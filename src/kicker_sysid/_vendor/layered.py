"""Explicit motor/drive/transmission/friction layers, in SI units.

The same arithmetic runs on Python scalars and Torch tensors. This module
produces joint effort and a friction *budget*. It never integrates the robot:
MuJoCo owns that state; the separate pendulum replay is only a test bench.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Electrical:
    resistance: float
    inductance: float
    kt: float
    ke: float
    voltage_limit: float
    current_limit: float
    rotor_inertia: float = 0.0
    ambient_temperature: float = 25.0
    resistance_tempco: float = 0.00393
    thermal_resistance: float | None = None
    thermal_capacity: float | None = None


@dataclass(frozen=True)
class Transmission:
    ratio: float = 1.0  # omega_motor / omega_joint
    extra_joint_armature: float = 0.0  # gearbox inertia, excludes rotor and link
    efficiency: float = 1.0  # optional constant-efficiency approximation


@dataclass(frozen=True)
class Controller:
    domain: str = "torque"  # joint Nm, motor A, or motor V
    mode: str = "position"  # position PD or direct effort input
    kp: float = 0.0  # domain-units/rad of JOINT error
    kd: float = 0.0  # domain-units/(rad/s) of JOINT error
    effort_limit: float = 10.0  # joint electromagnetic effort bound, Nm
    stall_torque: float = 0.0  # optional ideal-drive torque-speed envelope, joint Nm
    no_load_speed: float = 0.0  # joint rad/s, not a hard velocity clamp
    response_time: float = 0.0  # ideal current/torque closed-loop time constant
    delay_steps: int = 0  # command delay, in physics steps
    max_delay_steps: int = 32


@dataclass(frozen=True)
class Friction:
    model: str = "coulomb"  # none, coulomb, stribeck, m6
    coulomb: float = 0.0  # all friction parameters are JOINT/output-side
    viscous: float = 0.0
    stribeck: float = 0.0
    stribeck_velocity: float = 0.1
    stribeck_exponent: float = 2.0
    load_motor: float = 0.0
    load_external: float = 0.0
    load_motor_stribeck: float = 0.0
    load_external_stribeck: float = 0.0
    quadratic_motor: float = 0.0  # units 1/Nm
    quadratic_external: float = 0.0


@dataclass(frozen=True)
class Encoder:
    bias: float = 0.0  # q_measured = q_true + bias; opposite sign to PACE's bias


@dataclass(frozen=True)
class ActuatorSpec:
    motor: Electrical | None = None  # None: identified ideal joint torque source
    transmission: Transmission = field(default_factory=Transmission)
    controller: Controller = field(default_factory=Controller)
    friction: Friction = field(default_factory=Friction)
    encoder: Encoder = field(default_factory=Encoder)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for layer in (self.motor, self.transmission, self.controller, self.friction, self.encoder):
            if layer is not None:
                for key, value in asdict(layer).items():
                    if isinstance(value, (float, int)) and not math.isfinite(value):
                        raise ValueError(f"Non-finite {type(layer).__name__}.{key}")
        c, t, f, m = self.controller, self.transmission, self.friction, self.motor
        if c.domain not in ("voltage", "current", "torque") or c.mode not in ("position", "direct"):
            raise ValueError("Unknown controller domain or mode")
        if m is None and c.domain != "torque":
            raise ValueError("Voltage/current drives require electrical parameters")
        if m is None and (t.ratio != 1 or t.efficiency != 1):
            raise ValueError("An ideal joint torque source is already output-side; use ratio=efficiency=1")
        if c.effort_limit <= 0 or min(c.kp, c.kd, c.response_time, c.stall_torque, c.no_load_speed) < 0:
            raise ValueError("Invalid controller gains or effort limits")
        if bool(c.stall_torque) != bool(c.no_load_speed):
            raise ValueError("Specify both stall_torque and no_load_speed, or neither")
        if not (isinstance(c.delay_steps, int) and isinstance(c.max_delay_steps, int)
                and 0 <= c.delay_steps <= c.max_delay_steps <= 1024):
            raise ValueError("Delay must be integer physics steps within max_delay_steps")
        if t.ratio <= 0 or t.extra_joint_armature < 0 or not 0 < t.efficiency <= 1:
            raise ValueError("Invalid transmission parameters")
        if f.model not in ("none", "coulomb", "stribeck", "m6"):
            raise ValueError("Unknown friction model")
        if any(v < 0 for v in asdict(f).values() if isinstance(v, (int, float))):
            raise ValueError("Friction coefficients must be non-negative")
        if f.stribeck_velocity <= 0 or f.stribeck_exponent <= 0:
            raise ValueError("Stribeck velocity and exponent must be positive")
        if f.model == "m6" and t.efficiency != 1:
            raise ValueError("M6 already models gearbox losses; use efficiency=1")
        if m is not None:
            if c.stall_torque or c.no_load_speed:
                raise ValueError("Electrical models already derive the torque-speed envelope; do not add a second empirical envelope")
            if min(m.resistance, m.kt, m.ke, m.voltage_limit, m.current_limit) <= 0 or min(m.inductance, m.rotor_inertia) < 0:
                raise ValueError("Invalid electrical parameters")
            if (m.thermal_resistance is None) != (m.thermal_capacity is None):
                raise ValueError("Thermal model needs both resistance and capacity")
            if m.thermal_resistance is not None and min(m.thermal_resistance, m.thermal_capacity) <= 0:
                raise ValueError("Thermal resistance/capacity must be positive")

    @property
    def joint_armature(self):
        return self.transmission.extra_joint_armature + (self.motor.rotor_inertia * self.transmission.ratio**2 if self.motor else 0.0)

    def to_dict(self):
        return {"schema": "sim2motor.actuator.v1", **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("schema", None) != "sim2motor.actuator.v1":
            raise ValueError("Expected sim2motor.actuator.v1; legacy specs need explicit conversion")
        for name, kind in (("motor", Electrical), ("transmission", Transmission),
                           ("controller", Controller), ("friction", Friction), ("encoder", Encoder)):
            if name in data and data[name] is not None:
                data[name] = kind(**data[name])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


class ScalarOps:
    exp = staticmethod(math.exp)
    abs = staticmethod(abs)
    maximum = staticmethod(max)
    minimum = staticmethod(min)
    where = staticmethod(lambda condition, a, b: a if condition else b)


class TorchOps:
    """Optional backend. Importing the scalar model does not require torch."""
    def __init__(self):
        import torch
        self.torch = torch
        self.exp = torch.exp
        self.abs = torch.abs

    def _pair(self, a, b):
        t = a if self.torch.is_tensor(a) else b
        return (self.torch.as_tensor(a, device=t.device, dtype=t.dtype),
                self.torch.as_tensor(b, device=t.device, dtype=t.dtype))

    def maximum(self, a, b):
        return self.torch.maximum(*self._pair(a, b))

    def minimum(self, a, b):
        return self.torch.minimum(*self._pair(a, b))

    def where(self, condition, a, b):
        if isinstance(condition, bool):
            return a if condition else b
        return self.torch.where(condition, a, b)


def clip(x, lo, hi, ops):
    return ops.minimum(ops.maximum(x, lo), hi)


def friction_budget(parameters, model, motor_effort, external_effort, velocity, ops=ScalarOps):
    """BAM-style output-side budget. CPU and GPU use this exact function.

    Quadratic term is enabled only for opposing torques; equal magnitudes
    enable neither branch. This explicitly chooses BAM's CPU convention.
    """
    p = lambda key: parameters["friction." + key]
    zero = velocity * 0.0
    if model == "none":
        return zero, zero
    dry = zero + p("coulomb")
    if model in ("stribeck", "m6"):
        exponent = (ops.abs(velocity) / p("stribeck_velocity")) ** p("stribeck_exponent")
        s = ops.exp(-ops.minimum(exponent, 700.0))
        dry = dry + s * p("stribeck")
        if model == "m6":
            dry = dry + ops.abs(p("load_motor") * motor_effort - p("load_external") * external_effort)
            dry = dry + s * ops.abs(p("load_motor_stribeck") * motor_effort - p("load_external_stribeck") * external_effort)
            am, ae = ops.abs(motor_effort), ops.abs(external_effort)
            quad = ops.where(am > ae, p("quadratic_external") * ae**2, zero)
            quad = quad + ops.where(ae > am, p("quadratic_motor") * am**2, zero)
            dry = dry + s * ops.where(motor_effort * external_effort < 0, quad, zero)
    return dry, zero + p("viscous")


class ActuatorKernel:
    """Drive + motor + rigid gearbox; returns effort before native friction.

    `template` is 0.0 for scalar replay or a (worlds, joints) tensor. Numeric
    parameters may be replaced with broadcast-compatible tensors for sysid.
    Commands are (joint_position_target, joint_velocity_target, feedforward).
    Feedforward is in the configured controller domain, never inferred.
    """
    def __init__(self, spec: ActuatorSpec, template=0.0, ops=ScalarOps):
        self.spec, self.ops, self.zero = spec, ops, template * 0.0
        self.parameters = {}
        for name in ("motor", "transmission", "controller", "friction", "encoder"):
            obj = getattr(spec, name)
            if obj is not None:
                for key, value in asdict(obj).items():
                    if isinstance(value, (int, float)):
                        self.parameters[name + "." + key] = self.zero + value
        self.history = []
        self.reset()

    def reset(self, mask=None):
        ambient = self.parameters.get("motor.ambient_temperature", self.zero + 25.0)
        if mask is None:
            self.current, self.temperature, self.filtered_effort = self.zero, ambient, self.zero
            self.history = []
            self.fresh = self.zero == 0
        else:
            self.current = self.ops.where(mask, self.zero, self.current)
            self.temperature = self.ops.where(mask, ambient, self.temperature)
            self.filtered_effort = self.ops.where(mask, self.zero, self.filtered_effort)
            self.fresh = self.fresh | mask

    def _delay(self, commands):
        if not self.history:
            self.history = [(self.zero, self.zero, self.zero)] * (self.spec.controller.max_delay_steps + 1)
        # A reset restores a zero command history, including for partial resets.
        # Do not pre-fill history with the first command: that bypasses delay.
        self.history = [commands] + [tuple(self.ops.where(self.fresh, self.zero, old)
            for old in item) for item in self.history[:-1]]
        self.fresh = self.zero != 0
        lag = self.parameters["controller.delay_steps"]
        output = self.history[0]
        for index, item in enumerate(self.history[1:], 1):
            output = tuple(self.ops.where(lag == index, value, prev) for value, prev in zip(item, output))
        self._delay_output = output
        return output

    def limit_effort(self, effort, velocity):
        p, o = self.parameters, self.ops
        lim = p["controller.effort_limit"]
        low, high = -lim, lim
        if self.spec.controller.stall_torque > 0:
            peak, speed = p["controller.stall_torque"], p["controller.no_load_speed"]
            v = clip(velocity, -speed * (1 + lim / peak), speed * (1 + lim / peak), o)
            low = o.maximum(-lim, peak * (-1 - v / speed))
            high = o.minimum(lim, peak * (1 - v / speed))
        return clip(effort, low, high, o)

    def step(self, commands, position, velocity, dt, external_effort=0.0, enabled=True):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        p, o, cfg = self.parameters, self.ops, self.spec
        q_target, v_target, ff = self._delay(tuple(self.zero + x for x in commands))
        error = q_target - position - p["encoder.bias"]
        request = ff
        if cfg.controller.mode == "position":
            request = request + p["controller.kp"] * error + p["controller.kd"] * (v_target - velocity)
        n, eta = p["transmission.ratio"], p["transmission.efficiency"]
        applied_voltage = self.zero
        if cfg.motor is None:
            effort = self.limit_effort(request, velocity)
            if cfg.controller.response_time > 0:
                decay = o.exp(-dt / p["controller.response_time"])
                effort = effort + (self.filtered_effort - effort) * decay
            effort = self.limit_effort(effort, velocity)
            effort = o.where(enabled, effort, self.zero)
            self.filtered_effort = effort
        else:
            r = o.maximum(p["motor.resistance"] * (1 + p["motor.resistance_tempco"] * (self.temperature - p["motor.ambient_temperature"])), 1e-9)
            emf = p["motor.ke"] * n * velocity
            decay = o.exp(-r * dt / o.maximum(p["motor.inductance"], 1e-15))
            decay = o.where(p["motor.inductance"] > 0, decay, self.zero)
            gain = o.maximum(1 - decay, 1e-15)
            current_limit, bus = p["motor.current_limit"], p["motor.voltage_limit"]
            # Convert the joint command cap into phase-current bounds, so
            # current, delivered torque and copper heat remain consistent.
            positive_factor = o.where(velocity >= 0, eta, 1 / eta)
            negative_factor = o.where(velocity <= 0, eta, 1 / eta)
            positive_limit = o.minimum(current_limit, p["controller.effort_limit"] / (n * p["motor.kt"] * positive_factor))
            negative_limit = o.minimum(current_limit, p["controller.effort_limit"] / (n * p["motor.kt"] * negative_factor))
            if cfg.controller.domain == "voltage":
                requested_voltage = request
            else:
                desired_current = request
                if cfg.controller.domain == "torque":
                    request = self.limit_effort(request, velocity)
                    # Invert the same drive/backdrive efficiency convention.
                    factor = o.where(request * velocity >= 0, eta, 1 / eta)
                    desired_current = request / (n * factor * p["motor.kt"])
                desired_current = clip(desired_current, -negative_limit, positive_limit, o)
                if cfg.controller.response_time > 0:
                    desired_current = desired_current + (self.current - desired_current) * o.exp(-dt / p["controller.response_time"])
                requested_voltage = emf + r * (desired_current - decay * self.current) / gain
            # Current limiting acts through drive voltage, followed by physical
            # bus saturation. Backdriving can therefore exceed the current limit.
            vlo = emf + r * (-negative_limit - decay * self.current) / gain
            vhi = emf + r * (positive_limit - decay * self.current) / gain
            applied_voltage = clip(clip(requested_voltage, vlo, vhi, o), -bus, bus, o)
            current = decay * self.current + gain * (applied_voltage - emf) / r
            # Disabled means an open circuit, not short-circuit braking.
            self.current = o.where(enabled, current, self.zero)
            applied_voltage = o.where(enabled, applied_voltage, self.zero)
            rotor_effort = p["motor.kt"] * self.current
            factor = o.where(rotor_effort * velocity >= 0, eta, 1 / eta)
            effort = n * factor * rotor_effort
            if cfg.motor.thermal_capacity is not None:
                rt, ct = p["motor.thermal_resistance"], p["motor.thermal_capacity"]
                equilibrium = p["motor.ambient_temperature"] + self.current**2 * r * rt
                self.temperature = equilibrium + (self.temperature - equilibrium) * o.exp(-dt / (rt * ct))
        dry, viscous = friction_budget(p, cfg.friction.model, effort, self.zero + external_effort, self.zero + velocity, o)
        return {"effort": effort, "frictionloss": dry, "damping": viscous,
                "current": self.current if cfg.motor else None,
                "temperature": self.temperature if cfg.motor else None,
                "voltage": applied_voltage if cfg.motor else None,
                "encoder_position": position + p["encoder.bias"]}


def legacy_to_layered(data, *, joint_armature, controller=None):
    """Explicit migration: old reflected_inertia may contain the bench load.

    Never infer armature from that ambiguous field. Unsupported legacy physics
    is rejected instead of silently disappearing from the exported simulation.
    """
    if joint_armature is None:
        raise ValueError("Specify actuator-only joint_armature; exclude the bench/link inertia")
    ext = data.get("simulation_extensions") or {}
    unsupported = ("backlash_gap_rad", "backlash_stiffness", "backlash_damping", "lugre_sigma0", "cogging_amplitude", "saturation_current", "deadzone_voltage", "encoder_noise_std", "encoder_bias", "kt_tempco", "integral_limit")
    active = [key for key in unsupported if ext.get(key, 0)]
    if ext.get("temperature_segments"):
        active.append("temperature_segments")
    if active:
        raise ValueError("Legacy effects need explicit modeling before migration: " + ", ".join(active))
    n = float(data.get("gear_ratio", 1.0))
    eta = float(ext.get("gearbox_efficiency", data.get("gearbox_efficiency", 1.0)))
    if eta != 1.0 or ext.get("friction_load_coefficient", 0):
        raise ValueError("Refit gearbox losses in joint coordinates; legacy efficiency/load multiplier cannot be migrated implicitly")
    kt = float(data.get("motor_constant_kt", data.get("kt")))
    current_limit = ext.get("current_limit", data.get("current_limit"))
    if not current_limit:
        raise ValueError("Explicit phase current_limit is required; peak_torque may be in another coordinate system")
    motor = Electrical(resistance=float(data["resistance"]), inductance=float(data["inductance"]),
        kt=kt, ke=float(data.get("motor_constant_ke", data.get("ke"))), voltage_limit=float(data["voltage_range"][1]), current_limit=float(current_limit),
        ambient_temperature=float(data.get("ambient_temperature", 25.0)),
        resistance_tempco=float(ext.get("resistance_tempco", .00393)),
        thermal_resistance=data.get("thermal_resistance"),
        thermal_capacity=(data["thermal_time_constant"] / data["thermal_resistance"] if data.get("thermal_resistance") and data.get("thermal_time_constant") else None))
    c = controller or Controller(domain="voltage", mode="direct", effort_limit=float(data["peak_torque"]), delay_steps=int(ext.get("actuator_delay_steps", 0)))
    return ActuatorSpec(motor=motor, transmission=Transmission(ratio=n, extra_joint_armature=joint_armature), controller=c,
        friction=Friction(model="stribeck", coulomb=float(data.get("friction_static", 0)) * n,
                          viscous=float(data.get("friction_dynamic", 0)) * n*n,
                          stribeck=float(ext.get("friction_stribeck", 0)) * n,
                          stribeck_velocity=float(ext.get("friction_stribeck_velocity", .01)) / n,
                          stribeck_exponent=float(ext.get("friction_stribeck_exponent", 2))),
        encoder=Encoder(bias=float(ext.get("encoder_bias", 0))),
        metadata={"source_motor_id": data.get("motor_id"), "migration": "explicit joint armature; confirm all units"})
