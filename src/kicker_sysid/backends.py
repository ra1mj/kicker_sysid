"""Scalar numerical and native MuJoCo motor benches."""

import math
from xml.sax.saxutils import escape

from ._vendor.layered import ActuatorKernel
from .spec import Bench


class CPUBackend:
    name = "cpu"

    def __init__(self, spec, bench: Bench):
        self.spec, self.bench = spec, bench
        self.kernel = ActuatorKernel(spec)
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is not None and list(env_ids) != [0]:
            raise ValueError("Scalar backend only has environment 0")
        self.q, self.v = self.bench.initial_position, self.bench.initial_velocity
        if self.bench.locked:
            self.v = 0.0
        self.kernel.reset()

    def observe(self, env=0):
        if env != 0:
            raise IndexError(env)
        return dict(
            position=self.q,
            velocity=self.v,
            current=self.kernel.current if self.spec.motor else None,
            temperature=self.kernel.temperature if self.spec.motor else None,
        )

    def _drive(self, command, measured_position):
        q = self.q if measured_position is None else measured_position - self.spec.encoder.bias
        return self.kernel.step(
            (command.position, command.velocity, command.effort),
            q,
            self.v,
            self.bench.dt,
            command.external_torque,
            command.enabled,
        )

    def step(self, command, measured_position=None):
        out = self._drive(command, measured_position)
        if not self.bench.locked:
            dt = self.bench.dt
            inertia = self.bench.load_inertia + self.spec.joint_armature
            momentum = inertia * self.v + dt * (out["effort"] + command.external_torque)
            dry = max(-out["frictionloss"] * dt, min(out["frictionloss"] * dt, momentum))
            self.v = (momentum - dry) / (inertia + dt * out["damping"])
            self.q += dt * self.v
        self._check_state()
        return out

    def _check_state(self):
        if not math.isfinite(self.q + self.v) or abs(self.v) > 1e6:
            raise RuntimeError("Unstable simulation; check timestep, gains and model")

    def close(self):
        pass


def bench_xml(spec, bench, *, actuator=True):
    """Load inertia belongs to the body; actuator-only inertia belongs to armature."""
    inertia = bench.load_inertia
    dry = max(1e-12, spec.friction.coulomb + spec.friction.stribeck)
    return f'''<mujoco model="kicker_motor_bench">
  <option timestep="{bench.dt}" gravity="0 0 0" integrator="implicitfast" tolerance="1e-10" iterations="100"/>
  <worldbody><light pos="0 0 2"/><body name="rotor" pos="0 0 0.3">
    <joint name="shaft" type="hinge" axis="0 0 1" armature="{spec.joint_armature}"
      damping="{spec.friction.viscous}" frictionloss="{dry}" solreffriction="0.002 1" solimpfriction="0.9999 0.9999 0.001"/>
    <inertial pos="0 0 0" mass="1" diaginertia="{inertia} {inertia} {inertia}"/>
    <geom name="flywheel" type="cylinder" size="0.08 0.015" rgba="0.2 0.55 0.9 1" contype="0" conaffinity="0"/>
    <geom type="capsule" fromto="0 0 0.025 0.08 0 0.025" size="0.005" rgba="1 0.5 0.1 1" contype="0" conaffinity="0"/>
  </body></worldbody>
  {"<actuator><motor name='drive' joint='shaft' gear='1'/></actuator>" if actuator else ""}
  <custom><text name="runtime_required" data="{escape("Use kicker_sysid runtime for electrical dynamics, delays and temperature")}"/></custom>
</mujoco>'''


class MuJoCoBackend(CPUBackend):
    name = "mujoco"

    def __init__(self, spec, bench, *, viewer=False):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is unavailable; install kicker-sysid[mujoco]") from exc
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(bench_xml(spec, bench))
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        super().__init__(spec, bench)
        if viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qvel[0] = self.q, self.v
        self.mj.mj_forward(self.model, self.data)

    def step(self, command, measured_position=None):
        out = self._drive(command, measured_position)
        self.model.dof_frictionloss[0] = out["frictionloss"]
        self.model.dof_damping[0] = out["damping"]
        self.data.ctrl[0] = out["effort"]
        self.data.qfrc_applied[0] = command.external_torque
        self.mj.mj_step(self.model, self.data)
        if self.bench.locked:
            self.data.qpos[0], self.data.qvel[0] = self.bench.initial_position, 0.0
            self.mj.mj_forward(self.model, self.data)
        self.q, self.v = float(self.data.qpos[0]), float(self.data.qvel[0])
        self._check_state()
        if self.viewer:
            self.viewer.sync()
        return out

    def close(self):
        if self.viewer:
            self.viewer.close()


def make_backend(name, spec, bench, *, device="cuda:0", viewer=False, num_envs=1):
    if num_envs != 1 and name != "mjlab":
        raise ValueError("Batched environments require the mjlab backend")
    if viewer and name != "mujoco":
        raise ValueError("The native viewer requires the mujoco backend")
    if name == "cpu":
        return CPUBackend(spec, bench)
    if name == "mujoco":
        return MuJoCoBackend(spec, bench, viewer=viewer)
    if name == "mjlab":
        from .mjlab_backend import MJLabBackend

        return MJLabBackend(spec, bench, device=device, num_envs=num_envs)
    raise ValueError(f"Unknown backend {name!r}")
