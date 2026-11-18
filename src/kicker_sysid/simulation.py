"""Reproducible sampling, command holding and public simulation API."""

import math
from dataclasses import asdict

import numpy as np

from .backends import make_backend
from .signals import Command, make_commands
from .spec import Bench, Sensors


def simulate(
    spec,
    bench=None,
    commands=None,
    *,
    backend="cpu",
    sensors=None,
    experiment=None,
    name="run",
    device="cuda:0",
    viewer=False,
):
    bench, sensors = bench or Bench(), sensors or Sensors()
    if commands is None:
        exp = dict(experiment or {"kind": "multisine", "duration": 2.0})
        commands = make_commands(dt=bench.dt, domain=spec.controller.domain, mode=spec.controller.mode, **exp)
    commands = list(commands)
    if len(commands) < 3:
        raise ValueError("At least three commands required")
    for c in commands:
        if not isinstance(c, Command) or not all(math.isfinite(v) for v in asdict(c).values()):
            raise ValueError("Commands must contain finite SI values")
    engine = make_backend(backend, spec, bench, device=device, viewer=viewer)
    rng = np.random.default_rng(sensors.seed)
    stride = round(bench.control_dt / bench.dt)
    rows = []
    held = commands[0]
    try:
        for k, command in enumerate(commands):
            if k % stride == 0:
                held = command
            # External load and enable are environmental/drive inputs, not PD targets.
            applied = Command(
                held.position, held.velocity, held.effort, command.external_torque, command.enabled
            )
            state = engine.observe()
            position = state["position"] + spec.encoder.bias
            if sensors.encoder_resolution:
                position = round(position / sensors.encoder_resolution) * sensors.encoder_resolution
            position += rng.normal(0, sensors.position_noise_std)
            row = {
                "t_s": k * bench.dt,
                "position_rad": position,
                "velocity_rad_s": state["velocity"] + rng.normal(0, sensors.velocity_noise_std),
                "position_target_rad": applied.position,
                "velocity_target_rad_s": applied.velocity,
                "effort_command": applied.effort,
                "external_torque_nm": applied.external_torque,
                "enabled": int(applied.enabled),
            }
            if spec.motor:
                row.update(
                    current_a=state["current"] + rng.normal(0, sensors.current_noise_std),
                    temperature_c=state["temperature"] + rng.normal(0, sensors.temperature_noise_std),
                    electromagnetic_torque_nm=spec.motor.kt * state["current"],
                )
            out = engine.step(applied, measured_position=position)
            # Inputs/drive effort act on [t_k,t_{k+1}); state channels are at t_k.
            row["drive_torque_nm"] = out["effort"]
            if out["voltage"] is not None:
                row["applied_voltage_v"] = out["voltage"]
            rows.append(row)
    finally:
        engine.close()
    return {
        "schema": "kicker.trace.v1",
        "name": name,
        "backend": backend,
        "bench": asdict(bench),
        "sensors": asdict(sensors),
        "controller": asdict(spec.controller),
        "sampling": "state_before_step; applied_voltage_v and drive_torque_nm act during following interval",
        "torque_label": "motor_electromagnetic",
        "provenance": "synthetic",
        "samples": rows,
    }


def commands_from_trace(trace):
    return [
        Command(
            position=r.get("position_target_rad", 0),
            velocity=r.get("velocity_target_rad_s", 0),
            effort=r["effort_command"],
            external_torque=r.get("external_torque_nm", 0),
            enabled=bool(r.get("enabled", 1)),
        )
        for r in trace["samples"]
    ]
