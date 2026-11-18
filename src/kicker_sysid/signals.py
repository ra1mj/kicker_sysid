"""Deterministic commands shared across physics backends."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Command:
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0  # V, A or output Nm according to controller.domain
    external_torque: float = 0.0  # signed output-side applied torque
    enabled: bool = True


def make_commands(
    kind,
    duration,
    dt,
    *,
    amplitude=3.0,
    frequency=1.0,
    seed=42,
    domain="voltage",
    mode="direct",
    load_torque=0.0,
):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive")
    n = round(duration / dt)
    if n < 3 or abs(n * dt - duration) > dt * 1e-6:
        raise ValueError("duration must contain at least 3 whole physics steps")
    rng = np.random.default_rng(seed)
    values = []
    block = max(1, round(0.02 / dt))
    random_values = rng.uniform(-amplitude, amplitude, (n + block - 1) // block)
    for k in range(n):
        t = k * dt
        enabled = True
        if kind == "multisine":
            u = amplitude * (
                0.55 * math.sin(2 * math.pi * frequency * t)
                + 0.3 * math.sin(2 * math.pi * frequency * 2.7 * t + 0.3)
                + 0.15 * math.sin(2 * math.pi * frequency * 6.1 * t)
            )
        elif kind == "chirp":
            u = amplitude * math.sin(2 * math.pi * (frequency * t + 2 * t * t / duration))
        elif kind == "prbs":
            u = float(random_values[k // block])
        elif kind == "step":
            u = amplitude if t >= duration * 0.1 else 0.0
        elif kind == "coast":
            enabled = t < duration / 3
            u = amplitude if enabled else 0.0
        elif kind == "thermal":
            u = amplitude if t < duration * 0.6 else 0.0
        elif kind == "reverse":
            u = amplitude if t < duration / 2 else -amplitude
        else:
            raise ValueError(f"Unknown experiment: {kind}")
        values.append(
            Command(
                position=u if mode == "position" else 0.0,
                effort=u if mode == "direct" else 0.0,
                external_torque=load_torque,
                enabled=enabled,
            )
        )
    return values
