"""Public model types; all values use SI units."""

import math
from dataclasses import asdict, dataclass

from ._vendor.layered import ActuatorSpec, Controller, Electrical, Encoder, Friction, Transmission

__all__ = [
    "ActuatorSpec",
    "Controller",
    "Electrical",
    "Encoder",
    "Friction",
    "Transmission",
    "Bench",
    "Sensors",
]


@dataclass(frozen=True)
class Bench:
    dt: float = 0.001
    control_dt: float = 0.001
    load_inertia: float = 0.0002  # output side; excludes all actuator inertia
    initial_position: float = 0.0
    initial_velocity: float = 0.0
    locked: bool = False

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("Bench values must be finite")
        if min(self.dt, self.control_dt, self.load_inertia) <= 0:
            raise ValueError("dt, control_dt and known load_inertia must be positive")
        ratio = self.control_dt / self.dt
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-8:
            raise ValueError("control_dt must be an integer multiple of physics dt")


@dataclass(frozen=True)
class Sensors:
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    current_noise_std: float = 0.0
    temperature_noise_std: float = 0.0
    encoder_resolution: float = 0.0  # radians/count; zero disables quantization
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("Sensor seed must be an integer")
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid sensor {key}")


def demo_spec():
    return ActuatorSpec(
        motor=Electrical(
            resistance=2.0,
            inductance=0.008,
            kt=0.1,
            ke=0.1,
            voltage_limit=24.0,
            current_limit=4.0,
            rotor_inertia=0.0003,
            thermal_resistance=3.0,
            thermal_capacity=2.0,
        ),
        controller=Controller(domain="voltage", mode="direct", effort_limit=1.0),
        friction=Friction(model="coulomb", coulomb=0.008, viscous=0.001),
        metadata={
            "status": "synthetic_example",
            "description": "Lumped motor constants, not product ratings",
        },
    )
