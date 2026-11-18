"""Generic motor simulation, identification, validation and export."""

from .identification import fit
from .simulation import simulate
from .spec import ActuatorSpec, Bench, Controller, Electrical, Encoder, Friction, Sensors, Transmission
from .workflow import export, validate

__all__ = [
    "ActuatorSpec",
    "Bench",
    "Controller",
    "Electrical",
    "Encoder",
    "Friction",
    "Sensors",
    "Transmission",
    "simulate",
    "fit",
    "validate",
    "export",
]
