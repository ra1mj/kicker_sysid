"""Optional public mjlab actuator integration; importing this requires the mjlab extra."""

from ._vendor.mjlab_actuator import (
    LayeredActuator,
    LayeredActuatorCfg,
    layered_init,
    layered_joint_position,
    prepare_simulation,
)

__all__ = [
    "LayeredActuator",
    "LayeredActuatorCfg",
    "layered_init",
    "layered_joint_position",
    "prepare_simulation",
]
