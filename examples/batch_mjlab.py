"""Run independent motor parameters on a real mjlab CUDA batch."""

import json

from kicker_sysid.mjlab_backend import MJLabBackend
from kicker_sysid.signals import Command
from kicker_sysid.spec import Bench, demo_spec

engine = MJLabBackend(demo_spec(), Bench(), num_envs=4, device="cuda:0")
engine.set_parameters({"motor.resistance": [[1.5], [2.0], [2.5], [3.0]]})
for _ in range(100):
    engine.step([Command(effort=u) for u in (2.0, 2.0, -2.0, -2.0)])
before = [engine.observe(i) for i in range(4)]
engine.reset([1, 3])
print(json.dumps({"before": before, "after_partial_reset": [engine.observe(i) for i in range(4)]}, indent=2))
engine.close()
