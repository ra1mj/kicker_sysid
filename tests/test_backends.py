from dataclasses import replace

import numpy as np
import pytest

from kicker_sysid import Bench, simulate
from kicker_sysid._vendor.layered import ActuatorKernel, TorchOps
from kicker_sysid.signals import Command
from kicker_sysid.spec import demo_spec


def test_scalar_tensor_kernel_and_partial_delay_reset():
    torch = pytest.importorskip("torch")
    spec = demo_spec()
    spec = replace(spec, controller=replace(spec.controller, delay_steps=2))
    scalar = [ActuatorKernel(spec), ActuatorKernel(spec)]
    tensor = ActuatorKernel(spec, torch.zeros((2, 1), dtype=torch.float64), TorchOps())
    for k in range(30):
        q = torch.tensor([[k * 0.01], [-k * 0.02]], dtype=torch.float64)
        v = torch.tensor([[0.4], [-0.3]], dtype=torch.float64)
        if k == 12:
            tensor.reset(torch.tensor([[True], [False]]))
            scalar[0].reset()
        out = tensor.step((0.1, 0, 2), q, v, 0.001)
        for i in range(2):
            expected = scalar[i].step((0.1, 0, 2), q[i, 0].item(), v[i, 0].item(), 0.001)
            for key in ("current", "effort", "temperature", "frictionloss"):
                assert out[key][i, 0].item() == pytest.approx(expected[key], abs=1e-10)


@pytest.mark.integration
def test_cpu_mujoco_trajectory_agreement():
    pytest.importorskip("mujoco")
    spec = demo_spec()
    args = dict(
        bench=Bench(), experiment={"kind": "chirp", "duration": 0.5, "amplitude": 4.0, "frequency": 2.0}
    )
    a = simulate(spec, backend="cpu", **args)
    b = simulate(spec, backend="mujoco", **args)
    for key in ("position_rad", "velocity_rad_s", "current_a"):
        x = np.array([r[key] for r in a["samples"]])
        y = np.array([r[key] for r in b["samples"]])
        assert np.sqrt(np.mean((x - y) ** 2)) / max(np.ptp(x), 0.1) < 0.02


@pytest.mark.integration
@pytest.mark.gpu
def test_real_mjlab_batch_isolation_parameters_and_reset():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from kicker_sysid.mjlab_backend import MJLabBackend

    spec = demo_spec()
    engine = MJLabBackend(spec, Bench(), num_envs=3, device="cuda:0")
    engine.set_parameters(
        {"motor.resistance": [[1.0], [2.0], [3.0]], "controller.delay_steps": [[0], [1], [2]]}
    )
    for _ in range(20):
        engine.step([Command(effort=1), Command(effort=1), Command(effort=-1)])
    states = [engine.observe(i) for i in range(3)]
    assert states[0]["current"] > states[1]["current"] > 0
    assert states[2]["current"] < 0
    untouched_current = engine.kernel.current[1:].clone()
    untouched_q = engine.robot.data.joint_pos[1:].clone()
    engine.reset([0])
    assert engine.observe(0)["current"] == 0
    assert engine.observe(0)["position"] == pytest.approx(0, abs=1e-7)
    assert torch.equal(engine.kernel.current[1:], untouched_current)
    assert torch.allclose(engine.robot.data.joint_pos[1:], untouched_q)
    engine.set_parameters({"joint_armature": [[0.001], [0.002], [0.003]]})
    engine.reset()
    for _ in range(5):
        engine.step(Command(effort=1))
    batched = engine.step({"effort": torch.tensor([[1.0], [0.0], [-1.0]], device="cuda:0")})
    assert batched["effort"].shape == (3, 1)
    assert batched["effort"].device.type == "cuda"
    assert all(np.isfinite(engine.observe(i)["position"]) for i in range(3))
    with pytest.raises(ValueError):
        engine.set_parameters({"controller.delay_steps": [[0.5], [1], [2]]})


@pytest.mark.integration
@pytest.mark.gpu
def test_mjlab_smooth_trace_matches_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    spec = demo_spec()
    args = dict(experiment={"kind": "multisine", "duration": 0.3, "amplitude": 3.0})
    a = simulate(spec, backend="cpu", **args)
    b = simulate(spec, backend="mjlab", **args)
    for key in ("position_rad", "velocity_rad_s", "current_a"):
        x = np.array([r[key] for r in a["samples"]])
        y = np.array([r[key] for r in b["samples"]])
        assert np.sqrt(np.mean((x - y) ** 2)) / max(np.ptp(x), 0.1) < 0.02
