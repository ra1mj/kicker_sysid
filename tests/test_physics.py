import math
from dataclasses import replace

import pytest

from kicker_sysid import (
    ActuatorSpec,
    Bench,
    Controller,
    Friction,
    Sensors,
    Transmission,
    simulate,
)
from kicker_sysid._vendor.layered import ActuatorKernel
from kicker_sysid.backends import CPUBackend, MuJoCoBackend
from kicker_sysid.signals import Command
from kicker_sysid.spec import demo_spec


def test_locked_rl_matches_analytic_and_torque():
    spec = demo_spec()
    spec = replace(spec, motor=replace(spec.motor, thermal_resistance=None, thermal_capacity=None))
    kernel = ActuatorKernel(spec)
    for k in range(1, 51):
        out = kernel.step((0, 0, 2), 0, 0, 0.001)
        expected = 1 - math.exp(-spec.motor.resistance * 0.001 * k / spec.motor.inductance)
        assert out["current"] == pytest.approx(expected, rel=1e-12)
        assert out["effort"] == pytest.approx(expected * spec.motor.kt, rel=1e-12)


def test_constant_torque_and_known_load():
    spec = ActuatorSpec(
        controller=Controller(domain="torque", mode="direct"),
        transmission=Transmission(extra_joint_armature=0.02),
        friction=Friction(model="none"),
    )
    backend = CPUBackend(spec, Bench(dt=0.001, control_dt=0.001, load_inertia=0.08))
    for _ in range(100):
        backend.step(Command(effort=0.2))
    assert backend.v == pytest.approx(0.2)
    assert backend.q == pytest.approx(0.0101)


def test_passive_friction_dissipates_and_sticks():
    spec = demo_spec()
    backend = CPUBackend(spec, Bench(initial_velocity=3))
    speeds = []
    for _ in range(500):
        backend.step(Command(enabled=False))
        speeds.append(backend.v)
    assert all(a >= b >= 0 for a, b in zip(speeds, speeds[1:]))
    assert speeds[-1] == 0
    backend.step(Command(enabled=False, external_torque=spec.friction.coulomb * 0.9))
    assert backend.v == 0
    backend.step(Command(enabled=False, external_torque=spec.friction.coulomb * 1.1))
    assert backend.v > 0


def test_command_delay_including_initial_command_and_reset():
    spec = ActuatorSpec(controller=Controller(domain="torque", mode="direct", delay_steps=2))
    kernel = ActuatorKernel(spec)
    assert [kernel.step((0, 0, 1), 0, 0, 0.001)["effort"] for _ in range(4)] == [0, 0, 1, 1]
    kernel.reset()
    assert kernel.step((0, 0, 2), 0, 0, 0.001)["effort"] == 0


def test_limits_and_reverse_symmetry():
    spec = demo_spec()
    a, b = ActuatorKernel(spec), ActuatorKernel(spec)
    for _ in range(50):
        x = a.step((0, 0, 100), 0, 0, 0.001)
        y = b.step((0, 0, -100), 0, 0, 0.001)
        assert abs(x["current"]) <= spec.motor.current_limit + 1e-12
        assert abs(x["voltage"]) <= spec.motor.voltage_limit
        assert x["effort"] == pytest.approx(-y["effort"])


def test_temperature_heats_and_cools():
    spec = demo_spec()
    kernel = ActuatorKernel(spec)
    for _ in range(1000):
        kernel.step((0, 0, 3), 0, 0, 0.01)
    hot = kernel.temperature
    assert hot > spec.motor.ambient_temperature + 1
    for _ in range(1000):
        kernel.step((0, 0, 0), 0, 0, 0.01)
    assert spec.motor.ambient_temperature < kernel.temperature < hot


def test_transmission_inertia_and_power_conventions():
    base = demo_spec()
    a = replace(base, transmission=Transmission(ratio=5, efficiency=0.8, extra_joint_armature=0.001))
    b = replace(a, transmission=replace(a.transmission, efficiency=1))
    assert a.joint_armature == b.joint_armature == pytest.approx(0.001 + 25 * base.motor.rotor_inertia)
    # At rest, the output torque equals N*eta*Kt*i, without applying N again in the engine.
    out = ActuatorKernel(a).step((0, 0, 1), 0, 0, 0.001)
    assert out["effort"] == pytest.approx(5 * 0.8 * base.motor.kt * out["current"])


def test_sensor_reproducibility_and_control_sampling():
    spec = replace(demo_spec(), controller=Controller(domain="voltage", mode="position", kp=3, kd=0.1))
    sensors = Sensors(position_noise_std=0.001, encoder_resolution=0.001, seed=8)
    bench = Bench(control_dt=0.004)
    a = simulate(
        spec, bench, sensors=sensors, experiment={"kind": "step", "duration": 0.05, "amplitude": 0.1}
    )
    b = simulate(
        spec, bench, sensors=sensors, experiment={"kind": "step", "duration": 0.05, "amplitude": 0.1}
    )
    assert a["samples"] == b["samples"]
    assert a["samples"][5]["position_target_rad"] == 0
    assert a["samples"][8]["position_target_rad"] == 0.1
    with pytest.raises(ValueError, match="integer"):
        Bench(control_dt=0.0015)


@pytest.mark.integration
def test_native_mujoco_static_friction_release():
    pytest.importorskip("mujoco")
    spec = demo_spec()
    backend = MuJoCoBackend(spec, Bench())
    for _ in range(100):
        backend.step(Command(enabled=False, external_torque=spec.friction.coulomb * 0.5))
    assert abs(backend.v) < 0.003
    for _ in range(100):
        backend.step(Command(enabled=False, external_torque=spec.friction.coulomb * 2))
    assert backend.v > 0.2
