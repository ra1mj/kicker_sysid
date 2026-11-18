from dataclasses import replace

import pytest

from kicker_sysid import ActuatorSpec, Bench, Friction, simulate
from kicker_sysid.identification import change_parameters, evaluate, fit, fit_trajectory
from kicker_sysid.spec import demo_spec


def test_stribeck_fit_generalizes_to_different_frequency():
    truth = replace(
        demo_spec(),
        friction=Friction(
            model="stribeck", coulomb=0.008, viscous=0.001, stribeck=0.012, stribeck_velocity=2.0
        ),
    )
    truth = replace(truth, controller=replace(truth.controller, domain="current"))
    bench = Bench(dt=0.002, control_dt=0.002)
    train = [
        simulate(
            truth,
            bench,
            experiment={"kind": "multisine", "duration": 0.8, "amplitude": 0.35, "frequency": 2.5},
            name="stribeck_train",
        )
    ]
    holdout = [
        simulate(
            truth,
            bench,
            experiment={"kind": "chirp", "duration": 0.8, "amplitude": 0.4, "frequency": 1.5},
            name="stribeck_holdout",
        )
    ]
    initial = replace(truth, friction=replace(truth.friction, stribeck=0.025, stribeck_velocity=4.0))
    values, _ = fit_trajectory(
        initial,
        train,
        {"friction.stribeck": [0.001, 0.03], "friction.stribeck_velocity": [0.2, 5.0]},
        generations=24,
        population=10,
        seed=12,
    )
    selected = change_parameters(initial, values)
    assert evaluate(selected, holdout)["score"] < evaluate(initial, holdout)["score"] * 0.25
    assert values["friction.stribeck"] == pytest.approx(0.012, rel=0.2)
    assert values["friction.stribeck_velocity"] == pytest.approx(2.0, rel=0.2)


@pytest.mark.integration
def test_parameters_recover_from_native_mujoco_observations():
    pytest.importorskip("mujoco")
    truth = demo_spec()
    initial = replace(
        truth,
        motor=replace(truth.motor, resistance=2.7, inductance=0.012, kt=0.08, ke=0.13, rotor_inertia=0.0006),
        friction=replace(truth.friction, coulomb=0.015, viscous=0.002),
    )
    train = [
        simulate(
            truth,
            Bench(),
            backend="mujoco",
            name="native_train",
            experiment={"kind": "prbs", "duration": 1.5, "amplitude": 4.0},
        )
    ]
    validation = [
        simulate(
            truth,
            Bench(load_inertia=0.0004),
            backend="mujoco",
            name="native_holdout",
            experiment={"kind": "chirp", "duration": 1.0, "amplitude": 3.0, "frequency": 2.0},
        )
    ]
    report = fit(
        {
            "schema": "kicker.dataset.v1",
            "initial_model": initial.to_dict(),
            "train": train,
            "validation": validation,
        }
    )
    fitted = ActuatorSpec.from_dict(report["selected_model"])
    assert report["accepted"]
    assert report["validation_rmse_ratio"] < 0.5
    assert fitted.motor.rotor_inertia == pytest.approx(truth.motor.rotor_inertia, rel=0.05)
    assert fitted.friction.coulomb == pytest.approx(truth.friction.coulomb, rel=0.05)
    assert fitted.friction.viscous == pytest.approx(truth.friction.viscous, rel=0.05)


def test_response_time_not_identifiable_in_voltage_domain():
    truth = demo_spec()
    trace = simulate(truth, experiment={"kind": "chirp", "duration": 0.1}, name="voltage")
    with pytest.raises(ValueError, match="not observable"):
        fit_trajectory(truth, [trace], {"controller.response_time": [0, 0.01]})


def test_trajectory_fit_rejects_zero_excitation():
    spec = demo_spec()
    trace = simulate(
        spec,
        Bench(locked=True),
        experiment={"kind": "step", "duration": 0.05, "amplitude": 0},
        name="unexcited",
    )
    with pytest.raises(ValueError, match="sensitivity"):
        fit_trajectory(spec, [trace], {"controller.delay_steps": [0, 5]})
