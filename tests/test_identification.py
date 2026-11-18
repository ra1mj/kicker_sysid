from copy import deepcopy
from dataclasses import replace

import pytest

from kicker_sysid import ActuatorSpec, Bench, fit, simulate
from kicker_sysid.data import load_trace, save_trace, split_dataset
from kicker_sysid.identification import change_parameters, evaluate, fit_trajectory, get_parameter
from kicker_sysid.spec import demo_spec
from kicker_sysid.workflow import generate_demo


@pytest.fixture(scope="module")
def identified(tmp_path_factory):
    folder = tmp_path_factory.mktemp("motor_demo")
    manifest = generate_demo(folder)
    report = fit(folder / "dataset.json", output=folder)
    return folder, manifest, report


def test_known_parameter_recovery_and_holdout(identified):
    _, _, report = identified
    truth = demo_spec()
    result = ActuatorSpec.from_dict(report["selected_model"])
    assert report["accepted"]
    assert report["validation_rmse_ratio"] < 0.5
    for name in (
        "motor.resistance",
        "motor.inductance",
        "motor.kt",
        "motor.ke",
        "motor.rotor_inertia",
        "friction.coulomb",
        "friction.viscous",
        "motor.thermal_resistance",
        "motor.thermal_capacity",
    ):
        assert report["parameters"][name]["status"] == "fitted"
        assert get_parameter(result, name) == pytest.approx(get_parameter(truth, name), rel=0.05)


def test_missing_torque_never_fits_kt(identified):
    folder, manifest, _ = identified
    _, train, validation = split_dataset(folder / "dataset.json")
    for trace in train + validation:
        trace.pop("torque_label", None)
        for row in trace["samples"]:
            row.pop("electromagnetic_torque_nm", None)
    data = {**manifest, "train": train, "validation": validation}
    result = fit(data)
    assert result["parameters"]["motor.kt"]["status"] == "unidentifiable"
    assert result["candidate_model"]["motor"]["kt"] == manifest["initial_model"]["motor"]["kt"]


def test_no_excitation_and_insufficient_heat_keep_priors():
    spec = demo_spec()
    train = simulate(
        spec, Bench(locked=True), experiment={"kind": "step", "duration": 0.05, "amplitude": 0}, name="zero"
    )
    validation = simulate(
        spec,
        Bench(locked=True),
        experiment={"kind": "step", "duration": 0.06, "amplitude": 0},
        name="zero_holdout",
    )
    result = fit(
        {
            "schema": "kicker.dataset.v1",
            "initial_model": spec.to_dict(),
            "train": [train],
            "validation": [validation],
        }
    )
    assert result["parameters"]["motor.resistance"]["status"] == "unidentifiable"
    assert result["parameters"]["motor.thermal_capacity"]["status"] == "unidentifiable"
    assert result["selected_model"] == spec.to_dict()


def test_rejects_duplicate_split_and_bad_timestamps(identified):
    folder, manifest, _ = identified
    _, train, validation = split_dataset(folder / "dataset.json")
    data = {**manifest, "train": train, "validation": [deepcopy(train[0])]}
    with pytest.raises(ValueError, match="Duplicate"):
        split_dataset(data)
    renamed = deepcopy(train[0])
    renamed["name"] = "renamed_csv_copy"
    renamed["samples"] = [{key: float(value) for key, value in row.items()} for row in renamed["samples"]]
    with pytest.raises(ValueError, match="Duplicate"):
        split_dataset({**manifest, "train": train, "validation": [renamed]})
    broken = deepcopy(train[0])
    broken["samples"][1]["t_s"] = 0
    with pytest.raises(ValueError, match="Timestamps"):
        split_dataset({**manifest, "train": [broken], "validation": validation})


def test_csv_roundtrip_missing_channels_and_checksum(tmp_path):
    trace = simulate(demo_spec(), experiment={"kind": "step", "duration": 0.03})
    trace["samples"][2].pop("current_a")
    path = save_trace(trace, tmp_path)
    loaded = load_trace(path)
    assert "current_a" not in loaded["samples"][2]
    assert loaded["samples"] == trace["samples"]
    with (tmp_path / "samples.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_trace(path)


def test_cma_recovers_delay_and_response_on_unseen_trajectory():
    truth = demo_spec()
    truth = replace(
        truth,
        controller=replace(
            truth.controller, domain="current", response_time=0.008, delay_steps=2, max_delay_steps=5
        ),
    )
    bench = Bench(dt=0.002, control_dt=0.002)
    train = [
        simulate(
            truth,
            bench,
            experiment={"kind": "prbs", "duration": 0.36, "amplitude": 0.5, "seed": 5},
            name="fit",
        )
    ]
    holdout = [
        simulate(
            truth,
            bench,
            experiment={"kind": "multisine", "duration": 0.4, "amplitude": 0.5, "frequency": 4},
            name="holdout",
        )
    ]
    initial = replace(truth, controller=replace(truth.controller, response_time=0.02, delay_steps=0))
    values, diag = fit_trajectory(
        initial,
        train,
        {"controller.response_time": [0.001, 0.03], "controller.delay_steps": [0, 5]},
        generations=20,
        population=10,
        seed=42,
    )
    fitted = change_parameters(initial, values)
    assert values["controller.delay_steps"] == 2
    assert values["controller.response_time"] == pytest.approx(0.008, rel=0.1)
    assert evaluate(fitted, holdout)["score"] < evaluate(initial, holdout)["score"] * 0.25
    assert len(diag["train_loss"]) == 20


def test_validation_regression_retains_baseline(identified, monkeypatch):
    import kicker_sysid.identification as module

    folder, _, _ = identified
    original = module.evaluate
    count = 0

    def bad_candidate(spec, traces, *args, **kwargs):
        nonlocal count
        result = original(spec, traces, *args, **kwargs)
        count += 1
        result["score"] = 1.0 if count == 1 else 2.0
        return result

    monkeypatch.setattr(module, "evaluate", bad_candidate)
    result = fit(folder / "dataset.json")
    assert not result["accepted"]
    assert result["selected_model"] != result["candidate_model"]
