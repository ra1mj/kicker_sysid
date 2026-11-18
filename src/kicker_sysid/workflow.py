"""End-to-end demos, backend validation, plots and portable model export."""

import hashlib
import importlib.metadata
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .backends import bench_xml
from .data import save_trace, split_dataset, write_json
from .identification import fit, get_parameter, replay
from .simulation import simulate
from .spec import ActuatorSpec, Bench, demo_spec


def export(spec, folder, bench=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    bench = bench or Bench()
    write_json(folder / "model.json", spec.to_dict())
    write_json(folder / "bench.json", asdict(bench))
    (folder / "bench.xml").write_text(bench_xml(spec, bench), encoding="utf-8")
    (folder / "replay.py").write_text(
        '''"""Run after installing kicker-sysid; XML alone omits motor dynamics."""
from pathlib import Path
from kicker_sysid import ActuatorSpec, Bench, simulate
from kicker_sysid.data import read_json, save_trace
root = Path(__file__).resolve().parent
model = ActuatorSpec.load(root / "model.json")
trace = simulate(model, Bench(**read_json(root / "bench.json")), backend="mujoco")
save_trace(trace, root / "replay_output")
''',
        encoding="utf-8",
    )
    manifest = {
        "schema": "kicker.export.v1",
        "requires_runtime": "kicker-sysid==0.1.0",
        "note": "bench.xml is the mechanical bench only; replay.py executes the complete electrical/control/thermal model",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                folder / "model.json",
                folder / "bench.json",
                folder / "bench.xml",
                folder / "replay.py",
            )
        },
    }
    write_json(folder / "manifest.json", manifest)
    return manifest


def _plot(truth, before, after, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("position_rad", "velocity_rad_s", "current_a", "temperature_c")
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    for ax, key in zip(axes, keys):
        for trace, label, color in (
            (truth, "observed", "black"),
            (before, "initial", "#d97706"),
            (after, "identified", "#167d9a"),
        ):
            rows = [r for r in trace["samples"] if key in r]
            ax.plot([r["t_s"] for r in rows], [r[key] for r in rows], label=label, color=color, linewidth=1.1)
        ax.set_ylabel(key)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(truth["name"] + " — held-out validation")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validate(run, *, backend="all", device="cuda:0", output=None, progress=None):
    run = Path(run)
    _, _, traces = split_dataset(run / "dataset.json")
    spec = ActuatorSpec.load(run / "model.json")
    names = ["cpu", "mujoco", "mjlab"] if backend == "all" else [backend]
    results = []
    passed = True
    for trace in traces:
        reference = replay(spec, trace)
        for name in names:
            if progress:
                progress(f"Validate {trace['name']} on {name}")
            predicted = reference if name == "cpu" else replay(spec, trace, name, device)
            save_trace(predicted, run / "validation" / name / trace["name"])
            metrics = {}
            for key in ("position_rad", "velocity_rad_s", "current_a", "temperature_c"):
                a = np.array([r[key] for r in reference["samples"] if key in r])
                b = np.array([r[key] for r in predicted["samples"] if key in r])
                if not len(a):
                    continue
                rmse = float(np.sqrt(np.mean((a - b) ** 2)))
                floor = {"position_rad": 0.05, "velocity_rad_s": 0.1, "current_a": 0.1, "temperature_c": 1.0}[
                    key
                ]
                nrmse = rmse / max(float(np.ptp(a)), floor)
                metrics[key] = {"rmse": rmse, "nrmse": nrmse}
            ok = all(m["nrmse"] <= 0.02 for m in metrics.values())
            passed = passed and ok
            results.append({"trace": trace["name"], "backend": name, "passed": ok, "metrics": metrics})
    result = {
        "schema": "kicker.validation.v1",
        "passed": passed,
        "reference_backend": "cpu",
        "nrmse_threshold": 0.02,
        "device": device,
        "results": results,
    }
    write_json(output or run / "backend_validation.json", result)
    return result


def generate_demo(folder):
    folder = Path(folder)
    truth = demo_spec()
    initial = replace(
        truth,
        motor=replace(
            truth.motor,
            resistance=2.8,
            inductance=0.014,
            kt=0.075,
            ke=0.13,
            rotor_inertia=0.0006,
            thermal_resistance=4.5,
            thermal_capacity=3.2,
        ),
        friction=replace(truth.friction, coulomb=0.015, viscous=0.002),
    )
    experiments = [
        (
            "train",
            "locked_prbs",
            Bench(locked=True),
            {"kind": "prbs", "duration": 1.5, "amplitude": 5, "seed": 42},
        ),
        (
            "train",
            "moving_multisine",
            Bench(),
            {"kind": "multisine", "duration": 3.0, "amplitude": 5.0, "frequency": 2.3},
        ),
        ("train", "coast", Bench(), {"kind": "coast", "duration": 1.5, "amplitude": 3.0}),
        (
            "train",
            "heating_cooling",
            Bench(dt=0.01, control_dt=0.01, locked=True),
            {"kind": "thermal", "duration": 30.0, "amplitude": 3.0},
        ),
        (
            "validation",
            "heldout_chirp",
            Bench(load_inertia=0.00035),
            {"kind": "chirp", "duration": 2.0, "amplitude": 4.0, "frequency": 1.7},
        ),
        (
            "validation",
            "heldout_locked",
            Bench(locked=True),
            {"kind": "prbs", "duration": 1.0, "amplitude": 3.5, "seed": 731},
        ),
        (
            "validation",
            "heldout_thermal",
            Bench(dt=0.01, control_dt=0.01, locked=True),
            {"kind": "thermal", "duration": 20.0, "amplitude": 2.8},
        ),
    ]
    dataset = {
        "schema": "kicker.dataset.v1",
        "initial_model": initial.to_dict(),
        "train": [],
        "validation": [],
        "provenance": "synthetic independent experiments; all product ratings deliberately unspecified",
    }
    for split, name, bench, experiment in experiments:
        trace = simulate(truth, bench, experiment=experiment, name=name)
        path = save_trace(trace, folder / "data" / name)
        dataset[split].append(str(path.relative_to(folder)))
    # Ground truth is kept outside the dataset and never supplied to fit().
    write_json(folder / "generation_truth.json", truth.to_dict())
    write_json(folder / "dataset.json", dataset)
    return dataset


def demo(folder="runs/demo", *, backend="all", device="cuda:0", progress=print):
    folder = Path(folder)
    progress("Generate independent motor experiments")
    generate_demo(folder)
    progress("Fit electrical, torque, mechanical and thermal stages")
    report = fit(folder / "dataset.json", output=folder)
    truth = ActuatorSpec.load(folder / "generation_truth.json")
    selected = ActuatorSpec.from_dict(report["selected_model"])
    keys = (
        "motor.resistance",
        "motor.inductance",
        "motor.kt",
        "motor.ke",
        "motor.rotor_inertia",
        "friction.coulomb",
        "friction.viscous",
        "motor.thermal_resistance",
        "motor.thermal_capacity",
    )
    errors = {k: abs(get_parameter(selected, k) / get_parameter(truth, k) - 1) for k in keys}
    manifest, _, validation = split_dataset(folder / "dataset.json")
    baseline = ActuatorSpec.from_dict(manifest["initial_model"])
    for trace in validation:
        _plot(trace, replay(baseline, trace), replay(selected, trace), folder / (trace["name"] + ".png"))
    export(selected, folder / "export")
    comparison = validate(folder, backend=backend, device=device, progress=progress)
    versions = {}
    for name in (
        "kicker-sysid",
        "numpy",
        "scipy",
        "cmaes",
        "mujoco",
        "mjlab",
        "mujoco-warp",
        "warp-lang",
        "torch",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "schema": "kicker.demo.v1",
        "passed": bool(
            report["accepted"]
            and max(errors.values()) <= 0.05
            and report["validation_rmse_ratio"] is not None
            and report["validation_rmse_ratio"] <= 0.5
            and comparison["passed"]
        ),
        "parameter_relative_errors": errors,
        "validation_rmse_ratio": report["validation_rmse_ratio"],
        "backends_passed": comparison["passed"],
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": 42,
    }
    write_json(folder / "demo_report.json", result)
    return result
