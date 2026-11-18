"""Encoder-only delay/response identification; electrical parameters remain fixed."""

import argparse
from dataclasses import replace
from pathlib import Path

from kicker_sysid import Bench, fit, simulate
from kicker_sysid.data import save_trace, write_json
from kicker_sysid.spec import demo_spec

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="runs/dynamics")
parser.add_argument("--generations", type=int, default=25)
args = parser.parse_args()
folder = Path(args.output)
truth = demo_spec()
truth = replace(
    truth,
    controller=replace(
        truth.controller, domain="current", response_time=0.008, delay_steps=2, max_delay_steps=5
    ),
)
initial = replace(truth, controller=replace(truth.controller, response_time=0.02, delay_steps=0))
dataset = {"schema": "kicker.dataset.v1", "initial_model": initial.to_dict(), "train": [], "validation": []}
for split, kind, duration in (("train", "prbs", 0.36), ("validation", "multisine", 0.4)):
    trace = simulate(
        truth,
        Bench(dt=0.002, control_dt=0.002),
        name=split,
        experiment={"kind": kind, "duration": duration, "amplitude": 0.5, "frequency": 4.0, "seed": 5},
    )
    trace.pop("torque_label", None)
    for row in trace["samples"]:
        for key in (
            "electromagnetic_torque_nm",
            "current_a",
            "temperature_c",
            "drive_torque_nm",
            "applied_voltage_v",
        ):
            row.pop(key, None)
    path = save_trace(trace, folder / split)
    dataset[split].append(str(path.relative_to(folder)))
write_json(folder / "dataset.json", dataset)
bounds = {"controller.response_time": [0.001, 0.03], "controller.delay_steps": [0, 5]}
write_json(folder / "bounds.json", bounds)
report = fit(
    folder / "dataset.json",
    output=folder,
    trajectory_bounds=bounds,
    generations=args.generations,
    population=10,
    seed=42,
    progress=lambda g, n, loss: print(f"{g}/{n}: {loss:.6g}", flush=True),
)
print("accepted:", report["accepted"], "validation RMSE ratio:", report["validation_rmse_ratio"])
raise SystemExit(0 if report["accepted"] else 1)
