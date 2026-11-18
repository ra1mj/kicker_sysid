"""Versioned trace manifests and lossless optional CSV channels."""

import csv
import hashlib
import json
import math
from pathlib import Path

from .spec import Bench, Controller


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_trace(trace):
    if trace.get("schema") != "kicker.trace.v1":
        raise ValueError("Expected kicker.trace.v1")
    bench = Bench(**trace["bench"])
    Controller(**trace["controller"])
    rows = trace["samples"]
    if len(rows) < 3:
        raise ValueError("At least 3 rows required")
    if not trace.get("name"):
        raise ValueError("Trace needs a unique name")
    t0 = rows[0].get("t_s", 0)
    for k, row in enumerate(rows):
        if "t_s" not in row or "effort_command" not in row:
            raise ValueError("Missing t_s or effort_command")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in row.values()):
            raise ValueError("CSV channels must be finite numeric values; omit missing channels")
        if abs(row["t_s"] - t0 - k * bench.dt) > bench.dt * 1e-6:
            raise ValueError("Timestamps must be strictly increasing at bench.dt; resample first")
        if row.get("enabled", 1) not in (0, 1):
            raise ValueError("enabled must be 0 or 1")
    return trace


def save_trace(trace, folder):
    validate_trace(trace)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = trace["samples"]
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (folder / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {k: v for k, v in trace.items() if k != "samples"}
    manifest["csv"] = "samples.csv"
    manifest["csv_sha256"] = hashlib.sha256((folder / "samples.csv").read_bytes()).hexdigest()
    write_json(folder / "trace.json", manifest)
    return folder / "trace.json"


def load_trace(path):
    path = Path(path)
    trace = read_json(path)
    if "samples" not in trace:
        csv_path = path.parent / trace["csv"]
        if (
            trace.get("csv_sha256")
            and hashlib.sha256(csv_path.read_bytes()).hexdigest() != trace["csv_sha256"]
        ):
            raise ValueError(f"CSV checksum mismatch: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as f:
            trace["samples"] = [
                {k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)
            ]
    return validate_trace(trace)


def split_dataset(dataset, root=None):
    if isinstance(dataset, (str, Path)):
        path = Path(dataset)
        root, dataset = path.parent, read_json(path)
    root = Path(root or ".")
    if dataset.get("schema") != "kicker.dataset.v1":
        raise ValueError("Expected kicker.dataset.v1")
    train, validation = [], []
    identities, names = set(), set()
    for split, output in (("train", train), ("validation", validation)):
        for item in dataset.get(split, []):
            trace = load_trace(root / item) if isinstance(item, str) else validate_trace(item)
            # Normalize numeric representations and time origins so CSV copies
            # cannot evade the whole-record training/validation separation.
            start = trace["samples"][0]["t_s"]
            canonical = [
                {key: float(value - start if key == "t_s" else value) for key, value in row.items()}
                for row in trace["samples"]
            ]
            digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
            if digest in identities or trace["name"] in names:
                raise ValueError(
                    "Duplicate experiment: training/validation must be disjoint complete records"
                )
            identities.add(digest)
            names.add(trace["name"])
            output.append(trace)
    if not train or not validation:
        raise ValueError("Both training and held-out validation records are required")
    return dataset, train, validation
