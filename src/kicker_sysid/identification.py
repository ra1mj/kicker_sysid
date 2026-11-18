"""Staged identification from observed channels, with whole-record holdouts."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .data import split_dataset, write_json
from .simulation import commands_from_trace, simulate
from .spec import ActuatorSpec, Bench, Controller, Sensors

PARAMETERS = (
    "motor.resistance",
    "motor.inductance",
    "motor.ke",
    "motor.kt",
    "motor.rotor_inertia",
    "friction.coulomb",
    "friction.viscous",
    "friction.stribeck",
    "friction.stribeck_velocity",
    "controller.response_time",
    "controller.delay_steps",
    "motor.thermal_resistance",
    "motor.thermal_capacity",
)


class InsufficientExcitation(ValueError):
    """A requested parameter is not observable in the supplied records."""


def change_parameters(spec, values):
    data = spec.to_dict()
    for name, value in values.items():
        layer, key = name.split(".")
        if layer not in data or data[layer] is None or key not in data[layer]:
            raise ValueError(f"Unknown model parameter {name}")
        data[layer][key] = int(round(value)) if name == "controller.delay_steps" else float(value)
    return ActuatorSpec.from_dict(data)


def get_parameter(spec, name):
    layer, key = name.split(".")
    return getattr(getattr(spec, layer), key) if getattr(spec, layer) is not None else None


def _rank(matrix):
    matrix = np.asarray(matrix, dtype=float)
    scales = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.maximum(scales, 1e-30)
    rank = int(np.linalg.matrix_rank(normalized, tol=1e-8))
    condition = float(np.linalg.cond(normalized))
    if rank < matrix.shape[1] or not np.isfinite(condition) or condition > 1e8:
        raise InsufficientExcitation(f"Insufficient excitation: rank {rank}/{matrix.shape[1]}")
    return scales, {"rank": rank, "condition": condition, "samples": len(matrix)}


def _linear(matrix, target, lower=0.0):
    matrix, target = np.asarray(matrix), np.asarray(target)
    scales, diag = _rank(matrix)
    solution = lsq_linear(matrix / scales, target, bounds=(lower, np.inf), tol=1e-12)
    if not solution.success:
        raise ValueError("Bounded least squares did not converge")
    diag["residual_rmse"] = float(np.sqrt(np.mean((matrix @ (solution.x / scales) - target) ** 2)))
    return solution.x / scales, diag


def _pairs(traces, required, *, enabled_only=False):
    for trace in traces:
        rows = trace["samples"]
        for a, b in zip(rows[:-1], rows[1:]):
            if all(k in a and k in b for k in required) and (not enabled_only or a.get("enabled", 1)):
                yield trace, a, b


def fit_electrical(spec, traces):
    pairs = list(_pairs(traces, ("current_a", "velocity_rad_s", "applied_voltage_v"), enabled_only=True))
    if len(pairs) < 12:
        raise ValueError("Need >=12 voltage/current/velocity transitions")
    i0 = np.array([a["current_a"] for _, a, _ in pairs])
    i1 = np.array([b["current_a"] for _, _, b in pairs])
    u = np.array([a["applied_voltage_v"] for _, a, _ in pairs])
    w = np.array([a["velocity_rad_s"] * spec.transmission.ratio for _, a, _ in pairs])
    dt = np.array([t["bench"]["dt"] for t, _, _ in pairs])
    temp = np.array([a.get("temperature_c", spec.motor.ambient_temperature) for _, a, _ in pairs])
    factor = 1 + spec.motor.resistance_tempco * (temp - spec.motor.ambient_temperature)
    # Lock-only records cannot identify Ke. Keep the prior and fit R/L alone.
    names = ["motor.resistance", "motor.inductance"]
    matrix = np.column_stack(((i0 + i1) * 0.5 * factor, (i1 - i0) / dt))
    if np.ptp(w) > 0.1:
        matrix = np.column_stack((matrix, w))
        names.append("motor.ke")
    _, diag = _rank(matrix)
    initial = np.array([get_parameter(spec, n) for n in names])
    initial = np.maximum(initial, [1e-6, 1e-8] + ([1e-6] if len(names) == 3 else []))

    def residual(x):
        r = x[0] * factor
        decay = np.exp(-r * dt / x[1])
        ke = x[2] if len(x) == 3 else spec.motor.ke
        return decay * i0 + (1 - decay) * (u - ke * w) / r - i1

    result = least_squares(
        residual,
        initial,
        bounds=(np.full(len(names), 1e-12), np.full(len(names), np.inf)),
        x_scale="jac",
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
        max_nfev=500,
    )
    if not result.success:
        raise ValueError("Electrical fit did not converge")
    _rank(result.jac)
    diag["residual_rmse_a"] = float(np.sqrt(np.mean(result.fun**2)))
    return dict(zip(names, result.x)), diag


def fit_kt(spec, traces):
    trusted = [t for t in traces if t.get("torque_label") == "motor_electromagnetic"]
    samples = [
        r
        for t in trusted
        for r in t["samples"]
        if "current_a" in r and "electromagnetic_torque_nm" in r and abs(r["current_a"]) > 0.05
    ]
    if len(samples) < 8 or np.ptp([r["current_a"] for r in samples]) < 0.1:
        raise ValueError(
            "Kt needs explicitly labelled motor electromagnetic torque calibration with current excitation"
        )
    x, diag = _linear(
        [[r["current_a"], 1] for r in samples],
        [r["electromagnetic_torque_nm"] for r in samples],
        lower=-np.inf,
    )
    if x[0] <= 0:
        raise ValueError("Kt fit is nonphysical; verify torque/current coordinate conventions")
    diag["torque_sensor_bias_nm"] = float(x[1])
    return {"motor.kt": x[0]}, diag


def fit_mechanical(spec, traces):
    if spec.friction.model not in ("none", "coulomb", "stribeck"):
        raise ValueError("Mechanical regression supports none/coulomb/stribeck only")
    matrix, target = [], []
    n, eta = spec.transmission.ratio, spec.transmission.efficiency
    for t, a, b in _pairs(traces, ("velocity_rad_s", "current_a", "external_torque_nm")):
        if t["bench"].get("locked"):
            continue
        v0, v1 = a["velocity_rad_s"], b["velocity_rad_s"]
        cutoff = max(0.02, 5 * spec.friction.stribeck_velocity if spec.friction.model == "stribeck" else 0.02)
        if abs(v1) < cutoff or v0 * v1 <= 0:
            continue
        acceleration = (v1 - v0) / t["bench"]["dt"]
        rotor_torque = spec.motor.kt * b["current_a"]
        factor = eta if rotor_torque * v0 >= 0 else 1 / eta
        drive = n * factor * rotor_torque
        known_inertia = t["bench"]["load_inertia"] + spec.transmission.extra_joint_armature
        row = [n * n * acceleration]
        if spec.friction.model != "none":
            row += [np.sign(v1), v1]
        matrix.append(row)
        target.append(drive + a["external_torque_nm"] - known_inertia * acceleration)
    if len(matrix) < 20:
        raise ValueError(
            "Need >=20 moving samples with known load and independent acceleration/speed excitation"
        )
    result, diag = _linear(matrix, target)
    names = ["motor.rotor_inertia"] + (
        [] if spec.friction.model == "none" else ["friction.coulomb", "friction.viscous"]
    )
    return dict(zip(names, result)), diag


def fit_thermal(spec, traces):
    pairs = list(_pairs(traces, ("current_a", "temperature_c")))
    if len(pairs) < 30:
        raise ValueError("Need current and temperature records including heating and cooling")
    t0 = np.array([a["temperature_c"] for _, a, _ in pairs])
    t1 = np.array([b["temperature_c"] for _, _, b in pairs])
    i1 = np.array([b["current_a"] for _, _, b in pairs])
    dt = np.array([t["bench"]["dt"] for t, _, _ in pairs])
    ambient = spec.motor.ambient_temperature
    r = spec.motor.resistance * (1 + spec.motor.resistance_tempco * (t0 - ambient))
    power = i1 * i1 * r
    if np.ptp(t0) < 1 or not np.any((power < 0.01) & (t0 > ambient + 0.5)) or not np.any(power > 0.1):
        raise ValueError("Need >=1 C temperature span and both heating and elevated-temperature cooling")
    _, diag = _rank(np.column_stack((power, t0 - ambient)))

    def residual(x):
        rt, ct = x
        equilibrium = ambient + power * rt
        return equilibrium + (t0 - equilibrium) * np.exp(-dt / (rt * ct)) - t1

    result = least_squares(
        residual,
        [spec.motor.thermal_resistance or 3, spec.motor.thermal_capacity or 2],
        bounds=([1e-6, 1e-6], [1e6, 1e8]),
        x_scale="jac",
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )
    if not result.success:
        raise ValueError("Thermal fit did not converge")
    _rank(result.jac)
    diag["residual_rmse_c"] = float(np.sqrt(np.mean(result.fun**2)))
    return dict(zip(("motor.thermal_resistance", "motor.thermal_capacity"), result.x)), diag


def replay(spec, trace, backend="cpu", device="cuda:0"):
    # Controller gains/domain are measured experiment metadata. Identified
    # response and delay override their baseline metadata values.
    controller = Controller(
        **{
            **trace["controller"],
            "response_time": spec.controller.response_time,
            "delay_steps": spec.controller.delay_steps,
            "max_delay_steps": spec.controller.max_delay_steps,
        }
    )
    bench = Bench(**trace["bench"])
    return simulate(
        replace(spec, controller=controller),
        bench,
        commands_from_trace(trace),
        backend=backend,
        sensors=Sensors(),
        name=trace["name"],
        device=device,
    )


def evaluate(spec, traces, backend="cpu", device="cuda:0"):
    scores = []
    reports = []
    for trace in traces:
        predicted = replay(spec, trace, backend, device)
        metrics = {}
        for key in ("position_rad", "velocity_rad_s", "current_a", "temperature_c"):
            pairs = [
                (a[key], b[key])
                for a, b in zip(trace["samples"], predicted["samples"])
                if key in a and key in b
            ]
            if not pairs:
                continue
            a, b = np.array(pairs).T
            rmse = float(np.sqrt(np.mean((a - b) ** 2)))
            scale = max(
                float(np.ptp(a)),
                {"position_rad": 0.05, "velocity_rad_s": 0.1, "current_a": 0.1, "temperature_c": 1.0}[key],
            )
            metrics[key] = {"rmse": rmse, "nrmse": rmse / scale}
        if not metrics:
            raise ValueError("Validation needs at least one measured response channel")
        # Equal weight per complete record and observed channel, including locked electrical tests.
        scores.append(float(np.mean([x["nrmse"] ** 2 for x in metrics.values()])))
        reports.append({"name": trace["name"], "metrics": metrics})
    return {"score": float(np.mean(scores)), "traces": reports, "backend": backend}


def fit_trajectory(spec, traces, bounds, *, generations=30, population=12, seed=42, progress=None):
    from cmaes import CMA

    allowed = {
        "friction.stribeck",
        "friction.stribeck_velocity",
        "controller.response_time",
        "controller.delay_steps",
    }
    if not bounds or set(bounds) - allowed or generations < 1 or population < 4:
        raise ValueError("CMA-ES requires supported bounds, generations>=1, population>=4")
    if any(k.startswith("friction.stribeck") for k in bounds) and spec.friction.model != "stribeck":
        raise ValueError("Enable the Stribeck model before fitting its parameters")
    if "controller.response_time" in bounds and any(t["controller"]["domain"] == "voltage" for t in traces):
        raise InsufficientExcitation("Current-loop response_time is not observable in direct voltage drive")
    names = list(bounds)
    ranges = np.array([bounds[k] for k in names], dtype=float)
    if (
        ranges.shape != (len(names), 2)
        or not np.isfinite(ranges).all()
        or (ranges[:, 0] >= ranges[:, 1]).any()
    ):
        raise ValueError("Invalid CMA-ES bounds")
    for end in (0, 1):
        change_parameters(spec, dict(zip(names, ranges[:, end])))
    initial = np.array([get_parameter(spec, k) for k in names])
    if ((initial < ranges[:, 0]) | (initial > ranges[:, 1])).any():
        raise ValueError("Initial parameters must be inside bounds")
    width = ranges[:, 1] - ranges[:, 0]
    if "controller.delay_steps" in names:
        delay_bounds = ranges[names.index("controller.delay_steps")]
        if not np.equal(delay_bounds, np.round(delay_bounds)).all():
            raise ValueError("Delay search bounds must be integer physics steps")

    def response_vector(model):
        values = []
        for trace in traces:
            predicted = replay(model, trace)
            for key in ("position_rad", "velocity_rad_s", "current_a", "temperature_c"):
                observed = [r[key] for r in trace["samples"] if key in r]
                if not observed:
                    continue
                scale = max(float(np.ptp(observed)), 0.1)
                values.extend(
                    b[key] / scale
                    for a, b in zip(trace["samples"], predicted["samples"])
                    if key in a and key in b
                )
        if not values:
            raise InsufficientExcitation("Trajectory fitting needs measured response channels")
        return np.asarray(values)

    base_response = response_vector(spec)
    columns = []
    for index, name in enumerate(names):
        step = 1.0 if name == "controller.delay_steps" else width[index] * 0.02
        value = initial[index] + step
        if value > ranges[index, 1]:
            value = initial[index] - step
        perturbation = response_vector(change_parameters(spec, {name: value})) - base_response
        if np.linalg.norm(perturbation) < 1e-10:
            raise InsufficientExcitation(f"No observable trajectory sensitivity to {name}")
        columns.append(perturbation)
    _, sensitivity = _rank(np.column_stack(columns))
    optimizer = CMA(
        mean=(initial - ranges[:, 0]) / width,
        sigma=0.2,
        bounds=np.tile([0.0, 1.0], (len(names), 1)),
        population_size=population,
        seed=seed,
    )
    best, best_score = spec, evaluate(spec, traces)["score"]
    history = []
    for generation in range(generations):
        solutions = []
        for _ in range(population):
            x = optimizer.ask()
            try:
                candidate = change_parameters(spec, dict(zip(names, ranges[:, 0] + x * width)))
                score = evaluate(candidate, traces)["score"]
            except (ValueError, OverflowError, RuntimeError):
                score = 1e20
            solutions.append((x, score))
            if score < best_score:
                best, best_score = candidate, score
        optimizer.tell(solutions)
        history.append(best_score)
        if progress:
            progress(generation + 1, generations, best_score)
    return {n: get_parameter(best, n) for n in names}, {
        "method": "CMA-ES",
        "train_loss": history,
        "generations": generations,
        "population": population,
        "seed": seed,
        "local_sensitivity": sensitivity,
    }


def fit(
    dataset,
    spec=None,
    *,
    output=None,
    trajectory_bounds=None,
    generations=30,
    population=12,
    seed=42,
    progress=None,
):
    manifest, train, validation = split_dataset(dataset)
    baseline = spec or ActuatorSpec.from_dict(manifest["initial_model"])
    if baseline.motor is None:
        raise ValueError("Motor SysID requires an electrical motor model")
    candidate = baseline
    status = {
        n: {
            "status": "kept",
            "reason": "Not requested or not supported by this experiment",
            "initial": get_parameter(baseline, n),
        }
        for n in PARAMETERS
    }
    diagnostics = {}
    stages = [
        ("electrical", fit_electrical, PARAMETERS[:3]),
        ("torque_constant", fit_kt, ["motor.kt"]),
        ("mechanical", fit_mechanical, ["motor.rotor_inertia", "friction.coulomb", "friction.viscous"]),
        ("thermal", fit_thermal, ["motor.thermal_resistance", "motor.thermal_capacity"]),
    ]
    for stage, func, names in stages:
        try:
            values, diag = func(candidate, train)
            candidate = change_parameters(candidate, values)
            if stage == "mechanical":
                diag["fixed_parameters"] = {
                    "motor.kt": candidate.motor.kt,
                    "kt_source": "fitted" if status["motor.kt"]["status"] == "fitted" else "prior",
                    "transmission.ratio": candidate.transmission.ratio,
                    "transmission.efficiency": candidate.transmission.efficiency,
                    "transmission.extra_joint_armature": candidate.transmission.extra_joint_armature,
                }
                diag["known_load_inertia"] = {t["name"]: t["bench"]["load_inertia"] for t in train}
            diagnostics[stage] = diag
            for n in names:
                status[n] = {
                    "status": "fitted" if n in values else "unidentifiable",
                    "reason": stage if n in values else "Required excitation absent; prior retained",
                    "initial": get_parameter(baseline, n),
                    "candidate": get_parameter(candidate, n),
                }
        except ValueError as exc:
            diagnostics[stage] = {"reason": str(exc)}
            for n in names:
                status[n].update(status="unidentifiable", reason=str(exc))
    if trajectory_bounds:
        try:
            values, diag = fit_trajectory(
                candidate,
                train,
                trajectory_bounds,
                generations=generations,
                population=population,
                seed=seed,
                progress=progress,
            )
            candidate = change_parameters(candidate, values)
            diagnostics["trajectory"] = diag
            for n, v in values.items():
                status[n].update(status="fitted", reason="CMA-ES on training records only", candidate=v)
        except InsufficientExcitation as exc:
            diagnostics["trajectory"] = {"reason": str(exc)}
            for name in trajectory_bounds:
                status[name].update(status="unidentifiable", reason=str(exc))
    before, after = evaluate(baseline, validation), evaluate(candidate, validation)
    regressions = []
    for old, new in zip(before["traces"], after["traces"]):
        for key, metric in old["metrics"].items():
            # A global average must not hide a worsened response channel.
            if new["metrics"][key]["nrmse"] > metric["nrmse"] * 1.001 + 1e-7:
                regressions.append({"trace": old["name"], "channel": key})
    accepted = not regressions and (
        after["score"] < before["score"] or (before["score"] <= 1e-16 and after["score"] <= 1e-16)
    )
    selected = candidate if accepted else baseline
    for n in status:
        status[n]["selected"] = get_parameter(selected, n)
    report = {
        "schema": "kicker.fit.v1",
        "accepted": accepted,
        "selected_model": selected.to_dict(),
        "candidate_model": candidate.to_dict(),
        "parameters": status,
        "diagnostics": diagnostics,
        "validation_before": before,
        "validation_after": after,
        "regressed_channels": regressions,
        "validation_rmse_ratio": math.sqrt(after["score"] / before["score"])
        if before["score"] > 0
        else (0.0 if after["score"] == 0 else None),
        "train_records": [t["name"] for t in train],
        "validation_records": [t["name"] for t in validation],
        "seed": seed,
        "rejection_reason": None
        if accepted
        else "Candidate did not improve independent validation; baseline retained",
    }
    if output:
        output = Path(output)
        write_json(output / "fit_report.json", report)
        write_json(output / "model.json", selected.to_dict())
    return report
