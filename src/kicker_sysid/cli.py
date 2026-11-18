"""Command line entry points. Failures always have a nonzero exit status."""

import argparse
import json
import sys
from pathlib import Path

from .data import read_json, save_trace
from .identification import fit
from .simulation import simulate
from .spec import ActuatorSpec, Bench, Sensors
from .workflow import demo, export, validate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kicker-sysid", description="Motor simulation and identification")
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "simulate", "validate"):
        p = subs.add_parser(command)
        p.add_argument(
            "--backend",
            choices=["cpu", "mujoco", "mjlab"] + (["all"] if command != "simulate" else []),
            default="cpu",
        )
        p.add_argument("--device", default="cuda:0")
        if command == "demo":
            p.add_argument("--output", default="runs/demo")
        elif command == "simulate":
            p.add_argument("--config", required=True)
            p.add_argument("--output", default="runs/simulation")
            p.add_argument("--viewer", action="store_true")
        else:
            p.add_argument("--run", required=True)
    p = subs.add_parser("fit")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model")
    p.add_argument("--output")
    p.add_argument("--trajectory-bounds", help="JSON object with supported parameter bounds")
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p = subs.add_parser("export")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo(args.output, backend=args.backend, device=args.device)
        elif args.command == "simulate":
            config = read_json(args.config)
            if args.viewer and args.backend != "mujoco":
                raise ValueError("--viewer requires --backend mujoco")
            model = ActuatorSpec.from_dict(config["model"])
            trace = simulate(
                model,
                Bench(**config.get("bench", {})),
                backend=args.backend,
                sensors=Sensors(**config.get("sensors", {})),
                experiment=config.get("experiment"),
                name=config.get("name", "simulation"),
                device=args.device,
                viewer=args.viewer,
            )
            save_trace(trace, args.output)
            export(model, Path(args.output) / "export", Bench(**config.get("bench", {})))
            result = {
                "passed": True,
                "backend": args.backend,
                "samples": len(trace["samples"]),
                "output": args.output,
            }
        elif args.command == "fit":
            result = fit(
                args.dataset,
                ActuatorSpec.load(args.model) if args.model else None,
                output=args.output or Path(args.dataset).parent,
                trajectory_bounds=read_json(args.trajectory_bounds) if args.trajectory_bounds else None,
                generations=args.generations,
                population=args.population,
                seed=args.seed,
                progress=lambda g, n, loss: print(f"CMA-ES {g}/{n}: {loss:.6g}", flush=True),
            )
            print(
                json.dumps(
                    {
                        "accepted": result["accepted"],
                        "validation_rmse_ratio": result["validation_rmse_ratio"],
                    },
                    indent=2,
                )
            )
            return 0 if result["accepted"] else 1
        elif args.command == "validate":
            result = validate(args.run, backend=args.backend, device=args.device, progress=print)
        else:
            result = export(ActuatorSpec.load(args.model), args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("passed", True) else 1
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f"kicker-sysid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
