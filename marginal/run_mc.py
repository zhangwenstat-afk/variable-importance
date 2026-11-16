#!/usr/bin/env python3
"""Run three-layer marginal Monte Carlo experiments; see README_zh.md."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()

import argparse
import json
import os
from pathlib import Path

# Avoid oversubscribed BLAS in both serial and multiprocess runs.
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from marginal_mc.simulation import execute
from marginal_mc.reporting import summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).parent/"configs"/"smoke.json")
    p.add_argument("--out", type=Path, default=Path("results/smoke"))
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-summary", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--seed", type=int, help="override the config seed")
    p.add_argument("--reps", type=int, help="override evaluation repetitions only")
    p.add_argument("--calibration-reps", type=int)
    p.add_argument("--ns", nargs="+", type=int)
    p.add_argument("--errors", nargs="+", choices=["symmetric", "right", "left"])
    p.add_argument("--layers", nargs="+", help="layer IDs from the selected config")
    p.add_argument("--studies", nargs="+", help="study names from the selected config")
    p.add_argument("--variance", choices=["empirical", "manuscript"])
    p.add_argument("--g-estimator", choices=["spacing", "oracle"])
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for attr, key in [("seed", "seed"), ("reps", "repetitions"), ("ns", "sample_sizes"),
                      ("errors", "errors"), ("variance", "variance"),
                      ("g_estimator", "g_estimator")]:
        value = getattr(args, attr)
        if value is not None:
            config[key] = value
    if args.calibration_reps is not None:
        config.setdefault("size_adjustment", {})["calibration_reps"] = args.calibration_reps
    for option, key, identifier in [(args.layers, "layers", "id"),
                                     (args.studies, "studies", "name")]:
        if option is not None:
            unknown = set(option)-{entry[identifier] for entry in config[key]}
            if unknown:
                p.error(f"unknown {key}: {sorted(unknown)}")
            config[key] = [entry for entry in config[key] if entry[identifier] in option]
    execute(config, args.out, args.jobs, args.resume, args.dry_run)
    if not args.dry_run and not args.no_summary:
        summarize(args.out)
    if not args.dry_run and args.plots:
        from marginal_mc.plotting import make_plots
        make_plots(args.out)


if __name__ == "__main__":
    main()
