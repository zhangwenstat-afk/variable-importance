#!/usr/bin/env python3
"""Run the conditional (S=X) Monte Carlo experiment."""
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

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from conditional_mc.simulation import execute
from conditional_mc.reporting import summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-summary", action="store_true")
    p.add_argument("--reps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--c-semi", type=float)
    p.add_argument("--c-conf", type=float)
    p.add_argument("--c-n-factor", type=float)
    p.add_argument("--ns", nargs="+", type=int)
    p.add_argument("--errors", nargs="+", choices=["symmetric", "right", "left"])
    p.add_argument("--locations", nargs="+", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--shard", help="K/N, run batches whose md5 falls in bucket K")
    p.add_argument("--shard-except", help="K/N, run every remaining batch except bucket K")
    p.add_argument("--only-files", type=Path, help="text file of batch csv names to run")
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.reps is not None:
        config["repetitions"] = args.reps
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed
    if args.c_semi is not None:
        config["c_semi"] = args.c_semi
    if args.c_conf is not None:
        config["c_conf"] = args.c_conf
    if args.c_n_factor is not None:
        config["c_n_factor"] = args.c_n_factor
    if args.ns is not None:
        config["sample_sizes"] = args.ns
    if args.errors is not None:
        config["errors"] = args.errors
    if args.locations is not None:
        config["locations"] = args.locations
    execute(config, args.out, args.jobs, args.resume, args.dry_run,
            shard=args.shard, shard_except=args.shard_except, only_files=args.only_files)
    if not args.dry_run and not args.no_summary:
        summarize(args.out)


if __name__ == "__main__":
    main()
