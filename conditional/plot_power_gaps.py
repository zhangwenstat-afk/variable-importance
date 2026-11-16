#!/usr/bin/env python3
"""Paper conditional power figures at n = 1000, 4000 and 16000.

Rejection rates use inference-valid replications. Error bars are pointwise
95% Monte Carlo intervals. Rows are right-skewed then left-skewed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(os.environ.get(
    "MC_PROJECT",
    "/data/eason/mc_results",
)).expanduser().resolve()
DEFAULT_RUN = (
    "conditional_infer_y20_w_hfhalf_cn075_spacing_seed20261006_gaps_r1000"
)


def power_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = raw.groupby(["error", "n", "s0", "method", "gap"], sort=True)
    for keys, block in grouped:
        valid = block[block.inference_valid == True]
        reps = int(len(valid))
        reject = float(valid.reject_nominal.mean()) if reps else math.nan
        mcse = math.sqrt(reject * (1.0 - reject) / reps) if reps else math.nan
        record = dict(zip(["error", "n", "s0", "method", "gap"], keys))
        record.update(reps=int(len(block)), infer_reps=reps, reject=reject, mcse=mcse)
        rows.append(record)
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame, n: int, path: Path) -> None:
    errors = ("right", "left")
    s0s = (-0.5, 0.0, 0.5)
    labels = {-0.5: r"$s_0=-0.5$", 0.0: r"$s_0=0$", 0.5: r"$s_0=0.5$"}
    ylabels = {"right": "Right-skewed", "left": "Left-skewed"}
    styles = (
        ("semi", "o", "-", "#1f77b4"),
        ("conf", "s", "-", "#ff7f0e"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.4), sharex=True, sharey=True, squeeze=False)
    sub = table[table.n == n]
    for i, error in enumerate(errors):
        for j, s0 in enumerate(s0s):
            ax = axes[i, j]
            block = sub[(sub.error == error) & ((sub.s0 - s0).abs() < 1e-9)]
            for method, marker, linestyle, color in styles:
                group = block[block.method == method].sort_values("gap")
                ax.errorbar(
                    group.gap, group.reject, yerr=1.96 * group.mcse,
                    fmt=marker + linestyle, color=color, capsize=2, label=method,
                )
            ax.axhline(0.05, linestyle=":", color="0.35")
            ax.axvline(0.0, linestyle=":", color="0.35")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(labels[s0])
            if j == 0:
                ax.set_ylabel(f"{ylabels[error]}\nRejection rate")
            if i == len(errors) - 1:
                ax.set_xlabel(r"$\Delta(s_0)-\Delta_0$")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    raw_dir = PROJECT / args.run
    raw_path = raw_dir / "raw_all.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, usecols=[
            "error", "n", "s0", "method", "gap", "inference_valid", "reject_nominal",
        ])
    else:
        parts = [pd.read_csv(path, usecols=[
            "error", "n", "s0", "method", "gap", "inference_valid", "reject_nominal",
        ]) for path in sorted((raw_dir / "raw").glob("*.csv"))]
        raw = pd.concat(parts, ignore_index=True)
    table = power_table(raw)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "power_curves.csv", index=False)
    for n in (1000, 4000, 16000):
        draw(table, n, args.outdir / f"cond_delta_power_n{n}.png")


if __name__ == "__main__":
    main()
