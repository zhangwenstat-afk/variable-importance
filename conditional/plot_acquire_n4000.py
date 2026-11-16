#!/usr/bin/env python3
"""Paper acquisition figure at n = 4000 on the 11-point grid."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()

import argparse
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get(
    "MC_PROJECT",
    "/data/eason/mc_results",
)).expanduser().resolve()
DEFAULT_RUN = (
    "conditional_acquire_y20_w_hfhalf_cn075_spacing_seed20261006_n4000_11pt_r1000"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    summary = PROJECT / args.run / "summaries" / "summary.csv"
    frame = pd.read_csv(summary)
    frame["delta_true"] = 0.2 + frame["gap"]
    errors = [("left", "Left-skewed"), ("right", "Right-skewed")]
    methods = [
        ("semi", "semi", "o", "#1f77b4"),
        ("conf", "conf", "s", "#ff7f0e"),
    ]
    z = 1.96
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True, sharey="row")
    for j, (error, title) in enumerate(errors):
        block = frame[frame.error == error].sort_values("s0")
        semi = block[block.method == "semi"].sort_values("s0")
        ax = axes[0, j]
        ax.axhline(0.20, color="0.45", ls=":", lw=0.8)
        ax.axvline(0.0, color="0.45", ls=":", lw=0.8)
        ax.plot(semi.s0, semi.delta_true, color="black", lw=1.4, marker="D", ms=3.6,
                label="true value")
        for method, label, marker, color in methods:
            group = block[block.method == method].sort_values("s0")
            mean = group.mean_delta.to_numpy()
            half = group.ci_length_delta.to_numpy() / 2.0
            ax.fill_between(group.s0, mean - half, mean + half, color=color, alpha=0.16, lw=0)
            ax.plot(group.s0, mean, color=color, lw=1.5, marker=marker, ms=4.2, label=label)
        ax.set_title(title)
        ax.set_ylim(-0.16, 0.64)
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.set_ylabel(r"Gain $\Delta(s)$")
            ax.legend(frameon=False, fontsize=8)
        ax = axes[1, j]
        ax.axhline(0.05, color="0.45", ls=":", lw=0.8)
        ax.axvline(0.0, color="0.45", ls=":", lw=0.8)
        for method, label, marker, color in methods:
            group = block[block.method == method].sort_values("s0")
            probability = group.size_or_power.to_numpy()
            reps = group.reps.to_numpy()
            se = np.sqrt(probability * (1.0 - probability) / reps)
            ax.errorbar(group.s0, probability, yerr=z * se, color=color, lw=1.5,
                        marker=marker, ms=4.2, capsize=2, label=label)
        ax.set_ylim(-0.02, 0.58)
        ax.set_xlabel(r"$s$")
        ax.set_xticks(np.linspace(-0.5, 0.5, 5))
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.set_ylabel("Rejection probability")
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outdir / "acquire_n4000.png", dpi=200)
    fig.savefig(args.outdir / "acquire_n4000.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
