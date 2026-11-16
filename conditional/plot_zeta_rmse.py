#!/usr/bin/env python3
"""Redraw Figure 3 and appendix Figure A3 from the published estimation summaries.

Figure 3 is zeta_x. Figure A3 is zeta_w. Rows are left-skewed then right-skewed.
Each location is read from the estimation summary that the paper table uses.
"""
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
import pandas as pd

PROJECT = Path(os.environ.get(
    "MC_PROJECT",
    "/home/eason/variable importance/code_new/marginal_mc_project",
)).expanduser().resolve()
RESULTS = PROJECT / "results"
SOURCES = {
    -0.5: "conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s0m05_0_r1000",
    0.0: "conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s00_wcv05_0625_r16_32_r1000",
    0.5: "conditional_est_y20_w_hfhalf_cn075_spacing_seonly_seed20261006_s05_r1000",
}
H_CONST = 1.5
H_EXPONENT = {"semi": -0.25, "conf": -0.4}
STYLE = {
    "semi": dict(color="#1f77b4", marker="o", ls="-", label="Semi"),
    "conf": dict(color="#ff7f0e", marker="s", ls="--", label="Conf"),
}


def load(sources: dict[float, str]) -> pd.DataFrame:
    parts = []
    for s0, name in sources.items():
        frame = pd.read_csv(RESULTS / name / "summaries" / "summary.csv")
        frame = frame[(frame.s0 - s0).abs() < 1e-9].copy()
        if frame.empty or not (frame.reps == 1000).all():
            raise RuntimeError(f"unexpected summary for s0={s0} in {name}")
        if "sqrt_nh_rmse_x" not in frame:
            nh = frame.n * H_CONST * frame.n ** frame.method.map(H_EXPONENT)
            frame = frame.assign(
                sqrt_nh_rmse_x=nh ** 0.5 * frame.rmse_x,
                sqrt_nh_rmse_w=nh ** 0.5 * frame.rmse_w,
            )
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def plot(summary: pd.DataFrame, column: str, dest: Path) -> None:
    s0s = [-0.5, 0.0, 0.5]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True, sharey=True)
    rows = (("left", "Left-skewed errors"), ("right", "Right-skewed errors"))
    for i, (error, label) in enumerate(rows):
        for j, s0 in enumerate(s0s):
            ax = axes[i, j]
            for method, style in STYLE.items():
                group = summary[
                    (summary.error == error) & ((summary.s0 - s0).abs() < 1e-9)
                    & (summary.method == method)
                ].sort_values("n")
                ax.plot(group.n, group[column], ms=6, **style)
            ax.set_xscale("log", base=10)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"$s_0={s0:g}$")
            else:
                ax.set_xlabel(r"Sample size $n$ (log scale)")
            if j == 0:
                ax.set_ylabel(label + "\n" + r"$\sqrt{nh}\times$ RMSE")
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    if dest.suffix.lower() == ".png":
        fig.savefig(dest.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    summary = load(SOURCES)
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot(summary, "sqrt_nh_rmse_x", args.outdir / "cond_zeta_sqrt_nh_rmse.png")
    plot(summary, "sqrt_nh_rmse_w", args.outdir / "cond_zeta_w_sqrt_nh_rmse.png")


if __name__ == "__main__":
    main()
