"""Summaries and figures for the conditional experiment."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(output: str | Path) -> dict[str, Path]:
    out = Path(output)
    raw = pd.concat((pd.read_csv(p) for p in sorted((out / "raw").glob("*.csv"))),
                    ignore_index=True)
    raw.to_csv(out / "raw_all.csv", index=False)
    if "estimator_valid" in raw.columns:
        ok = raw[raw["estimator_valid"] == True].copy()
    else:
        ok = raw.copy()
    group = ["study_type", "error", "n", "s0", "method", "gap"]
    group = [c for c in group if c in ok.columns]
    rows = []
    for keys, block in ok.groupby(group, dropna=False):
        key = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        rec = {**key, "reps": len(block),
               "fail_rate": 1.0 - float(block["estimator_valid"].mean()) if "estimator_valid" in block else 0.0}
        for col, name in (("bias_x", "bias_x"), ("bias_w", "bias_w"),
                          ("bias_delta", "bias_delta"),
                          ("zeta_x_est", "mean_zeta_x"), ("zeta_w_est", "mean_zeta_w"),
                          ("delta_est", "mean_delta"),
                          ("se_x", "mean_se_x"), ("se_w", "mean_se_w"),
                          ("se_delta", "mean_se_delta"),
                          ("pred_cover_x", "pred_cover_x"), ("pred_cover_w", "pred_cover_w"),
                          ("pred_length_x", "pred_length_x"), ("pred_length_w", "pred_length_w"),
                          ("n_eff", "mean_n_eff"), ("n_window", "mean_n_window"),
                          ("h", "h"), ("nh", "nh"),
                          ("ci_x_cover", "ci_cover_x"), ("ci_w_cover", "ci_cover_w"),
                          ("ci_delta_cover", "ci_cover_delta"),
                          ("ci_delta_length", "ci_length_delta"),
                          ("reject_nominal", "size_or_power"),
                          ("T", "mean_T")):
            if col in block:
                rec[name] = float(block[col].mean())
        for col, name in (("zeta_x_est", "sd_x"), ("zeta_w_est", "sd_w"),
                          ("delta_est", "sd_delta"), ("T", "sd_T")):
            if col in block and len(block) > 1:
                rec[name] = float(block[col].std(ddof=1))
        if "bias_x" in block:
            rec["rmse_x"] = float(np.sqrt(np.mean(np.square(block["bias_x"]))))
            rec["rmse_w"] = float(np.sqrt(np.mean(np.square(block["bias_w"]))))
            rec["rmse_delta"] = float(np.sqrt(np.mean(np.square(block["bias_delta"]))))
        if "nh" in rec and "rmse_x" in rec:
            rec["sqrt_nh_rmse_x"] = math_sqrt(rec["nh"]) * rec["rmse_x"]
            rec["sqrt_nh_rmse_w"] = math_sqrt(rec["nh"]) * rec["rmse_w"]
            rec["sqrt_nh_rmse_delta"] = math_sqrt(rec["nh"]) * rec["rmse_delta"]
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values(group)
    dest = out / "summaries"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "summary.csv"
    summary.to_csv(path, index=False)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    _rmse_figure(summary, figdir / "rmse_by_s0.png")
    _sqrt_nh_figure(summary, figdir / "sqrt_nh_rmse_by_s0.png")
    _power_figure(summary, figdir / "delta_power.png")
    return {"summary": path}


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x)) if np.isfinite(x) and x > 0 else np.nan


def _rmse_figure(summary: pd.DataFrame, dest: Path) -> None:
    est = summary[summary.study_type == "estimation"] if "study_type" in summary else summary
    if est.empty or "rmse_x" not in est:
        return
    s0s = sorted(est.s0.unique())
    methods = [m for m in ("semi", "conf", "conf_hsemi") if m in set(est.method)]
    fig, axes = plt.subplots(len(s0s), 2, figsize=(9.6, 2.8 * len(s0s)),
                             sharex=True, squeeze=False)
    for i, s0 in enumerate(s0s):
        for j, (col, title) in enumerate((("rmse_x", r"RMSE of $\zeta_x(s_0)$"),
                                          ("rmse_w", r"RMSE of $\zeta_w(s_0)$"))):
            ax = axes[i, j]
            block = est[est.s0 == s0]
            for method in methods:
                g = block[block.method == method].sort_values("n")
                if g.empty:
                    continue
                ax.plot(g.n, g[col], marker="o", label=method)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"$s_0={s0:g}$")
            if i == len(s0s) - 1:
                ax.set_xlabel("Sample size n")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def _sqrt_nh_figure(summary: pd.DataFrame, dest: Path) -> None:
    est = summary[summary.study_type == "estimation"] if "study_type" in summary else summary
    if est.empty or "sqrt_nh_rmse_x" not in est:
        return
    s0s = sorted(est.s0.unique())
    methods = [m for m in ("semi", "conf", "conf_hsemi") if m in set(est.method)]
    fig, axes = plt.subplots(len(s0s), 2, figsize=(9.6, 2.8 * len(s0s)),
                             sharex=True, squeeze=False)
    for i, s0 in enumerate(s0s):
        for j, col in enumerate(("sqrt_nh_rmse_x", "sqrt_nh_rmse_w")):
            ax = axes[i, j]
            block = est[est.s0 == s0]
            for method in methods:
                g = block[block.method == method].sort_values("n")
                if g.empty:
                    continue
                ax.plot(g.n, g[col], marker="o", label=method)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(r"$\sqrt{nh}\times$RMSE " + (r"$\zeta_x$" if j == 0 else r"$\zeta_w$"))
            if j == 0:
                ax.set_ylabel(f"$s_0={s0:g}$")
            if i == len(s0s) - 1:
                ax.set_xlabel("Sample size n")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def _power_figure(summary: pd.DataFrame, dest: Path) -> None:
    inf = summary[summary.study_type == "inference"] if "study_type" in summary else pd.DataFrame()
    if inf.empty or "size_or_power" not in inf:
        return
    ns = [n for n in (1000, 4000, 16000) if n in set(inf.n.unique())]
    s0s = sorted(inf.s0.unique())
    if not ns:
        ns = sorted(inf.n.unique())[:3]
    fig, axes = plt.subplots(len(s0s), len(ns), figsize=(3.8 * len(ns), 2.7 * len(s0s)),
                             sharex=True, sharey=True, squeeze=False)
    methods = [m for m in ("semi", "conf", "conf_hsemi") if m in set(inf.method)]
    for i, s0 in enumerate(s0s):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            block = inf[(inf.s0 == s0) & (inf.n == n)]
            for method in methods:
                g = block[block.method == method].sort_values("gap")
                if g.empty:
                    continue
                ax.plot(g.gap, g.size_or_power, marker="o", label=method)
            ax.axhline(0.05, linestyle=":", color="0.4")
            ax.axvline(0.0, linestyle=":", color="0.4")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(f"n={n}")
            if j == 0:
                ax.set_ylabel(f"$s_0={s0:g}$  rejection")
            if i == len(s0s) - 1:
                ax.set_xlabel(r"$\Delta(s_0)-\Delta_0$")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
