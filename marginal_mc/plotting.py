"""One-axes-per-file matplotlib figures, with no manually selected color palette."""
from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reporting import read_raw


def slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _read_optional(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _finish(fig, ax, path: Path, title: str, xlabel: str, ylabel: str, note: str = "") -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best")
    if note:
        fig.text(0.02, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.035 if note else 0, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_plots(output: str | Path) -> list[str]:
    output = Path(output)
    summaries = output/"summaries"
    if not (summaries/"estimation_summary.csv").exists():
        from .reporting import summarize
        summarize(output)
    est = _read_optional(summaries/"estimation_summary.csv")
    infer = _read_optional(summaries/"inference_summary.csv")
    folder = output/"figures"
    folder.mkdir(exist_ok=True)
    files = []
    estimation = est[est["study_type"] == "estimation"]
    for keys, block in estimation.groupby(["study", "error", "kind", "target"], sort=True):
        study, error, kind, target = keys
        for metric, label in [("rmse", "RMSE"), ("sqrt_n_rmse", "sqrt(n) x RMSE"),
                              ("sqrt_n_bias", "sqrt(n) x Bias"),
                              (f"sqrt_n_remainder_{target}_rms", "RMS of sqrt(n) expansion remainder")]:
            if metric not in block or block[metric].notna().sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(7.0, 4.6))
            for (layer, method), group in block.groupby(["layer", "method"], sort=True):
                group = group.sort_values("n")
                ax.plot(group["n"], group[metric], marker="o",
                        linestyle="-" if method == "semi" else "--",
                        label=f"{layer}: {method}")
            ax.set_xscale("log", base=2)
            if metric == "rmse":
                ax.set_yscale("log")
                ns = np.sort(block["n"].unique())
                anchor = float(block.loc[block.n == ns[0], "rmse"].median())
                ax.plot(ns, anchor*(ns/ns[0])**(-0.5), linestyle=":", label="reference n^(-1/2)")
                if kind == "controlled":
                    # A common illustrative reference; exact gamma is in the layer ID.
                    ax.plot(ns, anchor*(ns/ns[0])**(-0.35), linestyle="-.", label="reference n^(-0.35)")
            elif metric == "sqrt_n_bias":
                ax.axhline(0, linestyle=":")
            filename = f"est_{slug(study)}_{error}_{kind}_{target}_{slug(metric)}.png"
            note = f"Valid replications per point: {int(block.R_est_valid.min())} to {int(block.R_est_valid.max())}."
            _finish(fig, ax, folder/filename, f"{error} | {kind} | target={target}", "n", label, note)
            files.append(filename)
    if len(infer):
        for keys, block in infer.groupby(["study", "error", "n", "layer"], sort=True):
            study, error, n, layer = keys
            local = (block["gap_type"] == "local").all()
            xcol, xlabel = (("gap_param", "Local alternative c: Delta = Delta0 + c/sqrt(n)")
                           if local else ("gap", "True Delta - Delta0"))
            fig, ax = plt.subplots(figsize=(7.0, 4.7))
            for method, group in block.groupby("method"):
                group = group.sort_values(xcol)
                for col, suffix, style in [("reject_nominal", "nominal", "-"),
                                           ("reject_size_adjusted", "size-adjusted", "--")]:
                    if col in group and group[col].notna().any():
                        ax.errorbar(group[xcol], group[col],
                                    yerr=1.96*group[col+"_mcse"], marker="o",
                                    capsize=2, linestyle=style, label=f"{method}: {suffix}")
                if local and method == "semi" and "oracle_normal_benchmark" in group:
                    ax.plot(group[xcol], group["oracle_normal_benchmark"], linestyle=":",
                            label="semi oracle-normal benchmark")
            ax.axhline(float(block.test_level.iloc[0]), linestyle=":", label="nominal level")
            ax.axvline(0, linestyle=":")
            ax.set_ylim(-0.03, 1.03)
            filename = f"inference_{slug(study)}_{error}_n{n}_{slug(layer)}.png"
            note = (f"R={int(block.R_requested.iloc[0])}; bars: 1.96 x MCSE; "
                    "size-adjusted bars condition on calibration.")
            _finish(fig, ax, folder/filename,
                    f"{error} | n={n} | {layer} | Delta0={block.delta0.iloc[0]:g}",
                    xlabel, "Rejection rate", note)
            files.append(filename)
    diag = _read_optional(summaries/"nuisance_diagnostics.csv")
    if len(diag):
        diag = diag[(diag.study_type == "estimation") & (diag.kind != "oracle")]
        for keys, block in diag.groupby(["study", "error", "layer"], sort=True):
            study, error, layer = keys
            for suffix, title in [
                ("n_quarter_m_sup_grid_mean",
                 "n^(1/4) x grid mean-function error"),
                ("n_quarter_f_sup_grid_mean",
                 "n^(1/4) x full-grid density error"),
                ("n_quarter_f_q_sup_grid_mean",
                 "n^(1/4) x density error at +/- q"),
                ("n_quarter_a_sup_grid_mean",
                 "n^(1/4) x A(q) contrast error"),
            ]:
                fig, ax = plt.subplots(figsize=(7.0, 4.6))
                drew = False
                for target in ("x", "w"):
                    column = target+"_"+suffix
                    if column in block and block[column].notna().any():
                        group = block.sort_values("n")
                        ax.plot(group.n, group[column], marker="o", label=target)
                        drew = True
                if not drew:
                    plt.close(fig)
                    continue
                ax.set_xscale("log", base=2)
                filename = f"nuisance_{slug(study)}_{error}_{slug(layer)}_{suffix}.png"
                _finish(fig, ax, folder/filename, f"{error} | {layer}", "n", title,
                        "Finite-grid diagnostics, not a proof of uniform convergence.")
                files.append(filename)
    raw = read_raw(output)
    raw = raw[(raw.role == "evaluation") & (raw.study_type == "estimation")
              & (raw.method == "conf") & raw.estimator_valid & (raw.kind != "oracle")]
    for keys, block in raw.groupby(["study", "error", "layer"], sort=True):
        study, error, layer = keys
        block = block[block.n == block.n.max()]
        for target in ("x", "w", "delta"):
            a, b = f"bias_pred_{target}", f"noise_removed_{target}"
            if a not in block or b not in block:
                continue
            subset = block[[a, b]].dropna()
            if not len(subset):
                continue
            fig, ax = plt.subplots(figsize=(5.8, 5.0))
            ax.scatter(subset[a], subset[b], alpha=0.5, s=14, label="replications")
            lo, hi = float(subset.min().min()), float(subset.max().max())
            pad = max(0.01, 0.1*(hi-lo))
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], linestyle="--", label="45-degree reference")
            filename = f"theorem2_{slug(study)}_{error}_{slug(layer)}_{target}.png"
            _finish(fig, ax, folder/filename,
                    f"Theorem 2 | {error} | {layer} | n={int(block.n.max())} | {target}",
                    "Predicted first-order nuisance bias", "Estimate - truth - oracle quantile fluctuation",
                    "Constant shifts can give a vertical cloud; see remainder RMS for convergence.")
            files.append(filename)
    (folder/"figure_index.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(f"Wrote {len(files)} PNG figures to {folder}")
    return files
