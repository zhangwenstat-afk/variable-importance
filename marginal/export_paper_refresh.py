#!/usr/bin/env python3
"""Build paper Table 1 / Table 2 snippets and the two main figures from an MC run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ERROR_LABEL = {
    "symmetric": "Symmetric",
    "left": "Left-skewed",
    "right": "Right-skewed",
}
ERROR_ORDER = ["symmetric", "left", "right"]
NS = [1000, 2000, 4000, 8000, 16000]


def present_errors(frame: pd.DataFrame) -> list[str]:
    found = set(frame.error.unique())
    return [error for error in ERROR_ORDER if error in found]


def _pair(semi, conf, digits: int) -> str:
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(semi)}/{fmt.format(conf)}"


def _get(block: pd.DataFrame, method: str, column: str, scale: float = 1.0) -> float:
    row = block.loc[block.method == method]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column]) * scale


def write_table1(est: pd.DataFrame, folder: Path) -> Path | None:
    est = est[(est.study_type == "estimation") & (est.target.isin(["x", "w"]))].copy()
    if est.empty:
        return None
    lines = [
        r"\begin{tabular}{lccccccccc}",
        r"\hline",
        r"& & \multicolumn{4}{c}{$\zeta_\x$} & \multicolumn{4}{c}{$\zeta_\w$} \\",
        r"\cline{3-6}\cline{7-10}",
        r"Error & $n$ & Bias & SD & SE & RMSE & Bias & SD & SE & RMSE \\",
        r"\hline",
    ]
    rows_csv = []
    for error in present_errors(est):
        sub = est[est.error == error]
        ns = sorted(sub.n.unique())
        for i, n in enumerate(ns):
            block_x = sub[(sub.n == n) & (sub.target == "x")]
            block_w = sub[(sub.n == n) & (sub.target == "w")]
            cells = []
            for block in (block_x, block_w):
                for col in ("bias", "sd", "mean_se", "rmse"):
                    cells.append(_pair(_get(block, "semi", col, 100),
                                       _get(block, "conf", col, 100), 2))
            prefix = (rf"\multirow{{{len(ns)}}}{{*}}{{{ERROR_LABEL[error]}}} & {n}"
                      if i == 0 else f" & {n}")
            lines.append(" & ".join([prefix, *cells]) + r" \\")
            rows_csv.append({
                "error": error, "n": int(n),
                "x_bias_semi": _get(block_x, "semi", "bias"),
                "x_bias_conf": _get(block_x, "conf", "bias"),
                "x_sd_semi": _get(block_x, "semi", "sd"),
                "x_sd_conf": _get(block_x, "conf", "sd"),
                "x_se_semi": _get(block_x, "semi", "mean_se"),
                "x_se_conf": _get(block_x, "conf", "mean_se"),
                "x_rmse_semi": _get(block_x, "semi", "rmse"),
                "x_rmse_conf": _get(block_x, "conf", "rmse"),
                "w_bias_semi": _get(block_w, "semi", "bias"),
                "w_bias_conf": _get(block_w, "conf", "bias"),
                "w_sd_semi": _get(block_w, "semi", "sd"),
                "w_sd_conf": _get(block_w, "conf", "sd"),
                "w_se_semi": _get(block_w, "semi", "mean_se"),
                "w_se_conf": _get(block_w, "conf", "mean_se"),
                "w_rmse_semi": _get(block_w, "semi", "rmse"),
                "w_rmse_conf": _get(block_w, "conf", "rmse"),
            })
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    folder.mkdir(parents=True, exist_ok=True)
    tex = folder / "table1_zeta.tex"
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows_csv).to_csv(folder / "table1_zeta.csv", index=False)
    return tex


def write_table2(infer: pd.DataFrame, folder: Path) -> Path | None:
    boundary = infer[(infer.gap.abs() < 1e-12)].copy()
    if boundary.empty:
        return None
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Error & $n$ & Size & Bias & Coverage & CI length & SD($T$) \\",
        r"\hline",
    ]
    rows_csv = []
    for error in present_errors(boundary):
        sub = boundary[boundary.error == error]
        ns = sorted(sub.n.unique())
        for i, n in enumerate(ns):
            block = sub[sub.n == n]
            size = _pair(_get(block, "semi", "reject_nominal", 100),
                         _get(block, "conf", "reject_nominal", 100), 1)
            bias = _pair(_get(block, "semi", "delta_bias", 100),
                         _get(block, "conf", "delta_bias", 100), 1)
            cov = _pair(_get(block, "semi", "delta_coverage", 100),
                        _get(block, "conf", "delta_coverage", 100), 1)
            length = _pair(_get(block, "semi", "delta_ci_length_mean", 100),
                           _get(block, "conf", "delta_ci_length_mean", 100), 1)
            sd_t = _pair(_get(block, "semi", "T_sd"),
                         _get(block, "conf", "T_sd"), 3)
            prefix = (rf"\multirow{{{len(ns)}}}{{*}}{{{ERROR_LABEL[error]}}} & {n}"
                      if i == 0 else f" & {n}")
            lines.append(" & ".join([prefix, size, bias, cov, length, sd_t]) + r" \\")
            rows_csv.append({
                "error": error, "n": int(n),
                "size_semi": _get(block, "semi", "reject_nominal"),
                "size_conf": _get(block, "conf", "reject_nominal"),
                "bias_semi": _get(block, "semi", "delta_bias"),
                "bias_conf": _get(block, "conf", "delta_bias"),
                "coverage_semi": _get(block, "semi", "delta_coverage"),
                "coverage_conf": _get(block, "conf", "delta_coverage"),
                "ci_length_semi": _get(block, "semi", "delta_ci_length_mean"),
                "ci_length_conf": _get(block, "conf", "delta_ci_length_mean"),
                "T_sd_semi": _get(block, "semi", "T_sd"),
                "T_sd_conf": _get(block, "conf", "T_sd"),
            })
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    folder.mkdir(parents=True, exist_ok=True)
    tex = folder / "table2_delta_calib.tex"
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows_csv).to_csv(folder / "table2_delta_calib.csv", index=False)
    return tex


def figure1_rmse(est: pd.DataFrame, folder: Path) -> Path | None:
    est = est[(est.study_type == "estimation") & (est.target.isin(["x", "w"]))].copy()
    if est.empty:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    targets = ("x", "w")
    target_title = {"x": r"$\zeta_x$ RMSE", "w": r"$\zeta_w$ RMSE"}
    method_label = {"semi": "Semiparametric", "conf": "Conformal"}
    errors = present_errors(est)
    if not errors:
        return None
    fig, axes = plt.subplots(len(errors), len(targets), figsize=(9.6, 3.6 * len(errors)),
                             sharex=True, sharey=False, squeeze=False)
    for i, error in enumerate(errors):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            block = est[(est.error == error) & (est.target == target)]
            if block.empty:
                ax.set_visible(False)
                continue
            for method, ls, marker in (("semi", "-", "o"), ("conf", "--", "s")):
                group = block[block.method == method].sort_values("n")
                if group.empty:
                    continue
                ax.plot(group.n, group.rmse, linestyle=ls, marker=marker,
                        label=method_label[method])
            ax.set_xscale("log", base=10)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
            if i == 0:
                ax.set_title(target_title[target])
            if j == 0:
                ax.set_ylabel(f"{ERROR_LABEL[error]} errors\nRMSE (log scale)")
            else:
                ax.set_ylabel("RMSE (log scale)")
            if i == len(errors) - 1:
                ax.set_xlabel("Sample size $n$ (log scale)")
    fig.tight_layout()
    path = folder / "zeta_rmse_by_error.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure2_power(infer: pd.DataFrame, folder: Path) -> Path | None:
    if infer.empty:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    ns_plot = [n for n in (1000, 4000, 16000) if n in set(infer.n.unique())]
    errors = [e for e in ("right", "left") if e in set(infer.error.unique())]
    if not ns_plot or not errors:
        return None
    fig, axes = plt.subplots(len(errors), len(ns_plot), figsize=(11.6, 3.2 * len(errors)),
                             sharex=True, sharey=True, squeeze=False)
    for i, error in enumerate(errors):
        for j, n in enumerate(ns_plot):
            ax = axes[i, j]
            block = infer[(infer.error == error) & (infer.n == n)]
            for method, marker in (("semi", "o"), ("conf", "s")):
                group = block[block.method == method].sort_values("gap")
                if group.empty:
                    continue
                yerr = 1.96 * group["reject_nominal_mcse"] if "reject_nominal_mcse" in group else None
                ax.errorbar(group.gap, group.reject_nominal, yerr=yerr,
                            marker=marker, capsize=2, label=method)
            level = float(block.test_level.iloc[0]) if len(block) else 0.05
            ax.axhline(level, linestyle=":", color="0.35")
            ax.axvline(0.0, linestyle=":", color="0.35")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(f"n={n}")
            if j == 0:
                ax.set_ylabel(f"{ERROR_LABEL[error]}\nRejection rate")
            if i == len(errors) - 1:
                ax.set_xlabel(r"$\Delta(\beta)-\Delta_0$")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = folder / "delta_power_nominal.png"
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    summaries = output / "summaries"
    figures = output / "figures"
    tables = output / "paper_tables"
    written = []
    def _read(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    est = _read(summaries / "estimation_summary.csv")
    infer = _read(summaries / "inference_summary.csv")
    if len(est):
        for path in (write_table1(est, tables), figure1_rmse(est, figures)):
            if path is not None:
                written.append(str(path))
    if len(infer):
        for path in (write_table2(infer, tables), figure2_power(infer, figures)):
            if path is not None:
                written.append(str(path))
    print("Wrote " + (", ".join(written) if written else "nothing"))


if __name__ == "__main__":
    main()
