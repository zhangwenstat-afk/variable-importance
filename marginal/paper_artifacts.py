#!/usr/bin/env python3
"""Build paper Figure 1 (sqrt-n RMSE) and Figure 2 (power), and copy them."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd

from export_paper_refresh import ERROR_LABEL, figure2_power, present_errors


def figure1_sqrt_n_rmse(est: pd.DataFrame, dest: Path) -> Path | None:
    est = est[(est.study_type == "estimation") & (est.target.isin(["x", "w"]))].copy()
    if est.empty:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    targets = ("x", "w")
    target_title = {
        "x": r"$\sqrt{n}\times$ RMSE of $\zeta_x$",
        "w": r"$\sqrt{n}\times$ RMSE of $\zeta_w$",
    }
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
            for method, ls, marker in (("semi", "-", "o"), ("conf", "--", "s")):
                group = block[block.method == method].sort_values("n")
                if group.empty:
                    continue
                ax.plot(group.n, group.sqrt_n_rmse, linestyle=ls, marker=marker,
                        label=method_label[method])
            ax.set_xscale("log", base=10)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
            if i == 0:
                ax.set_title(target_title[target])
            if j == 0:
                ax.set_ylabel(f"{ERROR_LABEL[error]} errors\n"+r"$\sqrt{n}\times$ RMSE")
            else:
                ax.set_ylabel(r"$\sqrt{n}\times$ RMSE")
            if i == len(errors) - 1:
                ax.set_xlabel(r"Sample size $n$ (log scale)")
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    if dest.suffix.lower() == ".png":
        fig.savefig(dest.with_suffix(".pdf"))
    plt.close(fig)
    return dest


def _copy(src: Path, *folders: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, folder / src.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fig2-from", nargs="+", help="inference_summary.csv paths to concatenate")
    p.add_argument("--fig2-out", type=Path)
    p.add_argument("--fig1-from", type=Path, help="estimation_summary.csv")
    p.add_argument("--fig1-out", type=Path)
    p.add_argument("--copy-to", nargs="*", type=Path, default=[])
    args = p.parse_args()
    written = []
    if args.fig2_from:
        infer = pd.concat([pd.read_csv(path) for path in args.fig2_from], ignore_index=True)
        folder = args.fig2_out.parent if args.fig2_out else Path("figures")
        path = figure2_power(infer, folder)
        if path is None:
            raise RuntimeError("figure 2 could not be built")
        if args.fig2_out and path.resolve() != args.fig2_out.resolve():
            args.fig2_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, args.fig2_out)
            path = args.fig2_out
        written.append(path)
        _copy(path, *args.copy_to)
    if args.fig1_from:
        est = pd.read_csv(args.fig1_from)
        dest = args.fig1_out or Path("figures/zeta_sqrt_n_rmse_by_error.png")
        path = figure1_sqrt_n_rmse(est, dest)
        if path is None:
            raise RuntimeError("figure 1 could not be built")
        written.append(path)
        _copy(path, *args.copy_to)
    print("Wrote", ", ".join(str(path) for path in written) if written else "nothing")


if __name__ == "__main__":
    main()
