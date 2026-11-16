#!/usr/bin/env python3
"""Rebuild spacing standard errors at a new c_n factor.

Point estimates, scores and f_S stay the finished factor-0.5 inference rows.
The nuisance is refit only to recover weighted |residual| quantiles at
p ± c_n. Zeta and Delta are copied, not replaced.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()


import argparse
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from scipy.stats import norm

from conditional_mc.estimators import (
    K2_INT,
    bandwidths,
    cross_fit_nuisance,
    kernel_h,
    method_h,
    spacing_c,
    weighted_quantile,
)
from conditional_mc.simulation import make_cells, validate_config
from marginal_mc.dgp import ErrorLaw, generate, stable_rng

PROJECT = (
    Path(os.environ["MC_PROJECT"]).expanduser().resolve()
    if os.environ.get("MC_PROJECT")
    else Path(__file__).resolve().parent
)
ALPHA = 0.1
P_LEVEL = 1.0 - ALPHA
S0 = 0.5
BASE_FACTOR = 0.5
NEW_FACTOR = 0.75
SOURCES = (
    PROJECT / "results/conditional_infer_y20_w_hfhalf_cn05_midroot_seed20261006_n1k4k_r1000",
    PROJECT / "results/conditional_infer_y20_w_hfhalf_cn05_midroot_seed20261006_n8000_r1000",
    PROJECT / "results/conditional_infer_y20_w_hfhalf_cn05_midroot_seed20261006_n16000_r1000",
)
CONFIG_PATH = PROJECT / "configs/conditional_infer_y20_w_hfhalf_cn05_midroot_seed20261006_n1k4k_r1000.json"
USECOLS = [
    "error", "n", "s0", "rep", "method", "beta", "study_type", "gap",
    "delta_est", "delta_true", "delta0", "zeta_x_est", "zeta_w_est",
    "zeta_x_true", "zeta_w_true", "bias_x", "bias_w", "bias_delta",
    "estimator_valid", "f_s", "nh", "h", "c_n", "score_xx", "score_ww",
    "score_xw", "rho", "ginv_x", "ginv_w", "se_x", "se_w", "se_delta",
]


def _se_from_g(method: str, f_s: float, nh: float, gx: float, gw: float,
               score_xx: float, score_ww: float, score_xw: float, rho: float
               ) -> tuple[float, float, float]:
    if not all(math.isfinite(v) and v > 0 for v in (f_s, nh, gx, gw)):
        return math.nan, math.nan, math.nan
    if method == "semi":
        if not all(math.isfinite(v) for v in (score_xx, score_ww, score_xw)):
            return math.nan, math.nan, math.nan
        sig_xx = score_xx / (f_s * gx * gx)
        sig_ww = score_ww / (f_s * gw * gw)
        sig_xw = score_xw / (f_s * gx * gw)
        v_x = (K2_INT / f_s) * sig_xx
        v_w = (K2_INT / f_s) * sig_ww
        v_delta = (K2_INT / f_s) * (sig_xx + sig_ww - 2.0 * sig_xw)
    else:
        if not math.isfinite(rho):
            return math.nan, math.nan, math.nan
        v_x = (K2_INT / f_s) * (ALPHA * P_LEVEL / (gx * gx))
        v_w = (K2_INT / f_s) * (ALPHA * P_LEVEL / (gw * gw))
        v_delta = (K2_INT / f_s) * (
            ALPHA * P_LEVEL / (gx * gx) + ALPHA * P_LEVEL / (gw * gw)
            - 2.0 * rho / (gx * gw)
        )

    def _se(v: float) -> float:
        return math.sqrt(v / nh) if math.isfinite(v) and v > 0.0 else math.nan

    return _se(v_x), _se(v_w), _se(v_delta)


def _inv_g(residual: np.ndarray, weights: np.ndarray, c_n: float) -> float:
    if not (math.isfinite(c_n) and c_n > 0):
        return math.nan
    absolute = np.abs(residual)
    hi = weighted_quantile(absolute, weights, min(0.999, P_LEVEL + c_n))
    lo = weighted_quantile(absolute, weights, max(1e-4, P_LEVEL - c_n))
    if not (math.isfinite(hi) and math.isfinite(lo)):
        return math.nan
    return (hi - lo) / (2.0 * c_n)


def _g(inv: float) -> float:
    return 1.0 / inv if math.isfinite(inv) and inv > 0 else math.nan


def _rel(new: float, old: float) -> float:
    if not (math.isfinite(new) and math.isfinite(old)):
        return math.nan
    scale = max(abs(old), 1e-12)
    return abs(new - old) / scale


def load_saved(errors: set[str], sample_sizes: set[int], cache: Path) -> pd.DataFrame:
    if cache.exists():
        out = pd.read_csv(cache)
        print(f"loaded cached ingredients {len(out)}", flush=True)
    else:
        out = _scan_saved(cache)
    out = out[out["error"].isin(errors) & out["n"].isin(sample_sizes)]
    return out


def _scan_saved(cache: Path) -> pd.DataFrame:
    frames = []
    for root in SOURCES:
        raw = root / "raw"
        for path in sorted(raw.glob("*.csv")):
            frame = pd.read_csv(path, usecols=USECOLS)
            frame = frame[np.isclose(frame["s0"].to_numpy(dtype=float), S0)]
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise SystemExit("no saved s0=0.5 rows")
    out = pd.concat(frames, ignore_index=True)
    keys = ["error", "n", "rep", "method"]
    if out.duplicated(keys).any():
        check = ["ginv_x", "ginv_w", "delta_est", "se_delta", "score_xx", "rho"]
        worst = 0.0
        for _, block in out.groupby(keys, sort=False):
            if len(block) == 1:
                continue
            base = block.iloc[0]
            for _, row in block.iloc[1:].iterrows():
                for col in check:
                    a, b = float(base[col]), float(row[col])
                    if math.isfinite(a) or math.isfinite(b):
                        worst = max(worst, _rel(a, b) if math.isfinite(a) and math.isfinite(b) else math.inf)
        out = out.drop_duplicates(keys, keep="first")
        print(f"dropped duplicate saved rows; max relative gap {worst:.3e}", flush=True)
        if worst > 1e-8:
            raise SystemExit("duplicate saved rows disagree")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"cached ingredients {len(out)}", flush=True)
    return out


def production_config(sample_sizes: list[int]) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["locations"] = [S0]
    config["sample_sizes"] = sample_sizes
    config["c_n_factor"] = BASE_FACTOR
    config.pop("name", None)
    return validate_config(config)


def one(job: dict) -> dict:
    config = job["config"]
    layer = job["layer"]
    error = job["error"]
    n = int(job["n"])
    rep = int(job["rep"])
    beta = float(job["beta"])
    law = ErrorLaw(error, config["mixture_p"], config["mixture_a"], config["error_base"])
    s0_key = f"{S0:.6g}"
    data_key = (
        config["error_base"], error, n, f"{beta:.12g}", "inference",
        s0_key, f"{0.0:.6g}", rep,
    )
    rng = stable_rng(int(config["seed"]), "cond_data", *data_key)
    data = generate(n, beta, law, rng, config["mean_fn"])
    fitted = cross_fit_nuisance(data, law, layer, config, beta, S0)
    hs = bandwidths(n, float(config["c_semi"]), float(config["c_conf"]))
    by_method = {}
    for method in ("semi", "conf"):
        h = method_h(method, hs)
        c50 = spacing_c(n, h, BASE_FACTOR)
        c75 = spacing_c(n, h, NEW_FACTOR)
        inv50 = {"x": [], "w": []}
        inv75 = {"x": [], "w": []}
        for j, ev in enumerate(data.folds):
            weights = kernel_h(data.x[ev], S0, h)
            for target in ("x", "w"):
                residual = fitted["resid"][target][j]
                inv50[target].append(_inv_g(residual, weights, c50))
                inv75[target].append(_inv_g(residual, weights, c75))
        by_method[method] = {
            "c_n_050": c50,
            "c_n_075": c75,
            "ginv_x_050": float(np.nanmean(inv50["x"])),
            "ginv_w_050": float(np.nanmean(inv50["w"])),
            "ginv_x_075": float(np.nanmean(inv75["x"])),
            "ginv_w_075": float(np.nanmean(inv75["w"])),
        }
    z_test = float(job["z_test"])
    z_ci = float(job["z_ci"])
    rows = []
    for saved in job["rows"]:
        method = str(saved["method"])
        got = by_method[method]
        gx = _g(got["ginv_x_075"])
        gw = _g(got["ginv_w_075"])
        se_x, se_w, se_delta = _se_from_g(
            method, float(saved["f_s"]), float(saved["nh"]), gx, gw,
            float(saved["score_xx"]), float(saved["score_ww"]), float(saved["score_xw"]),
            float(saved["rho"]),
        )
        se_x0, se_w0, se_d0 = _se_from_g(
            method, float(saved["f_s"]), float(saved["nh"]),
            _g(got["ginv_x_050"]), _g(got["ginv_w_050"]),
            float(saved["score_xx"]), float(saved["score_ww"]), float(saved["score_xw"]),
            float(saved["rho"]),
        )
        estimator_ok = bool(saved["estimator_valid"])
        inference_ok = (
            estimator_ok and math.isfinite(se_delta) and se_delta > 0
            and math.isfinite(se_x) and math.isfinite(se_w)
        )
        delta_est = float(saved["delta_est"])
        delta_true = float(saved["delta_true"])
        delta0 = float(saved["delta0"])
        if inference_ok:
            stat = (delta_est - delta0) / se_delta
            reject = float(stat > z_test)
            cover = float(abs(delta_est - delta_true) <= z_ci * se_delta)
            length = 2.0 * z_ci * se_delta
        else:
            stat = reject = cover = length = math.nan
        rows.append({
            "error": error,
            "n": n,
            "s0": S0,
            "rep": rep,
            "method": method,
            "study_type": "inference",
            "gap": 0.0,
            "beta": beta,
            "estimator_valid": estimator_ok,
            "inference_valid": inference_ok,
            "delta_est": delta_est,
            "delta_true": delta_true,
            "delta0": delta0,
            "zeta_x_est": float(saved["zeta_x_est"]),
            "zeta_w_est": float(saved["zeta_w_est"]),
            "bias_x": float(saved["bias_x"]) if estimator_ok else math.nan,
            "bias_w": float(saved["bias_w"]) if estimator_ok else math.nan,
            "bias_delta": float(saved["bias_delta"]) if estimator_ok else math.nan,
            "f_s": float(saved["f_s"]),
            "nh": float(saved["nh"]),
            "h": float(saved["h"]),
            "c_n": got["c_n_075"],
            "c_n_050": got["c_n_050"],
            "ginv_x_saved": float(saved["ginv_x"]),
            "ginv_w_saved": float(saved["ginv_w"]),
            "ginv_x_050": got["ginv_x_050"],
            "ginv_w_050": got["ginv_w_050"],
            "ginv_x": got["ginv_x_075"],
            "ginv_w": got["ginv_w_075"],
            "ginv_x_relerr": _rel(got["ginv_x_050"], float(saved["ginv_x"])),
            "ginv_w_relerr": _rel(got["ginv_w_050"], float(saved["ginv_w"])),
            "se_delta_saved": float(saved["se_delta"]),
            "se_delta_050": se_d0,
            "se_x_050": se_x0,
            "se_w_050": se_w0,
            "se_relerr": _rel(se_d0, float(saved["se_delta"])),
            "se_x": se_x,
            "se_w": se_w,
            "se_delta": se_delta,
            "T": stat,
            "reject_nominal": reject,
            "ci_delta_cover": cover,
            "ci_delta_length": length,
        })
    dest = Path(job["dest"])
    tmp = dest.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, dest)
    return {"rep": rep, "n": n, "error": error, "rows": len(rows)}


def write_summary(out: Path) -> None:
    raw = pd.concat(
        (pd.read_csv(path) for path in sorted((out / "raw").glob("*.csv"))),
        ignore_index=True,
    )
    raw.to_csv(out / "raw_all.csv", index=False)
    ok = raw[raw["estimator_valid"] == True].copy()
    group = ["study_type", "error", "n", "s0", "method", "gap"]
    rows = []
    for keys, block in ok.groupby(group, dropna=False):
        rec = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        inf = block[block["inference_valid"] == True]
        rec["reps"] = int(len(block))
        rec["infer_reps"] = int(len(inf))
        rec["bias_delta"] = float(block["bias_delta"].mean())
        rec["ginv_relerr_max"] = float(np.nanmax(block[["ginv_x_relerr", "ginv_w_relerr"]].to_numpy()))
        rec["se_relerr_max"] = float(np.nanmax(block["se_relerr"].to_numpy()))
        if len(inf):
            rec["size"] = float(inf["reject_nominal"].mean())
            rec["ci_cover_delta"] = float(inf["ci_delta_cover"].mean())
            rec["ci_length_delta"] = float(inf["ci_delta_length"].mean())
            rec["mean_se_delta"] = float(inf["se_delta"].mean())
            rec["sd_T"] = float(inf["T"].std(ddof=1)) if len(inf) > 1 else math.nan
            rec["mean_T"] = float(inf["T"].mean())
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values(group)
    dest = out / "summaries"
    dest.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dest / "summary.csv", index=False)
    print(f"summary rows {len(summary)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--errors", default="left,right")
    parser.add_argument("--ns", default="1000,2000,4000,8000,16000")
    parser.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    errors = set(args.errors.split(","))
    sample_sizes = [int(v) for v in args.ns.split(",")]
    out = args.out
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config = production_config(sample_sizes)
    cells = {
        (cell["error"], int(cell["n"])): cell
        for cell in make_cells(config)
        if cell["error"] in errors and int(cell["n"]) in sample_sizes
    }
    saved = load_saved(errors, set(sample_sizes), out / "ingredients_s05.csv")
    saved = saved[saved["rep"] < args.reps]
    expected = {(error, n, rep) for error, n in cells for rep in range(args.reps)}
    have = set(zip(saved["error"], saved["n"].astype(int), saved["rep"].astype(int)))
    missing = sorted(expected - have)
    if missing:
        raise SystemExit(f"saved rows missing {len(missing)} replications, first {missing[:5]}")
    z_test = float(norm.ppf(1.0 - float(config["test_level"])))
    z_ci = float(norm.ppf(0.5 + 0.5 * float(config["ci_level"])))
    layer = config["layers"][0]
    jobs = []
    skipped = 0
    for (error, n, rep), block in saved.groupby(["error", "n", "rep"], sort=True):
        dest = raw_dir / f"{error}_n{int(n)}_r{int(rep):04d}.csv"
        if dest.exists():
            skipped += 1
            continue
        cell = cells[(error, int(n))]
        jobs.append({
            "config": config,
            "layer": layer,
            "error": error,
            "n": int(n),
            "rep": int(rep),
            "beta": float(cell["beta"]),
            "z_test": z_test,
            "z_ci": z_ci,
            "dest": str(dest),
            "rows": block.to_dict("records"),
        })
    print(
        f"saved={len(saved)} jobs={len(jobs)} skipped={skipped} workers={args.jobs}",
        flush=True,
    )
    if jobs:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                fut.result()
                if i % 25 == 0 or i == len(futures):
                    print(f"done={i}/{len(futures)}", flush=True)
    done = len(list(raw_dir.glob("*.csv")))
    target = len(expected)
    print(f"files={done}/{target}", flush=True)
    if done == target:
        write_summary(out)


if __name__ == "__main__":
    main()
