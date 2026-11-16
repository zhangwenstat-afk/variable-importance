"""Monte Carlo summaries, independent size-adjusted rejection, and audit counts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

GROUP = ["study", "study_type", "role", "error", "n", "layer", "kind", "method",
         "delta0", "delta_target", "gap", "gap_type", "gap_param", "calibration_key"]


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def read_raw(output: str | Path) -> pd.DataFrame:
    files = sorted((Path(output)/"raw").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no raw batch CSV files found")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True, sort=False)
    if df.duplicated(["cell_id", "rep", "layer", "method"]).any():
        raise ValueError("duplicate replication rows detected; do not concatenate overlapping runs")
    for column in ["zeta_x_est", "zeta_w_est", "delta_est", "se_x", "se_w", "se_delta"]:
        if column not in df:
            df[column] = np.nan
    df["estimator_valid"] = truthy(df["estimator_valid"])
    df["inference_valid"] = truthy(df["inference_valid"])
    df["T_delta"] = (df["delta_est"]-df["delta0"])/df["se_delta"]
    return df


def rate_summary(indicators: np.ndarray, prefix: str, total: int | None = None) -> dict[str, float]:
    valid = np.asarray(indicators, dtype=float)
    valid = valid[np.isfinite(valid)]
    N = len(valid)
    if N == 0:
        out = {prefix: math.nan, prefix+"_mcse": math.nan,
               prefix+"_wilson_lo": math.nan, prefix+"_wilson_hi": math.nan}
        if total is not None and total > 0:
            out[prefix+"_all_lower"], out[prefix+"_all_upper"] = 0.0, 1.0
        return out
    successes = float(valid.sum())
    p = successes/N
    z = norm.ppf(0.975)
    denom = 1+z*z/N
    center = (p+z*z/(2*N))/denom
    half = z*math.sqrt(p*(1-p)/N+z*z/(4*N*N))/denom
    out = {prefix: p, prefix+"_mcse": math.sqrt(p*(1-p)/N),
           prefix+"_wilson_lo": center-half, prefix+"_wilson_hi": center+half}
    if total is not None:
        out[prefix+"_all_lower"] = successes/total
        out[prefix+"_all_upper"] = (successes+total-N)/total
    return out


def _rms(values: pd.Series) -> float:
    a = pd.to_numeric(values, errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a*a))) if len(a) else math.nan


def estimation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, block in df[df["role"] == "evaluation"].groupby(GROUP, dropna=False, sort=True):
        meta = dict(zip(GROUP, keys))
        for target in ("x", "w", "delta"):
            name = "delta_est" if target == "delta" else f"zeta_{target}_est"
            true_name = "delta_true" if target == "delta" else f"zeta_{target}"
            valid = block["estimator_valid"] & np.isfinite(block[name])
            estimates = block.loc[valid, name]
            errors = estimates-block.loc[valid, true_name]
            R = len(errors)
            se = block.loc[valid, f"se_{target}"]
            infer_valid = np.isfinite(se) & (se > 0)
            zcrit = norm.ppf((1+float(block["ci_level"].iloc[0]))/2)
            cover = (errors[infer_valid].abs() <= zcrit*se[infer_valid]).to_numpy(float)
            sd = float(estimates.std(ddof=1)) if R > 1 else math.nan
            rmse = _rms(errors)
            mean_se = float(se[infer_valid].mean()) if infer_valid.any() else math.nan
            out = {**meta, "target": target, "truth": float(block[true_name].iloc[0]),
                   "R_requested": len(block), "R_est_valid": R,
                   "R_ci_valid": int(infer_valid.sum()), "est_failure_rate": 1-R/len(block),
                   "bias": float(errors.mean()) if R else math.nan,
                   "bias_mcse": sd/math.sqrt(R) if R else math.nan,
                   "sd": sd, "rmse": rmse, "mean_se": mean_se,
                   "se_sd_ratio": mean_se/sd if sd > 0 else math.nan,
                   "sqrt_n_rmse": math.sqrt(meta["n"])*rmse,
                   "sqrt_n_bias": math.sqrt(meta["n"])*(float(errors.mean()) if R else math.nan),
                   "ci_length_mean": 2*zcrit*mean_se}
            if R > 1 and rmse > 0:
                out["rmse_mcse"] = float((errors**2).std(ddof=1))/(2*rmse*math.sqrt(R))
            else:
                out["rmse_mcse"] = math.nan
            out.update(rate_summary(cover, "coverage", len(block)))
            standardized = errors[infer_valid]/se[infer_valid]
            out["studentized_mean"] = float(standardized.mean())
            out["studentized_sd"] = float(standardized.std(ddof=1))
            for column in (f"sqrt_n_remainder_{target}", f"bias_pred_{target}",
                           f"noise_removed_{target}",
                           f"B_{target}_bar" if target != "delta" else None):
                if column and column in block:
                    values = block.loc[valid, column]
                    out[column+"_mean"] = float(values.mean())
                    out[column+"_rms"] = _rms(values)
            if target != "delta":
                for column in (f"{target}_B_abs", f"{target}_B_cancel",
                               f"{target}_mean_D", f"{target}_mean_abs_D",
                               f"{target}_B_mc", f"{target}_B_abs_mc",
                               f"{target}_B_cancel_mc",
                               f"{target}_delta_e", f"{target}_neg_half_D_delta2",
                               f"{target}_so_score", f"{target}_mean_delta2"):
                    if column in block:
                        values = pd.to_numeric(block.loc[valid, column], errors="coerce")
                        out[column+"_mean"] = float(values.mean())
            for column in (f"so_bias_pred_{target}",
                           "so_unexplained_delta" if target == "delta" else None):
                if column and column in block:
                    values = pd.to_numeric(block.loc[valid, column], errors="coerce")
                    out[column+"_mean"] = float(values.mean())
                    out[column+"_rms"] = _rms(values)
            rows.append(out)
    return pd.DataFrame(rows)


def calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    calibration = df[df["role"] == "calibration"]
    for keys, block in calibration.groupby(["calibration_key", "layer", "method"], sort=True):
        valid = block["inference_valid"] & np.isfinite(block["T_delta"])
        t = block.loc[valid, "T_delta"].to_numpy(float)
        level = float(block["test_level"].iloc[0])
        critical = float(np.quantile(t, 1-level, method="inverted_cdf")) if len(t) else math.nan
        boundary_oracle_variance = (float((block.loc[valid, "se_delta_oracle_if"]**2
                                                *block.loc[valid, "n"]).mean())
                                    if "se_delta_oracle_if" in block else math.nan)
        rows.append({"calibration_key": keys[0], "layer": keys[1], "method": keys[2],
                     "error": block["error"].iloc[0], "n": block["n"].iloc[0],
                     "delta0": block["delta0"].iloc[0], "R_calibration": len(block),
                     "R_calibration_valid": len(t), "critical_size_adjusted": critical,
                     "oracle_variance_boundary": boundary_oracle_variance,
                     "calibration_failure_rate": 1-len(t)/len(block)})
    return pd.DataFrame(rows)


def inference_summary(df: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    if len(calibration):
        lookup = {(r["calibration_key"], r["layer"], r["method"]): r
                  for r in calibration.to_dict("records")}
    rows = []
    evaluation = df[(df["role"] == "evaluation") & (df["study_type"] != "estimation")]
    for keys, block in evaluation.groupby(GROUP, dropna=False, sort=True):
        meta = dict(zip(GROUP, keys))
        valid = block["inference_valid"] & np.isfinite(block["T_delta"])
        t = block.loc[valid, "T_delta"].to_numpy(float)
        level = float(block["test_level"].iloc[0])
        out = {**meta, "R_requested": len(block), "R_inference_valid": len(t),
               "inference_failure_rate": 1-len(t)/len(block),
               "T_mean": float(np.mean(t)) if len(t) else math.nan,
               "T_sd": float(np.std(t, ddof=1)) if len(t) > 1 else math.nan,
               "test_level": level,
               "regime": ("boundary_size" if abs(meta["gap"]) < 1e-10
                          else "null_interior" if meta["gap"] < 0 else "power")}
        out.update(rate_summary((t > norm.ppf(1-level)).astype(float), "reject_nominal", len(block)))
        cal = lookup.get((meta["calibration_key"], meta["layer"], meta["method"]))
        if cal and np.isfinite(cal["critical_size_adjusted"]):
            out["critical_size_adjusted"] = cal["critical_size_adjusted"]
            out["R_calibration_valid"] = cal["R_calibration_valid"]
            out.update(rate_summary((t > cal["critical_size_adjusted"]).astype(float),
                                    "reject_size_adjusted", len(block)))
            v0 = cal["oracle_variance_boundary"]
            out["oracle_normal_benchmark"] = (float(norm.sf(norm.ppf(1-level)
                        -math.sqrt(meta["n"])*meta["gap"]/math.sqrt(v0))) if v0 > 0 else math.nan)
        ci_critical = norm.ppf((1+float(block["ci_level"].iloc[0]))/2)
        error = block.loc[valid, "delta_est"]-block.loc[valid, "delta_true"]
        se = block.loc[valid, "se_delta"]
        coverage = (error.abs() <= ci_critical*se).to_numpy(float)
        out.update(rate_summary(coverage, "delta_coverage", len(block)))
        out["delta_ci_length_mean"] = float((2*ci_critical*se).mean()) if len(t) else math.nan
        out["delta_bias"] = float(error.mean()) if len(t) else math.nan
        out["delta_bias_mcse"] = (float(error.std(ddof=1))/math.sqrt(len(error))
                                  if len(error) > 1 else math.nan)
        out["mean_se"] = float(se.mean()) if len(t) else math.nan
        for column in ("so_bias_pred_delta", "so_unexplained_delta",
                       "x_delta_e", "w_delta_e",
                       "x_neg_half_D_delta2", "w_neg_half_D_delta2",
                       "so_bias_pred_x", "so_bias_pred_w"):
            if column in block:
                values = pd.to_numeric(block.loc[valid, column], errors="coerce")
                finite = values[np.isfinite(values)]
                out[column+"_mean"] = float(finite.mean()) if len(finite) else math.nan
        rows.append(out)
    return pd.DataFrame(rows)


def cancellation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Fold-averaged B_v, A_v, and weighted-error cancellation on diagnostic reps."""
    if "diagnostics_computed" not in df:
        return pd.DataFrame()
    d = df[(df["role"] == "evaluation") & (df["method"] == "semi")
           & truthy(df["diagnostics_computed"])]
    if d.empty:
        return pd.DataFrame()
    keys = [col for col in GROUP if col != "method"]
    if "mean_fn" in d and "mean_fn" not in keys:
        keys = ["mean_fn"] + keys
    rows = []
    for group_keys, block in d.groupby(keys, dropna=False, sort=True):
        meta = dict(zip(keys, group_keys if isinstance(group_keys, tuple) else (group_keys,)))
        out = {**meta, "R_diag": len(block)}
        for target in ("x", "w"):
            out[f"mean_A_{target}"] = float(block[f"mean_A_{target}"].iloc[0])
            for column, name in ((f"B_{target}_bar", f"B_{target}"),
                                 (f"{target}_B_abs", f"B_abs_{target}"),
                                 (f"{target}_B_cancel", f"B_cancel_{target}"),
                                 (f"{target}_mean_D", f"mean_D_{target}"),
                                 (f"{target}_mean_abs_D", f"mean_abs_D_{target}"),
                                 (f"{target}_B_mc", f"B_mc_{target}"),
                                 (f"{target}_B_abs_mc", f"B_abs_mc_{target}"),
                                 (f"{target}_B_cancel_mc", f"B_cancel_mc_{target}"),
                                 (f"{target}_mean_D_mc", f"mean_D_mc_{target}")):
                if column in block:
                    values = pd.to_numeric(block[column], errors="coerce")
                    out[name+"_mean"] = float(values.mean())
                    out[name+"_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else math.nan
        rows.append(out)
    return pd.DataFrame(rows)


def second_order_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Score-level second-order remainder vs observed Delta bias, on diagnostic reps."""
    required = ["delta_error", "T_delta", "se_delta", "so_bias_pred_delta"]
    if any(col not in df for col in required):
        return pd.DataFrame()
    d = df[(df["role"] == "evaluation") & (df["method"] == "semi")
           & df["estimator_valid"]]
    if "diagnostics_computed" in d:
        d = d[truthy(d["diagnostics_computed"])]
    if d.empty:
        return pd.DataFrame()
    keys = [col for col in GROUP if col != "method"]
    rows = []
    for group_keys, block in d.groupby(keys, dropna=False, sort=True):
        meta = dict(zip(keys, group_keys if isinstance(group_keys, tuple) else (group_keys,)))
        infer = block["inference_valid"] & np.isfinite(block["T_delta"])
        t = block.loc[infer, "T_delta"].to_numpy(float)
        se = pd.to_numeric(block.loc[infer, "se_delta"], errors="coerce")
        err = pd.to_numeric(block["delta_error"], errors="coerce")
        so = pd.to_numeric(block["so_bias_pred_delta"], errors="coerce")
        if "so_unexplained_delta" in block:
            unexplained = pd.to_numeric(block["so_unexplained_delta"], errors="coerce")
        else:
            unexplained = err-so
        level = float(block["test_level"].iloc[0]) if "test_level" in block else 0.05
        out = {
            **meta, "R_diag": len(block),
            "delta_bias": float(err.mean()),
            "delta_bias_mcse": float(err.std(ddof=1))/math.sqrt(len(err)) if len(err) > 1 else math.nan,
            "mean_se": float(se.mean()) if infer.any() else math.nan,
            "T_mean": float(np.mean(t)) if len(t) else math.nan,
            "T_sd": float(np.std(t, ddof=1)) if len(t) > 1 else math.nan,
        }
        out.update(rate_summary((t > norm.ppf(1-level)).astype(float), "reject_nominal",
                                len(block)))
        for column, name in (("x_delta_e", "mean_delta_e_x"),
                             ("w_delta_e", "mean_delta_e_w"),
                             ("x_neg_half_D_delta2", "neg_half_D_delta2_x"),
                             ("w_neg_half_D_delta2", "neg_half_D_delta2_w"),
                             ("so_bias_pred_x", "so_bias_pred_x"),
                             ("so_bias_pred_w", "so_bias_pred_w")):
            if column in block:
                values = pd.to_numeric(block[column], errors="coerce")
                out[name] = float(values.mean())
        out["so_bias_pred_delta"] = float(so.mean()) if so.notna().any() else math.nan
        out["so_unexplained_delta"] = (float(unexplained.mean())
                                       if unexplained.notna().any() else math.nan)
        n = float(meta["n"])
        out["sqrt_n_delta_bias"] = math.sqrt(n)*out["delta_bias"]
        out["sqrt_n_so_bias_pred"] = math.sqrt(n)*out["so_bias_pred_delta"]
        out["sqrt_n_unexplained"] = math.sqrt(n)*out["so_unexplained_delta"]
        rows.append(out)
    return pd.DataFrame(rows)


def diagnostic_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["role"] == "evaluation") & (df["method"] == "semi")]
    columns = [col for col in df if col.startswith(("x_", "w_")) and col not in {"x", "w"}]
    numeric = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    keys = [col for col in GROUP if col != "method"]
    if not numeric:
        return pd.DataFrame()
    grouped = d.groupby(keys, dropna=False)[numeric].agg(["mean", "max", "count"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index()


def summarize(output: str | Path) -> dict[str, Any]:
    output = Path(output)
    df = read_raw(output)
    folder = output/"summaries"
    folder.mkdir(exist_ok=True)
    calibration = calibration_summary(df)
    tables = {"estimation_summary": estimation_summary(df),
              "inference_summary": inference_summary(df, calibration),
              "calibration_critical_values": calibration,
              "nuisance_diagnostics": diagnostic_summary(df),
              "cancellation_summary": cancellation_summary(df),
              "second_order_summary": second_order_summary(df)}
    for name, table in tables.items():
        table.to_csv(folder/f"{name}.csv", index=False)
    failed = df[df["status"] != "ok"]
    failed.to_csv(folder/"failed_replications.csv", index=False)
    status = df.groupby(["layer", "method", "status"], dropna=False).size().reset_index(name="rows")
    status.to_csv(folder/"status_counts.csv", index=False)
    audit = {"raw_rows": len(df), "failed_estimation_rows": len(failed),
             "invalid_inference_rows": int((~df["inference_valid"]).sum()),
             "min_evaluation_reps_per_cell_layer_method": int(df[df.role == "evaluation"]
                         .groupby(["cell_id", "layer", "method"]).size().min()),
             "independent_calibration_rows": int((df.role == "calibration").sum()),
             "size_adjustment_mcse_note": "MCSE conditions on independently simulated critical values; calibration uncertainty is additional.",
             "failure_note": "Reported rates use valid replicates; *_all_lower/upper bound rates under arbitrary outcomes for failed replicates."}
    (folder/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return audit
