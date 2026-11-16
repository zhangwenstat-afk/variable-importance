"""Fold-specific estimating equations, cross-fitted variance, and truth diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .dgp import (Data, ErrorLaw, covariate_quadrature, diagnostic_grid,
                  sample_covariates, stable_rng, true_mean)
from .nuisance import (AnalyticNuisance, LearnedPrepared, Nuisance, Prepared,
                       fit_nuisance)

Array = NDArray[np.float64]


def empirical_quantile(values: Array, p: float) -> float:
    """Left inverse of the empirical CDF, i.e. an empirical check-loss minimizer.

    This is NOT the ceil((M+1)*p) split-conformal coverage correction.
    """
    if not 0 < p < 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("invalid empirical quantile arguments")
    return float(np.quantile(values, p, method="inverted_cdf"))


@dataclass
class RootResult:
    q: float
    score: float
    success: bool
    evaluations: int
    bracket_expansions: int


def solve_semi(residual: Array, prepared: Prepared, p: float,
               q0: float) -> RootResult:
    """A local approximate root initialized by the conf quantile.

    The empirical CDF is linearly interpolated ONLY during root search.
    This interpolation differs from the step CDF by at most 1/M, so the
    returned solution is checked against the ORIGINAL unsmoothed score.
    No fixed-bandwidth smoothing of the indicator, no one-step substitution,
    and no oracle quantile enters the solver.
    """
    M = len(residual)
    absolute = np.abs(residual)
    values = np.r_[0.0, np.sort(absolute)]
    levels = np.arange(M+1, dtype=float)/M
    evaluations = 0
    cache: dict[float, float] = {}

    def correction(q: float) -> float:
        nonlocal evaluations
        if q not in cache:
            cache[q] = prepared.correction(q, residual)
            evaluations += 1
        return cache[q]

    def score_original(q: float) -> float:
        return p-float(np.mean(absolute <= q))-correction(q)

    def score_continuous(q: float) -> float:
        return p-float(np.interp(q, values, levels, left=0.0, right=1.0))-correction(q)

    exact_at_q0 = score_original(q0)
    # Includes exact oracle symmetry; avoids numerical changes to its conf root.
    if abs(exact_at_q0) <= (1.0+1e-8)/M:
        return RootResult(q0, exact_at_q0, True, evaluations, 0)
    center = score_continuous(q0)
    radius = max(0.05, 0.10*q0)
    bracket = None
    expansion = 0
    for expansion in range(12):
        left, right = max(0.0, q0-radius), q0+radius
        sl, sr = score_continuous(left), score_continuous(right)
        candidates = []
        if sl >= 0 >= center:
            candidates.append((left, q0))
        if center >= 0 >= sr:
            candidates.append((q0, right))
        if candidates:
            bracket = min(candidates, key=lambda pair: pair[1]-pair[0])
            break
        radius *= 1.8
    if bracket is None:
        return RootResult(q0, exact_at_q0, False, evaluations, expansion)
    try:
        q = float(brentq(score_continuous, *bracket, xtol=1e-10,
                         rtol=1e-11, maxiter=120))
    except (RuntimeError, ValueError, FloatingPointError):
        return RootResult(q0, exact_at_q0, False, evaluations, expansion)
    score = score_original(q)
    success = (q >= 0 and np.isfinite(q) and np.isfinite(score)
               and abs(score) <= max(1.05/M, 1e-8))
    return RootResult(q, score, bool(success), evaluations, expansion)


def spacing_inverse_density(fold_residuals: list[Array], alpha: float,
                            n: int) -> tuple[float, float]:
    p = 1-alpha
    c = 0.5*min(alpha, p)*min(1.0, (n/500.0)**(-0.2))
    spacings = [(empirical_quantile(np.abs(r), p+c)
                 - empirical_quantile(np.abs(r), p-c))/(2*c)
                for r in fold_residuals]
    inv_g = float(np.mean(spacings))
    return inv_g, c


def b_functionals(D: Array, error: Array, weights: Array) -> dict[str, float]:
    """B = E[D e], B_abs = E[|D e|], and the cancellation ratio |B|/B_abs."""
    D = np.asarray(D, dtype=float)
    error = np.asarray(error, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if D.shape != error.shape or D.shape != weights.shape:
        raise ValueError("D, error, and weights must share a shape")
    weighted = D*error
    B = float(weights @ weighted)
    B_abs = float(weights @ np.abs(weighted))
    mean_D = float(weights @ D)
    mean_abs_D = float(weights @ np.abs(D))
    cancel = abs(B)/B_abs if B_abs > 0 else math.nan
    return {"B": B, "B_abs": B_abs, "B_cancel": cancel,
            "mean_D": mean_D, "mean_abs_D": mean_abs_D}


def _weight_stats_checkpoint(fitted: Nuisance) -> tuple[list | None, int]:
    stats = getattr(fitted, "weight_stats", None)
    if isinstance(stats, list):
        return stats, len(stats)
    return None, 0


def nuisance_diagnostic(fitted: Nuisance, target: str, beta: float,
                        law: ErrorLaw, q: float, mean_A: float,
                        settings: dict[str, Any],
                        mean_fn: str = "paper",
                        diag_keys: tuple = ()) -> dict[str, float]:
    stats, n_before = _weight_stats_checkpoint(fitted)
    beta_residual = beta if target == "x" else 0.0
    xq, zq, weights = covariate_quadrature(target, int(settings.get("b_nodes", 24)))
    prep_q = fitted.prepare(xq, zq)
    error = prep_q.mean-true_mean(xq, zq, target, beta, mean_fn)
    A_true = law.A(q, xq, beta_residual)
    e_A = prep_q.A(q)-A_true
    D_slope = law.D(q, xq, beta_residual)
    quad = b_functionals(A_true, error, weights)
    if isinstance(fitted, AnalyticNuisance):
        # Constant mean shift integrates exactly against E[A].
        quad["B"] = fitted.mean_shift*mean_A
        if quad["B_abs"] > 0:
            quad["B_cancel"] = abs(quad["B"])/quad["B_abs"]
    delta_e = float(weights @ (error*e_A))
    mean_D_delta2 = float(weights @ (D_slope*error*error))
    quad.update({
        "delta_e": delta_e,
        "mean_D_delta2": mean_D_delta2,
        "neg_half_D_delta2": -0.5*mean_D_delta2,
        "so_score": delta_e-0.5*mean_D_delta2,
        "mean_delta2": float(weights @ (error*error)),
    })
    out = {**quad, "m_sup_grid": math.nan, "m_rmse_grid": math.nan,
           "f_sup_grid": math.nan, "f_q_plus_sup_grid": math.nan,
           "f_q_minus_sup_grid": math.nan, "f_q_sup_grid": math.nan,
           "a_sup_grid": math.nan, "a_rmse_grid": math.nan}
    xg, zg = diagnostic_grid(target, int(settings.get("grid_side", 15)))
    prep = fitted.prepare(xg, zg)
    m_error = prep.mean-true_mean(xg, zg, target, beta, mean_fn)
    out["m_sup_grid"] = float(np.max(np.abs(m_error)))
    out["m_rmse_grid"] = float(np.sqrt(np.mean(m_error*m_error)))
    density_points = int(settings.get("density_points", 21))
    if density_points > 0:
        radius = float(settings.get("density_radius", 0.35))
        half_width = max(6.0, q+radius)
        ts = np.unique(np.r_[np.linspace(-half_width, half_width, density_points),
                             q+np.linspace(-radius, radius, 5),
                             -q+np.linspace(-radius, radius, 5)])
        f_error = 0.0
        for t in ts:
            oracle = law.conditional_pdf(float(t), xg, beta_residual)
            f_error = max(f_error, float(np.max(np.abs(prep.pdf(float(t))-oracle))))
        oracle_plus = law.conditional_pdf(q, xg, beta_residual)
        oracle_minus = law.conditional_pdf(-q, xg, beta_residual)
        estimated_plus = prep.pdf(q)
        estimated_minus = prep.pdf(-q)
        estimated_a = prep.A(q)
        oracle_a = oracle_plus-oracle_minus
        plus_error = estimated_plus-oracle_plus
        minus_error = estimated_minus-oracle_minus
        a_error = estimated_a-oracle_a
        out.update({"f_sup_grid": f_error,
                    "f_q_plus_sup_grid": float(np.max(np.abs(plus_error))),
                    "f_q_minus_sup_grid": float(np.max(np.abs(minus_error))),
                    "f_q_sup_grid": float(max(np.max(np.abs(plus_error)),
                                              np.max(np.abs(minus_error)))),
                    "a_sup_grid": float(np.max(np.abs(a_error))),
                    "a_rmse_grid": float(np.sqrt(np.mean(a_error*a_error)))})
    check_nodes = int(settings.get("b_check_nodes", 0))
    if check_nodes:
        xq, zq, weights = covariate_quadrature(target, check_nodes)
        error = fitted.predict_mean(xq, zq)-true_mean(xq, zq, target, beta, mean_fn)
        D = law.A(q, xq, beta_residual)
        out["B_quad_difference"] = b_functionals(D, error, weights)["B"]-out["B"]
    mc_size = int(settings.get("b_mc_size", 0))
    if mc_size > 0:
        seed = int(settings.get("seed", 0))
        rng = stable_rng(seed, "independent_v", target, *diag_keys)
        if target == "x":
            xm = rng.uniform(-1.0, 1.0, size=mc_size)
            zm = np.zeros(mc_size)
        else:
            xm, zm = sample_covariates(mc_size, rng)
        error = fitted.predict_mean(xm, zm)-true_mean(xm, zm, target, beta, mean_fn)
        D = law.A(q, xm, beta_residual)
        mc = b_functionals(D, error, np.full(mc_size, 1.0/mc_size))
        out.update({f"{key}_mc": value for key, value in mc.items()})
    if stats is not None:
        del stats[n_before:]
    return out


@dataclass
class TargetFit:
    q: dict[str, float]
    fold_q: dict[str, list[float]]
    scores: dict[str, Array]
    inverse_g: float
    root_success: bool
    auxiliary: dict[str, float]
    B: float


def fit_target(data: Data, target: str, beta: float, law: ErrorLaw,
               layer: dict[str, Any], config: dict[str, Any],
               truth: dict[str, float], signs: tuple[float, float],
               do_diagnostics: bool, diag_keys: tuple = ()) -> TargetFit:
    n = len(data.y)
    alpha, p = float(config["alpha"]), 1-float(config["alpha"])
    mean_fn = config.get("mean_fn", "paper")
    fold_q: dict[str, list[float]] = {"semi": [], "conf": []}
    scores = {"semi": np.zeros(n), "conf": np.zeros(n)}
    residuals, roots, Bs = [], [], []
    nuisance_stats: list[dict[str, float]] = []
    grid_stats: list[dict[str, float]] = []
    negative_pdf, pdf_evaluations = 0, 0
    for j, ev in enumerate(data.folds):
        tr = data.folds[1-j]
        fitted = fit_nuisance(target, beta, law, data.x[tr], data.z[tr], data.y[tr],
                              {**layer, "sample_size": n}, signs[1-j], mean_fn)
        prep = fitted.prepare(data.x[ev], data.z[ev])
        r = data.y[ev]-prep.mean
        residuals.append(r)
        q_conf = empirical_quantile(np.abs(r), p)
        root = solve_semi(r, prep, p, q_conf)
        roots.append(root)
        fold_q["conf"].append(q_conf)
        fold_q["semi"].append(root.q)
        scores["conf"][ev] = p-(np.abs(r) <= q_conf).astype(float)
        scores["semi"][ev] = (p-(np.abs(r) <= root.q).astype(float)
                                 -r*prep.A(root.q))
        if isinstance(prep, LearnedPrepared):
            negative_pdf += prep.negative_count
            pdf_evaluations += prep.total_count
        if do_diagnostics:
            diag_settings = {**config["diagnostics"], "seed": config["seed"]}
            diag = nuisance_diagnostic(fitted, target, beta, law, truth[f"zeta_{target}"],
                                       truth[f"mean_A_{target}"], diag_settings,
                                       mean_fn, (*diag_keys, j))
            Bs.append(diag["B"])
            grid_stats.append(diag)
        elif isinstance(fitted, AnalyticNuisance):
            # Exact analytic B is essentially free even when grid diagnostics are off.
            Bs.append(fitted.mean_shift*truth[f"mean_A_{target}"])
        nuisance_stats.append(fitted.diagnostics())
    if config.get("g_estimator", "spacing") == "oracle":
        inv_g, c = 1/truth[f"g_{target}"], math.nan
    else:
        inv_g, c = spacing_inverse_density(residuals, alpha, n)
    auxiliary: dict[str, float] = {
        "spacing_c": c,
        "root_abs_score_max": max(abs(r.score) for r in roots),
        "root_M_abs_score_max": (n/2)*max(abs(r.score) for r in roots),
        "root_evaluations": float(sum(r.evaluations for r in roots)),
        "root_bracket_expansions_max": float(max(r.bracket_expansions for r in roots)),
        "negative_density_fraction_solver": negative_pdf/max(1, pdf_evaluations),
    }
    for key in sorted(set().union(*(d.keys() for d in nuisance_stats))):
        auxiliary[key] = float(np.mean([d.get(key, math.nan) for d in nuisance_stats]))
    if grid_stats:
        auxiliary["m_sup_grid"] = max(d["m_sup_grid"] for d in grid_stats)
        auxiliary["f_sup_grid"] = max(d["f_sup_grid"] for d in grid_stats)
        auxiliary["m_rmse_grid"] = float(np.mean([d["m_rmse_grid"] for d in grid_stats]))
        auxiliary["n_quarter_m_sup_grid"] = n**0.25*auxiliary["m_sup_grid"]
        auxiliary["n_quarter_f_sup_grid"] = n**0.25*auxiliary["f_sup_grid"]
        for key in ("f_q_plus_sup_grid", "f_q_minus_sup_grid",
                    "f_q_sup_grid", "a_sup_grid", "a_rmse_grid"):
            if key in grid_stats[0]:
                auxiliary[key] = float(np.mean(
                    [d[key] for d in grid_stats]
                )) if key == "a_rmse_grid" else max(
                    d[key] for d in grid_stats
                )
                auxiliary["n_quarter_"+key] = n**0.25*auxiliary[key]
        if "B_quad_difference" in grid_stats[0]:
            auxiliary["B_quad_difference"] = max(abs(d["B_quad_difference"]) for d in grid_stats)
        for key in ("B_abs", "B_cancel", "mean_D", "mean_abs_D",
                    "B_mc", "B_abs_mc", "B_cancel_mc", "mean_D_mc", "mean_abs_D_mc",
                    "delta_e", "mean_D_delta2", "neg_half_D_delta2", "so_score",
                    "mean_delta2"):
            if key in grid_stats[0]:
                auxiliary[key] = float(np.mean([d[key] for d in grid_stats]))
    return TargetFit({m: float(np.mean(v)) for m, v in fold_q.items()}, fold_q, scores,
                     inv_g, all(r.success for r in roots), auxiliary,
                     float(np.mean(Bs)) if len(Bs) == 2 else math.nan)


def _se(variance: float, n: int) -> float:
    return math.sqrt(variance/n) if math.isfinite(variance) and variance > 0 else math.nan


def estimate_layer(data: Data, law: ErrorLaw, layer: dict[str, Any],
                  config: dict[str, Any], truth: dict[str, float],
                  signs: tuple[float, float], do_diagnostics: bool,
                  diag_keys: tuple = ()) -> list[dict[str, Any]]:
    n, beta = len(data.y), truth["beta"]
    p, alpha = 1-config["alpha"], config["alpha"]
    mean_fn = config.get("mean_fn", "paper")
    fits = {v: fit_target(data, v, beta, law, layer, config, truth, signs,
                          do_diagnostics, diag_keys) for v in ("x", "w")}
    gx, gw = fits["x"].inverse_g, fits["w"].inverse_g
    rows = []
    for method in config.get("methods", ("semi", "conf")):
        sx, sw = fits["x"].scores[method], fits["w"].scores[method]
        xi_x, xi_w = sx*gx, sw*gw
        diff = xi_x-xi_w
        v_emp_x = float(np.var(xi_x, ddof=1))
        v_emp_w = float(np.var(xi_w, ddof=1))
        v_emp_delta = float(np.var(diff, ddof=1))
        if method == "semi":
            v_ms_x, v_ms_w = float(np.mean(xi_x**2)), float(np.mean(xi_w**2))
            v_ms_delta = float(np.mean(diff**2))
        else:
            ix, iw = p-sx, p-sw
            rho = float(np.mean(ix*iw)-p*p)
            v_ms_x, v_ms_w = alpha*p*gx*gx, alpha*p*gw*gw
            v_ms_delta = v_ms_x+v_ms_w-2*rho*gx*gw
        chosen = ((v_ms_x, v_ms_w, v_ms_delta)
                  if config.get("variance", "empirical") == "manuscript"
                  else (v_emp_x, v_emp_w, v_emp_delta))
        estimator_ok = (method == "conf" or
                        (fits["x"].root_success and fits["w"].root_success))
        normalization_ok = np.isfinite([gx, gw]).all() and gx > 0 and gw > 0
        inference_ok = estimator_ok and normalization_ok and all(np.isfinite(chosen)) and chosen[2] > 0
        row: dict[str, Any] = {
            "method": method, "status": "ok" if estimator_ok else "root_failed",
            "estimator_valid": bool(estimator_ok), "inference_valid": bool(inference_ok),
            "zeta_x_est": fits["x"].q[method], "zeta_w_est": fits["w"].q[method],
            "delta_est": fits["x"].q[method]-fits["w"].q[method],
            "ginv_x": gx, "ginv_w": gw,
            "se_x": _se(chosen[0], n), "se_w": _se(chosen[1], n),
            "se_delta": _se(chosen[2], n),
            "v_x_empirical": v_emp_x, "v_w_empirical": v_emp_w,
            "v_delta_empirical": v_emp_delta,
            "v_x_manuscript": v_ms_x, "v_w_manuscript": v_ms_w,
            "v_delta_manuscript": v_ms_delta,
            "se_delta_manuscript": _se(v_ms_delta, n),
            "se_delta_empirical": _se(v_emp_delta, n),
        }
        oracle_linear = {}
        predicted_bias = {}
        oracle_influence = {}
        for target in ("x", "w"):
            fit = fits[target]
            eps = data.y-true_mean(data.x, data.z, target, beta, mean_fn)
            q, g = truth[f"zeta_{target}"], truth[f"g_{target}"]
            psi = p-(np.abs(eps) <= q).astype(float)
            phi = psi-eps*law.A(q, data.x, beta if target == "x" else 0)
            # The score derivative is -g; the estimator's expansion has +phi/g.
            oracle_influence[target] = (phi if method == "semi" else psi)/g
            oracle_linear[target] = float(np.mean(oracle_influence[target]))
            predicted_bias[target] = -fit.B/g
            row[f"B_{target}_bar"] = fit.B
            row[f"bias_pred_{target}"] = predicted_bias[target]
            row[f"linear_{target}"] = oracle_linear[target]
            error = fit.q[method]-q
            row[f"noise_removed_{target}"] = error-oracle_linear[target]
            remainder = error-oracle_linear[target]
            if method == "conf":
                remainder -= predicted_bias[target]
            row[f"sqrt_n_remainder_{target}"] = math.sqrt(n)*remainder
            so_score = fit.auxiliary.get("so_score", math.nan)
            row[f"so_bias_pred_{target}"] = (so_score/g if math.isfinite(so_score)
                                             else math.nan)
            for key, value in fit.auxiliary.items():
                row[f"{target}_{key}"] = value
        row["bias_pred_delta"] = predicted_bias["x"]-predicted_bias["w"]
        row["linear_delta"] = oracle_linear["x"]-oracle_linear["w"]
        row["delta_error"] = row["delta_est"]-truth["delta_true"]
        row["noise_removed_delta"] = row["delta_error"]-row["linear_delta"]
        row["sqrt_n_remainder_delta"] = math.sqrt(n)*(row["noise_removed_delta"]
            -(row["bias_pred_delta"] if method == "conf" else 0.0))
        so_x, so_w = row["so_bias_pred_x"], row["so_bias_pred_w"]
        row["so_bias_pred_delta"] = (so_x-so_w
                                     if math.isfinite(so_x) and math.isfinite(so_w)
                                     else math.nan)
        row["so_unexplained_delta"] = (row["noise_removed_delta"]-row["so_bias_pred_delta"]
                                       if math.isfinite(row["so_bias_pred_delta"])
                                       else math.nan)
        row["se_delta_oracle_if"] = _se(float(np.var(oracle_influence["x"]
                                                                -oracle_influence["w"], ddof=1)), n)
        rows.append(row)
    return rows
