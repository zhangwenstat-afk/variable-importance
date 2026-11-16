"""Localized conformal and semiparametric estimators at a fixed s0=X."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from marginal_mc.dgp import Data, ErrorLaw, m0
from marginal_mc.nuisance import fit_nuisance

from .oracle import residual_coverage, z_nodes

Array = NDArray[np.float64]
K2_INT = 0.6  # Epanechnikov ∫ K^2 = 3/5


def epanechnikov(u: Array) -> Array:
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    mask = np.abs(u) <= 1.0
    out[mask] = 0.75 * (1.0 - u[mask] * u[mask])
    return out


def kernel_h(x: Array, s0: float, h: float) -> Array:
    if h <= 0 or not math.isfinite(h):
        raise ValueError("bandwidth must be positive and finite")
    return epanechnikov((np.asarray(x, dtype=float) - s0) / h) / h


# Local RMSE on the n=1000/4000 pilot is smallest near these c values.
# c=2 at s0=0.5 and r=12 at every s0 sat on the edge of that grid, so those
# sides extend one step further. At s0=0.5 the r minimizer is bimodal.
MEAN_BW_BY_S0: dict[float, dict[str, list[float]]] = {
    -0.5: {
        "m_multipliers": [0.25, 0.5, 0.75],
        "z_scales": [6.0, 8.0, 12.0, 16.0],
    },
    0.0: {
        "m_multipliers": [0.5, 0.75, 1.0, 1.25],
        "z_scales": [4.0, 6.0, 8.0, 12.0, 16.0],
    },
    # m''(0.3) is about 0, so this location uses the same large-c grid as s0=0.5.
    0.3: {
        "m_multipliers": [1.0, 1.5, 2.0, 2.5, 3.0],
        "z_scales": [2.0, 4.0, 8.0, 12.0, 16.0],
    },
    0.5: {
        "m_multipliers": [1.0, 1.5, 2.0, 2.5, 3.0],
        "z_scales": [2.0, 4.0, 8.0, 12.0, 16.0],
    },
}


# m''(x)=1-sin(pi x) is 2, 1, 0 at s0=-0.5, 0, 0.5. The n=4000 pilot
# puts c near 1, 1-1.5, and 2-3 respectively. r=16 is the edge because
# the conditional mean is linear in Z.
SINE_QUAD_BW_BY_S0: dict[float, dict[str, list[float]]] = {
    -0.5: {
        "m_multipliers": [0.75, 1.0, 1.25, 1.5],
        "z_scales": [4.0, 8.0, 12.0, 16.0],
    },
    0.0: {
        "m_multipliers": [1.0, 1.5, 2.0],
        "z_scales": [4.0, 8.0, 12.0, 16.0],
    },
    0.5: {
        "m_multipliers": [1.5, 2.0, 2.5, 3.0, 4.0],
        "z_scales": [2.0, 4.0, 8.0, 12.0, 16.0],
    },
}
MEAN_BW_BY_MEAN: dict[str, dict[float, dict[str, list[float]]]] = {
    "paper": MEAN_BW_BY_S0,
    "sine_quad": SINE_QUAD_BW_BY_S0,
}


def mean_bw_grid(s0: float, mean_fn: str = "paper") -> dict[str, list[float]] | None:
    table = MEAN_BW_BY_MEAN.get(mean_fn, {})
    for key, grid in table.items():
        if abs(float(s0) - key) <= 1e-8:
            return grid
    return None


# Settled mean-bandwidth menu for m_w at s0=-0.5. It is not written into
# MEAN_BW_BY_S0, so m_x and every other location keep the shared grid.
W_S0_M05_BW: dict[str, list[float]] = {
    "m_multipliers": [0.1875, 0.25],
    "z_scales": [32.0, 64.0],
}


# Settled mean-bandwidth menu for m_w at s0=0, chosen by the local inner CV.
# It is not written into MEAN_BW_BY_S0, so m_x and every other location keep
# the shared grid.
W_S0_0_BW: dict[str, list[float]] = {
    "m_multipliers": [0.5, 0.625],
    "z_scales": [16.0, 32.0],
}


def apply_w_s0_0_bandwidth(layer: dict[str, Any], s0: float) -> dict[str, Any]:
    """Use c in {0.5, 0.625} and r in {16, 32} for m_w at s0=0.

    This is the settled m_w menu at s0=0. Other locations and m_x are
    unchanged. The shared MEAN_BW_BY_S0 table is not modified.
    """
    if abs(float(s0)) > 1e-8:
        return layer
    if layer.get("kind") not in {"learned", "learned_mean_oracle_density"}:
        return layer
    selection = layer.get("bandwidth_selection")
    if not isinstance(selection, dict) or selection.get("method") != "inner_cv":
        return layer
    out = dict(layer)
    sel = deepcopy(selection)
    sel["m_multipliers"] = list(W_S0_0_BW["m_multipliers"])
    sel["z_scales"] = list(W_S0_0_BW["z_scales"])
    out["bandwidth_selection"] = sel
    return out


def apply_w_s0_m05_bandwidth(layer: dict[str, Any], s0: float) -> dict[str, Any]:
    """Use c in {0.1875, 0.25} and r in {32, 64} for m_w at s0=-0.5.

    Other locations and m_x are unchanged. The shared MEAN_BW_BY_S0 table
    is not modified.
    """
    if abs(float(s0) - (-0.5)) > 1e-8:
        return layer
    if layer.get("kind") not in {"learned", "learned_mean_oracle_density"}:
        return layer
    selection = layer.get("bandwidth_selection")
    if not isinstance(selection, dict) or selection.get("method") != "inner_cv":
        return layer
    out = dict(layer)
    sel = deepcopy(selection)
    sel["m_multipliers"] = list(W_S0_M05_BW["m_multipliers"])
    sel["z_scales"] = list(W_S0_M05_BW["z_scales"])
    out["bandwidth_selection"] = sel
    return out


def apply_s0_mean_bandwidth(layer: dict[str, Any], s0: float,
                            radius: float = 0.25,
                            mean_fn: str = "paper") -> dict[str, Any]:
    """Use an s0-specific mean-bandwidth grid and score CV near s0.

    Density multipliers are left unchanged. Non-learned layers are returned
    as-is. Unknown locations keep their configured c and r grid.
    """
    if layer.get("kind") not in {"learned", "learned_mean_oracle_density"}:
        return layer
    selection = layer.get("bandwidth_selection")
    if not isinstance(selection, dict) or selection.get("method") != "inner_cv":
        return layer
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("mean CV radius must be finite and positive")
    out = dict(layer)
    sel = deepcopy(selection)
    grid = mean_bw_grid(s0, mean_fn)
    if grid is not None:
        sel["m_multipliers"] = list(grid["m_multipliers"])
        sel["z_scales"] = list(grid["z_scales"])
    sel["cv_center"] = float(s0)
    sel["cv_radius"] = float(radius)
    out["bandwidth_selection"] = sel
    return out


def bandwidths(n: int, c_semi: float, c_conf: float) -> dict[str, float]:
    return {
        "semi": float(c_semi) * n ** (-0.25),
        "conf": float(c_conf) * n ** (-0.4),
        "conf_hsemi": float(c_semi) * n ** (-0.25),
    }


def method_h(method: str, hs: dict[str, float]) -> float:
    return hs[method]


def spacing_c(n: int, h: float, factor: float = 1.0) -> float:
    return float(min(0.05, 0.15 * float(factor) * (n * h / 2.0) ** (-0.2)))


def weighted_quantile(values: Array, weights: Array, p: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not 0 < p < 1:
        raise ValueError("p must lie in (0,1)")
    keep = weights > 0
    if not np.any(keep):
        return math.nan
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    total = float(cdf[-1])
    if total <= 0:
        return math.nan
    idx = int(np.searchsorted(cdf, p * total, side="left"))
    return float(values[min(idx, len(values) - 1)])


def _weighted_mean(values: Array, weights: Array) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return math.nan
    return float(np.sum(weights * values) / total)


# Continuous-root tolerance used only for the auxiliary levels p±c_n.
# The step-function score can remain as large as one kernel weight after the
# interpolated score has already reached zero.
_CONTINUOUS_ROOT_TOL = 1e-7


def _score_grid(absolute: Array, weights: Array, mass: float,
                grid: str) -> tuple[Array, Array]:
    """Nodes for linear interpolation of the weighted |residual| distribution.

    "right" keeps every observation, including zero weights, and places the
    node at the right end of its probability jump. "midpoint_positive" drops
    zero-weight observations and places each node at the middle of its jump.
    """
    if grid == "right":
        order = np.argsort(absolute, kind="mergesort")
        return np.r_[0.0, absolute[order]], np.r_[0.0, np.cumsum(weights[order]) / mass]
    if grid != "midpoint_positive":
        raise ValueError("grid must be 'right' or 'midpoint_positive'")
    keep = weights > 0
    vals, ws = absolute[keep], weights[keep]
    order = np.argsort(vals, kind="mergesort")
    vals, ws = vals[order], ws[order]
    levels = (np.cumsum(ws) - 0.5 * ws) / mass
    return np.r_[0.0, vals], np.r_[0.0, levels]


def solve_semi_local(residual: Array, a_fn, weights: Array, p: float,
                     q0: float, *, accept: str = "step",
                     allow_early_q0: bool = True,
                     grid: str = "right") -> tuple[float, float, bool]:
    """Solve the local semiparametric score, starting from q0.

    accept="step" is the point-estimate rule: a conformal start whose step
    score is already within 1/n_+ is returned as-is, and a Brent root is kept
    only when its step score is within 1.05/n_+. accept="continuous" judges
    the interpolated score instead. The returned score is always the step score.
    grid="right" is that historical interpolant. grid="midpoint_positive"
    uses only positive weights and midpoint jumps; callers that select it also
    turn off the early return.
    """
    if accept not in ("step", "continuous"):
        raise ValueError("accept must be 'step' or 'continuous'")
    residual = np.asarray(residual, dtype=float)
    weights = np.asarray(weights, dtype=float)
    absolute = np.abs(residual)
    mass = float(np.sum(weights))
    if mass <= 0:
        return q0, math.nan, False

    vals, levels = _score_grid(absolute, weights, mass, grid)

    def corr(q: float) -> float:
        return _weighted_mean(residual * np.asarray(a_fn(q), dtype=float), weights)

    def score_step(q: float) -> float:
        return p - _weighted_mean(absolute <= q, weights) - corr(q)

    def score_cont(q: float) -> float:
        return p - float(np.interp(q, vals, levels, left=0.0, right=1.0)) - corr(q)

    s0 = score_step(q0)
    n_pos = max(1.0, float(np.sum(weights > 0)))
    if (allow_early_q0 and np.isfinite(s0)
            and abs(s0) <= (1.0 + 1e-8) / n_pos):
        return float(q0), float(s0), True
    center = score_cont(q0)
    radius = max(0.05, 0.10 * max(q0, 1e-6))
    bracket = None
    for _ in range(14):
        left, right = max(0.0, q0 - radius), q0 + radius
        sl, sr = score_cont(left), score_cont(right)
        candidates = []
        if np.isfinite(sl) and np.isfinite(center) and sl >= 0 >= center:
            candidates.append((left, q0))
        if np.isfinite(center) and np.isfinite(sr) and center >= 0 >= sr:
            candidates.append((q0, right))
        if candidates:
            bracket = min(candidates, key=lambda pair: pair[1] - pair[0])
            break
        radius *= 1.8
    if bracket is None:
        return float(q0), float(s0) if np.isfinite(s0) else math.nan, False
    try:
        q = float(brentq(score_cont, *bracket, xtol=1e-10, rtol=1e-11, maxiter=120))
    except (RuntimeError, ValueError, FloatingPointError):
        return float(q0), float(s0) if np.isfinite(s0) else math.nan, False
    score = score_step(q)
    if accept == "continuous":
        cont = score_cont(q)
        ok = (q >= 0 and np.isfinite(q) and np.isfinite(cont)
              and abs(cont) <= _CONTINUOUS_ROOT_TOL)
    else:
        ok = (q >= 0 and np.isfinite(q) and np.isfinite(score)
              and abs(score) <= max(1.05 / n_pos, 1e-8))
    return q, float(score) if np.isfinite(score) else math.nan, bool(ok)


def _fold_quantiles(residual: Array, weights: Array, a_fn, p: float,
                    c_n: float, *, grid: str = "right",
                    allow_early_q0: bool = True) -> dict[str, float]:
    q_conf = weighted_quantile(np.abs(residual), weights, p)
    q_plus = weighted_quantile(np.abs(residual), weights, min(0.999, p + c_n))
    q_minus = weighted_quantile(np.abs(residual), weights, max(1e-4, p - c_n))
    q_semi, score, ok = solve_semi_local(
        residual, a_fn, weights, p, q_conf, grid=grid, allow_early_q0=allow_early_q0,
        accept="continuous" if grid == "midpoint_positive" else "step",
    )
    inv_g = (q_plus - q_minus) / (2.0 * c_n) if np.isfinite(q_plus) and np.isfinite(q_minus) else math.nan
    return {
        "conf": q_conf,
        "semi": q_semi,
        "q_plus": float(q_plus) if np.isfinite(q_plus) else math.nan,
        "q_minus": float(q_minus) if np.isfinite(q_minus) else math.nan,
        "inv_g": inv_g,
        "score": score,
        "success": float(ok),
        "n_pos": float(np.sum(weights > 0)),
        "mass": float(np.sum(weights)),
    }


def cross_fit_nuisance(data: Data, law: ErrorLaw, layer: dict[str, Any],
                       config: dict[str, Any], beta: float,
                       s0: float | None = None) -> dict[str, Any]:
    """Fit each target on each training fold once.

    s0 localizes the mean-bandwidth CV only when mean_bw_by_s0 is set.
    A curve reuses the returned object at every location.
    """
    n = len(data.y)
    mean_fn = config.get("mean_fn", "paper")
    fold_fits: dict[str, list] = {"x": [], "w": []}
    fold_prep: dict[str, list] = {"x": [], "w": []}
    fold_resid: dict[str, list] = {"x": [], "w": []}
    for j, ev in enumerate(data.folds):
        tr = data.folds[1 - j]
        for target in ("x", "w"):
            fit_layer = layer
            if s0 is not None and config.get("mean_bw_by_s0", True):
                fit_layer = apply_s0_mean_bandwidth(
                    layer, s0, float(config.get("mean_cv_radius", 0.25)),
                    mean_fn,
                )
            if s0 is not None and target == "w":
                fit_layer = apply_w_s0_m05_bandwidth(fit_layer, s0)
                fit_layer = apply_w_s0_0_bandwidth(fit_layer, s0)
            fitted = fit_nuisance(target, beta, law, data.x[tr], data.z[tr], data.y[tr],
                                  {**fit_layer, "sample_size": n}, 1.0, mean_fn)
            prep = fitted.prepare(data.x[ev], data.z[ev])
            fold_fits[target].append(fitted)
            fold_prep[target].append(prep)
            fold_resid[target].append(data.y[ev] - prep.mean)
    return {"fits": fold_fits, "prep": fold_prep, "resid": fold_resid}


def estimate_at_s0(data: Data, law: ErrorLaw, layer: dict[str, Any],
                   config: dict[str, Any], s0: float, beta: float,
                   truth: dict[str, float],
                   cross_fit: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    n = len(data.y)
    alpha = float(config["alpha"])
    p = 1.0 - alpha
    mean_fn = config.get("mean_fn", "paper")
    hs = bandwidths(n, float(config["c_semi"]), float(config["c_conf"]))
    methods = list(config.get("methods", ("semi", "conf", "conf_hsemi")))

    if cross_fit is None:
        cross_fit = cross_fit_nuisance(data, law, layer, config, beta, s0)
    fold_fits = cross_fit["fits"]
    fold_prep = cross_fit["prep"]
    fold_resid = cross_fit["resid"]

    rows = []
    z_quad, w_quad = z_nodes(96)
    x_quad = np.full(z_quad.shape, s0)
    for method in methods:
        h = method_h(method, hs)
        c_n = spacing_c(n, h, float(config.get("c_n_factor", 1.0)))
        root_mode = str(config.get("semi_root", "production"))
        if root_mode not in {"production", "no_early", "midpoint_positive"}:
            raise ValueError("semi_root must be production, no_early, or midpoint_positive")
        # production: right-end interpolant, including zero weights, early return.
        # no_early: same interpolant, but do not return the pilot quantile early.
        # midpoint_positive: positive weights only, midpoint nodes, no early return.
        root_grid = "midpoint_positive" if root_mode == "midpoint_positive" else "right"
        allow_early = root_mode == "production"
        kind = "semi" if method == "semi" else "conf"
        fold_q: dict[str, list[float]] = {"x": [], "w": []}
        fold_inv_g: dict[str, list[float]] = {"x": [], "w": []}
        fold_inv_g_slope: dict[str, list[float]] = {"x": [], "w": []}
        slope_fold_ok: dict[str, list[bool]] = {"x": [], "w": []}
        slope_fold_unordered: dict[str, list[bool]] = {"x": [], "w": []}
        slope_fold_step: dict[str, list[float]] = {"x": [], "w": []}
        slope_fold_jump: dict[str, list[float]] = {"x": [], "w": []}
        fold_f: list[float] = []
        fold_phi: dict[str, list[Array]] = {"x": [], "w": []}
        fold_ind: dict[str, list[Array]] = {"x": [], "w": []}
        fold_w: list[Array] = []
        fold_n: list[int] = []
        successes = []
        n_eff = []
        n_window = []
        for j, ev in enumerate(data.folds):
            wloc = kernel_h(data.x[ev], s0, h)
            fold_w.append(wloc)
            fold_n.append(len(ev))
            fold_f.append(float(np.mean(wloc)))
            n_eff.append(float(np.sum(wloc) * h))
            n_window.append(int(np.sum(np.abs(data.x[ev] - s0) <= h)))
            for target in ("x", "w"):
                residual = fold_resid[target][j]
                prep = fold_prep[target][j]
                stats = _fold_quantiles(
                    residual, wloc, prep.A, p, c_n, grid=root_grid,
                    allow_early_q0=allow_early,
                )
                q = stats[kind]
                fold_q[target].append(q)
                fold_inv_g[target].append(stats["inv_g"])
                if method == "semi":
                    successes.append(stats["success"])
                    p_hi = min(0.999, p + c_n)
                    p_lo = max(1e-4, p - c_n)
                    mass = float(np.sum(wloc))
                    jump = float(np.max(wloc) / mass) if mass > 0 else math.nan
                    if np.isfinite(stats["q_plus"]) and np.isfinite(stats["q_minus"]) and c_n > 0:
                        q_hi, step_hi, ok_hi = solve_semi_local(
                            residual, prep.A, wloc, p_hi, stats["q_plus"],
                            accept="continuous", allow_early_q0=False, grid=root_grid)
                        q_lo, step_lo, ok_lo = solve_semi_local(
                            residual, prep.A, wloc, p_lo, stats["q_minus"],
                            accept="continuous", allow_early_q0=False, grid=root_grid)
                        steps = [v for v in (step_hi, step_lo) if np.isfinite(v)]
                        slope_fold_step[target].append(float(np.max(np.abs(steps))) if steps else math.nan)
                        slope_fold_jump[target].append(jump)
                        ordered = (ok_hi and ok_lo and np.isfinite(q_hi) and np.isfinite(q_lo)
                                   and q_hi > q_lo)
                        slope_fold_ok[target].append(bool(ordered))
                        slope_fold_unordered[target].append(bool(
                            ok_hi and ok_lo and np.isfinite(q_hi) and np.isfinite(q_lo)
                            and not q_hi > q_lo))
                        if ordered:
                            fold_inv_g_slope[target].append((q_hi - q_lo) / (2.0 * c_n))
                        else:
                            fold_inv_g_slope[target].append(math.nan)
                    else:
                        slope_fold_ok[target].append(False)
                        slope_fold_unordered[target].append(False)
                        slope_fold_step[target].append(math.nan)
                        slope_fold_jump[target].append(jump)
                        fold_inv_g_slope[target].append(math.nan)
                else:
                    successes.append(float(np.isfinite(q)))
                    fold_inv_g_slope[target].append(math.nan)
                a = np.asarray(prep.A(q), dtype=float) if method == "semi" else 0.0
                phi = (p - (np.abs(residual) <= q).astype(float)
                       - (residual * a if method == "semi" else 0.0))
                fold_phi[target].append(phi)
                fold_ind[target].append((np.abs(residual) <= q).astype(float))

        f_s = float(np.mean(fold_f))
        inv_g = {v: float(np.nanmean(fold_inv_g[v])) for v in ("x", "w")}
        g_spacing = {v: (1.0 / inv_g[v] if np.isfinite(inv_g[v]) and inv_g[v] > 0 else math.nan)
                     for v in ("x", "w")}
        inv_g_slope = {v: float(np.nanmean(fold_inv_g_slope[v])) if fold_inv_g_slope[v] else math.nan
                       for v in ("x", "w")}
        g_slope = {v: (1.0 / inv_g_slope[v] if np.isfinite(inv_g_slope[v]) and inv_g_slope[v] > 0 else math.nan)
                   for v in ("x", "w")}
        g_true = {}
        for target, zeta_key, beta_residual in (
            ("x", "zeta_x", beta), ("w", "zeta_w", 0.0),
        ):
            zeta = float(truth[zeta_key])
            g_true[target] = float(
                law.conditional_pdf(zeta, s0, beta_residual)
                + law.conditional_pdf(-zeta, s0, beta_residual)
            )
        g_estimator = str(config.get("g_estimator", "spacing"))
        if config.get("oracle_g"):
            g = dict(g_true)
        elif g_estimator == "semi_slope" and method == "semi":
            g = dict(g_slope)
        else:
            g = dict(g_spacing)
        q_hat = {v: float(np.mean(fold_q[v])) for v in ("x", "w")}
        delta_hat = q_hat["x"] - q_hat["w"]

        def weighted_cross(a: list[Array], b: list[Array]) -> float:
            acc, den = 0.0, 0
            for j in range(2):
                acc += float(np.mean(a[j] * b[j] * fold_w[j]))
                den += 1
            return acc / den

        if method == "semi":
            sxx = weighted_cross(fold_phi["x"], fold_phi["x"])
            sww = weighted_cross(fold_phi["w"], fold_phi["w"])
            sxw = weighted_cross(fold_phi["x"], fold_phi["w"])
            rho = math.nan
            denom_x = f_s * g["x"] * g["x"] if np.isfinite(g["x"]) else math.nan
            denom_w = f_s * g["w"] * g["w"] if np.isfinite(g["w"]) else math.nan
            denom_xw = f_s * g["x"] * g["w"] if np.isfinite(g["x"]) and np.isfinite(g["w"]) else math.nan
            sig_xx = sxx / denom_x if denom_x not in (0, math.nan) and np.isfinite(denom_x) else math.nan
            sig_ww = sww / denom_w if denom_w not in (0, math.nan) and np.isfinite(denom_w) else math.nan
            sig_xw = sxw / denom_xw if denom_xw not in (0, math.nan) and np.isfinite(denom_xw) else math.nan
            v_delta = (K2_INT / f_s) * (sig_xx + sig_ww - 2.0 * sig_xw) if np.isfinite(f_s) and f_s > 0 else math.nan
            v_x = (K2_INT / f_s) * sig_xx if np.isfinite(f_s) and f_s > 0 else math.nan
            v_w = (K2_INT / f_s) * sig_ww if np.isfinite(f_s) and f_s > 0 else math.nan
        else:
            sxx = sww = sxw = math.nan
            rho = (weighted_cross(fold_ind["x"], fold_ind["w"]) / f_s - p * p
                   if np.isfinite(f_s) and f_s > 0 else math.nan)
            gx, gw = g["x"], g["w"]
            if all(np.isfinite(v) and v > 0 for v in (f_s, gx, gw)) and np.isfinite(rho):
                v_delta = (K2_INT / f_s) * (
                    alpha * p / (gx * gx) + alpha * p / (gw * gw) - 2.0 * rho / (gx * gw)
                )
                v_x = (K2_INT / f_s) * (alpha * p / (gx * gx))
                v_w = (K2_INT / f_s) * (alpha * p / (gw * gw))
            else:
                v_delta = v_x = v_w = math.nan

        nh = n * h
        se = {
            "x": math.sqrt(v_x / nh) if np.isfinite(v_x) and v_x > 0 else math.nan,
            "w": math.sqrt(v_w / nh) if np.isfinite(v_w) and v_w > 0 else math.nan,
            "delta": math.sqrt(v_delta / nh) if np.isfinite(v_delta) and v_delta > 0 else math.nan,
        }

        center_x = float(0.5 * (
            fold_fits["x"][0].predict_mean(np.array([s0]), np.array([0.0]))[0]
            + fold_fits["x"][1].predict_mean(np.array([s0]), np.array([0.0]))[0]
        ))
        center_w = 0.5 * (
            fold_fits["w"][0].predict_mean(x_quad, z_quad)
            + fold_fits["w"][1].predict_mean(x_quad, z_quad)
        )
        m0s = float(m0(np.array([s0]), mean_fn)[0])
        loc_x = (m0s - center_x) + beta * z_quad
        loc_w = m0s + beta * z_quad - center_w
        pred_x = residual_coverage(loc_x, w_quad, q_hat["x"], s0, law)
        pred_w = residual_coverage(loc_w, w_quad, q_hat["w"], s0, law)

        estimator_ok = all(np.isfinite(q_hat[v]) and q_hat[v] >= 0 for v in ("x", "w"))
        root_rate = float(np.mean(successes)) if successes else 0.0
        inference_ok = (estimator_ok and np.isfinite(se["delta"]) and se["delta"] > 0
                        and np.isfinite(se["x"]) and np.isfinite(se["w"]))
        length_x = 2.0 * q_hat["x"]
        length_w = 2.0 * q_hat["w"]
        density_diag: dict[str, float] = {}
        for target in ("x", "w"):
            diags = [fit.diagnostics() for fit in fold_fits[target] if hasattr(fit, "diagnostics")]
            for key in ("cv_f_multiplier", "cv_y_multiplier", "density_feature_x_only",
                        "h_f", "h_f_cv", "hf_post_multiplier"):
                vals = [float(d[key]) for d in diags if key in d]
                density_diag[f"{key}_{target}"] = float(np.mean(vals)) if vals else math.nan
        rows.append({
            "method": method,
            "s0": s0,
            "h": h,
            "nh": nh,
            "c_n": c_n,
            "status": "ok" if estimator_ok else "failed",
            "estimator_valid": bool(estimator_ok),
            "inference_valid": bool(inference_ok),
            "zeta_x_est": q_hat["x"],
            "zeta_w_est": q_hat["w"],
            "delta_est": delta_hat,
            "se_x": se["x"],
            "se_w": se["w"],
            "se_delta": se["delta"],
            "v_x": v_x,
            "v_w": v_w,
            "v_delta": v_delta,
            "f_s": f_s,
            "ginv_x": inv_g["x"],
            "ginv_w": inv_g["w"],
            "g_spacing_x": g_spacing["x"],
            "g_spacing_w": g_spacing["w"],
            "ginv_slope_x": inv_g_slope["x"],
            "ginv_slope_w": inv_g_slope["w"],
            "g_slope_x": g_slope["x"],
            "g_slope_w": g_slope["w"],
            "slope_folds_x": int(sum(slope_fold_ok["x"])) if method == "semi" else math.nan,
            "slope_folds_w": int(sum(slope_fold_ok["w"])) if method == "semi" else math.nan,
            "slope_unordered_x": int(sum(slope_fold_unordered["x"])) if method == "semi" else math.nan,
            "slope_unordered_w": int(sum(slope_fold_unordered["w"])) if method == "semi" else math.nan,
            "slope_step_x": (float(np.nanmax(slope_fold_step["x"]))
                             if method == "semi" and np.any(np.isfinite(slope_fold_step["x"])) else math.nan),
            "slope_step_w": (float(np.nanmax(slope_fold_step["w"]))
                             if method == "semi" and np.any(np.isfinite(slope_fold_step["w"])) else math.nan),
            "slope_jump_x": (float(np.nanmax(slope_fold_jump["x"]))
                             if method == "semi" and np.any(np.isfinite(slope_fold_jump["x"])) else math.nan),
            "slope_jump_w": (float(np.nanmax(slope_fold_jump["w"]))
                             if method == "semi" and np.any(np.isfinite(slope_fold_jump["w"])) else math.nan),
            "g_true_x": g_true["x"],
            "g_true_w": g_true["w"],
            "score_xx": sxx,
            "score_ww": sww,
            "score_xw": sxw,
            "rho": rho,
            "n_eff": float(np.mean(n_eff)),
            "n_window": float(np.mean(n_window)),
            "root_success_rate": root_rate,
            "pred_cover_x": pred_x,
            "pred_cover_w": pred_w,
            "pred_length_x": length_x,
            "pred_length_w": length_w,
            "center_x": center_x,
            **density_diag,
        })
    return rows


def estimate_curve(data: Data, law: ErrorLaw, layer: dict[str, Any],
                   config: dict[str, Any], locations: list[float], beta: float,
                   truth_by_s0: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Local roots along one sample.

    With mean_bw_by_s0, each location gets its own nuisance fit, including
    the settled m_w menus. Otherwise one fit is reused at every location.
    """
    rows: list[dict[str, Any]] = []
    if config.get("mean_bw_by_s0", True):
        for s0 in locations:
            truth = truth_by_s0[f"{float(s0):.6g}"]
            rows.extend(estimate_at_s0(
                data, law, layer, config, float(s0), beta, truth,
            ))
        return rows
    shared = dict(config)
    shared["mean_bw_by_s0"] = False
    fitted = cross_fit_nuisance(data, law, layer, shared, beta, s0=None)
    for s0 in locations:
        truth = truth_by_s0[f"{float(s0):.6g}"]
        rows.extend(estimate_at_s0(
            data, law, layer, shared, float(s0), beta, truth, cross_fit=fitted,
        ))
    return rows
