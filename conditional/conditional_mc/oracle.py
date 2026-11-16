"""Deterministic conditional targets zeta_v(s) and Delta(s; beta)."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from marginal_mc.dgp import ErrorLaw, SQRT3, sigma
from marginal_mc.oracle_calibration import standardized_cdf

ALPHA_DEFAULT = 0.10
NODES_DEFAULT = 160


def error_cdf(law: ErrorLaw, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    probs, means, stds = law.components
    u = (t[..., None] - means) / stds
    return np.sum(probs * standardized_cdf(u, law.error_base, **law._base_kw()), axis=-1)


def abs_coverage(law: ErrorLaw, q: float) -> float:
    return float(error_cdf(law, q) - error_cdf(law, -q))


@lru_cache(maxsize=64)
def q0_abs(error: str, alpha: float, mixture_p: float, mixture_a: float,
           error_base: str, emg_lambda: float) -> float:
    law = ErrorLaw(error, mixture_p, mixture_a, error_base, emg_lambda)
    target = 1.0 - alpha
    return float(brentq(lambda q: abs_coverage(law, q) - target, 1e-8, 20.0,
                        xtol=1e-12, rtol=1e-12))


def z_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    u, w = leggauss(nodes)
    return SQRT3 * u, w / 2.0


def _z_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return z_nodes(nodes)


def zeta_w(s: float, q0: float) -> float:
    return float(sigma(np.array([s]))[0] * q0)


def zeta_x(s: float, beta: float, law: ErrorLaw, q0: float, nodes: int,
           alpha: float = ALPHA_DEFAULT) -> float:
    sig = float(sigma(np.array([s]))[0])
    z, w = _z_nodes(nodes)
    target = 1.0 - alpha

    def coverage(q: float) -> float:
        return float(np.sum(w * (
            error_cdf(law, (q - beta * z) / sig) - error_cdf(law, (-q - beta * z) / sig)
        )))

    lo, hi = max(1e-6, 0.2 * sig * q0), max(4.0 * sig * q0 + abs(beta) * SQRT3, 1.0)
    for _ in range(12):
        if coverage(hi) >= target:
            break
        hi *= 1.6
    return float(brentq(lambda q: coverage(q) - target, lo, hi, xtol=1e-12, rtol=1e-12))


def calibrate_beta(s: float, target_delta: float, law: ErrorLaw, q0: float,
                   nodes: int, max_beta: float = 2.5,
                   alpha: float = ALPHA_DEFAULT) -> float:
    zw = zeta_w(s, q0)
    if target_delta <= 0:
        raise ValueError("target_delta must be positive")

    def gap(beta: float) -> float:
        return zeta_x(s, beta, law, q0, nodes, alpha) - zw - target_delta

    lo, f_lo = 0.0, -target_delta
    for hi in np.linspace(0.02, max_beta, 400):
        f_hi = gap(float(hi))
        if f_lo * f_hi <= 0.0:
            return float(brentq(gap, lo, float(hi), xtol=1e-12, rtol=1e-12))
        lo, f_lo = float(hi), f_hi
    raise RuntimeError(f"no beta crossing at s={s} for Delta={target_delta}")


@lru_cache(maxsize=1024)
def conditional_truth(error: str, s0: float, alpha: float, beta: float | None,
                      target_delta: float | None, nodes: int = NODES_DEFAULT,
                      mixture_p: float = 0.2, mixture_a: float = 5.0,
                      error_base: str = "normal",
                      emg_lambda: float = 1.5) -> dict[str, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    law = ErrorLaw(error, mixture_p, mixture_a, error_base, emg_lambda)
    q0 = q0_abs(error, alpha, mixture_p, mixture_a, error_base, emg_lambda)
    zw = zeta_w(s0, q0)
    if beta is None:
        if target_delta is None:
            raise ValueError("need beta or target_delta")
        beta = calibrate_beta(s0, float(target_delta), law, q0, nodes, alpha=alpha)
    else:
        beta = float(beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be finite and positive")
    zx = zeta_x(s0, beta, law, q0, nodes, alpha)
    return {
        "beta": beta,
        "s0": float(s0),
        "q0": q0,
        "sigma_s": float(sigma(np.array([s0]))[0]),
        "zeta_x": zx,
        "zeta_w": zw,
        "delta_true": zx - zw,
    }


def residual_coverage(loc: np.ndarray, weights: np.ndarray, half: float,
                      s: float, law: ErrorLaw) -> float:
    """P(|loc(Z) + sigma(s) e0| <= half) with Z-quadrature nodes/weights."""
    sig = float(sigma(np.array([s]))[0])
    loc = np.asarray(loc, dtype=float)
    return float(np.sum(weights * (
        error_cdf(law, (half - loc) / sig) - error_cdf(law, (-half - loc) / sig)
    )))
