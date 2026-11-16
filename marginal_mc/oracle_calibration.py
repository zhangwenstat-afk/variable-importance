#!/usr/bin/env python3
"""Oracle calibration for the proposed marginal simulation (not Monte Carlo results).

DGP
---
X ~ Uniform[-1, 1], Z ~ Uniform[-sqrt(3), sqrt(3)], independently.
Y = m0(X) + beta*Z + (1 + 0.4*X)*epsilon,
m0(x) = x + 2*x**2 + 0.5*sin(pi*x) by default, x + 2*x**2 when
mean_fn='quadratic', or x + 0.5*x**2 when mean_fn='quad_half'.
Residual laws, zeta, and beta calibration do not depend on m0.

Errors:
  symmetric: epsilon = G.
  right:     epsilon = [G + a*(B-p)] / sqrt(1+a*a*p*(1-p)),
             B ~ Bernoulli(p), independent of G; defaults p=0.2, a=5.
  left:      the negative of the right-skewed error.
  G is N(0,1) by default, or the unit-variance t_4 law G = T_4/sqrt(2)
  when error_base='t4'. When error_base='exponnorm', the skewed laws use the
  standardized exponential-modified normal
  epsilon = [G + lambda*(E-1)] / sqrt(1+lambda^2), E ~ Exp(1), default lambda=1.5.

All errors have mean zero and variance one. The code computes oracle residual
quantiles, Delta(beta), beta values for prescribed Delta, and E[A_v]/g_v.
It uses exact location-scale mixture / uniform-Z convolution formulas and
deterministic Gauss-Legendre integration over X. No estimator-performance
simulation is run.

Requirements: Python 3.10+, numpy, scipy.
Usage:
  python marginal_oracle_calibration.py
  python marginal_oracle_calibration.py --alpha 0.05 --nodes 320
  python marginal_oracle_calibration.py --deltas 0.10 0.15 0.20 0.25 0.30 0.40
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr
from scipy.stats import exponnorm

SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT_3 = math.sqrt(3.0)
ERROR_BASES = ("normal", "t4", "exponnorm")
EMG_LAMBDA_DEFAULT = 1.5


def _emg_shape(emg_lambda: float) -> tuple[float, float]:
    lam = float(emg_lambda)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("emg_lambda must be finite and positive")
    return lam, math.sqrt(1.0 + lam * lam)


def _emg_w(u: np.ndarray, emg_lambda: float, skew_sign: float) -> tuple[np.ndarray, float, float]:
    """Map standardized EMG points to SciPy exponnorm(K=λ, loc=0, scale=1)."""
    lam, scale = _emg_shape(emg_lambda)
    sign = 1.0 if float(skew_sign) >= 0.0 else -1.0
    return scale * (sign * np.asarray(u, dtype=float)) + lam, lam, scale


def _emg_cdf_W_integral(w: np.ndarray, lam: float) -> np.ndarray:
    """Antiderivative of the exponnorm(K=λ) CDF."""
    alpha = 1.0 / lam
    c = 0.5 * alpha * alpha
    w = np.asarray(w, dtype=float)
    phi = np.exp(-0.5 * w * w) / SQRT_2PI
    Phi = ndtr(w)
    exp_Phi = np.exp(np.clip(c - alpha * w + log_ndtr(w - alpha), -745.0, 745.0))
    return w * Phi + phi + exp_Phi / alpha - Phi / alpha


def standardized_pdf(u: np.ndarray, base: str, emg_lambda: float = EMG_LAMBDA_DEFAULT,
                     skew_sign: float = 1.0) -> np.ndarray:
    """PDF of the unit-variance base law G, or of standardized EMG when requested."""
    if base == "exponnorm":
        w, lam, scale = _emg_w(u, emg_lambda, skew_sign)
        return scale * exponnorm.pdf(w, lam)
    if base == "normal":
        return np.exp(-0.5 * np.asarray(u, dtype=float) ** 2) / SQRT_2PI
    return 3.0 * np.power(np.asarray(u, dtype=float) ** 2 + 2.0, -2.5)


def standardized_pdf_derivative(u: np.ndarray, base: str,
                                emg_lambda: float = EMG_LAMBDA_DEFAULT,
                                skew_sign: float = 1.0) -> np.ndarray:
    """Derivative of the unit-variance base-law PDF."""
    if base == "exponnorm":
        w, lam, scale = _emg_w(u, emg_lambda, skew_sign)
        sign = 1.0 if float(skew_sign) >= 0.0 else -1.0
        alpha = 1.0 / lam
        c = 0.5 * alpha * alpha
        phi_shift = np.exp(-0.5 * (w - alpha) ** 2) / SQRT_2PI
        exp_term = np.exp(np.clip(c - alpha * w, -745.0, 745.0))
        deriv_W = alpha * exp_term * (-alpha * ndtr(w - alpha) + phi_shift)
        return sign * (scale ** 2) * deriv_W
    if base == "normal":
        return -np.asarray(u, dtype=float) * standardized_pdf(u, base)
    u = np.asarray(u, dtype=float)
    return -15.0 * u * np.power(u * u + 2.0, -3.5)


def standardized_cdf(u: np.ndarray, base: str, emg_lambda: float = EMG_LAMBDA_DEFAULT,
                     skew_sign: float = 1.0) -> np.ndarray:
    """CDF of the unit-variance base law G."""
    if base == "exponnorm":
        w, lam, _scale = _emg_w(u, emg_lambda, skew_sign)
        cdf_right = exponnorm.cdf(w, lam)
        return cdf_right if float(skew_sign) >= 0.0 else 1.0 - cdf_right
    if base == "normal":
        return ndtr(u)
    root = np.sqrt(np.asarray(u, dtype=float) ** 2 + 2.0)
    v = np.asarray(u, dtype=float) / root
    return 0.5 + 0.75 * v - 0.25 * v * v * v


def standardized_sf(u: np.ndarray, base: str, emg_lambda: float = EMG_LAMBDA_DEFAULT,
                    skew_sign: float = 1.0) -> np.ndarray:
    """Survival function of the unit-variance base law G."""
    if base == "exponnorm":
        w, lam, _scale = _emg_w(u, emg_lambda, skew_sign)
        sf_right = exponnorm.sf(w, lam)
        return sf_right if float(skew_sign) >= 0.0 else 1.0 - sf_right
    if base == "normal":
        return ndtr(-np.asarray(u, dtype=float))
    root = np.sqrt(np.asarray(u, dtype=float) ** 2 + 2.0)
    v = np.asarray(u, dtype=float) / root
    return 0.5 - 0.75 * v + 0.25 * v * v * v


def standardized_cdf_integral(u: np.ndarray, base: str,
                              emg_lambda: float = EMG_LAMBDA_DEFAULT,
                              skew_sign: float = 1.0) -> np.ndarray:
    """Antiderivative of the standardized CDF, used for Uniform-Z convolution."""
    if base == "exponnorm":
        w, lam, scale = _emg_w(u, emg_lambda, skew_sign)
        G = _emg_cdf_W_integral(w, lam) / scale
        if float(skew_sign) >= 0.0:
            return G
        return np.asarray(u, dtype=float) + G
    if base == "normal":
        u = np.asarray(u, dtype=float)
        return u * ndtr(u) + np.exp(-0.5 * u * u) / SQRT_2PI
    u = np.asarray(u, dtype=float)
    return 0.5 * u + (u * u + 1.0) / (2.0 * np.sqrt(u * u + 2.0))


@dataclass(frozen=True)
class BoundaryMetrics:
    zeta: float
    g: float
    mean_A: float
    kappa: float  # E[A_v(V)] / g_v


class MarginalOracle:
    """Oracle residual distributions: beta=0 is the full-model residual."""

    def __init__(self, error: str, alpha: float = 0.10, nodes: int = 160,
                 mixture_p: float = 0.20, mixture_a: float = 5.0,
                 error_base: str = "normal",
                 emg_lambda: float = EMG_LAMBDA_DEFAULT) -> None:
        if error not in {"symmetric", "right", "left"}:
            raise ValueError("error must be symmetric, right, or left")
        if error_base not in ERROR_BASES:
            raise ValueError("error_base must be normal, t4, or exponnorm")
        if error_base == "exponnorm" and error == "symmetric":
            raise ValueError("exponnorm error_base requires a skewed error")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if nodes < 16:
            raise ValueError("use at least 16 quadrature nodes")
        if not 0.0 < mixture_p < 1.0 or mixture_a < 0.0:
            raise ValueError("invalid mixture parameters")
        self.error, self.alpha = error, float(alpha)
        self.error_base = error_base
        self.emg_lambda = float(emg_lambda) if error_base == "exponnorm" else EMG_LAMBDA_DEFAULT
        if error_base == "exponnorm":
            _emg_shape(self.emg_lambda)
        self.skew_sign = -1.0 if error_base == "exponnorm" and error == "left" else 1.0
        self.x, weights = leggauss(nodes)
        self.weights = weights / 2.0
        self.sigma = 1.0 + 0.4 * self.x
        if error == "symmetric" or error_base == "exponnorm":
            probs = np.array([1.0])
            means = np.array([0.0])
            stds = np.array([1.0])
        else:
            scale = math.sqrt(1.0 + mixture_a**2 * mixture_p * (1.0-mixture_p))
            probs = np.array([1.0-mixture_p, mixture_p])
            means = np.array([-mixture_a*mixture_p,
                              mixture_a*(1.0-mixture_p)]) / scale
            if error == "left":
                means = -means
            stds = np.array([1.0, 1.0]) / scale
        self.probs = probs
        self.locs = self.sigma[:, None] * means
        self.scales = self.sigma[:, None] * stds
        self.zeta_w = self.quantile(0.0)

    def _base_kw(self) -> dict[str, float]:
        return {"emg_lambda": self.emg_lambda, "skew_sign": self.skew_sign}

    @staticmethod
    def _validate_beta(beta: float) -> float:
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        return float(beta)

    def conditional_cdf(self, t: float, beta: float) -> np.ndarray:
        """CDF of beta*Z + sigma(X)*epsilon at the quadrature X values."""
        beta = self._validate_beta(beta)
        a = beta * SQRT_3
        if a < 1e-7:
            u = (t - self.locs) / self.scales
            return np.clip(standardized_cdf(u, self.error_base, **self._base_kw()) @ self.probs, 0.0, 1.0)
        u_hi = (t + a - self.locs) / self.scales
        u_lo = (t - a - self.locs) / self.scales
        integ_hi = self.scales * standardized_cdf_integral(u_hi, self.error_base, **self._base_kw())
        integ_lo = self.scales * standardized_cdf_integral(u_lo, self.error_base, **self._base_kw())
        values = ((integ_hi - integ_lo) / (2.0 * a)) @ self.probs
        return np.clip(values, 0.0, 1.0)

    def _cdf_difference(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        diff = np.where(
            lo > 0,
            standardized_sf(lo, self.error_base, **self._base_kw())
            - standardized_sf(hi, self.error_base, **self._base_kw()),
            standardized_cdf(hi, self.error_base, **self._base_kw())
            - standardized_cdf(lo, self.error_base, **self._base_kw()),
        )
        return diff

    def conditional_pdf(self, t: float, beta: float) -> np.ndarray:
        beta = self._validate_beta(beta)
        a = beta * SQRT_3
        if a < 1e-7:
            u = (t - self.locs) / self.scales
            return (standardized_pdf(u, self.error_base, **self._base_kw()) / self.scales) @ self.probs
        lo = (t - a - self.locs) / self.scales
        hi = (t + a - self.locs) / self.scales
        return (self._cdf_difference(lo, hi) / (2.0 * a)) @ self.probs

    def absolute_cdf(self, zeta: float, beta: float) -> float:
        if zeta < 0.0:
            return 0.0
        return float(self.weights @ (self.conditional_cdf(zeta, beta)
                                     - self.conditional_cdf(-zeta, beta)))

    def quantile(self, beta: float) -> float:
        beta = self._validate_beta(beta)
        target = 1.0-self.alpha
        hi = max(2.0, 2.0+beta*SQRT_3)
        while self.absolute_cdf(hi, beta) < target:
            hi *= 2.0
            if hi > 1e5:
                raise RuntimeError("failed to bracket the residual quantile")
        return float(brentq(lambda q: self.absolute_cdf(q, beta)-target,
                            0.0, hi, xtol=1e-12, rtol=1e-12))

    def delta(self, beta: float) -> float:
        return self.quantile(beta)-self.zeta_w

    def calibrate_beta(self, target_delta: float, max_beta: float = 8.0) -> float:
        """Find the first crossing on a fixed grid; no global monotonicity assumed."""
        if not math.isfinite(target_delta) or target_delta < 0.0:
            raise ValueError("target_delta must be finite and nonnegative")
        if target_delta == 0.0:
            return 0.0
        lo, f_lo = 0.0, -target_delta
        for hi in np.linspace(0.025, max_beta, 320):
            f_hi = self.delta(float(hi))-target_delta
            if f_lo*f_hi <= 0.0:
                return float(brentq(lambda b: self.delta(b)-target_delta,
                                    lo, float(hi), xtol=1e-11, rtol=1e-11))
            lo, f_lo = float(hi), f_hi
        raise RuntimeError(f"no beta crossing found up to {max_beta}")

    def boundary_metrics(self, beta: float) -> BoundaryMetrics:
        zeta = self.quantile(beta)
        positive = self.conditional_pdf(zeta, beta)
        negative = self.conditional_pdf(-zeta, beta)
        g = float(self.weights @ (positive+negative))
        mean_A = float(self.weights @ (positive-negative))
        if g <= 0.0:
            raise RuntimeError("nonpositive oracle density at the quantile")
        return BoundaryMetrics(zeta, g, mean_A, mean_A/g)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--nodes", type=int, default=160)
    parser.add_argument("--mixture-p", type=float, default=0.20)
    parser.add_argument("--mixture-a", type=float, default=5.0)
    parser.add_argument("--error-base", choices=list(ERROR_BASES), default="normal")
    parser.add_argument("--emg-lambda", type=float, default=EMG_LAMBDA_DEFAULT)
    parser.add_argument("--deltas", nargs="+", type=float,
                        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60])
    args = parser.parse_args(argv)
    errors = ("right", "left") if args.error_base == "exponnorm" else ("symmetric", "right", "left")
    for error in errors:
        oracle = MarginalOracle(error, args.alpha, args.nodes,
                                args.mixture_p, args.mixture_a, args.error_base,
                                args.emg_lambda)
        w = oracle.boundary_metrics(0.0)
        print(f"\nerror={error}; error_base={args.error_base}; alpha={args.alpha}; nodes={args.nodes}")
        print(f"zeta_w={w.zeta:.9f}; g_w={w.g:.9f}; kappa_w={w.kappa:.9f}")
        print(" Delta       beta          zeta_x        kappa_x       kappa_x-kappa_w")
        for delta in args.deltas:
            beta = oracle.calibrate_beta(delta)
            x = oracle.boundary_metrics(beta)
            print(f" {delta:5.2f}   {beta:12.9f}  {x.zeta:12.9f}"
                  f"  {x.kappa:12.9f}  {x.kappa-w.kappa:16.9f}")


if __name__ == "__main__":
    main()
