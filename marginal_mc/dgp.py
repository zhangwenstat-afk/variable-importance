"""Data generation and analytic truth; no Monte Carlo approximation to target values."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import leggauss

from .oracle_calibration import (ERROR_BASES, EMG_LAMBDA_DEFAULT, MarginalOracle,
                                standardized_pdf, standardized_pdf_derivative,
                                standardized_cdf, standardized_sf)

FloatArray = NDArray[np.float64]
SQRT3 = math.sqrt(3.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
MEAN_FNS = ("paper", "quadratic", "quad_half", "sine_quad")


def m0(x: FloatArray, mean_fn: str = "paper") -> FloatArray:
    x = np.asarray(x, dtype=float)
    if mean_fn == "paper":
        return x + 2.0 * x*x + 0.5 * np.sin(np.pi*x)
    if mean_fn == "quadratic":
        return x + 2.0 * x*x
    if mean_fn == "quad_half":
        return x + 0.5 * x*x
    # m''(x) = 1 - sin(pi x): 2 at -0.5, 1 at 0, 0 at 0.5.
    if mean_fn == "sine_quad":
        return x + 0.5 * x*x + np.sin(np.pi * x) / (np.pi ** 2)
    raise ValueError("mean_fn must be paper, quadratic, quad_half, or sine_quad")


def sigma(x: FloatArray) -> FloatArray:
    return 1.0 + 0.4*x


def features(x: FloatArray, z: FloatArray, target: str) -> FloatArray:
    """Use deterministic support scaling, NOT a full-sample estimated transform."""
    if target == "x":
        return np.asarray(x, dtype=float)[:, None]
    if target == "w":
        return np.column_stack((x, z / SQRT3))
    raise ValueError("target must be 'x' or 'w'")


def true_mean(x: FloatArray, z: FloatArray, target: str, beta: float,
              mean_fn: str = "paper") -> FloatArray:
    if target not in {"x", "w"}:
        raise ValueError("unknown target")
    return m0(x, mean_fn) + (beta*z if target == "w" else 0.0)


def stable_rng(seed: int, *keys: Any) -> np.random.Generator:
    """Seeds do not depend on process scheduling, layer order, or batch size."""
    payload = json.dumps(keys, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode()
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    words = np.frombuffer(digest, dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([int(seed), *words]))


@dataclass(frozen=True)
class ErrorLaw:
    error: str
    mixture_p: float = 0.2
    mixture_a: float = 5.0
    error_base: str = "normal"
    emg_lambda: float = EMG_LAMBDA_DEFAULT

    def __post_init__(self) -> None:
        if self.error not in {"symmetric", "right", "left"}:
            raise ValueError("error must be symmetric, right, or left")
        if self.error_base not in ERROR_BASES:
            raise ValueError("error_base must be normal, t4, or exponnorm")
        if self.error_base == "exponnorm" and self.error == "symmetric":
            raise ValueError("exponnorm error_base requires a skewed error")
        if not 0 < self.mixture_p < 1 or self.mixture_a < 0:
            raise ValueError("invalid mixture parameters")
        if self.error_base == "exponnorm":
            lam = float(self.emg_lambda)
            if not math.isfinite(lam) or lam <= 0.0:
                raise ValueError("emg_lambda must be finite and positive")

    @property
    def skew_sign(self) -> float:
        return -1.0 if self.error_base == "exponnorm" and self.error == "left" else 1.0

    def _base_kw(self) -> dict[str, float]:
        return {"emg_lambda": float(self.emg_lambda), "skew_sign": self.skew_sign}

    @property
    def components(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        if self.error == "symmetric" or self.error_base == "exponnorm":
            return np.array([1.0]), np.array([0.0]), np.array([1.0])
        p, a = self.mixture_p, self.mixture_a
        scale = math.sqrt(1 + a*a*p*(1-p))
        means = np.array([-a*p, a*(1-p)]) / scale
        if self.error == "left":
            means = -means
        return np.array([1-p, p]), means, np.full(2, 1/scale)

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        if self.error_base == "exponnorm":
            lam = float(self.emg_lambda)
            scale = math.sqrt(1.0 + lam * lam)
            eps = (rng.normal(size=n) + lam * (rng.exponential(size=n) - 1.0)) / scale
            return -eps if self.error == "left" else eps
        if self.error_base == "normal":
            g = rng.normal(size=n)
        else:
            g = rng.standard_t(4, size=n) / math.sqrt(2.0)
        if self.error == "symmetric":
            return g
        p, a = self.mixture_p, self.mixture_a
        eps = (g + a*(rng.binomial(1, p, size=n)-p)) / math.sqrt(1+a*a*p*(1-p))
        return -eps if self.error == "left" else eps

    def conditional_pdf(self, t: float | FloatArray, x: FloatArray,
                        beta_residual: float) -> FloatArray:
        """PDF of beta_residual*Z + sigma(x)*epsilon, after integrating over Z.

        beta_residual=0 for w; beta_residual=beta for x.
        t and x must broadcast to the same shape. Z is NOT conditioned on here.
        """
        t, x = np.broadcast_arrays(np.asarray(t, float), np.asarray(x, float))
        if beta_residual < 0 or not math.isfinite(beta_residual):
            raise ValueError("beta_residual must be finite and nonnegative")
        probs, means, stds = self.components
        loc = sigma(x)[..., None]*means
        scale = sigma(x)[..., None]*stds
        center = t[..., None] - loc
        a = beta_residual*SQRT3
        if a < 1e-7:
            u = center/scale
            out = np.sum(probs*standardized_pdf(u, self.error_base, **self._base_kw())/scale, axis=-1)
        else:
            lo, hi = (center-a)/scale, (center+a)/scale
            # Survival-function form avoids cancellation for two positive tails.
            diff = np.where(
                lo > 0,
                standardized_sf(lo, self.error_base, **self._base_kw())
                -standardized_sf(hi, self.error_base, **self._base_kw()),
                standardized_cdf(hi, self.error_base, **self._base_kw())
                -standardized_cdf(lo, self.error_base, **self._base_kw()),
            )
            out = np.sum(probs*diff/(2*a), axis=-1)
        return np.maximum(out, 0.0)

    def A(self, q: float | FloatArray, x: FloatArray,
          beta_residual: float) -> FloatArray:
        return (self.conditional_pdf(q, x, beta_residual)
                - self.conditional_pdf(-np.asarray(q), x, beta_residual))

    def conditional_pdf_derivative(self, t: float | FloatArray, x: FloatArray,
                                   beta_residual: float) -> FloatArray:
        """d/dt of the conditional residual PDF, after integrating over Z."""
        t, x = np.broadcast_arrays(np.asarray(t, float), np.asarray(x, float))
        if beta_residual < 0 or not math.isfinite(beta_residual):
            raise ValueError("beta_residual must be finite and nonnegative")
        probs, means, stds = self.components
        loc = sigma(x)[..., None]*means
        scale = sigma(x)[..., None]*stds
        center = t[..., None] - loc
        a = beta_residual*SQRT3
        if a < 1e-7:
            u = center/scale
            out = np.sum(
                probs*standardized_pdf_derivative(u, self.error_base, **self._base_kw())/(scale*scale),
                axis=-1,
            )
        else:
            lo, hi = (center-a)/scale, (center+a)/scale
            out = np.sum(
                probs*(standardized_pdf(hi, self.error_base, **self._base_kw())
                       -standardized_pdf(lo, self.error_base, **self._base_kw()))/(2*a*scale),
                axis=-1,
            )
        return out

    def D(self, q: float | FloatArray, x: FloatArray,
          beta_residual: float) -> FloatArray:
        """f'(ζ|V) - f'(-ζ|V) at the true residual quantile."""
        q = np.asarray(q, float)
        return (self.conditional_pdf_derivative(q, x, beta_residual)
                - self.conditional_pdf_derivative(-q, x, beta_residual))


@dataclass
class Data:
    x: FloatArray
    z: FloatArray
    y: FloatArray
    eps: FloatArray
    folds: tuple[NDArray[np.int64], NDArray[np.int64]]


def sample_covariates(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    if n < 1:
        raise ValueError("covariate sample size must be positive")
    x = rng.uniform(-1, 1, size=n)
    z = rng.uniform(-SQRT3, SQRT3, size=n)
    return x, z


def generate(n: int, beta: float, law: ErrorLaw, rng: np.random.Generator,
             mean_fn: str = "paper") -> Data:
    if n < 40 or n % 2:
        raise ValueError("Use an even n >= 40; equal folds match the manuscript average.")
    x, z = sample_covariates(n, rng)
    eps = law.sample(n, rng)
    y = m0(x, mean_fn) + beta*z + sigma(x)*eps
    perm = rng.permutation(n)
    return Data(x, z, y, eps, (perm[:n//2], perm[n//2:]))


@lru_cache(maxsize=512)
def truth(error: str, alpha: float, target_delta: float, nodes: int = 160,
          mixture_p: float = 0.2, mixture_a: float = 5.0,
          error_base: str = "normal", beta: float | None = None,
          emg_lambda: float = EMG_LAMBDA_DEFAULT) -> dict[str, float]:
    # Reflection plus symmetric Z leaves absolute residual quantiles unchanged.
    calibration_error = "right" if error == "left" else error
    oracle = MarginalOracle(calibration_error, alpha, nodes, mixture_p, mixture_a,
                            error_base, emg_lambda)
    if beta is None:
        beta = oracle.calibrate_beta(target_delta)
    else:
        beta = float(beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("fixed beta must be finite and positive")
    mx, mw = oracle.boundary_metrics(beta), oracle.boundary_metrics(0.0)
    sign = -1.0 if error == "left" else 1.0
    return {"beta": beta, "zeta_x": mx.zeta, "zeta_w": mw.zeta,
            "delta_true": mx.zeta-mw.zeta, "g_x": mx.g, "g_w": mw.g,
            "mean_A_x": sign*mx.mean_A, "mean_A_w": sign*mw.mean_A,
            "kappa_x": sign*mx.kappa, "kappa_w": sign*mw.kappa}


@lru_cache(maxsize=32)
def covariate_quadrature(target: str, nodes: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    a, weights = leggauss(nodes)
    weights = weights / 2
    if target == "x":
        return a, np.zeros_like(a), weights
    xx, uu = np.meshgrid(a, a, indexing="ij")
    ww = np.outer(weights, weights)
    return xx.ravel(), SQRT3*uu.ravel(), ww.ravel()


@lru_cache(maxsize=32)
def diagnostic_grid(target: str, side: int) -> tuple[FloatArray, FloatArray]:
    """Includes the covariate support boundary; finite grid is not a sup-norm proof."""
    u = np.linspace(-1, 1, side)
    if target == "x":
        return u, np.zeros_like(u)
    xx, uu = np.meshgrid(u, u, indexing="ij")
    return xx.ravel(), SQRT3*uu.ravel()
