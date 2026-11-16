"""Training-only kernel estimates of the conditional quantile Jacobians.

``prepare_g_fits`` separates density estimation for inference from the nuisance
fits used by the point estimator.  It can reconstruct the x density's LOO
residuals using a local quadratic mean and mix that density with the original
density.  Neither the point estimator's residuals nor its A correction changes.

The defaults reproduce ``recompute_kernel_g_se.g_from_folds``: evaluate the
positive-part density at both signs of the pooled quantile, integrate full
feature w fits over Z using 48 nodes, and average the two fold densities.
All tuning options affect x only; w retains the original density fit.  The
optional skewness rule uses each training fold's original LOO residuals only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from conditional_mc.oracle import z_nodes
from marginal_mc.nuisance import (
    LearnedPrepared, LocalLinear, _density_features, _loo_residuals,
)


@dataclass(frozen=True)
class KernelGOptions:
    """Parameters for an additional, independent x density fit.

    ``x_density_mix`` is the weight on the alternative *density*, after the
    existing positive-part projection; it never interpolates residuals or
    reciprocals of densities.  Zero gives the original density exactly.
    ``response_bw_multiplier`` and ``x_hf_multiplier`` scale the already
    selected x density bandwidths, without rerunning density CV.  The response
    bandwidth is not rescaled by the new residual standard deviation.
    ``x_mean_degree=None`` preserves the original mean degree, including when
    its bandwidth changes.  Explicit degrees 1 and 2 are supported.
    ``apply_if='right_skew'`` gates all changes separately in each fold using
    the standardized third central moment of its original training residuals.
    """

    x_mean_degree: int | None = None
    x_mean_h_multiplier: float = 1.0
    x_density_mix: float = 1.0
    response_bw_multiplier: float = 1.0
    x_hf_multiplier: float = 1.0
    apply_if: str = "always"
    skew_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (self.x_mean_degree is not None
                and (isinstance(self.x_mean_degree, bool) or self.x_mean_degree not in {1, 2})):
            raise ValueError("x_mean_degree must be None, 1 or 2")
        for name in ("x_mean_h_multiplier", "response_bw_multiplier", "x_hf_multiplier"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        mix = float(self.x_density_mix)
        if not math.isfinite(mix) or not 0 <= mix <= 1:
            raise ValueError("x_density_mix must be between 0 and 1")
        if self.apply_if not in {"always", "right_skew"}:
            raise ValueError("apply_if must be 'always' or 'right_skew'")
        if not math.isfinite(float(self.skew_threshold)):
            raise ValueError("skew_threshold must be finite")


def _skewness(residual: np.ndarray) -> float:
    centered = np.asarray(residual, dtype=float) - np.mean(residual)
    variance = float(np.mean(centered * centered))
    if not math.isfinite(variance):
        raise ValueError("training residuals must be finite")
    if variance == 0:
        return 0.0
    return float(np.mean(centered ** 3) / variance ** 1.5)


def _average_folds(values: list[float]) -> float:
    # Match the legacy nanmean, without warning when both roots are invalid.
    values_array = np.asarray(values, dtype=float)
    return float(np.nanmean(values_array)) if np.any(~np.isnan(values_array)) else math.nan


class PreparedKernelG:
    """Reusable g evaluator, with alternative LOO residuals computed once.

    Original fitted objects and their diagnostic lists are never mutated.
    Density query weights are cached across Semi/Conf quantile evaluations.
    ``diagnostics`` describes training decisions; ``evaluation_diagnostics``
    reports the densities at a particular pair of quantiles.
    """

    def __init__(self, fits: Mapping[str, Any], options: KernelGOptions) -> None:
        if any(target not in fits or len(fits[target]) != 2 for target in ("x", "w")):
            raise ValueError("kernel g requires two training fits for each of x and w")
        self.options = options
        self._fits = {target: tuple(fits[target]) for target in ("x", "w")}
        self._alternatives: dict[int, tuple[np.ndarray, LocalLinear, float, float]] = {}
        self._queries: dict[tuple[str, int, float], tuple[Any, Any, np.ndarray]] = {}
        self.diagnostics: dict[str, float] = {}
        self._z, self._zw = z_nodes(48)
        for fold, fitted in enumerate(self._fits["x"]):
            if not fitted.residual_density or fitted.mean_fit.train.shape[1] != 1:
                raise ValueError("x kernel g requires a one-dimensional residual-density fit")
            original = np.asarray(fitted.density_response, dtype=float)
            skew = _skewness(original)
            triggered = options.apply_if == "always" or skew > float(options.skew_threshold)
            degree = (fitted.mean_smoother.degree if options.x_mean_degree is None
                      else int(options.x_mean_degree))
            mean_changed = (degree != fitted.mean_smoother.degree
                            or float(options.x_mean_h_multiplier) != 1.0)
            density_changed = (float(options.response_bw_multiplier) != 1.0
                               or float(options.x_hf_multiplier) != 1.0)
            applied = bool(triggered and float(options.x_density_mix) > 0
                           and (mean_changed or density_changed))
            prefix = f"x_fold{fold}_"
            hm = float(fitted.hm)
            hf = float(fitted.hf)
            b = float(fitted.b)
            fallback = expanded = pinv = 0.0
            if applied:
                hm *= float(options.x_mean_h_multiplier)
                residual = original
                if mean_changed:
                    old = fitted.mean_smoother
                    mean_smoother = LocalLinear(
                        fitted.mean_fit.train, hm, old.min_neighbors,
                        condition_limit=old.condition_limit, expand=old.expand,
                        degree=degree,
                    )
                    residual, result, fallback = _loo_residuals(
                        mean_smoother, fitted.mean_fit.train, fitted.mean_fit.response,
                    )
                    expanded, pinv = result.expanded_rows, result.pinv_rows
                density_smoother = fitted.density_smoother
                hf *= float(options.x_hf_multiplier)
                if float(options.x_hf_multiplier) != 1.0:
                    old = fitted.density_smoother
                    density_smoother = LocalLinear(
                        fitted.density_train, hf, old.min_neighbors,
                        condition_limit=old.condition_limit, expand=old.expand,
                        degree=old.degree,
                    )
                b *= float(options.response_bw_multiplier)
                self._alternatives[fold] = (residual, density_smoother, b, hf)
            self.diagnostics.update({prefix + name: float(value) for name, value in {
                "training_skew": skew, "triggered": triggered, "applied": applied,
                "mix": float(options.x_density_mix) if applied else 0.0,
                "base_hm": fitted.hm, "density_mean_hm": hm,
                "base_hf": fitted.hf, "density_hf": hf,
                "base_b": fitted.b, "density_b": b,
                "loo_fallback_rows": fallback, "mean_expanded_rows": expanded,
                "mean_pinv_rows": pinv,
            }.items()})

    def _query(self, target: str, fold: int, s0: float):
        key = (target, fold, float(s0))
        if key in self._queries:
            return self._queries[key]
        fitted = self._fits[target][fold]
        mode = getattr(fitted, "density_feature_mode", "all")
        if target == "x" or mode == "x_only":
            x, z, quadrature = np.array([s0]), np.array([0.0]), np.ones(1)
        else:
            x, z, quadrature = np.full(self._z.shape, s0), self._z, self._zw
        density_query = _density_features(x, z, target, mode)
        weights = fitted.density_smoother.weights(density_query).matrix
        # For residual densities the query mean drops out of every offset.
        # Avoid predict_mean(), which appends to the original diagnostics.
        mean = np.zeros(len(x)) if fitted.residual_density else fitted.mean_fit.predict(x, z)[0]
        original = LearnedPrepared(
            mean, weights, fitted.density_response, fitted.b,
            fitted.positive_part, fitted.direct_contrast, fitted.residual_density,
        )
        alternative = None
        if target == "x" and fold in self._alternatives:
            residual, smoother, b, _ = self._alternatives[fold]
            if smoother is not fitted.density_smoother:
                weights = smoother.weights(density_query).matrix
            alternative = LearnedPrepared(
                mean, weights, residual, b, fitted.positive_part,
                fitted.direct_contrast, residual_density=True,
            )
        self._queries[key] = original, alternative, quadrature
        return self._queries[key]

    def evaluate_fold(self, target: str, fold: int, s0: float, q: float) -> float:
        """Evaluate one fold's f(q|s0)+f(-q|s0), with optional density mixing."""
        if target not in {"x", "w"} or fold not in {0, 1}:
            raise ValueError("target must be x or w and fold must be 0 or 1")
        if not math.isfinite(float(q)) or q < 0:
            return math.nan
        if not math.isfinite(float(s0)):
            raise ValueError("s0 must be finite")
        original, alternative, quadrature = self._query(target, fold, float(s0))

        def density(prepared):
            values = np.asarray(prepared.pdf(float(q)), dtype=float)
            values = values + np.asarray(prepared.pdf(float(-q)), dtype=float)
            if values.shape != quadrature.shape or not np.isfinite(values).all():
                return math.nan
            return float(np.sum(quadrature * values))

        if alternative is None:
            return density(original)
        mix = float(self.options.x_density_mix)
        if mix == 1.0:
            return density(alternative)
        return (1.0 - mix) * density(original) + mix * density(alternative)

    def evaluate(self, target: str, s0: float, q: float) -> float:
        """Average the two fold densities before any variance-formula inversion."""
        return _average_folds([self.evaluate_fold(target, j, s0, q) for j in range(2)])

    def evaluation_diagnostics(self, s0: float, q_by_target: Mapping[str, float]) -> dict[str, float]:
        """Return gx/gw and their fold values at the specified pooled roots."""
        result = {}
        for target in ("x", "w"):
            values = [self.evaluate_fold(target, j, s0, q_by_target[target]) for j in range(2)]
            result[f"g{target}"] = _average_folds(values)
            for j, value in enumerate(values):
                result[f"g{target}_fold{j}"] = value
        return result


def validate_options(options: Mapping[str, Any] | KernelGOptions | None = None) -> dict[str, Any]:
    """Validate an option mapping and return normalized JSON-safe defaults.

    Unknown keys are rejected, so a misspelled option cannot silently leave
    the original inference settings in place.
    """
    if options is None:
        checked = KernelGOptions()
    elif isinstance(options, KernelGOptions):
        checked = options
    elif isinstance(options, Mapping):
        checked = KernelGOptions(**dict(options))
    else:
        raise TypeError("kernel g options must be a mapping or KernelGOptions")
    result = asdict(checked)
    if result["x_mean_degree"] is not None:
        result["x_mean_degree"] = int(result["x_mean_degree"])
    for name in ("x_mean_h_multiplier", "x_density_mix", "response_bw_multiplier",
                 "x_hf_multiplier", "skew_threshold"):
        result[name] = float(result[name])
    return result


def prepare_g_fits(fits: Mapping[str, Any], options: Mapping[str, Any] | KernelGOptions | None = None) -> PreparedKernelG:
    """Prepare reusable inference densities from existing two-fold nuisance fits.

    ``fits`` is the ``['fits']`` entry returned by ``cross_fit_nuisance``.
    Options are validated eagerly, including unknown option names.  No true
    parameters, error-law labels, quantiles, or evaluation outcomes enter the
    training decision.  For reproducible metadata use ``asdict(result.options)``.
    """
    return PreparedKernelG(fits, KernelGOptions(**validate_options(options)))
