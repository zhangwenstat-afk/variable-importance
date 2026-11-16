"""Oracle, controlled-rate, and genuinely training-data-only nuisance estimators.

The default learned conditional density is a local-linear regression of
K_b(Y-y) on V, then evaluated at y=m_hat(V)+t.  An optional residual-LOO path
directly smooths training-fold leave-one-out residuals.  Neither path is a
reimplementation of lpcde, and no oracle parameter, evaluation response, or
diagnostic grid is used for fitting.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Protocol, Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.spatial import cKDTree

from .dgp import ErrorLaw, SQRT2PI, features, true_mean

Array = NDArray[np.float64]


class Prepared(Protocol):
    mean: Array
    def pdf(self, t: float | Array) -> Array: ...
    def A(self, q: float) -> Array: ...
    def correction(self, q: float, residual: Array) -> float: ...


class Nuisance(Protocol):
    def predict_mean(self, x: Array, z: Array) -> Array: ...
    def prepare(self, x: Array, z: Array) -> Prepared: ...
    def diagnostics(self) -> dict[str, float]: ...


@dataclass
class AnalyticPrepared:
    mean: Array
    x: Array
    law: ErrorLaw
    beta_residual: float
    contamination: float = 0.0
    dagger_sd: float = 2.0
    a_shift: float = 0.0

    def pdf(self, t: float | Array) -> Array:
        oracle = self.law.conditional_pdf(t, self.x, self.beta_residual)
        dagger = np.exp(-0.5*(np.asarray(t)/self.dagger_sd)**2)/(SQRT2PI*self.dagger_sd)
        return (1-self.contamination)*oracle + self.contamination*dagger

    def A(self, q: float) -> Array:
        # The contaminating N(0, dagger_sd**2) density is symmetric.
        return ((1-self.contamination)*self.law.A(q, self.x, self.beta_residual)
                + self.a_shift)

    def correction(self, q: float, residual: Array) -> float:
        return float(np.mean(residual*self.A(q)))


@dataclass
class AnalyticNuisance:
    target: str
    beta: float
    law: ErrorLaw
    mean_shift: float = 0.0
    contamination: float = 0.0
    dagger_sd: float = 2.0
    mean_fn: str = "paper"
    a_shift: float = 0.0

    def predict_mean(self, x: Array, z: Array) -> Array:
        return true_mean(x, z, self.target, self.beta, self.mean_fn) + self.mean_shift

    def prepare(self, x: Array, z: Array) -> AnalyticPrepared:
        return AnalyticPrepared(self.predict_mean(x, z), x, self.law,
                                self.beta if self.target == "x" else 0.0,
                                self.contamination, self.dagger_sd, self.a_shift)

    def diagnostics(self) -> dict[str, float]:
        return {"mean_shift": self.mean_shift, "contamination": self.contamination,
                "a_shift": self.a_shift,
                "ll_expanded_rows": 0.0, "ll_pinv_rows": 0.0}


@dataclass
class WeightResult:
    matrix: csr_array
    expanded_rows: int
    pinv_rows: int
    min_neighbors: int
    max_condition: float


def _bandwidth_vector(bandwidth: float | Array, d: int) -> Array:
    h = np.asarray(bandwidth, dtype=float).reshape(-1)
    if h.size == 1:
        h = np.full(d, float(h[0]), dtype=float)
    if h.size != d or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("bandwidth must be a finite positive scalar or length-d vector")
    return h


def _poly_n_params(d: int, degree: int) -> int:
    if degree not in {1, 2, 3}:
        raise ValueError("local polynomial degree must be 1, 2, or 3")
    n_params = 1
    for k in range(1, degree + 1):
        n_params = n_params * (d + k) // k
    return n_params


def _poly_design(u: Array, degree: int) -> Array:
    """Columns are monomials of u up to the given total degree."""
    n, d = u.shape if u.ndim == 2 else (0, 0)
    if n == 0:
        return np.empty((0, _poly_n_params(d, degree)))
    cols = [np.ones(n), u]
    if degree >= 2:
        quads = [u[:, i]*u[:, j] for i in range(d) for j in range(i, d)]
        cols.append(np.column_stack(quads))
    if degree >= 3:
        cubics = [u[:, i]*u[:, j]*u[:, k]
                  for i in range(d) for j in range(i, d) for k in range(j, d)]
        cols.append(np.column_stack(cubics))
    return np.column_stack(cols)


class LocalLinear:
    """Boundary-adaptive local-polynomial equivalent weights with a product kernel.

    Degree 1/2/3 are local linear/quadratic/cubic. In normalized support
    coordinates, K(u)=prod(0.75*(1-u_k**2)_+). Bandwidth may be a scalar or a
    per-coordinate vector. If expand=True, a sparse or singular row enlarges
    the radius; if expand=False, the window stays fixed.
    """
    def __init__(self, train: Array, bandwidth: float | Array,
                 min_neighbors: int = 12, condition_limit: float = 1e10,
                 expand: bool = True, degree: int = 1) -> None:
        self.train = np.asarray(train, dtype=float)
        if self.train.ndim != 2 or not np.isfinite(self.train).all():
            raise ValueError("train must be a finite two-dimensional array")
        if degree not in {1, 2, 3}:
            raise ValueError("local polynomial degree must be 1, 2, or 3")
        d = self.train.shape[1]
        self.degree = int(degree)
        self.n_params = _poly_n_params(d, self.degree)
        self.h = _bandwidth_vector(bandwidth, d)
        self.inv_h = 1.0 / self.h
        self.scaled = self.train * self.inv_h
        self.tree = cKDTree(self.scaled)
        self.min_neighbors = min(max(min_neighbors, self.n_params+3), len(train))
        self.condition_limit = condition_limit
        self.expand = bool(expand)

    def weights(self, query: Array) -> WeightResult:
        query = np.atleast_2d(np.asarray(query, float))
        if query.shape[1] != self.train.shape[1]:
            raise ValueError("query dimension mismatch")
        qscaled = query * self.inv_h
        neighborhoods = self.tree.query_ball_point(qscaled, 1.0, p=np.inf)
        vals, cols = [], []
        indptr = [0]
        expanded, pinv, min_count, max_cond = 0, 0, len(self.train), 0.0
        d = self.train.shape[1]
        p = self.n_params
        e0 = np.zeros(p)
        e0[0] = 1.0
        attempts = 14 if self.expand else 1
        for row, qs in enumerate(qscaled):
            radius = 1.0
            idx = np.asarray(neighborhoods[row], dtype=np.int64)
            did_expand = False
            coefficients = None
            kernel = np.empty(0)
            design = np.empty((0, p))
            used_idx = idx
            cond = math.inf
            for _ in range(attempts):
                if self.expand and len(idx) < self.min_neighbors:
                    distances, _ = self.tree.query(qs, k=self.min_neighbors, p=np.inf)
                    radius = max(radius*1.5, float(np.max(np.atleast_1d(distances)))*1.05+1e-12)
                    idx = np.asarray(self.tree.query_ball_point(qs, radius, p=np.inf), dtype=np.int64)
                    did_expand = True
                u = (self.scaled[idx]-qs)/radius if len(idx) else np.empty((0, d))
                kernel = np.prod(0.75*np.maximum(1-u*u, 0.0), axis=1) if len(idx) else np.empty(0)
                keep = kernel > 0
                idx, u, kernel = idx[keep], u[keep], kernel[keep]
                used_idx = idx.copy()
                design = _poly_design(u, self.degree)
                moment = design.T @ (kernel[:, None]*design) if len(idx) else np.eye(p)
                cond = float(np.linalg.cond(moment)) if len(idx) else math.inf
                if len(idx) >= p+1 and np.isfinite(cond) and cond < self.condition_limit:
                    coefficients = np.linalg.solve(moment, e0)
                    break
                if not self.expand:
                    break
                radius *= 1.5
                idx = np.asarray(self.tree.query_ball_point(qs, radius, p=np.inf), dtype=np.int64)
                did_expand = True
            if coefficients is None:
                if len(used_idx) == 0:
                    _, nn = self.tree.query(qs, k=1, p=np.inf)
                    idx = np.asarray([int(np.atleast_1d(nn)[0])], dtype=np.int64)
                    equivalent = np.ones(1)
                    pinv += 1
                    cond = math.inf
                else:
                    idx = used_idx
                    coefficients = np.linalg.pinv(moment, rcond=1e-12) @ e0
                    equivalent = kernel*(design @ coefficients)
                    pinv += 1
            else:
                equivalent = kernel*(design @ coefficients)
            total = float(equivalent.sum())
            if not np.isfinite(equivalent).all() or abs(total) < 1e-10:
                raise FloatingPointError("local-linear weights could not be constructed")
            equivalent /= total  # remove only floating-point normalization error
            vals.extend(equivalent.tolist())
            cols.extend(idx.tolist())
            indptr.append(len(vals))
            expanded += int(did_expand)
            min_count = min(min_count, len(idx))
            max_cond = max(max_cond, cond)
        matrix = csr_array((np.asarray(vals), np.asarray(cols, dtype=np.int64),
                            np.asarray(indptr, dtype=np.int64)),
                           shape=(len(query), len(self.train)))
        return WeightResult(matrix, expanded, pinv, min_count, max_cond)


def _inner_cv_splits(n: int, folds: int, max_points: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Deterministic training-only inner folds and bounded validation queries."""
    if n < 2:
        raise ValueError("inner CV requires at least two training observations")
    folds = min(max(int(folds), 2), n)
    all_idx = np.arange(n, dtype=np.int64)
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for fold in range(folds):
        validation_all = all_idx[fold::folds]
        if len(validation_all) == 0:
            continue
        validation = validation_all
        if max_points > 0 and len(validation) > max_points:
            positions = np.linspace(0, len(validation)-1, max_points, dtype=int)
            validation = validation[positions]
        training = np.setdiff1d(all_idx, validation_all, assume_unique=True)
        splits.append((training, validation))
    if len(splits) < 2:
        raise ValueError("inner CV requires at least two nonempty folds")
    return splits


def _multipliers(settings: dict[str, Any], key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(value) for value in settings.get(key, default))
    if not values or not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise ValueError(f"{key} must contain finite positive values")
    return values


def _mean_h(base_bandwidth: float, multiplier: float, d: int, z_scale: float) -> float | Array:
    hx = base_bandwidth*multiplier
    if d == 1:
        return hx
    return np.array([hx, hx*z_scale], dtype=float)


def _cv_mean_bandwidth(
    train: Array,
    y: Array,
    base_bandwidth: float,
    min_neighbors: int,
    settings: dict[str, Any],
    expand: bool = True,
    z_scale: float = 1.0,
    degree: int = 1,
    search_ratio: bool = False,
) -> tuple[float, float, float, float]:
    """Select m bandwidth using only the current outer training fold.

    For a 2D mean, anisotropic CV jointly searches hx = c * b_N and
    hu = r * hx. The selected ratio r is returned as z_scale.
    """
    splits = _inner_cv_splits(
        len(train),
        int(settings.get("folds", 3)),
        int(settings.get("max_points", 512)),
    )
    multipliers = _multipliers(
        settings,
        "m_multipliers",
        (0.5, 0.75, 1.0, 1.25, 1.5),
    )
    d = train.shape[1]
    if search_ratio and d > 1 and "z_scales" in settings:
        z_scales = _multipliers(
            settings,
            "z_scales",
            (1.0, 1.5, 2.0, 3.0, 4.0),
        )
    else:
        z_scales = (1.0 if d == 1 else float(z_scale),)
    cv_center = settings.get("cv_center")
    local = cv_center is not None and math.isfinite(float(cv_center))
    cv_radius = float(settings.get("cv_radius", 0.25)) if local else math.nan
    if local and (not math.isfinite(cv_radius) or cv_radius <= 0):
        raise ValueError("cv_radius must be finite and positive")
    scores: list[tuple[float, float, float]] = []
    for multiplier in multipliers:
        for ratio in z_scales:
            fold_scores: list[float] = []
            h = _mean_h(base_bandwidth, multiplier, d, ratio)
            for fit_idx, validation_idx in splits:
                smoother = LocalLinear(
                    train[fit_idx],
                    h,
                    min_neighbors,
                    expand=expand,
                    degree=degree,
                )
                weights = smoother.weights(train[validation_idx])
                fitted = np.asarray(weights.matrix @ y[fit_idx]).ravel()
                sq = (y[validation_idx]-fitted)**2
                if local:
                    u = (train[validation_idx, 0] - float(cv_center)) / cv_radius
                    mask = np.abs(u) <= 1.0
                    ww = np.zeros(len(u), dtype=float)
                    ww[mask] = 0.75 * (1.0 - u[mask] * u[mask])
                    total = float(np.sum(ww))
                    fold_scores.append(
                        float(np.dot(ww, sq) / total) if total > 0 else float(np.mean(sq))
                    )
                else:
                    fold_scores.append(float(np.mean(sq)))
            scores.append((float(np.mean(fold_scores)), multiplier, ratio))
    best_score, best_multiplier, best_ratio = min(scores, key=lambda item: item[0])
    return (
        base_bandwidth*best_multiplier,
        best_multiplier,
        best_score,
        best_ratio,
    )


def _conditional_response_density(
    weights: WeightResult,
    y_train: Array,
    y_query: Array,
    response_bandwidth: float,
) -> Array:
    """Evaluate a local-linear conditional response density on query points."""
    y_query = np.asarray(y_query, dtype=float)
    out = np.empty(len(y_query), dtype=float)
    for row in range(len(y_query)):
        start, stop = weights.matrix.indptr[row], weights.matrix.indptr[row+1]
        indices = weights.matrix.indices[start:stop]
        coefficients = weights.matrix.data[start:stop]
        u = (y_query[row]-y_train[indices])/response_bandwidth
        out[row] = float(np.sum(
            coefficients*np.exp(-0.5*u*u)/(SQRT2PI*response_bandwidth)
        ))
    return out


def _cv_density_bandwidth(
    train: Array,
    y: Array,
    base_bandwidth: float,
    response_scale: float,
    min_neighbors: int,
    settings: dict[str, Any],
) -> tuple[float, float, float, float, float, float]:
    """Select covariate and response bandwidths by training-only log likelihood."""
    splits = _inner_cv_splits(
        len(train),
        int(settings.get("folds", 3)),
        int(settings.get("max_points", 512)),
    )
    f_multipliers = _multipliers(
        settings,
        "f_multipliers",
        (0.75, 1.0, 1.25),
    )
    y_multipliers = _multipliers(
        settings,
        "y_multipliers",
        (0.75, 1.0, 1.25),
    )
    scores: list[tuple[float, float, float, float]] = []
    for f_multiplier in f_multipliers:
        for y_multiplier in y_multipliers:
            response_bandwidth = (
                y_multiplier*response_scale*base_bandwidth
            )
            fold_scores: list[float] = []
            invalid_fractions: list[float] = []
            for fit_idx, validation_idx in splits:
                smoother = LocalLinear(
                    train[fit_idx],
                    base_bandwidth*f_multiplier,
                    min_neighbors,
                )
                weights = smoother.weights(train[validation_idx])
                density = _conditional_response_density(
                    weights,
                    y[fit_idx],
                    y[validation_idx],
                    response_bandwidth,
                )
                invalid = (~np.isfinite(density)) | (density <= 1e-12)
                invalid_fractions.append(float(np.mean(invalid)))
                fold_scores.append(float(-np.mean(np.log(np.maximum(density, 1e-12)))))
            scores.append((
                float(np.mean(fold_scores)),
                f_multiplier,
                y_multiplier,
                float(np.mean(invalid_fractions)),
            ))
    best = min(scores, key=lambda item: item[0])
    score, f_multiplier, y_multiplier, invalid_fraction = best
    return (
        base_bandwidth*f_multiplier,
        y_multiplier*response_scale*base_bandwidth,
        f_multiplier,
        y_multiplier,
        score,
        invalid_fraction,
    )


def _density_features(
    x: Array,
    z: Array,
    target: str,
    mode: str,
) -> Array:
    """Return the covariates used by the conditional-density smoother."""
    if mode == "all":
        return features(x, z, target)
    if mode == "x_only":
        return np.asarray(x, dtype=float)[:, None]
    raise ValueError("density_feature_mode must be 'all' or 'x_only'")


def _loo_residuals(
    smoother: LocalLinear,
    train: Array,
    y: Array,
) -> tuple[Array, WeightResult, float]:
    """Compute training-only LOO residuals for a linear smoother.

    The exact LOO identity is used whenever the diagonal is numerically
    stable.  A small internal cross-fit fallback avoids dividing by a nearly
    interpolating diagonal on exceptional rows.
    """
    weights = smoother.weights(train)
    fitted = np.asarray(weights.matrix @ y).ravel()
    diagonal = np.asarray(weights.matrix.diagonal(), dtype=float).ravel()
    safe = np.isfinite(diagonal) & (np.abs(1.0-diagonal) > 0.05)
    if len(diagonal) == len(y) and np.all(safe):
        residual = (y-fitted)/(1.0-diagonal)
        if np.isfinite(residual).all():
            return residual, weights, 0.0

    folds = min(3, len(y))
    residual = np.empty(len(y), dtype=float)
    all_idx = np.arange(len(y), dtype=np.int64)
    for fold in range(folds):
        validation = all_idx[fold::folds]
        fitting = np.setdiff1d(all_idx, validation, assume_unique=True)
        local = LocalLinear(
            train[fitting], smoother.h, smoother.min_neighbors,
            expand=smoother.expand, degree=smoother.degree,
        )
        validation_weights = local.weights(train[validation])
        residual[validation] = (
            y[validation]
            - np.asarray(validation_weights.matrix @ y[fitting]).ravel()
        )
    if not np.isfinite(residual).all():
        raise FloatingPointError("LOO residuals are not finite")
    return residual, weights, float(np.sum(~safe))


class LearnedPrepared:
    def __init__(self, mean: Array, weights: csr_array, y_train: Array,
                 response_bandwidth: float, positive_part: bool,
                 direct_contrast: bool = False,
                 residual_density: bool = False) -> None:
        self.mean = np.asarray(mean)
        self.weights = weights
        self.rows = np.repeat(np.arange(weights.shape[0]), np.diff(weights.indptr))
        if residual_density:
            self.offset = np.asarray(y_train, dtype=float)[weights.indices]
        else:
            self.offset = (
                np.asarray(y_train, dtype=float)[weights.indices]
                - self.mean[self.rows]
            )
        self.b = response_bandwidth
        self.positive_part = positive_part
        self.direct_contrast = direct_contrast
        self.pdf_calls = 0
        self.negative_count = 0
        self.total_count = 0

    def raw_pdf(self, t: float | Array) -> Array:
        value = np.asarray(t)
        at_edges = value if value.ndim == 0 else value[self.rows]
        u = (at_edges-self.offset)/self.b
        edge_pdf = self.weights.data*np.exp(-0.5*u*u)/(SQRT2PI*self.b)
        return np.bincount(self.rows, weights=edge_pdf, minlength=len(self.mean))

    def pdf(self, t: float | Array) -> Array:
        out = self.raw_pdf(t)
        self.pdf_calls += 1
        self.negative_count += int(np.sum(out < 0))
        self.total_count += len(out)
        return np.maximum(out, 0.0) if self.positive_part else out

    def A(self, q: float) -> Array:
        if self.direct_contrast:
            # A(q) is signed, so estimate the paired contrast before
            # applying the optional nonnegative density projection.
            left = self.raw_pdf(q)
            right = self.raw_pdf(-q)
            self.pdf_calls += 2
            self.negative_count += int(np.sum(left < 0) + np.sum(right < 0))
            self.total_count += 2*len(left)
            return left-right
        return self.pdf(q)-self.pdf(-q)

    def correction(self, q: float, residual: Array) -> float:
        return float(np.mean(residual*self.A(q)))


@dataclass
class MeanFit:
    smoother: LocalLinear
    train: Array
    response: Array
    hm: float
    hz: float
    z_scale: float
    cv_m_multiplier: float
    cv_m_score: float
    beta: float
    anisotropic: float
    additive: float
    expand: bool
    target: str
    poly_degree: float
    undersmooth: float

    def predict(self, x: Array, z: Array) -> tuple[Array, WeightResult]:
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        if self.additive:
            wr = self.smoother.weights(x[:, None])
            mean = np.asarray(wr.matrix @ self.response).ravel() + self.beta*z
            return mean, wr
        wr = self.smoother.weights(features(x, z, self.target))
        return np.asarray(wr.matrix @ self.response).ravel(), wr


def _fit_mean_regression(
    x: Array,
    z: Array,
    y: Array,
    target: str,
    settings: dict[str, Any],
    selection: dict[str, Any],
    bandwidth_cv: bool,
) -> MeanFit:
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    min_neighbors = int(settings.get("min_neighbors", 12))
    expand = bool(settings.get("expand_radius", True))
    degree = int(settings.get("mean_degree", 1))
    if degree not in {1, 2, 3}:
        raise ValueError("mean_degree must be 1, 2, or 3")
    undersmooth = float(settings.get("undersmooth", 1.0))
    if not math.isfinite(undersmooth) or undersmooth <= 0:
        raise ValueError("undersmooth must be finite and positive")
    mean_model = settings.get("mean_model", "local_linear")
    if mean_model not in {"local_linear", "additive"}:
        raise ValueError("mean_model must be local_linear or additive")
    bandwidth_mode = settings.get("mean_bandwidth", "isotropic")
    if bandwidth_mode not in {"isotropic", "anisotropic"}:
        raise ValueError("mean_bandwidth must be isotropic or anisotropic")
    additive = mean_model == "additive" and target == "w"
    beta = 0.0
    if additive:
        zc = z - float(z.mean())
        denom = float(np.dot(zc, zc))
        beta = float(np.dot(zc, y-float(y.mean()))/denom) if denom > 1e-14 else 0.0
        train = x[:, None]
        response = y - beta*z
    else:
        train = features(x, z, target)
        response = y
    d = train.shape[1]
    anisotropic = bandwidth_mode == "anisotropic" and d > 1
    logn_over_n = math.log(max(n, 3))/n
    # Keep the usual d-dimensional local-polynomial rate even when the
    # window is allowed to be anisotropic: for 2D local linear this is
    # (log n / n)^{1/6}, not a 1D-style 1/5 exponent.
    base_hm = logn_over_n**(1/(2*degree+2+d))
    if anisotropic:
        z_scale = float(settings.get("z_scale", 3.0))
        if not math.isfinite(z_scale) or z_scale <= 0:
            raise ValueError("z_scale must be finite and positive")
    else:
        z_scale = 1.0
    cv_m_multiplier = 1.0
    cv_m_score = math.nan
    if bandwidth_cv:
        hm, cv_m_multiplier, cv_m_score, z_scale = _cv_mean_bandwidth(
            train, response, base_hm, min_neighbors, selection,
            expand=expand, z_scale=z_scale, degree=degree,
            search_ratio=anisotropic,
        )
    else:
        hm = float(settings.get("m_bw_mult", 0.8))*base_hm
    hm = hm*undersmooth
    smoother = LocalLinear(
        train, _mean_h(hm, 1.0, d, z_scale), min_neighbors,
        expand=expand, degree=degree,
    )
    hz = float(smoother.h[-1]) if d > 1 else math.nan
    return MeanFit(
        smoother, train, response, float(hm), hz, z_scale,
        cv_m_multiplier, cv_m_score, beta, float(anisotropic),
        float(additive), expand, target, float(degree), undersmooth,
    )


class LearnedNuisance:
    """Training-only local-linear nuisance estimates.

    Response-bandwidth scale comes from training-fold leave-one-out residuals.
    The response-density path uses a bounded pilot, while the optional
    residual-density path uses the full LOO residual vector and can select a
    lower-dimensional density feature set by inner cross-validation.  It never
    uses true residuals or validation Y.
    """
    def __init__(self, x: Array, z: Array, y: Array, target: str,
                 settings: dict[str, Any]) -> None:
        self.target = target
        self.x = np.asarray(x, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.y = np.asarray(y, float)
        self.train = features(self.x, self.z, target)
        n, d = self.train.shape
        base = math.log(max(n, 3))/n
        self.positive_part = bool(settings.get("density_positive_part", True))
        self.direct_contrast = bool(settings.get("direct_contrast", False))
        self.density_estimator = settings.get("density_estimator", "response")
        if self.density_estimator not in {"response", "residual_loo"}:
            raise ValueError("density_estimator must be response or residual_loo")
        self.density_feature_mode = settings.get("density_feature_mode", "all")
        if self.density_feature_mode not in {"all", "x_only"}:
            raise ValueError("density_feature_mode must be all or x_only")
        self.density_feature_selection = settings.get(
            "density_feature_selection", "fixed"
        )
        if self.density_feature_selection not in {"fixed", "inner_cv"}:
            raise ValueError(
                "density_feature_selection must be fixed or inner_cv"
            )
        if (self.density_estimator == "response"
                and self.density_feature_mode != "all"):
            raise ValueError(
                "response density estimation requires density_feature_mode=all"
            )
        if (self.density_feature_selection == "inner_cv"
                and self.density_estimator != "residual_loo"):
            raise ValueError(
                "density feature selection requires density_estimator=residual_loo"
            )
        min_neighbors = int(settings.get("min_neighbors", 12))
        selection = settings.get("bandwidth_selection", {})
        if not isinstance(selection, dict):
            raise ValueError("bandwidth_selection must be an object")
        self.bandwidth_cv = selection.get("method", "rule") == "inner_cv"
        if (self.density_feature_selection == "inner_cv"
                and not self.bandwidth_cv):
            raise ValueError(
                "density feature selection requires inner_cv bandwidth selection"
            )
        self.cv_m_multiplier = 1.0
        self.cv_f_multiplier = 1.0
        self.cv_y_multiplier = 1.0
        self.cv_m_score = math.nan
        self.cv_density_score = math.nan
        self.cv_density_invalid_fraction = math.nan
        self.cv_density_feature_score = math.nan
        self.cv_inner_folds = 0.0
        self.cv_max_points = 0.0
        self.density_loo_fallback = 0.0
        if self.bandwidth_cv:
            self.cv_inner_folds = float(selection.get("folds", 3))
            self.cv_max_points = float(selection.get("max_points", 512))
        self.mean_fit = _fit_mean_regression(
            self.x, self.z, self.y, target, settings, selection, self.bandwidth_cv,
        )
        self.hm = self.mean_fit.hm
        self.cv_m_multiplier = self.mean_fit.cv_m_multiplier
        self.cv_m_score = self.mean_fit.cv_m_score
        self.mean_smoother = self.mean_fit.smoother
        self.weight_stats: list[WeightResult] = []
        self.density_response = self.y
        self.residual_density = self.density_estimator == "residual_loo"
        if self.residual_density:
            residual, full_weights, fallback_rows = _loo_residuals(
                self.mean_smoother, self.mean_fit.train, self.mean_fit.response
            )
            self.weight_stats.append(full_weights)
            self.density_response = residual
            self.density_loo_fallback = fallback_rows
            scale = float(np.std(residual, ddof=1))
        else:
            n = len(self.y)
            pilot_size = min(
                n,
                max(
                    int(settings.get("bandwidth_pilot_size", 256)),
                    int(math.ceil(math.sqrt(n))),
                ),
            )
            pilot_idx = np.unique(np.linspace(0, n-1, pilot_size, dtype=int))
            pilot_weights = self.mean_smoother.weights(self.mean_fit.train[pilot_idx])
            self.weight_stats.append(pilot_weights)
            fitted = np.asarray(pilot_weights.matrix @ self.mean_fit.response).ravel()
            # The corresponding diagonal element of each pilot row.
            diagonal = np.array([
                pilot_weights.matrix[i:i+1, int(j):int(j)+1].toarray()[0, 0]
                for i, j in enumerate(pilot_idx)
            ])
            safe = np.abs(1-diagonal) > 0.05
            loo_residual = (
                (self.mean_fit.response[pilot_idx][safe]-fitted[safe])/(1-diagonal[safe])
            )
            if len(loo_residual) < 8:
                raise FloatingPointError(
                    "not enough stable training LOO pilot residuals"
                )
            # The pilot grows at least as sqrt(n); the scale is bounded in
            # probability.  LOO holds the realized local bandwidth fixed.
            scale = float(np.std(loo_residual, ddof=1))
        if not math.isfinite(scale) or scale <= 1e-12:
            raise FloatingPointError("degenerate response bandwidth scale")
        self.response_scale = scale

        candidate_modes = [self.density_feature_mode]
        if self.density_feature_selection == "inner_cv":
            candidate_modes = list(settings.get(
                "density_feature_candidates", ["all", "x_only"]
            ))
            if self.target != "w":
                candidate_modes = ["all"]
            if not candidate_modes or any(
                mode not in {"all", "x_only"} for mode in candidate_modes
            ):
                raise ValueError(
                    "density_feature_candidates must contain all or x_only"
                )
        candidate_fits: list[tuple[str, Array, tuple[float, ...]]] = []
        for mode in candidate_modes:
            density_train = _density_features(
                self.x, self.z, self.target, mode
            )
            density_dim = density_train.shape[1]
            base_hf = base**(1/(5+density_dim))
            if self.bandwidth_cv:
                cv_settings = selection
                if self.target == "x" and "f_multipliers_x" in selection:
                    cv_settings = dict(selection)
                    cv_settings["f_multipliers"] = selection["f_multipliers_x"]
                fit = _cv_density_bandwidth(
                    density_train,
                    self.density_response,
                    base_hf,
                    scale,
                    min_neighbors,
                    cv_settings,
                )
            else:
                hf = float(settings.get("f_bw_mult", 0.65))*base_hf
                b = float(settings.get("y_bw_mult", 0.8))*scale*base_hf
                fit = (
                    hf, b, float(settings.get("f_bw_mult", 0.65)),
                    float(settings.get("y_bw_mult", 0.8)), math.nan, math.nan,
                )
            candidate_fits.append((mode, density_train, fit))
        if self.density_feature_selection == "inner_cv":
            selected_mode, self.density_train, selected_fit = min(
                candidate_fits, key=lambda item: item[2][4]
            )
        else:
            selected_mode, self.density_train, selected_fit = candidate_fits[0]
        (
            self.hf,
            self.b,
            self.cv_f_multiplier,
            self.cv_y_multiplier,
            self.cv_density_score,
            self.cv_density_invalid_fraction,
        ) = selected_fit
        self.h_f_cv = self.hf
        self.hf_post_multiplier = 1.0
        if self.target == "w" and "w_hf_post_multiplier" in settings:
            post = float(settings["w_hf_post_multiplier"])
            if not math.isfinite(post) or post <= 0:
                raise ValueError("w_hf_post_multiplier must be positive")
            self.hf *= post
            self.hf_post_multiplier = post
        if self.target == "x" and "x_hf_post_multiplier" in settings:
            post = float(settings["x_hf_post_multiplier"])
            if not math.isfinite(post) or post <= 0:
                raise ValueError("x_hf_post_multiplier must be positive")
            self.hf *= post
            self.hf_post_multiplier = post
        self.density_feature_mode = selected_mode
        self.density_feature_dim = self.density_train.shape[1]
        self.density_feature_x_only = float(selected_mode == "x_only")
        self.density_candidate_count = float(len(candidate_fits))
        self.cv_density_feature_score = self.cv_density_score
        self.density_smoother = LocalLinear(
            self.density_train, self.hf, min_neighbors
        )
        if self.hm <= 0 or self.hf <= 0 or self.b <= 0:
            raise ValueError("bandwidth multipliers must be positive")

    def predict_mean(self, x: Array, z: Array) -> Array:
        mean, wr = self.mean_fit.predict(x, z)
        self.weight_stats.append(wr)
        return mean

    def prepare(self, x: Array, z: Array) -> LearnedPrepared:
        mean = self.predict_mean(x, z)
        density_query = _density_features(
            x, z, self.target, self.density_feature_mode
        )
        wr = self.density_smoother.weights(density_query)
        self.weight_stats.append(wr)
        return LearnedPrepared(
            mean,
            wr.matrix,
            self.density_response,
            self.b,
            self.positive_part,
            self.direct_contrast,
            self.residual_density,
        )

    def diagnostics(self) -> dict[str, float]:
        return {"h_m": self.hm, "h_f": self.hf, "b_y": self.b,
                "h_m_z": self.mean_fit.hz,
                "z_scale": self.mean_fit.z_scale,
                "mean_anisotropic": self.mean_fit.anisotropic,
                "mean_additive": self.mean_fit.additive,
                "expand_radius": float(self.mean_fit.expand),
                "additive_beta": self.mean_fit.beta,
                "poly_degree": self.mean_fit.poly_degree,
                "undersmooth": self.mean_fit.undersmooth,
                "feature_dim": float(self.train.shape[1]),
                "density_feature_dim": float(self.density_feature_dim),
                "density_feature_x_only": self.density_feature_x_only,
                "density_residual_loo": float(self.residual_density),
                "density_feature_selection_cv": float(
                    self.density_feature_selection == "inner_cv"
                ),
                "density_candidate_count": self.density_candidate_count,
                "density_loo_fallback_rows": self.density_loo_fallback,
                "bandwidth_cv": float(self.bandwidth_cv),
                "direct_contrast": float(self.direct_contrast),
                "cv_m_multiplier": self.cv_m_multiplier,
                "cv_f_multiplier": self.cv_f_multiplier,
                "cv_y_multiplier": self.cv_y_multiplier,
                "h_f_cv": self.h_f_cv,
                "hf_post_multiplier": self.hf_post_multiplier,
                "cv_m_score": self.cv_m_score,
                "cv_density_score": self.cv_density_score,
                "cv_density_feature_score": self.cv_density_feature_score,
                "cv_density_invalid_fraction": self.cv_density_invalid_fraction,
                "cv_inner_folds": self.cv_inner_folds,
                "cv_max_points": self.cv_max_points,
                "response_scale": self.response_scale,
                "ll_expanded_rows": float(sum(s.expanded_rows for s in self.weight_stats)),
                "ll_pinv_rows": float(sum(s.pinv_rows for s in self.weight_stats)),
                "ll_min_neighbors": float(min(s.min_neighbors for s in self.weight_stats)),
                "ll_max_condition": float(max(s.max_condition for s in self.weight_stats))}


class LearnedMeanOracleDensityNuisance:
    """Learn the regression mean while using the analytic conditional density.

    This diagnostic layer keeps the training-only mean fit, including its
    bandwidth selection, but replaces the response-density nuisance and its
    A-correction with the known DGP truth.  It is intended to separate mean
    regression error from conditional-density estimation error.
    """
    def __init__(
        self,
        x: Array,
        z: Array,
        y: Array,
        target: str,
        settings: dict[str, Any],
        beta: float,
        law: ErrorLaw,
    ) -> None:
        self.target = target
        self.y = np.asarray(y, float)
        self.train = features(x, z, target)
        self.beta = float(beta)
        self.law = law
        selection = settings.get("bandwidth_selection", {})
        if not isinstance(selection, dict):
            raise ValueError("bandwidth_selection must be an object")
        self.bandwidth_cv = selection.get("method", "rule") == "inner_cv"
        self.cv_inner_folds = float(selection.get("folds", 3)) if self.bandwidth_cv else 0.0
        self.cv_max_points = float(selection.get("max_points", 512)) if self.bandwidth_cv else 0.0
        self.mean_fit = _fit_mean_regression(
            x, z, y, target, settings, selection, self.bandwidth_cv,
        )
        self.hm = self.mean_fit.hm
        self.cv_m_multiplier = self.mean_fit.cv_m_multiplier
        self.cv_m_score = self.mean_fit.cv_m_score
        self.mean_smoother = self.mean_fit.smoother
        self.weight_stats: list[WeightResult] = []
        n = len(self.y)
        pilot_size = min(n, max(int(settings.get("bandwidth_pilot_size", 256)),
                                int(math.ceil(math.sqrt(n)))))
        pilot_idx = np.unique(np.linspace(0, n-1, pilot_size, dtype=int))
        self.weight_stats.append(self.mean_smoother.weights(self.mean_fit.train[pilot_idx]))

    def predict_mean(self, x: Array, z: Array) -> Array:
        mean, wr = self.mean_fit.predict(x, z)
        self.weight_stats.append(wr)
        return mean

    def prepare(self, x: Array, z: Array) -> AnalyticPrepared:
        mean = self.predict_mean(x, z)
        beta_residual = self.beta if self.target == "x" else 0.0
        return AnalyticPrepared(mean, np.asarray(x, float), self.law, beta_residual)

    def diagnostics(self) -> dict[str, float]:
        return {
            "h_m": self.hm,
            "h_f": math.nan,
            "b_y": math.nan,
            "h_m_z": self.mean_fit.hz,
            "z_scale": self.mean_fit.z_scale,
            "mean_anisotropic": self.mean_fit.anisotropic,
            "mean_additive": self.mean_fit.additive,
            "expand_radius": float(self.mean_fit.expand),
            "additive_beta": self.mean_fit.beta,
            "poly_degree": self.mean_fit.poly_degree,
            "undersmooth": self.mean_fit.undersmooth,
            "feature_dim": float(self.train.shape[1]),
            "bandwidth_cv": float(self.bandwidth_cv),
            "cv_m_multiplier": self.cv_m_multiplier,
            "cv_f_multiplier": math.nan,
            "cv_y_multiplier": math.nan,
            "cv_m_score": self.cv_m_score,
            "cv_density_score": math.nan,
            "cv_density_invalid_fraction": math.nan,
            "cv_inner_folds": self.cv_inner_folds,
            "cv_max_points": self.cv_max_points,
            "response_scale": math.nan,
            "density_oracle": 1.0,
            "ll_expanded_rows": float(sum(s.expanded_rows for s in self.weight_stats)),
            "ll_pinv_rows": float(sum(s.pinv_rows for s in self.weight_stats)),
            "ll_min_neighbors": float(min(s.min_neighbors for s in self.weight_stats)),
            "ll_max_condition": float(max(s.max_condition for s in self.weight_stats)),
        }


def fit_nuisance(target: str, beta: float, law: ErrorLaw,
                 x_train: Array, z_train: Array, y_train: Array,
                 settings: dict[str, Any], shift_sign: float = 1.0,
                 mean_fn: str = "paper") -> Nuisance:
    kind = settings["kind"]
    if kind == "oracle":
        return AnalyticNuisance(target, beta, law, mean_fn=mean_fn)
    if kind == "controlled":
        n = len(y_train)
        shift = float(settings.get("c_m", 1.0))*n**(-float(settings["gamma_m"]))*shift_sign
        lam = float(settings.get("c_f", 1.0))*n**(-float(settings["gamma_f"]))
        if not 0 <= lam <= 1:
            raise ValueError("controlled contamination must lie in [0,1]")
        return AnalyticNuisance(target, beta, law, shift, lam,
                                float(settings.get("dagger_sd", 2.0)),
                                mean_fn)
    if kind == "learned":
        # Deliberately do not pass beta or law into the learned estimator.
        return LearnedNuisance(x_train, z_train, y_train, target, settings)
    if kind == "learned_mean_oracle_density":
        return LearnedMeanOracleDensityNuisance(
            x_train, z_train, y_train, target, settings, beta, law
        )
    if kind == "fake_mean_oracle_density":
        n = int(settings.get("sample_size", 2 * len(y_train)))
        if n < 1:
            raise ValueError("fake mean requires a positive sample_size")
        gamma = float(settings["gamma_m_x"] if target == "x" else settings["gamma_m_w"])
        shift = float(settings.get("c_m", 1.0)) * (n ** (-gamma)) * shift_sign
        return AnalyticNuisance(target, beta, law, shift, 0.0, 2.0, mean_fn)
    if kind == "fake_mean_fake_A":
        n = int(settings.get("sample_size", 2 * len(y_train)))
        if n < 1:
            raise ValueError("fake mean/A requires a positive sample_size")
        gamma = float(settings["gamma_m_x"] if target == "x" else settings["gamma_m_w"])
        shift = float(settings.get("c_m", 1.0)) * (n ** (-gamma)) * shift_sign
        a_shift = (float(settings.get("c_a", 1.0))
                   * (n ** (-float(settings["gamma_a"]))) * shift_sign)
        return AnalyticNuisance(target, beta, law, shift, 0.0, 2.0, mean_fn, a_shift)
    raise ValueError(f"unknown layer kind: {kind}")
