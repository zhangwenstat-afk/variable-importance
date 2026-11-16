"""Deterministic correctness checks, not claims about Monte Carlo operating characteristics."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()

from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

from marginal_mc.dgp import (ErrorLaw, generate, stable_rng, truth, true_mean,
                             covariate_quadrature, features)
from marginal_mc.oracle_calibration import MarginalOracle
from marginal_mc.nuisance import AnalyticNuisance, LocalLinear, fit_nuisance
from marginal_mc.estimators import (empirical_quantile, solve_semi, estimate_layer,
                                    spacing_inverse_density, nuisance_diagnostic)
from conditional_mc.estimators import (
    MEAN_BW_BY_S0, apply_s0_mean_bandwidth, apply_w_s0_0_bandwidth,
    apply_w_s0_m05_bandwidth, mean_bw_grid,
)
from marginal_mc.simulation import validate_config, make_cells, run_replication, execute
from marginal_mc.reporting import read_raw, summarize, rate_summary
from make_configs import (
    base, controlled, ORACLE, LEARNED, LEARNED_RESIDUAL_CV,
)


def small_config() -> dict:
    c = base("unittest")
    c.update({"errors": ["right"], "sample_sizes": [120], "repetitions": 2,
              "batch_size": 1, "layers": [deepcopy(ORACLE), controlled(.35), deepcopy(LEARNED)]})
    c["diagnostics"].update({"every": 1, "grid_side": 5, "density_points": 5, "b_nodes": 8})
    return validate_config(c)


class DistributionTests(unittest.TestCase):
    def test_mixture_exact_moments(self):
        for error in ("symmetric", "right", "left"):
            p, mu, sd = ErrorLaw(error).components
            self.assertAlmostEqual(float(p.sum()), 1)
            self.assertAlmostEqual(float(p @ mu), 0)
            self.assertAlmostEqual(float(p @ (mu*mu+sd*sd)), 1)

    def test_conditional_density_integrates_to_one(self):
        for error in ("symmetric", "right", "left"):
            law = ErrorLaw(error)
            for x in (-1.0, 0.3, 1.0):
                for beta in (0.0, 0.6):
                    integral = quad(lambda t: float(law.conditional_pdf(t, np.array(x), beta)),
                                    -np.inf, np.inf, epsabs=2e-8)[0]
                    self.assertAlmostEqual(integral, 1.0, places=6)

    def test_reflection_and_conditional_symmetry(self):
        x = np.linspace(-1, 1, 101)
        right, left, symmetric = ErrorLaw("right"), ErrorLaw("left"), ErrorLaw("symmetric")
        for beta in (0, 0.6):
            np.testing.assert_allclose(right.conditional_pdf(1.7, x, beta),
                                       left.conditional_pdf(-1.7, x, beta), atol=1e-14)
            np.testing.assert_allclose(symmetric.A(1.7, x, beta), 0, atol=1e-14)

    def test_analytic_oracle_pdf_agreement(self):
        for error in ("symmetric", "right", "left"):
            oracle = MarginalOracle(error)
            law = ErrorLaw(error)
            for beta in (0.0, 0.6):
                np.testing.assert_allclose(oracle.conditional_pdf(1.7, beta),
                                          law.conditional_pdf(1.7, oracle.x, beta), atol=1e-13)

    def test_calibration_matches_previous_table(self):
        a, b, c = [truth(error, .1, .2) for error in ("symmetric", "right", "left")]
        self.assertAlmostEqual(a["beta"], .522002, places=5)
        self.assertAlmostEqual(b["beta"], .604597, places=5)
        self.assertAlmostEqual(b["zeta_w"], 1.766949, places=5)
        self.assertAlmostEqual(c["beta"], b["beta"], places=12)
        self.assertAlmostEqual(c["kappa_x"], -b["kappa_x"], places=12)
        self.assertAlmostEqual(c["kappa_w"], -b["kappa_w"], places=12)
        for result in (a, b, c):
            self.assertAlmostEqual(result["delta_true"], .2, places=9)

    def test_conditional_density_derivative_bound(self):
        law = ErrorLaw("right")
        L = 5/(.6**2*math.sqrt(2*math.pi*math.e))
        x = np.repeat(np.linspace(-1, 1, 25), 41)
        t = np.tile(np.linspace(-5, 5, 41), 25)
        h = 1e-5
        for beta in (0, .6):
            derivative = (law.conditional_pdf(t+h, x, beta)
                          -law.conditional_pdf(t-h, x, beta))/(2*h)
            self.assertLessEqual(float(np.max(np.abs(derivative))), L+1e-6)

    def test_conditional_density_derivative_matches_finite_difference(self):
        law = ErrorLaw("right")
        x = np.linspace(-1, 1, 21)
        h = 1e-6
        for beta in (0.0, 0.6):
            for t in (-2.0, 0.0, 1.7):
                numeric = (law.conditional_pdf(t+h, x, beta)
                           -law.conditional_pdf(t-h, x, beta))/(2*h)
                analytic = law.conditional_pdf_derivative(t, x, beta)
                np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-8)
            q = 1.7
            np.testing.assert_allclose(
                law.D(q, x, beta),
                (law.conditional_pdf_derivative(q, x, beta)
                 -law.conditional_pdf_derivative(-q, x, beta)),
                atol=1e-14,
            )

    def test_second_order_terms_with_oracle_A_and_constant_shift(self):
        law = ErrorLaw("right")
        tr = truth("right", .1, .2)
        shift = 0.2
        fitted = AnalyticNuisance("w", tr["beta"], law, mean_shift=shift)
        diag = nuisance_diagnostic(
            fitted, "w", tr["beta"], law, tr["zeta_w"], tr["mean_A_w"],
            {"b_nodes": 48, "grid_side": 5, "density_points": 0},
        )
        xq, _, weights = covariate_quadrature("w", 48)
        expected = -0.5*float(weights @ (law.D(tr["zeta_w"], xq, 0.0)*(shift**2)))
        self.assertAlmostEqual(diag["delta_e"], 0.0, places=12)
        self.assertAlmostEqual(diag["neg_half_D_delta2"], expected, places=10)
        self.assertAlmostEqual(diag["so_score"], expected, places=10)

    def test_learned_oracle_density_has_zero_A_error(self):
        law = ErrorLaw("right")
        tr = truth("right", .1, .2)
        d = generate(160, tr["beta"], law, stable_rng(34, "so-oracle-A"))
        settings = deepcopy(LEARNED)
        settings["kind"] = "learned_mean_oracle_density"
        fitted = fit_nuisance("x", tr["beta"], law, d.x[d.folds[0]], d.z[d.folds[0]],
                              d.y[d.folds[0]], settings)
        diag = nuisance_diagnostic(
            fitted, "x", tr["beta"], law, tr["zeta_x"], tr["mean_A_x"],
            {"b_nodes": 16, "grid_side": 5, "density_points": 0},
        )
        self.assertAlmostEqual(diag["delta_e"], 0.0, places=12)
        self.assertTrue(math.isfinite(diag["neg_half_D_delta2"]))

    def test_population_orthogonality_and_conf_bias_sign(self):
        oracle = MarginalOracle("right")
        beta = oracle.calibrate_beta(.2)
        for b in (0.0, beta):
            metrics = oracle.boundary_metrics(b)
            q, mean_A = metrics.zeta, metrics.mean_A
            def probability(q_value, shift):
                return float(oracle.weights @ (oracle.conditional_cdf(q_value+shift, b)
                                                -oracle.conditional_cdf(-q_value+shift, b)))
            def semi_score(shift):
                return .9-probability(q, shift)+shift*mean_A
            h = 1e-4
            self.assertLess(abs((semi_score(h)-semi_score(-h))/(2*h)), 1e-6)
            qplus = brentq(lambda a: probability(a, h)-.9, .01, 6)
            qminus = brentq(lambda a: probability(a, -h)-.9, .01, 6)
            self.assertAlmostEqual((qplus-qminus)/(2*h), -metrics.kappa, places=5)

    def test_t4_base_is_unit_variance_and_calibrates(self):
        # t_4 has no fourth moment, so sample variance is not a reliable check.
        pdf_mass = quad(lambda g: 3.0 * (g * g + 2.0) ** (-2.5), -np.inf, np.inf)[0]
        second = quad(lambda g: g * g * 3.0 * (g * g + 2.0) ** (-2.5), -np.inf, np.inf)[0]
        self.assertAlmostEqual(pdf_mass, 1.0, places=8)
        self.assertAlmostEqual(second, 1.0, places=8)
        for error in ("symmetric", "right", "left"):
            law = ErrorLaw(error, error_base="t4")
            oracle = MarginalOracle(error, error_base="t4")
            for beta in (0.0, 0.6):
                np.testing.assert_allclose(
                    oracle.conditional_pdf(1.7, beta),
                    law.conditional_pdf(1.7, oracle.x, beta),
                    atol=1e-12,
                )
            for x in (-1.0, 0.3, 1.0):
                for beta in (0.0, 0.6):
                    integral = quad(
                        lambda t: float(law.conditional_pdf(t, np.array(x), beta)),
                        -np.inf, np.inf, epsabs=2e-7,
                    )[0]
                    self.assertAlmostEqual(integral, 1.0, places=5)
        result = truth("right", 0.1, 0.2, error_base="t4")
        self.assertAlmostEqual(result["delta_true"], 0.2, places=9)
        self.assertGreater(result["beta"], 0.0)
        self.assertGreater(result["zeta_w"], 0.0)
        self.assertGreater(result["zeta_x"], result["zeta_w"])

    def test_exponnorm_base_moments_reflection_and_calibration(self):
        from marginal_mc.oracle_calibration import (
            standardized_cdf, standardized_cdf_integral, standardized_pdf,
            standardized_pdf_derivative,
        )
        lam = 1.5
        scale = math.sqrt(1.0 + lam * lam)
        skew = 2.0 * lam**3 / (scale ** 3)
        self.assertAlmostEqual(skew, 1.152, places=3)
        pdf_mass = quad(lambda g: float(standardized_pdf(g, "exponnorm", lam, 1.0)),
                        -np.inf, np.inf, epsabs=2e-8)[0]
        mean = quad(lambda g: g * float(standardized_pdf(g, "exponnorm", lam, 1.0)),
                    -np.inf, np.inf, epsabs=2e-8)[0]
        second = quad(lambda g: g * g * float(standardized_pdf(g, "exponnorm", lam, 1.0)),
                      -np.inf, np.inf, epsabs=2e-8)[0]
        third = quad(lambda g: g**3 * float(standardized_pdf(g, "exponnorm", lam, 1.0)),
                     -np.inf, np.inf, epsabs=2e-8)[0]
        self.assertAlmostEqual(pdf_mass, 1.0, places=6)
        self.assertAlmostEqual(mean, 0.0, places=6)
        self.assertAlmostEqual(second, 1.0, places=6)
        self.assertAlmostEqual(third, skew, places=5)
        grid = np.linspace(-3.5, 3.5, 21)
        np.testing.assert_allclose(
            standardized_pdf(grid, "exponnorm", lam, -1.0),
            standardized_pdf(-grid, "exponnorm", lam, 1.0),
            atol=1e-12,
        )
        h = 1e-5
        for u in (-1.2, 0.3, 1.7):
            num = (float(standardized_cdf(u+h, "exponnorm", lam, 1.0))
                   - float(standardized_cdf(u-h, "exponnorm", lam, 1.0))) / (2*h)
            self.assertAlmostEqual(num, float(standardized_pdf(u, "exponnorm", lam, 1.0)), places=5)
            num_d = (float(standardized_pdf(u+h, "exponnorm", lam, 1.0))
                     - float(standardized_pdf(u-h, "exponnorm", lam, 1.0))) / (2*h)
            self.assertAlmostEqual(num_d, float(standardized_pdf_derivative(u, "exponnorm", lam, 1.0)), places=4)
            num_i = (float(standardized_cdf_integral(u+h, "exponnorm", lam, 1.0))
                     - float(standardized_cdf_integral(u-h, "exponnorm", lam, 1.0))) / (2*h)
            self.assertAlmostEqual(num_i, float(standardized_cdf(u, "exponnorm", lam, 1.0)), places=5)
            num_i_left = (float(standardized_cdf_integral(u+h, "exponnorm", lam, -1.0))
                          - float(standardized_cdf_integral(u-h, "exponnorm", lam, -1.0))) / (2*h)
            self.assertAlmostEqual(
                num_i_left, float(standardized_cdf(u, "exponnorm", lam, -1.0)), places=5
            )
        for error in ("right", "left"):
            law = ErrorLaw(error, error_base="exponnorm", emg_lambda=lam)
            oracle = MarginalOracle(error, error_base="exponnorm", emg_lambda=lam)
            for beta in (0.0, 0.6):
                np.testing.assert_allclose(
                    oracle.conditional_pdf(1.7, beta),
                    law.conditional_pdf(1.7, oracle.x, beta),
                    atol=1e-11,
                )
            for x in (-1.0, 0.3, 1.0):
                for beta in (0.0, 0.6):
                    integral = quad(
                        lambda t: float(law.conditional_pdf(t, np.array(x), beta)),
                        -np.inf, np.inf, epsabs=2e-7,
                    )[0]
                    self.assertAlmostEqual(integral, 1.0, places=5)
        right, left = ErrorLaw("right", error_base="exponnorm"), ErrorLaw("left", error_base="exponnorm")
        x = np.linspace(-1, 1, 51)
        np.testing.assert_allclose(right.conditional_pdf(1.7, x, 0.6),
                                   left.conditional_pdf(-1.7, x, 0.6), atol=1e-12)
        result = truth("right", 0.1, 0.2, error_base="exponnorm", emg_lambda=lam)
        flipped = truth("left", 0.1, 0.2, error_base="exponnorm", emg_lambda=lam)
        self.assertAlmostEqual(result["delta_true"], 0.2, places=9)
        self.assertAlmostEqual(flipped["beta"], result["beta"], places=12)
        self.assertAlmostEqual(flipped["kappa_x"], -result["kappa_x"], places=12)
        self.assertGreater(result["beta"], 0.0)

    def test_quadrature_weights_and_features(self):
        x, z, weights = covariate_quadrature("w", 16)
        self.assertAlmostEqual(float(weights.sum()), 1)
        self.assertAlmostEqual(float(weights @ (z*z)), 1)
        self.assertAlmostEqual(float(weights @ (x*x)), 1/3)
        self.assertLessEqual(float(np.max(np.abs(features(x, z, "w")))), 1)

    def test_data_generation_is_reproducible(self):
        law = ErrorLaw("right")
        a = generate(120, .6, law, stable_rng(10, "example"))
        b = generate(120, .6, law, stable_rng(10, "example"))
        np.testing.assert_array_equal(a.y, b.y)
        for fa, fb in zip(a.folds, b.folds):
            np.testing.assert_array_equal(fa, fb)
        self.assertEqual(len(set(a.folds[0]) & set(a.folds[1])), 0)
        np.testing.assert_array_equal(np.sort(np.r_[*a.folds]), np.arange(120))

    def test_quadratic_mean_keeps_residuals_and_calibration(self):
        from marginal_mc.dgp import m0
        x = np.linspace(-1, 1, 51)
        np.testing.assert_allclose(np.diff(m0(x, "quadratic"), n=2),
                                  4.0*(x[1]-x[0])**2, atol=1e-12)
        self.assertFalse(np.allclose(m0(x, "paper"), m0(x, "quadratic")))
        paper = truth("right", .1, .2)
        quadratic = truth("right", .1, .2)
        for key in ("beta", "zeta_x", "zeta_w", "mean_A_x", "mean_A_w"):
            self.assertEqual(paper[key], quadratic[key])
        law = ErrorLaw("right")
        a = generate(120, .6, law, stable_rng(10, "example"), "paper")
        b = generate(120, .6, law, stable_rng(10, "example"), "quadratic")
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.z, b.z)
        np.testing.assert_array_equal(a.eps, b.eps)
        np.testing.assert_allclose(b.y, a.y - 0.5*np.sin(np.pi*a.x), atol=1e-14)

    def test_sine_quad_curvature(self):
        from marginal_mc.dgp import m0
        x = np.array([-0.5, 0.0, 0.5])
        h = 1e-4
        m2 = (m0(x + h, "sine_quad") - 2 * m0(x, "sine_quad") + m0(x - h, "sine_quad")) / h**2
        np.testing.assert_allclose(m2, 1.0 - np.sin(np.pi * x), atol=1e-5)

    def test_quad_half_mean_is_x_plus_half_x_squared(self):
        from marginal_mc.dgp import m0
        x = np.linspace(-1, 1, 51)
        np.testing.assert_allclose(m0(x, "quad_half"), x + 0.5*x*x)
        np.testing.assert_allclose(np.diff(m0(x, "quad_half"), n=2),
                                  (x[1]-x[0])**2, atol=1e-12)
        paper = truth("right", .1, .2, mixture_a=5.0, beta=0.5)
        half = truth("right", .1, .2, mixture_a=5.0, beta=0.5)
        for key in ("beta", "zeta_x", "zeta_w", "mean_A_x", "mean_A_w"):
            self.assertEqual(paper[key], half[key])
        law = ErrorLaw("right")
        a = generate(120, .5, law, stable_rng(10, "example"), "paper")
        b = generate(120, .5, law, stable_rng(10, "example"), "quad_half")
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.eps, b.eps)
        np.testing.assert_allclose(
            b.y, a.y - 1.5*a.x*a.x - 0.5*np.sin(np.pi*a.x), atol=1e-14)

    def test_fixed_beta_does_not_recalibrate_delta(self):
        calibrated = truth("right", .1, .2, mixture_a=10.0)
        fixed = truth("right", .1, .2, mixture_a=10.0, beta=0.7)
        self.assertAlmostEqual(fixed["beta"], 0.7, places=12)
        self.assertNotAlmostEqual(fixed["delta_true"], 0.2, places=3)
        self.assertLess(fixed["delta_true"], calibrated["delta_true"])
        self.assertGreater(fixed["zeta_x"], fixed["zeta_w"])
        self.assertAlmostEqual(calibrated["beta"], 0.826004, places=5)


class NuisanceTests(unittest.TestCase):
    def test_local_linear_no_expand_and_anisotropic_h(self):
        rng = stable_rng(21, "aniso-ll")
        u = rng.uniform(-0.6, 0.6, (400, 2))
        query = np.array([[0.0, 0.0], [0.1, -0.2], [-0.15, 0.1]])
        result = LocalLinear(u, [0.35, 0.9], expand=False).weights(query)
        y = 2+u[:, 0]-3*u[:, 1]
        np.testing.assert_allclose(result.matrix @ np.ones(len(u)), 1, atol=1e-11)
        np.testing.assert_allclose(result.matrix @ y, 2+query[:, 0]-3*query[:, 1], atol=1e-9)
        self.assertEqual(result.expanded_rows, 0)

    def test_additive_mean_recovers_partial_linear(self):
        law = ErrorLaw("right")
        rng = stable_rng(22, "additive-m")
        d = generate(240, 0.6, law, rng)
        tr, ev = d.folds
        settings = deepcopy(LEARNED)
        settings["kind"] = "learned_mean_oracle_density"
        settings["mean_model"] = "additive"
        settings["expand_radius"] = False
        fitted = fit_nuisance("w", 0.6, law, d.x[tr], d.z[tr], d.y[tr], settings)
        diag = fitted.diagnostics()
        self.assertEqual(diag["mean_additive"], 1.0)
        self.assertEqual(diag["expand_radius"], 0.0)
        pred = fitted.predict_mean(d.x[ev], d.z[ev])
        truth_m = true_mean(d.x[ev], d.z[ev], "w", 0.6)
        self.assertLess(float(np.sqrt(np.mean((pred-truth_m)**2))), 0.45)

    def test_local_quadratic_reproduces_quadratics(self):
        rng = stable_rng(24, "local-quad")
        u = rng.uniform(-0.7, 0.7, (500, 2))
        query = np.array([[0.0, 0.0], [0.12, -0.2], [-0.18, 0.15]])
        result = LocalLinear(u, 0.55, expand=False, degree=2).weights(query)
        y = 1+2*u[:, 0]-u[:, 1]+0.5*u[:, 0]**2-0.3*u[:, 0]*u[:, 1]+0.8*u[:, 1]**2
        qy = 1+2*query[:, 0]-query[:, 1]+0.5*query[:, 0]**2-0.3*query[:, 0]*query[:, 1]+0.8*query[:, 1]**2
        np.testing.assert_allclose(result.matrix @ np.ones(len(u)), 1, atol=1e-10)
        np.testing.assert_allclose(result.matrix @ y, qy, atol=1e-8)
        self.assertEqual(result.expanded_rows, 0)

    def test_local_cubic_reproduces_cubics(self):
        rng = stable_rng(25, "local-cubic")
        u = rng.uniform(-0.75, 0.75, (800, 2))
        query = np.array([[0.0, 0.0], [0.1, -0.15], [-0.12, 0.08]])
        result = LocalLinear(u, 0.7, expand=False, degree=3).weights(query)
        y = (1+2*u[:, 0]-u[:, 1]+0.4*u[:, 0]**2-0.2*u[:, 0]*u[:, 1]+0.5*u[:, 1]**2
             +0.3*u[:, 0]**3-0.15*u[:, 0]**2*u[:, 1]+0.25*u[:, 0]*u[:, 1]**2
             -0.1*u[:, 1]**3)
        qy = (1+2*query[:, 0]-query[:, 1]+0.4*query[:, 0]**2
              -0.2*query[:, 0]*query[:, 1]+0.5*query[:, 1]**2
              +0.3*query[:, 0]**3-0.15*query[:, 0]**2*query[:, 1]
              +0.25*query[:, 0]*query[:, 1]**2-0.1*query[:, 1]**3)
        np.testing.assert_allclose(result.matrix @ np.ones(len(u)), 1, atol=1e-10)
        np.testing.assert_allclose(result.matrix @ y, qy, atol=1e-7)
        self.assertEqual(result.expanded_rows, 0)
        self.assertEqual(result.pinv_rows, 0)

    def test_anisotropic_mean_sets_hz_gt_hx(self):
        law = ErrorLaw("right")
        d = generate(200, 0.6, law, stable_rng(23, "aniso-m"))
        tr = d.folds[0]
        settings = deepcopy(LEARNED)
        settings["kind"] = "learned_mean_oracle_density"
        settings["mean_bandwidth"] = "anisotropic"
        settings["expand_radius"] = False
        settings["z_scale"] = 3.0
        fitted = fit_nuisance("w", 0.6, law, d.x[tr], d.z[tr], d.y[tr], settings)
        diag = fitted.diagnostics()
        self.assertEqual(diag["mean_anisotropic"], 1.0)
        self.assertGreater(diag["h_m_z"], diag["h_m"]*2.5)

    def test_anisotropic_mean_keeps_twodim_base_rate(self):
        law = ErrorLaw("right")
        d = generate(200, 0.6, law, stable_rng(23, "aniso-base"))
        tr = d.folds[0]
        settings = deepcopy(LEARNED)
        settings["kind"] = "learned_mean_oracle_density"
        settings["mean_bandwidth"] = "anisotropic"
        settings["expand_radius"] = False
        settings["z_scale"] = 3.0
        fitted = fit_nuisance("w", 0.6, law, d.x[tr], d.z[tr], d.y[tr], settings)
        diag = fitted.diagnostics()
        n = len(tr)
        logn_over_n = math.log(max(n, 3))/n
        base = logn_over_n**(1/6)
        self.assertAlmostEqual(diag["h_m"]/0.8, base, places=10)
        self.assertAlmostEqual(diag["h_m_z"]/diag["h_m"], 3.0, places=10)
        old = logn_over_n**(1/5)
        self.assertGreater(abs(diag["h_m"]/0.8 - old), 1e-4)

    def test_joint_anisotropic_cv_selects_ratio(self):
        law = ErrorLaw("right")
        d = generate(400, 0.6, law, stable_rng(26, "joint-cr"))
        tr = d.folds[0]
        settings = deepcopy(LEARNED)
        settings["kind"] = "learned_mean_oracle_density"
        settings["mean_bandwidth"] = "anisotropic"
        settings["expand_radius"] = False
        settings["bandwidth_selection"] = {
            "method": "inner_cv",
            "folds": 3,
            "max_points": 256,
            "m_multipliers": [0.75, 1.0, 1.25],
            "z_scales": [1.0, 2.0, 4.0],
            "f_multipliers": [1.0],
            "y_multipliers": [1.0],
        }
        fitted = fit_nuisance("w", 0.6, law, d.x[tr], d.z[tr], d.y[tr], settings)
        diag = fitted.diagnostics()
        self.assertEqual(diag["mean_anisotropic"], 1.0)
        self.assertGreater(diag["z_scale"], 1.0)
        self.assertAlmostEqual(diag["h_m_z"]/diag["h_m"], diag["z_scale"], places=8)
        n = len(tr)
        base = (math.log(max(n, 3))/n)**(1/6)
        self.assertAlmostEqual(diag["h_m"]/diag["cv_m_multiplier"], base, places=8)

    def test_s0_mean_bandwidth_grid_is_local_and_does_not_mutate(self):
        layer = {
            "kind": "learned",
            "bandwidth_selection": {
                "method": "inner_cv",
                "m_multipliers": [0.5, 0.75, 1.0],
                "z_scales": [4.0, 5.0, 6.0, 8.0],
                "f_multipliers": [0.75],
            },
        }
        original = deepcopy(layer["bandwidth_selection"])
        fitted_layer = apply_s0_mean_bandwidth(layer, -0.5, 0.25)
        self.assertEqual(layer["bandwidth_selection"], original)
        self.assertEqual(
            fitted_layer["bandwidth_selection"]["m_multipliers"],
            mean_bw_grid(-0.5)["m_multipliers"],
        )
        self.assertEqual(fitted_layer["bandwidth_selection"]["cv_center"], -0.5)
        self.assertEqual(fitted_layer["bandwidth_selection"]["f_multipliers"], [0.75])
        zero = apply_s0_mean_bandwidth(layer, 0.0, 0.25)
        self.assertEqual(zero["bandwidth_selection"]["cv_center"], 0.0)
        self.assertEqual(
            zero["bandwidth_selection"]["m_multipliers"],
            [0.5, 0.75, 1.0, 1.25],
        )
        shared = deepcopy(MEAN_BW_BY_S0[-0.5])
        widened = apply_w_s0_m05_bandwidth(fitted_layer, -0.5)
        self.assertEqual(MEAN_BW_BY_S0[-0.5], shared)
        self.assertEqual(widened["bandwidth_selection"]["m_multipliers"], [0.1875, 0.25])
        self.assertEqual(widened["bandwidth_selection"]["z_scales"], [32.0, 64.0])
        self.assertEqual(widened["bandwidth_selection"]["cv_center"], -0.5)
        self.assertEqual(fitted_layer["bandwidth_selection"]["m_multipliers"], [0.25, 0.5, 0.75])
        self.assertIs(apply_w_s0_m05_bandwidth(fitted_layer, 0.0), fitted_layer)
        fixed = apply_w_s0_0_bandwidth(zero, 0.0)
        self.assertEqual(fixed["bandwidth_selection"]["m_multipliers"], [0.5, 0.625])
        self.assertEqual(fixed["bandwidth_selection"]["z_scales"], [16.0, 32.0])
        self.assertEqual(zero["bandwidth_selection"]["m_multipliers"], [0.5, 0.75, 1.0, 1.25])
        self.assertIs(apply_w_s0_0_bandwidth(zero, -0.5), zero)
        law = ErrorLaw("right")
        d = generate(300, 0.2, law, stable_rng(27, "local-cv"))
        tr = d.folds[0]
        settings = deepcopy(LEARNED)
        settings["kind"] = "learned_mean_oracle_density"
        settings["mean_bandwidth"] = "anisotropic"
        settings["expand_radius"] = False
        settings["bandwidth_selection"] = fitted_layer["bandwidth_selection"]
        fitted = fit_nuisance("w", 0.2, law, d.x[tr], d.z[tr], d.y[tr], settings)
        diag = fitted.diagnostics()
        self.assertIn(diag["cv_m_multiplier"], {0.25, 0.5, 0.75})
        self.assertIn(diag["z_scale"], {6.0, 8.0, 12.0, 16.0})

    def test_fake_mean_oracle_density_uses_full_n_exponents(self):
        law = ErrorLaw("right")
        x, z = np.linspace(-1, 1, 20), np.zeros(20)
        settings = {
            "kind": "fake_mean_oracle_density",
            "gamma_m_x": 0.4,
            "gamma_m_w": 1.0 / 3.0,
            "c_m": 1.0,
            "sample_size": 1000,
        }
        y = np.zeros(40)
        fx = fit_nuisance("x", .6, law, np.zeros(40), np.zeros(40), y, settings)
        fw = fit_nuisance("w", .6, law, np.zeros(40), np.zeros(40), y, settings)
        self.assertAlmostEqual(fx.mean_shift, 1000 ** (-0.4), places=12)
        self.assertAlmostEqual(fw.mean_shift, 1000 ** (-1.0 / 3.0), places=12)
        self.assertEqual(fx.contamination, 0.0)
        np.testing.assert_allclose(
            fx.prepare(x, z).mean - true_mean(x, z, "x", .6),
            1000 ** (-0.4), atol=1e-14,
        )
        np.testing.assert_allclose(
            fw.prepare(x, z).A(1.7), law.A(1.7, x, 0.0), atol=1e-14,
        )

    def test_fake_mean_fake_A_adds_n_exponents(self):
        law = ErrorLaw("right")
        x, z = np.linspace(-1, 1, 20), np.zeros(20)
        settings = {
            "kind": "fake_mean_fake_A",
            "gamma_m_x": 0.4,
            "gamma_m_w": 1.0 / 3.0,
            "gamma_a": 1.0 / 3.0,
            "c_m": 1.0,
            "c_a": 1.0,
            "sample_size": 1000,
        }
        y = np.zeros(40)
        fx = fit_nuisance("x", .6, law, np.zeros(40), np.zeros(40), y, settings)
        fw = fit_nuisance("w", .6, law, np.zeros(40), np.zeros(40), y, settings)
        self.assertAlmostEqual(fx.mean_shift, 1000 ** (-0.4), places=12)
        self.assertAlmostEqual(fw.mean_shift, 1000 ** (-1.0 / 3.0), places=12)
        self.assertAlmostEqual(fx.a_shift, 1000 ** (-1.0 / 3.0), places=12)
        self.assertAlmostEqual(fw.a_shift, 1000 ** (-1.0 / 3.0), places=12)
        np.testing.assert_allclose(
            fw.prepare(x, z).A(1.7),
            law.A(1.7, x, 0.0) + 1000 ** (-1.0 / 3.0),
            atol=1e-14,
        )

    def test_controlled_nuisance_exact_rates(self):
        law = ErrorLaw("right")
        x, z = np.linspace(-1, 1, 20), np.zeros(20)
        layer = controlled(.35)
        y = np.zeros(100)
        fitted = fit_nuisance("x", .6, law, np.zeros(100), np.zeros(100), y, layer)
        expected = 100**(-.35)
        self.assertAlmostEqual(fitted.mean_shift, expected)
        self.assertAlmostEqual(fitted.contamination, expected)
        prep = fitted.prepare(x, z)
        np.testing.assert_allclose(prep.mean-true_mean(x, z, "x", .6), expected, atol=1e-14)
        np.testing.assert_allclose(prep.A(1.7), (1-expected)*law.A(1.7, x, .6), atol=1e-14)

    def test_learned_fit_does_not_use_oracle_parameters(self):
        law = ErrorLaw("right")
        d = generate(120, .6, law, stable_rng(30, "no-leakage"))
        tr, ev = d.folds
        first = fit_nuisance("w", .6, law, d.x[tr], d.z[tr], d.y[tr], LEARNED)
        second = fit_nuisance("w", 99., ErrorLaw("symmetric"), d.x[tr], d.z[tr], d.y[tr], LEARNED)
        p1, p2 = first.prepare(d.x[ev], d.z[ev]), second.prepare(d.x[ev], d.z[ev])
        np.testing.assert_array_equal(p1.mean, p2.mean)
        np.testing.assert_array_equal(p1.pdf(1.7), p2.pdf(1.7))

    def test_learned_mean_oracle_density_layer_separates_nuisances(self):
        law = ErrorLaw("right")
        d = generate(160, .6, law, stable_rng(30, "oracle-density"))
        tr, ev = d.folds
        settings = deepcopy(LEARNED)
        settings["kind"] = "learned_mean_oracle_density"
        first = fit_nuisance("w", .6, law, d.x[tr], d.z[tr], d.y[tr], settings)
        second = fit_nuisance(
            "w", 99.0, ErrorLaw("symmetric"),
            d.x[tr], d.z[tr], d.y[tr], settings,
        )
        p1, p2 = first.prepare(d.x[ev], d.z[ev]), second.prepare(d.x[ev], d.z[ev])
        np.testing.assert_array_equal(p1.mean, p2.mean)
        self.assertFalse(np.allclose(p1.pdf(1.7), p2.pdf(1.7)))
        self.assertEqual(first.diagnostics()["density_oracle"], 1.0)

    def test_learned_inner_cv_is_training_only_and_two_dimensional(self):
        law = ErrorLaw("right")
        d = generate(160, .6, law, stable_rng(31, "inner-cv"))
        tr, ev = d.folds
        settings = deepcopy(LEARNED)
        settings["bandwidth_selection"] = {
            "method": "inner_cv",
            "folds": 3,
            "max_points": 24,
            "m_multipliers": [0.75, 1.0, 1.25],
            "f_multipliers": [0.75, 1.0, 1.25],
            "y_multipliers": [0.75, 1.0, 1.25],
        }
        first = fit_nuisance("w", .6, law, d.x[tr], d.z[tr], d.y[tr], settings)
        second = fit_nuisance(
            "w", 99.0, ErrorLaw("symmetric"),
            d.x[tr], d.z[tr], d.y[tr], settings,
        )
        first_diag, second_diag = first.diagnostics(), second.diagnostics()
        self.assertEqual(first_diag["feature_dim"], 2.0)
        self.assertEqual(first_diag["bandwidth_cv"], 1.0)
        for key in (
            "h_m", "h_f", "b_y", "cv_m_multiplier", "cv_f_multiplier",
            "cv_y_multiplier", "cv_m_score", "cv_density_score",
        ):
            self.assertEqual(first_diag[key], second_diag[key])
            self.assertTrue(math.isfinite(first_diag[key]))
        prepared = first.prepare(d.x[ev], d.z[ev])
        self.assertTrue(np.isfinite(prepared.pdf(1.7)).all())

    def test_residual_density_is_training_only_and_supports_x_only(self):
        law = ErrorLaw("right")
        d = generate(180, .6, law, stable_rng(32, "residual-density"))
        tr, ev = d.folds
        settings = deepcopy(LEARNED)
        settings.update({
            "density_estimator": "residual_loo",
            "density_feature_mode": "x_only",
            "direct_contrast": True,
        })
        first = fit_nuisance("w", .6, law, d.x[tr], d.z[tr], d.y[tr], settings)
        second = fit_nuisance(
            "w", 99.0, ErrorLaw("symmetric"),
            d.x[tr], d.z[tr], d.y[tr], settings,
        )
        first_diag, second_diag = first.diagnostics(), second.diagnostics()
        self.assertEqual(first_diag["density_feature_dim"], 1.0)
        self.assertEqual(first_diag["density_residual_loo"], 1.0)
        self.assertEqual(first_diag["density_loo_fallback_rows"], 0.0)
        for key in (
            "h_f", "b_y", "density_feature_dim", "density_feature_x_only",
            "response_scale", "density_loo_fallback_rows",
        ):
            self.assertEqual(first_diag[key], second_diag[key])
        prepared = first.prepare(
            np.array([-1.0, 1.0]), np.array([-np.sqrt(3.0), np.sqrt(3.0)])
        )
        self.assertTrue(np.isfinite(prepared.pdf(1.7)).all())
        self.assertTrue(np.isfinite(prepared.A(1.7)).all())

    def test_residual_density_inner_cv_selects_dimension_without_oracle(self):
        law = ErrorLaw("right")
        d = generate(180, .6, law, stable_rng(33, "residual-density-cv"))
        tr, ev = d.folds
        settings = deepcopy(LEARNED)
        settings.update({
            "density_estimator": "residual_loo",
            "density_feature_selection": "inner_cv",
            "density_feature_candidates": ["all", "x_only"],
            "direct_contrast": True,
            "bandwidth_selection": {
                "method": "inner_cv",
                "folds": 3,
                "max_points": 24,
                "m_multipliers": [0.75, 1.0, 1.25],
                "f_multipliers": [0.75, 1.0, 1.25],
                "y_multipliers": [0.75, 1.0, 1.25],
            },
        })
        first = fit_nuisance("w", .6, law, d.x[tr], d.z[tr], d.y[tr], settings)
        second = fit_nuisance(
            "w", 99.0, ErrorLaw("symmetric"),
            d.x[tr], d.z[tr], d.y[tr], settings,
        )
        first_diag, second_diag = first.diagnostics(), second.diagnostics()
        self.assertEqual(first_diag["density_candidate_count"], 2.0)
        self.assertIn(first_diag["density_feature_dim"], (1.0, 2.0))
        self.assertEqual(first_diag["density_feature_dim"],
                         second_diag["density_feature_dim"])
        self.assertEqual(first_diag["cv_density_score"],
                         second_diag["cv_density_score"])
        prepared = first.prepare(d.x[ev], d.z[ev])
        self.assertTrue(np.isfinite(prepared.A(1.7)).all())

    def test_positive_part_cannot_increase_pointwise_error(self):
        truth_pdf = np.array([.1, .4, .0, 1.0])
        estimate = np.array([-.1, .2, -.3, 1.2])
        self.assertTrue(np.all(np.abs(np.maximum(estimate, 0)-truth_pdf)
                               <= np.abs(estimate-truth_pdf)))


class EstimatorTests(unittest.TestCase):
    def test_quantile_definition(self):
        values = np.arange(1., 11.)
        self.assertEqual(empirical_quantile(values, .9), 9.)
        self.assertEqual(empirical_quantile(values, .91), 10.)
        inv_g, c = spacing_inverse_density([values, values], .1, 20)
        self.assertGreater(inv_g, 0)
        self.assertLess(.9+c, 1)

    def test_original_score_root_tolerance(self):
        for error in ("symmetric", "right", "left"):
            law = ErrorLaw(error)
            d = generate(200, .6, law, stable_rng(40, error))
            fitted = AnalyticNuisance("w", .6, law, .15, .2)
            prep = fitted.prepare(d.x, d.z)
            residual = d.y-prep.mean
            q0 = empirical_quantile(np.abs(residual), .9)
            result = solve_semi(residual, prep, .9, q0)
            self.assertTrue(result.success)
            self.assertLessEqual(len(residual)*abs(result.score), 1.05)

    def test_oracle_symmetry_equals_conf(self):
        c = small_config()
        law = ErrorLaw("symmetric")
        tr = truth("symmetric", .1, .2)
        d = generate(200, tr["beta"], law, stable_rng(41, "symmetry"))
        rows = estimate_layer(d, law, ORACLE, c, tr, (1., 1.), False)
        for key in ("zeta_x_est", "zeta_w_est", "delta_est", "se_delta"):
            self.assertAlmostEqual(rows[0][key], rows[1][key], places=12)

    def test_theorem_one_sign_and_delta_covariance(self):
        c = small_config()
        law = ErrorLaw("right")
        tr = truth("right", .1, .2)
        d = generate(200, tr["beta"], law, stable_rng(42, "influence"))
        row = estimate_layer(d, law, ORACLE, c, tr, (1., 1.), False)[0]
        eps = d.y-true_mean(d.x, d.z, "w", tr["beta"])
        q, g = tr["zeta_w"], tr["g_w"]
        phi = .9-(np.abs(eps) <= q)-eps*law.A(q, d.x, 0.)
        self.assertAlmostEqual(row["linear_w"], float(phi.mean()/g), places=12)
        self.assertAlmostEqual(row["sqrt_n_remainder_w"],
             math.sqrt(200)*(row["zeta_w_est"]-q-float(phi.mean()/g)), places=12)
        self.assertNotAlmostEqual(row["v_delta_empirical"],
                                  row["v_x_empirical"]+row["v_w_empirical"], places=5)

    def test_diagnostics_do_not_change_estimates(self):
        c = small_config()
        cell = make_cells(c)[0]
        a = run_replication(cell, 0, c)
        c["diagnostics"]["enabled"] = False
        b = run_replication(cell, 0, c)
        for first, second in zip(a, b):
            for key in ("zeta_x_est", "zeta_w_est", "delta_est", "se_delta"):
                self.assertAlmostEqual(first[key], second[key], places=12)

    def test_controlled_boundary_and_random_shift_settings_execute(self):
        c = small_config()
        c["layers"] = ([controlled(g) for g in (.30, .25, .20)]
                       + [controlled(.35, 2), controlled(.35, shift="negative"),
                          controlled(.35, shift="rademacher")])
        rows = run_replication(make_cells(c)[0], 0, c)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["estimator_valid"] for row in rows))
        for row in rows:
            if row["method"] == "conf" and row["shift_mode"] == "negative":
                self.assertLess(row["bias_pred_delta"], 0)

    def test_switch_to_manuscript_variance(self):
        c = small_config()
        c["variance"] = "manuscript"
        rows = run_replication(make_cells(c)[0], 0, c)
        for row in rows:
            if row["inference_valid"]:
                self.assertAlmostEqual(row["se_delta"]**2*120, row["v_delta_manuscript"], places=12)


class ExperimentTests(unittest.TestCase):
    def test_b_cancellation_ratio_and_independent_covariates(self):
        from marginal_mc.estimators import b_functionals
        D = np.array([1.0, -1.0, 0.5])
        error = np.array([2.0, 2.0, 0.0])
        weights = np.full(3, 1/3)
        stats = b_functionals(D, error, weights)
        self.assertAlmostEqual(stats["B"], 0.0, places=12)
        self.assertGreater(stats["B_abs"], 0)
        self.assertAlmostEqual(stats["B_cancel"], 0.0, places=12)
        aligned = b_functionals(np.ones(3), np.ones(3), weights)
        self.assertAlmostEqual(aligned["B_cancel"], 1.0, places=12)
        c = small_config()
        c["mean_fn"] = "quadratic"
        c["layers"] = [deepcopy(LEARNED)]
        c["diagnostics"].update({"density_points": 0, "b_mc_size": 200, "b_nodes": 8})
        cell = make_cells(c)[0]
        rows = run_replication(cell, 0, c)
        conf = next(row for row in rows if row["method"] == "conf")
        for target in ("x", "w"):
            cancel = conf[f"{target}_B_cancel"]
            self.assertTrue(0 <= cancel <= 1)
            self.assertTrue(math.isfinite(conf[f"{target}_B_mc"]))
            self.assertTrue(math.isfinite(conf[f"{target}_mean_D"]))
        self.assertAlmostEqual(conf["x_mean_D"], cell["mean_A_x"], places=2)

    def test_invalid_config_rejected(self):
        c = small_config()
        for key, bad in [("alpha", 1.), ("sample_sizes", [101]), ("variance", "naive")]:
            changed = deepcopy(c)
            changed[key] = bad
            with self.assertRaises(ValueError):
                validate_config(changed)
        changed = deepcopy(c)
        changed["mean_fn"] = "cubic"
        with self.assertRaises(ValueError):
            validate_config(changed)
        changed = deepcopy(c)
        changed["layers"][2]["bandwidth_selection"] = {
            "method": "inner_cv", "folds": 1,
        }
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_residual_config_validates(self):
        c = small_config()
        c["layers"] = [deepcopy(LEARNED_RESIDUAL_CV)]
        validated = validate_config(c)
        self.assertEqual(
            validated["layers"][0]["density_estimator"], "residual_loo"
        )
        self.assertEqual(
            validated["layers"][0]["density_feature_selection"], "inner_cv"
        )

    def test_local_alternative_and_calibration_cells(self):
        c = small_config()
        c["studies"] = [{"name": "local", "type": "local_power", "delta0": .2,
                         "local_cs": [0., 2.]}]
        c["size_adjustment"] = {"enabled": True, "calibration_reps": 3}
        cells = make_cells(c)
        self.assertEqual(len(cells), 3)
        self.assertEqual(sum(cell["role"] == "calibration" for cell in cells), 1)
        self.assertAlmostEqual(cells[1]["delta_target"], .2+2/math.sqrt(120), places=10)
        self.assertEqual(len({cell["calibration_key"] for cell in cells}), 1)
        a = stable_rng(1, "data", "right", 120, ".2", "evaluation", 0).normal(size=10)
        b = stable_rng(1, "data", "right", 120, ".2", "calibration", 0).normal(size=10)
        self.assertFalse(np.array_equal(a, b))

    def test_failure_rate_bounds(self):
        result = rate_summary(np.array([1., 0., 0., 0.]), "reject", total=5)
        self.assertEqual(result["reject"], .25)
        self.assertEqual(result["reject_all_lower"], .2)
        self.assertEqual(result["reject_all_upper"], .4)

    def test_entirely_invalid_rate_has_honest_bounds(self):
        result = rate_summary(np.array([], dtype=float), "reject", total=5)
        self.assertTrue(math.isnan(result["reject"]))
        self.assertEqual(result["reject_all_lower"], 0.)
        self.assertEqual(result["reject_all_upper"], 1.)

    def test_parallel_reproducibility_resume_and_reporting(self):
        c = small_config()
        c["diagnostics"]["enabled"] = False
        c["studies"] = [{"name": "inference", "type": "inference", "delta0": .2,
                         "gaps": [0., .1]}]
        c["size_adjustment"] = {"enabled": True, "calibration_reps": 2}
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"serial", Path(folder)/"parallel"
            execute(c, a, jobs=1)
            execute(c, b, jobs=2)
            sort = ["cell_id", "rep", "layer", "method"]
            da = read_raw(a).sort_values(sort).reset_index(drop=True)
            db = read_raw(b).sort_values(sort).reset_index(drop=True)
            pd.testing.assert_frame_equal(da, db, check_exact=False, rtol=1e-12, atol=1e-12)
            execute(c, a, jobs=1, resume=True)
            self.assertEqual(len(read_raw(a)), len(da))
            audit = summarize(a)
            self.assertEqual(audit["raw_rows"], 36)
            self.assertEqual(audit["failed_estimation_rows"], 0)
            infer = pd.read_csv(a/"summaries"/"inference_summary.csv")
            self.assertTrue(infer["critical_size_adjusted"].notna().all())
            changed = deepcopy(c)
            changed["repetitions"] = 3
            with self.assertRaises(ValueError):
                execute(changed, a, resume=True)

    def test_acquisition_grid_holds_beta_and_matches_anchors(self):
        from conditional_mc.simulation import make_cells, validate_config
        grid = [round(-0.5 + 0.05 * i, 2) for i in range(21)]
        beta = 0.6779790806045591
        config = validate_config({
            "errors": ["left", "right"],
            "sample_sizes": [4000],
            "locations": grid,
            "repetitions": 1,
            "fixed_beta": beta,
            "mean_bw_by_s0": False,
            "layers": [{"id": "x", "kind": "learned"}],
            "studies": [{"name": "acquire", "type": "acquisition", "delta0": 0.2}],
        })
        cells = make_cells(config)
        self.assertEqual(len(cells), 2)
        for cell in cells:
            self.assertEqual(cell["study_type"], "acquisition")
            self.assertEqual(cell["beta"], beta)
            self.assertEqual(cell["locations"], grid)
            anchors = {
                -0.5: 0.2782437443730734,
                0.0: 0.2,
                0.5: 0.14636441331728411,
            }
            for s0, delta in anchors.items():
                got = cell["truth_by_s0"][f"{s0:.6g}"]["delta_true"]
                self.assertAlmostEqual(got, delta, places=6)
            self.assertEqual(cell["truth_by_s0"]["0"]["beta"], beta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
