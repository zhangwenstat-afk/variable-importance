"""Kernel-g correctness checks, independent of Monte Carlo tuning outcomes."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()


import math
import unittest

import numpy as np

from conditional_mc.kernel_g import prepare_g_fits, validate_options
from marginal_mc.nuisance import (
    LearnedNuisance, LearnedPrepared, LocalLinear, _density_features, _loo_residuals,
)
from recompute_kernel_g_se import g_from_folds


class KernelGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20261016)
        settings = {
            "density_estimator": "residual_loo", "m_bw_mult": 0.75,
            "f_bw_mult": 0.65, "y_bw_mult": 0.2, "min_neighbors": 12,
        }
        cls.fits = {"x": [], "w": []}
        cls.w_x_only = []
        cls.x_quadratic = []
        for fold in range(2):
            x = rng.uniform(-1.2, 1.2, 120)
            z = rng.uniform(-math.sqrt(3), math.sqrt(3), len(x))
            error = rng.exponential(1.0, len(x)) - 1.0
            if fold:
                error = -error
            y = 2.0 * x * x + 0.3 * z + error
            cls.fits["x"].append(LearnedNuisance(x, z, y, "x", settings))
            cls.fits["w"].append(LearnedNuisance(x, z, y, "w", settings))
            cls.w_x_only.append(LearnedNuisance(
                x, z, y, "w", dict(settings, density_feature_mode="x_only"),
            ))
            cls.x_quadratic.append(LearnedNuisance(
                x, z, y, "x", dict(settings, mean_degree=2),
            ))

    def test_default_matches_legacy_for_both_w_feature_modes(self):
        for w_fits in (self.fits["w"], self.w_x_only):
            fits = {"x": self.fits["x"], "w": w_fits}
            prepared = prepare_g_fits(fits)
            for target in ("x", "w"):
                for s0 in (-0.5, 0.0, 0.5):
                    for q in (0.7, 1.5, 2.5):
                        with self.subTest(target=target, s0=s0, q=q):
                            self.assertAlmostEqual(
                                prepared.evaluate(target, s0, q),
                                g_from_folds(fits, target, s0, q), places=12,
                            )

    def test_default_preserves_an_existing_quadratic_mean(self):
        fits = {"x": self.x_quadratic, "w": self.fits["w"]}
        prepared = prepare_g_fits(fits)
        self.assertIsNone(validate_options()["x_mean_degree"])
        for fold in range(2):
            self.assertEqual(prepared.diagnostics[f"x_fold{fold}_applied"], 0.0)
        for s0 in (-0.5, 0.0, 0.5):
            self.assertAlmostEqual(
                prepared.evaluate("x", s0, 1.5),
                g_from_folds(fits, "x", s0, 1.5), places=12,
            )

    def test_training_objects_are_unchanged(self):
        snapshots = [
            (f, f.density_response.copy(), f.mean_fit.response.copy(), f.mean_fit.train.copy(),
             f.hm, f.hf, f.b, list(f.weight_stats))
            for group in self.fits.values() for f in group
        ]
        prepared = prepare_g_fits(self.fits, {
            "x_mean_degree": 2, "x_mean_h_multiplier": 1.5,
            "response_bw_multiplier": 1.1, "x_hf_multiplier": 1.2,
        })
        for s0 in (-0.5, 0.0, 0.5):
            prepared.evaluation_diagnostics(s0, {"x": 1.5, "w": 1.2})
        for f, residual, response, train, hm, hf, b, stats in snapshots:
            np.testing.assert_array_equal(f.density_response, residual)
            np.testing.assert_array_equal(f.mean_fit.response, response)
            np.testing.assert_array_equal(f.mean_fit.train, train)
            self.assertEqual((f.hm, f.hf, f.b), (hm, hf, b))
            self.assertEqual(len(f.weight_stats), len(stats))
            self.assertTrue(all(a is b for a, b in zip(f.weight_stats, stats)))

    def test_mixture_is_of_densities_and_w_is_unchanged(self):
        opts = {"x_mean_degree": 2, "x_mean_h_multiplier": 1.5}
        original = prepare_g_fits(self.fits)
        quadratic = prepare_g_fits(self.fits, opts)
        half = prepare_g_fits(self.fits, dict(opts, x_density_mix=0.5))
        zero = prepare_g_fits(self.fits, dict(opts, x_density_mix=0.0))
        for fold in range(2):
            for s0 in (-0.5, 0.0, 0.5):
                g0 = original.evaluate_fold("x", fold, s0, 1.5)
                g1 = quadratic.evaluate_fold("x", fold, s0, 1.5)
                self.assertEqual(zero.evaluate_fold("x", fold, s0, 1.5), g0)
                self.assertAlmostEqual(half.evaluate_fold("x", fold, s0, 1.5),
                                       (g0 + g1) / 2, places=12)
                self.assertEqual(quadratic.evaluate_fold("w", fold, s0, 1.5),
                                 original.evaluate_fold("w", fold, s0, 1.5))

    def test_right_skew_trigger_uses_each_training_fold(self):
        opts = {"x_mean_degree": 2, "x_mean_h_multiplier": 1.5}
        original = prepare_g_fits(self.fits)
        quadratic = prepare_g_fits(self.fits, opts)
        adaptive = prepare_g_fits(self.fits, dict(opts, apply_if="right_skew"))
        for fold in range(2):
            residual = self.fits["x"][fold].density_response
            centered = residual - residual.mean()
            skew = float(np.mean(centered ** 3) / np.mean(centered ** 2) ** 1.5)
            self.assertAlmostEqual(adaptive.diagnostics[f"x_fold{fold}_training_skew"], skew)
            self.assertEqual(adaptive.diagnostics[f"x_fold{fold}_applied"], float(fold == 0))
            reference = quadratic if fold == 0 else original
            for q in (0.7, 1.5, 2.5):
                self.assertEqual(adaptive.evaluate_fold("x", fold, 0.0, q),
                                 reference.evaluate_fold("x", fold, 0.0, q))

    def test_quadratic_density_matches_independent_loo_reconstruction(self):
        fitted = self.fits["x"][0]
        old = fitted.mean_smoother
        smoother = LocalLinear(
            fitted.mean_fit.train, fitted.hm * 1.5, old.min_neighbors,
            condition_limit=old.condition_limit, expand=old.expand, degree=2,
        )
        residual, _, _ = _loo_residuals(smoother, fitted.mean_fit.train, fitted.mean_fit.response)
        query = _density_features(np.array([0.]), np.array([0.]), "x", fitted.density_feature_mode)
        weights = fitted.density_smoother.weights(query).matrix
        reference = LearnedPrepared(np.zeros(1), weights, residual, fitted.b,
                                    fitted.positive_part, fitted.direct_contrast, True)
        prepared = prepare_g_fits(self.fits, {"x_mean_degree": 2, "x_mean_h_multiplier": 1.5})
        expected = float(reference.pdf(1.5)[0] + reference.pdf(-1.5)[0])
        self.assertEqual(prepared.evaluate_fold("x", 0, 0., 1.5), expected)

    def test_invalid_options_are_rejected(self):
        for options in (
            {"misspelled_option": 1}, {"x_density_mix": -0.1}, {"x_density_mix": 1.1},
            {"x_mean_h_multiplier": 0}, {"response_bw_multiplier": math.nan},
            {"x_hf_multiplier": math.inf}, {"x_mean_degree": True}, {"x_mean_degree": 3},
            {"apply_if": "wrong"}, {"skew_threshold": math.inf}, [],
        ):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                validate_options(options)


if __name__ == "__main__":
    unittest.main()
