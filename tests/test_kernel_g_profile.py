"""Policy boundaries and safe CLI resume checks for optional kernel-g tuning."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()


from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from conditional_mc.kernel_g_profile import (
    profile_id, resolve_g_options, validate_profile,
)
import recompute_kernel_g_se as recompute


def example_profile():
    return {
        "version": 1, "name": "test-quadratic", "methods": ["semi"],
        "rules": [{"locations": [-0.5, 0.0], "n_min": 1000, "n_max": 4000,
                   "options": {"x_mean_degree": 2, "x_mean_h_multiplier": 1.5,
                               "x_density_mix": 0.5, "apply_if": "right_skew"}}],
    }


class ProfileTests(unittest.TestCase):
    def test_method_sample_and_location_boundaries(self):
        profile = validate_profile(example_profile())
        for n in (1000, 2000, 4000):
            for location in (-0.5, 0.0):
                self.assertEqual(resolve_g_options(profile, "semi", n, location)["x_mean_degree"], 2)
        for method, n, location in (("conf", 2000, 0.0), ("conf_hsemi", 2000, 0.0),
                                    ("semi", 999, 0.0), ("semi", 4001, 0.0),
                                    ("semi", 2000, 0.5)):
            self.assertIsNone(resolve_g_options(profile, method, n, location))
        self.assertIsNone(resolve_g_options(None, "semi", 1000, -0.5))
        selected = resolve_g_options(profile, "semi", 1000, -0.5)
        selected["x_density_mix"] = 0.0
        self.assertEqual(resolve_g_options(profile, "semi", 1000, -0.5)["x_density_mix"], 0.5)

    def test_ambiguous_rules_rejected_at_resolution(self):
        profile = example_profile()
        profile["rules"].append(deepcopy(profile["rules"][0]))
        profile["rules"][1]["n_min"] = 4000
        profile["rules"][1]["n_max"] = 8000
        validated = validate_profile(profile)
        self.assertIsNotNone(resolve_g_options(validated, "semi", 3999, -0.5))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            resolve_g_options(validated, "semi", 4000, -0.5)

    def test_unknown_keys_and_invalid_options_rejected(self):
        for scope in ("profile", "rule", "options"):
            profile = example_profile()
            target = {"profile": profile, "rule": profile["rules"][0],
                      "options": profile["rules"][0]["options"]}[scope]
            target["x_density_mxi"] = 0.5
            with self.subTest(scope=scope), self.assertRaises((ValueError, TypeError)):
                validate_profile(profile)
        for key, value in (("x_density_mix", 1.1), ("x_mean_degree", 3),
                           ("x_mean_h_multiplier", 0.0), ("apply_if", "oracle_right")):
            profile = example_profile()
            profile["rules"][0]["options"][key] = value
            with self.subTest(option=key), self.assertRaises(ValueError):
                validate_profile(profile)

    def test_validation_preserves_caller_and_hash_tracks_parameters(self):
        original = example_profile()
        frozen = deepcopy(original)
        first = validate_profile(original)
        self.assertEqual(original, frozen)
        self.assertEqual(profile_id(first), profile_id(validate_profile(frozen)))
        changed = deepcopy(first)
        changed["rules"][0]["options"]["x_density_mix"] = 1.0
        self.assertNotEqual(profile_id(first), profile_id(changed))


class CliResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        (self.output / "raw").mkdir(parents=True)
        self.config = {"layers": [{}], "seed": 1, "mixture_p": 0.5, "mixture_a": 1.0}
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.profile = validate_profile(example_profile())
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(json.dumps(self.profile))
        pd.DataFrame([{"method": "semi", "n": 1000}]).to_csv(self.source / "batch.csv", index=False)
        pd.DataFrame([{"method": "semi", "g_profile_id": profile_id(self.profile)}]).to_csv(
            self.output / "raw" / "batch.csv", index=False)

    def write_manifest(self):
        project = Path(recompute.__file__).resolve().parent
        implementation = [Path(recompute.__file__).resolve(), project / "conditional_mc/kernel_g.py",
                          project / "conditional_mc/kernel_g_profile.py",
                          project / "conditional_mc/estimators.py", project / "marginal_mc/nuisance.py"]
        identity = {"config": self.config, "g_profile": self.profile,
                    "sources": [str(self.source.resolve())],
                    "implementation_hashes": {str(path.relative_to(project)):
                                              hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in implementation}}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        (self.output / "kernel_g_run.json").write_text(json.dumps({"fingerprint": fingerprint, **identity}))

    def invoke(self, *, processor=None, with_profile=True, summarize_only=False):
        argv = ["recompute_kernel_g_se", "--config", str(self.config_path),
                "--sources", str(self.source),
                "--output", str(self.output), "--jobs", "1"]
        if with_profile:
            argv.extend(["--g-config", str(self.profile_path)])
        if summarize_only:
            argv.append("--summarize-only")
        effect = processor if processor is not None else AssertionError("completed checkpoint must not refit")
        with patch("sys.argv", argv), patch.object(recompute, "collect_frames", return_value=pd.DataFrame()), \
                patch.object(recompute, "inference_table", return_value=pd.DataFrame()), \
                patch.object(recompute, "process_file", side_effect=effect) as process_mock, \
                patch("builtins.print"):
            recompute.main()
        return process_mock

    def test_identical_manifest_skips_completed_batch(self):
        self.write_manifest()
        self.invoke()
        result = recompute.process_file(str(self.source / "batch.csv"),
                                        str(self.output / "raw" / "batch.csv"),
                                        1, 0.5, 1.0, {}, {}, self.profile)
        self.assertTrue(result["ok"])
        self.assertTrue(result["skipped"])

    def test_changed_profile_rejected_despite_equal_row_counts(self):
        self.write_manifest()
        changed = deepcopy(self.profile)
        changed["rules"][0]["options"]["x_density_mix"] = 1.0
        self.profile_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "different run configuration"):
            self.invoke()

    def test_existing_output_without_tuning_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError, "no run provenance"):
            self.invoke()

    def test_incompatible_equal_count_checkpoint_is_recomputed(self):
        self.write_manifest()
        for old_profile in ("different-profile", None):
            row = {"method": "semi"}
            if old_profile is not None:
                row["g_profile_id"] = old_profile
            pd.DataFrame([row]).to_csv(self.output / "raw" / "batch.csv", index=False)
            with self.subTest(old_profile=old_profile):
                processor = self.invoke(processor=lambda *args: {
                    "ok": True, "file": "batch.csv", "rows": 1, "skipped": False,
                })
                processor.assert_called_once()
                self.assertEqual(processor.call_args.args[0], str(self.source / "batch.csv"))
                self.assertEqual(processor.call_args.args[-1], self.profile)

    def test_legacy_baseline_without_manifest_cannot_adopt_new_configuration(self):
        pd.DataFrame([{"method": "semi"}]).to_csv(self.output / "raw" / "batch.csv", index=False)
        with self.assertRaisesRegex(ValueError, "no run provenance"):
            self.invoke(with_profile=False)
        self.assertFalse((self.output / "kernel_g_run.json").exists())

    def test_legacy_baseline_can_still_be_summarized_without_manifest(self):
        pd.DataFrame([{"method": "semi"}]).to_csv(self.output / "raw" / "batch.csv", index=False)
        processor = self.invoke(with_profile=False, summarize_only=True)
        processor.assert_not_called()
        self.assertFalse((self.output / "kernel_g_run.json").exists())


if __name__ == "__main__":
    unittest.main()
