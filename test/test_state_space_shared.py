"""Shared state-space setup and statistics preserve each experiment's contract."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np

import helper
import state_space_geometry as geometry
import state_space_structure as structure
from test_state_space_structure import _pca_cache


class SharedStatisticsTests(unittest.TestCase):
    def test_matches_existing_ols_and_task_label_permutation(self):
        pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
        distances = np.array([0.5, 1.2, 3., 2., 1.7, 4.])
        angles = np.array([15., 27., 31., 40., 30., 62.])
        r_values, scatter = geometry.distance_angle_statistics(
            pairs, distances, angles, list("abcd"), n_permutations=40, seed=3)
        _, _, r, slope, intercept, p = helper.linear_regression(
            distances, angles, log=False, through_origin=False)
        np.testing.assert_array_equal(r_values, (r, slope, p))
        self.assertEqual(scatter["regression"]["intercept"], intercept)
        expected = geometry.mantel_pearson_permutation(
            geometry.pair_values_to_matrix(distances, pairs, 4),
            geometry.pair_values_to_matrix(angles, pairs, 4), n_permutations=40, seed=3)
        np.testing.assert_array_equal(scatter["permutation"]["null"], expected["null"])
        self.assertEqual(scatter["permutation"]["p_value"], expected["p_value"])
        self.assertEqual(scatter["pair_rules"], [(a, b) for a, b in
                                                (("a", "b"), ("a", "c"), ("a", "d"),
                                                 ("b", "c"), ("b", "d"), ("c", "d"))])

    def test_incomplete_pairs_allow_regression_but_not_label_permutations(self):
        values, scatter = geometry.distance_angle_statistics(
            [(0, 1), (1, 2)], [1., 2.], [20., 40.], list("abc"),
            pair_requirement="have defined response angles in both directions")
        self.assertAlmostEqual(values[0], 1.)
        self.assertIsNone(scatter["permutation"])
        self.assertIn("only 2 of 3", scatter["permutation_skipped"])
        self.assertIn("defined response angles", scatter["permutation_skipped"])
        self.assertIsNone(scatter["regression"]["skipped"])

    def test_degenerate_or_nonfinite_values_are_not_silently_dropped(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        for distances, angles, reason in (
                ([1., 1., 1.], [10., 20., 30.], "constant"),
                ([1., 2., 3.], [10., np.nan, 30.], "undefined angles"),
                ([1., np.nan, 3.], [10., 20., 30.], "undefined distances")):
            with self.subTest(reason=reason):
                values, scatter = geometry.distance_angle_statistics(pairs, distances, angles, list("abc"))
                self.assertTrue(np.isnan(values).all())
                self.assertIsNone(scatter["permutation"])
                self.assertIn(reason, scatter["regression"]["skipped"])
                np.testing.assert_array_equal(scatter["dists"], distances)
                np.testing.assert_array_equal(scatter["angles_deg"], angles)
                self.assertEqual(len(scatter["pair_rules"]), 3)


class StructureReuseTests(unittest.TestCase):
    def test_category_test_reuses_distances_and_preserves_feature_metadata(self):
        data = _pca_cache("fixture", 1e-4, 4)
        record = data["pca_results"]["eff_mod"]
        categories, _ = structure.task_categories(data)
        expected = geometry.category_separation_permutation(
            record["task_centroids"], categories, n_permutations=30, seed=2)
        with patch.object(geometry, "pdist", wraps=geometry.pdist) as distances:
            result = structure.analyze_representation(
                record, data["all_rules"], categories, n_permutations=30, seed=2)
        self.assertEqual(distances.call_count, 1)
        for key, value in expected.items():
            if isinstance(value, (np.ndarray, float, int)):
                np.testing.assert_array_equal(result["separation"][key], value)
            else:
                self.assertEqual(result["separation"][key], value)


class EvaluationImportTests(unittest.TestCase):
    def test_transplant_import_does_not_import_shift_or_select_a_device_or_change_plot_style(self):
        script = """
import _bootstrap
import sys
import torch
import matplotlib as mpl
from unittest.mock import patch
before = dict(mpl.rcParams)
with patch.object(torch.cuda, "is_available", side_effect=AssertionError("device chosen at import")):
    import state_space_evaluation
    import state_space_transplant
assert "state_space_shift" not in sys.modules
assert dict(mpl.rcParams) == before
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).resolve().parent,
            text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
