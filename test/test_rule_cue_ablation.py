"""Tests for the rule-cue ablation of the clustering features."""

import unittest
from types import SimpleNamespace

import _bootstrap  # noqa: F401
import numpy as np
import torch

import rule_cue_ablation as ablation


class RuleCueTest(unittest.TestCase):
    def test_without_rule_cue_zeroes_only_the_rule_channels(self):
        inputs = torch.randn(3, 4, 7)
        ablated = ablation.without_rule_cue(inputs, slice(4, 7))
        self.assertTrue(torch.all(ablated[..., 4:] == 0))
        self.assertTrue(torch.equal(ablated[..., :4], inputs[..., :4]))
        self.assertFalse(torch.all(inputs[..., 4:] == 0))

    def test_rule_channels_end_the_input(self):
        params = {"randomize_inputs": False, "hp": {"rule_start": 6, "n_rule": 15}}
        self.assertEqual(ablation.rule_channels(params, 21), slice(6, 21))
        with self.assertRaises(ValueError):
            ablation.rule_channels(params, 22)

    def test_first_stimulus_modality_reads_the_rings(self):
        rings = ablation.ring_channels({"fixate_off": True, "rule_start": 6, "num_ring": 2})
        self.assertEqual(rings, [slice(2, 4), slice(4, 6)])
        inputs = torch.zeros(3, 6, 6)
        inputs[0, 2:4, 2:4] = 1.0   # rule 0: ring 1 in stim1
        inputs[1, 4:, 4:6] = 1.0    # rule 1: ring 2 in go1 (no stim1)
        inputs[2, 2:4, 2:6] = 1.0   # rule 2: both rings in stim1
        trials = [SimpleNamespace(epochs={"stim1": (2, 4), "go1": (4, None)}),
                  SimpleNamespace(epochs={"go1": (4, None)}),
                  SimpleNamespace(epochs={"stim1": (2, 4), "go1": (4, None)})]
        modality = ablation.first_stimulus_modality(inputs, np.array([0, 1, 2]), trials, rings)
        self.assertEqual(modality, {0: "1", 1: "2", 2: "both"})


class RowWindowsTest(unittest.TestCase):
    def test_rows_are_period_major_and_skip_missing_epochs(self):
        trials = [
            SimpleNamespace(epochs={"fix1": (None, 5), "stim1": (5, 10), "go1": (10, None)}),
            SimpleNamespace(epochs={"fix1": (None, 5), "go1": (np.array([5, 5]), None)}),
        ]
        self.assertEqual(ablation.row_windows(trials),
                         [(0, "stim1", (5, 10)), (0, "go1", (10, None)), (1, "go1", (5, None))])


class ReadoutTest(unittest.TestCase):
    def test_normalize_columns_matches_the_clustering(self):
        features = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]])
        expected = np.array([[1 / 3, 0.0, 0.5], [1.0, 0.0, 1.0]])
        np.testing.assert_allclose(ablation.normalize_columns(features), expected)

    def test_pair_means_sorts_pairs_by_kind(self):
        # Rows: task 0 stim1, task 1 stim1, task 0 go1, task 1 go1.
        correlation = np.array([[1.0, 0.2, 0.9, 0.1],
                                [0.2, 1.0, 0.0, 0.7],
                                [0.9, 0.0, 1.0, 0.4],
                                [0.1, 0.7, 0.4, 1.0]])
        means = ablation.pair_means(correlation, [0, 1, 0, 1],
                                    ["stim1", "stim1", "go1", "go1"])
        self.assertAlmostEqual(means["same_task"]["mean"], 0.8)
        self.assertAlmostEqual(means["same_period"]["mean"], 0.3)
        self.assertAlmostEqual(means["other"]["mean"], 0.05)
        self.assertEqual([means[kind]["n"] for kind in ablation.PAIR_KINDS], [2, 2, 2])

    def test_relative_row_distance_averages_to_one(self):
        rows = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
        distance = ablation.relative_row_distance(rows)
        np.testing.assert_allclose(distance[np.triu_indices(3, k=1)].mean(), 1.0)
        np.testing.assert_allclose(distance[0, 1] / distance[0, 2], 5.0)
        np.testing.assert_allclose(np.diag(distance), 0.0)

    def test_matched_rows_of_a_rescaled_copy_correlate_perfectly(self):
        first = np.random.default_rng(0).random((4, 10))
        np.testing.assert_allclose(ablation.matched_row_correlation(first, 3 * first + 1), 1.0)


if __name__ == "__main__":
    unittest.main()
