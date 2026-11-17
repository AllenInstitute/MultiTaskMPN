"""Tests for the rule-cue ablation of the clustering features."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
        trials = [SimpleNamespace(batch_size=1, tdim=6, epochs={"stim1": (2, 4), "go1": (4, None)}),
                  SimpleNamespace(batch_size=1, tdim=6, epochs={"go1": (4, None)}),
                  SimpleNamespace(batch_size=1, tdim=6,
                                  epochs={"stim1": (2, 4), "go1": (4, None)})]
        modality = ablation.first_stimulus_modality(inputs, np.array([0, 1, 2]), trials, rings)
        self.assertEqual(modality, {0: "1", 1: "2", 2: "both"})

    def test_first_stimulus_modality_is_judged_trial_by_trial(self):
        rings = ablation.ring_channels({"fixate_off": True, "rule_start": 6, "num_ring": 2})
        inputs = torch.zeros(4, 6, 6)
        inputs[0, 2:4, 2:4] = 1.0   # rule 0, trial 1: ring 1
        inputs[1, 3:5, 4:6] = 1.0   # rule 0, trial 2: ring 2, in its own later window
        inputs[1, 1:3, 2:4] = 1.0   # ... ring 1 carries input only outside that window
        inputs[2, 2:4, 2:4] = 1.0   # rule 1, trial 1: ring 1
        inputs[3, 2:4, 2:6] = 1.0   # rule 1, trial 2: both rings at once
        trials = [SimpleNamespace(batch_size=2, tdim=6,
                                  epochs={"stim1": (np.array([2, 3]), np.array([4, 5]))}),
                  SimpleNamespace(batch_size=2, tdim=6, epochs={"stim1": (2, 4)})]
        modality = ablation.first_stimulus_modality(
            inputs, np.array([0, 0, 1, 1]), trials, rings)
        self.assertEqual(modality, {0: "1/2", 1: "both"})


class RowWindowsTest(unittest.TestCase):
    def test_rows_are_period_major_with_per_trial_windows(self):
        trials = [
            SimpleNamespace(batch_size=2, tdim=np.array([15, 12]), epochs={
                "fix1": (None, 5), "stim1": (5, np.array([10, 8])),
                "go1": (np.array([10, 8]), None)}),
            SimpleNamespace(batch_size=2, tdim=8, epochs={"fix1": (None, 5),
                                                          "go1": (np.array([5, 6]), None)}),
        ]
        rows = ablation.row_windows(trials)
        self.assertEqual([(rule, phase) for rule, phase, _ in rows],
                         [(0, "stim1"), (0, "go1"), (1, "go1")])
        for (_, _, window), expected in zip(rows, (([5, 5], [10, 8]), ([10, 8], [15, 12]),
                                                   ([5, 6], [8, 8]))):
            np.testing.assert_array_equal(window, expected)

    def test_features_preserve_trial_variance_and_exclude_padding(self):
        # Trial 1 ends at step 4, trial 2 at step 3: what follows is padding.
        values = np.array([[0., 0., 1., 3., 1000., 1000.],
                           [0., 0., 3., 1000., 1000., 1000.]])[..., None]
        states = {"input1": values, "hidden1": 2 * values, "M1": values[..., None]}
        trial = SimpleNamespace(batch_size=2, tdim=np.array([4, 3]), epochs={"go1": (2, None)})
        windows = [window for _, _, window in ablation.row_windows([trial])]
        with patch.object(ablation.multiple_task_analysis, "record_tracked_states",
                          return_value=(None, states)):
            features = ablation.task_features(None, None, windows)
        pooled = np.var([1., 3., 3.])
        for name, scale in (("input", 1.), ("hidden", 4.), ("modulation", 1.)):
            np.testing.assert_allclose(features[name], [[scale * pooled]])

    def test_default_trial_seed_is_the_networks_clustering_seed(self):
        own = ablation.multiple_task_analysis.evaluation_trial_seed
        self.assertEqual(ablation.resolve_trial_seed(427), own(427))
        self.assertEqual(ablation.resolve_trial_seed(427, 3), own(427, 3))
        self.assertNotEqual(ablation.resolve_trial_seed(427), ablation.resolve_trial_seed(749))
        self.assertEqual(ablation.resolve_trial_seed(427, 3, trial_seed=2), 2)
        # Only an explicit trial seed is named in the result file.
        default = ablation.result_stem(427, "L21e4")
        self.assertFalse(default.endswith(f"_trials{own(427)}"))
        self.assertEqual(ablation.result_stem(427, "L21e4", trial_seed=0), default + "_trials0")
        self.assertIsNone(ablation.parse_args(["--seed", "427"]).trial_seed)

    def test_trials_are_the_clustering_evaluation_trials(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = ablation.feedback.run_name(7, "L21e4")
            (root / f"param_{name}_param.json").write_text(
                json.dumps({"task_params": {}, "train_params": {}, "net_params": {}}))
            converted = {"rules": ["delaygo"], "hp": {}}
            with patch.object(ablation.feedback, "CHECKPOINT_DIR", root), \
                    patch.object(ablation.mpn_tasks, "convert_and_init_multitask_params",
                                 return_value=(converted, {}, {})), \
                    patch.object(ablation.multiple_task_analysis, "generate_evaluation_trials",
                                 return_value=(("x", "y", "m"), (None, ["trial"], [0, 0]))
                                 ) as generate:
                result = ablation.evaluation_trials(7, "L21e4", trial_seed=3)
        generate.assert_called_once_with(converted, 3)
        self.assertEqual(result[:3], (converted, "x", ["trial"]))
        np.testing.assert_array_equal(result[3], [0, 0])


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
