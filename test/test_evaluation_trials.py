"""Tests for the clustering analysis' evaluation trials: one reproducible
trial set per network, with per-trial timing and both stimulus modalities
per task."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np

import _bootstrap  # noqa: F401
import multiple_task_analysis


class EvaluationTrialSeedTests(unittest.TestCase):
    def test_seed_is_a_reproducible_function_of_the_network(self):
        seed = multiple_task_analysis.evaluation_trial_seed(427)
        self.assertEqual(seed, multiple_task_analysis.evaluation_trial_seed("427"))
        self.assertIsInstance(seed, int)
        self.assertTrue(0 <= seed < 2 ** 32)          # a valid RandomState seed

    def test_every_network_gets_its_own_seed(self):
        seeds = [multiple_task_analysis.evaluation_trial_seed(run, init)
                 for run in (11, 427, 749) for init in (None, 0, 1, 2)]
        self.assertEqual(len(set(seeds)), len(seeds))


class GenerateEvaluationTrialsTests(unittest.TestCase):
    def test_generator_is_reseeded_with_per_trial_timing_and_both_modalities(self):
        task_params = {"hp": {"seed": 1, "rng": np.random.RandomState(1)},
                       "rules": ["delaygo", "delayanti"]}
        with patch.object(multiple_task_analysis.mpn_tasks, "generate_trials_wrap",
                          return_value="trials") as generate:
            result = multiple_task_analysis.generate_evaluation_trials(task_params, 99)
        self.assertEqual(result, "trials")
        self.assertEqual(task_params["hp"]["seed"], 99)
        np.testing.assert_array_equal(
            task_params["hp"]["rng"].rand(5), np.random.RandomState(99).rand(5))
        n_per_task = multiple_task_analysis.EVALUATION_TRIALS_PER_TASK
        self.assertEqual(n_per_task % 2, 0)
        self.assertEqual(task_params["hp"]["batch_size_train"], n_per_task)
        generate.assert_called_once_with(
            task_params, n_per_task, rules=["delaygo", "delayanti"], mode_input="random_batch",
            device="cpu", verbose=False, balanced_stimulus_modalities=True)

    def test_the_analysis_reads_its_trials_from_the_seeded_generator(self):
        source = inspect.getsource(multiple_task_analysis.main)
        self.assertIn("evaluation_trial_seed(seed, untrained_init_seed)", source)
        self.assertIn("generate_evaluation_trials(task_params_c, trial_seed)", source)
        self.assertNotIn("generate_trials_wrap", source)
        # Both cluster caches record the trial set beside the variance method.
        self.assertEqual(source.count("**evaluation_provenance"), 2)
        self.assertIn('"evaluation_timing": "per_trial"', source)
        # Every (task, period) row is the pooled variance of per-trial windows.
        self.assertIn("trial_epoch_windows(trial, phase)", source)
        self.assertEqual(source.count("period_variance(clustering_data, test_task, rule_idx, "
                                      "period_time)"), 2)
        self.assertNotIn("task_variance_period_numpy", source)

    def test_period_variance_pools_one_rules_trials_in_their_own_windows(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(6, 8, 2))
        test_task = np.array([0, 0, 1, 1, 1, 0])
        windows = (np.array([1, 0, 3]), np.array([5, 2, 8]))      # rule 1's three trials
        expected = np.concatenate([values[2, 1:5], values[3, 0:2], values[4, 3:8]]).var(axis=0)
        np.testing.assert_allclose(
            multiple_task_analysis.period_variance(values, test_task, 1, windows), expected)


if __name__ == "__main__":
    unittest.main()
