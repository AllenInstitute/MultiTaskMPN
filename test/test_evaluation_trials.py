"""Tests for the clustering analysis' evaluation trials: one reproducible
trial set per network, with both stimulus modalities per task."""

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
    def test_generator_is_reseeded_and_both_modalities_are_requested(self):
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
            task_params, n_per_task, rules=["delaygo", "delayanti"], mode_input="random",
            device="cpu", verbose=False, long_all=False, balanced_stimulus_modalities=True)

    def test_the_analysis_reads_its_trials_from_the_seeded_generator(self):
        source = inspect.getsource(multiple_task_analysis.main)
        self.assertIn("evaluation_trial_seed(seed, untrained_init_seed)", source)
        self.assertIn("generate_evaluation_trials(task_params_c, trial_seed)", source)
        self.assertNotIn("generate_trials_wrap", source)
        # Both cluster caches record the trial set beside the variance method.
        self.assertEqual(source.count("**evaluation_provenance"), 2)


if __name__ == "__main__":
    unittest.main()
