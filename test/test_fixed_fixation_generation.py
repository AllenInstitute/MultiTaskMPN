"""Tests for the opt-in fixed-fixation trial transformation."""

from types import SimpleNamespace
import inspect
import unittest

import numpy as np

import _bootstrap  # noqa: F401
import mpn_tasks


def _fake_trial():
    """Small noiseless scalar-timing trial with fixation ending at step 3."""
    trial = SimpleNamespace()
    trial.dt = 40
    trial.batch_size = 2
    trial.tdim = 8
    trial.tdim_max = 8
    trial.epochs = {
        "fix1": (None, 3),
        "stim1": (3, 5),
        "go1": (5, 8),
    }
    trial.x = np.zeros((8, 2, 2), dtype=np.float32)
    trial.x[:3, :, 0] = 1.0
    trial.x[3:5, :, 1] = 2.0
    trial.y = np.zeros((8, 2, 1), dtype=np.float32)
    trial.y[:5, :, 0] = 1.0
    trial.y_loc = -np.ones((8, 2), dtype=np.float32)
    trial.c_mask = np.zeros((8, 2, 1), dtype=np.float32)
    trial.c_mask[2:, :, 0] = 1.0
    return trial


class FixedFixationGenerationTests(unittest.TestCase):
    def test_extends_all_trial_arrays_and_epochs_together(self):
        trial = mpn_tasks._set_trial_fixation_steps(_fake_trial(), 5)

        self.assertEqual(trial.tdim, 10)
        self.assertEqual(trial.x.shape[0], 10)
        self.assertEqual(trial.y.shape[0], 10)
        self.assertEqual(trial.y_loc.shape[0], 10)
        self.assertEqual(trial.c_mask.shape[0], 10)
        self.assertEqual(trial.epochs["fix1"], (None, 5))
        self.assertEqual(trial.epochs["stim1"], (5, 7))
        self.assertEqual(trial.epochs["go1"], (7, 10))
        np.testing.assert_array_equal(trial.x[:5, :, 0], 1.0)
        np.testing.assert_array_equal(trial.x[5:7, :, 1], 2.0)

    def test_crops_all_trial_arrays_and_epochs_together(self):
        trial = mpn_tasks._set_trial_fixation_steps(_fake_trial(), 2)

        self.assertEqual(trial.tdim, 7)
        self.assertEqual(trial.x.shape[0], 7)
        self.assertEqual(trial.epochs["fix1"], (None, 2))
        self.assertEqual(trial.epochs["stim1"], (2, 4))
        self.assertEqual(trial.epochs["go1"], (4, 7))
        np.testing.assert_array_equal(trial.x[:2, :, 0], 1.0)
        np.testing.assert_array_equal(trial.x[2:4, :, 1], 2.0)

    def test_alignment_is_opt_in(self):
        parameter = inspect.signature(
            mpn_tasks.generate_trials_wrap).parameters["fixed_fixation_steps"]
        self.assertIsNone(parameter.default)


if __name__ == "__main__":
    unittest.main()
