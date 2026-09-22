"""Tests for opt-in balanced DelayDM stimulus directions."""

import inspect
import unittest

import numpy as np

import _bootstrap  # noqa: F401
import mpn_tasks


def _config(seed=17):
    return {
        "dt": 40,
        "rng": np.random.RandomState(seed),
        "n_eachring": 8,
        "in_out_mode": "low_dim",
        "fixate_off": True,
        "n_input": 6,
        "n_output": 3,
        "loss_type": "lsq",
        "sigma_x": 0.0,
        "alpha": 0.2,
        "get_meta": True,
    }


def _delaydm_trial(config, stim_mod):
    return mpn_tasks.delaydm_(
        config, "random", stim_mod, False, True,
        "long", "normal", "normal", "normal", False,
        batch_size=32, balanced_stim1=True)


class BalancedStimulusGenerationTests(unittest.TestCase):
    def test_every_direction_has_four_trials_and_siblings_are_aligned(self):
        config = _config()
        rng_state = config["rng"].get_state()
        first = _delaydm_trial(config, 1)
        config["rng"].set_state(rng_state)
        second = _delaydm_trial(config, 2)

        directions, counts = np.unique(first.meta["stim1"], return_counts=True)
        np.testing.assert_array_equal(directions, np.arange(8))
        np.testing.assert_array_equal(counts, np.full(8, 4))
        np.testing.assert_array_equal(first.meta["stim1"], second.meta["stim1"])
        np.testing.assert_allclose(
            first.meta["stim1_strs"], second.meta["stim1_strs"])

    def test_nondivisible_trial_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            mpn_tasks._balanced_stimulus_locations(
                np.random.RandomState(1), 30, 8)

    def test_balancing_is_opt_in(self):
        parameter = inspect.signature(
            mpn_tasks.generate_trials_wrap
        ).parameters["balanced_stimulus_directions"]
        self.assertFalse(parameter.default)


if __name__ == "__main__":
    unittest.main()
