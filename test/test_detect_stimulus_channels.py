"""Tests for locating the stimulus ring of a gradient-solve template."""

import unittest
from unittest import mock

import numpy as np
import torch

import _bootstrap  # noqa: F401
import grad_fixed_points
import mpn_tasks


def _delaydm_trial(stim_mod, fixate_off):
    """Eight noiseless DelayDM trials, one per direction, on the low-dim
    layout: fixation (and fixation-off), then the modality 1 and 2 rings."""
    n_fixate = 2 if fixate_off else 1
    config = {"dt": 40, "rng": np.random.RandomState(3), "n_eachring": 8,
              "in_out_mode": "low_dim", "fixate_off": fixate_off,
              "n_input": n_fixate + 4, "n_output": 3, "loss_type": "lsq",
              "sigma_x": 0.0, "alpha": 0.2, "get_meta": True}
    return mpn_tasks.delaydm_(
        config, "random", stim_mod, False, True, "long", "normal", "normal",
        "normal", False, batch_size=8, balanced_stim1=True)


def _ring(stim_mod, fixate_off):
    """The (sin, cos) channels `mpn_tasks.Trial.add` writes a modality to."""
    start = (2 if fixate_off else 1) + 2 * (stim_mod - 1)
    return start, start + 1


class DetectStimulusChannelsTests(unittest.TestCase):
    def test_every_direction_pairs_the_ring_after_the_fixation_channels(self):
        for fixate_off in (True, False):
            n_fixate = 2 if fixate_off else 1
            for stim_mod in (1, 2):
                trial = _delaydm_trial(stim_mod, fixate_off)
                stim_on, stim_off = trial.epochs["stim1"]
                self.assertEqual(sorted(trial.meta["stim1"]), list(range(8)))
                if fixate_off:
                    # The fixation-off channel is silent during fixation, so
                    # counting the channels on there finds one fixation
                    # channel, not two.
                    self.assertFalse(trial.x[:stim_on, :, 1].any())
                for k in range(8):
                    with self.subTest(fixate_off=fixate_off,
                                      stim_mod=stim_mod, trial=k):
                        self.assertEqual(
                            grad_fixed_points.detect_stimulus_channels(
                                trial.x[:, k, :], (0, stim_on),
                                (stim_on, stim_off), n_fixate),
                            _ring(stim_mod, fixate_off))

    def test_a_stimulus_channel_among_the_fixation_channels_raises(self):
        template = np.zeros((10, 6))
        template[:4, 0] = 1.0
        template[4:6, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "fixation channel"):
            grad_fixed_points.detect_stimulus_channels(
                template, (0, 4), (4, 6), n_fixate=2)


class DenseStimulusTemplateTests(unittest.TestCase):
    def test_the_dense_stimulus_follows_the_task_layout(self):
        for fixate_off in (True, False):
            trial = _delaydm_trial(1, fixate_off)
            # Direction 2 (90 degrees): only the ring's sin channel is on.
            k = list(trial.meta["stim1"]).index(2)
            data = torch.as_tensor(
                trial.x[:, k:k + 1].transpose(1, 0, 2).copy())
            task_params = {"hp": {"fixate_off": fixate_off}}
            with mock.patch.object(
                    mpn_tasks, "convert_and_init_multitask_params",
                    return_value=(task_params, {}, {})), \
                    mock.patch.object(mpn_tasks, "get_prefs"), \
                    mock.patch.object(
                        mpn_tasks, "generate_trials_wrap",
                        return_value=((data,), (None, [trial], None))):
                template = grad_fixed_points.build_dense_stimulus_template(
                    {"task_params": {}, "train_params": {}, "net_params": {}},
                    "delaydm1", task_seed=0, n_interp=8)

            ring = _ring(1, fixate_off)
            with self.subTest(fixate_off=fixate_off):
                self.assertEqual(template["stim_channels"], ring)
                stim_on, stim_off = trial.epochs["stim1"]
                angles = template["stimulus_angle"]
                expected = np.stack([np.sin(angles), np.cos(angles)], axis=1)
                np.testing.assert_allclose(
                    template["batch"][:, stim_on:stim_off][:, :, list(ring)],
                    np.repeat(expected[:, None, :], stim_off - stim_on,
                              axis=1), atol=1e-6)
                # Every other input is the generated trial's.
                outside = np.ones(template["batch"].shape[1:], dtype=bool)
                outside[stim_on:stim_off, ring[0]:ring[1] + 1] = False
                np.testing.assert_array_equal(
                    template["batch"][:, outside],
                    np.repeat(data.numpy()[:, outside], 8, axis=0))


if __name__ == "__main__":
    unittest.main()
