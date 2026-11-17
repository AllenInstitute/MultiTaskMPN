"""Regression checks for trial-time variance, epoch bounds and cache provenance."""

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import helper
from core.task_variance import (VARIANCE_METHOD, NEURON_VARIANTS, LESION_MODULATION_TYPES,
                                trial_epoch_bounds, trial_epoch_windows,
                                variance_cache_is_current,
                                require_current_variance, cluster_caches_current)


class TaskVarianceTests(unittest.TestCase):
    def test_trial_differences_survive_when_the_mean_trajectory_is_zero(self):
        h = np.array([[[-1.], [1.]], [[1.], [-1.]]])
        self.assertEqual(h.mean(axis=0).var(), 0)
        np.testing.assert_array_equal(helper.task_variance_period_numpy(h), [1.])

    def test_time_variation_common_to_every_trial_is_retained(self):
        h = np.tile(np.array([[[0.], [2.]]]), (3, 1, 1))
        self.assertEqual(h.var(axis=0).sum(), 0)
        np.testing.assert_array_equal(helper.task_variance_period_numpy(h), [1.])

    def test_trials_have_equal_weight_without_reweighting_stimulus_bins(self):
        h = np.array([0., 0., 0., 4.]).reshape(4, 1, 1)
        np.testing.assert_array_equal(helper.task_variance_period_numpy(h), [3.])
        np.testing.assert_array_equal(helper.task_variance_period_numpy(h, ddof=1), [4.])
        # A legacy positional stimulus-label argument cannot be silently ignored.
        with self.assertRaises(TypeError):
            helper.task_variance_period_numpy(h, np.array([0, 0, 0, 1]))

    def test_large_offset_does_not_destroy_small_float32_variance(self):
        h = np.array([100000., 100000.125, 100000.25], dtype=np.float32).reshape(1, 3, 1)
        np.testing.assert_allclose(helper.task_variance_period_numpy(h), [1 / 96], rtol=1e-12)

    def test_empty_or_unflattened_epochs_fail(self):
        for shape in ((0, 2, 3), (2, 0, 3), (2, 3), (2, 3, 4, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                helper.task_variance_period_numpy(np.zeros(shape))


class TrialWindowTests(unittest.TestCase):
    """Per-trial timing: each trial has its own window of a period."""

    def test_windows_follow_each_trial_and_stop_at_its_own_end(self):
        trial = SimpleNamespace(batch_size=3, tdim=np.array([6, 4, 5]), epochs={
            "fix1": (None, np.array([2, 1, 2])), "go1": (np.array([4, 2, 3]), None),
            "stim1": (2, 3)})
        np.testing.assert_array_equal(trial_epoch_windows(trial, "fix1"), ([0, 0, 0], [2, 1, 2]))
        np.testing.assert_array_equal(trial_epoch_windows(trial, "go1"), ([4, 2, 3], [6, 4, 5]))
        np.testing.assert_array_equal(trial_epoch_windows(trial, "stim1"), ([2, 2, 2], [3, 3, 3]))
        # Aligned batches (scalar length and edges) give the same window per trial.
        aligned = SimpleNamespace(batch_size=2, tdim=7, epochs={"go1": (3, None)})
        np.testing.assert_array_equal(trial_epoch_windows(aligned, "go1"), ([3, 3], [7, 7]))
        trial.epochs["go1"] = (np.array([4, 5, 3]), None)          # starts after trial 2 ends
        with self.assertRaisesRegex(ValueError, "exceeds its trial's length"):
            trial_epoch_windows(trial, "go1")
        trial.epochs["go1"] = (np.array([4, 2]), None)
        with self.assertRaisesRegex(ValueError, "expected 3 trials"):
            trial_epoch_windows(trial, "go1")

    def test_pooled_variance_uses_each_trials_own_window(self):
        rng = np.random.default_rng(0)
        h = rng.normal(size=(4, 9, 3))
        starts, ends = np.array([1, 2, 0, 5]), np.array([4, 2, 6, 9])   # trial 2: empty
        pooled = np.concatenate([h[0, 1:4], h[2, 0:6], h[3, 5:9]])
        np.testing.assert_allclose(
            helper.task_variance_trial_windows(h, starts, ends), pooled.var(axis=0))
        np.testing.assert_allclose(
            helper.task_variance_trial_windows(h, starts, ends, ddof=1), pooled.var(axis=0, ddof=1))
        # With one window for every trial it is the aligned estimator.
        np.testing.assert_allclose(
            helper.task_variance_trial_windows(h, np.full(4, 2), np.full(4, 7)),
            helper.task_variance_period_numpy(h[:, 2:7]))
        # `trials` picks one task's rows out of the whole evaluation set.
        np.testing.assert_allclose(
            helper.task_variance_trial_windows(h, [1, 5], [4, 9], trials=[0, 3]),
            np.concatenate([h[0, 1:4], h[3, 5:9]]).var(axis=0))
        # Padding beyond a trial's end never enters.
        padded = np.array([[0., 0., 1., 3., 1000., 1000.], [0., 0., 3., 1000., 1000., 1000.]])[..., None]
        np.testing.assert_array_equal(
            helper.task_variance_trial_windows(padded, [2, 2], [4, 3]), [8. / 9.])

    def test_pooled_variance_rejects_bad_windows(self):
        h = np.zeros((2, 5, 1))
        for starts, ends, message in (([0, 0], [6, 2], "outside the recorded steps"),
                                      ([3, 0], [2, 2], "outside the recorded steps"),
                                      ([1, 2], [1, 2], "at least one"),
                                      ([0], [2], "one length")):
            with self.assertRaisesRegex(ValueError, message):
                helper.task_variance_trial_windows(h, starts, ends)


class EpochBoundsTests(unittest.TestCase):
    def test_none_end_stops_before_other_tasks_padding(self):
        trial = SimpleNamespace(tdim=4, epochs={"go1": (2, None)})
        start, end = trial_epoch_bounds(trial, "go1")
        h = np.array([0., 0., 1., 3., 1000., 1000.]).reshape(1, 6, 1)
        self.assertEqual((start, end), (2, 4))
        np.testing.assert_array_equal(helper.task_variance_period_numpy(h[:, start:end]), [1.])

    def test_aligned_arrays_are_accepted_and_unaligned_ones_rejected(self):
        trial = SimpleNamespace(tdim=np.array([7, 7]), epochs={"go1": (np.array([3, 3]), None)})
        self.assertEqual(trial_epoch_bounds(trial, "go1"), (3, 7))
        trial.epochs["go1"] = (np.array([3, 4]), None)
        with self.assertRaisesRegex(ValueError, "aligned"):
            trial_epoch_bounds(trial, "go1")


class VarianceCacheTests(unittest.TestCase):
    def test_paper_heatmaps_reject_old_variance_pickles(self):
        import paper_plot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            neuron = root / "cluster_info_run.pkl"
            synapse = root / "cluster_info_mod_run.pkl"
            current = {"input_normalized": {"variance_method": VARIANCE_METHOD}}
            with patch.object(paper_plot, "DATA_DIR", root), patch.object(paper_plot, "ANAME", "run"):
                neuron.write_bytes(pickle.dumps({"input_normalized": {}}))
                synapse.write_bytes(pickle.dumps({"modulation_all_normalized": {}}))
                with self.assertRaisesRegex(ValueError, "stale task-variance"):
                    paper_plot._load_cluster_info()
                with patch("builtins.print"):
                    self.assertIsNone(paper_plot._load_cluster_info_mod())
                neuron.write_bytes(pickle.dumps(current))
                self.assertEqual(paper_plot._load_cluster_info(), current)

    def test_overmembership_does_not_render_old_estimator_results(self):
        import paper_plot
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            path = run / "modulation_all_prepost_belonging_run_normalized.pkl"
            path.write_bytes(pickle.dumps({"prepost_belonging_results": [{}]}))
            with patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_find_experiment_dirs", return_value=[run]), \
                    patch.object(paper_plot, "_save_fig") as save, patch("builtins.print") as printed:
                paper_plot.plot_overmembership_norm()
            save.assert_not_called()
            self.assertIn("stale task-variance", str(printed.call_args_list))

    def test_missing_or_mixed_methods_are_rejected(self):
        current = {"input_normalized": {"variance_method": VARIANCE_METHOD}}
        self.assertTrue(variance_cache_is_current(current))
        for payload in ({}, {"input_normalized": {}},
                        {**current, "hidden_normalized": {"variance_method": "s1_mean"}}):
            with self.subTest(payload=payload):
                self.assertFalse(variance_cache_is_current(payload))
                with self.assertRaisesRegex(ValueError, "rerun"):
                    require_current_variance(payload, "fixture")

    def test_completion_requires_two_full_current_pickles(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("neurons.pkl", "synapses.pkl")]
            for path, keys in zip(paths, (NEURON_VARIANTS, LESION_MODULATION_TYPES)):
                path.write_bytes(pickle.dumps({key: {"variance_method": VARIANCE_METHOD} for key in keys}))
            self.assertTrue(cluster_caches_current(paths))
            paths[1].write_bytes(pickle.dumps({"modulation_all_normalized": {"variance_method": VARIANCE_METHOD}}))
            self.assertFalse(cluster_caches_current(paths))
            paths[1].write_bytes(b"unfinished")
            self.assertFalse(cluster_caches_current(paths))
            paths[1].unlink()
            self.assertFalse(cluster_caches_current(paths))


if __name__ == "__main__":
    unittest.main()
