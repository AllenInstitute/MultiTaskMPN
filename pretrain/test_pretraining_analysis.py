"""Run: python -m unittest discover -s pretrain -p test_pretraining_analysis.py.

Synthetic regression checks only; no checkpoint loading or full analysis runs.
"""

import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import pretraining_analysis as analysis


def eigenvalue_pr(covariance):
    """Reference implementation used before the trace/Frobenius optimization."""
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0, None)
    total = eigenvalues.sum()
    squared_total = np.sum(eigenvalues ** 2)
    return float(total ** 2 / squared_total) if total and squared_total else 0.0


class AnalysisOptimizationTests(unittest.TestCase):
    def setUp(self):
        timing = patch.object(analysis, "TIMING_ENABLED", False)
        timing.start()
        self.addCleanup(timing.stop)
        self.rng = np.random.default_rng(42)

    def test_pr_matches_eigenvalue_reference(self):
        for dtype in (np.float32, np.float64):
            for samples, features in ((40, 6), (6, 40), (12, 1)):
                with self.subTest(dtype=dtype, shape=(samples, features)):
                    data = self.rng.normal(size=(samples, features)).astype(dtype)
                    centered = data - data.mean(axis=0)
                    covariance = centered.T @ centered / samples
                    np.testing.assert_allclose(analysis._pr_from_data(centered),
                                               eigenvalue_pr(covariance), rtol=2e-5, atol=2e-6)
        rank_one = np.outer(np.arange(1.0, 7.0), np.arange(1.0, 7.0))
        self.assertAlmostEqual(analysis._participation_ratio(rank_one), 1.0)
        self.assertAlmostEqual(analysis._participation_ratio(np.eye(6)), 6.0)
        self.assertEqual(analysis._pr_from_data(np.zeros((4, 12))), 0.0)

    def test_pr_scaling_avoids_overflow_and_underflow(self):
        for scale in (1e-250, 1.0, 1e250):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(analysis._participation_ratio(np.eye(5) * scale), 5.0)

    def test_pca_results_and_angles_match_reference_pr_path(self):
        for datatype, shape in (("hidden", (5, 7, 6)),
                                ("modulation", (5, 7, 2, 3)),
                                ("modulation_weighted", (5, 7, 2, 3))):
            for center_on in ("X", "Y"):
                with self.subTest(datatype=datatype, center_on=center_on):
                    source = self.rng.normal(size=shape).astype(np.float32)
                    target = self.rng.normal(size=shape).astype(np.float32) + 2
                    kwargs = {"n_components": 4, "datatype": datatype,
                              "center_on": center_on, "angle_k": 3}
                    with patch.object(analysis, "_participation_ratio", side_effect=eigenvalue_pr):
                        reference = analysis.pca_cross_variance(source, target, **kwargs)
                    actual = analysis.pca_cross_variance(source, target, **kwargs)
                    self.assertEqual(set(actual), set(reference))
                    for key in actual:
                        np.testing.assert_allclose(actual[key], reference[key], rtol=2e-5, atol=2e-6)
                    with patch.object(analysis, "_pr_from_data", side_effect=AssertionError("PR called")), \
                            patch.object(analysis, "_participation_ratio", side_effect=AssertionError("PR called")):
                        curves = analysis.pca_cross_variance(source, target, compute_pr=False, **kwargs)
                    self.assertEqual(set(curves), set(actual) - {"PR_X", "PR_Y", "PR_Y_in_Xbasis"})
                    for key in curves:
                        np.testing.assert_array_equal(curves[key], actual[key])

    def test_direction_averages_skip_pr_without_changing_curves(self):
        source = self.rng.normal(size=(8, 6, 4))
        target = self.rng.normal(size=(8, 6, 4))
        directions = np.tile([0, 1], 4)
        reference = [analysis.pca_cross_variance(source[directions == direction],
                                                target[directions == direction], n_components=3)
                     for direction in range(2)]
        with patch.object(analysis, "_pr_from_data", side_effect=AssertionError("PR called")), \
                patch.object(analysis, "_participation_ratio", side_effect=AssertionError("PR called")):
            actual = analysis.direction_averaged_cve(source, target, directions, directions,
                                                      n_components=3, datatype="hidden", n_dirs=2)
        self.assertEqual(actual["n_dirs_used"], 2)
        np.testing.assert_array_equal(actual["cev_Y_mean"],
                                      np.mean([result["cev_Y"] for result in reference], axis=0))
        np.testing.assert_array_equal(actual["cev_Y_self_mean"],
                                      np.mean([result["cev_Y_self"] for result in reference], axis=0))

    def test_period_slice_content_and_copy_scope(self):
        copied_shapes = []

        class TrackedArray(np.ndarray):
            def __getitem__(self, key):
                if isinstance(key, np.ndarray):
                    copied_shapes.append(self.shape)
                return super().__getitem__(key)

        epochs = {"task": {"stim1": (2, 10)}}
        mask = np.array([True, False, True, False])
        for shape in ((4, 20, 5), (4, 20, 2, 3)):
            data = np.arange(np.prod(shape)).reshape(shape)
            expected = data[mask, :, :][:, 4:10, :]
            actual = analysis.period_slice(data.view(TrackedArray), epochs, "task", "stim1",
                                            shift_percentage=0.25, mask=mask)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(copied_shapes[-1][1], 6)
            view = analysis.period_slice(data, epochs, "task", "stim1", shift_percentage=0.25)
            np.testing.assert_array_equal(view, data[:, 4:10])
            self.assertTrue(np.shares_memory(view, data))

    def test_sanity_check_only_infers_plotted_trials(self):
        inputs = self.rng.normal(size=(25, 6, 9)).astype(np.float32)
        targets = inputs[:, :, :3] * 2
        stage = {"test_input_np": inputs, "test_output_np": targets,
                 "test_task": np.zeros(25, dtype=int),
                 "task_params": np.array({"rules": ["delayanti"]}, dtype=object)}
        batch_sizes = []

        def forward(batch, run_mode):
            self.assertEqual(run_mode, "minimal")
            batch_sizes.append(len(batch))
            return batch[:, :, :3] * 2, None, None

        model = SimpleNamespace(iterate_sequence_batch=forward)
        for n_trials, expected_batches in ((10, [8, 2, 8, 2]),
                                           (30, [8, 8, 8, 1, 8, 8, 8, 1])):
            with self.subTest(n_trials=n_trials), \
                    patch.object(analysis, "load_final_net", return_value=model), \
                    patch.object(analysis, "_plot_input_output_panel") as plot:
                batch_sizes.clear()
                analysis.run_final_net_sanity_check(1, torch.device("cpu"), stage, stage,
                                                     n_trials=n_trials)
                self.assertEqual(batch_sizes, expected_batches)
                self.assertEqual(plot.call_count, 2)
                for call in plot.call_args_list:
                    np.testing.assert_array_equal(call.args[1], inputs[:n_trials])
                    np.testing.assert_array_equal(call.args[2], targets[:n_trials])
                    np.testing.assert_array_equal(call.args[3], targets[:n_trials])
                    np.testing.assert_array_equal(call.args[5], stage["test_task"][:n_trials])

    def test_timing_does_not_change_return_or_swallow_failures(self):
        @analysis._timed_analysis
        def operation(fail=False):
            if fail:
                raise ValueError("expected failure")
            return 42

        output = io.StringIO()
        with patch.object(analysis, "TIMING_ENABLED", True), redirect_stdout(output):
            self.assertEqual(operation(), 42)
            with self.assertRaisesRegex(ValueError, "expected failure"):
                operation(fail=True)
        self.assertEqual(output.getvalue().count("[timing] operation"), 2)


if __name__ == "__main__":
    unittest.main()