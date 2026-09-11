"""Focused checks: python -m unittest discover -s pretrain -p test_pretraining_post.py."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import DEFAULT, patch

import numpy as np
import torch

import pretraining_post as post
import _bootstrap  # noqa: F401
from net_helpers import BaseNetwork


class CliTests(unittest.TestCase):
    def run_cli(self, argv):
        names = ("run_evaluation", "run_m_intervention", "plot_memory_pca",
                 "plot_accuracies", "plot_rule_cue", "plot_rule_cue_errors",
                 "plot_m_intervention", "load_accuracies")
        with patch.multiple(post, **{name: DEFAULT for name in names}) as mocks:
            mocks["run_evaluation"].return_value = ([], [])
            mocks["run_m_intervention"].return_value = ([], [])
            post.main(argv)
        return mocks

    def test_default_runs_all_experiments(self):
        mocks = self.run_cli([])
        self.assertEqual(mocks["run_evaluation"].call_count, 2)
        calls = mocks["run_evaluation"].call_args_list
        self.assertFalse(calls[0].args[0].rule_cue)
        self.assertTrue(calls[1].args[0].rule_cue)
        self.assertEqual(mocks["plot_memory_pca"].call_count, len(post.GROUPS))
        for name in ("run_m_intervention", "plot_accuracies", "plot_rule_cue",
                     "plot_rule_cue_errors", "plot_m_intervention"):
            mocks[name].assert_called_once()
        mocks["load_accuracies"].assert_not_called()

    def test_explicit_modes_are_isolated(self):
        expected = {
            "accuracy": {"run_evaluation", "plot_accuracies"},
            "memory-pca": {"plot_memory_pca"},
            "rule-cue": {"run_evaluation", "plot_rule_cue", "plot_rule_cue_errors"},
            "m-intervention": {"run_m_intervention", "plot_m_intervention"},
            "plot-only": {"load_accuracies", "plot_accuracies"},
        }
        for flag, names in expected.items():
            with self.subTest(flag=flag):
                mocks = self.run_cli([f"--{flag}"])
                self.assertEqual({name for name, mock in mocks.items() if mock.called}, names)

    def test_shared_filters_without_mode_still_run_all(self):
        mocks = self.run_cli(["--seed", "134", "--ruleset", "fdanti_delaygo", "--n-trials", "12"])
        mocks["plot_memory_pca"].assert_called_once()
        self.assertEqual(mocks["plot_memory_pca"].call_args.args[-3:], (134, 0, "fdanti_delaygo"))
        for name in ("run_evaluation", "run_m_intervention"):
            for call in mocks[name].call_args_list:
                self.assertEqual((call.args[0].seed, call.args[0].ruleset, call.args[0].n_trials),
                                 (134, "fdanti_delaygo", 12))

    def test_default_continues_after_failure_but_reports_it(self):
        with patch.object(post, "_run_analysis", side_effect=[ValueError("failed accuracy"), None, None, None]) as run:
            with self.assertRaisesRegex(SystemExit, "failed accuracy"):
                post.main([])
            self.assertEqual(run.call_count, 4)

    def test_conflicting_experiment_flags_are_rejected(self):
        with patch.object(post, "_run_analysis") as run:
            with self.assertRaises(SystemExit) as error:
                post.main(["--accuracy", "--rule-cue"])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


class ApplyRuleCueTests(unittest.TestCase):
    def setUp(self):
        self.epochs = {"stim1": (2, 5), "delay1": (5, 8)}
        self.inputs = torch.ones((3, 10, 9))
        self.channel = 7  # rule_column 1 in a 9-channel input with 3 cue columns

    def test_gating_boundaries(self):
        expectations = {
            "intact": np.ones(10, dtype=bool),
            "off_after_context": np.arange(10) < 2,
            "off_after_stimulus": np.arange(10) < 5,
            "off_after_memory": np.arange(10) < 8,
            "on_only_response": np.arange(10) >= 8,
        }
        self.assertEqual(set(expectations), set(post.CUE_CONDITIONS))
        for condition, keep in expectations.items():
            with self.subTest(condition=condition):
                altered = post.apply_rule_cue(self.inputs, self.epochs, 1, condition)
                np.testing.assert_array_equal(altered[:, :, self.channel].numpy(),
                                              np.tile(keep, (3, 1)).astype(np.float32))
                others = torch.ones(9, dtype=torch.bool)
                others[self.channel] = False
                self.assertTrue(torch.equal(altered[:, :, others], self.inputs[:, :, others]))

    def test_conditions_needing_delay_return_none_without_it(self):
        epochs = {"stim1": (2, 5)}
        self.assertIsNone(post.apply_rule_cue(self.inputs, epochs, 1, "off_after_memory"))
        self.assertIsNone(post.apply_rule_cue(self.inputs, epochs, 1, "on_only_response"))
        self.assertIsNotNone(post.apply_rule_cue(self.inputs, epochs, 1, "off_after_stimulus"))
        with self.assertRaises(ValueError):
            post.apply_rule_cue(self.inputs, self.epochs, 1, "bogus")


class TraceVsMTests(unittest.TestCase):
    def test_rowwise_pearson(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        b = torch.tensor([[2.0, 4.0, 6.0], [3.0, 2.0, 1.0], [5.0, 6.0, 7.0]])
        result = post._rowwise_pearson(a, b)
        torch.testing.assert_close(result, torch.tensor([1.0, -1.0, 0.0]))

    def test_plot_requires_trace_stats_and_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            stale = [{"ruleset": "fdanti_delaygo", "aname": "old", "task": {}}]
            self.assertIsNone(post.plot_m_trace_similarity(stale, Path(directory), "f", 200))
            runs = []
            for index, ruleset in enumerate(post.GROUPS):
                runs.append({"ruleset": ruleset, "aname": f"synthetic{index}", "task": {
                    "trace_vs_m": {
                        "mean_abs_trace": 0.2 + index, "mean_abs_m_intact": 0.3 + index,
                        "mean_abs_l1_distance": 0.1 + index,
                        "mean_corr_trace_m_intact": 0.5 - index,
                        "std_corr_trace_m_intact": 0.1,
                        "mean_corr_trace_m_nocue": -0.2, "std_corr_trace_m_nocue": 0.1,
                    }}})
            path = post.plot_m_trace_similarity(runs, Path(directory), "synthetic", 200)
            image = post.plt.imread(path)
            self.assertGreater(float(image.std()), 0.01)


class _ToyPlasticLayer:
    """M grows by +1 everywhere per step; bounds clamp overrides at ±10."""

    def __init__(self):
        self.W = torch.zeros(2, 3)
        self.modulation_bounds = True
        self.M_bounds = torch.stack((torch.full((2, 3), 10.0),
                                     torch.full((2, 3), -10.0)))

    def reset_state(self, B):
        self.M = torch.zeros(B, 2, 3)


class _ToyModel:
    n_output = 1

    def __init__(self):
        self.mp_layers = [_ToyPlasticLayer()]

    def reset_state(self, B):
        self.mp_layers[0].reset_state(B)

    def network_step(self, current_input, run_mode, seq_idx):
        layer = self.mp_layers[0]
        output = layer.M.sum(dim=(1, 2), keepdim=False)[:, None]
        layer.M = layer.M + 1
        return output, None, None


class RolloutMInterventionTests(unittest.TestCase):
    """Output at step t reads M before its update, so out = 6t for the toy."""

    def setUp(self):
        self.model = _ToyModel()
        self.inputs = torch.zeros(4, 6, 5)
        self.boundaries = torch.tensor([2, 3, 4, 4])

    def test_capture_reads_pre_response_state(self):
        outputs, captured = post._rollout_m_intervention(
            self.model, self.inputs, self.boundaries, "cpu", batch_size=3,
            capture_m=True)
        steps = torch.arange(6, dtype=torch.float)
        torch.testing.assert_close(outputs, (6 * steps).expand(4, 6)[:, :, None])
        for trial, boundary in enumerate(self.boundaries.tolist()):
            torch.testing.assert_close(captured[trial],
                                       torch.full((2, 3), float(boundary)))

    def test_override_applies_at_boundary_and_is_clamped(self):
        override = torch.full((4, 2, 3), 100.0)
        outputs, captured = post._rollout_m_intervention(
            self.model, self.inputs, self.boundaries, "cpu", batch_size=3,
            m_override=override, capture_m=True)
        for trial, boundary in enumerate(self.boundaries.tolist()):
            torch.testing.assert_close(captured[trial],
                                       torch.full((2, 3), float(boundary)))
            expected = torch.arange(6, dtype=torch.float) * 6
            # From the boundary on, M restarts from the clamped override (10).
            post_steps = torch.arange(6 - boundary, dtype=torch.float)
            expected[boundary:] = (10 + post_steps) * 6
            torch.testing.assert_close(outputs[trial, :, 0], expected)

    def test_boundary_outside_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            post._rollout_m_intervention(self.model, self.inputs,
                                         torch.tensor([2, 3, 4, 6]), "cpu", 3)
        with self.assertRaises(ValueError):
            post._rollout_m_intervention(self.model, self.inputs,
                                         torch.tensor([0, 3, 4, 4]), "cpu", 3)


class ResponseDirectionTests(unittest.TestCase):
    def setUp(self):
        self.masks = np.ones((4, 20, 3), dtype=np.float32)
        self.masks[:, :2] = 0
        self.masks[:, 6:8] = 0
        self.masks[0, 16:] = 0
        self.masks[1, 18:] = 0
        self.targets = np.zeros_like(self.masks)
        self.targets[:, :, 2] = 1
        self.outputs = self.targets.copy()
        self.outputs[1, :, 2] = -1
        self.outputs[2] = 0
        self.outputs[3, :, 1] = 1
        self.outputs[3, :, 2] = 0

    def diagnose(self, outputs=None):
        return post.response_direction_diagnostics(
            self.outputs if outputs is None else outputs, self.targets, self.masks)

    def test_categories_windows_and_histogram(self):
        report = self.diagnose()
        self.assertEqual(report["scoring_windows_steps"], [[10, 16], [10, 18], [11, 20], [11, 20]])
        self.assertEqual(report["n_timepoints"], 32)
        np.testing.assert_allclose([report[key] for key in post.ERROR_CATEGORIES],
                                   np.array([6, 9, 8, 9]) / 32 * 100)
        self.assertAlmostEqual(sum(report[key] for key in post.ERROR_CATEGORIES), 100)
        self.assertEqual(sum(report["angle_histogram_counts"]), 23)
        self.assertEqual(report["sector_half_width_deg"], 22.5)
        self.assertEqual(report["amplitude_threshold"], 0.15)

    def test_each_timepoint_has_exactly_one_category(self):
        for vector, expected in [([0, 1], "target_pct"), ([0, -1], "pro_pct"),
                                 ([1, 0], "other_pct"), ([0, 0], "low_amplitude_pct"),
                                 ([0, 0.05], "low_amplitude_pct"),
                                 ([0, 0.149], "low_amplitude_pct"), ([0, 0.15], "target_pct")]:
            with self.subTest(vector=vector):
                outputs = np.zeros_like(self.outputs)
                outputs[:, :, 1:3] = vector
                report = self.diagnose(outputs)
                for key in post.ERROR_CATEGORIES:
                    self.assertEqual(report[key], 100 if key == expected else 0)

    def test_low_amplitude_and_rotation(self):
        report = self.diagnose(np.zeros_like(self.outputs))
        self.assertEqual(report["low_amplitude_pct"], 100)
        self.assertIsNone(report["mean_angular_error_deg"])
        self.assertIsNone(report["angular_resultant_length"])
        self.assertEqual(sum(report["angle_histogram_counts"]), 0)
        json.dumps(report, allow_nan=False)
        baseline = self.diagnose()
        angle = np.deg2rad(179)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.outputs[:, :, 1:3] = self.outputs[:, :, 1:3] @ rotation
        self.targets[:, :, 1:3] = self.targets[:, :, 1:3] @ rotation
        rotated = self.diagnose()
        for key in post.ERROR_CATEGORIES:
            self.assertAlmostEqual(rotated[key], baseline[key])
        with self.assertRaises(ValueError):
            post.response_direction_diagnostics(self.outputs, self.targets, self.masks, n_directions=9)

    def test_batching_preserves_actual_accuracy_and_raw_outputs(self):
        model = SimpleNamespace(loss_type="MSE", prefs=torch.arange(8) * (2 * torch.pi / 8))
        model.compute_acc = lambda *args, **kwargs: BaseNetwork.compute_acc(model, *args, **kwargs)
        calls = []
        outputs = torch.from_numpy(self.outputs.copy())

        def forward(batch, run_mode):
            self.assertEqual(run_mode, "minimal")
            calls.append(len(batch))
            return outputs[batch[:, 0, 0].long()].clone(), None, None

        model.iterate_sequence_batch = forward
        inputs = torch.zeros((4, 20, 9))
        inputs[:, :, 0] = torch.arange(4)[:, None]
        scoring_inputs = torch.zeros_like(inputs)
        scoring_inputs[:, :, 0] = 1
        scoring_inputs[:, :, -1] = 1
        targets, masks = torch.from_numpy(self.targets), torch.from_numpy(self.masks)
        expected, _ = model.compute_acc(outputs.clone(), targets, masks, scoring_inputs, isvalid=True)
        report = {}
        value = post._evaluate_inputs(model, inputs, targets, masks, scoring_inputs, 3, "cpu",
                                       direction_report=report)
        self.assertEqual(value, float(expected))
        self.assertEqual(calls, [3, 1])
        self.assertEqual(report, self.diagnose())
        np.testing.assert_array_equal(outputs.numpy(), self.outputs)

    def test_seed_balanced_summary_and_plot(self):
        runs = []
        for seed, outputs in enumerate((self.targets, -self.targets)):
            report = self.diagnose(outputs)
            conditions = {condition: {"accuracy_pct": 100 - seed * 100,
                                      "delta_accuracy_pp": 0, "error_direction": report}
                          for condition in post.CUE_CONDITIONS}
            runs.append({"ruleset": "fdanti_delaygo", "seed": seed, "aname": f"synthetic{seed}",
                         "tasks": [{"task": "delayanti", "conditions": conditions}]})
        summary = post.summarize_rule_cue(runs)
        stats = summary["fdanti_delaygo"]["delayanti"]["intact"]["error_direction"]
        self.assertEqual(stats["target_pct"], {"n_seeds": 2, "mean": 50, "std": 50})
        self.assertEqual(stats["pro_pct"], {"n_seeds": 2, "mean": 50, "std": 50})
        self.assertEqual(stats["other_pct"], {"n_seeds": 2, "mean": 0, "std": 0})
        json.dumps(summary, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            path = post.plot_rule_cue_errors(runs, Path(directory), "synthetic", 200)
            image = post.plt.imread(path)
            self.assertGreater(float(image.std()), 0.01)


if __name__ == "__main__":
    unittest.main()