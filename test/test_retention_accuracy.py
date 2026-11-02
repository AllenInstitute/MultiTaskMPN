"""Regression checks for the M retention (lambda 0.99 vs 0.9) accuracy figures."""

import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import _bootstrap  # noqa: F401
import paper_plot


def _regular_config(m_time_scale):
    return {
        "task_params": {"ruleset": "everything"},
        "train_params": {"weight_reg": "L2", "reg_lambda": 1e-4,
                         "batch_size": 128, "n_batches": 128},
        "net_params": {"net_type": "dmpn", "n_neurons": [1, 300, 1],
                       "linear_embed": 300, "activation": "tanh",
                       "input_layer_add": True,
                       "ml_params": {"m_time_scale": m_time_scale}},
    }


class RetentionAccuracyTests(unittest.TestCase):
    def group(self, configs, results, **kwargs):
        def open_config(path, *args, **_kwargs):
            name = path.name.removeprefix("param_").removesuffix("_param.json")
            if name not in configs:
                raise FileNotFoundError(name)
            return io.StringIO(json.dumps(configs[name]))

        with patch.object(Path, "open", new=open_config):
            return paper_plot._retention_accuracy_groups(results, **kwargs)

    def test_groups_by_tag_and_checks_the_saved_retention(self):
        slow = _regular_config(4000)
        fast = _regular_config(400)
        no_time_scale = copy.deepcopy(slow)
        del no_time_scale["net_params"]["ml_params"]["m_time_scale"]
        mb2 = copy.deepcopy(slow)
        mb2["net_params"]["ml_params"]["m_bounds"] = [-2.0, 2.0]
        narrow = copy.deepcopy(fast)
        narrow["net_params"]["linear_embed"] = 200
        configs = {"slow": slow, "slow2": slow, "fast": fast,
                   "mislabeled": fast, "no_time_scale": no_time_scale,
                   "mb2": mb2, "narrow": narrow, "other_l2": slow}
        results = {
            "slow": {"feature": "L21e4", "acc": 0.9},
            "slow2": {"feature": "L21e4", "acc": 0.8},
            "fast": {"feature": "L21e4lam0.9", "acc": 0.7},
            # tagged as the 4000 ms cohort but trained at 400 ms
            "mislabeled": {"feature": "L21e4", "acc": 0.99},
            "no_time_scale": {"feature": "L21e4", "acc": 0.99},
            "mb2": {"feature": "L21e4", "acc": 0.99},
            "narrow": {"feature": "L21e4lam0.9", "acc": 0.99},
            "other_l2": {"feature": "L21e3", "acc": 0.99},
            "relu": {"feature": "L21e4relu", "acc": 0.99},
            "missing_config": {"feature": "L21e4", "acc": 0.99},
        }
        with contextlib.redirect_stdout(io.StringIO()) as notes:
            groups = self.group(configs, results)
        self.assertEqual(list(groups), ["L21e4", "L21e4lam0.9"])
        self.assertEqual(groups, {"L21e4": [90.0, 80.0], "L21e4lam0.9": [70.0]})
        self.assertIn("skipped mislabeled", notes.getvalue())
        self.assertIn("skipped no_time_scale", notes.getvalue())
        self.assertIn("skipped missing_config", notes.getvalue())
        self.assertNotIn("relu", notes.getvalue())

    def test_per_task_groups_skip_legacy_and_invalid_scores(self):
        configs = {"slow": _regular_config(4000), "legacy": _regular_config(4000),
                   "fast": _regular_config(400)}
        results = {
            "slow": {"feature": "L21e4", "acc": 0.9,
                     "acc_per_task": {"fdgo": 0.5, "delaygo": None, "bad": float("nan")}},
            "legacy": {"feature": "L21e4", "acc": 0.9},
            "fast": {"feature": "L21e4lam0.9", "acc": 0.7, "acc_per_task": {"fdgo": 1.0}},
        }
        with contextlib.redirect_stdout(io.StringIO()):
            groups = self.group(configs, results, per_task=True)
        self.assertEqual(groups, {"L21e4": {"fdgo": [50.0]}, "L21e4lam0.9": {"fdgo": [100.0]}})

    def test_mann_whitney_needs_two_runs_per_cohort(self):
        self.assertTrue(np.isnan(paper_plot._mann_whitney_p([90.0], [70.0, 71.0])))
        self.assertTrue(np.isnan(paper_plot._mann_whitney_p([], [])))
        separated = paper_plot._mann_whitney_p([90.0, 91.0, 92.0, 93.0, 94.0],
                                                  [70.0, 71.0, 72.0, 73.0, 74.0])
        self.assertAlmostEqual(separated, 2 / 252, places=6)
        self.assertGreater(paper_plot._mann_whitney_p([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.5)

    def test_holm_adjustment(self):
        np.testing.assert_allclose(paper_plot._holm_adjusted([0.01, 0.04, 0.03]),
                                   [0.03, 0.06, 0.06])
        adjusted = paper_plot._holm_adjusted([0.2, np.nan, 0.001])
        np.testing.assert_allclose(adjusted[[0, 2]], [0.2, 0.002])
        self.assertTrue(np.isnan(adjusted[1]))
        self.assertTrue(np.all(np.isnan(paper_plot._holm_adjusted([np.nan]))))
        # step-down monotonicity: the larger raw p inherits the capped smaller one
        np.testing.assert_array_equal(paper_plot._holm_adjusted([0.9, 0.8]), [1.0, 1.0])

    def _render(self, plot, groups):
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "open", return_value=io.StringIO("{}")), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_retention_accuracy_groups", return_value=groups), \
                patch.object(paper_plot, "_save_fig") as save, \
                contextlib.redirect_stdout(io.StringIO()) as console:
            plot()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        return save.call_args, console.getvalue()

    def test_overall_plot_two_columns_test_and_registration(self):
        groups = {"L21e4": [90.0, 91.0, 92.0, 93.0, 94.0],
                  "L21e4lam0.9": [70.0, 71.0, 72.0, 73.0, 74.0]}
        call, _ = self._render(paper_plot.plot_retention_accuracy, groups)
        figure, output_path = call.args[:2]
        axis = figure.axes[0]
        self.assertEqual(output_path.name, "multitask_retention_accuracy.png")
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["$\\lambda = 0.99$\n(4000 ms)", "$\\lambda = 0.9$\n(400 ms)"])
        self.assertEqual(len(axis.collections), 2)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [92.0, 72.0])
        self.assertEqual(axis.get_title(), "Mann-Whitney p = 0.008")
        self.assertIn("L21e4 n=5, L21e4lam0.9 n=5", call.kwargs["extra"])
        self.assertIs(paper_plot.FIGURES_BY_MODE["acc_plot"]["retention_accuracy"],
                      paper_plot.plot_retention_accuracy)

    def test_overall_plot_reports_a_missing_cohort(self):
        call, console = self._render(paper_plot.plot_retention_accuracy,
                                     {"L21e4": [90.0], "L21e4lam0.9": []})
        axis = call.args[0].axes[0]
        self.assertEqual(axis.get_title(), "")
        self.assertEqual(len(axis.collections), 1)
        self.assertIn("No L21e4lam0.9 performance data",
                      [text.get_text() for text in axis.texts])
        self.assertIn("--merge --feature L21e4lam0.9", console)

    def test_task_plot_annotates_holm_p_per_task(self):
        groups = {
            "L21e4": {"fdgo": [90.0, 91.0, 92.0, 93.0, 94.0], "delaygo": [90.0]},
            "L21e4lam0.9": {"fdgo": [70.0, 71.0, 72.0, 73.0, 74.0], "delaygo": [60.0]},
        }
        call, console = self._render(paper_plot.plot_retention_task_accuracy, groups)
        figure, output_path = call.args
        self.assertEqual(output_path.name, "multitask_retention_task_accuracy.png")
        self.assertEqual(len(figure.axes), 15)
        panels = dict(zip(paper_plot._TASK_DISPLAY, figure.axes))
        self.assertEqual(panels["fdgo"].get_title(loc="left"), "DelayPro")
        self.assertEqual(panels["fdgo"].get_title(loc="right"), "Holm p=.008")
        self.assertEqual(panels["delaygo"].get_title(loc="right"), "")
        self.assertEqual(panels["fdanti"].texts[0].get_text(), "No data")
        self.assertEqual([label.get_text() for label in figure.axes[-1].get_xticklabels()],
                         ["$\\lambda = 0.99$\n(4000 ms)", "$\\lambda = 0.9$\n(400 ms)"])
        self.assertEqual(figure.axes[-1].get_xlabel(), "Modulation retention")
        self.assertIn("DelayPro: Mann-Whitney p=0.00794, Holm-adjusted p=0.00794", console)
        self.assertIs(paper_plot.FIGURES_BY_MODE["acc_plot"]["retention_task_accuracy"],
                      paper_plot.plot_retention_task_accuracy)


if __name__ == "__main__":
    unittest.main()
