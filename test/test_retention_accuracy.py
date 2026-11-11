"""Regression checks for accuracy at lambda = 0.99, 0.95 and 0.9."""

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
        configs = {"longest": _regular_config(400000),
                   "slowest": _regular_config(40000), "near": _regular_config(2000),
                   "slow": slow, "slow2": slow, "middle": _regular_config(800), "fast": fast,
                   "mislabeled": fast, "no_time_scale": no_time_scale,
                   "mislabeled_middle": fast,
                   "mb2": mb2, "narrow": narrow, "other_l2": slow}
        results = {
            "longest": {"feature": "L21e4lam0.9999", "acc": 0.97},
            "slowest": {"feature": "L21e4lam0.999", "acc": 0.95},
            "slow": {"feature": "L21e4", "acc": 0.9},
            "slow2": {"feature": "L21e4", "acc": 0.8},
            "near": {"feature": "L21e4lam0.98", "acc": 0.88},
            "middle": {"feature": "L21e4lam0.95", "acc": 0.85},
            "fast": {"feature": "L21e4lam0.9", "acc": 0.7},
            # tagged as the 4000 ms cohort but trained at 400 ms
            "mislabeled": {"feature": "L21e4", "acc": 0.99},
            "mislabeled_middle": {"feature": "L21e4lam0.95", "acc": 0.99},
            "no_time_scale": {"feature": "L21e4", "acc": 0.99},
            "mb2": {"feature": "L21e4", "acc": 0.99},
            "narrow": {"feature": "L21e4lam0.9", "acc": 0.99},
            "other_l2": {"feature": "L21e3", "acc": 0.99},
            "relu": {"feature": "L21e4relu", "acc": 0.99},
            "missing_config": {"feature": "L21e4", "acc": 0.99},
        }
        with contextlib.redirect_stdout(io.StringIO()) as notes:
            groups = self.group(configs, results)
        self.assertEqual(list(groups), ["L21e4lam0.9999", "L21e4lam0.999", "L21e4",
                                        "L21e4lam0.98", "L21e4lam0.95", "L21e4lam0.9"])
        self.assertEqual(groups, {"L21e4lam0.9999": [97.0],
                                 "L21e4lam0.999": [95.0], "L21e4": [90.0, 80.0],
                                 "L21e4lam0.98": [88.0],
                                 "L21e4lam0.95": [85.0], "L21e4lam0.9": [70.0]})
        self.assertIn("skipped mislabeled", notes.getvalue())
        self.assertIn("skipped mislabeled_middle", notes.getvalue())
        self.assertIn("skipped no_time_scale", notes.getvalue())
        self.assertIn("skipped missing_config", notes.getvalue())
        self.assertNotIn("relu", notes.getvalue())

    def test_per_task_groups_skip_legacy_and_invalid_scores(self):
        configs = {"slow": _regular_config(4000), "legacy": _regular_config(4000),
                   "middle": _regular_config(800), "fast": _regular_config(400)}
        results = {
            "slow": {"feature": "L21e4", "acc": 0.9,
                     "acc_per_task": {"fdgo": 0.5, "delaygo": None, "bad": float("nan")}},
            "legacy": {"feature": "L21e4", "acc": 0.9},
            "middle": {"feature": "L21e4lam0.95", "acc": 0.8, "acc_per_task": {"fdgo": 0.75}},
            "fast": {"feature": "L21e4lam0.9", "acc": 0.7, "acc_per_task": {"fdgo": 1.0}},
        }
        with contextlib.redirect_stdout(io.StringIO()):
            groups = self.group(configs, results, per_task=True)
        self.assertEqual(groups, {"L21e4lam0.9999": {}, "L21e4lam0.999": {},
                                 "L21e4": {"fdgo": [50.0]},
                                 "L21e4lam0.98": {},
                                 "L21e4lam0.95": {"fdgo": [75.0]},
                                 "L21e4lam0.9": {"fdgo": [100.0]}})

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

    def test_pairwise_holm_covers_each_retention_against_the_reference(self):
        groups = {"L21e4lam0.9999": [100, 101, 102, 103, 104],
                  "L21e4lam0.999": [95, 96, 97, 98, 99],
                  "L21e4": [90, 91, 92, 93, 94],
                  "L21e4lam0.98": [85, 86, 87, 88, 89],
                  "L21e4lam0.95": [80, 81, 82, 83, 84],
                  "L21e4lam0.9": [70, 71, 72, 73, 74]}
        records = paper_plot._retention_pairwise_tests(groups)
        self.assertEqual([label for _, label, _, _ in records],
                         ["0.9999 vs 0.99", "0.999 vs 0.99", "0.99 vs 0.98", "0.99 vs 0.95",
                          "0.99 vs 0.9"])
        np.testing.assert_allclose([p for _, _, p, _ in records], 2 / 252)
        np.testing.assert_allclose([p for _, _, _, p in records], 5 * 2 / 252)
        # Without the 0.999 cohort its pair drops out of the family.
        groups["L21e4lam0.999"] = []
        records = paper_plot._retention_pairwise_tests(groups)
        self.assertTrue(np.isnan(records[1][3]))
        np.testing.assert_allclose([p for _, _, _, p in records[:1] + records[2:]], 4 * 2 / 252)

    def test_task_holm_covers_all_75_tests_jointly(self):
        groups = {tag: {task: list(range(start, start + 5)) for task in paper_plot._TASK_DISPLAY}
                  for tag, start in (("L21e4lam0.9999", 100), ("L21e4lam0.999", 95), ("L21e4", 90),
                                     ("L21e4lam0.98", 85), ("L21e4lam0.95", 80),
                                     ("L21e4lam0.9", 70))}
        records = paper_plot._retention_pairwise_tests(groups, per_task=True)
        self.assertEqual(len(records), 75)
        self.assertEqual({task for task, _, _, _ in records}, set(paper_plot._TASK_DISPLAY))
        np.testing.assert_allclose([p for _, _, p, _ in records], 2 / 252)
        np.testing.assert_allclose([p for _, _, _, p in records], 75 * 2 / 252)

    def test_missing_cohorts_keep_the_remaining_test(self):
        groups = {"L21e4lam0.9999": [], "L21e4lam0.999": [], "L21e4": [90, 91, 92, 93, 94],
                  "L21e4lam0.98": [], "L21e4lam0.95": [], "L21e4lam0.9": [70, 71, 72, 73, 74]}
        records = paper_plot._retention_pairwise_tests(groups)
        valid = [(label, p) for _, label, _, p in records if np.isfinite(p)]
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0][0], "0.99 vs 0.9")
        self.assertAlmostEqual(valid[0][1], 2 / 252)
        self.assertTrue(np.isnan(records[0][2]))
        self.assertTrue(np.isnan(records[2][3]))

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

    def test_overall_plot_six_columns_test_and_registration(self):
        groups = {"L21e4lam0.9999": [100.0, 101.0, 102.0, 103.0, 104.0],
                  "L21e4lam0.999": [95.0, 96.0, 97.0, 98.0, 99.0],
                  "L21e4": [90.0, 91.0, 92.0, 93.0, 94.0],
                  "L21e4lam0.98": [85.0, 86.0, 87.0, 88.0, 89.0],
                  "L21e4lam0.95": [80.0, 81.0, 82.0, 83.0, 84.0],
                  "L21e4lam0.9": [70.0, 71.0, 72.0, 73.0, 74.0]}
        call, console = self._render(paper_plot.plot_retention_accuracy, groups)
        figure, output_path = call.args[:2]
        axis = figure.axes[0]
        self.assertEqual(output_path.name, "multitask_retention_accuracy.png")
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["$\\lambda = 0.9999$\n(400000 ms)",
                          "$\\lambda = 0.999$\n(40000 ms)", "$\\lambda = 0.99$\n(4000 ms)",
                          "$\\lambda = 0.98$\n(2000 ms)",
                          "$\\lambda = 0.95$\n(800 ms)", "$\\lambda = 0.9$\n(400 ms)"])
        self.assertEqual(len(axis.collections), 6)
        np.testing.assert_allclose(axis.lines[0].get_ydata(),
                                   [102.0, 97.0, 92.0, 87.0, 82.0, 72.0])
        p_text = paper_plot._short_p(5 * 2 / 252)
        self.assertEqual(axis.get_title(), "Mann-Whitney, Holm\n" + "\n".join(
            f"{pair}: {p_text}" for pair in ("0.9999 vs 0.99", "0.999 vs 0.99", "0.99 vs 0.98",
                                             "0.99 vs 0.95", "0.99 vs 0.9")))
        self.assertIn("L21e4lam0.9999 n=5, L21e4lam0.999 n=5, L21e4 n=5, L21e4lam0.98 n=5, "
                      "L21e4lam0.95 n=5, L21e4lam0.9 n=5", call.kwargs["extra"])
        self.assertIn("0.99 vs 0.95: Mann-Whitney p=0.00794, Holm-adjusted p=0.0397", console)
        self.assertGreater(figure.get_size_inches()[0], 3.5)
        self.assertIs(paper_plot.FIGURES_BY_MODE["acc_plot"]["retention_accuracy"],
                      paper_plot.plot_retention_accuracy)

    def test_overall_plot_reports_a_missing_cohort(self):
        call, console = self._render(paper_plot.plot_retention_accuracy,
                                     {"L21e4lam0.9999": [], "L21e4lam0.999": [],
                                      "L21e4": [90.0], "L21e4lam0.98": [],
                                      "L21e4lam0.95": [], "L21e4lam0.9": []})
        axis = call.args[0].axes[0]
        self.assertEqual(axis.get_title(), "")
        self.assertEqual(len(axis.collections), 1)
        self.assertIn("No L21e4lam0.9999/L21e4lam0.999/L21e4lam0.98/L21e4lam0.95/L21e4lam0.9 "
                      "performance data", [text.get_text() for text in axis.texts])
        self.assertIn("--merge --feature L21e4lam0.9999 --feature L21e4lam0.999 "
                      "--feature L21e4lam0.98 --feature L21e4lam0.95 --feature L21e4lam0.9",
                      console)

    def test_overall_plot_with_middle_cohort_missing(self):
        call, console = self._render(paper_plot.plot_retention_accuracy,
                                     {"L21e4lam0.9999": [], "L21e4lam0.999": [],
                                      "L21e4": [90, 91, 92, 93, 94],
                                      "L21e4lam0.98": [], "L21e4lam0.95": [],
                                      "L21e4lam0.9": [70, 71, 72, 73, 74]})
        axis = call.args[0].axes[0]
        self.assertEqual(len(axis.collections), 2)
        self.assertEqual(axis.get_title(), "Mann-Whitney, Holm\n0.99 vs 0.9: p=.008")
        self.assertIn("--merge --feature L21e4lam0.9999 --feature L21e4lam0.999 "
                      "--feature L21e4lam0.98 --feature L21e4lam0.95", console)

    def test_task_plot_annotates_holm_p_per_task(self):
        groups = {
            "L21e4lam0.9999": {},
            "L21e4lam0.999": {},
            "L21e4": {"fdgo": [90.0, 91.0, 92.0, 93.0, 94.0], "delaygo": [90.0]},
            "L21e4lam0.98": {},
            "L21e4lam0.95": {"fdgo": [80.0, 81.0, 82.0, 83.0, 84.0]},
            "L21e4lam0.9": {"fdgo": [70.0, 71.0, 72.0, 73.0, 74.0], "delaygo": [60.0]},
        }
        call, console = self._render(paper_plot.plot_retention_task_accuracy, groups)
        figure, output_path = call.args
        self.assertEqual(output_path.name, "multitask_retention_task_accuracy.png")
        self.assertEqual(len(figure.axes), 15)
        panels = dict(zip(paper_plot._TASK_DISPLAY, figure.axes))
        self.assertEqual(panels["fdgo"].get_title(loc="left"), "DelayPro")
        self.assertEqual(panels["fdgo"].texts[0].get_text(),
                         "0.99 vs 0.95: Holm p=.02\n0.99 vs 0.9: Holm p=.02")
        self.assertEqual(len(panels["delaygo"].texts), 0)
        self.assertEqual(panels["fdanti"].texts[0].get_text(), "No data")
        self.assertEqual([label.get_text() for label in figure.axes[-1].get_xticklabels()],
                         ["$\\lambda = 0.9999$\n(400000 ms)",
                          "$\\lambda = 0.999$\n(40000 ms)", "$\\lambda = 0.99$\n(4000 ms)",
                          "$\\lambda = 0.98$\n(2000 ms)",
                          "$\\lambda = 0.95$\n(800 ms)", "$\\lambda = 0.9$\n(400 ms)"])
        self.assertEqual(figure.axes[-1].get_xlabel(), "Modulation retention")
        self.assertIn("DelayPro, 0.99 vs 0.95: Mann-Whitney p=0.00794, "
                      "Holm-adjusted p=0.0159", console)
        self.assertIs(paper_plot.FIGURES_BY_MODE["acc_plot"]["retention_task_accuracy"],
                      paper_plot.plot_retention_task_accuracy)
        figure.canvas.draw()
        annotation = panels["fdgo"].texts[0].get_window_extent(figure.canvas.get_renderer())
        self.assertGreater(annotation.x0, panels["fdgo"].get_window_extent().x1)
        self.assertLess(annotation.x1, figure.bbox.x1)

    def test_empty_results_skip_both_figures(self):
        for plot, per_task in ((paper_plot.plot_retention_accuracy, False),
                               (paper_plot.plot_retention_task_accuracy, True)):
            groups = {tag: {} if per_task else [] for tag, _, _ in paper_plot._RETENTION_SPECS}
            with patch.object(Path, "exists", return_value=True), \
                    patch.object(Path, "open", return_value=io.StringIO("{}")), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_retention_accuracy_groups", return_value=groups), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    contextlib.redirect_stdout(io.StringIO()) as console:
                plot()
            save.assert_not_called()
            self.assertIn("any retention", console.getvalue())


if __name__ == "__main__":
    unittest.main()
