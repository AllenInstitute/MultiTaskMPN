"""Regression checks for the accuracy comparison across M retentions."""

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

    def test_each_retention_is_tested_against_the_reference(self):
        reference, names, label = paper_plot._retention_reference()
        self.assertEqual((reference, names[reference], label), (2, "0.99", "$\\lambda = 0.99$"))
        self.assertEqual(names, ["0.9999", "0.999", "0.99", "0.98", "0.95", "0.9"])
        columns = [[100, 101, 102, 103, 104], [95, 96, 97, 98, 99], [90, 91, 92, 93, 94],
                   [85, 86, 87, 88, 89], [80, 81, 82, 83, 84], [70, 71, 72, 73, 74]]
        raw, adjusted = paper_plot._reference_column_tests({None: columns}, reference)[None]
        tested = [0, 1, 3, 4, 5]
        np.testing.assert_allclose(raw[tested], 2 / 252)
        np.testing.assert_allclose(adjusted[tested], 5 * 2 / 252)
        self.assertTrue(np.isnan(raw[reference]) and np.isnan(adjusted[reference]))
        # Without the 0.999 cohort its test drops out of the family.
        columns[1] = []
        raw, adjusted = paper_plot._reference_column_tests({None: columns}, reference)[None]
        self.assertTrue(np.isnan(adjusted[1]))
        np.testing.assert_allclose(adjusted[[0, 3, 4, 5]], 4 * 2 / 252)

    def test_task_holm_covers_all_75_tests_jointly(self):
        groups = {tag: {task: list(range(start, start + 5)) for task in paper_plot._TASK_DISPLAY}
                  for tag, start in (("L21e4lam0.9999", 100), ("L21e4lam0.999", 95), ("L21e4", 90),
                                     ("L21e4lam0.98", 85), ("L21e4lam0.95", 80),
                                     ("L21e4lam0.9", 70))}
        _, names, label = paper_plot._retention_reference()
        with contextlib.redirect_stdout(io.StringIO()) as console:
            arguments = paper_plot._task_reference_tests(groups, "L21e4", names, label)
        column_p = arguments["column_p"]
        self.assertEqual(set(column_p), set(paper_plot._TASK_DISPLAY))
        finite = np.concatenate([p_values[np.isfinite(p_values)] for p_values in column_p.values()])
        self.assertEqual(finite.size, 75)
        np.testing.assert_allclose(finite, 75 * 2 / 252)
        self.assertTrue(all(np.isnan(p_values[2]) for p_values in column_p.values()))
        self.assertEqual(console.getvalue().count("Holm-adjusted"), 75)
        self.assertEqual(len(arguments["legend_handles"]), 1)

    def test_missing_cohorts_keep_the_remaining_test(self):
        columns = [[], [], [90, 91, 92, 93, 94], [], [], [70, 71, 72, 73, 74]]
        raw, adjusted = paper_plot._reference_column_tests({None: columns}, 2)[None]
        np.testing.assert_array_equal(np.isfinite(adjusted), [False] * 5 + [True])
        self.assertAlmostEqual(adjusted[5], 2 / 252)
        self.assertAlmostEqual(raw[5], 2 / 252)

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

    @staticmethod
    def _marks(axis):
        """(text, column) of every significance mark drawn on `axis`."""
        return [(text.get_text(), text.xy[0]) for text in axis.texts
                if isinstance(text, paper_plot.mpl.text.Annotation)]

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
                         ["0.9999", "0.999", "0.99", "0.98", "0.95", "0.9"])
        self.assertEqual(axis.get_xlabel(), "Modulation retention $\\lambda$")
        dots = [item for item in axis.collections
                if isinstance(item, paper_plot.mpl.collections.PathCollection)]
        self.assertEqual(len(dots), 6)
        # every column also carries its mean with an SD error bar
        self.assertEqual(len(axis.containers), 6)
        np.testing.assert_allclose(
            [bar.lines[0].get_ydata()[0] for bar in axis.containers],
            [102.0, 97.0, 92.0, 87.0, 82.0, 72.0])
        # the red connector joins the medians
        np.testing.assert_allclose(axis.lines[0].get_ydata(),
                                   [102.0, 97.0, 92.0, 87.0, 82.0, 72.0])
        self.assertEqual(axis.lines[0].get_color(), paper_plot._TREND_LINE_COLOR)
        self.assertLessEqual(len(axis.get_yticks()), paper_plot._STAT_SCATTER_MAX_TICKS + 1)
        # Holm p = 5 * 2 / 252 = 0.04 at every retention but the reference.
        self.assertEqual(self._marks(axis), [("*", 0.0), ("*", 1.0), ("*", 3.0),
                                             ("*", 4.0), ("*", 5.0)])
        self.assertEqual(axis.get_title(), "")
        self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                         ["median", "mean \u00b1 SD",
                          "Holm p < 0.05 / 0.01 / 0.001\nvs $\\lambda = 0.99$ "
                          "(Mann\u2013Whitney)"])
        # Each mark sits above the highest network of its column.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for mark in axis.texts:
            top = axis.transData.transform((mark.xy[0], mark.xy[1]))[1]
            self.assertGreaterEqual(mark.get_window_extent(renderer).y0, top)
        self.assertIn("L21e4lam0.9999 n=5, L21e4lam0.999 n=5, L21e4 n=5, L21e4lam0.98 n=5, "
                      "L21e4lam0.95 n=5, L21e4lam0.9 n=5", call.kwargs["extra"])
        self.assertIn("0.95 vs 0.99: Mann-Whitney p=0.00794, Holm-adjusted p=0.0397", console)
        np.testing.assert_allclose(figure.get_size_inches(), paper_plot._accuracy_figsize(6))
        self.assertLess(figure.get_size_inches()[0], 3.5)
        self.assertIs(paper_plot.FIGURES_BY_MODE["acc_plot"]["retention_accuracy"],
                      paper_plot.plot_retention_accuracy)

    def test_overall_plot_reports_a_missing_cohort(self):
        call, console = self._render(paper_plot.plot_retention_accuracy,
                                     {"L21e4lam0.9999": [], "L21e4lam0.999": [],
                                      "L21e4": [90.0], "L21e4lam0.98": [],
                                      "L21e4lam0.95": [], "L21e4lam0.9": []})
        axis = call.args[0].axes[0]
        self.assertEqual(self._marks(axis), [])
        self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                         ["median", "mean \u00b1 SD"])
        # one network: its dot and its mean, without an SD bar
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(len(axis.containers), 1)
        # the empty cohorts keep their columns
        self.assertEqual(len(axis.get_xticklabels()), 6)
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
        # two cohorts: their dots and their SD bars
        self.assertEqual(len(axis.collections), 4)
        self.assertEqual(len(axis.containers), 2)
        self.assertEqual(self._marks(axis), [("**", 5.0)])      # Holm p = .008
        self.assertIn("--merge --feature L21e4lam0.9999 --feature L21e4lam0.999 "
                      "--feature L21e4lam0.98 --feature L21e4lam0.95", console)

    def test_significance_marks_by_level(self):
        figure, axis = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        marks = [paper_plot._annotate_significance(axis, 0, 0, p)
                 for p in (0.2, 0.04, 0.009, 0.0009)]
        self.assertEqual([mark.get_text() for mark in marks], ["n.s.", "*", "**", "***"])
        self.assertLess(marks[0].get_fontsize(), marks[1].get_fontsize())
        self.assertEqual(marks[0].get_color(), "0.45")
        self.assertIsNone(paper_plot._annotate_significance(axis, 0, 0, np.nan))

    def test_task_plot_marks_holm_p_per_task(self):
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
        # Holm p = 2 * 2 / 252 = .016 in both tested fdgo columns.
        self.assertEqual(self._marks(panels["fdgo"]), [("*", 4.0), ("*", 5.0)])
        self.assertEqual(len(panels["delaygo"].texts), 0)
        self.assertEqual(panels["fdanti"].texts[0].get_text(), "No data")
        self.assertEqual([label.get_text() for label in figure.axes[-1].get_xticklabels()],
                         ["0.9999", "0.999", "0.99", "0.98", "0.95", "0.9"])
        self.assertEqual(figure.get_supxlabel(), "Modulation retention $\\lambda$")
        self.assertIn("DelayPro, 0.95 vs 0.99: Mann-Whitney p=0.00794, "
                      "Holm-adjusted p=0.0159", console)
        self.assertIs(paper_plot.FIGURES_BY_MODE["acc_plot"]["retention_task_accuracy"],
                      paper_plot.plot_retention_task_accuracy)
        self.assertEqual(len(figure.legends), 1)
        self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()[:2]],
                         ["median", "mean \u00b1 SD"])
        self.assertIn("vs $\\lambda = 0.99$", figure.legends[0].get_texts()[2].get_text())
        # fdgo has three cohorts of five runs: three means with their SD bars
        self.assertEqual([len(bar.lines[2]) for bar in panels["fdgo"].containers], [1, 1, 1])
        # a grid of three panels per row, wider than tall
        np.testing.assert_allclose(figure.get_size_inches(), paper_plot._TASK_ACCURACY_FIGSIZE)
        self.assertEqual(figure.axes[0].get_gridspec().get_geometry(), (5, 3))
        # The marks sit between the panel and its title.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        panel = panels["fdgo"].get_window_extent()
        title_box = panels["fdgo"]._left_title.get_window_extent(renderer)
        for mark in panels["fdgo"].texts:
            box = mark.get_window_extent(renderer)
            self.assertGreaterEqual(box.y0, panel.y1)
            self.assertLess(box.x1, figure.bbox.x1)
            self.assertFalse(box.overlaps(title_box))

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
