"""Checks of the accuracy figures' tests against the default setting."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import _bootstrap  # noqa: F401
import paper_plot

LOW = [70.0, 71.0, 72.0, 73.0, 74.0]
HIGH = [90.0, 91.0, 92.0, 93.0, 94.0]


def _marks(axis):
    """(text, x) of every significance mark drawn on `axis`."""
    return [(text.get_text(), text.xy[0]) for text in axis.texts
            if isinstance(text, paper_plot.mpl.text.Annotation)]


class ReferenceColumnTests(unittest.TestCase):
    def test_holm_is_joint_over_the_panels(self):
        tests = paper_plot._reference_column_tests(
            {"a": [HIGH, LOW, [50.0]], "b": [HIGH, HIGH, LOW]}, 0)
        self.assertEqual(list(tests), ["a", "b"])
        raw_a, adjusted_a = tests["a"]
        raw_b, adjusted_b = tests["b"]
        # reference and the one-run column are not tested
        np.testing.assert_array_equal(np.isfinite(raw_a), [False, True, False])
        np.testing.assert_array_equal(np.isfinite(raw_b), [False, True, True])
        np.testing.assert_allclose([raw_a[1], raw_b[2]], 2 / 252)
        # three valid tests: the two smallest p share 3 * p, the identical
        # groups keep p = 1
        np.testing.assert_allclose([adjusted_a[1], adjusted_b[2]], 3 * 2 / 252)
        self.assertEqual(adjusted_b[1], 1.0)
        self.assertEqual(paper_plot._reference_column_tests({}, 0), {})

    def test_printed_lines_name_the_reference_and_the_panel(self):
        tests = paper_plot._reference_column_tests({"a": [HIGH, LOW, [50.0]]}, 0)
        with contextlib.redirect_stdout(io.StringIO()) as console:
            paper_plot._print_reference_tests(tests, ["ref", "low", "single"], 0, {"a": "Panel A"})
        self.assertEqual(console.getvalue(),
                         "  Panel A, low vs ref: Mann-Whitney p=0.00794, "
                         "Holm-adjusted p=0.00794\n")

    def test_marks_sit_over_each_tested_column(self):
        figure, axis = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(paper_plot._mark_reference_tests(
                axis, [0, 1], [LOW, HIGH], None, ["low", "high"]), {})
            self.assertEqual(_marks(axis), [])
            tests = paper_plot._mark_reference_tests(axis, [0, 1, 2], [LOW, HIGH, []], 1,
                                                     ["low", "high", "none"])
        self.assertEqual(_marks(axis), [("**", 0)])
        self.assertEqual(axis.texts[0].xy, (0, 74.0))
        self.assertEqual(
            [handle.get_label() for handle in
             paper_plot._reference_test_legend_handles(tests, "high", wrap=True)],
            ["Holm p < 0.05 / 0.01 / 0.001\nvs high (Mann\u2013Whitney)"])
        self.assertEqual(paper_plot._reference_test_legend_handles({}, "high"), [])

    def test_run_columns_draw_the_median_in_red(self):
        figure, axis = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        sweep = paper_plot._draw_run_columns(axis, [0, 1, 2], [LOW, [], HIGH], connect=True)
        self.assertEqual([handle.get_label() for handle in sweep], ["median"])
        # empty columns are skipped; the connector joins the medians
        np.testing.assert_allclose(axis.lines[0].get_xydata(), [[0, 72.0], [2, 92.0]])
        self.assertEqual(axis.lines[0].get_color(), paper_plot._TREND_LINE_COLOR)
        self.assertEqual(len(axis.collections), 2)
        np.testing.assert_allclose(axis.collections[0].get_offsets()[:, 0],
                                   np.linspace(-0.1, 0.1, 5))
        dashes = paper_plot._draw_run_columns(axis, [4], [HIGH], colors=["k"])
        self.assertEqual([handle.get_label() for handle in dashes], ["median, IQR"])
        # dots, then the interquartile bar and the median dash of the column
        self.assertEqual(len(axis.collections), 5)
        np.testing.assert_allclose(axis.collections[3].get_segments()[0], [[4, 91.0], [4, 93.0]])
        np.testing.assert_allclose(axis.collections[4].get_segments()[0][:, 1], 92.0)

    def test_run_columns_can_add_the_mean_with_its_sd(self):
        figure, axis = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        handles = paper_plot._draw_run_columns(axis, [0, 1, 2], [LOW, [70.0], []],
                                               connect=True, mean_sd=True)
        self.assertEqual([handle.get_label() for handle in handles],
                         ["median", "mean \u00b1 SD"])
        # the connector first, then one error bar per non-empty column
        self.assertEqual(axis.lines[0].get_color(), paper_plot._TREND_LINE_COLOR)
        self.assertEqual(len(axis.containers), 2)
        offset = paper_plot._ACCURACY_MEAN_OFFSET
        five, single = axis.containers
        np.testing.assert_allclose(five.lines[0].get_xydata(), [[offset, 72.0]])
        # sample SD of 70..74 is sqrt(2.5); one run has a mean but no bar
        np.testing.assert_allclose(five.lines[2][0].get_segments()[0][:, 1],
                                   [72.0 - np.sqrt(2.5), 72.0 + np.sqrt(2.5)])
        np.testing.assert_allclose(single.lines[0].get_xydata(), [[1 + offset, 70.0]])
        self.assertEqual(len(single.lines[2]), 0)
        self.assertEqual(paper_plot._mean_sd([70.0]), (70.0, 0.0))

    def test_accuracy_axis_has_few_ticks_and_room_for_the_marks(self):
        figure, axis = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        paper_plot._set_accuracy_ylim(axis, [61.0, 97.0])
        low, high = axis.get_ylim()
        self.assertEqual(low, 55.0)
        self.assertGreater(high, 100.0)
        np.testing.assert_array_equal(axis.get_yticks(), [60, 70, 80, 90, 100])
        self.assertEqual(paper_plot._accuracy_figsize(4)[1], paper_plot._accuracy_figsize(6)[1])
        self.assertLess(paper_plot._accuracy_figsize(4)[0], paper_plot._accuracy_figsize(6)[0])


class AccuracyFigureReferenceTests(unittest.TestCase):
    def _render(self, plot, results):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "performance_results.json"
            result_path.write_text(json.dumps(results))
            with patch.object(paper_plot, "PERF_RESULT_PATH", result_path), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    contextlib.redirect_stdout(io.StringIO()) as console:
                plot()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        return figure.axes[0], console.getvalue()

    @staticmethod
    def _results(cohorts):
        """feature tag -> (L2, accuracies in %) as performance entries."""
        return {f"{feature}_{index}": {"feature": feature, "l2_info": l2, "acc": value / 100}
                for feature, (l2, values) in cohorts.items()
                for index, value in enumerate(values)}

    def test_l2_figure_tests_each_strength_against_1e4(self):
        axis, console = self._render(paper_plot.plot_l2_vs_accuracy, self._results({
            "L21e5": (1e-5, [93.0]), "L21e4": (1e-4, HIGH), "L21e3": (1e-3, LOW),
            "L21e2": (1e-2, HIGH), "L21e4relu": (1e-4, LOW)}))
        # one run at 1e-5 is not tested; the activation cohort is not an L2 one
        self.assertEqual(_marks(axis), [("*", 2), ("n.s.", 3)])
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"])
        self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                         ["median", "mean \u00b1 SD", "Holm p < 0.05 / 0.01 / 0.001\n"
                                                    "vs L2 = $10^{-4}$ (Mann\u2013Whitney)"])
        self.assertIn("L2=0.001 vs L2=0.0001: Mann-Whitney p=0.00794, Holm-adjusted p=0.0159",
                      console)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [93.0, 92.0, 72.0, 92.0])

    def test_l2_figure_without_the_default_strength_is_not_tested(self):
        axis, console = self._render(paper_plot.plot_l2_vs_accuracy, self._results({
            "L21e3": (1e-3, LOW), "L21e2": (1e-2, HIGH)}))
        self.assertEqual(_marks(axis), [])
        self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                         ["median", "mean \u00b1 SD"])
        self.assertNotIn("Mann-Whitney", console)

    def test_l2_groups_keep_bare_tags_only_and_sort_by_strength(self):
        results = {
            "strong": {"feature": "L21e2", "l2_info": 1e-2, "acc": 0.6,
                       "acc_per_task": {"fdgo": 0.5, "delaygo": None}},
            "default_a": {"feature": "L21e4", "l2_info": 1e-4, "acc": 0.9,
                          "acc_per_task": {"fdgo": 1.0}},
            "default_b": {"feature": "L21e4", "l2_info": 1e-4, "acc": 0.8},
            "relu": {"feature": "L21e4relu", "l2_info": 1e-4, "acc": 0.99,
                     "acc_per_task": {"fdgo": 0.1}},
            "retention": {"feature": "L21e4lam0.9", "l2_info": 1e-4, "acc": 0.99},
        }
        self.assertEqual(paper_plot._l2_accuracy_groups(results),
                         {1e-4: [90.0, 80.0], 1e-2: [60.0]})
        with contextlib.redirect_stdout(io.StringIO()):
            per_task = paper_plot._l2_accuracy_groups(results, per_task=True)
        # a run without per-task scores adds nothing; a missing score is left out
        self.assertEqual(per_task, {1e-4: {"fdgo": [100.0]}, 1e-2: {"fdgo": [50.0]}})

    def test_l2_task_figure_tests_each_strength_against_1e4(self):
        def cohort(feature, l2, values):
            return {f"{feature}_{index}": {"feature": feature, "l2_info": l2, "acc": value / 100,
                                           "acc_per_task": {"fdgo": value / 100}}
                    for index, value in enumerate(values)}

        results = {**cohort("L21e4", 1e-4, HIGH), **cohort("L21e3", 1e-3, LOW),
                   **cohort("L21e5", 1e-5, [93.0]), **cohort("L21e4relu", 1e-4, LOW)}
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "performance_results.json"
            result_path.write_text(json.dumps(results))
            with patch.object(paper_plot, "PERF_RESULT_PATH", result_path), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    contextlib.redirect_stdout(io.StringIO()) as console:
                paper_plot.plot_l2_task_accuracy()
        figure, output_path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(output_path.name, "multitask_l2_task_accuracy.png")
        self.assertEqual(len(figure.axes), 15)
        panels = dict(zip(paper_plot._TASK_DISPLAY, figure.axes))
        self.assertEqual([label.get_text() for label in figure.axes[-1].get_xticklabels()],
                         ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$"])
        self.assertEqual(figure.get_supxlabel(), "L2 regularization strength")
        # one valid test: 1e-3 against the default; one run at 1e-5 is not tested
        self.assertEqual(_marks(panels["fdgo"]), [("**", 2)])
        np.testing.assert_allclose(panels["fdgo"].lines[0].get_ydata(), [93.0, 92.0, 72.0])
        self.assertIn("vs L2 = $10^{-4}$", figure.legends[0].get_texts()[-1].get_text())
        self.assertIn("DelayPro, L2=0.001 vs L2=0.0001: Mann-Whitney p=0.00794, "
                      "Holm-adjusted p=0.00794", console.getvalue())
        self.assertIs(paper_plot.FIGURES_BY_MODE["acc_plot"]["l2_task_accuracy"],
                      paper_plot.plot_l2_task_accuracy)

    def test_activation_figure_tests_each_activation_against_tanh(self):
        axis, console = self._render(paper_plot.plot_l2e4_activation_accuracy, self._results({
            "L21e4relu": (1e-4, [95.0, 96.0, 97.0, 98.0, 99.0]), "L21e4sigmoid": (1e-4, LOW),
            "L21e4": (1e-4, HIGH)}))
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["ReLU", "Sigmoid", "Tanh"])
        self.assertEqual(_marks(axis), [("*", 0), ("*", 1)])
        self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                         ["median, IQR", "mean \u00b1 SD", "Holm p < 0.05 / 0.01 / 0.001\n"
                                                         "vs Tanh (Mann\u2013Whitney)"])
        self.assertIn("Sigmoid vs Tanh: Mann-Whitney p=0.00794, Holm-adjusted p=0.0159", console)
        # categories carry no red connector, and each keeps its own color
        self.assertNotIn(paper_plot._TREND_LINE_COLOR, [line.get_color() for line in axis.lines])
        self.assertEqual(len(axis.containers), 3)
        self.assertEqual([mark.xy[1] for mark in axis.texts], [99.0, 74.0])
        colors = [paper_plot.mpl.colors.to_hex(dots.get_facecolor()[0])
                  for dots in axis.collections
                  if isinstance(dots, paper_plot.mpl.collections.PathCollection)]
        self.assertEqual(len(set(colors)), 3)

    def test_activation_figure_without_tanh_is_not_tested(self):
        axis, _ = self._render(paper_plot.plot_l2e4_activation_accuracy, self._results({
            "L21e4relu": (1e-4, HIGH), "L21e4sigmoid": (1e-4, LOW)}))
        self.assertEqual(_marks(axis), [])
        self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                         ["median, IQR", "mean \u00b1 SD"])


if __name__ == "__main__":
    unittest.main()
