"""Regression checks for metadata-based projection and hidden accuracy grouping."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import _bootstrap  # noqa: F401
import paper_plot


class ProjectionAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "net_params": {"n_neurons": [1, 300, 1], "activation": "tanh",
                           "linear_embed": 200, "input_layer_add": True,
                           "ml_params": {"m_time_scale": 4000}},
            "train_params": {"weight_reg": "L2", "reg_lambda": 1e-4},
        }

    def group(self, configs, results, *, vary_hidden=False, per_task=False):
        def open_config(path, *args, **kwargs):
            name = path.name.removeprefix("param_").removesuffix("_param.json")
            if name not in configs:
                raise FileNotFoundError(name)
            return io.StringIO(json.dumps(configs[name]))

        with patch.object(Path, "open", new=open_config):
            return paper_plot._projection_accuracy_groups(results, vary_hidden=vary_hidden,
                                                          per_task=per_task)

    def test_per_task_grouping_and_missing_scores(self):
        for vary_hidden in (False, True):
            with self.subTest(vary_hidden=vary_hidden):
                config = copy.deepcopy(self.config)
                if vary_hidden:
                    config["net_params"].update(n_neurons=[1, 200, 1], linear_embed=300)
                configs = {name: config for name in ("first", "second", "legacy")}
                results = {
                    "first": {"acc": 0.9, "acc_per_task": {"fdgo": 0.5, "delaygo": None}},
                    "second": {"acc": 0.9, "acc_per_task": {"fdgo": 1.0, "delaygo": 0.8,
                                                           "invalid": float("nan")}},
                    "legacy": {"acc": 0.9},
                }
                self.assertEqual(self.group(configs, results, vary_hidden=vary_hidden, per_task=True),
                                 {200: {"fdgo": [50.0, 100.0], "delaygo": [80.0]}})

    def test_task_lines_use_index_positions_and_seed_means(self):
        groups = {10: {"fdgo": [50.0, 100.0], "delaygo": [80.0]},
                  100: {"fdgo": [90.0]}, 300: {"fdgo": [100.0], "delaygo": [95.0]}}
        for vary_hidden in (False, True):
            with self.subTest(vary_hidden=vary_hidden), \
                    patch.object(paper_plot, "_projection_accuracy_groups", return_value=groups) as grouping, \
                    patch.object(Path, "exists", return_value=True), \
                    patch.object(Path, "open", return_value=io.StringIO(json.dumps({
                        "run": {"acc_per_task": {"fdgo": 1.0, "delaygo": 0.8}}}))), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot._plot_dimension_task_accuracy(vary_hidden=vary_hidden)
                figure, path = save.call_args.args
                try:
                    self.assertEqual(len(figure.axes), 15)
                    panels = dict(zip(paper_plot._TASK_DISPLAY, figure.axes))
                    axis = figure.axes[-1]
                    np.testing.assert_array_equal(axis.get_xticks(), [0, 1, 2])
                    self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                                     ["10", "100", "300"])
                    np.testing.assert_allclose(panels["fdgo"].lines[0].get_ydata(), [75, 90, 100])
                    np.testing.assert_allclose(panels["delaygo"].lines[0].get_ydata(), [80, np.nan, 95])
                    for task, panel in panels.items():
                        self.assertEqual(panel.get_title(loc="left"), paper_plot._TASK_DISPLAY[task])
                        np.testing.assert_array_equal(panel.lines[0].get_xdata(), [0, 1, 2])
                        self.assertEqual(panel.lines[0].get_marker(), "o")
                    np.testing.assert_array_equal(panels["fdgo"].collections[0].get_offsets(),
                                                  [[0, 50], [0, 100]])
                    dots = [item for item in panels["fdgo"].collections
                            if isinstance(item, paper_plot.mpl.collections.PathCollection)]
                    self.assertEqual(sum(len(item.get_offsets()) for item in dots), 4)
                    # beside the dots, each width's mean; an SD bar where it has two runs
                    offset = paper_plot._ACCURACY_MEAN_OFFSET
                    np.testing.assert_allclose(
                        [bar.lines[0].get_xydata()[0] for bar in panels["fdgo"].containers],
                        [[offset, 75.0], [1 + offset, 90.0], [2 + offset, 100.0]])
                    self.assertEqual([len(bar.lines[2]) for bar in panels["fdgo"].containers],
                                     [1, 0, 0])
                    self.assertEqual(len(panels["delaygo"].collections), 2)
                    self.assertGreater(panels["fdgo"].get_ylim()[1], 100)
                    self.assertEqual(panels["fdanti"].texts[0].get_text(), "No data")
                    self.assertLessEqual(figure.get_figheight(), 12)
                    self.assertTrue(grouping.call_args.kwargs["per_task"])
                    self.assertEqual(grouping.call_args.kwargs["vary_hidden"], vary_hidden)
                    self.assertIn("hidden" if vary_hidden else "projection", path.name)
                finally:
                    paper_plot.plt.close(figure)

    def test_hidden_sweep_uses_metadata_and_fixed_projection(self):
        configs = {}
        for name, hidden_dim, projection in (("misleading_hidden300", 100, 300),
                                             ("another_seed", 100, 300),
                                             ("arbitrary", 400, 300),
                                             ("wrong_projection", 300, 200)):
            config = copy.deepcopy(self.config)
            config["net_params"].update(n_neurons=[1, hidden_dim, 1], linear_embed=projection)
            configs[name] = config
        results = {name: {"acc": 0.9, "hidden_size": 999, "proj_dim": 999} for name in configs}
        self.assertEqual(self.group(configs, results, vary_hidden=True),
                         {100: [90.0, 90.0], 400: [90.0]})

    def test_hidden_sweep_rejects_other_activation_l2_and_multilayer(self):
        for section, key, value in (("net_params", "activation", "relu"),
                                    ("train_params", "reg_lambda", 1e-3),
                                    ("train_params", "weight_reg", "L1"),
                                    ("net_params", "ml_params", {"m_time_scale": 400}),
                                    ("net_params", "n_neurons", [1, 100, 100, 1])):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config["net_params"]["linear_embed"] = 300
                config[section][key] = value
                self.assertEqual(self.group({"run": config}, {"run": {"acc": 0.9}},
                                            vary_hidden=True), {})

    def test_metadata_overrides_names_and_cached_dimensions(self):
        renamed = copy.deepcopy(self.config)
        renamed["net_params"]["linear_embed"] = 75
        configs = {"L21e4proj10": self.config, "arbitrary_name": renamed,
                   "another_seed": self.config}
        results = {name: {"acc": 0.9, "feature": "L21e4proj400", "proj_dim": 400,
                          "hidden_size": 400, "l2_info": 1e-2} for name in configs}
        grouped = self.group(configs, results)
        self.assertEqual(list(grouped), [75, 200])
        self.assertEqual(grouped, {75: [90.0], 200: [90.0, 90.0]})

    def test_excludes_nonmatching_configurations(self):
        for section, field, value in (
            ("net_params", "n_neurons", [1, 400, 1]),
            ("net_params", "activation", "relu"),
            ("net_params", "input_layer_add", False),
            ("train_params", "weight_reg", "L1"),
            ("train_params", "reg_lambda", 1e-3),
            ("net_params", "ml_params", {"m_time_scale": 400}),
        ):
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[section][field] = value
                self.assertEqual(self.group({"L21e4": config}, {"L21e4": {"acc": 0.9}}), {})

    def test_only_the_default_bounds_join_the_sweep(self):
        """The 300-wide group is the default cohort the other widths are
        tested against, so the wider-bound L21e4mb2 runs stay out of it."""
        default = copy.deepcopy(self.config)
        default["net_params"]["linear_embed"] = 300
        explicit = copy.deepcopy(default)
        explicit["net_params"]["ml_params"]["m_bounds"] = [-1.0, 1.0]
        wider = copy.deepcopy(default)
        wider["net_params"]["ml_params"]["m_bounds"] = [-2.0, 2.0]
        configs = {"default": default, "explicit": explicit, "wider": wider}
        results = {"default": {"acc": 0.9}, "explicit": {"acc": 0.8}, "wider": {"acc": 0.7}}
        for vary_hidden in (False, True):
            with self.subTest(vary_hidden=vary_hidden):
                self.assertEqual(self.group(configs, results, vary_hidden=vary_hidden),
                                 {300: [90.0, 80.0]})

    def _render(self, plot, groups, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "performance_results.json"
            result_path.write_text("{}")
            with patch.object(paper_plot, "_projection_accuracy_groups", return_value=groups), \
                    patch.object(paper_plot, "PERF_RESULT_PATH", result_path), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    contextlib.redirect_stdout(io.StringIO()) as console:
                plot(**kwargs)
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        return figure, console.getvalue()

    @staticmethod
    def _marks(axis):
        return [(text.get_text(), text.xy[0]) for text in axis.texts
                if isinstance(text, paper_plot.mpl.text.Annotation)]

    def test_overall_sweep_marks_each_width_against_300(self):
        groups = {10: [70.0, 71.0, 72.0, 73.0, 74.0], 100: [91.5, 92.5, 93.5],
                  300: [90.0, 91.0, 92.0, 93.0, 94.0], 400: [95.0]}
        for vary_hidden, name in ((False, "projection"), (True, "hidden")):
            with self.subTest(vary_hidden=vary_hidden):
                figure, console = self._render(paper_plot._plot_dimension_accuracy, groups,
                                               vary_hidden=vary_hidden)
                axis = figure.axes[0]
                # Holm over the two valid tests; one run at 400 is not tested.
                self.assertEqual(self._marks(axis), [("*", 0.0), ("n.s.", 1.0)])
                self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                                 ["median", "mean \u00b1 SD",
                                  "Holm p < 0.05 / 0.01 / 0.001\n"
                                  f"vs {name} dimension 300 (Mann\u2013Whitney)"])
                # the mean of every width, beside its dots
                np.testing.assert_allclose(
                    [bar.lines[0].get_xydata()[0] for bar in axis.containers],
                    [[index + paper_plot._ACCURACY_MEAN_OFFSET, mean]
                     for index, mean in enumerate([72.0, 92.5, 92.0, 95.0])])
                # medians joined in red, a compact panel with few ticks
                np.testing.assert_allclose(axis.lines[0].get_ydata(), [72.0, 92.5, 92.0, 95.0])
                np.testing.assert_allclose(figure.get_size_inches(),
                                           paper_plot._accuracy_figsize(4))
                self.assertLessEqual(len(axis.get_yticks()),
                                     paper_plot._STAT_SCATTER_MAX_TICKS + 1)
                self.assertIn("10 vs 300: Mann-Whitney p=0.00794, Holm-adjusted p=0.0159", console)
                self.assertNotIn("400 vs 300", console)

    def test_sweep_without_the_default_width_is_not_tested(self):
        figure, console = self._render(paper_plot._plot_dimension_accuracy,
                                       {10: [70.0, 71.0], 100: [90.0, 91.0]}, vary_hidden=False)
        self.assertEqual(self._marks(figure.axes[0]), [])
        self.assertEqual([text.get_text() for text in figure.axes[0].get_legend().get_texts()],
                         ["median", "mean \u00b1 SD"])
        self.assertNotIn("Mann-Whitney", console)

    def test_task_sweep_marks_each_width_against_300(self):
        groups = {10: {"fdgo": [70.0, 71.0, 72.0, 73.0, 74.0]},
                  300: {"fdgo": [90.0, 91.0, 92.0, 93.0, 94.0], "delaygo": [95.0, 96.0]},
                  400: {"delaygo": [97.0]}}
        figure, console = self._render(paper_plot._plot_dimension_task_accuracy, groups,
                                       vary_hidden=True)
        panels = dict(zip(paper_plot._TASK_DISPLAY, figure.axes))
        self.assertEqual(self._marks(panels["fdgo"]), [("**", 0.0)])
        self.assertEqual(self._marks(panels["delaygo"]), [])
        self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()[:2]],
                         ["median", "mean \u00b1 SD"])
        self.assertIn("vs hidden dimension 300", figure.legends[0].get_texts()[2].get_text())
        self.assertIn("DelayPro, 10 vs 300: Mann-Whitney p=0.00794, Holm-adjusted p=0.00794",
                      console)

    def test_missing_retention_is_skipped_not_defaulted(self):
        """core/mpn.py falls back to lam_clamp without m_time_scale, so a config
        that lacks it is not assumed to be the 4000 ms cohort."""
        config = copy.deepcopy(self.config)
        del config["net_params"]["ml_params"]
        self.assertEqual(self.group({"run": config}, {"run": {"acc": 0.9}}), {})
        config["net_params"]["ml_params"] = {}
        self.assertEqual(self.group({"run": config}, {"run": {"acc": 0.9}}), {})

    def test_missing_metadata_does_not_fall_back_to_labels(self):
        config = copy.deepcopy(self.config)
        del config["net_params"]["linear_embed"]
        results = {"missing_config": {"acc": 0.9},
                   "L21e4proj100": {"acc": 0.9, "proj_dim": 100}}
        self.assertEqual(self.group({"L21e4proj100": config}, results), {})


if __name__ == "__main__":
    unittest.main()
