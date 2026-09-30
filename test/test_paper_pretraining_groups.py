"""Pretraining group selection applies to every paper plot and cache format."""

import json
import pickle
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


class PretrainingGroupTests(unittest.TestCase):
    def setUp(self):
        self.original_groups = paper_plot.PRETRAINING_GROUPS
        self.original_bound = paper_plot.PRETRAINING_BOUND
        self.addCleanup(paper_plot._set_pretraining_groups, self.original_groups)
        self.addCleanup(paper_plot._set_pretraining_bound, self.original_bound)
        paper_plot._set_pretraining_bound("mb1")

    def write_fixtures(self, root, combined=False, rulesets=None):
        rulesets = rulesets or tuple(paper_plot._PRETRAINING_RULESET_STYLES)
        addon = paper_plot.PRETRAINING_ADDON_NAME
        transfer = {}
        vectors = {}
        for ruleset in rulesets:
            parents = ruleset.split("_")
            cosines = {task: [0.4] for task in parents}
            pairs = ({"__".join(parents): [0.2]} if len(parents) == 2 else {})
            vectors[ruleset] = {
                "stage1_tasks": parents, "final_task": "delayanti",
                "cos_novel_by_task": cosines, "cos_pretrained_pairs": pairs,
                "in_span_fraction": [0.8],
            }
            transfer[ruleset] = {"per_seed_iters": [[1., 10.]], "n_seeds": 1}
            result = {
                "learning": {"acc_iter_post": [1, 10, 100], "acc_post": [.5, .8, 1.]},
                "rule_vectors": {
                    "pretrained_tasks": parents, "novel_task": "delayanti",
                    "cos_novel_by_task": {task: values[0] for task, values in cosines.items()},
                    "cos_pretrained_pairs": {pair: values[0] for pair, values in pairs.items()},
                    "in_span_fraction": .8,
                },
            }
            aggregate = {"ruleset": ruleset}
            for representation in ("hidden", "modulation", "modulation_weighted"):
                result[representation] = {}
                for period in ("stimulus", "response"):
                    result[representation][period] = {
                        "cev_Y_self": [.5, .8, 1.], "cev_Y": [.2, .4, .6],
                    }
                    result[representation][f"angles_{period}"] = [.1, .2, .3]
                    aggregate[f"{representation}_{period}_self"] = [[.5, .8, 1.]]
                    aggregate[f"{representation}_{period}_cross"] = [[.2, .4, .6]]
            with (root / f"{ruleset}_dmpn_seed1_{addon}_result.pkl").open("wb") as stream:
                pickle.dump(result, stream)
            probe = {"ruleset": ruleset, "random_probe": {"accuracy_pct": [50., 60.]}}
            with (root / f"backbone_probe_{ruleset}_dmpn_seed1_{addon}.json").open("w") as stream:
                json.dump(probe, stream)
            if combined:
                with (root / f"{ruleset}_dmpn_{addon}_aggregate.pkl").open("wb") as stream:
                    pickle.dump(aggregate, stream)
        if combined:
            for suffix, data in (
                    ("transfer_speed", {"thresholds": np.array([.5, .8]), "by_ruleset": transfer}),
                    ("rule_vectors", {"by_ruleset": vectors})):
                with (root / f"combined_dmpn_{addon}_{suffix}.pkl").open("wb") as stream:
                    pickle.dump(data, stream)

    def test_every_plot_respects_groups_with_combined_and_per_seed_data(self):
        for combined in (False, True):
            with self.subTest(combined=combined), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_fixtures(root, combined)
                for groups in ("motifs", "all", "motifs"):
                    paper_plot._set_pretraining_groups(groups)
                    expected = {paper_plot._PRETRAINING_RULESET_STYLES[ruleset][0]
                                for ruleset in paper_plot._PRETRAINING_GROUPS[groups]}
                    for name, function in paper_plot.FIGURES_BY_MODE["pretraining"].items():
                        with self.subTest(groups=groups, figure=name), chdir(root), \
                                patch.object(paper_plot, "PRETRAINING_ANALYSIS_DIR", root), \
                                patch.object(paper_plot, "OUT_DIR", root / "figures"), \
                                patch.object(paper_plot, "SHOW_LEGEND", True), \
                                patch.object(paper_plot, "_save_fig") as save:
                            function()
                            save.assert_called_once()
                            figure = save.call_args.args[0]
                            try:
                                if name == "backbone_probe":
                                    labels = {label.get_text().replace("\n", " ")
                                              for label in figure.axes[0].get_xticklabels()}
                                    log_entry = (root / "log/backbone_probe.log").read_text().splitlines()[-1]
                                    self.assertIn(f"feature={paper_plot.PRETRAINING_ADDON_NAME}", log_entry)
                                    self.assertIn(f"groups={groups}", log_entry)
                                    self.assertIn("bound=mb1", log_entry)
                                else:
                                    labels = set(figure.axes[0].get_legend_handles_labels()[1])
                                self.assertEqual(labels, expected | (
                                    {"Self"} if name.startswith("aggregate_cve") else set()))
                                if name == "rule_vectors":
                                    self.assertEqual(len(figure.axes[0].patches),
                                                     4 if groups == "motifs" else 6)
                                    np.testing.assert_allclose(
                                        figure.get_size_inches(),
                                        [3.78 if groups == "motifs" else 6.3, 1.6])
                                    self.assertFalse(any(line.get_marker() == "."
                                             for line in figure.axes[0].lines))
                                    bars = [container for container in figure.axes[0].containers
                                        if isinstance(container, paper_plot.mpl.container.BarContainer)]
                                    self.assertEqual(len(bars), len(figure.axes[0].patches))
                                    self.assertTrue(all(bar.errorbar is not None
                                            and bar.errorbar.has_yerr for bar in bars))
                                    dividers = [line for line in figure.axes[0].lines
                                                if len(line.get_xdata()) == 2
                                                and np.allclose(line.get_xdata(), [1.5, 1.5])]
                                    self.assertEqual(len(dividers), int(groups == "motifs"))
                                    if dividers:
                                        np.testing.assert_array_equal(dividers[0].get_ydata(), [0, 1])
                            finally:
                                paper_plot.plt.close(figure)

    def test_linear_trajectory_preserves_log_curves_and_uses_separate_output(self):
        self.assertIs(paper_plot.ALL_FIGURES["learning_trajectory_linear"],
                      paper_plot.plot_learning_trajectory_linear)
        for bound in ("mb1", "mb2"):
            paper_plot._set_pretraining_bound(bound)
            with self.subTest(bound=bound), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_fixtures(root)
                for groups in ("motifs", "all"):
                    paper_plot._set_pretraining_groups(groups)
                    with self.subTest(groups=groups), \
                            patch.object(paper_plot, "PRETRAINING_ANALYSIS_DIR", root), \
                            patch.object(paper_plot, "OUT_DIR", root / "figures"), \
                            patch.object(paper_plot, "SHOW_LEGEND", True), \
                            patch.object(paper_plot, "_save_fig") as save:
                        paper_plot.plot_learning_trajectory()
                        paper_plot.plot_learning_trajectory_linear()
                    self.assertEqual(save.call_count, 2)
                    log_figure, log_path = save.call_args_list[0].args
                    linear_figure, linear_path = save.call_args_list[1].args
                    self.addCleanup(paper_plot.plt.close, log_figure)
                    self.addCleanup(paper_plot.plt.close, linear_figure)
                    self.assertEqual(log_path.name, "learning_trajectory.png")
                    self.assertEqual(linear_path.name, "learning_trajectory_linear.png")
                    log_axis, linear_axis = log_figure.axes[0], linear_figure.axes[0]
                    self.assertEqual(log_axis.get_xscale(), "log")
                    self.assertEqual(linear_axis.get_xscale(), "linear")
                    np.testing.assert_allclose(log_figure.get_size_inches(), linear_figure.get_size_inches())
                    self.assertEqual(log_axis.get_ylim(), linear_axis.get_ylim())
                    self.assertEqual(len(log_axis.lines), len(linear_axis.lines))
                    for log_line, linear_line in zip(log_axis.lines, linear_axis.lines):
                        np.testing.assert_array_equal(log_line.get_xdata(), linear_line.get_xdata())
                        np.testing.assert_array_equal(log_line.get_ydata(), linear_line.get_ydata())
                        self.assertEqual(log_line.get_color(), linear_line.get_color())
                        self.assertEqual(log_line.get_alpha(), linear_line.get_alpha())
                        self.assertEqual(log_line.get_label(), linear_line.get_label())

    def test_control_only_data_does_not_create_motif_figures(self):
        paper_plot._set_pretraining_groups("motifs")
        for combined in (False, True):
            with self.subTest(combined=combined), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_fixtures(root, combined, rulesets=("fdanti", "fdgo"))
                with chdir(root), \
                        patch.object(paper_plot, "PRETRAINING_ANALYSIS_DIR", root), \
                        patch.object(paper_plot, "OUT_DIR", root / "figures"), \
                        patch.object(paper_plot, "_save_fig") as save:
                    for function in paper_plot.FIGURES_BY_MODE["pretraining"].values():
                        function()
                    save.assert_not_called()

    def test_cli_sets_groups_before_dispatch_and_preserves_bound_options(self):
        for selection in ([], ["--pretraining-groups", "motifs"],
                          ["--pretraining-groups", "all"]):
            for target in (["pretraining"], ["--only", "learning_trajectory"]):
                with self.subTest(selection=selection, target=target), \
                        tempfile.TemporaryDirectory() as directory:
                    paper_plot._set_pretraining_groups("motifs")
                    observed = []
                    function = Mock(side_effect=lambda: observed.append((
                        paper_plot.PRETRAINING_GROUPS, paper_plot.PRETRAINING_BOUND)))
                    figures = {"learning_trajectory": function}
                    with patch.object(paper_plot, "FIGURES_BY_MODE", {"pretraining": figures}), \
                            patch.object(paper_plot, "ALL_FIGURES", figures), \
                            patch.object(paper_plot, "OUT_DIR", Path(directory)), \
                            patch("sys.argv", ["paper_plot.py", *target, *selection,
                                               "--pretraining-bound", "mod2"]), \
                            patch("builtins.print") as output:
                        paper_plot.main()
                    expected = selection[-1] if selection else "all"
                    self.assertEqual(observed, [(expected, "mb2")])
                    self.assertTrue(any(f"groups={expected}" in str(call)
                                        for call in output.call_args_list))

    def test_cli_logs_selected_feature_for_default_and_mod2(self):
        for arguments, bound in (([], "mb1"), (["--pretraining-bound", "mod2"], "mb2")):
            with self.subTest(bound=bound), tempfile.TemporaryDirectory() as directory:
                figures = {"learning_trajectory": Mock()}
                with patch.object(paper_plot, "FIGURES_BY_MODE", {"pretraining": figures}), \
                        patch.object(paper_plot, "ALL_FIGURES", figures), \
                        patch.object(paper_plot, "OUT_DIR", Path(directory)), \
                        patch("sys.argv", ["paper_plot.py", "--only", "learning_trajectory", *arguments]), \
                        patch("builtins.print") as output:
                    paper_plot.main()
                feature = paper_plot._PRETRAINING_BOUND_ADDONS[bound]
                self.assertTrue(any(f"feature={feature};" in str(call)
                                    for call in output.call_args_list))

    def test_invalid_groups_are_rejected_before_dispatch(self):
        with self.assertRaises(ValueError):
            paper_plot._set_pretraining_groups("invalid")
        with patch("sys.argv", ["paper_plot.py", "pretraining", "--pretraining-groups", "invalid"]), \
                patch.object(paper_plot, "_ensure_out_dir") as ensure, \
                self.assertRaises(SystemExit) as error:
            paper_plot.main()
        self.assertEqual(error.exception.code, 2)
        ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()