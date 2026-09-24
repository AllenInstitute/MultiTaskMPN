"""Checks for the public paper-figure mode grouping."""

import inspect
import json
import pickle
import tempfile
import numpy as np
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import paper_plot


class PaperPlotModeTests(unittest.TestCase):
    def test_overmembership_variants_align_both_na_with_uniform_gray_bars(self):
        variants = (
            (paper_plot.plot_overmembership_norm,
             "modulation_all_prepost_belonging_{aname}_normalized.pkl",
             "overmembership_normalized.png"),
            (paper_plot.plot_overmembership_unnorm,
             "modulation_all_prepost_belonging_{aname}_unnormalized.pkl",
             "overmembership_unnormalized.png"),
            (paper_plot.plot_overmembership_weighted,
             "modulation_all_weighted_prepost_belonging_{aname}_unnormalized.pkl",
             "overmembership_weighted.png"),
            (paper_plot.plot_overmembership_var_weighted,
             "modulation_all_var_weighted_prepost_belonging_{aname}_unnormalized.pkl",
             "overmembership_var_weighted.png"),
        )
        observations = (np.array([2., 4., 8.]), np.array([4., 6., 8., 10.]))
        controls = (np.full(3, 2.), np.full(4, 2.))
        with tempfile.TemporaryDirectory() as directory:
            run_dirs = [Path(directory) / name for name in ("seed1", "seed2")]
            for run_dir, offset in zip(run_dirs, (0., 2.)):
                run_dir.mkdir()
                data = {"prepost_belonging_results": [{
                    "bar_name_lst": [["Share-Pre", "Share-Post", "Neither"],
                                     ["Share-Pre-Cluster", "Share-Post-Cluster",
                                      "Share-Both-Cluster", "Neither"]],
                    "bar_all_lst": [values + offset for values in observations],
                    "bar_all_ctrl_lst": controls,
                }]}
                for _, template, _ in variants:
                    with (run_dir / template.format(aname=run_dir.name)).open("wb") as stream:
                        pickle.dump(data, stream)
            for render, _, filename in variants:
                with self.subTest(figure=filename), \
                        patch.object(paper_plot, "_ensure_out_dir"), \
                        patch.object(paper_plot, "_find_experiment_dirs", return_value=run_dirs), \
                        patch.object(paper_plot, "_save_fig") as save_fig:
                    render()
                    save_fig.assert_called_once()
                    figure, output_path = save_fig.call_args.args[:2]
                    self.addCleanup(paper_plot.plt.close, figure)
                    self.assertEqual(output_path.name, f"multitask_{filename}")
                    self.assertEqual(len(figure.axes), 2)
                    for row_index, axis in enumerate(figure.axes):
                        positions = [0, 1, 3] if row_index == 0 else [0, 1, 2, 3]
                        self.assertEqual(len(axis.patches), len(positions))
                        np.testing.assert_allclose(
                            [bar.get_x() + bar.get_width() / 2 for bar in axis.patches],
                            positions)
                        for bar in axis.patches:
                            np.testing.assert_allclose(bar.get_facecolor(), (0.6, 0.6, 0.6, 1.))
                        np.testing.assert_allclose(
                            [bar.get_height() for bar in axis.patches],
                            (observations[row_index] + 1 - controls[row_index]) / controls[row_index])
                        np.testing.assert_array_equal(axis.get_xticks(), [0, 1, 2, 3])
                        expected_labels = (["Pre", "Post", "Both", "Neither"]
                                           if row_index == 0 else
                                           ["Pre Cl.", "Post Cl.", "Both Cl.", "Neither"])
                        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                                         expected_labels)
                        self.assertEqual([text.get_text() for text in axis.texts],
                                         ["N/A"] if row_index == 0 else [])
                        bars = next(container for container in axis.containers
                                    if isinstance(container, paper_plot.mpl.container.BarContainer))
                        segments = bars.errorbar.lines[2][0].get_segments()
                        self.assertEqual(len(segments), len(positions))
                        np.testing.assert_allclose([segment[0, 0] for segment in segments],
                                                   positions)
                        np.testing.assert_allclose(
                            [segment[1, 1] - segment[0, 1] for segment in segments], 1.)
                        scatters = [collection for collection in axis.collections
                                    if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
                        self.assertEqual(len(scatters), 2)
                        for scatter, offset in zip(scatters, (0., 2.)):
                            offsets = scatter.get_offsets()
                            np.testing.assert_allclose(np.rint(offsets[:, 0]), positions)
                            np.testing.assert_allclose(
                                offsets[:, 1],
                                (observations[row_index] + offset - controls[row_index]) / controls[row_index])
                    label = figure.axes[0].texts[0]
                    self.assertEqual(label.xy, (2, 0))
                    figure.canvas.draw()
                    zero_y = figure.axes[0].transData.transform((2, 0))[1]
                    self.assertAlmostEqual(
                        label.get_window_extent().y0 - zero_y, 5 * figure.dpi / 72)
                    for position in range(4):
                        self.assertAlmostEqual(
                            figure.axes[0].get_xaxis_transform().transform((position, 0))[0],
                            figure.axes[1].get_xaxis_transform().transform((position, 0))[0])

    def test_lesion_figures_have_their_own_leison_mode(self):
        expected = {
            "lesion_heatmap": paper_plot.plot_lesion_heatmap,
            "lesion_cluster_sizes": paper_plot.plot_lesion_cluster_sizes,
            "cluster_corr_vs_lesion": paper_plot.plot_cluster_corr_vs_lesion,
            "om_vs_lesion": paper_plot.plot_om_vs_lesion,
            "cross_seed_summary": paper_plot.plot_cross_seed_summary,
        }
        self.assertEqual(paper_plot.FIGURES_BY_MODE["leison"], expected)
        self.assertTrue(set(expected).isdisjoint(
            paper_plot.FIGURES_BY_MODE["multiple_tasks"]))
        self.assertEqual(set(paper_plot.FIGURES_BY_MODE["multiple_tasks"]), {
            "input", "hidden", "modulation", "heatmap_colorbar",
            "overmembership_norm", "overmembership_unnorm",
            "overmembership_weighted", "overmembership_var_weighted",
            "overmembership_examples",
            "input_weight_correlation",
        })
        for name, function in expected.items():
            self.assertIs(paper_plot.ALL_FIGURES[name], function)
        names = [name for group in paper_plot.FIGURES_BY_MODE.values()
                 for name in group]
        self.assertEqual(len(names), len(set(names)))

    def test_leison_cli_dispatch_and_existing_single_figure_names(self):
        cases = [
            (["leison"], set(paper_plot.FIGURES_BY_MODE["leison"])),
            (["multiple_tasks"], set(paper_plot.FIGURES_BY_MODE["multiple_tasks"])),
            (["multiple_tasks", "leison"],
             set(paper_plot.FIGURES_BY_MODE["multiple_tasks"])
             | set(paper_plot.FIGURES_BY_MODE["leison"])),
        ]
        cases.extend((["--only", name], {name})
                     for name in paper_plot.FIGURES_BY_MODE["leison"])
        for arguments, expected in cases:
            with self.subTest(arguments=arguments), \
                    tempfile.TemporaryDirectory() as directory:
                modes = {
                    mode: {name: Mock(name=name) for name in group}
                    for mode, group in paper_plot.FIGURES_BY_MODE.items()
                }
                figures = {name: function for group in modes.values()
                           for name, function in group.items()}
                with patch.object(paper_plot, "FIGURES_BY_MODE", modes), \
                        patch.object(paper_plot, "ALL_FIGURES", figures), \
                        patch.object(paper_plot, "OUT_DIR", Path(directory)), \
                        patch.object(paper_plot, "_set_pretraining_bound"), \
                        patch("sys.argv", ["paper_plot.py", *arguments]), \
                        patch("builtins.print") as output:
                    paper_plot.main()
                self.assertEqual(
                    {name for name, function in figures.items() if function.called},
                    expected,
                )
                for name in expected:
                    figures[name].assert_called_once_with()
                if "leison" in arguments:
                    self.assertTrue(any(
                        f"leison: {paper_plot.ANAME}" in str(call.args[0])
                        for call in output.call_args_list if call.args))

    def test_state_space_figures_have_their_own_mode(self):
        expected = {
            "state_space_combined",
            "state_space_dist_angle",
            "state_space_r_values",
        }
        self.assertEqual(set(paper_plot.FIGURES_BY_MODE["state_space"]), expected)
        self.assertTrue(expected.isdisjoint(
            paper_plot.FIGURES_BY_MODE["multiple_tasks"]))
        self.assertTrue(expected.issubset(paper_plot.ALL_FIGURES))

    def test_state_space_r_values_are_grouped_by_four_l2_strengths(self):
        result_dict = {}
        for index, strength in enumerate(paper_plot.STATE_SPACE_L2_STRENGTHS):
            result_dict[f"run{index}"] = {
                "l2_info": strength,
                "rval_dict": {
                    "hidden": (0.1 + index, 1.0, 0.01),
                    "mod": (0.2 + index, 1.0, 0.01),
                    "eff_mod": (0.3 + index, 1.0, 0.01),
                },
            }

        data_types = ["hidden", "mod", "eff_mod"]
        grouped = paper_plot._state_space_r_values_by_l2(
            result_dict, data_types)

        self.assertEqual(tuple(grouped), paper_plot.STATE_SPACE_L2_STRENGTHS)
        for index, strength in enumerate(paper_plot.STATE_SPACE_L2_STRENGTHS):
            np.testing.assert_allclose(
                [grouped[strength][dt][0] for dt in data_types],
                [0.1 + index, 0.2 + index, 0.3 + index],
            )

    @staticmethod
    def _state_space_pca(centers):
        return {
            "all_rules": ["fdgo", "delaygo", "fdanti", "delayanti"],
            "pca_results": {"eff_mod": {
                "X_2d": np.zeros((8, 2)),
                "ctx_rule_labels": np.repeat(np.arange(4), 2),
                "task_centroids": np.asarray(centers, dtype=float),
                "task_trial_counts": np.full(4, 2),
                "centroid_space": "original_features",
            }},
        }

    def test_state_space_trial_uses_high_dimensional_centers_within_l2_1e4(self):
        self.assertEqual(paper_plot.STATE_SPACE_EXAMPLE_L2, 1e-4)
        good = self._state_space_pca([
            [0, 0, -5], [0, 0, -4], [0, 0, 4], [0, 0, 5],
        ])
        mixed = self._state_space_pca([
            [0, 0, -5], [0, 0, 4], [0, 0, -4], [0, 0, 5],
        ])
        self.assertGreater(
            paper_plot._eff_mod_task_centroid_metrics(good)["score"],
            paper_plot._eff_mod_task_centroid_metrics(mixed)["score"],
        )

        result_dict = {
            "mixed": {"l2_info": 1e-4},
            "best1": {"l2_info": 1e-4},
            "best2": {"l2_info": 1e-4},
            "aaa_previous_l2": {"l2_info": 1e-3},
            "better_but_other_l2": {"l2_info": 1e-2},
        }
        pca_data = {
            "mixed": mixed,
            "best1": good,
            "best2": good,
            "aaa_previous_l2": good,
            "better_but_other_l2": good,
        }
        # Equal scores use the alphabetically first aname, independent of dict
        # insertion order, so regeneration cannot silently switch examples.
        selected = paper_plot._best_state_space_trial(
            result_dict, pca_loader=pca_data.get)
        self.assertEqual(selected[0], "best1")
        self.assertAlmostEqual(selected[1], 0.8)
        report = paper_plot._state_space_centroid_ranking(
            result_dict, pca_loader=pca_data.get)
        self.assertEqual(report["n_scored"], 3)
        self.assertEqual([row["aname"] for row in report["rankings"]],
                         ["best1", "best2", "mixed"])
        scores = [row["score"] for row in report["rankings"]]
        self.assertAlmostEqual(report["summary"]["score"]["mean"], np.mean(scores))
        self.assertAlmostEqual(report["summary"]["score"]["std"], np.std(scores, ddof=1))

    def test_state_space_trial_returns_none_without_valid_centers(self):
        self.assertIsNone(paper_plot._best_state_space_trial({
            "missing": {},
            "wrong_l2": {"l2_info": 1e-2},
            "missing_pca": {"l2_info": 1e-4},
        }, pca_loader=lambda _: None))

    def test_state_space_ranking_does_not_fall_back_to_two_dimensional_points(self):
        legacy = {"all_rules": ["fdgo", "delaygo", "fdanti", "delayanti"],
                  "pca_results": {"eff_mod": {"X_2d": np.ones((8, 2)),
                                             "ctx_rule_labels": np.repeat(np.arange(4), 2)}}}
        report = paper_plot._state_space_centroid_ranking(
            {"legacy": {"l2_info": 1e-4}}, pca_loader=lambda _: legacy)
        self.assertIsNone(report["best_aname"])
        self.assertEqual(report["n_candidates"], 1)
        self.assertEqual(report["n_scored"], 0)
        self.assertEqual(report["skipped"][0]["aname"], "legacy")

    def test_state_space_selection_uses_original_analysis_and_ignores_separate_cache(self):
        data = self._state_space_pca([[0, 0, -5], [0, 0, -4],
                                      [0, 0, 4], [0, 0, 5]])
        data.update(aname="new_run", l2_info=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix, content in (("state_space_context", {"obsolete": True}),
                                    ("state_space_pca", data)):
                with (root / f"{prefix}_new_run_noise0.01.pkl").open("wb") as stream:
                    pickle.dump(content, stream)
            saved_results = {"new_run": {"l2_info": 1e-4,
                                         "scatter": {"hidden": "retained"}}}
            result_path = root / "initial_condition_distance_vs_angle_results.pkl"
            with result_path.open("wb") as stream:
                pickle.dump(saved_results, stream)
            with patch.object(paper_plot, "STATE_SPACE_DIR", root), \
                    patch.object(paper_plot, "RVAL_RESULT_PATH", result_path), \
                    patch.object(paper_plot, "OUT_DIR", root / "plots"):
                selected = paper_plot._load_best_state_space_trial()
            self.assertEqual(selected[0], "new_run")
            self.assertAlmostEqual(selected[1], 0.8)
            self.assertEqual(selected[2], saved_results["new_run"])
            with (root / "plots/multitask_state_space_centroid_scores.json").open() as stream:
                report = json.load(stream)
            self.assertEqual(report["best_aname"], "new_run")
            self.assertEqual(report["n_scored"], 1)
            self.assertIsNone(report["summary"]["score"]["std"])

    def test_state_space_pca_outputs_include_raw_modulation(self):
        data = {"pca_results": {}}
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             return_value=("selected", 0.5, {})), \
                patch.object(paper_plot, "_load_state_space_pca", return_value=data), \
                patch.object(paper_plot, "_plot_state_space_panel") as plot_panel:
            paper_plot.plot_state_space_combined()

        self.assertEqual(
            [call.args[1:] for call in plot_panel.call_args_list],
            [
                ("hidden", "Hidden state", "state_space_hidden.png"),
                ("mod", "Modulation", "state_space_mod.png"),
                ("eff_mod", "Eff. modulation", "state_space_eff_mod.png"),
            ],
        )

    def test_state_space_dist_angle_renders_saved_nonzero_intercepts(self):
        entry = {"rval_dict": {}, "scatter": {}}
        for key, intercept in (("hidden", 15.), ("eff_mod", -3.)):
            entry["rval_dict"][key] = (0.9, 2., 0.02)
            entry["scatter"][key] = {
                "dists": np.array([2., 3., 4.]),
                "angles_deg": np.array([4., 6., 8.]) + intercept,
                "regression": {"through_origin": False, "intercept": intercept},
            }
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             return_value=("selected", 0.5, entry)), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_state_space_dist_angle()

        save_fig.assert_called_once()
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        for axis, intercept in zip(figure.axes, (15., -3.)):
            line = axis.lines[0]
            np.testing.assert_allclose(line.get_ydata(),
                                       intercept + 2 * line.get_xdata())
            self.assertIn("nominal p", axis.get_legend().get_texts()[0].get_text())

    def test_state_space_dist_angle_rejects_legacy_zero_intercept_cache(self):
        for regression in ({}, {"through_origin": True, "intercept": 0.}):
            entry = {"rval_dict": {"hidden": (0.9, 2., 0.02)}, "scatter": {
                "hidden": {"dists": np.array([2., 3., 4.]),
                           "angles_deg": np.array([4., 6., 8.]),
                           "regression": regression},
            }}
            with self.subTest(regression=regression), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_load_best_state_space_trial",
                                 return_value=("selected", 0.5, entry)), \
                    patch.object(paper_plot, "_save_fig") as save_fig, \
                    patch("builtins.print") as output:
                paper_plot.plot_state_space_dist_angle()
            save_fig.assert_not_called()
            self.assertIn("legacy cache", output.call_args.args[0])

    def test_cross_seed_summary_does_not_write_csv(self):
        source = inspect.getsource(paper_plot.plot_cross_seed_summary)
        self.assertNotIn("cross_seed_summary.csv", source)

    def test_two_in_multiple_renders_both_sibling_families(self):
        loaded = {"representations": {}}
        with patch.object(paper_plot, "_load_pkl_or_skip", return_value=loaded) as load, \
                patch.object(
                    paper_plot,
                    "_plot_multitask_sibling_fixed_point_geometry_representation",
                ) as render, \
                patch.object(
                    paper_plot,
                    "_plot_multitask_long_delay_endpoint_representation",
                ) as endpoint_render:
            paper_plot.plot_multitask_delaydm_fixed_point_geometry()

        self.assertEqual(
            [call.args[1] for call in render.call_args_list],
            ["delaydm1", "delaydm1", "dmcgo", "dmcgo"],
        )
        self.assertEqual(
            [call.args[0].name for call in load.call_args_list],
            [
                f"delaydm1_delay_pc_projections_{paper_plot.DELAYDM_ANAME}.pkl",
                f"dmcgo_delay_pc_projections_{paper_plot.DMCGO_ANAME}.pkl",
                f"delaydm1_long_delay_endpoint_pc_projections_"
                f"{paper_plot.DELAYDM_ANAME}.pkl",
                f"dmcgo_long_delay_endpoint_pc_projections_"
                f"{paper_plot.DMCGO_ANAME}.pkl",
            ],
        )
        self.assertEqual(
            [call.args[1] for call in endpoint_render.call_args_list],
            ["delaydm1", "delaydm1", "dmcgo", "dmcgo"],
        )
        self.assertEqual(
            [call.args[0].parent for call in load.call_args_list],
            [paper_plot.TWO_IN_MULTIPLES_DIR / aname for aname in (
                paper_plot.DELAYDM_ANAME, paper_plot.DMCGO_ANAME,
                paper_plot.DELAYDM_ANAME, paper_plot.DMCGO_ANAME,
            )],
        )
        self.assertIn("--families delaydm1",
                      paper_plot._multitask_sibling_hint("delaydm1"))
        self.assertIn("--families dmcgo",
                      paper_plot._multitask_sibling_hint("dmcgo"))
        self.assertIn("--method gradient",
                      paper_plot._multitask_sibling_hint("dmcgo"))
        self.assertIn(
            "--method long_delay_endpoint",
            paper_plot._multitask_sibling_hint(
                "dmcgo", "long_delay_endpoint"),
        )

    def test_two_in_multiple_uses_family_runs_for_metrics_and_hints(self):
        runs = {
            "delaydm1": "everything_seed408_L21e4+hidden300+batch128+angle",
            "dmcgo": "everything_seed921_L21e4+hidden300+batch128+angle",
        }
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        for family, aname in runs.items():
            rules = paper_plot._MULTITASK_SIBLING_FAMILIES[family]
            data = [
                {"results": {"longdelay": {
                    "fixed_WM": points + offset,
                    "stim": np.arange(3),
                }}}
                for offset in (0.0, 1.0)
            ]
            with self.subTest(family=family), \
                    patch.object(paper_plot, "DELAYDM_ANAME", runs["delaydm1"]), \
                    patch.object(paper_plot, "DMCGO_ANAME", runs["dmcgo"]), \
                    patch.object(paper_plot, "_load_pkl_or_skip",
                                 side_effect=data) as load:
                metrics = paper_plot._sibling_alignment_metrics(family, rules)
                self.assertEqual(metrics["n_pairs"], 3)
                self.assertEqual(
                    [call.args[0] for call in load.call_args_list],
                    [paper_plot.TWO_IN_MULTIPLES_DIR / aname
                     / f"fixed_points_grad_{aname}_{rule}.pkl" for rule in rules],
                )
                seed = 408 if family == "delaydm1" else 921
                for method in ("gradient", "long_delay_endpoint"):
                    self.assertEqual(
                        paper_plot._multitask_sibling_hint(family, method),
                        "Run: python multiple_task/sibling_delay_analysis.py "
                        f"--seed {seed} --feature L21e4 --families {family} "
                        f"--method {method}",
                    )

    def test_two_in_multiple_panels_are_square_without_titles_or_ticks(self):
        points = np.arange(48, dtype=float).reshape(8, 6) / 100
        entry = {
            "proj": points,
            "trajectory_proj": np.stack([2 * points, points], axis=1),
            "task_idx": np.repeat([0, 1], 4),
            "stim_idx": np.tile(np.arange(4), 2),
            "group_labels": np.tile([0, 0, 1, 1], 2),
        }
        data = {"representations": {"e_modulation": entry, "hidden": entry}}
        metrics = {"translation_explained": 0.9, "geometry_r": 0.8, "n_pairs": 4}
        with patch.object(paper_plot, "_load_pkl_or_skip", return_value=data), \
                patch.object(paper_plot, "_sibling_alignment_metrics",
                             return_value=metrics), \
                patch.object(paper_plot, "_best_sibling_endpoint_pc_pair",
                             return_value=((3, 4), 1.0, "test score")), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_multitask_delaydm_fixed_point_geometry()

        self.assertEqual(save_fig.call_count, 10)
        legend_names = set()
        for call in save_fig.call_args_list:
            figure, output_path = call.args[:2]
            self.addCleanup(paper_plot.plt.close, figure)
            with self.subTest(figure=output_path.name):
                if output_path.stem.endswith("_legend"):
                    legend_names.add(output_path.name)
                    self.assertEqual(len(figure.axes), 0)
                    self.assertEqual(len(figure.legends), 1)
                    family = ("delaydm1" if "delaydm" in output_path.name
                              else "dmcgo")
                    legend = figure.legends[0]
                    self.assertEqual(
                        [text.get_text() for text in legend.get_texts()],
                        [paper_plot._TASK_DISPLAY[rule]
                         for rule in paper_plot._MULTITASK_SIBLING_FAMILIES[family]],
                    )
                    self.assertEqual(
                        [line.get_marker() for line in legend.get_lines()],
                        list(paper_plot._MULTITASK_RULE_MARKERS),
                    )
                    self.assertEqual(
                        [line.get_linestyle() for line in legend.get_lines()],
                        ["-", "--"],
                    )
                    continue
                self.assertFalse(figure.legends)
                original_limits = [(axis.get_xlim(), axis.get_ylim())
                                   for axis in figure.axes]
                figure.canvas.draw()
                for axis, (xlim, ylim) in zip(figure.axes, original_limits):
                    self.assertEqual(axis.get_box_aspect(), 1)
                    bounds = axis.get_window_extent()
                    self.assertAlmostEqual(bounds.width, bounds.height, places=5)
                    np.testing.assert_allclose(axis.get_xlim(), xlim)
                    np.testing.assert_allclose(axis.get_ylim(), ylim)
                    if "long_delay_endpoint" in output_path.name:
                        self.assertIsNone(axis.get_legend())
                    for location in ("left", "center", "right"):
                        self.assertEqual(axis.get_title(loc=location), "")
                    for minor in (False, True):
                        self.assertEqual(axis.get_xticks(minor=minor).size, 0)
                        self.assertEqual(axis.get_yticks(minor=minor).size, 0)
                        self.assertEqual(axis.get_xticklabels(minor=minor), [])
                        self.assertEqual(axis.get_yticklabels(minor=minor), [])
                    self.assertTrue(axis.get_xlabel().startswith("Joint Delay PC"))

        self.assertEqual(legend_names, {
            "multitask_delaydm_long_delay_endpoint_legend.png",
            "multitask_dmcgo_long_delay_endpoint_legend.png",
        })

    def test_long_delay_endpoint_separate_legends_respect_global_toggle(self):
        points = np.arange(48, dtype=float).reshape(8, 6) / 100
        entry = {
            "proj": points,
            "trajectory_proj": np.stack([2 * points, points], axis=1),
            "task_idx": np.repeat([0, 1], 4),
            "stim_idx": np.tile(np.arange(4), 2),
            "group_labels": np.tile([0, 0, 1, 1], 2),
        }
        data = {"representations": {"e_modulation": entry, "hidden": entry}}
        with patch.object(paper_plot, "SHOW_LEGEND", False), \
                patch.object(paper_plot, "_best_sibling_endpoint_pc_pair",
                             return_value=((3, 4), 1.0, "test score")), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            for family, rules in paper_plot._MULTITASK_SIBLING_FAMILIES.items():
                for plot_name, suffix in (("e_modulation", "emodulation"),
                                          ("hidden", "hidden")):
                    rendered = paper_plot._plot_multitask_long_delay_endpoint_representation(
                        data, family, rules, plot_name, suffix)
                    self.assertTrue(rendered)

        self.assertEqual(save_fig.call_count, 4)
        for call in save_fig.call_args_list:
            figure, output_path = call.args[:2]
            self.addCleanup(paper_plot.plt.close, figure)
            self.assertFalse(output_path.stem.endswith("_legend"))
            self.assertFalse(figure.legends)
            self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))

    def test_long_delay_endpoint_selects_best_task_specific_pc_pair(self):
        labels = np.repeat(np.arange(4), 4)
        centers = np.array([
            [-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0],
        ])
        points = np.zeros((labels.size, 6), dtype=float)
        offsets = np.tile(np.array([[-0.05, 0.0], [0.05, 0.0],
                                    [0.0, -0.05], [0.0, 0.05]]), (4, 1))
        points[:, 2:4] = centers[labels] + offsets
        entry = {"proj": points, "stim_idx": labels}

        pair, score, metric = paper_plot._best_sibling_endpoint_pc_pair(
            entry, "delaydm1")

        self.assertEqual(pair, (3, 4))
        self.assertGreater(score, 0.9)
        self.assertEqual(
            metric, "direction separation / within-direction dispersion")

        dmc_entry = {
            "proj": points,
            "group_labels": labels % 2,
            "task_idx": np.tile([0, 1], labels.size // 2),
        }
        _, _, dmc_metric = paper_plot._best_sibling_endpoint_pc_pair(
            dmc_entry, "dmcgo")
        self.assertEqual(
            dmc_metric, "cross-task category balanced accuracy")

    def test_long_delay_endpoint_figure_has_default_and_best_pc_panels(self):
        labels = np.repeat(np.arange(4), 4)
        centers = np.array([
            [-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0],
        ])
        points = np.zeros((labels.size, 6), dtype=float)
        offsets = np.tile(np.array([[-0.05, 0.0], [0.05, 0.0],
                                    [0.0, -0.05], [0.0, 0.05]]), (4, 1))
        points[:, 2:4] = centers[labels] + offsets
        trajectories = np.stack(
            [100.0 * points, 10.0 * points, points], axis=1)
        data = {"representations": {"e_modulation": {
            "proj": points,
            "trajectory_proj": trajectories,
            "task_idx": np.repeat([0, 1], labels.size // 2),
            "stim_idx": labels,
            "task_names": ["delaydm1", "delaydm2"],
            "explained_variance_ratio": np.arange(6, 0, -1) / 21,
        }}}

        with patch.object(paper_plot, "_save_fig") as save_fig:
            rendered = paper_plot._plot_multitask_long_delay_endpoint_representation(
                data, "delaydm1", ("delaydm1", "delaydm2"),
                "e_modulation", "emodulation")

        self.assertTrue(rendered)
        figure = save_fig.call_args.args[0]
        for call in save_fig.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        self.assertEqual(len(figure.axes), 2)
        self.assertTrue(figure.axes[0].get_xlabel().startswith("Joint Delay PC1"))
        self.assertTrue(figure.axes[0].get_ylabel().startswith("Joint Delay PC2"))
        self.assertTrue(figure.axes[1].get_xlabel().startswith("Joint Delay PC3"))
        self.assertTrue(figure.axes[1].get_ylabel().startswith("Joint Delay PC4"))
        self.assertEqual(figure.axes[0].get_title(), "")
        self.assertEqual(figure.axes[1].get_title(), "")
        # The first trajectory sample is ~400 units away, but axes are framed
        # automatically around the ~4-unit endpoints with the expanded
        # DelayDM effective-modulation margin.
        expected_xlim, expected_ylim = paper_plot._adaptive_pc_limits(
            points, 2, 3,
            padding=paper_plot._MULTITASK_DELAYDM_EMODULATION_LIMIT_PADDING)
        np.testing.assert_allclose(figure.axes[1].get_xlim(), expected_xlim)
        np.testing.assert_allclose(figure.axes[1].get_ylim(), expected_ylim)
        self.assertLess(figure.axes[1].get_xlim()[1], 10.0)
        self.assertLess(figure.axes[1].get_ylim()[1], 10.0)

    def test_dmcgo_emodulation_uses_explicit_axis_limits(self):
        group_labels = np.tile(np.repeat([0, 1], 4), 2)
        points = np.zeros((group_labels.size, 6), dtype=float)
        points[:, 2] = np.where(group_labels == 0, -0.2, 0.2)
        points[:, 3] = np.tile([0.0, 0.1], group_labels.size // 2)
        trajectories = np.stack(
            [100.0 * points, 10.0 * points, points], axis=1)
        data = {"representations": {"e_modulation": {
            "proj": points,
            "trajectory_proj": trajectories,
            "task_idx": np.repeat([0, 1], group_labels.size // 2),
            "stim_idx": np.tile(np.arange(8), 2),
            "group_labels": group_labels,
            "task_names": ["dmcgo", "dmcnogo"],
        }}}

        with patch.object(paper_plot, "_save_fig") as save_fig, \
                patch.object(
                    paper_plot, "_best_sibling_endpoint_pc_pair",
                    return_value=((3, 4), 1.0, "cross-task category balanced accuracy")), \
                patch.object(paper_plot, "_adaptive_pc_limits") as adaptive_limits:
            rendered = paper_plot._plot_multitask_long_delay_endpoint_representation(
                data, "dmcgo", ("dmcgo", "dmcnogo"),
                "e_modulation", "emodulation")

        self.assertTrue(rendered)
        adaptive_limits.assert_not_called()
        figure = save_fig.call_args.args[0]
        for call in save_fig.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        self.assertIsNone(figure.axes[0].get_legend())
        self.assertIsNone(figure.axes[1].get_legend())
        expected_limits = (
            ((-5.2, 5.2), (-0.55, 0.75)),
            ((-0.52, 0.52), (-0.29, 0.49)),
        )
        for axis, (expected_xlim, expected_ylim) in zip(figure.axes, expected_limits):
            np.testing.assert_allclose(axis.get_xlim(), expected_xlim)
            np.testing.assert_allclose(axis.get_ylim(), expected_ylim)

    def test_dmcgo_hidden_zooms_to_endpoint_limits(self):
        group_labels = np.tile(np.repeat([0, 1], 4), 2)
        points = np.zeros((group_labels.size, 6), dtype=float)
        points[:, 3] = np.where(group_labels == 0, -4.0, 4.0)
        points[:, 3] += np.tile([-0.05, 0.05], group_labels.size // 2)
        trajectories = np.stack(
            [100.0 * points, 10.0 * points, points], axis=1)
        data = {"representations": {"hidden": {
            "proj": points,
            "trajectory_proj": trajectories,
            "task_idx": np.repeat([0, 1], group_labels.size // 2),
            "stim_idx": np.tile(np.arange(8), 2),
            "group_labels": group_labels,
            "task_names": ["dmcgo", "dmcnogo"],
        }}}

        with patch.object(paper_plot, "_save_fig") as save_fig:
            rendered = paper_plot._plot_multitask_long_delay_endpoint_representation(
                data, "dmcgo", ("dmcgo", "dmcnogo"), "hidden", "hidden")

        self.assertTrue(rendered)
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        best_pair, _, _ = paper_plot._best_sibling_endpoint_pc_pair(
            data["representations"]["hidden"], "dmcgo")
        for axis, (pc_x, pc_y) in zip(figure.axes, ((1, 2), best_pair)):
            expected_xlim, expected_ylim = paper_plot._adaptive_pc_limits(
                points, pc_x - 1, pc_y - 1,
                padding=paper_plot._MULTITASK_ENDPOINT_LIMIT_PADDING)
            np.testing.assert_allclose(axis.get_xlim(), expected_xlim)
            np.testing.assert_allclose(axis.get_ylim(), expected_ylim)
        self.assertLess(figure.axes[1].get_ylim()[1], 10.0)

    def test_dmcgo_endpoints_color_by_stimulus_in_both_representations(self):
        stim_idx = np.tile(np.arange(8), 2)
        task_idx = np.repeat([0, 1], 8)
        points = np.arange(96, dtype=float).reshape(16, 6) / 1000
        entry = {
            "proj": points,
            "trajectory_proj": np.stack([2 * points, points], axis=1),
            "task_idx": task_idx,
            "stim_idx": stim_idx,
            "group_labels": (stim_idx // 4 + task_idx) % 2,
            "task_names": ["dmcgo", "dmcnogo"],
        }
        expected_colors = np.asarray([
            paper_plot.mpl.colors.to_rgb(paper_plot.stim_color(int(stim), 8))
            for stim in stim_idx
        ])
        expected_legend = [
            paper_plot._TASK_DISPLAY[rule] for rule in entry["task_names"]
        ]
        for plot_name, suffix in (("e_modulation", "emodulation"),
                                  ("hidden", "hidden")):
            with self.subTest(representation=plot_name), \
                    patch.object(paper_plot, "_save_fig") as save_fig:
                rendered = paper_plot._plot_multitask_long_delay_endpoint_representation(
                    {"representations": {plot_name: entry}}, "dmcgo",
                    ("dmcgo", "dmcnogo"), plot_name, suffix)

                self.assertTrue(rendered)
                figure = save_fig.call_args.args[0]
                for call in save_fig.call_args_list:
                    self.addCleanup(paper_plot.plt.close, call.args[0])
                if plot_name == "e_modulation":
                    self.assertEqual(save_fig.call_count, 2)
                    legend_figure = save_fig.call_args_list[0].args[0]
                    self.assertEqual(
                        [text.get_text() for text
                         in legend_figure.legends[0].get_texts()],
                        expected_legend,
                    )
                else:
                    self.assertEqual(save_fig.call_count, 1)
                for axis in figure.axes:
                    self.assertEqual(len(axis.lines), stim_idx.size)
                    np.testing.assert_allclose([
                        paper_plot.mpl.colors.to_rgb(line.get_color())
                        for line in axis.lines
                    ], expected_colors)
                    self.assertEqual(len(axis.collections), 2 * stim_idx.size)
                    np.testing.assert_allclose([
                        collection.get_facecolors()[0, :3]
                        for collection in axis.collections
                    ], np.tile(expected_colors, (2, 1)))
                    self.assertIsNone(axis.get_legend())

    def test_missing_long_delay_endpoint_does_not_skip_gradient_plots(self):
        loaded = {"representations": {}}
        with patch.object(
                paper_plot, "_load_pkl_or_skip",
                side_effect=[loaded, loaded, None, None]), \
                patch.object(
                    paper_plot,
                    "_plot_multitask_sibling_fixed_point_geometry_representation",
                ) as gradient_render, \
                patch.object(
                    paper_plot,
                    "_plot_multitask_long_delay_endpoint_representation",
                ) as endpoint_render:
            paper_plot.plot_multitask_delaydm_fixed_point_geometry()

        self.assertEqual(gradient_render.call_count, 4)
        endpoint_render.assert_not_called()

    def test_old_endpoint_projection_without_trajectory_is_skipped(self):
        data = {"representations": {"hidden": {
            "proj": np.zeros((4, 6)),
        }}}
        rendered = paper_plot._plot_multitask_long_delay_endpoint_representation(
            data, "delaydm1", ("delaydm1", "delaydm2"),
            "hidden", "hidden")
        self.assertFalse(rendered)

    def test_two_in_multiple_does_not_filter_solver_endpoints_by_convergence(self):
        source = "\n".join([
            inspect.getsource(paper_plot._sibling_alignment_metrics),
            inspect.getsource(
                paper_plot._plot_multitask_sibling_fixed_point_geometry_representation),
        ])
        self.assertNotIn("is_fixed", source)

    # One synthetic pair of seeds per trained activation, mirroring the real
    # ordering of the diagnosis: ReLU is positive-only and zero-derivative
    # like sigmoid yet has a low offset, many trial-specific M dimensions and
    # the best accuracy.
    _ACTIVATION_FIXTURE = (
        # activation, offset, saturation, common, trial_dim, accuracy
        ("linear", 0.01, 0.00, 0.25, 18.0, 0.91),
        ("relu", 0.07, 0.96, 0.35, 12.0, 0.94),
        ("softplus", 0.90, 0.95, 0.91, 2.7, 0.71),
        ("sigmoid", 0.94, 0.96, 0.99, 1.6, 0.67),
        ("tanh", 0.01, 0.002, 0.25, 17.0, 0.93),
    )

    @classmethod
    def _activation_diagnosis_records(cls):
        records = []
        for (activation, offset, saturation, common, trial_dim,
                accuracy) in cls._ACTIVATION_FIXTURE:
            for delta in (-0.01, 0.01):
                records.append({
                    "activation": activation,
                    "database_accuracy": accuracy + delta,
                    "embedding": {
                        "offset_energy_fraction": offset + delta / 10,
                    },
                    "hidden": {
                        "derivative_fraction_lt_0p1": saturation + delta / 10,
                    },
                    "final_modulation": {
                        "common_across_trials_energy_fraction": common + delta / 10,
                        "trial_effective_dimension": trial_dim + delta * 10,
                    },
                })
        return records

    def test_activation_diagnosis_specs_match_accuracy_panel(self):
        # Diagnosis panels must show the same five activations, in the same
        # order, as plot_l2e4_activation_accuracy, so they read as companions.
        self.assertEqual(
            [spec[0] for spec in paper_plot._ACTIVATION_DIAGNOSIS_SPECS],
            ["linear", "relu", "softplus", "sigmoid", "tanh"])

    def test_activation_modulation_dimension_figure_is_not_a_percentage(self):
        with patch.object(
                paper_plot, "_load_activation_diagnosis_records",
                return_value=self._activation_diagnosis_records()), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_l2e4_activation_modulation_dimension()

        figure, output_path = save_fig.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        axis = figure.axes[0]
        self.assertEqual(
            output_path.name,
            "multitask_l2e4_activation_modulation_dimension.png")
        self.assertIn("dimension", axis.get_ylabel().lower())
        # Raw participation ratios: the axis starts at zero and reaches the
        # largest fixture value (18) rather than being clipped to 100 %.
        self.assertEqual(axis.get_ylim()[0], 0.0)
        self.assertGreater(axis.get_ylim()[1], 18.0)
        self.assertLess(axis.get_ylim()[1], 25.0)

    def test_activation_diagnosis_percentage_figures_are_registered(self):
        expected = {
            "l2e4_activation_embedding_offset":
                paper_plot.plot_l2e4_activation_embedding_offset,
            "l2e4_activation_modulation_dimension":
                paper_plot.plot_l2e4_activation_modulation_dimension,
            "l2e4_activation_modulation_common_accuracy":
                paper_plot.plot_l2e4_activation_modulation_common_accuracy,
        }
        self.assertNotIn(
            "l2e4_activation_hidden_saturation",
            paper_plot.FIGURES_BY_MODE["acc_plot"])
        for name, function in expected.items():
            self.assertIs(paper_plot.FIGURES_BY_MODE["acc_plot"][name], function)

        with patch.object(
                paper_plot, "_load_activation_diagnosis_records",
                return_value=self._activation_diagnosis_records()), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_l2e4_activation_embedding_offset()

        figure, output_path = save_fig.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        # Five seed scatters plus one error-bar collection per activation.
        self.assertGreaterEqual(len(figure.axes[0].collections), 5)
        self.assertEqual(
            output_path.name,
            "multitask_l2e4_activation_embedding_offset.png",
        )
        self.assertIn("offset energy", figure.axes[0].get_ylabel())
        self.assertLess(figure.axes[0].get_ylim()[0], 0.0)
        self.assertLess(figure.axes[0].get_ylim()[1], 105.0)

    def test_activation_modulation_accuracy_figure_has_seed_scatter_only(self):
        with patch.object(
                paper_plot, "_load_activation_diagnosis_records",
                return_value=self._activation_diagnosis_records()), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_l2e4_activation_modulation_common_accuracy()

        figure, output_path = save_fig.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        axis = figure.axes[0]
        # One scatter collection per activation, no fit line, no annotation.
        self.assertEqual(len(axis.collections), 5)
        self.assertEqual(len(axis.lines), 0)
        self.assertFalse(axis.texts)
        self.assertEqual(axis.get_ylabel(), "Mean accuracy\nacross tasks (%)")
        self.assertEqual(
            output_path.name,
            "multitask_l2e4_activation_modulation_common_accuracy.png",
        )

    def test_pretraining_bound_switches_every_artifact_pattern(self):
        original_bound = paper_plot.PRETRAINING_BOUND
        try:
            self.assertEqual(paper_plot._parse_pretraining_bound("mod1"), "mb1")
            self.assertEqual(paper_plot._parse_pretraining_bound("mb2"), "mb2")

            paper_plot._set_pretraining_bound("mb2")
            self.assertEqual(paper_plot.PRETRAINING_BOUND, "mb2")
            self.assertEqual(
                paper_plot.PRETRAINING_ADDON_NAME,
                "+hidden200+L21e3mb2+batch128+angle",
            )
            with patch.object(Path, "glob", return_value=[]) as glob:
                paper_plot._pretraining_result_pkls()
                glob.assert_called_once_with(
                    "*_dmpn_seed*_+hidden200+L21e3mb2+batch128+angle_result.pkl")
            with patch.object(Path, "glob", return_value=[]) as glob:
                paper_plot._pretraining_combined_pkls("aggregate")
                glob.assert_called_once_with(
                    "*_dmpn_+hidden200+L21e3mb2+batch128+angle_aggregate.pkl")
        finally:
            paper_plot._set_pretraining_bound(original_bound)

    def test_aggregate_cve_panel_includes_delaypro_control(self):
        rulesets = (
            "fdanti", "fdgo", "fdanti_delaygo", "fdgo_delaygo")
        by_ruleset = {
            ruleset: {
                "hidden_stimulus_self": [
                    np.array([0.4, 0.7, 0.9]),
                    np.array([0.5, 0.8, 1.0]),
                ],
                "hidden_stimulus_cross": [
                    np.array([0.1, 0.3, 0.5]),
                    np.array([0.2, 0.4, 0.6]),
                ],
            }
            for ruleset in rulesets
        }
        colors = {
            ruleset: color for ruleset, color in zip(
                rulesets, ("orange", "purple", "red", "blue"))
        }
        labels = {
            "fdanti": "DelayAnti",
            "fdgo": "DelayPro",
            "fdanti_delaygo": "Relevant motif",
            "fdgo_delaygo": "Irrelevant motif",
        }
        figure, axis = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)

        paper_plot._plot_aggregate_cve_panel(
            axis, by_ruleset, "hidden", "stimulus", colors, labels,
            x_lim=3, x_ticks=np.arange(1, 4), show_legend=True,
        )

        self.assertEqual(
            axis.get_legend_handles_labels()[1],
            ["Self", "DelayAnti", "DelayPro", "Relevant motif",
             "Irrelevant motif"],
        )

    def test_pretraining_figures_name_and_color_all_four_conditions(self):
        styles = paper_plot._PRETRAINING_RULESET_STYLES
        self.assertEqual(
            {ruleset: label for ruleset, (label, _) in styles.items()},
            {
                "fdgo_delaygo": "Irrelevant motif",
                "fdanti_delaygo": "Relevant motif",
                "fdanti": "DelayAnti",
                "fdgo": "DelayPro",
            },
        )
        self.assertEqual(len({color for _, color in styles.values()}), 4)
        for function in (
                paper_plot.plot_learning_trajectory,
                paper_plot.plot_transfer_speed,
                paper_plot.plot_rule_vectors,
                paper_plot.plot_pretraining_principal_angles,
                paper_plot.plot_backbone_probe):
            self.assertIn(
                "_PRETRAINING_RULESET_STYLES", inspect.getsource(function))

    def test_pretraining_compact_figure_layout(self):
        self.assertEqual(
            paper_plot._PRETRAINING_TRAJECTORY_FIGSIZE,
            (3.0, 2.2 * 2 / 3),
        )
        self.assertEqual(
            paper_plot._TRANSFER_SPEED_FIGSIZE,
            (3.3, 2.2 * 2 / 3 * 1.1),
        )
        self.assertGreater(
            paper_plot._TRANSFER_SPEED_FIGSIZE[0],
            paper_plot._PRETRAINING_TRAJECTORY_FIGSIZE[0],
        )
        self.assertGreater(
            paper_plot._TRANSFER_SPEED_FIGSIZE[1],
            paper_plot._PRETRAINING_TRAJECTORY_FIGSIZE[1],
        )
        self.assertEqual(paper_plot._TRANSFER_SPEED_YTICKS, (50, 75, 100))
        self.assertLess(paper_plot._BACKBONE_PROBE_FIGSIZE[0], 4.2)
        self.assertLess(paper_plot._BACKBONE_PROBE_FIGSIZE[1], 2.4)

        transfer_source = inspect.getsource(paper_plot.plot_transfer_speed)
        learning_source = inspect.getsource(paper_plot.plot_learning_trajectory)
        backbone_source = inspect.getsource(paper_plot.plot_backbone_probe)
        self.assertIn("_TRANSFER_SPEED_FIGSIZE", transfer_source)
        self.assertIn("_PRETRAINING_TRAJECTORY_FIGSIZE", learning_source)
        self.assertIn("set_yticks(_TRANSFER_SPEED_YTICKS)", transfer_source)
        self.assertIn("_BACKBONE_PROBE_FIGSIZE", backbone_source)


if __name__ == "__main__":
    unittest.main()
