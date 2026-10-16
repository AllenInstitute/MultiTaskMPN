"""Checks for the public paper-figure mode grouping."""

import copy
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
    @staticmethod
    def modulation_entry(values, row_labels, col_labels):
        """A minimal saved modulation clustering entry (two G groupings)."""
        n_rows, n_cols = values.shape
        result = {"row_order": np.arange(n_rows)[::-1], "col_order": np.arange(n_cols),
                  "row_tol_labels": row_labels, "col_tol_labels": col_labels,
                  "row_tol_k": len(set(row_labels)), "col_tol_k": len(set(col_labels))}
        return {"cell_vars_rules_sorted_norm": values,
                "tb_break_name": np.array(["delaygo-stim1", "delaygo-go1", "fdgo-go1"][:n_rows]),
                "result_all_lst": [result, result]}

    def test_abs_weighted_clustered_modulation_uses_log_scale_and_own_colorbar(self):
        values = np.array([[0.0, 2e-3, 0.5], [1e-3, 0.0, 4.0], [2.0, 3.0, 0.0]])
        cache = {
            "modulation_all_normalized": self.modulation_entry(
                values / values.max(axis=0, keepdims=True), [1, 1, 2], [1, 2, 2]),
            "modulation_all_abs_weighted_unnormalized": self.modulation_entry(
                values, [1, 2, 2], [1, 1, 2]),
        }
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=cache), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_clustered_modulation_abs_weighted()
            paper_plot.plot_clustered_modulation()
        names = [call.args[1].name for call in save_fig.call_args_list]
        self.assertEqual(names, ["multitask_clustered_modulation_abs_weighted.png",
                                 "multitask_clustered_modulation_abs_weighted_colorbar.png",
                                 "multitask_clustered_modulation_normalized.png"])
        for call in save_fig.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        heatmap = save_fig.call_args_list[0].args[0].axes[0].collections[0]
        self.assertIsInstance(heatmap.norm, paper_plot.mpl.colors.LogNorm)
        self.assertEqual((heatmap.norm.vmin, heatmap.norm.vmax), (1e-4, 4.0))
        # Silent synapses are floored to the color minimum, never masked.
        drawn = heatmap.get_array()
        self.assertEqual(float(np.min(drawn)), 1e-4)
        self.assertFalse(np.ma.is_masked(drawn))
        colorbar_norm = save_fig.call_args_list[1].args[0].axes[0].collections[-1].norm
        self.assertIsInstance(colorbar_norm, paper_plot.mpl.colors.LogNorm)
        normalized = save_fig.call_args_list[2].args[0].axes[0].collections[0]
        self.assertNotIsInstance(normalized.norm, paper_plot.mpl.colors.LogNorm)
        self.assertEqual((normalized.norm.vmin, normalized.norm.vmax),
                         paper_plot._MULTITASK_HEATMAP_VLIM)

    def test_abs_weighted_clustered_modulation_skips_caches_without_the_variant(self):
        cache = {"modulation_all_normalized": self.modulation_entry(
            np.ones((2, 2)), [1, 2], [1, 2])}
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=cache), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save_fig, \
                patch("builtins.print") as output:
            paper_plot.plot_clustered_modulation_abs_weighted()
        save_fig.assert_not_called()
        self.assertIn("modulation_all_abs_weighted_unnormalized", output.call_args.args[0])

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
            (paper_plot.plot_overmembership_abs_weighted,
             "modulation_all_abs_weighted_prepost_belonging_{aname}_unnormalized.pkl",
             "overmembership_abs_weighted.png"),
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
                                           ["Pre\ncluster", "Post\ncluster", "Both\ncluster", "Neither"])
                        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                                         expected_labels)
                        # Upright 7 pt tick labels (two lines on the cluster row).
                        self.assertTrue(all(label.get_rotation() == 0 and label.get_fontsize() == 7
                                            for label in axis.get_xticklabels()))
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
                        self.assertEqual(scatters, [])
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

    def test_lesion_figures_have_their_own_lesion_mode(self):
        expected = {
            "lesion_heatmap": paper_plot.plot_lesion_heatmap,
            "lesion_cluster_sizes": paper_plot.plot_lesion_cluster_sizes,
            "cluster_corr_vs_lesion": paper_plot.plot_cluster_corr_vs_lesion,
            "cluster_corr_vs_lesion_weighted": paper_plot.plot_cluster_corr_vs_lesion_weighted,
            "cluster_corr_vs_lesion_seeds": paper_plot.plot_cluster_corr_vs_lesion_seeds,
            "om_vs_lesion": paper_plot.plot_om_vs_lesion,
            "om_vs_lesion_seeds": paper_plot.plot_om_vs_lesion_seeds,
            "plasticity_share": paper_plot.plot_plasticity_share,
            "plasticity_share_cohort": paper_plot.plot_plasticity_share_cohort,
            "causal_vs_activity_tasks": paper_plot.plot_causal_vs_activity,
            "causal_vs_activity_seeds": paper_plot.plot_causal_vs_activity_seeds,
            "task_specificity": paper_plot.plot_task_specificity,
        }
        self.assertEqual(paper_plot.FIGURES_BY_MODE["lesion"], expected)
        self.assertTrue(set(expected).isdisjoint(
            paper_plot.FIGURES_BY_MODE["multiple_tasks"]))
        self.assertEqual(set(paper_plot.FIGURES_BY_MODE["multiple_tasks"]), {
            "input", "hidden", "modulation", "modulation_abs_weighted", "heatmap_colorbar",
            "overmembership_norm", "overmembership_unnorm",
            "overmembership_weighted", "overmembership_var_weighted",
            "overmembership_abs_weighted", "overmembership_examples",
            "input_weight_correlation",
        })
        for name, function in expected.items():
            self.assertIs(paper_plot.ALL_FIGURES[name], function)
        names = [name for group in paper_plot.FIGURES_BY_MODE.values()
                 for name in group]
        self.assertEqual(len(names), len(set(names)))

    def test_lesion_cli_dispatch_and_existing_single_figure_names(self):
        cases = [
            (["lesion"], set(paper_plot.FIGURES_BY_MODE["lesion"])),
            (["multiple_tasks"], set(paper_plot.FIGURES_BY_MODE["multiple_tasks"])),
            (["multiple_tasks", "lesion"],
             set(paper_plot.FIGURES_BY_MODE["multiple_tasks"])
             | set(paper_plot.FIGURES_BY_MODE["lesion"])),
        ]
        cases.extend((["--only", name], {name})
                     for name in paper_plot.FIGURES_BY_MODE["lesion"])
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
                if "lesion" in arguments:
                    self.assertTrue(any(
                        f"lesion: {paper_plot.LESION_ANAME}" in str(call.args[0])
                        for call in output.call_args_list if call.args))

    def test_state_space_figures_have_their_own_mode(self):
        expected = {
            "state_space_combined",
            "state_space_untrained",
            "state_space_dist_angle",
            "state_space_r_values",
            "state_space_rdm",
            "state_space_rdm_example",
            "state_space_category_test",
            "state_space_category_vs_untrained",
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
                "scatter": {
                    name: {"permutation": {"p_value": 0.01 * (offset + index)}}
                    for offset, name in enumerate(("hidden", "mod", "eff_mod"), start=1)
                },
            }
        # A run whose cache predates the permutation test is left out, and so
        # is a run of a strength outside the four cohorts.
        result_dict["legacy"] = {"l2_info": 1e-4,
                                 "rval_dict": {"hidden": (0.9, 1.0, 0.01)}}
        result_dict["other"] = {"l2_info": 1e-6,
                                "rval_dict": {"hidden": (0.9, 1.0, 0.01)},
                                "scatter": {"hidden": {"permutation": {"p_value": 0.5}}}}

        data_types = ["hidden", "mod", "eff_mod"]
        grouped = paper_plot._state_space_r_values_by_l2(
            result_dict, data_types)

        self.assertEqual(tuple(grouped), paper_plot.STATE_SPACE_L2_STRENGTHS)
        for index, strength in enumerate(paper_plot.STATE_SPACE_L2_STRENGTHS):
            self.assertEqual([len(grouped[strength][dt]) for dt in data_types], [1, 1, 1])
            np.testing.assert_allclose(
                [grouped[strength][dt][0]["r"] for dt in data_types],
                [0.1 + index, 0.2 + index, 0.3 + index],
            )
            np.testing.assert_allclose(
                [grouped[strength][dt][0]["p_perm"] for dt in data_types],
                [0.01 * (offset + index) for offset in (1, 2, 3)],
            )
            self.assertEqual(grouped[strength]["hidden"][0]["aname"], f"run{index}")

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

    def test_state_space_strong_l2_panels_select_within_the_1e2_cohort(self):
        good = self._state_space_pca([[0, 0, -5], [0, 0, -4], [0, 0, 4], [0, 0, 5]])
        mixed = self._state_space_pca([[0, 0, -5], [0, 0, 4], [0, 0, -4], [0, 0, 5]])
        result_dict = {"strong_mixed": {"l2_info": 1e-2}, "strong_good": {"l2_info": 1e-2},
                       "weak_good": {"l2_info": 1e-4}}
        pca_data = {"strong_mixed": mixed, "strong_good": good, "weak_good": good}
        self.assertEqual(paper_plot._l2_feature_tag(1e-2), "L21e2")
        self.assertEqual(paper_plot._l2_feature_tag(1e-4), "L21e4")
        selected = paper_plot._best_state_space_trial(result_dict, pca_loader=pca_data.get,
                                                      strength=1e-2)
        self.assertEqual(selected[0], "strong_good")
        report = paper_plot._state_space_centroid_ranking(result_dict, pca_loader=pca_data.get,
                                                          strength=1e-2)
        self.assertEqual(report["l2"], 1e-2)
        self.assertEqual([row["aname"] for row in report["rankings"]],
                         ["strong_good", "strong_mixed"])
        # The PCA figures select one run per cohort, the 1e-4 example on the
        # left and the 1e-2 run on the right, and keep the unsuffixed names.
        selections = {1e-4: ("weak_good", 0.8, {}), 1e-2: ("strong_good", 0.8, {})}
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             side_effect=selections.get) as selector, \
                patch.object(paper_plot, "_load_state_space_pca",
                             side_effect=pca_data.get) as loader, \
                patch.object(paper_plot, "_plot_state_space_figure") as plot_figure:
            paper_plot.plot_state_space_combined()
        self.assertEqual([call.args[0] for call in selector.call_args_list], [1e-4, 1e-2])
        self.assertEqual([call.args[0] for call in loader.call_args_list],
                         ["weak_good", "strong_good"])
        self.assertEqual([call.args[3] for call in plot_figure.call_args_list],
                         ["state_space_hidden.png", "state_space_mod.png",
                          "state_space_eff_mod.png"])
        for call in plot_figure.call_args_list:
            self.assertEqual([strength for strength, _ in call.args[0]], [1e-4, 1e-2])
            self.assertEqual([data is good for _, data in call.args[0]], [True, True])
        # Without the strong cohort nothing is drawn: a figure needs both panels.
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             side_effect={1e-4: ("weak_good", 0.8, {})}.get), \
                patch.object(paper_plot, "_load_state_space_pca", side_effect=pca_data.get), \
                patch.object(paper_plot, "_plot_state_space_figure") as plot_figure, \
                patch("builtins.print"):
            paper_plot.plot_state_space_combined()
        plot_figure.assert_not_called()

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
                patch.object(paper_plot, "_plot_state_space_figure") as plot_figure:
            paper_plot.plot_state_space_combined()

        self.assertEqual(
            [call.args[1:] for call in plot_figure.call_args_list],
            [
                ("hidden", "Hidden state", "state_space_hidden.png"),
                ("mod", "Modulation", "state_space_mod.png"),
                ("eff_mod", "Eff. modulation", "state_space_eff_mod.png"),
            ],
        )

    def test_state_space_panels_are_square_at_the_sweep_subpanel_size(self):
        rules = ["fdgo", "delaydm1", "dmcgo"]
        rng = np.random.RandomState(0)
        data = {
            "all_rules": rules,
            "pca_results": {"eff_mod": {
                "X_2d": rng.normal(size=(12, 2)),
                "ctx_rule_labels": np.repeat(np.arange(3), 4),
            }},
        }
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot._plot_state_space_figure(
                [(1e-4, data), (1e-2, data)], "eff_mod", "Eff. modulation",
                "state_space_eff_mod.png")
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(Path(save_fig.call_args.args[1]).name, "multitask_state_space_eff_mod.png")
        self.assertEqual(tuple(figure.get_size_inches()), paper_plot._STATE_SPACE_FIGSIZE)
        # One panel per cohort, titled by its L2; the representation is named
        # once, on the left panel; one color key for the whole figure.
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         [r"$L_2 = 10^{-4}$", r"$L_2 = 10^{-2}$"])
        self.assertEqual([axis.get_ylabel() for axis in figure.axes],
                         ["Eff. modulation\nPC2", "PC2"])
        self.assertEqual(len(figure.legends), 1)
        self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()],
                         [label for label, _ in paper_plot._STATE_SPACE_LEGEND])
        # Each drawn box matches a rule-cue sweep subpanel (about 1.6 in).
        figure.canvas.draw()
        for axis in figure.axes:
            self.assertEqual(axis.get_box_aspect(), 1)
            box = axis.get_window_extent().transformed(
                figure.dpi_scale_trans.inverted())
            self.assertAlmostEqual(box.width, box.height, places=2)
            self.assertGreater(box.width, 1.5)
            self.assertLess(box.width, 1.75)

    @staticmethod
    def _untrained_cache(untrained=True):
        rules = list(paper_plot._RULE_MOTIF)
        rng = np.random.RandomState(1)
        labels = np.repeat(np.arange(len(rules)), 4)
        return {
            "all_rules": rules, "untrained": untrained, "init_seed": 0,
            "pca_results": {key: {
                "X_2d": rng.normal(size=(labels.size, 2)),
                "ctx_rule_labels": labels,
                "task_centroids": rng.normal(size=(len(rules), 5)),
            } for key in ("hidden", "mod", "eff_mod")},
        }

    def _draw_untrained(self, cache):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if cache is not None:
                with (root / "state_space_pca_run_init0_noise0.01.pkl").open("wb") as stream:
                    pickle.dump(cache, stream)
            with patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "STATE_SPACE_UNTRAINED_DIR", root), \
                    patch.object(paper_plot, "_load_best_state_space_trial",
                                 return_value=("run", 0.1, {})), \
                    patch.object(paper_plot, "_save_fig") as save_fig, \
                    patch("builtins.print") as output:
                paper_plot.plot_state_space_untrained()
        return save_fig, output

    def test_state_space_untrained_figure_has_one_panel_per_representation(self):
        save_fig, _ = self._draw_untrained(self._untrained_cache())
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(Path(save_fig.call_args.args[1]).name,
                         "multitask_state_space_untrained.png")
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["Hidden state", "Modulation", "Eff. modulation"])
        self.assertEqual([axis.get_ylabel() for axis in figure.axes],
                         ["Untrained\nPC2", "PC2", "PC2"])
        self.assertEqual(len(figure.legends), 1)
        extra = save_fig.call_args.kwargs["extra"]
        self.assertIn("config of run, init seed 0", extra)
        self.assertIn("eff_mod S = ", extra)
        # The trained PCA figures' panel size (about 1.6 in square boxes).
        figure.canvas.draw()
        for axis in figure.axes:
            box = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
            self.assertAlmostEqual(box.width, box.height, places=2)
            self.assertGreater(box.width, 1.5)
            self.assertLess(box.width, 1.75)

    def test_state_space_untrained_figure_skips_missing_or_trained_caches(self):
        for cache, reason in ((None, "no untrained-network PCA cache"),
                              (self._untrained_cache(untrained=False), "not marked")):
            with self.subTest(reason=reason):
                save_fig, output = self._draw_untrained(cache)
                save_fig.assert_not_called()
                message = output.call_args.args[0]
                self.assertIn(reason, message)
                self.assertIn("state_space_shift.py --untrained --checkpoint "
                              "multiple_tasks/savednet_run.pt --init-seed 0", message)

    def test_state_space_dist_angle_renders_saved_nonzero_intercepts(self):
        entry = {"rval_dict": {}, "scatter": {}}
        for key, intercept in (("hidden", 15.), ("eff_mod", -3.), ("eff_mod_hebb", 2.)):
            entry["rval_dict"][key] = (0.9, 2., 0.02)
            entry["scatter"][key] = {
                "dists": np.array([2., 3., 4.]),
                "angles_deg": np.array([4., 6., 8.]) + intercept,
                "regression": {"through_origin": False, "intercept": intercept},
                "permutation": {"p_value": 0.004},
            }
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             return_value=("selected", 0.5, entry)), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_state_space_dist_angle()

        save_fig.assert_called_once()
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(len(figure.axes), 3)
        for axis, intercept in zip(figure.axes, (15., -3., 2.)):
            line = axis.lines[0]
            np.testing.assert_allclose(line.get_ydata(),
                                       intercept + 2 * line.get_xdata())
            legend_text = axis.get_legend().get_texts()[0].get_text()
            # The saved task-label permutation p is reported, never the
            # nominal OLS p of the (r, slope, p) tuple.
            self.assertEqual(legend_text, "r = 0.90\npermutation p = 0.004")

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

    @staticmethod
    def _state_space_structure(rules, centers, p_value=0.0005):
        """A state_space_structure.py per-run cache for two representations,
        with the tree and distances computed from `centers` and the saved
        statistics set by hand."""
        from core.state_space_geometry import task_centroid_distances, task_centroid_linkage

        distances = task_centroid_distances(centers)
        tree = task_centroid_linkage(distances)
        null = np.linspace(-0.2, 0.2, 50)
        factor_stats = {
            "mean_cosine": 0.8, "shared_variance_fraction": 0.9, "p_value": p_value,
            "pairwise_cosines": np.array([0.7, 0.8, 0.9]), "null": null,
            "null_quantile_95": 0.15, "null_quantile_025": -0.18, "null_quantile_975": 0.18,
            "pairs": [("fdgo", "fdanti"), ("delaygo", "delayanti"), ("reactgo", "reactanti")],
        }
        pooled = {**factor_stats, "mean_cosine": 0.5, "p_value": 0.3}
        pooled.pop("pairwise_cosines")
        pooled.pop("pairs")
        representation = {
            "distances": distances, "linkage": tree["linkage"],
            "leaf_order": tree["leaf_order"], "linkage_method": "average",
            "separation": {"score": 0.42, "p_value": p_value, "null": null,
                           "null_quantile_95": 0.1},
            "factor_vectors": {
                "factors": {"pro to anti": dict(factor_stats),
                            "delay to memory": {**factor_stats, "p_value": 0.4}},
                "pooled": pooled, "skipped_factors": [],
            },
        }
        return {"aname": "selected", "all_rules": list(rules), "l2_info": 1e-4,
                "representations": {"hidden": representation,
                                    "eff_mod": copy.deepcopy(representation)}}

    def test_state_space_rdm_example_orders_the_heatmap_by_the_drawn_tree(self):
        rules = ["fdgo", "delaydm1", "delaygo", "dmcgo"]
        centers = np.array([[0., 0.], [10., 0.], [1., 0.], [10., 1.]])
        structure = self._state_space_structure(rules, centers)
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             return_value=("selected", 0.5, {})), \
                patch.object(paper_plot, "_load_state_space_structure",
                             return_value=structure) as loader, \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_state_space_rdm_example()
        loader.assert_called_once_with("selected")
        # Effective modulation only, followed by its own colorbar figure.
        self.assertEqual([Path(call.args[1]).name for call in save_fig.call_args_list],
                         ["multitask_state_space_rdm_example_eff_mod.png",
                          "multitask_state_space_rdm_example_colorbar_eff_mod.png"])
        figure = save_fig.call_args_list[0].args[0]
        for call in save_fig.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        self.assertEqual(len(figure.axes), 2)          # tree and matrix, no colorbar
        tree_axis, heat_axis = figure.axes
        order = list(structure["representations"]["eff_mod"]["leaf_order"])
        # Close tasks sit next to each other, and the heatmap follows the tree.
        self.assertEqual(abs(order.index(0) - order.index(2)), 1)
        self.assertEqual(abs(order.index(1) - order.index(3)), 1)
        image = heat_axis.images[0].get_array()
        distances = structure["representations"]["eff_mod"]["distances"]
        ordered = distances[np.ix_(order, order)]
        lower = np.tril_indices(len(order))
        np.testing.assert_allclose(np.asarray(image)[lower], ordered[lower])
        # Only the lower triangle is drawn: the cells above the diagonal are masked.
        self.assertTrue(np.all(np.ma.getmaskarray(image)[np.triu_indices(len(order), 1)]))
        self.assertFalse(np.any(np.ma.getmaskarray(image)[lower]))
        colorbar_figure = save_fig.call_args_list[1].args[0]
        colorbar_axis = colorbar_figure.axes[0]
        self.assertEqual([label.get_text() for label in colorbar_axis.get_xticklabels()],
                         ["0", f"{ordered.max():.1f}"])
        self.assertEqual(colorbar_axis.get_xlabel(), "Centroid distance")
        self.assertEqual([label.get_text() for label in heat_axis.get_xticklabels()],
                         [paper_plot._TASK_DISPLAY[rules[index]] for index in order])
        for label, index in zip(heat_axis.get_yticklabels(), order):
            np.testing.assert_allclose(
                paper_plot.mpl.colors.to_rgb(label.get_bbox_patch().get_facecolor()),
                paper_plot.mpl.colors.to_rgb(paper_plot._RULE_MOTIF[rules[index]][1]))
        legend_text = tree_axis.get_legend().get_texts()[0].get_text()
        self.assertIn("S = 0.420", legend_text)
        self.assertIn("p < 0.001", legend_text)

    def test_state_space_rdm_draws_the_example_l2_cohort_mean(self):
        rules = ["fdgo", "delaydm1", "delaygo", "dmcgo"]
        structure = self._state_space_structure(rules, np.array([[0., 0.], [10., 0.],
                                                                 [1., 0.], [10., 1.]]))
        cohort = {"n_runs": 2, "anames": ["run_a", "run_b"], "all_rules": rules,
                  "representations": structure["representations"]}
        for entry in cohort["representations"].values():
            entry["consistency"] = {"median": 0.91, "min": 0.76, "max": 0.95, "n_runs": 2}
        combined = {"runs": {}, "cohorts": {1e-2: {"n_runs": 0}, 1e-4: cohort}}
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_pkl_or_skip", return_value=combined), \
                patch.object(paper_plot, "_load_best_state_space_trial") as selector, \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_state_space_rdm()
        selector.assert_not_called()                    # the cohort mean needs no example run
        self.assertEqual([Path(call.args[1]).name for call in save_fig.call_args_list],
                         ["multitask_state_space_rdm_eff_mod.png",
                          "multitask_state_space_rdm_colorbar_eff_mod.png"])
        for call in save_fig.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        figure = save_fig.call_args_list[0].args[0]
        tree_axis, heat_axis = figure.axes
        order = list(cohort["representations"]["eff_mod"]["leaf_order"])
        ordered = cohort["representations"]["eff_mod"]["distances"][np.ix_(order, order)]
        lower = np.tril_indices(len(order))
        np.testing.assert_allclose(np.asarray(heat_axis.images[0].get_array())[lower],
                                   ordered[lower])
        legend_text = tree_axis.get_legend().get_texts()[0].get_text()
        self.assertIn("mean of 2 runs", legend_text)
        self.assertIn("10^{-4}", legend_text)
        self.assertIn("S = 0.420", legend_text)
        self.assertIn("median 0.91 (0.76 to 0.95)", legend_text)
        colorbar_axis = save_fig.call_args_list[1].args[0].axes[0]
        self.assertEqual(colorbar_axis.get_xlabel(), "Distance / mean distance")

    def test_state_space_rdm_skips_results_without_cohort_means(self):
        for combined in ({"runs": {}}, {"runs": {}, "cohorts": {1e-3: {"n_runs": 1}}}):
            with self.subTest(combined=combined), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_load_pkl_or_skip", return_value=combined), \
                    patch.object(paper_plot, "_save_fig") as save_fig, \
                    patch("builtins.print") as output:
                paper_plot.plot_state_space_rdm()
            save_fig.assert_not_called()
            self.assertIn("Skipped", output.call_args.args[0])

    def test_state_space_structure_figures_skip_without_the_cache(self):
        for figure in (paper_plot.plot_state_space_rdm_example,):
            with self.subTest(figure=figure.__name__), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_load_best_state_space_trial",
                                 return_value=("selected", 0.5, {})), \
                    patch.object(paper_plot, "_load_state_space_structure", return_value=None), \
                    patch.object(paper_plot, "_save_fig") as save_fig, \
                    patch("builtins.print") as output:
                figure()
            save_fig.assert_not_called()
            self.assertIn("state_space_structure.py", output.call_args.args[0])

    def test_state_space_category_test_groups_runs_by_l2_and_counts_significance(self):
        runs = {}
        for index, strength in enumerate(paper_plot.STATE_SPACE_L2_STRENGTHS):
            for seed in range(2):
                significant = seed == 0 or index == 1
                runs[f"everything_seed{seed}_L2{index}"] = {
                    "l2_info": strength,
                    "representations": {
                        key: {"separation": {"score": 0.1 * (index + 1) + 0.05 * seed,
                                             "p_value": 0.01 if significant else 0.5,
                                             "null_quantile_95": 0.08}}
                        for key in ("hidden", "mod", "eff_mod")},
                }
        runs["other_l2"] = {"l2_info": 3e-3, "representations": {}}
        combined = {"runs": runs, "n_permutations": 10, "permutation_seed": 0}
        grouped = paper_plot._state_space_runs_by_l2(runs)
        self.assertEqual(tuple(grouped), paper_plot.STATE_SPACE_L2_STRENGTHS)
        self.assertEqual(sum(len(entries) for entries in grouped.values()), 8)
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_pkl_or_skip", return_value=combined), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            paper_plot.plot_state_space_category_test()
        save_fig.assert_called_once()
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(Path(save_fig.call_args.args[1]).name,
                         "multitask_state_space_category_test.png")
        self.assertEqual(len(figure.axes), 2)
        for axis in figure.axes:
            counts = [label.get_text().split("\n")[1] for label in axis.get_xticklabels()]
            self.assertEqual(counts, ["1/2", "2/2", "1/2", "1/2"])
            points = [collection for collection in axis.collections
                      if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
            self.assertEqual(len(points), 8)
            filled = sum(paper_plot.mpl.colors.to_hex(collection.get_facecolors()[0])
                         == paper_plot.mpl.colors.to_hex(paper_plot._SCATTER_POINT_COLOR)
                         for collection in points)
            self.assertEqual(filled, 5)

    @staticmethod
    def _separation_summary(score, p_value):
        return {"representations": {key: {"separation": {
            "score": score, "p_value": p_value, "null_quantile_95": 0.08}}
            for key in ("hidden", "mod", "eff_mod")}}

    def _structure_with_untrained(self):
        runs = {}
        for index, strength in enumerate(paper_plot.STATE_SPACE_L2_STRENGTHS):
            for seed in range(2):
                runs[f"everything_seed{seed}_L2{index}"] = {
                    "l2_info": strength, **self._separation_summary(0.1 * (index + 1), 0.01)}
        untrained = {
            f"run_init{seed}": {"config_source": "run", "init_seed": seed,
                                **self._separation_summary(0.002 * seed, 0.6)}
            for seed in range(2)}
        # An untrained network of another run's config is not drawn.
        untrained["other_init0"] = {"config_source": "other", "init_seed": 0,
                                    **self._separation_summary(0.5, 0.01)}
        return {"runs": runs, "untrained": untrained}

    def _draw_category_vs_untrained(self, combined):
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_pkl_or_skip", return_value=combined), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             return_value=("run", 0.1, {})), \
                patch.object(paper_plot, "_save_fig") as save_fig, \
                patch("builtins.print") as output:
            paper_plot.plot_state_space_category_vs_untrained()
        return save_fig, output

    def test_state_space_category_vs_untrained_puts_untrained_left_of_the_cohorts(self):
        save_fig, _ = self._draw_category_vs_untrained(self._structure_with_untrained())
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(Path(save_fig.call_args.args[1]).name,
                         "multitask_state_space_category_vs_untrained.png")
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["Hidden state", "Eff. modulation"])
        for axis in figure.axes:
            labels = [label.get_text() for label in axis.get_xticklabels()]
            self.assertEqual(labels[0], "Untrained\n0/2")
            self.assertEqual([label.split("\n")[1] for label in labels[1:]], ["2/2"] * 4)
            np.testing.assert_allclose(axis.get_xticks(), [0.0, 1.5, 2.5, 3.5, 4.5])
            # Two untrained networks of the example config and eight trained runs.
            points = [collection for collection in axis.collections
                      if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
            self.assertEqual(len(points), 10)
            dividers = [line for line in axis.lines
                        if np.allclose(line.get_xdata(), [0.75, 0.75])]
            self.assertEqual(len(dividers), 1)
        self.assertIn("untrained 0/2", save_fig.call_args.kwargs["extra"])
        self.assertIn("L2 1e-02 2/2", save_fig.call_args.kwargs["extra"])

    def test_state_space_category_vs_untrained_skips_without_untrained_networks(self):
        for untrained in ({}, {"other_init0": {"config_source": "other", "init_seed": 0,
                                               **self._separation_summary(0.5, 0.01)}}):
            combined = self._structure_with_untrained()
            combined["untrained"] = untrained
            with self.subTest(untrained=list(untrained)):
                save_fig, output = self._draw_category_vs_untrained(combined)
                save_fig.assert_not_called()
                message = output.call_args.args[0]
                self.assertIn("--untrained --checkpoint multiple_tasks/savednet_run.pt", message)
                self.assertIn("state_space_structure.py", message)

    def test_removed_cross_seed_summary_is_not_available(self):
        self.assertFalse(hasattr(paper_plot, "plot_cross_seed_summary"))
        self.assertNotIn("cross_seed_summary", paper_plot.ALL_FIGURES)
        for group in paper_plot.FIGURES_BY_MODE.values():
            self.assertNotIn("cross_seed_summary", group)

    def test_two_in_multiple_renders_both_sibling_families(self):
        loaded = {"representations": {}}
        # No rule-cue sweep: a cached sweep on disk would add an overlaid copy
        # per representation (covered by test_sibling_cue_interp).
        with patch.object(paper_plot, "_load_pkl_or_skip", return_value=loaded) as load, \
                patch.object(paper_plot, "_cue_interp_endpoint_overlay",
                             return_value=None), \
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
        # An empty run directory: the endpoint figure must not pick up a
        # rule-cue sweep cached on disk (test_sibling_cue_interp covers that).
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(paper_plot, "TWO_IN_MULTIPLES_DIR", Path(directory)), \
                patch.object(paper_plot, "_load_pkl_or_skip", return_value=data), \
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

    def test_pretraining_lambda_switches_every_artifact_pattern(self):
        original_bound = paper_plot.PRETRAINING_BOUND
        original_lambda = paper_plot.PRETRAINING_LAMBDA
        try:
            self.assertEqual(paper_plot.PRETRAINING_LAMBDA, "0.99")
            self.assertEqual(
                paper_plot.PRETRAINING_ADDON_NAME,
                "+hidden200+L21e3+batch128+angle",
            )

            paper_plot._set_pretraining_lambda("0.9")
            self.assertEqual(paper_plot.PRETRAINING_LAMBDA, "0.9")
            self.assertEqual(
                paper_plot.PRETRAINING_ADDON_NAME,
                "+hidden200+L21e3lam0.9+batch128+angle",
            )
            with patch.object(Path, "glob", return_value=[]) as glob:
                paper_plot._pretraining_result_pkls()
                glob.assert_called_once_with(
                    "*_dmpn_seed*_+hidden200+L21e3lam0.9+batch128+angle_result.pkl")
            with patch.object(Path, "glob", return_value=[]) as glob:
                paper_plot._pretraining_combined_pkls("transfer_speed")
                glob.assert_called_once_with(
                    "*_dmpn_+hidden200+L21e3lam0.9+batch128+angle_transfer_speed.pkl")
            self.assertEqual(
                paper_plot._pretraining_ruleset_from_result_name(
                    "fdanti_delaygo_dmpn_seed3_+hidden200+L21e3lam0.9+batch128+angle_result.pkl"),
                "fdanti_delaygo")
            self.assertIsNone(paper_plot._pretraining_ruleset_from_result_name(
                "fdanti_delaygo_dmpn_seed11_+hidden200+L21e3+batch128+angle_result.pkl"))

            # The bound and lambda tags compose in the training script's order.
            paper_plot._set_pretraining_bound("mb2")
            self.assertEqual(
                paper_plot.PRETRAINING_ADDON_NAME,
                "+hidden200+L21e3mb2lam0.9+batch128+angle",
            )
            paper_plot._set_pretraining_lambda("0.99")
            self.assertEqual(
                paper_plot.PRETRAINING_ADDON_NAME,
                "+hidden200+L21e3mb2+batch128+angle",
            )
            with self.assertRaises(ValueError):
                paper_plot._set_pretraining_lambda("0.95")
            self.assertEqual(paper_plot.PRETRAINING_LAMBDA, "0.99")
        finally:
            paper_plot._set_pretraining_bound(original_bound)
            paper_plot._set_pretraining_lambda(original_lambda)

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
