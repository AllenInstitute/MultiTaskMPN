"""Checks for the public paper-figure mode grouping."""

import inspect
import numpy as np
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import paper_plot


class PaperPlotModeTests(unittest.TestCase):
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
    def _state_space_pca(points):
        return {
            "all_rules": ["fdgo", "delaygo", "fdanti", "delayanti"],
            "pca_results": {"eff_mod": {
                "X_2d": np.asarray(points, dtype=float),
                "ctx_rule_labels": np.repeat(np.arange(4), 2),
            }},
        }

    def test_state_space_trial_uses_best_color_clustering_within_l2_1e3(self):
        good = self._state_space_pca([
            [-5.1, 0], [-4.9, 0], [-5.0, 0.1], [-5.0, -0.1],
            [4.9, 0], [5.1, 0], [5.0, 0.1], [5.0, -0.1],
        ])
        mixed = self._state_space_pca([
            [-5.1, 0], [-4.9, 0], [4.9, 0], [5.1, 0],
            [-5.0, 0.1], [-5.0, -0.1], [5.0, 0.1], [5.0, -0.1],
        ])
        self.assertGreater(
            paper_plot._eff_mod_color_clustering_score(good),
            paper_plot._eff_mod_color_clustering_score(mixed),
        )

        result_dict = {
            "mixed": {"l2_info": 1e-3},
            "best1": {"l2_info": 1e-3},
            "best2": {"l2_info": 1e-3},
            "better_but_other_l2": {"l2_info": 1e-2},
        }
        pca_data = {
            "mixed": mixed,
            "best1": good,
            "best2": good,
            "better_but_other_l2": good,
        }
        # Equal scores use the alphabetically first aname, independent of dict
        # insertion order, so regeneration cannot silently switch examples.
        selected = paper_plot._best_state_space_trial(
            result_dict, pca_loader=pca_data.get)
        self.assertEqual(selected[0], "best1")
        self.assertAlmostEqual(
            selected[1], paper_plot._eff_mod_color_clustering_score(good))

    def test_state_space_trial_returns_none_without_valid_color_clustering(self):
        self.assertIsNone(paper_plot._best_state_space_trial({
            "missing": {},
            "wrong_l2": {"l2_info": 1e-2},
            "missing_pca": {"l2_info": 1e-3},
        }, pca_loader=lambda _: None))

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
                f"dmcgo_delay_pc_projections_{paper_plot.DELAYDM_ANAME}.pkl",
                f"delaydm1_long_delay_endpoint_pc_projections_"
                f"{paper_plot.DELAYDM_ANAME}.pkl",
                f"dmcgo_long_delay_endpoint_pc_projections_"
                f"{paper_plot.DELAYDM_ANAME}.pkl",
            ],
        )
        self.assertEqual(
            [call.args[1] for call in endpoint_render.call_args_list],
            ["delaydm1", "delaydm1", "dmcgo", "dmcgo"],
        )
        self.assertTrue(all(
            call.args[0].parent
            == paper_plot.TWO_IN_MULTIPLES_DIR / paper_plot.DELAYDM_ANAME
            for call in load.call_args_list
        ))
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
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(len(figure.axes), 2)
        self.assertTrue(figure.axes[0].get_xlabel().startswith("Joint Delay PC1"))
        self.assertTrue(figure.axes[0].get_ylabel().startswith("Joint Delay PC2"))
        self.assertTrue(figure.axes[1].get_xlabel().startswith("Joint Delay PC3"))
        self.assertTrue(figure.axes[1].get_ylabel().startswith("Joint Delay PC4"))
        self.assertIn("Default PC1-PC2", figure.axes[0].get_title())
        self.assertIn("Best task-specific projection", figure.axes[1].get_title())
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

    def test_dmcgo_emodulation_zooms_to_endpoint_limits(self):
        group_labels = np.tile(np.repeat([0, 1], 4), 2)
        points = np.zeros((group_labels.size, 6), dtype=float)
        points[:, 3] = np.where(group_labels == 0, -4.0, 4.0)
        points[:, 3] += np.tile([-0.05, 0.05], group_labels.size // 2)
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

        with patch.object(paper_plot, "_save_fig") as save_fig:
            rendered = paper_plot._plot_multitask_long_delay_endpoint_representation(
                data, "dmcgo", ("dmcgo", "dmcnogo"),
                "e_modulation", "emodulation")

        self.assertTrue(rendered)
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertLess(figure.axes[1].get_ylim()[1], 10.0)

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
        self.assertLess(figure.axes[1].get_ylim()[1], 10.0)

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


if __name__ == "__main__":
    unittest.main()
