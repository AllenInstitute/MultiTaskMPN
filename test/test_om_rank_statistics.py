"""Rank association and descriptive medians for clustered OM scatter data."""

import ast
import inspect
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import spearmanr
import lesion_plot


class OmRankStatisticsTests(unittest.TestCase):
    def test_single_and_combined_export_paths_save_matching_rank_statistics(self):
        main_node = ast.parse(inspect.getsource(lesion_plot.main)).body[0]
        names = {"plot_overmembership_vs_lesion_diff", "_plot_om_vs_lesion_combined"}
        functions = [node for node in main_node.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual(len(functions), 2)
        namespace = dict(vars(lesion_plot))
        namespace["linregress"] = Mock(side_effect=AssertionError("No OM regression"))
        summary_function = lesion_plot._om_scatter_summary
        namespace["_om_scatter_summary"] = lambda *args: summary_function(*args, n_perm=7)
        namespace["OM_N_PERM"] = 7
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<om-export-test>", "exec"), namespace)
        base = "modulation_all_var_weighted_unnormalized"
        combined_effects = np.array([[[0.0, 0.2], [0.6, 0.9]],
                                     [[0.1, 0.3], [0.5, 0.8]]])
        results = {"combined_lesion_norm": {
            "combined_random_accs": combined_effects,
            "combined_accs": np.zeros((2, 2, 2)), "pre_n": 2, "post_n": 2,
        }, "mod_lesion": {}}
        for mode, factor in (("zero_W", 1.), ("freeze_M", 0.5)):
            results["mod_lesion"][f"{base}__{mode}"] = {
                "all_comb_names_mod": ["mod_nolesion", "mod_c3", "mod_c7"],
                "modrandomtask_accs": np.ones((2, 3)),
                "modtask_accs": 1 - factor * np.array([[0., 0.05, 0.7], [0., 0.1, 0.6]]),
            }
        cluster_info = {base: {"global_assignment_fixed_k20": {
            "om_stack": np.array([[[3., 2.], [0.1, 0.2]], [[0.1, 0.2], [2., 3.]]]),
            "all_choice_order": [3, 7], "n_in": 2, "n_hid": 2,
            "cluster_size_percent": [0.5, 0.5], "n_active_block": np.full((2, 2), 100.),
        }}}
        captured = []

        def capture(figure, *args, **kwargs):
            captured.append(figure)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(lesion_plot.plt.Figure, "savefig", autospec=True, side_effect=capture):
            namespace["plot_overmembership_vs_lesion_diff"](
                results, cluster_info, {}, "norm", base, "freeze_M", "run", directory)
            namespace["_plot_om_vs_lesion_combined"](
                results, cluster_info, {}, "norm", base, ("zero_W", "freeze_M"), "run", directory)
            prefix = "om_vs_lesion_diff_var-weighted-unnormalized"
            with (Path(directory) / f"{prefix}_freeze-M_norm_run.pkl").open("rb") as stream:
                single = pickle.load(stream)
            with (Path(directory) / f"{prefix}_combined_norm_run.pkl").open("rb") as stream:
                combined = pickle.load(stream)
        self.assertEqual(single["schema_version"], 2)
        self.assertEqual(combined["schema_version"], 2)
        paired = combined["mode_data"]["freeze_M"]
        for key in ("om_vals", "lesion_diffs"):
            np.testing.assert_allclose(single[key], paired[key])
        for key in ("rho", "p_perm", "null_rho"):
            np.testing.assert_allclose(single["association"][key], paired["association"][key])
        self.assertEqual(len(captured), 2)
        for figure in captured:
            for axis in figure.axes:
                self.assertEqual(axis.get_ylim()[0], 0)
                self.assertEqual(len(axis.lines), 1)
                self.assertEqual(axis.lines[0].get_label(), "Binned median")
        namespace["linregress"].assert_not_called()

    def inputs(self):
        profiles = np.array([[0., 0.], [0.4, 0.8], [0.9, 0.2]])
        footprints = [np.array([0., 0., 4.]), np.array([1., 3.]), np.array([2., 5.])]
        combined = [np.array([[0.8, 0.6], [0.5, 0.5], [0., 0.]]),
                    np.array([[0., 0.], [0.5, 0.7]]),
                    np.array([[0., 0.], [0.8, 0.3]])]
        return profiles, footprints, combined

    def test_permutations_recompute_spearman_for_whole_unequal_footprints(self):
        profiles, footprints, combined = self.inputs()

        def statistic(assignment):
            values = np.concatenate([footprints[index] for index in assignment])
            distances = np.concatenate([
                np.abs(combined[index] - profiles[cluster]).mean(axis=1)
                for cluster, index in enumerate(assignment)])
            return spearmanr(values, distances).statistic

        rng = np.random.default_rng(7)
        expected_null = np.array([statistic(rng.permutation(3)) for _ in range(23)])
        rho, probability, null = lesion_plot._om_scatter_perm_test(
            profiles, footprints, combined, n_perm=23, seed=7)
        self.assertAlmostEqual(rho, statistic(np.arange(3)))
        np.testing.assert_allclose(null, expected_null)
        self.assertAlmostEqual(probability, (1 + np.count_nonzero(expected_null <= rho)) / 24)
        self.assertLess(rho, 0)

    def test_quantile_medians_keep_zeros_ties_and_nonnegative_values(self):
        values = np.array([0., 0., 0., 1., 2., 3., 4., 5.])
        distances = np.array([0., 2., 4., 3., 0., 1., 2., 1.])
        record = lesion_plot._om_binned_medians(values, distances, n_bins=2)
        np.testing.assert_allclose(record["x"], [0., 3.5])
        np.testing.assert_allclose(record["y"], [2.5, 1.])
        np.testing.assert_array_equal(record["counts"], [4, 4])
        tied = lesion_plot._om_binned_medians(np.zeros(4), [0., 1., 2., 3.])
        np.testing.assert_allclose(tied["x"], [0.])
        np.testing.assert_allclose(tied["y"], [1.5])
        np.testing.assert_array_equal(tied["counts"], [4])

    def test_summary_saves_matching_statistic_null_and_medians_without_regression(self):
        record = lesion_plot._om_scatter_summary(*self.inputs(), n_perm=11, seed=3)
        association = record["association"]
        self.assertEqual(association["statistic"], "spearman")
        self.assertEqual(association["permutation_unit"], "modulation_cluster_footprint")
        self.assertEqual(association["side"], "less")
        self.assertEqual(association["n_clusters"], 3)
        self.assertEqual(association["n_perm"], 11)
        self.assertEqual(association["null_rho"].shape, (11,))
        self.assertAlmostEqual(association["rho"],
                               spearmanr(record["om_vals"], record["lesion_diffs"]).statistic)
        self.assertNotIn("regression", record)
        self.assertEqual(record["binned_medians"]["counts"].sum(), len(record["om_vals"]))

    def test_degenerate_correlation_is_not_reported_as_significant(self):
        profiles, footprints, combined = self.inputs()
        rho, probability, null = lesion_plot._om_scatter_perm_test(
            profiles, [np.zeros(len(values)) for values in footprints], combined,
            n_perm=5)
        self.assertTrue(np.isnan(rho))
        self.assertTrue(np.isnan(probability))
        self.assertTrue(np.isnan(null).all())
        _, probability, _ = lesion_plot._om_scatter_perm_test(
            profiles[:1], footprints[:1], combined[:1], n_perm=5)
        self.assertTrue(np.isnan(probability))


if __name__ == "__main__":
    unittest.main()