"""Scale-free tuning-vs-lesion statistics exported by lesion_plot.py."""

import unittest

import _bootstrap  # noqa: F401
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import lesion_plot


def synthetic_lesion(n_tasks=8, n_clusters=6, n_repeats=10, seed=3):
    """Controls near 0.9; clusters 0, 1, 2, 4 carry distinct strong effects,
    cluster 3 has exactly no effect, cluster 5 is the trailing 'unresponsive' one."""
    rng = np.random.default_rng(seed)
    control_raw = 0.9 + rng.normal(0.0, 0.02, size=(n_tasks, n_clusters, n_repeats))
    control_mean = control_raw.mean(axis=2)
    patterns = np.zeros((n_tasks, n_clusters))
    for cluster in (0, 1, 2, 4):
        patterns[:, cluster] = 0.2 + 0.3 * rng.random(n_tasks)
    patterns[:, 5] = 0.005 * rng.random(n_tasks)
    lesion_acc = control_mean - patterns
    cluster_means = rng.random((5, n_clusters))
    labels = [f"mod_c{index + 1}" for index in range(n_clusters)]
    return cluster_means, lesion_acc, control_raw, labels


class ZScoreTests(unittest.TestCase):
    def test_effect_and_floored_sd(self):
        control_raw = np.stack([np.full((2, 3), 0.8), np.full((2, 3), 0.9)], axis=2)
        lesion_acc = np.array([[0.5, 0.85, 0.2], [0.85, 0.85, 0.85]])
        scored = lesion_plot._z_scored_lesion_effects(lesion_acc, control_raw, sd_floor=0.1)
        np.testing.assert_allclose(scored["effect"], 0.85 - lesion_acc, atol=1e-12)
        # Two repeats 0.1 apart have ddof=1 sd 0.0707, below the 0.1 floor.
        np.testing.assert_allclose(scored["sd"], 0.1)
        np.testing.assert_allclose(scored["z"], (0.85 - lesion_acc) / 0.1, atol=1e-10)
        self.assertEqual(scored["n_repeats"], 2)

    def test_rejects_shape_mismatch_single_repeat_and_bad_floor(self):
        control_raw = np.full((2, 3, 4), 0.8)
        with self.assertRaises(ValueError):
            lesion_plot._z_scored_lesion_effects(np.zeros((2, 2)), control_raw)
        with self.assertRaises(ValueError):
            lesion_plot._z_scored_lesion_effects(np.zeros((2, 3)), control_raw[:, :, :1])
        with self.assertRaises(ValueError):
            lesion_plot._z_scored_lesion_effects(np.zeros((2, 3)), control_raw, sd_floor=0.0)


class SignificanceTests(unittest.TestCase):
    def test_strong_clusters_pass_and_null_cluster_fails(self):
        _, lesion_acc, control_raw, _ = synthetic_lesion()
        scored = lesion_plot._z_scored_lesion_effects(lesion_acc, control_raw)
        result = lesion_plot._cluster_effect_significance(control_raw, scored["sd"], scored["z"])
        self.assertEqual(result["null"].shape, (6 * 10,))
        self.assertEqual(result["statistic"], "sum_abs_z")
        np.testing.assert_array_equal(result["significant"], [True, True, True, False, True, False])
        self.assertEqual(result["per_cluster"][3], 0.0)
        self.assertGreater(result["threshold"], 0.0)

    def test_null_is_control_only_and_uses_leave_one_out_means(self):
        control_raw = np.arange(24, dtype=float).reshape(2, 3, 4) / 100
        sd = np.full((2, 3), 0.01)
        z = np.zeros((2, 3))
        result = lesion_plot._cluster_effect_significance(control_raw, sd, z, quantile=0.5)
        others = (control_raw.sum(axis=2, keepdims=True) - control_raw) / 3
        expected = np.abs((others - control_raw) / sd[:, :, None]).sum(axis=0).ravel()
        np.testing.assert_allclose(np.sort(result["null"]), np.sort(expected))
        self.assertEqual(result["threshold"], float(np.quantile(expected, 0.5)))
        with self.assertRaises(ValueError):
            lesion_plot._cluster_effect_significance(control_raw, sd, z, quantile=1.0)


class MantelTests(unittest.TestCase):
    def matrices(self, seed=0):
        rng = np.random.default_rng(seed)
        profiles = rng.random((5, 6))
        matrix = np.corrcoef(profiles.T)
        return matrix, 1.0 - matrix

    def test_identical_structure_gives_rho_one_and_small_p(self):
        x_matrix, y_matrix = self.matrices()
        result = lesion_plot._mantel_spearman(x_matrix, -y_matrix, n_perm=400, seed=1)
        self.assertAlmostEqual(result["rho"], 1.0)
        self.assertLess(result["p_perm"], 0.05)
        self.assertEqual((result["n_clusters"], result["n_pairs"]), (6, 15))
        self.assertEqual(result["side"], "two-sided")
        self.assertEqual(result["permutation_unit"], "cluster_label")
        self.assertEqual(result["null_rho"].shape, (400,))

    def test_rho_matches_spearman_on_lower_triangle_and_is_relabeling_invariant(self):
        x_matrix, y_matrix = self.matrices(seed=4)
        tri = np.tril_indices(6, k=-1)
        result = lesion_plot._mantel_spearman(x_matrix, y_matrix, n_perm=5)
        self.assertAlmostEqual(result["rho"], spearmanr(x_matrix[tri], y_matrix[tri]).statistic)
        perm = np.random.default_rng(2).permutation(6)
        relabeled = lesion_plot._mantel_spearman(
            x_matrix[np.ix_(perm, perm)], y_matrix[np.ix_(perm, perm)], n_perm=5)
        self.assertAlmostEqual(relabeled["rho"], result["rho"])

    def test_constant_matrix_yields_nan_and_bad_inputs_raise(self):
        x_matrix, y_matrix = self.matrices()
        result = lesion_plot._mantel_spearman(np.ones((6, 6)), y_matrix, n_perm=3)
        self.assertTrue(np.isnan(result["rho"]) and np.isnan(result["p_perm"]))
        with self.assertRaises(ValueError):
            lesion_plot._mantel_spearman(x_matrix[:2, :2], y_matrix[:2, :2], n_perm=3)
        with self.assertRaises(ValueError):
            lesion_plot._mantel_spearman(x_matrix, y_matrix[:5, :5], n_perm=3)
        with self.assertRaises(ValueError):
            lesion_plot._mantel_spearman(x_matrix, y_matrix, n_perm=0)


class SummaryTests(unittest.TestCase):
    def test_summary_excludes_last_and_null_clusters_and_matches_definitions(self):
        cluster_means, lesion_acc, control_raw, labels = synthetic_lesion()
        summary = lesion_plot._tuning_vs_lesion_summary(
            cluster_means, lesion_acc, control_raw, labels,
            exclude_last_cluster=True, n_perm=50, seed=0)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["x_definition"], lesion_plot.CLUSTER_CORR_X_DEFINITION)
        self.assertEqual(summary["y_definition"], lesion_plot.CLUSTER_CORR_Y_DEFINITION)
        self.assertTrue(summary["exclude_last_cluster"])
        self.assertEqual(summary["included_clusters"], ["mod_c1", "mod_c2", "mod_c3", "mod_c5"])
        self.assertEqual(summary["excluded_clusters"],
                         {"last": ["mod_c6"], "not_significant": ["mod_c4"], "degenerate": []})

        keep = [0, 1, 2, 4]
        scored = lesion_plot._z_scored_lesion_effects(lesion_acc, control_raw)
        tri = np.tril_indices(4, k=-1)
        np.testing.assert_allclose(summary["tuning_corr"],
                                   np.corrcoef(cluster_means[:, keep].T)[tri])
        np.testing.assert_allclose(summary["lesion_profile_dissim"],
                                   (1.0 - np.corrcoef(scored["z"][:, keep].T))[tri])
        self.assertEqual(summary["association"]["n_clusters"], 4)
        self.assertEqual(summary["association"]["n_pairs"], 6)
        slope, intercept = np.polyfit(summary["tuning_corr"], summary["lesion_profile_dissim"], 1)
        self.assertEqual(summary["trend_line"]["method"], "ols")
        self.assertAlmostEqual(summary["trend_line"]["slope"], slope)
        self.assertAlmostEqual(summary["trend_line"]["intercept"], intercept)
        self.assertIsNone(lesion_plot._descriptive_trend_line([0.3, 0.3, 0.3], [1., 2., 3.]))
        self.assertEqual(summary["n_repeats"], 10)
        self.assertEqual(summary["sd_floor"], lesion_plot.CLUSTER_CORR_SD_FLOOR)

        # The L1 supplement reproduces the former figure over the first five clusters.
        supplement = summary["l1"]
        base = slice(0, 5)
        tri5 = np.tril_indices(5, k=-1)
        effect = scored["effect"][:, base]
        self.assertEqual(supplement["cluster_labels"], labels[:5])
        self.assertEqual(supplement["y_definition"], lesion_plot.CLUSTER_CORR_L1_DEFINITION)
        np.testing.assert_allclose(supplement["tuning_cos_sim"],
                                   cosine_similarity(cluster_means[:, base].T)[tri5])
        np.testing.assert_allclose(supplement["lesion_l1_dist"],
                                   squareform(pdist(effect.T, metric="cityblock"))[tri5])
        magnitude = np.abs(effect).sum(axis=0)
        np.testing.assert_allclose(supplement["effect_magnitude_sum"],
                                   (magnitude[:, None] + magnitude[None, :])[tri5])
        self.assertEqual(supplement["association_tuning"]["n_clusters"], 5)
        self.assertEqual(supplement["association_magnitude"]["n_pairs"], 10)

    def test_without_exclusion_the_last_cluster_is_judged_by_significance(self):
        cluster_means, lesion_acc, control_raw, labels = synthetic_lesion()
        summary = lesion_plot._tuning_vs_lesion_summary(
            cluster_means, lesion_acc, control_raw, labels, n_perm=20)
        self.assertEqual(summary["excluded_clusters"]["last"], [])
        self.assertEqual(summary["excluded_clusters"]["not_significant"], ["mod_c4", "mod_c6"])
        self.assertEqual(summary["l1"]["cluster_labels"], labels)

    def test_too_few_significant_clusters_and_mismatched_inputs_raise(self):
        cluster_means, lesion_acc, control_raw, labels = synthetic_lesion()
        no_effect = control_raw.mean(axis=2)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            lesion_plot._tuning_vs_lesion_summary(cluster_means, no_effect, control_raw,
                                                  labels, n_perm=5)
        with self.assertRaises(ValueError):
            lesion_plot._tuning_vs_lesion_summary(cluster_means[:, :5], lesion_acc,
                                                  control_raw, labels, n_perm=5)
        with self.assertRaises(ValueError):
            lesion_plot._tuning_vs_lesion_summary(cluster_means, lesion_acc, control_raw,
                                                  labels[:-1] + [labels[0]], n_perm=5)


if __name__ == "__main__":
    unittest.main()
