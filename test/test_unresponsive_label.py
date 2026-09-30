"""The unresponsive cluster is identified by an explicit label, not by position."""

import unittest

import _bootstrap  # noqa: F401
import numpy as np

import clustering
import lesion_plot


def _variance_matrix_with_silent_columns(seed=0):
    """(features, neurons) matrix: 3 tuned groups plus 4 silent neurons at
    positions that are NOT the last columns."""
    rng = np.random.default_rng(seed)
    groups = [rng.random(8) + 2 * k for k in range(3)]
    columns = []
    for group in groups:
        columns.extend([group + rng.normal(0, 0.05, 8) for _ in range(6)])
    silent_positions = [1, 5, 9, 12]
    for position in silent_positions:
        columns.insert(position, np.full(8, 1e-7))
    return np.stack(columns, axis=1), silent_positions


class UnresponsiveLabelHelperTests(unittest.TestCase):
    def test_label_is_read_from_mask_and_none_without_unresponsive_entries(self):
        labels = np.array([1, 4, 2, 4, 3, 1])
        mask = np.array([False, True, False, True, False, False])
        self.assertEqual(clustering.unresponsive_label(labels, mask), 4)
        self.assertIsNone(clustering.unresponsive_label(labels, np.zeros(6, bool)))

    def test_ambiguous_or_shared_labels_are_rejected(self):
        labels = np.array([1, 4, 2, 5])
        with self.assertRaises(ValueError):
            clustering.unresponsive_label(labels, np.array([False, True, False, True]))
        with self.assertRaises(ValueError):
            clustering.unresponsive_label(np.array([1, 4, 4, 2]),
                                          np.array([False, True, False, False]))
        with self.assertRaises(ValueError):
            clustering.unresponsive_label(labels, np.array([False, True]))

    def test_legacy_result_without_mask_uses_the_k_plus_one_convention(self):
        legacy = {"col_tol_labels": np.array([1, 2, 3, 2, 1]), "col_tol_k": 2}
        self.assertEqual(clustering.unresponsive_label_from_result(legacy, "col"), 3)
        legacy_none = {"col_tol_labels": np.array([1, 2, 2, 1]), "col_tol_k": 2}
        self.assertIsNone(clustering.unresponsive_label_from_result(legacy_none, "col"))


class ClusteringResultTests(unittest.TestCase):
    def test_repeat_clustering_records_mask_and_label_of_non_last_silent_columns(self):
        V, silent = _variance_matrix_with_silent_columns()
        result = clustering.cluster_variance_matrix_repeat(
            V, k_min=2, k_max=5, n_repeats=1, silhouette_tol=0.05, tol_k_select="min")
        mask = result["col_unresponsive_mask"]
        np.testing.assert_array_equal(np.where(mask)[0], silent)
        label = result["col_unresponsive_label"]
        self.assertEqual(label, result["col_tol_k"] + 1)
        np.testing.assert_array_equal(np.asarray(result["col_tol_labels"])[mask], label)
        self.assertEqual(clustering.unresponsive_label_from_result(result, "col"), label)
        self.assertIsNone(result["row_unresponsive_label"])
        self.assertFalse(result["row_unresponsive_mask"].any())

    def test_fixed_k_recut_returns_the_label_and_keeps_silent_columns_together(self):
        V, silent = _variance_matrix_with_silent_columns()
        result = clustering.cluster_variance_matrix_repeat(
            V, k_min=2, k_max=5, n_repeats=1, silhouette_tol=0.05, tol_k_select="min")
        entry = {"result": result}
        clusters, label = clustering.fixed_k_col_clusters(
            entry, 4, return_unresponsive_label=True)
        self.assertEqual(label, 5)
        np.testing.assert_array_equal(clusters[label], silent)
        plain = clustering.fixed_k_col_clusters(entry, 4)
        self.assertEqual(sorted(plain), sorted(clusters))
        for key in clusters:
            np.testing.assert_array_equal(plain[key], clusters[key])

    def test_fixed_k_recut_reports_no_label_when_nothing_is_silent(self):
        V, _ = _variance_matrix_with_silent_columns()
        V = V[:, [c for c in range(V.shape[1]) if V[:, c].max() > 1e-3]]
        result = clustering.cluster_variance_matrix_repeat(
            V, k_min=2, k_max=5, n_repeats=1, silhouette_tol=0.05, tol_k_select="min")
        clusters, label = clustering.fixed_k_col_clusters(
            {"result": result}, 3, return_unresponsive_label=True)
        self.assertIsNone(label)
        self.assertEqual(sorted(clusters), [1, 2, 3])

    def test_group_clustering_expands_the_mask_to_synapses_and_labels_every_k(self):
        V, silent = _variance_matrix_with_silent_columns()
        groups = [[c] for c in range(V.shape[1])]
        result = clustering.cluster_variance_matrix_forgroup(
            V, k_min=2, k_max=5, col_groups_all_lst=[groups],
            silhouette_tol=0.05, tol_k_select="min")
        np.testing.assert_array_equal(np.where(result["col_unresponsive_mask"])[0], silent)
        for k, labels in result["col_labels_by_k"].items():
            self.assertEqual(result["col_unresponsive_label_by_k"][k], k + 1)
            np.testing.assert_array_equal(np.asarray(labels)[silent], k + 1)
        self.assertEqual(result["col_unresponsive_label"], result["col_k"] + 1)


class LesionPlotLookupTests(unittest.TestCase):
    def test_explicit_fields_win_over_the_last_cluster_rule(self):
        ga = {"n_in": 5, "n_hid": 6, "unresponsive_input_index": 1,
              "unresponsive_hidden_index": None}
        self.assertEqual(lesion_plot._unresponsive_grid_index(ga, "input", "unnorm"), 1)
        self.assertIsNone(lesion_plot._unresponsive_grid_index(ga, "hidden", "unnorm"))
        cdata = {"pre_n": 5, "post_n": 6, "unresponsive_pre_label": 2,
                 "unresponsive_post_label": None}
        self.assertEqual(lesion_plot._combined_unresponsive_index(cdata, "pre", "unnorm"), 1)
        self.assertIsNone(lesion_plot._combined_unresponsive_index(cdata, "post", "unnorm"))
        entry = {"all_comb_names_lesion": ["pre_nolesion", "pre_c1", "pre_c2", "post_nolesion",
                                           "post_c1", "post_c2", "post_c3"],
                 "unresponsive_conditions": ["post_c2"]}
        self.assertEqual(lesion_plot._unresponsive_condition_names(entry, legacy_last=True),
                         {"post_c2"})
        mod = {"all_comb_names_mod": ["mod_nolesion", "mod_c1", "mod_c2", "mod_c3"],
               "unresponsive_label": 2}
        self.assertEqual(lesion_plot._mod_unresponsive_label(mod, legacy_last=True), 2)
        mod_none = {"all_comb_names_mod": ["mod_nolesion", "mod_c1"], "unresponsive_label": None}
        self.assertIsNone(lesion_plot._mod_unresponsive_label(mod_none, legacy_last=True))

    def test_legacy_caches_fall_back_to_the_last_cluster_only_for_unnormalized(self):
        ga = {"n_in": 5, "n_hid": 6}
        self.assertEqual(lesion_plot._unresponsive_grid_index(ga, "input", "unnorm"), 4)
        self.assertEqual(lesion_plot._unresponsive_grid_index(ga, "hidden", "unnorm"), 5)
        self.assertIsNone(lesion_plot._unresponsive_grid_index(ga, "input", "norm"))
        cdata = {"pre_n": 5, "post_n": 6}
        self.assertEqual(lesion_plot._combined_unresponsive_index(cdata, "post", "unnorm"), 5)
        self.assertIsNone(lesion_plot._combined_unresponsive_index(cdata, "post", "norm"))
        entry = {"all_comb_names_lesion": ["pre_nolesion", "pre_c1", "pre_c2", "post_nolesion",
                                           "post_c1", "post_c2", "post_c3"]}
        self.assertEqual(lesion_plot._unresponsive_condition_names(entry, legacy_last=True),
                         {"pre_c2", "post_c3"})
        self.assertEqual(lesion_plot._unresponsive_condition_names(entry), set())
        mod = {"all_comb_names_mod": ["mod_nolesion", "mod_c1", "mod_c3"]}
        self.assertEqual(lesion_plot._mod_unresponsive_label(mod, legacy_last=True), 3)
        self.assertIsNone(lesion_plot._mod_unresponsive_label(mod))
        np.testing.assert_array_equal(lesion_plot._indices_without(4, 1), [0, 2, 3])
        np.testing.assert_array_equal(lesion_plot._indices_without(3, None), [0, 1, 2])


class TuningSummaryExclusionTests(unittest.TestCase):
    @staticmethod
    def _synthetic(unresponsive_position):
        rng = np.random.default_rng(3)
        n_tasks, n_clusters, n_rep = 6, 6, 10
        cluster_means = rng.random((8, n_clusters))
        control_raw = 0.9 + 0.02 * rng.standard_normal((n_tasks, n_clusters, n_rep))
        lesion_acc = control_raw.mean(axis=2) - rng.random((n_tasks, n_clusters)) * 0.4
        # the unresponsive cluster has no effect and a flat tuning profile
        lesion_acc[:, unresponsive_position] = control_raw[:, unresponsive_position, :].mean(axis=1)
        cluster_means[:, unresponsive_position] = 0.0
        labels = [f"mod_c{k}" for k in range(1, n_clusters + 1)]
        return cluster_means, lesion_acc, control_raw, labels

    def test_explicit_label_excludes_a_non_last_cluster(self):
        cluster_means, lesion_acc, control_raw, labels = self._synthetic(2)
        summary = lesion_plot._tuning_vs_lesion_summary(
            cluster_means, lesion_acc, control_raw, labels,
            unresponsive_labels={"mod_c3"}, n_perm=20)
        self.assertEqual(summary["excluded_clusters"]["last"], ["mod_c3"])
        self.assertNotIn("mod_c3", summary["l1"]["cluster_labels"])
        self.assertEqual(len(summary["l1"]["cluster_labels"]), 5)
        self.assertTrue(summary["exclude_last_cluster"])
        self.assertEqual(summary["unresponsive_source"], "explicit_label")

    def test_explicit_empty_set_marks_the_policy_without_dropping_anything(self):
        cluster_means, lesion_acc, control_raw, labels = self._synthetic(5)
        summary = lesion_plot._tuning_vs_lesion_summary(
            cluster_means, lesion_acc, control_raw, labels,
            unresponsive_labels=set(), n_perm=20)
        self.assertEqual(summary["excluded_clusters"]["last"], [])
        self.assertEqual(summary["l1"]["cluster_labels"], labels)
        self.assertTrue(summary["exclude_last_cluster"])

    def test_explicit_last_label_matches_the_legacy_positional_rule(self):
        cluster_means, lesion_acc, control_raw, labels = self._synthetic(5)
        explicit = lesion_plot._tuning_vs_lesion_summary(
            cluster_means, lesion_acc, control_raw, labels,
            unresponsive_labels={"mod_c6"}, n_perm=20)
        legacy = lesion_plot._tuning_vs_lesion_summary(
            cluster_means, lesion_acc, control_raw, labels,
            exclude_last_cluster=True, n_perm=20)
        self.assertEqual(explicit["included_clusters"], legacy["included_clusters"])
        np.testing.assert_allclose(explicit["tuning_corr"], legacy["tuning_corr"])
        self.assertEqual(legacy["unresponsive_source"], "last_cluster")
        self.assertIsNone(lesion_plot._tuning_vs_lesion_summary(
            cluster_means, lesion_acc, control_raw, labels, n_perm=20)["unresponsive_source"])


if __name__ == "__main__":
    unittest.main()
