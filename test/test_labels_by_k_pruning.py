"""Saved modulation clustering results keep per-k labels only where they are read."""

import unittest

import _bootstrap  # noqa: F401
import numpy as np

import clustering


def _grouped_result():
    rng = np.random.default_rng(0)
    groups = [rng.random(6) + 3 * k for k in range(4)]
    V = np.stack([g + rng.normal(0, 0.05, 6) for g in groups for _ in range(5)], axis=1)
    return clustering.cluster_variance_matrix_forgroup(
        V, k_min=2, k_max=8, col_groups_all_lst=[[[c] for c in range(V.shape[1])]],
        silhouette_tol=0.05, tol_k_select="min")


class PruneLabelsByKTests(unittest.TestCase):
    def test_keeps_only_requested_k_and_records_them(self):
        result = _grouped_result()
        self.assertGreater(len(result["col_labels_by_k"]), 2)
        fixed_k, optimal_k = 5, int(result["col_k"])
        pruned = clustering.prune_labels_by_k(result, {fixed_k, optimal_k})
        self.assertEqual(sorted(pruned["col_labels_by_k"]), sorted({fixed_k, optimal_k}))
        self.assertEqual(pruned["col_labels_by_k_kept"], sorted({fixed_k, optimal_k}))
        self.assertTrue(pruned["col_labels_by_k_pruned"])
        for name in ("col_unresponsive_label_by_k", "col_cut_distance_by_k"):
            self.assertEqual(sorted(pruned[name]), sorted({fixed_k, optimal_k}))
        np.testing.assert_array_equal(pruned["col_labels_by_k"][fixed_k],
                                      result["col_labels_by_k"][fixed_k])
        np.testing.assert_array_equal(pruned["col_labels_by_k"][optimal_k], pruned["col_labels"])

    def test_original_result_and_row_side_are_untouched(self):
        result = _grouped_result()
        n_before = len(result["col_labels_by_k"])
        pruned = clustering.prune_labels_by_k(result, {int(result["col_k"])})
        self.assertEqual(len(result["col_labels_by_k"]), n_before)
        self.assertNotIn("col_labels_by_k_pruned", result)
        self.assertIs(pruned["row_labels_by_k"], result["row_labels_by_k"])
        self.assertIs(pruned["col_linkage"], result["col_linkage"])

    def test_missing_k_and_missing_tables_are_ignored(self):
        result = _grouped_result()
        pruned = clustering.prune_labels_by_k(result, {999})
        self.assertEqual(pruned["col_labels_by_k"], {})
        self.assertEqual(pruned["col_labels_by_k_kept"], [])
        minimal = clustering.prune_labels_by_k({"col_k": 3}, {3})
        self.assertEqual(minimal["col_labels_by_k_kept"], [])


if __name__ == "__main__":
    unittest.main()
