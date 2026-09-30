"""Task-specificity counts, dispersion null and compositional sharing statistics."""

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import lesion_plot


TASKS = ["fdgo", "fdanti", "reactgo", "reactanti", "delaygo", "delayanti",
         "delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm",
         "dmsgo", "dmsnogo", "dmcgo", "dmcnogo"]


class RelationTests(unittest.TestCase):
    def test_every_pair_has_exactly_one_relation_and_counts_match(self):
        labels = lesion_plot._pair_relation_labels(TASKS)
        self.assertEqual(len(labels), 15 * 14 // 2)
        counts = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        self.assertEqual(counts["response rule"], 5)
        self.assertEqual(counts["timing"], 6)
        self.assertEqual(counts["modality"], 2)
        self.assertEqual(counts["context cue"], 2)
        self.assertEqual(counts["integration family"], 6)
        self.assertEqual(counts["match/category family"], 4)
        self.assertEqual(counts["other"], 105 - 25)
        for relation, pairs in lesion_plot.TASK_PAIR_RELATIONS.items():
            for first, second in pairs:
                self.assertIn(first, TASKS)
                self.assertIn(second, TASKS)

    def test_duplicate_relation_listing_is_rejected(self):
        with patch.dict(lesion_plot.TASK_PAIR_RELATIONS,
                        {"extra": [("fdgo", "fdanti")]}):
            with self.assertRaisesRegex(ValueError, "two relations"):
                lesion_plot._pair_relation_labels(TASKS)


class CountTests(unittest.TestCase):
    def test_counts_and_jaccard(self):
        sig = np.array([[1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0]], dtype=bool)
        np.testing.assert_array_equal(lesion_plot._task_specificity_counts(sig), [2, 2, 1, 0])
        jaccard = lesion_plot._jaccard_matrix(sig)
        self.assertEqual(jaccard[0, 1], 1.0)
        self.assertEqual(jaccard[0, 2], 0.0)
        self.assertTrue(np.isnan(jaccard[3, 3]))   # both sets empty: undefined
        self.assertEqual(jaccard[0, 3], 0.0)      # one empty set: no overlap

    def test_dispersion_null_preserves_per_task_totals(self):
        rng = np.random.default_rng(0)
        sig = rng.random((6, 10)) < 0.3
        result = lesion_plot._count_dispersion_test(sig, n_perm=200, seed=1)
        self.assertEqual(result["null_var"].shape, (200,))
        self.assertTrue(0 < result["p_perm"] <= 1)
        # A perfectly modular matrix (each task hits one private cluster) has
        # zero variance, the least dispersed possible outcome: p is large.
        modular = np.eye(6, 10, dtype=bool)
        self.assertGreater(lesion_plot._count_dispersion_test(modular, n_perm=100)["p_perm"], 0.5)
        # A hub matrix (every task hits the same two clusters) is maximally
        # dispersed relative to the shuffle: p is small.
        hub = np.zeros((6, 10), dtype=bool)
        hub[:, :2] = True
        self.assertLess(lesion_plot._count_dispersion_test(hub, n_perm=400)["p_perm"], 0.05)


class SharingTests(unittest.TestCase):
    def test_related_pairs_sharing_is_detected(self):
        rng = np.random.default_rng(3)
        sig = rng.random((15, 12)) < 0.15
        # make every response-rule pair depend on the same clusters
        for first, second in lesion_plot.TASK_PAIR_RELATIONS["response rule"]:
            sig[TASKS.index(second)] = sig[TASKS.index(first)]
        result = lesion_plot._sharing_by_relation(sig, TASKS, n_perm=500, seed=0)
        rule = result["by_relation"]["response rule"]
        self.assertEqual(len(rule["pairs"]), 5)
        self.assertAlmostEqual(rule["mean"], 1.0)
        self.assertLess(rule["p_perm"], 0.02)
        self.assertGreater(result["by_relation"]["other"]["mean"], 0.0)
        self.assertEqual(result["relations"][-1], "other")
        self.assertEqual(result["permutation_unit"], "task_label")
        self.assertIn("related", result["by_relation"])

    def test_summary_bundles_types_and_validates_shapes(self):
        rng = np.random.default_rng(5)
        sig_by_type = {"hidden": {"sig": rng.random((15, 8)) < 0.3,
                                  "cluster_labels": [f"h{k}" for k in range(8)]}}
        summary = lesion_plot._task_specificity_summary(sig_by_type, TASKS, n_perm=20)
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(list(summary["types"]), ["hidden"])
        self.assertEqual(summary["types"]["hidden"]["dispersion"]["counts"].shape, (8,))
        with self.assertRaisesRegex(ValueError, "shape"):
            lesion_plot._task_specificity_summary(
                {"hidden": {"sig": np.zeros((14, 8), bool), "cluster_labels": list("abcdefgh")}},
                TASKS, n_perm=5)


if __name__ == "__main__":
    unittest.main()
