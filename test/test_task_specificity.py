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

    def test_null_count_pmf_is_the_exact_shuffle_distribution(self):
        # Two tasks impairing 2 and 1 of 4 clusters: a cluster is hit with
        # probability 1/2 and 1/4, independently, so its count is
        # Poisson-binomial: [3/8, 1/2, 1/8].
        sig = np.array([[1, 1, 0, 0],
                        [1, 0, 0, 0]], dtype=bool)
        np.testing.assert_allclose(lesion_plot._null_count_pmf(sig), [0.375, 0.5, 0.125])
        # Clusters x pmf is the per-count histogram the shuffle averages to,
        # and the dispersion test saves it.
        rng = np.random.default_rng(2)
        sig = rng.random((8, 12)) < 0.4
        result = lesion_plot._count_dispersion_test(sig, n_perm=10)
        pmf = result["null_count_pmf"]
        self.assertEqual(pmf.shape, (9,))
        self.assertAlmostEqual(pmf.sum(), 1.0)
        shuffled = np.zeros(9)
        n_draws = 4000
        for _ in range(n_draws):
            counts = np.stack([rng.permutation(row) for row in sig]).sum(axis=0)
            shuffled += np.bincount(counts, minlength=9)
        np.testing.assert_allclose(shuffled / n_draws, 12 * pmf, atol=0.1)


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

    def test_sharing_null_reproduces_the_permuted_matrix_loop(self):
        rng = np.random.default_rng(11)
        sig = rng.random((len(TASKS), 9)) < 0.3
        result = lesion_plot._sharing_by_relation(sig, TASKS, n_perm=150, seed=4)
        jaccard = lesion_plot._jaccard_matrix(sig)
        labels = lesion_plot._pair_relation_labels(TASKS)
        groups = {name: [pair for pair, label in labels.items() if label == name]
                  for name in result["relations"]}
        groups["related"] = [pair for pair, label in labels.items()
                             if label != lesion_plot.TASK_PAIR_OTHER]

        def mean(matrix, pairs):
            values = np.array([matrix[i, j] for i, j in pairs], dtype=float)
            values = values[np.isfinite(values)]
            return float(values.mean()) if values.size else np.nan

        loop_rng = np.random.default_rng(4)
        for index in range(150):
            perm = loop_rng.permutation(len(TASKS))
            shuffled = jaccard[np.ix_(perm, perm)]
            for name, pairs in groups.items():
                expected = mean(shuffled, pairs)
                actual = result["by_relation"][name]["null_mean"][index]
                if np.isnan(expected):
                    self.assertTrue(np.isnan(actual))
                else:
                    self.assertEqual(actual, expected)

    def test_summary_bundles_types_and_validates_shapes(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(5)
        effect = rng.normal(scale=0.1, size=(15, 8))
        sizes = np.array([1, 2, 3, 5, 8, 13, 21, 34], dtype=float)
        sig_by_type = {"hidden": {"sig": rng.random((15, 8)) < 0.3, "effect": effect,
                                  "sizes": sizes,
                                  "cluster_labels": [f"h{k}" for k in range(8)]}}
        summary = lesion_plot._task_specificity_summary(sig_by_type, TASKS, n_perm=20)
        self.assertEqual(summary["schema_version"], lesion_plot.TASK_SPECIFICITY_SCHEMA_VERSION)
        self.assertEqual(lesion_plot.TASK_SPECIFICITY_SCHEMA_VERSION, 3)
        self.assertEqual(lesion_plot.TASK_SPECIFICITY_MOD_MODE, "freeze_M")
        self.assertEqual(list(summary["types"]), ["hidden"])
        hidden = summary["types"]["hidden"]
        self.assertEqual(hidden["dispersion"]["counts"].shape, (8,))
        # Beside the significance counts: the counts above every fixed effect
        # threshold, the cluster sizes, and how each count follows size.
        self.assertEqual(summary["effect_thresholds"],
                         list(lesion_plot.TASK_SPECIFICITY_EFFECT_THRESHOLDS))
        self.assertEqual(list(hidden["threshold_counts"]), [0.02, 0.05, 0.1, 0.2])
        for threshold, counts in hidden["threshold_counts"].items():
            np.testing.assert_array_equal(counts, (effect > threshold).sum(axis=0))
            self.assertAlmostEqual(hidden["size_correlation"][threshold],
                                   spearmanr(sizes, counts).correlation)
        np.testing.assert_array_equal(hidden["cluster_sizes"], sizes)
        self.assertAlmostEqual(hidden["size_correlation"]["significant"],
                               spearmanr(sizes, hidden["dispersion"]["counts"]).correlation)
        np.testing.assert_array_equal(hidden["effect"], effect)
        # Undefined correlations: too few clusters, or a constant count.
        self.assertTrue(np.isnan(lesion_plot._size_rank_correlation([1, 2], [3, 4])))
        self.assertTrue(np.isnan(lesion_plot._size_rank_correlation([1, 2, 3], [2, 2, 2])))
        self.assertEqual(lesion_plot._size_rank_correlation([1, 2, 3], [1, 2, 3]), 1.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            lesion_plot._task_specificity_summary(
                {"hidden": {"sig": np.zeros((14, 8), bool), "effect": np.zeros((14, 8)),
                            "sizes": sizes, "cluster_labels": list("abcdefgh")}},
                TASKS, n_perm=5)
        with self.assertRaisesRegex(ValueError, "effect or cluster sizes"):
            lesion_plot._task_specificity_summary(
                {"hidden": {"sig": np.zeros((15, 8), bool), "effect": np.zeros((15, 7)),
                            "sizes": sizes, "cluster_labels": list("abcdefgh")}},
                TASKS, n_perm=5)


if __name__ == "__main__":
    unittest.main()
