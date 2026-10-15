"""Checks for category-balanced high-dimensional task-center distances."""

import unittest

import _bootstrap  # noqa: F401
import numpy as np

from core.state_space_geometry import (
    TASK_FACTOR_PAIRS,
    category_separation_permutation,
    category_separation_permutation_from_distances,
    context_task_centroids,
    factor_vector_consistency,
    mean_task_distance_matrix,
    task_centroid_distances,
    task_centroid_linkage,
    task_centroid_separation,
    task_centroid_separation_from_distances,
    task_distance_consistency,
)


class StateSpaceGeometryTests(unittest.TestCase):
    def test_task_means_do_not_weight_other_tasks_by_trial_count(self):
        states = np.array([[0., 2.], [2., 4.], [10., 20.]])
        centers, counts = context_task_centroids(states, [0, 0, 1], 2)
        np.testing.assert_allclose(centers, [[1., 3.], [10., 20.]])
        np.testing.assert_array_equal(counts, [2, 1])
        repeated, _ = context_task_centroids(
            np.repeat(states, [3, 3, 1], axis=0), [0] * 6 + [1], 2)
        np.testing.assert_allclose(repeated, centers)

    def test_distances_are_category_balanced_not_pair_count_weighted(self):
        centers = np.array([[0.], [2.], [10.], [12.], [14.]])
        metrics = task_centroid_separation(centers, [0, 0, 1, 1, 1])
        within = (2. + (2. + 4. + 2.) / 3) / 2
        between = 11.
        self.assertAlmostEqual(metrics["within_distance"], within)
        self.assertAlmostEqual(metrics["between_distance"], between)
        self.assertAlmostEqual(metrics["score"], (between - within) / (between + within))
        self.assertEqual(metrics["within_pair_count"], 4)
        self.assertEqual(metrics["between_pair_count"], 6)
        from_distances = task_centroid_separation_from_distances(
            task_centroid_distances(centers), [0, 0, 1, 1, 1])
        self.assertAlmostEqual(from_distances["score"], metrics["score"])
        self.assertEqual(from_distances["n_categories"], 2)

    def test_unplotted_dimensions_determine_the_score(self):
        centers = np.array([[0., 0., -5.], [0., 0., -4.],
                            [0., 0., 4.], [0., 0., 5.]])
        metrics = task_centroid_separation(centers, [0, 0, 1, 1])
        self.assertAlmostEqual(metrics["score"], 0.8)
        self.assertEqual(metrics["n_features"], 3)
        with self.assertRaises(ValueError):
            task_centroid_separation(centers[:, :2], [0, 0, 1, 1])

    def test_score_is_translation_and_scale_invariant(self):
        centers = np.array([[0., 0.], [1., 0.], [8., 0.], [9., 0.]])
        original = task_centroid_separation(centers, [0, 0, 1, 1])
        transformed = task_centroid_separation(7 * centers + 30, [0, 0, 1, 1])
        self.assertAlmostEqual(original["score"], transformed["score"])
        self.assertAlmostEqual(transformed["within_distance"],
                               7 * original["within_distance"])

    def test_missing_tasks_and_nonfinite_states_are_rejected(self):
        with self.assertRaises(ValueError):
            context_task_centroids(np.ones((2, 3)), [0, 0], 2)
        with self.assertRaises(ValueError):
            context_task_centroids([[np.nan], [1.]], [0, 1], 2)

    def test_degenerate_category_geometry_is_rejected(self):
        for centers, categories in (
                (np.zeros((4, 3)), [0, 0, 1, 1]),
                (np.ones((3, 3)), [0, 0, 1]),
                (np.ones((4, 3)), [0, 0, 0, 0]),
                (np.full((4, 3), np.nan), [0, 0, 1, 1])):
            with self.subTest(categories=categories), self.assertRaises(ValueError):
                task_centroid_separation(centers, categories)


class CategoryPermutationTests(unittest.TestCase):
    centers = np.array([[0.], [1.], [2.], [20.], [21.], [22.], [40.], [41.], [42.]])
    categories = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]

    def test_grouped_tasks_beat_the_shuffled_label_null(self):
        result = category_separation_permutation(self.centers, self.categories,
                                                 n_permutations=400, seed=1)
        self.assertAlmostEqual(
            result["score"], task_centroid_separation(self.centers, self.categories)["score"])
        self.assertEqual(result["null"].shape, (400,))
        self.assertEqual(result["n_permutations"], 400)
        self.assertLess(result["p_value"], 0.05)
        self.assertLess(result["null_quantile_95"], result["score"])
        self.assertGreater(result["p_value"], 0.0)
        self.assertEqual(result["n_categories"], 3)

    def test_shuffled_categories_are_not_significant(self):
        shuffled = ["a", "b", "c", "a", "b", "c", "a", "b", "c"]
        result = category_separation_permutation(self.centers, shuffled,
                                                 n_permutations=400, seed=1)
        self.assertGreater(result["p_value"], 0.2)
        # Shuffling labels never changes category sizes, so every null score is
        # a valid category-balanced score of the same geometry.
        self.assertTrue(np.all(np.abs(result["null"]) <= 1.0))

    def test_null_is_reproducible_by_seed(self):
        first = category_separation_permutation(self.centers, self.categories,
                                                n_permutations=50, seed=3)
        second = category_separation_permutation(self.centers, self.categories,
                                                 n_permutations=50, seed=3)
        third = category_separation_permutation(self.centers, self.categories,
                                                n_permutations=50, seed=4)
        np.testing.assert_array_equal(first["null"], second["null"])
        self.assertFalse(np.array_equal(first["null"], third["null"]))
        with self.assertRaises(ValueError):
            category_separation_permutation(self.centers, self.categories, n_permutations=0)

    def test_distance_matrix_version_matches_the_centroid_version(self):
        from_centroids = category_separation_permutation(
            self.centers, self.categories, n_permutations=50, seed=3)
        from_distances = category_separation_permutation_from_distances(
            task_centroid_distances(self.centers), self.categories, n_permutations=50, seed=3)
        np.testing.assert_array_equal(from_centroids["null"], from_distances["null"])
        self.assertEqual(from_centroids["score"], from_distances["score"])
        self.assertEqual(from_centroids["p_value"], from_distances["p_value"])
        self.assertNotIn("n_features", from_distances)
        with self.assertRaises(ValueError):
            category_separation_permutation_from_distances(
                np.array([[0., 1.], [2., 0.]]), ["a", "b"], n_permutations=5)


class CohortDistanceTests(unittest.TestCase):
    centers = np.array([[0.], [1.], [2.], [20.], [21.], [22.]])

    def test_mean_matrix_puts_runs_of_different_scale_on_one_footing(self):
        base = task_centroid_distances(self.centers)
        mean, scales = mean_task_distance_matrix([base, 3.0 * base])
        upper = np.triu_indices(6, 1)
        np.testing.assert_allclose(scales, [base[upper].mean(), 3.0 * base[upper].mean()])
        np.testing.assert_allclose(mean, base / base[upper].mean())
        self.assertAlmostEqual(mean[upper].mean(), 1.0)
        # Without the normalization the larger run would dominate; with it, a
        # second run of a different shape counts as much as the first.
        other = task_centroid_distances(self.centers[::-1] ** 2)
        mean_two, _ = mean_task_distance_matrix([base, 10.0 * other])
        mean_unit, _ = mean_task_distance_matrix([base, other])
        np.testing.assert_allclose(mean_two, mean_unit)

    def test_consistency_is_scale_free_and_drops_for_a_different_geometry(self):
        base = task_centroid_distances(self.centers)
        same = task_distance_consistency([base, 3.0 * base, base + 0.0])
        self.assertEqual(same["n_runs"], 3)
        self.assertEqual(same["n_run_pairs"], 3)
        np.testing.assert_allclose(same["pairwise_spearman"], 1.0)
        self.assertAlmostEqual(same["median"], 1.0)
        shuffled = task_centroid_distances(np.array([[0.], [20.], [1.], [21.], [2.], [22.]]))
        mixed = task_distance_consistency([base, shuffled])
        self.assertLess(mixed["median"], 0.9)
        single = task_distance_consistency([base])
        self.assertEqual(single["n_run_pairs"], 0)
        self.assertTrue(np.isnan(single["median"]))
        with self.assertRaises(ValueError):
            mean_task_distance_matrix([base, base[:4, :4]])
        with self.assertRaises(ValueError):
            mean_task_distance_matrix([])


class LinkageTests(unittest.TestCase):
    def test_average_linkage_places_close_tasks_next_to_each_other(self):
        centers = np.array([[0.], [20.], [40.], [1.], [21.], [41.]])
        tree = task_centroid_linkage(task_centroid_distances(centers))
        order = list(tree["leaf_order"])
        self.assertEqual(sorted(order), list(range(6)))
        self.assertEqual(tree["linkage"].shape, (5, 4))
        self.assertEqual(tree["method"], "average")
        for pair in ((0, 3), (1, 4), (2, 5)):
            self.assertEqual(abs(order.index(pair[0]) - order.index(pair[1])), 1)

    def test_linkage_requires_a_symmetric_zero_diagonal_matrix(self):
        with self.assertRaises(ValueError):
            task_centroid_linkage(np.array([[0., 1.], [2., 0.]]))
        with self.assertRaises(ValueError):
            task_centroid_linkage(np.array([[1., 1.], [1., 1.]]))


def _additive_centroids(rules, noise=0.05, seed=0):
    """Centroids whose factors add: each task is the sum of the offsets of the
    factors it carries (anti, memory, reaction, modality 2, context cue,
    category), plus a small random component."""
    offsets = {
        "anti": np.array([3., 0, 0, 0, 0, 0, 0]),
        "memory": np.array([0, 2., 0, 0, 0, 0, 0]),
        "reaction": np.array([0, 0, 2., 0, 0, 0, 0]),
        "modality2": np.array([0, 0, 0, 2., 0, 0, 0]),
        "context": np.array([0, 0, 0, 0, 2., 0, 0]),
        "category": np.array([0, 0, 0, 0, 0, 2., 0]),
        "integration": np.array([0, 0, 0, 0, 0, 0, 4.]),
    }
    membership = {
        "fdgo": (), "fdanti": ("anti",),
        "delaygo": ("memory",), "delayanti": ("memory", "anti"),
        "reactgo": ("reaction",), "reactanti": ("reaction", "anti"),
        "dmsgo": (), "dmsnogo": ("anti",),
        "dmcgo": ("category",), "dmcnogo": ("category", "anti"),
        "delaydm1": ("integration",), "delaydm2": ("integration", "modality2"),
        "contextdelaydm1": ("integration", "context"),
        "contextdelaydm2": ("integration", "context", "modality2"),
        "multidelaydm": ("integration",),
    }
    rng = np.random.default_rng(seed)
    return np.stack([
        sum((offsets[name] for name in membership[rule]), np.zeros(7))
        + noise * rng.normal(size=7)
        for rule in rules
    ])


class FactorVectorTests(unittest.TestCase):
    rules = list(TASK_FACTOR_PAIRS["pro to anti"][0]) + [
        "delaygo", "delayanti", "reactgo", "reactanti", "dmsgo", "dmsnogo",
        "dmcgo", "dmcnogo", "delaydm1", "delaydm2", "contextdelaydm1",
        "contextdelaydm2", "multidelaydm"]

    def test_additive_code_gives_parallel_vectors_for_every_factor(self):
        centroids = _additive_centroids(self.rules)
        result = factor_vector_consistency(centroids, self.rules, n_permutations=300, seed=0)
        self.assertEqual(list(result["factors"]), list(TASK_FACTOR_PAIRS))
        self.assertEqual(result["skipped_factors"], [])
        for factor, stats in result["factors"].items():
            with self.subTest(factor=factor):
                self.assertGreater(stats["mean_cosine"], 0.95)
                self.assertGreater(stats["shared_variance_fraction"], 0.95)
                self.assertLess(stats["p_value"], 0.05)
                self.assertEqual(len(stats["pairs"]), len(TASK_FACTOR_PAIRS[factor]))
                self.assertEqual(stats["pairwise_cosines"].shape,
                                 (len(stats["pairs"]) * (len(stats["pairs"]) - 1) // 2,))
                self.assertEqual(stats["null"].shape, (300,))
        self.assertGreater(result["pooled"]["mean_cosine"], 0.95)
        self.assertLess(result["pooled"]["p_value"], 0.05)

    def test_gram_shortcut_matches_direct_cosines_and_pooled_null(self):
        centroids = np.random.default_rng(5).normal(size=(len(self.rules), 9))
        result = factor_vector_consistency(centroids, self.rules, n_permutations=40, seed=2)
        position = {rule: index for index, rule in enumerate(self.rules)}
        for factor, stats in result["factors"].items():
            vectors = np.stack([centroids[position[b]] - centroids[position[a]]
                                for a, b in stats["pairs"]])
            unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            cosines = unit @ unit.T
            expected = cosines[np.triu_indices(len(vectors), 1)]
            np.testing.assert_allclose(stats["pairwise_cosines"], expected, atol=1e-10)
            self.assertAlmostEqual(stats["mean_cosine"], expected.mean())
            singular = np.linalg.svd(vectors, compute_uv=False)
            self.assertAlmostEqual(stats["shared_variance_fraction"],
                                   singular[0] ** 2 / np.sum(singular ** 2))
        factor_nulls = np.stack([stats["null"] for stats in result["factors"].values()], axis=1)
        np.testing.assert_allclose(result["pooled"]["null"], factor_nulls.mean(axis=1))
        self.assertAlmostEqual(
            result["pooled"]["mean_cosine"],
            np.mean([stats["mean_cosine"] for stats in result["factors"].values()]))

    def test_random_centroids_are_not_consistently_parallel(self):
        centroids = np.random.default_rng(11).normal(size=(len(self.rules), 50))
        result = factor_vector_consistency(centroids, self.rules, n_permutations=300, seed=0)
        self.assertLess(abs(result["pooled"]["mean_cosine"]), 0.3)
        self.assertGreater(result["pooled"]["p_value"], 0.05)

    def test_factors_without_two_pairs_are_skipped_not_computed(self):
        rules = ["fdgo", "fdanti", "delaygo", "delayanti", "delaydm1"]
        centroids = _additive_centroids(rules)
        result = factor_vector_consistency(centroids, rules, n_permutations=20, seed=0)
        self.assertEqual(set(result["factors"]), {"pro to anti", "delay to memory"})
        self.assertEqual(set(result["skipped_factors"]),
                         set(TASK_FACTOR_PAIRS) - {"pro to anti", "delay to memory"})
        with self.assertRaises(ValueError):
            factor_vector_consistency(centroids[:2], ["fdgo", "delaydm1"], n_permutations=20)
        with self.assertRaises(ValueError):
            factor_vector_consistency(centroids, rules[:-1], n_permutations=20)


if __name__ == "__main__":
    unittest.main()
