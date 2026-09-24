"""Checks for category-balanced high-dimensional task-center distances."""

import unittest

import _bootstrap  # noqa: F401
import numpy as np

from core.state_space_geometry import context_task_centroids, task_centroid_separation


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


if __name__ == "__main__":
    unittest.main()