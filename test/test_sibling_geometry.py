"""Tests for the fixed-point separation score that picks the sibling PC pair."""

import unittest

import numpy as np

from core.sibling_geometry import (
    DELAYDM_METRIC_NAME,
    DMCGO_METRIC_NAME,
    best_task_specific_pc_pair,
    separation_components,
    separation_score,
)


SQUARE = np.array([[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05]])


class SiblingGeometryTests(unittest.TestCase):
    def test_score_rewards_separated_compact_groups(self):
        groups = np.repeat(np.arange(4), 4)
        compact = np.column_stack((groups * 4.0, np.zeros(16))) + np.tile(SQUARE, (4, 1))
        diffuse = compact + np.tile(40.0 * SQUARE, (4, 1))
        self.assertGreater(separation_score(compact, groups),
                           separation_score(diffuse, groups))

    def test_components_use_each_groups_nearest_competitor(self):
        # Groups 0 and 1 sit 1 apart, group 2 far away: the between term is
        # the mean nearest-centroid distance (1, 1, 9), not the mean of all.
        groups = np.repeat(np.arange(3), 4)
        centers = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        points = centers[groups] + np.tile(SQUARE, (3, 1))
        between, within = separation_components(points, groups)
        self.assertAlmostEqual(between, (1.0 + 1.0 + 9.0) / 3)
        self.assertAlmostEqual(within, 0.05)

    def test_score_needs_two_groups_of_two_points(self):
        with self.assertRaises(ValueError):
            separation_score(np.zeros((4, 2)), np.zeros(4))
        with self.assertRaises(ValueError):
            separation_score(np.zeros((3, 2)), np.array([0, 0, 1]))

    def test_best_pair_separates_delaydm_directions(self):
        directions = np.repeat(np.arange(4), 4)
        centers = np.array([[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]])
        points = np.zeros((directions.size, 6))
        points[:, 2:4] = centers[directions] + np.tile(SQUARE, (4, 1))
        pair, score, metric = best_task_specific_pc_pair(
            points, "delaydm1", stim_idx=directions)
        self.assertEqual(pair, (3, 4))
        self.assertGreater(score, 1.0)
        self.assertEqual(metric, DELAYDM_METRIC_NAME)

    def test_best_pair_separates_dmc_response_categories_over_both_tasks(self):
        # PC1 separates the two tasks, PC2 the response categories with the
        # same sign in both tasks; PC3-PC6 are small noise. Pooled over both
        # tasks the category plane must avoid PC1, whose task gap would count
        # as within-category spread.
        task = np.repeat([0, 1], 8)
        categories = np.tile(np.repeat([0, 1], 4), 2)
        rng = np.random.RandomState(0)
        points = rng.normal(scale=0.05, size=(task.size, 6))
        points[:, 0] += np.where(task == 0, -5.0, 5.0)
        points[:, 1] += np.where(categories == 0, -1.0, 1.0)
        pair, score, metric = best_task_specific_pc_pair(
            points, "dmcgo", group_labels=categories)
        self.assertNotIn(1, pair)
        self.assertIn(2, pair)
        self.assertGreater(score, 5.0)
        self.assertEqual(metric, DMCGO_METRIC_NAME)

    def test_best_pair_requires_the_family_labels(self):
        points = np.zeros((8, 6))
        with self.assertRaises(ValueError):
            best_task_specific_pc_pair(points, "dmcgo")
        with self.assertRaises(ValueError):
            best_task_specific_pc_pair(points, "delaydm1")


if __name__ == "__main__":
    unittest.main()
