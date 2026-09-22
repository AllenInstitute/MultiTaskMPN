"""Tests for sibling-family task-specific endpoint selection metrics."""

import unittest

import numpy as np

from core.sibling_geometry import (
    best_task_specific_pc_pair,
    delaydm_direction_score,
    dmcgo_cross_task_category_score,
)


class SiblingGeometryTests(unittest.TestCase):
    def test_delaydm_score_rewards_separated_compact_directions(self):
        directions = np.repeat(np.arange(4), 4)
        compact = np.column_stack((directions * 4.0, np.zeros(16)))
        compact += np.tile(
            [[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05]],
            (4, 1))
        diffuse = compact.copy()
        diffuse += np.tile(
            [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]],
            (4, 1))

        self.assertGreater(
            delaydm_direction_score(compact, directions),
            delaydm_direction_score(diffuse, directions))

    def test_dmc_score_requires_category_generalization_across_tasks(self):
        task_idx = np.repeat([0, 1], 8)
        categories = np.tile(np.repeat([0, 1], 4), 2)
        aligned = np.column_stack((
            np.where(categories == 0, -2.0, 2.0),
            np.where(task_idx == 0, 0.0, 5.0),
        ))
        reversed_categories = categories.copy()
        reversed_categories[task_idx == 1] = 1 - reversed_categories[task_idx == 1]

        self.assertEqual(
            dmcgo_cross_task_category_score(
                aligned, categories, task_idx),
            1.0,
        )
        self.assertEqual(
            dmcgo_cross_task_category_score(
                aligned, reversed_categories, task_idx),
            0.0,
        )

    def test_best_pair_uses_direction_metric(self):
        directions = np.repeat(np.arange(4), 4)
        points = np.zeros((directions.size, 6), dtype=float)
        centers = np.array([
            [-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0],
        ])
        points[:, 2:4] = centers[directions] + np.tile(
            [[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05]],
            (4, 1))

        pair, score, metric = best_task_specific_pc_pair(
            points, "delaydm1", stim_idx=directions)

        self.assertEqual(pair, (3, 4))
        self.assertGreater(score, 1.0)
        self.assertEqual(
            metric, "direction separation / within-direction dispersion")


if __name__ == "__main__":
    unittest.main()
