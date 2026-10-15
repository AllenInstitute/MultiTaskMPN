"""Tests for sibling-family task-specific endpoint selection metrics."""

import unittest

import numpy as np

from core.sibling_geometry import (
    DELAYDM_SWEEP_METRIC_NAME,
    best_delaydm_sweep_pc_pair,
    best_task_specific_pc_pair,
    delaydm_direction_score,
    delaydm_sweep_score,
    dmcgo_cross_task_category_score,
    path_roughness,
)


def _sweep(n_alpha=6, directions=4, per_direction=2, spread=0.05):
    """Straight parallel paths: direction d sits at x = 4 d and every trial
    drifts up by one unit over the sweep; trials of a direction differ by
    `spread` in y. Returns (paths (n_alpha, n_trials, 2), stim_idx)."""
    alphas = np.linspace(0.0, 1.0, n_alpha)
    stim = np.repeat(np.arange(directions), per_direction)
    offsets = np.tile(np.linspace(-spread, spread, per_direction), directions)
    paths = np.zeros((n_alpha, stim.size, 2))
    paths[:, :, 0] = 4.0 * stim[None, :]
    paths[:, :, 1] = alphas[:, None] + offsets[None, :]
    return paths, stim


class SiblingGeometryTests(unittest.TestCase):
    def test_path_roughness_is_zero_for_straight_paths_and_high_for_zigzags(self):
        paths, _ = _sweep()
        np.testing.assert_allclose(path_roughness(paths), 0.0, atol=1e-12)
        zigzag = paths.copy()
        zigzag[1::2, :, 0] += 1.0        # every other alpha steps sideways
        self.assertTrue(np.all(path_roughness(zigzag) > 0.5))
        # A path that barely moves is smooth once the floor exceeds its length.
        still = np.zeros((6, 1, 2))
        still[:, 0, 0] = 1e-3 * np.array([0, 1, 0, 1, 0, 1])
        self.assertGreater(path_roughness(still)[0], 1.0)
        np.testing.assert_allclose(path_roughness(still, floor=1.0), 0.0, atol=0.01)
        with self.assertRaises(ValueError):
            path_roughness(paths[:2])

    def test_sweep_score_penalizes_collapse_along_alpha_and_zigzags(self):
        paths, stim = _sweep()
        score, parts = delaydm_sweep_score(paths, stim)
        self.assertAlmostEqual(parts["smoothness"], 1.0)
        self.assertGreater(parts["separation"], 10.0)
        self.assertAlmostEqual(score, parts["separation"] * parts["smoothness"])
        # Directions that merge at an intermediate alpha lose their minimum
        # separation even though both ends are intact.
        collapsed = paths.copy()
        collapsed[3, :, 0] = 0.0
        collapsed_score, collapsed_parts = delaydm_sweep_score(collapsed, stim)
        self.assertLess(collapsed_parts["separation"], parts["separation"] / 10)
        self.assertLess(collapsed_score, score)
        # Zigzagging paths keep their separation but lose smoothness.
        zigzag = paths.copy()
        zigzag[1::2, :, 1] += 0.5
        zigzag_score, zigzag_parts = delaydm_sweep_score(zigzag, stim)
        self.assertLess(zigzag_parts["smoothness"], 0.8)
        self.assertLess(zigzag_score, score)

    def test_best_sweep_pair_prefers_the_plane_separating_all_directions(self):
        paths, stim = _sweep()
        # Six PCs: PC1-PC2 hold the clean sweep, PC3 collapses the directions
        # at one intermediate alpha, PC4 zigzags, PC5-PC6 are noise. Every PC
        # carries the trials' small within-direction offsets so no plane has a
        # degenerate (zero) spread.
        rng = np.random.RandomState(0)
        offsets = paths[:, :, 1] - paths[:, :, 1].mean(axis=1, keepdims=True)
        proj = np.zeros((paths.shape[0], paths.shape[1], 6))
        proj[:, :, :2] = paths
        proj[:, :, 2] = paths[:, :, 0] * np.where(
            np.arange(paths.shape[0]) == 3, 0.0, 1.0)[:, None] + offsets
        proj[:, :, 3] = paths[:, :, 0] + np.where(
            np.arange(paths.shape[0]) % 2, 1.5, 0.0)[:, None] + offsets
        proj[:, :, 4:] = rng.normal(scale=0.3, size=(paths.shape[0], paths.shape[1], 2))
        pair, score, name = best_delaydm_sweep_pc_pair([proj, proj], [stim, stim])
        self.assertEqual(pair, (1, 2))
        self.assertEqual(name, DELAYDM_SWEEP_METRIC_NAME)
        self.assertGreater(score, 0.0)
        # The weaker base rule decides: a collapsed second sweep lowers the
        # score of every pair that relies on PC1.
        weak = proj.copy()
        weak[:, :, 0] = 0.0
        _, weak_score, _ = best_delaydm_sweep_pc_pair([proj, weak], [stim, stim])
        self.assertLess(weak_score, score)
        with self.assertRaises(ValueError):
            best_delaydm_sweep_pc_pair([proj[:, :, :1]], [stim])

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
