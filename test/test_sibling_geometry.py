"""Tests for the fixed-point separation score that picks the sibling PC pair
and the full-space translation and category-axis metrics of the cohort."""

import unittest

import numpy as np

from core.sibling_geometry import (
    DELAYDM_METRIC_NAME,
    DMC_SWEEP_KEEP_EVERY_ALPHA_METRIC_NAME,
    DMC_SWEEP_METRIC_NAME,
    DMCGO_METRIC_NAME,
    SWEEP_METRIC_NAME,
    best_dmc_sweep_pc_pair,
    best_sweep_pc_pair,
    best_task_specific_pc_pair,
    category_axis_cosine,
    dmc_sweep_axis_scores,
    remove_task_offsets,
    separation_components,
    separation_score,
    sweep_separation_score,
    task_specific_pc_score,
    translation_metrics,
)


SQUARE = np.array([[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05]])


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
        task_idx = np.tile([0, 1], directions.size // 2)
        pair, score, metric = best_task_specific_pc_pair(
            points, "delaydm1", stim_idx=directions, task_idx=task_idx)
        self.assertEqual(pair, (3, 4))
        self.assertGreater(score, 1.0)
        self.assertEqual(metric, DELAYDM_METRIC_NAME)

    def test_delaydm_score_removes_each_tasks_offset(self):
        # Both tasks hold the same four directions, the second translated by
        # 20 units: pooled raw, the same direction of the two tasks is far
        # apart; with each task's mean removed it overlaps again.
        directions = np.repeat(np.arange(4), 4)
        centers = np.array([[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]])
        task_idx = np.tile([0, 1], directions.size // 2)
        points = centers[directions] + np.tile(SQUARE, (4, 1))
        points[task_idx == 1, 0] += 20.0
        aligned = remove_task_offsets(points, task_idx)
        for task in (0, 1):
            np.testing.assert_allclose(aligned[task_idx == task].mean(axis=0), 0.0,
                                       atol=1e-12)
        raw = separation_score(points, directions)
        score, _ = task_specific_pc_score(
            points, "delaydm1", stim_idx=directions, task_idx=task_idx)
        self.assertLess(raw, 1.0)
        self.assertGreater(score, 10 * raw)
        self.assertAlmostEqual(score, separation_score(aligned, directions))
        with self.assertRaises(ValueError):
            task_specific_pc_score(points, "delaydm1", stim_idx=directions)

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

    def test_sweep_score_is_the_worst_alpha_of_every_base_rule(self):
        paths, stim = _sweep()
        clean = sweep_separation_score([paths, paths], [stim, stim])
        self.assertAlmostEqual(clean, min(separation_score(p, stim) for p in paths))
        # A trial hopping to another direction's attractor at one alpha
        # lowers the score even though every other alpha is intact.
        hopping = paths.copy()
        hopping[3, 0, 0] = 4.0
        self.assertLess(sweep_separation_score([paths, hopping], [stim, stim]),
                        clean / 10)
        with self.assertRaises(ValueError):
            sweep_separation_score([paths[:, :, :1]], [stim])

    def test_best_sweep_pair_separates_the_directions_at_every_alpha(self):
        # Every PC carries small noise. PC1 spaces the directions 8 apart but
        # merges them at one intermediate alpha; PC3 spaces them only 2 apart,
        # at every alpha. The PC1 planes separate best at alpha = 0, but only
        # a plane with PC3 keeps the directions apart along the whole sweep.
        paths, stim = _sweep()
        rng = np.random.RandomState(0)
        proj = rng.normal(scale=0.1, size=(paths.shape[0], paths.shape[1], 6))
        proj[:, :, 0] += 2.0 * paths[:, :, 0] * np.where(
            np.arange(paths.shape[0]) == 3, 0.0, 1.0)[:, None]
        proj[:, :, 2] += 0.5 * paths[:, :, 0]
        self.assertGreater(separation_score(proj[0][:, [0, 1]], stim),
                           separation_score(proj[0][:, [2, 3]], stim))
        pair, score, metric = best_sweep_pc_pair([proj, proj], [stim, stim])
        self.assertIn(3, pair)
        self.assertGreater(score, 5.0)
        self.assertEqual(metric, SWEEP_METRIC_NAME)
        with self.assertRaises(ValueError):
            best_sweep_pc_pair([proj[:, :, :1]], [stim])

    @staticmethod
    def _dmc_sweep(n_alpha=5, n_trials=16, seed=0):
        """A DMC sweep with every PC noisy: PC4 keeps the response clusters
        4 apart in the same order, PC2 keeps them 1 apart, PC5 swaps them
        (+1 at alpha = 0, -1 at alpha = 1, crossing at alpha = 0.5) and PC1,
        PC3 and PC6 carry no clusters. Returns (projection, labels)."""
        rng = np.random.RandomState(seed)
        labels = np.tile([0, 1], n_trials // 2)
        sign = np.where(labels == 1, 1.0, -1.0)
        alphas = np.linspace(0.0, 1.0, n_alpha)
        proj = rng.normal(scale=0.05, size=(n_alpha, n_trials, 6))
        proj[:, :, 3] += 2.0 * sign[None, :]
        proj[:, :, 1] += 0.5 * sign[None, :]
        proj[:, :, 4] += (1.0 - 2.0 * alphas)[:, None] * sign[None, :]
        return proj, labels

    def test_dmc_axis_scores_split_kept_and_swapped_pcs(self):
        proj, labels = self._dmc_sweep()
        keep, swap = dmc_sweep_axis_scores(proj, labels)
        self.assertEqual(int(np.argmax(keep)), 3)
        self.assertEqual(int(np.argmax(swap)), 4)
        # A PC scores as kept or as swapped, never both.
        self.assertTrue(np.all((keep == 0) | (swap == 0)))
        self.assertEqual(swap[3], 0.0)
        self.assertEqual(keep[4], 0.0)
        with self.assertRaises(ValueError):
            dmc_sweep_axis_scores(proj, np.zeros(labels.size))

    def test_keep_at_every_alpha_rejects_a_pc_that_flips_mid_sweep(self):
        # PC4 keeps the order at both ends but reverses it at the middle
        # alpha: kept at the ends, not kept at every alpha, where PC2 (1 apart
        # throughout) takes over.
        proj, labels = self._dmc_sweep()
        sign = np.where(labels == 1, 1.0, -1.0)
        proj[2, :, 3] = -2.0 * sign
        ends_keep, _ = dmc_sweep_axis_scores(proj, labels)
        every_keep, every_swap = dmc_sweep_axis_scores(
            proj, labels, keep_at_every_alpha=True)
        self.assertGreater(ends_keep[3], 0.0)
        self.assertEqual(every_keep[3], 0.0)
        self.assertEqual(int(np.argmax(every_keep)), 1)
        pair, _, metric = best_dmc_sweep_pc_pair(
            [proj], [labels], keep_at_every_alpha=True)
        self.assertEqual(pair, (2, 5))
        self.assertEqual(metric, DMC_SWEEP_KEEP_EVERY_ALPHA_METRIC_NAME)

    def test_best_dmc_sweep_pair_puts_the_kept_pc_first(self):
        proj, labels = self._dmc_sweep()
        pair, score, metric = best_dmc_sweep_pc_pair([proj], [labels])
        self.assertEqual(pair, (4, 5))
        keep, swap = dmc_sweep_axis_scores(proj, labels)
        self.assertAlmostEqual(score, min(keep[3], swap[4]))
        self.assertEqual(metric, DMC_SWEEP_METRIC_NAME)
        # Without a swapping PC only noise can swap, so the pair scores at
        # noise level however well the kept PC separates.
        unswapped = proj.copy()
        unswapped[:, :, 4] = proj[0, :, 4][None, :]
        _, flat_score, _ = best_dmc_sweep_pc_pair([unswapped], [labels])
        self.assertLess(flat_score, 0.1 * score)

    def test_best_pair_requires_the_family_labels(self):
        points = np.zeros((8, 6))
        with self.assertRaises(ValueError):
            best_task_specific_pc_pair(points, "dmcgo")
        with self.assertRaises(ValueError):
            best_task_specific_pc_pair(points, "delaydm1")

    def test_translation_metrics_measure_one_shared_displacement(self):
        rng = np.random.default_rng(0)
        first = rng.normal(size=(6, 2, 4))          # flattened like a W * M endpoint
        shift = np.full((2, 4), 3.0)
        metrics = translation_metrics(first, first + shift)
        self.assertAlmostEqual(metrics["translation_explained"], 1.0)
        np.testing.assert_allclose(metrics["translation"], shift.ravel())
        self.assertAlmostEqual(metrics["geometry_r"], 1.0)
        self.assertAlmostEqual(metrics["raw_rms"], np.linalg.norm(shift))
        self.assertAlmostEqual(metrics["residual_rms"], 0.0)
        self.assertEqual(metrics["n_pairs"], 6)
        # A displacement that averages out over the conditions is no translation.
        scattered = translation_metrics(first, -first)
        self.assertLess(scattered["translation_explained"], 0.5)
        self.assertAlmostEqual(scattered["geometry_r"], 1.0)
        with self.assertRaisesRegex(ValueError, "three"):
            translation_metrics(first[:2], first[:2])
        with self.assertRaisesRegex(ValueError, "shape"):
            translation_metrics(first, first[:5])

    def test_category_axis_cosine_reads_the_shared_reversed_or_own_code(self):
        task = np.repeat([0, 1], 4)
        category = np.tile([0, 0, 1, 1], 2)
        points = np.zeros((8, 3))
        points[:4, 0] = category[:4]                 # task 0 codes the category on axis 0
        points[4:, 2] = 5.0                          # task 1 sits elsewhere
        points[4:, 0] = category[4:]
        self.assertAlmostEqual(category_axis_cosine(points, task, category), 1.0)
        points[4:, 0] = 1 - category[4:]
        self.assertAlmostEqual(category_axis_cosine(points, task, category), -1.0)
        points[4:, 0] = 0.0
        points[4:, 1] = category[4:]
        self.assertAlmostEqual(category_axis_cosine(points, task, category), 0.0)
        points[4:, 1] = 0.0                          # task 1 does not separate the categories
        self.assertTrue(np.isnan(category_axis_cosine(points, task, category)))
        with self.assertRaisesRegex(ValueError, "two tasks"):
            category_axis_cosine(points, np.zeros(8, dtype=int), category)
        with self.assertRaisesRegex(ValueError, "lacks"):
            category_axis_cosine(points, task, np.r_[category[:4], np.zeros(4, dtype=int)])


if __name__ == "__main__":
    unittest.main()
