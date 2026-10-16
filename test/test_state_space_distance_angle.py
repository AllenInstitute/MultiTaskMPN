"""Distance-angle statistics of state_space_shift.py: the task-label
permutation test, the vectorized pair statistics, the Hebbian-only step, the
single-checkpoint CLI and the paper figures that read the saved values."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import _bootstrap  # noqa: F401
import state_space_shift as state_space
import state_space_geometry as geometry
import paper_plot


def _loop_pair_statistics(h0_1, h0_2, dh1, dh2):
    """The per-pair loop the vectorized statistics replaced."""
    pair_dists, pair_angles = [], []
    for i in range(len(h0_1)):
        for j in range(len(h0_2)):
            d = np.linalg.norm(h0_1[i] - h0_2[j])
            n1 = np.linalg.norm(dh1[i])
            n2 = np.linalg.norm(dh2[j])
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            cosang = np.clip(np.dot(dh1[i], dh2[j]) / (n1 * n2), -1.0, 1.0)
            pair_dists.append(d)
            pair_angles.append(np.degrees(np.arccos(cosang)))
    return np.mean(pair_dists), np.mean(pair_angles), len(pair_dists)


def _symmetric(rng, n):
    matrix = rng.normal(size=(n, n))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    return matrix


class MantelTests(unittest.TestCase):
    def test_pair_values_to_matrix_round_trip(self):
        n = 4
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        values = np.arange(len(pairs), dtype=float) + 1
        matrix = geometry.pair_values_to_matrix(values, pairs, n)
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), 0)
        np.testing.assert_array_equal(matrix[np.triu_indices(n, 1)], values)
        with self.assertRaises(ValueError):
            geometry.pair_values_to_matrix(values[:-1], pairs[:-1], n)
        with self.assertRaises(ValueError):
            geometry.pair_values_to_matrix(values, pairs[:-1] + [pairs[0]], n)

    def test_mantel_matches_the_per_permutation_correlation(self):
        rng = np.random.default_rng(1)
        n = 6
        x = _symmetric(rng, n)
        y = x + 0.3 * _symmetric(rng, n)
        result = geometry.mantel_pearson_permutation(x, y, n_permutations=50, seed=3)
        upper = np.triu_indices(n, 1)
        self.assertAlmostEqual(result["r"], np.corrcoef(x[upper], y[upper])[0, 1])
        expected_rng = np.random.default_rng(3)
        expected = []
        for _ in range(50):
            order = expected_rng.permutation(n)
            permuted = y[np.ix_(order, order)]
            expected.append(np.corrcoef(x[upper], permuted[upper])[0, 1])
        np.testing.assert_allclose(result["null"], expected)
        self.assertEqual(result["p_value"],
                         (1 + np.sum(np.abs(expected) >= abs(result["r"]))) / 51)
        self.assertEqual(result["sided"], "two-sided")
        self.assertEqual(result["n_permutations"], 50)

    def test_mantel_detects_a_perfect_association_and_rejects_bad_inputs(self):
        rng = np.random.default_rng(0)
        n = 8
        points = rng.normal(size=(n, 3))
        x = np.linalg.norm(points[:, None] - points[None], axis=-1)
        result = geometry.mantel_pearson_permutation(x, 2 * x + 1, n_permutations=999, seed=0)
        self.assertAlmostEqual(result["r"], 1.0)
        self.assertLess(result["p_value"], 0.01)
        with self.assertRaises(ValueError):
            geometry.mantel_pearson_permutation(x, np.ones_like(x) - np.eye(n), n_permutations=10)
        with self.assertRaises(ValueError):
            geometry.mantel_pearson_permutation(x, x[:-1, :-1], n_permutations=10)
        with self.assertRaises(ValueError):
            geometry.mantel_pearson_permutation(x, x, n_permutations=0)


class PairStatisticsTests(unittest.TestCase):
    def test_vectorized_pair_statistics_match_the_loop(self):
        rng = np.random.default_rng(5)
        h0_1, h0_2 = rng.normal(size=(4, 7)), rng.normal(size=(3, 7))
        dh1, dh2 = rng.normal(size=(4, 7)), rng.normal(size=(3, 7))
        dh2[1] = 0.0  # a zero step is skipped by both implementations
        got = state_space._cross_pair_statistics(h0_1, h0_2, dh1, dh2)
        expected = _loop_pair_statistics(h0_1, h0_2, dh1, dh2)
        np.testing.assert_allclose(got[:2], expected[:2])
        self.assertEqual(got[2], expected[2])
        self.assertEqual(got[2], 8)
        self.assertIsNone(state_space._cross_pair_statistics(
            h0_1, h0_2, np.zeros_like(dh1), dh2))

    def test_first_step_removes_the_decay_term_with_retention(self):
        data = np.random.default_rng(2).normal(size=(5, 4, 3))
        trials, onset = np.array([0, 2]), 2
        initial, full = state_space._first_step(data, trials, onset)
        np.testing.assert_allclose(initial, data[trials, 1])
        np.testing.assert_allclose(full, data[trials, 2] - data[trials, 1])
        _, hebbian = state_space._first_step(data, trials, onset, retention=0.9)
        np.testing.assert_allclose(hebbian, data[trials, 2] - 0.9 * data[trials, 1])
        np.testing.assert_allclose(hebbian - full, 0.1 * initial)

    def test_hebbian_step_angles_follow_the_update_rule(self):
        """States built as x[t] = lam x[t-1] + outer(post, pre): the Hebbian-
        only angle between two tasks is the angle between their outer products
        whatever their initial conditions; the full step is not."""
        rng = np.random.default_rng(7)
        lam, n_rules, n_trials, n_post, n_pre = 0.9, 3, 2, 4, 5
        rule_indices = np.repeat(np.arange(n_rules), n_trials)
        stimulus = np.zeros(rule_indices.size, dtype=int)
        initial = rng.normal(size=(rule_indices.size, n_post * n_pre))
        posts = rng.normal(size=(n_rules, n_post))
        pres = rng.normal(size=(n_rules, n_pre))
        hebbian = np.stack([np.outer(posts[rule], pres[rule]).ravel() for rule in rule_indices])
        data = np.stack([initial, lam * initial + hebbian], axis=1)
        onset_by_rule = {rule: 1 for rule in range(n_rules)}

        pairs, distances, angles = state_space._task_pair_distance_angle(
            data, rule_indices, stimulus, onset_by_rule, n_rules, retention=lam, n_stimuli=1)
        self.assertEqual(pairs, [(0, 1), (0, 2), (1, 2)])
        for (first, second), angle in zip(pairs, angles):
            cosine = (np.dot(posts[first], posts[second]) * np.dot(pres[first], pres[second])
                      / (np.linalg.norm(posts[first]) * np.linalg.norm(posts[second])
                         * np.linalg.norm(pres[first]) * np.linalg.norm(pres[second])))
            self.assertAlmostEqual(angle, np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        first_trials = initial[rule_indices == 0]
        second_trials = initial[rule_indices == 1]
        self.assertAlmostEqual(distances[0], np.mean([
            np.linalg.norm(a - b) for a in first_trials for b in second_trials]))
        _, full_distances, full_angles = state_space._task_pair_distance_angle(
            data, rule_indices, stimulus, onset_by_rule, n_rules, n_stimuli=1)
        np.testing.assert_allclose(full_distances, distances)
        self.assertFalse(np.allclose(full_angles, angles))

    def test_step_norm_summary(self):
        lam = 0.8
        data = np.random.default_rng(3).normal(size=(4, 3, 6))
        rule_indices = np.array([0, 0, 1, 1])
        onset_by_rule = {0: 1, 1: 2}
        norms = state_space._step_norm_summary(data, rule_indices, onset_by_rule, 2, retention=lam)
        initial = np.concatenate([data[:2, 0], data[2:, 1]])
        following = np.concatenate([data[:2, 1], data[2:, 2]])
        self.assertAlmostEqual(norms["initial"], np.linalg.norm(initial, axis=1).mean())
        self.assertAlmostEqual(norms["full_step"],
                               np.linalg.norm(following - initial, axis=1).mean())
        self.assertAlmostEqual(norms["decay_term"],
                               (1 - lam) * np.linalg.norm(initial, axis=1).mean())
        self.assertAlmostEqual(norms["hebbian_term"],
                               np.linalg.norm(following - lam * initial, axis=1).mean())
        self.assertEqual(set(state_space._step_norm_summary(data, rule_indices, onset_by_rule, 2)),
                         {"initial", "full_step"})

    def test_permutation_is_skipped_for_an_incomplete_pair_set(self):
        permutation, reason = state_space._permutation_test_or_reason(
            [(0, 1), (0, 2)], np.array([1., 2.]), np.array([3., 4.]), 3)
        self.assertIsNone(permutation)
        self.assertIn("2 of 3", reason)


class RetentionTests(unittest.TestCase):
    def test_reads_scalar_lam_and_checks_it_against_m_time_scale(self):
        state_dict = {"mp_layer1.lam": torch.tensor(0.99)}
        self.assertAlmostEqual(state_space._modulation_retention(
            state_dict, {"ml_params": {"m_time_scale": 4000}}, 40), 0.99, places=6)
        self.assertAlmostEqual(state_space._modulation_retention(state_dict, {}, 40), 0.99,
                               places=6)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            state_space._modulation_retention(state_dict, {"ml_params": {"m_time_scale": 400}}, 40)
        with self.assertRaises(KeyError):
            state_space._modulation_retention({}, {}, 40)
        with self.assertRaisesRegex(ValueError, "scalar"):
            state_space._modulation_retention({"mp_layer1.lam": torch.full((3,), 0.99)}, {}, 40)
        with self.assertRaisesRegex(ValueError, r"\(0, 1\)"):
            state_space._modulation_retention({"mp_layer1.lam": torch.tensor(1.0)}, {}, 40)


class CheckpointRunTests(unittest.TestCase):
    def test_run_checkpoints_merges_into_existing_results_without_clearing(self):
        def fake_eval(path, output_dir):
            return (state_space._checkpoint_aname(path), 300, 1e-4,
                    {"hidden": (0.5, 1.0, 0.01)}, {"hidden": {}}, 12, 480.0, 0.99)

        old = "everything_seed1_L21e4+hidden300+batch128+angle"
        new = "everything_seed2_L21e4+hidden300+batch128+angle"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            (out / "keep.png").write_bytes(b"x")
            with (out / state_space.RESULTS_NAME).open("wb") as stream:
                pickle.dump({old: {"l2_info": 1e-4}}, stream)
            checkpoint = root / f"savednet_{new}.pt"
            checkpoint.touch()
            with patch.object(state_space, "eval_one", side_effect=fake_eval), \
                    patch.object(state_space, "_matches_target_cohort", return_value=True):
                result = state_space.run_checkpoints([str(checkpoint)], out)
            with (out / state_space.RESULTS_NAME).open("rb") as stream:
                saved = pickle.load(stream)
            self.assertEqual(set(saved), {old, new})
            entry = saved[new]
            self.assertEqual(entry["schema_version"], state_space.DISTANCE_ANGLE_SCHEMA_VERSION)
            self.assertEqual(entry["retention"], 0.99)
            self.assertEqual(entry["n_permutations"], state_space.DISTANCE_ANGLE_N_PERMUTATIONS)
            self.assertEqual(entry["fixed_fixation_ms"], 480.0)
            self.assertTrue((out / "keep.png").exists())
            self.assertEqual(result.keys(), saved.keys())
            with self.assertRaises(FileNotFoundError):
                state_space.run_checkpoints([str(root / "savednet_missing.pt")], out)

    def test_cli_defaults_to_the_batch_run(self):
        args = state_space._parse_args([])
        self.assertIsNone(args.checkpoint)
        self.assertEqual(Path(args.out_dir), state_space.STATE_SPACE_DIR)
        args = state_space._parse_args(["--checkpoint", "a.pt", "b.pt", "--out-dir", "scratch"])
        self.assertEqual(args.checkpoint, ["a.pt", "b.pt"])
        self.assertEqual(args.out_dir, "scratch")
        with patch.object(state_space, "run_checkpoints") as run_checkpoints, \
                patch.object(state_space, "run_all") as run_all, \
                patch.object(state_space, "summarize") as summarize:
            state_space.main(["--checkpoint", "a.pt", "--out-dir", "scratch"])
            run_checkpoints.assert_called_once_with(["a.pt"], Path("scratch"))
            run_all.assert_not_called()
            summarize.assert_called_once_with(Path("scratch"))
            state_space.main([])
            run_all.assert_called_once_with(state_space.STATE_SPACE_DIR)


def _distance_angle_entry(with_permutation=True, with_hebb=True):
    """A results entry as state_space_shift.py saves it, with ten task pairs."""
    rng = np.random.default_rng(0)
    names = ["hidden", "mod", "eff_mod"] + (["mod_hebb", "eff_mod_hebb"] if with_hebb else [])
    entry = {"l2_info": 1e-4, "rval_dict": {}, "scatter": {}}
    for index, name in enumerate(names):
        x = rng.uniform(1, 3, 10)
        entry["rval_dict"][name] = (0.9 - 0.1 * index, 10.0, 0.001)
        scatter = {
            "dists": x,
            "angles_deg": 10 * x + rng.normal(size=10),
            "regression": {"through_origin": False, "intercept": 1.0},
            "step_norms": {"initial": 1.0, "full_step": 0.5,
                           "decay_term": 0.05, "hebbian_term": 0.48},
        }
        if with_permutation:
            scatter["permutation"] = {"p_value": 0.02 if index % 2 == 0 else 0.3}
        entry["scatter"][name] = scatter
    return entry


class StateSpaceDistanceAngleFigureTests(unittest.TestCase):
    def _draw(self, function, entry):
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             return_value=("run", 0.1, entry)), \
                patch.object(paper_plot, "_save_fig") as save_fig:
            function()
        if save_fig.called:
            self.addCleanup(paper_plot.plt.close, save_fig.call_args.args[0])
        return save_fig

    def test_distance_angle_panel_requires_the_permutation_test(self):
        with patch("builtins.print") as output:
            self.assertIsNone(paper_plot._distance_angle_panel(
                _distance_angle_entry(with_permutation=False), "hidden"))
        self.assertIn("permutation", str(output.call_args_list[0]))
        panel = paper_plot._distance_angle_panel(_distance_angle_entry(), "hidden")
        self.assertEqual((panel["r"], panel["slope"], panel["intercept"], panel["p_perm"]),
                         (0.9, 10.0, 1.0, 0.02))

    def test_dist_angle_figure_draws_hidden_and_both_modulation_steps(self):
        save_fig = self._draw(paper_plot.plot_state_space_dist_angle, _distance_angle_entry())
        figure = save_fig.call_args.args[0]
        self.assertEqual(Path(save_fig.call_args.args[1]).name,
                         "multitask_state_space_dist_angle.png")
        # hidden, eff_mod and eff_mod_hebb are the fixture's representations
        # 0, 2 and 4: the permutation p, never the nominal one, is reported.
        texts = [text.get_text() for ax in figure.axes for text in ax.get_legend().get_texts()]
        self.assertEqual(texts, ["r = 0.90\npermutation p = 0.020",
                                 "r = 0.70\npermutation p = 0.020",
                                 "r = 0.50\npermutation p = 0.020"])
        self.assertEqual([ax.get_title() for ax in figure.axes],
                         ["Hidden state", "Eff. modulation",
                          "Eff. modulation,\nHebbian term only"])
        # The two W * M panels share one y range; the hidden panel keeps its own.
        self.assertEqual(figure.axes[1].get_ylim(), figure.axes[2].get_ylim())
        self.assertNotEqual(figure.axes[0].get_ylim(), figure.axes[1].get_ylim())
        self.assertIn("eff_mod decay term / full step = 0.100",
                      save_fig.call_args.kwargs["extra"])

    def test_dist_angle_figure_skips_a_legacy_cache(self):
        save_fig = self._draw(paper_plot.plot_state_space_dist_angle,
                              _distance_angle_entry(with_permutation=False))
        save_fig.assert_not_called()

    def test_dist_angle_figure_skips_caches_without_the_hebbian_step(self):
        save_fig = self._draw(paper_plot.plot_state_space_dist_angle,
                              _distance_angle_entry(with_hebb=False))
        save_fig.assert_not_called()

    def _draw_r_values(self, result_dict, untrained_path, selected=("run", 0.1, {})):
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_pkl_or_skip", return_value=result_dict), \
                patch.object(paper_plot, "STATE_SPACE_UNTRAINED_RVAL_PATH", untrained_path), \
                patch.object(paper_plot, "_load_best_state_space_trial", return_value=selected), \
                patch.object(paper_plot, "_save_fig") as save_fig, \
                patch("builtins.print") as output:
            paper_plot.plot_state_space_r_values()
        figure = save_fig.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        return figure, save_fig, output

    def test_r_values_figure_hatches_hebbian_bars_and_fills_significant_runs(self):
        result_dict = {f"run{index}": _distance_angle_entry() for index in range(2)}
        with tempfile.TemporaryDirectory() as tmp:
            # Without untrained results the four cohort panels stay as they were.
            figure, save_fig, output = self._draw_r_values(result_dict, Path(tmp) / "missing.pkl")
        self.assertEqual(Path(save_fig.call_args.args[1]).name,
                         "multitask_state_space_r_values.png")
        self.assertEqual(len(figure.axes), len(paper_plot.STATE_SPACE_L2_STRENGTHS))
        self.assertIn("--untrained", output.call_args_list[0].args[0])
        panel = figure.axes[list(paper_plot.STATE_SPACE_L2_STRENGTHS).index(1e-4)]
        bars = [patch_ for patch_ in panel.patches
                if isinstance(patch_, paper_plot.mpl.patches.Rectangle)]
        self.assertEqual([bar.get_hatch() for bar in bars], [None, None, None, "///", "///"])
        np.testing.assert_allclose([bar.get_height() for bar in bars],
                                   [0.9, 0.8, 0.7, 0.6, 0.5])
        # Two runs per bar: the dots of the significant representations
        # (hidden, eff_mod, eff_mod_hebb) are filled black, the others white.
        dots = [collection for collection in panel.collections
                if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
        faces = [tuple(collection.get_facecolor()[0][:3]) for collection in dots]
        self.assertEqual(faces, [(0, 0, 0)] * 2 + [(1, 1, 1)] * 2 + [(0, 0, 0)] * 2
                         + [(1, 1, 1)] * 2 + [(0, 0, 0)] * 2)
        empty = [ax for ax in figure.axes if ax is not panel]
        self.assertTrue(all(any(text.get_text() == "No data" for text in ax.texts)
                            for ax in empty))

    def test_r_values_figure_adds_the_untrained_networks_as_a_first_panel(self):
        result_dict = {f"run{index}": _distance_angle_entry() for index in range(2)}
        untrained = {}
        for seed, config in ((0, "run"), (1, "run"), (0, "other")):
            entry = _distance_angle_entry()
            entry.update(untrained=True, init_seed=seed, config_source=config, l2_info=None)
            # Untrained r values sit 0.5 below the trained fixture's.
            entry["rval_dict"] = {name: (r - 0.5, slope, p)
                                  for name, (r, slope, p) in entry["rval_dict"].items()}
            untrained[f"{config}_init{seed}"] = entry
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "untrained_results.pkl"
            with path.open("wb") as stream:
                pickle.dump(untrained, stream)
            figure, _, _ = self._draw_r_values(result_dict, path)
        self.assertEqual(len(figure.axes), len(paper_plot.STATE_SPACE_L2_STRENGTHS) + 1)
        self.assertEqual(figure.axes[0].get_title(), "Untrained")
        self.assertEqual(figure.axes[0].get_ylabel(), "R-value")
        bars = [patch_ for patch_ in figure.axes[0].patches
                if isinstance(patch_, paper_plot.mpl.patches.Rectangle)]
        np.testing.assert_allclose([bar.get_height() for bar in bars],
                                   [0.4, 0.3, 0.2, 0.1, 0.0], atol=1e-12)
        # Two networks of the example run's config per bar; the other
        # config's network is left out.
        dots = [collection for collection in figure.axes[0].collections
                if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
        self.assertEqual(len(dots), 10)
        # The trained cohorts keep their panels, shifted one to the right.
        self.assertEqual(figure.axes[1 + list(paper_plot.STATE_SPACE_L2_STRENGTHS).index(1e-4)]
                         .get_title(), r"$L_2 = 10^{-4}$")


if __name__ == "__main__":
    unittest.main()
