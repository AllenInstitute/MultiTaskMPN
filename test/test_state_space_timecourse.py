"""Multi-step distance-angle analysis: alignment, fixed distances and cache preservation."""

import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import torch

import paper_plot
import state_space_shift as state_space
import state_space_evaluation as evaluation_utils
from test_modulation_feedback import _tiny_network


def _states():
    rng = np.random.default_rng(19)
    rules = list(state_space.RULE_MOTIF_MAPPING)[:4]
    return {
        "aname": "everything_seed202_L21e4+hidden300+batch128+angle",
        "hidden_size": 3, "l2_info": 1e-4, "retention": 0.9,
        "dt_ms": 40, "fixed_fixation_steps": 2, "accuracy": float("nan"),
        "all_rules": rules,
        "rule_indices": np.repeat(np.arange(4), 4),
        "stimulus_labels": np.tile([0, 0, 1, 1], 4),
        "rules_epochs": {rule: {"fix1": (0, 2), "stim1": (2, 7),
                               "go1": (7, None)} for rule in rules},
        **{key: rng.normal(size=(16, 7, 3)) for key in ("hidden", "mod", "eff_mod")},
    }


def _analysis(states, step=1, initial_scatter=None):
    return state_space.distance_angle_analysis(
        {key: states[key] for key in ("hidden", "mod", "eff_mod")},
        states["rule_indices"], states["stimulus_labels"],
        dict.fromkeys(range(4), 2), states["all_rules"], states["retention"],
        step=step, initial_scatter=initial_scatter)


class TimecourseStatisticsTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(state_space, "DISTANCE_ANGLE_N_PERMUTATIONS", 20)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_later_updates_are_local_and_decay_uses_immediate_predecessor(self):
        data = np.array([[[0., 0.], [10., 3.], [11., 5.], [14., 9.]]])
        trials = np.array([0])
        initial, step = state_space._step_vectors(data, trials, onset=2, step=2)
        np.testing.assert_array_equal(initial, [[10., 3.]])
        np.testing.assert_array_equal(step, [[3., 4.]])
        _, remainder = state_space._step_vectors(data, trials, 2, retention=0.8, step=2)
        np.testing.assert_allclose(remainder, [[5.2, 5.]])
        norms = state_space._step_norm_summary(data, np.array([0]), {0: 2}, 1,
                                               retention=0.8, step=2)
        self.assertAlmostEqual(norms["full_step"], 5.)
        self.assertAlmostEqual(norms["decay_term"], 0.2 * np.linalg.norm([11., 5.]))
        with self.assertRaises(ValueError):
            state_space._step_vectors(data, trials, 2, step=3)

    def test_step_norms_are_summarized_once_per_source(self):
        states = _states()
        with patch.object(state_space, "_step_norm_summary",
                          wraps=state_space._step_norm_summary) as summary:
            _, scatter = _analysis(states)
        self.assertEqual(summary.call_count, 3)  # hidden, mod, eff_mod
        for source in ("mod", "eff_mod"):
            self.assertEqual(scatter[source]["step_norms"], scatter[f"{source}_hebb"]["step_norms"])
            self.assertIsNot(scatter[source]["step_norms"], scatter[f"{source}_hebb"]["step_norms"])
        self.assertEqual(set(scatter["hidden"]["step_norms"]), {"initial", "full_step"})

    def test_timecourse_preserves_first_step_and_fixed_distances(self):
        states = _states()
        original_r, original_scatter = _analysis(states)
        with patch.object(state_space, "_draw_distance_angle_figure"):
            r, scatter = state_space._distance_angle_for_states(states, Path("."), "run", n_steps=5)
        self.assertEqual(r, original_r)
        course = states["timecourse"]
        self.assertEqual(course["n_steps"], 5)
        for key in r:
            np.testing.assert_array_equal(scatter[key]["angles_deg"],
                                          original_scatter[key]["angles_deg"])
            for k, entry in course["by_step"].items():
                later = entry["scatter"][key]
                self.assertEqual(later["step"]["index"], k)
                self.assertEqual(later["pair_rules"], scatter[key]["pair_rules"])
                np.testing.assert_array_equal(later["dists"], scatter[key]["dists"])
        # Independently compute one later task-pair angle by loops, not the vectorized helper.
        data = states["hidden"]
        per_stimulus = []
        for label in (0, 1):
            angles = []
            for a in np.flatnonzero((states["rule_indices"] == 0) & (states["stimulus_labels"] == label)):
                for b in np.flatnonzero((states["rule_indices"] == 1) & (states["stimulus_labels"] == label)):
                    u, v = data[a, 6] - data[a, 5], data[b, 6] - data[b, 5]
                    cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                    angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            per_stimulus.append(np.mean(angles))
        self.assertAlmostEqual(course["by_step"][5]["scatter"]["hidden"]["angles_deg"][0],
                               np.mean(per_stimulus))
        _, saved = state_space._result_entry(state_space._evaluation_tuple(states, r, scatter))
        self.assertIs(saved["timecourse"], course)

    def test_undefined_later_pair_keeps_distances_and_marks_r_undefined(self):
        states = _states()
        _, first = _analysis(states)
        states["hidden"][:4, 3] = states["hidden"][:4, 2]  # task 0 does not move at step 2
        with patch.object(state_space, "cdist", side_effect=AssertionError("distance recomputed")):
            r, scatter = _analysis(states, step=2, initial_scatter=first)
        self.assertTrue(np.isnan(r["hidden"][0]))
        self.assertIn("undefined angles", scatter["hidden"]["regression"]["skipped"])
        self.assertEqual(scatter["hidden"]["pair_rules"], first["hidden"]["pair_rules"])
        np.testing.assert_array_equal(scatter["hidden"]["dists"], first["hidden"]["dists"])
        self.assertEqual(np.isnan(scatter["hidden"]["angles_deg"]).sum(), 3)
        self.assertTrue(np.isfinite(r["mod"][0]))  # a different representation remains usable

    def test_window_respects_each_trial_boundary_and_available_record(self):
        epochs = {"a": {"fix1": (0, 2), "stim1": (2, np.array([7, 8]))},
                  "b": {"fix1": (0, 2), "stim1": (2, None)}}
        state_space._validate_step_window(epochs, 5, aligned_length=7)
        with self.assertRaisesRegex(ValueError, "a:.*boundary"):
            state_space._validate_step_window(epochs, 6, aligned_length=20)
        with self.assertRaisesRegex(ValueError, "too short"):
            state_space._validate_step_window(epochs, 5, aligned_length=6)
        with self.assertRaisesRegex(ValueError, "b:.*boundary"):
            state_space._validate_step_window(
                epochs, 5, aligned_length=20, trial_lengths={"a": 10, "b": [6, 9]})
        with self.assertRaises(ValueError):
            state_space._validate_step_window(epochs, 0)


class ShortForwardTests(unittest.TestCase):
    def test_real_network_short_trace_matches_full_trace_including_last_m_update(self):
        model = _tiny_network(eta=3.)
        inputs = torch.randn(12, 12, 4, generator=torch.Generator().manual_seed(41))
        rules = list(state_space.RULE_MOTIF_MAPPING)[:4]
        epochs = {rule: {"fix1": (0, 2), "stim1": (2, 7)} for rule in rules}
        with torch.no_grad():
            _, _, full = model.iterate_sequence_batch(inputs, run_mode="track_states",
                                                      save_to_cpu=True, detach_saved=True)
        evaluation = {
            "model": model, "state_dict": model.state_dict(), "aname": "run",
            "hidden_size": 6, "l2_info": 1e-4, "task_params": {"rules": rules},
            "trials": [SimpleNamespace(tdim=12) for _ in rules],
            "rule_indices": np.repeat(np.arange(4), 3), "retention": 0.99, "dt_ms": 40,
            "fixed_fixation_steps": 2, "inputs": inputs,
        }
        with patch.object(evaluation_utils, "prepare_evaluation", return_value=evaluation), \
                patch.object(state_space, "device", torch.device("cpu")), \
                patch.object(state_space.helper, "generate_response_stimulus",
                             return_value=(None, np.zeros(12), None, epochs)), \
                patch.object(model, "iterate_sequence_batch", wraps=model.iterate_sequence_batch) as forward, \
                patch.object(model, "compute_acc") as accuracy:
            states = state_space._context_states("run", trace_steps=5)
        accuracy.assert_not_called()
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(forward.call_args.args[0].shape[1], 8)  # 2 context + 5 + extra record
        np.testing.assert_array_equal(states["hidden"], full["hidden1"][:, :7].numpy())
        np.testing.assert_array_equal(states["mod"], full["M1"][:, 1:8].numpy().reshape(12, 7, -1))
        _, step = state_space._step_vectors(states["mod"], np.arange(12), onset=2, step=5)
        expected = full["M1"][:, 7].double() - full["M1"][:, 6].double()
        np.testing.assert_array_equal(step, expected.numpy().reshape(12, -1))
        self.assertTrue(np.isnan(states["accuracy"]))

    def test_skip_pca_batch_preserves_existing_files_and_merges_results(self):
        states = _states()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / f"savednet_{states['aname']}.pt"
            checkpoint.touch()
            out = root / "out"
            out.mkdir()
            sentinel = out / "state_space_structure_results.pkl"
            sentinel.write_bytes(b"unchanged")
            with (out / state_space.RESULTS_NAME).open("wb") as stream:
                pickle.dump({"old": {"kept": True}}, stream)
            with patch.object(state_space.mpf, "list_pt_files", return_value=[checkpoint]), \
                    patch.object(evaluation_utils, "select_target_checkpoints", return_value=[checkpoint]), \
                    patch.object(evaluation_utils, "matches_target_cohort", return_value=True), \
                    patch.object(state_space, "_context_states", return_value=states) as context, \
                    patch.object(state_space, "_save_context_pca") as pca, \
                    patch.object(state_space, "_draw_distance_angle_figure"), \
                    patch.object(state_space, "DISTANCE_ANGLE_N_PERMUTATIONS", 10):
                results = state_space.run_all(out, n_steps=5, skip_pca=True)
                stem, control = state_space.eval_untrained(checkpoint, out, 3, n_steps=5, skip_pca=True)
            pca.assert_not_called()
            context_ms = state_space.FIXED_FIXATION_MS
            self.assertEqual(context.call_args_list[0].kwargs,
                             {"trace_steps": 5, "context_ms": context_ms})
            self.assertEqual(context.call_args.kwargs,
                             {"untrained_init_seed": 3, "trace_steps": 5, "context_ms": context_ms})
            self.assertEqual(sentinel.read_bytes(), b"unchanged")
            self.assertEqual(results["old"], {"kept": True})
            self.assertEqual(results[states["aname"]]["timecourse"]["n_steps"], 5)
            self.assertEqual(control["timecourse"]["n_steps"], 5)
            self.assertTrue(control["untrained"])
            self.assertTrue(stem.endswith("_init3"))

    def test_cli_forwards_options_and_rejects_zero_steps(self):
        with patch.object(state_space, "run_all") as run, \
                patch.object(state_space, "summarize"):
            state_space.main(["--steps", "5", "--skip-pca"])
        run.assert_called_once_with(state_space.STATE_SPACE_DIR, n_steps=5, skip_pca=True,
                                    context_ms=state_space.FIXED_FIXATION_MS)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            state_space._parse_args(["--steps", "0"])


def _entry(n_steps=5, offset=0.):
    r = {key: (0.2 + offset, 1., 0.1) for key, _, _ in state_space.DISTANCE_ANGLE_REPRESENTATIONS}
    return {
        "schema_version": state_space.DISTANCE_ANGLE_SCHEMA_VERSION,
        "m_alignment": state_space.M_ALIGNMENT, "l2_info": 1e-4,
        "timecourse": {"schema_version": 1, "n_steps": n_steps, "dt_ms": 40,
                       "distance_reference": "context_end_step1_pairs",
                       "by_step": {k: {"rval_dict": copy.deepcopy(r)}
                                   for k in range(1, n_steps + 1)}},
    }


class TimecourseFigureTests(unittest.TestCase):
    def _draw(self, trained, controls=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controls_path = root / "untrained.pkl"
            if controls is not None:
                with controls_path.open("wb") as stream:
                    pickle.dump(controls, stream)
            with patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_load_pkl_or_skip", return_value=trained), \
                    patch.object(paper_plot, "STATE_SPACE_UNTRAINED_RVAL_PATH", controls_path), \
                    patch.object(paper_plot, "_load_best_state_space_trial",
                                 side_effect=AssertionError("PCA ranking used")), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_state_space_r_timecourse()
        if save.called:
            fig = save.call_args.args[0]
            self.addCleanup(paper_plot.plt.close, fig)
            self.assertEqual(save.call_args.args[1].name, "multitask_state_space_r_timecourse.png")
            return fig
        return None

    def test_uses_common_horizon_mean_sd_and_one_untrained_source(self):
        name = _states()["aname"]
        other = name.replace("seed202", "seed303")
        controls = {}
        for source in (name, other):
            for seed in (0, 1):
                controls[f"{source}_init{seed}"] = {
                    **_entry(), "untrained": True, "init_seed": seed,
                    "config_source": source, "l2_info": None}
        fig = self._draw({name: _entry(), other: _entry(n_steps=3, offset=0.2)}, controls)
        self.assertEqual(len(fig.axes), 5)
        self.assertIn("n=2", fig.axes[0].get_title())
        ax = fig.axes[2]  # L2=1e-4
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.3] * 3)
        self.assertIn("40 ms", ax.get_xlabel())
        bounds = ax.collections[0].get_paths()[0].vertices[:, 1]
        self.assertAlmostEqual(bounds.min(), 0.3 - np.std([0.2, 0.4], ddof=1))
        self.assertAlmostEqual(bounds.max(), 0.3 + np.std([0.2, 0.4], ddof=1))

    def test_undefined_time_point_does_not_silently_drop_a_network(self):
        entry = _entry()
        entry["timecourse"]["by_step"][2]["rval_dict"]["hidden"] = (np.nan, np.nan, np.nan)
        fig = self._draw({"a": _entry(), "b": entry})
        values = fig.axes[2].lines[0].get_ydata()
        self.assertTrue(np.isnan(values[1]))
        np.testing.assert_allclose(values[[0, 2, 3, 4]], 0.2)

    def test_old_cache_and_mixed_time_steps_are_not_plotted(self):
        self.assertIsNone(self._draw({"old": {"l2_info": 1e-4}}))
        stale = _entry()
        stale.pop("m_alignment")
        self.assertIsNone(self._draw({"stale": stale}))
        other = _entry()
        other["timecourse"]["dt_ms"] = 20
        self.assertIsNone(self._draw({"a": _entry(), "b": other}))


if __name__ == "__main__":
    unittest.main()
