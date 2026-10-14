"""Tests for the rule-cue interpolation sweep of the sibling-family analysis."""

import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import torch

import paper_plot
import sibling_delay_analysis as sibling


RULES = ("delaydm1", "delaydm2")
ALL_RULES = ["fdgo", "delaydm1", "dmcgo", "delaydm2", "dmcnogo"]
RULE_START = 5
N_INPUT = RULE_START + len(ALL_RULES)


def _task_params():
    return {
        "rules": ALL_RULES,
        "randomize_inputs": False,
        "long_delay": "normal",
        "dt": 40,
        "hp": {
            "rules": ALL_RULES,
            "rule_start": RULE_START,
            "task_info": True,
            "batch_size_train": 1,
        },
    }


def _cue_inputs(n_per_rule, time_steps, rules=RULES):
    """Aligned sibling trials: fixation on channel 0, each rule's cue on."""
    channels = [RULE_START + ALL_RULES.index(rule) for rule in rules]
    x = torch.zeros(n_per_rule * len(rules), time_steps, N_INPUT)
    x[:, :, 0] = 1.0
    for rule_i, channel in enumerate(channels):
        rows = slice(rule_i * n_per_rule, (rule_i + 1) * n_per_rule)
        x[rows, :, channel] = 1.0
        x[rows, :, 1] = torch.arange(n_per_rule, dtype=torch.float)[:, None]
    return x, channels


class CueChannelModel:
    """Endpoint M and hidden read the two cue channels and the stimulus."""

    def __init__(self, channels):
        self.channels = list(channels)

    def reset_state(self, B=1):
        self.batch_size = B

    def network_step(self, current_input, run_mode="minimal", seq_idx=None):
        db = None
        if run_mode == "track_states":
            cue = current_input[:, self.channels]        # (B, 2)
            stim = current_input[:, 1:2]                 # (B, 1)
            m = torch.cat([cue, stim], dim=1)            # (B, 3)
            db = {
                "M1": m[:, None, :].repeat(1, 2, 1),      # (B, 2, 3)
                "hidden1": torch.cat([m, m], dim=1),      # (B, 6)
            }
        return None, None, db


class CueInterpInputTests(unittest.TestCase):
    def test_cue_channels_follow_the_full_rule_list(self):
        self.assertEqual(
            sibling.cue_channel_indices(_task_params(), RULES),
            (RULE_START + 1, RULE_START + 3))
        self.assertEqual(
            sibling.cue_channel_indices(_task_params(), ("dmcgo", "dmcnogo")),
            (RULE_START + 2, RULE_START + 4))

    def test_cue_channels_refuse_randomized_or_stripped_inputs(self):
        randomized = _task_params()
        randomized["randomize_inputs"] = True
        with self.assertRaisesRegex(ValueError, "raw"):
            sibling.cue_channel_indices(randomized, RULES)
        stripped = _task_params()
        stripped["hp"]["task_info"] = False
        with self.assertRaisesRegex(ValueError, "task_info"):
            sibling.cue_channel_indices(stripped, RULES)

    def test_alphas_are_evenly_spaced_from_zero_to_one(self):
        np.testing.assert_allclose(
            sibling.cue_interp_alphas(4), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(sibling.cue_interp_alphas(
            sibling.SIBLING_CUE_INTERP_N_ALPHA).size, 11)
        with self.assertRaises(ValueError):
            sibling.cue_interp_alphas(0)

    def test_interpolation_moves_only_the_two_cue_channels(self):
        x, (base, target) = _cue_inputs(2, 3)
        base_trials = x[:2]
        for alpha in (0.0, 0.3, 1.0):
            shifted = sibling.cue_interpolated_inputs(
                base_trials, base, target, alpha)
            self.assertIsInstance(shifted, torch.Tensor)
            np.testing.assert_allclose(shifted[..., base], 1.0 - alpha)
            np.testing.assert_allclose(shifted[..., target], alpha)
            untouched = [c for c in range(N_INPUT) if c not in (base, target)]
            np.testing.assert_array_equal(
                shifted[..., untouched], base_trials[..., untouched])
        # The input itself is never modified, and NumPy arrays work too.
        np.testing.assert_allclose(base_trials[..., target], 0.0)
        as_numpy = sibling.cue_interpolated_inputs(
            base_trials.numpy(), base, target, 0.5)
        self.assertIsInstance(as_numpy, np.ndarray)
        np.testing.assert_allclose(as_numpy[..., base], 0.5)
        with self.assertRaises(ValueError):
            sibling.cue_interpolated_inputs(base_trials, base, base, 0.5)
        with self.assertRaises(ValueError):
            sibling.cue_interpolated_inputs(base_trials, base, target, 1.5)

    def test_cue_check_rejects_a_base_trial_whose_cue_is_off(self):
        x, (base, target) = _cue_inputs(2, 3)
        sibling._check_cue_channels(x[:2], base, target, "delaydm1", "delaydm2")
        with self.assertRaisesRegex(ValueError, "not held at 1"):
            sibling._check_cue_channels(
                x[2:], base, target, "delaydm1", "delaydm2")

    def test_base_rules_cover_both_families(self):
        self.assertEqual(
            sibling.SIBLING_CUE_INTERP_BASE_RULES["delaydm1"],
            ("delaydm1", "delaydm2"))
        self.assertEqual(
            sibling.SIBLING_CUE_INTERP_BASE_RULES["dmcgo"], ("dmcgo",))
        with self.assertRaisesRegex(ValueError, "not in"):
            sibling._cue_interp_base_rules("dmcgo", list(RULES))


class CueInterpArtifactTests(unittest.TestCase):
    def test_artifact_names_belong_to_their_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint_sweep, endpoint_proj = sibling.cue_interp_artifact_paths(
                root, "run", "delaydm1", "long_delay_endpoint")
            gradient_sweep, gradient_proj = sibling.cue_interp_artifact_paths(
                root, "run", "delaydm1", "gradient")
            for path in (endpoint_sweep, endpoint_proj):
                self.assertEqual(
                    sibling._artifact_method(path), "long_delay_endpoint")
            for path in (gradient_sweep, gradient_proj):
                self.assertEqual(sibling._artifact_method(path), "gradient")
            self.assertTrue(sibling.is_sibling_artifact(gradient_sweep))

    def test_full_method_cleanup_also_removes_the_methods_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gradient_sweep, _ = sibling.cue_interp_artifact_paths(
                root, "run", "delaydm1", "gradient")
            endpoint_sweep, _ = sibling.cue_interp_artifact_paths(
                root, "run", "delaydm1", "long_delay_endpoint")
            gradient_sweep.touch()
            endpoint_sweep.touch()
            removed = sibling._clean_stale_sibling_artifacts(
                root, ("delaydm1",), method="gradient")
            self.assertEqual(removed, [gradient_sweep])
            self.assertTrue(endpoint_sweep.exists())

    def test_cli_exposes_the_cue_interp_options(self):
        parser = sibling.build_arg_parser()
        args = parser.parse_args([
            "--seed", "408", "--feature", "L21e4", "--families", "delaydm1",
            "--method", "long_delay_endpoint"])
        self.assertEqual(args.cue_interp_n_alpha,
                         sibling.SIBLING_CUE_INTERP_N_ALPHA)
        args = parser.parse_args([
            "--seed", "408", "--feature", "L21e4", "--families", "dmcgo",
            "--method", "gradient", "--cue-interp-n-alpha", "4"])
        self.assertEqual(args.cue_interp_n_alpha, 4)
        with patch.object(sibling, "run_sibling_analysis") as run:
            sibling.main([
                "--seed", "408", "--feature", "L21e4", "--families", "dmcgo",
                "--method", "gradient"])
        run.assert_called_once_with(
            408, "L21e4", ["dmcgo"], "gradient",
            cue_interp_n_alpha=sibling.SIBLING_CUE_INTERP_N_ALPHA)


class CueInterpEndpointSweepTests(unittest.TestCase):
    def test_endpoint_sweep_runs_each_base_rule_over_alpha(self):
        n_per_rule, time_steps = 4, 7
        x, channels = _cue_inputs(n_per_rule, time_steps)
        trials = [
            SimpleNamespace(
                epochs={"delay1": (2, 6)},
                meta={"stim1": np.arange(n_per_rule) % 2,
                      "stim1_strs": 0.5 + 0.1 * np.arange(n_per_rule)})
            for _ in RULES
        ]
        model = CueChannelModel(channels)
        generated = {}

        def fake_generate(tp, n_batches, **kwargs):
            generated["tp"] = tp
            generated["n_batches"] = n_batches
            generated["kwargs"] = kwargs
            return (x, None, None), (None, trials, None)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sibling.mpn_tasks, "generate_trials_wrap",
                             side_effect=fake_generate):
            out_path = sibling.extract_cue_interp_long_delay_endpoints(
                "run", directory, "delaydm1", RULES, model,
                torch.device("cpu"), _task_params(),
                np.ones((2, 3), dtype=np.float32), layer_index=1,
                n_alpha=2, n_trials_per_rule=n_per_rule)
            with open(out_path, "rb") as stream:
                artifact = pickle.load(stream)

        self.assertEqual(generated["tp"]["long_delay"], "long")
        self.assertEqual(generated["tp"]["rules"], list(RULES))
        self.assertEqual(generated["n_batches"], n_per_rule)
        self.assertTrue(generated["kwargs"]["align_periods"])
        self.assertTrue(generated["kwargs"]["balanced_stimulus_directions"])
        self.assertEqual(
            out_path.name,
            "delaydm1_cue_interp_long_delay_endpoints_run.pkl")
        self.assertEqual(artifact["method"], "long_delay_endpoint")
        self.assertEqual(artifact["kind"], "cue_interp")
        self.assertEqual(artifact["base_rules"], list(RULES))
        np.testing.assert_allclose(artifact["alphas"], [0.0, 0.5, 1.0])
        self.assertEqual(artifact["delay_window"], (2, 6))
        self.assertEqual(artifact["delay_ms"], 160)
        self.assertEqual(
            artifact["cue_channels"],
            {"delaydm1": channels[0], "delaydm2": channels[1]})

        for base_rule, target_rule, base_channel in (
                ("delaydm1", "delaydm2", 0), ("delaydm2", "delaydm1", 1)):
            sweep = artifact["sweeps"][base_rule]
            self.assertEqual(sweep["target_rule"], target_rule)
            wm = sweep["representations"]["fixed_WM"]
            hidden = sweep["representations"]["fixed_hidden"]
            self.assertEqual(wm.shape, (3, n_per_rule, 2, 3))
            self.assertEqual(hidden.shape, (3, n_per_rule, 6))
            # The fake endpoint copies the two cue channels, so the sweep must
            # read (1 - alpha, alpha) in base/target order at every alpha.
            for ai, alpha in enumerate(artifact["alphas"]):
                np.testing.assert_allclose(
                    wm[ai, :, 0, base_channel], 1.0 - alpha, atol=1e-6)
                np.testing.assert_allclose(
                    wm[ai, :, 0, 1 - base_channel], alpha, atol=1e-6)
            # The stimulus channel is untouched across alpha.
            np.testing.assert_allclose(
                wm[:, :, 0, 2], np.tile(np.arange(n_per_rule), (3, 1)))
            np.testing.assert_array_equal(sweep["stim_idx"],
                                          np.arange(n_per_rule) % 2)
            np.testing.assert_allclose(
                sweep["stimulus_magnitude"], 0.5 + 0.1 * np.arange(n_per_rule))

    def test_endpoint_sweep_rejects_a_misordered_rule_batch(self):
        n_per_rule, time_steps = 2, 5
        x, channels = _cue_inputs(n_per_rule, time_steps)
        x = x.flip(0)
        trials = [SimpleNamespace(epochs={"delay1": (1, 4)},
                                  meta={"stim1": np.zeros(n_per_rule, int)})
                  for _ in RULES]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sibling.mpn_tasks, "generate_trials_wrap",
                             return_value=((x, None, None),
                                           (None, trials, None))), \
                self.assertRaisesRegex(ValueError, "task order"):
            sibling.extract_cue_interp_long_delay_endpoints(
                "run", directory, "delaydm1", RULES, CueChannelModel(channels),
                torch.device("cpu"), _task_params(),
                np.ones((2, 3), dtype=np.float32), n_alpha=1,
                n_trials_per_rule=n_per_rule)


class CueInterpGradientSweepTests(unittest.TestCase):
    def test_gradient_sweep_solves_each_alpha_from_the_delay_endpoint(self):
        time_steps, n_points = 6, 3
        delay_window = (2, 5)
        x, channels = _cue_inputs(n_points, time_steps, rules=("dmcgo",))
        batch = x.numpy()
        cue_channels = sibling.cue_channel_indices(
            _task_params(), ("dmcgo", "dmcnogo"))

        def fake_template(cfg, rule, task_seed, n_interp, magnitudes,
                          device="cpu", stim_channels=None, periods=None):
            self.assertEqual(rule, "dmcgo")
            self.assertEqual(periods, ("longdelay",))
            self.assertEqual(n_interp, sibling.SIBLING_FP_N_STIM)
            self.assertEqual(tuple(magnitudes), (1.0,))
            return {
                "batch": batch.copy(),
                "period_win": {"longdelay": delay_window},
                "stim": np.arange(n_points),
                "angles": np.arange(n_points) * 1.0,
                "stimulus_angle": np.arange(n_points) * 1.0,
                "stimulus_magnitude": np.ones(n_points),
                "stim_channels": (1, 2),
                "task_params": _task_params(),
            }

        class FakeNet:
            def __init__(self):
                self.seen_inputs = []

            def iterate_sequence_batch(self, inputs, run_mode="minimal",
                                       save_to_cpu=False, detach_saved=False):
                self.seen_inputs.append(inputs.clone())
                cue = inputs[:, :, cue_channels]                  # (B, T, 2)
                m = torch.cat([cue, inputs[:, :, 1:2]], dim=2)    # (B, T, 3)
                return None, None, {"M1": m[:, :, None, :].repeat(1, 1, 2, 1)}

        solved = []

        def fake_solve(network, init_M, const_input, **kwargs):
            solved.append((np.asarray(init_M).copy(),
                           np.asarray(const_input).copy(), kwargs))
            return np.asarray(init_M) + 1.0, [], np.zeros(init_M.shape[0])

        def fake_views(net, fixed_M, const_input, final_speeds, W, device,
                       rel_tol_undamped):
            n = fixed_M.shape[0]
            return {
                "fixed_WM": fixed_M * W[None],
                "fixed_hidden": fixed_M.reshape(n, -1),
                "fixed_out_cos": np.zeros(n),
                "rel_step": np.full(n, 1e-4),
                "rel_step_undamped": np.full(n, 1e-2),
                "is_fixed": np.array([True] * (n - 1) + [False]),
                "is_fixed_strict": np.array([True] * (n - 1) + [False]),
                "rel_tol": 1e-4,
                "rel_tol_undamped": rel_tol_undamped,
            }

        net = FakeNet()
        W = np.full((2, 3), 2.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sibling, "build_dense_stimulus_template",
                             side_effect=fake_template), \
                patch.object(sibling, "find_modulation_fixed_points",
                             side_effect=fake_solve), \
                patch.object(sibling, "derive_fixed_point_views",
                             side_effect=fake_views):
            out_path = sibling.solve_cue_interp_fixed_points(
                "run", directory, "dmcgo", ("dmcgo", "dmcnogo"), net,
                torch.device("cpu"), {"task_params": {}}, W, n_alpha=2,
                steps=50, lbfgs_steps=5)
            with open(out_path, "rb") as stream:
                artifact = pickle.load(stream)

        self.assertEqual(out_path.name, "dmcgo_cue_interp_fixed_points_run.pkl")
        self.assertEqual(artifact["method"], "gradient")
        self.assertEqual(artifact["base_rules"], ["dmcgo"])
        self.assertEqual(artifact["steps"], 50)
        self.assertEqual(artifact["lbfgs_steps"], 5)
        self.assertEqual(len(solved), 3)
        sweep = artifact["sweeps"]["dmcgo"]
        self.assertEqual(sweep["target_rule"], "dmcnogo")
        self.assertEqual(sweep["period"], delay_window)
        self.assertEqual(sweep["t_seed"], delay_window[1] - 1)
        self.assertEqual(sweep["t_input"], sum(delay_window) // 2)
        self.assertEqual(sweep["representations"]["fixed_WM"].shape,
                         (3, n_points, 2, 3))
        self.assertEqual(sweep["representations"]["fixed_hidden"].shape,
                         (3, n_points, 6))
        self.assertEqual(sweep["is_fixed"].shape, (3, n_points))
        self.assertFalse(sweep["is_fixed"][:, -1].any())
        for ai, alpha in enumerate(artifact["alphas"]):
            init_M, const_input, kwargs = solved[ai]
            # Seeded from the recorded end-of-delay M of the alpha-shifted
            # input, solved under its mid-delay input.
            np.testing.assert_allclose(init_M[:, 0, 0], 1.0 - alpha, atol=1e-6)
            np.testing.assert_allclose(init_M[:, 0, 1], alpha, atol=1e-6)
            np.testing.assert_allclose(
                const_input[:, cue_channels[0]], 1.0 - alpha, atol=1e-6)
            np.testing.assert_allclose(
                const_input[:, cue_channels[1]], alpha, atol=1e-6)
            self.assertEqual(kwargs["steps"], 50)
            self.assertEqual(kwargs["lbfgs_steps"], 5)
            # fixed_WM = (init_M + 1) * W with W = 2.
            np.testing.assert_allclose(
                sweep["representations"]["fixed_WM"][ai, :, 0, 0],
                2.0 * (2.0 - alpha), atol=1e-5)


class CueInterpProjectionTests(unittest.TestCase):
    def _basis(self, scale=1.0):
        identity = np.eye(6, dtype=np.float32)
        return {
            "representations": {
                key: {
                    "mean": np.zeros(6, dtype=np.float32),
                    "components": scale * identity,
                    "explained_variance_ratio": np.arange(1, 7) / 21,
                }
                for key in ("fixed_hidden", "fixed_WM")
            },
            "candidate_selection": {"strategy": "test"},
        }

    def _sweep(self, method, n_alpha=3, n_points=4):
        alphas = np.linspace(0, 1, n_alpha)
        hidden = np.arange(n_alpha * n_points * 6, dtype=np.float32).reshape(
            n_alpha, n_points, 6)
        wm = (100 + np.arange(n_alpha * n_points * 6, dtype=np.float32)
              ).reshape(n_alpha, n_points, 2, 3)
        sweep = {
            "target_rule": "delaydm2",
            "stim_idx": np.arange(n_points) % 2,
            "stimulus_magnitude": np.ones(n_points),
            "representations": {"fixed_hidden": hidden, "fixed_WM": wm},
        }
        if method == "gradient":
            sweep["is_fixed"] = np.ones((n_alpha, n_points), dtype=bool)
            sweep["rel_step_undamped"] = np.zeros((n_alpha, n_points))
        return {
            "kind": "cue_interp",
            "method": method,
            "task_names": list(RULES),
            "alphas": alphas,
            "base_rules": ["delaydm1"],
            "cue_channels": {"delaydm1": 6, "delaydm2": 8},
            "sweeps": {"delaydm1": sweep},
        }

    def test_projection_uses_the_methods_joint_basis(self):
        for method, basis_name in (
                ("long_delay_endpoint",
                 "delaydm1_long_delay_endpoint_delay_trajectory_pca_run.pkl"),
                ("gradient", "delaydm1_delay_trajectory_pca_run.pkl")):
            with self.subTest(method=method), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with (root / basis_name).open("wb") as stream:
                    pickle.dump(self._basis(scale=2.0), stream)
                sweep = self._sweep(method)
                sweep_path, expected_out = sibling.cue_interp_artifact_paths(
                    root, "run", "delaydm1", method)
                with sweep_path.open("wb") as stream:
                    pickle.dump(sweep, stream)

                out_path = sibling.save_cue_interp_pc_projections(
                    "run", root, "delaydm1", RULES, sweep_path, method)

                self.assertEqual(out_path, expected_out)
                with out_path.open("rb") as stream:
                    projected = pickle.load(stream)
                self.assertEqual(projected["kind"], "cue_interp")
                self.assertEqual(projected["method"], method)
                self.assertEqual(projected["basis_scope"], "joint")
                self.assertEqual(projected["pca_selection"],
                                 {"strategy": "test"})
                np.testing.assert_allclose(projected["alphas"], [0, 0.5, 1])
                hidden_out = projected["representations"]["hidden"]
                entry = hidden_out["sweeps"]["delaydm1"]
                self.assertEqual(entry["target_rule"], "delaydm2")
                np.testing.assert_allclose(
                    entry["proj"],
                    2.0 * sweep["sweeps"]["delaydm1"]["representations"]
                    ["fixed_hidden"])
                wm_entry = (projected["representations"]["e_modulation"]
                            ["sweeps"]["delaydm1"])
                self.assertEqual(wm_entry["proj"].shape, (3, 4, 6))
                np.testing.assert_allclose(
                    wm_entry["proj"][0, 0],
                    2.0 * (100 + np.arange(6)))
                self.assertEqual("is_fixed" in entry, method == "gradient")
                np.testing.assert_allclose(
                    hidden_out["explained_variance_ratio"],
                    np.arange(1, 7) / 21)

    def test_projection_rejects_a_sweep_of_the_other_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "delaydm1_delay_trajectory_pca_run.pkl").open(
                    "wb") as stream:
                pickle.dump(self._basis(), stream)
            sweep_path = root / "sweep.pkl"
            with sweep_path.open("wb") as stream:
                pickle.dump(self._sweep("long_delay_endpoint"), stream)
            with self.assertRaisesRegex(ValueError, "not a gradient"):
                sibling.save_cue_interp_pc_projections(
                    "run", root, "delaydm1", RULES, sweep_path, "gradient")


class CueInterpFamilyWiringTests(unittest.TestCase):
    def test_base_analysis_precedes_the_sweep_and_zero_alphas_disable_it(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sibling, "_run_sibling_family_base") as base, \
                patch.object(sibling, "_run_cue_interp") as sweep:
            sibling._run_sibling_family(
                aname="run", save_name="save", save_dir=Path(directory),
                family="dmcgo", method="long_delay_endpoint", model=None,
                device=None, task_params={}, train_params={},
                net_params={}, converted_task_params={}, W_fp=None,
                cue_interp_n_alpha=4)
            base.assert_called_once()
            sweep.assert_called_once()
            self.assertEqual(sweep.call_args.kwargs["rules"],
                             ["dmcgo", "dmcnogo"])
            self.assertEqual(sweep.call_args.kwargs["n_alpha"], 4)
            self.assertEqual(sweep.call_args.kwargs["method"],
                             "long_delay_endpoint")
            base.reset_mock()
            sweep.reset_mock()
            sibling._run_sibling_family(
                aname="run", save_name="save", save_dir=Path(directory),
                family="dmcgo", method="long_delay_endpoint", model=None,
                device=None, task_params={}, train_params={},
                net_params={}, converted_task_params={}, W_fp=None,
                cue_interp_n_alpha=0)
            base.assert_called_once()
            sweep.assert_not_called()


class CueInterpPaperFigureTests(unittest.TestCase):
    def _projection(self, method, base_rules, n_alpha=3, n_points=4):
        alphas = np.linspace(0, 1, n_alpha)
        reps = {}
        for plot_name in ("e_modulation", "hidden"):
            sweeps = {}
            for k, base_rule in enumerate(base_rules):
                proj = np.zeros((n_alpha, n_points, 6), dtype=np.float32)
                proj[:, :, 0] = alphas[:, None] + k
                proj[:, :, 1] = np.arange(n_points)[None, :] / 10
                sweep = {
                    "target_rule": [r for r in RULES if r != base_rule][0]
                    if base_rule in RULES else "dmcnogo",
                    "proj": proj,
                    "stim_idx": np.arange(n_points) % 2,
                    "stimulus_magnitude": np.ones(n_points),
                }
                if method == "gradient":
                    good = np.ones((n_alpha, n_points), dtype=bool)
                    good[-1, -1] = False
                    sweep["is_fixed"] = good
                sweeps[base_rule] = sweep
            reps[plot_name] = {
                "explained_variance_ratio": np.arange(1, 7) / 21,
                "sweeps": sweeps,
            }
        return {
            "kind": "cue_interp",
            "method": method,
            "alphas": alphas,
            "base_rules": list(base_rules),
            "representations": reps,
        }

    def test_figure_names_cover_every_family_method_and_view(self):
        self.assertIn("sibling_cue_interp",
                      paper_plot.FIGURES_BY_MODE["two_in_multiple"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family, base_rules in (("delaydm1", RULES), ("dmcgo", ("dmcgo",))):
                aname = paper_plot._multitask_sibling_aname(family)
                run_dir = root / aname
                run_dir.mkdir(parents=True, exist_ok=True)
                for method in paper_plot._MULTITASK_CUE_INTERP_METHODS:
                    if family == "dmcgo" and method == "gradient":
                        continue   # exercise the skip path
                    path = paper_plot._multitask_cue_interp_projection_path(
                        run_dir, family, aname, method)
                    with path.open("wb") as stream:
                        pickle.dump(self._projection(method, base_rules), stream)
            with patch.object(paper_plot, "TWO_IN_MULTIPLES_DIR", root), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as output:
                paper_plot.plot_multitask_sibling_cue_interp()

        saved = [Path(call.args[1]).name for call in save.call_args_list]
        expected = []
        for family_name, methods in (
                ("delaydm", ("gradient", "long_delay_endpoint")),
                ("dmcgo", ("long_delay_endpoint",))):
            for method in methods:
                for suffix in ("emodulation", "hidden"):
                    for view in ("2d", "3d"):
                        expected.append(
                            f"multitask_{family_name}_cue_interp_{method}_"
                            f"{suffix}_{view}.png")
        self.assertEqual(saved, expected)
        skipped = [call.args[0] for call in output.call_args_list
                   if "Skipped" in str(call.args[0])]
        self.assertTrue(any("dmcgo_cue_interp_delay_pc_projections" in text
                            for text in skipped))
        # Every saved figure is a closed matplotlib figure with the right
        # column count: two base rules for DelayDM, one for DMC.
        for call in save.call_args_list:
            fig = call.args[0]
            name = Path(call.args[1]).name
            n_axes = len(fig.axes)
            if "_2d" in name:
                self.assertEqual(n_axes, 4 if "delaydm" in name else 2)
            else:
                self.assertEqual(n_axes, 2 if "delaydm" in name else 1)
            paper_plot.plt.close(fig)

    def test_endpoint_figure_overlays_the_cached_cue_sweep(self):
        points = np.arange(48, dtype=float).reshape(8, 6) / 100
        base_entry = {
            "proj": points,
            "trajectory_proj": np.stack([2 * points, points], axis=1),
            "task_idx": np.repeat([0, 1], 4),
            "stim_idx": np.tile(np.arange(4), 2),
        }
        base_data = {"representations": {"e_modulation": base_entry,
                                         "hidden": base_entry}}
        n_alpha, n_points = 3, 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aname = paper_plot._multitask_sibling_aname("delaydm1")
            run_dir = root / aname
            run_dir.mkdir(parents=True)
            sweep_path = paper_plot._multitask_cue_interp_projection_path(
                run_dir, "delaydm1", aname, "long_delay_endpoint")
            with sweep_path.open("wb") as stream:
                pickle.dump(self._projection(
                    "long_delay_endpoint", RULES, n_alpha, n_points), stream)
            with patch.object(paper_plot, "TWO_IN_MULTIPLES_DIR", root), \
                    patch.object(paper_plot, "SHOW_LEGEND", True), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_best_sibling_endpoint_pc_pair",
                                 return_value=((3, 4), 1.0, "test score")), \
                    patch.object(paper_plot, "_save_fig") as save:
                sweep = paper_plot._cue_interp_endpoint_overlay(
                    "delaydm1", "e_modulation")
                self.assertIsNotNone(sweep)
                self.assertTrue(
                    paper_plot._plot_multitask_long_delay_endpoint_representation(
                        base_data, "delaydm1", RULES, "e_modulation",
                        "emodulation", cue_sweep=sweep))
                # Without the cache the loader yields None and the figure
                # still renders, overlay-free.
                (run_dir / sweep_path.name).unlink()
                self.assertIsNone(paper_plot._cue_interp_endpoint_overlay(
                    "delaydm1", "hidden"))
                self.assertTrue(
                    paper_plot._plot_multitask_long_delay_endpoint_representation(
                        base_data, "delaydm1", RULES, "hidden", "hidden"))

        figures = {Path(call.args[1]).name: call.args[0]
                   for call in save.call_args_list}
        self.assertEqual(set(figures), {
            "multitask_delaydm_long_delay_endpoint_legend.png",
            "multitask_delaydm_long_delay_endpoint_emodulation.png",
            "multitask_delaydm_long_delay_endpoint_hidden.png"})
        overlaid = figures["multitask_delaydm_long_delay_endpoint_emodulation.png"]
        plain = figures["multitask_delaydm_long_delay_endpoint_hidden.png"]
        for fig in (overlaid, plain):
            self.addCleanup(paper_plot.plt.close, fig)
        for ax in overlaid.axes:
            dotted = [line for line in ax.lines if line.get_linestyle() == ":"]
            # One dotted alpha path per sweep trial: two base rules x 4 trials.
            self.assertEqual(len(dotted), 2 * n_points)
            # Every alpha = 1 end is a hollow marker, drawn above the endpoints.
            hollow = [coll for coll in ax.collections
                      if coll.get_zorder() == 4]
            self.assertEqual(len(hollow), 2 * n_points)
        for ax in plain.axes:
            self.assertFalse(
                [line for line in ax.lines if line.get_linestyle() == ":"])
        legend = figures["multitask_delaydm_long_delay_endpoint_legend.png"]
        self.addCleanup(paper_plot.plt.close, legend)
        labels = [text.get_text() for text in legend.legends[0].get_texts()]
        self.assertEqual(len(labels), 4)
        self.assertTrue(any("IntegrationModality1 stimulus" in label
                            for label in labels))
        extra = [call.kwargs.get("extra", "") for call in save.call_args_list]
        self.assertTrue(any("cue sweep overlaid: 3 alphas" in text
                            for text in extra))
        self.assertTrue(any("no cue sweep" in text for text in extra))

    def test_endpoint_plane_comes_from_the_base_endpoint_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aname = "run"
            base_entry = {
                "proj": np.random.RandomState(0).normal(size=(8, 6)),
                "task_idx": np.repeat([0, 1], 4),
                "stim_idx": np.tile([0, 1, 2, 3], 2),
            }
            with (root / f"delaydm1_long_delay_endpoint_pc_projections_"
                         f"{aname}.pkl").open("wb") as stream:
                pickle.dump({"representations": {"hidden": base_entry}}, stream)
            with patch.object(paper_plot, "_best_sibling_endpoint_pc_pair",
                              return_value=((3, 5), 0.4, "direction metric")):
                pair, note = paper_plot._multitask_cue_interp_plane(
                    "delaydm1", "hidden", "long_delay_endpoint", root, aname)
            self.assertEqual(pair, (3, 5))
            self.assertIn("direction metric", note)
            pair, note = paper_plot._multitask_cue_interp_plane(
                "delaydm1", "e_modulation", "long_delay_endpoint", root, aname)
            self.assertEqual(
                pair, paper_plot._MULTITASK_SIBLING_ENDPOINT_DEFAULT_PC_PLANE)
            self.assertEqual(note, "default")
            pair, note = paper_plot._multitask_cue_interp_plane(
                "dmcgo", "hidden", "gradient", root, aname)
            self.assertEqual(
                pair, paper_plot._MULTITASK_SIBLING_PC_PLANES["dmcgo"]["hidden"])
            self.assertEqual(note, "configured")


if __name__ == "__main__":
    unittest.main()
