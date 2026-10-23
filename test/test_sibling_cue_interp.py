"""Tests for the rule-cue interpolation sweep of the sibling-family analysis."""

import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import numpy as np
import torch

import mpn_tasks
import paper_plot
import sibling_delay_analysis as sibling


# Pin the sibling runs: with DELAYDM_ANAME / DMCGO_ANAME = None paper_plot
# ranks every run cached under two_in_multiples/, which these tests must not
# read (test_sibling_run_selection covers that selection).
PINNED_SIBLING_RUN = "everything_seed921_L21e4+hidden300+batch128+angle"


def setUpModule():
    for name in ("DELAYDM_ANAME", "DMCGO_ANAME"):
        patcher = patch.object(paper_plot, name, PINNED_SIBLING_RUN)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


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

    def test_cli_has_no_cue_interp_options(self):
        parser = sibling.build_arg_parser()
        option_strings = {
            option for action in parser._actions
            for option in action.option_strings}
        self.assertFalse(
            {option for option in option_strings if "cue" in option})
        self.assertEqual(sibling.SIBLING_CUE_INTERP_N_ALPHA, 10)
        with patch.object(sibling, "run_sibling_analysis") as run:
            sibling.main([
                "--seed", "408", "--feature", "L21e4", "--families", "dmcgo",
                "--method", "gradient"])
        run.assert_called_once_with(408, "L21e4", ["dmcgo"], "gradient")


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
                             side_effect=fake_generate), \
                patch.object(sibling, "_modulation_leak", return_value=0.01):
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
            # The fake's state is constant on a constant input: every alpha is
            # already a fixed point at the end of the delay, so none settles.
            self.assertEqual(sweep["is_fixed"].shape, (3, n_per_rule))
            self.assertTrue(sweep["is_fixed"].all())
            np.testing.assert_array_equal(sweep["settle_ms"], [0, 0, 0])

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
                self.assertEqual(projected["pca_source"], str(root / basis_name))
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

    def test_dmc_projection_saves_the_base_trials_memory_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "dmcgo_delay_trajectory_pca_run.pkl").open(
                    "wb") as stream:
                pickle.dump(self._basis(), stream)
            sweep = self._sweep("gradient", n_points=8)
            sweep["task_names"] = ["dmcgo", "dmcnogo"]
            sweep["base_rules"] = ["dmcgo"]
            sweep["sweeps"]["dmcgo"] = sweep["sweeps"].pop("delaydm1")
            sweep["sweeps"]["dmcgo"]["target_rule"] = "dmcnogo"
            sweep["sweeps"]["dmcgo"]["stim_idx"] = np.arange(8)
            sweep_path, _ = sibling.cue_interp_artifact_paths(
                root, "run", "dmcgo", "gradient")
            with sweep_path.open("wb") as stream:
                pickle.dump(sweep, stream)

            out_path = sibling.save_cue_interp_pc_projections(
                "run", root, "dmcgo", ("dmcgo", "dmcnogo"), sweep_path,
                "gradient")
            with out_path.open("rb") as stream:
                projected = pickle.load(stream)
        entry = projected["representations"]["hidden"]["sweeps"]["dmcgo"]
        # DMCGo stimuli 0-3 (category 1) respond to category 1 (label 1), 4-7
        # respond to category 2 (label 0).
        np.testing.assert_array_equal(entry["group_labels"],
                                      [1, 1, 1, 1, 0, 0, 0, 0])

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
    def test_base_analysis_precedes_the_sweep(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sibling, "_run_sibling_family_base",
                             side_effect=lambda **kw: calls.append("base")), \
                patch.object(sibling, "_run_cue_interp",
                             side_effect=lambda **kw: calls.append(
                                 ("sweep", kw["method"], kw["rules"]))):
            sibling._run_sibling_family(
                aname="run", save_name="save", save_dir=Path(directory),
                family="dmcgo", method="long_delay_endpoint", model=None,
                device=None, task_params={}, train_params={},
                net_params={}, converted_task_params={}, W_fp=None)
        self.assertEqual(calls, [
            "base", ("sweep", "long_delay_endpoint", ["dmcgo", "dmcnogo"])])

    def test_sweep_runner_uses_the_fixed_grid_and_budgets(self):
        with patch.object(sibling, "extract_cue_interp_long_delay_endpoints",
                          return_value="sweep.pkl") as endpoints, \
                patch.object(sibling, "solve_cue_interp_fixed_points",
                             return_value="sweep.pkl") as gradient, \
                patch.object(sibling, "save_cue_interp_pc_projections") as save:
            for method in ("long_delay_endpoint", "gradient"):
                sibling._run_cue_interp(
                    aname="run", save_dir=Path("out"), family="delaydm1",
                    rules=list(RULES), method=method, model=None, device=None,
                    cfg_fp={}, converted_task_params={}, W_fp=None)
        self.assertEqual(endpoints.call_args.kwargs["n_alpha"],
                         sibling.SIBLING_CUE_INTERP_N_ALPHA)
        self.assertEqual(gradient.call_args.kwargs["n_alpha"],
                         sibling.SIBLING_CUE_INTERP_N_ALPHA)
        self.assertEqual(gradient.call_args.kwargs["steps"],
                         sibling.SIBLING_FIXED_POINT_STEPS)
        self.assertEqual(gradient.call_args.kwargs["lbfgs_steps"],
                         sibling.SIBLING_CUE_INTERP_LBFGS_STEPS)
        self.assertEqual(
            [call.args[-1] for call in save.call_args_list],
            ["long_delay_endpoint", "gradient"])


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
                aname = paper_plot._multitask_sibling_aname(family, "e_modulation", "sweep")
                run_dir = root / aname
                run_dir.mkdir(parents=True, exist_ok=True)
                for method in paper_plot._MULTITASK_CUE_INTERP_METHODS:
                    if family == "dmcgo" and method == "gradient":
                        continue   # exercise the skip path
                    path = paper_plot._multitask_cue_interp_projection_path(
                        run_dir, family, aname, method)
                    projection = self._projection(method, base_rules)
                    if family == "dmcgo":
                        # Stimuli from both halves, so both memory clusters
                        # appear in the legend.
                        for rep in projection["representations"].values():
                            for sweep in rep["sweeps"].values():
                                sweep["stim_idx"] = np.array([0, 2, 4, 6])
                    with path.open("wb") as stream:
                        pickle.dump(projection, stream)
            with patch.object(paper_plot, "TWO_IN_MULTIPLES_DIR", root), \
                    patch.object(paper_plot, "SHOW_LEGEND", True), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as output:
                paper_plot.plot_multitask_sibling_cue_interp()

        saved = [Path(call.args[1]).name for call in save.call_args_list]
        expected = []
        # Per representation: the two trial views against alpha. A legend
        # figure follows each (family, method) that needs one (the gradient
        # convergence markers for DelayDM, the cluster lines for DMC).
        for family_name, methods in (
                ("delaydm", ("gradient", "long_delay_endpoint")),
                ("dmcgo", ("long_delay_endpoint",))):
            for method in methods:
                for suffix in ("emodulation", "hidden"):
                    for view in ("2d", "3d"):
                        expected.append(
                            f"multitask_{family_name}_cue_interp_{method}_"
                            f"{suffix}_{view}.png")
                if family_name == "dmcgo" or method == "gradient":
                    expected.append(
                        f"multitask_{family_name}_cue_interp_{method}_legend.png")
        self.assertEqual(saved, expected)
        legends = {Path(call.args[1]).name: call.args[0]
                   for call in save.call_args_list
                   if Path(call.args[1]).stem.endswith("_legend")}
        dmc_legend = legends["multitask_dmcgo_cue_interp_long_delay_endpoint_legend.png"]
        self.assertEqual(
            [text.get_text() for text in dmc_legend.legends[0].get_texts()],
            [r"Respond to category 1 ($\alpha$ = 0)",
             r"Respond to category 2 ($\alpha$ = 0)"])
        gradient_legend = legends["multitask_delaydm_cue_interp_gradient_legend.png"]
        self.assertEqual(
            [text.get_text() for text in gradient_legend.legends[0].get_texts()],
            ["Converged", "Not converged"])
        for fig in legends.values():
            paper_plot.plt.close(fig)
        skipped = [call.args[0] for call in output.call_args_list
                   if "Skipped" in str(call.args[0])]
        self.assertTrue(any("dmcgo_cue_interp_delay_pc_projections" in text
                            for text in skipped))
        # Every saved panel figure has the right column count: two base rules
        # for DelayDM, one for DMC.
        for call in save.call_args_list:
            fig = call.args[0]
            name = Path(call.args[1]).name
            if name.endswith("_legend.png"):
                continue
            n_axes = len(fig.axes)
            if name.endswith("_2d.png"):
                self.assertEqual(n_axes, 4 if "delaydm" in name else 2)
            else:
                self.assertEqual(n_axes, 2 if "delaydm" in name else 1)
            paper_plot.plt.close(fig)

    def test_dmc_endpoint_overlay_colors_each_path_by_its_alpha0_cluster(self):
        # DMCGo stimuli 0 and 2 form response cluster 1, 4 and 6 cluster 0;
        # the cache has no saved labels, so they are derived from the stimuli.
        points = np.arange(48, dtype=float).reshape(8, 6) / 100
        task_idx = np.repeat([0, 1], 4)
        stim_idx = np.tile([0, 2, 4, 6], 2)
        base_entry = {
            "proj": points,
            "trajectory_proj": np.stack([2 * points, points], axis=1),
            "task_idx": task_idx, "stim_idx": stim_idx,
            "group_labels": ((task_idx == 0) == (stim_idx < 4)).astype(int),
        }
        sweep = self._projection("long_delay_endpoint", ("dmcgo",))[
            "representations"]["hidden"]["sweeps"]["dmcgo"]
        sweep["stim_idx"] = np.array([0, 2, 4, 6])
        cue_sweep = (np.linspace(0, 1, 3), {"dmcgo": sweep})
        with patch.object(paper_plot, "SHOW_LEGEND", False), \
                patch.object(paper_plot, "_best_sibling_endpoint_pc_pair",
                             return_value=((3, 4), 1.0, "test score")), \
                patch.object(paper_plot, "_save_fig") as save:
            self.assertTrue(
                paper_plot._plot_multitask_long_delay_endpoint_representation(
                    {"representations": {"hidden": base_entry}}, "dmcgo",
                    ("dmcgo", "dmcnogo"), "hidden", "hidden", cue_sweep=cue_sweep))
        fig = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, fig)
        expected = [paper_plot.mpl.colors.to_hex(
                        paper_plot._CUE_INTERP_CLUSTER_COLORS[label])
                    for label in (1, 1, 0, 0)]
        for ax in fig.axes:
            dotted = [line for line in ax.lines if line.get_linestyle() == ":"]
            self.assertEqual(
                [paper_plot.mpl.colors.to_hex(line.get_color()) for line in dotted],
                expected)

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
            aname = paper_plot._multitask_sibling_aname("delaydm1", "e_modulation")
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
        # The overlaid copy and its legend carry the `_cue_interp` suffix, so
        # the plain figure names are untouched.
        self.assertEqual(set(figures), {
            "multitask_delaydm_long_delay_endpoint_legend_cue_interp.png",
            "multitask_delaydm_long_delay_endpoint_emodulation_cue_interp.png",
            "multitask_delaydm_long_delay_endpoint_hidden.png"})
        overlaid = figures[
            "multitask_delaydm_long_delay_endpoint_emodulation_cue_interp.png"]
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
        legend = figures[
            "multitask_delaydm_long_delay_endpoint_legend_cue_interp.png"]
        self.addCleanup(paper_plot.plt.close, legend)
        labels = [text.get_text() for text in legend.legends[0].get_texts()]
        self.assertEqual(len(labels), 4)
        self.assertTrue(any("IntegrationModality1 stimulus" in label
                            for label in labels))
        extra = [call.kwargs.get("extra", "") for call in save.call_args_list]
        self.assertTrue(any("cue sweep overlaid: 3 alphas" in text
                            for text in extra))

    def test_driver_saves_the_plain_figure_and_an_overlaid_copy(self):
        base_data = {"representations": {}}
        marker = object()

        def load(path, hint=""):
            return base_data if "long_delay_endpoint" in path.name else None

        def overlay(family, plot_name):
            return marker if plot_name == "e_modulation" else None

        with patch.object(paper_plot, "_load_pkl_or_skip", side_effect=load), \
                patch.object(paper_plot, "_cue_interp_endpoint_overlay",
                             side_effect=overlay), \
                patch.object(
                    paper_plot,
                    "_plot_multitask_long_delay_endpoint_representation",
                ) as render:
            paper_plot.plot_multitask_delaydm_fixed_point_geometry()

        calls = [(call.args[1], call.args[3], call.kwargs.get("cue_sweep"))
                 for call in render.call_args_list]
        # Per family: plain e_modulation, overlaid e_modulation, plain hidden
        # (no sweep for hidden, so no second copy).
        self.assertEqual(calls, [
            ("delaydm1", "e_modulation", None),
            ("delaydm1", "e_modulation", marker),
            ("delaydm1", "hidden", None),
            ("dmcgo", "e_modulation", None),
            ("dmcgo", "e_modulation", marker),
            ("dmcgo", "hidden", None),
        ])

    def test_dmc_sweep_lines_are_colored_by_memory_cluster(self):
        n_points = 8
        sweep = {"target_rule": "dmcnogo", "stim_idx": np.arange(n_points),
                 "proj": np.zeros((3, n_points, 6))}
        rules = ("dmcgo", "dmcnogo")
        # Derived from the base rule and stimulus when the cache has no labels.
        np.testing.assert_array_equal(
            paper_plot._cue_interp_sweep_groups("dmcgo", rules, "dmcgo", sweep),
            [1, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            paper_plot._cue_interp_sweep_groups(
                "dmcgo", rules, "dmcnogo", sweep),
            [0, 0, 0, 0, 1, 1, 1, 1])
        # Saved labels win; DelayDM has no clusters.
        sweep["group_labels"] = np.ones(n_points, dtype=int)
        np.testing.assert_array_equal(
            paper_plot._cue_interp_sweep_groups("dmcgo", rules, "dmcgo", sweep),
            np.ones(n_points))
        self.assertIsNone(paper_plot._cue_interp_sweep_groups(
            "delaydm1", RULES, "delaydm1", {"stim_idx": np.arange(4)}))

        projection = self._projection("long_delay_endpoint", ("dmcgo",),
                                      n_points=n_points)
        entry = projection["representations"]["e_modulation"]
        entry["sweeps"]["dmcgo"]["stim_idx"] = np.arange(n_points)
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "SHOW_LEGEND", True), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot._render_multitask_cue_interp_2d(
                entry, projection["alphas"], "dmcgo", "long_delay_endpoint",
                (1, 2), Path("out.png"), "test")
        fig = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, fig)
        for ax in fig.axes:
            colors = {paper_plot.mpl.colors.to_hex(line.get_color())
                      for line in ax.lines}
            self.assertEqual(colors, {
                paper_plot.mpl.colors.to_hex(c)
                for c in paper_plot._CUE_INTERP_CLUSTER_COLORS.values()})
            # The legend lives in its own figure, not in the panels.
            self.assertIsNone(ax.get_legend())
        self.assertIn("alpha = 0 cluster", save.call_args.kwargs["extra"])
        handles = paper_plot._cue_interp_legend_handles(
            "dmcgo", ("dmcgo", "dmcnogo"), "long_delay_endpoint", entry)
        self.assertEqual([h.get_label() for h in handles],
                         [r"Respond to category 1 ($\alpha$ = 0)",
                          r"Respond to category 2 ($\alpha$ = 0)"])
        self.assertEqual(
            len(paper_plot._cue_interp_legend_handles(
                "dmcgo", ("dmcgo", "dmcnogo"), "gradient", entry)), 4)

    def test_dmc_category_colorwheel_follows_the_task_categories(self):
        n_stim = paper_plot._SIBLING_N_STIM
        locations = paper_plot._DMC_STIMULUS_LOCATIONS
        # Location k is direction k, and the task's category (location < pi)
        # is the half split the DMC cluster labels use (direction < 4).
        np.testing.assert_array_equal(
            mpn_tasks.get_clostest_pref_idx(
                locations, 2 * np.pi * np.arange(n_stim) / n_stim),
            np.arange(n_stim))
        np.testing.assert_array_equal(locations < np.pi,
                                      np.arange(n_stim) < n_stim // 2)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(paper_plot, "OUT_DIR", Path(directory)), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot.plot_onetask_stimulus_colorwheel()
        for call in save.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        # One call saves the plain wheel and its DMC category companion.
        self.assertEqual(
            [Path(call.args[1]).name for call in save.call_args_list],
            ["onetask_stimulus_colorwheel.png",
             "onetask_stimulus_colorwheel_dmc_category.png"])
        ax = save.call_args_list[1].args[0].axes[0]
        to_hex = paper_plot.mpl.colors.to_hex
        colors = {label: to_hex(color) for label, color
                  in paper_plot._CUE_INTERP_CLUSTER_COLORS.items()}
        # The upper arc is category 1 in its cluster color, the lower one
        # category 2, each labelled in that color.
        arcs = [line for line in ax.lines if line.get_linestyle() == "-"]
        self.assertEqual(len(arcs), 2)
        for line in arcs:
            label = 1 if np.mean(line.get_ydata()) > 0 else 0
            self.assertEqual(to_hex(line.get_color()), colors[label])
        self.assertEqual({text.get_text(): to_hex(text.get_color())
                          for text in ax.texts},
                         {"Category 1": colors[1], "Category 2": colors[0]})
        # Eight dots, none on the boundary.
        dots = np.vstack([coll.get_offsets() for coll in ax.collections])
        self.assertEqual(len(dots), n_stim)
        self.assertTrue(np.all(np.abs(dots[:, 1]) > 0.1))

        # DelayDM has no memory clusters: its lines take the stimulus
        # direction's rainbow color (stim_idx 0 and 1 in the fixture) and
        # there is no cluster legend.
        delaydm = self._projection("long_delay_endpoint", RULES)
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "SHOW_LEGEND", True), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot._render_multitask_cue_interp_2d(
                delaydm["representations"]["hidden"], delaydm["alphas"],
                "delaydm1", "long_delay_endpoint", (1, 2), Path("out.png"),
                "test")
        fig = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, fig)
        rainbow = {paper_plot.mpl.colors.to_hex(paper_plot.stim_color(k, 8))
                   for k in (0, 1)}
        for ax in fig.axes:
            self.assertEqual(
                {paper_plot.mpl.colors.to_hex(line.get_color())
                 for line in ax.lines}, rainbow)
        self.assertIsNone(fig.axes[0].get_legend())
        self.assertIn("stimulus direction", save.call_args.kwargs["extra"])

    def test_trial_view_panels_are_square_with_alpha_ticks_and_tight_limits(self):
        delaydm = self._projection("long_delay_endpoint", RULES, n_points=4)
        entry = delaydm["representations"]["hidden"]
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "SHOW_LEGEND", True), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot._render_multitask_cue_interp_2d(
                entry, delaydm["alphas"], "delaydm1", "long_delay_endpoint",
                (1, 2), Path("out.png"), "test")
        fig = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, fig)
        # Every panel, not only the bottom row, labels the sweep ends, and
        # every panel is square.
        for ax in fig.axes:
            labels = [text.get_text() for text in ax.get_xticklabels()]
            self.assertEqual(labels, ["0", "1"])
            self.assertTrue(all(text.get_visible()
                                for text in ax.get_xticklabels()))
            self.assertEqual(ax.get_box_aspect(), 1)
        # y limits hug each panel's own lines, 8% below and more room above:
        # PC1 spans alpha in [0, 1] for the first base and [1, 2] for the second.
        pad = paper_plot._MULTITASK_CUE_INTERP_Y_PADDING
        top = paper_plot._MULTITASK_CUE_INTERP_Y_PADDING_TOP
        self.assertGreater(top, pad)
        np.testing.assert_allclose(fig.axes[0].get_ylim(), (-pad, 1 + top))
        np.testing.assert_allclose(fig.axes[1].get_ylim(), (1 - pad, 2 + top))
        # DelayDM endpoints need no legend at all...
        self.assertEqual(paper_plot._cue_interp_legend_handles(
            "delaydm1", RULES, "long_delay_endpoint", entry), [])
        # ...unless one stayed above the residual criterion and is drawn hollow.
        settled = np.ones((3, 4), dtype=bool)
        settled[-1, 0] = False
        entry["sweeps"]["delaydm1"]["is_fixed"] = settled
        self.assertEqual(
            [handle.get_label() for handle in paper_plot._cue_interp_legend_handles(
                "delaydm1", RULES, "long_delay_endpoint", entry)],
            ["Converged", "Not converged"])

    def test_one_trial_per_stimulus_matches_magnitudes_across_directions(self):
        stim = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        magnitude = np.array([0.7, 1.1, 1.3, 0.9, 1.0, 1.4, 0.8, 1.2, 1.5])
        indices, target = paper_plot._one_trial_per_stimulus(stim, magnitude)
        # 1.1 is the magnitude every direction can match within 0.1
        # (1.1, 1.0, 1.2); 1.0 and 1.2 would leave one direction 0.2 away.
        self.assertEqual(target, 1.1)
        np.testing.assert_array_equal(indices, [1, 4, 7])
        np.testing.assert_array_equal(stim[indices], [0, 1, 2])
        # Constant magnitudes (DMC) deduplicate to the first trial per direction.
        indices, target = paper_plot._one_trial_per_stimulus(
            np.array([3, 0, 3, 0]), np.ones(4))
        np.testing.assert_array_equal(indices, [1, 0])
        self.assertEqual(target, 1.0)
        with self.assertRaises(ValueError):
            paper_plot._one_trial_per_stimulus(np.array([0, 1]), np.ones(3))

    def test_one_trial_per_stimulus_picks_only_allowed_trials(self):
        stim = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        magnitude = np.array([0.7, 1.1, 1.35, 0.9, 1.0, 1.4, 0.8, 1.2, 1.6])
        # Direction 1 may only use 1.4, so the target moves to 1.35, which
        # every direction matches within 0.15 (1.35, 1.4, 1.2).
        allowed = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        indices, target = paper_plot._one_trial_per_stimulus(
            stim, magnitude, allowed)
        self.assertEqual(target, 1.35)
        np.testing.assert_array_equal(indices, [2, 5, 7])
        with self.assertRaises(ValueError):
            paper_plot._one_trial_per_stimulus(
                stim, magnitude, np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool))
        with self.assertRaises(ValueError):
            paper_plot._one_trial_per_stimulus(stim, magnitude, np.ones(3, bool))

    def test_attractor_switch_score_ignores_the_common_motion(self):
        # Every trial drifts one unit per alpha along PC1 (the cue's common
        # motion); PC2 holds the trials 0-3 apart, and trial 2 hops 0.5 up at
        # the third step; PC3 is flat.
        n_alpha = 5
        proj = np.zeros((n_alpha, 4, 3))
        proj[:, :, 0] = np.arange(n_alpha)[:, None]
        proj[:, :, 1] = np.arange(4)[None, :]
        proj[3:, 2, 1] += 0.5
        scores = paper_plot._cue_interp_attractor_switch_scores(proj)
        np.testing.assert_allclose(scores, [0.0, 0.0, 0.5 / 3.0, 0.0])
        with self.assertRaises(ValueError):
            paper_plot._cue_interp_attractor_switch_scores(proj[:1])

    @staticmethod
    def _hopping_sweep():
        """Six trials, two per direction, on one PC spanning 0-10 over four
        alphas: trial 0 hops 2 (score 0.2), trial 2 hops 3 (0.3) and trial 3
        hops 1.5 (0.15), each at its own step so the common motion stays
        zero; the rest stay put. By magnitude alone trials 0, 2 and 4 would
        be drawn."""
        proj = np.tile(np.arange(0.0, 12.0, 2.0)[None, :, None], (4, 1, 1))
        proj[1:, 0, 0] += 2.0
        proj[2:, 2, 0] += 3.0
        proj[3:, 3, 0] += 1.5
        return {"target_rule": "delaydm2", "proj": proj,
                "stim_idx": np.array([0, 0, 1, 1, 2, 2]),
                "stimulus_magnitude": np.array([1.0, 0.9, 1.0, 1.2, 1.0, 1.0]),
                "is_fixed": np.ones((4, 6), dtype=bool)}

    def test_attractor_stable_trials_keep_the_smoothest_when_none_stays(self):
        allowed, fallback = paper_plot._attractor_stable_trials(
            self._hopping_sweep())
        # Direction 1 has no trial under the threshold and keeps trial 3,
        # the smaller hop.
        np.testing.assert_array_equal(allowed, [0, 1, 0, 1, 1, 1])
        self.assertEqual(fallback, [1])

    def test_avoid_attractor_switches_option_redraws_the_delaydm_view(self):
        sweep = self._hopping_sweep()
        with patch.object(paper_plot, "SIBLING_ONE_TRIAL_PER_STIMULUS", True):
            plain = paper_plot._cue_interp_sweep_view(sweep, "delaydm1")
        np.testing.assert_array_equal(plain["proj"][0, :, 0], [0.0, 4.0, 8.0])
        self.assertNotIn("attractor_switch_fallback", plain)
        # The option implies the one-trial view on its own.
        with patch.object(paper_plot, "SIBLING_AVOID_ATTRACTOR_SWITCHES", True):
            view = paper_plot._cue_interp_sweep_view(sweep, "delaydm1")
            dmc = paper_plot._cue_interp_sweep_view(sweep, "dmcgo")
        # Trials 1, 3 and 4: the allowed trials matched at magnitude 1.0.
        np.testing.assert_array_equal(view["proj"][0, :, 0], [2.0, 6.0, 8.0])
        np.testing.assert_array_equal(view["stimulus_magnitude"], [0.9, 1.2, 1.0])
        self.assertEqual(view["one_trial_target_magnitude"], 1.0)
        self.assertEqual(view["attractor_switch_fallback"], [1])
        self.assertEqual(view["is_fixed"].shape, (4, 3))
        self.assertEqual(
            paper_plot._cue_interp_view_note({"delaydm1": view}),
            "; one trial per stimulus, magnitude ≈ 1; trials switching "
            "attractor avoided (none stays, smoothest drawn: delaydm1 "
            "direction 1)")
        # DMC is never filtered: its large step is the cluster swap.
        np.testing.assert_array_equal(dmc["proj"], plain["proj"])
        self.assertNotIn("attractor_switch_fallback", dmc)

    def test_one_trial_option_subsets_the_drawn_sweeps(self):
        projection = self._projection("gradient", RULES, n_points=4)
        entry = projection["representations"]["hidden"]
        sweep = entry["sweeps"]["delaydm1"]
        sweep["stimulus_magnitude"] = np.array([0.8, 1.2, 1.0, 1.0])
        # Off by default: the sweep is drawn as is.
        self.assertIs(paper_plot._cue_interp_sweep_view(sweep, "delaydm1"), sweep)
        with patch.object(paper_plot, "SIBLING_ONE_TRIAL_PER_STIMULUS", True):
            view = paper_plot._cue_interp_sweep_view(sweep, "delaydm1")
            # stim 0 -> trial 2 (magnitude 1.0), stim 1 -> trial 3 (1.0).
            self.assertEqual(view["proj"].shape, (3, 2, 6))
            np.testing.assert_array_equal(view["stim_idx"], [0, 1])
            np.testing.assert_array_equal(view["stimulus_magnitude"], [1.0, 1.0])
            np.testing.assert_allclose(view["proj"][:, 1, 1], np.full(3, 0.3),
                                       rtol=1e-6)
            self.assertEqual(view["is_fixed"].shape, (3, 2))
            self.assertEqual(view["one_trial_target_magnitude"], 1.0)
            with patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot._render_multitask_cue_interp_2d(
                    entry, projection["alphas"], "delaydm1", "gradient",
                    (1, 2), Path("out.png"), "test")
            fig = save.call_args.args[0]
            self.addCleanup(paper_plot.plt.close, fig)
            self.assertEqual(len(fig.axes), 4)
            for ax in fig.axes:
                self.assertEqual(len(ax.lines), 2)
            self.assertIn("one trial per stimulus, magnitude ≈ 1, 1",
                          save.call_args.kwargs["extra"])
        # The entry itself is untouched by the view.
        self.assertEqual(sweep["proj"].shape, (3, 4, 6))

    def test_cli_option_turns_on_one_trial_per_stimulus(self):
        self.assertFalse(paper_plot.SIBLING_ONE_TRIAL_PER_STIMULUS)
        self.addCleanup(setattr, paper_plot, "SIBLING_ONE_TRIAL_PER_STIMULUS",
                        False)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(paper_plot, "FIGURES_BY_MODE",
                             {"two_in_multiple": {"dummy": Mock()}}), \
                patch.object(paper_plot, "ALL_FIGURES", {"dummy": Mock()}), \
                patch.object(paper_plot, "OUT_DIR", Path(directory)), \
                patch.object(paper_plot, "_set_pretraining_bound"), \
                patch("sys.argv", ["paper_plot.py", "--only", "dummy",
                                   "--sibling-one-trial-per-stimulus"]), \
                patch("builtins.print"):
            paper_plot.main()
        self.assertTrue(paper_plot.SIBLING_ONE_TRIAL_PER_STIMULUS)

    def test_cli_option_avoids_attractor_switches_in_the_one_trial_view(self):
        self.assertFalse(paper_plot.SIBLING_AVOID_ATTRACTOR_SWITCHES)
        self.assertFalse(paper_plot.SIBLING_ONE_TRIAL_PER_STIMULUS)
        for name in ("SIBLING_AVOID_ATTRACTOR_SWITCHES",
                     "SIBLING_ONE_TRIAL_PER_STIMULUS"):
            self.addCleanup(setattr, paper_plot, name, False)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(paper_plot, "FIGURES_BY_MODE",
                             {"two_in_multiple": {"dummy": Mock()}}), \
                patch.object(paper_plot, "ALL_FIGURES", {"dummy": Mock()}), \
                patch.object(paper_plot, "OUT_DIR", Path(directory)), \
                patch.object(paper_plot, "_set_pretraining_bound"), \
                patch("sys.argv", ["paper_plot.py", "--only", "dummy",
                                   "--sibling-avoid-attractor-switches"]), \
                patch("builtins.print"):
            paper_plot.main()
        self.assertTrue(paper_plot.SIBLING_AVOID_ATTRACTOR_SWITCHES)
        self.assertTrue(paper_plot.SIBLING_ONE_TRIAL_PER_STIMULUS)

    def test_cli_option_switches_the_dmc_kept_pc_to_every_alpha(self):
        self.assertFalse(paper_plot.SIBLING_DMC_KEEP_AT_EVERY_ALPHA)
        self.addCleanup(setattr, paper_plot, "SIBLING_DMC_KEEP_AT_EVERY_ALPHA",
                        False)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(paper_plot, "FIGURES_BY_MODE",
                             {"two_in_multiple": {"dummy": Mock()}}), \
                patch.object(paper_plot, "ALL_FIGURES", {"dummy": Mock()}), \
                patch.object(paper_plot, "OUT_DIR", Path(directory)), \
                patch.object(paper_plot, "_set_pretraining_bound"), \
                patch("sys.argv", ["paper_plot.py", "--only", "dummy",
                                   "--sibling-dmc-keep-every-alpha"]), \
                patch("builtins.print"):
            paper_plot.main()
        self.assertTrue(paper_plot.SIBLING_DMC_KEEP_AT_EVERY_ALPHA)
        self.assertEqual(paper_plot._dmc_sweep_metric_name(),
                         paper_plot.DMC_SWEEP_KEEP_EVERY_ALPHA_METRIC_NAME)

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

    def test_delaydm_plane_is_chosen_on_the_sweep_itself(self):
        # Every PC carries noise. PC1 separates the directions widely but
        # merges them at one intermediate alpha; PC3 separates them less but
        # at every alpha, so the chosen plane holds PC3 for both methods.
        n_alpha, n_points = 6, 8
        stim = np.repeat(np.arange(4), 2)
        rng = np.random.RandomState(1)
        proj = rng.normal(scale=0.1, size=(n_alpha, n_points, 6))
        proj[:, :, 0] += 8.0 * stim[None, :] * np.where(
            np.arange(n_alpha) == 3, 0.0, 1.0)[:, None]
        proj[:, :, 2] += 2.0 * stim[None, :]
        entry = {"sweeps": {
            base: {"target_rule": target, "proj": proj.copy(),
                   "stim_idx": stim, "stimulus_magnitude": np.ones(n_points)}
            for base, target in (("delaydm1", "delaydm2"),
                                 ("delaydm2", "delaydm1"))}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("gradient", "long_delay_endpoint"):
                pair, note = paper_plot._multitask_cue_interp_plane(
                    "delaydm1", "e_modulation", method, root, "run",
                    entry=entry)
                self.assertIn(3, pair)
                self.assertIn(paper_plot.SWEEP_METRIC_NAME, note)
            # A sweep that cannot be scored (one trial per direction) falls
            # back to the base figure's plane instead of failing.
            single = {"sweeps": {"delaydm1": {
                "target_rule": "delaydm2", "proj": proj[:, ::2],
                "stim_idx": stim[::2], "stimulus_magnitude": np.ones(4)}}}
            with patch("builtins.print") as output:
                pair, note = paper_plot._multitask_cue_interp_plane(
                    "delaydm1", "e_modulation", "long_delay_endpoint", root,
                    "run", entry=single)
            self.assertEqual(
                pair, paper_plot._MULTITASK_SIBLING_ENDPOINT_DEFAULT_PC_PLANE)
            self.assertIn("falling back", output.call_args.args[0])

    def test_dmc_plane_pairs_the_kept_pc_with_the_swapped_pc(self):
        # DMCGo stimuli 0-3 form response cluster 1 at alpha = 0. PC3 keeps
        # the clusters apart in the same order, PC5 swaps them, the other PCs
        # are noise; the plane puts the kept PC first, for both methods, and
        # derives the clusters from the stimuli when none are saved.
        n_alpha = 5
        stim = np.tile(np.arange(8), 2)
        sign = np.where(stim < 4, 1.0, -1.0)
        rng = np.random.RandomState(2)
        proj = rng.normal(scale=0.05, size=(n_alpha, stim.size, 6))
        proj[:, :, 2] += sign[None, :]
        proj[:, :, 4] += (1.0 - 2.0 * np.linspace(0, 1, n_alpha))[:, None] * sign[None, :]
        entry = {"sweeps": {"dmcgo": {
            "target_rule": "dmcnogo", "proj": proj, "stim_idx": stim,
            "stimulus_magnitude": np.ones(stim.size)}}}
        with tempfile.TemporaryDirectory() as directory:
            for method in ("gradient", "long_delay_endpoint"):
                pair, note = paper_plot._multitask_cue_interp_plane(
                    "dmcgo", "hidden", method, Path(directory), "run", entry=entry)
                self.assertEqual(pair, (3, 5))
                self.assertIn(paper_plot.DMC_SWEEP_METRIC_NAME, note)


if __name__ == "__main__":
    unittest.main()
