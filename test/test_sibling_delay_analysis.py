"""Tests for the standalone sibling-family analysis boundary."""

import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import sibling_delay_analysis as sibling
import torch


class SiblingDelayAnalysisTests(unittest.TestCase):
    def test_family_names_expand_to_both_rules(self):
        self.assertEqual(
            sibling.SIBLING_FAMILIES["delaydm1"],
            ("delaydm1", "delaydm2"),
        )
        self.assertEqual(
            sibling.SIBLING_FAMILIES["dmcgo"],
            ("dmcgo", "dmcnogo"),
        )

    def test_run_identifier_matches_existing_convention(self):
        self.assertEqual(
            sibling.build_aname(921, "L21e4"),
            "everything_seed921_L21e4+hidden300+batch128+angle",
        )
        self.assertEqual(sibling.SIBLING_ANALYSIS_DIR, Path("two_in_multiples"))

    def test_cleanup_is_scoped_to_selected_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            delay_paths = [
                root / "fixed_points_grad_run_delaydm1.pkl",
                root / "fixed_points_grad_run_delaydm2.pkl",
                root / "delaydm1_delay_pc_projections_run.pkl",
            ]
            retained_paths = [
                root / "fixed_points_grad_run_dmcgo.pkl",
                root / "cluster_info_run.pkl",
            ]
            for path in delay_paths + retained_paths:
                path.touch()

            removed = sibling._clean_stale_sibling_artifacts(
                root, ("delaydm1",))

            self.assertEqual(set(removed), set(delay_paths))
            self.assertTrue(all(not path.exists() for path in delay_paths))
            self.assertTrue(all(path.exists() for path in retained_paths))

    def test_method_cleanup_preserves_the_other_fixed_point_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gradient_basis = root / "delaydm1_delay_trajectory_pca_run.pkl"
            gradient = root / "delaydm1_delay_pc_projections_run.pkl"
            endpoint_basis = (
                root
                / "delaydm1_long_delay_endpoint_delay_trajectory_pca_run.pkl"
            )
            legacy_endpoint_basis = (
                root / "delaydm1_delay_trajectory_pca_delaydm1_only_run.pkl"
            )
            endpoint = root / "delaydm1_long_delay_endpoints_run.pkl"
            for path in (gradient_basis, gradient, endpoint_basis, endpoint):
                path.touch()
            with legacy_endpoint_basis.open("wb") as stream:
                pickle.dump(
                    {"pca_strategy": "randomized_candidates"}, stream)

            removed = sibling._clean_stale_sibling_artifacts(
                root, ("delaydm1",), method="long_delay_endpoint")

            self.assertEqual(
                set(removed),
                {endpoint_basis, legacy_endpoint_basis, endpoint})
            self.assertTrue(gradient_basis.exists())
            self.assertTrue(gradient.exists())

    def test_artifact_ownership_covers_both_families(self):
        self.assertTrue(sibling.is_sibling_artifact(
            "fixed_points_grad_run_delaydm2.pkl"))
        self.assertTrue(sibling.is_sibling_artifact(
            "dmcgo_delay_pc_projections_run.pkl"))
        self.assertFalse(sibling.is_sibling_artifact("cluster_info_run.pkl"))

    def test_cli_requires_an_explicit_family(self):
        parser = sibling.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--seed", "921", "--feature", "L21e4",
                "--method", "gradient",
            ])
        args = parser.parse_args([
            "--seed", "921", "--feature", "L21e4",
            "--families", "delaydm1",
            "--method", "long_delay_endpoint",
        ])
        self.assertEqual(args.families, ["delaydm1"])
        self.assertEqual(args.method, "long_delay_endpoint")

    def test_cli_requires_an_explicit_method(self):
        parser = sibling.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--seed", "921", "--feature", "L21e4",
                "--families", "delaydm1",
            ])

    def test_long_delay_endpoint_tracks_only_the_last_delay_frames(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def reset_state(self, B=1):
                self.batch_size = B

            def network_step(self, current_input, run_mode="minimal", seq_idx=None):
                self.calls.append((seq_idx, run_mode))
                db = None
                if run_mode == "track_states":
                    db = {
                        "M1": torch.full((self.batch_size, 2, 3), float(seq_idx)),
                        "hidden1": torch.full(
                            (self.batch_size, 2), float(seq_idx + 1)),
                    }
                return None, None, db

        model = FakeModel()
        modulation, hidden, residual, n_settle = (
            sibling._run_to_settled_endpoint(
                model, torch.zeros(4, 9, 5), delay_stop=6, layer_index=1,
                leak=0.5, max_settle_steps=0))

        # The last two delay frames give the residual; no settling allowed.
        self.assertEqual(model.calls, [
            (0, "minimal"), (1, "minimal"), (2, "minimal"),
            (3, "minimal"), (4, "track_states"), (5, "track_states"),
        ])
        np.testing.assert_array_equal(modulation, np.full((4, 2, 3), 5.0))
        np.testing.assert_array_equal(hidden, np.full((4, 2), 6.0))
        # ||M5 - M4|| / ||M4|| / leak = (1 / 4) / 0.5.
        np.testing.assert_allclose(residual, np.full(4, 0.5))
        self.assertEqual(n_settle, 0)

    def test_settling_runs_on_the_last_delay_input_until_the_criterion(self):
        class SettlingModel:
            """M approaches 1 as 1 + 0.5 ** seq_idx; hidden is seq_idx."""

            def __init__(self):
                self.inputs = []

            def reset_state(self, B=1):
                self.batch_size = B

            def network_step(self, current_input, run_mode="minimal", seq_idx=None):
                self.inputs.append((seq_idx, float(current_input[0, 0])))
                db = None
                if run_mode == "track_states":
                    db = {
                        "M1": torch.full((self.batch_size, 2, 3),
                                         1.0 + 0.5 ** seq_idx),
                        "hidden1": torch.full(
                            (self.batch_size, 6), float(seq_idx)),
                    }
                return None, None, db

        inputs = torch.arange(6, dtype=torch.float)[None, :, None].repeat(2, 1, 4)
        # With leak 0.5 the residual after step t is 0.5**(t-1) / (1 + 0.5**(t-1)),
        # first at most 1e-2 at t = 8: five settling steps past the last delay
        # step, t = 3.
        model = SettlingModel()
        modulation, hidden, residual, n_settle = (
            sibling._run_to_settled_endpoint(
                model, inputs, delay_stop=4, layer_index=1, leak=0.5,
                max_settle_steps=50))
        self.assertEqual(n_settle, 5)
        np.testing.assert_allclose(modulation, np.full((2, 2, 3), 1 + 0.5 ** 8))
        np.testing.assert_array_equal(hidden, np.full((2, 6), 8.0))
        self.assertTrue(np.all(residual <= sibling.DEFAULT_REL_TOL_UNDAMPED))
        # Settling repeats the last delay input (3) at the next sequence indices.
        self.assertEqual(model.inputs[4:], [(t, 3.0) for t in range(4, 9)])

        # The cap stops it early and the residual says so.
        modulation, _, residual, n_settle = sibling._run_to_settled_endpoint(
            SettlingModel(), inputs, delay_stop=4, layer_index=1, leak=0.5,
            max_settle_steps=2)
        self.assertEqual(n_settle, 2)
        np.testing.assert_allclose(modulation, np.full((2, 2, 3), 1 + 0.5 ** 5))
        self.assertTrue(np.all(residual > sibling.DEFAULT_REL_TOL_UNDAMPED))

        # The projected trajectory runs from delay onset to the settled end.
        identity = np.eye(6, dtype=np.float32)
        basis = {"representations": {
            key: {"mean": np.zeros(6, dtype=np.float32), "components": identity}
            for key in ("fixed_hidden", "fixed_WM")}}
        modulation, _, trajectories, sample_indices, residual, n_settle = (
            sibling._run_long_delay_projected_trajectory(
                SettlingModel(), inputs, 1, 4, 1,
                np.ones((2, 3), dtype=np.float32), basis, leak=0.5,
                max_settle_steps=50, max_samples=20))
        self.assertEqual(n_settle, 5)
        np.testing.assert_array_equal(sample_indices, np.arange(1, 9))
        np.testing.assert_allclose(
            trajectories["fixed_WM"][0, :, 0], 1 + 0.5 ** np.arange(1, 9))
        np.testing.assert_allclose(
            trajectories["fixed_WM"][0, -1, 0], modulation[0, 0, 0])

    def test_long_delay_trajectory_is_projected_during_forward_pass(self):
        class FakeModel:
            def reset_state(self, B=1):
                self.batch_size = B

            def network_step(self, current_input, run_mode="minimal", seq_idx=None):
                db = None
                if run_mode == "track_states":
                    db = {
                        "M1": torch.full(
                            (self.batch_size, 2, 3), float(seq_idx)),
                        "hidden1": torch.full(
                            (self.batch_size, 6), float(seq_idx + 1)),
                    }
                return None, None, db

        identity = np.eye(6, dtype=np.float32)
        basis = {"representations": {
            key: {"mean": np.zeros(6, dtype=np.float32),
                  "components": identity}
            for key in ("fixed_hidden", "fixed_WM")
        }}
        modulation, hidden, trajectories, sample_indices, _, n_settle = (
            sibling._run_long_delay_projected_trajectory(
                FakeModel(), torch.zeros(2, 8, 4), 2, 7, 1,
                np.ones((2, 3), dtype=np.float32), basis, leak=1.0,
                max_settle_steps=0, max_samples=3))

        self.assertEqual(n_settle, 0)
        np.testing.assert_array_equal(sample_indices, [2, 4, 6])
        np.testing.assert_array_equal(modulation, np.full((2, 2, 3), 6.0))
        np.testing.assert_array_equal(hidden, np.full((2, 6), 7.0))
        self.assertEqual(trajectories["fixed_hidden"].shape, (2, 3, 6))
        self.assertEqual(trajectories["fixed_WM"].shape, (2, 3, 6))
        np.testing.assert_array_equal(
            trajectories["fixed_hidden"][0, :, 0], [3.0, 5.0, 7.0])
        np.testing.assert_array_equal(
            trajectories["fixed_WM"][0, :, 0], [2.0, 4.0, 6.0])

    def test_normal_delay_streams_small_batches_into_incremental_pca(self):
        class FakeModel:
            acc_measure = "angle"

            def __init__(self):
                self.max_batch = 0

            def reset_state(self, B=1):
                self.batch_size = B
                self.max_batch = max(self.max_batch, B)

            def network_step(self, current_input, run_mode="minimal", seq_idx=None):
                value = current_input[:, :1] + float(seq_idx)
                output = value.repeat(1, 3)
                db = None
                if run_mode == "track_states":
                    db = {
                        "M1": value[:, None, :].repeat(1, 2, 3),
                        "hidden1": value.repeat(1, 6),
                    }
                return output, None, db

            def compute_acc(self, *args, **kwargs):
                return torch.tensor(0.75), None

        n_trials, time_steps = 12, 5
        norm_input = torch.zeros(n_trials, time_steps, 4)
        norm_input[:, :, 0] = torch.arange(n_trials)[:, None]
        norm_output = torch.zeros(n_trials, time_steps, 3)
        norm_mask = torch.ones(n_trials, time_steps, 3)
        norm_task = np.repeat([0, 1], n_trials // 2)
        trials = [SimpleNamespace(epochs={"delay1": (1, 4)}),
                  SimpleNamespace(epochs={"delay1": (1, 4)})]

        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel()
            acc, per_task, preview = sibling.stream_normal_delay_analysis(
                "run", Path(directory), "delaydm1",
                ("delaydm1", "delaydm2"), model, torch.device("cpu"),
                norm_input, norm_output, norm_mask, trials, norm_task,
                np.ones((2, 3), dtype=np.float32), gpu_batch_size=4,
                pca_chunk_samples=8, n_components=2)

            self.assertEqual(model.max_batch, 4)
            self.assertEqual(acc, 0.75)
            self.assertEqual(per_task, {"delaydm1": 0.75, "delaydm2": 0.75})
            self.assertEqual(tuple(preview.shape), (4, time_steps, 3))
            with (Path(directory)
                  / "delaydm1_delay_trajectory_pca_run.pkl").open("rb") as stream:
                artifact = pickle.load(stream)
            self.assertEqual(
                artifact["trial_counts"], {"delaydm1": 6, "delaydm2": 6})
            self.assertEqual(artifact["gpu_batch_size"], 4)

        self.assertEqual(sibling.SIBLING_NORMAL_TRIALS_PER_RULE, 100)
        self.assertEqual(sibling.SIBLING_NORMAL_GPU_BATCH_SIZE, 200)

    def test_randomized_pca_strategy_fits_one_seeded_basis(self):
        class FakeModel:
            acc_measure = "angle"

            def reset_state(self, B=1):
                self.batch_size = B

            def network_step(self, current_input, run_mode="minimal", seq_idx=None):
                value = current_input[:, :1] + float(seq_idx)
                db = None
                if run_mode == "track_states":
                    db = {
                        "M1": value[:, None, :].repeat(1, 2, 3),
                        "hidden1": value.repeat(1, 6),
                    }
                return value.repeat(1, 3), None, db

            def compute_acc(self, *args, **kwargs):
                return torch.tensor(0.75), None

        n_trials, time_steps = 12, 5
        norm_input = torch.zeros(n_trials, time_steps, 4)
        norm_input[:, :, 0] = torch.arange(n_trials)[:, None]
        trials = [SimpleNamespace(epochs={"delay1": (1, 4)}),
                  SimpleNamespace(epochs={"delay1": (1, 4)})]
        with tempfile.TemporaryDirectory() as directory:
            sibling.stream_normal_delay_analysis(
                "run", Path(directory), "delaydm1",
                ("delaydm1", "delaydm2"), FakeModel(), torch.device("cpu"),
                norm_input, torch.zeros(n_trials, time_steps, 3),
                torch.ones(n_trials, time_steps, 3), trials,
                np.repeat([0, 1], n_trials // 2),
                np.ones((2, 3), dtype=np.float32), gpu_batch_size=4,
                pca_chunk_samples=8, n_components=2,
                pca_strategy="randomized")

            with (Path(directory)
                  / "delaydm1_long_delay_endpoint_delay_trajectory_pca_run.pkl"
                  ).open("rb") as stream:
                artifact = pickle.load(stream)
            self.assertEqual(artifact["version"], 7)
            self.assertEqual(artifact["pca_strategy"], "randomized")
            self.assertNotIn("pca_candidates", artifact)
            for key in ("fixed_hidden", "fixed_WM"):
                record = artifact["representations"][key]
                self.assertEqual(
                    record["method"],
                    "PCA(svd_solver=randomized, random_state="
                    f"{sibling.SIBLING_LONG_DELAY_PCA_SEED})")
                self.assertEqual(record["components"].shape[0], 2)

    def test_aligned_delay_window_validation(self):
        trials = [SimpleNamespace(epochs={"delay1": (4, 12)}),
                  SimpleNamespace(epochs={"delay1": (4, 12)})]
        self.assertEqual(
            sibling._aligned_delay1_window(("delaydm1", "delaydm2"), trials),
            (4, 12),
        )
        trials[1].epochs["delay1"] = (4, 13)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            sibling._aligned_delay1_window(("delaydm1", "delaydm2"), trials)

    def test_dmc_category_split_holds_when_top_direction_is_not_drawn(self):
        # Directions 0-3 lie below pi (category 1), 4-7 above. A random DMC
        # draw without direction 7 must still split between 3 and 4.
        stim_idx = np.array([0, 3, 4, 6, 0, 3, 4, 6])
        task_idx = np.repeat([0, 1], 4)
        np.testing.assert_array_equal(
            sibling._dmc_category_labels(task_idx, stim_idx),
            [1, 1, 0, 0, 0, 0, 1, 1])

    def test_long_delay_endpoints_project_into_saved_delay_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aname = "run"
            family = "delaydm1"
            rules = ("delaydm1", "delaydm2")
            identity = np.eye(6, dtype=np.float32)
            basis = {
                "representations": {
                    key: {
                        "mean": np.zeros(6, dtype=np.float32),
                        "components": identity,
                        "explained_variance_ratio": np.arange(1, 7) / 21,
                    }
                    for key in ("fixed_hidden", "fixed_WM")
                }
            }
            basis_path = (
                root
                / f"{family}_long_delay_endpoint_delay_trajectory_pca_{aname}.pkl"
            )
            with basis_path.open("wb") as stream:
                pickle.dump(basis, stream)

            hidden = np.arange(24, dtype=np.float32).reshape(4, 6)
            effective = (100 + np.arange(24, dtype=np.float32)).reshape(4, 2, 3)
            endpoints = {
                "method": "long_delay_endpoint",
                "task_names": list(rules),
                "delay_window": (3, 103),
                "delay_steps": 100,
                "delay_ms": 4000,
                "task_idx": np.array([0, 0, 1, 1]),
                "condition_idx": np.array([0, 1, 0, 1]),
                "stim_idx": np.array([2, 5, 2, 5]),
                "stimulus_magnitude": np.ones(4),
                "trajectory_basis_scope": "joint",
                "trajectory_sample_indices": np.array([3, 53, 102]),
                "trajectory_delay_steps": np.array([0, 50, 99]),
                "trajectory_time_ms": np.array([0, 2000, 3960]),
                "representations": {
                    "fixed_hidden": hidden,
                    "fixed_WM": effective,
                },
                "projected_trajectories": {
                    "fixed_hidden": np.repeat(hidden[:, None, :], 3, axis=1),
                    "fixed_WM": np.repeat(
                        effective.reshape(4, 1, 6), 3, axis=1),
                },
            }
            endpoint_path = root / f"{family}_long_delay_endpoints_{aname}.pkl"
            with endpoint_path.open("wb") as stream:
                pickle.dump(endpoints, stream)

            with patch.object(sibling, "_save_pc_pair_gallery",
                              return_value=root / "gallery.png"):
                output_path = sibling.save_long_delay_endpoint_pc_projections(
                    aname, root, family, rules, endpoint_path)

            with output_path.open("rb") as stream:
                projected = pickle.load(stream)
            self.assertEqual(projected["method"], "long_delay_endpoint")
            self.assertEqual(projected["version"], 3)
            self.assertEqual(projected["delay_ms"], 4000)
            np.testing.assert_array_equal(
                projected["representations"]["hidden"]["proj"], hidden)
            np.testing.assert_array_equal(
                projected["representations"]["e_modulation"]["proj"],
                effective.reshape(4, 6))
            np.testing.assert_array_equal(
                projected["representations"]["hidden"]["condition_idx"],
                [0, 1, 0, 1])
            self.assertEqual(
                projected["representations"]["hidden"]["trajectory_proj"].shape,
                (4, 3, 6))
            np.testing.assert_array_equal(
                projected["representations"]["hidden"]["trajectory_time_ms"],
                [0, 2000, 3960])
            # An endpoint cache from before settling ends with the trial's delay.
            self.assertEqual(projected["settle_ms"], 0)
            self.assertEqual(projected["endpoint_delay_ms"], 4000)
            self.assertNotIn("is_fixed", projected["representations"]["hidden"])

            # A settled cache carries its settling time and convergence flags.
            endpoints.update(settle_ms=1200, endpoint_delay_ms=5200,
                             is_fixed=np.array([True, True, False, True]),
                             rel_step_undamped=np.array([1e-3, 2e-3, 0.5, 4e-3]))
            with endpoint_path.open("wb") as stream:
                pickle.dump(endpoints, stream)
            with patch.object(sibling, "_save_pc_pair_gallery",
                              return_value=root / "gallery.png"):
                output_path = sibling.save_long_delay_endpoint_pc_projections(
                    aname, root, family, rules, endpoint_path)
            with output_path.open("rb") as stream:
                projected = pickle.load(stream)
            self.assertEqual(projected["settle_ms"], 1200)
            self.assertEqual(projected["endpoint_delay_ms"], 5200)
            for plot_name in ("hidden", "e_modulation"):
                np.testing.assert_array_equal(
                    projected["representations"][plot_name]["is_fixed"],
                    [True, True, False, True])


if __name__ == "__main__":
    unittest.main()
