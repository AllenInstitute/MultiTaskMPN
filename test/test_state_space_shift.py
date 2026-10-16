"""Regression tests for the focused state-space checkpoint cohort."""

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import numpy as np
import torch
import state_space_shift as state_space
import paper_plot


class StateSpaceCohortTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "net_params": {
                "activation": "tanh",
                "input_layer_add": True,
                "linear_embed": 300,
                "n_neurons": [1, 300, 1],
            },
            "train_params": {
                "weight_reg": "L2",
                "reg_lambda": 1e-3,
            },
        }

    def add_run(self, root, name, config=None):
        checkpoint = root / f"savednet_{name}.pt"
        checkpoint.touch()
        if config is not None:
            path = root / f"param_{name}_param.json"
            path.write_text(json.dumps(config))
        return str(checkpoint)

    def test_selects_all_exact_tanh_300x300_target_l2_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_l21e3 = self.add_run(
                root, "everything_seed86_L21e3+hidden300+batch128+angle",
                self.config)
            config_l21e5 = copy.deepcopy(self.config)
            config_l21e5["train_params"]["reg_lambda"] = 1e-5
            target_l21e5 = self.add_run(
                root, "everything_seed223_L21e5+hidden300+batch128+angle",
                config_l21e5)
            config_l21e4 = copy.deepcopy(self.config)
            config_l21e4["train_params"]["reg_lambda"] = 1e-4
            target_l21e4 = self.add_run(
                root, "everything_seed408_L21e4+hidden300+batch128+angle",
                config_l21e4)
            config_l21e2 = copy.deepcopy(self.config)
            config_l21e2["train_params"]["reg_lambda"] = 1e-2
            target_l21e2 = self.add_run(
                root, "everything_seed132_L21e2+hidden300+batch128+angle",
                config_l21e2)

            variants = []
            for suffix, section, field, value in (
                ("relu", "net_params", "activation", "relu"),
                ("proj", "net_params", "linear_embed", 200),
                ("hidden", "net_params", "n_neurons", [1, 200, 1]),
                ("noinput", "net_params", "input_layer_add", False),
                ("reg", "train_params", "reg_lambda", 1e-4),
            ):
                config = copy.deepcopy(self.config)
                config[section][field] = value
                variants.append(self.add_run(
                    root,
                    f"everything_seed1{suffix}_L21e3+hidden300+batch128+angle",
                    config,
                ))

            wrong_feature = self.add_run(
                root, "everything_seed999_L21e6+hidden300+batch128+angle",
                self.config)
            mismatched_feature_and_config = self.add_run(
                root, "everything_seed997_L21e5+hidden300+batch128+angle",
                self.config)
            missing_config = self.add_run(
                root, "everything_seed998_L21e3+hidden300+batch128+angle")

            selected = state_space._select_target_checkpoints(
                variants + [target_l21e5, target_l21e4, wrong_feature,
                            target_l21e3, mismatched_feature_and_config,
                            target_l21e2, missing_config])
            self.assertEqual(
                selected,
                [target_l21e5, target_l21e4, target_l21e3, target_l21e2])

    def test_checkpoint_name_validation(self):
        with self.assertRaisesRegex(ValueError, "savednet"):
            state_space._checkpoint_aname("multiple_tasks/model.pt")

    def test_cleanup_removes_top_level_files_but_preserves_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.pkl").write_bytes(b"old")
            (root / "old.png").write_bytes(b"old")
            nested = root / "keep"
            nested.mkdir()
            (nested / "sentinel.txt").write_text("keep")

            removed = state_space._clean_state_space_results(root)

            self.assertEqual(set(removed), {"old.pkl", "old.png"})
            self.assertEqual(list(root.iterdir()), [nested])
            self.assertEqual((nested / "sentinel.txt").read_text(), "keep")


class StateSpaceContextRecordTests(unittest.TestCase):
    def test_eval_one_writes_centroids_to_original_pca_cache_and_keeps_scatter(self):
        rules = list(paper_plot._RULE_MOTIF)
        n_trials = 2 * len(rules)
        labels = np.repeat(np.arange(len(rules)), 2)
        context_steps = int(state_space.FIXED_FIXATION_MS / 40)
        sequence_length = context_steps + 2
        random = np.random.default_rng(42)
        hidden = torch.tensor(random.normal(size=(n_trials, sequence_length, 3)),
                              dtype=torch.float32)
        modulation = torch.tensor(random.normal(size=(n_trials, sequence_length, 3, 3)),
                                  dtype=torch.float32)
        weights = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
        config = {"task_params": {}, "train_params": {}, "net_params": {}}
        converted = {"rules": rules, "hp": {"dt": 40}}
        trials = [SimpleNamespace(epochs={"fix1": (0, context_steps)})
                  for _ in rules]
        tensors = tuple(torch.zeros(n_trials, sequence_length, 3) for _ in range(3))
        model = Mock(acc_measure="angle")
        model.load_state_dict.return_value = ([], [])
        model.compute_acc.return_value = (1.0, None)
        model.iterate_sequence_batch.side_effect = [
            (None, None, None),
            (None, None, {"hidden1": hidden, "M1": modulation}),
        ]
        aname = "everything_seed408_L21e4+hidden3+batch128+angle"
        pca_figures = {}

        def capture_figure(figure, path, **kwargs):
            if Path(path).name.startswith("state_space_shift_"):
                for name in ("hidden", "mod", "eff_mod"):
                    if Path(path).name == f"state_space_shift_{aname}_{name}_noise0.01.png":
                        pca_figures[name] = figure

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"param_{aname}_param.json").write_text(json.dumps(config))
            with patch.object(state_space, "STATE_SPACE_DIR", root), \
                    patch.object(state_space, "device", torch.device("cpu")), \
                    patch.object(state_space.torch, "load", return_value={
                        "net_params": {}, "state_dict": {
                            "mp_layer1.W": weights,
                            "mp_layer1.lam": torch.tensor(0.99)}}), \
                    patch.object(state_space.mpn, "DeepMultiPlasticNet", return_value=model), \
                    patch.object(state_space.mpn_tasks, "convert_and_init_multitask_params",
                                 return_value=(converted, {}, {})), \
                    patch.object(state_space.mpn_tasks, "generate_trials_wrap",
                                 return_value=(tensors, (None, trials, labels))), \
                    patch.object(state_space.helper, "generate_response_stimulus",
                                 return_value=(None, np.zeros(n_trials, dtype=int), None,
                                               {rule: trials[0].epochs for rule in rules})), \
                    patch.object(state_space.helper, "linear_regression",
                                 return_value=(np.array([0., 1.]), np.array([12., 13.]),
                                               0.8, 1.0, 12.0, 0.01)) as regression, \
                    patch.object(state_space.plt.Figure, "savefig", autospec=True,
                                 side_effect=capture_figure):
                result = state_space.eval_one(root / f"savednet_{aname}.pt")
            with (root / f"state_space_pca_{aname}_noise0.01.pkl").open("rb") as stream:
                artifact = pickle.load(stream)

        saved_names = {"hidden", "mod", "eff_mod", "mod_hebb", "eff_mod_hebb"}
        self.assertEqual(result[0], aname)
        self.assertEqual(set(result[3]), saved_names)
        self.assertEqual(set(result[4]), saved_names)
        self.assertAlmostEqual(result[7], 0.99, places=6)
        self.assertEqual(regression.call_count, 5)
        self.assertTrue(all(call.kwargs["through_origin"] is False
                            and call.kwargs["log"] is False
                            for call in regression.call_args_list))
        n_pairs = len(rules) * (len(rules) - 1) // 2
        for name in saved_names:
            scatter = result[4][name]
            self.assertEqual(result[3][name], (0.8, 1.0, 0.01))
            self.assertEqual(scatter["regression"]["intercept"], 12.0)
            self.assertIs(scatter["regression"]["through_origin"], False)
            self.assertEqual(len(scatter["pair_rules"]), n_pairs)
            self.assertEqual(scatter["dists"].shape, (n_pairs,))
            self.assertIsNone(scatter["permutation_skipped"])
            permutation = scatter["permutation"]
            self.assertEqual(permutation["n_permutations"],
                             state_space.DISTANCE_ANGLE_N_PERMUTATIONS)
            self.assertEqual(permutation["sided"], "two-sided")
            self.assertTrue(0 < permutation["p_value"] <= 1)
            self.assertIn("initial", scatter["step_norms"])
        for name in ("hidden", "mod", "eff_mod"):
            self.assertEqual(result[4][name]["step"]["kind"], state_space.STEP_FULL)
            self.assertIsNone(result[4][name]["step"]["retention"])
        for name, source in (("mod_hebb", "mod"), ("eff_mod_hebb", "eff_mod")):
            step = result[4][name]["step"]
            self.assertEqual((step["kind"], step["source"]), (state_space.STEP_HEBBIAN, source))
            self.assertAlmostEqual(step["retention"], 0.99, places=6)
            self.assertIn("decay_term", result[4][name]["step_norms"])
        self.assertNotIn("decay_term", result[4]["hidden"]["step_norms"])
        # The Hebbian step of the mocked states differs from the full step, so
        # the two definitions give different angles for the same pairs.
        np.testing.assert_array_equal(result[4]["mod"]["dists"], result[4]["mod_hebb"]["dists"])
        self.assertFalse(np.allclose(result[4]["mod"]["angles_deg"],
                                     result[4]["mod_hebb"]["angles_deg"]))
        self.assertEqual(artifact["evaluation_seed"], state_space.EVALUATION_SEED)
        self.assertEqual(artifact["rule_motif_mapping"], paper_plot._RULE_MOTIF)
        # A trained run's cache is marked as such and records its accuracy.
        self.assertIs(artifact["untrained"], False)
        self.assertEqual(artifact["accuracy"], 1.0)
        for name, states in (("hidden", hidden), ("mod", modulation),
                             ("eff_mod", modulation * weights)):
            record = artifact["pca_results"][name]
            expected = states[:, context_steps - 1].numpy().reshape(len(rules), 2, -1)
            np.testing.assert_allclose(record["task_centroids"],
                                       expected.mean(axis=1, dtype=np.float64))
            np.testing.assert_array_equal(record["task_trial_counts"], [2] * len(rules))
            self.assertEqual(record["centroid_space"], "original_features")
            self.assertEqual(record["X_2d"].shape, (n_trials, 2))
            axis = pca_figures[name].axes[1]
            self.assertEqual(len(axis.collections), 6)
            for collection, (category, color) in zip(
                    axis.collections, paper_plot._STATE_SPACE_LEGEND):
                self.assertEqual(collection.get_label(), category)
                np.testing.assert_allclose(
                    collection.get_facecolors()[0, :3],
                    paper_plot.mpl.colors.to_rgb(color))
                selected = np.isin(record["ctx_rule_labels"], [
                    index for index, rule in enumerate(rules)
                    if paper_plot._RULE_MOTIF[rule][1] == color
                ])
                np.testing.assert_allclose(collection.get_offsets(), record["X_2d"][selected])

    def test_free_intercept_regression_matches_standard_ols(self):
        from scipy.stats import linregress

        distances = np.arange(1., 7.)
        angles = 15 + 3 * distances + np.array([0.2, -0.1, 0.3, -0.2, 0.1, -0.3])
        expected = linregress(distances, angles)
        x_fit, y_fit, correlation, slope, intercept, p_value = (
            state_space.helper.linear_regression(
                distances, angles, log=False, through_origin=False))
        self.assertGreater(intercept, 14)
        np.testing.assert_allclose(y_fit, intercept + slope * x_fit)
        np.testing.assert_allclose(
            [correlation, slope, intercept, p_value],
            [expected.rvalue, expected.slope, expected.intercept, expected.pvalue])

    def test_pca_cache_centroids_retain_all_original_dimensions(self):
        states = np.random.default_rng(3).normal(size=(8, 3, 4)).astype(np.float32)
        labels = np.repeat(np.arange(4), 2)
        record = state_space._context_pca_record(states, labels, 4, batch_size=3)
        self.assertEqual(record["X_2d"].shape, (8, 2))
        self.assertEqual(record["task_centroids"].shape, (4, 12))
        np.testing.assert_allclose(
            record["task_centroids"], states.reshape(4, 2, 12).mean(axis=1),
            atol=1e-7)
        np.testing.assert_array_equal(record["task_trial_counts"], [2] * 4)
        self.assertEqual(record["centroid_space"], "original_features")

    def test_display_pca_is_batched_without_modifying_source(self):
        states = np.random.default_rng(7).normal(size=(10, 3, 4)).astype(np.float32)
        labels = np.repeat(np.arange(5), 2)
        batch_lengths = []
        original_partial_fit = state_space.IncrementalPCA.partial_fit

        def track_batch(pca, samples, *args, **kwargs):
            batch_lengths.append(len(samples))
            return original_partial_fit(pca, samples, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            mapped = np.memmap(Path(directory) / "states.dat", mode="w+",
                               dtype=np.float32, shape=states.shape)
            try:
                mapped[:] = states
                with patch.object(state_space.IncrementalPCA, "partial_fit", track_batch):
                    record = state_space._context_pca_record(
                        mapped, labels, 5, batch_size=3)
                np.testing.assert_array_equal(mapped, states)
            finally:
                mapped._mmap.close()
        self.assertEqual(batch_lengths, [3, 3, 4])
        np.testing.assert_allclose(
            record["task_centroids"],
            states.reshape(5, 2, 12).mean(axis=1, dtype=np.float64))
        self.assertEqual(record["X_2d"].shape, (10, 2))
        self.assertTrue(np.isfinite(record["X_2d"]).all())
        self.assertEqual(record["pca_method"], "IncrementalPCA")


class UntrainedNetworkTests(unittest.TestCase):
    """The untrained control: a fresh initialization of a run's architecture,
    analyzed on the same trials and cached apart from the trained runs."""

    aname = "everything_seed921_L21e4+hidden300+batch128+angle"

    @staticmethod
    def _recording_constructor(draws):
        """A DeepMultiPlasticNet stand-in that records the NumPy and torch
        random numbers available at construction, as the real initializer
        draws its weights from both generators."""
        def construct(net_params, verbose=False, forzihan=True):
            draws.append((np.random.rand(3), torch.rand(3)))
            model = Mock()
            model.state_dict.return_value = {"mp_layer1.W": torch.ones(2, 2),
                                             "mp_layer1.lam": torch.tensor(0.99)}
            return model
        return construct

    def test_untrained_network_is_seeded_and_loads_no_weights(self):
        checkpoint = {"net_params": {"n_neurons": [1, 2, 1]},
                      "state_dict": {"mp_layer1.W": torch.full((2, 2), 7.0)}}
        draws = []
        with patch.object(state_space.mpn, "DeepMultiPlasticNet",
                          side_effect=self._recording_constructor(draws)) as constructor:
            first, first_state = state_space._build_network(checkpoint, untrained_init_seed=5)
            np.random.rand(10)  # whatever ran in between, the seed fixes the weights
            _, again_state = state_space._build_network(checkpoint, untrained_init_seed=5)
            state_space._build_network(checkpoint, untrained_init_seed=6)
        first.load_state_dict.assert_not_called()
        np.testing.assert_array_equal(draws[0][0], draws[1][0])
        self.assertTrue(torch.equal(draws[0][1], draws[1][1]))
        self.assertFalse(np.array_equal(draws[0][0], draws[2][0]))
        # W and lam come from the fresh network, never from the trained weights.
        np.testing.assert_array_equal(first_state["mp_layer1.W"].numpy(), np.ones((2, 2)))
        self.assertEqual(constructor.call_args.args[0], checkpoint["net_params"])

        trained_model = Mock()
        trained_model.load_state_dict.return_value = ([], [])
        with patch.object(state_space.mpn, "DeepMultiPlasticNet", return_value=trained_model):
            model, state_dict = state_space._build_network(checkpoint)
        trained_model.load_state_dict.assert_called_once_with(checkpoint["state_dict"], strict=True)
        self.assertIs(state_dict, checkpoint["state_dict"])

    def _states(self):
        rules = list(paper_plot._RULE_MOTIF)
        n_trials, n_steps = 2 * len(rules), 4
        rng = np.random.default_rng(0)
        epochs = {rule: {"fix1": (0, 2)} for rule in rules}
        return {
            "aname": self.aname, "hidden_size": 300, "l2_info": 1e-4, "retention": 0.99,
            "dt_ms": 40, "fixed_fixation_steps": 2, "accuracy": 0.1, "all_rules": rules,
            "rule_indices": np.repeat(np.arange(len(rules)), 2),
            "stimulus_labels": np.zeros(n_trials, dtype=int), "rules_epochs": epochs,
            "hidden": rng.normal(size=(n_trials, n_steps, 3)),
            "mod": rng.normal(size=(n_trials, n_steps, 4)),
            "eff_mod": rng.normal(size=(n_trials, n_steps, 4)),
        }

    def test_eval_untrained_writes_a_marked_cache_to_the_untrained_subdirectory(self):
        states = self._states()
        rval_dict = {"hidden": (0.4, 1.0, 0.01)}
        scatter_dict = {"hidden": {"permutation": {"p_value": 0.02}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(state_space, "_context_states",
                              return_value=states) as context_states, \
                    patch.object(state_space, "_draw_context_pca_figure") as draw, \
                    patch.object(state_space, "_distance_angle_for_states",
                                 return_value=(rval_dict, scatter_dict)) as distance_angle:
                stem, entry = state_space.eval_untrained(
                    root / f"savednet_{self.aname}.pt", root, init_seed=3)
            self.assertEqual(context_states.call_args.kwargs, {"untrained_init_seed": 3})
            # The distance-angle control runs on the same states, with its
            # diagnostic figure named after the untrained network.
            self.assertEqual(distance_angle.call_args.args,
                             (states, root / "untrained", f"{self.aname}_init3"))
            self.assertEqual(stem, f"{self.aname}_init3")
            self.assertEqual(state_space.untrained_file_stem(self.aname, 3), stem)
            path = root / "untrained" / f"state_space_pca_{self.aname}_init3_noise0.01.pkl"
            self.assertTrue(all("_init3_" in call.args[3].name for call in draw.call_args_list))
            # Only the untrained subdirectory is written: no top-level PCA
            # cache or results pickle for the trained readers to pick up.
            self.assertEqual([item.name for item in root.iterdir()], ["untrained"])
            self.assertEqual(draw.call_count, 3)
            with path.open("rb") as stream:
                cache = pickle.load(stream)
        # The results entry has a trained run's fields, the untrained marks
        # and no L2 strength, so no cohort grouping can take it in.
        self.assertEqual(entry["rval_dict"], rval_dict)
        self.assertEqual(entry["scatter"], scatter_dict)
        self.assertEqual(entry["schema_version"], state_space.DISTANCE_ANGLE_SCHEMA_VERSION)
        self.assertEqual(entry["retention"], 0.99)
        self.assertIsNone(entry["l2_info"])
        self.assertEqual(entry["config_l2_info"], 1e-4)
        self.assertEqual((entry["untrained"], entry["init_seed"], entry["config_source"]),
                         (True, 3, self.aname))
        self.assertIs(cache["untrained"], True)
        self.assertEqual(cache["init_seed"], 3)
        self.assertEqual(cache["config_source"], self.aname)
        self.assertEqual(cache["accuracy"], 0.1)
        self.assertEqual(set(cache["pca_results"]), {"hidden", "mod", "eff_mod"})
        self.assertEqual(cache["pca_results"]["mod"]["task_centroids"].shape,
                         (len(paper_plot._RULE_MOTIF), 4))

    def test_untrained_cache_is_invisible_to_structure_and_survives_cleanup(self):
        import state_space_structure
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            untrained = root / "untrained"
            untrained.mkdir()
            cache = untrained / f"state_space_pca_{self.aname}_init0_noise0.01.pkl"
            cache.write_bytes(b"x")
            self.assertEqual(state_space_structure.pca_cache_paths(root), [])
            state_space._clean_state_space_results(root)
            self.assertTrue(cache.exists())

    def test_cli_untrained_needs_a_checkpoint_and_owns_the_init_seed(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                state_space._parse_args(["--untrained"])
            with self.assertRaises(SystemExit):
                state_space._parse_args(["--checkpoint", "a.pt", "--init-seed", "4"])
        with patch.object(state_space, "run_untrained") as run_untrained, \
                patch.object(state_space, "run_checkpoints") as run_checkpoints, \
                patch.object(state_space, "summarize") as summarize:
            state_space.main(["--untrained", "--checkpoint", "a.pt", "--out-dir", "scratch"])
            state_space.main(["--untrained", "--checkpoint", "a.pt", "--init-seed", "4", "5"])
        self.assertEqual([call.args for call in run_untrained.call_args_list],
                         [(["a.pt"], Path("scratch"), [state_space.UNTRAINED_INIT_SEED]),
                          (["a.pt"], state_space.STATE_SPACE_DIR, [4, 5])])
        run_checkpoints.assert_not_called()
        # The summary covers the untrained networks' own results only.
        self.assertEqual([call.args for call in summarize.call_args_list],
                         [(Path("scratch") / "untrained",),
                          (state_space.STATE_SPACE_DIR / "untrained",)])
        with self.assertRaises(FileNotFoundError):
            state_space.run_untrained(["missing/savednet_x.pt"])

    def test_run_untrained_merges_one_entry_per_seed_into_its_own_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / f"savednet_{self.aname}.pt"
            checkpoint.touch()
            untrained = root / "untrained"
            untrained.mkdir()
            results_path = untrained / state_space.RESULTS_NAME
            with results_path.open("wb") as stream:
                pickle.dump({"kept_init9": {"init_seed": 9}}, stream)
            with patch.object(state_space, "eval_untrained",
                              side_effect=lambda path, out, seed: (f"net_init{seed}",
                                                                   {"init_seed": seed})
                              ) as evaluate, \
                    patch.object(state_space, "_matches_target_cohort", return_value=True):
                merged = state_space.run_untrained([str(checkpoint)], root, init_seeds=[2, 0])
                with self.assertRaises(ValueError):
                    state_space.run_untrained([str(checkpoint)], root, init_seeds=[1, 1])
            with results_path.open("rb") as stream:
                saved = pickle.load(stream)
            # The trained runs' results pickle is never created or touched.
            self.assertFalse((root / state_space.RESULTS_NAME).exists())
        self.assertEqual([call.args[2] for call in evaluate.call_args_list], [2, 0])
        self.assertEqual(set(saved), {"kept_init9", "net_init2", "net_init0"})
        self.assertEqual(saved, merged)
        self.assertEqual(saved["net_init2"], {"init_seed": 2})


if __name__ == "__main__":
    unittest.main()
