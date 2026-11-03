"""Tests of the modulation transplant experiment
(multiple_task/state_space_transplant.py) and its paper figure."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import torch

import mpn
import paper_plot
import state_space_shift as shift
import state_space_transplant as transplant

RULES = ["fdgo", "fdanti", "delaygo", "delayanti"]
N_INPUT = 6
CONTEXT_STEPS = 3


def _tiny_network(seed=0, hidden=5):
    """A real one-layer MPN with random weights (eta 1, so the updates are
    large enough to reach the M bounds)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net_params = {
        "n_neurons": [N_INPUT, hidden, 2], "linear_embed": 4, "input_layer_add": True,
        "input_layer_add_trainable": True, "input_layer_bias": False, "output_bias": False,
        "activation": "tanh", "output_matrix": "", "loss_type": "MSE", "acc_measure": "angle",
        "dt": 40,
        "ml_params": {"bias": True, "mp_type": "mult", "m_update_type": "hebb_assoc",
                      "eta_type": "scalar", "eta_train": True, "lam_type": "scalar",
                      "m_time_scale": 400, "lam_train": False, "W_freeze": False,
                      "m_bounds": [-1.0, 1.0]},
    }
    with patch("builtins.print"):
        model = mpn.DeepMultiPlasticNet(net_params, verbose=False, forzihan=True)
    model.eval()
    return model


def _trials(trials_per_rule=3, steps=5, seed=1):
    """Inputs with one rule channel per task on throughout, a stimulus
    channel from the onset step on and small noise: (inputs, rule_indices)."""
    rng = np.random.default_rng(seed)
    rule_indices = np.repeat(np.arange(len(RULES)), trials_per_rule)
    inputs = np.zeros((rule_indices.size, steps, N_INPUT))
    for trial, rule in enumerate(rule_indices):
        inputs[trial, :, rule] = 1.0
        inputs[trial, CONTEXT_STEPS:, len(RULES) + trial % 2] = 1.0
    inputs += 0.01 * rng.normal(size=inputs.shape)
    return torch.tensor(inputs, dtype=torch.float32), rule_indices


def _evaluation(model, inputs, rule_indices, aname, evaluation_seed=42):
    """A prepared evaluation as state_space_shift.prepare_evaluation returns it."""
    return {"aname": aname, "hidden_size": 5, "l2_info": 1e-4, "model": model,
            "state_dict": {}, "task_params": {"rules": RULES}, "inputs": inputs,
            "outputs": None, "masks": None, "trials": None, "rule_indices": rule_indices,
            "retention": 0.9, "dt_ms": 40, "fixed_fixation_steps": CONTEXT_STEPS,
            "evaluation_seed": evaluation_seed}


class BranchStepTests(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_network()
        self.inputs, self.rule_indices = _trials()
        self.x_context = self.inputs[:, CONTEXT_STEPS - 1]
        self.x_onset = self.inputs[:, CONTEXT_STEPS]

    def test_self_transplant_reproduces_the_uninterrupted_run(self):
        m_context, _ = transplant.modulation_after_steps(self.model, self.inputs, CONTEXT_STEPS)
        m_onset, hidden = transplant.modulation_after_steps(self.model, self.inputs,
                                                            CONTEXT_STEPS + 1)
        native = transplant.branch_step(self.model, m_context, self.x_context, self.x_onset)
        torch.testing.assert_close(native["h_onset"], hidden[:, CONTEXT_STEPS])
        torch.testing.assert_close(native["m_after"], m_onset)
        check = {"max_abs_hidden_diff": 0.0, "max_abs_m_diff": 0.0,
                 "tolerance": transplant.NATIVE_CHECK_TOLERANCE}
        transplant._check_native_branch(check, native, hidden[:, CONTEXT_STEPS], m_onset)
        self.assertLessEqual(max(check["max_abs_hidden_diff"], check["max_abs_m_diff"]),
                             transplant.NATIVE_CHECK_TOLERANCE)
        # A start state that is not the trial's own is caught.
        perturbed = transplant.branch_step(self.model, m_context + 0.05, self.x_context,
                                           self.x_onset)
        with self.assertRaises(RuntimeError):
            transplant._check_native_branch(check, perturbed, hidden[:, CONTEXT_STEPS], m_onset)

    def test_each_branch_subtracts_its_own_start_and_reads_the_context_response_frozen(self):
        m_context, _ = transplant.modulation_after_steps(self.model, self.inputs, CONTEXT_STEPS)
        branch = transplant.branch_step(self.model, m_context, self.x_context, self.x_onset)
        layer = transplant.plastic_layer(self.model)
        step = (branch["m_after"] - m_context).double()
        np.testing.assert_allclose(branch["mod"], step.flatten(1).numpy())
        np.testing.assert_allclose(branch["eff_mod"],
                                   (step * layer.W.detach().double()).flatten(1).numpy())
        with torch.no_grad():
            layer.M = m_context.clone()
            _, activities, _ = self.model.forward(self.x_context)
        np.testing.assert_allclose(branch["h_context"], activities[-1].double().numpy())
        np.testing.assert_allclose(branch["hidden"],
                                   branch["h_onset"].double().numpy() - branch["h_context"],
                                   atol=1e-6)
        self.assertTrue(0.0 <= branch["clipped_fraction"] <= 1.0)
        # The step is the network's own update: large enough here to clip.
        self.assertGreater(branch["clipped_fraction"], 0.0)

    def test_paired_angles_are_zero_for_identical_steps_and_nan_for_zero_steps(self):
        steps = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        other = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        angles = transplant.paired_angles(steps, other)
        np.testing.assert_allclose(angles[:2], [0.0, 90.0])
        self.assertTrue(np.isnan(angles[2]))
        self.assertEqual(transplant._mean_angle(angles), 45.0)
        self.assertTrue(np.isnan(transplant._mean_angle(np.array([np.nan]))))

    def test_plastic_layer_rejects_networks_with_other_memory(self):
        self.assertIs(transplant.plastic_layer(self.model), self.model.mp_layers[0])
        with patch.object(self.model, "recurrent_active", True):
            with self.assertRaises(ValueError):
                transplant.plastic_layer(self.model)
        with patch.object(self.model, "mp_layers", []):
            with self.assertRaises(ValueError):
                transplant.plastic_layer(self.model)


class TransplantAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_network()
        self.inputs, self.rule_indices = _trials()
        self.results = transplant.transplant_analysis(
            self.model, self.inputs, self.rule_indices, RULES, CONTEXT_STEPS, 0.9)

    def test_every_task_pair_gets_both_hosting_directions_and_the_permutation_test(self):
        rval_dict, scatter_dict, _, _ = self.results
        n_pairs = len(RULES) * (len(RULES) - 1) // 2
        self.assertEqual(set(rval_dict), set(transplant.REPRESENTATIONS))
        for name in transplant.REPRESENTATIONS:
            scatter = scatter_dict[name]
            self.assertEqual(scatter["dists"].shape, (n_pairs,))
            self.assertEqual(scatter["angles_deg_by_host"].shape, (n_pairs, 2))
            np.testing.assert_allclose(scatter["angles_deg"],
                                       scatter["angles_deg_by_host"].mean(axis=1))
            self.assertEqual(len(scatter["pair_rules"]), n_pairs)
            self.assertEqual(scatter["pair_rules"][0], ("fdgo", "fdanti"))
            self.assertIsNone(scatter["permutation_skipped"])
            self.assertTrue(0 < scatter["permutation"]["p_value"] <= 1)
            self.assertEqual(scatter["permutation"]["n_permutations"],
                             shift.DISTANCE_ANGLE_N_PERMUTATIONS)
            self.assertIs(scatter["regression"]["through_origin"], False)
            self.assertTrue(np.all(scatter["angles_deg"] > 0))
            self.assertEqual(set(scatter["within_task_angles_deg"]), set(RULES))
            # Same-task donors differ by input noise only: their angle is the
            # floor of the cross-task angles.
            self.assertGreater(scatter["within_task_angle_deg"], 0.0)
            self.assertLess(scatter["within_task_angle_deg"], scatter["angles_deg"].min())
            self.assertEqual(scatter["n_steps_skipped"], 0)
            self.assertEqual(scatter["step"]["definition"], transplant.STEP_DEFINITIONS[name])
            self.assertTrue(-1.0 <= rval_dict[name][0] <= 1.0)

    def test_distances_are_the_m_distances_of_the_paired_trials(self):
        _, scatter_dict, _, _ = self.results
        m_context, _ = transplant.modulation_after_steps(self.model, self.inputs, CONTEXT_STEPS)
        hosts = np.flatnonzero(self.rule_indices == 0)
        donors = np.flatnonzero(self.rule_indices == 1)
        expected = (m_context[hosts] - m_context[donors]).flatten(1).double().norm(dim=1).mean()
        for name in transplant.REPRESENTATIONS:
            np.testing.assert_allclose(scatter_dict[name]["dists"][0], float(expected), rtol=1e-6)

    def test_step_norms_and_the_native_check_are_recorded(self):
        _, _, step_norms, native_check = self.results
        self.assertEqual(set(step_norms["mod"]),
                         {"initial", "native_step", "transplant_step", "decay_term",
                          "clipped_fraction_native", "clipped_fraction_transplant"})
        self.assertEqual(set(step_norms["eff_mod"]),
                         {"initial", "native_step", "transplant_step", "decay_term"})
        self.assertEqual(set(step_norms["hidden"]), {"initial", "native_step", "transplant_step"})
        self.assertAlmostEqual(step_norms["mod"]["decay_term"], 0.1 * step_norms["mod"]["initial"])
        self.assertTrue(0.0 <= step_norms["mod"]["clipped_fraction_transplant"] <= 1.0)
        self.assertGreater(step_norms["mod"]["native_step"], 0.0)
        self.assertLessEqual(max(native_check["max_abs_hidden_diff"],
                                 native_check["max_abs_m_diff"]),
                             transplant.NATIVE_CHECK_TOLERANCE)


class OutputTests(unittest.TestCase):
    """The entries and files of eval_one, eval_untrained and the batch runs,
    all under <out-dir>/transplant/."""

    aname = "everything_seed202_L21e4+hidden5+batch128+angle"

    def _prepare(self, path, untrained_init_seed=None, evaluation_seed=shift.EVALUATION_SEED):
        seed = 0 if untrained_init_seed is None else 10 + untrained_init_seed
        model = _tiny_network(seed=seed)
        inputs, rule_indices = _trials()
        return _evaluation(model, inputs, rule_indices, self.aname, evaluation_seed)

    def test_batch_runs_write_marked_results_and_figures_under_the_transplant_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / f"savednet_{self.aname}.pt"
            checkpoint.touch()
            out = root / "state_space"
            with patch.object(shift, "prepare_evaluation", side_effect=self._prepare), \
                    patch.object(shift, "_matches_target_cohort", return_value=True), \
                    patch.object(shift, "device", torch.device("cpu")), \
                    patch("builtins.print"):
                trained = transplant.run_checkpoints([str(checkpoint)], out)
                untrained = transplant.run_untrained([str(checkpoint)], out, init_seeds=(0, 1))
                other_seed = transplant.run_checkpoints([str(checkpoint)], out,
                                                        evaluation_seed=7)
            transplant_dir = out / "transplant"
            with (transplant_dir / "transplant_results.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            with (transplant_dir / "untrained" / "transplant_results.pkl").open("rb") as stream:
                saved_untrained = pickle.load(stream)
            figures = sorted(path.name for path in transplant_dir.glob("*.png"))
            untrained_figures = sorted(path.name
                                       for path in (transplant_dir / "untrained").glob("*.png"))
            other_pickle = (transplant_dir / "transplant_results_eval7.pkl").is_file()
            top_level = sorted(path.name for path in out.iterdir())
        self.assertEqual(set(trained), {self.aname})
        entry = saved[self.aname]
        self.assertEqual(entry["schema_version"], transplant.TRANSPLANT_SCHEMA_VERSION)
        self.assertEqual(entry["experiment"], transplant.EXPERIMENT)
        self.assertIs(entry["untrained"], False)
        self.assertEqual(entry["l2_info"], 1e-4)
        self.assertEqual(entry["evaluation_seed"], shift.EVALUATION_SEED)
        self.assertEqual(entry["all_rules"], RULES)
        self.assertEqual(entry["fixed_fixation_ms"], CONTEXT_STEPS * 40)
        self.assertEqual(set(entry["rval_dict"]), set(transplant.REPRESENTATIONS))
        self.assertIn("native_check", entry)
        # Another evaluation seed gets its own pickle and figure; the main
        # results keep the standard seed.
        self.assertTrue(other_pickle)
        self.assertEqual(other_seed[self.aname]["evaluation_seed"], 7)
        self.assertEqual(figures, [f"transplant_{self.aname}.png",
                                   f"transplant_{self.aname}_eval7.png"])
        # The untrained networks: own pickle and marks, l2_info None.
        self.assertEqual(set(untrained), {f"{self.aname}_init0", f"{self.aname}_init1"})
        for seed in (0, 1):
            entry = saved_untrained[f"{self.aname}_init{seed}"]
            self.assertIs(entry["untrained"], True)
            self.assertEqual(entry["init_seed"], seed)
            self.assertEqual(entry["config_source"], self.aname)
            self.assertIsNone(entry["l2_info"])
            self.assertEqual(entry["config_l2_info"], 1e-4)
        self.assertEqual(untrained_figures, [f"transplant_{self.aname}_init0.png",
                                             f"transplant_{self.aname}_init1.png"])
        # Nothing is written at the top level of the state-space directory.
        self.assertEqual(top_level, ["transplant"])

    def test_results_names_follow_the_evaluation_seed(self):
        self.assertEqual(transplant.results_name(shift.EVALUATION_SEED), "transplant_results.pkl")
        self.assertEqual(transplant.results_name(7), "transplant_results_eval7.pkl")
        self.assertEqual(transplant.transplant_dir_of(), shift.STATE_SPACE_DIR / "transplant")
        self.assertEqual(transplant.untrained_dir_of("x"), Path("x") / "transplant" / "untrained")

    def test_cli_untrained_needs_a_checkpoint_and_owns_the_init_seed(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                transplant._parse_args(["--untrained"])
            with self.assertRaises(SystemExit):
                transplant._parse_args(["--checkpoint", "a.pt", "--init-seed", "4"])
        args = transplant._parse_args([])
        self.assertEqual((args.checkpoint, args.untrained, args.init_seed, args.evaluation_seed),
                         (None, False, [shift.UNTRAINED_INIT_SEED], shift.EVALUATION_SEED))
        with patch.object(transplant, "run_untrained", return_value={}) as run_untrained, \
                patch.object(transplant, "run_checkpoints", return_value={}) as run_checkpoints, \
                patch.object(transplant, "run_all", return_value={}) as run_all, \
                patch("builtins.print"):
            transplant.main(["--untrained", "--checkpoint", "a.pt", "--init-seed", "3", "4",
                             "--evaluation-seed", "9", "--out-dir", "scratch"])
            transplant.main(["--checkpoint", "a.pt"])
            transplant.main([])
        self.assertEqual(run_untrained.call_args.args, (["a.pt"], Path("scratch"), [3, 4], 9))
        self.assertEqual(run_checkpoints.call_args.args,
                         (["a.pt"], shift.STATE_SPACE_DIR, shift.EVALUATION_SEED))
        self.assertEqual(run_all.call_args.args, (shift.STATE_SPACE_DIR, shift.EVALUATION_SEED))


def _transplant_entry(offset=0.0, schema=transplant.TRANSPLANT_SCHEMA_VERSION):
    """A results entry as state_space_transplant.py saves it, ten task pairs."""
    rng = np.random.default_rng(0)
    entry = {"l2_info": 1e-4, "schema_version": schema, "experiment": transplant.EXPERIMENT,
             "rval_dict": {}, "scatter": {}}
    for index, name in enumerate(transplant.REPRESENTATIONS):
        x = rng.uniform(1, 3, 10)
        entry["rval_dict"][name] = (0.9 - 0.1 * index - offset, 10.0, 0.001)
        entry["scatter"][name] = {
            "dists": x, "angles_deg": 10 * x,
            "regression": {"through_origin": False, "intercept": 1.0},
            "permutation": {"p_value": 0.02 if index % 2 == 0 else 0.3},
            "within_task_angle_deg": 1.0,
        }
    return entry


class TransplantFigureTests(unittest.TestCase):
    def _draw(self, result_dict, untrained, selected="run0"):
        with tempfile.TemporaryDirectory() as tmp:
            trained_path = Path(tmp) / "transplant_results.pkl"
            untrained_path = Path(tmp) / "untrained_results.pkl"
            with trained_path.open("wb") as stream:
                pickle.dump(result_dict, stream)
            if untrained is not None:
                with untrained_path.open("wb") as stream:
                    pickle.dump(untrained, stream)
            with patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "STATE_SPACE_TRANSPLANT_RVAL_PATH", trained_path), \
                    patch.object(paper_plot, "STATE_SPACE_TRANSPLANT_UNTRAINED_RVAL_PATH",
                                 untrained_path), \
                    patch.object(paper_plot, "_load_best_state_space_trial",
                                 return_value=(selected, 0.1, {})), \
                    patch.object(paper_plot, "_save_fig") as save_fig, \
                    patch("builtins.print") as output:
                paper_plot.plot_state_space_transplant_r_values()
        if save_fig.called:
            self.addCleanup(paper_plot.plt.close, save_fig.call_args.args[0])
        return save_fig, output

    def test_figure_draws_three_bars_per_cohort_and_the_untrained_panel(self):
        result_dict = {f"run{index}": _transplant_entry() for index in range(2)}
        untrained = {}
        for seed in (0, 1):
            entry = _transplant_entry(offset=0.5)
            entry.update(untrained=True, init_seed=seed, config_source="run0", l2_info=None)
            untrained[f"run0_init{seed}"] = entry
        save_fig, _ = self._draw(result_dict, untrained)
        figure = save_fig.call_args.args[0]
        self.assertEqual(Path(save_fig.call_args.args[1]).name,
                         "multitask_state_space_transplant_r_values.png")
        self.assertEqual(len(figure.axes), len(paper_plot.STATE_SPACE_L2_STRENGTHS) + 1)
        self.assertEqual(figure.axes[0].get_title(), "Untrained")
        self.assertEqual(figure.axes[0].get_ylabel(), "R-value")
        panel = figure.axes[1 + list(paper_plot.STATE_SPACE_L2_STRENGTHS).index(1e-4)]
        self.assertEqual(panel.get_title(), r"$L_2 = 10^{-4}$")
        bars = [patch_ for patch_ in panel.patches
                if isinstance(patch_, paper_plot.mpl.patches.Rectangle)]
        np.testing.assert_allclose([bar.get_height() for bar in bars], [0.9, 0.8, 0.7])
        self.assertEqual([bar.get_hatch() for bar in bars], [None, None, None])
        self.assertEqual([text.get_text() for text in panel.get_xticklabels()],
                         ["Hidden", "Mod.", "Eff.\nMod."])
        untrained_bars = [patch_ for patch_ in figure.axes[0].patches
                          if isinstance(patch_, paper_plot.mpl.patches.Rectangle)]
        np.testing.assert_allclose([bar.get_height() for bar in untrained_bars], [0.4, 0.3, 0.2],
                                   atol=1e-12)
        # Two runs per bar; hidden and eff_mod are significant (filled).
        dots = [collection for collection in panel.collections
                if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
        faces = [tuple(collection.get_facecolor()[0][:3]) for collection in dots]
        self.assertEqual(faces, [(0, 0, 0)] * 2 + [(1, 1, 1)] * 2 + [(0, 0, 0)] * 2)

    def test_figure_without_untrained_results_names_the_transplant_command(self):
        result_dict = {"run0": _transplant_entry()}
        save_fig, output = self._draw(result_dict, None)
        figure = save_fig.call_args.args[0]
        self.assertEqual(len(figure.axes), len(paper_plot.STATE_SPACE_L2_STRENGTHS))
        self.assertIn("state_space_transplant.py --untrained", output.call_args_list[0].args[0])

    def test_entries_below_the_transplant_schema_are_left_out(self):
        result_dict = {"old": _transplant_entry(schema=0), "run0": _transplant_entry()}
        save_fig, output = self._draw(result_dict, None)
        panel = save_fig.call_args.args[0].axes[list(paper_plot.STATE_SPACE_L2_STRENGTHS).index(1e-4)]
        dots = [collection for collection in panel.collections
                if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
        self.assertEqual(len(dots), 3)
        self.assertTrue(any("state_space_transplant.py" in call.args[0]
                            for call in output.call_args_list))


if __name__ == "__main__":
    unittest.main()
