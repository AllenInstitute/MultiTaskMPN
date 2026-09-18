"""Static CLI-contract checks for the pretraining post-analysis driver."""

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pretrain import pretraining_post, pretraining_utils


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pretrain" / "pretraining_post.py"


class PretrainingPostCLITests(unittest.TestCase):
    def test_ruleset_order_and_groups_include_delaypro(self):
        self.assertEqual(
            pretraining_post.POST_RULESET_ORDER,
            ("fdanti_delaygo", "fdgo_delaygo", "fdanti", "fdgo"),
        )
        self.assertEqual(
            pretraining_post.GROUPS,
            {
                "fdanti_delaygo": (
                    "Proper motif", ("fdanti", "delaygo", "delayanti")),
                "fdgo_delaygo": (
                    "Improper motif", ("fdgo", "delaygo", "delayanti")),
                "fdanti": ("DelayAnti", ("fdanti", "delayanti")),
                "fdgo": ("DelayPro", ("fdgo", "delayanti")),
            },
        )
        self.assertGreaterEqual(
            len(pretraining_post.COLORS), len(pretraining_post.GROUPS))

    def test_checkpoint_discovery_preserves_stable_seed_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ruleset in pretraining_post.POST_RULESET_ORDER:
                for seed in range(10, 20):
                    aname = pretraining_utils.run_name(
                        ruleset, "dmpn", seed, 200, "L21e3")
                    (root / f"savednet_{aname}.pt").touch()
            decoy = pretraining_utils.run_name(
                "fdanti_delaygo", "dmpn", 999, 100, "L21e3")
            (root / f"savednet_{decoy}.pt").touch()

            matches = pretraining_post.discover_checkpoints(
                root, "L21e3", 200, total_seed=3, selection_seed=17)

        selected = {
            ruleset: [seed for _, match_ruleset, seed in matches
                      if match_ruleset == ruleset]
            for ruleset in pretraining_post.POST_RULESET_ORDER
        }
        self.assertEqual(selected, {
            "fdanti_delaygo": [14, 17, 19],
            "fdgo_delaygo": [10, 11, 17],
            "fdanti": [11, 17, 18],
            "fdgo": [16, 18, 19],
        })

    def test_backbone_direction_sweep_has_no_duplicate_rays(self):
        one_dimensional = pretraining_post._backbone_direction_coefficients(1)
        self.assertEqual(one_dimensional, ((-1.0,), (0.0,), (1.0,)))

        two_dimensional = np.asarray(
            pretraining_post._backbone_direction_coefficients(2))
        self.assertEqual(
            two_dimensional.shape,
            (len(pretraining_post.BACKBONE_DIRECTION_ANGLES_DEG), 2),
        )
        np.testing.assert_allclose(
            np.linalg.norm(two_dimensional, axis=1), 1.0, atol=1e-12)
        rounded = np.round(two_dimensional, decimals=12)
        self.assertEqual(np.unique(rounded, axis=0).shape[0], len(rounded))

    def test_backbone_direction_plot_supports_one_and_two_rule_spans(self):
        runs = []
        for ruleset, (_, tasks) in pretraining_post.GROUPS.items():
            n_pretraining = len(tasks) - 1
            directions = pretraining_post._backbone_direction_coefficients(
                n_pretraining)
            if n_pretraining == 1:
                angles = None
                labels = ["negative", "zero", "positive"]
                accuracies = [20.0, 10.0, 70.0]
                projection_coefficients = [1.0]
            else:
                angles = list(pretraining_post.BACKBONE_DIRECTION_ANGLES_DEG)
                labels = None
                accuracies = (50.0 + 30.0 * np.cos(np.deg2rad(angles))).tolist()
                projection_coefficients = [1.0, 1.0]
            runs.append({
                "aname": f"synthetic_{ruleset}",
                "ruleset": ruleset,
                "span_direction_sweep_norm_matched": {
                    "n_pretraining": n_pretraining,
                    "target_norm": 1.0,
                    "angles_deg": angles,
                    "direction_labels": labels,
                    "coefficient_directions": [list(values) for values in directions],
                    "accuracy_pct": accuracies,
                },
                "named_points": {"learned_projection": {
                    "coefficients": projection_coefficients}},
                "named_points_norm_matched": {"learned_projection": {
                    "coefficients": projection_coefficients,
                    "accuracy_pct": 75.0}},
            })

        with (tempfile.TemporaryDirectory() as directory,
              mock.patch.object(pretraining_post, "summarize_backbone_probe",
                                return_value={ruleset: {}
                                              for ruleset in pretraining_post.GROUPS})):
            path = pretraining_post.plot_backbone_direction_sweep(
                runs, Path(directory), "L21e3", 200)
            self.assertTrue(path.is_file())
            self.assertTrue(path.name.startswith(
                "backbone_span_direction_norm_matched_"))

    def test_backbone_random_plot_supports_every_group_color(self):
        runs = [{
            "aname": f"synthetic_{ruleset}",
            "ruleset": ruleset,
            "random_probe": {
                "loss": [1.0],
                "loss_out": [0.5],
                "accuracy_pct": [25.0],
            },
        } for ruleset in pretraining_post.GROUPS]

        with (tempfile.TemporaryDirectory() as directory,
              mock.patch.object(
                  pretraining_post, "summarize_backbone_probe",
                  return_value={ruleset: {}
                                for ruleset in pretraining_post.GROUPS})):
            path = pretraining_post.plot_backbone_probe_random(
                runs, Path(directory), "L21e3", 200)

        self.assertTrue(path.name.startswith("backbone_probe_random_"))

    def test_default_experiment_list_contains_every_analysis_flag(self):
        tree = ast.parse(SOURCE.read_text())
        assignments = {
            target.id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(
            ast.literal_eval(assignments["EXPERIMENTS"]),
            (
                "accuracy",
                "memory_pca",
                "rule_cue",
                "m_intervention",
                "rule_vector_intervention",
                "rule_vector_magnitude_sweep",
                "backbone_probe",
                "pathway_gain",
            ),
        )
        self.assertNotIn("EXPLICIT_ONLY_EXPERIMENTS", assignments)

    def test_memory_pca_total_seed_uses_each_selected_checkpoint(self):
        args = SimpleNamespace(
            rule_vector_magnitude_sweep=False,
            rule_vector_intervention=False,
            backbone_probe=False,
            pathway_gain=False,
            memory_pca=True,
            m_intervention=False,
            rule_cue=False,
            ruleset=None,
            checkpoint_dir=Path("checkpoints"),
            output_dir=Path("figures"),
            feature="L21e3",
            hidden=200,
            seed=None,
            total_seed=2,
            test_seed=17,
        )

        def selected_checkpoints(_root, _feature, _hidden, ruleset, **kwargs):
            self.assertEqual(kwargs, {"total_seed": 2, "selection_seed": 17})
            return [(Path(f"{ruleset}_{seed}.pt"), ruleset, seed)
                    for seed in (11, 29)]

        with (mock.patch.object(pretraining_post, "discover_checkpoints",
                                side_effect=selected_checkpoints),
              mock.patch.object(pretraining_post, "plot_memory_pca") as plot):
            pretraining_post._run_analysis(args, device="cpu")

        self.assertEqual(plot.call_count, 2 * len(pretraining_post.GROUPS))
        for ruleset in pretraining_post.GROUPS:
            for seed in (11, 29):
                plot.assert_any_call(
                    args.checkpoint_dir, args.output_dir, args.feature,
                    args.hidden, seed, args.test_seed, ruleset, device="cpu")


if __name__ == "__main__":
    unittest.main()
