"""Regression tests for the focused state-space checkpoint cohort."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
import state_space_shift as state_space


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


if __name__ == "__main__":
    unittest.main()
