#!/usr/bin/env python
"""Decay and filename contracts without importing the CUDA-starting trainer."""

import ast
import copy
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from test_pretraining_saving import load_functions

from pretrain import (
    _bootstrap,  # noqa: F401
    pretraining_post,
    pretraining_utils,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pretrain" / "pretraining.py"


def load_producer():
    tree = ast.parse(SOURCE.read_text())
    constants = {
        "DT_MS", "M_TIME_SCALE", "LEGACY_M_LAMBDA", "M_BOUNDS",
        "RULES_DICT", "RULES_DICT_FREQUENCY",
    }
    functions = {
        "_feature_with_bounds", "_feature_with_lambda", "_current_basic_params",
        "_build_experiment_hyp_dicts", "_build_file_tag",
    }
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in functions)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in constants
                         for target in node.targets))]
    namespace = {"np": np, "copy": copy}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace


class PretrainingDecayTests(unittest.TestCase):
    def setUp(self):
        self.producer = load_producer()

    def feature(self, bounds, time_scale):
        feature = self.producer["_feature_with_bounds"]("L21e3", bounds)
        return self.producer["_feature_with_lambda"](
            feature, dt=40, m_time_scale=time_scale)

    def test_legacy_lambda_preserves_names_for_both_bounds(self):
        self.assertEqual(self.feature((-1, 1), 4000), "L21e3")
        self.assertEqual(self.feature((-2, 2), 4000), "L21e3mb2")

    def test_nonlegacy_lambda_tags_are_distinct_and_use_actual_dt(self):
        self.assertEqual(self.feature((-1, 1), 400), "L21e3lam0.9")
        self.assertEqual(self.feature((-2, 2), 400), "L21e3mb2lam0.9")
        self.assertEqual(self.feature((-2, 2), 800), "L21e3mb2lam0.95")
        self.assertEqual(
            self.producer["_feature_with_lambda"]("L21e3", dt=20, m_time_scale=400),
            "L21e3lam0.95")
        self.assertEqual(self.feature((-1, 1), 40), "L21e3lam0")

    def test_invalid_time_configuration_is_rejected(self):
        for dt, time_scale in ((0, 400), (-1, 400), (np.nan, 400),
                               (np.inf, 400), (40, 0), (40, 20),
                               (40, np.nan), (40, np.inf)):
            with self.subTest(dt=dt, time_scale=time_scale), self.assertRaises(ValueError):
                self.producer["_feature_with_lambda"](
                    "L21e3", dt=dt, m_time_scale=time_scale)

    def test_run_trial_labels_both_stages_and_matches_consumers(self):
        tree = ast.parse(SOURCE.read_text())
        trial = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "run_trial")
        assignment = next(node for node in trial.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "feature"
                                  for target in node.targets))
        namespace = {**self.producer, "feature": "L21e3"}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), "exec"), namespace)  # noqa: S102
        feature = namespace["feature"]
        self.assertEqual(feature, "L21e3mb2lam0.9")
        for ruleset in pretraining_utils.RULESET_SPECS:
            stage1, stage2 = self.producer["_build_experiment_hyp_dicts"](
                feature, ruleset, "delayanti", n_hidden=200, chosen_network="dmpn")
            for stage in (stage1, stage2):
                with redirect_stdout(io.StringIO()):
                    task, train, model = self.producer["_current_basic_params"](
                        stage, train=True, n_hidden=200, mpn_depth=1)
                self.assertEqual(task["dt"], 40)
                self.assertEqual(model["ml_params"]["m_time_scale"], 400)
                self.assertFalse(model["ml_params"]["lam_train"])
                self.assertEqual(model["ml_params"]["m_bounds"], (-2.0, 2.0))
                stage["addon_name"] += f"+batch{train['n_batches']}+{model['acc_measure']}"
                self.assertEqual(stage["addon_name"], pretraining_utils.variant_addon(200, feature))
            self.assertEqual(
                self.producer["_build_file_tag"](stage1, stage2, 123),
                pretraining_utils.run_name(ruleset, "dmpn", 123, 200, feature))

    def test_real_plastic_layer_has_configured_fixed_lambda(self):
        from mpn import MultiPlasticLayer

        for time_scale, expected in ((400, 0.9), (4000, 0.99)):
            self.producer["M_TIME_SCALE"] = time_scale
            with redirect_stdout(io.StringIO()):
                task, _, model = self.producer["_current_basic_params"](
                    {"task_type": "multitask", "ruleset": "fdgo", "chosen_network": "dmpn"},
                    train=True, n_hidden=4, mpn_depth=1)
            plastic = {**model["ml_params"], "dt": task["dt"], "n_input": 4, "n_output": 4}
            layer = MultiPlasticLayer(plastic, output_matrix="", verbose=False)
            self.assertAlmostEqual(float(layer.state_dict()["lam"]), expected, places=6)
            self.assertFalse(layer.lam.requires_grad)
            self.assertEqual(layer.m_time_scale, time_scale)

    def test_downstream_discovery_separates_legacy_and_new_variants(self):
        import os
        import re

        features = ("L21e3", "L21e3mb2", "L21e3lam0.9", "L21e3mb2lam0.9")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, feature in enumerate(features, start=1):
                name = pretraining_utils.run_name("fdgo", "dmpn", seed, 200, feature)
                (root / f"param_{name}_result.npz").touch()
                (root / f"savednet_{name}.pt").touch()
            for seed, feature in enumerate(features, start=1):
                analysis = load_functions(
                    "pretrain/pretraining_analysis.py", ["discover_seeds"],
                    {"os": os, "re": re, "ruleset": "fdgo", "chosen_network": "dmpn",
                     "basepath": str(root), "addon_name": pretraining_utils.variant_addon(200, feature)})
                self.assertEqual(analysis["discover_seeds"](), [seed])
                matches = pretraining_post.discover_checkpoints(root, feature, 200, ruleset="fdgo")
                self.assertEqual([entry[2] for entry in matches], [seed])


if __name__ == "__main__":
    unittest.main()