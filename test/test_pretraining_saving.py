"""Saving-contract tests without importing the trainer's CUDA startup code."""

import ast
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"Missing test target in {path}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class PretrainingSavingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.producer = load_functions(
            "pretrain/pretraining.py", ["_modulation_extraction"], {"np": np})
        cls.post = load_functions(
            "pretrain/pretraining_utils.py", ["load_task_params"],
            {"np": np, "copy": copy})
        cls.analysis = load_functions(
            "pretrain/pretraining_analysis.py",
            ["_stage1_end_iteration", "period_slice"], {"np": np})

    def test_extraction_preserves_values_layout_and_copies(self):
        for trials in (3, 6):
            for layer_index in (0, 1):
                with self.subTest(trials=trials, layer_index=layer_index):
                    modulation = np.arange(trials * 5 * 12, dtype=np.float32).reshape(trials, 5, 3, 4)
                    hidden = np.arange(trials * 5 * 3, dtype=np.float32).reshape(trials * 5, 3)
                    saved_modulation, saved_hidden = self.producer["_modulation_extraction"](
                        {f"M{layer_index}": modulation, f"hidden{layer_index}": hidden},
                        5, layer_index, trials)
                    np.testing.assert_array_equal(saved_modulation, modulation)
                    np.testing.assert_array_equal(saved_hidden, hidden.reshape(trials, 5, 3))
                    self.assertEqual(saved_modulation.dtype, modulation.dtype)
                    self.assertEqual(saved_hidden.dtype, hidden.dtype)
                    self.assertFalse(np.shares_memory(saved_modulation, modulation))
                    self.assertFalse(np.shares_memory(saved_hidden, hidden))

    def test_vanilla_extraction_and_unsupported_type(self):
        for trials in (3, 6):
            hidden = np.arange(trials * 5 * 3, dtype=np.float32).reshape(trials * 5, 3)
            modulation, saved_hidden = self.producer["_modulation_extraction"](
                {"hidden": hidden}, 5, 1, trials, nettype="vanilla")
            self.assertIsNone(modulation)
            np.testing.assert_array_equal(saved_hidden, hidden.reshape(trials, 5, 3))
        with self.assertRaisesRegex(ValueError, "Unsupported nettype"):
            self.producer["_modulation_extraction"]({}, 5, 1, 6, nettype="unknown")

    def test_saved_npz_contract_and_downstream_reads(self):
        tree = ast.parse((ROOT / "pretrain/pretraining.py").read_text())
        trial = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "run_trial")
        save_calls = [node for node in ast.walk(trial)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and ast.unparse(node.func) == "np.savez_compressed"]
        expected = {
            "stage1_output_path": {"test_input_np", "test_output_np", "rules_epochs", "task_params", "test_task"},
            "stage2_output_path": {"test_input_np", "test_output_np", "rules_epochs2", "task_params", "test_task"},
            "result_path": {"Ms_orig_stage1", "Ms_orig_stage2", "hs_stage1", "hs_stage2",
                            "valid_acc_iter", "valid_acc", "stage1_end_iter", "pretrain_stop"},
        }
        actual = {ast.unparse(node.args[0]): {keyword.arg for keyword in node.keywords}
                  for node in save_calls}
        self.assertEqual(actual, expected)
        rng = np.random.default_rng(8)
        namespace = {
            "np": np,
            "test_input_np": rng.normal(size=(6, 5, 9)).astype(np.float32),
            "test_input2_np": rng.normal(size=(3, 5, 9)).astype(np.float32),
            "test_output_np": rng.normal(size=(6, 5, 3)).astype(np.float32),
            "test_output2_np": rng.normal(size=(3, 5, 3)).astype(np.float32),
            "rules_epochs": {"fdanti": {"stim1": (1, 3)}, "delaygo": {"go1": (3, 5)}},
            "rules_epochs2": {"delayanti": {"stim1": (1, 3), "go1": (3, 5)}},
            "task_params": {"rules": ["fdanti", "delaygo"], "dt": 40, "hp": {"batch_size_train": 6}},
            "task_params2": {"rules": ["delayanti"], "dt": 40, "hp": {"batch_size_train": 3}},
            "test_task": [0, 1, 0, 1, 0, 1],
            "test_task2": [0, 0, 0],
            "pretrain_stop": 9,
            "stage1_end_iter": 10,
            "net": SimpleNamespace(hist={"iters_monitor": [0, 5, 10, 10, 15, 20],
                                         "valid_acc": [0.0, 0.5, 0.8, 0.8, 0.9, 1.0]}),
        }
        for stage, trials in ((1, 6), (2, 3)):
            namespace[f"Ms_orig_stage{stage}"] = rng.normal(size=(trials, 5, 3, 4)).astype(np.float32)
            namespace[f"hs_stage{stage}"] = rng.normal(size=(trials, 5, 3)).astype(np.float32)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            namespace.update(stage1_output_path=root / "output_test_stage1.npz",
                             stage2_output_path=root / "output_test_stage2.npz",
                             result_path=root / "result.npz")
            for call in save_calls:
                statement = ast.Expr(value=call)
                module = ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[]))
                exec(compile(module, "<trainer-save>", "exec"), namespace)
                path_key = ast.unparse(call.args[0])
                with np.load(namespace[path_key], allow_pickle=True) as saved:
                    self.assertEqual(set(saved.files), expected[path_key])
                    for keyword in call.keywords:
                        value = eval(compile(ast.Expression(keyword.value), "<saved-value>", "eval"), namespace)
                        if isinstance(value, dict):
                            self.assertEqual(saved[keyword.arg].item(), value)
                        else:
                            np.testing.assert_array_equal(saved[keyword.arg], value)
                            self.assertEqual(saved[keyword.arg].dtype, np.asarray(value).dtype)
            for stage, params_key in ((1, "task_params"), (2, "task_params2")):
                params = self.post["load_task_params"](root, "test", f"stage{stage}")
                self.assertEqual(params, namespace[params_key])
                params["hp"]["batch_size_train"] = 1
                self.assertEqual(self.post["load_task_params"](root, "test", f"stage{stage}"), namespace[params_key])
            with np.load(namespace["result_path"], allow_pickle=True) as saved:
                boundary = self.analysis["_stage1_end_iteration"](saved)
                self.assertEqual(boundary, 10)
                post_mask = saved["valid_acc_iter"] > boundary
                np.testing.assert_array_equal(saved["valid_acc_iter"][post_mask] - boundary, [5, 10])
                np.testing.assert_array_equal(saved["valid_acc"][post_mask], [0.9, 1.0])
                with np.load(namespace["stage1_output_path"], allow_pickle=True) as stage1:
                    selected = self.analysis["period_slice"](
                        saved["hs_stage1"], stage1["rules_epochs"].item(), "fdanti", "stim1",
                        mask=stage1["test_task"] == 0)
                    np.testing.assert_array_equal(selected, namespace["hs_stage1"][[0, 2, 4], 1:3])


if __name__ == "__main__":
    unittest.main()
