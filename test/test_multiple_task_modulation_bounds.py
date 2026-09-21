"""Checks for multi-task modulation-bound configuration and run naming."""

import ast
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "multiple_task" / "multiple_task.py"


def _load_bound_helpers():
    tree = ast.parse(SOURCE.read_text())
    names = {"_normalize_m_bounds", "_addon_name_with_bounds"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError(f"Missing modulation-bound helpers in {SOURCE}")
    namespace = {"np": np, "M_BOUNDS": (-1.0, 1.0)}
    exec(  # noqa: S102 -- execute selected functions from trusted local source
        compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace


class MultipleTaskModulationBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_bound_helpers()

    def test_default_bounds_preserve_legacy_addon_name(self):
        self.assertEqual(
            self.helpers["_addon_name_with_bounds"]("L21e4proj300", (-1, 1)),
            "L21e4proj300",
        )

    def test_nondefault_bounds_are_encoded_like_pretraining(self):
        add_bounds = self.helpers["_addon_name_with_bounds"]
        self.assertEqual(add_bounds("L21e4proj300", (-2, 2)), "L21e4proj300mb2")
        self.assertEqual(add_bounds("L21e4proj300", (-0.5, 2)),
                         "L21e4proj300mb-0.5to2")

    def test_invalid_bounds_fail_before_training(self):
        normalize = self.helpers["_normalize_m_bounds"]
        for bounds in ((1,), (1, -1), (0, np.inf), "bad"):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                normalize(bounds)

    def test_bound_is_wired_to_network_and_effective_run_name(self):
        source = SOURCE.read_text()
        tree = ast.parse(source)
        current_params = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "current_basic_params"
        )
        run_trial = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_trial"
        )
        self.assertIn("'m_bounds': _normalize_m_bounds(M_BOUNDS)",
                      ast.unparse(current_params))
        self.assertIn("run_addon_name = _addon_name_with_bounds(ADDON_NAME)",
                      ast.unparse(run_trial))
        self.assertIn("'addon_name': run_addon_name + f'+hidden{N_HIDDEN}'",
                      ast.unparse(run_trial))

    def test_selected_run_uses_regular_dimensions_and_l2_with_wide_bounds(self):
        tree = ast.parse(SOURCE.read_text())
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"N_HIDDEN", "ADDON_NAME", "M_BOUNDS"}
        }
        self.assertEqual(assignments, {
            "N_HIDDEN": 300,
            "ADDON_NAME": "L21e4",
            "M_BOUNDS": (-2.0, 2.0),
        })

        current_params = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "current_basic_params"
        )
        params_source = ast.unparse(current_params)
        self.assertIn("'reg_lambda': 0.0001", params_source)
        self.assertIn("'linear_embed': 300", params_source)
        self.assertIn("'activation': 'tanh'", params_source)


if __name__ == "__main__":
    unittest.main()
