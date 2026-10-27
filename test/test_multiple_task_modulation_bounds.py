"""Checks for multi-task modulation-bound configuration and run naming."""

import ast
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "multiple_task" / "multiple_task.py"


def _load_bound_helpers():
    tree = ast.parse(SOURCE.read_text())
    names = {"_normalize_m_bounds", "_addon_name_with_bounds", "_addon_name_with_lambda"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError(f"Missing modulation-bound helpers in {SOURCE}")
    namespace = {"np": np, "M_BOUNDS": (-1.0, 1.0), "DT_MS": 40, "M_TIME_SCALE": 4000,
                 "LEGACY_M_LAMBDA": 0.99}
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

    def test_lambda_099_keeps_the_legacy_name_and_others_are_tagged(self):
        add_lambda = self.helpers["_addon_name_with_lambda"]
        self.assertEqual(add_lambda("L21e4", 4000, 40), "L21e4")
        self.assertEqual(add_lambda("L21e4"), "L21e4")
        self.assertEqual(add_lambda("L21e4", 400, 40), "L21e4lam0.9")
        self.assertEqual(add_lambda("L21e4mb2", 400, 40), "L21e4mb2lam0.9")
        for m_time_scale in (20, np.nan, np.inf):
            with self.subTest(m_time_scale=m_time_scale), self.assertRaises(ValueError):
                add_lambda("L21e4", m_time_scale, 40)

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
        hyp_dict = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_hyp_dict"
        )
        effective = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "effective_addon_name"
        )
        setting = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "setting_run_name"
        )
        self.assertIn("'m_bounds': _normalize_m_bounds(M_BOUNDS)",
                      ast.unparse(current_params))
        self.assertIn("'m_time_scale': M_TIME_SCALE", ast.unparse(current_params))
        self.assertIn("'dt': DT_MS", ast.unparse(current_params))
        self.assertIn("return _addon_name_with_lambda(_addon_name_with_bounds(ADDON_NAME))",
                      ast.unparse(effective))
        self.assertIn("run_addon_name = effective_addon_name()", ast.unparse(run_trial))
        self.assertIn("hyp_dict = _hyp_dict(effective_addon_name())", ast.unparse(setting))
        self.assertIn("hyp_dict = _hyp_dict(run_addon_name)", ast.unparse(run_trial))
        self.assertIn("'addon_name': run_addon_name + f'+hidden{N_HIDDEN}'",
                      ast.unparse(hyp_dict))

    def test_selected_run_reproduces_the_saved_l21e4_cohort(self):
        """The script is set to extend the saved L21e4 cohort: its dimensions,
        L2 strength, default (-1, 1) bounds and lambda = 0.99, drawing seeds
        new to that setting (SEED_LIST None; core/seed_selection.py guards
        against overwriting a checkpoint, see test_training_seed_growth)."""
        tree = ast.parse(SOURCE.read_text())
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"N_HIDDEN", "ADDON_NAME", "M_BOUNDS", "SEED_LIST",
                              "DT_MS", "M_TIME_SCALE"}
        }
        self.assertEqual(assignments, {
            "N_HIDDEN": 300,
            "ADDON_NAME": "L21e4",
            "M_BOUNDS": (-1.0, 1.0),
            "SEED_LIST": None,
            "DT_MS": 40,
            "M_TIME_SCALE": 4000,
        })
        self.assertEqual(
            self.helpers["_addon_name_with_bounds"]("L21e4", assignments["M_BOUNDS"]), "L21e4")
        self.assertEqual(
            self.helpers["_addon_name_with_lambda"](
                "L21e4", assignments["M_TIME_SCALE"], assignments["DT_MS"]), "L21e4")

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
