"""CLI-contract checks for the pretraining analysis driver."""

import argparse
import ast
import unittest
from pathlib import Path

from pretrain import pretraining_utils

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pretrain" / "pretraining_analysis.py"


def _load_functions(*names):
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        missing = set(names) - {node.name for node in functions}
        raise AssertionError(f"Missing test target(s): {missing}")
    namespace = {
        "argparse": argparse,
        "positive_int": pretraining_utils.positive_int,
    }
    exec(  # noqa: S102 -- execute selected functions from trusted local source
        compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace


class PretrainingAnalysisCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = _load_functions("_parse_args")

    def test_default_variant_matches_pretraining_post(self):
        args = self.functions["_parse_args"]([])
        self.assertEqual(args.hidden, 200)
        self.assertEqual(args.feature, "L21e3")
        self.assertEqual(
            pretraining_utils.variant_addon(args.hidden, args.feature),
            "+hidden200+L21e3+batch128+angle",
        )

    def test_ruleset_order_includes_both_single_task_controls(self):
        tree = ast.parse(SOURCE.read_text())
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id == "ANALYSIS_RULESET_ORDER"
                    for target in node.targets)
        )
        self.assertEqual(
            ast.literal_eval(assignment.value),
            ("fdgo_delaygo", "fdanti_delaygo", "fdanti", "fdgo"),
        )

    def test_custom_hidden_and_feature(self):
        args = self.functions["_parse_args"]([
            "--hidden", "96", "--feature", "L21e4",
            "--total-seed", "3", "--test-seed", "17",
        ])
        self.assertEqual(args.hidden, 96)
        self.assertEqual(args.feature, "L21e4")
        self.assertEqual(args.total_seed, 3)
        self.assertEqual(args.test_seed, 17)
        self.assertEqual(
            pretraining_utils.variant_addon(args.hidden, args.feature),
            "+hidden96+L21e4+batch128+angle",
        )

    def test_hidden_must_be_positive(self):
        with self.assertRaises(SystemExit):
            self.functions["_parse_args"](["--hidden", "0"])


if __name__ == "__main__":
    unittest.main()
