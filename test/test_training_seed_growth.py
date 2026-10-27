"""Every training script names its runs through one function that the seed
scan reuses, so a batch adds seeds new to the configured setting and never
overwrites a saved run of it."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import seed_selection as selection
from script_namespace import load_script_namespace


ROOT = Path(__file__).resolve().parents[1]
NAMING = ["_hyp_dict", "run_name", "setting_run_name"]
SCRIPTS = {
    "multiple_task": {
        "source": ROOT / "multiple_task" / "multiple_task.py",
        "functions": ["_normalize_m_bounds", "_addon_name_with_bounds", "_addon_name_with_lambda",
                      "effective_addon_name", "validate_reg_lambda_addon_name",
                      "current_basic_params", "full_addon_name"] + NAMING,
        "constants": ["RULESET", "CHOSEN_NETWORK", "N_HIDDEN", "n_hidden", "ADDON_NAME", "M_BOUNDS",
                      "DT_MS", "M_TIME_SCALE", "LEGACY_M_LAMBDA",
                      "train", "rules_dict", "rules_dict_frequency", "mpn_depth"],
        "name_regex": r"^everything_seed7_L21e4\+hidden300\+batch128\+angle$",
        "more_stimulus": False, "per_run_directory": False,
    },
    "one_task": {
        "source": ROOT / "one_task" / "one_task.py",
        "functions": ["current_basic_params"] + NAMING,
        "constants": ["RULESET", "CHOSEN_NETWORK", "ADDON_NAME", "train", "rules_dict", "mpn_depth"],
        "name_regex": r"^delaygo_seed7_L2\de\d\+hidden\d+\+batch\d+\+\w+$",
        "more_stimulus": True, "per_run_directory": False,
    },
    "one_task_rnn": {
        "source": ROOT / "one_task" / "one_task_rnn.py",
        "functions": ["current_basic_params"] + NAMING,
        "constants": ["RULESET", "CHOSEN_NETWORK", "ADDON_NAME", "train", "rules_dict"],
        "name_regex": r"^delaygo_seed7_rnnL2\de\d\+hidden\d+\+batch\d+\+\w+$",
        "more_stimulus": True, "per_run_directory": False,
    },
    "two_task": {
        "source": ROOT / "two_task" / "two_task.py",
        "functions": NAMING,
        "constants": ["RULESET", "CHOSEN_NETWORK", "N_HIDDEN", "ADDON_NAME"],
        "name_regex": r"^delaygofamily_seed7_reg1e3\+hidden200$",
        "more_stimulus": True, "per_run_directory": True,
    },
}


def _run_name(spec, **kwargs):
    namespace = load_script_namespace(spec["source"], spec["functions"], spec["constants"])
    with patch("builtins.print"):                      # the parameter builders print their settings
        return namespace["setting_run_name"](**kwargs)


def _mark(root, name, prefix, suffix, per_run_directory):
    if per_run_directory:
        (root / name).mkdir(exist_ok=True)
        (root / name / f"{prefix}{name}{suffix}").write_bytes(b"")
    else:
        (root / f"{prefix}{name}{suffix}").write_bytes(b"")


class TrainingSeedGrowthTests(unittest.TestCase):
    def test_every_script_names_its_setting_the_saved_way(self):
        for script, spec in SCRIPTS.items():
            with self.subTest(script=script):
                run_name = _run_name(spec)
                self.assertRegex(run_name(7), spec["name_regex"])
                self.assertEqual(selection.seed_pattern(run_name).fullmatch(run_name(123)).group(1), "123")
                if spec["more_stimulus"]:
                    denser = _run_name(spec, more_stimulus=True)
                    self.assertEqual(denser(7), run_name(7) + "+morestimulus")
                    self.assertIsNone(selection.seed_pattern(run_name).fullmatch(denser(7)))

    def test_every_script_skips_its_saved_and_started_seeds_only(self):
        for script, spec in SCRIPTS.items():
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_name = _run_name(spec)
                per_run = spec["per_run_directory"]
                _mark(root, run_name(5), *selection.COMPLETED_FILE, per_run)
                _mark(root, run_name(5), *selection.STARTED_FILE, per_run)
                _mark(root, run_name(9), *selection.STARTED_FILE, per_run)
                other = run_name(5) + ("+morestimulus" if spec["more_stimulus"] else "+other")
                _mark(root, other, *selection.COMPLETED_FILE, per_run)
                _mark(root, other.replace("seed5", "seed6"), *selection.COMPLETED_FILE, per_run)
                with patch("builtins.print") as output:
                    seeds = selection.choose_seeds(3, None, root, run_name, per_run_directory=per_run,
                                                   pool=range(1, 12), rng=random.Random(0))
                    with self.assertRaisesRegex(ValueError, r"seeds \[5\]"):
                        selection.choose_seeds(3, [5], root, run_name, per_run_directory=per_run)
                    self.assertEqual(selection.choose_seeds(3, [6, 9], root, run_name,
                                                            per_run_directory=per_run), [6, 9])
                self.assertEqual(len(set(seeds)), 3)
                self.assertTrue({5, 9}.isdisjoint(seeds))
                printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
                self.assertIn("completed seeds [5], started but unfinished [9]", printed)


if __name__ == "__main__":
    unittest.main()
