"""The seed selection shared by the training batch scripts."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import seed_selection as selection


def _name(seed):
    return f"delaygo_seed{seed}_L21e3+hidden200+batch128+angle"


def _other(seed):
    return f"delaygo_seed{seed}_L21e3+hidden200+batch128+angle+morestimulus"


def _touch(directory, names):
    for name in names:
        (Path(directory) / name).write_bytes(b"")


def _printed(output):
    return "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)


class SeedPatternTests(unittest.TestCase):
    def test_pattern_captures_the_seed_and_escapes_the_rest(self):
        pattern = selection.seed_pattern(_name)
        self.assertEqual(pattern.fullmatch(_name(123)).group(1), "123")
        self.assertIsNone(pattern.fullmatch(_other(123)))
        self.assertIsNone(pattern.fullmatch("delaygo_seed12_L21e3xhidden200+batch128+angle"))
        self.assertIsNone(pattern.fullmatch(_name("x")))

    def test_the_name_must_embed_the_seed_exactly_once(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            selection.seed_pattern(lambda seed: "fixed_name")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            selection.seed_pattern(lambda seed: f"a{seed}_b{seed}")


class DirectoryScanTests(unittest.TestCase):
    def test_flat_layout_lists_seeds_with_the_marker_file(self):
        with tempfile.TemporaryDirectory() as directory:
            _touch(directory, [f"savednet_{_name(5)}.pt", f"param_{_name(5)}_param.json",
                               f"param_{_name(9)}_param.json", f"savednet_{_other(7)}.pt",
                               f"loss_{_name(5)}.png"])
            self.assertEqual(selection.seeds_with_file(directory, _name, "savednet_", ".pt"), [5])
            self.assertEqual(selection.seeds_with_file(directory, _name, "param_", "_param.json"), [5, 9])
            self.assertEqual(selection.seeds_with_file(directory, _other, "savednet_", ".pt"), [7])
            self.assertEqual(selection.seeds_with_file(Path(directory) / "missing", _name, "savednet_", ".pt"), [])

    def test_per_run_directories_list_seeds_whose_folder_holds_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, files in ((21, ["savednet_", "param_"]), (22, ["param_"]), (23, [])):
                (root / _name(seed)).mkdir()
                for prefix in files:
                    suffix = ".pt" if prefix == "savednet_" else "_param.json"
                    (root / _name(seed) / f"{prefix}{_name(seed)}{suffix}").write_bytes(b"")
            (root / _name(24)).write_bytes(b"")                       # a stray file, not a run folder
            (root / _other(21)).mkdir()
            (root / _other(21) / f"savednet_{_other(21)}.pt").write_bytes(b"")
            self.assertEqual(selection.seeds_with_file(root, _name, "savednet_", ".pt", True), [21])
            self.assertEqual(selection.seeds_with_file(root, _name, "param_", "_param.json", True), [21, 22])
            self.assertEqual(selection.seeds_with_file(root / "missing", _name, "savednet_", ".pt", True), [])


class DrawTests(unittest.TestCase):
    def test_draw_follows_the_rng_and_skips_taken_seeds(self):
        taken = set(range(1, 996))
        for _ in range(3):
            self.assertEqual(sorted(selection.draw_new_seeds(5, taken, rng=random.Random(0))),
                             [996, 997, 998, 999, 1000])
        drawn = selection.draw_new_seeds(5, {123, 558}, rng=random.Random(1))
        self.assertEqual(len(set(drawn)), 5)
        self.assertTrue(set(drawn) <= set(selection.SEED_POOL) - {123, 558})
        self.assertEqual(selection.draw_new_seeds(5, {123, 558}, rng=random.Random(1)), drawn)
        self.assertEqual(selection.draw_new_seeds(1, {1, 2}, pool=range(1, 4)), [3])
        with self.assertRaisesRegex(ValueError, "unused seeds remain"):
            selection.draw_new_seeds(6, taken)


class SelectSeedsTests(unittest.TestCase):
    def test_fresh_draw_skips_completed_and_started_seeds(self):
        with patch("builtins.print") as output:
            seeds = selection.select_seeds(3, None, [408, 921], [11, 408], label="L21e4",
                                           pool=range(1, 20), rng=random.Random(0))
        self.assertEqual(len(set(seeds)), 3)
        self.assertTrue({408, 921, 11}.isdisjoint(seeds))
        self.assertIn("Runs of L21e4: completed seeds [408, 921], started but unfinished [11]",
                      _printed(output))

    def test_explicit_seeds_are_validated_against_the_saved_runs(self):
        with patch("builtins.print"):
            self.assertEqual(selection.select_seeds(5, [42, 101], [408], label="L21e4"), [42, 101])
            with self.assertRaisesRegex(ValueError, r"seeds \[408\] already have a saved run of L21e4"):
                selection.select_seeds(5, [408, 42], [408], label="L21e4")
            with self.assertRaisesRegex(ValueError, "distinct"):
                selection.select_seeds(5, [42, 42], [])
            with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                selection.select_seeds(5, [42, -1], [])
            with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                selection.select_seeds(5, ["42"], [])
        with patch("builtins.print") as output:
            self.assertEqual(selection.select_seeds(5, [11], [408], [11]), [11])
        self.assertIn("Warning: seeds [11] have a started but unfinished run", _printed(output))


class ChooseSeedsTests(unittest.TestCase):
    def test_choose_seeds_scans_the_markers_and_labels_the_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            _touch(directory, [f"savednet_{_name(5)}.pt", f"param_{_name(5)}_param.json",
                               f"param_{_name(9)}_param.json", f"savednet_{_other(5)}.pt"])
            with patch("builtins.print") as output:
                seeds = selection.choose_seeds(2, None, directory, _name, pool=range(1, 12),
                                               rng=random.Random(0))
                with self.assertRaisesRegex(ValueError, r"seeds \[5\]"):
                    selection.choose_seeds(2, [5], directory, _name)
                self.assertEqual(selection.choose_seeds(2, [9, 10], directory, _name), [9, 10])
        self.assertEqual(len(seeds), 2)
        self.assertTrue({5, 9}.isdisjoint(seeds))
        self.assertIn(f"Runs of {_name('*')}: completed seeds [5], started but unfinished [9]",
                      _printed(output))


if __name__ == "__main__":
    unittest.main()
