"""Legacy cache spelling is adapted in memory without migrating database files."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from core.lesion_cache import load_lesion_pickle, normalize_lesion_cache, resolve_lesion_cache_path


class LesionCacheTests(unittest.TestCase):
    def test_nested_metadata_is_normalized_without_mutating_input_or_numbers(self):
        numbers = np.array([[0.2, np.nan], [0.8, -0.1]])
        labels = np.array(["pre_noleison", "post_c1"])
        objects = np.empty(2, dtype=object)
        objects[:] = ["mod_noleison", numbers]
        original = {
            "leison_unnorm": {"all_comb_names_leison": labels, "effect": numbers},
            "mod_leison": {"conditions": objects},
            ("combined_leison_norm", 1): ["post_noleison", ("lesion", 3.5)],
        }
        result = normalize_lesion_cache(original)
        self.assertIs(result["lesion_unnorm"]["effect"], numbers)
        np.testing.assert_array_equal(result["lesion_unnorm"]["all_comb_names_lesion"],
                                      ["pre_nolesion", "post_c1"])
        self.assertEqual(result["mod_lesion"]["conditions"][0], "mod_nolesion")
        self.assertIs(result["mod_lesion"]["conditions"][1], numbers)
        self.assertEqual(result[("combined_lesion_norm", 1)], ["post_nolesion", ("lesion", 3.5)])
        self.assertIn("leison_unnorm", original)
        self.assertEqual(labels[0], "pre_noleison")
        self.assertEqual(objects[0], "mod_noleison")

    def test_load_legacy_and_prefer_canonical_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "cluster_corr_vs_normalized_leison_effect_run.pkl"
            canonical = root / "cluster_corr_vs_normalized_lesion_effect_run.pkl"
            payload = pickle.dumps({"leison": {"labels": ["pre_noleison"], "value": 0.25}})
            legacy.write_bytes(payload)
            before = legacy.stat()
            self.assertEqual(resolve_lesion_cache_path(canonical), legacy)
            self.assertEqual(load_lesion_pickle(canonical),
                             {"lesion": {"labels": ["pre_nolesion"], "value": 0.25}})
            self.assertEqual(legacy.read_bytes(), payload)
            self.assertEqual(legacy.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertFalse(canonical.exists())
            canonical.write_bytes(pickle.dumps({"lesion": "new"}))
            self.assertEqual(load_lesion_pickle(canonical), {"lesion": "new"})
            self.assertEqual(resolve_lesion_cache_path(canonical), canonical)

    def test_paper_raw_reader_normalizes_legacy_keys_without_writing(self):
        import paper_plot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f"lesion_prune_results_{paper_plot.LESION_ANAME}.pkl"
            payload = pickle.dumps({"leison_unnorm": {
                "all_comb_names_leison": ["post_noleison", "post_c1"],
                "lesion_units": {"post_noleison": 0, "post_c1": 3}},
                "mod_leison": {"example": {"all_comb_names_mod": ["mod_noleison"]}}})
            path.write_bytes(payload)
            with patch.object(paper_plot, "LESION_DIR", root):
                loaded = paper_plot._load_lesion_results()
            self.assertEqual(loaded["lesion_unnorm"]["all_comb_names_lesion"],
                             ["post_nolesion", "post_c1"])
            self.assertEqual(loaded["lesion_unnorm"]["lesion_units"]["post_c1"], 3)
            self.assertIn("mod_lesion", loaded)
            self.assertEqual(path.read_bytes(), payload)

    def test_collisions_raise_and_missing_cache_is_not_created(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize_lesion_cache({"leison": 1, "lesion": 2})
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing_lesion.pkl"
            with self.assertRaises(FileNotFoundError):
                load_lesion_pickle(missing)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()