"""Legacy cache spelling is adapted in memory without migrating database files."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from core.lesion_cache import (controls_pickle_path, load_lesion_pickle, normalize_lesion_cache,
                               overlay_lesion_controls, resolve_lesion_cache_path)


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

class LesionControlsOverlayTests(unittest.TestCase):
    """overlay_lesion_controls swaps the single-cluster records of a lesion
    pickle for a lesion_controls pickle's, in memory, after checking that
    both describe the same clusters."""

    @staticmethod
    def neuron(names, accs, units=None):
        record = {"all_comb_names_lesion": list(names), "all_tasks": ["a", "b"],
                  "ihtask_accs": accs}
        if units is not None:
            record["lesion_units"] = units
        return record

    def lesion_pickle(self):
        names = ["pre_nolesion", "pre_c1", "post_nolesion", "post_c1"]
        units = {"pre_nolesion": 0, "pre_c1": 3, "post_nolesion": 0, "post_c1": 2}
        clusters = {1: [0, 1], 2: [2, 3, 4]}
        mod = {"all_comb_names_mod": ["mod_nolesion", "mod_c1", "mod_c2"], "all_tasks": ["a", "b"],
               "mod_col_clusters": clusters, "mod_lesion_mode": "zero_W", "mod_G_idx": 1,
               "mod_k": 2, "modtask_accs": "old"}
        return {
            "lesion": self.neuron(names, "old", units),
            "random_lesion": {"all_comb_names_lesion": names, "all_tasks": ["a", "b"],
                              "ihrandomtask_accs_raw": "old"},
            "lesion_unnorm": self.neuron(names, "old", units),
            "random_lesion_unnorm": {"all_comb_names_lesion": names, "all_tasks": ["a", "b"],
                                     "ihrandomtask_accs_raw": "old"},
            "combined_lesion_norm": {"combined_accs": "old"},
            "mod_lesion": {"t__zero_W": mod, "t__freeze_M": {**mod, "mod_lesion_mode": "freeze_M"},
                           "u__zero_W": {**mod}},
            "repeat_num": 10, "fixed_k": 2,
        }

    def controls_pickle(self, pool="responsive"):
        lesion = self.lesion_pickle()
        controls = {"schema_version": 1, "control_pool": pool, "repeat_num": 30, "test_seed": 0,
                    "test_checksums": {"a": "x", "b": "y"}, "mod_lesion": {}}
        for key in ("lesion", "random_lesion", "lesion_unnorm", "random_lesion_unnorm"):
            controls[key] = {**lesion[key]}
            for field in ("ihtask_accs", "ihrandomtask_accs_raw"):
                if field in controls[key]:
                    controls[key][field] = "new"
            controls[key]["condition_pools"] = [None, pool, None, pool]
        for key in ("t__zero_W", "t__freeze_M"):
            controls["mod_lesion"][key] = {k: v for k, v in lesion["mod_lesion"][key].items()
                                           if k not in ("mod_G_idx", "mod_k")}
            controls["mod_lesion"][key]["modtask_accs"] = "new"
        return controls

    def test_controls_pickle_path_sits_beside_the_lesion_pickle(self):
        path = Path("multiple_tasks_perf/run/lesion_prune_results_run.pkl")
        self.assertEqual(controls_pickle_path(path, "responsive"),
                         Path("multiple_tasks_perf/run/lesion_controls_responsive_run.pkl"))

    def test_matching_controls_replace_the_single_cluster_records_only(self):
        lesion = self.lesion_pickle()
        controls = self.controls_pickle()
        overlaid, kept = overlay_lesion_controls(lesion, controls)
        self.assertEqual(kept, ["u__zero_W"])
        self.assertEqual(overlaid["lesion"]["ihtask_accs"], "new")
        self.assertEqual(overlaid["random_lesion_unnorm"]["ihrandomtask_accs_raw"], "new")
        self.assertEqual(overlaid["lesion"]["condition_pools"], [None, "responsive", None, "responsive"])
        self.assertEqual(overlaid["mod_lesion"]["t__freeze_M"]["modtask_accs"], "new")
        self.assertEqual(overlaid["mod_lesion"]["t__freeze_M"]["mod_lesion_mode"], "freeze_M")
        self.assertEqual(overlaid["mod_lesion"]["t__zero_W"]["mod_G_idx"], 1)   # kept from lesion.py
        self.assertEqual(overlaid["mod_lesion"]["u__zero_W"]["modtask_accs"], "old")
        self.assertEqual(overlaid["combined_lesion_norm"]["combined_accs"], "old")
        self.assertEqual((overlaid["control_pool"], overlaid["repeat_num"], overlaid["test_seed"]),
                         ("responsive", 30, 0))
        self.assertEqual(overlaid["combined_control_pool"], "all")
        self.assertEqual(overlaid["controls_schema_version"], 1)
        # Inputs untouched.
        self.assertEqual(lesion["lesion"]["ihtask_accs"], "old")
        self.assertEqual(lesion["mod_lesion"]["t__zero_W"]["modtask_accs"], "old")
        self.assertNotIn("control_pool", lesion)

    def test_mismatched_controls_are_rejected(self):
        lesion = self.lesion_pickle()
        for break_it, message in (
                (lambda c: c["lesion"]["all_comb_names_lesion"].pop(), "lesion: condition names differ"),
                (lambda c: c["lesion_unnorm"]["lesion_units"].update({"pre_c1": 4}),
                 "lesion_unnorm: cluster sizes differ"),
                (lambda c: c["random_lesion"].update({"all_tasks": ["a"]}), "random_lesion: tasks differ"),
                (lambda c: c["mod_lesion"]["t__zero_W"]["mod_col_clusters"].update({2: [2, 3]}),
                 "t__zero_W: synapse clusters differ"),
                (lambda c: c["mod_lesion"].update({"v__zero_W": c["mod_lesion"]["t__zero_W"]}),
                 "v__zero_W: not in the lesion pickle"),
                (lambda c: c.update({"control_pool": "silent"}), "control_pool"),
                (lambda c: c.pop("lesion_unnorm"), "lesion_unnorm: missing")):
            controls = self.controls_pickle()
            controls["mod_lesion"]["t__zero_W"]["mod_col_clusters"] = dict(
                controls["mod_lesion"]["t__zero_W"]["mod_col_clusters"])
            break_it(controls)
            with self.assertRaisesRegex(ValueError, message):
                overlay_lesion_controls(lesion, controls)


if __name__ == "__main__":
    unittest.main()
