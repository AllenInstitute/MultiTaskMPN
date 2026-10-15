"""Tests of the state-space structure analysis over the saved PCA caches."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np

import paper_plot
import state_space_structure as structure
from core.state_space_geometry import TASK_FACTOR_PAIRS


RULES = list(paper_plot._RULE_MOTIF)


def _pca_cache(aname, l2_info, seed):
    """A synthetic state_space_shift PCA cache: category-structured centroids
    (same-color tasks share an offset) for the three representations."""
    rng = np.random.default_rng(seed)
    colors = sorted({color for _, color in paper_plot._RULE_MOTIF.values()})
    offsets = {color: 6.0 * rng.normal(size=8) for color in colors}
    pca_results = {}
    for representation in structure.REPRESENTATIONS:
        centroids = np.stack([offsets[paper_plot._RULE_MOTIF[rule][1]] + rng.normal(size=8)
                              for rule in RULES])
        pca_results[representation] = {
            "X_2d": np.zeros((2 * len(RULES), 2)),
            "ctx_rule_labels": np.repeat(np.arange(len(RULES)), 2),
            "task_centroids": centroids,
            "task_trial_counts": np.full(len(RULES), 2),
            "centroid_space": "original_features",
        }
    return {
        "pca_results": pca_results,
        "all_rules": list(RULES),
        "rule_motif_mapping": dict(paper_plot._RULE_MOTIF),
        "noise_level": 0.01,
        "l2_info": l2_info,
        "aname": aname,
    }


class TaskCategoryTests(unittest.TestCase):
    def test_categories_are_the_shared_colors_not_the_names(self):
        data = _pca_cache("run", 1e-4, 0)
        categories, names = structure.task_categories(data)
        self.assertEqual(len(categories), len(RULES))
        self.assertEqual(categories[RULES.index("dmsgo")], categories[RULES.index("reactgo")])
        self.assertNotEqual(names[RULES.index("dmsgo")], names[RULES.index("reactgo")])
        self.assertEqual(len(set(categories)), 6)

    def test_tasks_without_a_category_are_rejected(self):
        data = _pca_cache("run", 1e-4, 0)
        data["all_rules"].append("unknown_task")
        with self.assertRaisesRegex(ValueError, "unknown_task"):
            structure.task_categories(data)


class AnalyzeRunTests(unittest.TestCase):
    def test_run_result_holds_distances_tree_and_both_tests_per_representation(self):
        data = _pca_cache("everything_seed1_L21e4+hidden300+batch128+angle", 1e-4, 0)
        result = structure.analyze_run(data, n_permutations=30, seed=0)
        self.assertEqual(result["aname"], data["aname"])
        self.assertEqual(result["l2_info"], 1e-4)
        self.assertEqual(result["all_rules"], RULES)
        self.assertEqual(result["n_permutations"], 30)
        self.assertEqual(set(result["representations"]), set(structure.REPRESENTATIONS))
        for entry in result["representations"].values():
            self.assertEqual(entry["distances"].shape, (len(RULES), len(RULES)))
            self.assertEqual(sorted(entry["leaf_order"]), list(range(len(RULES))))
            self.assertEqual(entry["linkage"].shape, (len(RULES) - 1, 4))
            self.assertEqual(entry["linkage_method"], structure.LINKAGE_METHOD)
            separation = entry["separation"]
            self.assertEqual(separation["null"].shape, (30,))
            self.assertLess(separation["p_value"], 0.1)       # built with category structure
            self.assertEqual(set(entry["factor_vectors"]["factors"]), set(TASK_FACTOR_PAIRS))
            self.assertEqual(entry["factor_vectors"]["skipped_factors"], [])

    def test_legacy_caches_without_original_centroids_are_rejected(self):
        data = _pca_cache("run", 1e-4, 0)
        del data["pca_results"]["hidden"]["centroid_space"]
        with self.assertRaisesRegex(ValueError, "state_space_shift"):
            structure.analyze_run(data, n_permutations=5)

    def test_summary_drops_null_arrays_but_keeps_statistics(self):
        result = structure.analyze_run(_pca_cache("run", 1e-3, 1), n_permutations=10)
        summary = structure.summarize_run(result)
        self.assertEqual(summary["l2_info"], 1e-3)
        for entry in summary["representations"].values():
            self.assertNotIn("null", entry["separation"])
            self.assertIn("null_quantile_95", entry["separation"])
            self.assertIn("p_value", entry["separation"])
            for stats in entry["factor_vectors"].values():
                self.assertNotIn("null", stats)
                self.assertNotIn("pairwise_cosines", stats)
                self.assertIn("mean_cosine", stats)
                self.assertIn("n_pairs", stats)
            self.assertIn("null_quantile_975", entry["pooled"])
            self.assertEqual(entry["skipped_factors"], [])


class CohortMeanTests(unittest.TestCase):
    def test_cohort_mean_averages_normalized_matrices_and_tests_them(self):
        results = {
            f"everything_seed{seed}_L21e4+hidden300+batch128+angle": structure.analyze_run(
                _pca_cache(f"everything_seed{seed}_L21e4+hidden300+batch128+angle", 1e-4, seed),
                n_permutations=15)
            for seed in (1, 2, 3)
        }
        cohort = structure.cohort_mean_structure(results, n_permutations=25, seed=4)
        self.assertEqual(cohort["n_runs"], 3)
        self.assertEqual(cohort["anames"], sorted(results))
        self.assertEqual(cohort["all_rules"], RULES)
        self.assertEqual(set(cohort["representations"]), set(structure.REPRESENTATIONS))
        entry = cohort["representations"]["eff_mod"]
        upper = np.triu_indices(len(RULES), 1)
        expected = np.mean([
            results[aname]["representations"]["eff_mod"]["distances"]
            / results[aname]["representations"]["eff_mod"]["distances"][upper].mean()
            for aname in cohort["anames"]], axis=0)
        np.testing.assert_allclose(entry["distances"], expected)
        self.assertEqual(entry["run_scales"].shape, (3,))
        self.assertEqual(sorted(entry["leaf_order"]), list(range(len(RULES))))
        self.assertEqual(entry["separation"]["null"].shape, (25,))
        self.assertEqual(entry["separation"]["seed"], 4)
        self.assertEqual(entry["consistency"]["n_run_pairs"], 3)
        self.assertTrue(-1.0 <= entry["consistency"]["median"] <= 1.0)

    def test_cohort_mean_rejects_runs_with_different_task_orders(self):
        results = {
            aname: structure.analyze_run(_pca_cache(aname, 1e-4, seed), n_permutations=5)
            for seed, aname in enumerate(("run_a", "run_b"))
        }
        results["run_b"]["all_rules"] = list(reversed(results["run_b"]["all_rules"]))
        with self.assertRaisesRegex(ValueError, "run_b"):
            structure.cohort_mean_structure(results, n_permutations=5)
        with self.assertRaises(ValueError):
            structure.cohort_mean_structure({}, n_permutations=5)

    def test_runs_are_grouped_by_exact_l2(self):
        results = {"b": {"l2_info": 1e-4}, "a": {"l2_info": 1e-4}, "c": {"l2_info": 1e-3}}
        grouped = structure.group_runs_by_l2(results)
        self.assertEqual({key: list(value) for key, value in grouped.items()},
                         {1e-4: ["a", "b"], 1e-3: ["c"]})


class RunAllTests(unittest.TestCase):
    def test_run_all_writes_per_run_caches_and_the_combined_summary(self):
        runs = [("everything_seed1_L21e4+hidden300+batch128+angle", 1e-4, 0),
                ("everything_seed2_L21e3+hidden300+batch128+angle", 1e-3, 1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for aname, l2_info, seed in runs:
                with (root / f"state_space_pca_{aname}_noise0.01.pkl").open("wb") as stream:
                    pickle.dump(_pca_cache(aname, l2_info, seed), stream)
            (root / "state_space_pca_other_noise0.1.pkl").write_bytes(b"other noise level")
            with patch("builtins.print") as output:
                combined = structure.run_all(root, n_permutations=12, seed=3)
            per_run = {}
            for aname, _, _ in runs:
                with structure.structure_cache_path(aname, root).open("rb") as stream:
                    per_run[aname] = pickle.load(stream)
            with (root / structure.COMBINED_RESULT_NAME).open("rb") as stream:
                saved_combined = pickle.load(stream)

        self.assertEqual(set(combined["runs"]), {aname for aname, _, _ in runs})
        self.assertEqual(saved_combined["n_permutations"], 12)
        self.assertEqual(saved_combined["permutation_seed"], 3)
        self.assertEqual(saved_combined["runs"].keys(), combined["runs"].keys())
        for aname, l2_info, _ in runs:
            self.assertEqual(per_run[aname]["n_permutations"], 12)
            self.assertEqual(per_run[aname]["permutation_seed"], 3)
            self.assertEqual(combined["runs"][aname]["l2_info"], l2_info)
            np.testing.assert_allclose(
                combined["runs"][aname]["representations"]["eff_mod"]["separation"]["score"],
                per_run[aname]["representations"]["eff_mod"]["separation"]["score"])
        self.assertEqual(set(saved_combined["cohorts"]), {1e-4, 1e-3})
        for strength, aname in ((1e-4, runs[0][0]), (1e-3, runs[1][0])):
            cohort = saved_combined["cohorts"][strength]
            self.assertEqual(cohort["anames"], [aname])
            self.assertEqual(cohort["n_runs"], 1)
            np.testing.assert_allclose(
                cohort["representations"]["hidden"]["distances"],
                per_run[aname]["representations"]["hidden"]["distances"]
                / cohort["representations"]["hidden"]["run_scales"][0])
            self.assertEqual(cohort["representations"]["hidden"]["separation"]["null"].shape,
                             (12,))
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("L2 = 0.0001", printed)
        self.assertIn("L2 = 0.001", printed)
        self.assertIn("pooled over factors", printed)
        self.assertIn("mean RDM over 1 runs", printed)

    def test_run_all_requires_pca_caches(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.assertRaisesRegex(FileNotFoundError, "state_space_shift"):
            structure.run_all(directory, n_permutations=5)


if __name__ == "__main__":
    unittest.main()
