"""Tests of the state-space-vs-lesion link: the Spearman Mantel test and its
within-family restricted version, the two lesion-profile measures (Pearson
similarity and L1/C distance), the analysis over the saved caches and the
five paper figures that redraw it."""

import math
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import spearmanr

import paper_plot
import state_space_vs_lesion as link
from core import state_space_geometry as geometry


RULES = list(paper_plot._RULE_MOTIF)
UPPER = np.triu_indices(len(RULES), 1)
FAMILIES = link.task_families(RULES)
SAME_FAMILY = np.array([FAMILIES[i] == FAMILIES[j] for i, j in zip(*UPPER)])
NEURON_CONDITIONS = [f"pre_c{k}" for k in range(1, 4)] + [f"post_c{k}" for k in range(1, 5)]


def _distance_matrix(rng, n, dim=3):
    points = rng.normal(size=(n, dim))
    return np.linalg.norm(points[:, None] - points[None], axis=-1)


def _similarity_matrix(rng, n):
    profiles = rng.normal(size=(n, 6))
    return np.corrcoef(profiles)


def _l1_distance(effect):
    """The L1/C distance matrix of a (tasks, clusters) effect matrix."""
    effect = np.asarray(effect, dtype=float)
    return np.abs(effect[:, None, :] - effect[None, :, :]).mean(axis=2)


def _profile(tasks, similarity, distance, excluded=None):
    return {"tasks": list(tasks), "similarity": similarity, "distance": distance,
            "excluded": excluded}


def _structure_cache(aname, l2_info, seed):
    rng = np.random.default_rng(seed)
    return {"aname": aname, "l2_info": l2_info, "all_rules": list(RULES),
            "m_alignment": link.M_ALIGNMENT,
            "representations": {representation: {"distances": _distance_matrix(rng, len(RULES))}
                                for representation in link.REPRESENTATIONS}}


def _tasksim_cache(aname, tasks, similarity):
    return {"variance_method": link.VARIANCE_METHOD,
            "schema_version": 2, "aname": aname, "tasks": list(tasks),
            "causal_similarity": similarity, "y_definition": link.NEURON_SIMILARITY_DEFINITION}


def _effect_record(effect, tasks, conditions):
    return {"effect": effect, "tasks": list(tasks), "conditions": list(conditions),
            "definition": "random_minus_lesion", "units": "fraction"}


def _effects_cache(aname, tasks, effect, conditions, neuron_effect=None):
    """The normalized-effects cache with the synapse entry and, when given,
    the neuron entry (NEURON_CONDITIONS)."""
    synapse = _effect_record(effect, tasks, conditions)
    synapse["mod_lesion_mode"] = link.SYNAPSE_LESION_MODE
    entries = {link.SYNAPSE_EFFECT_ENTRY: synapse}
    if neuron_effect is not None:
        entries[link.NEURON_EFFECT_ENTRY] = _effect_record(neuron_effect, tasks, NEURON_CONDITIONS)
    return {"variance_method": link.VARIANCE_METHOD,
            "schema_version": 1, "aname": aname, "primary_modulation_mode": link.SYNAPSE_LESION_MODE,
            "entries": entries}


def _prune_results(conditions, unresponsive_label="missing"):
    record = {"all_comb_names_mod": ["mod_nolesion"] + list(conditions),
              "mod_lesion_mode": link.SYNAPSE_LESION_MODE}
    if unresponsive_label != "missing":
        record["unresponsive_label"] = unresponsive_label
    return {"variance_method": link.VARIANCE_METHOD,
            "mod_lesion": {link.SYNAPSE_EFFECT_ENTRY: record}}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream)


class SpearmanMantelTests(unittest.TestCase):
    def test_rho_and_null_are_spearman_over_the_permuted_pairs(self):
        rng = np.random.default_rng(1)
        n = 7
        x = _distance_matrix(rng, n)
        y = _similarity_matrix(rng, n)
        result = geometry.mantel_spearman_permutation(x, y, n_permutations=40, seed=3)
        upper = np.triu_indices(n, 1)
        self.assertAlmostEqual(result["rho"], spearmanr(x[upper], y[upper]).correlation)
        expected_rng = np.random.default_rng(3)
        expected = []
        for _ in range(40):
            order = expected_rng.permutation(n)
            permuted = y[np.ix_(order, order)]
            expected.append(spearmanr(x[upper], permuted[upper]).correlation)
        np.testing.assert_allclose(result["null"], expected)
        self.assertEqual(result["p_value"],
                         (1 + np.sum(np.abs(expected) >= abs(result["rho"]))) / 41)
        self.assertEqual(result["statistic"], "spearman")
        self.assertEqual(result["sided"], "two-sided")
        self.assertNotIn("r", result)

    def test_rho_is_invariant_to_monotone_transforms_and_detects_perfect_order(self):
        rng = np.random.default_rng(0)
        n = 8
        x = _distance_matrix(rng, n)
        y = _similarity_matrix(rng, n)
        plain = geometry.mantel_spearman_permutation(x, y, n_permutations=30, seed=0)
        warped = geometry.mantel_spearman_permutation(x ** 3, np.exp(y), n_permutations=30, seed=0)
        self.assertAlmostEqual(plain["rho"], warped["rho"])
        np.testing.assert_allclose(plain["null"], warped["null"])
        reversed_order = geometry.mantel_spearman_permutation(x, -np.sqrt(x), n_permutations=999)
        self.assertAlmostEqual(reversed_order["rho"], -1.0)
        self.assertLess(reversed_order["p_value"], 0.01)
        with self.assertRaises(ValueError):
            geometry.mantel_spearman_permutation(x, x[:-1, :-1], n_permutations=10)

    def test_restricted_orders_shuffle_items_within_their_groups_only(self):
        groups = np.array(["a", "a", "a", "b", "b", "c", "a", "b"])
        members = [np.flatnonzero(groups == label) for label in np.unique(groups)]
        orders = geometry._restricted_orders(members, len(groups), 200, np.random.default_rng(0))
        self.assertEqual(orders.shape, (200, len(groups)))
        for order in orders:
            self.assertEqual(sorted(order), list(range(len(groups))))
            np.testing.assert_array_equal(groups[order], groups)
        # Groups of several items are shuffled, singletons never move.
        self.assertTrue((orders[:, 5] == 5).all())
        self.assertGreater(len({tuple(order) for order in orders}), 100)

    def test_restricted_test_keeps_the_group_structure_in_its_null(self):
        rng = np.random.default_rng(2)
        n = 9
        groups = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 2)
        x = _distance_matrix(rng, n)
        y = _similarity_matrix(rng, n)
        result = geometry.mantel_spearman_restricted_permutation(x, y, groups, n_permutations=60, seed=1)
        upper = np.triu_indices(n, 1)
        same = groups[upper[0]] == groups[upper[1]]
        self.assertAlmostEqual(result["rho"], spearmanr(x[upper], y[upper]).correlation)
        self.assertAlmostEqual(result["within"]["rho"],
                               spearmanr(x[upper][same], y[upper][same]).correlation)
        self.assertEqual(result["within"]["n_pairs"], 6 + 3 + 1)
        self.assertEqual(result["group_sizes"], {"a": 4, "b": 3, "c": 2})
        self.assertEqual((result["statistic"], result["sided"]), ("spearman", "two-sided"))
        self.assertEqual(result["null"].shape, (60,))
        self.assertEqual(result["within"]["null"].shape, (60,))
        self.assertEqual(result["p_value"],
                         (1 + np.sum(np.abs(result["null"]) >= abs(result["rho"]))) / 61)
        # The null is the Spearman rho over the pairs of y with its labels
        # permuted within groups: replaying the orders reproduces it.
        members = [np.flatnonzero(groups == label) for label in np.unique(groups)]
        orders = geometry._restricted_orders(members, n, 60, np.random.default_rng(1))
        expected_all = [spearmanr(x[upper], y[np.ix_(o, o)][upper]).correlation for o in orders]
        expected_within = [spearmanr(x[upper][same], y[np.ix_(o, o)][upper][same]).correlation
                           for o in orders]
        np.testing.assert_allclose(result["null"], expected_all)
        np.testing.assert_allclose(result["within"]["null"], expected_within)
        # An association carried by group membership alone is part of the
        # null: y that only tells same-group from different-group pairs,
        # plus a within-group part unrelated to x, is never significant.
        block = np.where(groups[:, None] == groups[None, :], 1.0, 0.0) * 10 + _similarity_matrix(rng, n)
        np.fill_diagonal(block, 0.0)
        membership_only = geometry.mantel_spearman_restricted_permutation(
            x, block, groups, n_permutations=200, seed=0)
        self.assertGreater(membership_only["p_value"], 0.2)
        # One group holding every item is the unrestricted test, and the
        # same-group pairs are then all pairs.
        one_group = geometry.mantel_spearman_restricted_permutation(
            x, y, np.zeros(n, dtype=int), n_permutations=30, seed=0)
        self.assertAlmostEqual(one_group["rho"], result["rho"])
        self.assertAlmostEqual(one_group["within"]["rho"], result["rho"])
        self.assertEqual(one_group["within"]["n_pairs"], len(upper[0]))
        with self.assertRaisesRegex(ValueError, "at least two items"):
            geometry.mantel_spearman_restricted_permutation(x, y, np.arange(n), n_permutations=5)
        with self.assertRaisesRegex(ValueError, "same-group pairs"):
            geometry.mantel_spearman_restricted_permutation(
                x, y, np.array([0, 0] + list(range(1, n - 1))), n_permutations=5)
        with self.assertRaisesRegex(ValueError, "group label per item"):
            geometry.mantel_spearman_restricted_permutation(x, y, groups[:-1], n_permutations=5)


class LesionProfileTests(unittest.TestCase):
    aname = "everything_seed5_L21e4+hidden300+batch128+angle"
    conditions = ["mod_c1", "mod_c2", "mod_c3", "mod_c4", "mod_c5"]

    def write_synapse_caches(self, root, effect, unresponsive_label="missing", neuron_effect=None):
        _write(root / "norm" / self.aname / f"normalized_lesion_effects_{self.aname}.pkl",
               _effects_cache(self.aname, RULES, effect, self.conditions, neuron_effect))
        _write(root / "perf" / self.aname / f"lesion_prune_results_{self.aname}.pkl",
               _prune_results(self.conditions, unresponsive_label))

    def assert_profile(self, profile, effect, excluded):
        """`profile` holds the Pearson and L1/C matrices of `effect`."""
        self.assertEqual(profile["tasks"], RULES)
        self.assertEqual(profile["excluded"], excluded)
        np.testing.assert_allclose(profile["similarity"], np.corrcoef(effect))
        np.testing.assert_allclose(profile["distance"], _l1_distance(effect))
        self.assertTrue((np.diag(profile["distance"]) == 0).all())

    def test_profile_measures_are_pearson_and_mean_absolute_difference(self):
        effect = np.array([[0.0, 0.2, 0.4], [0.1, 0.1, 0.1], [0.4, 0.2, 0.0]])
        with self.assertRaisesRegex(ValueError, "constant"):
            link._profile_measures(effect, "cache")
        effect[1] = [0.0, 0.3, 0.3]
        similarity, distance = link._profile_measures(effect, "cache")
        np.testing.assert_allclose(similarity, np.corrcoef(effect))
        np.testing.assert_allclose(distance[0], [0.0, 0.2 / 3, 0.8 / 3])
        np.testing.assert_allclose(distance, distance.T)
        with self.assertRaisesRegex(ValueError, "invalid"):
            link._profile_measures(effect[:, :2], "cache")

    def test_synapse_profiles_drop_the_recorded_unresponsive_class(self):
        rng = np.random.default_rng(2)
        effect = rng.normal(size=(len(RULES), len(self.conditions)))
        effect[:, 1] = -0.8          # a huge constant column, as the real class has
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_synapse_caches(root, effect, unresponsive_label=2)
            self.assert_profile(link.synapse_cluster_profiles(self.aname, root / "norm", root / "perf"),
                                effect[:, [0, 2, 3, 4]], "mod_c2")
            # No flagged class: every cluster stays.
            self.write_synapse_caches(root, effect, unresponsive_label=None)
            self.assert_profile(link.synapse_cluster_profiles(self.aname, root / "norm", root / "perf"),
                                effect, None)
            # Legacy lesion pickles without the label: the last cluster, as
            # lesion_plot reads them.
            self.write_synapse_caches(root, effect)
            self.assert_profile(link.synapse_cluster_profiles(self.aname, root / "norm", root / "perf"),
                                effect[:, :4], "mod_c5")
            # A label that names no saved condition is an error, not silently kept.
            self.write_synapse_caches(root, effect, unresponsive_label=9)
            with self.assertRaisesRegex(ValueError, "mod_c9"):
                link.synapse_cluster_profiles(self.aname, root / "norm", root / "perf")
            self.assertIsNone(link.synapse_cluster_profiles("other_run", root / "norm", root / "perf"))

    def test_synapse_profiles_read_freeze_m_like_the_lesion_heatmap(self):
        """The synapse panels use the lesion heatmap's mode, recorded in the
        results so paper_plot can tell; an entry of another mode is an error."""
        self.assertEqual(link.SYNAPSE_LESION_MODE, "freeze_M")
        self.assertEqual(link.SYNAPSE_LESION_MODE, paper_plot.LESION_HEATMAP_MODE)
        self.assertEqual(link.SYNAPSE_EFFECT_ENTRY,
                         "modulation_all_var_weighted_unnormalized__freeze_M")
        self.assertEqual(link.SYNAPSE_SIMILARITY_DEFINITION, paper_plot._SSL_SYNAPSE_DEFINITION)
        self.assertEqual(link.SYNAPSE_DISTANCE_DEFINITION, paper_plot._SSL_SYNAPSE_DISTANCE_DEFINITION)
        # paper_plot reads each measure from the keys the script writes.
        self.assertEqual([(measure["key"],) + measure["entry_keys"] for measure in paper_plot._SSL_MEASURES],
                         sorted(link.MEASURES, key=lambda row: row[0] != "distance"))

    def test_task_families_cover_the_ruleset_once(self):
        members = [task for family in link.TASK_FAMILIES.values() for task in family]
        self.assertEqual(sorted(members), sorted(RULES))
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(FAMILIES[:2], ["go", "go"])
        self.assertEqual(int(SAME_FAMILY.sum()), 15 + 10 + 6)
        with self.assertRaisesRegex(ValueError, "no family"):
            link.task_families(RULES + ["unknown"])
        # paper_plot names and colors the families in the script's order.
        self.assertEqual(tuple(link.TASK_FAMILIES),
                         tuple(family for family, _, _ in paper_plot._SSL_FAMILIES))
        rng = np.random.default_rng(5)
        effect = rng.normal(size=(len(RULES), len(self.conditions)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_synapse_caches(root, effect, unresponsive_label=None)
            path = root / "norm" / self.aname / f"normalized_lesion_effects_{self.aname}.pkl"
            mislabeled = _effects_cache(self.aname, RULES, effect, self.conditions)
            mislabeled["entries"][link.SYNAPSE_EFFECT_ENTRY]["mod_lesion_mode"] = "zero_W"
            _write(path, mislabeled)
            with self.assertRaisesRegex(ValueError, "lesion mode"):
                link.synapse_cluster_profiles(self.aname, root / "norm", root / "perf")

    def test_neuron_profiles_pair_the_saved_matrix_with_the_effects_that_made_it(self):
        rng = np.random.default_rng(3)
        synapse = rng.normal(size=(len(RULES), len(self.conditions)))
        neuron = rng.normal(size=(len(RULES), len(NEURON_CONDITIONS)))
        similarity = np.corrcoef(neuron)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "norm" / self.aname / f"causal_vs_activity_tasksim_hidden_{self.aname}.pkl"
            _write(path, _tasksim_cache(self.aname, RULES, similarity))
            # Both caches are needed.
            self.write_synapse_caches(root, synapse, unresponsive_label=None)
            with self.assertRaisesRegex(ValueError, link.NEURON_EFFECT_ENTRY):
                link.neuron_cluster_profiles(self.aname, root / "norm")
            self.write_synapse_caches(root, synapse, unresponsive_label=None, neuron_effect=neuron)
            profile = link.neuron_cluster_profiles(self.aname, root / "norm")
            self.assertEqual(profile["tasks"], RULES)
            self.assertIsNone(profile["excluded"])
            np.testing.assert_array_equal(profile["similarity"], similarity)
            np.testing.assert_allclose(profile["distance"], _l1_distance(neuron))
            # The saved matrix must be the correlation of the saved effects.
            other = _similarity_matrix(rng, len(RULES))
            _write(path, _tasksim_cache(self.aname, RULES, other))
            with self.assertRaisesRegex(ValueError, "reproduce"):
                link.neuron_cluster_profiles(self.aname, root / "norm")
            legacy = _tasksim_cache(self.aname, RULES, similarity)
            legacy["schema_version"] = 1
            _write(path, legacy)
            with self.assertRaisesRegex(ValueError, "legacy"):
                link.neuron_cluster_profiles(self.aname, root / "norm")
            other = _tasksim_cache(self.aname, RULES, similarity)
            other["y_definition"] = "something else"
            _write(path, other)
            with self.assertRaisesRegex(ValueError, "definition"):
                link.neuron_cluster_profiles(self.aname, root / "norm")
            self.assertIsNone(link.neuron_cluster_profiles("other_run", root / "norm"))
            path.unlink()
            self.assertIsNone(link.neuron_cluster_profiles(self.aname, root / "norm"))


class AnalyzeRunTests(unittest.TestCase):
    def test_entry_aligns_lesion_tasks_and_tests_every_pairing_of_both_measures_fully_and_within_families(self):
        aname = "everything_seed7_L21e4+hidden300+batch128+angle"
        cache = _structure_cache(aname, 1e-4, 0)
        rng = np.random.default_rng(4)
        neuron_similarity = _similarity_matrix(rng, len(RULES))
        neuron_distance = _distance_matrix(rng, len(RULES))
        # The lesion caches list the tasks in another order; the entry must
        # follow the structure cache's order.
        order = rng.permutation(len(RULES))
        shuffled_tasks = [RULES[index] for index in order]
        eff_mod = cache["representations"]["eff_mod"]["distances"]
        # Synapse profiles in perfect inverse (similarity) and direct (distance) order.
        synapse_similarity, synapse_distance = -eff_mod, 2.0 * eff_mod
        profiles = {
            "neuron": _profile(shuffled_tasks, neuron_similarity[np.ix_(order, order)],
                               neuron_distance[np.ix_(order, order)]),
            "synapse": _profile(RULES, synapse_similarity, synapse_distance, "mod_c21"),
        }
        entry = link.analyze_run(
            aname, 1e-4, RULES, {name: value["distances"] for name, value in cache["representations"].items()},
            profiles, n_permutations=25, seed=1, m_alignment=cache["m_alignment"])
        self.assertEqual(entry["aname"], aname)
        self.assertEqual(entry["l2_info"], 1e-4)
        self.assertEqual(len(entry["pairs"]), len(UPPER[0]))
        self.assertEqual(entry["pairs"][0], (RULES[0], RULES[1]))
        self.assertEqual(entry["excluded_conditions"], {"synapse": "mod_c21"})
        self.assertEqual(entry["families"], dict(zip(RULES, FAMILIES)))
        np.testing.assert_array_equal(entry["pair_same_family"], SAME_FAMILY)
        self.assertEqual(entry["similarity_definitions"]["synapse"], link.SYNAPSE_SIMILARITY_DEFINITION)
        self.assertEqual(entry["profile_distance_definitions"],
                         {"neuron": link.NEURON_DISTANCE_DEFINITION,
                          "synapse": link.SYNAPSE_DISTANCE_DEFINITION})
        np.testing.assert_allclose(entry["similarities"]["neuron"], neuron_similarity[UPPER])
        np.testing.assert_allclose(entry["profile_distances"]["neuron"], neuron_distance[UPPER])
        matrices = {"similarity": {"neuron": neuron_similarity, "synapse": synapse_similarity},
                    "distance": {"neuron": neuron_distance, "synapse": synapse_distance}}
        for representation in link.REPRESENTATIONS:
            distances = cache["representations"][representation]["distances"]
            np.testing.assert_allclose(entry["distances"][representation], distances[UPPER])
            for measure, values_key, tests_key, trends_key, restricted_key in link.MEASURES:
                for cluster_type in ("neuron", "synapse"):
                    values = matrices[measure][cluster_type]
                    np.testing.assert_allclose(entry[values_key][cluster_type], values[UPPER])
                    test = entry[tests_key][representation][cluster_type]
                    self.assertEqual(test["statistic"], "spearman")
                    self.assertEqual(test["n_permutations"], 25)
                    self.assertAlmostEqual(
                        test["rho"], spearmanr(distances[UPPER], values[UPPER]).correlation)
                    # The restricted test: the same rho, its p from
                    # within-family permutations, and the same-family rho.
                    restricted = entry[restricted_key][representation][cluster_type]
                    self.assertAlmostEqual(restricted["rho"], test["rho"])
                    self.assertEqual(restricted["n_permutations"], 25)
                    self.assertEqual(restricted["group_sizes"], {"decision": 5, "go": 6, "matching": 4})
                    self.assertAlmostEqual(
                        restricted["within"]["rho"],
                        spearmanr(distances[UPPER][SAME_FAMILY], values[UPPER][SAME_FAMILY]).correlation)
                    self.assertEqual(restricted["within"]["n_pairs"], int(SAME_FAMILY.sum()))
                    # Each family's own test over its pairs alone.
                    self.assertEqual(list(restricted["per_family"]), list(link.TASK_FAMILIES))
                    for family, members in link.TASK_FAMILIES.items():
                        own = np.array([FAMILIES[i] == family and FAMILIES[j] == family
                                        for i, j in zip(*UPPER)])
                        family_test = restricted["per_family"][family]
                        self.assertAlmostEqual(
                            family_test["rho"],
                            spearmanr(distances[UPPER][own], values[UPPER][own]).correlation)
                        self.assertEqual(family_test["n_pairs"], int(own.sum()))
                        self.assertEqual(family_test["n_items"], len(members))
                        self.assertEqual(family_test["n_distinct_permutations"],
                                         math.factorial(len(members)))
                        self.assertEqual(family_test["null"].shape, (25,))
                    # The descriptive guide is the OLS line through the same pairs.
                    trend = entry[trends_key][representation][cluster_type]
                    slope, intercept = np.polyfit(distances[UPPER], values[UPPER], 1)
                    self.assertEqual(trend["method"], "ols")
                    self.assertAlmostEqual(trend["slope"], slope)
                    self.assertAlmostEqual(trend["intercept"], intercept)
        self.assertEqual(entry["schema_version"], 5)
        self.assertEqual(entry["control_pool"], "all")
        self.assertIsNone(link.descriptive_trend_line(np.ones(4), np.arange(4.)))
        self.assertAlmostEqual(entry["tests"]["eff_mod"]["synapse"]["rho"], -1.0)
        self.assertAlmostEqual(entry["profile_distance_tests"]["eff_mod"]["synapse"]["rho"], 1.0)
        self.assertLess(entry["tests"]["eff_mod"]["synapse"]["p_value"], 0.1)
        self.assertAlmostEqual(entry["profile_distance_restricted_tests"]["eff_mod"]["synapse"]["within"]["rho"], 1.0)
        with self.assertRaisesRegex(ValueError, "different tasks"):
            link.analyze_run(aname, 1e-4, RULES,
                             {"hidden": cache["representations"]["hidden"]["distances"]},
                             {"neuron": _profile(RULES[:-1] + ["unknown"], neuron_similarity,
                                                 neuron_distance)},
                             n_permutations=5, m_alignment=cache["m_alignment"])


class RunAllTests(unittest.TestCase):
    def test_run_all_analyzes_runs_with_both_caches_and_writes_the_results(self):
        with_lesions = "everything_seed1_L21e4+hidden300+batch128+angle"
        without = "everything_seed2_L21e3+hidden300+batch128+angle"
        rng = np.random.default_rng(5)
        conditions = [f"mod_c{k}" for k in range(1, 6)]
        neuron_effect = rng.normal(size=(len(RULES), len(NEURON_CONDITIONS)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_space = root / "state_space"
            for aname, l2_info, seed in ((with_lesions, 1e-4, 0), (without, 1e-3, 1)):
                _write(state_space / f"state_space_structure_{aname}_noise{link.NOISE_LEVEL}.pkl",
                       _structure_cache(aname, l2_info, seed))
            (state_space / "state_space_structure_other_noise0.1.pkl").write_bytes(b"other noise")
            _write(root / "norm" / with_lesions / f"causal_vs_activity_tasksim_hidden_{with_lesions}.pkl",
                   _tasksim_cache(with_lesions, RULES, np.corrcoef(neuron_effect)))
            _write(root / "norm" / with_lesions / f"normalized_lesion_effects_{with_lesions}.pkl",
                   _effects_cache(with_lesions, RULES, rng.normal(size=(len(RULES), 5)), conditions,
                                  neuron_effect))
            _write(root / "perf" / with_lesions / f"lesion_prune_results_{with_lesions}.pkl",
                   _prune_results(conditions, 5))
            with patch("builtins.print") as output:
                results = link.run_all(state_space, root / "norm", root / "perf",
                                       n_permutations=12, seed=2)
            with (state_space / link.RESULT_NAME).open("rb") as stream:
                saved = pickle.load(stream)
            printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
            # Without lesion caches anywhere, the run is an error, not an empty file.
            with self.assertRaisesRegex(FileNotFoundError, "lesion_plot"):
                link.run_all(state_space, root / "elsewhere", root / "perf", n_permutations=3)
            with self.assertRaisesRegex(FileNotFoundError, "state_space_structure"):
                link.run_all(root / "empty", root / "norm", root / "perf", n_permutations=3)
        self.assertEqual(set(results), {with_lesions})
        self.assertEqual(saved.keys(), results.keys())
        entry = saved[with_lesions]
        self.assertEqual(entry["schema_version"], 5)
        self.assertEqual(entry["n_permutations"], 12)
        self.assertEqual(entry["permutation_seed"], 2)
        self.assertEqual(entry["excluded_conditions"], {"synapse": "mod_c5"})
        self.assertEqual(entry["tests"]["hidden"]["neuron"]["null"].shape, (12,))
        self.assertEqual(entry["profile_distance_tests"]["hidden"]["neuron"]["null"].shape, (12,))
        np.testing.assert_allclose(entry["profile_distances"]["neuron"], _l1_distance(neuron_effect)[UPPER])
        self.assertIn(f"{without}: no lesion caches", printed)
        self.assertIn("L2 = 0.0001: 1 runs", printed)
        self.assertIn("eff_mod distance vs synapse lesion-profile similarity", printed)
        self.assertIn("eff_mod distance vs synapse lesion-profile distance", printed)
        self.assertIn("vs L1/C distance rho", printed)
        self.assertIn("within-family permutation:", printed)
        self.assertIn("per family (rho median", printed)
        self.assertEqual(entry["restricted_tests"]["hidden"]["neuron"]["within"]["null"].shape, (12,))
        self.assertEqual(entry["restricted_tests"]["hidden"]["neuron"]["per_family"]["matching"]
                         ["n_distinct_permutations"], 24)


class PaperFigureTests(unittest.TestCase):
    def results(self, anames, seed=0):
        rng = np.random.default_rng(seed)
        combined = {}
        for aname in anames:
            cache = _structure_cache(aname, 1e-4, int(rng.integers(1000)))
            distances = {name: value["distances"] for name, value in cache["representations"].items()}
            # Synapse profiles follow the W * M geometry (similarity inversely,
            # distance directly); neuron profiles are unrelated noise.
            combined[aname] = link.analyze_run(
                aname, 1e-4, RULES, distances,
                {"neuron": _profile(RULES, _similarity_matrix(rng, len(RULES)),
                                    _distance_matrix(rng, len(RULES))),
                 "synapse": _profile(RULES, -distances["eff_mod"] + 0.1 * _similarity_matrix(rng, len(RULES)),
                                     distances["eff_mod"] + 0.1 * _distance_matrix(rng, len(RULES)),
                                     "mod_c21")},
                n_permutations=20, seed=0, m_alignment=cache["m_alignment"])
        return combined

    def render(self, function, combined, anames, show_legend=True):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for aname in anames:
                (base / aname).mkdir()
            result_path = base / "results.pkl"
            if combined is not None:
                _write(result_path, combined)
            with patch.object(paper_plot, "STATE_SPACE_VS_LESION_RESULT_PATH", result_path), \
                    patch.object(paper_plot, "LESION_NORM_DIR", base / paper_plot.LESION_ANAME), \
                    patch.object(paper_plot, "SHOW_LEGEND", show_legend), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as output:
                function()
        for call in save.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        return save, output

    def test_scatter_draws_both_measures_of_the_saved_pairs_with_the_saved_statistics(self):
        collections = paper_plot.mpl.collections
        aname = paper_plot.LESION_ANAME
        combined = self.results([aname])
        save, _ = self.render(paper_plot.plot_state_space_vs_lesion, combined, [aname])
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, "multitask_state_space_vs_lesion.png")
        # Rows: hidden-state then W * M distance; columns: per cluster type
        # the L1/C distance then the Pearson similarity, the cluster type
        # titled once over its pair (no per-axes titles).
        self.assertEqual(len(figure.axes), 8)
        self.assertEqual([text.get_text() for text in figure.texts],
                         ["Neuron clusters", "Synapse clusters"])
        self.assertTrue(all(axis.get_title() == "" for axis in figure.axes))
        top = figure.axes[:4]
        for pair, text in zip((top[:2], top[2:]), figure.texts):
            boxes = [axis.get_position() for axis in pair]
            self.assertAlmostEqual(text.get_position()[0], (boxes[0].x0 + boxes[1].x1) / 2)
            self.assertGreater(text.get_position()[1], max(box.y1 for box in boxes))
        entry = combined[aname]
        panels = [(representation, cluster_type, measure)
                  for representation in ("hidden", "eff_mod")
                  for cluster_type in ("neuron", "synapse")
                  for measure in paper_plot._SSL_MEASURES]
        for axis, (representation, cluster_type, measure) in zip(figure.axes, panels):
            values, test, trend, _ = paper_plot._ssl_panel(entry, measure, representation, cluster_type)
            points = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
            self.assertEqual(len(points), 1)
            offsets = points[0].get_offsets()
            np.testing.assert_allclose(offsets[:, 0], entry["distances"][representation])
            np.testing.assert_allclose(offsets[:, 1], values)
            # One red line: the saved OLS guide, not refitted here.
            self.assertEqual(len(axis.lines), 1)
            line = axis.lines[0]
            np.testing.assert_allclose(line.get_ydata(),
                                       trend["intercept"] + trend["slope"] * line.get_xdata())
            self.assertEqual(paper_plot.mpl.colors.to_hex(line.get_color()),
                             paper_plot.mpl.colors.to_hex(paper_plot._TREND_LINE_COLOR))
            legend_text = axis.get_legend().get_texts()[0].get_text()
            self.assertIn(f"{test['rho']:.2f}", legend_text)
            self.assertIn(paper_plot._permutation_p_text(test["p_value"]), legend_text)
            self.assertIn(paper_plot._SSL_DISTANCE_LABELS[representation], axis.get_xlabel())
            # The distance axis starts at zero; the similarity axis is left to the data.
            self.assertEqual(axis.get_ylim()[0] == 0, measure["key"] == "distance")
            # Undecorated dots and the scatter family's tick size.
            self.assertEqual(len(points[0].get_edgecolors()), 0)
            self.assertTrue(all(label.get_fontsize() == paper_plot._STAT_SCATTER_TICK_FONTSIZE
                                for label in axis.get_xticklabels() + axis.get_yticklabels()))
        # Each panel is one single-scatter square; y labels name each measure
        # once per row, on the neuron pair.
        np.testing.assert_allclose(figure.get_size_inches(),
                                   np.asarray(paper_plot._LESION_SCATTER_FIGSIZE) * (4, 2))
        self.assertEqual([axis.get_ylabel() for axis in figure.axes[:4]],
                         [paper_plot._SSL_MEASURES[0]["ylabel"], paper_plot._SSL_MEASURES[1]["ylabel"],
                          "", ""])
        self.assertIn("L1/C", figure.axes[0].get_ylabel())
        self.assertIn("(r)", figure.axes[1].get_ylabel())
        self.assertAlmostEqual(entry["tests"]["eff_mod"]["synapse"]["rho"], -1.0, places=0)
        self.assertAlmostEqual(entry["profile_distance_tests"]["eff_mod"]["synapse"]["rho"], 1.0, places=0)
        self.assertIn("eff_mod/synapse/similarity: rho = -", save.call_args.kwargs["extra"])
        self.assertIn("eff_mod/synapse/distance: rho = 0.", save.call_args.kwargs["extra"])

    def test_scatter_is_skipped_without_results_for_the_selected_run(self):
        aname = paper_plot.LESION_ANAME
        sibling = aname.replace("seed921", "seed11") if "seed921" in aname else aname.replace("seed", "seed1", 1)
        save, output = self.render(paper_plot.plot_state_space_vs_lesion, None, [aname])
        save.assert_not_called()
        save, output = self.render(paper_plot.plot_state_space_vs_lesion, self.results([sibling]), [aname])
        save.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("state_space_vs_lesion.py", printed)
        for tests_key in ("tests", "profile_distance_tests", "restricted_tests"):
            broken = self.results([aname])
            del broken[aname][tests_key]["hidden"]["synapse"]
            save, output = self.render(paper_plot.plot_state_space_vs_lesion, broken, [aname])
            save.assert_not_called()
            printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
            self.assertIn("incompatible", printed)
        # Results of an earlier schema (no per-family tests) are skipped with the rerun hint.
        legacy = self.results([aname])
        legacy[aname]["schema_version"] = 4
        save, output = self.render(paper_plot.plot_state_space_vs_lesion, legacy, [aname])
        save.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("schema", printed)
        # Results computed with zero_W synapse lesions (before the switch to
        # freeze_M) are skipped, not drawn as freeze_M, for either measure.
        for definitions, legacy_definition in (
                ("similarity_definitions",
                 "pearson_corr_of_task_zero_W_lesion_effect_profiles_over_var_weighted_synapse_clusters"),
                ("profile_distance_definitions",
                 "mean_over_var_weighted_synapse_clusters_abs_task_zero_W_lesion_effect_difference")):
            legacy = self.results([aname])
            legacy[aname][definitions]["synapse"] = legacy_definition
            save, output = self.render(paper_plot.plot_state_space_vs_lesion, legacy, [aname])
            save.assert_not_called()
            printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
            self.assertIn("another lesion mode", printed)

    def test_old_variance_summary_is_rejected_even_with_current_state_alignment(self):
        aname = paper_plot.LESION_ANAME
        combined = self.results([aname])
        combined[aname].pop("variance_method")
        save, output = self.render(paper_plot.plot_state_space_vs_lesion, combined, [aname])
        save.assert_not_called()
        self.assertIn("stale task-variance", str(output.call_args_list))

    def test_seed_figure_draws_every_sibling_run_per_measure_with_the_cohort_test(self):
        import re

        collections = paper_plot.mpl.collections
        aname = paper_plot.LESION_ANAME
        anames = [re.sub(r"seed\d+", f"seed{seed}", aname) for seed in (3, 5, 8)]
        combined = self.results(anames)
        combined["everything_seed4_L21e9+hidden300+batch128+angle"] = combined[anames[0]]  # another cohort
        save, _ = self.render(paper_plot.plot_state_space_vs_lesion_seeds, combined, anames)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, "multitask_state_space_vs_lesion_seeds.png")
        # Rows: L1/C distance then Pearson similarity; columns: neuron then
        # synapse clusters, titled on the top row.
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["Neuron clusters", "Synapse clusters", "", ""])
        panels = [(measure, cluster_type) for measure in paper_plot._SSL_MEASURES
                  for cluster_type in ("neuron", "synapse")]
        for axis, (measure, cluster_type) in zip(figure.axes, panels):
            dots = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
            self.assertEqual(len(dots), 3 * 2)              # 3 runs x 2 representations
            offsets = np.concatenate([c.get_offsets() for c in dots])
            for position, representation in enumerate(("hidden", "eff_mod")):
                tests = [paper_plot._ssl_panel(combined[name], measure, representation, cluster_type)[1]
                         for name in anames]
                rhos = sorted(test["rho"] for test in tests)
                drawn = sorted(offsets[np.isclose(offsets[:, 0], position, atol=0.3), 1])
                np.testing.assert_allclose(drawn, rhos)
                filled = sum(np.allclose(c.get_facecolors()[0][:3],
                                         paper_plot.mpl.colors.to_rgb(paper_plot._LESION_POINT_COLOR))
                             for c in dots if np.isclose(c.get_offsets()[0][0], position, atol=0.3))
                self.assertEqual(filled, sum(test["p_value"] < 0.05 for test in tests))
                label = axis.get_xticklabels()[position].get_text()
                self.assertIn(paper_plot._wilcoxon_vs_zero([test["rho"] for test in tests])[1], label)
            dashes = [c for c in axis.collections if isinstance(c, collections.LineCollection)]
            medians = sorted(c.get_segments()[0][0, 1] for c in dashes)
            expected = sorted(
                np.median([paper_plot._ssl_panel(combined[name], measure, representation, cluster_type)[1]["rho"]
                           for name in anames])
                for representation in ("hidden", "eff_mod"))
            np.testing.assert_allclose(medians, expected)
        self.assertIn("Hidden", figure.axes[0].get_xticklabels()[0].get_text())
        self.assertIn("L1/C", figure.axes[0].get_ylabel())
        self.assertIn("similarity", figure.axes[2].get_ylabel())
        self.assertEqual(figure.axes[1].get_ylabel(), "")
        self.assertEqual(len(figure.legends), 1)
        extra = save.call_args.kwargs["extra"]
        self.assertIn("3 runs", extra)
        self.assertIn("distance/synapse/eff_mod: 3/3 rho > 0", extra)
        self.assertIn("similarity/synapse/eff_mod: 3/3 rho < 0", extra)
        # One run is not a cohort.
        save, output = self.render(paper_plot.plot_state_space_vs_lesion_seeds,
                                   self.results(anames[:1]), anames[:1])
        save.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("need >= 2", printed)

    def test_families_scatter_colors_same_family_pairs_by_family_and_adds_the_restricted_p(self):
        collections = paper_plot.mpl.collections
        aname = paper_plot.LESION_ANAME
        combined = self.results([aname])
        save, _ = self.render(paper_plot.plot_state_space_vs_lesion_families, combined, [aname])
        # The grid, then its color key as a separate small figure.
        self.assertEqual([call.args[1].name for call in save.call_args_list],
                         ["multitask_state_space_vs_lesion_families.png",
                          "multitask_state_space_vs_lesion_families_legend.png"])
        figure, path = save.call_args_list[0].args[:2]
        key = save.call_args_list[1].args[0]
        self.assertEqual(len(figure.axes), 8)
        self.assertEqual([text.get_text() for text in figure.texts],
                         ["Neuron clusters", "Synapse clusters"])
        entry = combined[aname]
        same = np.asarray(entry["pair_same_family"], dtype=bool)
        family_masks = [np.array([FAMILIES[i] == family and FAMILIES[j] == family
                                  for i, j in zip(*UPPER)])
                        for family, _, _ in paper_plot._SSL_FAMILIES]
        panels = [(representation, cluster_type, measure)
                  for representation in ("hidden", "eff_mod")
                  for cluster_type in ("neuron", "synapse")
                  for measure in paper_plot._SSL_MEASURES]
        for axis, (representation, cluster_type, measure) in zip(figure.axes, panels):
            values, test, trend, restricted = paper_plot._ssl_panel(
                entry, measure, representation, cluster_type)
            x = np.asarray(entry["distances"][representation])
            points = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
            # Every pair in grey, then each family's pairs again in its color on top.
            self.assertEqual(len(points), 1 + len(paper_plot._SSL_FAMILIES))
            np.testing.assert_allclose(points[0].get_offsets(), np.column_stack([x, values]))
            self.assertEqual(paper_plot.mpl.colors.to_hex(points[0].get_facecolors()[0]),
                             paper_plot.mpl.colors.to_hex(paper_plot._LESION_POINT_COLOR))
            for collection, mask, (_, _, color) in zip(points[1:], family_masks,
                                                       paper_plot._SSL_FAMILIES):
                np.testing.assert_allclose(collection.get_offsets(),
                                           np.column_stack([x[mask], values[mask]]))
                self.assertEqual(paper_plot.mpl.colors.to_hex(collection.get_facecolors()[0]),
                                 paper_plot.mpl.colors.to_hex(color))
                self.assertLess(collection.get_zorder(), axis.lines[0].get_zorder())
            # The panel's legend holds the statistics only.
            texts = [text.get_text() for text in axis.get_legend().get_texts()]
            self.assertEqual(len(texts), 1)
            self.assertIn(f"{test['rho']:.2f}", texts[0])
            self.assertIn(paper_plot._permutation_p_text(test["p_value"]), texts[0])
            self.assertIn("within-family " + paper_plot._permutation_p_text(restricted["p_value"]),
                          texts[0])
            self.assertEqual(len(axis.lines), 1)
            self.assertEqual(axis.get_ylim()[0] == 0, measure["key"] == "distance")
        extra = save.call_args_list[0].kwargs["extra"]
        self.assertIn("within-family p = ", extra)
        self.assertIn("same-family pairs rho = ", extra)
        # The panels are as large as the full-pairs grid's; the color key
        # names the families with their pair counts and colors.
        np.testing.assert_allclose(figure.get_size_inches(),
                                   np.asarray(paper_plot._LESION_SCATTER_FIGSIZE) * (4, 2))
        self.assertEqual(len(key.axes), 0)
        legend = key.legends[0]
        self.assertEqual([text.get_text() for text in legend.get_texts()],
                         ["go (15 pairs)", "decision (10 pairs)", "matching (6 pairs)",
                          f"different families ({(~same).sum()} pairs)"])
        self.assertEqual([paper_plot.mpl.colors.to_hex(handle.get_color())
                          for handle in legend.legend_handles],
                         [paper_plot.mpl.colors.to_hex(color) for _, _, color in paper_plot._SSL_FAMILIES]
                         + [paper_plot.mpl.colors.to_hex(paper_plot._LESION_POINT_COLOR)])
        # Without legends the grid is saved alone.
        save, _ = self.render(paper_plot.plot_state_space_vs_lesion_families, combined, [aname],
                              show_legend=False)
        save.assert_called_once()

    def test_families_seed_figure_draws_the_same_family_rho_per_run(self):
        import re

        collections = paper_plot.mpl.collections
        aname = paper_plot.LESION_ANAME
        anames = [re.sub(r"seed\d+", f"seed{seed}", aname) for seed in (3, 5, 8)]
        combined = self.results(anames)
        save, _ = self.render(paper_plot.plot_state_space_vs_lesion_families_seeds, combined, anames)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, "multitask_state_space_vs_lesion_families_seeds.png")
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["Neuron clusters", "Synapse clusters", "", ""])
        panels = [(measure, cluster_type) for measure in paper_plot._SSL_MEASURES
                  for cluster_type in ("neuron", "synapse")]
        for axis, (measure, cluster_type) in zip(figure.axes, panels):
            dots = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
            self.assertEqual(len(dots), 3 * 2)
            offsets = np.concatenate([c.get_offsets() for c in dots])
            for position, representation in enumerate(("hidden", "eff_mod")):
                within = [paper_plot._ssl_panel(combined[name], measure, representation, cluster_type)[3]["within"]
                          for name in anames]
                drawn = sorted(offsets[np.isclose(offsets[:, 0], position, atol=0.3), 1])
                np.testing.assert_allclose(drawn, sorted(test["rho"] for test in within))
                filled = sum(np.allclose(c.get_facecolors()[0][:3],
                                         paper_plot.mpl.colors.to_rgb(paper_plot._LESION_POINT_COLOR))
                             for c in dots if np.isclose(c.get_offsets()[0][0], position, atol=0.3))
                self.assertEqual(filled, sum(test["p_value"] < 0.05 for test in within))
                label = axis.get_xticklabels()[position].get_text()
                self.assertIn(paper_plot._wilcoxon_vs_zero([test["rho"] for test in within])[1], label)
        self.assertIn("same-family pairs", figure.axes[0].get_ylabel())
        self.assertEqual(len(figure.legends), 1)
        self.assertEqual(figure.legends[0].get_texts()[0].get_text(),
                         "within-family permutation p < 0.05")
        self.assertIn("3 runs", save.call_args.kwargs["extra"])

    def test_family_seed_figure_draws_each_family_rho_per_run(self):
        import re

        collections = paper_plot.mpl.collections
        aname = paper_plot.LESION_ANAME
        anames = [re.sub(r"seed\d+", f"seed{seed}", aname) for seed in (3, 5, 8)]
        combined = self.results(anames)
        save, _ = self.render(paper_plot.plot_state_space_vs_lesion_family_seeds, combined, anames)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, "multitask_state_space_vs_lesion_family_seeds.png")
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["Neuron clusters", "Synapse clusters", "", ""])
        representation = paper_plot._SSL_PER_FAMILY_REPRESENTATION
        panels = [(measure, cluster_type) for measure in paper_plot._SSL_MEASURES
                  for cluster_type in ("neuron", "synapse")]
        for axis, (measure, cluster_type) in zip(figure.axes, panels):
            dots = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
            self.assertEqual(len(dots), 3 * len(paper_plot._SSL_FAMILIES))
            offsets = np.concatenate([c.get_offsets() for c in dots])
            for position, (family, label, _) in enumerate(paper_plot._SSL_FAMILIES):
                tests = [paper_plot._ssl_panel(combined[name], measure, representation, cluster_type)[3]
                         ["per_family"][family] for name in anames]
                drawn = sorted(offsets[np.isclose(offsets[:, 0], position, atol=0.3), 1])
                np.testing.assert_allclose(drawn, sorted(test["rho"] for test in tests))
                filled = sum(np.allclose(c.get_facecolors()[0][:3],
                                         paper_plot.mpl.colors.to_rgb(paper_plot._LESION_POINT_COLOR))
                             for c in dots if np.isclose(c.get_offsets()[0][0], position, atol=0.3))
                self.assertEqual(filled, sum(test["p_value"] < 0.05 for test in tests))
                tick = axis.get_xticklabels()[position].get_text()
                self.assertIn(label, tick)
                self.assertIn(paper_plot._wilcoxon_vs_zero([test["rho"] for test in tests])[1], tick)
        self.assertIn("one family's pairs", figure.axes[0].get_ylabel())
        self.assertEqual(figure.legends[0].get_texts()[0].get_text(),
                         "within-family permutation p < 0.05")
        self.assertIn("distance/synapse/matching:", save.call_args.kwargs["extra"])

    def test_figures_are_registered_in_the_lesion_mode(self):
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["state_space_vs_lesion"],
                      paper_plot.plot_state_space_vs_lesion)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["state_space_vs_lesion_seeds"],
                      paper_plot.plot_state_space_vs_lesion_seeds)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["state_space_vs_lesion_families"],
                      paper_plot.plot_state_space_vs_lesion_families)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["state_space_vs_lesion_families_seeds"],
                      paper_plot.plot_state_space_vs_lesion_families_seeds)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["state_space_vs_lesion_family_seeds"],
                      paper_plot.plot_state_space_vs_lesion_family_seeds)


if __name__ == "__main__":
    unittest.main()
