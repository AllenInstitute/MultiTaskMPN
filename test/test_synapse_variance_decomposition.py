"""Tests of the synapse tuning variance decomposition by endpoint clusters."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from scipy.cluster.hierarchy import linkage

import synapse_variance_decomposition as decomposition


def _one_hot(labels, n_groups):
    design = np.zeros((labels.size, n_groups))
    design[np.arange(labels.size), labels] = 1.0
    return design


def _centered(values):
    return values - values.mean(axis=0)


class SumOfSquaresTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n, self.p = 400, 5
        self.a = self.rng.integers(0, 4, self.n)
        self.b = self.rng.integers(0, 3, self.n)
        self.values = _centered(self.rng.normal(size=(self.n, self.p)) + self.a[:, None] * 0.5)

    def test_group_sum_of_squares_matches_the_one_way_anova(self):
        expected = sum(np.sum(self.a == g) * np.sum(self.values[self.a == g].mean(axis=0) ** 2)
                       for g in range(4))
        self.assertAlmostEqual(decomposition.explained_by_groups(self.a, 4, self.values), expected)
        # A label with no members does not break the computation.
        self.assertAlmostEqual(decomposition.explained_by_groups(self.a, 6, self.values), expected)

    def test_additive_sum_of_squares_matches_the_explicit_least_squares_fit(self):
        design = np.hstack([_one_hot(self.a, 4), _one_hot(self.b, 3)])
        fitted = design @ np.linalg.lstsq(design, self.values, rcond=None)[0]
        self.assertAlmostEqual(decomposition.explained_by_additive(self.a, 4, self.b, 3, self.values),
                               float((fitted ** 2).sum()))
        pre = decomposition.explained_by_groups(self.a, 4, self.values)
        post = decomposition.explained_by_groups(self.b, 3, self.values)
        additive = decomposition.explained_by_additive(self.a, 4, self.b, 3, self.values)
        block, n_block = decomposition.joint_labels(self.a, 4, self.b, 3)
        block_ss = decomposition.explained_by_groups(block, n_block, self.values)
        self.assertGreaterEqual(additive + 1e-9, max(pre, post))
        self.assertGreaterEqual(block_ss + 1e-9, additive)

    def test_within_group_permutation_keeps_every_group_composition(self):
        groups = self.rng.integers(0, 5, self.n)
        labels = self.rng.integers(0, 7, self.n)
        permuted = decomposition.permute_within_groups(labels, groups, self.rng)
        for g in range(5):
            self.assertEqual(sorted(permuted[groups == g]), sorted(labels[groups == g]))
        self.assertFalse(np.array_equal(permuted, labels))


class DecompositionTests(unittest.TestCase):
    def synthetic(self, kind, seed=1, n_input=12, n_hidden=10, n_features=6, noise=0.05):
        """Active synapses of a small grid whose features follow `kind`:
        'additive' (pre-cluster + post-cluster effects) or 'interaction'
        (block effects with zero marginals)."""
        rng = np.random.default_rng(seed)
        pre_index, post_index = np.meshgrid(np.arange(n_input), np.arange(n_hidden))
        pre_index, post_index = pre_index.ravel(), post_index.ravel()
        neuron_labels = {"pre": rng.integers(1, 4, n_input), "post": rng.integers(1, 3, n_hidden)}
        pre = neuron_labels["pre"][pre_index] - 1
        post = neuron_labels["post"][post_index] - 1
        if kind == "additive":
            effects_pre = rng.normal(size=(3, n_features))
            effects_post = rng.normal(size=(2, n_features))
            signal = effects_pre[pre] + effects_post[post]
        else:
            cells = rng.normal(size=(3, 2, n_features))
            cells -= cells.mean(axis=0, keepdims=True)
            cells -= cells.mean(axis=1, keepdims=True)
            signal = cells[pre, post]
        features = signal + noise * rng.normal(size=signal.shape)
        synapse_labels = rng.integers(1, 5, pre_index.size)
        return (_centered(features), pre_index, n_input, post_index, n_hidden,
                neuron_labels, synapse_labels)

    def test_additive_structure_is_credited_to_the_additive_model(self):
        result = decomposition.decompose(*self.synthetic("additive"), n_permutations=40, seed=0)
        self.assertGreater(result["additive"]["r2"], 0.95)
        self.assertLess(result["interaction"]["r2"], 0.03)
        self.assertLess(result["pre"]["p_value"], 0.05)
        self.assertLess(result["post"]["p_value"], 0.05)
        self.assertEqual(result["n_groups"], {"pre": 3, "post": 2, "synapse": 4, "block_occupied": 6})
        for model in decomposition.CLUSTER_MODELS:
            self.assertEqual(result[model]["null"].shape, (40,))
            self.assertEqual(result[model]["sided"], "greater")
        # The endpoint neurons explain at least what their clusters do.
        neuron = result["neuron_level"]
        self.assertGreaterEqual(neuron["pre_neuron"]["r2"] + 1e-9, result["pre"]["r2"])
        self.assertGreaterEqual(neuron["additive_neuron"]["r2"] + 1e-9, result["additive"]["r2"])
        self.assertLess(neuron["additive_neuron"]["p_value"], 0.05)

    def test_interaction_structure_is_credited_to_the_block_model(self):
        result = decomposition.decompose(*self.synthetic("interaction"), n_permutations=40, seed=0)
        self.assertLess(result["additive"]["r2"], 0.15)
        self.assertGreater(result["block"]["r2"], 0.9)
        self.assertGreater(result["interaction"]["r2"], 0.8)
        self.assertLess(result["interaction"]["p_value"], 0.05)

    def test_synapse_clusters_aligned_with_the_residual_explain_beyond_the_block(self):
        data = list(self.synthetic("additive", noise=1.0))
        centered, pre_index, _, post_index, _, neuron_labels, _ = data
        # Synapse labels that split the synapses by the sign of their first
        # residual feature carry structure no block holds.
        pre = neuron_labels["pre"][pre_index] - 1
        post = neuron_labels["post"][post_index] - 1
        block, n_block = decomposition.joint_labels(pre, 3, post, 2)
        residual = centered.copy()
        for cell in np.unique(block):
            residual[block == cell] -= centered[block == cell].mean(axis=0)
        data[6] = (residual[:, 0] > 0).astype(int) + 1
        result = decomposition.decompose(*data, n_permutations=40, seed=0)
        self.assertGreater(result["synapse_beyond_block"]["r2"], 0.03)
        self.assertLess(result["synapse_beyond_block"]["p_value"], 0.05)
        self.assertGreaterEqual(result["block_plus_synapse"]["r2"] + 1e-9, result["block"]["r2"])


def _neuron_entry(rng, n_neurons, n_unresponsive, n_tolerance):
    """A cluster_info entry whose saved dendrogram fixed_k_col_clusters can re-cut."""
    unresponsive = np.zeros(n_neurons, dtype=bool)
    unresponsive[rng.choice(n_neurons, n_unresponsive, replace=False)] = True
    active = np.where(~unresponsive)[0]
    tol_labels = np.zeros(n_neurons, dtype=int)
    tol_labels[active] = rng.integers(1, n_tolerance + 1, active.size)
    tol_labels[unresponsive] = n_tolerance + 1
    clusters = {label: np.where(tol_labels == label)[0] for label in range(1, n_tolerance + 2)}
    return {
        "result": {"col_tol_labels": tol_labels, "col_unresponsive_mask": unresponsive,
                   "col_tol_k": n_tolerance,
                   "col_linkage": linkage(rng.normal(size=(active.size, 4)), method="ward")},
        "col_clusters": clusters, "unresponsive_label": n_tolerance + 1,
        "unresponsive_neurons": np.where(unresponsive)[0],
    }


def _write_run(root, aname, rng, n_input=30, n_hidden=28, n_features=7):
    run_dir = root / aname
    run_dir.mkdir(parents=True)
    n_synapses = n_input * n_hidden
    cluster_info = {"input_unnormalized": _neuron_entry(rng, n_input, 3, 3),
                    "hidden_unnormalized": _neuron_entry(rng, n_hidden, 2, 4)}
    unresponsive = np.zeros(n_synapses, dtype=bool)
    unresponsive[rng.choice(n_synapses, 200, replace=False)] = True
    fixed = np.where(unresponsive, decomposition.FIXED_K + 1,
                     rng.integers(1, decomposition.FIXED_K + 1, n_synapses))
    tolerance = np.where(unresponsive, 6, rng.integers(1, 6, n_synapses))
    synapse_result = {"col_labels_by_k": {10: tolerance, decomposition.FIXED_K: fixed},
                      "col_unresponsive_label_by_k": {10: 6, decomposition.FIXED_K: decomposition.FIXED_K + 1},
                      "col_labels": tolerance, "col_unresponsive_label": 6,
                      "col_unresponsive_mask": unresponsive, "col_tol_labels": tolerance, "col_tol_k": 5}
    features = np.abs(rng.normal(size=(n_features, n_synapses)))
    negative = rng.random(n_synapses) < 0.4
    features[:, negative] *= -1                       # W < 0 flips every period's feature
    features[:, unresponsive] *= 0.01
    cluster_info_mod = {decomposition.MODULATION_TYPE: {
        "result_all_name_lst": ["G=100", "G=300", "G=1000"],
        "result_all_lst": [{}, synapse_result, {}],
        "cell_vars_rules_sorted_norm": features, "mod_shape": (n_hidden, n_input),
        "tb_break_name": np.array([f"period{k}" for k in range(n_features)]),
    }}
    with (run_dir / f"cluster_info_{aname}.pkl").open("wb") as stream:
        pickle.dump(cluster_info, stream)
    with (run_dir / f"cluster_info_mod_{aname}.pkl").open("wb") as stream:
        pickle.dump(cluster_info_mod, stream)
    return unresponsive, negative[~unresponsive]


class RunTests(unittest.TestCase):
    def test_run_reads_the_pickles_decomposes_both_label_sets_and_saves(self):
        aname = "everything_seed5_L21e4+hidden300+batch128+angle"
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unresponsive, negative = _write_run(root, aname, rng)
            (root / "everything_seed6_L21e3+hidden300+batch128+angle").mkdir()   # no pickles
            self.assertEqual(decomposition.candidate_runs(root), [(5, "L21e4", aname)])
            self.assertEqual(decomposition.select_runs(None, "L21e4", decomposition.candidate_runs(root)),
                             [(5, "L21e4", aname)])
            with self.assertRaisesRegex(SystemExit, "No run matches"):
                decomposition.select_runs(7, None, decomposition.candidate_runs(root))
            with patch("builtins.print") as output:
                entries = decomposition.run([aname], root, n_permutations=15, seed=2)
            with (root / aname / decomposition.result_name(aname)).open("rb") as stream:
                saved = pickle.load(stream)
        self.assertEqual(len(entries), 1)
        entry = saved
        self.assertEqual(entry["aname"], aname)
        self.assertEqual((entry["seed"], entry["feature"]), (5, "L21e4"))
        self.assertEqual(entry["n_synapses"], 30 * 28)
        self.assertEqual(entry["n_active"], 30 * 28 - int(unresponsive.sum()))
        self.assertEqual(entry["n_permutations"], 15)
        self.assertEqual(entry["primary"], {"version": "magnitude", "label_set": "fixed_k20"})
        self.assertEqual(set(entry["versions"]), {"magnitude", "signed"})
        magnitude = entry["versions"]["magnitude"]
        self.assertEqual(magnitude["feature_transform"], "log1p(|W| * Var(M))")
        self.assertEqual(entry["versions"]["signed"]["feature_transform"], "signed_log1p(W * Var(M))")
        self.assertTrue(0 <= magnitude["sign_of_w_r2"] <= 1)
        self.assertEqual(magnitude["n_negative_w"], int(negative.sum()))
        self.assertEqual(set(magnitude["label_sets"]), {"fixed_k20", "tolerance"})
        fixed = magnitude["label_sets"]["fixed_k20"]
        self.assertEqual(fixed["n_groups"]["pre"], decomposition.FIXED_K + 1)   # 20 active + unresponsive
        self.assertEqual(fixed["n_groups"]["synapse"], decomposition.FIXED_K)   # active synapses only
        self.assertEqual(fixed["unresponsive"], {"pre": 21, "post": 21, "synapse": 21})
        tolerance = magnitude["label_sets"]["tolerance"]
        self.assertEqual(tolerance["n_groups"], {"pre": 4, "post": 5, "synapse": 5,
                                                 "block_occupied": tolerance["n_groups"]["block_occupied"]})
        for results in entry["versions"].values():
            for result in results["label_sets"].values():
                for model in decomposition.CLUSTER_MODELS:
                    self.assertTrue(0 < result[model]["p_value"] <= 1)
                    self.assertEqual(result[model]["null"].shape, (15,))
                for model in decomposition.NEURON_MODELS:
                    self.assertEqual(result["neuron_level"][model]["null"].shape, (15,))
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("[magnitude / fixed_k20]", printed)
        self.assertIn("[signed / fixed_k20]", printed)
        self.assertIn("L21e4: 1 runs", printed)
        self.assertIn("synapse_beyond_block", printed)
        self.assertIn("residual (1 - block)", printed)

    def test_a_grouping_other_than_g300_is_refused(self):
        aname = "everything_seed5_L21e4+hidden300+batch128+angle"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, aname, np.random.default_rng(0))
            path = root / aname / f"cluster_info_mod_{aname}.pkl"
            with path.open("rb") as stream:
                cluster_info_mod = pickle.load(stream)
            cluster_info_mod[decomposition.MODULATION_TYPE]["result_all_name_lst"][1] = "G=100"
            with path.open("wb") as stream:
                pickle.dump(cluster_info_mod, stream)
            with self.assertRaisesRegex(ValueError, "G=300"):
                decomposition.load_run(aname, root)


if __name__ == "__main__":
    unittest.main()
