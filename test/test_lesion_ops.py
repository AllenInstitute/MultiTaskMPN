"""core/lesion_ops.py: the in-place lesions, their controls and the cluster
bookkeeping that lesion.py and lesion_controls.py share."""

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import torch

import lesion_ops
import mpn


def tiny_network():
    """A 4-input, 5-embedding, 6-hidden multiplicative MPN with random weights."""
    net_params = {
        "net_type": "dmpn", "n_neurons": [4, 6, 3], "linear_embed": 5, "activation": "tanh",
        "output_matrix": "", "input_layer_add": True, "input_layer_add_trainable": True,
        "input_layer_bias": False, "output_bias": False, "loss_type": "MSE",
        "acc_measure": "angle", "dt": 40,
        "ml_params": {"bias": True, "mp_type": "mult", "m_update_type": "hebb_assoc",
                      "eta_type": "scalar", "lam_type": "scalar", "m_time_scale": 4000,
                      "eta_train": True, "lam_train": False, "W_freeze": False},
    }
    torch.manual_seed(0)
    model = mpn.DeepMultiPlasticNet(net_params, verbose=False, forzihan=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter))
    return model


def cluster_info_fixture():
    """Two fixed-k neuron entries: input (embedding) and hidden clusters."""
    return {
        "input_normalized_k2": {"col_clusters": {1: np.array([0, 2]), 2: np.array([1, 3, 4])},
                                "unresponsive_label": None},
        "hidden_normalized_k2": {"col_clusters": {1: np.array([0, 1]), 2: np.array([2, 3]),
                                                  3: np.array([4, 5])},
                                 "unresponsive_label": 3},
    }


def state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def assert_state_equal(test, before, after):
    test.assertEqual(set(before), set(after))
    for name in before:
        test.assertTrue(torch.equal(before[name], after[name]), name)


class NeuronLesionTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_network()
        self.cluster_info = cluster_info_fixture()
        self.before = state(self.model)

    def test_pre_lesion_zeroes_the_embedding_units_and_restores(self):
        saved, n_units = lesion_ops.lesion_prepost_inplace(
            self.model, self.cluster_info, 1, "pre", "normalized_k2")
        self.assertEqual(n_units, 2)
        W_in = self.model.W_initial_linear.weight.detach()
        W = self.model.mp_layer1.W.detach()
        self.assertTrue(torch.equal(W_in[[0, 2]], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(W[:, [0, 2]], torch.zeros(6, 2)))
        self.assertTrue(torch.equal(W_in[[1, 3, 4]], self.before["W_initial_linear.weight"][[1, 3, 4]]))
        self.assertTrue(torch.equal(W[:, [1, 3, 4]], self.before["mp_layer1.W"][:, [1, 3, 4]]))
        lesion_ops.restore_lesion(self.model, saved)
        assert_state_equal(self, self.before, state(self.model))

    def test_post_lesion_zeroes_the_hidden_units_and_restores(self):
        saved, n_units = lesion_ops.lesion_prepost_inplace(
            self.model, self.cluster_info, 2, "post", "normalized_k2")
        self.assertEqual(n_units, 2)
        self.assertTrue(torch.equal(self.model.mp_layer1.W.detach()[[2, 3]], torch.zeros(2, 5)))
        self.assertTrue(torch.equal(self.model.mp_layer1.b.detach()[[2, 3]], torch.zeros(2)))
        self.assertTrue(torch.equal(self.model.W_output.detach()[:, [2, 3]], torch.zeros(3, 2)))
        self.assertTrue(torch.equal(self.model.mp_layer1.W.detach()[[0, 1, 4, 5]],
                                    self.before["mp_layer1.W"][[0, 1, 4, 5]]))
        lesion_ops.restore_lesion(self.model, saved)
        assert_state_equal(self, self.before, state(self.model))

    def test_baseline_is_a_no_op(self):
        saved, n_units = lesion_ops.lesion_prepost_inplace(
            self.model, self.cluster_info, None, "pre", "normalized_k2")
        self.assertEqual((saved, n_units), ({}, 0))
        lesion_ops.restore_lesion(self.model, saved)
        assert_state_equal(self, self.before, state(self.model))

    def test_random_control_draws_the_cluster_size_from_the_pool(self):
        pool = np.array([1, 3, 4])
        draws = []
        for _ in range(2):
            saved, n_units = lesion_ops.lesion_prepost_inplace(
                self.model, self.cluster_info, 1, "pre", "normalized_k2",
                random=True, pool=pool, rng=np.random.default_rng(7))
            draws.append(sorted(saved["neuron_index"].tolist()))
            lesion_ops.restore_lesion(self.model, saved)
            self.assertEqual(n_units, 2)
            self.assertTrue(set(draws[-1]) <= set(pool.tolist()))
            self.assertEqual(len(set(draws[-1])), 2)
        self.assertEqual(draws[0], draws[1])  # same seed, same draw
        assert_state_equal(self, self.before, state(self.model))

    def test_default_pool_is_every_assigned_unit(self):
        np.testing.assert_array_equal(
            lesion_ops.default_neuron_pool(self.cluster_info["hidden_normalized_k2"]),
            np.arange(6))
        with patch.object(np.random, "choice", wraps=np.random.choice) as choice:
            saved, _ = lesion_ops.lesion_prepost_inplace(
                self.model, self.cluster_info, 1, "post", "normalized_k2", random=True)
        np.testing.assert_array_equal(choice.call_args.args[0], np.arange(6))
        lesion_ops.restore_lesion(self.model, saved)

    def test_pool_smaller_than_the_cluster_is_rejected(self):
        with self.assertRaises(ValueError):
            lesion_ops.lesion_prepost_inplace(
                self.model, self.cluster_info, 2, "pre", "normalized_k2",
                random=True, pool=np.array([0, 1]))

    def test_unresponsive_conditions_are_read_from_the_labels(self):
        labels, conditions = lesion_ops.unresponsive_neuron_conditions(
            self.cluster_info, "normalized_k2")
        self.assertEqual(labels, {"pre": None, "post": 3})
        self.assertEqual(conditions, ["post_c3"])
        self.assertEqual(lesion_ops.neuron_cluster_size(self.cluster_info, ("post", 3), "normalized_k2"), 2)
        self.assertEqual(lesion_ops.neuron_cluster_size(self.cluster_info, ("post", None), "normalized_k2"), 0)


class SynapseLesionTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_network()
        self.before = state(self.model)
        self.n_pre = 5  # W is (6 post, 5 pre); flat 7 -> post 1, pre 2
        self.clusters = {1: [0, 7, 11], 2: [3, 4]}

    def test_zero_w_zeroes_the_cluster_synapses_and_restores(self):
        saved, n = lesion_ops.lesion_modulation_inplace(
            self.model, 1, self.clusters, self.n_pre, mode="zero_W")
        self.assertEqual(n, 3)
        W = self.model.mp_layer1.W.detach()
        for post, pre in ((0, 0), (1, 2), (2, 1)):
            self.assertEqual(W[post, pre].item(), 0.0)
        self.assertEqual(W[0, 3].item(), self.before["mp_layer1.W"][0, 3].item())
        lesion_ops.restore_lesion(self.model, saved)
        assert_state_equal(self, self.before, state(self.model))

    def test_freeze_m_sets_and_clears_the_plasticity_mask(self):
        saved, n = lesion_ops.lesion_modulation_inplace(
            self.model, 1, self.clusters, self.n_pre, mode="freeze_M")
        self.assertEqual(n, 3)
        post_t, pre_t = self.model.mp_layer1._plasticity_freeze_mask
        self.assertEqual(post_t.tolist(), [0, 1, 2])
        self.assertEqual(pre_t.tolist(), [0, 2, 1])
        assert_state_equal(self, self.before, state(self.model))  # W untouched
        lesion_ops.restore_lesion(self.model, saved)
        self.assertIsNone(self.model.mp_layer1._plasticity_freeze_mask)

    def test_random_control_draws_from_the_pool_or_the_whole_layer(self):
        rng = np.random.default_rng(3)
        saved, n = lesion_ops.lesion_modulation_inplace(
            self.model, 1, self.clusters, self.n_pre, mode="zero_W", random=True,
            pool=np.array([20, 21, 22, 23]), rng=rng)
        flat = (saved["post_t"] * self.n_pre + saved["pre_t"]).tolist()
        self.assertEqual(len(flat), 3)
        self.assertTrue(set(flat) <= {20, 21, 22, 23})
        lesion_ops.restore_lesion(self.model, saved)
        with patch.object(np.random, "choice", wraps=np.random.choice) as choice:
            saved, _ = lesion_ops.lesion_modulation_inplace(
                self.model, 2, self.clusters, self.n_pre, mode="zero_W", random=True)
        np.testing.assert_array_equal(choice.call_args.args[0], np.arange(30))
        lesion_ops.restore_lesion(self.model, saved)
        assert_state_equal(self, self.before, state(self.model))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            lesion_ops.lesion_modulation_inplace(self.model, 1, self.clusters, self.n_pre, mode="drop")


class EvaluateConditionsTests(unittest.TestCase):
    def test_controls_are_cached_per_key_and_the_model_is_restored(self):
        model = tiny_network()
        cluster_info = cluster_info_fixture()
        before = state(model)
        conditions, names = lesion_ops.neuron_conditions(2, 0)
        self.assertEqual(names, ["pre_nolesion", "pre_c1", "pre_c2", "post_nolesion"])
        calls = []

        def evaluate(net, test_set):
            calls.append(test_set)
            return float(net.mp_layer1.W.detach().abs().sum())

        def lesion(condition, random):
            saved, _ = lesion_ops.lesion_prepost_inplace(
                model, cluster_info, condition[1], condition[0], "normalized_k2",
                random=random, rng=np.random.default_rng(0))
            return saved

        def control_key(condition):
            # Both clusters are keyed as size 2 so they share one control set;
            # the baselines are their own control (None).
            return None if condition[1] is None else (condition[0], 2)

        accs, means, raw = lesion_ops.evaluate_conditions(
            model, "trials", conditions, lesion, control_key, repeats=4, evaluate=evaluate)
        full = float(before["mp_layer1.W"].abs().sum())
        self.assertEqual(accs[0], full)
        self.assertLess(accs[1], full)
        self.assertLess(accs[2], full)
        self.assertEqual(accs[3], full)
        self.assertEqual(len(calls), 4 + 4 * 1)      # conditions + repeats x unique keys
        self.assertEqual(raw[0], [full] * 4)          # a None key repeats the condition
        self.assertEqual(raw[3], [full] * 4)
        self.assertEqual(raw[1], raw[2])               # shared draws, identical copies
        self.assertEqual(means[1], float(np.mean(raw[1])))
        self.assertTrue(all(value < full for value in raw[1]))
        assert_state_equal(self, before, state(model))


class ClusterBookkeepingTests(unittest.TestCase):
    def test_run_name_and_variant_keys(self):
        self.assertEqual(lesion_ops.run_name(921, "L21e4"),
                         "everything_seed921_L21e4+hidden300+batch128+angle")
        self.assertEqual(lesion_ops.neuron_variant(20), "normalized_k20")
        self.assertEqual(lesion_ops.neuron_variant(20, normalized=False), "unnormalized_k20")
        self.assertEqual(lesion_ops.neuron_cluster_key("pre", "normalized_k20"), "input_normalized_k20")
        self.assertEqual(lesion_ops.neuron_cluster_key("post", "unnormalized_k20"), "hidden_unnormalized_k20")
        with self.assertRaises(ValueError):
            lesion_ops.neuron_cluster_key("mod", "normalized_k20")

    def test_infer_fixed_k_reads_the_global_assignment_key(self):
        self.assertEqual(lesion_ops.infer_fixed_k({"t": {"global_assignment_fixed_k20": 1}}),
                         (20, "global_assignment_fixed_k20"))
        self.assertEqual(lesion_ops.infer_fixed_k({"t": {"result_all_lst": []}}),
                         (lesion_ops.FALLBACK_FIXED_K, None))

    def test_add_fixed_k_neuron_clusters_injects_every_base(self):
        base = {"row_clusters": "rows", "tb_break_name": "breaks",
                "cell_vars_rules_sorted_norm": "V", "result": "result"}
        cluster_info = {name: dict(base) for name in lesion_ops.NEURON_CLUSTER_BASES}
        recut = ({1: np.array([0]), 2: np.array([1, 2])}, 2)
        with patch.object(lesion_ops.clustering, "fixed_k_col_clusters", return_value=recut) as fk:
            keys = lesion_ops.add_fixed_k_neuron_clusters(cluster_info, 20)
        self.assertEqual(keys, [f"{name}_k20" for name in lesion_ops.NEURON_CLUSTER_BASES])
        self.assertEqual(fk.call_count, 4)
        entry = cluster_info["hidden_unnormalized_k20"]
        self.assertIs(entry["col_clusters"], recut[0])
        self.assertEqual(entry["unresponsive_label"], 2)
        self.assertEqual((entry["row_clusters"], entry["result"]), ("rows", "result"))

    def test_fixed_k_synapse_clusters_prefers_the_saved_fixed_k(self):
        result = {"col_labels_by_k": {2: [1, 1, 2, 3, 3], 4: [1, 2, 3, 4, 5]},
                  "col_unresponsive_mask": [False, False, False, True, True],
                  "col_k": 2, "col_labels": [1, 1, 2, 3, 3]}
        entry = {"result_all_lst": [None, result]}
        clusters, unresponsive, k_used = lesion_ops.fixed_k_synapse_clusters(entry, 1, 2)
        self.assertEqual((clusters, unresponsive, k_used), ({1: [0, 1], 2: [2], 3: [3, 4]}, 3, 2))
        # Unsaved fixed k: the smallest saved k.
        clusters, unresponsive, k_used = lesion_ops.fixed_k_synapse_clusters(entry, 1, 7)
        self.assertEqual(k_used, 2)
        # No per-k labels and no mask: the optimal k and its k + 1 convention.
        legacy = {"result_all_lst": [None, {"col_k": 2, "col_labels": [1, 2, 2, 3, 3]}]}
        clusters, unresponsive, k_used = lesion_ops.fixed_k_synapse_clusters(legacy, 1, 2)
        self.assertEqual((clusters, unresponsive, k_used), ({1: [0], 2: [1, 2], 3: [3, 4]}, 3, 2))

    def test_condition_lists(self):
        conditions, names = lesion_ops.neuron_conditions(2, 1)
        self.assertEqual(conditions, [("pre", None), ("pre", 1), ("pre", 2), ("post", None), ("post", 1)])
        self.assertEqual(names, ["pre_nolesion", "pre_c1", "pre_c2", "post_nolesion", "post_c1"])
        conditions, names = lesion_ops.synapse_conditions({3: [0], 1: [1, 2]})
        self.assertEqual(conditions, [("mod", None), ("mod", 1), ("mod", 3)])
        self.assertEqual(names, ["mod_nolesion", "mod_c1", "mod_c3"])

    def test_test_set_checksum_identifies_the_input_batch(self):
        inputs = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        same = lesion_ops.test_set_checksum((inputs.clone(), None, None))
        self.assertEqual(same, lesion_ops.test_set_checksum((inputs, "out", "mask")))
        self.assertNotEqual(same, lesion_ops.test_set_checksum((inputs + 1, None, None)))
        self.assertEqual(len(same), 40)


if __name__ == "__main__":
    unittest.main()
