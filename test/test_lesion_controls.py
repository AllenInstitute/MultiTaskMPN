"""multiple_task/lesion_controls.py: activity-matched controls of the
single-cluster lesions, scored on a seeded test set and kept in their own
pickle."""

import argparse
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import torch

import lesion_controls
import lesion_ops
from test_lesion_ops import tiny_network

TASKS = ["fdgo", "delaygo"]


def neuron_entries():
    """Fixed-k entries at k = 2: input without an unresponsive class, hidden
    with one (label 3) larger than the two responsive units."""
    return {
        "input": {"col_clusters": {1: np.array([0, 2]), 2: np.array([1, 3, 4])},
                  "unresponsive_label": None},
        "hidden": {"col_clusters": {1: np.array([0]), 2: np.array([1]), 3: np.array([2, 3, 4, 5])},
                   "unresponsive_label": 3},
    }


def cluster_info_fixture():
    entries = neuron_entries()
    return {f"{side}_{variant}_k2": dict(entries[side])
            for side in ("input", "hidden") for variant in ("normalized", "unnormalized")}


def synapse_fixture():
    """Three synapse clusters of the 6 x 5 plastic layer: the unresponsive
    class (label 2) holds 25 of the 30 synapses."""
    responsive = [0, 7, 11, 1, 2]
    clusters = {1: [0, 7, 11], 2: sorted(set(range(30)) - set(responsive)), 3: [1, 2]}
    return clusters, 2


class RecordingEvaluate:
    """Stands in for lesion_ops.evaluate_accuracy: logs which hidden rows and
    synapses are lesioned at each call and returns a state-dependent score."""

    def __init__(self):
        self.zero_rows, self.zero_synapses, self.frozen = [], [], []

    def __call__(self, model, test_set):
        W = model.mp_layer1.W.detach()
        self.zero_rows.append(tuple(torch.where((W == 0).all(dim=1))[0].tolist()))
        self.zero_synapses.append(tuple(torch.where(W.flatten() == 0)[0].tolist()))
        mask = getattr(model.mp_layer1, "_plasticity_freeze_mask", None)
        self.frozen.append(None if mask is None else tuple((mask[0] * 5 + mask[1]).tolist()))
        return float(W.abs().sum())


class PoolTests(unittest.TestCase):
    def test_responsive_pools_leave_out_the_unresponsive_class(self):
        entries = neuron_entries()
        np.testing.assert_array_equal(lesion_controls.responsive_neuron_pool(entries["input"]),
                                      np.arange(5))
        np.testing.assert_array_equal(lesion_controls.responsive_neuron_pool(entries["hidden"]),
                                      [0, 1])
        clusters, unresponsive = synapse_fixture()
        np.testing.assert_array_equal(
            lesion_controls.responsive_synapse_pool(clusters, unresponsive, 30), [0, 1, 2, 7, 11])
        np.testing.assert_array_equal(
            lesion_controls.responsive_synapse_pool(clusters, None, 30), np.arange(30))

    def test_choose_pool_falls_back_to_all_for_oversized_conditions(self):
        pools = {"all": np.arange(6), "responsive": np.array([0, 1])}
        self.assertEqual(lesion_controls.choose_pool(pools, "responsive", 2)[0], "responsive")
        self.assertEqual(lesion_controls.choose_pool(pools, "responsive", 4)[0], "all")
        self.assertEqual(lesion_controls.choose_pool(pools, "all", 2)[0], "all")

    def test_sweeps_name_conditions_pools_and_keys(self):
        cluster_info = cluster_info_fixture()
        entries = neuron_entries()
        pools = {side: {"all": lesion_ops.default_neuron_pool(entries[name]),
                        "responsive": lesion_controls.responsive_neuron_pool(entries[name])}
                 for side, name in (("pre", "input"), ("post", "hidden"))}
        sweep = lesion_controls.neuron_sweep(cluster_info, "normalized_k2", pools, "responsive")
        self.assertEqual(sweep["names"], ["pre_nolesion", "pre_c1", "pre_c2",
                                          "post_nolesion", "post_c1", "post_c2", "post_c3"])
        self.assertEqual(sweep["sizes"], [0, 2, 3, 0, 1, 1, 4])
        self.assertEqual(sweep["condition_pools"],
                         [None, "responsive", "responsive", None, "responsive", "responsive", "all"])
        self.assertEqual(sweep["keys"], [None, ("pre", 2, "responsive"), ("pre", 3, "responsive"),
                                         None, ("post", 1, "responsive"), ("post", 1, "responsive"),
                                         ("post", 4, "all")])
        clusters, unresponsive = synapse_fixture()
        synapse_pools = {"all": np.arange(30),
                         "responsive": lesion_controls.responsive_synapse_pool(clusters, unresponsive, 30)}
        sweep = lesion_controls.synapse_sweep(clusters, synapse_pools, "responsive")
        self.assertEqual(sweep["names"], ["mod_nolesion", "mod_c1", "mod_c2", "mod_c3"])
        self.assertEqual(sweep["condition_pools"], [None, "responsive", "all", "responsive"])
        self.assertEqual(sweep["keys"], [None, (3, "responsive"), (25, "all"), (2, "responsive")])
        # Conditions once, plus repeats per distinct key, per task.
        self.assertEqual(lesion_controls.forward_count(2, sweep["keys"], 30), 2 * (4 + 30 * 3))


class ConsistencyCheckTests(unittest.TestCase):
    def results(self):
        clusters, unresponsive = synapse_fixture()
        record = {"mod_col_clusters": clusters, "unresponsive_label": unresponsive, "mod_G_idx": 1}
        names = ["pre_nolesion", "pre_c1", "post_nolesion", "post_c1"]
        sizes = {"pre_nolesion": 0, "pre_c1": 2, "post_nolesion": 0, "post_c1": 1}
        return {"fixed_k": 2,
                "lesion": {"all_comb_names_lesion": names, "lesion_units": sizes},
                "lesion_unnorm": {"all_comb_names_lesion": names, "lesion_units": sizes},
                "mod_lesion": {"t__zero_W": record, "t__freeze_M": dict(record)}}

    def test_matching_clusters_pass_and_mismatches_are_named(self):
        results = self.results()
        names = results["lesion"]["all_comb_names_lesion"]
        sizes = results["lesion"]["lesion_units"]
        clusters, unresponsive = synapse_fixture()
        neuron = {"lesion": (names, sizes), "lesion_unnorm": (names, sizes)}
        lesion_controls.check_against_lesion_results(results, 2, neuron, {"t": (clusters, unresponsive)})
        with self.assertRaisesRegex(ValueError, "fixed_k"):
            lesion_controls.check_against_lesion_results(results, 3, neuron, {"t": (clusters, unresponsive)})
        with self.assertRaisesRegex(ValueError, "lesion_unnorm: cluster sizes differ"):
            lesion_controls.check_against_lesion_results(
                results, 2, {"lesion": (names, sizes), "lesion_unnorm": (names, {**sizes, "pre_c1": 3})},
                {"t": (clusters, unresponsive)})
        with self.assertRaisesRegex(ValueError, "condition names differ"):
            lesion_controls.check_against_lesion_results(
                results, 2, {"lesion": (names[:-1], sizes), "lesion_unnorm": (names, sizes)},
                {"t": (clusters, unresponsive)})
        other = {**clusters, 3: [1, 5]}
        with self.assertRaisesRegex(ValueError, "t__zero_W: synapse clusters differ"):
            lesion_controls.check_against_lesion_results(results, 2, neuron, {"t": (other, unresponsive)})
        with self.assertRaisesRegex(ValueError, "unresponsive label differs"):
            lesion_controls.check_against_lesion_results(results, 2, neuron, {"t": (clusters, None)})
        with self.assertRaisesRegex(ValueError, "u__zero_W: missing"):
            lesion_controls.check_against_lesion_results(results, 2, neuron, {"u": (clusters, unresponsive)})


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_network()
        self.before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.test_sets = {task: f"{task} trials" for task in TASKS}

    def test_neuron_sweep_draws_controls_from_the_condition_pool(self):
        cluster_info = cluster_info_fixture()
        entries = neuron_entries()
        pools = {side: {"all": lesion_ops.default_neuron_pool(entries[name]),
                        "responsive": lesion_controls.responsive_neuron_pool(entries[name])}
                 for side, name in (("pre", "input"), ("post", "hidden"))}
        sweep = lesion_controls.neuron_sweep(cluster_info, "normalized_k2", pools, "responsive")
        evaluate = RecordingEvaluate()
        lesion_record, random_record = lesion_controls.score_neuron_sweep(
            self.model, cluster_info, sweep, pools, self.test_sets, TASKS, repeats=5,
            rng=np.random.default_rng(1), evaluate=evaluate, log=lambda *_: None)
        self.assertEqual(np.asarray(lesion_record["ihtask_accs"]).shape, (2, 7))
        self.assertEqual(np.asarray(random_record["ihrandomtask_accs_raw"]).shape, (2, 7, 5))
        self.assertEqual(lesion_record["lesion_units"]["post_c3"], 4)
        self.assertEqual(lesion_record["unresponsive_conditions"], ["post_c3"])
        self.assertEqual(random_record["condition_pools"], sweep["condition_pools"])
        # Baselines are their own control; the two size-1 hidden clusters
        # share one control set.
        full = float(self.before["mp_layer1.W"].abs().sum())
        raw = random_record["ihrandomtask_accs_raw"][0]
        self.assertEqual(raw[0], [full] * 5)
        self.assertEqual(raw[4], raw[5])
        # Single-unit hidden controls came from the responsive units only;
        # the four-unit unresponsive condition fell back to every unit.
        single = {rows for rows in evaluate.zero_rows if len(rows) == 1}
        self.assertTrue(single <= {(0,), (1,)}, single)
        quads = {rows for rows in evaluate.zero_rows if len(rows) == 4}
        self.assertTrue(any(set(rows) - {0, 1} for rows in quads), quads)
        # Conditions once, repeats per distinct key, per task; model restored.
        self.assertEqual(len(evaluate.zero_rows),
                         lesion_controls.forward_count(2, sweep["keys"], 5))
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, self.before[name]), name)

    def test_synapse_sweep_scores_both_modes_within_the_pool(self):
        clusters, unresponsive = synapse_fixture()
        pools = {"all": np.arange(30),
                 "responsive": lesion_controls.responsive_synapse_pool(clusters, unresponsive, 30)}
        sweep = lesion_controls.synapse_sweep(clusters, pools, "responsive")
        for mode in lesion_ops.SYNAPSE_LESION_MODES:
            evaluate = RecordingEvaluate()
            record = lesion_controls.score_synapse_sweep(
                self.model, sweep, clusters, unresponsive, 5, mode, pools, self.test_sets, TASKS,
                repeats=3, rng=np.random.default_rng(2), evaluate=evaluate, log=lambda *_: None)
            self.assertEqual(record["mod_lesion_mode"], mode)
            self.assertEqual(record["all_comb_names_mod"], ["mod_nolesion", "mod_c1", "mod_c2", "mod_c3"])
            self.assertEqual(np.asarray(record["modrandomtask_accs_raw"]).shape, (2, 4, 3))
            self.assertEqual(record["condition_pools"], [None, "responsive", "all", "responsive"])
            lesioned = evaluate.zero_synapses if mode == "zero_W" else evaluate.frozen
            small = {syn for syn in lesioned if syn is not None and 0 < len(syn) <= 3}
            self.assertTrue(small and all(set(syn) <= {0, 1, 2, 7, 11} for syn in small), small)
            self.assertTrue(any(syn is not None and len(syn) == 25 for syn in lesioned))
            for name, tensor in self.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, self.before[name]), name)
            self.assertIsNone(getattr(self.model.mp_layer1, "_plasticity_freeze_mask", None))


class MainTests(unittest.TestCase):
    def test_main_writes_the_controls_pickle_next_to_the_lesion_pickle(self):
        cluster_info = cluster_info_fixture()
        clusters, unresponsive = synapse_fixture()
        type_key = "modulation_all_normalized"
        cluster_info_mod = {type_key: {"global_assignment_fixed_k2": None,
                                       "result_all_lst": [None, {"col_labels_by_k": {2: None}}]}}
        names = ["pre_nolesion", "pre_c1", "pre_c2", "post_nolesion", "post_c1", "post_c2", "post_c3"]
        sizes = dict(zip(names, [0, 2, 3, 0, 1, 1, 4]))
        record = {"mod_col_clusters": clusters, "unresponsive_label": unresponsive,
                  "mod_G_idx": 1, "mod_k": 2}
        results = {"fixed_k": 2,
                   "lesion": {"all_comb_names_lesion": names, "lesion_units": sizes},
                   "lesion_unnorm": {"all_comb_names_lesion": names, "lesion_units": sizes},
                   "mod_lesion": {f"{type_key}__zero_W": record, f"{type_key}__freeze_M": dict(record)}}
        model = tiny_network()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aname = lesion_ops.run_name(7, "L21e4")
            with patch.object(lesion_controls, "load_lesion_pickle", return_value=results), \
                    patch.object(lesion_ops, "load_cluster_info",
                                 return_value=({}, cluster_info_mod)), \
                    patch.object(lesion_ops, "load_network", return_value=(model, {}, TASKS)), \
                    patch.object(lesion_ops, "add_fixed_k_neuron_clusters",
                                 side_effect=lambda ci, k: ci.update(cluster_info) or list(cluster_info)), \
                    patch.object(lesion_ops, "fixed_k_synapse_clusters",
                                 return_value=(clusters, unresponsive, 2)), \
                    patch.object(lesion_ops, "make_test_sets",
                                 return_value={task: (torch.zeros(1, 2, 3), None, None) for task in TASKS}), \
                    patch.object(lesion_ops, "evaluate_accuracy", RecordingEvaluate()), \
                    patch("builtins.print"):
                path = lesion_controls.main(7, "L21e4", pool="responsive", repeats=2,
                                            test_seed=5, device=torch.device("cpu"), root=root)
            self.assertEqual(path, root / "multiple_tasks_perf" / aname
                             / f"lesion_controls_responsive_{aname}.pkl")
            with path.open("rb") as stream:
                output = pickle.load(stream)
        self.assertEqual(output["schema_version"], lesion_controls.SCHEMA_VERSION)
        self.assertEqual((output["control_pool"], output["repeat_num"], output["test_seed"]),
                         ("responsive", 2, 5))
        self.assertEqual(output["all_tasks"], TASKS)
        self.assertEqual(set(output["test_checksums"]), set(TASKS))
        self.assertEqual(output["lesion"]["all_comb_names_lesion"], names)
        self.assertEqual(output["random_lesion_unnorm"]["condition_pools"][-1], "all")
        self.assertEqual(np.asarray(output["random_lesion"]["ihrandomtask_accs_raw"]).shape, (2, 7, 2))
        self.assertEqual(set(output["mod_lesion"]), {f"{type_key}__zero_W", f"{type_key}__freeze_M"})
        self.assertEqual(output["mod_lesion"][f"{type_key}__freeze_M"]["mod_G_idx"], 1)
        np.testing.assert_array_equal(output["pools"]["synapses"][type_key], [0, 1, 2, 7, 11])
        np.testing.assert_array_equal(output["pools"]["neurons"]["normalized_k2"]["post"], [0, 1])

    def test_main_rejects_an_unknown_pool(self):
        with self.assertRaises(ValueError):
            lesion_controls.main(7, "L21e4", pool="silent")


class SelectionTests(unittest.TestCase):
    def test_runs_are_discovered_filtered_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, feature, lesioned in ((11, "L21e4", True), (12, "L21e4", True),
                                            (13, "L21e3", True), (14, "L21e4", False)):
                aname = lesion_ops.run_name(seed, feature)
                (root / "multiple_tasks_perf" / aname).mkdir(parents=True)
                if lesioned:
                    lesion_controls.lesion_results_path(aname, root).touch()
            lesion_controls.controls_pickle_path(lesion_ops.run_name(12, "L21e4"), "responsive", root).touch()
            self.assertEqual(lesion_controls.discover_runs(root),
                             [(11, "L21e4"), (12, "L21e4"), (13, "L21e3")])
            args = lesion_controls.parse_args(["--seed", "all", "--feature", "L21e4"])
            self.assertEqual(lesion_controls.select_runs(args, root), [(11, "L21e4"), (12, "L21e4")])
            args = lesion_controls.parse_args(["--seed", "all", "--feature", "L21e4", "--skip-completed"])
            self.assertEqual(lesion_controls.select_runs(args, root), [(11, "L21e4")])
            args = lesion_controls.parse_args(["--seed", "13"])
            self.assertEqual(lesion_controls.select_runs(args, root), [(13, "L21e3")])
            with self.assertRaises(SystemExit):
                lesion_controls.select_runs(lesion_controls.parse_args(["--seed", "14"]), root)
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                lesion_controls.parse_args(["--seed", "all"])
            args = lesion_controls.parse_args(["--seed", "11", "--pool", "all", "--repeats", "50",
                                               "--mod-types", "modulation_all_normalized"])
            self.assertEqual((args.pool, args.repeats, args.mod_types, args.test_seed),
                             ("all", 50, ["modulation_all_normalized"], 0))
            self.assertIsInstance(args, argparse.Namespace)


if __name__ == "__main__":
    unittest.main()
