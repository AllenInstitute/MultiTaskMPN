"""lesion_plot.py's control pool: the lesion-vs-control test, the task
specificity masks under each pool and the pool selection of main()."""

import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import t as t_dist

import lesion_plot

TASKS = ["a", "b", "c"]


def neuron_records(rng, effects, unresponsive="post_c3"):
    """lesion_unnorm / random_lesion_unnorm records of two input and three
    hidden clusters whose (task, cluster) effects are `effects` (T, C) on
    top of control repeats drawn around 0.9."""
    names = ["pre_nolesion", "pre_c1", "pre_c2", "post_nolesion", "post_c1", "post_c2", "post_c3"]
    keep = [1, 2, 4, 5, 6]
    raw = 0.9 + 0.01 * rng.standard_normal((len(TASKS), len(names), 12))
    accs = raw.mean(axis=2).copy()
    accs[:, keep] -= np.asarray(effects, dtype=float)
    return ({"all_comb_names_lesion": names, "all_tasks": TASKS, "ihtask_accs": accs,
             "lesion_units": dict(zip(names, [0, 4, 5, 0, 1, 2, 150])),
             "unresponsive_conditions": [unresponsive]},
            {"all_comb_names_lesion": names, "all_tasks": TASKS, "ihrandomtask_accs_raw": raw})


def synapse_record(rng, effects, mode="freeze_M"):
    names = ["mod_nolesion", "mod_c1", "mod_c2", "mod_c3"]
    raw = 0.9 + 0.01 * rng.standard_normal((len(TASKS), len(names), 12))
    accs = raw.mean(axis=2).copy()
    accs[:, 1:] -= np.asarray(effects, dtype=float)
    return {"all_comb_names_mod": names, "all_tasks": TASKS, "modtask_accs": accs,
            "modrandomtask_accs_raw": raw, "mod_lesion_mode": mode,
            "mod_col_clusters": {1: [0, 1], 2: [2, 3, 4], 3: list(range(5, 30))},
            "unresponsive_label": 3}


def results_fixture(rng, neuron_effects, synapse_effects, pool=None):
    lesion, random = neuron_records(rng, neuron_effects)
    key = f"{lesion_plot.TASK_SPECIFICITY_MOD_TYPE}__{lesion_plot.TASK_SPECIFICITY_MOD_MODE}"
    results = {"lesion_unnorm": lesion, "random_lesion_unnorm": random,
               "mod_lesion": {key: synapse_record(rng, synapse_effects)}, "repeat_num": 12}
    if pool is not None:
        results["control_pool"] = pool
        results["test_seed"] = 0
    return results


class ControlTestTests(unittest.TestCase):
    def test_prediction_t_against_the_control_repeats(self):
        raw = np.array([[[0.9, 0.92, 0.88, 0.91]]])        # (1 task, 1 cluster, 4 repeats)
        test = lesion_plot._control_test(np.array([[0.80]]), raw)
        effect = raw.mean() - 0.80
        sd = raw.std(ddof=1)
        statistic = effect / (sd * np.sqrt(1 + 1 / 4))
        np.testing.assert_allclose(test["effect"], [[effect]])
        np.testing.assert_allclose(test["statistic"], [[statistic]])
        np.testing.assert_allclose(test["p"], [[t_dist.sf(statistic, df=3)]])
        self.assertEqual(test["n_repeats"], 4)
        # A lesion no worse than its controls is not significant; a zero-spread
        # control set is floored, not divided by zero.
        self.assertGreater(lesion_plot._control_test(np.array([[0.95]]), raw)["p"][0, 0], 0.5)
        flat = lesion_plot._control_test(np.array([[0.9]]), np.full((1, 1, 5), 0.9))
        self.assertTrue(np.isfinite(flat["statistic"]).all())
        self.assertEqual(flat["effect"][0, 0], 0.0)
        with self.assertRaises(ValueError):
            lesion_plot._control_test(np.zeros((1, 2)), np.zeros((1, 1, 3)))

    def test_bh_fdr_mask_keeps_the_shape(self):
        p = np.array([[0.001, 0.2], [0.004, 0.9]])
        mask = lesion_plot._bh_fdr_mask(p, q=0.05)
        np.testing.assert_array_equal(mask, [[True, False], [True, False]])
        self.assertFalse(lesion_plot._bh_fdr_mask(np.array([0.3, 0.5])).any())

    def test_control_meta_reads_the_pool_fields(self):
        self.assertEqual(lesion_plot._control_meta({"repeat_num": 10}),
                         {"control_pool": "all", "control_repeats": 10, "control_test_seed": None})
        self.assertEqual(lesion_plot._control_meta({"control_pool": "responsive", "repeat_num": 30,
                                                    "test_seed": 4}),
                         {"control_pool": "responsive", "control_repeats": 30, "control_test_seed": 4})


class TaskSpecificityMaskTests(unittest.TestCase):
    def test_masks_drop_baselines_and_unresponsive_classes(self):
        rng = np.random.default_rng(0)
        neuron_effects = np.array([[0.2, 0.0, 0.2, 0.0, 0.3],
                                   [0.2, 0.0, 0.0, 0.0, 0.3],
                                   [0.0, 0.0, 0.2, 0.0, 0.3]])
        synapse_effects = np.array([[0.2, 0.0, 0.3], [0.2, 0.2, 0.3], [0.0, 0.0, 0.3]])
        results = results_fixture(rng, neuron_effects, synapse_effects)
        masks = lesion_plot._task_specificity_masks(results, TASKS)
        self.assertEqual(set(masks), {"input", "hidden", "modulation"})
        self.assertEqual(masks["input"]["cluster_labels"], ["i1", "i2"])
        self.assertEqual(masks["hidden"]["cluster_labels"], ["h1", "h2"])      # h3 unresponsive
        self.assertEqual(masks["modulation"]["cluster_labels"], ["c1", "c2"])  # c3 unresponsive
        np.testing.assert_array_equal(masks["input"]["sig"].sum(axis=0), [2, 0])
        np.testing.assert_array_equal(masks["hidden"]["sig"].sum(axis=0), [2, 0])
        np.testing.assert_array_equal(masks["modulation"]["sig"].sum(axis=0), [2, 1])
        np.testing.assert_array_equal(masks["input"]["sizes"], [4, 5])
        np.testing.assert_array_equal(masks["modulation"]["sizes"], [2, 3])
        self.assertEqual(masks["hidden"]["effect"].shape, (3, 2))
        # Without the synapse record or the neuron records, those types are absent.
        self.assertEqual(set(lesion_plot._task_specificity_masks(
            {"mod_lesion": results["mod_lesion"]}, TASKS)), {"modulation"})
        self.assertEqual(lesion_plot._task_specificity_masks({}, TASKS), {})

    def test_pool_counts_follow_each_pools_results(self):
        rng = np.random.default_rng(1)
        strong = np.full((3, 5), 0.2)
        weak = np.zeros((3, 5))
        by_pool = {"all": results_fixture(rng, strong, np.full((3, 3), 0.2)),
                   "responsive": results_fixture(rng, weak, np.zeros((3, 3)), pool="responsive")}
        reference = lesion_plot._task_specificity_masks(by_pool["all"], TASKS)
        with patch("builtins.print"):
            counts = lesion_plot._task_specificity_pool_counts(by_pool, TASKS, reference)
        np.testing.assert_array_equal(counts["all"]["hidden"], [3, 3])
        np.testing.assert_array_equal(counts["responsive"]["hidden"], [0, 0])
        np.testing.assert_array_equal(counts["responsive"]["modulation"], [0, 0])
        # A pool whose clusters differ is left out of that type.
        other = results_fixture(rng, weak, np.zeros((3, 3)))
        other["lesion_unnorm"]["all_comb_names_lesion"][1] = "pre_c9"
        with patch("builtins.print") as printed:
            counts = lesion_plot._task_specificity_pool_counts({"odd": other}, TASKS, reference)
        self.assertEqual(set(counts["odd"]), {"hidden", "modulation"})
        self.assertTrue(any("input clusters differ" in str(call) for call in printed.call_args_list))


class ControlPoolResultsTests(unittest.TestCase):
    def test_available_pools_are_overlaid_from_their_pickles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lesion_path = root / "lesion_prune_results_run.pkl"
            base = {"lesion": {"all_comb_names_lesion": ["pre_nolesion"], "all_tasks": TASKS},
                    "random_lesion": {"all_comb_names_lesion": ["pre_nolesion"], "all_tasks": TASKS},
                    "lesion_unnorm": {"all_comb_names_lesion": ["pre_nolesion"], "all_tasks": TASKS},
                    "random_lesion_unnorm": {"all_comb_names_lesion": ["pre_nolesion"],
                                             "all_tasks": TASKS},
                    "mod_lesion": {}, "repeat_num": 10}
            controls = {**{k: dict(v) for k, v in base.items() if isinstance(v, dict)},
                        "schema_version": 1, "control_pool": "responsive", "repeat_num": 30,
                        "test_seed": 0}
            with (root / "lesion_controls_responsive_run.pkl").open("wb") as stream:
                pickle.dump(controls, stream)
            by_pool = lesion_plot._control_pool_results(base, lesion_path)
            self.assertEqual(set(by_pool), {"all", "responsive"})
            self.assertIs(by_pool["all"], base)
            self.assertEqual(by_pool["responsive"]["control_pool"], "responsive")
            self.assertEqual(by_pool["responsive"]["repeat_num"], 30)
            with (root / "lesion_controls_all_run.pkl").open("wb") as stream:
                pickle.dump({**controls, "control_pool": "all", "repeat_num": 20}, stream)
            by_pool = lesion_plot._control_pool_results(base, lesion_path)
            self.assertEqual(set(by_pool), {"all", "responsive", "all_same_trials"})
            self.assertEqual(by_pool["all_same_trials"]["repeat_num"], 20)
            self.assertEqual(by_pool["all_same_trials"]["control_pool"], "all")

    def test_main_rejects_unknown_pools_and_missing_controls(self):
        """main() resolves its directories against the working directory, so
        the test runs inside a temporary one: a run without the requested
        pool's controls is refused BEFORE its output directory is cleared,
        and its existing caches survive."""
        self.assertEqual(lesion_plot.DEFAULT_CONTROL_POOL, "all")
        with self.assertRaises(ValueError):
            lesion_plot.main(921, "L21e4", control_pool="silent")
        aname = "everything_seed921_L21e4+hidden300+batch128+angle"
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                existing = Path("multiple_tasks_norm") / aname / "task_specificity_old.pkl"
                existing.parent.mkdir(parents=True)
                existing.write_bytes(b"keep")
                with patch.object(lesion_plot, "load_lesion_pickle", return_value={"repeat_num": 10}), \
                        patch.object(lesion_plot, "_control_pool_results", return_value={"all": {}}), \
                        patch("builtins.print"):
                    with self.assertRaisesRegex(FileNotFoundError, "lesion_controls.py --seed 921"):
                        lesion_plot.main(921, "L21e4", control_pool="responsive")
                self.assertEqual(existing.read_bytes(), b"keep")
            finally:
                os.chdir(previous)

    def test_unrequested_mismatching_pool_is_left_out(self):
        """A stale responsive pickle must not abort a --control-pool all run,
        but is an error when it is the pool asked for."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lesion_path = root / "lesion_prune_results_run.pkl"
            base = {key: {"all_comb_names_lesion": ["pre_nolesion"], "all_tasks": TASKS}
                    for key in ("lesion", "random_lesion", "lesion_unnorm", "random_lesion_unnorm")}
            base.update({"mod_lesion": {}, "repeat_num": 10})
            stale = {**{k: {"all_comb_names_lesion": ["pre_c7"], "all_tasks": TASKS}
                        for k in ("lesion", "random_lesion", "lesion_unnorm", "random_lesion_unnorm")},
                     "mod_lesion": {}, "schema_version": 1, "control_pool": "responsive",
                     "repeat_num": 30, "test_seed": 0}
            with (root / "lesion_controls_responsive_run.pkl").open("wb") as stream:
                pickle.dump(stale, stream)
            with patch("builtins.print") as printed:
                by_pool = lesion_plot._control_pool_results(base, lesion_path, required="all")
            self.assertEqual(set(by_pool), {"all"})
            self.assertTrue(any("responsive left out" in str(call) for call in printed.call_args_list))
            with self.assertRaisesRegex(ValueError, "condition names differ"):
                lesion_plot._control_pool_results(base, lesion_path, required="responsive")


if __name__ == "__main__":
    unittest.main()
