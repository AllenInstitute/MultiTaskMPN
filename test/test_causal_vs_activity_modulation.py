"""The synapse side of lesion_plot's causal-vs-activity pairing: |W| Var(M)
task profiles against freeze_M lesion profiles without the unresponsive class,
saved in the cache format paper_plot.py reads."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import lesion_plot
import paper_plot

TASKS = ["fdgo", "reactgo", "delaygo", "dmsgo"]
EFFECT_KEY = f"{lesion_plot.CAUSAL_VS_ACTIVITY_MOD_TYPE}__{lesion_plot.CAUSAL_VS_ACTIVITY_MOD_MODE}"


def clustering_entry(rng, n_synapses=40):
    """Two periods per task; the stored feature is signed like W * Var(M)."""
    rows = [f"{task}-{period}" for period in ("stim1", "go1") for task in TASKS]
    magnitude = rng.random((len(rows), n_synapses))
    sign = np.where(rng.random(n_synapses) < 0.5, -1.0, 1.0)
    return {"cell_vars_rules_sorted_norm": magnitude * sign, "tb_break_name": np.array(rows)}


def effect_record(rng):
    """Normalized effects in another task order, with an unresponsive class
    (mod_c4) that impairs every task by the same large amount."""
    order = list(reversed(TASKS))
    effect = rng.random((len(order), 4)) * 0.1
    effect[:, 3] = 0.9
    return {"effect": effect, "tasks": order,
            "conditions": ["mod_c1", "mod_c2", "mod_c3", "mod_c4"],
            "definition": "random_minus_lesion", "units": "fraction",
            "mod_lesion_mode": lesion_plot.CAUSAL_VS_ACTIVITY_MOD_MODE}


class ModulationPairingTests(unittest.TestCase):
    def test_period_rows_of_a_task_are_averaged(self):
        features = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        names = ["a-stim1", "b-stim1", "a-go1", "b-go1"]
        np.testing.assert_allclose(lesion_plot._period_averaged_task_profiles(features, names, ["b", "a"]),
                                   [[5., 6.], [3., 4.]])

    def test_tuning_uses_magnitude_and_lesions_drop_the_unresponsive_class(self):
        rng = np.random.default_rng(0)
        entry, record = clustering_entry(rng), effect_record(rng)
        result = lesion_plot._modulation_causal_vs_activity(entry, record, TASKS, unresponsive_label=4)
        profiles = lesion_plot._period_averaged_task_profiles(
            np.abs(entry["cell_vars_rules_sorted_norm"]), entry["tb_break_name"], TASKS)
        np.testing.assert_allclose(result["activity_similarity"], np.corrcoef(profiles))
        rows = [record["tasks"].index(task) for task in TASKS]
        expected = np.corrcoef(np.asarray(record["effect"])[rows][:, :3])
        np.testing.assert_allclose(result["causal_similarity"], expected)
        self.assertEqual(result["excluded_conditions"], ["mod_c4"])
        iu = np.triu_indices(len(TASKS), k=1)
        np.testing.assert_allclose(result["causal_pairs"], expected[iu])
        self.assertEqual(result["association"]["n_clusters"], len(TASKS))

    def test_without_an_unresponsive_class_every_cluster_is_kept(self):
        rng = np.random.default_rng(1)
        entry, record = clustering_entry(rng), effect_record(rng)
        result = lesion_plot._modulation_causal_vs_activity(entry, record, TASKS, unresponsive_label=None)
        rows = [record["tasks"].index(task) for task in TASKS]
        np.testing.assert_allclose(result["causal_similarity"],
                                   np.corrcoef(np.asarray(record["effect"])[rows]))
        self.assertEqual(result["excluded_conditions"], [])

    def test_unknown_unresponsive_class_is_an_error(self):
        rng = np.random.default_rng(2)
        with self.assertRaises(ValueError):
            lesion_plot._modulation_causal_vs_activity(
                clustering_entry(rng), effect_record(rng), TASKS, unresponsive_label=9)

    def test_saved_cache_is_read_by_paper_plot(self):
        rng = np.random.default_rng(3)
        aname = "everything_seed1_L21e4+hidden300+batch128+angle"
        results = {"mod_lesion": {EFFECT_KEY: {"unresponsive_label": 4}}}
        effects = {"entries": {EFFECT_KEY: effect_record(rng)}}
        cluster_info_mod = {lesion_plot.CAUSAL_VS_ACTIVITY_MOD_TYPE: clustering_entry(rng)}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(lesion_plot, "_plot_causal_vs_activity_diagnostic") as diagnostic:
            lesion_plot._save_modulation_causal_vs_activity(
                results, effects, cluster_info_mod, TASKS, aname, directory)
            path = Path(directory) / f"causal_vs_activity_tasksim_modulation_{aname}.pkl"
            with path.open("rb") as handle:
                saved = pickle.load(handle)
            entry = paper_plot._read_causal_vs_activity(path, "modulation", aname)
        diagnostic.assert_called_once()
        self.assertEqual(saved["mod_lesion_mode"], "freeze_M")
        self.assertEqual(saved["excluded_conditions"], ["mod_c4"])
        np.testing.assert_allclose(entry["x"], saved["activity_pairs"])
        self.assertEqual(entry["n_tasks"], len(TASKS))

    def test_missing_freeze_m_lesion_saves_nothing(self):
        rng = np.random.default_rng(4)
        cluster_info_mod = {lesion_plot.CAUSAL_VS_ACTIVITY_MOD_TYPE: clustering_entry(rng)}
        with tempfile.TemporaryDirectory() as directory:
            lesion_plot._save_modulation_causal_vs_activity(
                {"mod_lesion": {}}, {"entries": {}}, cluster_info_mod, TASKS, "run", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
