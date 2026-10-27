"""Tests for the untrained control of the row-cluster analyses."""

import inspect
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np

import multiple_task_analysis
import multiple_task_diagnosis as diagnosis
import sibling_row_sharing as sharing
import untrained_clustering

ANAME = "everything_seed749_L21e4+hidden300+batch128+angle"


class OutputNameTests(unittest.TestCase):
    def test_untrained_outputs_get_their_own_name_and_directory(self):
        self.assertEqual(
            multiple_task_analysis.output_names(749, "everything", "L21e4+hidden300+batch128+angle"),
            (ANAME, Path("multiple_tasks_analysis") / ANAME))
        name, save_dir = multiple_task_analysis.output_names(
            749, "everything", "L21e4+hidden300+batch128+angle", untrained_init_seed=3)
        self.assertEqual(name, f"{ANAME}_init3")
        self.assertEqual(save_dir, Path("multiple_tasks_analysis") / "untrained" / f"{ANAME}_init3")

    def test_the_analysis_builds_the_network_through_the_shared_builder(self):
        source = inspect.getsource(multiple_task_analysis.main)
        self.assertIn("build_network(checkpoint, untrained_init_seed)", source)
        self.assertNotIn("load_state_dict", source)


class DriverTests(unittest.TestCase):
    def test_runs_every_init_seed_and_skips_the_clustered_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "untrained"
            with patch.object(multiple_task_analysis, "UNTRAINED_ANALYSIS_DIR", root), \
                    patch.object(multiple_task_analysis, "main") as analysis, \
                    patch("builtins.print"):
                for path in untrained_clustering.completion_markers(749, "L21e4", 1):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("done")
                self.assertTrue(untrained_clustering.is_completed(749, "L21e4", 1))
                self.assertFalse(untrained_clustering.is_completed(749, "L21e4", 0))
                untrained_clustering.main(["--feature", "L21e4", "--seed", "749",
                                           "--init-seed", "0", "1", "2", "--skip-completed"])
        self.assertEqual([call.kwargs["untrained_init_seed"] for call in analysis.call_args_list],
                         [0, 2])
        self.assertTrue(all(call.args == (749, "L21e4") for call in analysis.call_args_list))


def _write_run(root, aname, labels, names):
    run_dir = root / aname
    run_dir.mkdir(parents=True)
    neuron = {"result": {"row_tol_labels": labels}, "tb_break_name": names}
    with open(run_dir / f"cluster_info_{aname}.pkl", "wb") as stream:
        pickle.dump({"input_normalized": neuron, "hidden_normalized": neuron}, stream)
    modulation = {"result_all_lst": [None, {"row_tol_labels": labels}], "tb_break_name": names}
    with open(run_dir / f"cluster_info_mod_{aname}.pkl", "wb") as stream:
        pickle.dump({"modulation_all_normalized": modulation}, stream)


class UntrainedDiscoveryTests(unittest.TestCase):
    def test_untrained_runs_are_found_under_their_own_root(self):
        names = np.array(["fdgo-stim1", "fdanti-stim1", "fdgo-go1", "fdanti-go1"])
        labels = np.array([0, 1, 2, 2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for init_seed in (0, 1):
                _write_run(root, f"{ANAME}_init{init_seed}", labels, names)
            self.assertEqual(diagnosis._discover_anames("L21e4", root=root),
                             [f"{ANAME}_init0", f"{ANAME}_init1"])
            self.assertEqual(diagnosis._discover_anames("L21e4", 749, root=root),
                             [f"{ANAME}_init0", f"{ANAME}_init1"])
            self.assertEqual(diagnosis._discover_anames("L21e3", root=root), [])
            loaded = diagnosis._load_experiment(f"{ANAME}_init0", root=root)
            np.testing.assert_array_equal(loaded["modulation"]["row_labels"], labels)

    def test_the_sharing_analysis_of_untrained_runs_writes_its_own_tag(self):
        names = np.array(["fdgo-stim1", "fdanti-stim1", "fdgo-go1", "fdanti-go1"])
        labels = np.array([0, 1, 2, 2])
        with tempfile.TemporaryDirectory() as directory:
            root, out = Path(directory) / "untrained", Path(directory) / "out"
            for init_seed in (0, 1, 2):
                _write_run(root, f"{ANAME}_init{init_seed}", labels, names)
            out.mkdir()
            (out / "sibling_row_sharing_L21e4.pkl").write_text("trained")
            with patch.object(sharing, "UNTRAINED_DIR", root), \
                    patch.object(sharing, "OUT_DIR", out), \
                    patch("builtins.print"):
                sharing.main(feature="L21e4", untrained=True)
            self.assertEqual((out / "sibling_row_sharing_L21e4.pkl").read_text(), "trained")
            with open(out / "sibling_row_sharing_L21e4_untrained.pkl", "rb") as stream:
                payload = pickle.load(stream)
        self.assertTrue(payload["untrained"])
        self.assertIsNone(payload["seed"])
        self.assertEqual(len(payload["experiments"]), 3)
        self.assertEqual(set(payload["organization"]["representations"]),
                         {"input", "hidden", "modulation"})


if __name__ == "__main__":
    unittest.main()
