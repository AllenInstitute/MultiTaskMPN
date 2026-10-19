"""run_pipeline.py processes the saved multi-task checkpoints that match
--feature and --seed and, with --skip-completed, leaves out the runs whose
three steps already wrote their last output, so a rerun after more training
adds the pipeline outputs of the new checkpoints only."""

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from script_namespace import load_script_namespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "multiple_task" / "run_pipeline.py"
FUNCTIONS = ["run_name", "pipeline_markers", "is_completed", "discover_runs",
             "select_runs", "parse_args"]
CONSTANTS = ["CHECKPOINT_DIR", "ANALYSIS_DIR", "PERF_DIR", "NORM_DIR", "CHECKPOINT_PATTERN"]
# parse_args reads the sibling families and methods from the heavy analysis
# module; the stub keeps the test free of the training stack.
SIBLING_STUB = SimpleNamespace(SIBLING_FAMILIES={"delaydm1": None, "dmcgo": None},
                               SIBLING_METHODS=("gradient", "long_delay_endpoint"))


class RunPipelineSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_script_namespace(
            SOURCE, FUNCTIONS, CONSTANTS,
            extra={"argparse": argparse, "sibling_delay_analysis": SIBLING_STUB})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoints = self.root / self.ns["CHECKPOINT_DIR"]
        self.checkpoints.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _checkpoint(self, seed, feature, hidden=300):
        (self.checkpoints
         / f"savednet_everything_seed{seed}_{feature}+hidden{hidden}+batch128+angle.pt").touch()

    def _write_markers(self, seed, feature, steps=None, skip_paths=()):
        """Create the last output of the given steps (all steps by default)."""
        markers = self.ns["pipeline_markers"](self.ns["run_name"](seed, feature), self.root)
        for step, paths in markers.items():
            if steps is not None and step not in steps:
                continue
            for path in paths:
                if path.name in skip_paths:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

    def test_markers_are_the_last_output_of_each_step(self):
        aname = self.ns["run_name"](921, "L21e4")
        self.assertEqual(aname, "everything_seed921_L21e4+hidden300+batch128+angle")
        markers = self.ns["pipeline_markers"](aname, self.root)
        self.assertEqual(list(markers), ["multiple_task_analysis", "lesion", "lesion_plot"])
        self.assertEqual(
            [path.relative_to(self.root).as_posix() for paths in markers.values() for path in paths],
            [f"multiple_tasks_analysis/{aname}/cluster_info_{aname}.pkl",
             f"multiple_tasks_analysis/{aname}/cluster_info_mod_{aname}.pkl",
             f"multiple_tasks_perf/{aname}/lesion_prune_results_{aname}.pkl",
             f"multiple_tasks_norm/{aname}/lesion_interaction_norm_{aname}.pkl",
             f"multiple_tasks_norm/{aname}/lesion_interaction_unnorm_{aname}.pkl"])

    def test_discover_runs_orders_by_feature_then_seed(self):
        for seed, feature in [(101, "L21e4"), (11, "L21e4"), (86, "L21e3"), (921, "L21e4")]:
            self._checkpoint(seed, feature)
        self._checkpoint(5, "L21e4proj300bottleneck100", hidden=100)
        (self.checkpoints / "param_everything_seed7_L21e4+hidden300+batch128+angle_param.json").touch()
        (self.checkpoints / "savednet_everything_seed8_L21e4+hidden300+batch128+angle.pt.bak").touch()
        self.assertEqual(
            self.ns["discover_runs"](self.checkpoints),
            [(86, "L21e3"), (11, "L21e4"), (101, "L21e4"), (921, "L21e4"),
             (5, "L21e4proj300bottleneck100")])

    def test_skip_completed_leaves_out_runs_with_every_marker(self):
        runs = [(11, "L21e4"), (408, "L21e4"), (749, "L21e4"), (842, "L21e4"), (86, "L21e3")]
        self._write_markers(408, "L21e4")                                   # complete
        self._write_markers(749, "L21e4", steps=["multiple_task_analysis", "lesion"])
        self._write_markers(842, "L21e4",
                            skip_paths=["lesion_interaction_unnorm_"
                                        "everything_seed842_L21e4+hidden300+batch128+angle.pkl"])
        self._write_markers(86, "L21e3")                                    # other cohort
        selected, skipped, missing = self.ns["select_runs"](
            runs, feature="L21e4", skip_completed=True, root=self.root)
        self.assertEqual(selected, [(11, "L21e4"), (749, "L21e4"), (842, "L21e4")])
        self.assertEqual(skipped, [(408, "L21e4")])
        self.assertEqual(missing, [])
        self.assertTrue(self.ns["is_completed"](self.ns["run_name"](86, "L21e3"), self.root))
        self.assertFalse(self.ns["is_completed"](self.ns["run_name"](749, "L21e4"), self.root))

    def test_without_skip_completed_every_matching_run_is_selected(self):
        runs = [(11, "L21e4"), (408, "L21e4"), (86, "L21e3")]
        self._write_markers(408, "L21e4")
        selected, skipped, missing = self.ns["select_runs"](runs, feature="L21e4", root=self.root)
        self.assertEqual(selected, [(11, "L21e4"), (408, "L21e4")])
        self.assertEqual((skipped, missing), ([], []))

    def test_seed_order_and_missing_seeds_are_kept_with_skipping(self):
        runs = [(11, "L21e4"), (42, "L21e4"), (408, "L21e4"), (42, "L21e3")]
        self._write_markers(408, "L21e4")
        selected, skipped, missing = self.ns["select_runs"](
            runs, feature="L21e4", seeds=[408, 42, 999, 11, 42], skip_completed=True,
            root=self.root)
        self.assertEqual(selected, [(42, "L21e4"), (11, "L21e4")])
        self.assertEqual(skipped, [(408, "L21e4")])
        self.assertEqual(missing, [999])

    def test_skip_completed_needs_feature_or_seed(self):
        parse_args = self.ns["parse_args"]
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--skip-completed"])
        self.assertTrue(parse_args(["--skip-completed", "--feature", "L21e4"]).skip_completed)
        self.assertEqual(parse_args(["--skip-completed", "--seed", "11", "42"]).seed, [11, 42])
        self.assertFalse(parse_args(["--feature", "L21e4"]).skip_completed)


if __name__ == "__main__":
    unittest.main()
