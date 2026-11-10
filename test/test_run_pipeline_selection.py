"""run_pipeline.py processes the saved multi-task checkpoints that match
--feature and --seed and, with --skip-completed, leaves out the runs whose
steps already wrote their last output, so a rerun after more training
adds the pipeline outputs of the new checkpoints only; --resume skips the
steps whose outputs use the current variance estimator and are newer than their inputs.
The lesion plots are a step only with --with-plots; otherwise they belong to
run_pipeline_analysis.py."""

import argparse
import pickle
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import lesion_cache
from core.task_variance import (VARIANCE_METHOD, NEURON_VARIANTS, LESION_MODULATION_TYPES,
                                cluster_caches_current)
from script_namespace import load_script_namespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "multiple_task" / "run_pipeline.py"
FUNCTIONS = ["run_name", "pipeline_steps", "lesion_plot_pools", "pipeline_markers",
             "step_is_completed", "is_completed",
             "discover_runs", "select_runs", "run_pipeline", "parse_args", "main"]
CONSTANTS = ["CHECKPOINT_DIR", "ANALYSIS_DIR", "PERF_DIR", "CHECKPOINT_PATTERN",
             "DEFAULT_CONTROL_POOL", "DEFAULT_CONTROL_REPEATS", "STEPS", "PLOT_STEP"]
# lesion_controls' pool constants, read by the markers and the CLI defaults.
CONTROLS_STUB = SimpleNamespace(CONTROL_POOLS=("responsive", "all"), DEFAULT_POOL="responsive",
                                DEFAULT_REPEATS=30, main=Mock())
# parse_args reads the sibling families and methods from the heavy analysis
# module; the stub keeps the test free of the training stack.
SIBLING_STUB = SimpleNamespace(SIBLING_FAMILIES={"delaydm1": None, "dmcgo": None},
                               SIBLING_METHODS=("gradient", "long_delay_endpoint"))


class RunPipelineStartupTests(unittest.TestCase):
    def test_import_and_parse_without_test_bootstrap_or_pythonpath(self):
        # Direct script execution starts with multiple_task/ on sys.path.
        # A separate interpreter avoids test/_bootstrap.py masking missing paths.
        script = """
import run_pipeline
args = run_pipeline.parse_args([
    "--feature", "L21e4", "--resume", "--skip-completed",
])
assert args.feature == "L21e4"
assert args.resume and args.skip_completed
assert args.control_pool == "responsive"
assert args.control_repeats == 30
"""
        completed = subprocess.run(
            [sys.executable, "-E", "-c", script], cwd=SOURCE.parent,
            env={**os.environ, "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": ""},
            text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class RunPipelineSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_script_namespace(
            SOURCE, FUNCTIONS, CONSTANTS,
            extra={"argparse": argparse, "pickle": pickle, "sibling_delay_analysis": SIBLING_STUB,
                   "lesion_controls": CONTROLS_STUB, "lesion_cache": lesion_cache,
                   "cluster_caches_current": cluster_caches_current})

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
                if step == "multiple_task_analysis":
                    keys = LESION_MODULATION_TYPES if path.name.startswith("cluster_info_mod_") else NEURON_VARIANTS
                    path.write_bytes(pickle.dumps({key: {"variance_method": VARIANCE_METHOD} for key in keys}))
                else:
                    payload = {"variance_method": VARIANCE_METHOD}
                    if step == "lesion_controls":
                        payload["repeat_num"] = 30
                    path.write_bytes(pickle.dumps(payload))

    def test_markers_are_the_last_output_of_each_step(self):
        aname = self.ns["run_name"](921, "L21e4")
        self.assertEqual(aname, "everything_seed921_L21e4+hidden300+batch128+angle")
        markers = self.ns["pipeline_markers"](aname, self.root)
        self.assertEqual(list(markers), list(self.ns["STEPS"]))
        self.assertEqual(list(markers), ["multiple_task_analysis", "lesion", "lesion_controls",
                                         "lesion_plot"])
        # The plots are a pipeline step only on request.
        self.assertEqual(self.ns["pipeline_steps"](),
                         ("multiple_task_analysis", "lesion", "lesion_controls"))
        self.assertEqual(self.ns["pipeline_steps"](with_plots=True), self.ns["STEPS"])
        self.assertEqual(
            [path.relative_to(self.root).as_posix() for paths in markers.values() for path in paths],
            [f"multiple_tasks_analysis/{aname}/cluster_info_{aname}.pkl",
             f"multiple_tasks_analysis/{aname}/cluster_info_mod_{aname}.pkl",
             f"multiple_tasks_perf/{aname}/lesion_prune_results_{aname}.pkl",
             f"multiple_tasks_perf/{aname}/lesion_controls_responsive_{aname}.pkl",
             f"multiple_tasks_norm/{aname}/lesion_interaction_norm_{aname}.pkl",
             f"multiple_tasks_norm/{aname}/lesion_interaction_unnorm_{aname}.pkl",
             f"multiple_tasks_norm_responsive/{aname}/lesion_interaction_norm_{aname}.pkl",
             f"multiple_tasks_norm_responsive/{aname}/lesion_interaction_unnorm_{aname}.pkl"])
        # The controls marker follows the pool the pipeline is run with, and
        # with the all pool lesion_plot runs once, into multiple_tasks_norm/.
        markers = self.ns["pipeline_markers"](aname, self.root, "all")
        self.assertEqual(markers["lesion_controls"][0].name, f"lesion_controls_all_{aname}.pkl")
        self.assertEqual([path.parent.parent.name for path in markers["lesion_plot"]],
                         ["multiple_tasks_norm", "multiple_tasks_norm"])
        self.assertEqual(self.ns["lesion_plot_pools"]("all"), ("all",))
        self.assertEqual(self.ns["lesion_plot_pools"]("responsive"), ("all", "responsive"))

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

    def test_lambda_tagged_runs_are_found_but_kept_out_of_the_base_feature(self):
        """multiple_task.py tags runs with a non-0.99 lambda (L21e4lam0.9). The
        dot must not hide them from discovery, and the base feature must not
        pick them up."""
        self._checkpoint(11, "L21e4")
        self._checkpoint(12, "L21e4lam0.9")
        runs = self.ns["discover_runs"](self.checkpoints)
        self.assertEqual(runs, [(11, "L21e4"), (12, "L21e4lam0.9")])
        select = self.ns["select_runs"]
        self.assertEqual(select(runs, feature="L21e4", root=self.root)[0], [(11, "L21e4")])
        self.assertEqual(select(runs, feature="L21e4lam0.9", root=self.root)[0],
                         [(12, "L21e4lam0.9")])

    def test_skip_completed_leaves_out_runs_with_every_marker(self):
        runs = [(11, "L21e4"), (408, "L21e4"), (749, "L21e4"), (842, "L21e4"), (86, "L21e3")]
        self._write_markers(408, "L21e4")                                   # complete
        self._write_markers(749, "L21e4", steps=["multiple_task_analysis", "lesion"])
        self._write_markers(842, "L21e4",
                            skip_paths=["lesion_interaction_unnorm_"
                                        "everything_seed842_L21e4+hidden300+batch128+angle.pkl"])
        self._write_markers(86, "L21e3")                                    # other cohort
        # Seed 842 lacks one plot cache: complete for the pipeline, whose
        # steps end with the controls, but not when the plots are a step.
        selected, skipped, missing = self.ns["select_runs"](
            runs, feature="L21e4", skip_completed=True, root=self.root)
        self.assertEqual(selected, [(11, "L21e4"), (749, "L21e4")])
        self.assertEqual(skipped, [(408, "L21e4"), (842, "L21e4")])
        self.assertEqual(missing, [])
        selected, skipped, _ = self.ns["select_runs"](
            runs, feature="L21e4", skip_completed=True, root=self.root, with_plots=True)
        self.assertEqual(selected, [(11, "L21e4"), (749, "L21e4"), (842, "L21e4")])
        self.assertEqual(skipped, [(408, "L21e4")])
        aname = self.ns["run_name"](842, "L21e4")
        self.assertTrue(self.ns["is_completed"](aname, self.root))
        self.assertFalse(self.ns["is_completed"](aname, self.root, with_plots=True))
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

    def _stubbed_pipeline(self):
        """(namespace, calls) with every step's main recording (step, args, kwargs)."""
        calls = []
        stubs = {name: SimpleNamespace(main=Mock(side_effect=lambda *a, _n=name, **k: calls.append((_n, a, k))))
                 for name in ("multiple_task_analysis", "lesion", "lesion_plot")}
        controls = SimpleNamespace(**vars(CONTROLS_STUB))
        controls.main = Mock(side_effect=lambda *a, **k: calls.append(("lesion_controls", a, k)))
        ns = load_script_namespace(
            SOURCE, FUNCTIONS, CONSTANTS,
            extra={"argparse": argparse, "pickle": pickle, "sibling_delay_analysis": SIBLING_STUB,
                   "lesion_controls": controls, "lesion_cache": lesion_cache,
                   "cluster_caches_current": cluster_caches_current,
                   "time": __import__("time"), **stubs})
        return ns, calls

    def _run_from_root(self, run_pipeline, *args, **kwargs):
        """run_pipeline with the fixture root as the working directory
        (pipeline_markers resolves against ".")."""
        import os
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            with patch("builtins.print"):
                run_pipeline(*args, **kwargs)
        finally:
            os.chdir(previous)

    def test_resume_skips_the_steps_whose_outputs_exist(self):
        """A run lesioned before step 3 existed gets its controls (and, with
        the plots as a step, its plots) without a second clustering or lesion
        pass; without --resume every step runs."""
        ns, calls = self._stubbed_pipeline()
        self._write_markers(749, "L21e4", steps=["multiple_task_analysis", "lesion"])
        self._run_from_root(ns["run_pipeline"], 749, "L21e4", resume=True,
                            control_pool="responsive", control_repeats=7, with_plots=True)
        # lesion_plot runs for lesion.py's controls first (the historical
        # caches), then for the activity-matched pool.
        self.assertEqual([name for name, _, _ in calls],
                         ["lesion_controls", "lesion_plot", "lesion_plot"])
        self.assertEqual(calls[0][1:], ((749, "L21e4"), {"pool": "responsive", "repeats": 7}))
        self.assertEqual(calls[1][1:], ((749, "L21e4"), {"control_pool": "all"}))
        self.assertEqual(calls[2][1:], ((749, "L21e4"), {"control_pool": "responsive"}))
        calls.clear()
        self._run_from_root(ns["run_pipeline"], 749, "L21e4", resume=False, with_plots=True)
        self.assertEqual([name for name, _, _ in calls],
                         ["multiple_task_analysis", "lesion", "lesion_controls", "lesion_plot",
                          "lesion_plot"])
        self.assertEqual(calls[1][1:], ((749, "L21e4"), {}))
        self.assertEqual(calls[2][2], {"pool": "responsive", "repeats": 30})

    def test_plots_are_left_to_the_post_analysis_by_default(self):
        """Without with_plots the pipeline ends with the controls and names
        run_pipeline_analysis.py; a run with those three outputs runs nothing
        even when it has no plot cache."""
        ns, calls = self._stubbed_pipeline()

        def cluster(*args, **kwargs):
            calls.append(("multiple_task_analysis", args, kwargs))
            self._write_markers(749, "L21e4", steps=["multiple_task_analysis"])

        ns["multiple_task_analysis"].main.side_effect = cluster
        self._run_from_root(ns["run_pipeline"], 749, "L21e4", resume=False)
        self.assertEqual([name for name, _, _ in calls],
                         ["multiple_task_analysis", "lesion", "lesion_controls"])
        calls.clear()
        self._write_markers(749, "L21e4",
                            steps=["multiple_task_analysis", "lesion", "lesion_controls"])
        import os
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            with patch("builtins.print") as printed:
                ns["run_pipeline"](749, "L21e4", resume=True)
        finally:
            os.chdir(previous)
        self.assertEqual(calls, [])
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertIn("Step 3/3", output)
        self.assertIn("run_pipeline_analysis.py --feature L21e4", output)

    def test_resume_replots_after_computing_controls(self):
        """A run plotted before the controls step existed has lesion_plot
        outputs already; new controls must still get new plots, while a run
        with every marker runs nothing."""
        ns, calls = self._stubbed_pipeline()
        self._write_markers(749, "L21e4",
                            steps=["multiple_task_analysis", "lesion", "lesion_plot"])
        self._run_from_root(ns["run_pipeline"], 749, "L21e4", resume=True, with_plots=True)
        self.assertEqual([name for name, _, _ in calls],
                         ["lesion_controls", "lesion_plot", "lesion_plot"])
        calls.clear()
        self._write_markers(749, "L21e4")
        self._run_from_root(ns["run_pipeline"], 749, "L21e4", resume=True, with_plots=True)
        self.assertEqual(calls, [])
        # With the all pool the plots are made once, with lesion.py's controls.
        self._run_from_root(ns["run_pipeline"], 749, "L21e4", resume=False, control_pool="all",
                            with_plots=True)
        self.assertEqual([(name, kwargs) for name, _, kwargs in calls][-2:],
                         [("lesion_controls", {"pool": "all", "repeats": 30}),
                          ("lesion_plot", {"control_pool": "all"})])

    def test_main_warns_before_redoing_lesioned_runs_without_resume(self):
        """Selecting a lesioned run without --resume redoes hours of work; the
        warning names it, and --resume silences it."""
        ns, calls = self._stubbed_pipeline()
        self._checkpoint(749, "L21e4")
        self._checkpoint(11, "L21e4")
        self._write_markers(749, "L21e4", steps=["multiple_task_analysis", "lesion"])
        import os
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            with patch("sys.argv", ["run_pipeline.py", "--feature", "L21e4"]), \
                    patch("builtins.print") as printed:
                ns["main"]()
            warnings = [str(call) for call in printed.call_args_list if "WARNING" in str(call)]
            self.assertEqual(len(warnings), 1)
            self.assertIn("(749, 'L21e4')", warnings[0])
            self.assertNotIn("(11, 'L21e4')", warnings[0])
            self.assertEqual([name for name, _, _ in calls].count("multiple_task_analysis"), 2)
            calls.clear()
            with patch("sys.argv", ["run_pipeline.py", "--feature", "L21e4", "--resume"]), \
                    patch("builtins.print") as printed:
                ns["main"]()
            self.assertFalse(any("WARNING" in str(call) for call in printed.call_args_list))
        finally:
            os.chdir(previous)

    def test_old_variance_caches_are_not_completed_and_resume_rebuilds_dependents(self):
        ns, calls = self._stubbed_pipeline()
        self._write_markers(749, "L21e4")
        aname = ns["run_name"](749, "L21e4")
        markers = ns["pipeline_markers"](aname, self.root)
        for path in markers["multiple_task_analysis"]:
            data = pickle.loads(path.read_bytes())
            for entry in data.values():
                entry.pop("variance_method")
            path.write_bytes(pickle.dumps(data))
        self.assertFalse(ns["is_completed"](aname, self.root))
        def rebuild(*args, **kwargs):
            calls.append(("multiple_task_analysis", args, kwargs))
            self._write_markers(749, "L21e4", steps=["multiple_task_analysis"])
        ns["multiple_task_analysis"].main.side_effect = rebuild
        self._run_from_root(ns["run_pipeline"], 749, "L21e4", resume=True, with_plots=True)
        self.assertEqual([name for name, _, _ in calls],
                         ["multiple_task_analysis", "lesion", "lesion_controls", "lesion_plot", "lesion_plot"])

    def test_current_clusters_do_not_make_older_lesions_current(self):
        self._write_markers(749, "L21e4")
        aname = self.ns["run_name"](749, "L21e4")
        markers = self.ns["pipeline_markers"](aname, self.root)
        for path in markers["lesion"]:
            os.utime(path, ns=(1, 1))
        self.assertFalse(self.ns["is_completed"](aname, self.root))

    def test_newer_unmarked_lesion_results_are_not_current(self):
        self._write_markers(749, "L21e4")
        aname = self.ns["run_name"](749, "L21e4")
        path = self.ns["pipeline_markers"](aname, self.root)["lesion"][0]
        path.write_bytes(pickle.dumps({"lesion": {}}))
        self.assertFalse(self.ns["step_is_completed"](
            "lesion", self.ns["pipeline_markers"](aname, self.root)))

    def test_control_pool_flags(self):
        parse_args = self.ns["parse_args"]
        args = parse_args(["--feature", "L21e4"])
        self.assertEqual((args.control_pool, args.control_repeats, args.resume, args.with_plots),
                         ("responsive", 30, False, False))
        self.assertTrue(parse_args(["--feature", "L21e4", "--with-plots"]).with_plots)
        args = parse_args(["--feature", "L21e4", "--control-pool", "all", "--control-repeats",
                           "12", "--resume", "--skip-completed"])
        self.assertEqual((args.control_pool, args.control_repeats, args.resume), ("all", 12, True))
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--control-pool", "silent"])

    def test_requested_repeats_invalidate_controls_and_plots_only(self):
        ns, calls = self._stubbed_pipeline()
        self._write_markers(749, "L21e4")
        aname = ns["run_name"](749, "L21e4")
        self.assertTrue(ns["is_completed"](aname, self.root, control_repeats=30))
        self.assertFalse(ns["is_completed"](aname, self.root, control_repeats=7))
        self._run_from_root(ns["run_pipeline"], 749, "L21e4", resume=True, control_repeats=7,
                            with_plots=True)
        self.assertEqual([name for name, _, _ in calls],
                         ["lesion_controls", "lesion_plot", "lesion_plot"])
        self.assertEqual(calls[0][2]["repeats"], 7)

    def test_cli_skip_completed_uses_requested_control_repeats(self):
        ns, calls = self._stubbed_pipeline()
        self._checkpoint(749, "L21e4")
        self._write_markers(749, "L21e4")
        with patch("sys.argv", ["run_pipeline.py", "--feature", "L21e4", "--resume",
                                "--skip-completed", "--control-repeats", "7"]):
            self._run_from_root(ns["main"])
        self.assertEqual([name for name, _, _ in calls], ["lesion_controls"])
        self.assertEqual(calls[0][2]["repeats"], 7)
        calls.clear()
        with patch("sys.argv", ["run_pipeline.py", "--feature", "L21e4", "--resume",
                                "--skip-completed", "--control-repeats", "7", "--with-plots"]):
            self._run_from_root(ns["main"])
        self.assertEqual([name for name, _, _ in calls],
                         ["lesion_controls", "lesion_plot", "lesion_plot"])

    def test_skip_completed_needs_feature_or_seed(self):
        parse_args = self.ns["parse_args"]
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--skip-completed"])
        self.assertTrue(parse_args(["--skip-completed", "--feature", "L21e4"]).skip_completed)
        self.assertEqual(parse_args(["--skip-completed", "--seed", "11", "42"]).seed, [11, 42])
        self.assertFalse(parse_args(["--feature", "L21e4"]).skip_completed)


if __name__ == "__main__":
    unittest.main()
