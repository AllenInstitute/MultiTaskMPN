"""run_pipeline_analysis.py: the post-analysis commands of one cohort in
their dependency order, stage and seed selection, the runs it leaves out,
dry runs and the failure stop."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

import run_pipeline_analysis as pipeline

PYTHON = [pipeline.sys.executable, "-u"]
FEATURE = ["--feature", "L21e4"]


def _lesioned(root, seed, feature="L21e4", controls=("responsive",)):
    """A run directory with a lesion pickle and the given pools' controls."""
    aname = pipeline.run_name(seed, feature)
    directory = Path(root) / pipeline.PERF_DIR / aname
    directory.mkdir(parents=True)
    (directory / f"lesion_prune_results_{aname}.pkl").touch()
    for pool in controls:
        (directory / f"lesion_controls_{pool}_{aname}.pkl").touch()


def _plot(seed, pool):
    return PYTHON + [pipeline.LESION_PLOT, "--seed", str(seed), *FEATURE, "--control-pool", pool]


class StageCommandTests(unittest.TestCase):
    def setUp(self):
        # Never look at the real cache directories: every test works in a temp dir.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    def commands(self, argv):
        args = pipeline.parse_args(list(argv))
        commands, notes = [], []
        for stage in args.stages:
            stage_commands, stage_notes = pipeline.stage_commands(stage, args)
            commands += stage_commands
            notes += stage_notes
        return commands, notes

    def test_default_run_is_the_documented_sequence_without_figures(self):
        _lesioned(self.root, 749)
        _lesioned(self.root, 11)
        _lesioned(self.root, 86, feature="L21e3")              # another cohort
        _lesioned(self.root, 12, feature="L21e4lam0.9")        # a tagged cohort is not L21e4
        (self.root / pipeline.PERF_DIR / pipeline.run_name(5, "L21e4")).mkdir()   # no lesion pickle
        commands, notes = self.commands(FEATURE)
        self.assertEqual(commands, [
            _plot(11, "all"), _plot(11, "responsive"), _plot(749, "all"), _plot(749, "responsive"),
            PYTHON + [pipeline.ROW_SHARING, *FEATURE],
            PYTHON + [pipeline.VARIANCE, "--seed", "all", *FEATURE, "--skip-completed"],
            PYTHON + [pipeline.VS_LESION],
        ])
        self.assertEqual(notes, [])
        self.assertEqual(pipeline.lesioned_seeds("L21e4"), [11, 749])
        self.assertEqual(pipeline.lesioned_seeds("L21e4lam0.9"), [12])

    def test_runs_without_their_inputs_are_left_out_and_named(self):
        _lesioned(self.root, 749)
        _lesioned(self.root, 303, controls=())                 # controls still running
        commands, notes = self.commands([*FEATURE, "--stages", "lesion_plot"])
        self.assertEqual(commands, [_plot(303, "all"), _plot(749, "all"), _plot(749, "responsive")])
        self.assertEqual(notes, ["lesion_plot left out seed 303, responsive pool: "
                                 "no responsive controls pickle"])
        # A requested seed keeps its order and is named when it has no lesions.
        commands, notes = self.commands([*FEATURE, "--stages", "lesion_plot", "--seed", "749", "692",
                                         "--control-pool", "all"])
        self.assertEqual(commands, [_plot(749, "all")])
        self.assertEqual(notes, ["lesion_plot left out seed 692, all pool: no lesion pickle"])

    def test_options_reach_the_commands_and_stages_keep_their_order(self):
        _lesioned(self.root, 749)
        commands, _ = self.commands([*FEATURE, "--stages", "figures", "variance", "row_sharing",
                                     "--seed", "749", "921", "--untrained", "--force"])
        self.assertEqual(commands, [
            PYTHON + [pipeline.ROW_SHARING, *FEATURE],
            PYTHON + [pipeline.ROW_SHARING, *FEATURE, "--untrained"],
            PYTHON + [pipeline.VARIANCE, "--seed", "749", "921", *FEATURE],
            PYTHON + [pipeline.PAPER_PLOT, "multiple_tasks", "lesion", "--multitask-l2", "1e4"],
            PYTHON + [pipeline.PAPER_PLOT, "lesion", "--multitask-l2", "1e4",
                      "--control-pool", "responsive"],
        ])
        commands, _ = self.commands([*FEATURE, "--stages", "figures", "--control-pool", "all"])
        self.assertEqual(commands, [
            PYTHON + [pipeline.PAPER_PLOT, "multiple_tasks", "lesion", "--multitask-l2", "1e4"]])
        self.assertEqual(pipeline.control_pools("all"), ("all",))
        self.assertEqual(pipeline.control_pools("responsive"), ("all", "responsive"))
        with patch("sys.stderr"):
            for argv in ([], [*FEATURE, "--stages", "shift"], [*FEATURE, "--control-pool", "silent"],
                         ["--feature", "L21e4lam0.9", "--stages", "figures"]):
                with self.assertRaises(SystemExit):
                    pipeline.parse_args(argv)

    def test_dry_run_prints_every_command_and_runs_nothing(self):
        _lesioned(self.root, 749)
        _lesioned(self.root, 303, controls=())
        with patch.object(pipeline, "run_command") as run, patch("builtins.print") as output:
            pipeline.main(pipeline.parse_args([*FEATURE, "--dry-run"]))
        run.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn(f"$ {' '.join(_plot(749, 'responsive'))}", printed)
        self.assertIn("Stage 4/4: vs_lesion", printed)
        self.assertIn("listed for L21e4", printed)
        # What was left out is repeated at the end.
        self.assertEqual(printed.count("left out seed 303, responsive pool"), 2)

    def test_a_failing_command_stops_the_pipeline(self):
        _lesioned(self.root, 749)
        ran = []

        def fail_on_row_sharing(command):
            ran.append(command)
            if pipeline.ROW_SHARING in command:
                raise subprocess.CalledProcessError(1, command)

        with patch.object(pipeline, "run_command", side_effect=fail_on_row_sharing), \
                patch("builtins.print"), self.assertRaises(subprocess.CalledProcessError):
            pipeline.main(pipeline.parse_args(FEATURE))
        self.assertEqual(ran, [_plot(749, "all"), _plot(749, "responsive"),
                               PYTHON + [pipeline.ROW_SHARING, *FEATURE]])

    def test_run_command_echoes_output_and_raises_on_failure(self):
        with patch("builtins.print") as output:
            pipeline.run_command([pipeline.sys.executable, "-c", "print('hello')"])
        self.assertIn("hello\n", [call.args[0] for call in output.call_args_list if call.args])
        with patch("builtins.print"), self.assertRaises(subprocess.CalledProcessError):
            pipeline.run_command([pipeline.sys.executable, "-c", "raise SystemExit(3)"])


if __name__ == "__main__":
    unittest.main()
