"""run_state_space_pipeline.py: the documented commands in order, stage
selection, dry runs, the failure stop and the --fresh move."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

import run_state_space_pipeline as pipeline

PYTHON = [pipeline.sys.executable, "-u"]
UNTRAINED = ["--untrained", "--checkpoint", pipeline.EXAMPLE_CHECKPOINT,
             "--init-seed", "0", "1", "2", "3", "4", "5", "6", "7"]


def _commands(argv=()):
    args = pipeline.parse_args(list(argv))
    return [command for stage in args.stages for command in pipeline.stage_commands(stage, args)]


class StageCommandTests(unittest.TestCase):
    def test_default_run_is_the_documented_sequence(self):
        self.assertEqual(_commands(), [
            PYTHON + [pipeline.SHIFT, "--steps", "5"],
            PYTHON + [pipeline.SHIFT, "--steps", "5", *UNTRAINED],
            PYTHON + [pipeline.STRUCTURE],
            PYTHON + [pipeline.VS_LESION],
            PYTHON + [pipeline.SHIFT, "--steps", "5", "--skip-pca", "--context-ms", "240"],
            PYTHON + [pipeline.SHIFT, "--steps", "5", "--skip-pca", "--context-ms", "240", *UNTRAINED],
            PYTHON + [pipeline.SHIFT, "--steps", "5", "--skip-pca", "--context-ms", "960"],
            PYTHON + [pipeline.SHIFT, "--steps", "5", "--skip-pca", "--context-ms", "960", *UNTRAINED],
            PYTHON + [pipeline.TRANSPLANT],
            PYTHON + [pipeline.TRANSPLANT, *UNTRAINED],
            PYTHON + [pipeline.PAPER_PLOT, "state_space"],
        ])

    def test_options_reach_the_commands_and_stages_keep_their_order(self):
        commands = _commands(["--stages", "figures", "untrained", "--steps", "3",
                              "--checkpoint", "a.pt", "--init-seed", "4"])
        self.assertEqual(commands, [
            PYTHON + [pipeline.SHIFT, "--steps", "3", "--untrained", "--checkpoint", "a.pt",
                      "--init-seed", "4"],
            PYTHON + [pipeline.PAPER_PLOT, "state_space"],
        ])
        self.assertEqual(_commands(["--stages", "context", "--context-ms"]), [])
        with patch("sys.stderr"):
            for argv in (["--steps", "0"], ["--context-ms", "0"], ["--stages", "pca"],
                         ["--fresh", "--stages", "structure"]):
                with self.assertRaises(SystemExit):
                    pipeline.parse_args(argv)


class RunTests(unittest.TestCase):
    def setUp(self):
        # Never run against the real state_space/: every test works in a temp dir.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        previous = os.getcwd()
        os.chdir(directory.name)
        self.addCleanup(os.chdir, previous)

    def test_dry_run_prints_every_command_and_runs_nothing(self):
        with patch.object(pipeline, "run_command") as run, \
                patch.object(pipeline, "move_aside") as move, \
                patch("builtins.print") as output:
            pipeline.main(pipeline.parse_args(["--dry-run", "--fresh"]))
        run.assert_not_called()
        move.assert_not_called()
        printed = [call.args[0] for call in output.call_args_list]
        self.assertEqual(sum(line.startswith("$ ") for line in printed), 11)
        self.assertEqual(sum("--- Stage" in line for line in printed), 7)

    def test_a_failing_command_stops_the_pipeline(self):
        calls = []

        def fail_second(command):
            calls.append(command)
            if len(calls) == 2:
                raise subprocess.CalledProcessError(1, command)

        with patch.object(pipeline, "run_command", side_effect=fail_second), \
                patch("builtins.print"), self.assertRaises(subprocess.CalledProcessError):
            pipeline.main(pipeline.parse_args(["--stages", "structure", "vs_lesion", "figures"]))
        self.assertEqual(calls, [PYTHON + [pipeline.STRUCTURE], PYTHON + [pipeline.VS_LESION]])

    def test_untrained_stages_need_the_checkpoint_before_anything_runs(self):
        with patch.object(pipeline, "run_command") as run, patch("builtins.print"):
            with self.assertRaises(FileNotFoundError):
                pipeline.main(pipeline.parse_args(["--stages", "shift", "transplant",
                                                   "--checkpoint", "missing.pt"]))
        run.assert_not_called()

    def test_fresh_moves_the_directory_aside_and_keeps_its_files(self):
        with tempfile.TemporaryDirectory() as directory:
            state_space = Path(directory) / "state_space"
            (state_space / "untrained").mkdir(parents=True)
            (state_space / "untrained" / "keep.pkl").write_bytes(b"x")
            with patch.object(pipeline, "STATE_SPACE_DIR", state_space), \
                    patch.object(pipeline, "run_command"), patch("builtins.print") as output:
                pipeline.main(pipeline.parse_args(["--fresh", "--stages", "shift"]))
            self.assertFalse(state_space.exists())
            moved = [path for path in Path(directory).iterdir()
                     if path.name.startswith("state_space_old_")]
            self.assertEqual(len(moved), 1)
            self.assertEqual((moved[0] / "untrained" / "keep.pkl").read_bytes(), b"x")
            self.assertTrue(any("Moved" in call.args[0] for call in output.call_args_list))
            # Nothing to move: no message, no error.
            with patch.object(pipeline, "STATE_SPACE_DIR", state_space), \
                    patch.object(pipeline, "run_command"), patch("builtins.print") as output:
                pipeline.main(pipeline.parse_args(["--fresh", "--stages", "shift"]))
            self.assertFalse(any("Moved" in call.args[0] for call in output.call_args_list))

    def test_run_command_streams_output_and_raises_on_failure(self):
        with patch("builtins.print") as output:
            pipeline.run_command([pipeline.sys.executable, "-c", "print('hello')"])
        self.assertIn("hello\n", [call.args[0] for call in output.call_args_list])
        with patch("builtins.print"), self.assertRaises(subprocess.CalledProcessError):
            pipeline.run_command([pipeline.sys.executable, "-c", "raise SystemExit(3)"])


if __name__ == "__main__":
    unittest.main()
