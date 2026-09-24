#!/usr/bin/env python
"""Verify runner logging with mocks and tiny children, never real analyses."""

import io
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

from pretrain import run_pretraining_analysis_pipeline as pipeline


class PretrainingAnalysisPipelineTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log_dir = Path(directory.name)
        real_tee = pipeline.tee_output
        self.logger = self.enterContext(patch.object(
            pipeline, "tee_output",
            side_effect=lambda name, log_dir: real_tee(name, log_dir=self.log_dir)))

    def read_log(self):
        paths = list(self.log_dir.glob("pretraining_analysis_pipeline_*.log"))
        self.assertEqual(len(paths), 1)
        return paths[0].read_text(encoding="utf-8")

    def expected_calls(self):
        return [
            call([pipeline.sys.executable, "-u", str(pipeline.HERE / script),
                  "--feature", feature])
            for script, feature in (
                ("pretraining_analysis.py", "L21e3"),
                ("pretraining_analysis.py", "L21e3mb2"),
                ("pretraining_post.py", "L21e3"),
                ("pretraining_post.py", "L21e3mb2"),
            )
        ]

    def test_all_commands_run_in_order_from_repository_root(self):
        output = io.StringIO()
        with (
            patch.object(pipeline, "_run_command", return_value=0) as run,
            redirect_stdout(output),
        ):
            pipeline.main()
        self.assertEqual(run.call_args_list, self.expected_calls())
        self.assertEqual(pipeline.ROOT, Path(__file__).resolve().parents[1])
        self.assertIn("All four analyses completed", output.getvalue())
        self.assertEqual(self.read_log(), output.getvalue())
        self.logger.assert_called_once_with("pretraining_analysis_pipeline", log_dir=pipeline.ROOT / "log")

    def test_failure_stops_remaining_commands_and_propagates_exit_code(self):
        log_root = self.log_dir
        for failed_step in range(4):
            with self.subTest(failed_step=failed_step):
                self.log_dir = log_root / f"failure_{failed_step}"
                results = [0] * failed_step + [7]
                output = io.StringIO()
                with (
                    patch.object(pipeline, "_run_command", side_effect=results) as run,
                    redirect_stdout(output),
                    self.assertRaises(SystemExit) as raised,
                ):
                    pipeline.main()
                self.assertEqual(raised.exception.code, 7)
                self.assertEqual(run.call_args_list, self.expected_calls()[:failed_step + 1])
                self.assertIn("FAILED:", output.getvalue())
                self.assertNotIn("All four analyses completed", output.getvalue())
                self.assertIn(output.getvalue(), self.read_log())

    def test_child_stdout_stderr_and_working_directory_are_logged(self):
        output = io.StringIO()
        command = [pipeline.sys.executable, "-u", "-c",
                   ("import os, sys; print(os.getcwd()); print('child stdout'); "
                    "print('child stderr', file=sys.stderr); sys.exit(7)")]
        with redirect_stdout(output), pipeline.tee_output("pretraining_analysis_pipeline", log_dir=self.log_dir):
            self.assertEqual(pipeline._run_command(command), 7)
        self.assertIn(str(pipeline.ROOT), output.getvalue())
        self.assertIn("child stdout", output.getvalue())
        self.assertIn("child stderr", output.getvalue())
        self.assertEqual(self.read_log(), output.getvalue())

    def test_launch_error_is_logged_and_stops_pipeline(self):
        output = io.StringIO()
        with (
            patch.object(pipeline, "_run_command", side_effect=OSError("cannot launch")) as run,
            redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            pipeline.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(run.call_args_list, self.expected_calls()[:1])
        self.assertIn("cannot launch", self.read_log())
        self.assertEqual(self.read_log(), output.getvalue())

    def test_script_entrypoint(self):
        with (
            patch("run_logging.tee_output", self.logger),
            patch.object(pipeline.subprocess, "Popen") as popen,
            patch.dict("sys.modules", {"_bootstrap": pipeline._bootstrap}),
            redirect_stdout(io.StringIO()),
        ):
            child = popen.return_value.__enter__.return_value
            child.stdout = ["entrypoint child output\n"]
            child.wait.return_value = 0
            runpy.run_path(str(pipeline.HERE / "run_pretraining_analysis_pipeline.py"),
                           run_name="__main__")
        self.assertEqual([item.args[0] for item in popen.call_args_list],
                         [item.args[0] for item in self.expected_calls()])
        self.assertTrue(all(item.kwargs["cwd"] == pipeline.ROOT for item in popen.call_args_list))
        self.assertEqual(self.read_log().count("entrypoint child output"), 4)
        self.assertIn("All four analyses completed", self.read_log())


if __name__ == "__main__":
    unittest.main()