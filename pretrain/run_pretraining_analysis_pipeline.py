#!/usr/bin/env python
"""Run pretraining analysis and post-analysis for both modulation-bound variants.

Usage from the repository root:
    python pretrain/run_pretraining_analysis_pipeline.py

Runs pretraining_analysis.py for L21e3 then L21e3mb2, followed by
pretraining_post.py for L21e3 then L21e3mb2. Each command uses the current
Python interpreter and the repository root as its working directory.
Runs sequentially, stops on the first failure, and does not launch training.
The analysis scripts retain their defaults and overwrite matching outputs.
Runner and child stdout/stderr are streamed to the terminal and to
log/pretraining_analysis_pipeline_<timestamp>_<pid>.log on every invocation.
"""

import shlex
import subprocess
import sys
from pathlib import Path
from time import perf_counter

if __package__:
    from . import _bootstrap
else:
    import _bootstrap  # noqa: F401
from run_logging import tee_output

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEPS = (
    ("pretraining_analysis.py", "L21e3"),
    ("pretraining_analysis.py", "L21e3mb2"),
    ("pretraining_post.py", "L21e3"),
    ("pretraining_post.py", "L21e3mb2"),
)


def _run_command(command):
    """Stream merged child stdout/stderr through the parent's tee stream."""
    with subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()


def main():
    with tee_output("pretraining_analysis_pipeline", log_dir=ROOT / "log"):
        started = perf_counter()
        for step, (script, feature) in enumerate(STEPS, start=1):
            command = [sys.executable, "-u", str(HERE / script), "--feature", feature]
            print(f"\n[{step}/{len(STEPS)}] $ {shlex.join(command)}", flush=True)
            step_started = perf_counter()
            try:
                returncode = _run_command(command)
            except OSError as error:
                print(f"FAILED to launch {script} --feature {feature}: {error}", flush=True)
                raise SystemExit(1) from error
            elapsed = perf_counter() - step_started
            if returncode != 0:
                print(f"FAILED: {script} --feature {feature} "
                      f"(exit {returncode}, {elapsed:.1f}s)", flush=True)
                raise SystemExit(returncode)
            print(f"Completed step {step}/{len(STEPS)} ({elapsed:.1f}s)", flush=True)
        print(f"\nAll four analyses completed ({perf_counter() - started:.1f}s total).",
              flush=True)


if __name__ == "__main__":
    main()