"""
Run the single-task pipeline end to end:
  1. one_task.py           — train the single-task MPN(s) (writes ./onetask/)
  2. one_task_analysis.py  — post-training analysis of every run in ./onetask/

Each step runs in its own subprocess so the heavy training / CUDA state is
fully released between stages (mirrors how the scripts are meant to be run from
the command line). Training configuration (ruleset, seeds, hidden size, ...)
lives inside one_task.py itself; this pipeline just chains the two scripts.

By default the analysis step processes exactly the runs that the training step
just produced (read from the manifest one_task.py writes to
./onetask/last_run_anames.txt), not every run on disk. Pass --all to analyze
every run instead.

Usage:
    python run_one_task_pipeline.py                 # train, then analyze just-trained runs
    python run_one_task_pipeline.py --all           # ... analyze ALL runs on disk
    python run_one_task_pipeline.py --aname <name>  # analyze only this run
    python run_one_task_pipeline.py --skip-train     # analysis only (uses manifest)
    python run_one_task_pipeline.py --skip-analysis  # training only
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent      # one_task/ (holds the scripts)
ROOT = HERE.parent                          # repo root (holds data dirs)
TRAIN_SCRIPT = HERE / "one_task.py"
ANALYSIS_SCRIPT = HERE / "one_task_analysis.py"
MANIFEST_PATH = ROOT / "onetask" / "last_run_anames.txt"


def _read_manifest():
    """Run identifiers produced by the most recent one_task.py invocation."""
    if not MANIFEST_PATH.exists():
        return []
    return [ln.strip() for ln in MANIFEST_PATH.read_text().splitlines() if ln.strip()]


def _run(script, extra_args=None):
    """Run a script with the current interpreter; raise on non-zero exit."""
    cmd = [sys.executable, str(script), *(extra_args or [])]
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))  # data paths are relative to repo root
    dt = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(f"  FAILED ({script.name}, exit {result.returncode}, {dt:.1f}s)")
    print(f"    done ({script.name}, {dt:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Single-task train + analyze pipeline.")
    parser.add_argument("--aname", type=str, default=None,
                        help="Analyze only this run identifier (overrides the "
                             "manifest).")
    parser.add_argument("--all", action="store_true",
                        help="Analyze every run on disk instead of only the "
                             "runs just trained (the manifest).")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; run analysis only.")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis; run training only.")
    args = parser.parse_args()

    print(f"{'='*60}\n  Single-task pipeline\n{'='*60}")
    t0 = time.time()

    if not args.skip_train:
        print("\n--- Step 1/2: one_task (train) ---")
        _run(TRAIN_SCRIPT)
    else:
        print("\n--- Step 1/2: one_task (train) — SKIPPED ---")

    if not args.skip_analysis:
        print("\n--- Step 2/2: one_task_analysis ---")
        if args.aname:
            # A single explicit run.
            _run(ANALYSIS_SCRIPT, ["--aname", args.aname])
        elif args.all:
            # Every run on disk (one_task_analysis.py default with no --aname).
            _run(ANALYSIS_SCRIPT, [])
        else:
            # Only the runs just trained, one analysis call each.
            anames = _read_manifest()
            if not anames:
                print("  No manifest found; falling back to analyzing ALL runs. "
                      "(Run training first, or pass --all explicitly.)")
                _run(ANALYSIS_SCRIPT, [])
            else:
                print(f"  Analyzing {len(anames)} just-trained run(s): {anames}")
                for a in anames:
                    _run(ANALYSIS_SCRIPT, ["--aname", a])
    else:
        print("\n--- Step 2/2: one_task_analysis — SKIPPED ---")

    print(f"\n  Pipeline complete ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
