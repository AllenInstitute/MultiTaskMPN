"""
Run every state-space analysis end to end.

The stages are the commands of docs/state_space.md in their order, each in
its own process (the shift stages hold one network's full modulation trace,
about 30 GiB, which only a process exit returns in full):

  shift       state_space_shift.py --steps N: the context-end PCA caches,
              the distance-angle results and the N-step timecourse of the
              whole cohort; clears the top-level files of state_space/ first
  untrained   the same for untrained networks with the example run's config,
              one per --init-seed, into state_space/untrained/
  structure   state_space_structure.py: category structure and task-vector
              additivity of the saved centroids, trained and untrained
  vs_lesion   state_space_vs_lesion.py: context-end task distance vs
              lesion-profile similarity, for the runs with lesion caches
  context     shift and untrained again at every --context-ms length, from
              the short trace (--skip-pca), into state_space/context<N>ms/
  transplant  state_space_transplant.py, the cohort and then the untrained
              networks, into state_space/transplant/
  figures     paper_plot.py state_space, which clears paper_plot/ first as
              every mode run does; rerun the other modes afterwards

--stages picks a subset, run in this order, to resume after an interruption;
a failing command stops the pipeline. --dry-run prints the commands instead
of running them. --fresh moves an existing state_space/ aside
(state_space_old_<timestamp>/, nothing is deleted), since the untrained,
context and transplant pickles merge by network name and would otherwise
keep entries of networks no longer in the cohort. The transplant run on
another draw of the evaluation trials (--evaluation-seed) stays a manual
command. Run from the repository root; stdout and stderr go to log/ as well.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from run_logging import tee_output
from state_space_evaluation import STATE_SPACE_DIR, UNTRAINED_INIT_SEED

STAGES = ("shift", "untrained", "structure", "vs_lesion", "context", "transplant", "figures")
SHIFT = "multiple_task/state_space_shift.py"
STRUCTURE = "multiple_task/state_space_structure.py"
VS_LESION = "multiple_task/state_space_vs_lesion.py"
TRANSPLANT = "multiple_task/state_space_transplant.py"
PAPER_PLOT = "figures/paper_plot.py"
# The run whose config the untrained networks take (docs/state_space.md); the
# untrained weights depend only on the setting and the init seed, so any run
# of the setting gives the same networks.
EXAMPLE_CHECKPOINT = "multiple_tasks/savednet_everything_seed202_L21e4+hidden300+batch128+angle.pt"
DEFAULT_INIT_SEEDS = tuple(range(UNTRAINED_INIT_SEED, UNTRAINED_INIT_SEED + 8))
DEFAULT_CONTEXT_MS = (240.0, 960.0)
DEFAULT_STEPS = 5


def stage_commands(stage, args):
    """The commands of one stage, each a subprocess argument list."""
    python = [sys.executable, "-u"]
    steps = ["--steps", str(args.steps)]
    untrained = ["--untrained", "--checkpoint", args.checkpoint,
                 "--init-seed", *(str(seed) for seed in args.init_seed)]
    if stage == "shift":
        return [python + [SHIFT, *steps]]
    if stage == "untrained":
        return [python + [SHIFT, *steps, *untrained]]
    if stage == "structure":
        return [python + [STRUCTURE]]
    if stage == "vs_lesion":
        return [python + [VS_LESION]]
    if stage == "context":
        commands = []
        for length in args.context_ms:
            context = [*steps, "--skip-pca", "--context-ms", f"{length:g}"]
            commands.append(python + [SHIFT, *context])
            commands.append(python + [SHIFT, *context, *untrained])
        return commands
    if stage == "transplant":
        return [python + [TRANSPLANT], python + [TRANSPLANT, *untrained]]
    if stage == "figures":
        return [python + [PAPER_PLOT, "state_space"]]
    raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}.")


def run_command(command):
    """Run one command, echoing its output line by line (through the log
    tee), and raise CalledProcessError when it fails."""
    print(f"$ {' '.join(command)}", flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def move_aside(state_space_dir):
    """Rename an existing state-space directory to <name>_old_<timestamp>;
    returns the new path, or None when there was nothing to move."""
    if not state_space_dir.exists():
        return None
    target = state_space_dir.with_name(
        f"{state_space_dir.name}_old_{datetime.now():%Y%m%d_%H%M%S}")
    state_space_dir.rename(target)
    return target


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the state-space analyses end to end (see the module docstring).")
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=list(STAGES), metavar="STAGE",
        help=f"stages to run, always in the order {', '.join(STAGES)} (default: all)")
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, metavar="N",
        help=f"stimulus steps of the distance-angle timecourse (default: {DEFAULT_STEPS})")
    parser.add_argument(
        "--context-ms", type=float, nargs="*", default=list(DEFAULT_CONTEXT_MS), metavar="MS",
        help="Context lengths of the context stage besides the default one "
             f"(default: {' '.join(f'{ms:g}' for ms in DEFAULT_CONTEXT_MS)}; none skips the stage)")
    parser.add_argument(
        "--checkpoint", default=EXAMPLE_CHECKPOINT, metavar="PATH",
        help="run whose config the untrained networks take (default: the example run)")
    parser.add_argument(
        "--init-seed", type=int, nargs="+", default=list(DEFAULT_INIT_SEEDS), metavar="SEED",
        help="seeds of the untrained networks, one network per seed "
             f"(default: {' '.join(map(str, DEFAULT_INIT_SEEDS))})")
    parser.add_argument(
        "--fresh", action="store_true",
        help=f"move an existing {STATE_SPACE_DIR}/ to {STATE_SPACE_DIR}_old_<timestamp>/ "
             "first (needs the shift stage)")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands and run nothing")
    args = parser.parse_args(argv)
    args.stages = [stage for stage in STAGES if stage in args.stages]
    if args.steps < 1:
        parser.error("--steps must be positive")
    if any(not length > 0 for length in args.context_ms):
        parser.error("--context-ms lengths must be positive")
    if args.fresh and "shift" not in args.stages:
        parser.error("--fresh needs the shift stage, which rebuilds the directory")
    return args


def main(args):
    needs_checkpoint = {"untrained", "context", "transplant"} & set(args.stages)
    if needs_checkpoint and not args.dry_run and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint} not found; it is needed by "
                                f"{', '.join(sorted(needs_checkpoint))}.")
    if args.fresh and not args.dry_run:
        moved = move_aside(STATE_SPACE_DIR)
        if moved is not None:
            print(f"Moved {STATE_SPACE_DIR}/ to {moved}/")
    start = time.time()
    for index, stage in enumerate(args.stages, 1):
        print(f"\n--- Stage {index}/{len(args.stages)}: {stage} ---", flush=True)
        stage_start = time.time()
        for command in stage_commands(stage, args):
            if args.dry_run:
                print(f"$ {' '.join(command)}")
            else:
                run_command(command)
        if not args.dry_run:
            print(f"    done ({(time.time() - stage_start) / 60:.1f} min)", flush=True)
    print(f"\nState-space pipeline {'listed' if args.dry_run else 'complete'}: "
          f"{len(args.stages)} stage(s) in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.dry_run:
        main(arguments)
    else:
        with tee_output("run_state_space_pipeline"):
            main(arguments)
