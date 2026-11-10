"""
Run every post-analysis of one multi-task cohort.

run_pipeline.py evaluates the networks (clustering, lesions, lesion
controls; hours per run). Everything here only reads those outputs, so it is
rerun for the whole feature whenever more runs finish or the post-processing
code changes. The stages are the documented commands in their dependency
order, each command in its own process:

  lesion_plot  lesion_plot.py for every run of the feature with a lesion
               pickle: with lesion.py's controls (multiple_tasks_norm/) and,
               unless --control-pool is all, with that pool's
               (multiple_tasks_norm_<pool>/). A run without the lesion pickle
               or the pool's controls pickle is left out and listed.
  row_sharing  sibling_row_sharing.py: the cohort's row-cluster sharing
               summary; with --untrained also the untrained controls' summary
  variance     synapse_variance_decomposition.py --skip-completed: about five
               minutes per run, so current results are reused unless --force
  vs_lesion    state_space_vs_lesion.py: every run with state-space structure
               and lesion_plot caches, whatever its feature (seconds)
  figures      paper_plot.py multiple_tasks lesion for the feature's cohort
               (and the lesion mode again for the other control pool); it
               clears paper_plot/ first as every mode run does, so it is not
               a default stage: rerun the other modes afterwards

--stages picks a subset, run in this order; --seed restricts the per-run
stages (lesion_plot, variance) to those seeds, while the cohort summaries
always cover the feature; --dry-run prints the commands instead of running
them. A failing command stops the pipeline. The state-space analyses that
evaluate networks have their own driver, run_state_space_pipeline.py; the
sibling-task geometry (sibling_delay_analysis.py) redraws its trials and is
never rerun from here. Run from the repository root; stdout and stderr go to
log/ as well.

    python multiple_task/run_pipeline_analysis.py --feature L21e4
    python multiple_task/run_pipeline_analysis.py --feature L21e4 --stages lesion_plot vs_lesion
    python multiple_task/run_pipeline_analysis.py --feature L21e4 --seed 303 692 921
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from run_logging import tee_output

STAGES = ("lesion_plot", "row_sharing", "variance", "vs_lesion", "figures")
DEFAULT_STAGES = tuple(stage for stage in STAGES if stage != "figures")
LESION_PLOT = "multiple_task/lesion_plot.py"
ROW_SHARING = "multiple_task/sibling_row_sharing.py"
VARIANCE = "multiple_task/synapse_variance_decomposition.py"
VS_LESION = "multiple_task/state_space_vs_lesion.py"
PAPER_PLOT = "figures/paper_plot.py"
PERF_DIR = Path("multiple_tasks_perf")
# lesion.py's own controls, always plotted, and the pool lesion_controls.py
# adds by default (run_pipeline.DEFAULT_CONTROL_POOL).
BASE_POOL = "all"
CONTROL_POOLS = ("responsive", "all")
DEFAULT_CONTROL_POOL = "responsive"
# paper_plot selects a cohort by its L2 strength (--multitask-l2 1e4), which
# only the plain L2 features name.
L2_FEATURE_PATTERN = re.compile(r"L2(1e\d+)$")


def run_name(seed, feature):
    """The run identifier the analysis scripts derive their paths from."""
    return f"everything_seed{seed}_{feature}+hidden300+batch128+angle"


def control_pools(control_pool):
    """The pools lesion_plot is run with: lesion.py's always, then
    `control_pool` when it is another one."""
    return (BASE_POOL,) if control_pool == BASE_POOL else (BASE_POOL, control_pool)


def lesioned_seeds(feature, root=Path(".")):
    """Seeds of the feature's runs with a lesion pickle, ascending."""
    pattern = re.compile(rf"everything_seed(\d+)_{re.escape(feature)}\+hidden300\+batch128\+angle$")
    seeds = []
    for directory in (Path(root) / PERF_DIR).glob("everything_seed*"):
        match = pattern.match(directory.name)
        if match and (directory / f"lesion_prune_results_{directory.name}.pkl").exists():
            seeds.append(int(match.group(1)))
    return sorted(seeds)


def lesion_plot_jobs(feature, control_pool, seeds=None, root=Path(".")):
    """(jobs, left_out): the (seed, pool) pairs lesion_plot can run for, in
    seed order with lesion.py's pool first, and the (seed, pool, reason)
    triples it cannot: a requested seed without a lesion pickle, or a run
    without the pool's controls pickle (lesion_controls.py)."""
    available = lesioned_seeds(feature, root)
    chosen = available if seeds is None else list(dict.fromkeys(seeds))
    jobs, left_out = [], []
    for seed in chosen:
        aname = run_name(seed, feature)
        for pool in control_pools(control_pool):
            if seed not in available:
                left_out.append((seed, pool, "no lesion pickle"))
            elif pool != BASE_POOL and not (
                    Path(root) / PERF_DIR / aname / f"lesion_controls_{pool}_{aname}.pkl").exists():
                left_out.append((seed, pool, f"no {pool} controls pickle"))
            else:
                jobs.append((seed, pool))
    return jobs, left_out


def stage_commands(stage, args, root=Path(".")):
    """(commands, notes) of one stage: each command a subprocess argument
    list, each note a line about something the stage leaves out."""
    python = [sys.executable, "-u"]
    feature = ["--feature", args.feature]
    if stage == "lesion_plot":
        jobs, left_out = lesion_plot_jobs(args.feature, args.control_pool, args.seed, root)
        commands = [python + [LESION_PLOT, "--seed", str(seed), *feature, "--control-pool", pool]
                    for seed, pool in jobs]
        return commands, [f"lesion_plot left out seed {seed}, {pool} pool: {reason}"
                          for seed, pool, reason in left_out]
    if stage == "row_sharing":
        commands = [python + [ROW_SHARING, *feature]]
        if args.untrained:
            commands.append(python + [ROW_SHARING, *feature, "--untrained"])
        return commands, []
    if stage == "variance":
        seeds = ["all"] if args.seed is None else [str(seed) for seed in args.seed]
        skip = [] if args.force else ["--skip-completed"]
        return [python + [VARIANCE, "--seed", *seeds, *feature, *skip]], []
    if stage == "vs_lesion":
        return [python + [VS_LESION]], []
    if stage == "figures":
        strength = ["--multitask-l2", L2_FEATURE_PATTERN.fullmatch(args.feature).group(1)]
        commands = [python + [PAPER_PLOT, "multiple_tasks", "lesion", *strength]]
        if args.control_pool != BASE_POOL:
            commands.append(python + [PAPER_PLOT, "lesion", *strength,
                                      "--control-pool", args.control_pool])
        return commands, []
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the post-analyses of one multi-task cohort (see the module docstring).")
    parser.add_argument("--feature", required=True,
                        help="feature tag of the cohort (e.g. L21e4)")
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=list(DEFAULT_STAGES), metavar="STAGE",
        help=f"stages to run, always in the order {', '.join(STAGES)} "
             f"(default: {', '.join(DEFAULT_STAGES)})")
    parser.add_argument(
        "--seed", type=int, nargs="+", default=None, metavar="SEED",
        help="restrict the per-run stages (lesion_plot, variance) to these seeds "
             "(default: every run of the feature)")
    parser.add_argument(
        "--control-pool", choices=CONTROL_POOLS, default=DEFAULT_CONTROL_POOL,
        help="random-control pool lesion_plot reads in addition to lesion.py's "
             f"(default: {DEFAULT_CONTROL_POOL}; all plots lesion.py's only)")
    parser.add_argument(
        "--untrained", action="store_true",
        help="row_sharing also summarizes the untrained controls")
    parser.add_argument(
        "--force", action="store_true",
        help="variance recomputes every selected run instead of reusing current results")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands and run nothing")
    args = parser.parse_args(argv)
    args.stages = [stage for stage in STAGES if stage in args.stages]
    if "figures" in args.stages and L2_FEATURE_PATTERN.fullmatch(args.feature) is None:
        parser.error(f"the figures stage needs a plain L2 feature such as L21e4, not {args.feature}")
    return args


def main(args):
    start = time.time()
    notes = []
    for index, stage in enumerate(args.stages, 1):
        print(f"\n--- Stage {index}/{len(args.stages)}: {stage} ---", flush=True)
        stage_start = time.time()
        commands, stage_notes = stage_commands(stage, args)
        for note in stage_notes:
            print(f"    {note}")
        notes.extend(stage_notes)
        for command in commands:
            if args.dry_run:
                print(f"$ {' '.join(command)}")
            else:
                run_command(command)
        if not args.dry_run:
            print(f"    done ({(time.time() - stage_start) / 60:.1f} min)", flush=True)
    print(f"\nPost-analysis pipeline {'listed' if args.dry_run else 'complete'} for "
          f"{args.feature}: {len(args.stages)} stage(s) in {(time.time() - start) / 60:.1f} min")
    if notes:
        print("Left out:")
        for note in notes:
            print(f"  {note}")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.dry_run:
        main(arguments)
    else:
        with tee_output("run_pipeline_analysis"):
            main(arguments)
