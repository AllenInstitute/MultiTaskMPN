"""
Run the full analysis pipeline for each experiment:
  optional: sibling_delay_analysis — DelayDM/DMC fixed-point geometry, selected
            explicitly with --families
  1. multiple_task_analysis  — weight structure and clustering (produces the
     cluster_info pickles steps 2-3 need)
  2. lesion                  — lesion & pruning experiments
  3. lesion_plot             — normalized lesion effect plots

Run selection: every checkpoint in multiple_tasks/ that matches --feature and
--seed. With --skip-completed, runs whose three steps have all written their
last output file (pipeline_markers) are left out, so rerunning the command
after more training adds the pipeline outputs of the new checkpoints only. A
run missing any marker is redone from step 1: the lesion steps clear their own
output directories, so no partial state is reused.
"""
import argparse
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from run_logging import tee_output
import multiple_task_analysis
import sibling_delay_analysis
import lesion
import lesion_plot

CHECKPOINT_DIR = Path("multiple_tasks")
ANALYSIS_DIR = Path("multiple_tasks_analysis")
PERF_DIR = Path("multiple_tasks_perf")
NORM_DIR = Path("multiple_tasks_norm")
CHECKPOINT_PATTERN = re.compile(
    r"savednet_everything_seed(\d+)_([^+]+)\+hidden\d+\+batch\d+\+angle\.pt")


def run_name(seed, feature):
    """The run identifier the pipeline steps derive their paths from."""
    return f"everything_seed{seed}_{feature}+hidden300+batch128+angle"


def pipeline_markers(aname, root=Path(".")):
    """The last file each pipeline step writes for a run, keyed by step.

    multiple_task_analysis.main ends with cluster_info_mod (cluster_info
    comes earlier, and the lesion steps need both); lesion.main ends with
    lesion_prune_results; lesion_plot.main ends with the two
    lesion_interaction pickles.
    """
    root = Path(root)
    analysis_dir = root / ANALYSIS_DIR / aname
    return {
        "multiple_task_analysis": [analysis_dir / f"cluster_info_{aname}.pkl",
                                   analysis_dir / f"cluster_info_mod_{aname}.pkl"],
        "lesion": [root / PERF_DIR / aname / f"lesion_prune_results_{aname}.pkl"],
        "lesion_plot": [root / NORM_DIR / aname / f"lesion_interaction_{vtag}_{aname}.pkl"
                        for vtag in ("norm", "unnorm")],
    }


def is_completed(aname, root=Path(".")):
    """True when every pipeline step has written its last output for the run."""
    return all(path.exists()
               for paths in pipeline_markers(aname, root).values()
               for path in paths)


def discover_runs(checkpoint_dir=CHECKPOINT_DIR):
    """(seed, feature) of every saved multi-task checkpoint, by seed within feature."""
    runs = []
    for path in Path(checkpoint_dir).glob("savednet_everything_seed*+angle.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            runs.append((int(match.group(1)), match.group(2)))
    return sorted(runs, key=lambda run: (run[1], run[0]))


def select_runs(runs, feature=None, seeds=None, skip_completed=False, root=Path(".")):
    """Apply the command-line filters to the discovered (seed, feature) runs.

    Returns (selected, skipped, missing): the runs to process, in the order
    of --seed when given; the completed runs left out by skip_completed; and
    the requested seeds without a checkpoint.
    """
    if feature is not None:
        runs = [run for run in runs if run[1] == feature]
    missing = []
    if seeds is not None:
        requested = list(dict.fromkeys(seeds))  # de-duplicate, keep order
        available = {seed for seed, _ in runs}
        missing = [seed for seed in requested if seed not in available]
        # Run in the order the seeds were given on the command line.
        runs = [run for seed in requested for run in runs if run[0] == seed]
    skipped = []
    if skip_completed:
        skipped = [run for run in runs if is_completed(run_name(*run), root)]
        runs = [run for run in runs if run not in skipped]
    return runs, skipped, missing


def run_pipeline(seed, feature, families=(), sibling_method=None):
    aname = run_name(seed, feature)
    print(f"\n{'='*60}")
    print(f"  Pipeline start: {aname}")
    print(f"{'='*60}")

    t0 = time.time()

    if families:
        print("\n--- Optional: sibling_delay_analysis ---")
        sibling_delay_analysis.run_sibling_analysis(
            seed, feature, families, sibling_method)

    print("\n--- Step 1/3: multiple_task_analysis ---")
    t1 = time.time()
    multiple_task_analysis.main(seed, feature, clean=False)
    print(f"    done ({time.time() - t1:.1f}s)")

    if not all(path.exists()
               for path in pipeline_markers(aname)["multiple_task_analysis"]):
        print("    Skipping lesion steps: cluster files not found")
        return

    print("\n--- Step 2/3: lesion ---")
    t2 = time.time()
    lesion.main(seed, feature)
    print(f"    done ({time.time() - t2:.1f}s)")

    print("\n--- Step 3/3: lesion_plot ---")
    t3 = time.time()
    lesion_plot.main(seed, feature)
    print(f"    done ({time.time() - t3:.1f}s)")

    print(f"\n  Pipeline complete: {aname} ({time.time() - t0:.1f}s total)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default=None,
                        help="Only run models with this feature (e.g. 'L21e4')")
    parser.add_argument("--seed", type=int, nargs="+", default=None,
                        help="Only run the model(s) with these seeds, in the "
                             "given order (e.g. --seed 299 365 487). Combine "
                             "with --feature to disambiguate.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Leave out runs whose three steps have all written "
                             "their last output, so only checkpoints without "
                             "pipeline outputs are processed. Requires --feature "
                             "unless --seed is given.")
    parser.add_argument("--families", nargs="+", default=[],
                        choices=list(sibling_delay_analysis.SIBLING_FAMILIES),
                        help="Optional sibling-task families to analyze before "
                             "clustering. Default: none. For sibling analysis "
                             "only, run sibling_delay_analysis.py directly.")
    parser.add_argument(
        "--sibling-method", choices=sibling_delay_analysis.SIBLING_METHODS,
        help=("Fixed-point method for --families. Required when sibling "
              "families are selected."))
    args = parser.parse_args(argv)
    if args.families and args.sibling_method is None:
        parser.error("--sibling-method is required when --families is used")
    if args.sibling_method is not None and not args.families:
        parser.error("--sibling-method requires --families")
    if args.skip_completed and args.feature is None and args.seed is None:
        parser.error("--skip-completed requires --feature (or --seed) so the "
                     "batch is explicit")
    return args


def main():
    args = parse_args()
    runs, skipped, missing = select_runs(
        discover_runs(), args.feature, args.seed, args.skip_completed)
    if missing:
        print(f"WARNING: no checkpoint for seed(s) {missing}"
              + (f" with feature {args.feature}" if args.feature else "")
              + "; skipping them.")
    if skipped:
        print(f"Skipping {len(skipped)} completed run(s): {skipped}")

    print(f"Running {len(runs)} models: {runs}")

    for seed, feature in runs:
        run_pipeline(seed, feature, families=tuple(args.families),
                     sibling_method=args.sibling_method)


if __name__ == "__main__":
    with tee_output("run_pipeline"):
        main()
