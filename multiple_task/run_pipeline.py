"""
Run the full analysis pipeline for each experiment:
  1. multiple_task_analysis  — stage 2 clustering always runs (produces the
     cluster_info pickles steps 2-3 need); stage 1 sibling-task fixed points
     (dmcgo/delaydm families) are skipped by default, opt in with --families
  2. leison                  — lesion & pruning experiments
  3. leison_plot             — normalized lesion effect plots
"""
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from run_logging import tee_output
import multiple_task_analysis
import leison
import leison_plot


def run_pipeline(seed, feature, families=()):
    aname = f"everything_seed{seed}_{feature}+hidden300+batch128+angle"
    print(f"\n{'='*60}")
    print(f"  Pipeline start: {aname}")
    print(f"{'='*60}")

    t0 = time.time()

    print("\n--- Step 1/3: multiple_task_analysis ---")
    t1 = time.time()
    if not families:
        print("    Sibling-family (shared_run) analysis skipped; running clustering only")
    # families=() skips shared_run but stage 2 (clustering) still runs and
    # writes the cluster_info pickles steps 2-3 consume.
    multiple_task_analysis.main(seed, feature, clean=False, families=families)
    print(f"    done ({time.time() - t1:.1f}s)")

    cluster_path = Path(f"./multiple_tasks_analysis/{aname}/cluster_info_{aname}.pkl")
    cluster_path_mod = Path(f"./multiple_tasks_analysis/{aname}/cluster_info_mod_{aname}.pkl")
    if not cluster_path.exists() or not cluster_path_mod.exists():
        print("    Skipping leison steps: cluster files not found")
        return

    print("\n--- Step 2/3: leison ---")
    t2 = time.time()
    leison.main(seed, feature)
    print(f"    done ({time.time() - t2:.1f}s)")

    print("\n--- Step 3/3: leison_plot ---")
    t3 = time.time()
    leison_plot.main(seed, feature)
    print(f"    done ({time.time() - t3:.1f}s)")

    print(f"\n  Pipeline complete: {aname} ({time.time() - t0:.1f}s total)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default=None,
                        help="Only run models with this feature (e.g. 'L21e4')")
    parser.add_argument("--seed", type=int, default=None,
                        help="Only run the model with this seed (e.g. 749). "
                             "Combine with --feature to disambiguate.")
    parser.add_argument("--families", nargs="+", default=[],
                        choices=list(multiple_task_analysis.SHARED_RUN_FAMILIES),
                        help="Sibling-task families to solve in step 1 "
                             "(dmcgo/delaydm fixed points). Default: none — "
                             "shared_run is skipped, clustering still runs.")
    args = parser.parse_args()

    saved_nets = sorted(Path("multiple_tasks").glob("savednet_everything_seed*+angle.pt"))
    param_lst = []
    for p in saved_nets:
        m = re.match(
            r"savednet_everything_seed(\d+)_(\w+)\+hidden\d+\+batch\d+\+angle\.pt",
            p.name,
        )
        if m:
            param_lst.append((int(m.group(1)), m.group(2)))

    if args.feature:
        param_lst = [(s, f) for s, f in param_lst if f == args.feature]
    if args.seed is not None:
        param_lst = [(s, f) for s, f in param_lst if s == args.seed]

    print(f"Running {len(param_lst)} models: {param_lst}")

    for seed, feature in param_lst:
        run_pipeline(seed, feature, families=tuple(args.families))


if __name__ == "__main__":
    with tee_output("run_pipeline"):
        main()
