"""
Untrained control of the row-cluster analyses: cluster fresh networks.

For one trained run's config (--seed, --feature) it builds an untrained
network per --init-seed (network_loading.build_network: the same
architecture, weights drawn after seeding NumPy and torch with the init seed,
none of the trained weights) and runs the complete
`multiple_task_analysis.main` clustering on it, exactly as for the trained
run. Each network's outputs go to
multiple_tasks_analysis/untrained/{aname}_init{k}/ (cluster_info_..._init{k}.pkl,
cluster_info_mod_..._init{k}.pkl and the figures), apart from the trained
results; no lesion step runs. A historical S1-conditioned run took about
80 min on a GPU node (init seed 0 of seed 749: 4782 s); timing for the new
trial-time estimator has not yet been measured. Rerunning a
killed batch with --skip-completed skips networks with both complete
trial_time cluster caches and redoes interrupted or older-estimator runs
from the start.

L2 acts only during training, so the networks depend on the config's
architecture (hidden width, modulation bounds and decay), not on its L2
strength: the five networks of seed 749's L21e4 config are, init seed for
init seed, the networks any L21e3, L21e2 or L21e5 config would give, and
`sibling_row_sharing.py --untrained` pools them as the control of every L2
cohort of that architecture (core/multitask_naming.py). Cluster a new set
only for another architecture, such as the lambda = 0.9 cohort.

Usage:
    python multiple_task/untrained_clustering.py --feature L21e4 --seed 749 \\
        --init-seed 0 1 2 3 4
    # leave out the networks already clustered
    python multiple_task/untrained_clustering.py --feature L21e4 --seed 749 \\
        --init-seed 0 1 2 3 4 --skip-completed
    # then the row-cluster organization of the untrained networks
    python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained
"""
import argparse
import time

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from run_logging import tee_output
import multiple_task_analysis
from core.task_variance import cluster_caches_current


def run_name(seed, feature):
    """The trained run whose config the untrained networks take."""
    return f"everything_seed{seed}_{feature}+hidden300+batch128+angle"


def completion_markers(seed, feature, init_seed):
    """The two cluster pickles the clustering of one untrained network writes
    last (cluster_info, then cluster_info_mod)."""
    name, save_dir = multiple_task_analysis.output_names(
        seed, "everything", f"{feature}+hidden300+batch128+angle", init_seed)
    return [save_dir / f"cluster_info_{name}.pkl", save_dir / f"cluster_info_mod_{name}.pkl"]


def is_completed(seed, feature, init_seed):
    """Both full cluster pickles must use current trial_time variance."""
    return cluster_caches_current(completion_markers(seed, feature, init_seed))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cluster untrained networks of one run's config, the control "
                    "of the row-cluster analyses.")
    parser.add_argument("--feature", required=True,
                        help="Feature of the run whose config is used (e.g. 'L21e4').")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed of the run whose config is used (e.g. 749).")
    parser.add_argument("--init-seed", type=int, nargs="+", required=True,
                        help="Init seeds of the untrained networks, one network each.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Leave out networks with both complete, current trial_time cluster pickles.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    init_seeds = list(dict.fromkeys(args.init_seed))
    if args.skip_completed:
        done = [k for k in init_seeds if is_completed(args.seed, args.feature, k)]
        if done:
            print(f"Skipping {len(done)} clustered network(s): init seeds {done}")
        init_seeds = [k for k in init_seeds if k not in done]
    print(f"Untrained networks of {run_name(args.seed, args.feature)}: "
          f"init seeds {init_seeds}")
    for init_seed in init_seeds:
        print(f"\n{'=' * 60}\n  Untrained network, init seed {init_seed}\n{'=' * 60}")
        start = time.time()
        multiple_task_analysis.main(args.seed, args.feature, clean=True,
                                    untrained_init_seed=init_seed)
        print(f"    done ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    with tee_output("untrained_clustering"):
        main()
