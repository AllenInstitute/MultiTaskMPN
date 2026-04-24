"""
Run the full analysis pipeline for each experiment:
  1. multiple_task_analysis  — clustering & post-training analysis
  2. leison                  — lesion & pruning experiments
  3. leison_plot             — normalized lesion effect plots
"""
import re
import time
from pathlib import Path

import multiple_task_analysis
import leison
import leison_plot


def run_pipeline(seed, feature):
    aname = f"everything_seed{seed}_{feature}+hidden300+batch128+angle"
    print(f"\n{'='*60}")
    print(f"  Pipeline start: {aname}")
    print(f"{'='*60}")

    t0 = time.time()

    print(f"\n--- Step 1/3: multiple_task_analysis ---")
    t1 = time.time()
    multiple_task_analysis.main(seed, feature)
    print(f"    done ({time.time() - t1:.1f}s)")

    cluster_path = Path(f"./multiple_tasks/{aname}/cluster_info_{aname}.pkl")
    cluster_path_mod = Path(f"./multiple_tasks/{aname}/cluster_info_mod_{aname}.pkl")
    if not cluster_path.exists() or not cluster_path_mod.exists():
        print(f"    Skipping leison steps: cluster files not found")
        return

    print(f"\n--- Step 2/3: leison ---")
    t2 = time.time()
    leison.main(seed, feature)
    print(f"    done ({time.time() - t2:.1f}s)")

    print(f"\n--- Step 3/3: leison_plot ---")
    t3 = time.time()
    leison_plot.main(seed, feature)
    print(f"    done ({time.time() - t3:.1f}s)")

    print(f"\n  Pipeline complete: {aname} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    saved_nets = sorted(Path("multiple_tasks").glob("savednet_everything_seed*+angle.pt"))
    param_lst = []
    for p in saved_nets:
        m = re.match(
            r"savednet_everything_seed(\d+)_(\w+)\+hidden\d+\+batch\d+\+angle\.pt",
            p.name,
        )
        if m:
            param_lst.append((int(m.group(1)), m.group(2)))

    print(f"Found {len(param_lst)} saved models: {param_lst}")

    # Override to run specific experiments:
    # param_lst = [(749, "L21e4")]

    for seed, feature in param_lst:
        run_pipeline(seed, feature)
