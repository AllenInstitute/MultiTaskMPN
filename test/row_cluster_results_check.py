"""Read-only audit of current row-sharing caches and paper PNGs.

Run from the repository root. Renders figures in memory for pixel comparison;
does not rerun clustering/permutations or replace any analysis/figure files.
Select the cohort used to generate the PNGs, e.g. --multitask-l2 1e4. Use the
same plotting environment as the PNG generation for exact pixel comparisons.
"""
import argparse
import io
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image
from scipy.stats import mannwhitneyu

import paper_plot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multitask-l2", choices=paper_plot.MULTITASK_L2_CHOICES)
    args = parser.parse_args(argv)
    if args.multitask_l2 is not None:
        paper_plot._set_multitask_l2(args.multitask_l2)
    feature = paper_plot.multitask_feature()
    cohorts = {}
    for tag, untrained in ((feature, False), ("untrained", True)):
        data = paper_plot._load_row_cluster_sharing(untrained=untrained)
        if data is None:
            raise SystemExit("Current trained and untrained results are required; see rerun hint.")
        print("CACHE", tag, "version", data["version"], "networks", len(data["experiments"]))
        cohorts[tag] = data
        for view in ("variance_partition_all", "variance_partition_matched_tasks"):
            partition = data.get(view)
            if not partition:
                continue
            if partition.get("schema_version") != 2:
                raise ValueError(
                    f"{tag}/{view}: stale schema; run "
                    + paper_plot._row_sharing_view_command(view, data, untrained))
            samples = paper_plot._variance_partition_samples(partition, data["experiments"])
            for rep, values in samples.items():
                stats = partition["representations"][rep]
                models = {key: np.asarray(values) for key, values in stats["model_ss"].items()}
                for key, other in (("task", "period"), ("period", "task")):
                    np.testing.assert_allclose(stats["ss"][key],
                                               (models[key] + models["joint"] - models[other]) / 2)
                np.testing.assert_allclose(stats["ss"]["residual"],
                                           np.asarray(stats["ss"]["total"]) - models["joint"])
                print("VARIANCE_PARTITION", tag, view, rep, "valid n", len(values),
                      "task/period/residual mean", values.mean(axis=0).tolist() if len(values) else [])
        for mode, by in (("same_task_all", "tasks"), ("same_period_all", "periods"),
                         ("same_task_matched_tasks", "tasks"),
                         ("same_period_matched_tasks", "periods")):
            if mode not in data:
                continue
            entry = data[mode]
            print("METHOD", tag, mode, entry["permutation"])
            for key, reps in [("ALL", entry["pooled"]), *entry[by].items()]:
                counts = [[(name, run["n_cells"]) for name, run in stats["per_run"].items()]
                          for stats in reps.values()]
                assert all(value == counts[0] for value in counts)
                cells = []
                for rep, stats in reps.items():
                    shared, chance = paper_plot._row_sharing_run_values(stats)
                    np.testing.assert_allclose(shared.mean(), stats["pct_same_cluster"])
                    np.testing.assert_allclose(chance.mean(), 100 * stats["chance_mean"])
                    assert 0 <= stats["p_perm"] <= stats["q_fdr"] + 1e-12 <= 1 + 1e-12
                    cells.append(f"{rep}: {shared.mean():.3f}% / null {chance.mean():.3f}% "
                                 f"p={stats['p_perm']:.6g} q={stats['q_fdr']:.6g}")
                print("BAR", tag, mode, key, "; ".join(cells))
            for rep in ("input", "hidden", "modulation"):
                above, below = [], []
                for key, reps in entry[by].items():
                    stats = reps[rep]
                    if stats["q_fdr"] < 0.05:
                        target = (above if stats["pct_same_cluster"] > 100 * stats["chance_mean"]
                                  else below)
                        target.append(key)
                print("SIGNIFICANT", tag, mode, rep, "above", above, "below", below)

        for cut_key in ("cut_robustness_all", "cut_robustness_matched_tasks"):
            cut_entry = data.get(cut_key)
            if cut_entry is None:
                continue
            classes = ("task", "period", "neither", "any")
            for k in cut_entry["k_values"]:
                cut = cut_entry["cuts"][k]
                assert cut["anames"] == data["experiments"]
                for i, name in enumerate(cut["anames"]):
                    counts = cut["counts"][name]
                    # The matched cut keeps the fixed 25 rows; the all-task cut
                    # keeps every row the organization check counted.
                    expected_pairs = ([50, 50, 200, 300] if cut_key.endswith("_matched_tasks") else [
                        data["organization"]["representations"]["modulation"]["counts"][name][c]["n_pairs"]
                        for c in classes])
                    assert [counts[c]["n_pairs"] for c in classes] == expected_pairs
                    baseline = counts["any"]["n_shared"] / counts["any"]["n_pairs"]
                    for pair_class in classes[:3]:
                        expected = (counts[pair_class]["n_shared"] / counts[pair_class]["n_pairs"]
                                    / baseline) if baseline else np.nan
                        np.testing.assert_allclose(cut["enrichment"][pair_class][i], expected)
                delta = np.subtract(cut["enrichment"]["task"], cut["enrichment"]["period"])
                np.testing.assert_allclose(cut["task_bias"], delta)
                print("CUT", tag, cut_key, k, "actual", sorted(set(cut["actual_k"])),
                      "delta median", float(np.nanmedian(delta)), "positive", int(sum(delta > 0)),
                      "finite", int(sum(np.isfinite(delta))), "period median",
                      float(np.nanmedian(cut["enrichment"]["period"])))

    for mode in ("organization", "organization_matched_tasks"):
        if not all(mode in data for data in cohorts.values()):
            continue
        for rep in ("input", "hidden", "modulation"):
            entries = [cohorts[tag][mode]["representations"][rep] for tag in (feature, "untrained")]
            for pair_class in ("period", "task"):
                values = [paper_plot._organization_values(
                    entry, pair_class, include_zero=mode.endswith("_matched_tasks")) for entry in entries]
                p_value = mannwhitneyu(*values, alternative="two-sided").pvalue if all(v.size for v in values) else np.nan
                print("ORGANIZATION", mode, rep, pair_class,
                      "trained/untrained medians", [float(np.median(v)) if v.size else np.nan for v in values],
                      "n", [len(v) for v in values], "raw Mann-Whitney p", float(p_value))

    def compare_figure(fig, out_path, extra=""):
        path = Path(out_path)
        if paper_plot.SHOW_LEGEND:
            path = path.with_name(f"{path.stem}_n{path.suffix}")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        with Image.open(buffer) as generated, Image.open(path) as saved:
            new, old = np.asarray(generated.convert("RGBA")), np.asarray(saved.convert("RGBA"))
            identical = new.shape == old.shape and np.array_equal(new, old)
        print("PNG_MATCH", path, identical, "shape", old.shape, extra)
        paper_plot.plt.close(fig)
        assert identical, f"{path} does not match the current cache and plotting code"

    print("PAPER_SELECTION", paper_plot.multitask_feature(), "legend", paper_plot.SHOW_LEGEND)
    with patch.object(paper_plot, "_save_fig", side_effect=compare_figure), \
            patch.object(paper_plot, "_ensure_out_dir"):
        figures = {"organization": "organization", "same_task_seeds": "same_task_all",
                   "same_period_seeds": "same_period_all",
                   "same_period_synapses": "same_period_all",
                   "same_task_matched_tasks_seeds": "same_task_matched_tasks",
                   "same_period_matched_tasks_seeds": "same_period_matched_tasks",
                   "organization_matched_tasks": "organization_matched_tasks",
                   "counts_seeds": "cluster_counts",
                   "cut_robustness": "cut_robustness_all",
                   "cut_robustness_matched_tasks": "cut_robustness_matched_tasks"}
        for name, entry_key in figures.items():
            if entry_key in cohorts[feature]:
                getattr(paper_plot, f"plot_row_cluster_{name}")()
            else:
                print("PNG_SKIP", name, "missing trained summary", entry_key)
        for view in ("variance_partition_all", "variance_partition_matched_tasks"):
            if view in cohorts[feature]:
                getattr(paper_plot, f"plot_{view}")()


if __name__ == "__main__":
    main()
