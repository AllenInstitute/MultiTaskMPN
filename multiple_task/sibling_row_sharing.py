"""
Row-cluster sharing of sibling tasks: seven pairs, summarized by phase type.

`multiple_task_diagnosis.py` checks three sibling pairs. This script asks the
broader question the clustered heatmaps raise (`paper_plot.py`'s
`clustered_{input,hidden,modulation}_normalized`): do the rows of two sibling
tasks in one trial phase fall in the same row cluster, and does that depend
on the phase and on the representation? It checks seven pairs in two groups:

  Pro / anti  DelayPro vs DelayAnti (fdgo/fdanti), MemoryPro vs MemoryAnti
              (delaygo/delayanti), ReactPro vs ReactAnti (reactgo/reactanti),
              ReactMatch2Sample vs ReactNonMatch2Sample (dmsgo/dmsnogo),
              ReactCategoryPro vs ReactCategoryAnti (dmcgo/dmcnogo);
  Modality    ContextIntModality1 vs 2 (contextdelaydm1/2),
              IntegrationModality1 vs 2 (delaydm1/2).

Each pair, phase and representation is counted and tested against chance
exactly as in `multiple_task_diagnosis.py` (`diagnose_pair_same_cluster`),
whose loading, chance level and test this script reuses.

The legacy summary pools each group's pairs by phase type: Stimulus (stim1, stim2),
Memory (delay1, delay2) and Response (go1). Per run it counts the group's
(pair, phase) cells of that type whose two rows share a cluster and subtracts
the count chance predicts, the sum of the cells' chance levels; a two-sided
Wilcoxon signed-rank test over the runs asks whether that excess is zero.
Runs are independent, but the cells of one run come from one clustering, so
the per-cell tests are not pooled. A summary cell whose mean chance is at
least UNSPECIFIC_CHANCE (input and hidden in the memory periods, where every
task's row falls in one cluster) is reported but not tested: sharing there
is not specific to the pair. These exploratory diagnostics are separate from
the restricted permutation tests used by the paper's sharing bar figures.

The same-task and same-period checks use all available row pairs, identical
across representations. Their nulls permute cluster labels among tasks within
each phase (same task) or among available phases within each task (same period).
They preserve the row slots and block-specific cluster-label distributions.
The statistic is the equal mean of each network's sharing fraction; chance
is its permutation-null mean. Stars use BH-adjusted two-sided permutation p
values, corrected within a cohort and analysis type over all unique reported
tests (individual, all-pooled and sibling-group-pooled, across representations).
The paper's seeds figures show these means and q values with SD across networks.
These are conditional tests of row organization, not evidence of causality or
a direct test of trained versus untrained networks.

The organization check puts the two questions on one scale for every task's
rows (`summarize_organization`): per run and representation, the fold
enrichment of two rows of one period from different tasks, and of two rows
of one task from different periods, sharing a row cluster over any two rows
sharing one, the chance the cluster sizes give, so the representations,
with their different numbers of row clusters, compare. It is tested against
1 over the runs, and the runs' task-minus-period enrichment is compared
between representations (paired Wilcoxon).

Writes multiple_tasks_diagnosis/sibling_row_sharing_{tag}.{pkl,json}, the
per-pair figure sibling_row_sharing_pairs_{tag}.png (the layout of
multiple_task_diagnosis.py's figure, one row per pair), the summary figure
sibling_row_sharing_summary_{tag}.png and the same-task figure
sibling_row_sharing_same_task_{tag}.png, with the tag {feature}[_seed{seed}].
A run replaces its own tag's files only.

With --untrained it analyzes the untrained controls under
multiple_tasks_analysis/untrained/ (`untrained_clustering.py`) instead of the
trained runs. L2 acts only during training, so an untrained network built
from an L21e4 run's config is the same network as one built from an L21e3
run's config with the same init seed: one set of controls serves every L2
cohort of an architecture. The run therefore takes every untrained network
whose config shares --feature's architecture (the feature tag without its L2
strength, core/multitask_naming.py) and writes the result under the tag
``untrained_sharing_tag(feature)``: `untrained` for the default architecture
(L21e5 to L21e2), `lam0.9_untrained` for the lambda = 0.9 cohort. --seed is
then the config's run and every init seed of it enters (tag
`untrained_seed{seed}`).

Usage:
    python multiple_task/sibling_row_sharing.py --feature L21e4
    python multiple_task/sibling_row_sharing.py --feature L21e3
    # the untrained controls of every L2 cohort of the default architecture
    python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained
"""
import json
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are saved, never shown
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from multitask_naming import architecture_tag, untrained_sharing_tag
import multiple_task_diagnosis as diag

OUT_DIR = diag.OUT_DIR
# Where untrained_clustering.py writes the untrained controls' cluster info
# (multiple_task_analysis.UNTRAINED_ANALYSIS_DIR).
UNTRAINED_DIR = diag.MULTI_DIR / "untrained"

# Sibling pairs (raw rule names), grouped; the figures keep this order.
PAIR_GROUPS = {
    "Pro / anti": (("fdgo", "fdanti"), ("delaygo", "delayanti"),
                   ("reactgo", "reactanti"), ("dmsgo", "dmsnogo"),
                   ("dmcgo", "dmcnogo")),
    "Modality": (("contextdelaydm1", "contextdelaydm2"), ("delaydm1", "delaydm2")),
}

# Phase types of the summary and the trial phases each pools.
PHASE_TYPES = {
    "Stimulus": ("stim1", "stim2"),
    "Memory": ("delay1", "delay2"),
    "Response": ("go1",),
}

REPRESENTATIONS = tuple(diag._REP_COLORS)

# A summary cell at or above this mean chance is not tested: every task's row
# of those phases shares one cluster, so sharing says nothing about the pair.
UNSPECIFIC_CHANCE = 0.95
SIGNIFICANCE = 0.05

# Saved-result layout: 2 adds same-task, 3 all-task and organization,
# 4 same-period, 5 restricted permutations with common row pairs, equal
# network means and BH-adjusted q values. paper_plot requires version 5.
RESULT_VERSION = 5
# Prefixes of the files one run writes, each followed by "{tag}.{suffix}".
_OUTPUT_PREFIXES = ("sibling_row_sharing_", "sibling_row_sharing_pairs_",
                    "sibling_row_sharing_summary_", "sibling_row_sharing_same_task_")


def _clear_previous_results(tag):
    """Remove the earlier files of `tag` written by this script; other tags'
    files and multiple_task_diagnosis.py's files are kept. Returns the removed
    file names."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    removed = []
    for prefix in _OUTPUT_PREFIXES:
        for path in sorted(OUT_DIR.glob(f"{prefix}{tag}.*")):
            path.unlink()
            removed.append(path.name)
    return removed


def _pair_key(rule_a, rule_b):
    """Key of one pair's result in the saved payload."""
    return f"{rule_a}_vs_{rule_b}"


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze_pairs(experiments):
    """{group: {pair key: diagnose_pair_same_cluster result}} for every pair
    of PAIR_GROUPS. A pair whose rules no run clusters gets a result with no
    phases."""
    return {group: {_pair_key(a, b): diag.diagnose_pair_same_cluster(experiments, a, b)
                    for a, b in pairs}
            for group, pairs in PAIR_GROUPS.items()}


def _excess_test(excess):
    """Two-sided Wilcoxon signed-rank p of the per-run excess over chance
    being zero. Runs with no excess carry no sign and are dropped; with none
    left the p is 1."""
    excess = np.asarray(excess, dtype=float)
    nonzero = excess[np.abs(excess) > 1e-12]
    if nonzero.size == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided").pvalue)


def _summarize_cells(details):
    """Pool the per-run cells of one group, phase type and representation.

    `details` are the `detail` entries of the pooled cells (one per run and
    cell: aname, same, chance). Per run, `observed` is the number of cells
    whose two rows share a cluster and `expected` the sum of their chance
    levels; `excess` (observed - expected per run) is what the Wilcoxon test
    reads. Returns None without cells."""
    details = [entry for entry in details if np.isfinite(entry["chance"])]
    if not details:
        return None
    per_run = {}
    for entry in details:
        run = per_run.setdefault(entry["aname"], {"observed": 0, "expected": 0.0,
                                                  "n_cells": 0})
        run["observed"] += int(entry["same"])
        run["expected"] += float(entry["chance"])
        run["n_cells"] += 1
    excess = [run["observed"] - run["expected"] for run in per_run.values()]
    n_same = sum(run["observed"] for run in per_run.values())
    n_total = sum(run["n_cells"] for run in per_run.values())
    chance_mean = float(np.mean([entry["chance"] for entry in details]))
    unspecific = chance_mean >= UNSPECIFIC_CHANCE
    return {
        "pct_same_cluster": 100.0 * n_same / n_total,
        "n_same": n_same,
        "n_total": n_total,
        "n_runs": len(per_run),
        "chance_mean": chance_mean,
        "per_run": per_run,
        "excess": excess,
        "mean_excess": float(np.mean(excess)),
        "unspecific": unspecific,
        "p_excess": float("nan") if unspecific else _excess_test(excess),
    }


def summarize_by_phase_type(pair_results):
    """{group: {phase type: {rep: pooled stats}}} (`_summarize_cells`) over
    every pair of the group and every phase of the type the pair has. A phase
    type no pair of the group has is left out."""
    summary = {}
    for group, results in pair_results.items():
        summary[group] = {}
        for phase_type, phases in PHASE_TYPES.items():
            by_rep = {}
            for rep in REPRESENTATIONS:
                details = [entry
                           for result in results.values()
                           for phase in phases
                           if phase in result["per_phase"]
                           and rep in result["per_phase"][phase]
                           for entry in result["per_phase"][phase][rep]["detail"]]
                stats = _summarize_cells(details)
                if stats is not None:
                    by_rep[rep] = stats
            if by_rep:
                summary[group][phase_type] = by_rep
    return summary


def _split_row_names(tb_break_name):
    """(rules, phases) arrays of '{rule}-{phase}' row names."""
    parts = [str(name).rsplit("-", 1) for name in tb_break_name]
    return (np.array([part[0] for part in parts]),
            np.array([part[1] if len(part) == 2 else "" for part in parts]))


def _aligned_rows(aname, data):
    """Canonical rows shared by all representations; missing slots stay missing."""
    names, aligned = None, {}
    for rep in REPRESENTATIONS:
        if rep not in data:
            raise ValueError(f"{aname}: missing {rep} clustering")
        row_names = np.asarray(data[rep]["tb_break_name"], dtype=str)
        labels = np.asarray(data[rep]["row_labels"])
        if (row_names.ndim != 1 or labels.ndim != 1
                or row_names.size != labels.size
                or np.unique(row_names).size != row_names.size):
            raise ValueError(f"{aname}/{rep}: invalid or duplicate row names/labels")
        order = np.argsort(row_names)
        current = row_names[order]
        if names is None:
            names = current
        elif not np.array_equal(names, current):
            raise ValueError(f"{aname}: representations have different task/phase rows")
        aligned[rep] = labels[order]
    rules, phases = _split_row_names(names)
    if np.any(rules == "") or np.any(phases == ""):
        raise ValueError(f"{aname}: expected task-phase row names")
    return rules, phases, aligned


def _restricted_permutations(blocks, n_permutations, rng):
    """Independent uniform shuffles within blocks; one map for all representations."""
    indices = np.broadcast_to(np.arange(len(blocks)),
                              (n_permutations, len(blocks))).copy()
    for block in np.unique(blocks):
        rows = np.flatnonzero(blocks == block)
        indices[:, rows] = rng.permuted(
            np.broadcast_to(rows, (n_permutations, rows.size)), axis=1)
    return indices


def _permutation_p(observed, null):
    """Two-sided, inclusive equal-tail Monte Carlo p, with add-one correction."""
    null = np.asarray(null)
    tolerance = 1e-12
    lower = (1 + np.count_nonzero(null <= observed + tolerance)) / (null.size + 1)
    upper = (1 + np.count_nonzero(null >= observed - tolerance)) / (null.size + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def _adjust_fdr(statistics):
    """BH-adjust all unique reported tests within one mode and cohort."""
    finite = [stats for stats in statistics if np.isfinite(stats["p_perm"])]
    if not finite:
        return
    order = np.argsort([stats["p_perm"] for stats in finite])
    ranked = np.array([finite[i]["p_perm"] for i in order])
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    for index, q_value in zip(order, np.minimum(adjusted, 1.0)):
        finite[index]["q_fdr"] = float(q_value)


def _sharing_permutations(experiments, mode, n_permutations, permutation_seed):
    """Fixed row pairs with a label null conditional on the other factor.
    Average fractions equally across runs, including degenerate runs.
    Preserve dependence among pairs sharing a row in every permutation."""
    if mode not in ("task", "period"):
        raise ValueError("mode must be task or period")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    names = [name for name, _ in experiments]
    if len(names) != len(set(names)):
        raise ValueError("duplicate experiment names")
    rng = np.random.default_rng(permutation_seed)
    accumulated = {}
    group_rules = {group: {rule for pair in pairs for rule in pair}
                   for group, pairs in PAIR_GROUPS.items()}
    for aname, data in sorted(experiments, key=lambda item: item[0]):
        rules, phases, labels = _aligned_rows(aname, data)
        i, j = np.triu_indices(len(rules), k=1)
        same_rule, same_phase = rules[i] == rules[j], phases[i] == phases[j]
        keep = same_rule & ~same_phase if mode == "task" else same_phase & ~same_rule
        i, j = i[keep], j[keep]
        if i.size == 0:
            continue
        keys = rules[i] if mode == "task" else phases[i]
        selectors = {("pooled", "all"): np.ones(i.size, dtype=bool)}
        selectors.update({("individual", key): keys == key for key in np.unique(keys)})
        if mode == "task":
            selectors.update({("group", group): np.isin(keys, list(members))
                              for group, members in group_rules.items()})
        permutations = _restricted_permutations(
            phases if mode == "task" else rules, n_permutations, rng)
        for rep in REPRESENTATIONS:
            shared = labels[rep][i] == labels[rep][j]
            shuffled = labels[rep][permutations]
            null_shared = shuffled[:, i] == shuffled[:, j]
            for key, mask in selectors.items():
                n_cells = int(mask.sum())
                if not n_cells:
                    continue
                observed = int(shared[mask].sum())
                null_counts = null_shared[:, mask].sum(axis=1)
                acc = accumulated.setdefault((key, rep), {
                    "per_run": {}, "null_sum": np.zeros(n_permutations)})
                acc["per_run"][aname] = {
                    "observed": observed, "expected": float(null_counts.mean()),
                    "n_cells": n_cells}
                acc["null_sum"] += null_counts / n_cells

    results = {}
    for (key, rep), acc in accumulated.items():
        runs = acc["per_run"]
        observed = float(np.mean([run["observed"] / run["n_cells"]
                                  for run in runs.values()]))
        null = acc["null_sum"] / len(runs)
        results.setdefault(key, {})[rep] = {
            "pct_same_cluster": 100 * observed,
            "chance_mean": float(null.mean()),
            "n_same": sum(run["observed"] for run in runs.values()),
            "n_total": sum(run["n_cells"] for run in runs.values()),
            "n_runs": len(runs), "per_run": runs,
            "p_perm": _permutation_p(observed, null), "q_fdr": float("nan")}
    _adjust_fdr([stats for by_rep in results.values() for stats in by_rep.values()])
    by = "tasks" if mode == "task" else "periods"
    display_order = list(diag._TASK_DISPLAY if mode == "task" else diag._PHASE_DISPLAY)
    keys = [key for kind, key in results if kind == "individual"]
    keys.sort(key=lambda key: (display_order.index(key) if key in display_order
                               else len(display_order), key))
    metadata = {
        "null_definition": ("shuffle labels among tasks within each phase" if mode == "task"
                            else "shuffle labels among available phases within each task"),
        "test_statistic": "equal mean of per-network sharing fractions",
        "pair_selection": "all available pairs; identical row sets across representations",
        "n_permutations": int(n_permutations), "permutation_seed": int(permutation_seed),
        "p_definition": "two-sided inclusive equal tails with add-one correction",
        "n_fdr_tests": sum(len(by_rep) for by_rep in results.values()),
        "fdr_family": ("BH within this cohort and analysis type: all individual, all-pooled "
                       "and sibling-group-pooled tests across representations; repeated "
                       "panels of the same test counted once"),
    }
    entry = {"pooled": results.get(("pooled", "all"), {}),
             by: {key: results[("individual", key)] for key in keys},
             "permutation": metadata}
    groups = {}
    if mode == "task":
        for group, pairs in PAIR_GROUPS.items():
            groups[group] = {
                "pooled": results.get(("group", group), {}),
                "tasks": {rule: entry["tasks"][rule] for pair in pairs for rule in pair
                          if rule in entry["tasks"]},
                "permutation": metadata}
    return entry, groups


def summarize_same_task(experiments, n_permutations=10000, permutation_seed=0):
    """Sibling-group view of the same restricted tests used by the paper."""
    return _sharing_permutations(experiments, "task", n_permutations, permutation_seed)[1]


def summarize_same_task_all(experiments, n_permutations=10000, permutation_seed=0):
    """Every available task's phase pairs; no label-dependent pair exclusion."""
    return _sharing_permutations(experiments, "task", n_permutations, permutation_seed)[0]


def summarize_same_period_all(experiments, n_permutations=10000, permutation_seed=0):
    """Every available phase's task pairs, including single-phase tasks."""
    return _sharing_permutations(experiments, "period", n_permutations, permutation_seed)[0]


# Row-pair classes of the organization check: two rows of one period from
# different tasks, two rows of one task from different periods, rows that
# differ in both, and any two rows (the chance baseline).
ORGANIZATION_CLASSES = ("period", "task", "neither")


def _organization_counts(rep_data):
    """{class: {"n_pairs", "n_shared"}} over every two rows of one
    experiment's representation: "period" for one phase and two tasks, "task"
    for one task and two phases, "neither" for two of each and "any" for all
    pairs, with how many of the pairs share a row cluster."""
    rules, phases = _split_row_names(rep_data["tb_break_name"])
    labels = np.asarray(rep_data["row_labels"])
    upper = np.triu_indices(labels.size, k=1)
    same_rule = (rules[:, None] == rules[None, :])[upper]
    same_phase = (phases[:, None] == phases[None, :])[upper]
    shared = (labels[:, None] == labels[None, :])[upper]
    masks = {"period": same_phase & ~same_rule, "task": same_rule & ~same_phase,
             "neither": ~same_rule & ~same_phase, "any": np.ones(shared.size, dtype=bool)}
    return {name: {"n_pairs": int(mask.sum()), "n_shared": int(shared[mask].sum())}
            for name, mask in masks.items()}


def _enrichment(counts, name):
    """Fold enrichment of class `name`: its fraction of pairs in one row
    cluster over that of any two rows (NaN without pairs or sharing)."""
    pairs, base = counts[name], counts["any"]
    if pairs["n_pairs"] == 0 or base["n_shared"] == 0:
        return float("nan")
    return (pairs["n_shared"] / pairs["n_pairs"]) / (base["n_shared"] / base["n_pairs"])


def summarize_organization(experiments):
    """Are the rows grouped by period or by task, per representation?

    For every run, every row of every task enters: the fold enrichment
    (`_enrichment`) of each ORGANIZATION_CLASSES class over any two rows,
    whose sharing is the chance the run's cluster sizes give, so the
    representations, with their different numbers of row clusters, compare.
    Returns {"representations": {rep: {"anames", "counts", "enrichment":
    {class: per-run list}, "median": {class}, "p_vs_one": {class}}},
    "contrast_p": {"{a}_vs_{b}": p}}: `p_vs_one` is the two-sided Wilcoxon p
    of the runs' enrichment against 1 and `contrast_p` the paired one of the
    runs' task-minus-period enrichment between two representations."""
    result = {}
    for rep in REPRESENTATIONS:
        counts = {aname: _organization_counts(data[rep])
                  for aname, data in experiments if rep in data}
        if not counts:
            continue
        enrichment = {name: [_enrichment(run, name) for run in counts.values()]
                      for name in ORGANIZATION_CLASSES}
        finite = {name: np.asarray(values, dtype=float)[np.isfinite(values)]
                  for name, values in enrichment.items()}
        result[rep] = {
            "anames": list(counts),
            "counts": counts,
            "enrichment": enrichment,
            "median": {name: float(np.median(values)) if values.size else float("nan")
                       for name, values in finite.items()},
            "p_vs_one": {name: _excess_test(values - 1.0) if values.size else float("nan")
                         for name, values in finite.items()},
        }
    contrast_p = {}
    reps = list(result)
    for i, rep_a in enumerate(reps):
        for rep_b in reps[i + 1:]:
            contrast = {rep: dict(zip(result[rep]["anames"],
                                      np.subtract(result[rep]["enrichment"]["task"],
                                                  result[rep]["enrichment"]["period"])))
                        for rep in (rep_a, rep_b)}
            common = [aname for aname in contrast[rep_a] if aname in contrast[rep_b]]
            difference = np.array([contrast[rep_a][aname] - contrast[rep_b][aname]
                                   for aname in common])
            difference = difference[np.isfinite(difference)]
            contrast_p[f"{rep_a}_vs_{rep_b}"] = (_excess_test(difference)
                                                 if difference.size else float("nan"))
    return {"representations": result, "contrast_p": contrast_p}


# ─── Figures ─────────────────────────────────────────────────────────────────

_SUMMARY_PANEL_IN = (1.35, 1.45)          # (width, height) of one summary panel
_SUMMARY_MARKS = {"significant": "*", "unspecific": "†"}


def _summary_mark(stats):
    """'†' for an untested (unspecific) cell, '*' for p < SIGNIFICANCE in
    either direction (the bar against its chance dash shows which)."""
    if stats["unspecific"]:
        return _SUMMARY_MARKS["unspecific"]
    if stats["p_excess"] < SIGNIFICANCE:
        return _SUMMARY_MARKS["significant"]
    return ""


def _draw_summary_panel(ax, by_rep):
    """One group x phase-type panel: a bar per representation with the % of
    pooled (run, pair, phase) cells whose two rows share a cluster, a dash at
    the mean chance, one dot per run (that run's % of cells), and
    n_same/n_total above with its `_summary_mark`."""
    for x, rep in enumerate(REPRESENTATIONS):
        stats = by_rep.get(rep)
        if stats is None:
            ax.text(x, 3, "n/a", ha="center", va="bottom", fontsize=6, color="0.4")
            continue
        height = stats["pct_same_cluster"]
        chance = 100 * stats["chance_mean"]
        ax.bar(x, height, width=0.72, color=diag._REP_COLORS[rep], edgecolor="k",
               linewidth=0.5, alpha=0.85, zorder=2)
        ax.hlines(chance, x - 0.36, x + 0.36, color=diag._CHANCE_COLOR,
                  linewidth=1.4, zorder=4)
        runs = list(stats["per_run"].values())
        per_run = np.array([100 * run["observed"] / run["n_cells"] for run in runs])
        ax.scatter(x + np.linspace(-0.22, 0.22, per_run.size), per_run, s=4,
                   color="0.15", alpha=0.45, linewidths=0, zorder=3)
        top = max(height, chance, float(per_run.max()))
        ax.text(x, top + 3, f"{stats['n_same']}/{stats['n_total']}{_summary_mark(stats)}",
                ha="center", va="bottom", fontsize=6)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(REPRESENTATIONS) - 0.4)
    ax.set_ylim(0, diag._Y_TOP)
    ax.set_yticks([0, 50, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)


def plot_summary(summary, out_path, title):
    """A row per pair group and a column per phase type (`_draw_summary_panel`);
    a group without a phase type leaves its cell empty."""
    groups = [group for group in PAIR_GROUPS if summary.get(group)]
    phase_types = [phase_type for phase_type in PHASE_TYPES
                   if any(phase_type in summary[group] for group in groups)]
    if not groups or not phase_types:
        print("  [plot] no summary cells; skipping the summary figure.")
        return
    width, height = _SUMMARY_PANEL_IN
    fig, axes = plt.subplots(
        len(groups), len(phase_types), sharey=True, squeeze=False,
        layout="constrained",
        figsize=(1.6 + width * len(phase_types), 0.9 + height * len(groups)))
    for row, group in enumerate(groups):
        for col, phase_type in enumerate(phase_types):
            ax = axes[row, col]
            if row == 0:
                phases = ", ".join(diag._PHASE_DISPLAY[p] for p in PHASE_TYPES[phase_type])
                ax.set_title(f"{phase_type}\n({phases})", fontsize=8)
            if phase_type in summary[group]:
                _draw_summary_panel(ax, summary[group][phase_type])
            else:
                ax.axis("off")
        axes[row, 0].set_ylabel(f"{group}\n({len(PAIR_GROUPS[group])} pairs)",
                                rotation=0, ha="right", va="center", fontsize=8,
                                labelpad=6)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="k",
                             linewidth=0.5, alpha=0.85, label=rep)
               for rep, color in diag._REP_COLORS.items()]
    handles += [
        plt.Line2D([], [], color=diag._CHANCE_COLOR, linewidth=1.4, label="chance"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=2.5,
                   color="0.15", alpha=0.45, label="one run"),
        plt.Line2D([], [], linestyle="none", marker="$*$", markersize=5, color="k",
                   label=f"differs from chance, Wilcoxon p < {SIGNIFICANCE}"),
        plt.Line2D([], [], linestyle="none", marker="$†$", markersize=5, color="k",
                   label=f"chance ≥ {100 * UNSPECIFIC_CHANCE:.0f}%, not tested"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
               fontsize=7, handlelength=1.2, columnspacing=1.0)
    fig.suptitle(f"{title}\n% of (run, pair, phase) cells whose two task rows "
                 "share a row cluster", fontsize=9)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Same-task figure: a block of three bars (one per representation) per task,
# after the group's pooled block; blocks are _SAME_TASK_BLOCK apart on x.
_SAME_TASK_BLOCK = 1.0
_SAME_TASK_BAR = 0.26
_SAME_TASK_GAP = 0.5                      # extra space after the pooled block


def _draw_same_task_block(ax, x0, by_rep, show_runs):
    """Mean sharing over runs, permutation-null chance, and a BH-FDR star.
    Add one dot per run when `show_runs`."""
    for k, rep in enumerate(REPRESENTATIONS):
        stats = by_rep.get(rep)
        if stats is None:
            continue
        x = x0 + (k - 1) * _SAME_TASK_BAR
        height, chance = stats["pct_same_cluster"], 100 * stats["chance_mean"]
        ax.bar(x, height, width=_SAME_TASK_BAR * 0.9, color=diag._REP_COLORS[rep],
               edgecolor="k", linewidth=0.4, alpha=0.85, zorder=2)
        ax.hlines(chance, x - _SAME_TASK_BAR * 0.45, x + _SAME_TASK_BAR * 0.45,
                  color=diag._CHANCE_COLOR, linewidth=1.2, zorder=4)
        top = max(height, chance)
        if show_runs:
            runs = list(stats["per_run"].values())
            per_run = np.array([100 * run["observed"] / run["n_cells"] for run in runs])
            ax.scatter(x + np.linspace(-0.08, 0.08, per_run.size), per_run, s=3,
                       color="0.15", alpha=0.45, linewidths=0, zorder=3)
            top = max(top, float(per_run.max()))
        if stats["q_fdr"] < SIGNIFICANCE:
            ax.text(x, top + 2, "*", ha="center", va="bottom", fontsize=7)


def plot_same_task(same_task, out_path, title):
    """A row per pair group: the group's pooled block ("All") and then one
    block per task (`_draw_same_task_block`), every row on one x scale so the
    bars line up."""
    groups = [group for group in PAIR_GROUPS if same_task.get(group, {}).get("pooled")]
    if not groups:
        print("  [plot] no same-task cells; skipping the same-task figure.")
        return
    blocks = {}
    for group in groups:
        entry = same_task[group]
        rules = [rule for pair in PAIR_GROUPS[group] for rule in pair
                 if rule in entry["tasks"]]
        blocks[group] = ([("All", entry["pooled"], True)]
                         + [(diag._TASK_DISPLAY.get(rule, rule), entry["tasks"][rule], False)
                            for rule in rules])
    n_blocks = max(len(group_blocks) for group_blocks in blocks.values())
    fig, axes = plt.subplots(len(groups), 1, squeeze=False, layout="constrained",
                             figsize=(1.4 + 0.62 * n_blocks, 0.9 + 1.9 * len(groups)))
    for row, group in enumerate(groups):
        ax = axes[row, 0]
        centers = [b * _SAME_TASK_BLOCK + (_SAME_TASK_GAP if b else 0.0)
                   for b in range(len(blocks[group]))]
        for x0, (_, by_rep, show_runs) in zip(centers, blocks[group]):
            _draw_same_task_block(ax, x0, by_rep, show_runs)
        ax.set_xticks(centers)
        ax.set_xticklabels([label for label, _, _ in blocks[group]], rotation=35,
                           ha="right", fontsize=7)
        ax.set_xlim(-0.6, (n_blocks - 1) * _SAME_TASK_BLOCK + _SAME_TASK_GAP + 0.6)
        ax.set_ylim(0, diag._Y_TOP)
        ax.set_yticks([0, 50, 100])
        ax.spines["left"].set_bounds(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"{group} tasks", fontsize=8, loc="left")
        ax.set_ylabel("% of phase pairs\nin one cluster", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="k",
                             linewidth=0.4, alpha=0.85, label=rep)
               for rep, color in diag._REP_COLORS.items()]
    handles += [
        plt.Line2D([], [], color=diag._CHANCE_COLOR, linewidth=1.2,
                   label="chance (restricted permutation)"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=2.5,
                   color="0.15", alpha=0.45, label="one run (All)"),
        plt.Line2D([], [], linestyle="none", marker="$*$", markersize=5, color="k",
                   label=f"permutation, BH-FDR q < {SIGNIFICANCE}"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               fontsize=7, handlelength=1.2, columnspacing=1.0)
    fig.suptitle(f"{title}\nMean over runs: % of a task's phase pairs "
                 "in one row cluster", fontsize=9)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─── Driver ──────────────────────────────────────────────────────────────────

def _stats_line(rep, stats):
    """One printed line of pooled stats (`_summarize_cells`)."""
    test = (f"not tested (chance >= {100 * UNSPECIFIC_CHANCE:.0f}%)"
            if stats["unspecific"] else f"Wilcoxon p = {stats['p_excess']:.2g}")
    return (f"{rep:<11s}: {stats['pct_same_cluster']:5.1f}%  "
            f"({stats['n_same']}/{stats['n_total']} cells over {stats['n_runs']} "
            f"runs; chance {100 * stats['chance_mean']:.0f}%, mean excess per run "
            f"{stats['mean_excess']:+.2f}; {test})")


def _print_summary(summary):
    """One line per group, phase type and representation."""
    print("\n=== Summary by phase type ===")
    for group, by_type in summary.items():
        for phase_type, by_rep in by_type.items():
            print(f"  [{group}] {phase_type}")
            for rep, stats in by_rep.items():
                print(f"      {_stats_line(rep, stats)}")


def _sharing_stats_line(rep, stats):
    """The same run-mean statistic and restricted test displayed in the paper."""
    return (f"{rep:<11s}: mean {stats['pct_same_cluster']:.1f}% over "
            f"{stats['n_runs']} runs; {stats['n_same']}/{stats['n_total']} total pairs; "
            f"null mean {100 * stats['chance_mean']:.1f}%; "
            f"permutation p = {stats['p_perm']:.3g}, BH q = {stats['q_fdr']:.3g}")


def _print_same_task(same_task):
    """Per group, the pooled same-task lines, then one line per task and
    representation."""
    print("\n=== Same task, two phases ===")
    for group, entry in same_task.items():
        print(f"  [{group}] all tasks")
        for rep, stats in entry["pooled"].items():
            print(f"      {_sharing_stats_line(rep, stats)}")
        for rule, by_rep in entry["tasks"].items():
            print(f"    {diag._TASK_DISPLAY.get(rule, rule)}")
            for rep, stats in by_rep.items():
                print(f"      {_sharing_stats_line(rep, stats)}")


def _print_same_period(same_period):
    """The pooled same-period lines, then one line per phase and
    representation."""
    print("\n=== Same phase, two tasks ===")
    print("  all phases")
    for rep, stats in same_period["pooled"].items():
        print(f"      {_sharing_stats_line(rep, stats)}")
    for phase, by_rep in same_period["periods"].items():
        print(f"    {diag._PHASE_DISPLAY.get(phase, phase)}")
        for rep, stats in by_rep.items():
            print(f"      {_sharing_stats_line(rep, stats)}")


def _print_organization(organization):
    """Per representation, the median enrichment of each class and its test
    against 1, then the paired tests between representations."""
    print("\n=== Period or task: fold enrichment over any two rows ===")
    for rep, entry in organization["representations"].items():
        cells = "; ".join(f"{name} {entry['median'][name]:.2f}x "
                          f"(p = {entry['p_vs_one'][name]:.2g})"
                          for name in ORGANIZATION_CLASSES)
        print(f"  {rep:<11s}: median over {len(entry['anames'])} runs: {cells}")
    for name, p_value in organization["contrast_p"].items():
        print(f"  task - period enrichment, {name.replace('_vs_', ' vs ')}: "
              f"paired Wilcoxon p = {p_value:.2g}")


def main(feature="L21e4", seed=None, untrained=False,
         n_permutations=10000, permutation_seed=0):
    """Analyze every clustered run of `feature` (or one `seed`), or with
    `untrained` every untrained control of `feature`'s architecture (module
    docstring), save the results and draw the three figures."""
    tag = feature if feature else "all"
    if seed is not None:
        tag += f"_seed{seed}"
    root = None
    if untrained:
        if not feature:
            raise ValueError("--untrained needs --feature to name the architecture of the controls.")
        root = UNTRAINED_DIR
        tag = untrained_sharing_tag(feature)
        if seed is not None:
            tag = f"{tag}_seed{seed}"
        anames = diag._discover_anames(seed=seed, root=root,
                                       architecture=architecture_tag(feature))
    else:
        anames = diag._discover_anames(feature, seed)

    print(f"Discovered {len(anames)} {'untrained ' if untrained else ''}experiment(s) "
          f"(feature={feature}, seed={seed})")
    experiments = []
    for aname in anames:
        data = diag._load_experiment(aname, root=root)
        if data is not None:
            experiments.append((aname, data))
    print(f"Loaded cluster info for {len(experiments)}/{len(anames)} experiment(s).")
    if not experiments:
        print("No experiments with usable cluster info; nothing to analyze.")
        return None

    pair_results = analyze_pairs(experiments)
    for group, results in pair_results.items():
        for result in results.values():
            diag._print_result(f"{group}: {result['rule_a_display']} vs "
                               f"{result['rule_b_display']}", result)
    summary = summarize_by_phase_type(pair_results)
    _print_summary(summary)
    same_task_all, same_task = _sharing_permutations(
        experiments, "task", n_permutations, permutation_seed)
    _print_same_task(same_task)
    _print_same_task({"All tasks": same_task_all})
    same_period_all = summarize_same_period_all(
        experiments, n_permutations, permutation_seed)
    _print_same_period(same_period_all)
    organization = summarize_organization(experiments)
    _print_organization(organization)

    # Validate and compute before replacing any previous result files.
    removed = _clear_previous_results(tag)
    print(f"Cleared {len(removed)} previous {tag} result file(s) from {OUT_DIR}/")
    title = f"Sibling row-cluster sharing ({tag}, {len(experiments)} experiments)"
    diag.plot_combined_same_cluster(
        {key: result for results in pair_results.values()
         for key, result in results.items()},
        OUT_DIR / f"sibling_row_sharing_pairs_{tag}.png", title=title)
    plot_summary(summary, OUT_DIR / f"sibling_row_sharing_summary_{tag}.png", title)
    plot_same_task(same_task, OUT_DIR / f"sibling_row_sharing_same_task_{tag}.png",
                   f"Same-task row-cluster sharing ({tag}, {len(experiments)} experiments)")

    payload = {
        "version": RESULT_VERSION,
        # The untrained controls belong to an architecture, not to one L2
        # cohort (module docstring); their config runs' features are listed.
        "feature": None if untrained else feature,
        "architecture": architecture_tag(feature) if feature else None,
        "config_features": sorted({diag._run_feature(aname) for aname, _ in experiments}),
        "seed": seed,
        "mod_g_index": diag.MOD_G_INDEX,
        "experiments": [aname for aname, _ in experiments],
        "pair_groups": {group: [list(pair) for pair in pairs]
                        for group, pairs in PAIR_GROUPS.items()},
        "untrained": bool(untrained),
        "phase_types": {name: list(phases) for name, phases in PHASE_TYPES.items()},
        "unspecific_chance": UNSPECIFIC_CHANCE,  # legacy sibling phase-type summary only
        "pairs": pair_results,
        "summary": summary,
        "same_task": same_task,
        "same_task_all": same_task_all,
        "same_period_all": same_period_all,
        "organization": organization,
    }
    pkl_path = OUT_DIR / f"sibling_row_sharing_{tag}.pkl"
    json_path = OUT_DIR / f"sibling_row_sharing_{tag}.json"
    with open(pkl_path, "wb") as stream:
        pickle.dump(payload, stream)
    with open(json_path, "w") as stream:
        json.dump(diag._json_safe(payload), stream, indent=2)
    print(f"\nSaved: {pkl_path}")
    print(f"Saved: {json_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Row-cluster sharing: restricted same-task/same-period "
                    "permutation tests, organization and legacy sibling-pair diagnostics.")
    parser.add_argument("--feature", type=str, default="L21e4",
                        help="Only analyze models with this exact feature "
                             "(default: 'L21e4').")
    parser.add_argument("--seed", type=int, default=None,
                        help="Only analyze the model with this seed (with "
                             "--untrained: the config's run).")
    parser.add_argument("--untrained", action="store_true",
                        help="Analyze the untrained controls of "
                             "untrained_clustering.py instead of the trained runs.")
    parser.add_argument("--n-permutations", type=int, default=10000,
                        help="Restricted row-label shuffles for task/period tests (default: 10000).")
    parser.add_argument("--permutation-seed", type=int, default=0,
                        help="RNG seed for the restricted permutations (default: 0).")
    args = parser.parse_args()
    main(feature=args.feature, seed=args.seed, untrained=args.untrained,
         n_permutations=args.n_permutations, permutation_seed=args.permutation_seed)
