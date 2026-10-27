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

The summary pools each group's pairs by phase type: Stimulus (stim1, stim2),
Memory (delay1, delay2) and Response (go1). Per run it counts the group's
(pair, phase) cells of that type whose two rows share a cluster and subtracts
the count chance predicts, the sum of the cells' chance levels; a two-sided
Wilcoxon signed-rank test over the runs asks whether that excess is zero.
Runs are independent, but the cells of one run come from one clustering, so
the per-cell tests are not pooled. A summary cell whose mean chance is at
least UNSPECIFIC_CHANCE (input and hidden in the memory periods, where every
task's row falls in one cluster) is reported but not tested: sharing there
is not specific to the pair.

Sharing within a phase says whether a representation groups rows by period;
the same-task check asks whether it groups them by task. For every task of
the pairs and every two of its phases, it records whether the task's two
rows share a row cluster, against the chance that a row of the one phase
and a row of the other from two different tasks share one
(`_cross_phase_chance`), so the phase combination is controlled for. The
cells are pooled per task and per group and tested like the summary
(per-run excess over chance, Wilcoxon over the runs); cells whose own
chance is at least UNSPECIFIC_CHANCE (the two memory periods in input and
hidden) are left out and counted. The same check also runs over every task
of the runs (`summarize_same_task_all`), the per-task view of the paper.

The same-period check is its mirror over every task (`summarize_same_period_all`):
for every phase and every two tasks with a row of it, whether the two rows
share a row cluster, against the chance that a row of the one task and a row
of the other from two different phases share one (`_cross_period_chance`), so
how alike the two tasks are anyway is controlled for. Cells are pooled per
phase and over the phases and tested as above, and cells at chance of at
least UNSPECIFIC_CHANCE are left out and counted.

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
sibling_row_sharing_same_task_{tag}.png, with the tag
{feature}[_seed{seed}][_untrained]. A run replaces its own tag's files only.
With --untrained it analyzes the untrained controls under
multiple_tasks_analysis/untrained/ (`untrained_clustering.py`; --seed is then
the config's run and every init seed of it enters) instead of the trained
runs.

Usage:
    python multiple_task/sibling_row_sharing.py --feature L21e4
    python multiple_task/sibling_row_sharing.py --feature L21e3
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

# Saved-result layout of this script: 2 adds the same-task check, 3 its
# all-task version and the organization check, 4 the same-period check
# (paper_plot reads only 4).
RESULT_VERSION = 4
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


def _cross_phase_chance(rep_data, phase_a, phase_b):
    """Chance that one task's `phase_a` and `phase_b` rows share a row
    cluster in one experiment's representation: the fraction of pairs of a
    `phase_a` row and a `phase_b` row from two different tasks that carry the
    same label (NaN without such a pair). It is the baseline a task's own two
    rows of those phases are read against, as `_same_phase_chance` is for a
    sibling pair of one phase."""
    rules, phases = _split_row_names(rep_data["tb_break_name"])
    labels = np.asarray(rep_data["row_labels"])
    in_a, in_b = phases == phase_a, phases == phase_b
    pairs = rules[in_a][:, None] != rules[in_b][None, :]
    if not pairs.any():
        return float("nan")
    return float((labels[in_a][:, None] == labels[in_b][None, :])[pairs].mean())


def _same_task_cells(experiments, rules):
    """{rep: {rule: [cell, ...]}}: one cell per run and pair of a task's
    phases, with whether the two rows share a row cluster and the chance of
    that phase combination (`_cross_phase_chance`)."""
    phase_order = list(diag._PHASE_DISPLAY)
    cells = {rep: {rule: [] for rule in rules} for rep in REPRESENTATIONS}
    for aname, data in experiments:
        for rep in REPRESENTATIONS:
            if rep not in data:
                continue
            row_rules, row_phases = _split_row_names(data[rep]["tb_break_name"])
            labels = np.asarray(data[rep]["row_labels"])
            chances = {}
            for rule in rules:
                rows = sorted(np.flatnonzero(row_rules == rule),
                              key=lambda i: (phase_order.index(row_phases[i])
                                             if row_phases[i] in phase_order
                                             else len(phase_order)))
                for k, i in enumerate(rows):
                    for j in rows[k + 1:]:
                        phases = (str(row_phases[i]), str(row_phases[j]))
                        if phases not in chances:
                            chances[phases] = _cross_phase_chance(data[rep], *phases)
                        cells[rep][rule].append({
                            "aname": aname, "phases": list(phases),
                            "same": bool(labels[i] == labels[j]),
                            "chance": chances[phases]})
    return cells


def summarize_same_task(experiments):
    """{group: {"pooled": {rep: stats}, "tasks": {rule: {rep: stats}},
    "n_unspecific_cells": {rep: count}}}: whether a task's rows of two of
    its phases share a row cluster more often than two different tasks'
    rows of those phases, pooled per task and per group (`_summarize_cells`).
    Cells whose own chance is at least UNSPECIFIC_CHANCE are left out of the
    stats and counted."""
    rules = [rule for pairs in PAIR_GROUPS.values() for pair in pairs for rule in pair]
    cells = _same_task_cells(experiments, rules)
    return {group: _cell_entry(cells, [rule for pair in pairs for rule in pair], "tasks")
            for group, pairs in PAIR_GROUPS.items()}


def _cell_entry(cells, keys, by):
    """The stats of `keys` (tasks or phases) from `_same_task_cells` or
    `_same_period_cells` output: per key under `by` and pooled over `keys`,
    cells at chance >= UNSPECIFIC_CHANCE left out and counted."""
    entry = {"pooled": {}, by: {}, "n_unspecific_cells": {}}
    for rep in REPRESENTATIONS:
        pooled, n_unspecific = [], 0
        for key in keys:
            specific = [cell for cell in cells[rep][key]
                        if not cell["chance"] >= UNSPECIFIC_CHANCE]
            n_unspecific += len(cells[rep][key]) - len(specific)
            stats = _summarize_cells(specific)
            if stats is not None:
                entry[by].setdefault(key, {})[rep] = stats
            pooled += specific
        stats = _summarize_cells(pooled)
        if stats is not None:
            entry["pooled"][rep] = stats
        entry["n_unspecific_cells"][rep] = n_unspecific
    return entry


def _present_rules(experiments):
    """Every rule with a clustered row in some run, in diag._TASK_DISPLAY
    order (unknown rules last, by name)."""
    order = list(diag._TASK_DISPLAY)
    rules = {str(rule) for _, data in experiments for rep_data in data.values()
             for rule in _split_row_names(rep_data["tb_break_name"])[0]}
    return sorted(rules, key=lambda rule: (order.index(rule) if rule in order
                                           else len(order), rule))


def summarize_same_task_all(experiments):
    """The same-task check over every task of the runs (`_cell_entry`), not
    only the sibling pairs'; a task with one clustered phase has no cells."""
    rules = _present_rules(experiments)
    return _cell_entry(_same_task_cells(experiments, rules), rules, "tasks")


def _cross_period_chance(rep_data, rule_a, rule_b):
    """Chance that two tasks' rows of one phase share a row cluster in one
    experiment's representation: the fraction of pairs of a `rule_a` row and
    a `rule_b` row of two different phases that carry the same label (NaN
    without such a pair). It is the baseline the two tasks' rows of one phase
    are read against, as `_cross_phase_chance` is for one task's rows of two
    phases, so how alike the two tasks are anyway is controlled for."""
    rules, phases = _split_row_names(rep_data["tb_break_name"])
    labels = np.asarray(rep_data["row_labels"])
    in_a, in_b = rules == rule_a, rules == rule_b
    pairs = phases[in_a][:, None] != phases[in_b][None, :]
    if not pairs.any():
        return float("nan")
    return float((labels[in_a][:, None] == labels[in_b][None, :])[pairs].mean())


def _same_period_cells(experiments, phases):
    """{rep: {phase: [cell, ...]}}: one cell per run, phase and pair of tasks
    with a row of the phase, with whether the two rows share a row cluster and
    the chance of that task pair (`_cross_period_chance`)."""
    cells = {rep: {phase: [] for phase in phases} for rep in REPRESENTATIONS}
    for aname, data in experiments:
        for rep in REPRESENTATIONS:
            if rep not in data:
                continue
            row_rules, row_phases = _split_row_names(data[rep]["tb_break_name"])
            labels = np.asarray(data[rep]["row_labels"])
            chances = {}
            for phase in phases:
                rows = np.flatnonzero(row_phases == phase)
                for k, i in enumerate(rows):
                    for j in rows[k + 1:]:
                        rules = tuple(sorted((str(row_rules[i]), str(row_rules[j]))))
                        if rules not in chances:
                            chances[rules] = _cross_period_chance(data[rep], *rules)
                        cells[rep][phase].append({
                            "aname": aname, "rules": list(rules),
                            "same": bool(labels[i] == labels[j]),
                            "chance": chances[rules]})
    return cells


def summarize_same_period_all(experiments):
    """{"pooled": {rep: stats}, "periods": {phase: {rep: stats}},
    "n_unspecific_cells": {rep: count}}: whether two different tasks' rows of
    one phase share a row cluster more often than the same two tasks' rows of
    two different phases, per phase of the runs (in diag._PHASE_DISPLAY
    order) and pooled over them (`_cell_entry`): the per-phase view of the
    organization check's period class."""
    present = {str(phase) for _, data in experiments for rep_data in data.values()
               for phase in _split_row_names(rep_data["tb_break_name"])[1]}
    phases = [phase for phase in diag._PHASE_DISPLAY if phase in present]
    return _cell_entry(_same_period_cells(experiments, phases), phases, "periods")


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
    """One block: a bar per representation with the % of cells whose two
    rows share a cluster, a dash at the mean chance, '*' above when the
    per-run excess differs from zero in either direction (p < SIGNIFICANCE),
    and one dot per run when `show_runs`."""
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
        if stats["p_excess"] < SIGNIFICANCE:
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
        left_out = same_task[group]["n_unspecific_cells"]
        note = ", ".join(f"{rep} {count}" for rep, count in left_out.items() if count)
        ax.set_title(f"{group} tasks" + (f" (cells at chance ≥ "
                                          f"{100 * UNSPECIFIC_CHANCE:.0f}% left out: "
                                          f"{note})" if note else ""),
                     fontsize=8, loc="left")
        ax.set_ylabel("% of phase pairs\nin one cluster", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="k",
                             linewidth=0.4, alpha=0.85, label=rep)
               for rep, color in diag._REP_COLORS.items()]
    handles += [
        plt.Line2D([], [], color=diag._CHANCE_COLOR, linewidth=1.2,
                   label="chance (two tasks' rows of those phases)"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=2.5,
                   color="0.15", alpha=0.45, label="one run (All)"),
        plt.Line2D([], [], linestyle="none", marker="$*$", markersize=5, color="k",
                   label=f"differs from chance, Wilcoxon p < {SIGNIFICANCE}"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               fontsize=7, handlelength=1.2, columnspacing=1.0)
    fig.suptitle(f"{title}\n% of (run, phase pair) cells whose two rows of one "
                 "task share a row cluster", fontsize=9)
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


def _print_same_task(same_task):
    """Per group, the pooled same-task lines, then one line per task and
    representation."""
    print("\n=== Same task, two phases ===")
    for group, entry in same_task.items():
        left_out = ", ".join(f"{rep} {count}" for rep, count
                             in entry["n_unspecific_cells"].items() if count)
        print(f"  [{group}] all tasks"
              + (f" (cells at chance >= {100 * UNSPECIFIC_CHANCE:.0f}% left out: "
                 f"{left_out})" if left_out else ""))
        for rep, stats in entry["pooled"].items():
            print(f"      {_stats_line(rep, stats)}")
        for rule, by_rep in entry["tasks"].items():
            print(f"    {diag._TASK_DISPLAY.get(rule, rule)}")
            for rep, stats in by_rep.items():
                print(f"      {_stats_line(rep, stats)}")


def _print_same_period(same_period):
    """The pooled same-period lines, then one line per phase and
    representation."""
    print("\n=== Same phase, two tasks ===")
    left_out = ", ".join(f"{rep} {count}" for rep, count
                         in same_period["n_unspecific_cells"].items() if count)
    print("  all phases" + (f" (cells at chance >= {100 * UNSPECIFIC_CHANCE:.0f}% left out: "
                            f"{left_out})" if left_out else ""))
    for rep, stats in same_period["pooled"].items():
        print(f"      {_stats_line(rep, stats)}")
    for phase, by_rep in same_period["periods"].items():
        print(f"    {diag._PHASE_DISPLAY.get(phase, phase)}")
        for rep, stats in by_rep.items():
            print(f"      {_stats_line(rep, stats)}")


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


def main(feature="L21e4", seed=None, untrained=False):
    """Analyze every clustered run of `feature` (or one `seed`), or with
    `untrained` every untrained control of them, save the results and draw
    the three figures."""
    tag = feature if feature else "all"
    if seed is not None:
        tag += f"_seed{seed}"
    root = None
    if untrained:
        tag += "_untrained"
        root = UNTRAINED_DIR
    removed = _clear_previous_results(tag)
    print(f"Cleared {len(removed)} previous {tag} result file(s) from {OUT_DIR}/")

    anames = diag._discover_anames(feature, seed, root=root)
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
    same_task = summarize_same_task(experiments)
    _print_same_task(same_task)
    same_task_all = summarize_same_task_all(experiments)
    _print_same_task({"All tasks": same_task_all})
    same_period_all = summarize_same_period_all(experiments)
    _print_same_period(same_period_all)
    organization = summarize_organization(experiments)
    _print_organization(organization)

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
        "feature": feature,
        "seed": seed,
        "mod_g_index": diag.MOD_G_INDEX,
        "experiments": [aname for aname, _ in experiments],
        "pair_groups": {group: [list(pair) for pair in pairs]
                        for group, pairs in PAIR_GROUPS.items()},
        "untrained": bool(untrained),
        "phase_types": {name: list(phases) for name, phases in PHASE_TYPES.items()},
        "unspecific_chance": UNSPECIFIC_CHANCE,
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
        description="Row-cluster sharing of seven sibling-task pairs, per pair "
                    "and summarized by phase type.")
    parser.add_argument("--feature", type=str, default="L21e4",
                        help="Only analyze models with this exact feature "
                             "(default: 'L21e4').")
    parser.add_argument("--seed", type=int, default=None,
                        help="Only analyze the model with this seed (with "
                             "--untrained: the config's run).")
    parser.add_argument("--untrained", action="store_true",
                        help="Analyze the untrained controls of "
                             "untrained_clustering.py instead of the trained runs.")
    args = parser.parse_args()
    main(feature=args.feature, seed=args.seed, untrained=args.untrained)
