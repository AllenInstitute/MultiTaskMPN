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

The matched-task supplement retains the original cluster labels but restricts
every phase to the same five integration tasks (ten task pairs). It repeats
the restricted-null tests in a separate BH family and saves two prespecified
paired phase contrasts, Stimulus 2 minus 1 and Memory 2 minus 1. Contrasts use
whole-network sign flips and a separate six-test BH family; they do not test
trained versus untrained. Companion same-task and organization checks use
the identical 25 rows. Same-task nulls shuffle only the five selected tasks
within each phase; its five task tests plus pooled (three representations)
form an 18-test BH family without legacy sibling summaries. Organization
recomputes its baseline over the 300 selected row pairs (50 same-period,
50 same-task, 200 neither). No trials or clusterings are regenerated.
If any network lacks a required task/phase row, the driver skips the matched
supplements and saves the all-task results without their optional keys.
Malformed or inconsistent row data still raise an error.
An optional cut-robustness supplement recuts the full modulation row trees
at K=4,6,...,20 before selecting those same rows. It retains tied merge
heights and records actual cluster counts. Its all-task twin
(`cut_robustness_all`) keeps every row of every task instead, against the
all-pairs baseline of the organization check, and needs no five-task grid.
--cut-robustness-only adds both checks to an existing result using exactly
its saved cohort, without rerunning permutations or discovering additional
networks.

The variance-partition supplements use saved normalized feature matrices,
independently of cluster labels or K. One retains all observed task-period rows;
the other selects the same five-task/five-phase grid. Both use additive task +
period least-squares models and average each factor's incremental explained sum
of squares over the two entry orders. This equals the orthogonal decomposition
on the balanced matched grid. On the incomplete all-task grid it is an explicit
allocation of overlapping explanatory information, not unique main effects.
Observed rows have equal weight; nonexistent phases are never filled with zero.
Normalization is retained from all tasks; subsets are not renormalized.
Fractions are computed per network before cohort averaging. The residual includes
nonadditive interaction and estimation noise; constant profiles are undefined.
--variance-partition-only updates both supplements for the saved cohort without
repeating sharing permutations. Missing features skip them in a normal run;
in update-only mode they leave the existing summary untouched. Missing matched
rows skip only the matched supplement; all-task results are still available.

The organization check puts the two questions on one scale for every task's
rows (`summarize_organization`): per run and representation, the fold
enrichment of two rows of one period from different tasks, and of two rows
of one task from different periods, sharing a row cluster over any two rows
sharing one, the chance the cluster sizes give, so the representations,
with their different numbers of row clusters, compare. It is tested against
1 over the runs, and the runs' task-minus-period enrichment is compared
between representations (paired Wilcoxon). The result also records each
network's selected cluster counts (`cluster_counts`: the `row_tol_k` and
`col_tol_k` of every representation) for the paper's cluster-count figure.

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
    # recompute both cut checks for the exact cohort of an existing matched summary
    python multiple_task/sibling_row_sharing.py --feature L21e4 --cut-robustness-only
    # add task/period feature fractions without repeating sharing permutations
    python multiple_task/sibling_row_sharing.py --feature L21e4 --variance-partition-only
    python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained --variance-partition-only
"""
import json
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are saved, never shown
import matplotlib.pyplot as plt
from scipy.stats import permutation_test, wilcoxon
from scipy.cluster.hierarchy import fcluster

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from core.multitask_naming import architecture_tag, untrained_sharing_tag
if __package__:
    from . import multiple_task_diagnosis as diag
else:
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

# Fixed before inspecting cluster labels; identical in every cohort/network.
MATCHED_TASKS = ("contextdelaydm1", "contextdelaydm2", "delaydm1", "delaydm2",
                 "multidelaydm")
MATCHED_PHASES = ("stim1", "stim2", "delay1", "delay2", "go1")
PERIOD_CONTRASTS = (("stim2", "stim1"), ("delay2", "delay1"))
CUT_ROBUSTNESS_K = tuple(range(4, 21, 2))  # 4, 6, ..., 20
VARIANCE_COMPONENTS = ("task", "period", "residual")
VARIANCE_NORMALIZATION = "per-feature maximum over all task-period rows; subset not renormalized"
VARIANCE_ALLOCATION = "mean incremental explained sum of squares over both factor orders"
VARIANCE_ROW_WEIGHTING = "equal observed task-period rows"
VARIANCE_VIEWS = {"variance_partition_all": False, "variance_partition_matched_tasks": True}

# A summary cell at or above this mean chance is not tested: every task's row
# of those phases shares one cluster, so sharing says nothing about the pair.
UNSPECIFIC_CHANCE = 0.95
SIGNIFICANCE = 0.05

# Saved-result layout: 2 adds same-task, 3 all-task and organization,
# 4 same-period, 5 restricted permutations with common row pairs, equal
# network means and BH-adjusted q values; 6 requires trial_time variance
# with actual trial endpoints. paper_plot requires version 6.
# The matched-task, cut, cluster-count and variance supplements are additive and carry
# their own schema_version; existing v6 all-task results remain valid without
# them (a figure that needs one asks for a rerun).
RESULT_VERSION = 6
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


def _aligned_cohort(experiments):
    """(aname, row names, phases, {representation: row labels}) of every
    network of `experiments` (an iterable of (aname, clustering), so full
    source caches can be read one network at a time), rows in one canonical
    order. Raises ValueError for a duplicate or missing network, missing or
    nonfinite labels, or networks with different task-period rows, so a
    cohort statistic never changes its denominator silently."""
    names, anames = None, set()
    for aname, data in experiments:
        if aname in anames:
            raise ValueError(f"{aname}: duplicate network in co-clustering cohort")
        if data is None:
            raise ValueError(f"{aname}: missing current clustering")
        rules, phases, labels = _aligned_rows(aname, data)
        current = np.array([f"{rule}-{phase}" for rule, phase in zip(rules, phases)])
        if current.size == 0:
            raise ValueError(f"{aname}: no task-period rows")
        if names is None:
            names = current
        elif not np.array_equal(names, current):
            raise ValueError(f"{aname}: networks have different task/phase rows")
        for rep, values in labels.items():
            if not np.isfinite(values).all():
                raise ValueError(f"{aname}/{rep}: nonfinite cluster labels")
        anames.add(aname)
        yield aname, current, phases, labels


def summarize_row_coassignment(experiments):
    """Fraction of networks placing each pair of task-period rows in one cluster.

    Align row identities across all networks and representations before counting.
    Compare labels only within a network: relabeling its clusters cannot change
    the result. Every cell uses the same network denominator; missing rows or
    representations reject the cohort rather than becoming zero or reducing N.
    The symmetric matrices include the unit diagonal and are descriptive raw
    frequencies, without a chance correction or reclustering. `joint` is the
    same fraction for sharing a cluster in every representation at once,
    within one network. Accepts an iterator so full source caches can be read
    one network at a time.
    """
    names, counts, joint, anames = None, {}, None, []
    for aname, current, _, labels in _aligned_cohort(experiments):
        if names is None:
            names = current
            counts = {rep: np.zeros((len(names), len(names)), dtype=int)
                      for rep in REPRESENTATIONS}
            joint = np.zeros((len(names), len(names)), dtype=int)
        together = np.ones((len(names), len(names)), dtype=bool)
        for rep, values in labels.items():
            same = values[:, None] == values[None, :]
            counts[rep] += same
            together &= same
        joint += together
        anames.append(aname)
    if not anames:
        raise ValueError("No networks for co-clustering")
    return {"row_names": names.tolist(), "experiments": anames, "n_runs": len(anames),
            "representations": {rep: count / len(anames) for rep, count in counts.items()},
            "joint": joint / len(anames)}


# Chronological order of the trial periods; a period outside it sorts after.
PERIOD_ORDER = ("stim1", "delay1", "stim2", "delay2", "go1")


def summarize_period_pair_coassignment(experiments):
    """Per network, how often two rows of given period types share a row cluster.

    For every unordered pair of period types (a period with itself included:
    two rows of one period, which are of different tasks), count in each
    network the row pairs of those two types and the ones that fall in one
    cluster, per representation; `fractions` holds that network-level
    probability (networks x period pairs), so cohort statistics weight every
    network equally. A pair of different periods includes the rows of one
    task as well as those of two tasks. `any` is the same fraction over all
    row pairs of the network, the level its cluster sizes alone give. Rows
    are aligned and validated as in summarize_row_coassignment, so every
    network counts the same row pairs (`n_row_pairs`). Raw descriptive
    fractions: no chance correction, no test.
    """
    pairs, masks, upper, anames = None, None, None, []
    fractions = {rep: [] for rep in REPRESENTATIONS}
    overall = {rep: [] for rep in REPRESENTATIONS}
    for aname, _, phases, labels in _aligned_cohort(experiments):
        if pairs is None:
            present = sorted(set(phases.tolist()), key=lambda phase: (
                PERIOD_ORDER.index(phase) if phase in PERIOD_ORDER else len(PERIOD_ORDER), phase))
            upper = np.triu(np.ones((phases.size, phases.size), dtype=bool), k=1)
            first, second = phases[:, None], phases[None, :]
            masks = {(a, b): upper & (((first == a) & (second == b)) | ((first == b) & (second == a)))
                     for index, a in enumerate(present) for b in present[index:]}
            pairs = [pair for pair, mask in masks.items() if mask.any()]
            if not pairs:
                raise ValueError(f"{aname}: fewer than two task-period rows")
        for rep, values in labels.items():
            same = values[:, None] == values[None, :]
            fractions[rep].append([same[masks[pair]].mean() for pair in pairs])
            overall[rep].append(same[upper].mean())
        anames.append(aname)
    if not anames:
        raise ValueError("No networks for co-clustering")
    return {"schema_version": 1, "pairs": pairs,
            "n_row_pairs": [int(masks[pair].sum()) for pair in pairs],
            "experiments": anames, "n_runs": len(anames),
            "representations": {rep: {"fractions": np.asarray(fractions[rep], dtype=float),
                                      "any": np.asarray(overall[rep], dtype=float)}
                                for rep in REPRESENTATIONS}}


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


def _sharing_permutations(experiments, mode, n_permutations, permutation_seed,
                          include_groups=True):
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
        if mode == "task" and include_groups:
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
    if mode == "task" and include_groups:
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


def _matched_period_contrasts(periods, n_permutations, permutation_seed):
    """Swap whole phase summaries within networks, never individual task pairs.

    Exact sign enumeration when feasible, otherwise Monte Carlo sign flips.
    The null assumes the two phase summaries are exchangeable within a network.
    These six tests form their own BH family, separate from the chance tests.
    """
    contrasts = {}
    for later, earlier in PERIOD_CONTRASTS:
        by_rep = {}
        for rep in REPRESENTATIONS:
            first = periods[earlier][rep]["per_run"]
            second = periods[later][rep]["per_run"]
            if set(first) != set(second):
                raise ValueError("paired period comparisons require identical networks")
            differences = {
                name: 100 * (second[name]["observed"] / second[name]["n_cells"]
                             - first[name]["observed"] / first[name]["n_cells"])
                for name in sorted(first)}
            values = np.array(list(differences.values()))
            p_value = 1.0
            if values.size > 1 and np.any(values):
                p_value = float(permutation_test(
                    (values,), lambda x, axis: np.mean(x, axis=axis),
                    permutation_type="samples", vectorized=True,
                    n_resamples=n_permutations, alternative="two-sided",
                    random_state=permutation_seed).pvalue)
            by_rep[rep] = {
                "mean_difference_pp": float(values.mean()),
                "sd_difference_pp": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "per_run_difference_pp": differences, "n_runs": len(differences),
                "p_perm": p_value, "q_fdr": float("nan")}
        contrasts[f"{later}_minus_{earlier}"] = by_rep
    _adjust_fdr([stats for by_rep in contrasts.values() for stats in by_rep.values()])
    return {
        "contrasts": contrasts,
        "test_definition": "two-sided paired sign-flip test of the mean network difference",
        "null_definition": "phase summaries exchangeable within each network",
        "permutation_unit": "network (all task pairs swapped together)",
        "n_resamples": int(n_permutations), "permutation_seed": int(permutation_seed),
        "p_definition": "inclusive equal tails; exact if 2**n_runs <= n_resamples, else add-one",
        "n_fdr_tests": len(PERIOD_CONTRASTS) * len(REPRESENTATIONS),
        "fdr_family": "BH within this cohort: two prespecified phase contrasts x three representations",
    }


class MissingMatchedTaskRows(ValueError):
    """Valid row data do not cover the fixed five-task, five-phase grid."""


class MissingVarianceFeatures(ValueError):
    """A cache lacks a representation's saved normalized feature matrix."""


def partition_task_period_variance(features, tasks=None, periods=None):
    """Allocate task/period explained SS by averaging the two factor entry orders.

    Accept a complete task x period x feature grid, or observed row x feature
    profiles with task/period labels. Fit task-only, period-only and additive
    models with intercepts, giving residual SS S_T, S_P and S_TP. Task receives
    [(S_0 - S_T) + (S_P - S_TP)] / 2, period the symmetric expression, and
    residual S_TP, where S_0 is centered total SS. Divide after summing features.
    On a balanced grid this is the orthogonal two-factor decomposition. On an
    incomplete grid it allocates overlapping explanatory information equally
    over orders; it does not identify unique independent contributions.

    Rows are equally weighted; absent task-period cells are not imputed. SVD
    projections handle rank-deficient designs, including confounded factors.
    Residual includes nonadditive interaction and estimation noise. Constant
    profiles have undefined fractions. No neuron/synapse is a statistical replicate.
    """
    values = np.asarray(features, dtype=np.float64)
    if tasks is None and periods is None:
        if values.ndim != 3 or min(values.shape[:2]) < 2:
            raise ValueError("Variance partition requires a finite task x period x feature grid")
        n_tasks, n_periods, n_features = values.shape
        tasks = np.repeat(np.arange(n_tasks), n_periods)
        periods = np.tile(np.arange(n_periods), n_tasks)
        values = values.reshape(n_tasks * n_periods, n_features)
    tasks, periods = np.asarray(tasks), np.asarray(periods)
    if (values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0
            or not np.isfinite(values).all() or tasks.shape != (values.shape[0],)
            or periods.shape != tasks.shape):
        raise ValueError("Variance partition requires finite row x feature data and aligned factor labels")
    # Subtract one cell first so even a non-binary constant (e.g. 0.3) has
    # exactly zero variation, without a rounding artifact from its mean.
    centered = values - values[:1]
    centered -= centered.mean(axis=0, keepdims=True)
    total = float(np.sum(centered ** 2))
    # Full indicator matrices each span the intercept, including the joint
    # rank-deficient design. Only their column spaces enter the calculation.
    task_design = (tasks[:, None] == np.unique(tasks)[None, :]).astype(float)
    period_design = (periods[:, None] == np.unique(periods)[None, :]).astype(float)
    sse = {}
    for key, design in (("task", task_design), ("period", period_design),
                        ("joint", np.column_stack((task_design, period_design)))):
        u, singular, _ = np.linalg.svd(design, full_matrices=False)
        rank = int(np.sum(singular > singular[0] * max(design.shape) * np.finfo(float).eps))
        basis = u[:, :rank]
        residual = centered - basis @ (basis.T @ centered)
        sse[key] = float(np.sum(residual ** 2))
    tolerance = 1e-10 * total
    if (not np.isfinite([total, *sse.values()]).all()
            or any(sse[key] > total + tolerance or sse["joint"] > sse[key] + tolerance
                   for key in ("task", "period"))):
        raise ValueError("Variance partition has inconsistent nested-model sums of squares")
    # Clip only floating-point subtraction error; genuine nesting violations
    # above are errors. No threshold is applied to small nonzero effects.
    sse = {key: min(value, total) for key, value in sse.items()}
    sse["joint"] = min(sse.values())
    ss = {"task": ((total - sse["task"]) + (sse["period"] - sse["joint"])) / 2,
          "period": ((total - sse["period"]) + (sse["task"] - sse["joint"])) / 2,
          "residual": sse["joint"], "total": total}
    if not np.isclose(sum(ss[key] for key in VARIANCE_COMPONENTS), ss["total"],
                      rtol=1e-10, atol=0):
        raise ValueError("Variance partition sums of squares do not reconstruct the total")
    fractions = {key: ss[key] / ss["total"] if ss["total"] > 0 else float("nan")
                 for key in VARIANCE_COMPONENTS}
    return {"ss": ss, "fractions": fractions, "n_features": values.shape[1],
            "model_ss": {key: total - value for key, value in sse.items()}}


def _partition_features(aname, data, matched=True):
    """Align all observed or the fixed 25 rows, then use the same decomposition."""
    expected = ([f"{task}-{phase}" for task in MATCHED_TASKS for phase in MATCHED_PHASES]
                if matched else None)
    partitions = {}
    for rep in REPRESENTATIONS:
        rows = data.get(rep, {})
        names = np.asarray(rows.get("tb_break_name", []), dtype=str)
        if names.ndim != 1 or np.unique(names).size != names.size:
            raise ValueError(f"{aname}/{rep}: invalid or duplicate feature row names")
        if expected is None:
            expected = sorted(names.tolist())
        if matched:
            missing = sorted(set(expected) - set(names))
            if missing:
                raise MissingMatchedTaskRows(f"{aname}/{rep}: missing feature rows {missing}")
        elif sorted(names.tolist()) != expected:
            raise ValueError(f"{aname}/{rep}: inconsistent all-task feature rows")
        if "variance_features" not in rows:
            raise MissingVarianceFeatures(f"{aname}/{rep}: missing normalized variance features")
        features = np.asarray(rows["variance_features"])
        if (features.ndim != 2 or features.shape[0] != names.size
                or features.shape[1] == 0 or not np.isfinite(features).all()):
            raise ValueError(f"{aname}/{rep}: invalid normalized variance features")
        lookup = {name: i for i, name in enumerate(names)}
        tasks, periods = _split_row_names(expected)
        if not np.all(tasks != "") or not np.all(periods != ""):
            raise ValueError(f"{aname}/{rep}: malformed task-period feature row names")
        partitions[rep] = partition_task_period_variance(
            features[[lookup[name] for name in expected]], tasks, periods)
        partitions[rep]["row_names"] = expected
    return partitions


def summarize_variance_partitions(runs, matched=True):
    """Pack per-network partitions without retaining large feature matrices."""
    if not runs or len({name for name, _ in runs}) != len(runs):
        raise ValueError("Variance partition requires a nonempty, unique network cohort")
    row_names = runs[0][1][REPRESENTATIONS[0]]["row_names"]
    reps = {}
    for rep in REPRESENTATIONS:
        entries = [data[rep] for _, data in runs]
        if any(entry["row_names"] != row_names for entry in entries):
            raise ValueError("Variance partition requires identical task-period rows across networks/representations")
        reps[rep] = {
            "anames": [name for name, _ in runs],
            "ss": {key: [entry["ss"][key] for entry in entries]
                   for key in (*VARIANCE_COMPONENTS, "total")},
            "fractions": {key: [entry["fractions"][key] for entry in entries]
                          for key in VARIANCE_COMPONENTS},
            "n_features": [entry["n_features"] for entry in entries],
            "n_valid": sum(entry["ss"]["total"] > 0 for entry in entries),
            "model_ss": {key: [entry["model_ss"][key] for entry in entries]
                         for key in ("task", "period", "joint")},
        }
    tasks, periods = _split_row_names(row_names)
    return {"schema_version": 2, "variance_method": diag.VARIANCE_METHOD,
            "task_names": list(MATCHED_TASKS) if matched else sorted(set(tasks)),
            "phases": list(MATCHED_PHASES) if matched else sorted(set(periods)),
            "row_names": row_names, "row_weighting": VARIANCE_ROW_WEIGHTING,
            "allocation": VARIANCE_ALLOCATION,
            "normalization": VARIANCE_NORMALIZATION, "representations": reps}


def _partition_variance_views(aname, data):
    """Compute both views while one network's feature matrices are in memory."""
    views = {}
    for key, matched in VARIANCE_VIEWS.items():
        try:
            views[key] = _partition_features(aname, data, matched=matched)
        except MissingMatchedTaskRows as exc:
            print(f"  [skip] {key}: {exc}")
    return views


def _summarize_variance_views(runs):
    """Never silently drop networks: missing views skip that whole supplement."""
    entries = {}
    for key, matched in VARIANCE_VIEWS.items():
        if all(key in views for _, views in runs):
            entries[key] = summarize_variance_partitions(
                [(name, views[key]) for name, views in runs], matched=matched)
        else:
            print(f"  [skip] {key}: unavailable for part of the cohort; retaining other summaries.")
    return entries


def _matched_task_rows(experiments):
    """One validated 5 x 5 row grid for all matched-task analyses."""
    expected = {f"{rule}-{phase}" for rule in MATCHED_TASKS for phase in MATCHED_PHASES}
    filtered = []
    for aname, data in experiments:
        rules, phases, labels = _aligned_rows(aname, data)
        names = np.array([f"{rule}-{phase}" for rule, phase in zip(rules, phases)])
        keep = np.isin(rules, MATCHED_TASKS) & np.isin(phases, MATCHED_PHASES)
        missing = expected - set(names[keep])
        if missing:
            raise MissingMatchedTaskRows(
                f"{aname}: matched-task analysis missing rows {sorted(missing)}")
        filtered.append((aname, {
            rep: {"tb_break_name": names[keep], "row_labels": values[keep]}
            for rep, values in labels.items()}))
    if not filtered:
        raise ValueError("matched-task analysis requires at least one network")
    return filtered


def _matched_task_metadata():
    return {
        "schema_version": 1, "task_names": list(MATCHED_TASKS),
        "phases": list(MATCHED_PHASES),
        "clustering": "original full-task cluster labels retained; no reclustering",
    }


def summarize_same_period_matched_tasks(experiments, n_permutations=10000,
                                        permutation_seed=0):
    """Same five tasks and ten pairs per phase, retaining full-task clustering."""
    filtered = _matched_task_rows(experiments)
    entry = summarize_same_period_all(filtered, n_permutations, permutation_seed)
    entry.update(_matched_task_metadata())
    entry.update({
        "tasks": list(MATCHED_TASKS),  # retained for existing supplement readers
        "n_pairs_per_period": len(MATCHED_TASKS) * (len(MATCHED_TASKS) - 1) // 2,
        "period_contrasts": _matched_period_contrasts(
            entry["periods"], n_permutations, permutation_seed),
    })
    entry["permutation"]["pair_selection"] = "all pairs of the fixed five tasks in each phase"
    return entry


def summarize_same_task_matched_tasks(experiments, n_permutations=10000,
                                      permutation_seed=0):
    """Five tasks, ten phase pairs each; null shuffles only these five tasks."""
    entry, _ = _sharing_permutations(
        _matched_task_rows(experiments), "task", n_permutations, permutation_seed,
        include_groups=False)
    entry.update(_matched_task_metadata())
    entry["n_pairs_per_task"] = len(MATCHED_PHASES) * (len(MATCHED_PHASES) - 1) // 2
    entry["permutation"]["pair_selection"] = "all phase pairs of each of the fixed five tasks"
    entry["permutation"]["fdr_family"] = (
        "BH within this cohort: five matched tasks plus pooled, across three representations")
    return entry


def summarize_organization_matched_tasks(experiments):
    """Task/phase enrichment against all 300 pairs of the selected 25 rows."""
    entry = summarize_organization(_matched_task_rows(experiments))
    entry.update(_matched_task_metadata())
    entry["baseline"] = "sharing among all pairs of the selected 25 rows"
    return entry


class MissingRowLinkage(ValueError):
    """An older cluster cache has labels but no saved modulation row tree."""


def _recut_modulation_trees(experiments, k_values, selected_rows):
    """Recut every network's full modulation row tree at each K of `k_values`
    and count the organization classes among the rows `selected_rows` names
    ({aname: row names}), or among every row when it is None.

    maxclust preserves tied merge heights instead of arbitrarily splitting
    identical profiles. K is an upper bound on active clusters; the actual
    full-tree count (`actual_k`) and the count among the selected rows
    (`matched_k`, equal to it when every row is selected) are saved. Missing
    sharing gives NaN, not 0. Curves are descriptive sensitivity checks, not
    independent replications.
    """
    k_values = tuple(k_values)
    if (not k_values or len(set(k_values)) != len(k_values)
            or any(not isinstance(k, (int, np.integer)) or k < 2 for k in k_values)):
        raise ValueError("k_values must contain distinct integers >= 2")
    cuts = {k: {"anames": [], "actual_k": [], "matched_k": [], "counts": {},
                "enrichment": {name: [] for name in ORGANIZATION_CLASSES},
                "task_bias": []} for k in k_values}
    for aname, data in experiments:
        rows = data["modulation"]
        if "row_linkage" not in rows:
            raise MissingRowLinkage(f"{aname}: missing modulation row_linkage")
        tree = np.asarray(rows["row_linkage"], dtype=float)
        labels = np.asarray(rows["row_labels"])
        mask = np.asarray(rows.get("row_unresponsive_mask", np.zeros(labels.size)), dtype=bool)
        if (mask.shape != labels.shape or tree.ndim != 2 or tree.shape[1] != 4
                or tree.shape[0] + 1 != int((~mask).sum())):
            raise ValueError(f"{aname}: modulation tree leaves do not align with active rows")
        if max(k_values) > tree.shape[0] + 1:
            raise ValueError(f"{aname}: requested K exceeds the number of active rows")
        # The saved tree is in original row order, not heatmap leaf order.
        # Verify that it reproduces the existing partition before using it.
        original = labels[~mask]
        recovered = fcluster(tree, len(np.unique(original)), criterion="maxclust")
        if not np.array_equal(original[:, None] == original, recovered[:, None] == recovered):
            raise ValueError(f"{aname}: modulation tree does not reproduce saved row labels")
        names = np.asarray(rows["tb_break_name"])
        keep = (np.ones(names.size, dtype=bool) if selected_rows is None
                else np.isin(names, selected_rows[aname]))
        for k, cut in cuts.items():
            recut = np.empty(labels.size, dtype=int)
            recut[~mask] = fcluster(tree, k, criterion="maxclust")
            recut[mask] = k + 1  # retain an existing held-out unresponsive class
            counts = _organization_counts({"tb_break_name": names[keep],
                                           "row_labels": recut[keep]})
            cut["anames"].append(aname)
            cut["actual_k"].append(int(np.unique(recut).size))
            cut["matched_k"].append(int(np.unique(recut[keep]).size))
            cut["counts"][aname] = counts
            for name in ORGANIZATION_CLASSES:
                cut["enrichment"][name].append(_enrichment(counts, name))
            cut["task_bias"].append(_enrichment(counts, "task") - _enrichment(counts, "period"))
    return {"representation": "modulation", "k_values": list(k_values), "cuts": cuts,
            "cut_method": "maxclust; tied heights retained; actual_k includes any unresponsive class"}


def summarize_cut_robustness_matched_tasks(experiments, k_values=CUT_ROBUSTNESS_K):
    """Recut full-task modulation trees, then select the fixed 25 rows
    (`_recut_modulation_trees`), against the baseline of all 300 pairs among them."""
    matched = _matched_task_rows(experiments)  # validate the entire fixed cohort
    selected = {aname: rows["modulation"]["tb_break_name"] for aname, rows in matched}
    return {**_matched_task_metadata(),
            "clustering": "full-task modulation row trees recut before selecting the 25 rows",
            **_recut_modulation_trees(experiments, k_values, selected),
            "baseline": "sharing among all pairs of the selected 25 rows"}


def summarize_cut_robustness_all(experiments, k_values=CUT_ROBUSTNESS_K):
    """Recut full-task modulation trees and keep every row of every task.

    The all-task twin of `summarize_cut_robustness_matched_tasks`: as in
    `summarize_organization`, the baseline is sharing among all pairs of a
    network's rows, so the periods differ in task composition. It needs no
    five-task grid, only the rows every representation shares.
    """
    for aname, data in experiments:
        _aligned_rows(aname, data)
    return {"schema_version": 1,
            "clustering": "full-task modulation row trees recut; every row of every task retained",
            **_recut_modulation_trees(experiments, k_values, None),
            "baseline": "sharing among all pairs of every row"}


class MissingClusterCounts(ValueError):
    """A cluster cache lacks the selected K of an axis (older or synthetic)."""


def summarize_cluster_counts(experiments):
    """The number of clusters the clustering selected, per network and
    representation: `row_k` over the task x period rows and `col_k` over the
    columns (neurons of input and hidden, synapses of modulation at the
    plotted pre-grouping), the silhouette-tolerance K (`row_tol_k`,
    `col_tol_k`) whose labels every row-cluster analysis reads."""
    counts = {}
    for rep in REPRESENTATIONS:
        rows = [(aname, data[rep]) for aname, data in experiments if rep in data]
        if not rows:
            continue
        missing = [aname for aname, rep_data in rows
                   if "row_k" not in rep_data or "col_k" not in rep_data]
        if missing:
            raise MissingClusterCounts(f"{rep}: no selected K saved for {missing}")
        counts[rep] = {"anames": [aname for aname, _ in rows],
                       "row_k": [int(rep_data["row_k"]) for _, rep_data in rows],
                       "col_k": [int(rep_data["col_k"]) for _, rep_data in rows]}
    return {"schema_version": 1, "representations": counts,
            "k_definition": ("silhouette-tolerance K (row_tol_k, col_tol_k) of the normalized "
                             f"clustering; modulation at result_all_lst[{diag.MOD_G_INDEX}]"),
            "column_units": {"input": "neurons", "hidden": "neurons", "modulation": "synapses"}}


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


def _print_cut_robustness(title, entry):
    """Per K, the clusters the cut formed (full tree, selected rows) and the
    cohort medians of the synapse task bias and same-period enrichment."""
    print(f"\n=== {title}: modulation row trees recut at K={entry['k_values']} ===")
    for k in entry["k_values"]:
        cut = entry["cuts"][k]
        print(f"  K={k:<3d}: clusters, full tree {min(cut['actual_k'])}-{max(cut['actual_k'])}, "
              f"selected rows {min(cut['matched_k'])}-{max(cut['matched_k'])}; median "
              f"task - period enrichment {np.nanmedian(cut['task_bias']):.2f}; median "
              f"same-period enrichment {np.nanmedian(cut['enrichment']['period']):.2f}")


def _print_cluster_counts(entry):
    """Per representation and axis, the median and range of the selected K."""
    print("\n=== Selected number of clusters (silhouette tolerance) ===")
    for rep, counts in entry["representations"].items():
        cells = "; ".join(
            f"{axis} K median {np.median(counts[f'{axis}_k']):g}, "
            f"range {min(counts[f'{axis}_k'])}-{max(counts[f'{axis}_k'])}"
            for axis in ("row", "col"))
        print(f"  {rep:<11s}: {cells} over {len(counts['anames'])} runs")


def _update_cut_robustness(tag, root, feature, seed, untrained):
    """Replace the cut supplements of a current summary without changing its cohort."""
    path = OUT_DIR / f"sibling_row_sharing_{tag}.pkl"
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    matched_metadata = payload.get("organization_matched_tasks", {})
    if (payload.get("version") != RESULT_VERSION or payload.get("seed") != seed
            or payload.get("variance_method") != diag.VARIANCE_METHOD
            or bool(payload.get("untrained")) != untrained
            or payload.get("feature") != (None if untrained else feature)
            or payload.get("architecture") != architecture_tag(feature)
            or payload.get("mod_g_index") != diag.MOD_G_INDEX
            or "representations" not in payload.get("organization", {})
            or matched_metadata.get("schema_version") != 1
            or matched_metadata.get("task_names") != list(MATCHED_TASKS)
            or matched_metadata.get("phases") != list(MATCHED_PHASES)):
        raise ValueError("Need a current matched-task summary; run without --cut-robustness-only first")
    experiments = []
    for aname in payload["experiments"]:
        data = diag._load_experiment(aname, root=root)
        if data is None or "modulation" not in data:
            raise ValueError(f"{aname}: saved cohort member is missing; summary left unchanged")
        experiments.append((aname, data))
    matched = _matched_task_rows(experiments)
    for rows, cached in ((experiments, payload["organization"]["representations"]),
                         (matched, payload["organization_matched_tasks"]["representations"])):
        for rep in REPRESENTATIONS:
            fresh = {aname: _organization_counts(data[rep]) for aname, data in rows}
            if fresh != cached[rep]["counts"]:
                raise ValueError("Cluster caches changed; recompute the full summary first")
    payload["cut_robustness_all"] = summarize_cut_robustness_all(experiments)
    payload["cut_robustness_matched_tasks"] = summarize_cut_robustness_matched_tasks(experiments)
    _save_payload(payload, tag)
    print(f"Added cut robustness (all tasks and matched tasks) for the existing "
          f"{len(experiments)} networks; all previous statistics retained.")
    return payload


def _save_payload(payload, tag):
    pkl_path = OUT_DIR / f"sibling_row_sharing_{tag}.pkl"
    json_path = OUT_DIR / f"sibling_row_sharing_{tag}.json"
    with open(pkl_path, "wb") as stream:
        pickle.dump(payload, stream)
    with open(json_path, "w") as stream:
        json.dump(diag._json_safe(payload), stream, indent=2)
    print(f"\nSaved: {pkl_path}")
    print(f"Saved: {json_path}")


def _update_variance_partition(tag, root, feature, seed, untrained):
    """Update features for exactly the saved cohort; validate before writing anything."""
    path = OUT_DIR / f"sibling_row_sharing_{tag}.pkl"
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if (payload.get("version") != RESULT_VERSION or payload.get("seed") != seed
            or payload.get("variance_method") != diag.VARIANCE_METHOD
            or bool(payload.get("untrained")) != untrained
            or payload.get("feature") != (None if untrained else feature)
            or payload.get("architecture") != (architecture_tag(feature) if feature else None)
            or payload.get("mod_g_index") != diag.MOD_G_INDEX
            or "representations" not in payload.get("organization", {})):
        raise ValueError("Need a current sharing summary; run without --variance-partition-only first")
    runs = []
    for aname in payload["experiments"]:
        data = diag._load_experiment(aname, root=root, include_features=True)
        if data is None or any(rep not in data for rep in REPRESENTATIONS):
            raise ValueError(f"{aname}: saved cohort member is missing; summary left unchanged")
        for rep in REPRESENTATIONS:
            cached = payload["organization"]["representations"][rep]["counts"][aname]
            if _organization_counts(data[rep]) != cached:
                raise ValueError("Cluster caches changed; recompute the full sharing summary first")
        runs.append((aname, _partition_variance_views(aname, data)))
        del data
    entries = _summarize_variance_views(runs)
    for key in VARIANCE_VIEWS:
        payload.pop(key, None)
    payload.update(entries)
    _save_payload(payload, tag)
    print(f"Updated {len(entries)} variance-partition views for the existing {len(runs)} networks; "
          "all other statistics retained.")
    return payload


def main(feature="L21e4", seed=None, untrained=False,
         n_permutations=10000, permutation_seed=0, cut_robustness_only=False,
         variance_partition_only=False):
    """Analyze every clustered run of `feature` (or one `seed`), or with
    `untrained` every untrained control of `feature`'s architecture (module
    docstring), save the results and draw the three diagnostic figures.
    With `cut_robustness_only`, append cuts to the saved cohort instead.
    With `variance_partition_only`, update its all-task and matched feature decompositions.
    """
    if cut_robustness_only and variance_partition_only:
        raise ValueError("Choose only one supplement-only update")
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
    if cut_robustness_only:
        return _update_cut_robustness(tag, root, feature, seed, untrained)
    if variance_partition_only:
        return _update_variance_partition(tag, root, feature, seed, untrained)
    if untrained:
        anames = diag._discover_anames(seed=seed, root=root,
                                       architecture=architecture_tag(feature))
    else:
        anames = diag._discover_anames(feature, seed)

    print(f"Discovered {len(anames)} {'untrained ' if untrained else ''}experiment(s) "
          f"(feature={feature}, seed={seed})")
    experiments = []
    variance_runs = []
    for aname in anames:
        data = diag._load_experiment(aname, root=root, include_features=True)
        if data is not None and "modulation" not in data:
            print(f"  [skip] {aname}: clustering incomplete (modulation required)")
            continue
        if data is not None:
            try:
                variance_runs.append((aname, _partition_variance_views(aname, data)))
            except MissingVarianceFeatures as exc:
                variance_runs.append((aname, {}))
                print(f"  [skip] variance partition: {exc}")
            finally:
                for rows in data.values():
                    rows.pop("variance_features", None)
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
    optional_results = _summarize_variance_views(variance_runs)
    try:
        optional_results["cluster_counts"] = summarize_cluster_counts(experiments)
        _print_cluster_counts(optional_results["cluster_counts"])
    except MissingClusterCounts as exc:
        print(f"  [skip] cluster counts: {exc}.")
    try:
        optional_results["cut_robustness_all"] = summarize_cut_robustness_all(experiments)
        _print_cut_robustness("All tasks", optional_results["cut_robustness_all"])
    except MissingRowLinkage as exc:
        print(f"  [skip] cut robustness: {exc}; retaining the other summaries.")

    matched_results = {}
    try:
        # Build together so a missing grid cannot leave a partial supplement
        # or silently change which networks enter the matched comparisons.
        matched_results = {
            "same_period_matched_tasks": summarize_same_period_matched_tasks(
                experiments, n_permutations, permutation_seed),
            "same_task_matched_tasks": summarize_same_task_matched_tasks(
                experiments, n_permutations, permutation_seed),
            "organization_matched_tasks": summarize_organization_matched_tasks(experiments),
        }
    except MissingMatchedTaskRows as exc:
        print(f"  [skip] matched-task supplements: {exc}; "
              "saving all-task results without matched-task supplements.")
    if matched_results:
        same_period_matched_tasks = matched_results["same_period_matched_tasks"]
        print("\nMatched tasks: " + ", ".join(MATCHED_TASKS))
        _print_same_period(same_period_matched_tasks)
        print("\n=== Paired period contrasts (percentage points; separate six-test BH family) ===")
        for contrast, by_rep in same_period_matched_tasks["period_contrasts"]["contrasts"].items():
            for rep, stats in by_rep.items():
                print(f"  {contrast} {rep}: {stats['mean_difference_pp']:+.2f} pp; "
                      f"n={stats['n_runs']}; p={stats['p_perm']:.4g}, q={stats['q_fdr']:.4g}")
        _print_same_task({"Matched five tasks": matched_results["same_task_matched_tasks"]})
        print("\nMatched five tasks: organization against the selected 25-row baseline")
        _print_organization(matched_results["organization_matched_tasks"])
        if "cut_robustness_all" in optional_results:  # every network saved its row tree
            matched_results["cut_robustness_matched_tasks"] = (
                summarize_cut_robustness_matched_tasks(experiments))
            _print_cut_robustness("Matched five tasks",
                                  matched_results["cut_robustness_matched_tasks"])

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
        "variance_method": diag.VARIANCE_METHOD,
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
        **optional_results,
        **matched_results,
    }
    _save_payload(payload, tag)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Row-cluster sharing: restricted same-task/same-period "
                    "permutation tests, organization, task/period variance partition "
                    "and legacy sibling-pair diagnostics.")
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
    update = parser.add_mutually_exclusive_group()
    update.add_argument("--cut-robustness-only", action="store_true",
                        help="Recompute both cut checks (all tasks, matched tasks) for the "
                             "existing summary's exact saved cohort; leave previous "
                             "statistics unchanged.")
    update.add_argument("--variance-partition-only", action="store_true",
                        help="Update all-task and matched task/period variance fractions "
                             "for the saved cohort, without rerunning sharing permutations.")
    args = parser.parse_args()
    main(feature=args.feature, seed=args.seed, untrained=args.untrained,
         n_permutations=args.n_permutations, permutation_seed=args.permutation_seed,
         cut_robustness_only=args.cut_robustness_only,
         variance_partition_only=args.variance_partition_only)
