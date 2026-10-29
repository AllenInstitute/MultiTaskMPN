"""
Cohort statistics of the sibling-task fixed-point geometry.

The sibling figures of paper_plot.py (``delaydm_fixed_point_geometry``,
``sibling_cue_interp``) are illustrations: each shows the run of the cohort,
and the pair of its six saved PCs, in which the fixed points it draws
separate best. This script asks what holds over every run, in the full state
space, from the settled long-delay endpoint and rule-cue sweep caches that
``sibling_delay_analysis.py --method long_delay_endpoint`` writes under
``two_in_multiples/<aname>/`` (nothing is re-simulated; a cohort takes a few
minutes, most of it reading the W * M sweeps). Per run, family (DelayDM:
delaydm1 / delaydm2; DMC: dmcgo / dmcnogo) and representation (the hidden
state and W * M at the settled endpoint, ``REPRESENTATIONS``):

DelayDM, whose two rules present the stimulus on different modalities:
- ``translation_explained``: the two tasks' endpoints matched by (stimulus
  direction, magnitude); the fraction of the mean squared cross-task
  displacement that one shared vector accounts for (1: the two tasks store
  the stimulus at places that differ by a pure translation;
  ``core.sibling_geometry.translation_metrics``, with ``geometry_r``).
- ``alignment_gain``: the separation of the pooled endpoints by stimulus
  direction (``separation_score``) with each task's mean removed, divided by
  the raw separation; above 1 when the same direction clusters across tasks
  once the translation is gone.
- ``cue_traversed``: along the rule-cue sweep from a base rule's own cue
  (alpha = 0) to the sibling's (alpha = 1), the displacement of each trial's
  settled endpoint projected onto the task translation, as a fraction of it
  (median over the trials of both base rules; 1: the cue alone carries the
  state the whole way).
- ``cue_end_residual``: the distance of the alpha = 1 endpoint from the
  sibling task's endpoint of the same stimulus condition, as a fraction of
  the task translation's length (median; 0: it lands on it), and
  ``cue_end_nearer_sibling``, the fraction of trials whose alpha = 1 endpoint
  is nearer that sibling endpoint than the trial's own.

DMC, whose two rules share the stimulus and differ in which category to
respond to:
- ``category_axis_cosine``: the cosine of the two tasks' stimulus-category
  axes (category 1, the locations below pi, minus category 2), 1 for one
  shared category code, -1 when the code reverses between the tasks (a
  "respond to" code), 0 when each task codes the category along its own
  direction (``core.sibling_geometry.category_axis_cosine``). The
  separations by task, and by response and by stimulus category with the
  task means removed, are saved beside it.
- ``cue_end_residual``: the distance of a DMCGo trial's alpha = 1 endpoint
  (its stimulus under DMCNoGo's cue, a trained input) from the DMCNoGo
  endpoints' centroid of the same stimulus category, as a fraction of the
  distance between the two tasks' centroids of that category (median; 0: it
  lands on the sibling's endpoint), and ``cue_end_nearer_same``, the
  fraction of trials whose alpha = 1 endpoint is nearer that centroid than
  the other category's.
- ``category_separation_<rule>`` (``dmcgo``, ``dmcnogo``): within one rule,
  the separation of its endpoints by stimulus category (``separation_score``,
  full space), the memory of the category itself. Beside it,
  ``category_best_other_split_<rule>`` is the largest separation any of the
  other 34 balanced four-against-four splits of the eight stimulus
  directions gives, ``category_is_best_split_<rule>`` whether the category
  split beats them all (the endpoints group by the category boundary, not
  by some other half of the ring), and
  ``category_nearest_centroid_accuracy_<rule>`` the leave-one-out
  nearest-centroid accuracy of the category (0.5 chance, 1 perfect).

A run enters when both families' endpoint and sweep caches exist and every
point met the fixed-point criterion (``is_fixed``); other runs are listed
with the reason. Output: ``two_in_multiples/sibling_cohort_results.pkl``
(the feature, the definitions above, ``runs`` keyed by run name with every
statistic per family and representation, and ``skipped``), which
``paper_plot.py`` draws in ``two_in_multiple`` mode, one statistic per figure
(``sibling_cohort_translation``, ``sibling_cohort_alignment``,
``sibling_cohort_cue_transfer``, ``sibling_cohort_category_axis``,
``sibling_cohort_category_memory``). Rerun it after
``sibling_delay_analysis.py`` rewrites any run's caches.

    python multiple_task/sibling_cohort.py --feature L21e4
"""
import argparse
from itertools import combinations
from pathlib import Path
import pickle
import re

import numpy as np

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from sibling_geometry import (category_axis_cosine, remove_task_offsets, separation_score,
                              translation_metrics)

SIBLING_DIR = Path("two_in_multiples")
RESULT_NAME = "sibling_cohort_results.pkl"
# 2: the within-rule DMC category statistics (category_*_<rule>) were added.
SCHEMA_VERSION = 2
DEFAULT_FEATURE = "L21e4"
RUN_PATTERN = re.compile(
    r"everything_seed(?P<seed>\d+)_(?P<feature>[^+]+)\+hidden\d+\+batch\d+\+angle")
# Representation name (as paper_plot's sibling figures call it) -> key of the
# settled states in the endpoint and sweep caches.
REPRESENTATIONS = {"hidden": "fixed_hidden", "e_modulation": "fixed_WM"}
FAMILIES = {"delaydm1": ("delaydm1", "delaydm2"), "dmcgo": ("dmcgo", "dmcnogo")}
# The task generator's stimulus directions; DMC category 1 is the lower half
# (sibling_delay_analysis._dmc_category_labels).
N_STIM = 8
MAGNITUDE_DECIMALS = 6
# One-line definitions of the statistics the paper figures draw, saved with
# the results; the others saved beside them are described in the module
# docstring.
DEFINITIONS = {
    "translation_explained": "fraction of the mean squared cross-task displacement of matched "
                             "DelayDM endpoints explained by one shared translation",
    "alignment_gain": "DelayDM direction separation with task means removed / raw, full space",
    "cue_traversed": "median fraction of the DelayDM task translation traversed by the "
                     "rule-cue sweep from alpha = 0 to 1",
    "cue_end_residual": "median distance of the alpha = 1 endpoint from the sibling task's "
                        "endpoint of the same stimulus, relative to the task distance "
                        "(DelayDM: the matched endpoint and the translation length; DMC: "
                        "the DMCNoGo centroid of the stimulus category and the Go-NoGo "
                        "centroid distance of that category)",
    "category_axis_cosine": "cosine of the two DMC tasks' stimulus-category axes, full space "
                            "(1 shared code, -1 reversed code, 0 orthogonal)",
    "category_separation": "within one DMC rule, separation of its endpoints by stimulus "
                           "category, full space",
    "category_best_other_split": "largest separation over the other balanced four-against-four "
                                 "splits of the eight stimulus directions",
    "category_is_best_split": "the category split separates better than every other balanced "
                              "split",
    "category_nearest_centroid_accuracy": "leave-one-out nearest-centroid accuracy of the "
                                          "category within one rule",
}


def endpoint_path(run_dir, family, aname):
    return Path(run_dir) / f"{family}_long_delay_endpoints_{aname}.pkl"


def sweep_path(run_dir, family, aname):
    return Path(run_dir) / f"{family}_cue_interp_long_delay_endpoints_{aname}.pkl"


def _load(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"no {path.name}")
    with path.open("rb") as stream:
        return pickle.load(stream)


def _settled(mask, context):
    """Raise unless `mask`, the saved fixed-point criterion, exists and holds."""
    if mask is None:
        raise ValueError(f"{context} predate settling (no is_fixed mask); rerun "
                         "multiple_task/sibling_delay_analysis.py")
    mask = np.asarray(mask, dtype=bool)
    if not mask.all():
        raise ValueError(f"{context}: {int((~mask).sum())}/{mask.size} points above the "
                         "fixed-point criterion")


def _states(record, rep_key, context):
    """The settled states of one representation as (n, dim) rows (the sweep
    caches hold (alpha, trial, ...), returned as (alpha, trial, dim))."""
    values = record.get("representations", {}).get(rep_key)
    if values is None:
        raise ValueError(f"{context}: no {rep_key} states")
    values = np.asarray(values, dtype=float)
    lead = 2 if values.ndim > 2 and record.get("is_fixed") is not None \
        and np.ndim(record["is_fixed"]) == 2 else 1
    return values.reshape(*values.shape[:lead], -1)


def _condition_keys(stim_idx, magnitude):
    return list(zip(np.asarray(stim_idx, dtype=int).tolist(),
                    np.round(np.asarray(magnitude, dtype=float), MAGNITUDE_DECIMALS).tolist()))


def matched_conditions(endpoint, context):
    """(rows of task 0, rows of task 1, stimulus directions) of the DelayDM
    endpoints matched by (direction, magnitude): both tasks hold every
    condition once."""
    task_idx = np.asarray(endpoint["task_idx"], dtype=int)
    keys = _condition_keys(endpoint["stim_idx"], endpoint["stimulus_magnitude"])
    by_task = {}
    for task in (0, 1):
        rows = np.flatnonzero(task_idx == task)
        lookup = {keys[row]: int(row) for row in rows}
        if len(lookup) != rows.size:
            raise ValueError(f"{context}: task {task} repeats a (direction, magnitude) condition")
        by_task[task] = lookup
    if set(by_task[0]) != set(by_task[1]) or len(by_task[0]) < 3:
        raise ValueError(f"{context}: the two tasks' (direction, magnitude) conditions differ")
    common = sorted(by_task[0])
    return (np.array([by_task[0][key] for key in common]),
            np.array([by_task[1][key] for key in common]),
            np.array([key[0] for key in common]), common)


def delaydm_statistics(endpoint, sweep_cache, rules, rep_key):
    """The DelayDM statistics of one representation (module docstring)."""
    context = f"{rules[0]} {rep_key}"
    _settled(endpoint.get("is_fixed"), f"{context} endpoints")
    points = _states(endpoint, rep_key, context)
    first_rows, second_rows, stim, keys = matched_conditions(endpoint, context)
    first, second = points[first_rows], points[second_rows]
    metrics = translation_metrics(first, second)
    translation = metrics["translation"]
    length = float(np.linalg.norm(translation))
    if length == 0:
        raise ValueError(f"{context}: the two tasks' endpoints coincide")
    pooled = np.concatenate([first, second])
    task_idx = np.repeat([0, 1], first.shape[0])
    stim_pooled = np.concatenate([stim, stim])
    separation_raw = separation_score(pooled, stim_pooled)
    separation_aligned = separation_score(remove_task_offsets(pooled, task_idx), stim_pooled)

    index = {key: position for position, key in enumerate(keys)}
    traversed, residual, nearer = [], [], []
    for base_rule, sweep in sweep_cache["sweeps"].items():
        _settled(sweep.get("is_fixed"), f"{context} sweep from {base_rule}")
        path = _states(sweep, rep_key, f"{context} sweep from {base_rule}")
        if path.ndim != 3 or path.shape[0] < 2:
            raise ValueError(f"{context}: the sweep from {base_rule} needs two or more alphas")
        base_task = list(rules).index(base_rule)
        own, sibling = (first, second) if base_task == 0 else (second, first)
        shared = translation if base_task == 0 else -translation
        trial_keys = _condition_keys(sweep["stim_idx"], sweep["stimulus_magnitude"])
        rows = [index[key] for key in trial_keys if key in index]
        if len(rows) != path.shape[1]:
            raise ValueError(f"{context}: a sweep trial from {base_rule} has no matched endpoint")
        traversed.extend(((path[-1] - path[0]) @ shared) / length ** 2)
        to_sibling = np.linalg.norm(path[-1] - sibling[rows], axis=1)
        to_own = np.linalg.norm(path[-1] - own[rows], axis=1)
        residual.extend(to_sibling / length)
        nearer.extend(to_sibling < to_own)
    if not traversed:
        raise ValueError(f"{context}: no rule-cue sweep")
    return {
        "translation_explained": float(metrics["translation_explained"]),
        "geometry_r": float(metrics["geometry_r"]),
        "translation_length": length,
        "separation_raw": float(separation_raw),
        "separation_aligned": float(separation_aligned),
        "alignment_gain": float(separation_aligned / separation_raw),
        "cue_traversed": float(np.median(traversed)),
        "cue_end_residual": float(np.median(residual)),
        "cue_end_nearer_sibling": float(np.mean(nearer)),
        "n_conditions": int(first.shape[0]),
        "n_sweep_trials": len(traversed),
    }


def dmc_category(stim_idx):
    """1 for the stimulus locations below pi (category 1), 0 above."""
    stim_idx = np.asarray(stim_idx, dtype=int)
    half = max(N_STIM, int(stim_idx.max()) + 1) // 2
    return (stim_idx < half).astype(int)


def balanced_splits(n_stim=N_STIM):
    """Every split of `n_stim` stimulus directions into two groups of equal
    size, each as the frozenset of directions in the group holding direction
    0, so no split is listed twice; the category split (directions below
    n_stim / 2, `dmc_category`) comes first."""
    half = n_stim // 2
    category = frozenset(range(half))
    others = [frozenset((0,) + rest) for rest in combinations(range(1, n_stim), half - 1)]
    return [category] + [split for split in others if split != category]


def nearest_centroid_accuracy(points, labels):
    """Leave-one-out nearest-centroid accuracy of a two-class labeling: every
    point against the two class centroids computed without it."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError("nearest-centroid accuracy needs two classes")
    sums = {label: points[labels == label].sum(axis=0) for label in classes}
    counts = {label: int(np.sum(labels == label)) for label in classes}
    if min(counts.values()) < 2:
        raise ValueError("each class needs at least two points")
    correct = 0
    for point, label in zip(points, labels):
        other = classes[1] if label == classes[0] else classes[0]
        own_centroid = (sums[label] - point) / (counts[label] - 1)
        other_centroid = sums[other] / counts[other]
        correct += np.linalg.norm(point - own_centroid) < np.linalg.norm(point - other_centroid)
    return correct / points.shape[0]


def category_memory_statistics(points, stim_idx):
    """The within-rule category statistics of one rule's endpoints (module
    docstring): the separation by category, the best of the other balanced
    splits, whether the category split is the best, and the nearest-centroid
    accuracy. A split that leaves a group below two points is skipped; the
    category split itself must be scorable."""
    points = np.asarray(points, dtype=float)
    stim_idx = np.asarray(stim_idx, dtype=int)
    separations = []
    for index, split in enumerate(balanced_splits(max(N_STIM, int(stim_idx.max()) + 1))):
        labels = np.isin(stim_idx, list(split)).astype(int)
        try:
            separations.append(separation_score(points, labels))
        except ValueError:
            if index == 0:
                raise
            separations.append(np.nan)
    category, others = separations[0], np.array(separations[1:], dtype=float)
    best_other = float(np.nanmax(others)) if np.isfinite(others).any() else np.nan
    return {
        "category_separation": float(category),
        "category_best_other_split": best_other,
        "category_is_best_split": bool(np.isfinite(best_other) and category > best_other),
        "category_nearest_centroid_accuracy": float(
            nearest_centroid_accuracy(points, dmc_category(stim_idx))),
    }


def dmc_statistics(endpoint, sweep_cache, rules, rep_key):
    """The DMC statistics of one representation (module docstring)."""
    context = f"{rules[0]} {rep_key}"
    _settled(endpoint.get("is_fixed"), f"{context} endpoints")
    points = _states(endpoint, rep_key, context)
    task_idx = np.asarray(endpoint["task_idx"], dtype=int)
    category = dmc_category(endpoint["stim_idx"])
    response = np.asarray(endpoint["group_labels"], dtype=int)
    # The saved response label is the category under the base rule (task 0)
    # and the other category under its sibling; anything else is another
    # labeling than the one documented here.
    if not np.array_equal(response, np.where(task_idx == 0, category, 1 - category)):
        raise ValueError(f"{context}: group_labels are not the response categories")
    aligned = remove_task_offsets(points, task_idx)
    centroids = {task: {label: points[(task_idx == task) & (category == label)].mean(axis=0)
                        for label in (0, 1)} for task in (0, 1)}
    residual, nearer = [], []
    for base_rule, sweep in sweep_cache["sweeps"].items():
        _settled(sweep.get("is_fixed"), f"{context} sweep from {base_rule}")
        path = _states(sweep, rep_key, f"{context} sweep from {base_rule}")
        if path.ndim != 3 or path.shape[0] < 2:
            raise ValueError(f"{context}: the sweep from {base_rule} needs two or more alphas")
        base_task = list(rules).index(base_rule)
        target_task = 1 - base_task
        trial_category = dmc_category(sweep["stim_idx"])
        same = np.stack([centroids[target_task][label] for label in trial_category])
        other = np.stack([centroids[target_task][1 - label] for label in trial_category])
        gap = np.array([np.linalg.norm(centroids[base_task][label] - centroids[target_task][label])
                        for label in trial_category])
        to_same = np.linalg.norm(path[-1] - same, axis=1)
        residual.extend(to_same / gap)
        nearer.extend(to_same < np.linalg.norm(path[-1] - other, axis=1))
    if not residual:
        raise ValueError(f"{context}: no rule-cue sweep")
    memory = {}
    for task, rule in enumerate(rules):
        rows = task_idx == task
        statistics = category_memory_statistics(points[rows],
                                                np.asarray(endpoint["stim_idx"])[rows])
        memory.update({f"{key}_{rule}": value for key, value in statistics.items()})
    return {
        "category_axis_cosine": float(category_axis_cosine(points, task_idx, category)),
        "separation_task": float(separation_score(points, task_idx)),
        "separation_response_aligned": float(separation_score(aligned, response)),
        "separation_stimulus_aligned": float(separation_score(aligned, category)),
        "cue_end_residual": float(np.median(residual)),
        "cue_end_nearer_same": float(np.mean(nearer)),
        **memory,
        "n_endpoints": int(points.shape[0]),
        "n_sweep_trials": len(residual),
    }


STATISTICS = {"delaydm1": delaydm_statistics, "dmcgo": dmc_statistics}


def analyze_run(run_dir, aname):
    """{family: {representation: statistics}} of one cached run, or
    ValueError naming the first cache that is missing or unsettled."""
    families = {}
    for family, rules in FAMILIES.items():
        endpoint = _load(endpoint_path(run_dir, family, aname))
        sweep_cache = _load(sweep_path(run_dir, family, aname))
        if sweep_cache.get("kind") != "cue_interp" or not sweep_cache.get("sweeps"):
            raise ValueError(f"{sweep_path(run_dir, family, aname).name} is not a rule-cue sweep")
        families[family] = {name: STATISTICS[family](endpoint, sweep_cache, rules, rep_key)
                            for name, rep_key in REPRESENTATIONS.items()}
    return families


def cohort_run_dirs(sibling_dir, feature):
    """[(run directory, seed)] of the cached runs of one feature, by seed."""
    runs = []
    for run_dir in Path(sibling_dir).iterdir():
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if run_dir.is_dir() and match and match.group("feature") == feature:
            runs.append((run_dir, int(match.group("seed"))))
    return sorted(runs, key=lambda item: item[1])


def _iqr_text(values):
    values = np.asarray(values, dtype=float)
    return (f"{np.median(values):.2f} [{np.percentile(values, 25):.2f}, "
            f"{np.percentile(values, 75):.2f}]")


def print_summary(results):
    """Per family and representation the median and quartiles of each
    statistic over the runs, with the counts the figures report."""
    runs = list(results["runs"].values())
    print("=" * 78)
    print(f"Sibling-task cohort statistics, {results['feature']}: {len(runs)} runs, "
          f"{len(results['skipped'])} skipped")
    print("=" * 78)
    for family, rules in FAMILIES.items():
        print(f"\n{rules[0]} / {rules[1]}")
        for name in REPRESENTATIONS:
            entries = [entry["families"][family][name] for entry in runs]
            print(f"  {name}:")
            for key in entries[0]:
                values = np.array([entry[key] for entry in entries], dtype=float)
                if key.startswith("n_"):
                    continue
                if key.startswith("category_is_best_split"):
                    print(f"    {key:44s} {int(values.sum())}/{values.size} runs")
                    continue
                note = ""
                if key == "alignment_gain" or key.startswith("category_separation"):
                    note = f"; {int(np.sum(values > 1))}/{values.size} runs above 1"
                elif key == "category_axis_cosine":
                    note = f"; {int(np.sum(values < 0))}/{values.size} runs below 0"
                print(f"    {key:44s} {_iqr_text(values)}{note}")


def run_all(sibling_dir=SIBLING_DIR, feature=DEFAULT_FEATURE, out_dir=None):
    """Analyze every cached run of `feature`, write the results next to the
    caches (or under `out_dir`) and return them."""
    sibling_dir = Path(sibling_dir)
    out_dir = sibling_dir if out_dir is None else Path(out_dir)
    candidates = cohort_run_dirs(sibling_dir, feature)
    if not candidates:
        raise FileNotFoundError(
            f"No {feature} run under {sibling_dir}/; run python multiple_task/"
            f"sibling_delay_analysis.py --seed <seed> --feature {feature} "
            "--families delaydm1 dmcgo --method long_delay_endpoint first.")
    results = {"schema_version": SCHEMA_VERSION, "feature": feature,
               "representations": list(REPRESENTATIONS), "families": dict(FAMILIES),
               "definitions": dict(DEFINITIONS), "runs": {}, "skipped": []}
    for run_dir, seed in candidates:
        aname = run_dir.name
        try:
            families = analyze_run(run_dir, aname)
        except (KeyError, ValueError, OSError, EOFError, pickle.UnpicklingError) as error:
            results["skipped"].append({"aname": aname, "seed": seed, "reason": str(error)})
            print(f"{aname}: skipped ({error})")
            continue
        results["runs"][aname] = {"aname": aname, "seed": seed, "families": families}
        delaydm = families["delaydm1"]["e_modulation"]
        dmc = families["dmcgo"]["e_modulation"]
        print(f"{aname}: W*M translation explained {delaydm['translation_explained']:.2f}, "
              f"cue traversed {delaydm['cue_traversed']:.2f}; DMC category-axis cosine "
              f"{dmc['category_axis_cosine']:+.2f}, cue end residual {dmc['cue_end_residual']:.2f}")
    if not results["runs"]:
        raise FileNotFoundError(f"No {feature} run under {sibling_dir}/ has settled endpoint and "
                                "rule-cue sweep caches of both families.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / RESULT_NAME
    with out_path.open("wb") as stream:
        pickle.dump(results, stream)
    print(f"Saved {out_path}")
    print_summary(results)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cohort statistics of the sibling-task fixed-point geometry, in the full "
                    "state space, from the settled long-delay endpoint and rule-cue sweep caches.")
    parser.add_argument("--feature", default=DEFAULT_FEATURE,
                        help=f"Feature tag of the cohort (default {DEFAULT_FEATURE}).")
    parser.add_argument("--sibling-dir", default=str(SIBLING_DIR),
                        help=f"Directory of the per-run caches (default {SIBLING_DIR}).")
    parser.add_argument("--out-dir", default=None,
                        help=f"Where to write {RESULT_NAME} (default: the cache directory).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_all(args.sibling_dir, args.feature, args.out_dir)


if __name__ == "__main__":
    main()
