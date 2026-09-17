"""Shared metadata and lightweight helpers for pretraining analyses.

This module deliberately contains no model, plotting, or analysis code.  It
defines only the experiment contract shared by ``pretraining_analysis.py`` and
``pretraining_post.py``: ruleset metadata, run naming, deterministic seed
selection, saved task layout, and task-parameter loading.
"""

import argparse
import copy

import numpy as np

FINAL_TASK = "delayanti"

# This catalog contains the rulesets currently supported by both analysis
# drivers.  A script's presentation/iteration order remains local so sharing
# metadata cannot silently reorder figure panels.
RULESET_SPECS = {
    "fdgo_delaygo": {
        "label": "Improper motif",
        "stage1_tasks": ("fdgo", "delaygo"),
        "stage2_tasks": (FINAL_TASK,),
    },
    "fdanti_delaygo": {
        "label": "Proper motif",
        "stage1_tasks": ("fdanti", "delaygo"),
        "stage2_tasks": (FINAL_TASK,),
    },
    "fdanti": {
        "label": "DelayAnti",
        "stage1_tasks": ("fdanti",),
        "stage2_tasks": (FINAL_TASK,),
    },
}

RULE_DISPLAY_NAMES = {
    "fdgo": "DelayPro",
    "fdanti": "DelayAnti",
    "delaygo": "MemoryPro",
    "delayanti": "MemoryAnti",
}


def display_rule(rule):
    """Figure display name for an internal rule name."""
    return RULE_DISPLAY_NAMES.get(rule, rule)


def positive_int(value):
    """Argparse type requiring a strictly positive integer."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def stage1_tasks_for(ruleset):
    """Ordered Stage-1 tasks for a supported analysis ruleset."""
    try:
        return list(RULESET_SPECS[ruleset]["stage1_tasks"])
    except KeyError as exc:
        raise ValueError(f"Unknown pretraining ruleset: {ruleset!r}") from exc


def stage2_tasks_for(ruleset):
    """Ordered Stage-2 tasks for a supported analysis ruleset."""
    try:
        return list(RULESET_SPECS[ruleset]["stage2_tasks"])
    except KeyError as exc:
        raise ValueError(f"Unknown pretraining ruleset: {ruleset!r}") from exc


def variant_addon(hidden, feature, batch=128, metric="angle"):
    """Filename suffix used by the pretraining producer and consumers."""
    return f"+hidden{hidden}+{feature}+batch{batch}+{metric}"


def run_name(ruleset, network, seed, hidden, feature, batch=128,
             metric="angle"):
    """Canonical run identifier without an artifact-specific prefix/suffix."""
    addon = variant_addon(hidden, feature, batch=batch, metric=metric)
    return f"{ruleset}_{network}_seed{seed}_{addon}"


def selection_rng(test_seed, ruleset):
    """Stable per-ruleset RNG, independent of ruleset traversal order."""
    entropy = [int(test_seed), *ruleset.encode("utf-8")]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def select_seeds(seeds, total_seed, test_seed, ruleset):
    """Return all sorted seeds or a stable random subset of size total_seed."""
    seeds = sorted(int(seed) for seed in seeds)
    if total_seed is None:
        return seeds
    if len(seeds) < total_seed:
        raise ValueError(
            f"Requested --total-seed {total_seed}, but only {len(seeds)} "
            f"matching {ruleset} seed(s) exist"
        )
    rng = selection_rng(test_seed, ruleset)
    indices = rng.choice(len(seeds), size=total_seed, replace=False)
    return sorted(seeds[int(index)] for index in indices)


def build_task_layout(stage1, stage2, ruleset, input_width=None, context=None):
    """Validate saved stage metadata and return the final cue-column layout."""
    prefix = f"{context}: " if context else ""
    expected_stage1 = stage1_tasks_for(ruleset)
    expected_stage2 = stage2_tasks_for(ruleset)
    saved_stage1 = list(stage1["rules"])
    saved_stage2 = list(stage2["rules"])
    if saved_stage1 != expected_stage1 or saved_stage2 != expected_stage2:
        raise ValueError(
            f"{prefix}{ruleset}: unexpected stage task configuration: "
            f"stage1={saved_stage1}, stage2={saved_stage2}; expected "
            f"stage1={expected_stage1}, stage2={expected_stage2}"
        )

    rule_start = int(stage1["hp"]["rule_start"])
    if int(stage2["hp"]["rule_start"]) != rule_start:
        raise ValueError(f"{prefix}{ruleset}: inconsistent rule_start across stages")
    n_pretraining = len(expected_stage1)
    expected_width = rule_start + n_pretraining + len(expected_stage2)
    if input_width is not None and int(input_width) != expected_width:
        raise ValueError(
            f"{prefix}{ruleset}: input width {input_width} != expected "
            f"{expected_width} for {n_pretraining} pretraining rule(s)"
        )

    return {
        "stage1_rules": expected_stage1,
        "stage2_rules": expected_stage2,
        "rule_start": rule_start,
        "n_pretraining": n_pretraining,
        "novel_rule_index": n_pretraining,
        "novel_weight_column": rule_start + n_pretraining,
        "pretraining_weight_slice": slice(
            rule_start, rule_start + n_pretraining),
        "rule_input_slice": slice(rule_start, expected_width),
        "input_width": expected_width,
    }


def load_task_params(root, aname, stage):
    """Load an independent copy of one run's saved task parameters."""
    with np.load(root / f"output_{aname}_{stage}.npz", allow_pickle=True) as data:
        return copy.deepcopy(data["task_params"].item())
