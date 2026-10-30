"""Read lesion caches while retaining compatibility with the historical typo."""

import pickle
from pathlib import Path

import numpy as np


def resolve_lesion_cache_path(path):
    """Prefer a correctly spelled file, falling back to an existing legacy name."""
    path = Path(path)
    if path.exists():
        return path
    legacy = path.with_name(path.name.replace("lesion", "leison"))
    return legacy if legacy.exists() else path


def normalize_lesion_cache(value):
    """Correct metadata in memory without modifying input containers or numbers.

    Numerical arrays are reused, not copied. Colliding canonical/legacy keys
    are rejected instead of silently discarding either record. Only strings
    containing the historical spelling are changed; cache schemas stay intact.
    """
    if isinstance(value, str):
        return value.replace("leison", "lesion")
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            canonical_key = normalize_lesion_cache(key)
            if canonical_key in normalized:
                raise ValueError(f"Conflicting legacy and canonical lesion cache key: {canonical_key!r}")
            normalized[canonical_key] = normalize_lesion_cache(item)
        return normalized
    if isinstance(value, list):
        return [normalize_lesion_cache(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_lesion_cache(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "U":
            return np.char.replace(value, "leison", "lesion")
        if value.dtype.kind == "O":
            normalized = np.empty_like(value)
            for index in np.ndindex(value.shape):
                normalized[index] = normalize_lesion_cache(value[index])
            return normalized
    return value


def load_lesion_pickle(path):
    """Load a trusted local pickle with legacy filename/key support, read-only."""
    with resolve_lesion_cache_path(path).open("rb") as stream:
        return normalize_lesion_cache(pickle.load(stream))

# ── Activity-matched controls (lesion_controls.py) ───────────────────────────

# Pools a lesion_controls pickle can draw its random controls from.
CONTROL_POOLS = ("all", "responsive")
SINGLE_CLUSTER_RECORDS = ("lesion", "random_lesion", "lesion_unnorm", "random_lesion_unnorm")
MOD_BASELINE_NAMES = ("mod_nolesion", "mod_cNone")


def controls_pickle_path(lesion_pickle_path, pool):
    """The lesion_controls pickle of `pool` beside a lesion_prune_results pickle."""
    path = Path(lesion_pickle_path)
    return path.with_name(path.name.replace("lesion_prune_results_", f"lesion_controls_{pool}_"))


def _same_sequence(left, right):
    return [str(item) for item in left] == [str(item) for item in right]


def _same_clusters(left, right):
    return ({int(label): [int(m) for m in members] for label, members in left.items()}
            == {int(label): [int(m) for m in members] for label, members in right.items()})


def overlay_lesion_controls(results, controls):
    """`results` (a lesion_prune_results pickle) with its single-cluster
    records replaced by those of `controls` (a lesion_controls pickle): the
    neuron sweeps of both variants and every modulation record the controls
    pickle holds. Neither input is modified.

    The replaced records must name the same conditions and tasks, with the
    same cluster sizes and synapse clusters, else ValueError: the controls
    were then computed for other clusters. Returns (overlaid, kept), where
    `kept` lists the modulation records the controls pickle lacks, which keep
    lesion.py's controls. The overlaid dict carries the pool, repeat count,
    test seed and checksums of the controls and `combined_control_pool`
    "all", since the combined lesions are not rescored.
    """
    pool = controls.get("control_pool")
    if pool not in CONTROL_POOLS:
        raise ValueError(f"lesion_controls pickle has no valid control_pool: {pool!r}")
    problems = []
    for key in SINGLE_CLUSTER_RECORDS:
        if key not in controls or key not in results:
            problems.append(f"{key}: missing")
            continue
        if not _same_sequence(controls[key]["all_comb_names_lesion"],
                              results[key]["all_comb_names_lesion"]):
            problems.append(f"{key}: condition names differ")
        if not _same_sequence(controls[key]["all_tasks"], results[key]["all_tasks"]):
            problems.append(f"{key}: tasks differ")
        units, saved_units = controls[key].get("lesion_units"), results[key].get("lesion_units")
        if units is not None and saved_units is not None and (
                {str(n): int(u) for n, u in units.items()}
                != {str(n): int(u) for n, u in saved_units.items()}):
            problems.append(f"{key}: cluster sizes differ")
    overlaid_mod = dict(results.get("mod_lesion", {}))
    for key, record in controls.get("mod_lesion", {}).items():
        saved = overlaid_mod.get(key)
        if saved is None:
            problems.append(f"mod_lesion {key}: not in the lesion pickle")
            continue
        if not _same_sequence(record["all_comb_names_mod"], saved["all_comb_names_mod"]):
            problems.append(f"mod_lesion {key}: condition names differ")
        if not _same_clusters(record["mod_col_clusters"], saved["mod_col_clusters"]):
            problems.append(f"mod_lesion {key}: synapse clusters differ")
        if record.get("mod_lesion_mode") != saved.get("mod_lesion_mode"):
            problems.append(f"mod_lesion {key}: lesion mode differs")
    if problems:
        raise ValueError("lesion_controls pickle does not match the lesion pickle: "
                         + "; ".join(problems))
    for key, record in controls.get("mod_lesion", {}).items():
        merged = dict(overlaid_mod[key])
        merged.update(record)
        overlaid_mod[key] = merged
    kept = sorted(set(results.get("mod_lesion", {})) - set(controls.get("mod_lesion", {})))
    overlaid = dict(results)
    for key in SINGLE_CLUSTER_RECORDS:
        overlaid[key] = controls[key]
    overlaid["mod_lesion"] = overlaid_mod
    overlaid["control_pool"] = pool
    overlaid["repeat_num"] = controls["repeat_num"]
    overlaid["test_seed"] = controls.get("test_seed")
    overlaid["test_checksums"] = controls.get("test_checksums")
    overlaid["controls_schema_version"] = controls.get("schema_version")
    overlaid["combined_control_pool"] = "all"
    return overlaid, kept
