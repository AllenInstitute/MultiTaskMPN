"""
Post-processing and visualization of lesion experiment results.

Reads the raw lesion pickle produced by lesion.py and computes normalized
effects (random_accuracy - cluster_accuracy) to identify which neuron or
synapse clusters are selectively important for specific tasks. Produces:

1. Normalized lesion heatmaps — (task × cluster) matrices showing how much
   each cluster lesion impairs each task beyond the random-lesion baseline.
2. Combined heatmaps — side-by-side zero_W vs freeze_M modulation lesions
    with shared color scale, comparing removal of selected connections (zero_W)
    with freezing their M at its initial value while preserving W (freeze_M).
3. Violin plots — distribution of normalized effect across tasks for each
   cluster, highlighting clusters with broad vs. task-specific roles. The
   modulation panels cover every variant in core/modulation_variants.py, in
   its order and colors.
4. Cluster similarity vs lesion effect — compares cluster tuning similarity
   (Pearson correlation of cluster mean activity/modulation profiles) with
   lesion-profile dissimilarity (one minus the correlation of per-task effects
   z-scored against their random-control repeats), over clusters whose effect
   exceeds a control-only null, to test whether similarly-tuned clusters have
   similar causal roles. Both axes are scale-free and significance comes from
   a cluster-label (Mantel-type) Spearman permutation test, because cluster
   pairs share clusters. The former tuning-cosine vs effect-L1 scatter is kept
   as a supplement together with L1 vs summed effect magnitude, the size
   confound that produced its positive correlation.
5. Overmembership vs lesion difference — relates modulation cluster enrichment
   in (input, hidden) neuron pairs to the functional similarity (task-profile
   L1 distance) between modulation lesion and combined neuron lesion effects,
    using pooled Spearman rho and a negative-sided cluster-permutation p
    (footprint ownership shuffled). Scatter plots include quantile-bin medians,
    not a fitted regression line; neither a linear nor decreasing shape is imposed.
6. Causal dependency map — z-scores every (task, cluster) lesion effect
   against its stored random-control repeats (one-sided, BH-FDR across
   cells), biclusters the masked dependency matrix, and Mantel-tests whether
   the task organization implied by CAUSAL dependence matches the one
   implied by ACTIVITY tuning (cluster_info variance profiles). The paper
   variant pairs the unnormalized neuron clusters' lesion effects with raw
   (unnormalized) task-variance features; the normalized pairing is kept as a
   reference. Task-pair similarities, the two-sided Spearman label-permutation
   test and an OLS guide line are saved to
   causal_vs_activity_tasksim_{side}_{aname}.pkl for paper_plot.py (one file
   per activity side, hidden and input).
7. Combined-lesion interaction map — I(i,j) = combined − single_i − single_j
   per (input, hidden) cluster pair (I < 0 sub-additive/redundant, I > 0
   synergistic), regressed against each block's peak synapse-cluster
   over-membership to test whether anatomical co-location explains
   functional interaction. A saturation control separates genuine pathway
   redundancy from floor effects (a cluster that alone crushes a task to
   its accuracy floor makes ANY second lesion look sub-additive): (a) a
   headroom-restricted view keeps only cells whose single effects leave
   room for additive damage, and (b) a multiplicative survival baseline
   replaces the additive one on the bounded accuracy scale.
8. OM profile prediction — predicts each synapse cluster's full PER-TASK
    own-damage profile (not just its task-averaged magnitude) from the
   OM-weighted combination of the combined-lesion task profiles, and sweeps
   a concentration exponent alpha (weights OM^α: α=0 uniform anatomy-free
   baseline … α=∞ argmax block only) to ask whether a synapse cluster's
   function is carried by its whole anatomical footprint or by its few
   most-enriched blocks.
9. Plasticity-dependence decomposition — for every (task, synapse cluster)
   cell whose zero_W effect is significant, plasticity share =
    freeze_M effect / zero_W effect. This compares two intervention effects;
    it is not an exact additive partition of static and plastic contributions.
    Aggregated per task and compared between memory-family
   tasks (delay/dm/dms/dmc) and reaction-family tasks (fd/react) — the
   MPN prediction is that working-memory tasks run on M.
11. Task specificity and compositional sharing — how many tasks each
    (unresponsive-excluded) input, hidden or var-weighted synapse cluster
    significantly impairs, with a per-task shuffle null for the dispersion of
    that count, and the Jaccard overlap of impaired clusters for every task
    pair grouped by the task component the pair differs in (response rule,
    timing, modality, context cue, family) with a task-label permutation p.
    Saved to task_specificity_{aname}.pkl for paper_plot.py.
10. Protective-cluster dissection — decomposes every NEGATIVE normalized
    lesion effect (cluster lesion hurting LESS than the size-matched random
    control) into own damage vs control damage on the shared test set, to
    separate the mechanical reading (the cluster is inert and the control
    sampled critical hub neurons) from genuine protection (removing the
    cluster IMPROVES absolute accuracy above the intact baseline).

Outputs saved to ./multiple_tasks_norm/{aname}/.

The paper-facing normalized-effect cache uses zero_W for the primary
modulation lesion comparison with input/hidden neuron lesions. Both zero_W
and freeze_M records remain available, with explicit intervention metadata,
for the plasticity-dependence and cross-mode comparisons.

Effect matrices are saved as accuracy differences in fraction units; multiplying
by 100 for display gives percentage points, not a relative percent decrease.
This script computes per-run effects, scatter coordinates, OM rank statistics
and bin medians, other regressions and permutation/Mantel statistics.
paper_plot.py reads these for the main heatmap
and scatter figures, but its cluster-size figure derives counts/percentages
from saved memberships.

Entry points: run_pipeline.py calls `main(seed, feature)` as pipeline step 3.
Standalone, `python multiple_task/lesion_plot.py --seed 749 --feature L21e4`
(run from the repository root) re-plots one COMPLETED lesion run. Passing
`--seed all --feature L21e4` re-plots all completed runs matching that
feature. Single-run filters must still select exactly one run — anything
ambiguous or unmatched is an error.
All old files in multiple_tasks_norm/{aname}/ are cleared at the start of
every run, so stale figures never survive a re-plot.
"""
import os
from pathlib import Path
import numpy as np

import pickle
from scipy.stats import linregress, norm as gauss_norm, pearsonr, mannwhitneyu, spearmanr

import matplotlib.pyplot as plt
import matplotlib as mpl

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper
import clustering
from modulation_variants import (LESION_MODULATION_TYPES, MODULATION_TYPE_COLORS,
                                 MODULATION_VARIANCE_WEIGHTS)
from lesion_cache import load_lesion_pickle

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], 
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,   
    "ps.fonttype": 42,
})

OM_MIN_EXPECTED = 3.0


def _om_point_mask(ga, om_idx, *, skip_input=(), skip_hidden=(), min_expected=OM_MIN_EXPECTED):
    """Mask stable OM blocks for one modulation cluster.

    Keeps only blocks whose expected surviving-synapse count under the OM null
    is at least `min_expected`. If older cached OM metadata lacks the expected
    count ingredients, falls back to the non-skipped grid.
    """
    n_in = int(ga["n_in"])
    n_hid = int(ga["n_hid"])
    mask = np.ones((n_in, n_hid), dtype=bool)
    if skip_input:
        mask[np.array(sorted(skip_input), dtype=int), :] = False
    if skip_hidden:
        mask[:, np.array(sorted(skip_hidden), dtype=int)] = False

    cluster_size_percent = ga.get("cluster_size_percent")
    n_active_block = ga.get("n_active_block")
    if cluster_size_percent is None or n_active_block is None:
        return mask, None

    cluster_size_percent = np.asarray(cluster_size_percent, dtype=float)
    n_active_block = np.asarray(n_active_block, dtype=float)
    if om_idx >= cluster_size_percent.shape[0]:
        return mask, None

    expected = n_active_block * cluster_size_percent[om_idx]
    mask &= expected >= float(min_expected)
    return mask, expected


OM_N_PERM = 1000

# Baseline (no-lesion) condition names of the neuron lesions, old and new spelling.
NEURON_BASELINE_KEYS = frozenset({"pre_cNone", "post_cNone", "pre_nolesion", "post_nolesion"})


# ── Unresponsive-class lookups ──────────────────────────────────────────────
#
# The clustering appends the unresponsive (silent) neurons or synapses as one
# extra class. lesion.py and multiple_task_analysis.py now record its label
# explicitly (`unresponsive_label`, `unresponsive_conditions`,
# `unresponsive_{input,hidden}_index`, `unresponsive_{pre,post}_label`); these
# helpers read those fields. Caches written before the fields existed fall back
# to the historical convention — the unnormalized clusterings put the class
# last — which is what the fields were introduced to replace.

def _unresponsive_grid_index(ga, side, variant):
    """0-based row/column of the unresponsive class in an OM grid, or None.

    ga: an OM cache (`global_assignment*`); side: "input" or "hidden";
    variant: "norm" / "unnorm", used only by the legacy fallback.
    """
    if side not in ("input", "hidden"):
        raise ValueError("side must be 'input' or 'hidden'")
    key = f"unresponsive_{side}_index"
    if key in ga:
        return None if ga[key] is None else int(ga[key])
    if variant == "unnorm":
        return int(ga["n_in" if side == "input" else "n_hid"]) - 1
    return None


def _combined_unresponsive_index(cdata, side, variant):
    """0-based grid index of the unresponsive class in a combined-lesion entry, or None."""
    if side not in ("pre", "post"):
        raise ValueError("side must be 'pre' or 'post'")
    key = f"unresponsive_{side}_label"
    if key in cdata:
        return None if cdata[key] is None else int(cdata[key]) - 1
    if variant == "unnorm":
        return int(cdata["pre_n" if side == "pre" else "post_n"]) - 1
    return None


def _unresponsive_condition_names(entry, legacy_last=False):
    """Condition names ("pre_c21", "post_c21") of the unresponsive classes.

    entry: a `lesion` / `lesion_unnorm` record. Without the explicit field the
    legacy rule (last condition of each side) applies only when `legacy_last`.
    """
    if "unresponsive_conditions" in entry:
        return {str(name) for name in entry["unresponsive_conditions"]}
    if not legacy_last:
        return set()
    names = [n for n in entry["all_comb_names_lesion"] if n not in NEURON_BASELINE_KEYS]
    found = set()
    for prefix in ("pre_c", "post_c"):
        side = [n for n in names if n.startswith(prefix)]
        if side:
            found.add(side[-1])
    return found


def _mod_unresponsive_label(mod_data, legacy_last=False):
    """Cluster id of the unresponsive synapse class in a mod_lesion record, or None."""
    if "unresponsive_label" in mod_data:
        label = mod_data["unresponsive_label"]
        return None if label is None else int(label)
    if not legacy_last:
        return None
    ids = sorted(int(n.replace("mod_c", "")) for n in mod_data["all_comb_names_mod"]
                 if n.startswith("mod_c"))
    return ids[-1] if ids else None


def _indices_without(n, index):
    """np.arange(n) with one optional index removed."""
    keep = np.arange(int(n))
    return keep if index is None else keep[keep != int(index)]


def _om_scatter_perm_test(mod_profiles, row_om, row_cm, n_perm=OM_N_PERM, seed=0):
    """Spearman cluster-permutation p-value for the OM vs profile-L1 scatter.

    The scatter's (mod cluster, block) points are massively non-independent:
    each cluster's lesion profile is reused across all its blocks and each
    block's combined profile across all clusters, so linregress p-values treat
    ~20 clusters' worth of information as thousands of independent samples.
    The honest null keeps every lesion effect fixed and permutes WHICH cluster
    owns WHICH OM footprint — row_om[k] (masked OM values) and row_cm[k] (the
    matching blocks' combined-lesion task profiles) travel together with their
    stability mask — recomputing pooled Spearman rho each time. One-sided
    toward negative rho (hypothesis: higher OM -> more similar lesion profiles).

    mod_profiles: (C, T) per-cluster modulation-lesion task profiles.
    row_om: length-C list of (B_k,) masked OM values per footprint.
    row_cm: length-C list of (B_k, T) matching blocks' task profiles.
    y for a (cluster c, footprint k) pairing is the per-block task-profile
    L1/T distance: mean_t |mod_profiles[c, t] - row_cm[k][b, t]|.
    Returns (rho_obs, p_perm, null_rho); no regression model is fitted.
    """
    nC = len(mod_profiles)
    if n_perm < 1:
        raise ValueError("Need at least one cluster permutation.")

    def _pooled_rho(assign):
        x = np.concatenate([row_om[k] for k in assign])
        y = np.concatenate([
            np.mean(np.abs(row_cm[k] - mod_profiles[c][None, :]), axis=1)
            for c, k in enumerate(assign)])
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        return float(spearmanr(x, y).statistic)

    rho_obs = _pooled_rho(np.arange(nC))
    if nC < 2:
        return rho_obs, np.nan, np.full(n_perm, np.nan)
    rng = np.random.default_rng(seed)
    null_rho = np.array([_pooled_rho(rng.permutation(nC)) for _ in range(n_perm)])
    finite = np.isfinite(null_rho)
    if not np.isfinite(rho_obs) or not finite.any():
        return rho_obs, np.nan, null_rho
    p_perm = (1.0 + np.sum(null_rho[finite] <= rho_obs)) / (finite.sum() + 1.0)
    return rho_obs, float(p_perm), null_rho


def _om_binned_medians(om_vals, lesion_diffs, n_bins=5):
    """Describe the scatter with quantile-bin medians, keeping tied OM together."""
    om_vals = np.asarray(om_vals, dtype=float)
    lesion_diffs = np.asarray(lesion_diffs, dtype=float)
    if (om_vals.ndim != 1 or om_vals.size == 0 or om_vals.shape != lesion_diffs.shape
            or not np.isfinite(om_vals).all() or not np.isfinite(lesion_diffs).all()
            or np.any(om_vals < 0) or np.any(lesion_diffs < 0)
            or not isinstance(n_bins, int) or n_bins < 1):
        raise ValueError("Binned medians require finite nonnegative scatter pairs and positive n_bins.")
    edges = np.unique(np.quantile(om_vals, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.searchsorted(edges[1:-1], om_vals, side="right")
    groups = [bin_ids == index for index in np.unique(bin_ids)]
    return {
        "method": "quantile",
        "n_bins_requested": n_bins,
        "edges": edges,
        "x": np.asarray([np.median(om_vals[group]) for group in groups]),
        "y": np.asarray([np.median(lesion_diffs[group]) for group in groups]),
        "counts": np.asarray([np.count_nonzero(group) for group in groups]),
    }


def _om_scatter_summary(mod_profiles, row_om, row_cm, n_perm=OM_N_PERM,
                        seed=0, n_bins=5):
    """Shared single/combined export: scatter, rank permutation test and medians."""
    mod_profiles = np.asarray(mod_profiles, dtype=float)
    if (mod_profiles.ndim != 2 or mod_profiles.size == 0
            or len(row_om) != len(mod_profiles) or len(row_cm) != len(mod_profiles)
            or not np.isfinite(mod_profiles).all()):
        raise ValueError("Modulation profiles and cluster footprints must align.")
    row_om = [np.asarray(values, dtype=float) for values in row_om]
    row_cm = [np.asarray(values, dtype=float) for values in row_cm]
    for footprint, profiles in zip(row_om, row_cm):
        if (footprint.ndim != 1 or footprint.size == 0
                or profiles.shape != (footprint.size, mod_profiles.shape[1])
                or not np.isfinite(profiles).all()):
            raise ValueError("Each OM footprint must match its combined-lesion profiles.")
    om_vals = np.concatenate(row_om)
    lesion_diffs = np.concatenate([
        np.mean(np.abs(combined - modulation[None, :]), axis=1)
        for modulation, combined in zip(mod_profiles, row_cm)])
    medians = _om_binned_medians(om_vals, lesion_diffs, n_bins=n_bins)
    rho, p_perm, null_rho = _om_scatter_perm_test(
        mod_profiles, row_om, row_cm, n_perm=n_perm, seed=seed)
    return {
        "om_vals": om_vals,
        "lesion_diffs": lesion_diffs,
        "association": {
            "statistic": "spearman", "rho": rho, "p_perm": p_perm,
            "null_rho": null_rho, "n_perm": int(n_perm),
            "n_valid_perm": int(np.isfinite(null_rho).sum()),
            "n_clusters": len(mod_profiles), "seed": int(seed), "side": "less",
            "permutation_unit": "modulation_cluster_footprint",
        },
        "binned_medians": medians,
    }


def _om_pred_perm_test(pred, actual, n_perm=OM_N_PERM, seed=0):
    """Cluster-permutation p-value for the per-cluster OM-weighted prediction.

    One scalar per cluster on each side, but the same reuse concern applies to
    any downstream pooling, and the parametric p assumes exchangeable errors.
    The null permutes which cluster owns which OM-predicted value (i.e. which
    footprint), keeping the actual damages fixed. One-sided toward positive r.
    Returns (r_obs, p_perm, null_r).
    """
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    if np.std(pred) < 1e-12 or np.std(actual) < 1e-12:
        return np.nan, np.nan, np.array([])
    r_obs = float(np.corrcoef(pred, actual)[0, 1])
    rng = np.random.default_rng(seed)
    null_r = np.array([
        float(np.corrcoef(pred[rng.permutation(pred.size)], actual)[0, 1])
        for _ in range(n_perm)])
    p_perm = (1.0 + np.sum(null_r >= r_obs)) / (n_perm + 1.0)
    return r_obs, float(p_perm), null_r


def _normalized_effect_record(effect, tasks, conditions):
    """Package an already computed effect matrix with its exact axis identities."""
    values = np.asarray(effect, dtype=float)
    tasks, conditions = list(tasks), list(conditions)
    if values.shape != (len(tasks), len(conditions)):
        raise ValueError("Normalized effect matrix does not match its task/condition labels")
    if len(set(tasks)) != len(tasks) or len(set(conditions)) != len(conditions):
        raise ValueError("Normalized effect labels must be unique")
    return {"effect": values, "tasks": tasks, "conditions": conditions,
            "definition": "random_minus_lesion", "units": "fraction"}


def _normalized_modulation_effect_record(mod_type_key, mod_data, default_tasks):
    """Compute random-minus-lesion effects and export explicit mode metadata.

    Keep task and condition order, excluding baseline columns. A key suffix
    must agree with any saved mode; legacy unsuffixed records use their saved
    mode, defaulting to zero_W when that field is absent.
    """
    if "__" in mod_type_key:
        base_key, mode = mod_type_key.rsplit("__", 1)
    else:
        base_key = mod_type_key
        mode = mod_data.get("mod_lesion_mode", "zero_W")
    if mode not in ("zero_W", "freeze_M"):
        raise ValueError(f"Unknown modulation lesion mode: {mode}")
    if mod_data.get("mod_lesion_mode", mode) != mode:
        raise ValueError(f"Modulation lesion mode disagrees with key {mod_type_key}")
    conditions = list(mod_data["all_comb_names_mod"])
    lesion_acc = np.asarray(mod_data["modtask_accs"], dtype=float)
    random_acc = np.asarray(mod_data["modrandomtask_accs"], dtype=float)
    if (lesion_acc.ndim != 2 or random_acc.shape != lesion_acc.shape
            or lesion_acc.shape[1] != len(conditions)):
        raise ValueError("Modulation lesion/control accuracy shapes must match conditions")
    selected = [index for index, name in enumerate(conditions)
                if name not in ("mod_nolesion", "mod_cNone")]
    record = _normalized_effect_record(
        (random_acc - lesion_acc)[:, selected],
        mod_data.get("all_tasks", default_tasks),
        [conditions[index] for index in selected])
    record["mod_lesion_mode"] = mode
    return base_key, mode, record


# ── Cluster tuning similarity vs lesion-profile similarity (docstring #4) ──
#
# The lesion-effect L1 distance between two clusters, sum_t |e_i(t) - e_j(t)|,
# is dominated by how LARGE the two effects are, not by how different their
# task profiles are (on seed 749 it correlates at r = 0.92 with the summed
# effect magnitude). Tuning cosine similarity, by contrast, is scale-free and
# is highest between the large, high-|W| clusters whose mean profiles converge
# on the population-average profile. Both axes therefore track cluster size and
# weight, which manufactured a positive tuning-similarity/L1 correlation that
# said nothing about profile shape. The scale-free comparison below separates
# the two: x is the Pearson correlation between cluster mean tuning profiles,
# y is one minus the Pearson correlation between per-task lesion effects after
# each effect is z-scored against its own size-matched random-control repeats.
# Only clusters whose lesion effect exceeds the control null are compared, and
# significance comes from permuting cluster labels (Mantel-type test), because
# the C(C-1)/2 cluster pairs share C clusters and are not independent samples.
# The L1 version is kept as an explicit supplement together with its magnitude
# confound so the change of metric is documented in the data.
CLUSTER_CORR_SCHEMA_VERSION = 2
CLUSTER_CORR_N_PERM = 10000
# One accuracy percentage point. Saturated tasks give control repeats with
# (near-)zero spread; without a floor their tiny effects would receive huge z
# values and dominate the profile correlation.
CLUSTER_CORR_SD_FLOOR = 0.01
CLUSTER_CORR_SIG_QUANTILE = 0.95
CLUSTER_CORR_X_DEFINITION = "pearson_corr_of_cluster_mean_tuning_profiles"
CLUSTER_CORR_Y_DEFINITION = "one_minus_pearson_corr_of_z_scored_effect_profiles"
CLUSTER_CORR_L1_DEFINITION = "sum_over_tasks_abs_effect_difference"


def _z_scored_lesion_effects(lesion_acc, control_raw, sd_floor=CLUSTER_CORR_SD_FLOOR):
    """Z-score (task, cluster) lesion effects against their control repeats.

    lesion_acc: (T, C) accuracy after lesioning each cluster.
    control_raw: (T, C, R) accuracy under R size-matched random lesions.
    Returns effect = control mean - lesion (fraction units), the floored
    control standard deviation and z = effect / sd, all shaped (T, C).
    """
    lesion_acc = np.asarray(lesion_acc, dtype=float)
    control_raw = np.asarray(control_raw, dtype=float)
    if (lesion_acc.ndim != 2 or control_raw.ndim != 3
            or control_raw.shape[:2] != lesion_acc.shape or control_raw.shape[2] < 2
            or not np.isfinite(lesion_acc).all() or not np.isfinite(control_raw).all()):
        raise ValueError("Need (T, C) lesion accuracies and matching (T, C, R>=2) "
                         "finite control repeats")
    if not sd_floor > 0:
        raise ValueError("The control SD floor must be positive")
    effect = control_raw.mean(axis=2) - lesion_acc
    sd = np.maximum(control_raw.std(axis=2, ddof=1), float(sd_floor))
    return {"effect": effect, "sd": sd, "z": effect / sd,
            "n_repeats": int(control_raw.shape[2]), "sd_floor": float(sd_floor)}


def _cluster_effect_significance(control_raw, sd, z, quantile=CLUSTER_CORR_SIG_QUANTILE):
    """Flag clusters whose summed |z| exceeds a control-only null.

    The statistic is sum_t |z(t, c)|. Its null is built without any lesion:
    each control repeat k plays the role of the lesioned network against the
    mean of the other R-1 repeats, z-scored with the same floored sd, and the
    resulting statistics are pooled over clusters and repeats. A cluster is
    kept when its statistic exceeds the null's `quantile`.
    """
    control_raw = np.asarray(control_raw, dtype=float)
    sd = np.asarray(sd, dtype=float)
    z = np.asarray(z, dtype=float)
    n_rep = control_raw.shape[2]
    if not 0 < quantile < 1:
        raise ValueError("The significance quantile must lie strictly inside (0, 1)")
    statistic = np.abs(z).sum(axis=0)                         # (C,)
    total = control_raw.sum(axis=2, keepdims=True)
    held_out = control_raw
    others_mean = (total - held_out) / (n_rep - 1)
    null_z = (others_mean - held_out) / sd[:, :, None]         # (T, C, R)
    null = np.abs(null_z).sum(axis=0).ravel()                 # (C * R,)
    threshold = float(np.quantile(null, quantile))
    return {"statistic": "sum_abs_z", "per_cluster": statistic,
            "threshold": threshold, "quantile": float(quantile),
            "null": null, "significant": statistic > threshold}


def _mantel_spearman(x_matrix, y_matrix, n_perm=CLUSTER_CORR_N_PERM, seed=0):
    """Two-sided Spearman Mantel test between two square similarity matrices.

    Spearman rho is computed over the strict lower triangles. The null
    permutes the cluster labels of `y_matrix` (rows and columns together) so
    every pair keeps its dependence on the shared clusters. Returns a dict
    with rho, the permutation p, the null rhos and the test metadata.
    """
    x_matrix = np.asarray(x_matrix, dtype=float)
    y_matrix = np.asarray(y_matrix, dtype=float)
    n = x_matrix.shape[0]
    if (x_matrix.ndim != 2 or x_matrix.shape != (n, n) or y_matrix.shape != (n, n)
            or n < 3 or not np.isfinite(x_matrix).all() or not np.isfinite(y_matrix).all()):
        raise ValueError("Mantel test needs two finite square matrices over >= 3 clusters")
    if n_perm < 1:
        raise ValueError("Need at least one label permutation")
    tri = np.tril_indices(n, k=-1)
    x = x_matrix[tri]

    def _rho(matrix):
        y = matrix[tri]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        return float(spearmanr(x, y).statistic)

    rho = _rho(y_matrix)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for index in range(n_perm):
        perm = rng.permutation(n)
        null[index] = _rho(y_matrix[np.ix_(perm, perm)])
    finite = np.isfinite(null)
    if np.isfinite(rho) and finite.any():
        p_perm = float((1.0 + np.sum(np.abs(null[finite]) >= abs(rho))) / (finite.sum() + 1.0))
    else:
        p_perm = np.nan
    return {"statistic": "spearman", "rho": rho, "p_perm": p_perm, "null_rho": null,
            "n_perm": int(n_perm), "n_valid_perm": int(finite.sum()), "seed": int(seed),
            "side": "two-sided", "permutation_unit": "cluster_label",
            "n_clusters": int(n), "n_pairs": int(len(x))}


def _tuning_vs_lesion_summary(cluster_means, lesion_acc, control_raw, cluster_labels, *,
                              exclude_last_cluster=False, unresponsive_labels=None,
                              n_perm=CLUSTER_CORR_N_PERM,
                              seed=0, sd_floor=CLUSTER_CORR_SD_FLOOR,
                              quantile=CLUSTER_CORR_SIG_QUANTILE):
    """Build the scale-free tuning-vs-lesion comparison and its L1 supplement.

    cluster_means: (F, C) mean tuning profile per cluster (F features, e.g.
    task x period variances). lesion_acc: (T, C). control_raw: (T, C, R).
    cluster_labels: C labels in the shared column order.

    Main comparison (`tuning_corr`, `lesion_profile_dissim`): the unresponsive
    cluster is dropped first, then clusters without a significant lesion effect
    or with a constant profile on either side. `unresponsive_labels` names that
    class explicitly (a collection of labels, possibly empty when the run has
    no unresponsive class); the older `exclude_last_cluster=True` drops the
    last cluster and is kept for caches without the explicit label. The saved
    `exclude_last_cluster` flag records that an unresponsive exclusion was
    applied by either route. Supplement (`l1`): the former figure's tuning
    cosine similarity and effect L1 distance over the same clusters minus only
    the unresponsive one, plus the summed effect magnitude of each pair that
    explains the L1 distance.
    """
    cluster_means = np.asarray(cluster_means, dtype=float)
    cluster_labels = list(cluster_labels)
    n_clusters = len(cluster_labels)
    if (cluster_means.ndim != 2 or cluster_means.shape[1] != n_clusters
            or len(set(cluster_labels)) != n_clusters
            or not np.isfinite(cluster_means).all()):
        raise ValueError("Cluster means must be finite, (features, clusters) and match the labels")
    scored = _z_scored_lesion_effects(lesion_acc, control_raw, sd_floor=sd_floor)
    if scored["effect"].shape[1] != n_clusters:
        raise ValueError("Lesion columns must match the cluster labels")
    significance = _cluster_effect_significance(control_raw, scored["sd"], scored["z"],
                                                quantile=quantile)

    if unresponsive_labels is not None:
        unresponsive = {str(label) for label in unresponsive_labels}
        base = np.array([index for index in range(n_clusters)
                         if str(cluster_labels[index]) not in unresponsive], dtype=int)
        unresponsive_source = "explicit_label"
    elif exclude_last_cluster and n_clusters > 1:
        base = np.arange(n_clusters - 1)
        unresponsive_source = "last_cluster"
    else:
        base = np.arange(n_clusters)
        unresponsive_source = None
    exclude_last_cluster = bool(exclude_last_cluster or unresponsive_labels is not None)
    effect_base = scored["effect"][:, base]
    means_base = cluster_means[:, base]
    tuning_cos = _cosine_similarity_matrix(means_base)
    l1 = np.abs(effect_base[:, :, None] - effect_base[:, None, :]).sum(axis=0)
    magnitude = np.abs(effect_base).sum(axis=0)
    magnitude_sum = magnitude[:, None] + magnitude[None, :]
    tri_base = np.tril_indices(len(base), k=-1)
    supplement = {
        "y_definition": CLUSTER_CORR_L1_DEFINITION,
        "cluster_labels": [cluster_labels[index] for index in base],
        "tuning_cos_sim": tuning_cos[tri_base], "lesion_l1_dist": l1[tri_base],
        "effect_magnitude_sum": magnitude_sum[tri_base],
        "association_tuning": _mantel_spearman(tuning_cos, l1, n_perm=n_perm, seed=seed),
        "association_magnitude": _mantel_spearman(magnitude_sum, l1, n_perm=n_perm, seed=seed),
    }

    constant_tuning = np.std(cluster_means, axis=0) < 1e-12
    constant_z = np.std(scored["z"], axis=0) < 1e-12
    excluded = {
        "last": [cluster_labels[index] for index in range(n_clusters) if index not in base],
        "not_significant": [cluster_labels[index] for index in base
                            if not significance["significant"][index]],
        "degenerate": [cluster_labels[index] for index in base
                       if significance["significant"][index]
                       and (constant_tuning[index] or constant_z[index])],
    }
    keep = np.array([index for index in base
                     if significance["significant"][index]
                     and not constant_tuning[index] and not constant_z[index]], dtype=int)
    if len(keep) < 3:
        raise ValueError(f"Only {len(keep)} clusters with a significant, non-degenerate "
                         "lesion profile; need at least 3 for a pairwise comparison")
    tuning_corr = np.corrcoef(cluster_means[:, keep].T)
    lesion_dissim = 1.0 - np.corrcoef(scored["z"][:, keep].T)
    tri = np.tril_indices(len(keep), k=-1)
    return {
        "schema_version": CLUSTER_CORR_SCHEMA_VERSION,
        "x_definition": CLUSTER_CORR_X_DEFINITION,
        "y_definition": CLUSTER_CORR_Y_DEFINITION,
        "exclude_last_cluster": exclude_last_cluster,
        "unresponsive_source": unresponsive_source,
        "cluster_labels": cluster_labels,
        "included_clusters": [cluster_labels[index] for index in keep],
        "excluded_clusters": excluded,
        "tuning_corr": tuning_corr[tri],
        "lesion_profile_dissim": lesion_dissim[tri],
        "tuning_corr_matrix": tuning_corr,
        "lesion_profile_dissim_matrix": lesion_dissim,
        "effect": scored["effect"], "z": scored["z"],
        "n_repeats": scored["n_repeats"], "sd_floor": scored["sd_floor"],
        "significance": significance,
        "association": _mantel_spearman(tuning_corr, lesion_dissim, n_perm=n_perm, seed=seed),
        "trend_line": _descriptive_trend_line(tuning_corr[tri], lesion_dissim[tri]),
        "l1": supplement,
    }


def _descriptive_trend_line(x, y):
    """Ordinary least-squares line through the scatter, as a visual guide only.

    The reported association is Spearman rho with a cluster-label permutation
    p; this line is not that statistic (a rank correlation has no line in the
    raw coordinates) and carries no inferential claim. Returns None when x is
    constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-12:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return {"method": "ols", "slope": float(slope), "intercept": float(intercept),
            "role": "descriptive_guide_for_spearman_scatter"}


def _tuning_profiles_for_variant(cluster_means, mod_type_key):
    """Cluster mean tuning profiles as compared across clusters.

    For the signed W*Var(M) variant, positive- and negative-W synapses form
    separate clusters whose mean profiles differ in sign but not in shape, so
    the profiles are compared by magnitude (elementwise absolute value); a
    pair of same-shape clusters of opposite sign would otherwise read as
    anti-correlated. Non-negative variants pass through unchanged. Returns
    (profiles, description).
    """
    base = mod_type_key.replace("_unnormalized", "").replace("_normalized", "")
    cluster_means = np.asarray(cluster_means, dtype=float)
    if MODULATION_VARIANCE_WEIGHTS.get(base) == "signed":
        return np.abs(cluster_means), "absolute value of cluster mean (signed feature)"
    return cluster_means, "cluster mean"


def _cosine_similarity_matrix(columns):
    """Cosine similarity between the columns of a (features, items) array."""
    columns = np.asarray(columns, dtype=float)
    norms = np.linalg.norm(columns, axis=0)
    safe = np.where(norms > 0, norms, 1.0)
    unit = columns / safe
    similarity = unit.T @ unit
    return np.clip(similarity, -1.0, 1.0)


# ── Task specificity of clusters and compositional sharing (docstring #11) ──
#
# A cluster's lesion profile marks which tasks it significantly impairs. The
# number of tasks per cluster measures its specificity; whether that count
# distribution is more dispersed than a per-task-rate-preserving shuffle tells
# specialized/hub organization apart from uniform mixing (a null that keeps
# how many clusters each task depends on but breaks cluster identity). The
# same masks give, for every task pair, the Jaccard overlap of the clusters
# they depend on; grouping pairs by which task component they differ in asks
# whether shared clusters follow the compositional structure of the battery.
TASK_SPECIFICITY_N_PERM = 5000
TASK_PAIR_RELATIONS = {
    "response rule": [("fdgo", "fdanti"), ("delaygo", "delayanti"),
                      ("reactgo", "reactanti"), ("dmsgo", "dmsnogo"),
                      ("dmcgo", "dmcnogo")],
    "timing": [("fdgo", "delaygo"), ("fdgo", "reactgo"), ("delaygo", "reactgo"),
               ("fdanti", "delayanti"), ("fdanti", "reactanti"),
               ("delayanti", "reactanti")],
    "modality": [("delaydm1", "delaydm2"), ("contextdelaydm1", "contextdelaydm2")],
    "context cue": [("delaydm1", "contextdelaydm1"), ("delaydm2", "contextdelaydm2")],
    "integration family": [("delaydm1", "contextdelaydm2"), ("delaydm2", "contextdelaydm1"),
                           ("multidelaydm", "delaydm1"), ("multidelaydm", "delaydm2"),
                           ("multidelaydm", "contextdelaydm1"),
                           ("multidelaydm", "contextdelaydm2")],
    "match/category family": [("dmsgo", "dmcgo"), ("dmsgo", "dmcnogo"),
                              ("dmsnogo", "dmcgo"), ("dmsnogo", "dmcnogo")],
}
TASK_PAIR_OTHER = "other"


def _task_specificity_counts(sig):
    """Number of tasks each cluster significantly impairs; sig is (T, C) bool."""
    sig = np.asarray(sig, dtype=bool)
    if sig.ndim != 2:
        raise ValueError("sig must be a (tasks, clusters) boolean matrix")
    return sig.sum(axis=0)


def _count_dispersion_test(sig, n_perm=TASK_SPECIFICITY_N_PERM, seed=0):
    """Variance of tasks-per-cluster against a per-task shuffle of cluster identity.

    Each permutation shuffles every task row independently across clusters,
    preserving how many clusters each task depends on while destroying which
    clusters co-occur. One-sided p for the observed variance exceeding the
    null (more specialized/hub-like than mixing).
    """
    sig = np.asarray(sig, dtype=bool)
    counts = _task_specificity_counts(sig)
    if n_perm < 1:
        raise ValueError("Need at least one permutation")
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for index in range(n_perm):
        shuffled = np.stack([rng.permutation(row) for row in sig])
        null[index] = shuffled.sum(axis=0).var()
    observed = float(counts.var())
    return {"counts": counts, "observed_var": observed, "null_var": null,
            "p_perm": float((1.0 + np.sum(null >= observed)) / (n_perm + 1.0)),
            "n_perm": int(n_perm), "seed": int(seed), "side": "greater",
            "null": "independent per-task shuffle of cluster identity"}


def _pair_relation_labels(tasks):
    """Relation of every task pair (i < j): a TASK_PAIR_RELATIONS key or 'other'."""
    tasks = list(tasks)
    lookup = {}
    for relation, pairs in TASK_PAIR_RELATIONS.items():
        for first, second in pairs:
            key = frozenset((first, second))
            if key in lookup:
                raise ValueError(f"Task pair {first}/{second} listed under two relations")
            lookup[key] = relation
    labels = {}
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            labels[(i, j)] = lookup.get(frozenset((tasks[i], tasks[j])), TASK_PAIR_OTHER)
    return labels


def _jaccard_matrix(sig):
    """Jaccard overlap of significant-cluster sets for every task pair; NaN if both empty."""
    sig = np.asarray(sig, dtype=bool).astype(float)
    intersection = sig @ sig.T
    sizes = sig.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / union, np.nan)


def _sharing_by_relation(sig, tasks, n_perm=TASK_SPECIFICITY_N_PERM, seed=0):
    """Mean Jaccard sharing per task-pair relation with a task-label permutation p.

    The null relabels tasks (rows and columns of the Jaccard matrix together),
    keeping every pairwise overlap and shuffling which pairs count as related.
    One-sided: related pairs share more than the shuffled assignment predicts.
    """
    tasks = list(tasks)
    jaccard = _jaccard_matrix(sig)
    if jaccard.shape != (len(tasks), len(tasks)):
        raise ValueError("sig rows must match the task list")
    labels = _pair_relation_labels(tasks)
    relations = [name for name in TASK_PAIR_RELATIONS
                 if any(label == name for label in labels.values())] + [TASK_PAIR_OTHER]
    groups = {name: [pair for pair, label in labels.items() if label == name] for name in relations}
    related = [pair for pair, label in labels.items() if label != TASK_PAIR_OTHER]

    def _mean(matrix, pairs):
        values = np.array([matrix[i, j] for i, j in pairs], dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else np.nan

    rng = np.random.default_rng(seed)
    null = {name: np.empty(n_perm) for name in relations + ["related"]}
    for index in range(n_perm):
        perm = rng.permutation(len(tasks))
        shuffled = jaccard[np.ix_(perm, perm)]
        for name in relations:
            null[name][index] = _mean(shuffled, groups[name])
        null["related"][index] = _mean(shuffled, related)
    out = {"jaccard": jaccard, "tasks": tasks, "relations": relations, "n_perm": int(n_perm),
           "seed": int(seed), "side": "greater", "permutation_unit": "task_label",
           "by_relation": {}}
    for name, pairs in list(groups.items()) + [("related", related)]:
        observed = _mean(jaccard, pairs)
        finite = null[name][np.isfinite(null[name])]
        p_perm = (float((1.0 + np.sum(finite >= observed)) / (finite.size + 1.0))
                  if np.isfinite(observed) and finite.size else np.nan)
        out["by_relation"][name] = {
            "pairs": [(tasks[i], tasks[j]) for i, j in pairs],
            "values": np.array([jaccard[i, j] for i, j in pairs], dtype=float),
            "mean": observed, "p_perm": p_perm, "null_mean": null[name]}
    return out


def _task_specificity_summary(sig_by_type, tasks, n_perm=TASK_SPECIFICITY_N_PERM, seed=0):
    """Counts, dispersion tests and relation sharing for each cluster type."""
    summary = {"schema_version": 1, "tasks": list(tasks), "n_perm": int(n_perm),
               "types": {}}
    for type_name, entry in sig_by_type.items():
        sig = np.asarray(entry["sig"], dtype=bool)
        if sig.shape != (len(tasks), len(entry["cluster_labels"])):
            raise ValueError(f"{type_name}: sig shape does not match tasks and clusters")
        summary["types"][type_name] = {
            "cluster_labels": list(entry["cluster_labels"]),
            "dispersion": _count_dispersion_test(sig, n_perm=n_perm, seed=seed),
            "sharing": _sharing_by_relation(sig, tasks, n_perm=n_perm, seed=seed),
            "sig": sig,
        }
    return summary


def main(seed, feature):
    aname = f"everything_seed{seed}_{feature}+hidden300+batch128+angle"
    print(f"aname: {aname}")

    pickle_dir = f"./multiple_tasks_perf/{aname}"
    save_dir = f"./multiple_tasks_norm/{aname}"
    os.makedirs(save_dir, exist_ok=True)
    # Clear ALL previous outputs for this run before plotting anything, so a
    # re-plot can never leave stale figures from an older code version behind.
    _old_files = [f for f in Path(save_dir).iterdir() if f.is_file()]
    for _old in _old_files:
        _old.unlink()
    if _old_files:
        print(f"Cleared {len(_old_files)} old file(s) from {save_dir}")

    pickle_name = f"{pickle_dir}/lesion_prune_results_{aname}.pkl"
    results = load_lesion_pickle(pickle_name)
        
    # handle both old pickle names ("pre_cNone") and new ("pre_nolesion") after rename fix
    baseline_keys = set(NEURON_BASELINE_KEYS)
    mod_lesion_results = results.get("mod_lesion", {})
    normalized_effects = {"schema_version": 1, "aname": aname,
                          "primary_modulation_mode": "zero_W", "entries": {}}

    def compute_and_plot_normalized_lesion(lesion_key, random_key, savename, xlabel_suffix=""):
        """Compute normalized lesion effect (random - cluster) and plot its heatmap.
        Returns (select_props, all_comb_names_filtered) for downstream use.
        """
        all_comb_names = results[lesion_key]["all_comb_names_lesion"]
        def _rename(k):
            return k.replace("pre_c", "i").replace("post_c", "h")
        all_comb_names_filtered = [_rename(k) for k in all_comb_names if k not in baseline_keys]
        tasks = results[lesion_key]["all_tasks"]

        ihtask = np.asarray(results[lesion_key]["ihtask_accs"], dtype=float)
        ihrandom = np.asarray(results[random_key]["ihrandomtask_accs"], dtype=float)

        props = []
        for key_idx, key in enumerate(all_comb_names):
            if key not in baseline_keys:
                props.append(ihrandom[:, key_idx] - ihtask[:, key_idx])

        props = np.array(props).T  # (n_tasks, n_clusters)
        normalized_effects["entries"][lesion_key] = _normalized_effect_record(
            props, tasks, [key for key in all_comb_names if key not in baseline_keys])
        suffix = f" {xlabel_suffix}" if xlabel_suffix else ""
        print(f"[{savename}] select_props: {props.shape}")

        helper.plot_heatmap(props, all_comb_names_filtered, tasks,
                            xlabel=f"Lesion Condition{suffix}", ylabel="Task",
                            savename=savename, aname=aname, label="Normalized Accuracy",
                            vmin=None, vmax=None, save_dir=save_dir)

        return props, all_comb_names_filtered

    select_props, all_comb_names_lesion_ = compute_and_plot_normalized_lesion(
        "lesion", "random_lesion", "normalized_lesion",
    )
    all_tasks = results["lesion"]["all_tasks"]

    select_props_unnorm = None
    all_comb_names_unnorm_ = None
    if "lesion_unnorm" in results and "random_lesion_unnorm" in results:
        select_props_unnorm, all_comb_names_unnorm_ = compute_and_plot_normalized_lesion(
            "lesion_unnorm", "random_lesion_unnorm",
            "normalized_lesion_unnorm", xlabel_suffix="(unnorm)",
        )

    # Combined violin: 4 panels — input/hidden × normalized/unnormalized
    if select_props_unnorm is not None:
        _n_input_norm_v = len([n for n in all_comb_names_lesion_ if n.startswith("i")])
        _n_input_unnorm_v = len([n for n in all_comb_names_unnorm_ if n.startswith("i")])

        _ih_panels = [
            ("Input (normalized)", select_props[:, :_n_input_norm_v],
             [n for n in all_comb_names_lesion_ if n.startswith("i")]),
            ("Hidden (normalized)", select_props[:, _n_input_norm_v:],
             [n for n in all_comb_names_lesion_ if n.startswith("h")]),
            ("Input (unnormalized)", select_props_unnorm[:, :_n_input_unnorm_v],
             [n for n in all_comb_names_unnorm_ if n.startswith("i")]),
            ("Hidden (unnormalized)", select_props_unnorm[:, _n_input_unnorm_v:],
             [n for n in all_comb_names_unnorm_ if n.startswith("h")]),
        ]
        max_cls = max(p.shape[1] for _, p, _ in _ih_panels)
        _all_ih_vals = np.concatenate([p.ravel() * 100 for _, p, _ in _ih_panels])
        _ih_ylim = (min(_all_ih_vals.min() * 1.1, -1), max(_all_ih_vals.max() * 1.1, 1))

        fig_w = max(4, 0.45 * max_cls + 1.5)
        fig_ih, axes_ih = plt.subplots(4, 1, figsize=(fig_w, 1.8 * 4), dpi=300)

        for panel_idx, (label, props_panel, cnames_panel) in enumerate(_ih_panels):
            ax_v = axes_ih[panel_idx]
            n_cls = props_panel.shape[1]
            violin_data = [props_panel[:, ci] * 100 for ci in range(n_cls)]
            parts = ax_v.violinplot(violin_data, positions=range(n_cls),
                                    showmeans=True, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("steelblue")
                pc.set_alpha(0.6)
            parts["cmeans"].set_color("tomato")
            parts["cmeans"].set_linewidth(1.0)

            for ci in range(n_cls):
                ax_v.scatter(
                    np.full(len(violin_data[ci]), ci), violin_data[ci],
                    color="black", s=5, alpha=0.3, zorder=3,
                )

            ax_v.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
            ax_v.set_xticks(range(n_cls))
            ax_v.set_xticklabels(cnames_panel, rotation=25, ha="right", fontsize=8)
            ax_v.set_ylim(_ih_ylim)
            ax_v.set_ylabel("Effect (%)", fontsize=9)
            ax_v.set_title(label, fontsize=9)
            ax_v.spines["top"].set_visible(False)
            ax_v.spines["right"].set_visible(False)
            ax_v.tick_params(labelsize=8)

        fig_ih.tight_layout()
        fig_ih.savefig(f"{save_dir}/normalized_lesion_violin_all_{aname}.png", dpi=300)
        plt.close(fig_ih)
        print("Saved combined input/hidden violin plot (4 panels)")

    # Histogram of mean lesion effect per cluster for input/hidden (4 categories)
    # Split select_props into input and hidden based on the all_comb structure
    # all_comb_names_lesion_ has i1..iN then h1..hM (after _rename)
    _n_input_norm = len([n for n in all_comb_names_lesion_ if n.startswith("i")])

    _hist_data = {}
    _hist_data["Input (norm)"] = select_props[:, :_n_input_norm].mean(axis=0) * 100
    _hist_data["Hidden (norm)"] = select_props[:, _n_input_norm:].mean(axis=0) * 100
    if select_props_unnorm is not None and all_comb_names_unnorm_ is not None:
        _n_input_unnorm = len([n for n in all_comb_names_unnorm_ if n.startswith("i")])
        _hist_data["Input (unnorm)"] = select_props_unnorm[:, :_n_input_unnorm].mean(axis=0) * 100
        _hist_data["Hidden (unnorm)"] = select_props_unnorm[:, _n_input_unnorm:].mean(axis=0) * 100

    if _hist_data:
        _hist_colors = {"Input (norm)": "#2171b5", "Hidden (norm)": "#cb181d",
                        "Input (unnorm)": "#6baed6", "Hidden (unnorm)": "#fc9272"}

        # Build task-specific data (all task × cluster values, not averaged)
        _hist_data_individual = {}
        _hist_data_individual["Input (norm)"] = select_props[:, :_n_input_norm].ravel() * 100
        _hist_data_individual["Hidden (norm)"] = select_props[:, _n_input_norm:].ravel() * 100
        if select_props_unnorm is not None and all_comb_names_unnorm_ is not None:
            _n_input_unnorm = len([n for n in all_comb_names_unnorm_ if n.startswith("i")])
            _hist_data_individual["Input (unnorm)"] = select_props_unnorm[:, :_n_input_unnorm].ravel() * 100
            _hist_data_individual["Hidden (unnorm)"] = select_props_unnorm[:, _n_input_unnorm:].ravel() * 100

        _all_hist_vals = np.concatenate(list(_hist_data.values()))
        _hist_bins = np.linspace(_all_hist_vals.min(), _all_hist_vals.max(), 15)
        _all_indiv_vals = np.concatenate(list(_hist_data_individual.values()))
        _hist_bins_indiv = np.linspace(_all_indiv_vals.min(), _all_indiv_vals.max(), 25)

        fig_hist, (ax_mean, ax_indiv, ax_stats) = plt.subplots(
            1, 3, figsize=(11, 3), dpi=300, gridspec_kw={"width_ratios": [1, 1, 0.7]})

        # Left: cluster-averaged
        for label, vals in _hist_data.items():
            _mean = np.mean(vals)
            _med = np.median(vals)
            _lbl = f"{label} (μ={_mean:.2f}, md={_med:.2f})"
            ax_mean.hist(vals, bins=_hist_bins, alpha=0.5, label=_lbl, color=_hist_colors.get(label))
        ax_mean.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_mean.set_xlabel("Mean effect per cluster (%)", fontsize=8)
        ax_mean.set_ylabel("# Clusters", fontsize=8)
        ax_mean.set_title("Cluster-averaged", fontsize=8)
        ax_mean.legend(fontsize=5.5, frameon=False)
        ax_mean.spines["top"].set_visible(False)
        ax_mean.spines["right"].set_visible(False)
        ax_mean.tick_params(labelsize=7)

        # Middle: task-specific (all individual values)
        for label, vals in _hist_data_individual.items():
            ax_indiv.hist(vals, bins=_hist_bins_indiv, alpha=0.5, label=label, color=_hist_colors.get(label))
        ax_indiv.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_indiv.set_xlabel("Effect per (task, cluster) (%)", fontsize=8)
        ax_indiv.set_ylabel("# (task, cluster) pairs", fontsize=8)
        ax_indiv.set_title("Task-specific", fontsize=8)
        ax_indiv.legend(fontsize=5.5, frameon=False)
        ax_indiv.spines["top"].set_visible(False)
        ax_indiv.spines["right"].set_visible(False)
        ax_indiv.tick_params(labelsize=7)

        # Right: summary statistics (std and %>0) as grouped bars
        _ih_type_tags = list(_hist_data_individual.keys())
        _ih_stds = [np.std(v) for v in _hist_data_individual.values()]
        _ih_pct_pos = [(v > 0).mean() * 100 for v in _hist_data_individual.values()]

        _x_stats = np.arange(len(_ih_type_tags))
        _bar_w = 0.35
        ax_stats_twin = ax_stats.twinx()
        ax_stats.bar(_x_stats - _bar_w / 2, _ih_stds, _bar_w,
                     color="steelblue", alpha=0.7, label="Std (%)")
        ax_stats_twin.bar(_x_stats + _bar_w / 2, _ih_pct_pos, _bar_w,
                          color="tomato", alpha=0.7, label="% > 0")
        ax_stats.set_xticks(_x_stats)
        ax_stats.set_xticklabels(_ih_type_tags, rotation=25, ha="right", fontsize=6)
        ax_stats.set_ylabel("Std (%)", fontsize=8, color="steelblue")
        ax_stats_twin.set_ylabel("% > 0", fontsize=8, color="tomato")
        ax_stats.set_title("Informativeness", fontsize=8)
        ax_stats.spines["top"].set_visible(False)
        ax_stats_twin.spines["top"].set_visible(False)
        ax_stats.tick_params(labelsize=7)
        ax_stats_twin.tick_params(labelsize=7)
        ax_stats.legend(loc="upper left", fontsize=6, frameon=False)
        ax_stats_twin.legend(loc="upper right", fontsize=6, frameon=False)

        fig_hist.tight_layout()
        fig_hist.savefig(f"{save_dir}/normalized_lesion_hist_mean_{aname}.png", dpi=300)
        plt.close(fig_hist)
        print("Saved input/hidden histogram (cluster-averaged + task-specific + stats)")

    # ── Ranked cluster importance ──
    def _plot_ranked_effect(data_dict, colors, title, savepath):
        """Plot sorted mean effect per category on one axis."""
        fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)
        for label, vals in data_dict.items():
            sorted_vals = np.sort(vals)[::-1]
            ax.plot(range(1, len(sorted_vals) + 1), sorted_vals,
                    marker="o", markersize=4, linewidth=1.2,
                    color=colors.get(label), label=label, alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xlabel("Cluster rank (sorted by effect)", fontsize=9)
        ax.set_ylabel("Mean normalized effect (%)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(savepath, dpi=300)
        plt.close(fig)
        print(f"Saved {os.path.basename(savepath)}")

    # Ranked effect for input/hidden
    if _hist_data:
        _ih_rank_colors = {"Input (norm)": "#2171b5", "Hidden (norm)": "#cb181d",
                           "Input (unnorm)": "#6baed6", "Hidden (unnorm)": "#fc9272"}
        _plot_ranked_effect(_hist_data, _ih_rank_colors,
                            "Ranked cluster importance (input/hidden)",
                            f"{save_dir}/normalized_lesion_ranked_{aname}.png")

    # ══════════════════════════════════════════════════════════════════
    # Causal dependency map — significance layer + biclustering, and the
    # causal-vs-activity task-organization comparison (module docstring #6).
    #
    # Statistics: every (task, cluster) lesion effect is z-scored against
    # its own size-matched random-control distribution, using the raw
    # repeats lesion.py stores (ihrandomtask_accs_raw):
    #     z = (mean_ctrl - acc_lesion) / std_ctrl,   p = Phi(-z)  one-sided
    # The normal approximation is justified because each control accuracy
    # is itself a mean over test_n_batch trials. BH-FDR at q = 0.05 across
    # all cells of a variant. NB lesion.py's control cache shares draws
    # between same-(side, size) conditions, so p-values of such cells are
    # correlated — fine for a per-cell mask, but the cells are not fully
    # independent.
    # ══════════════════════════════════════════════════════════════════
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage as _sch_linkage, \
        leaves_list as _sch_leaves

    def _bh_fdr_mask(p, q=0.05):
        """Benjamini-Hochberg rejection mask, same shape as p."""
        flat = np.asarray(p, float).ravel()
        m = flat.size
        order = np.argsort(flat)
        passed = flat[order] <= q * np.arange(1, m + 1) / m
        k = (np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
        mask = np.zeros(m, dtype=bool)
        mask[order[:k]] = True
        return mask.reshape(np.asarray(p).shape)

    def _causal_dependency(lesion_key, random_key, vtag):
        """FDR-masked (task, cluster) dependency matrix + biclustered view.

        Returns the UNMASKED effect matrix (n_tasks, n_clusters) for reuse
        by the task-similarity comparison below (correlations pool over all
        clusters and are robust to per-cell noise, so they use the full
        matrix; the masked matrix drives the module-map figure only)."""
        names_all = results[lesion_key]["all_comb_names_lesion"]
        keep_idx = [k for k, n in enumerate(names_all) if n not in baseline_keys]
        cnames = [names_all[k].replace("pre_c", "i").replace("post_c", "h")
                  for k in keep_idx]
        accs = np.asarray(results[lesion_key]["ihtask_accs"], float)[:, keep_idx]
        raw = np.asarray(results[random_key]["ihrandomtask_accs_raw"],
                         float)[:, keep_idx, :]
        E = raw.mean(axis=2) - accs                       # (n_tasks, n_clusters)
        # Floor the control std: saturated or degenerate controls can have
        # zero spread, which would declare negligible effects significant.
        std = np.maximum(raw.std(axis=2), 1e-4)
        z = E / std
        pvals = gauss_norm.sf(z)                          # one-sided: worse than control
        sig = _bh_fdr_mask(pvals, q=0.05)
        E_sig = np.where(sig, E, 0.0)

        # Bicluster the MASKED matrix so the ordering is driven by
        # significant structure, not by sub-threshold noise.
        row_order = _sch_leaves(_sch_linkage(E_sig, method="ward"))
        col_order = _sch_leaves(_sch_linkage(E_sig.T, method="ward"))

        n_sig, n_cells = int(sig.sum()), E.size
        vmax = max(float(np.abs(E).max()) * 100, 1e-6)
        n_cl = len(cnames)
        fig, axs = plt.subplots(
            2, 1, figsize=(max(10, 0.32 * n_cl + 2.5),
                           2 * (0.32 * len(all_tasks) + 1.6)), dpi=300)
        sns.heatmap(E * 100, ax=axs[0], cmap="RdBu_r", center=0,
                    vmin=-vmax, vmax=vmax,
                    xticklabels=cnames, yticklabels=all_tasks,
                    cbar_kws={"label": "Effect (%)", "shrink": 0.8})
        _sr, _sc = np.nonzero(sig)
        axs[0].scatter(_sc + 0.5, _sr + 0.5, s=4, color="black", zorder=3)
        axs[0].set_title(
            f"Normalized effect; dots = significant "
            f"({n_sig}/{n_cells} cells, one-sided BH-FDR q=0.05)", fontsize=9)

        sns.heatmap(E_sig[np.ix_(row_order, col_order)] * 100, ax=axs[1],
                    cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=[cnames[c] for c in col_order],
                    yticklabels=[all_tasks[r] for r in row_order],
                    cbar_kws={"label": "Effect (%)", "shrink": 0.8})
        axs[1].set_title("FDR-masked dependency matrix, biclustered (Ward)",
                         fontsize=9)
        for ax in axs:
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Cluster", fontsize=8)
            ax.set_ylabel("Task", fontsize=8)
        fig.suptitle(f"Causal dependency map [{vtag}]", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/causal_dependency_{vtag}_{aname}.png", dpi=300)
        plt.close(fig)

        with open(f"{save_dir}/causal_dependency_{vtag}_{aname}.pkl", "wb") as _f:
            pickle.dump({
                "effect": E, "z": z, "p": pvals, "sig": sig,
                "cluster_names": cnames, "tasks": list(all_tasks),
                "row_order": np.asarray(row_order),
                "col_order": np.asarray(col_order),
                "fdr_q": 0.05, "n_sig": n_sig,
            }, _f)
        print(f"[causal-dep {vtag}] {n_sig}/{n_cells} significant cells "
              f"(BH-FDR q=0.05); saved map + pkl")
        # Unresponsive classes, in the renamed ("i3"/"h21") cluster names, read
        # from the lesion pickle; legacy pickles: last cluster of each side of
        # the unnormalized variant.
        unresponsive_names = {
            name.replace("pre_c", "i").replace("post_c", "h")
            for name in _unresponsive_condition_names(results[lesion_key],
                                                      legacy_last=(vtag == "unnorm"))}
        _causal_sig[vtag] = {"sig": sig, "cluster_names": cnames,
                             "unresponsive_names": unresponsive_names}
        return E

    _causal_sig = {}   # reused by the task-specificity section below
    E_dep_norm = _causal_dependency("lesion", "random_lesion", "norm")
    E_dep_unnorm = None
    if "lesion_unnorm" in results and "random_lesion_unnorm" in results:
        E_dep_unnorm = _causal_dependency("lesion_unnorm", "random_lesion_unnorm", "unnorm")

    # ── Causal vs activity task organization ────────────────────────────
    # Task-task similarity from lesion-dependency profiles vs from task-averaged
    # activity tuning. The paper-facing variant is UNNORMALIZED on both sides
    # (unnormalized neuron clusters' lesion effects; raw task-variance features),
    # matching the other lesion figures. Normalization divides each neuron's
    # variance by its maximum over rules and thereby discards the amplitude that
    # predicts causal impact: across the seven L2=1e-4 seeds the unnormalized
    # pairing gives hidden rho 0.27-0.50 (all seven p < 0.05) against 0.11-0.35
    # (three of seven) for the normalized pairing, while the input side stays
    # near zero under both. The normalized pairing is kept in the cache as a
    # reference variant. Periods of a task are averaged into one activity
    # vector; a response-period-only variant is not used because it also makes
    # the input side correlate.
    def _mantel(S_ref, S_other, n_perm=10000, seed=0):
        """One-sided Pearson Mantel test (positive association) on upper triangles."""
        n = S_ref.shape[0]
        iu = np.triu_indices(n, k=1)
        y = S_other[iu]
        r_obs = float(np.corrcoef(S_ref[iu], y)[0, 1])
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(n)
            if np.corrcoef(S_ref[np.ix_(perm, perm)][iu], y)[0, 1] >= r_obs:
                count += 1
        return r_obs, (count + 1) / (n_perm + 1)

    def _activity_task_similarity(ci_entry, tasks):
        """Task x task Pearson correlation of period-averaged task-variance profiles."""
        V = ci_entry["cell_vars_rules_sorted_norm"]
        tb = ci_entry["tb_break_name"]
        A_task = np.stack([
            V[[r for r, nm in enumerate(tb) if str(nm).split("-")[0] == t]].mean(axis=0)
            for t in tasks
        ])                                   # (n_tasks, n_neurons)
        return np.corrcoef(A_task)

    def _causal_vs_activity_variant(S_act, S_les, tasks):
        """Pairs, two-sided Spearman label-permutation test and OLS guide for one pairing."""
        iu = np.triu_indices(len(tasks), k=1)
        return {
            "activity_similarity": S_act, "causal_similarity": S_les,
            "activity_pairs": S_act[iu], "causal_pairs": S_les[iu],
            "association": _mantel_spearman(S_act, S_les, n_perm=CLUSTER_CORR_N_PERM, seed=0),
            "trend_line": _descriptive_trend_line(S_act[iu], S_les[iu]),
        }

    _ci_path = f"./multiple_tasks_analysis/{aname}/cluster_info_{aname}.pkl"
    if os.path.exists(_ci_path) and E_dep_unnorm is not None:
        with open(_ci_path, "rb") as _f:
            cluster_info = pickle.load(_f)  # also reused by later sections

        _S_les = {"unnormalized": np.corrcoef(E_dep_unnorm),
                  "normalized": np.corrcoef(E_dep_norm)}
        for side in ["hidden", "input"]:
            _variants = {}
            for variant in ("unnormalized", "normalized"):
                S_act = _activity_task_similarity(cluster_info[f"{side}_{variant}"], all_tasks)
                if not (np.isfinite(_S_les[variant]).all() and np.isfinite(S_act).all()):
                    print(f"[causal-vs-activity] {side}/{variant}: non-finite similarity, skipping")
                    continue
                _variants[variant] = _causal_vs_activity_variant(S_act, _S_les[variant], all_tasks)
            if "unnormalized" not in _variants:
                continue
            primary = _variants["unnormalized"]
            S_act, S_les = primary["activity_similarity"], primary["causal_similarity"]
            r_m, p_m = _mantel(S_act, S_les)
            _assoc_sp = primary["association"]
            with open(f"{save_dir}/causal_vs_activity_tasksim_{side}_{aname}.pkl", "wb") as _f:
                pickle.dump({
                    "schema_version": 2, "aname": aname, "side": side,
                    "neuron_variant": "unnormalized",
                    "tasks": list(all_tasks),
                    **primary,
                    "x_definition": "pearson_corr_of_task_mean_activity_variance_profiles",
                    "y_definition": "pearson_corr_of_task_lesion_effect_profiles_over_neuron_clusters",
                    "lesion_source": "causal_dependency_unnorm (input + hidden neuron clusters)",
                    "pearson_mantel_one_sided": {"r": r_m, "p_perm": p_m, "n_perm": 10000},
                    "reference_variants": {name: value for name, value in _variants.items()
                                           if name != "unnormalized"},
                }, _f)

            fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), dpi=300)
            for ax, S, ttl in [(axs[0], S_les, "Causal (lesion profiles, unnormalized clusters)"),
                               (axs[1], S_act, f"Activity ({side} tuning, unnormalized)")]:
                sns.heatmap(S, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                            xticklabels=all_tasks, yticklabels=all_tasks,
                            cbar_kws={"shrink": 0.75})
                ax.set_title(ttl, fontsize=9)
                ax.tick_params(labelsize=6)
            axs[2].scatter(primary["activity_pairs"], primary["causal_pairs"], s=14, alpha=0.7,
                           color="steelblue", edgecolors="none")
            axs[2].set_xlabel(f"Activity task similarity ({side})", fontsize=8)
            axs[2].set_ylabel("Causal task similarity", fontsize=8)
            _ref = _variants.get("normalized")
            _ref_txt = (f"; normalized pairing rho={_ref['association']['rho']:.2f}, "
                        f"p={_ref['association']['p_perm']:.3f}" if _ref else "")
            axs[2].set_title(f"Spearman rho={_assoc_sp['rho']:.2f}, two-sided "
                             f"p={_assoc_sp['p_perm']:.4f}; Pearson Mantel r={r_m:.2f}, "
                             f"one-sided p={p_m:.4f}{_ref_txt}", fontsize=7)
            axs[2].spines["top"].set_visible(False)
            axs[2].spines["right"].set_visible(False)
            fig.suptitle("Do tasks that look alike (activity) depend on the "
                         "same clusters (lesion)?", fontsize=9)
            fig.tight_layout()
            fig.savefig(f"{save_dir}/causal_vs_activity_tasksim_{side}_{aname}.png",
                        dpi=300)
            plt.close(fig)
            print(f"[causal-vs-activity] {side} (unnormalized): Spearman rho={_assoc_sp['rho']:.2f}, "
                  f"p={_assoc_sp['p_perm']:.4f}; Pearson Mantel r={r_m:.2f}, p={p_m:.4f}"
                  f"{_ref_txt}; cache saved")
    elif E_dep_unnorm is None:
        print("[causal-vs-activity] unnormalized lesion results not found, skipping")
    else:
        print("[causal-vs-activity] cluster_info pickle not found, skipping")

    # ══════════════════════════════════════════════════════════════════
    # Protective-cluster dissection (module docstring #10).
    # Many cluster lesions have NEGATIVE normalized effect — the size-
    # matched random control hurts more than the cluster lesion. Two very
    # different readings that this block separates:
    #   (i)  mechanical — the cluster itself is inert (own damage ≈ 0);
    #        the negativity comes entirely from the random control
    #        sampling critical (hub) neurons;
    #   (ii) genuine protection — removing the cluster IMPROVES absolute
    #        accuracy above the intact baseline (disinhibition-like).
    # Decomposition per (task, cluster) cell, all on the SAME trials (each
    # task's test set is shared by the baseline, cluster and control runs
    # inside lesion.py, and the forward pass is deterministic):
    #   own_damage  = baseline − lesion acc        (< 0 ⇒ improvement)
    #   ctrl_damage = baseline − mean control acc
    #   normalized effect (random − lesion) ≡ own_damage − ctrl_damage
    # Noise scale: accuracies are means over ~_N_EVAL_TRIALS trials, so a
    # binomial-style se = sqrt(base(1−base)/N), with a sqrt(2) independence
    # (upper-bound) factor for the paired difference. These are heuristic
    # SCREENING thresholds, not formal tests — a genuinely protective
    # cluster must show systematic improvement across tasks, not one cell.
    # ══════════════════════════════════════════════════════════════════
    # lesion.py's per-task trial count; pickles written before it was stored
    # used 200.
    _N_EVAL_TRIALS = int(results.get("test_n_batch", 200))

    for _pc_vtag, _pc_lesion, _pc_random in [
        ("norm", "lesion", "random_lesion"),
        ("unnorm", "lesion_unnorm", "random_lesion_unnorm"),
    ]:
        if _pc_lesion not in results or _pc_random not in results:
            continue
        _names_pc = results[_pc_lesion]["all_comb_names_lesion"]
        _acc_pc = np.asarray(results[_pc_lesion]["ihtask_accs"], float)
        _rnd_pc = np.asarray(results[_pc_random]["ihrandomtask_accs"], float)
        _units_pc = results[_pc_lesion].get("lesion_units", {})

        # The two no-lesion baselines are both no-op forwards on the same
        # test set and should be identical; average defensively if not.
        _b_pre = _acc_pc[:, _names_pc.index("pre_nolesion")]
        _b_post = _acc_pc[:, _names_pc.index("post_nolesion")]
        if not np.allclose(_b_pre, _b_post, atol=1e-6):
            print(f"[protective {_pc_vtag}] warning: the two no-lesion "
                  "baselines differ; using their mean")
        base_pc = (_b_pre + _b_post) / 2.0                        # (T,)

        keep_pc = [k for k, n in enumerate(_names_pc) if n not in baseline_keys]
        cn_pc = [_names_pc[k].replace("pre_c", "i").replace("post_c", "h")
                 for k in keep_pc]
        is_input_pc = np.array([n.startswith("i") for n in cn_pc])
        sizes_pc = np.array([float(_units_pc.get(_names_pc[k], np.nan))
                             for k in keep_pc])

        own = base_pc[:, None] - _acc_pc[:, keep_pc]    # (T, C); < 0 = improvement
        ctrl = base_pc[:, None] - _rnd_pc[:, keep_pc]   # (T, C)
        E_pc = own - ctrl                               # ≡ random − lesion
        se_pc = np.sqrt(np.clip(base_pc * (1 - base_pc), 1e-6, None)
                        / _N_EVAL_TRIALS)               # (T,)
        z_own = own / (np.sqrt(2.0) * se_pc[:, None])

        improving = (own < 0) & (z_own < -2)            # true improvement cells
        damaging = (own > 0) & (z_own > 2)
        inert = ~improving & ~damaging
        neg = E_pc < 0
        n_neg = int(neg.sum())
        _frac = lambda m: (100.0 * (neg & m).sum() / n_neg) if n_neg else 0.0

        # Per-cluster verdict on the task-mean own damage
        mean_own = own.mean(axis=0)
        se_mean = np.sqrt((2.0 * se_pc ** 2).sum()) / len(all_tasks)
        z_cl = mean_own / se_mean
        protective_cand = z_cl < -2
        damaging_cl = z_cl > 2

        fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.9), dpi=300)

        # P1: cell-level decomposition — own vs control damage
        for _mask, _col, _lbl in [(is_input_pc, "#4292c6", "input"),
                                  (~is_input_pc, "#e6550d", "hidden")]:
            axs[0].scatter(own[:, _mask].ravel() * 100,
                           ctrl[:, _mask].ravel() * 100,
                           s=8, alpha=0.45, color=_col, edgecolors="none",
                           label=_lbl)
        _lim = [min(own.min(), ctrl.min()) * 100, max(own.max(), ctrl.max()) * 100]
        axs[0].plot(_lim, _lim, color="grey", linestyle="--", linewidth=0.6,
                    label="effect = 0")
        axs[0].axvline(0, color="black", linewidth=0.6)
        axs[0].set_xlabel("Own damage (%)  [< 0 = lesion IMPROVES accuracy]",
                          fontsize=8)
        axs[0].set_ylabel("Random-control damage (%)", fontsize=8)
        axs[0].set_title("Above diagonal = negative normalized effect;\n"
                         "left of x=0 = candidate true protection", fontsize=8)
        axs[0].legend(fontsize=6, frameon=False, loc="upper left")

        # P2: per-cluster task-mean own damage, sorted, class-colored
        _ord_pc = np.argsort(mean_own)
        _cols = ["#d73027" if protective_cand[c]
                 else ("#2171b5" if damaging_cl[c] else "#bdbdbd")
                 for c in _ord_pc]
        axs[1].bar(np.arange(len(_ord_pc)), mean_own[_ord_pc] * 100,
                   yerr=2 * se_mean * 100, color=_cols, edgecolor="black",
                   linewidth=0.3, error_kw={"elinewidth": 0.4})
        axs[1].axhline(0, color="black", linewidth=0.6)
        axs[1].set_xticks(np.arange(len(_ord_pc)))
        axs[1].set_xticklabels([cn_pc[c] for c in _ord_pc], rotation=60,
                               ha="right", fontsize=5)
        axs[1].set_ylabel("Task-mean own damage (%)", fontsize=8)
        axs[1].set_title("red = protective candidate (z < −2)\n"
                         "blue = damaging, grey = inert", fontsize=8)

        # P3: the mechanical driver — control damage grows with lesion size
        _fin = np.isfinite(sizes_pc)
        _ctrl_cl = ctrl.mean(axis=0)
        for _mask, _col, _lbl in [(is_input_pc & _fin, "#4292c6", "input"),
                                  (~is_input_pc & _fin, "#e6550d", "hidden")]:
            axs[2].scatter(sizes_pc[_mask], _ctrl_cl[_mask] * 100, s=16,
                           alpha=0.75, color=_col, edgecolors="none", label=_lbl)
        if _fin.sum() > 2:
            _sl, _ic, _r, _pv, _ = linregress(sizes_pc[_fin], _ctrl_cl[_fin] * 100)
            _xf = np.linspace(np.nanmin(sizes_pc), np.nanmax(sizes_pc), 50)
            axs[2].plot(_xf, _sl * _xf + _ic, color="grey", linewidth=0.8)
            axs[2].text(0.05, 0.95, f"r={_r:.2f}", transform=axs[2].transAxes,
                        va="top", fontsize=7)
        axs[2].set_xlabel("Lesion size (# neurons)", fontsize=8)
        axs[2].set_ylabel("Task-mean control damage (%)", fontsize=8)
        axs[2].set_title("Mechanical driver: bigger random draws\n"
                         "hit more critical neurons", fontsize=8)
        axs[2].legend(fontsize=6, frameon=False)

        for ax in axs:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"Protective-cluster dissection [{_pc_vtag}]", fontsize=9)
        fig.tight_layout()
        _path = f"{save_dir}/protective_clusters_{_pc_vtag}_{aname}"
        fig.savefig(f"{_path}.png", dpi=300)
        plt.close(fig)

        with open(f"{_path}.pkl", "wb") as _f:
            pickle.dump({
                "own_damage": own, "ctrl_damage": ctrl, "effect": E_pc,
                "z_own": z_own, "improving": improving, "damaging": damaging,
                "baseline": base_pc, "cluster_names": cn_pc,
                "lesion_sizes": sizes_pc, "tasks": list(all_tasks),
                "task_mean_own": mean_own, "z_cluster": z_cl,
                "protective_candidates": [cn_pc[c] for c in
                                          np.flatnonzero(protective_cand)],
                "n_eval_trials_assumed": _N_EVAL_TRIALS,
            }, _f)
        _cand = [cn_pc[c] for c in np.flatnonzero(protective_cand)] or ["none"]
        print(f"[protective {_pc_vtag}] negative-effect cells: {n_neg}/{E_pc.size} "
              f"— of these, {_frac(inert):.0f}% inert (mechanical), "
              f"{_frac(damaging):.0f}% damaging-but-less-than-control, "
              f"{_frac(improving):.0f}% true improvement; "
              f"protective cluster candidates: {', '.join(_cand)}")

    # Normalized combined lesion effect (input × hidden) for both norm and unnorm
    for vtag in ["norm", "unnorm"]:
        ckey = f"combined_lesion_{vtag}"
        if ckey not in results or not results[ckey]:
            print(f"[combined] skipping {vtag}: key not found in pickle")
            continue

        cdata = results[ckey]
        combined_accs = np.asarray(cdata["combined_accs"], dtype=float)
        combined_random_accs = np.asarray(cdata["combined_random_accs"], dtype=float)
        c_all_tasks = cdata["all_tasks"]
        c_pre_n = cdata["pre_n"]
        c_post_n = cdata["post_n"]

        combined_norm_effect = combined_random_accs - combined_accs  # (n_tasks, pre_n, post_n)
        flat_names = [f"i{pi}_h{qi}" for pi in range(1, c_pre_n + 1) for qi in range(1, c_post_n + 1)]
        combined_norm_effect_flat = combined_norm_effect.reshape(len(c_all_tasks), -1)

        print(f"[combined {vtag}] normalized effect shape: {combined_norm_effect_flat.shape}")

        if c_pre_n * c_post_n <= 100:
            helper.plot_heatmap(
                combined_norm_effect_flat, flat_names, c_all_tasks,
                xlabel=f"Combined Lesion (input, hidden) [{vtag}]", ylabel="Task",
                savename=f"normalized_combined_lesion_{vtag}",
                aname=aname, label="Normalized Accuracy",
                vmin=None, vmax=None, save_dir=save_dir,
            )
        else:
            print(f"[combined {vtag}] Skipping heatmap: {c_pre_n} × {c_post_n} = {c_pre_n * c_post_n} > 100")

    # ── Additivity analysis: is combined(i,j) = single(i) + single(j)? ──
    # For each (input_cluster, hidden_cluster) pair and each task, compare the
    # combined lesion effect to the sum of individual input and hidden lesion effects.
    # Deviation from the diagonal reveals nonlinear interaction:
    #   above diagonal → super-additive (shared computation, synergistic damage)
    #   below diagonal → sub-additive (redundancy, partial compensation)
    _n_input_norm = len([n for n in all_comb_names_lesion_ if n.startswith("i")])
    for vtag in ["norm", "unnorm"]:
        ckey = f"combined_lesion_{vtag}"
        if ckey not in results or not results[ckey]:
            continue
        cdata = results[ckey]
        combined_accs = np.asarray(cdata["combined_accs"], dtype=float)
        combined_random_accs = np.asarray(cdata["combined_random_accs"], dtype=float)
        combined_effect = combined_random_accs - combined_accs  # (n_tasks, pre_n, post_n)
        c_pre_n = cdata["pre_n"]
        c_post_n = cdata["post_n"]

        # Get single-cluster effects for this variant
        if vtag == "norm":
            _sp = select_props  # (n_tasks, pre_n + post_n)
            _n_pre = _n_input_norm
        else:
            if select_props_unnorm is None:
                continue
            _sp = select_props_unnorm
            _n_pre = len([n for n in all_comb_names_unnorm_ if n.startswith("i")])

        _input_effects = _sp[:, :_n_pre]     # (n_tasks, pre_n)
        _hidden_effects = _sp[:, _n_pre:]    # (n_tasks, post_n)

        if _input_effects.shape[1] != c_pre_n or _hidden_effects.shape[1] != c_post_n:
            print(f"[additivity {vtag}] dimension mismatch, skipping")
            continue

        # Build paired arrays: sum of singles vs combined, for all (task, i, j) triples
        sum_singles = []
        combined_vals = []
        for pi in range(c_pre_n):
            for qi in range(c_post_n):
                for ti in range(len(all_tasks)):
                    s = _input_effects[ti, pi] + _hidden_effects[ti, qi]
                    c = combined_effect[ti, pi, qi]
                    sum_singles.append(s)
                    combined_vals.append(c)

        sum_singles = np.array(sum_singles) * 100
        combined_vals = np.array(combined_vals) * 100

        # Also compute task-averaged version (one point per i,j pair)
        sum_singles_mean = []
        combined_vals_mean = []
        for pi in range(c_pre_n):
            for qi in range(c_post_n):
                s = np.mean(_input_effects[:, pi] + _hidden_effects[:, qi]) * 100
                c = np.mean(combined_effect[:, pi, qi]) * 100
                sum_singles_mean.append(s)
                combined_vals_mean.append(c)
        sum_singles_mean = np.array(sum_singles_mean)
        combined_vals_mean = np.array(combined_vals_mean)

        # Figure: 2 panels — left: all (task,i,j) points; right: task-averaged per (i,j) pair
        fig_add, (ax_all, ax_mean) = plt.subplots(1, 2, figsize=(7, 3.3), dpi=300)

        # Left panel: all points
        ax_all.scatter(sum_singles, combined_vals, alpha=0.15, s=8, edgecolors="none", color="steelblue")
        _lim = [min(sum_singles.min(), combined_vals.min()),
                max(sum_singles.max(), combined_vals.max())]
        ax_all.plot(_lim, _lim, color="black", linewidth=0.7, linestyle="--", alpha=0.7)
        slope, intercept, r, p, _ = linregress(sum_singles, combined_vals)
        x_fit = np.linspace(_lim[0], _lim[1], 100)
        ax_all.plot(x_fit, slope * x_fit + intercept, color="tomato", linewidth=1.0)
        p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
        ax_all.text(0.05, 0.95, f"r = {r:.2f}, slope = {slope:.2f}\n{p_str}",
                    transform=ax_all.transAxes, va="top", ha="left", fontsize=7)
        ax_all.set_xlabel("Sum of single effects (%)", fontsize=8)
        ax_all.set_ylabel("Combined effect (%)", fontsize=8)
        ax_all.set_title("All (task, input, hidden) triples", fontsize=8)
        ax_all.spines["top"].set_visible(False)
        ax_all.spines["right"].set_visible(False)
        ax_all.tick_params(labelsize=7)

        # Right panel: task-averaged
        ax_mean.scatter(sum_singles_mean, combined_vals_mean, alpha=0.6, s=20, edgecolors="none", color="steelblue")
        _lim_m = [min(sum_singles_mean.min(), combined_vals_mean.min()),
                  max(sum_singles_mean.max(), combined_vals_mean.max())]
        ax_mean.plot(_lim_m, _lim_m, color="black", linewidth=0.7, linestyle="--", alpha=0.7)
        slope_m, intercept_m, r_m, p_m, _ = linregress(sum_singles_mean, combined_vals_mean)
        x_fit_m = np.linspace(_lim_m[0], _lim_m[1], 100)
        ax_mean.plot(x_fit_m, slope_m * x_fit_m + intercept_m, color="tomato", linewidth=1.0)
        p_str_m = f"p = {p_m:.2e}" if p_m < 0.001 else f"p = {p_m:.3f}"
        ax_mean.text(0.05, 0.95, f"r = {r_m:.2f}, slope = {slope_m:.2f}\n{p_str_m}",
                     transform=ax_mean.transAxes, va="top", ha="left", fontsize=7)
        ax_mean.set_xlabel("Sum of single effects (%)", fontsize=8)
        ax_mean.set_ylabel("Combined effect (%)", fontsize=8)
        ax_mean.set_title("Task-averaged per (input, hidden) pair", fontsize=8)
        ax_mean.spines["top"].set_visible(False)
        ax_mean.spines["right"].set_visible(False)
        ax_mean.tick_params(labelsize=7)

        fig_add.suptitle(f"Additivity test [{vtag}]: combined vs sum of singles", fontsize=9)
        fig_add.tight_layout()
        fig_add.savefig(f"{save_dir}/additivity_{vtag}_{aname}.png", dpi=300)
        plt.close(fig_add)
        print(f"Saved additivity plot [{vtag}]")

    mod_baseline_keys = {"mod_nolesion"}

    # First pass: compute normalized effect and collect by clustering type.
    # Individual heatmaps are not plotted; combined (zero_W | freeze_M) panels are plotted below.
    from collections import defaultdict
    mod_by_type = defaultdict(dict)

    for mod_type_key, mod_data in mod_lesion_results.items():
        base_key, mode, effect_record = _normalized_modulation_effect_record(
            mod_type_key, mod_data, all_tasks)
        mod_select_props = effect_record["effect"]
        all_comb_names_mod_ = effect_record["conditions"]
        normalized_effects["entries"][f"{base_key}__{mode}"] = effect_record

        print(f"[{mod_type_key}] mod_select_props: {mod_select_props.shape}")

        mod_by_type[base_key][mode] = {
            "select_props": mod_select_props,
            "cluster_names": all_comb_names_mod_,
        }

    with open(f"{save_dir}/normalized_lesion_effects_{aname}.pkl", "wb") as handle:
        pickle.dump(normalized_effects, handle)

    # Combined violin plot for all modulation types (zero_W only), one vertical
    # subpanel per registered variant, in registry order.
    _mod_violin_order = list(LESION_MODULATION_TYPES)
    _mod_violin_data = []
    for bk in _mod_violin_order:
        if bk in mod_by_type and "zero_W" in mod_by_type[bk]:
            _mod_violin_data.append((bk, mod_by_type[bk]["zero_W"]))

    if _mod_violin_data:
        n_panels = len(_mod_violin_data)
        max_clusters = max(d["select_props"].shape[1] for _, d in _mod_violin_data)

        # Shared y-limits across all panels
        _all_vals = np.concatenate([d["select_props"].ravel() * 100 for _, d in _mod_violin_data])
        _ylim = (min(_all_vals.min() * 1.1, -1), max(_all_vals.max() * 1.1, 1))

        fig_w = max(4, 0.45 * max_clusters + 1.5)
        fig_v, axes_v = plt.subplots(n_panels, 1, figsize=(fig_w, 1.8 * n_panels), dpi=300)
        if n_panels == 1:
            axes_v = [axes_v]

        for panel_idx, (bk, mode_data) in enumerate(_mod_violin_data):
            ax_v = axes_v[panel_idx]
            props = mode_data["select_props"]
            cnames = mode_data["cluster_names"]
            n_cls = len(cnames)
            type_tag = bk.replace("modulation_all_", "").replace("_", "-")

            violin_data = [props[:, ci] * 100 for ci in range(n_cls)]
            parts = ax_v.violinplot(violin_data, positions=range(n_cls),
                                    showmeans=True, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("steelblue")
                pc.set_alpha(0.6)
            parts["cmeans"].set_color("tomato")
            parts["cmeans"].set_linewidth(1.0)

            for ci in range(n_cls):
                ax_v.scatter(
                    np.full(len(violin_data[ci]), ci), violin_data[ci],
                    color="black", s=5, alpha=0.3, zorder=3,
                )

            ax_v.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
            ax_v.set_xticks(range(n_cls))
            ax_v.set_xticklabels(cnames, rotation=25, ha="right", fontsize=8)
            ax_v.set_ylim(_ylim)
            ax_v.set_ylabel("Effect (%)", fontsize=9)
            ax_v.set_title(f"{type_tag} (zero_W)", fontsize=9)
            ax_v.spines["top"].set_visible(False)
            ax_v.spines["right"].set_visible(False)
            ax_v.tick_params(labelsize=8)

        fig_v.tight_layout()
        fig_v.savefig(f"{save_dir}/normalized_mod_lesion_violin_all_{aname}.png", dpi=300)
        plt.close(fig_v)
        print(f"Saved combined modulation violin plot ({n_panels} panels)")

        # Ranked mean effect comparison: sorted cluster rank vs mean effect per type.
        # Steeper curve = better separation between critical and dispensable clusters.
        _rank_colors = MODULATION_TYPE_COLORS
        _mod_rank_data = {
            bk.replace("modulation_all_", "").replace("_", "-"): d["select_props"].mean(axis=0) * 100
            for bk, d in _mod_violin_data
        }
        _plot_ranked_effect(_mod_rank_data, _rank_colors,
                            "Ranked cluster importance (modulation)",
                            f"{save_dir}/normalized_mod_lesion_ranked_{aname}.png")

        # Cluster size vs normalized lesion effect
        # Tests whether larger clusters are more important after size-matching control.
        _size_colors = MODULATION_TYPE_COLORS
        fig_size, ax_size = plt.subplots(figsize=(4.5, 3.5), dpi=300)
        for bk, mode_data in _mod_violin_data:
            type_tag = bk.replace("modulation_all_", "").replace("_", "-")
            mean_per_cluster = mode_data["select_props"].mean(axis=0) * 100
            # Get cluster sizes from the lesion pickle
            _mod_rkey = f"{bk}__zero_W"
            if _mod_rkey in mod_lesion_results:
                _col_cls = mod_lesion_results[_mod_rkey]["mod_col_clusters"]
                _sorted_ids = sorted(_col_cls.keys())
                sizes = np.array([len(_col_cls[cid]) for cid in _sorted_ids])
                _sl, _ic, _r, _p, _ = linregress(sizes, mean_per_cluster)
                _lbl = f"{type_tag} (r={_r:.2f}, sl={_sl:.2e})"
                ax_size.scatter(sizes, mean_per_cluster, alpha=0.6, s=25,
                                edgecolors="none", color=_size_colors.get(type_tag),
                                label=_lbl)
                _xfit = np.linspace(sizes.min(), sizes.max(), 50)
                ax_size.plot(_xfit, _sl * _xfit + _ic,
                             color=_size_colors.get(type_tag), linewidth=0.8, alpha=0.7)
        ax_size.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_size.set_xlabel("Cluster size (# synapses)", fontsize=8)
        ax_size.set_ylabel("Mean normalized effect (%)", fontsize=8)
        ax_size.set_title("Cluster size vs functional importance", fontsize=9)
        ax_size.legend(fontsize=7, frameon=False)
        ax_size.spines["top"].set_visible(False)
        ax_size.spines["right"].set_visible(False)
        ax_size.tick_params(labelsize=7)
        fig_size.tight_layout()
        fig_size.savefig(f"{save_dir}/normalized_mod_lesion_size_vs_effect_{aname}.png", dpi=300)
        plt.close(fig_size)
        print("Saved modulation cluster size vs effect plot")

        # Histogram for modulation (cluster-averaged + task-specific + summary stats)
        _mod_hist_colors = MODULATION_TYPE_COLORS
        _all_mod_means = np.concatenate([d["select_props"].mean(axis=0) * 100 for _, d in _mod_violin_data])
        _mod_hist_bins = np.linspace(_all_mod_means.min(), _all_mod_means.max(), 15)
        _all_mod_indiv = np.concatenate([d["select_props"].ravel() * 100 for _, d in _mod_violin_data])
        _mod_hist_bins_indiv = np.linspace(_all_mod_indiv.min(), _all_mod_indiv.max(), 25)

        fig_mhist, (ax_mmean, ax_mindiv, ax_stats) = plt.subplots(
            1, 3, figsize=(11, 3), dpi=300, gridspec_kw={"width_ratios": [1, 1, 0.7]})

        # Left: cluster-averaged
        for bk, mode_data in _mod_violin_data:
            type_tag = bk.replace("modulation_all_", "").replace("_", "-")
            mean_per_cluster = mode_data["select_props"].mean(axis=0) * 100
            _mean = np.mean(mean_per_cluster)
            _med = np.median(mean_per_cluster)
            _lbl = f"{type_tag} (μ={_mean:.2f}, md={_med:.2f})"
            ax_mmean.hist(mean_per_cluster, bins=_mod_hist_bins, alpha=0.5,
                          label=_lbl, color=_mod_hist_colors.get(type_tag, None))
        ax_mmean.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_mmean.set_xlabel("Mean effect per cluster (%)", fontsize=8)
        ax_mmean.set_ylabel("# Clusters", fontsize=8)
        ax_mmean.set_title("Cluster-averaged", fontsize=8)
        ax_mmean.legend(fontsize=5.5, frameon=False)
        ax_mmean.spines["top"].set_visible(False)
        ax_mmean.spines["right"].set_visible(False)
        ax_mmean.tick_params(labelsize=7)

        # Middle: task-specific (all individual values)
        for bk, mode_data in _mod_violin_data:
            type_tag = bk.replace("modulation_all_", "").replace("_", "-")
            all_vals = mode_data["select_props"].ravel() * 100
            ax_mindiv.hist(all_vals, bins=_mod_hist_bins_indiv, alpha=0.5,
                           label=type_tag, color=_mod_hist_colors.get(type_tag, None))
        ax_mindiv.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_mindiv.set_xlabel("Effect per (task, cluster) (%)", fontsize=8)
        ax_mindiv.set_ylabel("# (task, cluster) pairs", fontsize=8)
        ax_mindiv.set_title("Task-specific", fontsize=8)
        ax_mindiv.legend(fontsize=5.5, frameon=False)
        ax_mindiv.spines["top"].set_visible(False)
        ax_mindiv.spines["right"].set_visible(False)
        ax_mindiv.tick_params(labelsize=7)

        # Right: summary statistics (std and %>0) as grouped bars
        _mod_type_tags = []
        _mod_stds = []
        _mod_pct_pos = []
        for bk, mode_data in _mod_violin_data:
            type_tag = bk.replace("modulation_all_", "").replace("_", "-")
            all_vals = mode_data["select_props"].ravel() * 100
            _mod_type_tags.append(type_tag)
            _mod_stds.append(np.std(all_vals))
            _mod_pct_pos.append((all_vals > 0).mean() * 100)

        _x_stats = np.arange(len(_mod_type_tags))
        _bar_w = 0.35
        ax_stats_twin = ax_stats.twinx()
        bars1 = ax_stats.bar(_x_stats - _bar_w / 2, _mod_stds, _bar_w,
                             color="steelblue", alpha=0.7, label="Std (%)")
        bars2 = ax_stats_twin.bar(_x_stats + _bar_w / 2, _mod_pct_pos, _bar_w,
                                   color="tomato", alpha=0.7, label="% > 0")
        ax_stats.set_xticks(_x_stats)
        ax_stats.set_xticklabels(_mod_type_tags, rotation=25, ha="right", fontsize=7)
        ax_stats.set_ylabel("Std (%)", fontsize=8, color="steelblue")
        ax_stats_twin.set_ylabel("% > 0", fontsize=8, color="tomato")
        ax_stats.set_title("Informativeness", fontsize=8)
        ax_stats.spines["top"].set_visible(False)
        ax_stats_twin.spines["top"].set_visible(False)
        ax_stats.tick_params(labelsize=7)
        ax_stats_twin.tick_params(labelsize=7)
        ax_stats.legend(loc="upper left", fontsize=6, frameon=False)
        ax_stats_twin.legend(loc="upper right", fontsize=6, frameon=False)

        fig_mhist.tight_layout()
        fig_mhist.savefig(f"{save_dir}/normalized_mod_lesion_hist_mean_{aname}.png", dpi=300)
        plt.close(fig_mhist)
        print("Saved modulation histogram (cluster-averaged + task-specific + stats)")

    # Pairwise ARI between modulation clustering types
    # Uses the cluster assignments from the lesion pickle (mod_col_clusters)
    # to compute adjusted Rand index — measures how similar two clusterings are.
    if len(mod_by_type) >= 2:
        from sklearn.metrics import adjusted_rand_score
        import seaborn as _sns_ari

        _ari_types = []
        _ari_labels_lst = []
        for mod_result_key, mod_data in mod_lesion_results.items():
            if "zero_W" not in mod_result_key:
                continue
            base_key = mod_result_key.rsplit("__", 1)[0]
            type_tag = base_key.replace("modulation_all_", "").replace("_", "-")
            col_clusters = mod_data["mod_col_clusters"]
            # Reconstruct full label array from col_clusters dict
            max_idx = max(max(v) for v in col_clusters.values())
            labels = np.zeros(max_idx + 1, dtype=int)
            for lab, idxs in col_clusters.items():
                labels[np.array(idxs)] = lab
            _ari_types.append(type_tag)
            _ari_labels_lst.append(labels)

        _n_ari = len(_ari_types)
        if _n_ari >= 2:
            _ari_mat = np.full((_n_ari, _n_ari), np.nan)
            for i in range(1, _n_ari):
                for j in range(i):
                    ari = adjusted_rand_score(_ari_labels_lst[i], _ari_labels_lst[j])
                    _ari_mat[i, j] = ari

            # Plot modulation ARI heatmap only
            _ari_mask = np.triu(np.ones((_n_ari, _n_ari), dtype=bool), k=0)
            fig_ari, ax_ari = plt.subplots(figsize=(4, 3.5), dpi=300)
            hm_ari = _sns_ari.heatmap(
                _ari_mat, mask=_ari_mask, annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=0.0, vmax=1.0, center=0.5,
                xticklabels=_ari_types, yticklabels=_ari_types,
                cbar_kws={"label": "ARI", "shrink": 0.75, "aspect": 20},
                linewidths=0.5, linecolor="white", square=True, ax=ax_ari,
                annot_kws={"fontsize": 10, "fontweight": "bold"},
            )
            ax_ari.set_xticklabels(_ari_types, rotation=25, ha="right", fontsize=8)
            ax_ari.set_yticklabels(_ari_types, rotation=0, fontsize=8)
            ax_ari.set_title("Pairwise ARI (modulation)", fontsize=9)
            ax_ari.tick_params(axis="both", length=1.5, width=0.5)
            for spine in ax_ari.spines.values():
                spine.set_linewidth(0.5)
            cbar = hm_ari.collections[0].colorbar
            cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
            cbar.ax.yaxis.label.set_size(8)
            cbar.outline.set_linewidth(0.5)
            fig_ari.tight_layout()
            fig_ari.savefig(f"{save_dir}/clustering_ari_{aname}.png", dpi=300)
            plt.close(fig_ari)
            print(f"Saved modulation ARI comparison ({_n_ari} types)")

    # Second pass: for each clustering type that has both lesion modes,
    # plot side-by-side heatmaps and a scatter comparison.
    import seaborn as sns

    for base_key, modes_dict in mod_by_type.items():
        base_tag = base_key.replace("modulation_all_", "").replace("_", "-")

        # If only one mode, plot a single heatmap and skip comparison
        if len(modes_dict) < 2 or "zero_W" not in modes_dict or "freeze_M" not in modes_dict:
            for mode, mode_data in modes_dict.items():
                mode_tag = mode.replace("_", "-")
                helper.plot_heatmap(
                    mode_data["select_props"], mode_data["cluster_names"], all_tasks,
                    xlabel=f"Modulation Lesion ({mode})", ylabel="Task",
                    savename=f"normalized_mod_lesion_{base_tag}_{mode_tag}",
                    aname=aname, label="Normalized Accuracy",
                    vmin=None, vmax=None, save_dir=save_dir,
                )
            continue

        zw = modes_dict["zero_W"]["select_props"]
        fm = modes_dict["freeze_M"]["select_props"]
        cluster_names = modes_dict["zero_W"]["cluster_names"]

        base_tag = base_key.replace("modulation_all_", "").replace("_", "-")

        # --- Side-by-side heatmaps (zero_W | freeze_M) with shared color scale ---
        abs_max = max(np.nanmax(np.abs(zw)), np.nanmax(np.abs(fm))) * 100
        vmin_shared, vmax_shared = -abs_max, abs_max

        n_tasks_ = len(all_tasks)
        n_conds_ = len(cluster_names)
        panel_w = max(3, 0.35 * n_conds_ + 1.4)
        fig_h = max(3, 0.35 * n_tasks_ + 1.2)
        fig_hm, axes_hm = plt.subplots(
            1, 2, figsize=(panel_w * 2 + 1.0, fig_h), dpi=300,
        )

        for ax_hm, mat, mode_label in [
            (axes_hm[0], zw, "zero_W"),
            (axes_hm[1], fm, "freeze_M"),
        ]:
            hm = sns.heatmap(
                mat * 100, cmap="RdBu_r",
                vmin=vmin_shared, vmax=vmax_shared, center=0.0,
                annot=False,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Normalized effect (%)", "shrink": 0.75,
                           "pad": 0.03, "aspect": 25},
                xticklabels=cluster_names, yticklabels=all_tasks, ax=ax_hm,
            )
            ax_hm.set_xticklabels(cluster_names, rotation=25, ha="right", fontsize=7)
            ax_hm.set_yticklabels(all_tasks, rotation=0, fontsize=7)
            ax_hm.set_xlabel("Modulation Cluster", fontsize=8)
            ax_hm.set_ylabel("Task", fontsize=8)
            ax_hm.set_title(mode_label, fontsize=9)
            ax_hm.tick_params(axis="both", length=1.5, pad=2, width=0.5)
            for spine in ax_hm.spines.values():
                spine.set_linewidth(0.5)
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
            cbar.ax.yaxis.label.set_size(7)
            cbar.outline.set_linewidth(0.5)

        fig_hm.suptitle(f"Normalized modulation lesion effect — {base_tag}", fontsize=10)
        fig_hm.tight_layout()
        _hm_path = f"{save_dir}/normalized_mod_lesion_{base_tag}_combined_heatmap_{aname}"
        fig_hm.savefig(f"{_hm_path}.png", dpi=300)
        plt.close(fig_hm)
        print(f"Saved combined heatmap for {base_key}")

        # --- Scatter: zero_W vs freeze_M ---
        x = zw.ravel()
        y = fm.ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        slope, intercept, r, p, _ = linregress(x, y)

        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)
        ax.scatter(x, y, alpha=0.5, s=20, edgecolors="none", color="steelblue")

        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.2)

        p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.05, 0.95, f"r = {r:.2f}, slope = {slope:.2f}\n{p_str}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)

        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

        ax.set_xlabel("zero_W (normalized effect)")
        ax.set_ylabel("freeze_M (normalized effect)")
        ax.set_title(f"{base_tag}: zero_W vs freeze_M")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(f"{save_dir}/normalized_mod_lesion_compare_{base_tag}_{aname}.png", dpi=300)
        plt.close(fig)
        print(f"Saved comparison scatter for {base_key}")

    # ══════════════════════════════════════════════════════════════════
    # Plasticity-dependence decomposition (module docstring #9).
    # share(task, cluster) = freeze_M effect / zero_W effect — the fraction
    # of a synapse cluster's contribution to a task that flows through the
    # plastic channel M (freeze_M removes only plasticity, W stays), rather
    # than the static wiring. share ≈ 1: the contribution IS the plasticity;
    # share ≈ 0: static wiring suffices; share > 1: frozen plasticity is
    # worse than removing the synapses altogether.
    #
    # The ratio is computed ONLY on cells whose zero_W effect is significant
    # (z against the stored random-control repeats, one-sided BH-FDR q=0.05
    # per clustering type) — this keeps the denominator away from zero,
    # where the ratio is meaningless.
    #
    # Testable MPN prediction: memory-family tasks (delay/dm/dms/dmc — the
    # tasks whose working memory must be held in M) carry a higher
    # plasticity share than reaction-family tasks (fd/react). Tested with a
    # one-sided Mann-Whitney U on the per-task median shares (task level,
    # the conservative unit — cells within a task are correlated).
    # ══════════════════════════════════════════════════════════════════
    def _task_family(t):
        return ("memory"
                if ("delay" in t or t.startswith("dms") or t.startswith("dmc"))
                else "reaction")

    _FAM_COLORS = {"reaction": "#d95f02", "memory": "#1b9e77"}

    _mod_sig = {}   # zero_W significance per modulation type, for task specificity
    for _pd_type in LESION_MODULATION_TYPES:
        zw = mod_lesion_results.get(f"{_pd_type}__zero_W")
        fm = mod_lesion_results.get(f"{_pd_type}__freeze_M")
        if zw is None or fm is None:
            continue
        _names_pd = zw["all_comb_names_mod"]
        _keep_pd = [k for k, n in enumerate(_names_pd) if n not in mod_baseline_keys]
        _cids = [int(_names_pd[k].replace("mod_c", "")) for k in _keep_pd]

        # zero_W significance from its stored raw control repeats
        _acc_zw = np.asarray(zw["modtask_accs"], float)[:, _keep_pd]
        _raw_zw = np.asarray(zw["modrandomtask_accs_raw"], float)[:, _keep_pd, :]
        E_zw = _raw_zw.mean(axis=2) - _acc_zw                    # (T, C)
        _std_zw = np.maximum(_raw_zw.std(axis=2), 1e-4)
        sig_zw = _bh_fdr_mask(gauss_norm.sf(E_zw / _std_zw), q=0.05)

        # freeze_M effect (numerator; needs no own significance test)
        _acc_fm = np.asarray(fm["modtask_accs"], float)[:, _keep_pd]
        _rnd_fm = np.asarray(fm["modrandomtask_accs"], float)[:, _keep_pd]
        E_fm = _rnd_fm - _acc_fm                                 # (T, C)

        share = np.full_like(E_zw, np.nan)
        share[sig_zw] = E_fm[sig_zw] / E_zw[sig_zw]              # sig ⇒ E_zw > 0
        _mod_sig[_pd_type] = {
            "sig": sig_zw, "cluster_ids": list(_cids),
            "unresponsive_label": _mod_unresponsive_label(
                zw, legacy_last=("unnormalized" in _pd_type))}

        n_sig_pd = int(sig_zw.sum())
        type_tag = _pd_type.replace("modulation_all_", "").replace("_", "-")
        if n_sig_pd < 5:
            print(f"[plasticity-share] {type_tag}: only {n_sig_pd} significant "
                  "cells, skipping")
            continue

        _fam = np.array([_task_family(t) for t in all_tasks])
        _order_t = ([i for i in range(len(all_tasks)) if _fam[i] == "reaction"]
                    + [i for i in range(len(all_tasks)) if _fam[i] == "memory"])
        _n_react = int((_fam == "reaction").sum())

        # Per-task median share over that task's significant cells
        task_med = np.array([
            np.nanmedian(share[t]) if np.isfinite(share[t]).any() else np.nan
            for t in range(len(all_tasks))])
        _mem_vals = task_med[(_fam == "memory") & np.isfinite(task_med)]
        _rea_vals = task_med[(_fam == "reaction") & np.isfinite(task_med)]
        if len(_mem_vals) >= 2 and len(_rea_vals) >= 2:
            _U, _p_mwu = mannwhitneyu(_mem_vals, _rea_vals, alternative="greater")
        else:
            _U, _p_mwu = np.nan, np.nan

        fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=300,
                                gridspec_kw={"width_ratios": [1.4, 1.2, 0.7]})

        # P1: share heatmap, non-significant cells masked, families grouped
        _sh_ord = share[_order_t]
        sns.heatmap(_sh_ord, ax=axs[0], cmap="viridis", vmin=0, vmax=1.25,
                    mask=np.isnan(_sh_ord),
                    xticklabels=[f"c{c}" for c in _cids],
                    yticklabels=[all_tasks[i] for i in _order_t],
                    cbar_kws={"label": "Plasticity share", "shrink": 0.85})
        axs[0].axhline(_n_react, color="white", linewidth=2.0)
        axs[0].axhline(_n_react, color="black", linewidth=0.6)
        for _ti, _t_idx in enumerate(_order_t):
            axs[0].get_yticklabels()[_ti].set_color(_FAM_COLORS[_fam[_t_idx]])
        axs[0].tick_params(labelsize=6)
        axs[0].set_title(f"freeze_M / zero_W on significant cells "
                         f"({n_sig_pd} cells)\nreaction family above line, "
                         "memory below", fontsize=8)

        # P2: per-task cell distributions + medians, family-colored
        for _pos, _t_idx in enumerate(_order_t):
            _vals = share[_t_idx][np.isfinite(share[_t_idx])]
            _col = _FAM_COLORS[_fam[_t_idx]]
            if _vals.size:
                axs[1].scatter(np.full(_vals.size, _pos)
                               + np.linspace(-0.15, 0.15, _vals.size),
                               _vals, s=9, alpha=0.55, color=_col,
                               edgecolors="none")
                axs[1].scatter([_pos], [np.median(_vals)], marker="_",
                               s=220, color="black", linewidths=1.6, zorder=3)
        axs[1].axhline(1.0, color="grey", linestyle="--", linewidth=0.6, alpha=0.6)
        axs[1].axhline(0.0, color="grey", linewidth=0.5, alpha=0.6)
        axs[1].set_xticks(range(len(_order_t)))
        axs[1].set_xticklabels([all_tasks[i] for i in _order_t],
                               rotation=45, ha="right", fontsize=6)
        for _pos, _t_idx in enumerate(_order_t):
            axs[1].get_xticklabels()[_pos].set_color(_FAM_COLORS[_fam[_t_idx]])
        axs[1].set_ylabel("Plasticity share", fontsize=8)
        axs[1].set_title("Per-task shares (dash = task median)", fontsize=8)

        # P3: family comparison at the task level (one point per task)
        for _fi, _fname in enumerate(["reaction", "memory"]):
            _vals = task_med[(_fam == _fname) & np.isfinite(task_med)]
            axs[2].scatter(np.full(_vals.size, _fi)
                           + np.linspace(-0.08, 0.08, max(_vals.size, 1))[:_vals.size],
                           _vals, s=28, alpha=0.8, color=_FAM_COLORS[_fname],
                           edgecolors="black", linewidths=0.3)
            if _vals.size:
                axs[2].hlines(np.median(_vals), _fi - 0.2, _fi + 0.2,
                              color="black", linewidth=1.6)
        axs[2].set_xticks([0, 1])
        axs[2].set_xticklabels(["reaction\n(fd/react)", "memory\n(delay/dm)"],
                               fontsize=7)
        axs[2].set_xlim(-0.5, 1.5)
        axs[2].set_ylabel("Task median share", fontsize=8)
        _p_str = (f"p={_p_mwu:.3f}" if np.isfinite(_p_mwu) else "p=n/a")
        axs[2].set_title(f"memory > reaction?\nMWU one-sided {_p_str}", fontsize=8)

        for ax in axs[1:]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"Plasticity-dependence decomposition — {type_tag}",
                     fontsize=9)
        fig.tight_layout()
        _path = f"{save_dir}/plasticity_share_{type_tag}_{aname}"
        fig.savefig(f"{_path}.png", dpi=300)
        plt.close(fig)

        with open(f"{_path}.pkl", "wb") as _f:
            pickle.dump({
                "share": share, "sig_zero_w": sig_zw,
                "effect_zero_w": E_zw, "effect_freeze_m": E_fm,
                "cluster_ids": _cids, "tasks": list(all_tasks),
                "task_family": list(_fam), "task_median_share": task_med,
                "mwu_U": _U, "mwu_p_one_sided": _p_mwu,
                "mod_type": _pd_type, "fdr_q": 0.05,
            }, _f)
        print(f"[plasticity-share] {type_tag}: {n_sig_pd} sig cells; "
              f"overall median share={np.nanmedian(share):.2f}; "
              f"memory={np.nanmedian(_mem_vals) if _mem_vals.size else np.nan:.2f} "
              f"vs reaction={np.nanmedian(_rea_vals) if _rea_vals.size else np.nan:.2f}; "
              f"MWU one-sided {_p_str}")

    # ── Overmembership vs lesion difference ──
    def plot_overmembership_vs_lesion_diff(
        results, cluster_info_mod, cluster_info, variant, mod_type_key,
        mod_lesion_mode, aname, save_dir,
    ):
        """For each (mod_cluster, input_cluster, hidden_cluster) triple, scatter
        overmembership vs the task-profile L1/T distance (mean over tasks of
        |normalized effect difference|) between modulation lesion and combined
        (input+hidden) lesion.

        variant: "norm" or "unnorm"
        mod_type_key: e.g. "modulation_all_normalized"
        mod_lesion_mode: "zero_W" or "freeze_M"
        cluster_info: unused (the unresponsive classes are read from the OM
            cache's recorded indices); kept for call compatibility
        """
        ckey = f"combined_lesion_{variant}"
        if ckey not in results or not results[ckey]:
            print(f"[om_vs_lesion] skipping {variant}: combined lesion data not found")
            return
        mod_result_key = f"{mod_type_key}__{mod_lesion_mode}"
        if mod_result_key not in results["mod_lesion"]:
            print(f"[om_vs_lesion] skipping: {mod_result_key} not found in mod_lesion")
            return
        if mod_type_key not in cluster_info_mod:
            print(f"[om_vs_lesion] skipping: {mod_type_key} not in cluster_info_mod")
            return
        # Find the fixed-k overmembership key dynamically
        _mod_keys = cluster_info_mod[mod_type_key]
        _fk_ga_keys = [k for k in _mod_keys if k.startswith("global_assignment_fixed_k")]
        if _fk_ga_keys:
            ga = _mod_keys[_fk_ga_keys[0]]
        else:
            ga = _mod_keys.get("global_assignment")
        if ga is None:
            print(f"[om_vs_lesion] skipping: no overmembership data for {mod_type_key}")
            return

        # --- Overmembership data ---
        om_stack = ga["om_stack"]                    # (N_cls_om, n_in, n_hid)
        all_choice_order = ga["all_choice_order"]    # list of cluster IDs sorted by size desc
        n_in = ga["n_in"]
        n_hid = ga["n_hid"]
        om_id_to_idx = {cid: idx for idx, cid in enumerate(all_choice_order)}

        # --- Identify unresponsive cluster indices (0-based) to exclude ---
        # Read from the OM cache's explicit unresponsive_{input,hidden}_index
        # (legacy caches: last row/column of the unnormalized grids; none for
        # normalized). Modulation clusters enter below only when their ID is in
        # the saved active OM population and their footprint has supported blocks.
        skip_input = {idx for idx in [_unresponsive_grid_index(ga, "input", variant)]
                      if idx is not None}
        skip_hidden = {idx for idx in [_unresponsive_grid_index(ga, "hidden", variant)]
                       if idx is not None}
        if skip_input or skip_hidden:
            print(f"[om_vs_lesion] excluding unresponsive: input idx={sorted(skip_input)}, "
                  f"hidden idx={sorted(skip_hidden)}")

        # --- Modulation lesion effect (random - cluster), per task ---
        mod_data = results["mod_lesion"][mod_result_key]
        mod_baseline_keys = {"mod_nolesion"}
        all_comb_names_mod = mod_data["all_comb_names_mod"]
        modtask_accs = np.asarray(mod_data["modtask_accs"], dtype=float)
        modrandomtask_accs = np.asarray(mod_data["modrandomtask_accs"], dtype=float)

        mod_effects = {}
        for key_idx, key in enumerate(all_comb_names_mod):
            if key in mod_baseline_keys:
                continue
            cid = int(key.replace("mod_c", ""))
            mod_effects[cid] = modrandomtask_accs[:, key_idx] - modtask_accs[:, key_idx]

        # --- Combined lesion effect (random - cluster), per task ---
        cdata = results[ckey]
        combined_accs = np.asarray(cdata["combined_accs"], dtype=float)
        combined_random_accs = np.asarray(cdata["combined_random_accs"], dtype=float)
        combined_effect = combined_random_accs - combined_accs  # (n_tasks, pre_n, post_n)
        c_pre_n = cdata["pre_n"]
        c_post_n = cdata["post_n"]

        if n_in != c_pre_n or n_hid != c_post_n:
            print(f"[om_vs_lesion] cluster count mismatch: om ({n_in},{n_hid}) vs combined ({c_pre_n},{c_post_n})")
            return

        # --- Build scatter data, keeping the per-cluster (footprint) structure
        # so the permutation test can shuffle cluster -> footprint ownership.
        # y is the task-profile L1/T distance mean_t |mod(t) - combined(t)|,
        # so profile-shape mismatches count even when the task means agree
        # (the old |mean - mean| was blind to those). ---
        mod_profiles = []   # per cluster: (T,) mod lesion effect task profile
        row_om_list = []    # per cluster: masked OM values of its footprint
        row_cm_list = []    # per cluster: (B, T) matching blocks' task profiles
        labels = []

        for cid in sorted(mod_effects.keys()):
            if cid not in om_id_to_idx:
                continue
            om_idx = om_id_to_idx[cid]
            point_mask, _ = _om_point_mask(
                ga, om_idx, skip_input=skip_input, skip_hidden=skip_hidden
            )
            if not np.any(point_mask):
                continue
            mod_profiles.append(np.asarray(mod_effects[cid], float))
            # Boolean indexing is row-major, matching the (pi, qi) loop order
            # this replaces, so labels stay aligned with the flat arrays.
            row_om_list.append(np.asarray(om_stack[om_idx][point_mask], float))
            row_cm_list.append(combined_effect[:, point_mask].T)  # (B, T)
            for pi, qi in np.argwhere(point_mask):
                labels.append(f"m{cid}_i{pi+1}_h{qi+1}")

        if not row_om_list:
            print(f"[om_vs_lesion] no data points to plot")
            return
        if sum(len(values) for values in row_om_list) < 2:
            print(f"[om_vs_lesion] no data points to plot")
            return

        summary = _om_scatter_summary(mod_profiles, row_om_list, row_cm_list)
        om_vals, lesion_diffs = summary["om_vals"], summary["lesion_diffs"]
        association, medians = summary["association"], summary["binned_medians"]
        rho, p_perm = association["rho"], association["p_perm"]

        fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
        ax.scatter(om_vals, lesion_diffs, alpha=0.25, s=20, edgecolors="none", color="steelblue")
        ax.plot(medians["x"], medians["y"], "o-", color="tomato", linewidth=1.2,
                markersize=4, label="Binned median")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", frameon=False, fontsize=7)

        _pp_str = (f"p_perm = {p_perm:.3f}" if np.isfinite(p_perm) else "p_perm = n/a")
        ax.text(0.05, 0.95,
            f"Spearman rho = {rho:.2f}\n"
                f"{_pp_str} ({OM_N_PERM} perms, {len(mod_profiles)} clusters)\n"
            f"n = {len(om_vals)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)

        ax.set_xlabel("Over-membership")
        ax.set_ylabel("Task-profile L1 distance (mean |Δ| over tasks)\n(mod cluster vs combined input+hidden)")
        type_tag = mod_type_key.replace("modulation_all_", "").replace("_", "-")
        mode_tag = mod_lesion_mode.replace("_", "-")
        ax.set_title(f"OM vs lesion diff — {type_tag} {mode_tag} [{variant}]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        savepath = f"{save_dir}/om_vs_lesion_diff_{type_tag}_{mode_tag}_{variant}_{aname}.png"
        fig.savefig(savepath, dpi=300)
        plt.close(fig)
        print(f"[om_vs_lesion] saved: {savepath}")

        # Save the scatter data so the figure is reproducible without re-deriving
        # the OM × combined-lesion matching from the two upstream pickles.
        data_path = f"{save_dir}/om_vs_lesion_diff_{type_tag}_{mode_tag}_{variant}_{aname}.pkl"
        with open(data_path, "wb") as _f:
            pickle.dump({
                "schema_version": 2,
                **summary,
                "aname": aname,
                "labels": labels,
                "y_definition": "task-profile L1/T: mean_t |mod_effect(t) - combined_effect(t)|",
                "mod_type_key": mod_type_key,
                "mod_lesion_mode": mod_lesion_mode,
                "variant": variant,
                "min_expected": OM_MIN_EXPECTED,
                "skip_input": sorted(skip_input),
                "skip_hidden": sorted(skip_hidden),
            }, _f)
        print(f"[om_vs_lesion] saved data: {data_path}")

    def _plot_om_vs_lesion_combined(results, cluster_info_mod, cluster_info, variant,
                                    base_key, modes, aname, save_dir):
        """Plot zero_W and freeze_M overmembership vs lesion diff side-by-side.

        The per-mode scatter data is rebuilt inline with the same derivation
        as plot_overmembership_vs_lesion_diff (duplicated, not shared — keep
        the two in sync when changing either)."""
        type_tag = base_key.replace("modulation_all_", "").replace("_", "-")

        # Collect scatter data for each mode
        mode_data_all = {}
        for mode in ["zero_W", "freeze_M"]:
            if mode not in modes:
                continue
            mod_result_key = f"{base_key}__{mode}"
            if mod_result_key not in results["mod_lesion"]:
                continue
            if base_key not in cluster_info_mod:
                continue
            _mod_keys = cluster_info_mod[base_key]
            _fk_ga_keys = [k for k in _mod_keys if k.startswith("global_assignment_fixed_k")]
            if _fk_ga_keys:
                ga = _mod_keys[_fk_ga_keys[0]]
            else:
                ga = _mod_keys.get("global_assignment")
            if ga is None:
                continue

            om_stack = ga["om_stack"]
            all_choice_order = ga["all_choice_order"]
            n_in = ga["n_in"]
            n_hid = ga["n_hid"]
            om_id_to_idx = {cid: idx for idx, cid in enumerate(all_choice_order)}

            skip_input = {idx for idx in [_unresponsive_grid_index(ga, "input", variant)]
                          if idx is not None}
            skip_hidden = {idx for idx in [_unresponsive_grid_index(ga, "hidden", variant)]
                           if idx is not None}

            mod_data = results["mod_lesion"][mod_result_key]
            mod_baseline_keys_ = {"mod_nolesion"}
            all_comb_names_mod = mod_data["all_comb_names_mod"]
            modtask_accs = np.asarray(mod_data["modtask_accs"], dtype=float)
            modrandomtask_accs = np.asarray(mod_data["modrandomtask_accs"], dtype=float)

            mod_effects = {}
            for key_idx, key in enumerate(all_comb_names_mod):
                if key in mod_baseline_keys_:
                    continue
                cid = int(key.replace("mod_c", ""))
                mod_effects[cid] = modrandomtask_accs[:, key_idx] - modtask_accs[:, key_idx]

            ckey = f"combined_lesion_{variant}"
            if ckey not in results or not results[ckey]:
                continue
            cdata = results[ckey]
            combined_effect = np.asarray(cdata["combined_random_accs"], dtype=float) - np.asarray(cdata["combined_accs"], dtype=float)
            c_pre_n = cdata["pre_n"]
            c_post_n = cdata["post_n"]
            if n_in != c_pre_n or n_hid != c_post_n:
                continue

            # Keep the per-cluster (footprint) structure for the permutation
            # test — same derivation as plot_overmembership_vs_lesion_diff
            # (y = task-profile L1/T distance, see there).
            mod_profiles = []
            row_om_list = []
            row_cm_list = []
            for cid in sorted(mod_effects.keys()):
                if cid not in om_id_to_idx:
                    continue
                om_idx = om_id_to_idx[cid]
                point_mask, _ = _om_point_mask(
                    ga, om_idx, skip_input=skip_input, skip_hidden=skip_hidden
                )
                if not np.any(point_mask):
                    continue
                mod_profiles.append(np.asarray(mod_effects[cid], float))
                row_om_list.append(np.asarray(om_stack[om_idx][point_mask], float))
                row_cm_list.append(combined_effect[:, point_mask].T)  # (B, T)

            if row_om_list:
                if sum(len(values) for values in row_om_list) >= 2:
                    mode_data_all[mode] = _om_scatter_summary(
                        mod_profiles, row_om_list, row_cm_list)

        if len(mode_data_all) < 2:
            return

        # Panel 3: per-cluster prediction — use OM-weighted combined effect to predict
        # modulation cluster's mean own damage (one point per cluster).
        # For each mod cluster: predicted_effect = sum(OM[i,j] * combined_effect_mean[i,j]) / sum(OM[i,j])
        # Uses zero_W mode for the prediction.
        _pred_x, _pred_y = [], []
        mod_result_key_zw = f"{base_key}__zero_W"
        if mod_result_key_zw in results["mod_lesion"] and base_key in cluster_info_mod:
            _mod_keys_p = cluster_info_mod[base_key]
            _fk_ga_keys_p = [k for k in _mod_keys_p if k.startswith("global_assignment_fixed_k")]
            ga_p = _mod_keys_p[_fk_ga_keys_p[0]] if _fk_ga_keys_p else _mod_keys_p.get("global_assignment")
            if ga_p is not None:
                om_stack_p = ga_p["om_stack"]
                all_choice_order_p = ga_p["all_choice_order"]
                n_in_p, n_hid_p = ga_p["n_in"], ga_p["n_hid"]
                om_id_to_idx_p = {cid: idx for idx, cid in enumerate(all_choice_order_p)}

                ckey_p = f"combined_lesion_{variant}"
                if ckey_p in results and results[ckey_p]:
                    cdata_p = results[ckey_p]
                    comb_eff_p = (np.asarray(cdata_p["combined_random_accs"], dtype=float)
                                  - np.asarray(cdata_p["combined_accs"], dtype=float))
                    comb_mean_p = comb_eff_p.mean(axis=0)  # (pre_n, post_n)

                    mod_data_p = results["mod_lesion"][mod_result_key_zw]
                    _mt_p = np.asarray(mod_data_p["modtask_accs"], dtype=float)
                    _baseline_idx_p = mod_data_p["all_comb_names_mod"].index("mod_nolesion")
                    _base_p = _mt_p[:, _baseline_idx_p]
                    for key_idx, key in enumerate(mod_data_p["all_comb_names_mod"]):
                        if key == "mod_nolesion":
                            continue
                        cid = int(key.replace("mod_c", ""))
                        if cid not in om_id_to_idx_p:
                            continue
                        om_idx = om_id_to_idx_p[cid]
                        point_mask, _ = _om_point_mask(
                            ga_p, om_idx,
                            skip_input=skip_input,
                            skip_hidden=skip_hidden,
                        )
                        if not np.any(point_mask):
                            continue
                        om_profile = np.where(point_mask, om_stack_p[om_idx], 0.0)
                        # Predicted effect: OM-weighted average of combined effects
                        if om_profile.sum() > 0:
                            predicted = (om_profile * comb_mean_p).sum() / om_profile.sum()
                        else:
                            predicted = 0.0
                        actual = (_base_p - _mt_p[:, key_idx]).mean()
                        _pred_x.append(predicted * 100)
                        _pred_y.append(actual * 100)

        _has_pred = len(_pred_x) >= 3

        fig, axes = plt.subplots(1, 3 if _has_pred else 2,
                                 figsize=(10.5 if _has_pred else 7, 3.2), dpi=300)
        for ax, mode in zip(axes[:2], ["zero_W", "freeze_M"]):
            if mode not in mode_data_all:
                ax.set_visible(False)
                continue
            summary = mode_data_all[mode]
            om_vals, lesion_diffs = summary["om_vals"], summary["lesion_diffs"]
            association, medians = summary["association"], summary["binned_medians"]
            rho, p_perm = association["rho"], association["p_perm"]
            n_clusters = association["n_clusters"]

            ax.scatter(om_vals, lesion_diffs, alpha=0.25, s=12, edgecolors="none", color="steelblue")
            ax.plot(medians["x"], medians["y"], "o-", color="tomato", linewidth=1.0,
                    markersize=3, label="Binned median")
            ax.set_ylim(bottom=0)
            ax.legend(loc="lower left", frameon=False, fontsize=6)

            _pp_str = (f"p_perm = {p_perm:.3f}" if np.isfinite(p_perm)
                       else "p_perm = n/a")
            ax.text(0.05, 0.95,
                    f"Spearman rho = {rho:.2f}\n{_pp_str} ({n_clusters} clusters)\n"
                    f"n = {len(om_vals)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7)
            ax.set_xlabel("Over-membership", fontsize=8)
            ax.set_ylabel("Profile L1 dist. (mean |Δ| over tasks)", fontsize=8)
            mode_tag = mode.replace("_", "-")
            ax.set_title(f"{mode_tag}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)

        # Panel 3: predicted vs actual per cluster
        _pred_p_perm = np.nan
        if _has_pred:
            ax_pred = axes[2]
            _pred_x = np.array(_pred_x)
            _pred_y = np.array(_pred_y)
            slope_p, intercept_p, r_p, p_p, _ = linregress(_pred_x, _pred_y)
            _, _pred_p_perm, _ = _om_pred_perm_test(_pred_x, _pred_y)

            ax_pred.scatter(_pred_x, _pred_y, alpha=0.6, s=25, edgecolors="none", color="steelblue")
            _lim_p = [min(_pred_x.min(), _pred_y.min()), max(_pred_x.max(), _pred_y.max())]
            ax_pred.plot(_lim_p, _lim_p, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
            x_fit_p = np.linspace(_pred_x.min(), _pred_x.max(), 100)
            ax_pred.plot(x_fit_p, slope_p * x_fit_p + intercept_p, color="tomato", linewidth=1.0)

            _pp_str_p = (f"p_perm = {_pred_p_perm:.3f}"
                         if np.isfinite(_pred_p_perm) else "p_perm = n/a")
            ax_pred.text(0.05, 0.95,
                         f"r = {r_p:.2f}\n{_pp_str_p}\n"
                         f"n = {len(_pred_x)} (naive p = {p_p:.1e})",
                         transform=ax_pred.transAxes, va="top", ha="left", fontsize=7)
            ax_pred.set_xlabel("OM-predicted effect (%)", fontsize=8)
            ax_pred.set_ylabel("Actual mod own damage (%)", fontsize=8)
            ax_pred.set_title("Per-cluster prediction", fontsize=9)
            ax_pred.spines["top"].set_visible(False)
            ax_pred.spines["right"].set_visible(False)
            ax_pred.tick_params(labelsize=7)

        fig.suptitle(f"OM vs lesion — {type_tag} [{variant}]", fontsize=9)
        fig.tight_layout()
        savepath = f"{save_dir}/om_vs_lesion_diff_{type_tag}_combined_{variant}_{aname}.png"
        fig.savefig(savepath, dpi=300)
        plt.close(fig)
        print(f"[om_vs_lesion] saved combined: {savepath}")
        _perm_summary = ", ".join(
            f"{mode} Spearman p_perm={values['association']['p_perm']:.3f}"
            for mode, values in mode_data_all.items())
        print(f"[om_vs_lesion] {type_tag} [{variant}] permutation "
              f"({OM_N_PERM} perms): {_perm_summary}"
              + (f", prediction p_perm={_pred_p_perm:.3f}"
                 if np.isfinite(_pred_p_perm) else ""))

        # Save per-mode scatter data and the per-cluster prediction so the
        # combined figure and paper_plot can be reproduced directly from this
        # pickle without rebuilding the matching or repeating the test.
        data_path = f"{save_dir}/om_vs_lesion_diff_{type_tag}_combined_{variant}_{aname}.pkl"
        with open(data_path, "wb") as _f:
            pickle.dump({
                "schema_version": 2,
                "mode_data": mode_data_all,
                "aname": aname,
                "prediction": ({"predicted_pct": np.asarray(_pred_x),
                                "actual_pct": np.asarray(_pred_y),
                                "actual_own_damage_pct": np.asarray(_pred_y),
                                "actual_definition": "baseline_minus_lesion",
                                "p_perm": _pred_p_perm}
                               if _has_pred else None),
                "base_key": base_key,
                "variant": variant,
                "min_expected": OM_MIN_EXPECTED,
                "n_perm": OM_N_PERM,
                "perm_side": {"scatter": "one-sided (rho <= rho_obs)",
                              "prediction": "one-sided (r >= r_obs)"},
                "y_definition": "task-profile L1/T: mean_t |mod_effect(t) - combined_effect(t)|",
            }, _f)
        print(f"[om_vs_lesion] saved combined data: {data_path}")

    def _om_profile_prediction(results, cluster_info_mod, variant, base_key,
                               aname, save_dir):
        """Predict each synapse cluster's PER-TASK own-damage profile from
        its OM footprint over the combined-lesion map, and sweep a
        concentration exponent (module docstring #8).

            pred_c(t; α) = Σij w_ij · CE(t, i, j) / Σij w_ij,   w = OM[c]^α

        α = 0 is the uniform, anatomy-free baseline (every cluster gets the
        same predicted profile, so the task-mean regression is undefined
        there and reported as NaN); α = 1 is the plain OM weighting used by
        the om_vs_lesion panel-3 prediction; α → ∞ keeps only the argmax
        block. zero_W lesion mode only, mirroring that panel; unresponsive
        input/hidden classes excluded for the unnorm variant, as elsewhere.
        """
        mod_result_key = f"{base_key}__zero_W"
        if (mod_result_key not in results["mod_lesion"]
                or base_key not in cluster_info_mod):
            return
        _mk = cluster_info_mod[base_key]
        _fk_keys = [k for k in _mk if k.startswith("global_assignment_fixed_k")]
        ga = _mk[_fk_keys[0]] if _fk_keys else _mk.get("global_assignment")
        if ga is None:
            return
        om_stack = np.asarray(ga["om_stack"], float)
        om_id_to_idx = {cid: idx for idx, cid in enumerate(ga["all_choice_order"])}
        n_in, n_hid = ga["n_in"], ga["n_hid"]

        ckey = f"combined_lesion_{variant}"
        if ckey not in results or not results[ckey]:
            return
        cdata = results[ckey]
        if n_in != cdata["pre_n"] or n_hid != cdata["post_n"]:
            print(f"[om-profile] {base_key}: OM grid ({n_in},{n_hid}) ≠ combined "
                  f"grid ({cdata['pre_n']},{cdata['post_n']}), skipping")
            return
        CE = (np.asarray(cdata["combined_random_accs"], float)
              - np.asarray(cdata["combined_accs"], float))       # (T, P, H)
        sel_i = _indices_without(n_in, _unresponsive_grid_index(ga, "input", variant))
        sel_h = _indices_without(n_hid, _unresponsive_grid_index(ga, "hidden", variant))
        CE_s = CE[:, sel_i][:, :, sel_h]                         # (T, P', H')

        mod_data = results["mod_lesion"][mod_result_key]
        _mt = np.asarray(mod_data["modtask_accs"], float)
        _baseline_idx = mod_data["all_comb_names_mod"].index("mod_nolesion")
        _base = _mt[:, _baseline_idx]
        clusters, om_rows, actual = [], [], []
        for key_idx, key in enumerate(mod_data["all_comb_names_mod"]):
            if key == "mod_nolesion":
                continue
            cid = int(key.replace("mod_c", ""))
            if cid not in om_id_to_idx:
                continue
            om_idx = om_id_to_idx[cid]
            point_mask, _ = _om_point_mask(
                ga, om_idx,
                skip_input=set(range(n_in)) - set(sel_i.tolist()),
                skip_hidden=set(range(n_hid)) - set(sel_h.tolist()),
            )
            point_mask = point_mask[np.ix_(sel_i, sel_h)]
            if not np.any(point_mask):
                continue
            W = np.where(point_mask, om_stack[om_idx][np.ix_(sel_i, sel_h)], 0.0)
            if W.sum() <= 0:
                continue
            clusters.append(cid)
            om_rows.append(W)
            actual.append(_base - _mt[:, key_idx])
        if len(clusters) < 3:
            print(f"[om-profile] {base_key}: only {len(clusters)} usable "
                  "clusters, skipping")
            return
        om_rows = np.stack(om_rows)                              # (C, P', H')
        actual = np.stack(actual)                                # (C, T)
        nC = actual.shape[0]

        def _predict(alpha):
            """(C, T) predicted profiles with weights OM^alpha (normalized)."""
            if np.isinf(alpha):
                w = np.zeros_like(om_rows)
                flat = om_rows.reshape(nC, -1)
                w.reshape(nC, -1)[np.arange(nC), flat.argmax(axis=1)] = 1.0
            elif alpha == 0:
                w = np.ones_like(om_rows)
            else:
                w = om_rows ** alpha
            w = w / w.sum(axis=(1, 2), keepdims=True)
            return np.einsum("cij,tij->ct", w, CE_s)

        # ── Analysis 1 (α = 1): per-cluster task-profile prediction ──
        pred1 = _predict(1.0)                                    # (C, T)
        prof_r = np.array([
            pearsonr(pred1[c], actual[c])[0]
            if np.std(pred1[c]) > 1e-12 and np.std(actual[c]) > 1e-12
            else np.nan
            for c in range(nC)])

        # ── Analysis 2: concentration-exponent sweep ──
        alphas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
        sweep = []
        for a in alphas:
            pr = _predict(a)
            r_cell = pearsonr(pr.ravel(), actual.ravel())[0]
            xm, ym = pr.mean(axis=1), actual.mean(axis=1)
            if np.std(xm) > 1e-12:
                _sl, _, r_mag, _, _ = linregress(xm, ym)
            else:
                _sl, r_mag = np.nan, np.nan
            sweep.append({"alpha": a, "r_cell": float(r_cell),
                          "r_mag": float(r_mag) if np.isfinite(r_mag) else np.nan,
                          "slope": float(_sl) if np.isfinite(_sl) else np.nan})
        _finite = [s for s in sweep if np.isfinite(s["r_cell"])]
        best = max(_finite, key=lambda s: s["r_cell"])
        pred_best = _predict(best["alpha"])

        type_tag = base_key.replace("modulation_all_", "").replace("_", "-")
        fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), dpi=300)

        # P1: per-cluster profile r, sorted
        _ord = np.argsort(-np.nan_to_num(prof_r, nan=-2))
        axs[0].bar(np.arange(nC), prof_r[_ord],
                   color=["#2171b5" if v > 0 else "#cb181d" for v in prof_r[_ord]],
                   edgecolor="black", linewidth=0.3)
        axs[0].axhline(np.nanmean(prof_r), color="grey", linestyle="--",
                       linewidth=0.7, label=f"mean={np.nanmean(prof_r):.2f}")
        axs[0].set_xticks(np.arange(nC))
        axs[0].set_xticklabels([f"MC{clusters[c]}" for c in _ord],
                               rotation=60, ha="right", fontsize=5)
        axs[0].set_ylabel("Task-profile r (α=1)", fontsize=8)
        axs[0].set_ylim(-1, 1)
        axs[0].legend(fontsize=6, frameon=False)
        axs[0].set_title("Does the OM footprint predict\nWHICH tasks a cluster serves?",
                         fontsize=8)

        # P2: pooled (cluster, task) cells at α=1
        _r_cell1 = pearsonr(pred1.ravel(), actual.ravel())[0]
        axs[1].scatter(pred1.ravel() * 100, actual.ravel() * 100, s=7,
                       alpha=0.4, color="steelblue", edgecolors="none")
        _lim = [min(pred1.min(), actual.min()) * 100,
                max(pred1.max(), actual.max()) * 100]
        axs[1].plot(_lim, _lim, color="grey", linestyle="--", linewidth=0.6)
        axs[1].set_xlabel("Predicted effect (%)", fontsize=8)
        axs[1].set_ylabel("Actual own damage (%)", fontsize=8)
        axs[1].set_title(f"All (cluster, task) cells, α=1\nr={_r_cell1:.2f}, "
                         f"n={pred1.size}", fontsize=8)

        # P3: alpha sweep
        _x = np.arange(len(alphas))
        _xl = ["0", "0.5", "1", "2", "4", "8", "max"]
        axs[2].plot(_x, [s["r_cell"] for s in sweep], "-o", markersize=4,
                    color="#2171b5", label="cell-level r")
        axs[2].plot(_x, [s["r_mag"] for s in sweep], "-s", markersize=4,
                    color="#6baed6", label="task-mean r")
        axs[2].set_xticks(_x)
        axs[2].set_xticklabels(_xl)
        axs[2].set_xlabel("Concentration exponent α", fontsize=8)
        axs[2].set_ylabel("Prediction r", fontsize=8)
        axs[2].legend(fontsize=6, frameon=False, loc="lower right")
        _tw = axs[2].twinx()
        _tw.plot(_x, [s["slope"] for s in sweep], ":d", markersize=4,
                 color="tomato")
        _tw.axhline(1.0, color="tomato", linestyle="--", linewidth=0.5, alpha=0.5)
        _tw.set_ylabel("Task-mean slope", fontsize=8, color="tomato")
        _tw.tick_params(labelsize=7, colors="tomato")
        axs[2].set_title(f"footprint (α small) vs peak blocks (α large)\n"
                         f"best α={_xl[alphas.index(best['alpha'])]} "
                         f"(cell r={best['r_cell']:.2f})", fontsize=8)

        # P4: task-mean scatter at the best α
        _xm, _ym = pred_best.mean(axis=1) * 100, actual.mean(axis=1) * 100
        axs[3].scatter(_xm, _ym, s=20, alpha=0.7, color="steelblue",
                       edgecolors="none")
        _lim = [min(_xm.min(), _ym.min()), max(_xm.max(), _ym.max())]
        axs[3].plot(_lim, _lim, color="grey", linestyle="--", linewidth=0.6)
        if np.std(_xm) > 1e-12:
            _sl, _ic, _r, _p, _ = linregress(_xm, _ym)
            _xf = np.linspace(_xm.min(), _xm.max(), 50)
            axs[3].plot(_xf, _sl * _xf + _ic, color="tomato", linewidth=1.0)
            axs[3].text(0.05, 0.95, f"r={_r:.2f}\nslope={_sl:.2f}",
                        transform=axs[3].transAxes, va="top", fontsize=7)
        axs[3].set_xlabel("Predicted task-mean effect (%)", fontsize=8)
        axs[3].set_ylabel("Actual task-mean own damage (%)", fontsize=8)
        axs[3].set_title(f"Per-cluster magnitude at best α", fontsize=8)

        for ax in axs:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"OM profile prediction — {type_tag} [{variant}] (zero_W)",
                     fontsize=9)
        fig.tight_layout()
        _path = f"{save_dir}/om_profile_prediction_{type_tag}_{variant}_{aname}"
        fig.savefig(f"{_path}.png", dpi=300)
        plt.close(fig)

        with open(f"{_path}.pkl", "wb") as _f:
            pickle.dump({
                "clusters": clusters, "om_rows": om_rows,
                "actual_profiles": actual, "predicted_profiles_alpha1": pred1,
                "predicted_profiles_best": pred_best,
                "profile_r_alpha1": prof_r,
                "alpha_sweep": sweep, "best_alpha": best["alpha"],
                "tasks": list(all_tasks), "base_key": base_key,
                "variant": variant,
                "actual_definition": "baseline_minus_lesion",
            }, _f)
        print(f"[om-profile] {type_tag} [{variant}]: profile r mean="
              f"{np.nanmean(prof_r):.2f} ({np.nanmean(prof_r > 0) * 100:.0f}%>0); "
              f"cell r(α=1)={_r_cell1:.2f}; best α="
              f"{_xl[alphas.index(best['alpha'])]} (cell r={best['r_cell']:.2f}, "
              f"slope={best['slope']:.2f})")

    # ══════════════════════════════════════════════════════════════════
    # Task specificity of clusters and compositional sharing (docstring #11).
    # Significance masks: unnormalized input/hidden neuron clusters from the
    # causal-dependency block, var-weighted zero_W synapse clusters from the
    # plasticity-share block (both one-sided BH-FDR q=0.05 against control
    # repeats). The unresponsive cluster of each type, identified by its
    # recorded label, is excluded.
    # ══════════════════════════════════════════════════════════════════
    _spec_types = {}
    if "unnorm" in _causal_sig:
        _cs = _causal_sig["unnorm"]
        for side, prefix in (("input", "i"), ("hidden", "h")):
            # drop the unresponsive class, identified by name (see _causal_dependency)
            cols = [k for k, n in enumerate(_cs["cluster_names"])
                    if n.startswith(prefix) and n not in _cs["unresponsive_names"]]
            if len(cols) >= 2:
                _spec_types[side] = {"sig": _cs["sig"][:, cols],
                                     "cluster_labels": [_cs["cluster_names"][k] for k in cols]}
    _spec_mod_key = "modulation_all_var_weighted_unnormalized"
    if _spec_mod_key in _mod_sig:
        _ms = _mod_sig[_spec_mod_key]
        # drop the unresponsive synapse class, identified by its recorded label
        keep = [k for k, cid in enumerate(_ms["cluster_ids"]) if cid != _ms["unresponsive_label"]]
        if len(keep) >= 2:
            _spec_types["modulation"] = {"sig": _ms["sig"][:, keep],
                                         "cluster_labels": [f"c{_ms['cluster_ids'][k]}" for k in keep]}
    if _spec_types:
        _spec = _task_specificity_summary(_spec_types, all_tasks)
        _spec["aname"] = aname
        _spec["modulation_type"] = _spec_mod_key
        _spec["neuron_variant"] = "unnormalized"
        _spec["significance"] = "one-sided z vs control repeats, BH-FDR q=0.05"
        with open(f"{save_dir}/task_specificity_{aname}.pkl", "wb") as _f:
            pickle.dump(_spec, _f)

        _type_colors = {"input": "#2171b5", "hidden": "#cb181d", "modulation": "#e7298a"}
        fig, axs = plt.subplots(1, 1 + len(_spec_types), figsize=(4.2 * (1 + len(_spec_types)), 3.4),
                                dpi=300, squeeze=False)
        ax0 = axs[0, 0]
        n_tasks = len(all_tasks)
        for type_name, entry in _spec["types"].items():
            counts = entry["dispersion"]["counts"]
            hist = np.bincount(counts, minlength=n_tasks + 1) / counts.size
            ax0.plot(range(n_tasks + 1), hist, "o-", markersize=3, linewidth=1,
                     color=_type_colors[type_name],
                     label=f"{type_name} (n={counts.size}; dispersion p="
                           f"{entry['dispersion']['p_perm']:.3f})")
        ax0.set_xlabel("# tasks a cluster significantly impairs")
        ax0.set_ylabel("Fraction of clusters")
        ax0.legend(fontsize=6, frameon=False)
        ax0.set_title("Task specificity (unresponsive clusters excluded)", fontsize=8)
        for ax, (type_name, entry) in zip(axs[0, 1:], _spec["types"].items()):
            sharing = entry["sharing"]
            for pos, name in enumerate(sharing["relations"]):
                rel = sharing["by_relation"][name]
                values = rel["values"][np.isfinite(rel["values"])]
                if values.size:
                    ax.scatter(np.full(values.size, pos) + np.linspace(-0.15, 0.15, values.size),
                               values, s=10, alpha=0.6, color=_type_colors[type_name],
                               edgecolors="none")
                    ax.scatter([pos], [np.nanmedian(values)], marker="_", s=200,
                               color="black", linewidths=1.5, zorder=3)
                    ax.text(pos, 1.02, f"p={rel['p_perm']:.2f}", ha="center",
                            va="bottom", fontsize=5.5)
            ax.set_xticks(range(len(sharing["relations"])))
            ax.set_xticklabels(sharing["relations"], rotation=35, ha="right", fontsize=6)
            ax.set_ylim(-0.02, 1.12)
            ax.set_ylabel("Jaccard overlap of impaired clusters", fontsize=7)
            related = sharing["by_relation"]["related"]
            ax.set_title(f"{type_name}: related pairs mean {related['mean']:.2f}, "
                         f"p={related['p_perm']:.3f}", fontsize=8)
        for ax in axs[0]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/task_specificity_{aname}.png", dpi=300)
        plt.close(fig)
        for type_name, entry in _spec["types"].items():
            disp = entry["dispersion"]
            rel = entry["sharing"]["by_relation"]
            print(f"[task-specificity] {type_name}: tasks/cluster median "
                  f"{np.median(disp['counts']):.0f} (range {disp['counts'].min()}-"
                  f"{disp['counts'].max()}), dispersion var {disp['observed_var']:.2f} vs null "
                  f"{disp['null_var'].mean():.2f} p={disp['p_perm']:.3f}; related-pair Jaccard "
                  f"{rel['related']['mean']:.2f} vs other {rel['other']['mean']:.2f} "
                  f"p={rel['related']['p_perm']:.3f}")
    else:
        print("[task-specificity] no significance masks available, skipping")

    # Load cluster_info and cluster_info_mod for overmembership analysis
    cluster_path = f"./multiple_tasks_analysis/{aname}/cluster_info_{aname}.pkl"
    cluster_mod_path = f"./multiple_tasks_analysis/{aname}/cluster_info_mod_{aname}.pkl"
    if os.path.exists(cluster_mod_path) and os.path.exists(cluster_path):
        with open(cluster_mod_path, "rb") as f:
            cluster_info_mod = pickle.load(f)
        with open(cluster_path, "rb") as f:
            cluster_info = pickle.load(f)

        # Group by (base_key, variant) to plot both modes side-by-side
        from collections import defaultdict
        _om_groups = defaultdict(dict)
        for mod_result_key in results["mod_lesion"]:
            base_key, mode = mod_result_key.rsplit("__", 1)
            if "normalized" in base_key and "unnormalized" not in base_key:
                variant = "norm"
            else:
                variant = "unnorm"
            _om_groups[(base_key, variant)][mode] = mod_result_key

        for (base_key, variant), modes in _om_groups.items():
            if "zero_W" in modes and "freeze_M" in modes:
                # Plot both modes as two subplots in one figure
                _plot_om_vs_lesion_combined(
                    results, cluster_info_mod, cluster_info, variant, base_key,
                    modes, aname, save_dir,
                )
            else:
                for mode in modes:
                    plot_overmembership_vs_lesion_diff(
                        results, cluster_info_mod, cluster_info, variant, base_key, mode,
                        aname, save_dir,
                    )
            _om_profile_prediction(results, cluster_info_mod, variant, base_key,
                                   aname, save_dir)
    else:
        print(f"[om_vs_lesion] cluster pickle(s) not found, skipping")

    # ── Cluster tuning similarity vs lesion-profile similarity (docstring #4) ──
    def plot_cluster_corr_vs_lesion(panels, savesuffix, aname, save_dir,
                                    exclude_last_cluster=False, unresponsive_labels=None):
        """5×N diagnostic figure and the paper-facing scatter cache.

        panels: ordered {name: {"cluster_means": (F, C), "lesion_acc": (T, C),
        "control_raw": (T, C, R), "cluster_labels": [C labels],
        optional "mod_lesion_mode"}}. `unresponsive_labels` (a collection of
        cluster labels, or None) names the unresponsive class to drop in every
        panel; `exclude_last_cluster` is the legacy positional rule. Every
        statistic is computed by `_tuning_vs_lesion_summary`; this function
        only draws and saves.
        Rows: tuning-correlation heatmap, lesion-profile dissimilarity heatmap
        (both over the compared clusters), the scale-free scatter with its
        Spearman rho and label-permutation p, then the L1 supplement: tuning
        cosine vs L1 and summed effect magnitude vs L1.
        """
        n_cols = len(panels)
        fig, axs = plt.subplots(5, n_cols, figsize=(4.5 * n_cols, 18), dpi=300,
                                squeeze=False)
        scatter_save_data = {}

        def _rank_label(association):
            if not np.isfinite(association["rho"]):
                return "constant x or y"
            return (f"rho = {association['rho']:.2f}\n"
                    f"permutation p = {association['p_perm']:.3f} "
                    f"(n = {association['n_clusters']} clusters)")

        for col, (name, panel) in enumerate(panels.items()):
            try:
                summary = _tuning_vs_lesion_summary(
                    panel["cluster_means"], panel["lesion_acc"], panel["control_raw"],
                    panel["cluster_labels"], exclude_last_cluster=exclude_last_cluster,
                    unresponsive_labels=unresponsive_labels)
            except ValueError as error:
                print(f"[cluster_corr_vs_lesion] {name}: {error}; panel skipped")
                for row in range(5):
                    axs[row, col].set_axis_off()
                axs[0, col].set_title(f"{name}: skipped ({error})", fontsize=7)
                continue
            summary["aname"] = aname
            if "mod_lesion_mode" in panel:
                summary["mod_lesion_mode"] = panel["mod_lesion_mode"]
            summary["tuning_profile"] = panel.get("tuning_profile", "cluster mean")
            scatter_save_data[name] = summary

            labels = [str(label).replace("pre_c", "i").replace("post_c", "h")
                      .replace("mod_c", "c") for label in summary["included_clusters"]]
            n = len(labels)
            tril_idx = np.tril_indices(n, k=-1)
            for row, (matrix, cmap, limits, cbar_label, title) in enumerate((
                    (summary["tuning_corr_matrix"], "RdBu_r", (-1, 1), "Pearson r",
                     "tuning profile correlation"),
                    (summary["lesion_profile_dissim_matrix"], "viridis", (0, 2),
                     "1 - r", "lesion profile dissimilarity (z-scored)"))):
                ax = axs[row, col]
                shown = np.full((n, n), np.nan)
                shown[tril_idx] = matrix[tril_idx]
                image = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=limits[0],
                                  vmax=limits[1], origin="upper")
                fig.colorbar(image, ax=ax, shrink=0.8, label=cbar_label)
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(labels, rotation=90, fontsize=6)
                ax.set_yticklabels(labels, fontsize=6)
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Cluster")
                ax.set_title(f"{name}: {title}")

            ax_main = axs[2, col]
            ax_main.scatter(summary["tuning_corr"], summary["lesion_profile_dissim"],
                            alpha=0.6, s=30, edgecolors="none", color="steelblue")
            ax_main.text(0.05, 0.95, _rank_label(summary["association"]),
                         transform=ax_main.transAxes, va="top", ha="left", fontsize=8)
            trend = summary["trend_line"]
            if trend is not None:
                x_line = np.linspace(summary["tuning_corr"].min(), summary["tuning_corr"].max(), 50)
                ax_main.plot(x_line, trend["intercept"] + trend["slope"] * x_line,
                             color="tomato", linewidth=1.2, label="OLS guide")
            ax_main.set_xlabel("Tuning profile correlation")
            ax_main.set_ylabel("Lesion profile dissimilarity (1 - r)")
            ax_main.set_ylim(bottom=0)
            excluded = summary["excluded_clusters"]
            ax_main.set_title(
                f"{name}: {n} clusters compared; excluded "
                f"{len(excluded['last'])} last, {len(excluded['not_significant'])} "
                f"non-significant, {len(excluded['degenerate'])} degenerate", fontsize=7)

            supplement = summary["l1"]
            for row, (x_values, x_label, association) in enumerate((
                    (supplement["tuning_cos_sim"], "Tuning cosine similarity",
                     supplement["association_tuning"]),
                    (supplement["effect_magnitude_sum"],
                     "Summed effect magnitude of the pair",
                     supplement["association_magnitude"])), start=3):
                ax = axs[row, col]
                ax.scatter(x_values, supplement["lesion_l1_dist"], alpha=0.6, s=30,
                           edgecolors="none", color="grey")
                ax.text(0.05, 0.95, _rank_label(association), transform=ax.transAxes,
                        va="top", ha="left", fontsize=8)
                ax.set_xlabel(x_label)
                ax.set_ylabel("Lesion effect L1 distance")
                ax.set_title(f"{name}: L1 supplement", fontsize=8)

            print(f"[cluster_corr_vs_lesion] {name}: rho={summary['association']['rho']:+.2f} "
                  f"p_perm={summary['association']['p_perm']:.3f} over {n} clusters; "
                  f"L1 vs tuning cos rho={supplement['association_tuning']['rho']:+.2f}, "
                  f"L1 vs magnitude rho={supplement['association_magnitude']['rho']:+.2f}")

        fig.tight_layout()
        fig.savefig(f"{save_dir}/cluster_corr_vs_{savesuffix}_{aname}.png", dpi=300)
        plt.close(fig)
        print(f"Saved cluster_corr_vs_{savesuffix}")

        scatter_pkl_path = f"{save_dir}/cluster_corr_vs_{savesuffix}_{aname}.pkl"
        with open(scatter_pkl_path, "wb") as _f:
            pickle.dump(scatter_save_data, _f)
        print(f"Saved scatter data: {scatter_pkl_path}")

    def _neuron_lesion_panels(lesion_key, random_key, cluster_means_by_side):
        """Split neuron lesion accuracies and control repeats into input/hidden panels.

        cluster_means_by_side: {"input": (name, (F, C)), "hidden": (name, (F, C))}.
        Columns follow the saved condition order with baselines removed.
        """
        names = results[lesion_key]["all_comb_names_lesion"]
        acc = np.asarray(results[lesion_key]["ihtask_accs"], dtype=float)
        raw = np.asarray(results[random_key]["ihrandomtask_accs_raw"], dtype=float)
        panels = {}
        for side, prefix in (("input", "pre_c"), ("hidden", "post_c")):
            index = [k for k, key in enumerate(names)
                     if key.startswith(prefix) and key not in baseline_keys]
            panel_name, means = cluster_means_by_side[side]
            if means.shape[1] != len(index):
                raise ValueError(f"{panel_name}: {means.shape[1]} cluster means but "
                                 f"{len(index)} lesion conditions")
            panels[panel_name] = {
                "cluster_means": means, "lesion_acc": acc[:, index],
                "control_raw": raw[:, index, :],
                "cluster_labels": [names[k] for k in index],
            }
        return panels

    # --- Normalized variant ---
    cluster_means_norm = results["cluster_similarity"]["cluster_means"]
    # Keys may be "input_normalized" or "input_normalized_k{N}" depending on FIXED_K
    _input_norm_key = [k for k in cluster_means_norm if k.startswith("input_normalized")][0]
    _hidden_norm_key = [k for k in cluster_means_norm if k.startswith("hidden_normalized")][0]
    plot_cluster_corr_vs_lesion(
        _neuron_lesion_panels("lesion", "random_lesion", {
            "input": (_input_norm_key, np.asarray(cluster_means_norm[_input_norm_key], float)),
            "hidden": (_hidden_norm_key, np.asarray(cluster_means_norm[_hidden_norm_key], float)),
        }),
        "normalized_lesion_effect", aname, save_dir,
    )

    # --- Unnormalized variant ---
    # Cluster means are computed on-the-fly from cluster_info (not in the lesion pickle)
    if ("lesion_unnorm" in results and "random_lesion_unnorm" in results
            and os.path.exists(cluster_path)):
        try:
            cluster_info
        except NameError:
            with open(cluster_path, "rb") as f:
                cluster_info = pickle.load(f)

        _fixed_k_plot = results.get("fixed_k", 20)

        cluster_means_unnorm = {}
        for side in ("input", "hidden"):
            name = f"{side}_unnormalized"
            if name not in cluster_info:
                continue
            ci = cluster_info[name]
            V = ci["cell_vars_rules_sorted_norm"]
            col_clusters = clustering.fixed_k_col_clusters(ci, _fixed_k_plot)
            n_clusters = len(col_clusters)
            cluster_means = np.stack(
                [V[:, col_clusters[c]].mean(axis=1) for c in range(1, n_clusters + 1)],
                axis=1,
            )
            cluster_means_unnorm[side] = (f"{name}_k{_fixed_k_plot}", cluster_means)

        if len(cluster_means_unnorm) == 2:
            plot_cluster_corr_vs_lesion(
                _neuron_lesion_panels("lesion_unnorm", "random_lesion_unnorm",
                                      cluster_means_unnorm),
                "normalized_lesion_effect_unnorm", aname, save_dir,
                unresponsive_labels=_unresponsive_condition_names(
                    results["lesion_unnorm"], legacy_last=True),
            )

    # --- Modulation variant ---
    # For each modulation clustering type × lesion mode, the cluster means come
    # from cell_vars_rules_sorted_norm and the cluster assignments actually
    # used in the lesion; the lesion side uses that mode's raw control repeats.
    if os.path.exists(cluster_mod_path):
        try:
            cluster_info_mod
        except NameError:
            with open(cluster_mod_path, "rb") as f:
                cluster_info_mod = pickle.load(f)

        for mod_type_key, modes_dict in mod_by_type.items():
            if mod_type_key not in cluster_info_mod:
                continue
            mod_ci = cluster_info_mod[mod_type_key]
            V_mod = mod_ci["cell_vars_rules_sorted_norm"]   # (n_features, n_synapses)

            # Use the actual cluster assignments saved in the lesion pickle
            # (guaranteed to match the lesion columns).
            _result_key = f"{mod_type_key}__{next(iter(modes_dict.keys()))}"
            col_clusters_mod = mod_lesion_results[_result_key]["mod_col_clusters"]
            unique_labels = sorted(col_clusters_mod.keys())
            cluster_means_mod = np.stack(
                [V_mod[:, col_clusters_mod[lab]].mean(axis=1) for lab in unique_labels],
                axis=1,
            )  # (n_features, n_mod_clusters)
            cluster_means_mod, _tuning_note = _tuning_profiles_for_variant(
                cluster_means_mod, mod_type_key)

            for mode in modes_dict:
                mod_data = mod_lesion_results[f"{mod_type_key}__{mode}"]
                names_mod = list(mod_data["all_comb_names_mod"])
                index = [k for k, key in enumerate(names_mod)
                         if key not in ("mod_nolesion", "mod_cNone")]
                lesion_labels = [names_mod[k] for k in index]
                if lesion_labels != [f"mod_c{lab}" for lab in unique_labels]:
                    print(f"[mod corr_vs_lesion] cluster order mismatch for "
                          f"{mod_type_key}__{mode}: lesion={lesion_labels}, "
                          f"similarity={unique_labels}, skipping")
                    continue
                if "modrandomtask_accs_raw" not in mod_data:
                    print(f"[mod corr_vs_lesion] {mod_type_key}__{mode} has no stored "
                          "control repeats; rerun lesion.py to enable z-scoring, skipping")
                    continue

                type_tag = mod_type_key.replace("modulation_all_", "").replace("_", "-")
                mode_tag = mode.replace("_", "-")
                mod_name = f"{type_tag}_{mode_tag}"
                # Unnormalized variants carry an unresponsive synapse class; its
                # recorded label (legacy pickles: the last cluster) is dropped.
                # Normalized variants have none and keep every cluster.
                if "unnormalized" in mod_type_key:
                    _unres_mod = _mod_unresponsive_label(mod_data, legacy_last=True)
                    _unres_mod_labels = set() if _unres_mod is None else {f"mod_c{_unres_mod}"}
                else:
                    _unres_mod_labels = None
                plot_cluster_corr_vs_lesion(
                    {mod_name: {
                        "cluster_means": cluster_means_mod,
                        "lesion_acc": np.asarray(mod_data["modtask_accs"], float)[:, index],
                        "control_raw": np.asarray(mod_data["modrandomtask_accs_raw"],
                                                  float)[:, index, :],
                        "cluster_labels": lesion_labels,
                        "mod_lesion_mode": mode,
                        "tuning_profile": _tuning_note,
                    }},
                    f"mod_lesion_effect_{type_tag}_{mode_tag}", aname, save_dir,
                    unresponsive_labels=_unres_mod_labels,
                )

    # ══════════════════════════════════════════════════════════════════
    # Interaction map from the combined lesions (module docstring #7).
    #     I(i, j) = combined_effect(i, j) − single_effect(i) − single_effect(j)
    # per task, in normalized-effect units (random − lesion):
    #     I < 0  sub-additive — the two clusters overlap / are redundant
    #     I > 0  super-additive — synergy (each compensates for the other)
    # The task-averaged map is the primary view: singles were measured at
    # test_n_batch (200) and combined at combined_test_n_batch (100) with
    # independent noise, and averaging over tasks suppresses it.
    # If the modulation OM data is already in memory (loaded by the
    # om_vs_lesion section above), the interaction strength is regressed
    # against each (input, hidden) block's PEAK synapse-cluster enrichment:
    # does anatomical co-location explain functional interaction? The
    # large cluster_info_mod pickle is deliberately NOT re-loaded here.
    # ══════════════════════════════════════════════════════════════════
    try:
        _cim_for_interaction = cluster_info_mod
    except NameError:
        _cim_for_interaction = None

    for vtag, singles, names_f in [
        ("norm", select_props, all_comb_names_lesion_),
        ("unnorm", select_props_unnorm, all_comb_names_unnorm_),
    ]:
        ckey = f"combined_lesion_{vtag}"
        if singles is None or ckey not in results or not results[ckey]:
            print(f"[interaction {vtag}] missing singles or combined data, skipping")
            continue
        cdata = results[ckey]
        c_pre_n, c_post_n = cdata["pre_n"], cdata["post_n"]
        n_pre_s = len([n for n in names_f if n.startswith("i")])
        n_post_s = len(names_f) - n_pre_s
        if n_pre_s != c_pre_n or n_post_s != c_post_n:
            print(f"[interaction {vtag}] single/combined cluster count mismatch "
                  f"({n_pre_s},{n_post_s}) vs ({c_pre_n},{c_post_n}), skipping")
            continue

        CE = (np.asarray(cdata["combined_random_accs"], float)
              - np.asarray(cdata["combined_accs"], float))     # (T, P, H)
        SI = singles[:, :n_pre_s]                               # (T, P)
        SH = singles[:, n_pre_s:]                               # (T, H)
        I_task = CE - SI[:, :, None] - SH[:, None, :]           # (T, P, H)
        I_avg = I_task.mean(axis=0)                             # (P, H)

        # ── Saturation control ────────────────────────────────────────
        # A high-effect cluster can saturate a task (accuracy at floor); on
        # a bounded scale ANY second lesion then looks sub-additive, pathway
        # overlap or not. Two complementary controls:
        #   (a) headroom-restricted cells — keep (task, i, j) cells whose two
        #       single effects are both damaging AND together claim at most
        #       half of the task's available range: saturation cannot
        #       explain sub-additivity there;
        #   (b) multiplicative (survival) baseline — expected combined damage
        #       under independence on the bounded scale is
        #       1 − (1 − d_i)(1 − d_j) with d = effect / headroom, so
        #       I_mult = CE − headroom · (1 − (1 − d_i)(1 − d_j)); pure
        #       saturation gives I_mult ≈ 0, genuine overlap stays negative.
        # The floor is the worst accuracy OBSERVED in this variant's combined
        # grid — an upper bound on the true floor, which UNDERestimates the
        # headroom and OVERcorrects toward saturation: redundancy surviving
        # this control is therefore claimed conservatively.
        _base_t = np.asarray(cdata.get(
            "combined_baseline",
            np.asarray(cdata["combined_random_accs"], float)
              .reshape(len(all_tasks), -1).max(axis=1)), float)     # (T,)
        _floor_t = np.asarray(cdata["combined_accs"], float)\
            .reshape(len(all_tasks), -1).min(axis=1)                # (T,)
        headroom = np.maximum(_base_t - _floor_t, 1e-3)             # (T,)

        d_i = np.clip(SI / headroom[:, None], 0.0, 1.0)             # (T, P)
        d_h = np.clip(SH / headroom[:, None], 0.0, 1.0)             # (T, H)
        E_pred_mult = headroom[:, None, None] * (
            1.0 - (1.0 - d_i[:, :, None]) * (1.0 - d_h[:, None, :]))
        I_mult_task = CE - E_pred_mult                              # (T, P, H)
        I_mult_avg = I_mult_task.mean(axis=0)                       # (P, H)

        _hr_mask = ((SI[:, :, None] > 0) & (SH[:, None, :] > 0) &
                    (SI[:, :, None] + SH[:, None, :]
                     <= 0.5 * headroom[:, None, None]))             # (T, P, H)
        _hr_vals = I_task[_hr_mask]

        # Peak OM per (input, hidden) block from the matching modulation
        # clustering, when available in memory.
        om_max = None
        if _cim_for_interaction is not None:
            _om_type = ("modulation_all_normalized" if vtag == "norm"
                        else "modulation_all_unnormalized")
            _mk = _cim_for_interaction.get(_om_type, {})
            _fk_keys = [k for k in _mk if k.startswith("global_assignment_fixed_k")]
            _ga = _mk[_fk_keys[0]] if _fk_keys else _mk.get("global_assignment")
            if _ga is not None and _ga["n_in"] == c_pre_n and _ga["n_hid"] == c_post_n:
                om_max = _ga["om_stack"].max(axis=0)            # (P, H)
            elif _ga is not None:
                print(f"[interaction {vtag}] OM grid ({_ga['n_in']},{_ga['n_hid']}) "
                      f"≠ combined grid ({c_pre_n},{c_post_n}); OM panel skipped")

        n_panels = 3 if om_max is not None else 2
        fig, axs = plt.subplots(1, n_panels, figsize=(4.3 * n_panels, 3.8), dpi=300)

        _v = max(float(np.abs(I_avg).max()) * 100, 1e-6)
        sns.heatmap(I_avg * 100, ax=axs[0], cmap="RdBu_r", center=0,
                    vmin=-_v, vmax=_v,
                    xticklabels=[f"h{j}" for j in range(1, c_post_n + 1)],
                    yticklabels=[f"i{i}" for i in range(1, c_pre_n + 1)],
                    cbar_kws={"label": "Interaction (%)", "shrink": 0.8})
        axs[0].set_title("Task-averaged interaction\n(<0 redundant, >0 synergistic)",
                         fontsize=8)
        axs[0].tick_params(labelsize=5)

        axs[1].hist(I_avg.ravel() * 100, bins=25, color="steelblue",
                    edgecolor="black", linewidth=0.3, alpha=0.8)
        axs[1].axvline(0, color="grey", linestyle="--", linewidth=0.7)
        axs[1].set_title(
            f"mean={I_avg.mean() * 100:+.2f}%  |  "
            f"{(I_avg < 0).mean() * 100:.0f}% sub-additive", fontsize=8)
        axs[1].set_xlabel("Interaction (%)", fontsize=8)
        axs[1].set_ylabel("# (input, hidden) pairs", fontsize=8)
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)

        om_reg = None
        if om_max is not None:
            # The unresponsive input/hidden class (recorded in the combined
            # lesion entry; legacy pickles: last cluster of the unnorm variant)
            # is excluded from the regression, kept in the heatmap.
            _sel_i = _indices_without(c_pre_n, _combined_unresponsive_index(cdata, "pre", vtag))
            _sel_h = _indices_without(c_post_n, _combined_unresponsive_index(cdata, "post", vtag))
            x = om_max[np.ix_(_sel_i, _sel_h)].ravel()
            y = I_avg[np.ix_(_sel_i, _sel_h)].ravel() * 100
            _sl, _ic, _r, _pv, _ = linregress(x, y)
            om_reg = {"slope": _sl, "intercept": _ic, "r": _r, "p": _pv}
            axs[2].scatter(x, y, s=10, alpha=0.5, color="steelblue",
                           edgecolors="none")
            _xf = np.linspace(x.min(), x.max(), 50)
            axs[2].plot(_xf, _sl * _xf + _ic, color="tomato", linewidth=1.0)
            _ps = f"p = {_pv:.2e}" if _pv < 0.001 else f"p = {_pv:.3f}"
            axs[2].text(0.05, 0.95, f"r = {_r:.2f}\n{_ps}\nn = {len(x)}",
                        transform=axs[2].transAxes, va="top", fontsize=7)
            axs[2].set_xlabel("Peak synapse-cluster OM of block", fontsize=8)
            axs[2].set_ylabel("Interaction (%)", fontsize=8)
            axs[2].set_title("Anatomical co-location vs interaction", fontsize=8)
            axs[2].spines["top"].set_visible(False)
            axs[2].spines["right"].set_visible(False)

        fig.suptitle(f"Combined-lesion interaction map [{vtag}]", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/lesion_interaction_{vtag}_{aname}.png", dpi=300)
        plt.close(fig)

        # ── Saturation-control figure ──
        fig2, ax2 = plt.subplots(1, 3, figsize=(12.9, 3.8), dpi=300)

        # P1: additive vs multiplicative interaction, one point per pair.
        # Pairs whose y is pulled toward 0 owed their (additive)
        # sub-additivity to saturation; pairs staying below 0 keep a
        # genuine-overlap interpretation.
        ax = ax2[0]
        ax.scatter(I_avg.ravel() * 100, I_mult_avg.ravel() * 100, s=10,
                   alpha=0.5, color="steelblue", edgecolors="none")
        _lim = [min(I_avg.min(), I_mult_avg.min()) * 100,
                max(I_avg.max(), I_mult_avg.max()) * 100]
        ax.plot(_lim, _lim, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Additive interaction (%)", fontsize=8)
        ax.set_ylabel("Multiplicative-baseline interaction (%)", fontsize=8)
        ax.set_title("y pulled to 0 = sub-additivity was saturation;\n"
                     "y still < 0 = genuine overlap", fontsize=8)

        # P2: interaction restricted to cells with headroom.
        ax = ax2[1]
        if _hr_vals.size:
            ax.hist(_hr_vals * 100, bins=25, color="steelblue",
                    edgecolor="black", linewidth=0.3, alpha=0.8)
            ax.axvline(0, color="grey", linestyle="--", linewidth=0.7)
            ax.set_title(
                f"{_hr_vals.size} (task, pair) cells with headroom\n"
                f"mean={_hr_vals.mean() * 100:+.2f}%  |  "
                f"{(_hr_vals < 0).mean() * 100:.0f}% sub-additive", fontsize=8)
            ax.set_xlabel("Interaction (%)", fontsize=8)
            ax.set_ylabel("# cells", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no cells pass the headroom criterion",
                    ha="center", va="center", fontsize=8,
                    transform=ax.transAxes)
            ax.set_title("Headroom-restricted cells", fontsize=8)

        # P3: the most sub-additive pairs (additive ranking) — do they
        # survive the multiplicative saturation control?
        ax = ax2[2]
        _k = min(10, I_avg.size)
        _worst = np.argsort(I_avg, axis=None)[:_k]
        _wi, _wj = np.unravel_index(_worst, I_avg.shape)
        _pos = np.arange(_k)
        ax.bar(_pos - 0.2, I_avg[_wi, _wj] * 100, width=0.4,
               color="steelblue", label="additive")
        ax.bar(_pos + 0.2, I_mult_avg[_wi, _wj] * 100, width=0.4,
               color="tomato", label="multiplicative")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xticks(_pos)
        ax.set_xticklabels([f"i{i + 1}×h{j + 1}" for i, j in zip(_wi, _wj)],
                           rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Interaction (%)", fontsize=8)
        ax.set_title("Top sub-additive pairs under both baselines", fontsize=8)
        ax.legend(fontsize=6, frameon=False)

        for ax in ax2:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)
        fig2.suptitle(f"Interaction saturation control [{vtag}]", fontsize=9)
        fig2.tight_layout()
        fig2.savefig(f"{save_dir}/lesion_interaction_saturation_{vtag}_{aname}.png",
                     dpi=300)
        plt.close(fig2)

        with open(f"{save_dir}/lesion_interaction_{vtag}_{aname}.pkl", "wb") as _f:
            pickle.dump({
                "interaction_per_task": I_task, "interaction_avg": I_avg,
                "single_input_effects": SI, "single_hidden_effects": SH,
                "combined_effects": CE, "om_max": om_max, "om_regression": om_reg,
                "tasks": list(all_tasks), "vtag": vtag,
                # saturation control
                "interaction_mult_per_task": I_mult_task,
                "interaction_mult_avg": I_mult_avg,
                "baseline_per_task": _base_t,
                "floor_per_task": _floor_t,
                "headroom_per_task": headroom,
                "headroom_mask": _hr_mask,
                "headroom_criterion":
                    "both singles > 0 and their sum <= 0.5 * headroom",
            }, _f)
        _worst10 = np.argsort(I_avg, axis=None)[:min(10, I_avg.size)]
        print(f"[interaction {vtag}] mean={I_avg.mean() * 100:+.2f}%, "
              f"{(I_avg < 0).mean() * 100:.0f}% sub-additive"
              + (f", OM r={om_reg['r']:.2f} (p={om_reg['p']:.1e})"
                 if om_reg else ", OM unavailable"))
        print(f"[interaction {vtag}] saturation control: top-{len(_worst10)} "
              f"sub-additive pairs, additive mean="
              f"{I_avg.flat[_worst10].mean() * 100:+.1f}% -> multiplicative "
              f"mean={I_mult_avg.flat[_worst10].mean() * 100:+.1f}%; "
              f"headroom cells n={_hr_vals.size}"
              + (f", {(_hr_vals < 0).mean() * 100:.0f}% < 0"
                 if _hr_vals.size else ""))


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser(
        description="Regenerate the lesion post-processing figures for ONE "
                    "completed lesion run (reads multiple_tasks_perf/, writes "
                    "multiple_tasks_norm/; cheap — no model forwards). Use "
                    "--seed all with --feature to batch re-plot every saved "
                    "run for that feature. "
                    "Run from the repository root.")
    parser.add_argument(
        "--seed", type=str, default=None,
        help="Seed of the run to plot (e.g. 749), or 'all' to plot every "
             "completed run matching --feature.")
    parser.add_argument("--feature", type=str, default=None,
                        help="Feature tag of the run to plot (e.g. 'L21e4').")
    args = parser.parse_args()

    # Discover plottable runs: perf directories holding a finished
    # lesion_prune_results pickle. The pattern mirrors main()'s hard-coded
    # aname (ruleset 'everything', hidden300, batch128, +angle) — runs with
    # other configurations would not resolve inside main() and are not
    # offered here.
    _pat = re.compile(r"everything_seed(\d+)_(\w+)\+hidden300\+batch128\+angle$")
    candidates = []
    for _d in sorted(Path("multiple_tasks_perf").glob("everything_seed*")):
        _m = _pat.match(_d.name)
        if _m and (_d / f"lesion_prune_results_{_d.name}.pkl").exists():
            candidates.append((int(_m.group(1)), _m.group(2)))

    if args.seed == "all":
        if args.feature is None:
            raise SystemExit("--seed all requires --feature so the batch run is explicit.")
        matches = [(s, f) for s, f in candidates if f == args.feature]
        _avail = ", ".join(f"seed{s}/{f}" for s, f in candidates) or "none found"
        if len(matches) == 0:
            raise SystemExit(
                f"No completed lesion runs match feature={args.feature!r}. "
                f"Available runs: {_avail}")
        print(f"Plotting {len(matches)} lesion run(s) for feature='{args.feature}'")
        for _seed, _feature in matches:
            print(f"Plotting lesion results for seed={_seed}, feature='{_feature}'")
            main(_seed, _feature)
        raise SystemExit(0)

    try:
        seed_filter = None if args.seed is None else int(args.seed)
    except ValueError as exc:
        raise SystemExit("--seed must be an integer or 'all'.") from exc

    matches = [(s, f) for s, f in candidates
               if (seed_filter is None or s == seed_filter)
               and (args.feature is None or f == args.feature)]

    _avail = ", ".join(f"seed{s}/{f}" for s, f in candidates) or "none found"
    if len(matches) == 0:
        raise SystemExit(
            f"No completed lesion run matches seed={args.seed} "
            f"feature={args.feature!r}. Available runs: {_avail}")
    if len(matches) > 1:
        _hits = ", ".join(f"seed{s}/{f}" for s, f in matches)
        raise SystemExit(
            f"Ambiguous selection: {len(matches)} runs match seed={args.seed} "
            f"feature={args.feature!r} -> {_hits}. "
            f"Pass --seed (and --feature if needed) to select exactly one.")

    _seed, _feature = matches[0]
    print(f"Plotting lesion results for seed={_seed}, feature='{_feature}'")
    main(_seed, _feature)
