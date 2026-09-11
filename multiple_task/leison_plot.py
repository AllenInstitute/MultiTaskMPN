"""
Post-processing and visualization of lesion experiment results.

Reads the raw lesion pickle produced by leison.py and computes normalized
effects (random_accuracy - cluster_accuracy) to identify which neuron or
synapse clusters are selectively important for specific tasks. Produces:

1. Normalized lesion heatmaps — (cluster × task) matrices showing how much
   each cluster lesion impairs each task beyond the random-lesion baseline.
2. Combined heatmaps — side-by-side zero_W vs freeze_M modulation lesions
   with shared color scale, revealing whether a synapse cluster's contribution
   comes from static connectivity or dynamic plasticity.
3. Violin plots — distribution of normalized effect across tasks for each
   cluster, highlighting clusters with broad vs. task-specific roles.
4. Cluster similarity vs lesion effect — correlates cluster tuning similarity
   (from activity profiles) with functional lesion similarity (from accuracy
   patterns) to test whether similarly-tuned clusters have similar causal roles.
5. Overmembership vs lesion difference — relates modulation cluster enrichment
   in (input, hidden) neuron pairs to the functional similarity (task-profile
   L1 distance) between modulation lesion and combined neuron lesion effects,
   with a cluster-permutation p (footprint ownership shuffled) alongside the
   naive per-point regression p.
6. Causal dependency map — z-scores every (task, cluster) lesion effect
   against its stored random-control repeats (one-sided, BH-FDR across
   cells), biclusters the masked dependency matrix, and Mantel-tests whether
   the task organization implied by CAUSAL dependence matches the one
   implied by ACTIVITY tuning (cluster_info variance profiles).
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
   freeze_M effect / zero_W effect, i.e. the fraction of the cluster's
   contribution that flows through the plastic channel M rather than the
   static weight W. Aggregated per task and compared between memory-family
   tasks (delay/dm/dms/dmc) and reaction-family tasks (fd/react) — the
   MPN prediction is that working-memory tasks run on M.
10. Protective-cluster dissection — decomposes every NEGATIVE normalized
    lesion effect (cluster lesion hurting LESS than the size-matched random
    control) into own damage vs control damage on the shared test set, to
    separate the mechanical reading (the cluster is inert and the control
    sampled critical hub neurons) from genuine protection (removing the
    cluster IMPROVES absolute accuracy above the intact baseline).

Outputs saved to ./multiple_tasks_norm/{aname}/.

Entry points: run_pipeline.py calls `main(seed, feature)` as pipeline step 3.
Standalone, `python multiple_task/leison_plot.py --seed 749 --feature L21e4`
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


def _om_scatter_perm_test(mod_profiles, row_om, row_cm, n_perm=OM_N_PERM, seed=0):
    """Cluster-permutation p-value for the OM vs profile-L1 scatter.

    The scatter's (mod cluster, block) points are massively non-independent:
    each cluster's lesion profile is reused across all its blocks and each
    block's combined profile across all clusters, so linregress p-values treat
    ~20 clusters' worth of information as thousands of independent samples.
    The honest null keeps every lesion effect fixed and permutes WHICH cluster
    owns WHICH OM footprint — row_om[k] (masked OM values) and row_cm[k] (the
    matching blocks' combined-lesion task profiles) travel together with their
    stability mask — recomputing the pooled Pearson r each time. One-sided
    toward negative r (hypothesis: higher OM -> more similar lesion profiles).

    mod_profiles: (C, T) per-cluster modulation-lesion task profiles.
    row_om: length-C list of (B_k,) masked OM values per footprint.
    row_cm: length-C list of (B_k, T) matching blocks' task profiles.
    y for a (cluster c, footprint k) pairing is the per-block task-profile
    L1/T distance: mean_t |mod_profiles[c, t] - row_cm[k][b, t]|.
    Returns (r_obs, p_perm, null_r).
    """
    nC = len(mod_profiles)

    def _pooled_r(assign):
        x = np.concatenate([row_om[k] for k in assign])
        y = np.concatenate([
            np.mean(np.abs(row_cm[k] - mod_profiles[c][None, :]), axis=1)
            for c, k in enumerate(assign)])
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    r_obs = _pooled_r(np.arange(nC))
    rng = np.random.default_rng(seed)
    null_r = np.array([_pooled_r(rng.permutation(nC)) for _ in range(n_perm)])
    finite = np.isfinite(null_r)
    if not np.isfinite(r_obs) or not finite.any():
        return r_obs, np.nan, null_r
    p_perm = (1.0 + np.sum(null_r[finite] <= r_obs)) / (finite.sum() + 1.0)
    return r_obs, float(p_perm), null_r


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
    with open(pickle_name, 'rb') as f:
        results = pickle.load(f)
        
    # handle both old pickle names ("pre_cNone") and new ("pre_noleison") after rename fix
    baseline_keys = {"pre_cNone", "post_cNone", "pre_noleison", "post_noleison"}
    mod_leison_results = results.get("mod_leison", {})

    def compute_and_plot_normalized_lesion(leison_key, random_key, savename, xlabel_suffix=""):
        """Compute normalized lesion effect (random - cluster) and plot its heatmap.
        Returns (select_props, all_comb_names_filtered) for downstream use.
        """
        all_comb_names = results[leison_key]["all_comb_names_leison"]
        def _rename(k):
            return k.replace("pre_c", "i").replace("post_c", "h")
        all_comb_names_filtered = [_rename(k) for k in all_comb_names if k not in baseline_keys]
        tasks = results[leison_key]["all_tasks"]

        ihtask = np.asarray(results[leison_key]["ihtask_accs"], dtype=float)
        ihrandom = np.asarray(results[random_key]["ihrandomtask_accs"], dtype=float)

        props = []
        for key_idx, key in enumerate(all_comb_names):
            if key not in baseline_keys:
                props.append(ihrandom[:, key_idx] - ihtask[:, key_idx])

        props = np.array(props).T  # (n_tasks, n_clusters)
        suffix = f" {xlabel_suffix}" if xlabel_suffix else ""
        print(f"[{savename}] select_props: {props.shape}")

        helper.plot_heatmap(props, all_comb_names_filtered, tasks,
                            xlabel=f"Lesion Condition{suffix}", ylabel="Task",
                            savename=savename, aname=aname, label="Normalized Accuracy",
                            vmin=None, vmax=None, save_dir=save_dir)

        return props, all_comb_names_filtered

    select_props, all_comb_names_leison_ = compute_and_plot_normalized_lesion(
        "leison", "random_leison", "normalized_leison",
    )
    all_tasks = results["leison"]["all_tasks"]

    select_props_unnorm = None
    all_comb_names_unnorm_ = None
    if "leison_unnorm" in results and "random_leison_unnorm" in results:
        select_props_unnorm, all_comb_names_unnorm_ = compute_and_plot_normalized_lesion(
            "leison_unnorm", "random_leison_unnorm",
            "normalized_leison_unnorm", xlabel_suffix="(unnorm)",
        )

    # Combined violin: 4 panels — input/hidden × normalized/unnormalized
    if select_props_unnorm is not None:
        _n_input_norm_v = len([n for n in all_comb_names_leison_ if n.startswith("i")])
        _n_input_unnorm_v = len([n for n in all_comb_names_unnorm_ if n.startswith("i")])

        _ih_panels = [
            ("Input (normalized)", select_props[:, :_n_input_norm_v],
             [n for n in all_comb_names_leison_ if n.startswith("i")]),
            ("Hidden (normalized)", select_props[:, _n_input_norm_v:],
             [n for n in all_comb_names_leison_ if n.startswith("h")]),
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
        fig_ih.savefig(f"{save_dir}/normalized_leison_violin_all_{aname}.png", dpi=300)
        plt.close(fig_ih)
        print("Saved combined input/hidden violin plot (4 panels)")

    # Histogram of mean lesion effect per cluster for input/hidden (4 categories)
    # Split select_props into input and hidden based on the all_comb structure
    # all_comb_names_leison_ has i1..iN then h1..hM (after _rename)
    _n_input_norm = len([n for n in all_comb_names_leison_ if n.startswith("i")])

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
        fig_hist.savefig(f"{save_dir}/normalized_leison_hist_mean_{aname}.png", dpi=300)
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
                            f"{save_dir}/normalized_leison_ranked_{aname}.png")

    # ══════════════════════════════════════════════════════════════════
    # Causal dependency map — significance layer + biclustering, and the
    # causal-vs-activity task-organization comparison (module docstring #6).
    #
    # Statistics: every (task, cluster) lesion effect is z-scored against
    # its own size-matched random-control distribution, using the raw
    # repeats leison.py stores (ihrandomtask_accs_raw):
    #     z = (mean_ctrl - acc_lesion) / std_ctrl,   p = Phi(-z)  one-sided
    # The normal approximation is justified because each control accuracy
    # is itself a mean over test_n_batch trials. BH-FDR at q = 0.05 across
    # all cells of a variant. NB leison.py's control cache shares draws
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

    def _causal_dependency(leison_key, random_key, vtag):
        """FDR-masked (task, cluster) dependency matrix + biclustered view.

        Returns the UNMASKED effect matrix (n_tasks, n_clusters) for reuse
        by the task-similarity comparison below (correlations pool over all
        clusters and are robust to per-cell noise, so they use the full
        matrix; the masked matrix drives the module-map figure only)."""
        names_all = results[leison_key]["all_comb_names_leison"]
        keep_idx = [k for k, n in enumerate(names_all) if n not in baseline_keys]
        cnames = [names_all[k].replace("pre_c", "i").replace("post_c", "h")
                  for k in keep_idx]
        accs = np.asarray(results[leison_key]["ihtask_accs"], float)[:, keep_idx]
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
        return E

    E_dep_norm = _causal_dependency("leison", "random_leison", "norm")
    if "leison_unnorm" in results and "random_leison_unnorm" in results:
        _causal_dependency("leison_unnorm", "random_leison_unnorm", "unnorm")

    # ── Causal vs activity task organization (norm variant) ─────────────
    # Task-task similarity from lesion-dependency profiles vs from
    # task-averaged activity tuning; one-sided Mantel permutation test.
    def _mantel(S_ref, S_other, n_perm=10000, seed=0):
        """One-sided Mantel test (positive association) on upper triangles."""
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

    _ci_path = f"./multiple_tasks_analysis/{aname}/cluster_info_{aname}.pkl"
    if os.path.exists(_ci_path):
        with open(_ci_path, "rb") as _f:
            cluster_info = pickle.load(_f)  # also reused by later sections

        S_les = np.corrcoef(E_dep_norm)     # task × task, causal profiles
        for side in ["hidden", "input"]:
            V = cluster_info[f"{side}_normalized"]["cell_vars_rules_sorted_norm"]
            tb = cluster_info[f"{side}_normalized"]["tb_break_name"]
            A_task = np.stack([
                V[[r for r, nm in enumerate(tb)
                   if str(nm).split("-")[0] == t]].mean(axis=0)
                for t in all_tasks
            ])                               # (n_tasks, n_neurons)
            S_act = np.corrcoef(A_task)
            if not (np.isfinite(S_les).all() and np.isfinite(S_act).all()):
                print(f"[causal-vs-activity] {side}: non-finite similarity, skipping")
                continue
            r_m, p_m = _mantel(S_act, S_les)

            fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), dpi=300)
            for ax, S, ttl in [(axs[0], S_les, "Causal (lesion profiles)"),
                               (axs[1], S_act, f"Activity ({side} tuning)")]:
                sns.heatmap(S, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                            xticklabels=all_tasks, yticklabels=all_tasks,
                            cbar_kws={"shrink": 0.75})
                ax.set_title(ttl, fontsize=9)
                ax.tick_params(labelsize=6)
            iu = np.triu_indices(len(all_tasks), k=1)
            axs[2].scatter(S_act[iu], S_les[iu], s=14, alpha=0.7,
                           color="steelblue", edgecolors="none")
            axs[2].set_xlabel(f"Activity task similarity ({side})", fontsize=8)
            axs[2].set_ylabel("Causal task similarity", fontsize=8)
            axs[2].set_title(f"Mantel r={r_m:.2f}, p={p_m:.4f}", fontsize=9)
            axs[2].spines["top"].set_visible(False)
            axs[2].spines["right"].set_visible(False)
            fig.suptitle("Do tasks that look alike (activity) depend on the "
                         "same clusters (lesion)?", fontsize=9)
            fig.tight_layout()
            fig.savefig(f"{save_dir}/causal_vs_activity_tasksim_{side}_{aname}.png",
                        dpi=300)
            plt.close(fig)
            print(f"[causal-vs-activity] {side}: Mantel r={r_m:.2f}, p={p_m:.4f}")
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
    # inside leison.py, and the forward pass is deterministic):
    #   own_damage  = baseline − lesion acc        (< 0 ⇒ improvement)
    #   ctrl_damage = baseline − mean control acc
    #   normalized effect (random − lesion) ≡ own_damage − ctrl_damage
    # Noise scale: accuracies are means over ~_N_EVAL_TRIALS trials, so a
    # binomial-style se = sqrt(base(1−base)/N), with a sqrt(2) independence
    # (upper-bound) factor for the paired difference. These are heuristic
    # SCREENING thresholds, not formal tests — a genuinely protective
    # cluster must show systematic improvement across tasks, not one cell.
    # ══════════════════════════════════════════════════════════════════
    _N_EVAL_TRIALS = 200   # leison.py's test_n_batch (not stored in the pickle)

    for _pc_vtag, _pc_leison, _pc_random in [
        ("norm", "leison", "random_leison"),
        ("unnorm", "leison_unnorm", "random_leison_unnorm"),
    ]:
        if _pc_leison not in results or _pc_random not in results:
            continue
        _names_pc = results[_pc_leison]["all_comb_names_leison"]
        _acc_pc = np.asarray(results[_pc_leison]["ihtask_accs"], float)
        _rnd_pc = np.asarray(results[_pc_random]["ihrandomtask_accs"], float)
        _units_pc = results[_pc_leison].get("lesion_units", {})

        # The two no-lesion baselines are both no-op forwards on the same
        # test set and should be identical; average defensively if not.
        _b_pre = _acc_pc[:, _names_pc.index("pre_noleison")]
        _b_post = _acc_pc[:, _names_pc.index("post_noleison")]
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
        ckey = f"combined_leison_{vtag}"
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
                savename=f"normalized_combined_leison_{vtag}",
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
    _n_input_norm = len([n for n in all_comb_names_leison_ if n.startswith("i")])
    for vtag in ["norm", "unnorm"]:
        ckey = f"combined_leison_{vtag}"
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

    # Normalized modulation lesion effect for each clustering type
    mod_baseline_keys = {"mod_noleison"}

    # First pass: compute normalized effect and collect by clustering type.
    # Individual heatmaps are not plotted; combined (zero_W | freeze_M) panels are plotted below.
    from collections import defaultdict
    mod_by_type = defaultdict(dict)

    for mod_type_key, mod_data in mod_leison_results.items():
        all_comb_names_mod = mod_data["all_comb_names_mod"]
        all_comb_names_mod_ = [k for k in all_comb_names_mod if k not in mod_baseline_keys]
        modtask_accs = np.asarray(mod_data["modtask_accs"], dtype=float)
        modrandomtask_accs = np.asarray(mod_data["modrandomtask_accs"], dtype=float)

        mod_select_props = []
        for key_idx, key in enumerate(all_comb_names_mod):
            if key not in mod_baseline_keys:
                mod_select_props.append(modrandomtask_accs[:, key_idx] - modtask_accs[:, key_idx])

        mod_select_props = np.array(mod_select_props).T

        if "__" in mod_type_key:
            base_key, mode = mod_type_key.rsplit("__", 1)
        else:
            base_key = mod_type_key
            mode = mod_data.get("mod_lesion_mode", "zero_W")

        print(f"[{mod_type_key}] mod_select_props: {mod_select_props.shape}")

        mod_by_type[base_key][mode] = {
            "select_props": mod_select_props,
            "cluster_names": all_comb_names_mod_,
        }

    # Combined violin plot for all modulation types (zero_W only), 4 vertical subpanels
    _mod_violin_order = [
        "modulation_all_normalized",
        "modulation_all_unnormalized",
        "modulation_all_var_weighted_unnormalized",
        "modulation_all_weighted_unnormalized",
    ]
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
        fig_v.savefig(f"{save_dir}/normalized_mod_leison_violin_all_{aname}.png", dpi=300)
        plt.close(fig_v)
        print(f"Saved combined modulation violin plot ({n_panels} panels)")

        # Ranked mean effect comparison: sorted cluster rank vs mean effect per type.
        # Steeper curve = better separation between critical and dispensable clusters.
        _rank_colors = {
            "normalized": "#1b9e77",
            "unnormalized": "#d95f02",
            "var-weighted-unnormalized": "#e7298a",
            "weighted-unnormalized": "#7570b3",
        }
        _mod_rank_data = {
            bk.replace("modulation_all_", "").replace("_", "-"): d["select_props"].mean(axis=0) * 100
            for bk, d in _mod_violin_data
        }
        _plot_ranked_effect(_mod_rank_data, _rank_colors,
                            "Ranked cluster importance (modulation)",
                            f"{save_dir}/normalized_mod_leison_ranked_{aname}.png")

        # Cluster size vs normalized lesion effect
        # Tests whether larger clusters are more important after size-matching control.
        _size_colors = {
            "normalized": "#1b9e77",
            "unnormalized": "#d95f02",
            "var-weighted-unnormalized": "#e7298a",
            "weighted-unnormalized": "#7570b3",
        }
        fig_size, ax_size = plt.subplots(figsize=(4.5, 3.5), dpi=300)
        for bk, mode_data in _mod_violin_data:
            type_tag = bk.replace("modulation_all_", "").replace("_", "-")
            mean_per_cluster = mode_data["select_props"].mean(axis=0) * 100
            # Get cluster sizes from the lesion pickle
            _mod_rkey = f"{bk}__zero_W"
            if _mod_rkey in mod_leison_results:
                _col_cls = mod_leison_results[_mod_rkey]["mod_col_clusters"]
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
        fig_size.savefig(f"{save_dir}/normalized_mod_leison_size_vs_effect_{aname}.png", dpi=300)
        plt.close(fig_size)
        print("Saved modulation cluster size vs effect plot")

        # Histogram for modulation (cluster-averaged + task-specific + summary stats)
        _mod_hist_colors = {
            "normalized": "#1b9e77",
            "unnormalized": "#d95f02",
            "var-weighted-unnormalized": "#e7298a",
            "weighted-unnormalized": "#7570b3",
        }
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
        fig_mhist.savefig(f"{save_dir}/normalized_mod_leison_hist_mean_{aname}.png", dpi=300)
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
        for mod_result_key, mod_data in mod_leison_results.items():
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
                    savename=f"normalized_mod_leison_{base_tag}_{mode_tag}",
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
        _hm_path = f"{save_dir}/normalized_mod_leison_{base_tag}_combined_heatmap_{aname}"
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
        fig.savefig(f"{save_dir}/normalized_mod_leison_compare_{base_tag}_{aname}.png", dpi=300)
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

    for _pd_type in ["modulation_all_normalized", "modulation_all_unnormalized",
                     "modulation_all_weighted_unnormalized",
                     "modulation_all_var_weighted_unnormalized"]:
        zw = mod_leison_results.get(f"{_pd_type}__zero_W")
        fm = mod_leison_results.get(f"{_pd_type}__freeze_M")
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
        cluster_info: neuron clustering pickle (needed to identify unresponsive clusters)
        """
        ckey = f"combined_leison_{variant}"
        if ckey not in results or not results[ckey]:
            print(f"[om_vs_lesion] skipping {variant}: combined lesion data not found")
            return
        mod_result_key = f"{mod_type_key}__{mod_lesion_mode}"
        if mod_result_key not in results["mod_leison"]:
            print(f"[om_vs_lesion] skipping: {mod_result_key} not found in mod_leison")
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
        # The overmembership om_stack uses fixed-k clusters. For unnormalized data,
        # the unresponsive cluster is the last one (index n_in-1 / n_hid-1).
        # For normalized: no unresponsive cluster exists.
        # Fixed-k modulation labels (from col_labels_by_k) have no separate
        # unresponsive cluster, so no modulation exclusion is needed.
        skip_input = set()
        skip_hidden = set()
        if variant == "unnorm":
            skip_input.add(n_in - 1)
            skip_hidden.add(n_hid - 1)
            print(f"[om_vs_lesion] excluding unresponsive: input idx={n_in-1}, hidden idx={n_hid-1}")

        # --- Modulation lesion effect (random - cluster), per task ---
        mod_data = results["mod_leison"][mod_result_key]
        mod_baseline_keys = {"mod_noleison"}
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
        mod_profiles = np.array(mod_profiles)
        om_vals = np.concatenate(row_om_list)
        lesion_diffs = np.concatenate([
            np.mean(np.abs(cm - mp[None, :]), axis=1)
            for mp, cm in zip(mod_profiles, row_cm_list)])

        if len(om_vals) < 2:
            print(f"[om_vs_lesion] no data points to plot")
            return

        # Naive regression (slope/line for display; its p treats every point
        # as independent and is kept only for reference) + the honest
        # cluster-permutation p (see _om_scatter_perm_test).
        slope, intercept, r, p, _ = linregress(om_vals, lesion_diffs)
        _, p_perm, _null_r = _om_scatter_perm_test(mod_profiles, row_om_list, row_cm_list)

        fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
        ax.scatter(om_vals, lesion_diffs, alpha=0.5, s=20, edgecolors="none", color="steelblue")

        x_line = np.linspace(om_vals.min(), om_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.2)

        _pp_str = (f"p_perm = {p_perm:.3f}" if np.isfinite(p_perm) else "p_perm = n/a")
        ax.text(0.05, 0.95,
                f"r = {r:.2f}, slope = {slope:.2f}\n"
                f"{_pp_str} ({OM_N_PERM} perms, {len(mod_profiles)} clusters)\n"
                f"n = {len(om_vals)} (naive p = {p:.1e})",
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
                "om_vals": om_vals,
                "lesion_diffs": lesion_diffs,
                "labels": labels,
                "regression": {"slope": slope, "intercept": intercept, "r": r, "p": p},
                "permutation": {"p_perm": p_perm, "null_r": _null_r,
                                "n_perm": OM_N_PERM, "n_clusters": len(mod_profiles),
                                "side": "one-sided (r <= r_obs)"},
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
            if mod_result_key not in results["mod_leison"]:
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

            skip_input = set()
            skip_hidden = set()
            if variant == "unnorm":
                skip_input.add(n_in - 1)
                skip_hidden.add(n_hid - 1)

            mod_data = results["mod_leison"][mod_result_key]
            mod_baseline_keys_ = {"mod_noleison"}
            all_comb_names_mod = mod_data["all_comb_names_mod"]
            modtask_accs = np.asarray(mod_data["modtask_accs"], dtype=float)
            modrandomtask_accs = np.asarray(mod_data["modrandomtask_accs"], dtype=float)

            mod_effects = {}
            for key_idx, key in enumerate(all_comb_names_mod):
                if key in mod_baseline_keys_:
                    continue
                cid = int(key.replace("mod_c", ""))
                mod_effects[cid] = modrandomtask_accs[:, key_idx] - modtask_accs[:, key_idx]

            ckey = f"combined_leison_{variant}"
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
                mod_profiles = np.array(mod_profiles)
                om_vals = np.concatenate(row_om_list)
                lesion_diffs = np.concatenate([
                    np.mean(np.abs(cm - mp[None, :]), axis=1)
                    for mp, cm in zip(mod_profiles, row_cm_list)])
                if len(om_vals) >= 2:
                    _, _p_perm, _ = _om_scatter_perm_test(
                        mod_profiles, row_om_list, row_cm_list)
                    mode_data_all[mode] = (om_vals, lesion_diffs,
                                           _p_perm, len(mod_profiles))

        if len(mode_data_all) < 2:
            return

        # Panel 3: per-cluster prediction — use OM-weighted combined effect to predict
        # modulation cluster's mean own damage (one point per cluster).
        # For each mod cluster: predicted_effect = sum(OM[i,j] * combined_effect_mean[i,j]) / sum(OM[i,j])
        # Uses zero_W mode for the prediction.
        _pred_x, _pred_y = [], []
        mod_result_key_zw = f"{base_key}__zero_W"
        if mod_result_key_zw in results["mod_leison"] and base_key in cluster_info_mod:
            _mod_keys_p = cluster_info_mod[base_key]
            _fk_ga_keys_p = [k for k in _mod_keys_p if k.startswith("global_assignment_fixed_k")]
            ga_p = _mod_keys_p[_fk_ga_keys_p[0]] if _fk_ga_keys_p else _mod_keys_p.get("global_assignment")
            if ga_p is not None:
                om_stack_p = ga_p["om_stack"]
                all_choice_order_p = ga_p["all_choice_order"]
                n_in_p, n_hid_p = ga_p["n_in"], ga_p["n_hid"]
                om_id_to_idx_p = {cid: idx for idx, cid in enumerate(all_choice_order_p)}

                ckey_p = f"combined_leison_{variant}"
                if ckey_p in results and results[ckey_p]:
                    cdata_p = results[ckey_p]
                    comb_eff_p = (np.asarray(cdata_p["combined_random_accs"], dtype=float)
                                  - np.asarray(cdata_p["combined_accs"], dtype=float))
                    comb_mean_p = comb_eff_p.mean(axis=0)  # (pre_n, post_n)

                    mod_data_p = results["mod_leison"][mod_result_key_zw]
                    _mt_p = np.asarray(mod_data_p["modtask_accs"], dtype=float)
                    _baseline_idx_p = mod_data_p["all_comb_names_mod"].index("mod_noleison")
                    _base_p = _mt_p[:, _baseline_idx_p]
                    for key_idx, key in enumerate(mod_data_p["all_comb_names_mod"]):
                        if key == "mod_noleison":
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
            om_vals, lesion_diffs, p_perm, n_clusters = mode_data_all[mode]
            slope, intercept, r, p, _ = linregress(om_vals, lesion_diffs)

            ax.scatter(om_vals, lesion_diffs, alpha=0.4, s=12, edgecolors="none", color="steelblue")
            x_line = np.linspace(om_vals.min(), om_vals.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.0)

            _pp_str = (f"p_perm = {p_perm:.3f}" if np.isfinite(p_perm)
                       else "p_perm = n/a")
            ax.text(0.05, 0.95,
                    f"r = {r:.2f}\n{_pp_str} ({n_clusters} clusters)\n"
                    f"n = {len(om_vals)} (naive p = {p:.1e})",
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
            f"{mode} p_perm={vals[2]:.3f}" for mode, vals in mode_data_all.items())
        print(f"[om_vs_lesion] {type_tag} [{variant}] permutation "
              f"({OM_N_PERM} perms): {_perm_summary}"
              + (f", prediction p_perm={_pred_p_perm:.3f}"
                 if np.isfinite(_pred_p_perm) else ""))

        # Save per-mode scatter data and the per-cluster prediction so the
        # combined figure (and paper_plot's re-derivation of the same matching)
        # can be reproduced directly from this pickle.
        data_path = f"{save_dir}/om_vs_lesion_diff_{type_tag}_combined_{variant}_{aname}.pkl"
        with open(data_path, "wb") as _f:
            pickle.dump({
                "mode_data": {
                    mode: {"om_vals": vals[0], "lesion_diffs": vals[1],
                           "p_perm": vals[2], "n_clusters": vals[3]}
                    for mode, vals in mode_data_all.items()
                },
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
                "perm_side": {"scatter": "one-sided (r <= r_obs)",
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
        if (mod_result_key not in results["mod_leison"]
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

        ckey = f"combined_leison_{variant}"
        if ckey not in results or not results[ckey]:
            return
        cdata = results[ckey]
        if n_in != cdata["pre_n"] or n_hid != cdata["post_n"]:
            print(f"[om-profile] {base_key}: OM grid ({n_in},{n_hid}) ≠ combined "
                  f"grid ({cdata['pre_n']},{cdata['post_n']}), skipping")
            return
        CE = (np.asarray(cdata["combined_random_accs"], float)
              - np.asarray(cdata["combined_accs"], float))       # (T, P, H)
        sel_i = np.arange(n_in - 1 if variant == "unnorm" else n_in)
        sel_h = np.arange(n_hid - 1 if variant == "unnorm" else n_hid)
        CE_s = CE[:, sel_i][:, :, sel_h]                         # (T, P', H')

        mod_data = results["mod_leison"][mod_result_key]
        _mt = np.asarray(mod_data["modtask_accs"], float)
        _baseline_idx = mod_data["all_comb_names_mod"].index("mod_noleison")
        _base = _mt[:, _baseline_idx]
        clusters, om_rows, actual = [], [], []
        for key_idx, key in enumerate(mod_data["all_comb_names_mod"]):
            if key == "mod_noleison":
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
        for mod_result_key in results["mod_leison"]:
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

    # ── Cluster similarity vs normalized lesion effect ──
    def plot_cluster_corr_vs_lesion(corr_matrices_dict, select_props_mat, slices_dict,
                                    savesuffix, aname, save_dir,
                                    cluster_means_dict=None,
                                    exclude_last_cluster=False):
        """3×N figure: for each cluster type (column),
        row 0 = cluster tuning cosine similarity heatmap,
        row 1 = lesion effect L1 distance heatmap,
        row 2 = scatter of tuning cosine sim vs lesion L1 distance.

        corr_matrices_dict only supplies the panel names and cluster counts;
        both plotted matrices are computed here — tuning similarity from
        cluster_means_dict, lesion L1 distance from select_props_mat.

        If exclude_last_cluster=True, the last cluster (unresponsive) is excluded
        from the scatter plot (row 2) but still shown in the heatmaps."""
        n_cols = len(corr_matrices_dict)
        fig, axs = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 11), dpi=300,
                                squeeze=False)

        from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
        from scipy.spatial.distance import squareform as _squareform, pdist as _pdist

        scatter_save_data = {}

        for col, (name, corr_matrix) in enumerate(corr_matrices_dict.items()):
            lesion_vecs = select_props_mat[:, slices_dict[name]].T  # (n_clusters, n_tasks)

            # Cluster tuning: cosine similarity between cluster mean profiles.
            # Required — a silent fallback here (the old np.eye placeholder)
            # would draw a meaningless identity-similarity panel instead of
            # failing, so missing means are treated as a caller error.
            if cluster_means_dict is None or name not in cluster_means_dict:
                raise ValueError(
                    f"cluster_means_dict must provide {name!r}: the tuning-"
                    "similarity panel is computed from cluster mean profiles"
                )
            tuning_cos = _cosine_sim(cluster_means_dict[name].T)
            # Lesion effect: L1 distance between lesion effect vectors
            lesion_l1 = _squareform(_pdist(lesion_vecs, metric="cityblock"))

            n = corr_matrix.shape[0]
            tril_idx = np.tril_indices(n, k=-1)
            cluster_labels = [str(i) for i in range(n)]

            ax0 = axs[0, col]
            mat0 = np.full((n, n), np.nan)
            mat0[tril_idx] = tuning_cos[tril_idx]
            im0 = ax0.imshow(mat0, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                             origin="upper")
            fig.colorbar(im0, ax=ax0, shrink=0.8, label="Cosine sim.")
            ax0.set_xticks(range(n))
            ax0.set_yticks(range(n))
            ax0.set_xticklabels(cluster_labels)
            ax0.set_yticklabels(cluster_labels)
            ax0.set_xlabel("Cluster index")
            ax0.set_ylabel("Cluster index")
            ax0.set_title(f"{name}: tuning cosine similarity")

            ax1 = axs[1, col]
            mat1 = np.full((n, n), np.nan)
            mat1[tril_idx] = lesion_l1[tril_idx]
            im1 = ax1.imshow(mat1, aspect="auto", cmap="viridis",
                             origin="upper")
            fig.colorbar(im1, ax=ax1, shrink=0.8, label="L1 distance")
            ax1.set_xticks(range(n))
            ax1.set_yticks(range(n))
            ax1.set_xticklabels(cluster_labels)
            ax1.set_yticklabels(cluster_labels)
            ax1.set_xlabel("Cluster index")
            ax1.set_ylabel("Cluster index")
            ax1.set_title(f"{name}: lesion effect L1 distance")

            ax2 = axs[2, col]
            if exclude_last_cluster and n > 1:
                # Exclude pairs involving the last cluster (unresponsive)
                n_active = n - 1
                tril_idx_active = np.tril_indices(n_active, k=-1)
                x = tuning_cos[:n_active, :n_active][tril_idx_active]
                y = lesion_l1[:n_active, :n_active][tril_idx_active]
            else:
                x = tuning_cos[tril_idx]
                y = lesion_l1[tril_idx]
            ax2.scatter(x, y, alpha=0.6, s=30, edgecolors="none", color="steelblue")

            if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                slope, intercept, r, p, _ = linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax2.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.2)

                p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
                ax2.text(0.05, 0.95, f"r = {r:.2f}\n{p_str}",
                         transform=ax2.transAxes, va="top", ha="left", fontsize=8)
            else:
                ax2.text(0.05, 0.95, "constant x or y", transform=ax2.transAxes,
                         va="top", ha="left", fontsize=8)

            ax2.set_xlabel("Tuning cosine similarity")
            ax2.set_ylabel("Lesion effect L1 distance")
            ax2.set_title(f"{name} clusters")

            scatter_save_data[name] = {
                "tuning_cos_sim": x.tolist(),
                "lesion_l1_dist": y.tolist(),
            }

        fig.tight_layout()
        fig.savefig(f"{save_dir}/cluster_corr_vs_{savesuffix}_{aname}.png", dpi=300)
        plt.close(fig)
        print(f"Saved cluster_corr_vs_{savesuffix}")

        # Save scatter data for paper_plot reuse
        scatter_pkl_path = f"{save_dir}/cluster_corr_vs_{savesuffix}_{aname}.pkl"
        with open(scatter_pkl_path, "wb") as _f:
            pickle.dump(scatter_save_data, _f)
        print(f"Saved scatter data: {scatter_pkl_path}")

    # --- Normalized variant ---
    corr_matrices_norm = results["cluster_similarity"]["corr_matrices"]
    cluster_means_norm = results["cluster_similarity"]["cluster_means"]
    # Keys may be "input_normalized" or "input_normalized_k{N}" depending on FIXED_K
    _input_norm_key = [k for k in corr_matrices_norm if k.startswith("input_normalized")][0]
    _hidden_norm_key = [k for k in corr_matrices_norm if k.startswith("hidden_normalized")][0]
    pre_n = len(corr_matrices_norm[_input_norm_key])
    post_n = len(corr_matrices_norm[_hidden_norm_key])
    slices_norm = {
        _input_norm_key:  slice(0, pre_n),
        _hidden_norm_key: slice(pre_n, pre_n + post_n),
    }
    plot_cluster_corr_vs_lesion(
        corr_matrices_norm, select_props, slices_norm,
        "normalized_leison_effect", aname, save_dir,
        cluster_means_dict=cluster_means_norm,
    )

    # --- Unnormalized variant ---
    # Compute cluster similarity on-the-fly from cluster_info (not in the lesion pickle)
    if select_props_unnorm is not None and os.path.exists(cluster_path):
        try:
            cluster_info
        except NameError:
            with open(cluster_path, "rb") as f:
                cluster_info = pickle.load(f)

        _fixed_k_plot = results.get("fixed_k", 20)

        corr_matrices_unnorm = {}
        cluster_means_unnorm = {}
        _unnorm_keys = {}
        for name in ["input_unnormalized", "hidden_unnormalized"]:
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
            fk_name = f"{name}_k{_fixed_k_plot}"
            corr_matrices_unnorm[fk_name] = np.corrcoef(cluster_means.T)
            cluster_means_unnorm[fk_name] = cluster_means
            _unnorm_keys[name] = fk_name

        if "input_unnormalized" in _unnorm_keys and "hidden_unnormalized" in _unnorm_keys:
            _ik = _unnorm_keys["input_unnormalized"]
            _hk = _unnorm_keys["hidden_unnormalized"]
            pre_n_u = len(corr_matrices_unnorm[_ik])
            post_n_u = len(corr_matrices_unnorm[_hk])
            slices_unnorm = {
                _ik: slice(0, pre_n_u),
                _hk: slice(pre_n_u, pre_n_u + post_n_u),
            }
            plot_cluster_corr_vs_lesion(
                corr_matrices_unnorm, select_props_unnorm, slices_unnorm,
                "normalized_leison_effect_unnorm", aname, save_dir,
                cluster_means_dict=cluster_means_unnorm,
                exclude_last_cluster=True,
            )

    # --- Modulation variant ---
    # For each modulation clustering type × lesion mode, compute cluster similarity
    # from cell_vars_rules_sorted_norm + the actual cluster assignments used in lesion,
    # then compare against the normalized modulation lesion effect.
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
            V_mod = mod_ci["cell_vars_rules_sorted_norm"]   # (n_tasks, n_synapses)

            # Use the actual cluster assignments saved in the lesion pickle
            # (guaranteed to match mod_select_props columns).
            _any_mode = next(iter(modes_dict.values()))
            _result_key = f"{mod_type_key}__{next(iter(modes_dict.keys()))}"
            mod_data_ref = mod_leison_results[_result_key]
            col_clusters_mod = mod_data_ref["mod_col_clusters"]
            unique_labels = sorted(col_clusters_mod.keys())
            n_mod_clusters = len(unique_labels)
            cluster_means_mod = np.stack(
                [V_mod[:, col_clusters_mod[lab]].mean(axis=1) for lab in unique_labels],
                axis=1,
            )  # (n_tasks, n_mod_clusters)
            corr_matrix_mod = np.corrcoef(cluster_means_mod.T)

            for mode, mode_data in modes_dict.items():
                mod_select_props = mode_data["select_props"]   # (n_tasks, n_mod_clusters)
                n_lesion_cols = mod_select_props.shape[1]

                # mod_select_props columns are ordered by sorted cluster IDs,
                # excluding the no-lesion baseline. The similarity matrix rows/cols
                # follow unique_labels (also sorted). They should match.
                if n_lesion_cols != n_mod_clusters:
                    print(f"[mod corr_vs_lesion] column mismatch for {mod_type_key}__{mode}: "
                          f"lesion={n_lesion_cols}, similarity={n_mod_clusters}, skipping")
                    continue

                type_tag = mod_type_key.replace("modulation_all_", "").replace("_", "-")
                mode_tag = mode.replace("_", "-")
                mod_name = f"{type_tag}_{mode_tag}"

                _is_unnorm_mod = "unnormalized" in mod_type_key
                plot_cluster_corr_vs_lesion(
                    {mod_name: corr_matrix_mod},
                    mod_select_props,
                    {mod_name: slice(0, n_mod_clusters)},
                    f"mod_leison_effect_{type_tag}_{mode_tag}", aname, save_dir,
                    cluster_means_dict={mod_name: cluster_means_mod},
                    exclude_last_cluster=_is_unnorm_mod,
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
    # 3+ GB cluster_info_mod pickle is deliberately NOT re-loaded here.
    # ══════════════════════════════════════════════════════════════════
    try:
        _cim_for_interaction = cluster_info_mod
    except NameError:
        _cim_for_interaction = None

    for vtag, singles, names_f in [
        ("norm", select_props, all_comb_names_leison_),
        ("unnorm", select_props_unnorm, all_comb_names_unnorm_),
    ]:
        ckey = f"combined_leison_{vtag}"
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
            # For the unnorm variant the LAST input/hidden cluster is the
            # unresponsive class (same convention as om_vs_lesion above) —
            # excluded from the regression, kept in the heatmap.
            _sel_i = np.arange(c_pre_n - 1 if vtag == "unnorm" else c_pre_n)
            _sel_h = np.arange(c_post_n - 1 if vtag == "unnorm" else c_post_n)
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
