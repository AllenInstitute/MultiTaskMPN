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
   in (input, hidden) neuron pairs to the functional similarity between
   modulation lesion and combined neuron lesion effects.

Outputs saved to ./multiple_tasks_norm/{aname}/.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

import pickle
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper

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

def main(seed, feature):
    aname = f"everything_seed{seed}_{feature}+hidden300+batch128+angle"
    print(f"aname: {aname}")

    pickle_dir = f"./multiple_tasks_perf/{aname}"
    save_dir = f"./multiple_tasks_norm/{aname}"
    os.makedirs(save_dir, exist_ok=True)
    for _old in Path(save_dir).iterdir():
        if _old.is_file():
            _old.unlink()

    pickle_name = f"{pickle_dir}/lesion_prune_results_{aname}.pkl"
    with open(pickle_name, 'rb') as f:
        results = pickle.load(f)
        
    # handle both old pickle names ("pre_cNone") and new ("pre_noleison") after rename fix
    baseline_keys = {"pre_cNone", "post_cNone", "pre_noleison", "post_noleison"}
    mod_leison_results = results.get("mod_leison", {})

    from scipy.cluster.hierarchy import fcluster as _fcluster

    def _derive_fixed_k_clusters(ci_entry, fk):
        """Cut dendrogram at fk, return col_clusters dict {label: indices}.
        Unresponsive neurons get label fk+1."""
        res = ci_entry["result"]
        lnk = res["col_linkage"]
        tol_k = res["col_tol_k"]
        tol_labels = res["col_tol_labels"]
        unres_mask = tol_labels == (tol_k + 1)
        active_labels = _fcluster(lnk, fk, criterion="maxclust")
        full_labels = np.zeros(len(tol_labels), dtype=int)
        full_labels[~unres_mask] = active_labels
        if unres_mask.any():
            full_labels[unres_mask] = fk + 1
        return {int(lab): np.where(full_labels == lab)[0] for lab in np.unique(full_labels) if lab > 0}

    def _derive_fixed_k_labels(ci_entry, fk):
        """Cut dendrogram at fk, return full label array (same as _derive_fixed_k_clusters
        but returns the raw array instead of a dict)."""
        res = ci_entry["result"]
        lnk = res["col_linkage"]
        tol_k = res["col_tol_k"]
        tol_labels = res["col_tol_labels"]
        unres_mask = tol_labels == (tol_k + 1)
        active_labels = _fcluster(lnk, fk, criterion="maxclust")
        full_labels = np.zeros(len(tol_labels), dtype=int)
        full_labels[~unres_mask] = active_labels
        if unres_mask.any():
            full_labels[unres_mask] = fk + 1
        return full_labels

    def compute_and_plot_normalized_lesion(leison_key, random_key, savename, xlabel_suffix=""):
        """Compute normalized lesion effect (random - cluster), plot heatmap + violin.
        Returns (select_props, all_comb_names_filtered, ihtask_accs) for downstream use.
        """
        import seaborn as sns

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

        return props, all_comb_names_filtered, ihtask

    select_props, all_comb_names_leison_, ihtask_accs = compute_and_plot_normalized_lesion(
        "leison", "random_leison", "normalized_leison",
    )
    all_comb_names_leison = results["leison"]["all_comb_names_leison"]
    all_tasks = results["leison"]["all_tasks"]

    select_props_unnorm = None
    all_comb_names_unnorm_ = None
    if "leison_unnorm" in results and "random_leison_unnorm" in results:
        select_props_unnorm, all_comb_names_unnorm_, _ = compute_and_plot_normalized_lesion(
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
    _fixed_k_val = results.get("fixed_k", 20)
    # Split select_props into input and hidden based on the all_comb structure
    # all_comb_names_leison_ has i1..iN then h1..hM (after _rename)
    _n_input_norm = len([n for n in all_comb_names_leison_ if n.startswith("i")])
    _n_hidden_norm = len([n for n in all_comb_names_leison_ if n.startswith("h")])

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

    # ── Task-specific lesion trajectories per modulation type ──
    # For each modulation type (subpanel), sort clusters by mean effect,
    # then plot each task as a line across the sorted cluster ranks.
    # This shows how individual tasks respond to each cluster's removal.
    _traj_mod_data = []
    _traj_order = [
        "modulation_all_normalized",
        "modulation_all_unnormalized",
        "modulation_all_var_weighted_unnormalized",
        "modulation_all_weighted_unnormalized",
    ]
    for bk in _traj_order:
        rkey = f"{bk}__zero_W"
        if rkey not in mod_leison_results:
            continue
        mod_data = mod_leison_results[rkey]
        mt = np.asarray(mod_data["modtask_accs"], dtype=float)
        mr = np.asarray(mod_data["modrandomtask_accs"], dtype=float)
        effect = (mr[:, 1:] - mt[:, 1:]) * 100  # (n_tasks, n_clusters)
        type_tag = bk.replace("modulation_all_", "").replace("_", "-")
        _traj_mod_data.append((type_tag, effect))

    if _traj_mod_data:
        n_panels = len(_traj_mod_data)
        fig_traj, axes_traj = plt.subplots(n_panels, 1,
                                            figsize=(5, 2.2 * n_panels), dpi=300)
        if n_panels == 1:
            axes_traj = [axes_traj]

        # Use a colormap for tasks
        _task_cmap = plt.cm.get_cmap("tab20", len(all_tasks))

        for panel_idx, (type_tag, effect) in enumerate(_traj_mod_data):
            ax = axes_traj[panel_idx]
            n_cls = effect.shape[1]

            # Each task sorts clusters by its own effect (descending)
            for ti, task in enumerate(all_tasks):
                sorted_vals = np.sort(effect[ti])[::-1]
                ax.plot(range(n_cls), sorted_vals, alpha=0.5, linewidth=0.8,
                        color=_task_cmap(ti), label=task if panel_idx == 0 else None)

            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.set_xlim(-0.5, n_cls - 0.5)
            ax.set_xlabel("Cluster rank (sorted by mean effect)", fontsize=8)
            ax.set_ylabel("Effect (%)", fontsize=8)
            ax.set_title(f"{type_tag} (zero_W)", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)

        # Single legend for all panels
        axes_traj[0].legend(fontsize=5, frameon=False, ncol=3, loc="upper right")

        fig_traj.tight_layout()
        fig_traj.savefig(f"{save_dir}/task_trajectories_mod_{aname}.png", dpi=300)
        plt.close(fig_traj)
        print("Saved modulation task trajectories plot")


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
    from matplotlib.ticker import MaxNLocator

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
    
    # ── Overmembership vs lesion difference ──
    def plot_overmembership_vs_lesion_diff(
        results, cluster_info_mod, cluster_info, variant, mod_type_key,
        mod_lesion_mode, aname, save_dir,
    ):
        """For each (mod_cluster, input_cluster, hidden_cluster) triple, scatter
        overmembership vs |task-averaged normalized lesion effect difference|
        between modulation lesion and combined (input+hidden) lesion.

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

        unres_mod_label = None

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
            if unres_mod_label is not None and cid == unres_mod_label:
                continue
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

        # --- Build scatter data ---
        om_vals = []
        lesion_diffs = []
        labels = []

        for cid in sorted(mod_effects.keys()):
            if cid not in om_id_to_idx:
                continue
            om_idx = om_id_to_idx[cid]
            mod_eff = mod_effects[cid]  # (n_tasks,)

            for pi in range(n_in):
                if pi in skip_input:
                    continue
                for qi in range(n_hid):
                    if qi in skip_hidden:
                        continue
                    om_val = om_stack[om_idx, pi, qi]
                    comb_eff = combined_effect[:, pi, qi]  # (n_tasks,)
                    diff = np.abs(np.mean(mod_eff) - np.mean(comb_eff))
                    om_vals.append(om_val)
                    lesion_diffs.append(diff)
                    labels.append(f"m{cid}_i{pi+1}_h{qi+1}")

        om_vals = np.array(om_vals)
        lesion_diffs = np.array(lesion_diffs)

        if len(om_vals) == 0:
            print(f"[om_vs_lesion] no data points to plot")
            return

        slope, intercept, r, p, _ = linregress(om_vals, lesion_diffs)

        fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
        ax.scatter(om_vals, lesion_diffs, alpha=0.5, s=20, edgecolors="none", color="steelblue")

        x_line = np.linspace(om_vals.min(), om_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.2)

        p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.05, 0.95, f"r = {r:.2f}, slope = {slope:.2f}\n{p_str}\nn = {len(om_vals)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)

        ax.set_xlabel("Over-membership")
        ax.set_ylabel("|Mean lesion effect difference|\n(mod cluster vs combined input+hidden)")
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

    def _plot_om_vs_lesion_combined(results, cluster_info_mod, cluster_info, variant,
                                    base_key, modes, aname, save_dir, single_plot_fn):
        """Plot zero_W and freeze_M overmembership vs lesion diff side-by-side."""
        type_tag = base_key.replace("modulation_all_", "").replace("_", "-")

        # Collect scatter data for each mode
        mode_data_all = {}
        for mode in ["zero_W", "freeze_M"]:
            if mode not in modes:
                continue
            # Reuse the logic from the single-plot function to get om_vals/lesion_diffs
            # by calling a data-only variant
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

            om_vals = []
            lesion_diffs = []
            for cid in sorted(mod_effects.keys()):
                if cid not in om_id_to_idx:
                    continue
                om_idx = om_id_to_idx[cid]
                mod_eff = mod_effects[cid]
                for pi in range(n_in):
                    if pi in skip_input:
                        continue
                    for qi in range(n_hid):
                        if qi in skip_hidden:
                            continue
                        om_val = om_stack[om_idx, pi, qi]
                        comb_eff = combined_effect[:, pi, qi]
                        diff = np.abs(np.mean(mod_eff) - np.mean(comb_eff))
                        om_vals.append(om_val)
                        lesion_diffs.append(diff)

            if len(om_vals) > 0:
                mode_data_all[mode] = (np.array(om_vals), np.array(lesion_diffs))

        if len(mode_data_all) < 2:
            return

        # Panel 3: per-cluster prediction — use OM-weighted combined effect to predict
        # modulation cluster's mean lesion effect (one point per cluster).
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
                    _mr_p = np.asarray(mod_data_p["modrandomtask_accs"], dtype=float)
                    for key_idx, key in enumerate(mod_data_p["all_comb_names_mod"]):
                        if key == "mod_noleison":
                            continue
                        cid = int(key.replace("mod_c", ""))
                        if cid not in om_id_to_idx_p:
                            continue
                        om_idx = om_id_to_idx_p[cid]
                        om_profile = om_stack_p[om_idx]  # (n_in, n_hid)
                        # Predicted effect: OM-weighted average of combined effects
                        if om_profile.sum() > 0:
                            predicted = (om_profile * comb_mean_p).sum() / om_profile.sum()
                        else:
                            predicted = 0.0
                        actual = (_mr_p[:, key_idx] - _mt_p[:, key_idx]).mean()
                        _pred_x.append(predicted * 100)
                        _pred_y.append(actual * 100)

        _has_pred = len(_pred_x) >= 3

        fig, axes = plt.subplots(1, 3 if _has_pred else 2,
                                 figsize=(10.5 if _has_pred else 7, 3.2), dpi=300)
        for ax, mode in zip(axes[:2], ["zero_W", "freeze_M"]):
            if mode not in mode_data_all:
                ax.set_visible(False)
                continue
            om_vals, lesion_diffs = mode_data_all[mode]
            slope, intercept, r, p, _ = linregress(om_vals, lesion_diffs)

            ax.scatter(om_vals, lesion_diffs, alpha=0.4, s=12, edgecolors="none", color="steelblue")
            x_line = np.linspace(om_vals.min(), om_vals.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.0)

            p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
            ax.text(0.05, 0.95, f"r = {r:.2f}\n{p_str}\nn = {len(om_vals)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7)
            ax.set_xlabel("Over-membership", fontsize=8)
            ax.set_ylabel("|Lesion effect diff|", fontsize=8)
            mode_tag = mode.replace("_", "-")
            ax.set_title(f"{mode_tag}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)

        # Panel 3: predicted vs actual per cluster
        if _has_pred:
            ax_pred = axes[2]
            _pred_x = np.array(_pred_x)
            _pred_y = np.array(_pred_y)
            slope_p, intercept_p, r_p, p_p, _ = linregress(_pred_x, _pred_y)

            ax_pred.scatter(_pred_x, _pred_y, alpha=0.6, s=25, edgecolors="none", color="steelblue")
            _lim_p = [min(_pred_x.min(), _pred_y.min()), max(_pred_x.max(), _pred_y.max())]
            ax_pred.plot(_lim_p, _lim_p, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
            x_fit_p = np.linspace(_pred_x.min(), _pred_x.max(), 100)
            ax_pred.plot(x_fit_p, slope_p * x_fit_p + intercept_p, color="tomato", linewidth=1.0)

            p_str_p = f"p = {p_p:.2e}" if p_p < 0.001 else f"p = {p_p:.3f}"
            ax_pred.text(0.05, 0.95, f"r = {r_p:.2f}\n{p_str_p}\nn = {len(_pred_x)}",
                         transform=ax_pred.transAxes, va="top", ha="left", fontsize=7)
            ax_pred.set_xlabel("OM-predicted effect (%)", fontsize=8)
            ax_pred.set_ylabel("Actual mod lesion effect (%)", fontsize=8)
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

    # Load cluster_info and cluster_info_mod for overmembership analysis
    cluster_path = f"./multiple_tasks/{aname}/cluster_info_{aname}.pkl"
    cluster_mod_path = f"./multiple_tasks/{aname}/cluster_info_mod_{aname}.pkl"
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
                    modes, aname, save_dir, plot_overmembership_vs_lesion_diff,
                )
            else:
                for mode in modes:
                    plot_overmembership_vs_lesion_diff(
                        results, cluster_info_mod, cluster_info, variant, base_key, mode,
                        aname, save_dir,
                    )
    else:
        print(f"[om_vs_lesion] cluster pickle(s) not found, skipping")

    # ── Cluster similarity vs normalized lesion effect ──
    def plot_cluster_corr_vs_lesion(corr_matrices_dict, select_props_mat, slices_dict,
                                    savesuffix, aname, save_dir,
                                    l1_dist_matrices_dict=None, cluster_means_dict=None,
                                    exclude_last_cluster=False):
        """3×N figure: for each cluster type (column),
        row 0 = cluster tuning cosine similarity heatmap,
        row 1 = lesion effect L1 distance heatmap,
        row 2 = scatter of tuning cosine sim vs lesion L1 distance.

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

            # Cluster tuning: cosine similarity between cluster mean profiles
            if cluster_means_dict is not None and name in cluster_means_dict:
                tuning_cos = _cosine_sim(cluster_means_dict[name].T)
            else:
                tuning_cos = _cosine_sim(np.eye(corr_matrix.shape[0]))
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
    l1_dist_norm = results["cluster_similarity"]["l1_dist_matrices"]
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
        l1_dist_matrices_dict=l1_dist_norm, cluster_means_dict=cluster_means_norm,
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
        l1_dist_unnorm = {}
        cluster_means_unnorm = {}
        _unnorm_keys = {}
        for name in ["input_unnormalized", "hidden_unnormalized"]:
            if name not in cluster_info:
                continue
            ci = cluster_info[name]
            V = ci["cell_vars_rules_sorted_norm"]
            col_clusters = _derive_fixed_k_clusters(ci, _fixed_k_plot)
            n_clusters = len(col_clusters)
            cluster_means = np.stack(
                [V[:, col_clusters[c]].mean(axis=1) for c in range(1, n_clusters + 1)],
                axis=1,
            )
            fk_name = f"{name}_k{_fixed_k_plot}"
            corr_matrices_unnorm[fk_name] = np.corrcoef(cluster_means.T)
            from scipy.spatial.distance import squareform as _sq_u, pdist as _pd_u
            l1_dist_unnorm[fk_name] = _sq_u(_pd_u(cluster_means.T, metric="cityblock"))
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
                l1_dist_matrices_dict=l1_dist_unnorm, cluster_means_dict=cluster_means_unnorm,
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

                from scipy.spatial.distance import squareform as _sq_m, pdist as _pd_m
                l1_mod = _sq_m(_pd_m(cluster_means_mod.T, metric="cityblock"))
                _is_unnorm_mod = "unnormalized" in mod_type_key
                plot_cluster_corr_vs_lesion(
                    {mod_name: corr_matrix_mod},
                    mod_select_props,
                    {mod_name: slice(0, n_mod_clusters)},
                    f"mod_leison_effect_{type_tag}_{mode_tag}", aname, save_dir,
                    l1_dist_matrices_dict={mod_name: l1_mod},
                    cluster_means_dict={mod_name: cluster_means_mod},
                    exclude_last_cluster=_is_unnorm_mod,
                )


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default=None,
                        help="Only run models with this feature (e.g. 'L21e4')")
    args = parser.parse_args()

    saved_nets = sorted(Path("multiple_tasks").glob("savednet_everything_seed*+angle.pt"))
    param_lst = []
    for p in saved_nets:
        m = re.match(r"savednet_everything_seed(\d+)_(\w+)\+hidden\d+\+batch\d+\+angle\.pt", p.name)
        if m:
            param_lst.append((int(m.group(1)), m.group(2)))

    if args.feature:
        param_lst = [(s, f) for s, f in param_lst if f == args.feature]

    print(f"Running {len(param_lst)} models: {param_lst}")

    for seed, feature in param_lst:
        main(seed, feature)
