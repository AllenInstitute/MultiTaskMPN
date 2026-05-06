import os
from pathlib import Path
import numpy as np
import pandas as pd

import pickle
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl

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

    # Combined violin: normalized + unnormalized input/hidden in one figure (2 panels)
    if select_props_unnorm is not None:
        _ih_panels = [
            ("normalized", select_props, all_comb_names_leison_),
            ("unnormalized", select_props_unnorm, all_comb_names_unnorm_),
        ]
        max_cls = max(p.shape[1] for _, p, _ in _ih_panels)
        _all_ih_vals = np.concatenate([p.ravel() * 100 for _, p, _ in _ih_panels])
        _ih_ylim = (min(_all_ih_vals.min() * 1.1, -1), max(_all_ih_vals.max() * 1.1, 1))

        fig_w = max(4, 0.45 * max_cls + 1.5)
        fig_ih, axes_ih = plt.subplots(2, 1, figsize=(fig_w, 2.2 * 2), dpi=300)

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
            ax_v.set_xticklabels(cnames_panel, rotation=45, ha="right", fontsize=6)
            ax_v.set_ylim(_ih_ylim)
            ax_v.set_ylabel("Effect (%)", fontsize=7)
            ax_v.set_title(f"Input & Hidden ({label})", fontsize=8)
            ax_v.spines["top"].set_visible(False)
            ax_v.spines["right"].set_visible(False)
            ax_v.tick_params(labelsize=6)

        fig_ih.tight_layout()
        fig_ih.savefig(f"{save_dir}/normalized_leison_violin_all_{aname}.png", dpi=300)
        plt.close(fig_ih)
        print("Saved combined input/hidden violin plot (2 panels)")

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
        from scipy.stats import mannwhitneyu
        import seaborn as sns

        _hist_colors = {"Input (norm)": "#2171b5", "Hidden (norm)": "#cb181d",
                        "Input (unnorm)": "#6baed6", "Hidden (unnorm)": "#fc9272"}
        _all_hist_vals = np.concatenate(list(_hist_data.values()))
        _hist_bins = np.linspace(_all_hist_vals.min(), _all_hist_vals.max(), 15)

        _ih_labels = list(_hist_data.keys())
        _ih_n = len(_ih_labels)
        _ih_pvals = np.full((_ih_n, _ih_n), np.nan)
        for i in range(1, _ih_n):
            for j in range(i):
                _, p = mannwhitneyu(_hist_data[_ih_labels[i]], _hist_data[_ih_labels[j]],
                                    alternative="less")
                _ih_pvals[i, j] = p

        fig_hp, (ax_hist, ax_p) = plt.subplots(1, 2, figsize=(7.5, 3), dpi=300,
                                                gridspec_kw={"width_ratios": [1.3, 1]})
        for label, vals in _hist_data.items():
            _mean = np.mean(vals)
            _med = np.median(vals)
            _lbl = f"{label} (μ={_mean:.2f}, md={_med:.2f})"
            ax_hist.hist(vals, bins=_hist_bins, alpha=0.5, label=_lbl, color=_hist_colors.get(label))
        ax_hist.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_hist.set_xlabel("Mean normalized lesion effect (%)", fontsize=8)
        ax_hist.set_ylabel("# Clusters", fontsize=8)
        ax_hist.legend(fontsize=6, frameon=False)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.tick_params(labelsize=7)

        from matplotlib.colors import LogNorm
        _lower_mask = np.triu(np.ones((_ih_n, _ih_n), dtype=bool), k=0)
        _pvals_plot = np.where(np.isnan(_ih_pvals), 1.0, _ih_pvals)
        _pvals_plot = np.clip(_pvals_plot, 1e-10, 1.0)
        sns.heatmap(_pvals_plot, mask=_lower_mask, annot=True, fmt=".2g",
                    cmap="YlOrRd_r", norm=LogNorm(vmin=1e-4, vmax=1.0),
                    xticklabels=_ih_labels, yticklabels=_ih_labels,
                    cbar_kws={"label": "p-value", "shrink": 0.8}, ax=ax_p,
                    linewidths=0.5, linecolor="white", square=True)
        ax_p.set_xticklabels(_ih_labels, rotation=45, ha="right", fontsize=6)
        ax_p.set_yticklabels(_ih_labels, rotation=0, fontsize=6)
        ax_p.set_title("MW-U (row < col)", fontsize=8)
        ax_p.tick_params(labelsize=6)

        fig_hp.tight_layout()
        fig_hp.savefig(f"{save_dir}/normalized_leison_hist_mean_{aname}.png", dpi=300)
        plt.close(fig_hp)
        print("Saved input/hidden histogram + p-value heatmap")

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
    mod_leison_results = results["mod_leison"]

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
        fig_v, axes_v = plt.subplots(n_panels, 1, figsize=(fig_w, 2.2 * n_panels), dpi=300)
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
            ax_v.set_xticklabels(cnames, rotation=45, ha="right", fontsize=6)
            ax_v.set_ylim(_ylim)
            ax_v.set_ylabel("Effect (%)", fontsize=7)
            ax_v.set_title(f"{type_tag} (zero_W)", fontsize=8)
            ax_v.spines["top"].set_visible(False)
            ax_v.spines["right"].set_visible(False)
            ax_v.tick_params(labelsize=6)

        fig_v.tight_layout()
        fig_v.savefig(f"{save_dir}/normalized_mod_leison_violin_all_{aname}.png", dpi=300)
        plt.close(fig_v)
        print(f"Saved combined modulation violin plot ({n_panels} panels)")

        # Histogram + p-value heatmap for modulation (one figure, two panels)
        from scipy.stats import mannwhitneyu as _mwu_mod
        import seaborn as _sns_mod

        _mod_hist_colors = {
            "normalized": "#1b9e77",
            "unnormalized": "#d95f02",
            "var-weighted-unnormalized": "#e7298a",
            "weighted-unnormalized": "#7570b3",
        }
        _mod_labels = [bk.replace("modulation_all_", "").replace("_", "-") for bk, _ in _mod_violin_data]
        _mod_means_lst = [d["select_props"].mean(axis=0) * 100 for _, d in _mod_violin_data]
        _all_mod_means = np.concatenate(_mod_means_lst)
        _mod_hist_bins = np.linspace(_all_mod_means.min(), _all_mod_means.max(), 15)

        _mod_n = len(_mod_labels)
        _mod_pvals = np.full((_mod_n, _mod_n), np.nan)
        for i in range(1, _mod_n):
            for j in range(i):
                _, p = _mwu_mod(_mod_means_lst[i], _mod_means_lst[j], alternative="less")
                _mod_pvals[i, j] = p

        fig_mhp, (ax_mhist, ax_mp) = plt.subplots(1, 2, figsize=(7.5, 3), dpi=300,
                                                    gridspec_kw={"width_ratios": [1.3, 1]})
        for bk, mode_data in _mod_violin_data:
            type_tag = bk.replace("modulation_all_", "").replace("_", "-")
            mean_per_cluster = mode_data["select_props"].mean(axis=0) * 100
            _mean = np.mean(mean_per_cluster)
            _med = np.median(mean_per_cluster)
            _lbl = f"{type_tag} (μ={_mean:.2f}, md={_med:.2f})"
            ax_mhist.hist(mean_per_cluster, bins=_mod_hist_bins, alpha=0.5,
                          label=_lbl, color=_mod_hist_colors.get(type_tag, None))
        ax_mhist.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_mhist.set_xlabel("Mean normalized lesion effect (%)", fontsize=8)
        ax_mhist.set_ylabel("# Clusters", fontsize=8)
        ax_mhist.legend(fontsize=6, frameon=False)
        ax_mhist.spines["top"].set_visible(False)
        ax_mhist.spines["right"].set_visible(False)
        ax_mhist.tick_params(labelsize=7)

        from matplotlib.colors import LogNorm as _LogNorm_mod
        _mod_lower_mask = np.triu(np.ones((_mod_n, _mod_n), dtype=bool), k=0)
        _mod_pvals_plot = np.where(np.isnan(_mod_pvals), 1.0, _mod_pvals)
        _mod_pvals_plot = np.clip(_mod_pvals_plot, 1e-10, 1.0)
        _sns_mod.heatmap(_mod_pvals_plot, mask=_mod_lower_mask, annot=True, fmt=".2g",
                         cmap="YlOrRd_r", norm=_LogNorm_mod(vmin=1e-4, vmax=1.0),
                         xticklabels=_mod_labels, yticklabels=_mod_labels,
                         cbar_kws={"label": "p-value", "shrink": 0.8}, ax=ax_mp,
                         linewidths=0.5, linecolor="white", square=True)
        ax_mp.set_xticklabels(_mod_labels, rotation=45, ha="right", fontsize=6)
        ax_mp.set_yticklabels(_mod_labels, rotation=0, fontsize=6)
        ax_mp.set_title("MW-U (row < col)", fontsize=8)
        ax_mp.tick_params(labelsize=6)

        fig_mhp.tight_layout()
        fig_mhp.savefig(f"{save_dir}/normalized_mod_leison_hist_mean_{aname}.png", dpi=300)
        plt.close(fig_mhp)
        print("Saved modulation histogram + p-value heatmap")

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
            ax_hm.set_xticklabels(cluster_names, rotation=45, ha="right", fontsize=7)
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

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), dpi=300)
        for ax, mode in zip(axes, ["zero_W", "freeze_M"]):
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

        fig.suptitle(f"OM vs lesion diff — {type_tag} [{variant}]", fontsize=9)
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
                                    savesuffix, aname, save_dir):
        """3×N figure: for each cluster type (column),
        row 0 = cluster similarity heatmap, row 1 = lesion effect correlation heatmap,
        row 2 = scatter of similarity vs lesion effect correlation."""
        n_cols = len(corr_matrices_dict)
        fig, axs = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 11), dpi=300,
                                squeeze=False)

        for col, (name, corr_matrix) in enumerate(corr_matrices_dict.items()):
            data_corr = np.corrcoef(select_props_mat[:, slices_dict[name]].T)

            n = corr_matrix.shape[0]
            tril_idx = np.tril_indices(n, k=-1)
            cluster_labels = [str(i) for i in range(n)]

            ax0 = axs[0, col]
            mat0 = np.full((n, n), np.nan)
            mat0[tril_idx] = corr_matrix[tril_idx]
            im0 = ax0.imshow(mat0, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                             origin="upper")
            fig.colorbar(im0, ax=ax0, shrink=0.8, label="Correlation")
            ax0.set_xticks(range(n))
            ax0.set_yticks(range(n))
            ax0.set_xticklabels(cluster_labels)
            ax0.set_yticklabels(cluster_labels)
            ax0.set_xlabel("Cluster index")
            ax0.set_ylabel("Cluster index")
            ax0.set_title(f"{name}: cluster similarity (corr_matrix)")

            ax1 = axs[1, col]
            mat1 = np.full((n, n), np.nan)
            mat1[tril_idx] = data_corr[tril_idx]
            im1 = ax1.imshow(mat1, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                             origin="upper")
            fig.colorbar(im1, ax=ax1, shrink=0.8, label="Correlation")
            ax1.set_xticks(range(n))
            ax1.set_yticks(range(n))
            ax1.set_xticklabels(cluster_labels)
            ax1.set_yticklabels(cluster_labels)
            ax1.set_xlabel("Cluster index")
            ax1.set_ylabel("Cluster index")
            ax1.set_title(f"{name}: Normalized lesion effect correlation")

            ax2 = axs[2, col]
            x = corr_matrix[tril_idx]
            y = data_corr[tril_idx]
            ax2.scatter(x, y, alpha=0.6, s=30, edgecolors="none", color="steelblue")

            slope, intercept, r, p, _ = linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax2.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.2)

            p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
            ax2.text(0.05, 0.95, f"r = {r:.2f}\n{p_str}", transform=ax2.transAxes,
                     va="top", ha="left", fontsize=8)
            ax2.set_xlabel(f"{name} cluster similarity (corr_matrix)")
            ax2.set_ylabel("Normalized lesion effect correlation")
            ax2.set_title(f"{name} clusters: scatter")

        fig.tight_layout()
        fig.savefig(f"{save_dir}/cluster_corr_vs_{savesuffix}_{aname}.png", dpi=300)
        plt.close(fig)
        print(f"Saved cluster_corr_vs_{savesuffix}")

    # --- Normalized variant ---
    corr_matrices_norm = results["cluster_similarity"]["corr_matrices"]
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
    )

    # --- Unnormalized variant ---
    # Compute cluster similarity on-the-fly from cluster_info (not in the lesion pickle)
    if select_props_unnorm is not None and os.path.exists(cluster_path):
        try:
            cluster_info
        except NameError:
            with open(cluster_path, "rb") as f:
                cluster_info = pickle.load(f)

        # Derive fixed-k clusters on the fly (same as leison.py) to match select_props_unnorm
        from scipy.cluster.hierarchy import fcluster as _fcluster_plot
        _fixed_k_plot = results.get("fixed_k", 20)

        def _derive_fk_clusters_plot(ci_entry, fk):
            res = ci_entry["result"]
            lnk = res["col_linkage"]
            tol_k = res["col_tol_k"]
            tol_labels = res["col_tol_labels"]
            unres_mask = tol_labels == (tol_k + 1)
            active_labels = _fcluster_plot(lnk, fk, criterion="maxclust")
            full_labels = np.zeros(len(tol_labels), dtype=int)
            full_labels[~unres_mask] = active_labels
            if unres_mask.any():
                full_labels[unres_mask] = fk + 1
            return {int(lab): np.where(full_labels == lab)[0] for lab in np.unique(full_labels) if lab > 0}

        corr_matrices_unnorm = {}
        _unnorm_keys = {}
        for name in ["input_unnormalized", "hidden_unnormalized"]:
            if name not in cluster_info:
                continue
            ci = cluster_info[name]
            V = ci["cell_vars_rules_sorted_norm"]
            col_clusters = _derive_fk_clusters_plot(ci, _fixed_k_plot)
            n_clusters = len(col_clusters)
            cluster_means = np.stack(
                [V[:, col_clusters[c]].mean(axis=1) for c in range(1, n_clusters + 1)],
                axis=1,
            )
            fk_name = f"{name}_k{_fixed_k_plot}"
            corr_matrices_unnorm[fk_name] = np.corrcoef(cluster_means.T)
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

                plot_cluster_corr_vs_lesion(
                    {mod_name: corr_matrix_mod},
                    mod_select_props,
                    {mod_name: slice(0, n_mod_clusters)},
                    f"mod_leison_effect_{type_tag}_{mode_tag}", aname, save_dir,
                )


if __name__ == "__main__":
    main(749, "L21e4")
