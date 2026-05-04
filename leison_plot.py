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

        # Violin plot: distribution of normalized lesion effect across tasks per cluster
        n_clusters = len(all_comb_names_filtered)
        fig_v, ax_v = plt.subplots(
            figsize=(max(4, 0.6 * n_clusters + 1.5), 3.5), dpi=300,
        )
        violin_data = [props[:, ci] * 100 for ci in range(n_clusters)]
        parts = ax_v.violinplot(violin_data, positions=range(n_clusters),
                                showmeans=True, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("tomato")
        parts["cmeans"].set_linewidth(1.2)

        for ci in range(n_clusters):
            ax_v.scatter(
                np.full(len(violin_data[ci]), ci), violin_data[ci],
                color="black", s=8, alpha=0.4, zorder=3,
            )

        ax_v.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
        ax_v.set_xticks(range(n_clusters))
        ax_v.set_xticklabels(all_comb_names_filtered, rotation=45, ha="right", fontsize=7)
        ax_v.set_ylabel("Normalized lesion effect (%)", fontsize=8)
        ax_v.set_xlabel(f"Lesion Condition{suffix}", fontsize=8)
        ax_v.spines["top"].set_visible(False)
        ax_v.spines["right"].set_visible(False)
        fig_v.tight_layout()
        fig_v.savefig(f"{save_dir}/{savename}_violin_{aname}.png", dpi=300)
        plt.close(fig_v)

        return props, all_comb_names_filtered, ihtask

    select_props, all_comb_names_leison_, ihtask_accs = compute_and_plot_normalized_lesion(
        "leison", "random_leison", "normalized_leison",
    )
    all_comb_names_leison = results["leison"]["all_comb_names_leison"]
    all_tasks = results["leison"]["all_tasks"]

    select_props_unnorm = None
    if "leison_unnorm" in results and "random_leison_unnorm" in results:
        select_props_unnorm, _, _ = compute_and_plot_normalized_lesion(
            "leison_unnorm", "random_leison_unnorm",
            "normalized_leison_unnorm", xlabel_suffix="(unnorm)",
        )

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

        helper.plot_heatmap(
            combined_norm_effect_flat, flat_names, c_all_tasks,
            xlabel=f"Combined Lesion (input, hidden) [{vtag}]", ylabel="Task",
            savename=f"normalized_combined_leison_{vtag}",
            aname=aname, label="Normalized Accuracy",
            vmin=None, vmax=None, save_dir=save_dir,
        )

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
        ga = cluster_info_mod[mod_type_key].get("global_assignment")
        if ga is None:
            print(f"[om_vs_lesion] skipping: global_assignment is None for {mod_type_key}")
            return

        # --- Overmembership data ---
        om_stack = ga["om_stack"]                    # (N_cls_om, n_in, n_hid)
        all_choice_order = ga["all_choice_order"]    # list of cluster IDs sorted by size desc
        n_in = ga["n_in"]
        n_hid = ga["n_hid"]
        om_id_to_idx = {cid: idx for idx, cid in enumerate(all_choice_order)}

        # --- Identify unresponsive cluster indices (0-based) to exclude ---
        # For unnormalized: last cluster in input/hidden is unresponsive.
        # For normalized: skip_unresponsive_detection=True, no unresponsive cluster.
        # Modulation unresponsive cluster is already excluded from all_choice_order.
        skip_input = set()
        skip_hidden = set()
        if variant == "unnorm":
            input_key = "input_unnormalized"
            hidden_key = "hidden_unnormalized"
            if input_key in cluster_info and "result" in cluster_info[input_key]:
                input_k = cluster_info[input_key]["result"]["col_tol_k"]
                skip_input.add(input_k)  # 0-based index of unresponsive cluster
                print(f"[om_vs_lesion] excluding unresponsive input cluster (0-based idx={input_k})")
            if hidden_key in cluster_info and "result" in cluster_info[hidden_key]:
                hidden_k = cluster_info[hidden_key]["result"]["col_tol_k"]
                skip_hidden.add(hidden_k)
                print(f"[om_vs_lesion] excluding unresponsive hidden cluster (0-based idx={hidden_k})")

            # Also exclude unresponsive modulation cluster from lesion effects
            mod_result_all = cluster_info_mod[mod_type_key]["result_all_lst"]
            mod_G_idx = results["mod_leison"][mod_result_key].get("mod_G_idx", 1)
            unres_mod_label = mod_result_all[mod_G_idx]["col_k"] + 1
        else:
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

    # Load cluster_info and cluster_info_mod for overmembership analysis
    cluster_path = f"./multiple_tasks/{aname}/cluster_info_{aname}.pkl"
    cluster_mod_path = f"./multiple_tasks/{aname}/cluster_info_mod_{aname}.pkl"
    if os.path.exists(cluster_mod_path) and os.path.exists(cluster_path):
        with open(cluster_mod_path, "rb") as f:
            cluster_info_mod = pickle.load(f)
        with open(cluster_path, "rb") as f:
            cluster_info = pickle.load(f)

        for mod_result_key in results["mod_leison"]:
            base_key, mode = mod_result_key.rsplit("__", 1)
            if "normalized" in base_key and "unnormalized" not in base_key:
                variant = "norm"
            else:
                variant = "unnorm"
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
    pre_n = len(corr_matrices_norm["input_normalized"])
    post_n = len(corr_matrices_norm["hidden_normalized"])
    slices_norm = {
        "input_normalized":  slice(0, pre_n),
        "hidden_normalized": slice(pre_n, pre_n + post_n),
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

        corr_matrices_unnorm = {}
        for name in ["input_unnormalized", "hidden_unnormalized"]:
            if name not in cluster_info:
                continue
            ci = cluster_info[name]
            V = ci["cell_vars_rules_sorted_norm"]
            col_clusters = ci["col_clusters"]
            n_clusters = len(col_clusters)
            cluster_means = np.stack(
                [V[:, col_clusters[c]].mean(axis=1) for c in range(1, n_clusters + 1)],
                axis=1,
            )
            corr_matrices_unnorm[name] = np.corrcoef(cluster_means.T)

        if "input_unnormalized" in corr_matrices_unnorm and "hidden_unnormalized" in corr_matrices_unnorm:
            pre_n_u = len(corr_matrices_unnorm["input_unnormalized"])
            post_n_u = len(corr_matrices_unnorm["hidden_unnormalized"])
            slices_unnorm = {
                "input_unnormalized":  slice(0, pre_n_u),
                "hidden_unnormalized": slice(pre_n_u, pre_n_u + post_n_u),
            }
            plot_cluster_corr_vs_lesion(
                corr_matrices_unnorm, select_props_unnorm, slices_unnorm,
                "normalized_leison_effect_unnorm", aname, save_dir,
            )

    # --- Modulation variant ---
    # For each modulation clustering type × lesion mode, compute cluster similarity
    # from cell_vars_rules_sorted_norm + col_labels at G=300 (index 1), then compare
    # against the normalized modulation lesion effect.
    MOD_G_IDX = 1
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
            result_all = mod_ci["result_all_lst"][MOD_G_IDX]
            col_labels_mod = result_all["col_labels"]

            # Build cluster-mean profiles (same approach as input/hidden)
            unique_labels = sorted(set(int(l) for l in col_labels_mod))
            col_clusters_mod = {lab: np.where(col_labels_mod == lab)[0] for lab in unique_labels}
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
