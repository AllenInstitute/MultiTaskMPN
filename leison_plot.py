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

if __name__ == "__main__": 
    aname = "everything_seed749_L21e4+hidden300+batch128+angle"

    save_dir = f"./multiple_tasks_perf/{aname}"
    pickle_name = f"{save_dir}/lesion_prune_results_{aname}.pkl"
    with open(pickle_name, 'rb') as f:
        results = pickle.load(f)
        
    # handle both old pickle names ("pre_cNone") and new ("pre_noleison") after rename fix
    baseline_keys = {"pre_cNone", "post_cNone", "pre_noleison", "post_noleison"}

    all_comb_names_leison = results["leison"]["all_comb_names_leison"]
    all_comb_names_leison_ = [k for k in all_comb_names_leison if k not in baseline_keys]
    all_tasks = results["leison"]["all_tasks"]
    
    ihtask_accs = np.asarray(results["leison"]["ihtask_accs"], dtype=float)
    ihrandomtask_accs = np.asarray(results["random_leison"]["ihrandomtask_accs"], dtype=float)

    select_props = []
    for key_idx, key in enumerate(all_comb_names_leison):
        if key not in baseline_keys:
            leison = ihtask_accs[:, key_idx]
            random_leison = ihrandomtask_accs[:, key_idx]
            noleison = ihtask_accs[:, 0]  # first column means no leison
            random_leison_diff = noleison - random_leison
            normalized_leison_diff = random_leison - leison
            select_props.append(normalized_leison_diff)

    select_props = np.array(select_props).T
    print(f"select_props: {select_props.shape}")

    helper.plot_heatmap(select_props, all_comb_names_leison_, all_tasks,
                        xlabel="Lesion Condition", ylabel="Task", savename="normalized_leison",
                        aname=aname, label="Normalized Accuracy",
                        vmin=None, vmax=None, save_dir=save_dir)

    # Normalized modulation lesion effect for each clustering type
    mod_baseline_keys = {"mod_noleison"}
    mod_leison_results = results["mod_leison"]

    # First pass: compute normalized effect for each entry and plot individual heatmaps.
    # Also collect results grouped by clustering type for cross-method comparison.
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
                leison = modtask_accs[:, key_idx]
                random_leison = modrandomtask_accs[:, key_idx]
                mod_select_props.append(random_leison - leison)

        mod_select_props = np.array(mod_select_props).T

        if "__" in mod_type_key:
            base_key, mode = mod_type_key.rsplit("__", 1)
            type_tag = base_key.replace("modulation_all_", "").replace("_", "-") + "_" + mode.replace("_", "-")
        else:
            base_key = mod_type_key
            mode = mod_data.get("mod_lesion_mode", "zero_W")
            type_tag = base_key.replace("modulation_all_", "").replace("_", "-")

        lesion_mode_label = mod_data.get("mod_lesion_mode", "zero_W")
        print(f"[{mod_type_key}] mod_select_props: {mod_select_props.shape}")

        helper.plot_heatmap(mod_select_props, all_comb_names_mod_, all_tasks,
                            xlabel=f"Modulation Lesion Condition ({lesion_mode_label})",
                            ylabel="Task",
                            savename=f"normalized_mod_leison_{type_tag}",
                            aname=aname, label="Normalized Accuracy",
                            vmin=None, vmax=None, save_dir=save_dir)

        mod_by_type[base_key][mode] = {
            "select_props": mod_select_props,
            "cluster_names": all_comb_names_mod_,
        }

    # Second pass: for each clustering type that has both lesion modes,
    # plot a side-by-side comparison (zero_W | freeze_M | difference).
    for base_key, modes_dict in mod_by_type.items():
        if len(modes_dict) < 2:
            continue
        if "zero_W" not in modes_dict or "freeze_M" not in modes_dict:
            continue

        zw = modes_dict["zero_W"]["select_props"]
        fm = modes_dict["freeze_M"]["select_props"]
        cluster_names = modes_dict["zero_W"]["cluster_names"]
        diff = fm - zw

        vabs = max(np.nanmax(np.abs(zw)), np.nanmax(np.abs(fm)))
        dabs = np.nanmax(np.abs(diff))

        base_tag = base_key.replace("modulation_all_", "").replace("_", "-")

        fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 0.4 * len(all_tasks) + 1.5))

        panels = [
            (axes[0], zw,   "zero_W",  "RdBu_r", -vabs, vabs),
            (axes[1], fm,   "freeze_M", "RdBu_r", -vabs, vabs),
            (axes[2], diff, "freeze_M − zero_W", "PiYG", -dabs, dabs),
        ]
        for ax, mat, title, cmap, vlo, vhi in panels:
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi)
            fig.colorbar(im, ax=ax, shrink=0.6, label="Normalized Accuracy")
            ax.set_xticks(range(len(cluster_names)))
            ax.set_xticklabels(cluster_names, rotation=45, ha="right")
            ax.set_yticks(range(len(all_tasks)))
            ax.set_yticklabels(all_tasks)
            ax.set_xlabel("Modulation Cluster")
            ax.set_ylabel("Task")
            ax.set_title(f"{base_tag}: {title}")

        fig.tight_layout()
        fig.savefig(f"{save_dir}/normalized_mod_leison_compare_{base_tag}_{aname}.png", dpi=300)
        plt.close(fig)
        print(f"Saved comparison plot for {base_key}")
    
    # this part focuses on input & hidden analysis only 
    # drop the two baseline columns from ihtask_accs so its layout matches select_props:
    #   keep only columns where the condition is not a no-lesion baseline
    keep_cols = [i for i, k in enumerate(all_comb_names_leison) if k not in baseline_keys]
    ihtask_accs_no_baseline = ihtask_accs[:, keep_cols]  # (n_tasks, pre_n + post_n)
    print(f"ihtask_accs: {ihtask_accs.shape} -> no_baseline: {ihtask_accs_no_baseline.shape}; select_props: {select_props.shape}")

    corr_matrices = results["cluster_similarity"]["corr_matrices"]
    pre_n = len(corr_matrices["input_normalized"])
    post_n = len(corr_matrices["hidden_normalized"])
    # columns [0:pre_n] = input clusters, [pre_n:pre_n+post_n] = hidden clusters
    slices = {
        "input_normalized":  slice(0, pre_n),
        "hidden_normalized": slice(pre_n, pre_n + post_n),
    }

    metrics = [
        ("normalized_leison_effect", select_props,            "Normalized lesion effect correlation"),
        ("raw_accuracy",             ihtask_accs_no_baseline, "Raw accuracy correlation"),
    ]

    for savesuffix, data_mat, ylabel_label in metrics:
        fig, axs = plt.subplots(3, 2, figsize=(9, 11))

        for col, (name, corr_matrix) in enumerate(corr_matrices.items()):
            data_corr = np.corrcoef(data_mat[:, slices[name]].T)  # (n_clusters, n_clusters)

            n = corr_matrix.shape[0]
            tril_idx = np.tril_indices(n, k=-1)
            cluster_labels = [str(i) for i in range(n)]

            # --- Row 0: corr_matrix lower-triangle heatmap ---
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

            # --- Row 1: data_corr lower-triangle heatmap ---
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
            ax1.set_title(f"{name}: {ylabel_label} (data_corr)")

            # --- Row 2: scatter + linear regression ---
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
            ax2.set_ylabel(ylabel_label)
            ax2.set_title(f"{name} clusters: scatter")

        fig.tight_layout()
        fig.savefig(f"{save_dir}/cluster_corr_vs_{savesuffix}_{aname}.png", dpi=300)
        plt.close(fig)  
        
        
            
    
    