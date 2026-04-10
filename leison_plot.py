import numpy as np
import pandas as pd

import pickle
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib.ticker import MaxNLocator

import leison as leison_helper
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
    
    pickle_name = f"./multiple_tasks_perf/lesion_prune_results_{aname}.pkl"
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
                        vmin=None, vmax=None)

    # # normalized modulation lesion effect
    # mod_baseline_keys = {"mod_noleison"}
    # print(results.keys())
    # all_comb_names_mod = results["mod_leison"]["all_comb_names_mod"]
    # all_comb_names_mod_ = [k for k in all_comb_names_mod if k not in mod_baseline_keys]
    # modtask_accs = np.asarray(results["mod_leison"]["modtask_accs"], dtype=float)
    # modrandomtask_accs = np.asarray(results["mod_leison"]["modrandomtask_accs"], dtype=float)

    # mod_select_props = []
    # for key_idx, key in enumerate(all_comb_names_mod):
    #     if key not in mod_baseline_keys:
    #         leison = modtask_accs[:, key_idx]
    #         random_leison = modrandomtask_accs[:, key_idx]
    #         normalized_leison_diff = random_leison - leison
    #         mod_select_props.append(normalized_leison_diff)

    # mod_select_props = np.array(mod_select_props).T
    # print(f"mod_select_props: {mod_select_props.shape}")

    # helper.plot_heatmap(mod_select_props, all_comb_names_mod_, all_tasks,
    #                     xlabel="Modulation Lesion Condition", ylabel="Task",
    #                     savename="normalized_mod_leison",
    #                     aname=aname, label="Normalized Accuracy",
    #                     vmin=None, vmax=None)
    
    # this part focuses on input & hidden analysis only 
    # drop the two baseline columns from ihtask_accs so its layout matches select_props:
    #   keep only columns where the condition is not a no-lesion baseline
    keep_cols = [i for i, k in enumerate(all_comb_names_leison) if k not in baseline_keys]
    ihtask_accs_no_baseline = ihtask_accs[:, keep_cols]  # (n_tasks, pre_n + post_n)
    print(f"ihtask_accs: {ihtask_accs.shape} -> no_baseline: {ihtask_accs_no_baseline.shape}; select_props: {select_props.shape}")

    corr_matrices = results["cluster_similarity"]["corr_matrices"]
    pre_n = len(corr_matrices["input"])
    post_n = len(corr_matrices["hidden"])
    # columns [0:pre_n] = input clusters, [pre_n:pre_n+post_n] = hidden clusters
    slices = {
        "input":  slice(0, pre_n),
        "hidden": slice(pre_n, pre_n + post_n),
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
        fig.savefig(f"./multiple_tasks_perf/cluster_corr_vs_{savesuffix}_{aname}.png", dpi=300)
        plt.close(fig)  
        
        
            
    
    