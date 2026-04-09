import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

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

import seaborn as sns 
import pickle 
import numpy as np 
import os 

if __name__ == "__main__":
    paths = [
        "modulation_all_clustering_result_all_everything_seed749_L21e4+hidden300+batch128+angle_normalized.pkl", 
        "modulation_all_clustering_result_all_everything_seed749_L21e4+hidden300+batch128+angle_unnormalized.pkl", 
        "modulation_all_weighted_clustering_result_all_everything_seed749_L21e4+hidden300+batch128+angle_unnormalized.pkl"
    ]
    
    os.makedirs("multiple_task_heatmaps", exist_ok=True)
    
    for idx, path in enumerate(paths):
        print(f"Processing {path}...")
        with open("multiple_tasks/" + path, "rb") as f: 
            data = pickle.load(f)
        
        select = 2 
        heatmap = data["cell_vars_rules_sorted_norm_all_lst"][select]
        boundaries = data["modulation_cluster_boundary"][select]
        xboundary, yboundary = boundaries[0], boundaries[1]
        
        plot_data = np.asarray(heatmap, dtype=float)

        # normalization rule
        if idx >= 1:
            # LogNorm requires strictly positive values
            vmax = np.nanmax(plot_data)
            if vmax <= 0:
                raise ValueError("For idx >= 1, heatmap must contain at least one positive value for LogNorm.")
            
            plot_data = np.clip(plot_data, 1e-3, None)
            norm = LogNorm(vmin=1e-3, vmax=vmax)
        else:
            norm = Normalize(vmin=0, vmax=1)

        fig, ax = plt.subplots(1,1,figsize=(10,4))

        sns.heatmap(
            plot_data,
            ax=ax,
            cmap="coolwarm",
            cbar=True,
            norm=norm,
            linewidths=0,
        )

        # clearer separators
        for x in xboundary:
            ax.axhline(x, color="black", linestyle="-", linewidth=1.5, alpha=0.9, zorder=10)

        for y in yboundary:
            ax.axvline(y, color="black", linestyle="-", linewidth=1.5, alpha=0.9, zorder=10)
        
        ax.set_title(f"Neuron Class Number: {len(yboundary)+1}")
        fig.tight_layout()
        fig.savefig(
            os.path.join("multiple_task_heatmaps", path.replace(".pkl", ".png")),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)
        
        # for idx == 2: also plot the portion after the first vertical cutoff
        if idx == 2 and len(yboundary) > 0:
            first_cut = yboundary[0]
            plot_data_right = plot_data[:, first_cut:]

            # rebase yboundary to the cropped region
            yboundary_right = [y - first_cut for y in yboundary[1:]]

            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
            sns.heatmap(
                plot_data_right,
                ax=ax2,
                cmap="coolwarm",
                cbar=True,
                norm=norm,
                linewidths=0,
            )
            for x in xboundary:
                ax2.axhline(x, color="black", linestyle="-", linewidth=1.5, alpha=0.9, zorder=10)
            for y in yboundary_right:
                ax2.axvline(y, color="black", linestyle="-", linewidth=1.5, alpha=0.9, zorder=10)
            ax2.set_title(f"After First Vertical Cut (col {first_cut}+); Neuron Class Number: {len(yboundary_right)+1}")
            fig2.tight_layout()
            fig2.savefig(
                os.path.join("multiple_task_heatmaps", path.replace(".pkl", "_after_first_cut.png")),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig2)
        
        