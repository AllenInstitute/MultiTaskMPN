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

def main(seed, feature):
    """
    """
    addname = f"result_all_everything_seed{seed}_{feature}+hidden300+batch128+angle"
    paths = [
        f"modulation_all_clustering_{addname}_normalized.pkl", 
        f"modulation_all_clustering_{addname}_unnormalized.pkl", 
        f"modulation_all_weighted_clustering_{addname}_unnormalized.pkl"
    ]
    
    os.makedirs("multiple_task_heatmaps", exist_ok=True)
    
    for idx, path in enumerate(paths):
        print(f"Processing {path}...")
        with open("multiple_tasks/" + path, "rb") as f: 
            data = pickle.load(f)
        
        select = 1
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
        
        # for idx == 2: also plot with the largest cluster removed
        if idx == 2 and len(yboundary) > 0:
            edges = [0] + list(yboundary) + [plot_data.shape[1]]
            cluster_sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
            largest_idx = int(np.argmax(cluster_sizes))

            # remove the largest cluster's column range and rebase boundaries
            plot_data_excl = np.concatenate([
                plot_data[:, :edges[largest_idx]],
                plot_data[:, edges[largest_idx + 1]:]
            ], axis=1)
            remaining_sizes = [s for i, s in enumerate(cluster_sizes) if i != largest_idx]
            yboundary_excl = list(np.cumsum(remaining_sizes)[:-1])

            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
            sns.heatmap(
                plot_data_excl,
                ax=ax2,
                cmap="coolwarm",
                cbar=True,
                norm=norm,
                linewidths=0,
            )
            for x in xboundary:
                ax2.axhline(x, color="black", linestyle="-", linewidth=1.5, alpha=0.9, zorder=10)
            for y in yboundary_excl:
                ax2.axvline(y, color="black", linestyle="-", linewidth=1.5, alpha=0.9, zorder=10)
            ax2.set_title(f"Largest Cluster Removed (cluster {largest_idx}, size {cluster_sizes[largest_idx]}); Neuron Class Number: {len(yboundary_excl)+1}")
            fig2.tight_layout()
            fig2.savefig(
                os.path.join("multiple_task_heatmaps", path.replace(".pkl", "_largest_removed.png")),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig2)
        
if __name__ == "__main__":
    seed_lst = [921, 749, 842, 408]
    # seed_lst = [921]
    for seed in seed_lst:
        main(seed, "L21e4")