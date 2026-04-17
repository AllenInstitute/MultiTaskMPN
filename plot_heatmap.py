import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from pathlib import Path

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
    # modulation + unnormalized modulation + weighted modulation (unnormalized)
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
        
        # for idx == 2: grid of block heatmaps + column-cluster sum filtered heatmap
        if idx == 2 and len(yboundary) > 0:
            col_edges = [0] + list(yboundary) + [plot_data.shape[1]]
            row_edges = [0] + list(xboundary) + [plot_data.shape[0]]
            n_col_cls = len(col_edges) - 1
            n_row_cls = len(row_edges) - 1

            # Mean of each (row_cluster × col_cluster) block; shape (n_row_cls, n_col_cls)
            block_means = np.array([
                [plot_data[row_edges[ri]:row_edges[ri + 1],
                           col_edges[ci]:col_edges[ci + 1]].mean()
                 for ci in range(n_col_cls)]
                for ri in range(n_row_cls)
            ])
            # Score each column cluster by summing its block means across row clusters
            # (size-independent, unlike raw entry sum)
            col_sums = block_means.sum(axis=0)   # shape (n_col_cls,)
            fig_grid, ax_grid = plt.subplots(
                1, 1, figsize=(max(4, n_col_cls), max(2, n_row_cls))
            )
            sns.heatmap(block_means, ax=ax_grid, cmap="coolwarm", cbar=True,
                        norm=norm, linewidths=0.5, annot=True, fmt=".2g",
                        xticklabels=[f"C{ci}" for ci in range(n_col_cls)],
                        yticklabels=[f"R{ri}" for ri in range(n_row_cls)])
            ax_grid.set_title(
                f"Block means — {n_row_cls} row × {n_col_cls} col clusters\n"
                f"col sums: {', '.join(f'C{ci}={col_sums[ci]:.2g}' for ci in range(n_col_cls))}",
                fontsize=8
            )
            fig_grid.tight_layout()
            fig_grid.savefig(
                os.path.join("multiple_task_heatmaps", path.replace(".pkl", "_block_grid.png")),
                dpi=200, bbox_inches="tight"
            )
            plt.close(fig_grid)

            # --- Figure 2: remove column clusters whose sum is below threshold ---
            threshold_frac = 0.10   # remove if sum < 10% of max column-cluster sum
            threshold_val  = threshold_frac * col_sums.max()
            kept    = np.where(col_sums >= threshold_val)[0]
            removed = np.where(col_sums <  threshold_val)[0]

            if len(kept) == 0:
                kept = np.array([int(np.argmax(col_sums))])   # always keep at least one
                removed = np.array([i for i in range(n_col_cls) if i not in kept])

            kept_blocks = [plot_data[:, col_edges[ci]:col_edges[ci + 1]] for ci in kept]
            plot_data_filt = np.concatenate(kept_blocks, axis=1)
            kept_sizes     = [col_edges[ci + 1] - col_edges[ci] for ci in kept]
            yboundary_filt = list(np.cumsum(kept_sizes)[:-1])

            kept_desc = ", ".join(
                f"C{ci}(size={col_edges[ci+1]-col_edges[ci]}, sum={col_sums[ci]:.2g})"
                for ci in kept
            )
            removed_desc = (
                ", ".join(
                    f"C{ci}(size={col_edges[ci+1]-col_edges[ci]}, sum={col_sums[ci]:.2g})"
                    for ci in removed
                )
                if len(removed) > 0 else "none"
            )
            title_filt = (
                f"Col clusters kept: {len(kept)}/{n_col_cls}  "
                f"(threshold={threshold_frac:.0%} × max={col_sums.max():.3g})\n"
                f"Kept: {kept_desc}\n"
                f"Removed: {removed_desc}"
            )

            fig_filt, ax_filt = plt.subplots(1, 1, figsize=(10, 4))
            sns.heatmap(
                plot_data_filt, ax=ax_filt, cmap="coolwarm",
                cbar=True, norm=norm, linewidths=0,
            )
            for x in xboundary:
                ax_filt.axhline(x, color="black", linestyle="-", linewidth=1.5,
                                alpha=0.9, zorder=10)
            for y in yboundary_filt:
                ax_filt.axvline(y, color="black", linestyle="-", linewidth=1.5,
                                alpha=0.9, zorder=10)
            ax_filt.set_title(title_filt, fontsize=8)
            fig_filt.tight_layout()
            fig_filt.savefig(
                os.path.join("multiple_task_heatmaps", path.replace(".pkl", "_col_filtered.png")),
                dpi=300, bbox_inches="tight"
            )
            plt.close(fig_filt)
        
if __name__ == "__main__":
    # Clean up old output files
    for f in Path("multiple_task_heatmaps").glob("*.png"):
        f.unlink()
        
    seed = 749
    L2 = "L21e4"
    main(seed, L2)