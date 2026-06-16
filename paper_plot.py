"""
Paper figure generation for the MultiTaskMPN project.

Each public function produces one publication-ready figure and saves it
to the `paper_plot/` directory. Run the script directly to generate all
figures, or import individual functions as needed.

Figures are grouped into three modes by the experiment they depend on:
    one_task         single-task training analyses
    multiple_tasks   full multi-task network (clustering, lesion, state space)
    pretraining      pretraining → post-training transfer analyses

Usage:
    python paper_plot.py                       # generate every mode
    python paper_plot.py all                   # same as above
    python paper_plot.py one_task              # only the one-task figures
    python paper_plot.py multiple_tasks        # only the multi-task figures
    python paper_plot.py pretraining           # only the pretraining figures
    python paper_plot.py --only input          # generate a single figure
"""
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import fcluster

# ─── Global style ────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ─── Paths ───────────────────────────────────────────────────────────────────
ANAME = "everything_seed749_L21e4+hidden300+batch128+angle"
DATA_DIR = Path("multiple_tasks") / ANAME
OUT_DIR = Path("paper_plot")

# Categorical color cycle (matches multiple_task_analysis.py)
c_vals = [
    "#e53e3e", "#3182ce", "#38a169", "#d69e2e", "#d53f8c",
    "#4c51bf", "#dd6b20", "#0ea5e9", "#22c55e", "#a855f7",
    "#f43f5e", "#0f766e", "#b83280", "#ca8a04", "#2b6cb0",
] * 10


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _breaks(lbls):
    """Cluster boundary positions from an ordered label array."""
    idx = np.nonzero(np.diff(lbls))[0] + 1
    return idx.tolist()


# Task name → Driscoll et al. 2024 display name
_TASK_DISPLAY = {
    "fdgo": "DelayPro",
    "fdanti": "DelayAnti",
    "delaygo": "MemoryPro",
    "delayanti": "MemoryAnti",
    "reactgo": "ReactGo",
    "reactanti": "ReactAnti",
    "delaydm1": "IntegrationModality1",
    "delaydm2": "IntegrationModality2",
    "contextdelaydm1": "ContextIntModality1",
    "contextdelaydm2": "ContextIntModality2",
    "multidelaydm": "IntegrationMultimodal",
    "dmsgo": "ReactMatch2Sample",
    "dmsnogo": "ReactNonMatch2Sample",
    "dmcgo": "ReactCategoryPro",
    "dmcnogo": "ReactCategoryAnti",
}

# Task → computation-category motif and color (matches state_space_shift.py)
_RULE_MOTIF = {
    "fdgo":            ("Pro Delayed",    "#3182ce"),  # blue
    "fdanti":          ("Anti Delayed",   "#e53e3e"),  # red
    "delaygo":         ("Pro Delayed",    "#3182ce"),
    "delayanti":       ("Anti Delayed",   "#e53e3e"),
    "reactgo":         ("Pro Reaction",   "#38a169"),  # green
    "reactanti":       ("Anti Reaction",  "#dd6b20"),  # orange
    "contextdelaydm1": ("Pro Integration", "#4682b4"),  # steelblue
    "contextdelaydm2": ("Pro Integration", "#4682b4"),
    "delaydm1":        ("Pro Integration", "#4682b4"),
    "delaydm2":        ("Pro Integration", "#4682b4"),
    "multidelaydm":    ("Pro Integration", "#4682b4"),
    "dmsgo":           ("Categorization", "#38a169"),
    "dmsnogo":         ("Categorization", "#dd6b20"),
    "dmcgo":           ("Categorization", "#ff1493"),  # deeppink
    "dmcnogo":         ("Categorization", "#ff1493"),
}


# Phase suffix → display name and background color
_PHASE_DISPLAY = {
    "stim1": "Stimulus 1",
    "stim2": "Stimulus 2",
    "delay1": "Memory 1",
    "delay2": "Memory 2",
    "go1": "Response",
}
_PHASE_COLORS = {
    "stim1": "#c3b1e1",   # purple
    "stim2": "#bfdbfe",   # light blue
    "delay1": "#bbf7d0",  # light green
    "delay2": "#fed7aa",  # light orange
    "go1": "#d1d5db",     # light gray
}


def _relabel_tb_name(name):
    """Convert '{rule}-{phase}' to '{DisplayRule}-{DisplayPhase}'."""
    for phase, disp in _PHASE_DISPLAY.items():
        if name.endswith(f"-{phase}"):
            rule = name[: -(len(phase) + 1)]
            rule_disp = _TASK_DISPLAY.get(rule, rule)
            return f"{rule_disp}-{disp}"
    return name


def _phase_of(name):
    """Return the phase suffix of a '{rule}-{phase}' label, or None."""
    for phase in _PHASE_DISPLAY:
        if name.endswith(f"-{phase}"):
            return phase
    return None


def _color_phase_ticklabels(ax, ordered_names, axis="y"):
    """Set a background highlight on each tick label based on its phase."""
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lab, name in zip(labels, ordered_names):
        phase = _phase_of(name)
        if phase is not None:
            lab.set_bbox(dict(facecolor=_PHASE_COLORS[phase], edgecolor="none",
                              boxstyle="round,pad=0.15", alpha=0.8))


def _color_motif_ticklabels(ax, task_names, axis="y"):
    """Set a background highlight on each tick label based on its task's
    computation-category motif (see _RULE_MOTIF). `task_names` is the list of
    raw rule names in tick order."""
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lab, task in zip(labels, task_names):
        color = _RULE_MOTIF.get(task, (None, None))[1]
        if color is not None:
            lab.set_bbox(dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.15", alpha=0.5))


def _load_cluster_info():
    """Load the cluster_info pickle for the target model."""
    pkl_path = DATA_DIR / f"cluster_info_{ANAME}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Cluster info not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ─── Figure: Clustered variance matrix ───────────────────────────────────────

def _recut_labels(linkage, k, original_labels):
    """
    Re-cut a dendrogram at a different k.

    The linkage matrix was built on the "active" subset (excluding any
    unresponsive neurons marked with label = original_k + 1). This
    function cuts the linkage at the new k, then maps back to the full
    label array preserving the unresponsive label if present.
    """
    original_k = linkage.shape[0]  # n_obs - 1 gives linkage rows
    n_obs = linkage.shape[0] + 1
    original_labels = np.asarray(original_labels)
    unique_orig = np.unique(original_labels)

    # Detect unresponsive cluster (label > original_k stored in result)
    max_label = unique_orig.max()
    # If there's an unresponsive cluster, its label = col_tol_k + 1
    # which equals n_obs + 1 (since linkage has n_obs - 1 rows → n_obs active neurons)
    has_unresponsive = (max_label > n_obs)
    unres_mask = original_labels == max_label if has_unresponsive else np.zeros(len(original_labels), dtype=bool)

    new_active_labels = fcluster(linkage, t=k, criterion="maxclust")

    full_labels = np.zeros(len(original_labels), dtype=int)
    full_labels[~unres_mask] = new_active_labels
    if has_unresponsive:
        full_labels[unres_mask] = k + 1

    return full_labels


def _compute_order_from_labels(linkage, labels):
    """
    Compute a display order that groups neurons by cluster label,
    with within-cluster ordering derived from the dendrogram leaf order.
    """
    from scipy.cluster.hierarchy import leaves_list
    leaf_order = leaves_list(linkage)

    labels = np.asarray(labels)
    n = len(labels)

    # Map from linkage leaf order (active neurons only) to full array
    unique_labels = np.unique(labels)
    active_mask = labels <= labels.max()  # all are active in this context

    # Build order: group by cluster, within each cluster use dendrogram order
    ordered = []
    for lab in sorted(unique_labels):
        members = set(np.where(labels == lab)[0])
        # Keep dendrogram order among members
        for idx in leaf_order:
            if idx in members:
                ordered.append(idx)
        # Any members not in leaf_order (e.g. unresponsive) appended at end
        remaining = members - set(ordered)
        ordered.extend(sorted(remaining))

    return np.array(ordered, dtype=int)


def _add_col_cluster_strip(ax, cl_ordered, cbreaks):
    """Add a thin colored strip below the heatmap, one color per column cluster,
    to visually group the x-axis columns by their cluster assignment."""
    n_cols = len(cl_ordered)
    # Cluster boundaries as [start, end) spans
    bounds = [0] + list(cbreaks) + [n_cols]
    n_clusters = len(bounds) - 1

    strip = ax.inset_axes([0, -0.06, 1, 0.04], transform=ax.transAxes)
    cmap_clusters = plt.get_cmap("tab20")
    for ci in range(n_clusters):
        start, end = bounds[ci], bounds[ci + 1]
        strip.axvspan(start, end, color=cmap_clusters(ci % 20), lw=0)
    strip.set_xlim(0, n_cols)
    strip.set_ylim(0, 1)
    strip.set_xticks([])
    strip.set_yticks([])
    for s in strip.spines.values():
        s.set_visible(False)
    return strip


def _add_row_cluster_strip(ax, rl_ordered, rbreaks):
    """Add a thin colored strip to the right of the heatmap, one color per row
    cluster, to visually group the y-axis rows by their cluster assignment."""
    n_rows = len(rl_ordered)
    bounds = [0] + list(rbreaks) + [n_rows]
    n_clusters = len(bounds) - 1

    strip = ax.inset_axes([1.01, 0, 0.025, 1], transform=ax.transAxes)
    cmap_clusters = plt.get_cmap("tab20")
    for ci in range(n_clusters):
        start, end = bounds[ci], bounds[ci + 1]
        strip.axhspan(start, end, color=cmap_clusters(ci % 20), lw=0)
    # Heatmap rows increase downward; match that orientation
    strip.set_ylim(n_rows, 0)
    strip.set_xlim(0, 1)
    strip.set_xticks([])
    strip.set_yticks([])
    for s in strip.spines.values():
        s.set_visible(False)
    return strip


def _plot_clustered_variance(
    cell_vars, result, tb_break_name,
    title="", cmap="magma", vmin=0, vmax=1,
    figsize=(8, 7),
    row_k_override=None,
    col_k_override=None,
):
    """
    Create a single-panel figure of the clustered task-variance matrix
    with cluster boundaries.

    Parameters
    ----------
    row_k_override : int, optional
        Override the number of row (session) clusters by re-cutting the
        stored dendrogram at this k.
    col_k_override : int, optional
        Override the number of col (neuron) clusters by re-cutting the
        stored dendrogram at this k.

    Returns (fig, ax).
    """
    # Determine row labels and order
    if row_k_override is not None:
        rl_full = _recut_labels(result["row_linkage"], row_k_override, result["row_tol_labels"])
        row_order = _compute_order_from_labels(result["row_linkage"], rl_full)
        row_k = row_k_override
    else:
        rl_full = np.asarray(result["row_tol_labels"])
        row_order = result["row_order"]
        row_k = result["row_tol_k"]

    # Determine col labels and order
    if col_k_override is not None:
        cl_full = _recut_labels(result["col_linkage"], col_k_override, result["col_tol_labels"])
        col_order = _compute_order_from_labels(result["col_linkage"], cl_full)
        col_k = col_k_override
    else:
        cl_full = np.asarray(result["col_tol_labels"])
        col_order = result["col_order"]
        col_k = result["col_tol_k"]

    ordered = cell_vars[np.ix_(row_order, col_order)]

    rl = rl_full[row_order]
    cl = cl_full[col_order]
    rbreaks = _breaks(rl)
    cbreaks = _breaks(cl)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    hm = sns.heatmap(
        ordered, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar=True, cbar_kws={"shrink": 0.4, "label": "Normalized variance"},
    )
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.0f}", f"{vmax:.0f}"])
    cbar.ax.tick_params(labelsize=12)

    for rb in rbreaks:
        ax.axhline(rb, color="0.6", lw=0.5, zorder=3, alpha=0.6)
    for cb in cbreaks:
        ax.axvline(cb, color="0.6", lw=0.5, zorder=3, alpha=0.6)

    ordered_names = tb_break_name[row_order]
    display_names = [_relabel_tb_name(nm) for nm in ordered_names]
    ax.set_yticks(np.arange(len(ordered_names)) + 0.5)
    ax.set_yticklabels(display_names, rotation=0, ha="right", va="center", fontsize=6)
    _color_phase_ticklabels(ax, ordered_names, axis="y")

    ax.set_xticks([])

    # Column-cluster grouping strip beneath the x-axis
    _add_col_cluster_strip(ax, cl, cbreaks)
    # Row-cluster grouping strip to the right of the y-axis
    _add_row_cluster_strip(ax, rl, rbreaks)

    fig.tight_layout()
    return fig, ax


def plot_clustered_input():
    """
    Figure: Clustered normalized task-variance matrix for the INPUT layer.
    """
    _ensure_out_dir()
    cluster_info = _load_cluster_info()
    data = cluster_info["input_normalized"]

    fig, _ = _plot_clustered_variance(
        cell_vars=data["cell_vars_rules_sorted_norm"],
        result=data["result"],
        tb_break_name=data["tb_break_name"],
        title="Input Layer — Normalized Task Variance",
    )

    out_path = OUT_DIR / "clustered_input_normalized.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_clustered_hidden(col_k_override=20):
    """
    Figure: Clustered normalized task-variance matrix for the HIDDEN layer.
    """
    _ensure_out_dir()
    cluster_info = _load_cluster_info()
    data = cluster_info["hidden_normalized"]

    fig, _ = _plot_clustered_variance(
        cell_vars=data["cell_vars_rules_sorted_norm"],
        result=data["result"],
        tb_break_name=data["tb_break_name"],
        title="Hidden Layer — Normalized Task Variance",
        col_k_override=col_k_override,
    )

    out_path = OUT_DIR / "clustered_hidden_normalized.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Figure: Clustered modulation variance matrix ────────────────────────────

def _load_cluster_info_mod():
    """Load the modulation cluster_info pickle for the target model."""
    pkl_path = DATA_DIR / f"cluster_info_mod_{ANAME}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Modulation cluster info not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def plot_clustered_modulation(G_index=1):
    """
    Figure: Clustered normalized task-variance matrix for MODULATION synapses.

    Uses the G=300 KMeans pre-grouping result (index 1 in result_all_lst).
    The figure is 2x wider than input/hidden figures to accommodate the
    90,000 synapse columns.
    """
    _ensure_out_dir()
    mod_info = _load_cluster_info_mod()
    mod_data = mod_info["modulation_all_normalized"]

    cell_vars = mod_data["cell_vars_rules_sorted_norm"]
    tb_break_name = mod_data["tb_break_name"]
    result = mod_data["result_all_lst"][G_index]

    row_order = result["row_order"]
    col_order = result["col_order"]
    ordered = cell_vars[np.ix_(row_order, col_order)]

    rl = np.asarray(result["row_tol_labels"])[row_order]
    cl = np.asarray(result["col_tol_labels"])[col_order]
    rbreaks = _breaks(rl)
    cbreaks = _breaks(cl)

    row_k = result["row_tol_k"]
    col_k = result["col_tol_k"]

    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    hm = sns.heatmap(
        ordered, ax=ax, cmap="magma", vmin=0, vmax=1,
        cbar=True, cbar_kws={"shrink": 0.4, "label": "Normalized variance"},
    )
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])
    cbar.ax.tick_params(labelsize=12)

    for rb in rbreaks:
        ax.axhline(rb, color="0.6", lw=0.5, zorder=3, alpha=0.6)
    for cb in cbreaks:
        ax.axvline(cb, color="0.6", lw=0.5, zorder=3, alpha=0.6)

    ordered_names = tb_break_name[row_order]
    display_names = [_relabel_tb_name(nm) for nm in ordered_names]
    ax.set_yticks(np.arange(len(ordered_names)) + 0.5)
    ax.set_yticklabels(display_names, rotation=0, ha="right", va="center", fontsize=6)
    _color_phase_ticklabels(ax, ordered_names, axis="y")

    ax.set_xticks([])

    # Column-cluster grouping strip beneath the x-axis
    _add_col_cluster_strip(ax, cl, cbreaks)
    # Row-cluster grouping strip to the right of the y-axis
    _add_row_cluster_strip(ax, rl, rbreaks)

    fig.tight_layout()

    out_path = OUT_DIR / "clustered_modulation_normalized.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Figure: L2 vs Accuracy ──────────────────────────────────────────────────

PERF_RESULT_PATH = Path("multiple_tasks_perf") / "performance_results.json"


def plot_l2_vs_accuracy():
    """Figure: Test accuracy (%) vs L2 regularization strength."""
    import json as _json

    _ensure_out_dir()
    if not PERF_RESULT_PATH.exists():
        print(f"  Skipped: {PERF_RESULT_PATH} not found. Run multiple_task_performance.py first.")
        return

    with open(PERF_RESULT_PATH) as f:
        result_dict = _json.load(f)

    l2_vals = np.array([e["l2_info"] for e in result_dict.values()])
    acc_vals = np.array([e["acc"] for e in result_dict.values()]) * 100

    fig, ax = plt.subplots(1, 1, figsize=(2.3, 3))
    ax.scatter(l2_vals, acc_vals, color="#3182ce", edgecolors="k",
               linewidths=0.5, s=40, alpha=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("L2 regularization strength")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)

    fig.tight_layout()
    out_path = OUT_DIR / "l2_vs_accuracy.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Figure: State space PCA ─────────────────────────────────────────────────

STATE_SPACE_DIR = Path("state_space")


def _plot_state_space_pca(X_2d, ctx_rule_labels, all_rules, rule_motif_mapping,
                          title="", figsize=(2.5, 2.5), show_legend=True):
    """
    Scatter of context-endpoint PCA colored by computation category.
    Returns (fig, ax).
    """
    category_order = [
        "Pro Delayed",
        "Anti Delayed",
        "Pro Reaction",
        "Anti Reaction",
        "Pro Integration",
        "Categorization",
    ]
    category_to_color = {cat: col for _, (cat, col) in rule_motif_mapping.items()}

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for cat in category_order:
        rule_idxs_in_cat = [
            idx for idx, rule in enumerate(all_rules)
            if rule_motif_mapping[rule][0] == cat
        ]
        sel = np.isin(ctx_rule_labels, rule_idxs_in_cat)
        ax.scatter(
            X_2d[sel, 0], X_2d[sel, 1],
            label=cat, color=category_to_color[cat],
            alpha=0.5, s=18, edgecolors="none",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=10, pad=6)
    if show_legend:
        ax.legend(frameon=True, loc="best", fontsize=5, markerscale=1.0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax


def _load_state_space_pca():
    """Load the PCA pickle for the target model."""
    pattern = f"state_space_pca_{ANAME}_noise*.pkl"
    matches = list(STATE_SPACE_DIR.glob(pattern))
    if not matches:
        return None
    return pickle.load(open(matches[0], "rb"))


def plot_state_space_combined():
    """Figure: Context-end PCA for hidden (top) and eff_mod (bottom) stacked vertically."""
    _ensure_out_dir()
    data = _load_state_space_pca()
    if data is None:
        print("  Skipped: state_space PCA pickle not found. Run state_space_shift.py first.")
        return

    all_rules = data["all_rules"]
    rule_motif_mapping = data["rule_motif_mapping"]

    category_order = [
        "Pro Delayed", "Anti Delayed", "Pro Reaction",
        "Anti Reaction", "Pro Integration", "Categorization",
    ]
    category_to_color = {cat: col for _, (cat, col) in rule_motif_mapping.items()}

    fig, axes = plt.subplots(2, 1, figsize=(2.5, 4.5), sharex=True)

    panels = [
        ("hidden", "Hidden state", False),
        ("eff_mod", "Eff. modulation", True),
    ]

    for ax, (key, ylabel_prefix, show_legend) in zip(axes, panels):
        pca = data["pca_results"][key]
        X_2d = pca["X_2d"]
        ctx_rule_labels = pca["ctx_rule_labels"]

        for cat in category_order:
            rule_idxs_in_cat = [
                idx for idx, rule in enumerate(all_rules)
                if rule_motif_mapping[rule][0] == cat
            ]
            sel = np.isin(ctx_rule_labels, rule_idxs_in_cat)
            ax.scatter(
                X_2d[sel, 0], X_2d[sel, 1],
                label=cat, color=category_to_color[cat],
                alpha=0.5, s=14, edgecolors="none",
            )

        ax.set_ylabel(f"{ylabel_prefix}\nPC2", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        if show_legend:
            ax.legend(frameon=True, loc="best", fontsize=5, markerscale=1.0)

    axes[1].set_xlabel("PC1", fontsize=8)
    axes[0].tick_params(labelbottom=False)

    fig.tight_layout()
    out_path = OUT_DIR / "state_space_combined.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


RVAL_RESULT_PATH = STATE_SPACE_DIR / "initial_condition_distance_vs_angle_results.pkl"


def plot_state_space_r_values():
    """Figure: Mean R-values (initial-condition distance vs trajectory angle) for hidden & eff_mod."""
    _ensure_out_dir()
    if not RVAL_RESULT_PATH.exists():
        print(f"  Skipped: {RVAL_RESULT_PATH} not found. Run state_space_shift.py first.")
        return

    with open(RVAL_RESULT_PATH, "rb") as f:
        result_dict = pickle.load(f)

    data_types = ["hidden", "mod", "eff_mod"]
    labels = ["Hidden", "Mod.", "Eff. Mod."]
    colors = ["#3182ce", "#dd6b20", "#38a169"]

    r_values = {dt: [] for dt in data_types}
    for results in result_dict.values():
        for dt in data_types:
            if dt in results["rval_dict"]:
                r_values[dt].append(results["rval_dict"][dt][0])

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 3))

    positions = np.arange(len(data_types))
    r_means = [np.mean(r_values[dt]) for dt in data_types]
    r_stds = [np.std(r_values[dt]) for dt in data_types]

    ax.bar(positions, r_means, yerr=r_stds, capsize=4,
           color=colors, edgecolor="k", linewidth=0.6, width=0.6)

    for dt_idx, dt in enumerate(data_types):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(r_values[dt]))
        ax.scatter(positions[dt_idx] + jitter, r_values[dt],
                   color="k", s=15, alpha=0.5, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("R-value")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)

    fig.tight_layout()
    out_path = OUT_DIR / "state_space_r_values.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Figure: Over-membership ─────────────────────────────────────────────────

def _find_experiment_dirs():
    """Return all experiment subfolders under multiple_tasks/ matching the
    same feature/hidden/batch signature as ANAME (any seed)."""
    import re as _re
    # ANAME = everything_seed{seed}_{feature}+hidden{h}+batch{b}+angle
    m = _re.match(r"everything_seed\d+_(.+)$", ANAME)
    suffix = m.group(1) if m else ""
    base = Path("multiple_tasks")
    dirs = sorted(base.glob(f"everything_seed*_{suffix}"))
    return [d for d in dirs if d.is_dir()]


def _plot_overmembership_single(pkl_template, out_filename):
    """
    Plot a 2×1 over-membership figure (top: same-neuron, bottom: same neuron-cluster),
    aggregated across all available experiments (seeds) that have the matching
    prepost_belonging pickle. Bars show the mean over-membership; error bars show
    the standard error across experiments.

    pkl_template: a filename template containing "{aname}", e.g.
        "modulation_all_prepost_belonging_{aname}_unnormalized.pkl"
    """
    _ensure_out_dir()

    # Collect per-experiment over-membership for each row (G=100, optimal-k entry).
    per_row_over = {0: [], 1: []}      # row_idx -> list of (n_bars,) arrays
    bar_name_lst = None
    n_experiments = 0

    for exp_dir in _find_experiment_dirs():
        aname = exp_dir.name
        pkl_path = exp_dir / pkl_template.format(aname=aname)
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        entry = data["prepost_belonging_results"][0]  # G=100, optimal k
        if bar_name_lst is None:
            bar_name_lst = entry["bar_name_lst"]
        for row_idx in range(2):
            obs = np.array(entry["bar_all_lst"][row_idx], dtype=float)
            ctrl = np.array(entry["bar_all_ctrl_lst"][row_idx], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                over = np.where(ctrl > 0, (obs - ctrl) / ctrl, 0.0)
            per_row_over[row_idx].append(over)
        n_experiments += 1

    if n_experiments == 0:
        print(f"  Skipped: no experiments with pickle '{pkl_template}'.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(2.5, 3.2))

    for row_idx in range(2):
        ax = axes[row_idx]
        stacked = np.vstack(per_row_over[row_idx])      # (n_exp, n_bars)
        mean = stacked.mean(axis=0)
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(mean)

        bar_names = bar_name_lst[row_idx]
        short_names = [n.replace("Share-", "").replace("-Cluster", " Cl.") for n in bar_names]

        colors = ["#3182ce", "#e53e3e", "#38a169", "#718096"][:len(mean)]
        x = np.arange(len(mean))
        ax.bar(x, mean, yerr=sem, capsize=3, color=colors,
               edgecolor="k", linewidth=0.5, width=0.6, zorder=2)
        # Overlay individual experiment points
        for over in per_row_over[row_idx]:
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(over))
            ax.scatter(x + jitter, over, color="k", s=8, alpha=0.5, zorder=3)
        ax.axhline(0, color="k", lw=0.5, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(hspace=0.45)
    out_path = OUT_DIR / out_filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  (n={n_experiments} experiments)")


def plot_overmembership_unnorm():
    """Figure: Over-membership for unnormalized modulation (G=100), aggregated across seeds."""
    _plot_overmembership_single(
        "modulation_all_prepost_belonging_{aname}_unnormalized.pkl",
        "overmembership_unnormalized.png",
    )


def plot_overmembership_weighted():
    """Figure: Over-membership for weighted unnormalized modulation (G=100), aggregated across seeds."""
    _plot_overmembership_single(
        "modulation_all_weighted_prepost_belonging_{aname}_unnormalized.pkl",
        "overmembership_weighted.png",
    )


def plot_overmembership_var_weighted():
    """Figure: Over-membership for var-weighted unnormalized modulation (G=100), aggregated across seeds."""
    _plot_overmembership_single(
        "modulation_all_var_weighted_prepost_belonging_{aname}_unnormalized.pkl",
        "overmembership_var_weighted.png",
    )


# ─── Figure: Lesion heatmap ──────────────────────────────────────────────────

LESION_DIR = Path("multiple_tasks_perf") / ANAME


def _load_lesion_results():
    """Load the lesion/prune results pickle."""
    pkl_path = LESION_DIR / f"lesion_prune_results_{ANAME}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def plot_lesion_heatmap():
    """
    Figure: Normalized lesion effect heatmap (unnormalized clusters).

    Two panels stacked vertically:
      Top — input (pre) cluster lesion effect (tasks × input clusters)
      Bottom — hidden (post) cluster lesion effect (tasks × hidden clusters)

    Normalized effect = random_acc - cluster_acc (positive = cluster matters).
    """
    _ensure_out_dir()
    data = _load_lesion_results()
    if data is None:
        print("  Skipped: lesion results not found. Run leison.py first.")
        return

    lu = data["leison_unnorm"]
    ru = data["random_leison_unnorm"]

    all_tasks = lu["all_tasks"]
    all_tasks_display = [_TASK_DISPLAY.get(t, t) for t in all_tasks]
    comb_names = lu["all_comb_names_leison"]
    ihtask_accs = np.array(lu["ihtask_accs"])
    random_accs = np.array(ru["ihrandomtask_accs"])

    pre_idx = [i for i, n in enumerate(comb_names) if n.startswith("pre_c")]
    post_idx = [i for i, n in enumerate(comb_names) if n.startswith("post_c")]

    effect_pre = (random_accs[:, pre_idx] - ihtask_accs[:, pre_idx]) * 100
    effect_post = (random_accs[:, post_idx] - ihtask_accs[:, post_idx]) * 100

    pre_labels = [f"C{i+1}" for i in range(len(pre_idx))]
    post_labels = [f"C{i+1}" for i in range(len(post_idx))]

    # Modulation freeze_M lesion (weighted unnormalized)
    mod_entry = data["mod_leison"]["modulation_all_weighted_unnormalized__freeze_M"]
    mod_accs = np.array(mod_entry["modtask_accs"])
    mod_random = np.array(mod_entry["modrandomtask_accs"])
    mod_comb = mod_entry["all_comb_names_mod"]
    mod_idx = [i for i, n in enumerate(mod_comb) if n.startswith("mod_c")]
    effect_mod = (mod_random[:, mod_idx] - mod_accs[:, mod_idx]) * 100
    mod_labels = [f"C{i+1}" for i in range(len(mod_idx))]

    vmax = max(np.abs(effect_post).max(), np.abs(effect_mod).max())

    fig, axes = plt.subplots(
        2, 1, figsize=(6, 5.5),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1},
    )

    panels = [
        (axes[0], effect_post, post_labels),
        (axes[1], effect_mod, mod_labels),
    ]

    for idx, (ax, effect, cluster_labels) in enumerate(panels):
        sns.heatmap(
            effect, ax=ax, cmap="RdBu_r", center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=cluster_labels,
            yticklabels=all_tasks_display,
            cbar=False,
        )
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=6, rotation=0)
        ax.tick_params(axis="x", labelsize=6)
        if idx < 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Cluster", fontsize=8)

        # Color each task tick label's background by its computation-category motif
        _color_motif_ticklabels(ax, all_tasks, axis="y")

    # Shared colorbar
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.04)
    cbar.set_label("Normalized effect (%)", fontsize=8)
    # Ticks every 30, symmetric around 0, within [-vmax, vmax]
    _tick_max = int(np.floor(vmax / 30.0)) * 30
    _ticks = np.arange(-_tick_max, _tick_max + 1, 30)
    cbar.set_ticks(_ticks)
    cbar.set_ticklabels([f"{t:.0f}" for t in _ticks])
    cbar.ax.tick_params(labelsize=10)
    out_path = OUT_DIR / "lesion_heatmap_unnorm.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Figure: OM vs lesion ────────────────────────────────────────────────────

def _load_cluster_info_mod():
    """Load the modulation cluster_info pickle for the target model."""
    pkl_path = DATA_DIR / f"cluster_info_mod_{ANAME}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def plot_om_vs_lesion():
    """
    Figure: Over-membership predicts modulation lesion effect.

    Panel 1 (left): OM vs |lesion effect diff| for freeze_M mode.
    Panel 2 (right): Per-cluster OM-predicted effect vs actual mod lesion effect.

    Uses weighted-unnormalized modulation, fixed_k20 global assignment,
    and combined_leison_unnorm.
    """
    _ensure_out_dir()
    results = _load_lesion_results()
    if results is None:
        print("  Skipped: lesion results not found.")
        return

    cluster_info_mod = _load_cluster_info_mod()
    if cluster_info_mod is None:
        print("  Skipped: cluster_info_mod not found.")
        return

    base_key = "modulation_all_var_weighted_unnormalized"
    variant = "unnorm"

    if base_key not in cluster_info_mod:
        print(f"  Skipped: {base_key} not in cluster_info_mod.")
        return

    mod_keys = cluster_info_mod[base_key]
    fk_ga_keys = [k for k in mod_keys if k.startswith("global_assignment_fixed_k")]
    ga = mod_keys[fk_ga_keys[0]] if fk_ga_keys else mod_keys.get("global_assignment")
    if ga is None:
        print("  Skipped: no global_assignment found.")
        return

    om_stack = ga["om_stack"]
    all_choice_order = ga["all_choice_order"]
    n_in = ga["n_in"]
    n_hid = ga["n_hid"]
    om_id_to_idx = {cid: idx for idx, cid in enumerate(all_choice_order)}

    # Skip unresponsive clusters (last index in unnorm)
    skip_input = {n_in - 1}
    skip_hidden = {n_hid - 1}

    ckey = f"combined_leison_{variant}"
    if ckey not in results:
        print(f"  Skipped: {ckey} not in results.")
        return
    cdata = results[ckey]
    combined_effect = np.asarray(cdata["combined_random_accs"], dtype=float) - np.asarray(cdata["combined_accs"], dtype=float)
    comb_mean = combined_effect.mean(axis=0)  # (pre_n, post_n)

    # --- Panel 1: OM vs |lesion effect diff| for freeze_M ---
    mod_result_key_fm = f"{base_key}__freeze_M"
    mod_data_fm = results["mod_leison"][mod_result_key_fm]
    modtask_accs_fm = np.asarray(mod_data_fm["modtask_accs"], dtype=float)
    modrandom_fm = np.asarray(mod_data_fm["modrandomtask_accs"], dtype=float)
    all_comb_names_fm = mod_data_fm["all_comb_names_mod"]

    mod_effects_fm = {}
    for key_idx, key in enumerate(all_comb_names_fm):
        if key == "mod_noleison":
            continue
        cid = int(key.replace("mod_c", ""))
        mod_effects_fm[cid] = modrandom_fm[:, key_idx] - modtask_accs_fm[:, key_idx]

    om_vals, lesion_diffs = [], []
    for cid in sorted(mod_effects_fm.keys()):
        if cid not in om_id_to_idx:
            continue
        om_idx = om_id_to_idx[cid]
        mod_eff = mod_effects_fm[cid]
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

    om_vals = np.array(om_vals)
    lesion_diffs = np.array(lesion_diffs)

    # --- Panel 2: per-cluster prediction (zero_W) ---
    mod_result_key_zw = f"{base_key}__zero_W"
    mod_data_zw = results["mod_leison"][mod_result_key_zw]
    modtask_accs_zw = np.asarray(mod_data_zw["modtask_accs"], dtype=float)
    modrandom_zw = np.asarray(mod_data_zw["modrandomtask_accs"], dtype=float)

    pred_x, pred_y = [], []
    for key_idx, key in enumerate(mod_data_zw["all_comb_names_mod"]):
        if key == "mod_noleison":
            continue
        cid = int(key.replace("mod_c", ""))
        if cid not in om_id_to_idx:
            continue
        om_idx = om_id_to_idx[cid]
        om_profile = om_stack[om_idx]
        if om_profile.sum() > 0:
            predicted = (om_profile * comb_mean).sum() / om_profile.sum()
        else:
            predicted = 0.0
        actual = (modrandom_zw[:, key_idx] - modtask_accs_zw[:, key_idx]).mean()
        pred_x.append(predicted * 100)
        pred_y.append(actual * 100)

    pred_x = np.array(pred_x)
    pred_y = np.array(pred_y)

    # --- Figure 1: OM vs |lesion diff| ---
    from scipy.stats import linregress as _linregress
    fig1, ax1 = plt.subplots(1, 1, figsize=(3, 2.8))
    slope, intercept, r, p, _ = _linregress(om_vals, lesion_diffs)
    ax1.scatter(om_vals, lesion_diffs, color="#3182ce", edgecolors="k",
                linewidths=0.5, s=40, alpha=0.8, zorder=3)
    x_line = np.linspace(om_vals.min(), om_vals.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.2, zorder=4)
    p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
    ax1.legend([f"r = {r:.2f}, {p_str}"], loc="upper right", fontsize=7, frameon=True)
    ax1.set_xlabel("Over-membership", fontsize=8)
    ax1.set_ylabel("|Lesion effect diff|", fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()
    out_path1 = OUT_DIR / "om_vs_lesion_scatter.png"
    fig1.savefig(out_path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {out_path1}")

    # --- Figure 2: predicted vs actual ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(3, 2.8))
    slope_p, intercept_p, r_p, p_p, _ = _linregress(pred_x, pred_y)
    ax2.scatter(pred_x, pred_y, color="#3182ce", edgecolors="k",
                linewidths=0.5, s=40, alpha=0.8, zorder=3)
    lim = [min(pred_x.min(), pred_y.min()), max(pred_x.max(), pred_y.max())]
    ax2.plot(lim, lim, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    x_fit = np.linspace(pred_x.min(), pred_x.max(), 100)
    ax2.plot(x_fit, slope_p * x_fit + intercept_p, color="tomato", linewidth=1.2, zorder=4)
    p_str_p = f"p = {p_p:.2e}" if p_p < 0.001 else f"p = {p_p:.3f}"
    ax2.legend([f"r = {r_p:.2f}, {p_str_p}"], loc="upper left", fontsize=7, frameon=True)
    ax2.set_xlabel("OM-predicted effect (%)", fontsize=8)
    ax2.set_ylabel("Actual mod lesion effect (%)", fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    out_path2 = OUT_DIR / "om_vs_lesion_prediction.png"
    fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path2}")


# ─── Figure: Fixed-point PCA trajectories ────────────────────────────────────

def plot_fixed_points(addtask="delaydm1", plot_name="e_modulation"):
    """
    Figure: Fixed-point trajectories in delay-period PCA space.

    Plots how memory-state fixed points evolve as the stimulus is interpolated
    between two conditions. One panel per PC pair (PC1-2, PC1-3, PC2-3).

    addtask: "delaydm1" or "dmcgo"
    plot_name: "hidden", "e_modulation", or "m_modulation"
    """
    _ensure_out_dir()

    pkl_path = DATA_DIR / f"{addtask}_fixed_points_{ANAME}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path.name} not found. Run multiple_task_analysis.py shared_run first.")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if plot_name not in data:
        print(f"  Skipped: '{plot_name}' not in {pkl_path.name}.")
        return

    entry = data[plot_name]
    fixed_points_all_arr = np.asarray(entry["fixed_points_all_arr"])  # (n_alpha, n_stim, n_pc)
    stim_labels = entry["stim_labels"]
    trial_num = entry["trial_num"]

    n_alpha, n_stim, n_pc = fixed_points_all_arr.shape
    pcs = [[0, 1], [0, 2], [1, 2]]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for pc_idx, (pc_x, pc_y) in enumerate(pcs):
        ax = axes[pc_idx]
        for stim in range(n_stim):
            traj_fp = fixed_points_all_arr[:, stim, :]
            color = c_vals[stim_labels[stim]]
            ax.plot(traj_fp[:, pc_x], traj_fp[:, pc_y], "-o",
                    color=color, linewidth=1.2, markersize=3, alpha=0.5,
                    label=f"stim {(stim // trial_num + 1)}" if stim % trial_num == 0 else None)
            ax.scatter(traj_fp[0, pc_x], traj_fp[0, pc_y], color=color,
                       marker="s", s=40, zorder=3)
            ax.scatter(traj_fp[-1, pc_x], traj_fp[-1, pc_y], color=color,
                       marker="^", s=40, zorder=3)

        ax.set_xlabel(f"Memory State PC{pc_x+1}", fontsize=9)
        ax.set_ylabel(f"Memory State PC{pc_y+1}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if pc_idx == 0:
            ax.legend(frameon=True, fontsize=6)

    fig.tight_layout()
    out_path = OUT_DIR / f"fixed_points_{addtask}_{plot_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fixed_points_all():
    """Generate fixed-point figures for both addtasks and all representations."""
    for addtask in ["delaydm1", "dmcgo"]:
        for plot_name in ["hidden", "e_modulation", "m_modulation"]:
            plot_fixed_points(addtask=addtask, plot_name=plot_name)


# ─── Figure: Input weight correlation ────────────────────────────────────────

def plot_input_weight_correlation():
    """
    Figure: Pearson correlation between columns of W_initial_linear (input weight).

    Each column corresponds to an input feature: 6 stimulus channels + 15 task indicators.
    """
    import torch
    import json as _json

    _ensure_out_dir()

    ckpt_path = Path("multiple_tasks") / f"savednet_{ANAME}.pt"
    param_path = Path("multiple_tasks") / f"param_{ANAME}_param.json"

    if not ckpt_path.exists():
        print(f"  Skipped: {ckpt_path} not found.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_W = ckpt["state_dict"]["W_initial_linear.weight"].numpy()

    with open(param_path) as f:
        cfg = _json.load(f)
    rules = cfg["task_params"]["rules"]

    all_input = ["Fix On", "Fix Off", "Stim 1 Cos", "Stim 1 Sin", "Stim 2 Cos", "Stim 2 Sin"] + rules
    input_corr = np.corrcoef(input_W.T)
    mask = np.triu(np.ones_like(input_corr, dtype=bool), k=0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    hm = sns.heatmap(input_corr, ax=ax, cmap="coolwarm", center=0, mask=mask,
                     vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.5})
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(labelsize=9)
    ax.set_xticks(np.arange(len(all_input)) + 0.5)
    ax.set_xticklabels(all_input, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(all_input)) + 0.5)
    ax.set_yticklabels(all_input, rotation=0, fontsize=7)

    fig.tight_layout()
    out_path = OUT_DIR / "input_weight_correlation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Figure: Cluster tuning vs lesion effect ─────────────────────────────────

LESION_NORM_DIR = Path("multiple_tasks_norm") / ANAME


def plot_cluster_corr_vs_lesion():
    """
    Figure: Scatter of cluster tuning cosine similarity vs lesion effect L1 distance.

    Produces separate figures for each variant (normalized, unnormalized).
    Each figure has one subplot per cluster type (input, hidden).
    """
    from scipy.stats import linregress as _linregress

    _ensure_out_dir()
    if not LESION_NORM_DIR.exists():
        print("  Skipped: multiple_tasks_norm dir not found. Run leison_plot.py first.")
        return

    variants = [
        ("normalized_leison_effect", "norm"),
        ("normalized_leison_effect_unnorm", "unnorm"),
    ]

    for suffix, short_tag in variants:
        pkl_path = LESION_NORM_DIR / f"cluster_corr_vs_{suffix}_{ANAME}.pkl"
        if not pkl_path.exists():
            print(f"  Skipped: {pkl_path.name} not found.")
            continue

        with open(pkl_path, "rb") as f:
            scatter_data = pickle.load(f)

        for name, data in scatter_data.items():
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.8))
            x = np.array(data["tuning_cos_sim"])
            y = np.array(data["lesion_l1_dist"])

            ax.scatter(x, y, color="#3182ce", edgecolors="k",
                       linewidths=0.5, s=40, alpha=0.8, zorder=3)

            if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                slope, intercept, r, p, _ = _linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, color="tomato",
                        linewidth=1.2, zorder=4)
                p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
                ax.legend([f"r = {r:.2f}, {p_str}"], loc="upper right",
                          fontsize=7, frameon=True)

            ax.set_xlabel("Tuning cosine similarity", fontsize=8)
            ax.set_ylabel("Lesion effect L1 distance", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)

            fig.tight_layout()
            # e.g. "input_normalized_k20" -> "input_norm"
            clean_name = name.replace("_normalized", "_norm").replace("_unnormalized", "_unnorm").replace("_k20", "")
            out_path = OUT_DIR / f"cluster_corr_vs_lesion_{clean_name}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


# ─── Figure: Transfer speed ──────────────────────────────────────────────────

PRETRAINING_ANALYSIS_DIR = Path("pretraining_analysis")


def plot_transfer_speed():
    """
    Figure: Transfer speed — iterations to reach accuracy thresholds during
    post-training, comparing fdgo_delaygo vs fdanti_delaygo rulesets (L21e3).

    Loads from the combined transfer_speed.pkl if available; otherwise falls
    back to loading individual per-seed result pickles.
    """
    import re as _re

    _ensure_out_dir()
    if not PRETRAINING_ANALYSIS_DIR.exists():
        print("  Skipped: pretraining_analysis/ not found.")
        return

    # Try loading the combined pickle first (saved by pretraining_analysis.py)
    ts_pkl = list(PRETRAINING_ANALYSIS_DIR.glob("*_transfer_speed.pkl"))
    if ts_pkl:
        with open(ts_pkl[0], "rb") as f:
            ts_data = pickle.load(f)
        thresholds = ts_data["thresholds"]
        by_ruleset_mats = ts_data["by_ruleset"]
    else:
        # Fallback: load individual seed pickles
        addon_name = "+hidden200+L21e3+batch128+angle"
        pkls = sorted(PRETRAINING_ANALYSIS_DIR.glob(f"*_dmpn_seed*_{addon_name}_result.pkl"))
        if not pkls:
            print("  Skipped: no pretraining result pickles found.")
            return

        by_ruleset_raw = {}
        for p in pkls:
            m = _re.match(
                r'(.+)_dmpn_seed\d+_\+hidden200\+L21e3\+batch128\+angle_result\.pkl', p.name
            )
            if m:
                ruleset = m.group(1)
                with open(p, "rb") as f:
                    by_ruleset_raw.setdefault(ruleset, []).append(pickle.load(f))

        if not by_ruleset_raw:
            print("  Skipped: no valid results loaded.")
            return

        def _first_iter_to(iters, acc, threshold):
            iters = np.asarray(iters)
            acc = np.asarray(acc)
            hits = np.where(acc >= threshold)[0]
            return float(iters[hits[0]]) if hits.size else np.nan

        thresholds = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 0.99])
        by_ruleset_mats = {}
        for rs, seed_results in by_ruleset_raw.items():
            per_seed_mat = np.asarray([
                [_first_iter_to(sr["learning"]["acc_iter_post"],
                                sr["learning"]["acc_post"], th)
                 for th in thresholds]
                for sr in seed_results
            ], dtype=float)
            by_ruleset_mats[rs] = {"per_seed_iters": per_seed_mat, "n_seeds": len(seed_results)}

    ys = thresholds * 100
    ruleset_colors = {
        "fdgo_delaygo": "#3182ce",
        "fdanti_delaygo": "#e53e3e",
    }
    ruleset_labels = {
        "fdgo_delaygo": "Irrelevant motif",
        "fdanti_delaygo": "Relevant motif",
    }

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.2))

    for rs, rs_data in by_ruleset_mats.items():
        color = ruleset_colors.get(rs, "#718096")
        label = ruleset_labels.get(rs, rs)
        per_seed_mat = np.asarray(rs_data["per_seed_iters"], dtype=float)
        n_seeds = rs_data["n_seeds"]

        mean_vals = np.nanmean(per_seed_mat, axis=0)
        std_vals = np.nanstd(per_seed_mat, axis=0)
        ax.plot(mean_vals, ys, "s-", color=color, linewidth=2.0,
                markersize=5, label=label)
        ax.fill_betweenx(ys, mean_vals - std_vals, mean_vals + std_vals,
                         color=color, alpha=0.15)

    ax.set_xlabel("Iterations to reach threshold")
    ax.set_ylabel("Accuracy threshold (%)")
    ax.set_xscale("log")
    ax.set_yticks([50, 75, 100])
    ax.legend(fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "transfer_speed.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Figure: Rule vectors ────────────────────────────────────────────────────

def plot_rule_vectors():
    """
    Figure: Pairwise cosine similarity between rule-input vectors.

    Shows how the novel task's learned rule vector relates to the two
    pretrained rule vectors, for each ruleset (relevant vs irrelevant motif).
    """
    import re as _re

    _ensure_out_dir()
    if not PRETRAINING_ANALYSIS_DIR.exists():
        print("  Skipped: pretraining_analysis/ not found.")
        return

    # Try combined pkl first
    rv_pkls = list(PRETRAINING_ANALYSIS_DIR.glob("*_rule_vectors.pkl"))
    if rv_pkls:
        with open(rv_pkls[0], "rb") as f:
            rv_data = pickle.load(f)
        by_ruleset = rv_data["by_ruleset"]
    else:
        # Fallback: load from individual seed pickles
        addon_name = "+hidden200+L21e3+batch128+angle"
        pkls = sorted(PRETRAINING_ANALYSIS_DIR.glob(f"*_dmpn_seed*_{addon_name}_result.pkl"))
        if not pkls:
            print("  Skipped: no pretraining result pickles found.")
            return

        stage1_tasks_map = {
            "fdgo_delaygo": ["fdgo", "delaygo"],
            "fdanti_delaygo": ["fdanti", "delaygo"],
        }
        final_task = "delayanti"

        by_ruleset = {}
        for p in pkls:
            m = _re.match(
                r'(.+)_dmpn_seed\d+_\+hidden200\+L21e3\+batch128\+angle_result\.pkl', p.name
            )
            if m:
                rs = m.group(1)
                with open(p, "rb") as f:
                    data = pickle.load(f)
                if "rule_vectors" in data:
                    entry = by_ruleset.setdefault(rs, {
                        "cos_novel_pre0": [], "cos_novel_pre1": [],
                        "cos_pre0_pre1": [], "in_span_fraction": [],
                        "stage1_tasks": stage1_tasks_map.get(rs, [rs]),
                        "final_task": final_task,
                    })
                    rv = data["rule_vectors"]
                    entry["cos_novel_pre0"].append(rv["cos_novel_pre0"])
                    entry["cos_novel_pre1"].append(rv["cos_novel_pre1"])
                    entry["cos_pre0_pre1"].append(rv["cos_pre0_pre1"])
                    entry["in_span_fraction"].append(rv["in_span_fraction"])

    if not by_ruleset:
        print("  Skipped: no rule vector data found.")
        return

    ruleset_colors = {
        "fdgo_delaygo": "#3182ce",
        "fdanti_delaygo": "#e53e3e",
    }
    ruleset_labels = {
        "fdgo_delaygo": "Irrelevant motif",
        "fdanti_delaygo": "Relevant motif",
    }

    task_display_names = {
        "delayanti": "MemoryAnti",
        "fdanti": "DelayAnti",
        "fdgo": "DelayPro",
        "delaygo": "MemoryPro",
    }

    cos_keys = ["cos_novel_pre0", "cos_novel_pre1", "cos_pre0_pre1"]
    rs_list = sorted(by_ruleset.keys())

    # Drop the within-pretraining baseline (pre0 ↔ pre1) so each motif keeps
    # the two novel-vs-pretrained comparisons:
    #   relevant   (fdanti_delaygo): MemoryAnti↔DelayAnti + MemoryAnti↔MemoryPro
    #   irrelevant (fdgo_delaygo):   MemoryAnti↔DelayPro  + MemoryAnti↔MemoryPro
    # Dropped: DelayAnti↔MemoryPro (fdanti↔delaygo), DelayPro↔MemoryPro (fdgo↔delaygo).
    excluded_pairs = {
        frozenset(("fdanti", "delaygo")),
        frozenset(("fdgo", "delaygo")),
    }

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4))

    # Bars are packed within a ruleset group; `group_gap` separates groups.
    bar_step = 1.0
    bar_width = 0.8
    group_gap = 0.6

    all_x, all_labels = [], []
    next_x = 0.0  # running left edge for the next group
    for rs in rs_list:
        color = ruleset_colors.get(rs, "#718096")
        label = ruleset_labels.get(rs, rs)
        s1_tasks = by_ruleset[rs].get("stage1_tasks", [rs])
        final_task = by_ruleset[rs].get("final_task", "novel")

        ft = task_display_names.get(final_task, final_task)
        t0 = task_display_names.get(s1_tasks[0], s1_tasks[0])
        t1 = task_display_names.get(s1_tasks[1], s1_tasks[1])

        # (key, label, underlying raw task pair) for the three comparisons.
        bar_specs = [
            ("cos_novel_pre0", f"{ft}\n↔ {t0}", (final_task, s1_tasks[0])),
            ("cos_novel_pre1", f"{ft}\n↔ {t1}", (final_task, s1_tasks[1])),
            ("cos_pre0_pre1", f"{t0}\n↔ {t1}", (s1_tasks[0], s1_tasks[1])),
        ]
        # Drop the excluded task pairs; keep remaining bars packed (no gaps).
        bar_specs = [
            (k, lbl, pair) for (k, lbl, pair) in bar_specs
            if frozenset(pair) not in excluded_pairs
        ]
        if not bar_specs:
            continue

        keys = [k for (k, _, _) in bar_specs]
        cos_labels = [lbl for (_, lbl, _) in bar_specs]

        xs_group = next_x + bar_step * np.arange(len(keys))
        next_x = xs_group[-1] + bar_step + group_gap
        means = np.array([np.mean(by_ruleset[rs][k]) for k in keys])
        stds = np.array([np.std(by_ruleset[rs][k]) for k in keys])

        ax.bar(xs_group, means, bar_width, yerr=stds, capsize=2,
               color=color, alpha=0.8, edgecolor="k", linewidth=0.5,
               label=label)

        for k_idx, k in enumerate(keys):
            vals = np.array(by_ruleset[rs][k])
            ax.plot(np.full_like(vals, xs_group[k_idx]), vals,
                    "k.", markersize=3, alpha=0.6)

        all_x.extend(xs_group.tolist())
        all_labels.extend(cos_labels)

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=0, ha="center", fontsize=6)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Cosine similarity")
    ax.legend(fontsize=7, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "rule_vectors.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Figure: Aggregate CVE ───────────────────────────────────────────────────

def _load_aggregate_cve_by_ruleset(analysis_types, periods):
    """
    Load aggregate CVE curves keyed by ruleset.

    Prefers the combined `*_aggregate.pkl` files written by
    pretraining_analysis.py; falls back to reconstructing the aggregate from
    the per-seed `*_result.pkl` files. Returns a dict {ruleset: agg_dict}
    where agg_dict holds `{dtype}_{period}_self` / `_cross` lists of per-seed
    curves. Returns {} if no data is available.
    """
    import re as _re

    if not PRETRAINING_ANALYSIS_DIR.exists():
        return {}

    # Try combined aggregate pkls first
    agg_pkls = sorted(PRETRAINING_ANALYSIS_DIR.glob("*_dmpn_*_aggregate.pkl"))

    by_ruleset = {}
    if agg_pkls:
        for p in agg_pkls:
            with open(p, "rb") as f:
                data = pickle.load(f)
            rs = data["ruleset"]
            by_ruleset[rs] = data
        return by_ruleset

    # Fallback: reconstruct from individual seed pickles
    addon_name = "+hidden200+L21e3+batch128+angle"
    pkls = sorted(PRETRAINING_ANALYSIS_DIR.glob(f"*_dmpn_seed*_{addon_name}_result.pkl"))
    if not pkls:
        return {}

    raw_by_rs = {}
    for p in pkls:
        m = _re.match(
            r'(.+)_dmpn_seed\d+_\+hidden200\+L21e3\+batch128\+angle_result\.pkl', p.name
        )
        if m:
            rs = m.group(1)
            with open(p, "rb") as f:
                raw_by_rs.setdefault(rs, []).append(pickle.load(f))

    for rs, seed_results in raw_by_rs.items():
        agg = {"ruleset": rs}
        for dtype in analysis_types:
            for period in periods:
                all_self, all_cross = [], []
                for sr in seed_results:
                    if dtype not in sr:
                        continue
                    if period not in sr[dtype]:
                        continue
                    res = sr[dtype][period]
                    all_self.append(res["cev_Y_self"])
                    all_cross.append(res["cev_Y"])
                if all_self:
                    min_len = min(min(len(c) for c in all_self),
                                  min(len(c) for c in all_cross))
                    agg[f"{dtype}_{period}_self"] = [c[:min_len] for c in all_self]
                    agg[f"{dtype}_{period}_cross"] = [c[:min_len] for c in all_cross]
        by_ruleset[rs] = agg

    return by_ruleset


def _plot_aggregate_cve_panel(ax, by_ruleset, dtype, period, ruleset_colors,
                              ruleset_labels, x_lim, x_ticks, show_legend):
    """
    Draw one CVE panel: novel-in-own-PCs (self, black) plus novel-in-
    pretraining-PCs (cross, colored per ruleset), with per-seed thin lines
    and seed-mean thick lines. Shared by the full and stimulus-only figures.
    """
    key_self = f"{dtype}_{period}_self"
    key_cross = f"{dtype}_{period}_cross"

    # Plot self (black) — same across rulesets, just use the first available
    self_plotted = False
    for rs in ["fdanti_delaygo", "fdgo_delaygo"]:
        if rs not in by_ruleset:
            continue
        agg = by_ruleset[rs]
        if key_self not in agg:
            continue
        if not self_plotted:
            all_self = np.array(agg[key_self])
            min_len = all_self.shape[1]
            xs = np.arange(1, min_len + 1)
            for i in range(all_self.shape[0]):
                ax.plot(xs, all_self[i], color="black", linewidth=0.5, alpha=0.2)
            mean_self = np.mean(all_self, axis=0)
            ax.plot(xs, mean_self, color="black", linewidth=2.0,
                    label="Self" if show_legend else None)
            self_plotted = True

    # Plot cross (colored by ruleset)
    for rs in ["fdanti_delaygo", "fdgo_delaygo"]:
        if rs not in by_ruleset:
            continue
        agg = by_ruleset[rs]
        if key_cross not in agg:
            continue

        color = ruleset_colors.get(rs, "#718096")
        label = ruleset_labels.get(rs, rs)

        all_cross = np.array(agg[key_cross])
        min_len = all_cross.shape[1]
        xs = np.arange(1, min_len + 1)

        for i in range(all_cross.shape[0]):
            ax.plot(xs, all_cross[i], color=color, linewidth=0.5,
                    alpha=0.25, linestyle="--")

        mean_cross = np.mean(all_cross, axis=0)
        ax.plot(xs, mean_cross, color=color, linewidth=2.0, linestyle="--",
                label=label if show_legend else None)

    ax.set_xlim(0, x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(fontsize=7, frameon=True)


def plot_aggregate_cve():
    """
    Figure: Cumulative variance explained (CVE) of the novel task in its own
    PCs vs in the pretraining task PCs. Overlays fdgo_delaygo (irrelevant motif)
    and fdanti_delaygo (relevant motif) on the same axes.

    Produces a 3×2 grid: rows = hidden, modulation, modulation_weighted;
    columns = stimulus, response. Each panel saved as a separate file.
    """
    _ensure_out_dir()

    analysis_types = ["hidden", "modulation", "modulation_weighted"]
    periods = ["stimulus", "response"]

    by_ruleset = _load_aggregate_cve_by_ruleset(analysis_types, periods)
    if not by_ruleset:
        print("  Skipped: no aggregate data found.")
        return

    ruleset_colors = {
        "fdgo_delaygo": "#3182ce",
        "fdanti_delaygo": "#e53e3e",
    }
    ruleset_labels = {
        "fdgo_delaygo": "Irrelevant motif",
        "fdanti_delaygo": "Relevant motif",
    }

    # 2×2 combined figure: rows = [hidden, modulation_weighted], cols = [stimulus, response]
    panel_layout = [
        ("hidden", "stimulus"),
        ("hidden", "response"),
        ("modulation_weighted", "stimulus"),
        ("modulation_weighted", "response"),
    ]
    x_lim_map = {"hidden": 20, "modulation_weighted": 1000}
    x_tick_map = {"hidden": np.arange(0, 21, 5), "modulation_weighted": np.arange(0, 1001, 200)}

    dtype_titles = {"hidden": "Hidden", "modulation_weighted": "Effective Modulation"}
    period_titles = {"stimulus": "Stimulus Period", "response": "Response Period"}

    fig, axes = plt.subplots(2, 2, figsize=(6, 4.5))

    for idx, (dtype, period) in enumerate(panel_layout):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        _plot_aggregate_cve_panel(
            ax, by_ruleset, dtype, period, ruleset_colors, ruleset_labels,
            x_lim=x_lim_map[dtype], x_ticks=x_tick_map[dtype],
            show_legend=(row == 0 and col == 0))
        ax.set_title(f"{dtype_titles[dtype]} — {period_titles[period]}",
                     fontsize=8, pad=4)
        if col > 0:
            ax.set_yticklabels([])

    fig.text(0.5, 0.005, "# PCs", ha="center", fontsize=9)
    fig.text(0.005, 0.5, "MemoryAnti Variance Explained", va="center",
             rotation="vertical", fontsize=9)
    fig.tight_layout(rect=[0.03, 0.02, 1, 1])
    out_path = OUT_DIR / "aggregate_cve.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_aggregate_cve_stimulus():
    """
    Figure: stimulus-period-only CVE. Single row, two columns — hidden (left)
    and effective modulation (right) — overlaying the relevant and irrelevant
    motif rulesets, same conventions as plot_aggregate_cve.
    """
    _ensure_out_dir()

    analysis_types = ["hidden", "modulation", "modulation_weighted"]
    periods = ["stimulus", "response"]

    by_ruleset = _load_aggregate_cve_by_ruleset(analysis_types, periods)
    if not by_ruleset:
        print("  Skipped: no aggregate data found.")
        return

    ruleset_colors = {
        "fdgo_delaygo": "#3182ce",
        "fdanti_delaygo": "#e53e3e",
    }
    ruleset_labels = {
        "fdgo_delaygo": "Irrelevant motif",
        "fdanti_delaygo": "Relevant motif",
    }

    x_lim_map = {"hidden": 20, "modulation_weighted": 1000}
    x_tick_map = {"hidden": np.arange(0, 21, 5),
                  "modulation_weighted": np.arange(0, 1001, 200)}
    dtype_titles = {"hidden": "Hidden", "modulation_weighted": "Effective Modulation"}

    # One row, two columns: hidden | effective modulation, both stimulus period.
    col_dtypes = ["hidden", "modulation_weighted"]

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.6))

    for col, dtype in enumerate(col_dtypes):
        ax = axes[col]
        _plot_aggregate_cve_panel(
            ax, by_ruleset, dtype, "stimulus", ruleset_colors, ruleset_labels,
            x_lim=x_lim_map[dtype], x_ticks=x_tick_map[dtype],
            show_legend=(col == 0))
        ax.set_title(f"{dtype_titles[dtype]} — Stimulus Period",
                     fontsize=8, pad=4)
        if col > 0:
            ax.set_yticklabels([])

    fig.text(0.5, 0.005, "# PCs", ha="center", fontsize=9)
    fig.text(0.005, 0.5, "MemoryAnti Variance Explained", va="center",
             rotation="vertical", fontsize=9)
    fig.tight_layout(rect=[0.03, 0.04, 1, 1])
    out_path = OUT_DIR / "aggregate_cve_stimulus.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

# ─── Figure: single-task fixon/task cancellation ("show") ───────────────────

ONETASK_DIR = Path("onetask")
# Which single-task run to plot (set to the desired aname under onetask/{aname}/).
ONETASK_ANAME = "delaygo_seed865_hidden200+batch128+angle"


def plot_onetask_show():
    """
    Figure: per-stimulus fixon / task / combine modulation-component traces
    (the single-task "cancellation" figure), reloaded from the pickle saved by
    one_task_analysis.py. Shows how the fixon and task contributions cancel
    until the response period.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"show_{ONETASK_ANAME}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path} not found. Run one_task_analysis.py first.")
        return

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    per_stim = d["per_stimulus"]
    stimulus_start = d.get("stimulus_start")
    stimulus_end = d.get("stimulus_end")
    response_start = d.get("response_start")
    stim_labels = sorted(per_stim.keys())
    # Show only the 2nd and 3rd stimulus panels.
    sel_labels = stim_labels[1:3]
    if len(sel_labels) < 2:
        print(f"  Skipped: need >=3 stimuli, only {len(stim_labels)} available.")
        return

    # Trial periods: fixation | stimulus | memory(delay) | response, bounded by
    # the saved break times. Shade each with the canonical phase colors.
    period_spans = []
    if stimulus_start is not None and stimulus_end is not None and response_start is not None:
        period_spans = [
            (0, stimulus_start, _PHASE_COLORS["stim1"], "Fixation"),
            (stimulus_start, stimulus_end, _PHASE_COLORS["stim1"], "Stimulus"),
            (stimulus_end, response_start, _PHASE_COLORS["delay1"], "Memory"),
            (response_start, None, _PHASE_COLORS["go1"], "Response"),
        ]
        # Give fixation a neutral shade distinct from stimulus.
        period_spans[0] = (0, stimulus_start, "#f0f0f0", "Fixation")

    fig, axes = plt.subplots(len(sel_labels), 1, figsize=(3.4, 1.8 * len(sel_labels)),
                             squeeze=False)
    for i, lab in enumerate(sel_labels):
        ax = axes[i, 0]
        tr = per_stim[lab]
        T = len(tr["combine"])
        # Period shading behind the traces.
        for start, end, color, _name in period_spans:
            ax.axvspan(start, end if end is not None else T, color=color, alpha=0.35, lw=0, zorder=0)
        ax.plot(tr["fixon"], color=c_vals[0], label="Fixon", zorder=2)
        ax.plot(tr["task"], color=c_vals[1], label="Task", zorder=2)
        ax.plot(tr["combine"], color=c_vals[2], linewidth=2.5, label="Combine", zorder=3)
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        ax.set_xlim(0, T - 1)
        ax.set_ylim([-2.0, 2.0])
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(frameon=True, fontsize=6, loc="best")
        if i == len(sel_labels) - 1:
            ax.set_xlabel("Timestep", fontsize=9)
        else:
            ax.set_xticklabels([])

    # Shared y-label centered across the panels.
    fig.supylabel("Modulation component", fontsize=9)

    fig.tight_layout()
    out_path = OUT_DIR / "onetask_show.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_onetask_modulation_heatmap():
    """
    Figure: per-stimulus change of the fixon-synapse modulation across the
    hidden units, for the trained single-task network. Two stacked heatmaps
    (rows = stimulus directions, cols = hidden neurons): top = change during
    the stimulus period, bottom = change during the response period.
    Reloaded from the pickle saved by one_task_analysis.py.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"modulation_heatmap_{ONETASK_ANAME}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path} not found. Run one_task_analysis.py first.")
        return

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    stim_change = np.asarray(d["stim_change"])
    resp_change = np.asarray(d["response_change"])
    # Robust symmetric color limit: a few columns carry extreme values that
    # would otherwise wash the bulk of the map to flat gray. Clip the scale to
    # the 99th percentile of |change| so the typical structure is visible.
    both = np.concatenate([np.abs(stim_change).ravel(), np.abs(resp_change).ravel()])
    vmax = float(np.percentile(both, 99))

    fig, axes = plt.subplots(2, 1, figsize=(4, 4),
                             gridspec_kw={"hspace": 0.08})
    n_stim, n_hidden = stim_change.shape
    for ax, mat in [(axes[0], stim_change), (axes[1], resp_change)]:
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        # Keep short tick marks (no numeric labels) so the axes read cleanly.
        ax.set_xticks(np.linspace(0, n_hidden - 1, 5))
        ax.set_yticks(np.arange(n_stim))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", length=3, width=0.6, color="0.4")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_edgecolor("0.4")

    # Shared centered x / y labels placed close to the panels.
    axes[1].set_xlabel("Hidden", fontsize=9)               # bottom panel only
    fig.text(0.04, 0.5, "Stimuli", va="center", rotation="vertical", fontsize=9)

    # Shared single colorbar across both panels.
    cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.03)
    cbar.ax.tick_params(labelsize=7)

    out_path = OUT_DIR / "onetask_modulation_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_onetask_m_pca():
    """
    Figure: low-D PCA trajectory of the fixon-input modulation during the
    stimulus period, colored by stimulus direction (one trace per trial), for
    the trained single-task network. Only the PC1-PC2 plane is shown.
    Reloaded from the pickle saved by one_task_analysis.py — the PCA basis was
    fit on the fixon modulation itself (see one_task_analysis for rationale).
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"m_pca_{ONETASK_ANAME}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path} not found. Run one_task_analysis.py first.")
        return

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    lowd = np.asarray(d["lowd"])                 # (batch, T, n_pc)
    labels = np.asarray(d["labels"]).reshape(-1)
    s0, s1 = int(d["stimulus_start"]), int(d["stimulus_end"])

    # Only the PC1-PC2 plane.
    xpc, ypc = 0, 1
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    for i in range(lowd.shape[0]):
        color = c_vals[int(labels[i]) % len(c_vals)]
        ax.plot(lowd[i, s0:s1, xpc], lowd[i, s0:s1, ypc],
                marker="o", markersize=3, c=color, alpha=0.5)
    ax.set_xlabel(f"PC {xpc+1}", fontsize=10)
    ax.set_ylabel(f"PC {ypc+1}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "onetask_m_pca.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


ONETASK_DATA_DIR = Path("onetask_data")


def plot_onetask_corr_during_learning():
    """
    Figure: inter-stimulus cosine of the (fixon) modulation and hidden activity
    across training, aggregated over ALL single-task runs in onetask_data/.

    Each run contributes a corr_{aname}.npz (counter_lst, m_corr_stage,
    h_corr_stage) written by one_task_analysis.py. Runs are truncated to a
    common length; the mean across runs is plotted with a ±1 std band.
    Top panel = modulation cosine, bottom = hidden-activity cosine.
    """
    import glob as _glob

    _ensure_out_dir()
    files = sorted(_glob.glob(str(ONETASK_DATA_DIR / "corr_*.npz")))
    if not files:
        print("  Skipped: no corr_*.npz in onetask_data/. Run one_task_analysis.py first.")
        return

    counters, m_all, h_all = [], [], []
    for fpath in files:
        dd = np.load(fpath)
        counters.append(dd["counter_lst"])
        m_all.append(dd["m_corr_stage"])
        h_all.append(dd["h_corr_stage"])

    common = min(len(c) for c in counters)
    counters = np.array([c[:common] for c in counters])
    m_all = np.array([m[:common] for m in m_all])
    h_all = np.array([h[:common] for h in h_all])
    mean_counter = np.mean(counters, axis=0)
    n_runs = len(files)

    fig, axm = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    for ax, data, ylabel in [
        (axm[0], m_all, "Modulation\nstimulus similarity"),
        (axm[1], h_all, "Activity\nstimulus similarity"),
    ]:
        mean = data.mean(0)
        std = data.std(0)
        ax.plot(mean_counter, mean, "-o", color=c_vals[0], markersize=4)
        ax.fill_between(mean_counter, mean - std, mean + std,
                        color=c_vals[0], alpha=0.2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axm[1].set_xlabel("# Dataset", fontsize=11)
    axm[1].set_xscale("log")
    axm[0].set_title(f"n = {n_runs} runs", fontsize=9)

    fig.tight_layout()
    out_path = OUT_DIR / "onetask_corr_during_learning.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  (n={n_runs} runs)")


# ─── Figures grouped by mode ──────────────────────────────────────────────────
# Each mode maps a figure name → its plotting function. Figures only depend on
# the data produced by their corresponding experiment, so a mode can be run in
# isolation without touching the others' inputs.
#
#   one_task        single-task training analyses (multiple_task single-task run)
#   multiple_tasks  the full multi-task network: clustering, lesion, state space
#   pretraining     pretraining → post-training transfer analyses
FIGURES_BY_MODE = {
    "one_task": {
        "onetask_show": plot_onetask_show,
        "onetask_modulation_heatmap": plot_onetask_modulation_heatmap,
        "onetask_m_pca": plot_onetask_m_pca,
        "onetask_corr_during_learning": plot_onetask_corr_during_learning,
    },
    "multiple_tasks": {
        "fixed_points": plot_fixed_points_all,
        "input": plot_clustered_input,
        "hidden": plot_clustered_hidden,
        "modulation": plot_clustered_modulation,
        "l2_accuracy": plot_l2_vs_accuracy,
        "state_space_combined": plot_state_space_combined,
        "state_space_r_values": plot_state_space_r_values,
        "overmembership_unnorm": plot_overmembership_unnorm,
        "overmembership_weighted": plot_overmembership_weighted,
        "overmembership_var_weighted": plot_overmembership_var_weighted,
        "input_weight_correlation": plot_input_weight_correlation,
        "lesion_heatmap": plot_lesion_heatmap,
        "cluster_corr_vs_lesion": plot_cluster_corr_vs_lesion,
        "om_vs_lesion": plot_om_vs_lesion,
    },
    "pretraining": {
        "transfer_speed": plot_transfer_speed,
        "rule_vectors": plot_rule_vectors,
        "aggregate_cve": plot_aggregate_cve,
        "aggregate_cve_stimulus": plot_aggregate_cve_stimulus,
    },
}

# Flattened view: every figure across all modes, preserving mode order.
ALL_FIGURES = {
    name: fn
    for mode_figs in FIGURES_BY_MODE.values()
    for name, fn in mode_figs.items()
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate paper figures for one or more analysis modes."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", *FIGURES_BY_MODE.keys()],
        help="Which group of figures to generate. 'all' runs every mode "
             "(default).",
    )
    parser.add_argument(
        "--only",
        metavar="FIGURE",
        help="Generate a single figure by name (overrides mode).",
    )
    args = parser.parse_args()

    # Resolve the set of figures to generate.
    if args.only is not None:
        if args.only not in ALL_FIGURES:
            parser.error(
                f"unknown figure '{args.only}'. "
                f"Available: {', '.join(ALL_FIGURES)}"
            )
        figures = {args.only: ALL_FIGURES[args.only]}
        modes_run = "only"
    elif args.mode == "all":
        figures = ALL_FIGURES
        modes_run = "all"
    else:
        figures = FIGURES_BY_MODE[args.mode]
        modes_run = args.mode

    print(f"Experiment: {ANAME}")
    print(f"Output: {OUT_DIR}/")
    print(f"Mode: {modes_run} ({len(figures)} figure(s))")
    print()

    # Clear old figures. Only wipe the whole directory on a full 'all' run;
    # a single-mode (or --only) run leaves other modes' outputs untouched.
    _ensure_out_dir()
    if modes_run == "all":
        for f in OUT_DIR.iterdir():
            if f.is_file():
                f.unlink()

    import traceback

    failures = []
    for name, fn in figures.items():
        print(f"── Generating: {name} ──")
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR generating '{name}': {exc}")
            traceback.print_exc()
            failures.append(name)

    if failures:
        print(f"\nCompleted with {len(failures)} failed figure(s): {failures}")
    else:
        print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
