"""
Paper figure generation for the MultiTaskMPN project.

Each public function produces one publication-ready figure and saves it
to the `paper_plot/` directory. Run the script directly to generate all
figures, or import individual functions as needed.

Figures are grouped into modes by the experiment they depend on:
    one_task         single-task training analyses
    multiple_tasks   full multi-task network (clustering, lesion, state space)
    two_in_multiple  two-task probes within the multi-task network (DMC memory)
    pretraining      pretraining → post-training transfer analyses
    two_task         two-task network (cross-task / cross-period PCA)

Usage:
    python paper_plot.py                       # generate every mode
    python paper_plot.py all                   # same as above
    python paper_plot.py one_task              # only the one-task figures
    python paper_plot.py multiple_tasks        # only the multi-task figures
    python paper_plot.py two_in_multiple       # only the two-in-multiple figures
    python paper_plot.py pretraining           # only the pretraining figures
    python paper_plot.py two_task              # only the two-task figures
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

# Multi-task run used for the DMC category-memory probe (two_in_multiple mode).
# May differ from ANAME — set independently so the attractor figure can come
# from a different seed/regularization than the clustering/lesion figures.
DMC_ANAME = "everything_seed299_L21e3+hidden300+batch128+angle"

# Two-task run used for the cross-task / cross-period PCA figure (d_combine).
# The matching data file is written by two_task_analysis.py into
# twotasks/{TWOTASK_ANAME}/d_combine_{TWOTASK_ANAME}.pkl.
TWOTASK_ANAME = "delaygofamily_seed21_reg1e3+hidden200"
TWOTASKS_DIR = Path("twotasks")

# Two-task run used for the attractor first-subplot figure (independent of
# TWOTASK_ANAME so it can come from a different seed/regularization).
TWOTASK_ATTRACTOR_ANAME = "delaygofamily_seed21_reg1e3+hidden200"


def _twotask_seed_tag():
    """Seed substring of TWOTASK_ANAME (e.g. 'seed894'), for figure filenames.
    Falls back to the full aname if no 'seed<N>' token is present."""
    import re as _re
    m = _re.search(r"seed\d+", TWOTASK_ANAME)
    return m.group(0) if m else TWOTASK_ANAME

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


def _task_display_name(name):
    """Task-only display label for a '{rule}-{phase}' tick.

    Drops the phase/session suffix (e.g. 'Response', 'Memory1') and returns
    just the task display name (e.g. 'DelayPro'). Falls back to the full
    relabeled name if no phase suffix is present.
    """
    phase = _phase_of(name)
    if phase is not None:
        rule = name[: -(len(phase) + 1)]
        return _TASK_DISPLAY.get(rule, rule)
    return _relabel_tb_name(name)


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
    display_names = [_task_display_name(nm) for nm in ordered_names]
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
    display_names = [_task_display_name(nm) for nm in ordered_names]
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
    # Adaptive y-range: pad the observed accuracy span by 5% of its extent
    # (min 2 pts), clamped to the valid [0, 100] accuracy interval.
    lo, hi = float(acc_vals.min()), float(acc_vals.max())
    pad = max((hi - lo) * 0.05, 2.0)
    ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + pad))
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

    # Transform raw rule names to their display/session names; stimulus-channel
    # labels are already display-ready.
    rule_labels = [_TASK_DISPLAY.get(r, r) for r in rules]
    all_input = ["Fix On", "Fix Off", "Stim 1 Cos", "Stim 1 Sin",
                 "Stim 2 Cos", "Stim 2 Sin"] + rule_labels
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

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.2 * 2 / 3))  # height squeezed by 1/3

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
    ax.set_ylabel("Accuracy\nthreshold (%)", ha="center")
    ax.set_xscale("log")
    ax.set_yticks([50, 75, 100])
    ax.legend(fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "transfer_speed.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_learning_trajectory():
    """
    Figure: post-training learning trajectory — accuracy vs training iteration,
    comparing fdgo_delaygo vs fdanti_delaygo rulesets (L21e3). Same rulesets /
    colors as plot_transfer_speed, but plotting the full accuracy curve (mean ±
    std across seeds) rather than iterations-to-threshold.

    Reads per-seed result pickles (learning.acc_iter_post / learning.acc_post).
    Seeds are resampled onto a shared iteration grid before averaging, so it is
    robust to slightly different logging cadences across seeds.
    """
    import re as _re

    _ensure_out_dir()
    if not PRETRAINING_ANALYSIS_DIR.exists():
        print("  Skipped: pretraining_analysis/ not found.")
        return

    addon_name = "+hidden200+L21e3+batch128+angle"
    pkls = sorted(PRETRAINING_ANALYSIS_DIR.glob(f"*_dmpn_seed*_{addon_name}_result.pkl"))
    if not pkls:
        print("  Skipped: no pretraining result pickles found.")
        return

    # Collect each ruleset's per-seed (iterations, accuracy) trajectories.
    by_ruleset_traj = {}  # rs -> list of (iters, acc)
    for p in pkls:
        m = _re.match(
            r'(.+)_dmpn_seed\d+_\+hidden200\+L21e3\+batch128\+angle_result\.pkl', p.name
        )
        if not m:
            continue
        rs = m.group(1)
        with open(p, "rb") as f:
            data = pickle.load(f)
        learn = data.get("learning", {})
        if "acc_iter_post" not in learn or "acc_post" not in learn:
            continue
        iters = np.asarray(learn["acc_iter_post"], dtype=float)
        acc = np.asarray(learn["acc_post"], dtype=float)
        by_ruleset_traj.setdefault(rs, []).append((iters, acc))

    if not by_ruleset_traj:
        print("  Skipped: no learning trajectories found.")
        return

    ruleset_colors = {
        "fdgo_delaygo": "#3182ce",
        "fdanti_delaygo": "#e53e3e",
    }
    ruleset_labels = {
        "fdgo_delaygo": "Irrelevant motif",
        "fdanti_delaygo": "Relevant motif",
    }

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.2 * 2 / 3))  # match transfer_speed

    for rs in sorted(by_ruleset_traj.keys()):
        trajs = by_ruleset_traj[rs]
        color = ruleset_colors.get(rs, "#718096")
        label = ruleset_labels.get(rs, rs)

        # Shared iteration grid = intersection of every seed's [min, max] range,
        # log-spaced so the (log-x) curve is evenly sampled; interpolate each
        # seed onto it, then average.
        lo = max(t[0].min() for t in trajs)
        hi = min(t[0].max() for t in trajs)
        grid = np.unique(np.round(np.geomspace(max(lo, 1.0), hi, 400)).astype(int))
        grid = grid[grid >= 1].astype(float)
        resampled = np.array([np.interp(grid, it, ac) for (it, ac) in trajs])

        mean_vals = resampled.mean(axis=0) * 100
        std_vals = resampled.std(axis=0) * 100
        ax.plot(grid, mean_vals, "-", color=color, linewidth=2.0, label=label)
        ax.fill_between(grid, mean_vals - std_vals, mean_vals + std_vals,
                        color=color, alpha=0.15)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    ax.set_yticks([0, 50, 100])
    ax.set_ylim([0, 105])
    ax.legend(fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "learning_trajectory.png"
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

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4 * 2 / 3))  # height squeezed by 1/3

    # All bars are evenly spaced (no extra gap between column groups).
    bar_step = 1.0
    bar_width = 0.8
    group_gap = 0.0

    # Build the per-ruleset bar list first, then interleave columns across
    # rulesets so the colors alternate (red1, blue1, red2, blue2, ...) instead
    # of grouping all of one ruleset's bars together.
    per_rs_bars = {}  # rs -> list of (label, mean, std, vals)
    for rs in rs_list:
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
        per_rs_bars[rs] = [
            (lbl, float(np.mean(by_ruleset[rs][k])),
             float(np.std(by_ruleset[rs][k])), np.array(by_ruleset[rs][k]))
            for (k, lbl, _) in bar_specs
        ]

    # Interleave: for each column index, emit one bar per ruleset (red then
    # blue), so bars alternate color; a group_gap separates successive columns.
    all_x, all_labels = [], []
    labeled = set()  # ensure each ruleset appears once in the legend
    n_cols = max((len(bars) for bars in per_rs_bars.values()), default=0)
    x = 0.0
    for col in range(n_cols):
        for rs in rs_list:
            bars = per_rs_bars[rs]
            if col >= len(bars):
                continue
            lbl, mean, std, vals = bars[col]
            color = ruleset_colors.get(rs, "#718096")
            legend_label = None if rs in labeled else ruleset_labels.get(rs, rs)
            labeled.add(rs)

            ax.bar(x, mean, bar_width, yerr=std, capsize=2,
                   color=color, alpha=0.8, edgecolor="k", linewidth=0.5,
                   label=legend_label)
            ax.plot(np.full_like(vals, x), vals, "k.", markersize=3, alpha=0.6)

            all_x.append(x)
            all_labels.append(lbl)
            x += bar_step
        x += group_gap  # gap between successive interleaved column groups

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

    fig, axes = plt.subplots(2, 2, figsize=(6, 4.5 * 2 / 3))  # height squeezed by 1/3

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

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.6 * 2 / 3))  # height squeezed by 1/3

    for col, dtype in enumerate(col_dtypes):
        ax = axes[col]
        _plot_aggregate_cve_panel(
            ax, by_ruleset, dtype, "stimulus", ruleset_colors, ruleset_labels,
            x_lim=x_lim_map[dtype], x_ticks=x_tick_map[dtype],
            show_legend=False)
        ax.set_title(f"{dtype_titles[dtype]} — Stimulus Period",
                     fontsize=8, pad=4)
        if col > 0:
            ax.set_yticklabels([])

    fig.text(0.5, 0.005, "# PCs", ha="center", fontsize=9)
    fig.text(0.005, 0.5, "MemoryAnti\nVariance Explained", va="center",
             ha="center", rotation="vertical", fontsize=9)
    fig.tight_layout(rect=[0.03, 0.04, 1, 1])
    out_path = OUT_DIR / "aggregate_cve_stimulus.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

# ─── Figure: single-task fixon/task cancellation ("show") ───────────────────

ONETASK_DIR = Path("onetask")
# Which single-task run to plot (set to the desired aname under onetask/{aname}/).
ONETASK_ANAME = "delaygo_seed395_hidden200+batch128+angle"


def plot_onetask_example_trial():
    """
    Figures: one representative single-task trial, saved as TWO files:
      onetask_example_trial_input.png  — 4 vertically-stacked input subplots:
        Fixation, Modality 1 (cos+sin), Modality 2 (cos+sin), Task cue.
      onetask_example_trial_output.png — network vs target output.
    Reloaded from the pickle saved by one_task_analysis.py. Y-ticks are just
    [-1, 1] on every panel.

    Input channel layout (low_dim, no fixate_off): 0=Fixation, 1-2=Modality 1
    (cos,sin), 3-4=Modality 2 (cos,sin), 5=Task cue.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"example_trial_{ONETASK_ANAME}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path} not found. Run one_task_analysis.py first.")
        return

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    inp = np.asarray(d["input"])              # (T, n_input)
    net_out = np.asarray(d["net_output"])     # (T, n_output)
    target = np.asarray(d["target_output"])   # (T, n_output)
    out_labels = d["output_labels"]
    T = inp.shape[0]

    # Light palette for the faded target-output shadows (matches one_task_analysis).
    c_vals_l = ["#feb2b2", "#90cdf4", "#9ae6b4", "#d6bcfa", "#fbd38d",
                "#81e6d9", "#e2e8f0", "#fbb6ce", "#faf089"] * 10

    # Trial periods: fixation | stimulus | memory(delay) | response.
    stimulus_start = d.get("stimulus_start")
    stimulus_end = d.get("stimulus_end")
    response_start = d.get("response_start")
    period_spans = []
    if stimulus_start is not None and stimulus_end is not None and response_start is not None:
        period_spans = [
            (0, stimulus_start, "#f0f0f0"),
            (stimulus_start, stimulus_end, _PHASE_COLORS["stim1"]),
            (stimulus_end, response_start, _PHASE_COLORS["delay1"]),
            (response_start, None, _PHASE_COLORS["go1"]),
        ]

    def _shade(ax):
        for start, end, color in period_spans:
            ax.axvspan(start, end if end is not None else T - 1,
                       color=color, alpha=0.35, lw=0, zorder=0)

    def _style(ax, ylabel, last_row):
        ax.set_xlim(0, T - 1)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1, 1])          # only -1 and 1, as requested
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        # x tick labels only on the bottom subplot. NB: with sharex=True, calling
        # set_xticklabels([]) on a non-last axis blanks the shared tick text for
        # the bottom row too, so toggle visibility via tick_params instead.
        ax.tick_params(axis="x", labelbottom=last_row)
        if last_row:
            ax.set_xlabel("Time step", fontsize=10)

    # ── Input figure: 4 stacked subplots ────────────────────────────────────
    # (channel indices, per-channel colors+labels, panel ylabel)
    n_in = inp.shape[1]
    input_groups = [
        ([0], [c_vals[0]], ["Fixation"], "Fixation"),
        ([1, 2], [c_vals[1], c_vals[2]], ["Mod1 cos", "Mod1 sin"], "Modality 1"),
        ([3, 4], [c_vals[3], c_vals[4]], ["Mod2 cos", "Mod2 sin"], "Modality 2"),
        ([5], [c_vals[5]], ["Task cue"], "Task cue"),
    ]
    # keep only groups whose channels exist in this input
    input_groups = [(chs, cols, labs, ylab) for chs, cols, labs, ylab in input_groups
                    if all(ch < n_in for ch in chs)]

    figin, axin = plt.subplots(len(input_groups), 1, figsize=(4, 1.0 * len(input_groups)),
                               sharex=True, squeeze=False)
    for row, (chs, cols, labs, ylab) in enumerate(input_groups):
        ax = axin[row, 0]
        _shade(ax)
        for ch, col, lab in zip(chs, cols, labs):
            ax.plot(inp[:, ch], color=col, label=lab, zorder=2)
        _style(ax, ylab, last_row=(row == len(input_groups) - 1))
        ax.legend(fontsize=6, frameon=True, loc="upper right", ncol=len(chs))
    figin.tight_layout()
    out_in = OUT_DIR / "onetask_example_trial_input.png"
    figin.savefig(out_in, dpi=300, bbox_inches="tight")
    plt.close(figin)
    print(f"Saved: {out_in}")

    # ── Output figure: one stacked subplot per output channel ────────────────
    figout, axout = plt.subplots(net_out.shape[-1], 1,
                                 figsize=(4, 1.0 * net_out.shape[-1]),
                                 sharex=True, squeeze=False)
    for out_idx in range(net_out.shape[-1]):
        ax = axout[out_idx, 0]
        _shade(ax)
        lab = out_labels[out_idx] if out_idx < len(out_labels) else f"out {out_idx}"
        ax.plot(target[:, out_idx], color=c_vals_l[out_idx % len(c_vals_l)],
                linewidth=4, alpha=0.6, zorder=2, label="target")
        ax.plot(net_out[:, out_idx], color=c_vals[out_idx % len(c_vals)],
                zorder=3, label=lab)
        _style(ax, lab, last_row=(out_idx == net_out.shape[-1] - 1))
        ax.legend(fontsize=6, frameon=True, loc="upper right", ncol=2)
    figout.tight_layout()
    out_out = OUT_DIR / "onetask_example_trial_output.png"
    figout.savefig(out_out, dpi=300, bbox_inches="tight")
    plt.close(figout)
    print(f"Saved: {out_out}")


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
        # Colors matched to onetask_example_trial: Fixon = Fixation (c_vals[0]),
        # Task = Task Cue (c_vals[3]). Combine uses a new color not used there.
        ax.plot(tr["fixon"], color=c_vals[0], label="Fixon", zorder=2)
        ax.plot(tr["task"], color=c_vals[3], label="Task", zorder=2)
        ax.plot(tr["combine"], color=c_vals[4], linewidth=2.5, label="Combine", zorder=3)
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

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4),
                             gridspec_kw={"wspace": 0.08})
    n_stim, n_hidden = stim_change.shape
    titles = ["Stimulus period", "Response period"]
    for col, (ax, mat) in enumerate([(axes[0], stim_change), (axes[1], resp_change)]):
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
        ax.set_xlabel("Hidden", fontsize=9)
        ax.set_title(titles[col], fontsize=9)
    # y-label on the left panel only
    axes[0].set_ylabel("Stimuli", fontsize=9)

    out_path = OUT_DIR / "onetask_modulation_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_onetask_long_fixed_points():
    """
    Figure: per-period trajectory + fixed point (last frame) of the single-task
    network, in each period's own top-2 PCA. One grid: top row = hidden, bottom
    row = e_modulation; columns = periods (Fixation/Stimulus/Delay/Response).
    Color = stimulus. Reloaded from the pickle saved by one_task_analysis.py's
    long_period_fixed_points.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"long_fixed_points_{ONETASK_ANAME}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path} not found. Run one_task_analysis.py first.")
        return

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    present = d["present"]
    period_title = d["period_title"]
    data = d["data"]
    reps = [r for r in ("hidden", "e_modulation") if r in data]
    if not reps or not present:
        print("  Skipped: no rep/period data in pickle.")
        return

    n_row, n_col = len(reps), len(present)
    fig, axs = plt.subplots(n_row, n_col, figsize=(3.2 * n_col, 3.2 * n_row), squeeze=False)
    for r, rep in enumerate(reps):
        for cidx, v in enumerate(present):
            ax = axs[r][cidx]
            ent = data[rep].get(v)
            if ent is None:
                ax.axis("off")
                continue
            proj = np.asarray(ent["proj"])      # (batch, win_T, 2)
            stim = np.asarray(ent["stim"])
            for i in range(proj.shape[0]):
                col = c_vals[int(stim[i]) % len(c_vals)]
                p = proj[i]
                ax.plot(p[:, 0], p[:, 1], color=col, alpha=0.4, linewidth=0.8, zorder=2)
                ax.scatter(p[-1, 0], p[-1, 1], color=col, marker="o", s=45,
                           edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(period_title.get(v, v), fontsize=12)

    # Shared x/y labels for the whole grid (all panels share the same axes).
    fig.supxlabel("Delay PC1", fontsize=12)
    fig.supylabel("Delay PC2", fontsize=12)

    # stimulus-color legend on the top-left panel
    uniq_stim = sorted(set(int(s) for rep in reps for v in present
                           for s in np.asarray(data[rep][v]["stim"])))
    stim_handles = [plt.Line2D([0], [0], marker="o", linestyle="None",
                               markerfacecolor=c_vals[s % len(c_vals)],
                               markeredgecolor="black", markersize=6, label=f"stim {s}")
                    for s in uniq_stim]
    axs[0][0].legend(handles=stim_handles, frameon=True, fontsize=6, ncol=2,
                     title="stimulus", loc="best")

    fig.tight_layout()
    out_path = OUT_DIR / "onetask_long_fixed_points.png"
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


def plot_two_task_d_combine():
    """
    Figure: cross-task / cross-period PCA explained-variance heatmaps for the
    two-task network — one panel each for hidden activity and modulation.

    Reads the self-contained d_combine pickle written by two_task_analysis.py
    (twotasks/{TWOTASK_ANAME}/d_combine_{TWOTASK_ANAME}.pkl), which stores the
    already-permuted 8x8 FVE matrix, its tick labels, and the color range for
    each of "hidden" and "modulation".
    """
    _ensure_out_dir()
    pkl_path = TWOTASKS_DIR / TWOTASK_ANAME / f"d_combine_{TWOTASK_ANAME}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path} not found. Run two_task_analysis.py first.")
        return

    with open(pkl_path, "rb") as f:
        d_combine = pickle.load(f)

    names = [n for n in ("hidden", "modulation") if n in d_combine]
    # Shared color range across panels so a single colorbar applies to both.
    vmin = min(d_combine[n].get("vmin", 0.0) for n in names)
    vmax = max(d_combine[n].get("vmax", 1.0) for n in names)

    fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.9),
                            gridspec_kw={"wspace": 0.45})
    if len(names) == 1:
        axs = [axs]
    mesh = None
    for i, (ax, name) in enumerate(zip(axs, names)):
        e = d_combine[name]
        sns.heatmap(np.asarray(e["fve_k_all"]), ax=ax,
                    xticklabels=e["labels"], yticklabels=e["labels"],
                    annot=True, fmt=".2f", vmin=vmin, vmax=vmax, square=True,
                    cbar=False)
        mesh = ax.collections[0]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # One shared colorbar for all panels.
    fig.colorbar(mesh, ax=list(axs), shrink=0.8)
    out_path = OUT_DIR / f"twotask_d_combine_{_twotask_seed_tag()}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_m_pca_panels(data, title_prefix, out_name, legend_frameon=False,
                       show_legend=True):
    """Redraw the m_pca trajectory figure (PCA 1-2 only) from the data dict
    stashed by two_task_analysis.py (cell 86, "normal" variant). Mirrors the
    original alpha / marker / color conventions; tick/label sizing matches the
    two_task_attractor figures."""
    _ensure_out_dir()
    projected = np.asarray(data["projected_data"])          # (batch, T, 3)
    ltc = np.asarray(data["label_task_comb"])
    ts = data["time_stamps"]
    phases = data["phases"]
    transitions = data["transitions"]
    period_markers = data["period_markers"]
    markers_vals = data["markers_vals"]
    linestyles = data["linestyles"]

    batch_num = projected.shape[0]
    stim0, trial_end = ts["stimulus_start"], ts["trial_end"]
    a, bb = 0, 1  # PCA 1-2 only

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    legend_handles = [
        plt.Line2D([0], [0], marker=markers_vals[idx], linestyle="None", markersize=10,
                   markerfacecolor="k", markeredgecolor="k", label=label)
        for label, idx in period_markers.items()
    ]

    for i in range(batch_num):
        task = ltc[i, 1]
        if task not in (0, 1):
            continue
        color = c_vals[ltc[i, 0]]
        ls = linestyles[task]
        data_i = projected[i]
        seg = slice(stim0, trial_end)
        ax.plot(data_i[seg, a], data_i[seg, bb], c=color, linestyle=ls, alpha=0.5)
        for _, t0_key, t1_key, mk_idx in phases:
            sl = slice(ts[t0_key], ts[t1_key])
            ax.scatter(data_i[sl, a], data_i[sl, bb], c=color,
                       marker=markers_vals[mk_idx], alpha=0.8)
        for t_key, mk_idx in transitions:
            t = ts[t_key] - 1
            ax.scatter([data_i[t, a]], [data_i[t, bb]], c=color,
                       marker=markers_vals[mk_idx], alpha=1.0, s=60,
                       linewidths=0.6, zorder=10)
    ax.set_xlabel("PCA 1", fontsize=20)
    ax.set_ylabel("PCA 2", fontsize=20)
    ax.tick_params(axis="both", labelsize=15)
    if show_legend:
        ax.legend(handles=legend_handles, loc="upper right",
                  frameon=legend_frameon, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_two_task_m_pca():
    """
    Figure: PCA trajectories of hidden activity and modulation for the two-task
    network ("normal" variant). Reads the self-contained m_pca pickle written
    by two_task_analysis.py and emits one figure per panel type
    (hidden / modulation).
    """
    run_dir = TWOTASKS_DIR / TWOTASK_ANAME
    matches = sorted(run_dir.glob("m_pca_normal_*.pkl"))
    if not matches:
        print(f"  Skipped: no m_pca_normal_*.pkl in {run_dir}. Run two_task_analysis.py first.")
        return

    with open(matches[0], "rb") as f:
        m_pca = pickle.load(f)

    for name in ("hidden", "modulation"):
        if name not in m_pca:
            continue
        _plot_m_pca_panels(m_pca[name], title_prefix=f"{name} (normal)",
                           out_name=f"twotask_m_pca_{name}_{_twotask_seed_tag()}.png",
                           legend_frameon=(name == "hidden"),
                           show_legend=(name == "hidden"))


def _draw_attractor_cycle_pc12(ax, entry, show_ylabel=True):
    """Draw the PCA 1-2 fixed-point "cycle" for one (period, series) entry onto
    ax. Per stimulus, connects the fixed points across alpha steps; overlays
    dashed rings at the alpha indices in ring_indices. The x-label is shared at
    the figure level (set by the caller), so it is not drawn here."""
    pdf_all = np.asarray(entry["projected_data_fix_all"])  # (n_alpha, batch, 3)
    interpolation_label = entry["interpolation_label"]
    ring_indices = entry["ring_indices"]
    comb = [0, 1]  # PCA 1-2 only

    # Light shades for the dashed overlay rings (mirrors c_vals_l in analysis).
    c_vals_l = ["#feb2b2", "#90cdf4", "#9ae6b4", "#fbd38d", "#fbb6ce"] * 10

    for it_idx, it in enumerate(ring_indices):
        xy = pdf_all[it][:, [comb[0], comb[1]]]
        num_xy = xy.shape[0]
        for j in range(num_xy):
            ax.plot([xy[j % num_xy, 0], xy[(j + 1) % num_xy, 0]],
                    [xy[j % num_xy, 1], xy[(j + 1) % num_xy, 1]],
                    linestyle="--", linewidth=3, color=c_vals_l[it_idx])
    for i in range(len(interpolation_label)):
        fixed_points = pdf_all[:, i, :]
        color = c_vals[interpolation_label[i]]
        # Connecting line stays faint; the per-stimulus endpoint markers ramp
        # their opacity from 0 -> 1 across the alpha interpolation steps, so the
        # anti end is nearly transparent and the pro end is solid.
        ax.plot(fixed_points[:, comb[0]], fixed_points[:, comb[1]],
                "-", c=color, alpha=0.3, zorder=1)
        n_steps = fixed_points.shape[0]
        point_alphas = (np.linspace(0.0, 1.0, n_steps) if n_steps > 1
                        else np.array([1.0]))
        ax.scatter(fixed_points[:, comb[0]], fixed_points[:, comb[1]],
                   c=color, alpha=point_alphas, marker="o", zorder=2)

    if show_ylabel:
        ax.set_ylabel("PCA 2", fontsize=20)
    ax.tick_params(axis="both", labelsize=15)
    ax.spines[["top", "right"]].set_visible(False)


# Long-period variant -> clean display title.
_PERIOD_TITLE = {
    "longfixation": "Fixation",
    "longstimulus": "Stimulus",
    "longdelay": "Delay",
    "longresponse": "Response",
}


def plot_two_task_attractor_cycle():
    """
    Figure: interpolation fixed-point "cycle" plots (PCA 1-2 only) for the
    two-task network. One figure per series (hidden / modulation /
    w_modulation), each a 1x4 row with one panel per long-period variant
    (Fixation, Stimulus, Delay, Response). A single shared "PCA 1" x-label spans
    the row. Reads the self-contained pickle written by two_task_analysis.py.
    """
    _ensure_out_dir()
    run_dir = TWOTASKS_DIR / TWOTASK_ANAME
    matches = sorted(run_dir.glob("m_pca_attractor_cycle_*.pkl"))
    if not matches:
        print(f"  Skipped: no m_pca_attractor_cycle_*.pkl in {run_dir}. "
              f"Run two_task_analysis.py first.")
        return

    with open(matches[0], "rb") as f:
        ac = pickle.load(f)

    names = ["hidden", "modulation", "w_modulation"]
    periods = ["longfixation", "longstimulus", "longdelay", "longresponse"]

    def _render(name):
        fig, axs = plt.subplots(1, len(periods), figsize=(5 * len(periods), 5))
        for col, (ax, sname) in enumerate(zip(axs, periods)):
            key = f"{sname}|{name}"
            if key not in ac:
                ax.axis("off")
                continue
            _draw_attractor_cycle_pc12(ax, ac[key], show_ylabel=(col == 0))
            ax.set_title(_PERIOD_TITLE.get(sname, sname), fontsize=22)
        # Single shared x-label for the whole row.
        fig.supxlabel("PCA 1", fontsize=20)
        fig.tight_layout()
        out_path = OUT_DIR / f"twotask_attractor_cycle_{name}_{_twotask_seed_tag()}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def _render_combined(row_names):
        """Stack the given series as rows in one figure (one period per column).
        Period titles are drawn once on the top row (shared across rows); a
        single "PCA 1" x-label spans the whole figure."""
        present = [n for n in row_names
                   if any(f"{sname}|{n}" in ac for sname in periods)]
        if len(present) < 2:
            return
        nrows, ncols = len(present), len(periods)
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        for r, name in enumerate(present):
            for col, sname in enumerate(periods):
                ax = axs[r, col]
                key = f"{sname}|{name}"
                if key not in ac:
                    ax.axis("off")
                    continue
                _draw_attractor_cycle_pc12(ax, ac[key], show_ylabel=False)
                if r == 0:  # shared period titles on the top row only
                    ax.set_title(_PERIOD_TITLE.get(sname, sname), fontsize=22)
        fig.supxlabel("PCA 1", fontsize=20)
        fig.supylabel("PCA 2", fontsize=20)
        fig.tight_layout()
        tag = "_".join(present)
        out_path = OUT_DIR / f"twotask_attractor_cycle_{tag}_{_twotask_seed_tag()}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    plotted = 0
    for name in names:
        if not any(f"{sname}|{name}" in ac for sname in periods):
            continue
        _render(name)
        plotted += 1
    if plotted == 0:
        print(f"  Skipped: no expected entries found in {matches[0].name}.")
        return

    # Combined figure: hidden (top row) + w_modulation (bottom row).
    _render_combined(["hidden", "w_modulation"])


def plot_two_task_cancel():
    """
    Figure: fixon/task cancellation projection traces for the two-task network,
    for selected stimuli (default 2 & 6) × both task columns — 2x2 panels. Each
    panel shows Combine (= Fix On + Task + Bias), Fix On, Task+Bias, and Fixoff
    (if fixate_off). Reads the self-contained cancel pickle written by
    two_task_analysis.py.
    """
    _ensure_out_dir()
    run_dir = TWOTASKS_DIR / TWOTASK_ANAME
    matches = sorted(run_dir.glob("cancel_seed*.pkl"))
    if not matches:
        print(f"  Skipped: no cancel_seed*.pkl in {run_dir}. Run two_task_analysis.py first.")
        return

    with open(matches[0], "rb") as f:
        saved = pickle.load(f)
    stimuli = saved["stimuli"]
    markers = saved["markers"]

    stim_keys = sorted(stimuli.keys())
    n_rows = len(stim_keys)

    # Trial periods: fixation | stimulus | memory(delay) | response, bounded by
    # the saved marker times. Shade each with the canonical phase colors,
    # matching onetask_show (fixation gets a neutral gray).
    fix_end = markers["fixation_end"]
    stim_end = markers["stimulus_end"]
    delay_end = markers["delay_end"]
    period_spans = [
        (0, fix_end, "#f0f0f0", "Fixation"),
        (fix_end, stim_end, _PHASE_COLORS["stim1"], "Stimulus"),
        (stim_end, delay_end, _PHASE_COLORS["delay1"], "Memory"),
        (delay_end, None, _PHASE_COLORS["go1"], "Response"),
    ]

    fig, axs = plt.subplots(n_rows, 2, figsize=(3.4 * 2, 1.8 * n_rows),
                            squeeze=False)
    # column 0 = task1 (pro), column 1 = task2 (anti)
    cols = [
        ("fixon_proj1", "x_task1_proj", "fixoff_proj1", "Task 1 (pro)"),
        ("fixon_proj2", "x_task2_proj", "fixoff_proj2", "Task 2 (anti)"),
    ]
    for r, si in enumerate(stim_keys):
        e = stimuli[si]
        bias = e["bias_proj"]
        for c, (fixon_k, task_k, fixoff_k, col_name) in enumerate(cols):
            ax = axs[r][c]
            fixon = np.asarray(e[fixon_k])
            task = np.asarray(e[task_k])
            T = len(fixon)
            # Period shading behind the traces.
            for start, end, color, _name in period_spans:
                ax.axvspan(start, end if end is not None else T - 1,
                           color=color, alpha=0.35, lw=0, zorder=0)
            ax.axhline(0, color="0.6", lw=0.8, zorder=1)
            # Colors matched to onetask_show / onetask_example_trial:
            # Fixon = c_vals[0], Task = c_vals[3], Combine = c_vals[4].
            ax.plot(fixon, color=c_vals[0], label="Fixon", zorder=2)
            ax.plot(task + bias, color=c_vals[3], label="Task", zorder=2)
            if e.get("fixate_off"):
                ax.plot(np.asarray(e[fixoff_k]), color=c_vals[5],
                        label="Fixoff", zorder=2)
            ax.plot(fixon + task + bias, color=c_vals[4], linewidth=2.5,
                    label="Combine", zorder=3)
            ax.set_xlim(0, T - 1)
            ax.set_ylim([-1.5, 1.5])
            ax.set_title(f"Stimulus {si}; {col_name}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0 and c == 0:
                ax.legend(frameon=True, fontsize=6, loc="best")
            if r == n_rows - 1:
                ax.set_xlabel("Timestep", fontsize=9)
            else:
                ax.set_xticklabels([])

    # Shared y-label centered across the panels.
    fig.supylabel("Proj Cos Mag", fontsize=9)

    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_cancel_{_twotask_seed_tag()}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  (stimuli {stim_keys})")


def plot_two_task_attractor_first():
    """
    Figures: the first subplot ("Hidden" panel) of each of the two attractor
    figures, saved separately (style matches twotask_m_pca_hidden):
      twotask_attractor_first_overlearning_{seed} — cosine sim vs iteration
      twotask_attractor_first_posttraining_{seed}  — cosine sim vs trial epoch
    Reloaded from the attractor_first pickle written by two_task_analysis.py.
    """
    _ensure_out_dir()
    aname = TWOTASK_ATTRACTOR_ANAME
    pkl_path = TWOTASKS_DIR / aname / f"attractor_first_{aname}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path} not found. Run two_task_analysis.py first.")
        return

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    ol = d.get("over_learning_hidden")
    st = d.get("stage_posttraining_hidden")
    if ol is None or st is None:
        print("  Skipped: attractor_first pickle missing expected entries.")
        return

    import re as _re
    m = _re.search(r"seed\d+", aname)
    seed_tag = m.group(0) if m else aname

    # ── Figure 1: over-learning (cosine similarity vs iteration) ──────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5 * 0.75))  # height reduced by 1/4
    x = np.asarray(ol["counter_lst"])
    for i, name in enumerate(ol["break_names"]):
        mean = np.asarray(ol["mean"][i])
        std = np.asarray(ol["std"][i])
        ax.plot(x, mean, "-o", color=c_vals[i], label=name)
        ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=c_vals[i])
    ax.set_xscale("log")
    ax.set_xlabel(ol.get("xlabel", "Iteration"), fontsize=20)
    ax.set_ylabel(ol.get("ylabel", "Cosine Similarity"), fontsize=20)
    ax.set_ylim([0, 1.05])
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(frameon=True, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_attractor_first_overlearning_{seed_tag}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── Figure 2: post-training stage (cosine similarity vs trial epoch) ──────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5 * 0.75))  # height reduced by 1/4
    keys = st["keys"]
    xs = np.arange(len(keys))
    for i, name in enumerate(st["break_names"]):
        mean = np.asarray(st["mean"][i])
        std = np.asarray(st["std"][i])
        ax.plot(xs, mean, "-o", color=c_vals[i], label=name)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.3, color=c_vals[i])
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel(st.get("ylabel", "Cosine Similarity"), fontsize=20)
    ax.set_ylim([-1.1, 1.1])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])  # every 0.5
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_attractor_first_posttraining_{seed_tag}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_dmc_memory_attractor():
    """
    Figure: DMC category-memory attractors for each representation (hidden,
    m_modulation, e_modulation), each in its own optimal 2-PC plane (the plane
    with the highest group-separation 2D silhouette).

    Reloads the fixed-point pickle written by multiple_task_analysis.py's
    shared_run (multiple_tasks/{DMC_ANAME}/dmcgo_fixed_points_{DMC_ANAME}.pkl).
    DMC_ANAME is independent of ANAME, so this can come from a different
    seed/regularization than the other multi-task figures. Shows the delay-period
    trajectory + end-of-delay fixed points for 8 stimuli x 2 tasks (color =
    stimulus, marker = task: square = Pro/dmcgo, triangle = Anti/dmcnogo).
    One figure per representation.
    """
    _ensure_out_dir()
    pkl_path = Path("multiple_tasks") / DMC_ANAME / f"dmcgo_fixed_points_{DMC_ANAME}.pkl"
    if not pkl_path.exists():
        print(f"  Skipped: {pkl_path} not found. Run multiple_task_analysis.py "
              f"shared_run first.")
        return

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    task_markers = ["s", "^"]                 # square = task0/Pro, triangle = task1/Anti
    task_styles = ["-", "--"]                 # solid = task0/Pro, dashed = task1/Anti

    def _plot_one(rep):
        if rep not in d:
            print(f"  Skipped {rep}: not in pickle.")
            return
        e = d[rep]
        if e.get("best_plane") is None or "fp_by_task" not in e:
            print(f"  Skipped {rep}: pickle predates the optimal-plane format. "
                  f"Re-run multiple_task_analysis.py shared_run.")
            return

        fp = np.asarray(e["fp_by_task"])          # (n_task, n_stim, n_pc)
        traj_by_task = [np.asarray(t) for t in e.get("traj_by_task", [])]
        stim_labels = e["stim_labels"]
        task_names = e.get("task_names", ["Pro", "Anti"])
        bx, by = e["best_plane"]
        sil = e.get("best_plane_silhouette", float("nan"))
        n_task, n_stim, _ = fp.shape
        have_traj = len(traj_by_task) == n_task

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
        for t_idx in range(n_task):
            for stim in range(n_stim):
                col = c_vals[stim_labels[stim] % len(c_vals)]
                if have_traj:
                    tr = traj_by_task[t_idx][stim]   # (delay_T, n_pc)
                    ax.plot(tr[:, bx], tr[:, by], color=col, alpha=0.5,
                            linewidth=0.8, linestyle=task_styles[t_idx % len(task_styles)],
                            zorder=2)
                ax.scatter(fp[t_idx, stim, bx], fp[t_idx, stim, by],
                           facecolor=col, edgecolor="black", linewidth=0.5,
                           marker=task_markers[t_idx % len(task_markers)],
                           s=22, alpha=0.7, zorder=3)
        ax.set_xlabel(f"PC{bx+1}", fontsize=8)
        ax.set_ylabel(f"PC{by+1}", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        # hidden / m_modulation span a wide range, so use sparser ticks there.
        nbins = 3 if rep in ("hidden", "m_modulation") else None
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=nbins))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=nbins))
        ax.tick_params(labelsize=7)

        # combined legend (stimulus colors + task markers), placed ON the axes
        stim_handles = [plt.Line2D([0], [0], marker="o", linestyle="None",
                                   markerfacecolor=c_vals[lab % len(c_vals)],
                                   markeredgecolor="black", markersize=5, label=f"stim {lab}")
                        for lab in stim_labels]
        task_handles = [plt.Line2D([0], [0], marker=task_markers[i],
                                   linestyle=task_styles[i], color="0.5",
                                   markerfacecolor="0.7", markeredgecolor="black",
                                   markersize=6, label=task_names[i])
                        for i in range(n_task)]
        ax.legend(handles=stim_handles + task_handles, frameon=True, fontsize=5,
                  ncol=2, loc="best")

        fig.tight_layout()
        out_path = OUT_DIR / f"dmc_memory_attractor_{rep}_{DMC_ANAME}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}  (PC{bx+1}-PC{by+1}, 2D sil={sil:.3f})")

    for rep in ("hidden", "m_modulation", "e_modulation"):
        _plot_one(rep)


# ─── Figures grouped by mode ──────────────────────────────────────────────────
# Each mode maps a figure name → its plotting function. Figures only depend on
# the data produced by their corresponding experiment, so a mode can be run in
# isolation without touching the others' inputs.
#
#   one_task         single-task training analyses (multiple_task single-task run)
#   multiple_tasks   the full multi-task network: clustering, lesion, state space
#   two_in_multiple  two-task probes within the multi-task net (DMC memory attractor)
#   pretraining      pretraining → post-training transfer analyses
#   two_task         the two-task network: cross-task / cross-period PCA
FIGURES_BY_MODE = {
    "one_task": {
        "onetask_example_trial": plot_onetask_example_trial,
        "onetask_show": plot_onetask_show,
        "onetask_modulation_heatmap": plot_onetask_modulation_heatmap,
        "onetask_long_fixed_points": plot_onetask_long_fixed_points,
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
    "two_in_multiple": {
        "dmc_memory_attractor": plot_dmc_memory_attractor,
    },
    "pretraining": {
        "transfer_speed": plot_transfer_speed,
        "learning_trajectory": plot_learning_trajectory,
        "rule_vectors": plot_rule_vectors,
        "aggregate_cve": plot_aggregate_cve,
        "aggregate_cve_stimulus": plot_aggregate_cve_stimulus,
    },
    "two_task": {
        "twotask_d_combine": plot_two_task_d_combine,
        "twotask_m_pca": plot_two_task_m_pca,
        "twotask_attractor_cycle": plot_two_task_attractor_cycle,
        "twotask_cancel": plot_two_task_cancel,
        "twotask_attractor_first": plot_two_task_attractor_first,
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
        nargs="*",
        choices=["all", *FIGURES_BY_MODE.keys()],
        help="Which group(s) of figures to generate. Accepts multiple modes "
             "(e.g. 'one_task two_task'). 'all' runs every mode (default when "
             "none given).",
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
    else:
        modes = args.mode or ["all"]
        if "all" in modes:
            figures = ALL_FIGURES
            modes_run = "all"
        else:
            # Union of the selected modes, de-duplicated, preserving order.
            figures = {}
            for m in modes:
                figures.update(FIGURES_BY_MODE[m])
            modes_run = "+".join(modes)

    # Each mode reads a different experiment; print the relevant name(s) so the
    # source run is unambiguous. Pretraining aggregates across seeds, so it has
    # no single identifier.
    mode_experiment = {
        "one_task": ONETASK_ANAME,
        "multiple_tasks": ANAME,
        "two_in_multiple": DMC_ANAME,
        "two_task": TWOTASK_ANAME,
        "pretraining": "(aggregated across seeds)",
    }
    if modes_run in ("all", "only"):
        printed_modes = list(FIGURES_BY_MODE.keys())
    else:
        printed_modes = modes
    print("Experiment(s):")
    for m in printed_modes:
        print(f"  {m}: {mode_experiment.get(m, '?')}")
    print(f"Output: {OUT_DIR}/")
    print(f"Mode: {modes_run} ({len(figures)} figure(s))")
    print()

    # Clear old figures before (re)generating. The output directory is wiped on
    # every run — including a single-mode or --only run — so stale outputs never
    # linger.
    _ensure_out_dir()
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
