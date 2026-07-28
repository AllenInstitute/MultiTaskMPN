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

# ─── Global figure options ────────────────────────────────────────────────────
# Master toggle for legends across every figure. Set False to suppress all
# legends (useful for panels where the legend is documented in the caption).
# All figures route their legend calls through _legend(), so this one flag
# controls them uniformly.
SHOW_LEGEND = True


def _legend(ax, *args, **kwargs):
    """Draw a legend on `ax` only if the global SHOW_LEGEND flag is set.

    Drop-in replacement for ax.legend(...); returns the Legend or None. Any
    per-figure gating (e.g. a local show_legend) should be checked by the caller
    before calling this, so both conditions must hold for a legend to appear.
    """
    if not SHOW_LEGEND:
        return None
    return ax.legend(*args, **kwargs)


# ─── Shared label / IO helpers ────────────────────────────────────────────────

def _wrap(text, width=16):
    """Insert line breaks into `text` so no line exceeds ~`width` characters.

    Wraps on word boundaries (never mid-word), so long axis labels / tick labels
    stack onto multiple lines instead of running off the panel or overlapping a
    neighbor. Accepts a single string or an iterable of strings (returns a list
    for the latter). Non-string items pass through unchanged."""
    import textwrap
    if isinstance(text, str):
        return "\n".join(textwrap.wrap(text, width=width)) or text
    return [_wrap(t, width) if isinstance(t, str) else t for t in text]


def _save_fig(fig, out_path, extra=""):
    """Save `fig` at the standard dpi / tight bbox, close it, and print a line.
    `extra` appends to the "Saved: {out_path}" message (e.g. counts/params).

    When legends are enabled (SHOW_LEGEND), an `_n` suffix is appended to the
    filename stem so the legended figure does not overwrite the no-legend one
    (e.g. `foo.png` → `foo_n.png`)."""
    out_path = Path(out_path)
    if SHOW_LEGEND:
        out_path = out_path.with_name(f"{out_path.stem}_n{out_path.suffix}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}{extra}")


def _load_pkl_or_skip(pkl_path, hint="", use_name=False):
    """Return the unpickled object at `pkl_path`, or None (with a "Skipped"
    message) if it does not exist. `hint` is appended to the message (e.g.
    "Run one_task_analysis.py first."); `use_name` prints only the filename."""
    if not pkl_path.exists():
        shown = pkl_path.name if use_name else pkl_path
        print(f"  Skipped: {shown} not found.{(' ' + hint) if hint else ''}")
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _load_twotask_glob_or_skip(pattern):
    """Load the first pickle matching `pattern` under the configured two-task run
    dir, or return None (with a "Skipped" message) if none match."""
    run_dir = TWOTASKS_DIR / TWOTASK_ANAME
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        print(f"  Skipped: no {pattern} in {run_dir}. "
              f"Run two_task_analysis.py first.")
        return None
    with open(matches[0], "rb") as f:
        return pickle.load(f)


# ─── Paths & run identifiers ──────────────────────────────────────────────────
# All figure output goes here; the run identifiers ("aname") below select which
# trained run each mode's figures are drawn from. They are grouped by experiment
# family so a run can be swapped in one place.
OUT_DIR = Path("paper_plot")

# ── Multi-task (one full multi-task network) ──
ANAME = "everything_seed749_L21e4+hidden300+batch128+angle"
DATA_DIR = Path("multiple_tasks") / ANAME
# DMC category-memory probe (two_in_multiple mode). May differ from ANAME — set
# independently so the attractor figure can come from a different
# seed/regularization than the clustering/lesion figures.
DMC_ANAME = "everything_seed299_L21e3+hidden300+batch128+angle"
# delayDM integration-memory probe (two_in_multiple mode). Independent of
# ANAME/DMC_ANAME; matching data written by multiple_task_analysis.py's
# shared_run("delaydm1") into
# multiple_tasks/{DELAYDM_ANAME}/delaydm1_fixed_points_{DELAYDM_ANAME}.pkl.
DELAYDM_ANAME = "everything_seed408_L21e4+hidden300+batch128+angle"

# ── Two-task network ──
TWOTASKS_DIR = Path("twotasks")
# Cross-task / cross-period PCA figure (d_combine); data written by
# two_task_analysis.py into twotasks/{TWOTASK_ANAME}/d_combine_{TWOTASK_ANAME}.pkl.
TWOTASK_ANAME = "delaygofamily_seed21_reg1e3+hidden200"
# Attractor first-subplot figure (independent of TWOTASK_ANAME so it can come
# from a different seed/regularization).
TWOTASK_ATTRACTOR_ANAME = "delaygofamily_seed21_reg1e3+hidden200"

# ── Single-task network ──
ONETASK_DIR = Path("onetask")
# Default single-task run for the one-task figures (aname under onetask/{aname}/).
# Used by onetask_show, onetask_modulation_snapshot, onetask_long_fixed_points, etc.
ONETASK_ANAME = "delaygo_seed395_hidden200+batch128+angle"
# Runs used for the example-trial illustration. Set to ONETASK_ANAME so the
# input/output illustration comes from the SAME network as onetask_show; can be
# pointed at a different seed here if desired (they read separate example_trial
# pickles).
ONETASK_INPUT_ANAME = ONETASK_ANAME
ONETASK_OUTPUT_ANAME = ONETASK_ANAME


def _twotask_seed_tag():
    """Seed substring of TWOTASK_ANAME (e.g. 'seed894'), for figure filenames.
    Falls back to the full aname if no 'seed<N>' token is present."""
    import re as _re
    m = _re.search(r"seed\d+", TWOTASK_ANAME)
    return m.group(0) if m else TWOTASK_ANAME


def _read_twotask_n_stim(default=8):
    """Trained ring-direction count (n_eachring) for the configured two-task run,
    read from its saved param json; used only for the dashed trained-direction
    guide lines in the interp / stability figures. Falls back to `default`."""
    try:
        import json as _json
        p = TWOTASKS_DIR / TWOTASK_ANAME / f"param_{TWOTASK_ANAME}_param.json"
        if p.exists():
            cfg = _json.load(open(p))
            return int(cfg.get("task_params", {}).get("n_eachring", default))
    except Exception:
        pass
    return default


def _twotask_grad_fp_paths():
    """(rule, path) for every gradient fixed-point pickle of the configured
    two-task run — one per task rule, written by two_task_analysis.py as
    twotasks/{aname}/fixed_points_grad_{aname}_{rule}.pkl. Sorted by rule name."""
    run_dir = TWOTASKS_DIR / TWOTASK_ANAME
    prefix = f"fixed_points_grad_{TWOTASK_ANAME}_"
    out = []
    for p in sorted(run_dir.glob(f"{prefix}*.pkl")):
        rule = p.name[len(prefix):-len(".pkl")]
        out.append((rule, p))
    return out


TWOTASK_N_STIM = _read_twotask_n_stim()

# Categorical color cycle (matches multiple_task_analysis.py). Used for
# NON-stimulus categorical coloring (components, series, periods, tasks).
c_vals = [
    "#e53e3e", "#3182ce", "#38a169", "#d69e2e", "#d53f8c",
    "#4c51bf", "#dd6b20", "#0ea5e9", "#22c55e", "#a855f7",
    "#f43f5e", "#0f766e", "#b83280", "#ca8a04", "#2b6cb0",
] * 10

# Number of TRAINED ring stimulus directions (n_eachring in the task config) for
# the configured one-task run. Read from that run's saved param json so figures
# adapt to each experiment (e.g. the default 8 vs a `morestimulus` run's 1024),
# falling back to 8 if the json is unavailable. NB: this is the *trained*
# direction count — the dense fixed-point interpolation grid (n_interp) is
# separate and read per-figure from the pickle's own `stim`/`angles`.
def _read_onetask_n_stim(default=8):
    try:
        import json as _json
        p = ONETASK_DIR / f"param_{ONETASK_ANAME}_param.json"
        if p.exists():
            cfg = _json.load(open(p))
            return int(cfg.get("task_params", {}).get("n_eachring", default))
    except Exception:
        pass
    return default


ONETASK_N_STIM = _read_onetask_n_stim()

# ─── Stimulus color scheme ────────────────────────────────────────────────────
# A continuous rainbow ramp from red to purple, used ONLY to color by stimulus
# direction (ring index). Stimulus k of N maps to a hue sweeping from red
# (hue 0) through the spectrum to purple, so adjacent stimuli are adjacent
# colors and the ring reads as a smooth gradient. Use stim_color(k, n).


def stim_color(k, n=ONETASK_N_STIM):
    """Color for stimulus index k of n, on a red→purple rainbow ramp."""
    n = max(int(n), 1)
    # Sweep hue from 0 (red) to ~0.83 (purple/violet) across the n stimuli.
    frac = (k % n) / max(n - 1, 1)
    hue = 0.83 * frac
    return mpl.colors.hsv_to_rgb((hue, 0.85, 0.9))


def stim_colors(n=ONETASK_N_STIM):
    """List of n stimulus colors on the red→purple rainbow ramp."""
    return [stim_color(k, n) for k in range(n)]


def _shade(color, frac):
    """Shade `color` by `frac` in [-1, 1]: frac<0 darkens toward BLACK (frac=-1
    → black), frac=0 is the original, frac>0 lightens toward WHITE (frac=+1 →
    white). Used to shade a trajectory dark→bright along a sweep."""
    r, g, b = mpl.colors.to_rgb(color)
    if frac >= 0:
        return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)
    f = 1.0 + frac                       # frac in [-1,0] -> multiplier in [0,1]
    return (r * f, g * f, b * f)


def _fixed_point_mask(entry, n):
    """Boolean (n,) mask of which gradient fixed points converged.

    Reads the `is_fixed` array saved by one_task_analysis.py (relative-step <=
    rel_tol). Older pickles lack it — treat every point as converged so figures
    from those still render unchanged."""
    mask = entry.get("is_fixed")
    if mask is None:
        return np.ones(int(n), dtype=bool)
    return np.asarray(mask, dtype=bool)


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

# Period colorbar palette for the one-task / two-task period strip, ordered
# Fixation → Stimulus → Memory → Response. Stimulus/Memory/Response reuse the
# multi-task heatmap phase colors (_PHASE_COLORS stim1/delay1/go1) so the period
# bar is color-consistent with those figures; Fixation has no heatmap
# counterpart, so it gets a new pastel yellow in the same soft-pastel family.
_ONETASK_PERIOD_COLORS = [
    "#fef08a",                  # Fixation — new (pale yellow)
    _PHASE_COLORS["stim1"],     # Stimulus — matches heatmap Stimulus 1 (purple)
    _PHASE_COLORS["delay1"],    # Memory   — matches heatmap Memory 1 (light green)
    _PHASE_COLORS["go1"],       # Response — matches heatmap Response (light gray)
]

# ─── Input / output channel colors ────────────────────────────────────────────
# Colors for the example-trial input and output traces. A muted qualitative set,
# deliberately distinct from the vivid stimulus rainbow (stim_color) and the pale
# period-bar pastels (_ONETASK_PERIOD_COLORS). Within a modality the cos/sin
# channels share a hue as a (dark, light) pair. Fixation↔Fixation shares a color
# across the input and output figures. The response Cos/Sin get their OWN hue
# (purple), deliberately distinct from the stimulus modalities so the readout is
# not confused with an input modality.
_IO_FIXATION = "#555555"              # dark gray
_IO_MOD1 = ("#a6761d", "#dcb877")     # brown  (cos dark, sin light)
_IO_MOD2 = ("#1b9e77", "#8fded0")     # teal   (cos dark, sin light) = active stimulus
_IO_TASK = "#d95f02"                  # orange (single channel)
_IO_RESPONSE = ("#7e3ff2", "#c4a3f5")  # purple (cos dark, sin light) = readout


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


def _add_period_strip(ax, spans, xmax, height=0.05, pad=0.02):
    """Add a thin colored strip above `ax` marking trial periods, matching the
    cluster-strip style used in the multi-task heatmaps (colors only, no text).

    `spans` is a list of (start, end, color, ...) tuples in the parent axis's
    data-x coordinates; a trailing `end` of None runs to `xmax`. The strip is an
    inset axis placed just above the parent, spanning its full x-range so the
    period boundaries line up with the traces below.
    """
    strip = ax.inset_axes([0, 1.0 + pad, 1, height], transform=ax.transAxes)
    for span in spans:
        start, end, color = span[0], span[1], span[2]
        end = xmax if end is None else min(end, xmax)
        strip.axvspan(start, end, color=color, lw=0)
    strip.set_xlim(0, xmax)
    strip.set_ylim(0, 1)
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
    _save_fig(fig, out_path)


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
    _save_fig(fig, out_path)


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
    _save_fig(fig, out_path)


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
    _save_fig(fig, out_path)


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
        _legend(ax, frameon=True, loc="best", fontsize=5, markerscale=1.0)
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
            _legend(ax, frameon=True, loc="best", fontsize=5, markerscale=1.0)

    axes[1].set_xlabel("PC1", fontsize=8)
    axes[0].tick_params(labelbottom=False)

    fig.tight_layout()
    out_path = OUT_DIR / "state_space_combined.png"
    _save_fig(fig, out_path)


RVAL_RESULT_PATH = STATE_SPACE_DIR / "initial_condition_distance_vs_angle_results.pkl"


def plot_state_space_r_values():
    """Figure: Mean R-values (initial-condition distance vs trajectory angle) for hidden & eff_mod."""
    _ensure_out_dir()
    result_dict = _load_pkl_or_skip(RVAL_RESULT_PATH, "Run state_space_shift.py first.")
    if result_dict is None:
        return

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
    _save_fig(fig, out_path)


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
    _save_fig(fig, out_path, extra=f"  (n={n_experiments} experiments)")


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
    _save_fig(fig, out_path)


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
    _legend(ax1, [f"r = {r:.2f}, {p_str}"], loc="upper right", fontsize=7, frameon=True)
    ax1.set_xlabel("Over-membership", fontsize=8)
    ax1.set_ylabel("|Lesion effect diff|", fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()
    out_path1 = OUT_DIR / "om_vs_lesion_scatter.png"
    _save_fig(fig1, out_path1)

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
    _legend(ax2, [f"r = {r_p:.2f}, {p_str_p}"], loc="upper left", fontsize=7, frameon=True)
    ax2.set_xlabel("OM-predicted effect (%)", fontsize=8)
    ax2.set_ylabel("Actual mod lesion effect (%)", fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    out_path2 = OUT_DIR / "om_vs_lesion_prediction.png"
    _save_fig(fig2, out_path2)


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
            color = stim_color(stim_labels[stim], n_stim)
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
            _legend(ax, frameon=True, fontsize=6)

    fig.tight_layout()
    out_path = OUT_DIR / f"fixed_points_{addtask}_{plot_name}.png"
    _save_fig(fig, out_path)


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
    _save_fig(fig, out_path)


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
                _legend(ax, [f"r = {r:.2f}, {p_str}"], loc="upper right",
                          fontsize=7, frameon=True)

            ax.set_xlabel("Tuning cosine similarity", fontsize=8)
            ax.set_ylabel("Lesion effect L1 distance", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)

            fig.tight_layout()
            # e.g. "input_normalized_k20" -> "input_norm"
            clean_name = name.replace("_normalized", "_norm").replace("_unnormalized", "_unnorm").replace("_k20", "")
            out_path = OUT_DIR / f"cluster_corr_vs_lesion_{clean_name}.png"
            _save_fig(fig, out_path)


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
    _legend(ax, fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "transfer_speed.png"
    _save_fig(fig, out_path)


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
    _legend(ax, fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "learning_trajectory.png"
    _save_fig(fig, out_path)


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
    _legend(ax, fontsize=7, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "rule_vectors.png"
    _save_fig(fig, out_path)


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
        _legend(ax, fontsize=7, frameon=True)


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
    _save_fig(fig, out_path)


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
    _save_fig(fig, out_path)


# ─── One-task figures ─────────────────────────────────────────────────────────
# (Run identifiers ONETASK_ANAME / ONETASK_INPUT_ANAME / ONETASK_OUTPUT_ANAME and
# ONETASK_DIR are defined in the Paths & run identifiers block at the top.)


def plot_onetask_example_trial():
    """
    Figures: one representative single-task trial, saved as TWO files:
      onetask_example_trial_input.png  — 4 vertically-stacked input subplots:
        Fixation, Modality 1 (cos+sin), Modality 2 (cos+sin), Task cue.
        Read from ONETASK_INPUT_ANAME.
      onetask_example_trial_output.png — network vs target output.
        Read from ONETASK_OUTPUT_ANAME (may be a different seed than the input).
    Reloaded from the pickles saved by one_task_analysis.py. Y-ticks are just
    [-1, 1] on every panel.

    Input channel layout (low_dim, no fixate_off): 0=Fixation, 1-2=Modality 1
    (cos,sin), 3-4=Modality 2 (cos,sin), 5=Task cue.
    """
    _ensure_out_dir()

    def _load_example(aname):
        """Load an example-trial pickle for the given run, or None if missing."""
        p = ONETASK_DIR / aname / f"example_trial_{aname}.pkl"
        if not p.exists():
            print(f"  Skipped: {p} not found. Run one_task_analysis.py first.")
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def _period_spans(d):
        """Build the (start, end, color) period bar spans from a trial dict."""
        stimulus_start = d.get("stimulus_start")
        stimulus_end = d.get("stimulus_end")
        response_start = d.get("response_start")
        if stimulus_start is None or stimulus_end is None or response_start is None:
            return []
        fix_c, stim_c, mem_c, resp_c = _ONETASK_PERIOD_COLORS
        return [
            (0, stimulus_start, fix_c),
            (stimulus_start, stimulus_end, stim_c),
            (stimulus_end, response_start, mem_c),
            (response_start, None, resp_c),
        ]

    def _style(ax, ylabel, last_row, T):
        ax.set_xlim(0, T - 1)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1, 1])          # only -1 and 1, as requested
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        # Thin dashed zero baseline behind the traces.
        ax.axhline(0, color="0.6", lw=0.6, linestyle="--", zorder=1)
        # x tick labels only on the bottom subplot. NB: with sharex=True, calling
        # set_xticklabels([]) on a non-last axis blanks the shared tick text for
        # the bottom row too, so toggle visibility via tick_params instead.
        ax.tick_params(axis="x", labelbottom=last_row)
        if last_row:
            ax.set_xlabel("Time (ms)", fontsize=10)

    # ── Input figure: 4 stacked subplots ────────────────────────────────────
    d = _load_example(ONETASK_INPUT_ANAME)
    if d is None:
        return
    inp = np.asarray(d["input"])              # (T, n_input)
    T = inp.shape[0]
    period_spans = _period_spans(d)

    # (channel indices, per-channel colors+labels, panel ylabel). Colors: cos/sin
    # of a modality share a hue (dark/light); Fixation and Modality 2 (the active
    # stimulus) colors are reused in the output figure for the matching channels.
    n_in = inp.shape[1]
    input_groups = [
        ([0], [_IO_FIXATION], ["Fixation"], "Fixation"),
        ([1, 2], [_IO_MOD1[0], _IO_MOD1[1]], ["Mod1 cos", "Mod1 sin"], "Stimulus\nModality 1"),
        ([3, 4], [_IO_MOD2[0], _IO_MOD2[1]], ["Mod2 cos", "Mod2 sin"], "Stimulus\nModality 2"),
        ([5], [_IO_TASK], ["Task cue"], "Rule"),
    ]
    # keep only groups whose channels exist in this input
    input_groups = [(chs, cols, labs, ylab) for chs, cols, labs, ylab in input_groups
                    if all(ch < n_in for ch in chs)]

    figin, axin = plt.subplots(len(input_groups), 1, figsize=(3.4, 1.0 * len(input_groups)),
                               sharex=True, squeeze=False)
    for row, (chs, cols, labs, ylab) in enumerate(input_groups):
        ax = axin[row, 0]
        for ch, col, lab in zip(chs, cols, labs):
            ax.plot(inp[:, ch], color=col, label=lab, zorder=2)
        _style(ax, ylab, last_row=(row == len(input_groups) - 1), T=T)
        _legend(ax, fontsize=6, frameon=True, loc="upper right", ncol=len(chs))
    # Period colorbar above the top subplot (colors only, no shading behind traces).
    if period_spans:
        _add_period_strip(axin[0, 0], period_spans, xmax=T - 1)
    figin.tight_layout()
    out_in = OUT_DIR / "onetask_example_trial_input.png"
    _save_fig(figin, out_in)

    # ── Output figure: one stacked subplot per output channel ────────────────
    # Loaded from ONETASK_OUTPUT_ANAME (independent of the input run), with its
    # own trial timing for the period bar.
    d_out = _load_example(ONETASK_OUTPUT_ANAME)
    if d_out is None:
        return
    net_out = np.asarray(d_out["net_output"]).copy()   # (T, n_output)
    target = np.asarray(d_out["target_output"])        # (T, n_output)
    out_labels = d_out["output_labels"]
    T_out = net_out.shape[0]
    period_spans_out = _period_spans(d_out)

    # Illustrative transient error: for the first couple of timesteps of the
    # RESPONSE period, add an offset to the plotted Output Sin channel (channel 2)
    # that DECREASES to zero over those steps — so the sin readout starts slightly
    # off target at response onset and converges. Display-only; does not touch the
    # saved data or the target trace.
    ONETASK_OUT_ERR_STEPS = 2            # number of response steps to perturb
    ONETASK_OUT_ERR_MAG = 0.5            # initial additive error on Output Sin
    r0 = d_out.get("response_start")
    if r0 is not None and net_out.shape[-1] >= 3 and ONETASK_OUT_ERR_STEPS > 0:
        r0 = int(r0)
        sin_i = 2
        # Decaying weights 1 → 0 over the perturbed steps (linear ramp-down).
        weights = np.linspace(1.0, 0.0, ONETASK_OUT_ERR_STEPS, endpoint=False)
        for k in range(ONETASK_OUT_ERR_STEPS):
            t = r0 + k
            if t >= T_out:
                break
            net_out[t, sin_i] = net_out[t, sin_i] + ONETASK_OUT_ERR_MAG * weights[k]

    # Output channels are [Fixation, Output Cos, Output Sin]. Fixation shares the
    # input figure's fixation color; the response Cos/Sin get their OWN purple hue
    # (_IO_RESPONSE), distinct from the stimulus modalities so the readout isn't
    # confused with an input modality. Extra channels (if any) fall back to brown.
    out_colors = [_IO_FIXATION, _IO_RESPONSE[0], _IO_RESPONSE[1],
                  _IO_MOD1[0], _IO_MOD1[1]]
    # Panel y-labels by output-channel meaning: [Fixation, Response cosθ,
    # Response sinθ] on two lines; fall back to the pickle's own labels for any
    # extra channel. Mathtext $\cos\theta$ keeps cos/sin tight against θ.
    out_ylabels = ["Fixation", "Response\n" + r"$\cos\theta$",
                   "Response\n" + r"$\sin\theta$"]

    def _lighten(color, frac=0.55):
        """Blend a color toward white by `frac` (for the faded target shadow)."""
        r, g, b = mpl.colors.to_rgb(color)
        return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)

    figout, axout = plt.subplots(net_out.shape[-1], 1,
                                 figsize=(3.4, 1.0 * net_out.shape[-1]),
                                 sharex=True, squeeze=False)
    for out_idx in range(net_out.shape[-1]):
        ax = axout[out_idx, 0]
        lab = out_labels[out_idx] if out_idx < len(out_labels) else f"out {out_idx}"
        ylab = (out_ylabels[out_idx] if out_idx < len(out_ylabels)
                else lab)
        col = out_colors[out_idx % len(out_colors)]
        ax.plot(target[:, out_idx], color=_lighten(col),
                linewidth=4, alpha=0.7, zorder=2, label="target")
        ax.plot(net_out[:, out_idx], color=col,
                zorder=3, label=lab)
        _style(ax, ylab, last_row=(out_idx == net_out.shape[-1] - 1), T=T_out)
        _legend(ax, fontsize=6, frameon=True, loc="upper right", ncol=2)
    # Period colorbar above the top subplot (colors only, no shading behind traces).
    if period_spans_out:
        _add_period_strip(axout[0, 0], period_spans_out, xmax=T_out - 1)
    figout.tight_layout()
    out_out = OUT_DIR / "onetask_example_trial_output.png"
    _save_fig(figout, out_out)


def plot_onetask_stimulus_colorwheel():
    """
    Illustration: the stimulus color convention used throughout the one-task
    figures. Each of the N ring stimulus directions (angle = 2*pi*k/N) is drawn
    as a dot on the unit circle in its c_vals[k] color — the same mapping used
    to color trajectories/rings by stimulus. A pure legend/illustration; uses no
    saved data.
    """
    _ensure_out_dir()
    n = ONETASK_N_STIM

    fig, ax = plt.subplots(1, 1, figsize=(2.6, 2.6))
    # Faint guide circle.
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "-", color="0.8", lw=1.0, zorder=1)
    # Draw one dot per trained direction, but cap the number of drawn dots for
    # dense (`morestimulus`) runs so the wheel stays legible; colors still span
    # the full ring (0..n-1) so the ramp matches the trajectories.
    n_draw = n if n <= 24 else 24
    draw_k = np.linspace(0, n, n_draw, endpoint=False).astype(int)
    dot_size = 180 if n <= 24 else 60
    for k in draw_k:
        a = 2 * np.pi * k / n
        ax.scatter(np.cos(a), np.sin(a), color=stim_color(int(k), n), s=dot_size,
                   edgecolors="k", linewidths=0.5, zorder=3)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    out_path = OUT_DIR / "onetask_stimulus_colorwheel.png"
    _save_fig(fig, out_path)


def plot_onetask_show():
    """
    Figure: per-stimulus fixon / task / combine modulation-component traces
    (the single-task "cancellation" figure), reloaded from the pickle saved by
    one_task_analysis.py. Shows how the fixon and task contributions cancel
    until the response period.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"show_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return

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
    # the saved break times, drawn as a top colorbar (colors only, no shading).
    period_spans = []
    if stimulus_start is not None and stimulus_end is not None and response_start is not None:
        fix_c, stim_c, mem_c, resp_c = _ONETASK_PERIOD_COLORS
        period_spans = [
            (0, stimulus_start, fix_c, "Fixation"),
            (stimulus_start, stimulus_end, stim_c, "Stimulus"),
            (stimulus_end, response_start, mem_c, "Memory"),
            (response_start, None, resp_c, "Response"),
        ]

    fig, axes = plt.subplots(len(sel_labels), 1, figsize=(3.4, 1.8 * len(sel_labels)),
                             squeeze=False)
    for i, lab in enumerate(sel_labels):
        ax = axes[i, 0]
        tr = per_stim[lab]
        T = len(tr["combine"])
        # Colors + names matched to onetask_example_trial's input channels:
        # Fixation → _IO_FIXATION (dark gray), Rule → _IO_TASK (orange).
        # Combine uses a distinct color not used for either input channel.
        ax.plot(tr["fixon"], color=_IO_FIXATION, label="Fixation", zorder=2)
        ax.plot(tr["task"], color=_IO_TASK, label="Rule", zorder=2)
        ax.plot(tr["combine"], color=c_vals[4], linewidth=2.5, label="Combine", zorder=3)
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        ax.set_xlim(0, T - 1)
        ax.set_ylim([-2.0, 2.0])
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            _legend(ax, frameon=True, fontsize=6, loc="best")
            # Period colorbar above the top panel only.
            if period_spans:
                _add_period_strip(ax, period_spans, xmax=T - 1)
        if i == len(sel_labels) - 1:
            ax.set_xlabel("Time (ms)", fontsize=9)
        else:
            ax.set_xticklabels([])

    # Shared y-label centered across the panels, nudged rightward (larger x) so
    # it sits close to the axes rather than at the far figure edge.
    fig.supylabel("Readout projection", fontsize=9, x=0.06)

    fig.tight_layout()
    out_path = OUT_DIR / "onetask_show.png"
    _save_fig(fig, out_path)


def plot_onetask_modulation_snapshot():
    """
    Figure: full plasticity matrix M (hidden x input) at a single timepoint,
    laid out as a 2x2 grid — rows = stimulus (labels 1 and 5), columns =
    trial period (middle of the stimulus period / middle of the response
    period). Each cell is a hidden x input heatmap on a shared symmetric color
    scale. Reloaded from the pickle saved by one_task_analysis.py.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"modulation_snapshot_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return

    stims = d["stims"]
    snapshots = d["snapshots"]
    if len(stims) < 2:
        print(f"  Skipped: need 2 stimuli, only {stims} available.")
        return

    periods = [("stimulus", "Mid stimulus"), ("response", "Mid response")]

    # Shared symmetric color limit across all four cells. Clip to the 99th
    # percentile of |M| so a few extreme entries don't wash out the structure.
    all_vals = np.concatenate([
        np.abs(np.asarray(snapshots[s][pkey]).ravel())
        for s in stims for pkey, _ in periods
    ])
    vmax = float(np.percentile(all_vals, 99))

    fig, axes = plt.subplots(2, 2, figsize=(6, 6),
                             gridspec_kw={"wspace": 0.12, "hspace": 0.18})
    im = None
    for r, s in enumerate(stims):
        for c, (pkey, ptitle) in enumerate(periods):
            ax = axes[r, c]
            mat = np.asarray(snapshots[s][pkey], dtype=float)   # (hidden, input)
            im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.6)
                sp.set_edgecolor("0.4")
            if r == 0:
                ax.set_title(ptitle, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"Stimulus {s}\nHidden", fontsize=9)
            if r == len(stims) - 1:
                ax.set_xlabel("Input", fontsize=9)

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.04)
    cbar.set_label("Modulation (M)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    out_path = OUT_DIR / "onetask_modulation_snapshot.png"
    _save_fig(fig, out_path)


# Which single stimulus to show in plot_onetask_modulation_snapshot_single
# (must be one of the labels saved in the snapshot pickle, e.g. 1 or 5).
ONETASK_SNAPSHOT_SINGLE_STIM = 1


def _onetask_hcbar(vmax, out_name):
    """Save a standalone short/wide horizontal bwr colorbar (ticks only, no
    label) spanning [-vmax, vmax], to OUT_DIR/out_name."""
    figc = plt.figure(figsize=(1.5, 0.45))
    axc = figc.add_axes([0.05, 0.5, 0.9, 0.35])   # [left, bottom, width, height]
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap="bwr", norm=norm)
    sm.set_array([])
    cbar = figc.colorbar(sm, cax=axc, orientation="horizontal")
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels([f"{-vmax:.2g}", "0", f"{vmax:.2g}"])
    cbar.ax.tick_params(labelsize=8)
    cbar_path = OUT_DIR / out_name
    _save_fig(figc, cbar_path)


def _plot_onetask_snapshot_single(mat, stim, title, out_name, cbar_out_name,
                                  cbar_label):
    """Render a single-stimulus mid-response snapshot matrix as a square bwr
    heatmap (no colorbar), plus a SEPARATE small figure holding just a
    horizontal colorbar. `mat` is (hidden, input); color scale is symmetric,
    clipped to the 99th percentile of |mat|."""
    vmax = float(np.percentile(np.abs(mat).ravel(), 99))

    # Smaller panel so the axis labels/title read relatively larger.
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.2))
    ax.imshow(mat, cmap="bwr", vmin=-vmax, vmax=vmax,
              aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_edgecolor("0.4")
    # No title (was `title`); kept off per figure spec.
    ax.set_xlabel("Input", fontsize=12)
    ax.set_ylabel("Hidden", fontsize=12)
    fig.tight_layout()
    out_path = OUT_DIR / out_name
    _save_fig(fig, out_path)

    _onetask_hcbar(vmax, cbar_out_name)


def _plot_onetask_hidden_single(hidden_vec, stim, title, out_name, cbar_out_name):
    """Render the hidden-state vector at a single step as a thin vertical bwr
    strip (its own figure), plus a SEPARATE horizontal colorbar. Symmetric color
    scale clipped to the 99th percentile of |hidden|."""
    hv = np.asarray(hidden_vec, dtype=float).reshape(-1, 1)   # (hidden, 1)
    vmax = float(np.percentile(np.abs(hv).ravel(), 99)) or 1.0

    # Smaller panel so the axis label/title read relatively larger.
    fig, ax = plt.subplots(1, 1, figsize=(0.55, 2.2))
    ax.imshow(hv, cmap="bwr", vmin=-vmax, vmax=vmax,
              aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_edgecolor("0.4")
    # No title (was `title`); kept off per figure spec.
    ax.set_ylabel("Hidden", fontsize=12)
    fig.tight_layout()
    out_path = OUT_DIR / out_name
    _save_fig(fig, out_path)

    _onetask_hcbar(vmax, cbar_out_name)


def plot_onetask_modulation_snapshot_single():
    """
    Figures (each for a SINGLE stimulus, ONETASK_SNAPSHOT_SINGLE_STIM, at the
    middle of the response period), all as bwr heatmaps with a separate
    horizontal colorbar each:
      onetask_modulation_snapshot_single   — raw plasticity matrix M (hidden×input)
      onetask_emodulation_snapshot_single  — effective modulation W⊙M
      onetask_hidden_snapshot_single       — hidden-state vector (its own colorbar)
    Reuses the snapshot pickle saved by one_task_analysis.py.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"modulation_snapshot_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return

    stim = ONETASK_SNAPSHOT_SINGLE_STIM

    # Raw modulation M.
    if stim in d.get("snapshots", {}):
        _plot_onetask_snapshot_single(
            np.asarray(d["snapshots"][stim]["response"], dtype=float),
            stim,
            title=f"Stimulus {stim} — Mid response",
            out_name="onetask_modulation_snapshot_single.png",
            cbar_out_name="onetask_modulation_snapshot_single_cbar.png",
            cbar_label="Modulation (M)",
        )
    else:
        print(f"  Skipped raw M: stimulus {stim} not in snapshot pickle "
              f"(available: {d.get('stims')}).")

    # Effective modulation W⊙M (only if the analysis saved it).
    if stim in d.get("snapshots_eff", {}):
        _plot_onetask_snapshot_single(
            np.asarray(d["snapshots_eff"][stim]["response"], dtype=float),
            stim,
            title=f"Stimulus {stim} — Mid response",
            out_name="onetask_emodulation_snapshot_single.png",
            cbar_out_name="onetask_emodulation_snapshot_single_cbar.png",
            cbar_label="Effective modulation (W⊙M)",
        )
    else:
        print("  Skipped W⊙M: 'snapshots_eff' not in pickle "
              "(re-run one_task_analysis.py to add it).")

    # Hidden state at the same (mid-response) step — its own figure + colorbar.
    if stim in d.get("hidden_snapshots", {}):
        _plot_onetask_hidden_single(
            np.asarray(d["hidden_snapshots"][stim]["response"], dtype=float),
            stim,
            title=f"Stimulus {stim} — Mid response",
            out_name="onetask_hidden_snapshot_single.png",
            cbar_out_name="onetask_hidden_snapshot_single_cbar.png",
        )
    else:
        print("  Skipped hidden: 'hidden_snapshots' not in pickle "
              "(re-run one_task_analysis.py to add it).")


# Marker cycle for the full-trial PCA phases (matches one_task_analysis.py's
# markers_vals so the replotted figure uses the same per-phase markers).
_ONETASK_MARKERS = ['o', 'v', '*', '+', '>', '1', '2', '3', '4', 's',
                    'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']


def _plot_onetask_fulltrial_panel(d, ylabel, show_legend):
    """Render a single-panel full-trial PCA figure (PC1-PC2 only) from a
    fulltrial pickle dict written by one_task_analysis.py. Mirrors the original
    two-task style: per-trial trajectory line + per-phase markers + large solid
    transition markers at each period boundary. Returns the figure."""
    proj = np.asarray(d["lowd"])              # (batch, T, n_pc)
    labels = np.asarray(d["labels"]).reshape(-1)
    phases = [(n, int(t0), int(t1), int(mk)) for (n, t0, t1, mk) in d["phases"]]
    a, bb = 0, 1  # PC1-PC2 plane only

    # Transition timepoints = each non-fixation phase's start (its marker index).
    stimulus_start = int(d["stimulus_start"])
    stimulus_end = int(d["stimulus_end"])
    response_start = int(d["response_start"])
    # Marker index for a boundary = marker of the phase that starts at it.
    start_to_mk = {t0: mk for (_n, t0, _t1, mk) in phases}
    transition_ts = [(t, start_to_mk.get(t, 0))
                     for t in (stimulus_start, stimulus_end, response_start)]

    legend_handles = [
        plt.Line2D([0], [0], marker=_ONETASK_MARKERS[mk], linestyle="None",
                   markersize=10, markerfacecolor="k", markeredgecolor="k", label=name)
        for name, _t0, _t1, mk in phases
    ]

    n_stim = int(np.max(labels)) + 1 if len(labels) else ONETASK_N_STIM
    figd, ax = plt.subplots(1, 1, figsize=(5, 5))
    for i in range(proj.shape[0]):
        db = proj[i, :, :]
        color = stim_color(labels[i], n_stim)
        ax.plot(db[:, a], db[:, bb], c=color, alpha=0.25, zorder=2)
        for _name, t0, t1, mk in phases:
            sl = slice(t0, t1)
            ax.scatter(db[sl, a], db[sl, bb], color=color,
                       marker=_ONETASK_MARKERS[mk], alpha=0.5, zorder=3)
        for t, mk in transition_ts:
            tt = min(max(t - 1, 0), db.shape[0] - 1)
            ax.scatter([db[tt, a]], [db[tt, bb]], color=color,
                       marker=_ONETASK_MARKERS[mk], alpha=0.8, s=60,
                       linewidths=0.6, zorder=10)
    ax.set_xlabel(f"PC{a+1}", fontsize=18)
    ax.set_ylabel(f"PC{bb+1}", fontsize=18)
    ax.set_title(ylabel, fontsize=15)
    # Hide tick marks and numeric labels (PC axes are unitless here).
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        _legend(ax, handles=legend_handles, loc="upper right",
                  frameon=True, fontsize=10)
    figd.tight_layout()
    return figd


def _save_onetask_fulltrial_legend(d, out_name):
    """Save a STANDALONE per-phase legend figure (the same handles the fulltrial
    panels would draw), so the trajectory panels can stay legend-free."""
    phases = [(n, int(t0), int(t1), int(mk)) for (n, t0, t1, mk) in d["phases"]]
    handles = [
        plt.Line2D([0], [0], marker=_ONETASK_MARKERS[mk], linestyle="None",
                   markersize=10, markerfacecolor="k", markeredgecolor="k", label=name)
        for name, _t0, _t1, mk in phases
    ]
    figL = plt.figure(figsize=(1.6, 0.3 * max(len(handles), 1)))
    figL.legend(handles=handles, loc="center", frameon=True, fontsize=10)
    _save_fig(figL, OUT_DIR / out_name)


def plot_onetask_pca_fulltrial():
    """
    Figures: full-trial PCA trajectories (PC1-PC2 only) for the single-task
    network, replotted from the pickles written by one_task_analysis.py — one
    figure each for hidden activity and effective modulation:
      onetask_h_pca_fulltrial.png     — hidden activity, whole-trial PCA basis
      onetask_e_mod_pca_fulltrial.png — effective modulation W⊙M, whole-trial PCA
    Each is a single PC1-PC2 panel, colored by stimulus, with per-phase markers
    and period-boundary transition markers. The per-phase legend is emitted as a
    SEPARATE figure (onetask_pca_fulltrial_legend.png), not drawn on the panels.
    """
    _ensure_out_dir()
    specs = [
        ("h", "Hidden activity", "onetask_h_pca_fulltrial.png"),
        ("e_mod", "Effective modulation", "onetask_e_mod_pca_fulltrial.png"),
    ]
    legend_saved = False
    for tag, ylabel, out_name in specs:
        pkl_path = ONETASK_DIR / ONETASK_ANAME / f"{tag}_pca_fulltrial_{ONETASK_ANAME}.pkl"
        if not pkl_path.exists():
            print(f"  Skipped: {pkl_path} not found. Run one_task_analysis.py first.")
            continue
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        # Panels are drawn WITHOUT an inset legend.
        fig = _plot_onetask_fulltrial_panel(d, ylabel, show_legend=False)
        out_path = OUT_DIR / out_name
        _save_fig(fig, out_path)
        # Emit the shared per-phase legend once, as its own figure.
        if not legend_saved:
            _save_onetask_fulltrial_legend(d, "onetask_pca_fulltrial_legend.png")
            legend_saved = True


def plot_onetask_cancel():
    """
    Figure: fixon/task readout-cancellation across training, replotted from the
    pickle saved by one_task_analysis.py. Three curves — |Fix − Task| (combined
    residual, incl. bias), |Fix|, |Task| — of the mean readout-projection
    magnitude over the STIMULUS + DELAY period, vs number of training datasets
    (log x). Each curve shows a mean line with a ±1 SEM (std/sqrt(n) across
    trials) shaded band.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"cancel_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return

    counter = np.asarray(d["counter_lst"], dtype=float)
    cancel_mean = np.asarray(d["cancel_mean"], dtype=float)   # (stages, 3)
    # SEM band; fall back to the older "cancel_std" key for pre-SEM pickles.
    cancel_sem = np.asarray(d.get("cancel_sem", d.get("cancel_std")), dtype=float)
    labels = d["labels"]

    # Light palette for the ±SEM bands (matches one_task_analysis c_vals_l).
    c_vals_l = ["#feb2b2", "#90cdf4", "#9ae6b4", "#d6bcfa", "#fbd38d",
                "#81e6d9", "#e2e8f0", "#fbb6ce", "#faf089"] * 10

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    for k in range(cancel_mean.shape[1]):
        ax.plot(counter, cancel_mean[:, k], "-o", color=c_vals[k],
                markersize=3, label=labels[k])
        ax.fill_between(counter, cancel_mean[:, k] - cancel_sem[:, k],
                        cancel_mean[:, k] + cancel_sem[:, k],
                        color=c_vals_l[k], alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("# Dataset", fontsize=9)
    ax.set_ylabel("Readout projection magnitude", fontsize=9)
    # Fewer y-ticks for a cleaner axis.
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    _legend(ax, loc="best", fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "onetask_cancel.png"
    _save_fig(fig, out_path)


def plot_onetask_d_combine():
    """
    Figure: cross-period PCA explained-variance heatmaps for the single-task
    network — one panel each for hidden activity and effective modulation (W⊙M).
    Each 4x4 matrix (Fixation/Stimulus/Memory/Response) shows how well each
    period's top-k PCA subspace captures every other period's variance.
    Single-task analog of plot_two_task_d_combine; same color scheme (perceptually
    uniform `mako` cmap, shared 0-1 range, one shared colorbar). Reads
    d_combine_{aname}.pkl written by one_task_analysis.py.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"d_combine_{ONETASK_ANAME}.pkl"
    d_combine = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d_combine is None:
        return

    names = [n for n in ("hidden", "w_modulation") if n in d_combine]
    if not names:
        print("  Skipped: no series in d_combine pickle.")
        return
    # Shared color range across panels so a single colorbar applies to all.
    vmin = min(d_combine[n].get("vmin", 0.0) for n in names)
    vmax = max(d_combine[n].get("vmax", 1.0) for n in names)

    # Compact size, comparable to onetask_pc_cumvar.
    fig, axs = plt.subplots(1, len(names), figsize=(2.3 * len(names), 2.2),
                            gridspec_kw={"wspace": 0.15})
    if len(names) == 1:
        axs = [axs]
    title_map = {"hidden": "Hidden", "modulation": "Modulation",
                 "w_modulation": "Eff. modulation"}
    mesh = None
    for col, (ax, name) in enumerate(zip(axs, names)):
        e = d_combine[name]
        # y-tick labels only on the leftmost panel (all panels share the same
        # period rows); saves horizontal space so panels don't collide.
        ylabels = e["labels"] if col == 0 else False
        sns.heatmap(np.asarray(e["fve_k_all"]), ax=ax,
                    xticklabels=e["labels"], yticklabels=ylabels,
                    annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                    vmin=vmin, vmax=vmax, square=True,
                    cmap="mako", cbar=False)
        mesh = ax.collections[0]
        ax.set_title(title_map.get(name, name), fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        if col == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    # One shared colorbar for all panels.
    cb = fig.colorbar(mesh, ax=list(axs), shrink=0.8)
    cb.ax.tick_params(labelsize=7)
    out_path = OUT_DIR / "onetask_d_combine.png"
    _save_fig(fig, out_path)


def plot_onetask_pc_cumvar():
    """
    Figure: cumulative variance explained vs number of PCs, per trial period
    (the right panel of the cross-period dimensionality analysis). For each
    representation (hidden, effective modulation) a single panel plots, for each
    period, the fraction of that period's variance captured by its own top 1..N
    PCs — one curve per period (Fixation / Stimulus / Memory / Response), colored
    with the period-bar palette.

    Reads the `cumvar` array saved in d_combine_{aname}.pkl by
    one_task_analysis.py's cross_period_dimensionality. Skips gracefully if the
    pickle predates that field.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"d_combine_{ONETASK_ANAME}.pkl"
    d_combine = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d_combine is None:
        return

    names = [n for n in ("hidden", "w_modulation") if n in d_combine
             and d_combine[n].get("cumvar") is not None]
    if not names:
        print("  Skipped: 'cumvar' not in d_combine pickle "
              "(re-run one_task_analysis.py to add it).")
        return

    title_map = {"hidden": "Hidden", "w_modulation": "Eff. modulation"}
    # Period name -> period-bar color (Fixation/Stimulus/Memory/Response order).
    period_color = dict(zip(["Fixation", "Stimulus", "Memory", "Response"],
                            _ONETASK_PERIOD_COLORS))

    fig, axs = plt.subplots(1, len(names), figsize=(1.4 * len(names), 1.33),
                            squeeze=False)
    # Remember the period label/color order so the standalone legend below
    # matches the drawn curves exactly.
    legend_labels, legend_colors = None, None
    for ax, name in zip(axs[0], names):
        e = d_combine[name]
        cumvar = np.asarray(e["cumvar"], dtype=float)      # (n_period, max_pc)
        labels = e.get("labels", ["Fixation", "Stimulus", "Memory", "Response"])
        n_pc = cumvar.shape[1]
        xs = np.arange(1, n_pc + 1)
        cols = [period_color.get(lab, c_vals[i % len(c_vals)])
                for i, lab in enumerate(labels)]
        for i, lab in enumerate(labels):
            ax.plot(xs, cumvar[i], "-o", color=cols[i], markersize=3, label=lab)
        if legend_labels is None:
            legend_labels, legend_colors = labels, cols
        ax.set_title(title_map.get(name, name), fontsize=10)
        # Pad the limits a touch so the first x/y ticks sit off the origin corner.
        x_pad = 0.04 * (n_pc - 1)
        ax.set_xlim(1 - x_pad, n_pc + x_pad)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([1, n_pc])          # only endpoints (1 and 11)
        ax.set_yticks([0, 1])             # only 0 and 1
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        # No on-panel legend — the short figure has no room for it; the legend is
        # emitted as a separate figure below (when legends are enabled).

    # Shared x/y labels for both panels (y aligned to the leftmost panel), instead
    # of duplicating them on each subplot.
    fig.supxlabel("No. of PCs", fontsize=9)
    fig.supylabel("Var expl.", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "onetask_pc_cumvar.png"
    _save_fig(fig, out_path)

    # Standalone legend figure (only when legends are enabled, i.e. not
    # --no-legend), so the compact panels above stay uncluttered.
    if SHOW_LEGEND and legend_labels:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="-", color=c,
                              markersize=4, label=lab)
                   for lab, c in zip(legend_labels, legend_colors)]
        leg_fig = plt.figure(figsize=(1.4, 1.2))
        leg_fig.legend(handles=handles, loc="center", frameon=True, fontsize=8,
                       title="Period", title_fontsize=9)
        # This figure IS the legend; save it directly (bypass _save_fig's _n
        # suffix so its name is clean and it isn't gated on the flag again).
        leg_path = OUT_DIR / "onetask_pc_cumvar_legend.png"
        leg_fig.savefig(leg_path, dpi=300, bbox_inches="tight")
        plt.close(leg_fig)
        print(f"Saved: {leg_path}")


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
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return
    present = d["present"]
    period_title = d["period_title"]
    data = d["data"]
    reps = [r for r in ("hidden", "e_modulation") if r in data]
    if not reps or not present:
        print("  Skipped: no rep/period data in pickle.")
        return

    # Stimulus count, for the red→purple stimulus color ramp.
    n_stim = 1 + max(int(s) for rep in reps for v in present
                     for s in np.asarray(data[rep][v]["stim"]))

    n_row, n_col = len(reps), len(present)
    fig, axs = plt.subplots(n_row, n_col, figsize=(2.1 * n_col, 2.1 * n_row),
                            squeeze=False)
    # Track per-row (x, y) data ranges so every panel in a row shares one range.
    row_xy = {r: [] for r in range(n_row)}
    for r, rep in enumerate(reps):
        for cidx, v in enumerate(present):
            ax = axs[r][cidx]
            ent = data[rep].get(v)
            if ent is None:
                ax.axis("off")
                continue
            proj = np.asarray(ent["proj"])      # (batch, win_T, 2)
            stim = np.asarray(ent["stim"])
            # Hidden starts strictly at the period boundary (drop the leading
            # transition-in frame stored in the pickle); modulation keeps it.
            # `lead` = number of leading frames prepended by the analysis.
            lead = int(ent.get("lead", 0))
            disp_start = lead if rep == "hidden" else 0

            # For the hidden Stimulus panel, prepend each trial's fixation-period
            # endpoint so the stimulus trajectory literally continues from where
            # fixation ended. Trials share order and the delay PCA basis across
            # variants, so trial i's coordinates are directly comparable.
            fix_end = None
            if rep == "hidden" and v == "longstimulus":
                fix_ent = data.get("hidden", {}).get("longfixation")
                if fix_ent is not None:
                    fix_proj = np.asarray(fix_ent["proj"])   # (batch, win_T, 2)
                    if fix_proj.shape[0] == proj.shape[0]:
                        fix_end = fix_proj[:, -1, :]         # (batch, 2)

            def _dir_arrow(p, head_idx, color, filled=False):
                """Draw a directional arrowhead on trajectory p at head_idx,
                pointing along the direction of travel APPROACHING that point.
                `filled` uses a solid black-edged head for the endpoint; else a
                lighter head for the mid-trajectory direction cue.

                The tail is walked back until the head-to-tail displacement is a
                meaningful fraction of the trajectory's extent — otherwise a
                settled endpoint (where consecutive frames barely move) would
                yield an arbitrary/noisy direction. Falls back to a dot if the
                whole path is essentially stationary."""
                nP = p.shape[0]
                hi = head_idx % nP
                # Minimum meaningful step = a small fraction of the path extent.
                extent = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
                min_step = max(extent * 0.05, 1e-9)
                ti = hi - 1
                while ti > 0 and np.linalg.norm(p[hi] - p[ti]) < min_step:
                    ti -= 1
                if ti < 0 or np.linalg.norm(p[hi] - p[ti]) < min_step:
                    ax.scatter(p[hi, 0], p[hi, 1], color=color, s=12, zorder=3)
                    return
                # Endpoint and mid arrows are the same (small) size; the endpoint
                # is only slightly more opaque to read as the terminus. Use an
                # open ">"-style chevron head ("->") rather than a filled triangle.
                ax.annotate(
                    "", xy=(p[hi, 0], p[hi, 1]), xytext=(p[ti, 0], p[ti, 1]),
                    zorder=3,
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=0.8,
                        alpha=0.9 if filled else 0.6,
                        mutation_scale=9,
                    ),
                )

            for i in range(proj.shape[0]):
                col = stim_color(int(stim[i]), n_stim)
                p = proj[i, disp_start:, :]
                if fix_end is not None:
                    p = np.vstack([fix_end[i][None, :], p])  # prepend fixation end
                row_xy[r].append(p)
                ax.plot(p[:, 0], p[:, 1], color=col,
                        alpha=0.4, linewidth=0.8, zorder=2)
                # Mid-trajectory direction cue + endpoint arrow (replaces the
                # end circle) so each path's direction of travel is clear. The
                # mid arrow is placed at the ARC-LENGTH midpoint (50% of the
                # distance travelled), which is robust to non-uniform speed —
                # these trajectories jump most of their distance in the first few
                # frames and then settle, so a plain time- or nearest-to-centroid
                # midpoint would land right next to the start or the endpoint.
                # Skip the mid arrow where the trajectory barely moves / is not
                # informative: the fixation panels (both reps) and the hidden
                # delay/response panels (near-stationary settled points).
                show_mid = not (
                    v == "longfixation"
                    or (rep == "hidden" and v in ("longdelay", "longresponse"))
                )
                if show_mid and p.shape[0] >= 3:
                    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
                    cum = np.concatenate([[0.0], np.cumsum(seg)])
                    total = cum[-1]
                    if total > 0:
                        mid_idx = int(np.searchsorted(cum, 0.5 * total))
                    else:
                        mid_idx = p.shape[0] // 2
                    # keep off the endpoints so a forward arrow is always drawn
                    mid_idx = min(max(mid_idx, 1), p.shape[0] - 2)
                    _dir_arrow(p, mid_idx, col, filled=False)
                _dir_arrow(p, -1, col, filled=True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=7)
            if r == 0:
                ax.set_title(period_title.get(v, v), fontsize=11)

    # Give every panel in a row a common x/y range (padded), so panels within a
    # row are directly comparable and equally sized.
    for r in range(n_row):
        if not row_xy[r]:
            continue
        allp = np.vstack(row_xy[r])
        allp = allp[np.isfinite(allp).all(axis=1)]
        if allp.size == 0:
            continue
        (x0, y0), (x1, y1) = allp.min(axis=0), allp.max(axis=0)
        px = (x1 - x0) * 0.06 or 1e-3
        py = (y1 - y0) * 0.06 or 1e-3
        for cidx in range(n_col):
            ax = axs[r][cidx]
            if ax.has_data():
                ax.set_xlim(x0 - px, x1 + px)
                ax.set_ylim(y0 - py, y1 + py)

    # Shared x/y labels for the whole grid (all panels share the same axes).
    fig.supxlabel("Delay PC1", fontsize=11)
    fig.supylabel("Delay PC2", fontsize=11)

    fig.tight_layout()
    out_path = OUT_DIR / "onetask_long_fixed_points.png"
    _save_fig(fig, out_path)


def _grad_fp_2d_project(d, rep_key, pca):
    """Project a loaded grad-fp pickle's fixed points into the 2-PC `pca` for the
    2D figure of representation `rep_key`. Returns
    (periods, proj_by_period, traj_by_period, angle0_pt, n_stim):
      proj_by_period[v] : (batch, 2) PCA coords of that period's fixed points
      traj_by_period[v] : (win_T, 2) exemplar (angle-0) within-period trajectory,
                          present only where the pickle saved it
      angle0_pt[v]      : (2,) the exemplar-stimulus fixed point, for connectors
      n_stim            : stimulus-color count (dense ring size)
    Pure data prep — drawing lives in _draw_grad_fp_2d_row so several rules can
    share one figure."""
    results = d["results"]
    periods = list(results.keys())

    def _flat(arr):
        arr = np.asarray(arr, dtype=float)
        return arr.reshape(arr.shape[0], -1)

    n_stim = 1 + max(int(s) for v in periods for s in np.asarray(results[v]["stim"]))
    proj_by_period = {v: pca.transform(_flat(results[v][rep_key])) for v in periods}

    # Exemplar stimulus (angle 0): its within-period RECORDED trajectory and its
    # fixed point per period, for the anchored connector.
    _TRAJ_STIM = 0
    _traj_field = {"fixed_M": "traj_M", "fixed_WM": "traj_WM",
                   "fixed_hidden": "traj_hidden"}.get(rep_key)
    traj_by_period = {}
    angle0_pt = {}
    for v in periods:
        tr = results[v].get(_traj_field) if _traj_field else None
        if tr is not None and int(results[v].get("traj_stim", _TRAJ_STIM)) == _TRAJ_STIM:
            traj_by_period[v] = pca.transform(_flat(tr))   # (win_T, 2)
        st = np.asarray(results[v]["stim"], dtype=int)
        idx = np.where(st == _TRAJ_STIM)[0]
        if idx.size:
            angle0_pt[v] = proj_by_period[v][int(idx[0])]  # (2,) current FP

    return periods, proj_by_period, traj_by_period, angle0_pt, n_stim


def _draw_grad_fp_2d_row(axs_row, results, periods, proj_by_period, traj_by_period,
                         angle0_pt, n_stim, lim, show_period_titles=True,
                         row_label=None):
    """Draw one rule's four per-period 2D panels into the pre-created axes
    `axs_row` (length = n_col), using precomputed projections. Shared symmetric
    limit `lim` is passed in so multiple rows use IDENTICAL axes.
    `show_period_titles` prints the Fixation/Stimulus/… titles (typically only
    the top row); `row_label` writes a rotated label (e.g. the task rule) to the
    left of the row's first panel."""
    _TRAJ_STIM = 0
    _traj_col = stim_color(_TRAJ_STIM, n_stim)
    for j, (ax, v) in enumerate(zip(axs_row, periods)):
        e = results[v]
        fixed = proj_by_period[v]                   # (batch, 2)
        stim = np.asarray(e["stim"])
        good = _fixed_point_mask(e, fixed.shape[0])
        for i in range(fixed.shape[0]):
            col = stim_color(int(stim[i]), n_stim)
            if good[i]:
                # Converged fixed point: filled marker.
                ax.scatter(fixed[i, 0], fixed[i, 1], color=col, marker="o", s=18,
                           edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
            else:
                # Over-threshold (not stationary enough): hollow marker.
                ax.scatter(fixed[i, 0], fixed[i, 1], facecolor="none",
                           edgecolor=col, marker="o", s=18, linewidth=0.9,
                           alpha=0.85, zorder=3)
        # Trajectory anchored to the fixed points: previous period's FP (dashed
        # marker) → recorded within-period path → this period's FP (solid marker).
        tp = traj_by_period.get(v)
        if tp is not None and tp.shape[0] >= 1 and j >= 1:
            prev_v = periods[j - 1]
            if prev_v in angle0_pt and v in angle0_pt:
                p0, p1 = angle0_pt[prev_v], angle0_pt[v]
                xs = np.concatenate([[p0[0]], tp[:, 0], [p1[0]]])
                ys = np.concatenate([[p0[1]], tp[:, 1], [p1[1]]])
                ax.plot(xs, ys, color=_traj_col, linewidth=1.1, alpha=0.8, zorder=4)
                # Start: previous period's fixed point (dashed edge).
                ax.scatter([p0[0]], [p0[1]], color=_traj_col, marker="o", s=22,
                           edgecolor="black", linewidth=0.9, linestyle="--",
                           zorder=5)
                # End: current period's fixed point (solid edge).
                ax.scatter([p1[0]], [p1[1]], color=_traj_col, marker="o", s=22,
                           edgecolor="black", linewidth=0.5, zorder=5)
        if show_period_titles:
            ax.set_title(e.get("period_title", v), fontsize=11)
        if j == 0 and row_label is not None:
            ax.set_ylabel(row_label, fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")   # so a ring reads as circular, not stretched
        ax.tick_params(axis="both", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)


def _render_grad_fixed_points(d, rep_key, out_path, basis=None):
    """Single-rule 2D grad fixed-point figure (one row of period panels) from an
    already-loaded pickle dict `d`, laid out like onetask_long_fixed_points.
    Shared by the one-task and two-task grad fixed-point figures. `rep_key`
    selects which saved view to plot:
      "fixed_M"      raw modulation matrix M*        (the actual solved state)
      "fixed_WM"     effective modulation W⊙M*       (transform of M*)
      "fixed_hidden" hidden state produced by M*     (transform of M*)
    Each period's points (one per stimulus) are flattened and projected into a
    SHARED delay-period PCA; points are colored by stimulus.

    `basis`: an OPTIONAL pre-fitted 2-component PCA to project into. When None
    (the default, and the one-task behavior) the basis is fit on THIS pickle's
    own delay-period fixed points. When supplied (e.g. by the two-task driver,
    which passes the delayanti delay-period basis), every panel is projected into
    that EXTERNAL basis instead — so figures from different pickles/rules share
    one x-y plane and become directly comparable point-for-point."""
    _ensure_out_dir()
    results = d["results"]
    periods = list(results.keys())
    if not periods:
        print("  Skipped: no periods in grad fixed-point pickle.")
        return
    if any(results[v].get(rep_key) is None for v in periods):
        print(f"  Skipped '{rep_key}': not in pickle "
              f"(re-run one_task_analysis.py to add it).")
        return

    # Use the caller-supplied shared basis if given; otherwise fit on this
    # pickle's own delay-period fixed points (one-task / standalone behavior).
    if basis is None:
        basis = _fit_period_grad_fp_basis(d, rep_key)

    periods, proj, traj, angle0_pt, n_stim = _grad_fp_2d_project(d, rep_key, basis)
    lim = max(np.abs(np.vstack(list(proj.values()))).max() * 1.08, 1e-9)

    n_col = len(periods)
    # Match onetask_long_fixed_points' compact panel size.
    fig, axs = plt.subplots(1, n_col, figsize=(2.1 * n_col, 2.1), squeeze=False)
    _draw_grad_fp_2d_row(axs[0], results, periods, proj, traj, angle0_pt, n_stim,
                         lim, show_period_titles=True, row_label=None)

    # Shared x/y labels for the whole grid (all panels share the delay basis).
    fig.supxlabel("Delay PC1", fontsize=11)
    fig.supylabel("Delay PC2", fontsize=11)

    # (equal-aspect panels are incompatible with tight_layout; bbox_inches on
    # save handles trimming.)
    _save_fig(fig, out_path)


def _plot_onetask_grad_fixed_points(rep_key, out_name):
    """One-task wrapper: load the single-task grad-fp pickle and render `rep_key`."""
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"fixed_points_grad_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return
    _render_grad_fixed_points(d, rep_key, OUT_DIR / out_name)


def plot_onetask_grad_fixed_points():
    """
    Figures: gradient-solved fixed points for the single-task network, one figure
    per representation (each laid out like onetask_long_fixed_points, shared
    delay-period PCA, one panel per period, colored by stimulus). The filename
    suffix names the representation:
      onetask_grad_fixed_points_modulation.png   — raw modulation M* (solved state)
      onetask_grad_fixed_points_emodulation.png  — effective modulation W⊙M*
      onetask_grad_fixed_points_hidden.png       — hidden state produced by M*
    The W⊙M and hidden views are transforms of the same fixed points (not
    re-solved). Reads fixed_points_grad_{aname}.pkl.
    """
    _plot_onetask_grad_fixed_points("fixed_M", "onetask_grad_fixed_points_modulation.png")
    _plot_onetask_grad_fixed_points("fixed_WM", "onetask_grad_fixed_points_emodulation.png")
    _plot_onetask_grad_fixed_points("fixed_hidden", "onetask_grad_fixed_points_hidden.png")


def _fit_period_grad_fp_basis(d, rep_key, period="longdelay"):
    """Fit a 2-component PCA on ONE period's gradient fixed points of an
    already-loaded pickle dict `d`, for representation `rep_key`. `period` selects
    which trial epoch's fixed points define the basis (e.g. "longdelay" for the
    memory ring, "longstimulus" for the stimulus ring). Returns the fitted PCA
    (usable as the `basis` arg of the grad-fp renderers) or None if the pickle
    lacks that representation. Falls back to the first available period when the
    requested one is absent."""
    from sklearn.decomposition import PCA as _PCA
    results = d.get("results", {})
    periods = list(results.keys())
    if not periods or any(results[v].get(rep_key) is None for v in periods):
        return None
    basis_key = period if period in results else periods[0]
    arr = np.asarray(results[basis_key][rep_key], dtype=float)
    return _PCA(n_components=2, random_state=0).fit(arr.reshape(arr.shape[0], -1))


def _grad_fp_3d_project(d, rep_key, pca):
    """Project a loaded grad-fp pickle's fixed points into the 2-PC `pca` for the
    3D figure of representation `rep_key`. Returns
    (periods, proj_by_period, z_by_period, traj_by_period, n_stim):
      proj_by_period[v] : (batch, 2) x-y PCA coords of that period's fixed points
      z_by_period[v]    : (batch,) ideal cos-output target (cos θ in response,
                          else 0 — the task only demands an output in Response)
      traj_by_period[v] : (win_T, 2) exemplar (angle-0) within-period trajectory,
                          present only where the pickle saved it
      n_stim            : stimulus-color count (dense ring size)
    Pure data prep — the actual drawing lives in _draw_grad_fp_3d_row so several
    rules can share one figure."""
    results = d["results"]
    periods = list(results.keys())

    def _flat(arr):
        arr = np.asarray(arr, dtype=float)
        return arr.reshape(arr.shape[0], -1)

    n_stim = 1 + max(int(s) for v in periods for s in np.asarray(results[v]["stim"]))
    # Dense stimulus angles (radians), indexed by each fixed point's `stim`. Fall
    # back to evenly-spaced trained directions if the pickle lacks dense angles.
    dense_angles = np.asarray(d.get("angles", []), dtype=float)
    # Response offset: the required saccade angle is the stimulus angle for a PRO
    # rule but the OPPOSITE (stim + π) for an ANTI rule — see delaygo_ in
    # mpn_tasks.py (response_locs = stim_locs+π when anti_response). So the ideal
    # cos-output target is cos θ for pro and cos(θ+π) = −cos θ for anti. Keying off
    # the pickle's own rule keeps the delaygo and delayanti panels each correct.
    resp_offset = np.pi if "anti" in str(d.get("rule", "")).lower() else 0.0

    def _target_cos(v, e):
        """Ideal cos-output target: cos(response angle) in the response period,
        else 0 (the task only demands an output during Response)."""
        stim = np.asarray(e["stim"], dtype=int)
        if "response" not in v.lower():
            return np.zeros(stim.shape[0], dtype=float)
        if dense_angles.size and int(stim.max()) < dense_angles.size:
            ang = dense_angles[stim]
        else:
            ang = 2.0 * np.pi * stim / max(n_stim, 1)
        return np.cos(ang + resp_offset)

    proj_by_period = {v: pca.transform(_flat(results[v][rep_key])) for v in periods}
    z_by_period = {v: _target_cos(v, results[v]) for v in periods}

    # Within-period RECORDED trajectory of the exemplar stimulus (how the state
    # moves and converges during each period), projected into the same basis.
    _TRAJ_STIM = 0
    _traj_field = {"fixed_M": "traj_M", "fixed_WM": "traj_WM",
                   "fixed_hidden": "traj_hidden"}.get(rep_key)
    traj_by_period = {}
    for v in periods:
        tr = results[v].get(_traj_field) if _traj_field else None
        # Only the exemplar stimulus's trajectory is saved (traj_stim); ensure it
        # matches _TRAJ_STIM so it lines up with the connector endpoints.
        if tr is not None and int(results[v].get("traj_stim", _TRAJ_STIM)) == _TRAJ_STIM:
            traj_by_period[v] = pca.transform(_flat(tr))   # (win_T, 2)

    return periods, proj_by_period, z_by_period, traj_by_period, n_stim


def _draw_grad_fp_3d_row(fig, results, periods, proj_by_period, z_by_period,
                         traj_by_period, n_stim, lim, zmax, n_rows, row_idx,
                         n_col, show_period_titles=True, row_label=None):
    """Draw one rule's four per-period 3D panels into row `row_idx` of an
    (n_rows x n_col) subplot grid on `fig`, using precomputed projections. Shared
    x-y limit `lim` and symmetric z-limit `zmax` are passed in so that multiple
    rows/figures can use IDENTICAL axes (the two-task combined figure stacks
    delaygo over delayanti and shares both). `show_period_titles` prints the
    Fixation/Stimulus/… titles (typically only the top row); `row_label` writes a
    rotated label (e.g. the task rule) to the left of the row's first panel."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)

    # Angle-0 exemplar fixed point (x,y,z) per period, for the anchored connector.
    _TRAJ_STIM = 0
    angle0_pt = {}
    for v in periods:
        st = np.asarray(results[v]["stim"], dtype=int)
        idx = np.where(st == _TRAJ_STIM)[0]
        if idx.size:
            i0 = int(idx[0])
            angle0_pt[v] = (proj_by_period[v][i0, 0], proj_by_period[v][i0, 1],
                            float(z_by_period[v][i0]))
    _traj_col = stim_color(_TRAJ_STIM, n_stim)

    for j, v in enumerate(periods):
        ax = fig.add_subplot(n_rows, n_col, row_idx * n_col + j + 1,
                             projection="3d")
        e = results[v]
        xy = proj_by_period[v]                       # (batch, 2)
        z = z_by_period[v]
        stim = np.asarray(e["stim"])
        # Faint z=0 reference plane (drawn first so the colored fixed points sit
        # on top; low alpha keeps it a shadow, not an occluder).
        _pg = np.array([[-lim, lim], [-lim, lim]])
        ax.plot_surface(_pg, _pg.T, np.zeros((2, 2)), color="0.5", alpha=0.12,
                        edgecolor="none", shade=False, zorder=0)
        good = _fixed_point_mask(e, xy.shape[0])
        for i in range(xy.shape[0]):
            col = stim_color(int(stim[i]), n_stim)
            if good[i]:
                ax.scatter(xy[i, 0], xy[i, 1], z[i], color=col, marker="o", s=14,
                           edgecolor="black", linewidth=0.3, alpha=0.85)
            else:
                # Over-threshold point: hollow marker.
                ax.scatter(xy[i, 0], xy[i, 1], z[i], facecolor="none",
                           edgecolor=col, marker="o", s=14, linewidth=0.8,
                           alpha=0.85)
        # Exemplar-stimulus trajectory ANCHORED to the fixed points: it starts at
        # the PREVIOUS period's fixed point, follows the recorded within-period
        # path, and ends at THIS period's fixed point — so its endpoints coincide
        # with the solved fixed points (unlike the raw recorded window, whose ends
        # are recorded boundary states, not the relaxed fixed points). z sits at
        # this period's level along the path; the leading segment shows the jump
        # from the previous period's z. Start = dashed-edge, end = solid-edge.
        tp = traj_by_period.get(v)
        if tp is not None and tp.shape[0] >= 1 and j >= 1:
            prev_v = periods[j - 1]
            if prev_v in angle0_pt and v in angle0_pt:
                p0, p1 = angle0_pt[prev_v], angle0_pt[v]     # prev FP, current FP
                z_lvl = p1[2]                                # current period z
                xs = np.concatenate([[p0[0]], tp[:, 0], [p1[0]]])
                ys = np.concatenate([[p0[1]], tp[:, 1], [p1[1]]])
                zs = np.concatenate([[p0[2]], np.full(tp.shape[0], z_lvl), [p1[2]]])
                ax.plot(xs, ys, zs, color=_traj_col, linewidth=1.3, alpha=0.85,
                        zorder=5)
                # Start: previous period's fixed point (dashed edge).
                ax.scatter([p0[0]], [p0[1]], [p0[2]], color=_traj_col, marker="o",
                           s=30, edgecolor="black", linewidth=0.9,
                           linestyle="--", zorder=6)
                # End: current period's fixed point (solid edge).
                ax.scatter([p1[0]], [p1[1]], [p1[2]], color=_traj_col, marker="o",
                           s=30, edgecolor="black", linewidth=0.6, zorder=6)
        if show_period_titles:
            ax.set_title(e.get("period_title", v), fontsize=11, pad=-6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-zmax, zmax)
        # No tick labels, so pull the axis labels in tight against each axis.
        ax.set_xlabel("PC1", fontsize=7, labelpad=-15)
        ax.set_ylabel("PC2", fontsize=7, labelpad=-15)
        # z-axis label on EVERY panel, same small size/tight pad as x & y (was a
        # single larger shared label on the rightmost panel only).
        ax.set_zlabel("Output cos θ", fontsize=7, labelpad=-15)
        # Row label (e.g. the task rule) just to the left of the leftmost panel
        # (small negative x keeps it close to the 3D box rather than far out).
        if j == 0 and row_label is not None:
            ax.text2D(-0.10, 0.5, row_label, transform=ax.transAxes,
                      rotation=90, va="center", ha="right", fontsize=11)
        # Hide numeric tick labels on all three axes (keep the tick marks).
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(axis="both", labelsize=8, pad=-2)
        ax.view_init(elev=18, azim=-60)
        # Remove the grey background panes and gridlines.
        ax.grid(False)
        for _pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            _pane.pane.set_visible(False)


def _render_grad_fixed_points_3d(d, rep_key, out_path, basis=None):
    """Single-rule 3D grad fixed-point figure (one row of period panels) from an
    already-loaded pickle dict `d`: the x-y plane is a delay-period PCA of the
    fixed points, and the z-axis is the TARGET (expected) cos-output for each
    fixed point's stimulus, cos θ. z is ~0 for fixation/stimulus/delay and equals
    cos θ only in the Response panel — the ring lifts off the z=0 plane only
    there.

    `basis`: an OPTIONAL pre-fitted 2-component PCA to project into. When None
    (the default, and the one-task behavior) the basis is fit on THIS pickle's
    own delay-period fixed points. When supplied (e.g. by the two-task driver,
    which passes the delayanti delay-period basis), every panel is projected into
    that EXTERNAL basis instead — so figures from different pickles/rules share
    one x-y plane and become directly comparable point-for-point."""
    _ensure_out_dir()
    results = d["results"]
    periods = list(results.keys())
    if not periods:
        print("  Skipped: no periods in grad fixed-point pickle.")
        return
    if any(results[v].get(rep_key) is None for v in periods):
        print(f"  Skipped '{rep_key}': not in pickle "
              f"(re-run one_task_analysis.py to add it).")
        return

    # Use the caller-supplied shared basis if given; otherwise fit on this
    # pickle's own delay-period fixed points (one-task / standalone behavior).
    if basis is None:
        basis = _fit_period_grad_fp_basis(d, rep_key)

    periods, proj, zc, traj, n_stim = _grad_fp_3d_project(d, rep_key, basis)
    lim = max(np.abs(np.vstack(list(proj.values()))).max() * 1.08, 1e-9)
    zmax = max(np.abs(np.concatenate([zc[v].ravel() for v in periods])).max() * 1.1,
               1e-6)

    n_col = len(periods)
    # Compact panels: no tick labels, so each panel can be small and packed close.
    fig = plt.figure(figsize=(1.8 * n_col, 1.8))
    _draw_grad_fp_3d_row(fig, results, periods, proj, zc, traj, n_stim, lim, zmax,
                         n_rows=1, row_idx=0, n_col=n_col,
                         show_period_titles=True, row_label=None)
    # Per-panel z-labels now (no shared right-margin label), so use full width.
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.12)
    _save_fig(fig, out_path)


def _plot_onetask_grad_fixed_points_3d(rep_key, out_name):
    """One-task wrapper: load the single-task grad-fp pickle and render `rep_key`
    in 3D."""
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"fixed_points_grad_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return
    _render_grad_fixed_points_3d(d, rep_key, OUT_DIR / out_name)


def plot_onetask_grad_fixed_points_3d():
    """
    3D versions of the gradient fixed-point figures: x-y = shared delay-period
    PCA of the fixed points, z = cos-output readout (response cos θ), which is
    ~0 except in the response period. One figure per representation:
      onetask_grad_fixed_points_3d_modulation.png
      onetask_grad_fixed_points_3d_emodulation.png
      onetask_grad_fixed_points_3d_hidden.png
    Reads fixed_points_grad_{aname}.pkl.
    """
    _plot_onetask_grad_fixed_points_3d("fixed_M", "onetask_grad_fixed_points_3d_modulation.png")
    _plot_onetask_grad_fixed_points_3d("fixed_WM", "onetask_grad_fixed_points_3d_emodulation.png")
    _plot_onetask_grad_fixed_points_3d("fixed_hidden", "onetask_grad_fixed_points_3d_hidden.png")


def _render_interp_fixed_points(d, out_path, n_trained=ONETASK_N_STIM,
                                period="longdelay", src_name=""):
    """Continuous-attractor probe from an already-loaded pickle dict `d`. Shared
    by the one-task and two-task interp figures.

    Two panels (for the delay period by default):
      left  — the fixed points in a 2-PC PCA of the solved M*, colored by
              stimulus angle. A smooth, evenly-filled ring ⇒ continuous
              attractor; clustering onto ~8 points ⇒ discrete attractors.
              Over-threshold points (not stationary enough) are drawn hollow.
      right — scale-free relative step ‖F(M*)−M*‖/‖M*‖ vs angle (log y), with the
              rel_tol acceptance line. Uniformly below the line ⇒ every angle is a
              (slowly-varying) fixed point (continuous manifold); excursions above
              it ⇒ those angles did not settle to a fixed point.
    `n_trained` sets how many dashed trained-direction guide lines to draw;
    `src_name` names the pickle in the skip message."""
    _ensure_out_dir()
    angles = np.asarray(d.get("angles", []), dtype=float)
    results = d.get("results", {})
    if period not in results or angles.size == 0:
        print(f"  Skipped: period '{period}' or dense angles not in "
              f"{src_name} (re-run the analysis).")
        return
    e = results[period]
    fixed = np.asarray(e["fixed_M"], dtype=float)
    n = len(angles)
    # Scale-free relative step; fall back to sqrt(2q)/‖M‖ for older pickles that
    # lack the saved rel_step field.
    rel_step = e.get("rel_step")
    if rel_step is None:
        step_norm = np.sqrt(2.0 * np.asarray(e["final_speeds"], dtype=float))
        m_norm = np.maximum(np.linalg.norm(fixed.reshape(n, -1), axis=1), 1e-12)
        rel_step = step_norm / m_norm
    rel_step = np.asarray(rel_step, dtype=float)
    rel_tol = float(e.get("rel_tol", d.get("rel_tol", 0.05)))
    good = _fixed_point_mask(e, n)

    from sklearn.decomposition import PCA as _PCA
    proj = _PCA(n_components=2, random_state=0).fit_transform(
        fixed.reshape(n, -1))

    # Color each angle by its position on the ring (continuous rainbow ramp).
    cols = [stim_color(k, n) for k in range(n)]

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.4),
                            gridspec_kw={"wspace": 0.35})

    # Left: fixed points in PCA, connected in angle order to show the ring.
    # Over-threshold points are drawn hollow.
    ax = axs[0]
    ax.plot(np.append(proj[:, 0], proj[0, 0]), np.append(proj[:, 1], proj[0, 1]),
            "-", color="0.7", linewidth=0.8, alpha=0.6, zorder=1)
    for i in range(n):
        if good[i]:
            ax.scatter(proj[i, 0], proj[i, 1], color=cols[i], s=30,
                       edgecolor="black", linewidth=0.3, zorder=3)
        else:
            ax.scatter(proj[i, 0], proj[i, 1], facecolor="none", edgecolor=cols[i],
                       s=30, linewidth=1.1, zorder=3)
    ax.set_xlabel("FP PC1", fontsize=10)
    ax.set_ylabel("FP PC2", fontsize=10)
    ax.set_title(f"Fixed points ({n} angles)", fontsize=10)
    ax.set_aspect("equal")   # so a ring reads as circular, not stretched
    ax.spines[["top", "right"]].set_visible(False)

    # Right: relative step vs angle (continuity diagnostic). Mark the 8 trained
    # angles and the rel_tol acceptance line.
    ax = axs[1]
    deg = np.degrees(angles)
    ax.plot(deg, rel_step, "-o", color=c_vals[0], markersize=3)
    # Dashed lines at the trained ring directions — only when few enough to be
    # legible (a dense/`morestimulus` run has too many to mark).
    if n_trained <= 16:
        for k in range(n_trained):
            ax.axvline(360.0 * k / n_trained, color="0.8", lw=0.6,
                       linestyle="--", zorder=0)
    ax.axhline(rel_tol, color=c_vals[3], lw=1.0, linestyle="-",
               label=f"rel_tol = {rel_tol:g}", zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Stimulus angle (deg)", fontsize=10)
    ax.set_ylabel(r"Relative step  $\|F(M^*)-M^*\|/\|M^*\|$", fontsize=9)
    _dash_note = (f"\n(dashed = {n_trained} trained dirs)"
                  if n_trained <= 16 else "")
    ax.set_title(f"Relative step vs angle{_dash_note}", fontsize=9)
    _legend(ax, fontsize=7, loc="best")
    ax.spines[["top", "right"]].set_visible(False)

    # (equal-aspect left panel is incompatible with tight_layout; bbox_inches on
    # save handles trimming.)
    _save_fig(fig, out_path)


def plot_onetask_interp_fixed_points(period="longdelay"):
    """One-task continuous-attractor probe. Reads the shared
    fixed_points_grad_{aname}.pkl (written by one_task_analysis.py's
    _solve_period_modulation_fixed_points) and renders it."""
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"fixed_points_grad_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return
    _render_interp_fixed_points(d, OUT_DIR / "onetask_interp_fixed_points.png",
                                n_trained=ONETASK_N_STIM, period=period,
                                src_name=pkl_path.name)


def _render_fixed_point_stability(d, out_path, n_trained=ONETASK_N_STIM):
    """Linear-stability spectrum of the gradient fixed points, per period, from an
    already-loaded pickle dict `d`. Shared by the one-task and two-task figures.
    Reads the Jacobian eigenvalues saved in the grad-fp pickle (written by the
    stability pass in core/grad_fixed_points.py). Two rows:

      top  — leading Jacobian eigenvalues of F(M*) in the COMPLEX PLANE, one panel
             per period, colored by stimulus angle, with the unit circle. As a
             discrete map: points inside the circle are contracting, outside are
             expanding; an eigenvalue sitting AT (1, 0) is a marginal/neutral
             direction — the ring-attractor signature.
      bottom — spectral radius ρ = max|λ| vs stimulus angle, per period, with the
             ρ = 1 stability line. ρ < 1 ⇒ attracting fixed point.

    `n_trained` sets how many dashed trained-direction guide lines to draw.
    Skips gracefully if the pickle predates the stability pass.
    """
    _ensure_out_dir()
    results = d.get("results", {})
    periods = list(results.keys())
    if not periods or any(results[v].get("eigenvalues") is None for v in periods):
        print("  Skipped: 'eigenvalues' not in pickle "
              "(re-run the analysis to add the stability pass).")
        return

    angles = np.asarray(d.get("angles", []), dtype=float)
    deg = np.degrees(angles) if angles.size else None
    marg_tol = float(results[periods[0]].get("marginal_tol", 0.05))
    n_col = len(periods)

    # Stimulus color count (dense ring); eigenvalues share the stimulus of their
    # fixed point.
    n_stim = 1 + max(int(s) for v in periods for s in np.asarray(results[v]["stim"]))

    theta = np.linspace(0, 2 * np.pi, 200)
    fig, axs = plt.subplots(2, n_col, figsize=(2.6 * n_col, 5.0), squeeze=False)

    for j, v in enumerate(periods):
        e = results[v]
        eig = np.asarray(e["eigenvalues"])              # (batch, k) complex
        stim = np.asarray(e["stim"])
        rad = np.asarray(e["spectral_radius"], dtype=float)

        # ── Top row: eigenvalues in the complex plane ────────────────────────
        ax = axs[0][j]
        ax.plot(np.cos(theta), np.sin(theta), "-", color="0.7", lw=0.8, zorder=1)
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.axvline(0, color="0.85", lw=0.5, zorder=0)
        for i in range(eig.shape[0]):
            col = stim_color(int(stim[i]), n_stim)
            ax.scatter(eig[i].real, eig[i].imag, color=col, s=6, alpha=0.6,
                       edgecolor="none", zorder=3)
        ax.set_title(e.get("period_title", v), fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("Re(λ)", fontsize=9)
        if j == 0:
            ax.set_ylabel("Im(λ)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # ── Bottom row: spectral radius vs stimulus angle ────────────────────
        ax = axs[1][j]
        x = deg if deg is not None and deg.size == rad.size else np.arange(rad.size)
        ax.plot(x, rad, "-o", color=c_vals[0], markersize=3, zorder=3)
        ax.axhline(1.0, color=c_vals[3], lw=1.0, zorder=2,
                   label="ρ = 1")
        # Shade the marginal band [1-tol, 1+tol].
        ax.axhspan(1.0 - marg_tol, 1.0 + marg_tol, color="0.85", alpha=0.5, zorder=0)
        if deg is not None and n_trained <= 16:
            for kk in range(n_trained):
                ax.axvline(360.0 * kk / n_trained, color="0.9", lw=0.5,
                           linestyle="--", zorder=0)
        ax.set_xlabel("Stimulus angle (deg)" if deg is not None else "Stimulus index",
                      fontsize=9)
        if j == 0:
            ax.set_ylabel(r"Spectral radius  $\rho=\max|\lambda|$", fontsize=9)
            _legend(ax, fontsize=7, loc="best")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_onetask_fixed_point_stability():
    """One-task linear-stability spectrum. Reads the Jacobian eigenvalues saved
    in fixed_points_grad_{aname}.pkl and renders them."""
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"fixed_points_grad_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return
    _render_fixed_point_stability(
        d, OUT_DIR / "onetask_fixed_point_stability.png", n_trained=ONETASK_N_STIM)


# Reference rule whose delay-period PCA defines the SHARED x-y basis for BOTH
# the 2D and 3D two-task fixed-point figures. Every rule's figure is projected
# into this rule's delay basis so their rings are directly comparable
# point-for-point (rather than each rule using its own, incomparable, delay PCA).
_TWOTASK_FP_BASIS_RULE = "delayanti"

# Row order (top → bottom) for the combined two-task fixed-point figures (2D and
# 3D): each rule is one row, delaygo above delayanti.
_TWOTASK_FP_ROW_ORDER = ["delaygo", "delayanti"]


def _twotask_shared_fp_bases(paths, label, period="longdelay"):
    """Fit ONE shared 2-PC PCA per representation from the `_TWOTASK_FP_BASIS_RULE`
    reference rule's pickle, for the two-task grad fixed-point figures. `period`
    selects which trial epoch's fixed points define the basis ("longdelay" or
    "longstimulus"). `paths` is the (rule, path) list from
    _twotask_grad_fp_paths(); `label` tags the log line.

    Returns a dict rep_key -> fitted PCA (usable as the renderers' `basis` arg).
    An EMPTY dict means the reference pickle was missing or unreadable, in which
    case each rule falls back to its own basis (the per-pickle default)."""
    ref_paths = [p for (r, p) in paths if r == _TWOTASK_FP_BASIS_RULE]
    shared = {}
    if ref_paths:
        d_ref = _load_pkl_or_skip(ref_paths[0], "Run two_task_analysis.py first.")
        if d_ref is not None:
            for rep_key in ("fixed_M", "fixed_WM", "fixed_hidden"):
                b = _fit_period_grad_fp_basis(d_ref, rep_key, period=period)
                if b is not None:
                    shared[rep_key] = b
        print(f"  [{label}] shared x-y basis = '{_TWOTASK_FP_BASIS_RULE}' {period} "
              f"({len(shared)}/3 representations).")
    else:
        print(f"  [{label}] reference rule '{_TWOTASK_FP_BASIS_RULE}' pickle not "
              f"found; each rule uses its own {period} basis.")
    return shared


def _load_two_task_grad_fp_rules(paths):
    """Load each two-task rule's grad-fp pickle once, ordered top→bottom per
    `_TWOTASK_FP_ROW_ORDER` (rules not in that list are appended after, in
    discovery order). Returns an ordered list of (rule, loaded-pickle-dict),
    skipping rules whose pickle is missing/unreadable. Shared by the combined 2D
    and 3D two-task drivers so both stack rows in the same order."""
    by_rule = dict(paths)
    ordered_rules = ([r for r in _TWOTASK_FP_ROW_ORDER if r in by_rule]
                     + [r for r in by_rule if r not in _TWOTASK_FP_ROW_ORDER])
    rule_data = []
    for rule in ordered_rules:
        d = _load_pkl_or_skip(by_rule[rule], "Run two_task_analysis.py first.")
        if d is not None:
            rule_data.append((rule, d))
    return rule_data


def _render_two_task_grad_fp_2d_combined(rule_data, rep_key, out_path, basis,
                                         pc_label="Delay"):
    """Draw ALL two-task rules as stacked rows of a SINGLE 2D figure for one
    representation. `rule_data` is an ordered list of (rule, loaded-pickle-dict)
    (row order = top→bottom); every rule is projected into the shared `basis` and
    the rows share one symmetric x-y limit so panels are directly comparable
    across rows. Period titles print only on the top row; each row is labeled on
    the left (y-axis) by its task rule. `pc_label` names the shared PCA basis on
    the figure's x/y labels (e.g. "Delay" or "Stimulus")."""
    # Project every rule first, so a shared axis limit can span all rows.
    per_rule = []   # (rule, results, periods, proj, traj, angle0_pt, n_stim)
    for rule, d in rule_data:
        results = d["results"]
        periods = list(results.keys())
        if not periods or any(results[v].get(rep_key) is None for v in periods):
            print(f"  Skipped '{rep_key}' for rule '{rule}': not in pickle.")
            continue
        periods, proj, traj, a0, n_stim = _grad_fp_2d_project(d, rep_key, basis)
        per_rule.append((rule, results, periods, proj, traj, a0, n_stim))
    if not per_rule:
        print(f"  Skipped '{rep_key}': no rule had it.")
        return

    # Shared symmetric x-y limit across ALL rows (so rows are comparable).
    lim = max(np.abs(np.vstack([p for (_, _, _, proj, _, _, _) in per_rule
                                for p in proj.values()])).max() * 1.08, 1e-9)
    n_col = max(len(periods) for (_, _, periods, _, _, _, _) in per_rule)
    n_rows = len(per_rule)

    fig, axs = plt.subplots(n_rows, n_col, figsize=(2.1 * n_col, 2.1 * n_rows),
                            squeeze=False)
    for row_idx, (rule, results, periods, proj, traj, a0, n_stim) in enumerate(per_rule):
        _draw_grad_fp_2d_row(
            axs[row_idx], results, periods, proj, traj, a0, n_stim, lim,
            show_period_titles=(row_idx == 0),
            row_label=_TASK_DISPLAY.get(rule, rule))

    # Shared x/y labels for the whole grid (all panels share the same basis).
    fig.supxlabel(f"{pc_label} PC1", fontsize=11)
    fig.supylabel(f"{pc_label} PC2", fontsize=11)
    # (equal-aspect panels are incompatible with tight_layout; bbox_inches on
    # save handles trimming.)
    _save_fig(fig, out_path)


# The two-task grad fixed-point figures are produced in TWO variants that differ
# only by which trial period of the `_TWOTASK_FP_BASIS_RULE` defines the shared
# x-y PCA basis. Each entry: (period key, filename infix, axis-label prefix).
# The infix keeps the two variants' output files distinguishable.
_TWOTASK_FP_BASIS_VARIANTS = [
    ("longdelay",    "delaypc", "Delay"),
    ("longstimulus", "stimpc",  "Stimulus"),
]


def _plot_two_task_grad_fp_combined(stem_prefix, log_label, render_fn,
                                    with_pc_label):
    """Shared driver for the combined two-task grad fixed-point figures (2D and
    3D). Loads each rule's pickle once, then for every basis variant
    (`_TWOTASK_FP_BASIS_VARIANTS`) and every representation (raw M*, W⊙M*,
    hidden) fits/reuses the shared delayanti basis and calls `render_fn` to draw
    all rules as stacked rows into ONE figure. Output:
      {stem_prefix}_{seed}_{infix}_{suffix}.png

    stem_prefix   : filename stem before the seed tag ("twotask_grad_fixed_points"
                    for 2D, that + "_3d" for 3D).
    log_label     : short tag for the shared-basis log line ("twotask-2d"/"-3d").
    render_fn     : the combined renderer (_render_two_task_grad_fp_2d_combined or
                    _..._3d_combined); called as
                    render_fn(rule_data, rep_key, out_path, basis[, pc_label=...]).
    with_pc_label : pass the variant's axis-label prefix as pc_label= (2D only;
                    the 3D renderer uses generic PC1/PC2 axis labels)."""
    paths = _twotask_grad_fp_paths()
    if not paths:
        print("  Skipped: no fixed_points_grad_*_{rule}.pkl in "
              f"{TWOTASKS_DIR / TWOTASK_ANAME}. Run two_task_analysis.py first.")
        return
    tag = _twotask_seed_tag()
    rule_data = _load_two_task_grad_fp_rules(paths)
    if not rule_data:
        return

    for period, infix, pc_label in _TWOTASK_FP_BASIS_VARIANTS:
        shared_bases = _twotask_shared_fp_bases(paths, f"{log_label}/{infix}",
                                                period=period)
        for rep_key, suffix in (("fixed_M", "modulation"),
                                ("fixed_WM", "emodulation"),
                                ("fixed_hidden", "hidden")):
            # Shared basis for this representation; fall back to the first rule's
            # own basis for this period if the reference-rule pickle was absent.
            basis = shared_bases.get(rep_key) or _fit_period_grad_fp_basis(
                rule_data[0][1], rep_key, period=period)
            if basis is None:
                print(f"  Skipped '{rep_key}' ({period}): no basis could be fit.")
                continue
            out_path = OUT_DIR / f"{stem_prefix}_{tag}_{infix}_{suffix}.png"
            extra = {"pc_label": pc_label} if with_pc_label else {}
            render_fn(rule_data, rep_key, out_path, basis, **extra)


def plot_two_task_grad_fixed_points():
    """
    2D two-task gradient fixed-point figures — BOTH rules stacked as rows of a
    SINGLE figure per representation (top → bottom = _TWOTASK_FP_ROW_ORDER, i.e.
    delaygo over delayanti), colored by stimulus; period titles on the top row,
    task-rule labels down the left.

    TWO BASIS VARIANTS (`_TWOTASK_FP_BASIS_VARIANTS`): the same fixed points are
    plotted twice, differing only in which period of the `_TWOTASK_FP_BASIS_RULE`
    (delayanti) defines the shared x-y PCA basis — the DELAY ring vs the STIMULUS
    ring. The filename infix distinguishes them:
      twotask_grad_fixed_points_{seed}_delaypc_modulation.png  (+ emodulation, hidden)
      twotask_grad_fixed_points_{seed}_stimpc_modulation.png   (+ emodulation, hidden)

    SHARED BASIS: within each variant, all rows are projected into a SINGLE common
    x-y plane (fit once per representation) and share one symmetric x-y limit, so
    the delaygo and delayanti rings are directly comparable point-for-point. If
    the reference rule's pickle is missing, the basis falls back to the first
    available rule's own corresponding period.
    Reads twotasks/{aname}/fixed_points_grad_{aname}_{rule}.pkl.
    """
    _plot_two_task_grad_fp_combined(
        "twotask_grad_fixed_points", "twotask-2d",
        _render_two_task_grad_fp_2d_combined, with_pc_label=True)


def _render_two_task_grad_fp_3d_combined(rule_data, rep_key, out_path, basis):
    """Draw ALL two-task rules as stacked rows of a SINGLE 3D figure for one
    representation. `rule_data` is an ordered list of (rule, loaded-pickle-dict)
    (row order = top→bottom); every rule is projected into the shared `basis` and
    the rows share one x-y limit and one z-limit so the panels are directly
    comparable across rows. Period titles print only on the top row; each row is
    labeled on the left by its task rule."""
    # Project every rule first, so shared axis limits can span all rows.
    per_rule = []   # (rule, results, periods, proj, zc, traj, n_stim)
    for rule, d in rule_data:
        results = d["results"]
        periods = list(results.keys())
        if not periods or any(results[v].get(rep_key) is None for v in periods):
            print(f"  Skipped '{rep_key}' for rule '{rule}': not in pickle.")
            continue
        periods, proj, zc, traj, n_stim = _grad_fp_3d_project(d, rep_key, basis)
        per_rule.append((rule, results, periods, proj, zc, traj, n_stim))
    if not per_rule:
        print(f"  Skipped '{rep_key}': no rule had it.")
        return

    # Shared symmetric x-y and z limits across ALL rows (so rows are comparable).
    lim = max(np.abs(np.vstack([p for (_, _, _, proj, _, _, _) in per_rule
                                for p in proj.values()])).max() * 1.08, 1e-9)
    zmax = max(np.abs(np.concatenate([zc[v].ravel()
                                      for (_, _, periods, _, zc, _, _) in per_rule
                                      for v in periods])).max() * 1.1, 1e-6)
    n_col = max(len(periods) for (_, _, periods, _, _, _, _) in per_rule)
    n_rows = len(per_rule)

    fig = plt.figure(figsize=(1.8 * n_col, 1.8 * n_rows))
    for row_idx, (rule, results, periods, proj, zc, traj, n_stim) in enumerate(per_rule):
        _draw_grad_fp_3d_row(
            fig, results, periods, proj, zc, traj, n_stim, lim, zmax,
            n_rows=n_rows, row_idx=row_idx, n_col=n_col,
            show_period_titles=(row_idx == 0),
            row_label=_TASK_DISPLAY.get(rule, rule))
    # hspace sets the vertical gap between the two task rows (3D axes carry large
    # internal margins, so this is negative but less so than the tightest pack).
    # Per-panel z-labels now (no shared right-margin label), so use full width.
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top=0.94,
                        wspace=0.12, hspace=-0.05)
    _save_fig(fig, out_path)


def plot_two_task_grad_fixed_points_3d():
    """
    3D two-task gradient fixed-point figures — BOTH rules stacked as rows of a
    SINGLE figure per representation (top → bottom = _TWOTASK_FP_ROW_ORDER, i.e.
    delaygo over delayanti). x-y = shared PCA basis, z = ideal cos-output target.

    TWO BASIS VARIANTS (`_TWOTASK_FP_BASIS_VARIANTS`), like the 2D figures: the
    same fixed points are plotted twice, differing only in which period of the
    `_TWOTASK_FP_BASIS_RULE` (delayanti) defines the shared x-y PCA basis — the
    DELAY ring vs the STIMULUS ring. The filename infix distinguishes them:
      twotask_grad_fixed_points_3d_{seed}_delaypc_modulation.png  (+ emodulation, hidden)
      twotask_grad_fixed_points_3d_{seed}_stimpc_modulation.png   (+ emodulation, hidden)

    SHARED BASIS: within each variant, all rows are projected into a SINGLE common
    x-y plane (fit once per representation) and share one x-y/z limit, so the
    delaygo and delayanti rings are directly comparable point-for-point. If the
    reference rule's pickle is missing, the basis falls back to the first
    available rule's own corresponding period.
    Reads twotasks/{aname}/fixed_points_grad_{aname}_{rule}.pkl.
    """
    _plot_two_task_grad_fp_combined(
        "twotask_grad_fixed_points_3d", "twotask-3d",
        _render_two_task_grad_fp_3d_combined, with_pc_label=False)


def plot_two_task_interp_fixed_points(period="longdelay"):
    """
    Figure: continuous-attractor probe for the two-task network, one per task
    rule (twotask_interp_fixed_points_{seed}_{rule}.png). Same two-panel layout
    as the one-task version (FP ring in PCA + relative-step-vs-angle continuity
    diagnostic). Reads twotasks/{aname}/fixed_points_grad_{aname}_{rule}.pkl.
    """
    paths = _twotask_grad_fp_paths()
    if not paths:
        print("  Skipped: no fixed_points_grad_*_{rule}.pkl in "
              f"{TWOTASKS_DIR / TWOTASK_ANAME}. Run two_task_analysis.py first.")
        return
    tag = _twotask_seed_tag()
    for rule, pkl_path in paths:
        d = _load_pkl_or_skip(pkl_path, "Run two_task_analysis.py first.")
        if d is None:
            continue
        _render_interp_fixed_points(
            d, OUT_DIR / f"twotask_interp_fixed_points_{tag}_{rule}.png",
            n_trained=TWOTASK_N_STIM, period=period, src_name=pkl_path.name)


def _render_interp_alpha_fp_3d(d, rep_key, out_path, basis):
    """3D figure of the TASK-INTERPOLATION fixed points: one panel per trial
    period, x = interpolation level alpha (pro<->anti), y/z = the two PCs of the
    shared `basis` (delayanti delay-period). Each stimulus is one line traced
    across alpha, colored by stimulus. Reads the interp_fixed_points_{aname}.pkl
    written by two_task_analysis.py, where results[period][rep_key] has shape
    (n_alpha, n_stim, feat)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)
    _ensure_out_dir()
    results = d.get("results", {})
    # Order panels canonically Fixation -> Stimulus -> Delay -> Response
    # (the pickle stores them delay/response/stimulus/fixation); any period not in
    # the canonical list is appended after, in pickle order.
    _CANON = ["longfixation", "longstimulus", "longdelay", "longresponse"]
    periods = ([v for v in _CANON if v in results]
               + [v for v in results if v not in _CANON])
    if not periods or any(results[v].get(rep_key) is None for v in periods):
        print(f"  Skipped '{rep_key}': not in interp pickle "
              f"(re-run two_task_analysis.py with --interp-save-full for M*/W⊙M*).")
        return
    alphas = np.asarray(d["alphas"], dtype=float)

    def _proj(arr):
        # arr: (n_alpha, n_stim, feat) -> (n_alpha, n_stim, 2) in the shared basis.
        a = np.asarray(arr, dtype=float)
        na, ns = a.shape[0], a.shape[1]
        return basis.transform(a.reshape(na * ns, -1)).reshape(na, ns, 2)

    proj_by_period = {v: _proj(results[v][rep_key]) for v in periods}
    n_stim = proj_by_period[periods[0]].shape[1]
    # Shared symmetric PC limits across periods so panels are comparable.
    lim = max(np.abs(np.concatenate([p.reshape(-1, 2) for p in proj_by_period.values()])).max()
              * 1.08, 1e-9)

    n_col = len(periods)
    fig = plt.figure(figsize=(1.9 * n_col, 1.9))
    for j, v in enumerate(periods):
        ax = fig.add_subplot(1, n_col, j + 1, projection="3d")
        xy = proj_by_period[v]                       # (n_alpha, n_stim, 2)
        good = np.asarray(results[v].get("is_fixed",
                          np.ones(xy.shape[:2], bool)), dtype=bool)
        # Fixation has no stimulus tuning (all 8 lines coincide), so coloring by
        # stimulus is misleading — draw it black instead of the stimulus rainbow.
        is_fixation = "fixation" in v.lower()
        # Along each stimulus trajectory, shade markers DARK->BRIGHT across the
        # sweep: alpha=0 is a DARKENED version of the base color (toward black,
        # e.g. dark red), alpha=1 is a LIGHTENED tint (toward white), so the
        # pro<->anti direction is readable per line. Maps alpha in [0,1] to
        # _shade's frac in [-0.55, +0.6].
        na = xy.shape[0]
        t = (alphas - alphas.min()) / max(alphas.max() - alphas.min(), 1e-9)
        shade_frac = -0.55 + t * (0.6 - (-0.55))
        for s in range(n_stim):
            base = "black" if is_fixation else stim_color(s, n_stim)
            # Line across alpha for this stimulus (PC1=y, PC2=z vs alpha=x).
            ax.plot(alphas, xy[:, s, 0], xy[:, s, 1], "-", color=base,
                    linewidth=1.1, alpha=0.5, zorder=2)
            # Per-alpha shaded points: converged filled, over-threshold hollow.
            for ai in range(na):
                col = _shade(base, shade_frac[ai])
                if good[ai, s]:
                    ax.scatter(alphas[ai], xy[ai, s, 0], xy[ai, s, 1], color=col,
                               marker="o", s=12, edgecolor="black", linewidth=0.3,
                               alpha=0.9, zorder=3)
                else:
                    ax.scatter(alphas[ai], xy[ai, s, 0], xy[ai, s, 1],
                               facecolor="none", edgecolor=col, marker="o", s=12,
                               linewidth=0.7, alpha=0.9, zorder=3)
        ax.set_title(results[v].get("period_title", v), fontsize=11, pad=-6)
        ax.set_xlim(alphas.min(), alphas.max())
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        # More-negative labelpad pulls each axis label in closer to its axis.
        ax.set_xlabel(r"$\alpha$", fontsize=7, labelpad=-15)
        ax.set_ylabel("Delay PC1", fontsize=7, labelpad=-17)
        # z-axis label on EVERY panel, same small size as the other axis labels.
        ax.set_zlabel("Delay PC2", fontsize=7, labelpad=-17)
        ax.set_xticks([alphas.min(), alphas.max()])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(axis="both", labelsize=7, pad=-2)
        ax.view_init(elev=18, azim=-60)
        ax.grid(False)
        for _pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            _pane.pane.set_visible(False)

    # Per-panel z-labels (no shared right-margin label), so use the full width.
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.92, wspace=0.12)
    _save_fig(fig, out_path)


def plot_two_task_interp_alpha_fixed_points_3d():
    """
    3D figure of the task-interpolation fixed points: per trial period, x = the
    pro<->anti interpolation level alpha, y/z = the two PCs of the delayanti
    delay-period basis (the SAME shared basis as the grad-fixed-point 3D figures).
    Each of the 8 stimuli is one line traced across alpha, colored by stimulus, so
    the figure shows how each period's fixed points move as the task cue morphs
    from anti (alpha=0) to pro (alpha=1). One figure per representation:
      twotask_interp_alpha_fixed_points_3d_{seed}_modulation.png  (+ emodulation, hidden)
    Reads interp_fixed_points_{aname}.pkl (needs --interp-save-full for the
    modulation / emodulation variants; hidden is always available).
    """
    d = _load_twotask_glob_or_skip("interp_fixed_points_*.pkl")
    if d is None:
        return
    tag = _twotask_seed_tag()
    # Shared x-y (here y-z) basis: delayanti delay-period grad fixed points, one
    # per representation — identical to the grad-fixed-point 3D figures.
    paths = _twotask_grad_fp_paths()
    shared_bases = _twotask_shared_fp_bases(paths, "twotask-interp-alpha",
                                            period="longdelay") if paths else {}
    for rep_key, suffix in (("fixed_M", "modulation"),
                            ("fixed_WM", "emodulation"),
                            ("fixed_hidden", "hidden")):
        basis = shared_bases.get(rep_key)
        if basis is None:
            print(f"  Skipped '{rep_key}': no delayanti delay basis "
                  f"(need fixed_points_grad_*_delayanti.pkl).")
            continue
        _render_interp_alpha_fp_3d(
            d, rep_key,
            OUT_DIR / f"twotask_interp_alpha_fixed_points_3d_{tag}_{suffix}.png",
            basis)


def plot_two_task_interp_alpha_bifurcation(period="longstimulus", rep_key="fixed_WM",
                                           pc=0):
    """
    2D bifurcation diagram of the task-interpolation fixed points for a single
    period/representation: x = the pro<->anti interpolation level alpha, y = one
    PC of the delayanti delay-period basis (PC1 by default). Each of the 8 stimuli
    is one line traced across alpha (dark at alpha=0 -> bright at alpha=1), so a
    fan that collapses/splits as alpha varies reads as a bifurcation of the
    fixed-point structure. Defaults to the STIMULUS period, effective modulation
    (W⊙M). Reads interp_fixed_points_{aname}.pkl (needs --interp-save-full for
    the modulation / emodulation reps). Writes
      twotask_interp_alpha_bifurcation_{seed}_{period}_{rep}_pc{pc+1}.png
    """
    d = _load_twotask_glob_or_skip("interp_fixed_points_*.pkl")
    if d is None:
        return
    results = d.get("results", {})
    if period not in results or results[period].get(rep_key) is None:
        print(f"  Skipped: period '{period}' / '{rep_key}' not in interp pickle "
              f"(re-run two_task_analysis.py with --interp-save-full).")
        return
    paths = _twotask_grad_fp_paths()
    shared = _twotask_shared_fp_bases(paths, "twotask-bifurcation",
                                      period="longdelay") if paths else {}
    basis = shared.get(rep_key)
    if basis is None:
        print(f"  Skipped '{rep_key}': no delayanti delay basis "
              f"(need fixed_points_grad_*_delayanti.pkl).")
        return

    alphas = np.asarray(d["alphas"], dtype=float)
    arr = np.asarray(results[period][rep_key], dtype=float)   # (n_alpha, n_stim, feat)
    na, n_stim = arr.shape[0], arr.shape[1]
    proj = basis.transform(arr.reshape(na * n_stim, -1)).reshape(na, n_stim, 2)
    good = np.asarray(results[period].get("is_fixed",
                      np.ones((na, n_stim), bool)), dtype=bool)

    # Dark->bright shading along alpha (matches the 3D figure convention).
    t = (alphas - alphas.min()) / max(alphas.max() - alphas.min(), 1e-9)
    shade_frac = -0.55 + t * (0.6 - (-0.55))

    _ensure_out_dir()
    fig, ax = plt.subplots(1, 1, figsize=(2.6, 2.2))
    for s in range(n_stim):
        base = stim_color(s, n_stim)
        ax.plot(alphas, proj[:, s, pc], "-", color=base, linewidth=1.0,
                alpha=0.5, zorder=2)
        for ai in range(na):
            col = _shade(base, shade_frac[ai])
            if good[ai, s]:
                ax.scatter(alphas[ai], proj[ai, s, pc], color=col, marker="o",
                           s=16, edgecolor="black", linewidth=0.3, alpha=0.9, zorder=3)
            else:
                ax.scatter(alphas[ai], proj[ai, s, pc], facecolor="none",
                           edgecolor=col, marker="o", s=16, linewidth=0.7,
                           alpha=0.9, zorder=3)
    title = results[period].get("period_title", period)
    ax.set_xlabel(r"$\alpha$", fontsize=9)
    ax.set_ylabel(f"Delay PC{pc + 1}", fontsize=9)
    ax.set_title(f"{title} bifurcation", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _rep_tag = {"fixed_M": "modulation", "fixed_WM": "emodulation",
                "fixed_hidden": "hidden"}.get(rep_key, rep_key)
    out_path = OUT_DIR / (f"twotask_interp_alpha_bifurcation_{_twotask_seed_tag()}"
                          f"_{period}_{_rep_tag}_pc{pc + 1}.png")
    _save_fig(fig, out_path)


def plot_two_task_fixed_point_stability():
    """
    Figure: linear-stability spectrum of the two-task gradient fixed points, one
    per task rule (twotask_fixed_point_stability_{seed}_{rule}.png). Same two-row
    layout as the one-task version (eigenvalues in the complex plane + spectral
    radius vs angle). Reads twotasks/{aname}/fixed_points_grad_{aname}_{rule}.pkl.
    """
    paths = _twotask_grad_fp_paths()
    if not paths:
        print("  Skipped: no fixed_points_grad_*_{rule}.pkl in "
              f"{TWOTASKS_DIR / TWOTASK_ANAME}. Run two_task_analysis.py first.")
        return
    tag = _twotask_seed_tag()
    for rule, pkl_path in paths:
        d = _load_pkl_or_skip(pkl_path, "Run two_task_analysis.py first.")
        if d is None:
            continue
        _render_fixed_point_stability(
            d, OUT_DIR / f"twotask_fixed_point_stability_{tag}_{rule}.png",
            n_trained=TWOTASK_N_STIM)


def plot_two_task_d_combine():
    """
    Figure: cross-task / cross-period PCA explained-variance heatmaps for the
    two-task network — one panel each for hidden activity and effective
    modulation (W⊙M).

    Reads the self-contained d_combine pickle written by two_task_analysis.py
    (twotasks/{TWOTASK_ANAME}/d_combine_{TWOTASK_ANAME}.pkl), which stores the
    already-permuted 8x8 FVE matrix, its tick labels, and the color range for
    each of "hidden" and "w_modulation".
    """
    _ensure_out_dir()
    pkl_path = TWOTASKS_DIR / TWOTASK_ANAME / f"d_combine_{TWOTASK_ANAME}.pkl"
    d_combine = _load_pkl_or_skip(pkl_path, "Run two_task_analysis.py first.")
    if d_combine is None:
        return

    # Prefer the effective-modulation series; fall back to raw "modulation" for
    # older pickles that predate the W⊙M change.
    mod_key = "w_modulation" if "w_modulation" in d_combine else "modulation"
    names = [n for n in ("hidden", mod_key) if n in d_combine]
    # Shared color range across panels so a single colorbar applies to both.
    vmin = min(d_combine[n].get("vmin", 0.0) for n in names)
    vmax = max(d_combine[n].get("vmax", 1.0) for n in names)

    fig, axs = plt.subplots(1, len(names), figsize=(2.8 * len(names), 2.6),
                            gridspec_kw={"wspace": 0.15})
    if len(names) == 1:
        axs = [axs]
    title_map = {"hidden": "Hidden", "modulation": "Modulation",
                 "w_modulation": "Eff. modulation"}
    mesh = None
    for col, (ax, name) in enumerate(zip(axs, names)):
        e = d_combine[name]
        # y-tick labels only on the leftmost panel (all panels share the same
        # row labels); saves horizontal space so panels don't collide.
        ylabels = e["labels"] if col == 0 else False
        sns.heatmap(np.asarray(e["fve_k_all"]), ax=ax,
                    xticklabels=e["labels"], yticklabels=ylabels,
                    annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                    vmin=vmin, vmax=vmax, square=True,
                    cmap="mako", cbar=False)
        mesh = ax.collections[0]
        ax.set_title(title_map.get(name, name), fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        if col == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    # One shared colorbar for all panels.
    cb = fig.colorbar(mesh, ax=list(axs), shrink=0.8)
    cb.ax.tick_params(labelsize=7)
    out_path = OUT_DIR / f"twotask_d_combine_{_twotask_seed_tag()}.png"
    _save_fig(fig, out_path)


def plot_two_task_pc_cumvar():
    """
    Figure: cumulative variance explained vs number of PCs, per task and period,
    for the two-task network — the two-task analog of onetask_pc_cumvar. A 2x2
    grid: rows = representation (hidden / effective modulation), columns = task
    (Go / Anti). Each panel plots one curve per trial period (colored with the
    period-bar palette), showing how many PCs each period's trajectory needs.

    Reads the self-contained pc_cumvar pickle written by two_task_analysis.py
    (twotasks/{TWOTASK_ANAME}/pc_cumvar_{TWOTASK_ANAME}.pkl). Skips gracefully if
    the pickle predates that field.
    """
    _ensure_out_dir()
    d = _load_twotask_glob_or_skip("pc_cumvar_*.pkl")
    if d is None:
        return

    mod_key = "w_modulation" if "w_modulation" in d else "modulation"
    names = [n for n in ("hidden", mod_key) if n in d]
    if not names:
        print("  Skipped: no series in pc_cumvar pickle.")
        return

    rep_title = {"hidden": "Hidden", "w_modulation": "Eff. modulation",
                 "modulation": "Modulation"}
    # Period-bar palette, matching onetask_pc_cumvar. The two-task pickle stores
    # the periods under abbreviated names (context/stim/delay/resp) rather than
    # the one-task labels, but both list the same four trial epochs in the same
    # Fixation→Stimulus→Memory→Response order, so color BY POSITION into
    # _ONETASK_PERIOD_COLORS (falling back to the categorical cycle for any
    # extra periods) instead of keying on the mismatched names.
    def _period_col(pi):
        return (_ONETASK_PERIOD_COLORS[pi] if pi < len(_ONETASK_PERIOD_COLORS)
                else c_vals[pi % len(c_vals)])

    n_task = len(d[names[0]]["task_names"])
    # Per-panel size matches onetask_pc_cumvar (1.4 wide x 1.33 tall).
    fig, axs = plt.subplots(len(names), n_task,
                            figsize=(1.4 * n_task, 1.33 * len(names)),
                            squeeze=False)
    for r, name in enumerate(names):
        e = d[name]
        cumvar = np.asarray(e["cumvar"], dtype=float)     # (n_task, n_period, max_pc)
        pnames = e["period_names"]
        tnames = e["task_names"]
        n_pc = cumvar.shape[2]
        xs = np.arange(1, n_pc + 1)
        for ti in range(n_task):
            ax = axs[r][ti]
            for pi, pname in enumerate(pnames):
                ax.plot(xs, cumvar[ti, pi], "-o", color=_period_col(pi),
                        markersize=3, label=pname)
            # Pad limits so the first ticks sit off the origin corner.
            x_pad = 0.04 * (n_pc - 1)
            ax.set_xlim(1 - x_pad, n_pc + x_pad)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks([1, n_pc])
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(tnames[ti], fontsize=10)
            if r == len(names) - 1:
                ax.set_xlabel("No. of PCs", fontsize=9)
            if ti == 0:
                ax.set_ylabel(f"{rep_title.get(name, name)}\nVar expl.", fontsize=9)
            # No legend here — the period colors match onetask_pc_cumvar's shared
            # palette (colored BY POSITION into _ONETASK_PERIOD_COLORS), so the
            # standalone onetask_pc_cumvar_legend applies to this figure too.

    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_pc_cumvar_{_twotask_seed_tag()}.png"
    _save_fig(fig, out_path)


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
    n_stim = int(np.max(ltc[:, 0])) + 1 if len(ltc) else ONETASK_N_STIM

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
        color = stim_color(ltc[i, 0], n_stim)
        ls = linestyles[task]
        data_i = projected[i]
        seg = slice(stim0, trial_end)
        ax.plot(data_i[seg, a], data_i[seg, bb], c=color, linestyle=ls, alpha=0.5)
        for _, t0_key, t1_key, mk_idx in phases:
            sl = slice(ts[t0_key], ts[t1_key])
            ax.scatter(data_i[sl, a], data_i[sl, bb], color=color,
                       marker=markers_vals[mk_idx], alpha=0.8)
        for t_key, mk_idx in transitions:
            t = ts[t_key] - 1
            ax.scatter([data_i[t, a]], [data_i[t, bb]], color=color,
                       marker=markers_vals[mk_idx], alpha=1.0, s=60,
                       linewidths=0.6, zorder=10)
    ax.set_xlabel("PCA 1", fontsize=20)
    ax.set_ylabel("PCA 2", fontsize=20)
    ax.tick_params(axis="both", labelsize=15)
    if show_legend:
        _legend(ax, handles=legend_handles, loc="upper right",
                  frameon=legend_frameon, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / out_name
    _save_fig(fig, out_path)


def plot_two_task_m_pca():
    """
    Figure: PCA trajectories of hidden activity, modulation (raw M), and
    effective modulation (w_modulation = W⊙M) for the two-task network ("normal"
    variant). Reads the self-contained m_pca pickle written by
    two_task_analysis.py and emits one figure per panel type
    (hidden / modulation / w_modulation).
    """
    m_pca = _load_twotask_glob_or_skip("m_pca_normal_*.pkl")
    if m_pca is None:
        return

    for name in ("hidden", "modulation", "w_modulation"):
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
    n_stim = len(interpolation_label)
    for i in range(len(interpolation_label)):
        fixed_points = pdf_all[:, i, :]
        color = stim_color(interpolation_label[i], n_stim)
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
    ac = _load_twotask_glob_or_skip("m_pca_attractor_cycle_*.pkl")
    if ac is None:
        return

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
        _save_fig(fig, out_path)

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
        _save_fig(fig, out_path)

    plotted = 0
    for name in names:
        if not any(f"{sname}|{name}" in ac for sname in periods):
            continue
        _render(name)
        plotted += 1
    if plotted == 0:
        print("  Skipped: no expected entries found in m_pca_attractor_cycle pickle.")
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
    saved = _load_twotask_glob_or_skip("cancel_seed*.pkl")
    if saved is None:
        return
    stimuli = saved["stimuli"]
    markers = saved["markers"]

    stim_keys = sorted(stimuli.keys())
    n_rows = len(stim_keys)

    # Trial periods: fixation | stimulus | memory(delay) | response, bounded by
    # the saved marker times. Drawn as a top color bar (colors only, no shading),
    # matching the one-task figures' _ONETASK_PERIOD_COLORS grayscale palette.
    fix_end = markers["fixation_end"]
    stim_end = markers["stimulus_end"]
    delay_end = markers["delay_end"]
    fix_c, stim_c, mem_c, resp_c = _ONETASK_PERIOD_COLORS
    period_spans = [
        (0, fix_end, fix_c, "Fixation"),
        (fix_end, stim_end, stim_c, "Stimulus"),
        (stim_end, delay_end, mem_c, "Memory"),
        (delay_end, None, resp_c, "Response"),
    ]

    fig, axs = plt.subplots(n_rows, 2, figsize=(3.4 * 2, 1.8 * n_rows),
                            squeeze=False)
    # column 0 = task1 (pro), column 1 = task2 (anti)
    cols = [
        ("fixon_proj1", "x_task1_proj", "fixoff_proj1", "Task 1 (pro)"),
        ("fixon_proj2", "x_task2_proj", "fixoff_proj2", "Task 2 (anti)"),
    ]
    T = None
    for r, si in enumerate(stim_keys):
        e = stimuli[si]
        bias = e["bias_proj"]
        for c, (fixon_k, task_k, fixoff_k, col_name) in enumerate(cols):
            ax = axs[r][c]
            fixon = np.asarray(e[fixon_k])
            task = np.asarray(e[task_k])
            T = len(fixon)
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
            # Extra title pad on the top row so it clears the period bar above it.
            ax.set_title(f"Stimulus {si}; {col_name}", fontsize=9,
                         pad=14 if r == 0 else None)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0 and c == 0:
                _legend(ax, frameon=True, fontsize=6, loc="best")
            if r == n_rows - 1:
                ax.set_xlabel("Timestep", fontsize=9)
            else:
                ax.set_xticklabels([])

    # Period color bar above each top-row panel (colors only, no shading).
    if T is not None:
        for c in range(2):
            _add_period_strip(axs[0][c], period_spans, xmax=T - 1)

    # Shared y-label centered across the panels.
    fig.supylabel("Proj Cos Mag", fontsize=9)

    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_cancel_{_twotask_seed_tag()}.png"
    _save_fig(fig, out_path, extra=f"  (stimuli {stim_keys})")


def plot_two_task_outputsubspace_cancel():
    """
    Figure: output-subspace cancellation scatter for the two-task network.
    Two panels, each scattering per-stimulus values across three x-categories
    (Projection to Cosine Output / Orthogonal Complement / Random Vector):
      left  — "Cancelation between Same Stimulus" = |task1 + task2| projection
              (small for the cosine-output category means the pro & anti memory
               states cancel along the readout axis).
      right — "Magnitude of Projection" = the individual |task1| magnitude.
    Log y-axis. Reads the self-contained pickle written by two_task_analysis.py.
    """
    _ensure_out_dir()
    d = _load_twotask_glob_or_skip("outputsubspace_cancel_*.pkl")
    if d is None:
        return

    projs_all = np.asarray(d["projs_all"], dtype=float)   # (n_cat, n_stim, 3)
    cat_labels = d["category_labels"]
    n_cat, n_stim, _ = projs_all.shape

    fig, axs = plt.subplots(1, 2, figsize=(2.6 * 2, 2.6))
    # Spread the per-stimulus points around each integer x so they don't fully
    # overlap, and draw them semi-transparent so density is visible.
    jitter = np.linspace(-0.13, 0.13, n_stim) if n_stim > 1 else np.zeros(1)
    for i in range(n_cat):
        for k in range(n_stim):
            axs[0].scatter(i + jitter[k], projs_all[i, k, 0], color=c_vals[i],
                           alpha=0.45, s=16, edgecolors="none", zorder=2)
            axs[1].scatter(i + jitter[k], projs_all[i, k, 1], color=c_vals[i],
                           alpha=0.45, s=16, edgecolors="none", zorder=2)
    # Overlay geometric mean +/- 1 std computed in log space (matching the log
    # y-axis), so the error bars summarize the scatter of the 8 stimuli.
    for ax, val_idx in ((axs[0], 0), (axs[1], 1)):
        for i in range(n_cat):
            vals = projs_all[i, :, val_idx]
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            log_v = np.log10(vals)
            m, s = log_v.mean(), log_v.std()
            center = 10.0 ** m
            lo = center - 10.0 ** (m - s)
            hi = 10.0 ** (m + s) - center
            ax.errorbar(i, center, yerr=[[lo], [hi]], fmt="_",
                        color=c_vals[i], ecolor=c_vals[i], elinewidth=1.2,
                        capsize=3, markersize=11, markeredgewidth=1.6, zorder=5)
    for ax in axs:
        ax.set_xticks(list(range(n_cat)))
        # Wrap long category names onto multiple lines so they fit under the
        # small panel without overlapping their neighbors.
        ax.set_xticklabels(_wrap(cat_labels, width=12), fontsize=6)
        ax.tick_params(axis="both", which="both", labelsize=7)
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel(_wrap(d.get("combined_ylabel", "Cancelation between Same Stimulus"),
                            width=18), fontsize=7)
    axs[1].set_ylabel(_wrap(d.get("magnitude_ylabel", "Magnitude of Projection"),
                            width=18), fontsize=7)

    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_outputsubspace_cancel_{_twotask_seed_tag()}.png"
    _save_fig(fig, out_path)


def plot_two_task_w_gram_matrix():
    """
    Figure: input-embedding Gram matrix (W_initial_linear^T @ W_initial_linear)
    for the two-task network, FINAL training stage only. The 7x7 matrix shows,
    per pair of input channels (Fixon, Stim1 Cos/Sin, Stim2 Cos/Sin, Task1,
    Task2), the inner product of their learned embedding vectors: diagonal =
    per-channel gain (norm^2), off-diagonal = overlap between channel embeddings
    in hidden space. Reads the self-contained pickle written by
    two_task_analysis.py. Matches the analysis heatmap style (coolwarm, center 0).
    """
    _ensure_out_dir()
    d = _load_twotask_glob_or_skip("w_gram_matrix_*.pkl")
    if d is None:
        return

    keys = d["keys"]
    gram = np.asarray(d["gram_final"], dtype=float)   # (7, 7) final stage

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.9))
    sns.heatmap(gram, ax=ax, center=0, cmap="coolwarm", square=True,
                xticklabels=keys, yticklabels=keys, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7}, cbar_kws={"shrink": 0.8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_w_gram_matrix_{_twotask_seed_tag()}.png"
    _save_fig(fig, out_path)


def plot_two_task_w_hurt():
    """
    Figure: magnitude-pruning accuracy curve for the two-task network — test
    accuracy vs. the fraction of recurrent MP weights (W) zeroed out by smallest
    magnitude. A slow accuracy drop until high sparsity indicates the task
    computation is carried by a small subset of large-magnitude weights (W is
    prunable). Reads the self-contained pickle written by two_task_analysis.py.
    """
    _ensure_out_dir()
    d = _load_twotask_glob_or_skip("w_hurt_*.pkl")
    if d is None:
        return

    sparsity = np.asarray(d["sparsity_pct"], dtype=float)
    acc = np.asarray(d["accuracy"], dtype=float) * 100.0
    x = np.arange(len(sparsity))

    fig, ax = plt.subplots(1, 1, figsize=(2.4, 1.7))
    ax.plot(x, acc, "-o", color=c_vals[0], markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in sparsity], rotation=45, ha="right",
                       fontsize=7)
    ax.set_xlabel("Sparsity of W (%)", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_w_hurt_{_twotask_seed_tag()}.png"
    _save_fig(fig, out_path)


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
    d = _load_pkl_or_skip(pkl_path, "Run two_task_analysis.py first.")
    if d is None:
        return
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
    _legend(ax, frameon=True, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_attractor_first_overlearning_{seed_tag}.png"
    _save_fig(fig, out_path)

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
    _save_fig(fig, out_path)


def _plot_memory_attractor(pkl_path, out_prefix, aname):
    """
    Shared body for the DMC / delayDM memory-attractor figures.

    Reloads a fixed-point pickle written by multiple_task_analysis.py's
    shared_run and, for each representation (hidden, m_modulation, e_modulation),
    plots the delay-period trajectory + end-of-delay fixed points for 8 stimuli
    x 2 tasks in that representation's optimal 2-PC plane (the plane with the
    highest group-separation 2D silhouette). Color = stimulus, marker = task
    (square = task0/Pro, triangle = task1/Anti). One figure per representation.

    If the pickle sets `connect_endpoint_ring` (delayDM, where the endpoints form
    a stim1-angle ring), each task's fixed points are joined in stim order
    (0→1→…→7→0) to make the ring geometry visible; DMC leaves them unconnected.

    Files are saved as `{out_prefix}_{rep}_{aname}.png`.
    """
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
        connect_ring = bool(e.get("connect_endpoint_ring"))
        bx, by = e["best_plane"]
        sil = e.get("best_plane_silhouette", float("nan"))
        n_task, n_stim, _ = fp.shape
        have_traj = len(traj_by_task) == n_task

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
        for t_idx in range(n_task):
            for stim in range(n_stim):
                col = stim_color(stim_labels[stim], n_stim)
                if have_traj:
                    tr = traj_by_task[t_idx][stim]   # (delay_T, n_pc)
                    ax.plot(tr[:, bx], tr[:, by], color=col, alpha=0.5,
                            linewidth=0.8, linestyle=task_styles[t_idx % len(task_styles)],
                            zorder=2)
                ax.scatter(fp[t_idx, stim, bx], fp[t_idx, stim, by],
                           facecolor=col, edgecolor="black", linewidth=0.5,
                           marker=task_markers[t_idx % len(task_markers)],
                           s=22, alpha=0.7, zorder=3)
            # delayDM: close the stim1-angle ring (stim0→…→stimN→stim0).
            if connect_ring and n_stim >= 2:
                ring = np.concatenate(
                    [fp[t_idx, :, [bx, by]].T, fp[t_idx, :1, [bx, by]].T], axis=0)
                ax.plot(ring[:, 0], ring[:, 1], color="0.4", linewidth=0.8,
                        alpha=0.6, linestyle=task_styles[t_idx % len(task_styles)],
                        zorder=2)
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
                                   markerfacecolor=stim_color(lab, n_stim),
                                   markeredgecolor="black", markersize=5, label=f"stim {lab}")
                        for lab in stim_labels]
        task_handles = [plt.Line2D([0], [0], marker=task_markers[i],
                                   linestyle=task_styles[i], color="0.5",
                                   markerfacecolor="0.7", markeredgecolor="black",
                                   markersize=6, label=task_names[i])
                        for i in range(n_task)]
        _legend(ax, handles=stim_handles + task_handles, frameon=True, fontsize=5,
                  ncol=2, loc="best")

        fig.tight_layout()
        out_path = OUT_DIR / f"{out_prefix}_{rep}_{aname}.png"
        _save_fig(fig, out_path, extra=f"  (PC{bx+1}-PC{by+1}, 2D sil={sil:.3f})")

    for rep in ("hidden", "m_modulation", "e_modulation"):
        _plot_one(rep)


def plot_dmc_memory_attractor():
    """
    Figure: DMC category-memory attractors for each representation (hidden,
    m_modulation, e_modulation), each in its own optimal 2-PC plane.

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
    _plot_memory_attractor(pkl_path, "dmc_memory_attractor", DMC_ANAME)


def plot_delaydm_memory_attractor():
    """
    Figure: delayDM integration-memory attractors for each representation
    (hidden, m_modulation, e_modulation), each in its own optimal 2-PC plane.

    Reloads the fixed-point pickle written by multiple_task_analysis.py's
    shared_run("delaydm1") (multiple_tasks/{DELAYDM_ANAME}/
    delaydm1_fixed_points_{DELAYDM_ANAME}.pkl). DELAYDM_ANAME is independent of
    ANAME, so this can come from a different seed/regularization than the other
    multi-task figures. Shows the delay-period trajectory + end-of-delay fixed
    points for 8 stimuli x 2 tasks (color = stimulus, marker = task: square =
    delaydm1/Modality1, triangle = delaydm2/Modality2). Because the delayDM
    endpoints form a stim1-angle ring, each task's fixed points are joined in
    stim order. One figure per representation.
    """
    _ensure_out_dir()
    pkl_path = Path("multiple_tasks") / DELAYDM_ANAME / f"delaydm1_fixed_points_{DELAYDM_ANAME}.pkl"
    _plot_memory_attractor(pkl_path, "delaydm_memory_attractor", DELAYDM_ANAME)


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
        "onetask_stimulus_colorwheel": plot_onetask_stimulus_colorwheel,
        "onetask_show": plot_onetask_show,
        "onetask_modulation_snapshot": plot_onetask_modulation_snapshot,
        "onetask_modulation_snapshot_single": plot_onetask_modulation_snapshot_single,
        "onetask_pca_fulltrial": plot_onetask_pca_fulltrial,
        "onetask_cancel": plot_onetask_cancel,
        "onetask_d_combine": plot_onetask_d_combine,
        "onetask_pc_cumvar": plot_onetask_pc_cumvar,
        "onetask_long_fixed_points": plot_onetask_long_fixed_points,
        "onetask_grad_fixed_points": plot_onetask_grad_fixed_points,
        "onetask_grad_fixed_points_3d": plot_onetask_grad_fixed_points_3d,
        "onetask_interp_fixed_points": plot_onetask_interp_fixed_points,
        "onetask_fixed_point_stability": plot_onetask_fixed_point_stability,
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
        "delaydm_memory_attractor": plot_delaydm_memory_attractor,
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
        "twotask_pc_cumvar": plot_two_task_pc_cumvar,
        "twotask_m_pca": plot_two_task_m_pca,
        "twotask_attractor_cycle": plot_two_task_attractor_cycle,
        "twotask_cancel": plot_two_task_cancel,
        "twotask_outputsubspace_cancel": plot_two_task_outputsubspace_cancel,
        "twotask_grad_fixed_points": plot_two_task_grad_fixed_points,
        "twotask_grad_fixed_points_3d": plot_two_task_grad_fixed_points_3d,
        "twotask_interp_fixed_points": plot_two_task_interp_fixed_points,
        "twotask_interp_alpha_fixed_points_3d": plot_two_task_interp_alpha_fixed_points_3d,
        "twotask_interp_alpha_bifurcation": plot_two_task_interp_alpha_bifurcation,
        "twotask_fixed_point_stability": plot_two_task_fixed_point_stability,
        "twotask_w_gram_matrix": plot_two_task_w_gram_matrix,
        "twotask_w_hurt": plot_two_task_w_hurt,
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
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Suppress legends on every figure (overrides the SHOW_LEGEND "
             "default).",
    )
    args = parser.parse_args()

    # Apply the legend toggle globally; every figure routes through _legend(),
    # which reads this module-level flag.
    global SHOW_LEGEND
    if args.no_legend:
        SHOW_LEGEND = False

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
        "two_in_multiple": f"DMC={DMC_ANAME}, delayDM={DELAYDM_ANAME}",
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
    print(f"Legends: {'on' if SHOW_LEGEND else 'off'}")
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
