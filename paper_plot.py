"""
Paper figure generation for the MultiTaskMPN project.

Each public function produces one publication-ready figure and saves it
to the `paper_plot/` directory. Run the script directly to generate all
figures, or import individual functions as needed.

Figures are grouped into modes by the experiment they depend on:
    one_task         single-task training analyses
    multiple_tasks   full multi-task network (clustering, lesion, state space)
    acc_plot         accuracy comparisons (L2, activation, projection/hidden dims)
    two_in_multiple  delayDM fixed-point geometry probe of the multi-task network
    pretraining      pretraining → post-training transfer analyses
    two_task         two-task network (cross-task / cross-period PCA)

Usage:
    python paper_plot.py                       # generate every mode
    python paper_plot.py all                   # same as above
    python paper_plot.py one_task              # only the one-task figures
    python paper_plot.py multiple_tasks        # only the multi-task figures
    python paper_plot.py acc_plot              # only the accuracy figures
    python paper_plot.py two_in_multiple       # only the two-in-multiple figures
    python paper_plot.py pretraining           # only the pretraining figures
    python paper_plot.py two_task              # only the two-task figures
    python paper_plot.py --only input          # generate a single figure
"""
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def _save_standalone_colorbar(out_path, cmap, vmin, vmax, ticks=None,
                              ticklabels=None, label=None,
                              orientation="horizontal", figsize=(1.5, 0.45),
                              rect=(0.05, 0.5, 0.9, 0.35), labelsize=8,
                              label_fontsize=8):
    """Save JUST a colorbar as its own small figure at `out_path`.

    For panels that share one color scale: drawing the bar once beside them beats
    repeating it inside each, and a standalone bar can be placed and sized in the
    manuscript independently of the panels. `rect` is the bar's axes rectangle
    within the figure, leaving room for the tick labels and `label`."""
    figc = plt.figure(figsize=figsize)
    axc = figc.add_axes(rect)                     # [left, bottom, width, height]
    sm = mpl.cm.ScalarMappable(cmap=cmap,
                               norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = figc.colorbar(sm, cax=axc, orientation=orientation)
    if ticks is not None:
        cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
    if label:
        cbar.set_label(label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=labelsize)
    _save_fig(figc, out_path)


OM_MIN_EXPECTED = 5.0


def _om_point_mask(ga, om_idx, *, skip_input=(), skip_hidden=(), min_expected=OM_MIN_EXPECTED):
    """Mask stable OM blocks for one modulation cluster.

    Keeps only blocks whose expected surviving-synapse count under the OM null
    is at least `min_expected`. If older cached OM metadata lacks the expected
    count ingredients, falls back to the non-skipped grid.
    """
    n_in = int(ga["n_in"])
    n_hid = int(ga["n_hid"])
    mask = np.ones((n_in, n_hid), dtype=bool)
    if skip_input:
        mask[np.array(sorted(skip_input), dtype=int), :] = False
    if skip_hidden:
        mask[:, np.array(sorted(skip_hidden), dtype=int)] = False

    cluster_size_percent = ga.get("cluster_size_percent")
    n_active_block = ga.get("n_active_block")
    if cluster_size_percent is None or n_active_block is None:
        return mask, None

    cluster_size_percent = np.asarray(cluster_size_percent, dtype=float)
    n_active_block = np.asarray(n_active_block, dtype=float)
    if om_idx >= cluster_size_percent.shape[0]:
        return mask, None

    expected = n_active_block * cluster_size_percent[om_idx]
    mask &= expected >= float(min_expected)
    return mask, expected


OM_N_PERM = 1000


def _om_scatter_perm_test(mod_profiles, row_om, row_cm, n_perm=OM_N_PERM, seed=0):
    """Cluster-permutation p-value for the OM vs profile-L1 scatter.

    Mirrors multiple_task/leison_plot.py's helper of the same name (keep the
    two in sync): the scatter's (mod cluster, block) points are massively
    non-independent, so the parametric regression p is inflated. The null
    keeps every lesion effect fixed and permutes WHICH cluster owns WHICH OM
    footprint, recomputing the pooled Pearson r each time. One-sided toward
    negative r. Returns (r_obs, p_perm, null_r)."""
    nC = len(mod_profiles)

    def _pooled_r(assign):
        x = np.concatenate([row_om[k] for k in assign])
        y = np.concatenate([
            np.mean(np.abs(row_cm[k] - mod_profiles[c][None, :]), axis=1)
            for c, k in enumerate(assign)])
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    r_obs = _pooled_r(np.arange(nC))
    rng = np.random.default_rng(seed)
    null_r = np.array([_pooled_r(rng.permutation(nC)) for _ in range(n_perm)])
    finite = np.isfinite(null_r)
    if not np.isfinite(r_obs) or not finite.any():
        return r_obs, np.nan, null_r
    p_perm = (1.0 + np.sum(null_r[finite] <= r_obs)) / (finite.sum() + 1.0)
    return r_obs, float(p_perm), null_r


def _load_pkl_or_skip(pkl_path, hint="", use_name=False):
    """Return the unpickled object at `pkl_path`, or None (with a "Skipped"
    message) if it does not exist. `hint` is appended to the message (e.g.
    "Run one_task_analysis.py first."); `use_name` prints only the filename."""
    if not pkl_path.exists():
        shown = pkl_path.name if use_name else pkl_path
        print(f"  Skipped: {shown} not found.{(' ' + hint) if hint else ''}")
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and pkl_path.name.startswith(("fixed_points_grad_", "fixed_points_hidden_")):
        data["_fixed_point_source"] = str(pkl_path)
    return data


def _load_twotask_glob_or_skip(pattern):
    """Load the first pickle matching `pattern` under the configured two-task run
    dir, or return None (with a "Skipped" message) if none match."""
    run_dir = TWOTASKS_DIR / TWOTASK_ANAME
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        print(f"  Skipped: no {pattern} in {run_dir}. "
              f"Run two_task_analysis.py first.")
        return None
    return _load_pkl_or_skip(matches[0])


# ─── Paths & run identifiers ──────────────────────────────────────────────────
# All figure output goes here; the run identifiers ("aname") below select which
# trained run each mode's figures are drawn from. They are grouped by experiment
# family so a run can be swapped in one place.
OUT_DIR = Path("paper_plot")

# ── Multi-task (one full multi-task network) ──
ANAME = "everything_seed749_L21e4+hidden300+batch128+angle"
DATA_DIR = Path("multiple_tasks_analysis") / ANAME
# delayDM integration-memory probe (two_in_multiple mode). May differ from
# ANAME — set independently so the probe figure can come from a different
# seed/regularization than the clustering/lesion figures.
DELAYDM_ANAME = "everything_seed921_L21e4+hidden300+batch128+angle"
# Produced by multiple_task/sibling_delay_analysis.py, which writes into
# multiple_tasks_analysis/{aname}/ the two pickles the delayDM geometry figure reads:
#     fixed_points_grad_{aname}_{rule}.pkl   one per delayDM rule
#     delaydm1_delay_pc_projections_{aname}.pkl
#                                             joint six-PC delay-trajectory
#                                             coordinates of the fixed points

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

# ── Single-task vanilla RNN (the no-plasticity control) ──
# Trained by one_task/one_task_rnn.py and solved by one_task_rnn_analysis.py, which
# writes onetask_rnn/{aname}/fixed_points_hidden_{aname}.pkl in the SAME schema as
# the MPN's fixed_points_grad pickle — so the RNN 3D figure is rendered by the very
# same code, and any difference the reader sees is a difference in the networks,
# not in how the two were drawn.
ONETASK_RNN_DIR = Path("onetask_rnn")
ONETASK_RNN_ANAME = "delaygo_seed459_rnnL21e4+hidden200+batch128+angle"


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


def _read_twotask_dt(default=40):
    """Simulation time step (ms) for the configured two-task run, read from its
    saved param json (see SCHEME.md); used to relabel step-index x-axes in ms.
    Falls back to `default`."""
    try:
        import json as _json
        p = TWOTASKS_DIR / TWOTASK_ANAME / f"param_{TWOTASK_ANAME}_param.json"
        if p.exists():
            cfg = _json.load(open(p))
            return int(cfg.get("task_params", {}).get("dt", default))
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


def _read_onetask_dt(default=40):
    """Simulation time step (ms) for the configured one-task run, read from its
    saved param json (see SCHEME.md). Used to relabel step-index x-axes in ms for
    figures whose pickle predates the saved `dt` (e.g. the onetask_show traces)."""
    try:
        import json as _json
        p = ONETASK_DIR / f"param_{ONETASK_ANAME}_param.json"
        if p.exists():
            cfg = _json.load(open(p))
            return int(cfg.get("task_params", {}).get("dt", default))
    except Exception:
        pass
    return default

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


# ─── Sequential (dark → light) color scheme: interpolation level alpha ────────
# One sequential ramp, shared by every figure that traces fixed points across the
# pro<->anti interpolation level alpha (the task-interpolation figures). It
# deliberately replaces the stimulus rainbow there: in those figures the stimulus
# identity of an interpolated fixed point is not the claim — the DIRECTION of the
# sweep is — so a single universal dark(alpha=0) → light(alpha=1) scale carries the
# sweep and every stimulus line looks alike.
#
# A perceptually-uniform MULTI-HUE map (plasma: dark indigo → magenta → orange)
# rather than one hue's dark→light: it still lightens monotonically, so the sweep
# direction is unambiguous, but neighboring alphas are far easier to tell apart than
# in a monochrome ramp. The FULL plasma range is used, so alpha=1 lands on
# plasma's near-white yellow and the ramp spans the widest luminance it has
# (0.07 -> 0.91, against 0.08 -> 0.71 for the (0.03, 0.82) truncation used
# earlier). The known cost: HOLLOW (non-converged) markers are drawn in the ramp
# color as an outline only, so a hollow point near alpha=1 is close to invisible
# on white. Kept anyway because in the runs these figures are made from every
# point converges (88/88 per period/representation on seed21), and the wider ramp
# is what makes the sweep direction unmistakable. If a run does produce
# non-converged points at high alpha, give the hollow markers their own truncated
# stroke color rather than pulling this range back in — that keeps both.
_ALPHA_CMAP = "plasma"
_ALPHA_CMAP_RANGE = (0.0, 1.0)           # dark indigo (alpha=0) → yellow (alpha=1)


def _alpha_ramp_color(t):
    """Color at position `t` in [0, 1] along the alpha ramp (0 = dark, 1 = light)."""
    lo, hi = _ALPHA_CMAP_RANGE
    return mpl.colormaps[_ALPHA_CMAP](lo + (hi - lo) * float(np.clip(t, 0.0, 1.0)))


def _alpha_ramp_norm(alphas):
    """Map an alpha sweep to [0, 1] ramp positions (flat sweep → all 0)."""
    a = np.asarray(alphas, dtype=float)
    return (a - a.min()) / max(a.max() - a.min(), 1e-9)


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


# Filename prefix for the MULTI-TASK figures, matching what the one-task and
# two-task families already do (onetask_*, twotask_*): every figure in the
# `multiple_tasks` mode is named through `_multitask_out`, so a listing of
# paper_plot/ groups by experiment family instead of scattering these among the
# others. Change the prefix here and every multi-task figure follows.
MULTITASK_PREFIX = "multitask"


def _multitask_out(name):
    """OUT_DIR path for a multi-task figure, with the shared `multitask_` prefix."""
    return OUT_DIR / f"{MULTITASK_PREFIX}_{name}"


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
    "reactgo": "ReactPro",
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

# Task → computation-category motif and color. This table is the paper-side
# source of truth for motif colors: the state-space panels color from it too
# (the pickle's rule_motif_mapping supplies data, not colors), and
# state_space_shift.py mirrors these hexes for its own analysis-side figures.
# The dms pair keeps the go/anti pairing DELIBERATELY: ReactMatch2Sample
# (dmsgo) shares Pro Reaction's green and ReactNonMatch2Sample (dmsnogo)
# shares Anti Reaction's orange — match/non-match is a pro/anti response
# rule, so the label color carries that. Only the dmc pair takes
# Categorization's own deeppink. NB the state-space scatter colors by
# CATEGORY (dict last-wins => Categorization = deeppink there), so dms
# points are deeppink in that figure while their heatmap labels stay paired.
_RULE_MOTIF = {
    "fdgo":            ("Pro Delayed",    "#3182ce"),  # blue
    "fdanti":          ("Anti Delayed",   "#e53e3e"),  # red
    "delaygo":         ("Pro Delayed",    "#3182ce"),
    "delayanti":       ("Anti Delayed",   "#e53e3e"),
    "reactgo":         ("Pro Reaction",   "#38a169"),  # green
    "reactanti":       ("Anti Reaction",  "#dd6b20"),  # orange
    "contextdelaydm1": ("Pro Integration", "#805ad5"),  # purple
    "contextdelaydm2": ("Pro Integration", "#805ad5"),
    "delaydm1":        ("Pro Integration", "#805ad5"),
    "delaydm2":        ("Pro Integration", "#805ad5"),
    "multidelaydm":    ("Pro Integration", "#805ad5"),
    "dmsgo":           ("Categorization", "#38a169"),  # green, pairs reactgo
    "dmsnogo":         ("Categorization", "#dd6b20"),  # orange, pairs reactanti
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
    "stim1": "#c3b1e1",   # light purple     (L* 75)
    "stim2": "#9c79d8",   # medium purple    (L* 58)
    "delay1": "#bbf7d0",  # light green      (L* 92)
    "delay2": "#4dcb79",  # medium green     (L* 73)
    "go1": "#d1d5db",     # light gray       (L* 85)
}
# The two stimulus epochs SHARE a hue and the two memory epochs share another, so a
# reader sees at a glance that Stimulus 2 is another *stimulus* epoch rather than a
# different kind of thing — which the original palette (stim2 light blue, delay2
# light orange) hid by giving every epoch an unrelated hue.
#
# EACH PAIR SPANS TWO BRIGHTNESS TIERS, darker = later: stim2/delay2 are the deep
# members (L* 58/73) under their pastel partners (L* 75/92). An earlier attempt put
# the pairs on two tiers by darkening stim1/delay1 instead; that was reverted,
# because stim1/delay1 are also the published one-task period-bar pastels (see NB
# below). Darkening the "2" epochs restores the separation without touching them:
#   * stim1 vs stim2 is 33 CIELAB and delay1 vs delay2 is 37 (up from 9.7 / 23 when
#     all five sat on one light tier), so each pair is told apart by color, not
#     just by position in the heatmap.
#   * both pairs now agree on direction — the second epoch is the darker one — so
#     "darker = later" carries from one pair to the other.
#
# The two constraints that DO still hold, and must keep holding: these colors are
# the background highlight behind black tick labels (_color_phase_ticklabels,
# _PERIOD_LABEL_COLORS), and every one clears 5:1 black-text contrast (the new dark
# members are the low ends at 6.1:1 stim2 and 10.1:1 delay2; the pastels sit at
# 10.7-17.3:1). And the cluster strip shares the figure, so each is kept >= ~26
# CIELAB from every `_CLUSTER_COLORS` entry (34.7 stim2 vs the mauve, 35.4 delay2
# vs the moss green) and from the stimulus rainbow (26.6 stim2, 29.9 delay2).
#
# NB stim1 and delay1 are also `_ONETASK_PERIOD_COLORS[1:3]`, the Stimulus and Memory
# blocks of the one-task / two-task period bars (and cartoon.py's period strip, which
# imports them) — one definition drives both, which is why the dark tier lives on
# stim2/delay2 and stim1/delay1 must stay on the published pastels.

# Period colorbar palette for the one-task / two-task period strip, ordered
# Context → Stimulus → Memory → Response. Stimulus/Memory/Response reuse the
# multi-task heatmap phase colors (_PHASE_COLORS stim1/delay1/go1) so the period
# bar is color-consistent with those figures; Context has no heatmap
# counterpart, so it gets a new pastel yellow in the same soft-pastel family.
_ONETASK_PERIOD_COLORS = [
    "#fef08a",                  # Context — new (pale yellow)
    _PHASE_COLORS["stim1"],     # Stimulus — matches heatmap Stimulus 1 (light purple)
    _PHASE_COLORS["delay1"],    # Memory   — matches heatmap Memory 1 (light green)
    _PHASE_COLORS["go1"],       # Response — matches heatmap Response (light gray)
]

# Trial-period display vocabulary: the FIRST epoch is called "Context", never
# "Fixation" — it is where the rule cue establishes which computation the trial
# demands, and naming it after the incidental fixation signal describes the wrong
# thing. The canonical order is Context → Stimulus → Memory (Delay) → Response.
#
# This renames PERIODS ONLY. The input/output channel called "Fixation" (the
# fixation cue itself, `_IO_FIXATION`, the fixon trace, the readout's fixation
# channel) keeps its name: it is a signal, not an epoch. Internal keys are
# untouched too ("longfixation", "fix1", the "fixation" dict keys), because they
# index saved pickles — which is exactly why the mapping happens at DISPLAY time:
# pickles solved before this convention still carry period_title="Fixation", and
# `_period_display` normalizes them on the way to the axes.
_PERIOD_DISPLAY_RENAMES = {
    "fixation": "Context",
    "longfixation": "Context",
}


def _period_display(name):
    """Display name for a trial period, in the project's period vocabulary.

    Pass anything that is about to become a title / legend entry / tick label for
    a PERIOD; non-period strings and already-correct names pass through unchanged.
    """
    if not isinstance(name, str):
        return name
    return _PERIOD_DISPLAY_RENAMES.get(name.strip().lower(), name)


# ─── Input / output channel colors ────────────────────────────────────────────
# Colors for the example-trial input and output traces. A muted qualitative set,
# deliberately distinct from the vivid stimulus rainbow (stim_color) and the pale
# period-bar pastels (_ONETASK_PERIOD_COLORS). Within a modality the cos/sin
# channels share a hue as a (dark, light) pair. Fixation↔Fixation shares a color
# across the input and output figures. The response Cos/Sin get their OWN hue
# (purple), deliberately distinct from the stimulus modalities so the readout is
# not confused with an input modality.
_IO_FIXATION = "#555555"              # dark gray
_IO_FIXATION2 = "#aaaaaa"             # light gray = fixation-off (pairs with above)
# Mod1 and Mod2 share the SAME green cos/sin pair, so cos↔cos and sin↔sin match
# across the two stimulus modalities (they are the same physical channel, just a
# different modality). Within the pair the cos/sin keeps the (dark, light)
# convention.
_IO_MOD2 = ("#1b9e77", "#6fceae")     # green  (cos dark, sin light) = stimulus cos/sin
_IO_MOD1 = _IO_MOD2                   # Modality 1 shares Modality 2's cos/sin colors
_IO_TASK = "#d95f02"                  # orange (single channel)
_IO_TASK2 = "#fdae6b"                 # light orange = second (inactive) task cue
_IO_RESPONSE = ("#7e3ff2", "#c4a3f5")  # purple (cos dark, sin light) = readout
# Sum of the component traces in the cancellation figures (onetask_show,
# twotask_cancel). A deep blue: the classic high-contrast partner of the orange
# task cue, and far from the grays (fixation), the green (stimulus) and the
# purple (readout), so the emphasized "Combine" line never reads as a component.
_IO_COMBINE = "#2166ac"               # deep blue = Fix + Task (+ bias) sum


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


# Trial-period name -> period-bar color, keyed by the period word that ends a
# "{task} {period}" tick label. Covers both the one-task period names
# (Context/Stimulus/Memory/Response) and the abbreviated two-task ones
# (Context/Stim/Delay/Resp), so a tick labeled e.g. "Anti Stim" gets the same
# color as the Stimulus block of the input/output illustration's period strip.
_PERIOD_LABEL_COLORS = {
    "fixation": _ONETASK_PERIOD_COLORS[0],
    "context": _ONETASK_PERIOD_COLORS[0],
    "stimulus": _ONETASK_PERIOD_COLORS[1],
    "stim": _ONETASK_PERIOD_COLORS[1],
    "memory": _ONETASK_PERIOD_COLORS[2],
    "delay": _ONETASK_PERIOD_COLORS[2],
    "response": _ONETASK_PERIOD_COLORS[3],
    "resp": _ONETASK_PERIOD_COLORS[3],
    "go": _ONETASK_PERIOD_COLORS[3],
}


def _period_label_color(label):
    """Period-bar color for a '{task} {period}' tick label (None if unknown)."""
    word = str(label).replace("\n", " ").strip().split(" ")[-1].lower()
    return _PERIOD_LABEL_COLORS.get(word)


def _color_period_ticklabels(ax, labels, axis="y"):
    """Highlight each tick label with its trial period's color.

    Same background-bbox treatment as `_color_phase_ticklabels`, but keyed on
    the period word of a '{task} {period}' label (e.g. "Pro Stim") and using the
    period-bar palette shared with the input/output illustration figure.
    """
    texts = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for text, label in zip(texts, labels):
        color = _period_label_color(label)
        if color is not None:
            text.set_bbox(dict(facecolor=color, edgecolor="none",
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


# ─── Multi-task heatmap color scale ───────────────────────────────────────────
# The input / hidden / modulation task-variance heatmaps all show the SAME
# quantity on the SAME scale, so the scale is defined once here and the colorbar is
# a figure of its own (`plot_multitask_heatmap_colorbar`) rather than a strip
# repeated inside each panel — the same split the one-task modulation snapshots use
# (_plot_onetask_snapshot_single + _onetask_hcbar). Change the scale here and the
# heatmaps and their colorbar move together.
_MULTITASK_HEATMAP_CMAP = "viridis"
_MULTITASK_HEATMAP_VLIM = (0.0, 1.0)
_MULTITASK_HEATMAP_CLABEL = "Normalized variance"


# ─── Cluster (neuron-class) strip colors ──────────────────────────────────────
# The multi-task heatmaps mark each contiguous row/column cluster with a barcode
# block. Alternating light and dark gray is sufficient to show every boundary and
# keeps categorical color from competing with the heatmap's quantitative scale.
_CLUSTER_COLORS = [
    "#bdbdbd",   # light gray
    "#4d4d4d",   # dark gray
]


def _cluster_color(ci):
    """Color for cluster index `ci`, cycling `_CLUSTER_COLORS`."""
    return _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]


def _add_col_cluster_strip(ax, cl_ordered, cbreaks):
    """Add a thin colored strip below the heatmap, one color per column cluster,
    to visually group the x-axis columns by their cluster assignment."""
    n_cols = len(cl_ordered)
    # Cluster boundaries as [start, end) spans
    bounds = [0] + list(cbreaks) + [n_cols]
    n_clusters = len(bounds) - 1

    strip = ax.inset_axes([0, -0.06, 1, 0.04], transform=ax.transAxes)
    for ci in range(n_clusters):
        start, end = bounds[ci], bounds[ci + 1]
        strip.axvspan(start, end, color=_cluster_color(ci), lw=0)
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
    for ci in range(n_clusters):
        start, end = bounds[ci], bounds[ci + 1]
        strip.axhspan(start, end, color=_cluster_color(ci), lw=0)
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
    title="", cmap=_MULTITASK_HEATMAP_CMAP,
    vmin=_MULTITASK_HEATMAP_VLIM[0], vmax=_MULTITASK_HEATMAP_VLIM[1],
    figsize=(8.4, 7),          # 5% wider than the original 8 in
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

    # No colorbar: all three heatmaps share one scale, so it is published once as
    # its own figure (plot_multitask_heatmap_colorbar) instead of three times here.
    sns.heatmap(ordered, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)

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

    out_path = _multitask_out("clustered_input_normalized.png")
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

    out_path = _multitask_out("clustered_hidden_normalized.png")
    _save_fig(fig, out_path)


# ─── Figure: Clustered modulation variance matrix ────────────────────────────

def _load_cluster_info_mod():
    """Load the modulation cluster_info pickle for the target model, or None if
    it has not been produced yet (callers print their own "Skipped" message)."""
    pkl_path = DATA_DIR / f"cluster_info_mod_{ANAME}.pkl"
    if not pkl_path.exists():
        return None
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
    if mod_info is None:
        print(f"  Skipped: cluster_info_mod_{ANAME}.pkl not found. "
              "Run multiple_task_analysis.py first.")
        return
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

    # No colorbar here either — see plot_multitask_heatmap_colorbar.
    sns.heatmap(ordered, ax=ax, cmap=_MULTITASK_HEATMAP_CMAP,
                vmin=_MULTITASK_HEATMAP_VLIM[0], vmax=_MULTITASK_HEATMAP_VLIM[1],
                cbar=False)

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

    out_path = _multitask_out("clustered_modulation_normalized.png")
    _save_fig(fig, out_path)


def plot_multitask_heatmap_colorbar():
    """
    The colorbar for the multi-task task-variance heatmaps (input, hidden and
    modulation), as its own figure.

    Those three panels show the same quantity on the same scale
    (`_MULTITASK_HEATMAP_CMAP` over `_MULTITASK_HEATMAP_VLIM`), so none of them
    draws a colorbar: one bar published once serves all three, is not repeated
    three times at three different panel widths, and can be placed and sized in the
    manuscript independently of the panels. Reads no data — the scale is fixed.
    """
    _ensure_out_dir()
    vmin, vmax = _MULTITASK_HEATMAP_VLIM
    _save_standalone_colorbar(
        _multitask_out("heatmap_colorbar.png"),
        cmap=_MULTITASK_HEATMAP_CMAP, vmin=vmin, vmax=vmax,
        ticks=[vmin, vmax], ticklabels=[f"{vmin:.0f}", f"{vmax:.0f}"],
        label=_MULTITASK_HEATMAP_CLABEL,
        # Horizontal: the bar sits in the upper part of the figure, leaving the
        # space below it for the tick labels and then the axis label.
        orientation="horizontal", figsize=(1.9, 0.62),
        rect=(0.05, 0.62, 0.90, 0.24), labelsize=8, label_fontsize=8)


# ─── Figure: L2 vs Accuracy ──────────────────────────────────────────────────

PERF_RESULT_PATH = Path("multiple_tasks_perf") / "performance_results.json"


def _performance_feature_tag(model_name, result):
    """Return a performance entry's full feature tag, with legacy fallback."""
    import re as _re

    feature = result.get("feature")
    if feature is not None:
        return feature
    match = _re.search(r"_(L2[^+]*)\+hidden\d+", model_name)
    return match.group(1) if match else None


def plot_l2_vs_accuracy():
    """Figure: Tanh-model test accuracy (%) vs L2 regularization strength."""
    import json as _json
    import re as _re

    _ensure_out_dir()
    if not PERF_RESULT_PATH.exists():
        print(f"  Skipped: {PERF_RESULT_PATH} not found. Run multiple_task_performance.py first.")
        return

    with open(PERF_RESULT_PATH) as f:
        result_dict = _json.load(f)

    # A bare L2 tag denotes the default Tanh activation.  Exact matching
    # deliberately excludes activation experiments such as L21e4relu and
    # L21e4sigmoid from the regularization-strength comparison.
    tanh_results = [
        result
        for model_name, result in result_dict.items()
        if _re.fullmatch(
            r"L2\d+(?:\.\d+)?e\d+",
            _performance_feature_tag(model_name, result) or "",
        )
    ]
    if not tanh_results:
        print("  Skipped: no Tanh performance results with a bare L2 feature tag.")
        return

    l2_vals = np.array([e["l2_info"] for e in tanh_results])
    acc_vals = np.array([e["acc"] for e in tanh_results]) * 100

    fig, ax = plt.subplots(1, 1, figsize=(2.3, 3))
    ax.scatter(l2_vals, acc_vals, color="#3182ce", edgecolors="k",
               linewidths=0.5, s=40, alpha=0.8, zorder=3)

    # Overlay the across-seed mean at each L2 value and connect those means to
    # make the regularization trend visible without hiding the individual runs.
    unique_l2 = np.unique(l2_vals)
    mean_acc = np.array([acc_vals[l2_vals == l2].mean() for l2 in unique_l2])
    ax.plot(
        unique_l2, mean_acc, color="k", linewidth=1.2, marker="D",
        markerfacecolor="white", markeredgecolor="k", markeredgewidth=0.8,
        markersize=4, zorder=4,
    )
    ax.set_xscale("log")
    ax.set_xlabel("L2 regularization strength")
    ax.set_ylabel("Test accuracy (%)")
    # Adaptive y-range: pad the observed accuracy span by 5% of its extent
    # (min 2 pts), clamped to [0, 100], then snapped outward to multiples of 5
    # so the 5%-interval ticks land on the axis edges.
    lo, hi = float(acc_vals.min()), float(acc_vals.max())
    pad = max((hi - lo) * 0.05, 2.0)
    ax.set_ylim(np.floor(max(0.0, lo - pad) / 5.0) * 5.0,
                np.ceil(min(100.0, hi + pad) / 5.0) * 5.0)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)

    fig.tight_layout()
    out_path = _multitask_out("l2_vs_accuracy.png")
    _save_fig(fig, out_path, extra=f" (Tanh n={len(tanh_results)})")


# ─── Figure: L2=1e-4 activation comparison ──────────────────────────────────

def plot_l2e4_activation_accuracy():
    """Figure: Test accuracy at L2=1e-4 for every trained activation function.

    Compares linear, ReLU, softplus, sigmoid, and the default Tanh.  The full
    feature tag is read from each result's ``feature`` field when available and
    otherwise inferred from its model identifier.  The unadorned ``L21e4`` tag
    denotes the default Tanh activation.
    """
    import json as _json
    _ensure_out_dir()
    if not PERF_RESULT_PATH.exists():
        print(f"  Skipped: {PERF_RESULT_PATH} not found. Run multiple_task_performance.py first.")
        return

    with open(PERF_RESULT_PATH) as f:
        result_dict = _json.load(f)

    # Categorical distinction → c_vals (SCHEME.md); Tanh keeps the same blue
    # as the tanh points in plot_l2_vs_accuracy (c_vals[1]).
    group_specs = [
        ("L21e4linear", "Linear", c_vals[2]),
        ("L21e4relu", "ReLU", c_vals[6]),
        ("L21e4softplus", "Softplus", c_vals[3]),
        ("L21e4sigmoid", "Sigmoid", c_vals[9]),
        ("L21e4", "Tanh", c_vals[1]),
    ]
    accuracies = {feature: [] for feature, _, _ in group_specs}

    for model_name, result in result_dict.items():
        feature = _performance_feature_tag(model_name, result)
        if feature in accuracies:
            accuracies[feature].append(float(result["acc"]) * 100.0)

    # Drop activations with no evaluated checkpoints instead of skipping the
    # whole figure, so a partially populated database still plots.
    missing = [feature for feature, _, _ in group_specs if not accuracies[feature]]
    if missing:
        print(f"  Note: no performance results for {', '.join(missing)}; omitted.")
        group_specs = [spec for spec in group_specs if spec[0] not in missing]
    if not group_specs:
        print("  Skipped: no L2=1e-4 activation results found.")
        return

    # Same per-column width as plot_l2_vs_accuracy (2.3 in / 4 L2 values).
    fig, ax = plt.subplots(1, 1, figsize=(0.575 * len(group_specs), 3))
    positions = np.arange(len(group_specs))

    all_values = []
    for x, (feature, _, color) in zip(positions, group_specs):
        values = np.asarray(accuracies[feature], dtype=float)
        all_values.extend(values.tolist())
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(
            x + jitter, values, color=color, edgecolors="k", linewidths=0.5,
            s=40, alpha=0.8, zorder=3,
        )
        ax.errorbar(
            x, values.mean(), yerr=values.std(), fmt="D", color="k",
            markerfacecolor="white", markeredgewidth=0.8, markersize=4,
            capsize=3, linewidth=1.0, zorder=4,
        )

    labels = [label for _, label, _ in group_specs]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Activation function")
    ax.set_ylabel("Test accuracy (%)")
    lo, hi = min(all_values), max(all_values)
    pad = max((hi - lo) * 0.08, 2.0)
    ax.set_ylim(np.floor(max(0.0, lo - pad) / 5.0) * 5.0,
                np.ceil(min(100.0, hi + pad) / 5.0) * 5.0)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)

    fig.tight_layout()
    out_path = _multitask_out("l2e4_activation_accuracy.png")
    counts = ", ".join(
        f"{label} n={len(accuracies[feature])}"
        for feature, label, _ in group_specs
    )
    _save_fig(fig, out_path, extra=f" ({counts})")


# ─── Figure: L2=1e-4 projection-dimension comparison ─────────────────────────

def _projection_accuracy_groups(result_dict, config_dir=Path("multiple_tasks"), *, vary_hidden=False,
                                per_task=False):
    """Group saved widths at Tanh/L2=1e-4, fixing the other width to 300.

    By default vary projection at hidden300; vary_hidden selects hidden widths
    at projection300. Only single-hidden-layer configurations are included.
    Model identifiers locate configuration files only; feature tags and cached
    filename-derived dimensions are not used to select or group runs.
    per_task returns dimension -> task name -> accuracy list; missing or invalid
    task scores are omitted rather than replaced by overall accuracy or zero.
    """
    import json as _json

    accuracies = {}
    for model_name, result in result_dict.items():
        config_path = config_dir / f"param_{model_name}_param.json"
        try:
            with config_path.open() as handle:
                config = _json.load(handle)
            net_params = config["net_params"]
            train_params = config["train_params"]
            hidden_dims = net_params["n_neurons"][1:-1]
            if (len(hidden_dims) != 1
                    or net_params["activation"] != "tanh"
                    or train_params["weight_reg"] != "L2"
                    or not np.isclose(float(train_params["reg_lambda"]), 1e-4,
                                      rtol=1e-9, atol=0)):
                continue
            if not net_params["input_layer_add"]:
                continue
            dimension = net_params["linear_embed"]
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise ValueError("linear_embed must be a positive integer")
            hidden_dim = hidden_dims[0]
            if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
                raise ValueError("hidden dimension must be a positive integer")
            if vary_hidden:
                if dimension != 300:
                    continue
                dimension = hidden_dim
            elif hidden_dim != 300:
                continue
            accuracy = float(result["acc"])
            if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("accuracy must be finite and in [0, 1]")
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"  Note: skipped {model_name}: invalid or missing metadata/result ({error}).")
            continue
        if per_task:
            task_groups = accuracies.setdefault(dimension, {})
            task_scores = result.get("acc_per_task")
            if not isinstance(task_scores, dict):
                print(f"  Note: {model_name} has no acc_per_task; rerun multiple_task_performance.py.")
                continue
            for task, score in task_scores.items():
                if score is None:
                    continue
                try:
                    value = float(score)
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("task accuracy must be finite and in [0, 1]")
                except (ValueError, TypeError) as error:
                    print(f"  Note: skipped {model_name}/{task}: {error}.")
                    continue
                task_groups.setdefault(task, []).append(value * 100.0)
        else:
            accuracies.setdefault(dimension, []).append(accuracy * 100.0)
    return dict(sorted(accuracies.items()))


def plot_projection_dim_accuracy():
    """Plot accuracy vs metadata projection width at hidden300/Tanh/L2=1e-4.

    Read each evaluated run's configuration JSON and include all matching widths,
    regardless of feature naming. Use evenly spaced positions labeled by width.
    """
    _plot_dimension_accuracy(vary_hidden=False)


def plot_hidden_dim_accuracy():
    """Plot accuracy vs saved hidden width at projection300/Tanh/L2=1e-4."""
    _plot_dimension_accuracy(vary_hidden=True)


def plot_projection_dim_task_accuracy():
    """Plot stacked task panels vs projection width at hidden300/Tanh/L2=1e-4."""
    _plot_dimension_task_accuracy(vary_hidden=False)


def plot_hidden_dim_task_accuracy():
    """Plot stacked task panels vs hidden width at projection300/Tanh/L2=1e-4."""
    _plot_dimension_task_accuracy(vary_hidden=True)


def _plot_dimension_task_accuracy(*, vary_hidden):
    """Show 15 compact task panels with seed dots and connected seed means.

    Use paper task names/order and index positions labeled by actual dimensions.
    Each panel has an adaptive y range. Missing task/dimension pairs remain NaN;
    tasks without any scores retain an empty panel rather than disappearing.
    """
    import json as _json

    _ensure_out_dir()
    if not PERF_RESULT_PATH.exists():
        print(f"  Skipped: {PERF_RESULT_PATH} not found. Run multiple_task_performance.py first.")
        return
    with PERF_RESULT_PATH.open() as handle:
        result_dict = _json.load(handle)
    groups = _projection_accuracy_groups(result_dict, vary_hidden=vary_hidden, per_task=True)
    if not any(groups.values()):
        print("  Skipped: no metadata-matched per-task accuracies; rerun multiple_task_performance.py.")
        return
    fig, axes = plt.subplots(len(_TASK_DISPLAY), 1, figsize=(3.5, 11.5), sharex=True)
    positions = np.arange(len(groups))
    for ax, (task, display_name) in zip(axes, _TASK_DISPLAY.items()):
        entries = [tasks.get(task, []) for tasks in groups.values()]
        means = [float(np.mean(values)) if values else np.nan for values in entries]
        for position, values in zip(positions, entries):
            if values:
                ax.scatter(np.full(len(values), position), values, color=c_vals[1],
                           edgecolors="k", linewidths=0.4, s=12, alpha=0.8, zorder=3)
        ax.plot(positions, means, color="k", linewidth=1.0, marker="D",
                markerfacecolor="white", markeredgecolor="k", markeredgewidth=0.6,
                markersize=3, zorder=4)
        ax.set_title(display_name, loc="left", fontsize=7, pad=2)
        all_values = [value for values in entries for value in values]
        if all_values:
            lo, hi = min(all_values), max(all_values)
            pad = max((hi - lo) * 0.08, 2.0)
            ax.set_ylim(np.floor(max(0.0, lo - pad) / 5.0) * 5.0,
                        np.ceil((hi + pad) / 5.0) * 5.0)
            locator = mpl.ticker.MaxNLocator(nbins=2, steps=[1, 2, 5, 10])
            ticks = locator.tick_values(*ax.get_ylim())
            ax.set_yticks([tick for tick in ticks if 0 <= tick <= 100
                          and ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]])
        else:
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 100])
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="0.5")
        ax.tick_params(axis="both", labelsize=6, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)
        print(f"  {display_name}: " + ", ".join(f"{dimension} n={len(values)}"
                                       for dimension, values in zip(groups, entries)))
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([str(dimension) for dimension in groups])
    axes[-1].set_xlabel("Hidden dimension" if vary_hidden else "Projection dimension")
    fig.supylabel("Task test accuracy (%)", x=0.02, fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.045, hspace=0.9)
    dimension_name = "hidden" if vary_hidden else "projection"
    _save_fig(fig, _multitask_out(f"{dimension_name}_dim_task_accuracy.png"))


def _plot_dimension_accuracy(*, vary_hidden):
    """Render either width sweep at index positions labeled by actual dimensions."""
    import json as _json
    _ensure_out_dir()
    if not PERF_RESULT_PATH.exists():
        print(f"  Skipped: {PERF_RESULT_PATH} not found. Run multiple_task_performance.py first.")
        return

    with open(PERF_RESULT_PATH) as f:
        result_dict = _json.load(f)

    accuracies = _projection_accuracy_groups(result_dict, vary_hidden=vary_hidden)
    dimension_name = "hidden" if vary_hidden else "projection"
    fixed_name = "projection300" if vary_hidden else "hidden300"
    if not accuracies:
        print(f"  Skipped: no metadata-matched {fixed_name}/Tanh/L2=1e-4 {dimension_name} results.")
        return

    # Same layout as plot_l2_vs_accuracy: individual seeds as scatter, the
    # across-seed means connected to show the trend over the varying dimension.
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    positions = np.arange(len(accuracies))
    mean_acc = []
    all_values = []
    for position, entries in zip(positions, accuracies.values()):
        values = np.asarray(entries, dtype=float)
        all_values.extend(values.tolist())
        mean_acc.append(values.mean())
        ax.scatter(
            np.full(len(values), position), values, color=c_vals[1], edgecolors="k",
            linewidths=0.5, s=40, alpha=0.8, zorder=3,
        )
    ax.plot(
        positions, mean_acc, color="k", linewidth=1.2, marker="D",
        markerfacecolor="white", markeredgecolor="k", markeredgewidth=0.8,
        markersize=4, zorder=4,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(dimension) for dimension in accuracies])
    ax.set_xlabel("Hidden dimension" if vary_hidden else "Projection dimension")
    ax.set_ylabel("Test accuracy (%)")
    lo, hi = min(all_values), max(all_values)
    pad = max((hi - lo) * 0.08, 2.0)
    ax.set_ylim(np.floor(max(0.0, lo - pad) / 5.0) * 5.0,
                np.ceil(min(100.0, hi + pad) / 5.0) * 5.0)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)

    fig.tight_layout()
    out_path = _multitask_out(f"{dimension_name}_dim_accuracy.png")
    count_prefix = "hidden" if vary_hidden else "proj"
    counts = ", ".join(
        f"{count_prefix}{dimension} n={len(entries)}" for dimension, entries in accuracies.items()
    )
    _save_fig(fig, out_path, extra=f" ({counts})")


# ─── Figure: State space PCA ─────────────────────────────────────────────────

STATE_SPACE_DIR = Path("state_space")


def _load_state_space_pca():
    """Load the PCA pickle for the target model."""
    pattern = f"state_space_pca_{ANAME}_noise*.pkl"
    matches = list(STATE_SPACE_DIR.glob(pattern))
    if not matches:
        return None
    return pickle.load(open(matches[0], "rb"))


# Legend for the state-space panels: one entry per DISTINCT color in
# _RULE_MOTIF, labeled by what that color means across figures (dms shares
# the Reaction hues by the match=pro / non-match=anti pairing, so green and
# orange cover both).
_STATE_SPACE_LEGEND = [
    ("Pro Delayed", "#3182ce"),
    ("Anti Delayed", "#e53e3e"),
    ("Pro Reaction / Match", "#38a169"),
    ("Anti Reaction / Non-match", "#dd6b20"),
    ("Integration", "#805ad5"),
    ("Category", "#ff1493"),
]


def _plot_state_space_panel(data, key, ylabel_prefix, out_name):
    """One context-end PCA panel as its own figure, colored PER TASK from
    _RULE_MOTIF — the single color mapping every figure shares (same colors
    as the lesion-heatmap task labels). The pickle's rule_motif_mapping
    supplies data only, so recoloring never requires re-running
    state_space_shift.py."""
    all_rules = data["all_rules"]
    pca = data["pca_results"][key]
    X_2d, ctx_rule_labels = pca["X_2d"], pca["ctx_rule_labels"]

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.4))
    for idx, rule in enumerate(all_rules):
        sel = ctx_rule_labels == idx
        ax.scatter(X_2d[sel, 0], X_2d[sel, 1],
                   color=_RULE_MOTIF[rule][1], alpha=0.5, s=14,
                   edgecolors="none")

    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel(f"{ylabel_prefix}\nPC2", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    # Each figure stands alone, so each carries the color key (routed through
    # _legend, so --no-legend still suppresses it). Handles are built from the
    # shared color list, not the scatters, so one entry per color.
    handles = [mpl.lines.Line2D([0], [0], marker="o", linestyle="",
                                color=col, alpha=0.5, markersize=4, label=lab)
               for lab, col in _STATE_SPACE_LEGEND]
    _legend(ax, handles=handles, frameon=True, loc="best", fontsize=5,
            markerscale=1.0)

    fig.tight_layout()
    _save_fig(fig, _multitask_out(out_name))


def plot_state_space_combined():
    """
    Figures: context-end PCA colored by task category — hidden state and effective
    modulation, as TWO separate figures:
      multitask_state_space_hidden.png
      multitask_state_space_eff_mod.png

    They were previously one stacked 2-panel figure. Split because the two panels
    are separate PCA spaces whose PCs are not comparable, so nothing was gained by
    forcing them onto a shared x axis and into one layout — and each is now sized
    and placed independently. Each carries its own axis labels and category legend.
    """
    _ensure_out_dir()
    data = _load_state_space_pca()
    if data is None:
        print("  Skipped: state_space PCA pickle not found. Run state_space_shift.py first.")
        return

    for key, ylabel_prefix, out_name in (
        ("hidden", "Hidden state", "state_space_hidden.png"),
        ("eff_mod", "Eff. modulation", "state_space_eff_mod.png"),
    ):
        _plot_state_space_panel(data, key, ylabel_prefix, out_name)


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
    out_path = _multitask_out("state_space_r_values.png")
    _save_fig(fig, out_path)


def plot_state_space_dist_angle():
    """
    Figure: initial-condition distance vs first-step trajectory angle for the
    paper seed (ANAME) — the per-task-pair scatter that the R-value bars
    (plot_state_space_r_values) summarize.

    Two panels (hidden state, effective modulation). Each point is one task
    pair: x = mean Euclidean distance between the two tasks' pre-stimulus
    states (end of the Context period, matched by stimulus), y = mean angle
    (deg) between their first post-stimulus displacement vectors. The line and
    the annotated r/slope are the through-origin fit state_space_shift.py
    computed — read from the same pickle, never refit here, so this figure and
    the R-value bars cannot drift apart. Needs the raw scatter data
    state_space_shift.py now saves; older pickles (r-values only) are skipped
    with a message to re-run it.
    """
    _ensure_out_dir()
    result_dict = _load_pkl_or_skip(RVAL_RESULT_PATH, "Run state_space_shift.py first.")
    if result_dict is None:
        return
    entry = result_dict.get(ANAME)
    if entry is None:
        print(f"  Skipped: {ANAME} not in {RVAL_RESULT_PATH.name}.")
        return
    scatter = entry.get("scatter")
    if not scatter:
        print("  Skipped: pickle has no raw scatter data (older format stored "
              "only the r-values) — re-run state_space_shift.py.")
        return

    panels = [("hidden", "Hidden state"), ("eff_mod", "Eff. modulation")]
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.6))
    for ax, (key, title) in zip(axs, panels):
        sd = scatter.get(key)
        if sd is None:
            ax.set_visible(False)
            continue
        x = np.asarray(sd["dists"], float)
        y = np.asarray(sd["angles_deg"], float)
        r_value, slope, p_value = entry["rval_dict"][key]

        ax.scatter(x, y, color="#3182ce", edgecolors="k", linewidths=0.4,
                   s=22, alpha=0.75, zorder=3)
        x_fit = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_fit, slope * x_fit, color="tomato", linewidth=1.2, zorder=4)

        p_str = "p < 1e-4" if p_value < 1e-4 else f"p = {p_value:.3f}"
        _legend(ax, [f"r = {r_value:.2f}, {p_str}"], loc="lower right",
                fontsize=6, frameon=True)
        ax.set_xlabel("Initial-condition distance", fontsize=8)
        ax.set_ylabel("First-step angle (deg.)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save_fig(fig, _multitask_out("state_space_dist_angle.png"))


# ─── Figure: Over-membership ─────────────────────────────────────────────────

def _find_experiment_dirs():
    """Return all experiment subfolders under multiple_tasks_analysis/ matching
    the same feature/hidden/batch signature as ANAME (any seed)."""
    import re as _re
    # ANAME = everything_seed{seed}_{feature}+hidden{h}+batch{b}+angle
    m = _re.match(r"everything_seed\d+_(.+)$", ANAME)
    suffix = m.group(1) if m else ""
    base = Path("multiple_tasks_analysis")
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
    out_path = _multitask_out(out_filename)
    _save_fig(fig, out_path, extra=f"  (n={n_experiments} experiments)")


def plot_overmembership_norm():
    """Figure: Over-membership for normalized modulation (G=100), aggregated across seeds."""
    _plot_overmembership_single(
        "modulation_all_prepost_belonging_{aname}_normalized.pkl",
        "overmembership_normalized.png",
    )


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
        Figure: Normalized lesion effect heatmaps for unnormalized clusterings.

    Two panels stacked vertically:
            Top — hidden (post) cluster lesion effect from `leison_unnorm`
                        (tasks × hidden clusters)
            Bottom — modulation cluster lesion effect from
                             `modulation_all_var_weighted_unnormalized__freeze_M`
                             (tasks × modulation clusters)

        Normalized effect = random_acc - cluster_acc (positive = cluster matters).
        Reads the effect matrices exported by leison_plot.py, without deriving
        effects again from raw accuracy. Task and cluster identities are retained.
        The colorbar is saved as its own figure so the panel layout stays compact.
    """
    _ensure_out_dir()
    path = LESION_NORM_DIR / f"normalized_lesion_effects_{ANAME}.pkl"
    data = _load_pkl_or_skip(path, "Run multiple_task/leison_plot.py first.")
    if data is None:
        return

    try:
        if data["schema_version"] != 1 or data["aname"] != ANAME:
            raise ValueError("normalized-effect cache version/run does not match")
        hidden = data["entries"]["leison_unnorm"]
        modulation = data["entries"]["modulation_all_var_weighted_unnormalized__freeze_M"]
        for entry in (hidden, modulation):
            if entry["definition"] != "random_minus_lesion" or entry["units"] != "fraction":
                raise ValueError("unexpected effect definition or units")
            if np.shape(entry["effect"]) != (len(entry["tasks"]), len(entry["conditions"])):
                raise ValueError("effect axes do not match saved labels")
            if (len(set(entry["tasks"])) != len(entry["tasks"])
                    or len(set(entry["conditions"])) != len(entry["conditions"])):
                raise ValueError("duplicate task or cluster labels")
        all_tasks = hidden["tasks"]
        if set(all_tasks) != set(modulation["tasks"]):
            raise ValueError("hidden and modulation task sets differ")
        post_idx = [index for index, name in enumerate(hidden["conditions"]) if name.startswith("post_c")]
        mod_idx = [index for index, name in enumerate(modulation["conditions"]) if name.startswith("mod_c")]
        if not post_idx or not mod_idx:
            raise ValueError("no saved hidden or modulation cluster effects")
        mod_rows = [modulation["tasks"].index(task) for task in all_tasks]
        effect_post = np.asarray(hidden["effect"], dtype=float)[:, post_idx] * 100
        effect_mod = np.asarray(modulation["effect"], dtype=float)[np.ix_(mod_rows, mod_idx)] * 100
        if not np.isfinite(effect_post).all() or not np.isfinite(effect_mod).all():
            raise ValueError("non-finite saved effect values")
        post_labels = ["C" + hidden["conditions"][index].removeprefix("post_c") for index in post_idx]
        mod_labels = ["C" + modulation["conditions"][index].removeprefix("mod_c") for index in mod_idx]
    except (KeyError, TypeError, ValueError) as error:
        print(f"  Skipped: incompatible {path.name} ({error}). Run multiple_task/leison_plot.py again.")
        return

    all_tasks_display = [_TASK_DISPLAY.get(task, task) for task in all_tasks]
    vmax = max(np.abs(effect_post).max(), np.abs(effect_mod).max(), 1e-6)

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

    # Ticks every 30, symmetric around 0, within [-vmax, vmax]
    _tick_max = int(np.floor(vmax / 30.0)) * 30
    _ticks = np.arange(-_tick_max, _tick_max + 1, 30)
    out_path = _multitask_out("lesion_heatmap_unnorm.png")
    _save_fig(fig, out_path)
    _save_standalone_colorbar(
        _multitask_out("lesion_heatmap_unnorm_colorbar.png"),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        ticks=_ticks,
        ticklabels=[f"{t:.0f}" for t in _ticks],
        label="Normalized effect (%)",
        orientation="vertical",
        figsize=(0.7, 2.4),
        rect=(0.35, 0.08, 0.3, 0.84),
        labelsize=10,
        label_fontsize=8,
    )


def plot_lesion_cluster_sizes():
    """
    Figure: relative size of every cluster shown in the lesion heatmap
    (plot_lesion_heatmap), as two stacked bar panels sharing its column order:

        Top    — hidden (post) neuron clusters from the unnormalized
                 clustering: % of all hidden neurons per cluster
        Bottom — var-weighted-unnormalized modulation (synapse) clusters:
                 % of all clustered synapses per cluster

    Companion to the heatmap: it says how much substrate each column's lesion
    removes, so a big effect from a small cluster reads as selectivity rather
    than mass. Cluster indices match the heatmap's C1..Cn labels (the two
    panels' numberings are independent of each other, as there).
    """
    _ensure_out_dir()
    data = _load_lesion_results()
    if data is None:
        print("  Skipped: lesion results not found. Run leison.py first.")
        return

    # Top: hidden (post) cluster sizes, in the heatmap's column order
    lu = data["leison_unnorm"].get("lesion_units", {})
    comb_names = data["leison_unnorm"]["all_comb_names_leison"]
    post_names = [n for n in comb_names if n.startswith("post_c")]
    if not post_names or any(n not in lu for n in post_names):
        print("  Skipped: lesion_units missing for hidden clusters.")
        return
    hid_sizes = np.array([lu[n] for n in post_names], float)

    # Bottom: var-weighted modulation cluster sizes (sorted ids = heatmap order)
    mod_entry = data["mod_leison"].get(
        "modulation_all_var_weighted_unnormalized__freeze_M")
    if mod_entry is None:
        print("  Skipped: var-weighted freeze_M lesion entry not found.")
        return
    col_clusters = mod_entry["mod_col_clusters"]
    mod_sizes = np.array([len(col_clusters[c]) for c in sorted(col_clusters)],
                         float)

    panels = [
        (hid_sizes, f"Hidden neuron clusters (n = {int(hid_sizes.sum())} neurons)"),
        (mod_sizes, f"Modulation clusters (n = {int(mod_sizes.sum())} synapses)"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(6, 3.6), sharex=False)
    for ax, (sizes, title) in zip(axes, panels):
        pct = sizes / sizes.sum() * 100
        xs = np.arange(len(pct))
        ax.bar(xs, pct, color="#4682b4", edgecolor="k", linewidth=0.4, width=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"C{i + 1}" for i in xs], fontsize=6)
        ax.set_ylabel("Cluster size (%)", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[1].set_xlabel("Cluster", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, _multitask_out("lesion_cluster_sizes.png"))


# ─── Figure: OM vs lesion ────────────────────────────────────────────────────

def _load_om_lesion_scatter():
    """Load the exact upstream freeze-M scatter and statistics, without refiltering."""
    tag = "var-weighted-unnormalized"
    candidates = ((f"om_vs_lesion_diff_{tag}_combined_unnorm_{ANAME}.pkl", True),
                  (f"om_vs_lesion_diff_{tag}_freeze-M_unnorm_{ANAME}.pkl", False))
    for filename, combined in candidates:
        path = LESION_NORM_DIR / filename
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                saved = pickle.load(handle)
            base_key = saved["base_key"] if combined else saved["mod_type_key"]
            if (base_key != "modulation_all_var_weighted_unnormalized"
                    or saved["variant"] != "unnorm" or saved.get("aname", ANAME) != ANAME):
                raise ValueError("OM cache belongs to a different run or variant")
            if saved.get("y_definition") != "task-profile L1/T: mean_t |mod_effect(t) - combined_effect(t)|":
                raise ValueError("OM cache does not describe task-profile L1/T distance")
            if combined:
                entry = saved["mode_data"]["freeze_M"]
                p_perm, n_clusters, n_perm = entry["p_perm"], entry["n_clusters"], saved["n_perm"]
            else:
                if saved["mod_lesion_mode"] != "freeze_M":
                    raise ValueError("OM cache uses a different lesion mode")
                entry = saved
                permutation = saved["permutation"]
                p_perm, n_clusters, n_perm = permutation["p_perm"], permutation["n_clusters"], permutation["n_perm"]
            om_vals = np.asarray(entry["om_vals"], dtype=float)
            lesion_diffs = np.asarray(entry["lesion_diffs"], dtype=float)
            if (om_vals.ndim != 1 or om_vals.shape != lesion_diffs.shape or om_vals.size < 2
                    or not np.isfinite(om_vals).all() or not np.isfinite(lesion_diffs).all()):
                raise ValueError("invalid saved OM scatter coordinates")
            regression = {key: float(entry["regression"][key]) for key in ("slope", "intercept", "r", "p")}
            min_expected = float(saved["min_expected"])
            if not np.isfinite(min_expected) or min_expected < 0:
                raise ValueError("invalid saved OM threshold")
            return {"om_vals": om_vals, "lesion_diffs": lesion_diffs, "regression": regression,
                    "p_perm": float(p_perm), "n_clusters": int(n_clusters), "n_perm": int(n_perm),
                    "min_expected": min_expected, "path": path}
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"  Note: incompatible {path.name}: {error}.")
    print("  Skipped: no complete saved OM scatter/statistics. Run multiple_task/leison_plot.py again.")
    return None


def plot_om_vs_lesion():
    """Replot the saved var-weighted freeze-M OM/profile-L1 scatter and permutation p.

    The sample set and min_expected threshold come from leison_plot's cache.
    No raw lesion/cluster data, profile matching, regression fitting, or new
    permutation test is used here.
    """
    _ensure_out_dir()
    saved = _load_om_lesion_scatter()
    if saved is None:
        return
    om_vals, lesion_diffs = saved["om_vals"], saved["lesion_diffs"]
    regression = saved["regression"]
    slope, intercept, r, p = (regression[key] for key in ("slope", "intercept", "r", "p"))
    p_perm = saved["p_perm"]
    fig1, ax1 = plt.subplots(1, 1, figsize=(3, 2.8))
    ax1.scatter(om_vals, lesion_diffs, color="#3182ce", edgecolors="k",
                linewidths=0.5, s=40, alpha=0.8, zorder=3)
    x_line = np.linspace(om_vals.min(), om_vals.max(), 100)
    if np.isfinite(slope) and np.isfinite(intercept):
        ax1.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.2, zorder=4)
    _pp_str = (f"p = {p_perm:.3f}" if np.isfinite(p_perm) else "p = n/a")
    _legend(ax1, [f"r = {r:.2f}, {_pp_str}"], loc="upper right", fontsize=7, frameon=True)
    ax1.set_xlabel("Over-membership", fontsize=8)
    ax1.set_ylabel("Lesion profile L1 distance", fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()
    out_path1 = _multitask_out("om_vs_lesion_scatter.png")
    _save_fig(fig1, out_path1)
    print(f"  om_vs_lesion_scatter: r={r:.2f}, p_perm={p_perm:.3f} "
            f"({saved['n_perm']} saved permutations, {saved['n_clusters']} clusters, "
            f"min_expected={saved['min_expected']:g}, naive p={p:.1e}; {saved['path'].name})")


# ─── Figure: Fixed-point PCA trajectories ────────────────────────────────────

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
    out_path = _multitask_out("input_weight_correlation.png")
    _save_fig(fig, out_path)


# ─── Figure: Cluster tuning vs lesion effect ─────────────────────────────────

LESION_NORM_DIR = Path("multiple_tasks_norm") / ANAME


def _draw_saved_lesion_regression(axis, values, regression):
    """Draw a cached regression line and its original statistics without fitting."""
    if regression is None:
        return
    positions = np.linspace(values.min(), values.max(), 100)
    axis.plot(positions, regression["slope"] * positions + regression["intercept"],
              color="tomato", linewidth=1.2, zorder=4)
    label = f"r = {regression['r']:.2f}, p = {regression['p']:.3f}"
    _legend(axis, [label], loc="upper left", fontsize=7, frameon=True)


def plot_cluster_corr_vs_lesion():
    """
    Figure: saved cluster tuning cosine similarity vs lesion-effect L1 distance.
    Use the exact coordinates and regression computed by leison_plot.py. This is
    sum-over-tasks L1, not profile Pearson correlation or a Mantel test.

    Produces one scatter figure per variant (normalized, unnormalized) and
    cluster type (input, hidden). Incomplete caches are skipped, never refitted.
    """
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
            try:
                if (data.get("y_definition") != "sum_over_tasks_abs_effect_difference"
                        or data.get("aname", ANAME) != ANAME):
                    raise ValueError("saved metric or run does not match")
                x = np.asarray(data["tuning_cos_sim"], dtype=float)
                y = np.asarray(data["lesion_l1_dist"], dtype=float)
                if (x.ndim != 1 or x.shape != y.shape or x.size == 0
                        or not np.isfinite(x).all() or not np.isfinite(y).all()):
                    raise ValueError("invalid saved scatter coordinates")
                regression = data["regression"]
                if regression is not None:
                    regression = {key: float(regression[key])
                                  for key in ("slope", "intercept", "r", "p")}
                    if not all(np.isfinite(value) for value in regression.values()):
                        raise ValueError("non-finite saved regression statistics")
            except (KeyError, TypeError, ValueError) as error:
                print(f"  Skipped {name}: incomplete or incompatible L1 cache ({error}); "
                      "run multiple_task/leison_plot.py again.")
                continue
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.8))

            ax.scatter(x, y, color="#3182ce", edgecolors="k",
                       linewidths=0.5, s=40, alpha=0.8, zorder=3)

            ax.set_xlabel("Tuning cosine similarity", fontsize=8)
            ax.set_ylabel("Lesion effect L1 distance", fontsize=8)
            _draw_saved_lesion_regression(ax, x, regression)
            ax.spines[["top", "right"]].set_visible(False)

            fig.tight_layout()
            # e.g. "input_normalized_k20" -> "input_norm"
            clean_name = name.replace("_normalized", "_norm").replace("_unnormalized", "_unnorm").replace("_k20", "")
            out_path = _multitask_out(f"cluster_corr_vs_lesion_{clean_name}.png")
            _save_fig(fig, out_path)


# ─── Figure: Cross-seed summary of the var-weighted lesion results ──────────

def plot_cross_seed_summary():
    """
    Figure: cross-seed consistency of the var-weighted-unnormalized lesion
    results — one point per seed, five panels:

      P1  OM vs lesion-profile-L1 pooled r (zero_W & freeze_M);
          filled = cluster-permutation p < 0.05
      P2  plasticity share median (freeze_M effect / zero_W effect on
          significant cells)
      P3  zero_W vs freeze_M effect-map pattern correlation
      P4  tuning-similarity vs lesion-profile-correlation Mantel r (zero_W);
          filled = Mantel p < 0.05
      P5  per-cluster Spearman(memory-family bias, plasticity share) —
          are the memory-serving clusters the plasticity-dependent ones?

    Every number is read from the per-seed pickles leison_plot.py saves in
    multiple_tasks_norm/ (no model forwards, no cluster_info_mod). Each
    panel is annotated with the Fisher-z mean r (or plain mean) and a
    one-sided sign test across seeds. The raw per-seed numbers are also
    written to {MULTITASK_PREFIX}_cross_seed_summary.csv.
    """
    import re as _re
    from scipy.stats import spearmanr as _spearmanr, binomtest as _binomtest

    _ensure_out_dir()
    tag = "var-weighted-unnormalized"
    run_pattern = _re.sub(r"seed\d+", "seed*", ANAME)
    run_dirs = sorted(Path("multiple_tasks_norm").glob(run_pattern))
    if not run_dirs:
        print(f"  Skipped: no runs match multiple_tasks_norm/{run_pattern}.")
        return

    def _task_family(t):
        return ("memory"
                if ("delay" in t or t.startswith("dms") or t.startswith("dmc"))
                else "reaction")

    rows = []
    for run in run_dirs:
        aname = run.name
        seed = _re.search(r"seed(\d+)", aname).group(1)
        row = {"seed": seed}

        # P1: OM vs lesion-profile-L1 scatter, both modes
        p = run / f"om_vs_lesion_diff_{tag}_combined_unnorm_{aname}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            for mode in ["zero_W", "freeze_M"]:
                md = d["mode_data"].get(mode)
                if md is None or "p_perm" not in md:
                    continue
                row[f"om_r_{mode}"] = float(
                    np.corrcoef(md["om_vals"], md["lesion_diffs"])[0, 1])
                row[f"om_p_perm_{mode}"] = float(md["p_perm"])

        # P2/P3/P5: plasticity-share pickle
        p = run / f"plasticity_share_{tag}_{aname}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            share = np.asarray(d["share"], float)            # (T, C), NaN = n.s.
            E_zw = np.asarray(d["effect_zero_w"], float)     # (T, C)
            E_fm = np.asarray(d["effect_freeze_m"], float)
            row["share_median"] = float(np.nanmedian(share))
            row["pattern_r"] = float(
                np.corrcoef(E_zw.ravel(), E_fm.ravel())[0, 1])

            is_mem = np.array([_task_family(t) == "memory" for t in d["tasks"]])
            bias = E_zw[is_mem].mean(axis=0) - E_zw[~is_mem].mean(axis=0)
            # Per-cluster median share; clusters with no significant cell are
            # all-NaN columns — give them NaN without numpy's warning.
            _any = np.isfinite(share).any(axis=0)
            share_c = np.full(share.shape[1], np.nan)
            share_c[_any] = np.nanmedian(share[:, _any], axis=0)
            ok = np.isfinite(share_c)
            if ok.sum() >= 5:
                row["bias_share_rho"] = float(
                    _spearmanr(bias[ok], share_c[ok]).statistic)

        # P4: tuning similarity vs lesion profile correlation (Mantel)
        p = run / f"cluster_corr_vs_mod_leison_effect_{tag}_zero-W_{aname}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            entry = next(iter(d.values()))
            mantel = entry.get("mantel")
            if mantel is not None and np.isfinite(mantel.get("r", np.nan)):
                row["mantel_r"] = float(mantel["r"])
                row["mantel_p"] = float(mantel["p"])
        rows.append(row)

    def _col(key):
        return np.array([r.get(key, np.nan) for r in rows], float)

    def _fisher_mean(r):
        r = r[np.isfinite(r)]
        return float(np.tanh(np.arctanh(np.clip(r, -0.999, 0.999)).mean())) \
            if r.size else np.nan

    def _sign_p(vals, positive):
        """One-sided sign test that the seeds agree with the expected sign."""
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return np.nan
        k = int((v > 0).sum() if positive else (v < 0).sum())
        return float(_binomtest(k, v.size, 0.5, alternative="greater").pvalue)

    seeds = [r["seed"] for r in rows]
    xs = np.arange(len(rows))
    fig, axs = plt.subplots(1, 5, figsize=(14.5, 2.9))

    def _seed_axis(ax):
        ax.set_xticks(xs)
        ax.set_xticklabels(seeds, rotation=60, fontsize=6)
        ax.set_xlabel("Seed", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    # P1: OM scatter r, both modes, filled = p_perm < 0.05
    ax = axs[0]
    for off, mode, color in [(-0.15, "zero_W", "#3182ce"),
                             (0.15, "freeze_M", "#9ecae1")]:
        r = _col(f"om_r_{mode}")
        sig = _col(f"om_p_perm_{mode}") < 0.05
        fin = np.isfinite(r)
        ax.scatter(xs[fin & sig] + off, r[fin & sig], s=26,
                   facecolors=color, edgecolors=color, linewidths=0.8,
                   zorder=3, label=mode.replace("_", "-"))
        ax.scatter(xs[fin & ~sig] + off, r[fin & ~sig], s=26,
                   facecolors="white", edgecolors=color, linewidths=0.8, zorder=3)
        ax.axhline(_fisher_mean(r), color=color, linewidth=0.7,
                   linestyle="--", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.5)
    _z = _col("om_r_zero_W")
    ax.set_title(f"OM vs profile-L1 r\nsign p={_sign_p(_z, positive=False):.3f}",
                 fontsize=8)
    ax.set_ylabel("Pooled r", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    _seed_axis(ax)

    # P2: plasticity share median
    ax = axs[1]
    v = _col("share_median")
    ax.scatter(xs, v, s=26, color="#1b9e77", edgecolors="k", linewidths=0.4, zorder=3)
    ax.axhline(np.nanmean(v), color="#1b9e77", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":")
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    _k = int(np.nansum(v > 0.5))
    ax.set_title(f"Plasticity share (median)\nmean={np.nanmean(v):.2f}, "
                 f"{_k}/{int(np.isfinite(v).sum())} > 0.5", fontsize=8)
    ax.set_ylabel("freeze-M / zero-W", fontsize=8)
    ax.set_ylim(0, 1.1)
    _seed_axis(ax)

    # P3: zero_W vs freeze_M pattern correlation
    ax = axs[2]
    v = _col("pattern_r")
    ax.scatter(xs, v, s=26, color="#3182ce", edgecolors="k", linewidths=0.4, zorder=3)
    ax.axhline(_fisher_mean(v), color="#3182ce", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title(f"zero-W vs freeze-M map r\nFisher mean={_fisher_mean(v):.2f}",
                 fontsize=8)
    ax.set_ylabel("Pattern r", fontsize=8)
    ax.set_ylim(0, 1)
    _seed_axis(ax)

    # P4: Mantel r (tuning sim vs lesion profile corr)
    ax = axs[3]
    v = _col("mantel_r")
    pp = _col("mantel_p")
    sig = pp < 0.05
    fin = np.isfinite(v)
    ax.scatter(xs[fin & sig], v[fin & sig], s=26, color="#7e3ff2",
               edgecolors="#7e3ff2", linewidths=0.8, zorder=3)
    ax.scatter(xs[fin & ~sig], v[fin & ~sig], s=26, facecolors="white",
               edgecolors="#7e3ff2", linewidths=0.8, zorder=3)
    ax.axhline(0, color="grey", linewidth=0.5)
    if fin.any():
        ax.axhline(_fisher_mean(v), color="#7e3ff2", linewidth=0.7,
                   linestyle="--", alpha=0.7)
    ax.set_title(f"Tuning-sim vs lesion-corr Mantel r\n"
                 f"sign p={_sign_p(v, positive=True):.3f}", fontsize=8)
    ax.set_ylabel("Mantel r", fontsize=8)
    _seed_axis(ax)

    # P5: memory-bias x share Spearman
    ax = axs[4]
    v = _col("bias_share_rho")
    ax.scatter(xs, v, s=26, color="#d95f02", edgecolors="k", linewidths=0.4, zorder=3)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axhline(np.nanmean(v), color="#d95f02", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_title(f"Spearman(memory bias, share)\nsign p={_sign_p(v, positive=True):.3f}",
                 fontsize=8)
    ax.set_ylabel("rho", fontsize=8)
    ax.set_ylim(-1, 1)
    _seed_axis(ax)

    fig.suptitle(f"Cross-seed summary — {tag} ({len(rows)} seeds)", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, _multitask_out("cross_seed_summary.png"))

    # CSV with the raw per-seed numbers
    cols = ["seed", "om_r_zero_W", "om_p_perm_zero_W", "om_r_freeze_M",
            "om_p_perm_freeze_M", "share_median", "pattern_r",
            "mantel_r", "mantel_p", "bias_share_rho"]
    csv_path = OUT_DIR / f"{MULTITASK_PREFIX}_cross_seed_summary.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                (f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
                for c in cols) + "\n")
    print(f"Saved: {csv_path}")


# ─── Figure: Transfer speed ──────────────────────────────────────────────────

PRETRAINING_ANALYSIS_DIR = Path("pretraining_analysis")
PRETRAINING_ADDON_NAME = "+hidden200+L21e3+batch128+angle"


def _pretraining_result_pkls():
    """Return the configured per-seed pretraining result pickles."""
    pattern = f"*_dmpn_seed*_{PRETRAINING_ADDON_NAME}_result.pkl"
    return sorted(PRETRAINING_ANALYSIS_DIR.glob(pattern))


def _pretraining_ruleset_from_result_name(filename):
    """Extract the stage-1 ruleset from a configured pretraining result filename."""
    import re as _re

    pattern = rf"(.+)_dmpn_seed\d+_{_re.escape(PRETRAINING_ADDON_NAME)}_result\.pkl"
    match = _re.match(pattern, filename)
    return match.group(1) if match else None


def _transfer_speed_summary(per_seed_iters):
    """Summarize positive first-hit times among reaching seeds; NaN means not reached."""
    values = np.asarray(per_seed_iters, dtype=float)
    if values.ndim != 2 or np.isinf(values).any() or np.any(values <= 0):
        raise ValueError("Expected a seed-by-threshold matrix of positive times or NaN")
    reached = np.sum(np.isfinite(values), axis=0)
    quartiles = np.full((3, values.shape[1]), np.nan)
    for column, count in enumerate(reached):
        if count:
            quartiles[:, column] = np.percentile(values[np.isfinite(values[:, column]), column],
                                                  [25, 50, 75])
    return quartiles[1], quartiles[0], quartiles[2], reached, values.shape[0]


def plot_transfer_speed():
    """
    Figure: Transfer speed — iterations to reach accuracy thresholds during
    post-training, comparing fdgo_delaygo vs fdanti_delaygo rulesets.

    Loads from the combined transfer_speed.pkl if available; otherwise falls
    back to loading individual per-seed result pickles.
    Points show individual reaching seeds, lines their median, and bands their
    25th-75th percentiles. Non-reaching seeds are excluded from these summaries;
    this is conditional on reaching, not a survival estimate.
    """
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
        pkls = _pretraining_result_pkls()
        if not pkls:
            print("  Skipped: no pretraining result pickles found.")
            return

        by_ruleset_raw = {}
        for p in pkls:
            ruleset = _pretraining_ruleset_from_result_name(p.name)
            if ruleset is not None:
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

    fig, ax = plt.subplots(figsize=(3, 2.2 * 2 / 3))

    for rs, rs_data in by_ruleset_mats.items():
        color = ruleset_colors.get(rs, "#718096")
        label = ruleset_labels.get(rs, rs)
        per_seed_mat = np.asarray(rs_data["per_seed_iters"], dtype=float)
        medians, lower, upper, _, n_seeds = _transfer_speed_summary(per_seed_mat)
        if per_seed_mat.shape[1] != len(ys) or rs_data["n_seeds"] != n_seeds:
            raise ValueError(f"{rs}: transfer-speed shape or seed count mismatch")
        for column, threshold in enumerate(ys):
            times = per_seed_mat[:, column]
            times = times[np.isfinite(times)]
            ax.scatter(times, np.full(len(times), threshold), color=color,
                       s=12, alpha=0.55, linewidths=0, zorder=3)
        ax.plot(medians, ys, "s-", color=color, linewidth=1.5,
                markersize=5, label=label)
        ax.fill_betweenx(ys, lower, upper,
                         color=color, alpha=0.15)

    ax.set_xlabel("Iterations to reach threshold")
    ax.set_ylabel("Accuracy\nthreshold (%)", ha="center")
    ax.set_xscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.set_ylim(float(np.min(ys)) - 3, float(np.max(ys)) + 3)
    ax.set_title("Reaching seeds: median and IQR", fontsize=8)
    _legend(ax, fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "transfer_speed.png"
    _save_fig(fig, out_path)


def plot_backbone_probe():
    """Random-rule backbone probe: seed-mean accuracy with adaptive y limits.

    Reads the per-checkpoint JSONs in pretraining_analysis/ written by pretraining_post.py's
    --backbone-probe experiment (accuracy_pct is already in percent).
    Points average random rule initializations within each seed; diamonds show
    the across-seed mean with population SD. This is not the exact stage-2 init.
    A two-sided independent-seed permutation test compares group means, assuming
    exchangeability under the null; random initializations are not replicates.
    """
    import json
    import re as _re
    from scipy.stats import permutation_test

    groups = {
        "fdgo_delaygo": ("Irrelevant\nmotif", "#3182ce"),
        "fdanti_delaygo": ("Relevant\nmotif", "#e53e3e"),
    }
    values = {ruleset: [] for ruleset in groups}
    pattern = _re.compile(
        rf"backbone_probe_({'|'.join(map(_re.escape, groups))})_dmpn_seed\d+_"
        rf"{_re.escape(PRETRAINING_ADDON_NAME)}\.json")
    paths = sorted(PRETRAINING_ANALYSIS_DIR.glob(
        f"backbone_probe_*_dmpn_seed*_{PRETRAINING_ADDON_NAME}.json"))
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        ruleset = match.group(1)
        with path.open() as handle:
            run = json.load(handle)
        if run.get("ruleset") != ruleset:
            raise ValueError(f"{path.name}: ruleset metadata does not match filename")
        samples = np.asarray(run.get("random_probe", {}).get("accuracy_pct", []),
                             dtype=float)
        if samples.size == 0 or not np.isfinite(samples).all():
            print(f"  Note: {path.name}: missing or non-finite backbone probe acc; omitted.")
            continue
        values[ruleset].append(float(samples.mean()))

    if not any(values.values()):
        print("  Skipped: no usable backbone probe results; run "
              "pretrain/pretraining_post.py --backbone-probe first.")
        return

    _ensure_out_dir()
    fig, axis = plt.subplots(figsize=(2.6, 2.4))
    counts = []
    bounds = []
    for position, (ruleset, (_, color)) in enumerate(groups.items()):
        samples = np.asarray(values[ruleset])
        counts.append(f"{ruleset} acc n={samples.size}")
        if samples.size == 0:
            continue
        mean, std = samples.mean(), samples.std()
        bounds.extend([samples.min(), samples.max(), mean - std, mean + std])
        jitter = np.linspace(-0.10, 0.10, samples.size) if samples.size > 1 else np.zeros(1)
        axis.scatter(position + jitter, samples, color=color, s=24,
                     alpha=0.7, edgecolors="k", linewidths=0.4, zorder=3)
        axis.errorbar(position, mean, yerr=std,
                      fmt="D", color="k", markerfacecolor="white",
                      markersize=4, capsize=3, linewidth=1.0, zorder=4)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([label for label, _ in groups.values()])
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylabel("Accuracy (%)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    lower, upper = min(bounds), max(bounds)
    padding = max((upper - lower) * 0.1, 0.5)
    axis.set_ylim(lower - padding, upper + padding)
    axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    irrelevant = np.asarray(values["fdgo_delaygo"], dtype=float)
    relevant = np.asarray(values["fdanti_delaygo"], dtype=float)
    if min(irrelevant.size, relevant.size) >= 2:
        test = permutation_test(
            (relevant, irrelevant),
            lambda first, second: np.mean(first) - np.mean(second),
            permutation_type="independent", alternative="two-sided",
            n_resamples=9999, random_state=0,
        )
        stats_label = (
            f"Two-sided permutation p={test.pvalue:.3g}\n"
            f"Relevant - Irrelevant = {test.statistic:+.2f} pp\n"
            f"Seeds: relevant={relevant.size}, irrelevant={irrelevant.size}"
        )
    else:
        stats_label = "Permutation test unavailable: need >=2 seeds/group"
    print(f"  Backbone probe statistics: {stats_label.replace(chr(10), '; ')}")
    from datetime import datetime
    log_path = Path("log") / "backbone_probe.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"{PRETRAINING_ADDON_NAME}: {stats_label.replace(chr(10), '; ')}\n"
        )
    fig.suptitle("Random-rule backbone probe", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, OUT_DIR / "backbone_probe.png", extra=f" ({'; '.join(counts)})")


def plot_learning_trajectory():
    """
    Figure: post-training learning trajectory — accuracy vs training iteration,
    comparing fdgo_delaygo vs fdanti_delaygo rulesets. Same rulesets /
    colors as plot_transfer_speed, but plotting the full accuracy curve with
    transparent per-seed trajectories plus the mean rather than
    iterations-to-threshold.

    Reads per-seed result pickles (learning.acc_iter_post / learning.acc_post).
    Seeds are resampled onto a shared iteration grid before averaging, so it is
    robust to slightly different logging cadences across seeds.
    """
    _ensure_out_dir()
    if not PRETRAINING_ANALYSIS_DIR.exists():
        print("  Skipped: pretraining_analysis/ not found.")
        return

    pkls = _pretraining_result_pkls()
    if not pkls:
        print("  Skipped: no pretraining result pickles found.")
        return

    # Collect each ruleset's per-seed (iterations, accuracy) trajectories.
    by_ruleset_traj = {}  # rs -> list of (iters, acc)
    for p in pkls:
        rs = _pretraining_ruleset_from_result_name(p.name)
        if rs is None:
            continue
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

        for seed_curve in resampled:
            ax.plot(grid, seed_curve * 100, "-", color=color,
                linewidth=0.9, alpha=0.18)

        mean_vals = resampled.mean(axis=0) * 100
        ax.plot(grid, mean_vals, "-", color=color, linewidth=2.2, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    # ylim tops at 105 for headroom; explicit ticks stop at 100 so no >100% tick.
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim([0, 105])
    ax.tick_params(axis="both", labelsize=7)
    _legend(ax, fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "learning_trajectory.png"
    _save_fig(fig, out_path)


# ─── Figure: Rule vectors ────────────────────────────────────────────────────

def plot_rule_vectors():
    """
    Figure: Pairwise cosine similarity between rule-input vectors.

    Shows how the novel task's learned rule vector relates to every available
    pretrained rule vector, including the one-parent DelayAnti condition.
    """
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
        pkls = _pretraining_result_pkls()
        if not pkls:
            print("  Skipped: no pretraining result pickles found.")
            return

        stage1_tasks_map = {
            "fdgo_delaygo": ["fdgo", "delaygo"],
            "fdanti_delaygo": ["fdanti", "delaygo"],
            "fdanti": ["fdanti"],
        }
        final_task = "delayanti"

        by_ruleset = {}
        for p in pkls:
            rs = _pretraining_ruleset_from_result_name(p.name)
            if rs is not None:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                if "rule_vectors" in data:
                    rv = data["rule_vectors"]
                    stage1_tasks = list(rv.get(
                        "pretrained_tasks", stage1_tasks_map.get(rs, [rs])))
                    entry = by_ruleset.setdefault(rs, {
                        "cos_novel_by_task": {task: [] for task in stage1_tasks},
                        "cos_pretrained_pairs": {},
                        "in_span_fraction": [],
                        "stage1_tasks": stage1_tasks,
                        "final_task": rv.get("novel_task", final_task),
                    })
                    if entry["stage1_tasks"] != stage1_tasks:
                        raise ValueError(
                            f"{p}: inconsistent Stage-1 task list for {rs}")
                    named_novel = rv.get("cos_novel_by_task", {})
                    for task_index, task in enumerate(stage1_tasks):
                        value = named_novel.get(task, rv.get(f"cos_novel_pre{task_index}"))
                        if value is None:
                            raise KeyError(f"{p}: missing novel cosine for {task}")
                        entry["cos_novel_by_task"][task].append(value)
                    named_pairs = rv.get("cos_pretrained_pairs", {})
                    for pair, value in named_pairs.items():
                        entry["cos_pretrained_pairs"].setdefault(pair, []).append(value)
                    if (not named_pairs and len(stage1_tasks) == 2
                            and "cos_pre0_pre1" in rv):
                        pair = f"{stage1_tasks[0]}__{stage1_tasks[1]}"
                        entry["cos_pretrained_pairs"].setdefault(pair, []).append(
                            rv["cos_pre0_pre1"])
                    entry["in_span_fraction"].append(rv["in_span_fraction"])

    if not by_ruleset:
        print("  Skipped: no rule vector data found.")
        return

    ruleset_colors = {
        "fdgo_delaygo": "#3182ce",
        "fdanti_delaygo": "#e53e3e",
        "fdanti": "#dd6b20",
    }
    ruleset_labels = {
        "fdgo_delaygo": "Irrelevant motif",
        "fdanti_delaygo": "Relevant motif",
        "fdanti": "DelayAnti",
    }

    task_display_names = {
        "delayanti": "MemoryAnti",
        "fdanti": "DelayAnti",
        "fdgo": "DelayPro",
        "delaygo": "MemoryPro",
    }

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
    # rulesets instead of grouping all of one ruleset's bars together.
    per_rs_bars = {}  # rs -> list of (label, mean, std, vals)
    for rs in rs_list:
        s1_tasks = by_ruleset[rs].get("stage1_tasks", [rs])
        final_task = by_ruleset[rs].get("final_task", "novel")

        ft = task_display_names.get(final_task, final_task)
        if not s1_tasks:
            raise ValueError(f"{rs}: rule-vector data has no Stage-1 tasks")

        # (values, label, underlying raw task pair) for every comparison that
        # exists. The named schema supports the one-parent DelayAnti condition;
        # legacy flat keys remain a fallback for older two-parent pickles.
        bar_specs = []
        named_novel = by_ruleset[rs].get("cos_novel_by_task", {})
        for task_index, task in enumerate(s1_tasks):
            values = named_novel.get(
                task, by_ruleset[rs].get(f"cos_novel_pre{task_index}"))
            if values is None:
                raise KeyError(f"{rs}: missing novel cosine for {task}")
            task_label = task_display_names.get(task, task)
            bar_specs.append((values, f"{ft}\n↔ {task_label}",
                              (final_task, task)))

        named_pairs = by_ruleset[rs].get("cos_pretrained_pairs", {})
        for left_index, left_task in enumerate(s1_tasks):
            for right_index in range(left_index + 1, len(s1_tasks)):
                right_task = s1_tasks[right_index]
                pair_key = f"{left_task}__{right_task}"
                values = named_pairs.get(pair_key)
                if values is None and left_index == 0 and right_index == 1:
                    values = by_ruleset[rs].get("cos_pre0_pre1")
                if values is None:
                    raise KeyError(f"{rs}: missing pretrained cosine for {pair_key}")
                left_label = task_display_names.get(left_task, left_task)
                right_label = task_display_names.get(right_task, right_task)
                bar_specs.append((values, f"{left_label}\n↔ {right_label}",
                                  (left_task, right_task)))
        # Drop the excluded task pairs; keep remaining bars packed (no gaps).
        bar_specs = [
            (values, label, pair) for values, label, pair in bar_specs
            if frozenset(pair) not in excluded_pairs
        ]
        per_rs_bars[rs] = [
            (label, float(np.mean(values)), float(np.std(values)), np.asarray(values))
            for values, label, _ in bar_specs
        ]

    # The original two-ruleset figure had four bars. DelayAnti adds a
    # fifth, so scale width with the actual count to keep multiline labels apart.
    n_bars_total = sum(len(bars) for bars in per_rs_bars.values())
    fig.set_size_inches(max(3.6, 1.05 * n_bars_total), 2.4 * 2 / 3)

    # Interleave: for each column index, emit one bar per ruleset; a group_gap
    # separates successive columns.
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
    ax.set_xticklabels(all_labels, rotation=0, ha="center", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
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
    pkls = _pretraining_result_pkls()
    if not pkls:
        return {}

    raw_by_rs = {}
    for p in pkls:
        rs = _pretraining_ruleset_from_result_name(p.name)
        if rs is not None:
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
    and seed-mean thick lines. Shared by the stimulus and response figures.
    """
    key_self = f"{dtype}_{period}_self"
    key_cross = f"{dtype}_{period}_cross"
    seed_width = 0.9
    mean_width = 2.2
    self_alpha = 0.18
    cross_alpha = 0.18
    cross_style = "-"

    ruleset_order = ("fdanti", "fdanti_delaygo", "fdgo_delaygo")
    self_ruleset_order = ("fdanti_delaygo", "fdgo_delaygo", "fdanti")

    # Plot self (black) — use one available ruleset as the shared reference.
    self_plotted = False
    for rs in self_ruleset_order:
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
                ax.plot(xs, all_self[i], color="black", linewidth=seed_width, alpha=self_alpha)
            mean_self = np.mean(all_self, axis=0)
            ax.plot(xs, mean_self, color="black", linewidth=mean_width,
                    label="Self" if show_legend else None)
            self_plotted = True

    # Plot cross (colored by ruleset)
    for rs in ruleset_order:
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
            ax.plot(xs, all_cross[i], color=color, linewidth=seed_width,
                    alpha=cross_alpha, linestyle=cross_style)

        mean_cross = np.mean(all_cross, axis=0)
        ax.plot(xs, mean_cross, color=color, linewidth=mean_width, linestyle=cross_style,
                label=label if show_legend else None)

    if dtype == "modulation_weighted":
        ax.set_xscale("log")
        ax.set_xlim(1, x_lim)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1000"])
    else:
        ax.set_xlim(1, x_lim)
        ax.set_xticks(x_ticks)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        _legend(ax, fontsize=7, frameon=True)


def _plot_aggregate_cve_period(period):
    """
    Draw a one-row aggregate-CVE figure for a single task period.

    Supported periods are "stimulus" and "response". The layout is hidden
    on the left and effective modulation on the right.
    """
    """
    Figure: single-period-only CVE. Single row, two columns — hidden (left)
    and effective modulation (right) — overlaying the relevant and irrelevant
    motif rulesets.
    """
    if period not in {"stimulus", "response"}:
        raise ValueError(f"Unsupported aggregate CVE period: {period}")

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
        "fdanti": "#dd6b20",
    }
    ruleset_labels = {
        "fdgo_delaygo": "Irrelevant motif",
        "fdanti_delaygo": "Relevant motif",
        "fdanti": "DelayAnti",
    }

    x_lim_map = {"hidden": 20, "modulation_weighted": 1000}
    x_tick_map = {"hidden": np.array([1, 5, 10, 15, 20]),
                  "modulation_weighted": np.arange(0, 1001, 200)}
    dtype_titles = {"hidden": "Hidden", "modulation_weighted": "Effective Modulation"}
    period_title = f"{period.capitalize()} Period"

    col_dtypes = ["hidden", "modulation_weighted"]

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.6 * 2 / 3))

    for col, dtype in enumerate(col_dtypes):
        ax = axes[col]
        _plot_aggregate_cve_panel(
            ax, by_ruleset, dtype, period, ruleset_colors, ruleset_labels,
            x_lim=x_lim_map[dtype], x_ticks=x_tick_map[dtype],
            show_legend=(col == 0))
        ax.set_title(f"{dtype_titles[dtype]} — {period_title}",
                     fontsize=8, pad=4)
        if col > 0:
            ax.set_yticklabels([])

    fig.text(0.5, 0.005, "# PCs", ha="center", fontsize=9)
    fig.text(0.005, 0.5, "MemoryAnti\nVariance Explained", va="center",
             ha="center", rotation="vertical", fontsize=9)
    fig.tight_layout(rect=[0.03, 0.04, 1, 1])
    out_path = OUT_DIR / f"aggregate_cve_{period}.png"
    _save_fig(fig, out_path)


def plot_aggregate_cve_stimulus():
    """
    Figure: stimulus-period-only CVE. Single row, two columns — hidden (left)
    and effective modulation (right) — overlaying all three motif conditions.
    """
    _plot_aggregate_cve_period("stimulus")


def plot_pretraining_principal_angles():
    """Plot saved rank-filtered principal-angle spectra in degrees.

    Rows are periods, columns are representations. Means use the common
    available spectrum length; indices order angles, not individual PCs.
    Regenerate legacy analysis results after the numerical-rank correction.
    """
    groups = {
        "fdgo_delaygo": ("Irrelevant motif", "#3182ce"),
        "fdanti_delaygo": ("Relevant motif", "#e53e3e"),
        "fdanti": ("DelayAnti", "#dd6b20"),
    }
    representations = [("hidden", "Hidden"), ("modulation_weighted", "Effective Modulation")]
    periods = ["stimulus", "response"]
    spectra = {(period, dtype, ruleset): [] for period in periods
               for dtype, _ in representations for ruleset in groups}
    for path in _pretraining_result_pkls():
        ruleset = _pretraining_ruleset_from_result_name(path.name)
        if ruleset not in groups:
            continue
        with path.open("rb") as handle:
            result = pickle.load(handle)
        for period in periods:
            for dtype, _ in representations:
                angles = np.asarray(result.get(dtype, {}).get(f"angles_{period}", []), dtype=float)
                if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all():
                    print(f"  Note: {path.name}: missing/invalid {dtype} {period} angles; omitted.")
                    continue
                spectra[period, dtype, ruleset].append(np.degrees(angles))
    if not any(spectra.values()):
        print("  Skipped: no principal angles for the configured pretraining experiment.")
        return

    _ensure_out_dir()
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True)
    counts = []
    for row, period in enumerate(periods):
        for column, (dtype, title) in enumerate(representations):
            axis = axes[row, column]
            max_length = 1
            for ruleset, (label, color) in groups.items():
                curves = spectra[period, dtype, ruleset]
                if not curves:
                    continue
                common_length = min(map(len, curves))
                max_length = max(max_length, max(map(len, curves)))
                for curve in curves:
                    axis.plot(np.arange(1, len(curve) + 1), curve, "-",
                              color=color, linewidth=0.9, alpha=0.18)
                mean = np.mean([curve[:common_length] for curve in curves], axis=0)
                axis.plot(np.arange(1, common_length + 1), mean, "-",
                          color=color, linewidth=2.2, label=label)
                counts.append(f"{period}/{dtype}/{ruleset}: n={len(curves)}, common k={common_length}")
            axis.set_xlim(0.5, max_length + 0.5)
            axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
            axis.set_ylim(-2, 92)
            axis.set_yticks([0, 30, 60, 90])
            axis.set_title(f"{title} ({period.capitalize()})", fontsize=8)
            axis.tick_params(labelsize=7)
            axis.spines[["top", "right"]].set_visible(False)
            if column == 0:
                axis.set_ylabel("Principal angle (deg)", fontsize=8)
            if row == 1:
                axis.set_xlabel("Angle index (ascending)", fontsize=8)
            if axis.get_legend_handles_labels()[0]:
                _legend(axis, fontsize=6, frameon=False)
    fig.tight_layout()
    _save_fig(fig, OUT_DIR / "pretraining_principal_angles.png", extra=f" ({'; '.join(counts)})")


def plot_aggregate_cve_response():
    """
    Figure: response-period-only CVE. Single row, two columns — hidden (left)
    and effective modulation (right) — overlaying all three motif conditions,
    with the same conventions as plot_aggregate_cve_stimulus.
    """
    _plot_aggregate_cve_period("response")


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

    def _add_period_lines(ax, spans):
        """Draw a thin dashed vertical line at each period boundary (the start
        of every period after the first), so the epoch changes are marked on
        the trace panel itself as well as in the top color strip."""
        for span in spans:
            start = span[0]
            if start and start > 0:
                ax.axvline(start, color="0.5", lw=0.8, linestyle="--", zorder=1.5)

    def _style(ax, ylabel, last_row, T, dt=1):
        ax.set_xlim(0, T - 1)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1, 1])          # only -1 and 1, as requested
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        # Thin dashed zero baseline behind the traces.
        ax.axhline(0, color="0.6", lw=0.6, linestyle="--", zorder=1)
        # Traces are plotted against step index; relabel x ticks in ms (index *
        # dt), so the axis reads real time (see SCHEME.md). Halve the tick
        # frequency by doubling the auto-chosen spacing (fewer, less crowded
        # ticks), then format each tick's index as ms.
        auto_ticks = mticker.AutoLocator().tick_values(0, T - 1)
        if len(auto_ticks) >= 2:
            ax.xaxis.set_major_locator(
                mticker.MultipleLocator((auto_ticks[1] - auto_ticks[0]) * 2))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _pos: f"{x * dt:.0f}"))
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
    # Simulation step in ms (see SCHEME.md); older pickles predate it, fall back
    # to the project default so the time axis stays in ms.
    dt_in = d.get("dt", 40)
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
        # On the Rule panel, add a flat placeholder line at y=0 for a second
        # (inactive) task cue, in a lighter orange than the active cue.
        n_leg = len(chs)
        if ylab == "Rule":
            ax.plot(np.zeros(T), color=_IO_TASK2, label="Task cue 2", zorder=2)
            n_leg += 1
        _add_period_lines(ax, period_spans)
        _style(ax, ylab, last_row=(row == len(input_groups) - 1), T=T, dt=dt_in)
        _legend(ax, fontsize=6, frameon=True, loc="upper right", ncol=n_leg)
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
    dt_out = d_out.get("dt", 40)          # sim step in ms (see SCHEME.md)
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
    # Panel y-labels by output-channel meaning. The response cosθ and sinθ
    # channels are drawn TOGETHER on one panel (labeled "Response"); Fixation
    # keeps its own panel. Mathtext $\cos\theta$ keeps cos/sin tight against θ.
    out_ylabels = ["Fixation", "Response\n" + r"$\cos\theta$",
                   "Response\n" + r"$\sin\theta$"]
    # Legend labels: the full "Response cosθ" wording when a channel stands
    # alone, and short cos/sin inside the shared Response panel (where the panel
    # y-label already says "Response").
    out_leglabels = ["Fixation", "Response " + r"$\cos\theta$",
                     "Response " + r"$\sin\theta$"]
    out_shortlabels = ["Fixation", r"$\cos\theta$", r"$\sin\theta$"]

    def _lighten(color, frac=0.55):
        """Blend a color toward white by `frac` (for the faded target shadow)."""
        r, g, b = mpl.colors.to_rgb(color)
        return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)

    # Group output channels into panels: Fixation (channel 0) on its own, and
    # the response cos/sin channels (1, 2) together on a single panel so the
    # readout's two components share a figure. Extra channels fall back to their
    # own panel each.
    n_out = net_out.shape[-1]
    if n_out >= 3:
        panels = [[0], [1, 2]] + [[c] for c in range(3, n_out)]
    else:
        panels = [[c] for c in range(n_out)]

    figout, axout = plt.subplots(len(panels), 1,
                                 figsize=(3.4, 1.5 * len(panels)),
                                 sharex=True, squeeze=False)
    for row, chans in enumerate(panels):
        ax = axout[row, 0]
        combined = len(chans) > 1
        for j, ch in enumerate(chans):
            col = out_colors[ch % len(out_colors)]
            # Faded target "shadow" behind the network trace; not in the legend.
            ax.plot(target[:, ch], color=_lighten(col), linewidth=4,
                    alpha=0.7, zorder=2, label="_nolegend_")
            leglabels = out_shortlabels if combined else out_leglabels
            lab = (leglabels[ch] if ch < len(leglabels)
                   else (out_labels[ch] if ch < len(out_labels) else f"out {ch}"))
            ax.plot(net_out[:, ch], color=col, zorder=3, label=lab)
        _add_period_lines(ax, period_spans_out)
        # Panel y-label: per-channel meaning when alone, "Response" when the
        # cos/sin components are combined on one panel.
        if combined:
            ylab = "Response"
        else:
            ch0 = chans[0]
            ylab = (out_ylabels[ch0] if ch0 < len(out_ylabels)
                    else (out_labels[ch0] if ch0 < len(out_labels) else f"out {ch0}"))
        _style(ax, ylab, last_row=(row == len(panels) - 1), T=T_out, dt=dt_out)
        _legend(ax, fontsize=6, frameon=True, loc="upper right", ncol=2)
    # Period colorbar above the top subplot (colors only, no shading behind traces).
    if period_spans_out:
        _add_period_strip(axout[0, 0], period_spans_out, xmax=T_out - 1)
    figout.tight_layout()
    out_out = OUT_DIR / "onetask_example_trial_output.png"
    _save_fig(figout, out_out)


def _draw_modulation_magnitude(series, period_spans, dt, out_path):
    """Shared renderer for the one-/two-task modulation-magnitude figures.

    `series`       : list of (label, mean(T,), std(T,), color); each drawn as a
                     line with a ±std band.
    `period_spans` : list of (start, end_or_None, color) in STEP-INDEX units; used
                     for both the dashed period-boundary lines and the top color
                     strip. Boundaries are scaled to ms (× dt) to match the axis.
    `dt`           : simulation step in ms (see SCHEME.md); the time axis is ms.

    Styling matches the example-trial illustration: dashed period lines, a period
    color strip above the panel, and x ticks at the illustration's frequency (the
    auto spacing doubled). Saves to `out_path`.
    """
    T = len(series[0][1]) if series else 0
    t_ms = np.arange(T) * dt
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for lab, mean, std, col in series:
        ax.plot(t_ms, mean, "-", color=col, label=lab, zorder=2)
        ax.fill_between(t_ms, mean - std, mean + std,
                        color=col, alpha=0.2, lw=0, zorder=1)
    # Dashed vertical lines at each period boundary (in ms).
    for span in period_spans:
        start = span[0]
        if start and start > 0:
            ax.axvline(start * dt, color="0.5", lw=0.8, linestyle="--", zorder=1.5)
    ax.set_xlim(0, (T - 1) * dt)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Modulation magnitude", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    # X ticks at the same frequency as the input/output illustration: take the
    # auto-chosen ms spacing and double it (fewer, less crowded ticks); the axis
    # is already in ms so no per-tick rescale is needed (see SCHEME.md).
    auto_ticks = mticker.AutoLocator().tick_values(0, (T - 1) * dt)
    if len(auto_ticks) >= 2:
        ax.xaxis.set_major_locator(
            mticker.MultipleLocator((auto_ticks[1] - auto_ticks[0]) * 2))
    _legend(ax, fontsize=6, frameon=True, loc="upper right", ncol=2)
    # Period colorbar above the panel (colors only). The x-axis is in ms, so scale
    # the step-index span boundaries by dt to match (None runs to the axis end).
    if period_spans:
        spans_ms = [(s * dt, (None if e is None else e * dt), c)
                    for s, e, c in period_spans]
        _add_period_strip(ax, spans_ms, xmax=(T - 1) * dt)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_onetask_modulation_magnitude():
    """
    Figure: modulation-computation magnitude across trial time, one curve per
    input MEANING. For each input channel c, the plastic matrix M's modulation of
    that channel is the hidden-unit vector M · W_input[:, c]; its L2 magnitude over
    hidden units (mean ± std across trials) shows how strongly each input drives
    the plastic weights over the trial. The two stimulus modalities' cos/sin
    channels are combined into a SINGLE "Stimulus" trajectory (per-trial mean over
    that block), so the figure shows three curves — Fixation, Stimulus, Task cue —
    colored to match the example-trial input figure. Each curve carries a ±std
    band. Reloaded from modulation_magnitude_{aname}.pkl written by
    one_task_analysis.py.
    """
    _ensure_out_dir()
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"modulation_magnitude_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return

    channels = list(d["channels"])                 # raw input indices, in order
    labels = list(d["labels"])
    mag_mean = np.asarray(d["mag_mean"])            # (T, n_raw)
    dt = int(d.get("dt", 40))                       # sim step in ms (see SCHEME.md)
    T = mag_mean.shape[0]
    # Across-trial std band. Older pickles stored SEM (std/√n) under mag_sem; fall
    # back to it (approximate) so a stale pickle still renders.
    if "mag_std" in d:
        mag_std = np.asarray(d["mag_std"])          # (T, n_raw)
    else:
        mag_std = np.asarray(d.get("mag_sem", np.zeros_like(mag_mean)))

    fix_ch = 0
    task_ch = max(channels)
    # Combined stimulus trajectory: prefer the per-trial mean series (and its std)
    # saved by one_task_analysis.py. Older pickles predate it — fall back to the
    # (approximate) mean of the per-channel MEANS over the stimulus channels
    # (every component that is neither Fixation nor the Task cue), std unavailable.
    stim_cols = [i for i, (ch, lab) in enumerate(zip(channels, labels))
                 if lab not in ("Fixation", "Task cue")]
    if "stim_mag_mean" in d:
        stim_mean = np.asarray(d["stim_mag_mean"])
        stim_std = np.asarray(d.get("stim_mag_std", d.get("stim_mag_sem", np.zeros(T))))
    else:
        stim_mean = mag_mean[:, stim_cols].mean(axis=1) if stim_cols else np.zeros(T)
        stim_std = np.zeros(T)

    # Colors matched to the example-trial input figure: Fixation gray, the
    # combined Stimulus in the shared stimulus green, Task cue orange.
    series = [
        ("Fixation", mag_mean[:, fix_ch], mag_std[:, fix_ch], _IO_FIXATION),
        ("Stimulus", stim_mean, stim_std, _IO_MOD2[0]),
        ("Task cue", mag_mean[:, task_ch], mag_std[:, task_ch], _IO_TASK),
    ]

    # Period boundaries (stimulus / memory / response onsets), in step-index units.
    ss = d.get("stimulus_start")
    se = d.get("stimulus_end")
    rs = d.get("response_start")
    fix_c, stim_c, mem_c, resp_c = _ONETASK_PERIOD_COLORS
    period_spans = []
    if ss is not None and se is not None and rs is not None:
        period_spans = [(0, ss, fix_c), (ss, se, stim_c),
                        (se, rs, mem_c), (rs, None, resp_c)]

    _draw_modulation_magnitude(series, period_spans, dt,
                               OUT_DIR / "onetask_modulation_magnitude.png")


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
    # Show stimulus 5 and 2 (in that order), by label value.
    ONETASK_SHOW_STIM = [5, 2]
    sel_labels = [s for s in ONETASK_SHOW_STIM if s in per_stim]
    if len(sel_labels) < len(ONETASK_SHOW_STIM):
        missing = [s for s in ONETASK_SHOW_STIM if s not in per_stim]
        print(f"  Skipped: requested stimuli {ONETASK_SHOW_STIM} but "
              f"{missing} not saved (available: {stim_labels}).")
        return

    # Simulation step in ms (see SCHEME.md); the show pickle predates a saved dt,
    # so read it from the run's param json to relabel the x-axis in ms.
    dt = _read_onetask_dt()

    # Trial periods: fixation | stimulus | memory(delay) | response, bounded by
    # the saved break times, drawn as a top colorbar (colors only, no shading).
    period_spans = []
    if stimulus_start is not None and stimulus_end is not None and response_start is not None:
        fix_c, stim_c, mem_c, resp_c = _ONETASK_PERIOD_COLORS
        period_spans = [
            (0, stimulus_start, fix_c, "Context"),
            (stimulus_start, stimulus_end, stim_c, "Stimulus"),
            (stimulus_end, response_start, mem_c, "Memory"),
            (response_start, None, resp_c, "Response"),
        ]

    # Panels laid out HORIZONTALLY (one column per stimulus), styled like the
    # example-trial illustration: each panel gets the period color strip and
    # dashed period-boundary lines; the x-axis is relabeled in ms.
    fig, axes = plt.subplots(1, len(sel_labels),
                             figsize=(3.0 * len(sel_labels), 2.0),
                             sharey=True, squeeze=False)
    for i, lab in enumerate(sel_labels):
        ax = axes[0, i]
        tr = per_stim[lab]
        T = len(tr["combine"])
        # Colors + names matched to onetask_example_trial's input channels:
        # Fixation → _IO_FIXATION (dark gray), Rule → _IO_TASK (orange).
        # Combine uses _IO_COMBINE (deep blue), a hue used by no input channel.
        ax.plot(tr["fixon"], color=_IO_FIXATION, label="Fixation", zorder=2)
        ax.plot(tr["task"], color=_IO_TASK, label="Rule", zorder=2)
        ax.plot(tr["combine"], color=_IO_COMBINE, linewidth=2.5, label="Combine", zorder=3)
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        # Dashed vertical lines at each period boundary (stimulus/memory/response
        # onsets), matching the example-trial illustration.
        for span in period_spans:
            start = span[0]
            if start and start > 0:
                ax.axvline(start, color="0.5", lw=0.8, linestyle="--", zorder=1.5)
        ax.set_xlim(0, T - 1)
        ax.set_ylim([-1.5, 1.5])
        ax.set_yticks([-1, 0, 1])       # only -1, 0, 1 ticklabels
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        # X ticks at the same frequency as the input/output illustration: take
        # the auto-chosen spacing and double it (fewer, less crowded ticks), then
        # relabel each tick's step index in ms (index * dt); see SCHEME.md.
        auto_ticks = mticker.AutoLocator().tick_values(0, T - 1)
        if len(auto_ticks) >= 2:
            ax.xaxis.set_major_locator(
                mticker.MultipleLocator((auto_ticks[1] - auto_ticks[0]) * 2))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _pos: f"{x * dt:.0f}"))
        # Period colorbar above every panel.
        if period_spans:
            _add_period_strip(ax, period_spans, xmax=T - 1)
        if i == 0:
            _legend(ax, frameon=True, fontsize=6, loc="best")
            ax.set_ylabel("Readout projection", fontsize=9)

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
    _save_standalone_colorbar(
        OUT_DIR / out_name, cmap="bwr", vmin=-vmax, vmax=vmax,
        ticks=[-vmax, 0, vmax],
        ticklabels=[f"{-vmax:.2g}", "0", f"{vmax:.2g}"])


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
                   markersize=10, markerfacecolor="k", markeredgecolor="k",
                   label=_period_display(name))
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
                   markersize=10, markerfacecolor="k", markeredgecolor="k",
                   label=_period_display(name))
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
    Each 4x4 matrix (Context/Stimulus/Memory/Response) shows how well each
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

    # Panels STACKED VERTICALLY, one representation per row, so the two share one
    # x axis: the period tick labels are identical on both, and printing them once
    # under the bottom panel leaves the columns aligned down the figure.
    # constrained layout, not tight: it reserves room for the figure-level
    # supxlabel/supylabel, which tight_layout ignores — they would otherwise be
    # painted straight over the period tick labels.
    # No per-panel titles: the row order is fixed (hidden, then effective
    # modulation) and the caption names them, so the titles only cost height that
    # the matrices themselves can use.
    # Height per panel is matched to what `square=True` actually draws (the axes
    # box is as tall as it is wide, and the width is the figure minus the y tick
    # labels): any more and the squares float in vertical slack, opening a gap
    # between the two panels. The +0.55 is the rotated x tick labels plus supxlabel.
    fig, axs = plt.subplots(len(names), 1,
                            figsize=(1.9, 1.15 * len(names) + 0.55),
                            layout="constrained", squeeze=False)
    axs = axs[:, 0]
    for row, (ax, name) in enumerate(zip(axs, names)):
        e = d_combine[name]
        # Period names, in the display vocabulary (older pickles say "Fixation").
        plabels = [_period_display(v) for v in e["labels"]]
        last_row = row == len(names) - 1
        # No per-cell values (annot=False) — the color carries the FVE, and the
        # standalone colorbar reads it. Matches plot_two_task_d_combine.
        sns.heatmap(np.asarray(e["fve_k_all"]), ax=ax,
                    xticklabels=plabels if last_row else False,
                    yticklabels=plabels,
                    annot=False,
                    vmin=vmin, vmax=vmax, square=True,
                    cmap="mako", cbar=False)
        if last_row:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right",
                               rotation_mode="anchor", fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        # Highlight each period tick label with that period's own bar color, so a
        # row/column can be tied to the period strip of the input/output figure
        # without reading the word. Applied AFTER set_x/yticklabels, which
        # replaces the label artists. Same treatment plot_two_task_d_combine uses.
        if last_row:
            _color_period_ticklabels(ax, plabels, axis="x")
        _color_period_ticklabels(ax, plabels, axis="y")

    # One shared pair of axis labels for the stack. fve_k[i, j] is period i's
    # variance captured by the top-k subspace fit on period j, so COLUMNS are the
    # period the subspace was fit on and ROWS are the period it is then tested
    # against — hence x = Fit, y = Test, not the other way round.
    fig.supxlabel("Fit", fontsize=9)
    fig.supylabel("Test", fontsize=9)
    out_path = OUT_DIR / "onetask_d_combine.png"
    _save_fig(fig, out_path)

    # The colorbar goes out as its OWN figure rather than beside the panels: both
    # panels share one 0-1 scale, so a bar inside the stack would either squeeze
    # the heatmaps or be drawn twice, and a standalone bar can be placed and sized
    # in the manuscript independently of them.
    _save_standalone_colorbar(
        OUT_DIR / "onetask_d_combine_colorbar.png", cmap="mako",
        vmin=vmin, vmax=vmax, label="Frac. var. explained", labelsize=7)


def plot_onetask_pc_cumvar():
    """
    Figure: cumulative variance explained vs number of PCs, per trial period
    (the right panel of the cross-period dimensionality analysis). For each
    representation (hidden, effective modulation) a single panel plots, for each
    period, the fraction of that period's variance captured by its own top 1..N
    PCs — one curve per period (Context / Stimulus / Memory / Response), colored
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

    # Period name -> period-bar color (Context/Stimulus/Memory/Response order).
    period_color = dict(zip(["Context", "Stimulus", "Memory", "Response"],
                            _ONETASK_PERIOD_COLORS))

    # Panels STACKED VERTICALLY, one representation per row, sharing the x axis:
    # both plot the same 1..N PC index, so the ticks and the "No. of PCs" label
    # are drawn once under the bottom panel (sharex hides the upper row's).
    # No per-panel titles (same as onetask_d_combine: fixed row order, named in
    # the caption). Panel height is HALF the previous ~1.13 in of drawn axes; the
    # additive 0.45 is the shared x tick labels plus the supxlabel, which is fixed
    # overhead and must stay out of the per-panel term or "half" would shrink it too.
    fig, axs = plt.subplots(len(names), 1,
                            figsize=(1.7, 0.57 * len(names) + 0.45),
                            sharex=True, layout="constrained", squeeze=False)
    # Remember the period label/color order so the standalone legend below
    # matches the drawn curves exactly.
    legend_labels, legend_colors = None, None
    for ax, name in zip(axs[:, 0], names):
        e = d_combine[name]
        cumvar = np.asarray(e["cumvar"], dtype=float)      # (n_period, max_pc)
        # Normalize to the period vocabulary BEFORE the color lookup: pickles
        # written before the Context rename store "Fixation", which would miss
        # `period_color` and drop that curve onto the categorical fallback (red).
        labels = [_period_display(v)
                  for v in e.get("labels",
                                 ["Context", "Stimulus", "Memory", "Response"])]
        n_pc = cumvar.shape[1]
        xs = np.arange(1, n_pc + 1)
        cols = [period_color.get(lab, c_vals[i % len(c_vals)])
                for i, lab in enumerate(labels)]
        for i, lab in enumerate(labels):
            ax.plot(xs, cumvar[i], "-o", color=cols[i], markersize=3, label=lab)
        if legend_labels is None:
            legend_labels, legend_colors = labels, cols
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
    row = e_modulation; columns = periods (Context/Stimulus/Delay/Response).
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
                ax.set_title(_period_display(period_title.get(v, v)), fontsize=11)

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
    (periods, overlay, proj_by_period, traj_by_period, angle0_pt, n_stim):
      periods           : the trial periods that get a panel (see
                          _grad_fp_period_panels)
      overlay[period]   : other probes solved under that period's input, drawn in
                          the SAME panel
      proj_by_period[v] : (batch, 2) PCA coords of that probe's fixed points —
                          keyed by probe, so it covers the overlaid ones too
      traj_by_period[v] : (win_T, 2) exemplar (angle-0) within-period trajectory,
                          present only where the pickle saved it (period probes)
      angle0_pt[v]      : (2,) the exemplar-stimulus fixed point, for connectors
      n_stim            : stimulus-color count (dense ring size)
    Pure data prep — drawing lives in _draw_grad_fp_2d_row so several rules can
    share one figure."""
    results = d["results"]
    periods, overlay = _grad_fp_period_panels(results)
    drawn = periods + [n for p in periods for n in overlay.get(p, [])]

    def _flat(arr):
        arr = np.asarray(arr, dtype=float)
        return arr.reshape(arr.shape[0], -1)

    n_stim = 1 + max(int(s) for v in drawn for s in np.asarray(results[v]["stim"]))
    proj_by_period = {v: pca.transform(_flat(results[v][rep_key])) for v in drawn}

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

    return periods, overlay, proj_by_period, traj_by_period, angle0_pt, n_stim


def _draw_grad_fp_2d_row(axs_row, results, periods, proj_by_period, traj_by_period,
                         angle0_pt, n_stim, lim, show_period_titles=True,
                         row_label=None, overlay=None):
    """Draw one rule's four per-period 2D panels into the pre-created axes
    `axs_row` (length = n_col), using precomputed projections. Shared symmetric
    limit `lim` is passed in so multiple rows use IDENTICAL axes.
    `show_period_titles` prints the Context/Stimulus/… titles (typically only
    the top row); `row_label` writes a rotated label (e.g. the task rule) to the
    left of the row's first panel.

    `pc_label` prefixes the x/y axis labels (e.g. "Delay" -> "Delay PC1"), naming
    the period whose PCA defines the shared basis; None keeps the bare "PC1"/"PC2".

    `overlay[period]` (from _grad_fp_period_panels) lists further probes solved
    under that period's input — they are drawn into the SAME panel, marker-coded,
    so every fixed point belonging to a period appears in that period's panel."""
    _TRAJ_STIM = 0
    _traj_col = stim_color(_TRAJ_STIM, n_stim)
    overlay = overlay or {}
    for j, (ax, v) in enumerate(zip(axs_row, periods)):
        e = results[v]
        # This period's own probe first, then any probe solved under the same
        # input from a different starting state (memory-seeded, naive seeds).
        panel_probes = [v] + list(overlay.get(v, []))
        for name in panel_probes:
            pe = results[name]
            xy = proj_by_period[name]               # (batch, 2)
            # Naive-seeded probes have no stimulus label, so they are colored by
            # the reference angle they landed on (gray if they landed nowhere near
            # it) rather than by a `stim` that is only a seed index.
            cols, _ = _grad_fp_point_colors(pe, np.asarray(pe["stim"]), n_stim)
            _scatter_grad_fp(ax, xy, _grad_fp_probe_style(pe, name, base_s=18),
                             cols, _fixed_point_mask(pe, xy.shape[0]))
        _grad_fp_overlay_legend(ax, results, panel_probes)
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
                # Start = cross, end = arrowhead. The two ends used to be told
                # apart by their outline (dashed vs solid black), but fixed-point
                # markers carry no outline any more, so the distinction moves to
                # SHAPE — the same treatment the 3D renderer uses.
                ax.scatter([p0[0]], [p0[1]], color=_traj_col, marker="x", s=26,
                           linewidth=1.4, zorder=5)
                _pts = np.column_stack([xs, ys])
                # Aim the head from 85% of the way along the path, not from the
                # final segment: these paths cover most of their distance in the
                # first few steps and then creep, so the last segment alone can
                # point almost anywhere.
                _sl = np.linalg.norm(np.diff(_pts, axis=0), axis=1)
                _tot = float(_sl.sum())
                if _tot > 1e-9:
                    _cum = np.concatenate([[0.0], np.cumsum(_sl)])
                    _k = int(np.clip(np.searchsorted(_cum, 0.85 * _tot),
                                     0, len(_pts) - 2))
                    ax.annotate("", xy=(p1[0], p1[1]), xytext=tuple(_pts[_k]),
                                arrowprops=dict(arrowstyle="-|>", color=_traj_col,
                                                lw=1.1, shrinkA=0, shrinkB=0),
                                zorder=5)
        if show_period_titles:
            ax.set_title(_period_display(e.get("period_title", v)), fontsize=11)
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

    `basis`: an OPTIONAL stored 2-component PCA to project into. When None
    (the default, and the one-task behavior), load THIS pickle's saved
    delay-period basis. When supplied (e.g. by the two-task driver,
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

    if basis is None:
        basis = _load_period_grad_fp_basis(d, rep_key)
    if basis is None:
        return

    periods, overlay, proj, traj, angle0_pt, n_stim = _grad_fp_2d_project(
        d, rep_key, basis)
    lim = max(np.abs(np.vstack(list(proj.values()))).max() * 1.08, 1e-9)

    n_col = len(periods)
    # Match onetask_long_fixed_points' compact panel size.
    fig, axs = plt.subplots(1, n_col, figsize=(2.1 * n_col, 2.1), squeeze=False)
    _draw_grad_fp_2d_row(axs[0], results, periods, proj, traj, angle0_pt, n_stim,
                         lim, show_period_titles=True, row_label=None,
                         overlay=overlay)

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


# ─── Grouping several probes into ONE period panel ────────────────────────────
# The solver's battery has more entries than there are trial periods: besides each
# period's own ("diagonal") probe it holds probes solved under the SAME period
# input from a different starting state. Those belong in that period's subplot, so
# every fixed point of a period is drawn together and the panel layout stays one
# panel per period.
#
# Hue is reserved for stimulus direction (see SCHEME.md), so within a panel the
# probe is distinguished by MARKER, and fill keeps its usual meaning (filled =
# converged, hollow = over the rel_step threshold). Markers drawn as strokes have
# no face, so they take a plain `color=` and ignore the fill/edge distinction.
_GRAD_FP_STROKE_MARKERS = ("x", "+", "1", "2", "3", "4", "|", "_")
# A naive-seeded point farther than this relative distance from its reference
# fixed points did not land on them, so the ring angle it is labelled with is
# meaningless and it is drawn gray instead (matches the solver's 10% log line).
_GRAD_FP_RING_TOL = 0.1


def _grad_fp_probe_style(entry, name="", base_s=18):
    """Marker spec for one probe drawn inside its period's panel.

    The period's OWN probe keeps the original look — a small filled circle — so
    panels are pixel-identical wherever no extra probe exists. An overlaid probe
    gets a distinct shape: a large OPEN circle when it was seeded from another
    period's state (its points should be seen landing ON the structure they came
    from), an × for stimulus-free naive seeds. `base_s` scales to each figure's
    own point size (the 3D panels draw smaller)."""
    if entry.get("is_diagonal", True):
        return dict(marker="o", s=base_s, fill=True, z=3)
    if entry.get("seed_source") == "naive_rank1" or name.endswith("_naiveseed"):
        return dict(marker="x", s=base_s * 1.4, fill=True, z=2)
    return dict(marker="o", s=base_s * 4.0, fill=False, z=2)


def _grad_fp_seed_label(entry):
    """Short legend label naming where a probe's optimizer STARTED, which is the
    only thing that differs between the probes sharing a panel."""
    if entry.get("is_diagonal", True):
        return "own state"
    src = str(entry.get("seed_source", "") or "other")
    if src == "naive_rank1":
        return "naive seeds"
    stem = src[4:] if src.startswith("long") else src
    return f"{stem.capitalize()} seed"


def _grad_fp_same_input(results, a, b, tol=0.0):
    """Whether probes `a` and `b` were solved under the same constant input,
    measured from the saved `const_input` (False when a pickle predates it). Two
    probes at distance 0 pose the same fixed-point problem, so a result about one
    is a result about the other."""
    ca, cb = results[a].get("const_input"), results[b].get("const_input")
    if ca is None or cb is None:
        return False
    ca, cb = np.asarray(ca, dtype=float), np.asarray(cb, dtype=float)
    if ca.shape != cb.shape:
        return False
    return float(np.abs(ca - cb).max()) <= tol


def _grad_fp_period_panels(results):
    """Split a grad-fp pickle's probes into (panels, overlay).

    `panels` is the list of trial periods that each get their own subplot — the
    original figure layout. `overlay[panel]` lists the OTHER probes drawn in that
    panel (e.g. the memory-seeded and naive-seeded fixation probes both belong to
    the Context panel), so a period's fixed points are drawn together rather than
    spread over extra panels.

    A naive-seeded probe characterizes its INPUT's whole fixed-point set, and
    periods can share an input — fixation and delay do in the delaygo family, which
    is why the solver only solves that probe once. It is therefore mirrored into
    every panel with an identical input: the naive fixed points of the fixation
    input ARE the naive fixed points of the delay input. State-seeded probes are not
    mirrored, because their points already appear in the panel they were seeded
    from, as that panel's own probe.

    Probes name their period in `input_period`; a pickle predating the probe
    battery has exactly one entry per period and yields an empty overlay, leaving
    every figure unchanged. A probe whose period has no panel of its own (its
    diagonal probe was disabled) becomes its own panel rather than being dropped."""
    panels, overlay = [], {}
    for name, entry in results.items():
        panel = entry.get("input_period", name)
        if panel == name:
            panels.append(name)
        else:
            overlay.setdefault(panel, []).append(name)
    for panel in list(overlay):
        if panel not in panels:
            panels.extend(overlay.pop(panel))

    # Synthesized-seed probes (naive, and the RNN's trajectory-seeded family)
    # characterize their INPUT's whole fixed-point set rather than one recorded
    # state, so each is mirrored into every panel sharing that input.
    synth = [n for names in overlay.values() for n in names
             if results[n].get("seed_source") in ("naive_rank1", "traj_noise")
             or n.endswith(("_naiveseed", "_trajseed"))]
    for name in synth:
        home = results[name].get("input_period", name)
        for panel in panels:
            if panel == home or name in overlay.get(panel, []):
                continue
            if _grad_fp_same_input(results, name, panel):
                overlay.setdefault(panel, []).append(name)
    return panels, overlay


def _scatter_grad_fp(ax, xy, style, cols, good, z_vals=None, alpha=0.85):
    """Scatter one probe's fixed points: 2D, or 3D when `z_vals` is given.

    Converged points are filled and over-threshold ones hollow — the long-standing
    convention. Filled points carry NO outline: an outline on a small marker eats
    into the fill, and at 64 points a ring of hairlines reads as its own texture.
    The hollow markers keep their colored edge because there the edge IS the
    marker. `alpha` defaults to the long-standing 0.85."""
    stroke = style["marker"] in _GRAD_FP_STROKE_MARKERS
    # zorder is meaningless for 3D axes, and the 3D panels never set it.
    zkw = {} if z_vals is not None else {"zorder": style["z"]}
    for i in range(xy.shape[0]):
        pos = ((xy[i, 0], xy[i, 1]) if z_vals is None
               else (xy[i, 0], xy[i, 1], z_vals[i]))
        if stroke:
            ax.scatter(*pos, marker=style["marker"], s=style["s"], color=cols[i],
                       linewidth=0.9, alpha=alpha, **zkw)
        elif style["fill"] and good[i]:
            ax.scatter(*pos, marker=style["marker"], s=style["s"], color=cols[i],
                       edgecolor="none", alpha=alpha, **zkw)
        else:
            ax.scatter(*pos, marker=style["marker"], s=style["s"],
                       facecolor="none", edgecolor=cols[i], linewidth=0.9,
                       alpha=alpha, **zkw)


def _grad_fp_overlay_legend(ax, results, names, base_s=18, fontsize=6):
    """Marker legend for a panel holding more than one probe. Handles are gray so
    the legend reads the MARKER; hue belongs to the stimulus scale."""
    if len(names) < 2:
        return
    handles = []
    for name in names:
        entry = results[name]
        style = _grad_fp_probe_style(entry, name, base_s=base_s)
        stroke = style["marker"] in _GRAD_FP_STROKE_MARKERS
        label = _grad_fp_seed_label(entry)
        n_on_ring = entry.get("ring_dist")
        if n_on_ring is not None:
            rd = np.asarray(n_on_ring, dtype=float)
            label += (f" [{int((rd <= _GRAD_FP_RING_TOL).sum())}/{rd.size} on "
                      f"{entry.get('ring_ref', 'ref')}]")
        handles.append(plt.Line2D(
            [0], [0], marker=style["marker"], linestyle="None",
            markerfacecolor=("0.35" if (style["fill"] and not stroke) else "none"),
            markeredgecolor="0.35", markersize=np.sqrt(style["s"]) * 0.9,
            label=label))
    _legend(ax, handles=handles, frameon=True, fontsize=fontsize, loc="best")


# Added probes keep the SAME circle marker and size as every other fixed point in
# the 3D figures and are told apart by color alone (gray), because a 3D panel
# already spends its budget on a z axis and a viewing angle; extra marker shapes
# and point clouds there only obscure it. The 2D figures keep the full battery,
# marker-coded.
#
# Which added probes each 3D panel draws, named explicitly rather than derived, so
# the content of every panel is decided in exactly one place:
#   Context — the ring-alike memory-seeded ring (the point-and-ring coexistence),
#              plus the naive shell.
#   Delay    — the same naive shell. Delay shares fixation's input (the solver
#              measures the distance as 0), so that single solve is this panel's
#              naive result too — see _grad_fp_period_panels. This sharing is
#              licensed by that measurement, which is why it does not extend to the
#              response panel: its input differs by construction (fixation off), so
#              the fixation-input solutions are NOT fixed points of its map, and
#              drawing them there would put an untested claim under the same gray
#              circles the other panels use for genuine ones.
#   Response — its OWN naive-seeded solve, actually solved under the response input.
#   Stimulus — none; its own added ring is a second lower-amplitude branch that the
#              2D figures carry.
_GRAD_FP_3D_OVERLAY_PROBES = {
    "longfixation": ("longfixation_memseed", "longfixation_trajseed",
                     "longfixation_naiveseed"),
    "longdelay": ("longfixation_trajseed", "longfixation_naiveseed"),
    "longresponse": ("longresponse_trajseed", "longresponse_naiveseed"),
}
_GRAD_FP_3D_OVERLAY_COLOR = "0.55"

# Which stimulus is the exemplar whose within-period trajectory the 3D panels draw.
# Module-level so the renderers' axis-limit computation and _draw_grad_fp_3d_row
# pick the SAME anchor point (they must, or a panel can draw a point outside its
# own axes box).
_GRAD_FP_TRAJ_STIM = 0


def _grad_fp_3d_anchor_xyz(results, periods, draw_periods, proj_by_period,
                           z_by_period):
    """The trajectory anchor each drawn panel inherits from the period before it.

    `_draw_grad_fp_3d_row` starts every panel's exemplar trajectory at the previous
    period's `_GRAD_FP_TRAJ_STIM` fixed point and draws that point inside the panel.
    When the previous period has no panel of its own (fixation, delay), its points
    are otherwise absent from the figure's shared limits — so a caller that trims
    panels must fold these back in, or the anchor is drawn outside the axes box.

    Returns (xy, z) as ((k, 2), (k,)) arrays over the drawn panels that have one."""
    xy, z = [], []
    for v in draw_periods:
        i = periods.index(v)
        if i < 1:
            continue                      # first period inherits no anchor
        prev = periods[i - 1]
        if prev not in proj_by_period or prev not in z_by_period:
            continue
        st = np.asarray(results[prev]["stim"], dtype=int)
        idx = np.where(st == _GRAD_FP_TRAJ_STIM)[0]
        if not idx.size:
            continue
        i0 = int(idx[0])
        xy.append(proj_by_period[prev][i0, :2])
        z.append(float(np.asarray(z_by_period[prev], dtype=float)[i0]))
    if not xy:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(xy), np.asarray(z, dtype=float)


def _grad_fp_3d_overlay(period, overlay, results):
    """Which added probes get drawn in `period`'s 3D panel, per
    _GRAD_FP_3D_OVERLAY_PROBES (a period absent from it shows none). Names missing
    from this pickle are skipped, so an older or reduced battery still renders.

    `overlay` is accepted for signature parity with the 2D path and as the fallback
    when the table names nothing for this period."""
    wanted = _GRAD_FP_3D_OVERLAY_PROBES.get(period)
    if wanted is None:
        return []
    return [n for n in wanted if n in results]


def _grad_fp_3d_colors(period, entry, stim, n_stim):
    """Point colors for one probe in a 3D panel. Gray when the probe is an ADDED
    one (its points are not fixed points the trial itself reaches) or when the
    panel is FIXATION — that input carries no stimulus at all, so no hue in that
    panel would be earned. Contrast the Delay panel: the SAME input, but its points
    are seeded from a state that remembers the angle, so there the rainbow is
    earned. Every other panel colors its own probe by stimulus as always."""
    if "fixation" in period.lower() or not entry.get("is_diagonal", True):
        return [_GRAD_FP_3D_OVERLAY_COLOR] * len(stim)
    return _grad_fp_point_colors(entry, stim, n_stim)[0]


def _grad_fp_point_colors(entry, stim, n_stim):
    """(colors, n_on_ring) for one probe entry's points.

    Period-seeded probes are colored by stimulus direction. A naive-seeded probe
    has no stimulus — its `stim` is only a seed index — so its points take the
    color of the reference angle they LANDED on (`ring_angle_idx`), and gray when
    they landed nowhere near the reference (`ring_dist` > tol) or when the pickle
    predates that annotation. `n_on_ring` is None for period-seeded probes."""
    if entry.get("stim_is_stimulus", True):
        return [stim_color(int(s), n_stim) for s in stim], None
    ring_idx, ring_dist = entry.get("ring_angle_idx"), entry.get("ring_dist")
    if ring_idx is None:
        return ["0.55"] * len(stim), None
    ring_idx = np.asarray(ring_idx, dtype=int)
    ring_dist = (np.asarray(ring_dist, dtype=float) if ring_dist is not None
                 else np.zeros(ring_idx.size))
    cols = [stim_color(int(a), n_stim) if dd <= _GRAD_FP_RING_TOL else "0.55"
            for a, dd in zip(ring_idx, ring_dist)]
    return cols, int((ring_dist <= _GRAD_FP_RING_TOL).sum())


def _load_fixed_point_pca_record(data, rep_key, period):
    """Read a matching analysis-exported PCA record, never fitting a substitute."""
    source_name = data.get("_fixed_point_source")
    if source_name is None:
        print("  Skipped: fixed-point PCA source path unavailable; load through _load_pkl_or_skip.")
        return None
    source = Path(source_name)
    sidecar = source.with_suffix(".pca.npz")
    try:
        stat = source.stat()
        signature = {"name": source.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        artifact = data.get("_fixed_point_pca")
        if artifact is None:
            with np.load(sidecar, allow_pickle=True) as saved:
                artifact = saved["artifact"].item()
        if artifact.get("schema_version") != 1 or artifact.get("source") != signature:
            raise ValueError("PCA sidecar is outdated or belongs to a different source")
        record = artifact["bases"][period][rep_key]
        values = data["results"][period][rep_key]
        if (tuple(record["source_shape"]) != np.shape(values)
                or record["source_period"] != period or record["representation"] != rep_key
                or record["source_aname"] != data.get("aname")
                or record["source_rule"] != data.get("rule")):
            raise ValueError("PCA provenance does not match the requested fixed points")
        data["_fixed_point_pca"] = artifact
        return record
    except (OSError, KeyError, ValueError, TypeError) as error:
        print(f"  Skipped {period}/{rep_key}: missing or incompatible {sidecar} ({error}). "
              f"Run: python core/fixed_point_pca.py '{source}' --force")
        return None


def _load_period_grad_fp_basis(data, rep_key, period="longdelay"):
    """Load the requested period's two-PC basis from its fixed-point PCA sidecar."""
    record = _load_fixed_point_pca_record(data, rep_key, period)
    return _StoredPCABasis(record) if record is not None else None


class _StoredPCABasis:
    """Minimal transform-only PCA reconstructed from an analysis artifact."""

    def __init__(self, record, n_components=2):
        self.mean_ = np.asarray(record["mean"], dtype=float)
        components = np.asarray(record["components"], dtype=float)
        n_components = int(n_components)
        if components.ndim != 2 or components.shape[0] < n_components:
            raise ValueError(f"expected at least {n_components} PCA components, "
                             f"got {components.shape}")
        self.components_ = components[:n_components]
        if self.mean_.shape != (self.components_.shape[1],):
            raise ValueError(f"PCA mean/components disagree: {self.mean_.shape} "
                             f"vs {self.components_.shape}")

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.mean_.size:
            raise ValueError(f"PCA expected (*, {self.mean_.size}), got "
                             f"{values.shape}")
        return (values - self.mean_) @ self.components_.T


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
    rules can share one figure. Like the 2D version it returns
    (periods, overlay, ...): `periods` are the periods that get a panel and
    `overlay[period]` the further probes drawn inside that same panel, while the
    projection/z dicts are keyed by probe so they cover both."""
    results = d["results"]
    periods, overlay = _grad_fp_period_panels(results)
    drawn = periods + [n for p in periods for n in overlay.get(p, [])]

    def _flat(arr):
        arr = np.asarray(arr, dtype=float)
        return arr.reshape(arr.shape[0], -1)

    n_stim = 1 + max(int(s) for v in drawn for s in np.asarray(results[v]["stim"]))
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
        else 0 (the task only demands an output during Response).

        The period comes from the entry's `input_period`, not from the probe name:
        an overlaid probe such as "longresponse_naiveseed" is solved under the
        response input but its `stim` is only a seed index, so its angle is taken
        from the reference point it LANDED on (ring_angle_idx) — the same source
        as its color — and a point that landed nowhere near the reference gets 0
        rather than a fabricated height."""
        stim = np.asarray(e["stim"], dtype=int)
        period = str(e.get("input_period", v))
        if "response" not in period.lower():
            return np.zeros(stim.shape[0], dtype=float)
        if e.get("stim_is_stimulus", True):
            idx, valid = stim, np.ones(stim.shape[0], dtype=bool)
        else:
            ring_idx, ring_dist = e.get("ring_angle_idx"), e.get("ring_dist")
            if ring_idx is None:
                return np.zeros(stim.shape[0], dtype=float)
            idx = np.asarray(ring_idx, dtype=int)
            valid = (np.asarray(ring_dist, dtype=float) <= _GRAD_FP_RING_TOL
                     if ring_dist is not None
                     else np.ones(idx.shape[0], dtype=bool))
        if dense_angles.size and int(idx.max()) < dense_angles.size:
            ang = dense_angles[idx]
        else:
            ang = 2.0 * np.pi * idx / max(n_stim, 1)
        return np.where(valid, np.cos(ang + resp_offset), 0.0)

    proj_by_period = {v: pca.transform(_flat(results[v][rep_key])) for v in drawn}
    z_by_period = {v: _target_cos(v, results[v]) for v in drawn}

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

    return periods, overlay, proj_by_period, z_by_period, traj_by_period, n_stim


def _rotate_hidden_3d_content(rep_key, proj, traj):
    """For the hidden-state representation only, rotate the horizontal (PC1-PC2)
    content 90° about the z-axis. We rotate the DATA (both fixed-point `proj` and
    trajectory `traj`, each {period: (N, 2)}) rather than the camera azimuth, so
    the xyz axis box and labels stay in their original position and only the
    plotted content turns. Rotation preserves norms, so downstream shared limits
    are unaffected. Non-hidden representations are returned unchanged."""
    if rep_key != "fixed_hidden":
        return proj, traj
    rot = np.deg2rad(90.0)
    R = np.array([[np.cos(rot), -np.sin(rot)],
                  [np.sin(rot), np.cos(rot)]])
    proj = {v: xy @ R.T for v, xy in proj.items()}
    traj = {v: xy @ R.T for v, xy in traj.items()}
    return proj, traj


def _draw_grad_fp_3d_row(fig, results, periods, proj_by_period, z_by_period,
                         traj_by_period, n_stim, lim, zmax, n_rows, row_idx,
                         n_col, show_period_titles=True, row_label=None,
                         draw_periods=None, overlay=None, transpose=False,
                         pc_label=None):
    """Draw one rule's per-period 3D panels into row `row_idx` of an
    (n_rows x n_col) subplot grid on `fig`, using precomputed projections. Shared
    x-y limit `lim` and symmetric z-limit `zmax` are passed in so that multiple
    rows/figures can use IDENTICAL axes (the two-task combined figure stacks
    delaygo over delayanti and shares both). `show_period_titles` prints the
    Context/Stimulus/… titles (typically only the top row); `row_label` writes a
    rotated label (e.g. the task rule) to the left of the row's first panel.

    `transpose` swaps what the grid's two directions mean. Default (False): this
    call fills ROW `row_idx` with the rule's periods, so periods run along x and
    each rule is a row — periods get the panel titles and the rule gets the side
    label. With True it fills COLUMN `row_idx` instead, so RULES run along x and
    periods run down y; the labels swap with the axes they name — the rule titles
    its column (top row only) and each period labels its row (first column only).
    `show_period_titles` is then unused: which labels appear follows from the
    panel's own position in the grid.

    `draw_periods` selects which periods get their OWN panel (defaults to all of
    `periods`). A period omitted from `draw_periods` is still used as the previous-
    period anchor for the next drawn panel's trajectory connector — so e.g. the
    Stimulus panel can keep its fixation→stimulus trajectory even when the
    Context panel itself is not drawn.

    `overlay[period]` (from _grad_fp_period_panels) lists further probes solved
    under that period's input. Unlike the 2D figures, which draw the whole battery
    marker-coded, the 3D panels draw one added probe per period at most
    (_grad_fp_3d_overlay) in the SAME circle marker and size as everything else,
    in gray — see the note above _GRAD_FP_3D_OVERLAY_KIND. The dashed ring line and
    the trajectory connector stay tied to the period's OWN probe, which is the only
    one the trial actually traversed."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)
    if draw_periods is None:
        draw_periods = periods

    # Angle-0 exemplar fixed point (x,y,z) per period, for the anchored connector.
    # Built over ALL `periods` (incl. undrawn ones) so an undrawn previous period
    # can still anchor the next drawn panel's trajectory. Callers that trim panels
    # must include these in their axis limits — see _grad_fp_3d_anchor_xyz.
    _TRAJ_STIM = _GRAD_FP_TRAJ_STIM
    angle0_pt = {}
    for v in periods:
        st = np.asarray(results[v]["stim"], dtype=int)
        idx = np.where(st == _TRAJ_STIM)[0]
        if idx.size:
            i0 = int(idx[0])
            angle0_pt[v] = (proj_by_period[v][i0, 0], proj_by_period[v][i0, 1],
                            float(z_by_period[v][i0]))
    # NB no per-stimulus trajectory color here any more: the path is colored by
    # time (see the trajectory block below), not by which stimulus it belongs to.

    for j, v in enumerate(draw_periods):
        # `row_idx` indexes the row normally, the COLUMN when transposed; `j` (the
        # period) indexes the other direction.
        cell = (j * n_col + row_idx if transpose else row_idx * n_col + j) + 1
        ax = fig.add_subplot(n_rows, n_col, cell, projection="3d")
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
        # One circle style for every fixed point in these panels; only the color
        # differs (gray throughout the fixation panel — see _grad_fp_3d_colors).
        _fp_style = dict(marker="o", s=14, fill=True, z=3)
        # Same 0.85 as everywhere else. It was briefly dropped to 0.55 to keep the
        # exemplar trajectory the subject of these panels, but the trajectory now
        # carries its own weight (black, bold, white-cased), so the fixed points do
        # not need to be faded to make room for it. Lower this to demote them again.
        _FP_ALPHA = 0.85
        # This period's own probe.
        _scatter_grad_fp(ax, xy, _fp_style,
                         _grad_fp_3d_colors(v, e, stim, n_stim), good,
                         z_vals=np.asarray(z, dtype=float), alpha=_FP_ALPHA)
        # Plus this period's added fixed points, in gray (Context: the ring-alike
        # memory-seeded ring; Response: the naive-seeded solve — see
        # _GRAD_FP_3D_OVERLAY_KIND).
        for name in _grad_fp_3d_overlay(v, overlay, results):
            if name not in proj_by_period:      # not projected for this figure
                continue
            pe = results[name]
            pxy = proj_by_period[name]
            _scatter_grad_fp(ax, pxy, _fp_style,
                             _grad_fp_3d_colors(v, pe, np.asarray(pe["stim"]),
                                                n_stim),
                             _fixed_point_mask(pe, pxy.shape[0]),
                             z_vals=np.asarray(z_by_period[name], dtype=float),
                             alpha=_FP_ALPHA)
        # Connect the converged fixed points into their stimulus-ordered ring
        # with a thin dashed black line, tracing the ring-attractor manifold the
        # fixed points lie on. One representative point per stimulus (mean of its
        # converged points), ordered by stimulus index and closed into a loop.
        stim_int = stim.astype(int)
        z_arr = np.asarray(z, dtype=float)
        ring_pts = []
        for s in sorted(set(stim_int.tolist())):
            sel = (stim_int == s) & good
            if np.any(sel):
                ring_pts.append((xy[sel, 0].mean(), xy[sel, 1].mean(),
                                 z_arr[sel].mean()))
        if len(ring_pts) >= 2:
            rx = [p[0] for p in ring_pts] + [ring_pts[0][0]]
            ry = [p[1] for p in ring_pts] + [ring_pts[0][1]]
            rz = [p[2] for p in ring_pts] + [ring_pts[0][2]]
            ax.plot(rx, ry, rz, color="black", linewidth=0.45, linestyle="--",
                    alpha=0.45, zorder=2)
        # Exemplar-stimulus trajectory ANCHORED to the fixed points: it starts at
        # the PREVIOUS period's fixed point, follows the recorded within-period
        # path, and ends at THIS period's fixed point — so its endpoints coincide
        # with the solved fixed points (unlike the raw recorded window, whose ends
        # are recorded boundary states, not the relaxed fixed points). z sits at
        # this period's level along the path; the leading segment shows the jump
        # from the previous period's z. Start = dashed-edge, end = solid-edge.
        tp = traj_by_period.get(v)
        # Previous period = the one before `v` in the FULL period order (which may
        # be an undrawn fixation panel), so the first DRAWN panel still gets its
        # incoming trajectory anchored to the prior period's fixed point.
        _vi = periods.index(v)
        prev_v = periods[_vi - 1] if _vi >= 1 else None
        if tp is not None and tp.shape[0] >= 1 and v in angle0_pt:
            p1 = angle0_pt[v]                                # this period's FP
            z_lvl = p1[2]                                    # current period z
            if prev_v is not None and prev_v in angle0_pt:
                p0 = angle0_pt[prev_v]                       # previous period's FP
            else:
                # FIRST period: there is no previous fixed point to come from, so
                # start at the first RECORDED frame rather than skipping the panel
                # — one panel in four with no path at all reads as an omission.
                p0 = (float(tp[0, 0]), float(tp[0, 1]), z_lvl)
            pts = np.column_stack([
                np.concatenate([[p0[0]], tp[:, 0], [p1[0]]]),
                np.concatenate([[p0[1]], tp[:, 1], [p1[1]]]),
                np.concatenate([[p0[2]], np.full(tp.shape[0], z_lvl), [p1[2]]]),
            ])
            # BLACK and BOLD, over a white casing. Black is the one ink that no
            # stimulus hue can be confused with (the old stimulus-0 red was the
            # same hue as the fixed points the path runs between, so the line read
            # as part of the ring); the casing is what keeps it legible where it
            # crosses the markers, since 3D zorder is advisory at best.
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="white", linewidth=4.6,
                    alpha=0.95, solid_capstyle="round", zorder=4)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="black", linewidth=2.4,
                    solid_capstyle="round", zorder=5)

            # START: a cross. END: a bold arrowhead — so the two ends are told
            # apart by SHAPE, and the direction of travel is on the figure rather
            # than in the caption.
            # All of the arrow geometry is done in units of the DRAWN BOX, not in
            # data units. x/y are PCA units while z is output cos θ, and their
            # half-ranges differ by up to ~35x (raw M: lim 38 vs zmax 1.1), so a
            # length taken from `lim` and applied to a direction with any z
            # component overshoots the z axis by hundreds of percent and the arrow
            # shoots off the panel. Normalizing per axis makes the arrow a fixed
            # fraction of the box whatever the representation's scale.
            axis_half = np.array([lim, lim, zmax], dtype=float)
            pts_n = pts / axis_half
            seg_len = np.linalg.norm(np.diff(pts_n, axis=0), axis=1)
            total = float(seg_len.sum())
            if total > 1e-9:
                cum = np.concatenate([[0.0], np.cumsum(seg_len)])
                # Approach direction from 85% of the ARC LENGTH, not from the last
                # segment: these paths cover most of their distance in the first
                # few steps and then creep, so the final segment alone can point
                # almost anywhere.
                i_b = int(np.clip(np.searchsorted(cum, 0.85 * total),
                                  0, len(pts) - 2))
                j, d_n = i_b, pts_n[-1] - pts_n[i_b]
                while float(np.linalg.norm(d_n)) < 1e-9 and j > 0:
                    j -= 1
                    d_n = pts_n[-1] - pts_n[j]
                nrm = float(np.linalg.norm(d_n))
                if nrm > 1e-9:
                    # 20% of the box, but never longer than the path it terminates,
                    # so a barely-moving path (the delay one) does not sprout an
                    # arrow bigger than itself.
                    d = (d_n / nrm) * min(0.20, 0.9 * total) * axis_half
                    ax.quiver(pts[-1, 0] - d[0], pts[-1, 1] - d[1],
                              pts[-1, 2] - d[2], d[0], d[1], d[2],
                              color="black", arrow_length_ratio=0.62,
                              linewidth=2.4, zorder=6)
            # Start cross, over its own white casing for the same reason as the path.
            for _c, _s, _lw in (("white", 78, 3.8), ("black", 56, 2.0)):
                ax.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], color=_c,
                           marker="x", s=_s, linewidth=_lw, zorder=6)
        # Titles name whichever quantity runs along x: the period normally, the
        # task rule when transposed (and then only above the grid's first row).
        if transpose:
            if j == 0 and row_label is not None:
                ax.set_title(row_label, fontsize=11, pad=-6)
        elif show_period_titles:
            ax.set_title(_period_display(e.get("period_title", v)), fontsize=11, pad=-6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-zmax, zmax)
        # Force the drawn box to a cube so all three axes have equal on-screen
        # length (overrides matplotlib's default 4:4:3 box aspect); the data
        # ranges still differ per axis, this only equalizes the visual box.
        ax.set_box_aspect((1, 1, 1))
        # No tick labels, so pull the axis labels in tight against each axis.
        # `pc_label` names which period's PCA the shared x-y basis came from (e.g.
        # "Delay PC1"): the same basis is used by every panel, so without it the
        # reader cannot tell the delay-basis figure from the stimulus-basis one.
        _pc = f"{pc_label} " if pc_label else ""
        ax.set_xlabel(f"{_pc}PC1", fontsize=7, labelpad=-15)
        ax.set_ylabel(f"{_pc}PC2", fontsize=7, labelpad=-15)
        # z-axis label on EVERY panel, same small size/tight pad as x & y (was a
        # single larger shared label on the rightmost panel only).
        ax.set_zlabel("Output cos θ", fontsize=7, labelpad=-15)
        # Side label, just to the left of the leftmost panel (a small negative x
        # keeps it close to the 3D box rather than far out): it names whichever
        # quantity runs down y — the task rule normally, the PERIOD when transposed,
        # and then only on the grid's first column.
        _side = (_period_display(e.get("period_title", v)) if transpose else row_label)
        if (row_idx == 0 if transpose else j == 0) and _side is not None:
            ax.text2D(-0.10, 0.5, _side, transform=ax.transAxes,
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

    `basis`: an OPTIONAL stored 2-component PCA to project into. When None
    (the default, and the one-task behavior), load THIS pickle's saved
    delay-period basis. When supplied (e.g. by the two-task driver,
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

    pc_label = None
    if basis is None:
        basis_period = "longdelay"
        basis = _load_period_grad_fp_basis(d, rep_key, period=basis_period)
        if basis is None:
            return
        pc_label = _period_display(
            results[basis_period].get("period_title", basis_period))

    periods, overlay, proj, zc, traj, n_stim = _grad_fp_3d_project(d, rep_key, basis)
    # Rotate the hidden-state content 90° (see _rotate_hidden_3d_content); a no-op
    # for the modulation / eff-modulation representations.
    proj, traj = _rotate_hidden_3d_content(rep_key, proj, traj)

    # Limits span exactly what the panels draw: the period probes plus the few
    # overlaid ones the 3D figures keep (the naive probes are not drawn here, so
    # they must not stretch the axes either).
    drawn_3d = [v for p in periods
                for v in [p] + _grad_fp_3d_overlay(p, overlay, results)
                if v in proj]
    lim = max(np.abs(np.vstack([proj[v] for v in drawn_3d])).max() * 1.08, 1e-9)
    zmax = max(np.abs(np.concatenate([zc[v].ravel() for v in drawn_3d])).max() * 1.1,
               1e-6)

    n_col = len(periods)
    # Compact panels: no tick labels, so each panel can be small and packed close.
    fig = plt.figure(figsize=(1.8 * n_col, 1.8))
    _draw_grad_fp_3d_row(fig, results, periods, proj, zc, traj, n_stim, lim, zmax,
                         n_rows=1, row_idx=0, n_col=n_col,
                         show_period_titles=True, row_label=None,
                         overlay=overlay, pc_label=pc_label)
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


def plot_onetask_rnn_fixed_points_3d():
    """
    3D hidden-state fixed points of the single-task VANILLA RNN — the control for
    onetask_grad_fixed_points_3d_hidden, drawn by the same renderer so the two can
    be laid side by side: x-y = delay-period PCA of h*, z = the ideal cos-output
    target (0 outside Response), points colored by stimulus direction.

    The RNN has no modulation matrix, so `fixed_hidden` is its ONLY representation
    — there is no modulation/eff-modulation counterpart to draw.

      onetask_rnn_fixed_points_3d_hidden.png
    Reads onetask_rnn/{aname}/fixed_points_hidden_{aname}.pkl.
    """
    pkl_path = (ONETASK_RNN_DIR / ONETASK_RNN_ANAME
                / f"fixed_points_hidden_{ONETASK_RNN_ANAME}.pkl")
    d = _load_pkl_or_skip(pkl_path, "Run one_task/one_task_rnn_analysis.py first.")
    if d is None:
        return
    _render_grad_fixed_points_3d(
        d, "fixed_hidden", OUT_DIR / "onetask_rnn_fixed_points_3d_hidden.png")


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

    record = _load_fixed_point_pca_record(d, "fixed_M", period)
    if record is None:
        return
    proj = np.asarray(record["fit_projection"], dtype=float)
    if proj.shape != (n, 2):
        raise ValueError("Saved dense-angle PCA projection does not match the stimulus grid")

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
                       edgecolor="none", zorder=3)
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
        ax.set_title(_period_display(e.get("period_title", v)), fontsize=11)
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


# Stable / marginal / unstable class colors for the fixed-point classification
# figures: green = stable (attracting), gray = marginal (ring/neutral), red =
# unstable (expanding). Kept distinct from the stimulus rainbow and period pastels.
_FP_CLASS_COLORS = {"stable": "#2ca02c", "marginal": "#9ca3af", "unstable": "#d62728"}


def _draw_classification_bars(ax, per_period, class_names, add_legend=True):
    """Draw a per-period stacked bar of stable / marginal / unstable fixed-point
    counts onto `ax` (segments colored by _FP_CLASS_COLORS). `per_period` maps a
    period key -> {"period_title", "counts": {class: n}}. Shared by the one-task
    (single-axes) and two-task (one axes per rule) classification figures."""
    periods = list(per_period.keys())
    titles = [_period_display(per_period[v].get("period_title", v)) for v in periods]
    x = np.arange(len(periods))
    bottom = np.zeros(len(periods))
    for cname in class_names:
        counts = np.array([per_period[v]["counts"].get(cname, 0) for v in periods],
                          dtype=float)
        ax.bar(x, counts, bottom=bottom, width=0.7,
               color=_FP_CLASS_COLORS.get(cname, "0.5"), label=cname,
               edgecolor="white", linewidth=0.4)
        bottom += counts
    ax.set_xticks(x)
    ax.set_xticklabels(titles, rotation=20, ha="right", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    if add_legend:
        _legend(ax, fontsize=7, frameon=True, loc="upper right",
                ncol=len(class_names))


def _render_fixed_point_classification(d, out_path):
    """Per-period stacked-bar count of stable / marginal / unstable fixed points,
    from the classification pickle written by
    one_task_analysis.classify_fixed_point_stability. One bar per trial period; the
    bar segments are the class counts over the dense stimulus grid (converged
    points only). A compact 'result' view of the stability verdict, complementary
    to the eigenvalue/spectral-radius spectrum figure."""
    _ensure_out_dir()
    per_period = d.get("per_period", {})
    class_names = d.get("class_names", ["stable", "marginal", "unstable"])
    if not per_period:
        print("  Skipped: no periods in classification pickle.")
        return

    fig, ax = plt.subplots(figsize=(1.1 * len(per_period) + 1.2, 2.6))
    _draw_classification_bars(ax, per_period, class_names)
    ax.set_ylabel("Fixed-point count", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_onetask_fixed_point_classification():
    """One-task fixed-point stability classification: per-period stable / marginal
    / unstable counts. Reads fixed_point_classification_{aname}.pkl written by
    one_task_analysis.classify_fixed_point_stability."""
    pkl_path = ONETASK_DIR / ONETASK_ANAME / f"fixed_point_classification_{ONETASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run one_task_analysis.py first.")
    if d is None:
        return
    _render_fixed_point_classification(
        d, OUT_DIR / "onetask_fixed_point_classification.png")


# Reference rule whose delay-period PCA defines the SHARED x-y basis for BOTH
# the 2D and 3D two-task fixed-point figures. Every rule's figure is projected
# into this rule's delay basis so their rings are directly comparable
# point-for-point (rather than each rule using its own, incomparable, delay PCA).
_TWOTASK_FP_BASIS_RULE = "delayanti"

# Row order (top → bottom) for the combined two-task fixed-point figures (2D and
# 3D): each rule is one row, delaygo above delayanti.
_TWOTASK_FP_ROW_ORDER = ["delaygo", "delayanti"]


def _twotask_shared_fp_bases(paths, label, period="longdelay"):
    """Load ONE shared 2-PC basis per representation from the reference rule's
    (_TWOTASK_FP_BASIS_RULE) PCA sidecar, for the combined grad fixed-point figures.
    `period` selects which trial epoch's fixed points define the basis
    ("longdelay" or "longstimulus"). `paths` is a (rule, path) list; `label` tags
    the log line.

    Returns rep_key -> stored PCA. Missing bases skip the corresponding figures;
    no per-rule fallback is allowed, because that would change the comparison."""
    basis_rule = _TWOTASK_FP_BASIS_RULE
    ref_paths = [p for (r, p) in paths if r == basis_rule]
    shared = {}
    if ref_paths:
        d_ref = _load_pkl_or_skip(ref_paths[0], "Run two_task_analysis.py first.")
        if d_ref is not None:
            for rep_key in ("fixed_M", "fixed_WM", "fixed_hidden"):
                b = _load_period_grad_fp_basis(d_ref, rep_key, period=period)
                if b is not None:
                    shared[rep_key] = b
        print(f"  [{label}] shared x-y basis = '{basis_rule}' {period} "
              f"({len(shared)}/3 representations).")
    else:
        print(f"  [{label}] reference rule '{basis_rule}' pickle not "
              f"found; shared {period} figures will be skipped.")
    return shared


def _load_two_task_grad_fp_rules(paths):
    """Load each rule's grad-fp pickle once, ordered top→bottom per
    _TWOTASK_FP_ROW_ORDER (rules not in the list are appended after, in discovery
    order). Returns an ordered list of (rule, loaded-pickle-dict), skipping rules
    whose pickle is missing/unreadable. Shared by the combined 2D and 3D drivers
    so both order rows alike."""
    row_order = _TWOTASK_FP_ROW_ORDER
    by_rule = dict(paths)
    ordered_rules = ([r for r in row_order if r in by_rule]
                     + [r for r in by_rule if r not in row_order])
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
        periods, ovl, proj, traj, a0, n_stim = _grad_fp_2d_project(d, rep_key, basis)
        per_rule.append((rule, results, periods, ovl, proj, traj, a0, n_stim))
    if not per_rule:
        print(f"  Skipped '{rep_key}': no rule had it.")
        return

    # Shared symmetric x-y limit across ALL rows (so rows are comparable).
    lim = max(np.abs(np.vstack([p for (_, _, _, _, proj, _, _, _) in per_rule
                                for p in proj.values()])).max() * 1.08, 1e-9)
    n_col = max(len(periods) for (_, _, periods, _, _, _, _, _) in per_rule)
    n_rows = len(per_rule)

    fig, axs = plt.subplots(n_rows, n_col, figsize=(2.1 * n_col, 2.1 * n_rows),
                            squeeze=False)
    for row_idx, (rule, results, periods, ovl, proj, traj, a0,
                  n_stim) in enumerate(per_rule):
        _draw_grad_fp_2d_row(
            axs[row_idx], results, periods, proj, traj, a0, n_stim, lim,
            show_period_titles=(row_idx == 0),
            row_label=_TASK_DISPLAY.get(rule, rule), overlay=ovl)

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

# Periods the two-task 3D figure does NOT give a panel to, matched as substrings of
# the period key. Both are still kept in `periods`, so each remains the anchor for
# the next drawn panel's incoming trajectory (fixation → Stimulus, delay →
# Response) — they are dropped as panels, not as data.
#   Context — its fixed points carry no stimulus structure at all.
#   Delay    — the memory ring is what the x-y basis is already fit on, and the 2D
#              figures show it in full; the 3D figure keeps only the two periods
#              where the z axis says something, i.e. where the ring forms
#              (Stimulus) and where it lifts off the z=0 plane (Response).
# Shared with the task-INTERPOLATION 3D figure (_render_interp_alpha_fp_3d), so both
# two-task 3D figures show the same two epochs and read as a pair; there the skipped
# periods are dropped outright, nothing being anchored across periods. The 2D and
# one-task 3D figures still draw every period.
_TWOTASK_FP_3D_SKIP_PANELS = ("fixation", "delay")


def _plot_two_task_grad_fp_combined(stem_prefix, log_label, render_fn,
                                    with_pc_label):
    """Shared driver for the combined two-task grad fixed-point figures (2D and
    3D). Loads each rule's pickle once, then for every basis variant
    (`_TWOTASK_FP_BASIS_VARIANTS`) and every representation (raw M*, W⊙M*,
    hidden) loads the shared delayanti basis and calls `render_fn` to draw
    all rules as stacked rows into ONE figure. Output:
      {stem_prefix}_{seed}_{infix}_{suffix}.png

    stem_prefix   : filename stem before the seed tag ("twotask_grad_fixed_points"
                    for 2D, that + "_3d" for 3D).
    log_label     : short tag for the shared-basis log line ("twotask-2d"/"-3d").
    render_fn     : the combined renderer (_render_two_task_grad_fp_2d_combined or
                    _..._3d_combined); called as
                    render_fn(rule_data, rep_key, out_path, basis[, pc_label=...]).
    with_pc_label : pass the variant's axis-label prefix as pc_label=, so the panels
                    say which period's PCA the shared basis is ("Delay PC1" vs
                    "Stimulus PC1"). Both the 2D and 3D renderers take it."""
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
            basis = shared_bases.get(rep_key)
            if basis is None:
                print(f"  Skipped '{rep_key}' ({period}): no saved reference-rule basis.")
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


def _render_two_task_grad_fp_3d_combined(rule_data, rep_key, out_path, basis,
                                         pc_label=None):
    """Draw ALL two-task rules as stacked rows of a SINGLE 3D figure for one
    representation. `rule_data` is an ordered list of (rule, loaded-pickle-dict)
    (row order = top→bottom); every rule is projected into the shared `basis` and
    the rows share one x-y limit and one z-limit so the panels are directly
    comparable across rows. Period titles print only on the top row; each row is
    labeled on the left by its task rule."""
    # Project every rule first, so shared axis limits can span all rows.
    per_rule = []   # (rule, results, periods, draw_periods, ovl, proj, zc, traj, n_stim)
    for rule, d in rule_data:
        results = d["results"]
        periods = list(results.keys())
        if not periods or any(results[v].get(rep_key) is None for v in periods):
            print(f"  Skipped '{rep_key}' for rule '{rule}': not in pickle.")
            continue
        periods, ovl, proj, zc, traj, n_stim = _grad_fp_3d_project(d, rep_key, basis)
        # Draw only the Stimulus and Response panels; the fixation and delay panels
        # are skipped (_TWOTASK_FP_3D_SKIP_PANELS) but KEPT in `periods`, so each
        # still anchors the next drawn panel's incoming trajectory.
        # `draw_periods` = the panels actually drawn.
        draw_periods = [v for v in periods
                        if not any(skip in v.lower()
                                   for skip in _TWOTASK_FP_3D_SKIP_PANELS)]
        # NB: unlike the one-task figure, the two-task hidden panels are NOT
        # rotated — they use the same viewing angle as modulation / e_modulation.
        per_rule.append((rule, results, periods, draw_periods, ovl, proj, zc, traj,
                         n_stim))
    if not per_rule:
        print(f"  Skipped '{rep_key}': no rule had it.")
        return
    if not any(draw for (_, _, _, draw, _, _, _, _, _) in per_rule):
        print(f"  Skipped '{rep_key}': no period survived "
              f"_TWOTASK_FP_3D_SKIP_PANELS as a panel.")
        return

    # Shared symmetric x-y and z limits across ALL rows (so rows are comparable);
    # computed over the DRAWN periods only (the fixation and delay panels are not
    # drawn). Spans the drawn panels' own probes AND the probes actually overlaid
    # into them (the 3D figures keep only the ring-alike fixation one), so an
    # overlaid point is never silently outside the axes and an undrawn one never
    # stretches them.
    def _panel_probes(draw_periods, ovl, results, proj):
        return [v for p in draw_periods
                for v in [p] + _grad_fp_3d_overlay(p, ovl, results)
                if v in proj]

    # ... plus the single anchor point each panel inherits from the period BEFORE
    # it, which the panel draws as its trajectory's start. That period may itself be
    # undrawn (fixation, delay), and its anchor can sit outside the drawn probes'
    # own extent, so it has to be in the limits even though the rest of its ring is
    # deliberately not.
    def _anchor_xyz(rows):
        pts = [_grad_fp_3d_anchor_xyz(res, periods, draw_periods, proj, zc)
               for (_, res, periods, draw_periods, _, proj, zc, _, _) in rows]
        pts = [p for p in pts if p[0].size]
        if not pts:
            return np.zeros((0, 2)), np.zeros(0)
        return (np.vstack([xy for xy, _ in pts]),
                np.concatenate([z for _, z in pts]))

    anchor_xy, anchor_z = _anchor_xyz(per_rule)
    lim = max(np.abs(np.vstack(
        [proj[v] for (_, res, _, draw_periods, ovl, proj, _, _, _) in per_rule
         for v in _panel_probes(draw_periods, ovl, res, proj)]
        + [anchor_xy])).max() * 1.08, 1e-9)
    zmax = max(np.abs(np.concatenate(
        [zc[v].ravel() for (_, res, _, draw_periods, ovl, _, zc, _, _) in per_rule
         for v in _panel_probes(draw_periods, ovl, res, zc)]
        + [anchor_z])).max() * 1.1, 1e-6)
    # TASK RULE runs along x (one column per rule, named by the column title) and
    # TRIAL PERIOD down y (one row per period, named by the row's side label) — the
    # transpose of the original layout, so the two rules sit side by side for
    # comparison and the trial unfolds downward.
    n_col = len(per_rule)
    n_rows = max(len(draw_periods)
                 for (_, _, _, draw_periods, _, _, _, _, _) in per_rule)

    fig = plt.figure(figsize=(1.8 * n_col, 1.8 * n_rows))
    for col_idx, (rule, results, periods, draw_periods, ovl, proj, zc, traj,
                  n_stim) in enumerate(per_rule):
        _draw_grad_fp_3d_row(
            fig, results, periods, proj, zc, traj, n_stim, lim, zmax,
            n_rows=n_rows, row_idx=col_idx, n_col=n_col,
            row_label=_TASK_DISPLAY.get(rule, rule),
            draw_periods=draw_periods, overlay=ovl, transpose=True,
            pc_label=pc_label)
    # hspace sets the vertical gap between the period rows (3D axes carry large
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
        _render_two_task_grad_fp_3d_combined, with_pc_label=True)


# ═════════════════════════════════════════════════════════════════════════════
# Multi-task DelayDM fixed-point geometry
# ═════════════════════════════════════════════════════════════════════════════
# This is the sole retained two_in_multiple analysis. It reads the two solved
# delay-rule fixed-point pickles and their joint six-PC delay-trajectory
# projection; paper_plot never refits PCA on fixed points.
_DELAYDM_RULES = ("delaydm1", "delaydm2")
_DELAYDM_PAPER_PC_PLANES = {
    "hidden": (1, 2),
    "e_modulation": (1, 2),
}
_MULTITASK_FP_HINT = (
    "Run: python multiple_task/sibling_delay_analysis.py --seed 921 "
    "--feature L21e4 --families delaydm1")
_MULTITASK_RULE_MARKERS = ("s", "^")


def _paper_pc_indices(pair, projection, context):
    """Validate one 1-based paper PC pair and return NumPy column indices."""
    pc_x, pc_y = pair
    n_components = projection.shape[1]
    if not (1 <= pc_x <= n_components and 1 <= pc_y <= n_components
            and pc_x != pc_y):
        raise ValueError(f"invalid paper PC pair {(pc_x, pc_y)} for {context}; "
                         f"available PCs are 1..{n_components}")
    return pc_x, pc_y, pc_x - 1, pc_y - 1


def _adaptive_pc_limits(projection, pc_x, pc_y, padding=0.10):
    """Independent x/y limits from both DelayDM modalities' joint extent."""
    xy = np.asarray(projection[:, [pc_x, pc_y]], dtype=float)
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return (-1.0, 1.0), (-1.0, 1.0)
    lo = xy[finite].min(axis=0)
    hi = xy[finite].max(axis=0)
    center = 0.5 * (lo + hi)
    span = hi - lo
    fallback = max(float(np.max(span)), float(np.max(np.abs(center))), 1e-9) * 0.1
    span = np.where(span > np.finfo(float).eps, span, fallback)
    half = 0.5 * span * (1.0 + 2.0 * padding)
    return ((center[0] - half[0], center[0] + half[0]),
            (center[1] - half[1], center[1] + half[1]))


def _ordered_delaydm_conditions(stim_idx, magnitude, context):
    """Return a stable stimulus×magnitude order and reject ambiguous matches."""
    stim_idx = np.asarray(stim_idx, dtype=int)
    magnitude = np.asarray(magnitude, dtype=float)
    if stim_idx.shape != magnitude.shape:
        raise ValueError(f"{context}: stimulus and magnitude shapes disagree: "
                         f"{stim_idx.shape} vs {magnitude.shape}")
    order = np.lexsort((magnitude, stim_idx))
    keys = np.column_stack((stim_idx[order], np.round(magnitude[order], 8)))
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise ValueError(f"{context}: stimulus×magnitude conditions are not unique")
    return order, stim_idx[order], magnitude[order]


def _matched_delaydm_projection_indices(entry):
    """Indices pairing identical angle×magnitude conditions across modalities."""
    task_idx = np.asarray(entry["task_idx"], dtype=int)
    stim_idx = np.asarray(entry["stim_idx"], dtype=int)
    magnitude = np.asarray(
        entry.get("stimulus_magnitude", np.ones(stim_idx.size)), dtype=float)
    if task_idx.shape != stim_idx.shape or magnitude.shape != stim_idx.shape:
        raise ValueError("delayDM projection labels must have identical shapes")

    paired = []
    for task in (0, 1):
        idx = np.flatnonzero(task_idx == task)
        order, stim, mag = _ordered_delaydm_conditions(
            stim_idx[idx], magnitude[idx], f"delayDM task {task}")
        paired.append((idx[order], stim, mag))
    if (not np.array_equal(paired[0][1], paired[1][1])
            or not np.allclose(paired[0][2], paired[1][2], atol=1e-8, rtol=0)):
        raise ValueError("delayDM modalities do not contain the same "
                         "stimulus×magnitude conditions")
    return paired[0][0], paired[1][0]


def _condensed_euclidean(x):
    """Upper-triangle Euclidean distances without a scipy.spatial dependency."""
    x = np.asarray(x, dtype=float)
    gram = x @ x.T
    sq = np.maximum(
        np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0)
    return np.sqrt(sq)[np.triu_indices(x.shape[0], 1)]


def _delaydm_alignment_metrics(rep_key="fixed_WM", probe="longdelay"):
    """Quantify translation and geometry in the original representation.

    Translation explained is the fraction of the mean squared paired
    cross-task displacement accounted for by one shared displacement vector.
    Geometry correlation compares the two within-task pairwise-distance
    matrices and is therefore translation invariant. These metrics are computed
    before PCA; the trajectory-PC panels below only visualize them.
    """
    aname, rules = DELAYDM_ANAME, _DELAYDM_RULES
    run_dir = Path("multiple_tasks_analysis") / aname
    records = []
    for rule in rules:
        path = run_dir / f"fixed_points_grad_{aname}_{rule}.pkl"
        data = _load_pkl_or_skip(path, _MULTITASK_FP_HINT)
        if data is None:
            return None
        entry = data.get("results", {}).get(probe)
        if entry is None or entry.get(rep_key) is None:
            raise KeyError(f"{path}: missing {probe!r}/{rep_key!r}")
        values = np.asarray(entry[rep_key], dtype=float)
        values = values.reshape(values.shape[0], -1)
        stim = np.asarray(entry["stim"], dtype=int)
        magnitude = np.asarray(
            entry.get("stimulus_magnitude", np.ones(stim.size)), dtype=float)
        order, ordered_stim, ordered_mag = _ordered_delaydm_conditions(
            stim, magnitude, f"{rule}/{probe}")
        fixed = np.asarray(
            entry.get("is_fixed", np.ones(stim.size, bool)), dtype=bool)[order]
        records.append((values[order], ordered_stim, ordered_mag, fixed))

    if (not np.array_equal(records[0][1], records[1][1])
            or not np.allclose(records[0][2], records[1][2],
                               atol=1e-8, rtol=0)):
        raise ValueError("delayDM fixed-point pickles do not contain matching "
                         "stimulus×magnitude conditions")
    keep = records[0][3] & records[1][3]
    if int(keep.sum()) < 3:
        raise ValueError("need at least three converged matched delayDM conditions")
    first, second = records[0][0][keep], records[1][0][keep]

    displacement = second - first
    task_translation = displacement.mean(axis=0)
    total_squared = float(np.mean(np.sum(displacement ** 2, axis=1)))
    translation_squared = float(np.sum(task_translation ** 2))
    translation_explained = (translation_squared / total_squared
                             if total_squared > 0 else np.nan)
    residual = displacement - task_translation

    dist_first = _condensed_euclidean(first)
    dist_second = _condensed_euclidean(second)
    if np.std(dist_first) == 0 or np.std(dist_second) == 0:
        geometry_r = np.nan
    else:
        geometry_r = float(np.corrcoef(dist_first, dist_second)[0, 1])
    return {
        "translation_explained": translation_explained,
        "geometry_r": geometry_r,
        "n_pairs": int(keep.sum()),
        "raw_rms": float(np.sqrt(total_squared)),
        "residual_rms": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
    }


def _plot_multitask_delaydm_fixed_point_geometry_representation(
        plot_name, rep_key, output_suffix):
    """Render one delayDM representation before/after task translation.

    Both panels use that representation's configured joint delay-trajectory PC
    plane. Panel B subtracts only the mean paired modality displacement from
    modality 2. It never refits PCA and performs no rotation or scaling.
    Quantitative annotations are computed in the corresponding original
    high-dimensional representation, before PCA.
    """
    aname, rules = DELAYDM_ANAME, _DELAYDM_RULES
    pc_label = "Joint Delay"
    path = (Path("multiple_tasks_analysis") / aname
            / f"{rules[0]}_delay_pc_projections_{aname}.pkl")
    data = _load_pkl_or_skip(path, _MULTITASK_FP_HINT)
    if data is None:
        return
    entry = data.get("representations", {}).get(plot_name)
    if entry is None:
        raise KeyError(f"{path}: missing representation {plot_name!r}")

    projection = np.asarray(entry["proj"], dtype=float)
    task_idx = np.asarray(entry["task_idx"], dtype=int)
    stim_idx = np.asarray(entry["stim_idx"], dtype=int)
    is_fixed = np.asarray(
        entry.get("is_fixed", np.ones(projection.shape[0], bool)), dtype=bool)
    first_idx, second_idx = _matched_delaydm_projection_indices(entry)
    paired_fixed = is_fixed[first_idx] & is_fixed[second_idx]
    if not np.any(paired_fixed):
        raise ValueError("no converged matched delayDM fixed-point pairs")

    try:
        pc_x, pc_y = _DELAYDM_PAPER_PC_PLANES[plot_name]
    except KeyError as exc:
        raise KeyError(f"Set _DELAYDM_PAPER_PC_PLANES[{plot_name!r}] "
                       "in paper_plot.py") from exc
    pc_x, pc_y, bx, by = _paper_pc_indices(
        (pc_x, pc_y), projection,
        f"delayDM fixed-point geometry/joint/{plot_name}")

    # Because PCA projection is linear, subtracting this six-PC translation is
    # exactly the projection of subtracting its high-dimensional counterpart.
    translation = np.mean(
        projection[second_idx[paired_fixed]]
        - projection[first_idx[paired_fixed]], axis=0)
    aligned = projection.copy()
    aligned[task_idx == 1] -= translation
    metrics = _delaydm_alignment_metrics(rep_key=rep_key, probe="longdelay")
    if metrics is None:
        return

    task_names = list(entry.get("task_names", rules))
    n_stim = int(stim_idx.max()) + 1
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.75), squeeze=False)
    panels = ((projection, "a   Original state space"),
              (aligned, "b   Task offset removed"))
    for panel_index, (ax, (shown, title)) in enumerate(zip(axs[0], panels)):
        # Matched-condition connectors expose the displacement field in panel A
        # and the remaining non-translational mismatch in panel B.
        for i, j, converged in zip(first_idx, second_idx, paired_fixed):
            if not converged:
                continue
            ax.plot([shown[i, bx], shown[j, bx]],
                    [shown[i, by], shown[j, by]],
                    color="0.55", linewidth=0.45,
                    alpha=(0.18 if panel_index == 0 else 0.28), zorder=1)

        for task, rule in enumerate(task_names):
            sel_task = task_idx == task
            for stim in np.unique(stim_idx[sel_task]):
                sel = sel_task & (stim_idx == stim)
                color = stim_color(int(stim), n_stim)
                good, bad = sel & is_fixed, sel & ~is_fixed
                if np.any(good):
                    ax.scatter(shown[good, bx], shown[good, by], color=color,
                               edgecolor="white", linewidth=0.25,
                               marker=_MULTITASK_RULE_MARKERS[task % 2],
                               s=31, alpha=0.88, zorder=3)
                if np.any(bad):
                    ax.scatter(shown[bad, bx], shown[bad, by], color="none",
                               edgecolor=color, linewidth=0.9,
                               marker=_MULTITASK_RULE_MARKERS[task % 2],
                               s=31, zorder=3)

        if panel_index == 0:
            c0 = shown[first_idx[paired_fixed]][:, [bx, by]].mean(axis=0)
            c1 = shown[second_idx[paired_fixed]][:, [bx, by]].mean(axis=0)
            ax.annotate("", xy=c1, xytext=c0,
                        arrowprops=dict(arrowstyle="-|>", color="0.18",
                                        linewidth=1.2, mutation_scale=9),
                        zorder=4)
            midpoint = 0.5 * (c0 + c1)
            ax.annotate(r"$\Delta_{\mathrm{task}}$", xy=midpoint,
                        xytext=(3, 4), textcoords="offset points",
                        fontsize=8, color="0.18")
        else:
            trans_pct = 100.0 * metrics["translation_explained"]
            ax.text(0.03, 0.97,
                    f"Translation explained: {trans_pct:.0f}%\n"
                    f"Geometry correlation: $r$ = {metrics['geometry_r']:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
                    color="0.18",
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                              edgecolor="0.82", linewidth=0.6, alpha=0.92),
                    zorder=5)

        xlim, ylim = _adaptive_pc_limits(shown, bx, by, padding=0.13)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel(f"{pc_label} PC{pc_x}", fontsize=8.5)
        if panel_index == 0:
            ax.set_ylabel(f"{pc_label} PC{pc_y}", fontsize=8.5)
        ax.tick_params(length=2.5)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([], [],
                          marker=_MULTITASK_RULE_MARKERS[t % 2],
                          markerfacecolor="0.45", markeredgecolor="white",
                          color="none", markersize=5.5,
                          label=_TASK_DISPLAY.get(rule, rule))
               for t, rule in enumerate(task_names)]
    _legend(axs[0, 0], handles=handles, frameon=False, fontsize=6.5,
            loc="best", handletextpad=0.3)
    fig.tight_layout(w_pad=1.5)
    _save_fig(
        fig,
        _multitask_out(f"delaydm_fixed_point_geometry_{output_suffix}.png"),
        extra=(f"  ({pc_label} PC{pc_x}-PC{pc_y}; "
               f"representation={rep_key}; "
               f"translation={100 * metrics['translation_explained']:.1f}%; "
               f"geometry r={metrics['geometry_r']:.3f}; "
               f"n={metrics['n_pairs']})"))


def plot_multitask_delaydm_fixed_point_geometry():
    """DelayDM task-translation analysis for effective modulation and hidden.

    Produces two matched-layout figures from the same solved fixed points:
      * ``..._emodulation.png``: original high-dimensional W⊙M* metrics;
      * ``..._hidden.png``: original high-dimensional hidden-state metrics.
    Both use joint delay-trajectory PCA only for visualization.
    """
    for plot_name, rep_key, output_suffix in (
            ("e_modulation", "fixed_WM", "emodulation"),
            ("hidden", "fixed_hidden", "hidden")):
        _plot_multitask_delaydm_fixed_point_geometry_representation(
            plot_name, rep_key, output_suffix)


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
    across alpha, colored NOT by stimulus but by the universal dark→light alpha
    ramp (`_ALPHA_CMAP`). Reads the interp_fixed_points_{aname}.pkl
    written by two_task_analysis.py, where results[period][rep_key] has shape
    (n_alpha, n_stim, feat).

    Panels: Stimulus and Response only — the same two the two-task grad
    fixed-point 3D figure keeps (`_TWOTASK_FP_3D_SKIP_PANELS`), so the two figures
    show the same epochs at the same width and can be read as a pair. Unlike that
    figure nothing here is anchored across periods (each panel is its own alpha
    sweep), so the skipped periods are dropped outright, including from the shared
    PC limits."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)
    _ensure_out_dir()
    results = d.get("results", {})
    # Order panels canonically Context -> Stimulus -> Delay -> Response
    # (the pickle stores them delay/response/stimulus/fixation); any period not in
    # the canonical list is appended after, in pickle order.
    _CANON = ["longfixation", "longstimulus", "longdelay", "longresponse"]
    periods = ([v for v in _CANON if v in results]
               + [v for v in results if v not in _CANON])
    periods = [v for v in periods
               if not any(skip in v.lower() for skip in _TWOTASK_FP_3D_SKIP_PANELS)]
    if not periods or any(results[v].get(rep_key) is None for v in periods):
        print(f"  Skipped '{rep_key}': not in interp pickle "
              f"(re-run two_task_analysis.py).")
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

    # TRIAL PERIOD runs down y — one panel per row, named by a rotated label to the
    # left of the panel, exactly as in the transposed grad fixed-point 3D figure, so
    # the two figures read as a pair.
    #
    # Panel geometry is matched to that figure PANEL FOR PANEL: same 1.8 in grid
    # cell, same margins and row spacing, so each subplot rect comes out the same
    # (~1.57 x 1.70 in) and mplot3d therefore draws the cube at the same size in
    # both. The one departure is `left`, widened from that figure's 0.06 to keep the
    # rotated period labels inside the canvas of a figure a third as wide — which
    # happens to land the axes width on the same number, since there is no `wspace`
    # here to take out of it.
    n_rows = len(periods)
    fig = plt.figure(figsize=(1.8, 1.8 * n_rows))
    for j, v in enumerate(periods):
        ax = fig.add_subplot(n_rows, 1, j + 1, projection="3d")
        xy = proj_by_period[v]                       # (n_alpha, n_stim, 2)
        good = np.asarray(results[v].get("is_fixed",
                          np.ones(xy.shape[:2], bool)), dtype=bool)
        # NOT colored by stimulus: every line takes the SAME universal sequential
        # ramp, dark at alpha=0 → light at alpha=1 (_ALPHA_CMAP), so color encodes
        # only the sweep direction. This also retires the old fixation special case
        # (that panel was drawn black because its lines coincide and a rainbow there
        # would be misleading) — with one ramp everywhere, every panel matches.
        na = xy.shape[0]
        t = _alpha_ramp_norm(alphas)
        # Connector in the ramp's mid tone, so the markers carry the gradient.
        line_col = _alpha_ramp_color(0.5)
        for s in range(n_stim):
            # Line across alpha for this stimulus (PC1=y, PC2=z vs alpha=x).
            ax.plot(alphas, xy[:, s, 0], xy[:, s, 1], "-", color=line_col,
                    linewidth=1.1, alpha=0.5, zorder=2)
            # Per-alpha ramp points: converged filled, over-threshold hollow.
            for ai in range(na):
                col = _alpha_ramp_color(t[ai])
                if good[ai, s]:
                    ax.scatter(alphas[ai], xy[ai, s, 0], xy[ai, s, 1], color=col,
                               marker="o", s=12, edgecolor="none",
                               alpha=0.9, zorder=3)
                else:
                    ax.scatter(alphas[ai], xy[ai, s, 0], xy[ai, s, 1],
                               facecolor="none", edgecolor=col, marker="o", s=12,
                               linewidth=0.7, alpha=0.9, zorder=3)
        # Period name labels its ROW, to the left of the panel (same placement and
        # size as the grad fixed-point 3D figure's period labels).
        ax.text2D(-0.02, 0.5, _period_display(results[v].get("period_title", v)),
                  transform=ax.transAxes, rotation=90, va="center", ha="right",
                  fontsize=11)
        ax.set_xlim(alphas.min(), alphas.max())
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        # Cube box, same as the grad 3D panels. Stretching it along x (alpha) was
        # tried to spend the column's spare width: mplot3d caps the drawn box by the
        # panel HEIGHT, so a 2:1 stretch bought ~6% width and brought the alpha
        # axis's far tick label into the "Delay PC1" label. Not worth it.
        ax.set_box_aspect((1, 1, 1))
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
        # Transparent axes background. The rows are packed with a NEGATIVE hspace, so
        # each panel's rect overlaps its neighbor's; an opaque patch on the lower
        # panel would paint over the upper panel's alpha axis labels, which sit low
        # in its rect (this happened — the top panel lost its "0", "α" and "1").
        ax.patch.set_alpha(0.0)

    # Margins/row spacing copied from the grad fixed-point 3D figure (only `left`
    # differs, see above) so the subplot rects match. hspace is negative because 3D
    # axes carry large internal margins.
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.02, top=0.94, hspace=-0.05)
    _save_fig(fig, out_path)


def _render_interp_alpha_fp_2d(d, rep_key, out_path, basis, pc=0):
    """2D view of `_render_interp_alpha_fp_3d`: the same panels, the same alpha
    sweep, the same shared basis and the same marker conventions, with the z axis
    (Delay PC2) dropped — so this is that figure seen straight down PC2, x = alpha
    and y = Delay PC1. Reading one PC against alpha is what makes a fan
    splitting or merging legible as a curve rather than as depth in a cube.

    Deliberately shares three things with the 3D renderer so the two can be read
    as a pair, and any difference is the projection rather than the plotting:
      * the SAME panel set and order (canonical Context->Response, minus
        `_TWOTASK_FP_3D_SKIP_PANELS`), one period per row with the period named by
        a rotated label to the left;
      * the SAME shared symmetric limit, taken over BOTH PCs rather than only the
        one drawn. That makes the y axis here identical to the 3D figure's y axis,
        and it costs almost nothing: on this run max|PC2| / max|PC1| is 0.96-1.03
        for all three representations, so restricting it to PC1 would widen the
        range by at most ~4%;
      * the alpha ramp (`_ALPHA_CMAP`), the mid-tone connector, and filled =
        converged / hollow = over the relative-step threshold.

    It departs in two places, both because 2D has room the cube did not: the alpha
    axis is SHARED and labelled once at the bottom instead of redrawn per panel,
    and the PC axis keeps its tick labels (the 3D panels hide theirs as
    unreadable), since reading that coordinate is the point of this view.
    """
    _ensure_out_dir()
    results = d.get("results", {})
    _CANON = ["longfixation", "longstimulus", "longdelay", "longresponse"]
    periods = ([v for v in _CANON if v in results]
               + [v for v in results if v not in _CANON])
    periods = [v for v in periods
               if not any(skip in v.lower() for skip in _TWOTASK_FP_3D_SKIP_PANELS)]
    if not periods or any(results[v].get(rep_key) is None for v in periods):
        print(f"  Skipped '{rep_key}': not in interp pickle "
              f"(re-run two_task_analysis.py).")
        return
    alphas = np.asarray(d["alphas"], dtype=float)

    def _proj(arr):
        # arr: (n_alpha, n_stim, feat) -> (n_alpha, n_stim, 2) in the shared basis.
        a = np.asarray(arr, dtype=float)
        na, ns = a.shape[0], a.shape[1]
        return basis.transform(a.reshape(na * ns, -1)).reshape(na, ns, 2)

    proj_by_period = {v: _proj(results[v][rep_key]) for v in periods}
    n_stim = proj_by_period[periods[0]].shape[1]
    lim = max(np.abs(np.concatenate([p.reshape(-1, 2)
                                     for p in proj_by_period.values()])).max()
              * 1.08, 1e-9)

    n_rows = len(periods)
    # Sized to match the 3D figure AS SAVED, which is not the same as matching
    # figsize: both go through _save_fig's bbox_inches="tight", and that CROPS the
    # 3D figure (mplot3d leaves wide margins around each cube, and its rows use a
    # negative hspace so the rects overlap) while it EXPANDS this one by the label
    # margins. Measured at 300 dpi, the 3D figure lands at 1.95 x 3.47 in for two
    # period rows; asking for 1.8 x (1.64 x rows) here lands this one at
    # 1.99 x 3.43 in — same width, same ~1.7 in of height per row — so the pair
    # can sit side by side at one scale.
    # Constrained layout, not subplots_adjust: the shared PC label is vertically
    # CENTERED, which is exactly where the top panel's lowest tick label and the
    # bottom panel's highest one sit, so any hand-placed x either leaves a gap or
    # runs through "-1.5"/"1.5" (both were tried). Constrained layout measures the
    # tick extents and puts each label just outside them, which is what "close to
    # the axes" actually means here.
    fig, axs = plt.subplots(n_rows, 1, figsize=(1.8, 1.64 * n_rows),
                            sharex=True, squeeze=False, layout="constrained")
    t = _alpha_ramp_norm(alphas)
    line_col = _alpha_ramp_color(0.5)
    for j, v in enumerate(periods):
        ax = axs[j][0]
        xy = proj_by_period[v]                       # (n_alpha, n_stim, 2)
        good = np.asarray(results[v].get("is_fixed",
                          np.ones(xy.shape[:2], bool)), dtype=bool)
        for s in range(n_stim):
            ax.plot(alphas, xy[:, s, pc], "-", color=line_col, linewidth=1.1,
                    alpha=0.5, zorder=2)
            for ai in range(xy.shape[0]):
                col = _alpha_ramp_color(t[ai])
                if good[ai, s]:
                    ax.scatter(alphas[ai], xy[ai, s, pc], color=col, marker="o",
                               s=12, edgecolor="none", alpha=0.9, zorder=3)
                else:
                    ax.scatter(alphas[ai], xy[ai, s, pc], facecolor="none",
                               edgecolor=col, marker="o", s=12, linewidth=0.7,
                               alpha=0.9, zorder=3)
        # No period label on the panel. The rows stay in canonical order
        # (Stimulus above Response), so the epoch is carried by the caption rather
        # than by text competing with the curves for a 1.8 in column.
        #
        # x padded by 10% of the sweep on each side so the alpha=0 and alpha=1
        # markers sit inside the axes instead of being clipped in half by the
        # spines; the ticks stay at the true endpoints. Proportional rather than a
        # hardcoded -0.1/1.1 so a different sweep range pads correctly.
        _xpad = 0.1 * max(alphas.max() - alphas.min(), 1e-9)
        ax.set_xlim(alphas.min() - _xpad, alphas.max() + _xpad)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([alphas.min(), alphas.max()])
        # PC label on EVERY panel and NO y tick labels — both as in the 3D figure,
        # whose y/z tick labels are hidden too. The numbers were also what filled
        # the gap between the panels (only 4 px of it was actually blank), so
        # hiding them is what lets that gap read as a gap.
        ax.set_ylabel(f"Delay PC{pc + 1}", fontsize=7)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelleft=False)
        ax.spines[["top", "right"]].set_visible(False)
    # labelpad = -2, not the default 4: swept and measured on the saved PNG, the
    # blank between the x tick labels and the alpha label goes 31 px (pad 4) ->
    # 14 px (0) -> 6 px (-2), then back up to 14 px at -4 and -6 because by then
    # the label has moved INTO the tick labels and the band being measured is a
    # different one. -2 is therefore the tightest clean setting.
    axs[-1][0].set_xlabel(r"$\alpha$", fontsize=9, labelpad=-2)
    # Tight pads pull the labels in against their axes; `hspace` sets the gap
    # between the panels, and is chosen to MATCH the 3D figure's. That gap cannot
    # be copied from the 3D figure's own `hspace=-0.05`: there the value is
    # negative because mplot3d leaves wide margins inside each rect, so overlapping
    # rects still leave a wide gap between the drawn cubes. Measured on the saved
    # PNGs instead — the 3D panels are separated by 69 px (0.230 in) of blank at
    # 300 dpi — and hspace here is set to reproduce that.
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, hspace=0.15, wspace=0.0)
    _save_fig(fig, out_path)


def _plot_two_task_interp_alpha_fp(render_fn, stem, log_label):
    """Shared driver for the task-interpolation alpha figures (3D and its 2D
    view): load the interp pickle once, fetch the delayanti delay-period basis per
    representation — the SAME shared basis the grad fixed-point figures use — and
    render each representation through `render_fn`."""
    d = _load_twotask_glob_or_skip("interp_fixed_points_*.pkl")
    if d is None:
        return
    tag = _twotask_seed_tag()
    paths = _twotask_grad_fp_paths()
    shared_bases = _twotask_shared_fp_bases(paths, log_label,
                                            period="longdelay") if paths else {}
    for rep_key, suffix in (("fixed_M", "modulation"),
                            ("fixed_WM", "emodulation"),
                            ("fixed_hidden", "hidden")):
        basis = shared_bases.get(rep_key)
        if basis is None:
            print(f"  Skipped '{rep_key}': no delayanti delay basis "
                  f"(need fixed_points_grad_*_delayanti.pkl).")
            continue
        render_fn(d, rep_key, OUT_DIR / f"{stem}_{tag}_{suffix}.png", basis)


def plot_two_task_interp_alpha_fixed_points_2d():
    """
    2D view of the task-interpolation fixed-point figure: per trial period,
    x = the pro<->anti interpolation level alpha, y = PC1 of the delayanti
    delay-period basis. Exactly `twotask_interp_alpha_fixed_points_3d` seen
    straight down its PC2 axis — same panels, same basis, same alpha ramp
    (dark at alpha=0 -> light at alpha=1), filled = converged / hollow = not — so
    a fan that splits or merges with alpha reads as a curve instead of as depth in
    a cube. One figure per representation:
      twotask_interp_alpha_fixed_points_2d_{seed}_modulation.png  (+ emodulation, hidden)
    Reads interp_fixed_points_{aname}.pkl (which stores all three representations).
    """
    _plot_two_task_interp_alpha_fp(
        _render_interp_alpha_fp_2d,
        "twotask_interp_alpha_fixed_points_2d", "twotask-interp-alpha-2d")


def plot_two_task_interp_alpha_fixed_points_3d():
    """
    3D figure of the task-interpolation fixed points: per trial period, x = the
    pro<->anti interpolation level alpha, y/z = the two PCs of the delayanti
    delay-period basis (the SAME shared basis as the grad-fixed-point 3D figures).
    Each of the 8 stimuli is one line traced across alpha, so the figure shows how
    each period's fixed points move as the task cue morphs from anti (alpha=0) to
    pro (alpha=1). Color is NOT stimulus: every line uses one sequential ramp, dark
    at alpha=0 → light at alpha=1 (`_ALPHA_CMAP`), so hue carries the sweep
    direction only. One figure per representation:
      twotask_interp_alpha_fixed_points_3d_{seed}_modulation.png  (+ emodulation, hidden)
    Reads interp_fixed_points_{aname}.pkl (which stores all three representations).
    """
    # Shared y-z basis: delayanti delay-period grad fixed points, one per
    # representation — identical to the grad-fixed-point 3D figures, and to the 2D
    # view above, which goes through this same driver.
    _plot_two_task_interp_alpha_fp(
        _render_interp_alpha_fp_3d,
        "twotask_interp_alpha_fixed_points_3d", "twotask-interp-alpha")


def _interp_alphas_or_default(n_default=11):
    """The rule-input alpha sweep the interpolation figures use.

    Read from the interp pickle so an illustration of the color scheme matches the
    run's actual sampling; falls back to `n_default` evenly spaced values in [0, 1]
    (announced, not silent) when that pickle is missing or unreadable, so a pure
    legend figure still renders on a machine that cannot load it."""
    try:
        run_dir = TWOTASKS_DIR / TWOTASK_ANAME
        matches = sorted(run_dir.glob("interp_fixed_points_*.pkl"))
        if matches:
            with open(matches[0], "rb") as f:
                return np.asarray(pickle.load(f)["alphas"], dtype=float)
        print("  interp pickle not found; illustrating alpha with "
              f"{n_default} evenly spaced values.")
    except Exception as exc:                     # unreadable pickle, missing key
        print(f"  interp pickle unreadable ({type(exc).__name__}); illustrating "
              f"alpha with {n_default} evenly spaced values.")
    return np.linspace(0.0, 1.0, n_default)


def plot_two_task_alpha_colorscheme():
    """
    Illustration: the rule-input alpha color convention used by the two-task
    task-interpolation figures (`twotask_interp_alpha_fixed_points_3d` and its 2D
    view, `twotask_interp_alpha_fixed_points_2d`).

    ONE object carries the whole convention — a continuous ramp from the anti rule
    to the pro rule, with the sampled alphas strung along it as the very markers
    those figures draw, overhanging the ramp so each bead's own color reads against
    the page. The ramp is the alpha axis, so it needs no separate axis; the rule
    names sit at the ends in their end's color, so they need no arrows; and the
    markers sit on the color they encode, so they need no second row. That is what
    makes it small: every element does two jobs.

    A legend, not a measurement: colors come from `_alpha_ramp_color` (family 5 in
    SCHEME.md), so this figure and the figures it documents cannot drift apart. The
    only thing read from the run is which alphas were swept.
    """
    _ensure_out_dir()
    alphas = _interp_alphas_or_default()
    t = _alpha_ramp_norm(alphas)
    # alpha = 0 is the ANTI rule, alpha = 1 the PRO rule (the interpolation runs
    # anti → pro); names come from _TASK_DISPLAY so they match the other figures.
    end_labels = (_TASK_DISPLAY.get("delayanti", "delayanti"),
                  _TASK_DISPLAY.get("delaygo", "delaygo"))

    # Geometry in INCHES, then converted: the band has to be thinner than the marker
    # so the markers overhang it (below), and that relationship must survive anyone
    # changing the figure height.
    fig_h, margin_lo, margin_hi = 0.62, 0.02, 0.98
    marker_s = 44                                     # pt^2, as in the α figures
    band_in, marker_in = 0.045, marker_s ** 0.5 / 72.0
    fig, ax = plt.subplots(figsize=(2.4, fig_h))
    half = 0.5 * band_in / (fig_h * (margin_hi - margin_lo))

    # The ramp itself, as a thin continuous band: sampled densely and interpolated,
    # so it reads as the continuum alpha actually is, not as the sweep's steps.
    band_lo, band_hi = 0.5 - half, 0.5 + half
    grad = np.array([[_alpha_ramp_color(x) for x in np.linspace(0, 1, 256)]])
    ax.imshow(grad, extent=[0, 1, band_lo, band_hi], origin="lower", aspect="auto",
              interpolation="bilinear", zorder=2)

    # The sweep's alphas, drawn as the fixed-point markers themselves (same circle,
    # and, like them, no outline). They OVERHANG the band by design — a marker whose
    # fill matches the band exactly would otherwise vanish into it, and it is the
    # overhang that lets each bead's own color read against the page.
    assert marker_in > band_in, "markers must overhang the band to be visible"
    ax.scatter(t, np.full(t.size, 0.5), c=[_alpha_ramp_color(x) for x in t],
               marker="o", s=marker_s, edgecolor="none", zorder=3)

    # Rule names at the ends, each in its own end's color — that is the arrow's job
    # done by hue instead. The light end is darkened for text legibility on white.
    for x, lab, ha, ramp_t in ((0.0, end_labels[0], "left", 0.0),
                               (1.0, end_labels[1], "right", 1.0)):
        col = _alpha_ramp_color(ramp_t)
        col = tuple(0.78 * c for c in col[:3]) if ramp_t else col[:3]
        ax.text(x, 0.5 + 2.1 * half, lab, color=col, ha=ha, va="bottom",
                fontsize=9.5)

    # Endpoint values tucked under the band's ends, and the quantity itself centered
    # below — the only text that is not doing a second job.
    # Sits further from the band than the rule names above it: these three read as a
    # conventional axis annotation, and crowding them against the ramp made the band
    # look like it was underlining them.
    below = 0.5 - 3.6 * half
    ax.text(0.0, below, f"{alphas.min():g}", ha="center", va="top", fontsize=9.5,
            color="0.25")
    ax.text(1.0, below, f"{alphas.max():g}", ha="center", va="top", fontsize=9.5,
            color="0.25")
    ax.text(0.5, below, r"Rule interpolation $\alpha$", ha="center", va="top",
            fontsize=9.5)

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    # Hide the axes background too: with the frame off it draws nothing, but its
    # bbox still counts toward the save's tight bbox and would pad the strip with
    # blank rows above and below the content.
    ax.patch.set_visible(False)
    fig.subplots_adjust(left=0.03, right=0.97, top=margin_hi, bottom=margin_lo)
    _save_fig(fig, OUT_DIR / "twotask_alpha_colorscheme.png")


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


def plot_two_task_fixed_point_classification():
    """Two-task fixed-point stability classification: per-period stable / marginal
    / unstable counts, one panel per task rule (columns). Reads the combined
    fixed_point_classification_{aname}.pkl written by
    two_task_analysis.classify_fixed_point_stability."""
    _ensure_out_dir()
    pkl_path = (TWOTASKS_DIR / TWOTASK_ANAME
                / f"fixed_point_classification_{TWOTASK_ANAME}.pkl")
    d = _load_pkl_or_skip(pkl_path, "Run two_task_analysis.py first.")
    if d is None:
        return
    by_rule = d.get("by_rule", {})
    class_names = d.get("class_names", ["stable", "marginal", "unstable"])
    rules = [r for r in d.get("rules", list(by_rule.keys())) if r in by_rule]
    if not rules:
        print("  Skipped: no rules in classification pickle.")
        return

    fig, axs = plt.subplots(1, len(rules), figsize=(2.6 * len(rules) + 0.6, 2.6),
                            sharey=True, squeeze=False)
    for c, rule in enumerate(rules):
        ax = axs[0][c]
        # Legend on the first panel only (all panels share the same classes).
        _draw_classification_bars(ax, by_rule[rule]["per_period"], class_names,
                                  add_legend=(c == 0))
        ax.set_title(_TASK_DISPLAY.get(rule, rule), fontsize=10)
        if c == 0:
            ax.set_ylabel("Fixed-point count", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, OUT_DIR
              / f"twotask_fixed_point_classification_{_twotask_seed_tag()}.png")


def plot_two_task_d_combine():
    """
    Figure: cross-task / cross-period PCA explained-variance heatmaps for the
    two-task network — one panel each for hidden activity and effective
    modulation (W⊙M).

    Reads the self-contained d_combine pickle written by two_task_analysis.py
    (twotasks/{TWOTASK_ANAME}/d_combine_{TWOTASK_ANAME}.pkl), which stores the
    already-permuted 8x8 FVE matrix, its tick labels, and the color range for
    each of "hidden" and "w_modulation".

    The x/y tick labels ("{task} {period}") are highlighted with their trial
    period's color from the period-bar palette (_ONETASK_PERIOD_COLORS), the same
    colors as the input/output illustration figure's period strip.
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
        # "{task} {period}" labels, period part in the display vocabulary.
        plabels = [_period_display(v) for v in e["labels"]]
        # y-tick labels only on the leftmost panel (all panels share the same
        # row labels); saves horizontal space so panels don't collide.
        ylabels = plabels if col == 0 else False
        # No per-cell value annotations (annot=False) — the color encodes the FVE.
        sns.heatmap(np.asarray(e["fve_k_all"]), ax=ax,
                    xticklabels=plabels, yticklabels=ylabels,
                    annot=False,
                    vmin=vmin, vmax=vmax, square=True,
                    cmap="mako", cbar=False)
        mesh = ax.collections[0]
        ax.set_title(title_map.get(name, name), fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        if col == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        # Highlight each "{task} {period}" tick label with its trial-period color
        # from the period-bar palette, so the epoch a row/column belongs to reads
        # off the same colors as the input/output illustration's period strip.
        # Applied AFTER set_x/yticklabels, which replaces the label artists.
        _color_period_ticklabels(ax, plabels, axis="x")
        if col == 0:
            _color_period_ticklabels(ax, plabels, axis="y")

    # One shared colorbar for all panels. Ticks only at the ends and the middle:
    # the panels are read for which blocks are bright, not for a cell's exact
    # value (annot=False above), so three labels carry the scale and the default
    # denser set only adds clutter. FVE is bounded to [0, 1] and this pickle's
    # vmin/vmax are exactly that, so all three ticks sit inside the range.
    cb = fig.colorbar(mesh, ax=list(axs), shrink=0.8, ticks=[0.0, 0.5, 1.0])
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
    # Context→Stimulus→Memory→Response order, so color BY POSITION into
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
                   markerfacecolor="k", markeredgecolor="k",
                   label=_period_display(label))
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
    "longfixation": "Context",
    "longstimulus": "Stimulus",
    "longdelay": "Delay",
    "longresponse": "Response",
}


def plot_two_task_attractor_cycle():
    """
    Figure: interpolation fixed-point "cycle" plots (PCA 1-2 only) for the
    two-task network. One figure per series (hidden / modulation /
    w_modulation), each a 1x4 row with one panel per long-period variant
    (Context, Stimulus, Delay, Response). A single shared "PCA 1" x-label spans
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

    # Which stimuli get a row, in this order. Current pickles store every
    # stimulus, so this is the only place the selection lives.
    _PREF_STIM = [6, 2]
    stim_keys = [s for s in _PREF_STIM if s in stimuli]
    missing = [s for s in _PREF_STIM if s not in stimuli]
    if missing:
        # Older pickles saved only a subset of the stimuli. Top the selection up
        # from whatever they do have so the panel grid stays full, and name the
        # substitutes so the figure is never silently off-spec.
        fill = [s for s in sorted(stimuli) if s not in stim_keys][:len(missing)]
        print(f"  Note: stimuli {missing} not in the cancel pickle; showing "
              f"{fill} instead (re-run two_task_analysis.py to save all 8).")
        stim_keys += fill

    # Simulation step in ms (see SCHEME.md); the cancel pickle predates a saved
    # dt, so read it from the run's param json to relabel the x-axis in ms.
    dt = _read_twotask_dt()

    # Trial periods: fixation | stimulus | memory(delay) | response, bounded by
    # the saved marker times. Drawn as a top color bar (colors only, no shading),
    # matching the one-task figures' _ONETASK_PERIOD_COLORS grayscale palette.
    fix_end = markers["fixation_end"]
    stim_end = markers["stimulus_end"]
    delay_end = markers["delay_end"]
    fix_c, stim_c, mem_c, resp_c = _ONETASK_PERIOD_COLORS
    period_spans = [
        (0, fix_end, fix_c, "Context"),
        (fix_end, stim_end, stim_c, "Stimulus"),
        (stim_end, delay_end, mem_c, "Memory"),
        (delay_end, None, resp_c, "Response"),
    ]

    n_rows = len(stim_keys)
    # Original grid layout: one ROW per stimulus, two COLUMNS for the task cues
    # (col 0 = task1/pro, col 1 = task2/anti). Illustration-style additions per
    # panel: dashed period lines, ms x-axis, y-ticks [-1,0,1]; period strip on the
    # top row only, x labels on the bottom row only.
    cols = [
        ("fixon_proj1", "x_task1_proj", "fixoff_proj1", "MemoryPro"),
        ("fixon_proj2", "x_task2_proj", "fixoff_proj2", "MemoryAnti"),
    ]

    fig, axs = plt.subplots(n_rows, 2, figsize=(2.72 * 2, 1.8 * n_rows),
                            squeeze=False)
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
            # Colors matched to onetask_show, the one-task cancellation figure,
            # and hence to onetask_example_trial's input channels: Fixon →
            # _IO_FIXATION (dark gray), Task → _IO_TASK (orange), Combine →
            # _IO_COMBINE (deep blue, a hue used by neither input channel).
            # Fixoff takes the light-gray partner of the fixation color.
            ax.plot(fixon, color=_IO_FIXATION, label="Fixon", zorder=2)
            ax.plot(task + bias, color=_IO_TASK, label="Task", zorder=2)
            if e.get("fixate_off"):
                ax.plot(np.asarray(e[fixoff_k]), color=_IO_FIXATION2,
                        label="Fixoff", zorder=2)
            ax.plot(fixon + task + bias, color=_IO_COMBINE, linewidth=2.5,
                    label="Combine", zorder=3)
            # Dashed vertical lines at each period boundary, matching the
            # example-trial illustration.
            for span in period_spans:
                start = span[0]
                if start and start > 0:
                    ax.axvline(start, color="0.5", lw=0.8, linestyle="--",
                               zorder=1.5)
            ax.set_xlim(0, T - 1)
            ax.set_ylim([-1.5, 1.5])
            ax.set_yticks([-1, 0, 1])       # only -1, 0, 1 ticklabels
            # Extra title pad on the top row so it clears the period bar above it.
            ax.set_title(f"Stimulus {si}; {col_name}", fontsize=9,
                         pad=14 if r == 0 else None)
            ax.spines[["top", "right"]].set_visible(False)
            # X ticks at the same frequency as the input/output illustration: take
            # the auto-chosen spacing and double it (fewer, less crowded ticks),
            # then relabel each tick's step index in ms (index * dt); see SCHEME.md.
            auto_ticks = mticker.AutoLocator().tick_values(0, T - 1)
            if len(auto_ticks) >= 2:
                ax.xaxis.set_major_locator(
                    mticker.MultipleLocator((auto_ticks[1] - auto_ticks[0]) * 2))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _pos: f"{x * dt:.0f}"))
            if r == n_rows - 1:
                ax.set_xlabel("Time (ms)", fontsize=9)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if r == 0 and c == 0:
                _legend(ax, frameon=True, fontsize=6, loc="best")

    # Period color bar above each top-row panel (colors only, no shading).
    if T is not None:
        for c in range(2):
            _add_period_strip(axs[0][c], period_spans, xmax=T - 1)

    # Shared y-label centered across the panels.
    fig.supylabel("Proj Cos Mag", fontsize=9)

    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_cancel_{_twotask_seed_tag()}.png"
    _save_fig(fig, out_path, extra=f"  (stimuli {stim_keys})")


def plot_two_task_modulation_magnitude():
    """
    Figure: modulation-computation magnitude across trial time for the two-task
    network, one curve per input MEANING. For each input channel c, the plastic
    matrix M's modulation of that channel is the hidden-unit vector M · W_input[:, c];
    its L2 magnitude over hidden units (mean ± std across trials) shows how strongly
    each input drives the plastic weights over the trial. The stimulus channels are
    combined into a SINGLE "Stimulus" trajectory (per-trial mean), so the figure
    shows four curves — Fixation, Stimulus, Task cue 1, Task cue 2 — colored to
    match the example-trial input figure. Reloaded from
    modulation_magnitude_{aname}.pkl written by two_task_analysis.py.
    """
    _ensure_out_dir()
    pkl_path = TWOTASKS_DIR / TWOTASK_ANAME / f"modulation_magnitude_{TWOTASK_ANAME}.pkl"
    d = _load_pkl_or_skip(pkl_path, "Run two_task_analysis.py first.")
    if d is None:
        return

    labels = list(d["labels"])
    mean = np.asarray(d["mean"])                    # (n_curve, T)
    std = np.asarray(d["std"])                      # (n_curve, T)
    dt = int(d.get("dt", 40))                       # sim step in ms (see SCHEME.md)

    # Colors matched to the example-trial input figure: Fixation gray, combined
    # Stimulus in the shared stimulus green, the two task cues in orange /
    # light-orange (active vs second cue).
    label_color = {
        "Fixation": _IO_FIXATION,
        "Stimulus": _IO_MOD2[0],
        "Task cue 1": _IO_TASK,
        "Task cue 2": _IO_TASK2,
    }
    series = [(lab, mean[k], std[k], label_color.get(lab, c_vals[k % len(c_vals)]))
              for k, lab in enumerate(labels)]

    # Period boundaries (fixation / stimulus / delay ends), in step-index units.
    fe = d.get("fixation_end")
    se = d.get("stimulus_end")
    de = d.get("delay_end")
    fix_c, stim_c, mem_c, resp_c = _ONETASK_PERIOD_COLORS
    period_spans = []
    if fe is not None and se is not None and de is not None:
        period_spans = [(0, fe, fix_c), (fe, se, stim_c),
                        (se, de, mem_c), (de, None, resp_c)]

    _draw_modulation_magnitude(
        series, period_spans, dt,
        OUT_DIR / f"twotask_modulation_magnitude_{_twotask_seed_tag()}.png")


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

    fig, axs = plt.subplots(1, 2, figsize=(2.6 * 2, 1.9))
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
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
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


# ─── Figures grouped by mode ──────────────────────────────────────────────────
# Each mode maps a figure name → its plotting function. Figures only depend on
# the data produced by their corresponding experiment, so a mode can be run in
# isolation without touching the others' inputs.
#
#   one_task         single-task training analyses (multiple_task single-task run)
#   multiple_tasks   the full multi-task network: clustering, lesion, state space
#   acc_plot         accuracy comparisons across training configurations
#   two_in_multiple  delayDM fixed-point geometry probe of the multi-task net
#   pretraining      pretraining → post-training transfer analyses
#   two_task         the two-task network: cross-task / cross-period PCA
FIGURES_BY_MODE = {
    "one_task": {
        "onetask_example_trial": plot_onetask_example_trial,
        "onetask_modulation_magnitude": plot_onetask_modulation_magnitude,
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
        "onetask_rnn_fixed_points_3d": plot_onetask_rnn_fixed_points_3d,
        "onetask_interp_fixed_points": plot_onetask_interp_fixed_points,
        "onetask_fixed_point_stability": plot_onetask_fixed_point_stability,
        "onetask_fixed_point_classification": plot_onetask_fixed_point_classification,
    },
    "multiple_tasks": {
        "input": plot_clustered_input,
        "hidden": plot_clustered_hidden,
        "modulation": plot_clustered_modulation,
        "heatmap_colorbar": plot_multitask_heatmap_colorbar,
        "state_space_combined": plot_state_space_combined,
        "state_space_r_values": plot_state_space_r_values,
        "state_space_dist_angle": plot_state_space_dist_angle,
        "overmembership_norm": plot_overmembership_norm,
        "overmembership_unnorm": plot_overmembership_unnorm,
        "overmembership_weighted": plot_overmembership_weighted,
        "overmembership_var_weighted": plot_overmembership_var_weighted,
        "input_weight_correlation": plot_input_weight_correlation,
        "lesion_heatmap": plot_lesion_heatmap,
        "lesion_cluster_sizes": plot_lesion_cluster_sizes,
        "cluster_corr_vs_lesion": plot_cluster_corr_vs_lesion,
        "om_vs_lesion": plot_om_vs_lesion,
        "cross_seed_summary": plot_cross_seed_summary,
    },
    "acc_plot": {
        "l2_accuracy": plot_l2_vs_accuracy,
        "l2e4_activation_accuracy": plot_l2e4_activation_accuracy,
        "projection_dim_accuracy": plot_projection_dim_accuracy,
        "hidden_dim_accuracy": plot_hidden_dim_accuracy,
        "projection_dim_task_accuracy": plot_projection_dim_task_accuracy,
        "hidden_dim_task_accuracy": plot_hidden_dim_task_accuracy,
    },
    "two_in_multiple": {
        # DelayDM task-translation geometry inside the multi-task network, read
        # off TRUE gradient fixed points (solved by sibling_delay_analysis.py
        # with the same solver one_task and two_task use).
        "delaydm_fixed_point_geometry":
            plot_multitask_delaydm_fixed_point_geometry,
    },
    "pretraining": {
        "backbone_probe": plot_backbone_probe,
        "principal_angles": plot_pretraining_principal_angles,
        "transfer_speed": plot_transfer_speed,
        "learning_trajectory": plot_learning_trajectory,
        "rule_vectors": plot_rule_vectors,
        "aggregate_cve_stimulus": plot_aggregate_cve_stimulus,
        "aggregate_cve_response": plot_aggregate_cve_response,
    },
    "two_task": {
        "twotask_d_combine": plot_two_task_d_combine,
        "twotask_pc_cumvar": plot_two_task_pc_cumvar,
        "twotask_m_pca": plot_two_task_m_pca,
        "twotask_attractor_cycle": plot_two_task_attractor_cycle,
        "twotask_cancel": plot_two_task_cancel,
        "twotask_modulation_magnitude": plot_two_task_modulation_magnitude,
        "twotask_outputsubspace_cancel": plot_two_task_outputsubspace_cancel,
        "twotask_grad_fixed_points": plot_two_task_grad_fixed_points,
        "twotask_grad_fixed_points_3d": plot_two_task_grad_fixed_points_3d,
        "twotask_interp_fixed_points": plot_two_task_interp_fixed_points,
        "twotask_alpha_colorscheme": plot_two_task_alpha_colorscheme,
        "twotask_interp_alpha_fixed_points_2d": plot_two_task_interp_alpha_fixed_points_2d,
        "twotask_interp_alpha_fixed_points_3d": plot_two_task_interp_alpha_fixed_points_3d,
        "twotask_fixed_point_stability": plot_two_task_fixed_point_stability,
        "twotask_fixed_point_classification": plot_two_task_fixed_point_classification,
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
    # Modes are validated by hand, below, instead of with argparse's `choices=`.
    # With nargs="*" and no positional given, argparse on Python <= 3.11 checks the
    # EMPTY DEFAULT against `choices` and exits with "invalid choice: []" — which
    # made every `--only FIGURE` call fail on those interpreters, even though the
    # figure name was fine. (Python 3.12 fixed it, which is why it went unnoticed.)
    valid_modes = ("all", *FIGURES_BY_MODE.keys())
    parser.add_argument(
        "mode",
        nargs="*",
        metavar="MODE",
        help="Which group(s) of figures to generate: "
             f"{', '.join(valid_modes)}. Accepts multiple modes "
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

    bad_modes = [m for m in args.mode if m not in valid_modes]
    if bad_modes:
        parser.error(f"invalid mode(s): {', '.join(bad_modes)}. "
                     f"Choose from: {', '.join(valid_modes)}")

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
        "acc_plot": "(aggregated across seeds)",
        "two_in_multiple": DELAYDM_ANAME,
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

    # Clear old figures on "all" and on mode runs, so stale outputs from
    # renamed/removed figures never linger. A --only run must NOT wipe the
    # directory: it regenerates just its own file (savefig overwrites in
    # place), and wiping there used to silently delete every other figure —
    # including ones whose inputs are expensive to reload.
    _ensure_out_dir()
    if modes_run != "only":
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
