"""
Paper figure generation for the MultiTaskMPN project.

Each public function produces one publication-ready figure and saves it
to the `paper_plot/` directory. Run the script directly to generate all
figures, or import individual functions as needed.

Figures are grouped into modes by the experiment they depend on:
    one_task         single-task training analyses
    multiple_tasks   multi-task clustering and network structure
    lesion           multi-task lesion effects and cluster comparisons
    state_space      multi-task state-space geometry analyses
    acc_plot         accuracy comparisons (L2, activation, projection/hidden dims)
    two_in_multiple  sibling-task delay geometry and rule-cue sweeps in the multi-task network
    pretraining      pretraining → post-training transfer analyses
    two_task         two-task network (cross-task / cross-period PCA)

Usage:
    python figures/paper_plot.py                       # generate every mode
    python figures/paper_plot.py all                   # same as above
    python figures/paper_plot.py one_task              # only the one-task figures
    python figures/paper_plot.py multiple_tasks        # clustering and network structure
    python figures/paper_plot.py lesion                # only the lesion figures
    python figures/paper_plot.py --multitask-l2 1e4 multiple_tasks lesion
                                                        # both from the L2=1e-4 cohort
                                                        # (default cohort: MULTITASK_L2)
    python figures/paper_plot.py state_space            # only the state-space figures
    python figures/paper_plot.py acc_plot              # only the accuracy figures
    python figures/paper_plot.py two_in_multiple       # only the two-in-multiple figures
    python figures/paper_plot.py pretraining           # only the pretraining figures
    python figures/paper_plot.py pretraining --pretraining-bound mod2
                                                        # pretraining with M in [-2, 2]
    python figures/paper_plot.py pretraining --pretraining-groups motifs
                                                        # Relevant / Irrelevant only
    python figures/paper_plot.py pretraining --pretraining-groups all
                                                        # also DelayAnti / DelayPro (default)
    python figures/paper_plot.py two_task              # only the two-task figures
    python figures/paper_plot.py --only input          # generate a single figure
"""
import pickle
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, fcluster

import _bootstrap  # noqa: F401  (puts the repository root on sys.path)
from core.sibling_geometry import (DELAYDM_METRIC_NAME, DMC_SWEEP_METRIC_NAME,
                                   DMC_SWEEP_KEEP_EVERY_ALPHA_METRIC_NAME,
                                   DMCGO_METRIC_NAME, SWEEP_METRIC_NAME,
                                   best_dmc_sweep_pc_pair, best_sweep_pc_pair,
                                   best_task_specific_pc_pair, task_offsets)
from core.state_space_geometry import task_centroid_separation
from core.lesion_cache import load_lesion_pickle, normalize_lesion_cache, resolve_lesion_cache_path
from core.clustering import fixed_k_col_clusters
from pretrain.pretraining_utils import variant_addon

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


def _save_standalone_colorbar(out_path, cmap, vmin=None, vmax=None, ticks=None,
                              ticklabels=None, label=None,
                              orientation="horizontal", figsize=(1.5, 0.45),
                              rect=(0.05, 0.5, 0.9, 0.35), labelsize=8,
                              label_fontsize=8, norm=None):
    """Save JUST a colorbar as its own small figure at `out_path`.

    For panels that share one color scale: drawing the bar once beside them beats
    repeating it inside each, and a standalone bar can be placed and sized in the
    manuscript independently of the panels. `rect` is the bar's axes rectangle
    within the figure, leaving room for the tick labels and `label`. The scale is
    linear over [vmin, vmax] unless a matplotlib `norm` (e.g. LogNorm) is given."""
    figc = plt.figure(figsize=figsize)
    axc = figc.add_axes(rect)                     # [left, bottom, width, height]
    if norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
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


# One size and text style for every standalone vertical colorbar (lesion
# heatmap, OM examples), so the bars tile beside their panels identically: a
# slim 0.13 in bar with 10 pt tick labels and an 8 pt label.
_VERTICAL_COLORBAR_STYLE = dict(orientation="vertical", figsize=(0.7, 2.4),
                                rect=(0.3, 0.08, 0.18, 0.84), labelsize=10, label_fontsize=8)

OM_MIN_EXPECTED = 1.0


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

    Mirrors multiple_task/lesion_plot.py's helper of the same name (keep the
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
# The `multiple_tasks` (clustering) and `lesion` figures read ONE L2 cohort of
# the multi-task runs, everything_seed{seed}_L2{l2}+hidden300+batch128+angle.
# MULTITASK_L2 is the control: change it here, or pass `--multitask-l2 1e4` on
# the command line. Everything below follows it:
#   * the aggregate figures (overmembership, the `*_seeds` figures) summarize
#     every cached seed of that cohort (`_find_experiment_dirs`,
#     `_sibling_run_dirs`);
#   * the single-run figures use the cohort's designated seed from
#     MULTITASK_DESIGNATED_SEEDS when that seed has cached results, otherwise
#     the cached seed with the most complete caches (ties: lowest seed). Each
#     category resolves once, so related figures never mix runs, and a run
#     is never taken from another L2 cohort.
MULTITASK_L2 = "1e3"
MULTITASK_L2_CHOICES = ("1e3", "1e4")
_MULTITASK_RUN_SUFFIX = "+hidden300+batch128+angle"
MULTITASK_ANALYSIS_ROOT = Path("multiple_tasks_analysis")
LESION_PERF_ROOT = Path("multiple_tasks_perf")
LESION_NORM_ROOT = Path("multiple_tasks_norm")
# Designated example seed per cohort and figure category; None means "the
# cached seed with the most complete caches, lowest seed on ties". For
# L2=1e-4, seed 749 is the clustering example and seed 921 the lesion example
# (chosen for a clear hidden-neuron tuning-vs-lesion relation; see the figure
# legends for the current statistics). Its lesion caches come from
#     python multiple_task/lesion_plot.py --seed 921 --feature L21e4
MULTITASK_DESIGNATED_SEEDS = {
    "1e4": {"multiple_tasks": 749, "lesion": 921},
    "1e3": {"multiple_tasks": None, "lesion": None},
}
# Cache roots whose per-run directories decide whether a seed is available.
_MULTITASK_CACHE_ROOTS = {
    "multiple_tasks": (MULTITASK_ANALYSIS_ROOT,),
    "lesion": (LESION_PERF_ROOT, LESION_NORM_ROOT),
}


def multitask_feature(l2=None):
    """Training-script feature tag of an L2 cohort (e.g. 'L21e3')."""
    return f"L2{MULTITASK_L2 if l2 is None else l2}"


def multitask_aname(seed, l2=None):
    """Run identifier of one multi-task run; `seed` may be '*' for a glob."""
    return f"everything_seed{seed}_{multitask_feature(l2)}{_MULTITASK_RUN_SUFFIX}"


def _cached_multitask_seeds(l2, roots):
    """Map each cohort seed with cached results to the roots that hold them.

    A run counts as cached under a root when its own directory there holds at
    least one pickle carrying its name (legacy spellings included); empty
    directories and pickles of other runs do not count.
    """
    import re

    seed_pattern = re.compile(
        r"everything_seed(\d+)_" + re.escape(f"{multitask_feature(l2)}{_MULTITASK_RUN_SUFFIX}"))
    cached = {}
    for root in roots:
        for directory in Path(root).glob(multitask_aname("*", l2)):
            match = seed_pattern.fullmatch(directory.name)
            if match is None or not directory.is_dir():
                continue
            if any(path.is_file() for path in directory.glob(f"*{directory.name}*.pkl")):
                cached.setdefault(int(match.group(1)), set()).add(Path(root))
    return cached


def _resolve_multitask_run(l2, category):
    """Pick the run one figure category reads from the selected L2 cohort.

    The designated seed wins when it has cached results. Otherwise the cached
    seed covering the most cache roots is used (ties: lowest seed) and the
    switch is printed. With no cached run at all the designated name, or the
    cohort name with seed 0, is kept so the per-figure "Skipped" messages fire.
    """
    if l2 not in MULTITASK_L2_CHOICES:
        raise ValueError(f"MULTITASK_L2 must be one of {MULTITASK_L2_CHOICES}, got {l2!r}")
    designated = MULTITASK_DESIGNATED_SEEDS[l2][category]
    cached = _cached_multitask_seeds(l2, _MULTITASK_CACHE_ROOTS[category])
    if designated is not None and designated in cached:
        return multitask_aname(designated, l2)
    if not cached:
        print(f"  {category}: no cached {multitask_feature(l2)} run found under "
              + ", ".join(str(root) for root in _MULTITASK_CACHE_ROOTS[category]) + ".")
        return multitask_aname(0 if designated is None else designated, l2)
    seed = min(cached, key=lambda candidate: (-len(cached[candidate]), candidate))
    if designated is not None:
        print(f"  {category}: designated run {multitask_aname(designated, l2)} has no "
              f"cached results; using seed {seed}.")
    return multitask_aname(seed, l2)


def multitask_cohort_of(aname):
    """Cohort tag of a multi-task run name ('L21e3' for
    everything_seed86_L21e3+hidden300+batch128+angle), or None when the name
    is not a run of this run family."""
    import re

    match = re.fullmatch(r"everything_seed\d+_(L2[^+]*)" + re.escape(_MULTITASK_RUN_SUFFIX),
                         str(aname))
    return match.group(1) if match else None


def _check_same_cohort(l2, **runs):
    """Raise unless every named run belongs to the `l2` cohort.

    The `multiple_tasks` and `lesion` figures must describe one set of
    experiments; this is the guard that keeps a hand-edited or stale run name
    from mixing, say, an L2=1e-3 clustering with L2=1e-4 lesions."""
    expected = multitask_feature(l2)
    mismatched = {name: aname for name, aname in runs.items()
                  if multitask_cohort_of(aname) != expected}
    if mismatched:
        raise RuntimeError(f"multi-task figure runs must all come from the {expected} "
                           f"cohort; got {mismatched}")


def _set_multitask_l2(l2):
    """Select the L2 cohort and re-resolve every multi-task run name and path.

    Called once at import with MULTITASK_L2 and again by `main()` when
    `--multitask-l2` is given, so imported plotting functions and the command
    line see the same run selection. Both categories are resolved from the
    same `l2` and verified to share its cohort before anything is assigned, so
    the `multiple_tasks` and `lesion` figures can never describe different
    experiment sets (their example seeds may differ, their cohort cannot).
    """
    global MULTITASK_L2, ANAME, DATA_DIR, LESION_ANAME, LESION_DIR, LESION_NORM_DIR
    clustering = _resolve_multitask_run(l2, "multiple_tasks")
    lesion = _resolve_multitask_run(l2, "lesion")
    _check_same_cohort(l2, multiple_tasks=clustering, lesion=lesion)
    MULTITASK_L2 = l2
    ANAME = clustering
    LESION_ANAME = lesion
    DATA_DIR = MULTITASK_ANALYSIS_ROOT / ANAME
    LESION_DIR = LESION_PERF_ROOT / LESION_ANAME
    LESION_NORM_DIR = LESION_NORM_ROOT / LESION_ANAME


# Clustering run (`multiple_tasks` figures) and its analysis directory, and the
# lesion run shared by every single-run figure of the `lesion` mode with its
# multiple_tasks_perf / multiple_tasks_norm directories. All six are assigned by
# _set_multitask_l2 from the cohort above.
ANAME = DATA_DIR = LESION_ANAME = LESION_DIR = LESION_NORM_DIR = None
_set_multitask_l2(MULTITASK_L2)
# State-space example figures automatically select one run from the tanh,
# projection-300, hidden-300, L2=1e-4 cohort. The selected run has the highest
# category-balanced separation of task centroids in full effective-modulation
# space; the two-dimensional PCA is used only for display.
# This remains independent of MULTITASK_L2, ANAME and LESION_ANAME, which select
# the clustering and lesion paper figures above. The R-value summary still shows
# all four L2 cohorts.
# DelayDM and DMCGo sibling-task probes (two_in_multiple mode) select their
# trained runs independently of each other and of the clustering figures.
# With None each figure (endpoint or rule-cue sweep, per representation) takes
# its own run automatically: the cached SIBLING_L2 run whose drawn fixed points
# separate best (`_select_sibling_runs`; these figures are illustrations, so the
# most legible run is shown and the whole ranking is printed and saved). A run
# name pins every figure of that family to one run instead.
SIBLING_L2 = "1e4"
DELAYDM_ANAME = None
DMCGO_ANAME = None
# Rule-cue sweep figures (sibling_cue_interp and the `_cue_interp` endpoint
# overlays): draw every sweep trial (False), or one trial per stimulus
# direction (True, the default; see `_one_trial_per_stimulus`), at a
# magnitude matched across the directions and, for DelayDM, weighed against
# how far the trial's endpoint jumps along the sweep (jumps above
# _SIBLING_ATTRACTOR_SWITCH_FRACTION only). The settled DelayDM endpoints
# form a discrete set of attractors, and a trial near a basin boundary hops to a
# neighboring one as the cue moves, which draws as a jump; DMC sweeps are not
# scored, since their large step is the response-cluster swap the figures
# show. This is a selection for legibility, so the caption must say so.
# `--no-sibling-one-trial-per-stimulus` draws every trial instead.
SIBLING_ONE_TRIAL_PER_STIMULUS = True
# DMC rule-cue sweep figures: the top-row PC keeps the response clusters apart
# in the same order at both ends of the sweep (False), or at every alpha
# (True), which gives a cleaner top row but can leave a smaller swap on the
# bottom row (see `core.sibling_geometry.dmc_sweep_axis_scores`). It changes
# the DMC sweep figures' plane and run. `--sibling-dmc-keep-every-alpha` sets it.
SIBLING_DMC_KEEP_AT_EVERY_ALPHA = False
# Produced by multiple_task/sibling_delay_analysis.py, which writes into
# two_in_multiples/{aname}/ the artifacts the sibling figures read. Gradient
# method:
#     fixed_points_grad_{aname}_{rule}.pkl   one per sibling rule
#     {family}_delay_pc_projections_{aname}.pkl
#                                             joint six-PC delay-trajectory
#                                             coordinates of the fixed points
#     {family}_cue_interp_delay_pc_projections_{aname}.pkl
#                                             the rule-cue sweep in that basis
# Long-delay endpoint method:
#     {family}_long_delay_endpoint_pc_projections_{aname}.pkl
#     {family}_cue_interp_long_delay_endpoint_pc_projections_{aname}.pkl
TWO_IN_MULTIPLES_DIR = Path("two_in_multiples")

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

    New analyses judge the leak-normalized residual. For an older modulation
    pickle that already saved ``rel_step_undamped`` (or raw ``rel_step`` plus
    ``leak``), apply the new 1e-2 criterion at render time rather than preserving
    the former, overly permissive raw mask. Truly old pickles fall back to their
    saved mask, then to all-converged for compatibility."""
    mask = entry.get("is_fixed_strict")
    if mask is None and entry.get("rel_step_undamped") is not None:
        tol = float(entry.get("rel_tol_undamped", 1e-2))
        mask = np.asarray(entry["rel_step_undamped"], dtype=float) <= tol
    if (mask is None and entry.get("rel_step") is not None
            and entry.get("leak") is not None):
        tol = float(entry.get("rel_tol_undamped", 1e-2))
        mask = (np.asarray(entry["rel_step"], dtype=float)
                / max(float(entry["leak"]), 1e-12)) <= tol
    if mask is None:
        mask = entry.get("is_fixed")
    if mask is None:
        return np.ones(int(n), dtype=bool)
    return np.asarray(mask, dtype=bool)


def _fixed_point_marginal_tol(entry):
    """Raw Jacobian tolerance, scaled from the normalized 0.05 default."""
    if entry.get("marginal_tol_undamped") is not None:
        return (float(entry["marginal_tol_undamped"])
                * max(float(entry.get("leak", 1.0)), 1e-12))
    if entry.get("leak") is not None:
        # Reinterpret legacy modulation pickles on the corrected scale.
        return 5e-2 * max(float(entry["leak"]), 1e-12)
    return float(entry.get("marginal_tol", 0.05))


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
# Categorization's own deeppink. State-space panels and task-center scoring
# group by these colors, keeping the DMS match/non-match pairing consistent.
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
# CIELAB from every `_CLUSTER_COLORS` entry (the closest is 39.9, delay1 vs the
# sand block) and from the stimulus rainbow (26.6 stim2, 29.9 delay2).
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

# ─── Statistics-overlay style (association scatters) ─────────────────────────
# Every x-y scatter that reports a saved association (state-space distance vs
# angle, cluster tuning vs lesion, OM vs lesion, causal vs activity) and the
# per-run `_seeds` dot plots draw their points and trend / median overlay with
# these two colors, defined once so the figures agree (SCHEME.md family 4):
# points in c_vals' blue (grey in the lesion-mode figures, _LESION_POINT_COLOR
# below), the overlay in a warm red no other family uses.
_SCATTER_POINT_COLOR = c_vals[1]          # "#3182ce"
_TREND_LINE_COLOR = "tomato"
# The lesion-mode scatters (cluster tuning vs lesion, OM vs lesion, causal vs
# activity, state space vs lesion, plasticity share) and their `_seeds` and
# cohort plots draw their points in grey instead, significant and not alike
# (filled vs hollow grey ring), so the red overlay is their only color.
_LESION_POINT_COLOR = "0.45"
# The dots have no outline: a black edge only darkens dense clouds.
_SCATTER_POINT_STYLE = dict(color=_SCATTER_POINT_COLOR, edgecolors="none", linewidths=0,
                            s=22, alpha=0.75, zorder=3)
_TREND_LINE_STYLE = dict(color=_TREND_LINE_COLOR, linewidth=1.2, zorder=4)
# Dense scatters (hundreds of points, e.g. OM vs lesion) fade their dots.
_DENSE_SCATTER_ALPHA = 0.45
# Small panels keep a few ticks per axis: at most this many tick intervals.
# Five, not four, so a 0 to 0.9 axis can take the 0.2 step (0, 0.2, ..., 0.8)
# instead of falling back to 0 and 0.5 only once 0.25-type steps are barred.
_STAT_SCATTER_MAX_TICKS = 5
# Tick labels of these scatters and of the lesion `_seeds` dot plots, one
# point above paper_plot's 7 pt default so they read at the 2.5 in panel size.
_STAT_SCATTER_TICK_FONTSIZE = 8


def _short_tick(value, _position=None):
    """Tick label with at most two decimals and no trailing zeros: 0.4, 0.8, 1."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _invisible_legend_handle():
    """A zero-size, fully transparent legend handle, for text-only legends."""
    return mpl.patches.Patch(facecolor="none", edgecolor="none", linewidth=0)


def _stat_scatter(ax, x, y, legend_text, *, trend_line=None, xlabel="", ylabel="",
                  legend_handles=None, point_alpha=None, label_fontsize=8,
                  point_color=None):
    """Draw one association scatter the way every such figure draws it.

    Points in _SCATTER_POINT_STYLE (optionally faded with `point_alpha`, and
    in `point_color` instead of its blue, e.g. _LESION_POINT_COLOR), the
    saved OLS guide `trend_line` ({"slope", "intercept"}) in _TREND_LINE_STYLE
    spanning the data, and the statistics text as a frameless legend above the
    axes - the only place a 2.5 in panel can hold it without hiding points. With
    `legend_handles` (e.g. a median connector) the text becomes the legend
    title and the handles its entries. Spines, label sizes, tick counts and a
    square axes box (set_box_aspect(1): the data ranges differ between axes,
    so the box, not the data, is kept square, and panels with legends of
    different heights still line up) are set here so the figures cannot drift
    apart; axis limits are left to the caller.
    """
    style = dict(_SCATTER_POINT_STYLE)
    if point_alpha is not None:
        style["alpha"] = point_alpha
    if point_color is not None:
        style["color"] = point_color
    ax.scatter(x, y, **style)
    if trend_line is not None:
        positions = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(positions, trend_line["intercept"] + trend_line["slope"] * positions,
                **_TREND_LINE_STYLE)
    placement = dict(loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0,
                     fontsize=6, frameon=False)
    if legend_handles:
        _legend(ax, handles=legend_handles, title=legend_text, title_fontsize=6, **placement)
    else:
        _legend(ax, [_invisible_legend_handle()], [legend_text], handlelength=0,
                handletextpad=0, **placement)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=_STAT_SCATTER_TICK_FONTSIZE)
    # Round tick steps only (1, 2, 5 x a power of ten: 0.2, 0.5, 2 ...), never
    # the 0.25-type steps that put 0.75 on a correlation axis, and labels with
    # trailing zeros dropped (0.4, 0.8, 1).
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.MaxNLocator(_STAT_SCATTER_MAX_TICKS, steps=[1, 2, 5, 10]))
        axis.set_major_formatter(mticker.FuncFormatter(_short_tick))


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

def _recut_labels(linkage, k, original_labels, unresponsive_mask=None):
    """
    Re-cut a dendrogram at a different k.

    The linkage matrix was built on the "active" subset (excluding any
    unresponsive neurons marked with label = original_k + 1). This
    function cuts the linkage at the new k, then maps back to the full
    label array, giving the unresponsive neurons label k + 1.

    `unresponsive_mask` is the per-neuron flag the clustering saves
    (`result["col_unresponsive_mask"]`). Results saved before it existed are
    recognized by their length: the full label array is longer than the
    number of dendrogram leaves exactly when unresponsive neurons were held
    out, and those neurons carry the largest label.
    """
    n_active = linkage.shape[0] + 1     # leaves of the dendrogram
    original_labels = np.asarray(original_labels)

    if unresponsive_mask is not None:
        unres_mask = np.asarray(unresponsive_mask, dtype=bool)
        if unres_mask.shape != original_labels.shape:
            raise ValueError("unresponsive_mask must align with original_labels")
    elif len(original_labels) > n_active:
        unres_mask = original_labels == original_labels.max()
    else:
        unres_mask = np.zeros(len(original_labels), dtype=bool)
    if int((~unres_mask).sum()) != n_active:
        raise ValueError(f"{int((~unres_mask).sum())} active labels but the linkage has "
                         f"{n_active} leaves")

    new_active_labels = fcluster(linkage, t=k, criterion="maxclust")

    full_labels = np.zeros(len(original_labels), dtype=int)
    full_labels[~unres_mask] = new_active_labels
    if unres_mask.any():
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
    unique_labels = np.unique(labels)

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
# The input / hidden / normalized-modulation task-variance heatmaps all show the
# SAME quantity on the SAME scale, so the scale is defined once here and the
# colorbar is a figure of its own (`plot_multitask_heatmap_colorbar`) rather than
# a strip repeated inside each panel — the same split the one-task modulation
# snapshots use (_plot_onetask_snapshot_single + _onetask_hcbar). Change the scale
# here and the heatmaps and their colorbar move together. The one exception is
# the |W|*Var(M) heatmap (`plot_clustered_modulation_abs_weighted`), whose
# unnormalized values need a data-driven log scale and so export their own bar;
# it shares only the colormap.
_MULTITASK_HEATMAP_CMAP = "viridis"
_MULTITASK_HEATMAP_VLIM = (0.0, 1.0)
_MULTITASK_HEATMAP_CLABEL = "Normalized variance"


# ─── Cluster (neuron-class) strip colors ──────────────────────────────────────
# The multi-task heatmaps mark each contiguous row/column cluster with a barcode
# block. Two alternating tones are enough to show every boundary and keep
# categorical color from competing with the heatmap's quantitative scale.
#
# The tones are a light/dark BROWN pair (ColorBrewer BrBG ends), not gray: the
# Response tick labels sit on the light gray `_PHASE_COLORS["go1"]`, and the
# earlier light-gray block (#bdbdbd) was only 12 CIELAB from that label as
# drawn, so a gray block read as "Response". Brown is a hue nothing else in these
# figures uses (phase labels are purple/green/gray, the heatmap is viridis):
#   * sand is >= 39.9 CIELAB from every phase color (delay1), brown >= 70.0 (go1);
#   * >= 49.7 from every viridis color, so a block never reads as a heatmap cell;
#   * the pair is 44.1 apart, as far as the two grays were (43.9).
_CLUSTER_COLORS = [
    "#dfc27d",   # sand
    "#8c510a",   # dark brown
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
        rl_full = _recut_labels(result["row_linkage"], row_k_override, result["row_tol_labels"],
                                result.get("row_unresponsive_mask"))
        row_order = _compute_order_from_labels(result["row_linkage"], rl_full)
    else:
        rl_full = np.asarray(result["row_tol_labels"])
        row_order = result["row_order"]

    # Determine col labels and order
    if col_k_override is not None:
        cl_full = _recut_labels(result["col_linkage"], col_k_override, result["col_tol_labels"],
                                result.get("col_unresponsive_mask"))
        col_order = _compute_order_from_labels(result["col_linkage"], cl_full)
    else:
        cl_full = np.asarray(result["col_tol_labels"])
        col_order = result["col_order"]

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


# Color floor of the log-scaled |W|*Var(M) heatmap: the same 1e-4 the analysis
# script uses for its own unnormalized modulation heatmaps (LogNorm(vmin=1e-4)).
_ABS_WEIGHTED_MOD_KEY = "modulation_all_abs_weighted_unnormalized"
_ABS_WEIGHTED_HEATMAP_FLOOR = 1e-4


def _load_modulation_variant(mod_key):
    """The saved clustering entry of one modulation variant, or None with a
    "Skipped" message when the cache or the variant is missing."""
    mod_info = _load_cluster_info_mod()
    if mod_info is None:
        print(f"  Skipped: cluster_info_mod_{ANAME}.pkl not found. "
              "Run multiple_task_analysis.py first.")
        return None
    mod_data = mod_info.get(mod_key)
    if mod_data is None:
        print(f"  Skipped: {mod_key} not in cluster_info_mod_{ANAME}.pkl "
              "(older caches lack it); rerun multiple_task_analysis.py.")
    return mod_data


def _draw_clustered_modulation(mod_data, G_index, norm, out_filename):
    """Shared layout of the clustered modulation task-variance heatmaps.

    `norm` maps the saved feature values to colors: the fixed linear
    `_MULTITASK_HEATMAP_VLIM` scale for the normalized variant, a data-driven
    log scale for the unnormalized ones. Values below `norm.vmin` are floored
    so that a log scale never masks silent (zero) synapses. No colorbar is drawn
    in the panel; callers export the matching bar as its own figure.
    """
    cell_vars = mod_data["cell_vars_rules_sorted_norm"]
    tb_break_name = mod_data["tb_break_name"]
    result = mod_data["result_all_lst"][G_index]

    row_order = result["row_order"]
    col_order = result["col_order"]
    ordered = np.maximum(cell_vars[np.ix_(row_order, col_order)], norm.vmin)

    rl = np.asarray(result["row_tol_labels"])[row_order]
    cl = np.asarray(result["col_tol_labels"])[col_order]
    rbreaks = _breaks(rl)
    cbreaks = _breaks(cl)

    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    # No colorbar here — see plot_multitask_heatmap_colorbar and the
    # abs_weighted colorbar exported by plot_clustered_modulation_abs_weighted.
    sns.heatmap(ordered, ax=ax, cmap=_MULTITASK_HEATMAP_CMAP, norm=norm, cbar=False)

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
    _save_fig(fig, _multitask_out(out_filename))


def plot_clustered_modulation(G_index=1):
    """
    Figure: Clustered normalized task-variance matrix for MODULATION synapses.

    Uses the G=300 KMeans pre-grouping result (index 1 in result_all_lst).
    The figure is 2x wider than input/hidden figures to accommodate the
    90,000 synapse columns. Colors follow the shared normalized scale of
    plot_multitask_heatmap_colorbar.
    """
    _ensure_out_dir()
    mod_data = _load_modulation_variant("modulation_all_normalized")
    if mod_data is None:
        return
    _draw_clustered_modulation(
        mod_data, G_index, mpl.colors.Normalize(*_MULTITASK_HEATMAP_VLIM),
        "clustered_modulation_normalized.png")


def plot_clustered_modulation_abs_weighted(G_index=1):
    """
    Figure: Clustered |W|*Var(M) task-variance matrix for MODULATION synapses.

    The abs_weighted counterpart of plot_clustered_modulation: same layout, same
    G=300 pre-grouping index, but rows/columns follow the abs_weighted
    clustering and the cells show the unnormalized feature |W|*Var(M). Because
    these values span orders of magnitude and are not on the shared [0, 1]
    scale, colors are log-scaled from the 1e-4 floor the analysis script uses
    to the run's maximum, and the matching colorbar is exported separately as
    clustered_modulation_abs_weighted_colorbar.png. Runs analyzed before the
    abs_weighted variant existed have no entry and are skipped.
    """
    _ensure_out_dir()
    mod_data = _load_modulation_variant(_ABS_WEIGHTED_MOD_KEY)
    if mod_data is None:
        return
    cell_vars = np.asarray(mod_data["cell_vars_rules_sorted_norm"], dtype=float)
    vmax = float(np.nanmax(cell_vars))
    if not np.isfinite(vmax) or vmax <= _ABS_WEIGHTED_HEATMAP_FLOOR:
        raise ValueError(f"abs_weighted modulation features must exceed the "
                         f"{_ABS_WEIGHTED_HEATMAP_FLOOR:g} color floor (max {vmax:g})")
    norm = mpl.colors.LogNorm(vmin=_ABS_WEIGHTED_HEATMAP_FLOOR, vmax=vmax)
    _draw_clustered_modulation(mod_data, G_index, norm,
                               "clustered_modulation_abs_weighted.png")
    _save_standalone_colorbar(
        _multitask_out("clustered_modulation_abs_weighted_colorbar.png"),
        cmap=_MULTITASK_HEATMAP_CMAP, norm=norm,
        label=r"$|W|\,\mathrm{Var}(M)$",
        orientation="horizontal", figsize=(1.9, 0.62),
        rect=(0.05, 0.62, 0.90, 0.24), labelsize=8, label_fontsize=8)


def plot_multitask_heatmap_colorbar():
    """
    The colorbar for the multi-task task-variance heatmaps (input, hidden and
    normalized modulation), as its own figure.

    Those three panels show the same quantity on the same scale
    (`_MULTITASK_HEATMAP_CMAP` over `_MULTITASK_HEATMAP_VLIM`), so none of them
    draws a colorbar: one bar published once serves all three, is not repeated
    three times at three different panel widths, and can be placed and sized in the
    manuscript independently of the panels. Reads no data — the scale is fixed.
    The |W|*Var(M) heatmap is log-scaled and exports its own colorbar instead
    (`plot_clustered_modulation_abs_weighted`).
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
ACTIVATION_DIAGNOSIS_PATH = (
    Path("multiple_task") / "activation_function_diagnosis_results.json"
)
MODULATION_BOUND_L2_STRENGTHS = (1e-5, 1e-4, 1e-3, 1e-2)
_MODULATION_BOUND_SPECS = (
    ("mb1", (-1.0, 1.0), r"$M \in [-1, 1]$ (mb1)", "#0072B2"),
    ("mb2", (-2.0, 2.0), r"$M \in [-2, 2]$ (mb2)", "#D55E00"),
)


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
    ax.scatter(l2_vals, acc_vals, **_SCATTER_POINT_STYLE)

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


def _modulation_bound_accuracy_groups(
        result_dict, config_dir=Path("multiple_tasks")):
    """Collect matched regular-run overall accuracies by L2 and M bound.

    A regular run has the unadorned ``L21eX`` feature (historical mb1) or only
    the ``mb2`` suffix, one 300-unit hidden layer, a 300-dimensional learned
    input projection, Tanh activation, and the standard batch-128 setup. Older
    mb1 configs omit ``m_bounds`` because ``(-1, 1)`` is the model default.
    """
    import json as _json
    import re as _re

    groups = {
        strength: {
            bound_name: []
            for bound_name, _, _, _ in _MODULATION_BOUND_SPECS
        }
        for strength in MODULATION_BOUND_L2_STRENGTHS
    }
    bounds_to_name = {
        bounds: bound_name
        for bound_name, bounds, _, _ in _MODULATION_BOUND_SPECS
    }

    for model_name, result in result_dict.items():
        feature = _performance_feature_tag(model_name, result) or ""
        match = _re.fullmatch(r"L21e([2345])(mb2)?", feature)
        if match is None:
            continue
        config_path = config_dir / f"param_{model_name}_param.json"
        try:
            with config_path.open() as handle:
                config = _json.load(handle)
            task_params = config["task_params"]
            train_params = config["train_params"]
            net_params = config["net_params"]
            hidden_dims = net_params["n_neurons"][1:-1]
            if (task_params.get("ruleset") != "everything"
                    or len(hidden_dims) != 1 or hidden_dims[0] != 300
                    or net_params.get("linear_embed", 300) != 300
                    or net_params.get("activation") != "tanh"
                    or net_params.get("net_type") != "dmpn"
                    or not net_params.get("input_layer_add", False)
                    or train_params.get("weight_reg") != "L2"
                    or train_params.get("batch_size") != 128
                    or train_params.get("n_batches") != 128):
                continue

            l2_value = float(train_params["reg_lambda"])
            matched_l2 = next(
                (strength for strength in MODULATION_BOUND_L2_STRENGTHS
                 if np.isclose(l2_value, strength, rtol=1e-9, atol=0)),
                None,
            )
            if matched_l2 is None:
                continue
            raw_bounds = net_params.get("ml_params", {}).get(
                "m_bounds", (-1.0, 1.0))
            bounds = tuple(float(value) for value in raw_bounds)
            bound_name = bounds_to_name.get(bounds)
            expected_bound_name = "mb2" if match.group(2) else "mb1"
            if bound_name is None or bound_name != expected_bound_name:
                raise ValueError(
                    f"feature {feature!r} disagrees with m_bounds={bounds}")

            accuracy = float(result["acc"])
            if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError(f"invalid overall accuracy: {accuracy!r}")
            groups[matched_l2][bound_name].append(accuracy * 100.0)
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"  Note: skipped {model_name}: invalid or missing matched "
                  f"mb1/mb2 metadata/result ({error}).")
    return groups


def plot_modulation_bound_accuracy():
    """Four vertical L2 panels comparing matched mb1/mb2 overall accuracy."""
    import json as _json

    _ensure_out_dir()
    if not PERF_RESULT_PATH.exists():
        print(f"  Skipped: {PERF_RESULT_PATH} not found. Run "
              "multiple_task_performance.py first.")
        return
    with PERF_RESULT_PATH.open() as handle:
        result_dict = _json.load(handle)
    groups = _modulation_bound_accuracy_groups(result_dict)
    if not any(
            values
            for l2_groups in groups.values()
            for values in l2_groups.values()):
        print("  Skipped: no metadata-matched regular mb1/mb2 accuracies.")
        return

    x_positions = np.arange(len(_MODULATION_BOUND_SPECS), dtype=float)
    fig, axes = plt.subplots(
        len(MODULATION_BOUND_L2_STRENGTHS), 1,
        figsize=(3.5, 8.2), sharex=True)

    missing_features = []
    count_summary = []
    for panel_index, (ax, strength) in enumerate(
            zip(axes, MODULATION_BOUND_L2_STRENGTHS)):
        exponent = int(round(-np.log10(strength)))
        ax.set_title(
            f"{chr(ord('a') + panel_index)}   L2 = $10^{{-{exponent}}}$",
            loc="left", fontsize=9.5, pad=3)
        missing_labels = []
        panel_values = []
        means = []
        for position, (bound_name, _, _, _) in zip(
                x_positions, _MODULATION_BOUND_SPECS):
            values = np.asarray(groups[strength][bound_name], dtype=float)
            count_summary.append(
                f"L21e{exponent}{'mb2' if bound_name == 'mb2' else ''} "
                f"n={values.size}")
            if values.size == 0:
                means.append(np.nan)
                missing_labels.append(bound_name)
                missing_features.append(
                    f"L21e{exponent}{'mb2' if bound_name == 'mb2' else ''}")
                continue
            means.append(float(values.mean()))
            panel_values.extend(values.tolist())
            jitter = (np.linspace(-0.06, 0.06, values.size)
                      if values.size > 1 else np.zeros(1))
            ax.scatter(
                position + jitter, values, **_SCATTER_POINT_STYLE)
        means = np.asarray(means, dtype=float)
        finite_means = np.isfinite(means)
        ax.plot(
            x_positions[finite_means], means[finite_means], color="k",
            linewidth=1.2, marker="D", markerfacecolor="white",
            markeredgecolor="k", markeredgewidth=0.8,
            markersize=4, zorder=4)
        if missing_labels:
            ax.text(
                0.98, 0.08,
                "No " + "/".join(missing_labels) + " performance data",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color="0.42")
        if panel_values:
            lo, hi = min(panel_values), max(panel_values)
            pad = max((hi - lo) * 0.08, 2.0)
            ax.set_ylim(
                np.floor(max(0.0, lo - pad) / 5.0) * 5.0,
                np.ceil(min(100.0, hi + pad) / 5.0) * 5.0)
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        else:
            ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5,
                      color="0.82", zorder=0)
        ax.tick_params(axis="both", length=2.5)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels([
        "$[-1, 1]$\n(mb1)", "$[-2, 2]$\n(mb2)"])
    axes[-1].set_xlabel("Modulation bound")
    fig.supylabel("Overall test accuracy (%)", x=0.015, fontsize=9)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.985,
                        bottom=0.085, hspace=0.32)

    if missing_features:
        missing_features = list(dict.fromkeys(missing_features))
        mb2_missing = [feature for feature in missing_features
                       if feature.endswith("mb2")]
        print("  Note: missing matched performance data for "
              + ", ".join(missing_features) + ".")
        if mb2_missing:
            feature_args = " ".join(
                f"--feature {feature}" for feature in mb2_missing)
            print("  Evaluate available mb2 checkpoints with: python "
                  "multiple_task/multiple_task_performance.py --merge "
                  f"{feature_args}")

    _save_fig(
        fig, _multitask_out("modulation_bound_accuracy.png"),
        extra=" (" + ", ".join(count_summary) + ")")


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
        # Shared marker style; the color stays per activation function.
        ax.scatter(x + jitter, values, **{**_SCATTER_POINT_STYLE, "color": color})
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


# ─── Figures: L2=1e-4 activation-function diagnosis ─────────────────────────

# Same order and colors as plot_l2e4_activation_accuracy, so the diagnosis
# panels read as companions of the accuracy panel. ReLU and Linear are the
# controls: ReLU is positive-only and mostly zero-derivative yet trains best,
# so the panels below isolate the embedding offset and its effect on M rather
# than sign or saturation.
_ACTIVATION_DIAGNOSIS_SPECS = (
    ("linear", "Linear", c_vals[2]),
    ("relu", "ReLU", c_vals[6]),
    ("softplus", "Softplus", c_vals[3]),
    ("sigmoid", "Sigmoid", c_vals[9]),
    ("tanh", "Tanh", c_vals[1]),
)


def _load_activation_diagnosis_records():
    """Load per-checkpoint records produced by activation_function_diagnosis."""
    import json as _json

    if not ACTIVATION_DIAGNOSIS_PATH.exists():
        print(
            f"  Skipped: {ACTIVATION_DIAGNOSIS_PATH} not found. Run: python "
            "multiple_task/activation_function_diagnosis.py"
        )
        return None
    with ACTIVATION_DIAGNOSIS_PATH.open() as stream:
        data = _json.load(stream)
    records = data.get("records", [])
    if not records:
        print(f"  Skipped: no model records in {ACTIVATION_DIAGNOSIS_PATH}.")
        return None
    return records


def _diagnosis_record_value(record, path):
    """Read a dot-separated scalar metric from one diagnosis record."""
    value = record
    for key in path.split("."):
        value = value[key]
    return float(value)


def _plot_activation_diagnosis_metric(metric_path, ylabel, filename,
                                      percentage=True):
    """Draw a seed-level activation comparison for one diagnosis metric.

    ``percentage=True`` scales a fractional metric to percent and keeps the
    axis inside [0, 100]; otherwise the raw value is plotted from zero.
    """
    _ensure_out_dir()
    records = _load_activation_diagnosis_records()
    if records is None:
        return

    scale = 100.0 if percentage else 1.0
    # Same per-column width as plot_l2e4_activation_accuracy.
    fig, ax = plt.subplots(
        1, 1, figsize=(0.575 * len(_ACTIVATION_DIAGNOSIS_SPECS) + 0.3, 2.35))
    plotted = []
    for x, (activation, label, color) in enumerate(
            _ACTIVATION_DIAGNOSIS_SPECS):
        values = np.asarray([
            scale * _diagnosis_record_value(record, metric_path)
            for record in records
            if record.get("activation") == activation
        ], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        plotted.append((x, label, values))
        jitter = (np.linspace(-0.10, 0.10, values.size)
                  if values.size > 1 else np.zeros(1))
        ax.scatter(
            x + jitter, values, color=color, edgecolors="k", linewidths=0.5,
            s=38, alpha=0.8, zorder=3,
        )
        ax.errorbar(
            x, values.mean(), yerr=values.std(), fmt="D", color="k",
            markerfacecolor="white", markeredgewidth=0.8, markersize=4,
            capsize=3, linewidth=1.0, zorder=4,
        )

    if not plotted:
        plt.close(fig)
        print(f"  Skipped: metric {metric_path!r} has no finite values.")
        return
    ax.set_xticks([entry[0] for entry in plotted])
    ax.set_xticklabels([entry[1] for entry in plotted], rotation=45, ha="right")
    ax.set_xlabel("Activation function")
    ax.set_ylabel(ylabel)
    all_values = np.concatenate([entry[2] for entry in plotted])
    if percentage:
        value_span = max(float(np.ptp(all_values)), 1.0)
        padding = max(2.5, 0.03 * value_span)
        # Keep zero inside the panel with a small negative margin, so the
        # near-zero Tanh points and their error bars are not clipped against
        # the bottom spine.
        lower = min(-2.5, float(all_values.min()) - padding)
        upper = min(100.0, float(all_values.max()) + padding)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    else:
        padding = max(0.05 * float(all_values.max()), 0.5)
        ax.set_ylim(0.0, float(all_values.max()) + padding)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)
    fig.tight_layout()
    counts = ", ".join(
        f"{label} n={values.size}" for _, label, values in plotted)
    _save_fig(fig, _multitask_out(filename), extra=f" ({counts})")


def plot_l2e4_activation_embedding_offset():
    """Figure B: constant background (offset) energy in the plastic layer's
    presynaptic embedding. Softplus and sigmoid output ln 2 and 0.5 at zero
    input, so their embedding is mostly a constant vector; ReLU, linear and
    tanh output zero at zero input and sit near 0 %."""
    _plot_activation_diagnosis_metric(
        "embedding.offset_energy_fraction",
        "Input embedding\noffset energy (%)",
        "l2e4_activation_embedding_offset.png",
    )


def plot_l2e4_activation_modulation_dimension():
    """Figure C: trial-specific capacity of the final modulation M.

    Participation ratio of the end-of-trial M across trials. A constant
    presynaptic background makes the Hebbian outer product grow in one shared
    direction on every trial, so softplus / sigmoid collapse to a few
    dimensions while ReLU, linear and tanh keep an order of magnitude more.
    This replaces an earlier hidden-saturation panel: ReLU is as
    zero-derivative as sigmoid yet trains best, so saturation is not the
    explanatory variable.
    """
    _plot_activation_diagnosis_metric(
        "final_modulation.trial_effective_dimension",
        "Trial-specific dimension\nof final modulation",
        "l2e4_activation_modulation_dimension.png",
        percentage=False,
    )


def plot_l2e4_activation_modulation_common_accuracy():
    """Figure D: accuracy versus common-across-trial modulation energy."""
    _ensure_out_dir()
    records = _load_activation_diagnosis_records()
    if records is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(2.65, 2.35))
    all_x, all_y = [], []
    counts = []
    for activation, label, color in _ACTIVATION_DIAGNOSIS_SPECS:
        selected = [
            record for record in records
            if record.get("activation") == activation
        ]
        x = np.asarray([
            100.0 * _diagnosis_record_value(
                record,
                "final_modulation.common_across_trials_energy_fraction",
            )
            for record in selected
        ], dtype=float)
        y = np.asarray([
            100.0 * _diagnosis_record_value(record, "database_accuracy")
            for record in selected
        ], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size == 0:
            continue
        ax.scatter(
            x, y, color=color, edgecolors="k", linewidths=0.5, s=38,
            alpha=0.8, label=f"{label} (n={x.size})", zorder=3,
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
        counts.append(f"{label} n={x.size}")

    all_x = np.asarray(all_x, dtype=float)
    all_y = np.asarray(all_y, dtype=float)
    if all_x.size == 0:
        plt.close(fig)
        print("  Skipped: no finite modulation/accuracy pairs.")
        return

    ax.set_xlabel("Common-across-trial\nmodulation energy (%)")
    ax.set_ylabel("Mean accuracy\nacross tasks (%)")
    ax.set_xlim(0.0, 105.0)
    y_pad = max(2.0, 0.08 * (all_y.max() - all_y.min()))
    ax.set_ylim(max(0.0, all_y.min() - y_pad),
                min(100.0, all_y.max() + y_pad))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, color="0.85", zorder=0)
    _legend(ax, frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    _save_fig(
        fig,
        _multitask_out("l2e4_activation_modulation_common_accuracy.png"),
        extra=" (" + ", ".join(counts) + ")",
    )


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
        ax.scatter(np.full(len(values), position), values, **_SCATTER_POINT_STYLE)
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
RVAL_RESULT_PATH = STATE_SPACE_DIR / "initial_condition_distance_vs_angle_results.pkl"
STATE_SPACE_EXAMPLE_L2 = 1e-4
STATE_SPACE_NOISE_LEVEL = 0.01
STATE_SPACE_L2_STRENGTHS = (1e-5, 1e-4, 1e-3, 1e-2)


def _eff_mod_task_centroid_metrics(data):
    """Score full-dimensional task centers using the six paper color groups."""
    record = data["pca_results"]["eff_mod"]
    all_rules = list(data["all_rules"])
    if record.get("centroid_space") != "original_features":
        raise ValueError("Missing original-feature task centers; rerun state_space_shift.py.")
    counts = np.asarray(record["task_trial_counts"])
    if (len(set(all_rules)) != len(all_rules)
            or counts.shape != (len(all_rules),)
            or not np.all(np.isfinite(counts)) or np.any(counts <= 0)):
        raise ValueError("Every distinct task must have context samples.")
    categories = [_RULE_MOTIF[rule][1] for rule in all_rules]
    return task_centroid_separation(record["task_centroids"], categories)


def _l2_feature_tag(strength):
    """The run-name feature tag of an L2 strength: 1e-4 -> 'L21e4'."""
    return f"L21e{-int(round(np.log10(strength)))}"


def _state_space_centroid_ranking(result_dict, pca_loader=None, strength=STATE_SPACE_EXAMPLE_L2):
    """Rank all valid seeds of the L2 `strength` cohort and summarize their
    high-dimensional separation."""
    if pca_loader is None:
        pca_loader = _load_state_space_pca
    rankings, skipped = [], []
    for aname in sorted(result_dict):
        try:
            entry = result_dict[aname]
            l2_strength = float(entry["l2_info"])
        except (KeyError, TypeError, ValueError):
            continue
        if (not np.isfinite(l2_strength)
                or not np.isclose(l2_strength, strength, rtol=1e-6, atol=0.0)):
            continue
        try:
            data = pca_loader(aname)
            if data is None:
                raise ValueError("No state-space PCA cache at the configured noise level.")
            metrics = _eff_mod_task_centroid_metrics(data)
        except (KeyError, IndexError, TypeError, ValueError, OSError,
                EOFError, pickle.UnpicklingError) as exc:
            skipped.append({"aname": aname, "reason": str(exc)})
            continue
        rankings.append({"aname": aname, **metrics})
    rankings.sort(key=lambda row: (-row["score"], row["aname"]))
    summary = {}
    if rankings:
        for metric in ("score", "within_distance", "between_distance"):
            values = np.asarray([row[metric] for row in rankings])
            summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else None,
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return {
        "criterion": "(between_distance - within_distance) / (between_distance + within_distance)",
        "space": "original effective-modulation features (W * M)",
        "weighting": "equal within-category means and equal between-category-pair means",
        "l2": strength,
        "noise_level": STATE_SPACE_NOISE_LEVEL,
        "n_candidates": len(rankings) + len(skipped),
        "n_scored": len(rankings),
        "best_aname": rankings[0]["aname"] if rankings else None,
        "rankings": rankings,
        "summary": summary,
        "skipped": skipped,
    }


def _best_state_space_trial(result_dict, pca_loader=None, strength=STATE_SPACE_EXAMPLE_L2):
    """Select maximal task-center separation; break ties by run name."""
    report = _state_space_centroid_ranking(result_dict, pca_loader, strength)
    if not report["rankings"]:
        return None
    best = report["rankings"][0]
    return best["aname"], best["score"]


def _load_best_state_space_trial(strength=STATE_SPACE_EXAMPLE_L2):
    """Report all seed scores of the L2 `strength` cohort and select its example
    without a 2D fallback. The default cohort's ranking is saved as
    state_space_centroid_scores.json, another cohort's with its feature tag
    (state_space_centroid_scores_L21e2.json)."""
    result_dict = _load_pkl_or_skip(
        RVAL_RESULT_PATH, "Run python multiple_task/state_space_shift.py first.")
    if result_dict is None:
        return None
    report = _state_space_centroid_ranking(result_dict, strength=strength)
    _ensure_out_dir()
    suffix = "" if strength == STATE_SPACE_EXAMPLE_L2 else f"_{_l2_feature_tag(strength)}"
    report_path = _multitask_out(f"state_space_centroid_scores{suffix}.json")
    with report_path.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"  Task-center ranking: {report['n_scored']}/{report['n_candidates']} seeds "
          f"at L2={strength:.0e}; saved {report_path}")
    for row in report["rankings"]:
        print(f"    {row['aname']}: S={row['score']:.6f}, "
              f"within={row['within_distance']:.6g}, "
              f"between={row['between_distance']:.6g}")
    for row in report["skipped"]:
        print(f"    Skipped {row['aname']}: {row['reason']}")
    if report["skipped"] or not report["rankings"]:
        print("  Populate missing task centers: python multiple_task/state_space_shift.py")
    if not report["rankings"]:
        return None
    stats = report["summary"]["score"]
    spread = f"{stats['std']:.6f}" if stats["std"] is not None else "N/A (one seed)"
    print(f"  Across scored seeds: mean S={stats['mean']:.6f}, sample SD={spread}")
    best = report["rankings"][0]
    aname, score = best["aname"], best["score"]
    print(f"  State-space trial selected automatically: {aname} "
          f"(high-dimensional task-center separation={score:.6f})")
    return aname, score, result_dict[aname]


def _load_state_space_pca(aname):
    """Load state_space_shift's shared PCA and original-feature task centers."""
    path = STATE_SPACE_DIR / f"state_space_pca_{aname}_noise{STATE_SPACE_NOISE_LEVEL}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        return pickle.load(stream)


# Legend for the state-space panels: one entry per DISTINCT color in
# _RULE_MOTIF, labeled by what that color means across figures (dms shares
# the Reaction hues by the match=pro / non-match=anti pairing, so green and
# orange cover both).
_STATE_SPACE_LEGEND = [
    ("Pro Delayed", _SCATTER_POINT_COLOR),
    ("Anti Delayed", "#e53e3e"),
    ("Pro Reaction / Match", "#38a169"),
    ("Anti Reaction / Non-match", "#dd6b20"),
    ("Integration", "#805ad5"),
    ("Category", "#ff1493"),
]


# The two cohorts of the context-end PCA figures, drawn side by side: the
# default example cohort on the left and the cohort with the strongest
# category structure (state_space_category_test) on the right, each with its
# own automatically selected run.
STATE_SPACE_STRONG_L2 = 1e-2
STATE_SPACE_PCA_COHORTS = (STATE_SPACE_EXAMPLE_L2, STATE_SPACE_STRONG_L2)

# Figure size of one two-panel state-space PCA figure. With the square axes
# boxes, the two-line y label of the left panel, the panel titles and the
# two-row color key across the top this lands each drawn box at about 1.6 in,
# the size of one `_render_multitask_cue_interp_2d` subpanel (its 1.8 in
# column minus labels).
_STATE_SPACE_FIGSIZE = (4.4, 2.5)

# The three context-end PCA representations, as (cache key, display name).
_STATE_SPACE_PCA_REPRESENTATIONS = (
    ("hidden", "Hidden state"),
    ("mod", "Modulation"),
    ("eff_mod", "Eff. modulation"),
)


def _draw_state_space_panel(ax, data, key, ylabel_prefix=None):
    """One context-end PCA scatter on `ax`, colored PER TASK from _RULE_MOTIF
    — the single color mapping every figure shares (same colors as the
    lesion-heatmap task labels). The pickle's rule_motif_mapping supplies data
    only, so recoloring never requires re-running state_space_shift.py. The y
    label names the representation when `ylabel_prefix` is given (the left
    panel of a figure) and is a plain "PC2" otherwise."""
    all_rules = data["all_rules"]
    pca = data["pca_results"][key]
    X_2d, ctx_rule_labels = pca["X_2d"], pca["ctx_rule_labels"]
    for idx, rule in enumerate(all_rules):
        sel = ctx_rule_labels == idx
        ax.scatter(X_2d[sel, 0], X_2d[sel, 1],
                   color=_RULE_MOTIF[rule][1], alpha=0.5, s=14,
                   edgecolors="none")

    # A square box the size of one rule-cue sweep subpanel
    # (`_render_multitask_cue_interp_2d`, about 1.6 in), so the state-space
    # and sweep panels can sit side by side at one scale.
    ax.set_box_aspect(1)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2" if ylabel_prefix is None else f"{ylabel_prefix}\nPC2", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    # A few round ticks per axis whatever the PC range (the ranges differ by
    # orders of magnitude between representations and cohorts), labelled
    # without trailing zeros, as in the association scatters.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.MaxNLocator(4, steps=[1, 2, 5, 10]))
        axis.set_major_formatter(mticker.FuncFormatter(_short_tick))


def _state_space_legend_handles():
    """One color-key entry per DISTINCT category color, from the shared color
    list rather than the scatters (dms shares the Reaction hues)."""
    return [mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=col, alpha=0.5,
                             markersize=4, label=lab)
            for lab, col in _STATE_SPACE_LEGEND]


def _plot_state_space_figure(cohort_data, key, ylabel_prefix, out_name):
    """One context-end PCA figure of representation `key`: one panel per
    (strength, PCA cache) of `cohort_data`, left to right, each titled by its
    L2 strength, with one color key across the top of the figure (gated by
    SHOW_LEGEND like every legend; inside the axes it covered task clusters
    in some runs). The panels are separate PCA spaces, so each keeps its own
    axes; only the left one names the representation in its y label."""
    fig, axs = plt.subplots(1, len(cohort_data), figsize=_STATE_SPACE_FIGSIZE)
    for index, (ax, (strength, data)) in enumerate(zip(np.atleast_1d(axs), cohort_data)):
        _draw_state_space_panel(ax, data, key, ylabel_prefix if index == 0 else None)
        ax.set_title(rf"$L_2 = 10^{{{int(round(np.log10(strength)))}}}$", fontsize=8, pad=4)
    _state_space_figure_legend(fig, _state_space_legend_handles(), ncol=3, top=0.88)
    _save_fig(fig, _multitask_out(out_name))


def plot_state_space_combined():
    """
    Figures: context-end PCA colored by task category — hidden state, raw
    modulation and effective modulation, as THREE figures of two panels each:
      multitask_state_space_hidden.png
      multitask_state_space_mod.png
      multitask_state_space_eff_mod.png

    The left panel is the run of the L2 = STATE_SPACE_EXAMPLE_L2 cohort with
    the best high-dimensional task-center separation, the right panel the run
    of the L2 = STATE_SPACE_STRONG_L2 cohort, whose category structure is the
    strongest (state_space_category_test), selected by the same criterion
    (STATE_SPACE_PCA_COHORTS; the rankings are saved as
    state_space_centroid_scores.json and state_space_centroid_scores_L21e2.json).
    The panels are separate PCA spaces whose PCs are not comparable, so each
    keeps its own axes; ``mod`` plots M and ``eff_mod`` plots M elementwise
    multiplied by the learned plastic weight matrix W. A figure needs both
    cohorts: when either has no analyzed run or no PCA cache, the three
    figures are skipped with a message rather than drawn with one panel.
    """
    _ensure_out_dir()
    cohort_data = []
    for strength in STATE_SPACE_PCA_COHORTS:
        selected = _load_best_state_space_trial(strength)
        if selected is None:
            print(f"  Skipped: no analyzed state-space run at L2={strength:.0e}.")
            return
        aname = selected[0]
        data = _load_state_space_pca(aname)
        if data is None:
            print(f"  Skipped: state-space PCA pickle for {aname} not found. "
                  "Run state_space_shift.py first.")
            return
        cohort_data.append((strength, data))

    for key, ylabel_prefix in _STATE_SPACE_PCA_REPRESENTATIONS:
        _plot_state_space_figure(cohort_data, key, ylabel_prefix, f"state_space_{key}.png")


# Untrained control of the PCA figures: state_space_shift.py --untrained
# writes the PCA of a freshly initialized network with a run's architecture
# and config into this subdirectory, out of reach of the trained caches'
# readers. Three square panels at the trained figures' panel size.
STATE_SPACE_UNTRAINED_DIR = STATE_SPACE_DIR / "untrained"
# The untrained network drawn in the PCA figure, by the seed of its weights
# (state_space_shift.UNTRAINED_INIT_SEED, the default of --init-seed).
STATE_SPACE_UNTRAINED_INIT_SEED = 0
_STATE_SPACE_UNTRAINED_FIGSIZE = (6.4, 2.5)


def _untrained_command(aname, init_seeds=(STATE_SPACE_UNTRAINED_INIT_SEED,)):
    """The command that builds untrained networks from run `aname`'s config."""
    seeds = " ".join(str(seed) for seed in init_seeds)
    return ("python multiple_task/state_space_shift.py --untrained "
            f"--checkpoint multiple_tasks/savednet_{aname}.pt --init-seed {seeds}")


def _load_state_space_untrained_pca(aname, init_seed=STATE_SPACE_UNTRAINED_INIT_SEED):
    """The PCA cache of the untrained network built from run `aname`'s config
    with weights from `init_seed`, or None (with a message) when it is
    missing or not marked untrained."""
    path = (STATE_SPACE_UNTRAINED_DIR
            / f"state_space_pca_{aname}_init{init_seed}_noise{STATE_SPACE_NOISE_LEVEL}.pkl")
    command = _untrained_command(aname, (init_seed,))
    if not path.exists():
        print(f"  Skipped: no untrained-network PCA cache for {aname} "
              f"(init seed {init_seed}). Run: {command}")
        return None
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if data.get("untrained") is not True:
        print(f"  Skipped: {path} is not marked as an untrained network. Run: {command}")
        return None
    return data


def plot_state_space_untrained():
    """
    Figure: multitask_state_space_untrained.png — the context-end PCA of an
    UNTRAINED network, one panel per representation (hidden state, raw
    modulation M, effective modulation W * M), drawn like the trained PCA
    figures (plot_state_space_combined): same colors, panel size and color key.

    The network has the architecture and config of the selected
    L2 = STATE_SPACE_EXAMPLE_L2 example run (the left panels of the trained
    figures) but fresh random weights drawn from STATE_SPACE_UNTRAINED_INIT_SEED and eta
    at its initial value; no trained weight is used, and the training run's
    own initialization was never saved. It runs on the same evaluation trials
    (EVALUATION_SEED), so any task separation here comes from the rule-cue
    inputs passing through random weights, the baseline the trained figures
    are compared against. The L2 strength only acts during training, so one
    untrained network stands for every cohort. Each panel is its own PCA
    space; the save message adds the category separation S of each
    representation's full-dimensional task centroids. Read from
    state_space/untrained/; nothing is recomputed from network activity.
    """
    _ensure_out_dir()
    selected = _load_best_state_space_trial()
    if selected is None:
        return
    aname = selected[0]
    data = _load_state_space_untrained_pca(aname)
    if data is None:
        return

    fig, axs = plt.subplots(1, len(_STATE_SPACE_PCA_REPRESENTATIONS),
                            figsize=_STATE_SPACE_UNTRAINED_FIGSIZE)
    scores = []
    categories = [_RULE_MOTIF[rule][1] for rule in data["all_rules"]]
    for index, (ax, (key, title)) in enumerate(zip(axs, _STATE_SPACE_PCA_REPRESENTATIONS)):
        _draw_state_space_panel(ax, data, key, "Untrained" if index == 0 else None)
        ax.set_title(title, fontsize=8, pad=4)
        centroids = data["pca_results"][key].get("task_centroids")
        if centroids is not None:
            score = task_centroid_separation(centroids, categories)["score"]
            scores.append(f"{key} S = {score:.4f}")
    _state_space_figure_legend(fig, _state_space_legend_handles(), ncol=3, top=0.88)
    _save_fig(fig, _multitask_out("state_space_untrained.png"),
              extra=(f"  (config of {aname}, init seed {data.get('init_seed')}; "
                     f"{', '.join(scores)})"))


def _state_space_r_values_by_l2(result_dict, data_types):
    """Per analyzed L2 strength and representation, the runs' saved
    distance-angle r and two-sided task-label permutation p as
    {"r", "p_perm", "aname"} dicts. Runs of other strengths are left out; a
    run whose cache predates the permutation test (no saved p) is left out
    with a message rather than drawn with a made-up significance."""
    grouped = {
        strength: {data_type: [] for data_type in data_types}
        for strength in STATE_SPACE_L2_STRENGTHS
    }
    legacy = 0
    for aname, results in result_dict.items():
        saved_strength = results.get("l2_info")
        if saved_strength is None:
            continue
        strength = next(
            (candidate for candidate in STATE_SPACE_L2_STRENGTHS
             if np.isclose(float(saved_strength), candidate,
                           rtol=1e-6, atol=0.0)),
            None,
        )
        if strength is None:
            continue
        legacy += _add_r_value_entry(grouped[strength], aname, results, data_types)
    if legacy:
        print(f"  Skipped {legacy} run/representation entries without a saved task-label "
              "permutation test (legacy cache). Re-run multiple_task/state_space_shift.py.")
    return grouped


def _add_r_value_entry(group, name, results, data_types):
    """Append one results entry's saved r and permutation p to `group`
    ({data type: [{"r", "p_perm", "aname"}]}) for each of `data_types` it
    holds. Returns the number of data types left out because the entry has
    no saved permutation test (legacy cache)."""
    legacy = 0
    for data_type in data_types:
        if data_type not in results.get("rval_dict", {}):
            continue
        permutation = (results.get("scatter") or {}).get(data_type, {}).get("permutation")
        if not permutation or "p_value" not in permutation:
            legacy += 1
            continue
        group[data_type].append({
            "r": float(results["rval_dict"][data_type][0]),
            "p_perm": float(permutation["p_value"]),
            "aname": name,
        })
    return legacy


# The untrained networks' own distance-angle results (state_space_shift.py
# --untrained), keyed by file stem, next to their PCA caches.
STATE_SPACE_UNTRAINED_RVAL_PATH = STATE_SPACE_UNTRAINED_DIR / RVAL_RESULT_PATH.name


def _untrained_r_values(data_types):
    """{data type: [{"r", "p_perm", "aname"}]} of the untrained networks built
    from the selected example run's config (the networks of the untrained
    PCA and category figures), or None, with the command to run, when there
    are none."""
    if not STATE_SPACE_UNTRAINED_RVAL_PATH.exists():
        print(f"  No untrained distance-angle results ({STATE_SPACE_UNTRAINED_RVAL_PATH}); "
              "drawing the trained cohorts only. Add them with: "
              "python multiple_task/state_space_shift.py --untrained --checkpoint "
              "multiple_tasks/savednet_<example aname>.pt --init-seed 0 1 2 3 4 5 6 7")
        return None
    with STATE_SPACE_UNTRAINED_RVAL_PATH.open("rb") as stream:
        results = pickle.load(stream)
    selected = _load_best_state_space_trial()
    if selected is None:
        return None
    aname = selected[0]
    group = {data_type: [] for data_type in data_types}
    legacy = 0
    for stem in sorted(results):
        entry = results[stem]
        if entry.get("untrained") is True and entry.get("config_source") == aname:
            legacy += _add_r_value_entry(group, stem, entry, data_types)
    if not any(group.values()):
        print(f"  No untrained distance-angle results built from {aname}'s config; drawing "
              f"the trained cohorts only. Add them with: {_untrained_command(aname)}")
        return None
    if legacy:
        print(f"  Skipped {legacy} untrained network/representation entries without a saved "
              "permutation test.")
    return group


# Bars of the R-value figure: (saved name, label, color, hatched). A hatched
# bar is the Hebbian-only first step of the full-step bar of the same color
# (state_space_shift.DISTANCE_ANGLE_REPRESENTATIONS).
_STATE_SPACE_R_BARS = (
    ("hidden", "Hidden", _SCATTER_POINT_COLOR, False),
    ("mod", "Mod.", "#dd6b20", False),
    ("eff_mod", "Eff.\nMod.", "#38a169", False),
    ("mod_hebb", "Mod.\nHebb", "#dd6b20", True),
    ("eff_mod_hebb", "Eff. Mod.\nHebb", "#38a169", True),
)
_STATE_SPACE_HEBB_HATCH = "///"


def _draw_r_value_panel(ax, runs, title, jitter):
    """One panel of the R-value figure: per bar of _STATE_SPACE_R_BARS the
    mean r of `runs` ({data type: [{"r", "p_perm"}]}) with the sample SD, and
    one dot per run or network, filled when its permutation p is below 0.05.
    Returns the lowest r drawn (0 when none)."""
    lowest = 0.0
    if not any(runs[name] for name, _, _, _ in _STATE_SPACE_R_BARS):
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
    positions = np.arange(len(_STATE_SPACE_R_BARS))
    for position, (name, _, color, hatched) in zip(positions, _STATE_SPACE_R_BARS):
        values = np.array([run["r"] for run in runs[name]], dtype=float)
        if values.size == 0:
            continue
        lowest = min(lowest, float(values.min()))
        spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
        ax.bar(position, values.mean(), yerr=spread, capsize=4, color=color,
               edgecolor="k", linewidth=0.6, width=0.6,
               hatch=_STATE_SPACE_HEBB_HATCH if hatched else None)
        for offset, run in zip(jitter.uniform(-0.12, 0.12, values.size), runs[name]):
            significant = run["p_perm"] < _STATE_SPACE_SIGNIFICANCE
            ax.scatter(position + offset, run["r"], s=15, zorder=5,
                       color="k" if significant else "white",
                       edgecolors="k", linewidths=0.6)
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label, _, _ in _STATE_SPACE_R_BARS], fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)
    return lowest


def plot_state_space_r_values():
    """
    Figure: multitask_state_space_r_values.png — the distance-angle r of every
    analyzed run, one panel per L2 strength and one bar per representation:
    hidden state, raw modulation and effective modulation with the full first
    step, then (hatched, same colors) the two modulation representations with
    the Hebbian term of the step only (the decay-term control explained in
    plot_state_space_dist_angle). Bars are the mean over runs with
    the sample SD; the dots are the runs, filled when the run's saved two-sided
    task-label permutation p is below 0.05 and hollow otherwise. Nothing is
    recomputed; runs whose cache has no saved permutation test are left out.

    When the untrained networks built from the selected example run's config
    have distance-angle results (state_space_shift.py --untrained), a first
    panel, "Untrained", draws them the same way, one dot per init seed: the
    r that the rule-cue inputs and the Hebbian update give without any
    learning, against which the trained cohorts are read. Without them the
    figure keeps its four cohort panels and says how to add the control.
    """
    _ensure_out_dir()
    result_dict = _load_pkl_or_skip(RVAL_RESULT_PATH, "Run state_space_shift.py first.")
    if result_dict is None:
        return

    names = [name for name, _, _, _ in _STATE_SPACE_R_BARS]
    grouped = _state_space_r_values_by_l2(result_dict, names)
    panels = [(rf"$L_2 = 10^{{{int(round(np.log10(strength)))}}}$", grouped[strength])
              for strength in STATE_SPACE_L2_STRENGTHS]
    untrained = _untrained_r_values(names)
    if untrained is not None:
        panels.insert(0, ("Untrained", untrained))
    fig, axs = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 3.2), sharey=True)
    jitter = np.random.default_rng(42)
    lowest = min(_draw_r_value_panel(ax, runs, title, jitter)
                 for ax, (title, runs) in zip(axs, panels))
    axs[0].set_ylabel("R-value")
    axs[0].set_ylim(min(0.0, lowest - 0.05), 1.05)

    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="none", color="k", markersize=4,
                         label=f"permutation p < {_STATE_SPACE_SIGNIFICANCE}"),
        mpl.lines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                         markeredgecolor="k", markersize=4,
                         label=rf"p $\geq$ {_STATE_SPACE_SIGNIFICANCE}"),
        mpl.patches.Patch(facecolor="white", edgecolor="k", hatch=_STATE_SPACE_HEBB_HATCH,
                          label="Hebbian term only (decay term removed)"),
    ]
    _state_space_figure_legend(fig, handles)
    _save_fig(fig, _multitask_out("state_space_r_values.png"))


_DISTANCE_ANGLE_XLABEL = "Distance between\ninitial conditions"
_DISTANCE_ANGLE_YLABEL = "Angle between first step\nof trajectories (deg.)"


def _distance_angle_panel(entry, key):
    """The saved task-pair coordinates and statistics of one representation
    (`key`) of a distance-angle results entry written by state_space_shift.py,
    or None with a message when the cache predates the free-intercept fit or
    the task-label permutation test. Nothing is refitted or recomputed."""
    scatter = (entry.get("scatter") or {}).get(key)
    if scatter is None:
        print(f"  Skipped: no saved scatter for {key} (legacy cache). "
              "Re-run multiple_task/state_space_shift.py.")
        return None
    regression = scatter.get("regression", {})
    if regression.get("through_origin") is not False or "intercept" not in regression:
        print(f"  Skipped: {key} has no saved free-intercept regression (legacy cache). "
              "Re-run multiple_task/state_space_shift.py.")
        return None
    permutation = scatter.get("permutation")
    if not permutation or "p_value" not in permutation:
        reason = scatter.get("permutation_skipped") or "legacy cache"
        print(f"  Skipped: {key} has no saved task-label permutation test ({reason}). "
              "Re-run multiple_task/state_space_shift.py.")
        return None
    r_value, slope, _ = entry["rval_dict"][key]
    return {
        "x": np.asarray(scatter["dists"], float),
        "y": np.asarray(scatter["angles_deg"], float),
        "r": float(r_value),
        "slope": float(slope),
        "intercept": float(regression["intercept"]),
        "p_perm": float(permutation["p_value"]),
        "step_norms": scatter.get("step_norms"),
    }


def _draw_distance_angle_panel(ax, panel, title):
    """One distance-angle scatter: the saved task-pair points, the saved OLS
    guide and, as the legend, r with its two-sided task-label permutation p."""
    _stat_scatter(ax, panel["x"], panel["y"],
                  f"r = {panel['r']:.2f}\npermutation {_permutation_p_text(panel['p_perm'])}",
                  trend_line={"slope": panel["slope"], "intercept": panel["intercept"]},
                  # Two lines: the square axes box is narrower than the label.
                  xlabel=_DISTANCE_ANGLE_XLABEL, ylabel=_DISTANCE_ANGLE_YLABEL)
    # The statistics sit above the axes, so the panel title goes higher.
    ax.set_title(title, fontsize=9, pad=22)


# Panels of the distance-angle figure: (saved name, title). The last two share
# their initial conditions (x) and are drawn on one y range: the full first
# step of W * M and its Hebbian term alone (state_space_shift.py).
_STATE_SPACE_DIST_ANGLE_PANELS = (
    ("hidden", "Hidden state"),
    ("eff_mod", "Eff. modulation"),
    ("eff_mod_hebb", "Eff. modulation,\nHebbian term only"),
)


def plot_state_space_dist_angle():
    """
    Figure: multitask_state_space_dist_angle.png — initial-condition distance
    vs first-step trajectory angle for the automatically selected L2=1e-4
    state-space trial with the best task-center separation in full
    effective-modulation space; the per-task-pair scatter that the R-value
    bars (plot_state_space_r_values) summarize. The same selected trial
    supplies the hidden-state and effective-modulation PCA figures.

    Three panels: the hidden state and the effective modulation W * M with
    the full first step x[t] - x[t-1], then W * M with the Hebbian term of
    that step only. The modulation update is
    M[t] = lam * M[t-1] + eta * post (x) pre, so the full step contains
    -(1 - lam) * M[t-1]: a component along the initial condition itself,
    which would relate the step angle to the initial-condition distance
    without any task-dependent computation. The third panel removes it
    (x[t] - lam * x[t-1]), so a correlation that survives there is carried by
    the task pairs' post- and presynaptic activity at stimulus onset. The two
    W * M panels share their x coordinates and are drawn on one y range, so
    the change in angles is visible. Each point is one task pair: x = mean
    Euclidean distance between the two tasks' pre-stimulus states (end of the
    Context period, matched by stimulus), y = mean angle (deg) between their
    first post-stimulus displacement vectors. The line and r come from the
    free-intercept OLS fit state_space_shift.py computed; the p is its
    two-sided task-label permutation (Mantel-type) p, because task pairs share
    tasks. Nothing is refit; caches without the saved fit, the permutation
    test or the Hebbian step are skipped with a message to rerun the analysis.
    The saved message appends the decay term's share of the full W * M step.
    """
    _ensure_out_dir()
    selected = _load_best_state_space_trial()
    if selected is None:
        return
    _, _, entry = selected
    panels = []
    for key, title in _STATE_SPACE_DIST_ANGLE_PANELS:
        panel = _distance_angle_panel(entry, key)
        if panel is None:
            return
        panels.append((panel, title))
    fig, axs = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 2.6))
    for ax, (panel, title) in zip(axs, panels):
        _draw_distance_angle_panel(ax, panel, title)
    # One y range for the full and the Hebbian-only step of W * M.
    modulation = [panel for panel, _ in panels[1:]]
    low = min(float(panel["y"].min()) for panel in modulation)
    high = max(float(panel["y"].max()) for panel in modulation)
    pad = 0.05 * ((high - low) or 1.0)
    for ax in axs[1:]:
        ax.set_ylim(low - pad, high + pad)
    norms = modulation[0]["step_norms"] or {}
    note = ""
    if norms.get("full_step") and "decay_term" in norms:
        note = (f"  (eff_mod decay term / full step = "
                f"{norms['decay_term'] / norms['full_step']:.3f})")
    fig.tight_layout()
    _save_fig(fig, _multitask_out("state_space_dist_angle.png"), extra=note)


# ─── Figure: State space structure (category test, RDM, task vectors) ───────
# All three read the caches of multiple_task/state_space_structure.py: the
# per-run `state_space_structure_<aname>_noise0.01.pkl` (distances, linkage,
# null arrays) and the combined `state_space_structure_results.pkl` (every
# run's statistics). Nothing is recomputed here.

STATE_SPACE_STRUCTURE_RESULT_PATH = STATE_SPACE_DIR / "state_space_structure_results.pkl"
_STATE_SPACE_STRUCTURE_HINT = "Run python multiple_task/state_space_structure.py first."
# The two representations the structure figures show, as (key, title); raw M
# stays in the caches but, as in the distance-angle figure, is not drawn.
_STATE_SPACE_STRUCTURE_PANELS = (("hidden", "Hidden state"), ("eff_mod", "Eff. modulation"))
# The distance-matrix figures show the effective modulation only.
_STATE_SPACE_RDM_PANELS = (("eff_mod", "Eff. modulation"),)
_STATE_SPACE_RDM_FIGSIZE = (3.0, 3.5)
_STATE_SPACE_SIGNIFICANCE = 0.05


def _load_state_space_structure(aname):
    """Load state_space_structure.py's per-run cache, or None."""
    path = STATE_SPACE_DIR / f"state_space_structure_{aname}_noise{STATE_SPACE_NOISE_LEVEL}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        return pickle.load(stream)


def _permutation_p_text(p_value):
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def _run_offsets(n_runs, step, half_width):
    """x offsets that set the per-run points of one column side by side:
    `step` apart and centered, but never wider than +-`half_width` (the
    column's median dash), so a large cohort packs its points closer instead
    of spilling into the neighboring columns. The order is the runs' order,
    which carries no meaning."""
    if n_runs < 2:
        return np.zeros(n_runs)
    step = min(step, 2 * half_width / (n_runs - 1))
    return (np.arange(n_runs) - (n_runs - 1) / 2) * step


def _significance_marker(ax, x, y, significant, size=32, color=_SCATTER_POINT_COLOR,
                         hollow_edge="k"):
    """One statistic as a point: filled in `color` when its saved permutation
    p is below 0.05, hollow with a `hollow_edge` ring otherwise (the
    convention of every per-run `_seeds` figure; the lesion-mode ones pass
    _LESION_POINT_COLOR for both)."""
    ax.scatter([x], [y], s=size, zorder=3,
               color=color if significant else "white",
               edgecolors=color if significant else hollow_edge, linewidths=0.6)


def _significance_legend_handles(color=_SCATTER_POINT_COLOR, hollow_edge="k"):
    """Legend entries matching _significance_marker with the same colors."""
    filled = mpl.lines.Line2D([], [], marker="o", linestyle="none", color=color,
                              markersize=5, label=f"permutation p < {_STATE_SPACE_SIGNIFICANCE}")
    hollow = mpl.lines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                              markeredgecolor=hollow_edge, markersize=5,
                              label=rf"p $\geq$ {_STATE_SPACE_SIGNIFICANCE}")
    return [filled, hollow]


def _plot_state_space_rdm_panel(rules, entry, legend_text, out_name, colorbar_name,
                                colorbar_label, key):
    """One task distance matrix (`entry`: distances, linkage, leaf_order and
    separation, from a per-run or a cohort-mean cache), ordered by its saved
    average-linkage tree drawn above it, with `legend_text` as the legend. The
    matrix is symmetric, so only its lower triangle (with the zero diagonal)
    is drawn; tick labels carry the task display names on their category
    color. The color scale is exported as its own figure, `colorbar_name`, in
    the style of the task-variance heatmap colorbars."""
    rules = list(rules)
    distances = np.asarray(entry["distances"], dtype=float)
    fig = plt.figure(figsize=_STATE_SPACE_RDM_FIGSIZE)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.22, 1.0], hspace=0.04)
    ax_tree = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])

    # The dendrogram decides the leaf order on screen; the heatmap follows it,
    # and the saved order (same scipy routine) is checked rather than trusted.
    tree = dendrogram(np.asarray(entry["linkage"], dtype=float), ax=ax_tree, no_labels=True,
                      color_threshold=0.0, above_threshold_color="0.35")
    order = np.asarray(tree["leaves"], dtype=int)
    if not np.array_equal(order, np.asarray(entry["leaf_order"], dtype=int)):
        print(f"  Note: {key} dendrogram leaf order differs from the saved order; "
              "the drawn tree's order is used.")
    ax_tree.set_axis_off()

    ordered = distances[np.ix_(order, order)]
    vmax = float(ordered.max())
    lower = np.ma.masked_where(np.triu(np.ones_like(ordered, dtype=bool), k=1), ordered)
    cmap = plt.get_cmap(_MULTITASK_HEATMAP_CMAP).copy()
    cmap.set_bad("none")                       # masked upper triangle stays blank
    ax.imshow(lower, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal")
    names = [_TASK_DISPLAY.get(rules[index], rules[index]) for index in order]
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor", fontsize=5)
    ax.set_yticklabels(names, fontsize=5)
    ax.tick_params(length=0, pad=1.5)
    for axis in ("x", "y"):
        _color_motif_ticklabels(ax, [rules[index] for index in order], axis=axis)
    for spine in ax.spines.values():
        spine.set_visible(False)

    separation = entry["separation"]
    _legend(ax_tree, [_invisible_legend_handle()], [legend_text], handlelength=0,
            handletextpad=0, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0,
            fontsize=6, frameon=False)
    _save_fig(fig, _multitask_out(out_name),
              extra=f"  ({key}: S = {separation['score']:.4f}, "
                    f"{_permutation_p_text(separation['p_value'])})")
    # Each representation has its own distance scale, so each exports its own
    # bar, at the size and placement of plot_multitask_heatmap_colorbar.
    _save_standalone_colorbar(
        _multitask_out(colorbar_name), cmap=_MULTITASK_HEATMAP_CMAP, vmin=0.0, vmax=vmax,
        ticks=[0.0, vmax], ticklabels=["0", f"{vmax:.1f}"], label=colorbar_label,
        orientation="horizontal", figsize=(1.9, 0.62), rect=(0.05, 0.62, 0.90, 0.24),
        labelsize=8, label_fontsize=8)


def _state_space_cohort_structure(combined, strength):
    """The cohort-mean entry of the combined structure results for one L2
    strength, or None (with a message) when the cache predates cohort means
    or holds no runs of that strength."""
    cohorts = combined.get("cohorts")
    if cohorts is None:
        print(f"  Skipped: structure results hold no cohort means. {_STATE_SPACE_STRUCTURE_HINT}")
        return None
    for saved_strength, cohort in cohorts.items():
        if np.isclose(float(saved_strength), strength, rtol=1e-6, atol=0.0):
            return cohort
    print(f"  Skipped: no runs at L2={strength:.0e} in the structure results.")
    return None


def plot_state_space_rdm():
    """
    Figure: the mean 15 x 15 distance matrix between the full-dimensional
    context-end effective-modulation (W * M) task centroids over every run of
    the L2 = STATE_SPACE_EXAMPLE_L2 cohort:
      multitask_state_space_rdm_eff_mod.png

    Each run's matrix is divided by its mean task-pair distance before
    averaging (state_space_structure.py), so the scale is "distance / mean
    distance" and a run with a larger representation does not dominate. Only
    the lower triangle of the symmetric matrix is drawn. Tasks are ordered by
    the mean matrix's average-linkage tree drawn above it, so same-category
    tasks that cluster appear as blocks; tick labels sit on their category
    color. The legend reports the number of runs, the category separation S
    of the mean matrix with its one-sided label-permutation p, and the median
    seed-to-seed Spearman correlation of the runs' matrices (how far the mean
    stands for every run). Each representation's color scale is exported as a
    separate colorbar figure (multitask_state_space_rdm_colorbar_{key}.png) in
    the style of the task-variance heatmap colorbar. The single selected run's
    matrix is plot_state_space_rdm_example.
    """
    _ensure_out_dir()
    combined = _load_pkl_or_skip(STATE_SPACE_STRUCTURE_RESULT_PATH, _STATE_SPACE_STRUCTURE_HINT)
    if combined is None:
        return
    cohort = _state_space_cohort_structure(combined, STATE_SPACE_EXAMPLE_L2)
    if cohort is None:
        return
    for key, title in _STATE_SPACE_RDM_PANELS:
        if key not in cohort["representations"]:
            print(f"  Skipped {key}: not in the cohort means.")
            continue
        entry = cohort["representations"][key]
        separation, consistency = entry["separation"], entry["consistency"]
        text = (f"{title}, mean of {cohort['n_runs']} runs at "
                rf"$L_2 = 10^{{{int(round(np.log10(STATE_SPACE_EXAMPLE_L2)))}}}$: "
                f"S = {separation['score']:.3f}, {_permutation_p_text(separation['p_value'])}\n"
                rf"seed-to-seed Spearman $\rho$ median {consistency['median']:.2f} "
                f"({consistency['min']:.2f} to {consistency['max']:.2f})")
        _plot_state_space_rdm_panel(cohort["all_rules"], entry, text,
                                    f"state_space_rdm_{key}.png",
                                    f"state_space_rdm_colorbar_{key}.png",
                                    "Distance / mean distance", key)


def plot_state_space_rdm_example():
    """
    Figure: the 15 x 15 Euclidean distance matrix between the full-dimensional
    context-end effective-modulation task centroids of the automatically
    selected state-space trial (the run of the PCA and distance-angle figures):
      multitask_state_space_rdm_example_eff_mod.png

    Drawn like plot_state_space_rdm (lower triangle, tree-ordered, labels on
    their category color, separate colorbar figures
    multitask_state_space_rdm_example_colorbar_{key}.png) but from the
    per-run cache in the run's own distance units, with that run's saved
    category separation S and permutation p in the legend.
    """
    _ensure_out_dir()
    selected = _load_best_state_space_trial()
    if selected is None:
        return
    aname = selected[0]
    structure = _load_state_space_structure(aname)
    if structure is None:
        print(f"  Skipped: no state-space structure cache for {aname}. "
              f"{_STATE_SPACE_STRUCTURE_HINT}")
        return
    for key, title in _STATE_SPACE_RDM_PANELS:
        if key not in structure["representations"]:
            print(f"  Skipped {key}: not in the structure cache.")
            continue
        entry = structure["representations"][key]
        separation = entry["separation"]
        text = (f"{title}: S = {separation['score']:.3f}, "
                f"{_permutation_p_text(separation['p_value'])}")
        _plot_state_space_rdm_panel(structure["all_rules"], entry, text,
                                    f"state_space_rdm_example_{key}.png",
                                    f"state_space_rdm_example_colorbar_{key}.png",
                                    "Centroid distance", key)


def _state_space_runs_by_l2(runs):
    """Group the combined structure results by the four analyzed L2 strengths,
    skipping runs of any other strength."""
    grouped = {strength: {} for strength in STATE_SPACE_L2_STRENGTHS}
    for aname in sorted(runs):
        entry = runs[aname]
        strength = next(
            (candidate for candidate in STATE_SPACE_L2_STRENGTHS
             if np.isclose(float(entry["l2_info"]), candidate, rtol=1e-6, atol=0.0)),
            None)
        if strength is not None:
            grouped[strength][aname] = entry
    return grouped


def _l2_tick_label(strength):
    return rf"$10^{{{int(round(np.log10(strength)))}}}$"


def plot_state_space_category_test():
    """
    Figure: multitask_state_space_category_test.png — the category-balanced
    separation S of every analyzed run, per L2 cohort, for the hidden state
    and the effective modulation.

    One point per run (filled when its saved one-sided label-permutation p is
    below 0.05), its null's 95th percentile as a gray tick behind it, and the
    median over the cohort's runs as a dash. S near zero means the six paper
    categories group the context-end centroids no better than a random
    regrouping of the same sizes. Read from the combined
    state_space_structure.py results; nothing is recomputed.
    """
    _ensure_out_dir()
    combined = _load_pkl_or_skip(STATE_SPACE_STRUCTURE_RESULT_PATH, _STATE_SPACE_STRUCTURE_HINT)
    if combined is None:
        return
    grouped = _state_space_runs_by_l2(combined["runs"])
    columns = [(_l2_tick_label(strength), f"L2 {strength:.0e}", list(grouped[strength].values()))
               for strength in STATE_SPACE_L2_STRENGTHS]
    _plot_separation_columns(columns, np.arange(len(columns)), figsize=(4.6, 2.4),
                             out_name="state_space_category_test.png")


def _draw_separation_column(ax, position, runs, key):
    """One column of a category-separation figure at x = `position`: the saved
    S of representation `key` of every run (or untrained network) in `runs`
    as a point, filled when its one-sided label-permutation p is below 0.05,
    the range of their null 95th percentiles as a gray band behind them, and
    their median as a dash. Returns the "significant/total" count."""
    entries = [run["representations"][key] for run in runs if key in run["representations"]]
    scores = [entry["separation"]["score"] for entry in entries]
    quantiles = [entry["separation"]["null_quantile_95"] for entry in entries]
    if quantiles:
        # The runs' null 95th percentiles nearly coincide, so one band
        # from the lowest to the highest spans the column.
        ax.fill_between([position - 0.3, position + 0.3], min(quantiles),
                        max(quantiles), color="0.82", linewidth=0, zorder=1)
    offsets = _run_offsets(len(entries), 0.07, 0.3)
    significant = 0
    for offset, entry in zip(offsets, entries):
        separation = entry["separation"]
        is_significant = separation["p_value"] < _STATE_SPACE_SIGNIFICANCE
        significant += int(is_significant)
        _significance_marker(ax, position + offset, separation["score"], is_significant,
                             size=22)
    if scores:
        ax.hlines(np.median(scores), position - 0.3, position + 0.3,
                  color=_TREND_LINE_COLOR, linewidth=1.8, zorder=4)
    return f"{significant}/{len(entries)}"


def _plot_separation_columns(columns, positions, figsize, out_name, divider=None):
    """A two-panel category-separation figure (hidden state, effective
    modulation): one column per (tick label, summary name, runs) of `columns`
    at `positions`, each drawn by _draw_separation_column, with the
    significant/total count under its tick label, an optional dotted vertical
    line at x = `divider`, one shared legend and the counts in the save
    message."""
    fig, axs = plt.subplots(1, len(_STATE_SPACE_STRUCTURE_PANELS), figsize=figsize, sharey=True)
    summaries = []
    for ax, (key, title) in zip(axs, _STATE_SPACE_STRUCTURE_PANELS):
        ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
        counts = [_draw_separation_column(ax, position, runs, key)
                  for position, (_, _, runs) in zip(positions, columns)]
        if divider is not None:
            ax.axvline(divider, color="0.35", linestyle=":", linewidth=0.8, zorder=1)
        summaries.append(f"{key}: " + ", ".join(
            f"{name} {count}" for (_, name, _), count in zip(columns, counts)))
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{label}\n{count}" for (label, _, _), count
                            in zip(columns, counts)], fontsize=7)
        ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
        ax.set_xlabel(r"$L_2$ (runs with p < 0.05)", fontsize=8)
        ax.set_title(title, fontsize=9, pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    axs[0].set_ylabel("Category separation S", fontsize=8)
    band = mpl.patches.Patch(color="0.82", label="null 95th percentile (range over runs)")
    median = mpl.lines.Line2D([], [], color=_TREND_LINE_COLOR, linewidth=1.8,
                              label="median over runs")
    _state_space_figure_legend(fig, _significance_legend_handles() + [band, median])
    _save_fig(fig, _multitask_out(out_name), extra="  " + "; ".join(summaries))


def _untrained_structure_entries(combined, aname):
    """The structure summaries of the untrained networks built from run
    `aname`'s config (one per init seed), in name order; empty when the
    combined results hold none (or predate the untrained analysis)."""
    untrained = combined.get("untrained") or {}
    return [untrained[stem] for stem in sorted(untrained)
            if untrained[stem].get("config_source") == aname]


def plot_state_space_category_vs_untrained():
    """
    Figure: multitask_state_space_category_vs_untrained.png — the category
    separation S of untrained networks beside the trained runs of every L2
    cohort, for the hidden state and the effective modulation W * M.

    The left column holds the untrained networks of state_space_shift.py
    --untrained built from the selected L2 = STATE_SPACE_EXAMPLE_L2 example
    run's config, one point per init seed (fresh random weights, eta at its
    initial value, same evaluation trials); a dotted line and a wider gap set
    it apart from the four trained cohorts, which are drawn exactly as in
    plot_state_space_category_test. Every column shows one point per network,
    filled when its saved one-sided label-permutation p is below 0.05, the
    range of the networks' null 95th percentiles as a gray band, and the
    median as a dash; the tick label counts the significant networks. The
    untrained column is the baseline the rule-cue inputs give through random
    weights, so the trained cohorts' excess over it is what training adds.
    Read from the combined state_space_structure.py results (``untrained``
    and ``runs``); nothing is recomputed. Skipped, with the commands to run,
    when no untrained network of that config has been analyzed.
    """
    _ensure_out_dir()
    combined = _load_pkl_or_skip(STATE_SPACE_STRUCTURE_RESULT_PATH, _STATE_SPACE_STRUCTURE_HINT)
    if combined is None:
        return
    selected = _load_best_state_space_trial()
    if selected is None:
        return
    aname = selected[0]
    untrained = _untrained_structure_entries(combined, aname)
    if not untrained:
        print(f"  Skipped: the structure results hold no untrained network built from "
              f"{aname}'s config. Run: {_untrained_command(aname)}; then "
              "python multiple_task/state_space_structure.py")
        return
    grouped = _state_space_runs_by_l2(combined["runs"])
    columns = [("Untrained", "untrained", untrained)] + [
        (_l2_tick_label(strength), f"L2 {strength:.0e}", list(grouped[strength].values()))
        for strength in STATE_SPACE_L2_STRENGTHS]
    # The untrained column sits one and a half steps left of the first cohort.
    positions = np.array([0.0] + [1.5 + index for index in range(len(STATE_SPACE_L2_STRENGTHS))])
    _plot_separation_columns(columns, positions, figsize=(5.6, 2.4),
                             out_name="state_space_category_vs_untrained.png", divider=0.75)


def _state_space_figure_legend(fig, handles, ncol=None, top=0.92):
    """One frameless legend across the top of a two-panel figure (an axes
    legend would make tight_layout reserve its width for one panel), gated by
    SHOW_LEGEND like every legend: `ncol` columns (default: one row) and the
    figure laid out underneath, below the `top` fraction of its height."""
    if SHOW_LEGEND:
        fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles),
                   frameon=False, fontsize=6, columnspacing=1.0, handletextpad=0.4,
                   bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, top if SHOW_LEGEND else 1))


# ─── Figure: Over-membership ─────────────────────────────────────────────────

def _find_experiment_dirs():
    """Return all experiment subfolders under multiple_tasks_analysis/ matching
    the same feature/hidden/batch signature as ANAME (any seed), i.e. every
    analyzed run of the selected MULTITASK_L2 cohort."""
    import re as _re
    _check_same_cohort(MULTITASK_L2, multiple_tasks=ANAME)
    # ANAME = everything_seed{seed}_{feature}+hidden{h}+batch{b}+angle
    m = _re.match(r"everything_seed\d+_(.+)$", ANAME)
    suffix = m.group(1) if m else ""
    dirs = sorted(DATA_DIR.parent.glob(f"everything_seed*_{suffix}"))
    return [d for d in dirs if d.is_dir()]


def _plot_overmembership_single(pkl_template, out_filename):
    """
    Plot a 2×1 over-membership figure (top: same-neuron, bottom: same neuron-cluster),
    aggregated across all available experiments (seeds) that have the matching
    prepost_belonging pickle. Bars show the mean over-membership; error bars show
    the standard error across experiments, without individual seed points.
    The same-neuron Both slot is N/A:
    distinct modulation entries cannot share both neuron endpoints, making its
    observed and control counts zero and its relative over-membership undefined.

    pkl_template: a filename template containing "{aname}", e.g.
        "modulation_all_prepost_belonging_{aname}_unnormalized.pkl"
    One figure exists per modulation clustering variant of
    core/modulation_variants.py (normalized, unnormalized, weighted,
    var_weighted, abs_weighted); see the plot_overmembership_* wrappers below.
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
        # Upright labels: one word per bar on the neuron row, two lines on the
        # cluster row, so four 7 pt labels fit a 2.5 in panel without rotation.
        short_names = [n.replace("Share-", "").replace("-Cluster", "\ncluster") for n in bar_names]

        x = np.arange(len(mean))
        if row_idx == 0:
            short_names.insert(2, "Both")
            x[x >= 2] += 1
            ax.annotate("N/A", xy=(2, 0), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color="0.4")
        ax.bar(x, mean, yerr=sem, capsize=3, color="0.6",
               edgecolor="k", linewidth=0.5, width=0.6, zorder=2)
        ax.axhline(0, color="k", lw=0.5, zorder=0)
        ax.set_xticks(np.arange(len(short_names)))
        ax.set_xticklabels(short_names, rotation=0, ha="center", fontsize=7)
        ax.set_xlim(-0.6, len(short_names) - 0.4)
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


def plot_overmembership_abs_weighted():
    """Figure: Over-membership for |W|-weighted (abs_weighted) unnormalized modulation
    (G=100), aggregated across seeds. Fifth clustering variant of
    core/modulation_variants.py; runs analyzed before it existed have no pickle and
    are skipped."""
    _plot_overmembership_single(
        "modulation_all_abs_weighted_prepost_belonging_{aname}_unnormalized.pkl",
        "overmembership_abs_weighted.png",
    )


OVERMEMBERSHIP_EXAMPLE_FIXED_K = 20
# How many synapse (modulation) clusters plot_overmembership_examples draws,
# and which. With OVERMEMBERSHIP_EXAMPLE_CLUSTERS = None the examples are the
# OVERMEMBERSHIP_EXAMPLE_N most concentrated clusters of the resolved run,
# ranked by the highest observed/expected ratio over their well-supported
# blocks (expected count at least OVERMEMBERSHIP_EXAMPLE_SELECT_MIN_EXPECTED;
# ties to the larger expected count), so the choice follows MULTITASK_L2 and
# ANAME. A tuple of 1-based fixed-k cluster IDs forces the examples in that
# order; cluster numbers are not comparable across runs, so a forced tuple is
# meaningful for one run only and the figure skips when an ID is absent.
OVERMEMBERSHIP_EXAMPLE_N = 2
OVERMEMBERSHIP_EXAMPLE_CLUSTERS = None
# Blocks must expect at least this many synapses to rank the examples. The
# display threshold below is 1 synapse, where a ratio of 9 is nine synapses
# and mostly noise; at 5 expected a ratio of 6 is thirty synapses. Without
# this, the pick is the cluster with the luckiest near-empty block.
OVERMEMBERSHIP_EXAMPLE_SELECT_MIN_EXPECTED = 5.0
# Blocks expecting fewer synapses than this under the label-shuffle null are
# masked: their observed/expected ratio swings wildly with one or two synapses.
# Shared with the OM-vs-lesion scatter (OM_MIN_EXPECTED) so both figures judge
# block stability the same way.
OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED = OM_MIN_EXPECTED
# Same footprint as the lesion scatters, so the panels tile with them. The
# heatmap labels only the input and the hidden cluster of the deciding block:
# twenty labels per axis would be denser than the blocks they name, and the
# figure's point is which pathway the cluster occupies.
_OVERMEMBERSHIP_EXAMPLE_FIGSIZE = (2.0, 2.0)
# Dot-matrix companion: the largest observed count of any example fills this
# fraction of a cell's side (one scale for all examples); a measurable block
# with no synapse of the cluster is a small hollow grey ring, so it is told
# apart from an unmeasurable (blank) block.
_OVERMEMBERSHIP_DOT_MAX_FILL = 0.85
_OVERMEMBERSHIP_DOT_EMPTY_PT2 = 4.0
_OVERMEMBERSHIP_DOT_REFERENCE_COUNTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
# The dot matrix labels at most this many input and hidden clusters per axis:
# those holding the cluster's most significant enrichments, ranked by the
# Poisson z of their best shown block, (observed - expected) / sqrt(expected),
# which unlike the OM ratio is not inflated by a near-empty block. A label
# that would overlap a more significant neighbour's is dropped, never
# staggered onto a second row.
_OVERMEMBERSHIP_DOT_MAX_LABELS = 3
_OVERMEMBERSHIP_DOT_LEGEND_FIGSIZE = (0.8, 1.0)
# Synapse-map companion: every member synapse of the example cluster as one
# dot at its (input neuron, hidden neuron) position, neurons ordered by their
# fixed-k cluster (unresponsive class last, shaded), colored by the sign of
# the static weight W it modulates (a categorical distinction, SCHEME family 4;
# the var-weighted clustering is expected to keep one sign per cluster).
_OVERMEMBERSHIP_SYNAPSE_COLORS = {"W > 0": c_vals[1], "W < 0": c_vals[0]}
_OVERMEMBERSHIP_SYNAPSE_PT2 = 1.8
_OVERMEMBERSHIP_SYNAPSE_UNRESPONSIVE_SHADE = "0.94"
# The cluster-ordered synapse map leaves the unresponsive neurons out of both
# axes (their synapses, if any, are counted in the saved message): in the
# L2=1e-4 runs more than half of the hidden neurons are silent, and keeping
# them as a shaded band spends most of the panel on an empty region.
_OVERMEMBERSHIP_SYNAPSE_DROP_UNRESPONSIVE = True


def _ranked_om_clusters(ratios, expected, cluster_ids, skip_ids=()):
    """Candidate synapse clusters, most concentrated first.

    `ratios` and `expected` are (clusters, inputs, hiddens); a block counts
    when `expected` is finite there (callers pass masked arrays with
    unsupported blocks set to NaN). Each candidate is scored by the highest
    OM over its counted blocks, ties by that block's expected count. Clusters
    in `skip_ids` (the unresponsive class) or without a counted block are
    left out. Returns the indices in rank order."""
    scored = []
    for index, cluster_id in enumerate(cluster_ids):
        if cluster_id in skip_ids or not np.isfinite(expected[index]).any():
            continue
        om = np.where(np.isfinite(expected[index]), ratios[index], -np.inf)
        block = np.unravel_index(int(np.argmax(om)), om.shape)
        scored.append(((float(om[block]), float(expected[index][block])), index))
    return [index for _, index in sorted(scored, key=lambda item: item[0], reverse=True)]


def _om_significant_clusters(observed, expected, shown, max_labels=_OVERMEMBERSHIP_DOT_MAX_LABELS):
    """Indices of the input (axis 0) and hidden (axis 1) clusters holding the
    most significant enrichments of one synapse cluster: each cluster is
    scored by the Poisson z of its best shown block, only positive z counts
    (enrichment, not depletion), and at most `max_labels` are kept per side,
    most significant first. Returns (input indices, hidden indices)."""
    z = np.where(shown, (observed - expected) / np.sqrt(np.maximum(expected, 1e-12)), -np.inf)
    picks = []
    for axis in (1, 0):                      # best block per input row, per hidden column
        best = z.max(axis=axis)
        candidates = [int(index) for index in np.argsort(-best, kind="stable") if best[index] > 0]
        picks.append(candidates[:max_labels])
    return tuple(picks)


def _non_colliding_labels(indices, names, cell_pt, which, fontsize=7):
    """Keep, in significance order, each label whose text would not overlap
    an already kept one: x labels collide within one label width, y labels
    within one label height (0.65 em per character, 1.2 em line height, plus
    2 pt clearance), with one cell spanning `cell_pt` points. Returns the kept
    indices in axis order."""
    if not indices:
        return []
    if which == "x":
        extent_pt = 0.65 * fontsize * max(len(names[index]) for index in indices) + 2.0
    else:
        extent_pt = 1.2 * fontsize + 2.0
    kept = []
    for index in indices:
        if all(abs(index - other) * cell_pt >= extent_pt for other in kept):
            kept.append(index)
    return sorted(kept)


def _om_dot_cell_pt(n_in, n_hid, layout):
    """Side of one dot-matrix cell in points, from the figure size and the
    axes rectangle (`layout` has left/right/bottom/top fractions)."""
    width_in = _OVERMEMBERSHIP_EXAMPLE_FIGSIZE[0] * (layout["right"] - layout["left"])
    height_in = _OVERMEMBERSHIP_EXAMPLE_FIGSIZE[1] * (layout["top"] - layout["bottom"])
    return 72.0 * min(width_in / n_in, height_in / n_hid)


def _save_dot_size_key(out_path, references, max_area, max_observed, extra=""):
    """The dot matrices' size key as its own small figure: one grey disc per
    round reference count at the exact area the matrices use for it."""
    handles = [mpl.lines.Line2D([], [], marker="o", linestyle="none", color="0.5",
                                markersize=np.sqrt(max_area * count / max_observed),
                                label=f"{count:d}")
               for count in references]
    fig = plt.figure(figsize=_OVERMEMBERSHIP_DOT_LEGEND_FIGSIZE)
    fig.legend(handles=handles, loc="center", title="synapses", title_fontsize=6, fontsize=6,
               frameon=False, handletextpad=0.4, labelspacing=0.6)
    _save_fig(fig, out_path, extra=extra)


_OM_DOT_LAYOUT = dict(left=0.2, right=0.97, bottom=0.17, top=0.9)


def _fixed_k_neuron_labels(side, k=OVERMEMBERSHIP_EXAMPLE_FIXED_K):
    """Fixed-k cluster label of every neuron of the unnormalized `side`
    ("input" or "hidden") clustering, recut from the saved dendrogram by the
    shared `fixed_k_col_clusters` (labels 1..K, the unresponsive class last).
    Returns (labels, unresponsive label or None), or None when the cluster
    cache is missing."""
    try:
        info = _load_cluster_info()
    except FileNotFoundError:
        return None
    clusters, unresponsive = fixed_k_col_clusters(info[f"{side}_unnormalized"], k,
                                                  return_unresponsive_label=True)
    size = max(int(np.max(indices)) for indices in clusters.values() if len(indices)) + 1
    labels = np.zeros(size, dtype=int)
    for label, indices in clusters.items():
        labels[np.asarray(indices, dtype=int)] = label
    return labels, unresponsive


def _plastic_weights():
    """mp_layer1.W (post, pre) of ANAME's checkpoint, or None when absent."""
    path = Path("multiple_tasks") / f"savednet_{ANAME}.pt"
    if not path.exists():
        return None
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]["mp_layer1.W"].numpy()


def _synapse_cluster_members(entry, cluster_id, n_pre, expected_size):
    """(post, pre) neuron indices of the fixed-k synapse cluster `cluster_id`.

    The saved entry holds one clustering per synapse grouping G; the fixed-k
    assignment the examples are ranked on came from one of them, identified
    here as the grouping whose member count is closest to the assignment's
    own size (`expected_size`, cluster share x active synapses). Flat synapse
    indices are C-order over W's (post, pre). Returns (posts, pres, count
    mismatch) or None when no grouping saved fixed-k labels."""
    best = None
    for result in entry.get("result_all_lst", ()):
        labels = (result.get("col_labels_by_k") or {}).get(OVERMEMBERSHIP_EXAMPLE_FIXED_K)
        if labels is None:
            continue
        flat = np.nonzero(np.asarray(labels) == cluster_id)[0]
        mismatch = abs(flat.size - expected_size)
        if best is None or mismatch < best[0]:
            best = (mismatch, flat)
    if best is None:
        return None
    flat = best[1]
    return flat // n_pre, flat % n_pre, best[0]


def _neuron_axis_positions(labels, unresponsive):
    """Order neurons by fixed-k label with the unresponsive class last.
    Returns (position of each neuron, labels in axis order)."""
    labels = np.asarray(labels)
    key = labels.copy()
    if unresponsive is not None:
        key = np.where(labels == unresponsive, labels.max() + 1, labels)
    order = np.lexsort((np.arange(labels.size), key))
    position = np.empty(labels.size, dtype=int)
    position[order] = np.arange(labels.size)
    return position, labels[order]


def _plot_om_synapse_map(posts, pres, w_signs, input_axis, hidden_axis, labelled, title,
                         out_path, extra="", order_by_cluster=True,
                         drop_unresponsive=None):
    """One synapse cluster at neuron resolution: a dot per member synapse at
    (input neuron, hidden neuron), colored by the sign of its static weight.

    `input_axis` / `hidden_axis` are (position of each neuron, labels in axis
    order, unresponsive label) from _neuron_axis_positions: neurons are grouped
    by fixed-k cluster, boundaries drawn as thin lines and the unresponsive
    class shaded. `labelled` is (input cluster names, hidden cluster names),
    most significant first, to put on the axes at the center of each
    cluster's band (the clusters the dot matrix labels), plus U where it
    exists; a label whose text would overlap a higher-priority one is
    dropped. With `drop_unresponsive` (default
    _OVERMEMBERSHIP_SYNAPSE_DROP_UNRESPONSIVE) the unresponsive neurons are
    left out of both axes and member synapses touching them are not drawn
    (their number goes to the saved message); the axes then hold different
    neuron counts and are filled rather than kept square. With
    `order_by_cluster=False` the neurons keep their index order and the axes
    carry no cluster structure at all: no boundaries, no shading, no cluster
    labels, just the first and last neuron index. Same footprint as the
    other example panels; the color key is saved separately."""
    if drop_unresponsive is None:
        drop_unresponsive = _OVERMEMBERSHIP_SYNAPSE_DROP_UNRESPONSIVE
    fig, axis = plt.subplots(figsize=_OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
    fig.subplots_adjust(**_OM_DOT_LAYOUT)
    (in_pos, in_sorted, in_unres), (hid_pos, hid_sorted, hid_unres) = input_axis, hidden_axis
    n_dropped = 0
    if order_by_cluster and drop_unresponsive:
        # Keep responsive neurons only: positions are re-ranked among them,
        # sorted labels lose the U band, member synapses on U neurons go.
        keep_in = in_sorted[in_pos] != in_unres if in_unres is not None else np.ones(in_pos.size, bool)
        keep_hid = hid_sorted[hid_pos] != hid_unres if hid_unres is not None else np.ones(hid_pos.size, bool)
        member_ok = keep_in[pres] & keep_hid[posts]
        n_dropped = int((~member_ok).sum())
        posts, pres, w_signs = posts[member_ok], pres[member_ok], w_signs[member_ok]
        in_pos = np.argsort(np.argsort(in_pos[keep_in]))
        hid_pos = np.argsort(np.argsort(hid_pos[keep_hid]))
        in_sorted = in_sorted[in_sorted != in_unres] if in_unres is not None else in_sorted
        hid_sorted = hid_sorted[hid_sorted != hid_unres] if hid_unres is not None else hid_sorted
        in_unres = hid_unres = None
        # Index arrays must address the reduced neuron sets.
        pres = np.searchsorted(np.nonzero(keep_in)[0], pres)
        posts = np.searchsorted(np.nonzero(keep_hid)[0], posts)
    n_in, n_hid = in_pos.size, hid_pos.size
    if not order_by_cluster:
        in_pos, hid_pos = np.arange(n_in), np.arange(n_hid)
    for which, sorted_labels, unresponsive in (
            (("x", in_sorted, in_unres), ("y", hid_sorted, hid_unres)) if order_by_cluster else ()):
        boundaries = np.nonzero(np.diff(sorted_labels))[0] + 1
        for boundary in boundaries:
            (axis.axvline if which == "x" else axis.axhline)(boundary, color="0.85", lw=0.4, zorder=1)
        if unresponsive is not None and np.any(sorted_labels == unresponsive):
            start = int(np.argmax(sorted_labels == unresponsive))
            span = int(np.sum(sorted_labels == unresponsive))
            if which == "x":
                axis.axvspan(start, start + span, color=_OVERMEMBERSHIP_SYNAPSE_UNRESPONSIVE_SHADE,
                             lw=0, zorder=0)
            else:
                axis.axhspan(start, start + span, color=_OVERMEMBERSHIP_SYNAPSE_UNRESPONSIVE_SHADE,
                             lw=0, zorder=0)
    for name, color in _OVERMEMBERSHIP_SYNAPSE_COLORS.items():
        pick = (w_signs > 0) if name == "W > 0" else (w_signs < 0)
        if pick.any():
            axis.scatter(in_pos[pres[pick]] + 0.5, hid_pos[posts[pick]] + 0.5,
                         s=_OVERMEMBERSHIP_SYNAPSE_PT2, color=color, edgecolors="none",
                         zorder=3, label=name)
    axis.set_xlim(0, n_in)
    axis.set_ylim(n_hid, 0)
    axis.set_aspect("equal" if n_in == n_hid else "auto")
    neuron_pt = _om_dot_cell_pt(n_in, n_hid, _OM_DOT_LAYOUT)      # points per neuron
    if not order_by_cluster:
        axis.set_xticks([0.5, n_in - 0.5])
        axis.set_xticklabels(["1", str(n_in)])
        axis.set_yticks([0.5, n_hid - 0.5])
        axis.set_yticklabels(["1", str(n_hid)])
    for which, setter_ticks, setter_labels, sorted_labels, unresponsive, wanted in (
            (("x", axis.set_xticks, axis.set_xticklabels, in_sorted, in_unres, labelled[0]),
             ("y", axis.set_yticks, axis.set_yticklabels, hid_sorted, hid_unres, labelled[1]))
            if order_by_cluster else ()):
        # Candidate ticks in priority order: the dot matrix's labels, then U.
        candidates = []
        for name in wanted:
            members = np.nonzero(sorted_labels == int(name[1:]))[0]
            if members.size:
                candidates.append((float(members.mean()) + 0.5, name))
        if unresponsive is not None and np.any(sorted_labels == unresponsive):
            candidates.append((float(np.nonzero(sorted_labels == unresponsive)[0].mean()) + 0.5, "U"))
        extent_pt = (0.65 * 7 * max((len(name) for _, name in candidates), default=1) + 2.0
                     if which == "x" else 1.2 * 7 + 2.0)
        kept = []
        for tick, name in candidates:
            if all(abs(tick - other) * neuron_pt >= extent_pt for other, _ in kept):
                kept.append((tick, name))
        kept.sort()
        setter_ticks([tick for tick, _ in kept])
        setter_labels([name for _, name in kept])
    axis.tick_params(axis="both", length=2, labelsize=7, labelrotation=0)
    axis.set_title(title, fontsize=8)
    axis.set_xlabel("Input neuron", fontsize=8)
    axis.set_ylabel("Hidden neuron", fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    if order_by_cluster and drop_unresponsive:
        extra += (f" [unresponsive neurons left out: {n_in} input x {n_hid} hidden shown, "
                  f"{n_dropped} member synapses on them not drawn]")
    _save_fig(fig, out_path, extra=extra)


def _save_synapse_sign_key(out_path):
    """The synapse maps' color key as its own small figure."""
    handles = [mpl.lines.Line2D([], [], marker="o", linestyle="none", color=color, markersize=3,
                                label=name) for name, color in _OVERMEMBERSHIP_SYNAPSE_COLORS.items()]
    fig = plt.figure(figsize=_OVERMEMBERSHIP_DOT_LEGEND_FIGSIZE)
    fig.legend(handles=handles, loc="center", title="synapse", title_fontsize=6, fontsize=6,
               frameon=False, handletextpad=0.4, labelspacing=0.6)
    _save_fig(fig, out_path, extra="  (color key of the synapse maps: sign of W)")


def _plot_om_dot_matrix(ratio, shown, observed, input_names, hidden_names, vmax, max_observed,
                        labelled, title, out_path, extra=""):
    """The dot-matrix view of one synapse cluster's block map.

    `ratio`, `shown` and `observed` are (inputs, hiddens) over the kept
    clusters; x is the input (pre) cluster, y the hidden (post) cluster, as in
    the heatmap. Each shown block is a disc centered on its cell whose area is
    proportional to `observed` (`max_observed`, shared by all examples, fills
    _OVERMEMBERSHIP_DOT_MAX_FILL of the cell side) and whose color is OM on
    the shared viridis scale [0, vmax]; shown blocks with no synapse are small
    hollow grey rings; unshown blocks are blank. `labelled` is (input indices,
    hidden indices) in significance order; those whose names would overlap a
    more significant neighbour's are dropped (_non_colliding_labels). The
    size key is saved separately (_save_dot_size_key)."""
    n_in, n_hid = ratio.shape
    fig, axis = plt.subplots(figsize=_OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
    fig.subplots_adjust(**_OM_DOT_LAYOUT)
    cell_pt = _om_dot_cell_pt(n_in, n_hid, _OM_DOT_LAYOUT)
    max_area = (_OVERMEMBERSHIP_DOT_MAX_FILL * cell_pt) ** 2
    cmap = plt.get_cmap(_MULTITASK_HEATMAP_CMAP)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    rows, cols = np.nonzero(shown)
    filled = observed[rows, cols] > 0
    if filled.any():
        sizes = max_area * observed[rows, cols][filled] / max(max_observed, 1e-12)
        axis.scatter(rows[filled] + 0.5, cols[filled] + 0.5, s=sizes,
                     color=cmap(norm(ratio[rows, cols][filled])), edgecolors="none", zorder=3)
    if (~filled).any():
        axis.scatter(rows[~filled] + 0.5, cols[~filled] + 0.5, s=_OVERMEMBERSHIP_DOT_EMPTY_PT2,
                     facecolors="none", edgecolors="0.6", linewidths=0.5, zorder=3)
    axis.set_xlim(0, n_in)
    axis.set_ylim(n_hid, 0)
    axis.set_aspect("equal")
    kept = {}
    for which, setter_ticks, setter_labels, names, wanted in (
            ("x", axis.set_xticks, axis.set_xticklabels, input_names, labelled[0]),
            ("y", axis.set_yticks, axis.set_yticklabels, hidden_names, labelled[1])):
        kept[which] = _non_colliding_labels(wanted, names, cell_pt, which)
        setter_ticks([index + 0.5 for index in kept[which]])
        setter_labels([names[index] for index in kept[which]])
    axis.tick_params(axis="both", length=2, labelsize=7, labelrotation=0)
    axis.set_title(title, fontsize=8)
    axis.set_xlabel("Input cluster", fontsize=8)
    axis.set_ylabel("Hidden cluster", fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    # Kept labels, most significant first, for the synapse map's tick choice.
    kept_names = ([input_names[i] for i in labelled[0] if i in kept["x"]],
                  [hidden_names[j] for j in labelled[1] if j in kept["y"]])
    labelled_text = (f"; labelled: input {', '.join(input_names[i] for i in kept['x']) or 'none'}, "
                     f"hidden {', '.join(hidden_names[j] for j in kept['y']) or 'none'}")
    _save_fig(fig, out_path, extra=extra + labelled_text)
    return max_area, kept_names


def plot_overmembership_examples():
    """Show the most concentrated synapse clusters' input-by-hidden block
    enrichment, one heatmap and one dot matrix per example.

    Reads global_assignment_fixed_k20: input, hidden and modulation use their
    fixed-k unnormalized clusterings, with variance weighting for modulation.
    Extra endpoint group k+1, when present, is the unresponsive class (U).
    OM is observed/expected, with baseline 1, not pooled G=100 pair statistics.
    The null shuffles labels among surviving synapses, so expected counts are
    n_active_block * cluster_size_percent.

    The examples are OVERMEMBERSHIP_EXAMPLE_CLUSTERS, or with None the
    OVERMEMBERSHIP_EXAMPLE_N most concentrated clusters (_ranked_om_clusters),
    so they are the clearest cases of a synapse cluster living in one
    neuron-cluster block rather than hand-picked IDs that go stale when the
    run changes; the saved messages name the choices and the runners-up.
    Blocks whose expected count is below OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED
    (the OM_MIN_EXPECTED threshold shared with the OM-vs-lesion scatter) are
    masked and hatched, because a ratio over a handful of expected synapses is
    unstable; zero-expectation blocks are a special case of this. Input or
    hidden clusters with no shown block are dropped from an example's axes,
    which is what removes U and very small neuron clusters. Cells carry no
    numbers and no block is singled out; each heatmap axis labels only the
    cluster of the deciding block (the peak OM over well-supported blocks),
    and the fixed k, the masking rule and the dropped clusters go to the
    saved message rather than onto the figure.

    Outputs, rank r = 1..N: `overmembership_examples_<r>` (heatmap),
    `overmembership_examples_synapses_<r>` (the cluster at neuron
    resolution: one dot per member synapse at its input x hidden neuron
    position, neurons ordered by fixed-k cluster with the unresponsive class
    shaded, dot color the sign of the static weight W; skipped with a note
    when the checkpoint or the per-synapse labels are unavailable; color key
    saved once as `overmembership_examples_synapses_legend`), its
    `overmembership_examples_synapses_raw_<r>` twin with the neurons in
    their index order and no cluster structure on the axes, and
    `overmembership_examples_dots_<r>` (dot matrix: dot area proportional to
    the observed number of the cluster's synapses, color still OM, so a ratio
    over two synapses is a speck and one over thirty a disc; measurable
    blocks with no synapse are small hollow rings, unmeasurable blocks blank;
    up to _OVERMEMBERSHIP_DOT_MAX_LABELS input and hidden clusters labelled,
    those with the most significant enrichment, _om_significant_clusters,
    dropping a label that would overlap a more significant one). All examples
    share one color scale and one dot-area scale, saved once as
    `overmembership_examples_colorbar` (viridis, OM = 1 marked) and
    `overmembership_examples_dots_legend` (size key).
    The full cluster cache can be several GB; run on an allocated compute node.
    """
    mod_info = _load_cluster_info_mod()
    if mod_info is None:
        print("  Skipped OM examples: modulation cluster cache not found.")
        return
    entry = mod_info["modulation_all_var_weighted_unnormalized"]
    assignment_key = f"global_assignment_fixed_k{OVERMEMBERSHIP_EXAMPLE_FIXED_K}"
    assignment = entry.get(assignment_key)
    if (assignment is None or "n_active_block" not in assignment
            or assignment.get("fixed_k") != OVERMEMBERSHIP_EXAMPLE_FIXED_K):
        print(f"  Skipped OM examples: regenerate {assignment_key} with active-block counts.")
        return
    ratios = np.asarray(assignment["om_stack"], dtype=float)
    cluster_ids = list(assignment["all_choice_order"])
    fractions = np.asarray(assignment["cluster_size_percent"], dtype=float)
    active = np.asarray(assignment["n_active_block"], dtype=float)
    if (ratios.ndim != 3 or ratios.shape != (len(cluster_ids), *active.shape)
            or active.ndim != 2 or fractions.shape != (len(cluster_ids),)
            or len(set(cluster_ids)) != len(cluster_ids)
            or not np.isfinite(ratios).all() or np.any(ratios < 0)
            or not np.isfinite(active).all() or np.any(active < 0)
            or not np.isfinite(fractions).all() or np.any(fractions <= 0)
            or not np.isclose(fractions.sum(), 1.0)):
        raise ValueError("Invalid OM example cache: counts, ratios and cluster IDs must align.")
    expected = active[None, :, :] * fractions[:, None, None]
    stable = expected >= OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED
    select_min = max(OVERMEMBERSHIP_EXAMPLE_SELECT_MIN_EXPECTED, OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED)
    supported = expected >= select_min
    unresponsive = assignment.get("unresponsive_mod_label")
    skip_ids = () if unresponsive is None else (unresponsive,)
    n_post, n_pre = (int(value) for value in entry["mod_shape"]) if "mod_shape" in entry else (None, None)
    weights = _plastic_weights()
    neuron_axes = None
    if weights is not None and n_pre is not None and weights.shape == (n_post, n_pre):
        sides = {side: _fixed_k_neuron_labels(side) for side in ("input", "hidden")}
        if all(value is not None and value[0].size == size
               for value, size in zip(sides.values(), (n_pre, n_post))):
            neuron_axes = {side: (*_neuron_axis_positions(*value), value[1])
                           for side, value in sides.items()}
    if neuron_axes is None:
        print("  Note: synapse maps skipped (checkpoint, cluster cache or synapse shape unavailable).")
    del mod_info, assignment

    ranked = _ranked_om_clusters(ratios, np.where(supported, expected, np.nan), cluster_ids, skip_ids)
    if OVERMEMBERSHIP_EXAMPLE_CLUSTERS is None:
        chosen = ranked[:OVERMEMBERSHIP_EXAMPLE_N]
        if not chosen:
            print(f"  Skipped OM examples: no synapse cluster has a block with expected >= "
                  f"{select_min:g} synapses in {ANAME}.")
            return
    else:
        chosen = []
        for cluster_id in OVERMEMBERSHIP_EXAMPLE_CLUSTERS:
            if cluster_id not in cluster_ids:
                print(f"  Skipped OM examples: synapse cluster C{cluster_id} not found in {ANAME}.")
                return
            index = cluster_ids.index(cluster_id)
            if not stable[index].any():
                print(f"  Skipped OM examples: C{cluster_id} has no block with expected >= "
                      f"{OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED:g} synapses.")
                return
            chosen.append(index)
    # Peak OM over well-supported blocks of every ranked candidate, for the
    # saved messages (runners-up after the chosen examples).
    peaks = [(float(np.max(np.where(supported[index], ratios[index], -np.inf))), cluster_ids[index])
             for index in ranked]
    runners = ", ".join([f"C{cid} {peak:.1f}" for peak, cid in peaks
                         if cid not in {cluster_ids[index] for index in chosen}][:3])

    def _labels(indices):
        return ["U" if index == OVERMEMBERSHIP_EXAMPLE_FIXED_K else f"C{index + 1}"
                for index in indices]

    _ensure_out_dir()
    # One color scale and one dot-area scale across the examples.
    vmax = max(1.0, float(np.ceil(max(ratios[index][stable[index]].max() for index in chosen))))
    max_observed = max(float((ratios[index] * expected[index])[stable[index]].max()) for index in chosen)
    max_area = None
    synapse_maps_drawn = False
    for rank, cluster_index in enumerate(chosen, start=1):
        cluster_id = cluster_ids[cluster_index]
        ratio, shown = ratios[cluster_index], stable[cluster_index]
        keep_input = np.nonzero(shown.any(axis=1))[0]
        keep_hidden = np.nonzero(shown.any(axis=0))[0]
        dropped = {
            "input": _labels([i for i in range(active.shape[0]) if i not in keep_input]),
            "hidden": _labels([j for j in range(active.shape[1]) if j not in keep_hidden]),
        }
        # The deciding block: peak OM over well-supported blocks (over shown
        # blocks for a forced cluster without one). Its two clusters are the
        # heatmap's only axis labels, and it is named in the saved message.
        top = np.unravel_index(int(np.argmax(np.where(supported[cluster_index], ratio, -np.inf)))
                               if supported[cluster_index].any() else
                               int(np.argmax(np.where(shown, ratio, -np.inf))), ratio.shape)
        labelled = ({_labels([top[0]])[0]}, {_labels([top[1]])[0]})
        fig, axis = plt.subplots(figsize=_OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
        fig.subplots_adjust(**_OM_DOT_LAYOUT)
        sub_ratio = ratio[np.ix_(keep_input, keep_hidden)]
        sub_shown = shown[np.ix_(keep_input, keep_hidden)]
        axis.set_facecolor("white")
        sns.heatmap(
            sub_ratio.T, mask=~sub_shown.T, ax=axis, cmap=_MULTITASK_HEATMAP_CMAP,
            vmin=0, vmax=vmax,
            annot=False, linewidths=0.5, linecolor="white", cbar=False, square=True,
            xticklabels=_labels(keep_input), yticklabels=_labels(keep_hidden),
        )
        # Masked blocks are hatched in light grey, a tone outside the viridis
        # ramp, so they cannot be mistaken for a low OM.
        for input_pos_masked, hidden_pos_masked in np.argwhere(~sub_shown):
            axis.add_patch(mpl.patches.Rectangle(
                (input_pos_masked, hidden_pos_masked), 1, 1, facecolor="0.94",
                edgecolor="0.75", hatch="////", linewidth=0, zorder=1))
        axis.set_title(f"Synapse cluster C{cluster_id}", fontsize=8)
        axis.set_xlabel("Input cluster", fontsize=8)
        axis.set_ylabel("Hidden cluster", fontsize=8)
        input_names, hidden_names = _labels(keep_input), _labels(keep_hidden)
        for setter_ticks, setter_labels, names, wanted in (
                (axis.set_xticks, axis.set_xticklabels, input_names, labelled[0]),
                (axis.set_yticks, axis.set_yticklabels, hidden_names, labelled[1])):
            positions = [index + 0.5 for index, name in enumerate(names) if name in wanted]
            setter_ticks(positions)
            setter_labels([name for name in names if name in wanted])
        axis.tick_params(axis="both", length=2, labelsize=7, labelrotation=0)
        observed_top = ratio[top] * expected[cluster_index][top]
        dropped_text = "; ".join(f"{side} {', '.join(names)}" for side, names in dropped.items() if names)
        _save_fig(fig, _multitask_out(f"overmembership_examples_{rank}.png"),
                  extra=(f"  ({ANAME}; fixed k={OVERMEMBERSHIP_EXAMPLE_FIXED_K}; "
                         f"hatched: expected < {OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED:g} synapses; "
                         f"var-weighted unnormalized modulation; unnormalized input/hidden) "
                         f"example {rank}: synapse cluster C{cluster_id}: {int(sub_shown.sum())}/"
                         f"{sub_shown.size} blocks shown; peak OM {ratio[top]:.2f} at input "
                         f"{_labels([top[0]])[0]} / hidden {_labels([top[1]])[0]} ({observed_top:.0f} "
                         f"observed / {expected[cluster_index][top]:.2f} expected; blocks with "
                         f"expected >= {select_min:g} rank the examples)"
                         + (f"; runners-up: {runners}" if runners else "")
                         + (f"; dropped (no shown block): {dropped_text}" if dropped_text else "")))
        sub_expected = expected[cluster_index][np.ix_(keep_input, keep_hidden)]
        sub_observed = sub_ratio * sub_expected
        max_area, kept_names = _plot_om_dot_matrix(
            sub_ratio, sub_shown, sub_observed, input_names, hidden_names, vmax, max_observed,
            _om_significant_clusters(sub_observed, sub_expected, sub_shown),
            f"Synapse cluster C{cluster_id}", _multitask_out(f"overmembership_examples_dots_{rank}.png"),
            extra=f"  ({ANAME}; example {rank}; dot area = observed synapses of C{cluster_id}; color = OM")
        if neuron_axes is not None:
            members = _synapse_cluster_members(entry, cluster_id, n_pre,
                                               fractions[cluster_index] * active.sum())
            if members is None:
                print(f"  Note: synapse map of C{cluster_id} skipped (no fixed-k synapse labels saved).")
            else:
                posts, pres, mismatch = members
                w_signs = np.sign(weights[posts, pres])
                synapse_extra = (
                    f"  ({ANAME}; example {rank}; {posts.size} synapses of C{cluster_id}, "
                    f"{int((w_signs > 0).sum())} with W > 0, {int((w_signs < 0).sum())} with W < 0; "
                    f"{np.unique(pres).size} input and {np.unique(posts).size} hidden neurons; "
                    f"member count differs from the fixed-k assignment by {mismatch:.0f})")
                _plot_om_synapse_map(
                    posts, pres, w_signs, neuron_axes["input"], neuron_axes["hidden"], kept_names,
                    f"Synapse cluster C{cluster_id}",
                    _multitask_out(f"overmembership_examples_synapses_{rank}.png"), extra=synapse_extra)
                _plot_om_synapse_map(
                    posts, pres, w_signs, neuron_axes["input"], neuron_axes["hidden"], kept_names,
                    f"Synapse cluster C{cluster_id}",
                    _multitask_out(f"overmembership_examples_synapses_raw_{rank}.png"),
                    extra=synapse_extra + " [neurons in index order]", order_by_cluster=False)
                synapse_maps_drawn = True
    if neuron_axes is not None and synapse_maps_drawn:
        _save_synapse_sign_key(_multitask_out("overmembership_examples_synapses_legend.png"))
    references = [count for count in _OVERMEMBERSHIP_DOT_REFERENCE_COUNTS if count <= max_observed][-3:]
    if references:
        _save_dot_size_key(_multitask_out("overmembership_examples_dots_legend.png"), references,
                           max_area, max_observed,
                           extra=f"  (size key shared by the {len(chosen)} dot matrices: {references})")
    ticks = sorted({0.0, 1.0, vmax / 2, vmax})
    _save_standalone_colorbar(
        _multitask_out("overmembership_examples_colorbar.png"),
        cmap=_MULTITASK_HEATMAP_CMAP, vmin=0, vmax=vmax, ticks=ticks,
        ticklabels=[f"{tick:g}" for tick in ticks], label="OM (observed / expected)",
        **_VERTICAL_COLORBAR_STYLE)


# ─── Figure: Lesion heatmap ──────────────────────────────────────────────────
# LESION_DIR (multiple_tasks_perf/{LESION_ANAME}) is set by _set_multitask_l2.

# Synapse intervention drawn in the lesion heatmap, and read by its cluster-size
# companion, every modulation tuning-vs-lesion figure and the modulation side
# of causal-vs-activity. freeze_M keeps W and holds the selected synapses'
# modulation at its initial value, so the panel measures what their
# plasticity contributes. zero_W (W <- 0) stays in the same caches; the OM
# scatter still reads it (OM_LESION_MODE). Both modes lesion the same synapse
# clusters.
LESION_HEATMAP_MODE = "freeze_M"
_LESION_HEATMAP_MOD_KEY = f"modulation_all_var_weighted_unnormalized__{LESION_HEATMAP_MODE}"


def _load_lesion_results():
    """Load the lesion/prune results pickle."""
    pkl_path = LESION_DIR / f"lesion_prune_results_{LESION_ANAME}.pkl"
    if not pkl_path.exists():
        return None
    return load_lesion_pickle(pkl_path)


def plot_lesion_heatmap():
    """Plot cached random-control-adjusted effects for unnormalized clusters.

    Two task-by-cluster panels: hidden neuron clusters (post, from
    lesion_unnorm) above modulation clusters from
    modulation_all_var_weighted_unnormalized__freeze_M. The input (pre) effects
    saved in the same cache are not drawn. The modulation panel freezes the
    selected synapses' M at its initial value and keeps W
    (LESION_HEATMAP_MODE); a cache without freeze_M effects is skipped, never
    replaced by zero_W.

    lesion_plot.py computes random_acc - cluster_acc. Positive values mean the
    targeted lesion impairs accuracy more than the size-matched random control.
    This renderer selects columns, aligns task rows and multiplies by 100 to
    show accuracy percentage points; it does not recompute effects or tests.
    Each panel retains its own cluster identities. C21 is displayed as U for
    the fixed-k=20 results. Both share a symmetric color scale, with the
    colorbar saved separately.
    """
    _ensure_out_dir()
    path = LESION_NORM_DIR / f"normalized_lesion_effects_{LESION_ANAME}.pkl"
    data = _load_pkl_or_skip(path, "Run multiple_task/lesion_plot.py first.")
    if data is None:
        return

    try:
        data = normalize_lesion_cache(data)
        if data["schema_version"] != 1 or data["aname"] != LESION_ANAME:
            raise ValueError("normalized-effect cache version/run does not match")
        hidden = data["entries"]["lesion_unnorm"]
        modulation = data["entries"][_LESION_HEATMAP_MOD_KEY]
        if modulation.get("mod_lesion_mode", LESION_HEATMAP_MODE) != LESION_HEATMAP_MODE:
            raise ValueError(f"modulation heatmap requires {LESION_HEATMAP_MODE} effects")
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
            raise ValueError("missing saved hidden or modulation cluster effects")
        mod_rows = [modulation["tasks"].index(task) for task in all_tasks]
        effect_post = np.asarray(hidden["effect"], dtype=float)[:, post_idx] * 100
        effect_mod = np.asarray(modulation["effect"], dtype=float)[np.ix_(mod_rows, mod_idx)] * 100
        if not all(np.isfinite(effect).all() for effect in (effect_post, effect_mod)):
            raise ValueError("non-finite saved effect values")
        post_labels = ["C" + hidden["conditions"][index].removeprefix("post_c") for index in post_idx]
        mod_labels = ["C" + modulation["conditions"][index].removeprefix("mod_c") for index in mod_idx]
    except (KeyError, TypeError, ValueError) as error:
        print(f"  Skipped: incompatible {path.name} ({error}). Run multiple_task/lesion_plot.py again.")
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
        cluster_labels = ["U" if label == "C21" else label for label in cluster_labels]
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
        if idx < len(panels) - 1:
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
    _save_fig(fig, out_path, extra=f"  (modulation lesion: {LESION_HEATMAP_MODE})")
    _save_standalone_colorbar(
        _multitask_out("lesion_heatmap_unnorm_colorbar.png"),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        ticks=_ticks,
        ticklabels=[f"{t:.0f}" for t in _ticks],
        label="Normalized effect (%)",
        **_VERTICAL_COLORBAR_STYLE,
    )


def plot_lesion_cluster_sizes():
    """
    Figure: relative sizes of the hidden and modulation clusters in the lesion
    heatmap (plot_lesion_heatmap), as two bar panels retaining their column order:

        Top    — hidden (post) neuron clusters from the unnormalized
                 clustering: % of all hidden neurons per cluster
        Bottom — var-weighted-unnormalized modulation (synapse) clusters from
                 the freeze_M result (LESION_HEATMAP_MODE), matching the
                 heatmap: % of all clustered synapses per cluster

    Read hidden counts from saved lesion_units and compute modulation counts
    from the saved memberships of that same entry. Percentages are calculated
    here using each panel's summed counts; no lesion effects or significance
    tests are fitted.
    The input panel in the main heatmap has no size panel here. Numberings are
    independent and displayed as C1..Cn; unlike the heatmap, C21 is not relabeled
    U in this figure. Percentages use log y axes; zero-size clusters are labeled
    0% without a bar. Sizes provide context for interpreting lesion effects.
    """
    _ensure_out_dir()
    data = _load_lesion_results()
    if data is None:
        print("  Skipped: lesion results not found. Run lesion.py first.")
        return

    # Top: hidden (post) cluster sizes, in the heatmap's column order
    lu = data["lesion_unnorm"].get("lesion_units", {})
    comb_names = data["lesion_unnorm"]["all_comb_names_lesion"]
    post_names = [n for n in comb_names if n.startswith("post_c")]
    if not post_names or any(n not in lu for n in post_names):
        print("  Skipped: lesion_units missing for hidden clusters.")
        return
    hid_sizes = np.array([lu[n] for n in post_names], float)

    # Bottom: var-weighted modulation cluster sizes (sorted ids = heatmap order)
    mod_entry = data["mod_lesion"].get(_LESION_HEATMAP_MOD_KEY)
    if mod_entry is None:
        print(f"  Skipped: var-weighted {LESION_HEATMAP_MODE} lesion entry not found.")
        return
    if mod_entry.get("mod_lesion_mode", LESION_HEATMAP_MODE) != LESION_HEATMAP_MODE:
        print(f"  Skipped: modulation cluster sizes require a {LESION_HEATMAP_MODE} lesion entry.")
        return
    col_clusters = mod_entry["mod_col_clusters"]
    mod_sizes = np.array([len(col_clusters[c]) for c in sorted(col_clusters)],
                         float)

    panels = [
        (hid_sizes, f"Hidden neuron clusters (n = {int(hid_sizes.sum())} neurons)"),
        (mod_sizes, f"Modulation clusters (n = {int(mod_sizes.sum())} synapses)"),
    ]
    if any(not np.isfinite(sizes).all() or np.any(sizes < 0) or sizes.sum() <= 0
           for sizes, _ in panels):
        print("  Skipped: cluster sizes must be nonnegative with positive totals.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(6, 3.6), sharex=False)
    for ax, (sizes, title) in zip(axes, panels):
        pct = sizes / sizes.sum() * 100
        xs = np.arange(len(pct))
        positive = pct > 0
        ax.bar(xs[positive], pct[positive], color="#4682b4", edgecolor="k",
               linewidth=0.4, width=0.7)
        ax.set_yscale("log")
        ax.set_ylim(pct[positive].min() * 0.5, pct[positive].max() * 1.4)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
        for index in xs[~positive]:
            ax.text(index, 0.04, "0%", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"C{i + 1}" for i in xs], fontsize=6)
        ax.set_ylabel("Cluster size (%)\n(log scale)", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[1].set_xlabel("Cluster", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, _multitask_out("lesion_cluster_sizes.png"))


# ─── Figure: OM vs lesion ────────────────────────────────────────────────────

def _saved_om_rank_association(entry):
    """Validate cached rank/permutation identity without recomputing statistics."""
    association = entry["association"]
    if (association["statistic"] != "spearman" or association["side"] != "less"
            or association["permutation_unit"] != "modulation_cluster_footprint"):
        raise ValueError("OM scatter requires a negative-sided Spearman cluster permutation")
    rho, probability = float(association["rho"]), float(association["p_perm"])
    n_perm, n_clusters = int(association["n_perm"]), int(association["n_clusters"])
    if (n_perm < 1 or n_clusters < 1
            or not (np.isnan(rho) or np.isfinite(rho) and -1 <= rho <= 1)
            or not (np.isnan(probability) or np.isfinite(probability) and 0 <= probability <= 1)
            or np.isfinite(probability) and (not np.isfinite(rho) or n_clusters < 2)):
        raise ValueError("Invalid cached Spearman correlation or permutation metadata")
    return {"rho": rho, "p_perm": probability, "n_perm": n_perm, "n_clusters": n_clusters}


# Intervention shown in the OM-vs-lesion scatter: zero_W (selected synaptic
# weights set to zero). The other synapse-lesion figures follow
# LESION_HEATMAP_MODE (freeze_M) instead; both modes stay in the same caches.
OM_LESION_MODE = "zero_W"


def _load_om_lesion_scatter(run_dir=None, aname=None):
    """Load the saved OM_LESION_MODE scatter, Spearman test and bin medians,
    without fitting, of run `aname` from its post-processing directory
    `run_dir` (default: LESION_ANAME in LESION_NORM_DIR)."""
    run_dir = LESION_NORM_DIR if run_dir is None else Path(run_dir)
    aname = LESION_ANAME if aname is None else aname
    tag = "var-weighted-unnormalized"
    mode_tag = OM_LESION_MODE.replace("_", "-")
    candidates = ((f"om_vs_lesion_diff_{tag}_combined_unnorm_{aname}.pkl", True),
                  (f"om_vs_lesion_diff_{tag}_{mode_tag}_unnorm_{aname}.pkl", False))
    for filename, combined in candidates:
        path = resolve_lesion_cache_path(run_dir / filename)
        if not path.exists():
            continue
        try:
            saved = load_lesion_pickle(path)
            base_key = saved["base_key"] if combined else saved["mod_type_key"]
            if (base_key != "modulation_all_var_weighted_unnormalized"
                    or saved["variant"] != "unnorm" or saved.get("aname", aname) != aname):
                raise ValueError("OM cache belongs to a different run or variant")
            if saved.get("y_definition") != "task-profile L1/T: mean_t |mod_effect(t) - combined_effect(t)|":
                raise ValueError("OM cache does not describe task-profile L1/T distance")
            if saved.get("schema_version") != 2:
                raise ValueError("Legacy OM cache; rerun lesion_plot.py for Spearman and bin medians")
            if combined:
                entry = saved["mode_data"][OM_LESION_MODE]
            else:
                if saved["mod_lesion_mode"] != OM_LESION_MODE:
                    raise ValueError("OM cache uses a different lesion mode")
                entry = saved
            association = _saved_om_rank_association(entry)
            om_vals = np.asarray(entry["om_vals"], dtype=float)
            lesion_diffs = np.asarray(entry["lesion_diffs"], dtype=float)
            if (om_vals.ndim != 1 or om_vals.shape != lesion_diffs.shape or om_vals.size < 2
                    or not np.isfinite(om_vals).all() or not np.isfinite(lesion_diffs).all()
                    or np.any(om_vals < 0) or np.any(lesion_diffs < 0)):
                raise ValueError("invalid saved OM scatter coordinates")
            medians = entry["binned_medians"]
            bin_x, bin_y = np.asarray(medians["x"], float), np.asarray(medians["y"], float)
            counts = np.asarray(medians["counts"])
            if (medians["method"] != "quantile" or bin_x.ndim != 1 or bin_x.size == 0
                    or bin_y.shape != bin_x.shape or counts.shape != bin_x.shape
                    or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0)
                    or counts.sum() != om_vals.size
                    or not np.isfinite(bin_x).all() or not np.isfinite(bin_y).all()
                    or np.any(bin_y < 0) or np.any(np.diff(bin_x) <= 0)
                    or np.any(bin_x < om_vals.min()) or np.any(bin_x > om_vals.max())):
                raise ValueError("Invalid saved quantile-bin medians")
            min_expected = float(saved["min_expected"])
            if not np.isfinite(min_expected) or min_expected < 0:
                raise ValueError("invalid saved OM threshold")
            return {"om_vals": om_vals, "lesion_diffs": lesion_diffs, **association,
                    "binned_medians": {"x": bin_x, "y": bin_y, "counts": counts},
                    "min_expected": min_expected, "path": path, "mode": OM_LESION_MODE}
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"  Note: incompatible {path.name}: {error}.")
    print(f"  Skipped: no complete saved OM scatter/statistics for {aname}. "
          "Run multiple_task/lesion_plot.py again.")
    return None


def plot_om_vs_lesion():
    """Replot the saved OM/L1 scatter, bin medians and Spearman permutation p.

    The sample set and min_expected threshold come from lesion_plot's cache.
    No raw lesion/cluster data, profile matching, regression fitting, or new
    permutation test is used here. The scatter uses OM_LESION_MODE (zero_W),
    the same intervention as the primary modulation heatmap, so the two
    figures describe one lesion experiment.
    Its y values are mean absolute profile differences over tasks (L1/T).
    The median connector is descriptive, not a linear or monotone fit; the
    distance axis starts at zero. Legacy Pearson/regression caches are skipped.
    """
    _ensure_out_dir()
    saved = _load_om_lesion_scatter()
    if saved is None:
        return
    om_vals, lesion_diffs = saved["om_vals"], saved["lesion_diffs"]
    medians = saved["binned_medians"]
    rho = saved["rho"]
    p_perm = saved["p_perm"]
    fig1, ax1 = plt.subplots(1, 1, figsize=_LESION_SCATTER_FIGSIZE)
    median_line, = ax1.plot(medians["x"], medians["y"], "o-", markersize=3.5,
                            label="Binned median", **_TREND_LINE_STYLE)
    _pp_str = (f"p_perm = {p_perm:.3f}" if np.isfinite(p_perm) else "p_perm = n/a")
    _rho_str = (rf"Spearman $\rho$ = {rho:.2f}" if np.isfinite(rho)
                else r"Spearman $\rho$ = n/a")
    # Hundreds of points: fade them so the median connector stays readable.
    _stat_scatter(ax1, om_vals, lesion_diffs, f"{_rho_str}\n{_pp_str}",
                  legend_handles=[median_line], point_alpha=_DENSE_SCATTER_ALPHA,
                  xlabel="Over-membership", ylabel="Lesion-effect\ndistance (L1)",
                  point_color=_LESION_POINT_COLOR)
    ax1.set_ylim(bottom=0)
    fig1.tight_layout()
    out_path1 = _multitask_out("om_vs_lesion_scatter.png")
    _save_fig(fig1, out_path1)
    print(f"  om_vs_lesion_scatter [{saved['mode']}]: Spearman rho={rho:.2f}, p_perm={p_perm:.3f} "
            f"({saved['n_perm']} saved permutations, {saved['n_clusters']} clusters, "
            f"min_expected={saved['min_expected']:g}; {saved['path'].name})")


def _om_lesion_seed_runs():
    """{aname: saved OM_LESION_MODE association} of every sibling run of
    LESION_ANAME (same feature, any seed) whose OM cache validates and holds
    a defined rho and permutation p; incomplete, legacy or undefined caches
    are left out with the loader's note."""
    runs = {}
    for directory in _sibling_run_dirs():
        saved = _load_om_lesion_scatter(directory, directory.name)
        if saved is None:
            continue
        if not (np.isfinite(saved["rho"]) and np.isfinite(saved["p_perm"])):
            print(f"  Skipped {directory.name}: undefined saved OM rank statistics.")
            continue
        runs[directory.name] = saved
    return runs


def plot_om_vs_lesion_seeds():
    """
    Figure: multitask_om_vs_lesion_seeds.png — the OM-vs-lesion Spearman rho
    of every sibling run of LESION_ANAME, the cohort view of
    plot_om_vs_lesion.

    Each run contributes one point, the saved rho between a synapse cluster's
    over-membership in an (input, hidden) neuron-cluster block and the L1/T
    distance between its lesion profile and that block's combined lesion
    profile (OM_LESION_MODE, zero_W, as in the single-run scatter); filled
    markers have a saved one-sided (negative) modulation-cluster-footprint
    permutation p below 0.05, hollow ones do not. The dash is the median over
    runs. Because the per-run test is one-sided, the cohort-level test in the
    legend title is the one-sided Wilcoxon signed-rank p of the runs' rhos
    being below zero. Only saved per-run statistics are drawn; the save
    message counts the runs with negative and with significant rho.
    """
    _ensure_out_dir()
    runs = _om_lesion_seed_runs()
    if len(runs) < 2:
        print(f"  Skipped: {len(runs)} run(s) with a saved OM-vs-lesion association; need >= 2.")
        return
    rhos = np.array([saved["rho"] for saved in runs.values()], dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=_LESION_SCATTER_FIGSIZE)
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
    offsets = _run_offsets(len(runs), 0.03, 0.22)
    significant = 0
    for offset, saved in zip(offsets, runs.values()):
        is_significant = saved["p_perm"] < 0.05
        significant += int(is_significant)
        _significance_marker(ax, offset, saved["rho"], is_significant,
                             color=_LESION_POINT_COLOR, hollow_edge=_LESION_POINT_COLOR)
    ax.hlines(np.median(rhos), -0.22, 0.22, color=_TREND_LINE_COLOR, linewidth=1.8, zorder=4)
    p_value, p_text = _wilcoxon_vs_zero(rhos, alternative="less")
    ax.set_xticks([0])
    ax.set_xticklabels(["Synapse clusters"], fontsize=_STAT_SCATTER_TICK_FONTSIZE)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel(r"Spearman $\rho$" + "\n(OM vs lesion distance)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=_STAT_SCATTER_TICK_FONTSIZE)
    _seeds_legend(ax, _significance_legend_handles(_LESION_POINT_COLOR, _LESION_POINT_COLOR),
                  len(runs),
                  note="Wilcoxon signed-rank,\n" + rf"$\rho$ < 0 (one-sided): {p_text}")
    fig.tight_layout()
    _save_fig(fig, _multitask_out("om_vs_lesion_seeds.png"),
              extra=(f"  {int(np.sum(rhos < 0))}/{len(runs)} rho < 0, {significant} significant, "
                     f"one-sided Wilcoxon p = {p_value:.3f}"))


# ─── Figure: Plasticity share ────────────────────────────────────────────────

PLASTICITY_SHARE_MOD_TYPE = "modulation_all_var_weighted_unnormalized"
_PLASTICITY_SHARE_YLIM = (-0.6, 1.6)
_PLASTICITY_SHARE_FIGSIZE = (4.6, 2.8)
_PLASTICITY_SHARE_FAMILY_LABELS = {"reaction": "No working memory",
                                   "memory": "Working memory"}
# Cells are lesion-figure points (_LESION_POINT_COLOR, grey), so each run's
# median dash is drawn near black to stay distinct from them.
_PLASTICITY_SHARE_RUN_MEDIAN_COLOR = "0.15"
# Task labels slant at this angle from horizontal: low enough that the long
# names stay short vertically, high enough that neighbors do not overlap.
_PLASTICITY_SHARE_TICK_ROTATION = 45


def _load_plasticity_share(directory=None, aname=None):
    """Load a saved plasticity-share cache (LESION_ANAME in LESION_NORM_DIR by
    default; the cohort figure passes each sibling run), or None with a note.

    lesion_plot.py computes share(task, cluster) = freeze_M effect / zero_W
    effect on (task, synapse cluster) cells whose zero_W effect is significant
    and the per-task medians; the figure draws those values. The cache also
    holds a task-level Mann-Whitney U test, which the figure does not report.
    Nothing is recomputed here.
    """
    directory = LESION_NORM_DIR if directory is None else Path(directory)
    aname = LESION_ANAME if aname is None else aname
    tag = PLASTICITY_SHARE_MOD_TYPE.replace("modulation_all_", "").replace("_", "-")
    path = resolve_lesion_cache_path(directory / f"plasticity_share_{tag}_{aname}.pkl")
    if not path.exists():
        print(f"  Skipped: {path.name} not found. Run multiple_task/lesion_plot.py first.")
        return None
    try:
        saved = load_lesion_pickle(path)
        if saved.get("mod_type") != PLASTICITY_SHARE_MOD_TYPE:
            raise ValueError("cache belongs to another modulation clustering")
        share = np.asarray(saved["share"], dtype=float)
        tasks = list(saved["tasks"])
        family = [str(value) for value in saved["task_family"]]
        medians = np.asarray(saved["task_median_share"], dtype=float)
        if (share.ndim != 2 or share.shape[0] != len(tasks) or len(family) != len(tasks)
                or medians.shape != (len(tasks),)
                or set(family) - set(_PLASTICITY_SHARE_FAMILY_LABELS)
                or not np.isfinite(share[np.isfinite(share)]).all()):
            raise ValueError("share matrix, task labels or medians do not align")
        for index in range(len(tasks)):
            values = share[index][np.isfinite(share[index])]
            if values.size and not np.isclose(np.median(values), medians[index]):
                raise ValueError("saved task medians disagree with the saved shares")
        n_cells = int(np.isfinite(share).sum())
        if n_cells == 0:
            raise ValueError("no significant cells")
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"  Skipped: incompatible {path.name}: {error}. Run multiple_task/lesion_plot.py again.")
        return None
    return {"share": share, "tasks": tasks, "family": family, "medians": medians,
            "n_cells": n_cells, "path": path}


def _draw_plasticity_share_axes(ax, tasks, family, cells, medians, run_medians=None,
                                point_alpha=0.45):
    """Draw one plasticity-share panel: one column per task, no-working-memory
    tasks first, `cells[i]` (the task's share values) as pale points with
    values beyond _PLASTICITY_SHARE_YLIM as hollow triangles at the edge, both
    in _LESION_POINT_COLOR, `medians[i]` as the red dash and, when given,
    `run_medians[i]` (one value per run) as short near-black dashes. Returns
    the number of clipped cells."""
    order = ([index for index, kind in enumerate(family) if kind == "reaction"]
             + [index for index, kind in enumerate(family) if kind == "memory"])
    n_no_memory = sum(kind == "reaction" for kind in family)
    low, high = _PLASTICITY_SHARE_YLIM
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
    ax.axhline(1.0, color="0.6", linewidth=0.6, linestyle="--", zorder=1)
    if 0 < n_no_memory < len(order):
        ax.axvline(n_no_memory - 0.5, color="0.3", linewidth=0.8, linestyle=":", zorder=1)
    n_clipped = 0
    for position, index in enumerate(order):
        values = np.asarray(cells[index], dtype=float)
        if values.size:
            jitter = np.linspace(-0.22, 0.22, values.size) if values.size > 1 else np.zeros(1)
            inside = (values >= low) & (values <= high)
            ax.scatter(position + jitter[inside], values[inside], color=_LESION_POINT_COLOR,
                       edgecolors="none", s=14, alpha=point_alpha, zorder=2)
            for beyond, marker, edge in ((values > high, "^", high), (values < low, "v", low)):
                if beyond.any():
                    n_clipped += int(beyond.sum())
                    ax.scatter(position + jitter[beyond], np.full(beyond.sum(), edge),
                               marker=marker, facecolors="none", edgecolors=_LESION_POINT_COLOR,
                               linewidths=0.6, s=16, zorder=2)
        if run_medians is not None and len(run_medians[index]):
            ax.hlines(np.clip(run_medians[index], low, high), position - 0.3, position + 0.3,
                      color=_PLASTICITY_SHARE_RUN_MEDIAN_COLOR, linewidth=0.7, zorder=3)
        if np.isfinite(medians[index]):
            ax.scatter([position], [medians[index]], marker="_", s=160, color=_TREND_LINE_COLOR,
                       linewidths=1.8, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_TASK_DISPLAY.get(tasks[index], tasks[index]) for index in order],
                       rotation=_PLASTICITY_SHARE_TICK_ROTATION, ha="right",
                       rotation_mode="anchor", fontsize=6)
    _color_motif_ticklabels(ax, [tasks[index] for index in order], axis="x")
    for start, stop, kind in ((0, n_no_memory, "reaction"), (n_no_memory, len(order), "memory")):
        if stop > start:
            ax.text((start + stop - 1) / 2, high + 0.05 * (high - low),
                    _PLASTICITY_SHARE_FAMILY_LABELS[kind],
                    ha="center", va="bottom", fontsize=7, color="0.3", clip_on=False)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(low, high)
    ax.set_ylabel("Plasticity share\n(freeze M / zero W)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    return n_clipped


def _plasticity_share_median_handle(label="Task median"):
    return mpl.lines.Line2D([], [], color=_TREND_LINE_COLOR, marker="_", linestyle="none",
                            markersize=8, markeredgewidth=1.8, label=label)


def plot_plasticity_share():
    """
    Figure: plasticity share of each task's synapse-cluster dependence.

    One column per task, no-working-memory tasks first. Pale grey points are
    the saved share values, one per (task, synapse cluster) cell whose zero_W
    effect was significant; the red dash is the saved task median. Share 1
    means freezing the plasticity of that cluster costs the task as much as
    removing its weights; 0 means the static wiring suffices. Values beyond
    the y-limits are drawn as hollow triangles at the edge, so no cell is
    hidden. The figure shows the values only, with no test statistic; task
    labels carry their motif colors. Everything is read from lesion_plot.py's
    cache for LESION_ANAME and the var-weighted modulation clustering.
    """
    _ensure_out_dir()
    saved = _load_plasticity_share()
    if saved is None:
        return
    share = saved["share"]
    cells = [row[np.isfinite(row)] for row in share]
    fig, ax = plt.subplots(1, 1, figsize=_PLASTICITY_SHARE_FIGSIZE)
    n_clipped = _draw_plasticity_share_axes(ax, saved["tasks"], saved["family"], cells,
                                            saved["medians"])
    # The band below share 0 is empty for the memory tasks, so the legend sits
    # there instead of over the points at the right edge.
    _legend(ax, handles=[_plasticity_share_median_handle()], loc="lower center", fontsize=6,
            frameon=True)
    fig.tight_layout()
    _save_fig(fig, _multitask_out("plasticity_share.png"),
              extra=f"  [{LESION_ANAME}] {saved['n_cells']} cells, {n_clipped} beyond y-limits")


def _plasticity_share_cohort_runs():
    """The saved plasticity-share caches of every sibling run of LESION_ANAME
    (same feature, any seed) that validates and lists the same tasks with the
    same families as the first; a run with another task list is left out with
    a note. Returns the list of loaded caches."""
    runs = []
    for directory in _sibling_run_dirs():
        saved = _load_plasticity_share(directory, directory.name)
        if saved is None:
            continue
        if runs and (saved["tasks"] != runs[0]["tasks"] or saved["family"] != runs[0]["family"]):
            print(f"  Skipped {directory.name}: its task list differs from the other runs.")
            continue
        runs.append(saved)
    return runs


def plot_plasticity_share_cohort():
    """
    Figure: multitask_plasticity_share_cohort.png — the plasticity-share
    figure with every sibling run of LESION_ANAME overlaid.

    Same layout as plot_plasticity_share: one column per task, no-working-
    memory tasks first, share 1 meaning that freezing a cluster's plasticity
    costs the task as much as removing its weights. The pale points are the
    saved significant (task, synapse cluster) cells of all runs pooled per
    task, each short near-black dash is one run's saved task median, and the red
    dash is the median of the pooled cells. Values beyond the y-limits are
    hollow triangles at the edge; run medians beyond them are drawn at the
    edge. The legend title gives the run count. Only saved values are drawn;
    the figure is skipped below two runs.
    """
    _ensure_out_dir()
    runs = _plasticity_share_cohort_runs()
    if len(runs) < 2:
        print(f"  Skipped: {len(runs)} run(s) with saved plasticity-share caches; need >= 2.")
        return
    tasks, family = runs[0]["tasks"], runs[0]["family"]
    cells, run_medians = [], []
    for index in range(len(tasks)):
        rows = [saved["share"][index] for saved in runs]
        cells.append(np.concatenate([row[np.isfinite(row)] for row in rows]))
        run_medians.append(np.array([saved["medians"][index] for saved, row in zip(runs, rows)
                                     if np.isfinite(row).any()]))
    medians = [float(np.median(values)) if values.size else np.nan for values in cells]
    fig, ax = plt.subplots(1, 1, figsize=_PLASTICITY_SHARE_FIGSIZE)
    n_clipped = _draw_plasticity_share_axes(ax, tasks, family, cells, medians, run_medians,
                                            point_alpha=0.3)
    run_handle = mpl.lines.Line2D([], [], color=_PLASTICITY_SHARE_RUN_MEDIAN_COLOR, linewidth=0.7,
                                  label="Run median")
    # With every run pooled no band of the axes stays empty, so the legend
    # sits outside, to the right of the axes.
    _legend(ax, handles=[_plasticity_share_median_handle("Pooled median"), run_handle],
            loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0, fontsize=6,
            frameon=False, title=f"{len(runs)} runs", title_fontsize=6)
    fig.tight_layout()
    n_cells = int(sum(values.size for values in cells))
    _save_fig(fig, _multitask_out("plasticity_share_cohort.png"),
              extra=f"  {len(runs)} runs, {n_cells} cells, {n_clipped} beyond y-limits")


# ─── Figure: Task-pair sharing of impaired clusters, over runs ──────────────

# Display names of lesion_plot.TASK_PAIR_RELATIONS, the task component the
# two tasks of a pair differ in; "related" pools every such pair.
_TASK_SHARING_RELATION_LABELS = {
    "related": "All\nrelated", "response rule": "Response\nrule", "timing": "Timing",
    "modality": "Modality", "context cue": "Context\ncue",
    "integration family": "Integration\nfamily", "match/category family": "Match /\ncategory",
}
_TASK_SHARING_FIGSIZE = (4.2, 5.4)


def _short_p(p_value):
    """Compact p text for tight column annotations: 'p<.001', 'p=.004', 'p=.54'."""
    if not np.isfinite(p_value):
        return "p=n/a"
    if p_value < 0.001:
        return "p<.001"
    digits = 3 if p_value < 0.01 else 2
    return "p=" + f"{p_value:.{digits}f}".lstrip("0")


def _load_task_sharing(directory, aname):
    """The saved task-pair sharing statistics of run `aname` from its
    task-specificity cache in `directory`: {cluster type: {"relations":
    [...], "mean": {relation: mean Jaccard}, "p_perm": {relation: p}}},
    where the means and one-sided task-label permutation p cover every
    relation, the pooled "related" pairs and the reference "other" pairs
    (whose p is undefined). None, silently, when the run has no cache; None
    with a note when the cache is incompatible."""
    path = resolve_lesion_cache_path(Path(directory) / f"task_specificity_{aname}.pkl")
    if not path.exists():
        return None
    try:
        saved = load_lesion_pickle(path)
        if saved.get("schema_version") != 1 or saved.get("aname") != aname:
            raise ValueError("cache schema or run does not match")
        types = {}
        for type_name, entry in saved["types"].items():
            if type_name not in _TASK_SPECIFICITY_TYPES:
                raise ValueError(f"unknown cluster type {type_name!r}")
            sharing = entry["sharing"]
            if (sharing.get("permutation_unit") != "task_label"
                    or sharing.get("side") != "greater"):
                raise ValueError(f"{type_name}: sharing test is not the task-label permutation")
            relations = [name for name in sharing["relations"] if name != "other"]
            if not relations or "other" not in sharing["relations"]:
                raise ValueError(f"{type_name}: sharing has no relation groups or no reference")
            means, p_values = {}, {}
            for name in relations + ["other", "related"]:
                rel = sharing["by_relation"][name]
                means[name] = float(rel["mean"])
                p_values[name] = float(rel["p_perm"])
            if not all(np.isfinite(value) for value in means.values()):
                raise ValueError(f"{type_name}: non-finite mean sharing")
            types[type_name] = {"relations": relations, "mean": means, "p_perm": p_values}
        if not types:
            raise ValueError("no cluster types saved")
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"  Skipped {aname}: incompatible {path.name} ({error}). "
              "Run multiple_task/lesion_plot.py again.")
        return None
    return types


def _task_sharing_runs():
    """{aname: _load_task_sharing(...)} over the sibling runs of LESION_ANAME
    whose caches validate and agree with the first run on the cluster types
    and relation lists; a disagreeing run is left out with a note."""
    runs = {}
    for directory in _sibling_run_dirs():
        types = _load_task_sharing(directory, directory.name)
        if types is None:
            continue
        if runs:
            first = next(iter(runs.values()))
            if (set(types) != set(first)
                    or any(types[name]["relations"] != first[name]["relations"] for name in first)):
                print(f"  Skipped {directory.name}: its cluster types or relations differ "
                      "from the other runs.")
                continue
        runs[directory.name] = types
    return runs


def plot_task_sharing_cohort():
    """
    Figure: multitask_task_sharing_cohort.png — do tasks that differ in one
    component depend on the same clusters? Every sibling run of LESION_ANAME,
    one panel per cluster type.

    For each run and relation (the component the two tasks of a pair differ
    in, lesion_plot.TASK_PAIR_RELATIONS; "All related" pools them), the point
    is the excess sharing: the run's saved mean Jaccard overlap of impaired
    clusters over the relation's task pairs minus that over the unrelated
    ("other") pairs, so zero means related pairs share no more than unrelated
    ones and the panels are comparable although synapse clusters are shared
    far more than neuron clusters. A point is filled when the run's saved
    one-sided task-label permutation p for that relation is below 0.05. The
    dash is the median over runs, and the small text above each column is the
    cohort-level test, a one-sided Wilcoxon signed-rank p of the runs' excess
    sharing being above zero (computed here from the saved per-run means, as
    in the other cohort figures). The dotted line separates the pooled
    column from the relations. Skipped below two runs.
    """
    _ensure_out_dir()
    runs = _task_sharing_runs()
    if len(runs) < 2:
        print(f"  Skipped: {len(runs)} run(s) with saved task-sharing caches; need >= 2.")
        return
    first = next(iter(runs.values()))
    type_names = [name for name in _TASK_SPECIFICITY_TYPES if name in first]
    columns = ["related"] + first[type_names[0]]["relations"]
    fig, axs = plt.subplots(len(type_names), 1, figsize=_TASK_SHARING_FIGSIZE,
                            sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    offsets = _run_offsets(len(runs), 0.06, 0.3)
    summaries = []
    for ax, type_name in zip(axs, type_names):
        color = _TASK_SPECIFICITY_COLORS[type_name]
        ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
        ax.axvline(0.5, color="0.3", linewidth=0.8, linestyle=":", zorder=1)
        parts = []
        for position, name in enumerate(columns):
            excess = np.array([types[type_name]["mean"][name] - types[type_name]["mean"]["other"]
                               for types in runs.values()])
            p_values = np.array([types[type_name]["p_perm"][name] for types in runs.values()])
            for offset, value, p_value in zip(offsets, excess, p_values):
                significant = p_value < 0.05
                ax.scatter([position + offset], [value], s=18, zorder=3,
                           color=color if significant else "white",
                           edgecolors=color if significant else "k", linewidths=0.6)
            ax.hlines(np.median(excess), position - 0.3, position + 0.3,
                      color=_TREND_LINE_COLOR, linewidth=1.8, zorder=4)
            p_cohort, _ = _wilcoxon_vs_zero(excess, alternative="greater")
            ax.text(position, 0.98, _short_p(p_cohort), ha="center", va="top", fontsize=5,
                    color="0.3", transform=mpl.transforms.blended_transform_factory(
                        ax.transData, ax.transAxes))
            parts.append(f"{name} {int(np.sum(excess > 0))}/{len(runs)} > 0, "
                         f"{int(np.sum(p_values < 0.05))} significant, "
                         f"Wilcoxon p = {p_cohort:.3f}")
        summaries.append(f"{type_name}: " + "; ".join(parts))
        ax.set_title(_TASK_SPECIFICITY_LABELS[type_name], fontsize=8, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    axs[-1].set_xticks(np.arange(len(columns)))
    axs[-1].set_xticklabels([_TASK_SHARING_RELATION_LABELS.get(name, name) for name in columns],
                            fontsize=6)
    axs[-1].set_xlim(-0.6, len(columns) - 0.4)
    # Room above the points for the column p texts, the zero line always shown.
    low, high = axs[0].get_ylim()
    axs[0].set_ylim(min(low, -0.05), high + 0.18 * (high - low))
    axs[len(axs) // 2].set_ylabel("Excess shared impaired clusters\n(Jaccard, related - other pairs)",
                                  fontsize=8)
    filled = mpl.lines.Line2D([], [], marker="o", linestyle="none", color="0.3", markersize=5,
                              label="run, permutation p < 0.05")
    hollow = mpl.lines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                              markeredgecolor="k", markersize=5, label=r"p $\geq$ 0.05")
    median = mpl.lines.Line2D([], [], color=_TREND_LINE_COLOR, linewidth=1.8,
                              label="median over runs")
    # One legend across the top of the stacked panels (shared with the
    # two-panel state-space figures, which lay out the same way).
    _state_space_figure_legend(fig, [filled, hollow, median], ncol=3, top=0.95)
    _save_fig(fig, _multitask_out("task_sharing_cohort.png"),
              extra=f"  {len(runs)} runs; " + " | ".join(summaries))


# ─── Figure: Context-end task geometry vs causal dependence ──────────────────
# Both read the combined results of multiple_task/state_space_vs_lesion.py,
# which links the state-space structure caches to the lesion caches over the
# task pairs; nothing is recomputed here.

STATE_SPACE_VS_LESION_RESULT_PATH = Path("state_space") / "state_space_vs_lesion_results.pkl"
_STATE_SPACE_VS_LESION_HINT = "Run python multiple_task/state_space_vs_lesion.py first."
_SSL_CLUSTER_TYPES = (("neuron", "Neuron clusters"), ("synapse", "Synapse clusters"))
_SSL_REPRESENTATIONS = (("hidden", "Hidden"), ("eff_mod", r"W $\odot$ M"))
_SSL_DISTANCE_LABELS = {"hidden": "hidden-state centroids", "eff_mod": r"W $\odot$ M centroids"}


def _state_space_vs_lesion_entry(combined, aname):
    """The validated results entry of run `aname`, or None with a message."""
    entry = combined.get(aname)
    if entry is None:
        print(f"  Skipped: {aname} has no state-space-vs-lesion results. "
              f"{_STATE_SPACE_VS_LESION_HINT}")
        return None
    try:
        if entry["schema_version"] != 1 or entry["aname"] != aname:
            raise ValueError("schema or run does not match")
        n_pairs = len(entry["pairs"])
        for representation, _ in _SSL_REPRESENTATIONS:
            if np.asarray(entry["distances"][representation]).shape != (n_pairs,):
                raise ValueError(f"{representation} distances do not match the pairs")
            for cluster_type, _ in _SSL_CLUSTER_TYPES:
                test = entry["tests"][representation][cluster_type]
                if (test["statistic"] != "spearman" or test["sided"] != "two-sided"
                        or not np.isfinite(test["rho"]) or not np.isfinite(test["p_value"])):
                    raise ValueError(f"invalid test for {representation} vs {cluster_type}")
        for cluster_type, _ in _SSL_CLUSTER_TYPES:
            if np.asarray(entry["similarities"][cluster_type]).shape != (n_pairs,):
                raise ValueError(f"{cluster_type} similarities do not match the pairs")
    except (KeyError, TypeError, ValueError) as error:
        print(f"  Skipped: incompatible state-space-vs-lesion entry for {aname} ({error}). "
              f"{_STATE_SPACE_VS_LESION_HINT}")
        return None
    return entry


def plot_state_space_vs_lesion():
    """
    Figure: multitask_state_space_vs_lesion.png — do tasks whose context-end
    states lie closer depend on the same clusters? LESION_ANAME, a 2 x 2 grid:
    one row per representation of the context-end state (hidden state, W * M),
    one column per cluster type (neuron clusters, synapse clusters).

    Each point is one task pair: x the Euclidean distance between the two
    tasks' full-dimensional context-end centroids in that row's
    representation (the state-space analysis, state_space_structure.py), y
    the Pearson correlation of their lesion-effect profiles (random-control
    minus lesion accuracy), over all unnormalized input and hidden neuron
    clusters (the causal-vs-activity matrix) or over the var-weighted zero_W
    synapse clusters without the unresponsive class. Each panel's legend
    reports the saved Spearman rho with its two-sided task-label permutation
    (Mantel) p, from multiple_task/state_space_vs_lesion.py; a negative rho
    links the task geometry of the state-space chapter to the causal
    dependence of the lesion chapter. No line is drawn: the statistic is a
    rank correlation.
    """
    _ensure_out_dir()
    combined = _load_pkl_or_skip(STATE_SPACE_VS_LESION_RESULT_PATH, _STATE_SPACE_VS_LESION_HINT)
    if combined is None:
        return
    entry = _state_space_vs_lesion_entry(combined, LESION_ANAME)
    if entry is None:
        return
    # One _LESION_SCATTER_FIGSIZE square per panel, so each panel matches the
    # single-panel lesion scatters.
    n_rows, n_cols = len(_SSL_REPRESENTATIONS), len(_SSL_CLUSTER_TYPES)
    panel_width, panel_height = _LESION_SCATTER_FIGSIZE
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * panel_width, n_rows * panel_height))
    notes = []
    for row, (representation, _) in enumerate(_SSL_REPRESENTATIONS):
        x = np.asarray(entry["distances"][representation], dtype=float)
        for column, (cluster_type, title) in enumerate(_SSL_CLUSTER_TYPES):
            ax = axs[row, column]
            y = np.asarray(entry["similarities"][cluster_type], dtype=float)
            test = entry["tests"][representation][cluster_type]
            label = (rf"$\rho$ = {test['rho']:.2f} (n = {x.size} pairs)" + "\n"
                     f"permutation {_permutation_p_text(test['p_value'])}")
            _stat_scatter(ax, x, y, label,
                          xlabel=f"Context-end distance\n({_SSL_DISTANCE_LABELS[representation]})",
                          ylabel="Lesion-profile\nsimilarity (r)" if column == 0 else "",
                          point_color=_LESION_POINT_COLOR)
            if row == 0:
                ax.set_title(title, fontsize=9, pad=22)
            notes.append(f"{representation}/{cluster_type}: rho = {test['rho']:.3f}, "
                         f"p = {test['p_value']:.4f}")
    fig.tight_layout()
    _save_fig(fig, _multitask_out("state_space_vs_lesion.png"),
              extra=f"  [{LESION_ANAME}] " + "; ".join(notes))


def plot_state_space_vs_lesion_seeds():
    """
    Figure: multitask_state_space_vs_lesion_seeds.png — the Spearman rho of
    context-end distance vs lesion-profile similarity for every sibling run of
    LESION_ANAME, one panel per cluster type (neuron, synapse), hidden-state
    and W * M distances side by side.

    One point per run, filled when its saved two-sided task-label permutation
    p is below 0.05, the median over runs as a dash, and under each tick the
    two-sided Wilcoxon signed-rank p of the runs' rhos against zero, the
    cohort-level test computed here from the saved per-run rhos. Only runs of
    the selected cohort with saved results are drawn; skipped below two runs.
    """
    _ensure_out_dir()
    combined = _load_pkl_or_skip(STATE_SPACE_VS_LESION_RESULT_PATH, _STATE_SPACE_VS_LESION_HINT)
    if combined is None:
        return
    names = [directory.name for directory in _sibling_run_dirs()]
    entries = [entry for entry in (_state_space_vs_lesion_entry(combined, name)
                                   for name in names if name in combined)
               if entry is not None]
    if len(entries) < 2:
        print(f"  Skipped: {len(entries)} run(s) with state-space-vs-lesion results; need >= 2.")
        return
    fig, axs = plt.subplots(1, len(_SSL_CLUSTER_TYPES), figsize=(3.8, 2.3), sharey=True)
    positions = np.arange(len(_SSL_REPRESENTATIONS))
    offsets = _run_offsets(len(entries), 0.05, 0.3)
    summaries = []
    for ax, (cluster_type, title) in zip(axs, _SSL_CLUSTER_TYPES):
        ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
        tick_labels = []
        for position, (representation, label) in zip(positions, _SSL_REPRESENTATIONS):
            tests = [entry["tests"][representation][cluster_type] for entry in entries]
            rhos = np.array([test["rho"] for test in tests])
            for offset, test in zip(offsets, tests):
                _significance_marker(ax, position + offset, test["rho"],
                                     test["p_value"] < _STATE_SPACE_SIGNIFICANCE, size=22,
                                     color=_LESION_POINT_COLOR, hollow_edge=_LESION_POINT_COLOR)
            ax.hlines(np.median(rhos), position - 0.3, position + 0.3, color=_TREND_LINE_COLOR,
                      linewidth=1.8, zorder=4)
            p_cohort, p_text = _wilcoxon_vs_zero(rhos)
            tick_labels.append(f"{label}\n{p_text}")
            summaries.append(f"{cluster_type}/{representation}: "
                             f"{int(np.sum(rhos < 0))}/{len(rhos)} rho < 0, "
                             f"{sum(test['p_value'] < 0.05 for test in tests)} significant, "
                             f"Wilcoxon p = {p_cohort:.3f}")
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlim(-0.6, len(positions) - 0.4)
        ax.set_title(title, fontsize=9, pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    axs[0].set_ylabel(r"Spearman $\rho$" + "\n(distance vs lesion similarity)", fontsize=7)
    _state_space_figure_legend(
        fig, _significance_legend_handles(_LESION_POINT_COLOR, _LESION_POINT_COLOR) + [
            mpl.lines.Line2D([], [], color=_TREND_LINE_COLOR, linewidth=1.8,
                             label="median over runs")])
    _save_fig(fig, _multitask_out("state_space_vs_lesion_seeds.png"),
              extra=f"  {len(entries)} runs; " + "; ".join(summaries))


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


# ─── Figures: Row clusters by period or by task ──────────────────────────────
# Read from multiple_task/sibling_row_sharing.py, which saves one result per
# cohort as multiple_tasks_diagnosis/sibling_row_sharing_{feature}.pkl (every
# clustered run of that cohort) and, for the untrained networks of
# untrained_clustering.py, as sibling_row_sharing_{feature}_untrained.pkl;
# nothing is recomputed here. A row of the
# clustered input / hidden / modulation heatmaps is one task in one trial
# period, and these figures ask whether the row clusters group the rows by
# period or by task.
MULTITASK_DIAGNOSIS_ROOT = Path("multiple_tasks_diagnosis")
_ROW_SHARING_RESULT_VERSION = 4
_ROW_SHARING_REPRESENTATIONS = (("input", "Input\nneurons"), ("hidden", "Hidden\nneurons"),
                                ("modulation", "Synapses"))
_ROW_ORGANIZATION_PANELS = (("period", "Same period,\ndifferent task"),
                            ("task", "Same task,\ndifferent period"))
_ROW_ORGANIZATION_FIGSIZE = (4.2, 2.6)
_ROW_ORGANIZATION_YTICKS = (0.25, 0.5, 1, 2, 4, 8, 16)
# Within a representation's column the trained runs sit left of the untrained
# controls, each sub-column this far from the column's center.
_ROW_ORGANIZATION_SPLIT = 0.2
_ROW_ORGANIZATION_UNTRAINED_MEDIAN = "0.35"
_ROW_SAME_TASK_BAR = 0.26
_ROW_SAME_TASK_Y_TOP = 112                # head room for the stars above 100%
# Width per group of the per-task / per-period bar figures; with untrained
# networks each representation's slot holds a trained and an untrained bar.
_ROW_SHARING_GROUP_IN = {False: 0.42, True: 0.55}
_ROW_SHARING_CHANCE_COLOR = "0.25"
_ROW_SHARING_SIGNIFICANCE = 0.05


def _row_sharing_commands(untrained=False):
    """The commands that write the cohort's (or its untrained controls')
    row-cluster sharing result."""
    feature = multitask_feature()
    if not untrained:
        return f"python multiple_task/sibling_row_sharing.py --feature {feature}"
    seed = MULTITASK_DESIGNATED_SEEDS.get(MULTITASK_L2, {}).get("multiple_tasks") or "<seed>"
    return (f"python multiple_task/untrained_clustering.py --feature {feature} "
            f"--seed {seed} --init-seed 0 1 2 3 4, then python "
            f"multiple_task/sibling_row_sharing.py --feature {feature} --untrained")


def _load_row_cluster_sharing(untrained=False):
    """The selected cohort's saved row-cluster sharing result, or with
    `untrained` that of its untrained controls (untrained_clustering.py); None
    with a message when it is missing or older than the organization check."""
    feature = multitask_feature()
    tag = f"{feature}_untrained" if untrained else feature
    path = MULTITASK_DIAGNOSIS_ROOT / f"sibling_row_sharing_{tag}.pkl"
    command = f"Run {_row_sharing_commands(untrained)}"
    data = _load_pkl_or_skip(path, f"{command}.")
    if data is None:
        return None
    if (data.get("version") != _ROW_SHARING_RESULT_VERSION or data.get("feature") != feature
            or data.get("seed") is not None or bool(data.get("untrained")) != untrained):
        print(f"  Skipped: {path} is not a version-{_ROW_SHARING_RESULT_VERSION} {tag} "
              f"cohort result. {command}.")
        return None
    return data


def _load_untrained_row_sharing():
    """The untrained controls' row-cluster sharing result, or None with a
    message when there is none, so a figure draws the trained runs alone."""
    path = MULTITASK_DIAGNOSIS_ROOT / f"sibling_row_sharing_{multitask_feature()}_untrained.pkl"
    if path.exists():
        return _load_row_cluster_sharing(untrained=True)
    print(f"  No untrained control ({path.name}); drawing the trained runs only. "
          f"Run {_row_sharing_commands(untrained=True)}.")
    return None


def _untrained_provenance(experiments):
    """'seed 749 config, init 0, 1, 2' from the untrained networks' names
    ({aname}_init{k}), or '' when the names do not carry them."""
    import re

    configs = sorted({int(match.group(1)) for name in experiments
                      if (match := re.search(r"_seed(\d+)_", name))})
    inits = sorted(int(match.group(1)) for name in experiments
                   if (match := re.search(r"_init(\d+)$", name)))
    if not configs or not inits:
        return ""
    return (f"seed {', '.join(map(str, configs))} config, "
            f"init {', '.join(map(str, inits))}")


def _organization_values(entry, pair_class):
    """One representation's per-run enrichments of `pair_class` that a log
    axis can show (finite and positive)."""
    values = np.asarray(entry["enrichment"][pair_class], dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def plot_row_cluster_organization():
    """
    Figure: multitask_row_cluster_organization.png — are the rows of the
    clustered input, hidden and modulation heatmaps (one row per task and
    trial period) grouped by period or by task, and does training make them
    so? Every clustered run of the MULTITASK_L2 cohort, every task, beside
    untrained networks of one run's config when they are clustered.

    Left panel: two rows of one period from different tasks; right panel: two
    rows of one task from different periods. Each point is one network's fold
    enrichment, the fraction of such row pairs that share a row cluster over
    the fraction of any two rows that do, which is the chance the network's
    cluster sizes give; so 1 (dashed) is chance on both panels, and the
    representations compare although they have different numbers of row
    clusters (about 9 input, 6 hidden and 13 synapse clusters on the L2 =
    1e-4 runs). The y axis is logarithmic. In each representation's column
    the trained runs (filled, red median dash) sit left of the untrained
    networks (hollow, gray median dash: untrained_clustering.py, fresh
    weights of the designated run's config, clustered by the same pipeline).
    The text above a column is the two-sided Mann-Whitney p of trained
    against untrained, or, without untrained networks, the two-sided
    Wilcoxon signed-rank p of the trained runs against 1. The saved message
    adds the trained runs' paired Wilcoxon p of task-minus-period enrichment
    between representations. Read from
    multiple_tasks_diagnosis/sibling_row_sharing_{feature}[_untrained].pkl
    (multiple_task/sibling_row_sharing.py); skipped without the trained
    result, drawn without the untrained networks when theirs is missing.
    """
    from scipy.stats import mannwhitneyu

    _ensure_out_dir()
    data = _load_row_cluster_sharing()
    if data is None:
        return
    reps = data["organization"]["representations"]
    columns = [(rep, label) for rep, label in _ROW_SHARING_REPRESENTATIONS if rep in reps]
    if not columns:
        print("  Skipped: the result holds no organization check.")
        return
    untrained = _load_untrained_row_sharing()
    control = {} if untrained is None else untrained["organization"]["representations"]
    split = _ROW_ORGANIZATION_SPLIT if control else 0.0
    half_width = 0.13 if control else 0.28

    fig, axs = plt.subplots(1, len(_ROW_ORGANIZATION_PANELS), figsize=_ROW_ORGANIZATION_FIGSIZE,
                            sharey=True)
    low = high = 1.0
    tests = []
    for ax, (pair_class, title) in zip(axs, _ROW_ORGANIZATION_PANELS):
        ax.axhline(1.0, color="0.6", linewidth=0.7, linestyle="--", zorder=1)
        for position, (rep, _) in enumerate(columns):
            color = _TASK_SPECIFICITY_COLORS[rep]
            trained = _organization_values(reps[rep], pair_class)
            groups = [(position - split, trained, color, _TREND_LINE_COLOR)]
            fresh = (_organization_values(control[rep], pair_class) if rep in control
                     else np.array([]))
            if fresh.size:
                groups.append((position + split, fresh, "white",
                               _ROW_ORGANIZATION_UNTRAINED_MEDIAN))
            for x, values, face, median_color in groups:
                if not values.size:
                    continue
                ax.scatter(x + _run_offsets(values.size, 0.035, half_width), values, s=12,
                           color=face, edgecolors=color, linewidths=0.6, alpha=0.8,
                           zorder=3)
                ax.hlines(np.median(values), x - half_width - 0.02, x + half_width + 0.02,
                          color=median_color, linewidth=1.8, zorder=4)
                low, high = min(low, values.min()), max(high, values.max())
            if fresh.size and trained.size:
                p_value = float(mannwhitneyu(trained, fresh, alternative="two-sided").pvalue)
                tests.append(f"{pair_class} {rep} trained {np.median(trained):.2f}x vs "
                             f"untrained {np.median(fresh):.2f}x, Mann-Whitney p = "
                             f"{p_value:.2g}")
            else:
                p_value = reps[rep]["p_vs_one"][pair_class]
            ax.text(position, 0.98, _short_p(p_value), ha="center", va="top", fontsize=5,
                    color="0.3", transform=mpl.transforms.blended_transform_factory(
                        ax.transData, ax.transAxes))
        ax.set_title(title, fontsize=8)
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels([label for _, label in columns], fontsize=6)
        ax.set_xlim(-0.6, len(columns) - 0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    axs[0].set_yscale("log", base=2)
    # Room above the points for the column p texts.
    axs[0].set_ylim(low / 1.3, high * 2.2)
    ticks = [tick for tick in _ROW_ORGANIZATION_YTICKS if low / 1.3 <= tick <= high * 2.2]
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels([f"{tick:g}\u00d7" for tick in ticks])
    axs[0].yaxis.set_minor_locator(mticker.NullLocator())
    axs[0].set_ylabel("Row pairs in one cluster\n(fold over chance)", fontsize=8)
    handles = [mpl.lines.Line2D([], [], color=_TREND_LINE_COLOR, linewidth=1.8,
                                label="trained, median")]
    if control:
        handles.append(mpl.lines.Line2D([], [], color=_ROW_ORGANIZATION_UNTRAINED_MEDIAN,
                                        linewidth=1.8, label="untrained, median"))
    handles.append(mpl.lines.Line2D([], [], color="0.6", linewidth=0.7, linestyle="--",
                                    label="chance"))
    _state_space_figure_legend(fig, handles, ncol=len(handles), top=0.92)
    n_runs = max(len(reps[rep]["anames"]) for rep, _ in columns)
    medians = "; ".join(
        f"{rep} " + ", ".join(f"{name} {reps[rep]['median'][name]:.2f}x"
                              for name, _ in _ROW_ORGANIZATION_PANELS)
        for rep, _ in columns)
    contrasts = "; ".join(f"{name.replace('_vs_', ' vs ')} p = {p_value:.2g}"
                          for name, p_value in data["organization"]["contrast_p"].items())
    n_untrained = (max(len(entry["anames"]) for entry in control.values()) if control else 0)
    provenance = _untrained_provenance(untrained["experiments"]) if control else ""
    _save_fig(fig, _multitask_out("row_cluster_organization.png"),
              extra=(f"  {n_runs} {data['feature']} runs, {n_untrained} untrained networks"
                     + (f" ({provenance})" if provenance else "") + "; "
                     f"trained median enrichment: {medians}; task - period enrichment, "
                     f"paired Wilcoxon: {contrasts}"
                     + (f"; {'; '.join(tests)}" if tests else "")))


def _draw_row_sharing_bars(ax, x0, trained, untrained=None):
    """One group of a per-task / per-period row-sharing figure, centered at
    `x0`: per representation, the % of the trained runs' cells whose two rows
    share a row cluster (filled bar) and its mean chance (dash), with a star
    when the per-run excess over chance differs from zero (two-sided Wilcoxon
    p < _ROW_SHARING_SIGNIFICANCE). With `untrained` (possibly empty) every
    representation's slot is split: the trained bar left, the untrained
    networks' bar (hollow, its own chance dash) right. Both arguments are
    {rep: sibling_row_sharing._summarize_cells stats}."""
    split = untrained is not None
    width = _ROW_SAME_TASK_BAR * (0.45 if split else 0.9)
    for k, (rep, _) in enumerate(_ROW_SHARING_REPRESENTATIONS):
        center = x0 + (k - 1) * _ROW_SAME_TASK_BAR
        color = _TASK_SPECIFICITY_COLORS[rep]
        bars = [(trained.get(rep), center - width / 2 if split else center,
                 dict(color=color, edgecolor="k", linewidth=0.3))]
        if split:
            bars.append((untrained.get(rep), center + width / 2,
                         dict(color="white", edgecolor=color, linewidth=0.6)))
        for stats, x, style in bars:
            if stats is None:
                continue
            height, chance = stats["pct_same_cluster"], 100 * stats["chance_mean"]
            ax.bar(x, height, width=width * 0.92, zorder=2, **style)
            ax.hlines(chance, x - width * 0.46, x + width * 0.46,
                      color=_ROW_SHARING_CHANCE_COLOR, linewidth=1.0, zorder=3)
            if stats["p_excess"] < _ROW_SHARING_SIGNIFICANCE:
                ax.text(x, max(height, chance) + 1.5, "*", ha="center", va="bottom",
                        fontsize=6)


def _row_sharing_bar_figure(groups, labels, ylabel, untrained):
    """(fig, ax) of a per-task / per-period row-sharing figure: one
    `_draw_row_sharing_bars` group per (trained, untrained) entry of `groups`,
    labeled `labels` on x, on the 0-100% scale, with the legend; `untrained`
    says whether the untrained networks are drawn."""
    fig, ax = plt.subplots(1, 1, figsize=(1.0 + _ROW_SHARING_GROUP_IN[untrained] * len(groups),
                                          2.4))
    for position, (trained, control) in enumerate(groups):
        _draw_row_sharing_bars(ax, position, trained, control if untrained else None)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, _ROW_SAME_TASK_Y_TOP)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.spines["left"].set_bounds(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    handles = [mpl.patches.Patch(facecolor=_TASK_SPECIFICITY_COLORS[rep], edgecolor="k",
                                 linewidth=0.3, label=label.replace("\n", " "))
               for rep, label in _ROW_SHARING_REPRESENTATIONS]
    if untrained:
        handles.append(mpl.patches.Patch(facecolor="white", edgecolor="0.3", linewidth=0.6,
                                         label="untrained (hollow)"))
    handles += [mpl.lines.Line2D([], [], color=_ROW_SHARING_CHANCE_COLOR, linewidth=1.0,
                                 label="chance"),
                mpl.lines.Line2D([], [], linestyle="none", marker="$*$", color="k",
                                 markersize=5,
                                 label=f"differs from chance, p < {_ROW_SHARING_SIGNIFICANCE}")]
    # A narrow figure (few groups) wraps its legend onto more rows.
    ncol = len(handles) if len(groups) >= 8 else 4
    n_rows = -(-len(handles) // ncol)
    _state_space_figure_legend(fig, handles, ncol=ncol, top=0.9 if n_rows == 1 else 0.8)
    return fig, ax


def _row_sharing_untrained_note(untrained):
    """'; N untrained networks (seed S config, init ...)' for a save message,
    or '' without untrained networks."""
    if untrained is None:
        return ""
    provenance = _untrained_provenance(untrained["experiments"])
    return (f"; {len(untrained['experiments'])} untrained networks"
            + (f" ({provenance})" if provenance else ""))


def plot_row_cluster_same_task():
    """
    Figure: multitask_row_cluster_same_task.png — the per-task view of the
    right panel of row_cluster_organization: for each task, how often its own
    rows of two different trial periods share a row cluster.

    One group of three bars per task (input neurons, hidden neurons,
    synapses), each the % of (run, period pair) cells whose two rows share a
    row cluster. The dark dash is that task's chance: the fraction of a row of
    the one period and a row of the other from two different tasks sharing a
    cluster, averaged over the task's cells, so the period combination is
    controlled for (a stricter baseline than the organization figure's).
    Cells at chance >= 95% (the two memory periods in input and hidden, where
    every task's rows share one cluster) are left out. A star marks a per-run
    excess over chance that differs from zero (two-sided Wilcoxon signed-rank
    p < 0.05) in either direction; the bar against its dash gives the
    direction. When the untrained networks are clustered, each bar has a
    hollow twin on its right with their % and chance (their stars need at
    least six networks). Tasks with a single clustered period (ReactPro,
    ReactAnti) have no cells. Read from
    multiple_tasks_diagnosis/sibling_row_sharing_{feature}[_untrained].pkl
    (multiple_task/sibling_row_sharing.py).
    """
    _ensure_out_dir()
    data = _load_row_cluster_sharing()
    if data is None:
        return
    entry = data["same_task_all"]
    tasks = list(entry["tasks"])
    if not tasks:
        print("  Skipped: the result holds no same-task cells.")
        return
    untrained = _load_untrained_row_sharing()
    control = {} if untrained is None else untrained["same_task_all"]["tasks"]
    fig, _ = _row_sharing_bar_figure(
        [(entry["tasks"][rule], control.get(rule, {})) for rule in tasks],
        [_TASK_DISPLAY.get(rule, rule) for rule in tasks],
        "Period pairs of one task\nin one row cluster", untrained is not None)
    missing = [_TASK_DISPLAY.get(rule, rule) for rule in _TASK_DISPLAY if rule not in tasks]
    left_out = ", ".join(f"{rep} {count}" for rep, count in entry["n_unspecific_cells"].items()
                         if count)
    _save_fig(fig, _multitask_out("row_cluster_same_task.png"),
              extra=(f"  {len(data['experiments'])} {data['feature']} runs, {len(tasks)} tasks"
                     + _row_sharing_untrained_note(untrained)
                     + (f"; no period pairs: {', '.join(missing)}" if missing else "")
                     + (f"; cells at chance >= 95% left out: {left_out}" if left_out else "")))


def plot_row_cluster_same_period():
    """
    Figure: multitask_row_cluster_same_period.png — the per-period view of
    the left panel of row_cluster_organization: for each trial period, how
    often the rows of two different tasks in it share a row cluster.

    One group of three bars per period (input neurons, hidden neurons,
    synapses), each the % of (run, task pair) cells whose two rows share a
    row cluster. The dark dash is the chance: the fraction of a row of the
    one task and a row of the other from two different periods sharing a
    cluster, averaged over the period's cells, so how alike the two tasks are
    anyway is controlled for (the mirror of row_cluster_same_task's
    baseline). Cells at chance >= 95% are left out. Stars, the untrained
    networks' hollow bars and the input files are as in row_cluster_same_task;
    the period labels carry the heatmap phase colors.
    """
    _ensure_out_dir()
    data = _load_row_cluster_sharing()
    if data is None:
        return
    entry = data["same_period_all"]
    periods = list(entry["periods"])
    if not periods:
        print("  Skipped: the result holds no same-period cells.")
        return
    untrained = _load_untrained_row_sharing()
    control = {} if untrained is None else untrained["same_period_all"]["periods"]
    fig, ax = _row_sharing_bar_figure(
        [(entry["periods"][phase], control.get(phase, {})) for phase in periods],
        [_PHASE_DISPLAY.get(phase, phase) for phase in periods],
        "Task pairs of one period\nin one row cluster", untrained is not None)
    for label, phase in zip(ax.get_xticklabels(), periods):
        if phase in _PHASE_COLORS:
            label.set_bbox(dict(facecolor=_PHASE_COLORS[phase], edgecolor="none",
                                boxstyle="round,pad=0.15", alpha=0.8))
    left_out = ", ".join(f"{rep} {count}" for rep, count in entry["n_unspecific_cells"].items()
                         if count)
    _save_fig(fig, _multitask_out("row_cluster_same_period.png"),
              extra=(f"  {len(data['experiments'])} {data['feature']} runs, "
                     f"{len(periods)} periods" + _row_sharing_untrained_note(untrained)
                     + (f"; cells at chance >= 95% left out: {left_out}" if left_out else "")))


# ─── Figure: Cluster tuning vs lesion effect ─────────────────────────────────
# LESION_NORM_DIR (multiple_tasks_norm/{LESION_ANAME}) is set by _set_multitask_l2.
CLUSTER_CORR_SCHEMA_VERSION = 2
CLUSTER_CORR_X_DEFINITION = "pearson_corr_of_cluster_mean_tuning_profiles"
CLUSTER_CORR_Y_DEFINITION = "one_minus_pearson_corr_of_z_scored_effect_profiles"
# One size for every lesion-mode scatter figure (tuning vs lesion, OM vs lesion,
# causal vs activity and the per-run `_seeds` dot plots), so they tile as equal
# squares; the task-specificity count bars keep their own size below.
_LESION_SCATTER_FIGSIZE = (2.5, 2.5)
_TASK_SPECIFICITY_FIGSIZE = (3, 2.8)


def _seeds_legend(ax, handles, n_runs, note=None):
    """Legend of the per-run `_seeds` figures, placed above the axes (a 2.5 in
    panel has no empty corner for a box) with the run count as its title;
    `note` is appended to that title and may hold its own line breaks.
    tight_layout reserves the legend's width for the axes, so every title
    line must stay within the two-column handle row (about 1.4 in at 6 pt)
    or the axes shrink."""
    title = f"{n_runs} runs" + (f"; {note}" if note else "")
    return _legend(ax, handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                   borderaxespad=0.0, ncol=2, columnspacing=0.8, handletextpad=0.3,
                   fontsize=6, frameon=False, title=title, title_fontsize=6)


def _rank_association_label(association, unit=""):
    """Legend text for a saved Spearman label-permutation test; nothing is refitted."""
    rho = float(association["rho"])
    p_perm = float(association["p_perm"])
    if not (np.isfinite(rho) and np.isfinite(p_perm)):
        raise ValueError("non-finite saved rank statistics")
    p_text = "p < 0.001" if p_perm < 0.001 else f"p = {p_perm:.3f}"
    suffix = f" {unit}" if unit else ""
    # Two short lines: a one-line label is wider than the 2.5 in scatter axes and
    # would make tight_layout squeeze the plot to make room for it.
    return (rf"$\rho$ = {rho:.2f} (n = {int(association['n_clusters'])}{suffix})"
            "\n" f"permutation {p_text}")


def _saved_association(association):
    """Validate and copy the fields a legend needs from a saved association."""
    if association is None:
        raise ValueError("missing saved association")
    copied = {key: float(association[key]) for key in ("rho", "p_perm")}
    copied["n_clusters"] = int(association["n_clusters"])
    if association.get("statistic") != "spearman" or "cluster" not in str(
            association.get("permutation_unit", "")):
        raise ValueError("saved association is not a cluster-permutation Spearman test")
    return copied


def _saved_binned_medians(medians):
    """Validate and copy a saved quantile-bin summary (median x, median y with
    quartiles and count per bin); None when the cache predates it."""
    if medians is None:
        return None
    copied = {key: np.asarray(medians[key], dtype=float)
              for key in ("x", "y", "y_q25", "y_q75")}
    counts = np.asarray(medians["counts"])
    size = copied["x"].size
    if (size == 0 or any(value.shape != (size,) for value in copied.values())
            or counts.shape != (size,)
            or not all(np.isfinite(value).all() for value in copied.values())
            or np.any(counts < 1)
            or np.any(copied["y_q25"] > copied["y"]) or np.any(copied["y"] > copied["y_q75"])):
        raise ValueError("invalid saved binned medians")
    copied["counts"] = counts.astype(int)
    return copied


def _read_cluster_corr_entry(name, data, expected_entry, aname=None):
    """Validate one saved cluster-correlation entry and extract plot inputs.

    Only `schema_version` 2 caches are accepted: they hold the scale-free
    comparison (tuning profile correlation against one minus the correlation
    of z-scored lesion profiles, over clusters with a significant effect).
    Older caches with OLS regressions are rejected so they are regenerated by
    lesion_plot.py instead of being drawn silently. Entries must come from
    `aname` (default LESION_ANAME; the per-run `_seeds` figure passes each
    sibling run).
    """
    aname = LESION_ANAME if aname is None else aname
    if data.get("schema_version") != CLUSTER_CORR_SCHEMA_VERSION:
        raise ValueError("legacy cache without the scale-free comparison")
    if expected_entry is not None:
        exclude_last = "unnormalized" in expected_entry
        # The entry name ends in its lesion mode ("..._zero-W", "..._freeze-M").
        expected_mode = expected_entry.rsplit("_", 1)[1].replace("-", "_")
        if (name != expected_entry
                or data.get("exclude_last_cluster") is not exclude_last
                or data.get("mod_lesion_mode", expected_mode) != expected_mode):
            raise ValueError("modulation variant, mode or cluster exclusion does not match")
    if (data.get("x_definition") != CLUSTER_CORR_X_DEFINITION
            or data.get("y_definition") != CLUSTER_CORR_Y_DEFINITION
            or data.get("aname", aname) != aname):
        raise ValueError("saved metric or run does not match")
    x = np.asarray(data["tuning_corr"], dtype=float)
    y = np.asarray(data["lesion_profile_dissim"], dtype=float)
    if (x.ndim != 1 or x.size == 0 or x.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid saved scatter coordinates")
    return {"x": x, "y": y, "association": _saved_association(data["association"]),
            "binned_medians": _saved_binned_medians(data.get("binned_medians"))}


def plot_cluster_corr_vs_lesion():
    """
    Figure: cluster tuning-profile correlation vs lesion-profile similarity.

    Draws the coordinates and rank statistics saved by lesion_plot.py, seen
    from the similarity side: x is the Pearson correlation between cluster mean
    tuning profiles, y the Pearson correlation between per-task lesion effects
    z-scored against their random-control repeats (the caches store 1 - r; the
    renderer flips y, the sign of the saved Spearman rho and the saved bin
    medians, see _plot_cluster_corr_vs_lesion_variants), over clusters whose
    lesion effect exceeds the control null. The legend reports Spearman rho
    with a cluster-label permutation p, because cluster pairs share clusters.
    The saved quantile-bin medians (with their interquartile range) describe
    the shape of the relation, which is concentrated among the most similarly
    tuned pairs; they are descriptive, not the statistic.

    Produces input/hidden figures for both clustering variants, plus three
    modulation figures: normalized, var-weighted unnormalized (W*Var(M)) and
    abs-weighted unnormalized (|W|*Var(M)); Var(W*M) has its own entry point,
    plot_cluster_corr_vs_lesion_weighted. Every modulation figure uses the
    lesion heatmap's intervention, LESION_HEATMAP_MODE (freeze_M), and only
    that exact saved variant/mode entry, never the other mode as a
    substitute; its unnormalized variants drop the unresponsive cluster. All
    entries come from LESION_ANAME's caches. Legacy caches are skipped, not
    refitted, and the L1 supplement kept in the caches is not drawn here.
    """
    variants = [
        ("normalized_lesion_effect", None, None),
        ("normalized_lesion_effect_unnorm", None, None),
        _cluster_corr_modulation_variant("normalized", "norm"),
        _cluster_corr_modulation_variant("var-weighted-unnormalized", "var_weighted_unnorm"),
        _cluster_corr_modulation_variant("abs-weighted-unnormalized", "abs_weighted_unnorm"),
    ]
    _plot_cluster_corr_vs_lesion_variants(variants)


def plot_cluster_corr_vs_lesion_weighted():
    """Plot Var(W*M) tuning versus LESION_HEATMAP_MODE (freeze_M) lesion
    profiles from its exact saved cache.

    The weighted-unnormalized features are variance after multiplication by W,
    unlike var-weighted features W*Var(M). Keep the saved cluster memberships,
    last-cluster exclusion and rank statistics. Exports only the main figure,
    separately from the existing modulation figures.
    """
    _plot_cluster_corr_vs_lesion_variants([
        _cluster_corr_modulation_variant("weighted-unnormalized", "weighted_unnorm"),
    ])


def _cluster_corr_modulation_variant(type_tag, output_tag):
    """(cache suffix, output name, expected entry) of one modulation clustering
    type under LESION_HEATMAP_MODE, so every synapse tuning-vs-lesion figure
    reads the lesion heatmap's intervention. Cache names spell the mode with a
    hyphen ("freeze-M"), output names as the mode ("freeze_M")."""
    cache_mode = LESION_HEATMAP_MODE.replace("_", "-")
    return (f"mod_lesion_effect_{type_tag}_{cache_mode}",
            f"modulation_{output_tag}_{LESION_HEATMAP_MODE}",
            f"{type_tag}_{cache_mode}")


def _plot_cluster_corr_vs_lesion_variants(variants):
    """Render exact saved scatter variants without recomputing their analysis."""
    _ensure_out_dir()
    if not LESION_NORM_DIR.exists():
        print("  Skipped: multiple_tasks_norm dir not found. Run lesion_plot.py first.")
        return

    for suffix, output_name, expected_entry in variants:
        pkl_path = resolve_lesion_cache_path(
            LESION_NORM_DIR / f"cluster_corr_vs_{suffix}_{LESION_ANAME}.pkl")
        if not pkl_path.exists():
            print(f"  Skipped: {pkl_path.name} not found.")
            continue
        scatter_data = load_lesion_pickle(pkl_path)

        for name, data in scatter_data.items():
            try:
                entry = _read_cluster_corr_entry(name, data, expected_entry)
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                print(f"  Skipped {name}: incomplete or incompatible cache ({error}); "
                      "run multiple_task/lesion_plot.py again.")
                continue

            weighted = (expected_entry or "").startswith("weighted-unnormalized_")
            tuning_word = r"$\mathrm{Var}(WM)$ tuning" if weighted else "Tuning"
            # e.g. "input_normalized_k20" -> "input_norm"
            clean_name = output_name or name.replace("_normalized", "_norm").replace(
                "_unnormalized", "_unnorm").replace("_k20", "")

            # The caches store the lesion-profile DISSIMILARITY 1 - r and the
            # rank statistics of that quantity. The figure shows the similarity
            # r instead, which is the same comparison seen from the other side:
            # y flips to 1 - y, the Spearman rho changes sign (rank order
            # reversed; the two-sided permutation p is unchanged) and each bin
            # median flips with its quartiles swapped. Nothing is refitted.
            similarity = 1.0 - entry["y"]
            association = dict(entry["association"])
            association["rho"] = -association["rho"]
            fig, ax = plt.subplots(1, 1, figsize=_LESION_SCATTER_FIGSIZE)
            handles = []
            if entry["binned_medians"] is None:
                print(f"  Note: {name} cache has no saved binned medians; "
                      "rerun multiple_task/lesion_plot.py to add them.")
            else:
                handles.append(_draw_similarity_bin_medians(ax, entry["binned_medians"]))
            _stat_scatter(ax, entry["x"], similarity, _rank_association_label(association),
                          legend_handles=handles, xlabel=f"{tuning_word} correlation",
                          ylabel="Lesion similarity (r)", point_color=_LESION_POINT_COLOR)
            fig.tight_layout()
            _save_fig(fig, _multitask_out(f"cluster_corr_vs_lesion_{clean_name}.png"),
                      extra=f" [{LESION_ANAME}]")


def _draw_similarity_bin_medians(ax, medians):
    """Draw saved quantile-bin medians of the lesion DISSIMILARITY on the
    similarity axis: 1 - median per bin, with the interquartile range as a
    bar (the quartiles swap under the flip). Returns the legend handle."""
    center = 1.0 - medians["y"]
    lower = 1.0 - medians["y_q75"]
    upper = 1.0 - medians["y_q25"]
    return ax.errorbar(medians["x"], center, yerr=[center - lower, upper - center],
                       fmt="o-", markersize=3.5, elinewidth=0.8, capsize=0,
                       label="Bin median, IQR", **_TREND_LINE_STYLE)


_CLUSTER_CORR_SEED_SIDES = ("hidden", "input")
_CLUSTER_CORR_SEED_LABELS = {"hidden": "Hidden", "input": "Input"}


def _cluster_corr_seed_runs(sides=_CLUSTER_CORR_SEED_SIDES):
    """Saved neuron-cluster tuning-vs-lesion associations of every sibling run.

    Reads each run's `cluster_corr_vs_normalized_lesion_effect_unnorm` cache
    (unnormalized clustering, fixed k = 20) and returns {aname: {side:
    association}} with rho on the similarity sign the figures use. A run is
    kept only when every side validates, so hidden and input always come from
    the same runs; legacy or incomplete caches are skipped with a note."""
    runs = {}
    for directory in _sibling_run_dirs():
        aname = directory.name
        path = resolve_lesion_cache_path(
            directory / f"cluster_corr_vs_normalized_lesion_effect_unnorm_{aname}.pkl")
        if not path.exists():
            continue
        saved = load_lesion_pickle(path)
        entries = {}
        for side in sides:
            name = f"{side}_unnormalized_k20"
            try:
                entry = _read_cluster_corr_entry(name, saved[name], None, aname=aname)
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                print(f"  Skipped {aname} {name}: incomplete or incompatible cache ({error}).")
                entries = None
                break
            association = dict(entry["association"])
            association["rho"] = -association["rho"]      # dissimilarity -> similarity
            entries[side] = association
        if entries:
            runs[aname] = entries
    return runs


def _wilcoxon_vs_zero(values, alternative="two-sided"):
    """Wilcoxon signed-rank p of per-run statistics against zero, the
    cohort-level test of the per-run figures: two-sided by default, or
    `alternative` ("less", "greater") when the per-run test is itself
    directional. NaN below two runs or when every value is zero. Returns
    (p, legend text)."""
    from scipy.stats import wilcoxon

    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.all(values == 0):
        return np.nan, "p = n/a"
    p_value = float(wilcoxon(values, alternative=alternative).pvalue)
    return p_value, ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}")


def _plot_cluster_corr_seed_rhos(runs, out_name):
    """Renderer of the per-run rho figure: one joined hidden/input pair per
    run (filled when its saved permutation p is below 0.05), the median over
    runs as a dash, and in the legend title the two-sided Wilcoxon signed-rank
    p of each cluster type's rhos against zero (the axes are too narrow to
    hold a p under each tick label)."""
    sides = _CLUSTER_CORR_SEED_SIDES
    fig, ax = plt.subplots(1, 1, figsize=_LESION_SCATTER_FIGSIZE)
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
    positions = np.arange(len(sides))
    for run_index, entries in enumerate(runs.values()):
        offset = (run_index - (len(runs) - 1) / 2) * 0.03
        ax.plot(positions + offset, [entries[side]["rho"] for side in sides],
                color="0.75", linewidth=0.6, zorder=2)
        for position, side in zip(positions, sides):
            _significance_marker(ax, position + offset, entries[side]["rho"],
                                 entries[side]["p_perm"] < 0.05,
                                 color=_LESION_POINT_COLOR, hollow_edge=_LESION_POINT_COLOR)
    cohort_tests, test_texts = {}, []
    for position, side in zip(positions, sides):
        rhos = [entries[side]["rho"] for entries in runs.values()]
        ax.hlines(np.median(rhos), position - 0.22, position + 0.22, color=_TREND_LINE_COLOR,
                  linewidth=1.8, zorder=4)
        cohort_tests[side], p_text = _wilcoxon_vs_zero(rhos)
        test_texts.append(f"{side} {p_text}")
    ax.set_xticks(positions)
    ax.set_xticklabels([_CLUSTER_CORR_SEED_LABELS[side] for side in sides],
                       fontsize=_STAT_SCATTER_TICK_FONTSIZE)
    ax.set_xlim(-0.6, len(positions) - 0.4)
    ax.set_ylabel(r"Spearman $\rho$" + "\n(tuning vs lesion similarity)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=_STAT_SCATTER_TICK_FONTSIZE)
    _seeds_legend(ax, _significance_legend_handles(_LESION_POINT_COLOR, _LESION_POINT_COLOR),
                  len(runs),
                  note="Wilcoxon signed-rank vs 0:\n" + ", ".join(test_texts))
    fig.tight_layout()
    summary = ", ".join(
        f"{side}: {sum(e[side]['rho'] > 0 for e in runs.values())}/{len(runs)} rho > 0, "
        f"{sum(e[side]['p_perm'] < 0.05 for e in runs.values())} significant, "
        f"Wilcoxon p = {cohort_tests[side]:.3f}"
        for side in sides)
    _save_fig(fig, _multitask_out(out_name), extra="  " + summary)


def plot_cluster_corr_vs_lesion_seeds():
    """
    Figure: the cluster-level tuning-vs-lesion Spearman rho of every sibling
    run of LESION_ANAME, hidden and input neuron clusters side by side.

    The single-run `cluster_corr_vs_lesion` scatters have only 8-17 clusters
    each, so one run rarely reaches significance; this figure shows the whole
    cohort. Each run contributes one point per cluster type (rho of tuning
    correlation vs lesion-profile similarity, sign-flipped from the cached
    dissimilarity), joined by a thin line; filled markers have a saved
    cluster-label permutation p below 0.05, hollow ones do not. The dash is
    the median over runs, and the tick label carries the two-sided Wilcoxon
    signed-rank p of the runs' rhos against zero, the cohort-level test. Only
    saved per-run statistics enter; the saved message counts the runs with
    positive and with significant rho.
    """
    _ensure_out_dir()
    runs = _cluster_corr_seed_runs()
    if len(runs) < 2:
        print(f"  Skipped: {len(runs)} run(s) with saved cluster-level caches; need >= 2.")
        return
    _plot_cluster_corr_seed_rhos(runs, "cluster_corr_vs_lesion_seeds.png")


def _sibling_run_dirs():
    """Directories of LESION_ANAME's sibling runs (same feature, any seed),
    i.e. every post-processed run of the selected MULTITASK_L2 cohort."""
    import re as _re

    _check_same_cohort(MULTITASK_L2, lesion=LESION_ANAME)
    pattern = _re.sub(r"seed\d+", "seed*", LESION_ANAME)
    return sorted(LESION_NORM_DIR.parent.glob(pattern))


# ─── Figure: Causal vs activity task organization ────────────────────────────

CAUSAL_VS_ACTIVITY_SIDES = ("hidden", "input", "modulation")
# A run enters the seeds figure once both neuron sides are saved; the synapse
# side, added later to lesion_plot.py, is drawn for the runs that have it.
_CAUSAL_VS_ACTIVITY_NEURON_SIDES = ("hidden", "input")
_CAUSAL_VS_ACTIVITY_X_DEFINITION = "pearson_corr_of_task_mean_activity_variance_profiles"
_CAUSAL_VS_ACTIVITY_Y_DEFINITION = "pearson_corr_of_task_lesion_effect_profiles_over_neuron_clusters"
_CAUSAL_VS_ACTIVITY_MOD_X_DEFINITION = "pearson_corr_of_task_mean_abs_modulation_variance_profiles"
_CAUSAL_VS_ACTIVITY_MOD_Y_DEFINITION = "pearson_corr_of_task_lesion_effect_profiles_over_synapse_clusters"
_CAUSAL_VS_ACTIVITY_SIDE_LABELS = {"hidden": "Hidden activity", "input": "Input activity",
                                   "modulation": "Modulation"}
# Unnormalized neuron clusters and raw task-variance features on both neuron
# sides, matching the lesion heatmap and the hidden tuning-vs-lesion scatter.
_CAUSAL_VS_ACTIVITY_VARIANT = "unnormalized"
# Metadata each side's cache must carry. The synapse side pairs the
# var-weighted features (by magnitude) with the freeze_M lesion effects over
# that clustering's synapse clusters, the intervention of the lesion heatmap.
_CAUSAL_VS_ACTIVITY_NEURON_FIELDS = {"neuron_variant": _CAUSAL_VS_ACTIVITY_VARIANT,
                                     "x_definition": _CAUSAL_VS_ACTIVITY_X_DEFINITION,
                                     "y_definition": _CAUSAL_VS_ACTIVITY_Y_DEFINITION}
_CAUSAL_VS_ACTIVITY_CACHE_FIELDS = {
    "hidden": _CAUSAL_VS_ACTIVITY_NEURON_FIELDS,
    "input": _CAUSAL_VS_ACTIVITY_NEURON_FIELDS,
    "modulation": {"modulation_type": "modulation_all_var_weighted_unnormalized",
                   "mod_lesion_mode": LESION_HEATMAP_MODE,
                   "x_definition": _CAUSAL_VS_ACTIVITY_MOD_X_DEFINITION,
                   "y_definition": _CAUSAL_VS_ACTIVITY_MOD_Y_DEFINITION},
}


def _read_causal_vs_activity(path, side, aname):
    """Validate one saved task-level Mantel cache and return its plot inputs."""
    saved = load_lesion_pickle(path)
    if saved.get("schema_version") != 2:
        raise ValueError("cache predates the unnormalized pairing; rerun lesion_plot.py")
    if saved.get("aname") != aname or saved.get("side") != side:
        raise ValueError("cache belongs to another run or side")
    for field, expected in _CAUSAL_VS_ACTIVITY_CACHE_FIELDS[side].items():
        if saved.get(field) != expected:
            raise ValueError(f"cache has {field} = {saved.get(field)!r}, expected {expected!r}")
    x = np.asarray(saved["activity_pairs"], dtype=float)
    y = np.asarray(saved["causal_pairs"], dtype=float)
    n_tasks = len(saved["tasks"])
    if (x.ndim != 1 or x.shape != y.shape or x.size != n_tasks * (n_tasks - 1) // 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid saved task-pair similarities")
    association = _saved_association(saved["association"])
    if association["n_clusters"] != n_tasks:
        raise ValueError("permutation unit count does not match the tasks")
    trend = saved.get("trend_line")
    if trend is not None:
        trend = {key: float(trend[key]) for key in ("slope", "intercept")}
        if not all(np.isfinite(value) for value in trend.values()):
            raise ValueError("non-finite saved trend line")
    return {"x": x, "y": y, "association": association, "trend_line": trend,
            "n_tasks": n_tasks}


def _causal_vs_activity_path(directory, side, aname):
    return resolve_lesion_cache_path(
        Path(directory) / f"causal_vs_activity_tasksim_{side}_{aname}.pkl")


def plot_causal_vs_activity():
    """
    Figure: do tasks that look alike in activity depend on the same clusters?

    One figure per side (hidden, input, modulation). Each point is a pair of
    tasks: x is the Pearson correlation between the two tasks' mean
    task-variance profiles over that side's units, y the correlation between
    their lesion-effect profiles. The neuron sides compare activity over
    neurons with lesions of all input and hidden neuron clusters, both in the
    unnormalized variant (raw task-variance features, unnormalized neuron
    clusters). The modulation side compares |W| Var(M) over synapses with
    freeze_M lesions of the var-weighted synapse clusters, the unresponsive
    class left out. Coordinates, the two-sided Spearman task-label
    permutation test and the OLS guide line are read from lesion_plot.py's
    cache for LESION_ANAME; the style matches the cluster-level
    tuning-vs-lesion scatters. Caches of another variant are skipped.
    """
    _ensure_out_dir()
    for side in CAUSAL_VS_ACTIVITY_SIDES:
        path = _causal_vs_activity_path(LESION_NORM_DIR, side, LESION_ANAME)
        if not path.exists():
            print(f"  Skipped: {path.name} not found. Run multiple_task/lesion_plot.py first.")
            continue
        try:
            entry = _read_causal_vs_activity(path, side, LESION_ANAME)
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"  Skipped {side}: incompatible {path.name} ({error}); "
                  "run multiple_task/lesion_plot.py again.")
            continue
        fig, ax = plt.subplots(1, 1, figsize=_LESION_SCATTER_FIGSIZE)
        _stat_scatter(ax, entry["x"], entry["y"],
                      _rank_association_label(entry["association"], unit="tasks"),
                      trend_line=entry["trend_line"],
                      # Two lines each: the square axes box is about 1.5 in wide,
                      # narrower than either label on one line.
                      xlabel=f"Similarity:\n{_CAUSAL_VS_ACTIVITY_SIDE_LABELS[side].lower()}",
                      ylabel="Similarity:\nlesion profiles", point_color=_LESION_POINT_COLOR)
        fig.tight_layout()
        _save_fig(fig, _multitask_out(f"causal_vs_activity_tasks_{side}.png"),
                  extra=f" [{LESION_ANAME}] {entry['n_tasks']} tasks")


def _causal_vs_activity_runs():
    """Sibling runs of LESION_ANAME (same feature, any seed) with both neuron
    sides saved, each with the sides it has; a missing or unreadable
    modulation cache only leaves that side out."""
    runs = {}
    for directory in _sibling_run_dirs():
        entries = {}
        for side in CAUSAL_VS_ACTIVITY_SIDES:
            path = _causal_vs_activity_path(directory, side, directory.name)
            if not path.exists():
                continue
            try:
                entries[side] = _read_causal_vs_activity(path, side, directory.name)
            except (OSError, KeyError, TypeError, ValueError) as error:
                print(f"  Skipped {directory.name} ({side}): {error}.")
        if all(side in entries for side in _CAUSAL_VS_ACTIVITY_NEURON_SIDES):
            runs[directory.name] = entries
    return runs


# Layout of the seeds figure. Hidden and input share their lesion basis (the
# same neuron-cluster lesion profiles; only the activity side differs), so they
# sit side by side with each run's pair joined: input is the control for
# hidden. Modulation pairs other features with other lesions, so it stands
# apart behind a dotted divider and is not joined.
_CAUSAL_SEEDS_POSITIONS = {"hidden": 0.0, "input": 1.0, "modulation": 2.4}
_CAUSAL_SEEDS_DIVIDER = 1.7
_CAUSAL_SEEDS_JITTER = 0.15        # max horizontal spread of a column's points
_CAUSAL_SEEDS_MEDIAN_HALF_WIDTH = 0.27
_CAUSAL_SEEDS_POINT_SIZE = 20
_CAUSAL_SEEDS_LEGEND_FIGSIZE = (1.6, 0.7)
# Side of the panel's square axes box, in inches: the box tight_layout gives
# the single-run causal scatters at _LESION_SCATTER_FIGSIZE (their statistics
# legend takes the top), so the seeds panel beside them is drawn at one scale.
# test_causal_vs_activity_figures checks the two still agree.
_CAUSAL_SEEDS_BOX_INCHES = 1.49
_CAUSAL_SEEDS_BOX_ORIGIN = (0.3, 0.2)    # lower-left corner, figure fraction


def _causal_seeds_legend_handles():
    """Legend entries of the seeds figure: significant and non-significant
    runs, and the red median dash with its interquartile bar."""
    filled = mpl.lines.Line2D([], [], marker="o", linestyle="none", color=_LESION_POINT_COLOR,
                              markeredgecolor="none", markersize=4.5, label="permutation p < 0.05")
    hollow = mpl.lines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                              markeredgecolor=_LESION_POINT_COLOR, markeredgewidth=0.8,
                              markersize=4.5, label=r"permutation p $\geq$ 0.05")
    summary = mpl.lines.Line2D([], [], color=_TREND_LINE_COLOR, linewidth=1.8, marker="|",
                               markersize=6, markeredgewidth=1.0, label="median, IQR")
    return [filled, hollow, summary]


def _save_causal_seeds_legend(out_path):
    """Save the seeds figure's legend as its own small figure (only when
    legends are on), so the panel keeps the scatters' size."""
    if not SHOW_LEGEND:
        return
    fig = plt.figure(figsize=_CAUSAL_SEEDS_LEGEND_FIGSIZE)
    fig.legend(handles=_causal_seeds_legend_handles(), loc="center", frameon=False,
               fontsize=6, handletextpad=0.4, labelspacing=0.5)
    _save_fig(fig, out_path, extra="  (legend of causal_vs_activity_seeds)")


def plot_causal_vs_activity_seeds():
    """
    Figure: the task-level Mantel rho of every sibling run, hidden vs input
    vs modulation.

    One column per side, one point per run: filled when the run's saved
    permutation p is below 0.05, hollow otherwise, spread horizontally by a
    fixed random jitter (the runs have no order). A thin line joins each
    run's hidden and input points, the controlled pair; the bracket above
    them gives the paired Wilcoxon signed-rank p of hidden minus input over
    runs, the cohort-level test computed here from the saved rhos. Modulation
    stands apart (other features and lesions) and appears once any run has
    its cache. Each column carries its median (dash) and interquartile range
    (bar). Only saved per-run statistics are drawn.

    The panel has the size and square axes box of the single-run scatters
    (causal_vs_activity_tasks_*), so the two sit side by side as equals; its
    legend is saved as its own figure, causal_vs_activity_seeds_legend.png.
    """
    _ensure_out_dir()
    runs = _causal_vs_activity_runs()
    if len(runs) < 2:
        print(f"  Skipped: {len(runs)} run(s) with saved task-level Mantel caches; need >= 2.")
        return
    sides = [side for side in CAUSAL_VS_ACTIVITY_SIDES
             if any(side in entries for entries in runs.values())]
    associations = {side: [entries[side]["association"] for entries in runs.values()
                           if side in entries]
                    for side in sides}
    rhos = {side: np.array([saved["rho"] for saved in saved_list])
            for side, saved_list in associations.items()}
    significant = {side: np.array([saved["p_perm"] < 0.05 for saved in saved_list])
                   for side, saved_list in associations.items()}
    rng = np.random.default_rng(0)
    xs = {side: _CAUSAL_SEEDS_POSITIONS[side]
          + rng.uniform(-_CAUSAL_SEEDS_JITTER, _CAUSAL_SEEDS_JITTER, rhos[side].size)
          for side in sides}

    fig, ax = plt.subplots(1, 1, figsize=_LESION_SCATTER_FIGSIZE)
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
    if "modulation" in sides:
        ax.axvline(_CAUSAL_SEEDS_DIVIDER, color="0.7", linewidth=0.6, linestyle=":", zorder=1)
    # Every run has both neuron sides (_causal_vs_activity_runs), in run order.
    for run in range(len(runs)):
        ax.plot([xs["hidden"][run], xs["input"][run]],
                [rhos["hidden"][run], rhos["input"][run]],
                color="0.85", linewidth=0.5, zorder=2)
    for side in sides:
        hit = significant[side]
        ax.scatter(xs[side][hit], rhos[side][hit], s=_CAUSAL_SEEDS_POINT_SIZE,
                   color=_LESION_POINT_COLOR, edgecolors="none", zorder=3)
        ax.scatter(xs[side][~hit], rhos[side][~hit], s=_CAUSAL_SEEDS_POINT_SIZE,
                   facecolors="white", edgecolors=_LESION_POINT_COLOR, linewidths=0.8,
                   zorder=3)
        position = _CAUSAL_SEEDS_POSITIONS[side]
        q25, median, q75 = np.percentile(rhos[side], [25, 50, 75])
        ax.vlines(position, q25, q75, color=_TREND_LINE_COLOR, linewidth=1.0, zorder=4)
        ax.hlines(median, position - _CAUSAL_SEEDS_MEDIAN_HALF_WIDTH,
                  position + _CAUSAL_SEEDS_MEDIAN_HALF_WIDTH,
                  color=_TREND_LINE_COLOR, linewidth=1.8, zorder=4)

    # Paired test of hidden against its input control, as a bracket.
    p_paired, p_text = _wilcoxon_vs_zero(rhos["hidden"] - rhos["input"])
    values = np.concatenate(list(rhos.values()))
    span = max(values.max() - values.min(), 1e-6)
    base = max(rhos["hidden"].max(), rhos["input"].max()) + 0.06 * span
    top = base + 0.04 * span
    ax.plot([0.0, 0.0, 1.0, 1.0], [base, top, top, base], color="0.3", linewidth=0.7,
            zorder=4)
    # Only the p sits on the bracket (the test belongs in the caption), so the
    # label stays within the bracket and clear of the y tick labels.
    ax.text(0.5, top + 0.01 * span, p_text, ha="center", va="bottom", fontsize=6)
    bottom = min(values.min(), 0.0) - 0.06 * span
    ax.set_ylim(bottom, top + 0.14 * span)
    # The scatter family's tick steps and labels, with no tick beside the
    # bracket, which is not on the rho scale; one more interval than the
    # scatters allow covers the bracket's headroom, which then drops its ticks.
    ax.yaxis.set_major_locator(mticker.MaxNLocator(_STAT_SCATTER_MAX_TICKS + 1,
                                                   steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_short_tick))
    ax.set_yticks([tick for tick in ax.get_yticks() if bottom <= tick <= base])

    ax.set_xticks([_CAUSAL_SEEDS_POSITIONS[side] for side in sides])
    ax.set_xticklabels([side.capitalize() for side in sides],
                       fontsize=_STAT_SCATTER_TICK_FONTSIZE)
    ax.set_xlim(-0.55, max(_CAUSAL_SEEDS_POSITIONS[side] for side in sides) + 0.55)
    ax.set_ylabel(r"Task-level Mantel $\rho$" + "\n(activity vs lesion similarity)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=_STAT_SCATTER_TICK_FONTSIZE)
    # A fixed square box instead of tight_layout, which would hand this panel
    # (no legend on top) a larger box than the scatters; the saved image is
    # cropped to its content, so the margins only have to hold the labels.
    width, height = _LESION_SCATTER_FIGSIZE
    ax.set_position([*_CAUSAL_SEEDS_BOX_ORIGIN,
                     _CAUSAL_SEEDS_BOX_INCHES / width, _CAUSAL_SEEDS_BOX_INCHES / height])
    _save_causal_seeds_legend(_multitask_out("causal_vs_activity_seeds_legend.png"))
    _save_fig(fig, _multitask_out("causal_vs_activity_seeds.png"),
              extra="  " + ", ".join(
                  f"{side}: {int(np.sum(rhos[side] > 0))}/{rhos[side].size} rho > 0, "
                  f"{int(significant[side].sum())} significant, "
                  f"median {np.median(rhos[side]):.2f}"
                  for side in sides)
              + f"; hidden vs input paired Wilcoxon p = {p_paired:.2g}")


# ─── Figure: Task specificity ────────────────────────────────────────────────

_TASK_SPECIFICITY_TYPES = ("input", "hidden", "modulation")
_TASK_SPECIFICITY_LABELS = {"input": "Input neuron clusters", "hidden": "Hidden neuron clusters",
                            "modulation": "Synapse clusters"}
_TASK_SPECIFICITY_COLORS = {"input": c_vals[1], "hidden": c_vals[0], "modulation": c_vals[4]}
_TASK_SPECIFICITY_TICK_LABELS = {"input": "Input\nclusters", "hidden": "Hidden\nclusters",
                                 "modulation": "Synapse\nclusters"}


def _load_task_specificity():
    """Load the per-cluster impaired-task counts and their dispersion test
    (observed variance, the mixing null's variances and the permutation p)
    from lesion_plot.py's task-specificity cache for LESION_ANAME, or None
    with a note. The cache also holds per-relation task-pair sharing
    statistics, which no paper figure draws, so they are not read."""
    path = resolve_lesion_cache_path(LESION_NORM_DIR / f"task_specificity_{LESION_ANAME}.pkl")
    if not path.exists():
        print(f"  Skipped: {path.name} not found. Run multiple_task/lesion_plot.py first.")
        return None
    try:
        saved = load_lesion_pickle(path)
        if saved.get("schema_version") != 1 or saved.get("aname") != LESION_ANAME:
            raise ValueError("cache schema or run does not match")
        tasks = list(saved["tasks"])
        types = {}
        for type_name, entry in saved["types"].items():
            if type_name not in _TASK_SPECIFICITY_TYPES:
                raise ValueError(f"unknown cluster type {type_name!r}")
            dispersion = entry["dispersion"]
            counts = np.asarray(dispersion["counts"], dtype=int)
            p_disp = float(dispersion["p_perm"])
            null_var = np.asarray(dispersion["null_var"], dtype=float)
            if (counts.ndim != 1 or counts.size == 0 or counts.min() < 0
                    or counts.max() > len(tasks) or not np.isfinite(p_disp)
                    or dispersion.get("side") != "greater"
                    or null_var.ndim != 1 or null_var.size == 0
                    or not np.isfinite(null_var).all() or np.any(null_var < 0)):
                raise ValueError(f"{type_name}: invalid impaired-task counts or dispersion test")
            types[type_name] = {"counts": counts, "p_dispersion": p_disp,
                                "null_var_median": float(np.median(null_var))}
        if not types:
            raise ValueError("no cluster types saved")
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"  Skipped: incompatible {path.name}: {error}. Run multiple_task/lesion_plot.py again.")
        return None
    return {"tasks": tasks, "types": types}


def plot_task_specificity():
    """
    Figure: multitask_task_specificity_counts.png — how many tasks each
    cluster impairs, one column per cluster type.

    For input and hidden neuron clusters and synapse clusters (the
    unresponsive class excluded), every cluster is a dot at the number of
    tasks whose accuracy its zero_W lesion significantly lowers; tied values
    spread sideways, so each column reads as a dot histogram, and the dash is
    the median. The gray band is the spread the counts would have under the
    saved mixing null, which shuffles cluster identity within each task and so
    keeps every task's number of impaired clusters and the mean count per
    cluster: the mean ± one SD, the square root of the median null variance.
    Dots that spread beyond the band mean the type holds both specialized and
    hub-like clusters rather than a uniform mix; the tick label carries the
    cluster count and the saved one-sided permutation p of the observed
    variance exceeding the null. Everything comes from lesion_plot.py's
    cache; nothing is recomputed.
    """
    _ensure_out_dir()
    saved = _load_task_specificity()
    if saved is None:
        return
    n_tasks = len(saved["tasks"])
    types = [name for name in _TASK_SPECIFICITY_TYPES if name in saved["types"]]
    fig, ax = plt.subplots(1, 1, figsize=_TASK_SPECIFICITY_FIGSIZE)
    tick_labels, notes = [], []
    for position, type_name in enumerate(types):
        entry = saved["types"][type_name]
        counts = entry["counts"]
        null_sd = float(np.sqrt(entry["null_var_median"]))
        ax.fill_between([position - 0.3, position + 0.3], counts.mean() - null_sd,
                        counts.mean() + null_sd, color="0.85", linewidth=0, zorder=1)
        for value in np.unique(counts):
            ties = int(np.count_nonzero(counts == value))
            half_width = min(0.28, 0.07 * (ties - 1))
            ax.scatter(position + np.linspace(-half_width, half_width, ties),
                       np.full(ties, value), s=16, color=_TASK_SPECIFICITY_COLORS[type_name],
                       edgecolors="none", alpha=0.75, zorder=3)
        ax.hlines(np.median(counts), position - 0.3, position + 0.3, color=_TREND_LINE_COLOR,
                  linewidth=1.8, zorder=4)
        tick_labels.append(f"{_TASK_SPECIFICITY_TICK_LABELS[type_name]}\nn = {counts.size}\n"
                           f"{_permutation_p_text(entry['p_dispersion'])}")
        notes.append(f"{type_name}: median {np.median(counts):.0f} tasks, variance "
                     f"{counts.var():.1f} vs null {entry['null_var_median']:.1f}")
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_xlim(-0.6, len(types) - 0.4)
    ax.set_ylabel("Tasks impaired per cluster", fontsize=8)
    ax.set_ylim(-0.6, n_tasks + 0.6)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    median = mpl.lines.Line2D([], [], color=_TREND_LINE_COLOR, linewidth=1.8, label="median")
    band = mpl.patches.Patch(color="0.85", label="mixing null, mean ± SD")
    _legend(ax, handles=[median, band], loc="lower left", bbox_to_anchor=(0.0, 1.0),
            borderaxespad=0.0, ncol=2, columnspacing=0.8, handletextpad=0.4, fontsize=6,
            frameon=False)
    fig.tight_layout()
    _save_fig(fig, _multitask_out("task_specificity_counts.png"),
              extra=f" [{LESION_ANAME}] " + "; ".join(notes))


# ─── Figure: Transfer speed ──────────────────────────────────────────────────

PRETRAINING_ANALYSIS_DIR = Path("pretraining_analysis")
# Every pretraining figure reads the dmpn runs with hidden 200, L2 1e-3 and
# batch 128. pretraining.py appends two optional tags to that feature: the M
# bound when it is not the default [-1, 1] ("mb2") and the M retention per
# 40 ms step when it is not the historical 0.99 ("lam0.9", m_time_scale 400 ms
# instead of 4000 ms), giving L21e3, L21e3mb2, L21e3lam0.9 and L21e3mb2lam0.9.
_PRETRAINING_HIDDEN = 200
_PRETRAINING_FEATURE = "L21e3"
_PRETRAINING_BOUND_TAGS = {
    "mb1": "",
    "mb2": "mb2",
}
_PRETRAINING_BOUND_LIMITS = {
    "mb1": (-1, 1),
    "mb2": (-2, 2),
}
# CLI spelling of lambda -> (feature tag, m_time_scale in ms).
_PRETRAINING_LAMBDA_TAGS = {
    "0.99": "",
    "0.9": "lam0.9",
}
_PRETRAINING_LAMBDA_TIME_SCALES_MS = {
    "0.99": 4000,
    "0.9": 400,
}
_PRETRAINING_RULESET_STYLES = {
    "fdgo_delaygo": ("Irrelevant motif", _SCATTER_POINT_COLOR),
    "fdanti_delaygo": ("Relevant motif", "#e53e3e"),
    "fdanti": ("DelayAnti", "#dd6b20"),
    "fdgo": ("DelayPro", "#4c51bf"),
}
_PRETRAINING_GROUPS = {
    "motifs": ("fdgo_delaygo", "fdanti_delaygo"),
    "all": tuple(_PRETRAINING_RULESET_STYLES),
}
PRETRAINING_GROUPS = "all"
_PRETRAINING_TRAJECTORY_FIGSIZE = (3.0, 2.2 * 2 / 3)
# rule_vectors with --pretraining-groups motifs: the two pairs of bars are set
# apart by this many bar steps and separated by a dark divider, so the break
# between the two comparisons is read before the bars are.
_RULE_VECTORS_PAIR_GAP = 0.5
_RULE_VECTORS_DIVIDER_STYLE = dict(color="0.3", linewidth=1.0, zorder=1)
# x extent of that layout in bar steps (two pairs plus the gap); it is drawn at
# exactly _PRETRAINING_TRAJECTORY_FIGSIZE and other layouts scale their width
# from it so every layout keeps the same height and width per bar.
_RULE_VECTORS_MOTIFS_EXTENT = 4 + _RULE_VECTORS_PAIR_GAP
# The two-line "MemoryAnti / <-> partner" tick labels sit one bar step (about
# 0.5 in) apart at that width, which only a 5 pt font fits without overlap.
_RULE_VECTORS_TICK_FONTSIZE = 5
# Fraction of the x range left free of bars at each end; the default 0.05 is
# halved to give the tick labels room.
_RULE_VECTORS_X_MARGIN = 0.025
_TRANSFER_SPEED_FIGSIZE = (3.3, 2.2 * 2 / 3 * 1.1)
_TRANSFER_SPEED_YTICKS = (50, 75, 100)
_LEARNING_TRAJECTORY_YTICKS = (0, 50, 100)
_BACKBONE_PROBE_FIGSIZE = (3.4, 2.0)
PRETRAINING_BOUND = "mb1"
PRETRAINING_LAMBDA = "0.99"


def _pretraining_addon_name(bound, lam):
    """Filename suffix of the pretraining runs with this M bound and retention."""
    feature = (f"{_PRETRAINING_FEATURE}{_PRETRAINING_BOUND_TAGS[bound]}"
               f"{_PRETRAINING_LAMBDA_TAGS[lam]}")
    return variant_addon(_PRETRAINING_HIDDEN, feature)


PRETRAINING_ADDON_NAME = _pretraining_addon_name(PRETRAINING_BOUND, PRETRAINING_LAMBDA)


def _parse_pretraining_bound(value):
    """Normalize the user-facing mod1/mod2 aliases to repository mb1/mb2."""
    import argparse as _argparse

    normalized = str(value).strip().lower()
    aliases = {"mod1": "mb1", "mod2": "mb2", "mb1": "mb1", "mb2": "mb2"}
    if normalized not in aliases:
        raise _argparse.ArgumentTypeError(
            "expected mod1/mb1 for [-1,1] or mod2/mb2 for [-2,2]")
    return aliases[normalized]


def _set_pretraining_bound(bound):
    """Select the M bound of the pretraining runs read by every plot."""
    global PRETRAINING_BOUND, PRETRAINING_ADDON_NAME

    if bound not in _PRETRAINING_BOUND_TAGS:
        raise ValueError(f"Unknown pretraining bound: {bound!r}")
    PRETRAINING_BOUND = bound
    PRETRAINING_ADDON_NAME = _pretraining_addon_name(bound, PRETRAINING_LAMBDA)


def _set_pretraining_lambda(lam):
    """Select the M retention (lambda per step) of the pretraining runs read by every plot."""
    global PRETRAINING_LAMBDA, PRETRAINING_ADDON_NAME

    if lam not in _PRETRAINING_LAMBDA_TAGS:
        raise ValueError(f"Unknown pretraining lambda: {lam!r}")
    PRETRAINING_LAMBDA = lam
    PRETRAINING_ADDON_NAME = _pretraining_addon_name(PRETRAINING_BOUND, lam)


def _set_pretraining_groups(groups):
    """Select motif-only or motif-plus-single-task comparisons for all plots."""
    global PRETRAINING_GROUPS

    if groups not in _PRETRAINING_GROUPS:
        raise ValueError(f"Unknown pretraining groups: {groups!r}")
    PRETRAINING_GROUPS = groups


def _select_pretraining_rulesets(entries):
    """Keep selected conditions in their existing order without changing values."""
    selected = _PRETRAINING_GROUPS[PRETRAINING_GROUPS]
    return {ruleset: entry for ruleset, entry in entries.items() if ruleset in selected}


def _pretraining_combined_pkls(suffix):
    """Return combined pickles strictly matched to the selected M bound."""
    pattern = f"*_dmpn_{PRETRAINING_ADDON_NAME}_{suffix}.pkl"
    return sorted(PRETRAINING_ANALYSIS_DIR.glob(pattern))


def _pretraining_result_pkls():
    """Return per-seed results matching both the M bound and selected groups."""
    pattern = f"*_dmpn_seed*_{PRETRAINING_ADDON_NAME}_result.pkl"
    selected = _PRETRAINING_GROUPS[PRETRAINING_GROUPS]
    return [path for path in sorted(PRETRAINING_ANALYSIS_DIR.glob(pattern))
            if _pretraining_ruleset_from_result_name(path.name) in selected]


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
    post-training, comparing both motifs and both single-task controls.

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
    ts_pkl = _pretraining_combined_pkls("transfer_speed")
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

    by_ruleset_mats = _select_pretraining_rulesets(by_ruleset_mats)
    if not by_ruleset_mats:
        print("  Skipped: no transfer-speed data for the selected pretraining groups.")
        return
    ys = thresholds * 100
    fig, ax = plt.subplots(figsize=_TRANSFER_SPEED_FIGSIZE)

    for rs, rs_data in by_ruleset_mats.items():
        label, color = _PRETRAINING_RULESET_STYLES.get(
            rs, (rs, "#718096"))
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
    ax.set_yticks(_TRANSFER_SPEED_YTICKS)
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
    the across-seed mean with population SD for both motifs and both single-task
    controls. This is not the exact stage-2 init. A two-sided independent-seed
    permutation test compares the relevant and irrelevant motif means, assuming
    exchangeability under the null; random initializations are not replicates.
    """
    import json
    import re as _re
    from scipy.stats import permutation_test

    groups = {
        ruleset: (label.replace(" motif", "\nmotif"), color)
        for ruleset, (label, color) in _select_pretraining_rulesets(
            _PRETRAINING_RULESET_STYLES).items()
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
    fig, axis = plt.subplots(figsize=_BACKBONE_PROBE_FIGSIZE)
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
        axis.scatter(position + jitter, samples, color=color, s=18,
                     alpha=0.7, edgecolors="k", linewidths=0.4, zorder=3)
        axis.errorbar(position, mean, yerr=std,
                      fmt="D", color="k", markerfacecolor="white",
                      markersize=4, capsize=3, linewidth=1.0, zorder=4)
    axis.set_xticks(np.arange(len(groups)))
    axis.set_xticklabels([label for label, _ in groups.values()])
    axis.set_xlim(-0.5, len(groups) - 0.5)
    axis.set_ylabel("Accuracy (%)", fontsize=8)
    axis.tick_params(labelsize=6.5)
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
            f"feature={PRETRAINING_ADDON_NAME}; groups={PRETRAINING_GROUPS}; "
            f"bound={PRETRAINING_BOUND}; lambda={PRETRAINING_LAMBDA}: "
            f"{stats_label.replace(chr(10), '; ')}\n"
        )
    fig.suptitle("Random-rule backbone probe", fontsize=9)
    fig.tight_layout(pad=0.4)
    _save_fig(fig, OUT_DIR / "backbone_probe.png", extra=f" ({'; '.join(counts)})")


def plot_learning_trajectory(*, xscale="log"):
    """
    Figure: post-training learning trajectory — accuracy vs training iteration,
    comparing the relevant and irrelevant motifs plus the DelayAnti and
    DelayPro single-task controls. Transparent lines are individual seeds and
    thick lines are condition means.

    Reads per-seed result pickles (learning.acc_iter_post / learning.acc_post).
    Seeds are resampled onto a shared iteration grid before averaging, so it is
    robust to slightly different logging cadences across seeds. The linear-axis
    companion uses the same samples and curves, changing only the x-axis scale.
    """
    if xscale not in ("log", "linear"):
        raise ValueError(f"Unsupported learning-trajectory scale: {xscale!r}")
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

    fig, ax = plt.subplots(
        1, 1, figsize=_PRETRAINING_TRAJECTORY_FIGSIZE)

    for rs in sorted(by_ruleset_traj.keys()):
        trajs = by_ruleset_traj[rs]
        label, color = _PRETRAINING_RULESET_STYLES.get(
            rs, (rs, "#718096"))

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
    ax.set_xscale(xscale)
    # ylim tops at 105 for headroom; explicit ticks stop at 100 so no >100% tick.
    ax.set_yticks(_LEARNING_TRAJECTORY_YTICKS)
    ax.set_ylim([0, 105])
    ax.tick_params(axis="both", labelsize=7)
    _legend(ax, fontsize=6, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    filename = "learning_trajectory_linear.png" if xscale == "linear" else "learning_trajectory.png"
    out_path = OUT_DIR / filename
    _save_fig(fig, out_path)


def plot_learning_trajectory_linear():
    """Plot the same post-training learning curves with a linear iteration axis."""
    plot_learning_trajectory(xscale="linear")


# ─── Figure: Rule vectors ────────────────────────────────────────────────────

def plot_rule_vectors():
    """
    Figure: Pairwise cosine similarity between rule-input vectors.

    Shows how the novel task's learned rule vector relates to every available
    pretrained rule vector, including the DelayAnti and DelayPro single-task
    controls.
    """
    _ensure_out_dir()
    if not PRETRAINING_ANALYSIS_DIR.exists():
        print("  Skipped: pretraining_analysis/ not found.")
        return

    # Try combined pkl first
    rv_pkls = _pretraining_combined_pkls("rule_vectors")
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

    by_ruleset = _select_pretraining_rulesets(by_ruleset)
    if not by_ruleset:
        print("  Skipped: no rule vector data found.")
        return

    ruleset_colors = {
        ruleset: color
        for ruleset, (_, color) in _PRETRAINING_RULESET_STYLES.items()
    }
    ruleset_labels = {
        ruleset: label
        for ruleset, (label, _) in _PRETRAINING_RULESET_STYLES.items()
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

    # All bars are evenly spaced; only the motifs layout opens a gap between
    # its two column groups so the divider drawn below reads as a break.
    bar_step = 1.0
    bar_width = 0.8
    group_gap = _RULE_VECTORS_PAIR_GAP if PRETRAINING_GROUPS == "motifs" else 0.0

    # Build the per-ruleset bar list first, then interleave columns across
    # rulesets instead of grouping all of one ruleset's bars together.
    per_rs_bars = {}  # rs -> list of (label, mean, std)
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
            (label, float(np.mean(values)), float(np.std(values)))
            for values, label, _ in bar_specs
        ]

    # Match learning_trajectory: the motifs layout (two pairs and their gap)
    # takes exactly its size; the six-bar layout keeps the same height and
    # the same width per bar step, so its labels stay as legible.
    n_bars_total = sum(len(bars) for bars in per_rs_bars.values())
    figure_width, figure_height = _PRETRAINING_TRAJECTORY_FIGSIZE
    figure_width *= (n_bars_total + group_gap) / _RULE_VECTORS_MOTIFS_EXTENT
    fig, ax = plt.subplots(1, 1, figsize=(figure_width, figure_height))

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
            lbl, mean, std = bars[col]
            color = ruleset_colors.get(rs, "#718096")
            legend_label = None if rs in labeled else ruleset_labels.get(rs, rs)
            labeled.add(rs)

            ax.bar(x, mean, bar_width, yerr=std, capsize=2,
                   color=color, alpha=0.8, edgecolor="k", linewidth=0.5,
                   label=legend_label)

            all_x.append(x)
            all_labels.append(lbl)
            x += bar_step
        x += group_gap  # gap between successive interleaved column groups

    ax.margins(x=_RULE_VECTORS_X_MARGIN)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=0, ha="center",
                       fontsize=_RULE_VECTORS_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=7)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    if PRETRAINING_GROUPS == "motifs" and len(all_x) == 4:
        ax.axvline((all_x[1] + all_x[2]) / 2, **_RULE_VECTORS_DIVIDER_STYLE)
    ax.set_ylabel("Cosine similarity")
    # Upper right is free in both layouts: the bars there are negative.
    _legend(ax, fontsize=6, frameon=True, loc="upper right")
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
    agg_pkls = _pretraining_combined_pkls("aggregate")

    by_ruleset = {}
    if agg_pkls:
        for p in agg_pkls:
            with open(p, "rb") as f:
                data = pickle.load(f)
            rs = data["ruleset"]
            by_ruleset[rs] = data
        return _select_pretraining_rulesets(by_ruleset)

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

    ruleset_order = (
        "fdanti", "fdgo", "fdanti_delaygo", "fdgo_delaygo")
    self_ruleset_order = (
        "fdanti_delaygo", "fdgo_delaygo", "fdanti", "fdgo")

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
    motifs plus the DelayAnti and DelayPro single-task controls.
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
        ruleset: color
        for ruleset, (_, color) in _PRETRAINING_RULESET_STYLES.items()
    }
    ruleset_labels = {
        ruleset: label
        for ruleset, (label, _) in _PRETRAINING_RULESET_STYLES.items()
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
    and effective modulation (right) — overlaying both motifs and both
    single-task controls.
    """
    _plot_aggregate_cve_period("stimulus")


def plot_pretraining_principal_angles():
    """Plot saved rank-filtered principal-angle spectra in degrees.

    Rows are periods, columns are representations. Means use the common
    available spectrum length; indices order angles, not individual PCs.
    Regenerate legacy analysis results after the numerical-rank correction.
    """
    groups = _select_pretraining_rulesets(_PRETRAINING_RULESET_STYLES)
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
    and effective modulation (right) — overlaying both motifs and both
    single-task controls, with the same conventions as
    plot_aggregate_cve_stimulus.
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
    saved data. Also saves the DMC category companion
    (`_plot_dmc_category_colorwheel`).
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
    _plot_dmc_category_colorwheel()


def _plot_dmc_category_colorwheel():
    """
    Illustration: the DMC stimulus categories on the stimulus colorwheel
    (onetask_stimulus_colorwheel_dmc_category.png). The eight DMC locations
    (`_DMC_STIMULUS_LOCATIONS`) keep their direction colors; the guide circle
    is split into the two categories in the DMC figures' category colors
    (`_CUE_INTERP_CLUSTER_COLORS`): the upper half (location < pi) is
    category 1, which a DMCGo trial responds to as "category 1", the lower
    half category 2. The dashed line is the category boundary through 0 and
    pi. Uses no saved data.
    """
    _ensure_out_dir()
    fig, ax = plt.subplots(1, 1, figsize=(2.6, 2.9))
    ax.plot([-1.15, 1.15], [0.0, 0.0], "--", color="0.5", lw=0.8, zorder=0)
    for label, name, (start, stop), y, va in (
            (1, "Category 1", (0.0, np.pi), 1.2, "bottom"),
            (0, "Category 2", (np.pi, 2 * np.pi), -1.2, "top")):
        theta = np.linspace(start, stop, 100)
        ax.plot(np.cos(theta), np.sin(theta), "-", lw=4.0,
                color=_CUE_INTERP_CLUSTER_COLORS[label], solid_capstyle="butt",
                zorder=1)
        ax.text(0.0, y, name, ha="center", va=va, fontsize=8,
                color=_CUE_INTERP_CLUSTER_COLORS[label])
    for k, loc in enumerate(_DMC_STIMULUS_LOCATIONS):
        ax.scatter(np.cos(loc), np.sin(loc), color=stim_color(k, _SIBLING_N_STIM),
                   s=180, edgecolors="k", linewidths=0.5, zorder=3)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    _save_fig(fig, OUT_DIR / "onetask_stimulus_colorwheel_dmc_category.png")


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


def _plot_onetask_snapshot_single(mat, out_name, cbar_out_name):
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


def _plot_onetask_hidden_single(hidden_vec, out_name, cbar_out_name):
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
            out_name="onetask_modulation_snapshot_single.png",
            cbar_out_name="onetask_modulation_snapshot_single_cbar.png",
        )
    else:
        print(f"  Skipped raw M: stimulus {stim} not in snapshot pickle "
              f"(available: {d.get('stims')}).")

    # Effective modulation W⊙M (only if the analysis saved it).
    if stim in d.get("snapshots_eff", {}):
        _plot_onetask_snapshot_single(
            np.asarray(d["snapshots_eff"][stim]["response"], dtype=float),
            out_name="onetask_emodulation_snapshot_single.png",
            cbar_out_name="onetask_emodulation_snapshot_single_cbar.png",
        )
    else:
        print("  Skipped W⊙M: 'snapshots_eff' not in pickle "
              "(re-run one_task_analysis.py to add it).")

    # Hidden state at the same (mid-response) step — its own figure + colorbar.
    if stim in d.get("hidden_snapshots", {}):
        _plot_onetask_hidden_single(
            np.asarray(d["hidden_snapshots"][stim]["response"], dtype=float),
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
    Used by the one-task grad fixed-point figures. `rep_key` selects which saved
    view to plot:
      "fixed_M"      raw modulation matrix M*        (the actual solved state)
      "fixed_WM"     effective modulation W⊙M*       (transform of M*)
      "fixed_hidden" hidden state produced by M*     (transform of M*)
    Each period's points (one per stimulus) are flattened and projected into a
    SHARED delay-period PCA; points are colored by stimulus.

    `basis`: an optional stored 2-component PCA to project into. When None (the
    normal one-task path), load this pickle's saved delay-period basis. A supplied
    basis keeps this low-level renderer reusable without fitting PCA here."""
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
# probe is distinguished by MARKER. In 2D and one-task 3D, fill denotes
# convergence (filled = strict, hollow = over threshold); two-task gradient 3D
# deliberately renders every candidate solid. Stroke markers have no face, so
# they take a plain `color=` and ignore the fill/edge distinction.
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


def _grad_fp_3d_overlay(period, results):
    """Which added probes get drawn in `period`'s 3D panel, per
    _GRAD_FP_3D_OVERLAY_PROBES (a period absent from it shows none). Names missing
    from this pickle are skipped, so an older or reduced battery still renders.
    A period absent from the table intentionally receives no added probe."""
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
    rules can share one figure. The projection/z dictionaries are keyed by probe
    so they cover both the period probes and the explicitly selected 3D
    overlays."""
    results = d["results"]
    periods, _ = _grad_fp_period_panels(results)
    drawn = periods + [n for p in periods
                       for n in _grad_fp_3d_overlay(p, results)]

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

    return periods, proj_by_period, z_by_period, traj_by_period, n_stim


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
                         draw_periods=None, transpose=False,
                         pc_label=None, solid_candidates=False):
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

    Unlike the 2D figures, which draw the whole battery marker-coded, the 3D
    panels use `_grad_fp_3d_overlay` to select a small, explicit probe subset,
    drawn in gray with the same circle marker and size as everything else. The
    dashed candidate-ring line and trajectory connector stay tied to the period's
    own probe, which is the only one the trial actually traversed."""
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
        # The two-task paper figures request solid candidates; their convergence
        # quality is reported by the dedicated residual/classification figures
        # instead of by hollow markers here. Other callers retain filled=strict
        # and hollow=over-threshold. In either case `good` remains authoritative
        # for the dashed ring, which connects strict fixed points only.
        _fp_style = dict(marker="o", s=14, fill=True, z=3)
        # Same 0.85 as everywhere else. It was briefly dropped to 0.55 to keep the
        # exemplar trajectory the subject of these panels, but the trajectory now
        # carries its own weight (black, bold, white-cased), so the fixed points do
        # not need to be faded to make room for it. Lower this to demote them again.
        _FP_ALPHA = 0.85
        # This period's own probe.
        shown_good = (np.ones(xy.shape[0], dtype=bool)
                      if solid_candidates else good)
        _scatter_grad_fp(ax, xy, _fp_style,
                         _grad_fp_3d_colors(v, e, stim, n_stim), shown_good,
                         z_vals=np.asarray(z, dtype=float), alpha=_FP_ALPHA)
        # Plus this period's added fixed points, in gray (Context: the ring-alike
        # memory-seeded ring; Response: the naive-seeded solve — see
        # _GRAD_FP_3D_OVERLAY_KIND).
        for name in _grad_fp_3d_overlay(v, results):
            if name not in proj_by_period:      # not projected for this figure
                continue
            pe = results[name]
            pxy = proj_by_period[name]
            probe_good = (_fixed_point_mask(pe, pxy.shape[0])
                          if not solid_candidates
                          else np.ones(pxy.shape[0], dtype=bool))
            _scatter_grad_fp(ax, pxy, _fp_style,
                             _grad_fp_3d_colors(v, pe, np.asarray(pe["stim"]),
                                                n_stim),
                             probe_good,
                             z_vals=np.asarray(z_by_period[name], dtype=float),
                             alpha=_FP_ALPHA)
        # Connect strict fixed points into a stimulus-ordered candidate ring with
        # a thin dashed black line. This visualizes geometry only; it does not by
        # itself establish a continuous attractor. One representative point per
        # stimulus is ordered by stimulus index and closed into a loop.
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
        # `pc_label` names which period supplied the shared x-y PCA basis. The
        # same basis is used by every panel, so the axis label records that
        # provenance explicitly (for example, "Stimulus PC1").
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

    `basis`: an optional stored 2-component PCA to project into. When None (the
    normal one-task path), load this pickle's saved delay-period basis. A supplied
    basis is used by the vanilla-RNN control so it can share the MPN coordinates."""
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

    periods, proj, zc, traj, n_stim = _grad_fp_3d_project(d, rep_key, basis)
    # Rotate the hidden-state content 90° (see _rotate_hidden_3d_content); a no-op
    # for the modulation / eff-modulation representations.
    proj, traj = _rotate_hidden_3d_content(rep_key, proj, traj)

    # Limits span exactly what the panels draw: the period probes plus the few
    # overlaid ones the 3D figures keep (the naive probes are not drawn here, so
    # they must not stretch the axes either).
    drawn_3d = [v for p in periods
                for v in [p] + _grad_fp_3d_overlay(p, results)
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
                         pc_label=pc_label)
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
      left  — fixed-point candidates in a 2-PC PCA of the solved M*, colored by
              stimulus angle. Strict candidates are filled, approximate ones are
              hollow, and failed solves use x markers. Connecting segments are
              solid only when both endpoints are strict.
      right — leak-normalized relative residual versus angle (log y), with the
              strict and approximate thresholds. Uniformly below the strict line
              means every angle is stationary at the requested precision.
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
    # Prefer the leak-normalized residual, whose scale is comparable across
    # modulation time constants. Reconstruct it from a legacy raw residual when
    # the pickle records its leak; only truly old files remain on the raw scale.
    rel_step = e.get("rel_step_undamped")
    normalized = rel_step is not None
    if rel_step is None and e.get("rel_step") is not None:
        rel_step = np.asarray(e["rel_step"], dtype=float)
        if e.get("leak") is not None:
            rel_step = rel_step / max(float(e["leak"]), 1e-12)
            normalized = True
    if rel_step is None:
        step_norm = np.sqrt(2.0 * np.asarray(e["final_speeds"], dtype=float))
        m_norm = np.maximum(np.linalg.norm(fixed.reshape(n, -1), axis=1), 1e-12)
        rel_step = step_norm / m_norm
    rel_step = np.asarray(rel_step, dtype=float)
    if normalized:
        rel_tol = float(e.get(
            "rel_tol_undamped", d.get("rel_tol_undamped", 1e-2)))
        approx_tol = float(e.get(
            "approx_rel_tol_undamped",
            d.get("approx_rel_tol_undamped", 5e-2)))
    else:
        rel_tol = float(e.get("rel_tol", d.get("rel_tol", 0.05)))
        approx_tol = float(e.get("approx_rel_tol", 5.0 * rel_tol))
    if approx_tol < rel_tol:
        raise ValueError("approximate residual threshold must be >= strict threshold")
    finite = np.isfinite(rel_step)
    strict = finite & (rel_step <= rel_tol)
    approximate = finite & (rel_step > rel_tol) & (rel_step <= approx_tol)
    failed = ~(strict | approximate)

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

    # Left: candidates in PCA, connected in angle order. A segment is solid only
    # when both endpoints pass the strict residual criterion; approximate and
    # failed portions cannot silently complete an apparently strict ring.
    ax = axs[0]
    for i in range(n):
        j = (i + 1) % n
        linestyle = "-" if strict[i] and strict[j] else "--"
        ax.plot(proj[[i, j], 0], proj[[i, j], 1], linestyle,
                color="0.7", linewidth=0.8, alpha=0.6, zorder=1)
    for i in range(n):
        if strict[i]:
            ax.scatter(proj[i, 0], proj[i, 1], color=cols[i], s=30,
                       edgecolor="none", zorder=3)
        elif approximate[i]:
            ax.scatter(proj[i, 0], proj[i, 1], facecolor="none", edgecolor=cols[i],
                       s=30, linewidth=1.1, zorder=3)
        else:
            ax.scatter(proj[i, 0], proj[i, 1], color=cols[i], marker="x",
                       s=32, linewidth=1.1, zorder=3)
    ax.set_xlabel("Candidate PC1", fontsize=10)
    ax.set_ylabel("Candidate PC2", fontsize=10)
    ax.set_title("Fixed-point candidates\n"
                 f"{strict.sum()} strict / {approximate.sum()} approximate / "
                 f"{failed.sum()} failed", fontsize=10)
    quality_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="None", color="0.3",
                         markerfacecolor="0.3", markeredgecolor="none",
                         label="Strict"),
        mpl.lines.Line2D([], [], marker="o", linestyle="None", color="0.3",
                         markerfacecolor="none", markeredgecolor="0.3",
                         label="Approximate"),
        mpl.lines.Line2D([], [], marker="x", linestyle="None", color="0.3",
                         label="Failed"),
    ]
    _legend(ax, handles=quality_handles, fontsize=7, loc="best", frameon=False)
    ax.set_aspect("equal")   # so a ring reads as circular, not stretched
    ax.spines[["top", "right"]].set_visible(False)

    # Right: normalized residual vs angle (continuity diagnostic). Mark the
    # trained angles and both quality thresholds.
    ax = axs[1]
    deg = np.degrees(angles)
    ax.plot(deg, rel_step, "-", color="0.65", linewidth=0.8, zorder=1)
    quality_specs = (
        (strict, "#009E73", "o", f"Strict ({strict.sum()})"),
        (approximate, "#E69F00", "o",
         f"Approximate ({approximate.sum()})"),
        (failed, "#D55E00", "x", f"Failed ({failed.sum()})"),
    )
    for mask, color, marker, label in quality_specs:
        visible = mask & finite
        if np.any(visible):
            scatter_kw = {"edgecolor": "none"} if marker == "o" else {}
            ax.scatter(deg[visible], rel_step[visible], s=14, color=color,
                       marker=marker, linewidth=0.8, label=label, zorder=3,
                       **scatter_kw)
    # Dashed lines at the trained ring directions — only when few enough to be
    # legible (a dense/`morestimulus` run has too many to mark).
    if n_trained <= 16:
        for k in range(n_trained):
            ax.axvline(360.0 * k / n_trained, color="0.8", lw=0.6,
                       linestyle="--", zorder=0)
    ax.axhline(rel_tol, color="#374151", lw=1.0, linestyle="-",
               label=f"Strict threshold = {rel_tol:g}", zorder=2)
    ax.axhline(approx_tol, color="#6B7280", lw=1.0, linestyle="--",
               label=f"Approx. threshold = {approx_tol:g}", zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Stimulus angle (deg)", fontsize=10)
    ylabel = (r"Normalized relative step"
              if normalized else r"Relative step  $\|F(M^*)-M^*\|/\|M^*\|$")
    ax.set_ylabel(ylabel, fontsize=9)
    _dash_note = (f"\n(dashed = {n_trained} trained dirs)"
                  if n_trained <= 16 else "")
    metric_title = "Normalized residual" if normalized else "Relative step"
    ax.set_title(f"{metric_title} vs angle{_dash_note}", fontsize=9)
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
    already-loaded one-task pickle dict `d`. Reads the Jacobian eigenvalues saved
    in the grad-fp pickle (written by the stability pass in
    core/grad_fixed_points.py). Candidates that fail the strict fixed-point
    residual criterion are excluded from both rows. Two rows:

      top  — leading Jacobian eigenvalues of F(M*) in the COMPLEX PLANE, one panel
             per period, colored by stimulus angle, with the unit circle. As a
             discrete map: points inside the circle are contracting, outside are
             expanding; an eigenvalue sitting AT (1, 0) is a marginal/neutral
             direction and a necessary, but not sufficient, ring-attractor sign.
      bottom — spectral radius ρ = max|λ| vs stimulus angle, per period, with the
             ρ = 1 stability line. For a converged fixed point, ρ < 1 means
             locally attracting.

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
    marg_tol = _fixed_point_marginal_tol(results[periods[0]])
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
        is_fixed = _fixed_point_mask(e, rad.size)

        # ── Top row: eigenvalues in the complex plane ────────────────────────
        ax = axs[0][j]
        ax.plot(np.cos(theta), np.sin(theta), "-", color="0.7", lw=0.8, zorder=1)
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.axvline(0, color="0.85", lw=0.5, zorder=0)
        for i in np.flatnonzero(is_fixed):
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
        ax.plot(x[is_fixed], rad[is_fixed], "-o", color=c_vals[0],
                markersize=3, zorder=3)
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


# Stable / marginal / unstable colors use a colorblind-safe Okabe-Ito subset;
# gray remains the natural visual middle for the near-neutral marginal class.
_FP_CLASS_COLORS = {
    "stable": "#009E73",
    "marginal": "#9CA3AF",
    "unstable": "#D55E00",
    "approximate": "#56B4E9",
    "unconverged": "#E69F00",
}


def _fixed_point_classification_counts(entry):
    """Return counts with the approximate-residual band represented explicitly.

    New classification pickles already contain this category. For older ones,
    reconstruct it from the saved normalized residual without changing the file
    on disk. The strict and approximate defaults mirror core.grad_fixed_points.
    """
    counts = dict(entry.get("counts", {}))
    if "approximate" in counts:
        return counts
    residual = entry.get("rel_step_undamped")
    if residual is None:
        counts["approximate"] = 0
        return counts
    residual = np.asarray(residual, dtype=float)
    strict_tol = float(entry.get("rel_tol_undamped", 1e-2))
    approx_tol = float(entry.get("approx_rel_tol_undamped", 5e-2))
    finite = np.isfinite(residual)
    counts["approximate"] = int(np.sum(
        finite & (residual > strict_tol) & (residual <= approx_tol)))
    counts["unconverged"] = int(np.sum(
        ~finite | (residual > approx_tol)))
    return counts

_STABILITY_PHASE_ORDER = {
    "longfixation": 0,
    "longstimulus": 1,
    "longdelay": 2,
    "longresponse": 3,
}
_STABILITY_PHASE_LABELS = {
    "longfixation": "Context",
    "longstimulus": "Stimulus",
    "longdelay": "Delay",
    "longresponse": "Response",
}
_STABILITY_PROBE_SUFFIXES = (
    ("_memseed", "Memory seed", 1),
    ("_trajseed", "Trajectory seed", 2),
    ("_naiveseed", "Naive seed", 3),
)


def _stability_condition_parts(period, entry=None):
    """Return sortable group/phase metadata and a compact display label."""
    base, group, group_order = period, "Task epoch", 0
    for suffix, candidate_group, candidate_order in _STABILITY_PROBE_SUFFIXES:
        if period.endswith(suffix):
            base = period[:-len(suffix)]
            group, group_order = candidate_group, candidate_order
            break
    raw_title = ((entry or {}).get("period_title", base))
    phase = _STABILITY_PHASE_LABELS.get(base, _period_display(raw_title))
    # Saved probe titles may already include the seed description. Derive the
    # phase from the canonical key so the grouped y labels stay short.
    if group_order:
        phase = _STABILITY_PHASE_LABELS.get(base, _period_display(base))
    return group_order, _STABILITY_PHASE_ORDER.get(base, 99), group, phase


def _ordered_stability_periods(per_period):
    return sorted(
        per_period,
        key=lambda period: (*_stability_condition_parts(
            period, per_period[period])[:2], period),
    )


def _stability_condition_label(period, entry=None):
    _, _, group, phase = _stability_condition_parts(period, entry)
    return f"{group} · {phase}"


def _fixed_point_class(radius, tol):
    """Classify spectral radii with the same thresholds as the analysis."""
    radius = np.asarray(radius, dtype=float)
    classes = np.full(radius.shape, "marginal", dtype=object)
    classes[radius < 1.0 - tol] = "stable"
    classes[radius > 1.0 + tol] = "unstable"
    return classes


def _render_twotask_fixed_point_stability(d, out_path, rule,
                                           n_trained=TWOTASK_N_STIM):
    """Compact two-task stability diagnostic.

    The four task epochs retain eigenvalue-plane panels. All task epochs and
    initialization probes are then aligned as rows of one spectral-radius
    heatmap, replacing the former 2-by-N strip whose eleven narrow columns were
    difficult to compare.
    """
    _ensure_out_dir()
    results = d.get("results", {})
    periods = _ordered_stability_periods(results)
    if not periods or any(results[v].get("eigenvalues") is None for v in periods):
        print("  Skipped: 'eigenvalues' not in pickle "
              "(re-run the analysis to add the stability pass).")
        return

    main_periods = [v for v in periods
                    if _stability_condition_parts(v, results[v])[0] == 0]
    if not main_periods:
        print("  Skipped: no task-epoch stability entries in pickle.")
        return

    tol = _fixed_point_marginal_tol(results[periods[0]])
    max_stim = max(
        int(np.max(np.asarray(results[v]["stim"], dtype=int)))
        for v in periods
    )
    n_stim = max_stim + 1
    radius_grid = np.full((len(periods), n_stim), np.nan, dtype=float)
    for row, period in enumerate(periods):
        entry = results[period]
        stim = np.asarray(entry["stim"], dtype=int)
        radius = np.asarray(entry["spectral_radius"], dtype=float)
        is_fixed = _fixed_point_mask(entry, radius.size)
        for stimulus in np.unique(stim):
            selected = (stim == stimulus) & is_fixed & np.isfinite(radius)
            if np.any(selected):
                # If a future run stores several magnitudes for one direction,
                # show the least-stable fixed point rather than hiding it in an
                # average.
                radius_grid[row, stimulus] = np.max(radius[selected])

    all_eigenvalues = [
        np.asarray(results[v]["eigenvalues"])
        for v in main_periods
    ]
    finite_eigenvalues = np.concatenate([
        eig[np.isfinite(eig)] for eig in all_eigenvalues
    ])
    eig_extent = max(1.08, float(np.max(np.abs(finite_eigenvalues))) * 1.05)

    fig = plt.figure(figsize=(7.4, 6.2))
    grid = fig.add_gridspec(
        2, len(main_periods), height_ratios=(1.0, 1.45),
        hspace=0.42, wspace=0.30)
    theta = np.linspace(0.0, 2.0 * np.pi, 256)

    for col, period in enumerate(main_periods):
        entry = results[period]
        eig = np.asarray(entry["eigenvalues"])
        radius = np.asarray(entry["spectral_radius"], dtype=float)
        is_fixed = _fixed_point_mask(entry, radius.size)
        classes = _fixed_point_class(radius, tol)
        ax = fig.add_subplot(grid[0, col])

        # Remaining modes recede; the leading-|lambda| mode carries the actual
        # fixed-point verdict and receives the semantic class color.
        if eig.shape[1] > 1:
            other = eig[is_fixed, 1:].reshape(-1)
            ax.scatter(other.real, other.imag, s=5, color="#CBD5E1",
                       alpha=0.28, edgecolor="none", rasterized=True, zorder=1)
        leading = eig[:, 0]
        for class_name in ("stable", "marginal", "unstable"):
            selected = is_fixed & (classes == class_name)
            if np.any(selected):
                ax.scatter(leading[selected].real, leading[selected].imag,
                           s=14, color=_FP_CLASS_COLORS[class_name], alpha=0.85,
                           edgecolor="white", linewidth=0.25, zorder=3)

        ax.plot(np.cos(theta), np.sin(theta), color="#374151", lw=0.9, zorder=2)
        ax.plot((1.0 + tol) * np.cos(theta), (1.0 + tol) * np.sin(theta),
                color="#D55E00", lw=0.65, ls="--", alpha=0.65, zorder=2)
        ax.axhline(0, color="0.88", lw=0.5, zorder=0)
        ax.axvline(0, color="0.88", lw=0.5, zorder=0)
        ax.set_xlim(-eig_extent, eig_extent)
        ax.set_ylim(-eig_extent, eig_extent)
        ax.set_aspect("equal")
        _, _, _, phase = _stability_condition_parts(period, entry)
        ax.set_title(phase, fontsize=9.5)
        ax.set_xlabel(r"Re($\lambda$)", fontsize=8)
        if col == 0:
            ax.set_ylabel(r"Im($\lambda$)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6.5, length=2)
        ax.spines[["top", "right"]].set_visible(False)

    heat_ax = fig.add_subplot(grid[1, :])
    finite_radius = radius_grid[np.isfinite(radius_grid)]
    vmin = min(1.0 - tol, float(np.min(finite_radius)))
    vmax = max(1.0 + tol, float(np.max(finite_radius)))
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#F3F4F6")
    image = heat_ax.imshow(
        radius_grid, aspect="auto", interpolation="nearest", cmap=cmap,
        norm=norm, extent=(0.0, 360.0, len(periods), 0.0))

    heat_ax.set_yticks(np.arange(len(periods)) + 0.5)
    heat_ax.set_yticklabels([
        _stability_condition_label(v, results[v]) for v in periods
    ], fontsize=7.5)
    heat_ax.set_xticks([0, 90, 180, 270, 360])
    heat_ax.set_xlabel("Stimulus angle (deg)", fontsize=9)
    heat_ax.set_ylabel("Condition", fontsize=9)
    heat_ax.tick_params(axis="x", labelsize=7)
    if n_trained <= 16:
        for direction in range(n_trained):
            heat_ax.axvline(360.0 * direction / n_trained, color="white",
                            lw=0.45, ls=":", alpha=0.75)

    previous_group = _stability_condition_parts(periods[0], results[periods[0]])[0]
    for row, period in enumerate(periods[1:], start=1):
        group = _stability_condition_parts(period, results[period])[0]
        if group != previous_group:
            heat_ax.axhline(row, color="white", lw=2.0)
        previous_group = group
    heat_ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(image, ax=heat_ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"Spectral radius $\rho$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    for threshold in (1.0 - tol, 1.0, 1.0 + tol):
        if vmin <= threshold <= vmax:
            cbar.ax.axhline(threshold, color="black", lw=0.55,
                            ls="-" if threshold == 1.0 else "--")

    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=5,
                         color=_FP_CLASS_COLORS[name], label=name.capitalize())
        for name in ("stable", "marginal", "unstable")
    ]
    handles.append(mpl.lines.Line2D([], [], marker="o", linestyle="",
                                    markersize=4, color="#CBD5E1",
                                    label="Other eigenmodes"))
    if SHOW_LEGEND:
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                   fontsize=7.5, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{_TASK_DISPLAY.get(rule, rule)} fixed-point stability",
                 fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.18, right=0.94, top=0.88, bottom=0.09)
    _save_fig(fig, out_path)


def _draw_classification_bars(ax, per_period, class_names, add_legend=True):
    """Draw a per-period stacked bar of stable / marginal / unstable fixed-point
    counts onto `ax` (segments colored by _FP_CLASS_COLORS). `per_period` maps a
    period key -> {"period_title", "counts": {class: n}}. Used by the one-task
    classification figure."""
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


# Reference rule whose stimulus-period PCA defines the SHARED x-y basis for both
# the 2D and 3D two-task gradient fixed-point figures. Every rule is projected
# into this one reference basis so their geometry is directly comparable rather
# than each rule using its own, incomparable PCA coordinates.
_TWOTASK_FP_BASIS_RULE = "delayanti"

# Rule order for the combined two-task fixed-point figures. It runs top-to-bottom
# in 2D and left-to-right in 3D.
_TWOTASK_FP_RULE_ORDER = ["delaygo", "delayanti"]


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
    """Load each rule's grad-fp pickle once in `_TWOTASK_FP_RULE_ORDER` (rules
    absent from the list are appended in discovery order). Returns an ordered
    list of (rule, loaded-pickle-dict), skipping missing or unreadable files.
    The 2D renderer maps this order to rows; the 3D renderer maps it to columns."""
    rule_order = _TWOTASK_FP_RULE_ORDER
    by_rule = dict(paths)
    ordered_rules = ([r for r in rule_order if r in by_rule]
                     + [r for r in by_rule if r not in rule_order])
    rule_data = []
    for rule in ordered_rules:
        d = _load_pkl_or_skip(by_rule[rule], "Run two_task_analysis.py first.")
        if d is not None:
            rule_data.append((rule, d))
    return rule_data


def _render_two_task_grad_fp_2d_combined(rule_data, rep_key, out_path, basis,
                                         pc_label="Stimulus"):
    """Draw ALL two-task rules as stacked rows of a SINGLE 2D figure for one
    representation. `rule_data` is an ordered list of (rule, loaded-pickle-dict)
    (row order = top→bottom); every rule is projected into the shared `basis` and
    the rows share one symmetric x-y limit so panels are directly comparable
    across rows. Period titles print only on the top row; each row is labeled on
    the left (y-axis) by its task rule. `pc_label` names the stimulus-period
    shared PCA basis on the figure's x/y labels."""
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


# The two-task grad fixed-point figures use one shared x-y PCA basis: the
# reference rule's stimulus-period activity. The delay-PCA duplicate was removed
# because it showed the same fixed points in a second projection without adding
# a distinct analysis. Tuple fields: (period key, filename infix, axis label).
_TWOTASK_FP_BASIS = ("longstimulus", "stimpc", "Stimulus")

# Periods the two-task 3D figure does NOT give a panel to, matched as substrings of
# the period key. Both are still kept in `periods`, so each remains the anchor for
# the next drawn panel's incoming trajectory (fixation → Stimulus, delay →
# Response) — they are dropped as panels, not as data.
#   Context — its fixed points carry no stimulus structure at all.
#   Delay    — its geometry is already shown in full by the 2D figure and its
#              ideal-output z coordinate is zero. The 3D figure keeps the two
#              periods where the added z axis is most informative: where the
#              stimulus structure forms and where the response leaves z=0.
# Shared with the task-INTERPOLATION 3D figure (_render_interp_alpha_fp_3d), so both
# two-task 3D figures show the same two epochs and read as a pair; there the skipped
# periods are dropped outright, nothing being anchored across periods. The 2D and
# one-task 3D figures still draw every period.
_TWOTASK_FP_3D_SKIP_PANELS = ("fixation", "delay")


def _plot_two_task_grad_fp_combined(stem_prefix, log_label, render_fn,
                                    with_pc_label):
    """Shared driver for the combined two-task grad fixed-point figures (2D and
    3D). Loads each rule's pickle once, then for every representation (raw M*,
    W⊙M*, hidden) loads the shared delayanti stimulus-period basis and calls
    `render_fn` to draw all rules into ONE figure. The 2D renderer stacks rules
    as rows; the 3D renderer places rules in columns and periods in rows. Output:
      {stem_prefix}_{seed}_stimpc_{suffix}.png

    stem_prefix   : filename stem before the seed tag ("twotask_grad_fixed_points"
                    for 2D, that + "_3d" for 3D).
    log_label     : short tag for the shared-basis log line ("twotask-2d"/"-3d").
    render_fn     : the combined renderer (_render_two_task_grad_fp_2d_combined or
                    _..._3d_combined); called as
                    render_fn(rule_data, rep_key, out_path, basis[, pc_label=...]).
    with_pc_label : pass "Stimulus" as pc_label= so axes explicitly identify the
                    shared PCA basis. Both renderers accept it."""
    paths = _twotask_grad_fp_paths()
    if not paths:
        print("  Skipped: no fixed_points_grad_*_{rule}.pkl in "
              f"{TWOTASKS_DIR / TWOTASK_ANAME}. Run two_task_analysis.py first.")
        return
    tag = _twotask_seed_tag()
    rule_data = _load_two_task_grad_fp_rules(paths)
    if not rule_data:
        return

    period, infix, pc_label = _TWOTASK_FP_BASIS
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
    SINGLE figure per representation (top → bottom = _TWOTASK_FP_RULE_ORDER, i.e.
    delaygo over delayanti), colored by stimulus; period titles on the top row,
    task-rule labels down the left.

    STIMULUS PCA BASIS: the `_TWOTASK_FP_BASIS_RULE` (delayanti) stimulus period
    defines the shared x-y PCA basis. Outputs are:
      twotask_grad_fixed_points_{seed}_stimpc_modulation.png
      twotask_grad_fixed_points_{seed}_stimpc_emodulation.png
      twotask_grad_fixed_points_{seed}_stimpc_hidden.png

    SHARED BASIS: all rows are projected into one delayanti stimulus-period x-y
    plane (fit once per representation) and share one symmetric x-y limit, so the
    delaygo and delayanti geometry is directly comparable point-for-point. If the
    reference-rule pickle or saved basis is missing, that representation is
    skipped; no per-rule fallback is used.
    Reads twotasks/{aname}/fixed_points_grad_{aname}_{rule}.pkl.
    """
    _plot_two_task_grad_fp_combined(
        "twotask_grad_fixed_points", "twotask-2d",
        _render_two_task_grad_fp_2d_combined, with_pc_label=True)


def _render_two_task_grad_fp_3d_combined(rule_data, rep_key, out_path, basis,
                                         pc_label=None):
    """Draw ALL two-task rules as columns of a SINGLE 3D figure for one
    representation. `rule_data` is an ordered list of (rule, loaded-pickle-dict)
    (column order follows `_TWOTASK_FP_RULE_ORDER`); every rule is projected into
    the shared `basis`. Columns share one x-y limit and one z-limit, and trial
    periods run down the rows."""
    # Project every rule first, so shared axis limits can span all rows.
    per_rule = []   # (rule, results, periods, draw_periods, proj, zc, traj, n_stim)
    for rule, d in rule_data:
        results = d["results"]
        periods = list(results.keys())
        if not periods or any(results[v].get(rep_key) is None for v in periods):
            print(f"  Skipped '{rep_key}' for rule '{rule}': not in pickle.")
            continue
        periods, proj, zc, traj, n_stim = _grad_fp_3d_project(d, rep_key, basis)
        # Draw only the Stimulus and Response panels; the fixation and delay panels
        # are skipped (_TWOTASK_FP_3D_SKIP_PANELS) but KEPT in `periods`, so each
        # still anchors the next drawn panel's incoming trajectory.
        # `draw_periods` = the panels actually drawn.
        draw_periods = [v for v in periods
                        if not any(skip in v.lower()
                                   for skip in _TWOTASK_FP_3D_SKIP_PANELS)]
        # NB: unlike the one-task figure, the two-task hidden panels are NOT
        # rotated — they use the same viewing angle as modulation / e_modulation.
        per_rule.append((rule, results, periods, draw_periods, proj, zc, traj,
                         n_stim))
    if not per_rule:
        print(f"  Skipped '{rep_key}': no rule had it.")
        return
    if not any(draw for (_, _, _, draw, _, _, _, _) in per_rule):
        print(f"  Skipped '{rep_key}': no period survived "
              f"_TWOTASK_FP_3D_SKIP_PANELS as a panel.")
        return

    # Shared symmetric x-y and z limits across ALL rows (so rows are comparable);
    # computed over the DRAWN periods only (the fixation and delay panels are not
    # drawn). Spans the drawn panels' own probes AND the probes actually overlaid
    # into them (the 3D figures keep only their explicitly selected subset), so an
    # overlaid point is never silently outside the axes and an undrawn one never
    # stretches them.
    def _panel_probes(draw_periods, results, proj):
        return [v for p in draw_periods
                for v in [p] + _grad_fp_3d_overlay(p, results)
                if v in proj]

    # ... plus the single anchor point each panel inherits from the period BEFORE
    # it, which the panel draws as its trajectory's start. That period may itself be
    # undrawn (fixation, delay), and its anchor can sit outside the drawn probes'
    # own extent, so it has to be in the limits even though the rest of its ring is
    # deliberately not.
    def _anchor_xyz(rows):
        pts = [_grad_fp_3d_anchor_xyz(res, periods, draw_periods, proj, zc)
               for (_, res, periods, draw_periods, proj, zc, _, _) in rows]
        pts = [p for p in pts if p[0].size]
        if not pts:
            return np.zeros((0, 2)), np.zeros(0)
        return (np.vstack([xy for xy, _ in pts]),
                np.concatenate([z for _, z in pts]))

    anchor_xy, anchor_z = _anchor_xyz(per_rule)
    lim = max(np.abs(np.vstack(
        [proj[v] for (_, res, _, draw_periods, proj, _, _, _) in per_rule
         for v in _panel_probes(draw_periods, res, proj)]
        + [anchor_xy])).max() * 1.08, 1e-9)
    zmax = max(np.abs(np.concatenate(
        [zc[v].ravel() for (_, res, _, draw_periods, _, zc, _, _) in per_rule
         for v in _panel_probes(draw_periods, res, zc)]
        + [anchor_z])).max() * 1.1, 1e-6)
    # TASK RULE runs along x (one column per rule, named by the column title) and
    # TRIAL PERIOD down y (one row per period, named by the row's side label) — the
    # transpose of the original layout, so the two rules sit side by side for
    # comparison and the trial unfolds downward.
    n_col = len(per_rule)
    n_rows = max(len(draw_periods)
                 for (_, _, _, draw_periods, _, _, _, _) in per_rule)

    fig = plt.figure(figsize=(1.8 * n_col, 1.8 * n_rows))
    for col_idx, (rule, results, periods, draw_periods, proj, zc, traj,
                  n_stim) in enumerate(per_rule):
        _draw_grad_fp_3d_row(
            fig, results, periods, proj, zc, traj, n_stim, lim, zmax,
            n_rows=n_rows, row_idx=col_idx, n_col=n_col,
            row_label=_TASK_DISPLAY.get(rule, rule),
            draw_periods=draw_periods, transpose=True,
            pc_label=pc_label, solid_candidates=True)
    # hspace sets the vertical gap between the period rows (3D axes carry large
    # internal margins, so this is negative but less so than the tightest pack).
    # Per-panel z-labels now (no shared right-margin label), so use full width.
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top=0.94,
                        wspace=0.12, hspace=-0.05)
    _save_fig(fig, out_path)


def plot_two_task_grad_fixed_points_3d():
    """
    3D two-task gradient fixed-point figures — BOTH rules placed as columns of a
    SINGLE figure per representation (left → right follows
    _TWOTASK_FP_RULE_ORDER). Trial periods run down the rows; x-y is the shared PCA
    basis and z is the ideal cos-output target.

    STIMULUS PCA BASIS: as in the 2D figures, the `_TWOTASK_FP_BASIS_RULE`
    (delayanti) stimulus period defines the shared x-y PCA basis. Outputs are:
      twotask_grad_fixed_points_3d_{seed}_stimpc_modulation.png
      twotask_grad_fixed_points_3d_{seed}_stimpc_emodulation.png
      twotask_grad_fixed_points_3d_{seed}_stimpc_hidden.png

    SHARED BASIS: all rows are projected into one delayanti stimulus-period x-y
    plane (fit once per representation) and share one x-y/z limit, so the delaygo
    and delayanti geometry is directly comparable point-for-point. If the
    reference-rule pickle or saved basis is missing, that representation is
    skipped; no per-rule fallback is used.
    Reads twotasks/{aname}/fixed_points_grad_{aname}_{rule}.pkl.
    """
    _plot_two_task_grad_fp_combined(
        "twotask_grad_fixed_points_3d", "twotask-3d",
        _render_two_task_grad_fp_3d_combined, with_pc_label=True)


# ═════════════════════════════════════════════════════════════════════════════
# Multi-task sibling-family fixed-point geometry
# ═════════════════════════════════════════════════════════════════════════════
# This is the sole retained two_in_multiple analysis. It reads both gradient
# fixed-point and long-delay-endpoint artifacts in each family's joint six-PC
# delay-trajectory basis; paper_plot never refits PCA on fixed points.
_MULTITASK_SIBLING_FAMILIES = {
    "delaydm1": ("delaydm1", "delaydm2"),
    "dmcgo": ("dmcgo", "dmcnogo"),
}
_MULTITASK_SIBLING_OUTPUT_NAMES = {
    "delaydm1": "delaydm",
    "dmcgo": "dmcgo",
}
_MULTITASK_SIBLING_PC_PLANES = {
    family: {
        "hidden": (1, 2),
        "e_modulation": (1, 2),
    }
    for family in _MULTITASK_SIBLING_FAMILIES
}
_MULTITASK_SIBLING_ENDPOINT_DEFAULT_PC_PLANE = (1, 2)
_MULTITASK_RULE_MARKERS = ("s", "^")
_MULTITASK_ENDPOINT_LIMIT_PADDING = 0.13
_MULTITASK_DELAYDM_EMODULATION_LIMIT_PADDING = 0.18


def _pinned_sibling_aname(family):
    """The run DELAYDM_ANAME / DMCGO_ANAME pins for `family`, or None."""
    return {"delaydm1": DELAYDM_ANAME, "dmcgo": DMCGO_ANAME}[family]


def _multitask_sibling_aname(family, plot_name, figure="endpoint"):
    """Return the training run behind one sibling-family figure: `figure`
    ("endpoint" or "sweep", `_SIBLING_FIGURES`) of representation `plot_name`
    (`_SIBLING_REPRESENTATIONS`). The family's pinned run, or with None the
    run selected for that figure (resolved once per family)."""
    pinned = _pinned_sibling_aname(family)
    if pinned is not None:
        return pinned
    if family not in _SIBLING_SELECTED_RUNS:
        _SIBLING_SELECTED_RUNS[family] = _select_sibling_runs(family)
    return _SIBLING_SELECTED_RUNS[family][f"{figure} {plot_name}"]


# ─── Automatic choice of the sibling example runs ────────────────────────────
# The sibling figures are illustrations, so with DELAYDM_ANAME / DMCGO_ANAME =
# None each figure shows the cached SIBLING_L2 run whose drawn fixed points
# separate best in their best PC pair, chosen on its own. The endpoint figure
# of each representation is scored on its endpoints
# (`_best_sibling_endpoint_pc_pair`: stimulus directions for DelayDM, response
# categories for DMC, both tasks pooled) and the rule-cue sweep figure on the
# sweep's own fixed points (`_sibling_sweep_pc_pair`). A DelayDM sweep figure
# weighs that score by how smoothly its drawn trials follow the cue and ranks
# last the runs whose drawn directions collapse onto one line
# (`_cue_interp_drawn_legibility`, `_sibling_sweep_ranked_score`): the score
# is a ratio to the spread within a direction, so a run whose trials all land
# on one attractor can score well while drawing nothing apart. Figures of one
# family can therefore come from different networks. The endpoint figure's run also
# serves its `_cue_interp` overlay (the overlaid sweep shares that run's basis)
# and the gradient-method geometry figures; the sweep figure's run serves its
# 2d and 3d views. The whole ranking is printed and saved beside the figures.
_SIBLING_SELECTED_RUNS = {}
_SIBLING_REPRESENTATIONS = ("hidden", "e_modulation")
_SIBLING_FIGURES = ("endpoint", "sweep")


def _dmc_sweep_metric_name():
    """Name of the DMC sweep score under SIBLING_DMC_KEEP_AT_EVERY_ALPHA."""
    return (DMC_SWEEP_KEEP_EVERY_ALPHA_METRIC_NAME if SIBLING_DMC_KEEP_AT_EVERY_ALPHA
            else DMC_SWEEP_METRIC_NAME)


def _sibling_sweep_pc_pair(family, sweeps):
    """((pc_x, pc_y), score, metric_name) of the plane a rule-cue sweep is
    drawn in, chosen on the sweep's own fixed points.

    DelayDM: the pair whose fixed points stay best separated by stimulus
    direction at every alpha (`best_sweep_pc_pair`). DMC: the PC on which the
    response clusters keep their order from alpha = 0 to 1 beside the PC on
    which they swap under the sibling's cue (`best_dmc_sweep_pc_pair`; the
    kept PC at both ends, or at every alpha with
    SIBLING_DMC_KEEP_AT_EVERY_ALPHA), so each row of the figure shows the
    clusters: what was seen is kept, the response flips. `sweeps` maps each
    base rule to its sweep.
    """
    projections = [np.asarray(sweep["proj"], dtype=float) for sweep in sweeps.values()]
    if family == "dmcgo":
        rules = _MULTITASK_SIBLING_FAMILIES[family]
        return best_dmc_sweep_pc_pair(
            projections,
            [_cue_interp_sweep_groups(family, rules, base_rule, sweep)
             for base_rule, sweep in sweeps.items()],
            keep_at_every_alpha=SIBLING_DMC_KEEP_AT_EVERY_ALPHA)
    return best_sweep_pc_pair(
        projections,
        [np.asarray(sweep["stim_idx"], dtype=int) for sweep in sweeps.values()])


def _require_sibling_converged(is_fixed, context):
    """Raise unless a saved convergence mask exists and is all True."""
    if is_fixed is None:
        raise ValueError(f"{context} predate settling (no is_fixed mask); rerun "
                         "multiple_task/sibling_delay_analysis.py")
    mask = np.asarray(is_fixed, dtype=bool)
    if not mask.all():
        raise ValueError(f"{context}: {int((~mask).sum())}/{mask.size} points "
                         "above the fixed-point criterion")


def _sibling_run_scores(family, aname):
    """({figure: {representation: best-pair score}},
    {representation: sweep legibility}) of one cached run.

    Each figure is scored with the metric it picks its PC pair with, on the
    fixed points it draws: "endpoint" by `_best_sibling_endpoint_pc_pair`,
    "sweep" by `_sibling_sweep_pc_pair` on the long-delay rule-cue sweep; the
    sweep's drawn trials in that pair's panels also get their legibility
    (`_cue_interp_drawn_legibility`, None for DMC).
    Raises ValueError when a cache is missing, predates settling or holds a
    point above the fixed-point criterion, so such a run is never shown.
    """
    run_dir = TWO_IN_MULTIPLES_DIR / aname
    paths = {
        "endpoint": (run_dir
                     / f"{family}_long_delay_endpoint_pc_projections_{aname}.pkl"),
        "sweep": _multitask_cue_interp_projection_path(
            run_dir, family, aname, "long_delay_endpoint"),
    }
    caches = {}
    for kind, path in paths.items():
        if not path.exists():
            raise ValueError(f"no {path.name}")
        with path.open("rb") as stream:
            caches[kind] = pickle.load(stream)
    if caches["sweep"].get("kind") != "cue_interp":
        raise ValueError(f"{paths['sweep'].name} is not a rule-cue sweep projection")

    scores = {figure: {} for figure in _SIBLING_FIGURES}
    legibility = {}
    for plot_name in _SIBLING_REPRESENTATIONS:
        entry = caches["endpoint"].get("representations", {}).get(plot_name)
        sweeps = (caches["sweep"].get("representations", {})
                  .get(plot_name, {}).get("sweeps"))
        if entry is None or not sweeps:
            raise ValueError(f"no {plot_name} endpoints or rule-cue sweep")
        _require_sibling_converged(entry.get("is_fixed"), f"{plot_name} endpoints")
        for base_rule, sweep in sweeps.items():
            _require_sibling_converged(sweep.get("is_fixed"),
                                       f"{plot_name} sweep points from {base_rule}")
        scores["endpoint"][plot_name] = float(
            _best_sibling_endpoint_pc_pair(entry, family)[1])
        pair, sweep_score, _ = _sibling_sweep_pc_pair(family, sweeps)
        scores["sweep"][plot_name] = float(sweep_score)
        legibility[plot_name] = _cue_interp_drawn_legibility(sweeps, family, pair)
    return scores, legibility


def _sibling_sweep_ranked_score(score, legibility):
    """A sweep figure's best-pair `score` weighed by the smoothness of its
    drawn trials: times min(1, _SIBLING_ATTRACTOR_SWITCH_FRACTION / largest
    drawn switch score), so a run whose drawn lines jump no further than
    the switch threshold keeps its score. `legibility` None (DMC) leaves the
    score as is."""
    if legibility is None:
        return score
    largest = legibility["largest_attractor_switch"]
    if largest <= _SIBLING_ATTRACTOR_SWITCH_FRACTION:
        return score
    return score * _SIBLING_ATTRACTOR_SWITCH_FRACTION / largest


def _sibling_run_rank_key(row, figure, plot_name):
    """Sort key of one scored run for one figure, best first: a DelayDM sweep
    figure ranks the runs whose drawn directions collapse
    (direction spread below _SIBLING_MIN_DIRECTION_SPREAD) after the rest
    and orders each group by `_sibling_sweep_ranked_score`; every other
    figure by its score. Ties go to the lowest seed."""
    score = row["scores"][figure][plot_name]
    legibility = row["sweep_legibility"][plot_name] if figure == "sweep" else None
    if legibility is None:
        return (False, -score, row["seed"])
    return (legibility["direction_spread"] < _SIBLING_MIN_DIRECTION_SPREAD,
            -_sibling_sweep_ranked_score(score, legibility), row["seed"])


def _sibling_run_ranking(family):
    """Score every cached SIBLING_L2 run of `family` and rank them for every
    figure and representation (`_sibling_run_rank_key`). Runs that cannot be
    scored are listed with the reason."""
    import re

    pattern = re.compile(r"everything_seed(\d+)_" + re.escape(
        f"{multitask_feature(SIBLING_L2)}{_MULTITASK_RUN_SUFFIX}"))
    runs, skipped = [], []
    for run_dir in TWO_IN_MULTIPLES_DIR.glob(multitask_aname("*", SIBLING_L2)):
        match = pattern.fullmatch(run_dir.name)
        if match is None or not run_dir.is_dir():
            continue
        seed = int(match.group(1))
        try:
            scores, legibility = _sibling_run_scores(family, run_dir.name)
        except (KeyError, ValueError, OSError, EOFError,
                pickle.UnpicklingError) as exc:
            skipped.append({"aname": run_dir.name, "seed": seed, "reason": str(exc)})
            continue
        runs.append({"aname": run_dir.name, "seed": seed, "scores": scores,
                     "sweep_legibility": legibility})
    runs.sort(key=lambda row: row["seed"])
    skipped.sort(key=lambda row: row["seed"])
    choices = {}
    for figure in _SIBLING_FIGURES:
        for plot_name in _SIBLING_REPRESENTATIONS:
            ranked = sorted(runs, key=lambda row: _sibling_run_rank_key(
                row, figure, plot_name))
            ranking = []
            for row in ranked:
                entry = {"seed": row["seed"],
                         "score": row["scores"][figure][plot_name]}
                legible = row["sweep_legibility"][plot_name]
                if figure == "sweep" and legible is not None:
                    entry.update(legible)
                    entry["ranked_score"] = _sibling_sweep_ranked_score(
                        entry["score"], legible)
                ranking.append(entry)
            choices[f"{figure} {plot_name}"] = {
                "best_aname": ranked[0]["aname"] if ranked else None,
                "ranking": ranking,
            }
    dmc = family == "dmcgo"
    smoothness = ""
    if family in _SIBLING_ATTRACTOR_SWITCH_FAMILIES:
        smoothness = (
            f" x min(1, {_SIBLING_ATTRACTOR_SWITCH_FRACTION:g} / largest drawn "
            "attractor switch); runs whose drawn directions span less than "
            f"{_SIBLING_MIN_DIRECTION_SPREAD:g} of a panel last")
    return {
        "family": family,
        "cohort": multitask_feature(SIBLING_L2),
        "metrics": {
            "endpoint": "best-pair " + (DMCGO_METRIC_NAME if dmc else DELAYDM_METRIC_NAME),
            "sweep": ("best-pair "
                      + (_dmc_sweep_metric_name() if dmc else SWEEP_METRIC_NAME)
                      + smoothness),
        },
        "n_candidates": len(runs) + len(skipped),
        "choices": choices,
        "runs": runs,
        "skipped": skipped,
    }


def _select_sibling_runs(family):
    """Rank `family`'s cached runs for every figure, report the rankings and
    return {"<figure> <representation>": run}.

    The rankings are printed and saved as
    multitask_sibling_run_scores_{delaydm,dmcgo}.json. With no scorable run
    every figure gets the cohort's seed-0 name, so the figures print their
    usual "Skipped" messages.
    """
    report = _sibling_run_ranking(family)
    _ensure_out_dir()
    report_path = _multitask_out(
        f"sibling_run_scores_{_MULTITASK_SIBLING_OUTPUT_NAMES[family]}.json")
    with report_path.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    metrics = "; ".join(f"{figure}: {metric}"
                        for figure, metric in report["metrics"].items())
    print(f"  {family} sibling runs: {len(report['runs'])}/"
          f"{report['n_candidates']} cached {report['cohort']} runs scored "
          f"({metrics}); saved {report_path}")
    for row in report["runs"]:
        scores = ", ".join(f"{figure} {plot_name} {value:.3f}"
                           for figure, by_rep in row["scores"].items()
                           for plot_name, value in by_rep.items())
        legibility = "".join(
            f"; sweep {plot_name} largest switch "
            f"{legible['largest_attractor_switch']:.2f}, direction spread "
            f"{legible['direction_spread']:.3f}"
            for plot_name, legible in row["sweep_legibility"].items()
            if legible is not None)
        print(f"    seed {row['seed']}: {scores}{legibility}")
    for row in report["skipped"]:
        print(f"    Skipped seed {row['seed']}: {row['reason']}")
    if not report["runs"]:
        print(f"  {family}: no scorable {report['cohort']} run under "
              f"{TWO_IN_MULTIPLES_DIR}/. Run: python "
              "multiple_task/sibling_delay_analysis.py --seed <seed> "
              f"--feature {report['cohort']} --families {family} "
              "--method long_delay_endpoint")
        return {key: multitask_aname(0, SIBLING_L2) for key in report["choices"]}
    for key, choice in report["choices"].items():
        best = choice["ranking"][0]
        ranked = ("" if "ranked_score" not in best else
                  f", {best['ranked_score']:.3f} weighed by its largest drawn "
                  f"switch {best['largest_attractor_switch']:.2f}")
        print(f"  {family} {key} run selected automatically: seed {best['seed']} "
              f"(score {best['score']:.3f}{ranked})")
    return {key: choice["best_aname"] for key, choice in report["choices"].items()}


def _multitask_sibling_hint(family, method="gradient", aname=None):
    """Reconstruct the upstream command for one sibling run: `aname`, or by
    default the family's pinned run (a generic hint when none is pinned)."""
    import re as _re

    if aname is None:
        aname = _pinned_sibling_aname(family) or ""
    match = _re.fullmatch(
        r"everything_seed(\d+)_(.+)\+hidden\d+\+batch\d+\+angle", aname)
    if match is None:
        return "Run multiple_task/sibling_delay_analysis.py first."
    if method not in ("gradient", "long_delay_endpoint"):
        raise ValueError(f"unknown sibling fixed-point method {method!r}")
    seed, feature = match.groups()
    return ("Run: python multiple_task/sibling_delay_analysis.py "
            f"--seed {seed} --feature {feature} --families {family} "
            f"--method {method}")


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
    """Independent x/y limits from the joint extent of the drawn points."""
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


def _sibling_alignment_metrics(family, rules, aname, rep_key="fixed_WM",
                               probe="longdelay"):
    """Quantify sibling-task translation and geometry before PCA, from the
    gradient fixed points of run `aname`.

    Translation explained is the fraction of the mean squared paired
    cross-task displacement accounted for by one shared displacement vector.
    Geometry correlation compares the two within-task pairwise-distance
    matrices and is therefore translation invariant. These metrics are computed
    before PCA; the trajectory-PC panels below only visualize them.
    """
    run_dir = TWO_IN_MULTIPLES_DIR / aname
    records = []
    for rule in rules:
        path = run_dir / f"fixed_points_grad_{aname}_{rule}.pkl"
        data = _load_pkl_or_skip(path, _multitask_sibling_hint(family, aname=aname))
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
        records.append((values[order], ordered_stim, ordered_mag))

    if (not np.array_equal(records[0][1], records[1][1])
            or not np.allclose(records[0][2], records[1][2],
                               atol=1e-8, rtol=0)):
        raise ValueError(f"{family} fixed-point pickles do not contain matching "
                         "stimulus×magnitude conditions")
    first, second = records[0][0], records[1][0]
    if first.shape[0] < 3:
        raise ValueError("need at least three matched sibling-task conditions")

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
        "n_pairs": int(first.shape[0]),
        "raw_rms": float(np.sqrt(total_squared)),
        "residual_rms": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
    }


def _plot_multitask_sibling_fixed_point_geometry_representation(
        data, family, rules, plot_name, rep_key, output_suffix):
    """Render one sibling-family representation before/after translation.

    Both panels use that representation's configured joint delay-trajectory PC
    plane. Panel B subtracts only the mean paired modality displacement from
    modality 2. It never refits PCA and performs no rotation or scaling.
    Quantitative annotations are computed in the corresponding original
    high-dimensional representation, before PCA. `data` is the gradient
    projection cache of the run of this representation's endpoint figure.
    """
    aname = _multitask_sibling_aname(family, plot_name)
    pc_label = "Joint Delay"
    path = (TWO_IN_MULTIPLES_DIR / aname
            / f"{family}_delay_pc_projections_{aname}.pkl")
    entry = data.get("representations", {}).get(plot_name)
    if entry is None:
        raise KeyError(f"{path}: missing representation {plot_name!r}")

    projection = np.asarray(entry["proj"], dtype=float)
    task_idx = np.asarray(entry["task_idx"], dtype=int)
    first_idx, second_idx = _matched_delaydm_projection_indices(entry)
    if first_idx.size < 3:
        print(f"  Skipped {family}: only {first_idx.size} matched fixed-point "
              "pair(s); at least 3 are required for geometry metrics.")
        return False

    try:
        pc_x, pc_y = _MULTITASK_SIBLING_PC_PLANES[family][plot_name]
    except KeyError as exc:
        raise KeyError(f"Set _MULTITASK_SIBLING_PC_PLANES[{family!r}]"
                       f"[{plot_name!r}] "
                       "in paper_plot.py") from exc
    pc_x, pc_y, bx, by = _paper_pc_indices(
        (pc_x, pc_y), projection,
        f"delayDM fixed-point geometry/joint/{plot_name}")

    # Because PCA projection is linear, subtracting this six-PC translation is
    # exactly the projection of subtracting its high-dimensional counterpart.
    translation = np.mean(
        projection[second_idx] - projection[first_idx], axis=0)
    aligned = projection.copy()
    aligned[task_idx == 1] -= translation
    metrics = _sibling_alignment_metrics(
        family, rules, aname, rep_key=rep_key, probe="longdelay")
    if metrics is None:
        return False

    task_names = list(entry.get("task_names", rules))
    # Colored as the endpoint and sweep figures are: DelayDM by stimulus
    # direction, DMC by response category.
    labels, _ = _sibling_fixed_point_color_labels(family, entry)
    color_for = _sibling_color_for(family, int(labels.max()) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.75), squeeze=False)
    panels = (projection, aligned)
    for panel_index, (ax, shown) in enumerate(zip(axs[0], panels)):
        # Matched-condition connectors expose the displacement field in panel A
        # and the remaining non-translational mismatch in panel B.
        for i, j in zip(first_idx, second_idx):
            ax.plot([shown[i, bx], shown[j, bx]],
                    [shown[i, by], shown[j, by]],
                    color="0.55", linewidth=0.45,
                    alpha=(0.18 if panel_index == 0 else 0.28), zorder=1)

        for task, rule in enumerate(task_names):
            sel_task = task_idx == task
            for label in np.unique(labels[sel_task]):
                sel = sel_task & (labels == label)
                ax.scatter(shown[sel, bx], shown[sel, by], color=color_for(label),
                           edgecolor="white", linewidth=0.25,
                           marker=_MULTITASK_RULE_MARKERS[task % 2],
                           s=31, alpha=0.88, zorder=3)

        if panel_index == 0:
            c0 = shown[first_idx][:, [bx, by]].mean(axis=0)
            c1 = shown[second_idx][:, [bx, by]].mean(axis=0)
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
        ax.set_box_aspect(1)
        ax.set_xlabel(f"{pc_label} PC{pc_x}", fontsize=8.5)
        if panel_index == 0:
            ax.set_ylabel(f"{pc_label} PC{pc_y}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([], [],
                          marker=_MULTITASK_RULE_MARKERS[t % 2],
                          markerfacecolor="0.45", markeredgecolor="white",
                          color="none", markersize=5.5,
                          label=_TASK_DISPLAY.get(rule, rule))
               for t, rule in enumerate(task_names)]
    handles += _sibling_category_handles(family, labels)
    _legend(axs[0, 0], handles=handles, frameon=False, fontsize=6.5,
            loc="best", handletextpad=0.3)
    fig.tight_layout(w_pad=1.5)
    _save_fig(
        fig,
        _multitask_out(
            f"{_MULTITASK_SIBLING_OUTPUT_NAMES[family]}_fixed_point_geometry_"
            f"{output_suffix}.png"),
        extra=(f"  ({pc_label} PC{pc_x}-PC{pc_y}; "
               f"representation={rep_key}; "
               f"translation={100 * metrics['translation_explained']:.1f}%; "
               f"geometry r={metrics['geometry_r']:.3f}; "
               f"n={metrics['n_pairs']})"))
    return True


def _sibling_fixed_point_color_labels(family, entry):
    """(labels, name) that color the fixed points of a sibling projection
    (long-delay endpoints or gradient fixed points), matching the lines of the
    rule-cue sweep figures: the stimulus direction for DelayDM, the response
    category for DMC ("respond to category 1 / 2", the saved `group_labels`,
    or derived from task and stimulus as
    `sibling_delay_analysis._dmc_category_labels` derives them)."""
    if family not in ("delaydm1", "dmcgo"):
        raise ValueError(f"no long-delay clustering criterion for {family!r}")
    if "stim_idx" not in entry:
        raise KeyError(f"{family} endpoint projection has no stim_idx labels")
    stim = np.asarray(entry["stim_idx"], dtype=int)
    if family == "delaydm1":
        return stim, "stimulus direction"
    if "group_labels" in entry:
        return np.asarray(entry["group_labels"], dtype=int), "response category"
    task = np.asarray(entry["task_idx"], dtype=int)
    half = max(_SIBLING_N_STIM, int(stim.max()) + 1) // 2
    return ((task == 0) == (stim < half)).astype(int), "response category"


def _sibling_color_for(family, n_colors):
    """label -> color of the sibling fixed-point figures: the response
    cluster's color for DMC (`_CUE_INTERP_CLUSTER_COLORS`), the stimulus
    rainbow over `n_colors` directions for DelayDM."""
    if family == "dmcgo":
        return lambda label: _CUE_INTERP_CLUSTER_COLORS[int(label)]
    return lambda label: stim_color(int(label), n_colors)


def _sibling_category_handles(family, labels):
    """Legend lines for the response categories coloring a DMC figure, in the
    sweep figures' order and colors; none for DelayDM, whose stimulus colors
    are keyed by the stimulus colorwheel."""
    if family != "dmcgo":
        return []
    categories = sorted({int(label) for label in np.unique(labels)}, reverse=True)
    return [plt.Line2D([], [], color=_CUE_INTERP_CLUSTER_COLORS[label],
                       linewidth=1.4, label=_CUE_INTERP_CLUSTER_NAMES[label])
            for label in categories]


def _sibling_sweep_color_labels(family, rules, base_rule, sweep):
    """Per-trial labels coloring a sweep overlaid on the endpoint figure: the
    trial's response cluster at alpha = 0 for DMC
    (`_cue_interp_sweep_groups`), its stimulus direction for DelayDM."""
    groups = _cue_interp_sweep_groups(family, rules, base_rule, sweep)
    return np.asarray(sweep["stim_idx"], dtype=int) if groups is None else groups


def _best_sibling_endpoint_pc_pair(entry, family):
    """Select the six-PC pair that best separates the fixed points, both
    sibling tasks pooled (`best_task_specific_pc_pair`): by stimulus direction
    for DelayDM once each task's mean is removed, by response category for
    DMC."""
    projection = np.asarray(entry["proj"], dtype=float)
    return best_task_specific_pc_pair(
        projection, family,
        stim_idx=entry.get("stim_idx"),
        group_labels=entry.get("group_labels"),
        task_idx=entry.get("task_idx"))


# Families whose endpoint figure draws its selected plane task-aligned (each
# task's mean removed), the coordinates its PC pair is scored in.
_SIBLING_TASK_ALIGNED_FAMILIES = ("delaydm1",)


# Overlay the long-delay rule-cue sweep (see plot_multitask_sibling_cue_interp)
# on the endpoint figure when its projection cache exists. The sweep is
# projected onto the same joint basis, so its points sit in the same plane.
_MULTITASK_ENDPOINT_OVERLAY_CUE_INTERP = True


def _cue_interp_endpoint_overlay(family, plot_name):
    """(alphas, sweeps) of `family`'s long-delay rule-cue sweep, or None.

    Reads the endpoint-method projection written by
    `sibling_delay_analysis.save_cue_interp_pc_projections` from the run of
    the `plot_name` endpoint figure, whose basis the sweep shares; a missing
    or foreign cache only drops the overlay, never the endpoint figure."""
    if not _MULTITASK_ENDPOINT_OVERLAY_CUE_INTERP:
        return None
    aname = _multitask_sibling_aname(family, plot_name)
    path = _multitask_cue_interp_projection_path(
        TWO_IN_MULTIPLES_DIR / aname, family, aname, "long_delay_endpoint")
    if not path.exists():
        print(f"  {family} long-delay {plot_name}: no rule-cue sweep cache "
              f"({path.name}); drawing endpoints only.")
        return None
    with open(path, "rb") as f:
        sweep_data = pickle.load(f)
    entry = sweep_data.get("representations", {}).get(plot_name)
    if sweep_data.get("kind") != "cue_interp" or not entry or not entry.get("sweeps"):
        print(f"  {family} long-delay {plot_name}: {path.name} holds no "
              "rule-cue sweep for this representation; drawing endpoints only.")
        return None
    return (np.asarray(sweep_data["alphas"], dtype=float),
            _cue_interp_sweeps(entry, family))


def _draw_cue_interp_endpoint_overlay(ax, alphas, sweeps, task_names, bx, by,
                                      color_for, sweep_labels, offsets=None):
    """Overlay every sweep trial's path across alpha on one endpoint panel.

    Hue follows the panel's convention, `color_for` of each trial's entry in
    `sweep_labels` (base rule -> labels, `_sibling_sweep_color_labels`: the
    stimulus direction for DelayDM, the alpha = 0 response cluster for DMC),
    so the cue is carried by marker shape as it is for the endpoints. The
    alpha = 0 start gets no marker of its own: the sweep reruns the base
    rule's endpoint trials (same generator state, unchanged input at
    alpha = 0), so each path starts on that trial's endpoint, already drawn,
    up to the drift between the two runs' settle times. The alpha = 1 end
    takes the sibling's marker, hollow, marking where the sweep ends rather
    than an endpoint of the sibling's own trials. For
    DelayDM that end is an input never seen in training (each rule presents
    its stimulus on its own modality); for DMC it is an ordinary DMCNoGo
    input for the same stimulus, so the hollow marker lands on DMCNoGo's own
    endpoint. Intermediate alphas are dots along a dotted path. On a
    task-aligned panel `offsets` (task -> mean endpoint, as `task_offsets`
    returns) shifts each path by its base task's offset, the frame its
    alpha = 0 start shares with that task's endpoints."""
    for base_rule, sweep in sweeps.items():
        base_task = task_names.index(base_rule)
        target_task = task_names.index(sweep["target_rule"])
        proj = np.asarray(sweep["proj"], dtype=float)      # (n_alpha, n_points, n_pc)
        if offsets is not None:
            proj = proj - offsets[base_task]
        labels = sweep_labels[base_rule]
        for s in range(proj.shape[1]):
            path = proj[:, s][:, [bx, by]]
            finite = np.isfinite(path).all(axis=1)
            if np.count_nonzero(finite) < 2:
                continue
            color = color_for(labels[s])
            ax.plot(path[finite, 0], path[finite, 1], linestyle=":",
                    color=color, linewidth=0.7, alpha=0.7, zorder=2)
            if alphas.size > 2:
                mid = path[1:-1][finite[1:-1]]
                ax.scatter(mid[:, 0], mid[:, 1], color=color, marker=".",
                           s=7, edgecolor="none", alpha=0.7, zorder=2)
            end = path[-1]
            if np.isfinite(end).all():
                ax.scatter(end[0], end[1], facecolor="none", edgecolor=color,
                           marker=_MULTITASK_RULE_MARKERS[target_task % 2],
                           s=31, linewidth=0.8, alpha=0.95, zorder=4)


def _plot_multitask_long_delay_endpoint_representation(
        data, family, rules, plot_name, output_suffix, cue_sweep=None):
    """Plot default PC1-PC2 beside the best long-delay endpoint PC pair.

    Endpoints and trajectories are colored as the sweep figures' lines are
    (`_sibling_fixed_point_color_labels`): DelayDM by stimulus direction, DMC
    by response category (`_CUE_INTERP_CLUSTER_COLORS`); marker shape and
    trajectory line style give the task. For DelayDM the best pair is scored
    and drawn task-aligned, each task's mean endpoint removed, so the same
    stimulus of both tasks overlaps; PC1-PC2 stays raw and shows the offset.
    `cue_sweep` is the family's long-delay rule-cue sweep as returned by
    `_cue_interp_endpoint_overlay` (the caller loads it); when given, each
    sweep trial's path across alpha is overlaid in the same plane
    (`_draw_cue_interp_endpoint_overlay`), the limits include its points, and
    the figure and its legend are saved under a `_cue_interp` suffix so the
    plain figure keeps its name."""
    name_suffix = "" if cue_sweep is None else "_cue_interp"
    entry = data.get("representations", {}).get(plot_name)
    if entry is None:
        print(f"  Skipped {family} long-delay {plot_name}: representation "
              "not available.")
        return False
    if "trajectory_proj" not in entry:
        print(f"  Skipped {family} long-delay {plot_name}: projected delay "
              "trajectories are not available. Regenerate with "
              + _multitask_sibling_hint(
                  family, "long_delay_endpoint",
                  aname=_multitask_sibling_aname(family, plot_name)))
        return False
    try:
        pair, score, metric_name = _best_sibling_endpoint_pc_pair(
            entry, family)
    except (KeyError, ValueError) as exc:
        print(f"  Skipped {family} long-delay {plot_name}: {exc}")
        return False

    projection = np.asarray(entry["proj"], dtype=float)
    trajectories = np.asarray(entry["trajectory_proj"], dtype=float)
    task_idx = np.asarray(entry["task_idx"], dtype=int)
    labels, _ = _sibling_fixed_point_color_labels(family, entry)
    if (task_idx.shape != labels.shape
            or trajectories.ndim != 3
            or trajectories.shape[0] != projection.shape[0]
            or trajectories.shape[2] != projection.shape[1]):
        print(f"  Skipped {family} long-delay {plot_name}: endpoint and "
              "trajectory shapes disagree.")
        return False
    best_pc_x, best_pc_y, best_bx, best_by = _paper_pc_indices(
        pair, projection, f"{family}/long-delay-endpoint/{plot_name}")
    default_pc_x, default_pc_y, default_bx, default_by = _paper_pc_indices(
        _MULTITASK_SIBLING_ENDPOINT_DEFAULT_PC_PLANE, projection,
        f"{family}/long-delay-endpoint/{plot_name}/default")
    task_names = list(entry.get("task_names", rules))
    if len(task_names) != len(rules):
        task_names = list(rules)

    overlay = cue_sweep
    n_colors = int(labels.max()) + 1
    if overlay is not None:
        alphas, sweeps = overlay
        unknown = [rule for rule in sweeps if rule not in task_names]
        if unknown:
            print(f"  {family} long-delay {plot_name}: sweep base rules "
                  f"{unknown} are not in {task_names}; drawing endpoints only.")
            overlay = None
        else:
            sweep_labels = {
                base_rule: _sibling_sweep_color_labels(family, rules, base_rule, sweep)
                for base_rule, sweep in sweeps.items()}
            n_colors = max(n_colors, 1 + max(
                int(np.max(trial_labels)) for trial_labels in sweep_labels.values()))
    color_for = _sibling_color_for(family, n_colors)

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), squeeze=False)
    trajectory_linestyles = ("-", "--")
    evr = np.asarray(entry.get("explained_variance_ratio", []), dtype=float)
    # The selected plane of a task-aligned family is drawn as it is scored,
    # each task's mean endpoint removed (paths and the sweep shifted with
    # their task); the default PC1-PC2 plane keeps the raw coordinates, which
    # show the task offset itself.
    aligned = family in _SIBLING_TASK_ALIGNED_FAMILIES
    offsets = task_offsets(projection, task_idx) if aligned else None
    panel_specs = (
        (axs[0, 0], default_pc_x, default_pc_y, default_bx, default_by, None),
        (axs[0, 1], best_pc_x, best_pc_y, best_bx, best_by, offsets),
    )
    for ax, pc_x, pc_y, bx, by, panel_offsets in panel_specs:
        shift = (np.zeros_like(projection) if panel_offsets is None
                 else np.stack([panel_offsets[task] for task in task_idx]))
        points = projection - shift
        paths = trajectories - shift[:, None, :]
        limit_points = points
        if overlay is not None:
            limit_points = np.vstack([points] + [
                (np.asarray(sweep["proj"], dtype=float)
                 - (0.0 if panel_offsets is None
                    else panel_offsets[task_names.index(base_rule)])
                 ).reshape(-1, projection.shape[1])
                for base_rule, sweep in sweeps.items()])
        for trial_index in range(projection.shape[0]):
            path = paths[trial_index][:, [bx, by]]
            finite = np.isfinite(path).all(axis=1)
            if np.count_nonzero(finite) < 2:
                continue
            task = int(task_idx[trial_index])
            ax.plot(
                path[finite, 0], path[finite, 1],
                color=color_for(labels[trial_index]), linewidth=0.65,
                alpha=0.24,
                linestyle=trajectory_linestyles[task % 2], zorder=1)
            ax.scatter(
                path[finite, 0][0], path[finite, 1][0],
                color=color_for(labels[trial_index]), edgecolor="none",
                marker=".", s=8, alpha=0.35, zorder=2)

        finite_endpoints = np.isfinite(points[:, [bx, by]]).all(axis=1)
        for task, rule in enumerate(task_names):
            for label in np.unique(labels[task_idx == task]):
                selected = ((task_idx == task) & (labels == label)
                            & finite_endpoints)
                if np.any(selected):
                    ax.scatter(
                        points[selected, bx], points[selected, by],
                        color=color_for(label), edgecolor="white",
                        linewidth=0.25,
                        marker=_MULTITASK_RULE_MARKERS[task % 2], s=31,
                        alpha=0.88, zorder=3)
        if overlay is not None:
            _draw_cue_interp_endpoint_overlay(
                ax, alphas, sweeps, task_names, bx, by, color_for, sweep_labels,
                offsets=panel_offsets)

        x_var = f" ({100 * evr[bx]:.1f}%)" if evr.size > bx else ""
        y_var = f" ({100 * evr[by]:.1f}%)" if evr.size > by else ""
        aligned_note = "" if panel_offsets is None else ", task-aligned"
        ax.set_xlabel(f"Joint Delay PC{pc_x}{x_var}{aligned_note}")
        ax.set_ylabel(f"Joint Delay PC{pc_y}{y_var}{aligned_note}")
        limit_padding = _MULTITASK_ENDPOINT_LIMIT_PADDING
        if family == "delaydm1" and plot_name == "e_modulation":
            limit_padding = _MULTITASK_DELAYDM_EMODULATION_LIMIT_PADDING
        xlim, ylim = _adaptive_pc_limits(
            limit_points, bx, by, padding=limit_padding)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        ax.spines[["top", "right"]].set_visible(False)

    if SHOW_LEGEND and plot_name == "e_modulation":
        handles = [
            plt.Line2D([], [], marker=_MULTITASK_RULE_MARKERS[task % 2],
                       markerfacecolor="0.45", markeredgecolor="white",
                       color="0.45", linewidth=0.7,
                       linestyle=trajectory_linestyles[task % 2], markersize=5.5,
                       label=_TASK_DISPLAY.get(rule, rule))
            for task, rule in enumerate(task_names)
        ]
        category_handles = _sibling_category_handles(family, labels)
        handles += category_handles
        legend_height = 0.65 + 0.3 * len(category_handles)
        if overlay is not None:
            # One entry per sweep direction: the hollow end marker is the
            # sibling's shape, so each direction needs its own handle.
            for base_rule, sweep in sweeps.items():
                target_task = task_names.index(sweep["target_rule"])
                handles.append(plt.Line2D(
                    [], [], marker=_MULTITASK_RULE_MARKERS[target_task % 2],
                    markerfacecolor="none", markeredgecolor="0.45",
                    color="0.45", linewidth=0.7, linestyle=":", markersize=5.5,
                    label=(f"{_TASK_DISPLAY.get(base_rule, base_rule)} stimulus, "
                           f"{_TASK_DISPLAY.get(sweep['target_rule'], sweep['target_rule'])} "
                           r"cue ($\alpha$ 0 $\to$ 1)")))
            legend_height += 0.3 * len(sweeps)
        legend_fig = plt.figure(figsize=(2.0, legend_height))
        legend_fig.legend(handles=handles, loc="center", frameon=False,
                          fontsize=8, handletextpad=0.5)
        _save_fig(
            legend_fig,
            _multitask_out(
                f"{_MULTITASK_SIBLING_OUTPUT_NAMES[family]}_long_delay_endpoint_"
                f"legend{name_suffix}.png"))
    fig.tight_layout(w_pad=1.5)
    overlay_note = ("" if overlay is None else
                    f"; cue sweep overlaid: {alphas.size} alphas, bases {list(sweeps)}"
                    f"{_cue_interp_view_note(sweeps)}")
    # Hollow markers here already mean the sweep's alpha = 1 end, so endpoints
    # left above the residual criterion are reported rather than drawn hollow.
    unsettled = int(np.sum(~np.asarray(
        entry.get("is_fixed", np.ones(projection.shape[0], bool)), dtype=bool)))
    unsettled_note = ("" if unsettled == 0 else
                      f"; {unsettled} endpoints above the fixed-point criterion")
    _save_fig(
        fig,
        _multitask_out(
            f"{_MULTITASK_SIBLING_OUTPUT_NAMES[family]}_long_delay_endpoint_"
            f"{output_suffix}{name_suffix}.png"),
        extra=(f"  (default PC1-PC2; best PC{best_pc_x}-PC{best_pc_y}"
               f"{', task-aligned' if aligned else ''}; "
               f"metric={metric_name}; score={score:.3f}{overlay_note}"
               f"{unsettled_note})"))
    return True


def plot_multitask_delaydm_fixed_point_geometry():
    """Sibling-task translation analysis for DelayDM and DMCGo families.

    Produces effective-modulation and hidden-state figures for each family.
    All use their own family's joint delay-trajectory PCA only for
    visualization; quantitative metrics use the original high-dimensional
    representation. Each representation reads the run of its endpoint figure
    (`_multitask_sibling_aname`); a run's cache is loaded once however many
    representations use it.
    """
    for family, rules in _MULTITASK_SIBLING_FAMILIES.items():
        loaded = {}
        for plot_name, rep_key, output_suffix in (
                ("e_modulation", "fixed_WM", "emodulation"),
                ("hidden", "fixed_hidden", "hidden")):
            aname = _multitask_sibling_aname(family, plot_name)
            if aname not in loaded:
                path = (TWO_IN_MULTIPLES_DIR / aname
                        / f"{family}_delay_pc_projections_{aname}.pkl")
                loaded[aname] = _load_pkl_or_skip(
                    path, _multitask_sibling_hint(family, aname=aname))
            if loaded[aname] is None:
                continue
            rendered = _plot_multitask_sibling_fixed_point_geometry_representation(
                loaded[aname], family, rules, plot_name, rep_key, output_suffix)
            if rendered is False:
                break

    # Long-delay endpoints are an additional visualization, not a replacement
    # for the gradient fixed-point geometry above. Missing endpoint artifacts
    # skip only these panels so gradient-only runs remain fully supported.
    for family, rules in _MULTITASK_SIBLING_FAMILIES.items():
        loaded = {}
        for plot_name, output_suffix in (
                ("e_modulation", "emodulation"),
                ("hidden", "hidden")):
            aname = _multitask_sibling_aname(family, plot_name)
            if aname not in loaded:
                path = (TWO_IN_MULTIPLES_DIR / aname
                        / f"{family}_long_delay_endpoint_pc_projections_{aname}.pkl")
                loaded[aname] = _load_pkl_or_skip(
                    path, _multitask_sibling_hint(
                        family, "long_delay_endpoint", aname=aname))
            data = loaded[aname]
            if data is None:
                continue
            # The plain figure first, then, when the rule-cue sweep is
            # cached, a second copy with the sweep overlaid (`_cue_interp`).
            _plot_multitask_long_delay_endpoint_representation(
                data, family, rules, plot_name, output_suffix)
            cue_sweep = _cue_interp_endpoint_overlay(family, plot_name)
            if cue_sweep is not None:
                _plot_multitask_long_delay_endpoint_representation(
                    data, family, rules, plot_name, output_suffix,
                    cue_sweep=cue_sweep)


# ─── Rule-cue interpolation between the sibling rules ────────────────────────
# The multi-task counterpart of the two-task alpha figures
# (`twotask_interp_alpha_fixed_points_3d` / `_2d`): the stimulus is held at one
# base rule's trials while its cue fades into the sibling's over alpha, and the
# delay fixed point (gradient) or long-delay endpoint is traced across alpha in
# the family's joint delay basis. The markers take the alpha ramp (SCHEME
# family 5), dark at alpha=0 -> light at alpha=1, filled = converged, hollow =
# over the residual threshold (long-delay endpoints settle until they meet it,
# so they are hollow only past the settling cap; caches written before
# settling carry no mask and draw filled); the connecting
# lines carry the plain endpoint figure's grouping instead (see
# `_cue_interp_line_colors`). Read from the projections that
# sibling_delay_analysis.save_cue_interp_pc_projections writes.
_MULTITASK_CUE_INTERP_METHODS = ("gradient", "long_delay_endpoint")
_MULTITASK_CUE_INTERP_PANEL_IN = 1.8      # in; matches the two-task alpha figures
_MULTITASK_CUE_INTERP_LIMIT_PADDING = 0.12   # 3D view, shared over panels
_MULTITASK_CUE_INTERP_Y_PADDING = 0.08       # `2d` view, per panel, below the line range
_MULTITASK_CUE_INTERP_Y_PADDING_TOP = 0.25   # `2d` view, per panel, above the line range


def _multitask_cue_interp_projection_path(run_dir, family, aname, method):
    """Projection pickle of one family's rule-cue sweep for `method`."""
    if method == "gradient":
        return run_dir / f"{family}_cue_interp_delay_pc_projections_{aname}.pkl"
    if method == "long_delay_endpoint":
        return (run_dir / f"{family}_cue_interp_long_delay_endpoint_"
                          f"pc_projections_{aname}.pkl")
    raise ValueError(f"unknown sibling fixed-point method {method!r}")


def _multitask_cue_interp_plane(family, plot_name, method, run_dir, aname,
                                entry=None):
    """1-based PC pair the sweep is drawn in.

    The plane is chosen on the sweep itself when `entry` (the
    representation's sweeps) is given (`_sibling_sweep_pc_pair`: for DelayDM
    the pair whose fixed points stay best separated by stimulus direction at
    every alpha, for DMC the PC on which the response clusters keep their
    order beside the PC on which they swap, in that row order). When the
    sweep cannot be scored, the base figure's plane is used: the configured
    joint plane for the gradient method (`_MULTITASK_SIBLING_PC_PLANES`) and,
    for the endpoint method, the pair that best separates the run's fixed
    points (`_best_sibling_endpoint_pc_pair`), read from its cached base
    endpoint projections, with the default PC1-PC2 plane when those are
    missing (announced). Returns (pair, note) where the note names how the
    plane was chosen.
    """
    if entry and entry.get("sweeps"):
        try:
            pair, score, metric_name = _sibling_sweep_pc_pair(family, entry["sweeps"])
            return tuple(pair), f"best {metric_name} = {score:.3f}"
        except ValueError as exc:
            print(f"  {family}/{plot_name}: sweep-based PC pair unavailable "
                  f"({exc}); falling back to the base figure's plane.")
    if method == "gradient":
        return tuple(_MULTITASK_SIBLING_PC_PLANES[family][plot_name]), "configured"
    base_path = (run_dir
                 / f"{family}_long_delay_endpoint_pc_projections_{aname}.pkl")
    if base_path.exists():
        with open(base_path, "rb") as f:
            base = pickle.load(f)
        entry = base.get("representations", {}).get(plot_name)
        if entry is not None:
            try:
                pair, _, metric_name = _best_sibling_endpoint_pc_pair(
                    entry, family)
                return tuple(pair), f"best {metric_name}"
            except (KeyError, ValueError) as exc:
                print(f"  {family}/{plot_name}: best endpoint PC pair "
                      f"unavailable ({exc}); using the default plane.")
    return tuple(_MULTITASK_SIBLING_ENDPOINT_DEFAULT_PC_PLANE), "default"


def _one_trial_per_stimulus(stim_idx, magnitude, switch_scores=None):
    """Indices of one trial per stimulus direction at matched magnitudes,
    preferring trials that stay on one attractor along the sweep.

    Against a target magnitude a trial costs its distance from the target
    plus, if it switches attractor (score above
    _SIBLING_ATTRACTOR_SWITCH_FRACTION), _SIBLING_SMOOTHNESS_WEIGHT times its
    attractor-switch score (`switch_scores`, from
    `_cue_interp_attractor_switch_scores`; none counts as 0). A score at or
    below the threshold is drift, not a jump, and costs nothing: on the
    DelayDM magnitude grid every direction has a trial at every level, and
    letting that drift break the tie between exact matches would move the
    target off 1.0. The target is the trials' own magnitude at which the
    costliest direction pays least for its cheapest trial (ties broken toward
    1.0); each direction then contributes its cheapest trial (lowest index on
    ties). Returns (indices in direction order, target). Without switch
    scores the magnitudes alone decide; DMC trials all have magnitude 1 and
    are not scored, so this simply deduplicates them."""
    stim = np.asarray(stim_idx, dtype=int)
    magnitude = np.asarray(magnitude, dtype=float)
    if stim.shape != magnitude.shape or stim.size == 0:
        raise ValueError("stimulus indices and magnitudes must be non-empty "
                         "and of equal length")
    penalty = np.zeros(stim.shape)
    if switch_scores is not None:
        scores = np.asarray(switch_scores, dtype=float)
        if scores.shape != stim.shape:
            raise ValueError(f"{scores.size} switch scores for {stim.size} trials")
        penalty = np.where(scores > _SIBLING_ATTRACTOR_SWITCH_FRACTION,
                           _SIBLING_SMOOTHNESS_WEIGHT * scores, 0.0)
    members = [np.flatnonzero(stim == d) for d in np.unique(stim)]
    best = None
    for target in np.unique(magnitude):
        worst = max(float(np.min(np.abs(magnitude[m] - target) + penalty[m]))
                    for m in members)
        key = (worst, abs(float(target) - 1.0))
        if best is None or key < best[0]:
            best = (key, float(target))
    target = best[1]
    chosen = [int(m[np.argmin(np.abs(magnitude[m] - target) + penalty[m])])
              for m in members]
    return np.asarray(chosen, dtype=int), target


# A sweep trial switches attractor when, between two consecutive alphas, it
# moves away from the common motion of all trials by more than this fraction
# of a PC's span over the sweep (in any of the six cached PCs). On the
# 2026-10-22 seed 62 hidden sweep the trials it flags score 0.10-0.28 and all
# hop in the full hidden space too; the rest score 0.008-0.1. On the magnitude
# grid caches (2026-10-26) it flags 11/40 and 22/40 of that sweep's trials,
# which depart from the common motion by 0.08-0.29 in the full hidden space
# against a median of 0.05-0.08 for the rest.
_SIBLING_ATTRACTOR_SWITCH_FRACTION = 0.1
# How much a switching trial's attractor-switch score costs in the one-trial
# view against its magnitude's distance from the target
# (`_one_trial_per_stimulus`; scores at or below
# _SIBLING_ATTRACTOR_SWITCH_FRACTION cost nothing): a jump just over a tenth
# of a PC's span weighs as much as a magnitude 0.2 off. On the seed 62 hidden
# sweep of the magnitude grid caches (2026-10-26), where a DelayDM2 direction
# has no trial that stays, the magnitudes alone draw jumps of 0.21 and 0.24
# (DelayDM1 and DelayDM2 bases), 1 draws 0.14 and 0.19, and 2 leaves DelayDM1
# without a switching trial (largest score 0.09) and DelayDM2 at 0.19, every
# direction within 0.2 of its target.
_SIBLING_SMOOTHNESS_WEIGHT = 2.0
# Families whose sweeps are scored for attractor switches: the one-trial view
# weighs them and the sweep figure's run ranking weighs and checks the drawn
# trials (`_cue_interp_drawn_legibility`). DMC is not, as every trial takes
# the response-cluster swap.
_SIBLING_ATTRACTOR_SWITCH_FAMILIES = ("delaydm1",)
# A DelayDM sweep run whose drawn direction means span less than this
# fraction of a `2d` panel's range is ranked after every other run: its
# directions draw as one line. On the magnitude grid caches (2026-10-26) the
# one-trial hidden sweeps of seeds 177, 42 and 510 span under 0.01 (every
# drawn trial on one attractor) and seed 62 0.91.
_SIBLING_MIN_DIRECTION_SPREAD = 0.05


def _cue_interp_attractor_switch_scores(proj):
    """Each sweep trial's largest departure from the common motion.

    `proj` is (n_alpha, n_points, n_pc). Between consecutive alphas the
    median step over all trials is the motion the cue gives every endpoint;
    what a trial moves beyond it, per PC, is divided by that PC's span over
    the sweep (the y range of a `2d` panel). A trial that stays on its
    attractor scores near zero; one that hops to a neighboring attractor
    scores its jump as a fraction of the panel. Returns (n_points,)."""
    proj = np.asarray(proj, dtype=float)
    if proj.ndim != 3 or proj.shape[0] < 2:
        raise ValueError(f"expected (n_alpha >= 2, n_points, n_pc) sweep "
                         f"coordinates, got {proj.shape}")
    steps = np.diff(proj, axis=0)
    departure = steps - np.median(steps, axis=1, keepdims=True)
    span = proj.max(axis=(0, 1)) - proj.min(axis=(0, 1))
    span = np.where(span > 0, span, 1.0)   # a flat PC has no departure either
    return (np.abs(departure) / span).max(axis=(0, 2))


def _cue_interp_sweep_view(sweep, family):
    """The sweep as drawn: every trial, or, with SIBLING_ONE_TRIAL_PER_STIMULUS,
    one trial per stimulus direction (`_one_trial_per_stimulus`). For a
    `family` in _SIBLING_ATTRACTOR_SWITCH_FAMILIES that choice weighs each
    trial's attractor-switch score against its magnitude, and the view keeps
    the drawn trials' scores (`attractor_switch_scores`). The view is a
    shallow copy with the per-point arrays subset."""
    if not SIBLING_ONE_TRIAL_PER_STIMULUS:
        return sweep
    scores = (_cue_interp_attractor_switch_scores(sweep["proj"])
              if family in _SIBLING_ATTRACTOR_SWITCH_FAMILIES else None)
    indices, target = _one_trial_per_stimulus(
        sweep["stim_idx"], sweep["stimulus_magnitude"], scores)
    view = dict(sweep)
    view["proj"] = np.asarray(sweep["proj"])[:, indices]
    for key in ("stim_idx", "stimulus_magnitude", "group_labels"):
        if key in sweep:
            view[key] = np.asarray(sweep[key])[indices]
    for key in ("is_fixed", "rel_step_undamped"):
        if key in sweep:
            view[key] = np.asarray(sweep[key])[:, indices]
    view["one_trial_target_magnitude"] = target
    if scores is not None:
        view["attractor_switch_scores"] = scores[indices]
    return view


def _cue_interp_drawn_legibility(sweeps, family, pair):
    """How legible the `2d` panels of `sweeps` ({base rule: sweep}) are in
    the 1-based PC `pair`, over the trials the current view draws
    (`_cue_interp_sweep_view`), or None for a family whose sweeps are not
    scored for attractor switches (DMC).

    Returns {"largest_attractor_switch": the largest drawn
    `_cue_interp_attractor_switch_scores` value, each trial scored against
    its whole sweep; "direction_spread": per panel (base rule x PC) the
    median over alpha of the range of the drawn direction means over the
    panel's range, smallest over the panels}. A spread near 0 means the
    directions draw as one line, however well the separation ratio scores
    them."""
    if family not in _SIBLING_ATTRACTOR_SWITCH_FAMILIES:
        return None
    largest, spread = 0.0, np.inf
    for sweep in sweeps.values():
        view = _cue_interp_sweep_view(sweep, family)
        scores = view.get("attractor_switch_scores")
        if scores is None:   # every trial is drawn
            scores = _cue_interp_attractor_switch_scores(sweep["proj"])
        largest = max(largest, float(np.max(scores)))
        proj = np.asarray(view["proj"], dtype=float)
        stim = np.asarray(view["stim_idx"], dtype=int)
        for pc in pair:
            y = proj[:, :, pc - 1]
            means = np.stack([y[:, stim == d].mean(axis=1) for d in np.unique(stim)],
                             axis=1)
            span = float(y.max() - y.min())
            apart = means.max(axis=1) - means.min(axis=1)
            spread = min(spread, float(np.median(apart / span)) if span > 0 else 0.0)
    return {"largest_attractor_switch": largest, "direction_spread": spread}


def _cue_interp_sweeps(entry, family):
    """`entry["sweeps"]` with `_cue_interp_sweep_view` applied to each."""
    return {base_rule: _cue_interp_sweep_view(sweep, family)
            for base_rule, sweep in entry["sweeps"].items()}


def _cue_interp_view_note(sweeps):
    """Saved-message note when the sweeps show one trial per stimulus: the
    target magnitudes and, where the trials were scored for attractor
    switches, the largest drawn score and how many drawn trials switch."""
    targets = [sweep["one_trial_target_magnitude"] for sweep in sweeps.values()
               if "one_trial_target_magnitude" in sweep]
    if not targets:
        return ""
    note = ("; one trial per stimulus, magnitude ≈ "
            + ", ".join(f"{t:g}" for t in targets))
    scores = [np.asarray(sweep["attractor_switch_scores"], dtype=float)
              for sweep in sweeps.values() if "attractor_switch_scores" in sweep]
    if scores:
        drawn = np.concatenate(scores)
        n_switching = int(np.sum(drawn > _SIBLING_ATTRACTOR_SWITCH_FRACTION))
        note += (f"; largest attractor switch drawn {drawn.max():.2f}, "
                 f"{n_switching} of {drawn.size} drawn trials above "
                 f"{_SIBLING_ATTRACTOR_SWITCH_FRACTION:g}")
    return note


def _cue_interp_sweep_arrays(sweep, context):
    """(proj (n_alpha, n_points, n_pc), converged mask (n_alpha, n_points))."""
    proj = np.asarray(sweep["proj"], dtype=float)
    if proj.ndim != 3:
        raise ValueError(f"{context}: expected (n_alpha, n_points, n_pc) "
                         f"coordinates, got {proj.shape}")
    good = np.asarray(
        sweep.get("is_fixed", np.ones(proj.shape[:2], dtype=bool)), dtype=bool)
    if good.shape != proj.shape[:2]:
        raise ValueError(f"{context}: convergence mask {good.shape} does not "
                         f"match coordinates {proj.shape[:2]}")
    return proj, good


# Line colors of the sweep figures follow the clusters of each family's plain
# endpoint figure, while the markers keep the alpha ramp, so hue on the line
# and hue on the markers carry different things.
#   DMC: the plain figure separates two memory clusters that cut across task
#   and stimulus (`sibling_delay_analysis._dmc_category_labels`). Stimulus
#   category 1 is locations 0-3 (< pi), category 2 locations 4-7. DMCGo
#   responds when the second stimulus matches the first, DMCNoGo when it does
#   not, so a DMCGo trial with a category-1 stimulus and a DMCNoGo trial with
#   a category-2 stimulus both respond to a category-1 second stimulus. The
#   cluster is therefore the category the network must respond to: "respond
#   to category 1" = DMCGo 0-3 with DMCNoGo 4-7 (label 1), "respond to
#   category 2" = the complement (label 0). A line takes the cluster of its
#   base trial (alpha = 0), categorical c_vals colors (SCHEME family 4), so
#   the swap under the sibling's cue is visible.
#   DelayDM: the plain figure's clusters are the stimulus directions, so a
#   line takes its direction's rainbow color (family 1), as the endpoints do.
_CUE_INTERP_CLUSTER_COLORS = {1: c_vals[1], 0: c_vals[2]}   # blue, green
_CUE_INTERP_CLUSTER_NAMES = {1: "Respond to category 1",
                             0: "Respond to category 2"}
_SIBLING_N_STIM = 8      # the task generator's eight stimulus directions
# The eight DMC stimulus locations (`core.mpn_tasks.dmc_`), offset from the
# ring directions 2 pi k / 8 so none lies on the category boundary at 0 / pi;
# location k maps to direction index k, and location < pi is category 1.
_DMC_STIMULUS_LOCATIONS = np.array(
    [0.05, 0.30, 0.55, 0.80, 1.05, 1.30, 1.55, 1.80]) * np.pi


def _cue_interp_sweep_groups(family, rules, base_rule, sweep):
    """Per-point cluster labels of one sweep, or None for families without.

    Saved `group_labels` are used when the projection cache carries them;
    otherwise the DMC labels are derived from the base rule and stimulus
    exactly as the plain endpoint figure derives them."""
    if "group_labels" in sweep:
        return np.asarray(sweep["group_labels"], dtype=int)
    if family != "dmcgo":
        return None
    stim = np.asarray(sweep["stim_idx"], dtype=int)
    base_task = list(rules).index(base_rule)
    half = max(_SIBLING_N_STIM, int(stim.max()) + 1) // 2
    return ((base_task == 0) == (stim < half)).astype(int)


def _cue_interp_line_colors(groups, sweep):
    """One line color per sweep point: the memory cluster when the family has
    clusters (`groups`), else the stimulus direction's rainbow color."""
    if groups is not None:
        return [_CUE_INTERP_CLUSTER_COLORS[int(label)] for label in groups]
    stim = np.asarray(sweep["stim_idx"], dtype=int)
    n_stim = max(_SIBLING_N_STIM, int(stim.max()) + 1)
    return [stim_color(int(s), n_stim) for s in stim]


def _cue_interp_cluster_handles(groups_by_sweep):
    """Legend entries for the clusters present in a figure's sweeps."""
    present = sorted({int(label) for groups in groups_by_sweep.values()
                      if groups is not None for label in np.unique(groups)},
                     reverse=True)
    return [mpl.lines.Line2D([], [], color=_CUE_INTERP_CLUSTER_COLORS[label],
                             linewidth=1.4,
                             label=f"{_CUE_INTERP_CLUSTER_NAMES[label]} "
                                   r"($\alpha$ = 0)")
            for label in present]


def _cue_interp_legend_handles(family, rules, method, entry):
    """Legend entries of one family's sweep figures for `method`.

    The cluster lines when the family has memory clusters (DMC), and the
    filled / hollow convergence markers for the gradient method, or for an
    endpoint sweep with a point left above the residual criterion; the
    stimulus-direction lines of DelayDM are documented by the stimulus
    colorwheel instead. Empty when nothing needs explaining."""
    groups_by_sweep = {
        base_rule: _cue_interp_sweep_groups(family, rules, base_rule, sweep)
        for base_rule, sweep in entry["sweeps"].items()}
    handles = _cue_interp_cluster_handles(groups_by_sweep)
    unconverged = any(not np.all(sweep.get("is_fixed", True))
                      for sweep in entry["sweeps"].values())
    if method == "gradient" or unconverged:
        handles += [
            mpl.lines.Line2D([], [], marker="o", linestyle="None",
                             markerfacecolor="0.3", markeredgecolor="none",
                             markersize=4, label="Converged"),
            mpl.lines.Line2D([], [], marker="o", linestyle="None",
                             markerfacecolor="none", markeredgecolor="0.3",
                             markersize=4, label="Not converged"),
        ]
    return handles


def _save_cue_interp_legend(handles, out_path):
    """Save the sweep figures' legend as its own small figure."""
    if not SHOW_LEGEND or not handles:
        return False
    legend_fig = plt.figure(figsize=(2.0, 0.35 + 0.3 * len(handles)))
    legend_fig.legend(handles=handles, loc="center", frameon=False,
                      fontsize=8, handletextpad=0.5, handlelength=1.6)
    _save_fig(legend_fig, out_path)
    return True


def _draw_cue_interp_sweep(ax, alphas, y, good, line_colors, z=None):
    """Trace every sweep trial across alpha on `ax`.

    x is alpha; `y` (and `z` for a 3D axes) are (n_alpha, n_points)
    coordinates of the trials along alpha. Each line takes its entry of
    `line_colors` (cluster or stimulus direction, see
    `_cue_interp_line_colors`); each alpha's markers take the ramp color at
    that alpha, filled when converged and hollow otherwise.
    """
    t = _alpha_ramp_norm(alphas)
    for s in range(y.shape[1]):
        coords = (alphas, y[:, s]) if z is None else (alphas, y[:, s], z[:, s])
        ax.plot(*coords, "-", color=line_colors[s], linewidth=1.1, alpha=0.75,
                zorder=2)
    for ai in range(alphas.size):
        col = _alpha_ramp_color(t[ai])
        for mask, style in ((good[ai], dict(color=col, edgecolor="none")),
                            (~good[ai], dict(facecolor="none", edgecolor=col,
                                             linewidth=0.7))):
            if not np.any(mask):
                continue
            x_at = np.full(int(mask.sum()), alphas[ai])
            coords = ((x_at, y[ai, mask]) if z is None
                      else (x_at, y[ai, mask], z[ai, mask]))
            ax.scatter(*coords, marker="o", s=12, alpha=0.9, zorder=3, **style)


def _cue_interp_panel_ylim(values):
    """Axis limits hugging the lines drawn in one panel: their finite range
    padded by `_MULTITASK_CUE_INTERP_Y_PADDING` of that range below and by
    `_MULTITASK_CUE_INTERP_Y_PADDING_TOP` above."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-9)
    return (lo - _MULTITASK_CUE_INTERP_Y_PADDING * span,
            hi + _MULTITASK_CUE_INTERP_Y_PADDING_TOP * span)


def _cue_interp_column_title(base_rule, target_rule):
    """Two-line panel title: whose stimulus is held, whose cue it fades to."""
    return (f"{_TASK_DISPLAY.get(base_rule, base_rule)} stimulus\n"
            f"→ {_TASK_DISPLAY.get(target_rule, target_rule)} cue")


def _cue_interp_axis_limits(sweeps, index):
    """Shared limits of one PC over every base rule's sweep."""
    values = np.concatenate([
        np.asarray(sweep["proj"], dtype=float)[:, :, index].ravel()
        for sweep in sweeps.values()])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-9)
    pad = _MULTITASK_CUE_INTERP_LIMIT_PADDING * span
    return (lo - pad, hi + pad)


def _render_multitask_cue_interp_2d(entry, alphas, family, method, pair,
                                    out_path, plane_note, rules=None):
    """x = alpha, y = one Joint Delay PC; rows are the plane's two PCs and
    columns the base rules whose stimulus is held (two for DelayDM, one for
    DMC). Each panel's y limits hug its own lines (`_cue_interp_panel_ylim`),
    so columns are not on a shared scale; the 3D view below uses the same
    plane. Lines are colored by the plain endpoint figure's clusters: the
    base trial's memory cluster for DMC, the stimulus direction for DelayDM
    (`_cue_interp_line_colors`)."""
    _ensure_out_dir()
    rules = _MULTITASK_SIBLING_FAMILIES[family] if rules is None else rules
    sweeps = _cue_interp_sweeps(entry, family)
    n_cols = len(sweeps)
    first = next(iter(sweeps.values()))
    pc_x, pc_y, bx, by = _paper_pc_indices(
        pair, np.asarray(first["proj"])[0], f"{family}/cue-interp/{method}")
    groups_by_sweep = {
        base_rule: _cue_interp_sweep_groups(family, rules, base_rule, sweep)
        for base_rule, sweep in sweeps.items()}
    # No shared x axis: every panel keeps its own 0 / 1 tick labels, so a
    # panel read on its own still shows where the sweep starts and ends.
    # Square panels (box aspect 1), as in the other association figures; the
    # figure is tall enough for two squares plus titles and the alpha label.
    fig, axs = plt.subplots(
        2, n_cols,
        figsize=(_MULTITASK_CUE_INTERP_PANEL_IN * n_cols,
                 2 * _MULTITASK_CUE_INTERP_PANEL_IN + 0.9),
        squeeze=False, layout="constrained")
    xpad = 0.1 * max(alphas.max() - alphas.min(), 1e-9)
    for row, (pc, index) in enumerate(((pc_x, bx), (pc_y, by))):
        for col, (base_rule, sweep) in enumerate(sweeps.items()):
            ax = axs[row, col]
            proj, good = _cue_interp_sweep_arrays(
                sweep, f"{family}/cue-interp/{method}/{base_rule}")
            colors = _cue_interp_line_colors(groups_by_sweep[base_rule], sweep)
            drawn = proj[:, :, index]
            _draw_cue_interp_sweep(ax, alphas, drawn, good, line_colors=colors)
            ax.set_xlim(alphas.min() - xpad, alphas.max() + xpad)
            # y limits hug the lines of THIS panel, not shared across columns.
            ax.set_ylim(*_cue_interp_panel_ylim(drawn))
            ax.set_xticks([alphas.min(), alphas.max()])
            ax.set_xticklabels([f"{alphas.min():g}", f"{alphas.max():g}"])
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax.tick_params(axis="x", labelsize=7, labelbottom=True)
            ax.tick_params(axis="y", labelleft=False)
            ax.set_box_aspect(1)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"Joint Delay PC{pc}", fontsize=7)
            if row == 0:
                ax.set_title(_cue_interp_column_title(
                    base_rule, sweep["target_rule"]), fontsize=6.5)
            else:
                ax.set_xlabel(r"Rule cue $\alpha$", fontsize=8, labelpad=-2)
    # The legend is a separate figure (`_save_cue_interp_legend`), so the
    # panels hold nothing but the sweep.
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, hspace=0.15,
                                wspace=0.05)
    grouping = ("stimulus direction"
                if all(g is None for g in groups_by_sweep.values())
                else "alpha = 0 cluster")
    _save_fig(fig, out_path,
              extra=(f"  ({method}; Joint Delay PC{pc_x}-PC{pc_y}, {plane_note}; "
                     f"{alphas.size} alphas; bases {list(sweeps)}; lines colored "
                     f"by {grouping}{_cue_interp_view_note(sweeps)})"))


def _render_multitask_cue_interp_3d(entry, alphas, family, method, pair,
                                    out_path, plane_note, rules=None):
    """3D view of `_render_multitask_cue_interp_2d`: x = alpha, y/z = the two
    Joint Delay PCs of the same plane, one panel per base rule. Panel geometry
    and axis styling follow the two-task alpha 3D figure so the two read as a
    pair; lines carry the same group colors as the 2D view."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)
    _ensure_out_dir()
    rules = _MULTITASK_SIBLING_FAMILIES[family] if rules is None else rules
    sweeps = _cue_interp_sweeps(entry, family)
    n_cols = len(sweeps)
    first = next(iter(sweeps.values()))
    pc_x, pc_y, bx, by = _paper_pc_indices(
        pair, np.asarray(first["proj"])[0], f"{family}/cue-interp/{method}")
    ylim = _cue_interp_axis_limits(sweeps, bx)
    zlim = _cue_interp_axis_limits(sweeps, by)
    fig = plt.figure(figsize=(_MULTITASK_CUE_INTERP_PANEL_IN * n_cols, 1.9))
    for col, (base_rule, sweep) in enumerate(sweeps.items()):
        ax = fig.add_subplot(1, n_cols, col + 1, projection="3d")
        proj, good = _cue_interp_sweep_arrays(
            sweep, f"{family}/cue-interp/{method}/{base_rule}")
        _draw_cue_interp_sweep(
            ax, alphas, proj[:, :, bx], good, z=proj[:, :, by],
            line_colors=_cue_interp_line_colors(
                _cue_interp_sweep_groups(family, rules, base_rule, sweep),
                sweep))
        ax.set_xlim(alphas.min(), alphas.max())
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel(r"$\alpha$", fontsize=7, labelpad=-15)
        ax.set_ylabel(f"Delay PC{pc_x}", fontsize=7, labelpad=-17)
        ax.set_zlabel(f"Delay PC{pc_y}", fontsize=7, labelpad=-17)
        ax.set_xticks([alphas.min(), alphas.max()])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(axis="both", labelsize=7, pad=-2)
        ax.view_init(elev=18, azim=-60)
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_visible(False)
        ax.patch.set_alpha(0.0)
        ax.set_title(_cue_interp_column_title(base_rule, sweep["target_rule"]),
                     fontsize=6.5, y=0.97)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top=0.9, wspace=0.12)
    _save_fig(fig, out_path,
              extra=(f"  ({method}; Joint Delay PC{pc_x}-PC{pc_y}, {plane_note}; "
                     f"{alphas.size} alphas; bases {list(sweeps)}"
                     f"{_cue_interp_view_note(sweeps)})"))


def plot_multitask_sibling_cue_interp():
    """
    Figures: the rule cue interpolated between the two sibling rules of each
    family, for both sibling methods, in the family's joint delay basis:
      multitask_{delaydm,dmcgo}_cue_interp_{gradient,long_delay_endpoint}_{emodulation,hidden}_{2d,3d}.png
      multitask_{delaydm,dmcgo}_cue_interp_{gradient,long_delay_endpoint}_legend.png
    `2d` draws one line per trial against alpha for each PC of the plane and
    `3d` the trial lines in the plane against alpha, both colored by the
    trial's group (stimulus direction for DelayDM, memory cluster for DMC).
    The legend is its own figure per family and method, holding the DMC
    cluster lines and the gradient method's converged / hollow markers
    (absent when nothing needs explaining).
    The stimulus is held at a base rule's trials (both DelayDM rules in turn,
    since they use different modalities; DMCGo alone, since DMCGo and DMCNoGo
    share one stimulus structure) while the cue moves from the base rule
    (alpha=0) to the sibling (alpha=1). Gradient sweeps trace TRUE delay fixed
    points solved by optimization; endpoint sweeps trace each trial's long
    delay settled to the same residual criterion. The plane is chosen on the
    sweep itself (`_multitask_cue_interp_plane`); for DMC its first PC keeps
    the response clusters' order along the sweep and its second swaps them,
    so the `2d` rows read "what was seen" and "what to respond to". With
    SIBLING_ONE_TRIAL_PER_STIMULUS (on by default) every view shows one trial
    per stimulus direction, which for DelayDM weighs matched magnitudes
    against trials that stay on one attractor along the sweep. Each
    representation reads the run selected for its sweep figure
    (`_multitask_sibling_aname`) from
    two_in_multiples/{aname}/{family}_cue_interp_*pc_projections_{aname}.pkl,
    loaded once per run; a representation or method without that cache is
    skipped.
    """
    for family, rules in _MULTITASK_SIBLING_FAMILIES.items():
        for method in _MULTITASK_CUE_INTERP_METHODS:
            loaded = {}
            legend_handles = {}
            for plot_name, output_suffix in (
                    ("e_modulation", "emodulation"),
                    ("hidden", "hidden")):
                aname = _multitask_sibling_aname(family, plot_name, "sweep")
                run_dir = TWO_IN_MULTIPLES_DIR / aname
                path = _multitask_cue_interp_projection_path(
                    run_dir, family, aname, method)
                if aname not in loaded:
                    loaded[aname] = _load_pkl_or_skip(
                        path,
                        _multitask_sibling_hint(family, method, aname=aname)
                        + " (its rule-cue sweep writes this file).")
                    if (loaded[aname] is not None
                            and loaded[aname].get("kind") != "cue_interp"):
                        print(f"  Skipped {path.name}: not a rule-cue sweep "
                              "projection.")
                        loaded[aname] = None
                data = loaded[aname]
                if data is None:
                    continue
                alphas = np.asarray(data["alphas"], dtype=float)
                entry = data.get("representations", {}).get(plot_name)
                if not entry or not entry.get("sweeps"):
                    print(f"  Skipped {family}/{method}/{plot_name}: no sweep "
                          "in the projection cache.")
                    continue
                pair, plane_note = _multitask_cue_interp_plane(
                    family, plot_name, method, run_dir, aname, entry=entry)
                stem = (f"{_MULTITASK_SIBLING_OUTPUT_NAMES[family]}_cue_interp_"
                        f"{method}_{output_suffix}")
                _render_multitask_cue_interp_2d(
                    entry, alphas, family, method, pair,
                    _multitask_out(f"{stem}_2d.png"), plane_note)
                _render_multitask_cue_interp_3d(
                    entry, alphas, family, method, pair,
                    _multitask_out(f"{stem}_3d.png"), plane_note)
                for handle in _cue_interp_legend_handles(family, rules, method, entry):
                    legend_handles.setdefault(handle.get_label(), handle)
            # One legend per family and method, shared by both
            # representations (whose runs may differ, so it holds every entry
            # either needs) and all views; nothing to save when the sweep
            # needs no explanation (DelayDM endpoints).
            if legend_handles:
                _save_cue_interp_legend(
                    list(legend_handles.values()),
                    _multitask_out(
                        f"{_MULTITASK_SIBLING_OUTPUT_NAMES[family]}_cue_interp_"
                        f"{method}_legend.png"))


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
    representation, and render each representation through `render_fn`. This
    alpha-sweep view deliberately retains its delay basis; the gradient
    fixed-point figures now use a separate stimulus basis."""
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
    delay-period basis. This differs from the stimulus basis used by the
    gradient-fixed-point figures.
    Each of the 8 stimuli is one line traced across alpha, so the figure shows how
    each period's fixed points move as the task cue morphs from anti (alpha=0) to
    pro (alpha=1). Color is NOT stimulus: every line uses one sequential ramp, dark
    at alpha=0 → light at alpha=1 (`_ALPHA_CMAP`), so hue carries the sweep
    direction only. One figure per representation:
      twotask_interp_alpha_fixed_points_3d_{seed}_modulation.png  (+ emodulation, hidden)
    Reads interp_fixed_points_{aname}.pkl (which stores all three representations).
    """
    # Shared y-z basis: delayanti delay-period grad fixed points, one per
    # representation. The 2D alpha view above uses this same driver and basis;
    # the separate gradient-fixed-point figures use the stimulus-period basis.
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
    per task rule (twotask_fixed_point_stability_{seed}_{rule}.png). The four
    task epochs show their eigenvalue spectra; a shared heatmap below aligns the
    spectral radius across stimulus angle for every task epoch and seed probe.
    Reads twotasks/{aname}/fixed_points_grad_{aname}_{rule}.pkl.
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
        _render_twotask_fixed_point_stability(
            d, OUT_DIR / f"twotask_fixed_point_stability_{tag}_{rule}.png",
            rule=rule, n_trained=TWOTASK_N_STIM)


def plot_two_task_fixed_point_classification():
    """Organized two-task stability classification.

    MemoryPro and MemoryAnti are aligned as columns; conditions are horizontal
    rows grouped into task epochs, memory-seeded, trajectory-seeded, and
    naive-seeded probes. Segment labels report nonzero candidate counts. Strict
    fixed points are split by stability; candidates with normalized residual in
    (0.01, 0.05] are shown separately as approximate rather than being folded
    into the unconverged category.
    """
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

    # Backward-compatible in-memory upgrade for classification pickles written
    # before the approximate category was added. This lets paper_plot redraw an
    # existing fixed-point run without asking the expensive solver to run again.
    for rule in rules:
        for entry in by_rule[rule]["per_period"].values():
            entry["counts"] = _fixed_point_classification_counts(entry)

    reference_periods = _ordered_stability_periods(by_rule[rules[0]]["per_period"])
    include_approximate = any(
        by_rule[rule]["per_period"].get(period, {}).get("counts", {}).get(
            "approximate", 0) > 0
        for rule in rules for period in reference_periods
    )
    include_unconverged = any(
        by_rule[rule]["per_period"].get(period, {}).get("counts", {}).get(
            "unconverged", 0) > 0
        for rule in rules for period in reference_periods
    )
    plotted_classes = list(class_names)
    if include_approximate:
        plotted_classes.append("approximate")
    if include_unconverged:
        plotted_classes.append("unconverged")

    max_total = max(
        sum(by_rule[rule]["per_period"].get(period, {}).get("counts", {}).get(
            name, 0) for name in plotted_classes)
        for rule in rules for period in reference_periods
    )
    fig_height = max(3.8, 0.34 * len(reference_periods) + 1.35)
    fig, axs = plt.subplots(
        1, len(rules), figsize=(7.4, fig_height), sharex=True, sharey=True,
        squeeze=False)
    y = np.arange(len(reference_periods))

    for col, rule in enumerate(rules):
        ax = axs[0, col]
        per_period = by_rule[rule]["per_period"]
        left = np.zeros(len(reference_periods), dtype=float)
        for class_name in plotted_classes:
            counts = np.asarray([
                per_period.get(period, {}).get("counts", {}).get(class_name, 0)
                for period in reference_periods
            ], dtype=float)
            bars = ax.barh(
                y, counts, left=left, height=0.68,
                color=_FP_CLASS_COLORS.get(class_name, "0.5"),
                edgecolor="white", linewidth=0.5, label=class_name.capitalize())
            for bar, count in zip(bars, counts):
                if count < 3:
                    continue
                x_center = bar.get_x() + 0.5 * bar.get_width()
                text_color = ("#111827" if class_name in
                              {"marginal", "approximate"} else "white")
                ax.text(x_center, bar.get_y() + 0.5 * bar.get_height(),
                        f"{int(count)}", ha="center", va="center",
                        fontsize=6.5, color=text_color)
            left += counts

        ax.set_title(_TASK_DISPLAY.get(rule, rule), fontsize=10.5)
        ax.set_xlim(0, max_total)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
        ax.set_xlabel("Candidate count", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", color="0.9", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)

        previous_group = _stability_condition_parts(
            reference_periods[0], per_period.get(reference_periods[0], {}))[0]
        for row, period in enumerate(reference_periods[1:], start=1):
            group = _stability_condition_parts(
                period, per_period.get(period, {}))[0]
            if group != previous_group:
                ax.axhline(row - 0.5, color="0.78", lw=0.8)
            previous_group = group

    axs[0, 0].set_yticks(y)
    axs[0, 0].set_yticklabels([
        _stability_condition_label(
            period, by_rule[rules[0]]["per_period"].get(period, {}))
        for period in reference_periods
    ], fontsize=7.5)
    axs[0, 0].invert_yaxis()

    handles = [
        mpl.patches.Patch(color=_FP_CLASS_COLORS[name], label=name.capitalize())
        for name in plotted_classes
    ]
    if SHOW_LEGEND:
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.985))
    fig.subplots_adjust(left=0.25, right=0.98, top=0.88, bottom=0.11, wspace=0.08)
    _save_fig(fig, OUT_DIR
              / f"twotask_fixed_point_classification_{_twotask_seed_tag()}.png")


_TWOTASK_PCA_REP_ORDER = ("hidden", "modulation", "w_modulation")
_TWOTASK_PCA_REP_TITLES = {
    "hidden": "Hidden",
    "modulation": "Modulation",
    "w_modulation": "Eff. modulation",
}


def _twotask_pca_representations(data):
    """Available two-task PCA representations in the paper's fixed order.

    Old analysis pickles containing only hidden plus one modulation variant
    remain renderable; new pickles contain all three representations.
    """
    return [name for name in _TWOTASK_PCA_REP_ORDER if name in data]


def plot_two_task_d_combine():
    """
    Figure: cross-task / cross-period PCA explained-variance heatmaps for the
    two-task network — one panel each for hidden activity, raw modulation M, and
    effective modulation W⊙M.

    Reads the self-contained d_combine pickle written by two_task_analysis.py
    (twotasks/{TWOTASK_ANAME}/d_combine_{TWOTASK_ANAME}.pkl), which stores the
    already-permuted 8x8 FVE matrix, its tick labels, and the color range for
    each available representation. New analysis output contains "hidden",
    "modulation", and "w_modulation"; older two-panel pickles remain supported.

    The x/y tick labels ("{task} {period}") are highlighted with their trial
    period's color from the period-bar palette (_ONETASK_PERIOD_COLORS), the same
    colors as the input/output illustration figure's period strip.
    """
    _ensure_out_dir()
    pkl_path = TWOTASKS_DIR / TWOTASK_ANAME / f"d_combine_{TWOTASK_ANAME}.pkl"
    d_combine = _load_pkl_or_skip(pkl_path, "Run two_task_analysis.py first.")
    if d_combine is None:
        return

    names = _twotask_pca_representations(d_combine)
    if not names:
        print("  Skipped: no PCA representations in d_combine pickle.")
        return
    # Shared color range across panels so a single colorbar applies to both.
    vmin = min(d_combine[n].get("vmin", 0.0) for n in names)
    vmax = max(d_combine[n].get("vmax", 1.0) for n in names)

    fig, axs = plt.subplots(1, len(names), figsize=(2.8 * len(names), 2.6),
                            gridspec_kw={"wspace": 0.15})
    if len(names) == 1:
        axs = [axs]
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
        ax.set_title(_TWOTASK_PCA_REP_TITLES.get(name, name), fontsize=10)
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
    for the two-task network — the two-task analog of onetask_pc_cumvar. Rows are
    the available representations (hidden / raw modulation / effective
    modulation), and columns are tasks (Go / Anti). Each panel plots one curve
    per trial period, showing how many PCs each period's trajectory needs.

    Reads the self-contained pc_cumvar pickle written by two_task_analysis.py
    (twotasks/{TWOTASK_ANAME}/pc_cumvar_{TWOTASK_ANAME}.pkl). Skips gracefully if
    the pickle predates that field.
    """
    _ensure_out_dir()
    d = _load_twotask_glob_or_skip("pc_cumvar_*.pkl")
    if d is None:
        return

    names = _twotask_pca_representations(d)
    if not names:
        print("  Skipped: no series in pc_cumvar pickle.")
        return

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
                ax.set_ylabel(
                    f"{_TWOTASK_PCA_REP_TITLES.get(name, name)}\nVar expl.",
                    fontsize=9)
            # No legend here — the period colors match onetask_pc_cumvar's shared
            # palette (colored BY POSITION into _ONETASK_PERIOD_COLORS), so the
            # standalone onetask_pc_cumvar_legend applies to this figure too.

    fig.tight_layout()
    out_path = OUT_DIR / f"twotask_pc_cumvar_{_twotask_seed_tag()}.png"
    _save_fig(fig, out_path)


def _plot_m_pca_panels(data, out_name, legend_frameon=False, show_legend=True):
    """Redraw the m_pca trajectory figure (PCA 1-2 only) from the data dict
    stashed by two_task_analysis.py (cell 86, "normal" variant). Mirrors the
    original alpha / marker / color conventions."""
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
        _plot_m_pca_panels(m_pca[name],
                           out_name=f"twotask_m_pca_{name}_{_twotask_seed_tag()}.png",
                           legend_frameon=(name == "hidden"),
                           show_legend=(name == "hidden"))


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
    ax.set_yticks([0, 50, 100])
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
#   one_task         analyses of the dedicated one-task training run
#   multiple_tasks   multi-task clustering and network structure
#   lesion           multi-task lesion effects and cluster comparisons
#   state_space      context-state and trajectory geometry of the multi-task net
#   acc_plot         accuracy comparisons across training configurations
#   two_in_multiple  sibling fixed-point geometry probes of the multi-task net
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
        "modulation_abs_weighted": plot_clustered_modulation_abs_weighted,
        "heatmap_colorbar": plot_multitask_heatmap_colorbar,
        "overmembership_norm": plot_overmembership_norm,
        "overmembership_unnorm": plot_overmembership_unnorm,
        "overmembership_weighted": plot_overmembership_weighted,
        "overmembership_var_weighted": plot_overmembership_var_weighted,
        "overmembership_abs_weighted": plot_overmembership_abs_weighted,
        "overmembership_examples": plot_overmembership_examples,
        "input_weight_correlation": plot_input_weight_correlation,
        "row_cluster_organization": plot_row_cluster_organization,
        "row_cluster_same_task": plot_row_cluster_same_task,
        "row_cluster_same_period": plot_row_cluster_same_period,
    },
    "lesion": {
        "lesion_heatmap": plot_lesion_heatmap,
        "lesion_cluster_sizes": plot_lesion_cluster_sizes,
        "cluster_corr_vs_lesion": plot_cluster_corr_vs_lesion,
        "cluster_corr_vs_lesion_weighted": plot_cluster_corr_vs_lesion_weighted,
        "cluster_corr_vs_lesion_seeds": plot_cluster_corr_vs_lesion_seeds,
        "om_vs_lesion": plot_om_vs_lesion,
        "om_vs_lesion_seeds": plot_om_vs_lesion_seeds,
        "plasticity_share": plot_plasticity_share,
        "plasticity_share_cohort": plot_plasticity_share_cohort,
        "causal_vs_activity_tasks": plot_causal_vs_activity,
        "causal_vs_activity_seeds": plot_causal_vs_activity_seeds,
        "task_specificity": plot_task_specificity,
        "task_sharing_cohort": plot_task_sharing_cohort,
        # Links the state-space task geometry to the lesion dependence
        # (multiple_task/state_space_vs_lesion.py).
        "state_space_vs_lesion": plot_state_space_vs_lesion,
        "state_space_vs_lesion_seeds": plot_state_space_vs_lesion_seeds,
    },
    "state_space": {
        "state_space_combined": plot_state_space_combined,
        "state_space_untrained": plot_state_space_untrained,
        "state_space_dist_angle": plot_state_space_dist_angle,
        "state_space_r_values": plot_state_space_r_values,
        # Category structure and task-vector additivity of the context-end
        # task centroids, from state_space_structure.py.
        "state_space_rdm": plot_state_space_rdm,
        "state_space_rdm_example": plot_state_space_rdm_example,
        "state_space_category_test": plot_state_space_category_test,
        "state_space_category_vs_untrained": plot_state_space_category_vs_untrained,
    },
    "acc_plot": {
        "l2_accuracy": plot_l2_vs_accuracy,
        "modulation_bound_accuracy": plot_modulation_bound_accuracy,
        "l2e4_activation_accuracy": plot_l2e4_activation_accuracy,
        "l2e4_activation_embedding_offset":
            plot_l2e4_activation_embedding_offset,
        "l2e4_activation_modulation_dimension":
            plot_l2e4_activation_modulation_dimension,
        "l2e4_activation_modulation_common_accuracy":
            plot_l2e4_activation_modulation_common_accuracy,
        "projection_dim_accuracy": plot_projection_dim_accuracy,
        "hidden_dim_accuracy": plot_hidden_dim_accuracy,
        "projection_dim_task_accuracy": plot_projection_dim_task_accuracy,
        "hidden_dim_task_accuracy": plot_hidden_dim_task_accuracy,
    },
    "two_in_multiple": {
        # DelayDM and DMCGo sibling-task delay geometry inside the multi-task
        # network, from sibling_delay_analysis.py: the gradient fixed-point
        # translation figures when solved, and the long-delay endpoint figures
        # (plain, and with the rule-cue sweep overlaid when cached).
        "delaydm_fixed_point_geometry":
            plot_multitask_delaydm_fixed_point_geometry,
        # The rule cue interpolated between the sibling rules, traced in the
        # same joint delay basis for both sibling methods.
        "sibling_cue_interp": plot_multitask_sibling_cue_interp,
    },
    "pretraining": {
        "backbone_probe": plot_backbone_probe,
        "principal_angles": plot_pretraining_principal_angles,
        "transfer_speed": plot_transfer_speed,
        "learning_trajectory": plot_learning_trajectory,
        "learning_trajectory_linear": plot_learning_trajectory_linear,
        "rule_vectors": plot_rule_vectors,
        "aggregate_cve_stimulus": plot_aggregate_cve_stimulus,
        "aggregate_cve_response": plot_aggregate_cve_response,
    },
    "two_task": {
        "twotask_d_combine": plot_two_task_d_combine,
        "twotask_pc_cumvar": plot_two_task_pc_cumvar,
        "twotask_m_pca": plot_two_task_m_pca,
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
    parser.add_argument(
        "--multitask-l2",
        choices=MULTITASK_L2_CHOICES,
        default=None,
        help="L2 cohort read by the multiple_tasks and lesion figures "
             f"(default: MULTITASK_L2 = {MULTITASK_L2}). Single-run figures use "
             "the cohort's designated seed when cached, else the best-covered "
             "cached seed; aggregate figures summarize every cached seed.",
    )
    parser.add_argument(
        "--pretraining-bound",
        type=_parse_pretraining_bound,
        choices=("mb1", "mb2"),
        default="mb1",
        metavar="{mod1,mod2}",
        help="Pretraining modulation bound: mod1/mb1 selects M in [-1,1] "
             "(default); mod2/mb2 selects M in [-2,2].",
    )
    parser.add_argument(
        "--pretraining-lambda",
        choices=tuple(_PRETRAINING_LAMBDA_TAGS),
        default=PRETRAINING_LAMBDA,
        help="Pretraining M retention per 40 ms step: 0.99 selects the runs "
             "trained with m_time_scale 4000 ms (default; untagged file names), "
             "0.9 the lam0.9 runs trained with 400 ms. Combines with "
             "--pretraining-bound.",
    )
    parser.add_argument(
        "--pretraining-groups",
        choices=tuple(_PRETRAINING_GROUPS),
        default="all",
        help="Pretraining conditions: motifs shows Relevant/Irrelevant motifs "
             "only; all also includes DelayAnti and DelayPro (default). "
             "Applies to pretraining mode and individual --only figures.",
    )
    parser.add_argument(
        "--sibling-one-trial-per-stimulus",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In the rule-cue sweep figures (sibling_cue_interp and the "
             "_cue_interp endpoint overlays) draw one trial per stimulus "
             "direction instead of every sweep trial, at a magnitude matched "
             "across the directions and, for DelayDM, weighed against how far "
             "the trial jumps to another attractor along alpha (switch scores "
             "above 0.1 only; default: on, "
             "SIBLING_ONE_TRIAL_PER_STIMULUS). A selection for legibility, to "
             "be stated in the caption. The --no- form draws every trial. "
             "Output names are unchanged.",
    )
    parser.add_argument(
        "--sibling-dmc-keep-every-alpha",
        action="store_true",
        help="In the DMC rule-cue sweep figures choose the top-row PC that "
             "keeps the response clusters apart in the same order at every "
             "alpha, instead of at the two ends of the sweep only. Changes "
             "the DMC sweep figures' plane and run; output names are "
             "unchanged.",
    )
    args = parser.parse_args()

    bad_modes = [m for m in args.mode if m not in valid_modes]
    if bad_modes:
        parser.error(f"invalid mode(s): {', '.join(bad_modes)}. "
                     f"Choose from: {', '.join(valid_modes)}")

    # Apply before resolving/running figures because all pretraining loaders
    # build exact artifact patterns from this selected addon.
    _set_pretraining_bound(args.pretraining_bound)
    _set_pretraining_lambda(args.pretraining_lambda)
    _set_pretraining_groups(args.pretraining_groups)
    # Re-resolve the multi-task run names only when the cohort is overridden;
    # the import-time selection already reflects MULTITASK_L2.
    if args.multitask_l2 is not None and args.multitask_l2 != MULTITASK_L2:
        _set_multitask_l2(args.multitask_l2)

    # Apply the legend toggle globally; every figure routes through _legend(),
    # which reads this module-level flag.
    global SHOW_LEGEND, SIBLING_ONE_TRIAL_PER_STIMULUS, SIBLING_DMC_KEEP_AT_EVERY_ALPHA
    if args.no_legend:
        SHOW_LEGEND = False
    if args.sibling_one_trial_per_stimulus is not None:
        SIBLING_ONE_TRIAL_PER_STIMULUS = args.sibling_one_trial_per_stimulus
    if args.sibling_dmc_keep_every_alpha:
        SIBLING_DMC_KEEP_AT_EVERY_ALPHA = True

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
        "multiple_tasks": f"{ANAME} (L2={MULTITASK_L2} cohort; aggregates over its cached seeds)",
        "lesion": f"{LESION_ANAME} (L2={MULTITASK_L2} cohort; *_seeds figures over its cached seeds)",
        "state_space": (
            f"(auto-selects best high-dimensional task-center separation within L2={STATE_SPACE_EXAMPLE_L2:.0e}; "
            f"the PCA figures pair it with the L2={STATE_SPACE_STRONG_L2:.0e} cohort's selection; "
            "R-values and the category test split across L2 1e-5/1e-4/1e-3/1e-2 cohorts; "
            "distance-angle p is the saved task-label permutation p)"
        ),
        "acc_plot": "(aggregated across seeds)",
        # The automatic sibling selection scans every cached run, so it is
        # left to the figures that need it rather than run for this summary.
        "two_in_multiple": "; ".join(
            f"{family}: "
            + (_pinned_sibling_aname(family)
               or f"best-separating cached {multitask_feature(SIBLING_L2)} run per "
                  "figure (ranked when its figures run)")
            for family in _MULTITASK_SIBLING_FAMILIES
        ),
        "two_task": TWOTASK_ANAME,
        "pretraining": (
            f"(aggregated across seeds; groups={PRETRAINING_GROUPS}; {PRETRAINING_BOUND}, "
            f"lambda={PRETRAINING_LAMBDA} (m_time_scale "
            f"{_PRETRAINING_LAMBDA_TIME_SCALES_MS[PRETRAINING_LAMBDA]} ms); "
            f"feature={PRETRAINING_ADDON_NAME}; "
            f"M in [{_PRETRAINING_BOUND_LIMITS[PRETRAINING_BOUND][0]}, "
            f"{_PRETRAINING_BOUND_LIMITS[PRETRAINING_BOUND][1]}])"
        ),
    }
    if modes_run in ("all", "only"):
        printed_modes = list(FIGURES_BY_MODE.keys())
    else:
        printed_modes = modes
    print("Experiment(s):")
    if {"multiple_tasks", "lesion"} & set(printed_modes):
        print(f"  multi-task cohort: {multitask_feature()} (shared by multiple_tasks and lesion)")
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
