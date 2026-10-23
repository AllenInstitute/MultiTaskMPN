"""
Cross-experiment diagnostics on trained multi-task MPNs.

Like ``multiple_task_analysis.py``, this takes a ``--feature`` (and optional
``--seed``), discovers all matching experiments, and inspects them. But rather
than re-running the heavy analysis, it loads only the *cluster* information the
analysis already wrote — the same row/column cluster assignments that
``paper_plot.py`` uses for its ``clustered_input/hidden/modulation`` figures —
so the diagnostics are fast and memory-light.

Each diagnostic is a small function registered in ``DIAGNOSES`` via
``@_register``; new checks can be added without touching the driver. Every
registered diagnostic is run over the discovered experiments, its summary is
printed, and all results are saved to
``multiple_tasks_diagnosis/diagnosis_{tag}.{pkl,json}``, with the tag
``{feature}[_seed{seed}]``. A run replaces the earlier results of its own tag
only, so the cohorts' results sit side by side (``paper_plot.py`` reads
``diagnosis_{feature}.pkl`` for its ``sibling_row_cluster_sharing`` figure).
All requested same-cluster diagnostics are drawn in one figure
(``same_cluster_diagnoses_{tag}.png``), one task pair per row and one trial
phase per column.

Cluster data read per experiment (all under ``multiple_tasks_analysis/{aname}/``):
  - input / hidden  : ``cluster_info_{aname}.pkl`` →
        ``{input,hidden}_normalized → result → row_tol_labels`` (+ tb_break_name)
  - modulation      : ``cluster_info_mod_{aname}.pkl`` →
        ``modulation_all_normalized → result_all_lst[MOD_G_INDEX] → row_tol_labels``
        (MOD_G_INDEX=1, the G=300 pre-grouping variant plotted by paper_plot).

Row labels are indexed by ``tb_break_name`` entries of the form
``"{rule}-{phase}"`` (e.g. ``"dmcgo-delay1"``), where the phase is one of
stim1 / stim2 / delay1 / delay2 / go1.

Current diagnostics, each a pair of sibling tasks: for every trial phase both
tasks have, the fraction of experiments that place the two tasks' rows in the
SAME row cluster, reported separately for the input, hidden and modulation
representations (`diagnose_pair_same_cluster`):
  category_same_cluster
      ReactCategoryPro (dmcgo) vs ReactCategoryAnti (dmcnogo).
  context_modality_same_cluster
      ContextIntModality1 (contextdelaydm1) vs ContextIntModality2
      (contextdelaydm2).
  integration_modality_same_cluster
      IntegrationModality1 (delaydm1) vs IntegrationModality2 (delaydm2).
The representations have different numbers of row clusters (on the L21e4
runs about 9 input, 6 hidden and 13 modulation clusters for 49 rows), and
rows of one phase can cluster together whatever the task, so the percentages
are only read against the chance level of that representation and phase:
the fraction of pairs of rows of the phase from two different tasks that
share a cluster, per experiment (`_same_phase_chance`). On the L21e4 runs it
ranges from about 7% (modulation, Memory 1) to 100% (input and hidden in
both memory periods, where every task's row falls in one cluster). The number
of experiments in which a pair shares a cluster is tested against those
chance levels with an exact two-sided Poisson-binomial test
(`_poisson_binomial_p`).

Usage:
    python multiple_task/multiple_task_diagnosis.py --feature L21e4
    python multiple_task/multiple_task_diagnosis.py --feature L21e3 --seed 299
    python multiple_task/multiple_task_diagnosis.py            # defaults to L21e4
"""
import re
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are saved, never shown
import matplotlib.pyplot as plt

# Log every saved figure path (like multiple_task_analysis.py / paper_plot.py).
# Wrapping Figure.savefig once means any figure-producing diagnostic prints its
# destination automatically.
from matplotlib.figure import Figure as _Figure

if not getattr(_Figure.savefig, "_logs_path", False):
    _orig_savefig = _Figure.savefig

    def _savefig_logged(self, fname, *args, **kwargs):
        result = _orig_savefig(self, fname, *args, **kwargs)
        try:
            print(f"  Saved figure: {fname}")
        except Exception:
            pass
        return result

    _savefig_logged._logs_path = True
    _Figure.savefig = _savefig_logged

# Data lives at repo root (scripts are run from there); see README "Quick start".
MULTI_DIR = Path("multiple_tasks_analysis")
OUT_DIR = Path("multiple_tasks_diagnosis")

# Modulation clustering variant to read. The mod pickle stores several
# pre-grouping results in result_all_lst; index 1 is the G=300 KMeans variant
# that paper_plot.plot_clustered_modulation(G_index=1) displays.
MOD_G_INDEX = 1

# Raw rule name -> Driscoll et al. 2024 display name (matches paper_plot._TASK_DISPLAY).
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

# Trial-phase suffix -> display name (matches paper_plot._PHASE_DISPLAY).
_PHASE_DISPLAY = {
    "stim1": "Stimulus 1",
    "stim2": "Stimulus 2",
    "delay1": "Memory 1",
    "delay2": "Memory 2",
    "go1": "Response",
}

# Per-representation bar colors, those of paper_plot's task-specificity and
# sibling_row_cluster_sharing figures (_TASK_SPECIFICITY_COLORS), so the
# diagnostic and the paper figure of one result agree.
_REP_COLORS = {
    "input": "#3182ce",       # blue
    "hidden": "#e53e3e",      # red
    "modulation": "#d53f8c",  # pink
}


# Saved-result layout: 2 adds each representation's chance level and the
# test against it (paper_plot reads only this version).
RESULT_VERSION = 2
# Prefixes of the files one run writes, each followed by "{tag}.{suffix}".
_OUTPUT_PREFIXES = ("diagnosis_", "same_cluster_diagnoses_")


def _clear_previous_results(tag):
    """Remove the earlier results of `tag` (diagnosis_{tag}.pkl/json and
    same_cluster_diagnoses_{tag}.png); other tags' files are kept. Returns the
    removed file names."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    removed = []
    for prefix in _OUTPUT_PREFIXES:
        for path in sorted(OUT_DIR.glob(f"{prefix}{tag}.*")):
            path.unlink()
            removed.append(path.name)
    return removed


# ─── Experiment discovery & loading ──────────────────────────────────────────

def _discover_anames(feature=None, seed=None):
    """Experiment identifiers under multiple_tasks_analysis/ that have cluster info.

    Filters by regularization `feature` (e.g. 'L21e4') and/or `seed` when given,
    mirroring multiple_task_analysis.py's CLI. Sorted for deterministic output.
    """
    anames = []
    # The mod pickle is 'cluster_info_mod_...'; this glob only matches the
    # neuron-cluster pickle 'cluster_info_everything_...'.
    for p in sorted(MULTI_DIR.glob("everything_seed*/cluster_info_everything_seed*.pkl")):
        m = re.match(
            r"cluster_info_(everything_seed(\d+)_(\w+)\+hidden\d+\+batch\d+\+angle)\.pkl",
            p.name,
        )
        if not m:
            continue
        aname, s, f = m.group(1), int(m.group(2)), m.group(3)
        if feature is not None and f != feature:
            continue
        if seed is not None and s != seed:
            continue
        anames.append(aname)
    return anames


def _load_experiment(aname):
    """Row-cluster labels + tb_break_name per representation for one experiment.

    Returns {rep: {"row_labels": np.ndarray, "tb_break_name": np.ndarray}} for
    rep in {input, hidden, modulation}, or None if the neuron-cluster pickle is
    missing (input/hidden are required; modulation is included when available).
    """
    run_dir = MULTI_DIR / aname
    ci_path = run_dir / f"cluster_info_{aname}.pkl"
    mod_path = run_dir / f"cluster_info_mod_{aname}.pkl"

    if not ci_path.exists():
        print(f"  [skip] {aname}: missing {ci_path.name}")
        return None

    with open(ci_path, "rb") as f:
        ci = pickle.load(f)

    reps = {}
    for rep, key in (("input", "input_normalized"), ("hidden", "hidden_normalized")):
        entry = ci.get(key)
        if entry is None:
            print(f"  [skip] {aname}: cluster_info missing '{key}'")
            return None
        reps[rep] = {
            "row_labels": np.asarray(entry["result"]["row_tol_labels"]),
            "tb_break_name": np.asarray(entry["tb_break_name"]),
        }

    # Modulation (optional): different pickle, list of pre-grouping variants.
    if mod_path.exists():
        with open(mod_path, "rb") as f:
            cim = pickle.load(f)
        mentry = cim.get("modulation_all_normalized")
        result_all = (mentry or {}).get("result_all_lst", [])
        if mentry is not None and len(result_all) > MOD_G_INDEX:
            reps["modulation"] = {
                "row_labels": np.asarray(result_all[MOD_G_INDEX]["row_tol_labels"]),
                "tb_break_name": np.asarray(mentry["tb_break_name"]),
            }
        else:
            print(f"  [warn] {aname}: modulation cluster info incomplete; "
                  f"modulation representation skipped")
    else:
        print(f"  [warn] {aname}: missing {mod_path.name}; "
              f"modulation representation skipped")

    return reps


def _row_label(rep_data, session_name):
    """Row-cluster label for a '{rule}-{phase}' session, or None if absent."""
    tb = rep_data["tb_break_name"]
    idx = np.flatnonzero(tb == session_name)
    if idx.size == 0:
        return None
    return int(rep_data["row_labels"][idx[0]])


def _same_phase_chance(rep_data, phase):
    """Chance that two tasks' rows of `phase` share a row cluster in one
    experiment's representation: the fraction of pairs of rows of `phase` from
    two different tasks that carry the same label (NaN with fewer than two
    such rows). It is the baseline a sibling pair's sharing is read against,
    per phase, since rows of one phase can cluster together regardless of the
    task."""
    names = [str(name).rsplit("-", 1) for name in rep_data["tb_break_name"]]
    in_phase = np.array([len(parts) == 2 and parts[1] == phase for parts in names])
    rules = np.array([parts[0] for parts in names])[in_phase]
    labels = np.asarray(rep_data["row_labels"])[in_phase]
    pairs = np.triu(rules[:, None] != rules[None, :], k=1)
    if not pairs.any():
        return float("nan")
    return float((labels[:, None] == labels[None, :])[pairs].mean())


def _poisson_binomial_p(n_same, chances):
    """Exact two-sided p of `n_same` experiments sharing a cluster when each
    shares independently with its own probability in `chances` (a
    Poisson-binomial count): the summed probability of every count no more
    likely than the observed one. NaN when a chance is undefined."""
    chances = np.asarray(chances, dtype=float)
    if chances.size == 0 or not np.all(np.isfinite(chances)):
        return float("nan")
    if not 0 <= n_same <= chances.size:
        raise ValueError(f"{n_same} sharing experiments out of {chances.size}")
    pmf = np.array([1.0])
    for chance in chances:
        pmf = np.convolve(pmf, [1.0 - chance, chance])
    return float(min(1.0, pmf[pmf <= pmf[n_same] * (1.0 + 1e-7)].sum()))


# ─── Diagnostic registry ─────────────────────────────────────────────────────

DIAGNOSES = {}


def _register(name):
    """Register a diagnostic under `name` so the driver runs it automatically."""
    def deco(fn):
        DIAGNOSES[name] = fn
        return fn
    return deco


def _same_cluster_fraction(experiments, session_a, session_b, reps=("input", "hidden", "modulation")):
    """Per representation, the fraction of experiments whose `session_a` and
    `session_b` rows (two tasks of one '{rule}-{phase}' phase) share the same
    row cluster, against chance.

    Returns {rep: {pct_same_cluster, n_same, n_total, chance_mean,
    expected_same, p_vs_chance, detail: [...]}}: each experiment's chance
    level is `_same_phase_chance` of that phase, `expected_same` their sum and
    `p_vs_chance` the two-sided `_poisson_binomial_p` of n_same. Experiments
    missing a representation or either session row are skipped for that
    representation (and counted only in the representations where present).
    """
    phase = session_a.rsplit("-", 1)[-1]
    per_rep = {}
    for rep in reps:
        n_same, n_total, detail = 0, 0, []
        for aname, data in experiments:
            if rep not in data:
                continue
            la = _row_label(data[rep], session_a)
            lb = _row_label(data[rep], session_b)
            if la is None or lb is None:
                continue
            same = (la == lb)
            n_total += 1
            n_same += int(same)
            detail.append({"aname": aname, "label_a": la, "label_b": lb, "same": same,
                           "chance": _same_phase_chance(data[rep], phase)})
        pct = (100.0 * n_same / n_total) if n_total else float("nan")
        chances = [entry["chance"] for entry in detail]
        per_rep[rep] = {
            "pct_same_cluster": pct,
            "n_same": n_same,
            "n_total": n_total,
            "chance_mean": float(np.mean(chances)) if chances else float("nan"),
            "expected_same": float(np.sum(chances)) if chances else float("nan"),
            "p_vs_chance": _poisson_binomial_p(n_same, chances),
            "detail": detail,
        }
    return per_rep


def _phases_present(experiments, rule_a, rule_b):
    """Trial phases for which BOTH rules have a clustered row in at least one
    experiment, returned in canonical _PHASE_DISPLAY order."""
    present = set()
    for _, data in experiments:
        for rep_data in data.values():
            tb = set(rep_data["tb_break_name"].tolist())
            for phase in _PHASE_DISPLAY:
                if f"{rule_a}-{phase}" in tb and f"{rule_b}-{phase}" in tb:
                    present.add(phase)
    return [ph for ph in _PHASE_DISPLAY if ph in present]


def diagnose_pair_same_cluster(experiments, rule_a, rule_b):
    """Do two tasks share a row cluster, per trial phase? (pair-agnostic core)

    `rule_a` and `rule_b` are raw rule names (e.g. 'dmcgo', 'contextdelaydm1').
    Each contributes one clustered row per trial phase present in the analysis
    (stim1 / stim2 / delay1 / delay2 / go1). For every phase both tasks share,
    this reports — per representation (input, hidden, modulation) — the fraction
    of experiments in which the two rows land in the SAME row cluster, with its
    chance level and the test against it (`_same_cluster_fraction`).

    Returns a result dict consumable by _print_result and
    plot_combined_same_cluster.
    """
    a_disp = _TASK_DISPLAY.get(rule_a, rule_a)
    b_disp = _TASK_DISPLAY.get(rule_b, rule_b)
    phases = _phases_present(experiments, rule_a, rule_b)
    per_phase = {}
    for phase in phases:
        per_phase[phase] = _same_cluster_fraction(
            experiments, f"{rule_a}-{phase}", f"{rule_b}-{phase}")
    return {
        "description": (
            f"% of experiments where {a_disp} ({rule_a}) and {b_disp} ({rule_b}) "
            f"rows fall in the same row cluster, per trial phase and representation."
        ),
        "rule_a": rule_a,
        "rule_a_display": a_disp,
        "rule_b": rule_b,
        "rule_b_display": b_disp,
        "phases": phases,
        "per_phase": per_phase,
    }


@_register("category_same_cluster")
def diagnose_category_same_cluster(experiments):
    """ReactCategoryPro (dmcgo) vs ReactCategoryAnti (dmcnogo): same row cluster
    per trial phase, per representation."""
    return diagnose_pair_same_cluster(experiments, "dmcgo", "dmcnogo")


@_register("context_modality_same_cluster")
def diagnose_context_modality_same_cluster(experiments):
    """ContextIntModality1 (contextdelaydm1) vs ContextIntModality2
    (contextdelaydm2): same row cluster per trial phase, per representation."""
    return diagnose_pair_same_cluster(experiments, "contextdelaydm1", "contextdelaydm2")


@_register("integration_modality_same_cluster")
def diagnose_integration_modality_same_cluster(experiments):
    """IntegrationModality1 (delaydm1) vs IntegrationModality2 (delaydm2):
    same row cluster per trial phase, per representation."""
    return diagnose_pair_same_cluster(experiments, "delaydm1", "delaydm2")


# ─── Figure ──────────────────────────────────────────────────────────────────

# Panel size (inches) of the combined figure, and the y-axis top: the head
# room above 100% keeps each bar's n_same/n_total label clear of the panel
# above it and of the column header.
_PANEL_W_IN, _PANEL_H_IN = 1.2, 1.25
_Y_TOP = 118


# The chance-level dash drawn across each bar (`_same_phase_chance`).
_CHANCE_COLOR = "0.25"


def _draw_same_cluster_bars(ax, per_rep):
    """One trial-phase panel: a bar per representation with the same-cluster
    %, labelled n_same/n_total ("n/a" when no experiment has both rows), and a
    dash at the representation's mean chance level. The representations are
    told apart by color (the figure legend), so the panel has no x tick
    labels."""
    reps = list(per_rep)
    xs = np.arange(len(reps))
    pcts = np.array([per_rep[r]["pct_same_cluster"] for r in reps], dtype=float)
    heights = np.nan_to_num(pcts)
    ax.bar(xs, heights, width=0.75, color=[_REP_COLORS.get(r, "#718096") for r in reps],
           edgecolor="k", linewidth=0.5, alpha=0.85)
    for x, rep, pct, height in zip(xs, reps, pcts, heights):
        stats = per_rep[rep]
        label = (f"{stats['n_same']}/{stats['n_total']}" if np.isfinite(pct)
                 else "n/a")
        chance = 100 * stats.get("chance_mean", float("nan"))
        if np.isfinite(chance):
            ax.hlines(chance, x - 0.375, x + 0.375, color=_CHANCE_COLOR,
                      linewidth=1.4, zorder=3)
        # Above the bar or its chance dash, whichever is higher.
        ax.text(x, np.nanmax([height, chance]) + 3, label, ha="center", va="bottom",
                fontsize=6)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(reps) - 0.4)
    ax.set_ylim(0, _Y_TOP)
    ax.set_yticks([0, 50, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.spines[["top", "right"]].set_visible(False)


def plot_combined_same_cluster(results, out_path, title="Task-pair row-cluster sharing"):
    """All task-pair diagnostics in one table-like figure.

    A row per task pair, named on its left, and a column per trial phase in
    canonical order, headed on the top row, so one phase sits in the same
    column on every row; a pair without that phase leaves its cell empty.
    The y axis is the % of experiments whose two rows share a row cluster,
    labelled on the left column only; the dash across each bar is that
    representation's chance level in that phase. One legend below the panels
    gives the representation colors and the chance dash.
    """
    plot_items = [result for result in results.values() if result.get("phases")]
    if not plot_items:
        print("  [plot] no same-cluster diagnostics to combine; skipping figure.")
        return
    phases = [phase for phase in _PHASE_DISPLAY
              if any(phase in result["phases"] for result in plot_items)]

    n_rows, n_cols = len(plot_items), len(phases)
    fig, axes = plt.subplots(
        n_rows, n_cols, sharey=True, squeeze=False, layout="constrained",
        figsize=(1.6 + _PANEL_W_IN * n_cols, 0.7 + _PANEL_H_IN * n_rows))
    for row, result in enumerate(plot_items):
        for col, phase in enumerate(phases):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(_PHASE_DISPLAY[phase], fontsize=9)
            if phase in result["per_phase"]:
                _draw_same_cluster_bars(ax, result["per_phase"][phase])
            else:
                ax.axis("off")
        a_disp = result.get("rule_a_display", result.get("rule_a", "A"))
        b_disp = result.get("rule_b_display", result.get("rule_b", "B"))
        axes[row, 0].set_ylabel(f"{a_disp}\nvs\n{b_disp}", rotation=0,
                                ha="right", va="center", fontsize=8, labelpad=6)
    for ax in axes.flat:
        ax.tick_params(axis="y", labelsize=7)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="k",
                             linewidth=0.5, alpha=0.85, label=rep)
               for rep, color in _REP_COLORS.items()]
    handles.append(plt.Line2D([], [], color=_CHANCE_COLOR, linewidth=1.4,
                              label="chance (two tasks' rows of the phase)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.2)
    fig.suptitle(f"{title}\n% of experiments whose two task rows share a row cluster",
                 fontsize=10)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─── Driver ──────────────────────────────────────────────────────────────────

def _print_per_rep(per_rep, indent="    "):
    """Print one line per representation for a {rep: stats} mapping."""
    for rep, stats in per_rep.items():
        pct = stats["pct_same_cluster"]
        pct_str = "  n/a" if not np.isfinite(pct) else f"{pct:5.1f}%"
        print(f"{indent}{rep:<11s}: {pct_str}  "
              f"({stats['n_same']}/{stats['n_total']} experiments; chance "
              f"{100 * stats['chance_mean']:.0f}%, expected "
              f"{stats['expected_same']:.1f}, p = {stats['p_vs_chance']:.2g})")


def _print_result(name, result):
    """Human-readable summary of a diagnostic result to stdout."""
    print(f"\n=== {name} ===")
    print(f"  {result.get('description', '')}")
    # One block per trial phase.
    for phase, per_rep in result.get("per_phase", {}).items():
        print(f"  [{_PHASE_DISPLAY.get(phase, phase)}]")
        _print_per_rep(per_rep, indent="      ")


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays and bools to JSON-native types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def main(feature="L21e4", seed=None, which=None):
    """Run the requested diagnostics over all matching experiments and save.

    which : optional list of diagnostic names to run (default: all registered).
    """
    tag = feature if feature else "all"
    if seed is not None:
        tag += f"_seed{seed}"
    removed = _clear_previous_results(tag)
    print(f"Cleared {len(removed)} previous {tag} result file(s) from {OUT_DIR}/")

    anames = _discover_anames(feature, seed)
    print(f"Discovered {len(anames)} experiment(s) "
          f"(feature={feature}, seed={seed}): {anames}")
    if not anames:
        print("Nothing to diagnose.")
        return {}

    experiments = []  # (aname, {rep: {...}})
    for aname in anames:
        data = _load_experiment(aname)
        if data is not None:
            experiments.append((aname, data))
    print(f"Loaded cluster info for {len(experiments)}/{len(anames)} experiment(s).")
    if not experiments:
        print("No experiments with usable cluster info; nothing to diagnose.")
        return {}

    to_run = which if which else list(DIAGNOSES)
    unknown = [n for n in to_run if n not in DIAGNOSES]
    if unknown:
        raise SystemExit(f"Unknown diagnostic(s): {unknown}. "
                         f"Available: {list(DIAGNOSES)}")

    results = {}
    for name in to_run:
        results[name] = DIAGNOSES[name](experiments)
        _print_result(name, results[name])

    # One combined figure: one requested task-pair diagnostic per row.
    fig_path = OUT_DIR / f"same_cluster_diagnoses_{tag}.png"
    plot_combined_same_cluster(
        results, fig_path,
        title=f"Task-pair row-cluster sharing ({tag}, {len(experiments)} experiments)")

    # Save results (pkl keeps full detail incl. numpy; json is a portable copy).
    payload = {
        "version": RESULT_VERSION,
        "feature": feature,
        "seed": seed,
        "mod_g_index": MOD_G_INDEX,
        "experiments": [a for a, _ in experiments],
        "results": results,
    }
    pkl_path = OUT_DIR / f"diagnosis_{tag}.pkl"
    json_path = OUT_DIR / f"diagnosis_{tag}.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    with open(json_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    print(f"\nSaved: {pkl_path}")
    print(f"Saved: {json_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-experiment MPN diagnostics.")
    parser.add_argument("--feature", type=str, default="L21e4",
                        help="Only diagnose models with this exact feature "
                             "(default: 'L21e4').")
    parser.add_argument("--seed", type=int, default=None,
                        help="Only diagnose the model with this seed (e.g. 299).")
    parser.add_argument("--diagnosis", type=str, nargs="*", default=None,
                        help=f"Diagnostic(s) to run (default: all). "
                             f"Available: {list(DIAGNOSES)}")
    args = parser.parse_args()
    main(feature=args.feature, seed=args.seed, which=args.diagnosis)
