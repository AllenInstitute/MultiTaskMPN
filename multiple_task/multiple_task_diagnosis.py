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
``multiple_tasks_diagnosis/diagnosis_{feature}[_seed{seed}].{pkl,json}``.

Cluster data read per experiment (all under ``multiple_tasks/{aname}/``):
  - input / hidden  : ``cluster_info_{aname}.pkl`` →
        ``{input,hidden}_normalized → result → row_tol_labels`` (+ tb_break_name)
  - modulation      : ``cluster_info_mod_{aname}.pkl`` →
        ``modulation_all_normalized → result_all_lst[MOD_G_INDEX] → row_tol_labels``
        (MOD_G_INDEX=1, the G=300 pre-grouping variant plotted by paper_plot).

Row labels are indexed by ``tb_break_name`` entries of the form
``"{rule}-{phase}"`` (e.g. ``"dmcgo-delay1"``), where the phase is one of
stim1 / stim2 / delay1 / delay2 / go1.

Current diagnostics:
  category_delay_same_cluster
      For the category tasks ReactCategoryPro (dmcgo) and ReactCategoryAnti
      (dmcnogo), the fraction of experiments that place their delay-period
      (delay1) rows in the SAME row cluster — reported separately for the
      input, hidden, and modulation representations.

Usage:
    python multiple_task/multiple_task_diagnosis.py --feature L21e4
    python multiple_task/multiple_task_diagnosis.py --feature L21e3 --seed 299
    python multiple_task/multiple_task_diagnosis.py            # every experiment
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

# Data lives at repo root (scripts are run from there); see README "Workflow".
MULTI_DIR = Path("multiple_tasks")
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

# Trial-phase suffix -> display name (matches paper_plot._PHASE_DISPLAY).
_PHASE_DISPLAY = {
    "stim1": "Stimulus 1",
    "stim2": "Stimulus 2",
    "delay1": "Memory 1",
    "delay2": "Memory 2",
    "go1": "Response",
}

# Per-representation bar colors (matches the categorical palette elsewhere).
_REP_COLORS = {
    "input": "#3182ce",       # blue
    "hidden": "#e53e3e",      # red
    "modulation": "#38a169",  # green
}


# ─── Experiment discovery & loading ──────────────────────────────────────────

def _discover_anames(feature=None, seed=None):
    """Experiment identifiers under multiple_tasks/ that have cluster info.

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
    `session_b` rows share the same row cluster.

    Returns {rep: {pct_same_cluster, n_same, n_total, detail: [...]}}.
    Experiments missing a representation or either session row are skipped for
    that representation (and counted only in the representations where present).
    """
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
            detail.append({"aname": aname, "label_a": la, "label_b": lb, "same": same})
        pct = (100.0 * n_same / n_total) if n_total else float("nan")
        per_rep[rep] = {
            "pct_same_cluster": pct,
            "n_same": n_same,
            "n_total": n_total,
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
    of experiments in which the two rows land in the SAME row cluster.

    Returns a result dict consumable by _print_result and plot_pair_same_cluster.
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


# ─── Figure registry ─────────────────────────────────────────────────────────

# Diagnostic name -> plotter(result, out_path). A diagnostic gets a figure by
# registering here; the driver calls the plotter after computing the result.
PLOTTERS = {}


def _register_plot(name):
    """Register a plotter for the diagnostic `name`."""
    def deco(fn):
        PLOTTERS[name] = fn
        return fn
    return deco


@_register_plot("category_same_cluster")
@_register_plot("context_modality_same_cluster")
@_register_plot("integration_modality_same_cluster")
def plot_pair_same_cluster(result, out_path):
    """One subfigure per trial phase; each a barplot of the same-cluster % over
    representations (input / hidden / modulation). Pair-agnostic: the task pair
    is taken from the result's rule_a_display / rule_b_display."""
    per_phase = result["per_phase"]
    phases = result.get("phases") or list(per_phase)
    if not phases:
        print("  [plot] no phases to plot; skipping figure.")
        return

    n = len(phases)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0), sharey=True, squeeze=False)
    axes = axes[0]

    a_disp = result.get("rule_a_display", result.get("rule_a", "A"))
    b_disp = result.get("rule_b_display", result.get("rule_b", "B"))

    for ax, phase in zip(axes, phases):
        per_rep = per_phase[phase]
        reps = list(per_rep)
        pcts = [per_rep[r]["pct_same_cluster"] for r in reps]
        colors = [_REP_COLORS.get(r, "#718096") for r in reps]
        xs = np.arange(len(reps))
        ax.bar(xs, pcts, color=colors, edgecolor="k", linewidth=0.5, alpha=0.85)
        # Annotate each bar with n_same/n_total.
        for x, r, pct in zip(xs, reps, pcts):
            st = per_rep[r]
            if np.isfinite(pct):
                ax.text(x, pct + 2, f"{st['n_same']}/{st['n_total']}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(reps, rotation=30, ha="right", fontsize=8)
        ax.set_title(_PHASE_DISPLAY.get(phase, phase), fontsize=10)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Same-cluster experiments (%)", fontsize=9)
    fig.suptitle(f"{a_disp} vs {b_disp}: same row cluster", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─── Driver ──────────────────────────────────────────────────────────────────

def _print_per_rep(per_rep, indent="    "):
    """Print one line per representation for a {rep: stats} mapping."""
    for rep, stats in per_rep.items():
        pct = stats["pct_same_cluster"]
        pct_str = "  n/a" if not np.isfinite(pct) else f"{pct:5.1f}%"
        print(f"{indent}{rep:<11s}: {pct_str}  "
              f"({stats['n_same']}/{stats['n_total']} experiments)")


def _print_result(name, result):
    """Human-readable summary of a diagnostic result to stdout."""
    print(f"\n=== {name} ===")
    print(f"  {result.get('description', '')}")
    # Per-phase form (session-resolved): one block per phase.
    per_phase = result.get("per_phase")
    if per_phase:
        for phase, per_rep in per_phase.items():
            print(f"  [{_PHASE_DISPLAY.get(phase, phase)}]")
            _print_per_rep(per_rep, indent="      ")
        return
    # Flat form: a single {rep: stats} mapping.
    per_rep = result.get("per_representation")
    if per_rep:
        _print_per_rep(per_rep)


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


def main(feature=None, seed=None, which=None):
    """Run the requested diagnostics over all matching experiments and save.

    which : optional list of diagnostic names to run (default: all registered).
    """
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

    OUT_DIR.mkdir(exist_ok=True)
    tag = feature if feature else "all"
    if seed is not None:
        tag += f"_seed{seed}"

    results = {}
    for name in to_run:
        results[name] = DIAGNOSES[name](experiments)
        _print_result(name, results[name])
        # Figure (if this diagnostic registered a plotter).
        if name in PLOTTERS:
            fig_path = OUT_DIR / f"{name}_{tag}.png"
            PLOTTERS[name](results[name], fig_path)

    # Save results (pkl keeps full detail incl. numpy; json is a portable copy).
    payload = {
        "feature": feature,
        "seed": seed,
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
    parser.add_argument("--feature", type=str, default=None,
                        help="Only diagnose models with this feature (e.g. 'L21e4').")
    parser.add_argument("--seed", type=int, default=None,
                        help="Only diagnose the model with this seed (e.g. 299).")
    parser.add_argument("--diagnosis", type=str, nargs="*", default=None,
                        help=f"Diagnostic(s) to run (default: all). "
                             f"Available: {list(DIAGNOSES)}")
    args = parser.parse_args()
    main(feature=args.feature, seed=args.seed, which=args.diagnosis)
