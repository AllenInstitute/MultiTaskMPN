"""Shared delay-memory geometry analyses for sibling tasks.

This module owns delay-trajectory PCA fitting and complete six-PC fixed-point
projections for the delayDM and DMC sibling families. The parent
multiple_task_analysis.py only orchestrates data generation and fixed-point
solving before calling this module.
"""
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
import torch

__all__ = [
    "DELAY_PCA_SCOPES",
    "fit_delay_trajectory_pca",
    "save_sibling_fixed_point_pc_projections",
]

DELAY_PCA_SCOPES = ("joint", "first_task_only")


def _delay_pca_scope_spec(rules, basis_scope):
    """Metadata shared by PCA fitting and fixed-point projection."""
    if basis_scope == "joint":
        return {
            "fit_rule_indices": tuple(range(len(rules))),
            "source": "concatenated normal-trial delay1 trajectories from both tasks",
            "artifact_suffix": "",
            "title": "joint",
        }
    if basis_scope == "first_task_only":
        reference_rule = rules[0]
        return {
            "fit_rule_indices": (0,),
            "source": (f"normal-trial delay1 trajectories from "
                       f"{reference_rule} only"),
            "artifact_suffix": f"_{reference_rule}_only",
            "title": f"{reference_rule}-only",
        }
    raise ValueError(f"unknown basis_scope {basis_scope!r}; choose from "
                     f"{DELAY_PCA_SCOPES}")


def _tracked_numpy(value):
    """Convert one tracked-state entry to a single NumPy array."""
    if isinstance(value, (list, tuple)):
        return np.concatenate([_tracked_numpy(v) for v in value], axis=-1)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pca_record(pca, n_samples, method):
    """Portable PCA parameters consumed by paper_plot.py (no sklearn pickle)."""
    return {
        "mean": np.asarray(pca.mean_, dtype=np.float32),
        "components": np.asarray(pca.components_, dtype=np.float32),
        "explained_variance": np.asarray(pca.explained_variance_, dtype=np.float32),
        "explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype=np.float32),
        "n_samples": int(n_samples),
        "method": str(method),
    }


def fit_delay_trajectory_pca(aname, save_dir, addtask, rules, norm_db,
                             norm_trials, norm_task, W,
                             layer_index=1, chunk_samples=128,
                             n_components=6, basis_scope="joint"):
    """Fit a Delay-PC basis on joint or first-task-only delay1 trajectories.

    ``basis_scope="joint"`` uses every time point from both sibling tasks;
    ``basis_scope="first_task_only"`` uses only the family's first rule
    (delaydm1 or dmcgo). Both rules remain in the metadata because both sets of
    fixed points are later projected through the reference-task basis.

    Hidden activity is small enough for ordinary PCA. Effective modulation
    ``W ⊙ M(t)`` is flattened to hidden×embedding dimensions and fed to
    IncrementalPCA in chunks, avoiding a multi-gigabyte materialized matrix.
    ``n_components`` is six in the sibling analysis, so every displayed PC-pair
    view comes from trajectory PCA rather than a PCA refit on fixed points.

    The joint artifact keeps the historical filename; the reference-task artifact
    adds ``_{first_rule}_only`` before ``_{aname}``. paper_plot.py uses these
    stored bases directly and never refits on fixed points.
    """
    if len(norm_trials) != len(rules):
        raise ValueError(f"{addtask}: got {len(norm_trials)} trial objects for "
                         f"{len(rules)} rules")

    delay_windows = []
    for rule, trial in zip(rules, norm_trials):
        if "delay1" not in trial.epochs:
            raise KeyError(f"{rule}: trial has no delay1 epoch")
        start, stop = trial.epochs["delay1"]
        if np.ndim(start) or np.ndim(stop):
            raise ValueError(f"{rule}: delay1 must be a scalar aligned window, "
                             f"got {(start, stop)}")
        delay_windows.append((int(start), int(stop)))
    if len(set(delay_windows)) != 1:
        raise ValueError(f"{addtask}: sibling delay1 windows are not aligned: "
                         f"{dict(zip(rules, delay_windows))}")
    delay_start, delay_stop = delay_windows[0]
    if delay_stop <= delay_start:
        raise ValueError(f"{addtask}: empty delay1 window {delay_windows[0]}")

    hidden = _tracked_numpy(norm_db[f"hidden{layer_index}"])
    modulation = _tracked_numpy(norm_db[f"M{layer_index}"])
    if hidden.shape[:2] != modulation.shape[:2]:
        raise ValueError(f"{addtask}: hidden/M trajectory shapes disagree: "
                         f"{hidden.shape} vs {modulation.shape}")
    if delay_stop > hidden.shape[1]:
        raise ValueError(f"{addtask}: delay1 stop {delay_stop} exceeds recorded "
                         f"trajectory length {hidden.shape[1]}")

    task_idx = np.asarray(norm_task, dtype=int)
    if task_idx.size != hidden.shape[0]:
        raise ValueError(f"{addtask}: {task_idx.size} task labels for "
                         f"{hidden.shape[0]} trajectories")
    trial_counts = {rule: int(np.sum(task_idx == i))
                    for i, rule in enumerate(rules)}
    if any(v == 0 for v in trial_counts.values()):
        raise ValueError(f"{addtask}: a sibling task has no delay trajectories: "
                         f"{trial_counts}")

    scope = _delay_pca_scope_spec(rules, basis_scope)
    fit_mask = np.isin(task_idx, scope["fit_rule_indices"])
    fit_rules = [rules[i] for i in scope["fit_rule_indices"]]
    source = scope["source"]
    artifact_suffix = scope["artifact_suffix"]

    fit_trial_counts = {rule: trial_counts[rule] for rule in fit_rules}
    # Logical concatenation order follows the batch order returned by
    # generate_trials_wrap, restricted to the requested PCA-fitting task(s).
    h_delay = np.asarray(
        hidden[fit_mask, delay_start:delay_stop], dtype=np.float32)
    h_samples = h_delay.reshape(-1, h_delay.shape[-1])
    if h_samples.shape[0] < 2:
        raise ValueError(f"{addtask}: need at least two delay samples for PCA")
    n_components = int(n_components)
    if n_components < 2:
        raise ValueError(f"n_components must be >=2, got {n_components}")
    max_hidden_components = min(h_samples.shape)
    if n_components > max_hidden_components:
        raise ValueError(f"requested {n_components} hidden PCs but only "
                         f"{max_hidden_components} are available")
    hidden_pca = PCA(n_components=n_components, svd_solver="randomized",
                     random_state=0)
    hidden_pca.fit(h_samples)

    M_delay = np.asarray(modulation[fit_mask, delay_start:delay_stop])
    if M_delay.ndim < 4:
        raise ValueError(f"{addtask}: expected M as (batch,time,hidden,embed), "
                         f"got {M_delay.shape}")
    W = np.asarray(W, dtype=np.float32)
    if tuple(M_delay.shape[-2:]) != tuple(W.shape):
        raise ValueError(f"{addtask}: M/W shapes disagree: "
                         f"{M_delay.shape[-2:]} vs {W.shape}")
    M_samples = M_delay.reshape(-1, *M_delay.shape[-2:])
    n_samples = int(M_samples.shape[0])
    if n_samples < 2:
        raise ValueError(f"{addtask}: need at least two W⊙M delay samples for PCA")

    # IncrementalPCA sees every sample from the logical concatenation. Keep the
    # final chunk at least n_components large by merging a tiny tail into the
    # preceding chunk.
    chunk_samples = max(int(chunk_samples), n_components)
    wm_pca = IncrementalPCA(n_components=n_components)
    start = 0
    n_chunks = 0
    while start < n_samples:
        remaining = n_samples - start
        stop = (n_samples if remaining <= chunk_samples + n_components - 1
                else start + chunk_samples)
        wm_chunk = (np.asarray(M_samples[start:stop], dtype=np.float32)
                    * W[None, :, :]).reshape(stop - start, -1)
        wm_pca.partial_fit(wm_chunk)
        start = stop
        n_chunks += 1
    print(f"  [{addtask}/delay-pca/{basis_scope}] fitting trajectories: "
          f"{fit_trial_counts}, window=({delay_start}, {delay_stop}), "
          f"samples={n_samples}")
    print(f"  [{addtask}/delay-pca/{basis_scope}] hidden PCA EVR="
          f"{hidden_pca.explained_variance_ratio_.sum():.3f}; "
          f"W⊙M incremental PCA EVR={wm_pca.explained_variance_ratio_.sum():.3f} "
          f"in {n_chunks} chunks")

    artifact = {
        "version": 3,
        "aname": aname,
        "family": addtask,
        "rules": list(rules),
        "basis_scope": basis_scope,
        "fit_rules": fit_rules,
        "period": "delay1",
        "period_window": (delay_start, delay_stop),
        "source": source,
        "trial_counts": trial_counts,
        "fit_trial_counts": fit_trial_counts,
        "timepoints_per_trial": int(delay_stop - delay_start),
        "n_components": n_components,
        "representations": {
            "fixed_hidden": _pca_record(hidden_pca, h_samples.shape[0],
                                         "PCA(randomized)"),
            "fixed_WM": _pca_record(wm_pca, n_samples,
                                     f"IncrementalPCA(chunk_samples={chunk_samples})"),
        },
    }
    out_path = (Path(save_dir)
                / f"{addtask}_delay_trajectory_pca{artifact_suffix}_{aname}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Saved {basis_scope} delay-trajectory PCA basis: {out_path}")
    return out_path


_FIXED_POINT_REPS = (("hidden", "fixed_hidden"),
                     ("e_modulation", "fixed_WM"))


def _load_fixed_point_rules(aname, save_dir, rules):
    """Load sibling-rule fixed-point pickles in one shared, validated order."""
    per_rule = []
    for rule in rules:
        path = Path(save_dir) / f"fixed_points_grad_{aname}_{rule}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            per_rule.append((rule, pickle.load(f)))
    return per_rule


def _stimulus_color(stim, n_stim):
    """Match paper_plot.py's red-to-purple circular-stimulus color ramp."""
    frac = (int(stim) % n_stim) / max(n_stim - 1, 1)
    return mpl.colors.hsv_to_rgb((0.83 * frac, 0.85, 0.9))


def _adaptive_limits(xy, padding=0.10):
    """Independent x/y limits from the joint extent of both displayed cycles."""
    xy = np.asarray(xy, dtype=float)
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


def _save_pc_pair_gallery(save_dir, aname, addtask, artifact_suffix,
                          basis_scope, plot_name, entry):
    """Draw all 15 pairwise views of a six-PC fixed-point projection.

    This is a visual selection aid only: it computes no score, highlights no
    panel, and writes no selected-plane metadata. paper_plot.py remains the sole
    place where a paper PC pair is specified.
    """
    proj = np.asarray(entry["proj"], dtype=float)
    if proj.ndim != 2 or proj.shape[1] != 6:
        raise ValueError(f"{addtask}/{basis_scope}/{plot_name}: gallery expects "
                         f"six PCs, got {proj.shape}")
    task_idx = np.asarray(entry["task_idx"], dtype=int)
    stim_idx = np.asarray(entry["stim_idx"], dtype=int)
    is_fixed = np.asarray(entry["is_fixed"], dtype=bool)
    task_names = list(entry["task_names"])
    n_stim = int(stim_idx.max()) + 1
    evr = np.asarray(entry.get("explained_variance_ratio", []), dtype=float)
    pairs = [(x, y) for x in range(6) for y in range(x + 1, 6)]
    markers = ("s", "^")

    fig, axs = plt.subplots(3, 5, figsize=(13.5, 8.2), squeeze=False)
    for ax, (pc_x, pc_y) in zip(axs.flat, pairs):
        for task, rule in enumerate(task_names):
            sel_task = task_idx == task
            if addtask == "delaydm1":
                good_task = sel_task & is_fixed
                order = np.argsort(stim_idx[good_task])
                ring = proj[good_task][order][:, [pc_x, pc_y]]
                if ring.shape[0] >= 2:
                    ring = np.vstack([ring, ring[:1]])
                    ax.plot(ring[:, 0], ring[:, 1], color="0.60", lw=0.7,
                            alpha=0.7, linestyle=("-", "--")[task % 2],
                            zorder=1)
            for stim in np.unique(stim_idx[sel_task]):
                sel = sel_task & (stim_idx == stim)
                color = _stimulus_color(stim, n_stim)
                converged = bool(is_fixed[sel].all())
                ax.scatter(proj[sel, pc_x], proj[sel, pc_y],
                           color=color if converged else "none",
                           edgecolor="none" if converged else color,
                           linewidth=0 if converged else 0.9,
                           marker=markers[task % len(markers)], s=25, zorder=2)

        xlim, ylim = _adaptive_limits(proj[:, [pc_x, pc_y]])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        x_var = f" ({100 * evr[pc_x]:.1f}%)" if evr.size > pc_x else ""
        y_var = f" ({100 * evr[pc_y]:.1f}%)" if evr.size > pc_y else ""
        ax.set_xlabel(f"PC{pc_x + 1}{x_var}", fontsize=7)
        ax.set_ylabel(f"PC{pc_y + 1}{y_var}", fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([], [], marker=markers[t % len(markers)],
                          color="0.3", linestyle="", markersize=5, label=rule)
               for t, rule in enumerate(task_names)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=len(handles), frameon=True, fontsize=8)
    fig.suptitle(f"{addtask} | {basis_scope} delay PCA | {plot_name} | "
                 "all 15 PC pairs (no automatic selection)", fontsize=12,
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = (Path(save_dir)
            / f"{addtask}_delay_pc_gallery{artifact_suffix}_{plot_name}_{aname}.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"  Saved 15-pair PC gallery: {path}")
    return path


def save_sibling_fixed_point_pc_projections(
        aname, save_dir, addtask, rules, probe="longdelay", basis_scope="joint"):
    """Save one sibling family's fixed points in a six-PC trajectory basis.

    This upstream step deliberately makes no two-dimensional plane choice and
    computes no alignment score. The stored six-dimensional coordinates contain
    every possible PC-pair projection; paper_plot.py selects a pair explicitly.
    DMC category labels are included as metadata without scoring their separation.
    """
    if addtask not in ("delaydm1", "dmcgo"):
        raise ValueError("six-PC projection is defined for delayDM and DMC")
    if len(rules) != 2:
        raise ValueError(f"expected two sibling rules, got {rules}")

    save_dir = Path(save_dir)
    scope = _delay_pca_scope_spec(rules, basis_scope)
    artifact_suffix = scope["artifact_suffix"]
    basis_path = (save_dir
                  / f"{addtask}_delay_trajectory_pca{artifact_suffix}_{aname}.pkl")
    with open(basis_path, "rb") as f:
        basis_artifact = pickle.load(f)
    if basis_artifact.get("basis_scope") != basis_scope:
        raise ValueError(f"{basis_path}: expected basis_scope={basis_scope!r}, "
                         f"got {basis_artifact.get('basis_scope')!r}")
    if list(basis_artifact.get("rules", [])) != list(rules):
        raise ValueError(f"{basis_path}: expected rules {rules}, got "
                         f"{basis_artifact.get('rules')}")

    per_rule = _load_fixed_point_rules(aname, save_dir, rules)

    out = {
        "version": 1,
        "aname": aname,
        "family": addtask,
        "task_names": list(rules),
        "probe": probe,
        "basis_scope": basis_scope,
        "pca_source": str(basis_path),
        "pca_fit_source": basis_artifact["source"],
        "n_components": 6,
        "available_pc_pairs": [(x + 1, y + 1)
                               for x in range(6) for y in range(x + 1, 6)],
        "representations": {},
    }
    for plot_name, fp_key in _FIXED_POINT_REPS:
        record = basis_artifact.get("representations", {}).get(fp_key)
        if record is None:
            raise KeyError(f"{basis_path}: missing PCA representation {fp_key!r}")
        mean = np.asarray(record["mean"], dtype=float)
        components = np.asarray(record["components"], dtype=float)
        if components.ndim != 2 or components.shape[0] < 6:
            raise ValueError(f"{basis_path}: {fp_key} needs >=6 trajectory PCs, "
                             f"got {components.shape}")
        components = components[:6]

        projected, task_labels, stim_labels, fixed_labels = [], [], [], []
        for task, (rule, data) in enumerate(per_rule):
            entry = data.get("results", {}).get(probe)
            if entry is None or entry.get(fp_key) is None:
                raise KeyError(f"{rule}: missing {probe!r}/{fp_key!r}")
            values = np.asarray(entry[fp_key], dtype=float)
            values = values.reshape(values.shape[0], -1)
            if values.shape[1] != mean.size:
                raise ValueError(f"{rule}/{fp_key}: {values.shape[1]} features, "
                                 f"trajectory PCA expects {mean.size}")
            projected.append((values - mean) @ components.T)
            stim = np.asarray(entry["stim"], dtype=int)
            task_labels.extend([task] * stim.size)
            stim_labels.extend(stim.tolist())
            fixed_labels.extend(np.asarray(
                entry.get("is_fixed", np.ones(stim.size, bool)), dtype=bool).tolist())

        proj = np.vstack(projected)
        task_idx = np.asarray(task_labels, dtype=int)
        stim_idx = np.asarray(stim_labels, dtype=int)
        is_fixed = np.asarray(fixed_labels, dtype=bool)
        rep_out = {
            "proj": np.asarray(proj, dtype=np.float32),
            "task_idx": task_idx,
            "stim_idx": stim_idx,
            "is_fixed": is_fixed,
            "task_names": list(rules),
            "explained_variance_ratio": np.asarray(
                record.get("explained_variance_ratio", []), dtype=float)[:6],
        }
        if addtask == "dmcgo":
            n_stim = int(stim_idx.max()) + 1
            rep_out["group_labels"] = _dmc_category_labels(
                task_idx, stim_idx, n_stim)
        gallery_path = _save_pc_pair_gallery(
            save_dir, aname, addtask, artifact_suffix, basis_scope,
            plot_name, rep_out)
        rep_out["gallery_path"] = str(gallery_path)
        out["representations"][plot_name] = rep_out
        print(f"  [{addtask}/pc-projection/{basis_scope}] {plot_name}: saved "
              f"{proj.shape[0]} points in all six Delay PCs")

    out_path = (save_dir
                / f"{addtask}_delay_pc_projections{artifact_suffix}_{aname}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"  Saved {basis_scope} {addtask} fixed-point PC projections: "
          f"{out_path}")
    return out_path


def _dmc_category_labels(task_idx, stim_idx, n_stim):
    """DMC category label per solved fixed point, pooled across the two rules.

    The split cuts across task and stimulus identity:
        Group A: task0 stim {0..n/2-1} + task1 stim {n/2..n-1}
        Group B: task0 stim {n/2..n-1} + task1 stim {0..n/2-1}
    Thus group A iff (task == 0) == (stim < n_stim/2). A separating plane
    encodes remembered category rather than stimulus identity or task cue.
    """
    half = n_stim // 2
    return np.asarray([int((t == 0) == (a < half))
                       for t, a in zip(task_idx, stim_idx)], dtype=int)
