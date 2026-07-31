#!/usr/bin/env python
# coding: utf-8
"""
Post-training analysis of a two-task MPN.

Reloads the trained network + full training bundle saved by two_task.py and
reproduces ALL of the analyses that previously lived in the monolithic
`two_task_analysis.ipynb` (cells 13-106): cross-task / cross-period PCA,
attractor / cosine-similarity-over-learning, dPCA, fixon/task cancellation,
weight-structure heatmaps, magnitude pruning, and the interpolation /
fixed-point ring analyses.

Unlike one_task_analysis.py — which is purely trace-driven — the two-task
analysis repeatedly needs the LIVE trained network (it runs the net on freshly
interpolated inputs, prunes its weights, reads its weight matrices). So this
script rebuilds the network from the checkpoint and runs it; the per-stage
training traces are reloaded from the bundle and are aligned to exactly the
test trials two_task.py saved.

Matplotlib figures are written into ./twotasks/{aname}/ with the same filenames
as the notebook. Interactive Plotly figures (which the notebook displayed with
fig.show()) are saved as standalone .html files in the same directory.
Cross-run pickle summaries go to ./twotasks_data/.

Usage:
    python two_task_analysis.py                 # all runs in ./twotasks/
    python two_task_analysis.py --aname <name>  # a specific run
"""
import os
import gc
import copy
import glob
import json
import pickle
import argparse
from pathlib import Path
from itertools import chain

import numpy as np
import torch

from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
ticker.Locator.MAXTICKS = 10000
import seaborn as sns
from matplotlib.lines import Line2D

# Match the plotting style used in one_task_analysis.py / multiple_task_analysis.py
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

import plotly.graph_objects as go

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import networks as nets
import net_helpers
import mpn_tasks
import helper
import mpn
from grad_fixed_points import (solve_period_modulation_fixed_points,
                                derive_fixed_point_views, _PERIOD_TITLE)
from fixed_point import find_modulation_fixed_points

# ─── Plotting palette (notebook cell 2) ──────────────────────────────────────
# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e'] * 10
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089'] * 10
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16'] * 10
l_vals = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
markers_vals = ['o', 'v', '*', 'x', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
linestyles = ["-", "--", "-."]

OUT_DIR = Path("twotasks")
DATA_DIR = Path("twotasks_data")


# ═══════════════════════════════════════════════════════════════════════════
# Pure helper functions (notebook cells 19, 23, 25, 26, 34, 35, 39, 42, 59,
# 65, 81, 84, 89, 95, 96). These do not reference notebook globals; anything
# they need is passed explicitly.
# ═══════════════════════════════════════════════════════════════════════════
def dimensionality_measure(W, n_hidden):
    """Participation-ratio dimensionality (Recanatesi et al., 2019, Eq. 3).
    Returns a value in (0, 1]. (Notebook cell 19.)"""
    covW = np.cov(W)
    assert covW.shape[0] == n_hidden
    eigenvalues, eigenvectors = np.linalg.eig(covW)
    numerator = np.sum(eigenvalues) ** 2
    denominator = np.sum(eigenvalues ** 2)
    return (numerator / denominator) / W.shape[0]


def sample_non_nan(arr, k):
    """Pick `k` distinct (non-NaN) numbers from a 2-D array. (Cell 23.)"""
    pool = arr[~np.isnan(arr)]
    if k > pool.size:
        raise ValueError("k exceeds number of non-NaN entries.")
    return np.random.choice(pool, k, replace=False).tolist()


def assert_sums_close(arr_list, rtol=1e-5, atol=1e-8):
    """Assert every array in the list has (nearly) the same sum. (Cell 23.)"""
    assert len(arr_list) > 0, "Empty list."
    sums = np.array([np.sum(a) for a in arr_list], dtype=float)
    ref = sums[0]
    ok = np.isclose(sums, ref, rtol=rtol, atol=atol)
    if not np.all(ok):
        bad = np.where(~ok)[0]
        raise AssertionError(
            f"Sum mismatch at indices {bad.tolist()}.\n"
            f"ref_sum={ref}, bad_sums={sums[bad].tolist()}, all_sums={sums.tolist()}"
        )


def modulation_extraction(test_input, db, layer_index):
    """Extract (Ms, Ms_orig, hs, bs) from a recorded activation dict. (Cell 26.)"""
    def _to_numpy(x):
        try:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _concat_last(x):
        return np.concatenate(x, axis=-1) if isinstance(x, (list, tuple)) else x

    n_batch, max_seq_len = test_input.shape[0], test_input.shape[1]

    M_raw = _concat_last(_to_numpy(db[f"M{layer_index}"]))
    Ms = M_raw.reshape(n_batch, max_seq_len, -1)
    Ms_orig = M_raw

    bs = _concat_last(_to_numpy(db[f"b{layer_index}"]))

    H_raw = _concat_last(_to_numpy(db[f"hidden{layer_index}"]))
    hs = H_raw.reshape(n_batch, max_seq_len, -1)

    return Ms, Ms_orig, hs, bs


def modulation_magnitude_by_component(save_dir, aname, Ms_orig, W_input,
                                      shift_index, fixate_off, dt,
                                      fixation_end, stimulus_end, delay_end):
    """Modulation-computation magnitude across time, per input component, for the
    two-task network. Mirrors the one-task analysis: for each raw input channel c,
    the modulation applied to that channel is the hidden-unit vector obtained by
    contracting M's embedded-input axis with that channel's embedding column,
        p_c[b, t, :] = M[b, t, :, :] @ W_input[:, c]     (raw M — Hebbian trace)
    and its per-timestep magnitude is the L2 norm over hidden units, averaged over
    trials with a ±std band.

    Four summary curves: Fixation, the combined Stimulus (per-trial MEAN over the
    stimulus channels), and the two Task cues. The two-task input layout matches
    the cancellation cell: channel 0 = fixation; the stimulus block is
    [2-shift_index : 6-shift_index]; the last two channels are task cue 1 and 2.

    `Ms_orig` : (batch, T, hidden, embed) final-stage modulation.
    `W_input` : (embed, n_raw) input embedding (identity if no input layer).
    Saves modulation_magnitude_{aname}.png and .pkl (for paper_plot reuse).
    """
    T = Ms_orig.shape[1]
    n_raw = W_input.shape[1]
    # Project M onto every raw input column at once: (batch, T, hidden, n_raw),
    # then L2 norm over hidden units → per-trial magnitude per channel.
    proj = np.einsum("bthe,er->bthr", Ms_orig, W_input)
    mag = np.linalg.norm(proj, axis=2)                 # (batch, T, n_raw)
    n_batch = mag.shape[0]

    def _mean_std(series_bt):
        return series_bt.mean(axis=0), series_bt.std(axis=0)

    # Channel layout (same as the cancellation cell): fixation = 0; stimulus block
    # = [2-shift_index : 6-shift_index]; task cues = the last two channels.
    fix_ch = 0
    stim_cols = list(range(2 - shift_index, 6 - shift_index))
    stim_cols = [c for c in stim_cols if 0 <= c < n_raw]
    task1_ch, task2_ch = n_raw - 2, n_raw - 1

    fix_mean, fix_std = _mean_std(mag[:, :, fix_ch])
    task1_mean, task1_std = _mean_std(mag[:, :, task1_ch])
    task2_mean, task2_std = _mean_std(mag[:, :, task2_ch])
    if stim_cols:
        stim_mag = mag[:, :, stim_cols].mean(axis=2)   # per-trial mean over block
        stim_mean, stim_std = _mean_std(stim_mag)
    else:
        stim_mean = stim_std = np.zeros(T)

    # Four curves: Fixation, Stimulus (combined), Task cue 1, Task cue 2.
    series = [
        ("Fixation", fix_mean, fix_std, c_vals[6]),
        ("Stimulus", stim_mean, stim_std, c_vals[2]),
        ("Task cue 1", task1_mean, task1_std, c_vals[4]),
        ("Task cue 2", task2_mean, task2_std, c_vals[1]),
    ]
    t_ms = np.arange(T) * dt
    fig, ax = plt.subplots(figsize=(6, 3))
    for lab, mean, std, col in series:
        ax.plot(t_ms, mean, "-", color=col, label=lab)
        ax.fill_between(t_ms, mean - std, mean + std, color=col, alpha=0.25, lw=0)
    for bt in (fixation_end, stimulus_end, delay_end):
        if bt is not None:
            ax.axvline(bt * dt, color="0.5", lw=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Time (ms)", fontsize=13)
    ax.set_ylabel("Modulation magnitude\n(‖M·x_c‖₂)", fontsize=12)
    ax.legend(loc="best", frameon=True, fontsize=9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / f"modulation_magnitude_{aname}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {save_dir / f'modulation_magnitude_{aname}.png'}")

    with open(save_dir / f"modulation_magnitude_{aname}.pkl", "wb") as f:
        pickle.dump({
            "aname": aname,
            "dt": int(dt),
            "labels": [lab for lab, *_ in series],
            # Per-curve mean and across-trial std, stacked (n_curve, T).
            "mean": np.asarray([m for _, m, _, _ in series], dtype=float),
            "std": np.asarray([s for _, _, s, _ in series], dtype=float),
            "stim_channels": [int(c) for c in stim_cols],
            "fixation_end": None if fixation_end is None else int(fixation_end),
            "stimulus_end": None if stimulus_end is None else int(stimulus_end),
            "delay_end": None if delay_end is None else int(delay_end),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved modulation magnitude data: "
          f"{save_dir / f'modulation_magnitude_{aname}.pkl'}")


def analyze_similarity(Ms_orig, hs, net, net_params, label_task_comb, checktime,
                       compare="modulation", moddim=0):
    """Cosine-similarity structure of modulation / hidden states across the
    same-stimulus / same-response / different-stimulus groupings. (Cell 25.)"""
    inverse_modulation_ss_dt = []
    inverse_modulation_sr_dt = []
    modulation_save = [[], []]
    modulation_save_time = [[], []]
    hidden_save_time = [[], []]

    if net_params["input_layer_add"]:
        W = net.mp_layer1.W.data.detach().cpu().numpy()
    else:
        W = net.mp_layer0.W.data.detach().cpu().numpy()

    if compare == "w_modulation":
        Ms_orig = Ms_orig * W[None, None, :, :]

    # same stimulus (effectively anti-response), different task
    for k in range(8):
        ind1 = [i for i, lst in enumerate(label_task_comb) if np.array_equal(lst, [k, 0])]
        ind2 = [i for i, lst in enumerate(label_task_comb) if np.array_equal(lst, [k, 1])]
        ll = min(len(ind1), len(ind2))

        if net_params["input_layer_add"]:
            win = net.W_initial_linear.weight.data.detach().cpu().numpy()

        if compare in ("modulation", "w_modulation"):
            winadd = False if moddim is None else True
            if winadd:
                if net_params["input_layer_add"]:
                    if moddim == "Win":
                        Ms1_change_stimulus = [((Ms_orig[ind1[i], checktime, :, :]) @ win).flatten() for i in range(ll)]
                        Ms2_change_stimulus = [((Ms_orig[ind2[i], checktime, :, :]) @ win).flatten() for i in range(ll)]
                    else:
                        Ms1_change_stimulus = [((Ms_orig[ind1[i], checktime, :, :]) @ win)[:, moddim].flatten() for i in range(ll)]
                        Ms2_change_stimulus = [((Ms_orig[ind2[i], checktime, :, :]) @ win)[:, moddim].flatten() for i in range(ll)]
                else:
                    if moddim == "Win":
                        Ms1_change_stimulus = [((Ms_orig[ind1[i], checktime, :, :])).flatten() for i in range(ll)]
                        Ms2_change_stimulus = [((Ms_orig[ind2[i], checktime, :, :])).flatten() for i in range(ll)]
                    else:
                        Ms1_change_stimulus = [((Ms_orig[ind1[i], checktime, :, :]))[:, moddim].flatten() for i in range(ll)]
                        Ms2_change_stimulus = [((Ms_orig[ind2[i], checktime, :, :]))[:, moddim].flatten() for i in range(ll)]
            else:
                Ms1_change_stimulus = [(Ms_orig[ind1[i], checktime, :, :]).flatten() for i in range(ll)]
                Ms2_change_stimulus = [(Ms_orig[ind2[i], checktime, :, :]).flatten() for i in range(ll)]
        elif compare == "hidden":
            Ms1_change_stimulus = [hs[ind1[i], checktime, :].flatten() for i in range(ll)]
            Ms2_change_stimulus = [hs[ind2[i], checktime, :].flatten() for i in range(ll)]

        assert_sums_close(Ms1_change_stimulus, rtol=1e-3, atol=1e-3)
        assert_sums_close(Ms2_change_stimulus, rtol=1e-3, atol=1e-3)

        inverse_modulation_ss_dt.append(1 - cosine(Ms1_change_stimulus[0], Ms2_change_stimulus[0]))

        modulation_save[0].append(Ms1_change_stimulus[0])
        modulation_save[1].append(Ms2_change_stimulus[0])

        Ms1_all = Ms_orig[ind1[0], :, :, :]
        Ms2_all = Ms_orig[ind2[0], :, :, :]
        h1_all = hs[ind1[0], :, :]
        h2_all = hs[ind2[0], :, :]
        modulation_save_time[0].append(Ms1_all)
        modulation_save_time[1].append(Ms2_all)
        hidden_save_time[0].append(h1_all)
        hidden_save_time[1].append(h2_all)

    # same response, different task
    for k in range(8):
        ind1 = [i for i, lst in enumerate(label_task_comb) if np.array_equal(lst, [k, 0])]
        ind2 = [i for i, lst in enumerate(label_task_comb) if np.array_equal(lst, [(k + 4) % 8, 1])]
        ll = min(len(ind1), len(ind2))

        if compare in ("modulation", "w_modulation"):
            winadd = False if moddim is None else True
            if winadd:
                if net_params["input_layer_add"]:
                    if moddim == "Win":
                        Ms1_change_stimulus = [((Ms_orig[ind1[i], checktime, :, :]) @ win).flatten() for i in range(ll)]
                        Ms2_change_stimulus = [((Ms_orig[ind2[i], checktime, :, :]) @ win).flatten() for i in range(ll)]
                    else:
                        Ms1_change_stimulus = [((Ms_orig[ind1[i], checktime, :, :]) @ win)[:, moddim].flatten() for i in range(ll)]
                        Ms2_change_stimulus = [((Ms_orig[ind2[i], checktime, :, :]) @ win)[:, moddim].flatten() for i in range(ll)]
                else:
                    if moddim == "Win":
                        Ms1_change_stimulus = [((Ms_orig[ind1[i], checktime, :, :])).flatten() for i in range(ll)]
                        Ms2_change_stimulus = [((Ms_orig[ind2[i], checktime, :, :])).flatten() for i in range(ll)]
                    else:
                        Ms1_change_stimulus = [((Ms_orig[ind1[i], checktime, :, :]))[:, moddim].flatten() for i in range(ll)]
                        Ms2_change_stimulus = [((Ms_orig[ind2[i], checktime, :, :]))[:, moddim].flatten() for i in range(ll)]
            else:
                Ms1_change_stimulus = [(Ms_orig[ind1[i], checktime, :, :]).flatten() for i in range(ll)]
                Ms2_change_stimulus = [(Ms_orig[ind2[i], checktime, :, :]).flatten() for i in range(ll)]
        elif compare == "hidden":
            Ms1_change_stimulus = [hs[ind1[i], checktime, :].flatten() for i in range(ll)]
            Ms2_change_stimulus = [hs[ind2[i], checktime, :].flatten() for i in range(ll)]

        assert_sums_close(Ms1_change_stimulus, rtol=1e-3, atol=1e-3)
        assert_sums_close(Ms2_change_stimulus, rtol=1e-3, atol=1e-3)

        inverse_modulation_sr_dt.append(1 - cosine(Ms1_change_stimulus[0], Ms2_change_stimulus[0]))

    # same task, different stimulus
    repeat = 100
    modulation_matrices_all = []
    for _ in range(repeat):
        modulation_matrices = [
            np.full((len(modulation_save[0]), len(modulation_save[0])), np.nan),
            np.full((len(modulation_save[0]), len(modulation_save[0])), np.nan),
        ]
        for i in range(len(modulation_save[0])):
            for j in range(i + 1, len(modulation_save[0])):
                modulation_matrices[0][i, j] = 1 - cosine(modulation_save[0][i], modulation_save[0][j])
                modulation_matrices[1][i, j] = 1 - cosine(modulation_save[1][i], modulation_save[1][j])
        try:
            modulation_matrices_all.append([np.nanmean(sample_non_nan(modulation_matrices[0], 8)),
                                            np.nanmean(sample_non_nan(modulation_matrices[1], 8))])
        except Exception:
            modulation_matrices_all.append([np.nan, np.nan])

    modulation_matrices_all = np.array(modulation_matrices_all)

    result = [[np.mean(inverse_modulation_ss_dt), np.std(inverse_modulation_ss_dt)],
              [np.mean(inverse_modulation_sr_dt), np.std(inverse_modulation_sr_dt)],
              [np.mean(modulation_matrices_all[:, 0]), np.std(modulation_matrices_all[:, 0])],
              [np.mean(modulation_matrices_all[:, 1]), np.std(modulation_matrices_all[:, 1])]]

    return result, modulation_save_time, hidden_save_time


def input_change(U, X):
    """Global RMS gain from a 3-channel input U to its embedding X. (Cell 34.)"""
    eps = 1e-12
    u2 = np.sum(U ** 2, axis=-1)
    x2 = np.sum(X ** 2, axis=-1)
    g_rms = np.sqrt(x2.mean()) / (np.sqrt(u2.mean()) + eps)
    return g_rms


def cosine_sim(a, b):
    return 1.0 - cosine(a, b)


def vec_angle_deg(u, v, sign_invariant=False, eps=1e-12):
    """Angle (deg) between two vectors. (Cell 39.)"""
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    cu = np.linalg.norm(u)
    cv = np.linalg.norm(v)
    if cu < eps or cv < eps:
        return np.nan
    c = np.dot(u, v) / (cu * cv)
    c = np.clip(c, -1.0, 1.0)
    if sign_invariant:
        c = abs(c)
    return np.degrees(np.arccos(c))


def figure2A_pca_fve(H, task_id, periods, k=2, max_pcs=10, center="global",
                     flatten="trial_time", dtype=np.float64, return_cross_task=True):
    """Cross-period (and optional cross-task) PCA explained-variance. (Cell 42.)"""
    if hasattr(H, "detach"):
        H_np = H.detach().cpu().numpy()
    else:
        H_np = np.asarray(H)
    H_np = H_np.astype(dtype, copy=False)

    task_id = np.asarray(task_id)
    B, T, N = H_np.shape

    def _get_period_matrix(H_task, t0, t1):
        X = H_task[:, t0:t1, :]
        if flatten == "trial_time":
            X = X.reshape(-1, N)
        else:
            raise ValueError(f"Unsupported flatten mode: {flatten}")
        return X

    def _center(X, mean=None):
        if center == "none":
            mu = np.zeros((X.shape[1],), dtype=X.dtype) if mean is None else mean
            return X, mu
        if mean is None:
            mu = X.mean(axis=0)
        return X - mu, mu

    def _pca_svd(X, r):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r_eff = min(r, Vt.shape[0])
        V = Vt[:r_eff, :].T
        S = S[:r_eff]
        M = X.shape[0]
        denom = (np.sum(X * X) / max(M - 1, 1))
        evals = (S * S) / max(M - 1, 1)
        evr = evals / denom if denom > 0 else np.zeros_like(evals)
        return V, S, evr

    def _fve_project(X, V):
        tot = np.sum(X * X)
        if tot <= 0:
            return 0.0
        XV = X @ V
        Xhat = XV @ V.T
        num = np.sum(Xhat * Xhat)
        return float(num / tot)

    results = {}
    all_labels = []
    all_Xc = {}
    all_Vk = {}

    for task, per_dict in periods.items():
        idx = np.where(task_id == task)[0]
        if idx.size == 0:
            continue

        H_task = H_np[idx, :, :]
        period_names = list(per_dict.keys())
        P = len(period_names)

        pca_info = {}
        X_period_centered = {}

        for pname in period_names:
            t0, t1 = per_dict[pname]
            if not (0 <= t0 < t1 <= T):
                raise ValueError(f"[{task}:{pname}] invalid period bounds {(t0, t1)} for T={T}")
            X = _get_period_matrix(H_task, t0, t1)
            Xc, mu = _center(X)
            X_period_centered[pname] = Xc

            V, S, evr = _pca_svd(Xc, r=max(max_pcs, k))
            pca_info[pname] = {"components": V, "singular_values": S, "mean": mu, "evr": evr}

            if return_cross_task:
                key = (task, pname)
                all_labels.append(key)
                all_Xc[key] = Xc
                k_eff = min(k, V.shape[1])
                all_Vk[key] = V[:, :k_eff]

        fve_k = np.zeros((P, P), dtype=dtype)
        for i, px in enumerate(period_names):
            Xc = X_period_centered[px]
            for j, py in enumerate(period_names):
                V = pca_info[py]["components"]
                r_eff = min(k, V.shape[1])
                fve_k[i, j] = _fve_project(Xc, V[:, :r_eff])

        evr_curves = np.zeros((P, max_pcs), dtype=dtype)
        for i, pname in enumerate(period_names):
            evr = pca_info[pname]["evr"]
            evr_curves[i, :min(max_pcs, evr.shape[0])] = evr[:min(max_pcs, evr.shape[0])]

        results[task] = {"period_names": period_names, "fve_k": fve_k,
                         "evr_curves": evr_curves, "pca": pca_info}

    if return_cross_task:
        Q = len(all_labels)
        fve_k_all = np.zeros((Q, Q), dtype=dtype)
        for i, key_x in enumerate(all_labels):
            Xc = all_Xc[key_x]
            for j, key_y in enumerate(all_labels):
                Vk = all_Vk[key_y]
                fve_k_all[i, j] = _fve_project(Xc, Vk)
        results["__cross_task__"] = {"labels": all_labels, "fve_k_all": fve_k_all}

    return results


def principal_angle_cosines(W_proj, stim_idx, control_idx, eps=1e-12):
    """Principal-angle cosines between two column subspaces. (Cell 59.)"""
    S = W_proj[:, stim_idx]
    C = W_proj[:, control_idx]
    QS, _ = np.linalg.qr(S)
    QC, _ = np.linalg.qr(C)
    M = QS.T @ QC
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, 0.0, 1.0)
    return sigmas


def subspace_orthogonality_report(W_proj, stim_idx, control_idx):
    sigmas = principal_angle_cosines(W_proj, stim_idx, control_idx)
    max_cos = float(sigmas.max()) if sigmas.size else 0.0
    angles_deg = np.degrees(np.arccos(sigmas)) if sigmas.size else np.array([])
    return {"cosines": sigmas, "max_cos": max_cos, "angles_deg": angles_deg}


def bin_by_sorted_x(x, y, nbins=100, drop_nonfinite=True, return_counts=False):
    """Bin y by sorted x into nbins equal-count chunks. (Cell 65.)"""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same number of elements; got {x.size} vs {y.size}")
    if drop_nonfinite:
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
    n = x.size
    if n == 0:
        raise ValueError("No valid data points after filtering.")
    if nbins < 1:
        raise ValueError("nbins must be >= 1.")
    nb = min(nbins, n)
    idx = np.argsort(x)
    x_s = x[idx]
    y_s = y[idx]
    x_chunks = np.array_split(x_s, nb)
    y_chunks = np.array_split(y_s, nb)
    x_mean = np.array([c.mean() for c in x_chunks])
    y_mean = np.array([c.mean() for c in y_chunks])
    if return_counts:
        counts = np.array([c.size for c in x_chunks], dtype=int)
        return x_mean, y_mean, counts
    return x_mean, y_mean


def input_interpolation(test_input_long, test_output_long, label_task_comb_long,
                        expand_stimulus=True, n_alpha=20):
    """Build alpha-interpolations between pro and anti task inputs. (Cell 81.)

    For each stimulus k, the pro (task 0) and anti (task 1) input trials are
    linearly mixed as alpha*pro + (1-alpha)*anti over `n_alpha`+1 evenly-spaced
    alphas in [0, 1]. `n_alpha` controls the grid resolution (default 20 -> 21
    steps; pass 10 for the coarse 0.0, 0.1, ... 1.0 grid)."""
    assert test_input_long.shape[0] == label_task_comb_long.shape[0]
    pro_task, anti_task = {}, {}
    pro_task_answer, anti_task_answer = {}, {}
    for k in range(8):
        ind1 = [i for i, lst in enumerate(label_task_comb_long) if np.array_equal(lst, [k, 0])]
        ind1_sample = ind1[0]
        pro_task[k] = test_input_long[ind1_sample, :, :]
        pro_task_answer[k] = test_output_long[ind1_sample, :, :]

        ind2 = [i for i, lst in enumerate(label_task_comb_long) if np.array_equal(lst, [k, 1])]
        ind2_sample = ind2[0]
        anti_task[k] = test_input_long[ind2_sample, :, :]
        anti_task_answer[k] = test_output_long[ind2_sample, :, :]

    if expand_stimulus:
        base_len = len(pro_task)
        for i in range(base_len):
            i1, i2 = i % 8, (i + 1) % 8
            pro_task[base_len + i] = (pro_task[i1] + pro_task[i2]) / 2
            anti_task[base_len + i] = (anti_task[i1] + anti_task[i2]) / 2
            pro_task_answer[base_len + i] = (pro_task_answer[i1] + pro_task_answer[i2]) / 2
            anti_task_answer[base_len + i] = (anti_task_answer[i1] + anti_task_answer[i2]) / 2

        interleaved_keys = [k for pair in zip(range(base_len), range(base_len, 2 * base_len)) for k in pair]
        pro_task = {k: pro_task[k] for k in interleaved_keys}
        anti_task = {k: anti_task[k] for k in interleaved_keys}
        pro_task_answer = {k: pro_task_answer[k] for k in interleaved_keys}
        anti_task_answer = {k: anti_task_answer[k] for k in interleaved_keys}

    n = n_alpha
    alpha_lst = [i / n for i in range(n + 1)]

    stacked_pro = torch.stack([pro_task[k] for k in sorted(pro_task)])
    stacked_anti = torch.stack([anti_task[k] for k in sorted(anti_task)])
    stacked_pro_answer = torch.stack([pro_task_answer[k] for k in sorted(pro_task_answer)])
    stacked_anti_answer = torch.stack([anti_task_answer[k] for k in sorted(anti_task_answer)])

    stacked_interpolation = [alpha_lst[i] * stacked_pro + (1 - alpha_lst[i]) * stacked_anti
                             for i in range(len(alpha_lst))]
    stacked_interpolation_ans = [alpha_lst[i] * stacked_pro_answer + (1 - alpha_lst[i]) * stacked_anti_answer
                                 for i in range(len(alpha_lst))]

    return alpha_lst, stacked_interpolation, stacked_interpolation_ans


def ring_length(pts):
    """Closed-loop perimeter of an ordered point set. (Cell 89.)"""
    diffs = np.diff(pts, axis=0, append=pts[:1])
    return np.linalg.norm(diffs, axis=1).sum()


def normalize_lst(lst, value=None):
    """Normalize a list by its first (or a given) value. (Cell 92.)

    If the divisor is zero or non-finite (e.g. a ring perimeter that collapses to
    0 at the base alpha), the ratio is undefined — return NaNs instead of dividing
    (which would emit an "invalid value in scalar divide" RuntimeWarning)."""
    if value is None:
        value = lst[0]
    if not np.isfinite(value) or value == 0:
        return [np.nan for _ in lst]
    return [val_ / value for val_ in lst]


def classify_fixed_point_stability(aname, save_dir, rules):
    """Post-analysis: classify each gradient fixed point as stable / marginal /
    unstable from the linear-stability spectrum already saved by
    solve_period_modulation_fixed_points, one pickle per task rule
    (fixed_points_grad_{aname}_{rule}.pkl). Two-task analog of the one-task
    classifier — no recomputation, just re-packaging the per-point spectral radius
    ρ = max|λ| and the marginal tolerance into an explicit 3-way class per period:
      unstable  ρ > 1 + tol      (an expanding direction)
      marginal  |ρ − 1| ≤ tol    (neutral direction — the ring-attractor signature)
      stable    ρ < 1 − tol      (all directions contracting)
    Only converged fixed points (is_fixed) are classified.

    Writes ONE combined fixed_point_classification_{aname}.pkl (keyed by rule) and
    a human-readable .csv (one row per rule × period × fixed point). Skips a rule
    whose grad-fp pickle is missing or predates the stability pass.
    """
    import csv as _csv
    CLASS_NAMES = ["stable", "marginal", "unstable"]

    def _classify(rad, tol):
        code = np.full(rad.shape, -1, dtype=int)   # -1 = unconverged / NaN
        finite = np.isfinite(rad)
        code[finite & (rad > 1.0 + tol)] = 2
        code[finite & (np.abs(rad - 1.0) <= tol)] = 1
        code[finite & (rad < 1.0 - tol)] = 0
        return code

    by_rule = {}
    csv_rows = []
    for rule in rules:
        fp_pkl = save_dir / f"fixed_points_grad_{aname}_{rule}.pkl"
        if not fp_pkl.exists():
            print(f"  [fp-classify/{rule}] {fp_pkl.name} not found; skipping.")
            continue
        with open(fp_pkl, "rb") as f:
            d = pickle.load(f)
        results = d.get("results", {})
        angles = np.asarray(d.get("angles", []), dtype=float)
        periods = list(results.keys())
        if not periods or any(results[v].get("spectral_radius") is None for v in periods):
            print(f"  [fp-classify/{rule}] stability spectrum not in pickle "
                  f"(re-run the stability pass); skipping.")
            continue

        per_period = {}
        for v in periods:
            e = results[v]
            rad = np.asarray(e["spectral_radius"], dtype=float)
            stim = np.asarray(e["stim"], dtype=int)
            tol = float(e.get("marginal_tol", 0.05))
            is_fixed = np.asarray(e.get("is_fixed", np.ones(rad.shape, bool)), dtype=bool)
            code = _classify(rad, tol)
            code[~is_fixed] = -1
            counts = {name: int(np.sum(code == i)) for i, name in enumerate(CLASS_NAMES)}
            counts["unconverged"] = int(np.sum(~is_fixed))
            n_conv = int(is_fixed.sum())
            per_period[v] = {
                "period_title": e.get("period_title", v),
                "stim": stim,
                "spectral_radius": rad,
                "class_code": code,
                "is_fixed": is_fixed,
                "marginal_tol": tol,
                "counts": counts,
                "n_converged": n_conv,
                "frac_stable": (counts["stable"] / n_conv) if n_conv else float("nan"),
            }
            print(f"  [fp-classify/{rule}] {v}: {counts['stable']} stable / "
                  f"{counts['marginal']} marginal / {counts['unstable']} unstable "
                  f"(+{counts['unconverged']} unconverged) of {rad.size}")
            for i in range(rad.size):
                ang = float(angles[stim[i]]) if angles.size and stim[i] < angles.size else float("nan")
                cc = code[i]
                cname = CLASS_NAMES[cc] if cc >= 0 else "unconverged"
                csv_rows.append({
                    "rule": rule,
                    "period": v,
                    "period_title": e.get("period_title", v),
                    "stim_index": int(stim[i]),
                    "stim_angle_rad": ang,
                    "spectral_radius": float(rad[i]) if np.isfinite(rad[i]) else "",
                    "n_unstable": int(np.asarray(e["n_unstable"])[i])
                                  if e.get("n_unstable") is not None else "",
                    "class": cname,
                })
        by_rule[rule] = {"angles": angles, "per_period": per_period}

    if not by_rule:
        print("  [fp-classify] no rule produced a classification; nothing saved.")
        return

    out_pkl = save_dir / f"fixed_point_classification_{aname}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"aname": aname, "class_names": CLASS_NAMES,
                     "rules": list(by_rule.keys()), "by_rule": by_rule}, f)
    print(f"  Saved fixed-point classification: {out_pkl}")

    out_csv = save_dir / f"fixed_point_classification_{aname}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=[
            "rule", "period", "period_title", "stim_index", "stim_angle_rad",
            "spectral_radius", "n_unstable", "class"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved fixed-point classification table: {out_csv}")


# ═══════════════════════════════════════════════════════════════════════════
# Network reload
# ═══════════════════════════════════════════════════════════════════════════
def _rebuild_net(net_params):
    if net_params['net_type'] == 'mpn1':
        netFunction = mpn.MultiPlasticNet
    elif net_params['net_type'] == 'dmpn':
        netFunction = mpn.DeepMultiPlasticNet
    elif net_params['net_type'] == 'vanilla':
        netFunction = nets.VanillaRNN
    elif net_params['net_type'] == 'gru':
        netFunction = nets.GRU
    else:
        raise ValueError(f"Unknown net_type {net_params['net_type']}")
    net = netFunction(net_params, verbose=False)
    return net


def main(aname, fp_n_seeds=5, interp_n_alpha=10, run_fixed_points=True):
    # two_task.py saves each trial in a self-contained subfolder twotasks/{aname}/.
    # Fall back to the flat layout (files directly under twotasks/) for older runs.
    run_dir = OUT_DIR / aname
    ckpt_path = run_dir / f"savednet_{aname}.pt"
    bundle_path = run_dir / f"bundle_{aname}.pkl"
    if not ckpt_path.exists() or not bundle_path.exists():
        flat_ckpt = OUT_DIR / f"savednet_{aname}.pt"
        flat_bundle = OUT_DIR / f"bundle_{aname}.pkl"
        if flat_ckpt.exists() and flat_bundle.exists():
            ckpt_path, bundle_path = flat_ckpt, flat_bundle
        else:
            raise FileNotFoundError(f"Missing checkpoint/bundle for {aname}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device}")

    # ── Reload checkpoint & rebuild the live network ─────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_params = ckpt["net_params"]
    task_params = ckpt["task_params"]
    train_params = ckpt["train_params"]
    hyp_dict = ckpt["hyp_dict"]

    net = _rebuild_net(net_params)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()

    # ── Reload training bundle + test datasets ───────────────────────────────
    with open(bundle_path, "rb") as f:
        b = pickle.load(f)

    seed = b["seed"]
    shift_index = b["shift_index"]
    color_by = b["color_by"]
    n_hidden = net_params["n_neurons"][1]

    counter_lst = b["counter_lst"]
    netout_lst = b["netout_lst"]
    db_lst = b["db_lst"]
    Winput_lst = b["Winput_lst"]
    Winputbias_lst = b["Winputbias_lst"]
    Woutput_lst = b["Woutput_lst"]
    Wall_lst = b["Wall_lst"]
    marker_lst = b["marker_lst"]
    loss_lst = b["loss_lst"]
    acc_lst = b["acc_lst"]

    def _t(np_arr):
        return torch.as_tensor(np_arr, dtype=torch.float, device=device)

    test_input_np = b["test_input_np"]
    test_output_np = b["test_output_np"]
    test_input = _t(test_input_np)
    test_output = _t(test_output_np)
    test_mask = _t(b["test_mask_np"])

    test_input_longfixation = _t(b["test_input_longfixation_np"])
    test_output_longfixation = _t(b["test_output_longfixation_np"])
    test_input_longstimulus = _t(b["test_input_longstimulus_np"])
    test_output_longstimulus = _t(b["test_output_longstimulus_np"])
    test_input_longdelay = _t(b["test_input_longdelay_np"])
    test_output_longdelay = _t(b["test_output_longdelay_np"])
    test_input_longresponse = _t(b["test_input_longresponse_np"])
    test_output_longresponse = _t(b["test_output_longresponse_np"])

    labels = b["labels"]
    labels_stim = b["labels_stim"]
    labels_resp = b["labels_resp"]
    test_task = b["test_task"]
    test_task_longfixation = b["test_task_longfixation"]
    test_task_longstimulus = b["test_task_longstimulus"]
    test_task_longdelay = b["test_task_longdelay"]
    test_task_longresponse = b["test_task_longresponse"]
    label_task_comb = b["label_task_comb"]
    label_task_comb_longfixation = b["label_task_comb_longfixation"]
    label_task_comb_longstimulus = b["label_task_comb_longstimulus"]
    label_task_comb_longdelay = b["label_task_comb_longdelay"]
    label_task_comb_longresponse = b["label_task_comb_longresponse"]

    n_batch_all = test_input_np.shape[0]

    # Figures and analysis outputs live in the same per-run subfolder as the
    # training INPUTS (checkpoint / bundle / param). Clear previously-generated
    # analysis outputs — figures (.png / .html) AND analysis pickles (.pkl,
    # e.g. d_combine_*, m_pca_*, cancel_*, pc_cumvar_*, fixed_points_grad_*) — so
    # re-running is clean. Preserve the training inputs (the .pt checkpoint, the
    # param json, and the bundle_{aname}.pkl produced by two_task.py) by name.
    # Skip the wipe entirely on a --no-fixed-points run: the slow fixed-point
    # pickles are not regenerated, so leave the folder intact to preserve any
    # fixed_points_grad_* / interp_fixed_points_* from an earlier full run
    # (paper_plot still reads them).
    save_dir = OUT_DIR / aname
    save_dir.mkdir(parents=True, exist_ok=True)
    _preserve = {f"savednet_{aname}.pt", f"bundle_{aname}.pkl",
                 f"param_{aname}_param.json"}
    if run_fixed_points:
        for _old in save_dir.iterdir():
            if not _old.is_file() or _old.name in _preserve:
                continue
            if _old.suffix in (".png", ".html", ".pkl"):
                _old.unlink()

    def fp(stem):
        """Figure path under the per-run directory."""
        return str(save_dir / stem)

    # ── Final-stage network output / db (notebook cell 12) ───────────────────
    ind = -1
    net_out = netout_lst[0][ind]
    db = db_lst[0][ind]

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 13: plot_input_output  +  cells 14-18 (normal + 4 long variants)
    # ═════════════════════════════════════════════════════════════════════════
    def plot_input_output(test_input_np_, net_out_, test_output_np_, test_task_=None, tag="", batch_num=5, label=None):
        test_input_np_ = helper.to_ndarray(test_input_np_)
        net_out_ = helper.to_ndarray(net_out_)
        test_output_np_ = helper.to_ndarray(test_output_np_)

        # Trials are blocked by task (all delaygo, then all delayanti), so the
        # first batch_num rows would all be the same task. Instead pick rows that
        # alternate between the tasks so both delaygo and delayanti are visible —
        # this makes it easy to eyeball whether their periods are aligned.
        if test_task_ is not None:
            tt = np.asarray(test_task_)
            per_task = {int(t): [k for k in range(len(tt)) if int(tt[k]) == int(t)]
                        for t in np.unique(tt)}
            order = []
            ptr = {t: 0 for t in per_task}
            tasks_cyc = sorted(per_task)
            while len(order) < min(batch_num, len(tt)):
                progressed = False
                for t in tasks_cyc:
                    if ptr[t] < len(per_task[t]):
                        order.append(per_task[t][ptr[t]]); ptr[t] += 1
                        progressed = True
                        if len(order) >= min(batch_num, len(tt)):
                            break
                if not progressed:
                    break
            row_idxs = order
        else:
            row_idxs = list(range(min(batch_num, net_out_.shape[0])))

        fig_all, axs_all = plt.subplots(batch_num, 2, figsize=(4 * 2, batch_num * 2))

        if test_output_np_.shape[-1] == 1:
            for row, ax in enumerate(axs_all):
                batch_idx = row_idxs[row]
                ax.plot(net_out_[batch_idx, :, 0], color=c_vals[row])
                ax.plot(test_output_np_[batch_idx, :, 0], color=c_vals_l[row])
        else:
            for row in range(batch_num):
                batch_idx = row_idxs[row]
                for out_idx in range(test_output_np_.shape[-1]):
                    axs_all[row, 0].plot(net_out_[batch_idx, :, out_idx], color=c_vals[out_idx], label=out_idx)
                    axs_all[row, 0].plot(test_output_np_[batch_idx, :, out_idx], color=c_vals_l[out_idx], linewidth=5, alpha=0.5)
                    if test_task_ is not None:
                        outname = f"{task_params['rules'][test_task_[batch_idx]]}; {tag}"
                        axs_all[row, 0].set_title(outname)
                    axs_all[row, 0].legend()

                input_batch = test_input_np_[batch_idx, :, :]
                if task_params["randomize_inputs"]:
                    input_batch = input_batch @ np.linalg.pinv(task_params["randomize_matrix"])
                for inp_idx in range(input_batch.shape[-1]):
                    axs_all[row, 1].plot(input_batch[:, inp_idx], color=c_vals[inp_idx], label=inp_idx)
                    if test_task_ is not None:
                        axs_all[row, 1].set_title(f"{task_params['rules'][test_task_[batch_idx]]}; {tag}")
                    axs_all[row, 1].legend()

        for ax in axs_all.flatten():
            ax.set_ylim([-2, 2])
        fig_all.tight_layout()
        fig_all.savefig(fp(f"lowD_{hyp_dict['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}_{tag}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"lowD_{hyp_dict['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}_{tag}.png")))
        return fig_all, axs_all

    f14, _ = plot_input_output(test_input_np, net_out, test_output_np, test_task, tag="")
    plt.close(f14)
    f15, _ = plot_input_output(test_input_longdelay, netout_lst[3][ind], test_output_longdelay.detach().cpu().numpy(), test_task_longdelay, tag="longdelay")
    plt.close(f15)
    f16, _ = plot_input_output(test_input_longresponse, netout_lst[4][ind], test_output_longresponse.detach().cpu().numpy(), test_task_longresponse, tag="longresponse")
    plt.close(f16)
    f17, _ = plot_input_output(test_input_longstimulus, netout_lst[2][ind], test_output_longstimulus.detach().cpu().numpy(), test_task_longstimulus, tag="longstimulus")
    plt.close(f17)
    f18, _ = plot_input_output(test_input_longfixation, netout_lst[1][ind], test_output_longfixation.detach().cpu().numpy(), test_task_longfixation, tag="longfixation")
    plt.close(f18)

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 21: layer index
    # ═════════════════════════════════════════════════════════════════════════
    layer_index = 0
    if net_params["input_layer_add"]:
        layer_index += 1

    print(f"shift_index: {shift_index}")  # cell 31

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 27: attractor / cosine-similarity over learning + time stamps
    # ═════════════════════════════════════════════════════════════════════════
    result_attractor_all_h, result_attractor_all_m, result_attractor_all_wm, result_attractor_all_wmwin = [], [], [], []
    modulation_save_time = []
    pr_all = []
    test_input_long_all = [test_input, test_input_longfixation, test_input_longstimulus,
                           test_input_longdelay, test_input_longresponse]
    label_task_comb_long_all = [label_task_comb, label_task_comb_longfixation, label_task_comb_longstimulus,
                                label_task_comb_longdelay, label_task_comb_longresponse]
    time_stamps_usual, time_stamps_longfixation, time_stamps_longstimulus, time_stamps, time_stamps_longresponse = {}, {}, {}, {}, {}

    # ── Guard: the analysis uses a SINGLE shared timeline (read from one batch)
    # for both tasks, so it is only valid if the two tasks' periods are matched.
    # Verify that the fixation-drop timestep (response onset) is identical for a
    # task-0 trial and a task-1 trial in each variant; raise if not, since that
    # means the bundle was generated without period alignment (two_task.py
    # task_random_fix / align_periods) and cross-task comparisons would be made
    # at mismatched times.
    def _fix_drop_t(inp_long, trial_idx):
        ch0 = inp_long[trial_idx, :, 0].detach().cpu()
        nz = torch.nonzero(ch0 < 0.5, as_tuple=False)
        return int(nz[0].item()) if nz.numel() else -1

    def _assert_periods_matched(inp_long, ltc, name):
        ltc = np.asarray(ltc)
        t0 = [k for k, lst in enumerate(ltc) if int(lst[1]) == 0]
        t1 = [k for k, lst in enumerate(ltc) if int(lst[1]) == 1]
        if not t0 or not t1:
            return  # only one task present; nothing to compare
        d0, d1 = _fix_drop_t(inp_long, t0[0]), _fix_drop_t(inp_long, t1[0])
        assert d0 == d1, (
            f"[{name}] period mismatch across tasks: task0 response onset t={d0}, "
            f"task1 t={d1}. The bundle was generated without aligned periods; "
            f"regenerate with two_task.py (task_random_fix/align_periods=True)."
        )

    variant_names = ["normal", "longfixation", "longstimulus", "longdelay", "longresponse"]
    for _i in range(5):
        _assert_periods_matched(test_input_long_all[_i], label_task_comb_long_all[_i], variant_names[_i])

    cc = None
    for i in range(5):
        for db_attractor in db_lst[i]:
            _, M_long, h_long, _ = modulation_extraction(test_input_long_all[i], db_attractor, layer_index)

            prs = [dimensionality_measure(h_long[ii, :, :].T, n_hidden) for ii in range(h_long.shape[0])]
            if i == 0:
                pr_all.append([np.mean(prs), np.std(prs)])

            checktime_sample = test_input_long_all[i][0, :, 0].detach().cpu()
            mask = checktime_sample < 0.5
            idx = torch.nonzero(mask, as_tuple=False)
            checktime_attractor = idx[0].item()

            if i == 3:
                time_stamps["delay_end"] = checktime_attractor - 1
            elif i == 4:
                time_stamps_longresponse["delay_end"] = checktime_attractor - 1
            elif i == 0:
                time_stamps_usual["delay_end"] = checktime_attractor - 1
                cc = time_stamps_usual["delay_end"]
            elif i == 2:
                time_stamps_longstimulus["delay_end"] = checktime_attractor - 1
            elif i == 1:
                time_stamps_longfixation["delay_end"] = checktime_attractor - 1

            if i == 0:
                result_attractor_h, _, _ = analyze_similarity(M_long, h_long, net, net_params, label_task_comb_long_all[i], checktime=cc, compare="hidden")
                result_attractor_m, m_save, _ = analyze_similarity(M_long, h_long, net, net_params, label_task_comb_long_all[i], checktime=cc, compare="modulation", moddim=0)
                result_attractor_wm, _, _ = analyze_similarity(M_long, h_long, net, net_params, label_task_comb_long_all[i], checktime=cc, compare="w_modulation", moddim=None)
                result_attractor_wmwin, _, _ = analyze_similarity(M_long, h_long, net, net_params, label_task_comb_long_all[i], checktime=cc, compare="w_modulation", moddim="Win")

                result_attractor_all_h.append(result_attractor_h)
                result_attractor_all_m.append(result_attractor_m)
                result_attractor_all_wm.append(result_attractor_wm)
                result_attractor_all_wmwin.append(result_attractor_wmwin)
                modulation_save_time.append(m_save)

    # ── Cell 29: attractor-over-learning figure ──────────────────────────────
    figattractor, axsattractor = plt.subplots(1, 4, figsize=(4 * 4, 4))
    break_names = ["Same Stim", "Same Resp", "MemoryPro Diff Stim", "MemoryAnti Diff Stim"]

    def plot_mean_std(ax, x, mean, std, color, fill_color, label):
        ax.plot(x, mean, "-o", color=color, label=label)
        ax.fill_between(x, np.asarray(mean) - np.asarray(std), np.asarray(mean) + np.asarray(std), alpha=0.5, color=fill_color)

    panels = [(result_attractor_all_h, 0), (result_attractor_all_m, 1),
              (result_attractor_all_wm, 2), (result_attractor_all_wmwin, 3)]
    n_groups = len(result_attractor_all_h[0])
    for i in range(n_groups):
        for results, ax_idx in panels:
            mean = [rs[i][0] for rs in results]
            std = [rs[i][1] for rs in results]
            plot_mean_std(axsattractor[ax_idx], counter_lst, mean, std,
                          color=c_vals[i], fill_color=c_vals_l[i], label=break_names[i])
    for ax in axsattractor:
        ax.set_xscale("log")
        ax.legend()
        ax.set_ylabel("Cosine Similarity", fontsize=12)
        ax.set_xlabel("Iteration", fontsize=12)
    axsattractor[0].set_ylim([0, 1.05])
    axsattractor[1].set_ylim([0, 1.05])
    axsattractor[2].set_ylim([-1.05, 1.05])
    axsattractor[3].set_ylim([-1.05, 1.05])
    axsattractor[0].set_title("Hidden", fontsize=12)
    axsattractor[1].set_title("Modulation (Fix On Component)", fontsize=12)
    axsattractor[2].set_title(r"$W \odot \mathrm{Modulation}$", fontsize=12)
    axsattractor[3].set_title(r"$W_{\mathrm{in}} @ (W \odot \mathrm{Modulation})$", fontsize=12)
    figattractor.tight_layout()
    figattractor.savefig(fp(f"attractor_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"attractor_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(figattractor)

    # Cell 30: package learning_hm_similarity (saved later in cell 48)
    learning_hm_similarity = {
        "break_names": break_names,
        "counter_lst": counter_lst,
        "result_attractor_all_h": result_attractor_all_h,
        "result_attractor_all_m": result_attractor_all_m,
        "result_attractor_all_wmwin": result_attractor_all_wmwin,
    }

    # Save the FIRST subplot ("Hidden" panel) of attractor_*.png so paper_plot can
    # replot it: per break-name group, cosine-similarity mean/std vs iteration.
    attractor_hidden_first = {
        "break_names": list(break_names),
        "counter_lst": [float(c) for c in counter_lst],
        # per group: mean and std over training stages
        "mean": [[float(rs[i][0]) for rs in result_attractor_all_h]
                 for i in range(len(result_attractor_all_h[0]))],
        "std": [[float(rs[i][1]) for rs in result_attractor_all_h]
                for i in range(len(result_attractor_all_h[0]))],
        "ylabel": "Cosine Similarity", "xlabel": "Iteration", "title": "Hidden",
    }

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 32: full time-stamp extraction across variants
    # ═════════════════════════════════════════════════════════════════════════
    def time_stamp_extract(test_input_long, ts):
        stimulus_end = None
        chosen_batch = 0
        while stimulus_end is None:
            try:
                input_part = test_input_long[chosen_batch, :, 2 - shift_index:2 + 4 - shift_index].detach().cpu().numpy()
                input_part_sum = np.sum(input_part, axis=1)
                stimulus_end = np.where(input_part_sum > 0.5)[0][-1]
                stimulus_start = np.where(input_part_sum > 0.5)[0][0] - 1
            except IndexError:
                chosen_batch += 1
        ts["stimulus_start"] = stimulus_start + 1
        ts["stimulus_end"] = stimulus_end + 1
        ts["delay_start"] = stimulus_end + 1
        ts["trial_end"] = len(input_part_sum) - 1
        ts["fixation_end"] = stimulus_start
        ts["fixation_start"] = 0
        return ts

    time_stamps = time_stamp_extract(test_input_longdelay, time_stamps)
    time_stamps_longresponse = time_stamp_extract(test_input_longresponse, time_stamps_longresponse)
    time_stamps_usual = time_stamp_extract(test_input, time_stamps_usual)
    time_stamps_longstimulus = time_stamp_extract(test_input_longstimulus, time_stamps_longstimulus)
    time_stamps_longfixation = time_stamp_extract(test_input_longfixation, time_stamps_longfixation)
    print(f"time_stamps_usual: {time_stamps_usual}")

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 36 + 37 + 38 + 39: dPCA (optional dependency)
    # Run for BOTH the hidden activity and the effective modulation (W⊙M), so the
    # stimulus/task/time demixing can be compared across representations. Files:
    #   dpca_hidden_*        / dpca_hidden_var_*        / dpca_hidden_Pangle_*
    #   dpca_w_modulation_*  / dpca_w_modulation_var_*  / dpca_w_modulation_Pangle_*
    # ═════════════════════════════════════════════════════════════════════════
    try:
        from dPCA import dPCA

        # Trial index (mod1_j, mod2_j) per stimulus — matched via the saved
        # modulation snapshots. Representation-independent, so compute once.
        M_all = db_lst[0][-1][f"M{layer_index}"]
        all_hidden = db_lst[0][-1][f"hidden{layer_index}"]
        if net_params["input_layer_add"]:
            W_dpca = net.mp_layer1.W.data.detach().cpu().numpy()
        else:
            W_dpca = net.mp_layer0.W.data.detach().cpu().numpy()

        stim_trial_idx = []
        for i in range(8):
            mod1_stim1 = m_save[0][i]
            mod2_stim1 = m_save[1][i]
            mod1_j = mod2_j = None
            for j in range(M_all.shape[0]):
                if np.sum(np.abs(M_all[j, :, :, :] - mod1_stim1)) <= 1e-3:
                    mod1_j = j
                if np.sum(np.abs(M_all[j, :, :, :] - mod2_stim1)) <= 1e-3:
                    mod2_j = j
            stim_trial_idx.append((mod1_j, mod2_j))

        def _dpca_activity(dpca_name, j):
            """(N, k) activity for trial index j under representation `dpca_name`:
            hidden units, or the flattened effective modulation W⊙M per timestep."""
            if dpca_name == "hidden":
                return all_hidden[j].T                       # (N_neurons, k)
            # w_modulation: effective modulation W⊙M, (post,pre) flattened per step.
            wm = M_all[j] * W_dpca[None, :, :]               # (k, post, pre)
            return wm.reshape(wm.shape[0], -1).T             # (post*pre, k)

        # dPCA forms an N x N covariance internally, so the raw effective
        # modulation (N = post*pre, e.g. 40000) is infeasible. When the feature
        # dimension exceeds this cap, pre-reduce the activity with PCA and run
        # dPCA in that subspace (standard dPCA-on-large-populations practice).
        DPCA_MAX_FEATURES = 200

        # Self-contained data for paper_plot to replot the dPCA figures without
        # re-running dPCA, keyed by representation ("hidden" / "w_modulation").
        dpca_data = {}

        def _run_dpca(dpca_name):
            activity_list = [[_dpca_activity(dpca_name, j1), _dpca_activity(dpca_name, j2)]
                             for (j1, j2) in stim_trial_idx]

            N, k = activity_list[0][0].shape
            S = 8
            T = 2
            data_mean = np.zeros((N, S, T, k))
            for s in range(S):
                for t in range(T):
                    data_mean[:, s, t, :] = activity_list[s][t]

            # Pre-PCA for high-dimensional representations (w_modulation): fit on
            # all (S*T*k) samples in feature space, then project so the dPCA input
            # has at most DPCA_MAX_FEATURES "pseudo-neurons".
            if N > DPCA_MAX_FEATURES:
                flat = data_mean.reshape(N, -1).T           # (S*T*k, N)
                n_comp = min(DPCA_MAX_FEATURES, flat.shape[0])
                pca_pre = PCA(n_components=n_comp, random_state=42)
                reduced = pca_pre.fit_transform(flat).T     # (n_comp, S*T*k)
                data_mean = reduced.reshape(n_comp, S, T, k)
                print(f"  [dPCA/{dpca_name}] pre-PCA reduced {N} -> {n_comp} "
                      f"features ({pca_pre.explained_variance_ratio_.sum():.1%} var).")
                N = n_comp
            data_trials = data_mean[None, ...]

            dpca = dPCA.dPCA(labels='srt', n_components=min(100, N))
            dpca.protect = ['t']
            Z = dpca.fit_transform(data_mean, data_trials)

            figd, axsd = plt.subplots(len(Z.keys()), 1, figsize=(8, 4 * len(Z.keys())))
            if len(Z.keys()) == 1:
                axsd = [axsd]
            for idx, key in enumerate(Z.keys()):
                ax = axsd[idx]
                time = [ii for ii in range(time_stamps_usual['trial_end'] + 1)]
                for s in range(S):
                    ax.plot(time, Z[key][0, s, 0, :], color=c_vals[s], linestyle='-', alpha=0.8, label=f"Stimulus {s}")
                    ax.plot(time, Z[key][0, s, 1, :], color=c_vals[s], linestyle='--', alpha=0.8)
                ax.set_title(f"Top Stimulus Component ({key})")
                ax.grid(alpha=0.3)
                ax.legend()
                ax.axvline(time_stamps_usual["fixation_end"], linestyle="--", color="gray")
                ax.axvline(time_stamps_usual["stimulus_end"], linestyle="--", color="gray")
                ax.axvline(time_stamps_usual["delay_end"], linestyle="--", color="gray")
            figd.tight_layout()
            figd.savefig(fp(f"dpca_{dpca_name}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
            print("  Saved figure: " + str(fp(f"dpca_{dpca_name}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
            plt.close(figd)

            # Cell 38: variance per marginalization
            exp_var = dpca.explained_variance_ratio_
            keys_v = list(exp_var.keys())
            vals = [np.sum(exp_var[kk]) * 100 for kk in keys_v]
            order = np.argsort(vals)[::-1]
            keys_sorted = [keys_v[ii] for ii in order]
            vals_sorted = [vals[ii] for ii in order]
            figv, axv = plt.subplots(1, 1, figsize=(7, 4.5))
            bars = axv.bar(keys_sorted, vals_sorted)
            axv.set_ylabel("Explained variance (%)", fontsize=12)
            axv.set_xlabel("Marginalization", fontsize=12)
            axv.set_title("dPCA Variance Explained by Marginalization")
            for bb, v in zip(bars, vals_sorted):
                axv.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.5, f"{v:.2f}%", ha="center", va="bottom", fontsize=12)
            axv.set_ylim(0, max(vals_sorted) * 1.15 if len(vals_sorted) else 1)
            figv.tight_layout()
            figv.savefig(fp(f"dpca_{dpca_name}_var_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
            print("  Saved figure: " + str(fp(f"dpca_{dpca_name}_var_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
            plt.close(figv)

            # Cell 39: P-angle heatmap between marginalization axes
            P_angle = np.full((len(keys_v), len(keys_v)), np.nan)
            for i in range(len(keys_v)):
                for j in range(i + 1, len(keys_v)):
                    u = dpca.P[keys_v[i]][:, 0]
                    v = dpca.P[keys_v[j]][:, 0]
                    P_angle[i, j] = vec_angle_deg(u, v, sign_invariant=True)
            figpa, axpa = plt.subplots(1, 1, figsize=(4, 4))
            sns.heatmap(P_angle, ax=axpa, cmap="coolwarm", square=True, linewidths=0.5, linecolor="white",
                        cbar_kws={"shrink": 0.85, "label": "P_angle"}, xticklabels=keys_v, yticklabels=keys_v)
            axpa.set_xticklabels(keys_v, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
            axpa.set_yticklabels(keys_v, rotation=0, fontsize=10)
            figpa.tight_layout()
            figpa.savefig(fp(f"dpca_{dpca_name}_Pangle_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
            print("  Saved figure: " + str(fp(f"dpca_{dpca_name}_Pangle_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
            plt.close(figpa)

            # Stash everything paper_plot needs to replot all three dPCA figures.
            # Z_components[key] is the demixed top component per marginalization,
            # shape (S, T, k) — the (0, s, t, :) traces the component figure draws.
            dpca_data[dpca_name] = {
                "n_stim": int(S), "n_task": int(T),
                "time": np.arange(time_stamps_usual["trial_end"] + 1),
                "period_bounds": {kk: int(time_stamps_usual[kk]) for kk in
                                  ("fixation_end", "stimulus_end", "delay_end")},
                # component traces per marginalization key.
                "Z_components": {key: np.asarray(Z[key][0], dtype=np.float32)
                                 for key in Z.keys()},
                # explained variance (%) per marginalization (component figure order).
                "exp_var_keys": list(keys_v),
                "exp_var_pct": {kk: float(np.sum(exp_var[kk]) * 100) for kk in keys_v},
                # P-angle heatmap (deg) between the top encoder axes + its labels.
                "P_angle": np.asarray(P_angle, dtype=float),
                "P_angle_keys": list(keys_v),
                # feature dim actually fed to dPCA (post-pre-PCA for w_modulation).
                "n_features": int(N),
            }

        for _dpca_name in ("hidden", "w_modulation"):
            _run_dpca(_dpca_name)

        if dpca_data:
            dpca_path = save_dir / f"dpca_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.pkl"
            with open(dpca_path, "wb") as f:
                pickle.dump(dpca_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("  Saved data: " + str(dpca_path))
    except ImportError:
        print("  [warn] dPCA not installed; skipping dPCA figures (cells 36-39).")
    except Exception as e:
        print(f"  [warn] dPCA analysis failed; skipping (cells 36-39): {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 40: fixon/task cancellation projection (per stimulus) + projs
    # ═════════════════════════════════════════════════════════════════════════
    def trainable_parameters(model):
        """Return the names of trainable parameters (notebook cell 40)."""
        return [name for name, p in model.named_parameters() if p.requires_grad]

    print(trainable_parameters(net))

    m_save = modulation_save_time[-1]
    projs_all = [[], [], []]

    # self-contained data to replot selected stimuli of the cancel figure.
    # Match the one-task onetask_show selection (ONETASK_SHOW_STIM = [5, 2]) so
    # the single- and two-task cancellation figures show the same stimuli.
    cancel_data = {}
    cancel_save_stimuli = [5, 2]

    fig40, axs40 = plt.subplots(8, 2, figsize=(4 * 2, 8 * 2))
    for i in range(8):
        mod1_stim1 = m_save[0][i]
        mod2_stim1 = m_save[1][i]
        M_all = db_lst[0][-1][f"M{layer_index}"]
        mod1_j = mod2_j = None
        for j in range(M_all.shape[0]):
            if np.sum(np.abs(M_all[j, :, :, :] - mod1_stim1)) <= 1e-3:
                mod1_j = j
            if np.sum(np.abs(M_all[j, :, :, :] - mod2_stim1)) <= 1e-3:
                mod2_j = j

        all_input = db_lst[0][-1][f"input{layer_index}"]
        input_orig = test_input_np
        shrink = input_change(input_orig, all_input)

        if net_params["input_layer_add"]:
            W = net.mp_layer1.W.data.detach().cpu().numpy()
            W_out = net.W_output.data.detach().cpu().numpy()
            W_in = Winput_lst[-1]
            bias = net.mp_layer1.b.data.detach().cpu().numpy()

            if task_params["fixate_off"]:
                x_fix_on = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
                x_fix_off = np.array([0, 1, 0, 0, 0, 0, 0, 0])
                null = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
                x_task1 = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=float)
                x_task2 = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
            else:
                x_fix_on = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
                x_fix_off = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
                null = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
                x_task1 = np.array([0, 0, 0, 0, 0, 1, 0], dtype=float)
                x_task2 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)

            x_task1_all, x_task2_all = [], []
            x_fix_on_all, x_fix_off_all = [], []
            for Tt in range(mod1_stim1.shape[0]):
                if Tt <= time_stamps_usual["delay_end"]:
                    x_task1_all.append(x_task1)
                    x_task2_all.append(x_task2)
                    x_fix_on_all.append(x_fix_on)
                    x_fix_off_all.append(null)
                else:
                    x_task1_all.append(x_task1 + x_fix_off)
                    x_task2_all.append(x_task2 + x_fix_off)
                    x_fix_on_all.append(null)
                    x_fix_off_all.append(x_fix_off)

            Y_resp_cos, Y_resp_sin = W_out[1, :].reshape(1, -1), W_out[2, :].reshape(1, -1)

            fixon_proj1 = np.stack([Y_resp_cos @ (W + W * mod1_stim1[Tt]) @ (W_in @ x_fix_on_all[Tt]) for Tt in range(mod1_stim1.shape[0])], axis=0)
            fixon_proj2 = np.stack([Y_resp_cos @ (W + W * mod2_stim1[Tt]) @ (W_in @ x_fix_on_all[Tt]) for Tt in range(mod1_stim1.shape[0])], axis=0)
            x_task1_proj = np.stack([Y_resp_cos @ (W + W * mod1_stim1[Tt]) @ (W_in @ x_task1_all[Tt]) for Tt in range(mod1_stim1.shape[0])], axis=0)
            x_task2_proj = np.stack([Y_resp_cos @ (W + W * mod2_stim1[Tt]) @ (W_in @ x_task2_all[Tt]) for Tt in range(mod1_stim1.shape[0])], axis=0)
            fixoff_proj1 = np.stack([Y_resp_cos @ (W + W * mod1_stim1[Tt]) @ (W_in @ x_fix_off_all[Tt]) for Tt in range(mod1_stim1.shape[0])], axis=0)
            fixoff_proj2 = np.stack([Y_resp_cos @ (W + W * mod2_stim1[Tt]) @ (W_in @ x_fix_off_all[Tt]) for Tt in range(mod1_stim1.shape[0])], axis=0)

            bias_proj = Y_resp_cos @ bias

        axs40[i, 0].plot(fixon_proj1 + x_task1_proj + bias_proj, color=c_vals[0], label="Combine", linewidth=3, alpha=0.5)
        axs40[i, 0].plot(fixon_proj1, color=c_vals[1], label="Fix On")
        if task_params["fixate_off"]:
            axs40[i, 0].plot(fixoff_proj1, color=c_vals[3], label="Fixoff")
        axs40[i, 0].plot(x_task1_proj + bias_proj, color=c_vals[2], label="Task + Bias")
        axs40[i, 1].plot(fixon_proj2 + x_task2_proj + bias_proj, color=c_vals[0], label="Combine", linewidth=3, alpha=0.5)
        axs40[i, 1].plot(fixon_proj2, color=c_vals[1], label="Fix On")
        if task_params["fixate_off"]:
            axs40[i, 1].plot(fixoff_proj2, color=c_vals[3], label="Fixoff")
        axs40[i, 1].plot(x_task2_proj + bias_proj, color=c_vals[2], label="Task + Bias")

        # Stash the per-time-step projection traces for selected stimuli so
        # paper_plot can redraw both task columns (col 0 = task1/pro, col 1 =
        # task2/anti). Each "combine" = fixon + task + bias.
        if i in cancel_save_stimuli:
            cancel_data[i] = {
                "fixon_proj1": np.asarray(fixon_proj1).ravel(),
                "fixon_proj2": np.asarray(fixon_proj2).ravel(),
                "x_task1_proj": np.asarray(x_task1_proj).ravel(),
                "x_task2_proj": np.asarray(x_task2_proj).ravel(),
                "fixoff_proj1": np.asarray(fixoff_proj1).ravel(),
                "fixoff_proj2": np.asarray(fixoff_proj2).ravel(),
                "bias_proj": float(np.asarray(bias_proj).ravel()[0]),
                "fixate_off": bool(task_params["fixate_off"]),
            }

        # Average the fixon-probed memory state over the SECOND HALF of the
        # delay period (delay midpoint .. delay end), rather than a single
        # end-of-delay frame, so the projection estimate is more stable.
        delay_start = time_stamps_usual["delay_start"]
        delay_end = time_stamps_usual["delay_end"]
        half_start = (delay_start + delay_end) // 2
        delay_window = range(half_start, delay_end)
        h1 = np.mean([(W + W * mod1_stim1[Tt]) @ (W_in @ x_fix_on_all[Tt])
                      for Tt in delay_window], axis=0)
        h2 = np.mean([(W + W * mod2_stim1[Tt]) @ (W_in @ x_fix_on_all[Tt])
                      for Tt in delay_window], axis=0)

        y = Y_resp_cos.reshape(-1)
        y = y / (np.linalg.norm(y) + 1e-12)
        fixon_p1 = float(y @ h1.reshape(-1))
        fixon_p2 = float(y @ h2.reshape(-1))
        projs_all[0].append([np.abs(fixon_p1 + fixon_p2), np.abs(fixon_p1), np.abs(fixon_p2)])

        P_perp = np.eye(y.size) - np.outer(y, y)

        def proj_perp_norm(h_vec):
            h_vec = h_vec.reshape(-1)
            return np.linalg.norm(P_perp @ h_vec)

        perp_ctrl = proj_perp_norm(h1) + proj_perp_norm(h2)
        projs_all[1].append([np.abs(perp_ctrl), np.abs(proj_perp_norm(h1)), np.abs(proj_perp_norm(h2))])

        def random_proj(h_vec):
            proj_mag_all = []
            for _ in range(100):
                r = np.random.randn(h_vec.size)
                proj_mag = abs(h_vec @ r) / np.linalg.norm(r)
                proj_mag_all.append(proj_mag)
            return np.mean(proj_mag_all)

        random_ctrl = random_proj(h1) + random_proj(h2)
        projs_all[2].append([np.abs(random_ctrl), np.abs(random_proj(h1)), np.abs(random_proj(h2))])

    for i in range(8):
        for j in range(2):
            axs40[i, j].axvline(time_stamps_usual["fixation_end"], linestyle="--", c=c_vals[-1])
            axs40[i, j].axvline(time_stamps_usual["stimulus_end"], linestyle="--", c=c_vals[-1])
            axs40[i, j].axvline(time_stamps_usual["delay_end"], linestyle="--", c=c_vals[-1])
            axs40[i, j].set_ylim([-1.5, 1.5])
            axs40[i, j].set_xlabel("Timestep", fontsize=12)
            axs40[i, j].set_ylabel("Proj Cos Mag", fontsize=12)
            axs40[i, j].set_title(f"Stimulus {i}", fontsize=12)
            if i == 2 and j == 0:
                axs40[i, j].legend(loc="upper left", frameon=True)
    fig40.tight_layout()
    fig40.savefig(fp(f"cancel_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"cancel_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig40)

    # Save the cancel projection traces (selected stimuli) next to the figures
    # so paper_plot can replot them. Keyed by stimulus index (int).
    if cancel_data:
        cancel_save = {
            "stimuli": cancel_data,
            "markers": {k: time_stamps_usual[k]
                        for k in ("fixation_end", "stimulus_end", "delay_end")},
        }
        cancel_path = save_dir / f"cancel_seed{seed}_{hyp_dict['addon_name']}.pkl"
        with open(cancel_path, "wb") as f:
            pickle.dump(cancel_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  Saved data: " + str(cancel_path))

    # ── Modulation-computation magnitude across time, per input component ─────
    # L2 magnitude (over hidden units) of the modulation M applies to each input
    # channel, M @ W_input[:, c], averaged over trials (±std). Four curves:
    # Fixation, combined Stimulus, Task cue 1, Task cue 2. Uses the final-stage
    # normal-trial modulation and the trained input embedding.
    try:
        _, M_mag_all, _, _ = modulation_extraction(test_input, db_lst[0][-1], layer_index)
        if net_params["input_layer_add"] and len(Winput_lst) and Winput_lst[-1] is not None:
            W_input_mag = np.asarray(Winput_lst[-1])           # (embed, n_raw)
        else:
            W_input_mag = np.eye(M_mag_all.shape[-1])
        dt_mag = int(task_params.get("dt", 40))                # sim step in ms
        modulation_magnitude_by_component(
            save_dir, aname, M_mag_all, W_input_mag, shift_index,
            task_params.get("fixate_off", False), dt_mag,
            time_stamps_usual.get("fixation_end"),
            time_stamps_usual.get("stimulus_end"),
            time_stamps_usual.get("delay_end"))
    except Exception as exc:
        print(f"  [mod-magnitude] failed: {exc}")
        import traceback
        traceback.print_exc()

    # Cell 41: cancellation magnitude scatter
    fig41, axs41 = plt.subplots(1, 2, figsize=(4 * 2, 4))
    for i in range(len(projs_all)):
        for k in range(8):
            axs41[0].scatter(i, projs_all[i][k][0], color=c_vals[i])
            axs41[1].scatter(i, projs_all[i][k][1], color=c_vals[i])
    for ax in axs41:
        ax.set_xticks([i for i in range(len(projs_all))])
        ax.set_xticklabels(["Projection to Cosine Output", "Orthogonal Complement", "Random Vector"], rotation=10, fontsize=12)
        ax.tick_params(axis="both", which="both", labelsize=12)
        ax.set_yscale("log")
    axs41[0].set_ylabel("Cancelation between Same Stimulus", fontsize=12)
    axs41[1].set_ylabel("Magnitude of Projection", fontsize=12)
    fig41.tight_layout()
    fig41.savefig(fp(f"outputsubspace_cancel_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"outputsubspace_cancel_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig41)

    # Save the output-subspace cancellation scatter data so paper_plot can
    # replot it. projs_all[cat][stim] = [combined, |task1|, |task2|] where
    # combined = |task1 + task2| (left panel) and |task1| is the individual
    # magnitude (right panel). Categories: cosine-output / orthogonal / random.
    outputsubspace_data = {
        "projs_all": np.asarray(projs_all, dtype=float),   # (3 cat, 8 stim, 3 vals)
        "category_labels": ["Projection to Cosine Output",
                            "Orthogonal Complement", "Random Vector"],
        "combined_ylabel": "Cancelation between Same Stimulus",
        "magnitude_ylabel": "Magnitude of Projection",
    }
    outputsubspace_path = save_dir / f"outputsubspace_cancel_seed{seed}_{hyp_dict['addon_name']}.pkl"
    with open(outputsubspace_path, "wb") as f:
        pickle.dump(outputsubspace_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved data: " + str(outputsubspace_path))

    # ═════════════════════════════════════════════════════════════════════════
    # Cells 43-47: cross-task / cross-period PCA (figure2A_pca_fve)
    # ═════════════════════════════════════════════════════════════════════════
    H = db[f"hidden{layer_index}"]
    # Effective modulation W⊙M (the actual weight change applied to the recurrent
    # connections), flattened over (hidden, embed). W matches M's last two axes.
    M_raw = np.asarray(db[f"M{layer_index}"])            # (batch, T, hidden, embed)
    W_eff = (net.mp_layer1.W.data.detach().cpu().numpy() if net_params["input_layer_add"]
             else net.mp_layer0.W.data.detach().cpu().numpy())
    WM = (M_raw * W_eff[None, None, :, :]).reshape(M_raw.shape[0], M_raw.shape[1], -1)
    task_id = test_task
    periods = time_stamp_extract(test_input, time_stamps_usual)
    periods_ = {
        0: {"context": (0, periods["stimulus_start"] - 1),
            "stim": (periods["stimulus_start"], periods["stimulus_end"]),
            "delay": (periods["delay_start"], periods["delay_end"] - 1),
            "resp": (periods["delay_end"], periods["trial_end"])},
        1: {"context": (0, periods["stimulus_start"] - 1),
            "stim": (periods["stimulus_start"], periods["stimulus_end"]),
            "delay": (periods["delay_start"], periods["delay_end"] - 1),
            "resp": (periods["delay_end"], periods["trial_end"])},
    }

    top_k = 4
    res_H = figure2A_pca_fve(H, task_id, periods_, k=top_k, max_pcs=10, center="None")
    res_WM = figure2A_pca_fve(WM, task_id, periods_, k=top_k, max_pcs=10, center="None")

    data_all = [["hidden", res_H], ["w_modulation", res_WM]]
    pcs = {}
    name = "hidden"
    for name, res in data_all:  # cell 45
        fig45, axs45 = plt.subplots(1, 2, figsize=(4 * 2, 4))
        for task_index in range(2):
            for period_index in range(4):
                evr_curve = res[task_index]["evr_curves"][period_index]
                period_name = res[task_index]["period_names"][period_index]
                cev = np.cumsum(evr_curve, axis=0)
                axs45[task_index].plot([ii + 1 for ii in range(len(cev))], cev, "-o", color=c_vals[period_index], label=period_name)
                pcs[f"{name}_task{task_index}_{period_name}"] = cev
        axs45[0].set_ylabel("Go Task; Var. expl.", fontsize=15)
        axs45[1].set_ylabel("Anti Task; Var. expl.", fontsize=15)
        for ax in axs45:
            ax.set_xlabel("No. of PCs", fontsize=15)
            ax.legend(fontsize=12, frameon=True, loc="best")
            ax.set_title(name, fontsize=12)
        fig45.tight_layout()
        fig45.savefig(fp(f"dimension_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"dimension_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png")))
        plt.close(fig45)

    for name, res in data_all:  # cell 46
        fig46, axs46 = plt.subplots(1, 2, figsize=(4 * 2, 4))
        for task_index in range(2):
            fve_k = res[task_index]["fve_k"]
            sns.heatmap(fve_k, ax=axs46[task_index],
                        xticklabels=res_H[task_index]["period_names"],
                        yticklabels=res_H[task_index]["period_names"], annot=True, fmt=".2f")
        axs46[0].set_title(f"Go Task, k={top_k}", fontsize=15)
        axs46[1].set_title(f"Anti Task, k={top_k}", fontsize=15)
        fig46.tight_layout()
        fig46.savefig(fp(f"d_separate_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"d_separate_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png")))
        plt.close(fig46)

    fve_k_alls = []
    d_combine_data = {}  # self-contained data to replot the d_combine heatmaps
    for name, res in data_all:  # cell 47
        fig47, axs47 = plt.subplots(1, 1, figsize=(4 * 1, 4))
        fve_k_all = res["__cross_task__"]["fve_k_all"]
        labels_all = res["__cross_task__"]["labels"]
        permute = [0, 4, 1, 5, 2, 6, 3, 7]
        fve_k_all = fve_k_all[np.ix_(permute, permute)]
        labels_all = [labels_all[ii] for ii in permute]
        labels_alt = ["Pro Context", "Anti Context", "Pro Stim", "Anti Stim",
                      "Pro Memory", "Anti Memory", "Pro Response", "Anti Response"]
        sns.heatmap(fve_k_all, ax=axs47, xticklabels=labels_alt, yticklabels=labels_alt,
                    annot=True, fmt=".2f", vmin=0.0, vmax=1.0)
        fve_k_alls.append(fve_k_all)
        # Everything paper_plot needs to redraw this exact heatmap: the permuted
        # matrix, its tick labels, the color range, and top_k.
        d_combine_data[name] = {
            "fve_k_all": np.asarray(fve_k_all),
            "labels": labels_alt,
            "vmin": 0.0,
            "vmax": 1.0,
            "top_k": top_k,
        }
        fig47.tight_layout()
        fig47.savefig(fp(f"d_combine_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"d_combine_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png")))
        plt.close(fig47)

    # Save the d_combine matrices next to the figures so paper_plot can replot
    # them. Keyed by name ("hidden" / "modulation").
    d_combine_path = save_dir / f"d_combine_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.pkl"
    with open(d_combine_path, "wb") as f:
        pickle.dump(d_combine_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved data: " + str(d_combine_path))

    # Per-task cumulative-variance-vs-#PCs curves (the "dimension" figure data),
    # saved self-contained so paper_plot can replot them. Structure:
    #   pc_cumvar_data[rep]["task_names"]      -> ["Go", "Anti"]
    #   pc_cumvar_data[rep]["period_names"]    -> per-task list of period labels
    #   pc_cumvar_data[rep]["cumvar"]          -> (n_task, n_period, max_pc) array
    task_disp = ["Go", "Anti"]
    pc_cumvar_data = {}
    for name, res in data_all:
        n_task = 2
        pnames0 = res[0]["period_names"]
        max_pc = res[0]["evr_curves"].shape[1]
        cumvar = np.zeros((n_task, len(pnames0), max_pc), dtype=float)
        for ti in range(n_task):
            for pi in range(len(res[ti]["period_names"])):
                cumvar[ti, pi, :] = np.cumsum(res[ti]["evr_curves"][pi], axis=0)
        pc_cumvar_data[name] = {
            "task_names": task_disp,
            "period_names": pnames0,
            "cumvar": cumvar,
            "max_pc": int(max_pc),
        }
    pc_cumvar_path = save_dir / f"pc_cumvar_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.pkl"
    with open(pc_cumvar_path, "wb") as f:
        pickle.dump(pc_cumvar_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved data: " + str(pc_cumvar_path))

    # Cell 48: cross-run pickle summary
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_all = {"learning_hm_similarity": learning_hm_similarity, "pcs": pcs, "fve_k_alls": fve_k_alls}
    with open(DATA_DIR / f"seed{seed}_{hyp_dict['addon_name']}.pkl", "wb") as f:
        pickle.dump(save_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 49: input-correlation heatmaps (W_in, W, W @ W_in)
    # ═════════════════════════════════════════════════════════════════════════
    W_in = Winput_lst[-1]
    if net_params["input_layer_add"]:
        W = net.mp_layer1.W.data.detach().cpu().numpy()
    else:
        W = net.mp_layer0.W.data.detach().cpu().numpy()

    if task_params["fixate_off"]:
        input_label = ["Fix On", "Fix Off", "Stim 1 Cos", "Stim 1 Sin", "Stim 2 Cos", "Stim 2 Sin", "Task 1", "Task 2"]
    else:
        input_label = ["Fix On", "Stim 1 Cos", "Stim 1 Sin", "Stim 2 Cos", "Stim 2 Sin", "Task 1", "Task 2"]

    Wcombs = [W_in, W, W @ W_in]
    wcomb_names = ["W_in", "W", "WW_in"]
    wcomb_titles = [r"$W_{\mathrm{in}}$", r"$W$", r"$WW_{\mathrm{in}}$"]
    corr_upper_all = {}
    fig49, axs49 = plt.subplots(1, 3, figsize=(4 * 3, 4))
    for idx, Wcomb in enumerate(Wcombs):
        C = np.corrcoef(Wcomb, rowvar=False)
        C_upper = C.copy()
        C_upper[np.tril_indices_from(C_upper, k=-1)] = np.nan
        sns.heatmap(C_upper, cmap="coolwarm", ax=axs49[idx],
                    xticklabels=input_label if idx != 1 else False,
                    yticklabels=input_label if idx != 1 else False,
                    annot=True if idx != 1 else False, fmt=".2f", vmin=-1.0, vmax=1.0)
        corr_upper_all[wcomb_names[idx]] = np.asarray(C_upper, dtype=float)
    axs49[0].set_title(wcomb_titles[0], fontsize=12)
    axs49[1].set_title(wcomb_titles[1], fontsize=12)
    axs49[2].set_title(wcomb_titles[2], fontsize=12)
    fig49.tight_layout()
    fig49.savefig(fp(f"w_stim_corr_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"w_stim_corr_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png")))
    plt.close(fig49)

    # Save the input-correlation matrices so paper_plot can replot them. Each is
    # the upper-triangular Pearson correlation between columns of the weight
    # matrix (W_in / W / W@W_in). The labeled panels (W_in, WW_in) are over the
    # input channels; W is over hidden units so it carries no channel labels.
    w_stim_corr_data = {
        "corr_upper": corr_upper_all,          # {"W_in","W","WW_in"} -> upper-tri corr
        "input_label": input_label,
        "titles": {"W_in": "W_in", "W": "W", "WW_in": "WW_in"},
        "vmin": -1.0,
        "vmax": 1.0,
    }
    w_stim_corr_path = save_dir / f"w_stim_corr_{name}_seed{seed}_{hyp_dict['addon_name']}.pkl"
    with open(w_stim_corr_path, "wb") as f:
        pickle.dump(w_stim_corr_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved data: " + str(w_stim_corr_path))

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 54 + 55: half-period time stamps + attractor at before/after training
    # ═════════════════════════════════════════════════════════════════════════
    time_stamps_usual_copy = copy.deepcopy(time_stamps_usual)
    time_stamps_usual_copy["fixation_half"] = int(time_stamps_usual_copy["fixation_end"] / 2)
    time_stamps_usual_copy["stimulus_half"] = int((time_stamps_usual_copy["stimulus_end"] - time_stamps_usual_copy["stimulus_start"]) / 2) + time_stamps_usual_copy["stimulus_start"]
    time_stamps_usual_copy["delay_half"] = int((time_stamps_usual_copy["delay_end"] - time_stamps_usual_copy["delay_start"]) / 2) + time_stamps_usual_copy["delay_start"]
    time_stamps_usual_copy["response_half"] = int((time_stamps_usual_copy["trial_end"] - time_stamps_usual_copy["delay_end"]) / 2) + time_stamps_usual_copy["delay_end"]

    compare_values = [
        ["hidden", None, r"$h$"],
        ["modulation", 0, r"$M_{\mathrm{fix\ on}}$"],
        ["modulation", None, r"$M$"],
        ["w_modulation", None, r"$W \odot M$"],
        ["w_modulation", "Win", r"$(W \odot M)\,W_{\mathrm{in}}$"],
    ]
    cl = len(compare_values)
    hidden_over_time_save = None
    stages = [[0, "Before Training", "beforetraining"], [-1, "Post Training", "posttraining"]]

    compare_value = "w_modulation"
    mean_all_save = None
    result_attractor_end_all = {}
    stages_counter = []
    stage_hidden_first = None   # first subplot of the Post-Training stage figure
    for stage_idx, stage_name, save_name in stages:
        figae, axsae = plt.subplots(1, cl, figsize=(4 * cl, 4))
        for idx, (compare_value, moddim, compare_name) in enumerate(compare_values):
            print(f"{idx}: {compare_name}")
            _, M_end, h_end, _ = modulation_extraction(test_input, db_lst[0][stage_idx], layer_index)
            result_attractor_end_all = {}
            all_keys = ["fixation_half", "fixation_end", "stimulus_half", "stimulus_end",
                        "delay_half", "delay_end", "response_half", "trial_end"]
            for key_idx, key in enumerate(all_keys):
                result_attractor, _, hidden_over_time = analyze_similarity(
                    M_end, h_end, net, net_params, label_task_comb,
                    checktime=time_stamps_usual_copy[key], compare=compare_value, moddim=moddim)
                result_attractor_end_all[key] = result_attractor
                if key_idx == 0 and idx == 0:
                    hidden_over_time_save = hidden_over_time

            mean_all = []
            for i in range(len(result_attractor_end_all["trial_end"])):
                mean = [rs[i][0] for rs in result_attractor_end_all.values()]
                std = [rs[i][1] for rs in result_attractor_end_all.values()]
                stages_counter = [ii for ii in range(len(result_attractor_end_all))]
                axsae[idx].plot(stages_counter, mean, "-o", color=c_vals[i], label=f"{break_names[i]}")
                axsae[idx].fill_between(stages_counter, [mean[j] - std[j] for j in range(len(mean))],
                                        [mean[j] + std[j] for j in range(len(mean))], alpha=0.5, color=c_vals_l[i])
                mean_all.append(mean)

            axsae[idx].set_xticks(stages_counter)
            axsae[idx].set_xticklabels(list(result_attractor_end_all.keys()), rotation=45, ha="right", fontsize=12)
            axsae[idx].legend(fontsize=12, frameon=True, loc="best")
            axsae[idx].set_ylabel(f"Cosine Sim of {compare_name}", fontsize=12)
            axsae[idx].set_ylim([-1.1, 1.1])

            if stage_idx == -1 and idx == 3:
                mean_all_save = mean_all

            # capture the FIRST subplot (idx==0, hidden) of the Post-Training figure
            if stage_idx == -1 and idx == 0:
                keys_ordered = list(result_attractor_end_all.keys())
                stage_hidden_first = {
                    "break_names": list(break_names),
                    "keys": keys_ordered,        # x tick labels (fixation_half ... trial_end)
                    "mean": [[float(rs[i][0]) for rs in result_attractor_end_all.values()]
                             for i in range(len(result_attractor_end_all["trial_end"]))],
                    "std": [[float(rs[i][1]) for rs in result_attractor_end_all.values()]
                            for i in range(len(result_attractor_end_all["trial_end"]))],
                    "ylabel": f"Cosine Sim of {compare_name}", "title": "Post Training",
                }

        figae.suptitle(stage_name, fontsize=15)
        figae.tight_layout()
        figae.savefig(fp(f"attractor_stage{save_name}_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"attractor_stage{save_name}_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
        plt.close(figae)

    # Save the first-subplot data of both attractor figures for paper_plot reuse:
    #  - attractor_over_learning: "Hidden" panel (cosine sim vs iteration)
    #  - attractor_stage_posttraining: "hidden" panel (cosine sim vs trial epoch)
    attractor_first_path = save_dir / f"attractor_first_{aname}.pkl"
    with open(attractor_first_path, "wb") as f:
        pickle.dump({"over_learning_hidden": attractor_hidden_first,
                     "stage_posttraining_hidden": stage_hidden_first}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved data: " + str(attractor_first_path))

    # Cell 56 + 57: relative change of W⊙M since stimulus end
    stage_names = np.array(list(result_attractor_end_all.keys()))
    stages_counter = np.array(stages_counter)
    assert len(mean_all_save) == 4
    mean_all_save_use = [mean_all_save[0], mean_all_save[1]]
    fig57, axs57 = plt.subplots(1, 2, figsize=(4 * 2, 4))
    for idx, entry in enumerate(mean_all_save_use):
        entry = entry[3:]
        entry_norm = [np.abs(entry_ - entry[0]) / np.abs(entry[0]) for entry_ in entry]
        entry = [np.abs(entry_ - entry[0]) for entry_ in entry]
        axs57[0].plot([ii for ii in range(len(stages_counter[3:]))], entry_norm, "-o", color=c_vals[idx], label=break_names[idx])
        axs57[1].plot([ii for ii in range(len(stages_counter[3:]))], entry, "-o", color=c_vals[idx], label=break_names[idx])
    for ax in axs57:
        ax.set_xticks([ii for ii in range(len(stages_counter[3:]))])
        ax.set_xticklabels(stage_names[3:], rotation=45, ha="right", fontsize=12)
        ax.legend(fontsize=12, frameon=True, loc="best")
    axs57[0].set_ylabel("Rel Change since Stimulus Period End", fontsize=12)
    axs57[1].set_ylabel("Change since Stimulus Period End", fontsize=12)
    fig57.tight_layout()
    fig57.savefig(fp(f"wm_relchange_stimulusend_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"wm_relchange_stimulusend_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig57)

    # Cell 58: per-stage Winput Gram matrices
    keys = ["Fixon", "Stim 1 Cos", "Stim 1 Sin", "Stim 2 Cos", "Stim 2 Sin", "Task 1", "Task 2"]
    fig58, axs58 = plt.subplots(1, len(Winput_lst), figsize=(4 * len(Winput_lst), 4))
    if len(Winput_lst) == 1:
        axs58 = [axs58]
    for idx, Winput in enumerate(Winput_lst):
        sns.heatmap(Winput.T @ Winput, ax=axs58[idx], center=0, cmap="coolwarm", square=True,
                    xticklabels=keys, yticklabels=keys, annot=True, fmt=".2f")
        axs58[idx].set_title(f"Training Stage {idx + 1}", fontsize=12)
    fig58.tight_layout()
    fig58.savefig(fp(f"w_gram_matrix_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"w_gram_matrix_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig58)

    # Save the input-embedding Gram matrices so paper_plot can replot the FINAL
    # stage. gram_final = Winput_final.T @ Winput_final (raw inner products); the
    # cosine form is gram / sqrt(outer(diag, diag)). Also keep all stages for
    # reference. keys = the 7 input-channel names.
    grams_all = np.asarray([Winput.T @ Winput for Winput in Winput_lst], dtype=float)
    w_gram_data = {
        "keys": keys,
        "gram_final": grams_all[-1],          # (7, 7) raw Gram, final stage
        "grams_all": grams_all,               # (n_stage, 7, 7)
    }
    w_gram_path = save_dir / f"w_gram_matrix_seed{seed}_{hyp_dict['addon_name']}.pkl"
    with open(w_gram_path, "wb") as f:
        pickle.dump(w_gram_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved data: " + str(w_gram_path))

    # ═════════════════════════════════════════════════════════════════════════
    # Cells 60-63: subspace orthogonality + readout heatmap
    # ═════════════════════════════════════════════════════════════════════════
    fix_idx = 0
    task1_idx = 5
    task2_idx = 6
    stim_idx = [1, 2, 3, 4]
    control_idx = [fix_idx, task1_idx, task2_idx]
    W_in = Winput_lst[-1]
    if net_params["input_layer_add"]:
        W = net.mp_layer1.W.data.detach().cpu().numpy()
    else:
        W = net.mp_layer0.W.data.detach().cpu().numpy()
    W_output = net.W_output.data.detach().cpu().numpy()

    report = subspace_orthogonality_report(W_in, stim_idx, control_idx)
    print("W_in principal-angle cosines:", report["cosines"], "max:", report["max_cos"])
    report = subspace_orthogonality_report(W @ W_in, stim_idx, control_idx)
    print("W@W_in principal-angle cosines:", report["cosines"], "max:", report["max_cos"])

    all2all = W_output @ W @ W_in
    fig63, ax63 = plt.subplots(1, 1, figsize=(4, 4))
    sns.heatmap(all2all, ax=ax63, center=0, cmap="coolwarm", square=True,
                xticklabels=keys, yticklabels=["Fixon", "Stim Cos", "Stim Sin"])
    fig63.tight_layout()
    fig63.savefig(fp(f"all2all_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"all2all_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig63)

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 67: W, W_in, W@W_in heatmaps
    # ═════════════════════════════════════════════════════════════════════════
    if net_params["input_layer_add"]:
        W = net.mp_layer1.W.data.detach().cpu().numpy()
    else:
        W = net.mp_layer0.W.data.detach().cpu().numpy()
    W_in = Winput_lst[-1]
    fig67, axs67 = plt.subplots(1, 3, figsize=(4 * 3, 4))
    sns.heatmap(W, ax=axs67[0], cmap="coolwarm", square=True, center=0)
    sns.heatmap(W_in, ax=axs67[1], cmap="coolwarm", center=0)
    sns.heatmap(W @ W_in, ax=axs67[2], cmap="coolwarm", center=0)
    axs67[0].set_ylabel("MPN Postsynaptic Neuron", fontsize=12)
    axs67[0].set_xlabel("MPN Presynaptic Neuron", fontsize=12)
    axs67[1].set_ylabel("MPN Presynpatic Neuron", fontsize=12)
    axs67[1].set_xlabel("Input Neuron", fontsize=12)
    axs67[2].set_ylabel("MPN Postsynaptic Neuron", fontsize=12)
    axs67[2].set_xlabel("Input Neuron", fontsize=12)
    axs67[0].set_title("W", fontsize=12)
    axs67[1].set_title(r"$W_{\mathrm{proj}}$", fontsize=12)
    axs67[2].set_title(r"$W \, @ \, W_{\mathrm{proj}}$", fontsize=12)
    fig67.tight_layout()
    fig67.savefig(fp(f"w_wwin{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"w_wwin{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig67)

    # ═════════════════════════════════════════════════════════════════════════
    # Cells 68-72: input-channel activation + W receive profiles
    # ═════════════════════════════════════════════════════════════════════════
    W_in = Winput_lst[-1]
    if task_params["fixate_off"]:
        delay1 = np.array([1, 0, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
        delay2 = np.array([1, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
        fixon = np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        fixoff = np.array([0, 1, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        task1 = np.array([0, 0, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
        task2 = np.array([0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
    else:
        delay1 = np.array([1, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
        delay2 = np.array([1, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
        fixon = np.array([1, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        fixoff = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        task1 = np.array([0, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
        task2 = np.array([0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)

    fixon_act = np.abs(W_in @ fixon)
    fixoff_act = np.abs(W_in @ fixoff)
    task1_act = np.abs(W_in @ task1)
    task2_act = np.abs(W_in @ task2)

    # Cell 69-71 diagnostics (printed)
    delay1_act = W_in @ delay1
    delay2_act = W_in @ delay2
    print("cos(delay1, delay2):", cosine_sim(delay1.flatten(), delay2.flatten()))
    print("cos(W_in@delay1, W_in@delay2):", cosine_sim(delay1_act.flatten(), delay2_act.flatten()))
    print("opnorm_2(W_in):", np.linalg.svd(W_in, compute_uv=False)[0])
    print("gain delay1:", np.linalg.norm(W_in @ delay1) / np.linalg.norm(delay1))
    print("gain fixon:", np.linalg.norm(W_in @ fixon) / np.linalg.norm(fixon))
    print("gain task1:", np.linalg.norm(W_in @ task1) / np.linalg.norm(task1))

    # Cell 72: W receive profiles
    fig72, axs72 = plt.subplots(3, 1, figsize=(6, 3 * 3))
    sumWpost = np.sum(np.abs(W), axis=1)
    sumWpre = np.sum(np.abs(W), axis=0)
    axs72[0].plot(sumWpost / np.mean(sumWpost), color=c_vals[0], label="W_post")
    axs72[1].plot(sumWpre / np.mean(sumWpre), color=c_vals[0], label="W_pre")
    axs72[2].plot(fixon_act, color=c_vals[0], label="fixon")
    axs72[2].plot(fixoff_act, color=c_vals[1], label="fixoff")
    axs72[2].plot(task1_act, color=c_vals[2], label="task1")
    axs72[2].plot(task2_act, color=c_vals[3], label="task2")
    for ax in axs72:
        ax.set_xlabel("MPN Postsynaptic Neuron", fontsize=12)
        ax.set_ylabel("Normalized Total Weight", fontsize=10)
        ax.legend(fontsize=12, frameon=True)
    fig72.tight_layout()
    fig72.savefig(fp(f"wreceive_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"wreceive_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig72)

    # Cell 73: W magnitude histogram
    Wm = net.mp_layer1.W.data.detach().cpu().numpy() if net_params["input_layer_add"] else net.mp_layer0.W.data.detach().cpu().numpy()
    Wm = np.asarray(Wm)
    mags = np.abs(Wm).ravel()
    mags = mags[np.isfinite(mags)]
    Nm = mags.size
    fig73, ax73 = plt.subplots(1, 1, figsize=(4, 2))
    weights = np.ones_like(mags) / Nm * 100.0
    ax73.hist(mags, weights=weights, bins=50)
    ax73.set_yscale("log")
    ax73.set_xlabel("W Magnitude", fontsize=12)
    ax73.set_ylabel("Prop of Entries", fontsize=12)
    fig73.tight_layout()
    fig73.savefig(fp(f"w_magnitude_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"w_magnitude_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig73)

    # Cell 74: parameter / buffer device diagnostics (printed)
    for name_p, p in net.named_parameters():
        print(f"{name_p:50s}  {p.device}  {tuple(p.shape)}  {p.dtype}")
    for name_b, bbuf in net.named_buffers():
        print(f"[buffer] {name_b:43s}  {bbuf.device}  {tuple(bbuf.shape)}  {bbuf.dtype}")

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 76: magnitude pruning of W (uses the LIVE network)
    # ═════════════════════════════════════════════════════════════════════════
    W_orig = net.mp_layer1.W.detach().cpu().numpy().copy()
    K_lst = [0.0, 10.0, 50.0, 90.0, 95.0, 98.0, 99.0, 99.90]
    acc_K_lst = []
    for K in K_lst:
        with torch.no_grad():
            Wp = net.mp_layer1.W
            w_np = Wp.detach().cpu().numpy()
            n = w_np.size
            k = int(round(K / 100.0 * n))
            if k > 0:
                idx = np.argpartition(np.abs(w_np).ravel(), k - 1)[:k]
                w_flat = w_np.ravel()
                w_flat[idx] = 0.0
            Wp.copy_(torch.from_numpy(w_np).to(Wp.device))
            net_out_redo, _, db_redo = net.iterate_sequence_batch(test_input, run_mode='track_states')
            acc_K_lst.append(net.compute_acc(net_out_redo, test_output, test_mask, test_input, isvalid=False, mode="stimulus")[0].item())
            Wp.copy_(torch.as_tensor(W_orig, device=Wp.device, dtype=Wp.dtype))
    fig76, ax76 = plt.subplots(1, 1, figsize=(4, 2))
    ax76.plot([ii for ii in range(len(K_lst))], np.array(acc_K_lst) * 100, "-o", color=c_vals[0])
    ax76.set_xticks([ii for ii in range(len(K_lst))])
    ax76.set_xticklabels(K_lst)
    ax76.set_xlabel("Sparsity of W (%)", fontsize=12)
    ax76.set_ylabel("Accuracy", fontsize=12)
    fig76.tight_layout()
    fig76.savefig(fp(f"w_hurt_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"w_hurt_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig76)

    # Save the magnitude-pruning accuracy curve so paper_plot can replot it:
    # accuracy (fraction) vs. sparsity level (% of recurrent W entries zeroed by
    # smallest magnitude). Zero sparsity is the intact-network accuracy.
    w_hurt_data = {
        "compare_value": compare_value,
        "sparsity_pct": np.asarray(K_lst, dtype=float),      # (n_K,) % of W zeroed
        "accuracy": np.asarray(acc_K_lst, dtype=float),      # (n_K,) fraction correct
    }
    w_hurt_path = save_dir / f"w_hurt_{compare_value}_seed{seed}_{hyp_dict['addon_name']}.pkl"
    with open(w_hurt_path, "wb") as f:
        pickle.dump(w_hurt_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved data: " + str(w_hurt_path))

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 77: component cosine-similarity diagnostics (same stim / same resp)
    # ═════════════════════════════════════════════════════════════════════════
    input1 = db_lst[0][-1]["input1"] if net_params["input_layer_add"] else db_lst[0][-1][f"input{layer_index}"]
    hidden1 = db_lst[0][-1][f"hidden{layer_index}"]
    W = net.mp_layer1.W.data.detach().cpu().numpy() if net_params["input_layer_add"] else net.mp_layer0.W.data.detach().cpu().numpy()
    W_in = Winput_lst[-1]
    _, M_all, h_end, _ = modulation_extraction(test_input, db_lst[0][-1], layer_index)

    fig77, axs77 = plt.subplots(12, 1, figsize=(6, 13 * 2))
    maps = [lambda i: i, lambda i: (i + 4) % 8]
    map_names = ["Same Stim", "Same Resp"]

    for idx, map_ in enumerate(maps):
        for k in range(8):
            ind1 = [i for i, lst in enumerate(label_task_comb) if np.array_equal(lst, [k, 0])][0]
            ind2 = [i for i, lst in enumerate(label_task_comb) if np.array_equal(lst, [map_(k), 1])][0]
            input1_, input2_ = input1[ind1], input1[ind2]
            hidden1_, hidden2_ = hidden1[ind1], hidden1[ind2]
            cosines = [[], [], [], [], [], [], [], [], [], [], [], []]
            for t in range(1, input1_.shape[0]):
                h1, i1 = hidden1_[t].reshape(-1, 1), input1_[t].reshape(-1, 1)
                h2, i2 = hidden2_[t].reshape(-1, 1), input2_[t].reshape(-1, 1)
                M_tp1, M_tp2 = M_all[ind1, t - 1, :, :], M_all[ind2, t - 1, :, :]
                assert W.shape == M_tp1.shape == M_tp2.shape
                proj1, proj2 = ((h1 @ i1.T)).flatten(), ((h2 @ i2.T)).flatten()
                proj1W, proj2W = ((h1 @ i1.T) * W).flatten(), ((h2 @ i2.T) * W).flatten()
                ihM1, ihM2 = ((h1 @ i1.T) * M_tp1).flatten(), ((h2 @ i2.T) * M_tp2).flatten()
                ihWM1, ihWM2 = ((h1 @ i1.T) * W * M_tp1).flatten(), ((h2 @ i2.T) * W * M_tp2).flatten()
                proj1Wx, proj2Wx = ((W @ i1)).flatten(), ((W @ i2)).flatten()
                proj1x, proj2x = i1.flatten(), i2.flatten()
                proj1MW, proj2MW = (M_tp1 * W).flatten(), (M_tp2 * W).flatten()
                proj1MWx, proj2MWx = (((M_tp1 * W) @ i1)).flatten(), (((M_tp2 * W) @ i2)).flatten()
                proj1MWx_fake, proj2MWx_fake = (((M_tp1 * W) @ i2)).flatten(), (((M_tp2 * W) @ i1)).flatten()

                proj1M, proj2M = (M_tp1.flatten(), M_tp2.flatten())

                cosines[0].append(cosine_sim(proj1, proj2))
                cosines[1].append(cosine_sim(proj1W, proj2W))
                cosines[2].append(cosine_sim(proj1Wx, proj2Wx))
                cosines[3].append(cosine_sim(proj1x, proj2x))
                cosines[4].append(cosine_sim(proj1M, proj2M))
                cosines[5].append(cosine_sim(proj1MWx, proj2MWx))
                cosines[6].append(cosine_sim(proj1MW, proj2MW))
                cosines[7].append(cosine_sim(ihM1, ihM2))
                cosines[8].append(cosine_sim(ihWM1, ihWM2))
                cosines[9].append(cosine_sim(h1.flatten(), h2.flatten()))
                cosines[10].append(cosine_sim(proj1MWx_fake, proj1MWx))
                cosines[11].append(cosine_sim(proj2MWx_fake, proj1MWx))

            for u in range(len(cosines)):
                label = map_names[idx] if k == 0 else None
                axs77[u].plot(cosines[u], color=c_vals[idx], linestyle="-", label=label)

    for idx, ax in enumerate(axs77):
        ax.set_xlabel("Time Steps", fontsize=15)
        ax.axvline(time_stamps_usual["delay_end"], color=c_vals[2])
        if idx not in (7, 8):
            ax.set_ylim([-1.1, 1.1])
        ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.2)
        ax.tick_params(axis="both", which="minor", labelsize=14, length=3, width=1.0)
        ax.axhline(0.0, color=c_vals[3], linestyle="--")
        ax.legend(fontsize=12, frameon=True, loc="upper left")
    ylabels = [r"$(h_t x_t^{\top})$", r"$(h_t x_t^{\top}) \odot W$", r"$Wx_t$", r"$x_t$",
               r"$M_{t-1}$", r"$(M_{t-1} \odot W)x_t$", r"$M_{t-1} \odot W$",
               r"$(h_t x_t^{\top}) \odot M$", r"$(h_t x_t^{\top}) \odot W \odot M$",
               r"$h_t$", r"Fake Input", r"Fake Modulation"]
    for u, yl in enumerate(ylabels):
        axs77[u].set_ylabel(yl, fontsize=15)
    for ax in axs77.flatten():
        ax.set_ylim([-1.05, 1.05])
    fig77.tight_layout()
    fig77.savefig(fp(f"hxw_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"hxw_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig77)

    # Cell 79: response-input cosine diagnostics (printed)
    if task_params["fixate_off"]:
        resp1 = np.array([0, 1, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
        resp2 = np.array([0, 1, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
    else:
        resp1 = np.array([0, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
        resp2 = np.array([0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
    W_in = Winput_lst[-1]
    W = net.mp_layer1.W.data.detach().cpu().numpy()
    print("cos(W_in@resp1, W_in@resp2):", cosine_sim((W_in @ resp1).ravel(), (W_in @ resp2).ravel()))
    print("cos(W@W_in@resp1, W@W_in@resp2):", cosine_sim((W @ (W_in @ resp1)).ravel(), (W @ (W_in @ resp2)).ravel()))

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 81: alpha-interpolations between pro/anti inputs (uses test tensors)
    # ═════════════════════════════════════════════════════════════════════════
    alpha_lst, stacked_interpolation_ld, stacked_interpolation_answer_ld = input_interpolation(
        test_input_longdelay, test_output_longdelay, label_task_comb_longdelay, expand_stimulus=False)
    _, stacked_interpolation_lr, stacked_interpolation_answer_lr = input_interpolation(
        test_input_longresponse, test_output_longresponse, label_task_comb_longresponse, expand_stimulus=False)
    _, stacked_interpolation_ls, stacked_interpolation_answer_ls = input_interpolation(
        test_input_longstimulus, test_output_longstimulus, label_task_comb_longstimulus, expand_stimulus=False)
    _, stacked_interpolation_lf, stacked_interpolation_answer_lf = input_interpolation(
        test_input_longfixation, test_output_longfixation, label_task_comb_longfixation, expand_stimulus=False)

    # Cell 83: time-stamp / input map
    time_stamp_input_map = [
        [time_stamps_usual, test_input, "normal", 0, "delay_end", label_task_comb],
        [time_stamps, test_input_longdelay, "longdelay", 3, "delay_end", label_task_comb_longdelay],
        [time_stamps_longstimulus, test_input_longstimulus, "longstimulus", 2, "stimulus_end", label_task_comb_longstimulus],
        [time_stamps_longresponse, test_input_longresponse, "longresponse", 4, "trial_end", label_task_comb_longresponse],
        [time_stamps_longfixation, test_input_longfixation, "longfixation", 1, "fixation_end", label_task_comb_longfixation],
    ]

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 86: PCA trajectories of hidden / modulation for the NORMAL variant
    # ═════════════════════════════════════════════════════════════════════════
    zeros_pca = None
    wout_proj = None
    batch_num = None
    m_pca_normal_data = {}  # self-contained data to replot the "normal" m_pca figures
    # Only the "normal" variant is plotted (the long-period variants were dropped).
    for time_stamp_long, test_input_long, sname, db_index, _, label_task_comb_long in time_stamp_input_map:
        if sname != "normal":
            continue
        print(f"sname: {sname}; db_index: {db_index}")
        names = ["hidden", "modulation", "w_modulation"]
        # Recurrent weight for the effective modulation W⊙M (see analyze_similarity).
        W_mp = (net.mp_layer1.W.data.detach().cpu().numpy() if net_params["input_layer_add"]
                else net.mp_layer0.W.data.detach().cpu().numpy())
        for name in names:
            fighs, axshs = plt.subplots(1, 3, figsize=(5 * 3, 5 * 1))
            PCA_downsample = 3
            Ms, Ms_orig, hs, bs = modulation_extraction(test_input_long, db_lst[db_index][-1], layer_index)
            batch_num = Ms_orig.shape[0]
            if name == "modulation":
                data = Ms
            elif name == "w_modulation":
                # Effective modulation W⊙M, flattened like Ms (batch, T, hidden*embed).
                eff = Ms_orig * W_mp[None, None, :, :]
                data = eff.reshape(eff.shape[0], eff.shape[1], -1)
            elif name == "hidden":
                data = hs
            n_activity = data.shape[-1]
            activity_zero = np.zeros((1, n_activity))
            as_flat = data.reshape((-1, n_activity))
            pca = PCA(n_components=PCA_downsample, random_state=42)
            pca.fit(as_flat)
            if name == "hidden":
                wout = net.W_output.detach().cpu().numpy()
                wout_proj = pca.transform(wout)
            as_pca = pca.transform(as_flat)
            projected_data = as_pca.reshape((data.shape[0], data.shape[1], -1))
            zeros_pca = pca.transform(activity_zero)

            combination = [(0, 1), (0, 2), (1, 2)]
            phases = [("fix", "fixation_start", "fixation_end", 1),
                      ("stim", "stimulus_start", "delay_start", 2),
                      ("delay", "delay_start", "delay_end", 3),
                      ("resp", "delay_end", "trial_end", 0)]
            transitions = [("fixation_end", 1), ("delay_start", 2), ("delay_end", 3), ("trial_end", 0)]
            period_markers = {"Fixation": 1, "Stimulus": 2, "Delay": 3, "Response": 0}
            stim0 = time_stamp_long["stimulus_start"]
            trial_end = time_stamp_long["trial_end"]
            legend_handles = [Line2D([0], [0], marker=markers_vals[idx], linestyle="None", markersize=10,
                                     markerfacecolor="k", markeredgecolor="k", label=label)
                              for label, idx in period_markers.items()]

            for i in range(batch_num):
                # Use the CURRENT variant's labels (label_task_comb_long), not
                # the longdelay ones — variants can have different per-stimulus
                # batch ordering, so indexing with longdelay mismatches colors
                # (e.g. the "normal" figure collapsing 8 stimuli to ~4 colors).
                task = label_task_comb_long[i, 1]
                if task not in (0, 1):
                    continue
                color = c_vals[label_task_comb_long[i, 0]]
                ls = linestyles[task]
                data_i = projected_data[i]
                seg = slice(stim0, trial_end)
                for ax, (a, bb) in zip(axshs, combination):
                    ax.plot(data_i[seg, a], data_i[seg, bb], c=color, linestyle=ls,
                            alpha=0.05 if sname != "normal" else 0.25)
                    for _, t0_key, t1_key, mk_idx in phases:
                        sl = slice(time_stamp_long[t0_key], time_stamp_long[t1_key])
                        ax.scatter(data_i[sl, a], data_i[sl, bb], c=color, marker=markers_vals[mk_idx],
                                   alpha=0.05 if sname != "normal" else 0.5)
                    for t_key, mk_idx in transitions:
                        t = time_stamp_long[t_key] - 1
                        ax.scatter([data_i[t, a]], [data_i[t, bb]], c=color, marker=markers_vals[mk_idx],
                                   alpha=0.8, s=60, linewidths=0.6, zorder=10)
            for ax, (a, bb) in zip(axshs, combination):
                ax.set_xlabel(f"PCA {a+1}", fontsize=12)
                ax.set_ylabel(f"PCA {bb+1}", fontsize=12)
                ax.set_title(f"name: {name}; sname: {sname}", fontsize=15)
                ax.legend(handles=legend_handles, loc="upper right", frameon=False)
            fighs.tight_layout()
            fighs.savefig(fp(f"m_pca_{name}_seed{seed}_{hyp_dict['addon_name']}_{sname}.png"), dpi=300)
            print("  Saved figure: " + str(fp(f"m_pca_{name}_seed{seed}_{hyp_dict['addon_name']}_{sname}.png")))
            plt.close(fighs)

            # Stash everything paper_plot needs to redraw the "normal" panels:
            # the PCA-projected trajectories (batch, T, 3), the projected origin,
            # the readout projection (hidden only), the period/transition time
            # keys, and the time stamps + labels that drive coloring/markers.
            # Use this variant's own labels (label_task_comb_long), which for the
            # normal variant is label_task_comb.
            if sname == "normal":
                m_pca_normal_data[name] = {
                    "projected_data": np.asarray(projected_data),
                    "zeros_pca": np.asarray(zeros_pca),
                    "wout_proj": np.asarray(wout_proj) if name == "hidden" else None,
                    "label_task_comb": np.asarray(label_task_comb_long),
                    "time_stamps": dict(time_stamp_long),
                    "combination": combination,
                    "phases": phases,
                    "transitions": transitions,
                    "period_markers": period_markers,
                    "markers_vals": markers_vals,
                    "linestyles": linestyles,
                }

    # Save the "normal" m_pca trajectory data next to the figures so paper_plot
    # can replot them. Keyed by name ("hidden" / "modulation").
    if m_pca_normal_data:
        m_pca_path = save_dir / f"m_pca_normal_seed{seed}_{hyp_dict['addon_name']}.pkl"
        with open(m_pca_path, "wb") as f:
            pickle.dump(m_pca_normal_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  Saved data: " + str(m_pca_path))

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 91: interpolation fixed-point ring analysis (uses the LIVE network)
    # ═════════════════════════════════════════════════════════════════════════
    stacked_interpolation_lst = [stacked_interpolation_ld, stacked_interpolation_lr, stacked_interpolation_ls, stacked_interpolation_lf]
    time_stamps_lst = [time_stamps, time_stamps_longresponse, time_stamps_longstimulus, time_stamps_longfixation]
    stacked_interpolation_name_lst = ["longdelay", "longresponse", "longstimulus", "longfixation"]
    desire_period = [[time_stamps["delay_start"], time_stamps["delay_end"]],
                     [time_stamps_longresponse["delay_end"], time_stamps_longresponse["trial_end"]],
                     [time_stamps_longstimulus["stimulus_start"], time_stamps_longstimulus["stimulus_end"]],
                     [time_stamps_longfixation["fixation_start"], time_stamps_longfixation["fixation_end"]]]

    int_input_all = []
    raw_data_ring_all, raw_data_ring_magnitude_all, projected_data_ring_all = [], [], []
    PCA_downsample = 3
    int_index = 0
    name = "hidden"
    # self-contained data to replot the m_pca_attractor_cycle figures, keyed by
    # (sname, name) -> dict. Holds the per-alpha fixed-point PCA projections.
    attractor_cycle_data = {}

    for siindex, stacked_interpolation_ in enumerate(stacked_interpolation_lst):
        sname = stacked_interpolation_name_lst[siindex]
        print(f"sname: {sname}")
        names = ["hidden", "modulation", "w_modulation"]
        raw_data_ring = [[], [], []]
        raw_data_ring_magnitude = [[], [], []]
        projected_data_ring = [[], [], []]

        # The forward pass per alpha step is INDEPENDENT of the representation
        # ("hidden"/"modulation"/"w_modulation") — all three are derived from the
        # same tracked db. Run it ONCE per alpha here (rather than 3x inside the
        # name loop below) and cache the per-representation activity, so this cell
        # does n_alpha forward passes instead of 3*n_alpha.
        Wlocal = net.mp_layer1.W.data.detach().cpu().numpy()
        alpha_activity = []   # per alpha: {name -> (batch, T, n_activity)}
        for int_index, int_input in enumerate(stacked_interpolation_):
            if int_index == 0:
                int_input_all.append(int_input)   # alpha=0 input per period (Cell 91 downstream)
            _, _, db_intp = net.iterate_sequence_batch(
                int_input, run_mode='track_states', save_to_cpu=True, detach_saved=True)
            Ms, Ms_orig, hs, _ = modulation_extraction(int_input, db_intp, layer_index)
            alpha_activity.append({
                "hidden": hs,
                "modulation": Ms,
                "w_modulation": (Ms_orig * Wlocal[None, None, :, :]).reshape(
                    Ms.shape[0], Ms.shape[1], -1),
            })

        for nindex, name in enumerate(names):
            fighsadd, axshsadd = plt.subplots(1, 3, figsize=(5 * 3, 5 * 1))
            fig3dfix = go.Figure()
            combination = [[0, 1], [0, 2], [1, 2]]
            interpolation_label = [i for i in range(len(stacked_interpolation_[0]))]
            projected_data_fix_all = []
            pca_delay = None

            for int_index in range(len(stacked_interpolation_)):
                data = alpha_activity[int_index][name]   # cached; no forward pass
                n_activity = data.shape[-1]
                as_flat_wantperiod_ = data[:, desire_period[siindex][0]:desire_period[siindex][1], :]
                as_flat_wantperiod = as_flat_wantperiod_.reshape((-1, n_activity))
                as_flat_fixedpoint_raw = data[:, desire_period[siindex][1], :]

                raw_data_ring[names.index(name)].append(ring_length(as_flat_fixedpoint_raw))
                fixpt_norm = np.linalg.norm(as_flat_fixedpoint_raw, axis=1)
                raw_data_ring_magnitude[names.index(name)].append(fixpt_norm.mean())

                as_flat = data.reshape((-1, n_activity))
                if int_index == 0:
                    pca_delay = PCA(n_components=PCA_downsample, random_state=42)
                    activity_zero = np.zeros((1, n_activity))
                    pca_delay.fit(as_flat_wantperiod)
                as_pca = pca_delay.transform(as_flat)
                projected_data = as_pca.reshape((data.shape[0], data.shape[1], -1))
                projected_data_fix = projected_data[:, desire_period[siindex][1], :]
                projected_data_ring[names.index(name)].append(ring_length(projected_data_fix))
                projected_data_fix_all.append(projected_data_fix)

            for index, comb in enumerate(combination):
                select1 = [pa[:, comb[0]] for pa in projected_data_fix_all]
                min_select1 = min(arr.min() for arr in select1)
                select2 = [pa[:, comb[1]] for pa in projected_data_fix_all]
                min_select2 = min(arr.min() for arr in select2)
                epsilon = 1 if name == "hidden" else 10
                min_select1 -= epsilon
                min_select2 -= epsilon
                indices_lst = [0, 10, -1]
                for it_idx, it in enumerate(indices_lst):
                    xy = projected_data_fix_all[it][:, [comb[0], comb[1]]]
                    num_xy = xy.shape[0]
                    for xy_index in range(num_xy):
                        axshsadd[index].plot([xy[xy_index % num_xy, 0], xy[(xy_index + 1) % num_xy, 0]],
                                             [xy[xy_index % num_xy, 1], xy[(xy_index + 1) % num_xy, 1]],
                                             linestyle="--", linewidth=3, color=c_vals_l[it_idx])
                for i in range(len(interpolation_label)):
                    fixed_points = np.array([pdf[i, :] for pdf in projected_data_fix_all])
                    axshsadd[index].plot(fixed_points[:, comb[0]], fixed_points[:, comb[1]], "-o", c=c_vals[interpolation_label[i]])
                    axshsadd[index].set_xlabel(f"PCA {comb[0]+1}", fontsize=15)
                    axshsadd[index].set_ylabel(f"PCA {comb[1]+1}", fontsize=15)
                    if index == 0:
                        fig3dfix.add_trace(go.Scatter3d(
                            x=np.array(alpha_lst), y=fixed_points[:, 0], z=fixed_points[:, 1],
                            mode="lines+markers", line=dict(width=6, color=c_vals[interpolation_label[i]]),
                            marker=dict(size=5, color=c_vals[interpolation_label[i]], symbol="circle"),
                            opacity=0.5, name=f"Stimulus {i}", showlegend=True))

            fighsadd.suptitle(f"name: {name}; sname: {sname}", fontsize=20)
            fighsadd.tight_layout()
            fighsadd.savefig(fp(f"m_pca_attractor_cycle_{name}_seed{seed}_{hyp_dict['addon_name']}_{int_index}_{sname}.png"), dpi=300)
            print("  Saved figure: " + str(fp(f"m_pca_attractor_cycle_{name}_seed{seed}_{hyp_dict['addon_name']}_{int_index}_{sname}.png")))
            plt.close(fighsadd)

            # Stash everything paper_plot needs to redraw the attractor_cycle
            # panels for this (sname, name). projected_data_fix is the per-alpha
            # fixed-point PCA projection, shape (n_alpha, batch_num, 3) — all 3
            # PCs are kept so any panel (incl. PC2) can be replotted. The cycle
            # figure connects, per stimulus i, fixed_points across alpha steps,
            # and overlays dashed rings for alpha indices [0, 10, -1].
            attractor_cycle_data[(sname, name)] = {
                "projected_data_fix_all": np.asarray(projected_data_fix_all),
                "alpha_lst": np.asarray(alpha_lst),
                "interpolation_label": list(interpolation_label),
                "combination": [list(c) for c in combination],
                "ring_indices": [0, 10, -1],
            }
            fig3dfix.update_layout(
                title=dict(text=f"name: {name}; sname: {sname}", x=0.5, xanchor="center", y=0.95, font=dict(size=14)),
                scene=dict(domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
                           xaxis=dict(title="Alpha", tickfont=dict(size=12)),
                           yaxis=dict(title=f"PCA 1; Anti {sname}", tickfont=dict(size=12)),
                           zaxis=dict(title=f"PCA 2; Anti {sname}", tickfont=dict(size=12)),
                           aspectratio=dict(x=1, y=1, z=0.8)),
                width=650, height=650, margin=dict(l=10, r=10, t=35, b=10), showlegend=True)

        raw_data_ring_all.append(raw_data_ring)
        raw_data_ring_magnitude_all.append(raw_data_ring_magnitude)
        projected_data_ring_all.append(projected_data_ring)

    # Save the attractor_cycle fixed-point data next to the figures so paper_plot
    # can replot them. Keys are stringified "{sname}|{name}" (sname in
    # {longdelay, longresponse, longstimulus, longfixation}).
    if attractor_cycle_data:
        ac_save = {f"{sn}|{nm}": v for (sn, nm), v in attractor_cycle_data.items()}
        ac_path = save_dir / f"m_pca_attractor_cycle_seed{seed}_{hyp_dict['addon_name']}.pkl"
        with open(ac_path, "wb") as f:
            pickle.dump(ac_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  Saved data: " + str(ac_path))

    # Cell 93: ring perimeter vs alpha
    fig93, axs93 = plt.subplots(2, 2, figsize=(4 * 2, 4 * 2), sharex=True)
    ax_hd_hid, ax_hd_mod = axs93[0, 0], axs93[0, 1]
    ax_3d_hid, ax_3d_mod = axs93[1, 0], axs93[1, 1]
    for i, sname in enumerate(stacked_interpolation_name_lst):
        y_hd_hidden = normalize_lst(raw_data_ring_all[i][0])
        y_hd_mod = normalize_lst(raw_data_ring_all[i][1])
        y_3d_hidden = normalize_lst(projected_data_ring_all[i][0])
        y_3d_mod = normalize_lst(projected_data_ring_all[i][1])
        ax_hd_hid.plot(alpha_lst, y_hd_hidden, "-o", color=c_vals[i], alpha=0.9, label=sname)
        ax_hd_mod.plot(alpha_lst, y_hd_mod, "-o", color=c_vals[i], alpha=0.9, label=sname)
        ax_3d_hid.plot(alpha_lst, y_3d_hidden, "-o", color=c_vals[i], alpha=0.9, label=sname)
        ax_3d_mod.plot(alpha_lst, y_3d_mod, "-o", color=c_vals[i], alpha=0.9, label=sname)
    ax_hd_hid.set_title("High-D Ring Perimeter (Hidden)", fontsize=14)
    ax_hd_mod.set_title("High-D Ring Perimeter (Modulation)", fontsize=14)
    ax_3d_hid.set_title("3-D Ring Perimeter (Hidden)", fontsize=14)
    ax_3d_mod.set_title("3-D Ring Perimeter (Modulation)", fontsize=14)
    ax_hd_hid.set_ylabel("Normalized Ring Perimeter", fontsize=13)
    ax_3d_hid.set_ylabel("Normalized Ring Perimeter", fontsize=13)
    ax_3d_hid.set_xlabel("Alpha", fontsize=13)
    ax_3d_mod.set_xlabel("Alpha", fontsize=13)
    for ax in axs93.ravel():
        ax.set_yscale("log")
        ax.set_ylim([5e-2, 1e0 + 2e-1])
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.legend(frameon=True, fontsize=12)
    fig93.tight_layout()
    fig93.savefig(fp(f"m_pca_ring_ALL_{name}_seed{seed}_{hyp_dict['addon_name']}_{int_index}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"m_pca_ring_ALL_{name}_seed{seed}_{hyp_dict['addon_name']}_{int_index}.png")))
    plt.close(fig93)

    # ═════════════════════════════════════════════════════════════════════════
    # (Removed) Cells 98-105: stimulus-PCA + readout response trajectories and
    # endpoint-plane hulls (Plotly). These built in-memory go.Figure() objects
    # but never saved them (no write_html/write_image), so they produced no
    # output files — only console prints, including degenerate-fixation Qhull
    # warnings. Deleted as dead analysis.
    # ═════════════════════════════════════════════════════════════════════════

    # ── Gradient-based TRUE fixed points of the modulation matrix (per rule) ──
    # Mirror the one-task analysis: for each of the two rules, solve genuine fixed
    # points M* = F(M*; x) per trial period, seeded from a DENSE grid of stimulus
    # angles (continuous-attractor probe). One pickle per rule
    # (fixed_points_grad_{aname}_{rule}.pkl). Shared solver lives in
    # core/grad_fixed_points.py, used by both one_task and two_task.
    # These gradient solves are the slow part; skip when run_fixed_points=False.
    if net_params["input_layer_add"]:
        W_fp = net.mp_layer1.W.data.detach().cpu().numpy()
    else:
        W_fp = net.mp_layer0.W.data.detach().cpu().numpy()
    cfg_fp = {"task_params": task_params, "train_params": train_params,
              "net_params": net_params}
    if run_fixed_points:
        for _rule in task_params["rules"]:
            try:
                solve_period_modulation_fixed_points(
                    aname, save_dir, net, cfg_fp, device,
                    rule=_rule, out_suffix=f"_{_rule}",
                    layer_index=layer_index, W=W_fp, n_interp=64,
                    n_seeds=fp_n_seeds)
            except Exception as exc:
                print(f"  [grad-fp/{_rule}] failed: {exc}")
                import traceback
                traceback.print_exc()
    else:
        print("  [grad-fp] skipped (--no-fixed-points).")

    # ── Fixed-point stability classification (post-analysis) ─────────────────
    # Re-package the linear-stability spectrum saved in the per-rule
    # fixed_points_grad_*.pkl into an explicit stable / marginal / unstable class
    # per fixed point (no recomputation). Runs whenever those pickles exist (they
    # are produced above unless --no-fixed-points).
    try:
        classify_fixed_point_stability(aname, save_dir, task_params["rules"])
    except Exception as exc:
        print(f"  [fp-classify] failed: {exc}")
        import traceback
        traceback.print_exc()

    # ═════════════════════════════════════════════════════════════════════════
    # Task-interpolation fixed points: for each stimulus, linearly mix the pro
    # (task 0) and anti (task 1) INPUT while holding the stimulus IDENTICAL, at
    # alpha = 0.0, 0.1, ... 1.0, and solve the TRUE gradient fixed point
    # M* = F(M*; x) per trial period at each alpha. Unlike the dense-angle
    # per-rule sweep above (which varies the stimulus within one rule), this
    # sweeps the CONTINUOUS pro<->anti axis at fixed stimuli, so it probes how the
    # attractor for each period morphs as the task cue is interpolated. Cost is
    # linear in the alpha count (--interp-n-alpha; one solve per alpha per period;
    # the 8 stimuli are batched). Deterministic: reuses the fixed test tensors, no
    # template RNG (unlike the per-rule solver's n_seeds sweep).
    # Saves interp_fixed_points_{aname}.pkl (self-contained for paper_plot),
    # including the raw M*/W⊙M* matrices (large: (n_alpha,n_stim,hid,emb)).
    # This is a gradient solve too, so it is skipped when run_fixed_points=False.
    # ═════════════════════════════════════════════════════════════════════════
    if not run_fixed_points:
        print("  [interp-fp] skipped (--no-fixed-points).")
    else:
        try:
            _interp_alphas, _interp_ld, _ = input_interpolation(
                test_input_longdelay, test_output_longdelay,
                label_task_comb_longdelay, expand_stimulus=False, n_alpha=interp_n_alpha)
            _, _interp_lr, _ = input_interpolation(
                test_input_longresponse, test_output_longresponse,
                label_task_comb_longresponse, expand_stimulus=False, n_alpha=interp_n_alpha)
            _, _interp_ls, _ = input_interpolation(
                test_input_longstimulus, test_output_longstimulus,
                label_task_comb_longstimulus, expand_stimulus=False, n_alpha=interp_n_alpha)
            _, _interp_lf, _ = input_interpolation(
                test_input_longfixation, test_output_longfixation,
                label_task_comb_longfixation, expand_stimulus=False, n_alpha=interp_n_alpha)

            # (period name, per-alpha stacked inputs, (start, end) window in that
            # variant's timebase). Windows reuse the Cell-91 desire_period bounds.
            _interp_variants = [
                ("longdelay",     _interp_ld, desire_period[0]),
                ("longresponse",  _interp_lr, desire_period[1]),
                ("longstimulus",  _interp_ls, desire_period[2]),
                ("longfixation",  _interp_lf, desire_period[3]),
            ]
            _REL_TOL = 0.05
            interp_fp_data = {
                "alphas": np.asarray(_interp_alphas, dtype=float),
                "n_stim": int(_interp_ld[0].shape[0]),
                "rel_tol": _REL_TOL,
                "results": {},   # period -> per-alpha arrays
            }
            for sname, stacked, (ps, pe) in _interp_variants:
                n_stim = stacked[0].shape[0]
                # Solve one fixed point per (alpha, stimulus) for this period,
                # accumulating the raw M*/W⊙M* matrices plus the compact
                # hidden/cos-out/rel_step views.
                fixed_M_a, fixed_WM_a, fixed_hidden_a, fixed_out_cos_a = [], [], [], []
                rel_step_a, is_fixed_a = [], []
                for ai, x_stack in enumerate(stacked):
                    # x_stack is already a device tensor (n_stim, T, n_input).
                    _, _, db_i = net.iterate_sequence_batch(
                        x_stack, run_mode="track_states", save_to_cpu=True, detach_saved=True)
                    M_all = np.asarray(db_i[f"M{layer_index}"])        # (n_stim,T,hid,emb)
                    T = M_all.shape[1]
                    t_seed = min(pe, T - 1)                              # end-of-period M
                    t_mid = min((ps + pe) // 2, T - 1)                  # sustained input
                    init_M = M_all[:, t_seed, :, :]
                    const_input = np.asarray(x_stack.detach().cpu())[:, t_mid, :]

                    fixed_M, _, final_speeds = find_modulation_fixed_points(
                        net, init_M, const_input, steps=20000, learningRate=1e-3,
                        printPeriod=100000, loss_tol=1e-8, lbfgs_steps=2000,
                        device=device)
                    # Shared derivation of W⊙M*, hidden(M*), cos-out, and rel_step.
                    views = derive_fixed_point_views(net, fixed_M, const_input,
                                                     final_speeds, W_fp, device,
                                                     rel_tol=_REL_TOL)

                    fixed_M_a.append(fixed_M.astype(np.float32))
                    fixed_WM_a.append(views["fixed_WM"].astype(np.float32))
                    fixed_hidden_a.append(views["fixed_hidden"].astype(np.float32))
                    fixed_out_cos_a.append(views["fixed_out_cos"].astype(np.float32))
                    rel_step_a.append(views["rel_step"])
                    is_fixed_a.append(views["is_fixed"])
                    print(f"  [interp-fp/{sname}] alpha={_interp_alphas[ai]:.1f}: "
                          f"{int(views['is_fixed'].sum())}/{n_stim} converged "
                          f"(median rel_step {np.median(views['rel_step']):.2e})")

                interp_fp_data["results"][sname] = {
                    "period_title": _PERIOD_TITLE.get(sname, sname),
                    "period": (int(ps), int(pe)),
                    "fixed_M": np.stack(fixed_M_a),            # (n_alpha, n_stim, hid, emb)
                    "fixed_WM": np.stack(fixed_WM_a),
                    "fixed_hidden": np.stack(fixed_hidden_a),  # (n_alpha, n_stim, hidden)
                    "fixed_out_cos": np.stack(fixed_out_cos_a),  # (n_alpha, n_stim)
                    "rel_step": np.stack(rel_step_a),          # (n_alpha, n_stim)
                    "is_fixed": np.stack(is_fixed_a),
                }

            interp_fp_path = save_dir / f"interp_fixed_points_{aname}.pkl"
            with open(interp_fp_path, "wb") as f:
                pickle.dump(interp_fp_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("  Saved data: " + str(interp_fp_path))
        except Exception as exc:
            print(f"  [interp-fp] failed: {exc}")
            import traceback
            traceback.print_exc()

    print(f"All figures saved to {save_dir}/")


def _discover_anames():
    """Return all experiment identifiers (savednet_*.pt) under twotasks/, sorted
    by modification time (oldest first). Searches per-run subfolders
    (twotasks/{aname}/savednet_*.pt) as well as the legacy flat layout, and
    de-duplicates by identifier."""
    results = sorted(OUT_DIR.glob("*/savednet_*.pt"), key=lambda p: p.stat().st_mtime)
    results += sorted(OUT_DIR.glob("savednet_*.pt"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise FileNotFoundError("No savednet_*.pt found in ./twotasks/. Run two_task.py first.")
    seen, anames = set(), []
    for p in results:
        a = p.name[len("savednet_"):-len(".pt")]
        if a not in seen:
            seen.add(a)
            anames.append(a)
    return anames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aname", type=str, default=None,
                        help="Experiment identifier. Omit to analyze ALL runs in ./twotasks/.")
    parser.add_argument("--fp-n-seeds", type=int, default=5,
                        help="Number of random trial templates to try when solving "
                             "gradient fixed points (per rule); the best-converging "
                             "one is kept (default 5).")
    parser.add_argument("--interp-n-alpha", type=int, default=10,
                        help="Number of pro<->anti interpolation intervals for the "
                             "task-interpolation fixed points; yields n+1 alpha steps "
                             "in [0,1]. Cost scales linearly with this (one FP solve "
                             "per alpha per period). Default 10 (0.0, 0.1, ... 1.0).")
    parser.add_argument("--no-fixed-points", dest="run_fixed_points",
                        action="store_false",
                        help="Skip the time-consuming gradient fixed-point solvers "
                             "(per-rule fixed_points_grad_* and the task-interpolation "
                             "fixed points). On by default.")
    parser.set_defaults(run_fixed_points=True)
    args = parser.parse_args()

    anames = [args.aname] if args.aname else _discover_anames()
    print(f"Analyzing {len(anames)} run(s).")
    for a in anames:
        print(f"\n── Analyzing: {a} ──")
        try:
            main(a, fp_n_seeds=args.fp_n_seeds,
                 interp_n_alpha=args.interp_n_alpha,
                 run_fixed_points=args.run_fixed_points)
        except Exception as exc:
            print(f"  FAILED {a}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
