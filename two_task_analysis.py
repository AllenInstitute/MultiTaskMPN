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
from scipy.spatial import ConvexHull

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

import networks as nets
import net_helpers
import mpn_tasks
import helper
import mpn

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


def modulation_extraction(test_input, db, layer_index, cuda=False):
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


def l2_consecutive_diff(X):
    """Per-time-step L2 difference of an (T, N) trajectory. (Cell 84.)"""
    d = np.diff(X, axis=0)
    return np.linalg.norm(d, axis=1)


def input_interpolation(test_input_long, test_output_long, label_task_comb_long, expand_stimulus=True):
    """Build alpha-interpolations between pro and anti task inputs. (Cell 81.)"""
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

    n = 20
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


def ring_volume_nd(pts):
    T, D = pts.shape
    if T <= D:
        raise ValueError(f"Need at least D+1={D+1} non-coplanar points, got {T}.")
    hull = ConvexHull(pts)
    return hull.volume


def ring_volume_3d(pts):
    if pts.shape[1] != 3:
        raise ValueError("ring_volume_3d expects a 3-D point set.")
    hull = ConvexHull(pts)
    return hull.volume


def add_convex_hull_mesh(fig, x, y, z, row=None, col=None, name="Endpoint hull",
                         mesh_opacity=0.15, mesh_color="black", showlegend=True):
    """Add a 3D convex-hull mesh to a Plotly figure. (Cell 95.)"""
    pts = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
    if pts.shape[0] < 4:
        return
    try:
        hull = ConvexHull(pts)
    except Exception as e:
        print(f"ConvexHull failed: {e}")
        return
    tri = hull.simplices
    mesh = go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                     i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                     opacity=mesh_opacity, color=mesh_color, name=name, showlegend=showlegend)
    if row is not None and col is not None:
        fig.add_trace(mesh, row=row, col=col)
    else:
        fig.add_trace(mesh)


def best_fit_plane_normal(pts, center=True, eps=1e-12):
    """Unit normal of the best-fit plane to pts (N,3), via SVD. (Cell 96.)"""
    pts = np.asarray(pts, float)
    if pts.shape[0] < 3:
        return None
    X = pts - pts.mean(axis=0) if center else pts.copy()
    if np.linalg.norm(X) < eps:
        return None
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s.size < 3 and np.all(s < eps):
        return None
    n = Vt[-1]
    n_norm = np.linalg.norm(n)
    if n_norm < eps:
        return None
    return n / n_norm


def angle_between_unit_vectors(u, v, degrees=True, eps=1e-12):
    if u is None or v is None:
        return np.nan
    c = float(np.clip(np.abs(np.dot(u, v)), -1.0, 1.0))
    ang = np.arccos(c)
    return np.degrees(ang) if degrees else ang


def normalize_lst(lst, value=None):
    """Normalize a list by its first (or a given) value. (Cell 92.)"""
    if value is None:
        value = lst[0]
    return [val_ / value for val_ in lst]


# ═══════════════════════════════════════════════════════════════════════════
# Network reload
# ═══════════════════════════════════════════════════════════════════════════
def _rebuild_net(net_params, device):
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


def main(aname):
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

    net = _rebuild_net(net_params, device)
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

    # Figures are written into the same per-run subfolder that holds the
    # checkpoint / bundle / param. Clear only previously-generated figure outputs
    # (.png / .html) so re-running is clean without deleting the saved inputs.
    save_dir = OUT_DIR / aname
    save_dir.mkdir(parents=True, exist_ok=True)
    for _old in save_dir.iterdir():
        if _old.is_file() and _old.suffix in (".png", ".html"):
            _old.unlink()

    def fp(stem):
        """Figure path under the per-run directory."""
        return str(save_dir / stem)

    # ── Final-stage network output / db (notebook cell 12) ───────────────────
    ind = -1
    net_out = netout_lst[0][ind]
    db = db_lst[0][ind]

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 7: input-embedding Frobenius norm across training (input_layer_add)
    # ═════════════════════════════════════════════════════════════════════════
    if hyp_dict['chosen_network'] == "dmpn" and net_params["input_layer_add"]:
        fignorm, axsnorm = plt.subplots(1, 1, figsize=(4, 4))
        axsnorm.plot(counter_lst, [np.linalg.norm(Wm) for Wm in Winput_lst], "-o")
        axsnorm.set_xscale("log")
        axsnorm.set_ylabel("Frobenius Norm")
        fignorm.savefig(fp(f"winput_norm_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"winput_norm_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
        plt.close(fignorm)

    # Cell 8: final input-embedding heatmap
    if net_params["input_layer_add"]:
        Win_end = Winput_lst[-1]
        fig8, ax8 = plt.subplots(1, 1, figsize=(4, 4))
        sns.heatmap(Win_end, ax=ax8, center=0, cmap="coolwarm")
        fig8.savefig(fp(f"winput_end_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"winput_end_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
        plt.close(fig8)

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 13: plot_input_output  +  cells 14-18 (normal + 4 long variants)
    # ═════════════════════════════════════════════════════════════════════════
    def plot_input_output(test_input_np_, net_out_, test_output_np_, test_task_=None, tag="", batch_num=5, label=None):
        test_input_np_ = helper.to_ndarray(test_input_np_)
        net_out_ = helper.to_ndarray(net_out_)
        test_output_np_ = helper.to_ndarray(test_output_np_)

        fig_all, axs_all = plt.subplots(batch_num, 2, figsize=(4 * 2, batch_num * 2))

        if test_output_np_.shape[-1] == 1:
            for batch_idx, ax in enumerate(axs_all):
                ax.plot(net_out_[batch_idx, :, 0], color=c_vals[batch_idx])
                ax.plot(test_output_np_[batch_idx, :, 0], color=c_vals_l[batch_idx])
        else:
            for batch_idx in range(batch_num):
                for out_idx in range(test_output_np_.shape[-1]):
                    axs_all[batch_idx, 0].plot(net_out_[batch_idx, :, out_idx], color=c_vals[out_idx], label=out_idx)
                    axs_all[batch_idx, 0].plot(test_output_np_[batch_idx, :, out_idx], color=c_vals_l[out_idx], linewidth=5, alpha=0.5)
                    if test_task_ is not None:
                        outname = f"{task_params['rules'][test_task_[batch_idx]]}; {tag}"
                        axs_all[batch_idx, 0].set_title(outname)
                    axs_all[batch_idx, 0].legend()

                input_batch = test_input_np_[batch_idx, :, :]
                if task_params["randomize_inputs"]:
                    input_batch = input_batch @ np.linalg.pinv(task_params["randomize_matrix"])
                for inp_idx in range(input_batch.shape[-1]):
                    axs_all[batch_idx, 1].plot(input_batch[:, inp_idx], color=c_vals[inp_idx], label=inp_idx)
                    if test_task_ is not None:
                        axs_all[batch_idx, 1].set_title(f"{task_params['rules'][test_task_[batch_idx]]}; {tag}")
                    axs_all[batch_idx, 1].legend()

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
    # ═════════════════════════════════════════════════════════════════════════
    dpca_name = "hidden"
    try:
        from dPCA import dPCA

        activity_list = []
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
            all_hidden = db_lst[0][-1][f"hidden{layer_index}"]
            hidden1 = all_hidden[mod1_j].T
            hidden2 = all_hidden[mod2_j].T
            activity_list.append([hidden1, hidden2])

        N, k = activity_list[0][0].shape
        S = 8
        T = 2
        data_mean = np.zeros((N, S, T, k))
        for s in range(S):
            for t in range(T):
                data_mean[:, s, t, :] = activity_list[s][t]
        data_trials = data_mean[None, ...]

        dpca = dPCA.dPCA(labels='srt', n_components=100)
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

        T_delayend = time_stamps_usual["delay_end"] - 1
        h1 = (W + W * mod1_stim1[T_delayend]) @ (W_in @ x_fix_on_all[T_delayend])
        h2 = (W + W * mod2_stim1[T_delayend]) @ (W_in @ x_fix_on_all[T_delayend])

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

    # ═════════════════════════════════════════════════════════════════════════
    # Cells 43-47: cross-task / cross-period PCA (figure2A_pca_fve)
    # ═════════════════════════════════════════════════════════════════════════
    H = db[f"hidden{layer_index}"]
    M = db[f"M{layer_index}"].reshape(db[f"M{layer_index}"].shape[0], db[f"M{layer_index}"].shape[1], -1)
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
    res_M = figure2A_pca_fve(M, task_id, periods_, k=top_k, max_pcs=10, center="None")

    data_all = [["hidden", res_H], ["modulation", res_M]]
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
        fig47.tight_layout()
        fig47.savefig(fp(f"d_combine_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"d_combine_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png")))
        plt.close(fig47)

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
    fig49, axs49 = plt.subplots(1, 3, figsize=(4 * 3, 4))
    for idx, Wcomb in enumerate(Wcombs):
        C = np.corrcoef(Wcomb, rowvar=False)
        C_upper = C.copy()
        C_upper[np.tril_indices_from(C_upper, k=-1)] = np.nan
        sns.heatmap(C_upper, cmap="coolwarm", ax=axs49[idx],
                    xticklabels=input_label if idx != 1 else False,
                    yticklabels=input_label if idx != 1 else False,
                    annot=True if idx != 1 else False, fmt=".2f", vmin=-1.0, vmax=1.0)
    axs49[0].set_title(r"$W_{\mathrm{in}}$", fontsize=12)
    axs49[1].set_title(r"$W$", fontsize=12)
    axs49[2].set_title(r"$WW_{\mathrm{in}}$", fontsize=12)
    fig49.tight_layout()
    fig49.savefig(fp(f"w_stim_corr_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"w_stim_corr_{hyp_dict['ruleset']}_seed{seed}_{name}_{hyp_dict['addon_name']}.png")))
    plt.close(fig49)

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

        figae.suptitle(stage_name, fontsize=15)
        figae.tight_layout()
        figae.savefig(fp(f"attractor_stage{save_name}_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"attractor_stage{save_name}_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
        plt.close(figae)

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
    # Cell 64: delta_M / delta_MW magnitude over time
    # ═════════════════════════════════════════════════════════════════════════
    m_save = modulation_save_time[-1]
    delta_M_tnorm_all, delta_MW_tnorm_all = [], []
    for i in range(8):
        mod1_stim1 = m_save[0][i]
        mod2_stim1 = m_save[1][i]
        delta_M = mod1_stim1 - mod2_stim1
        delta_MW = (mod1_stim1 * W[None, :, :] - mod2_stim1 * W[None, :, :])
        delta_M_tnorm_all.append([np.linalg.norm(delta_M[t]) for t in range(delta_M.shape[0])])
        delta_MW_tnorm_all.append([np.linalg.norm(delta_MW[t]) for t in range(delta_MW.shape[0])])
    delta_M_tnorm_all = np.array(delta_M_tnorm_all)
    delta_MW_tnorm_all = np.array(delta_MW_tnorm_all)

    fig64, ax64 = plt.subplots(1, 1, figsize=(6, 2))
    m1 = np.mean(delta_M_tnorm_all, axis=0)
    s1 = np.std(delta_M_tnorm_all, axis=0)
    m2 = np.mean(delta_MW_tnorm_all, axis=0)
    s2 = np.std(delta_MW_tnorm_all, axis=0)
    x = np.arange(m1.shape[0])
    ax64.plot(x, m1, color=c_vals[0], label="delta_M")
    ax64.fill_between(x, m1 - s1, m1 + s1, color=c_vals_l[0], alpha=0.2)
    ax64.plot(x, m2, color=c_vals[1], label="delta_MW")
    ax64.fill_between(x, m2 - s2, m2 + s2, color=c_vals_l[1], alpha=0.2)
    ax64.set_yscale("log")
    for kkey in ["fixation_end", "stimulus_end", "delay_end", "trial_end"]:
        ax64.axvline(time_stamps_usual[kkey], linestyle="--", color=c_vals[2])
    ax64.set_xlabel("Timestep", fontsize=12)
    ax64.set_ylabel("Magnitude", fontsize=12)
    fig64.tight_layout()
    fig64.savefig(fp(f"m_magnitude_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"m_magnitude_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig64)

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 66: modulation magnitude vs W magnitude
    # ═════════════════════════════════════════════════════════════════════════
    time_cut = ["fixation_end", "stimulus_end", "delay_end", "trial_end"]
    fig66, axs66 = plt.subplots(2, 4, figsize=(4 * 4, 4 * 2))
    if net_params["input_layer_add"]:
        W = net.mp_layer1.W.data.detach().cpu().numpy()
    else:
        W = net.mp_layer0.W.data.detach().cpu().numpy()
    for t, time in enumerate(time_cut):
        for i in range(1):
            mod1_stim1 = m_save[0][i]
            mod2_stim1 = m_save[1][i]
            Mmod1 = mod1_stim1[time_stamps_usual[time]]
            Mmod2 = mod2_stim1[time_stamps_usual[time]]
            axs66[0, t].scatter(W.flatten(), np.abs(Mmod1).flatten(), alpha=0.1, c=c_vals[0])
            x1, y1 = bin_by_sorted_x(W.flatten(), np.abs(Mmod1).flatten())
            axs66[0, t].plot(x1, y1, "-o", c=c_vals[1])
            axs66[1, t].scatter(W.flatten(), np.abs(Mmod2).flatten(), alpha=0.1, c=c_vals[0])
            x2, y2 = bin_by_sorted_x(W.flatten(), np.abs(Mmod2).flatten())
            axs66[1, t].plot(x2, y2, "-o", c=c_vals[1])
    for ax in axs66.flatten():
        ax.set_xlabel("W Entry", fontsize=15)
        ax.set_ylabel(f"Abs(M) Entry at {time}", fontsize=15)
    fig66.tight_layout()
    fig66.savefig(fp(f"m_w_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"m_w_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fig66)

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

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 77: component cosine-similarity diagnostics (same stim / same resp)
    # ═════════════════════════════════════════════════════════════════════════
    input1 = db_lst[0][-1]["input1"] if net_params["input_layer_add"] else db_lst[0][-1][f"input{layer_index}"]
    hidden1 = db_lst[0][-1][f"hidden{layer_index}"]
    W = net.mp_layer1.W.data.detach().cpu().numpy() if net_params["input_layer_add"] else net.mp_layer0.W.data.detach().cpu().numpy()
    W_in = Winput_lst[-1]
    _, M_all, h_end, _ = modulation_extraction(test_input, db_lst[0][-1], layer_index)

    fig77, axs77 = plt.subplots(12, 1, figsize=(6, 13 * 2))
    fignorm2, axsnorm2 = plt.subplots(1, 1, figsize=(10, 3))
    figMdiff, axsMdiff = plt.subplots(1, 1, figsize=(10, 3))
    maps = [lambda i: i, lambda i: (i + 4) % 8]
    map_names = ["Same Stim", "Same Resp"]
    figcc, axscc = plt.subplots(8, 2, figsize=(10 * 2, 4 * 8))
    cc_all = [[], []]

    for idx, map_ in enumerate(maps):
        for k in range(8):
            ind1 = [i for i, lst in enumerate(label_task_comb) if np.array_equal(lst, [k, 0])][0]
            ind2 = [i for i, lst in enumerate(label_task_comb) if np.array_equal(lst, [map_(k), 1])][0]
            input1_, input2_ = input1[ind1], input1[ind2]
            hidden1_, hidden2_ = hidden1[ind1], hidden1[ind2]
            cosines = [[], [], [], [], [], [], [], [], [], [], [], []]
            norm_ratios = [[], []]
            M_diff = []
            component_compare_all = []
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

                A = M_tp1 * W
                B = M_tp2 * W

                def _fro_norm(X, eps=1e-12):
                    return np.linalg.norm(X, ord="fro") + eps

                fro_inner = float(np.sum(A * B))
                fro_cos = fro_inner / (_fro_norm(A) * _fro_norm(B))

                i1 = np.asarray(i1).reshape(-1, 1)
                i2 = np.asarray(i2).reshape(-1, 1)
                dirc_inner = float((i1.T @ A.T @ B @ i2)[0, 0])
                den1 = float((i1.T @ A.T @ A @ i1)[0, 0])
                den2 = float((i2.T @ B.T @ B @ i2)[0, 0])
                dirc_cos = dirc_inner / (np.sqrt(den1 + 1e-12) * np.sqrt(den2 + 1e-12))

                x1, _, _, _ = np.linalg.lstsq(W_in, i1, rcond=None)
                x2, _, _, _ = np.linalg.lstsq(W_in, i2, rcond=None)

                component_compare = []
                for nidx in range(W.shape[1]):
                    proj1MWxd = ((M_tp1[:, nidx] * W[:, nidx]) * i1[nidx])
                    component_compare.append(cosine_sim(proj1MWxd.flatten(), proj2MWx.flatten()))
                component_compare_all.append(component_compare)

                proj1WMWx, proj2WMWx = (((W + M_tp1 * W) @ i1)).flatten(), (((W + M_tp2 * W) @ i2)).flatten()
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
                norm_ratios[0].append(np.linalg.norm(proj1MWx) / np.linalg.norm(proj1WMWx))
                norm_ratios[1].append(np.linalg.norm(proj2MWx) / np.linalg.norm(proj2WMWx))
                M_diff.append(np.linalg.norm(proj1M - proj2M))

            for u in range(len(cosines)):
                label = map_names[idx] if k == 0 else None
                axs77[u].plot(cosines[u], color=c_vals[idx], linestyle="-", label=label)
            axsMdiff.plot(M_diff, color=c_vals[idx], linestyle="-")
            if idx == 0:
                for u in range(len(norm_ratios)):
                    axsnorm2.plot(norm_ratios[u], color=c_vals[idx], linestyle=linestyles[u])

            component_compare_all = np.array(component_compare_all)
            cc_all[idx].append(component_compare_all)
            sns.heatmap(component_compare_all, ax=axscc[k, idx], cmap="coolwarm")
            axscc[k, idx].axhline(y=time_stamps_usual["stimulus_end"], linewidth=2)
            axscc[k, idx].axhline(y=time_stamps_usual["delay_end"], linewidth=2)

    for idx, ax in enumerate(axs77):
        ax.set_xlabel("Time Steps", fontsize=15)
        ax.axvline(time_stamps_usual["delay_end"], color=c_vals[2])
        if idx not in (7, 8):
            ax.set_ylim([-1.1, 1.1])
        ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.2)
        ax.tick_params(axis="both", which="minor", labelsize=14, length=3, width=1.0)
        ax.axhline(0.0, color=c_vals[3], linestyle="--")
        ax.legend(fontsize=12, frameon=True, loc="upper left")
    axsnorm2.set_xlabel("Time Steps", fontsize=15)
    axsnorm2.axvline(time_stamps_usual["delay_end"], color=c_vals[2])
    axsnorm2.set_ylim([-1.1, 1.1])
    axsMdiff.set_xlabel("Time Steps", fontsize=15)
    axsMdiff.axvline(time_stamps_usual["delay_end"], color=c_vals[2])
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
    axsnorm2.set_ylabel(r"$||z^{\text{mod}}||/||z||$", fontsize=15)
    fignorm2.tight_layout()
    fignorm2.savefig(fp(f"zmodz_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"zmodz_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(fignorm2)
    axsMdiff.set_ylabel(r"$||M_{t-1}^A-M_{t-1}^B||$", fontsize=15)
    figMdiff.tight_layout()
    figMdiff.savefig(fp(f"mdiff_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"mdiff_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(figMdiff)
    figcc.tight_layout()
    figcc.savefig(fp(f"MWxcomponent_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
    print("  Saved figure: " + str(fp(f"MWxcomponent_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
    plt.close(figcc)

    # Cell 78: response-aligned component summaries
    timemarkers = [["delay_end", "trial_end"]]
    for ts, te in timemarkers:
        allcomps = [[], []]
        for i in range(8):
            samestim, sameresp = cc_all[0][i], cc_all[1][i]
            samestim = samestim[time_stamps_usual[ts]:time_stamps_usual[te], :]
            sameresp = sameresp[time_stamps_usual[ts]:time_stamps_usual[te], :]
            allcomps[0].append(np.mean(samestim, axis=0))
            allcomps[1].append(np.mean(sameresp, axis=0))
        fig78, axs78 = plt.subplots(2, 1, figsize=(6, 2 * 2))
        sns.heatmap(np.array(allcomps[0]), ax=axs78[0], cmap="coolwarm")
        sns.heatmap(np.array(allcomps[1]), ax=axs78[1], cmap="coolwarm")
        for ax in axs78:
            ax.set_xlabel("MPN Input Neuron Index", fontsize=15)
            ax.set_ylabel("Stimulus Type", fontsize=15)
        fig78.tight_layout()
        fig78.savefig(fp(f"mpninput_resp_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"mpninput_resp_{compare_value}_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png")))
        plt.close(fig78)
        samestimavg = np.mean(np.array(allcomps[0]), axis=0)
        samerespavg = np.mean(np.array(allcomps[1]), axis=0)
        print("cos(samestimavg, samerespavg):", cosine_sim(samestimavg, samerespavg))

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

    # Cell 85: per-time-step L2 diff of hidden / modulation
    for time_stamp_long, test_input_long, sname, db_index, end_time_name, _ in time_stamp_input_map:
        end_time = time_stamp_long[end_time_name]
        _, M_, h_, _ = modulation_extraction(test_input_long, db_lst[db_index][-1], layer_index)
        M_ = M_.reshape(M_.shape[0], M_.shape[1], -1)
        diff_h_all = np.array([l2_consecutive_diff(h_[bs_]) for bs_ in range(h_.shape[0])])
        diff_M_all = np.array([l2_consecutive_diff(M_[bs_]) for bs_ in range(M_.shape[0])])
        fig85, axs85 = plt.subplots(1, 2, figsize=(4 * 2, 4))
        axs85[0].plot(np.mean(diff_h_all, axis=0), color=c_vals[0])
        axs85[1].plot(np.mean(diff_M_all, axis=0), color=c_vals[0])
        for ax in axs85:
            ax.set_xlabel("Time Steps", fontsize=12)
            ax.set_yscale("log")
            ax.axvline(end_time, color=c_vals[1], linewidth=2)
            ax.set_title(sname, fontsize=15)
        axs85[0].set_ylabel("Hidden Per-Time-Step L2 Diff", fontsize=12)
        axs85[1].set_ylabel("Modulation Per-Time-Step L2 Diff", fontsize=12)
        fig85.tight_layout()
        fig85.savefig(fp(f"l2_timee_diff_seed{seed}_{hyp_dict['addon_name']}_{sname}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"l2_timee_diff_seed{seed}_{hyp_dict['addon_name']}_{sname}.png")))
        plt.close(fig85)

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 86: PCA trajectories of hidden / modulation per variant
    # ═════════════════════════════════════════════════════════════════════════
    projected_data_all = []
    zeros_pca = None
    wout_proj = None
    batch_num = None
    for time_stamp_long, test_input_long, sname, db_index, _, label_task_comb_long in time_stamp_input_map:
        print(f"sname: {sname}; db_index: {db_index}")
        names = ["hidden", "modulation"]
        for name in names:
            fighs, axshs = plt.subplots(1, 3, figsize=(5 * 3, 5 * 1))
            PCA_downsample = 3
            Ms, Ms_orig, hs, bs = modulation_extraction(test_input_long, db_lst[db_index][-1], layer_index)
            batch_num = Ms_orig.shape[0]
            if name == "modulation":
                data = Ms
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
            if db_index == 0:
                projected_data_all.append(projected_data)
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
                task = label_task_comb_longdelay[i, 1]
                if task not in (0, 1):
                    continue
                color = c_vals[label_task_comb_longdelay[i, 0]]
                ls = linestyles[task]
                data_i = projected_data[i]
                seg = slice(stim0, trial_end)
                for ax, (a, bb) in zip(axshs, combination):
                    ax.plot(data_i[seg, a], data_i[seg, bb], c=color, linestyle=ls, alpha=0.01)
                    for _, t0_key, t1_key, mk_idx in phases:
                        sl = slice(time_stamp_long[t0_key], time_stamp_long[t1_key])
                        ax.scatter(data_i[sl, a], data_i[sl, bb], c=color, marker=markers_vals[mk_idx],
                                   alpha=0.01 if sname != "normal" else 0.1)
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

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 88: 3D PCA trajectories + readout plane (Plotly -> HTML) + projection
    # ═════════════════════════════════════════════════════════════════════════
    for indd, projected_data in enumerate(projected_data_all):
        fig = go.Figure()
        for i in range(batch_num):
            data_batch = projected_data[i, :, :]
            fig.add_trace(go.Scatter3d(x=data_batch[:, 0], y=data_batch[:, 1], z=data_batch[:, 2],
                                       mode="lines", line=dict(width=2, color=c_vals[label_task_comb[i, 0]]),
                                       opacity=0.5, showlegend=False))
        fig.add_trace(go.Scatter3d(x=[zeros_pca[0, 0]], y=[zeros_pca[0, 1]], z=[zeros_pca[0, 2]],
                                   mode="markers", marker=dict(size=4, color="black"), showlegend=False))
        zero_pt = zeros_pca[0]
        v1 = wout_proj[0, :]
        v2 = wout_proj[1, :]
        traj_pts = projected_data[:, :, :].reshape(-1, 3)
        plane_half = 0.5 * np.linalg.norm(traj_pts - zero_pt, axis=1).max()
        u_hat = v1 / np.linalg.norm(v1)
        v2_proj = v2 - v2.dot(u_hat) * u_hat
        v_hat = v2_proj / np.linalg.norm(v2_proj)
        corners = np.array([-plane_half * u_hat - plane_half * v_hat,
                            plane_half * u_hat - plane_half * v_hat,
                            plane_half * u_hat + plane_half * v_hat,
                            -plane_half * u_hat + plane_half * v_hat])
        fig.add_trace(go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                                i=[0, 0], j=[1, 2], k=[2, 3], opacity=0.25, color="lightblue",
                                name="spanning plane", showscale=False))
        fig.update_layout(scene=dict(xaxis_title="PCA 1", yaxis_title="PCA 2", zaxis_title="PCA 3"),
                          width=600, height=600, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)

        response_half = int((time_stamps_usual["trial_end"] - time_stamps_usual["delay_end"]) / 2) + time_stamps_usual["delay_end"]
        endpoints = projected_data[:, response_half + 1, :]
        figproj, axproj = plt.subplots(1, 1, figsize=(4, 4))
        for ei in range(endpoints.shape[0]):
            endpoint = endpoints[ei, :] - zero_pt
            u_coord = endpoint.dot(u_hat)
            v_coord = endpoint.dot(v_hat)
            color_index = label_task_comb[ei, 0]
            task = label_task_comb[ei, 1]
            marker = "o" if task == 0 else "x"
            axproj.scatter(u_coord, v_coord, c=c_vals[color_index], marker=marker, alpha=0.5)
        figproj.tight_layout()
        figproj.savefig(fp(f"m_pca_readoutplane_seed{seed}_{hyp_dict['addon_name']}_{indd}.png"), dpi=300)
        print("  Saved figure: " + str(fp(f"m_pca_readoutplane_seed{seed}_{hyp_dict['addon_name']}_{indd}.png")))
        plt.close(figproj)

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

    for siindex, stacked_interpolation_ in enumerate(stacked_interpolation_lst):
        sname = stacked_interpolation_name_lst[siindex]
        print(f"sname: {sname}")
        names = ["hidden", "modulation", "w_modulation"]
        raw_data_ring = [[], [], []]
        raw_data_ring_magnitude = [[], [], []]
        projected_data_ring = [[], [], []]

        for nindex, name in enumerate(names):
            fighs, axshs = plt.subplots(1, 3, figsize=(5 * 3, 5 * 1))
            fighsadd, axshsadd = plt.subplots(1, 3, figsize=(5 * 3, 5 * 1))
            fig3dfix = go.Figure()
            combination = [[0, 1], [0, 2], [1, 2]]
            interpolation_label = [i for i in range(len(stacked_interpolation_[0]))]

            def numbered_markers(n):
                return [f'${i}$' for i in range(n)]

            marker_new = numbered_markers(len(stacked_interpolation_))
            projected_data_fix_all = []
            pca_delay = None

            for (int_index, int_input) in enumerate(stacked_interpolation_):
                if int_index == 0 and nindex == 0:
                    int_input_all.append(int_input)

                stack_output, _, db_intp = net.iterate_sequence_batch(
                    int_input, run_mode='track_states', save_to_cpu=True, detach_saved=True)
                Ms, Ms_orig, hs, bs = modulation_extraction(int_input, db_intp, layer_index, cuda=True)
                batch_num = Ms_orig.shape[0]

                if name == "hidden":
                    data = hs
                elif name == "modulation":
                    data = Ms
                elif name == "w_modulation":
                    Wlocal = net.mp_layer1.W.data.detach().cpu().numpy()
                    data = (Ms_orig * Wlocal[None, None, :, :]).reshape(Ms.shape[0], Ms.shape[1], -1)

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

                for i in range(batch_num):
                    data_batch = projected_data_fix[i, :]
                    for index, comb in enumerate(combination):
                        marker_value = marker_new[int_index] if int_index == 0 or int_index == len(stacked_interpolation_) - 1 else "o"
                        alpha_value = 0.1 if marker_value == "o" else 1.0
                        axshs[index].scatter(data_batch[comb[0]], data_batch[comb[1]], c=c_vals[interpolation_label[i]],
                                             marker=marker_value, alpha=alpha_value)
                        axshs[index].set_xlabel(f"PCA {comb[0]+1}; Anti {sname}", fontsize=15)
                        axshs[index].set_ylabel(f"PCA {comb[1]+1}; Anti {sname}", fontsize=15)

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

            fighs.suptitle(f"name: {name}; sname: {sname}", fontsize=20)
            fighs.tight_layout()
            fighs.savefig(fp(f"m_pca_attractor_{name}_seed{seed}_{hyp_dict['addon_name']}_{int_index}_{sname}.png"), dpi=300)
            print("  Saved figure: " + str(fp(f"m_pca_attractor_{name}_seed{seed}_{hyp_dict['addon_name']}_{int_index}_{sname}.png")))
            plt.close(fighs)
            fighsadd.suptitle(f"name: {name}; sname: {sname}", fontsize=20)
            fighsadd.tight_layout()
            fighsadd.savefig(fp(f"m_pca_attractor_cycle_{name}_seed{seed}_{hyp_dict['addon_name']}_{int_index}_{sname}.png"), dpi=300)
            print("  Saved figure: " + str(fp(f"m_pca_attractor_cycle_{name}_seed{seed}_{hyp_dict['addon_name']}_{int_index}_{sname}.png")))
            plt.close(fighsadd)
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
    # Cells 98-101: stimulus-PCA + readout response trajectories (Plotly -> HTML)
    # ═════════════════════════════════════════════════════════════════════════
    def traj(name):
        for siindex, stacked_interpolation in enumerate(stacked_interpolation_lst):
            sname = stacked_interpolation_name_lst[siindex]
            print(sname)
            fig3dresponse = go.Figure()
            N = len(stacked_interpolation)
            wout = net.W_output.detach().cpu().numpy()
            anti_go = [stacked_interpolation[0], stacked_interpolation[int((N + 1) / 2)], stacked_interpolation[-1]]
            _, _, db_intp_anti = net.iterate_sequence_batch(anti_go[0], run_mode="track_states", save_to_cpu=True, detach_saved=True)
            _, _, db_intp_go = net.iterate_sequence_batch(anti_go[2], run_mode="track_states", save_to_cpu=True, detach_saved=True)
            Ms_anti, Ms_orig_anti, hs_anti, bs_anti = modulation_extraction(int_input_all[siindex], db_intp_anti, layer_index)
            Ms_go, Ms_orig_go, hs_go, bs_go = modulation_extraction(int_input_all[siindex], db_intp_go, layer_index)
            if name == "hidden":
                data_anti, data_go = hs_anti, hs_go
            elif name == "modulation":
                data_anti, data_go = Ms_anti, Ms_go
            elif name == "w_modulation":
                Wlocal = net.mp_layer1.W.data.detach().cpu().numpy()
                data_anti = (Ms_orig_anti * Wlocal[None, None, :, :]).reshape(Ms_anti.shape[0], Ms_anti.shape[1], -1)
                data_go = (Ms_orig_go * Wlocal[None, None, :, :]).reshape(Ms_go.shape[0], Ms_go.shape[1], -1)
            n_activity = data_anti.shape[-1]
            as_flat_stim = data_anti[:, time_stamps_lst[siindex]["stimulus_start"]:time_stamps_lst[siindex]["stimulus_end"], :].reshape((-1, n_activity))
            data_anti_ = data_anti[:, desire_period[siindex][0]:desire_period[siindex][1], :]
            data_go_ = data_go[:, desire_period[siindex][0]:desire_period[siindex][1], :]
            as_flat_anti = data_anti_.reshape((-1, n_activity))
            as_flat_go = data_go_.reshape((-1, n_activity))
            pca_stim = PCA(n_components=PCA_downsample, random_state=42)
            pca_stim.fit(as_flat_stim)
            projected_data_anti = pca_stim.transform(as_flat_anti).reshape(data_anti_.shape[0], data_anti_.shape[1], -1)
            projected_data_go = pca_stim.transform(as_flat_go).reshape(data_go_.shape[0], data_go_.shape[1], -1)
            projected_data_stim_anti = projected_data_anti[:, :-1, :]
            projected_data_stim_go = projected_data_go[:, :-1, :]
            response_anti = -1 * hs_anti[:, desire_period[siindex][0]:desire_period[siindex][1], :] @ wout.T
            response_go = hs_go[:, desire_period[siindex][0]:desire_period[siindex][1], :] @ wout.T
            resp = 0
            for i in range(projected_data_stim_anti.shape[0]):
                fig3dresponse.add_trace(go.Scatter3d(x=projected_data_stim_anti[i, :, 0], y=projected_data_stim_anti[i, :, 1],
                                                     z=response_anti[i, :, resp + 1], mode="lines",
                                                     line=dict(width=8, color=c_vals[i]), name=f"Anti S{i}", showlegend=True))
                fig3dresponse.add_trace(go.Scatter3d(x=[projected_data_stim_anti[i, 0, 0]], y=[projected_data_stim_anti[i, 0, 1]],
                                                     z=[response_anti[i, 0, resp + 1]], mode="markers",
                                                     marker=dict(size=6, color=c_vals[i], symbol="circle-open"), showlegend=False))
                fig3dresponse.add_trace(go.Scatter3d(x=[projected_data_stim_anti[i, -1, 0]], y=[projected_data_stim_anti[i, -1, 1]],
                                                     z=[response_anti[i, -1, resp + 1]], mode="markers",
                                                     marker=dict(size=6, color=c_vals[i], symbol="circle"), showlegend=False))
                fig3dresponse.add_trace(go.Scatter3d(x=projected_data_stim_go[i, :, 0], y=projected_data_stim_go[i, :, 1],
                                                     z=response_go[i, :, resp + 1], mode="lines",
                                                     line=dict(width=8, color=c_vals[i], dash="dash"), name=f"Go S{i}", showlegend=True))
                fig3dresponse.add_trace(go.Scatter3d(x=[projected_data_stim_go[i, 0, 0]], y=[projected_data_stim_go[i, 0, 1]],
                                                     z=[response_go[i, 0, resp + 1]], mode="markers",
                                                     marker=dict(size=6, color=c_vals[i], symbol="diamond-open"), showlegend=False))
                fig3dresponse.add_trace(go.Scatter3d(x=[projected_data_stim_go[i, -1, 0]], y=[projected_data_stim_go[i, -1, 1]],
                                                     z=[response_go[i, -1, resp + 1]], mode="markers",
                                                     marker=dict(size=6, color=c_vals[i], symbol="diamond"), showlegend=False))
            x_anti = projected_data_stim_anti[:, -1, 0]
            y_anti = projected_data_stim_anti[:, -1, 1]
            z_anti = response_anti[:, -1, resp + 1]
            add_convex_hull_mesh(fig3dresponse, x_anti, y_anti, z_anti, name="Anti endpoint hull (cos)",
                                 mesh_opacity=0.18, mesh_color="black", showlegend=True)
            x_go = projected_data_stim_go[:, -1, 0]
            y_go = projected_data_stim_go[:, -1, 1]
            z_go = response_go[:, -1, resp + 1]
            add_convex_hull_mesh(fig3dresponse, x_go, y_go, z_go, name="Go endpoint hull (cos)",
                                 mesh_opacity=0.18, mesh_color="red", showlegend=True)
            ang_deg = angle_between_unit_vectors(
                best_fit_plane_normal(np.column_stack([x_anti, y_anti, z_anti])),
                best_fit_plane_normal(np.column_stack([x_go, y_go, z_go])), degrees=True)
            print(f"[{sname}] cosine: plane-normal angle = {ang_deg:.2f} deg")
            fig3dresponse.update_layout(template="plotly_white", width=1000, height=800,
                                        scene=dict(xaxis_title="Memoryanti Stimulus PCA 1",
                                                   yaxis_title="Memoryanti Stimulus PCA 2",
                                                   zaxis_title="cos θ", zaxis=dict(range=[-1.1, 1.1]), aspectmode="cube"))

    traj("hidden")
    traj("modulation")
    traj("w_modulation")

    # ═════════════════════════════════════════════════════════════════════════
    # Cells 102-105: endpoint planes across anti/median/go (Plotly -> HTML)
    # ═════════════════════════════════════════════════════════════════════════
    def traj_diff_alpha(name):
        for siindex, stacked_interpolation in enumerate(stacked_interpolation_lst):
            sname = stacked_interpolation_name_lst[siindex]
            print(sname)
            fig3dresponse_c = go.Figure()
            N = len(stacked_interpolation)
            wout = net.W_output.detach().cpu().numpy()
            anti_go = [stacked_interpolation[0], stacked_interpolation[int((N + 1) / 2)], stacked_interpolation[-1]]
            _, _, db_intp_anti = net.iterate_sequence_batch(anti_go[0], run_mode="track_states", save_to_cpu=True, detach_saved=True)
            _, _, db_intp_median = net.iterate_sequence_batch(anti_go[1], run_mode="track_states", save_to_cpu=True, detach_saved=True)
            _, _, db_intp_go = net.iterate_sequence_batch(anti_go[2], run_mode="track_states", save_to_cpu=True, detach_saved=True)
            Ms_anti, Ms_orig_anti, hs_anti, bs_anti = modulation_extraction(int_input_all[siindex], db_intp_anti, layer_index)
            Ms_median, Ms_orig_median, hs_median, bs_median = modulation_extraction(int_input_all[siindex], db_intp_median, layer_index)
            Ms_go, Ms_orig_go, hs_go, bs_go = modulation_extraction(int_input_all[siindex], db_intp_go, layer_index)
            if name == "hidden":
                data_anti, data_go, data_median = hs_anti, hs_go, hs_median
            elif name == "modulation":
                data_anti, data_go, data_median = Ms_anti, Ms_go, Ms_median
            elif name == "w_modulation":
                Wlocal = net.mp_layer1.W.data.detach().cpu().numpy()
                data_anti = (Ms_orig_anti * Wlocal[None, None, :, :]).reshape(Ms_anti.shape[0], Ms_anti.shape[1], -1)
                data_go = (Ms_orig_go * Wlocal[None, None, :, :]).reshape(Ms_go.shape[0], Ms_go.shape[1], -1)
                data_median = (Ms_orig_median * Wlocal[None, None, :, :]).reshape(Ms_median.shape[0], Ms_median.shape[1], -1)
            n_activity = data_anti.shape[-1]
            as_flat_stim = data_anti[:, time_stamps_lst[siindex]["stimulus_start"]:time_stamps_lst[siindex]["stimulus_end"], :].reshape((-1, n_activity))
            data_anti_ = data_anti[:, desire_period[siindex][0]:desire_period[siindex][1], :]
            data_go_ = data_go[:, desire_period[siindex][0]:desire_period[siindex][1], :]
            data_median_ = data_median[:, desire_period[siindex][0]:desire_period[siindex][1], :]
            as_flat_anti = data_anti_.reshape((-1, n_activity))
            as_flat_go = data_go_.reshape((-1, n_activity))
            as_flat_median = data_median_.reshape((-1, n_activity))
            pca_stim = PCA(n_components=PCA_downsample, random_state=42)
            pca_stim.fit(as_flat_stim)
            projected_data_anti = pca_stim.transform(as_flat_anti).reshape(data_anti_.shape[0], data_anti_.shape[1], -1)
            projected_data_go = pca_stim.transform(as_flat_go).reshape(data_go_.shape[0], data_go_.shape[1], -1)
            projected_data_median = pca_stim.transform(as_flat_median).reshape(data_median_.shape[0], data_median_.shape[1], -1)
            projected_data_stim_anti = projected_data_anti[:, :-1, :]
            projected_data_stim_go = projected_data_go[:, :-1, :]
            projected_data_stim_median = projected_data_median[:, :-1, :]
            response_anti = hs_anti[:, :desire_period[siindex][1], :] @ wout.T
            response_go = hs_go[:, :desire_period[siindex][1], :] @ wout.T
            response_median = hs_median[:, :desire_period[siindex][1], :] @ wout.T
            resp = 0
            for i in range(projected_data_stim_anti.shape[0]):
                fig3dresponse_c.add_trace(go.Scatter3d(x=[projected_data_stim_anti[i, -1, 0]], y=[projected_data_stim_anti[i, -1, 1]],
                                                       z=[response_anti[i, -1, resp + 1]], mode="markers",
                                                       marker=dict(size=6, color=c_vals[i], symbol="circle"),
                                                       legendgroup=f"S{i}", showlegend=False))
                fig3dresponse_c.add_trace(go.Scatter3d(x=[projected_data_stim_go[i, -1, 0]], y=[projected_data_stim_go[i, -1, 1]],
                                                       z=[response_go[i, -1, resp + 1]], mode="markers",
                                                       marker=dict(size=6, color=c_vals[i], symbol="diamond"),
                                                       legendgroup=f"S{i}", showlegend=False))
                fig3dresponse_c.add_trace(go.Scatter3d(x=[projected_data_stim_median[i, -1, 0]], y=[projected_data_stim_median[i, -1, 1]],
                                                       z=[response_median[i, -1, resp + 1]], mode="markers",
                                                       marker=dict(size=6, color=c_vals[i], symbol="square"),
                                                       legendgroup=f"S{i}", showlegend=False))
            x_end = projected_data_stim_anti[:, -1, 0]
            y_end = projected_data_stim_anti[:, -1, 1]
            z_end = response_anti[:, -1, resp + 1]
            add_convex_hull_mesh(fig3dresponse_c, x_end, y_end, z_end, name="Anti endpoint hull (cos)",
                                 mesh_opacity=0.18, mesh_color=c_vals[0], showlegend=True)
            x_go = projected_data_stim_go[:, -1, 0]
            y_go = projected_data_stim_go[:, -1, 1]
            z_go = response_go[:, -1, resp + 1]
            add_convex_hull_mesh(fig3dresponse_c, x_go, y_go, z_go, name="Go endpoint hull (cos)",
                                 mesh_opacity=0.18, mesh_color=c_vals[1], showlegend=True)
            x_median = projected_data_stim_median[:, -1, 0]
            y_median = projected_data_stim_median[:, -1, 1]
            z_median = response_median[:, -1, resp + 1]
            add_convex_hull_mesh(fig3dresponse_c, x_median, y_median, z_median, name="Median endpoint hull (cos)",
                                 mesh_opacity=0.18, mesh_color=c_vals[2], showlegend=True)
            pts_anti = np.column_stack([x_end, y_end, z_end])
            pts_go = np.column_stack([x_go, y_go, z_go])
            pts_median = np.column_stack([x_median, y_median, z_median])
            n_anti = best_fit_plane_normal(pts_anti)
            n_go = best_fit_plane_normal(pts_go)
            n_median = best_fit_plane_normal(pts_median)
            checknames = ["anti", "go", "median"]
            ns = [n_anti, n_go, n_median]
            for i in range(len(checknames)):
                for j in range(i + 1, len(checknames)):
                    ang_deg = angle_between_unit_vectors(ns[i], ns[j], degrees=True)
                    print(f"[{sname}] cosine: angle between {checknames[i]} - {checknames[j]} = {ang_deg:.2f} deg")
            fig3dresponse_c.update_layout(template="plotly_white", width=1000, height=800,
                                          scene=dict(xaxis_title="Memoryanti Stimulus PCA 1",
                                                     yaxis_title="Memoryanti Stimulus PCA 2",
                                                     zaxis_title="cos θ", zaxis=dict(range=[-1.1, 1.1]), aspectmode="cube"))

    traj_diff_alpha("hidden")
    traj_diff_alpha("modulation")
    traj_diff_alpha("w_modulation")

    # ═════════════════════════════════════════════════════════════════════════
    # Cell 106: stimulus-period PCA trajectories (anti / middle / go)
    # ═════════════════════════════════════════════════════════════════════════
    names = ["hidden", "modulation"]
    for siindex, stacked_interpolation in enumerate(stacked_interpolation_lst):
        N = len(stacked_interpolation)
        sname = stacked_interpolation_name_lst[siindex]
        for name in names:
            anti_go = [stacked_interpolation[0], stacked_interpolation[int((N + 1) / 2)], stacked_interpolation[-1]]
            _, _, db_intp_anti = net.iterate_sequence_batch(anti_go[0], run_mode='track_states', save_to_cpu=True, detach_saved=True)
            _, _, db_inp_middle = net.iterate_sequence_batch(anti_go[1], run_mode='track_states', save_to_cpu=True, detach_saved=True)
            _, _, db_intp_go = net.iterate_sequence_batch(anti_go[2], run_mode='track_states', save_to_cpu=True, detach_saved=True)
            Ms_anti, Ms_orig_anti, hs_anti, bs_anti = modulation_extraction(int_input_all[siindex], db_intp_anti, layer_index)
            Ms_middle, Ms_orig_middle, hs_middle, bs_middle = modulation_extraction(int_input_all[siindex], db_inp_middle, layer_index)
            Ms_go, Ms_orig_go, hs_go, bs_go = modulation_extraction(int_input_all[siindex], db_intp_go, layer_index)
            batch_num = Ms_orig_go.shape[0]
            if name == "hidden":
                data_anti, data_middle, data_go = hs_anti, hs_middle, hs_go
            elif name == "modulation":
                data_anti, data_middle, data_go = Ms_anti, Ms_middle, Ms_go
            n_activity = data_anti.shape[-1]
            as_flat_stim = data_anti[:, time_stamps_lst[siindex]["stimulus_start"]:time_stamps_lst[siindex]["stimulus_end"], :].reshape((-1, n_activity))
            as_flat_anti = data_anti.reshape((-1, n_activity))
            as_flat_middle = data_middle.reshape((-1, n_activity))
            as_flat_go = data_go.reshape((-1, n_activity))
            pca_stim = PCA(n_components=PCA_downsample, random_state=42)
            pca_stim.fit(as_flat_stim)
            projected_data_anti = pca_stim.transform(as_flat_anti).reshape((data_anti.shape[0], data_anti.shape[1], -1))
            projected_data_middle = pca_stim.transform(as_flat_middle).reshape((data_middle.shape[0], data_middle.shape[1], -1))
            projected_data_go = pca_stim.transform(as_flat_go).reshape((data_go.shape[0], data_go.shape[1], -1))
            ss = time_stamps_lst[siindex]["stimulus_start"]
            se = time_stamps_lst[siindex]["stimulus_end"]
            projected_data_stim_anti = projected_data_anti[:, ss:se, :]
            projected_data_stim_middle = projected_data_middle[:, ss:se, :]
            projected_data_stim_go = projected_data_go[:, ss:se, :]
            fig106, axs106 = plt.subplots(1, 3, figsize=(4 * 3, 4))
            combination = [[0, 1], [0, 2], [1, 2]]
            for comb_index, comb in enumerate(combination):
                for i in range(projected_data_stim_anti.shape[0]):
                    axs106[comb_index].plot(projected_data_stim_anti[i, :, comb[0]], projected_data_stim_anti[i, :, comb[1]], color=c_vals[i], linestyle=linestyles[0])
                    axs106[comb_index].plot(projected_data_stim_middle[i, :, comb[0]], projected_data_stim_middle[i, :, comb[1]], color=c_vals[i], linestyle=linestyles[1])
                    axs106[comb_index].plot(projected_data_stim_go[i, :, comb[0]], projected_data_stim_go[i, :, comb[1]], color=c_vals[i], linestyle=linestyles[2])
            for ax in axs106:
                ax.set_title(f"name: {name}; sname: {sname}", fontsize=12)
            fig106.tight_layout()
            fig106.savefig(fp(f"m_pca_stimulus_{name}_{sname}_seed{seed}_{hyp_dict['addon_name']}_{int_index}.png"), dpi=300)
            print("  Saved figure: " + str(fp(f"m_pca_stimulus_{name}_{sname}_seed{seed}_{hyp_dict['addon_name']}_{int_index}.png")))
            plt.close(fig106)

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
    args = parser.parse_args()

    anames = [args.aname] if args.aname else _discover_anames()
    print(f"Analyzing {len(anames)} run(s).")
    for a in anames:
        print(f"\n── Analyzing: {a} ──")
        try:
            main(a)
        except Exception as exc:
            print(f"  FAILED {a}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
