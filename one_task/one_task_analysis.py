#!/usr/bin/env python
# coding: utf-8
"""
Post-training analysis of a single-task MPN.

Reloads the per-stage training traces saved by one_task.py and reproduces the
single-task analyses:

1. Loss / accuracy across training.
2. Input weight matrix heatmap (W_initial_linear).
2b. Example single-trial input & network/target output.
3. Fixon vs task projection onto the readout — the "cancellation" mechanism
   (Eq. 2-7 sanity check) tracked across training. At the final stage also
   emits the exhaustive-search (es1/es2), fixon-task difference (diff), and
   per-stimulus cancellation (show) figures.
4. Modulation-change / synaptic & hidden correlation across learning.
5. Weight-component projection to output across learning.
6. Low-D PCA of the modulation matrix M during the stimulus period.

All outputs go into ./onetask/{aname}/. Aggregated correlation curves (across
seeds) are written to ./onetask_data/ and re-plotted if multiple runs exist.

Usage:
    python one_task_analysis.py                 # newest run in ./onetask/
    python one_task_analysis.py --aname <name>  # a specific run
"""
import os
import copy
import glob
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
ticker.Locator.MAXTICKS = 10000
import seaborn as sns

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper
import torch
import mpn
import networks as nets
import mpn_tasks
from grad_fixed_points import solve_period_modulation_fixed_points

# Match the plotting style used in multiple_task_analysis.py
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

c_vals = [
    "#e53e3e", "#3182ce", "#38a169", "#d69e2e", "#d53f8c",
    "#4c51bf", "#dd6b20", "#0ea5e9", "#22c55e", "#a855f7",
    "#f43f5e", "#0f766e", "#b83280", "#ca8a04", "#2b6cb0",
] * 10

c_vals_l = [
    "#feb2b2", "#90cdf4", "#9ae6b4", "#faf089", "#fbb6ce",
    "#c3dafe", "#fed7aa", "#bae6fd", "#bbf7d0", "#e9d5ff",
    "#fecdd3", "#a7f3d0", "#f9a8d4", "#fde68a", "#bfdbfe",
] * 10

c_vals_d = [
    "#9b2c2c", "#2c5282", "#276749", "#975a16", "#97266d",
    "#4338ca", "#7b341e", "#0369a1", "#15803d", "#6b21a8",
    "#9f1239", "#0f4c3a", "#702459", "#854d0e", "#1e3a8a",
] * 10

l_vals = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
markers_vals = ['o', 'v', '*', '+', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
linestyles = ["-", "--", "-."]

ONETASK_DIR = Path("onetask")
ONETASK_DATA_DIR = Path("onetask_data")


def _generate_random_orthonormal_matrix(N, num_columns=3):
    """N x num_columns matrix with orthonormal columns."""
    Q, _ = np.linalg.qr(np.random.randn(N, num_columns))
    return Q[:, :num_columns]


def _rebuild_net(net_params, device):
    """Instantiate the network class implied by net_params (no weights loaded)."""
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
    return netFunction(net_params, verbose=False)


def long_period_fixed_points(aname, save_dir, cfg, seed, shift_index, color_by,
                             fp_n_seeds=5, run_fixed_points=True):
    """Take the trained single-task network, generate test data with each trial
    period extended in turn (long fixation / stimulus / delay / response), fit a
    top-2 PCA on the pooled DELAY-period states, and scatter each variant's
    fixed point (last timestep) colored by stimulus. Mirrors the two-task
    attractor analysis. Done for both the hidden state and the effective
    modulation W ⊙ M. Requires the live checkpoint savednet_{aname}.pt."""
    ckpt_path = ONETASK_DIR / f"savednet_{aname}.pt"
    if not ckpt_path.exists():
        print(f"  [long-fp] checkpoint not found ({ckpt_path}); skipping.")
        return

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_params_ckpt = ckpt["net_params"]
    net = _rebuild_net(net_params_ckpt, device)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()

    # Rebuild task_params from the (un-converted) param json and re-run the
    # multitask conversion to populate hp / prefs / n_input needed for trial gen.
    task_params0 = copy.deepcopy(cfg["task_params"])
    train_params0 = copy.deepcopy(cfg["train_params"])
    net_params0 = copy.deepcopy(cfg["net_params"])

    layer_index = 1 if net_params_ckpt.get("input_layer_add", False) else 0

    # Each variant extends ONE period and is analyzed over THAT extended period
    # (its own epoch window), mirroring the two-task desire_period logic.
    variants = ["longfixation", "longstimulus", "longdelay", "longresponse"]
    period_key = {  # task_params flag to set "long", and the epoch to analyze
        "longfixation": ("long_fixation", "fix1"),
        "longstimulus": ("long_stimulus", "stim1"),
        "longdelay": ("long_delay", "delay1"),
        "longresponse": ("long_response", "go1"),
    }

    def _gen_and_run(variant):
        flag, epoch_name = period_key[variant]
        tp = copy.deepcopy(task_params0)
        tp["long_fixation"] = tp["long_stimulus"] = tp["long_delay"] = tp["long_response"] = "normal"
        tp[flag] = "long"
        tp, trp, npp = mpn_tasks.convert_and_init_multitask_params(
            (tp, copy.deepcopy(train_params0), copy.deepcopy(net_params0)))
        npp["prefs"] = mpn_tasks.get_prefs(tp["hp"])
        test_n_batch = trp["valid_n_batch"]
        tp["hp"]["batch_size_train"] = test_n_batch
        data, extra = mpn_tasks.generate_trials_wrap(
            tp, test_n_batch, rules=tp["rules"], mode_input="random", device=device)
        _, trials, _ = extra
        test_input = data[0]
        # stimulus labels for this variant (one ring stimulus per trial)
        stim = np.asarray(trials[0].meta["stim1"]).reshape(-1)
        ep = trials[0].epochs[epoch_name]    # (start, end_exclusive); start may be None (fix1)
        T = test_input.shape[1]
        win = (0 if ep[0] is None else int(ep[0]), T if ep[1] is None else int(ep[1]))
        _, _, db = net.iterate_sequence_batch(
            test_input.to(device), run_mode="track_states", save_to_cpu=True, detach_saved=True)
        return db, stim, win, np.asarray(test_input.detach().cpu())

    # Run each variant once; cache hidden + W⊙M states and the EXTENDED period
    # window to analyze for that variant.
    W = net.mp_layer1.W.data.detach().cpu().numpy() if net_params_ckpt.get("input_layer_add", False) \
        else net.mp_layer0.W.data.detach().cpu().numpy()
    cache = {}
    for v in variants:
        try:
            db, stim, win, var_input = _gen_and_run(v)
        except Exception as exc:
            print(f"  [long-fp] variant {v} failed: {exc}; skipping it.")
            continue
        h = np.asarray(db[f"hidden{layer_index}"])                  # (batch, T, hidden)
        M = np.asarray(db[f"M{layer_index}"])                       # (batch, T, hidden, hidden)
        mm = M.reshape(M.shape[0], M.shape[1], -1)                  # raw M flattened
        em = (M * W[None, None, :, :]).reshape(M.shape[0], M.shape[1], -1)  # W⊙M flattened
        # Keep the raw (unflattened) M and the input tensor for gradient-based
        # fixed-point finding (fixed_point.find_modulation_fixed_points).
        cache[v] = {"hidden": h, "m_modulation": mm, "e_modulation": em,
                    "M": M, "input": var_input,
                    "stim": stim, "period": tuple(win)}
        del db, M

    if not cache:
        print("  [long-fp] no variants succeeded; skipping figure.")
        return

    # One subplot per period (like the two-task per-period figures). Each period
    # gets its OWN top-2 PCA, fit on that period's extended-window states, and
    # shows: the per-trial trajectory over that window (line, colored by
    # stimulus) + the fixed point = last frame of the window (black-edged marker).
    present = [v for v in variants if v in cache]
    period_title = {"longfixation": "Fixation", "longstimulus": "Stimulus",
                    "longdelay": "Delay", "longresponse": "Response"}

    # Accumulate the projected 2-D trajectories so paper_plot can re-render this
    # without re-running the network: {rep: {period: {"proj": (batch,win_T,2),
    # "stim": (batch,)}}}.
    long_fp_save = {}

    # Source of the shared PCA basis: the (extended) DELAY window of the
    # longdelay variant. Every period panel in a row is projected into THIS same
    # delay-period basis, so PC1/PC2 mean the same axes across all panels.
    if "longdelay" not in cache:
        print("  [long-fp] no 'longdelay' variant; cannot build a shared delay "
              "PCA basis; skipping figure.")
        return

    for rep in ("hidden", "m_modulation", "e_modulation"):
        n_var = len(present)
        fig, axs = plt.subplots(1, n_var, figsize=(4 * n_var, 4), squeeze=False)
        long_fp_save[rep] = {}

        # Fit ONE PCA on the delay window of the longdelay variant for this rep.
        cd = cache["longdelay"]
        dps, dpe = cd["period"]
        delay_seg = cd[rep][:, dps:dpe, :]
        pca = PCA(n_components=2, random_state=0).fit(
            delay_seg.reshape(-1, delay_seg.shape[-1]))

        for ax, v in zip(axs[0], present):
            c = cache[v]
            ps, pe = c["period"]
            stim = c["stim"]
            # Save the window with one extra leading point (the transition-in
            # frame from the previous period) whenever one exists (ps>0). `lead`
            # records how many leading frames were prepended (0 for fixation,
            # which starts at t=0). The display rule below decides whether to
            # show it. Both the saved `proj` and `lead` are consumed by
            # paper_plot, so the display rule is applied there too.
            ps_ext = max(ps - 1, 0)
            lead = ps - ps_ext                              # 0 or 1
            seg = c[rep][:, ps_ext:pe, :]                   # (batch, win_T, feat)
            # project this period's window into the shared delay-period basis
            proj = pca.transform(seg.reshape(-1, seg.shape[-1])).reshape(
                seg.shape[0], seg.shape[1], 2)              # (batch, win_T, 2)
            # Hidden starts strictly at the period boundary (drop the leading
            # frame): it is an instantaneous readout of the current stimulus
            # input, so a leading frame already sits at a stimulus-specific
            # location. Modulation keeps the leading transition-in frame.
            disp_start = lead if rep == "hidden" else 0
            for i in range(seg.shape[0]):
                col = c_vals[int(stim[i]) % len(c_vals)]
                p = proj[i]                                 # (win_T, 2)
                ax.plot(p[disp_start:, 0], p[disp_start:, 1], color=col,
                        alpha=0.4, linewidth=0.8, zorder=2)
                ax.scatter(p[-1, 0], p[-1, 1], color=col, marker="o", s=45,
                           edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)
            ax.set_xlabel("Delay PC1", fontsize=10)
            ax.set_ylabel("Delay PC2", fontsize=10)
            ax.set_title(period_title.get(v, v), fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            long_fp_save[rep][v] = {"proj": np.asarray(proj, dtype=float),
                                    "stim": np.asarray(stim), "lead": int(lead)}

        # stimulus-color legend on the first subplot
        uniq_stim = sorted(set(int(s) for c in cache.values() for s in c["stim"]))
        stim_handles = [plt.Line2D([0], [0], marker="o", linestyle="None",
                                   markerfacecolor=c_vals[s % len(c_vals)],
                                   markeredgecolor="black", markersize=6, label=f"stim {s}")
                        for s in uniq_stim]
        axs[0, 0].legend(handles=stim_handles, frameon=True, fontsize=6, ncol=2,
                         title="stimulus", loc="best")
        fig.suptitle(f"{aname}  |  {rep} per-period trajectory + fixed point "
                     f"(shared delay-period PCA)", fontsize=10)
        fig.tight_layout()
        out = save_dir / f"long_fixed_points_{rep}_{aname}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved figure: {out}")

    # Persist the projected trajectories for paper_plot reuse.
    import pickle as _pickle
    long_pkl = save_dir / f"long_fixed_points_{aname}.pkl"
    with open(long_pkl, "wb") as _f:
        _pickle.dump({"aname": aname, "present": present,
                      "period_title": period_title, "data": long_fp_save}, _f)
    print(f"  Saved long_fixed_points data: {long_pkl}")

    # ── Gradient-based TRUE fixed points of the modulation matrix ────────────
    # In addition to the settling-endpoint proxy above, solve for genuine fixed
    # points M* = F(M*; x) of the modulation dynamics under each period's
    # constant input, via fixed_point.find_modulation_fixed_points. Seed the
    # optimizer at the recorded end-of-period M and hold the mid-period input
    # fixed during the relaxation.
    # Solve TRUE fixed points per period, seeded from a DENSE grid of stimulus
    # angles (64 by default) rather than only the 8 trained directions — this
    # also serves as the continuous-attractor probe (fixed_points_grad_*.pkl).
    # This gradient solve is the slow part; skip it when run_fixed_points=False.
    if run_fixed_points:
        try:
            solve_period_modulation_fixed_points(
                aname, save_dir, net, cfg, device, layer_index=layer_index, W=W,
                n_interp=64, n_seeds=fp_n_seeds)
        except Exception as exc:
            print(f"  [grad-fp] failed: {exc}")
            import traceback
            traceback.print_exc()
    else:
        print("  [grad-fp] skipped (--no-fixed-points).")

    # free GPU memory
    try:
        net.to("cpu")
    except Exception:
        pass
    del net
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cross_period_fve(H, periods, k=4, center="none", dtype=np.float64):
    """
    Cross-period PCA explained-variance for a SINGLE task (the one-task analog
    of two_task_analysis.figure2A_pca_fve's per-task block).

    H         : (batch, T, feat) states.
    periods   : ordered dict {period_name: (t0, t1)}.
    k         : number of PCs of the target period's subspace.

    Returns (period_names, fve_k) where fve_k[i, j] is the fraction of period i's
    variance captured by the top-k PCA subspace fit on period j. Diagonal ≈ how
    self-contained a period is in k PCs; off-diagonal = how much one period's
    geometry is shared with another's subspace.
    """
    H_np = (H.detach().cpu().numpy() if hasattr(H, "detach") else np.asarray(H))
    H_np = H_np.astype(dtype, copy=False)
    _, T, N = H_np.shape
    names = list(periods.keys())
    P = len(names)

    def _mat(t0, t1):
        return H_np[:, t0:t1, :].reshape(-1, N)

    def _center(X):
        if center == "none":
            return X
        return X - X.mean(axis=0)

    def _topk_components(X, r):
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        r_eff = min(r, Vt.shape[0])
        return Vt[:r_eff, :].T          # (N, r_eff)

    def _fve_project(X, V):
        tot = np.sum(X * X)
        if tot <= 0:
            return 0.0
        Xhat = (X @ V) @ V.T
        return float(np.sum(Xhat * Xhat) / tot)

    Xc = {}
    Vk = {}
    for nm in names:
        t0, t1 = periods[nm]
        if not (0 <= t0 < t1 <= T):
            raise ValueError(f"[{nm}] invalid period bounds {(t0, t1)} for T={T}")
        Xc[nm] = _center(_mat(t0, t1))
        Vk[nm] = _topk_components(Xc[nm], k)

    fve_k = np.zeros((P, P), dtype=dtype)
    for i, px in enumerate(names):
        for j, py in enumerate(names):
            fve_k[i, j] = _fve_project(Xc[px], Vk[py])
    return names, fve_k


def _period_cumvar(H, periods, max_pc=11, center="none", dtype=np.float64):
    """
    Per-period cumulative variance explained by a period's OWN top PCs.

    For each trial period, fit PCA on that period's states and return the
    cumulative fraction of variance captured by the top 1, 2, ..., max_pc PCs.
    This is the "top-11 PCs per period" scree/cumulative curve (the right panel
    of the cross-period dimensionality figure).

    Returns (period_names, cumvar) where cumvar[i] is a length-`max_pc` array of
    cumulative variance ratios for period i (monotone increasing toward 1).
    """
    H_np = (H.detach().cpu().numpy() if hasattr(H, "detach") else np.asarray(H))
    H_np = H_np.astype(dtype, copy=False)
    _, T, N = H_np.shape
    names = list(periods.keys())

    cumvar = np.zeros((len(names), max_pc), dtype=dtype)
    for i, nm in enumerate(names):
        t0, t1 = periods[nm]
        X = H_np[:, t0:t1, :].reshape(-1, N)
        if center != "none":
            X = X - X.mean(axis=0)
        # Singular values → variance per component (∝ s^2).
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        var = s ** 2
        tot = float(var.sum())
        r = min(max_pc, var.size)
        frac = (np.cumsum(var[:r]) / tot) if tot > 0 else np.zeros(r)
        cumvar[i, :r] = frac
        if r < max_pc:                       # fewer PCs than max_pc: hold at 1.0
            cumvar[i, r:] = frac[-1] if r > 0 else 0.0
    return names, cumvar


def cross_period_dimensionality(aname, save_dir, hs_final, Ms_final, W_eff,
                                stimulus_start, stimulus_end, response_start,
                                top_k=4):
    """
    One-task cross-period PCA explained-variance heatmaps (the single-task analog
    of two_task_analysis's d_combine figure). For hidden activity, modulation
    (raw M) and effective modulation (W⊙M), fit a top-k PCA on each trial
    period's states and measure how well it captures every other period's
    variance — a 4x4 (Fixation/Stimulus/Memory/Response) matrix per series.

    Saves d_combine_{aname}.png and .pkl (for paper_plot reuse).
    """
    T = hs_final.shape[1]
    # Trial periods (match the two-task period layout: fixation/stim/delay/resp).
    periods = {
        "Fixation": (0, max(stimulus_start - 1, 1)),
        "Stimulus": (stimulus_start, stimulus_end),
        "Memory": (stimulus_end, max(response_start - 1, stimulus_end + 1)),
        "Response": (response_start, T),
    }

    emod_full = (Ms_final * W_eff[None, None, :, :]).reshape(
        Ms_final.shape[0], Ms_final.shape[1], -1)

    series = [
        ("hidden", hs_final),
        ("w_modulation", emod_full),
    ]

    d_combine_data = {}
    fig, axs = plt.subplots(1, len(series), figsize=(4 * len(series), 3.8))
    if len(series) == 1:
        axs = [axs]
    max_pc = 11
    for ax, (name, H) in zip(axs, series):
        names, fve_k = _cross_period_fve(H, periods, k=top_k, center="none")
        # Per-period own-PCA cumulative variance curve (top max_pc PCs).
        _, cumvar = _period_cumvar(H, periods, max_pc=max_pc, center="none")
        sns.heatmap(fve_k, ax=ax, xticklabels=names, yticklabels=names,
                    annot=True, fmt=".2f", vmin=0.0, vmax=1.0, square=True,
                    cbar=True, cbar_kws={"shrink": 0.7})
        ax.set_title(f"{name} (k={top_k})", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        d_combine_data[name] = {
            "fve_k_all": np.asarray(fve_k),
            "labels": names,
            "vmin": 0.0,
            "vmax": 1.0,
            "top_k": int(top_k),
            # Right-panel data: cumulative variance vs #PCs, per period.
            "cumvar": np.asarray(cumvar),        # (n_period, max_pc)
            "max_pc": int(max_pc),
        }
    fig.tight_layout()
    fig.savefig(save_dir / f"d_combine_{aname}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {save_dir / f'd_combine_{aname}.png'}")

    import pickle as _pickle
    with open(save_dir / f"d_combine_{aname}.pkl", "wb") as _f:
        _pickle.dump(d_combine_data, _f)
    print(f"  Saved d_combine data: {save_dir / f'd_combine_{aname}.pkl'}")


def modulation_magnitude_by_component(aname, save_dir, Ms_orig, W_input,
                                      input_specs, dt, stimulus_start,
                                      stimulus_end, response_start):
    """Modulation-computation magnitude across time, per raw input component.

    For each raw input channel c, the modulation applied to that channel is the
    hidden-unit vector obtained by contracting the plasticity matrix M's
    embedded-input axis with that channel's input-embedding column:
        p_c[b, t, :] = M[b, t, :, :] @ W_input[:, c]        (raw M — Hebbian trace)
    Its per-timestep magnitude is the L2 norm over hidden units, ‖p_c[b, t, :]‖₂,
    averaged over trials with a ±std band. The two stimulus modalities' cos/sin
    channels are combined into a SINGLE "Stimulus" trajectory: the per-trial MEAN
    of their per-channel magnitudes, computed per trial and then averaged (so its
    std is not recoverable from the per-channel means alone). The figure therefore
    shows Fixation, the combined Stimulus, and the Task cue.

    `Ms_orig` : (batch, T, hidden, embed) final-stage modulation.
    `W_input` : (embed, n_raw) input embedding column-mapped to raw channels
                (identity when there is no input layer).
    `input_specs` : ordered [(raw_channel_index, label), ...] for the individual
                    per-channel magnitudes (kept in the pickle for reference).
    Saves modulation_magnitude_{aname}.png and .pkl (for paper_plot reuse).
    """
    import pickle as _pickle
    T = Ms_orig.shape[1]
    # Project M onto every raw input column at once: (batch, T, hidden, n_raw).
    # M @ W_input contracts the embed axis, mapping each hidden unit's modulation
    # to the raw input channels.
    proj = np.einsum("bthe,er->bthr", Ms_orig, W_input)
    # L2 norm over hidden units → per-trial magnitude per channel (batch, T, n_raw).
    mag = np.linalg.norm(proj, axis=2)
    mag_mean = mag.mean(axis=0)                       # (T, n_raw)
    mag_std = mag.std(axis=0)                         # (T, n_raw) across-trial std

    channels = [int(ch) for ch, _ in input_specs]
    labels = [lab for _, lab in input_specs]

    # Combine the stimulus channels (every component that is neither Fixation nor
    # the Task cue) into one trajectory: the per-trial MEAN of their per-channel
    # magnitudes, then mean ± std across trials.
    stim_channels = [ch for ch, lab in zip(channels, labels)
                     if lab not in ("Fixation", "Task cue")]
    if stim_channels:
        stim_mag = mag[:, :, stim_channels].mean(axis=2)  # (batch, T)
        stim_mag_mean = stim_mag.mean(axis=0)             # (T,)
        stim_mag_std = stim_mag.std(axis=0)
    else:
        stim_mag_mean = np.zeros(T)
        stim_mag_std = np.zeros(T)

    # Three summary curves: Fixation, combined Stimulus, Task cue (mean ± std).
    fix_ch = 0
    task_ch = max(channels)
    t_ms = np.arange(T) * dt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_ms, mag_mean[:, fix_ch], "-", color=c_vals[0], label="Fixation")
    ax.fill_between(t_ms, mag_mean[:, fix_ch] - mag_std[:, fix_ch],
                    mag_mean[:, fix_ch] + mag_std[:, fix_ch],
                    color=c_vals_l[0], alpha=0.3)
    ax.plot(t_ms, stim_mag_mean, "-", color=c_vals[2], label="Stimulus")
    ax.fill_between(t_ms, stim_mag_mean - stim_mag_std,
                    stim_mag_mean + stim_mag_std, color=c_vals_l[2], alpha=0.3)
    ax.plot(t_ms, mag_mean[:, task_ch], "-", color=c_vals[1], label="Task cue")
    ax.fill_between(t_ms, mag_mean[:, task_ch] - mag_std[:, task_ch],
                    mag_mean[:, task_ch] + mag_std[:, task_ch],
                    color=c_vals_l[1], alpha=0.3)
    # Dashed period boundaries (stimulus / memory / response onsets), in ms.
    for bt in (stimulus_start, stimulus_end, response_start):
        ax.axvline(bt * dt, color="0.5", lw=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Time (ms)", fontsize=13)
    ax.set_ylabel("Modulation magnitude\n(‖M·x_c‖₂)", fontsize=12)
    ax.legend(loc="best", frameon=True, fontsize=9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / f"modulation_magnitude_{aname}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {save_dir / f'modulation_magnitude_{aname}.png'}")

    with open(save_dir / f"modulation_magnitude_{aname}.pkl", "wb") as _f:
        _pickle.dump({
            "aname": aname,
            "dt": int(dt),
            "channels": channels,                            # raw input indices
            "labels": labels,                                # matching labels
            "mag_mean": np.asarray(mag_mean, dtype=float),   # (T, n_raw)
            "mag_std": np.asarray(mag_std, dtype=float),     # (T, n_raw) across-trial std
            # Combined stimulus trajectory (per-trial MEAN over the stimulus
            # channels, then mean ± std across trials); its std is not recoverable
            # from the per-channel means, so saved explicitly.
            "stim_channels": [int(c) for c in stim_channels],
            "stim_mag_mean": np.asarray(stim_mag_mean, dtype=float),  # (T,)
            "stim_mag_std": np.asarray(stim_mag_std, dtype=float),    # (T,)
            "stimulus_start": int(stimulus_start),
            "stimulus_end": int(stimulus_end),
            "response_start": int(response_start),
        }, _f)
    print(f"  Saved modulation magnitude data: "
          f"{save_dir / f'modulation_magnitude_{aname}.pkl'}")


def main(aname, fp_n_seeds=5, run_fixed_points=True):
    result_path = ONETASK_DIR / f"param_{aname}_result.npz"
    param_path = ONETASK_DIR / f"param_{aname}_param.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Result traces not found: {result_path}")

    with open(param_path) as f:
        cfg = json.load(f)
    task_params = cfg["task_params"]
    net_params = cfg["net_params"]
    fixate_off = task_params["fixate_off"]
    # Simulation time step in ms (see SCHEME.md): one recorded step = dt ms. Time
    # axes are labeled in ms (step index * dt), not raw step index.
    dt = task_params.get("dt", 40)

    data = np.load(result_path, allow_pickle=True)
    hyp_dict = data["hyp_dict"].item()
    seed = int(data["seed"])
    shift_index = int(data["shift_index"])
    color_by = str(data["color_by"])
    counter_lst = data["counter_lst"]
    loss_lst = data["loss_lst"]
    acc_lst = data["acc_lst"]
    # test_input_np is the SAVED validation set, kept because the per-stage
    # modulation traces (Ms_orig_stages etc.) are aligned to exactly these
    # trials.
    test_input_np = data["test_input_np"]
    test_output_np = data["test_output_np"]
    net_out_final = data["net_out_final"]   # final-stage network output on the test set
    input_matrix_final = data["input_matrix_final"]
    labels = data["labels"]
    Ms_orig_stages = data["Ms_orig_stages"]          # (stage, batch, T, hidden, input)
    hs_stages = data["hs_stages"]                    # (stage, batch, T, hidden)
    bs_stages = data["bs_stages"]                    # (stage, batch, T, hidden)
    Wall_stages = data["Wall_stages"]                # (stage, hidden, hidden) — MP layer W on embedded input
    Woutput_stages = data["Woutput_stages"]          # (stage, out, hidden)
    Winput_stages = data["Winput_stages"]            # (stage, hidden, n_raw_input) — input embedding
    input_layer_add = bool(net_params.get("input_layer_add", False))
    response_start = int(data["response_start"])
    stimulus_start = int(data["stimulus_start"])
    stimulus_end = int(data["stimulus_end"])
    all_breaks = data["all_breaks"].tolist()

    stages_num = Ms_orig_stages.shape[0]
    batch_nums = Ms_orig_stages.shape[1]
    print(f"stages={stages_num}, batch={batch_nums}, "
          f"stim=({stimulus_start},{stimulus_end}), response_start={response_start}")

    # counter_lst / loss_lst / acc_lst can differ in length by one because
    # train_network appends to them at slightly different points. Align every
    # per-stage quantity (and anything plotted against counter_lst) to a single
    # common length so x/y dimensions always match.
    n_common = min(len(counter_lst), len(loss_lst), len(acc_lst), stages_num)
    if n_common != stages_num or n_common != len(counter_lst):
        print(f"  [warn] trimming traces to common length {n_common} "
              f"(counter={len(counter_lst)}, loss={len(loss_lst)}, "
              f"acc={len(acc_lst)}, stages={stages_num})")
    counter_lst = np.asarray(counter_lst)[:n_common]
    loss_lst = np.asarray(loss_lst)[:n_common]
    acc_lst = np.asarray(acc_lst)[:n_common]
    # Use the trailing n_common stages so the final (best-trained) stage is kept.
    stage_sel = slice(stages_num - n_common, stages_num)
    Ms_orig_stages = Ms_orig_stages[stage_sel]
    hs_stages = hs_stages[stage_sel]
    bs_stages = bs_stages[stage_sel]
    Wall_stages = Wall_stages[stage_sel]
    Woutput_stages = Woutput_stages[stage_sel]
    if input_layer_add and Winput_stages.size > 0:
        Winput_stages = Winput_stages[stage_sel]
    stages_num = n_common

    save_dir = ONETASK_DIR / aname
    save_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale outputs before regenerating — but ONLY on a full run. When
    # --no-fixed-points is set the slow fixed-point pickles are not regenerated,
    # so leave the folder intact to preserve any fixed_points_* files from an
    # earlier full run (paper_plot still reads them).
    if run_fixed_points:
        for _old in save_dir.iterdir():
            if _old.is_file():
                _old.unlink()

    # ── Figure: loss / accuracy across training ──────────────────────────────
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(counter_lst, loss_lst, "-o", c=c_vals[0])
    ax1.set_ylabel("MSE Loss", color=c_vals[0], fontsize=15)
    ax1.tick_params(axis='y', colors=c_vals[0], labelsize=12)
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(counter_lst, acc_lst, "-o", c=c_vals[1])
    ax2.axhline(y=1 / 8, linestyle="--", label="By Chance")
    ax2.set_ylabel("Accuracy", color=c_vals[1], fontsize=15)
    ax2.tick_params(axis='y', colors=c_vals[1], labelsize=12)
    ax2.legend(loc='best', frameon=True, fontsize=12)
    ax1.set_xlabel("# Dataset", fontsize=15)
    ax1.set_xscale("log")
    fig.tight_layout()
    fig.savefig(save_dir / f"loss_acc_{aname}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {save_dir / f'loss_acc_{aname}.png'}")

    # ── Input weight matrix heatmap (notebook cell 9) ────────────────────────
    if net_params["input_layer_add"] and input_matrix_final.size > 0:
        figinp, axinp = plt.subplots(1, 1, figsize=(4, 4))
        sns.heatmap(input_matrix_final, ax=axinp, square=True, cmap='coolwarm')
        axinp.set_title("Input weight (W_initial_linear)", fontsize=12)
        figinp.tight_layout()
        figinp.savefig(save_dir / f"input_weight_{aname}.png", dpi=300)
        plt.close(figinp)
        print(f"  Saved figure: {save_dir / f'input_weight_{aname}.png'}")

    # ── Example single-trial input & output ──────────────────────────────────
    # One representative trial. Input layout (shift_index=1): channel 0 =
    # fixation, channels [1,2] and [3,4] are two stimulus (cos,sin) groups, the
    # last channel = task cue. Only ONE stimulus group is active per trial, so
    # we plot 4 channels: Fixation, Stim Cos, Stim Sin, Task Cue.
    # Output (3 channels): Fixation, Output Cos, Output Sin (network output).
    b0 = 0
    fix_ch = 0
    task_ch = test_input_np.shape[-1] - 1
    # The two candidate stimulus groups; pick the one carrying signal in trial b0.
    groups = [(1, 2), (3, 4)]
    group_energy = [np.abs(test_input_np[b0, :, list(g)]).sum() for g in groups]
    cos_ch, sin_ch = groups[int(np.argmax(group_energy))]

    figex, axex = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    in_specs = [(fix_ch, "Fixation"), (cos_ch, "Stim Cos"),
                (sin_ch, "Stim Sin"), (task_ch, "Task Cue")]
    for k, (ch, lab) in enumerate(in_specs):
        axex[0].plot(test_input_np[b0, :, ch], color=c_vals[k % len(c_vals)], label=lab)
    axex[0].set_ylabel("Input", fontsize=12)
    axex[0].set_title(f"Example trial (stimulus = {int(labels[b0, 0])})", fontsize=11)

    out_labels = ["Fixation", "Output Cos", "Output Sin"]
    for out_idx in range(min(test_output_np.shape[-1], len(out_labels))):
        # Target output as a faded shadow (no legend entry), network output on top.
        axex[1].plot(test_output_np[b0, :, out_idx], color=c_vals_l[out_idx % len(c_vals_l)],
                     linewidth=4, alpha=0.6)
        axex[1].plot(net_out_final[b0, :, out_idx], color=c_vals[out_idx % len(c_vals)],
                     label=out_labels[out_idx])
    axex[1].set_ylabel("Output", fontsize=12)
    axex[1].set_xlabel("Time (ms)", fontsize=12)
    # Traces are plotted against step index; relabel ticks in ms (index * dt).
    axex[1].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _pos: f"{x * dt:.0f}"))

    for ax in axex:
        ax.legend(fontsize=7, frameon=True, loc="best", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    figex.tight_layout()
    figex.savefig(save_dir / f"example_trial_{aname}.png", dpi=300)
    plt.close(figex)
    print(f"  Saved example trial figure: {save_dir / f'example_trial_{aname}.png'}")

    # Save the exact traces so paper_plot can re-render this figure identically:
    # the chosen trial's input channels (with their labels) and the network /
    # target output channels.
    import pickle as _pickle
    example_trial_pkl = {
        "aname": aname,
        # Simulation time step in ms (see SCHEME.md); lets paper_plot label the
        # example-trial time axis in ms (step index * dt) instead of step index.
        "dt": int(dt),
        "stimulus": int(labels[b0, 0]),
        "input_specs": [(int(ch), lab) for ch, lab in in_specs],
        "input": np.asarray(test_input_np[b0]),          # (T, n_input)
        "output_labels": list(out_labels),
        "net_output": np.asarray(net_out_final[b0]),      # (T, n_output)
        "target_output": np.asarray(test_output_np[b0]),  # (T, n_output)
        # Trial-period boundaries, for shading sessions like onetask_show.
        "stimulus_start": int(stimulus_start),
        "stimulus_end": int(stimulus_end),
        "response_start": int(response_start),
    }
    with open(save_dir / f"example_trial_{aname}.pkl", "wb") as _f:
        _pickle.dump(example_trial_pkl, _f)
    print(f"  Saved example trial data: {save_dir / f'example_trial_{aname}.pkl'}")

    # ── Fixon vs Task projection onto readout (cancellation) ─────────────────
    # For each training stage, project the modulated weight's response to the
    # fixation-on and task inputs onto the readout directions, and track how the
    # network learns to CANCEL the fixon contribution against the task input
    # before the response period.
    def plot_trajectory_by_index(label_index, stage_iter, verbose=False):
        """Replicates the notebook's plot_trajectory_by_index.

        Projects the modulated weight's response to decomposed input components
        (fixon/fixoff/stimulus/task) onto the readout directions. saver1 uses
        combined components (allX1); saver2 uses individual components (allX2).
        When verbose, emits the exhaustive-search, fixon-task-diff, and per-stim
        cancellation ("show") figures for this stage.

        The MP-layer weight W operates on the EMBEDDED input (W_initial_linear @
        raw_input), so each raw input component is mapped through W_input first;
        W_input is identity when there is no input layer."""
        W_ = Wall_stages[stage_iter]
        W_output = Woutput_stages[stage_iter]
        Ms_orig = Ms_orig_stages[stage_iter]
        bs = bs_stages[stage_iter]
        if input_layer_add and Winput_stages.size > 0:
            W_input = Winput_stages[stage_iter]
        else:
            W_input = np.eye(W_.shape[1])

        T = test_input_np.shape[1]

        if verbose:
            figsize1, figsize2 = 3, 6
            figexh1, axsexh1 = plt.subplots(3, 3, figsize=(figsize2 * 3, figsize1 * 3))
            figexh2, axsexh2 = plt.subplots(4, 3, figsize=(figsize2 * 3, figsize1 * 4))
            figdiff, axsdiff = plt.subplots(1, 2, figsize=(4 * 2, 2))

        task_labels_across_batch = []
        saver_shape1 = (3, 3)
        saver1 = np.empty((batch_nums, saver_shape1[0], saver_shape1[1]), dtype=object)
        saver_shape2 = (4, 3)
        saver2 = np.empty((batch_nums, saver_shape2[0] + 1, saver_shape2[1]), dtype=object)
        saver2_random = np.empty((batch_nums, saver_shape2[0] + 1, saver_shape2[1]), dtype=object)
        random_output_Y_lst = [_generate_random_orthonormal_matrix(W_output.shape[1]) for _ in range(10)]

        allX1name = ["x_fixon+x_task", "x_fixoff+x_task", "x_stimulus+x_fixon+x_task"]
        allX2name = ["x_fixon", "x_fixoff", "x_stimulus", "x_task"]
        allYname = ["y_fix", "Y_resp1", "Y_resp2"]

        for batch_iter in range(batch_nums):
            labels_for_batch = labels[batch_iter, 0]
            if labels_for_batch not in label_index:
                continue

            x_batch_taskinfo = test_input_np[batch_iter, :, :][:, 6 - shift_index:][0, :]
            task_specific = np.where(x_batch_taskinfo == 1)[0]
            assert len(task_specific) == 1
            task_specific = task_specific[0]
            task_labels_across_batch.append(task_specific)

            for i in range(saver_shape1[0]):
                for j in range(saver_shape1[1]):
                    saver1[batch_iter, i, j] = np.array([])
            for i in range(saver_shape2[0] + 1):
                for j in range(saver_shape2[1]):
                    saver2[batch_iter, i, j] = np.array([])
                    saver2_random[batch_iter, i, j] = np.array([])

            for time_iter in range(T):
                x = test_input_np[batch_iter, time_iter, :].reshape(-1, 1)
                input_length = len(x)
                x_fixon, x_fixoff, x_stimulus, x_task = [np.zeros((input_length, 1)) for _ in range(4)]
                x_fixon[0, 0] = x[0, 0]
                x_fixoff[1, 0] = x[1, 0] if fixate_off else 0
                x_stimulus[2 - shift_index:6 - shift_index, 0] = x[2 - shift_index:6 - shift_index, 0]
                x_task[6 - shift_index:, 0] = x[6 - shift_index:, 0]

                Mt = Ms_orig[batch_iter, time_iter, :, :]
                bt = bs[batch_iter, time_iter, :].reshape(-1, 1)
                middle = W_ + W_ * Mt

                y_fix = W_output[0, :].reshape(1, -1)
                Y_resp1 = W_output[1, :].reshape(1, -1)
                Y_resp2 = W_output[2, :].reshape(1, -1)

                # Combined-component inputs (allX1) and individual ones (allX2),
                # each embedded into the hidden-dim space via W_input.
                if fixate_off:
                    allX1 = [x_fixon + x_task, x_fixoff + x_task, x_stimulus + x_fixon + x_task]
                else:
                    allX1 = [x_fixon + x_task, x_task, x_stimulus + x_fixon + x_task]
                allX1 = [W_input @ xc for xc in allX1]
                allX2 = [W_input @ xc for xc in (x_fixon, x_fixoff, x_stimulus, x_task)]
                allY = [y_fix, Y_resp1, Y_resp2]

                for yiter in range(len(allY)):
                    for xiter in range(len(allX1)):
                        step1 = middle @ allX1[xiter] + bt
                        res1 = allY[yiter] @ step1
                        saver1[batch_iter, xiter, yiter] = np.append(
                            saver1[batch_iter, xiter, yiter], res1[0, 0])

                for y1 in range(len(allY)):
                    for x1 in range(len(allX2)):
                        step1 = middle @ allX2[x1]
                        res2 = allY[y1] @ step1
                        res2_random = [((rY[:, y1].reshape(1, -1)) @ middle @ allX2[x1])[0, 0]
                                       for rY in random_output_Y_lst]
                        saver2[batch_iter, x1, y1] = np.append(saver2[batch_iter, x1, y1], res2[0, 0])
                        saver2_random[batch_iter, x1, y1] = np.append(
                            saver2_random[batch_iter, x1, y1], np.mean(res2_random))

                # bias projection onto each readout
                for y_iter2 in range(len(allY)):
                    res2 = allY[y_iter2] @ bt
                    saver2[batch_iter, len(allX2), y_iter2] = np.append(
                        saver2[batch_iter, len(allX2), y_iter2], res2[0, 0])

            if verbose:
                ls = l_vals[task_specific % len(l_vals)]
                cb = c_vals[labels_for_batch % len(c_vals)]
                cbl = c_vals_l[labels_for_batch % len(c_vals_l)]
                for i in range(saver_shape1[0]):
                    for j in range(saver_shape1[1]):
                        axsexh1[i, j].plot(saver1[batch_iter, i, j], color=cb, linestyle=ls)
                for i in range(saver_shape2[0]):
                    for j in range(saver_shape2[1]):
                        axsexh2[i, j].plot(saver2[batch_iter, i, j], color=cb, linestyle=ls)
                axsdiff[0].plot(saver2[batch_iter, 0, 1] + saver2[batch_iter, 3, 1], color=cb, linestyle=ls)
                axsdiff[0].plot(saver2_random[batch_iter, 0, 1] + saver2_random[batch_iter, 3, 1], color=cbl, linestyle=ls)
                axsdiff[1].plot(saver2[batch_iter, 0, 2] + saver2[batch_iter, 3, 2], color=cb, linestyle=ls)
                axsdiff[1].plot(saver2_random[batch_iter, 0, 2] + saver2_random[batch_iter, 3, 2], color=cbl, linestyle=ls)

        if verbose:
            # per-stimulus fixon/task/combine cancellation ("show") figure
            figpaper, axspaper = plt.subplots(8, 1, figsize=(6, figsize1 * 8))
            temp_saver = []
            show_save = {}   # stimulus label -> {fixon, task, combine} traces
            for batch_iter in range(batch_nums):
                labels_for_batch = labels[batch_iter, 0]
                if labels_for_batch in label_index and labels_for_batch not in temp_saver:
                    f_fixon = saver2[batch_iter, 0, 1]
                    f_task = saver2[batch_iter, 3, 1]
                    f_bias = saver2[batch_iter, -1, 1]
                    k = len(temp_saver)
                    if k >= len(axspaper):
                        break
                    axspaper[k].plot(f_fixon, color=c_vals[0], linestyle=l_vals[0], label="Fixon")
                    axspaper[k].plot(f_task + f_bias, color=c_vals[1], linestyle=l_vals[1], label="Task")
                    axspaper[k].plot(f_fixon + f_task + f_bias, color=c_vals[2], linestyle=l_vals[3],
                                     linewidth=3, label="Combine")
                    axspaper[k].axhline(0, color=c_vals[3])
                    axspaper[k].set_xlabel("Time (ms)", fontsize=15)
                    # Traces are per step index; relabel ticks in ms (index * dt).
                    axspaper[k].xaxis.set_major_formatter(
                        ticker.FuncFormatter(lambda x, _pos: f"{x * dt:.0f}"))
                    axspaper[k].set_ylabel("Modulation Component", fontsize=15)
                    show_save[int(labels_for_batch)] = {
                        "fixon": np.asarray(f_fixon, dtype=float),
                        "task": np.asarray(f_task + f_bias, dtype=float),
                        "combine": np.asarray(f_fixon + f_task + f_bias, dtype=float),
                    }
                    temp_saver.append(labels_for_batch)
            for axsp in axspaper:
                axsp.legend(loc="best", frameon=True, fontsize=12)
                axsp.set_ylim([-2.0, 2.0])
            figpaper.tight_layout()
            figpaper.savefig(save_dir / f"show_{aname}.png", dpi=300)
            plt.close(figpaper)
            print(f"  Saved figure: {save_dir / f'show_{aname}.png'}")

            # Save the underlying traces so paper_plot can re-render this figure.
            import pickle as _pickle
            show_pkl = {
                "aname": aname,
                "stage_iter": int(stage_iter),
                "all_breaks": all_breaks,
                "response_start": int(response_start),
                "stimulus_start": int(stimulus_start),
                "stimulus_end": int(stimulus_end),
                "per_stimulus": show_save,  # {stim_label: {fixon, task, combine}}
            }
            with open(save_dir / f"show_{aname}.pkl", "wb") as _f:
                _pickle.dump(show_pkl, _f)
            print(f"  Saved show data: {save_dir / f'show_{aname}.pkl'}")

            for i in range(saver_shape1[0]):
                for j in range(saver_shape1[1]):
                    axsexh1[i, j].set_ylim([-1.2, 1.2])
                    axsexh1[i, j].set_title(f"{allX1name[i]} & {allYname[j]}")
            for i in range(saver_shape2[0]):
                for j in range(saver_shape2[1]):
                    axsexh2[i, j].set_ylim([-1.2, 1.2])
                    axsexh2[i, j].set_title(f"{allX2name[i]} & {allYname[j]}")
            for ax in np.concatenate((axsexh1.flatten(), axsexh2.flatten())):
                for bi, breaks in enumerate(all_breaks):
                    for bb in breaks:
                        ax.axvline(bb, linestyle="--", c=c_vals[bi % len(c_vals)])

            figexh1.suptitle(f"Exhaustive Search 1 {color_by} at Stage {stage_iter}")
            figexh1.tight_layout()
            figexh1.savefig(save_dir / f"es1_{aname}.png", dpi=300)
            plt.close(figexh1)
            print(f"  Saved figure: {save_dir / f'es1_{aname}.png'}")
            figexh2.suptitle(f"Exhaustive Search 2 {color_by} Stage {stage_iter}")
            figexh2.tight_layout()
            figexh2.savefig(save_dir / f"es2_{aname}.png", dpi=300)
            plt.close(figexh2)
            print(f"  Saved figure: {save_dir / f'es2_{aname}.png'}")
            axsdiff[0].set_title("Stimulus 1")
            axsdiff[1].set_title("Stimulus 2")
            figdiff.suptitle(f"Fixon-Task at Stage {stage_iter}")
            figdiff.tight_layout()
            figdiff.savefig(save_dir / f"diff_{aname}.png", dpi=300)
            plt.close(figdiff)
            print(f"  Saved figure: {save_dir / f'diff_{aname}.png'}")

        return task_labels_across_batch, saver2, saver2_random

    all_trajectory = []
    label_index = np.unique(labels)
    for stage_iter in range(stages_num):
        _, saver2, _ = plot_trajectory_by_index(
            label_index, stage_iter, verbose=(stage_iter == stages_num - 1))
        all_trajectory.append(saver2)

    def analyze_trajectory(save_trajectory):
        def process(trajectory):
            results = []
            for batch in trajectory:
                if batch[0, 1] is None:
                    continue
                # Average over the STIMULUS + DELAY period (stimulus_start:response_start).
                stim1_fixon = batch[0, 1][stimulus_start:response_start]
                stim1_task = batch[3, 1][stimulus_start:response_start]
                bias = batch[4, 1][stimulus_start:response_start]
                results.append([np.mean(np.abs(stim1_fixon + stim1_task + bias)),
                                np.mean(np.abs(stim1_fixon)), np.mean(np.abs(stim1_task))])
            return np.array(results)
        result = process(save_trajectory)
        # Per-quantity mean and standard error of the mean (std / sqrt(n)) across
        # trials, so the error band is the SEM rather than the raw trial spread.
        n = max(result.shape[0], 1)
        return result.mean(axis=0), result.std(axis=0) / np.sqrt(n)

    cancel_stats = [analyze_trajectory(all_trajectory[i]) for i in range(stages_num)]
    cancel_mean = np.array([s[0] for s in cancel_stats])   # (stages, 3)
    cancel_sem = np.array([s[1] for s in cancel_stats])    # (stages, 3) std/sqrt(n)

    figc, axc = plt.subplots(figsize=(6, 3))
    cancel_labels = [r"|Fix − Task|", r"|Fix|", r"|Task|"]
    for k in range(3):
        axc.plot(counter_lst, cancel_mean[:, k], "-o", color=c_vals[k],
                 label=cancel_labels[k])
        axc.fill_between(counter_lst, cancel_mean[:, k] - cancel_sem[:, k],
                         cancel_mean[:, k] + cancel_sem[:, k],
                         color=c_vals_l[k], alpha=0.2)
    axc.legend(loc="best", fontsize=12, frameon=True)
    axc.set_ylabel("Magnitude Projection", fontsize=15)
    axc.set_xlabel("# Dataset", fontsize=15)
    axc.set_xscale("log")
    figc.tight_layout()
    figc.savefig(save_dir / f"cancel_{aname}.png", dpi=300)
    plt.close(figc)
    print(f"  Saved figure: {save_dir / f'cancel_{aname}.png'}")

    # Save the cancel-curve data (stimulus+delay mean ± SEM = std/sqrt(n) across
    # trials, per training checkpoint) so paper_plot can re-render this figure.
    import pickle as _pickle
    with open(save_dir / f"cancel_{aname}.pkl", "wb") as _f:
        _pickle.dump({
            "aname": aname,
            "counter_lst": np.asarray(counter_lst, dtype=float),  # (stages,) # datasets
            "cancel_mean": np.asarray(cancel_mean, dtype=float),  # (stages, 3)
            "cancel_sem": np.asarray(cancel_sem, dtype=float),    # (stages, 3) std/sqrt(n)
            "labels": cancel_labels,                              # ["|Fix − Task|","|Fix|","|Task|"]
        }, _f)
    print(f"  Saved cancel data: {save_dir / f'cancel_{aname}.pkl'}")

    # ── Modulation-change / synaptic & hidden correlation across learning ────
    modulation_dict_diff_lst, modulation_dict_lst = [], []
    hidden_output_dict_lst, hidden_dict_lst = [], []

    # Fixon column of the input embedding (see m_pca note): M's last axis is the
    # embedded input, so the fixon-channel modulation effect on hidden units is
    # M @ W_input[:, fixon_col], not M[..., 0].
    _fixon_col = 0
    for stage_iter in range(stages_num):
        Woutput = Woutput_stages[stage_iter]
        Ms_orig = Ms_orig_stages[stage_iter]
        hs = hs_stages[stage_iter]
        hs_stimulus = hs[:, stimulus_start:stimulus_end, :]
        if input_layer_add and Winput_stages.size > 0:
            w_fixon = Winput_stages[stage_iter][:, _fixon_col]   # (embed,)
            Ms_fixon_proj = Ms_orig @ w_fixon                    # (batch, T, hidden)
        else:
            Ms_fixon_proj = Ms_orig[:, :, :, _fixon_col]
        # Per-period fixon-modulation traces (projected onto the raw fixon input).
        Mf_fix = Ms_fixon_proj[:, :stimulus_start, :]
        Mf_stimulus = Ms_fixon_proj[:, stimulus_start:stimulus_end, :]
        Mf_delay = Ms_fixon_proj[:, stimulus_end:response_start, :]
        Mf_response = Ms_fixon_proj[:, response_start:, :]
        Mf_all = [Mf_fix, Mf_stimulus, Mf_delay, Mf_response]

        modulation_diff_dict, modulation_dict, hidden_output_dict, hidden_dict = {}, {}, {}, {}
        for batch_iter in range(batch_nums):
            hs_stim_batch = hs_stimulus[batch_iter, :, :]
            hs_stim_out = hs_stim_batch @ Woutput.T
            # change of fixon modulation (end - start) per period
            Ms_fixon = [Mf[batch_iter, -1, :] - Mf[batch_iter, 0, :] for Mf in Mf_all]
            modulation_diff_dict[labels[batch_iter, 0]] = Ms_fixon
            # fixon modulation at end of stimulus (for synaptic-cosine analysis)
            modulation_dict[labels[batch_iter, 0]] = Mf_stimulus[batch_iter, -1, :]
            hidden_output_dict[labels[batch_iter, 0]] = hs_stim_out
            hidden_dict[labels[batch_iter, 0]] = hs_stim_batch[-1, :]
        modulation_dict_diff_lst.append(modulation_diff_dict)
        modulation_dict_lst.append(modulation_dict)
        hidden_output_dict_lst.append(hidden_output_dict)
        hidden_dict_lst.append(hidden_dict)

    modulation_change_stage = [[], [], [], []]
    m_corr_stage, h_corr_stage = [], []
    fig_hc, axs_hc = plt.subplots(2, 1, figsize=(6, 3 * 2))

    def analyze_hm_change(lst, i, index=None):
        md = lst[i]
        if index is None:
            md_m = [np.array(v) for v in md.values()]
        else:
            md_m = [np.array(v[index]) for v in md.values()]
        md_m = np.column_stack(md_m).T  # num_stimulus x hidden
        mc_stage = list(np.mean(np.abs(md_m), axis=1))
        synaptic_corr = cosine_similarity(md_m)
        # Mean pairwise cosine over the STRICT upper triangle (k=1): exclude the
        # all-ones diagonal and avoid double-counting / the zeroed lower triangle,
        # so the result is a genuine cosine in [-1, 1].
        n = synaptic_corr.shape[0]
        iu = np.triu_indices(n, k=1)
        mean_cos = float(np.nanmean(synaptic_corr[iu])) if n > 1 else np.nan
        return mean_cos, mc_stage, md_m

    for i in range(stages_num):
        m_mean_corr, _, _ = analyze_hm_change(modulation_dict_lst, i)
        h_mean_corr, _, _ = analyze_hm_change(hidden_dict_lst, i)
        m_corr_stage.append(m_mean_corr)
        h_corr_stage.append(h_mean_corr)
        md_m_diff_stim = md_m_diff_response = None
        for p in range(4):
            _, mc_stage, md_m_diff = analyze_hm_change(modulation_dict_diff_lst, i, p)
            if p == 1:
                md_m_diff_stim = md_m_diff
            elif p == 3:
                md_m_diff_response = md_m_diff
            modulation_change_stage[p].append(mc_stage)
        if i == stages_num - 1:
            sns.heatmap(md_m_diff_stim, ax=axs_hc[0], cmap="coolwarm")
            sns.heatmap(md_m_diff_response, ax=axs_hc[1], cmap="coolwarm")
            for ax_hc in axs_hc:
                ax_hc.set_xticks([]); ax_hc.set_yticks([])
                ax_hc.set_xlabel("Hidden", fontsize=15)
                ax_hc.set_ylabel("Stimuli", fontsize=15)
            fig_hc.tight_layout()
            fig_hc.savefig(save_dir / f"modulation_heatmap_{aname}.png", dpi=300)
            plt.close(fig_hc)
            print(f"  Saved figure: {save_dir / f'modulation_heatmap_{aname}.png'}")

            # Save the two heatmap matrices for paper_plot reuse.
            import pickle as _pickle
            with open(save_dir / f"modulation_heatmap_{aname}.pkl", "wb") as _f:
                _pickle.dump({
                    "aname": aname,
                    "stage_iter": int(i),
                    "stim_change": np.asarray(md_m_diff_stim, dtype=float),       # (n_stim, hidden)
                    "response_change": np.asarray(md_m_diff_response, dtype=float),
                }, _f)
            print(f"  Saved modulation heatmap data: "
                  f"{save_dir / f'modulation_heatmap_{aname}.pkl'}")

            # ── Full-M snapshots for the 2x2 (stimulus x period) figure ──────
            # Save the raw plasticity matrix M (hidden x input) at the middle of
            # the stimulus period and the middle of the response period, for two
            # example stimuli (labels 1 and 5). Reused by paper_plot's
            # plot_onetask_modulation_snapshot.
            T_full = Ms_orig.shape[1]
            t_mid_stim = (stimulus_start + stimulus_end) // 2
            t_mid_resp = (response_start + T_full) // 2
            # Map each stimulus label to its (first) batch row in this stage.
            label_to_batch = {}
            for batch_iter in range(batch_nums):
                label_to_batch.setdefault(int(labels[batch_iter, 0]), batch_iter)
            # Effective modulation W⊙M uses the recurrent MP-layer weight of
            # this stage (hidden x embed, matching Ms_orig's last two axes).
            W_eff_snap = Wall_stages[i]
            snapshot_stims = [1, 5]
            snapshots = {}          # raw M
            snapshots_eff = {}      # effective modulation W⊙M
            hidden_snapshots = {}   # hidden state at the same timepoints
            for s in snapshot_stims:
                if s not in label_to_batch:
                    continue
                b = label_to_batch[s]
                snapshots[s] = {
                    "stimulus": np.asarray(Ms_orig[b, t_mid_stim, :, :], dtype=float),
                    "response": np.asarray(Ms_orig[b, t_mid_resp, :, :], dtype=float),
                }
                snapshots_eff[s] = {
                    "stimulus": np.asarray(Ms_orig[b, t_mid_stim, :, :] * W_eff_snap, dtype=float),
                    "response": np.asarray(Ms_orig[b, t_mid_resp, :, :] * W_eff_snap, dtype=float),
                }
                # Hidden-state vector at the same two timepoints (for the hidden
                # illustration alongside the M / W⊙M snapshot figures).
                hidden_snapshots[s] = {
                    "stimulus": np.asarray(hs[b, t_mid_stim, :], dtype=float),
                    "response": np.asarray(hs[b, t_mid_resp, :], dtype=float),
                }
            with open(save_dir / f"modulation_snapshot_{aname}.pkl", "wb") as _f:
                _pickle.dump({
                    "aname": aname,
                    "stage_iter": int(i),
                    "stims": [s for s in snapshot_stims if s in snapshots],
                    "t_mid_stim": int(t_mid_stim),
                    "t_mid_resp": int(t_mid_resp),
                    "snapshots": snapshots,       # {stim: {"stimulus", "response"}} raw M
                    "snapshots_eff": snapshots_eff,  # {stim: {"stimulus", "response"}} W⊙M
                    "hidden_snapshots": hidden_snapshots,  # {stim: {"stimulus", "response"}} hidden vec
                }, _f)
            print(f"  Saved modulation snapshot data: "
                  f"{save_dir / f'modulation_snapshot_{aname}.pkl'}")

    modulation_change_stage = np.array(modulation_change_stage)
    m_corr_stage = np.array(m_corr_stage)
    h_corr_stage = np.array(h_corr_stage)
    period_names = ["Fixation", "Stimulus", "Delay", "Response"]

    figmc, axsmc = plt.subplots(3, 1, figsize=(6, 3 * 3))
    for i in range(4):
        mcs = modulation_change_stage[i]
        axsmc[0].plot(counter_lst, np.mean(mcs, axis=1), "-o", c=c_vals[i], label=period_names[i])
        axsmc[0].fill_between(counter_lst, np.mean(mcs, axis=1) - np.std(mcs, axis=1),
                              np.mean(mcs, axis=1) + np.std(mcs, axis=1), color=c_vals_l[i])
    axsmc[0].set_ylabel("Change of Modulation", fontsize=15)
    axsmc[0].legend(loc="best", frameon=True, fontsize=12)
    axsmc[1].plot(counter_lst, m_corr_stage, "-o")
    axsmc[1].set_ylabel("Synaptic Cosine\nbetween Stimulus", fontsize=13)
    axsmc[2].plot(counter_lst, h_corr_stage, "-o")
    axsmc[2].set_ylabel("Hidden Activity Cosine\nbetween Stimulus", fontsize=13)
    for ax in axsmc:
        ax.set_xlabel("# Dataset", fontsize=15)
        ax.set_xscale("log")
    figmc.tight_layout()
    figmc.savefig(save_dir / f"modulation_change_{aname}.png", dpi=300)
    plt.close(figmc)
    print(f"  Saved figure: {save_dir / f'modulation_change_{aname}.png'}")

    # Save the raw (un-normalized) mean cosine curves for cross-seed plotting.
    ONETASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ONETASK_DATA_DIR / f"corr_{aname}.npz",
             counter_lst=counter_lst,
             m_corr_stage=m_corr_stage,
             h_corr_stage=h_corr_stage)

    # ── Hidden-trajectory length across learning ─────────────────────────────
    def traj_length(arr):
        return np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))

    hidden_length_all = []
    for stage_iter in range(stages_num):
        hstage = hidden_output_dict_lst[stage_iter]
        hstage = {k: hstage[k] for k in sorted(hstage.keys())}
        hidden_length_all.append([traj_length(arr) for arr in hstage.values()])
    hidden_length_all = np.array(hidden_length_all)

    figt, axst = plt.subplots(figsize=(6, 3))
    for i in range(hidden_length_all.shape[1]):
        axst.plot(counter_lst, hidden_length_all[:, i], "-o", c=c_vals[i % len(c_vals)])
    axst.set_xlabel("# Dataset", fontsize=15)
    axst.set_xscale("log")
    axst.set_ylabel("Length of Hidden\nState Trajectory", fontsize=13)
    figt.tight_layout()
    figt.savefig(save_dir / f"length_hidden_state_{aname}.png", dpi=300)
    plt.close(figt)
    print(f"  Saved figure: {save_dir / f'length_hidden_state_{aname}.png'}")

    # ── Weight-component projection to output across learning ────────────────
    fixon_task_projoutput = []
    for stage_iter in range(stages_num):
        W = Wall_stages[stage_iter]
        W_output = Woutput_stages[stage_iter]
        bias = np.mean(bs_stages[stage_iter], axis=0)
        # W is in the embedded space; compose the input embedding so we read the
        # raw fixon / task channels' effective weight onto the hidden units.
        if input_layer_add and Winput_stages.size > 0:
            W_input = Winput_stages[stage_iter]
            W_fixon = (W @ W_input[:, 0]).reshape(-1, 1)
            W_task = (W @ W_input[:, 6 - shift_index]).reshape(-1, 1)
        else:
            W_fixon = W[:, 0].reshape(-1, 1)
            W_task = W[:, 6 - shift_index].reshape(-1, 1)
        fixon_output, task_output = W_output[1:, :] @ W_fixon, W_output[1:, :] @ W_task
        bias_output = np.mean(bias @ (W_output[1:, :].T), axis=0)
        fixon_task_projoutput.append([
            fixon_output[0][0] + bias_output[0], task_output[0][0],
            fixon_output[1][0] + bias_output[1], task_output[1][0],
        ])
    fixon_task_projoutput = np.array(fixon_task_projoutput)

    figw, axw = plt.subplots(figsize=(6, 3))
    axw.plot(counter_lst, fixon_task_projoutput[:, 0], marker="o", color=c_vals[0], linestyle=l_vals[0], label="fixon→out1")
    axw.plot(counter_lst, fixon_task_projoutput[:, 1], marker="o", color=c_vals[0], linestyle=l_vals[1], label="task→out1")
    axw.plot(counter_lst, fixon_task_projoutput[:, 0] + fixon_task_projoutput[:, 1], marker="o",
             color=c_vals[1], linestyle=l_vals[2], linewidth=1, label="fixon+task→out1")
    axw.axhline(0, color=c_vals[1], linestyle=l_vals[2])
    axw.legend(loc="lower left", fontsize=12, frameon=True)
    axw.set_xlabel("# Dataset", fontsize=15)
    axw.set_xscale("log")
    axw.set_ylabel("Weight Component\nProjection", fontsize=13)
    figw.tight_layout()
    figw.savefig(save_dir / f"w_to_output_{aname}.png", dpi=300)
    plt.close(figw)
    print(f"  Saved figure: {save_dir / f'w_to_output_{aname}.png'}")

    # ── Recover the fixon modulation and hidden state for the full-trial PCA ──
    #
    # The recorded modulation tensor M has shape (batch, T, hidden, embed): its
    # last axis is the EMBEDDED input (output of the trained input layer
    # W_initial_linear), NOT the raw input channels. So M[..., 0] would be
    # embedded-dim 0, a meaningless mixed coordinate — not fixon. The effective
    # modulation of the raw fixon input on each hidden unit is obtained by
    # contracting M's embedded-input axis with the fixon column of the embedding
    # (w_fixon = W_input[:, fixon_col], shape (embed,)):
    #     fixon_mod[b, t, hidden] = sum_e  M[b, t, hidden, e] * w_fixon[e]
    # giving a per-timestep hidden-unit vector (batch, T, hidden). (Without an
    # input layer, M's last axis IS the raw input, so slice it.)
    stage_iter = stages_num - 1
    PCA_downsample = 3
    Ms_orig = Ms_orig_stages[stage_iter]               # (batch, T, hidden, embed)
    fixon_col = 0  # raw input column for the fixation-on channel

    if input_layer_add and Winput_stages.size > 0:
        w_fixon = Winput_stages[stage_iter][:, fixon_col]   # (embed,)
        fixon_mod = Ms_orig @ w_fixon                       # (batch, T, hidden)
    else:
        fixon_mod = Ms_orig[:, :, :, fixon_col]             # raw input already

    hs_now = hs_stages[stage_iter]                          # (batch, T, hidden)

    import pickle as _pickle

    # ── FULL-trial trajectory in a whole-trial PCA basis (two-task style) ────
    # Matches the two-task m_pca_*_normal figure: fit the top-3 PCA on the
    # FULL-trial states pooled over all trials AND all timesteps, then plot the
    # entire trajectory across 3 PC planes (PC1-2 / PC1-3 / PC2-3), colored by
    # stimulus. Each trial-period is marked with a distinct marker over its
    # window, and each period boundary gets a large solid transition marker.
    T_total = fixon_mod.shape[1]
    # phase name -> (start, end_exclusive, marker index in markers_vals)
    phases = [("Fixation", 0, stimulus_start, 1),
              ("Stimulus", stimulus_start, stimulus_end, 2),
              ("Delay", stimulus_end, response_start, 3),
              ("Response", response_start, T_total, 0)]
    transition_ts = [(stimulus_start, 2), (stimulus_end, 3), (response_start, 0)]
    pcs_comb = [(0, 1), (0, 2), (1, 2)]
    legend_handles = [plt.Line2D([0], [0], marker=markers_vals[mk], linestyle="None",
                                 markersize=10, markerfacecolor="k", markeredgecolor="k",
                                 label=name)
                      for name, _, _, mk in phases]

    def _full_traj_pca(data_bt_feat, tag, ylabel, show_legend=True):
        """data_bt_feat: (batch, T, feat). PCA basis fit on the whole trial
        (every timestep of every trial, like two_task m_pca_*_normal), plotted in
        two-task style: line + per-phase markers + transition markers."""
        n_activity = data_bt_feat.shape[-1]
        as_flat = data_bt_feat.reshape(-1, n_activity)        # (batch*T, feat)
        pca_d = PCA(n_components=PCA_downsample, random_state=42)
        pca_d.fit(as_flat)
        proj = pca_d.transform(as_flat).reshape(
            data_bt_feat.shape[0], data_bt_feat.shape[1], PCA_downsample)

        figd, axsd = plt.subplots(1, 3, figsize=(5 * 3, 5), squeeze=False)
        for i in range(proj.shape[0]):
            db = proj[i, :, :]
            color = c_vals[labels[i, 0] % len(c_vals)]
            for col, (a, bb) in enumerate(pcs_comb):
                ax = axsd[0, col]
                # full-trial trajectory line
                ax.plot(db[:, a], db[:, bb], c=color, alpha=0.25, zorder=2)
                # per-phase markers over each phase window
                for _name, t0, t1, mk in phases:
                    sl = slice(t0, t1)
                    ax.scatter(db[sl, a], db[sl, bb], c=color,
                               marker=markers_vals[mk], alpha=0.5, zorder=3)
                # large solid transition markers at period boundaries
                for t, mk in transition_ts:
                    tt = min(max(t - 1, 0), db.shape[0] - 1)
                    ax.scatter([db[tt, a]], [db[tt, bb]], c=color,
                               marker=markers_vals[mk], alpha=0.8, s=60,
                               linewidths=0.6, zorder=10)
        for col, (a, bb) in enumerate(pcs_comb):
            ax = axsd[0, col]
            ax.set_xlabel(f"PCA {a+1}", fontsize=12)
            ax.set_ylabel(f"PCA {bb+1}", fontsize=12)
            ax.set_title(f"{ylabel}", fontsize=15)
            ax.tick_params(axis="both", labelsize=12)
            if show_legend:
                ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=10)
        figd.suptitle(f"{aname}  |  full trajectory in whole-trial PCA", fontsize=12)
        figd.tight_layout()
        out = save_dir / f"{tag}_pca_fulltrial_{aname}.png"
        figd.savefig(out, dpi=300)
        plt.close(figd)
        print(f"  Saved figure: {out}")

        with open(save_dir / f"{tag}_pca_fulltrial_{aname}.pkl", "wb") as _f:
            _pickle.dump({
                "aname": aname,
                "stage_iter": int(stage_iter),
                "lowd": np.asarray(proj, dtype=float),       # (batch, T, n_pc) FULL trial
                "labels": np.asarray(labels).reshape(-1),
                "pcs_comb": pcs_comb,
                "phases": [(n, int(t0), int(t1), int(mk)) for n, t0, t1, mk in phases],
                "stimulus_start": int(stimulus_start),
                "stimulus_end": int(stimulus_end),
                "response_start": int(response_start),
                "explained_variance_ratio": np.asarray(pca_d.explained_variance_ratio_, dtype=float),
            }, _f)
        print(f"  Saved {tag}_pca_fulltrial data: {save_dir / f'{tag}_pca_fulltrial_{aname}.pkl'}")

    _full_traj_pca(fixon_mod, "m", "fixon modulation", show_legend=False)
    _full_traj_pca(hs_now, "h", "hidden activity")

    # Full modulation: flatten M's (hidden, embed) feature axes into a single
    # vector per timestep and run the same whole-trial PCA over the ENTIRE
    # modulation matrix (every input channel), not just the fixon projection.
    all_mod = Ms_orig.reshape(Ms_orig.shape[0], Ms_orig.shape[1], -1)  # (batch, T, hidden*embed)
    _full_traj_pca(all_mod, "m_all", "all modulation", show_legend=False)

    # Effective modulation W⊙M: the actual weight change applied to the
    # recurrent connections (see e_modulation in long_period_fixed_points).
    # Wall is the (hidden, embed) MP-layer weight, matching M's last two axes.
    W_eff = Wall_stages[stage_iter]                                    # (hidden, embed)
    eff_mod = (Ms_orig * W_eff[None, None, :, :]).reshape(
        Ms_orig.shape[0], Ms_orig.shape[1], -1)                       # (batch, T, hidden*embed)
    _full_traj_pca(eff_mod, "e_mod", "effective modulation", show_legend=False)

    # ── Long-period fixed-point geometry (uses the LIVE trained network) ─────
    # Generate test data with each trial period extended in turn, fit a top-2
    # PCA on the pooled delay-period states, and scatter each variant's fixed
    # point (last delay frame) colored by stimulus — for hidden and W⊙M.
    try:
        long_period_fixed_points(aname, save_dir, cfg, seed, shift_index, color_by,
                                 fp_n_seeds=fp_n_seeds,
                                 run_fixed_points=run_fixed_points)
    except Exception as exc:
        print(f"  [long-fp] failed: {exc}")
        import traceback
        traceback.print_exc()

    # ── Cross-period PCA explained-variance (one-task analog of two-task's
    # d_combine): how much each trial period's subspace captures the others,
    # for hidden / modulation / effective modulation. Uses the final-stage
    # hs_now, Ms_orig, and W_eff computed above.
    try:
        cross_period_dimensionality(aname, save_dir, hs_now, Ms_orig, W_eff,
                                    stimulus_start, stimulus_end, response_start,
                                    top_k=2)
    except Exception as exc:
        print(f"  [d_combine] failed: {exc}")
        import traceback
        traceback.print_exc()

    # ── Modulation-computation magnitude across time, per input component ─────
    # For every raw input channel, the L2 magnitude (over hidden units) of the
    # modulation the plastic weights apply to that channel, M @ W_input[:, c],
    # averaged over trials. Uses the final-stage Ms_orig from the PCA section.
    try:
        n_raw = test_input_np.shape[-1]
        # Input-embedding columns mapped to raw channels (identity if no layer;
        # then M's last axis already IS the raw input).
        if input_layer_add and Winput_stages.size > 0:
            W_input_final = Winput_stages[stage_iter]          # (embed, n_raw)
        else:
            W_input_final = np.eye(Ms_orig.shape[-1])
        # Full component layout, named like the example-trial figure: fixation,
        # the two stimulus modalities' cos/sin, and the task cue (last channel).
        task_ch = n_raw - 1
        mag_specs = [(0, "Fixation")]
        for (cos_ch, sin_ch), mod_name in (((1, 2), "Mod1"), ((3, 4), "Mod2")):
            if sin_ch < task_ch:
                mag_specs += [(cos_ch, f"{mod_name} cos"),
                              (sin_ch, f"{mod_name} sin")]
        mag_specs.append((task_ch, "Task cue"))
        modulation_magnitude_by_component(
            aname, save_dir, Ms_orig, W_input_final, mag_specs, dt,
            stimulus_start, stimulus_end, response_start)
    except Exception as exc:
        print(f"  [mod-magnitude] failed: {exc}")
        import traceback
        traceback.print_exc()

    print(f"All figures saved to {save_dir}/")

    # ── Cross-seed aggregate of correlation curves ───────────────────────────
    _plot_aggregate_corr(save_dir, aname)


def _plot_aggregate_corr(save_dir, aname):
    """Average the m_corr / h_corr curves across all saved corr_*.npz runs."""
    files = sorted(glob.glob(str(ONETASK_DATA_DIR / "corr_*.npz")))
    if len(files) < 1:
        return
    counters, m_all, h_all = [], [], []
    for f in files:
        d = np.load(f)
        counters.append(d["counter_lst"])
        m_all.append(d["m_corr_stage"])
        h_all.append(d["h_corr_stage"])
    # only aggregate runs with a common length
    lens = [len(c) for c in counters]
    common = min(lens)
    counters = np.array([c[:common] for c in counters])
    m_all = np.array([m[:common] for m in m_all])
    h_all = np.array([h[:common] for h in h_all])

    mean_counter = np.mean(counters, axis=0)
    figm, axm = plt.subplots(2, 1, figsize=(6, 3 * 2))
    axm[0].plot(mean_counter, m_all.mean(0), "-o", color=c_vals[0])
    axm[0].fill_between(mean_counter, m_all.mean(0) - m_all.std(0), m_all.mean(0) + m_all.std(0),
                        color=c_vals_l[0], alpha=0.2)
    axm[0].set_ylabel("Cos of Modulation", fontsize=14)
    axm[1].plot(mean_counter, h_all.mean(0), "-o", color=c_vals[0])
    axm[1].fill_between(mean_counter, h_all.mean(0) - h_all.std(0), h_all.mean(0) + h_all.std(0),
                        color=c_vals_l[0], alpha=0.2)
    axm[1].set_ylabel("Cos of Hidden Activity", fontsize=14)
    for ax in axm:
        ax.set_xlabel("# Dataset", fontsize=15)
        ax.set_xscale("log")
    figm.tight_layout()
    figm.savefig(save_dir / f"modulation_analysis_during_learning_{aname}.png", dpi=300)
    plt.close(figm)
    print(f"Aggregated {len(files)} runs into modulation_analysis_during_learning_{aname}.png")


def _discover_anames():
    """Return all experiment identifiers (param_*_result.npz) in onetask/,
    sorted by modification time (oldest first)."""
    results = sorted(ONETASK_DIR.glob("param_*_result.npz"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise FileNotFoundError("No param_*_result.npz found in ./onetask/. Run one_task.py first.")
    return [p.name[len("param_"):-len("_result.npz")] for p in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aname", type=str, default=None,
                        help="Experiment identifier. Omit to analyze ALL runs in ./onetask/.")
    parser.add_argument("--fp-n-seeds", type=int, default=5,
                        help="Number of random trial templates to try when solving "
                             "gradient fixed points; the best-converging one is kept "
                             "(default 5).")
    parser.add_argument("--no-fixed-points", dest="run_fixed_points",
                        action="store_false",
                        help="Skip the time-consuming gradient fixed-point solver "
                             "(fixed_points_grad_*). On by default.")
    parser.set_defaults(run_fixed_points=True)
    args = parser.parse_args()

    anames = [args.aname] if args.aname else _discover_anames()
    print(f"Analyzing {len(anames)} run(s).")
    for a in anames:
        print(f"\n── Analyzing: {a} ──")
        try:
            main(a, fp_n_seeds=args.fp_n_seeds,
                 run_fixed_points=args.run_fixed_points)
        except Exception as exc:
            print(f"  FAILED {a}: {exc}")
            import traceback
            traceback.print_exc()
