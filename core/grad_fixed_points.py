"""
Shared driver for gradient-based modulation fixed-point analysis.

`core/fixed_point.py` provides the low-level optimizer
(`find_modulation_fixed_points`) that solves M* = F(M*; x) for a batch of
candidate modulation matrices under a constant input. This module wraps that
optimizer into the per-period, dense-stimulus analysis used by BOTH the
single-task (`one_task_analysis.py`) and two-task (`two_task_analysis.py`)
pipelines, so the logic lives in one place.

For each trial period (fixation / stimulus / delay / response) it:
  1. builds a dense grid of `n_interp` stimulus angles (bypassing the task
     generator's 8-way snapping) by copying one real trial template and
     overwriting its stimulus channels with (sin θ, cos θ);
  2. runs that batch once through the trained network to record M(t);
  3. seeds the optimizer at the end-of-period M and relaxes it while holding the
     period-midpoint input fixed, giving a TRUE fixed point M*;
  4. saves M*, its derived views (W⊙M*, hidden(M*), cos-output at M*), the
     scale-free convergence metric rel_step = ||F(M*)-M*|| / ||M*||, and an
     `is_fixed` mask (rel_step <= rel_tol).

Results are pickled to `fixed_points_grad_{aname}{out_suffix}.pkl`. The
`out_suffix` lets a multi-rule caller (two-task) write one file per rule.
"""

import copy
import pickle

import numpy as np
import torch

import mpn_tasks  # resolved via each experiment's _bootstrap (core/ on sys.path)
from fixed_point import (find_modulation_fixed_points,
                         characterize_fixed_point_stability)


# Long-period variant -> display title (shared with the plotting scripts).
_PERIOD_TITLE = {
    "longfixation": "Fixation",
    "longstimulus": "Stimulus",
    "longdelay": "Delay",
    "longresponse": "Response",
}


def solve_period_modulation_fixed_points(
        aname, save_dir, net, cfg, device,
        rule=None, out_suffix="",
        layer_index=1, W=None,
        n_interp=64, steps=200000, learningRate=1e-3,
        loss_tol=1e-8, lbfgs_steps=2000, rel_tol=0.05,
        stim_channels=None, n_seeds=5, seed_base=0,
        analyze_stability=True, n_eigs=16):
    """
    Solve TRUE gradient fixed points of the modulation matrix M per trial period.

    aname       : run identifier (used in the output filename).
    save_dir    : Path to write the pickle into.
    net         : trained (Deep)MultiPlasticNet (left untouched; the solver works
                  on a frozen deepcopy).
    cfg         : dict with keys "task_params", "train_params", "net_params"
                  (the two-task caller builds this from the checkpoint fields).
    rule        : task rule string to generate the dense-angle template for. If
                  None, uses cfg["task_params"]["rules"] (the single-task case).
    out_suffix  : appended to the pickle stem, e.g. "_delaygo" (multi-rule caller).
    layer_index : MP-layer db-key index (1 if net has an input embedding layer,
                  else 0); selects db["M{layer_index}"].
    W           : recurrent plastic weight matrix (hidden, embed) for the effective
                  modulation view W⊙M*; if None, that view is skipped.
    n_interp    : number of interpolated stimulus angles on [0, 2π).
    steps / learningRate / loss_tol / lbfgs_steps : optimizer settings passed to
                  find_modulation_fixed_points (Adam until loss<=loss_tol capped at
                  `steps`, then L-BFGS polishing).
    rel_tol     : a point counts as a fixed point when rel_step <= rel_tol.
    stim_channels : the two input channels holding the active ring's (sinθ, cosθ).
                  If None (default), they are AUTO-DETECTED from the template as
                  the channels energized during the stimulus window but ~zero
                  during fixation — robust to which ring modality a given rule
                  happens to drive (delaygo_ picks it at random). Pass an explicit
                  pair to override.
    n_seeds     : the trial template (stimulus modality, angle, epoch timing) is
                  drawn from the task RNG, so a single draw gives run-to-run
                  variability. We instead solve the whole per-period battery for
                  `n_seeds` DETERMINISTIC seeds (seed_base .. seed_base+n_seeds-1)
                  and keep the single seed whose fixed points converge best —
                  judged by the median rel_step over the STIMULUS + RESPONSE
                  periods only (the parts most sensitive to the template). This
                  makes the result both reproducible and the best of several
                  candidates. Set n_seeds=1 to solve a single deterministic seed.
    seed_base   : first task-RNG seed to try.
    analyze_stability : if True, linearize F about each M* and record the leading
                  Jacobian eigenvalues (see fixed_point.characterize_fixed_point_
                  stability). Adds spectral_radius / n_unstable / n_marginal /
                  is_stable / eigenvalues per period. A lone marginal (|λ|≈1)
                  direction with the rest contracting is the ring-attractor
                  signature. Only the SELECTED seed is analyzed (cheap: ~O(n_eigs)
                  backward passes per point).
    n_eigs      : number of leading eigenvalues per fixed point.

    Writes fixed_points_grad_{aname}{out_suffix}.pkl and returns its path (or None
    if no period could be solved).
    """
    def _hidden_from_M(M_np, x_np):
        """Hidden state and cos-output readout produced by setting the layer's
        modulation to M_np and running one forward pass under input x_np. M is
        restored afterward. Returns (hidden (batch,hidden), out_cos (batch,)).

        out_cos is output channel 1 ("Output Cos"); it should be ~0 during
        fixation/stimulus/delay (fixation-on holds the output at zero via the
        fixon/task cancellation) and nonzero only in the response period — this
        is the z-axis of the 3D fixed-point figures."""
        mp = net.mp_layers[0]
        saved_M, saved_M_pre = mp.M, getattr(mp, "M_pre", None)
        with torch.no_grad():
            mp.M = torch.as_tensor(M_np, dtype=torch.float, device=device)
            xin = torch.as_tensor(x_np, dtype=torch.float, device=device)
            output, mpl_activities, _ = net.forward(xin, run_mode="minimal")
            hid = np.asarray(mpl_activities[-1].detach().cpu())   # last = MP-layer post-act
            out = np.asarray(output.detach().cpu())               # (batch, n_output)
            out_cos = out[:, 1] if out.shape[-1] > 1 else out[:, 0]
        mp.M = saved_M
        if saved_M_pre is not None:
            mp.M_pre = saved_M_pre
        return hid, out_cos

    tag = f"[grad-fp{'/' + rule if rule else ''}]"

    def _detect_stim_channels(template, fix_on, fix_off, stim_on, stim_off):
        """The active ring's (sin θ, cos θ) channel pair. A ring channel carries
        energy during the stimulus window but is ~zero throughout fixation (this
        rejects the always-on fixation bit and the constant rule-cue channels).
        We detect the single most energetic such channel, then recover its
        PARTNER from the layout — low-dim rings are consecutive (sin, cos)
        2-blocks anchored just after the leading fixation channel(s). Detecting
        one channel suffices because for an angle on an axis (θ≈0/90/180/270°) one
        of sin/cos is exactly zero over the whole window."""
        stim_energy = (template[stim_on:stim_off] ** 2).sum(axis=0)
        fix_max = np.abs(template[fix_on:fix_off]).max(axis=0)
        stim_only = np.where(fix_max <= 1e-6, stim_energy, 0.0)
        c0 = int(np.argmax(stim_only))
        if stim_only[c0] <= 1e-9:
            raise ValueError(
                f"could not auto-detect a stimulus channel for rule={rule} "
                f"(stim-only energies {np.round(stim_only, 3).tolist()}); "
                f"pass stim_channels explicitly.")
        on_in_fix = fix_max > 1e-6
        n_fix = 0
        while n_fix < on_in_fix.size and on_in_fix[n_fix]:
            n_fix += 1
        ring_start = n_fix + ((c0 - n_fix) // 2) * 2
        if ring_start + 1 >= template.shape[1]:
            raise ValueError(
                f"stimulus ring pair ({ring_start},{ring_start+1}) out of range "
                f"for rule={rule} (n_input={template.shape[1]}).")
        return ring_start, ring_start + 1

    def _solve_one_seed(task_seed):
        """Build the dense-angle template for a FIXED task RNG seed and solve the
        per-period fixed points. Returns (results_dict, angles). Deterministic in
        task_seed, so re-running is reproducible."""
        # ── Dense interpolated-stimulus batch (one normal trial per angle) ────
        tp = copy.deepcopy(cfg["task_params"])
        tp["long_fixation"] = tp["long_stimulus"] = tp["long_delay"] = tp["long_response"] = "normal"
        tp, trp, npp = mpn_tasks.convert_and_init_multitask_params(
            (tp, copy.deepcopy(cfg["train_params"]), copy.deepcopy(cfg["net_params"])))
        npp["prefs"] = mpn_tasks.get_prefs(tp["hp"])
        tp["hp"]["batch_size_train"] = 1
        # Pin the task RNG so the template (stimulus modality, angle, epoch
        # timing) is deterministic for this seed — the whole point of the sweep.
        tp["hp"]["seed"] = int(task_seed)
        tp["hp"]["rng"] = np.random.RandomState(int(task_seed))
        gen_rules = [rule] if rule is not None else tp["rules"]
        data, extra = mpn_tasks.generate_trials_wrap(
            tp, 1, rules=gen_rules, mode_input="random", device=device)
        _, trials, _ = extra
        template = np.asarray(data[0].detach().cpu())[0]      # (T, n_input)
        T = template.shape[0]

        def _ep(name):
            e = trials[0].epochs[name]
            return (0 if e[0] is None else int(e[0]),
                    T if e[1] is None else int(e[1]))
        fix_on, fix_off = _ep("fix1")
        stim_on, stim_off = _ep("stim1")
        delay_on, delay_off = _ep("delay1")
        resp_on, resp_off = _ep("go1")
        period_win = {
            "longfixation": (fix_on, fix_off),
            "longstimulus": (stim_on, stim_off),
            "longdelay":    (delay_on, delay_off),
            "longresponse": (resp_on, resp_off),
        }

        if stim_channels is not None:
            ch_a, ch_b = int(stim_channels[0]), int(stim_channels[1])
        else:
            ch_a, ch_b = _detect_stim_channels(template, fix_on, fix_off,
                                               stim_on, stim_off)
        print(f"  {tag} seed={task_seed}: stimulus ring channels ({ch_a}, {ch_b})")

        # The ring's two channels hold (sin θ, cos θ) in generator order.
        angles = np.arange(n_interp) * (2 * np.pi / n_interp)
        batch = np.repeat(template[None, :, :], n_interp, axis=0)   # (n_interp,T,n_in)
        batch[:, stim_on:stim_off, ch_a] = np.sin(angles)[:, None]
        batch[:, stim_on:stim_off, ch_b] = np.cos(angles)[:, None]

        x = torch.as_tensor(batch, dtype=torch.float, device=device)
        _, _, db = net.iterate_sequence_batch(
            x, run_mode="track_states", save_to_cpu=True, detach_saved=True)
        M_all = np.asarray(db[f"M{layer_index}"])                  # (n_interp,T,hid,emb)
        stim = np.arange(n_interp)

        results = {}
        for v, (ps, pe) in period_win.items():
            if not (0 <= ps < pe <= T):
                continue
            t_seed = min(pe - 1, T - 1)
            t_mid = min((ps + pe) // 2, T - 1)
            init_M = M_all[:, t_seed, :, :]
            const_input = batch[:, t_mid, :]

            print(f"  {tag} seed={task_seed} {v}: solving {n_interp} fixed points "
                  f"(seed t={t_seed}, input t={t_mid})")
            fixed_M, loss_hist, final_speeds = find_modulation_fixed_points(
                net, init_M, const_input, steps=steps, learningRate=learningRate,
                printPeriod=max(steps // 20, 1), loss_tol=loss_tol,
                lbfgs_steps=lbfgs_steps, device=device)

            fixed_WM = (fixed_M * np.asarray(W)[None, :, :]) if W is not None else None
            fixed_hidden, fixed_out_cos = _hidden_from_M(fixed_M, const_input)

            # Scale-free convergence metric rel_step = ||F(M*)-M*|| / ||M*||;
            # final_speeds is q = 1/2||F-M||^2, so ||F-M|| = sqrt(2 q).
            fm = np.asarray(fixed_M, dtype=float).reshape(fixed_M.shape[0], -1)
            step_norm = np.sqrt(2.0 * np.asarray(final_speeds, dtype=float))
            m_norm = np.maximum(np.linalg.norm(fm, axis=1), 1e-12)
            rel_step = step_norm / m_norm
            is_fixed = rel_step <= rel_tol
            print(f"  {tag} seed={task_seed} {v}: {int(is_fixed.sum())}/{is_fixed.size} "
                  f"converged (rel_step<= {rel_tol:g}); "
                  f"median {np.median(rel_step):.2e} max {rel_step.max():.2e}")

            results[v] = {
                "period_title": _PERIOD_TITLE.get(v, v),
                "period": (int(ps), int(pe)),
                "t_seed": int(t_seed),
                "t_input": int(t_mid),
                "init_M": np.asarray(init_M, dtype=np.float32),
                "fixed_M": np.asarray(fixed_M, dtype=np.float32),
                "fixed_WM": (np.asarray(fixed_WM, dtype=np.float32)
                             if fixed_WM is not None else None),
                "fixed_hidden": np.asarray(fixed_hidden, dtype=np.float32),
                "fixed_out_cos": np.asarray(fixed_out_cos, dtype=np.float32),
                "final_speeds": np.asarray(final_speeds, dtype=float),
                "rel_step": np.asarray(rel_step, dtype=float),
                "is_fixed": np.asarray(is_fixed, dtype=bool),
                "rel_tol": float(rel_tol),
                "loss_hist": np.asarray(loss_hist, dtype=float),
                "stim": np.asarray(stim),
                # Constant input this period's M* was solved under; kept so the
                # (deferred) stability analysis can linearize F at the same point.
                "const_input": np.asarray(const_input, dtype=np.float32),
            }
        return results, angles

    def _selection_score(results):
        """Lower = better. Median rel_step over the STIMULUS + RESPONSE periods
        only (the parts most sensitive to the random template). Missing periods
        contribute nothing; if neither is present, fall back to all periods."""
        keys = [k for k in ("longstimulus", "longresponse") if k in results]
        if not keys:
            keys = list(results.keys())
        vals = np.concatenate([np.asarray(results[k]["rel_step"], float)
                               for k in keys]) if keys else np.array([np.inf])
        return float(np.median(vals))

    # ── Try n_seeds deterministic templates; keep the best-converging one ────
    best = None   # (score, task_seed, results, angles)
    for s in range(seed_base, seed_base + max(int(n_seeds), 1)):
        try:
            results, angles = _solve_one_seed(s)
        except Exception as exc:
            print(f"  {tag} seed={s} failed: {exc}")
            continue
        if not results:
            continue
        score = _selection_score(results)
        print(f"  {tag} seed={s}: selection score (stim+resp median rel_step) "
              f"= {score:.3e}")
        if best is None or score < best[0]:
            best = (score, s, results, angles)

    if best is None:
        print(f"  {tag} no seed produced fixed points; skipping save.")
        return None

    best_score, best_seed, results, angles = best
    print(f"  {tag} selected seed={best_seed} (score {best_score:.3e} over "
          f"{n_seeds} seed(s)).")

    # ── Linear-stability analysis on the SELECTED seed's fixed points ────────
    # Linearize F about each M* and record the leading Jacobian eigenvalues, so
    # the fixed points can be classified stable / marginal (ring) / unstable.
    if analyze_stability:
        for v, e in results.items():
            try:
                stab = characterize_fixed_point_stability(
                    net, e["fixed_M"], e["const_input"], k=n_eigs, device=device)
            except Exception as exc:
                print(f"  {tag} {v}: stability analysis failed: {exc}")
                continue
            e.update({
                "eigenvalues": stab["eigenvalues"],
                "spectral_radius": stab["spectral_radius"],
                "n_unstable": stab["n_unstable"],
                "n_marginal": stab["n_marginal"],
                "stab_is_stable": stab["is_stable"],
                "marginal_tol": stab["marginal_tol"],
            })
            rad = stab["spectral_radius"]
            print(f"  {tag} {v}: spectral radius median {np.nanmedian(rad):.3f} "
                  f"(max {np.nanmax(rad):.3f}); "
                  f"{int(stab['is_stable'].sum())}/{stab['is_stable'].size} stable, "
                  f"marginal-dir median {int(np.median(stab['n_marginal']))}")

    out_pkl = save_dir / f"fixed_points_grad_{aname}{out_suffix}.pkl"
    with open(out_pkl, "wb") as _f:
        pickle.dump({"aname": aname, "rule": rule, "n_interp": int(n_interp),
                     "rel_tol": float(rel_tol),
                     "n_seeds": int(n_seeds), "selected_seed": int(best_seed),
                     "selection_score": float(best_score),
                     "angles": np.asarray(angles, dtype=float),
                     "results": results}, _f)
    print(f"  Saved gradient fixed-point data: {out_pkl}")
    return out_pkl
