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

The battery above is the DIAGONAL of a more general design: the constant input a
fixed point is solved under and the state the optimizer starts from are two
independent choices (see the probe list below). That separation is what makes
multistability testable. In the delaygo family the fixation and delay inputs are
literally the same vector — the fixation bit and the rule cue are on in both and
the stimulus channels are zero in both — so the two periods pose the SAME
fixed-point problem, and the only reason the delay solve returns a ring while the
fixation solve returns a single point is the seed (all `n_interp` fixation seeds
are the same matrix, because nothing stimulus-specific has entered M yet).

Two things follow, and the solver reports both rather than assuming them:
  * the four periods can probe fewer than four maps, so the solver measures the
    pairwise period-input distances and groups periods that share an input
    (three groups for delaygo: fixation=delay, stimulus, response);
  * each probe's `across_angle_spread` says whether it found ONE fixed point
    (≈0, which the fixation probe must return by construction) or a manifold.

The off-diagonal probes then vary only the seed, one per distinct input, since a
diagonal probe only ever re-finds the one solution its own trajectory reached:
  * a memory-carrying state (the end of the delay);
  * the recorded M(t), jittered, half of it pulled toward the mean M — the only
    probe that can reach a fixed point no trajectory passes through, e.g. the
    interior of a ring;
  * random rank-one states carrying no stimulus information at all.

The rank-one battery came first and rests on M* = [eta/(1-lam)] h* x', which is
exact only while the modulation BOUNDS are slack. They are not: on a solved
delaygo run 3-9% of the entries sit at the bound and the top singular value
carries 78-83% of the spectrum rather than ~100%, so the true solutions are
rank-one on their interior support and clamped elsewhere, and the rank-one family
does not contain them. That is why the trajectory-seeded probe exists, and why it
is the one to trust when the two disagree.

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
# The first epoch is "Context", not "Fixation": it is where the rule cue sets up
# which computation the trial demands. Only the DISPLAY name changes — the period
# KEYS ("longfixation", "fix1") are unchanged, since they index saved pickles.
# paper_plot._period_display normalizes titles from pickles solved before this.
_PERIOD_TITLE = {
    "longfixation": "Context",
    "longstimulus": "Stimulus",
    "longdelay": "Delay",
    "longresponse": "Response",
}

# ─── Probes: (input period) × (seed source) ──────────────────────────────────
# A probe is the 4-tuple (name, input_period, seed_source, title). `input_period`
# picks the constant input the fixed points are solved under (that period's
# midpoint); `seed_source` picks where the optimizer starts. A seed source is
# either a period name (that period's LAST recorded M) or the special token below.
_NAIVE_RANK1 = "naive_rank1"     # random rank-one M, no stimulus information
_TRAJ_SEED = "traj_noise"        # recorded M(t) + jitter (Sussillo & Barak)
# Both synthesize their seeds instead of reading one recorded step, so for both
# `stim` is only a seed index and the points must be coloured by where they
# LANDED (ring_angle_idx), not by a stimulus they never carried.
_SYNTH_SEEDS = (_NAIVE_RANK1, _TRAJ_SEED)

# Off-diagonal probes:
#   *_memseed    the FIXATION input seeded from the end of the DELAY — a state
#                that already holds a stimulus-specific memory. If the ring
#                survives under the fixation input, the ring is a property of the
#                STATE (M), not of the delay input, and the two coexist: a
#                multistable network. NB when the fixation and delay inputs are
#                the same vector (the delaygo family) this is by construction the
#                same solve as the diagonal delay probe; it is kept because it
#                puts the point and the ring side by side under ONE named input,
#                and because solving it verifies the input identity instead of
#                assuming it.
#   *_naiveseed  one per DISTINCT period input, seeded from random rank-one
#                matrices (see _naive_rank1_seeds). These carry no stimulus
#                information, so whatever they find was discovered rather than
#                transplanted — and unlike a diagonal probe, which re-finds the
#                one solution the trial happened to visit, they sample the input's
#                whole fixed-point SET. Each point is annotated with its distance
#                to every same-input diagonal reference (ring_dist /
#                ring_angle_idx / ref_dist), so "did naive seeds land on the
#                structure the task uses?" is a number, not an eyeball judgement.
_MEMSEED_PROBE = ("longfixation_memseed", "longfixation", "longdelay",
                  "Fixation (memory seed)")


def _diagonal_probes(period_win):
    """The historical battery: every period solved under its own input, starting
    from its own end-of-period state."""
    return [(v, v, v, _PERIOD_TITLE.get(v, v)) for v in period_win]


def _naive_probe(period):
    """The naive-seed probe for one period's constant input."""
    return (f"{period}_naiveseed", period, _NAIVE_RANK1,
            f"{_PERIOD_TITLE.get(period, period)} (naive seeds)")


def _traj_probe(period):
    """The trajectory-seed probe for one period's constant input."""
    return (f"{period}_trajseed", period, _TRAJ_SEED,
            f"{_PERIOD_TITLE.get(period, period)} (trajectory seeds)")


def _same_input_groups(input_info, tol=0.0):
    """Group period names by IDENTICAL constant input, e.g.
    [["longfixation", "longdelay"], ["longstimulus"], ["longresponse"]].

    Periods in one group pose the same fixed-point problem, so their fixed-point
    sets are the same set and any probe differing only by which of them is named
    is a duplicate solve. Grouping is measured per run (input_info["dist"]), not
    assumed: with extra epochs, `fixate_off`, or another task family the fixation
    ≡ delay coincidence can fail, and then both deserve their own probe."""
    names, dist = input_info["periods"], input_info["dist"]
    groups = []
    for i, v in enumerate(names):
        for g in groups:
            if dist[i, names.index(g[0])] <= tol:
                g.append(v)
                break
        else:
            groups.append([v])
    return groups


def _extra_probes(present, input_info, cross_seed_probes=True,
                  naive_seed_probes=True, traj_seed_probes=True, tag=""):
    """The off-diagonal battery for the periods actually solved (`present`).

    One synthesized-seed probe per DISTINCT input rather than per period: with
    fixation ≡ delay the delay probes would repeat the fixation ones, and the
    skip is decided from the measured distances so it self-corrects on other
    tasks."""
    extra = []
    if cross_seed_probes and {"longfixation", "longdelay"} <= set(present):
        extra.append(_MEMSEED_PROBE)
    if naive_seed_probes or traj_seed_probes:
        for g in _same_input_groups(input_info):
            g_here = [v for v in g if v in present]
            if not g_here:
                continue
            if traj_seed_probes:
                extra.append(_traj_probe(g_here[0]))
            if naive_seed_probes:
                extra.append(_naive_probe(g_here[0]))
            if len(g_here) > 1 and tag:
                print(f"  {tag} synthesized-seed probes for {g_here[1:]} skipped: "
                      f"same input as {g_here[0]}, so they would be the identical "
                      f"solve.")
    return extra


def _input_pair_distance(input_info, a, b):
    """max|x_a − x_b| between two periods' constant inputs, or NaN if either is
    absent from this trial. 0 means the two periods pose the same map."""
    names = input_info["periods"]
    if a not in names or b not in names:
        return float("nan")
    return float(input_info["dist"][names.index(a), names.index(b)])


def _relative_spread(fixed_M):
    """max_i ||M*_i − mean|| / ||mean|| over a probe's solved points.

    The single number that separates "one fixed point" from "a manifold of them":
    0 means every point is the SAME matrix (which is what the fixation probe must
    return, since all its seeds are identical), while a large value means the
    points spread out along a ring."""
    A = np.asarray(fixed_M, dtype=np.float64)
    A = A.reshape(A.shape[0], -1)
    mu = A.mean(axis=0)
    return float(np.linalg.norm(A - mu[None, :], axis=1).max()
                 / max(np.linalg.norm(mu), 1e-12))


def _naive_rank1_seeds(n, x_embed, mp, seed=0):
    """`n` random rank-one modulation matrices M0 = [η/(1−λ)] ⊙ (v xᵀ).

    Every fixed point of the modulation dynamics is exactly rank one with the
    CURRENT INPUT as its presynaptic factor:

        M* = λM* + η h* xᵀ   ⇒   M* = [η/(1−λ)] h* xᵀ,

    so {v xᵀ} is the ambient family that contains every solution. Drawing the
    postsynaptic factor v uniformly on [-1, 1] (the range of the tanh hidden
    units) seeds the optimizer ON that family but at a random place on it, using
    no stimulus information whatsoever — the point of the naive probe.

    A seed that violates the layer's modulation bounds is infeasible (the
    dynamics could never occupy it), so each seed is RESCALED — not clipped — by
    the largest factor ≤ 1 that fits inside the bounds. Rescaling preserves the
    exact rank-one form and, more importantly, the SIGN pattern of v, which is
    what selects the latched branch: a unit whose self-gain exceeds 1 has an
    unstable quiet state, so any nonzero seed component grows to the latched
    value and the seed amplitude barely matters. Clipping, by contrast, would
    saturate rows independently and destroy the rank-one structure.

    n       : number of seeds (kept equal to the dense angle count so every saved
              array in the probe entry has the same leading dimension).
    x_embed : (n, pre) embedded MP-layer input rows the solve is run under.
    mp      : the multi-plastic layer — supplies η, λ and the modulation bounds.
    seed    : RNG seed, so the naive battery is reproducible.

    Returns (M0, fit_factors): the (n, post, pre) seeds and the per-seed rescale
    factor actually applied (1.0 = the raw seed already fit).
    """
    eta = mp.build_M_parameter(mp.eta, mp.eta_type).detach().cpu().numpy()
    lam = mp.build_M_parameter(mp.lam, mp.lam_type).detach().cpu().numpy()
    # Broadcastable to (post, pre) for every eta/lam type (scalar, pre/post
    # vector, full matrix); atleast_2d covers the scalar case's (1,) shape.
    scale = np.atleast_2d(eta / np.maximum(1.0 - lam, 1e-6))
    rng = np.random.RandomState(int(seed))
    v = rng.uniform(-1.0, 1.0, size=(int(n), int(mp.n_output)))          # (n, post)
    M0 = scale[None, :, :] * (v[:, :, None] * np.asarray(x_embed)[:, None, :])

    fit = np.ones(M0.shape[0])
    if getattr(mp, "modulation_bounds", False):
        hi = mp.M_bounds[0].detach().cpu().numpy()      # upper bounds (post, pre)
        lo = mp.M_bounds[1].detach().cpu().numpy()      # lower bounds (post, pre)
        # Per entry, how much of the way to its bound the seed may travel; the
        # per-seed factor is the tightest of them (zeros impose no limit).
        with np.errstate(divide="ignore", invalid="ignore"):
            room = np.where(M0 > 0, hi / M0, np.where(M0 < 0, lo / M0, np.inf))
        room = np.where(np.isfinite(room), room, np.inf)
        fit = np.minimum(1.0, room.reshape(M0.shape[0], -1).min(axis=1))
        M0 = M0 * fit[:, None, None]
    return M0.astype(np.float32), fit


def _trajectory_M_seeds(n, M_all, mp, noise_frac=0.25, seed=0):
    """`n` seeds drawn from the modulation matrices the network ACTUALLY visits.

    The canonical Sussillo & Barak (2013) sampler, moved to M. It exists because
    `_naive_rank1_seeds`'s justification does not survive contact with this
    network on two counts:

      * The rank-one family is NOT the family of solutions once the modulation
        BOUNDS bite. The true equation is M* = clamp(lam*M* + eta*h*x'), so
        interior entries obey the rank-one relation while saturated ones sit at
        +/- the bound, and the sum is not rank one. Measured on a solved
        delaygo run: 3-9% of entries at the bound and the top singular value
        carries only 78-83% of the spectrum, not ~100%.
      * Even where the family is right, drawing v uniformly in R^n_hidden samples
        it about as thinly as any uniform draw in a few hundred dimensions does.

    These seeds start from states the dynamics genuinely occupy instead, in two
    halves:
      * RECORDED M(t) drawn uniformly over (stimulus, time), plus Gaussian
        jitter. The jitter is what lets the optimizer leave the trajectory and
        settle on the saddles and repellers between attractors.
      * the same matrices pulled a RANDOM fraction of the way toward the MEAN
        recorded M, which is the only thing in the battery that ever seeds the
        INTERIOR of a ring — no trajectory passes through it.

    Seeds are CLIPPED to the layer's modulation bounds, not rescaled the way the
    rank-one seeds are. Rescaling exists there to preserve a global rank-one sign
    pattern; these seeds are not rank one to begin with, and the solutions they
    aim at have their largest entries pinned AT the bound, so shrinking the whole
    matrix would start them systematically small.

    noise_frac : jitter sigma as a fraction of the per-entry std across all
                 recorded matrices, so the scale follows the network.

    Returns (M0, how) with `how` describing the draw, for the caller to print.
    """
    rng = np.random.RandomState(int(seed))
    M = np.asarray(M_all, dtype=np.float64)
    flat = M.reshape(-1, *M.shape[-2:])                     # (stim*T, post, pre)
    base = flat[rng.randint(0, flat.shape[0], size=int(n))]
    n_pull = int(n) // 2
    if n_pull:
        mu = flat.mean(axis=0)
        frac = rng.uniform(0.0, 1.0, size=(n_pull, 1, 1))
        base[:n_pull] = mu[None] + frac * (base[:n_pull] - mu[None])
    base = base + noise_frac * flat.std(axis=0)[None] * rng.randn(*base.shape)

    clipped = 0.0
    if getattr(mp, "modulation_bounds", False):
        hi = mp.M_bounds[0].detach().cpu().numpy()
        lo = mp.M_bounds[1].detach().cpu().numpy()
        out = (base > hi[None]) | (base < lo[None])
        clipped = float(out.mean())
        base = np.clip(base, lo[None], hi[None])
    return (base.astype(np.float32),
            f"{int(n) - n_pull} recorded + {n_pull} pulled toward the mean, "
            f"jitter {noise_frac:g}x per-entry std, {100 * clipped:.1f}% of "
            f"entries clipped to the modulation bounds")


def _decay_factor(mp):
    """(lam, 1 - lam) for the modulation update M <- lam*M + eta*h*x', as
    representative scalars (the median over entries when lam is a vector or a
    full matrix).

    Both the residual and the Jacobian spectrum are compressed by (1 - lam):

        F(M) - M = -(1 - lam)M + eta*h*x' = (1 - lam)[M_target(M) - M]
        J = lam*I + (1 - lam) * d M_target / d M   =>   lam_J = lam + (1 - lam)mu

    With the default m_time_scale = 400 ms and dt = 40 ms, lam = 0.9, so the
    factor is 0.1: a residual reads TEN times more converged than the underlying
    map warrants, and the marginal band |lam_J - 1| <= tol is really
    |mu - 1| <= 10*tol. The RNN counterpart has 1 - alpha = 0.2, a different
    factor, so rel_step is NOT comparable between the two models. Nothing here
    changes a threshold; the normalized twins are recorded beside the raw ones so
    the looseness is visible instead of implicit."""
    lam = mp.build_M_parameter(mp.lam, mp.lam_type).detach().cpu().numpy()
    lam_med = float(np.median(np.atleast_1d(lam)))
    return lam_med, max(1.0 - lam_med, 1e-12)


def _annotate_ring_distance(results, probe_name, ref_names):
    """Record where a naive-seeded probe's points LANDED, relative to the fixed
    points the trial itself visits under the same input.

    `ref_names` are the diagonal probes solved under an identical input (the
    probe's own period plus any period grouped with it by _same_input_groups).
    For the fixation input that is normally BOTH the single fixation point and the
    delay ring, and the two references answer different questions — "did the seeds
    fall back to the trivial state?" versus "did they find the memory ring?" — so
    all of them are recorded:

      ref_dist[ref]     (n,) min over that reference's points of
                        ||M* − M*_ref|| / ||M*_ref||
      ref_nearest[ref]  (n,) the argmin's stimulus index in that reference
      ref_spread[ref]   scalar relative spread of the reference itself
                        (≈0 = the reference is a single point, large = a ring)

    The most ring-like reference (largest spread) additionally fills the plain
    ring_dist / ring_angle_idx / ring_ref fields, so a reader who wants one
    number gets the meaningful one. No-op if no reference is present. Mutates
    `results` in place."""
    e = results.get(probe_name)
    if e is None:
        return
    A = np.asarray(e["fixed_M"], dtype=np.float64)
    A = A.reshape(A.shape[0], -1)                             # (n, D)
    a_sq = (A ** 2).sum(axis=1)

    e["ref_dist"], e["ref_nearest"], e["ref_spread"] = {}, {}, {}
    for ref in ref_names:
        ref_e = results.get(ref)
        if ref_e is None:
            continue
        R = np.asarray(ref_e["fixed_M"], dtype=np.float64)
        R = R.reshape(R.shape[0], -1)                         # (m, D)
        r_sq = (R ** 2).sum(axis=1)
        r_norm = np.maximum(np.sqrt(r_sq), 1e-12)
        # ||a − r||² = |a|² + |r|² − 2a·r, then normalize each column by |r|.
        d2 = a_sq[:, None] + r_sq[None, :] - 2.0 * (A @ R.T)
        rel = np.sqrt(np.maximum(d2, 0.0)) / r_norm[None, :]
        j = np.argmin(rel, axis=1)
        e["ref_dist"][ref] = rel[np.arange(rel.shape[0]), j]
        e["ref_nearest"][ref] = np.asarray(ref_e["stim"], dtype=int)[j]
        e["ref_spread"][ref] = _relative_spread(ref_e["fixed_M"])

    if not e["ref_dist"]:
        return
    ringiest = max(e["ref_spread"], key=lambda r: e["ref_spread"][r])
    e["ring_ref"] = ringiest
    e["ring_dist"] = e["ref_dist"][ringiest]
    e["ring_angle_idx"] = e["ref_nearest"][ringiest]


def derive_fixed_point_views(net, fixed_M, const_input, final_speeds, W, device,
                             rel_tol=0.05):
    """Derive the standard views/metrics of solved modulation fixed points M*.

    Given the solver output `fixed_M` (B, post, pre), the constant input it was
    solved under (B, n_input), and the per-point speeds q(M*), returns a dict:
      fixed_WM      : effective modulation W⊙M* (or None if W is None)
      fixed_hidden  : hidden state produced by M* under const_input (B, hidden)
      fixed_out_cos : cos-output readout at M* (B,) — channel 1 (~0 except response)
      rel_step      : scale-free ||F(M*)-M*|| / ||M*|| (B,); speeds q = ½||F-M||²
      is_fixed      : rel_step <= rel_tol (B,)
    Shared by the per-period dense-angle solver and the task-interpolation sweep
    so both compute these identically. `net`'s stored modulation is restored
    after the forward pass, so this is side-effect free."""
    fixed_M = np.asarray(fixed_M)
    fixed_WM = (fixed_M * np.asarray(W)[None, :, :]) if W is not None else None

    mp = net.mp_layers[0]
    saved_M, saved_M_pre = mp.M, getattr(mp, "M_pre", None)
    with torch.no_grad():
        mp.M = torch.as_tensor(fixed_M, dtype=torch.float, device=device)
        output, mpl_activities, _ = net.forward(
            torch.as_tensor(np.asarray(const_input), dtype=torch.float, device=device),
            run_mode="minimal")
        fixed_hidden = np.asarray(mpl_activities[-1].detach().cpu())
        out = np.asarray(output.detach().cpu())
        fixed_out_cos = out[:, 1] if out.shape[-1] > 1 else out[:, 0]
    mp.M = saved_M
    if saved_M_pre is not None:
        mp.M_pre = saved_M_pre

    # rel_step = ||F(M*)-M*|| / ||M*||; final_speeds is q = ½||F-M||², so
    # ||F-M|| = sqrt(2 q).
    fm = fixed_M.reshape(fixed_M.shape[0], -1).astype(float)
    step_norm = np.sqrt(2.0 * np.asarray(final_speeds, dtype=float))
    m_norm = np.maximum(np.linalg.norm(fm, axis=1), 1e-12)
    rel_step = step_norm / m_norm
    return {
        "fixed_WM": fixed_WM,
        "fixed_hidden": fixed_hidden,
        "fixed_out_cos": fixed_out_cos,
        "rel_step": rel_step,
        "is_fixed": rel_step <= rel_tol,
    }


def solve_period_modulation_fixed_points(
        aname, save_dir, net, cfg, device,
        rule=None, out_suffix="",
        layer_index=1, W=None,
        n_interp=64, stim_magnitudes=None,
        steps=200000, learningRate=1e-3,
        loss_tol=1e-8, lbfgs_steps=2000, rel_tol=0.05,
        stim_channels=None, periods=None, save_all_trajectories=False,
        n_seeds=5, seed_base=0,
        analyze_stability=True, n_eigs=16,
        cross_seed_probes=True, naive_seed_probes=True,
        traj_seed_probes=True, traj_noise_frac=0.25, naive_rng_seed=0):
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
    stim_magnitudes : optional one-dimensional sequence of positive stimulus
                  magnitudes. Every angle is crossed with every magnitude, and
                  the optimizer is initialized from the corresponding recorded
                  end-of-period M. None keeps the historical unit-magnitude
                  sweep. The saved ``stim`` label remains the angle index, while
                  ``stimulus_magnitude`` identifies repeated points at that angle.
    steps / learningRate / loss_tol / lbfgs_steps : optimizer settings passed to
                  find_modulation_fixed_points (Adam until loss<=loss_tol capped at
                  `steps`, then L-BFGS polishing).
    rel_tol     : a point counts as a fixed point when rel_step <= rel_tol.
    periods     : restrict the solve to these period keys (a subset of
                  longfixation / longstimulus / longdelay / longresponse), in
                  canonical order regardless of the order given. None (default)
                  solves all four — the one-task / two-task behavior. Narrowing it
                  cuts the cost proportionally and narrows the probe battery with
                  it: the memory-seed probe needs both fixation and delay, and the
                  naive probes are built per distinct input among the periods
                  actually solved. "longdelay" is the FIRST delay epoch (`delay1`)
                  for every task family, including the two-delay ones (dmc,
                  delaydm), whose later epochs this solver does not probe.
    save_all_trajectories : also save the recorded within-period path of EVERY
                  stimulus (`traj_all_hidden`, `traj_all_WM`), not just the angle-0
                  exemplar. Off by default because the W⊙M one is
                  (n_interp, win_T, hidden*embed): ~55 MB at n_interp=8 but ~440 MB
                  at the 64 the one/two-task callers use. Turn it on when the
                  figures draw a path per stimulus (the multi-task sibling
                  families do).
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
    cross_seed_probes : add the "longfixation_memseed" probe — the FIXATION input
                  solved from the end-of-DELAY state. Same input as the plain
                  fixation probe, different basin, so a ring here means the
                  network is multistable and the ring lives in M rather than in
                  the delay input.
    naive_seed_probes : add one "{period}_naiveseed" probe per DISTINCT period
                  input — solved from random rank-one seeds that carry no stimulus
                  information (_naive_rank1_seeds), with each solved point's
                  distance to the same-input diagonal probes recorded. This is
                  both the control for "the ring was transplanted with the seed"
                  and the only probe that samples an input's fixed-point SET
                  rather than re-finding the one solution the trial visits.
                  Periods sharing an input get a single probe (see
                  _same_input_groups).
    traj_seed_probes : add one "{period}_trajseed" probe per DISTINCT period
                  input — solved from the modulation matrices the network actually
                  visits, jittered, half of them pulled toward the mean recorded M
                  (_trajectory_M_seeds). This is the probe that reaches fixed
                  points no trajectory passes through, and unlike the naive
                  battery it does not depend on the rank-one family being the
                  solution set — which the modulation BOUNDS break (measured:
                  3-9% of entries at the bound, top singular value carrying only
                  78-83% of the spectrum).
    traj_noise_frac : jitter for those seeds, as a fraction of the per-entry std.
    naive_rng_seed : RNG seed for the synthesized batteries (reproducibility).
                  All the off-diagonal probes are solved ONCE, on the selected
                  seed only: the selection score looks at the stimulus and
                  response periods, so solving them per candidate seed would cost
                  n_seeds× for nothing.

    Writes fixed_points_grad_{aname}{out_suffix}.pkl and returns its path (or None
    if no period could be solved).
    """
    # Decay factor, for the normalized twins of the convergence and stability
    # numbers. Stated once, up front, because every rel_step and every |lam_J - 1|
    # below is damped by it.
    decay_lam, decay_leak = _decay_factor(net.mp_layers[0])
    if stim_magnitudes is None:
        stimulus_magnitude_levels = np.array([1.0], dtype=float)
    else:
        stimulus_magnitude_levels = np.asarray(stim_magnitudes, dtype=float)
        if stimulus_magnitude_levels.ndim != 1 or stimulus_magnitude_levels.size == 0:
            raise ValueError("stim_magnitudes must be a non-empty 1D sequence")
        if (not np.isfinite(stimulus_magnitude_levels).all()
                or np.any(stimulus_magnitude_levels <= 0)):
            raise ValueError("stim_magnitudes must contain finite positive values")
    tag_pre = f"[grad-fp/{rule}]" if rule else "[grad-fp]"
    print(f"  {tag_pre} decay: lam={decay_lam:.3f}, 1-lam={decay_leak:.3f} — both "
          f"the residual and the eigenvalue band are compressed by that factor, "
          f"so rel_tol={rel_tol:g} is {rel_tol / decay_leak:.3g} on the undamped "
          f"map and every threshold below reads {1.0 / decay_leak:.1f}x tighter "
          f"than it is. Undamped twins are printed and saved beside the damped "
          f"numbers.")

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

    def _solve_one_seed(task_seed, probes=None):
        """Build the dense-angle template for a FIXED task RNG seed and solve the
        requested probes. `probes` is a list of (name, input_period, seed_source,
        title) tuples; None means the diagonal battery (each period from its own
        end state). Returns (results_dict, angles, input_info) where input_info
        holds the pairwise period-input distances (see below).
        Deterministic in task_seed, so re-running is reproducible."""
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
            "longdelay":    (delay_on, delay_off),   # = delay1, the FIRST delay
            "longresponse": (resp_on, resp_off),
        }
        if periods is not None:
            unknown = [v for v in periods if v not in period_win]
            if unknown:
                raise ValueError(f"unknown period(s) {unknown}; choose from "
                                 f"{list(period_win)}")
            # Filter by iterating period_win, so the canonical
            # fixation→stimulus→delay→response order survives whatever order the
            # caller listed. Everything downstream — the diagonal probes, the
            # measured input distances, the naive-probe grouping — is derived from
            # this dict, so restricting it here restricts the whole solve.
            keep = set(periods)
            period_win = {v: w for v, w in period_win.items() if v in keep}

        if stim_channels is not None:
            ch_a, ch_b = int(stim_channels[0]), int(stim_channels[1])
        else:
            ch_a, ch_b = _detect_stim_channels(template, fix_on, fix_off,
                                               stim_on, stim_off)
        print(f"  {tag} seed={task_seed}: stimulus ring channels ({ch_a}, {ch_b})")

        # The ring's two channels hold magnitude * (sin θ, cos θ) in
        # generator order. Keep `stim` as the ANGLE label when magnitudes are
        # repeated, so downstream hue continues to mean stimulus direction.
        angles = np.arange(n_interp) * (2 * np.pi / n_interp)
        stim = np.repeat(np.arange(n_interp, dtype=int),
                         stimulus_magnitude_levels.size)
        point_angles = angles[stim]
        point_magnitudes = np.tile(stimulus_magnitude_levels, n_interp)
        n_points = int(stim.size)
        batch = np.repeat(template[None, :, :], n_points, axis=0)
        batch[:, stim_on:stim_off, ch_a] = (
            point_magnitudes * np.sin(point_angles))[:, None]
        batch[:, stim_on:stim_off, ch_b] = (
            point_magnitudes * np.cos(point_angles))[:, None]

        x = torch.as_tensor(batch, dtype=torch.float, device=device)
        _, _, db = net.iterate_sequence_batch(
            x, run_mode="track_states", save_to_cpu=True, detach_saved=True)
        M_all = np.asarray(db[f"M{layer_index}"])                  # (n_points,T,hid,emb)
        hid_all = np.asarray(db[f"hidden{layer_index}"])           # (n_points,T,hidden)
        # Embedded MP-layer input x(t) (post input-embedding). Needed to build the
        # naive rank-one seeds, whose presynaptic factor must be the x the solve
        # runs under — every fixed point has that form (see _naive_rank1_seeds).
        # Only that probe needs it, so a net layout without the key stays usable.
        _x_key = f"input{layer_index}"
        x_embed_all = np.asarray(db[_x_key]) if _x_key in db else None
        # Exemplar stimulus whose within-period trajectory we record for the
        # figure (first magnitude at angle 0). `traj_stim` is the angle label,
        # retained for compatibility with paper_plot's exemplar lookup.
        traj_stim = 0

        # ── How many DISTINCT constant inputs does the period battery have? ───
        # Pairwise max|x_a - x_b| between the period midpoints. Two periods at
        # distance 0 pose the SAME fixed-point problem — their fixed-point sets
        # are one set — so the four periods can probe fewer than four maps. For
        # the delaygo family fixation and delay coincide (fixation bit and rule
        # cue on in both, stimulus channels zero in both), giving three maps for
        # four periods. Measured per run so a task where that fails (extra
        # epochs, fixate_off, another family) is caught instead of assumed, and
        # so the naive battery can skip the duplicate input by itself.
        in_names = [v for v, (a, b) in period_win.items() if 0 <= a < b <= T]
        in_t = {v: min((period_win[v][0] + period_win[v][1]) // 2, T - 1)
                for v in in_names}
        in_dist = np.zeros((len(in_names), len(in_names)))
        for i, a in enumerate(in_names):
            for j, b in enumerate(in_names):
                in_dist[i, j] = np.abs(batch[:, in_t[a], :]
                                       - batch[:, in_t[b], :]).max()
        input_info = {"periods": in_names, "t_mid": in_t, "dist": in_dist}
        groups = _same_input_groups(input_info)
        print(f"  {tag} seed={task_seed}: {len(groups)} distinct period input(s) "
              f"among {len(in_names)}: "
              + " | ".join("=".join(g) for g in groups))
        for i, a in enumerate(in_names):
            print(f"      max|x_{a} - x_*| = "
                  + "  ".join(f"{b}:{in_dist[i, j]:.2e}"
                              for j, b in enumerate(in_names) if j != i))

        if probes is None:
            probes = _diagonal_probes(period_win)

        results = {}
        for name, in_period, seed_src, title in probes:
            ps, pe = period_win.get(in_period, (0, 0))
            if not (0 <= ps < pe <= T):
                continue
            t_mid = min((ps + pe) // 2, T - 1)
            const_input = batch[:, t_mid, :]
            # Seed: a period's end state, or synthesized seeds carrying no
            # stimulus information (t_seed = -1 marks "not from a recorded step").
            if seed_src == _TRAJ_SEED:
                t_seed = -1
                init_M, how = _trajectory_M_seeds(
                    n_points, M_all, net.mp_layers[0],
                    noise_frac=traj_noise_frac, seed=naive_rng_seed)
                print(f"  {tag} {name}: seeds drawn {how}")
            elif seed_src == _NAIVE_RANK1:
                if x_embed_all is None:
                    print(f"  {tag} {name}: db has no '{_x_key}' (embedded MP-layer "
                          f"input), so naive rank-one seeds cannot be built; "
                          f"skipping the probe.")
                    continue
                t_seed = -1
                init_M, fit = _naive_rank1_seeds(
                    n_points, x_embed_all[:, t_mid, :], net.mp_layers[0],
                    seed=naive_rng_seed)
                print(f"  {tag} {name}: naive rank-one seeds, bound-fit factor "
                      f"median {np.median(fit):.3f} (1.0 = raw seed already fit)")
            else:
                qs, qe = period_win.get(seed_src, (0, 0))
                if not (0 <= qs < qe <= T):
                    print(f"  {tag} {name}: seed period '{seed_src}' absent from "
                          f"this trial; skipping the probe.")
                    continue
                t_seed = min(qe - 1, T - 1)
                init_M = M_all[:, t_seed, :, :]

            seed_desc = (f"{seed_src} t={t_seed}" if t_seed >= 0
                         else f"{seed_src} (rng {naive_rng_seed})")
            print(f"  {tag} seed={task_seed} {name}: solving {n_points} fixed "
                  f"points ({n_interp} angles x "
                  f"{stimulus_magnitude_levels.size} magnitudes; input "
                  f"{in_period} t={t_mid}, seed {seed_desc})")
            fixed_M, loss_hist, final_speeds = find_modulation_fixed_points(
                net, init_M, const_input, steps=steps, learningRate=learningRate,
                printPeriod=max(steps // 20, 1), loss_tol=loss_tol,
                lbfgs_steps=lbfgs_steps, device=device)

            fixed_WM = (fixed_M * np.asarray(W)[None, :, :]) if W is not None else None
            fixed_hidden, fixed_out_cos = _hidden_from_M(fixed_M, const_input)

            # Recorded within-period trajectory of the EXEMPLAR stimulus (angle 0):
            # how the state actually moves over this period, en route to the fixed
            # point. M(t) under the true (time-varying) period input, so it settles
            # NEAR — not exactly onto — M* (solved under sustained input).
            # DIAGONAL probes only: for an off-diagonal probe the recorded path
            # never visited these fixed points, so a connector would be fiction.
            # paper_plot draws no connector when traj_* is None.
            diagonal = (seed_src == in_period)
            traj_M_flat = traj_hidden = traj_WM = None
            traj_all_hidden = traj_all_WM = None
            if diagonal:
                traj_M = M_all[traj_stim, ps:pe, :, :]             # (win_T, hid, emb)
                traj_M_flat = traj_M.reshape(traj_M.shape[0], -1)  # (win_T, hid*emb)
                traj_WM = (traj_M * np.asarray(W)[None, :, :]).reshape(
                    traj_M.shape[0], -1) if W is not None else None
                traj_hidden = hid_all[traj_stim, ps:pe, :]         # (win_T, hidden)
                if save_all_trajectories:
                    # Same window, every stimulus — row i pairs with fixed point i,
                    # so a caller can project paths and endpoints through one basis.
                    traj_all_hidden = hid_all[:, ps:pe, :]         # (B, win_T, hidden)
                    if W is not None:
                        M_win = M_all[:, ps:pe, :, :]
                        traj_all_WM = (M_win * np.asarray(W)[None, None, :, :]
                                       ).reshape(M_win.shape[0], M_win.shape[1], -1)

            # Scale-free convergence metric rel_step = ||F(M*)-M*|| / ||M*||;
            # final_speeds is q = 1/2||F-M||^2, so ||F-M|| = sqrt(2 q).
            fm = np.asarray(fixed_M, dtype=float).reshape(fixed_M.shape[0], -1)
            step_norm = np.sqrt(2.0 * np.asarray(final_speeds, dtype=float))
            m_norm = np.maximum(np.linalg.norm(fm, axis=1), 1e-12)
            rel_step = step_norm / m_norm
            is_fixed = rel_step <= rel_tol
            # Decay-normalized twin (see _decay_factor): the measured residual is
            # damped by (1 - lam), so with lam = 0.9 every point reads ten times
            # more converged than the underlying map warrants. rel_tol is
            # deliberately NOT changed — it keeps one meaning across this
            # codebase and across the saved pickles — but the normalized number
            # is recorded and printed beside it.
            rel_step_undamped = rel_step / decay_leak
            print(f"  {tag} seed={task_seed} {name}: {int(is_fixed.sum())}/{is_fixed.size} "
                  f"converged (rel_step<= {rel_tol:g}); "
                  f"median {np.median(rel_step):.2e} max {rel_step.max():.2e}"
                  f" | undamped median {np.median(rel_step_undamped):.2e} "
                  f"max {rel_step_undamped.max():.2e}")

            results[name] = {
                "period_title": title,
                # Which input the points were solved under, and where the solve
                # started — the two axes that make this a probe rather than just
                # "the fixation period".
                "input_period": in_period,
                "seed_source": seed_src,
                "is_diagonal": bool(diagonal),
                # `stim` is the dense stimulus-angle index for period-seeded
                # probes. Naive seeds carry no stimulus, so there it is only a
                # seed index — colour those points by ring_angle_idx instead.
                "stim_is_stimulus": bool(seed_src not in _SYNTH_SEEDS),
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
                # The same residual relative to the UNDAMPED map, i.e. divided by
                # (1 - lam). See _decay_factor.
                "rel_step_undamped": np.asarray(rel_step_undamped, dtype=float),
                "lam": float(decay_lam),
                "leak": float(decay_leak),
                "is_fixed": np.asarray(is_fixed, dtype=bool),
                "rel_tol": float(rel_tol),
                "loss_hist": np.asarray(loss_hist, dtype=float),
                "stim": np.asarray(stim),
                "stimulus_angle": np.asarray(point_angles, dtype=float),
                "stimulus_magnitude": np.asarray(point_magnitudes, dtype=float),
                # Constant input this period's M* was solved under; kept so the
                # (deferred) stability analysis can linearize F at the same point.
                "const_input": np.asarray(const_input, dtype=np.float32),
                # Exemplar-stimulus within-period trajectory (angle 0) for the
                # figures: raw M, effective W⊙M, and hidden, one row per timestep.
                "traj_stim": int(traj_stim),
                "traj_M": (np.asarray(traj_M_flat, dtype=np.float32)
                           if traj_M_flat is not None else None),
                "traj_WM": (np.asarray(traj_WM, dtype=np.float32)
                            if traj_WM is not None else None),
                "traj_hidden": (np.asarray(traj_hidden, dtype=np.float32)
                                if traj_hidden is not None else None),
                # Per-stimulus paths (save_all_trajectories); row i pairs with
                # fixed point i. None unless the caller asked for them.
                "traj_all_hidden": (np.asarray(traj_all_hidden, dtype=np.float32)
                                    if traj_all_hidden is not None else None),
                "traj_all_WM": (np.asarray(traj_all_WM, dtype=np.float32)
                                if traj_all_WM is not None else None),
                # 0 ⇒ every point is the same matrix (one fixed point); large ⇒
                # the points spread along a manifold. See _relative_spread.
                "across_angle_spread": _relative_spread(fixed_M),
            }
            print(f"  {tag} seed={task_seed} {name}: across-stimulus spread of M* = "
                  f"{results[name]['across_angle_spread']:.3e}")
        return results, angles, input_info

    def _selection_score(results):
        """Lower = better. Median rel_step over the STIMULUS + RESPONSE periods
        only (the parts most sensitive to the random template). Missing periods
        contribute nothing; if neither is present, fall back to all periods."""
        keys = [k for k in ("longstimulus", "longresponse") if k in results]
        if not keys:
            keys = list(results.keys())
        vals = np.concatenate([np.asarray(results[k]["rel_step"], float)
                               for k in keys]) if keys else np.array([np.inf])
        return float(np.median(vals)), keys

    # ── Try n_seeds deterministic templates; keep the best-converging one ────
    best = None   # (score, task_seed, results, angles, input_info, score_periods)
    for s in range(seed_base, seed_base + max(int(n_seeds), 1)):
        try:
            results, angles, input_info = _solve_one_seed(s)
        except Exception as exc:
            print(f"  {tag} seed={s} failed: {exc}")
            continue
        if not results:
            continue
        score, score_periods = _selection_score(results)
        score_label = "+".join(score_periods) if score_periods else "none"
        print(f"  {tag} seed={s}: selection score ({score_label} median "
              f"rel_step) = {score:.3e}")
        if best is None or score < best[0]:
            best = (score, s, results, angles, input_info, score_periods)

    if best is None:
        print(f"  {tag} no seed produced fixed points; skipping save.")
        return None

    best_score, best_seed, results, angles, input_info, score_periods = best
    print(f"  {tag} selected seed={best_seed} (score {best_score:.3e} over "
          f"{n_seeds} seed(s)).")

    # ── Off-diagonal multistability probes (selected seed only) ──────────────
    # Same input as the plain fixation probe, different starting state. Solved
    # here rather than inside the sweep because the selection score only looks at
    # the stimulus and response periods — running these per candidate seed would
    # cost n_seeds× and change nothing. Re-entering _solve_one_seed rebuilds the
    # identical template (deterministic in the seed) for one extra forward pass.
    groups = _same_input_groups(input_info)
    extra = _extra_probes(list(results), input_info,
                          cross_seed_probes=cross_seed_probes,
                          naive_seed_probes=naive_seed_probes,
                          traj_seed_probes=traj_seed_probes, tag=tag)
    if extra:
        print(f"  {tag} solving {len(extra)} multistability probe(s) on the "
              f"selected seed={best_seed}: {[p[0] for p in extra]}")
        try:
            extra_results, _, _ = _solve_one_seed(best_seed, probes=extra)
            results.update(extra_results)
            # Naive seeds carry no stimulus label, so instead of a label record
            # WHERE they landed: distance to every diagonal probe solved under
            # the same input (for the fixation input that is both the single
            # fixation point and the delay ring — different questions, both worth
            # asking).
            for p in extra:
                if p[2] not in _SYNTH_SEEDS or p[0] not in results:
                    continue
                grp = next((g for g in groups if p[1] in g), [p[1]])
                refs = [v for v in grp
                        if v in results and results[v].get("is_diagonal")]
                _annotate_ring_distance(results, p[0], refs)
                e = results[p[0]]
                for ref, rd in e.get("ref_dist", {}).items():
                    rd = np.asarray(rd, dtype=float)
                    kind = ("ring" if e["ref_spread"][ref] > 1e-3 else "point")
                    print(f"  {tag} {p[0]}: landing distance to {ref} "
                          f"({kind}, spread {e['ref_spread'][ref]:.3e}) — median "
                          f"{np.median(rd):.3f}, min {rd.min():.3f}, "
                          f"{int((rd <= 0.1).sum())}/{rd.size} within 10%")
        except Exception as exc:
            print(f"  {tag} multistability probes failed: {exc}")
            import traceback
            traceback.print_exc()

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
            # Decay-normalized spectrum. lam_J = lam + (1 - lam)mu, so
            # |lam_J - 1| = (1 - lam)|mu - 1| and the marginal band is ten times
            # wider than it reads at lam = 0.9. Two consequences worth having as
            # numbers: `n_marginal_leak_only` counts directions that are marginal
            # ONLY because of the decay (|mu - 1| is outside the band), and mu
            # exposes that the M-dynamics' BULK spectrum sits at lam by
            # construction — every direction that does not move h has mu = 0 and
            # lam_J = lam — so "spectral radius < 1" is far weaker evidence here
            # than the same sentence about a vanilla RNN.
            ev = np.asarray(stab["eigenvalues"])
            mtol = float(stab["marginal_tol"])
            mu = (ev - decay_lam) / decay_leak
            leak_only = (np.abs(ev - 1.0) <= mtol) & (np.abs(mu - 1.0) > mtol)
            e.update({
                "eigenvalues_undamped": mu,
                "spectral_radius_undamped": np.abs(mu).max(axis=-1),
                "n_marginal_leak_only": leak_only.sum(axis=-1),
            })
            rad = stab["spectral_radius"]
            print(f"  {tag} {v}: spectral radius median {np.nanmedian(rad):.3f} "
                  f"(max {np.nanmax(rad):.3f}); "
                  f"{int(stab['is_stable'].sum())}/{stab['is_stable'].size} stable, "
                  f"marginal-dir median {int(np.median(stab['n_marginal']))}")
            print(f"  {tag} {v}: undamped spectral radius median "
                  f"{np.nanmedian(e['spectral_radius_undamped']):.3f}; of the "
                  f"marginal directions, a median of "
                  f"{int(np.median(e['n_marginal_leak_only']))} are marginal ONLY "
                  f"because of the decay (|mu-1| > {mtol:g})")

    out_pkl = save_dir / f"fixed_points_grad_{aname}{out_suffix}.pkl"
    with open(out_pkl, "wb") as _f:
        pickle.dump({"aname": aname, "rule": rule, "n_interp": int(n_interp),
                     "stimulus_magnitudes": np.asarray(
                         stimulus_magnitude_levels, dtype=float),
                     "n_stimulus_points": int(
                         n_interp * stimulus_magnitude_levels.size),
                     "rel_tol": float(rel_tol),
                     # Modulation decay, so a reader can undo the damping of
                     # rel_step and of the eigenvalue band (see _decay_factor).
                     "lam": float(decay_lam), "leak": float(decay_leak),
                     "n_seeds": int(n_seeds), "selected_seed": int(best_seed),
                     "selection_score": float(best_score),
                     "selection_periods": list(score_periods),
                     "angles": np.asarray(angles, dtype=float),
                     # Provenance of every entry in `results`: which input it was
                     # solved under and where the solve started.
                     "probes": [(v, e.get("input_period", v),
                                 e.get("seed_source", v), e.get("period_title", v))
                                for v, e in results.items()],
                     # Pairwise max|x_a - x_b| between the period midpoint inputs
                     # of the selected template, and the resulting grouping. A 0
                     # entry means those two periods solve the SAME map, so any
                     # difference between their panels is purely the seed — which
                     # is exactly the fixation-point vs delay-ring case.
                     "input_periods": list(input_info["periods"]),
                     "input_dist": np.asarray(input_info["dist"], dtype=float),
                     "input_groups": groups,
                     "input_diff_fix_delay": _input_pair_distance(
                         input_info, "longfixation", "longdelay"),
                     "naive_rng_seed": int(naive_rng_seed),
                     "results": results}, _f)
    print(f"  Saved gradient fixed-point data: {out_pkl}")
    return out_pkl
