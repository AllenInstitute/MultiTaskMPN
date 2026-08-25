#!/usr/bin/env python
# coding: utf-8
"""
Hidden-state fixed-point analysis of a single-task vanilla RNN.

The RNN counterpart of one_task_analysis.py's fixed-point section — and ONLY
that section. one_task_analysis.py runs a dozen analyses because the MPN has a
fast modulation matrix M whose evolution across training is the object of study;
an RNN has no M, so the one thing worth asking of it here is the same question
core/grad_fixed_points.py asks of the MPN, moved to the state the RNN actually
has:

    MPN:  M* = F(M*; x)   solved over the modulation matrix
    RNN:  h* = F(h*; x)   solved over the HIDDEN state, with
          F(h; x) = alpha*h + (1-alpha)*act(W_rec h + W_input x + b)

The analysis:
  1. builds a dense grid of `n_interp` stimulus angles by copying ONE real trial
     template and overwriting its stimulus channels with (sin θ, cos θ) — this
     bypasses the task generator's 8-way snapping, so the ring is sampled between
     the trained directions and a continuous attractor can be told from 8
     discrete ones;
  2. runs that batch through the trained RNN once to record h(t);
  3. solves a BATTERY of probes, each an independent choice of (which constant
     input to hold, where to start the optimizer) — see the probe block below;
  4. repeats 1-3 for `n_seeds` deterministic trial TEMPLATES and keeps the
     best-converging one, so the result is not hostage to a single draw of the
     task RNG;
  5. saves h*, the scale-free convergence metric
     rel_step = ||F(h*) − h*|| / ||h*||, an `is_fixed` mask, and the linear
     stability spectrum (a lone marginal eigenvalue with the rest contracting is
     the ring-attractor signature).

Step 3 is what makes this more than one solve per period, and it mirrors the
MPN's battery in core/grad_fixed_points.py. Seeding a period's own end state
under its own input — the diagonal probe — can only ever re-find the one solution
the trial itself visited, since every seed already sits on the trajectory. Fixed
points the network never visits (an interior point, a second branch) are
invisible to it by construction. So the battery also varies the SEED while
holding the input fixed:

    longfixation_memseed   Context input, seeded from the end of the Delay
    {period}_trajseed      one per DISTINCT input, seeded from the states the
                           network actually visits, jittered — plus half of them
                           pulled toward the mean state, which is the only way
                           anything ever seeds the INTERIOR of a ring
    {period}_naiveseed     one per DISTINCT input, seeded from random hidden
                           states carrying no stimulus information

Two different things are called a "seed" here, as in the MPN version. The probe
seed above is a STARTING STATE for the optimizer. The template seed (`n_seeds`,
step 4) is a task-RNG draw picking which trial the whole analysis runs on. The
off-diagonal probes are solved on the selected template only, since the template
choice is scored on the diagonal battery and re-solving them per candidate would
cost n_seeds x and change nothing.

Reads a run trained by one_task_rnn.py (./onetask_rnn/). Outputs, per run, into
./onetask_rnn/{aname}/:
  fixed_points_hidden_{aname}.pkl   — one entry per PROBE: h*, rel_step,
                                      is_fixed, spectrum, and for the
                                      synthesized-seed probes where each point
                                      landed relative to the diagonal ones
  fixed_points_hidden_{aname}.png   — the diagonal probes in a shared
                                      delay-period PCA, colored by stimulus

Usage:
    python one_task/one_task_rnn_analysis.py                  # every run found
    python one_task/one_task_rnn_analysis.py --aname <name>
    python one_task/one_task_rnn_analysis.py --n-interp 8     # trained dirs only
    python one_task/one_task_rnn_analysis.py --diagonal-only  # old, cheaper battery
    python one_task/one_task_rnn_analysis.py --fp-n-seeds 1   # one template only
"""
import copy
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import networks as nets
import mpn_tasks
from fixed_point import (find_hidden_fixed_points,
                         characterize_hidden_fixed_point_stability)

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

RNN_DIR = Path("onetask_rnn")

# Trial periods, and the epoch key each one is read from. Names follow the
# project's period vocabulary (SCHEME.md): the first epoch is Context, not
# "Fixation" — the epoch KEYS ("fix1", …) are the generator's and stay as they
# are. The internal period keys mirror grad_fixed_points.py so a reader can put
# the two analyses side by side.
PERIODS = {
    "longfixation": ("fix1", "Context"),
    "longstimulus": ("stim1", "Stimulus"),
    "longdelay": ("delay1", "Delay"),
    "longresponse": ("go1", "Response"),
}

# ─── Probes: (input period) × (seed source) ──────────────────────────────────
# The same battery core/grad_fixed_points.py runs for the MPN, moved from the
# modulation matrix M to the hidden state h. A probe is the 4-tuple
# (name, input_period, seed_source, title): `input_period` picks the constant
# input the fixed points are solved under (that period's midpoint), and
# `seed_source` picks where the optimizer STARTS — either a period name (that
# period's last recorded h) or the token below.
#
# Separating those two choices is the whole point. A DIAGONAL probe (input and
# seed from the same period) can only re-find the one solution the trial itself
# visited, because every one of its seeds sits on the recorded trajectory; it
# says nothing about the rest of that input's fixed-point set. Holding the input
# fixed and moving the SEED is what turns "here is where the network went" into
# "here is what the network could settle to" — e.g. an interior fixed point that
# no trajectory ever visits is invisible to the diagonal battery by construction.
_NAIVE_SEED = "naive_random"      # random hidden states, no stimulus information
_TRAJ_SEED = "traj_noise"         # recorded states + jitter (Sussillo & Barak)
# Both synthesize their seeds instead of reading one recorded step, so for both
# `stim` is only a seed index and the points must be coloured by where they
# LANDED (ring_angle_idx), not by a stimulus they never had.
_SYNTH_SEEDS = (_NAIVE_SEED, _TRAJ_SEED)

# Off-diagonal probes:
#   *_memseed    the CONTEXT input seeded from the end of the DELAY — a state
#                that already carries a stimulus-specific memory. If the ring
#                survives under the context input, the ring is a property of the
#                STATE, not of the delay input, and the two coexist: a
#                multistable network. When the two inputs are the same vector
#                (the delaygo family) this is by construction the same solve as
#                the diagonal delay probe; it is kept because it puts the single
#                point and the ring side by side under ONE named input, and
#                because solving it VERIFIES that input identity rather than
#                assuming it.
#   *_trajseed   one per DISTINCT period input, seeded from the states the
#                network ACTUALLY visits, jittered (see _trajectory_seeds). This
#                is the canonical Sussillo & Barak sampler and the probe that
#                does the real work: a uniform draw in the tanh box is nearly
#                orthogonal to the 2-3 dimensional manifold the dynamics occupy,
#                so the naive probe below covers very little of what matters.
#                The MPN battery has the same probe for the same reason — its
#                rank-one family looked like a principled alternative until the
#                modulation bounds turned out to break it.
#   *_naiveseed  one per DISTINCT period input, seeded from random hidden states
#                (see _naive_hidden_seeds). These carry no stimulus information,
#                so whatever they find was discovered rather than transplanted,
#                and they sample the input's whole fixed-point SET. Each solved
#                point is annotated with its distance to the same-input diagonal
#                references, so "did the naive seeds land on the structure the
#                task actually uses?" is a number rather than an eyeball call.
_MEMSEED_PROBE = ("longfixation_memseed", "longfixation", "longdelay",
                  "Context (memory seed)")

# Which stimulus is the exemplar whose within-period trajectory is saved. Named
# once so the saved path and the `traj_stim` annotation that tells a figure which
# point it connects to can never disagree.
_TRAJ_STIM = 0


def _diagonal_probes(present):
    """The historical battery: every period solved under its own input, starting
    from its own end-of-period state."""
    return [(v, v, v, PERIODS.get(v, (None, v))[1]) for v in present]


def _naive_probe(period):
    """The naive-seed probe for one period's constant input."""
    return (f"{period}_naiveseed", period, _NAIVE_SEED,
            f"{PERIODS.get(period, (None, period))[1]} (naive seeds)")


def _traj_probe(period):
    """The trajectory-seed probe for one period's constant input."""
    return (f"{period}_trajseed", period, _TRAJ_SEED,
            f"{PERIODS.get(period, (None, period))[1]} (trajectory seeds)")


def _same_input_groups(input_info, tol=0.0):
    """Group period names by IDENTICAL constant input, e.g.
    [["longfixation", "longdelay"], ["longstimulus"], ["longresponse"]].

    Periods in one group pose the same fixed-point problem, so their fixed-point
    sets are one set and any probe differing only by which of them is named is a
    duplicate solve. Grouped from the MEASURED distances (input_info["dist"]) so
    a task where the context ≡ delay coincidence fails is caught, not assumed."""
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
                  naive_seed_probes=True, traj_seed_probes=True):
    """The off-diagonal battery for the periods actually solved (`present`).

    One synthesized-seed probe per DISTINCT input rather than per period: with
    context ≡ delay the delay probes would repeat the context ones, and the skip
    is decided from the measured distances so it self-corrects on other tasks."""
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
            if len(g_here) > 1:
                print(f"  [hidden-fp] synthesized-seed probes for {g_here[1:]} "
                      f"skipped: same input as {g_here[0]}, so they would be the "
                      f"identical solve.")
    return extra


def _naive_hidden_seeds(n, n_hidden, activation="tanh", seed=0, scale_ref=None):
    """`n` random hidden states carrying no stimulus information.

The MPN's counterpart draws rank-one seeds, on the argument that
    M* = [η/(1−λ)] h* xᵀ makes the rank-one family the ambient set containing
    every solution. That argument holds only while the modulation bounds are
    slack, and they are not (see _trajectory_seeds), so it is not the stronger
    footing it appears to be. The hidden state has no such family to appeal to at
    all: h* may sit anywhere the units can reach, so the ambient set here is
    simply the activation's reachable range.

    For tanh that range is [-1, 1] per unit: the leaky update
    h ← αh + (1−α)tanh(·) started from h = 0 never leaves the hull of tanh's
    output, so a uniform draw on [-1, 1] covers exactly the reachable box. It is
    also the very distribution the MPN draws its postsynaptic factor from, and it
    lands at the right SCALE without being told — E‖v‖ = sqrt(n_hidden/3) ≈ 8.2
    for 200 units, against ≈ 6–8.5 for this run's recorded states.

    Any other activation has a different reachable set, so rather than assume one
    the seeds are matched to the RECORDED states' own per-unit scale
    (`scale_ref`, the (B, T, hidden) trace). Falls back to a unit Gaussian only
    when no trace is supplied.

    Returns (init_h, how) with `how` naming the distribution actually used, so
    the caller can print what it did instead of leaving it implicit.
    """
    rng = np.random.RandomState(int(seed))
    act = str(activation).lower()
    if act == "tanh":
        return (rng.uniform(-1.0, 1.0, size=(int(n), int(n_hidden))
                            ).astype(np.float32),
                "uniform[-1,1] (tanh range)")
    if act == "sigmoid":
        return (rng.uniform(0.0, 1.0, size=(int(n), int(n_hidden))
                            ).astype(np.float32),
                "uniform[0,1] (sigmoid range)")
    if scale_ref is not None:
        ref = np.asarray(scale_ref, dtype=np.float64).reshape(-1, int(n_hidden))
        mu, sd = ref.mean(axis=0), ref.std(axis=0)
        return ((mu[None, :] + sd[None, :] * rng.randn(int(n), int(n_hidden))
                 ).astype(np.float32),
                f"gaussian matched to recorded {act} states")
    return (rng.randn(int(n), int(n_hidden)).astype(np.float32),
            "unit gaussian (no reference trace)")


def _trajectory_seeds(n, h_all, noise_frac=0.25, seed=0):
    """`n` seeds drawn from the states the network ACTUALLY visits, jittered.

    The canonical Sussillo & Barak (2013) sampler, and the one thing the naive
    battery cannot do. A uniform draw in the tanh box is — in 200 dimensions —
    very nearly orthogonal to the two or three dimensional manifold the dynamics
    actually occupy, so 64 such points sample the interesting region essentially
    not at all. These seeds start inside that region instead.

    The MPN's rank-one seeds look better founded, since every unclamped
    modulation fixed point is exactly [η/(1−λ)] h* xᵀ. Measured on a solved
    delaygo run they are not: 3-9% of a fixed point's entries sit at the
    modulation bound and its top singular value carries 78-83% of the spectrum
    rather than ~100%, so the true solutions are rank-one on their interior
    support and clamped elsewhere, and the rank-one family does not contain
    them. Both models therefore get this probe.

    Two halves:
      * RECORDED states drawn uniformly over (stimulus, time), plus Gaussian
        jitter. The jitter is what lets the optimizer fall off the trajectory
        onto the saddles and repellers that sit between the attractors — without
        it these would only re-find the diagonal probes' answers.
      * the same states pulled a RANDOM fraction of the way toward the MEAN
        recorded state. The interior of a ring is on no trajectory at all, so
        nothing else in the battery ever seeds there. That this is not
        hypothetical was checked by hand on an earlier delaygo RNN: seeding at
        the delay ring's centroid converged to a genuine interior fixed point
        (rel_step 5.7e-8) sitting 3.05 from the centroid in a ring of radius
        6.23, which no diagonal probe can reach.

    noise_frac : jitter sigma as a fraction of each unit's own std across all
                 recorded states, so the scale follows the network rather than
                 being a hard-coded number.

    Returns (init_h, how) — `how` describes the draw, for the caller to print.
    """
    rng = np.random.RandomState(int(seed))
    flat = np.asarray(h_all, dtype=np.float64).reshape(-1, np.shape(h_all)[-1])
    base = flat[rng.randint(0, flat.shape[0], size=int(n))]
    n_pull = int(n) // 2
    if n_pull:
        mu = flat.mean(axis=0)
        frac = rng.uniform(0.0, 1.0, size=(n_pull, 1))
        base[:n_pull] = mu[None, :] + frac * (base[:n_pull] - mu[None, :])
    base = base + noise_frac * flat.std(axis=0)[None, :] * rng.randn(*base.shape)
    return (base.astype(np.float32),
            f"{int(n) - n_pull} recorded + {n_pull} pulled toward the mean, "
            f"jitter {noise_frac:g}x per-unit std")


def _leak_factor(net, net_params):
    """(alpha, 1 − alpha) for the update F(h) = alpha*h + (1 − alpha)*phi(.).

    Returns (0.0, 1.0) for a non-leaky net, where F(h) = phi(.) and nothing is
    damped. Read from the NETWORK first so the number describes the object the
    solve actually ran on, falling back to the saved config."""
    leaky = bool(getattr(net, "leaky", net_params.get("leaky", False)))
    if not leaky:
        return 0.0, 1.0
    a = getattr(net, "alpha", net_params.get("alpha", 0.0))
    a = float(a.item()) if hasattr(a, "item") else float(a)
    return a, max(1.0 - a, 1e-12)


def _relative_spread(fixed_h):
    """max_i ||h*_i − mean|| / ||mean|| over one probe's solved points.

    The one number separating "a single fixed point" from "a manifold of them":
    0 means every point is the SAME state, large means they spread along a ring.
    """
    A = np.asarray(fixed_h, dtype=np.float64)
    mu = A.mean(axis=0)
    return float(np.linalg.norm(A - mu[None, :], axis=1).max()
                 / max(np.linalg.norm(mu), 1e-12))


def _annotate_ring_distance(results, probe_name, ref_names):
    """Record where a synthesized-seed probe's points LANDED, relative to the
    fixed points the trial itself visits under the same input. Called for both
    the trajectory- and the naive-seeded probes.

    `ref_names` are the diagonal probes solved under an identical input. For the
    context input that is normally BOTH the single context point and the delay
    ring, and the two answer different questions — "did the seeds fall back to
    the trivial state?" versus "did they find the memory ring?" — so all of them
    are recorded:

      ref_dist[ref]     (n,) min over that reference's points of
                        ||h* − h*_ref|| / ||h*_ref||
      ref_nearest[ref]  (n,) the argmin's stimulus index in that reference
      ref_spread[ref]   scalar relative spread of the reference itself
                        (≈0 = a single point, large = a ring)

    The most ring-like reference (largest spread) additionally fills the plain
    ring_dist / ring_angle_idx / ring_ref fields, which is what paper_plot reads
    to colour such a point by the angle it landed on. Mutates `results` in place;
    a no-op when no reference is present."""
    e = results.get(probe_name)
    if e is None:
        return
    A = np.asarray(e["fixed_hidden"], dtype=np.float64)
    A = A.reshape(A.shape[0], -1)                             # (n, D)
    a_sq = (A ** 2).sum(axis=1)

    e["ref_dist"], e["ref_nearest"], e["ref_spread"] = {}, {}, {}
    for ref in ref_names:
        ref_e = results.get(ref)
        if ref_e is None:
            continue
        R = np.asarray(ref_e["fixed_hidden"], dtype=np.float64)
        R = R.reshape(R.shape[0], -1)                          # (m, D)
        r_sq = (R ** 2).sum(axis=1)
        r_norm = np.maximum(np.sqrt(r_sq), 1e-12)
        # ||a − r||² = |a|² + |r|² − 2a·r, then normalize each column by |r|.
        d2 = a_sq[:, None] + r_sq[None, :] - 2.0 * (A @ R.T)
        rel = np.sqrt(np.maximum(d2, 0.0)) / r_norm[None, :]
        j = np.argmin(rel, axis=1)
        e["ref_dist"][ref] = rel[np.arange(rel.shape[0]), j]
        e["ref_nearest"][ref] = np.asarray(ref_e["stim"], dtype=int)[j]
        e["ref_spread"][ref] = _relative_spread(ref_e["fixed_hidden"])

    if not e["ref_dist"]:
        return
    ringiest = max(e["ref_spread"], key=lambda r: e["ref_spread"][r])
    e["ring_ref"] = ringiest
    e["ring_dist"] = e["ref_dist"][ringiest]
    e["ring_angle_idx"] = e["ref_nearest"][ringiest]


def stim_color(k, n):
    """Stimulus color: red→purple rainbow, matching SCHEME.md's family 1."""
    frac = (k % n) / max(n - 1, 1)
    return mpl.colors.hsv_to_rgb((0.83 * frac, 0.85, 0.9))


def _rebuild_rnn(net_params):
    """Instantiate the RNN class implied by net_params (no weights loaded)."""
    if net_params["net_type"] == "vanilla":
        return nets.VanillaRNN(net_params, verbose=False)
    if net_params["net_type"] == "gru":
        return nets.GRU(net_params, verbose=False)
    raise ValueError(
        f"one_task_rnn_analysis only handles RNNs; got "
        f"net_type={net_params['net_type']!r}. Use one_task_analysis.py for the MPN.")


def _detect_stim_channels(template, fix_on, fix_off, stim_on, stim_off, rule):
    """The active ring's (sin θ, cos θ) channel pair in the input.

    A ring channel carries energy during the stimulus window but is ~zero
    throughout fixation, which rejects the always-on fixation bit and the
    constant rule cue. We find the single most energetic such channel and recover
    its PARTNER from the layout — low-dim rings are consecutive (sin, cos)
    2-blocks anchored just after the leading fixation channel(s). Detecting one
    suffices because for a stimulus on an axis (θ ≈ 0/90/180/270°) one of sin/cos
    is exactly zero across the whole window. Same rule as
    grad_fixed_points._detect_stim_channels, so both analyses pick the same pair.
    """
    stim_energy = (template[stim_on:stim_off] ** 2).sum(axis=0)
    fix_max = np.abs(template[fix_on:fix_off]).max(axis=0)
    stim_only = np.where(fix_max <= 1e-6, stim_energy, 0.0)
    c0 = int(np.argmax(stim_only))
    if stim_only[c0] <= 1e-9:
        raise ValueError(
            f"could not auto-detect a stimulus channel for rule={rule} "
            f"(stim-only energies {np.round(stim_only, 3).tolist()})")
    on_in_fix = fix_max > 1e-6
    n_fix = 0
    while n_fix < on_in_fix.size and on_in_fix[n_fix]:
        n_fix += 1
    ring_start = n_fix + ((c0 - n_fix) // 2) * 2
    if ring_start + 1 >= template.shape[1]:
        raise ValueError(f"stimulus ring pair out of range for rule={rule}")
    return ring_start, ring_start + 1


def _solve_one_seed(net, cfg, device,
                    n_interp=64, steps=20000, learningRate=1e-3,
                    loss_tol=1e-8, lbfgs_steps=2000, rel_tol=0.05,
                    analyze_stability=True, n_eigs=16, task_seed=0,
                    probes=None, naive_rng_seed=0, traj_noise_frac=0.25):
    """Solve h* = F(h*; x) for `probes` on ONE deterministic trial template.

    The template (stimulus modality, angle, epoch timing) is drawn from the task
    RNG, so it is fully determined by `task_seed` — re-running reproduces it.
    `probes` defaults to the diagonal battery; solve_hidden_fixed_points drives
    this for several seeds and then once more for the off-diagonal probes.

    Returns (results, angles, input_info). One entry per PROBE:
      fixed_hidden (n_interp, hidden)  the solved fixed points
      init_h       (n_interp, hidden)  the seeds they came from
      hidden_out   (n_interp, n_out)   the readout at h*
      rel_step     (n_interp,)         ||F(h*) − h*|| / ||h*||, scale-free
      is_fixed     (n_interp,)         rel_step <= rel_tol
      input_period / seed_source / is_diagonal — which probe this is
      plus the stability spectrum when `analyze_stability`.

    `input_info` reports which periods share a constant input, so the caller can
    build the synthesized batteries per DISTINCT input instead of per period.

    naive_rng_seed  : RNG seed for both synthesized seed families.
    traj_noise_frac : jitter for the trajectory seeds, as a fraction of each
                      unit's std across the recorded states.
    """
    # ── Dense interpolated-stimulus batch (one real trial, copied per angle) ──
    tp = copy.deepcopy(cfg["task_params"])
    tp["long_fixation"] = tp["long_stimulus"] = "normal"
    tp["long_delay"] = tp["long_response"] = "normal"
    tp, trp, npp = mpn_tasks.convert_and_init_multitask_params(
        (tp, copy.deepcopy(cfg["train_params"]), copy.deepcopy(cfg["net_params"])))
    npp["prefs"] = mpn_tasks.get_prefs(tp["hp"])
    tp["hp"]["batch_size_train"] = 1
    # Pin the task RNG so the template (stimulus modality, angle, epoch timing)
    # is deterministic — re-running the analysis reproduces the same figure.
    tp["hp"]["seed"] = int(task_seed)
    tp["hp"]["rng"] = np.random.RandomState(int(task_seed))
    data, extra = mpn_tasks.generate_trials_wrap(
        tp, 1, rules=tp["rules"], mode_input="random", device=device)
    _, trials, _ = extra
    template = np.asarray(data[0].detach().cpu())[0]        # (T, n_input)
    T = template.shape[0]
    rule = tp["rules"][0]

    def _ep(name):
        e = trials[0].epochs[name]
        return (0 if e[0] is None else int(e[0]),
                T if e[1] is None else int(e[1]))

    windows = {}
    for pkey, (epoch_name, _title) in PERIODS.items():
        try:
            windows[pkey] = _ep(epoch_name)
        except KeyError:
            print(f"  [hidden-fp] epoch {epoch_name} absent from this trial; "
                  f"skipping {pkey}.")

    fix_on, fix_off = windows["longfixation"]
    stim_on, stim_off = windows["longstimulus"]
    ch_a, ch_b = _detect_stim_channels(template, fix_on, fix_off,
                                       stim_on, stim_off, rule)
    print(f"  [hidden-fp] stimulus ring channels ({ch_a}, {ch_b})")

    angles = np.arange(n_interp) * (2 * np.pi / n_interp)
    batch = np.repeat(template[None, :, :], n_interp, axis=0)
    batch[:, stim_on:stim_off, ch_a] = np.sin(angles)[:, None]
    batch[:, stim_on:stim_off, ch_b] = np.cos(angles)[:, None]

    x = torch.as_tensor(batch, dtype=torch.float, device=device)
    with torch.no_grad():
        _, _, db = net.iterate_sequence_batch(
            x, run_mode="track_states", save_to_cpu=True, detach_saved=True)
    # Read the hidden trace from the tracked-state dict, NOT from
    # iterate_sequence_batch's second return value: that one does
    # `step_activity[-1]`, which is the last LAYER for an MPN but the last BATCH
    # ROW for an RNN (whose network_step returns a plain (B, H) tensor), and it
    # would silently broadcast one trial's state across the batch.
    h_all = np.asarray(db["hidden"])                        # (n_interp, T, hidden)
    print(f"  [hidden-fp] recorded hidden states {h_all.shape}")

    # ── How many DISTINCT constant inputs does the period battery have? ─────
    # Pairwise max|x_a − x_b| between the period midpoints. Two periods at
    # distance 0 pose the SAME fixed-point problem — their fixed-point sets are
    # one set — so four periods can probe fewer than four maps. In the delaygo
    # family Context and Delay coincide (fixation bit and rule cue on in both,
    # stimulus channels zero in both), giving three maps for four periods.
    # Measured per run so a task where that fails (extra epochs, fixate_off,
    # another family) is caught instead of assumed, and so the naive battery can
    # skip the duplicate input by itself.
    in_names = [v for v, (a, b) in windows.items() if 0 <= a < b <= T]
    in_t = {v: min((windows[v][0] + windows[v][1]) // 2, T - 1) for v in in_names}
    in_dist = np.zeros((len(in_names), len(in_names)))
    for i, a in enumerate(in_names):
        for j, b in enumerate(in_names):
            in_dist[i, j] = np.abs(batch[:, in_t[a], :]
                                   - batch[:, in_t[b], :]).max()
    input_info = {"periods": in_names, "t_mid": in_t, "dist": in_dist}
    groups = _same_input_groups(input_info)
    print(f"  [hidden-fp] seed={task_seed}: {len(groups)} distinct period "
          f"input(s) among {len(in_names)}: "
          + " | ".join("=".join(g) for g in groups))

    if probes is None:
        probes = _diagonal_probes(in_names)
    print(f"  [hidden-fp] seed={task_seed}: solving {len(probes)} probe(s): "
          + ", ".join(p[0] for p in probes))

    # Leak, for the normalized twins of the convergence and stability numbers.
    alpha, leak = _leak_factor(net, cfg["net_params"])

    results = {}
    for name, in_period, seed_src, title in probes:
        ps, pe = windows.get(in_period, (0, 0))
        if not (0 <= ps < pe <= T):
            continue
        t_mid = min((ps + pe) // 2, T - 1)
        const_input = batch[:, t_mid, :]                    # (n_interp, n_input)

        # Seed: a period's end state, or synthesized states carrying no stimulus
        # information (t_seed = -1 marks "not from a recorded step").
        if seed_src in _SYNTH_SEEDS:
            t_seed = -1
            if seed_src == _TRAJ_SEED:
                init_h, how = _trajectory_seeds(
                    n_interp, h_all, noise_frac=traj_noise_frac,
                    seed=naive_rng_seed)
            else:
                init_h, how = _naive_hidden_seeds(
                    n_interp, h_all.shape[-1],
                    activation=cfg["net_params"].get("activation", "tanh"),
                    seed=naive_rng_seed, scale_ref=h_all)
            print(f"  [hidden-fp] {name}: seeds drawn {how}")
        else:
            qs, qe = windows.get(seed_src, (0, 0))
            if not (0 <= qs < qe <= T):
                print(f"  [hidden-fp] {name}: seed period '{seed_src}' absent "
                      f"from this trial; skipping the probe.")
                continue
            t_seed = min(qe - 1, T - 1)
            init_h = h_all[:, t_seed, :]                    # (n_interp, hidden)

        # A probe is diagonal when its input and its seed come from the same
        # period — the only case in which the recorded path actually visited the
        # fixed points being solved for.
        diagonal = (seed_src == in_period)
        seed_desc = (f"{seed_src} t={t_seed}" if t_seed >= 0
                     else f"{seed_src} (rng {naive_rng_seed})")
        print(f"  [hidden-fp] {name}: solving {n_interp} fixed points "
              f"(input {in_period} t={t_mid}, seed {seed_desc})")
        fixed_h, loss_hist, final_speeds = find_hidden_fixed_points(
            net, init_h, const_input, steps=steps, learningRate=learningRate,
            printPeriod=max(steps // 10, 1), loss_tol=loss_tol,
            lbfgs_steps=lbfgs_steps, device=device)

        # Readout at the fixed point, so the figure can show what the network
        # would be reporting if it sat there forever.
        with torch.no_grad():
            h_t = torch.as_tensor(fixed_h, dtype=torch.float, device=device)
            out_h = (h_t @ net.W_output.T.to(device))
            if getattr(net, "b_output_active", False):
                out_h = out_h + net.b_output.unsqueeze(0).to(device)
            hidden_out = np.asarray(out_h.detach().cpu())

        # Scale-free convergence: final_speeds is q = 1/2||F-h||^2, so
        # ||F-h|| = sqrt(2q).
        step_norm = np.sqrt(2.0 * np.asarray(final_speeds, dtype=float))
        h_norm = np.maximum(np.linalg.norm(fixed_h, axis=1), 1e-12)
        rel_step = step_norm / h_norm
        is_fixed = rel_step <= rel_tol
        # Leak-normalized twin of rel_step. F(h) − h = (1 − a)(phi(.) − h), so the
        # residual we measure is DAMPED by the leak: with a = 0.8 every point
        # reads five times more converged than the underlying map warrants, and
        # rel_tol = 0.05 is really 0.25 on phi. The criterion is deliberately NOT
        # changed — rel_tol keeps one meaning across this codebase — but the
        # undamped number is recorded and printed beside it so nobody has to
        # rediscover the factor.
        rel_step_undamped = rel_step / leak
        print(f"  [hidden-fp] {name}: {int(is_fixed.sum())}/{is_fixed.size} "
              f"converged (rel_step <= {rel_tol:g}); "
              f"median {np.median(rel_step):.2e} max {rel_step.max():.2e}"
              + (f" | undamped median {np.median(rel_step_undamped):.2e} "
                 f"max {rel_step_undamped.max():.2e}" if leak < 1.0 else ""))

        entry = {
            "period_title": title,
            # Which input the points were solved under, and where the solve
            # started — the two axes that make this a probe rather than just
            # "the context period". paper_plot's grad-fixed-point renderers read
            # them to decide which panel a probe belongs to and whether it is
            # drawn as that panel's own points or as an overlay, which is what
            # lets the RNN pickle be drawn by the SAME code as the MPN one.
            "input_period": in_period,
            "seed_source": seed_src,
            "is_diagonal": bool(diagonal),
            # `stim` is the dense stimulus-angle index for period-seeded probes.
            # A synthesized seed carries no stimulus, so there it is only a seed
            # index — those points are coloured by ring_angle_idx instead.
            "stim_is_stimulus": bool(seed_src not in _SYNTH_SEEDS),
            "traj_stim": _TRAJ_STIM,
            "period": (int(ps), int(pe)),
            "t_input": int(t_mid),
            "t_seed": int(t_seed),
            "stim": np.arange(n_interp),
            "init_h": np.asarray(init_h, dtype=np.float32),
            # Named `fixed_hidden`, not `fixed_h`: that is the key the shared
            # renderers select with rep_key="fixed_hidden".
            "fixed_hidden": np.asarray(fixed_h, dtype=np.float32),
            "hidden_out": np.asarray(hidden_out, dtype=np.float32),
            "const_input": np.asarray(const_input, dtype=np.float32),
            "final_speeds": np.asarray(final_speeds, dtype=float),
            "rel_step": np.asarray(rel_step, dtype=float),
            # The same residual relative to the UNDAMPED map phi(.), i.e.
            # divided by (1 - alpha). See the comment where it is computed.
            "rel_step_undamped": np.asarray(rel_step_undamped, dtype=float),
            "alpha": float(alpha),
            "leak": float(leak),
            "is_fixed": np.asarray(is_fixed, dtype=bool),
            "rel_tol": float(rel_tol),
            "loss_hist": np.asarray(loss_hist, dtype=float),
            # Recorded within-period path of the exemplar stimulus, so a figure
            # can show how the state travelled to its fixed point. DIAGONAL
            # probes only: an off-diagonal probe's fixed points were never on the
            # recorded path, so a connector drawn to them would be fiction.
            # paper_plot draws no connector when this is None.
            "traj_hidden": (np.asarray(h_all[_TRAJ_STIM, ps:pe, :],
                                       dtype=np.float32) if diagonal else None),
            # 0 ⇒ every point is the same state (ONE fixed point); large ⇒ the
            # points spread along a manifold (a ring).
            "across_angle_spread": _relative_spread(fixed_h),
        }

        if analyze_stability:
            try:
                stab = characterize_hidden_fixed_point_stability(
                    net, fixed_h, const_input, k=n_eigs, device=device)
                entry.update({
                    "eigenvalues": stab["eigenvalues"],
                    "spectral_radius": stab["spectral_radius"],
                    "n_unstable": stab["n_unstable"],
                    "n_marginal": stab["n_marginal"],
                    "stab_is_stable": stab["is_stable"],
                    "marginal_tol": stab["marginal_tol"],
                })
                # Leak-normalized spectrum. J = a*I + (1 - a)*D*W, so every
                # eigenvalue is lam = a + (1 - a)*mu with mu an eigenvalue of the
                # UNDAMPED recurrent map, and |lam - 1| = (1 - a)|mu - 1|. With
                # a = 0.8 the marginal band is therefore five times wider than
                # it reads: a direction at mu = 1.2 — plainly expanding — lands
                # at lam = 1.04 and is counted marginal. `n_marginal_leak_only`
                # is exactly that population, so "one marginal direction = ring
                # attractor" can be checked rather than assumed. Computed from
                # the reported leading eigenvalues, which is where any |lam| ~ 1
                # direction necessarily sits.
                ev = np.asarray(stab["eigenvalues"])
                mtol = float(stab["marginal_tol"])
                mu = (ev - alpha) / leak
                leak_only = (np.abs(ev - 1.0) <= mtol) & (np.abs(mu - 1.0) > mtol)
                entry.update({
                    "eigenvalues_undamped": mu,
                    "spectral_radius_undamped": np.abs(mu).max(axis=-1),
                    "n_marginal_leak_only": leak_only.sum(axis=-1),
                })
                rad = stab["spectral_radius"]
                print(f"  [hidden-fp] {name}: spectral radius median "
                      f"{np.nanmedian(rad):.3f} (max {np.nanmax(rad):.3f}); "
                      f"{int(stab['is_stable'].sum())}/{stab['is_stable'].size} "
                      f"stable, marginal-dir median "
                      f"{int(np.median(stab['n_marginal']))}")
                if leak < 1.0:
                    print(f"  [hidden-fp] {name}: undamped spectral radius "
                          f"median {np.nanmedian(entry['spectral_radius_undamped']):.3f}; "
                          f"of the marginal directions, a median of "
                          f"{int(np.median(entry['n_marginal_leak_only']))} are "
                          f"marginal ONLY because of the leak "
                          f"(|mu-1| > {mtol:g})")
            except Exception as exc:
                print(f"  [hidden-fp] {name}: stability analysis failed: {exc}")

        print(f"  [hidden-fp] {name}: across-angle spread of h* = "
              f"{entry['across_angle_spread']:.3e}")
        results[name] = entry

    return results, angles, input_info


def _selection_score(results):
    """Lower = better. Median rel_step over the STIMULUS + RESPONSE probes only —
    the parts most sensitive to the random template, since those are the periods
    whose input actually depends on the drawn stimulus. Missing periods
    contribute nothing; if neither is present, fall back to all probes."""
    keys = [k for k in ("longstimulus", "longresponse") if k in results]
    if not keys:
        keys = list(results.keys())
    vals = (np.concatenate([np.asarray(results[k]["rel_step"], float)
                            for k in keys]) if keys else np.array([np.inf]))
    return float(np.median(vals))


def solve_hidden_fixed_points(net, cfg, device,
                              n_interp=64, steps=20000, learningRate=1e-3,
                              loss_tol=1e-8, lbfgs_steps=2000, rel_tol=0.05,
                              analyze_stability=True, n_eigs=16,
                              n_seeds=5, seed_base=0,
                              cross_seed_probes=True, naive_seed_probes=True,
                              traj_seed_probes=True, traj_noise_frac=0.25,
                              naive_rng_seed=0):
    """Solve h* = F(h*; x) over a dense stimulus ring, for the whole probe
    battery, on the best of `n_seeds` trial templates.

    Mirrors solve_period_modulation_fixed_points in core/grad_fixed_points.py,
    moved from M to h — including the two things that make a solve trustworthy
    rather than lucky:

      * MULTIPLE TEMPLATES. The trial template (stimulus modality, angle, epoch
        timing) comes from the task RNG, so one draw carries run-to-run
        variability that has nothing to do with the network. The diagonal battery
        is solved for `n_seeds` DETERMINISTIC templates and the single
        best-converging one is kept, judged by `_selection_score`. Reproducible
        and the best of several candidates, rather than whatever one draw gave.
      * MULTIPLE SEEDS PER TEMPLATE. Within the selected template, the
        off-diagonal probes re-solve the same inputs from different starting
        states — see the probe block at the top of this module.

    Returns (results, angles, selected_seed).

    n_seeds     : how many task-RNG templates to try (1 = a single template).
    seed_base   : first task-RNG seed to try.
    cross_seed_probes : add "longfixation_memseed" — the CONTEXT input seeded
                  from the end of the DELAY. Same input as the diagonal context
                  probe, different basin, so a ring here means the memory ring
                  and the single context point coexist.
    naive_seed_probes : add one "{period}_naiveseed" per DISTINCT period input,
                  seeded from random hidden states carrying no stimulus
                  information — the direct analogue of the MPN's naive battery.
    traj_seed_probes : add one "{period}_trajseed" per DISTINCT period input,
                  seeded from the states the network actually visits, jittered
                  (_trajectory_seeds). The probe that does the real work: a
                  uniform draw covers almost nothing of a 200-D space, and the
                  rank-one family the MPN's naive battery appeals to does not
                  survive its modulation bounds either.
    traj_noise_frac : jitter for those seeds, as a fraction of each unit's std.
    naive_rng_seed : RNG seed for both synthesized batteries (reproducibility).
    """
    common = dict(n_interp=n_interp, steps=steps, learningRate=learningRate,
                  loss_tol=loss_tol, lbfgs_steps=lbfgs_steps, rel_tol=rel_tol,
                  analyze_stability=analyze_stability, n_eigs=n_eigs,
                  naive_rng_seed=naive_rng_seed,
                  traj_noise_frac=traj_noise_frac)

    # State the leak factor ONCE, up front: every rel_step and every |lam - 1|
    # below is damped by it, so the thresholds are looser than they read.
    alpha, leak = _leak_factor(net, cfg["net_params"])
    if leak < 1.0:
        print(f"  [hidden-fp] leak: alpha={alpha:.3f}, 1-alpha={leak:.3f} — both "
              f"the residual and the eigenvalue band are compressed by that "
              f"factor, so rel_tol={rel_tol:g} is {rel_tol / leak:.3g} on the "
              f"undamped map and every threshold below reads "
              f"{1.0 / leak:.1f}x tighter than it is. Undamped twins are "
              f"printed and saved beside the damped numbers.")

    # ── Try n_seeds deterministic templates; keep the best-converging one ────
    best = None   # (score, task_seed, results, angles, input_info)
    for s in range(seed_base, seed_base + max(int(n_seeds), 1)):
        try:
            results, angles, input_info = _solve_one_seed(
                net, cfg, device, task_seed=s, **common)
        except Exception as exc:
            print(f"  [hidden-fp] seed={s} failed: {exc}")
            continue
        if not results:
            continue
        score = _selection_score(results)
        print(f"  [hidden-fp] seed={s}: selection score "
              f"(stim+resp median rel_step) = {score:.3e}")
        if best is None or score < best[0]:
            best = (score, s, results, angles, input_info)

    if best is None:
        raise RuntimeError(
            f"no task seed in [{seed_base}, {seed_base + max(int(n_seeds), 1)}) "
            f"produced hidden fixed points.")

    best_score, best_seed, results, angles, input_info = best
    print(f"  [hidden-fp] selected seed={best_seed} (score {best_score:.3e} "
          f"over {n_seeds} seed(s)).")

    # ── Off-diagonal probes, on the SELECTED template only ───────────────────
    # Solved here rather than inside the sweep because the selection score only
    # looks at the stimulus and response periods — running these per candidate
    # template would cost n_seeds× and change nothing. Re-entering
    # _solve_one_seed rebuilds the identical template (deterministic in the
    # seed) for one extra forward pass.
    groups = _same_input_groups(input_info)
    extra = _extra_probes(list(results), input_info,
                          cross_seed_probes=cross_seed_probes,
                          naive_seed_probes=naive_seed_probes,
                          traj_seed_probes=traj_seed_probes)
    if extra:
        print(f"  [hidden-fp] solving {len(extra)} multistability probe(s) on "
              f"the selected seed={best_seed}: {[p[0] for p in extra]}")
        extra_results, _, _ = _solve_one_seed(
            net, cfg, device, task_seed=best_seed, probes=extra, **common)
        results.update(extra_results)

        # A synthesized seed carries no stimulus label, so instead of a label
        # record WHERE it landed: distance to every diagonal probe under the same
        # input (for the context input that is both the single context point and
        # the delay ring — different questions, both worth asking).
        for name, in_period, seed_src, _title in extra:
            if seed_src not in _SYNTH_SEEDS or name not in results:
                continue
            grp = next((g for g in groups if in_period in g), [in_period])
            refs = [v for v in grp
                    if v in results and results[v].get("is_diagonal")]
            _annotate_ring_distance(results, name, refs)
            e = results[name]
            for ref, rd in e.get("ref_dist", {}).items():
                rd = np.asarray(rd, dtype=float)
                kind = "ring" if e["ref_spread"][ref] > 1e-3 else "point"
                print(f"  [hidden-fp] {name}: vs {ref} ({kind}) — median "
                      f"relative distance {np.median(rd):.3f}, "
                      f"{int((rd <= 0.1).sum())}/{rd.size} within 0.1")

    return results, angles, best_seed


def plot_hidden_fixed_points(results, angles, aname, out_path):
    """One panel per period: the solved hidden fixed points in a SHARED
    delay-period PCA, colored by stimulus.

    The basis is fit on the delay period's fixed points and reused by every
    panel, so PC1/PC2 mean the same axes across panels and the periods can be
    compared point for point. Converged points are filled, over-threshold ones
    hollow (the project's usual convention).

    DIAGONAL probes only — this is the quick local check that the solve worked.
    The off-diagonal probes (memory-, trajectory- and naive-seeded) are drawn by
    paper_plot, which overlays each onto the panel of the input it was solved
    under."""
    periods = [p for p in PERIODS if p in results]
    if not periods:
        print("  [hidden-fp] nothing solved; no figure.")
        return
    basis_key = "longdelay" if "longdelay" in results else periods[0]
    pca = PCA(n_components=2, random_state=0).fit(
        np.asarray(results[basis_key]["fixed_hidden"], dtype=float))
    pc_label = results[basis_key]["period_title"]

    proj = {p: pca.transform(np.asarray(results[p]["fixed_hidden"], dtype=float))
            for p in periods}
    lim = max(np.abs(np.vstack(list(proj.values()))).max() * 1.08, 1e-9)
    n_stim = len(angles)

    fig, axs = plt.subplots(1, len(periods), figsize=(2.3 * len(periods), 2.4),
                            squeeze=False)
    for ax, p in zip(axs[0], periods):
        e = results[p]
        xy = proj[p]
        good = np.asarray(e["is_fixed"], dtype=bool)
        cols = [stim_color(k, n_stim) for k in range(xy.shape[0])]
        # Connect the converged points in stimulus order: a smooth closed ring
        # means a continuous attractor, a few clumps mean discrete ones.
        if good.sum() >= 2:
            ring = xy[good]
            ax.plot(np.append(ring[:, 0], ring[0, 0]),
                    np.append(ring[:, 1], ring[0, 1]),
                    color="0.6", lw=0.6, linestyle="--", zorder=2)
        for i in range(xy.shape[0]):
            if good[i]:
                ax.scatter(xy[i, 0], xy[i, 1], color=cols[i], s=18,
                           edgecolor="none", zorder=3)
            else:
                ax.scatter(xy[i, 0], xy[i, 1], facecolor="none",
                           edgecolor=cols[i], s=18, linewidth=0.9, zorder=3)
        ax.set_title(f"{e['period_title']}\n"
                     f"{int(good.sum())}/{good.size} fixed", fontsize=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel(f"{pc_label} PC1", fontsize=9)
    fig.supylabel(f"{pc_label} PC2", fontsize=9)
    fig.suptitle(f"{aname}  |  hidden-state fixed points", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def main(aname, n_interp=64, steps=20000, rel_tol=0.05, analyze_stability=True,
         cross_seed_probes=True, naive_seed_probes=True, traj_seed_probes=True,
         fp_n_seeds=5):
    """Run the hidden-state fixed-point analysis for ONE trained RNN run."""
    param_path = RNN_DIR / f"param_{aname}_param.json"
    ckpt_path = RNN_DIR / f"savednet_{aname}.pt"
    for p in (param_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run one_task_rnn.py first.")

    with open(param_path) as f:
        cfg = json.load(f)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = _rebuild_rnn(ckpt["net_params"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()
    print(f"Loaded {type(net).__name__} on {device}")

    save_dir = RNN_DIR / aname
    save_dir.mkdir(parents=True, exist_ok=True)

    results, angles, selected_seed = solve_hidden_fixed_points(
        net, cfg, device, n_interp=n_interp, steps=steps,
        rel_tol=rel_tol, analyze_stability=analyze_stability,
        n_seeds=fp_n_seeds, cross_seed_probes=cross_seed_probes,
        naive_seed_probes=naive_seed_probes,
        traj_seed_probes=traj_seed_probes)

    out_pkl = save_dir / f"fixed_points_hidden_{aname}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"aname": aname, "n_interp": int(n_interp),
                     "rel_tol": float(rel_tol),
                     # Which of the `n_seeds` trial templates these fixed points
                     # came from, so a figure can be traced back to its template.
                     "n_seeds": int(fp_n_seeds),
                     "selected_seed": int(selected_seed),
                     # `rule` lets a reader (and paper_plot's z axis) know whether
                     # the required response is the stimulus angle (pro) or its
                     # opposite (anti).
                     "rule": str(cfg["task_params"]["rules"][0]),
                     "angles": np.asarray(angles, dtype=float),
                     "results": results}, f)
    print(f"Saved hidden fixed-point data: {out_pkl}")

    plot_hidden_fixed_points(results, angles, aname,
                             save_dir / f"fixed_points_hidden_{aname}.png")
    print(f"All outputs saved to {save_dir}/")


def _discover_anames():
    """Every run identifier (param_*_result.npz) under onetask_rnn/."""
    results = sorted(RNN_DIR.glob("param_*_result.npz"),
                     key=lambda p: p.stat().st_mtime)
    if not results:
        raise FileNotFoundError(
            f"No param_*_result.npz in ./{RNN_DIR}/. Run one_task_rnn.py first.")
    return [p.name[len("param_"):-len("_result.npz")] for p in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aname", type=str, default=None,
                        help="Run identifier. Omit to analyze every run found.")
    parser.add_argument("--n-interp", type=int, default=64,
                        help="Stimulus angles on the dense ring (default 64; "
                             "pass 8 for the trained directions only).")
    parser.add_argument("--steps", type=int, default=20000,
                        help="MAX Adam iterations per solve (it stops early once "
                             "the speed loss reaches the tolerance).")
    parser.add_argument("--rel-tol", type=float, default=0.05,
                        help="rel_step below which a point counts as a fixed "
                             "point (default 0.05).")
    parser.add_argument("--no-stability", dest="analyze_stability",
                        action="store_false",
                        help="Skip the Jacobian eigenvalue pass.")
    parser.add_argument("--fp-n-seeds", type=int, default=5,
                        help="How many deterministic trial templates to solve "
                             "the diagonal battery on, keeping the "
                             "best-converging one (default 5; 1 = single "
                             "template).")
    parser.add_argument("--diagonal-only", dest="extra_probes",
                        action="store_false",
                        help="Solve only each period under its own input from "
                             "its own end state, skipping the memory-, "
                             "trajectory- and naive-seed probes (the old, "
                             "cheaper battery).")
    parser.set_defaults(analyze_stability=True, extra_probes=True)
    args = parser.parse_args()

    anames = [args.aname] if args.aname else _discover_anames()
    print(f"Analyzing {len(anames)} run(s).")
    for a in anames:
        print(f"\n── Analyzing: {a} ──")
        try:
            main(a, n_interp=args.n_interp, steps=args.steps,
                 rel_tol=args.rel_tol, analyze_stability=args.analyze_stability,
                 cross_seed_probes=args.extra_probes,
                 naive_seed_probes=args.extra_probes,
                 traj_seed_probes=args.extra_probes,
                 fp_n_seeds=args.fp_n_seeds)
        except Exception as exc:
            print(f"  FAILED {a}: {exc}")
            import traceback
            traceback.print_exc()
