"""
Feedback and saturation of the plastic layer's modulation M.

The plastic layer of the multi-task MPN updates, per time step (core/mpn.py:
h is computed with the old M, then M is updated and clipped to its bounds):

    h_t = tanh((W * (1 + M_{t-1})) x_t + b)
    M_t = clip(lam M_{t-1} + eta h_t x_t^T, lower, upper)

so M feeds back on itself through h. Linearized about the trajectory, the row
M_i of post neuron i maps a perturbation of M_{t-1} as

    dM_i -> P_i (lam dM_i + eta (1 - h_i^2) <W_i * x, dM_i> x),

where P_i zeroes the synapses whose update was clipped this step. That is lam
times the identity plus a rank-one term, so on the unclipped synapses every
direction the neuron cannot read (orthogonal to W_i * x) decays by exactly
lam, and the one it reads, <W_i * x, M_i>, by

    mu_i = lam + eta (1 - h_i^2) sum_{j unclipped} W_ij x_j^2.

mu_i close to 1 keeps what h reads for longer than lam does, mu_i > 1 drives
the row to its bounds and mu_i < lam forgets it faster; lam itself cannot be
changed by training (lam_train is False in every run). A clipped synapse
holds its bound for as long as the drive pushes it there, which at lam = 0.99
takes only eta h_i x_j > 1 - lam = 0.01.

For one run, trained or untrained (--untrained-init-seed,
network_loading.build_network), this script steps the network through the
evaluation trials of the clustering analysis (multiple_task_analysis.py: 50
trials per task, mode_input="random", long_all=False) and records at every
step mu_i of every post neuron and the share of synapses clipped to each
bound, without keeping the M trajectory. mu_i is left out (NaN) for a row
whose synapses are all clipped, which has no mode left. A finite-difference
check through the real network step, in double precision, first confirms the
derivation: perturbations of M_{t-1} along a random direction, along the
readable mode (x on the unclipped synapses) and along a direction the neuron
cannot read must come out of one step as the linearized map, mu_i and lam
predict.

Writes multiple_tasks_diagnosis/modulation_feedback_{name}.{pkl,png}, with
name the run's aname, plus _init{k} for an untrained network: per task and
trial period the quantiles of mu_i, its shares above 1, below -1 and below lam, the
median time constant of the readable mode, dt / (1 - |mu|), and the shares of
synapses at each bound; the same pooled by period type; each task's
saturation time course; and the check's errors. The figure shows mu_i per
period type, the saturation time courses and the readable mode's time
constant against lam's.

Usage:
    python multiple_task/modulation_feedback.py --feature L21e4 --seed 749
    python multiple_task/modulation_feedback.py --feature L21e4 --seed 749 \\
        --untrained-init-seed 0
    # every run of the cohort
    python multiple_task/modulation_feedback.py --feature L21e4 --seed all
"""
import argparse
import copy
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are saved, never shown
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import mpn_tasks
from network_loading import build_network
from run_logging import tee_output
import multiple_task_analysis
import multiple_task_diagnosis as diag

CHECKPOINT_DIR = Path("multiple_tasks")
OUT_DIR = diag.OUT_DIR
RESULT_VERSION = 1
RUN_SUFFIX = "+hidden300+batch128+angle"
# The evaluation trials of the clustering analysis (multiple_task_analysis.py).
TRIALS_PER_TASK = 50
EVALUATION_SEED = 0
FORWARD_CHUNK = 50
# Finite-difference check: steps per trial, perturbation size (each probe has
# unit norm per row) and the probes' random seed.
CHECK_STEPS = 4
CHECK_EPSILON = 1e-6
CHECK_SEED = 0
QUANTILES = (5, 25, 50, 75, 95)
EPOCH_ORDER = ("fix1", "stim1", "delay1", "stim2", "delay2", "go1")
EPOCH_DISPLAY = {"fix1": "Context", **diag._PHASE_DISPLAY}
PERIOD_TYPES = {"Context": ("fix1",), "Stimulus": ("stim1", "stim2"),
                "Memory": ("delay1", "delay2"), "Response": ("go1",)}
HISTOGRAM_BINS = 80


def run_name(seed, feature):
    """The aname of one multi-task run."""
    return f"everything_seed{seed}_{feature}{RUN_SUFFIX}"


def result_stem(seed, feature, untrained_init_seed=None):
    """File stem of one network's result: the run's aname, plus _init{k} for an
    untrained network (multiple_task_analysis.output_names)."""
    name, _ = multiple_task_analysis.output_names(
        seed, "everything", f"{feature}{RUN_SUFFIX}", untrained_init_seed)
    return f"modulation_feedback_{name}"


def discover_seeds(feature, checkpoint_dir=CHECKPOINT_DIR):
    """Seeds of every saved checkpoint of `feature`, ascending."""
    pattern = re.compile(rf"savednet_everything_seed(\d+)_{re.escape(feature)}"
                         rf"{re.escape(RUN_SUFFIX)}\.pt")
    return sorted(int(match.group(1)) for path in Path(checkpoint_dir).glob("savednet_*.pt")
                  if (match := pattern.fullmatch(path.name)))


# ─── The linearized update ───────────────────────────────────────────────────

def plastic_layer(model):
    """The network's one plastic layer, after checking that it follows the
    update the derivation assumes (multiplicative, Hebbian, linear M, scalar
    eta and lam, no recurrent weights)."""
    if len(model.mp_layers) != 1 or getattr(model, "recurrent_active", False):
        raise ValueError("The derivation needs one plastic layer and no recurrent weights.")
    layer = model.mp_layers[0]
    expected = {"mp_type": "mult", "m_update_type": "hebb_assoc", "m_act": "linear",
                "eta_type": "scalar", "lam_type": "scalar"}
    found = {key: getattr(layer, key) for key in expected}
    if found != expected:
        raise ValueError(f"The derivation assumes the plastic layer {expected}, got {found}.")
    return layer


def layer_rates(layer):
    """The plastic layer's scalar (lam, eta)."""
    return (float(layer.lam.detach().reshape(-1)[0]), float(layer.eta.detach().reshape(-1)[0]))


def clipped_masks(layer):
    """(above, below): the synapses whose last update the layer clipped to its
    upper or its lower bound, from the unclipped update it keeps (M_pre)."""
    M_pre = layer.M_pre
    if not layer.modulation_bounds:
        none = torch.zeros_like(M_pre, dtype=torch.bool)
        return none, none
    return M_pre > layer.M_bounds[0], M_pre < layer.M_bounds[1]


def readable_mode_rate(h, x, W, unclipped, lam, eta):
    """mu_i = lam + eta (1 - h_i^2) sum_j unclipped_ij W_ij x_j^2, the factor
    by which one step scales what post neuron i reads of M.

    h (trials, post), x (trials, pre), W (post, pre) and `unclipped`
    (trials, post, pre) boolean, or None to sum over every synapse.
    Returns (trials, post).
    """
    drive = W.unsqueeze(0) * (x ** 2).unsqueeze(1)
    if unclipped is not None:
        drive = drive * unclipped
    return lam + eta * (1.0 - h ** 2) * drive.sum(-1)


def linear_response(delta, h, x, W, unclipped, lam, eta):
    """One step's linearized map applied to a perturbation `delta`
    (trials, post, pre) of M_{t-1}: P (lam delta + eta (1 - h^2) <W * x, delta> x)."""
    read = torch.einsum("bij,ij,bj->bi", delta, W, x)
    feedback = eta * ((1.0 - h ** 2) * read).unsqueeze(-1) * x.unsqueeze(1)
    return unclipped * (lam * delta + feedback)


def _unit_rows(probe):
    """`probe` scaled to unit norm per (trial, post neuron) row; all-zero rows
    stay zero."""
    norm = probe.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, probe / norm.clamp_min(torch.finfo(probe.dtype).tiny), probe)


def linearization_probes(x, W, unclipped, generator):
    """Three perturbations of M_{t-1}, unit norm per row: "random"; "readable",
    the readable mode's direction (x on the unclipped synapses), whose response
    is mu_i times it; and "unreadable", random on the unclipped synapses and
    orthogonal to W_i * x there, whose response is lam times it. A row with at
    most one unclipped synapse has no unreadable direction; its "unreadable"
    probe is zero instead of normalized round-off."""
    shape, dtype = unclipped.shape, x.dtype
    readout = unclipped * W.unsqueeze(0) * x.unsqueeze(1)
    unreadable = unclipped * torch.randn(shape, generator=generator, dtype=dtype, device=x.device)
    weight = ((unreadable * readout).sum(-1, keepdim=True)
              / (readout ** 2).sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny))
    orthogonal = unreadable - weight * readout
    kept = orthogonal.norm(dim=-1, keepdim=True) > 1e-6 * unreadable.norm(dim=-1, keepdim=True)
    return {
        "random": _unit_rows(torch.randn(shape, generator=generator, dtype=dtype, device=x.device)),
        "readable": _unit_rows(unclipped * x.unsqueeze(1)),
        "unreadable": _unit_rows(orthogonal * kept),
    }


def check_linearization(model, inputs, n_steps=CHECK_STEPS, epsilon=CHECK_EPSILON,
                        seed=CHECK_SEED):
    """Finite-difference check of the derivation through the real network
    step, on a double-precision copy of `model`.

    Runs `inputs` (trials, time, channels) and at `n_steps` evenly spaced steps
    perturbs M_{t-1} by `epsilon` times each probe (`linearization_probes`),
    takes the step again and compares the response with the prediction:
    the linearized map for "random", mu_i times the probe for "readable" and
    lam times it for "unreadable". The probes have unit norm per row, so a
    row's error is the norm of the difference relative to the perturbation.
    Returns {probe: {"median", "max"}} over the probed rows.
    """
    model = copy.deepcopy(model).double()
    layer = plastic_layer(model)
    lam, eta = layer_rates(layer)
    W = layer.W.detach()
    inputs = inputs.double()
    n_trials, n_time = inputs.shape[:2]
    steps = set(np.linspace(1, n_time - 1, n_steps).astype(int).tolist())
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    errors = {"random": [], "readable": [], "unreadable": []}
    with torch.no_grad():
        model.reset_state(B=n_trials)
        for t in range(n_time):
            M_prev = layer.M.clone()
            _, activities, _ = model.network_step(inputs[:, t], seq_idx=t)
            if t not in steps:
                continue
            M_next = layer.M.clone()
            x, h = activities[0], activities[1]
            above, below = clipped_masks(layer)
            unclipped = ~(above | below)
            mu = readable_mode_rate(h, x, W, unclipped, lam, eta)
            for name, probe in linearization_probes(x, W, unclipped, generator).items():
                layer.M = M_prev + epsilon * probe
                model.network_step(inputs[:, t], seq_idx=t)
                response = (layer.M - M_next) / epsilon
                expected = {"random": linear_response(probe, h, x, W, unclipped, lam, eta),
                            "readable": mu.unsqueeze(-1) * probe,
                            "unreadable": lam * probe}[name]
                probed = probe.norm(dim=-1) > 0
                errors[name].append((response - expected).norm(dim=-1)[probed].cpu().numpy())
            layer.M = M_next
    return {name: {"median": float(np.median(np.concatenate(values))),
                   "max": float(np.max(np.concatenate(values)))}
            for name, values in errors.items()}


# ─── Measurement ─────────────────────────────────────────────────────────────

def measure(model, inputs, chunk=FORWARD_CHUNK):
    """Step the network through `inputs` (trials, time, channels) in chunks of
    trials and record at every step: "mu" (trials, time, post), mu_i on the
    unclipped synapses (NaN for a row with every synapse clipped); "mu_all",
    mu_i summed over every synapse; "above" and "below" (trials, time), the
    shares of synapses clipped to the upper and the lower bound; and
    "rows_clipped" (trials, time), the share of rows with every synapse
    clipped. Also returns the network's outputs (trials, time, outputs)."""
    layer = plastic_layer(model)
    lam, eta = layer_rates(layer)
    W = layer.W.detach()
    n_trials, n_time = inputs.shape[:2]
    n_post = W.shape[0]
    record = {"mu": np.zeros((n_trials, n_time, n_post), dtype=np.float32),
              "mu_all": np.zeros((n_trials, n_time, n_post), dtype=np.float32),
              "above": np.zeros((n_trials, n_time), dtype=np.float32),
              "below": np.zeros((n_trials, n_time), dtype=np.float32),
              "rows_clipped": np.zeros((n_trials, n_time), dtype=np.float32)}
    outputs = []
    with torch.no_grad():
        for start in range(0, n_trials, chunk):
            batch = inputs[start:start + chunk]
            rows = slice(start, start + len(batch))
            model.reset_state(B=len(batch))
            batch_outputs = []
            for t in range(n_time):
                step_output, activities, _ = model.network_step(batch[:, t], seq_idx=t)
                x, h = activities[0], activities[1]
                above, below = clipped_masks(layer)
                unclipped = ~(above | below)
                mu = readable_mode_rate(h, x, W, unclipped, lam, eta)
                no_mode = ~unclipped.any(-1)
                record["mu"][rows, t] = mu.masked_fill(no_mode, float("nan")).cpu().numpy()
                record["mu_all"][rows, t] = readable_mode_rate(h, x, W, None, lam, eta).cpu().numpy()
                record["above"][rows, t] = above.float().mean((1, 2)).cpu().numpy()
                record["below"][rows, t] = below.float().mean((1, 2)).cpu().numpy()
                record["rows_clipped"][rows, t] = no_mode.float().mean(-1).cpu().numpy()
                batch_outputs.append(step_output)
            outputs.append(torch.stack(batch_outputs, dim=1).cpu())
    return record, torch.cat(outputs)


def _single_step(value):
    """One step index from an epoch boundary that every trial of a task shares."""
    values = np.unique(np.asarray(value))
    if values.size != 1:
        raise ValueError(f"Trials of one task disagree on an epoch boundary: {values}.")
    return int(values[0])


def epoch_bounds(trial):
    """{epoch: (start, end)} steps of one task's trials, in EPOCH_ORDER, with a
    missing start or end replaced by the trial's first or last step."""
    n_time = _single_step(trial.tdim)
    return {epoch: (0 if start is None else _single_step(start),
                    n_time if end is None else _single_step(end))
            for epoch in EPOCH_ORDER if epoch in trial.epochs
            for start, end in [trial.epochs[epoch]]}


def readable_tau(mu, dt_ms):
    """Time constant of the readable mode, dt / (1 - |mu|), the convention
    that gives lam its m_time_scale; infinite for |mu| >= 1, where the mode
    does not decay. A negative mu flips the mode's sign every step, so its
    magnitude decays by |mu|."""
    magnitude = np.abs(np.asarray(mu, dtype=float))
    tau = np.full(magnitude.shape, np.inf)
    decaying = magnitude < 1.0
    tau[decaying] = dt_ms / (1.0 - magnitude[decaying])
    return tau


def mu_stats(values, lam, dt_ms):
    """Quantiles of mu_i over `values` (NaN left out) and of |mu_i|, which
    order the readable mode's time constant (`readable_tau`); the shares of
    mu above 1, below -1 and below lam; and the median time constant."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    return {"n": int(values.size),
            "quantiles": {q: float(np.percentile(values, q)) for q in QUANTILES},
            "abs_quantiles": {q: float(np.percentile(np.abs(values), q)) for q in QUANTILES},
            "share_above_one": float(np.mean(values > 1.0)),
            "share_below_minus_one": float(np.mean(values < -1.0)),
            "share_below_lam": float(np.mean(values < lam)),
            "median_tau_ms": float(np.median(readable_tau(values, dt_ms)))}


def _period_type(epoch):
    return next(name for name, epochs in PERIOD_TYPES.items() if epoch in epochs)


def summarize(record, rule_indices, rules, epochs, lam, dt_ms):
    """Per task and epoch: `mu_stats`, the median of mu_i over every synapse
    and the mean shares of synapses at each bound and of rows with every
    synapse clipped. Per period type: `mu_stats` and a histogram of mu_i
    pooled over the tasks, on bins shared by the types. Per task: the mean
    saturation time course over its trials, up to its last step."""
    by_task, pooled, saturation = {}, {name: [] for name in PERIOD_TYPES}, {}
    for rule_index, rule in enumerate(rules):
        trials = rule_indices == rule_index
        by_task[rule] = {}
        for epoch, (start, end) in epochs[rule].items():
            mu = record["mu"][trials, start:end]
            by_task[rule][epoch] = {
                **mu_stats(mu, lam, dt_ms),
                "mu_all_median": float(np.median(record["mu_all"][trials, start:end])),
                "share_upper": float(record["above"][trials, start:end].mean()),
                "share_lower": float(record["below"][trials, start:end].mean()),
                "share_rows_clipped": float(record["rows_clipped"][trials, start:end].mean()),
            }
            pooled[_period_type(epoch)].append(mu[np.isfinite(mu)])
        n_time = max(end for _, end in epochs[rule].values())
        saturation[rule] = {side: record[side][trials, :n_time].mean(0)
                            for side in ("above", "below")}
    pooled = {name: np.concatenate(values) for name, values in pooled.items() if values}
    every = np.concatenate(list(pooled.values()))
    edges = np.linspace(*np.percentile(every, (0.5, 99.5)), HISTOGRAM_BINS + 1)
    by_type = {name: {**mu_stats(values, lam, dt_ms),
                      "histogram": np.histogram(values, bins=edges)[0]}
               for name, values in pooled.items()}
    return by_task, by_type, edges, saturation


# ─── Figure and printout ─────────────────────────────────────────────────────

def _tau_box(stats, dt_ms, cap):
    """ax.bxp statistics of the readable mode's time constant, from the
    quantiles of |mu|, which `readable_tau` maps monotonically; a time constant
    above `cap` (|mu| >= 1: no decay) is drawn at `cap`."""
    q = stats["abs_quantiles"]
    keys = (("whislo", 5), ("q1", 25), ("med", 50), ("q3", 75), ("whishi", 95))
    return {key: float(min(cap, readable_tau(q[percentile], dt_ms))) for key, percentile in keys}


def plot(payload, out_path):
    """mu_i per period type (top), each task's saturation time course and the
    readable mode's time constant per period type against lam's (bottom)."""
    lam, dt_ms = payload["lam"], payload["dt_ms"]
    by_type, edges = payload["by_period_type"], payload["histogram_edges"]
    fig = plt.figure(figsize=(13, 7.5))
    grid = fig.add_gridspec(2, len(PERIOD_TYPES), hspace=0.45, wspace=0.3)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for column, name in enumerate(PERIOD_TYPES):
        ax = fig.add_subplot(grid[0, column])
        if name not in by_type:
            ax.set_axis_off()
            continue
        stats = by_type[name]
        ax.fill_between(centers, stats["histogram"], step="mid", color="0.55", linewidth=0)
        ax.axvline(lam, color="tab:blue", linestyle="--", linewidth=1, label=f"λ = {lam:.3f}")
        ax.axvline(1.0, color="black", linewidth=1, label="1")
        ax.text(0.03, 0.97, f"μ > 1: {stats['share_above_one']:.1%}\n"
                            f"μ < -1: {stats['share_below_minus_one']:.1%}\n"
                            f"μ < λ: {stats['share_below_lam']:.1%}\n"
                            f"median τ: {stats['median_tau_ms']:.0f} ms",
                transform=ax.transAxes, va="top", fontsize=8)
        # Log counts: the bulk sits in a few bins just below lam, the tails
        # (fast forgetting, mu > 1) are orders of magnitude rarer.
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("μ of the readable mode", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if column == 0:
            ax.set_ylabel("post neurons x steps x trials", fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="center left")

    ax = fig.add_subplot(grid[1, :2])
    for index, (rule, course) in enumerate(payload["saturation"].items()):
        share = course["above"] + course["below"]
        ax.plot(np.arange(share.size) * dt_ms, 100 * share, linewidth=1,
                color=multiple_task_analysis.c_vals[index],
                label=diag._TASK_DISPLAY.get(rule, rule))
    ax.set_xlabel("Time in trial (ms)", fontsize=9)
    ax.set_ylabel("Synapses at a bound (%)", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 2:])
    tau_lam = dt_ms / (1.0 - lam)
    cap = 10 * tau_lam
    names = [name for name in PERIOD_TYPES if name in by_type]
    ax.bxp([_tau_box(by_type[name], dt_ms, cap) for name in names], showfliers=False,
           medianprops={"color": "black"})
    ax.axhline(tau_lam, color="tab:blue", linestyle="--", linewidth=1,
               label=f"λ alone: {tau_lam:.0f} ms")
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(f"Readable mode τ = Δt / (1 - |μ|) (ms)\n(capped at {cap / 1000:.0f} s)",
                  fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{payload['name']}: λ = {lam:.4f}, η = {payload['eta']:.3g}, "
                 f"accuracy {payload['accuracy']:.2f}", fontsize=11)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _stats_text(stats):
    q = stats["quantiles"]
    return (f"mu median {q[50]:.3f} (5-95%: {q[5]:.3f}-{q[95]:.3f}); "
            f"mu > 1 {stats['share_above_one']:.1%}, mu < -1 {stats['share_below_minus_one']:.1%}, "
            f"mu < lam {stats['share_below_lam']:.1%}; "
            f"median tau {stats['median_tau_ms']:.0f} ms")


def print_result(payload):
    """The check, the pooled period types and one line per task and epoch."""
    print(f"\n=== {payload['name']}: lam = {payload['lam']:.4f}, eta = {payload['eta']:.4g}, "
          f"accuracy {payload['accuracy']:.3f} ===")
    print("Finite-difference check (error per unit perturbation):")
    for name, error in payload["check"].items():
        print(f"  {name:<10s} median {error['median']:.2e}, max {error['max']:.2e}")
    print("Pooled by period type:")
    for name, stats in payload["by_period_type"].items():
        print(f"  {name:<9s} {_stats_text(stats)}")
    print("Per task and epoch:")
    for rule, by_epoch in payload["by_task"].items():
        for epoch, stats in by_epoch.items():
            print(f"  {diag._TASK_DISPLAY.get(rule, rule):<22s} {EPOCH_DISPLAY[epoch]:<10s} "
                  f"{_stats_text(stats)}; mu over all synapses {stats['mu_all_median']:.3f}; "
                  f"at bounds +{stats['share_upper']:.1%} / -{stats['share_lower']:.1%}, "
                  f"rows all clipped {stats['share_rows_clipped']:.1%}")


# ─── One run ─────────────────────────────────────────────────────────────────

def _evaluation_trials(seed, feature, trial_seed=EVALUATION_SEED):
    """The run's config and its evaluation trials, drawn with `trial_seed`:
    (task_params_c, inputs, outputs, masks, trials per task, task index per
    trial)."""
    with (CHECKPOINT_DIR / f"param_{run_name(seed, feature)}_param.json").open() as stream:
        raw = json.load(stream)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    task_params_c, _, _ = mpn_tasks.convert_and_init_multitask_params(
        (raw["task_params"], raw["train_params"], raw["net_params"]))
    task_params_c["hp"]["batch_size_train"] = TRIALS_PER_TASK
    task_params_c["hp"]["rng"] = np.random.RandomState(trial_seed)
    (inputs, outputs, masks), (_, trials, rule_indices) = mpn_tasks.generate_trials_wrap(
        task_params_c, TRIALS_PER_TASK, rules=task_params_c["rules"], mode_input="random",
        device="cpu", verbose=False, long_all=False)
    return task_params_c, inputs, outputs, masks, trials, np.asarray(rule_indices)


def analyze_run(seed, feature, untrained_init_seed=None, device=None):
    """Check the derivation on one network, measure mu_i and the saturation on
    its evaluation trials, and save the result and the figure."""
    start_time = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_DIR / f"savednet_{run_name(seed, feature)}.pt",
                            map_location="cpu", weights_only=False)
    if checkpoint["net_params"].get("activation") != "tanh":
        raise ValueError("The derivation uses tanh' = 1 - h^2.")
    model, _ = build_network(checkpoint, untrained_init_seed)
    model.to(device).eval()
    layer = plastic_layer(model)
    lam, eta = layer_rates(layer)

    task_params_c, inputs, outputs, masks, trials, rule_indices = _evaluation_trials(seed, feature)
    rules = list(task_params_c["rules"])
    dt_ms = float(task_params_c["hp"]["dt"])
    inputs = inputs.to(device)
    first_of_each_task = [int(np.flatnonzero(rule_indices == index)[0])
                          for index in range(len(rules))]
    check = check_linearization(model, inputs[first_of_each_task])
    record, net_out = measure(model, inputs)
    accuracy, _ = model.compute_acc(net_out.to(device), outputs.to(device), masks.to(device),
                                    inputs, isvalid=True, mode=model.acc_measure)
    epochs = {rule: epoch_bounds(trial) for rule, trial in zip(rules, trials)}
    by_task, by_type, edges, saturation = summarize(record, rule_indices, rules, epochs,
                                                    lam, dt_ms)

    stem = result_stem(seed, feature, untrained_init_seed)
    payload = {
        "version": RESULT_VERSION,
        "name": stem.removeprefix("modulation_feedback_"),
        "seed": seed,
        "feature": feature,
        "untrained_init_seed": untrained_init_seed,
        "lam": lam,
        "eta": eta,
        "dt_ms": dt_ms,
        "accuracy": float(accuracy),
        "n_trials_per_task": TRIALS_PER_TASK,
        "check": check,
        "rules": rules,
        "epochs": epochs,
        "by_task": by_task,
        "by_period_type": by_type,
        "histogram_edges": edges,
        "saturation": saturation,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{stem}.pkl", "wb") as stream:
        pickle.dump(payload, stream)
    plot(payload, OUT_DIR / f"{stem}.png")
    print_result(payload)
    print(f"Saved: {OUT_DIR / f'{stem}.pkl'} ({time.time() - start_time:.0f}s)")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Feedback (mu of the readable mode) and saturation of the plastic "
                    "layer's modulation on the evaluation trials of one or every run.")
    parser.add_argument("--feature", default="L21e4",
                        help="Feature of the runs (default 'L21e4').")
    parser.add_argument("--seed", required=True,
                        help="Seed of the run, or 'all' for every run of the feature.")
    parser.add_argument("--untrained-init-seed", type=int, default=None,
                        help="Analyze an untrained network of the run's config, drawn "
                             "with this init seed, instead of the trained one.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seeds = discover_seeds(args.feature) if args.seed == "all" else [int(args.seed)]
    if not seeds:
        print(f"No {args.feature} checkpoints in {CHECKPOINT_DIR}/.")
    for seed in seeds:
        analyze_run(seed, args.feature, args.untrained_init_seed)


if __name__ == "__main__":
    with tee_output("modulation_feedback"):
        main()
