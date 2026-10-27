"""
Remove the rule cue and see whether the rows of the clustering features
regroup.

The rows of the clustered heatmaps (multiple_task_analysis.py) are task x
period conditions. Input and hidden rows group by period, synaptic
modulation rows by task (sibling_row_sharing.py), in trained and untrained
networks alike. One proposed reason: the rule cue is on for the whole trial,
so it is a constant that adds no variance to the input and hidden features,
while M integrates it into a task-specific drift. This script tests the
cue's part directly. It runs one network on the evaluation trials of
modulation_feedback.py twice, once as they are and once with every rule-cue
channel set to zero (the cue and its input noise), and computes the input,
hidden and modulation features of both as the clustering does: per task and
period, the variance over stimulus conditions and time of the
stimulus-conditioned mean (helper.task_variance_period_numpy), each column
then divided by its maximum over the rows (the "_normalized" features of the
heatmaps). Rows follow the clustering's order (period-major) and windows; an
epoch without an end (go1) runs to the last step of the padded batch, as in
the clustering.

The readout needs no clustering. It uses the clustering's metric, the
Euclidean distance between two rows, divided by its mean over every two rows
(below 1: closer than a typical pair). Per representation and condition it
is the mean of that relative distance over three kinds of pair: one task in
two periods ("same task"), one period in two tasks ("same period") and the
rest ("other"). Row correlation was tried first and fails for input and
hidden: their period grouping is mostly row magnitude (no input feature
varies in the memory periods, so those rows are zero), which a correlation
discards. How much the cue changes the features themselves is the
correlation of each row with the same row of the other condition, and the
ratio of the summed unnormalized features.

mode="random" draws each task's stimulus modalities once per batch, so all
trials of a task share them. The script records the modality of each task's
first stimulus, and --trial-seed draws another batch, with other draws.

Writes multiple_tasks_diagnosis/rule_cue_ablation_{name}[_trials{s}].{pkl,png},
with name the run's aname, plus _init{k} for an untrained network, and the
_trials suffix for a trial seed other than the default.

Usage:
    python multiple_task/rule_cue_ablation.py --feature L21e4 --seed 749 \\
        --untrained-init-seed 0
    python multiple_task/rule_cue_ablation.py --feature L21e4 --seed 749 \\
        --untrained-init-seed 0 --trial-seed 3
    python multiple_task/rule_cue_ablation.py --feature L21e4 --seed 749
"""
import argparse
import pickle
import time

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are saved, never shown
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper
from network_loading import build_network
from run_logging import tee_output
import multiple_task_analysis
import multiple_task_diagnosis as diag
import modulation_feedback as feedback

OUT_DIR = diag.OUT_DIR
RESULT_VERSION = 2
# The clustering's periods, in its row order (multiple_task_analysis.py).
PHASES = ("stim1", "stim2", "delay1", "delay2", "go1")
# Representation -> key of the network's track_states record.
REPRESENTATIONS = {"input": "input1", "hidden": "hidden1", "modulation": "M1"}
CONDITIONS = {"cue": "Rule cue", "no_cue": "No rule cue"}
PAIR_KINDS = {"same_task": "Same task", "same_period": "Same period", "other": "Other"}
FORWARD_CHUNK = 25
# A ring counts as the stimulus modality when its mean |input| is this many
# times the other ring's; otherwise the stimulus is in "both".
MODALITY_RATIO = 5.0


def result_stem(seed, feature, untrained_init_seed=None, trial_seed=feedback.EVALUATION_SEED):
    """rule_cue_ablation_ + the run's aname, plus _init{k} for an untrained
    network and _trials{s} for a trial seed other than the default."""
    name, _ = multiple_task_analysis.output_names(
        seed, "everything", f"{feature}{feedback.RUN_SUFFIX}", untrained_init_seed)
    suffix = "" if trial_seed == feedback.EVALUATION_SEED else f"_trials{trial_seed}"
    return f"rule_cue_ablation_{name}{suffix}"


# ─── The rule cue and the stimulus modality ──────────────────────────────────

def rule_channels(task_params_c, n_channels):
    """The input channels of the rule cue, one per rule after the fixation and
    stimulus channels."""
    if task_params_c["randomize_inputs"]:
        raise ValueError("Randomized inputs mix the rule cue into every channel.")
    start, n_rule = task_params_c["hp"]["rule_start"], task_params_c["hp"]["n_rule"]
    if start + n_rule != n_channels:
        raise ValueError(f"Rule channels {start}..{start + n_rule - 1} do not end the "
                         f"{n_channels} input channels.")
    return slice(start, start + n_rule)


def check_rule_cue(inputs, rule_indices, channels):
    """Confirm that every trial's strongest rule channel on its first step is
    its own rule."""
    cued = inputs[:, 0, channels].argmax(-1).cpu().numpy()
    if not np.array_equal(cued, rule_indices):
        raise ValueError("The rule channels do not carry each trial's own rule.")


def without_rule_cue(inputs, channels):
    """A copy of `inputs` (trials, time, channels) with the rule channels zeroed."""
    ablated = inputs.clone()
    ablated[..., channels] = 0.0
    return ablated


def ring_channels(hp):
    """The input channels of the two stimulus rings, which lie between the
    fixation channels and the rule channels."""
    n_fixate = 2 if hp["fixate_off"] else 1
    width = (hp["rule_start"] - n_fixate) // hp["num_ring"]
    return [slice(n_fixate + ring * width, n_fixate + (ring + 1) * width)
            for ring in range(hp["num_ring"])]


def first_stimulus_modality(inputs, rule_indices, trials, rings):
    """Per rule index, the modality of its trials' first stimulus ("1", "2" or
    "both"), from the mean |input| of each ring over the stim1 epoch, or go1
    for a task whose stimulus comes with the response (ReactPro/Anti)."""
    modality = {}
    for rule_index, trial in enumerate(trials):
        epoch = "stim1" if "stim1" in trial.epochs else "go1"
        start, end = (_boundary(value) for value in trial.epochs[epoch])
        window = inputs[rule_indices == rule_index, start:end]
        energy = [float(window[..., ring].abs().mean()) for ring in rings]
        if energy[0] > MODALITY_RATIO * energy[1]:
            modality[rule_index] = "1"
        elif energy[1] > MODALITY_RATIO * energy[0]:
            modality[rule_index] = "2"
        else:
            modality[rule_index] = "both"
    return modality


# ─── Features ────────────────────────────────────────────────────────────────

def _boundary(value):
    return None if value is None else feedback._single_step(value)


def row_windows(trials):
    """(rule index, phase, (start, end)) of every row, period-major as in the
    clustering; `trials` holds one trial object per rule, in rule order. end
    is None for an epoch that runs to the end of the batch."""
    return [(rule_index, phase, tuple(_boundary(value) for value in trial.epochs[phase]))
            for phase in PHASES
            for rule_index, trial in enumerate(trials) if phase in trial.epochs]


def task_features(model, inputs, stim_labels, windows):
    """{representation: (len(windows), columns)} variance features of one
    task's trials, one row per (start, end) window."""
    _, states = multiple_task_analysis.record_tracked_states(model, inputs, FORWARD_CHUNK)
    features = {}
    for name, key in REPRESENTATIONS.items():
        values = states.pop(key)
        values = values.reshape(*values.shape[:2], -1)
        features[name] = np.stack([helper.task_variance_period_numpy(values[:, start:end],
                                                                     stim_labels)
                                   for start, end in windows])
        del values
    return features


def condition_features(model, inputs, rule_indices, stim_labels, windows):
    """{representation: (rows, columns)} features of every row of `windows`,
    one task's trials at a time so that only one task's M trajectory is held."""
    rows = {name: [None] * len(windows) for name in REPRESENTATIONS}
    for rule_index in np.unique(rule_indices):
        own_rows = [row for row, window in enumerate(windows) if window[0] == rule_index]
        if not own_rows:
            continue
        trials = np.flatnonzero(rule_indices == rule_index)
        features = task_features(model, inputs[trials], stim_labels[trials],
                                 [windows[row][2] for row in own_rows])
        for name, values in features.items():
            for row, value in zip(own_rows, values):
                rows[name][row] = value
    return {name: np.stack(values) for name, values in rows.items()}


# ─── Readout ─────────────────────────────────────────────────────────────────

def normalize_columns(features):
    """Each column divided by its maximum over the rows, 0 where that maximum
    is not positive (the clustering's "_normalized" features)."""
    peak = features.max(axis=0)
    return np.where(peak > 0, features / np.where(peak > 0, peak, 1.0), 0.0)


def relative_row_distance(normalized):
    """Euclidean distance between every two rows (the clustering's metric),
    divided by its mean over every two rows."""
    distance = squareform(pdist(normalized))
    return distance / distance[np.triu_indices_from(distance, k=1)].mean()


def pair_means(matrix, row_tasks, row_phases):
    """{kind: {"mean", "n"}}: the mean of a row-by-row `matrix` over the pairs
    of rows of each kind of PAIR_KINDS, and the number of pairs."""
    tasks, phases = np.asarray(row_tasks), np.asarray(row_phases)
    upper = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    same_task = tasks[:, None] == tasks[None, :]
    same_phase = phases[:, None] == phases[None, :]
    masks = {"same_task": upper & same_task & ~same_phase,
             "same_period": upper & same_phase & ~same_task,
             "other": upper & ~same_task & ~same_phase}
    return {kind: {"mean": float(np.nanmean(matrix[mask])), "n": int(mask.sum())}
            for kind, mask in masks.items()}


def matched_row_correlation(first, second):
    """Correlation of each row of `first` with the same row of `second`, over
    the columns that are not zero in every row of both."""
    kept = (first.max(axis=0) > 0) | (second.max(axis=0) > 0)
    a = first[:, kept] - first[:, kept].mean(axis=1, keepdims=True)
    b = second[:, kept] - second[:, kept].mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def summarize(features, row_tasks, row_phases):
    """Relative row distances and their pair means per condition and
    representation, and per representation the matched-row correlation
    between the conditions and the ratio of the summed unnormalized features
    (no cue / cue)."""
    normalized = {condition: {name: normalize_columns(values) for name, values in by_name.items()}
                  for condition, by_name in features.items()}
    distance = {condition: {name: relative_row_distance(values)
                            for name, values in by_name.items()}
                for condition, by_name in normalized.items()}
    means = {condition: {name: pair_means(values, row_tasks, row_phases)
                         for name, values in by_name.items()}
             for condition, by_name in distance.items()}
    matched = {name: matched_row_correlation(normalized["cue"][name], normalized["no_cue"][name])
               for name in REPRESENTATIONS}
    total_ratio = {name: float(features["no_cue"][name].sum() / features["cue"][name].sum())
                   for name in REPRESENTATIONS}
    return distance, means, matched, total_ratio


# ─── Figure and printout ─────────────────────────────────────────────────────

def _task_label(payload, rule_index):
    rule = payload["rules"][rule_index]
    return f"{diag._TASK_DISPLAY.get(rule, rule)} ({payload['modality'][rule_index]})"


def plot(payload, out_path):
    """Per representation (rows): the relative row distance with and without
    the cue, rows in task-major order with the tasks and the modality of
    their first stimulus marked, and the pair means."""
    row_tasks = np.asarray(payload["row_tasks"])
    order = sorted(range(len(row_tasks)),
                   key=lambda row: (row_tasks[row], PHASES.index(payload["row_phases"][row])))
    sorted_tasks = row_tasks[order]
    edges = np.flatnonzero(np.diff(sorted_tasks)) + 0.5
    present = np.unique(sorted_tasks)
    centers = [np.mean(np.flatnonzero(sorted_tasks == task)) for task in present]
    labels = [_task_label(payload, task) for task in present]

    fig, axes = plt.subplots(len(REPRESENTATIONS), 3, figsize=(13, 4 * len(REPRESENTATIONS)),
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for row_axes, name in zip(axes, REPRESENTATIONS):
        for ax, (condition, display) in zip(row_axes[:2], CONDITIONS.items()):
            image = ax.imshow(payload["distance"][condition][name][np.ix_(order, order)],
                              cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
            for edge in edges:
                ax.axhline(edge, color="white", linewidth=0.4)
                ax.axvline(edge, color="white", linewidth=0.4)
            ax.set_yticks(centers)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_xticks([])
            ax.set_title(f"{name.capitalize()}: {display}", fontsize=10)
        fig.colorbar(image, ax=row_axes[1], fraction=0.046, pad=0.02,
                     label="Distance / mean distance")

        ax = row_axes[2]
        positions = np.arange(len(PAIR_KINDS))
        width = 0.38
        for index, condition in enumerate(CONDITIONS):
            means = payload["pair_means"][condition][name]
            ax.bar(positions + (index - 0.5) * width, [means[kind]["mean"] for kind in PAIR_KINDS],
                   width, color=multiple_task_analysis.c_vals[index], label=CONDITIONS[condition])
        ax.axhline(1, color="black", linewidth=0.6, linestyle="--")
        # Headroom above the bars (all near or below ~1.1) keeps the legend
        # clear of the bars and of the line at 1.
        ax.set_ylim(0, max(1.4, ax.get_ylim()[1]))
        ax.set_xticks(positions)
        ax.set_xticklabels(PAIR_KINDS.values(), fontsize=9)
        ax.set_ylabel("Distance / mean distance\n(lower = closer)", fontsize=9)
        ax.set_title(f"matched rows r = {np.median(payload['matched'][name]):.2f} (median); "
                     f"total x{payload['total_ratio'][name]:.2f}", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{payload['name']}: rule cue removed (λ = {payload['lam']:.3f}, "
                 f"η = {payload['eta']:.3g}, trial seed {payload['trial_seed']}); "
                 "(k) = modality of the first stimulus", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_result(payload):
    print(f"\n=== {payload['name']}: lam = {payload['lam']:.4f}, eta = {payload['eta']:.4g}, "
          f"trial seed {payload['trial_seed']} ===")
    print("First-stimulus modality: " + ", ".join(
        _task_label(payload, index) for index in range(len(payload["rules"]))))
    print("Mean distance / mean over every two rows (lower = closer):")
    for name in REPRESENTATIONS:
        matched = payload["matched"][name]
        print(f"{name}: matched-row r median {np.median(matched):.3f} "
              f"(min {np.min(matched):.3f}); summed features no cue / cue "
              f"{payload['total_ratio'][name]:.3f}")
        for condition, display in CONDITIONS.items():
            means = payload["pair_means"][condition][name]
            print(f"  {display:<12s} " + ", ".join(
                f"{PAIR_KINDS[kind].lower()} {means[kind]['mean']:.3f} (n={means[kind]['n']})"
                for kind in PAIR_KINDS))


# ─── One run ─────────────────────────────────────────────────────────────────

def analyze_run(seed, feature, untrained_init_seed=None, trial_seed=feedback.EVALUATION_SEED,
                device=None):
    """Compute the features of one network with and without the rule cue,
    compare them, and save the result and the figure."""
    start_time = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        feedback.CHECKPOINT_DIR / f"savednet_{feedback.run_name(seed, feature)}.pt",
        map_location="cpu", weights_only=False)
    model, _ = build_network(checkpoint, untrained_init_seed)
    model.to(device).eval()
    lam, eta = feedback.layer_rates(feedback.plastic_layer(model))

    task_params_c, inputs, _, _, trials, rule_indices = feedback._evaluation_trials(
        seed, feature, trial_seed)
    rules = list(task_params_c["rules"])
    channels = rule_channels(task_params_c, inputs.shape[-1])
    check_rule_cue(inputs, rule_indices, channels)
    modality = first_stimulus_modality(inputs, rule_indices, trials,
                                       ring_channels(task_params_c["hp"]))
    # One label per trial, rule-major like the batch (the clustering's labels_stim1).
    stim_labels = np.concatenate([np.asarray(trial.meta["stim1"]).ravel() for trial in trials])
    if stim_labels.size != rule_indices.size:
        raise ValueError("Stimulus labels and trials do not line up.")
    windows = row_windows(trials)
    inputs = inputs.to(device)
    features = {
        "cue": condition_features(model, inputs, rule_indices, stim_labels, windows),
        "no_cue": condition_features(model, without_rule_cue(inputs, channels), rule_indices,
                                     stim_labels, windows),
    }
    row_tasks = [window[0] for window in windows]
    row_phases = [window[1] for window in windows]
    distance, means, matched, total_ratio = summarize(features, row_tasks, row_phases)

    stem = result_stem(seed, feature, untrained_init_seed, trial_seed)
    payload = {
        "version": RESULT_VERSION,
        "name": stem.removeprefix("rule_cue_ablation_"),
        "seed": seed,
        "feature": feature,
        "untrained_init_seed": untrained_init_seed,
        "trial_seed": trial_seed,
        "lam": lam,
        "eta": eta,
        "n_trials_per_task": feedback.TRIALS_PER_TASK,
        "rules": rules,
        "modality": modality,
        "rule_channels": (channels.start, channels.stop),
        "row_names": [f"{rules[task]}-{phase}" for task, phase in zip(row_tasks, row_phases)],
        "row_tasks": row_tasks,
        "row_phases": row_phases,
        "windows": windows,
        "features": {condition: {key: values.astype(np.float32) for key, values in by_name.items()}
                     for condition, by_name in features.items()},
        "distance": distance,
        "pair_means": means,
        "matched": matched,
        "total_ratio": total_ratio,
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
        description="Compare the clustering features of one network with and without "
                    "the rule cue.")
    parser.add_argument("--feature", default="L21e4",
                        help="Feature of the run (default 'L21e4').")
    parser.add_argument("--seed", type=int, required=True, help="Seed of the run.")
    parser.add_argument("--untrained-init-seed", type=int, default=None,
                        help="Analyze an untrained network of the run's config, drawn "
                             "with this init seed, instead of the trained one.")
    parser.add_argument("--trial-seed", type=int, default=feedback.EVALUATION_SEED,
                        help="Seed of the evaluation trials (default "
                             f"{feedback.EVALUATION_SEED}); another seed draws other "
                             "stimulus modalities.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    analyze_run(args.seed, args.feature, args.untrained_init_seed, args.trial_seed)


if __name__ == "__main__":
    with tee_output("rule_cue_ablation"):
        main()
