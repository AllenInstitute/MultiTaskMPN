"""Analyze pretraining checkpoints, cue/M interventions, and rule-vector probes.

Run from the repository root:
    python pretrain/pretraining_post.py
    python pretrain/pretraining_post.py --accuracy
    python pretrain/pretraining_post.py --memory-pca
    python pretrain/pretraining_post.py --rule-cue
    python pretrain/pretraining_post.py --m-intervention
    python pretrain/pretraining_post.py --rule-vector-intervention
    python pretrain/pretraining_post.py --rule-vector-magnitude-sweep
    python pretrain/pretraining_post.py --backbone-probe
    python pretrain/pretraining_post.py --pathway-gain
    python pretrain/pretraining_post.py --plot-only

Reads checkpoints from pretraining/, saves and reloads analysis data in
pretraining_analysis/, saves pooled figures in pretrain/fig/, and saves
per-checkpoint figures and seed-filtered accuracy plots in pretrain/fig_seed/.

Default (no experiment flag): run accuracy, memory PCA, rule-cue, and
M-intervention sequentially. An explicit experiment flag runs only that
experiment. Rule-vector interventions, magnitude sweeps, and backbone probes
require their explicit flags. --ruleset and --seed also filter default runs.
For checkpoint-batch analyses, --total-seed K randomly selects K matching
checkpoint seeds per motif; without it, every matching checkpoint is used.
Batch checkpoint selection is reproducible from --test-seed. Memory-PCA
checkpoint selection is independent of --test-seed; use --seed to fix it.
Default runs continue after an experiment fails and report failures at the end.
--accuracy generates fresh trials, saves accuracy JSON, then plots. Accuracy uses
the model's angle-based response-timepoint metric, not trial success counts.
--plot-only reads existing accuracy JSON without loading models. Accuracy bars show seed
means, error bars population SD, and dots individual seeds. Reads per-run
accuracy JSON files, excluding summary reports to avoid duplicate counts.
Use --memory-pca to project stimulus and response trajectories into the MemoryAnti memory
subspace, for hidden and effective modulation (W*M), in both motif groups. Saves
PNG figures only, selecting a random checkpoint per group unless filtered.
Memory PCA uses 256 trials per task, batches of 8, and automatic device selection;
--n-trials, --batch-size, and --device apply to the other experiments.
Use --rule-cue to compare intact cues, removal after context, stimulus,
or memory, and a cue present only during the response period, on paired
fresh trials. This tests cue dependence, not causal storage in M;
no plastic state is intervened on.
Use --m-intervention to test the functional effect of the cue-attributable
plastic trace (M with the cue minus M without it, same trial), by scaling
or transplanting it at response onset. Donors are selected by cyclic trial shift,
not by stimulus direction; the direction-mismatch fraction is reported.
Trace conditions have the response cue off; the intact replay retains it.
Overrides are clamped to modulation bounds, and subsequent M updates proceed
normally. These interventions do not by themselves establish that the trace
encodes task identity or separates additively from stimulus information.
--m-intervention also saves a trace-vs-M figure comparing the cue trace with
the plastic state at response onset (mean-|entry| L1 magnitudes/distance and
per-trial Pearson correlations, per motif group).
Use --rule-vector-intervention to replace the learned MemoryAnti rule-input
vector with its projection into the two-rule pretraining span or its orthogonal
residual. Raw and norm-matched versions separate direction from input strength;
matched-norm random vectors provide a control. This analysis uses final
checkpoints only, does not retrain the model, and saves separate accuracy and
vector-norm figures.
Use --rule-vector-magnitude-sweep to hold rule-vector direction fixed while
scanning its L2 norm relative to the learned MemoryAnti vector. It compares the
pretraining-span direction, orthogonal direction, and random directions using
final checkpoints only.
Use --backbone-probe for zero-shot MemoryAnti probes of the frozen backbone.
Stage 2 trains only the last input column, so replacing that column turns the
final checkpoint back into the end-of-stage-1 backbone under a counterfactual
rule input. Two untrained probes are evaluated: random rule vectors (the
negative control, reporting training-style loss and accuracy) and the
pretraining-span combination grid
a*v_pre0 + b*v_pre1 over an (a, b) coefficient grid from -4 to 4 on each axis
with spacing 0.25, plus single cues, the cue sum, and the learned vector's
least-squares projection into the span. A high-accuracy grid cell identifies
a rule-span solution on the evaluated trials without training a new vector.
The grid maximum carries selection bias over the grid evaluations; named
points are not selected by maximizing accuracy over the grid.
A second, norm-matched span grid rescales nonzero combinations and the learned
vector to the mean of that checkpoint's two pretrained cue norms. The origin
retains the zero-vector accuracy. This controls input strength within each
checkpoint; the target norm can differ across checkpoints. Accuracy is
constant along positive coefficient rays, excluding the origin, by construction.
Use --pathway-gain for a linear probe of the stimulus-to-readout pathway: per
trial, the embedded stimulus direction is pushed through W_eff = W + W*M and
projected onto the readout loading of the pro direction (the negated anti
target), so positive gain means the pathway drives pro and negative means
anti. M is read from rollouts with no cue, the cue trace at response onset,
and trial-end states with the response cue off or on. The probe ignores the
hidden tanh slope and bias paths, so the sign pattern across conditions is the
readout, not the magnitudes. For multiplicative M it also reports mean 1 + M
and bound-saturation fractions on each trial's top pathway synapses, split
into pro-driving and anti-driving contributions — under bounds 1 + M in
[0, 2], a negative summed gain can only arise from that re-weighting.
MemoryAnti cue conditions also report raw output magnitudes and angular errors
in the accuracy scoring window; these diagnostics do not redefine accuracy.
"""

import argparse
import copy
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "pretraining"
ANALYSIS_DIR = REPO_ROOT / "pretraining_analysis"
FIGURE_DIR = REPO_ROOT / "pretrain" / "fig"
SEED_FIGURE_DIR = REPO_ROOT / "pretrain" / "fig_seed"

GROUPS = {
    "fdanti_delaygo": ("Proper motif", ("fdanti", "delaygo", "delayanti")),
    "fdgo_delaygo": ("Improper motif", ("fdgo", "delaygo", "delayanti")),
}
COLORS = ("#3182ce", "#38a169", "#e53e3e")
# Matches two_task_analysis.py's c_vals[stimulus_index] trajectory colors.
STIMULUS_COLORS = ("#e53e3e", "#3182ce", "#38a169", "#805ad5",
                   "#dd6b20", "#319795", "#718096", "#d53f8c", "#d69e2e")

# Figure display names for the internal rule names. "Delay-" tasks keep the
# stimulus on (no memory demand); "Memory-" tasks turn it off. Note the trap:
# internal "delaygo"/"delayanti" are the MEMORY tasks. Filenames, JSON keys,
# and console logs keep the internal names; only figure text is converted.
RULE_DISPLAY_NAMES = {"fdgo": "DelayPro", "fdanti": "DelayAnti",
                      "delaygo": "MemoryPro", "delayanti": "MemoryAnti"}


def _display_rule(rule):
    """Figure display name for an internal rule name."""
    return RULE_DISPLAY_NAMES.get(rule, rule)


def _figure_path(output_dir, filename, *, seed_specific=False):
    """Separate default figure destinations while honoring custom directories."""
    output_dir = Path(output_dir)
    if seed_specific and output_dir.resolve() == FIGURE_DIR.resolve():
        output_dir = SEED_FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def discover_checkpoints(root, feature, hidden, ruleset=None, seed=None,
                         total_seed=None, selection_seed=0):
    pattern = re.compile(
        rf"savednet_({'|'.join(GROUPS)})_dmpn_seed(\d+)_"
        rf"\+hidden{hidden}\+{re.escape(feature)}\+batch128\+angle\.pt"
    )
    matches = []
    for path in sorted(root.glob("savednet_*.pt")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        run_ruleset, run_seed = match.group(1), int(match.group(2))
        if ruleset is not None and run_ruleset != ruleset:
            continue
        if seed is not None and run_seed != seed:
            continue
        matches.append((path, run_ruleset, run_seed))
    if total_seed is None:
        return matches

    rng = np.random.default_rng(selection_seed)
    selected = []
    active_rulesets = (ruleset,) if ruleset is not None else tuple(GROUPS)
    for group in active_rulesets:
        candidates = [item for item in matches if item[1] == group]
        if len(candidates) < total_seed:
            raise ValueError(
                f"Requested --total-seed {total_seed}, but only "
                f"{len(candidates)} matching {group} checkpoint(s) exist")
        indices = rng.choice(len(candidates), size=total_seed, replace=False)
        chosen = [candidates[int(index)] for index in indices]
        chosen.sort(key=lambda item: item[2])
        print(f"Selected {group} seeds: {[item[2] for item in chosen]}",
              flush=True)
        selected.extend(chosen)
    matches = selected
    return matches


def load_task_params(root, aname, stage):
    with np.load(root / f"output_{aname}_{stage}.npz", allow_pickle=True) as data:
        return copy.deepcopy(data["task_params"].item())


CUE_CONDITIONS = {
    "intact": "Intact",
    "off_after_context": "Off after\ncontext",
    "off_after_stimulus": "Off after\nstimulus",
    "off_after_memory": "Off after\nmemory",
    "on_only_response": "On only\nresponse",
}


def apply_rule_cue(inputs, epochs, rule_column, condition):
    """Return a copy with only the active task cue gated at per-trial boundaries.

    off_* conditions keep the cue up to a period boundary; on_only_response is
    the complement of off_after_memory (cue absent until the response onset).
    """
    import torch

    if condition not in CUE_CONDITIONS:
        raise ValueError(f"Unknown cue condition: {condition}")
    altered = inputs.clone()
    if condition == "intact":
        return altered
    if condition in ("off_after_memory", "on_only_response") and "delay1" not in epochs:
        return None
    if condition == "off_after_context":
        boundary = epochs["stim1"][0]
    elif condition == "off_after_stimulus":
        boundary = epochs["stim1"][1]
    else:
        boundary = epochs["delay1"][1]
    boundaries = torch.as_tensor(boundary, device=inputs.device)
    if boundaries.ndim == 0:
        boundaries = boundaries.expand(inputs.shape[0])
    if boundaries.shape != (inputs.shape[0],):
        raise ValueError("Epoch boundaries must be scalar or one per trial")
    if not torch.all((boundaries >= 0) & (boundaries <= inputs.shape[1])):
        raise ValueError("Epoch boundary outside sequence")
    times = torch.arange(inputs.shape[1], device=inputs.device)[None, :]
    keep = times < boundaries[:, None]
    if condition == "on_only_response":
        keep = ~keep
    channel = inputs.shape[-1] - 3 + rule_column
    altered[:, :, channel] *= keep
    return altered


ERROR_CATEGORIES = {
    "target_pct": ("Anti target", "#3182ce"),
    "low_amplitude_pct": ("Low amplitude", "#718096"),
    "pro_pct": ("Opposite (pro)", "#dd6b20"),
    "other_pct": ("Other direction", "#805ad5"),
}


def response_direction_diagnostics(output, targets, masks, n_directions=8,
                                   amplitude_threshold=0.15):
    """Partition MemoryAnti raw outputs into four mutually exclusive categories.

    Fractions count response timepoints, matching compute_acc's window. Low
    amplitude means vector norm < threshold (not compute_acc's component-wise
    rounding). First accept sufficiently large outputs within pi/n_directions
    of the anti target, then low-amplitude outputs, then the opposite (pro)
    sector within pi/n_directions of 180 deg, then all other directions.
    Other directions need not be random; angular statistics retain their structure.
    """
    import _bootstrap  # noqa: F401
    import helper

    output, targets, masks = (np.asarray(value) for value in (output, targets, masks))
    if output.shape != targets.shape or masks.shape != targets.shape or output.shape[-1] != 3:
        raise ValueError("Direction diagnostics require matching (batch, time, 3) arrays")
    if masks.dtype != np.float32:
        raise ValueError("Direction diagnostics require float32 cost masks")
    if n_directions < 4 or n_directions % 2 or amplitude_threshold <= 0:
        raise ValueError("Need an even number of directions >=4 and a positive amplitude threshold")
    selected = np.zeros(output.shape[:2], dtype=bool)
    windows = []
    for trial_index, mask in enumerate(masks):
        chunks = helper.find_zero_chunks(mask)
        if len(chunks) < 3:
            chunks.append([output.shape[1], output.shape[1]])
        if len(chunks) < 2:
            raise ValueError("Cannot identify response scoring window")
        start, end = chunks[-2][1] + 1, chunks[-1][0]
        start += int((end - start) / 4)
        if not np.all(mask[start:end, 1:3] > 0):
            raise ValueError("Response direction channels must both be scored")
        selected[trial_index, start:end] = True
        windows.append([start, end])
    if not selected.any():
        raise ValueError("No response timepoints for direction diagnostics")
    vectors = output[selected, 1:3]
    target_vectors = targets[selected, 1:3]
    if not np.isfinite(vectors).all() or not np.isfinite(target_vectors).all():
        raise ValueError("Non-finite response direction data")
    if np.any(np.linalg.norm(target_vectors, axis=1) <= 0):
        raise ValueError("MemoryAnti response target must have nonzero magnitude")
    magnitudes = np.linalg.norm(vectors, axis=1)
    low = magnitudes < amplitude_threshold
    target_angles = np.arctan2(target_vectors[:, 0], target_vectors[:, 1])
    output_angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    errors = np.angle(np.exp(1j * (output_angles - target_angles)))
    half_width = np.pi / n_directions
    target_match = (~low) & (np.abs(errors) <= half_width)
    pro = ~(target_match | low) & (np.abs(errors) >= np.pi - half_width)
    other = ~(target_match | low | pro)
    directional_errors = errors[~low]
    edges = np.linspace(-180, 180, 17)
    histogram, _ = np.histogram(np.degrees(directional_errors), bins=edges)
    resultant = np.mean(np.exp(1j * directional_errors)) if directional_errors.size else None
    return {
        "n_timepoints": int(len(magnitudes)),
        "n_directional_timepoints": int(directional_errors.size),
        "target_pct": float(target_match.mean() * 100),
        "low_amplitude_pct": float(low.mean() * 100),
        "pro_pct": float(pro.mean() * 100),
        "other_pct": float(other.mean() * 100),
        "mean_output_magnitude": float(magnitudes.mean()),
        "median_output_magnitude": float(np.median(magnitudes)),
        "mean_target_magnitude": float(np.linalg.norm(target_vectors, axis=1).mean()),
        "angular_resultant_length": float(abs(resultant)) if resultant is not None else None,
        "mean_angular_error_deg": (float(np.degrees(np.angle(resultant)))
                                   if resultant is not None and abs(resultant) > 1e-8 else None),
        "mean_abs_angular_error_deg": (float(np.degrees(np.abs(directional_errors)).mean())
                                       if directional_errors.size else None),
        "angle_histogram_counts": histogram.tolist(),
        "angle_histogram_edges_deg": edges.tolist(),
        "amplitude_threshold": amplitude_threshold,
        "sector_half_width_deg": float(np.degrees(half_width)),
        "scoring_windows_steps": windows,
        "definition": "raw sin/cos outputs; adequate-amplitude anti target, else low amplitude, else opposite (pro) sector, else other direction; low-norm outputs excluded from angular statistics",
    }


def _evaluate_inputs(model, inputs, targets, masks, scoring_inputs, batch_size, device,
                     direction_report=None, n_directions=8):
    """Score concatenated outputs using intact task labels and response masks."""
    import torch

    outputs = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            output, _, _ = model.iterate_sequence_batch(
                inputs[start:start + batch_size].to(device), run_mode="minimal")
            outputs.append(output.cpu())
        combined_output = torch.cat(outputs)
        if direction_report is not None:
            direction_report.update(response_direction_diagnostics(
                combined_output.numpy(), targets.cpu().numpy(), masks.cpu().numpy(),
                n_directions=n_directions))
        accuracy, _ = model.compute_acc(
            combined_output.to(device), targets.to(device), masks.to(device),
            scoring_inputs.to(device), isvalid=True, mode="angle",
        )
    value = float(accuracy)
    if not np.isfinite(value):
        raise ValueError("Non-finite accuracy")
    return value


def evaluate_task(model, task_params, rule, rule_column, n_trials, batch_size,
                  device, test_seed, *, pretraining_shift=0, pretraining_shift_pre=0,
                  rule_cue=False):
    import torch
    import _bootstrap  # noqa: F401
    import mpn_tasks

    params = copy.deepcopy(task_params)
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    params["hp"]["rng"] = np.random.RandomState(test_seed)
    params["hp"]["batch_size_train"] = n_trials
    (inputs, targets, masks), (_, trials, _) = mpn_tasks.generate_trials_wrap(
        params, n_trials, rules=[rule], mode_input="random_batch", device="cpu",
        pretraining_shift=pretraining_shift,
        pretraining_shift_pre=pretraining_shift_pre,
    )
    expected_inputs = model.W_initial_linear.in_features
    if inputs.shape[-1] != expected_inputs:
        raise ValueError(f"{rule}: input width {inputs.shape[-1]} != checkpoint {expected_inputs}")
    rule_offset = expected_inputs - 3
    if not torch.all(inputs[:, 0, rule_offset:].argmax(dim=-1) == rule_column):
        raise ValueError(f"{rule}: generated task cue is not in column {rule_column}")
    direction_report = {} if rule_cue and rule == "delayanti" else None
    value = _evaluate_inputs(model, inputs, targets, masks, inputs, batch_size, device,
                             direction_report=direction_report, n_directions=params["n_eachring"])
    result = {"task": rule, "accuracy": value, "accuracy_pct": value * 100,
            "n_trials": int(inputs.shape[0]), "test_seed": test_seed,
            "rule_column": rule_column}
    if rule_cue:
        conditions = {"intact": {"accuracy_pct": value * 100, "delta_accuracy_pp": 0.0}}
        if direction_report is not None:
            conditions["intact"]["error_direction"] = direction_report
        for condition in CUE_CONDITIONS:
            if condition == "intact":
                continue
            altered = apply_rule_cue(inputs, trials[0].epochs, rule_column, condition)
            if altered is None:
                conditions[condition] = {"accuracy_pct": None, "delta_accuracy_pp": None,
                                         "reason": "Task has no delay1 memory period"}
                continue
            direction_report = {} if rule == "delayanti" else None
            accuracy = _evaluate_inputs(model, altered, targets, masks, inputs, batch_size, device,
                                        direction_report=direction_report, n_directions=params["n_eachring"])
            conditions[condition] = {"accuracy_pct": accuracy * 100,
                                     "delta_accuracy_pp": (accuracy - value) * 100}
            if direction_report is not None:
                conditions[condition]["error_direction"] = direction_report
        result["conditions"] = conditions
        result["dt_ms"] = params["dt"]
        result["epochs_steps"] = {key: [None if bound is None else np.asarray(bound).tolist()
                                          for bound in bounds]
                                   for key, bounds in trials[0].epochs.items()}
    return result


def evaluate_checkpoint(path, run_ruleset, seed, args, device):
    import torch
    import _bootstrap  # noqa: F401
    import mpn

    aname = path.stem.removeprefix("savednet_")
    stage1 = load_task_params(args.checkpoint_dir, aname, "stage1")
    stage2 = load_task_params(args.checkpoint_dir, aname, "stage2")
    if stage1["rules"] != run_ruleset.split("_") or stage2["rules"] != ["delayanti"]:
        raise ValueError(f"{aname}: unexpected stage task configuration")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = mpn.DeepMultiPlasticNet(copy.deepcopy(checkpoint["net_params"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    tasks = []
    for column, rule in enumerate(stage1["rules"] + stage2["rules"]):
        posttraining = column == 2
        result = evaluate_task(
            model, stage2 if posttraining else stage1, rule, column,
            args.n_trials, args.batch_size, device, args.test_seed + column,
            pretraining_shift=2 if posttraining else 0,
            pretraining_shift_pre=0 if posttraining else 1,
            rule_cue=getattr(args, "rule_cue", False),
        )
        result["stage"] = "posttraining" if posttraining else "pretraining"
        tasks.append(result)
        print(f"seed={seed} {run_ruleset} task={rule}: "
              f"accuracy={result['accuracy_pct']:.2f}% (trials={result['n_trials']})")
        for condition, stats in result.get("conditions", {}).items():
            print(f"  {condition}: accuracy={stats['accuracy_pct']}, "
              f"delta_pp={stats['delta_accuracy_pp']}")
    return {"aname": aname, "ruleset": run_ruleset, "seed": seed, "tasks": tasks}


def _group_runs(runs):
    grouped = {ruleset: {task: [] for task in tasks}
               for ruleset, (_, tasks) in GROUPS.items()}
    for run in runs:
        for task in run["tasks"]:
            grouped[run["ruleset"]][task["task"]].append(task["accuracy_pct"])
    return grouped


def summarize(runs):
    return {ruleset: {task: {"n_seeds": len(values), "mean_accuracy_pct": float(np.mean(values)),
                             "std_accuracy_pct": float(np.std(values))}
                       for task, values in tasks.items() if values}
            for ruleset, tasks in _group_runs(runs).items() if any(tasks.values())}


def run_evaluation(args):
    import torch

    matches = discover_checkpoints(
        args.checkpoint_dir, args.feature, args.hidden, args.ruleset, args.seed,
        args.total_seed, args.test_seed)
    if not matches:
        raise ValueError("No checkpoints match the requested configuration")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    runs, failures = [], []
    rule_cue = getattr(args, "rule_cue", False)
    prefix = "rule_cue" if rule_cue else "accuracy"
    for path, ruleset, seed in matches:
        try:
            run = evaluate_checkpoint(path, ruleset, seed, args, device)
            with (args.input_dir / f"{prefix}_{run['aname']}.json").open("w") as handle:
                json.dump(run, handle, indent=2, allow_nan=False)
            runs.append(run)
        except Exception as error:
            failures.append({"checkpoint": str(path), "error": str(error)})
            print(f"FAILED {path.name}: {error}")
    summary = summarize_rule_cue(runs) if rule_cue else summarize(runs)
    tag = f"{args.ruleset or 'all'}_hidden{args.hidden}_{args.feature}"
    if args.seed is not None:
        tag += f"_seed{args.seed}"
    elif args.total_seed is not None:
        tag += f"_n{args.total_seed}"
    report = args.input_dir / f"{'rule_cue_' if rule_cue else ''}summary_{tag}.json"
    with report.open("w") as handle:
        json.dump({"settings": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                   "runs": runs, "summary": summary, "failures": failures},
                  handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {report}")
    return runs, failures


def summarize_rule_cue(runs):
    """Aggregate accuracy, paired changes, and available direction diagnostics over seeds."""
    summary = {}
    for ruleset, (_, tasks) in GROUPS.items():
        selected = [run for run in runs if run["ruleset"] == ruleset]
        if not selected:
            continue
        summary[ruleset] = {}
        for task in tasks:
            summary[ruleset][task] = {}
            for condition in CUE_CONDITIONS:
                entries = [entry["conditions"][condition] for run in selected
                           for entry in run["tasks"] if entry["task"] == task
                           and entry["conditions"][condition]["accuracy_pct"] is not None]
                stats = {"n_seeds": len(entries)}
                for key in ("accuracy_pct", "delta_accuracy_pp"):
                    values = [entry[key] for entry in entries]
                    stats[f"mean_{key}"] = float(np.mean(values)) if values else None
                    stats[f"std_{key}"] = float(np.std(values)) if values else None
                reports = [entry["error_direction"] for entry in entries if "error_direction" in entry]
                if reports:
                    direction_summary = {"n_seeds": len(reports)}
                    for key in (*ERROR_CATEGORIES, "mean_output_magnitude", "angular_resultant_length",
                                "mean_abs_angular_error_deg"):
                        values = [report[key] for report in reports if report[key] is not None]
                        direction_summary[key] = {
                            "n_seeds": len(values),
                            "mean": float(np.mean(values)) if values else None,
                            "std": float(np.std(values)) if values else None,
                        }
                    stats["error_direction"] = direction_summary
                summary[ruleset][task][condition] = stats
    return summary


def plot_rule_cue(runs, output_dir, feature, hidden):
    """Plot MemoryAnti accuracy in two motif panels, retaining all tasks in reports."""
    if not runs:
        return None
    summary = summarize_rule_cue(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True, sharey=True)
    positions = np.arange(len(CUE_CONDITIONS))
    for column, (ruleset, (title, tasks)) in enumerate(GROUPS.items()):
        axis = axes[column]
        for task, color in zip(tasks, COLORS):
            if task != "delayanti":
                continue
            for run in runs:
                if run["ruleset"] != ruleset:
                    continue
                entry = next(entry for entry in run["tasks"] if entry["task"] == task)
                values = [entry["conditions"][condition]["accuracy_pct"] for condition in CUE_CONDITIONS]
                axis.plot(positions, np.asarray(values, dtype=float), color=color,
                          linewidth=0.7, alpha=0.18)
            if ruleset not in summary:
                continue
            stats = summary[ruleset][task]
            means = np.array([stats[condition]["mean_accuracy_pct"] for condition in CUE_CONDITIONS], dtype=float)
            stds = np.array([stats[condition]["std_accuracy_pct"] for condition in CUE_CONDITIONS], dtype=float)
            axis.errorbar(positions, means, yerr=stds, fmt="o-", color=color,
                          linewidth=1.5, markersize=4, capsize=2,
                          label=f"MemoryAnti (n={stats['intact']['n_seeds']})")
        axis.set_title(title)
        axis.set_ylim(0, 105)
        axis.set_xticks(positions)
        axis.set_xticklabels(list(CUE_CONDITIONS.values()), fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        if ruleset in summary:
            axis.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"MemoryAnti rule-cue timing | hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = "_".join(run["aname"] for run in runs) if len(runs) == 1 else f"{'_'.join(summary)}_hidden{hidden}_{feature}"
    path = _figure_path(output_dir, f"rule_cue_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_rule_cue_errors(runs, output_dir, feature, hidden):
    """Show seed-balanced MemoryAnti error fractions and angular distributions."""
    reports = {ruleset: {condition: [] for condition in CUE_CONDITIONS} for ruleset in GROUPS}
    for run in runs:
        for task in run["tasks"]:
            if task["task"] != "delayanti":
                continue
            for condition in CUE_CONDITIONS:
                report = task["conditions"].get(condition, {}).get("error_direction")
                if report is not None:
                    reports[run["ruleset"]][condition].append(report)
    if not any(entries for conditions in reports.values() for entries in conditions.values()):
        print("Skipped error-direction plot: rerun --rule-cue to collect raw-output diagnostics")
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row")
    condition_colors = ("#718096", "#805ad5", "#dd6b20", "#3182ce", "#38a169")
    for column, (ruleset, (title, _)) in enumerate(GROUPS.items()):
        for position, condition in enumerate(CUE_CONDITIONS):
            entries = reports[ruleset][condition]
            if not entries:
                continue
            bottom = 0.0
            for key, (label, color) in ERROR_CATEGORIES.items():
                height = np.mean([entry[key] for entry in entries])
                axes[0, column].bar(position, height, bottom=bottom, width=0.65,
                                    color=color, label=label if position == 0 else None)
                bottom += height
            axes[0, column].text(position, 102, f"n={len(entries)}", ha="center", fontsize=7)
            directional = [entry for entry in entries if entry["n_directional_timepoints"] > 0]
            if directional:
                edges = np.asarray(directional[0]["angle_histogram_edges_deg"])
                if any(not np.array_equal(entry["angle_histogram_edges_deg"], edges) for entry in directional):
                    raise ValueError("Cannot combine different angular histogram bins")
                fractions = [np.asarray(entry["angle_histogram_counts"]) / entry["n_directional_timepoints"]
                             for entry in directional]
                axes[1, column].stairs(np.mean(fractions, axis=0) * 100, edges,
                                      color=condition_colors[position],
                                      label=f"{CUE_CONDITIONS[condition].replace(chr(10), ' ')} (n={len(directional)})")
        axes[0, column].set_title(title)
        axes[0, column].set_xticks(range(len(CUE_CONDITIONS)))
        axes[0, column].set_xticklabels(list(CUE_CONDITIONS.values()), fontsize=8)
        axes[0, column].set_ylim(0, 110)
        axes[0, column].set_yticks([0, 25, 50, 75, 100])
        axes[1, column].set_xlim(-180, 180)
        axes[1, column].set_xticks([-180, -90, 0, 90, 180])
        axes[1, column].set_xlabel("Angular error from anti target (deg)\n0 = anti target", fontsize=8)
        for axis in axes[:, column]:
            axis.spines[["top", "right"]].set_visible(False)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2)
    axes[0, 0].set_ylabel("Response timepoints (%)")
    axes[1, 0].set_ylabel("Non-low-amplitude timepoints per bin (%)")
    fig.suptitle(f"MemoryAnti rule-cue error directions | hidden{hidden} | {feature}")
    fig.tight_layout(h_pad=4)
    tag = runs[0]["aname"] if len(runs) == 1 else f"{'_'.join(dict.fromkeys(run['ruleset'] for run in runs))}_hidden{hidden}_{feature}"
    path = _figure_path(output_dir, f"rule_cue_errors_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# Scales 3-4 test the gain-crossing account of the improper motif: if the cue
# trace under-shoots a multiplicative sign inversion, accuracy without the
# online response cue should keep rising with scale and saturate near intact.
# Overrides remain clamped to the layer's modulation bounds, which caps how
# far large scales can actually push M.
M_TRACE_SCALES = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
M_CONDITIONS = {
    "replay_intact": "Intact\n(replay)",
    **{f"trace_scale_{scale:g}": f"Cue trace\nx{scale:g}" for scale in M_TRACE_SCALES},
    "trace_swap": "Swapped-trial\ncue trace",
}


def _clamp_to_m_bounds(layer, M):
    """Keep an intervened plastic state inside the layer's modulation bounds."""
    import torch

    if layer.modulation_bounds:
        return torch.clamp(M, min=layer.M_bounds[1], max=layer.M_bounds[0])
    return M


def _rollout_m_intervention(model, inputs, boundaries, device, batch_size,
                            m_override=None, capture_m=False):
    """Stepwise rollout that reads or replaces M at per-trial response onsets.

    At the top of each trial's boundary step — so M reflects everything before
    the response and nothing after — M is optionally recorded (capture_m) and
    optionally replaced with that trial's m_override row, clamped to the
    layer's modulation bounds. Normal M updates continue after replacement.
    Returns (outputs, captured_M or None) on CPU.
    """
    import torch

    layer = model.mp_layers[0]
    if not torch.all((boundaries > 0) & (boundaries < inputs.shape[1])):
        raise ValueError("Response onset must fall strictly inside the sequence")
    outputs = torch.empty(inputs.shape[0], inputs.shape[1], model.n_output)
    captured = torch.empty(inputs.shape[0], *layer.W.shape) if capture_m else None
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start:start + batch_size].to(device)
            bounds = boundaries[start:start + len(batch)].to(device)
            model.reset_state(B=len(batch))
            for timestep in range(batch.shape[1]):
                rows = torch.nonzero(bounds == timestep).squeeze(-1)
                if rows.numel():
                    rows_cpu = rows.cpu()
                    if capture_m:
                        captured[rows_cpu + start] = layer.M[rows].cpu()
                    if m_override is not None:
                        layer.M[rows] = _clamp_to_m_bounds(
                            layer, m_override[start:start + len(batch)][rows_cpu].to(device))
                output = model.network_step(batch[:, timestep], run_mode="minimal",
                                            seq_idx=timestep)[0]
                outputs[start:start + len(batch), timestep] = output.cpu()
    return outputs, captured


def _rowwise_pearson(a, b):
    """Per-row Pearson correlation of two (trials, features) tensors."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denominator = (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)
    return (a * b).sum(dim=1) / denominator


def _score_outputs(model, outputs, targets, masks, scoring_inputs, device):
    """Angle accuracy of precomputed outputs, matching _evaluate_inputs scoring."""
    accuracy, _ = model.compute_acc(
        outputs.to(device), targets.to(device), masks.to(device),
        scoring_inputs.to(device), isvalid=True, mode="angle")
    value = float(accuracy)
    if not np.isfinite(value):
        raise ValueError("Non-finite accuracy")
    return value


def evaluate_m_intervention_checkpoint(path, run_ruleset, seed, args, device):
    """Scale or transplant the cue-attributable plastic trace at response onset.

    The trace is defined by paired cue perturbations: M at response onset with the cue
    intact minus M on the identical trial with the cue channel silenced
    throughout. Trace conditions run the response with the cue off, so
    trace_scale_1 is expected to reproduce --rule-cue's off_after_memory, up to
    numerical differences and clamping. The replay check below compares intact
    stepwise and batched accuracy; it does not verify that scale-1 equivalence.
    trace_swap adds a cyclically shifted donor's trace to the recipient's no-cue
    M. Donors may share the recipient's stimulus (or be the same trial for a
    one-trial evaluation); the direction-mismatch fraction is saved. This probes trace
    transferability, not a guaranteed task/stimulus decomposition. Normal M
    updates continue during the response in all conditions.
    """
    import torch
    import _bootstrap  # noqa: F401
    import mpn
    import mpn_tasks

    aname = path.stem.removeprefix("savednet_")
    stage2 = load_task_params(args.checkpoint_dir, aname, "stage2")
    if stage2["rules"] != ["delayanti"]:
        raise ValueError(f"{aname}: unexpected stage-2 task configuration")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = mpn.DeepMultiPlasticNet(copy.deepcopy(checkpoint["net_params"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if len(model.mp_layers) != 1:
        raise ValueError("M intervention assumes a single plastic layer")

    params = copy.deepcopy(stage2)
    test_seed = args.test_seed + 2  # delayanti is rule column 2, matching evaluate_task
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    params["hp"]["rng"] = np.random.RandomState(test_seed)
    params["hp"]["batch_size_train"] = args.n_trials
    (inputs, targets, masks), (_, trials, _) = mpn_tasks.generate_trials_wrap(
        params, args.n_trials, rules=["delayanti"], mode_input="random_batch",
        device="cpu", pretraining_shift=2, pretraining_shift_pre=0)
    if inputs.shape[-1] != model.W_initial_linear.in_features:
        raise ValueError(f"delayanti: input width {inputs.shape[-1]} != checkpoint")
    if not torch.all(inputs[:, 0, -3:].argmax(dim=-1) == 2):
        raise ValueError("delayanti: generated task cue is not in column 2")
    epochs = trials[0].epochs
    if "delay1" not in epochs:
        raise ValueError("delayanti trials must have a delay1 memory period")
    boundaries = torch.as_tensor(epochs["delay1"][1])
    if boundaries.ndim == 0:
        boundaries = boundaries.expand(inputs.shape[0])
    boundaries = boundaries.long()

    inputs_nocue = inputs.clone()
    inputs_nocue[:, :, -1] = 0
    inputs_resp_off = apply_rule_cue(inputs, epochs, 2, "off_after_memory")

    outputs_intact, M_intact = _rollout_m_intervention(
        model, inputs, boundaries, device, args.batch_size, capture_m=True)
    outputs_nocue, M_nocue = _rollout_m_intervention(
        model, inputs_nocue, boundaries, device, args.batch_size, capture_m=True)
    trace = M_intact - M_nocue

    intact_acc = _score_outputs(model, outputs_intact, targets, masks, inputs, device)
    batch_acc = _evaluate_inputs(model, inputs, targets, masks, inputs,
                                 args.batch_size, device)
    if abs(intact_acc - batch_acc) > 1e-4:
        raise ValueError(f"Stepwise replay diverges from batch evaluation: "
                         f"{intact_acc} vs {batch_acc}")

    conditions = {"replay_intact": {"accuracy_pct": intact_acc * 100,
                                    "delta_accuracy_pp": 0.0}}
    layer = model.mp_layers[0]
    for scale in M_TRACE_SCALES:
        override = M_nocue + scale * trace
        # Fraction of override entries outside the modulation bounds — those
        # are clamped on application, so a large-scale condition whose
        # accuracy saturates may be bound-limited rather than trace-limited.
        clamped_frac = None
        if layer.modulation_bounds:
            bounds = layer.M_bounds.detach().cpu()
            clamped_frac = float(((override > bounds[0]) | (override < bounds[1]))
                                 .float().mean())
        outputs, _ = _rollout_m_intervention(model, inputs_resp_off, boundaries,
                                             device, args.batch_size, m_override=override)
        accuracy = _score_outputs(model, outputs, targets, masks, inputs, device)
        conditions[f"trace_scale_{scale:g}"] = {
            "accuracy_pct": accuracy * 100,
            "delta_accuracy_pp": (accuracy - intact_acc) * 100,
            "clamped_frac": clamped_frac}
    donor = np.roll(np.arange(inputs.shape[0]), 1)
    override = M_nocue + trace[donor]
    outputs, _ = _rollout_m_intervention(model, inputs_resp_off, boundaries,
                                         device, args.batch_size, m_override=override)
    accuracy = _score_outputs(model, outputs, targets, masks, inputs, device)
    conditions["trace_swap"] = {"accuracy_pct": accuracy * 100,
                                "delta_accuracy_pp": (accuracy - intact_acc) * 100}

    meta = getattr(trials[0], "meta", None)
    directions = (np.asarray(meta["stim1"]).reshape(-1)
                  if isinstance(meta, dict) and "stim1" in meta else None)
    swap_mismatch = (float(np.mean(directions[donor] != directions))
                     if directions is not None and directions.size == inputs.shape[0]
                     else None)

    trace_norms = trace.flatten(start_dim=1).norm(dim=1)
    base_norms = M_intact.flatten(start_dim=1).norm(dim=1).clamp(min=1e-12)
    flat_trace = trace.flatten(start_dim=1)
    flat_intact = M_intact.flatten(start_dim=1)
    flat_nocue = M_nocue.flatten(start_dim=1)
    corr_intact = _rowwise_pearson(flat_trace, flat_intact)
    corr_nocue = _rowwise_pearson(flat_trace, flat_nocue)
    trace_vs_m = {
        "mean_abs_trace": float(flat_trace.abs().mean()),
        "mean_abs_m_intact": float(flat_intact.abs().mean()),
        # Elementwise L1 distance |M_intact - trace|; identically |M_nocue|,
        # since trace = M_intact - M_nocue. Kept explicit for the figure.
        "mean_abs_l1_distance": float((flat_intact - flat_trace).abs().mean()),
        "mean_corr_trace_m_intact": float(corr_intact.mean()),
        "std_corr_trace_m_intact": float(corr_intact.std()),
        "mean_corr_trace_m_nocue": float(corr_nocue.mean()),
        "std_corr_trace_m_nocue": float(corr_nocue.std()),
    }
    task = {
        "task": "delayanti", "n_trials": int(inputs.shape[0]),
        "test_seed": test_seed, "dt_ms": params["dt"],
        "conditions": conditions,
        "references": {"nocue_accuracy_pct": _score_outputs(
            model, outputs_nocue, targets, masks, inputs, device) * 100},
        "trace_norm": {
            "mean_trace_frobenius": float(trace_norms.mean()),
            "std_trace_frobenius": float(trace_norms.std()),
            "mean_relative_trace": float((trace_norms / base_norms).mean()),
        },
        "swap_donor_direction_mismatch_frac": swap_mismatch,
        "trace_vs_m": trace_vs_m,
    }
    print(f"seed={seed} {run_ruleset} M intervention (delayanti):")
    for condition, stats in conditions.items():
        clamp_note = (f", clamped={stats['clamped_frac']:.3f}"
                      if stats.get("clamped_frac") is not None else "")
        print(f"  {condition}: accuracy={stats['accuracy_pct']:.2f}, "
              f"delta_pp={stats['delta_accuracy_pp']:.2f}{clamp_note}")
    return {"aname": aname, "ruleset": run_ruleset, "seed": seed, "task": task}


def summarize_m_intervention(runs):
    """Aggregate M-intervention accuracies and trace norms over checkpoint seeds."""
    summary = {}
    for ruleset in GROUPS:
        selected = [run for run in runs if run["ruleset"] == ruleset]
        if not selected:
            continue
        conditions = {}
        for condition in M_CONDITIONS:
            values = [run["task"]["conditions"][condition]["accuracy_pct"]
                      for run in selected]
            conditions[condition] = {"n_seeds": len(values),
                                     "mean_accuracy_pct": float(np.mean(values)),
                                     "std_accuracy_pct": float(np.std(values))}
            fractions = [run["task"]["conditions"][condition].get("clamped_frac")
                         for run in selected]
            fractions = [value for value in fractions if value is not None]
            if fractions:
                conditions[condition]["mean_clamped_frac"] = float(np.mean(fractions))
        summary[ruleset] = {"conditions": conditions}
        for key in ("mean_trace_frobenius", "mean_relative_trace"):
            values = [run["task"]["trace_norm"][key] for run in selected]
            summary[ruleset][key] = {"mean": float(np.mean(values)),
                                     "std": float(np.std(values))}
        stats = [run["task"]["trace_vs_m"] for run in selected
                 if "trace_vs_m" in run["task"]]
        if stats:
            summary[ruleset]["trace_vs_m"] = {
                key: {"n_seeds": len(stats),
                      "mean": float(np.mean([entry[key] for entry in stats])),
                      "std": float(np.std([entry[key] for entry in stats]))}
                for key in stats[0]}
    return summary


def run_m_intervention(args):
    import torch

    matches = discover_checkpoints(
        args.checkpoint_dir, args.feature, args.hidden, args.ruleset, args.seed,
        args.total_seed, args.test_seed)
    if not matches:
        raise ValueError("No checkpoints match the requested configuration")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    runs, failures = [], []
    for path, ruleset, seed in matches:
        try:
            run = evaluate_m_intervention_checkpoint(path, ruleset, seed, args, device)
            with (args.input_dir / f"m_intervention_{run['aname']}.json").open("w") as handle:
                json.dump(run, handle, indent=2, allow_nan=False)
            runs.append(run)
        except Exception as error:
            failures.append({"checkpoint": str(path), "error": str(error)})
            print(f"FAILED {path.name}: {error}")
    summary = summarize_m_intervention(runs)
    tag = f"{args.ruleset or 'all'}_hidden{args.hidden}_{args.feature}"
    if args.seed is not None:
        tag += f"_seed{args.seed}"
    elif args.total_seed is not None:
        tag += f"_n{args.total_seed}"
    report = args.input_dir / f"m_intervention_summary_{tag}.json"
    with report.open("w") as handle:
        json.dump({"settings": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                   "runs": runs, "summary": summary, "failures": failures},
                  handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {report}")
    return runs, failures


def plot_m_intervention(runs, output_dir, feature, hidden):
    """MemoryAnti accuracy under plastic-trace interventions, per motif panel."""
    if not runs:
        return None
    summary = summarize_m_intervention(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True, sharey=True)
    positions = np.arange(len(M_CONDITIONS))
    for column, (ruleset, (title, _)) in enumerate(GROUPS.items()):
        axis = axes[column]
        for run in runs:
            if run["ruleset"] != ruleset:
                continue
            values = [run["task"]["conditions"][condition]["accuracy_pct"]
                      for condition in M_CONDITIONS]
            axis.plot(positions, np.asarray(values, dtype=float), color="#e53e3e",
                      linewidth=0.7, alpha=0.18)
        if ruleset in summary:
            stats = summary[ruleset]["conditions"]
            means = np.array([stats[condition]["mean_accuracy_pct"]
                              for condition in M_CONDITIONS], dtype=float)
            stds = np.array([stats[condition]["std_accuracy_pct"]
                             for condition in M_CONDITIONS], dtype=float)
            axis.errorbar(positions, means, yerr=stds, fmt="o-", color="#e53e3e",
                          linewidth=1.5, markersize=4, capsize=2,
                          label=f"MemoryAnti (n={stats['replay_intact']['n_seeds']})")
            axis.legend(fontsize=8, frameon=False)
        axis.set_title(title)
        axis.set_ylim(0, 105)
        axis.set_xticks(positions)
        axis.set_xticklabels(list(M_CONDITIONS.values()), fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"MemoryAnti plastic-trace intervention at response onset "
                 f"(response cue off except intact replay) | hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = (runs[0]["aname"] if len(runs) == 1
           else f"{'_'.join(summary)}_hidden{hidden}_{feature}")
    path = _figure_path(
        output_dir, f"m_intervention_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_m_trace_similarity(runs, output_dir, feature, hidden):
    """Compare the cue-attributable trace with M at response onset, per motif.

    Left: mean absolute entry of the trace, the intact M, and their elementwise
    L1 distance (identically the no-cue M, since trace = M_intact - M_nocue).
    Right: per-trial Pearson correlation of the trace with the intact and
    no-cue M, averaged over trials; dots are individual seeds.
    """
    selected = {ruleset: [run for run in runs if run["ruleset"] == ruleset
                          and "trace_vs_m" in run["task"]] for ruleset in GROUPS}
    if not any(selected.values()):
        print("Skipped trace-vs-M plot: rerun --m-intervention to collect trace statistics")
        return None
    panels = (
        ("Mean |entry| (L1 / n entries)", {
            "mean_abs_trace": ("|ΔM|", "#3182ce"),
            "mean_abs_m_intact": ("|M intact|", "#718096"),
            "mean_abs_l1_distance": ("|M intact − ΔM|", "#38a169")}),
        ("Per-trial Pearson r (trial mean)", {
            "mean_corr_trace_m_intact": ("corr(ΔM, M intact)", "#805ad5"),
            "mean_corr_trace_m_nocue": ("corr(ΔM, M no-cue)", "#dd6b20")}),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, (ylabel, metrics) in zip(axes, panels):
        ticks, labels = [], []
        for group_index, (ruleset, (title, _)) in enumerate(GROUPS.items()):
            entries = selected[ruleset]
            base = group_index * (len(metrics) + 1)
            for metric_index, (key, (label, color)) in enumerate(metrics.items()):
                position = base + metric_index
                values = np.array([run["task"]["trace_vs_m"][key] for run in entries])
                if values.size:
                    axis.bar(position, values.mean(), yerr=values.std(), width=0.7,
                             color=color, alpha=0.8, capsize=3,
                             label=label if group_index == 0 else None)
                    jitter = (np.linspace(-0.15, 0.15, values.size)
                              if values.size > 1 else np.zeros(1))
                    axis.scatter(position + jitter, values, color="black", s=10,
                                 alpha=0.6, zorder=3)
            ticks.append(base + (len(metrics) - 1) / 2)
            labels.append(f"{title}\n(n={len(entries)})")
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Cue trace ΔM vs plastic state M at response onset | "
                 f"hidden{hidden} | {feature}")
    fig.tight_layout()
    plotted = [run for entries in selected.values() for run in entries]
    tag = (plotted[0]["aname"] if len(plotted) == 1
           else f"{'_'.join(ruleset for ruleset, entries in selected.items() if entries)}"
                f"_hidden{hidden}_{feature}")
    path = _figure_path(output_dir, f"m_trace_similarity_{tag}.png", seed_specific=len(plotted) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


PATHWAY_GAIN_CONDITIONS = {
    "baseline_w": "Frozen W\n(M = 0)",
    "nocue_onset": "No cue,\nresponse onset",
    "nocue_end": "No cue,\ntrial end",
    "trace_onset": "Cue trace,\nresponse onset",
    "trace_end_cue_off": "Trace, cue off\nin response (end)",
    "trace_end_cue_on": "Trace + online cue\n(trial end)",
}
PATHWAY_GAIN_TOP_FRAC = 0.01
PATHWAY_GAIN_DEPRESSED_BELOW = 0.1
PATHWAY_GAIN_ENHANCED_ABOVE = 1.9


def _rollout_capture_M_states(model, inputs, boundaries, device, batch_size):
    """Roll out inputs and capture M at response onset and after the last step.

    Onset follows _rollout_m_intervention's convention: M is read at the top of
    each trial's boundary step, so it reflects everything before the response
    and nothing after. Outputs are discarded. Returns CPU (M_onset, M_end).
    """
    import torch

    layer = model.mp_layers[0]
    if not torch.all((boundaries > 0) & (boundaries < inputs.shape[1])):
        raise ValueError("Response onset must fall strictly inside the sequence")
    M_onset = torch.empty(inputs.shape[0], *layer.W.shape)
    M_end = torch.empty_like(M_onset)
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start:start + batch_size].to(device)
            bounds = boundaries[start:start + len(batch)].to(device)
            model.reset_state(B=len(batch))
            for timestep in range(batch.shape[1]):
                rows = torch.nonzero(bounds == timestep).squeeze(-1)
                if rows.numel():
                    M_onset[rows.cpu() + start] = layer.M[rows].cpu()
                model.network_step(batch[:, timestep], run_mode="minimal",
                                   seq_idx=timestep)
            M_end[start:start + len(batch)] = layer.M.cpu()
    return M_onset, M_end


def evaluate_pathway_gain_checkpoint(path, run_ruleset, seed, args, device):
    """Sign of the stimulus-to-readout pathway gain through W_eff = W + W*M.

    Per trial, the stimulus-encoding direction u — the embedding-layer
    activation at the stimulus midpoint minus the pre-stimulus step, so the
    embedding tanh is included and everything but the stimulus cancels — is
    pushed through the plastic effective weights and projected onto the
    readout loading of the trial's PRO direction (the negated anti target,
    exact under the low_dim sin/cos encoding). Positive gain = the pathway
    drives the pro response; negative = anti. M is read from stepwise rollouts
    under no cue, the intact cue trace at response onset, and trial-end states
    with the response-period cue off (off_after_memory) or on (intact).
    This is a linear probe of one plastic layer and the frozen readout; it
    ignores the hidden tanh's local slope and bias paths, so the SIGN pattern
    across conditions is the readout, not the magnitudes.

    Under multiplicative bounds 1 + M lies in [0, 2], so no single synapse can
    flip the sign of its contribution; a negative summed gain must come from
    re-weighting pro-driving against anti-driving synapses. The per-synapse
    statistics quantify that directly: over each trial's top |a_i W_ij u_j|
    contributions, split by contribution sign, they report mean 1 + M and the
    fractions near the depression / enhancement bounds.
    """
    import torch
    import _bootstrap  # noqa: F401
    import mpn
    import mpn_tasks

    aname = path.stem.removeprefix("savednet_")
    stage2 = load_task_params(args.checkpoint_dir, aname, "stage2")
    if stage2["rules"] != ["delayanti"]:
        raise ValueError(f"{aname}: unexpected stage-2 task configuration")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = mpn.DeepMultiPlasticNet(copy.deepcopy(checkpoint["net_params"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if len(model.mp_layers) != 1:
        raise ValueError("Pathway gain assumes a single plastic layer")
    layer = model.mp_layers[0]

    params = copy.deepcopy(stage2)
    test_seed = args.test_seed + 2  # delayanti is rule column 2, matching evaluate_task
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    params["hp"]["rng"] = np.random.RandomState(test_seed)
    params["hp"]["batch_size_train"] = args.n_trials
    (inputs, targets, masks), (_, trials, _) = mpn_tasks.generate_trials_wrap(
        params, args.n_trials, rules=["delayanti"], mode_input="random_batch",
        device="cpu", pretraining_shift=2, pretraining_shift_pre=0)
    if inputs.shape[-1] != model.W_initial_linear.in_features:
        raise ValueError(f"delayanti: input width {inputs.shape[-1]} != checkpoint")
    if not torch.all(inputs[:, 0, -3:].argmax(dim=-1) == 2):
        raise ValueError("delayanti: generated task cue is not in column 2")
    epochs = trials[0].epochs
    if "delay1" not in epochs:
        raise ValueError("delayanti trials must have a delay1 memory period")
    boundaries = torch.as_tensor(epochs["delay1"][1])
    if boundaries.ndim == 0:
        boundaries = boundaries.expand(inputs.shape[0])
    boundaries = boundaries.long()

    inputs_nocue = inputs.clone()
    inputs_nocue[:, :, -1] = 0
    inputs_resp_off = apply_rule_cue(inputs, epochs, 2, "off_after_memory")

    M_trace_onset, M_intact_end = _rollout_capture_M_states(
        model, inputs, boundaries, device, args.batch_size)
    M_nocue_onset, M_nocue_end = _rollout_capture_M_states(
        model, inputs_nocue, boundaries, device, args.batch_size)
    _, M_respoff_end = _rollout_capture_M_states(
        model, inputs_resp_off, boundaries, device, args.batch_size)

    # Stimulus-encoding direction at the plastic layer's input, per trial.
    stim_start = torch.as_tensor(epochs["stim1"][0])
    stim_end = torch.as_tensor(epochs["stim1"][1])
    if stim_start.ndim == 0:
        stim_start = stim_start.expand(inputs.shape[0])
    if stim_end.ndim == 0:
        stim_end = stim_end.expand(inputs.shape[0])
    stim_start, stim_end = stim_start.long(), stim_end.long()
    if not torch.all(stim_start >= 1):
        raise ValueError("Need a pre-stimulus step to isolate the stimulus encoding")
    trial_index = torch.arange(inputs.shape[0])
    mid = (stim_start + stim_end) // 2

    with torch.no_grad():
        def embed(frames):
            return model.act_fn(model.W_initial_linear(frames.to(device)))

        u = embed(inputs[trial_index, mid]) - embed(inputs[trial_index, stim_start - 1])

        # Pro readout loading: negated, normalized anti target after onset.
        response_targets = targets[:, :, 1:3]
        time_index = torch.arange(inputs.shape[1])[None, :]
        scored = ((time_index >= boundaries[:, None])
                  & (response_targets.norm(dim=-1) > 1e-6)).float()
        counts = scored.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("No nonzero response target found after response onset")
        anti = (response_targets * scored.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
        anti = anti / anti.norm(dim=1, keepdim=True).clamp(min=1e-12)
        readout_loading = (-anti).to(device) @ model.W_output[1:3, :]

        W = layer.W.detach()

        def pathway_gain(M_batch):
            if M_batch is None:
                drive = torch.einsum('iI,BI->Bi', W, u)
            else:
                effective = layer.get_modulated_weights(M=M_batch.to(device))
                drive = torch.einsum('BiI,BI->Bi', effective, u)
            return torch.einsum('Bi,Bi->B', readout_loading, drive).cpu()

        condition_M = {
            "baseline_w": None,
            "nocue_onset": M_nocue_onset,
            "nocue_end": M_nocue_end,
            "trace_onset": M_trace_onset,
            "trace_end_cue_off": M_respoff_end,
            "trace_end_cue_on": M_intact_end,
        }
        gains = {name: pathway_gain(M_batch)
                 for name, M_batch in condition_M.items()}
        baseline_scale = float(gains["baseline_w"].abs().mean())
        if baseline_scale <= 0 or not np.isfinite(baseline_scale):
            raise ValueError("Degenerate frozen-W baseline pathway gain")
        conditions = {name: {
            "mean_gain": float(g.mean()),
            "std_gain": float(g.std()),
            "mean_gain_norm": float(g.mean()) / baseline_scale,
            "positive_frac": float((g > 0).float().mean()),
        } for name, g in gains.items()}

        # Per-synapse 1 + M statistics on the top pathway contributions.
        m_stats = None
        if layer.mp_type == "mult":
            contributions = torch.einsum('Bi,iI,BI->BiI', readout_loading, W, u)
            flat = contributions.flatten(start_dim=1)
            k = max(1, int(flat.shape[1] * PATHWAY_GAIN_TOP_FRAC))
            cutoffs = flat.abs().topk(k, dim=1).values[:, -1]
            top = flat.abs() >= cutoffs[:, None]
            synapse_masks = {"pro_driving": (top & (flat > 0)).float(),
                             "anti_driving": (top & (flat < 0)).float()}
            m_stats = {}
            for name, M_batch in condition_M.items():
                if M_batch is None:
                    continue
                one_plus = (1.0 + M_batch.to(device)).flatten(start_dim=1)
                entry = {}
                for side, mask in synapse_masks.items():
                    n_selected = mask.sum(dim=1).clamp(min=1)
                    entry[side] = {
                        "mean_one_plus_m": float(
                            ((one_plus * mask).sum(dim=1) / n_selected).mean()),
                        "frac_depressed": float(
                            (((one_plus < PATHWAY_GAIN_DEPRESSED_BELOW).float() * mask)
                             .sum(dim=1) / n_selected).mean()),
                        "frac_enhanced": float(
                            (((one_plus > PATHWAY_GAIN_ENHANCED_ABOVE).float() * mask)
                             .sum(dim=1) / n_selected).mean()),
                    }
                m_stats[name] = entry

    print(f"seed={seed} {run_ruleset} pathway gain (delayanti):")
    for name, stats in conditions.items():
        print(f"  {name}: gain/|baseline|={stats['mean_gain_norm']:+.3f}, "
              f"positive_frac={stats['positive_frac']:.2f}")
    return {
        "aname": aname,
        "ruleset": run_ruleset,
        "seed": seed,
        "task": "delayanti",
        "n_trials": int(inputs.shape[0]),
        "test_seed": test_seed,
        "baseline_abs_gain": baseline_scale,
        "conditions": conditions,
        "m_stats": m_stats,
        "settings": {"top_frac": PATHWAY_GAIN_TOP_FRAC,
                     "depressed_below": PATHWAY_GAIN_DEPRESSED_BELOW,
                     "enhanced_above": PATHWAY_GAIN_ENHANCED_ABOVE,
                     "mp_type": layer.mp_type},
    }


def summarize_pathway_gain(runs):
    """Aggregate normalized pathway gains and 1 + M statistics over seeds."""
    def stats(values):
        values = np.asarray(values, dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std())}

    summary = {}
    for ruleset in GROUPS:
        selected = [run for run in runs if run["ruleset"] == ruleset]
        if not selected:
            continue
        entry = {"n_seeds": len(selected), "conditions": {}}
        for condition in PATHWAY_GAIN_CONDITIONS:
            entry["conditions"][condition] = {
                key: stats([run["conditions"][condition][key] for run in selected])
                for key in ("mean_gain_norm", "positive_frac")}
        with_m = [run for run in selected if run["m_stats"]]
        if with_m:
            entry["m_stats"] = {}
            for condition in with_m[0]["m_stats"]:
                entry["m_stats"][condition] = {
                    side: {key: stats([run["m_stats"][condition][side][key]
                                       for run in with_m])
                           for key in ("mean_one_plus_m", "frac_depressed",
                                       "frac_enhanced")}
                    for side in ("pro_driving", "anti_driving")}
        summary[ruleset] = entry
    return summary


def run_pathway_gain(args):
    """Run the pathway-gain probe on every matching final checkpoint."""
    import torch

    matches = discover_checkpoints(
        args.checkpoint_dir, args.feature, args.hidden, args.ruleset, args.seed,
        args.total_seed, args.test_seed)
    if not matches:
        raise ValueError("No checkpoints match the requested configuration")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    runs, failures = [], []
    for path, ruleset, seed in matches:
        try:
            run = evaluate_pathway_gain_checkpoint(path, ruleset, seed, args, device)
            with (args.input_dir / f"pathway_gain_{run['aname']}.json").open("w") as handle:
                json.dump(run, handle, indent=2, allow_nan=False)
            runs.append(run)
        except Exception as error:
            failures.append({"checkpoint": str(path), "error": str(error)})
            print(f"FAILED {path.name}: {error}")

    summary = summarize_pathway_gain(runs)
    tag = f"{args.ruleset or 'all'}_hidden{args.hidden}_{args.feature}"
    if args.seed is not None:
        tag += f"_seed{args.seed}"
    elif args.total_seed is not None:
        tag += f"_n{args.total_seed}"
    report = args.input_dir / f"pathway_gain_summary_{tag}.json"
    with report.open("w") as handle:
        json.dump({"settings": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                   "runs": runs, "summary": summary, "failures": failures},
                  handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {report}")
    return runs, failures


def plot_pathway_gain(runs, output_dir, feature, hidden):
    """Normalized pathway gain per condition, one panel per motif.

    Gains are divided by each seed's mean |frozen-W gain|, so seeds share a
    scale; the zero line separates a pro-driving (positive) from an
    anti-driving (negative) stimulus-to-readout pathway.
    """
    if not runs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    positions = np.arange(len(PATHWAY_GAIN_CONDITIONS))
    for axis, (ruleset, (title, _)) in zip(axes, GROUPS.items()):
        selected = [run for run in runs if run["ruleset"] == ruleset]
        for run in selected:
            values = [run["conditions"][condition]["mean_gain_norm"]
                      for condition in PATHWAY_GAIN_CONDITIONS]
            axis.plot(positions, values, color="#319795", linewidth=0.7, alpha=0.25)
        if selected:
            matrix = np.asarray([[run["conditions"][condition]["mean_gain_norm"]
                                  for condition in PATHWAY_GAIN_CONDITIONS]
                                 for run in selected], dtype=float)
            axis.errorbar(positions, matrix.mean(axis=0), yerr=matrix.std(axis=0),
                          fmt="o-", color="#319795", linewidth=1.5, markersize=4,
                          capsize=2, label=f"MemoryAnti (n={len(selected)})")
            axis.legend(fontsize=8, frameon=False)
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels(list(PATHWAY_GAIN_CONDITIONS.values()), fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pathway gain / |frozen-W gain|\n(+ = pro, − = anti)")
    fig.suptitle(f"Stimulus-to-readout pathway gain through W_eff | "
                 f"hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = _backbone_figure_tag(runs, summarize_pathway_gain(runs), hidden, feature)
    path = _figure_path(output_dir, f"pathway_gain_{tag}.png",
                        seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_pathway_gain_m_stats(runs, output_dir, feature, hidden):
    """Mean 1 + M on top pro- vs anti-driving synapses, per motif and condition.

    1.0 means no modulation; toward 0 = depressed to the multiplicative bound;
    toward 2 = enhanced to the bound. Since 1 + M cannot go negative, a sign
    flip of the summed gain must show here as depression of the pro-driving
    synapses and/or enhancement of the anti-driving ones.
    """
    selected_all = [run for run in runs if run.get("m_stats")]
    if not selected_all:
        print("Skipped pathway-gain 1+M plot: no multiplicative-M statistics")
        return None
    m_conditions = [condition for condition in PATHWAY_GAIN_CONDITIONS
                    if condition != "baseline_w"]
    positions = np.arange(len(m_conditions))
    styles = {"pro_driving": ("Pro-driving synapses", "#dd6b20"),
              "anti_driving": ("Anti-driving synapses", "#805ad5")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for axis, (ruleset, (title, _)) in zip(axes, GROUPS.items()):
        selected = [run for run in selected_all if run["ruleset"] == ruleset]
        for side, (label, color) in styles.items():
            for run in selected:
                values = [run["m_stats"][condition][side]["mean_one_plus_m"]
                          for condition in m_conditions]
                axis.plot(positions, values, color=color, linewidth=0.7, alpha=0.2)
            if selected:
                matrix = np.asarray(
                    [[run["m_stats"][condition][side]["mean_one_plus_m"]
                      for condition in m_conditions] for run in selected],
                    dtype=float)
                axis.errorbar(positions, matrix.mean(axis=0), yerr=matrix.std(axis=0),
                              fmt="o-", color=color, linewidth=1.5, markersize=4,
                              capsize=2, label=f"{label} (n={len(selected)})")
        axis.axhline(1.0, color="gray", linewidth=0.9, linestyle="--")
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_title(title)
        axis.set_ylim(-0.1, 2.1)
        axis.set_xticks(positions)
        axis.set_xticklabels([PATHWAY_GAIN_CONDITIONS[condition]
                              for condition in m_conditions], fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
        if selected:
            axis.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Mean 1 + M on top pathway synapses")
    fig.suptitle(f"Multiplicative modulation on the stimulus-to-readout pathway | "
                 f"hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = _backbone_figure_tag(runs, summarize_pathway_gain(runs), hidden, feature)
    path = _figure_path(output_dir, f"pathway_gain_m_stats_{tag}.png",
                        seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


RULE_VECTOR_CONDITIONS = {
    "original": "Original",
    "parallel_raw": "Pretraining span\n(raw)",
    "parallel_norm_matched": "Pretraining span\n(norm matched)",
    "perpendicular_raw": "Orthogonal residual\n(raw)",
    "perpendicular_norm_matched": "Orthogonal residual\n(norm matched)",
    "random_norm_matched": "Random direction\n(norm matched)",
}
N_RANDOM_RULE_VECTORS = 10
RULE_VECTOR_SWEEP_RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4,
                            0.5, 0.75, 1.0, 1.25, 1.5)
N_SWEEP_RANDOM_DIRECTIONS = 3


def _decompose_rule_vector(pretrained_vectors, novel_vector):
    """Split a novel vector into pretrained-span and orthogonal components."""
    import torch

    pretrained_vectors = torch.as_tensor(pretrained_vectors)
    novel_vector = torch.as_tensor(novel_vector)
    if pretrained_vectors.ndim != 2 or novel_vector.ndim != 1:
        raise ValueError("Expected pretrained (features, rules) and novel (features,) vectors")
    if pretrained_vectors.shape[0] != novel_vector.shape[0]:
        raise ValueError("Pretrained and novel rule vectors must have the same width")
    if not torch.isfinite(pretrained_vectors).all() or not torch.isfinite(novel_vector).all():
        raise ValueError("Rule vectors contain non-finite values")

    left, singular_values, _ = torch.linalg.svd(pretrained_vectors, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values.max()) == 0:
        raise ValueError("Pretraining rule-vector span is empty")
    tolerance = (torch.finfo(singular_values.dtype).eps
                 * max(pretrained_vectors.shape) * singular_values.max())
    rank = int((singular_values > tolerance).sum())
    basis = left[:, :rank]
    parallel = basis @ (basis.T @ novel_vector)
    perpendicular = novel_vector - parallel
    novel_norm = novel_vector.norm()
    if float(novel_norm) == 0:
        raise ValueError("Novel rule vector has zero norm")

    def norm_matched(vector, name):
        vector_norm = vector.norm()
        if float(vector_norm) <= float(tolerance):
            raise ValueError(f"Cannot norm-match a zero {name} component")
        return vector * (novel_norm / vector_norm)

    return {
        "parallel_raw": parallel,
        "parallel_norm_matched": norm_matched(parallel, "parallel"),
        "perpendicular_raw": perpendicular,
        "perpendicular_norm_matched": norm_matched(perpendicular, "perpendicular"),
        "rank": rank,
        "novel_norm": float(novel_norm),
        "parallel_norm": float(parallel.norm()),
        "perpendicular_norm": float(perpendicular.norm()),
        "in_span_fraction": float(parallel.norm() / novel_norm),
        "reconstruction_error": float((parallel + perpendicular - novel_vector).norm()),
    }


def evaluate_rule_vector_intervention_checkpoint(path, run_ruleset, seed, args, device):
    """Causally replace the MemoryAnti rule vector in one final checkpoint."""
    import torch
    import _bootstrap  # noqa: F401
    import mpn
    import mpn_tasks

    aname = path.stem.removeprefix("savednet_")
    stage1 = load_task_params(args.checkpoint_dir, aname, "stage1")
    stage2 = load_task_params(args.checkpoint_dir, aname, "stage2")
    if stage1["rules"] != run_ruleset.split("_") or stage2["rules"] != ["delayanti"]:
        raise ValueError(f"{aname}: unexpected stage task configuration")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = mpn.DeepMultiPlasticNet(copy.deepcopy(checkpoint["net_params"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    params = copy.deepcopy(stage2)
    test_seed = args.test_seed + 2
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    params["hp"]["rng"] = np.random.RandomState(test_seed)
    params["hp"]["batch_size_train"] = args.n_trials
    (inputs, targets, masks), _ = mpn_tasks.generate_trials_wrap(
        params, args.n_trials, rules=["delayanti"], mode_input="random_batch",
        device="cpu", pretraining_shift=2, pretraining_shift_pre=0)
    if inputs.shape[-1] != model.W_initial_linear.in_features:
        raise ValueError(f"delayanti: input width {inputs.shape[-1]} != checkpoint")
    if not torch.all(inputs[:, 0, -3:].argmax(dim=-1) == 2):
        raise ValueError("delayanti: generated task cue is not in column 2")

    weight = model.W_initial_linear.weight
    original = weight[:, -1].detach().cpu().clone()
    pretrained = weight[:, -3:-1].detach().cpu().clone()
    components = _decompose_rule_vector(pretrained, original)
    vectors = {"original": original}
    vectors.update({condition: components[condition]
                    for condition in RULE_VECTOR_CONDITIONS
                    if condition not in ("original", "random_norm_matched")})

    def evaluate_vector(vector):
        with torch.no_grad():
            weight[:, -1].copy_(vector.to(device=device, dtype=weight.dtype))
        return _evaluate_inputs(model, inputs, targets, masks, inputs,
                                args.batch_size, device) * 100

    conditions = {}
    try:
        for condition, vector in vectors.items():
            accuracy = evaluate_vector(vector)
            conditions[condition] = {"accuracy_pct": accuracy}

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int((args.test_seed + seed + 10_000) % (2**63 - 1)))
        random_vectors = torch.randn(
            N_RANDOM_RULE_VECTORS, original.numel(), generator=generator,
            dtype=original.dtype)
        random_vectors *= original.norm() / random_vectors.norm(dim=1, keepdim=True)
        random_accuracies = [evaluate_vector(vector) for vector in random_vectors]
        conditions["random_norm_matched"] = {
            "accuracy_pct": float(np.mean(random_accuracies)),
            "std_across_vectors_pct": float(np.std(random_accuracies)),
            "samples_pct": random_accuracies,
            "n_vectors": N_RANDOM_RULE_VECTORS,
        }
    finally:
        with torch.no_grad():
            weight[:, -1].copy_(original.to(device=device, dtype=weight.dtype))

    baseline = conditions["original"]["accuracy_pct"]
    for stats in conditions.values():
        stats["delta_accuracy_pp"] = stats["accuracy_pct"] - baseline
    geometry = {key: value for key, value in components.items()
                if not hasattr(value, "shape")}
    print(f"seed={seed} {run_ruleset} rule-vector intervention (delayanti):")
    for condition, stats in conditions.items():
        print(f"  {condition}: accuracy={stats['accuracy_pct']:.2f}, "
              f"delta_pp={stats['delta_accuracy_pp']:.2f}")
    return {
        "aname": aname,
        "ruleset": run_ruleset,
        "seed": seed,
        "task": "delayanti",
        "n_trials": int(inputs.shape[0]),
        "test_seed": test_seed,
        "pretraining_rules": stage1["rules"],
        "geometry": geometry,
        "conditions": conditions,
    }


def summarize_rule_vector_intervention(runs):
    """Aggregate counterfactual rule-vector accuracy across checkpoint seeds."""
    summary = {}
    for ruleset in GROUPS:
        selected = [run for run in runs if run["ruleset"] == ruleset]
        if not selected:
            continue
        conditions = {}
        for condition in RULE_VECTOR_CONDITIONS:
            values = np.asarray([run["conditions"][condition]["accuracy_pct"]
                                 for run in selected], dtype=float)
            conditions[condition] = {
                "n_seeds": len(values),
                "mean_accuracy_pct": float(values.mean()),
                "std_accuracy_pct": float(values.std()),
            }
        span = np.asarray([run["geometry"]["in_span_fraction"]
                           for run in selected], dtype=float)
        summary[ruleset] = {
            "conditions": conditions,
            "in_span_fraction": {
                "n_seeds": len(span),
                "mean": float(span.mean()),
                "std": float(span.std()),
            },
        }
        for key in ("novel_norm", "parallel_norm", "perpendicular_norm"):
            values = np.asarray([run["geometry"][key] for run in selected], dtype=float)
            summary[ruleset][key] = {
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
    return summary


def plot_rule_vector_intervention(runs, output_dir, feature, hidden):
    """Plot MemoryAnti accuracy under within-checkpoint rule-vector replacements."""
    if not runs:
        return None
    summary = summarize_rule_vector_intervention(runs)
    positions = np.arange(len(RULE_VECTOR_CONDITIONS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True, sharey=True)
    for axis, (ruleset, (title, _)) in zip(axes, GROUPS.items()):
        selected = [run for run in runs if run["ruleset"] == ruleset]
        for run in selected:
            values = [run["conditions"][condition]["accuracy_pct"]
                      for condition in RULE_VECTOR_CONDITIONS]
            axis.plot(positions, values, "o-", color="#805ad5",
                      linewidth=0.7, markersize=2.5, alpha=0.2)
        if selected:
            stats = summary[ruleset]["conditions"]
            means = [stats[condition]["mean_accuracy_pct"]
                     for condition in RULE_VECTOR_CONDITIONS]
            stds = [stats[condition]["std_accuracy_pct"]
                    for condition in RULE_VECTOR_CONDITIONS]
            axis.errorbar(positions, means, yerr=stds, fmt="o-", color="#805ad5",
                          linewidth=1.8, markersize=5, capsize=3,
                          label=f"MemoryAnti (n={len(selected)})")
            axis.legend(fontsize=8, frameon=False)
        axis.set_title(title)
        axis.set_ylim(-5, 105)
        axis.set_xticks(positions)
        axis.set_xticklabels(list(RULE_VECTOR_CONDITIONS.values()),
                             rotation=25, ha="right", fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"MemoryAnti rule-vector causal decomposition | "
                 f"hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = (runs[0]["aname"] if len(runs) == 1
           else f"{'_'.join(summary)}_hidden{hidden}_{feature}")
    path = _figure_path(
        output_dir, f"rule_vector_intervention_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_rule_vector_norms(runs, output_dir, feature, hidden):
    """Plot original and decomposed MemoryAnti rule-vector L2 norms."""
    if not runs:
        return None
    summary = summarize_rule_vector_intervention(runs)
    norm_metrics = {
        "novel_norm": "Original\nMemoryAnti",
        "parallel_norm": "Pretraining-span\ncomponent",
        "perpendicular_norm": "Orthogonal\nresidual",
    }
    positions = np.arange(len(norm_metrics))
    max_norm = max(
        run["geometry"][key]
        for run in runs
        for key in norm_metrics
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharex=True, sharey=True)
    for axis, (ruleset, (title, _)) in zip(axes, GROUPS.items()):
        selected = [run for run in runs if run["ruleset"] == ruleset]
        for run in selected:
            values = [run["geometry"][key] for key in norm_metrics]
            axis.plot(positions, values, "o-", color="#319795",
                      linewidth=0.7, markersize=2.5, alpha=0.2)
        if selected:
            means = [summary[ruleset][key]["mean"] for key in norm_metrics]
            stds = [summary[ruleset][key]["std"] for key in norm_metrics]
            axis.errorbar(positions, means, yerr=stds, fmt="o-", color="#319795",
                          linewidth=1.8, markersize=5, capsize=3,
                          label=f"MemoryAnti (n={len(selected)})")
            span = summary[ruleset]["in_span_fraction"]
            axis.text(0.04, 0.94,
                      f"span/original = {span['mean']:.2f} ± {span['std']:.2f}",
                      transform=axis.transAxes, va="top", fontsize=7)
            axis.legend(fontsize=8, frameon=False, loc="center right")
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels(list(norm_metrics.values()), fontsize=8)
        axis.set_ylim(0, max_norm * 1.12)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Rule-vector L2 norm")
    fig.suptitle(f"MemoryAnti rule-vector norm decomposition | "
                 f"hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = (runs[0]["aname"] if len(runs) == 1
           else f"{'_'.join(summary)}_hidden{hidden}_{feature}")
    path = _figure_path(
        output_dir, f"rule_vector_norms_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def run_rule_vector_intervention(args):
    """Run final-checkpoint rule-vector decomposition and counterfactual tests."""
    import torch

    matches = discover_checkpoints(
        args.checkpoint_dir, args.feature, args.hidden, args.ruleset, args.seed,
        args.total_seed, args.test_seed)
    if not matches:
        raise ValueError("No checkpoints match the requested configuration")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    runs, failures = [], []
    for path, ruleset, seed in matches:
        try:
            run = evaluate_rule_vector_intervention_checkpoint(
                path, ruleset, seed, args, device)
            report = args.input_dir / f"rule_vector_intervention_{run['aname']}.json"
            with report.open("w") as handle:
                json.dump(run, handle, indent=2, allow_nan=False)
            runs.append(run)
        except Exception as error:
            failures.append({"checkpoint": str(path), "error": str(error)})
            print(f"FAILED {path.name}: {error}")

    summary = summarize_rule_vector_intervention(runs)
    tag = f"{args.ruleset or 'all'}_hidden{args.hidden}_{args.feature}"
    if args.seed is not None:
        tag += f"_seed{args.seed}"
    elif args.total_seed is not None:
        tag += f"_n{args.total_seed}"
    report = args.input_dir / f"rule_vector_intervention_summary_{tag}.json"
    with report.open("w") as handle:
        json.dump({"settings": {
            "feature": args.feature,
            "hidden": args.hidden,
            "n_trials": args.n_trials,
            "batch_size": args.batch_size,
            "device": str(device),
            "ruleset": args.ruleset,
            "seed": args.seed,
            "total_seed": args.total_seed,
            "test_seed": args.test_seed,
            "n_random_rule_vectors": N_RANDOM_RULE_VECTORS,
        }, "runs": runs, "summary": summary, "failures": failures},
                  handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {report}")
    return runs, failures


def evaluate_rule_vector_magnitude_sweep_checkpoint(path, run_ruleset, seed,
                                                    args, device):
    """Sweep MemoryAnti cue-vector norm along fixed counterfactual directions."""
    import torch
    import _bootstrap  # noqa: F401
    import mpn
    import mpn_tasks

    aname = path.stem.removeprefix("savednet_")
    stage1 = load_task_params(args.checkpoint_dir, aname, "stage1")
    stage2 = load_task_params(args.checkpoint_dir, aname, "stage2")
    if stage1["rules"] != run_ruleset.split("_") or stage2["rules"] != ["delayanti"]:
        raise ValueError(f"{aname}: unexpected stage task configuration")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = mpn.DeepMultiPlasticNet(copy.deepcopy(checkpoint["net_params"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    params = copy.deepcopy(stage2)
    test_seed = args.test_seed + 2
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    params["hp"]["rng"] = np.random.RandomState(test_seed)
    params["hp"]["batch_size_train"] = args.n_trials
    (inputs, targets, masks), _ = mpn_tasks.generate_trials_wrap(
        params, args.n_trials, rules=["delayanti"], mode_input="random_batch",
        device="cpu", pretraining_shift=2, pretraining_shift_pre=0)
    if inputs.shape[-1] != model.W_initial_linear.in_features:
        raise ValueError(f"delayanti: input width {inputs.shape[-1]} != checkpoint")
    if not torch.all(inputs[:, 0, -3:].argmax(dim=-1) == 2):
        raise ValueError("delayanti: generated task cue is not in column 2")

    weight = model.W_initial_linear.weight
    original = weight[:, -1].detach().cpu().clone()
    pretrained = weight[:, -3:-1].detach().cpu().clone()
    components = _decompose_rule_vector(pretrained, original)
    original_norm = original.norm()
    directions = {
        "parallel": components["parallel_raw"] / components["parallel_raw"].norm(),
        "perpendicular": (components["perpendicular_raw"]
                          / components["perpendicular_raw"].norm()),
    }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int((args.test_seed + seed + 20_000) % (2**63 - 1)))
    random_directions = torch.randn(
        N_SWEEP_RANDOM_DIRECTIONS, original.numel(), generator=generator,
        dtype=original.dtype)
    random_directions /= random_directions.norm(dim=1, keepdim=True)

    def evaluate_vector(vector):
        with torch.no_grad():
            weight[:, -1].copy_(vector.to(device=device, dtype=weight.dtype))
        return _evaluate_inputs(model, inputs, targets, masks, inputs,
                                args.batch_size, device) * 100

    ratios = list(RULE_VECTOR_SWEEP_RATIOS)
    curves = {}
    try:
        original_accuracy = evaluate_vector(original)
        zero_accuracy = evaluate_vector(torch.zeros_like(original))
        for name, direction in directions.items():
            curves[name] = {
                "accuracy_pct": [
                    zero_accuracy if ratio == 0 else evaluate_vector(
                        direction * (ratio * original_norm))
                    for ratio in ratios
                ],
            }

        random_samples = []
        for direction in random_directions:
            random_samples.append([
                zero_accuracy if ratio == 0 else evaluate_vector(
                    direction * (ratio * original_norm))
                for ratio in ratios
            ])
        random_samples = np.asarray(random_samples, dtype=float)
        curves["random"] = {
            "accuracy_pct": random_samples.mean(axis=0).tolist(),
            "std_across_directions_pct": random_samples.std(axis=0).tolist(),
            "samples_pct": random_samples.tolist(),
            "n_directions": N_SWEEP_RANDOM_DIRECTIONS,
        }
    finally:
        with torch.no_grad():
            weight[:, -1].copy_(original.to(device=device, dtype=weight.dtype))

    geometry = {key: value for key, value in components.items()
                if not hasattr(value, "shape")}
    print(f"seed={seed} {run_ruleset} rule-vector magnitude sweep: "
          f"original={original_accuracy:.2f}%, "
          f"span/original={geometry['in_span_fraction']:.3f}")
    return {
        "aname": aname,
        "ruleset": run_ruleset,
        "seed": seed,
        "task": "delayanti",
        "n_trials": int(inputs.shape[0]),
        "test_seed": test_seed,
        "pretraining_rules": stage1["rules"],
        "norm_ratios": ratios,
        "original_accuracy_pct": original_accuracy,
        "geometry": geometry,
        "curves": curves,
    }


def summarize_rule_vector_magnitude_sweep(runs):
    """Aggregate magnitude-sweep curves across checkpoint seeds."""
    summary = {}
    for ruleset in GROUPS:
        selected = [run for run in runs if run["ruleset"] == ruleset]
        if not selected:
            continue
        ratios = np.asarray(selected[0]["norm_ratios"], dtype=float)
        if any(not np.array_equal(np.asarray(run["norm_ratios"], dtype=float), ratios)
               for run in selected[1:]):
            raise ValueError(f"{ruleset}: inconsistent magnitude-sweep ratios")

        original = np.asarray([run["original_accuracy_pct"] for run in selected],
                              dtype=float)
        span_ratio = np.asarray([run["geometry"]["in_span_fraction"]
                                 for run in selected], dtype=float)
        curves = {}
        for name in ("parallel", "perpendicular", "random"):
            values = np.asarray([run["curves"][name]["accuracy_pct"]
                                 for run in selected], dtype=float)
            curves[name] = {
                "mean_accuracy_pct": values.mean(axis=0).tolist(),
                "std_accuracy_pct": values.std(axis=0).tolist(),
            }
        summary[ruleset] = {
            "n_seeds": len(selected),
            "norm_ratios": ratios.tolist(),
            "original_accuracy_pct": {
                "mean": float(original.mean()),
                "std": float(original.std()),
            },
            "raw_span_norm_ratio": {
                "mean": float(span_ratio.mean()),
                "std": float(span_ratio.std()),
            },
            "curves": curves,
        }
    return summary


def plot_rule_vector_magnitude_sweep(runs, output_dir, feature, hidden):
    """Plot MemoryAnti accuracy versus replacement rule-vector norm."""
    if not runs:
        return None
    summary = summarize_rule_vector_magnitude_sweep(runs)
    styles = {
        "parallel": ("Pretraining-span direction", "#805ad5"),
        "perpendicular": ("Orthogonal direction", "#dd6b20"),
        "random": ("Random directions", "#718096"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True, sharey=True)
    for axis, (ruleset, (title, _)) in zip(axes, GROUPS.items()):
        selected = [run for run in runs if run["ruleset"] == ruleset]
        if selected:
            stats = summary[ruleset]
            ratios = np.asarray(stats["norm_ratios"], dtype=float)
            for name, (label, color) in styles.items():
                for run in selected:
                    axis.plot(ratios, run["curves"][name]["accuracy_pct"],
                              color=color, linewidth=0.6, alpha=0.12)
                mean = np.asarray(stats["curves"][name]["mean_accuracy_pct"])
                std = np.asarray(stats["curves"][name]["std_accuracy_pct"])
                axis.plot(ratios, mean, "o-", color=color, linewidth=1.8,
                          markersize=3.5, label=label)
                axis.fill_between(ratios, mean - std, mean + std,
                                  color=color, alpha=0.12, linewidth=0)

            original = stats["original_accuracy_pct"]
            axis.axhline(original["mean"], color="black", linestyle="--",
                         linewidth=1.1,
                         label=f"Original ({original['mean']:.1f}%)")
            span = stats["raw_span_norm_ratio"]
            axis.axvline(span["mean"], color=styles["parallel"][1],
                         linestyle=":", linewidth=1.2,
                         label=f"Raw span norm ({span['mean']:.2f} ± {span['std']:.2f})")
            axis.legend(fontsize=7, frameon=False, loc="best")
        axis.set_title(title)
        axis.set_xlim(min(RULE_VECTOR_SWEEP_RATIOS),
                      max(RULE_VECTOR_SWEEP_RATIOS))
        axis.set_ylim(-5, 105)
        axis.set_xlabel("Replacement norm / original MemoryAnti norm")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MemoryAnti accuracy (%)")
    fig.suptitle(f"Rule-vector magnitude sweep | hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = (runs[0]["aname"] if len(runs) == 1
           else f"{'_'.join(summary)}_hidden{hidden}_{feature}")
    path = _figure_path(
        output_dir, f"rule_vector_magnitude_sweep_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def run_rule_vector_magnitude_sweep(args):
    """Run final-checkpoint rule-vector magnitude sweeps."""
    import torch

    matches = discover_checkpoints(
        args.checkpoint_dir, args.feature, args.hidden, args.ruleset, args.seed,
        args.total_seed, args.test_seed)
    if not matches:
        raise ValueError("No checkpoints match the requested configuration")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    runs, failures = [], []
    for path, ruleset, seed in matches:
        try:
            run = evaluate_rule_vector_magnitude_sweep_checkpoint(
                path, ruleset, seed, args, device)
            report = args.input_dir / f"rule_vector_magnitude_sweep_{run['aname']}.json"
            with report.open("w") as handle:
                json.dump(run, handle, indent=2, allow_nan=False)
            runs.append(run)
        except Exception as error:
            failures.append({"checkpoint": str(path), "error": str(error)})
            print(f"FAILED {path.name}: {error}")

    summary = summarize_rule_vector_magnitude_sweep(runs)
    tag = f"{args.ruleset or 'all'}_hidden{args.hidden}_{args.feature}"
    if args.seed is not None:
        tag += f"_seed{args.seed}"
    elif args.total_seed is not None:
        tag += f"_n{args.total_seed}"
    report = args.input_dir / f"rule_vector_magnitude_sweep_summary_{tag}.json"
    with report.open("w") as handle:
        json.dump({"settings": {
            "feature": args.feature,
            "hidden": args.hidden,
            "n_trials": args.n_trials,
            "batch_size": args.batch_size,
            "device": str(device),
            "ruleset": args.ruleset,
            "seed": args.seed,
            "total_seed": args.total_seed,
            "test_seed": args.test_seed,
            "norm_ratios": list(RULE_VECTOR_SWEEP_RATIOS),
            "n_random_directions": N_SWEEP_RANDOM_DIRECTIONS,
        }, "runs": runs, "summary": summary, "failures": failures},
                  handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {report}")
    return runs, failures


N_BACKBONE_RANDOM_VECTORS = 10
BACKBONE_GRID_COEFFS = tuple(np.round(np.linspace(-2.0, 2.0, 17), 4).tolist())
BACKBONE_NAMED_POINTS = {
    "zero": (0.0, 0.0),
    "pre0_cue": (1.0, 0.0),
    "pre1_cue": (0.0, 1.0),
    "cue_sum": (1.0, 1.0),
}


def load_stage1_train_params(checkpoint_dir, run_ruleset, seed, hidden, feature):
    """Load stage-1 train_params; the config filename omits the network name."""
    path = (checkpoint_dir / f"param_{run_ruleset}_seed{seed}_"
            f"+hidden{hidden}+{feature}+batch128+angle_param.json")
    with path.open() as handle:
        return json.load(handle)["train_params"]


def evaluate_backbone_probe_checkpoint(path, run_ruleset, seed, args, device):
    """Zero-shot MemoryAnti probes of the frozen backbone, no training involved.

    Replacing the last input column of the final checkpoint recovers the
    end-of-stage-1 backbone under a counterfactual rule input, since stage 2
    trains only that column. Random Kaiming-initialized rule vectors provide
    the negative control; training-style loss uses the saved regularization
    settings. The span grid tests whether a combination a*v_pre0 + b*v_pre1
    of the two pretrained rule vectors solves MemoryAnti without additional
    training. Named points identify interpretable
    combinations (single cues, cue sum, the learned vector's least-squares
    projection into the span). Nonzero coefficient pairs are also evaluated
    at the mean of the two pretrained cue norms for that checkpoint; positive
    coefficient rays share cached evaluations. The origin reuses the raw
    zero-vector accuracy. The learned vector is separately rescaled to the
    same target norm. All conditions share one fresh trial batch and
    the angle accuracy scoring used by the other experiments.
    """
    import math
    import torch
    import _bootstrap  # noqa: F401
    import mpn
    import mpn_tasks

    aname = path.stem.removeprefix("savednet_")
    stage1 = load_task_params(args.checkpoint_dir, aname, "stage1")
    stage2 = load_task_params(args.checkpoint_dir, aname, "stage2")
    if stage1["rules"] != run_ruleset.split("_") or stage2["rules"] != ["delayanti"]:
        raise ValueError(f"{aname}: unexpected stage task configuration")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = mpn.DeepMultiPlasticNet(copy.deepcopy(checkpoint["net_params"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    # compute_loss reads regularization attributes normally set by net.fit;
    # load_state_dict does not restore them, so re-inject from the stage-1
    # config to keep random-probe losses comparable to training-time loss.
    train_params = load_stage1_train_params(
        args.checkpoint_dir, run_ruleset, seed, args.hidden, args.feature)
    model.weight_reg = train_params.get("weight_reg", None)
    model.reg_lambda = train_params.get("reg_lambda", 0.0)
    model.activity_reg = train_params.get("activity_reg", None)
    model.reg_omit = train_params.get("reg_omit", [])
    model.gradient_type = train_params.get("gradient_type", "backprop")

    params = copy.deepcopy(stage2)
    test_seed = args.test_seed + 2  # delayanti is rule column 2, matching evaluate_task
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    params["hp"]["rng"] = np.random.RandomState(test_seed)
    params["hp"]["batch_size_train"] = args.n_trials
    (inputs, targets, masks), _ = mpn_tasks.generate_trials_wrap(
        params, args.n_trials, rules=["delayanti"], mode_input="random_batch",
        device="cpu", pretraining_shift=2, pretraining_shift_pre=0)
    if inputs.shape[-1] != model.W_initial_linear.in_features:
        raise ValueError(f"delayanti: input width {inputs.shape[-1]} != checkpoint")
    if not torch.all(inputs[:, 0, -3:].argmax(dim=-1) == 2):
        raise ValueError("delayanti: generated task cue is not in column 2")

    weight = model.W_initial_linear.weight
    original = weight[:, -1].detach().cpu().clone()
    v_pre0 = weight[:, -3].detach().cpu().clone()
    v_pre1 = weight[:, -2].detach().cpu().clone()

    basis = torch.stack([v_pre0, v_pre1], dim=1)
    coefficients = torch.linalg.lstsq(basis, original.unsqueeze(1)).solution.squeeze(1)
    projection = basis @ coefficients
    original_norm = float(original.norm())
    if original_norm == 0:
        raise ValueError("Learned MemoryAnti rule vector has zero norm")
    geometry = {
        "learned_coeff_pre0": float(coefficients[0]),
        "learned_coeff_pre1": float(coefficients[1]),
        "in_span_fraction": float(projection.norm()) / original_norm,
        "norm_pre0": float(v_pre0.norm()),
        "norm_pre1": float(v_pre1.norm()),
        "norm_original": original_norm,
        "cos_pre0_pre1": float(
            torch.dot(v_pre0, v_pre1)
            / (v_pre0.norm() * v_pre1.norm()).clamp(min=1e-12)),
    }

    def evaluate_vector(vector):
        with torch.no_grad():
            weight[:, -1].copy_(vector.to(device=device, dtype=weight.dtype))
        return _evaluate_inputs(model, inputs, targets, masks, inputs,
                                args.batch_size, device) * 100

    def evaluate_loss_and_accuracy(vector):
        """Full-batch forward for training-style loss plus the shared accuracy."""
        with torch.no_grad():
            weight[:, -1].copy_(vector.to(device=device, dtype=weight.dtype))
            outputs, hidden, _ = model.iterate_sequence_batch(
                inputs.to(device), run_mode="minimal")
            loss, loss_components, _ = model.compute_loss(
                outputs, targets.to(device), masks.to(device), hidden=hidden)
            accuracy, _ = model.compute_acc(
                outputs, targets.to(device), masks.to(device), inputs.to(device),
                isvalid=True, mode="angle")
        values = (float(loss), float(loss_components[0]), float(accuracy) * 100)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Non-finite random-probe loss or accuracy")
        return values

    try:
        original_accuracy = evaluate_vector(original)

        rng = np.random.default_rng(args.test_seed + seed)
        random_probe = {"loss": [], "loss_out": [], "accuracy_pct": [],
                        "n_vectors": N_BACKBONE_RANDOM_VECTORS}
        for _ in range(N_BACKBONE_RANDOM_VECTORS):
            new_column = torch.empty_like(original)
            torch.manual_seed(int(rng.integers(0, 2**31)))
            torch.nn.init.kaiming_uniform_(new_column.view(-1, 1).t(), a=math.sqrt(5))
            loss, loss_out, accuracy = evaluate_loss_and_accuracy(new_column)
            random_probe["loss"].append(loss)
            random_probe["loss_out"].append(loss_out)
            random_probe["accuracy_pct"].append(accuracy)

        named_points = {"original": {"a": None, "b": None,
                                     "accuracy_pct": original_accuracy}}
        named_coefficients = dict(BACKBONE_NAMED_POINTS)
        named_coefficients["learned_projection"] = (
            geometry["learned_coeff_pre0"], geometry["learned_coeff_pre1"])
        for name, (a, b) in named_coefficients.items():
            named_points[name] = {
                "a": a, "b": b,
                "accuracy_pct": evaluate_vector(a * v_pre0 + b * v_pre1)}

        grid = [[evaluate_vector(a * v_pre0 + b * v_pre1)
                 for b in BACKBONE_GRID_COEFFS]
                for a in BACKBONE_GRID_COEFFS]

        # Match input strength within this checkpoint using its mean pretrained
        # cue norm. Positive coefficient rays share one rescaled vector and
        # cached accuracy; the origin retains the raw zero-vector result.
        target_norm = float((v_pre0.norm() + v_pre1.norm()) / 2)
        direction_cache = {}

        def evaluate_direction(a, b):
            scale = math.hypot(a, b)
            if scale < 1e-12:
                return named_points["zero"]["accuracy_pct"]
            key = (round(a / scale, 9), round(b / scale, 9))
            if key not in direction_cache:
                vector = a * v_pre0 + b * v_pre1
                direction_cache[key] = evaluate_vector(
                    vector * (target_norm / float(vector.norm())))
            return direction_cache[key]

        named_points_norm_matched = {
            name: {"a": a, "b": b, "accuracy_pct": evaluate_direction(a, b)}
            for name, (a, b) in named_coefficients.items() if (a, b) != (0.0, 0.0)}
        named_points_norm_matched["learned_direction"] = {
            "a": None, "b": None,
            "accuracy_pct": evaluate_vector(original * (target_norm / original.norm()))}

        grid_norm_matched = [[evaluate_direction(a, b)
                              for b in BACKBONE_GRID_COEFFS]
                             for a in BACKBONE_GRID_COEFFS]
    finally:
        with torch.no_grad():
            weight[:, -1].copy_(original.to(device=device, dtype=weight.dtype))

    def _grid_summary(values):
        array = np.asarray(values, dtype=float)
        best = np.unravel_index(int(array.argmax()), array.shape)
        return array, {"accuracy_pct": float(array.max()),
                       "a": float(BACKBONE_GRID_COEFFS[best[0]]),
                       "b": float(BACKBONE_GRID_COEFFS[best[1]])}

    grid_array, grid_max = _grid_summary(grid)
    grid_nm_array, grid_max_norm_matched = _grid_summary(grid_norm_matched)

    print(f"seed={seed} {run_ruleset} backbone probe (delayanti):")
    print(f"  original: accuracy={original_accuracy:.2f}")
    for name in (*named_coefficients, "grid max"):
        stats = grid_max if name == "grid max" else named_points[name]
        print(f"  {name} (a={stats['a']:+.2f}, b={stats['b']:+.2f}): "
              f"accuracy={stats['accuracy_pct']:.2f}")
    print(f"  random mean: accuracy="
          f"{np.mean(random_probe['accuracy_pct']):.2f}")
    print(f"  norm matched to {target_norm:.3f}: "
          + ", ".join(f"{name}={stats['accuracy_pct']:.2f}"
                      for name, stats in named_points_norm_matched.items())
          + f", grid max={grid_max_norm_matched['accuracy_pct']:.2f}")
    return {
        "aname": aname,
        "ruleset": run_ruleset,
        "seed": seed,
        "task": "delayanti",
        "n_trials": int(inputs.shape[0]),
        "test_seed": test_seed,
        "pretraining_rules": stage1["rules"],
        "geometry": geometry,
        "random_probe": random_probe,
        "named_points": named_points,
        "span_grid": {"coefficients": list(BACKBONE_GRID_COEFFS),
                      "accuracy_pct": grid_array.tolist()},
        "grid_max": grid_max,
        "named_points_norm_matched": named_points_norm_matched,
        "span_grid_norm_matched": {"coefficients": list(BACKBONE_GRID_COEFFS),
                                   "target_norm": target_norm,
                                   "accuracy_pct": grid_nm_array.tolist()},
        "grid_max_norm_matched": grid_max_norm_matched,
    }


def _backbone_condition_value(run, key):
    """Accuracy of one probe condition, including the two derived ones."""
    if key == "grid_max":
        return run["grid_max"]["accuracy_pct"]
    if key == "random":
        return float(np.mean(run["random_probe"]["accuracy_pct"]))
    return run["named_points"][key]["accuracy_pct"]


def summarize_backbone_probe(runs):
    """Aggregate zero-shot probe accuracies and losses over checkpoint seeds."""
    def stats(values):
        values = np.asarray(values, dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std())}

    summary = {}
    for ruleset in GROUPS:
        selected = [run for run in runs if run["ruleset"] == ruleset]
        if not selected:
            continue
        entry = {"n_seeds": len(selected)}
        for key in ("original", *BACKBONE_NAMED_POINTS, "learned_projection",
                    "grid_max", "random"):
            entry[f"{key}_accuracy_pct"] = stats(
                [_backbone_condition_value(run, key) for run in selected])
        for key in ("loss", "loss_out"):
            entry[f"random_{key}"] = stats(
                [np.mean(run["random_probe"][key]) for run in selected])
        for key in ("in_span_fraction", "learned_coeff_pre0", "learned_coeff_pre1"):
            entry[key] = stats([run["geometry"][key] for run in selected])
        # Norm-matched fields exist only in results produced after they were
        # added; older JSONs are summarized without them.
        matched = [run for run in selected if "named_points_norm_matched" in run]
        if matched:
            entry["n_seeds_norm_matched"] = len(matched)
            for name in matched[0]["named_points_norm_matched"]:
                entry[f"{name}_norm_matched_accuracy_pct"] = stats(
                    [run["named_points_norm_matched"][name]["accuracy_pct"]
                     for run in matched])
            entry["grid_max_norm_matched_accuracy_pct"] = stats(
                [run["grid_max_norm_matched"]["accuracy_pct"] for run in matched])
        summary[ruleset] = entry
    return summary


def run_backbone_probe(args):
    """Run the zero-shot backbone probes on every matching final checkpoint."""
    import torch

    matches = discover_checkpoints(
        args.checkpoint_dir, args.feature, args.hidden, args.ruleset, args.seed,
        args.total_seed, args.test_seed)
    if not matches:
        raise ValueError("No checkpoints match the requested configuration")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    runs, failures = [], []
    for path, ruleset, seed in matches:
        try:
            run = evaluate_backbone_probe_checkpoint(path, ruleset, seed, args, device)
            with (args.input_dir / f"backbone_probe_{run['aname']}.json").open("w") as handle:
                json.dump(run, handle, indent=2, allow_nan=False)
            runs.append(run)
        except Exception as error:
            failures.append({"checkpoint": str(path), "error": str(error)})
            print(f"FAILED {path.name}: {error}")

    summary = summarize_backbone_probe(runs)
    tag = f"{args.ruleset or 'all'}_hidden{args.hidden}_{args.feature}"
    if args.seed is not None:
        tag += f"_seed{args.seed}"
    elif args.total_seed is not None:
        tag += f"_n{args.total_seed}"
    report = args.input_dir / f"backbone_probe_summary_{tag}.json"
    with report.open("w") as handle:
        json.dump({"settings": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                   "runs": runs, "summary": summary, "failures": failures},
                  handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {report}")
    return runs, failures


def _backbone_figure_tag(runs, summary, hidden, feature):
    return (runs[0]["aname"] if len(runs) == 1
            else f"{'_'.join(summary)}_hidden{hidden}_{feature}")


def plot_backbone_probe_random(runs, output_dir, feature, hidden):
    """Random-rule probe loss/accuracy per motif, seed-mean over random inits.

    Show total loss, output-only MSE, and accuracy as one boxplot per motif
    with per-seed dots. The output-only panel separates prediction error from
    the regularization contribution included in total loss.
    """
    if not runs:
        return None
    panels = (("Total loss (output + reg)", "loss"),
              ("Output MSE (label only)", "loss_out"),
              ("MemoryAnti accuracy (%)", "accuracy_pct"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for axis, (title, key) in zip(axes, panels):
        positions, labels = [], []
        for position, (ruleset, (group_title, _)) in enumerate(GROUPS.items()):
            values = np.asarray([np.mean(run["random_probe"][key]) for run in runs
                                 if run["ruleset"] == ruleset], dtype=float)
            positions.append(position)
            labels.append(f"{group_title}\n(n={values.size})")
            if values.size == 0:
                continue
            axis.boxplot([values], positions=[position], widths=0.5,
                         showfliers=False)
            jitter = (np.linspace(-0.1, 0.1, values.size)
                      if values.size > 1 else np.zeros(1))
            axis.scatter(position + jitter, values, color=COLORS[position],
                         s=18, alpha=0.75, zorder=3)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_title(title, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[2].set_ylim(-2, 102)
    fig.suptitle(f"Random-rule backbone probe on MemoryAnti | "
                 f"hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = _backbone_figure_tag(runs, summarize_backbone_probe(runs), hidden, feature)
    path = _figure_path(output_dir, f"backbone_probe_random_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_backbone_span_grid(runs, output_dir, feature, hidden, norm_matched=False):
    """Seed-mean zero-shot accuracy over the pretraining rule span, per motif.

    Heatmap axes are the coefficients (a, b) of a*v_pre0 + b*v_pre1 replacing
    the MemoryAnti rule vector with no stage-2 training. Overlaid markers show
    the single cues, the cue sum, and each seed's learned-vector projection
    into the span (reported to the console when it falls outside the grid).
    With norm_matched=True, nonzero combinations use each checkpoint's mean
    pretrained cue norm; the origin remains a zero-vector control. Positive
    coefficient rays share accuracy, excluding the origin. Input strength is
    matched within checkpoints, not necessarily across them. Saved grids must
    match BACKBONE_GRID_COEFFS; rerun --backbone-probe after changing the grid.
    """
    grid_key = "span_grid_norm_matched" if norm_matched else "span_grid"
    runs = [run for run in runs if grid_key in run]
    if not runs:
        if norm_matched:
            print("Skipped norm-matched span-grid plot: rerun --backbone-probe "
                  "to collect norm-matched grids")
        return None
    coefficients = np.asarray(BACKBONE_GRID_COEFFS, dtype=float)
    half_step = (coefficients[1] - coefficients[0]) / 2
    extent = (coefficients[0] - half_step, coefficients[-1] + half_step,
              coefficients[0] - half_step, coefficients[-1] + half_step)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharex=True, sharey=True,
                             constrained_layout=True)
    image = None
    for axis, (ruleset, (title, tasks)) in zip(axes, GROUPS.items()):
        selected = [run for run in runs if run["ruleset"] == ruleset]
        axis.set_title(f"{title} (n={len(selected)})", fontsize=9)
        axis.set_xlabel(f"a ({_display_rule(tasks[0])} rule vector)")
        if not selected:
            continue
        for run in selected:
            if not np.allclose(run[grid_key]["coefficients"], coefficients):
                raise ValueError(f"{run['aname']}: inconsistent span-grid coefficients")
        grids = np.asarray([run[grid_key]["accuracy_pct"] for run in selected],
                           dtype=float)
        # Row index is a, column index is b; transpose puts a on the x axis.
        image = axis.imshow(grids.mean(axis=0).T, origin="lower", extent=extent,
                            vmin=0, vmax=100, cmap="viridis", aspect="equal")
        first_panel = axis is axes[0]
        for name, marker in (("pre0_cue", "^"), ("pre1_cue", "s"), ("cue_sum", "P")):
            point = selected[0]["named_points"][name]
            axis.scatter(point["a"], point["b"], marker=marker, s=45,
                         facecolors="white", edgecolors="black", linewidths=0.8,
                         label=name if first_panel else None, zorder=3)
        projections = np.asarray(
            [[run["named_points"]["learned_projection"]["a"],
              run["named_points"]["learned_projection"]["b"]] for run in selected],
            dtype=float)
        outside = np.abs(projections) > extent[1]
        if outside.any():
            print(f"  Note: {ruleset}: {int(outside.any(axis=1).sum())} learned "
                  f"projection(s) fall outside the plotted grid; see the JSONs.")
        axis.scatter(projections[:, 0], projections[:, 1], marker="o", s=22,
                     facecolors="#e53e3e", edgecolors="white", linewidths=0.6,
                     label="learned projection" if first_panel else None, zorder=3)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
    axes[0].set_ylabel(
        f"b ({_display_rule(GROUPS['fdanti_delaygo'][1][1])} rule vector)")
    axes[0].legend(fontsize=6, loc="upper left", frameon=True, framealpha=0.85)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.85,
                     label="Zero-shot MemoryAnti accuracy (%)")
    variant = ("norm matched to mean pretrained cue norm" if norm_matched
               else "raw combinations")
    fig.suptitle(f"Pretraining-span rule combinations ({variant}), "
                 f"no stage-2 training | hidden{hidden} | {feature}")
    tag = _backbone_figure_tag(runs, summarize_backbone_probe(runs), hidden, feature)
    prefix = ("backbone_span_grid_norm_matched" if norm_matched
              else "backbone_span_grid")
    path = _figure_path(output_dir, f"{prefix}_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_backbone_conditions(runs, output_dir, feature, hidden):
    """Raw zero-shot MemoryAnti probe accuracies, one panel per motif.

    Best-grid accuracy is a maximum over the grid evaluations and therefore
    carries selection bias; named combinations are not chosen by that maximum.
    Norm-matched results are shown in the separate norm-matched span-grid plot.
    """
    if not runs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for axis, (ruleset, (title, tasks)) in zip(axes, GROUPS.items()):
        selected = [run for run in runs if run["ruleset"] == ruleset]
        conditions = (
            ("original", "Learned\nvector"),
            ("learned_projection", "Learned\nprojection"),
            ("grid_max", "Best grid\ncombination"),
            ("cue_sum", f"{_display_rule(tasks[0])} +\n{_display_rule(tasks[1])}"),
            ("pre0_cue", f"{_display_rule(tasks[0])}\nonly"),
            ("pre1_cue", f"{_display_rule(tasks[1])}\nonly"),
            ("zero", "Zero\nvector"),
            ("random", "Random\n(mean)"),
        )
        positions = np.arange(len(conditions))
        for run in selected:
            values = [_backbone_condition_value(run, key) for key, _ in conditions]
            axis.plot(positions, values, color="#805ad5", linewidth=0.7, alpha=0.18)
        if selected:
            matrix = np.asarray([[_backbone_condition_value(run, key)
                                  for key, _ in conditions] for run in selected],
                                dtype=float)
            axis.errorbar(positions, matrix.mean(axis=0), yerr=matrix.std(axis=0),
                          fmt="o-", color="#805ad5", linewidth=1.5, markersize=4,
                          capsize=2, label=f"MemoryAnti (n={len(selected)})")
            axis.legend(fontsize=8, frameon=False)
        axis.set_title(title)
        axis.set_ylim(-5, 105)
        axis.set_xticks(positions)
        axis.set_xticklabels([label for _, label in conditions], fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"Zero-shot MemoryAnti on the frozen backbone | "
                 f"hidden{hidden} | {feature}")
    fig.tight_layout()
    tag = _backbone_figure_tag(runs, summarize_backbone_probe(runs), hidden, feature)
    path = _figure_path(output_dir, f"backbone_probe_conditions_{tag}.png", seed_specific=len(runs) == 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def load_accuracies(input_dir, feature, hidden, ruleset=None, seed=None):
    pattern = re.compile(
        rf"accuracy_({'|'.join(GROUPS)})_dmpn_seed(\d+)_"
        rf"\+hidden{hidden}\+{re.escape(feature)}\+batch128\+angle\.json"
    )
    grouped = {ruleset: {task: [] for task in tasks}
               for ruleset, (_, tasks) in GROUPS.items()}
    for path in sorted(input_dir.glob("accuracy_*.json")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        if ruleset is not None and match.group(1) != ruleset:
            continue
        if seed is not None and int(match.group(2)) != seed:
            continue
        run_ruleset = match.group(1)
        with path.open() as handle:
            result = json.load(handle)
        if result["ruleset"] != run_ruleset or result["seed"] != int(match.group(2)):
            raise ValueError(f"{path}: metadata does not match filename")
        seen = set()
        for entry in result["tasks"]:
            task = entry["task"]
            if task not in grouped[run_ruleset] or task in seen:
                raise ValueError(f"{path}: unexpected or duplicate task {task}")
            seen.add(task)
            value = float(entry["accuracy_pct"])
            if not np.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"{path}: invalid accuracy {value}")
            grouped[run_ruleset][task].append(value)
    return grouped


def plot_accuracies(grouped, output_dir, feature, hidden, *, seed=None, ruleset=None):
    if not any(values for tasks in grouped.values() for values in tasks.values()):
        return None
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for axis, (group_ruleset, (title, tasks)) in zip(axes, GROUPS.items()):
        labels = []
        for index, (task, color) in enumerate(zip(tasks, COLORS)):
            values = np.asarray(grouped[group_ruleset][task], dtype=float)
            labels.append(f"{_display_rule(task)}\nn={values.size}")
            if values.size:
                axis.bar(index, values.mean(), yerr=values.std(), width=0.6,
                         color=color, alpha=0.7, capsize=3,
                         error_kw={"elinewidth": 1})
                jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.zeros(1)
                axis.scatter(index + jitter, values, color="black", s=12,
                             alpha=0.6, zorder=3)
            else:
                print(f"Missing accuracy: {group_ruleset}/{task}")
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(3))
        axis.set_xticklabels(labels)
        axis.set_xlim(-0.5, 2.5)
        axis.set_ylim(0, 110)
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"Final checkpoint accuracy | hidden{hidden} | {feature}", fontsize=10)
    fig.tight_layout()
    tag = f"hidden{hidden}_{feature}"
    if seed is not None:
        tag = f"{ruleset or 'all'}_seed{seed}_{tag}"
    path = _figure_path(output_dir, f"accuracy_by_motif_{tag}.png", seed_specific=seed is not None)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def _record_pca_periods(model, inputs, epochs, device, include_memory, include_response=False):
    """Retain only required periods, recording M before its timestep update."""
    import torch

    periods = {"stim1": epochs["stim1"]}
    if include_memory:
        periods["delay1"] = epochs["delay1"]
    if include_response:
        periods["go1"] = epochs["go1"]
    weight = model.mp_layer1.W.detach()
    recorded = {
        representation: {
            period: np.empty((len(inputs), end - start, width), dtype=np.float32)
            for period, (start, end) in periods.items()
        }
        for representation, width in [("hidden", model.n_hidden),
                                      ("effective_modulation", weight.numel())]
    }
    with torch.no_grad():
        for batch_start in range(0, len(inputs), 8):
            batch = inputs[batch_start:batch_start + 8].to(device)
            model.reset_state(B=len(batch))
            for timestep in range(max(end for _, end in periods.values())):
                _, _, state = model.network_step(batch[:, timestep], run_mode="track_states", seq_idx=timestep)
                for period, (start, end) in periods.items():
                    if start <= timestep < end:
                        selection = (slice(batch_start, batch_start + len(batch)), timestep - start)
                        recorded["hidden"][period][selection] = state["hidden1"].cpu().numpy()
                        recorded["effective_modulation"][period][selection] = (
                            state["M1"] * weight
                        ).flatten(start_dim=1).cpu().numpy()
    return recorded


def _plot_memory_representation(recorded, representation, first_rule, motif_title,
                                seed, aname, output_dir, period="stimulus", pca=None):
    """Fit or reuse a representation's MemoryAnti memory PCA and plot trajectories.

    The stimulus call also saves response trajectories using the same fitted PCA.
    """
    from sklearn.decomposition import PCA

    if period not in ("stimulus", "response"):
        raise ValueError(f"Unknown period: {period}")
    if pca is None:
        memory = recorded["delayanti"]["states"][representation]["delay1"]
        basis_data = memory.reshape(-1, memory.shape[-1])
        solver = "full" if representation == "hidden" else "randomized"
        pca = PCA(n_components=2, svd_solver=solver, random_state=0).fit(basis_data)
        if pca.singular_values_[1] <= np.finfo(basis_data.dtype).eps * max(basis_data.shape) * pca.singular_values_[0]:
            raise ValueError(f"MemoryAnti {representation} memory states do not support two nonzero PCs")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True)
    if period == "response":
        first_rule = "delaygo"
    first_title = _display_rule(first_rule)
    period_key = "stim1" if period == "stimulus" else "go1"
    for axis, (rule, title) in zip(axes, [(first_rule, first_title), ("delayanti", "MemoryAnti")]):
        entry = recorded[rule]
        activity = entry["states"][representation][period_key]
        for direction in range(entry["n_directions"]):
            selected = entry["directions"] == direction
            if not selected.any():
                raise ValueError(f"{rule}: no trials for stimulus direction {direction}")
            trajectory = pca.transform(activity[selected].mean(axis=0))
            color = STIMULUS_COLORS[direction % len(STIMULUS_COLORS)]
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.5,
                      label=f"{360 * direction / entry['n_directions']:g} deg")
            axis.scatter(*trajectory[0], facecolors="none", edgecolors=[color], s=28, zorder=3)
            axis.scatter(*trajectory[-1], color=[color], marker=">", s=28, zorder=3)
        axis.set_title(f"{title}: {period}-period trajectory", fontsize=9)
        axis.set_xlabel(f"Projection onto MemoryAnti memory PC1\n({pca.explained_variance_ratio_[0]:.1%} of memory-period variance)", fontsize=8)
        axis.set_aspect("equal", adjustable="box")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"Projection onto MemoryAnti memory PC2\n({pca.explained_variance_ratio_[1]:.1%} of memory-period variance)", fontsize=8)
    axes[1].legend(title="Stimulus", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    label = "Hidden" if representation == "hidden" else "Effective modulation (W * M)"
    fig.suptitle(f"{motif_title} | seed {seed} | {label}\nPCA fitted to MemoryAnti memory-period activity", fontsize=10)
    fig.tight_layout()
    prefix = f"memory_pca_{period}" if representation == "hidden" else f"memory_pca_{period}_effective_modulation"
    path = _figure_path(output_dir, f"{prefix}_{aname}.png", seed_specific=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}", flush=True)
    if period == "stimulus":
        _plot_memory_representation(recorded, representation, first_rule, motif_title,
                                    seed, aname, output_dir, period="response", pca=pca)
    return path


def plot_memory_pca(checkpoint_dir, output_dir, feature, hidden, seed=None, test_seed=0,
                    ruleset="fdgo_delaygo"):
    """Project stimulus and MemoryPro/MemoryAnti response paths in delay1 PCA.

    Fit two PCs to flattened trial/time samples from MemoryAnti's memory
    period, without feature scaling; transform direction-averaged trajectories
    using the fitted mean. Unless seed is supplied, select a checkpoint with
    SystemRandom; test_seed controls trial generation, not checkpoint selection.
    """
    import random
    import torch
    import _bootstrap  # noqa: F401
    import mpn
    import mpn_tasks

    matches = discover_checkpoints(checkpoint_dir, feature, hidden, ruleset, seed)
    if not matches:
        raise ValueError(f"No matching {ruleset} checkpoint")
    path, _, seed = random.SystemRandom().choice(matches)
    aname = path.stem.removeprefix("savednet_")
    print(f"Selected checkpoint: {aname}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = mpn.DeepMultiPlasticNet(copy.deepcopy(checkpoint["net_params"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    recorded = {}
    motif_title, tasks = GROUPS[ruleset]
    first_rule = tasks[0]
    for rule, stage, column in [(first_rule, "stage1", 0), ("delaygo", "stage1", 1), ("delayanti", "stage2", 2)]:
        params = load_task_params(checkpoint_dir, aname, stage)
        np.random.seed(test_seed)
        torch.manual_seed(test_seed)
        params["hp"]["rng"] = np.random.RandomState(test_seed)
        params["hp"]["batch_size_train"] = 256
        (inputs, _, _), (_, trials, _) = mpn_tasks.generate_trials_wrap(
            params, 256, rules=[rule], mode_input="random", device="cpu",
            pretraining_shift=2 if column == 2 else 0,
            pretraining_shift_pre=1 if column < 2 else 0,
        )
        if inputs.shape[-1] != model.W_initial_linear.in_features:
            raise ValueError(f"{rule}: checkpoint input width mismatch")
        if not torch.all(inputs[:, 0, -3:].argmax(dim=-1) == column):
            raise ValueError(f"{rule}: incorrect rule cue")
        recorded[rule] = {
            "states": _record_pca_periods(model, inputs, trials[0].epochs, device,
                                          column == 2, include_response=column > 0),
            "directions": np.asarray(trials[0].meta["stim1"]).reshape(-1),
            "epochs": trials[0].epochs, "dt": params["dt"],
            "n_directions": params["n_eachring"],
        }
        start, end = recorded[rule]["epochs"]["stim1"]
        print(f"{rule}: stimulus duration {(end - start) * params['dt']} ms", flush=True)
        if column > 0:
            start, end = recorded[rule]["epochs"]["go1"]
            print(f"{rule}: response duration {(end - start) * params['dt']} ms", flush=True)
    return [_plot_memory_representation(recorded, representation, first_rule,
                                        motif_title, seed, aname, output_dir)
            for representation in ("hidden", "effective_modulation")]


EXPERIMENTS = ("accuracy", "memory_pca", "rule_cue", "m_intervention")
EXPLICIT_ONLY_EXPERIMENTS = ("rule_vector_intervention",
                             "rule_vector_magnitude_sweep", "backbone_probe",
                             "pathway_gain", "plot_only")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(input_dir=ANALYSIS_DIR, output_dir=FIGURE_DIR,
                        checkpoint_dir=CHECKPOINT_DIR)
    parser.add_argument("--feature", default="L21e3")
    parser.add_argument("--hidden", type=positive_int, default=200)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--accuracy", action="store_true",
                       help="Evaluate fresh checkpoint accuracy only and plot it.")
    modes.add_argument("--memory-pca", action="store_true",
                       help="Run memory-PCA trajectory analysis only.")
    modes.add_argument("--plot-only", action="store_true",
                       help="Only plot cached accuracy JSON; do not run experiments.")
    modes.add_argument("--rule-cue", action="store_true",
                       help="Evaluate paired cue timing, including response-only cues, and plot accuracy/errors.")
    modes.add_argument("--m-intervention", action="store_true",
                       help="Causally scale/swap the cue-attributable plastic trace at response onset.")
    modes.add_argument("--rule-vector-intervention", action="store_true",
                       help="Causally decompose the final MemoryAnti rule vector without retraining.")
    modes.add_argument("--rule-vector-magnitude-sweep", action="store_true",
                       help="Sweep MemoryAnti replacement-vector norm along fixed directions.")
    modes.add_argument("--backbone-probe", action="store_true",
                       help="Zero-shot MemoryAnti probes: random rule vectors and "
                           "raw and norm-matched pretraining-span (a, b) grids.")
    modes.add_argument("--pathway-gain", action="store_true",
                       help="Sign of the stimulus-to-readout gain through W_eff "
                            "under cue/trace conditions, plus 1+M statistics "
                            "on the top pathway synapses.")
    parser.add_argument("--n-trials", type=positive_int, default=200,
                        help="Trials per task for evaluations; memory PCA uses a fixed 256.")
    parser.add_argument("--batch-size", type=positive_int, default=8,
                        help="Evaluation batch size; memory PCA uses a fixed 8.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Device for model evaluations; memory PCA selects automatically.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-seed", type=positive_int, default=None,
                        help="Randomly select K matching checkpoint seeds per motif.")
    parser.add_argument("--test-seed", type=int, default=0)
    parser.add_argument("--ruleset", choices=tuple(GROUPS), default=None)
    args = parser.parse_args(argv)
    if not 0 <= args.test_seed <= 2**32 - 3:
        parser.error("--test-seed must be between 0 and 2**32 - 3")
    if args.seed is not None and args.total_seed is not None:
        parser.error("--seed and --total-seed cannot be used together")
    if args.total_seed is not None and (args.memory_pca or args.plot_only):
        parser.error("--total-seed applies to checkpoint-batch analyses, not "
                     "--memory-pca or --plot-only")
    selected = [name for name in (*EXPERIMENTS, *EXPLICIT_ONLY_EXPERIMENTS)
                if getattr(args, name)]
    experiments = selected or EXPERIMENTS
    errors = []
    for experiment in experiments:
        experiment_args = copy.copy(args)
        for name in (*EXPERIMENTS, *EXPLICIT_ONLY_EXPERIMENTS):
            setattr(experiment_args, name, name == experiment)
        print(f"Running: --{experiment.replace('_', '-')}", flush=True)
        try:
            _run_analysis(experiment_args)
        except (Exception, SystemExit) as error:
            if selected:
                raise
            errors.append(f"{experiment}: {error}")
            print(f"FAILED {experiment}: {error}", flush=True)
    if errors:
        raise SystemExit("Analysis failures:\n" + "\n".join(errors))


def _run_analysis(args):
    """Run one selected analysis with its existing evaluation and plot pipeline."""
    if args.rule_vector_magnitude_sweep:
        runs, failures = run_rule_vector_magnitude_sweep(args)
        plot_rule_vector_magnitude_sweep(
            runs, args.output_dir, args.feature, args.hidden)
        if failures:
            raise SystemExit(
                f"{len(failures)} checkpoint(s) failed; see magnitude-sweep summary.")
        return
    if args.rule_vector_intervention:
        runs, failures = run_rule_vector_intervention(args)
        plot_rule_vector_intervention(runs, args.output_dir, args.feature, args.hidden)
        plot_rule_vector_norms(runs, args.output_dir, args.feature, args.hidden)
        if failures:
            raise SystemExit(f"{len(failures)} checkpoint(s) failed; see rule-vector summary.")
        return
    if args.backbone_probe:
        runs, failures = run_backbone_probe(args)
        plot_backbone_probe_random(runs, args.output_dir, args.feature, args.hidden)
        plot_backbone_span_grid(runs, args.output_dir, args.feature, args.hidden)
        plot_backbone_span_grid(runs, args.output_dir, args.feature, args.hidden,
                                norm_matched=True)
        plot_backbone_conditions(runs, args.output_dir, args.feature, args.hidden)
        if failures:
            raise SystemExit(f"{len(failures)} checkpoint(s) failed; see backbone-probe summary.")
        return
    if args.pathway_gain:
        runs, failures = run_pathway_gain(args)
        plot_pathway_gain(runs, args.output_dir, args.feature, args.hidden)
        plot_pathway_gain_m_stats(runs, args.output_dir, args.feature, args.hidden)
        if failures:
            raise SystemExit(f"{len(failures)} checkpoint(s) failed; see pathway-gain summary.")
        return
    if args.memory_pca:
        for ruleset in ([args.ruleset] if args.ruleset else GROUPS):
            plot_memory_pca(args.checkpoint_dir, args.output_dir, args.feature,
                            args.hidden, args.seed, args.test_seed, ruleset)
        return
    failures = []
    if args.m_intervention:
        runs, failures = run_m_intervention(args)
        plot_m_intervention(runs, args.output_dir, args.feature, args.hidden)
        plot_m_trace_similarity(runs, args.output_dir, args.feature, args.hidden)
        if failures:
            raise SystemExit(f"{len(failures)} checkpoint(s) failed; see M-intervention summary.")
        return
    if args.rule_cue:
        runs, failures = run_evaluation(args)
        plot_rule_cue(runs, args.output_dir, args.feature, args.hidden)
        plot_rule_cue_errors(runs, args.output_dir, args.feature, args.hidden)
        if failures:
            raise SystemExit(f"{len(failures)} checkpoint(s) failed; see rule-cue summary.")
        return
    accuracy_seed, accuracy_ruleset = args.seed, args.ruleset
    if args.plot_only:
        grouped = load_accuracies(args.input_dir, args.feature, args.hidden, args.ruleset, args.seed)
    else:
        runs, failures = run_evaluation(args)
        grouped = _group_runs(runs)
        if len(runs) == 1:
            accuracy_seed, accuracy_ruleset = runs[0]["seed"], runs[0]["ruleset"]
    if not plot_accuracies(grouped, args.output_dir, args.feature, args.hidden,
                           seed=accuracy_seed, ruleset=accuracy_ruleset):
        raise SystemExit("No matching successful accuracies; run --accuracy to evaluate checkpoints.")
    if failures:
        raise SystemExit(f"{len(failures)} checkpoint(s) failed; see summary report.")


if __name__ == "__main__":
    main()
