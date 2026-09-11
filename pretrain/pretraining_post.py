"""Analyze pretraining checkpoints: accuracy, memory PCA, cue timing, and M interventions.

Run from the repository root:
    python pretrain/pretraining_post.py
    python pretrain/pretraining_post.py --accuracy
    python pretrain/pretraining_post.py --memory-pca
    python pretrain/pretraining_post.py --rule-cue
    python pretrain/pretraining_post.py --m-intervention
    python pretrain/pretraining_post.py --plot-only

Default (no experiment flag): run accuracy, memory PCA, rule-cue, and
M-intervention sequentially. An explicit experiment flag runs only that
experiment. Shared filters also apply when running all experiments.
All-analysis runs continue after an experiment fails and report failures at the end.
--accuracy generates fresh trials, saves accuracy JSON, then plots. Accuracy uses
the model's angle-based response-timepoint metric, not trial success counts.
--plot-only reads existing accuracy JSON without loading models. Accuracy bars show seed
means, error bars population SD, and dots individual seeds. Reads per-run
accuracy JSON files, excluding summary reports to avoid duplicate counts.
Use --memory-pca to project stimulus and response trajectories into the MemoryAnti memory
subspace, for hidden and effective modulation, in both motif groups. Saves
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


GROUPS = {
    "fdanti_delaygo": ("Proper motif", ("fdanti", "delaygo", "delayanti")),
    "fdgo_delaygo": ("Improper motif", ("fdgo", "delaygo", "delayanti")),
}
COLORS = ("#3182ce", "#38a169", "#e53e3e")
# Matches two_task_analysis.py's c_vals[stimulus_index] trajectory colors.
STIMULUS_COLORS = ("#e53e3e", "#3182ce", "#38a169", "#805ad5",
                   "#dd6b20", "#319795", "#718096", "#d53f8c", "#d69e2e")


def discover_checkpoints(root, feature, hidden, ruleset=None, seed=None):
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

    matches = discover_checkpoints(args.checkpoint_dir, args.feature, args.hidden, args.ruleset, args.seed)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(run["aname"] for run in runs) if len(runs) == 1 else f"{'_'.join(summary)}_hidden{hidden}_{feature}"
    path = output_dir / f"rule_cue_{tag}.png"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = runs[0]["aname"] if len(runs) == 1 else f"{'_'.join(dict.fromkeys(run['ruleset'] for run in runs))}_hidden{hidden}_{feature}"
    path = output_dir / f"rule_cue_errors_{tag}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


M_TRACE_SCALES = (0.0, 0.5, 1.0, 2.0)
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
    for scale in M_TRACE_SCALES:
        override = M_nocue + scale * trace
        outputs, _ = _rollout_m_intervention(model, inputs_resp_off, boundaries,
                                             device, args.batch_size, m_override=override)
        accuracy = _score_outputs(model, outputs, targets, masks, inputs, device)
        conditions[f"trace_scale_{scale:g}"] = {
            "accuracy_pct": accuracy * 100,
            "delta_accuracy_pp": (accuracy - intact_acc) * 100}
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
        print(f"  {condition}: accuracy={stats['accuracy_pct']:.2f}, "
              f"delta_pp={stats['delta_accuracy_pp']:.2f}")
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

    matches = discover_checkpoints(args.checkpoint_dir, args.feature, args.hidden,
                                   args.ruleset, args.seed)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = (runs[0]["aname"] if len(runs) == 1
           else f"{'_'.join(summary)}_hidden{hidden}_{feature}")
    path = output_dir / f"m_intervention_{tag}.png"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = [run for entries in selected.values() for run in entries]
    tag = (plotted[0]["aname"] if len(plotted) == 1
           else f"{'_'.join(ruleset for ruleset, entries in selected.items() if entries)}"
                f"_hidden{hidden}_{feature}")
    path = output_dir / f"m_trace_similarity_{tag}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


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


def plot_accuracies(grouped, output_dir, feature, hidden):
    if not any(values for tasks in grouped.values() for values in tasks.values()):
        return None
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for axis, (ruleset, (title, tasks)) in zip(axes, GROUPS.items()):
        labels = []
        for index, (task, color) in enumerate(zip(tasks, COLORS)):
            values = np.asarray(grouped[ruleset][task], dtype=float)
            labels.append(f"{task}\nn={values.size}")
            if values.size:
                axis.bar(index, values.mean(), yerr=values.std(), width=0.6,
                         color=color, alpha=0.7, capsize=3,
                         error_kw={"elinewidth": 1})
                jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.zeros(1)
                axis.scatter(index + jitter, values, color="black", s=12,
                             alpha=0.6, zorder=3)
            else:
                print(f"Missing accuracy: {ruleset}/{task}")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"accuracy_by_motif_hidden{hidden}_{feature}.png"
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
    first_title = "DelayPro" if first_rule == "fdgo" else "DelayAnti"
    if period == "response":
        first_rule, first_title = "delaygo", "MemoryPro"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"memory_pca_{period}" if representation == "hidden" else f"memory_pca_{period}_effective_modulation"
    path = output_dir / f"{prefix}_{aname}.png"
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

    Like two_task_analysis's normal PCA trajectories, flatten trial/time for
    fitting and use the fitted mean in transform, without feature scaling.
    Here the basis is restricted to MemoryAnti memory (two PCs, not the pooled
    three-PC full-trial basis), and trajectories are averaged by direction.
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("pretraining_diagnosis"),
                        help="Evaluation JSON output directory; --plot-only reads its accuracy reports.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="PNG output directory.")
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
    parser.add_argument("--n-trials", type=positive_int, default=200,
                        help="Trials per task for evaluations; memory PCA uses a fixed 256.")
    parser.add_argument("--batch-size", type=positive_int, default=8,
                        help="Evaluation batch size; memory PCA uses a fixed 8.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Device for accuracy, rule-cue, and M-intervention; memory PCA selects automatically.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("pretraining"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-seed", type=int, default=0)
    parser.add_argument("--ruleset", choices=tuple(GROUPS), default=None)
    args = parser.parse_args(argv)
    if not 0 <= args.test_seed <= 2**32 - 3:
        parser.error("--test-seed must be between 0 and 2**32 - 3")
    selected = [name for name in (*EXPERIMENTS, "plot_only") if getattr(args, name)]
    experiments = selected or EXPERIMENTS
    errors = []
    for experiment in experiments:
        experiment_args = copy.copy(args)
        for name in (*EXPERIMENTS, "plot_only"):
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
    if args.plot_only:
        grouped = load_accuracies(args.input_dir, args.feature, args.hidden, args.ruleset, args.seed)
    else:
        runs, failures = run_evaluation(args)
        grouped = _group_runs(runs)
    if not plot_accuracies(grouped, args.output_dir, args.feature, args.hidden):
        raise SystemExit("No matching successful accuracies; run --accuracy to evaluate checkpoints.")
    if failures:
        raise SystemExit(f"{len(failures)} checkpoint(s) failed; see summary report.")


if __name__ == "__main__":
    main()