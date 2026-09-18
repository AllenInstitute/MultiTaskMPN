#!/usr/bin/env python
# coding: utf-8
"""Plot measured postsynaptic waveforms from an Allen multipatch experiment.

The three representative connections were selected from the 72 connections
displayed in ``biology_fast_synaptic_dynamics.png``.  All are excitatory
connections, and their fitted pulse-8 response is respectively about 3.05,
1.19, and 0.28 times the first-pulse response.  The plotted lines are trial
means; no fitted waveform or synthetic response is plotted.

The first run downloads three public Allen Synaptic Physiology NWB files.
Later runs reuse the copies in ``--cache-dir``.
"""
import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


OUT_DIR = Path("cartoon_plot")
NWB_URL_TEMPLATE = (
    "https://allen-synphys.s3-us-west-2.amazonaws.com/"
    "synphys-{experiment_id}.nwb"
)
DEFAULT_FREQUENCY_HZ = 50.0
DEFAULT_RECOVERY_MS = 250.0
DEFAULT_WINDOW_MS = (-4.0, 12.0)
DEFAULT_BASELINE_MS = (-1.0, -0.2)

# These examples are all excitatory connections and are among the 72 rows
# shown in biology_fast_synaptic_dynamics.png. Database IDs, scores, and
# response ratios come from synphys_r2.1_medium.sqlite; device IDs map the
# cells to recordings inside each experiment's NWB file.
EXAMPLES = (
    {
        "group": "facilitating",
        "title": "Facilitating",
        "experiment_id": "1524161846.498",
        "pair_database_id": 60101,
        "pair_external_id": ("1524161846.498", "8", "6"),
        "pre_device": 7,
        "post_device": 5,
        "induction_log2_change": 2.5384556560653215,
        "late_train_ratios": (6.3537, 5.8097, 3.0484),
    },
    {
        "group": "pseudolinear",
        "title": "Pseudolinear",
        "experiment_id": "1532552839.296",
        "pair_database_id": 67182,
        "pair_external_id": ("1532552839.296", "8", "5"),
        "pre_device": 7,
        "post_device": 4,
        "induction_log2_change": 0.24547987531141785,
        "late_train_ratios": (0.8669, 2.4222, 1.1855),
    },
    {
        "group": "depressing",
        "title": "Depressing",
        "experiment_id": "1539987094.832",
        "pair_database_id": 70801,
        "pair_external_id": ("1539987094.832", "3", "4"),
        "pre_device": 2,
        "post_device": 3,
        "induction_log2_change": -0.8112414840749544,
        "late_train_ratios": (0.5699, 0.6100, 0.2838),
    },
)

PULSES = (
    (0, "Pulse 1", "#4D4D4D", "-"),
    (3, "Pulse 4", "#E69F00", "-"),
    (7, "Pulse 8", "#D55E00", "-"),
    (8, "Pulse 9 (recovery)", "#56B4E9", (0, (3, 1.5))),
    (11, "Pulse 12", "#0072B2", (0, (1.5, 1.2))),
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})


def _download_nwb(experiment_id, cache_dir):
    """Download one experiment's NWB source file into a local cache."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"synphys-{experiment_id}.nwb"
    if destination.exists() and destination.stat().st_size > 100 * 1024 ** 2:
        print(f"Using cached NWB: {destination}")
        return destination

    temporary = destination.with_suffix(".nwb.download")
    source_url = NWB_URL_TEMPLATE.format(experiment_id=experiment_id)
    print(f"Downloading measured traces from: {source_url}")
    print(f"Destination: {destination}")
    try:
        with urllib.request.urlopen(source_url) as source, temporary.open("wb") as out:
            shutil.copyfileobj(source, out, length=1024 * 1024)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _load_dataset(nwb_path):
    try:
        from aisynphys.data import MultiPatchDataset
    except ImportError as exc:
        raise RuntimeError(
            "aisynphys is required to read the measured NWB traces. Install "
            "aisynphys and neuroanalysis, then rerun this script."
        ) from exc
    return MultiPatchDataset(str(nwb_path))


def _matching_sweeps(dataset, example, frequency_hz, recovery_ms):
    """Return IC sweeps matching the requested 12-pulse train protocol."""
    required_devices = {example["pre_device"], example["post_device"]}
    matches = []
    recovery_s = float(recovery_ms) / 1000.0
    for sweep_index, sweep in enumerate(dataset.contents):
        if not required_devices.issubset(set(sweep.devices)):
            continue
        try:
            recordings = [sweep[device] for device in required_devices]
            if any(recording.clamp_mode != "ic" for recording in recordings):
                continue
            stimulus = sweep[example["pre_device"]].stim_params()
        except (AttributeError, KeyError, TypeError):
            continue
        if stimulus is None:
            continue
        frequency, recovery = stimulus
        if (abs(float(frequency) - float(frequency_hz)) <= 0.5
                and abs(float(recovery) - recovery_s) <= 5e-3):
            matches.append((sweep_index, sweep))
    if not matches:
        raise RuntimeError(
            "No current-clamp sweeps match the requested 50 Hz / 250 ms "
            "pulse-train condition."
        )
    return matches


def _aligned_trace(recording, spike_time, time_ms, baseline_ms):
    """Return a baseline-subtracted measured trace aligned to one spike."""
    primary = recording["primary"]
    start_s = float(spike_time) + float(time_ms[0]) / 1000.0
    stop_s = float(spike_time) + float(time_ms[-1]) / 1000.0
    segment = primary.time_slice(start_s, stop_s)
    relative_ms = (segment.time_values - float(spike_time)) * 1000.0
    voltage_mv = np.asarray(segment.data, dtype=float) * 1000.0
    trace = np.interp(time_ms, relative_ms, voltage_mv)
    baseline_mask = (
        (time_ms >= float(baseline_ms[0]))
        & (time_ms <= float(baseline_ms[1]))
    )
    if not np.any(baseline_mask):
        raise ValueError("baseline window does not overlap the plotted window")
    return trace - np.median(trace[baseline_mask])


def load_measured_waveforms(datasets, frequency_hz=DEFAULT_FREQUENCY_HZ,
                            recovery_ms=DEFAULT_RECOVERY_MS,
                            window_ms=DEFAULT_WINDOW_MS,
                            baseline_ms=DEFAULT_BASELINE_MS):
    """Extract trial waveforms for pulses 1, 8, and the recovery pulse."""
    try:
        from neuroanalysis.analyzers.stim_pulse import PatchClampStimPulseAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "neuroanalysis is required to locate presynaptic spikes."
        ) from exc

    sweeps_by_group = {
        example["group"]: _matching_sweeps(
            datasets[example["experiment_id"]], example,
            frequency_hz, recovery_ms)
        for example in EXAMPLES
    }
    sample_dt_ms = min(
        float(sweep[example["post_device"]]["primary"].dt) * 1000.0
        for example in EXAMPLES
        for _, sweep in sweeps_by_group[example["group"]]
    )
    time_ms = np.arange(
        float(window_ms[0]), float(window_ms[1]) + sample_dt_ms / 2.0,
        sample_dt_ms,
    )
    waveforms = {
        example["group"]: {pulse_number: [] for pulse_number, *_ in PULSES}
        for example in EXAMPLES
    }
    sweep_ids = {example["group"]: [] for example in EXAMPLES}

    for example in EXAMPLES:
        group = example["group"]
        for sweep_index, sweep in sweeps_by_group[group]:
            pre_recording = sweep[example["pre_device"]]
            post_recording = sweep[example["post_device"]]
            pulses = PatchClampStimPulseAnalyzer.get(
                pre_recording).evoked_spikes()
            pulse_map = {int(pulse["pulse_n"]): pulse for pulse in pulses}
            trial = {}
            for pulse_number, *_ in PULSES:
                pulse = pulse_map.get(pulse_number)
                if pulse is None or len(pulse["spikes"]) != 1:
                    break
                spike_time = pulse["spikes"][0]["max_slope_time"]
                trial[pulse_number] = _aligned_trace(
                    post_recording, spike_time, time_ms, baseline_ms)
            if len(trial) != len(PULSES):
                continue
            for pulse_number, trace in trial.items():
                waveforms[group][pulse_number].append(trace)
            sweep_ids[group].append(int(sweep_index))

    for example in EXAMPLES:
        group = example["group"]
        for pulse_number, *_ in PULSES:
            traces = waveforms[group][pulse_number]
            if not traces:
                raise RuntimeError(
                    f"No complete measured traces were found for {group}, "
                    f"pulse {pulse_number + 1}."
                )
            waveforms[group][pulse_number] = np.vstack(traces)
    return time_ms, waveforms, sweep_ids


def plot_measured_waveforms(time_ms, waveforms, out_dir=OUT_DIR):
    """Plot trial-mean waveforms for three dynamics classes."""
    fig, axes = plt.subplots(
        3, 1, figsize=(3.76, 3.70), sharex=True, sharey=True)

    for ax, example in zip(axes, EXAMPLES):
        group = example["group"]
        for pulse_number, _, color, linestyle in PULSES:
            traces = waveforms[group][pulse_number]
            ax.plot(
                time_ms, np.mean(traces, axis=0), color=color,
                linewidth=1.5, linestyle=linestyle, zorder=3)

        ax.axhline(0, color="#B5B5B5", linewidth=0.5, zorder=0)
        pulse_8_ratio = example["late_train_ratios"][2]
        ax.set_title(
            f"{example['title']}    "
            rf"fitted $R_8/R_1={pulse_8_ratio:.2f}\times$",
            color="#222222", fontsize=8.8, loc="left", pad=3)
        ax.set_xlim(0.0, float(time_ms[-1]))
        ax.set_xticks([0, 5, 10])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=0.6, length=2.5)
        ax.grid(False)

    axes[-1].set_xlabel("Time from presynaptic spike (ms)")
    fig.supylabel(r"Mean PSP voltage $\Delta V_m$ (mV)", x=0.055)
    fig.subplots_adjust(
        left=0.18, right=0.985, bottom=0.13, top=0.97, hspace=0.35)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "biology_fast_synaptic_waveforms.png"
    fig.savefig(png_path, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")
    return png_path


def plot_waveform_legend(out_dir=OUT_DIR):
    """Save the pulse-condition legend as a separate compact figure."""
    handles = [
        Line2D([0], [0], color=color, linewidth=1.65,
               linestyle=linestyle, label=label)
        for _, label, color, linestyle in PULSES
    ]
    fig, ax = plt.subplots(figsize=(2.45, 1.65))
    ax.axis("off")
    ax.legend(
        handles=handles, loc="center left", ncol=1, handlelength=2.2,
        handletextpad=0.65, labelspacing=0.75)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_path = out_dir / "biology_fast_synaptic_waveforms_legend.png"
    fig.savefig(legend_path, dpi=400, facecolor="white", bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {legend_path}")
    return legend_path


def save_metadata(time_ms, waveforms, sweep_ids, out_dir=OUT_DIR):
    """Save provenance and simple response summaries for the plotted traces."""
    response_mask = (time_ms >= 1.0) & (time_ms <= 12.0)
    metadata = {
        "source": "Allen Institute Synaptic Physiology raw NWB recording",
        "source_urls": [
            NWB_URL_TEMPLATE.format(experiment_id=example["experiment_id"])
            for example in EXAMPLES
        ],
        "protocol": {
            "frequency_hz": DEFAULT_FREQUENCY_HZ,
            "recovery_ms": DEFAULT_RECOVERY_MS,
            "displayed_pulses_zero_based": [pulse[0] for pulse in PULSES],
            "alignment": "presynaptic spike maximum-slope time",
            "baseline_window_ms": list(DEFAULT_BASELINE_MS),
            "plot_window_ms": list(DEFAULT_WINDOW_MS),
            "display_window_ms": [0.0, float(DEFAULT_WINDOW_MS[1])],
            "filtering": "none",
        },
        "display": "colored lines are arithmetic means across measured "
                   "trials; individual trials are omitted",
        "connections": [],
    }
    for example in EXAMPLES:
        group = example["group"]
        pulse_summaries = {}
        for pulse_number, label, *_ in PULSES:
            traces = waveforms[group][pulse_number]
            means = traces[:, response_mask].mean(axis=1)
            peaks = traces[:, response_mask].max(axis=1)
            pulse_summaries[label] = {
                "n_trials": int(traces.shape[0]),
                "trial_mean_delta_vm_mv": means.tolist(),
                "trial_peak_delta_vm_mv": peaks.tolist(),
            }
        metadata["connections"].append({
            key: value for key, value in example.items()
        } | {
            "synapse_type": "excitatory",
            "selected_from_dynamics_heatmap": True,
            "included_nwb_sweep_indices": sweep_ids[group],
            "pulses": pulse_summaries,
        })

    out_dir = Path(out_dir)
    metadata_path = out_dir / "biology_fast_synaptic_waveforms_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Saved: {metadata_path}")
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR),
        help=f"output directory (default: {OUT_DIR})")
    parser.add_argument(
        "--cache-dir", default="~/.cache/aisynphys/waveforms",
        help="directory containing/downloading the three NWB source files")
    args = parser.parse_args()

    experiment_ids = {example["experiment_id"] for example in EXAMPLES}
    nwb_paths = {
        experiment_id: _download_nwb(experiment_id, args.cache_dir)
        for experiment_id in experiment_ids
    }
    datasets = {
        experiment_id: _load_dataset(path)
        for experiment_id, path in nwb_paths.items()
    }
    time_ms, waveforms, sweep_ids = load_measured_waveforms(datasets)
    for example in EXAMPLES:
        group = example["group"]
        print(f"{example['title']}: {len(sweep_ids[group])} measured trials")
    plot_measured_waveforms(time_ms, waveforms, out_dir=args.out_dir)
    plot_waveform_legend(out_dir=args.out_dir)
    save_metadata(time_ms, waveforms, sweep_ids, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
