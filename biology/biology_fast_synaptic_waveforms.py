#!/usr/bin/env python
# coding: utf-8
"""Plot measured postsynaptic waveforms from Allen multipatch experiments.

The three connections are the ones Campagnola et al. 2022 (Science 375,
eabj5861) show in Fig. 4A, taken from the paper's figure notebook
(aisynphys, branch figure-notebooks, doc/figure_notebooks/figure_04_dynamics.ipynb):
a depressing L5 ET -> Pv, a facilitating L5 IT -> Sst and a pseudo-linear
L6 Pyr -> Pv connection. EXAMPLES names them by experiment and cell ext_id;
everything else (database pair id, NWB device numbers, synapse type, the
paper's STP induction metric) is resolved from the medium database at run
time by `resolve_examples`. The plotted lines are trial means; no fitted
waveform or synthetic response is plotted.

Besides the waveform figure, the script writes
``biology_fast_synaptic_trial_amplitudes.json``, the data behind
``cartoon.plot_synaptic_train_amplitudes`` (Campagnola et al. 2022 Fig. 4A
redrawn). For every one of the 12 pulses of the same three connections it
holds:

* ``trial_amplitude_mv`` / ``trial_noise_mv``: the response's deconvolution
  fit amplitude (``dec_fit_reconv_amp``) and the matching baseline fit
  amplitude of every QC-passed response, read from the Synaptic Physiology
  medium database exactly as the paper's panel does
  (``aisynphys.dynamics.stim_sorted_pulse_amp``). Like the paper's
  ``plot_stim_sorted_pulse_amp``, all 50 Hz trains are pooled whatever their
  recovery delay, so pulses 9-12 mix delays; each trial's delay is stored.
* ``mean_waveform_mv``: the trial-mean PSP waveform (-1 to 8 ms after the
  presynaptic spike) from the raw NWB recording, over the same pooled 50 Hz
  trains - the black trace under each swarm.
* ``nwb_trial_amplitude_mv`` etc.: a secondary NWB-derived amplitude (mean
  voltage in a 1 ms window at the trial-mean peak latency, with a pre-spike
  noise counterpart and the per-trial peak), kept for cross-checking only.

The NWB recordings and the medium database are read from the shared project
cache (``SHARED_CACHE_DIR`` in biology_fast_synaptic_dynamics.py:
``waveforms/`` and ``database/``) when it exists; otherwise the NWB files are
downloaded into ``--cache-dir`` and the database into the aisynphys package
default. Both locations can be overridden on the command line.
"""
import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# The medium-database loader and the shared cache location are defined once,
# in the dynamics heatmap script.
from biology_fast_synaptic_dynamics import SHARED_CACHE_DIR, _load_database, default_cache_dir


OUT_DIR = Path("cartoon_plot")
NWB_URL_TEMPLATE = (
    "https://allen-synphys.s3-us-west-2.amazonaws.com/"
    "synphys-{experiment_id}.nwb"
)
DEFAULT_FREQUENCY_HZ = 50.0
DEFAULT_RECOVERY_MS = 250.0
DEFAULT_WINDOW_MS = (-4.0, 12.0)
DEFAULT_BASELINE_MS = (-1.0, -0.2)

# The paper's Fig. 4A example pairs, in the paper's row order. `experiment_id`
# is the NWB / experiment ext_id and `pair_external_id` its (experiment, pre
# cell, post cell) ext_ids, exactly as the figure notebook lists them; the
# class labels are the notebook's comments. `resolve_examples` adds the
# database pair id, the NWB device numbers of the two cells, the synapse type
# and the paper's stored STP metrics.
EXAMPLES = (
    {
        "group": "depressing",
        "title": "Depressing",
        "experiment_id": "1536790789.296",
        "pair_external_id": ("1536790789.296", "5", "3"),
        "pre_class": "L5 ET",
        "post_class": "Pv",
    },
    {
        "group": "facilitating",
        "title": "Facilitating",
        "experiment_id": "1490304528.810",
        "pair_external_id": ("1490304528.810", "8", "7"),
        "pre_class": "L5 IT",
        "post_class": "Sst",
    },
    {
        "group": "pseudolinear",
        "title": "Pseudo-linear",
        "experiment_id": "1522259564.089",
        "pair_external_id": ("1522259564.089", "4", "2"),
        "pre_class": "L6 Pyr",
        "post_class": "Pv",
    },
)

# Every pulse of the standard 12-pulse train (0-based), for the trial-amplitude
# export; PULSES below is the subset the waveform figure displays.
ALL_PULSES = tuple(range(12))
RESPONSE_WINDOW_MS = (1.0, 12.0)
AMPLITUDE_HALF_WIDTH_MS = 0.5
# Pre-spike center of the per-trial noise measurement (same width as the
# amplitude window; must lie inside DEFAULT_WINDOW_MS and before the baseline).
NOISE_CENTER_MS = -2.5
# Slice of the trial-mean waveform saved per pulse, as drawn by the paper's
# Fig. 4A (aisynphys plot_stim_sorted_pulse_amp: -1 to 8 ms), and the maximum
# number of samples kept for it.
MEAN_WAVEFORM_WINDOW_MS = (-1.0, 8.0)
MEAN_WAVEFORM_MAX_SAMPLES = 200

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


def _device_id(cell, ext_id):
    """NWB device (headstage) number of a cell: the database's electrode
    device id when stored, else the convention device = cell ext_id - 1."""
    electrode = getattr(cell, "electrode", None)
    device = getattr(electrode, "device_id", None)
    if device is None:
        return int(ext_id) - 1
    return int(device)


def resolve_examples(db, examples=EXAMPLES):
    """Look the example pairs up in the database and return enriched copies.

    Adds `pair_database_id`, `pre_device` / `post_device` (NWB headstages),
    `synapse_type` ('ex' / 'in'), `pre_cre_type` / `post_cre_type` and the
    paper's stored dynamics metrics (`stp_induction_50hz`,
    `stp_recovery_250ms`, `variability_resting_state`,
    `variability_stp_induced_state_50hz`; None when unavailable)."""
    resolved = []
    for example in examples:
        experiment_id, pre_id, post_id = example["pair_external_id"]
        experiment = db.experiment_from_ext_id(experiment_id)
        pair = experiment.pairs.get((pre_id, post_id))
        if pair is None:
            raise RuntimeError(f"pair {pre_id}->{post_id} not found in experiment {experiment_id}")
        if pair.synapse is None:
            raise RuntimeError(f"pair {pre_id}->{post_id} of {experiment_id} is not a synapse")
        dynamics = getattr(pair, "dynamics", None)
        metrics = {
            key: (None if dynamics is None or getattr(dynamics, key, None) is None
                  else float(getattr(dynamics, key)))
            for key in ("stp_induction_50hz", "stp_recovery_250ms",
                        "variability_resting_state", "variability_stp_induced_state_50hz")
        }
        resolved.append(dict(example) | {
            "pair_database_id": int(pair.id),
            "pre_device": _device_id(pair.pre_cell, pre_id),
            "post_device": _device_id(pair.post_cell, post_id),
            "synapse_type": str(pair.synapse.synapse_type),
            "pre_cre_type": getattr(pair.pre_cell, "cre_type", None),
            "post_cre_type": getattr(pair.post_cell, "cre_type", None),
        } | metrics)
        print(f"{example['title']}: {example['pre_class']}->{example['post_class']} "
              f"({resolved[-1]['pre_cre_type']}->{resolved[-1]['post_cre_type']}), pair id "
              f"{pair.id}, devices {resolved[-1]['pre_device']}->{resolved[-1]['post_device']}, "
              f"{resolved[-1]['synapse_type']}, STP induction 50 Hz = "
              f"{metrics['stp_induction_50hz']}")
    return resolved


def _matching_sweeps(dataset, example, frequency_hz, recovery_ms):
    """Return IC sweeps matching the requested 12-pulse train protocol.

    `recovery_ms=None` accepts every recovery delay at that induction
    frequency, which is how the paper pools trains for its Fig. 4A."""
    required_devices = {example["pre_device"], example["post_device"]}
    matches = []
    recovery_s = None if recovery_ms is None else float(recovery_ms) / 1000.0
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
        if abs(float(frequency) - float(frequency_hz)) > 0.5:
            continue
        if recovery_s is not None and abs(float(recovery) - recovery_s) > 5e-3:
            continue
        matches.append((sweep_index, sweep))
    if not matches:
        raise RuntimeError(
            f"No current-clamp sweeps match the requested {frequency_hz:g} Hz"
            + ("" if recovery_ms is None else f" / {recovery_ms:g} ms")
            + " pulse-train condition."
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


def _single_spike_times(pre_recording):
    """Map 0-based pulse number -> max-slope time of its single evoked spike.

    Pulses that evoked no spike or more than one are left out, and so are
    pulses whose single spike has no max-slope time (neuroanalysis reports
    None when it cannot place the rising phase): aligning such a pulse on
    another spike feature would shift it relative to every other trace. Callers
    decide per figure whether a trial must be complete."""
    try:
        from neuroanalysis.analyzers.stim_pulse import PatchClampStimPulseAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "neuroanalysis is required to locate presynaptic spikes."
        ) from exc
    pulses = PatchClampStimPulseAnalyzer.get(pre_recording).evoked_spikes()
    times = {}
    for pulse in pulses:
        if len(pulse["spikes"]) != 1:
            continue
        spike_time = pulse["spikes"][0].get("max_slope_time")
        if spike_time is not None and np.isfinite(float(spike_time)):
            times[int(pulse["pulse_n"])] = float(spike_time)
    return times


def _sweeps_and_time_axis(datasets, examples, frequency_hz, recovery_ms, window_ms):
    """Matching sweeps per group and the common time axis of the plotted window."""
    sweeps_by_group = {
        example["group"]: _matching_sweeps(
            datasets[example["experiment_id"]], example,
            frequency_hz, recovery_ms)
        for example in examples
    }
    sample_dt_ms = min(
        float(sweep[example["post_device"]]["primary"].dt) * 1000.0
        for example in examples
        for _, sweep in sweeps_by_group[example["group"]]
    )
    time_ms = np.arange(
        float(window_ms[0]), float(window_ms[1]) + sample_dt_ms / 2.0,
        sample_dt_ms,
    )
    return sweeps_by_group, time_ms


def load_measured_waveforms(datasets, examples, frequency_hz=DEFAULT_FREQUENCY_HZ,
                            recovery_ms=DEFAULT_RECOVERY_MS,
                            window_ms=DEFAULT_WINDOW_MS,
                            baseline_ms=DEFAULT_BASELINE_MS):
    """Extract trial waveforms for the displayed pulses (PULSES).

    `examples` are the resolved connections (`resolve_examples`). A trial is
    kept only when every displayed pulse evoked exactly one presynaptic spike,
    so the five mean waveforms of a connection come from the same trials."""
    sweeps_by_group, time_ms = _sweeps_and_time_axis(
        datasets, examples, frequency_hz, recovery_ms, window_ms)
    waveforms = {
        example["group"]: {pulse_number: [] for pulse_number, *_ in PULSES}
        for example in examples
    }
    sweep_ids = {example["group"]: [] for example in examples}

    for example in examples:
        group = example["group"]
        for sweep_index, sweep in sweeps_by_group[group]:
            post_recording = sweep[example["post_device"]]
            spike_times = _single_spike_times(sweep[example["pre_device"]])
            if any(pulse_number not in spike_times for pulse_number, *_ in PULSES):
                continue
            for pulse_number, *_ in PULSES:
                waveforms[group][pulse_number].append(_aligned_trace(
                    post_recording, spike_times[pulse_number], time_ms,
                    baseline_ms))
            sweep_ids[group].append(int(sweep_index))

    for example in examples:
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


def _trial_amplitudes(time_ms, traces, response_ms=RESPONSE_WINDOW_MS,
                      half_width_ms=AMPLITUDE_HALF_WIDTH_MS,
                      noise_center_ms=NOISE_CENTER_MS,
                      waveform_ms=MEAN_WAVEFORM_WINDOW_MS,
                      waveform_max_samples=MEAN_WAVEFORM_MAX_SAMPLES):
    """Per-pulse summary of one pulse's aligned trial traces.

    The trial-mean waveform's largest |deflection| inside `response_ms` fixes
    one latency for the pulse; each trial's amplitude is its mean voltage in a
    window of +-`half_width_ms` around that latency. Measuring every trial at
    the same latency keeps failures near zero (a per-trial maximum would report
    the noise peak instead). The same window applied before the spike, centered
    on `noise_center_ms`, gives each trial's noise amplitude, which the paper's
    adjusted coefficient of variation subtracts out. The per-trial maximum over
    the response window is kept for comparison with the waveform metadata, and
    the trial-mean waveform over `waveform_ms` (thinned to at most
    `waveform_max_samples` points) is what Fig. 4A draws under the dots.

    Returns a dict: peak_latency_ms, amplitude_mv, peak_mv, noise_mv (arrays
    one per trial), waveform_time_ms and mean_waveform_mv."""
    traces = np.asarray(traces, dtype=float)
    response_mask = (time_ms >= float(response_ms[0])) & (time_ms <= float(response_ms[1]))
    noise_mask = np.abs(time_ms - float(noise_center_ms)) <= float(half_width_ms)
    waveform_mask = (time_ms >= float(waveform_ms[0])) & (time_ms <= float(waveform_ms[1]))
    if traces.ndim != 2 or not (response_mask.any() and noise_mask.any() and waveform_mask.any()):
        raise ValueError("traces must be (n_trials, n_samples) covering the response, "
                         "noise and waveform windows")
    mean_trace = traces.mean(axis=0)
    window_times = time_ms[response_mask]
    peak_latency_ms = float(window_times[np.argmax(np.abs(mean_trace[response_mask]))])
    latency_mask = np.abs(time_ms - peak_latency_ms) <= float(half_width_ms)
    stride = max(1, int(np.ceil(waveform_mask.sum() / float(waveform_max_samples))))
    return {
        "peak_latency_ms": peak_latency_ms,
        "amplitude_mv": traces[:, latency_mask].mean(axis=1),
        "peak_mv": traces[:, response_mask].max(axis=1),
        "noise_mv": traces[:, noise_mask].mean(axis=1),
        "waveform_time_ms": time_ms[waveform_mask][::stride],
        "mean_waveform_mv": mean_trace[waveform_mask][::stride],
    }


def load_pulse_waveforms(datasets, examples, frequency_hz=DEFAULT_FREQUENCY_HZ,
                         recovery_ms=None,
                         window_ms=DEFAULT_WINDOW_MS,
                         baseline_ms=DEFAULT_BASELINE_MS,
                         pulses=ALL_PULSES):
    """Per-pulse NWB summaries of every pulse, per example connection: the
    trial-mean waveform the Fig.-4A redraw puts under each swarm, plus the
    secondary NWB amplitude measures of `_trial_amplitudes`.

    Unlike `load_measured_waveforms`, pulses are kept independently: a pulse
    contributes every trial in which it evoked exactly one well-timed
    presynaptic spike, whether or not the other pulses of that trial did. By
    default (`recovery_ms=None`) every train at `frequency_hz` is pooled, as
    the paper does; pass 250 to restrict to the 250 ms recovery protocol.

    Returns {group: {pulse: {"sweeps": [...], plus the per-pulse summary of
    `_trial_amplitudes`}}}."""
    sweeps_by_group, time_ms = _sweeps_and_time_axis(
        datasets, examples, frequency_hz, recovery_ms, window_ms)
    results = {}
    for example in examples:
        group = example["group"]
        traces = {pulse: [] for pulse in pulses}
        sweeps = {pulse: [] for pulse in pulses}
        n_skipped = 0
        for sweep_index, sweep in sweeps_by_group[group]:
            post_recording = sweep[example["post_device"]]
            spike_times = _single_spike_times(sweep[example["pre_device"]])
            for pulse in pulses:
                if pulse not in spike_times:
                    n_skipped += 1
                    continue
                traces[pulse].append(_aligned_trace(
                    post_recording, spike_times[pulse], time_ms, baseline_ms))
                sweeps[pulse].append(int(sweep_index))
        print(f"{example['title']}: {len(sweeps_by_group[group])} sweeps; "
              f"{n_skipped} pulse(s) without a single well-timed spike skipped")
        results[group] = {}
        for pulse in pulses:
            if not traces[pulse]:
                raise RuntimeError(
                    f"No measured trace with a single evoked spike for {group}, "
                    f"pulse {pulse + 1}.")
            summary = _trial_amplitudes(time_ms, np.vstack(traces[pulse]))
            summary["sweeps"] = sweeps[pulse]
            results[group][pulse] = summary
    return results


def load_fit_amplitudes(db, examples, frequency_hz=DEFAULT_FREQUENCY_HZ, pulses=ALL_PULSES):
    """Fitted PSP amplitudes of every QC-passed response, per example pair.

    This is what the paper's Fig. 4A plots: `aisynphys.dynamics.
    stim_sorted_pulse_amp` returns one row per QC-passed pulse response with
    the deconvolution fit amplitude (`dec_fit_reconv_amp`, volts) and the same
    fit applied to the pre-stimulus baseline (`baseline_dec_fit_reconv_amp`,
    the noise estimate behind the paper's adjusted CV). Following the paper's
    `plot_stim_sorted_pulse_amp`, only the induction frequency is filtered, so
    trains with every recovery delay are pooled. Amplitudes are sign-corrected
    (IPSPs positive) and converted to mV. Responses without a baseline fit are
    dropped so amplitude and noise arrays stay paired.

    Returns {group: {"synapse_type": str, "pulses": {pulse: {"amplitude_mv",
    "noise_mv", "recovery_ms", "sync_rec_ext_id"}}}}."""
    try:
        from aisynphys.dynamics import stim_sorted_pulse_amp
    except ImportError as exc:
        raise RuntimeError("aisynphys is required to read the fitted amplitudes.") from exc

    results = {}
    for example in examples:
        pair = db.query(db.Pair).filter(db.Pair.id == int(example["pair_database_id"])).one()
        data = stim_sorted_pulse_amp(pair, db=db)
        data = data[np.isclose(data["induction_frequency"].astype(float),
                               float(frequency_hz), atol=0.5)]
        n_all = int(len(data))
        data = data.dropna(subset=["dec_fit_reconv_amp", "baseline_dec_fit_reconv_amp"])
        if data.empty:
            raise RuntimeError(f"no fitted {frequency_hz:g} Hz responses for {example['group']}")
        synapse_type = str(pair.synapse.synapse_type)
        sign = 1.0 if synapse_type == "ex" else -1.0
        # r2.1 numbers the 12-pulse train 0..11; older builds used 1..12.
        pulse_start = 0 if (data["pulse_number"] == 0).any() else 1
        per_pulse = {}
        for pulse in pulses:
            rows = data[data["pulse_number"] == pulse + pulse_start]
            if rows.empty:
                raise RuntimeError(f"no fitted responses for {example['group']}, pulse {pulse + 1}")
            per_pulse[pulse] = {
                "amplitude_mv": sign * 1000.0 * rows["dec_fit_reconv_amp"].to_numpy(dtype=float),
                "noise_mv": sign * 1000.0 * rows["baseline_dec_fit_reconv_amp"].to_numpy(dtype=float),
                "recovery_ms": 1000.0 * rows["recovery_delay"].to_numpy(dtype=float),
                "sync_rec_ext_id": [str(value) for value in rows["sync_rec_ext_id"]],
            }
        print(f"{example['title']}: {len(data)} fitted {frequency_hz:g} Hz responses "
              f"({n_all - len(data)} without a baseline fit dropped); pulse 1 n = "
              f"{per_pulse[pulses[0]]['amplitude_mv'].size}")
        results[example["group"]] = {"synapse_type": synapse_type, "pulses": per_pulse}
    return results


def save_trial_amplitudes(fits, waveforms, examples, out_dir=OUT_DIR):
    """Write the per-pulse data cartoon.py reads.

    `fits` comes from `load_fit_amplitudes` (the primary amplitudes and noise),
    `waveforms` from `load_pulse_waveforms` (mean waveforms and the secondary
    NWB amplitudes), `examples` from `resolve_examples`. Pulse keys are 0-based strings;
    cartoon.load_synaptic_trial_amplitudes recognizes the file by its
    `amplitude_statistic` field."""
    metadata = {
        "source": "Allen Institute Synaptic Physiology: medium database "
                  "(fitted amplitudes) and raw NWB recordings (waveforms)",
        "source_urls": [
            "https://brain-map.org/our-research/connectivity/synaptic-physiology",
        ] + [
            NWB_URL_TEMPLATE.format(experiment_id=example["experiment_id"])
            for example in examples
        ],
        "selected_from": "Campagnola et al. 2022 Fig. 4A example pairs "
                         "(aisynphys figure_04_dynamics.ipynb)",
        "protocol": {
            "frequency_hz": DEFAULT_FREQUENCY_HZ,
            "recovery_ms": DEFAULT_RECOVERY_MS,
            "recovery_pooled": True,
            "pulses_zero_based": list(ALL_PULSES),
            "alignment": "presynaptic spike maximum-slope time",
            "baseline_window_ms": list(DEFAULT_BASELINE_MS),
            "response_window_ms": list(RESPONSE_WINDOW_MS),
            "filtering": "none",
        },
        "amplitude_statistic": (
            "deconvolution fit amplitude (dec_fit_reconv_amp) of every QC-passed "
            f"response, all {DEFAULT_FREQUENCY_HZ:g} Hz trains pooled regardless "
            "of recovery delay (as in the paper's Fig. 4A)"),
        "noise_statistic": "the same fit applied to the pre-stimulus baseline "
                           "(baseline_dec_fit_reconv_amp)",
        "nwb_amplitude_statistic": (
            "mean baseline-subtracted voltage in a "
            f"{2 * AMPLITUDE_HALF_WIDTH_MS:g} ms window centered on the "
            "trial-mean peak latency within the response window"),
        "nwb_peak_statistic": "per-trial maximum within the response window",
        "nwb_noise_statistic": (
            f"the NWB amplitude window applied {abs(NOISE_CENTER_MS):g} ms before "
            "the presynaptic spike"),
        "mean_waveform_window_ms": list(MEAN_WAVEFORM_WINDOW_MS),
        "waveform_trial_inclusion": "per pulse: every pooled-train trial whose "
                                    "pulse evoked exactly one well-timed spike",
        "connections": [],
    }

    def _per_pulse(per_pulse, key):
        return {str(p): np.asarray(per_pulse[p][key]).tolist() for p in per_pulse}

    for example in examples:
        group = example["group"]
        fit = fits[group]["pulses"]
        nwb = waveforms[group]
        if set(fit) != set(nwb):
            raise ValueError(f"{group}: fitted and NWB pulse sets differ")
        metadata["connections"].append({
            key: value for key, value in example.items() if key != "synapse_type"
        } | {
            "synapse_type": {"ex": "excitatory", "in": "inhibitory"}.get(
                fits[group]["synapse_type"], fits[group]["synapse_type"]),
            "n_trials": {str(p): int(fit[p]["amplitude_mv"].size) for p in fit},
            "trial_amplitude_mv": _per_pulse(fit, "amplitude_mv"),
            "trial_noise_mv": _per_pulse(fit, "noise_mv"),
            "trial_recovery_ms": _per_pulse(fit, "recovery_ms"),
            "trial_sync_rec_ext_id": {str(p): fit[p]["sync_rec_ext_id"] for p in fit},
            "waveform_time_ms": _per_pulse(nwb, "waveform_time_ms"),
            "mean_waveform_mv": _per_pulse(nwb, "mean_waveform_mv"),
            "waveform_n_trials": {str(p): len(nwb[p]["sweeps"]) for p in nwb},
            "waveform_nwb_sweep_indices": {str(p): nwb[p]["sweeps"] for p in nwb},
            "nwb_peak_latency_ms": {str(p): nwb[p]["peak_latency_ms"] for p in nwb},
            "nwb_trial_amplitude_mv": _per_pulse(nwb, "amplitude_mv"),
            "nwb_trial_peak_mv": _per_pulse(nwb, "peak_mv"),
            "nwb_trial_noise_mv": _per_pulse(nwb, "noise_mv"),
        })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "biology_fast_synaptic_trial_amplitudes.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Saved: {path}")
    return path


def plot_measured_waveforms(time_ms, waveforms, examples, out_dir=OUT_DIR):
    """Plot trial-mean waveforms for the three dynamics classes.

    Each title carries the connection's cell classes and the paper's stored
    50 Hz STP induction metric (negative: depressing, positive: facilitating)."""
    fig, axes = plt.subplots(
        len(examples), 1, figsize=(3.76, 3.70), sharex=True, sharey=True)

    for ax, example in zip(np.atleast_1d(axes), examples):
        group = example["group"]
        for pulse_number, _, color, linestyle in PULSES:
            traces = waveforms[group][pulse_number]
            ax.plot(
                time_ms, np.mean(traces, axis=0), color=color,
                linewidth=1.5, linestyle=linestyle, zorder=3)

        ax.axhline(0, color="#B5B5B5", linewidth=0.5, zorder=0)
        induction = example.get("stp_induction_50hz")
        metric = "" if induction is None else f"    STP induction = {induction:.2f}"
        ax.set_title(
            f"{example['title']}  {example['pre_class']}→{example['post_class']}{metric}",
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


def save_metadata(time_ms, waveforms, sweep_ids, examples, out_dir=OUT_DIR):
    """Save provenance and simple response summaries for the plotted traces."""
    response_mask = (time_ms >= 1.0) & (time_ms <= 12.0)
    metadata = {
        "source": "Allen Institute Synaptic Physiology raw NWB recording",
        "source_urls": [
            NWB_URL_TEMPLATE.format(experiment_id=example["experiment_id"])
            for example in examples
        ],
        "selected_from": "Campagnola et al. 2022 Fig. 4A example pairs",
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
    for example in examples:
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
            key: value for key, value in example.items() if key != "synapse_type"
        } | {
            "synapse_type": {"ex": "excitatory", "in": "inhibitory"}.get(
                example.get("synapse_type"), example.get("synapse_type")),
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
    nwb_default = default_cache_dir("waveforms") or Path("~/.cache/aisynphys/waveforms")
    parser.add_argument(
        "--cache-dir", default=str(nwb_default),
        help="directory containing/downloading the three NWB source files "
             f"(default: {nwb_default})")
    parser.add_argument(
        "--db-cache-dir", default=None,
        help="aisynphys cache directory holding the medium database used for "
             f"the fitted amplitudes; omit to use {SHARED_CACHE_DIR} when it "
             "exists, else the package default")
    args = parser.parse_args()

    # The database comes first: it tells us which NWB devices the pairs are on.
    db = _load_database(args.db_cache_dir)
    examples = resolve_examples(db)

    experiment_ids = {example["experiment_id"] for example in examples}
    nwb_paths = {
        experiment_id: _download_nwb(experiment_id, args.cache_dir)
        for experiment_id in experiment_ids
    }
    datasets = {
        experiment_id: _load_dataset(path)
        for experiment_id, path in nwb_paths.items()
    }
    time_ms, waveforms, sweep_ids = load_measured_waveforms(datasets, examples)
    for example in examples:
        group = example["group"]
        print(f"{example['title']}: {len(sweep_ids[group])} measured trials")
    plot_measured_waveforms(time_ms, waveforms, examples, out_dir=args.out_dir)
    plot_waveform_legend(out_dir=args.out_dir)
    save_metadata(time_ms, waveforms, sweep_ids, examples, out_dir=args.out_dir)
    # All 12 pulses for cartoon.plot_synaptic_train_amplitudes: fitted amplitudes
    # from the database (the paper's quantity), mean waveforms from the NWB files,
    # both pooled over every 50 Hz train as the paper does.
    waveforms = load_pulse_waveforms(datasets, examples, recovery_ms=None)
    fits = load_fit_amplitudes(db, examples)
    save_trial_amplitudes(fits, waveforms, examples, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
