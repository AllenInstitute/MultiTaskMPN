#!/usr/bin/env python
# coding: utf-8
"""Illustrate rapid cortical synaptic dynamics with Allen Institute data.

This script is independent of ``biology.py`` and its MICrONS connectivity
figure. It downloads/opens the Allen Synaptic Physiology *medium* database and
plots both a connection-level heatmap and population response trajectories for
QC-passed mouse cortical synapses during a 50 Hz, 12-pulse protocol with a
250 ms recovery interval.

The plotted values are fitted response amplitudes from real multipatch
recordings, not simulated waveforms. Amplitudes for each connection are
normalized to its median first-pulse response so excitatory and inhibitory
connections can be compared on the same axis. The script saves a compact JSON
sidecar identifying the selected connections and the exact values shown.
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


OUT_DIR = Path("cartoon_plot")
DEFAULT_FREQUENCY_HZ = 50.0
DEFAULT_RECOVERY_MS = 250.0
DEFAULT_MIN_SOURCE_EVENTS = 20
DEFAULT_MIN_TRAINS = 3
DEFAULT_N_SYNAPSES = 72
DEFAULT_STP_THRESHOLD = 0.25  # log2 change; about a 19% response change

_DEPRESSION_COLOR = "#0072B2"   # Okabe-Ito blue
_FACILITATION_COLOR = "#D55E00"  # Okabe-Ito vermillion
_PSEUDOLINEAR_COLOR = "#707070"
_NEUTRAL_COLOR = "#F7F7F7"
_DISPLAY_LIMIT = 2.0  # log2 response: 1/4x to 4x

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _load_database(cache_dir=None):
    """Load the current medium aisynphys database, downloading if necessary."""
    try:
        from aisynphys import config
        from aisynphys.database import SynphysDatabase
    except ImportError as exc:
        raise RuntimeError(
            "aisynphys is not installed. Follow the installation commands in "
            "the script handoff, then run this script in that environment."
        ) from exc

    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        config.cache_path = str(cache_dir)

        # load_current() always contacts the online version manifest, even when
        # the database is already downloaded. Prefer a complete local medium DB
        # so subsequent analysis also works from network-restricted nodes.
        local_databases = sorted(
            (path for path in (cache_dir / "database").glob(
                "synphys_*_medium.sqlite")
             if path.stat().st_size > 1024 ** 2),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if local_databases:
            local_path = local_databases[0]
            print(f"Loading cached medium database: {local_path}")
            return SynphysDatabase.load_sqlite(str(local_path))

    print("Loading the current medium Synaptic Physiology database.")
    print("The first run downloads a large SQLite file; later runs use the cache.")
    return SynphysDatabase.load_current("medium")


def _candidate_pairs(db, min_source_events):
    """Return QC-passed mouse synapses with a finite 50 Hz STP metric."""
    records = db.pair_query(
        experiment_type="standard multipatch",
        species="mouse",
        synapse=True,
        preload=["cell", "synapse"],
    ).all()

    # pair_query returns a bare Pair only when no extra entities are requested.
    # With preload enabled it returns SQLAlchemy rows whose first/named entity is
    # Pair. Keep the preload (it avoids thousands of lazy queries), but unwrap the
    # row explicitly. This works with both SQLAlchemy 1.3 KeyedTuple and 1.4 Row.
    pairs = [record if isinstance(record, db.Pair) else record.Pair
             for record in records]

    candidates = []
    for pair in pairs:
        dynamics = pair.dynamics
        if dynamics is None or not dynamics.qc_pass:
            continue
        score = dynamics.stp_induction_50hz
        n_events = dynamics.n_source_events
        if score is None or not np.isfinite(score):
            continue
        if n_events is None or int(n_events) < int(min_source_events):
            continue
        candidates.append((pair, float(score)))

    if len(candidates) < 2:
        raise RuntimeError(
            "Fewer than two QC-passed synapses have usable 50 Hz dynamics. "
            "Try lowering --min-source-events.")
    return candidates


def _normalize_pulse_frame(data, min_trains, pulse_numbers):
    """Convert one pair's fitted-event dataframe to train x pulse values."""
    matrix = data.pivot_table(
        index="sync_rec_ext_id", columns="pulse_number",
        values="dec_fit_reconv_amp", aggfunc="median")
    matrix = matrix.reindex(columns=pulse_numbers).to_numpy(dtype=float)

    # Keep only trains with a measured first response. A shared, robust scale is
    # preferable to dividing every train by its own first response: individual
    # synaptic failures can be near zero and would otherwise amplify noise by
    # tens of times. Absolute amplitude puts EPSPs and IPSPs on the same axis.
    first = np.abs(matrix[:, 0])
    valid_first = np.isfinite(first) & (first > np.finfo(float).eps)
    matrix = matrix[valid_first]
    first = first[valid_first]
    if matrix.shape[0] == 0:
        raise ValueError("no train has a finite first-pulse amplitude")
    first_pulse_scale = float(np.nanmedian(first))
    if not np.isfinite(first_pulse_scale) or first_pulse_scale <= 0:
        raise ValueError("first-pulse median cannot be used for normalization")
    normalized = np.abs(matrix) / first_pulse_scale

    per_pulse_n = np.isfinite(normalized).sum(axis=0)
    if int(per_pulse_n.min()) < int(min_trains):
        raise ValueError(
            f"only {int(per_pulse_n.min())} complete observations per pulse; "
            f"need {min_trains}")
    return normalized


def _load_pulse_matrices(candidates, db, frequency_hz, recovery_ms,
                         min_trains):
    """Load fitted amplitudes for every candidate in one database query.

    This is the event-level subset of ``aisynphys.dynamics.stim_sorted_pulse_amp``
    needed by this figure, applied to all candidate pairs at once. A single query
    avoids repeatedly joining the large pulse-response tables for each pair.
    """
    pair_ids = [int(pair.id) for pair, _ in candidates]
    recovery_s = float(recovery_ms) / 1000.0
    query = db.query(
        db.Pair.id.label("pair_id"),
        db.Synapse.synapse_type.label("synapse_type"),
        db.PulseResponseFit.dec_fit_reconv_amp,
        db.PulseResponse.ex_qc_pass,
        db.PulseResponse.in_qc_pass,
        db.StimPulse.pulse_number,
        db.MultiPatchProbe.induction_frequency,
        db.MultiPatchProbe.recovery_delay,
        db.SyncRec.ext_id.label("sync_rec_ext_id"),
    )
    query = query.join(
        db.PulseResponse, db.PulseResponseFit.pulse_response)
    query = query.join(db.Recording, db.PulseResponse.recording)
    query = query.join(db.SyncRec, db.Recording.sync_rec)
    query = query.join(
        db.PatchClampRecording, db.Recording.patch_clamp_recording)
    query = query.join(
        db.MultiPatchProbe, db.PatchClampRecording.multi_patch_probe)
    query = query.join(db.StimPulse, db.PulseResponse.stim_pulse)
    query = query.join(db.Pair, db.PulseResponse.pair)
    query = query.join(db.Synapse, db.Pair.synapse)
    query = query.filter(db.Pair.id.in_(pair_ids))
    query = query.filter(db.PatchClampRecording.clamp_mode == "ic")
    query = query.filter(
        db.MultiPatchProbe.induction_frequency.between(
            float(frequency_hz) - 0.5, float(frequency_hz) + 0.5))
    query = query.filter(
        db.MultiPatchProbe.recovery_delay.between(
            recovery_s - 5e-3, recovery_s + 5e-3))

    # aisynphys' JSON SQLAlchemy type predates statement caching. The warning is
    # harmless here and otherwise obscures the useful progress messages.
    try:
        from sqlalchemy.exc import SAWarning
    except ImportError:
        SAWarning = Warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"TypeDecorator JSONObject.*cache key",
            category=SAWarning)
        data = query.dataframe(rename_columns=False)
    if data.empty:
        raise RuntimeError(
            "No fitted pulse responses match the requested stimulus condition.")
    qc_pass = (
        ((data["synapse_type"] == "ex") & data["ex_qc_pass"].fillna(False))
        | ((data["synapse_type"] == "in") & data["in_qc_pass"].fillna(False))
    )
    data = data.loc[qc_pass & data["dec_fit_reconv_amp"].notna()]

    # The r2.1 database stores the standard multipatch train as pulses 0..11,
    # whereas some older aisynphys utilities/documentation assume 1..12.
    # Detect the convention so the script remains compatible with either.
    n_zero = int((data["pulse_number"] == 0).sum())
    n_twelve = int((data["pulse_number"] == 12).sum())
    pulse_start = 0 if n_zero > n_twelve else 1
    pulse_numbers = list(range(pulse_start, pulse_start + 12))
    print(
        f"Detected {pulse_start}-based pulse numbering; using database pulses "
        f"{pulse_numbers[0]}..{pulse_numbers[-1]}.")

    matrices = {}
    for pair_id, pair_data in data.groupby("pair_id", sort=False):
        try:
            matrices[int(pair_id)] = _normalize_pulse_frame(
                pair_data, min_trains=min_trains,
                pulse_numbers=pulse_numbers)
        except ValueError:
            continue
    if not matrices:
        raise RuntimeError(
            "No candidate pair has enough observations at all 12 pulses. "
            "Try lowering --min-trains.")
    return matrices


def _prepare_population_groups(candidates, matrices, stp_threshold):
    """Compute event-level trajectories and divide them by induction dynamics.

    The database's stored dynamics metrics were generated by older code that
    assumes 1-based pulse numbering. The r2.1 event table is predominantly
    0-based, so sorting is recalculated here directly from pulses 0..7: the
    median response at induction pulses 6--8 relative to pulse 1.
    """
    rows = []
    for pair, stored_score in candidates:
        matrix = matrices.get(int(pair.id))
        if matrix is None:
            continue
        median_curve = np.nanmedian(matrix, axis=0)
        if median_curve.shape != (12,) or not np.all(np.isfinite(median_curve)):
            continue
        reference = float(median_curve[0])
        if reference <= 0 or not np.isfinite(reference):
            continue

        relative = median_curve / reference
        if not np.all(np.isfinite(relative)) or np.any(relative <= 0):
            continue
        # Remove only severe ratio outliers before display clipping; these are
        # usually caused by a first response near the measurement noise floor.
        if float(np.nanmax(relative)) > 8.0:
            continue
        raw_log_response = np.log2(relative)
        log_response = np.clip(
            raw_log_response, -_DISPLAY_LIMIT, _DISPLAY_LIMIT)
        induction_score = float(np.nanmedian(raw_log_response[5:8]))
        heatmap_score = float(np.nanmedian(log_response[5:8]))
        rows.append({
            "pair": pair,
            "stored_stp_induction_50hz": float(stored_score),
            "induction_log2_change": induction_score,
            "heatmap_induction_log2_change": heatmap_score,
            "n_trains": int(matrix.shape[0]),
            "relative_response": relative,
            "log2_response": log_response,
        })

    threshold = float(stp_threshold)
    groups = {
        "facilitating": [
            row for row in rows
            if row["induction_log2_change"] >= threshold],
        "pseudolinear": [
            row for row in rows
            if abs(row["induction_log2_change"]) < threshold],
        "depressing": [
            row for row in rows
            if row["induction_log2_change"] <= -threshold],
    }
    for name, group in groups.items():
        if len(group) < 2:
            raise RuntimeError(
                f"Only {len(group)} {name} connections pass the log2-change "
                f"threshold {threshold:g}; lower --stp-threshold.")
    return rows, groups


def _select_heatmap_rows(rows, n_synapses):
    """Quantile-sample the full dynamics continuum for a compact heatmap."""
    if len(rows) < int(n_synapses):
        raise RuntimeError(
            f"Only {len(rows)} well-scaled connections are available; "
            f"requested {n_synapses}. Lower --n-synapses or --min-trains.")
    ranked = sorted(
        rows, key=lambda row: row["heatmap_induction_log2_change"],
        reverse=True)
    indices = np.linspace(
        0, len(ranked) - 1, int(n_synapses)).round().astype(int)
    return [ranked[index] for index in indices]


def _pulse_times_ms(frequency_hz, recovery_ms):
    """Times for eight induction pulses followed by four recovery pulses."""
    interval = 1000.0 / float(frequency_hz)
    induction = np.arange(8, dtype=float) * interval
    recovery = induction[-1] + float(recovery_ms) + np.arange(4) * interval
    return np.concatenate((induction, recovery))


def _cell_description(cell):
    """Return a compact, robust cell description for the metadata sidecar."""
    for attr in ("cre_type", "cell_class_nonsynaptic", "cell_class"):
        value = getattr(cell, attr, None)
        if value not in (None, "", "unknown"):
            return str(value)
    return "unclassified"


def _summarize_connection(row):
    pair = row["pair"]
    return {
        "pair_database_id": int(pair.id),
        "pair_external_id": str(pair.ext_id),
        "presynaptic_class": _cell_description(pair.pre_cell),
        "postsynaptic_class": _cell_description(pair.post_cell),
        "n_trains": int(row["n_trains"]),
        "recomputed_induction_log2_change": float(
            row["induction_log2_change"]),
        "displayed_induction_log2_change": float(
            row["heatmap_induction_log2_change"]),
        "stored_stp_induction_50hz": float(
            row["stored_stp_induction_50hz"]),
        "median_relative_response": row["relative_response"].tolist(),
        "displayed_log2_response": row["log2_response"].tolist(),
    }


def plot_connection_heatmap(rows, out_dir=OUT_DIR,
                            recovery_ms=DEFAULT_RECOVERY_MS):
    """Plot the compact connection-by-pulse heatmap."""
    heatmap = np.vstack([row["log2_response"] for row in rows])
    n_rows = heatmap.shape[0]
    cmap = LinearSegmentedColormap.from_list(
        "synaptic_dynamics",
        [_DEPRESSION_COLOR, _NEUTRAL_COLOR, _FACILITATION_COLOR])

    fig, ax = plt.subplots(figsize=(3.65, 3.05))
    image_kwargs = {
        "aspect": "auto",
        "interpolation": "nearest",
        "cmap": cmap,
        "vmin": -_DISPLAY_LIMIT,
        "vmax": _DISPLAY_LIMIT,
        "origin": "upper",
    }
    image = ax.imshow(
        heatmap[:, :8], extent=(-0.5, 7.5, n_rows - 0.5, -0.5),
        **image_kwargs)
    recovery_start = 8.25
    ax.imshow(
        heatmap[:, 8:],
        extent=(recovery_start, recovery_start + 4, n_rows - 0.5, -0.5),
        **image_kwargs)

    ax.add_patch(Rectangle(
        (-0.5, -0.5), 8, n_rows, fill=False, edgecolor="#555555",
        linewidth=0.45, clip_on=False))
    ax.add_patch(Rectangle(
        (recovery_start, -0.5), 4, n_rows, fill=False,
        edgecolor="#555555", linewidth=0.45, clip_on=False))
    induction_centers = np.arange(8, dtype=float)
    recovery_centers = recovery_start + 0.5 + np.arange(4, dtype=float)
    ax.set_xticks(np.concatenate((induction_centers, recovery_centers)))
    ax.set_xticklabels([str(i) for i in range(1, 13)])
    ax.tick_params(
        axis="x", top=False, labeltop=False, bottom=True, labelbottom=True,
        length=0, pad=2)
    ax.set_yticks([])
    ax.set_ylabel("Synaptic connections", labelpad=7)
    ax.set_xlim(-0.5, recovery_start + 4)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.text(
        (7.5 + recovery_start) / 2, 0.5, f"{recovery_ms:g} ms",
        transform=ax.get_xaxis_transform(), rotation=90,
        ha="center", va="center", fontsize=5.8, color="#666666")
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        image, ax=ax, fraction=0.037, pad=0.04, aspect=26,
        ticks=[-_DISPLAY_LIMIT, 0, _DISPLAY_LIMIT])
    colorbar.ax.set_yticklabels(["¼×", "1×", "4×"])
    colorbar.set_label("Response / pulse 1", rotation=270, labelpad=10)
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.tick_params(width=0.5, length=2)

    fig.subplots_adjust(left=0.17, right=0.88, bottom=0.10, top=0.985)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "biology_fast_synaptic_dynamics.png"
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")
    return png_path


def plot_population_dynamics(groups, out_dir=OUT_DIR,
                             frequency_hz=DEFAULT_FREQUENCY_HZ,
                             recovery_ms=DEFAULT_RECOVERY_MS):
    """Plot one compact population median-and-IQR trajectory panel."""
    times = _pulse_times_ms(frequency_hz, recovery_ms)
    summaries = {}
    styles = (
        ("pseudolinear", _PSEUDOLINEAR_COLOR),
        ("facilitating", _FACILITATION_COLOR),
        ("depressing", _DEPRESSION_COLOR),
    )

    fig, ax = plt.subplots(figsize=(3.65, 3.05))
    for key, color in styles:
        values = np.vstack([row["log2_response"] for row in groups[key]])
        median = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        summaries[key] = {"median": median, "q25": q25, "q75": q75}

        # Do not interpolate uncertainty across the unobserved recovery gap.
        for pulse_slice in (slice(0, 8), slice(8, 12)):
            x = times[pulse_slice]
            ax.fill_between(
                x, q25[pulse_slice], q75[pulse_slice], color=color,
                alpha=0.15, linewidth=0, zorder=1)
            ax.plot(
                x, median[pulse_slice], color=color, linewidth=1.55,
                marker="o", markersize=3.2, markerfacecolor="white",
                markeredgewidth=0.8, zorder=3)
        ax.plot(
            times[7:9], median[7:9], color=color, linewidth=0.8,
            linestyle=(0, (2, 2)), alpha=0.65, zorder=2)

    ax.axhline(0, color="#888888", linewidth=0.65,
               linestyle=(0, (2, 2)), zorder=0)
    ax.text(
        (times[7] + times[8]) / 2, 1.88, f"{recovery_ms:g} ms",
        ha="center", va="top", fontsize=6.5, color="#666666")
    ax.set_xlim(-12, times[-1] + 15)
    ax.set_ylim(-_DISPLAY_LIMIT, _DISPLAY_LIMIT)
    ax.set_xlabel("Time after first presynaptic spike (ms)")
    ax.set_ylabel("Response / pulse 1")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["¼×", "½×", "1×", "2×", "4×"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=2.5)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.45)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.17, top=0.985)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "biology_fast_synaptic_dynamics_population.png"
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")
    return png_path, summaries


def _save_metadata(heatmap_rows, groups, summaries, out_dir,
                   frequency_hz, recovery_ms, database_version,
                   stp_threshold):
    """Save the source values shared by both PNG figures."""
    times = _pulse_times_ms(frequency_hz, recovery_ms)
    metadata = {
        "source": "Allen Institute Synaptic Physiology medium database",
        "source_url": "https://brain-map.org/our-research/connectivity/"
                      "synaptic-physiology",
        "database_version": database_version,
        "normalization": "absolute fitted amplitude / median absolute "
                         "first-pulse amplitude, within each connection",
        "display_transform": "log2(relative response), clipped to [-2, 2]",
        "group_definition": "recomputed median log2 response at induction "
                            "pulses 6-8 above +threshold (facilitating) or "
                            "below -threshold (depressing); absolute change "
                            "below threshold is pseudolinear",
        "stp_threshold_log2": float(stp_threshold),
        "frequency_hz": float(frequency_hz),
        "recovery_ms": float(recovery_ms),
        "pulse_times_ms": times.tolist(),
        "heatmap_connections": [
            _summarize_connection(row) for row in heatmap_rows],
        "groups": {
            key: {
                "n_connections": len(groups[key]),
                "median_log2_response": summaries[key]["median"].tolist(),
                "q25_log2_response": summaries[key]["q25"].tolist(),
                "q75_log2_response": summaries[key]["q75"].tolist(),
                "connections": [
                    _summarize_connection(row) for row in groups[key]],
            }
            for key in ("facilitating", "pseudolinear", "depressing")
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "biology_fast_synaptic_dynamics_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print(f"Saved: {metadata_path}")
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"output directory (default: {OUT_DIR})")
    parser.add_argument("--cache-dir", default=None,
                        help="aisynphys download/cache directory; omit to use "
                             "the package default")
    parser.add_argument("--frequency-hz", type=float,
                        default=DEFAULT_FREQUENCY_HZ,
                        help=f"induction frequency (default: "
                             f"{DEFAULT_FREQUENCY_HZ:g} Hz)")
    parser.add_argument("--recovery-ms", type=float,
                        default=DEFAULT_RECOVERY_MS,
                        help=f"recovery interval (default: "
                             f"{DEFAULT_RECOVERY_MS:g} ms)")
    parser.add_argument("--min-source-events", type=int,
                        default=DEFAULT_MIN_SOURCE_EVENTS,
                        help="minimum QC-passed events used for a pair's dynamics "
                             f"metric (default: {DEFAULT_MIN_SOURCE_EVENTS})")
    parser.add_argument("--min-trains", type=int, default=DEFAULT_MIN_TRAINS,
                        help="minimum observations required at every pulse "
                             f"(default: {DEFAULT_MIN_TRAINS})")
    parser.add_argument("--n-synapses", type=int,
                        default=DEFAULT_N_SYNAPSES,
                        help="number of quantile-spaced connections in the "
                             f"heatmap (default: {DEFAULT_N_SYNAPSES})")
    parser.add_argument("--stp-threshold", type=float,
                        default=DEFAULT_STP_THRESHOLD,
                        help="minimum absolute induction log2-change used to "
                             "define facilitating/depressing groups "
                             f"(default: {DEFAULT_STP_THRESHOLD:g})")
    args = parser.parse_args()

    if args.frequency_hz <= 0 or args.recovery_ms <= 0:
        parser.error("--frequency-hz and --recovery-ms must be positive")
    if args.min_source_events < 1 or args.min_trains < 1:
        parser.error("--min-source-events and --min-trains must be positive")
    if args.n_synapses < 2:
        parser.error("--n-synapses must be at least 2")
    if args.stp_threshold < 0:
        parser.error("--stp-threshold must be nonnegative")

    db = _load_database(args.cache_dir)
    candidates = _candidate_pairs(db, args.min_source_events)
    print(f"Found {len(candidates)} QC-passed candidate synapses.")
    matrices = _load_pulse_matrices(
        candidates, db, args.frequency_hz, args.recovery_ms, args.min_trains)
    print(f"Found {len(matrices)} candidates with complete pulse-train data.")
    rows, groups = _prepare_population_groups(
        candidates, matrices, stp_threshold=args.stp_threshold)
    heatmap_rows = _select_heatmap_rows(rows, args.n_synapses)
    print(
        f"Population groups: {len(groups['facilitating'])} facilitating, "
        f"{len(groups['pseudolinear'])} pseudolinear, "
        f"{len(groups['depressing'])} depressing connections.")
    plot_connection_heatmap(
        heatmap_rows, out_dir=args.out_dir, recovery_ms=args.recovery_ms)
    _, summaries = plot_population_dynamics(
        groups, out_dir=args.out_dir,
        frequency_hz=args.frequency_hz, recovery_ms=args.recovery_ms)
    _save_metadata(
        heatmap_rows, groups, summaries, out_dir=args.out_dir,
        frequency_hz=args.frequency_hz, recovery_ms=args.recovery_ms,
        database_version=getattr(db, "version_name", None),
        stp_threshold=args.stp_threshold)


if __name__ == "__main__":
    main()
