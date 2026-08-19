#!/usr/bin/env python
# coding: utf-8
"""
Functional half of the MICrONS dataset: reading a two-photon scan and choosing
what to look at in it.

Everything here is data access and selection — no plotting, no matplotlib. The
figures that use it live in `biology.py` at the repository root, which imports
this module by putting `biology/` on `sys.path` (the `_bootstrap.py` idiom the
experiment directories use for `core/`). Its structural counterpart is
`connectivity_helper.py`; the two share no state, only this directory.

One scan is stored as netCDF (`functional_session_{session}_scan_{scan}.nc`). Per
unit it holds a deconvolved `activity` trace and the raw `fluorescence` it came
from, both (n_units x n_frames); per frame it holds `stim_on`, the flag saying
whether a stimulus was on the screen, plus pupil and treadmill traces. Units are
labelled by `brain_area`, `field`, `unit_id` and `oracle_score` — the
across-repeat reliability of a unit's response. See `test_new.py` for how the
scan lines up with the coregistration table and the wider dataset.

Read with **h5py**, not xarray: netCDF4 is HDF5 underneath, no netCDF backend is
installed in this environment, and the scan is >5 GB — h5py reads one unit's
window as a single small contiguous slice instead of pulling the array into
memory.
"""
from pathlib import Path

import h5py
import numpy as np

# The recordings sit beside this file, so locate them from it rather than from the
# working directory: `biology.py` runs from the repository root, but a notebook or
# a one-off script need not. Every loader still takes `data_dir` to override it.
DATA_DIR = Path(__file__).resolve().parent

# The scan read by default. Both the functional and the movie file are named after
# it; only the functional one is read here.
SESSION, SCAN = 4, 7

# The two per-unit traces the scan carries.
SIGNALS = ("activity", "fluorescence")

# Which units are eligible as examples, and how many are ranked before picking.
# The pool is the most reliable units of one area (highest `oracle_score`); the
# examples are then taken from the top of it, see `select_example_units`.
DEFAULT_AREA = "V1"
DEFAULT_POOL = 120

# A unit is not an example of anything if one frame is most of what it did in the
# window. Ceiling on that frame's share of the unit's total signal; the pool's
# median share is ~0.13, so this drops the worst ~15% rather than the typical unit.
MAX_EVENT_SHARE = 0.25

# Two example units showing the same response teach the reader nothing twice, so a
# candidate is skipped if it correlates with an already-chosen unit above this. Set
# where neighboring V1 units with overlapping receptive fields land, well above the
# ~0.1-0.3 of two units that merely share the stimulus.
MAX_PAIR_CORR = 0.6

# Default window length, in seconds of recording.
DEFAULT_DURATION_S = 60.0


# ─── Data access ──────────────────────────────────────────────────────────────

def functional_path(session=SESSION, scan=SCAN, data_dir=DATA_DIR):
    """Path of the functional netCDF for one session/scan."""
    return Path(data_dir) / f"functional_session_{session}_scan_{scan}.nc"


def load_session_meta(path):
    """Everything about a scan except the traces themselves.

    Reads only the per-unit and per-frame vectors — a few hundred kB — so the
    window and the example units can be chosen before touching the (n_units x
    n_frames) activity array. Returns a dict with `fps`, `frame_times` (s),
    `stim_on`, `oracle_score`, `brain_area`, `field`, `unit_id` and `n_frames`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The functional scans live in {DATA_DIR} and are "
            "not tracked by git; see the module docstring.")
    with h5py.File(path, "r") as h:
        frame_times = h["frame_times"][:]
        meta = {
            "path": path,
            # Frame rate is a property of the scan, so read it rather than assume
            # one (the SCHEME.md rule for dt, applied to a recording).
            "fps": float(h.attrs["fps"]),
            "frame_times": frame_times,
            "stim_on": h["stim_on"][:].astype(bool),
            "oracle_score": h["oracle_score"][:],
            "brain_area": h["brain_area"][:].astype(str),
            "field": h["field"][:],
            "unit_id": h["unit_id"][:],
            "n_units": h["activity"].shape[0],
            "n_frames": h["activity"].shape[1],
        }
    return meta


def load_traces(path, rows, f0, f1, signal="activity"):
    """Frames `f0:f1` of the units at `rows`, as an (len(rows), f1 - f0) array.

    One contiguous read per unit: the arrays are frame-major within a unit, so a
    window of a single unit is a few kB off disk however large the file is. Row
    order follows `rows` as given, not the file's."""
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}; got {signal!r}")
    with h5py.File(path, "r") as h:
        ds = h[signal]
        return np.stack([ds[int(r), f0:f1] for r in rows])


# ─── Choosing what to draw ────────────────────────────────────────────────────

def stim_spans(stim_on):
    """`stim_on` as a list of (start_frame, end_frame, is_on) blocks."""
    stim_on = np.asarray(stim_on, dtype=bool)
    edges = np.flatnonzero(np.diff(stim_on)) + 1
    bounds = np.concatenate(([0], edges, [stim_on.size]))
    return [(int(a), int(b), bool(stim_on[a]))
            for a, b in zip(bounds[:-1], bounds[1:])]


def default_window(meta, duration_s=DEFAULT_DURATION_S):
    """A `duration_s` window centered on a blank between two stimulus blocks.

    Centering on a gap rather than sitting inside one block is what puts a visible
    boundary in the period strip: the window then holds the end of one block, the
    blank screen, and the start of the next, so the strip carries the same
    blank→stimulus structure the task figures' period bar does.

    Only INTERIOR blanks count — ones with a stimulus block on both sides. The
    scan's longest blank by far is the tail after the last block (this one runs
    ~21 min), and centering on that would put the whole window in a gray screen: a
    figure of a stimulus-driven population with no stimulus in it. Between the
    interior blanks (all the same length here) the earliest wins, so the window is
    stable if the scan is re-exported. Falls back to the middle of the recording if
    the scan has no interior blank at all."""
    n_frames = meta["n_frames"]
    width = int(round(duration_s * meta["fps"]))
    spans = stim_spans(meta["stim_on"])
    interior = [(b - a, -a, a, b)                     # -a: earliest wins a tie
                for (a, b, on), before, after in zip(spans[1:-1], spans, spans[2:])
                if not on and before[2] and after[2]]
    if interior:
        _, _, a, b = max(interior)
        center = (a + b) // 2
    else:
        center = n_frames // 2
    f0 = int(np.clip(center - width // 2, 0, max(n_frames - width, 0)))
    return f0, min(f0 + width, n_frames)


def window_from_start(meta, start_s, duration_s=DEFAULT_DURATION_S):
    """The `duration_s` window beginning `start_s` seconds into the recording."""
    f0 = int(np.searchsorted(meta["frame_times"], start_s))
    return f0, min(f0 + int(round(duration_s * meta["fps"])), meta["n_frames"])


def select_example_units(meta, n_units, window, area=DEFAULT_AREA,
                         pool=DEFAULT_POOL, signal="activity",
                         max_event_share=MAX_EVENT_SHARE, max_corr=MAX_PAIR_CORR):
    """`n_units` example units: reliable, active in the window, and non-redundant.

    Three criteria, in this order. **Reliable** — rank the units of `area` by
    `oracle_score` (how repeatable a unit's response is across repeats of the same
    stimulus) and keep the top `pool`; a unit whose trace is mostly noise would
    make the figure look busy without showing anything. **Active in this window** —
    drop units whose single largest frame is more than `max_event_share` of
    everything they did in it, since one transient in a minute is a flat line with
    a spike on it, not an example of a response. **Not redundant** — walk down what
    survives in reliability order and accept a unit only if its windowed trace
    correlates with every already-accepted one below `max_corr`; if that leaves too
    few, the shortfall is filled from the rejects, still in reliability order.

    Reliability leads and correlation only vetoes, rather than the other way round,
    for a reason worth stating: "pick the units least like each other" sounds like
    the right way to show heterogeneity, but a trace that resembles nothing is most
    easily achieved by being noise, so that objective walks straight to the worst
    units in the pool. Ranking by reliability and using correlation as a veto gets
    the variety without paying for it in signal.

    Deterministic — no RNG anywhere in the selection. Returns row indices into the
    file's unit axis, ordered as chosen.
    """
    if n_units < 1:
        raise ValueError(f"n_units must be >= 1; got {n_units}")
    oracle = meta["oracle_score"]
    eligible = np.flatnonzero((meta["brain_area"] == area) & ~np.isnan(oracle))
    if eligible.size == 0:
        raise ValueError(f"no units with an oracle score in area {area!r}; "
                         f"areas present: {sorted(set(meta['brain_area']))}")
    ranked = eligible[np.argsort(-oracle[eligible],
                                 kind="stable")][:max(pool, n_units)]

    f0, f1 = window
    traces = load_traces(meta["path"], ranked, f0, f1, signal=signal)
    # Silent units carry no shape at all, and one-event units are the same thing
    # with a spike in it; neither is an example of a response, and both sail
    # through the correlation veto below by resembling nothing. Shares are measured
    # off each unit's own floor, so a fluorescence baseline does not dilute them.
    floored = traces - traces.min(axis=1, keepdims=True)
    event_share = floored.max(axis=1) / np.maximum(floored.sum(axis=1), 1e-9)
    keep = (traces.std(axis=1) > 0) & (event_share <= max_event_share)
    ranked, traces = ranked[keep], traces[keep]
    if ranked.size < n_units:
        raise ValueError(f"only {ranked.size} usable units in the pool for "
                         f"{n_units} examples; raise `pool` or `max_event_share`.")

    corr = np.abs(np.corrcoef(traces))
    chosen, rejected = [], []
    for i in range(len(ranked)):                   # reliability order
        if len(chosen) == n_units:
            break
        (chosen if not chosen or corr[i, chosen].max() < max_corr
         else rejected).append(i)
    chosen += rejected[:n_units - len(chosen)]     # too few: relax the veto
    return ranked[chosen]


def normalize_traces(traces, signal="activity"):
    """Put every trace on a common 0–1 display scale, one unit per trace.

    The two signals need different floors. Deconvolved `activity` already sits on
    zero between events, so it is only divided by its peak — subtracting anything
    would fake a baseline it does not have. Raw `fluorescence` sits on an arbitrary
    per-unit offset (it is neuropil-subtracted, so even the sign is arbitrary), so
    its 5th percentile is taken as that unit's baseline first.

    The scale is per unit and within the window, so the figure shows the *shape* of
    each response; it says nothing about which unit fired more."""
    traces = np.asarray(traces, dtype=float)
    if signal == "fluorescence":
        traces = traces - np.percentile(traces, 5, axis=1, keepdims=True)
    peak = np.max(np.abs(traces), axis=1, keepdims=True)
    return traces / np.maximum(peak, 1e-9)
