#!/usr/bin/env python
# coding: utf-8
"""
Example neural activity from a real recording — the measured counterpart of
`cartoon.py`'s synthetic traces.

`cartoon.py` draws what we are *claiming* about population dynamics; every trace
there is band-limited noise with a fixed seed. This module draws the same figure
from an actual two-photon recording, so the claim can be shown rather than
asserted. Same visual language (the raw glyph and the annotated panel, the
`c_vals` per-unit colors, the period strip, `_save_fig`), different provenance —
and that is the whole point of keeping the two modules apart.

This file is the FIGURE layer only. Reading the MICrONS dataset and choosing what
to show live in two helpers next to the data in `biology/`, one per half of it:

    activity_helper.py       the two-photon scan: traces, stimulus schedule,
                             which units make good examples
    connectivity_helper.py   the EM cell and synapse tables: which neurons are
                             wired to which, as a square connectivity matrix

Both are imported flat, after putting `biology/` on `sys.path` — the same shim the
experiment directories use for `core/` (see e.g. `one_task/_bootstrap.py`). It has
to be a path shim rather than `from biology import ...`: this module IS `biology`
as far as Python is concerned, so the directory of the same name is unreachable by
import.

Usage:
    python biology.py                       # writes into cartoon_plot/
    python biology.py --style annotated     # the labelled panel
    python biology.py --signal fluorescence # raw F instead of deconvolved
    python biology.py --n-units 8 --duration 90
    python biology.py --connectivity        # the L2IT connectivity matrix + summary
    python biology.py --connectivity --value binary count size --order id
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

_HELPERS = Path(__file__).resolve().parent / "biology"
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))

import activity_helper as act              # noqa: E402 (needs the path shim above)
import connectivity_helper as conn         # noqa: E402

# Style, palette and savers come from the real figure module, so a panel drawn
# here reads as the same document as the rest of the paper. `_save` is cartoon.py's
# legend-aware wrapper around `_save_fig`; the two figures must save alike, so it is
# imported rather than copied. Nothing else is shared with the cartoons.
from paper_plot import (c_vals, _ONETASK_PERIOD_COLORS,      # noqa: E402
                        _add_period_strip, _MULTITASK_HEATMAP_CMAP)
from cartoon import _save                                    # noqa: E402

# Figures land beside the cartoons they replace, so a panel can be swapped for its
# measured version without hunting across directories. Filenames are `biology_*`
# against the cartoons' `cartoon_*`, so nothing there is overwritten and the two
# provenances stay legible at a glance.
OUT_DIR = Path("cartoon_plot")

# Trial-period colors, reused from the task figures (SCHEME.md family 2) so this
# panel's strip matches the model ones. The recording has no fixation/memory/
# response structure — a stimulus is either on the screen or it is not — so only
# two of the four are used: the Stimulus purple while `stim_on` is set, and the
# pale Fixation yellow for the blank screen between blocks, that being the task
# figures' "nothing is being shown yet" epoch.
_STIM_ON_COLOR = _ONETASK_PERIOD_COLORS[1]
_BLANK_COLOR = _ONETASK_PERIOD_COLORS[0]


# ─── The figure ───────────────────────────────────────────────────────────────

def _period_spans(meta, window):
    """Period-strip spans for `window`, in seconds from the window's start.

    Built from the recording's own `stim_on`, so the strip reports the stimulus
    schedule rather than illustrating one."""
    f0, f1 = window
    t = meta["frame_times"]
    t0 = t[f0]
    spans = []
    for a, b, on in act.stim_spans(meta["stim_on"][f0:f1]):
        color = _STIM_ON_COLOR if on else _BLANK_COLOR
        label = "Stimulus" if on else "Blank"
        end = t[min(f0 + b, t.size - 1)] - t0
        spans.append((t[f0 + a] - t0, end, color, label))
    return spans


def plot_example_traces(out_dir=OUT_DIR, session=act.SESSION, scan=act.SCAN,
                        n_units=5, style="raw", signal="activity",
                        duration_s=act.DEFAULT_DURATION_S, start_s=None,
                        area=act.DEFAULT_AREA, pool=act.DEFAULT_POOL,
                        show_labels=True, data_dir=act.DATA_DIR):
    """
    `n_units` recorded units over one window of the MICrONS functional scan
    `session`/`scan` — the measured version of `cartoon.plot_rich_dynamics`.

    Units are the most reliable ones of `area`, skipping any that would repeat a
    unit already drawn (`activity_helper.select_example_units`); the window is
    `duration_s` long, by default straddling a blank screen so the period strip
    shows a stimulus boundary (`activity_helper.default_window`), or starting at
    `start_s` seconds into the recording if given.

    Two styles, matching the cartoon so the two can be shown side by side:
      "raw"       — thin black traces, no axes, no labels: the schematic glyph.
      "annotated" — `c_vals` per unit, the stimulus/blank period strip, a time axis
                    in seconds and a scale bar; for when the reader is meant to
                    compare units.

    `signal` picks the trace: "activity" (deconvolved, the spiking estimate) or
    "fluorescence" (the raw calcium signal it was inferred from). `show_labels`
    toggles the per-trace unit id and oracle score in the annotated style; being
    this figure's legend, they drive the `_n` filename suffix.
    """
    if style not in ("raw", "annotated"):
        raise ValueError(f"style must be 'raw' or 'annotated'; got {style!r}")
    raw = style == "raw"
    show_labels = show_labels and not raw

    meta = act.load_session_meta(act.functional_path(session, scan, data_dir))
    window = (act.default_window(meta, duration_s) if start_s is None
              else act.window_from_start(meta, start_s, duration_s))
    f0, f1 = window
    if f1 - f0 < 2:
        raise ValueError(f"window {window} is shorter than two frames; check "
                         "`start_s` and `duration_s`.")

    rows = act.select_example_units(meta, n_units, window, area=area, pool=pool,
                                    signal=signal)
    traces = act.normalize_traces(act.load_traces(meta["path"], rows, f0, f1,
                                                  signal=signal), signal=signal)
    # Real seconds from the window's start, from the scan's own frame clock —
    # never a frame index (SCHEME.md); at 6.3 Hz a 60 s window is ~380 frames.
    t = meta["frame_times"][f0:f1] - meta["frame_times"][f0]
    duration = float(t[-1])

    n_show = len(rows)
    figsize = ((3.3, 0.52 * n_show + 0.5) if raw else (4.6, 0.40 * n_show + 0.4))
    step = 1.30 if raw else 1.55       # vertical offset between traces
    fig, ax = plt.subplots(figsize=figsize)
    for i, trace in enumerate(traces):
        base = (n_show - 1 - i) * step
        col = "black" if raw else c_vals[i % len(c_vals)]
        if not raw:
            ax.axhline(base, color="0.88", lw=0.5, zorder=1)   # each unit's zero
        ax.plot(t, base + trace, color=col, lw=0.8 if raw else 1.0, zorder=3,
                solid_joinstyle="round")
        # Name the unit at the right end, in its own color — a legend without a
        # box. The cartoon names a motif here; a recording has no motif to name, so
        # it gets the unit's identity and how reliable it is.
        if show_labels:
            ax.text(duration * 1.015, base,
                    f"unit {meta['unit_id'][rows[i]]}\n"
                    f"oracle {meta['oracle_score'][rows[i]]:.2f}",
                    color=col, fontsize=6.0, va="center", ha="left", clip_on=False)

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.35, (n_show - 1) * step + 1.25)

    if raw:
        # Naked traces: the glyph is the whole message.
        ax.axis("off")
    else:
        spans = _period_spans(meta, window)
        # Dashed boundaries wherever the stimulus goes on or off inside the window.
        for start, _, _, _ in spans[1:]:
            ax.axvline(start, color="0.5", lw=0.8, linestyle="--", zorder=2)
        # Vertical scale bar instead of y ticks: each trace is scaled to its own
        # peak, so the number on the axis would mean nothing.
        bar_x = -0.04 * duration
        ax.plot([bar_x, bar_x], [0, 1.0], color="0.25", lw=1.4, clip_on=False,
                zorder=4)
        ax.text(1.55 * bar_x, 0.5, "peak", rotation=90, va="center", ha="center",
                fontsize=6.5, color="0.25")
        ax.set_yticks([])
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10],
                                                       integer=True))
        _add_period_strip(ax, spans, xmax=duration)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = ("biology_activity_traces.png" if raw
            else "biology_example_traces.png")
    if signal != "activity":
        name = name.replace(".png", f"_{signal}.png")
    _save(fig, out_dir / name,
          extra=f"  ({n_show} {area} units, session {session} scan {scan}, "
                f"{duration:.0f} s of {signal}, {style})",
          legend=show_labels)


# ─── Connectivity matrix figure ───────────────────────────────────────────────
# A sparse binary adjacency (3.6% of entries filled) is read for WHERE the marks
# are, so the page stays white and a connection is ink — the reverse of the
# multi-task task-variance heatmaps, which are dense and can afford a dark
# background. The weighted variants keep the paper's heatmap family (magma) but
# reversed, so that "no connection" is the blank page in every variant and only
# the meaning of the ink changes.
_CONN_INK = "#1a1a1a"                      # near-black: a connection
_CONN_CMAP = _MULTITASK_HEATMAP_CMAP + "_r"   # pale (few) → dark (many)

# The ramp starts a quarter of the way in, skipping magma_r's near-white end. That
# end is unusable here: the weakest value is also the most common one (92% of these
# connections are a single synapse), so leaving it near-white erases most of the
# matrix and leaves a panel of rare outliers on an empty page. Starting at a warm
# mid-tone keeps every connection visible while the ordering still reads.
_CONN_CMAP_RANGE = (0.25, 1.0)


def _conn_ramp(levels=256):
    """The connection ramp, truncated to `_CONN_CMAP_RANGE`, with zero left blank.

    `levels` under ~10 makes it a discrete scale, one color per integer, which is
    what a synapse count wants — a continuous bar invites reading 2.5 synapses."""
    base = plt.get_cmap(_CONN_CMAP)
    cmap = mcolors.ListedColormap(
        base(np.linspace(*_CONN_CMAP_RANGE, levels)))
    cmap.set_bad("white")                  # masked = no connection = blank page
    return cmap


_CONN_VALUES = {
    "binary": ("W", "connection"),
    "count": ("syn_count", "synapses per connection"),
    "size": ("total_syn", "summed synapse size (voxels)"),
}


def plot_connectivity_matrix(out_dir=OUT_DIR, value="binary", order="depth",
                             cell_type=conn.L2IT, proofread="both", region=None,
                             data_dir=None, result=None):
    """
    The square connectivity matrix of the proofread cells of one type, as a figure.

    Row i is neuron i's axon, column j is neuron j's dendrite, so the panel is
    directed and deliberately NOT symmetric: the mark at (i, j) says i synapses
    onto j. With `proofread="both"` every neuron on both axes has a reconstructed
    axon and a complete dendrite, which is what lets a blank cell mean "these two
    do not touch" rather than "not traced".

    `value` picks what the ink encodes: "binary" (a connection exists), "count"
    (how many synapses make it up) or "size" (their summed size, log-scaled — the
    values span two orders of magnitude, so a linear ramp would show one hot pixel
    and 1500 identical faint ones).

    `order` sorts both axes together, keeping the matrix square and the diagonal
    meaningful: "depth" (by `dist_pia`, superficial → deep, so any depth structure
    in the wiring lands on the diagonal) or "id" (`pt_root_id`, i.e. no ordering
    claim at all). Pass `result` to re-draw an `l2it_connectivity` output you
    already have instead of rebuilding it.
    """
    if value not in _CONN_VALUES:
        raise ValueError(f"value must be one of {sorted(_CONN_VALUES)}; "
                         f"got {value!r}")
    if order not in ("depth", "id"):
        raise ValueError(f"order must be 'depth' or 'id'; got {order!r}")

    if result is None:
        kwargs = {} if data_dir is None else {"data_dir": data_dir}
        result = conn.l2it_connectivity(cell_type=cell_type, proofread=proofread,
                                        region=region, **kwargs)
    key, meaning = _CONN_VALUES[value]
    matrix = np.asarray(result[key], dtype=float)
    cells = result["cells"]

    # One permutation applied to both axes — anything else would break the
    # correspondence between row i and column i that makes the matrix square.
    if order == "depth":
        perm = np.argsort(cells["dist_pia"].to_numpy(), kind="stable")
        depth_um = cells["dist_pia"].to_numpy()[perm] / 1000.0
        axis_note = (f"both axes sorted by depth, {depth_um[0]:.0f}–"
                     f"{depth_um[-1]:.0f} µm from pia")
    else:
        perm = np.arange(len(cells))
        axis_note = "both axes sorted by pt_root_id"
    matrix = matrix[np.ix_(perm, perm)]
    n = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    if value == "binary":
        cmap = mcolors.ListedColormap(["white", _CONN_INK])
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    else:
        # Zeros are "no connection", not "a very small one", so they are masked to
        # the blank page instead of taking the low end of the ramp.
        masked = np.ma.masked_where(matrix <= 0, matrix)
        top = int(masked.max())
        ticks = None
        if value == "count":
            # Small integers: one color per count, and a boundary norm so the bar
            # shows the counts themselves rather than a continuum through them.
            cmap = _conn_ramp(levels=top)
            norm = mcolors.BoundaryNorm(np.arange(0.5, top + 1.5), ncolors=top)
            ticks = np.arange(1, top + 1)
        else:
            # Sizes span two orders of magnitude, so a linear ramp would show one
            # hot pixel and 1500 indistinguishable faint ones.
            cmap = _conn_ramp()
            norm = mcolors.LogNorm(vmin=max(masked.min(), 1), vmax=top)
        im = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=ticks)
        cbar.set_label(meaning, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    # No axis labels: the panel is the matrix. What the axes are (rows presynaptic,
    # columns postsynaptic) and how they are ordered belongs in the caption — it is
    # the same sentence for every variant, so printing it three times on three
    # panels is ink that carries no per-figure information. The save line below
    # reports the ordering so the caption can be written from the run.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ax.tick_params(labelsize=6)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("0.6")

    edges = int(np.asarray(result["W"]).sum())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"biology_connectivity_{value}.png"
    _save(fig, out_dir / name,
          extra=f"  ({n}x{n} {proofread}-proofread {cell_type}, {edges} "
                f"connections, {100 * edges / max(n * (n - 1), 1):.2f}% density; "
                f"rows presynaptic, columns postsynaptic, {axis_note})")
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"Directory to write into (default: {OUT_DIR}, beside "
                             "the cartoons these replace).")
    parser.add_argument("--data-dir", default=str(act.DATA_DIR),
                        help="Directory holding the scans and EM tables "
                             "(default: the biology/ folder next to this file).")
    parser.add_argument("--session", type=int, default=act.SESSION,
                        help=f"Session number (default {act.SESSION}).")
    parser.add_argument("--scan", type=int, default=act.SCAN,
                        help=f"Scan number (default {act.SCAN}).")
    parser.add_argument("--style", choices=("raw", "annotated", "both"),
                        default="both",
                        help="'raw': thin black traces, no axes — the schematic "
                             "glyph. 'annotated': colored, with the period strip, "
                             "a time axis and unit labels. Default: both.")
    parser.add_argument("--signal", choices=act.SIGNALS, default="activity",
                        help="Deconvolved activity (default) or raw fluorescence.")
    parser.add_argument("--n-units", type=int, default=5,
                        help="How many example units to draw (default 5).")
    parser.add_argument("--area", default=act.DEFAULT_AREA,
                        help="Visual area to draw units from "
                             f"(default {act.DEFAULT_AREA}).")
    parser.add_argument("--duration", type=float, default=act.DEFAULT_DURATION_S,
                        help="Window length in seconds "
                             f"(default {act.DEFAULT_DURATION_S:g}).")
    parser.add_argument("--start", type=float, default=None,
                        help="Window start in seconds into the recording (default: "
                             "centered on a blank between two stimulus blocks).")
    parser.add_argument("--no-labels", dest="show_labels", action="store_false",
                        help="Annotated style only: drop the per-trace unit labels "
                             "(saves without the '_n' suffix).")
    parser.add_argument("--connectivity", action="store_true",
                        help="Build the proofread-L2IT connectivity matrix from the "
                             "EM tables, draw it and print its summary, instead of "
                             "drawing the activity traces.")
    parser.add_argument("--cell-type", default=conn.L2IT,
                        help="--connectivity only: annotated cell type "
                             f"(default {conn.L2IT}).")
    parser.add_argument("--proofread", default="both",
                        choices=conn.PROOFREAD_MODES,
                        help="--connectivity only: which reconstruction to require "
                             "(default both: axon AND dendrite).")
    parser.add_argument("--value", nargs="+", default=["binary"],
                        choices=sorted(_CONN_VALUES),
                        help="--connectivity only: what the ink encodes; one figure "
                             "per value given (default binary).")
    parser.add_argument("--order", default="depth", choices=("depth", "id"),
                        help="--connectivity only: how both axes are sorted "
                             "(default depth, superficial to deep).")
    parser.set_defaults(show_labels=True)
    args = parser.parse_args()

    if args.connectivity:
        # Built once and re-drawn, so N figures cost one pass over the EM tables.
        result = conn.l2it_connectivity(cell_type=args.cell_type,
                                        proofread=args.proofread,
                                        data_dir=args.data_dir)
        print(conn.connectivity_summary(result))
        for value in args.value:
            plot_connectivity_matrix(out_dir=args.out_dir, value=value,
                                     order=args.order, cell_type=args.cell_type,
                                     proofread=args.proofread, result=result)
        return

    styles = (("raw", "annotated") if args.style == "both" else (args.style,))
    for style in styles:
        plot_example_traces(out_dir=args.out_dir, data_dir=args.data_dir,
                            session=args.session, scan=args.scan,
                            n_units=args.n_units, style=style,
                            signal=args.signal, duration_s=args.duration,
                            start_s=args.start, area=args.area,
                            show_labels=args.show_labels)


if __name__ == "__main__":
    main()
