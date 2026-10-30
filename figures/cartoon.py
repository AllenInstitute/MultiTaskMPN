#!/usr/bin/env python
# coding: utf-8
"""
Rhetorical cartoons for the paper — figures that make a point rather than report a
measurement.

Nothing here reads a run. Every trace is SYNTHETIC and deterministic (fixed RNG
seed), so these figures are reproducible but must never be captioned as data. Keep
that separation strict: `paper_plot.py` shows what the network did, `cartoon.py`
shows what we are claiming about it.

Four cartoons so far: `plot_rich_dynamics` (heterogeneous single-unit responses in
one trial), `plot_fast_slow_curves` (two families of curves, fast and slow, the
ingredients of a richer dynamics), `plot_mouse_shape` (the mouse background silhouette,
traced from `cartoon_plot/mouse.png` so it can be redrawn, recolored and composed
with other panels instead of pasted as a bitmap) and `plot_synapse_lesion_schematic`
(the effective weight of one synapse over a trial, intact and with its modulation
frozen, the freeze_M intervention).

One exception to "synthetic": `plot_synaptic_train_amplitudes` redraws MEASURED
Allen Synaptic Physiology recordings (extracted by
`biology/biology_fast_synaptic_waveforms.py` into `cartoon_plot/`) in the layout of
Campagnola et al. 2022, Science 375, eabj5861, Fig. 4A. It is a biological
reference beside the cartoons, not a claim about our networks, and must be
captioned as data.

Style is imported from `paper_plot` rather than copied — rcParams, the categorical
palette (`c_vals`), the trial-period colors, the period strip and `_save_fig` — so
a cartoon dropped beside a real figure reads as the same document. See SCHEME.md
for the color families; the ten units here are a plain categorical distinction, so
they use `c_vals` and never the stimulus rainbow.

Usage:
    python figures/cartoon.py                  # writes into cartoon_plot/
    python figures/cartoon.py --out-dir /tmp   # somewhere else
    python figures/cartoon.py --figure mouse   # just the mouse silhouette
    python figures/cartoon.py --figure synaptic   # measured PSP trains (Fig. 4A layout)
    python figures/cartoon.py --figure lesion     # one synapse with its modulation frozen
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

# Style, palette and savers come from the real figure module so the two agree by
# construction. Importing it also applies its global rcParams.
import _bootstrap  # noqa: F401  (puts the repository root on sys.path for paper_plot)
import paper_plot as pp
from paper_plot import (c_vals, _ONETASK_PERIOD_COLORS,
                       _add_period_strip, _save_fig)

# Cartoons get their own output directory, kept separate from paper_plot/: these
# are claims, not measurements, so nothing here should land beside a real figure.
OUT_DIR = Path("cartoon_plot")

# Simulation step in ms — the project-wide convention (SCHEME.md). Trial time is
# always shown in ms, never in raw step index.
DT = 40

# Trial layout in STEPS: fixation | stimulus | memory | response.
FIX_END, STIM_END, DELAY_END, T_END = 12, 28, 64, 80

# Traces are drawn on a finer grid than the simulation's own steps, purely so the
# fast wiggle reads as recorded activity rather than as a smooth curve. The trial
# coordinate stays in step units, so the period marks still line up.
N_SAMPLES = 420


def _save(fig, out_path, extra="", legend=False):
    """Save through `paper_plot._save_fig` with this figure's own legend state.

    `_save_fig` appends its `_n` suffix from paper_plot's global SHOW_LEGEND, which
    no cartoon reads — each figure here knows whether it carries a legend, so drive
    the flag from that and restore it afterwards."""
    was = pp.SHOW_LEGEND
    pp.SHOW_LEGEND = bool(legend)
    try:
        _save_fig(fig, out_path, extra=extra)
    finally:
        pp.SHOW_LEGEND = was


def _smooth_noise(rng, n, scale, width=3.0):
    """Band-limited wiggle, so a trace looks recorded rather than drawn — white
    noise reads as spiky, and unsmoothed noise cannot be scaled sensibly."""
    pad = 3 * int(max(width, 1)) + 1
    x = rng.randn(n + 2 * pad)
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    k /= k.sum()
    return scale * np.convolve(x, k, mode="same")[pad:pad + n]


def _sig(t, center, width):
    """Smooth 0→1 step."""
    return 1.0 / (1.0 + np.exp(-(t - center) / width))


def _bump(t, center, width):
    """Smooth 0→1→0 bump."""
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def _motifs(t):
    """Ten single-unit response motifs, each a different way of being useful in a
    delayed-response trial. The variety IS the rhetorical point: one recurrent
    network, one task, and no two units doing the same thing in time.

    Returns a list of (label, trace) with every trace on a comparable scale."""
    fix, stim, delay = FIX_END, STIM_END, DELAY_END
    return [
        # Locked to stimulus onset and gone before the delay is over.
        ("transient", 1.05 * _bump(t, stim - 12, 2.6)),
        # The textbook memory unit: steps up at the stimulus, holds, releases at go.
        ("persistent", 0.95 * (_sig(t, fix + 2, 1.6) - _sig(t, delay, 1.6))),
        # Anticipatory ramp — carries time-to-go rather than the stimulus.
        ("ramp up", 0.9 * _sig(t, delay - 8, 7.0)),
        # Decaying memory: informative early in the delay, faded by the end.
        ("ramp down", 0.85 * (_sig(t, fix + 2, 1.5) * np.exp(-(t - fix) / 26.0)
                              * (t > fix))),
        # Rhythmic, strongest while the memory is held.
        ("oscillatory", 0.55 * np.sin(2 * np.pi * t / 9.0)
         * _bump(t, (stim + delay) / 2, 16.0)),
        # Silent until the go signal, then a burst — the readout side.
        ("response", 1.0 * _bump(t, delay + 5, 3.4)),
        # A bump in the middle of the delay: a basis element for elapsed time.
        ("delay bump", 0.9 * _bump(t, (stim + delay) / 2, 5.5)),
        # Mixed selectivity: the same unit reports stimulus AND response.
        ("mixed", 0.7 * _bump(t, stim - 11, 2.4) + 0.8 * _bump(t, delay + 6, 3.0)),
        # Suppressed by the stimulus, recovers — inhibition carries information too.
        # Kept shallower than the baseline so the trace stays positive, the way a
        # rate does: a unit can be silenced, not driven below zero.
        ("suppressed", -0.40 * _bump(t, stim - 9, 5.0)),
        # Slow drift across the whole trial: an integrator with a long time constant.
        ("integrator", 0.85 * np.tanh((t - fix) / 26.0) * (t > fix)),
    ]


# Order in which motifs enter the figure as `n_units` grows, chosen so that ANY
# prefix is a good spread over trial time rather than three variants of the same
# shape. Display order stays the canonical `_motifs` order, so the figure still
# reads top-to-bottom as onset → hold → ramp → mid-delay → readout.
_MOTIF_PRIORITY = ("persistent", "transient", "delay bump", "ramp up", "response",
                   "oscillatory", "ramp down", "mixed", "suppressed", "integrator")


def _pick_motifs(motifs, n_units):
    """The `n_units` most complementary motifs, kept in canonical display order."""
    if n_units is None or n_units >= len(motifs):
        return motifs
    if n_units < 1:
        raise ValueError(f"n_units must be >= 1; got {n_units}")
    keep = set(_MOTIF_PRIORITY[:n_units])
    return [m for m in motifs if m[0] in keep]


def _pink_trace(rng, n, octaves=7, w0=1.5, exponent=0.95, sharpen=1.9):
    """One naturalistic activity trace: 1/f-like noise, floored and sharpened.

    Real activity traces are not clean bumps — they are irregular and multi-peaked,
    with structure at every timescale at once. Summing band-limited noise over
    octaves with slower scales weighted more (`exponent`) gives exactly that: fast
    zigzag riding on slow humps of no fixed shape. Shifting to a zero floor and
    raising to `sharpen` > 1 flattens the quiet stretches and keeps the excursions,
    the way a rate or ΔF/F trace sits on a baseline and rises off it.

    Each call is an independent draw, so a stack of these reads as a handful of
    different recorded units rather than one shape repeated."""
    y = np.zeros(n)
    for k in range(octaves):
        width = w0 * (2.0 ** k)
        y += _smooth_noise(rng, n, width ** exponent, width=width)
    y -= y.min()
    y /= max(y.max(), 1e-9)
    return y ** sharpen


def _activity(rng, envelope, baseline=0.45, fast=0.17, slow=0.09):
    """Turn a slow motif envelope into something that reads as a recorded trace.

    Three ingredients, in the order they matter visually: the motif itself (what
    the unit is *for*), a slow random component so no two traces share a shape even
    where their motifs agree, and a fast band-limited wiggle for the recorded
    texture. A positive baseline keeps the trace mostly above zero, the way a rate
    or a ΔF/F trace sits on a floor, without hard-clipping it flat."""
    n = envelope.size
    y = (baseline + envelope
         + _smooth_noise(rng, n, slow, width=0.055 * n)
         + _smooth_noise(rng, n, fast, width=0.004 * n))
    return y / max(np.abs(y).max(), 1e-9)


def plot_rich_dynamics(out_dir=OUT_DIR, seed=0, style="raw", n_units=5,
                       show_labels=True):
    """
    Cartoon: `n_units` synthetic units in one delayed-response trial, illustrating
    that a single network's population response is dynamically heterogeneous —
    transient, persistent, ramping, oscillatory, response-locked and suppressed
    units all coexist under one task.

    `n_units` (default 5) selects from ten motifs by `_MOTIF_PRIORITY`, so a small
    figure still spans trial time instead of repeating one shape.

    Two styles of the same traces:
      "raw"       — the schematic-glyph look: thin black lines, no axes, no labels,
                    fast wiggle on top of each motif. Reads as "neural activity"
                    at a glance and drops into a slide or a schematic panel.
      "annotated" — the same traces as a figure: `c_vals` per unit, the trial-period
                    strip, dashed period boundaries, a time axis in ms and a scale
                    bar. Use when the reader is meant to compare motifs.

    Deliberately monochrome in "raw": unit identity carries no meaning here, so
    per-unit color would be decoration. "annotated" follows SCHEME.md — `c_vals`
    for the units (a plain categorical distinction, never the stimulus rainbow) and
    `_ONETASK_PERIOD_COLORS` for the period strip.

    `show_labels` toggles the per-trace motif names in the annotated style. They do
    the work of a legend, so the filename follows the project's convention:
    labelled → `..._n.png`, unlabelled → `....png`.
    """
    if style not in ("raw", "annotated"):
        raise ValueError(f"style must be 'raw' or 'annotated'; got {style!r}")
    raw = style == "raw"
    show_labels = show_labels and not raw

    rng = np.random.RandomState(seed)
    # Trial coordinate stays in step units; only the sampling is finer.
    t = np.linspace(0, T_END - 1, N_SAMPLES)
    if raw:
        # A texture glyph: independent 1/f-like traces, irregular and multi-peaked.
        # No motif is asserted, so nothing here is labelled or colored.
        motifs = [(None, _pink_trace(rng, N_SAMPLES)) for _ in range(n_units)]
    else:
        # Subset BEFORE drawing the noise, so a given seed gives each kept unit the
        # same trace whatever `n_units` is — the 5-unit figure is the 10-unit figure
        # with rows removed, not a different draw.
        motifs = [(lab, _activity(rng, env))
                  for lab, env in _pick_motifs(_motifs(t), n_units)]

    period_spans = [
        (0, FIX_END, _ONETASK_PERIOD_COLORS[0], "Fixation"),
        (FIX_END, STIM_END, _ONETASK_PERIOD_COLORS[1], "Stimulus"),
        (STIM_END, DELAY_END, _ONETASK_PERIOD_COLORS[2], "Memory"),
        (DELAY_END, None, _ONETASK_PERIOD_COLORS[3], "Response"),
    ]

    n_show = len(motifs)
    figsize = ((3.3, 0.52 * n_show + 0.5) if raw else (4.6, 0.40 * n_show + 0.4))
    step = 1.30 if raw else 1.55       # vertical offset between traces
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, trace) in enumerate(motifs):
        base = (len(motifs) - 1 - i) * step
        col = "black" if raw else c_vals[i % len(c_vals)]
        if not raw:
            ax.axhline(base, color="0.88", lw=0.5, zorder=1)   # each unit's zero
        ax.plot(t, base + trace, color=col, lw=0.8 if raw else 1.0, zorder=3,
                solid_joinstyle="round")
        # Name the motif at the right end, in the trace's own color: it does the
        # work of a legend without a box, and keeps the label next to its curve.
        if show_labels:
            ax.text(T_END + 1.0, base, label, color=col, fontsize=6.5,
                    va="center", ha="left", clip_on=False)

    ax.set_xlim(0, T_END - 1)
    ax.set_ylim(-0.35, (len(motifs) - 1) * step + 1.25)

    if raw:
        # Naked traces: the glyph is the whole message, so no axes, ticks, period
        # marks or labels compete with it.
        ax.axis("off")
    else:
        # Dashed period boundaries, matching the example-trial figures.
        for start, _, _, _ in period_spans[1:]:
            ax.axvline(start, color="0.5", lw=0.8, linestyle="--", zorder=2)
        # Vertical scale bar instead of y ticks: the units are arbitrary, the point
        # is the shape of each trace, not its value.
        ax.plot([-3.2, -3.2], [0, 1.0], color="0.25", lw=1.4, clip_on=False,
                zorder=4)
        ax.text(-4.4, 0.5, "1 a.u.", rotation=90, va="center", ha="center",
                fontsize=6.5, color="0.25")
        ax.set_yticks([])
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        # Ticks every 20 steps so the ms labels land on round numbers (index * dt).
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _p: f"{x * DT:.0f}"))
        _add_period_strip(ax, period_spans, xmax=T_END - 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = ("cartoon_activity_traces.png" if raw
            else "cartoon_rich_dynamics.png")
    # The motif labels are this figure's legend, so they drive the `_n` suffix.
    _save(fig, out_dir / name, extra=f"  ({len(motifs)} synthetic units, {style})",
          legend=show_labels)


# ─── Fast and slow timescales ─────────────────────────────────────────────────
# Two families of curves, distinguished by nothing but their timescale. Time is in
# ms (SCHEME.md); the two correlation times below are what "fast" and "slow" mean
# here, a factor of ~13 apart so the contrast needs no annotation.
FAST_TAU_MS = 70
SLOW_TAU_MS = 900
FAST_SLOW_T_MS = 3200          # trial length drawn, = T_END steps x DT

# Family colors: hue carries the family, lightness separates members within it
# (SCHEME.md family 4 — a plain categorical distinction).
_FAST_COLOR = c_vals[0]        # red
_SLOW_COLOR = c_vals[2]        # green


def _lighten(color, frac):
    """Blend `color` toward white by `frac` (0 = unchanged, 1 = white)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def _tau_width(tau_ms, n_samples, t_ms):
    """Gaussian smoothing width, in samples, for a correlation time of `tau_ms`."""
    return tau_ms * n_samples / t_ms


def _unit(y):
    """Center a curve and scale it to max |y| = 1.

    Centering matters once curves share one y axis: an off-center draw would other-
    wise crowd into half the panel and read as an offset rather than as a timescale."""
    y = y - y.mean()
    return y / max(np.abs(y).max(), 1e-9)


def _draw_curve_family(ax, t, curves, colors, lw=1.1, ylim=1.15):
    """Overlay one family of curves on `ax`, all sharing that panel's y axis.

    Overlaid rather than offset into rows: with a common y axis the curves are
    directly comparable to each other, and the panel reads as one family sampled
    several times instead of as several separate signals. `ylim` is the same for
    every family, so the only visible difference between panels is the timescale.

    No axes, ticks or baseline — the shape of the curves is the whole message, and
    the time and amplitude units are arbitrary, so any frame would only add ink."""
    for curve, color in zip(curves, colors):
        ax.plot(t, curve, color=color, lw=lw, zorder=3, solid_joinstyle="round")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-ylim, ylim)
    ax.set_axis_off()


def plot_fast_slow_curves(out_dir=OUT_DIR, seed=1, n_fast=3, n_slow=3):
    """
    Cartoon: two families of curves that differ only in their timescale — slow,
    smooth curves above fast, busy ones. Together they are the two ingredients of a
    richer dynamics; this figure shows just the ingredients.

    Each panel overlays its family on a single shared y axis, and both panels use
    the same limits, so a reader comparing them sees a difference in timescale and
    nothing else. Deliberately bare — no axes, titles, labels or legend, just the
    curves, stacked close so the two families read as one figure — and sized to
    match the width of the activity-trace cartoon so the two can sit side by side.
    The curves are band-limited noise at `SLOW_TAU_MS` / `FAST_TAU_MS`.
    """
    rng = np.random.RandomState(seed)
    n = N_SAMPLES
    t = np.linspace(0, FAST_SLOW_T_MS, n)
    slow = [_unit(_smooth_noise(rng, n, 1.0,
                                width=_tau_width(SLOW_TAU_MS, n, FAST_SLOW_T_MS)))
            for _ in range(n_slow)]
    fast = [_unit(_smooth_noise(rng, n, 1.0,
                                width=_tau_width(FAST_TAU_MS, n, FAST_SLOW_T_MS)))
            for _ in range(n_fast)]

    # A family reads as one thing if its members share a hue; lightness keeps them
    # individually followable.
    def _shades(color, k):
        return [_lighten(color, f) for f in np.linspace(0, 0.45, k)]

    # Width matches the activity-trace cartoon (3.3 in) so the figures pair up.
    # `hspace` sets the band of white between the two families — the only thing
    # separating them, since neither panel has a frame; the figure height carries it
    # so that widening the gap does not shrink the curves.
    fig, (ax_slow, ax_fast) = plt.subplots(2, 1, figsize=(3.3, 2.9), sharex=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.34)
    _draw_curve_family(ax_slow, t, slow, _shades(_SLOW_COLOR, n_slow))
    _draw_curve_family(ax_fast, t, fast, _shades(_FAST_COLOR, n_fast))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "cartoon_fast_slow.png",
          extra=f"  ({n_slow} slow, τ≈{SLOW_TAU_MS} ms; "
                f"{n_fast} fast, τ≈{FAST_TAU_MS} ms)")


# ─── Mouse silhouette ─────────────────────────────────────────────────────────
# Anchor points traced from the reference figure `cartoon_plot/mouse.png` — the
# gray shape *behind* its brain and gaze cone — in that image's own pixel frame:
# x to the right, y DOWN, canvas 342 x 372. Listed clockwise from the tip of the
# ear: ear, back, right flank, belly, then up the left flank to the ear notch.
#
# The trace is deliberately rough. In the reference the shape's fill fades to pure
# white toward the lower right, so its boundary there is white-on-white and cannot
# be measured — those anchors continue the visible curve rather than recover it,
# and the fill has faded out by the time they are reached. The wedge the gaze cone
# cuts out of the reference is *not* reproduced: the body continues under it.
_MOUSE_PNG_SIZE = (342, 372)
_MOUSE_OUTLINE = (
    (179, 16), (194, 26), (200, 41),                       # ear
    (219, 43), (243, 41), (266, 49), (282, 64),            # shoulder / back
    (288, 92), (284, 124), (277, 154), (267, 178),         # right flank
    (248, 192), (215, 199), (180, 201), (146, 199),        # belly, front to back
    (120, 200), (95, 197), (72, 190), (52, 186),
    (36, 178), (24, 165), (17, 151), (20, 136),            # rump
    (33, 123), (48, 112), (62, 100), (77, 88), (92, 76),   # left flank, up
    (108, 66), (124, 58), (139, 51),
    (146, 34), (161, 22),                                  # ear notch
)

# Fill: the reference's linear gradient, measured on mouse.png as a neutral gray
# ≈ 104/255 at pixel (20, 150) brightening to white by x ≈ 210, along a direction
# ~11° below horizontal. Reproduced as an alpha ramp on one gray rather than a
# ramp to white, so the shape fades into the page whatever color the page is.
_MOUSE_FILL = "#686868"
_MOUSE_GRAD_OPAQUE = (10.0, 148.0)      # pixel where the fill is fully opaque
_MOUSE_GRAD_CLEAR = (214.0, 186.0)      # pixel where it has faded out


def _catmull_rom_closed(points, per_segment=24):
    """Smooth closed curve through every point in `points` (uniform Catmull-Rom).

    The traced anchors are sparse and unevenly spaced, so a polygon through them
    reads as faceted. Catmull-Rom passes exactly through each anchor with a
    continuous tangent, which is what makes the result read as a drawn shape —
    and it keeps the anchors editable, since moving one moves the curve locally."""
    p = np.asarray(points, float)
    n = len(p)
    t = np.linspace(0, 1, per_segment, endpoint=False)[:, None]
    segments = []
    for i in range(n):
        p0, p1, p2, p3 = p[i - 1], p[i], p[(i + 1) % n], p[(i + 2) % n]
        segments.append(0.5 * (2 * p1 + (p2 - p0) * t
                               + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                               + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(segments)


def _mouse_gradient_alpha(n=256):
    """The fill's alpha field on a grid over the canvas, plus its extent.

    Alpha is 1 at `_MOUSE_GRAD_OPAQUE`, falls linearly along the line joining the
    two gradient anchors, and is 0 from `_MOUSE_GRAD_CLEAR` on. Returned in the
    y-up frame the figure is drawn in, ready for `imshow(..., origin="lower")`."""
    w, h = _MOUSE_PNG_SIZE
    x = np.linspace(0, w, n)
    y = np.linspace(0, h, n)
    x0, y0 = _MOUSE_GRAD_OPAQUE[0], h - _MOUSE_GRAD_OPAQUE[1]
    dx, dy = (_MOUSE_GRAD_CLEAR[0] - x0), (h - _MOUSE_GRAD_CLEAR[1]) - y0
    # Projection onto the gradient axis, in units of its length.
    u = ((x[None, :] - x0) * dx + (y[:, None] - y0) * dy) / (dx * dx + dy * dy)
    return np.clip(1.0 - u, 0.0, 1.0), (0, w, 0, h)


def draw_mouse_shape(ax, style="gradient", color=_MOUSE_FILL, zorder=0):
    """Draw the mouse background silhouette into `ax` and return its closed curve.

    Meant to be reusable: the same shape can back another cartoon panel (a brain,
    a gaze cone, a stimulus) instead of only standing alone. The curve is returned
    as an (N, 2) array in the axes' data coordinates — the reference image's pixel
    frame with y flipped up, so the mouse sits upright and 1 unit = 1 source pixel.

    Styles: "gradient" (as in the reference: gray at the head and rump, fading out
    toward the lower right), "flat" (one solid gray — for use as a mask or a
    silhouette), "outline" (the traced curve only, i.e. just the rough shape)."""
    if style not in ("gradient", "flat", "outline"):
        raise ValueError("style must be 'gradient', 'flat' or 'outline'; "
                         f"got {style!r}")
    h = _MOUSE_PNG_SIZE[1]
    curve = _catmull_rom_closed(_MOUSE_OUTLINE)
    xy = np.c_[curve[:, 0], h - curve[:, 1]]        # pixel frame → y-up

    if style == "outline":
        ax.add_patch(Polygon(xy, closed=True, facecolor="none", edgecolor=color,
                             lw=1.2, joinstyle="round", zorder=zorder))
    elif style == "flat":
        ax.add_patch(Polygon(xy, closed=True, facecolor=color, edgecolor="none",
                             zorder=zorder))
    else:
        # The gradient is an image clipped to the shape: one linear ramp, so the
        # fill cannot band or step the way a stack of nested patches would.
        clip = Polygon(xy, closed=True, facecolor="none", edgecolor="none",
                       zorder=zorder)
        ax.add_patch(clip)
        alpha, extent = _mouse_gradient_alpha()
        rgba = np.empty(alpha.shape + (4,))
        rgba[..., :3] = to_rgb(color)
        rgba[..., 3] = alpha
        img = ax.imshow(rgba, extent=extent, origin="lower",
                        interpolation="bilinear", zorder=zorder)
        img.set_clip_path(clip)
    return xy


def plot_mouse_shape(out_dir=OUT_DIR, style="gradient", margin=8):
    """
    Cartoon: the rough outline of the mouse in `cartoon_plot/mouse.png`, drawn from
    a traced path instead of a bitmap — so it scales, recolors, and can be composed
    with other panels. Nothing about it is measured; it is a background shape.

    `style` picks the look (see `draw_mouse_shape`); `margin` is the white border
    in source pixels.
    """
    fig, ax = plt.subplots(figsize=(2.6, 2.6 * _MOUSE_PNG_SIZE[1]
                                    / _MOUSE_PNG_SIZE[0]))
    xy = draw_mouse_shape(ax, style=style)
    ax.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
    ax.set_ylim(xy[:, 1].min() - margin, xy[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if style == "gradient" else f"_{style}"
    _save(fig, out_dir / f"cartoon_mouse_shape{suffix}.png",
          extra=f"  ({len(_MOUSE_OUTLINE)} traced anchors, {style})")


# ─── Measured synaptic dynamics (Allen Synaptic Physiology) ───────────────────
# Campagnola et al. 2022 (Science 375, eabj5861) Fig. 4A shows, for one
# depressing, one facilitating and one pseudo-linear excitatory connection, a
# 50 Hz 12-pulse train: over each pulse a swarm of grey dots, one per trial's
# PSP amplitude; under them, in black, that pulse's trial-mean PSP waveform
# (-1 to 8 ms after the spike). `plot_synaptic_train_amplitudes` draws the
# same elements for the paper's own three example pairs
# (biology/biology_fast_synaptic_waveforms.py), all dots in one grey (the paper also
# colors pulses 1 and 8 by a variability metric; that encoding is dropped
# here), with two departures from the paper's layout: only a subset of pulses is
# drawn (SYNAPTIC_PULSES, default 1, 4, 6, 10, 12: early, mid and late
# induction plus two recovery pulses), one slot each, and the x axis is
# labeled with each pulse's real time after the first presynaptic spike
# (SCHEME.md: trial time in ms); the uneven time gaps between slots are
# left to the tick labels, with no break glyph. For the recovery pulses that
# time assumes the protocol's 250 ms pause, while the pooled trials include
# pauses of 125 ms to 4 s (docs/biology.md). Within a slot one unit spans
# one 20 ms inter-pulse interval, and - as in the paper - each mean waveform
# is centered on its pulse's dot column rather than placed at its absolute
# latency (a PSP peaks 3-4 ms after the spike, which would otherwise shift
# every trace right of its dots). Rows are ticked at 0 and a fixed top
# (SYNAPTIC_YLIM_TOP) but their range follows the data past that top, so no
# dot is clipped.
#
# Data file, preferred: biology_fast_synaptic_trial_amplitudes.json (all 12
# pulses; the database's fitted amplitude and baseline-fit noise of every
# QC-passed response, all 50 Hz trains pooled as in the paper; per-pulse mean
# waveforms from the NWB recordings). Fallback: the older
# biology_fast_synaptic_waveforms_metadata.json, which only holds per-trial
# peaks for pulses 1, 4, 8, 9 and 12 - the figure then shows just those pulses
# with a mean dash instead of a waveform.
SYNAPTIC_AMPLITUDES_JSON = OUT_DIR / "biology_fast_synaptic_trial_amplitudes.json"
SYNAPTIC_LEGACY_JSON = OUT_DIR / "biology_fast_synaptic_waveforms_metadata.json"
# Row order of the paper figure, top to bottom.
SYNAPTIC_GROUP_ORDER = ("depressing", "facilitating", "pseudolinear")
SYNAPTIC_PULSES = (1, 4, 6, 10, 12)       # 1-based pulses drawn by default
SYNAPTIC_N_INDUCTION = 8                  # pulses before the recovery interval
SYNAPTIC_SWARM_HALF_WIDTH_MS = 7.0        # max swarm half-width around a pulse, in ms
# Tick top per row (mV): the row is ticked at 0 and this value. Its range runs
# from SYNAPTIC_YLIM_PAD of the top below zero (keeps failures scattered just
# under 0 visible) to the same pad above the top or above the highest dot or
# waveform, whichever is higher - the top is a reference, not a clip.
SYNAPTIC_YLIM_TOP = {"depressing": 1.0, "facilitating": 2.0, "pseudolinear": 5.0}
SYNAPTIC_YLIM_PAD = 0.08
_SYNAPTIC_MARKER_PT = 2.1                 # dot diameter in points (paper: 3.2, on a 6 in panel)
# Compact panel: width and per-row height in inches, plus the fixed margin that
# holds the x axis label; rows are separated by _SYNAPTIC_HSPACE of a row. The
# width sets the spacing between pulse columns (one slot each), so narrowing
# it pulls the spikes together.
_SYNAPTIC_FIGSIZE = (2.5, 0.55, 0.42)
_SYNAPTIC_HSPACE = 0.15
_SYNAPTIC_DOT_COLOR = (0.7, 0.7, 0.7)     # the paper's grey, for every dot
# The figure's key, saved as its own file (`_save_synaptic_legend`): what one
# dot is, and what the black mark under each swarm is - the trial-mean
# waveform, or the legacy file's mean-amplitude dash.
_SYNAPTIC_DOT_LABEL = "Single-trial PSP amplitude"
_SYNAPTIC_LINE_LABELS = {"waveform": "Trial-mean PSP waveform",
                         "dash": "Trial-mean PSP amplitude"}


def _parse_pulse_label(label):
    """'Pulse 9 (recovery)' -> 8 (0-based), for the legacy metadata keys."""
    match = re.match(r"Pulse (\d+)", str(label))
    if match is None:
        raise ValueError(f"unrecognized pulse label {label!r}")
    return int(match.group(1)) - 1


def _pulse_arrays(mapping, pulses):
    """{0-based pulse: float array} from a JSON {str pulse: list}, validated."""
    arrays = {int(pulse): np.asarray(values, dtype=float) for pulse, values in mapping.items()}
    if set(arrays) != set(pulses) or any(
            v.ndim != 1 or v.size == 0 or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("per-pulse arrays must cover every pulse with finite values")
    return arrays


def load_synaptic_trial_amplitudes(path=None):
    """Read the measured per-trial PSP data of the example connections.

    `path` defaults to SYNAPTIC_AMPLITUDES_JSON when it exists, else to the
    legacy waveform metadata. Returns a dict with `connections` (each: group,
    pair_external_id, classes ('L5 ET->Pv' or None), sign (+1 excitatory / -1
    inhibitory), amplitudes {0-based pulse: trial amplitudes in mV}, noise
    {pulse: trial noise amplitudes} or None, waveforms {pulse: (time_ms,
    mean_mv)} or None), `statistic` (how one trial's amplitude was measured),
    `adjusted` (whether the file carries the baseline-noise amplitudes the
    paper's adjusted CV needs; kept for completeness, the figure does not
    color by variability), `frequency_hz`, `recovery_ms` and `path`."""
    if path is None:
        path = (SYNAPTIC_AMPLITUDES_JSON if SYNAPTIC_AMPLITUDES_JSON.exists()
                else SYNAPTIC_LEGACY_JSON)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run biology/biology_fast_synaptic_waveforms.py first.")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)

    full_format = "amplitude_statistic" in data
    connections = []
    for entry in data["connections"]:
        noise = waveforms = None
        if full_format:
            pulses = [int(p) for p in entry["trial_amplitude_mv"]]
            amplitudes = _pulse_arrays(entry["trial_amplitude_mv"], pulses)
            if "trial_noise_mv" in entry:
                noise = _pulse_arrays(entry["trial_noise_mv"], pulses)
                if any(noise[p].shape != amplitudes[p].shape for p in pulses):
                    raise ValueError("noise amplitudes must match the trial amplitudes")
            if "mean_waveform_mv" in entry:
                times = _pulse_arrays(entry["waveform_time_ms"], pulses)
                means = _pulse_arrays(entry["mean_waveform_mv"], pulses)
                if any(times[p].shape != means[p].shape for p in pulses):
                    raise ValueError("waveform times and values must align")
                waveforms = {p: (times[p], means[p]) for p in pulses}
        else:
            pulses = [_parse_pulse_label(label) for label in entry["pulses"]]
            amplitudes = _pulse_arrays(
                {str(_parse_pulse_label(label)): values["trial_peak_delta_vm_mv"]
                 for label, values in entry["pulses"].items()}, pulses)
        synapse_type = str(entry.get("synapse_type", "excitatory")).lower()
        classes = (f"{entry['pre_class']}\u2192{entry['post_class']}"
                   if entry.get("pre_class") and entry.get("post_class") else None)
        connections.append({
            "group": entry["group"],
            "pair_external_id": tuple(entry.get("pair_external_id", ())),
            "classes": classes,
            "sign": -1.0 if synapse_type.startswith("in") else 1.0,
            "amplitudes": amplitudes,
            "noise": noise,
            "waveforms": waveforms,
        })
    protocol = data.get("protocol", {})
    return {
        "connections": connections,
        "statistic": (data["amplitude_statistic"] if full_format
                      else "per-trial peak within the response window (legacy metadata)"),
        "adjusted": all(conn["noise"] is not None for conn in connections),
        "frequency_hz": float(protocol.get("frequency_hz", 50.0)),
        "recovery_ms": float(protocol.get("recovery_ms", 250.0)),
        "path": path,
    }


def _beeswarm_offsets(y_px, marker_px, max_half_width_px):
    """Horizontal offsets (px) of a beeswarm: each point, in order of y, takes
    the position nearest the center at which it touches but does not overlap
    an already placed marker. When no such position fits within
    `max_half_width_px` the nearest candidate is clipped, accepting overlap."""
    y_px = np.asarray(y_px, dtype=float)
    offsets = np.zeros(y_px.size)
    placed = []
    for index in np.argsort(y_px, kind="stable"):
        y = y_px[index]
        candidates = [0.0]
        for x_other, y_other in placed:
            gap = abs(y - y_other)
            if gap < marker_px:
                reach = np.sqrt(marker_px ** 2 - gap ** 2)
                candidates.extend((x_other + reach, x_other - reach))
        chosen = None
        for candidate in sorted(candidates, key=abs):
            if abs(candidate) > max_half_width_px:
                continue
            if all((candidate - x_other) ** 2 + (y - y_other) ** 2 >= marker_px ** 2 - 1e-9
                   for x_other, y_other in placed):
                chosen = candidate
                break
        if chosen is None:
            chosen = float(np.clip(min(candidates, key=abs), -max_half_width_px, max_half_width_px))
        offsets[index] = chosen
        placed.append((chosen, y))
    return offsets


def _swarm_x(ax, x_center, values, marker_pt=_SYNAPTIC_MARKER_PT,
             half_width=SYNAPTIC_SWARM_HALF_WIDTH_MS):
    """Beeswarm x positions (data units) for `values` around `x_center`.

    Works in display pixels, so the axes limits must be final before the call;
    the figure's dpi converts the marker diameter from points."""
    values = np.asarray(values, dtype=float)
    marker_px = marker_pt * ax.figure.dpi / 72.0
    display = ax.transData.transform(np.column_stack([np.full(values.size, x_center), values]))
    center_px = ax.transData.transform((x_center, 0.0))[0]
    half_width_px = abs(ax.transData.transform((x_center + half_width, 0.0))[0] - center_px)
    offsets = _beeswarm_offsets(display[:, 1], marker_px, half_width_px)
    data = ax.transData.inverted().transform(np.column_stack([center_px + offsets, display[:, 1]]))
    return data[:, 0]


def synaptic_pulse_time_ms(pulse, frequency_hz, recovery_ms,
                           n_induction=SYNAPTIC_N_INDUCTION):
    """Time of a 0-based pulse's presynaptic spike after the first one: the
    induction pulses at the train's interval, the recovery pulses after the
    recovery interval (matching biology_fast_synaptic_dynamics._pulse_times_ms)."""
    interval = 1000.0 / float(frequency_hz)
    if pulse < n_induction:
        return pulse * interval
    return (n_induction - 1) * interval + float(recovery_ms) + (pulse - n_induction) * interval


def _synaptic_legend_figure(line_kinds=("waveform",)):
    """The PSP-train figure's key as a standalone figure, laid out like
    biology.py's connectivity legend: a grey dot for one trial's amplitude,
    then one black line per kind of trial-mean mark the rows draw
    ("waveform" or the legacy "dash", see _SYNAPTIC_LINE_LABELS)."""
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=_SYNAPTIC_DOT_COLOR,
                      markeredgecolor=_SYNAPTIC_DOT_COLOR, label=_SYNAPTIC_DOT_LABEL)]
    handles += [Line2D([], [], color="black", linewidth=1.5, label=_SYNAPTIC_LINE_LABELS[kind])
                for kind in line_kinds]
    fig = plt.figure(figsize=(2.8, 1.2))
    fig.legend(handles=handles, loc="center", ncol=1, frameon=False, fontsize=11,
               handlelength=1.6, handletextpad=0.65, labelspacing=0.5)
    return fig


def _save_synaptic_legend(out_dir, line_kinds=("waveform",)):
    """Save `_synaptic_legend_figure` beside the figure it explains, as
    biology.py saves its connectivity legend (transparent, 600 dpi), so the
    two keys match wherever they are placed. Returns the path."""
    fig = _synaptic_legend_figure(line_kinds)
    path = Path(out_dir) / "cartoon_synaptic_train_amplitudes_legend.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_synaptic_train_amplitudes(out_dir=OUT_DIR, data_path=None,
                                   pulses=SYNAPTIC_PULSES):
    """
    Measured figure (not a cartoon): the trial-by-trial PSP amplitudes and
    mean waveforms of Campagnola et al. 2022 Fig. 4A's three example pairs.

    One row per connection in the paper's order (depressing, facilitating,
    pseudo-linear), unlabeled as in the paper - the caption names them. The
    1-based `pulses` (default SYNAPTIC_PULSES = 1, 4, 6, 10, 12) each get one
    slot along x; the slot's tick is labeled with the pulse's real time after
    the first presynaptic spike in ms (20 ms per pulse at 50 Hz, plus the
    recovery interval before pulse 9); the uneven gaps between slots are left
    to those labels, with no break glyph. Inside a slot one unit is one
    inter-pulse interval; each pulse's trial-mean PSP waveform (-1 to 8 ms
    after its spike) is drawn in black with its time window centered on the
    slot, as the paper does, under a beeswarm of grey dots, one per trial's
    amplitude, centered on the slot too - so trace and dots line up and the
    waveform's absolute latency is not represented. All dots share one grey;
    the paper's variability coloring of pulses 1 and 8 is not reproduced. Rows
    are ticked at 0 and the fixed tops of SYNAPTIC_YLIM_TOP (1, 2 and 5 mV);
    each row's range extends past its top to the highest dot or waveform, so
    nothing is clipped. The key (grey dot = one trial's amplitude, black line
    = trial-mean waveform) is saved separately as
    cartoon_synaptic_train_amplitudes_legend.png.

    Legacy metadata (five pulses, no waveforms) draws a mean dash per pulse;
    the saved message and the key say so. Reads
    `load_synaptic_trial_amplitudes(data_path)`.
    """
    requested = sorted({int(pulse) - 1 for pulse in pulses})
    if not requested or requested[0] < 0 or requested[-1] > 11:
        raise ValueError(f"pulses must be 1-based pulse numbers within 1-12; got {list(pulses)}")
    data = load_synaptic_trial_amplitudes(data_path)
    by_group = {conn["group"]: conn for conn in data["connections"]}
    groups = [group for group in SYNAPTIC_GROUP_ORDER if group in by_group]
    if not groups:
        raise ValueError(f"no known STP group among {sorted(by_group)}")
    frequency_hz, recovery_ms = data["frequency_hz"], data["recovery_ms"]
    interval = 1000.0 / frequency_hz
    available = {pulse for group in groups for pulse in by_group[group]["amplitudes"]}
    shown = [pulse for pulse in requested if pulse in available]
    if not shown:
        raise ValueError(f"none of pulses {[p + 1 for p in requested]} in {data['path']}")
    # One slot per shown pulse; its tick names the pulse's real time.
    slot = {pulse: float(index) for index, pulse in enumerate(shown)}
    times = {pulse: synaptic_pulse_time_ms(pulse, frequency_hz, recovery_ms) for pulse in shown}
    ms_to_slot = 1.0 / interval
    n_rows = len(groups)

    width, row_height, margin = _SYNAPTIC_FIGSIZE
    height = row_height * n_rows + margin
    fig, axes = plt.subplots(n_rows, 1, figsize=(width, height), sharex=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.98, bottom=margin / height,
                        hspace=_SYNAPTIC_HSPACE)
    line_kinds = []                  # the trial-mean marks drawn, for the key
    for ax, group in zip(axes, groups):
        conn = by_group[group]
        sign = conn["sign"]
        pulses = [pulse for pulse in shown if pulse in conn["amplitudes"]]
        amplitudes = {p: sign * conn["amplitudes"][p] for p in pulses}
        waveforms = conn["waveforms"]
        kind = "waveform" if waveforms is not None else "dash"
        if kind not in line_kinds:
            line_kinds.append(kind)
        # Limits first: the swarm is laid out in display space. The range
        # follows the data, so no dot or waveform is clipped.
        ax.set_xlim(-0.5, len(shown) - 0.5)
        high = max(float(v.max()) for v in amplitudes.values())
        if waveforms is not None:
            high = max(high, max(float(np.abs(waveforms[p][1]).max()) for p in pulses))
        top = SYNAPTIC_YLIM_TOP.get(group)
        if top is None:
            low = min(float(v.min()) for v in amplitudes.values())
            pad = SYNAPTIC_YLIM_PAD * max(high - min(low, 0.0), 1e-6)
            ax.set_ylim(min(low, 0.0) - pad, high + pad)
        else:
            # Ticks at 0 and the row's fixed top; the top bounds the range only
            # when nothing reaches above it.
            pad = SYNAPTIC_YLIM_PAD * top
            ax.set_ylim(-pad, max(top, high) + pad)
            ax.set_yticks([0.0, top])
        for pulse in pulses:
            values = amplitudes[pulse]
            ax.scatter(_swarm_x(ax, slot[pulse], values,
                                half_width=SYNAPTIC_SWARM_HALF_WIDTH_MS * ms_to_slot),
                       values, s=_SYNAPTIC_MARKER_PT ** 2, color=_SYNAPTIC_DOT_COLOR,
                       edgecolors=_SYNAPTIC_DOT_COLOR, linewidths=0.3, zorder=3)
            if waveforms is not None:
                time_ms, mean_mv = waveforms[pulse]
                # Paper-style alignment: the waveform's time window is centered on
                # the dot column instead of starting at the spike time.
                centered_ms = time_ms - 0.5 * (float(time_ms[0]) + float(time_ms[-1]))
                ax.plot(slot[pulse] + centered_ms * ms_to_slot, np.abs(mean_mv), color="black",
                        lw=1.1, zorder=100, solid_capstyle="round")
            else:
                # No waveform in the legacy file: a dash at the trial mean.
                ax.plot([slot[pulse] - 0.3, slot[pulse] + 0.3], [values.mean()] * 2,
                        color="black", lw=1.1, zorder=100)
        ax.tick_params(labelsize=6, length=2, pad=1.5)
        if top is None:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=False))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.6)

    last = axes[-1]
    last.set_xticks([slot[pulse] for pulse in shown])
    last.set_xticklabels([f"{times[pulse]:g}" for pulse in shown])
    last.set_xlabel("Time after first presynaptic spike (ms)", fontsize=7, labelpad=2)
    fig.supylabel("PSP amplitude (mV)", fontsize=7, x=0.01)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ", ".join(f"{group}: {by_group[group]['classes'] or '?'}" for group in groups)
    pulse_text = ", ".join(f"{p + 1} ({times[p]:g} ms)" for p in shown)
    detail = (f"pulses {pulse_text} at {frequency_hz:g} Hz; rows top to bottom {rows}; "
              + f"amplitude = {data['statistic']}; "
              + ("mean waveforms" if data["connections"][0]["waveforms"] else "mean dashes (legacy)"))
    _save(fig, out_dir / "cartoon_synaptic_train_amplitudes.png",
          extra=f"  (MEASURED: {data['path'].name}; {detail})")
    _save_synaptic_legend(out_dir, line_kinds)


# ─── Lesion schematic: one synapse with its modulation frozen ────────────────
# A SYNAPSE cluster lesion in lesion.py acts on a set of entries of the plastic
# recurrent weight matrix mp_layer1.W. It comes in two modes: zero_W removes
# the synapses (W <- 0), freeze_M keeps W and holds the modulation at its
# initial value, so the effective weight stays at W instead of following
# M(t). This cartoon draws freeze_M only: one synapse's effective weight over
# a trial, intact and frozen. Both modes are scored against a size-matched
# random lesion; that comparison is left to the caption. Everything is
# invented and fixed, so the figure explains the procedure and must not be
# captioned as data.
_LESION_TRACE_COLOR = c_vals[1]             # the frozen synapse (categorical)
_LESION_W = 0.8                             # static weight of the traced synapse (a.u.)
# The y axis spans only the traces, not down to zero: W_eff stays within a
# few percent below W and rises above it, so a zero baseline would leave most
# of the panel empty.
_LESION_Y_MARGIN = 0.12                     # padding above and below, as a fraction of the span
# Text follows paper_plot: 7 pt ticks (its rcParams), 8 pt axis and panel
# titles (its lesion figures); only the line-side notes are smaller (6.5 pt).
_LESION_TICK_FONTSIZE = 7
_LESION_LABEL_FONTSIZE = 8
_LESION_NOTE_FONTSIZE = 6.5
_LESION_FIGSIZE = (2.8, 1.8)


def _modulation_trace(rng, t_ms, n_samples=N_SAMPLES):
    """A synthetic M(t) over one trial: zero at trial start (M_init), rising
    through the stimulus and memory periods with a Hebbian-looking wobble."""
    rise = 0.55 * _sig(t_ms, 900.0, 220.0) * (1.0 + 0.3 * np.sin(2 * np.pi * t_ms / 1100.0))
    return rise + _smooth_noise(rng, n_samples, 0.04, width=0.03 * n_samples)


def _freeze_m_traces(ax, t_ms, w, m, color):
    """W_eff(t) = W (1 + M(t)) of one lesioned synapse, intact and with its
    modulation frozen, labelled at the right end: intact (grey dashed) and
    M(t) <- M(0) (freeze_M, flat at W). The y axis covers the two traces
    with W as its only tick. The trial end is marked T rather than in ms:
    the trace is a cartoon, not a recorded trial. Returns the two traces."""
    intact = w * (1.0 + m)
    frozen = np.full_like(intact, w)
    ax.plot(t_ms, intact, color="0.55", lw=0.9, linestyle="--", zorder=2)
    ax.plot(t_ms, frozen, color=color, lw=1.4, zorder=3)
    x_label = t_ms[-1] + 0.03 * t_ms[-1]
    ax.text(x_label, intact[-1], "intact", color="0.45", fontsize=_LESION_NOTE_FONTSIZE,
            va="center", ha="left")
    ax.text(x_label, w, r"$M(t) \leftarrow M(0)$", color=color, fontsize=_LESION_NOTE_FONTSIZE,
            va="center", ha="left")
    low, high = min(intact.min(), w), max(intact.max(), w)
    pad = _LESION_Y_MARGIN * (high - low)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_ylim(low - pad, high + pad)
    ax.set_yticks([w])
    ax.set_yticklabels(["W"])
    ax.set_xticks([t_ms[0], t_ms[-1]])
    ax.set_xticklabels(["0", "T"])
    ax.set_xlabel("Trial time", fontsize=_LESION_LABEL_FONTSIZE, labelpad=1)
    ax.set_ylabel(r"$W_{\mathrm{eff}}$", fontsize=_LESION_LABEL_FONTSIZE, labelpad=1)
    ax.tick_params(labelsize=_LESION_TICK_FONTSIZE, length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    return intact, frozen


def plot_synapse_lesion_schematic(out_dir=OUT_DIR, seed=0):
    """
    Cartoon: what the freeze_M synapse lesion does to one synapse, which no
    wiring diagram can show. Its effective weight over a trial, intact
    (dashed, following M(t)) and with the modulation frozen (held at W).
    The size-matched random control that scores every lesion belongs in the
    caption.

    All values are invented and fixed by `seed`; this is the procedure of
    lesion.py, not data.
    """
    rng = np.random.RandomState(seed)
    t_ms = np.linspace(0.0, (T_END - 1) * DT, N_SAMPLES)
    m_trace = _modulation_trace(rng, t_ms)

    fig, ax = plt.subplots(figsize=_LESION_FIGSIZE)
    fig.subplots_adjust(left=0.13, right=0.67, top=0.87, bottom=0.21)
    _freeze_m_traces(ax, t_ms, _LESION_W, m_trace, _LESION_TRACE_COLOR)
    ax.set_title("Synapse lesion: frozen M", fontsize=_LESION_LABEL_FONTSIZE, pad=3)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "cartoon_synapse_lesion_schematic.png",
          extra="  (SYNTHETIC: one synapse, freeze_M)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                       help="Directory to write the cartoons into "
                            f"(default: {OUT_DIR}).")
    parser.add_argument("--figure",
                       choices=("traces", "fastslow", "mouse", "synaptic", "lesion", "all"),
                       default="all",
                       help="Which cartoon(s) to draw (default: all). 'synaptic' "
                            "is the measured Allen PSP-train figure; 'lesion' the "
                            "frozen-modulation synapse lesion trace.")
    parser.add_argument("--synaptic-data", default=None,
                       help="JSON of per-trial PSP amplitudes for --figure synaptic "
                            "(default: cartoon_plot/biology_fast_synaptic_trial_"
                            "amplitudes.json, else the legacy waveform metadata).")
    parser.add_argument("--synaptic-pulses", type=int, nargs="+", default=list(SYNAPTIC_PULSES),
                       metavar="PULSE",
                       help="Which 1-based pulses of the 12-pulse train --figure "
                            f"synaptic draws (default {' '.join(map(str, SYNAPTIC_PULSES))}; "
                            "e.g. 1 2 3 4 5 6 7 8 9 10 11 12 for all).")
    parser.add_argument("--mouse-style",
                       choices=("gradient", "flat", "outline", "all"),
                       default="gradient",
                       help="Mouse silhouette look: 'gradient' (default, as in "
                            "mouse.png), 'flat' or 'outline'.")
    parser.add_argument("--seed", type=int, default=0,
                       help="RNG seed for the synthetic wiggle (default 0).")
    parser.add_argument("--n-units", type=int, default=5,
                       help="How many units to draw (default 5; max 10 for the "
                            "annotated style, unlimited for raw).")
    parser.add_argument("--style", choices=("raw", "annotated", "both"),
                       default="raw",
                       help="'raw' (default): thin black traces, no axes — the "
                            "schematic glyph. 'annotated': colored, with the "
                            "period strip, a time axis and motif labels.")
    parser.add_argument("--no-labels", dest="show_labels", action="store_false",
                       help="Annotated style only: drop the per-trace motif names "
                            "(saves without the '_n' suffix, matching "
                            "paper_plot's convention).")
    parser.set_defaults(show_labels=True)
    args = parser.parse_args()
    if args.figure in ("traces", "all"):
        styles = (("raw", "annotated") if args.style == "both" else (args.style,))
        for style in styles:
            plot_rich_dynamics(out_dir=args.out_dir, seed=args.seed, style=style,
                               n_units=args.n_units, show_labels=args.show_labels)
    if args.figure in ("fastslow", "all"):
        plot_fast_slow_curves(out_dir=args.out_dir, seed=args.seed)
    if args.figure in ("mouse", "all"):
        mouse_styles = (("gradient", "flat", "outline")
                        if args.mouse_style == "all" else (args.mouse_style,))
        for style in mouse_styles:
            plot_mouse_shape(out_dir=args.out_dir, style=style)
    if args.figure in ("lesion", "all"):
        plot_synapse_lesion_schematic(out_dir=args.out_dir, seed=args.seed)
    if args.figure in ("synaptic", "all"):
        try:
            plot_synaptic_train_amplitudes(out_dir=args.out_dir,
                                           data_path=args.synaptic_data,
                                           pulses=args.synaptic_pulses)
        except FileNotFoundError as error:
            # The measured data are optional for an "all" run of the cartoons.
            print(f"  Skipped synaptic train figure: {error}")


if __name__ == "__main__":
    main()
