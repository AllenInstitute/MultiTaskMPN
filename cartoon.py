#!/usr/bin/env python
# coding: utf-8
"""
Rhetorical cartoons for the paper — figures that make a point rather than report a
measurement.

Nothing here reads a run. Every trace is SYNTHETIC and deterministic (fixed RNG
seed), so these figures are reproducible but must never be captioned as data. Keep
that separation strict: `paper_plot.py` shows what the network did, `cartoon.py`
shows what we are claiming about it.

Three cartoons so far: `plot_rich_dynamics` (heterogeneous single-unit responses in
one trial), `plot_fast_slow_curves` (two families of curves, fast and slow, the
ingredients of a richer dynamics) and `plot_mouse_shape` (the mouse background silhouette,
traced from `cartoon_plot/mouse.png` so it can be redrawn, recolored and composed
with other panels instead of pasted as a bitmap).

Style is imported from `paper_plot` rather than copied — rcParams, the categorical
palette (`c_vals`), the trial-period colors, the period strip and `_save_fig` — so
a cartoon dropped beside a real figure reads as the same document. See SCHEME.md
for the color families; the ten units here are a plain categorical distinction, so
they use `c_vals` and never the stimulus rainbow.

Usage:
    python cartoon.py                  # writes into cartoon_plot/
    python cartoon.py --out-dir /tmp   # somewhere else
    python cartoon.py --figure mouse   # just the mouse silhouette
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon

# Style, palette and savers come from the real figure module so the two agree by
# construction. Importing it also applies its global rcParams.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                       help="Directory to write the cartoons into "
                            f"(default: {OUT_DIR}).")
    parser.add_argument("--figure",
                       choices=("traces", "fastslow", "mouse", "all"),
                       default="all",
                       help="Which cartoon(s) to draw (default: all).")
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


if __name__ == "__main__":
    main()
