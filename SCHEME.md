# Project scheme

Shared conventions for the whole project: the simulation timing (below) and the
figure color families ([Figure color scheme](#figure-color-scheme)).

## Timing — Δt = 40 ms

The simulation time step is **Δt = 40 ms** (`task_params['dt'] = 40`, in
milliseconds). One network/recorded time step therefore corresponds to 40 ms of
simulated time. Task period lengths in [`core/mpn_tasks.py`](core/mpn_tasks.py)
are written in ms and divided by `dt` to get integer step counts (e.g.
`int(700/dt)`), and the leak `alpha = 0.2` implies a time constant τ = Δt/alpha
= 200 ms.

**Rule:** whenever a figure or readout refers to trial time, express it in the
actual unit (ms), not in raw time-step index — multiply the step index by `dt`.
Read `dt` from the run's saved params (`cfg["task_params"]["dt"]`) rather than
hard-coding it, so the conversion follows the run.

## Figure color scheme

Color conventions used by [`paper_plot.py`](paper_plot.py) (and mirrored in the
per-experiment analysis scripts). The goal is that each *meaning* has one color
family, and the families are visually distinct so a reader never confuses, say,
a stimulus color with a trial-period color.

Every color below is defined once at the top of `paper_plot.py`; change it there
and all figures follow.

## The four color families

| Family | What it encodes | Where it lives | Kept distinct from |
|---|---|---|---|
| **Stimulus** | ring stimulus direction (0–7) | `stim_color(k, n)` | period, input/output |
| **Period** | trial epoch (fixation/stimulus/memory/response) | `_ONETASK_PERIOD_COLORS`, `_PHASE_COLORS` | stimulus, input/output |
| **Input / output** | example-trial input & readout channels | `_IO_*` constants | stimulus, period |
| **Categorical** | everything else (components, series, tasks) | `c_vals` | — |

---

## 1. Stimulus color — red→purple rainbow

`stim_color(k, n)` maps stimulus index `k` of `n` to a hue sweeping from red
(hue 0) through the spectrum to purple (hue 0.83), so adjacent stimuli are
adjacent colors and a stimulus ring reads as a smooth gradient.

- Used **only** to color by stimulus direction: trajectory/ring/fixed-point
  figures, and the `onetask_stimulus_colorwheel` legend.
- `ONETASK_N_STIM = 8` is the default stimulus count.
- Example ramp (n=8): `#e62222` (red) → `#e6ad22` → `#93e622` → `#22e63d` →
  `#22e6c7` → `#2279e6` → `#5722e6` → `#e222e6` (violet).

## 2. Period (trial-epoch) colors

The period bar in the one-task / two-task figures reuses the **multi-task heatmap
phase colors** (`_PHASE_COLORS`) so it is color-consistent with the
input/hidden/modulation heatmaps.

`_ONETASK_PERIOD_COLORS` (ordered Fixation → Stimulus → Memory → Response):

| Period | Hex | Source |
|---|---|---|
| Fixation | `#fef08a` (pale yellow) | new — no heatmap counterpart |
| Stimulus | `#c3b1e1` (purple) | `_PHASE_COLORS["stim1"]` (Stimulus 1) |
| Memory | `#bbf7d0` (light green) | `_PHASE_COLORS["delay1"]` (Memory 1) |
| Response | `#d1d5db` (light gray) | `_PHASE_COLORS["go1"]` (Response) |

Full heatmap phase palette (`_PHASE_COLORS`): `stim1 #c3b1e1`, `stim2 #bfdbfe`,
`delay1 #bbf7d0`, `delay2 #fed7aa`, `go1 #d1d5db`.

## 3. Input / output channel colors

Colors for the `onetask_example_trial` input and output traces. A muted
qualitative set, distinct from the stimulus rainbow and the pale period pastels.
**Rules:** within a modality the cos/sin channels share a hue as a
`(dark, light)` pair; channels that mean the same thing across the input and
output figures share a color. The two stimulus modalities share **one green
cos/sin pair** (`_IO_MOD1` is an alias of `_IO_MOD2`), so cos↔cos and sin↔sin
match across the modalities — they are the same physical channel, differing only
in modality.

| Constant | Hex | Meaning |
|---|---|---|
| `_IO_FIXATION` | `#555555` (dark gray) | Fixation (input **and** output) |
| `_IO_MOD2` | `("#1b9e77", "#6fceae")` (green dark/light) | Stimulus cos / sin |
| `_IO_MOD1` | = `_IO_MOD2` (alias) | Modality 1 cos / sin — shares Modality 2's colors |
| `_IO_TASK` | `#d95f02` (orange) | Task cue (active) |
| `_IO_TASK2` | `#fdae6b` (light orange) | Second (inactive) task cue placeholder |
| `_IO_RESPONSE` | `("#7e3ff2", "#c4a3f5")` (purple dark/light) | Output (response) cos / sin |

Input↔output matching by meaning:
- **Fixation** (input) ↔ **Fixation** (output) → `_IO_FIXATION`.
- **Output Cos / Sin** get their **own purple hue** (`_IO_RESPONSE`), deliberately
  distinct from the stimulus modalities so the readout is not confused with an
  input modality.
- The faded target "shadow" in the output figure is each channel's own color
  lightened toward white (`_lighten`).

## 4. Categorical cycle — `c_vals`

The original qualitative palette. Used for **non-stimulus** categorical coloring
only: components (`|Fix−Task|`/`|Fix|`/`|Task|`), series/periods in CVE and
correlation curves, task columns, etc. Not used for stimulus (see family 1).

---

## When adding a figure

- Coloring **by stimulus direction** → `stim_color(k, n)` (never `c_vals`).
- A **trial-period bar/shading** → `_ONETASK_PERIOD_COLORS` / `_PHASE_COLORS`.
- **Example-trial input/output channels** → the `_IO_*` constants, matching
  input↔output by meaning.
- Any **other categorical** distinction → `c_vals`.
- Pass a single color to `scatter` via `color=` (not `c=`) — an RGB tuple in `c=`
  is treated as value-mappable and can mis-color 3-point scatters.
