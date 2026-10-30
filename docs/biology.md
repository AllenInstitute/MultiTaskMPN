# Biological reference panels

The introductory figure of the paper (Figure 1) mixes two panels drawn from
measured data by the code in this repository with hand-drawn schematics. This
file records where the data panels come from and the details their caption
leaves out, so the caption can stay an illustration.

| Panel | Content | Origin |
|---|---|---|
| a | Recurrent connectivity among cortical neurons | `python biology/biology.py` → `cartoon_plot/biology_connectivity_network.png` (+ `_legend.png`) |
| b | Neural dynamics shared across tasks (schematic) | hand-drawn |
| c | Fast synaptic dynamics | `python figures/cartoon.py --figure synaptic` → `cartoon_plot/cartoon_synaptic_train_amplitudes.png` (+ `_legend.png`) |
| d | Recurrent neural network (schematic) | hand-drawn |
| e | Dynamical motifs shared across tasks (schematic) | hand-drawn |
| f | Multi-plasticity network (schematic) | hand-drawn |

Both data panels are regenerated on the compute node with the `mpn`
environment; the data files they read sit beside the scripts (`biology/`,
ignored by git) and in the shared project cache named in
`biology/biology_fast_synaptic_dynamics.py`.

## Panel a: MICrONS basket-cell subnetwork

`biology/biology.py` draws a directed subnetwork of the MICrONS mouse
visual-cortex connectome (`minnie65`; cell annotations
`microns_cell_annos_CV_250904.feather`, synapse table
`synapses_minnie65_phase3_v1_1507_combined_filtered_incl_trafo_250904.feather`,
read by `biology/connectivity_helper.py`).

- **Cells.** Only basket cells (`cell_type == "BC"`) whose axon is proofread
  (`status_axon == "extended"`) and whose dendrite is complete
  (`full_dendrite`), the `"both"` proofreading mode: with any other mode an
  absent connection cannot be told from an untraced one.
- **Displayed core.** 48 of those cells (`DEFAULT_N_NODES`; the rendered
  figure holds 48 nodes), chosen deterministically: the cell with the largest
  directed degree seeds the set, and the cell with the most connections into
  the set is added until the size is reached (`_connectivity_core_indices`).
  Every node and edge shown is a measured connection; nothing is sampled or
  simulated.
- **Encoding.** Arrows point presynaptic → postsynaptic; connections need at
  least one synapse (`DEFAULT_MIN_SYNAPSES`). Node area scales linearly with
  a cell's incoming plus outgoing connections within the displayed core.
  Edge width and opacity grow with the logarithm of the number of synapses
  supporting the connection. The layout is a seeded spring layout
  (`DEFAULT_LAYOUT_SEED = 7`), so the figure is reproducible.
- **Citation.** MICrONS Consortium, *Nature* (2025), functional connectomics
  of mouse visual cortex.

## Panel c: measured short-term synaptic dynamics

`figures/cartoon.py --figure synaptic` redraws Campagnola et al. 2022
(*Science* 375, eabj5861) Fig. 4A for the same three connections, from the
data file `cartoon_plot/biology_fast_synaptic_trial_amplitudes.json` that
`biology/biology_fast_synaptic_waveforms.py` extracts from the Allen
Institute Synaptic Physiology medium database (fitted amplitudes) and raw
NWB recordings (waveforms).

- **Connections.** All three are excitatory mouse cortical connections
  recorded by multipatch electrophysiology: depressing L5 ET (Sim1) → Pv,
  facilitating L5 IT (Tlx3) → Sst, pseudo-linear L6 Pyr (Ntsr1) → Pv. The
  script leaves the rows unlabeled; the labels and the key
  (`cartoon_synaptic_train_amplitudes_legend.png`) are added when the figure
  is composed.
- **Protocol.** A 50 Hz train of eight presynaptic spikes followed, after a
  pause, by four recovery spikes at 50 Hz. The panel shows pulses 1, 4, 6,
  10 and 12 of the twelve (`SYNAPTIC_PULSES`), one slot each, ticked with
  the pulse's time after the first spike: 0, 60, 100, 410 and 450 ms.
- **Recovery pulses.** As in the original figure, every 50 Hz train of a
  connection is pooled whatever its recovery pause (pauses of 125 ms to 4 s,
  three to five trials each), so the dots at pulses 10 and 12 mix pauses,
  while their tick times assume the protocol's 250 ms pause
  (`synaptic_pulse_time_ms`). The caption therefore says "after a pause"
  without a number.
- **Dots and traces.** Grey dots are the deconvolution-fit amplitude
  (`dec_fit_reconv_amp`) of every QC-passed response, 18 to 25 trials per
  pulse. The black trace is the trial-mean PSP waveform from −1 to 8 ms
  after the presynaptic spike (aligned to the spike's maximum-slope time),
  over the trials whose pulse evoked exactly one well-timed spike. Each
  waveform is centered on its pulse's dot column, as in the paper, so the
  PSP's absolute latency is not represented. Rows are ticked at 0 and at 1,
  2 and 5 mV and extend past that top when a dot or waveform does. The
  paper's coloring of pulses 1 and 8 by variability is not reproduced.
- **Citation.** Campagnola et al., *Science* 375, eabj5861 (2022); Allen
  Institute Synaptic Physiology dataset.

## Panels b, d, e, f: schematics

These panels are drawn by hand and carry no data. Two points the caption
asserts about them follow from the code and the literature:

- The MPN of panel f is **feedforward**: `DeepMultiPlasticNet`
  (`core/mpn.py`) projects the input (`W_initial_linear`), passes it through
  the plastic layer `W ⊙ (1 + M)` and a nonlinearity, and reads out
  (`W_output`). The hidden state carries no memory of its own; the only
  memory is the plastic state **M**, which the Hebbian rule updates every
  40 ms step within a trial and which decays with the time constant in
  [SCHEME.md](../SCHEME.md). The optional hidden-to-hidden term
  (`recurrent_layer_add`) is off in every saved cohort. "Slow" synaptic
  dynamics are the learned weights **W**, "fast" ones **M**.
- The RNN results of panel e are Yang et al., *Nature Neuroscience* (2019;
  bioRxiv 2017) and Driscoll, Shenoy & Sussillo, *Nature Neuroscience*
  (2024), the "dynamical motifs" the panel names.
