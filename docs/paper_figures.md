# Paper figure panels and their sources

Which `paper_plot.py` figure (and which analysis cache) each panel of the
composed paper figures comes from, with the details the captions leave out.
Figure 1 is covered in [docs/biology.md](biology.md).

## Figure 2: the single-task MPN on MemoryPro

Run: `ONETASK_ANAME = delaygo_seed395_hidden200+batch128+angle` (200 hidden
units, λ = 0.9 from `m_time_scale = 400` ms, L2 = 1e-4), trained on
`delaygo` (MemoryPro) alone by `one_task/one_task.py`; every cache below is
written by `one_task/one_task_analysis.py` into `onetask/<run>/` and drawn
by `python figures/paper_plot.py one_task`.

| Panel | Figure(s) | Cache | Notes |
|---|---|---|---|
| a | `onetask_example_trial` (`_input.png`, `_output.png`) | example-trial pickle | Inputs: fixation, modality 1 cos/sin, modality 2 cos/sin, rule cues; outputs: fixation and response cos θ / sin θ with their targets. The network drawing between them is hand-drawn. |
| b | `onetask_modulation_snapshot_single` (`onetask_emodulation_snapshot_single.png`, `onetask_hidden_snapshot_single.png`, each with a `_cbar.png`) | `modulation_snapshot_<run>.pkl` | Stimulus 1 at the middle of the response period (`t_mid_resp`): the effective modulation W ⊙ M (hidden × input) and the hidden-state vector, each on a symmetric scale clipped at the 99th percentile of its own values (W ⊙ M: ±0.091; hidden: ±0.77). The raw M of the same step saturates at the bound ±1, so a ±0.77 bar belongs to the hidden strip, not to the matrix. |
| c | `onetask_d_combine` (left) and `onetask_pc_cumvar` (right) | `d_combine_<run>.pkl` (`cross_period_dimensionality`) | Left: for each pair of periods, the fraction of the "Test" period's variance captured by the top four PCs fitted on the "Fit" period (no centering), for hidden activity and W ⊙ M. Right: each period's own cumulative variance curve over 1–11 PCs, period colors as in the trial bar. |
| e–h | `onetask_grad_fixed_points_3d` (`_hidden.png`, top row; `_emodulation.png`, bottom row) | `fixed_points_grad_<run>.pkl` (+ `.pca.npz` sidecar) | Fixed points M* = F(M*; x) solved by gradient descent (`core/grad_fixed_points.py`) under each period's constant input for 64 stimulus angles, drawn as hidden(M*) and W ⊙ M*. x–y: the two-PC basis fitted on the memory-period ("longdelay") fixed points; z: the target cos-output of the point's stimulus (0 outside Response). Points colored by stimulus; grey points carry no stimulus identity: the Context panel's own solve (one point, all seeds alike) plus its memory-seeded ring and naive-seeded probes, the Memory panel's fixation-trajectory and naive probes (`_GRAD_FP_3D_OVERLAY_PROBES`). Black path: the recorded within-period trajectory of stimulus 0, anchored at the previous period's fixed point (×) and ending at this period's (arrowhead). Dashed ring: strict fixed points joined in stimulus order. Every candidate is drawn solid (`ONETASK_GRAD_FP_3D_SOLID_CANDIDATES = True`, as in the two-task 3D figures); convergence is reported by `onetask_interp_fixed_points` (residual per angle) and `onetask_fixed_point_classification`. In this run all context, memory and response candidates and the memory-seeded ring are strict fixed points (leak-normalized residual ≤ 0.01), 20 of the 64 stimulus-period candidates are approximate (0.011–0.043) and the random-seeded probes do not converge (0.11). The hidden row's x–y content is rotated 90° (`_rotate_hidden_3d_content`). |
| color wheel | `onetask_stimulus_colorwheel` | none | The stimulus-direction colors (`stim_color`). |
| i | `onetask_show` | `show_<run>.pkl` (`plot_trajectory_by_index`) | Stimuli 5 and 2. Each trace is the cos-output readout row applied to the modulated layer's linear response to one input component, `y_cos · (W + W ⊙ M(t)) · W_input x_component(t)`: the fixation input (grey), the rule input plus the layer bias (orange) and their sum (blue). The drawing of the two input pathways is hand-drawn. |

Composition notes: the composed figure has no panel d. Its fixed-point
panels were rendered before `_period_display` mapped the pickles' "Delay"
period to "Memory"; the current figures title that period "Memory" and label
the basis "Memory PC", matching the trial bars and
[SCHEME.md](../SCHEME.md).
