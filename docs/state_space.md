# State-space analyses

Context-end task geometry of the multi-task runs (`state_space_shift.py`,
`state_space_structure.py`), the sibling-task fixed-point geometry
(`sibling_delay_analysis.py`) and the `state_space` figures of `paper_plot.py`.

```bash
# optional sibling-task geometry only (does not rerun clustering or lesions)
python multiple_task/sibling_delay_analysis.py \
  --seed 921 --feature L21e4 --families delaydm1 --method gradient

# alternatively, run long-delay trials and keep each on its delay input until
# it settles to a fixed point (the gradient solver's residual criterion)
python multiple_task/sibling_delay_analysis.py \
  --seed 921 --feature L21e4 --families delaydm1 \
  --method long_delay_endpoint

# both methods end with a rule-cue interpolation sweep between the two sibling
# rules (11 alphas, fixed by SIBLING_CUE_INTERP_N_ALPHA)
python multiple_task/sibling_delay_analysis.py \
  --seed 921 --feature L21e4 --families delaydm1 dmcgo \
  --method long_delay_endpoint
python figures/paper_plot.py --only sibling_cue_interp

# additional state-space analysis
python multiple_task/state_space_shift.py
# category structure and task-vector additivity of the saved task centroids
python multiple_task/state_space_structure.py
# context-end task distance vs lesion-profile similarity, for runs that also
# have lesion_plot.py caches
python multiple_task/state_space_vs_lesion.py
python figures/paper_plot.py --only state_space_vs_lesion
python figures/paper_plot.py --only state_space_vs_lesion_seeds
# rank state-space seeds and plot the best example without clearing other figures
python figures/paper_plot.py --only state_space_combined
# untrained controls (fresh weights, the selected L2 = 1e-4 example run's
# config; seed 202 on the 2026-10-23 caches, printed by paper_plot when the
# control is missing), one per seed, then rerun the structure analysis so
# their category tests are included
python multiple_task/state_space_shift.py --untrained \
  --checkpoint multiple_tasks/savednet_everything_seed202_L21e4+hidden300+batch128+angle.pt \
  --init-seed 0 1 2 3 4 5 6 7
python multiple_task/state_space_structure.py
python figures/paper_plot.py --only state_space_untrained
python figures/paper_plot.py --only state_space_category_vs_untrained
python figures/paper_plot.py --only state_space_rdm
python figures/paper_plot.py --only state_space_rdm_example
python figures/paper_plot.py --only state_space_category_test

```

The state-space workflow is `state_space_shift.py` -> `state_space_structure.py`
-> `paper_plot.py`, with `state_space_vs_lesion.py` after
`state_space_structure.py` for the runs that also have lesion caches (see
[docs/multitask_analysis.md](multitask_analysis.md)).
The analysis saves original-feature task centroids, trial counts and display
PCA together in `state_space/state_space_pca_<aname>_noise0.01.pkl`, alongside
the existing distance-angle results. Display PCA is fitted in batches; the
analysis otherwise retains its full-trajectory workflow and memory requirements.
The batch analysis clears top-level files in `state_space/` before regenerating
the four standard L2 cohorts, so `state_space_structure.py` has to be rerun
after it; it rewrites the distance-angle results pickle after every run, so an
interrupted batch keeps what it finished.
`python multiple_task/state_space_shift.py --checkpoint multiple_tasks/savednet_<aname>.pt --out-dir DIR`
analyzes only the named checkpoints and merges their entries into the results
pickle of `DIR` (default `state_space/`) without clearing anything, which is
how one run is tested in a scratch directory.

`python multiple_task/state_space_structure.py` reads every PCA cache and
tests, for the hidden state, M and W ⊙ M of each run, two properties of the
full-dimensional task centroids (nothing is recomputed from network activity,
so it runs in seconds). (1) Category structure: the separation score S defined below
against a null that shuffles the six category labels over the 15 tasks
(10,000 permutations, seed 0, one-sided p), plus the 15 × 15 centroid distance
matrix and its average-linkage ordering. (2) Task-vector additivity: for the
task pairs differing in one factor (`core/state_space_geometry.py`
`TASK_FACTOR_PAIRS`: pro to anti, delay to memory, delay to reaction,
modality 1 to 2, context cue, match to category) the mean pairwise cosine of
the centroid-difference vectors and the share of their variance along one
direction, against a null that permutes which centroid belongs to which task
(one-sided p per factor and pooled over factors). Per L2 cohort it also
averages the runs' distance matrices after dividing each by its mean
task-pair distance, and gives the mean matrix its own tree and category test
plus the seed-to-seed Spearman correlation of the per-run matrices. Per-run
results with the null arrays go to
`state_space/state_space_structure_<aname>_noise0.01.pkl`; every run's
statistics and the cohort means to `state_space/state_space_structure_results.pkl`.
`paper_plot.py` draws them in `state_space` mode: `state_space_rdm` (the
lower triangle of the L2 = `STATE_SPACE_EXAMPLE_L2` cohort's mean W ⊙ M
matrix, ordered by its tree drawn above it, task labels on their category
color, run count, S, p and the seed-to-seed consistency in the legend, and
the "distance / mean distance" color scale as a separate
`multitask_state_space_rdm_colorbar_eff_mod` figure in the style of
`heatmap_colorbar`), `state_space_rdm_example` (the same drawing of the
selected single run's W ⊙ M matrix in its own distance units,
`multitask_state_space_rdm_example_eff_mod` with its own colorbar figure),
and `state_space_category_test` (every run's S per L2 cohort for the hidden
state and W ⊙ M, filled when p < 0.05, the range of the runs' null 95th
percentiles as a gray band, the cohort median as a dash). The task-vector
statistics are computed and cached but have no paper figure; raw M stays in
the caches.

Paper examples are selected within `STATE_SPACE_EXAMPLE_L2` (currently `1e-4`)
using full-dimensional effective-modulation task centers. Same-category task
distances are averaged equally across categories; different-category distances
are averaged equally across category pairs. The score is
`(between - within) / (between + within)`, with larger scores preferred.
Categories follow the six paper color groups. All valid seeds are ranked and
summarized in `paper_plot/multitask_state_space_centroid_scores.json`; missing
high-dimensional centers are reported, never replaced with a 2D score.
The best seed supplies all three PCA example panels, with names breaking ties.
Each of the three PCA figures (`multitask_state_space_{hidden,mod,eff_mod}.png`,
`--only state_space_combined`) has two panels: on the left that selected
L2 = 1e-4 run, on the right the run of the L2 = 1e-2 cohort
(`STATE_SPACE_STRONG_L2`), whose category structure is the strongest,
selected by the same criterion; that cohort's ranking is saved as
`paper_plot/multitask_state_space_centroid_scores_L21e2.json`. The two panels
are separate PCA spaces with their own axes, titled by their L2, under one
color key; a figure is skipped when either cohort has no analyzed run.

`python figures/paper_plot.py --only state_space_untrained` draws the untrained
control, `multitask_state_space_untrained.png`: the same context-end PCA of an
untrained network, one panel per representation (hidden state, M, W ⊙ M) at
the trained figures' panel size and colors. The network has the architecture
and config of the selected L2 = 1e-4 example run but fresh random weights
(seed `UNTRAINED_INIT_SEED` = 0, η at its initial value 1.0); no trained
weight is used, and the training run's own initialization was never saved.
It runs on the same evaluation trials, so its task separation is what the
rule-cue inputs produce through random weights. L2 acts only during training,
so one untrained network serves every cohort. Generate the caches first, one
network per `--init-seed` (eight here; each takes about as long as one
trained run), from the selected example run's checkpoint. That run is
re-ranked whenever the cohort's caches change (seed 921 among the first
seven L2 = 1e-4 runs, seed 202 among the 25 analyzed on 2026-10-23), and
the untrained figures only read controls built from the current one; when
they are missing, the skipped figures print the command:
```bash
python multiple_task/state_space_shift.py --untrained \
  --checkpoint multiple_tasks/savednet_everything_seed202_L21e4+hidden300+batch128+angle.pt \
  --init-seed 0 1 2 3 4 5 6 7
python multiple_task/state_space_structure.py
```
This writes only to `state_space/untrained/`: per seed the PCA cache
`state_space_pca_<aname>_init<seed>_noise0.01.pkl` (marked `untrained`, with
`init_seed` and `config_source`) and its diagnostic figures, and the networks'
own distance-angle results, `initial_condition_distance_vs_angle_results.pkl`
keyed by `<aname>_init<seed>`, with the fields of a trained run's entry
(permutation test and Hebbian-only steps included), the same marks, and
`l2_info` None (the config's L2 kept as `config_l2_info`). This is the
no-learning control of the distance-angle relation: the rule cue is on for the
whole trial, so the Hebbian step carries the cue as well, and distinct cues
could tie M distance to step angle without any training. The subdirectory
keeps these files out of the trained runs' structure glob, seed rankings and
R-value cohorts, and the batch run's cleanup leaves it in place. The PCA figure draws the network
of `STATE_SPACE_UNTRAINED_INIT_SEED` (0); its save message reports the
category separation S of each representation's untrained centroids.

`state_space_structure.py` gives every untrained network the same category
and task-vector tests as a trained run (`state_space/untrained/state_space_structure_<aname>_init<seed>_noise0.01.pkl`)
and stores their summaries under `untrained` in
`state_space_structure_results.pkl`, apart from `runs` and the L2 `cohorts`,
which they never enter. `python figures/paper_plot.py --only state_space_category_vs_untrained`
draws `multitask_state_space_category_vs_untrained.png`: the
`state_space_category_test` figure (hidden state and W ⊙ M, one point per
network, filled when its permutation p < 0.05, null 95th-percentile band,
median dash) with an "Untrained" column of the example config's untrained
networks to the left of the four L2 cohorts, set apart by a gap and a dotted
line. It is skipped, with the commands above, until the structure results
hold an untrained network of that config.

Distance-angle regression uses ordinary least squares with a fitted intercept:
`angle = intercept + slope * distance`. The existing `(r, slope, p)` tuples
remain unchanged in structure (`r` is Pearson correlation; that `p` is the
nominal OLS slope-test value, which assumes independent task pairs), while
`scatter[representation]` records the intercept (`regression`,
`through_origin=False`), the task names of every pair (`pair_rules`) and the
reported test, `permutation`: a two-sided task-label permutation (Mantel-type)
p of `r` from 10,000 relabelings of the 15 tasks
(`DISTANCE_ANGLE_N_PERMUTATIONS`, seed 0), because the 105 pairs share tasks.
Besides `hidden`, `mod` and `eff_mod`, whose first step is x[t] − x[t−1], each
entry holds `mod_hebb` and `eff_mod_hebb`. The modulation update is
M[t] = λ M[t−1] + η post ⊗ pre, clamped to the M bounds, so the full step
contains −(1 − λ) M[t−1]: a component along the initial condition itself, which
ties the step angle to the initial-condition distance without any
task-dependent computation. The Hebbian-only step x[t] − λ x[t−1] removes it,
with λ read from the checkpoint's `mp_layer1.lam` (saved as `retention`);
`step_norms` records the mean trial norms of the initial condition, the full
step and, for the modulation representations, its decay and Hebbian terms.
`paper_plot.py` draws saved values only: `state_space_dist_angle` shows three
panels of the selected run, the hidden state and W ⊙ M with the full first
step and W ⊙ M with the Hebbian term only (the two W ⊙ M panels on one y
range), each with r and the permutation p; `state_space_r_values` shows one
bar per representation and L2 cohort, raw M included (Hebbian-only bars
hatched, one dot per run, filled when its permutation p < 0.05), preceded by
an "Untrained" panel of the same bars when `state_space/untrained/` holds
distance-angle results of networks built from the selected example run's
config (one dot per init seed; without them the four cohort panels are drawn
and the command to add the control is printed). Caches
without the permutation test or the Hebbian steps are skipped, never refitted;
regenerate them with `state_space_shift.py`.

`python figures/paper_plot.py --only sibling_cue_interp` draws the multi-task analog
of the two-task pro/anti interpolation for the two sibling families (each
figure's run chosen automatically unless `DELAYDM_ANAME` / `DMCGO_ANAME` pin
one; see the end of this file):
`multitask_{delaydm,dmcgo}_cue_interp_{gradient,long_delay_endpoint}_{emodulation,hidden}_{2d,3d}.png`
and `multitask_{delaydm,dmcgo}_cue_interp_{gradient,long_delay_endpoint}_legend.png`.
`2d` draws one line per sweep trial against alpha for each PC of the plane
(square panels, alpha ticks on every panel, y limits hugging each panel's own
lines with 8% padding below and 25% above, so columns are not on a shared
scale) and `3d` the trial lines in the plane against alpha. The legend is a
separate figure per family and method,
holding the DMC cluster lines and, for the gradient method, the converged /
hollow markers (DelayDM endpoint sweeps need none). `python figures/paper_plot.py
--sibling-one-trial-per-stimulus --only sibling_cue_interp` draws one sweep
trial per stimulus direction instead of all of them, in `2d`, `3d`
and the `_cue_interp` endpoint overlays (`SIBLING_ONE_TRIAL_PER_STIMULUS`):
the target magnitude is the one every direction can match most closely (ties
toward 1.0) and each direction contributes its nearest trial, so the eight
DelayDM paths are at comparable stimulus strength (DMC trials, all at
magnitude 1, simply deduplicate). Output names are unchanged, so the option
replaces the all-trial figures.
`--sibling-avoid-attractor-switches` (implies the one-trial view;
`SIBLING_AVOID_ATTRACTOR_SWITCHES`) also makes each DelayDM direction prefer
a trial that stays on one attractor along the sweep. The settled DelayDM
endpoints are a discrete set of attractors, not a ring: on the seed 62 hidden
sweep the 32 trials of a base rule land on 7-16 distinct endpoints per alpha
(full 300-d hidden state, tolerance 0.01), shared across neighboring
directions, and they have mostly collapsed onto them 2.4 s into the delay. As
the cue moves the basin boundaries, a trial near one hops to a neighboring
attractor between two alphas, a full-space step of 0.13-0.26 against a median
of 0.04, which draws as a jump. The score
(`_cue_interp_attractor_switch_scores`) works in the six cached PCs: per
consecutive alpha pair, a trial's step minus the median step of all trials
(the common motion the cue gives every endpoint), per PC, over that PC's span
across the sweep; a trial whose largest value exceeds 0.1
(`_SIBLING_ATTRACTOR_SWITCH_FRACTION`) switches. The six-PC norm alone does not
work, since the common drift dominates it; the per-PC score flags 6/32 and
10/32 trials on seed 62, all of them trials that also hop in the full space.
The target magnitude is then re-chosen among the allowed trials, so the drawn
magnitudes match less closely (seed 62: 0.96 and 0.80, worst deviation 0.17 and
0.27, against 1.11 within 0.06), and each base rule chooses on its own. A
direction whose every trial switches keeps its smoothest trial (seed 62:
direction 0 in both columns), which the saved message names. DMC sweeps are
never filtered (`_SIBLING_ATTRACTOR_SWITCH_FAMILIES`), since their large step,
taken by every trial at the same alpha, is the response-cluster swap the
figures show. This is a selection for legibility, so the caption must state
it, e.g. "one trial per stimulus direction; trials whose endpoint switches to
a neighboring attractor along the sweep were excluded where possible".
`multiple_task/sibling_delay_analysis.py` runs the sweep after each method's
base analysis: the stimulus is held at one base rule's trials while its
one-hot rule cue fades out and the sibling's fades in,
`x_alpha = x_base + alpha * (e_sibling - e_base)`, over the 11 alphas 0.0,
0.1, ..., 1.0 fixed by `SIBLING_CUE_INTERP_N_ALPHA` (no CLI option; the
two-task sweep uses the same grid). The two-task sweep mixed whole trials, which is a cue
mix only because pro and anti share their stimulus; the DelayDM rules present
their stimulus on different modalities, so only the cue channels move here and
both rules serve as base in turn (`SIBLING_CUE_INTERP_BASE_RULES`), whereas
DMCGo and DMCNoGo share one stimulus structure and DMCGo alone is swept. For
DelayDM, alpha = 1 is an input the network never saw in training (each rule
presents its stimulus on its own modality); for DMC it is an ordinary, trained
DMCNoGo input for the same stimulus, and its long-delay endpoint lands on
DMCNoGo's own endpoint. The `long_delay_endpoint` sweep draws one fresh set of
long-delay trials (direction-balanced for DelayDM, random directions for DMC),
reruns the base rule's trials at every alpha and keeps the settled endpoint
(see below); the `gradient` sweep reuses the per-rule dense-angle template
(`core.grad_fixed_points.build_dense_stimulus_template`, eight directions, the
DelayDM magnitudes) and solves the true delay fixed point at every alpha with
the per-rule budgets. Both are projected onto the method's joint delay basis
(`*_cue_interp_*pc_projections_*.pkl`). The sweep figures pick their plane
on the sweep's own fixed points (see the end of this file): for DelayDM the
plane in which they stay best separated by stimulus direction at every
alpha, for DMC a PC on which the response clusters keep their order (top
row) beside a PC on which they swap (bottom row). Colors follow the alpha
ramp of SCHEME family 5 (dark at alpha = 0, light at alpha = 1; filled = converged,
hollow = not, for both methods). In the DMC figures the connecting lines are
colored by the base trial's memory cluster at alpha = 0. The two clusters are
the category the
network must respond to: "respond to category 1" holds DMCGo trials whose
first stimulus is in category 1 (locations 0-3, below pi) together with
DMCNoGo trials whose first stimulus is in category 2 (locations 4-7), since
both respond to a category-1 second stimulus; "respond to category 2" is the
complement. `python figures/paper_plot.py --only onetask_stimulus_colorwheel`
saves, beside the stimulus colorwheel, `onetask_stimulus_colorwheel_dmc_category`:
the eight DMC stimulus locations (0.05 pi to 1.80 pi, offset from the ring
directions so none lies on the boundary at 0 / pi) in their direction colors,
on a circle split into category 1 (upper half, blue) and category 2 (lower
half, green) in these cluster colors. The labels are read from the `group_labels` saved in the
projection cache or
derived from the stimulus for caches written before them, so the swap of the
two clusters under the sibling's cue is visible; DelayDM lines are colored by
their stimulus direction (the plain figure's clusters, SCHEME family 1), so
each direction's four trials can be followed across alpha. The long-delay
endpoint figures (`multitask_{delaydm,dmcgo}_long_delay_endpoint_{emodulation,hidden}`)
use the same colors for their endpoints and trajectories, the stimulus
direction for DelayDM and the response category for DMC (blue "respond to
category 1", green "respond to category 2", listed in the DMC endpoint
legend), with marker shape and trajectory line style giving the task; the
gradient-method geometry figures
(`multitask_{delaydm,dmcgo}_fixed_point_geometry_{emodulation,hidden}`) color
their fixed points the same way, the DMC categories in their in-panel legend.
When the endpoint sweep is cached, the base endpoint figure is saved unchanged
and a second copy with the `_cue_interp` suffix
(`..._long_delay_endpoint_{emodulation,hidden}_cue_interp`, legend
`..._long_delay_endpoint_legend_cue_interp`) overlays each sweep trial's path
across alpha in the same two planes: hue follows the endpoints (the stimulus
direction for DelayDM, the trial's alpha = 0 cluster for DMC),
the alpha = 0 start takes the base rule's marker (small, filled), the alpha =
1 end the sibling's marker, hollow, with dots at the intermediate alphas
along a dotted path; that legend gains one entry per sweep direction. The
sweep always runs together with the method's base
analysis, so rerunning a family regenerates its base artifacts from new random
trials as well.

**Long-delay endpoints are settled to fixed points.** The trials' own long
delay (DelayDM 5 s, DMC 10 s) is too short for the modulation to converge:
tau_M is 4 s at lambda = 0.99, and on seed 921 none of the 64 endpoints per
family met the gradient solver's criterion at the end of the trial (median
normalized residual 0.24 for DelayDM, 0.04 for DMC). Run past that delay on
the same input, the DelayDM W⊙M endpoints still moved 67% of their spread and
the direction separation of the e_modulation endpoint plane fell from 1.18 to
0.13; DMC moved under 10% and kept its geometry. Both the base endpoints and
every alpha of the endpoint sweep therefore keep running on the delay's
constant input after the trial's delay until every trial's leak-normalized
one-step residual ||F(M) - M|| / (||M|| (1 - lambda)) is at most
`DEFAULT_REL_TOL_UNDAMPED` (1e-2), the gradient solver's criterion, with
`SIBLING_LONG_DELAY_MAX_SETTLE_MS` (100 s) as the cap. The endpoint caches
record `settle_ms`, `endpoint_delay_ms`, each trial's `rel_step_undamped` and
`is_fixed`, and the projected trajectories run from delay onset to the
settled endpoint. A trial still above the criterion at the cap is drawn
hollow in the sweep figures and the PC-pair galleries (the automatic run
selection below skips such runs, so in `paper_plot.py` this happens only for a
pinned run); in the endpoint figure,
where hollow markers already mean the sweep's alpha = 1 end, its count is
printed with the saved figure instead. Endpoint caches written before
settling (endpoint version 2, sweep version 1) hold the unsettled end-of-trial
state; rerun the analysis to replace them.

**The PC pair and the example runs are chosen by fixed-point separation.**
Every figure picks two of its six joint delay PCs by how well they separate
the fixed points it draws, measured as the mean distance from each group's
centroid to the nearest other group's centroid divided by the mean distance
of the fixed points from their own group's centroid
(`core.sibling_geometry`). Each long-delay endpoint figure draws the default
PC1-PC2 plane beside the best pair of its endpoints
(`best_task_specific_pc_pair`), with both sibling tasks' fixed points pooled
and grouped by stimulus direction for DelayDM and by response category for
DMC ("respond to category 1 / 2", the memory clusters the sweep lines are
colored by), so the same stimulus or category of both tasks should cluster.
For DelayDM each task's mean endpoint is removed first (`remove_task_offsets`)
and the selected panel is drawn in those task-aligned coordinates (axis
labels "task-aligned"; trajectories and the overlaid sweep shifted with their
task), while PC1-PC2 stays raw. The two DelayDM tasks receive the stimulus on
different modalities and store it at places offset by a task-wide
translation: pooled raw, no run clusters the same direction across tasks (best
pair 0.91 of 25 runs on the 2026-10-22 caches, median 0.37; 0.13 in the full
W⊙M space), while with the offset removed seed 42 reaches 2.45 (median 0.65).
The aligned panel therefore shows a shared direction geometry up to a
translation, and the raw panel the translation itself; hidden states overlap
across tasks without it (1.26 raw, 1.33 aligned). DMC needs no alignment, its
response categories already share an axis across tasks. The DelayDM rule-cue
sweep figures score the sweep's own fixed points instead
(`best_sweep_pc_pair`): the separation by stimulus direction
at every alpha of both base rules, taken at its minimum, so a plane in which
the directions merge anywhere along the sweep, or a trial hops to another
direction's attractor, is not chosen. The endpoints alone cannot judge the
sweep, whose middle and alpha = 1 end (an input never trained) they never
see. The DMC sweep figure shows the two response clusters swapping under
the sibling's cue and draws each PC on its own row, so a separable-at-every-
alpha score would select against it and a pair scored as a plane can hide a
row with no clusters at all (an axis that only adds little spread). It
instead pairs two PCs chosen on their own (`best_dmc_sweep_pc_pair`): on each
PC the clusters, labelled at alpha = 0, are separated at both ends of the
sweep along that PC alone, and the smaller of the two scores the PC as kept
when the clusters keep their order and as swapped when it reverses. The best
kept PC is the top row ("what was seen") and the best swapped PC the bottom
row ("what to respond to"); the pair scores the smaller of the two. The
middle of the sweep is not scored, so a run whose fixed points hop between
the clusters mid-sweep can still be chosen. `python figures/paper_plot.py
--sibling-dmc-keep-every-alpha --only sibling_cue_interp`
(`SIBLING_DMC_KEEP_AT_EVERY_ALPHA`) instead requires the kept PC to keep the
order at every alpha and scores it by its smallest separation over all of
them: a cleaner top row, though the separation ratio is relative to the
clusters' spread, so the chosen swap can be small next to the row's range.
The swapped PC is judged at the ends either way, since a swap has to cross
somewhere in between; the option changes the DMC sweep figures' plane and
run, not their names. The sibling figures are
illustrations, so with `DELAYDM_ANAME` / `DMCGO_ANAME` = None (the default)
each figure (endpoint or sweep, per representation) also takes its own run:
the cached run of the `SIBLING_L2` cohort (L2 = 1e-4) with the highest
best-pair score of that figure, ties to the lowest seed; a run name in
either constant pins every
figure of that family to one run instead. The endpoint figure's run also
serves its `_cue_interp` overlay (the overlaid sweep shares that run's
basis, so it can look rougher than the standalone sweep figure) and the
gradient-method geometry figures; the sweep figure's run serves its `2d` and
`3d` views. Figures of one family can therefore show different networks.
Runs with a missing endpoint or sweep cache, a cache written before
settling, or any point above the fixed-point criterion are skipped. Every
run's scores and each figure's ranking are printed when the figures first
need them and saved as `multitask_sibling_run_scores_{delaydm,dmcgo}.json`
in `paper_plot/`; with no scorable run the figures print their usual
"Skipped" messages. Because the plane and the run are both picked for
legibility on the drawn fixed points, the figures show the best case of the
cohort, not a typical run: on the 2026-10-22 caches the chosen DelayDM
e_modulation sweep keeps its directions 3.0 apart (separation ratio) at its
worst alpha, against a cohort median of 0.71, so in most runs the directions
merge somewhere along the sweep. The saved rankings give the spread across
runs. The endpoint figures' axis limits follow the drawn points
(`_adaptive_pc_limits`) for every run.

