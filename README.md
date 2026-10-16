# MultiTaskMPN

Training and analysis of **Multi-Plastic Networks (MPNs)** on a battery of
cognitive tasks. A single recurrent network with Hebbian-like synaptic
plasticity learns many tasks at once; the structure of its plastic weights is
then analyzed to understand how task-specific computation is organized.

## Model

The core model is `DeepMultiPlasticNet` ([core/mpn.py](core/mpn.py)): a recurrent
network whose effective weights are modulated by a fast plasticity matrix **M**:

```
W_eff(t) = W + W ⊙ M(t)     (multiplicative)   or   W + M(t)   (additive)
```

**M** evolves by a Hebbian rule with learning-rate η and decay λ (each scalar,
pre/post-vector, or full matrix). In every training script η is learned and λ
is fixed (`lam_train = False`) at λ = 1 − Δt/`m_time_scale`, with Δt = 40 ms:
the single-task and pretraining runs use `m_time_scale` = 400 ms (λ = 0.9), the
two-task, multi-task and flexible-task runs use 4000 ms (λ = 0.99). See
[SCHEME.md](SCHEME.md) for where each model's time constant lives and how to
read λ back from a checkpoint. The network has three weight matrices:
`W_initial_linear` (input projection), `mp_layer1.W` (recurrent plastic weights),
and `W_output` (readout).

## Layout

Source is grouped by purpose. **Run scripts from the repository root**
(e.g. `python two_task/two_task.py`); data is written to the top-level data
directories. Experiment scripts import the shared library in `core/` through a
small `_bootstrap.py` shim that puts `core/` on `sys.path`, so flat imports
(`import mpn`, `import helper`) keep working.

| Path | Contents |
|---|---|
| `core/` | Shared library: model (`mpn`), tasks (`mpn_tasks`), training/base (`net_helpers`, `networks`), clustering, and utilities (`helper`, `color_func`, `plot_heatmap`) |
| `one_task/` | Single-task training, analysis, pipeline |
| `two_task/` | Two-task training, analysis, pipeline (+ notebooks) |
| `multiple_task/` | Multi-task training, analysis, lesion/pruning, state-space, pipeline |
| `pretrain/` | Pretraining → post-training transfer experiment + analysis |
| `flex_task/` | Flexible-task (RNN/MPN) training and analysis |
| `paper_plot.py` | Publication-figure generation (run from root) |

## Workflow

Each experiment family follows **train → analyze**, with multi-task adding
clustering and lesion/pruning. Hyperparameters (`hidden`, `batch`, `seed`,
regularization `feature`) are set inside each training script.

```bash
# multi-task: train, analyze (clustering), lesion, lesion plots
python multiple_task/multiple_task.py
python multiple_task/run_pipeline.py --seed 749 --feature L21e4
# several seeds of one cohort, run in the given order
python multiple_task/run_pipeline.py --feature L21e3 --seed 299 365 487 501 934

# optional sibling-task geometry only (does not rerun clustering or lesions)
python multiple_task/sibling_delay_analysis.py \
  --seed 921 --feature L21e4 --families delaydm1 --method gradient

# alternatively, use the end of a generated very-long delay
python multiple_task/sibling_delay_analysis.py \
  --seed 921 --feature L21e4 --families delaydm1 \
  --method long_delay_endpoint

# both methods end with a rule-cue interpolation sweep between the two sibling
# rules (11 alphas, fixed by SIBLING_CUE_INTERP_N_ALPHA)
python multiple_task/sibling_delay_analysis.py \
  --seed 921 --feature L21e4 --families delaydm1 dmcgo \
  --method long_delay_endpoint
python paper_plot.py --only sibling_cue_interp

# additional state-space analysis
python multiple_task/state_space_shift.py
# category structure and task-vector additivity of the saved task centroids
python multiple_task/state_space_structure.py
# context-end task distance vs lesion-profile similarity, for runs that also
# have lesion_plot.py caches
python multiple_task/state_space_vs_lesion.py
python paper_plot.py --only state_space_vs_lesion
python paper_plot.py --only state_space_vs_lesion_seeds
# rank state-space seeds and plot the best example without clearing other figures
python paper_plot.py --only state_space_combined
# untrained controls (fresh weights, the example run's config), one per seed,
# then rerun the structure analysis so their category tests are included
python multiple_task/state_space_shift.py --untrained \
  --checkpoint multiple_tasks/savednet_everything_seed921_L21e4+hidden300+batch128+angle.pt \
  --init-seed 0 1 2 3 4 5 6 7
python multiple_task/state_space_structure.py
python paper_plot.py --only state_space_untrained
python paper_plot.py --only state_space_category_vs_untrained
python paper_plot.py --only state_space_rdm
python paper_plot.py --only state_space_rdm_example
python paper_plot.py --only state_space_category_test

# single- / two-task (train + analyze chained by the pipeline)
python one_task/run_one_task_pipeline.py
python two_task/run_two_task_pipeline.py

# pretraining transfer: each run adds N_TRIALS (5) new seeds of PRETRAIN_RULESET
# (fdgo_delaygo) at the configured lambda (0.9); seeds with a completed run are
# never redrawn, so rerun it to grow the cohort, then refresh its analysis caches
python pretrain/pretraining.py
python pretrain/pretraining_analysis.py --feature L21e3lam0.9
python pretrain/pretraining_post.py --feature L21e3lam0.9

# paper figures
python paper_plot.py
# multi-task clustering, overmembership and network structure
python paper_plot.py multiple_tasks
# lesion effects and cluster comparisons
python paper_plot.py lesion
# pretraining: Relevant and Irrelevant motifs only
python paper_plot.py pretraining --pretraining-groups motifs
# pretraining: motifs plus DelayAnti and DelayPro (default)
python paper_plot.py pretraining --pretraining-groups all
# pretraining: the runs trained with lambda = 0.9 (m_time_scale 400 ms)
python paper_plot.py pretraining --pretraining-lambda 0.9
```

`--pretraining-groups {motifs,all}` controls the conditions shown in every
pretraining paper figure: backbone probe, transfer speed, learning trajectory,
rule vectors, principal angles, and both aggregate CVE figures. `motifs` selects
only Relevant/Irrelevant motifs; `all` also includes the DelayAnti/DelayPro
single-task controls and remains the default. The selection applies to both
combined and per-seed caches, including the CVE self-reference, and can be
combined with `--pretraining-bound mod2`, `--no-legend`, or `--only`, for example:
`python paper_plot.py --only learning_trajectory --pretraining-groups motifs`.
Other experiment modes are unaffected. Output filenames are unchanged, so
running the other group selection replaces the same figure; `--only` preserves
unrelated figures, while mode runs retain the output cleanup described below.

The pretraining figures read one variant of the `pretrain/pretraining.py`
runs (dmpn, hidden 200, L2 1e-3, batch 128), chosen by two options that select
the matching files in `pretraining_analysis/` by name and never mix variants:
`--pretraining-bound {mod1,mod2}` picks the M bound ([-1, 1], the default, or
[-2, 2], tagged `mb2`), and `--pretraining-lambda {0.99,0.9}` picks the M
retention per 40 ms step. `0.99` (default) is the historical cohort trained with
`m_time_scale` 4000 ms, whose file names carry no tag; `0.9` selects the runs
trained with 400 ms, tagged `lam0.9` (for example
`fdanti_delaygo_dmpn_seed3_+hidden200+L21e3lam0.9+batch128+angle_result.pkl`).
The two options combine (`L21e3mb2lam0.9`). Every pretraining figure aggregates
all cached seeds of the selected variant, and the run banner prints the
selected lambda, `m_time_scale` and feature string. Output filenames do not
encode the variant, so rendering another selection replaces the same figures.

`python paper_plot.py --only learning_trajectory_linear` exports
`learning_trajectory_linear_n.png`: the same seed and mean curves as
`learning_trajectory_n.png`, with a linear iteration axis instead of a log axis.
Both are included in `pretraining` mode and honor the group/bound options.
`rule_vectors_n.png` is drawn at the `learning_trajectory` size (the six-bar
`all` layout keeps that height and width per bar). Each bar keeps its two-line
"MemoryAnti ↔ partner" tick label, in a 5 pt font so that neighbouring labels
do not collide at that width.
For `--pretraining-groups motifs`, the four bars form two pairs (2 | 2) set
apart by a wider gap and a dark vertical divider; `all` keeps its evenly spaced
layout.

The `lesion` figure mode contains `lesion_heatmap`, `lesion_cluster_sizes`,
`cluster_corr_vs_lesion`, `cluster_corr_vs_lesion_weighted`,
`cluster_corr_vs_lesion_seeds`, `om_vs_lesion`, `om_vs_lesion_seeds`,
`plasticity_share`, `plasticity_share_cohort`, `causal_vs_activity_tasks`,
`causal_vs_activity_seeds`, `task_specificity`, `task_sharing_cohort`,
`state_space_vs_lesion` and `state_space_vs_lesion_seeds`. All but the four
`_seeds` and two `_cohort` figures read one run,
`LESION_ANAME`, independently of `ANAME`, which selects the `multiple_tasks`
clustering figures. These figures no longer run under `multiple_tasks`; use
`python paper_plot.py multiple_tasks lesion` for both groups. Existing figure
names and `--only FIGURE` commands are unchanged. The historical `leison`
spelling is now `lesion` in commands, module names, output names and cache keys:
use `multiple_task/lesion.py`, `multiple_task/lesion_plot.py` and the `lesion`
paper mode. Existing caches have NOT been migrated or renamed. Readers accept
their legacy filenames, keys and condition labels through a read-only adapter;
correctly spelled files take precedence when both exist. Conflicting new/old
keys in one cache raise an error rather than losing data. New results use only
the corrected spelling; this compatibility does not bypass schema validation.
Mode runs retain the existing
behavior of clearing top-level `paper_plot/` outputs; `--only` preserves other
figures.

The `multiple_tasks` and `lesion` figures read one L2 cohort of the
`everything_seed*_L2<l2>+hidden300+batch128+angle` runs, selected by
`MULTITASK_L2` in `paper_plot.py` (currently `1e3`) or per invocation with
`python paper_plot.py --multitask-l2 1e4 multiple_tasks lesion`. The aggregate
figures (overmembership, `cluster_corr_vs_lesion_seeds`, `om_vs_lesion_seeds`,
`plasticity_share_cohort`, `causal_vs_activity_seeds`, `task_sharing_cohort`,
`state_space_vs_lesion_seeds`) summarize every cached seed of that cohort. The single-run figures use the
cohort's designated seed from `MULTITASK_DESIGNATED_SEEDS` (749 for clustering
and 921 for lesions in the 1e4 cohort; none for 1e3) when its cache directory
holds pickles; otherwise `paper_plot` picks the cached seed covering the most of
the category's cache directories, lowest seed on ties, and prints the switch.
Each category resolves once, so its figures never mix runs, and a run is never
taken from another cohort; this also applies to `--only` and to imported plotting
functions. Empty directories and pickles of other runs do not count as cached
data. Missing or incompatible individual caches still use the existing per-figure
checks rather than switching seeds per figure.

`python paper_plot.py --only cluster_corr_vs_lesion` draws the four input/hidden
scatter figures and three modulation figures:
`multitask_cluster_corr_vs_lesion_modulation_norm_zero_W_n.png`,
`multitask_cluster_corr_vs_lesion_modulation_var_weighted_unnorm_zero_W_n.png`
and `multitask_cluster_corr_vs_lesion_modulation_abs_weighted_unnorm_zero_W_n.png`.
All use cached **zero_W** effects; the two unnormalized figures use var-weighted
(W·Var(M)) and abs-weighted (|W|·Var(M)) modulation and drop the unresponsive
cluster. A run whose `lesion_plot.py` caches predate the abs-weighted variant
skips only that figure. Each figure is scale-free
on both axes: x is the Pearson correlation between cluster mean tuning
profiles, y is the Pearson correlation between per-task lesion effects after
z-scoring each effect against its stored random-control repeats (control SD
floored at one accuracy point). The caches store the dissimilarity 1 − r and
the rank statistics of that quantity; `paper_plot.py` shows the similarity r,
so it flips y, the sign of the saved Spearman rho and the saved bin medians
(their quartiles swap; the two-sided permutation p is unchanged) without
refitting anything. Only clusters whose summed |z|
exceeds the 95th percentile of a control-only, leave-one-repeat-out null are
compared. The legend reports Spearman rho with a two-sided cluster-label
(Mantel-type) permutation p from 10,000 permutations, because the cluster pairs
share clusters and are not independent samples. The red connector through the
points joins the saved quantile-bin medians (`CLUSTER_CORR_N_BINS` = 4 bins of
the tuning correlation; tied values stay together) with the interquartile range
of each bin as a bar. It describes the shape of the relation, which is
concentrated among the most similarly tuned pairs rather than linear across the
range; it is not the reported statistic. `lesion_plot.py` still saves its OLS
guide for its own diagnostic figure, but `paper_plot.py` no longer draws it.
Caches written before the bin medians were added draw without the connector
until `lesion_plot.py` is rerun. All entries come from
`LESION_ANAME`. Coordinates
and statistics are read from the `schema_version=2` caches written by
`lesion_plot.py`; legacy caches holding OLS regressions are skipped, never
refitted. Missing or incompatible caches skip only the affected figure, with no
fallback to other variants or freeze_M. The caches also hold an L1
supplement (the former tuning-cosine vs lesion-effect L1 scatter and L1 vs the
pair's summed effect magnitude, which is what L1 mostly measures) that
`lesion_plot.py` draws in its per-run diagnostic figure; `paper_plot.py` does
not export it. The `_n` suffix is omitted with `--no-legend`.
`--only cluster_corr_vs_lesion_seeds` reads the same unnormalized hidden and
input caches for every sibling run of `LESION_ANAME` and plots each run's
similarity rho as a joined hidden/input pair, filled when the saved permutation
p is below 0.05, with the median over runs as a dash; a single run has only
8-17 clusters, so the cohort view is where the cluster-level relation should
be judged. The legend title carries the cohort-level test, a two-sided
Wilcoxon signed-rank p of the runs' rhos against zero for each cluster type,
computed in `paper_plot.py` from the saved per-run rhos (nothing below the run
level is recomputed). Like every lesion figure it describes the one cohort
selected by `--multitask-l2`; render it once per cohort to compare them.

`python paper_plot.py --only cluster_corr_vs_lesion_weighted` draws the separate
**Var(WM)** comparison from `LESION_ANAME`, exporting only
`multitask_cluster_corr_vs_lesion_modulation_weighted_unnorm_zero_W_n.png`.
The upstream `modulation_all_weighted_unnormalized`
features take variance **after** multiplying modulation by static W, giving
`W**2 * Var(M)`, unlike `var_weighted`, which gives `W * Var(M)`.
The plot reads only the `weighted-unnormalized_zero-W` scatter cache, drops
the last unresponsive cluster, and reuses the saved profile correlations,
z-scored lesion dissimilarities and permutation statistics without refitting.
No clustering or lesion experiment is rerun. Use an allocated compute node for
rendering; this `--only` command preserves all other figures.

The primary `lesion_heatmap` compares hidden neuron lesions with
var-weighted, unnormalized modulation **zero_W** lesions (selected synaptic
weights set to zero), as two task-by-cluster panels; the input neuron effects
saved in the same cache are not drawn, and the shared color scale covers only
the two drawn panels. Its `lesion_cluster_sizes` companion reads the same
zero_W cluster memberships. `lesion_plot.py` exports both zero_W and freeze_M
effects with explicit mode metadata; missing zero_W results are never replaced
by freeze_M in these two figures. Existing explicit zero_W cache keys remain
readable. The OM scatter reads the same zero_W mode (`paper_plot.OM_LESION_MODE`),
so heatmap, cluster sizes and scatter describe one lesion experiment; upstream
plasticity-share/cross-mode comparisons retain both intervention modes.

The OM-versus-lesion-profile scatter uses Spearman rank correlation and a
one-sided (negative association) modulation-cluster-footprint permutation test,
not linear regression. `lesion_plot.py` saves the scatter, `association`
(rho, matching permutation p, null statistics and test metadata), and
`binned_medians` in both single-mode and combined `schema_version=2` caches;
`paper_plot.py` takes the zero_W entry.
Up to five quantile bins provide median OM/L1 points and counts; tied OM values
stay together. The median connector is descriptive and is not forced to decrease.
`paper_plot.py` only reads these values, draws pale scatter points and the saved
median connector, and starts the L1 axis at zero. Pearson-only caches are skipped,
not silently reused. `python paper_plot.py --only om_vs_lesion_seeds` draws the
cohort view, `multitask_om_vs_lesion_seeds_n.png`: one point per sibling run of
`LESION_ANAME` (same feature, any seed), the saved zero_W Spearman rho of that
run's OM scatter, filled when its saved one-sided permutation p is below 0.05,
with the median over runs as a dash. Because the per-run test is one-sided
(negative association), the legend title carries the one-sided Wilcoxon
signed-rank p of the runs' rhos being below zero; the other `_seeds` figures
keep their two-sided test. Runs whose OM cache is missing, legacy or has an
undefined rho are left out with a note.

`python paper_plot.py --only plasticity_share` draws
`multitask_plasticity_share_n.png` from `plasticity_share_var-weighted-unnormalized_<aname>.pkl`.
For every (task, synapse cluster) cell whose zero_W effect is significant
(one-sided BH-FDR q = 0.05 against the stored random-control repeats),
`lesion_plot.py` saves the plasticity share = freeze_M effect / zero_W effect:
1 means freezing the cluster's plasticity costs the task as much as removing
its weights, 0 means the static wiring suffices. The figure shows one column
per task (no-working-memory tasks fdgo/fdanti/reactgo/reactanti first), the
saved cells as pale points and the saved task median as a dash; values outside
the fixed y-limits appear as hollow triangles at the edge. The figure shows
the values only: the one-sided task-level Mann-Whitney U test that
`lesion_plot.py` also saves (memory tasks exceeding no-memory tasks) is not
reported, and the former per-run `plasticity_share_seeds` figure was removed.
`--only plasticity_share_cohort` draws `multitask_plasticity_share_cohort_n.png`,
the same layout with every sibling run of `LESION_ANAME` (same feature, any
seed) overlaid: the saved significant cells of all runs pooled per task as
pale points, each run's saved task median as a short gray dash, and the median
of the pooled cells as the red dash; the legend title gives the run count. A
run whose task list differs from the others is left out with a note, and the
figure is skipped below two runs. The share compares two interventions and is
not an exact additive partition of static and plastic contributions.

`python paper_plot.py --only causal_vs_activity_tasks` draws
`multitask_causal_vs_activity_tasks_hidden_n.png` and `..._input_n.png` from
`causal_vs_activity_tasksim_<side>_<aname>.pkl`: one point per task pair, x the
Pearson correlation of the two tasks' period-averaged raw (unnormalized) task
variance profiles over that side's neurons, y the correlation of their
lesion-effect profiles over all unnormalized input and hidden neuron clusters
(the `causal_dependency_unnorm` matrix). Per-neuron normalization was dropped
because it discards the amplitude that predicts causal impact; the normalized
pairing is kept in the cache under `reference_variants`. The
legend reports the saved two-sided Spearman task-label (Mantel-type)
permutation p, and the line is the saved OLS guide (the cluster-level
tuning-vs-lesion scatters show bin medians instead, see above). `--only causal_vs_activity_seeds` reads the same
caches for every sibling run of `LESION_ANAME` (same feature, any seed) and
plots each run's hidden and input rho as a joined pair, filled when the saved
permutation p is below 0.05, with the median as a dash. A response-period-only
activity vector is not used: it also makes the input side correlate. Both
figures need `schema_version=2` caches written
by the current `lesion_plot.py`.

`python paper_plot.py --only task_specificity` draws
`multitask_task_specificity_counts_n.png` from `task_specificity_<aname>.pkl`
for input and hidden neuron clusters (unnormalized clustering) and
var-weighted zero_W synapse clusters, the unresponsive class excluded. A
cluster "impairs" a task when its zero_W lesion effect is significant
(one-sided z against the control repeats, BH-FDR q = 0.05, the
causal-dependency mask). The figure has one column per cluster type: every
cluster is a dot at the number of tasks it impairs (ties spread sideways, a
dot histogram), the dash is the median, and the gray band is the spread the
counts would have under the saved mixing null, which shuffles cluster identity
within each task and so keeps each task's number of impaired clusters and the
mean count (mean ± one SD, the square root of the median null variance).
Dots spreading beyond the band mean specialized and hub-like clusters rather
than a uniform mix; the tick label gives the cluster count and the saved
one-sided permutation p of the observed variance exceeding that null. Nothing
is recomputed in `paper_plot.py`. The same cache holds, per cluster type, the
task-pair Jaccard overlap of impaired clusters grouped by the component the
two tasks differ in (`TASK_PAIR_RELATIONS` in `lesion_plot.py`) with one-sided
task-label permutation tests. `python paper_plot.py --only task_sharing_cohort`
draws them at the cohort level, `multitask_task_sharing_cohort_n.png`: one
panel per cluster type and one column per relation (plus the pooled "All
related" column), each sibling run of `LESION_ANAME` a point at its excess
sharing, the saved mean Jaccard of the relation's pairs minus that of the
unrelated pairs, filled when the run's saved permutation p is below 0.05,
with the median over runs as a dash and, above each column, the one-sided
Wilcoxon signed-rank p of the runs' excess being above zero. Runs whose
cluster types or relation lists disagree are left out with a note; the
figure is skipped below two runs. The former per-run sharing figures were
removed.

`python multiple_task/state_space_vs_lesion.py` links the state-space and
lesion analyses over the task pairs. For every run with both a state-space
structure cache (`state_space/state_space_structure_<aname>_noise0.01.pkl`)
and lesion caches (`multiple_tasks_norm/<aname>/`), it relates the Euclidean
distance between the two tasks' full-dimensional context-end centroids (hidden
state and W ⊙ M, the saved distance matrices) to the similarity of their
lesion-effect profiles: for neuron clusters the saved `causal_similarity` of
`causal_vs_activity_tasksim_hidden_<aname>.pkl` (Pearson correlation of the
random-minus-lesion effects over all unnormalized input and hidden neuron
clusters, the causal-vs-activity matrix), for synapse clusters the Pearson
correlation of the var-weighted zero_W effects in
`normalized_lesion_effects_<aname>.pkl` over the synapse clusters without the
unresponsive class, whose label is read from the lesion pickle in
`multiple_tasks_perf/` (the class holds most synapses, so its size-matched
random control collapses the network and its effect is near −0.8 for every
task). Each (representation, cluster type) pair gets a two-sided Spearman
task-label permutation (Mantel) test (10,000 permutations, seed 0;
`core.state_space_geometry.mantel_spearman_permutation`), a negative rho
meaning that tasks whose context-end states lie closer depend on more similar
clusters. Nothing is recomputed from network activity, so it runs in seconds;
the results go to `state_space/state_space_vs_lesion_results.pkl` (one entry
per run with the pair coordinates, the tests with their null arrays and the
L2), which the state-space batch run clears like every top-level file there,
so rerun it after `state_space_structure.py` or `lesion_plot.py`.
`python paper_plot.py --only state_space_vs_lesion` draws
`multitask_state_space_vs_lesion_n.png` for `LESION_ANAME`: a 2 × 2 grid
with one row per representation (hidden-state and W ⊙ M centroid distance)
and one column per cluster type (neuron clusters, synapse clusters), one
point per task pair, x the centroid distance, y the lesion-profile
similarity, each panel's saved rho and permutation p in its legend and no
fitted line (the statistic is a rank correlation). `--only state_space_vs_lesion_seeds` draws
`multitask_state_space_vs_lesion_seeds_n.png`: per cluster type, every
sibling run's rho for the hidden-state and the W ⊙ M distance, filled when
its saved permutation p is below 0.05, the median over runs as a dash and
under each tick the two-sided Wilcoxon signed-rank p of the runs' rhos
against zero, computed in `paper_plot.py` from the saved per-run rhos; it is
skipped below two runs with results.

On a compute node, regenerate the post-processing cache and then the paper figure:
```bash
python multiple_task/lesion_plot.py --seed 921 --feature L21e4
python paper_plot.py --only om_vs_lesion
```
Use `--seed all` in the first command to update every matching run's OM results.
This reuses completed lesion experiments, but clears and regenerates the selected
runs' post-processing outputs in `multiple_tasks_norm/`.

## Modulation cluster pickle size

`cluster_info_mod_{aname}.pkl` stores each grouped modulation clustering
result with `col_labels_by_k` only at the fixed lesion k (20) and at that
result's tolerance-selected k (`clustering.prune_labels_by_k`, recorded in
`col_labels_by_k_kept`). These are the only k values `lesion.py` reads back;
the between-modulation metric sweep over other k runs inside
`multiple_task_analysis.py` before saving. Pickles written before this change
hold one 90,000-synapse label array per candidate k (up to k = G = 1000) and
are several GB; they remain readable.

## Unresponsive classes

Every clustering appends the silent (unresponsive) neurons or synapses as one
extra class, labelled `k + 1`. Since 2026-09-29 that class is recorded
explicitly instead of being inferred from its position: each clustering result
carries `col_unresponsive_mask` / `col_unresponsive_label` (and
`col_unresponsive_label_by_k` for the grouped modulation clusterings), each
`cluster_info_{aname}.pkl` entry carries `unresponsive_label` and
`unresponsive_neurons`, the OM caches carry `unresponsive_input_index` /
`unresponsive_hidden_index`, and `lesion_prune_results_{aname}.pkl` records
`unresponsive_labels` / `unresponsive_conditions` per neuron variant,
`unresponsive_pre_label` / `unresponsive_post_label` per combined lesion and
`unresponsive_label` per modulation lesion (`None` when nothing was flagged).
`lesion_plot.py` and `paper_plot.py` read these fields; caches written before
they existed fall back to the former rule (the last cluster of an unnormalized
variant). The `cluster_corr_vs_lesion`
caches keep their `exclude_last_cluster` flag, now meaning that an unresponsive
exclusion was applied, and add `unresponsive_source`.

## Modulation clustering variants

The plastic synapses of `mp_layer1` are clustered under five task-variance
feature definitions, registered once in `core/modulation_variants.py` and
shared by `multiple_task_analysis.py` (clustering), `lesion.py` (lesions) and
`lesion_plot.py` (per-variant comparisons, in this order and with fixed colors):

| saved name | feature |
|---|---|
| `modulation_all_normalized` | Var(M), each synapse divided by its max over rules |
| `modulation_all_unnormalized` | Var(M) |
| `modulation_all_weighted_unnormalized` | Var(W·M), i.e. W²·Var(M) |
| `modulation_all_var_weighted_unnormalized` | W·Var(M), signed: positive- and negative-W synapses form separate clusters (in caches produced before 2026-09-25 every negative-W synapse had been classified as unresponsive, so those results describe positive-W synapses only) |
| `modulation_all_abs_weighted_unnormalized` | \|W\|·Var(M), magnitude-weighted and sign-blind |

Two sign-aware steps are identities for the non-negative variants and matter
only for `var_weighted`: `clustering.unresponsive_row_mask` judges a synapse
unresponsive by its mean |feature| (the previous signed mean treated every
negative-W synapse as unresponsive), and `modulation_variants.signed_log1p`
compresses unnormalized features symmetrically before clustering. Because the
signed variant's negative clusters have negative mean profiles, `lesion_plot.py`
compares its cluster tuning profiles by absolute value (`tuning_profile` in the
cluster-correlation caches). `abs_weighted` was added after the others; existing
`cluster_info_mod_*.pkl` and `lesion_prune_results_*.pkl` files do not contain
it, and their `var_weighted` entries predate the sign fixes. To obtain it, rerun
the pipeline for the run (`python multiple_task/run_pipeline.py --seed 921
--feature L21e4`), which reclusters, lesions and re-plots all five variants.
The lesion paper figures keep reading the var-weighted variant, while the
`multiple_tasks` mode draws one aggregate over-membership figure per variant
(`overmembership_norm`, `overmembership_unnorm`, `overmembership_weighted`,
`overmembership_var_weighted`, `overmembership_abs_weighted`); a run without
the `abs_weighted` prepost-belonging pickle is skipped by that figure only.
`--only modulation_abs_weighted` draws the clustered task-variance heatmap of
the `abs_weighted` variant (`multitask_clustered_modulation_abs_weighted_n.png`),
the |W|·Var(M) counterpart of the normalized `modulation` heatmap: same layout
and G=300 grouping, but the unnormalized cells are log-scaled from a 1e-4 floor
(the analysis script's own floor) to the run's maximum, so it exports its own
colorbar, `multitask_clustered_modulation_abs_weighted_colorbar_n.png`, instead
of sharing `heatmap_colorbar`.

Key data outputs: `multiple_tasks/` (checkpoints, curves),
`multiple_tasks_analysis/` (per-run analysis figures, cluster info),
`two_in_multiples/` (sibling-task delay/fixed-point analysis),
`multiple_tasks_perf/` and `multiple_tasks_norm/` (lesion results/plots),
`onetask/`, `twotasks/`, `pretraining/`, `state_space/`, `paper_plot/`.

The state-space workflow is `state_space_shift.py` -> `state_space_structure.py`
-> `paper_plot.py`, with `state_space_vs_lesion.py` after
`state_space_structure.py` for the runs that also have lesion caches (see the
`lesion` figures above).
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
so it runs in seconds). (1) Category structure: the separation score S above
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

`python paper_plot.py --only state_space_untrained` draws the untrained
control, `multitask_state_space_untrained.png`: the same context-end PCA of an
untrained network, one panel per representation (hidden state, M, W ⊙ M) at
the trained figures' panel size and colors. The network has the architecture
and config of the selected L2 = 1e-4 example run but fresh random weights
(seed `UNTRAINED_INIT_SEED` = 0, η at its initial value 1.0); no trained
weight is used, and the training run's own initialization was never saved.
It runs on the same evaluation trials, so its task separation is what the
rule-cue inputs produce through random weights. L2 acts only during training,
so one untrained network serves every cohort. Generate the caches first, one
network per `--init-seed` (eight here, as many as the largest trained cohort;
each takes about as long as one trained run):
```bash
python multiple_task/state_space_shift.py --untrained \
  --checkpoint multiple_tasks/savednet_everything_seed921_L21e4+hidden300+batch128+angle.pt \
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
which they never enter. `python paper_plot.py --only state_space_category_vs_untrained`
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

`python paper_plot.py --only sibling_cue_interp` draws the multi-task analog
of the two-task pro/anti interpolation for the two sibling families
(`DELAYDM_ANAME`, `DMCGO_ANAME`):
`multitask_{delaydm,dmcgo}_cue_interp_{gradient,long_delay_endpoint}_{emodulation,hidden}_{2d,3d,plane}.png`
and `multitask_{delaydm,dmcgo}_cue_interp_{gradient,long_delay_endpoint}_legend.png`.
`2d` draws one line per sweep trial against alpha for each PC of the plane
(square panels, alpha ticks on every panel, y limits hugging each panel's own
lines with 8% padding below and 25% above, so columns are not on a shared
scale), `3d` the trial lines in the plane against alpha, and `plane` every
trial's path through the
plane from alpha = 0 to alpha = 1 without averaging (group color on the path,
stimulus direction for DelayDM and memory cluster for DMC, alpha ramp on its
markers), which shows the alpha = 0 and alpha = 1 configurations of the
trials directly. The legend is a separate figure per family and method,
holding the DMC cluster lines and, for the gradient method, the converged /
hollow markers (DelayDM endpoint sweeps need none). `python paper_plot.py
--sibling-one-trial-per-stimulus --only sibling_cue_interp` draws one sweep
trial per stimulus direction instead of all of them, in `2d`, `3d`, `plane`
and the `_cue_interp` endpoint overlays (`SIBLING_ONE_TRIAL_PER_STIMULUS`):
the target magnitude is the one every direction can match most closely (ties
toward 1.0) and each direction contributes its nearest trial, so the eight
DelayDM paths are at comparable stimulus strength (DMC trials, all at
magnitude 1, simply deduplicate). Output names are unchanged, so the option
replaces the all-trial figures.
`multiple_task/sibling_delay_analysis.py` runs the sweep after each method's
base analysis: the stimulus is held at one base rule's trials while its
one-hot rule cue fades out and the sibling's fades in,
`x_alpha = x_base + alpha * (e_sibling - e_base)`, over the 11 alphas 0.0,
0.1, ..., 1.0 fixed by `SIBLING_CUE_INTERP_N_ALPHA` (no CLI option; the
two-task sweep uses the same grid). The two-task sweep mixed whole trials, which is a cue
mix only because pro and anti share their stimulus; the DelayDM rules present
their stimulus on different modalities, so only the cue channels move here and
both rules serve as base in turn (`SIBLING_CUE_INTERP_BASE_RULES`), whereas
DMCGo and DMCNoGo share one stimulus structure and DMCGo alone is swept. alpha =
1 is an input the network never saw in training. The `long_delay_endpoint`
sweep reruns fresh balanced long-delay trials at every alpha and keeps the
delay-1 endpoint; the `gradient` sweep reuses the per-rule dense-angle template
(`core.grad_fixed_points.build_dense_stimulus_template`, eight directions, the
DelayDM magnitudes) and solves the true delay fixed point at every alpha with
the per-rule budgets. Both are projected onto the method's joint delay basis
(`*_cue_interp_*pc_projections_*.pkl`). The DelayDM sweep figures pick their
PC plane on the sweep itself (`core.sibling_geometry.best_delaydm_sweep_pc_pair`):
the pair maximizing the directions' minimum separation-over-spread ratio along
alpha times the paths' smoothness (one minus half the mean second-difference
share of each trial's path length), scored over all trials and taken as the
minimum over the two base rules, so a plane in which the directions merge at an
intermediate cue or the paths zigzag is not chosen. The DMC sweep figures use
the base figure's plane: the configured plane for gradient, the best
task-specific endpoint pair for endpoints. Colors follow the alpha ramp of SCHEME family 5 (dark at alpha =
0, light at alpha = 1; filled = converged, hollow = not; endpoints always
filled). In the DMC figures the connecting lines are colored by the base
trial's memory cluster at alpha = 0. The two clusters are the category the
network must respond to: "respond to category 1" holds DMCGo trials whose
first stimulus is in category 1 (locations 0-3, below pi) together with
DMCNoGo trials whose first stimulus is in category 2 (locations 4-7), since
both respond to a category-1 second stimulus; "respond to category 2" is the
complement. The labels are read from the `group_labels` saved in the
projection cache or
derived from the stimulus for caches written before them, so the swap of the
two clusters under the sibling's cue is visible; DelayDM lines are colored by
their stimulus direction (the plain figure's clusters, SCHEME family 1), so
each direction's four trials can be followed across alpha. When the endpoint
sweep is cached, the base endpoint figure
(`multitask_{delaydm,dmcgo}_long_delay_endpoint_{emodulation,hidden}`) is
saved unchanged and a second copy with the `_cue_interp` suffix
(`..._long_delay_endpoint_{emodulation,hidden}_cue_interp`, legend
`..._long_delay_endpoint_legend_cue_interp`) overlays each sweep trial's path
across alpha in the same two planes: hue stays with the stimulus direction,
the alpha = 0 start takes the base rule's marker (small, filled), the alpha =
1 end the sibling's marker, hollow, with dots at the intermediate alphas
along a dotted path; that legend gains one entry per sweep direction. The
sweep always runs together with the method's base
analysis, so rerunning a family regenerates its base artifacts from new random
trials as well.

Both `pretrain/pretraining_analysis.py` and `pretrain/pretraining_post.py` write
analysis data to `pretraining_analysis/`, pooled figures to `pretrain/fig/`, and
per-seed figures to `pretrain/fig_seed/`. The latter includes per-checkpoint
PCA/sanity checks, single-checkpoint diagnostics, and seed-filtered accuracy plots.
Existing outputs with matching filenames are overwritten. Training inputs and
checkpoints remain in `pretraining/`; `paper_plot.py` reads the centralized
analysis data and continues to save publication figures in `paper_plot/`.

## Naming convention

Checkpoints and result files share an identifier string:

```
{task}_seed{seed}_{feature}+hidden{hidden}+batch{batch}{accfeature}
# e.g. everything_seed749_L21e4+hidden300+batch128+angle
```

Analysis scripts parse this `aname` to locate the matching files.

## Requirements

Python 3.9+, PyTorch (CUDA optional), NumPy/SciPy/scikit-learn,
Matplotlib/seaborn, h5py/hdf5plugin, scienceplots.

## Acknowledgements

Parts of this codebase were written with the assistance of
[Claude Code](https://claude.ai/claude-code).
