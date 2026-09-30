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

# optional sibling-task geometry only (does not rerun clustering or lesions)
python multiple_task/sibling_delay_analysis.py \
  --seed 921 --feature L21e4 --families delaydm1 --method gradient

# alternatively, use the end of a generated very-long delay
python multiple_task/sibling_delay_analysis.py \
  --seed 921 --feature L21e4 --families delaydm1 \
  --method long_delay_endpoint

# additional state-space analysis
python multiple_task/state_space_shift.py
# rank state-space seeds and plot the best example without clearing other figures
python paper_plot.py --only state_space_combined

# single- / two-task (train + analyze chained by the pipeline)
python one_task/run_one_task_pipeline.py
python two_task/run_two_task_pipeline.py

# pretraining transfer
python pretrain/pretraining.py

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

`python paper_plot.py --only learning_trajectory_linear` exports
`learning_trajectory_linear_n.png`: the same seed and mean curves as
`learning_trajectory_n.png`, with a linear iteration axis instead of a log axis.
Both are included in `pretraining` mode and honor the group/bound options.
For `--only rule_vectors --pretraining-groups motifs`, a vertical divider
separates the four bars into two pairs (2 | 2); `all` keeps its existing layout.

The `lesion` figure mode contains `lesion_heatmap`, `lesion_cluster_sizes`,
`cluster_corr_vs_lesion`, `cluster_corr_vs_lesion_weighted`, `om_vs_lesion`,
`plasticity_share`, `plasticity_share_seeds`, `causal_vs_activity_tasks`,
`causal_vs_activity_seeds` and `task_specificity`. All but the two `_seeds`
figures read one run, `LESION_ANAME` (seed 921), independently of `ANAME`,
which keeps selecting the `multiple_tasks` clustering figures; seed 921 is the
run whose hidden-neuron tuning-vs-lesion effect is clearest of the seven
L2=1e-4 seeds. These figures no longer run under `multiple_tasks`; use
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

`python paper_plot.py --only cluster_corr_vs_lesion` draws the four input/hidden
scatter figures and two modulation figures:
`multitask_cluster_corr_vs_lesion_modulation_norm_zero_W_n.png` and
`multitask_cluster_corr_vs_lesion_modulation_var_weighted_unnorm_zero_W_n.png`.
Both use cached **zero_W** effects; the unnormalized figure uses var-weighted
modulation and drops the unresponsive cluster. Each figure is scale-free
on both axes: x is the Pearson correlation between cluster mean tuning
profiles, y is one minus the Pearson correlation between per-task lesion
effects after z-scoring each effect against its stored random-control repeats
(control SD floored at one accuracy point). Only clusters whose summed |z|
exceeds the 95th percentile of a control-only, leave-one-repeat-out null are
compared. The legend reports Spearman rho with a two-sided cluster-label
(Mantel-type) permutation p from 10,000 permutations, because the cluster pairs
share clusters and are not independent samples. The line through the points is
an ordinary least-squares fit saved by `lesion_plot.py` as a visual guide to the
trend; it is not the reported statistic and carries no p-value. Caches written
before the line was added draw without it until `lesion_plot.py` is rerun. All entries come from
`LESION_ANAME`; across the seven L2=1e-4 seeds the hidden-neuron unnormalized
entry's rho is negative in every run (similar tuning, similar lesion profile),
and seed 921 is where it is clearest. Coordinates
and statistics are read from the `schema_version=2` caches written by
`lesion_plot.py`; legacy caches holding OLS regressions are skipped, never
refitted. Missing or incompatible caches skip only the affected figure, with no
fallback to other variants or freeze_M. The caches also hold an L1
supplement (the former tuning-cosine vs lesion-effect L1 scatter and L1 vs the
pair's summed effect magnitude, which is what L1 mostly measures) that
`lesion_plot.py` draws in its per-run diagnostic figure; `paper_plot.py` does
not export it. The `_n` suffix is omitted with `--no-legend`.

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

The primary `lesion_heatmap` compares input/hidden neuron lesions with
var-weighted, unnormalized modulation **zero_W** lesions (selected synaptic
weights set to zero). Its `lesion_cluster_sizes` companion reads the same
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
`paper_plot.py` takes the zero_W entry (across the seven L2=1e-4 seeds zero_W
gives rho between -0.38 and -0.52, freeze_M between -0.23 and -0.47, all with
permutation p < 0.02).
Up to five quantile bins provide median OM/L1 points and counts; tied OM values
stay together. The median connector is descriptive and is not forced to decrease.
`paper_plot.py` only reads these values, draws pale scatter points and the saved
median connector, and starts the L1 axis at zero. Pearson-only caches are skipped,
not silently reused.

`python paper_plot.py --only plasticity_share` draws
`multitask_plasticity_share_n.png` from `plasticity_share_var-weighted-unnormalized_<aname>.pkl`.
For every (task, synapse cluster) cell whose zero_W effect is significant
(one-sided BH-FDR q = 0.05 against the stored random-control repeats),
`lesion_plot.py` saves the plasticity share = freeze_M effect / zero_W effect:
1 means freezing the cluster's plasticity costs the task as much as removing
its weights, 0 means the static wiring suffices. The figure shows one column
per task (no-working-memory tasks fdgo/fdanti/reactgo/reactanti first), the
saved cells as pale points, the saved task median as a dash, and the saved
one-sided task-level Mann-Whitney U p for memory tasks exceeding no-memory
tasks; values outside the fixed y-limits appear as hollow triangles at the
edge. Across the seven L2=1e-4 seeds the memory-family median exceeds the
no-memory median in six (seed 692 ties), with per-seed p between 0.001 and 0.7.
The share compares two interventions and is not an exact additive partition of
static and plastic contributions. `--only plasticity_share_seeds` reads the same
cache for every sibling run of `LESION_ANAME` and plots, per run, the median
over tasks of the saved task medians for the no-memory and memory families as a
joined pair, filled when the saved Mann-Whitney p is below 0.05, with the median
over runs as a dash.

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
permutation p, and the line is the saved OLS guide, matching the cluster-level
tuning-vs-lesion scatters. `--only causal_vs_activity_seeds` reads the same
caches for every sibling run of `LESION_ANAME` (same feature, any seed) and
plots each run's hidden and input rho as a joined pair, filled when the saved
permutation p is below 0.05, with the median as a dash. Across the seven
L2=1e-4 seeds the hidden rho is 0.27 to 0.50 with permutation p < 0.05 in all
seven, while the input rho stays within -0.11 to 0.14 and is never significant
(under the earlier normalized pairing hidden was 0.11 to 0.35 with three seeds
significant). A response-period-only activity vector is not used: it also makes
the input side correlate. Both figures need `schema_version=2` caches written
by the current `lesion_plot.py`.

`python paper_plot.py --only task_specificity` draws
`multitask_task_specificity_counts_n.png` and one
`multitask_task_sharing_relations_<type>_n.png` per cluster type (input and
hidden neuron clusters from the unnormalized clustering, var-weighted zero_W
synapse clusters; the unresponsive class excluded) from
`task_specificity_<aname>.pkl`. A cluster "impairs" a task when its zero_W
lesion effect is significant (one-sided z against the control repeats, BH-FDR
q = 0.05, the causal-dependency mask). The counts figure shows the fraction of
clusters impairing 0 to 15 tasks, with a one-sided permutation p for the count
variance exceeding a null that shuffles cluster identity within each task
(small p: specialized or hub-like rather than uniformly mixed). The sharing
figures show, per task pair, the Jaccard overlap of impaired clusters grouped
by the component the two tasks differ in (`TASK_PAIR_RELATIONS` in
`lesion_plot.py`: response rule, timing, modality, context cue, integration
family, match/category family, other), with a task-label permutation p per
group and for all related pairs pooled. Nothing is recomputed in
`paper_plot.py`.

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
variant), which holds for the seven L2=1e-4 seeds. The `cluster_corr_vs_lesion`
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
`paper_plot.py` is unchanged and keeps reading the var-weighted variant.

Key data outputs: `multiple_tasks/` (checkpoints, curves),
`multiple_tasks_analysis/` (per-run analysis figures, cluster info),
`two_in_multiples/` (sibling-task delay/fixed-point analysis),
`multiple_tasks_perf/` and `multiple_tasks_norm/` (lesion results/plots),
`onetask/`, `twotasks/`, `pretraining/`, `state_space/`, `paper_plot/`.

The state-space workflow is `state_space_shift.py` -> `paper_plot.py`.
The analysis saves original-feature task centroids, trial counts and display
PCA together in `state_space/state_space_pca_<aname>_noise0.01.pkl`, alongside
the existing distance-angle results. Display PCA is fitted in batches; the
analysis otherwise retains its full-trajectory workflow and memory requirements.
The batch analysis clears top-level files in `state_space/` before regenerating
the four standard L2 cohorts.

Paper examples are selected within `STATE_SPACE_EXAMPLE_L2` (currently `1e-4`)
using full-dimensional effective-modulation task centers. Same-category task
distances are averaged equally across categories; different-category distances
are averaged equally across category pairs. The score is
`(between - within) / (between + within)`, with larger scores preferred.
Categories follow the six paper color groups. All valid seeds are ranked and
summarized in `paper_plot/multitask_state_space_centroid_scores.json`; missing
high-dimensional centers are reported, never replaced with a 2D score.
The best seed supplies all three PCA example panels, with names breaking ties.

Distance-angle regression uses ordinary least squares with a fitted intercept:
`angle = intercept + slope * distance`. The existing `(r, slope, p)` tuples
remain unchanged in structure, while `scatter[representation]["regression"]`
records the intercept and `through_origin=False`. Paper plots use these saved
coefficients and skip legacy caches without a free-intercept fit. Reported
`r` is Pearson correlation; `p` is the nominal OLS slope-test value, which
assumes independent task pairs and is not adjusted for their shared tasks.

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
