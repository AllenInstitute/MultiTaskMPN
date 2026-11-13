# Multi-task clustering and lesion analyses

Details behind the `multiple_tasks` and `lesion` figure modes of
`paper_plot.py`: which caches each figure reads, what its statistics are, and
the conventions of the clustering caches. `python multiple_task/run_pipeline.py`
produces the per-run clustering, lesion and control caches, the steps that
evaluate the network; `python multiple_task/run_pipeline_analysis.py --feature
<feature>` then runs every post-analysis that only reads them (the lesion-plot
caches, the row-sharing summary, the synapse variance decomposition and the
state-space-vs-lesion link; see [Post-analysis of a cohort](#post-analysis-of-a-cohort)).
The commands are in the README; the paper figures are a separate step.

The trial-time variance revision invalidates the old clustering-dependent
results. Numerical observations below labeled as historical were computed
with S1-conditioned features and must be rechecked after rerunning; their
presence here does not imply that new results have been generated.

## Run selection of the pipeline

```bash
# multi-task: train, analyze (clustering), lesion, lesion controls. Every training
# script (one_task, one_task_rnn, two_task, multiple_task, pretraining) grows
# its cohort the same way (core/seed_selection.py): a run adds N_TRIALS seeds
# new to the setting configured in the script, never redrawing a seed with a
# saved checkpoint or a started run, so rerun it to add trials; --n-trials N
# changes the count and --seed S [S ...] trains explicit seeds (refused when
# their checkpoint exists). The runs just trained are listed in the output
# directory's last_run_anames.txt, which the pipelines read.
python multiple_task/multiple_task.py
python multiple_task/multiple_task.py --n-trials 3
python multiple_task/multiple_task.py --seed 11 42
python multiple_task/run_pipeline.py --seed 749 --feature L21e4
# several seeds of one cohort, run in the given order
python multiple_task/run_pipeline.py --feature L21e3 --seed 299 365 487 501 934
# Every checkpoint of a cohort needing current pipeline outputs. Completion
# requires the current variance marker, the requested --control-repeats,
# and outputs newer than their inputs:
# cluster_info and cluster_info_mod, lesion_prune_results and the
# lesion_controls pickle of --control-pool (responsive by default); with
# --with-plots, which makes lesion_plot the pipeline's fourth step as it was
# until 2026-11-09, also the two lesion_interaction pickles of each plotted
# pool (multiple_tasks_norm/ and multiple_tasks_norm_<pool>/). With --resume,
# current steps are reused; missing/outdated steps and their dependents
# rerun. Without --resume, selected runs restart at clustering.
# --skip-completed needs --feature or --seed; to spread a cohort over nodes,
# give each process its own --seed list (two bare runs would pick the same
# checkpoints). Historical timing was about an hour of clustering plus four
# to six hours of lesions and one to two hours of controls per synapse type;
# clustering with the new variance estimator has not yet been timed.
python multiple_task/run_pipeline.py --feature L21e4 --skip-completed
python multiple_task/run_pipeline.py --feature L21e4 --skip-completed --resume
```

## Post-analysis of a cohort

Everything downstream of the pipeline only reads its outputs, so it is rerun
for a whole feature whenever more runs finish or the post-processing code
changes (a new cache field, a new statistic). `run_pipeline_analysis.py`
issues the commands in their dependency order, each in its own process, and
stops at the first failure:

| Stage | Command | Output |
|---|---|---|
| `lesion_plot` | `lesion_plot.py --seed S --feature F --control-pool P` for every run with a lesion pickle, with `lesion.py`'s controls and then `--control-pool`'s (responsive by default) | `multiple_tasks_norm/<aname>/`, `multiple_tasks_norm_<pool>/<aname>/` |
| `row_sharing` | `sibling_row_sharing.py --feature F` (and `--untrained` with `--untrained`) | `multiple_tasks_diagnosis/sibling_row_sharing_*.pkl` |
| `variance` | `synapse_variance_decomposition.py --seed all --feature F --skip-completed` | one pickle per run in `multiple_tasks_analysis/<aname>/` |
| `vs_lesion` | `state_space_vs_lesion.py` | `state_space/state_space_vs_lesion_results.pkl` |
| `figures` (not a default stage) | `paper_plot.py multiple_tasks lesion --multitask-l2 <strength>`, then the lesion mode for the other pool | `paper_plot/`, `paper_plot_<pool>/` |

```bash
python multiple_task/run_pipeline_analysis.py --feature L21e4
# list the commands without running them
python multiple_task/run_pipeline_analysis.py --feature L21e4 --dry-run
# only some stages, always run in the order of the table
python multiple_task/run_pipeline_analysis.py --feature L21e4 --stages lesion_plot vs_lesion
# the per-run stages (lesion_plot, variance) for the runs that just finished
python multiple_task/run_pipeline_analysis.py --feature L21e4 --seed 303 692 921
```

A run without a lesion pickle, or without the controls pickle of a pool, is
left out of `lesion_plot` and named at the end instead of failing the batch.
The variance decomposition takes about five minutes per run, so the stage
reuses a saved result that is newer than the run's clustering pickles
(`--skip-completed`, `result_is_current`); `--force` recomputes. The cohort
summaries (`row_sharing`, `vs_lesion`) always cover every available run,
whatever `--seed` says. The `figures` stage clears `paper_plot/` like every
mode run, so the other modes must be redrawn after it; it needs a plain L2
feature such as `L21e4`. The analyses that evaluate networks stay with their
own drivers: `run_pipeline.py` and `run_state_space_pipeline.py`; the
sibling-task geometry (`sibling_delay_analysis.py`) redraws its trials and is
never rerun from here.

## Accuracy figures: tests against the default setting

Every accuracy figure of the `acc_plot` mode tests its other columns against
the column of the default setting, the `L21e4` cohort (L2 = 1e-4, Tanh,
M in [-1, 1], lambda = 0.99, 300-dimensional projection and hidden layer):

| Figure | Reference column | Tests |
|---|---|---|
| `l2_accuracy`, `l2_task_accuracy` | L2 = 1e-4 | one per other L2 strength (and task, up to 45) |
| `l2e4_activation_accuracy` | Tanh | one per other activation |
| `projection_dim_accuracy`, `hidden_dim_accuracy` | width 300 | one per other width |
| `projection_dim_task_accuracy`, `hidden_dim_task_accuracy` | width 300 | one per other width and task (up to 75) |
| `retention_accuracy`, `retention_task_accuracy` | lambda = 0.99 | one per other retention (and task) |
| `modulation_bound_accuracy` | mb1 of the same L2 | one per L2 panel |

The modulation-bound figure is the one exception to the global default: each
of its panels holds one L2 strength, so mb2 is tested against the default
bound at that strength rather than against `L21e4`, which would mix the two
factors. The tests are two-sided Mann-Whitney U tests, Holm-adjusted jointly
over all valid tests of one figure (`_reference_column_tests` in
`figures/paper_plot.py`); a test needs at least two networks in each group.
Each tested column carries the mark of its test, as in the other cohort
figures: one to three asterisks for Holm p < 0.05, 0.01, 0.001 and a small
grey "n.s." otherwise, over the column's highest network (in a row above
each panel in the task-grid figures). The reference column and a column
that was not tested carry no mark; a legend names the test and the
reference, and the exact raw and Holm p values are printed when the figure
is generated. The activation-diagnosis figures of the same mode
(`l2e4_activation_embedding_offset`, `l2e4_activation_modulation_dimension`,
`l2e4_activation_modulation_common_accuracy`) are not accuracy comparisons
and are not tested.

The figures share one drawing style, that of the per-run `_seeds` figures
(`_plot_accuracy_columns`): a 2.6 in tall panel as wide as its columns need,
at most five round y ticks, one blue dot per network and the summary over
networks in the statistics red of [SCHEME.md](../SCHEME.md). A sweep (L2,
width, retention) joins the medians of its columns; the activation figure,
whose columns are categories, draws each column's median dash and
interquartile bar and keeps one color per activation, as do its two
diagnosis companions. The one-panel figures of `_plot_accuracy_columns`
(L2, activation, the two widths, retention) and the per-task grids also give
each column its mean as a near-black diamond with the sample SD as an error
bar, just right of the dots, so the red median and the mean can be read
side by side.
`l2e4_activation_modulation_common_accuracy` is drawn like the other
association scatters (outline-free dots in the activation colors, a square
box, the legend above the panel). `modulation_bound_accuracy` is one panel with the two
bounds side by side at every L2 strength (mb1 filled, mb2 hollow) and the
mark of each test on a bracket over the pair. The per-task figures are a
grid of fifteen small panels, three per row, in the order of the task
table. Before 2026-11-13 these figures drew the across-network means as
connected black diamonds on larger panels with a tick every 5 %, the bound
figure had one panel per L2 and the task figures stacked their fifteen
panels in one column.

The width sweeps select their runs by configuration, and since 2026-11-13
they also require the default M bounds: before that the five `L21e4mb2`
runs were pooled into the 300-wide group (30 networks instead of 25).

## Accuracy across modulation retentions

The `acc_plot` figures `retention_accuracy` and `retention_task_accuracy`
compare lambda = 0.9999, 0.999, 0.99, 0.98, 0.95 and 0.9 at L2 = 1e-4, with the
same default network architecture and modulation bounds. The feature tags are
`L21e4lam0.9999`, `L21e4lam0.999`, `L21e4`, `L21e4lam0.98`, `L21e4lam0.95` and
`L21e4lam0.9`; their saved `m_time_scale` values must be 400000, 40000, 4000,
2000, 800 and 400 ms respectively. The lambda = 0.999 and 0.98 cohorts were
added on 2026-11-10 and 0.9999 (`--m-time-scale 400000`) on 2026-11-11.
At 0.999 (the script's `M_TIME_SCALE = 40000`) the time constant is 40 s, far
longer than a trial, so M barely decays within one (M is reset at every trial
onset, so nothing carries between trials at any retention). Train another
retention without editing the script, e.g.
`python multiple_task/multiple_task.py --m-time-scale 2000` for 0.98. Until a
cohort's runs are trained and evaluated its column stays empty. Each dot is
one network and the red line joins the across-network medians; the tick
labels are the lambda values (their time scales are listed above). These
figures use performance results, not clustering or lesion caches.

```bash
# Add the newly trained cohort without replacing the existing evaluations.
python multiple_task/multiple_task_performance.py --feature L21e4lam0.999 --feature L21e4lam0.98 --merge
python figures/paper_plot.py --only retention_accuracy
python figures/paper_plot.py --only retention_task_accuracy
```

Both figures report two-sided Mann-Whitney U tests of each retention against
the main lambda = 0.99 cohort (five pairs with six cohorts; before
2026-11-10 every pair of the three cohorts then present was tested). Holm
correction covers the overall comparisons in the first figure, and all 15
tasks times the pairs (up to 75 comparisons) jointly in the second. A pair needs at least two networks in each group; unavailable
comparisons have no p value and do not enter the correction. Missing cohorts
keep their axis position and produce a message with the evaluation command.

The marks follow [the convention of every accuracy figure](#accuracy-figures-tests-against-the-default-setting)
(before 2026-11-13 the p values were written on the two figures instead).

## Figures

The `lesion` figure mode contains `lesion_heatmap`, `lesion_cluster_sizes`,
`cluster_corr_vs_lesion`, `cluster_corr_vs_lesion_weighted`,
`cluster_corr_vs_lesion_seeds`, `om_vs_lesion`, `om_vs_lesion_seeds`,
`plasticity_share`, `plasticity_share_cohort_violin`,
`causal_vs_activity_tasks`,
`causal_vs_activity_seeds`, `task_specificity`, `task_specificity_seeds`,
`task_specificity_threshold_seeds`, `task_specificity_size`,
`task_specificity_pool_seeds`, `task_sharing_cohort`,
`state_space_vs_lesion`, `state_space_vs_lesion_seeds`,
`state_space_vs_lesion_families`, `state_space_vs_lesion_families_seeds` and
`state_space_vs_lesion_family_seeds`.
All but the nine `_seeds`, the size and three `_cohort` figures read one
run, `LESION_ANAME`,
independently of `ANAME`, which selects the `multiple_tasks` clustering
figures; the exception is `om_vs_lesion`, which reads `ANAME` (see the OM
scatter below). These figures no longer run under `multiple_tasks`; use
`python figures/paper_plot.py multiple_tasks lesion` for both groups. Existing figure
names and `--only FIGURE` commands are unchanged. The historical `leison`
spelling is now `lesion` in commands, module names, output names and cache keys:
use `multiple_task/lesion.py`, `multiple_task/lesion_plot.py` and the `lesion`
paper mode. Existing caches have NOT been migrated or renamed. Readers accept
their legacy filenames, keys and condition labels through a read-only adapter;
correctly spelled files take precedence when both exist. Conflicting new/old
keys in one cache raise an error rather than losing data. New results use only
the corrected spelling; this compatibility does not bypass schema validation.
Mode runs retain the existing
behavior of clearing top-level `paper_plot/` outputs (`paper_plot_<pool>/`
under `--control-pool`, see [Activity-matched controls](#activity-matched-controls));
`--only` preserves other figures.

The `multiple_tasks` and `lesion` figures read one L2 cohort of the
`everything_seed*_L2<l2>+hidden300+batch128+angle` runs, selected by
`MULTITASK_L2` in `paper_plot.py` (currently `1e3`) or per invocation with
`python figures/paper_plot.py --multitask-l2 1e4 multiple_tasks lesion`. The aggregate
figures (overmembership, `cluster_corr_vs_lesion_seeds`, `om_vs_lesion_seeds`,
`plasticity_share_cohort_violin`,
`causal_vs_activity_seeds`, `task_specificity_seeds`,
`task_specificity_threshold_seeds`, `task_specificity_size`,
`task_specificity_pool_seeds`, `task_sharing_cohort`,
`state_space_vs_lesion_seeds`, `state_space_vs_lesion_families_seeds`,
`state_space_vs_lesion_family_seeds`)
summarize every cached seed of that cohort. The single-run figures use the
cohort's designated seed from `MULTITASK_DESIGNATED_SEEDS` (749 for clustering
and 921 for lesions in the 1e4 cohort; none for 1e3) when its cache directory
holds pickles; otherwise `paper_plot` picks the cached seed covering the most of
the category's cache directories, lowest seed on ties, and prints the switch.
Each category resolves once, so its figures never mix runs, and a run is never
taken from another cohort; this also applies to `--only` and to imported plotting
functions. Empty directories and pickles of other runs do not count as cached
data. Missing or incompatible individual caches still use the existing per-figure
checks rather than switching seeds per figure.

`python figures/paper_plot.py --only cluster_corr_vs_lesion` draws the four input/hidden
scatter figures and three modulation figures:
`multitask_cluster_corr_vs_lesion_modulation_norm_freeze_M_n.png`,
`multitask_cluster_corr_vs_lesion_modulation_var_weighted_unnorm_freeze_M_n.png`
and `multitask_cluster_corr_vs_lesion_modulation_abs_weighted_unnorm_freeze_M_n.png`
(`--only cluster_corr_vs_lesion_weighted` adds the Var(W·M) one,
`..._modulation_weighted_unnorm_freeze_M_n.png`). Every modulation figure uses
cached **freeze_M** effects, the intervention of the lesion heatmap
(`paper_plot.LESION_HEATMAP_MODE`; changing it there moves all of them), and
reads only that mode's cache, never zero_W as a substitute. The
two unnormalized figures use var-weighted (W·Var(M)) and abs-weighted
(|W|·Var(M)) modulation and drop the unresponsive cluster. A run whose
`lesion_plot.py` caches predate the abs-weighted variant skips only that
figure. Each figure is scale-free
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
fallback to other variants or the other lesion mode. The caches also hold an L1
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

`python figures/paper_plot.py --only cluster_corr_vs_lesion_weighted` draws the separate
**Var(WM)** comparison from `LESION_ANAME`, exporting only
`multitask_cluster_corr_vs_lesion_modulation_weighted_unnorm_freeze_M_n.png`.
The upstream `modulation_all_weighted_unnormalized`
features take variance **after** multiplying modulation by static W, giving
`W**2 * Var(M)`, unlike `var_weighted`, which gives `W * Var(M)`.
The plot reads only the `weighted-unnormalized_freeze-M` scatter cache, drops
the last unresponsive cluster, and reuses the saved profile correlations,
z-scored lesion dissimilarities and permutation statistics without refitting.
No clustering or lesion experiment is rerun. Use an allocated compute node for
rendering; this `--only` command preserves all other figures.

The `lesion_heatmap` compares hidden neuron lesions with var-weighted,
unnormalized modulation **freeze_M** lesions (the selected synapses keep W and
their modulation is held at its initial value, M(t) <- M(0);
`paper_plot.LESION_HEATMAP_MODE`), as two task-by-cluster panels; the input
neuron effects saved in the same cache are not drawn, and the shared color
scale covers only the two drawn panels. Its `lesion_cluster_sizes` companion
reads the memberships of the same freeze_M entry; zero_W and freeze_M lesion
the same synapse clusters, so the sizes do not depend on the mode.
`lesion_plot.py` exports both zero_W and freeze_M effects with explicit mode
metadata; missing freeze_M results are never replaced by zero_W in these two
figures. Every modulation tuning-vs-lesion scatter, the modulation side of
causal-vs-activity, the synapse clusters of task specificity and task sharing
and the synapse panels of `state_space_vs_lesion` (below) follow the same
mode, and so do the OM scatter and its cohort view (`paper_plot.OM_LESION_MODE`,
equal to `LESION_HEATMAP_MODE`; they read **zero_W**, selected synaptic weights
set to zero, until 2026-11-08). Plasticity share compares the two modes by
definition.
`lesion_plot.py` records freeze_M as `primary_modulation_mode` in the
normalized-effect cache.

The `overmembership_examples_supported_<r>` panels (`--only overmembership_examples`,
run `ANAME`) draw the synapse clusters with the strongest block enrichment, by
default the two with the highest OM over blocks expecting at least five
synapses, as input × hidden block maps. These are most-enriched examples,
not representative clusters, and the selection does not perform a block-level
significance test. Each drawn block is a cell colored by
its OM = observed/expected − 1 on a diverging scale centered on chance
(`paper_plot._om_block_cmap_norm`: blue below chance down to −1, white at 0,
red above, up to the examples' peak OM over well-supported blocks; higher OM
saturates). Only the well-supported blocks are drawn, those expecting at
least five synapses, the ranking threshold; the map has a light grey
background, so grey is a block that is not assessed and white a block at
chance, and an input or hidden cluster without a well-supported block is
dropped from the axes. The deciding block (the peak OM over well-supported
blocks) is ringed in black, the marker the OM-vs-lesion scatter gives the
same block. The input and hidden clusters are sorted by the cluster's
observed/expected over their counted blocks (those expecting at least one
synapse, so the thin blocks the map leaves out still count), most enriched
first, and the bars above and to the right of the map show that marginal
ratio in log2 coordinates, with ticks labelled as observed/expected ratios
(chance 1). One symmetric axis covers every example and spans at least one
doubling. The shared colorbar `overmembership_examples_colorbar` marks
chance; the saved message names the axis order, the deciding block and the
runners-up. In the L21e4 run the examples are
diffuse, not concentrated: C3's three most enriched well-supported blocks hold
4% of its 3244 synapses (1% expected) and its ten fullest blocks 16%, against
33% for the ten largest blocks, because it avoids the largest input and hidden
clusters (0.4 to 0.6 of expected) and spreads at 1.5 to 2.4 times expected over
the small ones. The sorted axes and marginal bars show this profile. For C3
the map keeps 120 of the 288 counted blocks, holding 87% of its expected
synapses (C11: 86 of 236, 82%). Earlier versions were removed: unsorted dots
with area proportional to the observed count, which mostly showed the neuron
clusters' sizes; discs under rings sized by the expected count (2026-11-08),
whose small enriched blocks were specks; and, until 2026-11-10,
`overmembership_examples_dots_<r>`, which also drew the thin blocks as
smaller squares, whose saturated colors drew the eye to the least reliable
ratios.

The same call also saves `overmembership_examples_circles_<r>`, a reading aid
for what over-membership is rather than a map of the cluster: a few
well-supported blocks, each as two circles. The black ring is the block's
expected synapse count, drawn at one size in every cell and both figures;
the grey disc under it has observed/expected times the ring's area, so a disc
beyond its ring is enrichment, a disc inside it depletion and a ring alone a
block without a synapse. There is no color scale and there are no marginal
bars. The cells are those of the input and hidden clusters of the example's
three most enriched and three most depleted well-supported blocks
(`OVERMEMBERSHIP_CIRCLE_N_BLOCKS`, `paper_plot._om_circle_clusters`), in the
block map's axis order; a cell expecting fewer than five synapses stays empty,
and the deciding block is framed. In the L21e4 run that is a 5 × 5 grid for C3
(21 blocks drawn, holding 19% of its synapses, ratios 0 to 3.86) and a 3 × 5
grid for C11, whose three most enriched blocks share one input cluster (12
drawn, 16%, ratios 0.01 to 3.75). The blocks are the most extreme ones by
construction, so the figure must not be read as where the cluster's synapses
are; the saved message states the selection and the share. The key is
`overmembership_examples_circles_legend` (saved with the legends).

`overmembership_examples_overview` shows where the examples stand among the
synapse clusters: one column per cluster with a well-supported block, at its
synapse count (log axis), for the distribution of its OM over those blocks
(every block one sample, unweighted). It follows the per-run `_seeds`
figures' convention: the data value is a point in the statistics blue, here
the cluster's peak OM that ranks the examples, and the summary is the red
overlay, a dash at the median with a bar over the interquartile range; a thin
grey line spans the full range. The examples' peaks are ringed in black, as
the OM-vs-lesion scatter marks its example block, and named. The
examples are the top of that ranking, not typical clusters, and a peak is
the extreme of its own cluster. In the L21e4 run (seed 749, 19 assessable
clusters) C3 and C11 lead with peaks of 2.86 and 2.75, yet their median block
is close to chance (0.36 and 0.14; interquartile ranges −0.28 to 0.97 and
−0.28 to 0.62). C17 and C8 follow at 2.54 and 2.34 with medians below chance;
the two largest clusters, C14 and C13, hold 63% of the retained synapses and
peak near 1 (1.09 and 0.91); and the twelve smallest clusters (109 to 642
synapses) have no well-supported block above chance, their median at or near
−1, because their synapses lie in blocks expecting fewer than five. Three
cautions: the range widens with the number of blocks (120 for C3 against 2
to 28 for the small clusters), so the peak alone favors clusters large enough
to fill many assessable blocks; the expected-weighted mean OM of a cluster
over all its blocks is zero by construction, so the medians say where the
well-supported blocks sit, not whether a cluster is enriched overall; and
nothing here is tested. A figure caption should call the examples the most
enriched ones.

The OM-versus-lesion-profile figure has two panels with the same OM x axis.
The left panel retains task-profile L1/T distance, which includes differences
in overall damage. The right panel uses Pearson correlation between the two
random-minus-lesion task vectors, subtracting each vector's task mean and
normalizing its scale. Its y axis measures similarity of task selectivity,
not absolute damage. Both panels test OM association using Spearman rho and
whole modulation-cluster-footprint permutations: one-sided negative for L1,
one-sided positive for Pearson similarity. The Pearson on the right y axis
is distinct from the Spearman association reported above that panel.
`lesion_plot.py` saves the scatter, `association`
(rho, matching permutation p, null statistics and test metadata), and
`binned_medians` in both single-mode and combined `schema_version=2` caches.
The additive `profile_correlation` entry stores aligned Pearson values, its
own association and bin medians, and the excluded-point count. A centered
task vector with norm <= 1e-12 is treated as constant: its Pearson value is
NaN, not zero. Constant modulation profiles and constant combined profiles
are removed before permutation, so the valid sample set is fixed across
permutations. Both the single-run and cohort figures require this entry;
older v2 caches contain only the L1 result and must be refreshed. Refresh via
`python multiple_task/lesion_plot.py --seed 749 --feature L21e4 --control-pool all`, then
`python figures/paper_plot.py --only om_vs_lesion --multitask-l2 1e4 --control-pool all`
for the `_n` figure. This is post-processing of saved results, with no network
forwards. Pearson similarity alone does not establish a shared causal mechanism.
`paper_plot.py` takes the freeze_M entry (`OM_LESION_MODE`; the zero_W entry
until 2026-11-08, when seed 749's L1 association rho moved from −0.59 under zero_W to −0.41
under freeze_M, both with permutation p ≤ 0.002). **Run selection differs from the other
lesion figures:** the single-run scatter is drawn from `ANAME`, the clustering
example (seed 749 under `--multitask-l2 1e4`), the run of the
`overmembership_examples_supported_*` panels, so the example clusters are among its
points; other single-run lesion figures use `LESION_ANAME` (designated seed
921 at L2 = 1e-4). Both selections can fall back to another cached run when
the designated run is unavailable, as described above.
`paper_plot._om_scatter_run` holds the choice and the saved message names the
run. `om_vs_lesion_seeds` pools eligible sibling runs in the cohort. Because of the
shared run, the deciding block of `overmembership_examples_supported_1` (its most
enriched supported input × hidden block, from `_om_example_selection`) is
drawn on both scatters where defined, in the color the
examples' block maps give its OM, with a black ring ("Example block"), the
ring the block maps also draw on it. Each single-mode entry and each mode of
the combined cache saves `point_ids`, an N × 3 integer array in the same
order as the scatter: `(mod cluster, input cluster, hidden cluster)`, all
1-based. Matching uses this identity, so repeated OM values do not cause
ambiguity. The loader validates shape, positive integer labels and uniqueness;
the matched ratio must also agree with the current example before it is
highlighted. Old caches without IDs can still supply statistics, but their
highlight is skipped with a rerun message. Regenerate IDs with `lesion_plot.py`;
they are never inferred by matching OM values.
The block colors and scatter x coordinates display block over-membership as
`OM = observed / expected - 1`, with chance at zero. The cache retains raw
ratios for reconstructing observed counts; both scatter points and saved bin
x coordinates are shifted by one only when plotting. This translation leaves
the selected examples, Spearman correlation and permutation p values unchanged.
The shared block colorbar marks zero, the diverging scale's white center.
Up to five quantile bins provide median OM/comparison points and counts for
each panel; tied OM values stay together. The median connector is descriptive
and is not forced to increase or decrease.
`paper_plot.py` only reads these values, draws pale scatter points and the saved
median connector, starts the L1 axis at zero, and shows Pearson values from
−1 to 1 with a zero reference line. Old Pearson-regression caches are skipped;
they do not contain the new task-vector Pearson statistic.
`python figures/paper_plot.py --only om_vs_lesion_seeds` draws the
cohort view, `multitask_om_vs_lesion_seeds_n.png`, with two panels: OM versus
L1/T distance on the left and OM versus task-vector Pearson similarity on the
right. Each point is one sibling network's saved freeze_M **Spearman rho**
between OM and the chosen metric, not a task-vector Pearson r. Filled points
have a saved one-sided whole-footprint permutation p below 0.05; dashes mark
medians. The cohort Wilcoxon signed-rank tests assess rho < 0 on the left and
rho > 0 on the right. Both panels use the same networks, offsets and y scale.
Runs lacking either finite rho/permutation p are excluded from both panels
with a note; fewer than two complete runs skips the figure. Refresh every
completed run in the selected cohort (the `lesion_plot` stage of
`run_pipeline_analysis.py` does this for both pools), then plot:

```bash
python -u multiple_task/lesion_plot.py --seed all --feature L21e4 --control-pool all
python figures/paper_plot.py --only om_vs_lesion_seeds --multitask-l2 1e4 --control-pool all
```

This regenerates lesion post-processing caches and figures from saved raw
lesions and clustering. It does not rerun training, clustering, or lesion
forward passes. The unrelated `_seeds` figures retain their own tests.

`python figures/paper_plot.py --only plasticity_share` draws
`multitask_plasticity_share_n.png` from `plasticity_share_var-weighted-unnormalized_<aname>.pkl`.
For every (task, synapse cluster) cell whose zero_W effect is significant
(one-sided prediction t against the stored random-control repeats, BH-FDR
q = 0.05; see [Activity-matched controls](#activity-matched-controls)),
`lesion_plot.py` saves the plasticity share = freeze_M effect / zero_W effect:
1 means freezing the cluster's plasticity costs the task as much as removing
its weights, 0 means the static wiring suffices. The figure shows one column
per task (no-working-memory tasks fdgo/fdanti/reactgo/reactanti first), the
saved cells as pale grey points and the saved task median as a red dash;
values outside the fixed y-limits appear as hollow grey triangles at the edge.
Task labels slant at 45° (`_PLASTICITY_SHARE_TICK_ROTATION`), low enough to
keep the long names short and high enough that their colored boxes do not
overlap. The figure shows
the values only: the one-sided task-level Mann-Whitney U test that
`lesion_plot.py` also saves (memory tasks exceeding no-memory tasks) is not
reported, and the former per-run `plasticity_share_seeds` figure was removed.
`--only plasticity_share_cohort_violin` draws
`multitask_plasticity_share_cohort_violin_n.png`, the same layout with every
sibling run of `LESION_ANAME` (same feature, any seed) pooled: the saved
significant cells of all runs pooled per task as a Gaussian kernel density
(scipy's Scott bandwidth, cells within the y-limits only, every column scaled
to the same peak width so shapes compare rather than counts), each run's
saved task median as a short near-black dash (so it stays apart from the grey
density), and the median of the pooled cells as the red dash; the legend
title gives the run count. Cells beyond the limits are triangles at the edge,
and a task with fewer than three cells within them shows its points. A run
whose task list differs from the others is left out with a note, and the
figure is skipped below two runs. A point overlay of the same pooled cells
(`plasticity_share_cohort`) was removed on 2026-11-08 in favor of the density
version. The share compares two
interventions and is not an exact additive partition of static and plastic
contributions.

The share is not a measure of working memory, and the text has to say so. On
the 25 L21e4 runs the pooled medians of the four no-working-memory tasks are
DelayPro 0.55, DelayAnti 0.65, ReactPro 0.76 and ReactAnti 0.79 (0.69 over
their 1,331 cells): freezing a cluster's modulation costs these tasks more
than half of what removing its weights costs, although their stimulus stays
on and nothing has to be held across a delay. The working-memory tasks pool
at 0.83 (3,298 cells), but MemoryPro and MemoryAnti, the delayed counterparts
of DelayPro and DelayAnti, sit at 0.64 and 0.49, below the reaction tasks,
and the matching family is the highest (0.90-0.97); so the family contrast
is one of degree and does not follow the delay. The reason is in the
intervention: freeze_M holds M at M(0) for the whole trial, which removes
all of the cluster's within-trial modulation, written from the first step on
(the Hebbian term runs during the context and stimulus periods, before any
delay), not only the part that bridges a delay. The share therefore says how
much of a cluster's contribution to a task runs through its modulation rather
than its weights; which tasks need that modulation as a memory is the
question of a time-resolved freeze, which has not been run.

`python figures/paper_plot.py --only causal_vs_activity_tasks` draws
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
tuning-vs-lesion scatters show bin medians instead, see above).

A third side, `..._tasks_modulation_n.png` from
`causal_vs_activity_tasksim_modulation_<aname>.pkl`, asks the same of the
synapses: x the correlation of the two tasks' period-averaged var-weighted
modulation features over all synapses, taken by magnitude (|W| Var(M); the
signed W Var(M) carries the sign of W in every task alike and would inflate
every pair), y the correlation of their **freeze_M** lesion-effect profiles
over the var-weighted unnormalized synapse clusters, the intervention of the
lesion heatmap. The unresponsive class is left out of y: it holds most
synapses and impairs every task, which would raise every pair together
(`state_space_vs_lesion.py` drops it for the same reason). A run gets this
cache only from a `lesion_plot.py` run made after the side was added; until
then the figure is skipped for it.

`--only causal_vs_activity_seeds` reads the same
caches for every sibling run of `LESION_ANAME` (same feature, any seed) and
plots one point per run in a hidden, an input and a modulation column (fixed
random jitter; the runs have no order), filled when the saved permutation p is
below 0.05 and a hollow grey ring otherwise, with each column's median (dash)
and interquartile range (bar).
Hidden and input share their lesion basis, so input is hidden's control: a
thin line joins each run's pair and a bracket gives the paired Wilcoxon
signed-rank p of hidden minus input, computed from the saved per-run rhos.
Modulation pairs other features with other lesions, so it stands apart behind
a dotted divider and is not joined. A run needs both neuron sides; the
modulation column holds the runs that have its cache. The panel has the
figure size and the square axes box (about 1.49 in) of the single-run
scatters, so it sits beside `..._tasks_hidden_n.png` at one scale; it carries
no legend or run count, and its legend (filled and hollow markers, median and
IQR) is saved as `multitask_causal_vs_activity_seeds_legend_n.png`. The
bracket's test, the paired Wilcoxon signed-rank, belongs in the caption.
A response-period-only
activity vector is not used: it also makes the input side correlate. Both
figures need `schema_version=2` caches written
by the current `lesion_plot.py`.

`python figures/paper_plot.py --only task_specificity` draws
`multitask_task_specificity_counts_n.png` from `task_specificity_<aname>.pkl`
for input and hidden neuron clusters (unnormalized clustering) and
var-weighted synapse clusters, the unresponsive class excluded. A cluster
"impairs" a task when its lesion effect is significant (one-sided
prediction t against the control repeats, BH-FDR q = 0.05; the
unnormalized input and hidden clusters are one family, as in the
causal-dependency map). Synapse clusters are tested with their **freeze_M**
lesions (`lesion_plot.TASK_SPECIFICITY_MOD_MODE`, the lesion heatmap's mode),
against the freeze_M control repeats; caches of an earlier schema
(`schema_version=1`, zero_W; 2, without the threshold counts below; 3,
without the control pool and the per-pool counts) are skipped with a note,
so rerun `lesion_plot.py`. The figure has one row per
cluster type, sharing both axes:
every cluster is a dot stacked over the number of tasks it impairs (a dot
histogram counted in clusters), the red line is the median, and the gray
histogram behind the dots is the number of clusters the saved mixing null
expects at each count. That null shuffles cluster identity within each task,
keeping each task's number of impaired clusters, so one cluster's count is
Poisson-binomial; `lesion_plot.py` saves its exact distribution as
`null_count_pmf`, and the figure draws it times the number of clusters. Dots
piling up at both ends beyond the gray hump mean specialized and hub-like
clusters rather than a uniform mix; the right-hand row title gives the saved
one-sided permutation p of the observed count variance exceeding that null.
Nothing is recomputed in `paper_plot.py`.
`python figures/paper_plot.py --only task_specificity_seeds` draws the cohort,
`multitask_task_specificity_seeds_n.png`: one column per cluster type, each
sibling run of `LESION_ANAME` a grey point at the median of its clusters'
counts (filled when the run's dispersion p is below 0.05, runs tied on a value
side by side, each run's points joined), the median and interquartile range
over runs in red, and over each pair of neighboring columns a bracket with the
two-sided paired Wilcoxon signed-rank p of the runs' medians. It needs two
runs, and leaves out runs whose cluster types differ.

The significance criterion depends on the random control's spread, and that
spread grows with cluster size (bigger size-matched random lesions do more
variable damage; Spearman rho of size with the control SD is about 0.95 in
every type), so small clusters reach significance more easily than large
ones. `lesion_plot.py` therefore also saves, per cluster type, the number of
tasks each cluster impairs by a fixed effect threshold (its
random-minus-lesion effect above `TASK_SPECIFICITY_EFFECT_THRESHOLDS`, 2, 5,
10 and 20 accuracy points, a criterion the spread does not enter), the
cluster sizes (neurons, or synapses) and the Spearman correlation of size
with every count (`size_correlation`; NaN below three clusters or for a
constant count). `python figures/paper_plot.py --only
task_specificity_threshold_seeds` draws
`multitask_task_specificity_threshold_seeds_n.png`, the seeds figure counted
at `paper_plot._TASK_SPECIFICITY_FIGURE_THRESHOLD` (5 points) with every
point filled, its saved message giving the medians at every saved threshold;
`--only task_specificity_size` draws `multitask_task_specificity_size_n.png`,
per cluster type two columns of per-run correlations, under the significance
rule and under the threshold rule, a run's two joined, with the median and
interquartile range over runs, 0 marked and the two-sided Wilcoxon
signed-rank p of the runs' correlations against zero above each column
(computed in `paper_plot.py`); runs with an undefined correlation are left
out with a note. On the 25 L21e4 runs (development probe of 2026-10-29; the
two figures' saved messages give the same cohort medians) the median
correlation of size with the significance count is −0.57 (input), −0.50
(hidden) and −0.28 (synapse), against −0.20, −0.14 and +0.03 with the
5-point threshold, and the medians of the threshold counts are 6, 4 and 9.5
tasks against 2, 2.5 and 11.5 by significance: the synapse clusters still
impair the most tasks, by less, and the spread of the significance counts
partly reflects cluster size. Half of the hidden clusters hold three neurons
or fewer (median 10 of 20 per run; the unresponsive hidden class holds about
159 of 300 neurons), so the hidden row of the counts figure is largely a
statement about single neurons. The size dependence comes from the
controls themselves, which [Activity-matched controls](#activity-matched-controls)
redraws from the responsive units; the pool figure there compares the
counts under both.

The same cache (so the same schema requirement) holds, per cluster type, the
task-pair Jaccard overlap of impaired clusters grouped by the component the
two tasks differ in (`TASK_PAIR_RELATIONS` in `lesion_plot.py`) with one-sided
task-label permutation tests. `python figures/paper_plot.py --only task_sharing_cohort`
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
and lesion caches (`multiple_tasks_norm/<aname>/`, `lesion.py`'s controls),
it relates the Euclidean distance between the two tasks' full-dimensional
context-end centroids (hidden state and W ⊙ M, the saved distance matrices)
to two measures of how alike their lesion-effect profiles (random-minus-lesion
accuracy, one value per cluster) are, the two the OM-vs-lesion scatter uses:
the **Pearson correlation** of the two profiles, which removes each profile's
mean and scale, and their **L1/C distance**, the mean over clusters of the
absolute effect difference in accuracy fraction, which keeps the effect
magnitudes (added 2026-11-09, `schema_version=3`). Both are computed over the
same clusters: for neuron clusters all unnormalized input and hidden neuron
clusters, the `lesion_unnorm` entry of `normalized_lesion_effects_<aname>.pkl`,
whose correlation matrix is the saved `causal_similarity` of
`causal_vs_activity_tasksim_hidden_<aname>.pkl` (the causal-vs-activity
matrix; the script reads it and checks that the effects reproduce it); for
synapse clusters the var-weighted **freeze_M** effects (`SYNAPSE_LESION_MODE`,
the lesion heatmap's intervention) of the same cache over the synapse
clusters without the unresponsive class, whose label is read from the lesion
pickle in `multiple_tasks_perf/` (the class holds most synapses, so its
size-matched random control collapses the network and its effect is strongly
negative for every task). `paper_plot.py` skips results of an earlier schema
or whose synapse measures were computed with another mode (results written
before the switch from zero_W); rerun the script to refresh them. Each
(representation, cluster type, measure) gets a two-sided Spearman task-label
permutation (Mantel) test (10,000 permutations, seed 0;
`core.state_space_geometry.mantel_spearman_permutation`) and a descriptive
OLS guide line for the figure; tasks whose context-end states lie closer
depending on more similar clusters shows as a negative rho for the
similarity and a positive rho for the distance. The tasks form three
families (`TASK_FAMILIES`: the go, decision-making and matching families of
the README) whose members lie together in state space and share lesion
profiles, so the full test is significant whenever the families separate
on both axes. Each test is therefore also run with the task labels permuted
within families only (`mantel_spearman_restricted_permutation`, added
2026-11-09, keys `restricted_tests` and
`profile_distance_restricted_tests`): family membership is then part of the
null, and that p is the p of the association the arrangement of the tasks
within their families adds; its `within` part is the rho over the 31
same-family pairs alone, with its own p from the same permutations, and
its `per_family` part (`schema_version=5`) repeats that for each family on
its own pairs, with only that family's labels shuffled (`family_tests`;
the four matching tasks allow 24 distinct permutations, recorded as
`n_distinct_permutations`, so that p is coarse). In the L2 = 1e-4 cohort
the full tests are significant in most runs but the within-family ones in
2 to 5 of 19, and the same-family rhos sit near zero: the link lives at
the family level. Per family, the decision-making tasks are the exception
(their closer pairs do have more similar profiles), the go family's lesion
similarity follows pro/anti while its context-end distances do not, and the
matching tasks lie far apart yet share profiles. Nothing is recomputed from
network activity, so it runs in seconds; the results go to
`state_space/state_space_vs_lesion_results.pkl` (one entry per run with the
pair coordinates of both measures, the full and restricted tests with their
null arrays, the guide lines, the task families and the L2), which the
state-space batch run clears like every top-level file there, so rerun it
after `state_space_structure.py` or `lesion_plot.py`.
`python figures/paper_plot.py --only state_space_vs_lesion` draws
`multitask_state_space_vs_lesion_n.png` for `LESION_ANAME`: a 2 × 4 grid
with one row per representation (hidden-state and W ⊙ M centroid distance)
and, per cluster type (neuron clusters, synapse clusters, titled over the
pair), the OM scatter's pair of panels, the L1/C distance on the left (y
axis from zero) and the Pearson similarity on the right; one point per task
pair, x the centroid distance, each panel's saved rho and permutation p in
its legend and the script's saved OLS guide line, descriptive only as in the
causal-vs-activity scatter (the statistic is the rank correlation). Before
2026-11-09 the figure was the 2 × 2 Pearson half.
`--only state_space_vs_lesion_seeds` draws
`multitask_state_space_vs_lesion_seeds_n.png`: one row per measure (L1/C
distance above, Pearson similarity below) and one column per cluster type,
every sibling run's rho for the hidden-state and the W ⊙ M distance, filled
when its saved permutation p is below 0.05, the median over runs as a dash
and under each tick the two-sided Wilcoxon signed-rank p of the runs' rhos
against zero, computed in `paper_plot.py` from the saved per-run rhos; it is
skipped below two runs with results.
`--only state_space_vs_lesion_families` draws
`multitask_state_space_vs_lesion_families_n.png`, the same 2 × 4 grid with
the task families visible: the pairs of two tasks of one family in that
family's color (`_SSL_FAMILIES`: go blue, decision-making purple, matching
gold, the categorical `c_vals`) over the pairs of different families in
grey, and in each legend, under the full permutation p, the within-family
permutation p; the color key (family names and pair counts) is saved as
`multitask_state_space_vs_lesion_families_legend_n.png`, so the panels carry
the statistics only and keep the full-pairs grid's size.
`--only state_space_vs_lesion_families_seeds` draws
`multitask_state_space_vs_lesion_families_seeds_n.png`, the cohort grid
for the same-family pairs alone: per run the rho over those 31 pairs,
filled when its within-family permutation p is below 0.05, with the median
dash and the Wilcoxon p as in the full-pairs figure.
`--only state_space_vs_lesion_family_seeds` draws
`multitask_state_space_vs_lesion_family_seeds_n.png`, the same grid per
family: one tick per family, per run the rho over that family's pairs (15,
10 or 6) for the hidden-state distance only (the W ⊙ M distances rank the
pairs alike, rho 0.93), filled when the family's own permutation p is
below 0.05. All three were added 2026-11-09; the full-pairs figures are
unchanged.

On a compute node, regenerate the post-processing cache and then the paper figure:
```bash
python multiple_task/lesion_plot.py --seed 921 --feature L21e4
python figures/paper_plot.py --only om_vs_lesion
```
Use `--seed all` in the first command to update every matching run's OM results.
This reuses completed lesion experiments, but clears and regenerates the selected
runs' post-processing outputs in the control pool's cache directory
(`multiple_tasks_norm/` by default).

## Activity-matched controls

`lesion.py` compares every cluster lesion with size-matched random lesions
drawn from all units of that side or layer. In the historical S1-conditioned
analysis (these numerical values need rechecking), more than half of the hidden
neurons (median 159 of 300 per run) and of the plastic synapses (57%) belong
to the unresponsive class, so those controls are mostly silent units and
barely hurt the network (a random zero_W lesion costs about 2 accuracy
points, a random freeze_M lesion 0.7); the significance test then asks
whether a cluster matters more than a half-silent random set, and the
control spread tracks the cluster size (Spearman rho about 0.95), which is
what the threshold counts above work around.
`multiple_task/lesion_controls.py` scores the single-cluster conditions
again, on a seeded test set (`--test-seed`, default 0; the per-task input
checksums are saved so two pickles can be told apart), with the random
controls drawn from the **responsive** units only (`--pool responsive`, the
default: the units outside the unresponsive class of that neuron variant or
synapse clustering) or, for a comparison on the same trials, from every unit
(`--pool all`). It repeats the controls 30 times per (side or layer, size)
and task (`--repeats`), scores both neuron variants and every synapse
clustering type the lesion pickle holds (`--mod-types` to restrict) in both
lesion modes, and leaves the combined lesions and the pruning sweep to
`lesion.py`. A condition larger than the selected pool (the unresponsive
class itself) draws from every unit, and `condition_pools` records the pool
of each condition. The clusters, condition names and lesion operations are
those of `lesion.py` (shared in `core/lesion_ops.py`) and are checked against
the run's lesion pickle before anything is scored. The output,
`multiple_tasks_perf/<aname>/lesion_controls_<pool>_<aname>.pkl`, mirrors
the lesion pickle's `lesion` / `random_lesion` / `lesion_unnorm` /
`random_lesion_unnorm` / `mod_lesion` records (same shapes and condition
order) and adds `control_pool`, `repeat_num`, `test_seed`, `test_checksums`
and the pools; nothing is written into the lesion pickle, which `lesion.py`
no longer clears from its directory. A run costs about 45,000 forward passes
of 200 trials per synapse type (around 1.5 h on a GPU for one type, five
types in proportion); the script prints the planned count first.

```bash
python multiple_task/lesion_controls.py --seed 921 --feature L21e4
python multiple_task/lesion_controls.py --seed 921 --feature L21e4 --pool all
python multiple_task/lesion_controls.py --seed all --feature L21e4 --skip-completed
python multiple_task/lesion_plot.py --seed all --feature L21e4 --control-pool responsive
```

`lesion_plot.py --control-pool responsive` overlays that pickle on the
lesion pickle in memory (`core/lesion_cache.overlay_lesion_controls`) and
writes its caches to `multiple_tasks_norm_responsive/<aname>/` instead of
`multiple_tasks_norm/` (`lesion_cache.norm_root`), so the historical caches
stay in place: every single-cluster analysis (normalized effects, causal dependency, cluster
correlation, plasticity share, OM profiles, causal-vs-activity, task
specificity and sharing) then reads the activity-matched controls, while the
combined lesions keep `lesion.py`'s: the OM scatters and the OM profile
prediction compare synapse-cluster effects under the chosen pool with
combined effects under `lesion.py`'s (their caches record
`combined_control_pool`), and the interaction map takes its single effects
from `lesion.py`'s pickle so that its three terms share one control.
`state_space_vs_lesion.py` records the pool of the caches it reads
(`control_pool` per run). Its default, `lesion_plot.DEFAULT_CONTROL_POOL`
(`all`), reproduces the historical analyses; `run_pipeline.py` computes the
responsive pool as its third step (`--control-pool`, `--control-repeats`),
and `run_pipeline_analysis.py` (or the pipeline's optional fourth step,
`--with-plots`) plots both pools, `lesion.py`'s into
`multiple_tasks_norm/` and the responsive one into its own directory;
`--resume` lets runs lesioned before the controls step existed get their
controls without a second lesion pass. `paper_plot.py` reads
`multiple_tasks_norm/` by default (`LESION_CONTROL_POOL = "all"`), so its
lesion figures keep showing the historical analyses; `--control-pool
responsive` reads `multiple_tasks_norm_responsive/` and writes to
`paper_plot_responsive/` under the same figure names, so the two renders
can be compared and merged later. The pool figure below works from either
directory, since each pool's task-specificity cache holds the counts under
every pool available when it was written. `state_space_vs_lesion.py` reads
`multiple_tasks_norm/` and records that pool; under another pool its two
figures are skipped with a note. Every
cache `lesion_plot.py` writes records `control_pool`, `control_repeats` and
`control_test_seed`; cohort figures in `paper_plot.py` refuse to mix runs
whose task-specificity caches used different pools. A lesion is significant
against its controls by a one-sided prediction t test (`lesion_plot._control_test`:
effect / (sd · sqrt(1 + 1/R)) on R − 1 degrees of freedom, the control sd
floored at `CONTROL_SD_FLOOR`), BH-FDR at `CONTROL_FDR_Q` = 0.05, under
either pool; it replaced a Gaussian z that took the sample sd of 10 repeats
as known.

The task-specificity cache (schema 4) also saves, per cluster type, the
impaired-task count of every cluster under each control pool the run has
(`pool_counts`: `all`, lesion.py's controls on its own trials; `responsive`;
and `all_same_trials` when the `--pool all` pickle exists), whichever pool
its own statistics used. `python figures/paper_plot.py --only
task_specificity_pool_seeds` draws `multitask_task_specificity_pool_seeds_n.png`:
per cluster type one column per pool present in every run, each run a grey
point at the median of its clusters' counts, a run's columns joined, the
median and interquartile range over runs in red, and over each type the
two-sided paired Wilcoxon signed-rank p of the runs' medians between the
responsive and the all pool, computed in `paper_plot.py`. It is skipped when
the runs hold counts for one pool only.

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
`unresponsive_label` per modulation lesion (`None` when nothing was flagged);
the `lesion_controls_<pool>_{aname}.pkl` beside it (see
[Activity-matched controls](#activity-matched-controls)) carries the same
fields and, per condition, the pool its controls came from.
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

All three weighted variants are computed from Var(M) by
`weight_modulation_variance`; since 2026-10-27 that includes `weighted`, as
W²·Var(M), where the script used to take the variance of a weighted copy of the
M trajectory. The two agree up to rounding (on seed 749: median relative
difference 5e-9, largest absolute difference 3e-8 of the feature's maximum),
so `weighted` caches clustered before and after the change compare.

**Memory.** `multiple_task_analysis.py` holds the recorded M trajectory once,
750 trials x T steps x 300 x 300 float32 (T is the longest evaluation trial,
which varies from run to run: 24 GiB at T = 96, 32 GiB at T = 126), plus the
clustering's working memory. `record_tracked_states` writes every chunk of the
forward pass into one preallocated array. Before 2026-10-27 the script also
held the concatenated chunks and the weighted copy, two to three trajectories
in all, and a 100 GiB job was killed (out of memory, the shell printing
`Killed`) on long trials. The script prints the job's cgroup memory limit and
use at the start and after the forward pass (`core/memory_limit.py`); psutil's
node-wide numbers, printed before, hid the limit.

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
--feature L21e4 --with-plots`), which reclusters, lesions and re-plots all
five variants.
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

## Synapse tuning vs endpoint clusters

`python multiple_task/synapse_variance_decomposition.py` asks how much of a
synapse's task tuning is the cluster identity of its two endpoint neurons.
For every run with `cluster_info_<aname>.pkl` and `cluster_info_mod_<aname>.pkl`
(`--seed N [N ...] --feature TAG` for one or several runs, `--seed all
--feature TAG` for a cohort, no arguments for every run; `--skip-completed`
reuses a saved result newer than the run's clustering pickles, since the
permutations take about five minutes per run), it takes the active synapses of the
var-weighted clustering (the unresponsive synapse class excluded, as the
over-membership analysis does), their task-period features in two versions,
`magnitude` = `log1p(|W|·Var(M))` (primary; the sign of W is a per-synapse
attribute, not tuning: the signed clustering gives sign-pure clusters and the
sign alone accounted for about a fifth of the signed variance in the old
S1-conditioned analysis; this fraction needs rechecking) and `signed` =
`signed_log1p(W·Var(M))` (what the clustering saw), and reports the share of
the total sum of squares explained by least-squares fits of: the presynaptic
(input) neuron's cluster, the postsynaptic (hidden) neuron's cluster, both
additively, the pre × post block (interaction = block − additive), the
data-driven synapse clusters themselves (reference), the joint (block,
synapse cluster) cells (`synapse_beyond_block` = joint − block, the
structure the synapse clusters hold beyond the endpoint identity), and, at
the neuron level, the pre neuron, the post neuron and both additively (the
strongest endpoint model without an interaction). Two label sets are used
per version: `fixed_k20` (the lesion convention, k = 20 plus the
unresponsive class, G = 300 grouping) and `tolerance` (the analysis figures'
tolerance-selected clusterings). Nulls (1,000 permutations, seed 0,
one-sided p): the cluster models permute the cluster labels over the neurons
(same cluster sizes, so the specific clusters are tested against arbitrary
groupings), the synapse model permutes the synapse labels, the beyond-block
model permutes them within each block (which removes the mechanical gain of
the extra cells), and the neuron-level models permute the synapse positions.
Per run the result goes to
`multiple_tasks_analysis/<aname>/synapse_variance_decomposition_<aname>.pkl`
(every R², null array and p, the sign-of-W share and the group counts);
the run prints a per-cohort summary of the medians. Nothing is recomputed
from network activity; a run takes a few minutes on a compute node.


## Sibling-task row-cluster sharing

`multiple_task/multiple_task_diagnosis.py` asks whether two sibling tasks
land in the same row cluster of the task-variance clustering, the rows of the
clustered input, hidden and modulation heatmaps (input/hidden
`*_normalized`, modulation `modulation_all_normalized` with the G=300
grouping). It reads only the saved `cluster_info*` pickles of every clustered
run of one cohort, so it takes a few minutes on a CPU node:

```bash
python multiple_task/multiple_task_diagnosis.py --feature L21e4
# seven sibling pairs and their summary by phase type
python multiple_task/sibling_row_sharing.py --feature L21e4
# untrained control: cluster fresh networks of one run's config (about 80 min
# each on a GPU node; add --skip-completed to resume a killed batch), then
# their row-cluster organization; one set serves every L2 cohort of the
# architecture (see below), so the L21e3 figures draw these networks too
python multiple_task/untrained_clustering.py --feature L21e4 --seed 749 --init-seed 0 1 2 3 4
python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained
# paper figures: rows grouped by period or by task, and the per-task and
# per-period views
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_organization
# mean sharing and sample SD over networks
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_task_seeds
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_period_seeds
```

Three pairs are checked: ReactCategoryPro vs ReactCategoryAnti (dmcgo/dmcnogo),
ContextIntModality1 vs 2 and IntegrationModality1 vs 2, in every trial phase
both tasks have. For each pair, phase and representation the result is the
number of runs whose two rows share a row cluster. Each count is read against
a chance level: the fraction of pairs of rows of that phase from two
different tasks that share a cluster in that run and representation,
averaged over the runs (`_same_phase_chance`), with an exact two-sided
Poisson-binomial test over the runs' own chance levels
(`_poisson_binomial_p`). In the historical S1-conditioned L21e4 results, every
memory-period row of every task falls in one input and one hidden cluster
(chance 100%), so sharing in the memory periods is not specific to the
sibling pair, while the modulation chance stayed between about 7% and 20%.
These numerical observations need confirmation with the trial-time estimator.
The script writes `multiple_tasks_diagnosis/diagnosis_<tag>.{pkl,json}` and
its own figure `same_cluster_diagnoses_<tag>.png` (tag `<feature>` or
`<feature>_seed<seed>`), replacing earlier files of the same tag only, so the
cohorts' results sit side by side. No paper figure draws these results;
they are an intermediate check.

`multiple_task/sibling_row_sharing.py` runs the same per-pair test on seven
sibling pairs in two groups, the pro / anti pairs (DelayPro / DelayAnti,
MemoryPro / MemoryAnti, ReactPro / ReactAnti, ReactMatch2Sample /
ReactNonMatch2Sample, ReactCategoryPro / ReactCategoryAnti) and the modality
pairs (ContextIntModality1 / 2, IntegrationModality1 / 2), reusing the
diagnosis's loading, chance levels and test. It then pools each group by
phase type, Stimulus (stim1, stim2), Memory (delay1, delay2) and Response
(go1): per run, the number of the group's (pair, phase) cells of that type
whose two rows share a cluster minus the number chance predicts, tested
over the runs with a two-sided Wilcoxon signed-rank test (the cells of one
run share one clustering, so their per-cell tests are not pooled). A pooled
cell whose mean chance is at least 95% (input and hidden in the memory
periods) is reported but not tested. The per-pair and phase-type summaries
are legacy diagnostics; their signed-rank summaries are not used for the
paper's same-task/same-period
stars. Those paper analyses use the restricted permutation tests below.
The script writes
`multiple_tasks_diagnosis/sibling_row_sharing_<tag>.{pkl,json}`, the
per-pair figure `sibling_row_sharing_pairs_<tag>.png`, the phase-type summary
`sibling_row_sharing_summary_<tag>.png` and the same-task diagnostic
`sibling_row_sharing_same_task_<tag>.png`. The last diagnostic reuses the
paper analysis's permutation statistics, including the corrected q values.

The organization check (`summarize_organization`) puts period and task on
one scale with every task's rows: per run and representation, the fold
enrichment of two rows of one period from different tasks, and of two rows
of one task from different periods, sharing a row cluster over any two rows
sharing one (the chance the run's cluster sizes give, so the
representations, with their different numbers of row clusters, compare),
tested against 1 over the runs and, as task minus period enrichment, between
representations (paired Wilcoxon). Its baseline is the sharing fraction
over all row pairs, distinct from the restricted nulls below.

The same-task check runs over every task (`same_task_all`): all unordered
pairs of its available phase rows enter. ReactPro and ReactAnti have only
one clustered phase and therefore no same-task pairs. Its mirror,
`same_period_all`, includes all unordered task pairs with a row in the
phase, including single-phase tasks. The three representations must have
identical task×phase row sets within each network (row order may differ);
missing representations or mismatched rows raise an error. Missing phases
remain missing. No pair is removed because of its sharing or chance level.

Both checks use a restricted row-label permutation null:

- Same task: independently shuffle cluster labels among tasks within each
  phase of each network. This preserves each phase's cluster-label counts
  while testing a task's alignment across phases.
- Same period: independently shuffle labels among available phases within
  each task of each network. This preserves each task's cluster-label
  counts while testing alignment across tasks by phase.

These are conditional exchangeability nulls on the existing row slots,
not resampling trials or rerunning clustering. A permutation moves whole
row labels and then recounts all relevant pairs, preserving dependencies
among pairs sharing a row. The same row permutation is applied to all
three representations of a network. An unshufflable block stays fixed.
If every row is in one cluster, sharing and chance are both 100% and p=1;
such pairs remain in the results.

The statistic is the **equal mean across networks** of each network's
sharing fraction; the chance dash is the mean of that statistic's null
distribution. Defaults are 10,000 permutations and RNG seed 0
(`--n-permutations`, `--permutation-seed`). Two-sided p values use twice
the smaller inclusive tail with an add-one correction, capped at 1.
BH-FDR is applied separately for each cohort and analysis type, across all
representations and all unique reported tests: individual tasks/phases,
the all-pairs summary, and the sibling-group pooled summaries where
applicable. Repeated presentations of a task's test are counted once.
This does not control FDR jointly across L2 cohorts, or test the difference
between trained and untrained networks. The result saves `p_perm`,
`q_fdr`, per-network observed/expected counts, and the null definition,
permutation settings and FDR family under `permutation`.

Result **version 6** retains these statistical checks and requires
`variance_method="trial_time"`: variance is computed directly across trials
and time, without first averaging trials in S1 direction bins. Each epoch
ends at its own task's actual trial endpoint, excluding padded timesteps.
Input, hidden and modulation use the same definition; column normalization
and clustering settings are unchanged. Derived overmembership, lesion and
lesion-postprocessing caches also carry the estimator marker; readers reject
unmarked results instead of mixing them with new clusters. `--resume` and
`--skip-completed` check provenance, input/output times and the requested
control repeat count. Changing only `--control-repeats` with `--resume`
reuses clustering and lesions, then recomputes controls and plots.
Standalone `lesion_controls.py
--skip-completed` also checks the requested settings and upstream freshness.
Reclustering clears the run's old analysis artifacts, preserving independently
owned sibling-geometry files.

**Migration from v5 requires reclustering trained and untrained networks.**
Old cluster pickles have no estimator marker and are rejected, including by
`--skip-completed` and trained-pipeline `--resume`. After reclustering, rerun
`sibling_row_sharing.py` for each trained cohort and once per architecture
with `--untrained`, then regenerate the paper figures. Updating only the
sharing statistics cannot repair old variance features. The trained pipeline
also recomputes lesions and controls when their upstream clusters change.
The state-space trajectories, distances and transplant analyses are independent
of this change. The derived `state_space_vs_lesion.py` comparison depends on
lesion clusters and must be recomputed after the new lesion summaries. Previous numerical
conclusions from S1-conditioned clustering require confirmation after this
migration; existing PNGs are historical until regenerated.

Other files in `multiple_tasks_diagnosis/` need the same migration:
`multiple_task_diagnosis.py` regenerates `diagnosis_*` and
`same_cluster_diagnoses_*`, and `rule_cue_ablation.py` regenerates its own
variance-based results. These scripts are not run by `run_pipeline.py` or
`run_pipeline_analysis.py`.
The `modulation_feedback_*` results measure feedback dynamics and saturation
independently of the variance estimator and can be retained.

`paper_plot.py` reads these results and skips stale files with a rerun
command. `multitask_row_cluster_organization.png` retains its two
enrichment panels (period/task), network dots, median and chance=1.
Each panel fits its own y range and displays its own ticks; the original
organization figure uses logarithmic axes, and its matched-task supplement
uses linear axes to retain zero-sharing networks.
Its tests are unchanged: trained versus untrained uses raw, two-sided
Mann–Whitney p values; without controls it displays the saved Wilcoxon
comparison against 1. These p values are not BH-adjusted q values.

`multitask_row_cluster_counts_seeds_n.png` reports how many clusters the
clustering selected per network, from the `cluster_counts` entry of the
sharing result (schema 1: the `row_tol_k` and `col_tol_k` of the normalized
clustering, the modulation at its plotted G=300 pre-grouping, the K whose
labels every row-cluster figure reads): the task x period row clusters on
the left and the column clusters (neurons; synapses for modulation) on the
right, as medians with interquartile whiskers
(means with SD until 2026-11-10; the counts are skewed, the hidden column
count running from 5 to over 100), trained filled and untrained hollow as in
the sharing bar figures. A bracket over each pair carries the two-sided
Mann-Whitney test of trained against untrained computed in `paper_plot.py`:
a star for BH q < 0.05 over the figure's six tests, "n.s." otherwise; the
save message adds the medians, ranges, p and q per representation and axis.
The counts barely move with training, and the stars mark the exceptions. A sharing result
written before this entry existed asks for a rerun of the summary.

`multitask_variance_partition_matched_tasks_n.png` quantifies task and period
contributions to **normalized variance-profile differences**, independently of
clustering or the number of clusters. It uses the same complete five-task,
five-period grid as the matched sharing supplements. Input, hidden and raw
modulation features come from `cell_vars_rules_sorted_norm` in the current
cluster caches; rows are matched by `tb_break_name`. The original per-feature
maximum normalization across all task-period rows is retained; the subset is
not renormalized, and the modulation features are not K-means group averages.

For each network, write the feature vector for task q, phase l as
`v[q,l] = mu + task[q] + period[l] + residual[q,l]`, using marginal means
minus the overall mean for the two main effects. Centering is per feature.
On this balanced grid, `SS_task = 5 * sum_q ||task[q]||^2`,
`SS_period = 5 * sum_l ||period[l]||^2`, and
`SS_residual = sum_q,l ||residual[q,l]||^2` add to the centered total sum of
squares. Each fraction is its sum of squares divided by that total. Feature
sums are formed before division (not an equal average of per-feature ratios).
The residual contains task-by-period interaction and feature estimation noise;
there is only one estimated profile per cell, so these cannot be separated.
These fractions are descriptive, not chance-corrected, explained fractions of
raw trial activity, or causal contributions. They apply to the five-task subset.

The companion `multitask_variance_partition_all_n.png` uses every observed
task-period row, with equal weight per row. Tasks with more observed periods
therefore receive more total weight. Absent phases are not filled with zeros.
Because the all-task design is incomplete, task and period explanations can
overlap. Fit task-only, period-only and additive task + period models, each
including an intercept. Let their explained fractions of centered total sum
of squares be `R2_T`, `R2_P` and `R2_TP`. Allocate contributions by averaging
each factor's gain over the two entry orders:

```text
task     = 0.5 * (R2_T + R2_TP - R2_P)
period   = 0.5 * (R2_P + R2_TP - R2_T)
residual = 1 - R2_TP
```

These add to one. The first two are allocations of explanatory information,
not unique independent main effects; they depend on the observed row design
and are not adjusted for factor degrees of freedom or chance. The balanced
five-by-five design makes task and period orthogonal, so the same calculation
recovers the matched decomposition above. Both views call the **same**
`partition_task_period_variance` function, with SVD column-space projections
to handle redundant indicator columns or confounded factors. Feature sums
are pooled before division; no neuron or synapse is a statistical replicate.

The additive `variance_partition_all` and `variance_partition_matched_tasks`
entries (schema 2) save each network's allocated sums of squares and fractions,
the three models' explained sums of squares, feature counts, exact observed row
names, task/phase names, row weighting, allocation, normalization and variance
method. Schema-1 matched fractions remain mathematically valid, but a lightweight
`--variance-partition-only` refresh supplies the shared metadata and the all-task
entry; no reclustering is needed. A zero total gives undefined
fractions (NaN in pickle, null in JSON); those networks are omitted from that
representation's plot/tests and the valid n is reported in the save message.
A single panel groups bars by Task, Period and Interaction/residual, with
input/hidden/synapse colors and filled trained/hollow untrained twins matching
the sharing figures. Bars show network means with sample-SD whiskers. Fractions
sum to one within a valid network and representation. Two-sided asymptotic Mann-Whitney tests with
tie and continuity correction compare task and period fractions between cohorts;
BH correction uses a separate family of six tests per view (three
representations x task/period). Untestable comparisons retain a p=1 slot in that correction and have
no significance annotation. Bracketed stars denote trained/untrained q < 0.05,
not a comparison against chance; exact p/q values are printed in the save
message. Residual fractions have no separate test. Group comparisons are
computed from the saved network fractions when plotting, as in the other direct
trained/untrained comparisons; no neuron or synapse is treated as a replicate.

Normal sharing runs also compute both supplements, consuming one network's
features at a time. Missing features skip both for the whole cohort; missing
matched rows skip only the matched view. The all-task view requires identical
observed row sets across representations and networks, and its figure compares
cohorts only when their exact row sets and method metadata agree. Malformed data
raise an error. To add the views to an existing current summary without repeating the
sharing permutations, update the exact saved cohort and then plot:

```bash
python multiple_task/sibling_row_sharing.py --feature L21e4 --variance-partition-only
python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained --variance-partition-only
python figures/paper_plot.py --multitask-l2 1e4 --only variance_partition_matched_tasks
python figures/paper_plot.py --multitask-l2 1e4 --only variance_partition_all
```

The update checks provenance and row-organization counts against the saved
summary before writing. Missing features or changed cohort members leave the
summary untouched. If the saved cohort lacks the full matched grid, the update
retains only the new all-task view and removes any obsolete matched partition.
It does not discover new networks; to add networks, rerun the regular
sharing command. No forward passes, variance estimation, clustering or lesion
analysis need to be repeated. Both figures share their plotting, validation and
cohort-comparison code, including horizontal x labels and the same styling.

The permutation tests above apply to the two sharing bar figures:

- `multitask_row_cluster_same_task_seeds.png`: per-task mean sharing with
  sample SD across networks, restricted-null chance dash, and a star for
  **q_fdr < 0.05**.
- `multitask_row_cluster_same_period_seeds.png`: the corresponding
  per-phase view, with phase-colored tick labels.

Both ends of each SD whisker are displayed even when they extend outside
0–100%; SD describes spread and is not a confidence interval. Only the
seeds versions are retained: the former mean-only figures duplicated their
means, chance dashes and significance tests.

A star identifies a departure from the conditional null; the bar relative
to the dash shows the direction. It neither establishes causality nor
proves that training created the organization. Trained and untrained stars
use separate within-cohort tests. Their difference in significance is not
itself a significant difference between cohorts.

The follow-up `multitask_row_cluster_same_period_synapses_n.png` extracts only
the synapse bars from the all-task same-period figure. Means, sample SDs and
restricted-null chance dashes are unchanged. Its bracket annotations instead
test the directional alternative **trained > untrained** within each period:
one-sided Mann-Whitney U (`alternative="greater"`, trained samples first), using
tie- and continuity-corrected asymptotic p values. BH correction includes all
five periods, including those with lower trained sharing. A bracket over each
cohort pair carries a star for q < 0.05 and "n.s." otherwise (the exact p and
q are in the save message; they were printed on the figure until 2026-11-10,
when it took the size and styles of the other row-sharing bar figures); a
star supports greater sharing in trained networks.
This directional follow-up replaces this figure's earlier two-sided comparison.
Each network contributes one sharing fraction; task pairs are not independent
samples and the cohorts are not paired. The tests concern the distributions
of those fractions, not specifically the difference of their means. Zero
sharing is retained. Both cohorts are required. Since 2026-11-10 a bottom panel
repeats the comparison for the **chance-corrected** sharing, each network's
fraction minus its own restricted-null chance in percentage points (the
dashes of the top panel, per network), with the top panel's one-sided
Mann-Whitney U tests (trained > untrained) and their own BH correction over
the five periods; a pair in which the untrained networks are the higher
ones therefore reads "n.s." in both panels. The raw fractions depend
on the clustering's granularity, which differs between the cohorts: the
synapse row clusters number 10 to 20 with training (median 14) against 10
to 14 without (median 11, usually with one cluster of about 14 of the 49
rows), and within the trained cohort the cluster count and the pooled
sharing correlate at −0.66. Each network's null carries that granularity,
so the bottom panel shows what the period adds: in the L21e4 cohort
Stimulus 1 stays higher without training (+7.6 against +22.7 points),
Stimulus 2 stays higher with it, and Memory 1 reverses, above chance with
training (+2.7) and below without (−3.6) although its raw fraction is the
lower one. Stimulus 2 and Memory 2 have ten task pairs per network (five
tasks), so their fractions move in steps of ten points; the Memory 2 star of
the top panel rests on one shared pair in five of 25 networks and does not
survive the correction. The matched-task figure gets the same panel.
These comparisons are computed
from current v6 per-network counts while plotting; once clustering and sharing
have been migrated, this extra figure needs no further summary rerun:

```bash
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_period_synapses
```

The trained-only `multitask_row_cluster_coassignment_n.png` shows all task-period
row pairs as three heatmaps (input, hidden and modulation; currently 49 × 49).
For rows $i,j$ and representation $r$, the cell is
$C^{(r)}_{ij}=N^{-1}\sum_{s=1}^{N}\mathbf{1}[c^{(r,s)}_i=c^{(r,s)}_j]$,
using exactly the trained networks named in the current sharing summary.
Rows are aligned by task-period identity before counting; cluster numbers are
compared only within each network. A source with missing rows/representations,
or organization counts inconsistent with the saved summary, skips the figure
instead of changing its cohort or using cell-specific denominators. The
calculation reads current cluster caches at plot time and needs no new trials,
clustering, or permutation tests.

All panels use the same task order, chronological periods within tasks, and
0–1 color scale. The viridis palette, task-only tick labels with phase-colored
backgrounds, and thin gray boundaries follow the clustered task-variance
heatmaps. The colorbar denotes co-clustering frequency rather than normalized
variance. Lines separate tasks within the displayed lower triangle; the plot
does not impose a new consensus clustering or copy a single network's order.
The underlying matrices are symmetric with a diagonal of one (self-pairs);
the figure masks the upper triangle and diagonal, showing each distinct row
pair once. Masked cells are blank, not zero. These raw frequencies show which
particular pairs account for the aggregate sharing patterns; they are neither correlations nor
chance-corrected enrichment, and cluster number/size affect their baseline.
This is a descriptive companion to the sharing and organization figures, not
independent evidence or a trained/untrained comparison.

```bash
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_coassignment
```

`multitask_row_cluster_period_pairs_n.png` reduces those matrices to the period
types: for each representation, which two periods tend to share a row cluster.
For every unordered pair of period types (a period with itself included: two
rows of that period, of different tasks), each network counts its row pairs of
those two types and the ones that fall in one cluster
(`sibling_row_sharing.summarize_period_pair_coassignment`), which gives one
probability per network. A bar is the mean of that probability over the
networks, its whisker the sample SD; the pairs are sorted per panel, most
probable first (at most `_ROW_PERIOD_PAIRS_TOP` = 20; five periods give
fifteen), and the two period names of a row sit on their heatmap phase colors.
A pair of different periods pools the rows of one task with those of two
tasks. Each panel has its own x range: the full 0–100% when its bars pass
50% (input and hidden), a shorter one fitted to the bars otherwise (30% for
the synapses in the L21e4 cohort), so bar lengths are not comparable between
panels and the tick labels carry the scale. The dashed line is the cohort mean of the probability over all row
pairs, the level the networks' cluster sizes give; it differs between the
panels (the representations select different numbers of row clusters), so a
bar is read against its own panel's line. As with the co-clustering heatmaps
these are raw frequencies of the trained networks, with no chance correction,
test or untrained control. In the L21e4 cohort (25 networks) the input and
hidden rows of the two memory periods always share a cluster (Memory 1 with
Memory 1, Memory 1 with Memory 2, Memory 2 with Memory 2: 100%, against 30%
for any two rows), memory and response rows do so in 47% of their pairs, and
stimulus and memory rows never; the synapse rows are far less period-bound
(9% for any two rows), led by the stimulus-memory pairs of one epoch
(Stimulus 2 with Memory 2 18%, Stimulus 1 with Memory 1 15%), with Memory 2
with Memory 2 last at 2%.

```bash
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_period_pairs
```

`multitask_row_cluster_joint_pairs_n.png` asks which particular row pairs the
three clusterings agree on: for rows $i,j$ the joint frequency is
$J_{ij}=N^{-1}\sum_{s=1}^{N}\prod_{r}\mathbf{1}[c^{(r,s)}_i=c^{(r,s)}_j]$, the
fraction of networks in which the two rows share a row cluster in the input,
the hidden and the synapse clustering at once (`joint` of
`sibling_row_sharing.summarize_row_coassignment`). The figure lists the 20
pairs with the highest $J_{ij}$ (`_ROW_JOINT_PAIRS_TOP`), ties in the fixed
task and period order, as four bar panels sharing the rows: $C^{(r)}_{ij}$ for
input, hidden and synapses, then $J_{ij}$. Each row is labelled by its two
task-period rows on their period colors; a panel's dashed line is its mean
over all 1176 row pairs (30%, 29%, 9% and 2.7% in the L21e4 cohort). A
separate top 20 per representation would not be informative for input and
hidden, where 266 and 256 pairs sit at exactly 100%.

In the L21e4 cohort (25 networks) the seven leading pairs, at 92 to 100%, are
all one task's last memory period with its own response period (MemoryPro and
MemoryAnti: Memory 1 with Response; the five integration tasks: Memory 2
with Response). DelayPro's stimulus with ReactAnti's response follows at 88%,
then five pairs at 56 to 64% (a stimulus row with a reaction task's response
row, or two tasks' Stimulus 1 rows); 13 pairs reach 50%, and the rest of the
list sits at 32 to 36%. Three limits apply. The leading pairs share a cluster
in the input and hidden clusterings in every network, so $J_{ij}$ is close to
the synapse frequency: the figure shows which synapse-level pairings also
hold at the neuron level, not three independent agreements. The input rows of
periods without a changing input are all zero (23 of the 49) and cluster
together by construction, which accounts for part of the input panel's 100%.
And, as above, these are raw frequencies of the trained networks, with no
chance correction, test or untrained control.

```bash
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_joint_pairs
```

The matched-task supplement, `same_period_matched_tasks`, keeps the original
full-task clustering and restricts every phase to the same five tasks:
`delaydm1`, `delaydm2`, `contextdelaydm1`, `contextdelaydm2`, and `multidelaydm`.
Each network therefore contributes ten task pairs per phase, identically
across representations and trained/untrained cohorts. The matched-summary
functions reject missing required rows. If any network lacks them, the driver
prints the reason, skips the matched-task summaries and their cut robustness
for the whole cohort, and still saves the all-task results (the all-task cut
included) without those optional keys.
It does not drop networks or shrink the matched task set. Malformed rows,
duplicate rows and inconsistent row sets across representations still fail
validation. Runs without completed modulation clustering are explicitly
skipped by the driver.

Its means, SDs, restricted-null chance levels and tests are recomputed from
these rows. The 18 chance tests (five phases plus pooled, three
representations) form a separate BH family within each cohort. Because the
task set is identical across phases, the theoretical chance is the same
in every phase of a representation; finite permutation estimates can differ
slightly. The new figure is
`multitask_row_cluster_same_period_matched_tasks_seeds_n.png` with legends
enabled. Its stars still compare each phase with its own null.

Direct phase comparisons are saved under `period_contrasts` and printed:
Stimulus 2 minus Stimulus 1 and Memory 2 minus Memory 1, in percentage points.
First compute the difference within each network, then test the mean by
swapping whole phase summaries within networks (a paired sign-flip test).
This assumes within-network exchangeability of the two phase summaries
under the null; task pairs are not independent replicates. All sign maps
are enumerated when `2**n_runs <= n_permutations`; otherwise the test uses
Monte Carlo permutations with add-one correction. These six tests (two
contrasts, three representations) have their own BH family in each cohort.
Neither these tests nor the figure stars compare trained with untrained.

The direct synapse comparison for this same task population is
`multitask_row_cluster_same_period_synapses_matched_tasks_n.png`. It reads
`same_period_matched_tasks` and shares the all-task synapse figure's renderer
and statistics: one fraction per network (including zeros), mean ± sample SD,
restricted-null chance dashes, and one-sided Mann-Whitney U tests of
**trained > untrained**. Its five period tests form their own BH family,
separate from the all-task figure and the within-cohort chance tests.
Raw p and adjusted q compare cohorts; they do not test the chance dashes.
Both cohorts must have compatible matched-task summaries. The original
full-task clustering is retained, so no reclustering or summary rerun is
needed when these entries are already present:

```bash
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_period_synapses_matched_tasks
```

This comparison controls task composition across periods and cohorts. It
does not control evaluation-trial batches, so it is supplementary evidence
of a cohort difference rather than an isolated causal effect of training.

Two companion supplements use the same five tasks and original cluster labels:

- `same_task_matched_tasks`, drawn as
  `multitask_row_cluster_same_task_matched_tasks_seeds_n.png`: ten phase pairs
  per task. Means and SDs equal the corresponding five tasks in the original
  figure, but the null now shuffles only these five tasks within each phase.
  Chance and significance are recomputed; its five tasks plus pooled, across
  three representations, form an 18-test BH family in each cohort. No unused
  sibling-group tests enter this family.
- `organization_matched_tasks`, drawn as
  `multitask_row_cluster_organization_matched_tasks_n.png`: enrichment is
  recomputed against all 300 pairs of the selected 25 rows. Same-period and
  same-task comparisons each have 50 pairs; 200 pairs differ in both factors.
  Its linear axis includes zero-sharing networks in both the plot and the
  trained/untrained Mann–Whitney tests. These are raw two-sided p values,
  as in the original organization figure, not BH-adjusted q values.

The organization supplement checks whether the task/phase pattern persists
with a complete task-by-phase grid. The same-task supplement is its per-task
companion, not new evidence from the unchanged bar heights. These five
integration tasks are a restricted task family, so the supplements do not
replace the full-task overview or establish generality across all tasks.

Each of these three summaries has `schema_version=1` inside the version-6 payload, and
records `task_names`, `phases` and retention of the original clustering.
The period view also retains its original `tasks` metadata for compatibility.
Current v6 results without a supplement still draw the original figures;
a supplement asks for a rerun if its field is absent. After regenerating
the upstream trial-time cluster caches, compute the supplements:

```bash
conda activate mpn
python multiple_task/sibling_row_sharing.py --feature L21e4
python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_period_matched_tasks_seeds
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_task_matched_tasks_seeds
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_organization_matched_tasks
```

With current trial-time cluster caches, this step requires no network
forward pass or reclustering. It controls task composition; all phases use
variance pooled directly over trials and time. The input memory rows can share a
cluster because their features are near zero; that is not memory coding.

The further figure `multitask_row_cluster_cut_robustness_matched_tasks_n.png`
assesses cut sensitivity for synapses. The complete saved modulation row tree
is cut at the prespecified targets K=4,6,...,20, **before** selecting the same
25 rows (K=4 is coarser than the five periods; the saved modulation row
clusterings themselves hold 10 to 19 clusters). Its three panels show, for
the same-task (different period), same-period (different task) and unrelated
pairs among the selected 25 rows (50, 50 and 200 pairs), the fraction of
those pairs that fall in one cluster at each K, on one shared 0 to 1 axis.
Points are cohort means with SD whiskers over networks (SD describes spread
and is not a confidence interval; both cohorts are drawn at the same K); the
thin grey line of each cohort is the fraction over all 300 pairs, the baseline
of the organization figure's enrichments, so an enrichment is a colored
curve over the grey one. Fractions are calculated within each network before
averaging. The saved entries still hold the enrichments and
`Delta = E_same_task - E_same_period` per network and K. These repeated cuts
are sensitivity checks, not additional independent samples or new
significance tests.

The cut supplement is stored as `cut_robustness_matched_tasks` (schema 1).
`fcluster(..., criterion="maxclust")` retains tied merge heights, so K is an
upper bound on the number of active clusters. Actual full-tree cluster counts
(including any pre-existing held-out unresponsive class) and selected-row cluster
counts are saved for every network and K. The x ticks carry the target K,
with the full-tree count in parentheses when ties keep it below K; the plot
log reports, per cohort, the range over networks of the full-tree counts and
of the clusters formed among the 25 rows.
Identical profiles are not arbitrarily split to force a requested K. A zero
same-period enrichment remains zero; if no selected pair shares a cluster,
the baseline is zero and enrichment/Delta are undefined (NaN), not zero.
The saved tree must reproduce the original row partition before any recutting.

The all-task twin `multitask_row_cluster_cut_robustness_n.png` (stored as
`cut_robustness_all`, schema 1) recuts the same trees at the same K but keeps
every row of every task, so its pair classes and grey baseline are those of
all pairs of a network's rows (49 rows and 1176 pairs for a complete run:
258 same-period, 70 same-task, 848 neither), as in the organization figure.
The periods then
differ in task composition (five tasks in Stimulus 2 and Memory 2, fifteen in
Response), so this is the overview of the original organization check across
cut depths, not the controlled comparison. Its ticks follow the same rule. It needs no
five-task grid: the driver computes it whenever the modulation trees are saved.

Normal summary runs add both cut supplements when trees are available. Missing
trees skip only these supplements; missing matched rows still skip all matched
outputs, but not the all-task cut. To add both to existing v6 matched-task
results without repeating the permutation tests or including newly finished
networks, use:

```bash
python multiple_task/sibling_row_sharing.py --feature L21e4 --cut-robustness-only
python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained --cut-robustness-only
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_cut_robustness
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_cut_robustness_matched_tasks
```

This mode uses exactly the summary's saved network list. It validates the
task/phase metadata and checks that cluster caches reproduce both the saved
all-task and matched organization counts, leaving every previous statistic intact. If caches changed,
recompute the full summary first.
No network forward pass or new hierarchical clustering is needed.
To include newly completed networks, run the normal summary command without
`--cut-robustness-only`; for example,
`python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained`.
It discovers all complete controls, recomputes their statistics and cut
robustness, and skips incomplete modulation caches. Redraw the row-cluster
figures afterwards. Cut-only updates and plotting alone never add networks.

The following **historical S1-conditioned L21e4 snapshot**, checked on
2026-11-05 after adding init 5 and 6, predates the trial-time revision. It
contained 25 trained and seven untrained networks (init 0–6); at that time
all eight row-cluster PNGs matched in-memory renders of those caches.
These values and conclusions must be rechecked after the new analysis;
they are not results of the trial-time estimator or the current one-sided test.
The old synapse-only same-period follow-up used two-sided tests as follows:

| Period | Trained mean | Untrained mean | Raw p | BH q |
|---|---:|---:|---:|---:|
| Stimulus 1 | 19.08% | 35.90% | 0.0000788 | 0.000394 |
| Stimulus 2 | 20.00% | 10.00% | 0.0310 | 0.0517 |
| Memory 1 | 11.49% | 9.09% | 0.4059 | 0.4059 |
| Memory 2 | 20.40% | 8.57% | 0.0130 | 0.0326 |
| Response | 12.15% | 9.80% | 0.1833 | 0.2291 |

In that snapshot, only Stimulus 1 (lower in trained networks) and Memory 2 (higher) met q < 0.05.
The all-task results therefore do not show a uniform training-associated
increase in sharing. Task composition varies between periods; these direct
cohort comparisons are not tests of differences between periods.

At the original cut, all 25 trained and all seven untrained modulation networks
have positive Delta; medians are 1.579 and 3.943, respectively. Input and hidden
Delta values are negative in both cohorts. The full-tree recuts, on the K grid
of that time (5 to 25; K=25 was dropped on 2026-11-07 because the saved
clusterings hold at most 19 clusters, and the grid became 4, 6, ..., 20 on
2026-11-08), achieved exactly the requested K for
every network, with no undefined enrichments:

| K | Trained Delta median | Untrained Delta median | Trained same-period median | Untrained same-period median |
|---|---:|---:|---:|---:|
| 5 | 0.804 | 2.763 | 1.052 | 0.553 |
| 10 | 1.286 | 2.769 | 1.350 | 0.556 |
| 15 | 1.622 | 3.750 | 1.273 | 0.562 |
| 20 | 2.769 | 5.294 | 1.059 | 0.353 |
| 25 | 3.529 | 5.250 | 0.800 | 0.375 |

Thus the median task preference and the trained/untrained ordering persist at
all five cuts. This does not mean every network prefers task at every cut:
20/25 trained networks have positive Delta at K=5, 24/25 at K=10 (one tie),
and 25/25 at K=15,20,25. Nor is trained same-period enrichment always above
the all-pairs baseline: its median is below 1 at K=25. The robustness claim is
about the direction of the cohort medians across this tested K grid, not
significance at every K, arbitrary cluster settings, or all task families.

The **earlier L21e3 audit on 2026-11-05**, before the matched-task additions
and control expansion, covered 17 trained networks and five untrained networks
(seed 749 config, init 0–4). At that time the original organization,
same-task-seeds and same-period-seeds figures matched the caches pixel for
pixel. The following numerical conclusions refer to that historical snapshot:

- The organization figure's median same-period enrichment is 1.87x for
  input, 1.74x for hidden and 0.93x for M; same-task enrichment is 0.92x,
  0.88x and 4.03x. This is a relative preference for phase organization in
  input/hidden and task organization in M, not an exclusive division.
- The same-task figure shows M sharing above its restricted null in 11 of
  the 13 tasks with phase pairs after BH correction. DelayPro and DelayAnti
  have low M sharing but are not significantly below chance (q=0.149 and
  q=0.546); the older signed-rank claims for these two tasks do not hold.
- The same-period figure shows 100% input/hidden sharing in both memory
  phases, against permutation means of about 28–33%. M also has above-null
  sharing in Stimulus 1, so it is not devoid of phase organization.
- Untrained networks already show strong M task organization (median 4.36x
  versus trained 4.03x; raw Mann–Whitney p=0.256). These data do not establish
  that training creates or strengthens that overall organization, nor does
  a nonsignificant comparison establish equivalence.

Neither the historical L21e3 numbers nor the old L21e4 comparison above
describes the revised trial-time results. The manual, read-only check is
`python test/row_cluster_results_check.py --multitask-l2 1e4`: select the cohort
used to generate the PNGs and use the same plotting environment. It verifies
saved counts and means, includes matched summaries and cut statistics when
present, and compares all available row-cluster PNGs (up to eight for
L21e4) with in-memory renders without changing caches or figures. Without
the argument it uses `paper_plot.py`'s default cohort.

The untrained control asks whether training makes this organization or the
architecture alone does, for instance because M, with its 4 s memory, keeps
a trial's earlier periods in every later one. `untrained_clustering.py`
builds, for one run's config, an untrained network per init seed
(`core/network_loading.build_network`: the same architecture, weights drawn
after seeding NumPy and torch with the init seed, none of the trained ones;
the same builder the state-space control uses) and runs the complete
`multiple_task_analysis.main` clustering on it (`untrained_init_seed`), with
every output under
`multiple_tasks_analysis/untrained/<aname>_init<k>/`, named
`..._init<k>` (`multiple_task_analysis.output_names`), where no trained-run
reader looks. `--skip-completed` leaves out networks whose two complete
cluster pickles carry the current variance marker; no lesion step runs. `sibling_row_sharing.py --untrained`
then analyzes those networks. L2 acts only during training, so the untrained
network of an L21e4 config is, init seed for init seed, the untrained
network of any L21e3, L21e2 or L21e5 config (verified weight for weight on
2026-11-04); one set of controls therefore serves every L2 cohort of an
architecture. The analysis takes every untrained network whose config shares
`--feature`'s architecture (the feature tag without its L2 strength,
`core/multitask_naming.py`: `L21e4` and `L21e3` both give the default
architecture, `L21e4lam0.9` the lambda = 0.9 one) and writes
`sibling_row_sharing_untrained.pkl` (`sibling_row_sharing_lam0.9_untrained.pkl`
for the lambda = 0.9 cohort), which the figures of every `--multitask-l2`
read; `config_features` in the result lists the configs the networks came
from (it replaces the per-cohort `sibling_row_sharing_<feature>_untrained`
files of the first version). An untrained
network of seed 749's config is not degenerate: hidden std 0.12 (trained
0.16), M std 0.38, and up to about a quarter of its synapses reach their bounds
late in long trials, as in the trained network (`modulation_feedback.py`). The
organization figure draws the untrained networks, hollow with a gray median,
right of the trained runs in each representation's column, with the
two-sided Mann-Whitney p of trained against untrained above the column. The
per-task and per-period figures give every trained bar a hollow twin on its
right with the untrained networks' mean sharing and permutation chance.
There is no six-network threshold for the permutation test; few networks
still limit generalization across initializations. These figures draw the
trained runs alone, with the commands to run, while the untrained result is
missing. Old v4/v5 summaries require trial-time cluster caches before the
statistics command can regenerate them. Reclustering is necessary whenever
those caches are missing or were computed with the old S1-conditioned estimator.

## Modulation feedback and saturation

`multiple_task/modulation_feedback.py` asks how long M remembers, given
that M feeds back on itself through h. One step of the plastic layer is
`h_t = tanh((W * (1 + M_{t-1})) x_t + b)`, then
`M_t = clip(lam M_{t-1} + eta h_t x_t^T, -1, 1)`. Linearized, the row M_i
of post neuron i is lam times the identity plus a rank-one term, so on its
unclipped synapses every direction the neuron cannot read (orthogonal to
`W_i * x`) decays by exactly lam, whatever the training, and only the one it
reads, `<W_i * x, M_i>`, decays by
`mu_i = lam + eta (1 - h_i^2) sum_{j unclipped} W_ij x_j^2`; of the 90,000
synapse directions at most 300 are not set by lam. A clipped synapse stays at
its bound while `eta h_i x_j` exceeds `1 - lam` (0.01 at lam = 0.99).

```bash
python multiple_task/modulation_feedback.py --feature L21e4 --seed 749
python multiple_task/modulation_feedback.py --feature L21e4 --seed 749 --untrained-init-seed 0
python multiple_task/modulation_feedback.py --feature L21e4 --seed all
```

The script steps the network through the clustering analysis's evaluation
trials (50 per task, `mode_input="random"`, `long_all=False`), about a minute
per network on a CPU, and records at every step mu_i of every post neuron and
the share of synapses clipped to each bound. A finite-difference check
through the real network step, in double precision, comes first: one step's
response to a perturbation of M along a random direction, along the readable
mode and along an unreadable direction must match the linearized map, mu_i
and lam. The readable mode's time constant is `dt / (1 - |mu|)` (the
convention that gives lam its 4000 ms). It writes
`multiple_tasks_diagnosis/modulation_feedback_<aname>[_init<k>].{pkl,png}`:
per task and period the quantiles of mu_i, its shares above 1, below -1 and
below lam, the median time constant and the shares of synapses at each bound;
the same pooled by period type (Context, Stimulus, Memory, Response); each
task's saturation time course; and the check's errors.

On seed 749 (L21e4, eta = 2.15) the check holds to 1.2e-7 per unit
perturbation. The readable mode's mu has a median of 0.986-0.987 in every
period type, below lam for 80-84% of the post neurons and steps, so the
median time constant is 2.9-3.1 s against lam's 4 s; mu exceeds 1 for 4%
(Memory) to 14% (Response) and is never below -1, and a tail reaches mu of
about 0.4 (5th percentile in Memory). Synapses at a bound rise from about 1%
in the Context period to peaks of 15-25% late in the stimulus and memory
periods. An
untrained network of the same config (init seed 0, eta = 1) has mu
symmetric about lam (median 0.988, 5-95% 0.94-1.03, above 1 for 18-28%),
median time constants of 3.3-3.8 s and up to about 25% of synapses at a bound
late in long trials. In neither network does the feedback lengthen M's
typical memory: the unreadable directions keep lam's 4 s, the median readable
mode decays a little faster (more so after training), and where mu exceeds 1
the mode grows until its synapses clip, which the saturation share counts.

## Rule-cue ablation

`multiple_task/rule_cue_ablation.py` tests the rule cue's contribution to
task/period organization. Earlier S1-conditioned analyses suggested period
grouping for input/hidden and task grouping for modulation; this pattern
needs confirmation with trial-time features. The persistent rule cue can
change stimulus responses and accumulated modulation. The script runs one network, trained or untrained
(`--untrained-init-seed`, `network_loading.build_network`), on the evaluation
trials of `modulation_feedback.py` (50 per task, `mode_input="random"`, trial
seed 0) twice, once as they are and once with every rule-cue channel set to
zero (the cue and its input noise), and computes the input, hidden and
modulation features of both as the clustering does: per task and period, the
variance directly across trials and time, with no S1-bin averaging
(`helper.task_variance_period_numpy`), each column divided by its maximum
over the rows (the `_normalized` features of the heatmaps), the rows in the
clustering's period-major order and windows. Epochs stop at each task's real
trial end; padding is excluded. Rule-cue results are now version 3 with a
`variance_method="trial_time"` marker and must be regenerated as well.

```bash
python multiple_task/rule_cue_ablation.py --feature L21e4 --seed 749
python multiple_task/rule_cue_ablation.py --feature L21e4 --seed 749 --untrained-init-seed 0
# another evaluation batch, with other stimulus-modality draws
python multiple_task/rule_cue_ablation.py --feature L21e4 --seed 749 --untrained-init-seed 0 --trial-seed 2
```

The readout needs no clustering. It is the clustering's metric, the
Euclidean distance between two rows divided by its mean over every two rows
(below 1: closer than a typical pair), averaged per representation and
condition over three kinds of pair: one task in two periods ("same task"),
one period in two tasks ("same period") and the rest ("other"). Row
correlation was tried first and fails for input and hidden, whose period
grouping is mostly row magnitude (no input feature varies in the memory
periods, so those rows are zero), which a correlation discards. How much the
cue changes the features themselves is the correlation of each row with the
same row of the other condition and the ratio of the summed unnormalized
features (no cue / cue). `mode="random"` draws each task's stimulus
modalities once per batch, so all trials of a task share them; the script
records the modality of each task's first stimulus, and `--trial-seed` draws
another batch with other draws, so a result can be checked against the
modality draw. It writes
`multiple_tasks_diagnosis/rule_cue_ablation_<aname>[_init<k>][_trials<s>].{pkl,png}`
(result version 3): the distance matrices and pair means per condition and
representation, the matched-row correlations, the feature ratios and the
features themselves.

The historical S1-conditioned report covered only the untrained network of
seed 749's config (init seed 0), on trial seeds 0 and 2, which agreed.
These values have not been recomputed with trial-time variance. Without the cue the
synapse rows keep their task grouping (same-task relative distance 0.62 with
the cue, 0.63 without; same-period 1.02 and 0.98), the input rows are
unchanged (same-period 0.71 and 0.68, same-task 1.03 in both) and the hidden
rows group more by period (same-period 0.85 to 0.75, same-task 0.96 to
1.02); the features themselves change most in the modulation (matched-row r
median 0.33, summed features x0.76) and least in the input (0.60, x0.99).
That report suggested the cue alone did not explain task grouping; this
interpretation also needs rechecking. The trained run was not analyzed, and
another candidate, M's carry-over of a
trial's earlier periods (lambda = 0.99, a 4 s memory), is the question of
the lambda = 0.9 cohort (`L21e4lam0.9`) and of an untrained network with
lambda overridden, neither of which has been run through this script.
