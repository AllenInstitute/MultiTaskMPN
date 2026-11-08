# Multi-task clustering and lesion analyses

Details behind the `multiple_tasks` and `lesion` figure modes of
`paper_plot.py`: which caches each figure reads, what its statistics are, and
the conventions of the clustering caches. `python multiple_task/run_pipeline.py`
produces the per-run clustering, lesion, control and lesion-plot caches; the
commands are in the README. Row-sharing summaries and the paper figures are
separate steps.

The trial-time variance revision invalidates the old clustering-dependent
results. Numerical observations below labeled as historical were computed
with S1-conditioned features and must be rechecked after rerunning; their
presence here does not imply that new results have been generated.

## Run selection of the pipeline

```bash
# multi-task: train, analyze (clustering), lesion, lesion plots. Every training
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
# cluster_info and cluster_info_mod, lesion_prune_results, the
# lesion_controls pickle of --control-pool (responsive by default) and the
# two lesion_interaction pickles of each plotted pool (multiple_tasks_norm/
# and multiple_tasks_norm_<pool>/). With --resume, current steps are reused;
# missing/outdated steps and their dependents rerun. Without --resume,
# selected runs restart at clustering.
# --skip-completed needs --feature or --seed; to spread a cohort over nodes,
# give each process its own --seed list (two bare runs would pick the same
# checkpoints). Historical timing was about an hour of clustering plus four
# to six hours of lesions and one to two hours of controls per synapse type;
# clustering with the new variance estimator has not yet been timed.
python multiple_task/run_pipeline.py --feature L21e4 --skip-completed
python multiple_task/run_pipeline.py --feature L21e4 --skip-completed --resume
```

## Accuracy across modulation retentions

The `acc_plot` figures `retention_accuracy` and `retention_task_accuracy`
compare lambda = 0.99, 0.95 and 0.9 at L2 = 1e-4, with the same default
network architecture and modulation bounds. The feature tags are `L21e4`,
`L21e4lam0.95` and `L21e4lam0.9`; their saved `m_time_scale` values must be
4000, 800 and 400 ms respectively. Each dot is one network and the connected
diamonds are the across-network means. These figures use performance results,
not clustering or lesion caches.

```bash
# Add the newly trained cohort without replacing the existing evaluations.
python multiple_task/multiple_task_performance.py --feature L21e4lam0.95 --merge
python figures/paper_plot.py --only retention_accuracy
python figures/paper_plot.py --only retention_task_accuracy
```

Both figures report two-sided Mann-Whitney U tests for all three retention
pairs. Holm correction covers the three overall comparisons in the first
figure, and all 15 tasks times three pairs (up to 45 comparisons) jointly in
the second. A pair needs at least two networks in each group; unavailable
comparisons have no p value and do not enter the correction. Missing cohorts
keep their axis position and produce a message with the evaluation command.

## Figures

The `lesion` figure mode contains `lesion_heatmap`, `lesion_cluster_sizes`,
`cluster_corr_vs_lesion`, `cluster_corr_vs_lesion_weighted`,
`cluster_corr_vs_lesion_seeds`, `om_vs_lesion`, `om_vs_lesion_seeds`,
`plasticity_share`, `plasticity_share_cohort`, `plasticity_share_cohort_violin`,
`causal_vs_activity_tasks`,
`causal_vs_activity_seeds`, `task_specificity`, `task_specificity_seeds`,
`task_specificity_threshold_seeds`, `task_specificity_size`,
`task_specificity_pool_seeds`, `task_sharing_cohort`,
`state_space_vs_lesion` and `state_space_vs_lesion_seeds`. All but the seven
`_seeds`, the size and three `_cohort` figures read one run, `LESION_ANAME`,
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
`plasticity_share_cohort`, `plasticity_share_cohort_violin`,
`causal_vs_activity_seeds`, `task_specificity_seeds`,
`task_specificity_threshold_seeds`, `task_specificity_size`,
`task_specificity_pool_seeds`, `task_sharing_cohort`,
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
The plot reads only the `weighted-unnormalized_zero-W` scatter cache, drops
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
and the synapse column of `state_space_vs_lesion` (below) follow the same
mode. The OM scatter (`paper_plot.OM_LESION_MODE`) still reads **zero_W**
(selected synaptic weights set to zero); plasticity share compares the two
modes by definition.
`lesion_plot.py` records freeze_M as `primary_modulation_mode` in the
normalized-effect cache.

The OM-versus-lesion-profile scatter uses Spearman rank correlation and a
one-sided (negative association) modulation-cluster-footprint permutation test,
not linear regression. `lesion_plot.py` saves the scatter, `association`
(rho, matching permutation p, null statistics and test metadata), and
`binned_medians` in both single-mode and combined `schema_version=2` caches;
`paper_plot.py` takes the zero_W entry. **Run selection differs from the other
lesion figures:** the single-run scatter is drawn from `ANAME`, the clustering
example (seed 749 under `--multitask-l2 1e4`), the run of the
`overmembership_examples_dots_*` panels, so the example clusters are among its
points; every other single-run lesion figure uses `LESION_ANAME` (seed 921).
`paper_plot._om_scatter_run` holds the choice and the saved message names the
run. `om_vs_lesion_seeds` pools every sibling run either way. Because of the
shared run, the deciding block of `overmembership_examples_dots_1` (its most
significant input × hidden block; both panels take it from
`_om_example_selection`) is drawn on top of the scatter in the viridis color
the examples give its OM, with a black ring ("Example block"). The cache keeps
no block identity per point, so matching uses the cached observed/expected
ratio copied from the same `om_stack`; the highlight is skipped with a note
unless exactly one point matches.
The dot colors and scatter x coordinates display block over-membership as
`OM = observed / expected - 1`, with chance at zero. The cache retains raw
ratios for reconstructing observed counts; both scatter points and saved bin
x coordinates are shifted by one only when plotting. This translation leaves
the selected examples, Spearman correlation and permutation p values unchanged.
The shared dot colorbar marks zero; blocks with expected counts below one are
omitted, and supported blocks with no observed synapses (OM = -1) are hollow rings.
Up to five quantile bins provide median OM/L1 points and counts; tied OM values
stay together. The median connector is descriptive and is not forced to decrease.
`paper_plot.py` only reads these values, draws pale scatter points and the saved
median connector, and starts the L1 axis at zero. Pearson-only caches are skipped,
not silently reused. `python figures/paper_plot.py --only om_vs_lesion_seeds` draws the
cohort view, `multitask_om_vs_lesion_seeds_n.png`: one point per sibling run of
`LESION_ANAME` (same feature, any seed), the saved zero_W Spearman rho of that
run's OM scatter, filled when its saved one-sided permutation p is below 0.05,
with the median over runs as a dash. Because the per-run test is one-sided
(negative association), the legend title carries the one-sided Wilcoxon
signed-rank p of the runs' rhos being below zero; the other `_seeds` figures
keep their two-sided test. Runs whose OM cache is missing, legacy or has an
undefined rho are left out with a note.

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
`--only plasticity_share_cohort` draws `multitask_plasticity_share_cohort_n.png`,
the same layout with every sibling run of `LESION_ANAME` (same feature, any
seed) overlaid: the saved significant cells of all runs pooled per task as
pale grey points, each run's saved task median as a short near-black dash (so
it stays apart from the grey cells), and the median of the pooled cells as
the red dash; the legend title gives the run count. A
run whose task list differs from the others is left out with a note, and the
figure is skipped below two runs. `--only plasticity_share_cohort_violin`
draws `multitask_plasticity_share_cohort_violin_n.png`, the same pooled cells
per task as a Gaussian kernel density (scipy's Scott bandwidth, cells within
the y-limits only, every column scaled to the same peak width so shapes
compare rather than counts) under the same run and pooled medians; cells
beyond the limits stay triangles at the edge, and a task with fewer than
three cells within them shows its points. The share compares two
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
it relates the Euclidean
distance between the two tasks' full-dimensional context-end centroids (hidden
state and W ⊙ M, the saved distance matrices) to the similarity of their
lesion-effect profiles: for neuron clusters the saved `causal_similarity` of
`causal_vs_activity_tasksim_hidden_<aname>.pkl` (Pearson correlation of the
random-minus-lesion effects over all unnormalized input and hidden neuron
clusters, the causal-vs-activity matrix), for synapse clusters the Pearson
correlation of the var-weighted **freeze_M** effects (`SYNAPSE_LESION_MODE`,
the lesion heatmap's intervention) in `normalized_lesion_effects_<aname>.pkl`
over the synapse clusters without the unresponsive class, whose label is read
from the lesion pickle in `multiple_tasks_perf/` (the class holds most
synapses, so its size-matched random control collapses the network and its
effect is strongly negative for every task). `paper_plot.py` skips results
whose synapse similarity was computed with another mode (results written
before the switch from zero_W, `schema_version=1`); rerun the script to
refresh them. Each (representation, cluster type) pair gets a two-sided
Spearman task-label permutation (Mantel) test (10,000 permutations, seed 0;
`core.state_space_geometry.mantel_spearman_permutation`), a negative rho
meaning that tasks whose context-end states lie closer depend on more similar
clusters, and a descriptive OLS guide line for the figure. Nothing is
recomputed from network activity, so it runs in seconds; the results go to
`state_space/state_space_vs_lesion_results.pkl` (one entry per run with the
pair coordinates, the tests with their null arrays, the guide lines and the
L2), which the state-space batch run clears like every top-level file there,
so rerun it after `state_space_structure.py` or `lesion_plot.py`.
`python figures/paper_plot.py --only state_space_vs_lesion` draws
`multitask_state_space_vs_lesion_n.png` for `LESION_ANAME`: a 2 × 2 grid
with one row per representation (hidden-state and W ⊙ M centroid distance)
and one column per cluster type (neuron clusters, synapse clusters), one
point per task pair, x the centroid distance, y the lesion-profile
similarity, each panel's saved rho and permutation p in its legend and the
script's saved OLS guide line, descriptive only as in the causal-vs-activity
scatter (the statistic is the rank correlation).
`--only state_space_vs_lesion_seeds` draws
`multitask_state_space_vs_lesion_seeds_n.png`: per cluster type, every
sibling run's rho for the hidden-state and the W ⊙ M distance, filled when
its saved permutation p is below 0.05, the median over runs as a dash and
under each tick the two-sided Wilcoxon signed-rank p of the runs' rhos
against zero, computed in `paper_plot.py` from the saved per-run rhos; it is
skipped below two runs with results.

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
responsive pool as its third step (`--control-pool`, `--control-repeats`)
and its fourth step plots both pools, `lesion.py`'s into
`multiple_tasks_norm/` and the responsive one into its own directory, and
`--resume` lets runs lesioned before the step existed get their controls and
plots without a second lesion pass. `paper_plot.py` reads
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

## Synapse tuning vs endpoint clusters

`python multiple_task/synapse_variance_decomposition.py` asks how much of a
synapse's task tuning is the cluster identity of its two endpoint neurons.
For every run with `cluster_info_<aname>.pkl` and `cluster_info_mod_<aname>.pkl`
(`--seed N --feature TAG` for one run, `--seed all --feature TAG` for a
cohort, no arguments for every run), it takes the active synapses of the
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
variance-based results. These scripts are not run by `run_pipeline.py`.
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
right, as means with SD whiskers over networks, trained filled and untrained
hollow as in the sharing bar figures. The figure carries no test; the save
message adds the medians, ranges and the raw two-sided Mann-Whitney p of
trained against untrained per representation and axis. A sharing result
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
five periods, including those with lower trained sharing. Both raw p and
adjusted q are shown; a small value supports greater sharing in trained networks.
This directional follow-up replaces this figure's earlier two-sided comparison.
Each network contributes one sharing fraction; task pairs are not independent
samples and the cohorts are not paired. The tests concern the distributions
of those fractions, not specifically the difference of their means. Zero
sharing is retained. Both cohorts are required. These comparisons are computed
from current v6 per-network counts while plotting; once clustering and sharing
have been migrated, this extra figure needs no further summary rerun:

```bash
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_period_synapses
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
