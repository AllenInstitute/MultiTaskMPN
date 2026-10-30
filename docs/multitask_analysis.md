# Multi-task clustering and lesion analyses

Details behind the `multiple_tasks` and `lesion` figure modes of
`paper_plot.py`: which caches each figure reads, what its statistics are, and
the conventions of the clustering caches. `python multiple_task/run_pipeline.py`
produces those caches (clustering, lesions, lesion plots); the commands are in
the README.

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
# every checkpoint of a cohort that has no pipeline outputs yet (rerun after
# more training). A run counts as completed when all four steps wrote their
# last file: cluster_info and cluster_info_mod, lesion_prune_results, the
# lesion_controls pickle of --control-pool (responsive by default) and the
# two lesion_interaction pickles; a run missing any is redone from step 1,
# or with --resume only its missing steps are run (how runs lesioned before
# the controls step existed get their controls and plots).
# --skip-completed needs --feature or --seed; to spread a cohort over nodes,
# give each process its own --seed list (two bare runs would pick the same
# checkpoints). Each run takes about an hour of clustering plus four to six
# hours of lesions and one to two hours of controls per synapse type.
python multiple_task/run_pipeline.py --feature L21e4 --skip-completed
python multiple_task/run_pipeline.py --feature L21e4 --skip-completed --resume
```

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
behavior of clearing top-level `paper_plot/` outputs; `--only` preserves other
figures.

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
no block identity per point, so the point is found by its OM value, which the
scatter copies from the same `om_stack`; the highlight is skipped with a note
unless exactly one point matches.
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
and lesion caches (`multiple_tasks_norm/<aname>/`), it relates the Euclidean
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
runs' post-processing outputs in `multiple_tasks_norm/`.

## Activity-matched controls

`lesion.py` compares every cluster lesion with size-matched random lesions
drawn from all units of that side or layer. More than half of the hidden
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
lesion pickle in memory (`core/lesion_cache.overlay_lesion_controls`): every
single-cluster analysis (normalized effects, causal dependency, cluster
correlation, plasticity share, OM profiles, causal-vs-activity, task
specificity and sharing) then reads the activity-matched controls, while the
combined lesions keep `lesion.py`'s: the OM scatters and the OM profile
prediction compare synapse-cluster effects under the chosen pool with
combined effects under `lesion.py`'s (their caches record
`combined_control_pool`), and the interaction map takes its single effects
from `lesion.py`'s pickle so that its three terms share one control.
`state_space_vs_lesion.py` records the pool of the caches it reads
(`control_pool` per run). Its default, `lesion_plot.DEFAULT_CONTROL_POOL`
(`all`), reproduces the historical analyses; `run_pipeline.py` computes and
reads the responsive pool (`--control-pool`, `--control-repeats`) as its
third and fourth steps, and `--resume` lets runs lesioned before the step
existed get their controls and plots without a second lesion pass. Every
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
sign alone accounts for about a fifth of the signed variance) and `signed` =
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
# their row-cluster organization
python multiple_task/untrained_clustering.py --feature L21e4 --seed 749 --init-seed 0 1 2 3 4
python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained
# paper figures: rows grouped by period or by task, and the per-task and
# per-period views
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_organization
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_task
python figures/paper_plot.py --multitask-l2 1e4 --only row_cluster_same_period
```

Three pairs are checked: ReactCategoryPro vs ReactCategoryAnti (dmcgo/dmcnogo),
ContextIntModality1 vs 2 and IntegrationModality1 vs 2, in every trial phase
both tasks have. For each pair, phase and representation the result is the
number of runs whose two rows share a row cluster. Each count is read against
a chance level: the fraction of pairs of rows of that phase from two
different tasks that share a cluster in that run and representation,
averaged over the runs (`_same_phase_chance`), with an exact two-sided
Poisson-binomial test over the runs' own chance levels
(`_poisson_binomial_p`). The chance level matters: on the L21e4 runs every
memory-period row of every task falls in one input and one hidden cluster
(chance 100%), so sharing in the memory periods is not specific to the
sibling pair, while the modulation chance stays between about 7% and 20%.
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
periods) is reported but not tested. A third check asks whether rows are
grouped by task rather than by period: for every task of the pairs and every
two of its phases, whether the task's two rows share a row cluster, against
the chance that a row of the one phase and a row of the other from two
different tasks share one (`_cross_phase_chance`, so the phase combination is
controlled for), pooled per task and per group and tested the same way;
cells whose own chance is at least 95% (the two memory periods in input and
hidden) are left out and counted. It writes
`multiple_tasks_diagnosis/sibling_row_sharing_<tag>.{pkl,json}`, the
per-pair figure `sibling_row_sharing_pairs_<tag>.png`, the summary figure
`sibling_row_sharing_summary_<tag>.png` and the same-task figure
`sibling_row_sharing_same_task_<tag>.png`, replacing only its own tag's
files. A star marks a per-run excess that differs from zero in either
direction; the bar against its chance dash gives the direction.
On the 25 L21e4 runs the pooled pro / anti stimulus cells sit at chance in
every representation because the pairs disagree: DelayPro / DelayAnti and
MemoryPro / MemoryAnti share their stimulus rows in input and hidden (15-17
of 25 runs, chance about 33%) but not in modulation (2-3 of 25), the
category pair the other way round (2 of 25 in input and hidden, 16 of 25 in
modulation), and the match pair in none. Within a period, input and hidden
group the rows by period and input (every task's memory rows in one
cluster), modulation by neither. Across periods, modulation keeps a task's
rows together: 71% of the pro / anti tasks' phase pairs and 48% of the
modality tasks' share a cluster, against chances of 11% and 10%, while input
and hidden stay near chance (20% and 24% against 15% and 18% for pro / anti;
16% and 27% against 9% and 20% for the modality tasks). DelayPro and
DelayAnti are the exception: their stimulus and response rows share an input
cluster in every run (the stimulus stays on) but a modulation cluster in
none. The modulation keeps a 4 s memory of the trial (lambda = 0.99), longer
than a period, so part of this grouping may be the trial's history carried
in M rather than a separate task code.

The organization check (`summarize_organization`) puts period and task on
one scale with every task's rows: per run and representation, the fold
enrichment of two rows of one period from different tasks, and of two rows
of one task from different periods, sharing a row cluster over any two rows
sharing one (the chance the run's cluster sizes give, so the
representations, with their different numbers of row clusters, compare),
tested against 1 over the runs and, as task minus period enrichment, between
representations (paired Wilcoxon). On the 25 L21e4 runs the medians are, for
period and task, input 1.85x and 0.90x, hidden 1.57x and 0.95x, synapses
1.10x and 3.51x, and every pair of representations differs (p <= 7.5e-5).
The same-task check is also run over every task (`same_task_all`; ReactPro
and ReactAnti have one clustered period and so no cells). Its mirror, the
same-period check (`same_period_all`), asks for every phase and every two
tasks with a row of it whether the two rows share a row cluster, against the
chance that a row of the one task and a row of the other from two different
phases share one (`_cross_period_chance`, so how alike the two tasks are
anyway is controlled for), pooled per phase and tested the same way; cells at
chance of at least 95% are left out (on the L21e4 runs 118 input, 92 hidden
and 445 synapse cells: task pairs whose rows sit in one cluster in every
period). On the 25 L21e4 runs the period grouping of input and hidden is
mostly the memory periods: every two tasks' Memory 1 and Memory 2 rows share
a cluster (input 100%, hidden 99-100%, chance 18-31%), because nothing in the
input varies there. Stimulus rows share more than chance but far less
(input 32% and 39% against 15% and 22%, hidden 32% and 46% against 21% and
31%), and Response rows hardly (input 13% against 16%, below chance; hidden
27% against 21%). Synapse rows share 2-20% against 4-8% in every period. The
untrained networks show the same pattern (memory 100%, stimulus 38-55%,
response 20-25%). Result version 4 holds all three checks.

`paper_plot.py` (`multiple_tasks` mode) draws three figures from
`multiple_tasks_diagnosis/sibling_row_sharing_<MULTITASK_L2 feature>.pkl`,
skipped with the command to run when it is missing or older than version 4:
`multitask_row_cluster_organization.png`, the two enrichments side by side
(same period, different task; same task, different period), one point per
run in the representation colors, the median as a red dash, chance (1) as a
dashed line on a log axis and the Wilcoxon p against 1 above each column
(the Mann-Whitney p against the untrained networks once they exist, below);
and `multitask_row_cluster_same_task.png`, the per-task view of its right
panel: for every task with two clustered periods, three bars giving the % of
its (run, period pair) cells in one row cluster, a dash at the
period-combination chance of the same-task check (a stricter baseline than
the organization figure's) and a star where the per-run excess differs from
zero in either direction; and `multitask_row_cluster_same_period.png`, the
per-period view of its left panel: for every period, three bars giving the %
of its (run, task pair) cells in one row cluster with the task-pair chance
of the same-period check, the period labels in the heatmap phase colors.

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
reader looks. `--skip-completed` leaves out the networks whose two cluster
pickles exist; no lesion step runs. `sibling_row_sharing.py --untrained`
then analyzes those networks (tag `<feature>_untrained`). An untrained
network of seed 749's config is not degenerate: hidden std 0.12 (trained
0.16), M std 0.38, and up to about a quarter of its synapses reach their bounds
late in long trials, as in the trained network (`modulation_feedback.py`). The
organization figure draws the untrained networks, hollow with a gray median,
right of the trained runs in each representation's column, with the
two-sided Mann-Whitney p of trained against untrained above the column. The
per-task and per-period figures give every trained bar a hollow twin on its
right with the untrained networks' % and chance (a Wilcoxon star over the
networks needs at least six of them). All three draw the trained runs alone,
with the commands to run, while the untrained result is missing.

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

`multiple_task/rule_cue_ablation.py` asks whether the rule cue is what makes
the clustering rows group as they do (input and hidden rows by period,
synapse rows by task, in trained and untrained networks alike; see
[Sibling-task row-cluster sharing](#sibling-task-row-cluster-sharing)). The
cue is on for the whole trial, so it is a constant that adds no variance to
the input and hidden features, while M integrates it into a task-specific
drift. The script runs one network, trained or untrained
(`--untrained-init-seed`, `network_loading.build_network`), on the evaluation
trials of `modulation_feedback.py` (50 per task, `mode_input="random"`, trial
seed 0) twice, once as they are and once with every rule-cue channel set to
zero (the cue and its input noise), and computes the input, hidden and
modulation features of both as the clustering does: per task and period, the
variance over stimulus conditions and time of the stimulus-conditioned mean
(`helper.task_variance_period_numpy`), each column divided by its maximum
over the rows (the `_normalized` features of the heatmaps), the rows in the
clustering's period-major order and windows.

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
(result version 2): the distance matrices and pair means per condition and
representation, the matched-row correlations, the feature ratios and the
features themselves.

So far only the untrained network of seed 749's config (init seed 0) has
been analyzed, on trial seeds 0 and 2, which agree. Without the cue the
synapse rows keep their task grouping (same-task relative distance 0.62 with
the cue, 0.63 without; same-period 1.02 and 0.98), the input rows are
unchanged (same-period 0.71 and 0.68, same-task 1.03 in both) and the hidden
rows group more by period (same-period 0.85 to 0.75, same-task 0.96 to
1.02); the features themselves change most in the modulation (matched-row r
median 0.33, summed features x0.76) and least in the input (0.60, x0.99).
The cue is therefore not what groups the synapse rows by task. The trained
run has not been analyzed, and the remaining candidate, M's carry-over of a
trial's earlier periods (lambda = 0.99, a 4 s memory), is the question of
the lambda = 0.9 cohort (`L21e4lam0.9`) and of an untrained network with
lambda overridden, neither of which has been run through this script.
