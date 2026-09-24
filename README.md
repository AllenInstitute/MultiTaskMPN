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

**M** evolves by a Hebbian rule with learnable learning-rate η and decay λ (each
scalar, pre/post-vector, or full matrix). The network has three weight matrices:
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
# lesion effects and cross-seed lesion summaries (project spelling: leison)
python paper_plot.py leison
```

The `leison` figure mode contains `lesion_heatmap`, `lesion_cluster_sizes`,
`cluster_corr_vs_lesion`, `om_vs_lesion`, and `cross_seed_summary`.
These no longer run under `multiple_tasks`; use
`python paper_plot.py multiple_tasks leison` for both groups. Existing figure
names and `--only FIGURE` commands are unchanged. Mode runs retain the existing
behavior of clearing top-level `paper_plot/` outputs; `--only` preserves other
figures.

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
