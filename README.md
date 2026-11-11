# MultiTaskMPN

Training and analysis of **Multi-Plastic Networks (MPNs)** on a battery of
cognitive tasks. A single feedforward network with Hebbian-like synaptic
plasticity, whose only memory is the state of its plastic synapses, learns
many tasks at once; the structure of its plastic weights is
then analyzed to understand how task-specific computation is organized.

## Model

The core model is `DeepMultiPlasticNet` ([core/mpn.py](core/mpn.py)): a
feedforward network (input projection, plastic layer, readout; the hidden
state carries no memory of its own, see [SCHEME.md](SCHEME.md)) whose
effective weights are modulated by a fast plasticity matrix **M**:

```
W_eff(t) = W + W ⊙ M(t)     (multiplicative)   or   W + M(t)   (additive)
```

**M** evolves by a Hebbian rule with learning rate η and decay λ (each scalar,
pre/post-vector, or full matrix). The multi-task, pretraining and flex-task
scripts learn η; the single- and two-task scripts keep it fixed. Every script
fixes λ = 1 − Δt/`m_time_scale`, with Δt = 40 ms. The network has three
weight matrices: `W_initial_linear` (input projection), `mp_layer1.W`
(plastic weights from the projected input to the hidden layer) and
`W_output` (readout). The optional recurrent hidden-to-hidden term
(`recurrent_layer_add`) is off in every saved cohort.

| Experiment | hidden | batch | L2 | `m_time_scale` (λ) |
|---|---|---|---|---|
| multi-task (`multiple_task/`) | 300 | 128 | 1e-4 (cohorts from 1e-2 to 1e-5) | 4000 ms (0.99); 400000 ms (`lam0.9999`), 40000 ms (`lam0.999`), 2000 ms (`lam0.98`), 800 ms (`lam0.95`) and 400 ms (`lam0.9`) at L2 = 1e-4 |
| single-task (`one_task/`) | 200 | 128 | 1e-4 | 400 ms (0.9) |
| two-task (`two_task/`) | 200 | 128 | 1e-3 | 4000 ms (0.99) |
| pretraining (`pretrain/`) | 200 | 128 | 1e-3 | 400 ms (0.9); historical cohort 4000 ms (0.99) |
| flex-task (`flex_task/`) | 200 | 128 | 1e-4 | 4000 ms (0.99) |

Hyperparameters are set at the top of each training script and encoded in the
run name (see [Outputs](#outputs)). [SCHEME.md](SCHEME.md) explains the timing
convention, where each model's time constant lives and the figure colors.

## Tasks

The multi-task ruleset `everything` holds 15 tasks, named in the figures as
in parentheses: the go family `fdgo` (DelayPro), `fdanti` (DelayAnti),
`delaygo` (MemoryPro), `delayanti` (MemoryAnti), `reactgo` (ReactPro),
`reactanti` (ReactAnti); the decision-making family `delaydm1`
(IntegrationModality1), `delaydm2` (IntegrationModality2), `contextdelaydm1`
(ContextIntModality1), `contextdelaydm2` (ContextIntModality2),
`multidelaydm` (IntegrationMultimodal); and the matching family `dmsgo`
(ReactMatch2Sample), `dmsnogo` (ReactNonMatch2Sample), `dmcgo`
(ReactCategoryPro), `dmcnogo` (ReactCategoryAnti). The single-task, two-task
and pretraining experiments train subsets of the go family.

## Layout

Run every script from the repository root; data is written to the top-level
data directories. Experiment scripts import the shared library in `core/`
through a small `_bootstrap.py` shim.

| Path | Contents |
|---|---|
| `core/` | Shared library: model (`mpn`), tasks (`mpn_tasks`), training (`net_helpers`, `networks`), clustering, seed selection, utilities |
| `one_task/` | Single-task training, analysis, pipeline |
| `two_task/` | Two-task training, analysis, pipeline |
| `multiple_task/` | Multi-task training, clustering analysis, lesions, state space, pipeline |
| `pretrain/` | Pretraining → post-training transfer experiment and analysis |
| `flex_task/` | Flexible-task (RNN/MPN) training and analysis |
| `figures/` | Publication figures (`paper_plot.py`) and schematic cartoons (`cartoon.py`) |
| `biology/` | Biological reference figures: MICrONS connectivity, Allen Synaptic Physiology dynamics |
| `docs/` | Per-analysis documentation (see [Documentation](#documentation)) |
| `test/` | Unit tests (`python -m unittest discover -s test`) |

## Installation

Python 3.9+, PyTorch (CUDA optional), NumPy, SciPy, scikit-learn,
scikit-image, Matplotlib, seaborn, plotly, h5py, hdf5plugin, scienceplots,
psutil; the `biology/` scripts also need pandas. Training and the lesion
experiments need a GPU node. Historical pipeline timing was about five hours
per run plus lesion controls (one to two hours per synapse clustering type);
the revised trial-time clustering has not yet been benchmarked.

## Quick start

Each experiment family follows **train → analyze → figures**. The multi-task,
single-task, two-task and pretraining scripts add `N_TRIALS` new seeds of the
setting configured in the script and never overwrite a saved run, so rerun
one to grow a cohort; the first three take `--n-trials N` to change the count
and `--seed S [S ...]` to train explicit seeds, while `pretraining.py` reads
`N_TRIALS` and `SEED_LIST` from its header. `multiple_task.py` also takes
`--m-time-scale MS` to train another modulation retention.

```bash
# Multi-task: train, then cluster, lesion and score the lesion controls of
# the checkpoints needing current pipeline outputs (hours per run)
python multiple_task/multiple_task.py
python multiple_task/run_pipeline.py --feature L21e4 --skip-completed
# Reuse current steps; rebuild stale/missing results and their dependents.
# Old S1-conditioned caches need reclustering (see docs/multitask_analysis.md).
python multiple_task/run_pipeline.py --feature L21e4 --skip-completed --resume
# Post-analysis of the whole cohort from the saved outputs: lesion plots of
# both control pools, row-sharing summary, synapse variance decomposition and
# the state-space-vs-lesion link. Rerun it after more runs finish or after a
# change of the post-processing code (--stages, --seed, --dry-run).
python multiple_task/run_pipeline_analysis.py --feature L21e4
# Untrained controls: ten initializations, using seed 749's default config.
python multiple_task/untrained_clustering.py --feature L21e4 --seed 749 --init-seed 0 1 2 3 4 5 6 7 8 9 --skip-completed
# After the corresponding clustering batches finish, regenerate sharing summaries.
python multiple_task/sibling_row_sharing.py --feature L21e4
python multiple_task/sibling_row_sharing.py --feature L21e4 --untrained
# state-space geometry of the saved runs and its link to the lesions
python multiple_task/state_space_shift.py
python multiple_task/state_space_structure.py
python multiple_task/state_space_vs_lesion.py
# or every state-space analysis and figure end to end (see docs/state_space.md)
python multiple_task/run_state_space_pipeline.py
# sibling-task fixed-point geometry over every cached run (after
# sibling_delay_analysis.py, see docs/state_space.md)
python multiple_task/sibling_cohort.py --feature L21e4
# share of synapse tuning explained by the endpoint neurons' clusters (about
# five minutes per run; also the variance stage of run_pipeline_analysis.py)
python multiple_task/synapse_variance_decomposition.py --seed all --feature L21e4 --skip-completed

# Single- and two-task (train + analyze)
python one_task/run_one_task_pipeline.py
python two_task/run_two_task_pipeline.py

# Pretraining transfer: train, then analyze the cohort of one feature label
python pretrain/pretraining.py
python pretrain/pretraining_analysis.py --feature L21e3lam0.9
python pretrain/pretraining_post.py --feature L21e3lam0.9

# Paper figures: everything, one mode, or one figure
python figures/paper_plot.py
python figures/paper_plot.py multiple_tasks lesion --multitask-l2 1e4
# lesion figures from the activity-matched controls, into paper_plot_responsive/
python figures/paper_plot.py lesion --multitask-l2 1e4 --control-pool responsive
python figures/paper_plot.py pretraining --pretraining-lambda 0.9
python figures/paper_plot.py --only state_space_vs_lesion
```

## Outputs

Checkpoints and result files of a run share one identifier, which the
analysis scripts parse to find the run's files. Multi-task and single-task
runs use

```
{task}_seed{seed}_{feature}+hidden{hidden}+batch{batch}{accfeature}
# e.g. everything_seed749_L21e4+hidden300+batch128+angle
```

Two-task runs put the L2 tag and the width in one addon
(`delaygofamily_seed21_reg1e3+hidden200`); pretraining runs name the
pretraining ruleset and the network, and put the feature after the width
with the λ tag when λ ≠ 0.99
(`fdanti_delaygo_dmpn_seed109_+hidden200+L21e3lam0.9+batch128+angle`).

| Directory | Contents |
|---|---|
| `multiple_tasks/` | Multi-task checkpoints, recorded activity, learning curves |
| `multiple_tasks_analysis/` | Clustering results and per-run analysis figures |
| `multiple_tasks_perf/`, `multiple_tasks_norm/`, `multiple_tasks_norm_responsive/` | Lesion results (with the activity-matched controls of `lesion_controls.py`), lesion plots with `lesion.py`'s controls and with the activity-matched ones |
| `state_space/`, `two_in_multiples/` | State-space and sibling-task geometry caches |
| `onetask/`, `twotasks/` | Single- and two-task runs and analyses |
| `pretraining/`, `pretraining_analysis/` | Pretraining runs and analysis caches |
| `paper_plot/`, `paper_plot_responsive/` | Publication figures (the second holds the lesion figures rendered from the activity-matched controls) |
| `cartoon_plot/` | Schematic cartoons (`figures/cartoon.py`) |
| `log/` | stdout/stderr copies of the training, pipeline and pretraining runs (`core/run_logging.py`) |

## Documentation

- [docs/multitask_analysis.md](docs/multitask_analysis.md): clustering caches,
  the lesion figures and their statistics, the modulation clustering variants.
- [docs/state_space.md](docs/state_space.md): state-space and sibling-task
  geometry analyses and figures.
- [docs/pretraining.md](docs/pretraining.md): the transfer experiment, its
  figure options and output directories.
- [docs/biology.md](docs/biology.md): the data behind the biological
  reference panels of the introductory figure (MICrONS connectivity, Allen
  Synaptic Physiology dynamics) and which panels are schematics.
- [docs/paper_figures.md](docs/paper_figures.md): which figure and cache
  each panel of the composed paper figures comes from.
- [SCHEME.md](SCHEME.md): timing convention and figure color scheme.

## Acknowledgements

Parts of this codebase were written with the assistance of
[Claude Code](https://claude.ai/claude-code).
