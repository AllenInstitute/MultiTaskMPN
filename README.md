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
python multiple_task/multiple_task_analysis.py
python multiple_task/leison.py
python multiple_task/leison_plot.py
python multiple_task/state_space_shift.py

# single- / two-task (train + analyze chained by the pipeline)
python one_task/run_one_task_pipeline.py
python two_task/run_two_task_pipeline.py

# pretraining transfer
python pretrain/pretraining.py

# paper figures
python paper_plot.py
```

Key data outputs: `multiple_tasks/` (checkpoints, curves, cluster info),
`multiple_tasks_perf/` and `multiple_tasks_norm/` (lesion results/plots),
`onetask/`, `twotasks/`, `pretraining/`, `state_space/`, `paper_plot/`.

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
