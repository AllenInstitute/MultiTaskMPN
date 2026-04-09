# MultiTaskMPN

Research codebase for training and analyzing **Multi-Plastic Networks (MPNs)** on a battery of cognitive tasks. The core idea is that a single network with Hebbian-like synaptic plasticity can learn to solve many tasks simultaneously, and the structure of its plastic weights can be analyzed to understand how task-specific computation is organized.

---

## Model

The central model is `DeepMultiPlasticNet` ([mpn.py](mpn.py)), a recurrent network with one `MultiPlasticLayer` (`mp_layer1`) whose effective weights are modulated by a fast-timescale plasticity matrix **M**:

```
W_eff(t) = W + W ⊙ M(t)          (multiplicative)
         = W + M(t)               (additive)
```

**M** evolves by a Hebbian-like rule with learnable parameters:

| Parameter | Symbol | Description |
|---|---|---|
| Learning rate | η (eta) | Scales the Hebbian update |
| Decay | λ (lam) | Controls timescale of synaptic memory |

Both η and λ can be scalar, pre-vector, post-vector, or full matrix.

The full network (`DeepMultiPlasticNet`) has three weight matrices:
- `W_initial_linear` — input projection (pre-synaptic neurons)
- `mp_layer1.W` — recurrent plastic weights (hidden neurons)
- `W_output` — readout

---

## Workflow

```
Training → Analysis → Clustering → Lesion / Pruning
```

### 1. Training

```bash
python multiple_task.py
```

Trains the MPN on a set of cognitive tasks defined in `mpn_tasks.py`. Saves:
- `multiple_tasks/savednet_{aname}.pt` — model checkpoint
- `multiple_tasks/param_{aname}_param.json` — hyperparameters
- `multiple_tasks/param_{aname}_result.npz` — training curves

Key hyperparameters (set inside the script):
- `hidden` — number of recurrent units
- `batch` — batch size
- `seed` — random seed
- `feature` — regularization config (e.g. `L21e4`)

### 2. Post-training Analysis

```bash
python multiple_task_analysis.py
```

Loads a trained model, evaluates it on all tasks, and produces:
- Task-conditioned activity matrices
- Cluster analysis of input and hidden neurons
- Low-dimensional (PCA) trajectory plots
- Saves `cluster_info_{aname}_normalized.pkl` for downstream use

### 3. Clustering

```bash
python clustering.py
```

Implements hierarchical clustering (`clustering_metric.py`) with silhouette-score-based automatic selection of the number of clusters `k`. Clusters neurons by their task-tuning profiles.

### 4. Lesion & Pruning Analysis

```bash
python leison.py
```

Given a trained model and its cluster assignments, runs two experiments:

**Cluster lesion**: for each identified neuron cluster, zeros out all connections to/from that cluster and measures per-task accuracy. Pre-synaptic ("pre") and post-synaptic ("post") clusters are each lesioned independently in leave-one-out fashion.

**Random lesion**: lesions a size-matched random set of neurons (repeated 10×) as a control.

**Magnitude pruning**: zeros the lowest-magnitude fraction of `mp_layer1.W` at increasing sparsity levels (0–99.9%) to assess how much of the plastic weight matrix is functionally necessary.

Results are saved to `multiple_tasks_perf/lesion_prune_results_{aname}.pkl`.

### 5. State Space Analysis

```bash
python state_space_shift.py
```

Analyzes how the network's hidden-state geometry shifts across tasks using PCA and subspace angles.

---

## Key Files

| File | Purpose |
|---|---|
| `mpn.py` | Model definitions (`MultiPlasticLayer`, `DeepMultiPlasticNet`) |
| `mpn_tasks.py` | Task definitions and trial generators |
| `net_helpers.py` | Base network classes, weight initialization |
| `multiple_task.py` | Training loop |
| `multiple_task_analysis.py` | Post-training analysis and clustering pipeline |
| `clustering.py` | Hierarchical clustering with automatic `k` selection |
| `clustering_metric.py` | Cluster quality metrics |
| `leison.py` | Lesion and pruning experiments |
| `leison_plot.py` | Plotting utilities for lesion results |
| `state_space_shift.py` | State space / PCA analysis |
| `helper.py` | Shared utilities |
| `color_func.py` | Color palettes for plotting |

---

## Output Directories

| Directory | Contents |
|---|---|
| `multiple_tasks/` | Checkpoints, training curves, cluster info |
| `multiple_tasks_perf/` | Lesion/pruning heatmaps and result pickles |
| `state_space/` | State space figures |

---

## Requirements

- Python 3.9+
- PyTorch (CUDA optional, detected automatically in `leison.py`)
- NumPy, SciPy, scikit-learn
- Matplotlib, seaborn
- h5py, hdf5plugin
- scienceplots (for analysis notebooks)

---

## Naming Convention

Model checkpoints and result files use a shared identifier string:

```
{task}_seed{seed}_{feature}+hidden{hidden}+batch{batch}{accfeature}
# e.g. everything_seed749_L21e4+hidden300+batch128+angle
```

All analysis scripts read `aname` from this pattern to locate the correct files.
