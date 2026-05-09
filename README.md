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
Pretraining → Post-training Transfer
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

Given a trained model and its cluster assignments, runs lesion experiments using a **fixed number of clusters** (`FIXED_K`, inferred from the upstream `multiple_task_analysis.py` pickle). The same dendrogram is cut at this fixed k for input, hidden, and modulation clusters, ensuring consistent granularity across all analyses.

#### Experiments

**Single-cluster lesion (input & hidden)**: For each neuron cluster (both normalized and unnormalized variants), zeros out all connections to/from that cluster and measures per-task accuracy. Input ("pre") and hidden ("post") clusters are each lesioned independently in leave-one-out fashion.

**Random lesion**: For each cluster lesion condition, lesions a size-matched random set of neurons as a control. The normalized lesion effect is computed as `random_accuracy - cluster_accuracy`.

**Combined lesion (input × hidden)**: Simultaneously lesions one input cluster and one hidden cluster for all (pre_i, post_j) combinations. Random combined lesion serves as control.

**Modulation lesion**: For each modulation synapse cluster (derived from `col_labels_by_k[FIXED_K]`), two modes are tested:
- `zero_W`: zeros the static weight W at cluster synapses (removes connectivity)
- `freeze_M`: keeps W intact but freezes plasticity M at those synapses (removes learning)

**Magnitude pruning**: Zeros the lowest-magnitude fraction of `mp_layer1.W` at increasing sparsity levels (0–99.9%) to assess how much of the plastic weight matrix is functionally necessary.

#### Outputs

Results are saved to `multiple_tasks_perf/{aname}/lesion_prune_results_{aname}.pkl`.

### 4b. Lesion Plotting & Normalized Analysis

```bash
python leison_plot.py
```

Post-processes the lesion results to compute normalized effects and cross-analyses:

- **Normalized lesion heatmaps**: `random - cluster` effect for input/hidden and modulation clusters
- **Combined heatmaps**: Side-by-side `zero_W` vs `freeze_M` with shared color scale
- **Violin plots**: Distribution of normalized effect across tasks per cluster
- **Cluster similarity vs lesion effect**: Correlates cluster tuning similarity with functional lesion similarity (tests whether similar clusters have similar roles)
- **Overmembership vs lesion difference**: Relates modulation cluster enrichment in (input, hidden) pairs to the functional similarity between modulation lesion and combined lesion effects

Outputs are saved to `multiple_tasks_norm/{aname}/`.

### 5. State Space Analysis

```bash
python state_space_shift.py
```

Analyzes how the network's hidden-state geometry shifts across tasks using PCA and subspace angles.

### 6. Pretraining Transfer Experiment

```bash
python pretraining.py
```

Tests whether within-trial Hebbian plasticity can support learning a new task when all gradient-trained parameters are frozen except the task-indicator input column.

#### Protocol

1. **Stage 1 (Pretraining)**: Train a `DeepMultiPlasticNet` (200 hidden units) on a pair of tasks (e.g. `fdgo` + `delaygo`) until convergence (~60k datasets, with early stopping). All parameters are trainable. The input layer `W_initial_linear` is created with one extra column (zero-padded) to reserve space for the post-training task indicator.
2. **Stage 2 (Post-training)**: Freeze all parameters via `expand_and_freeze(option=1)`. Only the last column of `W_initial_linear` — the task-indicator-to-hidden weights for the new task — is trainable (via a gradient hook that masks all other columns). Train on a held-out task (e.g. `delayanti`) for 80k datasets. The plasticity matrix **M** still evolves within-trial via the Hebbian rule (eta, lam), but eta, lam, the static recurrent weight W, and the output layer W_output are all frozen.

The input is 9-dimensional: 6 stimulus/fixation channels + 3 task indicator channels (2 for pre-training tasks, 1 for post-training task). Each stage zero-pads the other stage's task indicator slots.

The experiment repeats over 5 random seeds. A sanity-check assertion verifies that all input weights except the last column remain unchanged between stages.

#### Key parameters

- Ruleset: `fdgo_delaygo` (pretraining) → `delayanti` (post-training)
- Hidden units: 200
- Input layout: `[fix1, fix2, r1cos, r1sin, r2cos, r2sin, task1, task2, task3]` — task slots are zero-padded per stage
- Stage 2 trainable parameters: last column of `W_initial_linear` only (200 weights)
- L2 regularization: 1e-4

#### Outputs (saved to `pretraining/`)

| File pattern | Contents |
|---|---|
| `savednet_{ruleset}_{net}_{seed}_{addon}.pt` | Network checkpoint (both stages) |
| `param_{ruleset}_{seed}_{addon}_param.json` | Hyperparameters |
| `param_{ruleset}_{net}_{seed}_{addon}_result.npz` | Hidden states, modulation, activations |
| `output_{ruleset}_{net}_{seed}_{addon}_stage{1,2}.npz` | Validation outputs per stage |
| `loss_*.png`, `lowD_*.png`, `input_prepost_*.png` | Diagnostic figures |

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
| `pretraining.py` | Pretraining → post-training transfer experiment |
| `helper.py` | Shared utilities |
| `color_func.py` | Color palettes for plotting |

---

## Output Directories

| Directory | Contents |
|---|---|
| `multiple_tasks/` | Checkpoints, training curves, cluster info |
| `multiple_tasks_perf/` | Lesion/pruning heatmaps and result pickles |
| `state_space/` | State space figures |
| `pretraining/` | Pretraining transfer experiment outputs |

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

---

## Acknowledgements

Parts of this codebase were written with the assistance of [Claude Code](https://claude.ai/claude-code).
