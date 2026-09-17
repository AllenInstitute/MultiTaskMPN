"""
Post-hoc analysis of the pretraining → post-training transfer experiment.

Compare novel-task variance in pretraining and novel-task PCA bases, without
held-out cross-validation. Analyze hidden states, M, W*M, principal angles,
rule vectors, and learning curves. For random-rule backbone probes, raw
pretraining-span grids, and norm-matched span-direction sweeps, use
pretraining_post.py --backbone-probe.

Run from the repository root (defaults: --hidden 200 --feature L21e3):
    python pretrain/pretraining_analysis.py
    python pretrain/pretraining_analysis.py --total-seed 3
    python pretrain/pretraining_analysis.py --hidden 100 --feature L21e4

Without --total-seed, every matching seed is analyzed. With --total-seed K,
K seeds are randomly selected from each ruleset using the same stable
(test-seed, ruleset) mapping as pretraining_post.py. Reads ./pretraining/ and
writes per-seed results and aggregate data
to ./pretraining_analysis/, aggregate figures to ./pretrain/fig/, and per-seed
figures to ./pretrain/fig_seed/; matching outputs overwrite.
"""

import argparse
import numpy as np
import pickle
import os
import re
import copy
from functools import lru_cache, wraps
from time import perf_counter

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
from scipy.stats import beta as beta_distribution

if __package__:
    from . import _bootstrap
else:
    import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import mpn

if __package__:
    from .pretraining_utils import (
        FINAL_TASK,
        RULESET_SPECS,
        build_task_layout,
        display_rule,
        positive_int,
        run_name,
        select_seeds as _select_seeds,
        stage1_tasks_for,
        variant_addon,
    )
else:
    from pretraining_utils import (
        FINAL_TASK,
        RULESET_SPECS,
        build_task_layout,
        display_rule,
        positive_int,
        run_name,
        select_seeds as _select_seeds,
        stage1_tasks_for,
        variant_addon,
    )

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20',
          '#319795', '#718096', '#d53f8c', '#d69e2e'] * 10

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,  
    "ps.fonttype": 42,
})


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
basepath = "./pretraining"
outpath = "./pretraining_analysis"
figpath = "./pretrain/fig"
seed_figpath = "./pretrain/fig_seed"
TIMING_ENABLED = True


def _timed_analysis(function):
    """Print inclusive wall time; nested timings must not be added together."""
    @wraps(function)
    def measured(*args, **kwargs):
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            if TIMING_ENABLED:
                detail = kwargs.get("datatype", "")
                print(f"  [timing] {function.__name__} {detail}: "
                      f"{perf_counter() - started:.3f} s", flush=True)
    return measured

# Existing outputs are preserved across runs; new outputs only overwrite
# files that share the same name. Aggregate figures embed `addon_name` in
# their filenames so different model variants (e.g. L2 strength, batch
# size) coexist in the same folder.
os.makedirs(outpath, exist_ok=True)
os.makedirs(figpath, exist_ok=True)
os.makedirs(seed_figpath, exist_ok=True)

# Period-basis choices are specific to this analysis. Shared task composition
# and display labels live in pretraining_utils.py.
BASIS_TASKS_BY_RULESET = {
    "fdgo_delaygo": {
        "stimulus": "fdgo", "response": "delaygo",
    },
    "fdanti_delaygo": {
        "stimulus": "fdanti", "response": "delaygo",
    },
    "fdanti": {
        "stimulus": "fdanti", "response": "fdanti",
    },
}

# Each ruleset is processed independently; learning/transfer curves are then
# combined into cross-ruleset figures. Keep this explicit order independent of
# the shared metadata catalog so refactors cannot silently reorder panels.
ANALYSIS_RULESET_ORDER = ("fdgo_delaygo", "fdanti_delaygo", "fdanti")

# Active ruleset / stage-1 tasks are (re)assigned at the top of each
# iteration of the main loop below. Functions that build file paths
# read `ruleset` from module scope at call time.
ruleset = None
stage1_tasks = None


def _basis_tasks_for(rs):
    """Stage-1 task whose activity defines each period's comparison basis."""
    try:
        return dict(BASIS_TASKS_BY_RULESET[rs])
    except KeyError as exc:
        raise ValueError(f"Unknown pretraining ruleset: {rs!r}") from exc


# Post-training task
final_task = FINAL_TASK


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", default="L21e3")
    parser.add_argument("--hidden", type=positive_int, default=200)
    parser.add_argument(
        "--total-seed", type=positive_int, default=None,
        help=("Randomly select K matching seeds from each ruleset; "
              "default: analyze every matching seed."),
    )
    parser.add_argument(
        "--test-seed", type=int, default=0,
        help="Random seed for reproducible --total-seed selection (default: 0).",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.test_seed <= 2**32 - 3:
        parser.error("--test-seed must be between 0 and 2**32 - 3")
    return args


# Per-ruleset plotting colors for the cross-ruleset combined figure.
ruleset_colors = {
    "fdgo_delaygo": c_vals[1],
    "fdanti_delaygo": c_vals[2],
    "fdanti": c_vals[4],
}

chosen_network = "dmpn"
N = 200
# PCA component cap for modulation / modulation_weighted analyses.
# Hidden requests up to N components. Modulation has bottleneck*proj features;
# call sites cap the requested components by both sample and feature counts.
N_MOD_PCS = 1000

# Principal angles reuse the leading directions of the CVE PCA fits.
# Return at most N_ANGLES values in ascending order, limited by the supported
# ranks of both bases (zero = shared direction, pi/2 = orthogonal).
N_ANGLES = 20

# Naming components that form addon_name in pretraining.py:
#   addon_name = f"+hidden{N}+{feature}+batch{batch}+{metric}"
metric = "angle"
feature = "L21e3"
addon_name = variant_addon(N, feature, metric=metric)

# ─────────────────────────────────────────────────────────────────────────────
# Path construction (matches pretraining.py naming convention)
# ─────────────────────────────────────────────────────────────────────────────
def _run_name(seed):
    """Canonical run name for the active ruleset and model variant."""
    return run_name(
        ruleset, chosen_network, seed, N, feature, batch=128, metric=metric)


def build_paths(seed):
    """Build file paths for a given seed, matching pretraining.py output naming."""
    base = _run_name(seed)
    return {
        "stage1_output": f"{basepath}/output_{base}_stage1.npz",
        "stage2_output": f"{basepath}/output_{base}_stage2.npz",
        "param_result": f"{basepath}/param_{base}_result.npz",
    }


def hist_path(seed):
    """Full training-history pickle path (see pretraining.py)."""
    return f"{basepath}/hist_{_run_name(seed)}.pkl"


def ckpt_path(seed):
    """Network checkpoint path (see pretraining.py)."""
    return f"{basepath}/savednet_{_run_name(seed)}.pt"


@lru_cache(maxsize=1)
def _load_checkpoint(path):
    """Cache one CPU checkpoint; clear before each seed analysis."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_mpn_W(seed):
    """Load frozen plastic-layer weights with their saved dimensions."""
    ckpt = _load_checkpoint(ckpt_path(seed))
    return ckpt["state_dict"]["mp_layer1.W"].numpy()


def _validate_saved_task_layout(seed, stage1_output, stage2_output):
    """
    Read the authoritative saved task metadata and validate the padded layout.

    pretraining.py saves both stages after padding them to the final network
    input width. That width is sensory columns through ``rule_start``, followed
    by every stage-1 rule cue and then the held-out stage-2 rule cue. It is 9
    for the two-parent motifs and 8 for the one-parent fdanti ablation.
    """
    task_params1 = stage1_output["task_params"].item()
    task_params2 = stage2_output["task_params"].item()
    shared_layout = build_task_layout(
        task_params1, task_params2, ruleset, context=f"seed {seed}")
    saved_stage1_tasks = shared_layout["stage1_rules"]
    saved_stage2_tasks = shared_layout["stage2_rules"]
    rule_start = shared_layout["rule_start"]
    expected_input_dim = shared_layout["input_width"]
    stage1_input = np.asarray(stage1_output["test_input_np"])
    stage2_input = np.asarray(stage2_output["test_input_np"])
    input_dims = (stage1_input.shape[-1], stage2_input.shape[-1])
    if input_dims != (expected_input_dim, expected_input_dim):
        raise ValueError(
            f"seed {seed}: padded input widths are {input_dims}, expected "
            f"{expected_input_dim} from rule_start={rule_start}, "
            f"{len(saved_stage1_tasks)} stage-1 rule(s), and "
            f"{len(saved_stage2_tasks)} stage-2 rule(s)"
        )

    test_task = np.asarray(stage1_output["test_task"], dtype=int)
    task_masks = {
        task: test_task == task_idx
        for task_idx, task in enumerate(saved_stage1_tasks)
    }
    empty_tasks = [task for task, mask in task_masks.items() if not np.any(mask)]
    if empty_tasks:
        raise ValueError(f"seed {seed}: no saved test trials for {empty_tasks}")

    basis_by_period = _basis_tasks_for(ruleset)
    missing_basis = set(basis_by_period.values()) - set(saved_stage1_tasks)
    if missing_basis:
        raise ValueError(
            f"seed {seed}: analysis basis task(s) {sorted(missing_basis)} are "
            f"not present in saved stage-1 tasks {saved_stage1_tasks}"
        )

    return {
        "stage1_tasks": saved_stage1_tasks,
        "stage2_tasks": saved_stage2_tasks,
        "basis_by_period": basis_by_period,
        "task_masks": task_masks,
        "rule_start": rule_start,
        "input_dim": expected_input_dim,
    }


def load_rule_vectors(seed, stage1_task_names, novel_task_name, rule_start,
                      expected_input_dim):
    """Extract named rule-cue columns using the saved, validated input layout."""
    ckpt = _load_checkpoint(ckpt_path(seed))
    W_in = ckpt["state_dict"]["W_initial_linear.weight"].numpy()
    if W_in.shape[1] != expected_input_dim:
        raise ValueError(
            f"seed {seed}: checkpoint input width {W_in.shape[1]} does not "
            f"match saved padded input width {expected_input_dim}"
        )

    pretrained = {
        task: W_in[:, rule_start + task_idx].copy()
        for task_idx, task in enumerate(stage1_task_names)
    }
    novel_col = rule_start + len(stage1_task_names)
    return pretrained, W_in[:, novel_col].copy(), novel_task_name


def _rule_vector_stats(pretrained_vectors, v_novel, novel_task):
    """
    Per-seed scalar summaries for any number of pretrained rule vectors.

    ``in_span_fraction`` is retained as the raw geometric quantity. Because its
    chance level grows with span rank, ``in_span_excess_over_random`` compares
    squared projection to the exact isotropic random-subspace expectation r/d.
    This permits a one-dimensional DelayAnti span and two-dimensional motif
    spans to be compared without treating their different ranks as equivalent.
    """
    if not pretrained_vectors:
        raise ValueError("At least one pretrained rule vector is required")

    task_names = list(pretrained_vectors)
    vectors = [np.asarray(pretrained_vectors[task]) for task in task_names]
    v_novel = np.asarray(v_novel)
    if any(vector.shape != v_novel.shape for vector in vectors):
        raise ValueError("All pretrained and novel rule vectors must have the same shape")

    basis = np.stack(vectors, axis=1)
    span_rank = int(np.linalg.matrix_rank(basis))
    if span_rank:
        U, _, _ = np.linalg.svd(basis, full_matrices=False)
        orthonormal_basis = U[:, :span_rank]
        v_proj = orthonormal_basis @ (orthonormal_basis.T @ v_novel)
    else:
        v_proj = np.zeros_like(v_novel)

    norm_novel = np.linalg.norm(v_novel)
    in_span = float(np.linalg.norm(v_proj) / norm_novel) if norm_novel > 0 else float("nan")
    in_span_squared = in_span ** 2
    ambient_dim = int(v_novel.size)
    random_expected_squared = span_rank / ambient_dim
    if span_rank == 0 or span_rank >= ambient_dim or not np.isfinite(in_span):
        random_percentile = float("nan")
        excess_over_random = float("nan")
    else:
        # For a fixed vector and a uniformly random rank-r subspace in R^d,
        # squared projection follows Beta(r/2, (d-r)/2).
        random_percentile = float(beta_distribution.cdf(
            np.clip(in_span_squared, 0.0, 1.0),
            span_rank / 2,
            (ambient_dim - span_rank) / 2,
        ))
        excess_over_random = float(
            (in_span_squared - random_expected_squared)
            / (1.0 - random_expected_squared)
        )

    cos_novel_by_task = {
        task: _cosine_sim(v_novel, vector)
        for task, vector in zip(task_names, vectors)
    }
    cos_pretrained_pairs = {}
    for left_idx, left_task in enumerate(task_names):
        for right_idx in range(left_idx + 1, len(task_names)):
            right_task = task_names[right_idx]
            cos_pretrained_pairs[f"{left_task}__{right_task}"] = _cosine_sim(
                vectors[left_idx], vectors[right_idx]
            )

    result = {
        "pretrained_tasks": task_names,
        "novel_task": novel_task,
        "cos_novel_by_task": cos_novel_by_task,
        "cos_pretrained_pairs": cos_pretrained_pairs,
        "in_span_fraction": in_span,
        "in_span_squared": in_span_squared,
        "span_rank": span_rank,
        "ambient_dim": ambient_dim,
        "random_span_expected_squared": random_expected_squared,
        "random_span_percentile": random_percentile,
        "in_span_excess_over_random": excess_over_random,
        "norm_pretrained": {
            task: float(np.linalg.norm(vector))
            for task, vector in zip(task_names, vectors)
        },
        "norm_novel": float(norm_novel),
    }

    # Preserve legacy scalar keys for existing two-parent paper code while the
    # named schema above remains authoritative and supports one-parent runs.
    for task_idx, task in enumerate(task_names):
        result[f"cos_novel_pre{task_idx}"] = cos_novel_by_task[task]
        result[f"norm_pre{task_idx}"] = result["norm_pretrained"][task]
    if len(task_names) == 2:
        pair_key = f"{task_names[0]}__{task_names[1]}"
        result["cos_pre0_pre1"] = cos_pretrained_pairs[pair_key]

    return result


def load_final_net(seed, device):
    """Reconstruct the post-stage-2 checkpoint in eval mode on device."""
    ckpt = _load_checkpoint(ckpt_path(seed))
    net_params = copy.deepcopy(ckpt["net_params"])
    net = mpn.DeepMultiPlasticNet(net_params).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def _plot_input_output_panel(
    fig_path, test_input_np, test_output_np, net_out_np,
    task_names, test_task, n_trials=10,
):
    """
    Two-column figure: left = net output (solid) over ground truth (faded),
    right = input channels. One row per trial. Titles identify the task.

    test_input_np  (batch, time, n_input)
    test_output_np (batch, time, n_output)
    net_out_np     (batch, time, n_output)
    task_names     list of rule names for this stage (stage-1 or stage-2)
    test_task      per-trial integer index into task_names
    """
    n_trials = min(n_trials, test_input_np.shape[0])
    fig, axs = plt.subplots(n_trials, 2, figsize=(4 * 2, 2 * n_trials))
    if n_trials == 1:
        axs = axs[np.newaxis, :]

    for b in range(n_trials):
        tname = (display_rule(task_names[int(test_task[b])])
                 if task_names is not None else "?")
        # Output column: net (solid) vs ground truth (thick faded).
        for oi in range(test_output_np.shape[-1]):
            axs[b, 0].plot(net_out_np[b, :, oi], color=c_vals[oi], linewidth=1)
            axs[b, 0].plot(test_output_np[b, :, oi], color=c_vals[oi],
                           linewidth=5, alpha=0.25)
        axs[b, 0].set_title(f"{tname} — out vs target")
        axs[b, 0].set_ylim([-1.6, 1.6])

        # Input column: every channel.
        for ii in range(test_input_np.shape[-1]):
            axs[b, 1].plot(test_input_np[b, :, ii], color=c_vals[ii], alpha=0.9)
        axs[b, 1].set_title(f"{tname} — input")
        axs[b, 1].set_ylim([-1.6, 1.6])

    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


@_timed_analysis
def run_final_net_sanity_check(
    seed, device, stage1_output, stage2_output, n_trials=10,
):
    """
    Load the final (post-stage-2) network and run it on BOTH stages' saved
    first n_trials test inputs from each stage. The npzs contain padded task
    indicator dimensions for each stage (see pretraining.py), so we reuse
    them verbatim and only need to reconstruct the network. Saves two
    figures: a stage-1 view (final net on pretraining test inputs) and a
    stage-2 view (final net on post-training test inputs).
    Only plotted trials are inferred, in batches of at most eight.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    net = load_final_net(seed, device)

    # Stage 1: pretraining tasks, evaluated with the *final* (post-stage-2) net.
    ti1 = np.asarray(stage1_output["test_input_np"])[:n_trials]
    to1 = np.asarray(stage1_output["test_output_np"])[:n_trials]
    task_params1 = stage1_output["task_params"].item()
    tt1 = np.asarray(stage1_output["test_task"])[:n_trials]

    # Stage 2: post-training task, evaluated with the final net.
    ti2 = np.asarray(stage2_output["test_input_np"])[:n_trials]
    to2 = np.asarray(stage2_output["test_output_np"])[:n_trials]
    task_params2 = stage2_output["task_params"].item()
    tt2 = np.asarray(stage2_output["test_task"])[:n_trials]

    def _run_on(ti_np):
        # Limit inference memory by processing at most eight trials at once.
        out_chunks = []
        bsz = 8
        with torch.no_grad():
            for s in range(0, ti_np.shape[0], bsz):
                e = min(s + bsz, ti_np.shape[0])
                batch = torch.as_tensor(ti_np[s:e], dtype=torch.float, device=device)
                out_batch, _, _ = net.iterate_sequence_batch(batch, run_mode="minimal")
                out_chunks.append(out_batch.detach().cpu().numpy())
        return np.concatenate(out_chunks, axis=0)

    out1 = _run_on(ti1)
    out2 = _run_on(ti2)

    checkname = _run_name(seed)
    _plot_input_output_panel(
        f"{seed_figpath}/{checkname}_finalnet_on_stage1.png",
        ti1, to1, out1, task_params1["rules"], tt1, n_trials=n_trials,
    )
    _plot_input_output_panel(
        f"{seed_figpath}/{checkname}_finalnet_on_stage2.png",
        ti2, to2, out2, task_params2["rules"], tt2, n_trials=n_trials,
    )


def discover_seeds():
    """Find all available seeds by scanning the pretraining output directory."""
    pattern = re.compile(
        rf"param_{re.escape(ruleset)}_{re.escape(chosen_network)}"
        rf"_seed(\d+)_{re.escape(addon_name)}_result\.npz"
    )
    seeds = []
    for fname in os.listdir(basepath):
        m = pattern.match(fname)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis utilities
# ─────────────────────────────────────────────────────────────────────────────
def _participation_ratio(cov_mat):
    """
    Effective dimensionality of a real symmetric PSD covariance matrix.
    Use trace(C)^2 / ||C||_F^2 without an eigendecomposition. Scale before
    float64 accumulation to avoid squaring large covariance values directly.
    Unlike clipping eigenvalues, this assumes PSD input, as produced by callers.
    Returns 0 for zero variance, 1 for rank 1, and d for isotropic d-dimensional variance.
    Higher PR → representations spread across more dimensions.
    """
    covariance = np.asarray(cov_mat, dtype=np.float64)
    scale = np.max(np.abs(covariance), initial=0.0)
    if scale == 0:
        return 0.0
    scaled = covariance / scale
    trace = np.trace(scaled)
    squared_norm = np.einsum("ij,ij->", scaled, scaled)
    return float(trace ** 2 / squared_norm)


def _mean_M_over_period(Ms_period):
    """Average an already sliced M over trials and time, retaining matrix axes."""
    return Ms_period.mean(axis=(0, 1))


def _cosine_sim(A, B):
    """Cosine similarity between two tensors, flattened."""
    a, b = np.asarray(A).ravel(), np.asarray(B).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def _stage1_end_iteration(final_param, history=None):
    """Resolve the history iteration boundary, with legacy single-update fallback."""
    if "stage1_end_iter" in final_param:
        return int(np.asarray(final_param["stage1_end_iter"]).item())
    if history is not None:
        if "stage1_end_iter" in history:
            return int(history["stage1_end_iter"])
        if "iter" in history.get("stage1", {}):
            return int(history["stage1"]["iter"])
    legacy_stop = np.asarray(final_param["pretrain_stop"]).item()
    if legacy_stop is None:
        raise ValueError("Missing stage1_end_iter and stage1 history; cannot split stages.")
    return int(legacy_stop) + 1


def _angles_from_pca(pca_X, pca_Y, shape_X, shape_Y, k):
    """Return ascending principal angles in radians from two existing PCA fits.

    Use at most k leading components from each fit, discarding directions
    below a dtype- and data-shape-dependent singular-value tolerance. The
    result length is limited by the smaller supported rank; an empty basis
    produces an empty array. Zero means shared, pi/2 means orthogonal.
    """
    if k < 1:
        return np.array([])
    def supported_basis(pca, shape):
        singular_values = pca.singular_values_[:k]
        tolerance = np.finfo(singular_values.dtype).eps * max(shape) * singular_values[0]
        return pca.components_[:k][singular_values > tolerance].T

    basis_X = supported_basis(pca_X, shape_X)
    basis_Y = supported_basis(pca_Y, shape_Y)
    if basis_X.shape[1] == 0 or basis_Y.shape[1] == 0:
        return np.array([])
    # scipy returns angles in descending order (largest first). Reverse
    # so index 0 is the most-aligned (smallest) angle, matching the
    # "top-k shared directions" convention used in the rest of the file.
    return subspace_angles(basis_X, basis_Y)[::-1]


def _pr_from_data(X_c):
    """
    PR from centered data, using whichever Gram side is smaller.

    Covariance (X_c.T @ X_c) and kernel (X_c @ X_c.T) share the same non-zero
    eigenvalues, so PR is identical; forming the smaller Gram matrix reduces
    work and storage. Essential for modulation (n_features = n_hidden² ≫
    n_samples), harmless for hidden.
    """
    n, f = X_c.shape
    if f <= n:
        M = (X_c.T @ X_c) / n
    else:
        M = (X_c @ X_c.T) / n
    return _participation_ratio(M)


def period_slice(op, epochs, task_name, key, *, shift_percentage=0, mask=None):
    """
    Extract one task epoch from op (batch × time × features).
    shift_percentage skips the leading fraction of the epoch (e.g. 0.25
    discards the transient onset and focuses on the steady-state window).
    mask selects a subset of trials (e.g. trials belonging to one task).
    """
    start, end = epochs[task_name][key]
    shift = int((end - start) * shift_percentage)
    period = op[:, start + shift:end, ...]
    if mask is not None:
        period = period[mask]
    return period


@_timed_analysis
def pca_cross_variance(
    X, Y, n_components=None, center_on="X", datatype="hidden", *, angle_k=None,
):
    """
    Driscoll-style subspace-overlap analysis (see Driscoll et al.,
    Nature Neurosci. 2024, Fig. 6c/k captions).

    angle_k optionally reuses these PCA fits for principal angles; randomized
    leading directions can differ from a separate lower-rank fit. The angles
    field is in radians, ascending, with length capped by both supported ranks.
    Basis convention:
      X = basis data   — the task whose top-k PCs define the reference
                         subspace (typically a pretraining task).
      Y = target data  — the task whose variance is being measured in
                         each basis (typically the novel / post-training
                         task, e.g. delayanti = Driscoll's MemoryAnti).

    Two cumulative-variance-explained curves are computed on the target:
      cev_Y_self : Y's variance captured by Y's own PCs   (Driscoll "black")
                   — self-reference, truncated to n_components.
      cev_Y      : Y's variance captured by X's PCs       (Driscoll "purple")
                   — measures how much of the target lives in the basis
                   task's subspace; high means the two tasks share
                   representational geometry, low means they're orthogonal.
      cev_X      : X's variance captured by X's own PCs — auxiliary
                   (sanity check that the basis spans its own variance).

    PR_X, PR_Y  : participation ratio of each dataset in its native space.
    PR_Y_in_Xbasis : effective dim of Y after projection into X's basis;
                   lower than PR_Y means the shared subspace is narrower
                   than the target's full representation.

    datatype controls reshaping before PCA
    ───────────────────────────────────────
    "hidden"              — (batch×time, n_hidden): neuron activations
    "modulation"          — (batch×time, bottleneck*proj): M flattened
    "modulation_weighted" — (batch×time, bottleneck*proj): W⊙M flattened (caller
                            must pre-multiply M by W elementwise)
    """
    if datatype == "hidden":
        X2d = X.reshape(-1, X.shape[-1])
        Y2d = Y.reshape(-1, Y.shape[-1])
    elif datatype in ("modulation", "modulation_weighted"):
        X2d = X.reshape(-1, X.shape[-1] * X.shape[-2])
        Y2d = Y.reshape(-1, Y.shape[-1] * Y.shape[-2])
    else:
        raise ValueError(f"Unknown datatype: {datatype}")

    assert X2d.shape[1] == Y2d.shape[1]

    # --- PCA on X (the reference / basis task) ---------------------------
    # Randomized SVD is much faster than the full solver when we only need
    # the top k components (k ≪ min(n_samples, n_features)), especially for
    # modulation where n_features = n_hidden² is large.
    pca_X = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca_X.fit(X2d)
    evr_X = pca_X.explained_variance_ratio_
    cev_X = np.cumsum(evr_X)

    # --- PCA on Y in its own basis (Driscoll's "black" reference curve) --
    # Both PCA fits use the same component limit.
    pca_Y = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca_Y.fit(Y2d)
    evr_Y_self = pca_Y.explained_variance_ratio_
    cev_Y_self = np.cumsum(evr_Y_self)

    # --- Y projected into X's basis (Driscoll's "purple" cross curve) ----
    # np.var below removes constant offsets for either centering choice.
    if center_on == "X":
        Y_centered = Y2d - pca_X.mean_
    elif center_on == "Y":
        Y_centered = Y2d - Y2d.mean(axis=0, keepdims=True)
    else:
        raise ValueError('center_on must be "X" or "Y"')

    Y_proj = Y_centered @ pca_X.components_.T  # project Y into X's PC basis

    var_total_Y = np.var(Y_centered, axis=0, ddof=0).sum()
    if var_total_Y == 0:
        evr_Y = np.zeros(pca_X.components_.shape[0])
    else:
        # Fraction of Y's total variance captured by each of X's PCs.
        evr_Y = np.var(Y_proj, axis=0, ddof=0) / var_total_Y
    cev_Y = np.cumsum(evr_Y)

    result = {
        "evr_X": evr_X, "cev_X": cev_X,
        "evr_Y_self": evr_Y_self, "cev_Y_self": cev_Y_self,
        "evr_Y": evr_Y, "cev_Y": cev_Y,
    }
    X_c = X2d - X2d.mean(axis=0, keepdims=True)
    result["PR_X"] = _pr_from_data(X_c)
    Y_c = Y2d - Y2d.mean(axis=0, keepdims=True)
    result["PR_Y"] = _pr_from_data(Y_c)
    Y_proj_c = Y_proj - Y_proj.mean(axis=0, keepdims=True)
    cov_Yp = (Y_proj_c.T @ Y_proj_c) / Y_proj_c.shape[0]
    result["PR_Y_in_Xbasis"] = _participation_ratio(cov_Yp)
    if angle_k is not None:
        result["angles"] = _angles_from_pca(
            pca_X, pca_Y, X2d.shape, Y2d.shape,
            min(angle_k, min(X2d.shape) - 1, min(Y2d.shape) - 1),
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = _parse_args()
    N = args.hidden
    feature = args.feature
    addon_name = variant_addon(N, feature, metric=metric)
    output_addon_name = (
        addon_name if args.total_seed is None
        else f"{addon_name}_n{args.total_seed}"
    )
    period_shift_percentage = 1 / 4

    # Device for the sanity-check re-runs of the final (post-stage-2) net.
    # Falls back to CPU if CUDA isn't available.
    sanity_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine which analyses to run based on network type
    if chosen_network == "dmpn":
        analysis_types = ["hidden", "modulation", "modulation_weighted"]
    else:
        analysis_types = ["hidden"]

    # Accumulate per-ruleset seed results for the cross-ruleset combined figure.
    all_results_by_ruleset = {}

    for active_ruleset in ANALYSIS_RULESET_ORDER:
        # Rebind module-level ruleset/stage1_tasks so the helper functions
        # (build_paths, ckpt_path, hist_path, discover_seeds) see the right
        # names for this iteration.
        ruleset = active_ruleset
        stage1_tasks = stage1_tasks_for(active_ruleset)

        print(f"\n{'#'*60}")
        print(f"  Ruleset: {ruleset}")
        print(f"{'#'*60}")

        seeds = discover_seeds()
        if not seeds:
            print(
                f"  No matching result files found in {basepath}/ for "
                f"ruleset={ruleset}, network={chosen_network}, addon={addon_name}; "
                f"skipping this ruleset."
            )
            continue
        print(f"Found {len(seeds)} seeds: {seeds}")

        if args.total_seed is not None:
            seeds = _select_seeds(
                seeds, args.total_seed, args.test_seed, ruleset
            )
            print(f"Selected {ruleset} seeds: {seeds}", flush=True)

        all_seed_results = []

        for seed in seeds:
            _load_checkpoint.cache_clear()
            print(f"\n{'='*60}")
            print(f"  Processing seed {seed}")
            print(f"{'='*60}")

            paths = build_paths(seed)
            for key, path in paths.items():
                if not os.path.exists(path):
                    print(f"  WARNING: missing {key} → {path}, skipping seed")
                    break
            else:
                # All files exist, proceed
                stage1_output = np.load(paths["stage1_output"], allow_pickle=True)
                stage2_output = np.load(paths["stage2_output"], allow_pickle=True)
                final_param = np.load(paths["param_result"], allow_pickle=True)

                # The saved npzs, not the filename, are authoritative for task
                # count and cue-column layout. This is essential for the
                # one-parent DelayAnti condition.
                layout = _validate_saved_task_layout(
                    seed, stage1_output, stage2_output
                )
                stage1_tasks = layout["stage1_tasks"]
                basis_by_period = layout["basis_by_period"]
                task_masks = layout["task_masks"]
                stimulus_basis_task = basis_by_period["stimulus"]
                response_basis_task = basis_by_period["response"]
                stimulus_basis_mask = task_masks[stimulus_basis_task]
                response_basis_mask = task_masks[response_basis_task]

                # Sanity-check figure: load the post-stage-2 network and run
                # it on both stages' test inputs (already correctly padded in
                # each stage's npz). Lets us visually verify the saved model
                # performs on both stages before trusting downstream PCA.
                try:
                    run_final_net_sanity_check(
                        seed, sanity_device, stage1_output, stage2_output,
                        n_trials=10,
                    )
                except (FileNotFoundError, KeyError, RuntimeError) as e:
                    print(f"  WARNING: final-net sanity check failed ({e}); "
                          f"continuing without it")

                seed_result = {
                    "seed": seed,
                    "stage1_tasks": list(stage1_tasks),
                    "final_task": final_task,
                    "basis_by_period": dict(basis_by_period),
                    "input_layout": {
                        "rule_start": layout["rule_start"],
                        "input_dim": layout["input_dim"],
                        "n_stage1_rules": len(stage1_tasks),
                        "n_stage2_rules": len(layout["stage2_tasks"]),
                    },
                }

                loading_started = perf_counter()
                stage1_hs = final_param["hs_stage1"]
                stage2_hs = final_param["hs_stage2"]
                stage1_ms = final_param["Ms_orig_stage1"]
                stage2_ms = final_param["Ms_orig_stage2"]
                saved_hidden = {int(stage1_hs.shape[-1]),
                                int(stage2_hs.shape[-1])}
                if saved_hidden != {N}:
                    raise ValueError(
                        f"seed {seed}: saved hidden widths {sorted(saved_hidden)} "
                        f"do not match --hidden {N}"
                    )

                stage1_rules_epochs = stage1_output["rules_epochs"].item()
                stage2_rules_epochs = stage2_output["rules_epochs2"].item()
                if TIMING_ENABLED:
                    print(f"  [timing] load state arrays and epochs: "
                          f"{perf_counter() - loading_started:.3f} s", flush=True)

                hp = hist_path(seed)
                hist = None
                if os.path.exists(hp):
                    with open(hp, "rb") as f:
                        hist = pickle.load(f)
                stop = _stage1_end_iteration(final_param, hist)
                acc_iter = final_param["valid_acc_iter"]
                acc = final_param["valid_acc"]
                # Exclude duplicate records at the stage boundary.
                post_mask = acc_iter > stop
                pre_mask = acc_iter < stop
                acc_iter_post = acc_iter[post_mask] - stop
                acc_post = acc[post_mask]
                acc_iter_pre = acc_iter[pre_mask]
                acc_pre = acc[pre_mask]

                # ---- Extract periods ----
                # period_shift_percentage skips the onset transient of each epoch
                # so the PCA captures steady-state geometry; applied consistently
                # to both stim and go periods.

                # Each period uses the scientifically specified Stage-1 basis.
                # For the DelayAnti condition, fdanti is the basis for both periods.
                stage1_stim = period_slice(
                    stage1_hs, stage1_rules_epochs, stimulus_basis_task, "stim1",
                    shift_percentage=period_shift_percentage,
                    mask=stimulus_basis_mask)
                final_stim = period_slice(
                    stage2_hs, stage2_rules_epochs, final_task, "stim1",
                    shift_percentage=period_shift_percentage)

                stage1_go = period_slice(
                    stage1_hs, stage1_rules_epochs, response_basis_task, "go1",
                    shift_percentage=period_shift_percentage,
                    mask=response_basis_mask)
                final_go = period_slice(
                    stage2_hs, stage2_rules_epochs, final_task, "go1",
                    shift_percentage=period_shift_percentage)

                # Hidden state analysis.
                # Driscoll convention: X = pretraining (basis), Y = novel
                # (delayanti). cev_Y = novel's variance in pretraining's PCs
                # (Driscoll "purple"); cev_Y_self = novel in its own PCs
                # (Driscoll "black").
                res_h_stim = pca_cross_variance(
                    stage1_stim, final_stim, n_components=N, datatype="hidden", angle_k=N_ANGLES)
                res_h_go = pca_cross_variance(
                    stage1_go, final_go, n_components=N, datatype="hidden", angle_k=N_ANGLES)
                seed_result["hidden"] = {
                    "stimulus": res_h_stim,
                    "response": res_h_go,
                    "angles_stimulus": res_h_stim.pop("angles"),
                    "angles_response": res_h_go.pop("angles"),
                }

                # Modulation analysis (dmpn only)
                if chosen_network == "dmpn" and stage1_ms.size > 0 and stage2_ms.size > 0:
                    stage1_stim_m = period_slice(
                        stage1_ms, stage1_rules_epochs, stimulus_basis_task, "stim1",
                        shift_percentage=period_shift_percentage,
                        mask=stimulus_basis_mask)
                    final_stim_m = period_slice(
                        stage2_ms, stage2_rules_epochs, final_task, "stim1",
                        shift_percentage=period_shift_percentage)
                    stage1_go_m = period_slice(
                        stage1_ms, stage1_rules_epochs, response_basis_task, "go1",
                        shift_percentage=period_shift_percentage,
                        mask=response_basis_mask)
                    final_go_m = period_slice(
                        stage2_ms, stage2_rules_epochs, final_task, "go1",
                        shift_percentage=period_shift_percentage)

                    # Clamp n_components to the smaller of the two slices'
                    # n_samples (sklearn errors if n_components > min(n, f)).
                    # Both slices must accommodate the same n_components since
                    # PCA is fit on each separately (self-reference + basis).
                    n_stim_samples = min(
                        final_stim_m.shape[0] * final_stim_m.shape[1],
                        stage1_stim_m.shape[0] * stage1_stim_m.shape[1])
                    n_go_samples = min(
                        final_go_m.shape[0] * final_go_m.shape[1],
                        stage1_go_m.shape[0] * stage1_go_m.shape[1])
                    n_stim_features = int(np.prod(stage1_stim_m.shape[-2:]))
                    n_go_features = int(np.prod(stage1_go_m.shape[-2:]))
                    n_comp_stim = min(
                        N_MOD_PCS, n_stim_samples, n_stim_features)
                    n_comp_go = min(
                        N_MOD_PCS, n_go_samples, n_go_features)

                    res_m_stim = pca_cross_variance(
                        stage1_stim_m, final_stim_m,
                        n_components=n_comp_stim, datatype="modulation", angle_k=N_ANGLES)
                    res_m_go = pca_cross_variance(
                        stage1_go_m, final_go_m,
                        n_components=n_comp_go, datatype="modulation", angle_k=N_ANGLES)
                    seed_result["modulation"] = {
                        "stimulus": res_m_stim,
                        "response": res_m_go,
                        "angles_stimulus": res_m_stim.pop("angles"),
                        "angles_response": res_m_go.pop("angles"),
                    }

                    # W⊙M analysis: multiplicative contribution of plasticity to
                    # W_eff. Stage 1 learns W; stage 2 freezes it, so the final
                    # checkpoint's W applies to both stages' saved M. No extra
                    # normalization is applied.
                    try:
                        W = load_mpn_W(seed)
                        # Broadcasting over (batch, time) axes.
                        stage1_stim_wm = stage1_stim_m * W
                        final_stim_wm = final_stim_m * W
                        stage1_go_wm = stage1_go_m * W
                        final_go_wm = final_go_m * W

                        res_wm_stim = pca_cross_variance(
                            stage1_stim_wm, final_stim_wm,
                            n_components=n_comp_stim,
                            datatype="modulation_weighted", angle_k=N_ANGLES)
                        res_wm_go = pca_cross_variance(
                            stage1_go_wm, final_go_wm,
                            n_components=n_comp_go,
                            datatype="modulation_weighted", angle_k=N_ANGLES)
                        seed_result["modulation_weighted"] = {
                            "stimulus": res_wm_stim,
                            "response": res_wm_go,
                            "angles_stimulus": res_wm_stim.pop("angles"),
                            "angles_response": res_wm_go.pop("angles"),
                        }
                    except (FileNotFoundError, KeyError) as e:
                        print(f"  WARNING: could not load W from checkpoint "
                              f"({e}); skipping modulation_weighted for this seed")

                    # ─── Period-matched M similarity ──────────────────────
                    # Every ruleset gets the two cross-stage comparisons that
                    # match its PCA bases. A within-stage-1 task-pair baseline
                    # exists only for genuine two-parent motifs; DelayAnti
                    # must not acquire a trivial fdanti-vs-itself baseline.
                    m_comparisons = []

                    def _append_m_comparison(
                        comparison_id, left_task, right_task, period,
                        comparison_type, left_period, right_period,
                    ):
                        left_mean = _mean_M_over_period(left_period)
                        right_mean = _mean_M_over_period(right_period)
                        m_comparisons.append({
                            "id": comparison_id,
                            "left_task": left_task,
                            "right_task": right_task,
                            "period": period,
                            "comparison_type": comparison_type,
                            "cos": _cosine_sim(left_mean, right_mean),
                            "frob": float(np.linalg.norm(left_mean - right_mean)),
                        })

                    _append_m_comparison(
                        "final_vs_stage1_stimulus_basis",
                        final_task, stimulus_basis_task, "stimulus",
                        "cross_stage", final_stim_m, stage1_stim_m,
                    )
                    _append_m_comparison(
                        "final_vs_stage1_response_basis",
                        final_task, response_basis_task, "response",
                        "cross_stage", final_go_m, stage1_go_m,
                    )

                    if len(stage1_tasks) >= 2:
                        task0, task1 = stage1_tasks[:2]
                        for period, epoch_key in (
                            ("stimulus", "stim1"), ("response", "go1")
                        ):
                            task0_period = period_slice(
                                stage1_ms, stage1_rules_epochs, task0, epoch_key,
                                shift_percentage=period_shift_percentage,
                                mask=task_masks[task0],
                            )
                            task1_period = period_slice(
                                stage1_ms, stage1_rules_epochs, task1, epoch_key,
                                shift_percentage=period_shift_percentage,
                                mask=task_masks[task1],
                            )
                            _append_m_comparison(
                                f"stage1_{task0}_vs_{task1}_{period}",
                                task0, task1, period, "within_stage1",
                                task0_period, task1_period,
                            )

                    seed_result["m_similarity"] = {
                        "comparisons": m_comparisons,
                    }
                elif chosen_network == "dmpn":
                    print("  WARNING: saved modulation arrays are empty; skipping modulation analyses")

                # Rule-input vector geometry. Cue columns are located from the
                # validated saved layout, so one- and two-parent conditions do
                # not silently slice different semantic channels.
                try:
                    pretrained_vectors, v_novel, novel_task = load_rule_vectors(
                        seed,
                        stage1_tasks,
                        final_task,
                        layout["rule_start"],
                        layout["input_dim"],
                    )
                    seed_result["rule_vectors"] = _rule_vector_stats(
                        pretrained_vectors, v_novel, novel_task
                    )
                except (FileNotFoundError, KeyError, ValueError) as e:
                    print(f"  WARNING: could not extract rule vectors "
                          f"({e}); skipping rule-vector analysis for this seed")

                # Learning curves
                seed_result["learning"] = {
                    "acc_iter_post": acc_iter_post,
                    "acc_post": acc_post,
                    "acc_iter_pre": acc_iter_pre,
                    "acc_pre": acc_pre,
                }

                # Stage2 history contains both stages; use the same boundary as accuracy.
                if hist is not None:
                    full_iter = np.asarray(hist["stage2"]["iters_monitor"])[1:]
                    full_out = np.asarray(hist["stage2"]["valid_loss_output_label"])[1:]
                    full_reg = np.asarray(hist["stage2"]["valid_loss_reg_term"])[1:]
                    n = min(len(full_iter), len(full_out), len(full_reg))
                    full_iter, full_out, full_reg = full_iter[:n], full_out[:n], full_reg[:n]

                    # Same split rule as accuracy: exclude the duplicated
                    # stage-boundary iter from both masks.
                    post_m = full_iter > stop
                    pre_m = full_iter < stop
                    seed_result["loss"] = {
                        "pre_iter": full_iter[pre_m],
                        "pre_out_loss": full_out[pre_m],
                        "pre_reg_loss": full_reg[pre_m],
                        "post_iter": full_iter[post_m] - stop,
                        "post_out_loss": full_out[post_m],
                        "post_reg_loss": full_reg[post_m],
                    }
                else:
                    print(f"  WARNING: missing training history {hp} — loss plot will skip this seed")

                all_seed_results.append(seed_result)

                # Save individual seed result
                checkname = _run_name(seed)
                with open(f"{outpath}/{checkname}_result.pkl", "wb") as f:
                    pickle.dump(seed_result, f)
                print(f"  Saved: {checkname}_result.pkl")

                # Per-seed PCA figure
                n_plots = len(analysis_types)
                fig, axs = plt.subplots(n_plots, 3, figsize=(9, 3 * n_plots))
                if n_plots == 1:
                    axs = axs[np.newaxis, :]

                period_to_stage1 = dict(basis_by_period)

                for row, dtype in enumerate(analysis_types):
                    if dtype not in seed_result:
                        continue
                    x_up = 20 if dtype == "hidden" else N_MOD_PCS

                    for period_idx, period in enumerate(["stimulus", "response"]):
                        res = seed_result[dtype][period]
                        xs = np.arange(1, len(res["cev_Y_self"]) + 1)
                        # Driscoll Fig. 6c/k convention:
                        #   black  = novel variance in novel PCs (self)
                        #   purple = novel variance in pretraining PCs (cross)
                        axs[row, period_idx].plot(
                            xs, res["cev_Y_self"], '-o', markersize=2,
                            color="black",
                            label=f"{display_rule(final_task)} in {display_rule(final_task)} PCs")
                        axs[row, period_idx].plot(
                            xs[:len(res["cev_Y"])], res["cev_Y"], '-o',
                            markersize=2, color=c_vals[3],
                            label=f"{display_rule(final_task)} in {display_rule(period_to_stage1[period])} PCs")
                        axs[row, period_idx].set_xlim(0, x_up)
                        axs[row, period_idx].set_ylim(0, 1.05)
                        axs[row, period_idx].set_xlabel("# PCs")
                        axs[row, period_idx].set_ylabel(
                            f"{display_rule(final_task)} variance explained")
                        axs[row, period_idx].set_title(f"{dtype} — {period}")
                        axs[row, period_idx].legend(fontsize=8)

                    # PR bar chart. Under Driscoll convention X = pretraining
                    # basis, Y = novel (delayanti). PR_X = pretraining's PR;
                    # PR_Y = novel's PR; PR_Y|X = PR of novel after projection
                    # into pretraining's basis. Bars grouped by period (stim
                    # first, then go) with a visible gap between groups.
                    res_stim = seed_result[dtype]["stimulus"]
                    res_go = seed_result[dtype]["response"]
                    pr_labels = [
                        "PR pre", f"PR {display_rule(final_task)}", f"PR {display_rule(final_task)}|pre",
                        "PR pre", f"PR {display_rule(final_task)}", f"PR {display_rule(final_task)}|pre",
                    ]
                    pr_vals = [res_stim["PR_X"], res_stim["PR_Y"], res_stim["PR_Y_in_Xbasis"],
                               res_go["PR_X"], res_go["PR_Y"], res_go["PR_Y_in_Xbasis"]]
                    # Tick positions: 0,1,2 (stim), then 4,5,6 (go) — gap at 3.
                    tick_positions = [0, 1, 2, 4, 5, 6]
                    colors = [c_vals[0], c_vals[3], c_vals[3],
                              c_vals[0], c_vals[3], c_vals[3]]
                    axs[row, 2].bar(tick_positions, pr_vals, width=0.7,
                                    color=colors, alpha=0.7)
                    axs[row, 2].set_xticks(tick_positions)
                    axs[row, 2].set_xticklabels(pr_labels, fontsize=7,
                                                rotation=35, ha="right")
                    # Period group annotations under the bars.
                    axs[row, 2].text(1, -0.18, "stim", transform=
                        axs[row, 2].get_xaxis_transform(),
                        ha="center", fontsize=8, fontweight="bold")
                    axs[row, 2].text(5, -0.18, "go", transform=
                        axs[row, 2].get_xaxis_transform(),
                        ha="center", fontsize=8, fontweight="bold")
                    axs[row, 2].set_ylabel("Participation Ratio")
                    axs[row, 2].set_title(f"{dtype} — dimensionality")

                fig.suptitle(f"Seed {seed} | {ruleset} | {chosen_network}", fontsize=12)
                fig.tight_layout()
                fig.savefig(f"{seed_figpath}/{checkname}_pca.png", dpi=300)
                plt.close(fig)
        # ─────────────────────────────────────────────────────────────────────
        # Aggregate across seeds (per-ruleset figures)
        # ─────────────────────────────────────────────────────────────────────
        if not all_seed_results:
            print(f"\nNo seeds processed successfully for {ruleset}. Skipping.")
            continue

        all_results_by_ruleset[ruleset] = all_seed_results
        print(f"\n{'='*60}")
        print(f"  Aggregating {len(all_seed_results)} seeds for {ruleset}")
        print(f"{'='*60}")

        # Cumulative variance explained summary
        n_plots = len(analysis_types)
        fig, axs = plt.subplots(n_plots, 2, figsize=(4 * 2, 4 * n_plots))
        if n_plots == 1:
            axs = axs[np.newaxis, :]

        period_to_stage1 = _basis_tasks_for(ruleset)

        for row, dtype in enumerate(analysis_types):
            x_up = 20 if dtype == "hidden" else N_MOD_PCS
            for col, period in enumerate(["stimulus", "response"]):
                all_self = []  # novel in its own basis (Driscoll black)
                all_cross = []  # novel in pretraining's basis (Driscoll purple)
                for sr_idx, sr in enumerate(all_seed_results):
                    if dtype not in sr:
                        continue
                    res = sr[dtype][period]
                    all_self.append(res["cev_Y_self"])
                    all_cross.append(res["cev_Y"])

                    xs_self = np.arange(1, len(res["cev_Y_self"]) + 1)
                    xs_cross = np.arange(1, len(res["cev_Y"]) + 1)
                    # Faded per-seed lines: thin for self (black-toned), colored per seed for cross.
                    axs[row, col].plot(xs_self, res["cev_Y_self"],
                                       color="black", alpha=0.2)
                    axs[row, col].plot(xs_cross, res["cev_Y"],
                                       color=c_vals[1 + sr_idx],
                                       alpha=0.6, label=f"seed {sr['seed']}")

                # Mean curves across seeds (thick lines).
                if all_self:
                    min_len = min(min(len(c) for c in all_self),
                                  min(len(c) for c in all_cross))
                    mean_self = np.mean([c[:min_len] for c in all_self], axis=0)
                    mean_cross = np.mean([c[:min_len] for c in all_cross], axis=0)
                    xs_mean = np.arange(1, min_len + 1)
                    axs[row, col].plot(xs_mean, mean_self, color="black",
                                       linewidth=2.5,
                                       label=f"{display_rule(final_task)} in {display_rule(final_task)} PCs (mean)")
                    axs[row, col].plot(xs_mean, mean_cross, color="gray",
                                       linewidth=2.5,
                                       label=f"{display_rule(final_task)} in {display_rule(period_to_stage1[period])} PCs (mean)")

                axs[row, col].set_xlim(0, x_up)
                axs[row, col].set_ylim(0, 1.05)
                axs[row, col].set_xlabel("# PCs")
                axs[row, col].set_ylabel(f"{display_rule(final_task)} variance explained")
                axs[row, col].set_title(f"{dtype} — {period}")
                axs[row, col].legend(fontsize=6)

        fig.suptitle(f"{ruleset} | {chosen_network} | {len(all_seed_results)} seeds", fontsize=12)
        fig.tight_layout()
        fig.savefig(
            f"{figpath}/{ruleset}_{chosen_network}_{output_addon_name}_aggregate.png",
            dpi=300,
        )
        plt.close(fig)

        # Save aggregate CVE data for paper_plot reuse
        aggregate_data = {"ruleset": ruleset, "analysis_types": analysis_types,
                          "periods": ["stimulus", "response"],
                          "final_task": final_task,
                          "stage1_tasks": list(stage1_tasks),
                          "basis_by_period": dict(period_to_stage1)}
        for dtype in analysis_types:
            for period in ["stimulus", "response"]:
                all_self, all_cross = [], []
                for sr in all_seed_results:
                    if dtype not in sr:
                        continue
                    res = sr[dtype][period]
                    all_self.append(res["cev_Y_self"])
                    all_cross.append(res["cev_Y"])
                if all_self:
                    min_len = min(min(len(c) for c in all_self),
                                  min(len(c) for c in all_cross))
                    aggregate_data[f"{dtype}_{period}_self"] = [c[:min_len] for c in all_self]
                    aggregate_data[f"{dtype}_{period}_cross"] = [c[:min_len] for c in all_cross]
        agg_pkl_path = (
            f"{outpath}/{ruleset}_{chosen_network}_"
            f"{output_addon_name}_aggregate.pkl"
        )
        with open(agg_pkl_path, "wb") as f:
            pickle.dump(aggregate_data, f)
        print(f"  Saved aggregate data: {agg_pkl_path}")

        # ─── Principal-angle spectra summary ───────────────────────────────
        # For each (datatype, period) pair, plot the top-k principal angles
        # between the pretraining and novel-task PC bases. Per-seed thin
        # lines + seed-mean thick line. Angle = 0 means the PC direction
        # is shared, π/2 means orthogonal. Complements the CVE panel above:
        # CVE is variance-weighted, angles are per-direction geometric.
        figpa, axspa = plt.subplots(n_plots, 2, figsize=(4 * 2, 3 * n_plots))
        if n_plots == 1:
            axspa = axspa[np.newaxis, :]
        for row, dtype in enumerate(analysis_types):
            for col, period in enumerate(["stimulus", "response"]):
                angle_key = f"angles_{period}"
                all_angles = []
                for sr_idx, sr in enumerate(all_seed_results):
                    if dtype not in sr or angle_key not in sr[dtype]:
                        continue
                    angs = np.asarray(sr[dtype][angle_key])
                    if angs.size == 0:
                        continue
                    all_angles.append(angs)
                    xs = np.arange(1, len(angs) + 1)
                    axspa[row, col].plot(
                        xs, np.degrees(angs),
                        color=c_vals[1 + sr_idx], alpha=0.4)

                if all_angles:
                    min_len = min(len(a) for a in all_angles)
                    mean_deg = np.mean(
                        [np.degrees(a[:min_len]) for a in all_angles], axis=0)
                    xs_mean = np.arange(1, min_len + 1)
                    axspa[row, col].plot(xs_mean, mean_deg, color="black",
                                         linewidth=2.5, label="mean")
                axspa[row, col].axhline(
                    0, color="gray", linestyle="--", linewidth=0.8)
                axspa[row, col].axhline(
                    90, color="gray", linestyle="--", linewidth=0.8)
                axspa[row, col].set_ylim([-5, 95])
                axspa[row, col].set_xlabel("shared-direction index (1 = most aligned)")
                axspa[row, col].set_ylabel("principal angle (deg)")
                axspa[row, col].set_title(f"{dtype} — {period}")
                axspa[row, col].legend(fontsize=6)

        figpa.suptitle(
            f"{ruleset} | {chosen_network} | principal angles "
            f"(pretraining vs {display_rule(final_task)})", fontsize=11)
        figpa.tight_layout()
        figpa.savefig(
            f"{figpath}/{ruleset}_{chosen_network}_"
            f"{output_addon_name}_principal_angles.png",
            dpi=300)
        plt.close(figpa)

        # ─── Per-ruleset accuracy + loss figure ────────────────────────────
        # Layout: rows = (post-training, pre-training); cols = (accuracy, loss).
        # Loss column is only populated for seeds whose training-history pkl exists.
        # Accuracy is plotted as a percentage (0–100).
        figlc, axslc = plt.subplots(2, 2, figsize=(4 * 2, 4 * 2))
        for sr_idx, sr in enumerate(all_seed_results):
            color = c_vals[1 + sr_idx]
            lc = sr["learning"]
            axslc[0, 0].plot(lc["acc_iter_post"], lc["acc_post"] * 100,
                             color=color, alpha=0.7, label=f"seed {sr['seed']}")
            axslc[1, 0].plot(lc["acc_iter_pre"], lc["acc_pre"] * 100,
                             color=color, alpha=0.7, label=f"seed {sr['seed']}")
            if "loss" in sr:
                ls = sr["loss"]
                axslc[0, 1].plot(ls["post_iter"], ls["post_out_loss"],
                                 color=color, alpha=0.7, label=f"seed {sr['seed']}")
                axslc[1, 1].plot(ls["pre_iter"], ls["pre_out_loss"],
                                 color=color, alpha=0.7, label=f"seed {sr['seed']}")

        axslc[0, 0].set_title("Post-training accuracy")
        axslc[0, 1].set_title("Post-training loss")
        axslc[1, 0].set_title("Pre-training accuracy")
        axslc[1, 1].set_title("Pre-training loss")
        axslc[0, 0].set_xlabel("# Datasets (post-training)")
        axslc[0, 1].set_xlabel("# Datasets (post-training)")
        axslc[1, 0].set_xlabel("# Datasets (pre-training)")
        axslc[1, 1].set_xlabel("# Datasets (pre-training)")
        for ax in axslc[:, 0]:
            ax.set_ylabel("Validation accuracy (%)")
            ax.set_ylim([0, 105])
        for ax in axslc[:, 1]:
            ax.set_ylabel("Validation loss")
            ax.set_yscale("log")
        for ax in axslc.flatten():
            ax.legend(fontsize=6)

        figlc.suptitle(f"{ruleset} | {chosen_network} | Learning curves", fontsize=12)
        figlc.tight_layout()
        figlc.savefig(
            f"{figpath}/{ruleset}_{chosen_network}_{output_addon_name}_learning.png",
            dpi=300,
        )
        plt.close(figlc)

        if not any("loss" in sr for sr in all_seed_results):
            print("No training-history pickles found; loss column in "
                  "learning figure will be empty.")

        # ─── Period-matched M similarity summary (per ruleset) ─────────────
        # Comparison records are dynamic: one-parent runs contain only the two
        # scientifically valid cross-stage comparisons, while two-parent runs
        # additionally contain the two within-stage-1 baselines.
        sim_seeds = [sr for sr in all_seed_results if "m_similarity" in sr]
        if sim_seeds:
            comparison_specs = sim_seeds[0]["m_similarity"]["comparisons"]
            comparison_ids = [entry["id"] for entry in comparison_specs]
            records_by_seed = {
                sr["seed"]: {
                    entry["id"]: entry
                    for entry in sr["m_similarity"]["comparisons"]
                }
                for sr in sim_seeds
            }
            sim_seeds = [
                sr for sr in sim_seeds
                if all(cid in records_by_seed[sr["seed"]]
                       for cid in comparison_ids)
            ]
            labels = [
                f"{display_rule(entry['left_task'])}"
                f"({entry['period']}) ↔ "
                f"{display_rule(entry['right_task'])}"
                f"({entry['period']})"
                for entry in comparison_specs
            ]

            def _render_msim_figure(cos_metric, frob_metric,
                                    title_tag, file_tag):
                """Build the 1×2 cosine/Frobenius-difference figure."""
                cos_values = {
                    cid: np.asarray([
                        records_by_seed[sr["seed"]][cid][cos_metric]
                        for sr in sim_seeds
                    ], dtype=float)
                    for cid in comparison_ids
                }
                frob_values = {
                    cid: np.asarray([
                        records_by_seed[sr["seed"]][cid][frob_metric]
                        for sr in sim_seeds
                    ], dtype=float)
                    for cid in comparison_ids
                }

                print(f"\n[{title_tag} {ruleset}, {len(sim_seeds)} seeds]")
                for cid in comparison_ids:
                    for metric, values in (
                        (cos_metric, cos_values[cid]),
                        (frob_metric, frob_values[cid]),
                    ):
                        print(
                            f"  {metric + '_' + cid:52s}  "
                            f"mean={values.mean():.4f}  "
                            f"std={values.std():.4f}  "
                            f"min={values.min():.4f}  "
                            f"max={values.max():.4f}"
                        )

                xs = np.arange(len(labels))
                figsim, axsim = plt.subplots(1, 2, figsize=(10, 3.5))
                for sr_idx, sr in enumerate(sim_seeds):
                    seed_records = records_by_seed[sr["seed"]]
                    axsim[0].plot(xs,
                                  [seed_records[cid][cos_metric]
                                   for cid in comparison_ids],
                                  "o-", color=c_vals[1 + sr_idx], alpha=0.7,
                                  label=f"seed {sr['seed']}")
                    axsim[1].plot(xs,
                                  [seed_records[cid][frob_metric]
                                   for cid in comparison_ids],
                                  "o-", color=c_vals[1 + sr_idx], alpha=0.7,
                                  label=f"seed {sr['seed']}")
                axsim[0].plot(xs,
                              [cos_values[cid].mean()
                               for cid in comparison_ids], "ks-",
                              linewidth=2, markersize=8, label="mean")
                axsim[1].plot(xs,
                              [frob_values[cid].mean()
                               for cid in comparison_ids], "ks-",
                              linewidth=2, markersize=8, label="mean")
                for ax in axsim:
                    ax.set_xticks(xs)
                    ax.set_xticklabels(labels, rotation=15)
                    ax.legend(fontsize=6)
                axsim[0].set_ylim([-0.05, 1.05])
                axsim[0].set_ylabel("Cosine similarity")
                axsim[0].set_title("M cosine similarity")
                axsim[1].set_ylim(bottom=0)
                axsim[1].set_ylabel("‖M_A − M_B‖ (Frobenius)")
                axsim[1].set_title("M difference magnitude")

                figsim.suptitle(f"{ruleset} | {chosen_network} | {title_tag}",
                                fontsize=11)
                figsim.tight_layout()
                figsim.savefig(
                    f"{figpath}/{ruleset}_{chosen_network}_"
                    f"{output_addon_name}_{file_tag}.png",
                    dpi=300)
                plt.close(figsim)

            _render_msim_figure(
                cos_metric="cos", frob_metric="frob",
                title_tag="period-matched M similarity",
                file_tag="m_similarity",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-ruleset combined accuracy + loss figure
    # Same 2×2 layout as the per-ruleset figure, but all seeds of each ruleset
    # share one color so the rulesets are visually separable.
    # ─────────────────────────────────────────────────────────────────────────
    def _mean_across_seeds(x_list, y_list, n_grid=200):
        """
        Average y(x) curves across seeds by interpolating each onto a common
        x-grid over their overlapping x range. Returns (grid, mean_y). If
        there's no overlap (e.g. stage-1 early-stopping differs too much),
        returns (None, None) and the caller should skip.
        """
        if not x_list:
            return None, None
        x_min = max(float(x[0]) for x in x_list)
        x_max = min(float(x[-1]) for x in x_list)
        if x_max <= x_min:
            return None, None
        grid = np.linspace(x_min, x_max, n_grid)
        interps = [np.interp(grid, x, y) for x, y in zip(x_list, y_list)]
        return grid, np.mean(interps, axis=0)

    if all_results_by_ruleset:
        figcmp, axscmp = plt.subplots(2, 2, figsize=(4 * 2, 4 * 2))
        for rs, seed_results in all_results_by_ruleset.items():
            color = ruleset_colors.get(rs, c_vals[0])

            # Accumulators for seed-mean curves.
            acc_post_x, acc_post_y = [], []
            acc_pre_x, acc_pre_y = [], []
            loss_post_x, loss_post_y = [], []
            loss_pre_x, loss_pre_y = [], []

            for sr in seed_results:
                lc = sr["learning"]
                axscmp[0, 0].plot(lc["acc_iter_post"], lc["acc_post"] * 100,
                                  color=color, alpha=0.3)
                axscmp[1, 0].plot(lc["acc_iter_pre"], lc["acc_pre"] * 100,
                                  color=color, alpha=0.3)
                acc_post_x.append(np.asarray(lc["acc_iter_post"]))
                acc_post_y.append(np.asarray(lc["acc_post"]) * 100)
                acc_pre_x.append(np.asarray(lc["acc_iter_pre"]))
                acc_pre_y.append(np.asarray(lc["acc_pre"]) * 100)

                if "loss" in sr:
                    ls = sr["loss"]
                    axscmp[0, 1].plot(ls["post_iter"], ls["post_out_loss"],
                                      color=color, alpha=0.3)
                    axscmp[1, 1].plot(ls["pre_iter"], ls["pre_out_loss"],
                                      color=color, alpha=0.3)
                    loss_post_x.append(np.asarray(ls["post_iter"]))
                    loss_post_y.append(np.asarray(ls["post_out_loss"]))
                    loss_pre_x.append(np.asarray(ls["pre_iter"]))
                    loss_pre_y.append(np.asarray(ls["pre_out_loss"]))

            # Mean curves (one label per ruleset in the legend).
            mean_panels = [
                (axscmp[0, 0], acc_post_x, acc_post_y),
                (axscmp[1, 0], acc_pre_x, acc_pre_y),
                (axscmp[0, 1], loss_post_x, loss_post_y),
                (axscmp[1, 1], loss_pre_x, loss_pre_y),
            ]
            label_used = False
            for ax, xs_list, ys_list in mean_panels:
                g, m = _mean_across_seeds(xs_list, ys_list)
                if g is None:
                    continue
                ax.plot(g, m, color=color, linewidth=2.5,
                        label=None if label_used else f"{rs} mean")
                label_used = True

        axscmp[0, 0].set_title("Post-training accuracy")
        axscmp[0, 1].set_title("Post-training loss")
        axscmp[1, 0].set_title("Pre-training accuracy")
        axscmp[1, 1].set_title("Pre-training loss")
        axscmp[0, 0].set_xlabel("# Datasets (post-training)")
        axscmp[0, 1].set_xlabel("# Datasets (post-training)")
        axscmp[1, 0].set_xlabel("# Datasets (pre-training)")
        axscmp[1, 1].set_xlabel("# Datasets (pre-training)")
        for ax in axscmp[:, 0]:
            ax.set_ylabel("Validation accuracy (%)")
            ax.set_ylim([0, 105])
            ax.set_yticks(np.arange(0, 101, 20))
        for ax in axscmp[:, 1]:
            ax.set_ylabel("Validation loss")
            ax.set_yscale("log")
        # Each ruleset attaches its label to only one panel (see `label_used`
        # above), so calling legend() unconditionally would put an empty box
        # on the other panels. Only call legend() when there's actually
        # something to label.
        for ax in axscmp.flatten():
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(handles, labels, fontsize=7)

        combined_tag = "_".join(all_results_by_ruleset.keys())
        figcmp.suptitle(f"{chosen_network} | Learning curves by ruleset", fontsize=12)
        figcmp.tight_layout()
        figcmp.savefig(
            f"{figpath}/{combined_tag}_{chosen_network}_"
            f"{output_addon_name}_learning.png",
            dpi=300,
        )
        plt.close(figcmp)

        # ─────────────────────────────────────────────────────────────────
        # Transfer-speed summary: iterations to first reach each accuracy
        # threshold during post-training. Lower = faster transfer. Per
        # seed + ruleset-mean, plotted as a function of threshold so you
        # can see where (if anywhere) the rulesets separate.
        # ─────────────────────────────────────────────────────────────────
        def _first_iter_to(iters, acc, threshold):
            """First iteration at which acc >= threshold (0-1 scale)."""
            iters = np.asarray(iters)
            acc = np.asarray(acc)
            hits = np.where(acc >= threshold)[0]
            return float(iters[hits[0]]) if hits.size else np.nan

        thresholds = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

        # Console summary.
        print("\n[Transfer speed — iterations to first reach threshold "
              "during post-training]")
        for rs, seed_results in all_results_by_ruleset.items():
            print(f"  {rs} ({len(seed_results)} seeds)")
            for th in thresholds:
                vals = [_first_iter_to(sr["learning"]["acc_iter_post"],
                                       sr["learning"]["acc_post"], th)
                        for sr in seed_results]
                vals = np.array(vals)
                n_reached = int(np.sum(~np.isnan(vals)))
                if n_reached > 0:
                    m = np.nanmean(vals)
                    s = np.nanstd(vals)
                    print(f"    acc >= {int(th*100):2d}%   reached by "
                          f"{n_reached}/{len(vals)}   mean iter = "
                          f"{m:7.1f} ± {s:6.1f}")
                else:
                    print(f"    acc >= {int(th*100):2d}%   reached by "
                          f"0/{len(vals)} seeds")

        # Figure: accuracy threshold (y) vs iterations-to-reach (x).
        # Reads like a learning curve — horizontal sweep from left to right
        # shows how fast each ruleset climbs to each threshold. One line
        # per ruleset; shaded band is ±1 std over seeds.
        figts, axts = plt.subplots(1, 1, figsize=(5, 3.8))
        ys = thresholds * 100
        for rs, seed_results in all_results_by_ruleset.items():
            color = ruleset_colors.get(rs, c_vals[0])
            per_seed_mat = np.asarray([
                [_first_iter_to(sr["learning"]["acc_iter_post"],
                                sr["learning"]["acc_post"], th)
                 for th in thresholds]
                for sr in seed_results
            ], dtype=float)
            # nanmean/nanstd in case a seed never hits the highest thresholds.
            mean_vals = np.nanmean(per_seed_mat, axis=0)
            std_vals = np.nanstd(per_seed_mat, axis=0)
            axts.plot(mean_vals, ys, "s-", color=color, linewidth=2.5,
                      markersize=7, label=f"{rs} (n={len(seed_results)})")
            # Horizontal ±1 std band at each threshold.
            axts.fill_betweenx(ys, mean_vals - std_vals, mean_vals + std_vals,
                               color=color, alpha=0.15)

        axts.set_xlabel("First post-training iter to reach threshold")
        axts.set_ylabel("Accuracy threshold (%)")
        axts.set_yticks(np.arange(0, 101, 10))
        axts.set_xscale("log")
        axts.set_title("Transfer speed — iterations to threshold")
        axts.legend(fontsize=7)

        figts.suptitle(f"{chosen_network} | Transfer speed by ruleset",
                       fontsize=12)
        figts.tight_layout()
        figts.savefig(
            f"{figpath}/{combined_tag}_{chosen_network}_"
            f"{output_addon_name}_transfer_speed.png",
            dpi=300)
        plt.close(figts)

        # Save transfer speed data for paper_plot reuse
        transfer_speed_data = {
            "thresholds": thresholds,
            "by_ruleset": {},
        }
        for rs, seed_results in all_results_by_ruleset.items():
            per_seed_mat = np.asarray([
                [_first_iter_to(sr["learning"]["acc_iter_post"],
                                sr["learning"]["acc_post"], th)
                 for th in thresholds]
                for sr in seed_results
            ], dtype=float)
            transfer_speed_data["by_ruleset"][rs] = {
                "per_seed_iters": per_seed_mat,
                "n_seeds": len(seed_results),
            }
        ts_pkl_path = (
            f"{outpath}/{combined_tag}_{chosen_network}_"
            f"{output_addon_name}_transfer_speed.pkl"
        )
        with open(ts_pkl_path, "wb") as f:
            pickle.dump(transfer_speed_data, f)
        print(f"  Saved transfer speed data: {ts_pkl_path}")

        # ─────────────────────────────────────────────────────────────────
        # Rule-input vector geometry, compared across rulesets.
        # Stage 2 adjusts only the novel task's input column. Compare that
        # vector with the pretrained columns to quantify its alignment with
        # their span; the vector width follows the checkpoint's projection size.
        # ─────────────────────────────────────────────────────────────────
        have_rule_vecs = any(
            "rule_vectors" in sr for rs_srs in all_results_by_ruleset.values()
            for sr in rs_srs)
        if have_rule_vecs:
            rs_list = [
                rs for rs, seed_results in all_results_by_ruleset.items()
                if any("rule_vectors" in sr for sr in seed_results)
            ]

            def _rule_vector_entries(rs):
                return [sr["rule_vectors"]
                        for sr in all_results_by_ruleset[rs]
                        if "rule_vectors" in sr]

            def _vals(rs, key):
                return np.asarray(
                    [entry[key] for entry in _rule_vector_entries(rs)],
                    dtype=float,
                )

            def _cosine_specs(rs):
                """(label, source-dict, source-key) for all valid pairs."""
                tasks = stage1_tasks_for(rs)
                specs = [
                    (
                        f"{display_rule(final_task)} ↔ {display_rule(task)}",
                        "cos_novel_by_task",
                        task,
                    )
                    for task in tasks
                ]
                for left_idx, left_task in enumerate(tasks):
                    for right_task in tasks[left_idx + 1:]:
                        specs.append((
                            f"{display_rule(left_task)} ↔ "
                            f"{display_rule(right_task)}",
                            "cos_pretrained_pairs",
                            f"{left_task}__{right_task}",
                        ))
                return specs

            def _nested_vals(rs, source, key):
                return np.asarray(
                    [entry[source][key]
                     for entry in _rule_vector_entries(rs)],
                    dtype=float,
                )

            # Console summary.
            print("\n[Rule-input vector geometry (stage-2 column vs "
                  "pretraining rule columns)]")
            for rs in rs_list:
                n_ok = len(_vals(rs, "in_span_fraction"))
                if n_ok == 0:
                    continue
                print(f"  {rs} ({n_ok} seeds)")
                for label, source, key in _cosine_specs(rs):
                    values = _nested_vals(rs, source, key)
                    print(
                        f"    cosine {label:28s}  "
                        f"mean={values.mean():.4f}  std={values.std():.4f}"
                    )
                for key in (
                    "in_span_fraction",
                    "in_span_excess_over_random",
                    "random_span_percentile",
                ):
                    values = _vals(rs, key)
                    print(
                        f"    {key:31s}  mean={values.mean():.4f}  "
                        f"std={values.std():.4f}"
                    )
                span_ranks = _vals(rs, "span_rank")
                print(f"    span_rank                       "
                      f"values={span_ranks.astype(int).tolist()}")

            # Figure: all valid pairwise cosine bars grouped by ruleset.
            # DelayAnti has one bar; two-parent motifs have three.
            # Error bars = std across seeds; black dots = per-seed values.
            figrv, axrv_cos = plt.subplots(1, 1, figsize=(7, 3.8))

            bar_width = 0.7
            all_x, all_labels = [], []
            cursor = 0

            for rs in rs_list:
                color = ruleset_colors.get(rs, c_vals[0])
                specs = _cosine_specs(rs)
                xs_group = cursor + np.arange(len(specs))
                values_by_pair = [
                    _nested_vals(rs, source, key)
                    for _, source, key in specs
                ]
                means = np.asarray([values.mean() for values in values_by_pair])
                stds = np.asarray([values.std() for values in values_by_pair])
                axrv_cos.bar(xs_group, means, bar_width,
                             yerr=stds, capsize=3, color=color,
                             alpha=0.8, label=RULESET_SPECS[rs]["label"])
                for pair_idx, values in enumerate(values_by_pair):
                    axrv_cos.plot(
                        np.full(values.shape, xs_group[pair_idx]), values,
                        "k.", markersize=3, alpha=0.6)
                all_x.extend(xs_group.tolist())
                all_labels.extend([label for label, _, _ in specs])
                cursor = int(xs_group[-1]) + 2

            axrv_cos.set_xticks(all_x)
            axrv_cos.set_xticklabels(all_labels, rotation=25, ha="right")
            axrv_cos.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
            axrv_cos.set_ylabel("Cosine similarity")
            axrv_cos.set_title("Rule-vector pairwise cosine")
            axrv_cos.legend(fontsize=7)

            figrv.suptitle(
                f"{chosen_network} | Rule-input vector analysis",
                fontsize=12)
            figrv.tight_layout()
            figrv.savefig(
                f"{figpath}/{combined_tag}_{chosen_network}_"
                f"{output_addon_name}_rule_vectors.png",
                dpi=300)
            plt.close(figrv)

            # Save a named schema that supports any number of Stage-1 rules.
            # Legacy flat keys remain present for two-parent paper code.
            rule_vec_data = {"schema_version": 2, "by_ruleset": {}}
            for rs in rs_list:
                s1_tasks = stage1_tasks_for(rs)
                entries = _rule_vector_entries(rs)
                rs_data = {
                    "cos_novel_by_task": {
                        task: _nested_vals(
                            rs, "cos_novel_by_task", task
                        ).tolist()
                        for task in s1_tasks
                    },
                    "cos_pretrained_pairs": {
                        key: _nested_vals(
                            rs, "cos_pretrained_pairs", key
                        ).tolist()
                        for key in entries[0]["cos_pretrained_pairs"]
                    },
                    "in_span_fraction": _vals(rs, "in_span_fraction").tolist(),
                    "in_span_squared": _vals(rs, "in_span_squared").tolist(),
                    "in_span_excess_over_random": _vals(
                        rs, "in_span_excess_over_random"
                    ).tolist(),
                    "random_span_expected_squared": _vals(
                        rs, "random_span_expected_squared"
                    ).tolist(),
                    "random_span_percentile": _vals(
                        rs, "random_span_percentile"
                    ).tolist(),
                    "span_rank": _vals(rs, "span_rank").astype(int).tolist(),
                    "ambient_dim": _vals(rs, "ambient_dim").astype(int).tolist(),
                    "norm_pretrained": {
                        task: _nested_vals(
                            rs, "norm_pretrained", task
                        ).tolist()
                        for task in s1_tasks
                    },
                    "norm_novel": _vals(rs, "norm_novel").tolist(),
                    "stage1_tasks": s1_tasks,
                    "final_task": final_task,
                }
                for task_idx in range(len(s1_tasks)):
                    rs_data[f"cos_novel_pre{task_idx}"] = _vals(
                        rs, f"cos_novel_pre{task_idx}"
                    ).tolist()
                    rs_data[f"norm_pre{task_idx}"] = _vals(
                        rs, f"norm_pre{task_idx}"
                    ).tolist()
                if len(s1_tasks) == 2:
                    rs_data["cos_pre0_pre1"] = _vals(
                        rs, "cos_pre0_pre1"
                    ).tolist()
                rule_vec_data["by_ruleset"][rs] = rs_data
            rv_pkl_path = (
                f"{outpath}/{combined_tag}_{chosen_network}_"
                f"{output_addon_name}_rule_vectors.pkl"
            )
            with open(rv_pkl_path, "wb") as f:
                pickle.dump(rule_vec_data, f)
            print(f"  Saved rule vector data: {rv_pkl_path}")

        print(f"\nDone. Data: {outpath}/; aggregate figures: {figpath}/; per-seed figures: {seed_figpath}/")
    else:
        print("\nNo rulesets produced results. Nothing to plot.")
