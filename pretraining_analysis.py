"""
Post-hoc analysis of the pretraining → post-training transfer experiment.

For each saved run, compares hidden-state (and optionally modulation) geometry
between pretraining tasks and the post-training task via cross-validated PCA:
  - Fit PCA on the post-training task representations
  - Project pretraining task representations into that basis
  - Measure cumulative variance explained and participation ratio

Aggregates results across seeds and produces summary figures.

Outputs saved to ./pretraining_analysis/.
"""

import numpy as np
import pickle
import os
import re
import copy
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

import mpn

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
# Existing outputs are preserved across runs; new outputs only overwrite
# files that share the same name. Aggregate figures embed `addon_name` in
# their filenames so different model variants (e.g. L2 strength, batch
# size) coexist in the same folder.
os.makedirs(outpath, exist_ok=True)

# Pretraining rulesets to run (must match rules_dict in pretraining.py).
# Each one is processed independently; their learning curves are then
# combined into a single cross-ruleset figure.
rulesets_to_run = ["fdgo_delaygo", "fdanti_delaygo"]

# Optional cap on how many seeds to analyze per ruleset. Useful for a fast
# iteration loop: set to a small int (e.g. 2) to skim through all the
# per-seed work in a fraction of the time, then set back to None to run
# every available seed for the final figures. When an int, seeds are
# sampled randomly without replacement from all that exist on disk; the
# random choice is reproducible via `seed_sample_rng_seed` below.
n_seeds_per_ruleset = None
seed_sample_rng_seed = 0

# Active ruleset / stage-1 tasks are (re)assigned at the top of each
# iteration of the main loop below. Functions that build file paths
# read `ruleset` from module scope at call time.
ruleset = None
stage1_tasks = None


def _stage1_tasks_for(rs):
    return ["fdanti", "delaygo"] if rs == "fdanti_delaygo" else ["fdgo", "delaygo"]


# Post-training task
final_task = "delayanti"

# Per-ruleset plotting colors for the cross-ruleset combined figure.
ruleset_colors = {
    "fdgo_delaygo": c_vals[1],
    "fdanti_delaygo": c_vals[2],
}

chosen_network = "dmpn"
N = 300
# PCA component cap for modulation / modulation_weighted analyses.
# Hidden uses N (the ambient dim). Modulation lives in N² = 40 000 dims, so a
# larger cap is needed to see whether CVE eventually saturates. sklearn still
# requires n_components ≤ min(n_samples, n_features); the call sites clamp to
# that limit to stay safe.
N_MOD_PCS = 1000

# Top-k cap for principal-angle analysis. Subspace angles are computed
# between the top-K PC bases of the pretraining and novel representations;
# the returned spectrum has K ascending values. Keep it modest because the
# cost scales like K³ and most of the interpretive content lies in the
# first few angles (zero-angle directions = shared axes, π/2 = orthogonal).
N_ANGLES = 20

# Naming components that form addon_name in pretraining.py:
#   addon_name = f"+hidden{N}+L21e4+batch{batch}+{metric}"
metric = "angle"
reg = "L21e3"
addon_name = f"+hidden{N}+{reg}+batch128+{metric}"

# ─────────────────────────────────────────────────────────────────────────────
# Path construction (matches pretraining.py naming convention)
# ─────────────────────────────────────────────────────────────────────────────
def build_paths(seed):
    """Build file paths for a given seed, matching pretraining.py output naming."""
    base = f"{ruleset}_{chosen_network}_seed{seed}_{addon_name}"
    return {
        "stage1_output": f"{basepath}/output_{base}_stage1.npz",
        "stage2_output": f"{basepath}/output_{base}_stage2.npz",
        "param_result": f"{basepath}/param_{base}_result.npz",
    }


def hist_path(seed):
    """Full training-history pickle path (see pretraining.py)."""
    return f"{basepath}/hist_{ruleset}_{chosen_network}_seed{seed}_{addon_name}.pkl"


def ckpt_path(seed):
    """Network checkpoint path (see pretraining.py)."""
    return f"{basepath}/savednet_{ruleset}_{chosen_network}_seed{seed}_{addon_name}.pt"


def config_path(seed):
    """
    Hyperparameter JSON path (see pretraining.py). Note: unlike the other
    per-seed files, this one omits `chosen_network` from its filename.
    """
    return f"{basepath}/param_{ruleset}_seed{seed}_{addon_name}_param.json"


def load_train_params(seed):
    """Load the stage-1 train_params dict saved alongside the checkpoint."""
    with open(config_path(seed)) as f:
        cfg = json.load(f)
    return cfg["train_params"]


def load_mpn_W(seed):
    """
    Load the frozen plastic-layer weight W from the saved checkpoint.
    Returns an (N, N) numpy array. W is identical across stage 1 and stage 2
    (frozen in expand_and_freeze), so a single W multiplies both stages' M.
    """
    ckpt = torch.load(ckpt_path(seed), map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["mp_layer1.W"].numpy()


def load_rule_vectors(seed):
    """
    Extract the 3 rule-input column vectors from the checkpoint.

    W_initial_linear.weight has shape (n_hidden, n_input); its last 3
    columns are the task-indicator rows in the input layout:
        [fix1, fix2, r1cos, r1sin, r2cos, r2sin, task1, task2, task3]
    Column 6 (task1)  = pretraining task 0 (e.g. fdgo / fdanti)
    Column 7 (task2)  = pretraining task 1 (delaygo)
    Column 8 (task3)  = post-training task (delayanti), trained in stage 2

    Returns (v_pre0, v_pre1, v_novel) as three (n_hidden,) numpy arrays.
    """
    ckpt = torch.load(ckpt_path(seed), map_location="cpu", weights_only=False)
    W_in = ckpt["state_dict"]["W_initial_linear.weight"].numpy()
    return W_in[:, -3], W_in[:, -2], W_in[:, -1]


def _rule_vector_stats(v_pre0, v_pre1, v_novel):
    """
    Per-seed scalar summaries of the rule-input vector geometry.

    Returns a dict with:
      - cos_pre0_pre1, cos_novel_pre0, cos_novel_pre1: pairwise cosines
      - in_span_fraction: ‖P_{span(pre0, pre1)} v_novel‖ / ‖v_novel‖
        (1 → novel is a linear combination of pretrained rule vectors;
         0 → novel is orthogonal to the pretrained rule subspace)
      - norm_pre0, norm_pre1, norm_novel: Frobenius norms for context
    """
    v_pre0 = np.asarray(v_pre0)
    v_pre1 = np.asarray(v_pre1)
    v_novel = np.asarray(v_novel)

    # Project v_novel onto span(v_pre0, v_pre1) via least-squares:
    # minimize ‖v_novel - (a·v_pre0 + b·v_pre1)‖ → coefficients from lstsq.
    basis = np.stack([v_pre0, v_pre1], axis=1)  # (n_hidden, 2)
    coeffs, _, _, _ = np.linalg.lstsq(basis, v_novel, rcond=None)
    v_proj = basis @ coeffs
    norm_novel = np.linalg.norm(v_novel)
    in_span = float(np.linalg.norm(v_proj) / norm_novel) if norm_novel > 0 else float("nan")

    return {
        "cos_pre0_pre1": _cosine_sim(v_pre0, v_pre1),
        "cos_novel_pre0": _cosine_sim(v_novel, v_pre0),
        "cos_novel_pre1": _cosine_sim(v_novel, v_pre1),
        "in_span_fraction": in_span,
        "norm_pre0": float(np.linalg.norm(v_pre0)),
        "norm_pre1": float(np.linalg.norm(v_pre1)),
        "norm_novel": float(norm_novel),
    }


def load_final_net(seed, device):
    """
    Reconstruct the post-stage-2 network from its checkpoint and return it
    in eval mode on `device`. Mirrors the reload pattern in pretraining.py.
    """
    ckpt = torch.load(ckpt_path(seed), map_location="cpu", weights_only=False)
    net_params = ckpt["net_params"]
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
        tname = task_names[int(test_task[b])] if task_names is not None else "?"
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


def evaluate_backbone_on_delayanti(
    seed, device, stage2_output, n_random_rule_inits=10, rng_seed=0,
    n_batch=128,
):
    """
    Measure the frozen backbone's starting-point competence on delayanti.

    Protocol
    ────────
    Load the post-stage-2 checkpoint. Everything in that state_dict except
    the last column of W_initial_linear is identical to the end-of-stage-1
    network (stage 2 only trains that column via the gradient hook in
    expand_and_freeze). We can therefore reconstruct the stage-2 STARTING
    point by overwriting that last column with a fresh random init, then
    run a forward pass on freshly generated delayanti test data to see how
    close the backbone is to solving delayanti BEFORE any stage-2 gradient
    step.

    Loss and accuracy are computed via the network's own `compute_loss` /
    `compute_acc` so the numbers are directly comparable to training-time
    loss/accuracy. This requires the trial mask, which is not saved in the
    npz, so a fresh batch is generated from the saved task_params.

    Returns dict of per-sample numpy arrays (length n_random_rule_inits):
        loss     — full training loss (output MSE + regularization)
        loss_out — output-label-MSE component only
        acc      — angle-based accuracy on the response period
    """
    import math
    import mpn_tasks  # local import so the helper stays self-contained

    # Load network and task config.
    net = load_final_net(seed, device)

    # compute_loss reads regularization attributes that are normally set by
    # net.fit(train_params, ...) during training. load_state_dict doesn't
    # restore those (they're not tensors), so we re-inject them from the
    # saved JSON config to match the training-time loss computation.
    train_params = load_train_params(seed)
    net.weight_reg = train_params.get("weight_reg", None)
    net.reg_lambda = train_params.get("reg_lambda", 0.0)
    net.activity_reg = train_params.get("activity_reg", None)
    net.reg_omit = train_params.get("reg_omit", [])
    net.gradient_type = train_params.get("gradient_type", "backprop")

    task_params = stage2_output["task_params"].item()

    # Generate a fresh test batch for delayanti. pretraining_shift=2 pads
    # the two stage-1 task-indicator slots with zeros so the input has the
    # correct 9-channel layout the checkpoint expects.
    task_params_test = copy.deepcopy(task_params)
    task_params_test["long_response"] = "normal"
    (test_input, test_output, test_mask), _ = mpn_tasks.generate_trials_wrap(
        task_params_test, n_batch, device=device, rules=["delayanti"],
        mode_input="random", pretraining_shift=2,
    )

    # Grab the trained rule-vector column so we can restore it afterward.
    orig_col = net.W_initial_linear.weight.data[:, -1].detach().clone()

    rng = np.random.default_rng(rng_seed + seed)
    loss_full, loss_out_only, acc_vals = [], [], []
    with torch.no_grad():
        for _ in range(n_random_rule_inits):
            # Kaiming-uniform init matching expand_and_freeze.
            new_col = torch.empty_like(orig_col)
            torch.manual_seed(int(rng.integers(0, 2**31)))
            torch.nn.init.kaiming_uniform_(new_col.view(-1, 1).t(), a=math.sqrt(5))
            net.W_initial_linear.weight.data[:, -1] = new_col

            # Forward through the full batch at once; compute_loss/acc
            # expect the whole batch, not mini-chunks.
            net_out, hidden, _ = net.iterate_sequence_batch(
                test_input, run_mode="minimal")

            loss, loss_components, _ = net.compute_loss(
                net_out, test_output, test_mask, hidden=hidden)
            acc, _ = net.compute_acc(
                net_out, test_output, test_mask, test_input,
                isvalid=True, mode=net.acc_measure)

            loss_full.append(float(loss.detach().cpu().item()))
            # loss_components is (output_label_term, reg_term); keep the
            # output-only term separately since reg_term is not meaningful
            # for a random rule vector (it penalizes all weights equally
            # regardless of the task-match).
            loss_out_only.append(float(loss_components[0]))
            acc_vals.append(float(acc.detach().cpu().item()))

            # Clear per-layer state before the next init.
            if hasattr(net, "reset_state"):
                try:
                    net.reset_state(B=1)
                except Exception:
                    pass

    net.W_initial_linear.weight.data[:, -1] = orig_col

    return {
        "loss": np.array(loss_full),
        "loss_out": np.array(loss_out_only),
        "acc": np.array(acc_vals),
    }


def run_final_net_sanity_check(
    seed, device, stage1_output, stage2_output, n_trials=10,
):
    """
    Load the final (post-stage-2) network and run it on BOTH stages' saved
    test inputs. The npzs already contain correctly zero-padded task
    indicator dimensions for each stage (see pretraining.py), so we reuse
    them verbatim and only need to reconstruct the network. Saves two
    figures: a stage-1 view (final net on pretraining test inputs) and a
    stage-2 view (final net on post-training test inputs).
    """
    net = load_final_net(seed, device)

    # Stage 1: pretraining tasks, evaluated with the *final* (post-stage-2) net.
    ti1 = np.asarray(stage1_output["test_input_np"])
    to1 = np.asarray(stage1_output["test_output_np"])
    task_params1 = stage1_output["task_params"].item()
    tt1 = np.asarray(stage1_output["test_task"])

    # Stage 2: post-training task, evaluated with the final net.
    ti2 = np.asarray(stage2_output["test_input_np"])
    to2 = np.asarray(stage2_output["test_output_np"])
    task_params2 = stage2_output["task_params"].item()
    tt2 = np.asarray(stage2_output["test_task"])

    def _run_on(ti_np):
        # Run in minibatches to mirror train_network's minibatch=8.
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

    checkname = f"{ruleset}_{chosen_network}_seed{seed}_{addon_name}"
    _plot_input_output_panel(
        f"{outpath}/{checkname}_finalnet_on_stage1.png",
        ti1, to1, out1, task_params1["rules"], tt1, n_trials=n_trials,
    )
    _plot_input_output_panel(
        f"{outpath}/{checkname}_finalnet_on_stage2.png",
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
    Effective dimensionality: (sum λ)² / sum λ².
    Returns 1 for a rank-1 matrix and d for isotropic d-dimensional variance.
    Higher PR → representations spread across more dimensions.
    """
    eigvals = np.linalg.eigvalsh(cov_mat)
    eigvals = np.clip(eigvals, 0, None)
    s1 = np.sum(eigvals)
    s2 = np.sum(eigvals ** 2)
    if s1 == 0 or s2 == 0:
        return 0.0
    return (s1 ** 2) / s2


def _mean_M_over_period(Ms_period):
    """
    Collapse a period slice of M down to a single (N, N) matrix.

    Ms_period shape: (batch, time_in_period, N, N), where time_in_period is
    already trimmed to the relevant epoch (after period_shift). Averaging
    over both batch and time handles variable trial lengths automatically —
    the output shape depends only on N, not on how many timesteps were in
    the slice. Uses the same period slices that feed the PCA analysis, so
    cosine similarity numbers here are directly comparable to the PCA.
    """
    return Ms_period.mean(axis=(0, 1))


def _stim_direction_indices(test_input_np, epochs, task_name, mask=None, n_dirs=8):
    """
    Return a per-trial integer stimulus-direction index in [0, n_dirs).

    Reads the r1cos (channel 2) and r1sin (channel 3) inputs at the midpoint
    of the task's stim1 period (where the stimulus is on), computes
    θ = atan2(sin, cos), and bins to the nearest of n_dirs evenly-spaced
    angles. Assumes sigma_x = 0 (noise-free inputs), which is the case in
    pretraining.py.
    """
    if mask is not None:
        test_input_np = test_input_np[mask]
    start, end = epochs[task_name]["stim1"]
    t_mid = (start + end) // 2
    r1cos = test_input_np[:, t_mid, 2]
    r1sin = test_input_np[:, t_mid, 3]
    thetas = np.arctan2(r1sin, r1cos)
    bin_size = 2 * np.pi / n_dirs
    return (np.round(thetas / bin_size).astype(int)) % n_dirs


def direction_averaged_cve(
    X_period, Y_period, dir_idxs_X, dir_idxs_Y,
    n_components, datatype, n_dirs=8,
):
    """
    Per-stimulus-direction CVE averaged across directions.

    Variance-decomposition diagnostic: the pooled CVE marginalizes over
    all stimulus directions before fitting PCA. If the two tasks share
    a per-direction subspace structure (e.g. a ring attractor whose axial
    geometry is identical at every θ but rotated), pooling washes that
    out. Running the full CVE analysis within each direction bin and
    averaging at the end reveals a shared structure that would otherwise
    be hidden.

    For each direction with trials on both sides, subset both period
    slices to those trials and call `pca_cross_variance`. Average the
    resulting `cev_Y` and `cev_Y_self` curves across directions. Returns
    length-min_len arrays truncated to the shortest achievable curve
    (a direction with few samples produces a shorter PCA spectrum).

    Note: alignment is by *stimulus* direction, not response direction —
    so for the go-period comparison a pro-task trial at stimulus θ is
    paired with a delayanti trial at stimulus θ (which responds toward
    θ+π). This tests whether the computational trajectories under the
    same input differ, regardless of target response.

    Returns
    -------
    dict with:
      cev_Y_mean       : (k,) direction-averaged novel-in-pretraining CVE
      cev_Y_self_mean  : (k,) direction-averaged novel-self CVE
      n_dirs_used      : int, how many direction bins contributed
    """
    cev_Y_list = []
    cev_Y_self_list = []
    for d in range(n_dirs):
        mask_X = (dir_idxs_X == d)
        mask_Y = (dir_idxs_Y == d)
        if not mask_X.any() or not mask_Y.any():
            continue
        X_d = X_period[mask_X]
        Y_d = Y_period[mask_Y]
        # Cap n_components to what this direction's subset can support
        # (some bins have only a handful of trials × timesteps).
        if datatype == "hidden":
            nX = X_d.shape[0] * X_d.shape[1]
            nY = Y_d.shape[0] * Y_d.shape[1]
        else:
            nX = X_d.shape[0] * X_d.shape[1]
            nY = Y_d.shape[0] * Y_d.shape[1]
        k_d = min(n_components, nX - 1, nY - 1)
        if k_d < 1:
            continue
        res_d = pca_cross_variance(
            X_d, Y_d, n_components=k_d, datatype=datatype)
        cev_Y_list.append(res_d["cev_Y"])
        cev_Y_self_list.append(res_d["cev_Y_self"])

    if not cev_Y_list:
        return {
            "cev_Y_mean": np.array([]),
            "cev_Y_self_mean": np.array([]),
            "n_dirs_used": 0,
        }
    min_len = min(len(c) for c in cev_Y_list)
    cev_Y_trunc = np.stack([c[:min_len] for c in cev_Y_list], axis=0)
    cev_self_trunc = np.stack([c[:min_len] for c in cev_Y_self_list], axis=0)
    return {
        "cev_Y_mean": cev_Y_trunc.mean(axis=0),
        "cev_Y_self_mean": cev_self_trunc.mean(axis=0),
        "n_dirs_used": len(cev_Y_list),
    }


def _mean_M_by_direction(M_period, dir_idxs, n_dirs=8):
    """
    Per-direction mean of M over batch and time.

    Returns a (n_dirs, N, N) array; bins with no trials produce NaNs.
    """
    N = M_period.shape[-1]
    out = np.full((n_dirs, N, N), np.nan, dtype=M_period.dtype)
    for d in range(n_dirs):
        trials = (dir_idxs == d)
        if trials.any():
            out[d] = M_period[trials].mean(axis=(0, 1))
    return out


def _stim_aligned_scalars(M_A_by_dir, M_B_by_dir):
    """
    Stimulus-aligned (direction-matched) cosine and Frobenius-difference.

    For each direction that has data on both sides, compute cos(M_A_θ, M_B_θ)
    and ‖M_A_θ − M_B_θ‖. Return the mean over directions for each.
    """
    n_dirs = M_A_by_dir.shape[0]
    cos_vals, frob_vals = [], []
    for d in range(n_dirs):
        A, B = M_A_by_dir[d], M_B_by_dir[d]
        if np.isnan(A).any() or np.isnan(B).any():
            continue
        cos_vals.append(_cosine_sim(A, B))
        frob_vals.append(float(np.linalg.norm(A - B)))
    if not cos_vals:
        return float("nan"), float("nan")
    return float(np.mean(cos_vals)), float(np.mean(frob_vals))


def _cosine_sim(A, B):
    """Cosine similarity between two tensors, flattened."""
    a, b = np.asarray(A).ravel(), np.asarray(B).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def principal_angles(X, Y, k, datatype="hidden"):
    """
    Principal angles (radians) between X's and Y's top-k PC subspaces.

    Complements CVE: CVE asks "what fraction of Y's variance lives in X's
    subspace?" which is variance-weighted. Principal angles ask "how
    aligned are the K dominant directions of X with the K dominant
    directions of Y?" — a per-direction geometric readout. 0 rad means a
    shared axis; π/2 means orthogonal. Reveals cases where CVE is low
    because the top-1 directions disagree but secondary directions still
    align (or vice versa).

    X, Y reshape follows the same rules as `pca_cross_variance`:
      "hidden"              — (batch×time, n_hidden)
      "modulation"          — (batch×time, n_hidden²)
      "modulation_weighted" — same flatten; caller pre-multiplies W⊙M.

    Returns
    -------
    angles : ndarray of shape (k_eff,), sorted ASCENDING. scipy's native
             order is descending (largest angle first, smallest last), so
             the returned array is reversed to put the most-aligned
             direction at index 0. k_eff = min(k, rank(X basis), rank(Y
             basis)) so you get at most k values.
    """
    if datatype == "hidden":
        X2d = X.reshape(-1, X.shape[-1])
        Y2d = Y.reshape(-1, Y.shape[-1])
    elif datatype in ("modulation", "modulation_weighted"):
        X2d = X.reshape(-1, X.shape[-1] * X.shape[-2])
        Y2d = Y.reshape(-1, Y.shape[-1] * Y.shape[-2])
    else:
        raise ValueError(f"Unknown datatype: {datatype}")

    # Clamp k to what each PCA can support.
    k_eff = min(k, min(X2d.shape) - 1, min(Y2d.shape) - 1)
    if k_eff < 1:
        return np.array([])

    pca_X = PCA(n_components=k_eff, svd_solver="randomized", random_state=0)
    pca_X.fit(X2d)
    pca_Y = PCA(n_components=k_eff, svd_solver="randomized", random_state=0)
    pca_Y.fit(Y2d)

    # components_ is (k, n_features); subspace_angles expects (n_features, k)
    # columns spanning each subspace.
    basis_X = pca_X.components_.T
    basis_Y = pca_Y.components_.T
    # scipy returns angles in descending order (largest first). Reverse
    # so index 0 is the most-aligned (smallest) angle, matching the
    # "top-k shared directions" convention used in the rest of the file.
    return subspace_angles(basis_X, basis_Y)[::-1]


def _pr_from_data(X_c):
    """
    PR from centered data, using whichever Gram side is smaller.

    Covariance (X_c.T @ X_c) and kernel (X_c @ X_c.T) share the same non-zero
    eigenvalues, so PR is identical; eigendecomposing the smaller matrix is
    strictly faster. Essential for modulation (n_features = n_hidden² ≫
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
    if mask is not None:
        op = op[mask, :, :]
    return op[:, start + shift:end, :]


def pca_cross_variance(X, Y, n_components=None, center_on="X", datatype="hidden"):
    """
    Driscoll-style subspace-overlap analysis (see Driscoll et al.,
    Nature Neurosci. 2024, Fig. 6c/k captions).

    Convention (matches Driscoll exactly):
      X = basis data   — the task whose top-k PCs define the reference
                         subspace (typically a pretraining task).
      Y = target data  — the task whose variance is being measured in
                         each basis (typically the novel / post-training
                         task, e.g. delayanti = Driscoll's MemoryAnti).

    Two cumulative-variance-explained curves are computed on the target:
      cev_Y_self : Y's variance captured by Y's own PCs   (Driscoll "black")
                   — saturates to 1.0, serves as the self-reference.
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
    "modulation"          — (batch×time, n_hidden²): full M matrix flattened
    "modulation_weighted" — (batch×time, n_hidden²): W⊙M flattened (caller
                            must pre-multiply M by W elementwise)
    """
    if datatype == "hidden":
        X2d = X.reshape(-1, X.shape[-1])
        Y2d = Y.reshape(-1, Y.shape[-1])
    elif datatype in ("modulation", "modulation_weighted"):
        # Full (W⊙)M flattened: n_hidden² features — very high-dimensional.
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
    # Separate PCA fit so the self-reference saturates at 1.0 naturally
    # and uses the same n_components truncation as the cross-basis curve.
    pca_Y = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca_Y.fit(Y2d)
    evr_Y_self = pca_Y.explained_variance_ratio_
    cev_Y_self = np.cumsum(evr_Y_self)

    # --- Y projected into X's basis (Driscoll's "purple" cross curve) ----
    # Center Y on X's mean: asks how much of Y's variance falls in X's
    # subspace when measured from X's origin (center_on="X" is the
    # standard choice and matches Driscoll's convention).
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

    # --- Participation ratios --------------------------------------------
    X_c = X2d - X2d.mean(axis=0, keepdims=True)
    PR_X = _pr_from_data(X_c)

    Y_c = Y2d - Y2d.mean(axis=0, keepdims=True)
    PR_Y = _pr_from_data(Y_c)

    # PR of Y after projection: how many of X's PCs does Y actually use?
    # Y_proj is (n_samples, n_components), centered on X's mean rather
    # than Y's mean. Mean-center Y_proj before the covariance product so
    # PR reflects Y's spread in X's basis, not the mean offset between
    # the two tasks (which would otherwise show up as an extra rank-1
    # eigenvalue and inflate PR). Keeps this PR consistent with how
    # evr_Y is computed via np.var above.
    Y_proj_c = Y_proj - Y_proj.mean(axis=0, keepdims=True)
    cov_Yp = (Y_proj_c.T @ Y_proj_c) / Y_proj_c.shape[0]
    PR_Y_in_Xbasis = _participation_ratio(cov_Yp)

    return {
        "evr_X": evr_X, "cev_X": cev_X,
        "evr_Y_self": evr_Y_self, "cev_Y_self": cev_Y_self,
        "evr_Y": evr_Y, "cev_Y": cev_Y,
        "PR_X": PR_X, "PR_Y": PR_Y, "PR_Y_in_Xbasis": PR_Y_in_Xbasis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
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

    for active_ruleset in rulesets_to_run:
        # Rebind module-level ruleset/stage1_tasks so the helper functions
        # (build_paths, ckpt_path, hist_path, discover_seeds) see the right
        # names for this iteration.
        ruleset = active_ruleset
        stage1_tasks = _stage1_tasks_for(active_ruleset)

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

        # Optional subsample for fast iteration. Draw without replacement
        # using a deterministic RNG so two runs with the same cap see the
        # same subset. Seeded per-ruleset so each ruleset picks its own
        # subset rather than aligning indices across rulesets.
        if n_seeds_per_ruleset is not None and len(seeds) > n_seeds_per_ruleset:
            rng = np.random.default_rng(seed_sample_rng_seed + hash(ruleset) % (2**31))
            seeds = sorted(rng.choice(seeds, size=n_seeds_per_ruleset,
                                      replace=False).tolist())
            print(f"Subsampled to {len(seeds)} seeds: {seeds}")

        all_seed_results = []

        for seed in seeds:
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

                # Sanity-check figure: load the post-stage-2 network and run
                # it on both stages' test inputs (already correctly padded in
                # each stage's npz). Lets us visually verify the saved model
                # actually solves both tasks before trusting downstream PCA.
                try:
                    run_final_net_sanity_check(
                        seed, sanity_device, stage1_output, stage2_output,
                        n_trials=10,
                    )
                except (FileNotFoundError, KeyError, RuntimeError) as e:
                    print(f"  WARNING: final-net sanity check failed ({e}); "
                          f"continuing without it")

                seed_result = {"seed": seed}

                # Starting-point probe: reconstruct the stage-2 init state
                # (rule-vector column reset to a random Kaiming-uniform) and
                # measure delayanti loss over several random draws. This
                # localizes the transfer advantage to the frozen backbone
                # rather than to anything the stage-2 optimizer does.
                try:
                    backbone_probe = evaluate_backbone_on_delayanti(
                        seed, sanity_device, stage2_output,
                        n_random_rule_inits=10,
                    )
                    seed_result["backbone_probe"] = backbone_probe
                except (FileNotFoundError, KeyError, RuntimeError) as e:
                    print(f"  WARNING: backbone-probe failed ({e}); "
                          f"continuing without it")

                test_task = stage1_output["test_task"]
                stage1_hs = final_param["hs_stage1"]
                stage2_hs = final_param["hs_stage2"]
                stage1_ms = final_param["Ms_orig_stage1"]
                stage2_ms = final_param["Ms_orig_stage2"]

                stage1_rules_epochs = stage1_output["rules_epochs"].item()
                stage2_rules_epochs = stage2_output["rules_epochs2"].item()

                # Validation accuracy.
                # `valid_acc_iter` is monotonic across both stages, but the
                # transition iter (stop+1) is recorded twice — once as the last
                # stage-1 validation, once as the first stage-2 validation.
                # Exclude the boundary from both sides so the post-training
                # curve starts cleanly at the first fresh stage-2 measurement.
                stop = final_param["pretrain_stop"]
                acc_iter = final_param["valid_acc_iter"]
                acc = final_param["valid_acc"]
                post_mask = acc_iter > stop + 1
                pre_mask = acc_iter < stop
                acc_iter_post = acc_iter[post_mask] - stop
                acc_post = acc[post_mask]
                acc_iter_pre = acc_iter[pre_mask]
                acc_pre = acc[pre_mask]

                # Trial masks for stage 1 tasks
                mask0 = (test_task == 0)
                mask1 = (test_task == 1)

                # ---- Extract periods ----
                # period_shift_percentage skips the onset transient of each epoch
                # so the PCA captures steady-state geometry; applied consistently
                # to both stim and go periods.

                # Stimulus period: compare stage1_tasks[0] vs final_task
                stage1_stim = period_slice(
                    stage1_hs, stage1_rules_epochs, stage1_tasks[0], "stim1",
                    shift_percentage=period_shift_percentage, mask=mask0)
                final_stim = period_slice(
                    stage2_hs, stage2_rules_epochs, final_task, "stim1",
                    shift_percentage=period_shift_percentage)

                # Go period: compare stage1_tasks[1] vs final_task
                stage1_go = period_slice(
                    stage1_hs, stage1_rules_epochs, stage1_tasks[1], "go1",
                    shift_percentage=period_shift_percentage, mask=mask1)
                final_go = period_slice(
                    stage2_hs, stage2_rules_epochs, final_task, "go1",
                    shift_percentage=period_shift_percentage)

                # Per-trial stimulus-direction index for each task, used
                # by the direction-averaged (stimulus-aligned) CVE below.
                # Alignment is by input stimulus direction, so a pro-task
                # trial at θ gets paired with a delayanti trial at θ even
                # though delayanti responds toward θ+π.
                stage1_test_input = np.asarray(stage1_output["test_input_np"])
                stage2_test_input = np.asarray(stage2_output["test_input_np"])
                dir_task0 = _stim_direction_indices(
                    stage1_test_input, stage1_rules_epochs,
                    stage1_tasks[0], mask=mask0)
                dir_task1 = _stim_direction_indices(
                    stage1_test_input, stage1_rules_epochs,
                    stage1_tasks[1], mask=mask1)
                dir_final = _stim_direction_indices(
                    stage2_test_input, stage2_rules_epochs, final_task)

                # Hidden state analysis.
                # Driscoll convention: X = pretraining (basis), Y = novel
                # (delayanti). cev_Y = novel's variance in pretraining's PCs
                # (Driscoll "purple"); cev_Y_self = novel in its own PCs
                # (Driscoll "black").
                res_h_stim = pca_cross_variance(
                    stage1_stim, final_stim, n_components=N, datatype="hidden")
                res_h_go = pca_cross_variance(
                    stage1_go, final_go, n_components=N, datatype="hidden")
                # Direction-averaged variant: run CVE within each stimulus
                # bin separately, then average. Diagnoses whether pooling
                # over direction is what makes the pooled CVE look low.
                dir_h_stim = direction_averaged_cve(
                    stage1_stim, final_stim, dir_task0, dir_final,
                    n_components=N, datatype="hidden")
                dir_h_go = direction_averaged_cve(
                    stage1_go, final_go, dir_task1, dir_final,
                    n_components=N, datatype="hidden")
                seed_result["hidden"] = {
                    "stimulus": res_h_stim,
                    "response": res_h_go,
                    "stimulus_dir_avg": dir_h_stim,
                    "response_dir_avg": dir_h_go,
                    "angles_stimulus": principal_angles(
                        stage1_stim, final_stim, k=N_ANGLES, datatype="hidden"),
                    "angles_response": principal_angles(
                        stage1_go, final_go, k=N_ANGLES, datatype="hidden"),
                }

                # Modulation analysis (dmpn only)
                if chosen_network == "dmpn" and stage1_ms.size > 0 and stage2_ms.size > 0:
                    stage1_stim_m = period_slice(
                        stage1_ms, stage1_rules_epochs, stage1_tasks[0], "stim1",
                        shift_percentage=period_shift_percentage, mask=mask0)
                    final_stim_m = period_slice(
                        stage2_ms, stage2_rules_epochs, final_task, "stim1",
                        shift_percentage=period_shift_percentage)
                    stage1_go_m = period_slice(
                        stage1_ms, stage1_rules_epochs, stage1_tasks[1], "go1",
                        shift_percentage=period_shift_percentage, mask=mask1)
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
                    n_comp_stim = min(N_MOD_PCS, n_stim_samples)
                    n_comp_go = min(N_MOD_PCS, n_go_samples)

                    res_m_stim = pca_cross_variance(
                        stage1_stim_m, final_stim_m,
                        n_components=n_comp_stim, datatype="modulation")
                    res_m_go = pca_cross_variance(
                        stage1_go_m, final_go_m,
                        n_components=n_comp_go, datatype="modulation")
                    k_angles_mod_stim = min(N_ANGLES, n_comp_stim)
                    k_angles_mod_go = min(N_ANGLES, n_comp_go)
                    dir_m_stim = direction_averaged_cve(
                        stage1_stim_m, final_stim_m, dir_task0, dir_final,
                        n_components=n_comp_stim, datatype="modulation")
                    dir_m_go = direction_averaged_cve(
                        stage1_go_m, final_go_m, dir_task1, dir_final,
                        n_components=n_comp_go, datatype="modulation")
                    seed_result["modulation"] = {
                        "stimulus": res_m_stim,
                        "response": res_m_go,
                        "stimulus_dir_avg": dir_m_stim,
                        "response_dir_avg": dir_m_go,
                        "angles_stimulus": principal_angles(
                            stage1_stim_m, final_stim_m,
                            k=k_angles_mod_stim, datatype="modulation"),
                        "angles_response": principal_angles(
                            stage1_go_m, final_go_m,
                            k=k_angles_mod_go, datatype="modulation"),
                    }

                    # W⊙M analysis: multiplicative contribution of plasticity to
                    # W_eff. W is frozen across both stages, so the same elementwise
                    # rescaling applies to stage-1 and stage-2 M alike. No extra
                    # normalization is applied (per user request).
                    try:
                        W = load_mpn_W(seed)  # (N, N)
                        # Broadcasting over (batch, time) axes.
                        stage1_stim_wm = stage1_stim_m * W
                        final_stim_wm = final_stim_m * W
                        stage1_go_wm = stage1_go_m * W
                        final_go_wm = final_go_m * W

                        res_wm_stim = pca_cross_variance(
                            stage1_stim_wm, final_stim_wm,
                            n_components=n_comp_stim,
                            datatype="modulation_weighted")
                        res_wm_go = pca_cross_variance(
                            stage1_go_wm, final_go_wm,
                            n_components=n_comp_go,
                            datatype="modulation_weighted")
                        dir_wm_stim = direction_averaged_cve(
                            stage1_stim_wm, final_stim_wm,
                            dir_task0, dir_final,
                            n_components=n_comp_stim,
                            datatype="modulation_weighted")
                        dir_wm_go = direction_averaged_cve(
                            stage1_go_wm, final_go_wm,
                            dir_task1, dir_final,
                            n_components=n_comp_go,
                            datatype="modulation_weighted")
                        seed_result["modulation_weighted"] = {
                            "stimulus": res_wm_stim,
                            "response": res_wm_go,
                            "stimulus_dir_avg": dir_wm_stim,
                            "response_dir_avg": dir_wm_go,
                            "angles_stimulus": principal_angles(
                                stage1_stim_wm, final_stim_wm,
                                k=k_angles_mod_stim,
                                datatype="modulation_weighted"),
                            "angles_response": principal_angles(
                                stage1_go_wm, final_go_wm,
                                k=k_angles_mod_go,
                                datatype="modulation_weighted"),
                        }
                    except (FileNotFoundError, KeyError) as e:
                        print(f"  WARNING: could not load W from checkpoint "
                              f"({e}); skipping modulation_weighted for this seed")

                    # ─── Period-matched M similarity across stages ────────
                    # Always compare M on the SAME period on both sides of
                    # each cosine (apples-to-apples), so all four numbers
                    # are measured in comparable windows.
                    # The two cross-stage comparisons match the PCA pairing
                    # (task-0 vs delayanti on stim; task-1 vs delayanti on
                    # go). The two within-stage-1 baselines measure how
                    # different the two pretraining tasks are from each
                    # other, separately on stim and on go.
                    stage1_task0_go_m = period_slice(
                        stage1_ms, stage1_rules_epochs, stage1_tasks[0], "go1",
                        shift_percentage=period_shift_percentage, mask=mask0)
                    stage1_task1_stim_m = period_slice(
                        stage1_ms, stage1_rules_epochs, stage1_tasks[1], "stim1",
                        shift_percentage=period_shift_percentage, mask=mask1)

                    M_final_stim = _mean_M_over_period(final_stim_m)
                    M_task0_stim = _mean_M_over_period(stage1_stim_m)
                    M_final_go = _mean_M_over_period(final_go_m)
                    M_task1_go = _mean_M_over_period(stage1_go_m)
                    M_task0_go = _mean_M_over_period(stage1_task0_go_m)
                    M_task1_stim = _mean_M_over_period(stage1_task1_stim_m)

                    # Frobenius norm of the M-difference, period-matched.
                    # Complements cosine: cosine is direction-only
                    # (scale-invariant), ‖ΔM‖ captures how much the actual
                    # magnitude of the change is in absolute units of M.
                    frob_diff = lambda A, B: float(np.linalg.norm(A - B))

                    # Stimulus-aligned variant: group trials by stimulus
                    # direction (8 bins), compute mean M per direction per
                    # (task, period), then take cosine and Frobenius of the
                    # difference per direction, and average those 8 values.
                    # Direction indices dir_task0/dir_task1/dir_final are
                    # already computed earlier in the per-seed block (used
                    # there by the direction-averaged CVE).

                    M_final_stim_d = _mean_M_by_direction(final_stim_m, dir_final)
                    M_task0_stim_d = _mean_M_by_direction(stage1_stim_m, dir_task0)
                    M_final_go_d = _mean_M_by_direction(final_go_m, dir_final)
                    M_task1_go_d = _mean_M_by_direction(stage1_go_m, dir_task1)
                    M_task0_go_d = _mean_M_by_direction(stage1_task0_go_m, dir_task0)
                    M_task1_stim_d = _mean_M_by_direction(stage1_task1_stim_m, dir_task1)

                    aligned = {}
                    (aligned["cos_final_vs_stage1_task0_stim"],
                     aligned["frob_final_vs_stage1_task0_stim"]) = \
                        _stim_aligned_scalars(M_final_stim_d, M_task0_stim_d)
                    (aligned["cos_final_vs_stage1_task1_go"],
                     aligned["frob_final_vs_stage1_task1_go"]) = \
                        _stim_aligned_scalars(M_final_go_d, M_task1_go_d)
                    (aligned["cos_stage1_task0_vs_task1_stim"],
                     aligned["frob_stage1_task0_vs_task1_stim"]) = \
                        _stim_aligned_scalars(M_task0_stim_d, M_task1_stim_d)
                    (aligned["cos_stage1_task0_vs_task1_go"],
                     aligned["frob_stage1_task0_vs_task1_go"]) = \
                        _stim_aligned_scalars(M_task0_go_d, M_task1_go_d)

                    seed_result["m_similarity"] = {
                        # Cross-stage, period-matched (direction-marginalized).
                        "cos_final_vs_stage1_task0_stim":
                            _cosine_sim(M_final_stim, M_task0_stim),
                        "cos_final_vs_stage1_task1_go":
                            _cosine_sim(M_final_go, M_task1_go),
                        # Within-stage-1 baselines, also period-matched.
                        "cos_stage1_task0_vs_task1_stim":
                            _cosine_sim(M_task0_stim, M_task1_stim),
                        "cos_stage1_task0_vs_task1_go":
                            _cosine_sim(M_task0_go, M_task1_go),
                        # Matching Frobenius-norm differences.
                        "frob_final_vs_stage1_task0_stim":
                            frob_diff(M_final_stim, M_task0_stim),
                        "frob_final_vs_stage1_task1_go":
                            frob_diff(M_final_go, M_task1_go),
                        "frob_stage1_task0_vs_task1_stim":
                            frob_diff(M_task0_stim, M_task1_stim),
                        "frob_stage1_task0_vs_task1_go":
                            frob_diff(M_task0_go, M_task1_go),
                        # Stimulus-aligned (per-direction then averaged).
                        "cos_aligned_final_vs_stage1_task0_stim":
                            aligned["cos_final_vs_stage1_task0_stim"],
                        "cos_aligned_final_vs_stage1_task1_go":
                            aligned["cos_final_vs_stage1_task1_go"],
                        "cos_aligned_stage1_task0_vs_task1_stim":
                            aligned["cos_stage1_task0_vs_task1_stim"],
                        "cos_aligned_stage1_task0_vs_task1_go":
                            aligned["cos_stage1_task0_vs_task1_go"],
                        "frob_aligned_final_vs_stage1_task0_stim":
                            aligned["frob_final_vs_stage1_task0_stim"],
                        "frob_aligned_final_vs_stage1_task1_go":
                            aligned["frob_final_vs_stage1_task1_go"],
                        "frob_aligned_stage1_task0_vs_task1_stim":
                            aligned["frob_stage1_task0_vs_task1_stim"],
                        "frob_aligned_stage1_task0_vs_task1_go":
                            aligned["frob_stage1_task0_vs_task1_go"],
                    }
                elif chosen_network == "dmpn":
                    print("  WARNING: Ms_orig is empty for this seed (vanilla fallback?)")

                # Rule-input vector geometry: how does the learned stage-2
                # rule vector relate to the two pretraining rule vectors?
                # Cheap to compute (load checkpoint, slice 3 columns) and
                # interpretive — a Driscoll-style "reuse" story would
                # predict high in_span_fraction.
                try:
                    v_pre0, v_pre1, v_novel = load_rule_vectors(seed)
                    seed_result["rule_vectors"] = _rule_vector_stats(
                        v_pre0, v_pre1, v_novel)
                except (FileNotFoundError, KeyError) as e:
                    print(f"  WARNING: could not extract rule vectors "
                          f"({e}); skipping rule-vector analysis for this seed")

                # Learning curves
                seed_result["learning"] = {
                    "acc_iter_post": acc_iter_post,
                    "acc_post": acc_post,
                    "acc_iter_pre": acc_iter_pre,
                    "acc_pre": acc_pre,
                }

                # Validation loss from training-history pickle.
                # stage2 hist contains BOTH stages concatenated (net object is
                # shared across stages in pretraining.py); split by pretrain_stop
                # the same way as accuracy.
                hp = hist_path(seed)
                if os.path.exists(hp):
                    with open(hp, "rb") as f:
                        hist = pickle.load(f)

                    full_iter = np.asarray(hist["stage2"]["iters_monitor"])[1:]
                    full_out = np.asarray(hist["stage2"]["valid_loss_output_label"])[1:]
                    full_reg = np.asarray(hist["stage2"]["valid_loss_reg_term"])[1:]
                    n = min(len(full_iter), len(full_out), len(full_reg))
                    full_iter, full_out, full_reg = full_iter[:n], full_out[:n], full_reg[:n]

                    # Same split rule as accuracy: exclude the duplicated
                    # stage-boundary iter from both masks.
                    post_m = full_iter > stop + 1
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
                checkname = f"{ruleset}_{chosen_network}_seed{seed}_{addon_name}"
                with open(f"{outpath}/{checkname}_result.pkl", "wb") as f:
                    pickle.dump(seed_result, f)
                print(f"  Saved: {checkname}_result.pkl")

                # Per-seed PCA figure
                n_plots = len(analysis_types)
                fig, axs = plt.subplots(n_plots, 3, figsize=(9, 3 * n_plots))
                if n_plots == 1:
                    axs = axs[np.newaxis, :]

                period_to_stage1 = {
                    "stimulus": stage1_tasks[0],
                    "response": stage1_tasks[1],
                }

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
                            label=f"{final_task} in {final_task} PCs")
                        axs[row, period_idx].plot(
                            xs[:len(res["cev_Y"])], res["cev_Y"], '-o',
                            markersize=2, color=c_vals[3],
                            label=f"{final_task} in {period_to_stage1[period]} PCs")
                        axs[row, period_idx].set_xlim(0, x_up)
                        axs[row, period_idx].set_ylim(0, 1.05)
                        axs[row, period_idx].set_xlabel("# PCs")
                        axs[row, period_idx].set_ylabel(
                            f"{final_task} variance explained")
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
                        "PR pre", f"PR {final_task}", f"PR {final_task}|pre",
                        "PR pre", f"PR {final_task}", f"PR {final_task}|pre",
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
                fig.savefig(f"{outpath}/{checkname}_pca.png", dpi=300)
                plt.close(fig)
                continue

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

        period_to_stage1 = {
            "stimulus": stage1_tasks[0],
            "response": stage1_tasks[1],
        }

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
                                       label=f"{final_task} in {final_task} PCs (mean)")
                    axs[row, col].plot(xs_mean, mean_cross, color="gray",
                                       linewidth=2.5,
                                       label=f"{final_task} in {period_to_stage1[period]} PCs (mean)")

                axs[row, col].set_xlim(0, x_up)
                axs[row, col].set_ylim(0, 1.05)
                axs[row, col].set_xlabel("# PCs")
                axs[row, col].set_ylabel(f"{final_task} variance explained")
                axs[row, col].set_title(f"{dtype} — {period}")
                axs[row, col].legend(fontsize=6)

        fig.suptitle(f"{ruleset} | {chosen_network} | {len(all_seed_results)} seeds", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{outpath}/{ruleset}_{chosen_network}_{addon_name}_aggregate.png", dpi=300)
        plt.close(fig)

        # Save aggregate CVE data for paper_plot reuse
        aggregate_data = {"ruleset": ruleset, "analysis_types": analysis_types,
                          "periods": ["stimulus", "response"],
                          "final_task": final_task, "stage1_tasks": list(stage1_tasks)}
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
        agg_pkl_path = f"{outpath}/{ruleset}_{chosen_network}_{addon_name}_aggregate.pkl"
        with open(agg_pkl_path, "wb") as f:
            pickle.dump(aggregate_data, f)
        print(f"  Saved aggregate data: {agg_pkl_path}")

        # ─── Stimulus-aligned (direction-averaged) CVE summary ─────────────
        # Same layout as the pooled CVE aggregate above (no PR column).
        # Each curve is already a per-seed direction-average across the 8
        # stimulus bins — alignment is by input stimulus direction, not
        # response direction.
        fig_d, axs_d = plt.subplots(n_plots, 2, figsize=(4 * 2, 4 * n_plots))
        if n_plots == 1:
            axs_d = axs_d[np.newaxis, :]

        for row, dtype in enumerate(analysis_types):
            x_up = 20 if dtype == "hidden" else N_MOD_PCS
            for col, period in enumerate(["stimulus", "response"]):
                key = f"{period}_dir_avg"
                all_self = []
                all_cross = []
                for sr_idx, sr in enumerate(all_seed_results):
                    if dtype not in sr or key not in sr[dtype]:
                        continue
                    entry = sr[dtype][key]
                    self_curve = np.asarray(entry.get("cev_Y_self_mean", []))
                    cross_curve = np.asarray(entry.get("cev_Y_mean", []))
                    if self_curve.size == 0 or cross_curve.size == 0:
                        continue
                    all_self.append(self_curve)
                    all_cross.append(cross_curve)

                    xs_self = np.arange(1, len(self_curve) + 1)
                    xs_cross = np.arange(1, len(cross_curve) + 1)
                    axs_d[row, col].plot(xs_self, self_curve,
                                         color="black", alpha=0.2)
                    axs_d[row, col].plot(xs_cross, cross_curve,
                                         color=c_vals[1 + sr_idx],
                                         alpha=0.6, label=f"seed {sr['seed']}")

                if all_self:
                    min_len = min(min(len(c) for c in all_self),
                                  min(len(c) for c in all_cross))
                    mean_self = np.mean([c[:min_len] for c in all_self], axis=0)
                    mean_cross = np.mean([c[:min_len] for c in all_cross], axis=0)
                    xs_mean = np.arange(1, min_len + 1)
                    axs_d[row, col].plot(
                        xs_mean, mean_self, color="black", linewidth=2.5,
                        label=f"{final_task} in {final_task} PCs (mean)")
                    pre_label = stage1_tasks[0] if period == "stimulus" else stage1_tasks[1]
                    axs_d[row, col].plot(
                        xs_mean, mean_cross, color="gray", linewidth=2.5,
                        label=f"{final_task} in {pre_label} PCs (mean)")

                axs_d[row, col].set_xlim(0, x_up)
                axs_d[row, col].set_ylim(0, 1.05)
                axs_d[row, col].set_xlabel("# PCs")
                axs_d[row, col].set_ylabel(f"{final_task} variance explained")
                axs_d[row, col].set_title(
                    f"{dtype} — {period} (direction-averaged)")
                axs_d[row, col].legend(fontsize=6)

        fig_d.suptitle(
            f"{ruleset} | {chosen_network} | stimulus-aligned CVE "
            f"({len(all_seed_results)} seeds)", fontsize=12)
        fig_d.tight_layout()
        fig_d.savefig(
            f"{outpath}/{ruleset}_{chosen_network}_{addon_name}_aggregate_stimaligned.png",
            dpi=300)
        plt.close(fig_d)

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
            f"(pretraining vs {final_task})", fontsize=11)
        figpa.tight_layout()
        figpa.savefig(
            f"{outpath}/{ruleset}_{chosen_network}_{addon_name}_principal_angles.png",
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
        figlc.savefig(f"{outpath}/{ruleset}_{chosen_network}_{addon_name}_learning.png", dpi=300)
        plt.close(figlc)

        if not any("loss" in sr for sr in all_seed_results):
            print("No training-history pickles found; loss column in "
                  "learning figure will be empty.")

        # ─── Period-matched M cosine similarity summary (per ruleset) ──────
        # Each comparison uses the same period slice that feeds the PCA:
        #   final vs task-0  → stimulus period
        #   final vs task-1  → response/go period
        #   task-0 vs task-1 → task-0 stim vs task-1 go (within-stage-1 baseline)
        # Low cross-stage cosine means stage-2 M is pointing somewhere
        # different from its pretraining counterpart in that period.
        sim_seeds = [sr for sr in all_seed_results if "m_similarity" in sr]
        if sim_seeds:
            labels = [
                f"final(stim) ↔ {stage1_tasks[0]}(stim)",
                f"final(go) ↔ {stage1_tasks[1]}(go)",
                f"{stage1_tasks[0]}(stim) ↔ {stage1_tasks[1]}(stim)",
                f"{stage1_tasks[0]}(go) ↔ {stage1_tasks[1]}(go)",
            ]
            suffixes = [
                "final_vs_stage1_task0_stim",
                "final_vs_stage1_task1_go",
                "stage1_task0_vs_task1_stim",
                "stage1_task0_vs_task1_go",
            ]

            def _render_msim_figure(prefix_cos, prefix_frob, title_tag, file_tag):
                """Build the 1×2 (cosine | Frobenius diff) figure for either
                the direction-marginalized or the stimulus-aligned keys."""
                cos_keys = [f"{prefix_cos}_{s}" for s in suffixes]
                frob_keys = [f"{prefix_frob}_{s}" for s in suffixes]
                stacked = {k: np.array([sr["m_similarity"][k] for sr in sim_seeds])
                           for k in cos_keys + frob_keys}

                print(f"\n[{title_tag} {ruleset}, {len(sim_seeds)} seeds]")
                for k, v in stacked.items():
                    print(f"  {k:52s}  mean={v.mean():.4f}  std={v.std():.4f}  "
                          f"min={v.min():.4f}  max={v.max():.4f}")

                xs = np.arange(len(labels))
                figsim, axsim = plt.subplots(1, 2, figsize=(10, 3.5))
                for sr_idx, sr in enumerate(sim_seeds):
                    axsim[0].plot(xs, [sr["m_similarity"][k] for k in cos_keys],
                                  "o-", color=c_vals[1 + sr_idx], alpha=0.7,
                                  label=f"seed {sr['seed']}")
                    axsim[1].plot(xs, [sr["m_similarity"][k] for k in frob_keys],
                                  "o-", color=c_vals[1 + sr_idx], alpha=0.7,
                                  label=f"seed {sr['seed']}")
                axsim[0].plot(xs, [stacked[k].mean() for k in cos_keys], "ks-",
                              linewidth=2, markersize=8, label="mean")
                axsim[1].plot(xs, [stacked[k].mean() for k in frob_keys], "ks-",
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
                    f"{outpath}/{ruleset}_{chosen_network}_{addon_name}_{file_tag}.png",
                    dpi=300)
                plt.close(figsim)

            # Direction-marginalized (unchanged behavior).
            _render_msim_figure(
                prefix_cos="cos", prefix_frob="frob",
                title_tag="period-matched M similarity",
                file_tag="m_similarity",
            )
            # Stimulus-aligned (new): per-direction, then averaged across
            # directions. Tests whether the unaligned numbers were hiding
            # anti-direction cancellation.
            _render_msim_figure(
                prefix_cos="cos_aligned", prefix_frob="frob_aligned",
                title_tag="stimulus-aligned M similarity",
                file_tag="m_similarity_stimaligned",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-ruleset combined accuracy + loss figure
    # Same 2×2 layout as the per-ruleset figure, but all seeds of each ruleset
    # share one color so the two rulesets are visually separable.
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
        figcmp.savefig(f"{outpath}/{combined_tag}_{chosen_network}_{addon_name}_learning.png", dpi=300)
        plt.close(figcmp)

        # ─────────────────────────────────────────────────────────────────
        # Transfer-speed summary: iterations to first reach each accuracy
        # threshold during post-training. Lower = faster transfer. Per
        # seed + ruleset-mean, plotted as a function of threshold so you
        # can see where (if anywhere) the two rulesets separate.
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
        axts.set_xscale("log")
        axts.set_title("Transfer speed — iterations to threshold")
        axts.legend(fontsize=7)

        figts.suptitle(f"{chosen_network} | Transfer speed by ruleset",
                       fontsize=12)
        figts.tight_layout()
        figts.savefig(
            f"{outpath}/{combined_tag}_{chosen_network}_{addon_name}_transfer_speed.png",
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
        ts_pkl_path = f"{outpath}/{combined_tag}_{chosen_network}_{addon_name}_transfer_speed.pkl"
        with open(ts_pkl_path, "wb") as f:
            pickle.dump(transfer_speed_data, f)
        print(f"  Saved transfer speed data: {ts_pkl_path}")

        # ─────────────────────────────────────────────────────────────────
        # Rule-input vector geometry, compared across rulesets.
        # The 200-dim vector learned in stage 2 is the only thing stage 2
        # adjusts. Where it lives relative to the two pretrained rule
        # vectors is the sharpest read on "does the novel task's rule
        # input reuse the pretrained rule subspace (Driscoll) or carve
        # out a new direction (MPN-distinctive)?"
        # ─────────────────────────────────────────────────────────────────
        have_rule_vecs = any(
            "rule_vectors" in sr for rs_srs in all_results_by_ruleset.values()
            for sr in rs_srs)
        if have_rule_vecs:
            rs_list = list(all_results_by_ruleset.keys())
            # Per-ruleset lists of scalars we care about.
            def _vals(rs, key):
                return np.array([sr["rule_vectors"][key]
                                 for sr in all_results_by_ruleset[rs]
                                 if "rule_vectors" in sr], dtype=float)

            # Console summary.
            print("\n[Rule-input vector geometry (stage-2 column vs "
                  "pretraining rule columns)]")
            for rs in rs_list:
                n_ok = len(_vals(rs, "in_span_fraction"))
                if n_ok == 0:
                    continue
                print(f"  {rs} ({n_ok} seeds)")
                for k in ("cos_pre0_pre1", "cos_novel_pre0",
                         "cos_novel_pre1", "in_span_fraction"):
                    v = _vals(rs, k)
                    print(f"    {k:22s}  mean={v.mean():.4f}  std={v.std():.4f}")

            # Figure: pairwise cosine bars grouped by ruleset. Each
            # ruleset gets its own trio of bars, x-tick-labeled with that
            # ruleset's actual stage-1 task names. Error bars = std across
            # seeds; overlaid black dots = per-seed values.
            figrv, axrv_cos = plt.subplots(1, 1, figsize=(7, 3.8))

            cos_keys = ["cos_novel_pre0", "cos_novel_pre1", "cos_pre0_pre1"]
            trio_width = 0.8
            bar_width = trio_width / len(cos_keys)
            # 4-unit separation between ruleset groups so the trios don't
            # overlap when there are multiple rulesets.
            group_step = len(cos_keys) + 1
            all_x, all_labels = [], []

            for rs_idx, rs in enumerate(rs_list):
                color = ruleset_colors.get(rs, c_vals[0])
                s1_tasks = _stage1_tasks_for(rs)
                cos_labels = [
                    f"{final_task} ↔ {s1_tasks[0]}",
                    f"{final_task} ↔ {s1_tasks[1]}",
                    f"{s1_tasks[0]} ↔ {s1_tasks[1]}",
                ]
                xs_group = rs_idx * group_step + np.arange(len(cos_keys))
                means = np.array([_vals(rs, k).mean() for k in cos_keys])
                stds = np.array([_vals(rs, k).std() for k in cos_keys])
                axrv_cos.bar(xs_group, means, bar_width * 2.5,
                             yerr=stds, capsize=3, color=color,
                             alpha=0.8, label=rs)
                for k_idx, k in enumerate(cos_keys):
                    vals = _vals(rs, k)
                    axrv_cos.plot(
                        np.full_like(vals, xs_group[k_idx]), vals,
                        "k.", markersize=3, alpha=0.6)
                all_x.extend(xs_group.tolist())
                all_labels.extend(cos_labels)

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
                f"{outpath}/{combined_tag}_{chosen_network}_{addon_name}_rule_vectors.png",
                dpi=300)
            plt.close(figrv)

            # Save rule vector data for paper_plot reuse
            rule_vec_data = {"by_ruleset": {}}
            for rs in rs_list:
                s1_tasks = _stage1_tasks_for(rs)
                rule_vec_data["by_ruleset"][rs] = {
                    "cos_novel_pre0": _vals(rs, "cos_novel_pre0").tolist(),
                    "cos_novel_pre1": _vals(rs, "cos_novel_pre1").tolist(),
                    "cos_pre0_pre1": _vals(rs, "cos_pre0_pre1").tolist(),
                    "in_span_fraction": _vals(rs, "in_span_fraction").tolist(),
                    "stage1_tasks": s1_tasks,
                    "final_task": final_task,
                }
            rv_pkl_path = f"{outpath}/{combined_tag}_{chosen_network}_{addon_name}_rule_vectors.pkl"
            with open(rv_pkl_path, "wb") as f:
                pickle.dump(rule_vec_data, f)
            print(f"  Saved rule vector data: {rv_pkl_path}")

        # ─────────────────────────────────────────────────────────────────
        # Backbone-probe figure: how close is the stage-2 STARTING state
        # (random rule-vector init, everything else frozen from stage-1 end)
        # to solving delayanti? Lower loss = pretraining backbone already
        # "knows" delayanti; explains asymptote differences between
        # rulesets directly.
        # ─────────────────────────────────────────────────────────────────
        have_probe = any(
            "backbone_probe" in sr
            for rs_srs in all_results_by_ruleset.values() for sr in rs_srs)
        if have_probe:
            print(f"\n[Backbone probe — stage-2 initial-state loss & "
                  f"accuracy on {final_task}, per ruleset]")
            # Collect per-seed stats: each seed's backbone_probe has
            # per-random-init arrays; aggregate to a single scalar per seed
            # (mean across random inits) so every seed contributes equally.
            per_seed_loss = {}
            per_seed_loss_out = {}
            per_seed_acc = {}
            for rs, seed_results in all_results_by_ruleset.items():
                loss_list, loss_out_list, acc_list = [], [], []
                for sr in seed_results:
                    if "backbone_probe" not in sr:
                        continue
                    loss_list.append(float(np.mean(sr["backbone_probe"]["loss"])))
                    loss_out_list.append(float(np.mean(sr["backbone_probe"]["loss_out"])))
                    acc_list.append(float(np.mean(sr["backbone_probe"]["acc"])))
                per_seed_loss[rs] = np.array(loss_list)
                per_seed_loss_out[rs] = np.array(loss_out_list)
                per_seed_acc[rs] = np.array(acc_list)
                if loss_list:
                    v_loss = np.array(loss_list)
                    v_acc = np.array(acc_list)
                    print(f"  {rs}  seeds={len(v_loss)}  "
                          f"loss mean={v_loss.mean():.4f}±{v_loss.std():.4f}  "
                          f"acc mean={v_acc.mean():.4f}±{v_acc.std():.4f}")

            # Figure: three panels (full loss, output-only loss, accuracy).
            # Each shows per-ruleset per-seed scatter + boxplot.
            figbp, axbp = plt.subplots(1, 3, figsize=(12, 3.8))
            panel_specs = [
                ("Total loss (output + reg)", per_seed_loss, "loss"),
                ("Output MSE (label only)", per_seed_loss_out, "output MSE"),
                (f"{final_task} accuracy", per_seed_acc, "accuracy"),
            ]
            for panel_idx, (title, data_by_rs, ylabel) in enumerate(panel_specs):
                ax = axbp[panel_idx]
                rs_list_here = list(data_by_rs.keys())
                positions = list(range(len(rs_list_here)))
                box_data = [data_by_rs[rs] for rs in rs_list_here]
                ax.boxplot(box_data, positions=positions, widths=0.5,
                           showfliers=False, patch_artist=False)
                for xpos, rs in enumerate(rs_list_here):
                    color = ruleset_colors.get(rs, c_vals[0])
                    vals = data_by_rs[rs]
                    jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.04
                    ax.plot(xpos + jitter, vals, "o", color=color,
                            alpha=0.7, markersize=7)
                ax.set_xticks(positions)
                ax.set_xticklabels(rs_list_here, rotation=10)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
            # Accuracy panel: anchor to [0, 1] for readability.
            axbp[2].set_ylim([-0.02, 1.02])

            figbp.suptitle(
                f"{chosen_network} | Backbone probe on {final_task} "
                f"(random rule-vector init, compute_loss/compute_acc)",
                fontsize=11)
            figbp.tight_layout()
            figbp.savefig(
                f"{outpath}/{combined_tag}_{chosen_network}_{addon_name}_backbone_probe.png",
                dpi=300)
            plt.close(figbp)

        print(f"\nDone. Results saved to {outpath}/")
    else:
        print("\nNo rulesets produced results. Nothing to plot.")
