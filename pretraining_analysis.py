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
import shutil
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

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
# Wipe stale analysis outputs from previous runs so results can't be
# confused across code versions — everything below is regenerated.
if os.path.isdir(outpath):
    shutil.rmtree(outpath)
os.makedirs(outpath, exist_ok=True)

# Pretraining rulesets to run (must match rules_dict in pretraining.py).
# Each one is processed independently; their learning curves are then
# combined into a single cross-ruleset figure.
rulesets_to_run = ["fdgo_delaygo", "fdanti_delaygo"]

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
N = 200
# PCA component cap for modulation / modulation_weighted analyses.
# Hidden uses N (the ambient dim). Modulation lives in N² = 40 000 dims, so a
# larger cap is needed to see whether CVE eventually saturates. sklearn still
# requires n_components ≤ min(n_samples, n_features); the call sites clamp to
# that limit to stay safe.
N_MOD_PCS = 1000

# Naming components that form addon_name in pretraining.py:
#   addon_name = f"+hidden{N}+L21e4+batch{batch}+{metric}"
metric = "angle"
addon_name = f"+hidden{N}+L21e4+batch128+{metric}"

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


def load_mpn_W(seed):
    """
    Load the frozen plastic-layer weight W from the saved checkpoint.
    Returns an (N, N) numpy array. W is identical across stage 1 and stage 2
    (frozen in expand_and_freeze), so a single W multiplies both stages' M.
    """
    ckpt = torch.load(ckpt_path(seed), map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["mp_layer1.W"].numpy()


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


def _mean_end_of_trial_M(Ms, mask=None):
    """
    Trial-averaged M at the final timestep of each trial.

    Ms shape: (batch, time, N, N). If `mask` is given (bool over the batch
    axis), restrict to those trials first. Returns (N, N) numpy array.
    """
    if mask is not None:
        Ms = Ms[mask]
    # -1 along time axis = last timestep of each trial; then average over trials.
    return Ms[:, -1, :, :].mean(axis=0)


def _cosine_sim(A, B):
    """Cosine similarity between two tensors, flattened."""
    a, b = np.asarray(A).ravel(), np.asarray(B).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


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
    Cross-validated PCA geometry analysis.

    Fits PCA on X (= post-training task, the "reference" subspace), then
    projects Y (= pre-training task) into that basis.  The cumulative variance
    explained (CVE) of Y in X's PCs measures how much the pre-training task
    recycles the same representational geometry as the post-training task.

    Interpretation of outputs
    ─────────────────────────
    cev_X  : CVE of X in its own PCs — should saturate to 1.0, confirming the
             basis spans X's full variance (reference curve).
    cev_Y  : CVE of Y projected into X's PCs.
             High → pre-training representations largely overlap with post-training.
             Low  → pre-training uses a distinct, orthogonal subspace.
    PR_X / PR_Y : participation ratio (effective dimensionality) of each
             representation in its native space.
    PR_Y_in_Xbasis : effective dimensionality of Y after projection into X's
             basis; lower than PR_Y means the shared subspace is more
             concentrated than Y's full representation.

    Observed pattern (fdgo_delaygo → delayanti)
    ────────────────────────────────────────────
    • Stimulus period: cev_Y is low (~0.5–0.7) — stimulus encoding is
      task-specific; pre-training tasks use a different stimulus subspace.
    • Response/go period: cev_Y is high (~0.9–1.0) — response-period dynamics
      are shared; the post-training task reuses the same go-period subspace.
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

    # Randomized SVD is much faster than the full solver when we only need
    # the top k components (k ≪ min(n_samples, n_features)), especially for
    # modulation where n_features = n_hidden² is large.
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(X2d)

    evr_X = pca.explained_variance_ratio_
    cev_X = np.cumsum(evr_X)

    # Center Y on X's mean: asks how much of Y's variance falls in X's subspace
    # when measured from X's origin (center_on="X" is the standard choice).
    if center_on == "X":
        Y_centered = Y2d - pca.mean_
    elif center_on == "Y":
        Y_centered = Y2d - Y2d.mean(axis=0, keepdims=True)
    else:
        raise ValueError('center_on must be "X" or "Y"')

    Y_proj = Y_centered @ pca.components_.T  # project Y into X's PC basis

    var_total_Y = np.var(Y_centered, axis=0, ddof=0).sum()
    if var_total_Y == 0:
        evr_Y = np.zeros(pca.components_.shape[0])
    else:
        # Fraction of Y's total variance captured by each of X's PCs.
        evr_Y = np.var(Y_proj, axis=0, ddof=0) / var_total_Y
    cev_Y = np.cumsum(evr_Y)

    X_c = X2d - X2d.mean(axis=0, keepdims=True)
    PR_X = _pr_from_data(X_c)

    Y_c = Y2d - Y2d.mean(axis=0, keepdims=True)
    PR_Y = _pr_from_data(Y_c)

    # PR of Y after projection: how many of X's PCs does Y actually use?
    # Y_proj is (n_samples, n_components), so this is always cheap.
    cov_Yp = (Y_proj.T @ Y_proj) / Y_proj.shape[0]
    PR_Y_in_Xbasis = _participation_ratio(cov_Yp)

    return {
        "evr_X": evr_X, "cev_X": cev_X,
        "evr_Y": evr_Y, "cev_Y": cev_Y,
        "PR_X": PR_X, "PR_Y": PR_Y, "PR_Y_in_Xbasis": PR_Y_in_Xbasis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    period_shift_percentage = 1 / 4

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

                seed_result = {"seed": seed}

                # Hidden state analysis
                res_h_stim = pca_cross_variance(
                    final_stim, stage1_stim, n_components=N, datatype="hidden")
                res_h_go = pca_cross_variance(
                    final_go, stage1_go, n_components=N, datatype="hidden")
                seed_result["hidden"] = {
                    "stimulus": res_h_stim,
                    "response": res_h_go,
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

                    # Clamp n_components to n_samples of the fit array
                    # (sklearn PCA errors if n_components > min(n, n_features)).
                    # n_samples = batch × time_in_period after period_slice.
                    n_stim_samples = final_stim_m.shape[0] * final_stim_m.shape[1]
                    n_go_samples = final_go_m.shape[0] * final_go_m.shape[1]
                    n_comp_stim = min(N_MOD_PCS, n_stim_samples)
                    n_comp_go = min(N_MOD_PCS, n_go_samples)

                    res_m_stim = pca_cross_variance(
                        final_stim_m, stage1_stim_m,
                        n_components=n_comp_stim, datatype="modulation")
                    res_m_go = pca_cross_variance(
                        final_go_m, stage1_go_m,
                        n_components=n_comp_go, datatype="modulation")
                    seed_result["modulation"] = {
                        "stimulus": res_m_stim,
                        "response": res_m_go,
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
                            final_stim_wm, stage1_stim_wm,
                            n_components=n_comp_stim,
                            datatype="modulation_weighted")
                        res_wm_go = pca_cross_variance(
                            final_go_wm, stage1_go_wm,
                            n_components=n_comp_go,
                            datatype="modulation_weighted")
                        seed_result["modulation_weighted"] = {
                            "stimulus": res_wm_stim,
                            "response": res_wm_go,
                        }
                    except (FileNotFoundError, KeyError) as e:
                        print(f"  WARNING: could not load W from checkpoint "
                              f"({e}); skipping modulation_weighted for this seed")

                    # ─── End-of-trial M similarity across stages ─────────
                    # Direct probe of whether stage-2 plasticity drifts away
                    # from its pretraining end-state under the same frozen W.
                    # If cos_sim ≈ 1, plasticity is mostly inert in stage 2
                    # (MPN ≈ vanilla RNN + input); if < 1, the new rule input
                    # actively reshapes W_eff during the post-training task.
                    M1_task0 = _mean_end_of_trial_M(stage1_ms, mask=mask0)
                    M1_task1 = _mean_end_of_trial_M(stage1_ms, mask=mask1)
                    M2_final = _mean_end_of_trial_M(stage2_ms)

                    frob = lambda A: float(np.linalg.norm(A))
                    seed_result["m_similarity"] = {
                        # Cross-stage, task-aligned (delayanti vs each pretraining task).
                        "cos_final_vs_stage1_task0": _cosine_sim(M2_final, M1_task0),
                        "cos_final_vs_stage1_task1": _cosine_sim(M2_final, M1_task1),
                        # Within-stage-1 baseline: how different are the two
                        # pretraining tasks' end-of-trial M from each other?
                        # Sets the scale for "meaningful" drift.
                        "cos_stage1_task0_vs_task1": _cosine_sim(M1_task0, M1_task1),
                        # Frobenius norms for magnitude context.
                        "fro_stage1_task0": frob(M1_task0),
                        "fro_stage1_task1": frob(M1_task1),
                        "fro_final": frob(M2_final),
                    }
                elif chosen_network == "dmpn":
                    print("  WARNING: Ms_orig is empty for this seed (vanilla fallback?)")

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
                        xs = np.arange(1, len(res["cev_X"]) + 1)
                        # X = post-training task (final_task, e.g. delayanti): PCA fit on this
                        # Y = one pre-training task (period-specific): projected into X's PC basis
                        axs[row, period_idx].plot(xs, res["cev_X"], '-o', markersize=2,
                                                 color=c_vals[0], label=f"{final_task} (fit)")
                        axs[row, period_idx].plot(xs, res["cev_Y"], '-o', markersize=2,
                                                 color=c_vals[1], label=f"{period_to_stage1[period]} (proj)")
                        axs[row, period_idx].set_xlim(0, x_up)
                        axs[row, period_idx].set_ylim(0, 1.05)
                        axs[row, period_idx].set_xlabel("# PCs")
                        axs[row, period_idx].set_ylabel("Cumulative Var. Explained")
                        axs[row, period_idx].set_title(f"{dtype} — {period}")
                        axs[row, period_idx].legend(fontsize=8)

                    # PR bar chart
                    res_stim = seed_result[dtype]["stimulus"]
                    res_go = seed_result[dtype]["response"]
                    pr_labels = ["PR_X\n(stim)", "PR_Y\n(stim)", "PR_Y|X\n(stim)",
                                 "PR_X\n(go)", "PR_Y\n(go)", "PR_Y|X\n(go)"]
                    pr_vals = [res_stim["PR_X"], res_stim["PR_Y"], res_stim["PR_Y_in_Xbasis"],
                               res_go["PR_X"], res_go["PR_Y"], res_go["PR_Y_in_Xbasis"]]
                    colors = [c_vals[0]] * 3 + [c_vals[1]] * 3
                    axs[row, 2].bar(range(len(pr_vals)), pr_vals, color=colors, alpha=0.7)
                    axs[row, 2].set_xticks(range(len(pr_vals)))
                    axs[row, 2].set_xticklabels(pr_labels, fontsize=7)
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
                all_cev_X = []
                all_cev_Y = []
                for sr_idx, sr in enumerate(all_seed_results):
                    if dtype not in sr:
                        continue
                    res = sr[dtype][period]
                    all_cev_X.append(res["cev_X"])
                    all_cev_Y.append(res["cev_Y"])

                    xs = np.arange(1, len(res["cev_X"]) + 1)
                    # thin lines = individual seeds; X = delayanti (fit), Y = one pre-train task (proj)
                    axs[row, col].plot(xs, res["cev_X"], color=c_vals[0], alpha=0.3)
                    axs[row, col].plot(xs, res["cev_Y"],
                                       color=c_vals[1 + sr_idx],
                                       alpha=0.6, label=f"seed {sr['seed']}")

                # Mean curves across seeds (thick lines)
                if all_cev_X:
                    min_len = min(len(c) for c in all_cev_X)
                    mean_X = np.mean([c[:min_len] for c in all_cev_X], axis=0)
                    mean_Y = np.mean([c[:min_len] for c in all_cev_Y], axis=0)
                    xs_mean = np.arange(1, min_len + 1)
                    # black = delayanti mean (reference), gray = pre-train mean (projected)
                    axs[row, col].plot(xs_mean, mean_X, color="black", linewidth=2.5,
                                       label=f"{final_task} mean")
                    axs[row, col].plot(xs_mean, mean_Y, color="gray", linewidth=2.5,
                                       label=f"{period_to_stage1[period]} mean")

                axs[row, col].set_xlim(0, x_up)
                axs[row, col].set_ylim(0, 1.05)
                axs[row, col].set_xlabel("# PCs")
                axs[row, col].set_ylabel("Cumulative Var. Explained")
                axs[row, col].set_title(f"{dtype} — {period}")
                axs[row, col].legend(fontsize=6)

        fig.suptitle(f"{ruleset} | {chosen_network} | {len(all_seed_results)} seeds", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{outpath}/{ruleset}_{chosen_network}_aggregate.png", dpi=300)
        plt.close(fig)

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
        figlc.savefig(f"{outpath}/{ruleset}_{chosen_network}_learning.png", dpi=300)
        plt.close(figlc)

        if not any("loss" in sr for sr in all_seed_results):
            print("No training-history pickles found; loss column in "
                  "learning figure will be empty.")

        # ─── End-of-trial M similarity summary (per ruleset) ───────────────
        # Reports how much the mean end-of-trial plasticity matrix drifts
        # from its pretraining counterpart once the new rule input drives
        # the frozen-W network. A ~1.0 across-stage cosine means plasticity
        # is essentially inert in stage 2; a lower value indicates the
        # novel task actively reshapes W_eff.
        sim_seeds = [sr for sr in all_seed_results if "m_similarity" in sr]
        if sim_seeds:
            keys = [
                "cos_final_vs_stage1_task0",
                "cos_final_vs_stage1_task1",
                "cos_stage1_task0_vs_task1",
            ]
            stacked = {k: np.array([sr["m_similarity"][k] for sr in sim_seeds])
                       for k in keys}
            stacked["fro_final/fro_stage1_task0"] = np.array([
                sr["m_similarity"]["fro_final"] / sr["m_similarity"]["fro_stage1_task0"]
                for sr in sim_seeds])
            stacked["fro_final/fro_stage1_task1"] = np.array([
                sr["m_similarity"]["fro_final"] / sr["m_similarity"]["fro_stage1_task1"]
                for sr in sim_seeds])

            print(f"\n[M-similarity {ruleset}, {len(sim_seeds)} seeds]")
            for k, v in stacked.items():
                print(f"  {k:38s}  mean={v.mean():.4f}  std={v.std():.4f}  "
                      f"min={v.min():.4f}  max={v.max():.4f}")

            # Small bar/strip figure: cosine similarities per seed.
            figsim, axsim = plt.subplots(1, 2, figsize=(4 * 2, 3))
            cos_labels = [
                f"final ↔ {stage1_tasks[0]}",
                f"final ↔ {stage1_tasks[1]}",
                f"{stage1_tasks[0]} ↔ {stage1_tasks[1]}",
            ]
            cos_vals = [stacked[k] for k in keys]
            xs = np.arange(len(cos_labels))
            for sr_idx, sr in enumerate(sim_seeds):
                seed_cos = [sr["m_similarity"][k] for k in keys]
                axsim[0].plot(xs, seed_cos, "o-", color=c_vals[1 + sr_idx],
                              alpha=0.7, label=f"seed {sr['seed']}")
            axsim[0].plot(xs, [v.mean() for v in cos_vals], "ks-",
                          linewidth=2, markersize=8, label="mean")
            axsim[0].set_xticks(xs)
            axsim[0].set_xticklabels(cos_labels, rotation=15)
            axsim[0].set_ylim([-0.05, 1.05])
            axsim[0].set_ylabel("Cosine similarity")
            axsim[0].set_title("End-of-trial M cosine similarity")
            axsim[0].legend(fontsize=6)

            fro_labels = [
                f"final / {stage1_tasks[0]}",
                f"final / {stage1_tasks[1]}",
            ]
            fro_keys = ["fro_final/fro_stage1_task0", "fro_final/fro_stage1_task1"]
            xs2 = np.arange(len(fro_labels))
            for sr_idx, sr in enumerate(sim_seeds):
                f0 = sr["m_similarity"]["fro_final"] / sr["m_similarity"]["fro_stage1_task0"]
                f1 = sr["m_similarity"]["fro_final"] / sr["m_similarity"]["fro_stage1_task1"]
                axsim[1].plot(xs2, [f0, f1], "o-", color=c_vals[1 + sr_idx],
                              alpha=0.7, label=f"seed {sr['seed']}")
            axsim[1].plot(xs2, [stacked[k].mean() for k in fro_keys], "ks-",
                          linewidth=2, markersize=8, label="mean")
            axsim[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
            axsim[1].set_xticks(xs2)
            axsim[1].set_xticklabels(fro_labels, rotation=15)
            axsim[1].set_ylabel("‖M_final‖ / ‖M_stage1‖")
            axsim[1].set_title("End-of-trial M magnitude ratio")
            axsim[1].legend(fontsize=6)

            figsim.suptitle(f"{ruleset} | {chosen_network} | "
                            f"M drift (stage 2 vs stage 1)", fontsize=11)
            figsim.tight_layout()
            figsim.savefig(f"{outpath}/{ruleset}_{chosen_network}_m_similarity.png",
                           dpi=300)
            plt.close(figsim)

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
        for ax in axscmp.flatten():
            ax.legend(fontsize=7)

        combined_tag = "_".join(all_results_by_ruleset.keys())
        figcmp.suptitle(f"{chosen_network} | Learning curves by ruleset", fontsize=12)
        figcmp.tight_layout()
        figcmp.savefig(f"{outpath}/{combined_tag}_{chosen_network}_learning.png", dpi=300)
        plt.close(figcmp)

        print(f"\nDone. Results saved to {outpath}/")
    else:
        print("\nNo rulesets produced results. Nothing to plot.")
