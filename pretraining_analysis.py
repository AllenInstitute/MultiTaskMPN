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
from pathlib import Path

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
os.makedirs(outpath, exist_ok=True)

# Pretraining ruleset key (must match rules_dict in pretraining.py)
# ruleset = "fdanti_delaygo"
ruleset = "fdgo_delaygo"
if ruleset == "fdanti_delaygo":
    # Individual task names within the pretraining ruleset
    stage1_tasks = ["fdanti", "delaygo"]
else:
    stage1_tasks = ["fdgo", "delaygo"]

# Post-training task
final_task = "delayanti"

chosen_network = "dmpn"
N = 200

# Naming components that form addon_name in pretraining.py:
#   addon_name = f"+hidden{N}+L21e4+batch{batch}+{metric}"
metric = "angle"
addon_name = f"+hidden{N}+L21e4+batch128+{metric}"

# Comparison: which pretraining task period to compare against post-training
# stage1_tasks[0] → stimulus period; stage1_tasks[1] → go/response period
stage1_compare_period = {
    stage1_tasks[0]: "stim1",
    stage1_tasks[1]: "go1",
}
final_compare_period = {
    "stim": "stim1",
    "go": "go1",
}

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
    • Modulation (pre/post): mirrors the hidden-state pattern, confirming that
      the reused geometry extends to the synaptic plasticity structure.
    • Full modulation (n_hidden² dims): too high-dimensional to interpret via
      CVE or PR; serves only as a sanity check that variance is captured.

    datatype controls reshaping before PCA
    ───────────────────────────────────────
    "hidden"         — (batch×time, n_hidden): neuron activations
    "modulation"     — (batch×time, n_hidden²): full M matrix flattened;
                       PR skipped (uninformative at this dimensionality)
    "modulation_pre" — each row of M as a pre-synaptic weight vector;
                       shape (batch×time×n_hidden, n_hidden)
    "modulation_post"— each column of M as a post-synaptic weight vector;
                       shape (batch×time×n_hidden, n_hidden)
    """
    if datatype == "hidden":
        X2d = X.reshape(-1, X.shape[-1])
        Y2d = Y.reshape(-1, Y.shape[-1])
    elif datatype == "modulation":
        # Full M flattened: n_hidden² features — very high-dimensional.
        X2d = X.reshape(-1, X.shape[-1] * X.shape[-2])
        Y2d = Y.reshape(-1, Y.shape[-1] * Y.shape[-2])
    elif datatype == "modulation_post":
        # Each column of M = incoming plasticity weights onto one post-synaptic neuron.
        X2d = X.reshape(-1, X.shape[-1])
        Y2d = Y.reshape(-1, Y.shape[-1])
    elif datatype == "modulation_pre":
        # Each row of M = outgoing plasticity weights from one pre-synaptic neuron.
        # Swap last two axes so rows become the sample axis before flattening.
        X2d = np.swapaxes(X, -1, -2).reshape(-1, X.shape[-2])
        Y2d = np.swapaxes(Y, -1, -2).reshape(-1, Y.shape[-2])
    else:
        raise ValueError(f"Unknown datatype: {datatype}")

    assert X2d.shape[1] == Y2d.shape[1]

    pca = PCA(n_components=n_components)
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

    if datatype != "modulation":
        X_c = X2d - X2d.mean(axis=0, keepdims=True)
        cov_X = (X_c.T @ X_c) / X_c.shape[0]
        PR_X = _participation_ratio(cov_X)

        Y_c = Y2d - Y2d.mean(axis=0, keepdims=True)
        cov_Y = (Y_c.T @ Y_c) / Y_c.shape[0]
        PR_Y = _participation_ratio(cov_Y)

        # PR of Y after projection: how many of X's PCs does Y actually use?
        cov_Yp = (Y_proj.T @ Y_proj) / Y_proj.shape[0]
        PR_Y_in_Xbasis = _participation_ratio(cov_Yp)
    else:
        # Full modulation is n_hidden²-dimensional; PR is not meaningful.
        PR_X, PR_Y, PR_Y_in_Xbasis = np.nan, np.nan, np.nan

    return {
        "evr_X": evr_X, "cev_X": cev_X,
        "evr_Y": evr_Y, "cev_Y": cev_Y,
        "PR_X": PR_X, "PR_Y": PR_Y, "PR_Y_in_Xbasis": PR_Y_in_Xbasis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    seeds = discover_seeds()
    if not seeds:
        raise FileNotFoundError(
            f"No matching result files found in {basepath}/ for "
            f"ruleset={ruleset}, network={chosen_network}, addon={addon_name}"
        )
    print(f"Found {len(seeds)} seeds: {seeds}")

    period_shift_percentage = 1 / 4

    # Determine which analyses to run based on network type
    if chosen_network == "dmpn":
        analysis_types = ["hidden", "modulation", "modulation_pre", "modulation_post"]
    else:
        analysis_types = ["hidden"]

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

            # Validation accuracy
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
            # Stimulus period: compare stage1_tasks[0] vs final_task
            stage1_stim = period_slice(
                stage1_hs, stage1_rules_epochs, stage1_tasks[0], "stim1", mask=mask0)
            final_stim = period_slice(
                stage2_hs, stage2_rules_epochs, final_task, "stim1")

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
                    stage1_ms, stage1_rules_epochs, stage1_tasks[0], "stim1", mask=mask0)
                final_stim_m = period_slice(
                    stage2_ms, stage2_rules_epochs, final_task, "stim1")
                stage1_go_m = period_slice(
                    stage1_ms, stage1_rules_epochs, stage1_tasks[1], "go1",
                    shift_percentage=period_shift_percentage, mask=mask1)
                final_go_m = period_slice(
                    stage2_ms, stage2_rules_epochs, final_task, "go1",
                    shift_percentage=period_shift_percentage)

                for dtype in ["modulation", "modulation_pre", "modulation_post"]:
                    n_comp = N if dtype != "modulation" else None
                    res_m_stim = pca_cross_variance(
                        final_stim_m, stage1_stim_m, n_components=n_comp, datatype=dtype)
                    res_m_go = pca_cross_variance(
                        final_go_m, stage1_go_m, n_components=n_comp, datatype=dtype)
                    seed_result[dtype] = {
                        "stimulus": res_m_stim,
                        "response": res_m_go,
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
                x_up = 20 if dtype != "modulation" else N**2 // 2

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

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate across seeds
    # ─────────────────────────────────────────────────────────────────────────
    if not all_seed_results:
        print("\nNo seeds processed successfully. Exiting.")
    else:
        print(f"\n{'='*60}")
        print(f"  Aggregating {len(all_seed_results)} seeds")
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
            x_up = 20 if dtype != "modulation" else N**2 // 2
            for col, period in enumerate(["stimulus", "response"]):
                all_cev_X = []
                all_cev_Y = []
                for sr in all_seed_results:
                    if dtype not in sr:
                        continue
                    res = sr[dtype][period]
                    all_cev_X.append(res["cev_X"])
                    all_cev_Y.append(res["cev_Y"])

                    xs = np.arange(1, len(res["cev_X"]) + 1)
                    # thin lines = individual seeds; X = delayanti (fit), Y = one pre-train task (proj)
                    axs[row, col].plot(xs, res["cev_X"], color=c_vals[0], alpha=0.3)
                    axs[row, col].plot(xs, res["cev_Y"],
                                       color=c_vals[1 + all_seed_results.index(sr)],
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

        # Learning curve summary
        figacc, axsacc = plt.subplots(2, 2, figsize=(4 * 2, 4 * 2))
        for sr in all_seed_results:
            lc = sr["learning"]
            for j in range(2):
                axsacc[0, j].plot(lc["acc_iter_post"], lc["acc_post"],
                                  color=c_vals[1 + all_seed_results.index(sr)],
                                  alpha=0.7, label=f"seed {sr['seed']}")
                axsacc[1, j].plot(lc["acc_iter_pre"], lc["acc_pre"],
                                  color=c_vals[1 + all_seed_results.index(sr)],
                                  alpha=0.7, label=f"seed {sr['seed']}")

        axsacc[0, 0].set_title("Post-training accuracy")
        axsacc[1, 0].set_title("Pre-training accuracy")
        for j in range(2):
            axsacc[0, j].set_xlabel("# Datasets (post-training)")
            axsacc[1, j].set_xlabel("# Datasets (pre-training)")
            axsacc[0, j].set_ylabel("Validation Accuracy")
            axsacc[1, j].set_ylabel("Validation Accuracy")
            axsacc[0, j].legend(fontsize=6)
            axsacc[1, j].legend(fontsize=6)
        axsacc[0, 0].set_ylim([0, 1])
        axsacc[0, 1].set_ylim([0.8, 1])
        axsacc[1, 0].set_ylim([0, 1])
        axsacc[1, 1].set_ylim([0.95, 1])

        figacc.suptitle(f"{ruleset} | {chosen_network} | Accuracy", fontsize=12)
        figacc.tight_layout()
        figacc.savefig(f"{outpath}/{ruleset}_{chosen_network}_accuracy.png", dpi=300)
        plt.close(figacc)

        print(f"\nDone. Results saved to {outpath}/")
