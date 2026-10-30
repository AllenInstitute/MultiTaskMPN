"""
Lesion and pruning experiments for a trained multi-task MPN.

Given a trained DeepMultiPlasticNet and its neuron/synapse cluster assignments
(from multiple_task_analysis.py), this script systematically ablates groups of
neurons or synapses and measures the resulting per-task accuracy change. The
goal is to identify which clusters are functionally necessary for which tasks,
revealing the network's modular organization.

Experiments:
1. Single-cluster lesion (input & hidden) — zero all connections to/from one
   neuron cluster at a time and measure per-task accuracy. A size-matched
   random lesion serves as control; normalized effect = random - cluster.
   Controls are computed once per (task, side, lesion size) and shared by
   same-size conditions — the random draw never depends on cluster identity,
   so shared controls are identically distributed (same for the combined and
   modulation lesions below, keyed by size pair / synapse count).
2. Combined lesion (input × hidden) — simultaneously lesion one input cluster
   and one hidden cluster for all (i, j) combinations to test interactions.
3. Modulation lesion — for each synapse cluster of every clustering variant
   listed in core/modulation_variants.py, either zero the static
   weight W at those synapses (remove connectivity) or freeze M at those
   synapses (remove plasticity), to dissect the contributions of static vs.
   plastic pathways.
4. Magnitude pruning — zero the lowest-magnitude fraction of W at increasing
   sparsity levels (0–99.9%) to assess weight redundancy.

All experiments score the same per-task test set, generated once at the start
(test_n_batch trials per task; the combined lesion uses the first
combined_test_n_batch of them), so baselines and effects are comparable across
experiments and no trials are regenerated inside the loops. The test set is
not seeded and not saved; lesion_controls.py scores the single-cluster
conditions again on a seeded one.

The lesion operations, the fixed-k cluster definitions and the
condition/control bookkeeping live in core/lesion_ops.py, shared with
lesion_controls.py so both scripts lesion exactly the same clusters.

Outputs:
  - Heatmaps of per-task accuracy under each lesion condition
  - lesion_prune_results_{aname}.pkl — full results dict for downstream
    analysis in lesion_plot.py

No CLI entry point: `main(seed, feature)` is invoked by run_pipeline.py.
"""
import os
import numpy as np
import seaborn as sns
import pickle
import gc
from scipy.spatial.distance import pdist, squareform

import matplotlib as mpl 
import matplotlib.pyplot as plt

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

import torch 

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper
import lesion_ops
from modulation_variants import LESION_MODULATION_TYPES

def main(seed, feature):
    """Run the lesion & pruning experiments for one trained network."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    repeat_num = 10

    aname = lesion_ops.run_name(seed, feature)

    save_dir = f"./multiple_tasks_perf/{aname}"
    # Not cleared on start: every file below is written under a fixed name, so
    # a rerun overwrites its own outputs, and lesion_controls.py keeps its
    # pickle in this directory.
    os.makedirs(save_dir, exist_ok=True)

    cluster_info, cluster_info_mod = lesion_ops.load_cluster_info(aname)
    print(cluster_info.keys())
        
    # need to remind what the clusters are
    cnames = ["input_normalized", "hidden_normalized"]
    for cname in cnames:
        print(f"{cname} clusters; row_clusters: {len(cluster_info[cname]['row_clusters'])}, col_clusters: {len(cluster_info[cname]['col_clusters'])}")

    model, task_params_c, all_tasks = lesion_ops.load_network(aname, device)
    print(model.state_dict().keys())
    
    # ── Fixed-k clustering for lesion experiments ──
    # For input/hidden: shared re-cut helper clustering.fixed_k_col_clusters.
    # For modulation: use result_all["col_labels_by_k"][FIXED_K] (already saved).
    # FIXED_K is inferred from the upstream pickle (global_assignment_fixed_k{N} key).
    FIXED_K, _fk_source = lesion_ops.infer_fixed_k(cluster_info_mod)
    if _fk_source is None:
        print(f"No global_assignment_fixed_k* found in cluster_info_mod; defaulting FIXED_K={FIXED_K}")
    else:
        print(f"Inferred FIXED_K={FIXED_K} from upstream pickle key: {_fk_source}")

    # Derive fixed-k clusters for input/hidden and inject into cluster_info
    # so that lesion_prepost_inplace can look them up via the variant string.
    for _fk_key in lesion_ops.add_fixed_k_neuron_clusters(cluster_info, FIXED_K):
        print(f"Derived {_fk_key}: {len(cluster_info[_fk_key]['col_clusters'])} clusters "
              f"(unresponsive label: {cluster_info[_fk_key]['unresponsive_label']})")

    VARIANT_NORM = lesion_ops.neuron_variant(FIXED_K, normalized=True)
    VARIANT_UNNORM = lesion_ops.neuron_variant(FIXED_K, normalized=False)
    _input_norm_key = lesion_ops.neuron_cluster_key("pre", VARIANT_NORM)
    _hidden_norm_key = lesion_ops.neuron_cluster_key("post", VARIANT_NORM)
    _input_unnorm_key = lesion_ops.neuron_cluster_key("pre", VARIANT_UNNORM)
    _hidden_unnorm_key = lesion_ops.neuron_cluster_key("post", VARIANT_UNNORM)

    pre_n = len(cluster_info[_input_norm_key]["col_clusters"])
    post_n = len(cluster_info[_hidden_norm_key]["col_clusters"])
    pre_n_unnorm = len(cluster_info[_input_unnorm_key]["col_clusters"])
    post_n_unnorm = len(cluster_info[_hidden_unnorm_key]["col_clusters"])
    print(f"Fixed k={FIXED_K}: norm pre={pre_n}, post={post_n}; unnorm pre={pre_n_unnorm}, post={post_n_unnorm}")

    # Cluster mean correlation matrices
    corr_matrices = {}
    l1_dist_matrices = {}
    cluster_means_cache = {}
    for name, n_clusters in [(_input_norm_key, pre_n), (_hidden_norm_key, post_n)]:
        V = cluster_info[name]["cell_vars_rules_sorted_norm"]
        col_clusters = cluster_info[name]["col_clusters"]
        cluster_means = np.stack(
            [V[:, col_clusters[c]].mean(axis=1) for c in range(1, n_clusters + 1)],
            axis=1
        )
        cluster_means_cache[name] = cluster_means
        corr_matrices[name] = np.corrcoef(cluster_means.T)
        l1_dist_matrices[name] = squareform(pdist(cluster_means.T, metric="cityblock"))
        print(f"{name} cluster corr matrix shape: {corr_matrices[name].shape}")

    for name, n_clusters in [(_input_norm_key, pre_n), (_hidden_norm_key, post_n)]:
        corr = corr_matrices[name]
        l1_dist = l1_dist_matrices[name]

        tick_labels = [f"c{c}" for c in range(1, n_clusters + 1)]
        upper_mask = np.triu(np.ones((n_clusters, n_clusters), dtype=bool), k=1)
        panel_size = max(2.5, 0.35 * n_clusters + 1.0)
        fig, axes = plt.subplots(1, 2, figsize=(panel_size * 2 + 1.0, panel_size), dpi=300)

        for ax, mat, title, cmap, vmin, vmax in [
            # Correlation: full symmetric range so negative correlations stay
            # visible — a [0, 1] range clamps them all to the bottom color.
            (axes[0], corr,    f"{name}: correlation",  "RdBu_r", -1.0, 1.0),
            # L1 distance is non-negative: sequential colormap, data-driven
            # range (matches the L1 panels in lesion_plot.py).
            (axes[1], l1_dist, f"{name}: L1 distance",  "viridis", None, None),
        ]:
            hm = sns.heatmap(mat, mask=upper_mask, cmap=cmap, vmin=vmin, vmax=vmax,
                        square=True, cbar=True,
                        linewidths=0.3, linecolor="white",
                        cbar_kws={"shrink": 0.75, "aspect": 25},
                        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(tick_labels, rotation=0, fontsize=7)
            ax.set_xlabel("Neuron Clusters", fontsize=8)
            ax.set_ylabel("Neuron Clusters", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis="both", length=1.5, pad=2, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
            cbar.outline.set_linewidth(0.5)

        fig.tight_layout()
        fig.savefig(f"{save_dir}/cluster_corr_{name}_{aname}.png", dpi=300)
        plt.close(fig)

    print(f"Cluster corr/L1 heatmaps done (k={FIXED_K}).")

    all_comb, all_comb_names_lesion = lesion_ops.neuron_conditions(pre_n, post_n)
    print(f"all_comb_names_lesion: {all_comb_names_lesion}")

    print(model.W_initial_linear.weight.shape, model.mp_layer1.W.shape, model.mp_layer1.b.shape, model.W_output.shape)

    # All modulation clustering types to run lesion experiments on (the five
    # variants registered in core/modulation_variants.py); the second
    # tuple element indexes result_all_lst in the upstream cluster_info_mod
    # pickle — the per-G clustering solutions saved by multiple_task_analysis.py
    # in its G_lst order (index 1 = G=300 for every type currently). The actual
    # G is read back from the pickle's result_all_name_lst, never hard-coded here.
    mod_type_lst_all = [(type_key, 1) for type_key in LESION_MODULATION_TYPES]
    mod_type_lst = [(k, g) for k, g in mod_type_lst_all if k in cluster_info_mod]
    print(f"Modulation types to lesion ({len(mod_type_lst)}): {[k for k,_ in mod_type_lst]}")

    # Modulation-matrix geometry straight from the loaded network — supports a
    # NON-SQUARE plastic layer (n_post hidden x n_pre embed). Flat synapse
    # indices are C-order over (post, pre):
    #   post = flat // n_pre,  pre = flat % n_pre.
    n_post_mod, n_pre_mod = model.mp_layer1.W.shape
    MM = n_post_mod * n_pre_mod
    _first_labels = cluster_info_mod[mod_type_lst[0][0]]["result_all_lst"][mod_type_lst[0][1]]["col_labels"]
    assert len(_first_labels) == MM, (
        f"cluster_info_mod has {len(_first_labels)} synapse labels but the "
        f"checkpoint's plastic layer is {n_post_mod}x{n_pre_mod}={MM} — "
        "analysis pickle and checkpoint do not match")

    def plot_lesion_unit_distribution(all_comb_, all_comb_names_, pre_n_, post_n_,
                                      variant_label, cluster_info_, savepath):
        """Bar chart showing how many neurons each lesion condition removes.
        Returns {condition_name: n_units} for the results pickle."""
        units = [lesion_ops.neuron_cluster_size(cluster_info_, comb, variant_label)
                 for comb in all_comb_]
        for idx, u in enumerate(units):
            print(f"[{variant_label}] Lesion condition: {all_comb_names_[idx]}, lesioned units: {u}")

        bar_colors = []
        for tag, ci in all_comb_:
            if ci is None:
                bar_colors.append("#bdbdbd")
            elif tag == "pre":
                bar_colors.append("#4292c6")
            else:
                bar_colors.append("#e6550d")

        n_bars = len(all_comb_names_)
        fig, ax = plt.subplots(1, 1, figsize=(max(3.5, 0.35 * n_bars + 1), 2.5), dpi=300)
        x = np.arange(n_bars)
        ax.bar(x, units, width=0.7, color=bar_colors, edgecolor="black", linewidth=0.3)

        max_u = max(units) if max(units) > 0 else 1
        for xi, v in enumerate(units):
            if v > 0:
                ax.text(xi, v + max_u * 0.02, str(v),
                        ha="center", va="bottom", fontsize=5.5)

        ax.set_xticks(x)
        ax.set_xticklabels(all_comb_names_, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Lesioned neurons", fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(axis="y", length=1.5, pad=2, width=0.5)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        ax.axvline(pre_n_ + 0.5, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
        ax.text(0.5 * pre_n_, max_u * 1.12,
                "Pre (input)", ha="center", fontsize=7, color="#4292c6", fontweight="bold")
        ax.text(pre_n_ + 1 + 0.5 * post_n_, max_u * 1.12,
                "Post (hidden)", ha="center", fontsize=7, color="#e6550d", fontweight="bold")

        fig.tight_layout()
        _base = savepath.rsplit(".", 1)[0]
        fig.savefig(f"{_base}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return dict(zip(all_comb_names_, units))

    lesion_units_norm = plot_lesion_unit_distribution(
        all_comb, all_comb_names_lesion, pre_n, post_n,
        VARIANT_NORM, cluster_info,
        f"{save_dir}/lesion_units_{aname}.png",
    )
    
    # ── Shared per-task test sets ──
    # One test set per task, generated once and reused by every experiment
    # below (single-cluster, pruning, combined and modulation lesions), so all
    # conditions of all experiments are scored on identical trials and the
    # baselines are directly comparable across experiments. The combined
    # lesion uses the first `combined_test_n_batch` trials of the same set.
    test_n_batch = 200
    combined_test_n_batch = 100
    test_sets = lesion_ops.make_test_sets(task_params_c, all_tasks, test_n_batch, device)

    def task_test_set(task, n_trials=None):
        """(input, output, mask) of one task's shared test set, optionally its first n_trials."""
        test_input, test_output, test_mask = test_sets[task]
        if n_trials is None or n_trials >= test_input.shape[0]:
            return test_input, test_output, test_mask
        return test_input[:n_trials], test_output[:n_trials], test_mask[:n_trials]

    print(f"Generated shared test sets: {len(test_sets)} tasks x {test_n_batch} trials "
          f"(combined lesion uses the first {combined_test_n_batch})")

    # L2 pruning for W
    K_lst = [0.0, 10.0, 50.0, 90.0, 95.0, 98.0, 99.0, 99.90]
    sparsity_lst = [k / 100.0 for k in K_lst]
    all_comb_names_prune = [f"prune_{k:.3f}%" for k in K_lst]

    # Precompute pruned W tensors once — mp_layer1.W doesn't change between tasks
    W_orig = model.mp_layer1.W.data.clone()
    pruned_Ws = []
    for sparsity in sparsity_lst:
        numel = W_orig.numel()
        k = int((1.0 - sparsity) * numel)
        W_p = W_orig.clone()
        if k <= 0:
            W_p.zero_()
        elif k < numel:
            w_abs = W_p.abs().view(-1)
            _, top_idx = torch.topk(w_abs, k, largest=True, sorted=False)
            mask = torch.zeros(numel, dtype=torch.bool, device=W_p.device)
            mask[top_idx] = True
            W_p.mul_(mask.view_as(W_p))
        pruned_Ws.append(W_p)

    def run_cluster_lesion(all_comb_, all_comb_names_, variant, label):
        """Leave-one-out cluster lesion + size-matched random lesion for all tasks.

        Random controls are cached per task by (side, lesion size): the random
        draw in lesion_prepost_inplace depends only on how many neurons are
        removed and on which side's weights are zeroed — never on the cluster
        identity — so same-(side, size) conditions have identically distributed
        controls and share one set of repeat_num draws.

        Returns (ihtask_accs, ihrandomtask_accs, ihrandomtask_accs_raw); the raw
        entry keeps every random repeat (n_tasks × n_conditions × repeat_num) —
        rows belonging to same-(side, size) conditions are identical copies of
        the shared draws. Downstream (lesion_plot.py) consumes only the
        per-condition control means.
        """
        ihtask_accs_, ihrandomtask_accs_, ihrandomtask_accs_raw_ = [], [], []

        def _lesion(comb, random):
            saved, _ = lesion_ops.lesion_prepost_inplace(
                model, cluster_info, comb[1], comb[0], variant, random=random)
            return saved

        def _control_key(comb):
            # The nolesion baseline (cluster None) is a no-op on both the
            # cluster and the control side, keyed as size 0.
            return (comb[0], lesion_ops.neuron_cluster_size(cluster_info, comb, variant))

        for task in all_tasks:
            print(f"[{label}] Evaluating task: {task}")
            ihaccs, ihrandomaccs, ihrandomaccs_raw = lesion_ops.evaluate_conditions(
                model, task_test_set(task), all_comb_, _lesion, _control_key, repeat_num)
            print(f"  [ctrl-cache/{label}] {task}: {len(set(map(_control_key, all_comb_)))} unique "
                  f"(side, size) controls for {len(all_comb_)} conditions")
            ihtask_accs_.append(ihaccs)
            ihrandomtask_accs_.append(ihrandomaccs)
            ihrandomtask_accs_raw_.append(ihrandomaccs_raw)
            # One gc pass per task is enough: tensors are freed promptly by
            # refcounting; gc only collects stray cycles, and calling it per
            # condition cost ~40 ms x thousands of calls.
            gc.collect()

        return ihtask_accs_, ihrandomtask_accs_, ihrandomtask_accs_raw_

    # ── Normalized cluster lesion (k=FIXED_K) ──
    ihtask_accs, ihrandomtask_accs, ihrandomtask_accs_raw = run_cluster_lesion(
        all_comb, all_comb_names_lesion, variant=VARIANT_NORM, label="norm"
    )

    # Pruning (normalized only) — runs in its own task loop
    wtask_accs = []
    for task in all_tasks:
        print(f"[pruning] Evaluating task: {task}")
        test_set = task_test_set(task)

        waccs = []
        for idx, W_pruned in enumerate(pruned_Ws):
            print(f"  Evaluating pruning condition: {all_comb_names_prune[idx]}")
            model.mp_layer1.W.data.copy_(W_pruned)
            waccs.append(lesion_ops.evaluate_accuracy(model, test_set))

        model.mp_layer1.W.data.copy_(W_orig)
        wtask_accs.append(waccs)
        gc.collect()  # once per task; see note in run_cluster_lesion

    helper.plot_heatmap(
        ihtask_accs, all_comb_names_lesion, all_tasks,
        xlabel="Lesion Condition", ylabel="Task", savename="lesion", aname=aname,
        save_dir=save_dir,
    )
    helper.plot_heatmap(
        wtask_accs, all_comb_names_prune, all_tasks,
        xlabel="Pruning Condition", ylabel="Task", savename="pruning", aname=aname,
        save_dir=save_dir,
    )
    helper.plot_heatmap(
        ihrandomtask_accs, all_comb_names_lesion, all_tasks,
        xlabel="Random Lesion Condition", ylabel="Task", savename="random_lesion", aname=aname,
        save_dir=save_dir,
    )

    # ── Unnormalized cluster lesion (k=FIXED_K) ──
    all_comb_unnorm, all_comb_names_lesion_unnorm = lesion_ops.neuron_conditions(
        pre_n_unnorm, post_n_unnorm)
    print(f"\n[unnormalized lesion] conditions: {all_comb_names_lesion_unnorm}")

    lesion_units_unnorm = plot_lesion_unit_distribution(
        all_comb_unnorm, all_comb_names_lesion_unnorm, pre_n_unnorm, post_n_unnorm,
        VARIANT_UNNORM, cluster_info,
        f"{save_dir}/lesion_units_unnorm_{aname}.png",
    )

    ihtask_accs_unnorm, ihrandomtask_accs_unnorm, ihrandomtask_accs_raw_unnorm = run_cluster_lesion(
        all_comb_unnorm, all_comb_names_lesion_unnorm, variant=VARIANT_UNNORM, label="unnorm"
    )

    helper.plot_heatmap(
        ihtask_accs_unnorm, all_comb_names_lesion_unnorm, all_tasks,
        xlabel="Lesion Condition (unnorm)", ylabel="Task",
        savename="lesion_unnorm", aname=aname, save_dir=save_dir,
    )
    helper.plot_heatmap(
        ihrandomtask_accs_unnorm, all_comb_names_lesion_unnorm, all_tasks,
        xlabel="Random Lesion Condition (unnorm)", ylabel="Task",
        savename="random_lesion_unnorm", aname=aname, save_dir=save_dir,
    )

    # ── Combined lesion: simultaneously lesion 1 input + 1 hidden cluster ──
    # Result shape per variant: (n_tasks, pre_n, post_n)
    # Fewer trials than the single-cluster sweep (the first combined_test_n_batch
    # of each task's shared test set): the pre_n × post_n grid multiplies
    # forward passes, and per-cell noise averages out over the grid.
    _combined_cache = {}
    for variant, pre_n_v, post_n_v in [
        (VARIANT_NORM,   pre_n,        post_n),
        (VARIANT_UNNORM, pre_n_unnorm, post_n_unnorm),
    ]:
        print(f"\n[combined lesion ({variant})] {pre_n_v} input × {post_n_v} hidden = {pre_n_v * post_n_v} combinations")

        combined_accs = np.zeros((len(all_tasks), pre_n_v, post_n_v))
        combined_random_accs = np.zeros((len(all_tasks), pre_n_v, post_n_v))
        combined_random_accs_raw = np.zeros((len(all_tasks), pre_n_v, post_n_v, repeat_num))
        combined_baseline = np.zeros(len(all_tasks))

        # Cluster sizes for the random-control cache below — the same size
        # expressions lesion_prepost_inplace uses to draw its random lesions.
        _pre_sizes = {i: lesion_ops.neuron_cluster_size(cluster_info, ("pre", i), variant)
                      for i in range(1, pre_n_v + 1)}
        _post_sizes = {i: lesion_ops.neuron_cluster_size(cluster_info, ("post", i), variant)
                       for i in range(1, post_n_v + 1)}

        for ti, task in enumerate(all_tasks):
            print(f"  [{variant}] task: {task}")
            test_set = task_test_set(task, combined_test_n_batch)

            # baseline (no lesion)
            combined_baseline[ti] = lesion_ops.evaluate_accuracy(model, test_set)

            # Per-task control cache: random draws depend only on the two
            # lesion sizes, so (pi, qi) cells with duplicate (pre size,
            # post size) share one set of repeat_num control draws.
            ctrl_cache = {}  # (pre size, post size) -> [repeat_num accs]

            for pi in range(1, pre_n_v + 1):
                for qi in range(1, post_n_v + 1):
                    # cluster lesion
                    saved_pre, _ = lesion_ops.lesion_prepost_inplace(
                        model, cluster_info, pi, "pre", variant)
                    saved_post, _ = lesion_ops.lesion_prepost_inplace(
                        model, cluster_info, qi, "post", variant)
                    combined_accs[ti, pi - 1, qi - 1] = lesion_ops.evaluate_accuracy(model, test_set)
                    lesion_ops.restore_lesion(model, saved_post)
                    lesion_ops.restore_lesion(model, saved_pre)

                    # random lesion (same sizes, repeated), cached by size pair
                    ctrl_key = (_pre_sizes[pi], _post_sizes[qi])
                    if ctrl_key not in ctrl_cache:
                        rset = []
                        for _ in range(repeat_num):
                            saved_pre_r, _ = lesion_ops.lesion_prepost_inplace(
                                model, cluster_info, pi, "pre", variant, random=True)
                            saved_post_r, _ = lesion_ops.lesion_prepost_inplace(
                                model, cluster_info, qi, "post", variant, random=True)
                            rset.append(lesion_ops.evaluate_accuracy(model, test_set))
                            lesion_ops.restore_lesion(model, saved_post_r)
                            lesion_ops.restore_lesion(model, saved_pre_r)
                        ctrl_cache[ctrl_key] = rset
                    combined_random_accs[ti, pi - 1, qi - 1] = np.mean(ctrl_cache[ctrl_key])
                    combined_random_accs_raw[ti, pi - 1, qi - 1] = ctrl_cache[ctrl_key]

            print(f"    [ctrl-cache] {task}: {len(ctrl_cache)} unique size-pairs "
                  f"for {pre_n_v * post_n_v} combinations")
            gc.collect()

        # Flatten to 2D for heatmap: columns = (pre_1,post_1), (pre_1,post_2), ...
        flat_names = [f"i{pi}_h{qi}" for pi in range(1, pre_n_v + 1) for qi in range(1, post_n_v + 1)]
        combined_accs_flat = combined_accs.reshape(len(all_tasks), -1)
        combined_random_flat = combined_random_accs.reshape(len(all_tasks), -1)

        vtag = "norm" if "unnormalized" not in variant else "unnorm"
        if pre_n_v * post_n_v <= 100:
            helper.plot_heatmap(
                combined_accs_flat, flat_names, all_tasks,
                xlabel=f"Combined Lesion (input, hidden) [{vtag}]", ylabel="Task",
                savename=f"combined_lesion_{vtag}", aname=aname, save_dir=save_dir,
            )
            helper.plot_heatmap(
                combined_random_flat, flat_names, all_tasks,
                xlabel=f"Random Combined Lesion (input, hidden) [{vtag}]", ylabel="Task",
                savename=f"combined_random_lesion_{vtag}", aname=aname, save_dir=save_dir,
            )
        else:
            print(f"  [{vtag}] Skipping combined heatmap: {pre_n_v} × {post_n_v} = {pre_n_v * post_n_v} > 100")

        saved_dict_key = f"combined_lesion_{vtag}"
        # store temporarily; will be added to saved_dict later
        _unres_labels_v, _ = lesion_ops.unresponsive_neuron_conditions(cluster_info, variant)
        _combined_cache[saved_dict_key] = {
            "combined_accs": combined_accs,
            "combined_random_accs": combined_random_accs,
            "combined_random_accs_raw": combined_random_accs_raw,
            "combined_baseline": combined_baseline,
            "all_tasks": all_tasks,
            "pre_n": pre_n_v,
            "post_n": post_n_v,
            "variant": variant,
            "test_n_batch": combined_test_n_batch,
            "test_set": f"first {combined_test_n_batch} trials of the shared per-task test set",
            # 1-based cluster labels of the unresponsive input / hidden class
            # (grid index = label - 1), or None
            "unresponsive_pre_label": _unres_labels_v["pre"],
            "unresponsive_post_label": _unres_labels_v["post"],
        }

    # Modulation lesion — loop over all clustering types and both lesion modes.
    # "zero_W": zero the static weight W at cluster synapses (original method).
    # "freeze_M": keep W intact but freeze plasticity (M stays at M_init) at cluster synapses.
    mod_lesion_results = {}

    for mod_type_key, mod_G_idx_cur in mod_type_lst:
        _G_name = cluster_info_mod[mod_type_key]["result_all_name_lst"][mod_G_idx_cur]
        print(f"\n[modulation lesion] type={mod_type_key}  {_G_name}")

        # Fixed-k labels if saved; otherwise the smallest saved k, else the
        # optimal k (lesion_ops.fixed_k_synapse_clusters).
        mod_col_clusters_cur, mod_unres_label_cur, mod_k_used = lesion_ops.fixed_k_synapse_clusters(
            cluster_info_mod[mod_type_key], mod_G_idx_cur, FIXED_K)
        if mod_k_used == FIXED_K:
            print(f"  Using fixed k={FIXED_K} for modulation clustering")
        else:
            print(f"  Fixed k={FIXED_K} not saved for this type; using k={mod_k_used}")
        print(f"  {len(mod_col_clusters_cur)} modulation clusters "
              f"(unresponsive label: {mod_unres_label_cur})")

        all_comb_mod, all_comb_names_mod = lesion_ops.synapse_conditions(mod_col_clusters_cur)

        # Plot cluster sizes for this modulation clustering type
        mod_cluster_ids_sorted = sorted(mod_col_clusters_cur.keys())
        mod_cluster_sizes = [len(mod_col_clusters_cur[ci]) for ci in mod_cluster_ids_sorted]
        mod_cluster_labels = [f"c{ci}" for ci in mod_cluster_ids_sorted]
        type_tag_size = mod_type_key.replace("modulation_all_", "").replace("_", "-")

        n_mod = len(mod_cluster_ids_sorted)
        fig_sz, ax_sz = plt.subplots(1, 1, figsize=(max(3.5, 0.35 * n_mod + 1), 2.5), dpi=300)
        ax_sz.bar(mod_cluster_labels, mod_cluster_sizes,
                  color="steelblue", edgecolor="black", linewidth=0.3)
        ax_sz.set_xticks(range(n_mod))
        ax_sz.set_xticklabels(mod_cluster_labels, rotation=45, ha="right", fontsize=7)
        ax_sz.set_ylabel("# Synapses", fontsize=8)
        ax_sz.set_xlabel(f"Modulation Cluster ({type_tag_size})", fontsize=8)
        ax_sz.tick_params(axis="both", length=1.5, pad=2, width=0.5)
        ax_sz.spines["top"].set_visible(False)
        ax_sz.spines["right"].set_visible(False)
        ax_sz.spines["left"].set_linewidth(0.5)
        ax_sz.spines["bottom"].set_linewidth(0.5)
        fig_sz.tight_layout()
        fig_sz.savefig(f"{save_dir}/mod_lesion_units_{type_tag_size}_{aname}.png", dpi=300)
        plt.close(fig_sz)
        print(f"  Saved modulation cluster sizes for {mod_type_key}: {dict(zip(mod_cluster_labels, mod_cluster_sizes))}")

        for mod_lesion_mode in lesion_ops.SYNAPSE_LESION_MODES:
            print(f"  [mode={mod_lesion_mode}]")

            def _lesion_mod(comb, random, _mode=mod_lesion_mode, _clusters=mod_col_clusters_cur):
                saved, _ = lesion_ops.lesion_modulation_inplace(
                    model, comb[1], _clusters, n_pre_mod, mode=_mode, random=random)
                return saved

            def _control_key_mod(comb, _clusters=mod_col_clusters_cur):
                # mod_nolesion (ci None) is a no-op on both sides: size 0. The
                # random modulation lesion draws n synapses uniformly from all
                # MM positions regardless of cluster identity, so same-size
                # clusters share one control set.
                return 0 if comb[1] is None else len(_clusters[comb[1]])

            modtask_accs = []
            modrandomtask_accs = []
            modrandomtask_accs_raw = []

            for task in all_tasks:
                print(f"    Evaluating task: {task}")
                modaccs, modrandomaccs, modrandomaccs_raw = lesion_ops.evaluate_conditions(
                    model, task_test_set(task), all_comb_mod, _lesion_mod, _control_key_mod,
                    repeat_num)
                print(f"      [ctrl-cache] {task}: {len(set(map(_control_key_mod, all_comb_mod)))} "
                      f"unique sizes for {len(all_comb_mod)} conditions")
                modtask_accs.append(modaccs)
                modrandomtask_accs.append(modrandomaccs)
                modrandomtask_accs_raw.append(modrandomaccs_raw)
                gc.collect()  # once per task; see note in run_cluster_lesion

            type_tag = mod_type_key.replace("modulation_all_", "").replace("_", "-")
            mode_tag = mod_lesion_mode.replace("_", "-")
            helper.plot_heatmap(
                modtask_accs, all_comb_names_mod, all_tasks,
                xlabel=f"Modulation Lesion Condition ({mod_lesion_mode})", ylabel="Task",
                savename=f"mod_lesion_{type_tag}_{mode_tag}", aname=aname,
                save_dir=save_dir,
            )
            helper.plot_heatmap(
                modrandomtask_accs, all_comb_names_mod, all_tasks,
                xlabel=f"Random Modulation Lesion Condition ({mod_lesion_mode})", ylabel="Task",
                savename=f"mod_random_lesion_{type_tag}_{mode_tag}", aname=aname,
                save_dir=save_dir,
            )

            result_key = f"{mod_type_key}__{mod_lesion_mode}"
            mod_lesion_results[result_key] = {
                "modtask_accs": modtask_accs,
                "modrandomtask_accs": modrandomtask_accs,
                "modrandomtask_accs_raw": modrandomtask_accs_raw,
                "all_comb_names_mod": all_comb_names_mod,
                "all_tasks": all_tasks,
                "mod_G_idx": mod_G_idx_cur,
                "mod_col_clusters": mod_col_clusters_cur,
                "mod_lesion_mode": mod_lesion_mode,
                # key of the unresponsive synapse class in mod_col_clusters
                # (condition f"mod_c{label}"), or None
                "unresponsive_label": mod_unres_label_cur,
                "mod_k": int(mod_k_used),
            }

    # Ordering note: corr/l1_dist row/col i → cluster i+1.
    # ihtask_accs columns: [0]=pre_nolesion, [1..pre_n]=lesion pre cluster 1..pre_n,
    #                       [pre_n+1]=post_nolesion, [pre_n+2..end]=lesion post cluster 1..post_n.
    # So ihtask_accs[:, 1:pre_n+1] aligns with corr_matrices["input"] and l1_dist_matrices["input"].
    _unres_labels_norm, _unres_conditions_norm = lesion_ops.unresponsive_neuron_conditions(
        cluster_info, VARIANT_NORM)
    _unres_labels_unnorm, _unres_conditions_unnorm = lesion_ops.unresponsive_neuron_conditions(
        cluster_info, VARIANT_UNNORM)
    saved_dict = {
        "lesion": {
            "ihtask_accs": ihtask_accs,
            "all_comb_names_lesion": all_comb_names_lesion,
            "all_tasks": all_tasks,
            "lesion_units": lesion_units_norm,
            # {"pre": label|None, "post": label|None} and the matching
            # condition names (e.g. "post_c21"); the unresponsive class is
            # identified by these, never by position.
            "unresponsive_labels": _unres_labels_norm,
            "unresponsive_conditions": _unres_conditions_norm,
        },
        "prune": {
            "wtask_accs": wtask_accs,
            "all_comb_names_prune": all_comb_names_prune,
            "all_tasks": all_tasks,
        },
        "random_lesion": {
            "ihrandomtask_accs": ihrandomtask_accs,
            "ihrandomtask_accs_raw": ihrandomtask_accs_raw,
            "all_comb_names_lesion": all_comb_names_lesion,
            "all_tasks": all_tasks,
        },
        "lesion_unnorm": {
            "ihtask_accs": ihtask_accs_unnorm,
            "all_comb_names_lesion": all_comb_names_lesion_unnorm,
            "all_tasks": all_tasks,
            "lesion_units": lesion_units_unnorm,
            "unresponsive_labels": _unres_labels_unnorm,
            "unresponsive_conditions": _unres_conditions_unnorm,
        },
        "random_lesion_unnorm": {
            "ihrandomtask_accs": ihrandomtask_accs_unnorm,
            "ihrandomtask_accs_raw": ihrandomtask_accs_raw_unnorm,
            "all_comb_names_lesion": all_comb_names_lesion_unnorm,
            "all_tasks": all_tasks,
        },
        "combined_lesion_norm": _combined_cache.get("combined_lesion_norm", {}),
        "combined_lesion_unnorm": _combined_cache.get("combined_lesion_unnorm", {}),
        "mod_lesion": mod_lesion_results,
        "cluster_similarity": {
            "corr_matrices": corr_matrices,
            "l1_dist_matrices": l1_dist_matrices,
            "cluster_means": cluster_means_cache,
        },
        "fixed_k": FIXED_K,
        "repeat_num": repeat_num,
        # Every experiment above scored the same per-task trials (the combined
        # lesion the first combined_test_n_batch of them).
        "test_n_batch": test_n_batch,
        "shared_test_set": True,
    }

    with open(f"{save_dir}/lesion_prune_results_{aname}.pkl", "wb") as f:
        pickle.dump(saved_dict, f)
    print(f"Saved lesion/pruning results to {save_dir}/lesion_prune_results_{aname}.pkl")
