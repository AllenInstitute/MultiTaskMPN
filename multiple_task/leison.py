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
3. Modulation lesion — for each synapse cluster in M, either zero the static
   weight W at those synapses (remove connectivity) or freeze M at those
   synapses (remove plasticity), to dissect the contributions of static vs.
   plastic pathways.
4. Magnitude pruning — zero the lowest-magnitude fraction of W at increasing
   sparsity levels (0–99.9%) to assess weight redundancy.

Outputs:
  - Heatmaps of per-task accuracy under each lesion condition
  - lesion_prune_results_{aname}.pkl — full results dict for downstream
    analysis in leison_plot.py

No CLI entry point: `main(seed, feature)` is invoked by run_pipeline.py.
"""
from pathlib import Path
import json
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
import mpn
import mpn_tasks
import helper
import clustering

def main(seed, feature):
    """Run the lesion & pruning experiments for one trained network."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    repeat_num = 10

    aname = f"everything_seed{seed}_{feature}+hidden300+batch128+angle"

    save_dir = f"./multiple_tasks_perf/{aname}"
    os.makedirs(save_dir, exist_ok=True)
    for _old in Path(save_dir).iterdir():
        if _old.is_file():
            _old.unlink()

    out_param_path = Path(f"multiple_tasks/param_{aname}_param.json")
    cluster_path = Path(f"./multiple_tasks_analysis/{aname}/cluster_info_{aname}.pkl")
    cluster_path_mod = Path(f"./multiple_tasks_analysis/{aname}/cluster_info_mod_{aname}.pkl")

    with out_param_path.open() as f:
        raw_cfg_param = json.load(f)

    with cluster_path.open("rb") as f:
        cluster_info = pickle.load(f)

    with cluster_path_mod.open("rb") as f:
        cluster_info_mod = pickle.load(f)
        
    print(cluster_info.keys())
        
    # need to remind what the clusters are
    cnames = ["input_normalized", "hidden_normalized"]
    for cname in cnames:
        print(f"{cname} clusters; row_clusters: {len(cluster_info[cname]['row_clusters'])}, col_clusters: {len(cluster_info[cname]['col_clusters'])}")
        
    task_params = raw_cfg_param["task_params"]
    train_params = raw_cfg_param["train_params"]
    net_params = raw_cfg_param["net_params"]
    
    netpathname = "multiple_tasks/" + f"savednet_{aname}.pt"
    checkpoint = torch.load(netpathname, map_location=device)

    state_dict = checkpoint["state_dict"]
    print(state_dict.keys())
    load_net_params = checkpoint["net_params"]
    print(load_net_params)
    
    model = mpn.DeepMultiPlasticNet(load_net_params, verbose=False, forzihan=True)

    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
    print("missing:", missing)
    print("unexpected:", unexpected)

    model.eval()
    model.to(device)

    task_params_c, _, _ = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )
    
    all_tasks = task_params_c['rules']
    
    # ── Fixed-k clustering for lesion experiments ──
    # Cut the saved dendrograms at FIXED_K instead of using the optimal k.
    # For input/hidden: shared re-cut helper clustering.fixed_k_col_clusters.
    # For modulation: use result_all["col_labels_by_k"][FIXED_K] (already saved).
    # FIXED_K is inferred from the upstream pickle (global_assignment_fixed_k{N} key).
    import re as _re

    _first_mod_type = next(iter(cluster_info_mod))
    _fk_keys = [k for k in cluster_info_mod[_first_mod_type] if k.startswith("global_assignment_fixed_k")]
    if _fk_keys:
        _fk_match = _re.search(r"fixed_k(\d+)", _fk_keys[0])
        FIXED_K = int(_fk_match.group(1))
        print(f"Inferred FIXED_K={FIXED_K} from upstream pickle key: {_fk_keys[0]}")
    else:
        # Fallback only for pickles predating the fixed-k key; must match
        # FIXED_K_OM in multiple_task_analysis.py.
        FIXED_K = 20
        print(f"No global_assignment_fixed_k* found in cluster_info_mod; defaulting FIXED_K={FIXED_K}")

    # Derive fixed-k clusters for input/hidden and inject into cluster_info
    # so that leison_prepost_inplace can look them up via the variant string.
    _base_names = ["input_normalized", "hidden_normalized",
                   "input_unnormalized", "hidden_unnormalized"]
    for _base in _base_names:
        _fk_key = f"{_base}_k{FIXED_K}"
        _fk_clusters = clustering.fixed_k_col_clusters(cluster_info[_base], FIXED_K)
        cluster_info[_fk_key] = {
            "col_clusters": _fk_clusters,
            "row_clusters": cluster_info[_base]["row_clusters"],
            "tb_break_name": cluster_info[_base]["tb_break_name"],
            "cell_vars_rules_sorted_norm": cluster_info[_base]["cell_vars_rules_sorted_norm"],
            "result": cluster_info[_base]["result"],
        }
        print(f"Derived {_fk_key}: {len(_fk_clusters)} clusters")

    _input_norm_key    = f"input_normalized_k{FIXED_K}"
    _hidden_norm_key   = f"hidden_normalized_k{FIXED_K}"
    _input_unnorm_key  = f"input_unnormalized_k{FIXED_K}"
    _hidden_unnorm_key = f"hidden_unnormalized_k{FIXED_K}"

    pre_n = len(cluster_info[_input_norm_key]["col_clusters"])
    post_n = len(cluster_info[_hidden_norm_key]["col_clusters"])
    pre_n_unnorm = len(cluster_info[_input_unnorm_key]["col_clusters"])
    post_n_unnorm = len(cluster_info[_hidden_unnorm_key]["col_clusters"])
    print(f"Fixed k={FIXED_K}: norm pre={pre_n}, post={post_n}; unnorm pre={pre_n_unnorm}, post={post_n_unnorm}")

    VARIANT_NORM = f"normalized_k{FIXED_K}"
    VARIANT_UNNORM = f"unnormalized_k{FIXED_K}"

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
            # range (matches the L1 panels in leison_plot.py).
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

    all_comb = (
        [("pre", None)] + [("pre", i) for i in range(1, pre_n + 1)] +
        [("post", None)] + [("post", i) for i in range(1, post_n + 1)]
    )
    rename = {"pre_cNone": "pre_noleison", "post_cNone": "post_noleison"}
    all_comb_names_leison = [rename.get(f"{tag}_c{i}", f"{tag}_c{i}") for tag, i in all_comb]
    print(f"all_comb_names_leison: {all_comb_names_leison}")

    print(model.W_initial_linear.weight.shape, model.mp_layer1.W.shape, model.mp_layer1.b.shape, model.W_output.shape)

    # Precompute max_N for each cluster name (used for random lesion sampling)
    max_N_cache = {
        name: max(arr.max() for arr in cluster_info[name]["col_clusters"].values())
        for name in [_input_norm_key, _hidden_norm_key, _input_unnorm_key, _hidden_unnorm_key]
    }

    # All modulation clustering types to run lesion experiments on; the second
    # tuple element indexes result_all_lst in the upstream cluster_info_mod
    # pickle — the per-G clustering solutions saved by multiple_task_analysis.py
    # in its G_lst order (index 1 = G=300 for every type currently). The actual
    # G is read back from the pickle's result_all_name_lst, never hard-coded here.
    mod_type_lst_all = [
        ("modulation_all_normalized",                1),
        ("modulation_all_unnormalized",              1),
        ("modulation_all_weighted_unnormalized",     1),
        ("modulation_all_var_weighted_unnormalized", 1),
    ]
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

    def leison_prepost_inplace(net, cluster_index, preorpost, random=False, variant="normalized"):
        """Apply lesion in-place on net; returns (saved_state, leison_units).
        Call restore_leison(net, saved_state) to undo.  No deepcopy."""
        name = f"input_{variant}" if preorpost == "pre" else f"hidden_{variant}"
        leison_units = 0
        saved = {}

        if cluster_index is not None:
            neuron_index = cluster_info[name]["col_clusters"][cluster_index]
            class_N = len(neuron_index)
            # Same count whether the target is the cluster itself or the
            # size-matched random control.
            leison_units = class_N

            if not random:
                neuron_index = torch.tensor(neuron_index, dtype=torch.long, device=net.W_output.device)
            else:
                vals = np.random.choice(np.arange(max_N_cache[name] + 1), size=class_N, replace=False)
                neuron_index = torch.tensor(vals, dtype=torch.long, device=net.W_output.device)

            with torch.no_grad():
                if preorpost == "pre":
                    saved = {
                        "preorpost": preorpost,
                        "neuron_index": neuron_index,
                        "W_initial": net.W_initial_linear.weight.data[neuron_index, :].clone(),
                        "mp_W_col": net.mp_layer1.W[:, neuron_index].clone(),
                    }
                    net.W_initial_linear.weight.data[neuron_index, :] = 0.0
                    net.mp_layer1.W[:, neuron_index] = 0.0
                elif preorpost == "post":
                    saved = {
                        "preorpost": preorpost,
                        "neuron_index": neuron_index,
                        "mp_W_row": net.mp_layer1.W[neuron_index, :].clone(),
                        "mp_b": net.mp_layer1.b[neuron_index].clone(),
                        "W_output": net.W_output[:, neuron_index].clone(),
                    }
                    net.mp_layer1.W[neuron_index, :] = 0.0
                    # the bias of MPN layer is on the postsynaptic side
                    net.mp_layer1.b[neuron_index] = 0.0
                    net.W_output[:, neuron_index] = 0.0

        return saved, leison_units

    def leison_modulation_inplace(net, cluster_index, mod_col_clusters_, MM_, n_pre_, random=False):
        """Lesion a modulation cluster by zeroing mp_layer1.W[post, pre] entries.
        W is (post, pre); C-order flat_idx k → post = k // n_pre_, pre = k % n_pre_
        (valid for square and non-square layers alike).
        Returns (saved_state, n_lesioned)."""
        if cluster_index is None:
            return {}, 0
        flat_idxs = np.array(mod_col_clusters_[cluster_index], dtype=int)
        n = len(flat_idxs)
        if random:
            flat_idxs = np.random.choice(MM_, size=n, replace=False)
        post_t = torch.tensor(flat_idxs // n_pre_, dtype=torch.long, device=net.mp_layer1.W.device)
        pre_t = torch.tensor(flat_idxs % n_pre_, dtype=torch.long, device=net.mp_layer1.W.device)
        with torch.no_grad():
            saved = {
                "preorpost": "modulation",
                "pre_t": pre_t,
                "post_t": post_t,
                "W_vals": net.mp_layer1.W[post_t, pre_t].clone(),
            }
            net.mp_layer1.W[post_t, pre_t] = 0.0
        return saved, n

    def leison_modulation_freeze_inplace(net, cluster_index, mod_col_clusters_, MM_, n_pre_, random=False):
        """Freeze plasticity at a modulation cluster: M stays at its initial value
        at those (post, pre) positions throughout the trial, but W is untouched.
        W is (post, pre); C-order flat_idx k → post = k // n_pre_, pre = k % n_pre_
        (valid for square and non-square layers alike).
        Returns (saved_state, n_frozen)."""
        if cluster_index is None:
            return {}, 0
        flat_idxs = np.array(mod_col_clusters_[cluster_index], dtype=int)
        n = len(flat_idxs)
        if random:
            flat_idxs = np.random.choice(MM_, size=n, replace=False)
        post_t = torch.tensor(flat_idxs // n_pre_, dtype=torch.long, device=net.mp_layer1.W.device)
        pre_t = torch.tensor(flat_idxs % n_pre_, dtype=torch.long, device=net.mp_layer1.W.device)
        net.mp_layer1.set_plasticity_freeze(post_t, pre_t)
        saved = {
            "preorpost": "modulation_freeze",
            "pre_t": pre_t,
            "post_t": post_t,
        }
        return saved, n

    def restore_leison(net, saved):
        """Restore weights modified by leison_prepost_inplace, leison_modulation_inplace,
        or leison_modulation_freeze_inplace."""
        if not saved:
            return
        with torch.no_grad():
            if saved["preorpost"] == "pre":
                neuron_index = saved["neuron_index"]
                net.W_initial_linear.weight.data[neuron_index, :] = saved["W_initial"]
                net.mp_layer1.W[:, neuron_index] = saved["mp_W_col"]
            elif saved["preorpost"] == "post":
                neuron_index = saved["neuron_index"]
                net.mp_layer1.W[neuron_index, :] = saved["mp_W_row"]
                net.mp_layer1.b[neuron_index] = saved["mp_b"]
                net.W_output[:, neuron_index] = saved["W_output"]
            elif saved["preorpost"] == "modulation":
                net.mp_layer1.W[saved["post_t"], saved["pre_t"]] = saved["W_vals"]
            elif saved["preorpost"] == "modulation_freeze":
                net.mp_layer1.clear_plasticity_freeze()

    def plot_lesion_unit_distribution(all_comb_, all_comb_names_, pre_n_, post_n_,
                                      variant_label, cluster_info_, savepath):
        """Bar chart showing how many neurons each lesion condition removes.
        Returns {condition_name: n_units} for the results pickle."""
        units = []
        for tag, ci in all_comb_:
            if ci is None:
                units.append(0)
            else:
                name = f"input_{variant_label}" if tag == "pre" else f"hidden_{variant_label}"
                units.append(len(cluster_info_[name]["col_clusters"][ci]))
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
        all_comb, all_comb_names_leison, pre_n, post_n,
        VARIANT_NORM, cluster_info,
        f"{save_dir}/lesion_units_{aname}.png",
    )
    
    # setup the evaluation dataset generator
    test_n_batch = 200
    task_params_c['hp']['batch_size_train'] = test_n_batch
    
    # L2 pruning for W
    K_lst = [0.0, 10.0, 50.0, 90.0, 95.0, 98.0, 99.0, 99.90]
    sparsity_lst = [k / 100.0 for k in K_lst]
    all_comb_prune = [("prune", k) for k in sparsity_lst]
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
        draw in leison_prepost_inplace depends only on how many neurons are
        removed and on which side's weights are zeroed — never on the cluster
        identity — so same-(side, size) conditions have identically distributed
        controls and share one set of repeat_num draws.

        Returns (ihtask_accs, ihrandomtask_accs, ihrandomtask_accs_raw); the raw
        entry keeps every random repeat (n_tasks × n_conditions × repeat_num) —
        rows belonging to same-(side, size) conditions are identical copies of
        the shared draws. Downstream (leison_plot.py) consumes only the
        per-condition control means.
        """
        ihtask_accs_, ihrandomtask_accs_, ihrandomtask_accs_raw_ = [], [], []

        for task in all_tasks:
            print(f"[{label}] Evaluating task: {task}")
            test_data, _ = mpn_tasks.generate_trials_wrap(
                task_params_c, test_n_batch, rules=[task],
                mode_input="random_batch", device=device, verbose=False
            )
            test_input, test_output, test_mask = test_data

            ihaccs, ihrandomaccs, ihrandomaccs_raw = [], [], []
            # Per-task control cache (each task has its own test set); see the
            # docstring for why sharing across same-(side, size) keys is exact.
            ctrl_cache = {}  # (preorpost, lesion size) -> [repeat_num accs]

            for idx, comb in enumerate(all_comb_):
                saved, _ = leison_prepost_inplace(
                    model, cluster_index=comb[1], preorpost=comb[0], variant=variant
                )
                with torch.inference_mode():
                    net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='minimal')
                    acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                               isvalid=True, mode=model.acc_measure)
                ihaccs.append(acc.item())
                restore_leison(model, saved)
                del net_out

                # Same size expression leison_prepost_inplace uses for its
                # random draw; the noleison baseline (cluster None) is a no-op
                # on both the cluster and the control side, keyed as size 0.
                if comb[1] is None:
                    ctrl_key = (comb[0], 0)
                else:
                    _cname = f"input_{variant}" if comb[0] == "pre" else f"hidden_{variant}"
                    ctrl_key = (comb[0], len(cluster_info[_cname]["col_clusters"][comb[1]]))

                if ctrl_key not in ctrl_cache:
                    rset = []
                    for _ in range(repeat_num):
                        saved_r, _ = leison_prepost_inplace(
                            model, cluster_index=comb[1], preorpost=comb[0],
                            random=True, variant=variant
                        )
                        with torch.inference_mode():
                            net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='minimal')
                            acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                                       isvalid=True, mode=model.acc_measure)
                        rset.append(acc.item())
                        restore_leison(model, saved_r)
                        del net_out
                    ctrl_cache[ctrl_key] = rset
                ihrandomaccs.append(np.mean(ctrl_cache[ctrl_key]))
                ihrandomaccs_raw.append(list(ctrl_cache[ctrl_key]))

            print(f"  [ctrl-cache/{label}] {task}: {len(ctrl_cache)} unique "
                  f"(side, size) controls for {len(all_comb_)} conditions")
            ihtask_accs_.append(ihaccs)
            ihrandomtask_accs_.append(ihrandomaccs)
            ihrandomtask_accs_raw_.append(ihrandomaccs_raw)
            # One gc pass per task is enough: tensors are freed promptly by
            # refcounting (del net_out above); gc only collects stray cycles,
            # and calling it per condition cost ~40 ms x thousands of calls.
            gc.collect()

        return ihtask_accs_, ihrandomtask_accs_, ihrandomtask_accs_raw_

    # ── Normalized cluster lesion (k=FIXED_K) ──
    ihtask_accs, ihrandomtask_accs, ihrandomtask_accs_raw = run_cluster_lesion(
        all_comb, all_comb_names_leison, variant=VARIANT_NORM, label="norm"
    )

    # Pruning (normalized only) — runs in its own task loop
    wtask_accs = []
    for task in all_tasks:
        print(f"[pruning] Evaluating task: {task}")
        test_data, _ = mpn_tasks.generate_trials_wrap(
            task_params_c, test_n_batch, rules=[task],
            mode_input="random_batch", device=device, verbose=False
        )
        test_input, test_output, test_mask = test_data

        waccs = []
        for idx, W_pruned in enumerate(pruned_Ws):
            print(f"  Evaluating pruning condition: {all_comb_names_prune[idx]}")
            model.mp_layer1.W.data.copy_(W_pruned)

            with torch.inference_mode():
                net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='minimal')
                acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                           isvalid=True, mode=model.acc_measure)
            waccs.append(acc.item())
            del net_out

        model.mp_layer1.W.data.copy_(W_orig)
        wtask_accs.append(waccs)
        gc.collect()  # once per task; see note in run_cluster_lesion

    helper.plot_heatmap(
        ihtask_accs, all_comb_names_leison, all_tasks,
        xlabel="Lesion Condition", ylabel="Task", savename="lesion", aname=aname,
        save_dir=save_dir,
    )
    helper.plot_heatmap(
        wtask_accs, all_comb_names_prune, all_tasks,
        xlabel="Pruning Condition", ylabel="Task", savename="pruning", aname=aname,
        save_dir=save_dir,
    )
    helper.plot_heatmap(
        ihrandomtask_accs, all_comb_names_leison, all_tasks,
        xlabel="Random Lesion Condition", ylabel="Task", savename="random_lesion", aname=aname,
        save_dir=save_dir,
    )

    # ── Unnormalized cluster lesion (k=FIXED_K) ──
    all_comb_unnorm = (
        [("pre", None)] + [("pre", i) for i in range(1, pre_n_unnorm + 1)] +
        [("post", None)] + [("post", i) for i in range(1, post_n_unnorm + 1)]
    )
    rename_unnorm = {"pre_cNone": "pre_noleison", "post_cNone": "post_noleison"}
    all_comb_names_leison_unnorm = [
        rename_unnorm.get(f"{tag}_c{i}", f"{tag}_c{i}") for tag, i in all_comb_unnorm
    ]
    print(f"\n[unnormalized lesion] conditions: {all_comb_names_leison_unnorm}")

    lesion_units_unnorm = plot_lesion_unit_distribution(
        all_comb_unnorm, all_comb_names_leison_unnorm, pre_n_unnorm, post_n_unnorm,
        VARIANT_UNNORM, cluster_info,
        f"{save_dir}/lesion_units_unnorm_{aname}.png",
    )

    ihtask_accs_unnorm, ihrandomtask_accs_unnorm, ihrandomtask_accs_raw_unnorm = run_cluster_lesion(
        all_comb_unnorm, all_comb_names_leison_unnorm, variant=VARIANT_UNNORM, label="unnorm"
    )

    helper.plot_heatmap(
        ihtask_accs_unnorm, all_comb_names_leison_unnorm, all_tasks,
        xlabel="Lesion Condition (unnorm)", ylabel="Task",
        savename="lesion_unnorm", aname=aname, save_dir=save_dir,
    )
    helper.plot_heatmap(
        ihrandomtask_accs_unnorm, all_comb_names_leison_unnorm, all_tasks,
        xlabel="Random Lesion Condition (unnorm)", ylabel="Task",
        savename="random_lesion_unnorm", aname=aname, save_dir=save_dir,
    )

    # ── Combined lesion: simultaneously lesion 1 input + 1 hidden cluster ──
    # Result shape per variant: (n_tasks, pre_n, post_n)
    # Smaller batch than the single-cluster sweep: the pre_n × post_n grid
    # multiplies forward passes, and per-cell noise averages out over the grid.
    combined_test_n_batch = 100
    task_params_c['hp']['batch_size_train'] = combined_test_n_batch
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
        # expressions leison_prepost_inplace uses to draw its random lesions.
        _pre_sizes = {i: len(cluster_info[f"input_{variant}"]["col_clusters"][i])
                      for i in range(1, pre_n_v + 1)}
        _post_sizes = {i: len(cluster_info[f"hidden_{variant}"]["col_clusters"][i])
                       for i in range(1, post_n_v + 1)}

        for ti, task in enumerate(all_tasks):
            print(f"  [{variant}] task: {task}")
            test_data, _ = mpn_tasks.generate_trials_wrap(
                task_params_c, combined_test_n_batch, rules=[task],
                mode_input="random_batch", device=device, verbose=False
            )
            test_input, test_output, test_mask = test_data

            # baseline (no lesion)
            with torch.inference_mode():
                net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='minimal')
                acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                           isvalid=True, mode=model.acc_measure)
            combined_baseline[ti] = acc.item()
            del net_out

            # Per-task control cache: random draws depend only on the two
            # lesion sizes, so (pi, qi) cells with duplicate (pre size,
            # post size) share one set of repeat_num control draws.
            ctrl_cache = {}  # (pre size, post size) -> [repeat_num accs]

            for pi in range(1, pre_n_v + 1):
                for qi in range(1, post_n_v + 1):
                    # cluster lesion
                    saved_pre, _ = leison_prepost_inplace(
                        model, cluster_index=pi, preorpost="pre", variant=variant
                    )
                    saved_post, _ = leison_prepost_inplace(
                        model, cluster_index=qi, preorpost="post", variant=variant
                    )
                    with torch.inference_mode():
                        net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='minimal')
                        acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                                   isvalid=True, mode=model.acc_measure)
                    combined_accs[ti, pi - 1, qi - 1] = acc.item()
                    restore_leison(model, saved_post)
                    restore_leison(model, saved_pre)
                    del net_out

                    # random lesion (same sizes, repeated), cached by size pair
                    ctrl_key = (_pre_sizes[pi], _post_sizes[qi])
                    if ctrl_key not in ctrl_cache:
                        rset = []
                        for _ in range(repeat_num):
                            saved_pre_r, _ = leison_prepost_inplace(
                                model, cluster_index=pi, preorpost="pre",
                                random=True, variant=variant
                            )
                            saved_post_r, _ = leison_prepost_inplace(
                                model, cluster_index=qi, preorpost="post",
                                random=True, variant=variant
                            )
                            with torch.inference_mode():
                                net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='minimal')
                                acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                                           isvalid=True, mode=model.acc_measure)
                            rset.append(acc.item())
                            restore_leison(model, saved_post_r)
                            restore_leison(model, saved_pre_r)
                            del net_out
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

        saved_dict_key = f"combined_leison_{vtag}"
        # store temporarily; will be added to saved_dict later
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
        }

    # Restore the full evaluation batch for the modulation lesion below.
    task_params_c['hp']['batch_size_train'] = test_n_batch

    # Modulation lesion — loop over all clustering types and both lesion modes.
    # "zero_W": zero the static weight W at cluster synapses (original method).
    # "freeze_M": keep W intact but freeze plasticity (M stays at M_init) at cluster synapses.
    mod_lesion_modes = ["zero_W", "freeze_M"]
    mod_leison_results = {}

    for mod_type_key, mod_G_idx_cur in mod_type_lst:
        _G_name = cluster_info_mod[mod_type_key]["result_all_name_lst"][mod_G_idx_cur]
        print(f"\n[modulation lesion] type={mod_type_key}  {_G_name}")

        mod_result_cur = cluster_info_mod[mod_type_key]["result_all_lst"][mod_G_idx_cur]
        # Use fixed-k labels if available; if FIXED_K is below k_min, use smallest available k
        if "col_labels_by_k" in mod_result_cur and FIXED_K in mod_result_cur["col_labels_by_k"]:
            mod_col_labels_cur = mod_result_cur["col_labels_by_k"][FIXED_K]
            print(f"  Using fixed k={FIXED_K} for modulation clustering")
        elif "col_labels_by_k" in mod_result_cur and mod_result_cur["col_labels_by_k"]:
            _available_ks = sorted(mod_result_cur["col_labels_by_k"].keys())
            _fallback_k = _available_ks[0]
            mod_col_labels_cur = mod_result_cur["col_labels_by_k"][_fallback_k]
            print(f"  Fixed k={FIXED_K} not in col_labels_by_k (range [{_available_ks[0]},{_available_ks[-1]}]); "
                  f"using smallest available k={_fallback_k}")
        else:
            mod_col_labels_cur = mod_result_cur["col_labels"]
            print(f"  col_labels_by_k not available, using optimal k={mod_result_cur['col_k']}")
        mod_col_clusters_cur = {}
        for flat_idx, label in enumerate(mod_col_labels_cur):
            mod_col_clusters_cur.setdefault(int(label), []).append(flat_idx)
        print(f"  {len(mod_col_clusters_cur)} modulation clusters")

        all_comb_mod = [("mod", None)] + [("mod", i) for i in sorted(mod_col_clusters_cur.keys())]
        all_comb_names_mod = ["mod_noleison"] + [f"mod_c{i}" for i in sorted(mod_col_clusters_cur.keys())]

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

        for mod_lesion_mode in mod_lesion_modes:
            print(f"  [mode={mod_lesion_mode}]")

            if mod_lesion_mode == "zero_W":
                _lesion_fn = leison_modulation_inplace
            else:
                _lesion_fn = leison_modulation_freeze_inplace

            modtask_accs = []
            modrandomtask_accs = []
            modrandomtask_accs_raw = []

            for task in all_tasks:
                print(f"    Evaluating task: {task}")
                test_data, _ = mpn_tasks.generate_trials_wrap(
                    task_params_c, test_n_batch, rules=[task],
                    mode_input="random_batch", device=device, verbose=False
                )
                test_input, test_output, test_mask = test_data

                modaccs = []
                modrandomaccs = []
                modrandomaccs_raw = []
                # Per-task control cache: the random modulation lesion draws
                # n synapses uniformly from all MM positions regardless of
                # cluster identity (and the lesion mode is fixed in this
                # loop), so same-size clusters share one control set.
                ctrl_cache = {}  # n lesioned synapses -> [repeat_num accs]

                for tag, ci in all_comb_mod:
                    saved, _ = _lesion_fn(
                        model, cluster_index=ci,
                        mod_col_clusters_=mod_col_clusters_cur, MM_=MM, n_pre_=n_pre_mod,
                    )
                    with torch.inference_mode():
                        net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='minimal')
                        acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                                   isvalid=True, mode=model.acc_measure)
                    modaccs.append(acc.item())
                    restore_leison(model, saved)
                    del net_out

                    # mod_noleison (ci None) is a no-op on both sides: size 0
                    n_syn = 0 if ci is None else len(mod_col_clusters_cur[ci])
                    if n_syn not in ctrl_cache:
                        rset = []
                        for _ in range(repeat_num):
                            saved_r, _ = _lesion_fn(
                                model, cluster_index=ci,
                                mod_col_clusters_=mod_col_clusters_cur, MM_=MM, n_pre_=n_pre_mod,
                                random=True,
                            )
                            with torch.inference_mode():
                                net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='minimal')
                                acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                                           isvalid=True, mode=model.acc_measure)
                            rset.append(acc.item())
                            restore_leison(model, saved_r)
                            del net_out
                        ctrl_cache[n_syn] = rset
                    modrandomaccs.append(np.mean(ctrl_cache[n_syn]))
                    modrandomaccs_raw.append(list(ctrl_cache[n_syn]))

                print(f"      [ctrl-cache] {task}: {len(ctrl_cache)} unique sizes "
                      f"for {len(all_comb_mod)} conditions")
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
            mod_leison_results[result_key] = {
                "modtask_accs": modtask_accs,
                "modrandomtask_accs": modrandomtask_accs,
                "modrandomtask_accs_raw": modrandomtask_accs_raw,
                "all_comb_names_mod": all_comb_names_mod,
                "all_tasks": all_tasks,
                "mod_G_idx": mod_G_idx_cur,
                "mod_col_clusters": mod_col_clusters_cur,
                "mod_lesion_mode": mod_lesion_mode,
            }

    # Ordering note: corr/l1_dist row/col i → cluster i+1.
    # ihtask_accs columns: [0]=pre_noleison, [1..pre_n]=lesion pre cluster 1..pre_n,
    #                       [pre_n+1]=post_noleison, [pre_n+2..end]=lesion post cluster 1..post_n.
    # So ihtask_accs[:, 1:pre_n+1] aligns with corr_matrices["input"] and l1_dist_matrices["input"].
    saved_dict = {
        "leison": {
            "ihtask_accs": ihtask_accs,
            "all_comb_names_leison": all_comb_names_leison,
            "all_tasks": all_tasks,
            "lesion_units": lesion_units_norm,
        },
        "prune": {
            "wtask_accs": wtask_accs,
            "all_comb_names_prune": all_comb_names_prune,
            "all_tasks": all_tasks,
        },
        "random_leison": {
            "ihrandomtask_accs": ihrandomtask_accs,
            "ihrandomtask_accs_raw": ihrandomtask_accs_raw,
            "all_comb_names_leison": all_comb_names_leison,
            "all_tasks": all_tasks,
        },
        "leison_unnorm": {
            "ihtask_accs": ihtask_accs_unnorm,
            "all_comb_names_leison": all_comb_names_leison_unnorm,
            "all_tasks": all_tasks,
            "lesion_units": lesion_units_unnorm,
        },
        "random_leison_unnorm": {
            "ihrandomtask_accs": ihrandomtask_accs_unnorm,
            "ihrandomtask_accs_raw": ihrandomtask_accs_raw_unnorm,
            "all_comb_names_leison": all_comb_names_leison_unnorm,
            "all_tasks": all_tasks,
        },
        "combined_leison_norm": _combined_cache.get("combined_leison_norm", {}),
        "combined_leison_unnorm": _combined_cache.get("combined_leison_unnorm", {}),
        "mod_leison": mod_leison_results,
        "cluster_similarity": {
            "corr_matrices": corr_matrices,
            "l1_dist_matrices": l1_dist_matrices,
            "cluster_means": cluster_means_cache,
        },
        "fixed_k": FIXED_K,
        "repeat_num": repeat_num,
    }

    with open(f"{save_dir}/lesion_prune_results_{aname}.pkl", "wb") as f:
        pickle.dump(saved_dict, f)