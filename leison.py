from pathlib import Path
import json
import os
import re
import numpy as np
import seaborn as sns
import pickle
import gc
import sys 
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

import mpn 
import mpn_tasks
import helper

def main(seed, feature):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    repeat_num = 10

    aname = f"everything_seed{seed}_{feature}+hidden300+batch128+angle"

    save_dir = f"./multiple_tasks_perf/{aname}"
    os.makedirs(save_dir, exist_ok=True)
    for _old in Path(save_dir).iterdir():
        if _old.is_file():
            _old.unlink()

    out_param_path = Path("multiple_tasks/" + f"/param_{aname}_param.json")
    cluster_path = Path(f"./multiple_tasks/{aname}/cluster_info_{aname}.pkl")
    cluster_path_mod = Path(f"./multiple_tasks/{aname}/cluster_info_mod_{aname}.pkl")

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
    
    pre_n = len(cluster_info["input_normalized"]["col_clusters"])
    post_n = len(cluster_info["hidden_normalized"]["col_clusters"])

    # Cluster mean correlation matrices, ordered by cluster number (1..n_clusters),
    # consistent with the lesion order used in leison_prepost (all_comb below).
    # For each neuron cluster: average across neurons → length-n_tasks profile.
    # Then correlate those profiles pairwise across neuron clusters.
    corr_matrices = {}
    l1_dist_matrices = {}
    cluster_means_cache = {}
    for name, n_clusters in [("input_normalized", pre_n), ("hidden_normalized", post_n)]:
        V = cluster_info[name]["cell_vars_rules_sorted_norm"]  # (n_tasks, n_neurons)
        col_clusters = cluster_info[name]["col_clusters"]      # {label: neuron indices}, 1-based
        cluster_means = np.stack(
            [V[:, col_clusters[c]].mean(axis=1) for c in range(1, n_clusters + 1)],
            axis=1
        )  # (n_tasks, n_clusters)
        cluster_means_cache[name] = cluster_means
        corr_matrices[name] = np.corrcoef(cluster_means.T)  # (n_clusters, n_clusters)
        l1_dist_matrices[name] = squareform(pdist(cluster_means.T, metric="cityblock"))  # (n_clusters, n_clusters)
        print(f"{name} cluster corr matrix shape: {corr_matrices[name].shape}")

    # plot the heatmaps for both correlation and L1 distance matrices for input and hidden clusters
    for name, n_clusters in [("input_normalized", pre_n), ("hidden_normalized", post_n)]:
        corr = corr_matrices[name]
        l1_dist = l1_dist_matrices[name]

        tick_labels = [f"c{c}" for c in range(1, n_clusters + 1)]
        upper_mask = np.triu(np.ones((n_clusters, n_clusters), dtype=bool), k=1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

        for ax, mat, title, cmap, vmin, vmax in [
            (axes[0], corr,    f"Correlation Between {name} Clusters",  "coolwarm", 0.0, 1.0),
            (axes[1], l1_dist, f"L1 Distance Between {name} Clusters",  "coolwarm", None, None),
        ]:
            sns.heatmap(mat, mask=upper_mask, cmap=cmap, vmin=vmin, vmax=vmax, 
                        square=True, cbar=True,
                        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(tick_labels, rotation=0, fontsize=9)
            ax.set_xlabel("Neuron Clusters", fontsize=15)
            ax.set_ylabel("Neuron Clusters", fontsize=15)
            ax.set_title(title, fontsize=12)

        fig.tight_layout()
        fig.savefig(f"{save_dir}/cluster_corr_{name}_{aname}.png", dpi=300)
        plt.close(fig)
        
    print(f"Plot the heatmaps for both correlation and L1 distance matrices for input and hidden clusters done.")
    
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
        for name in ["input_normalized", "hidden_normalized"]
    }

    # All modulation clustering types to run lesion experiments on.
    # G_lst index: 0=G100, 1=G300, 2=G1000.  G=300 (idx 1) is used for all types.
    G_lst_leison = [100, 300, 1000]
    mod_type_lst_all = [
        ("modulation_all_normalized",                1),
        ("modulation_all_unnormalized",              1),
        ("modulation_all_weighted_unnormalized",     1),
        ("modulation_all_var_weighted_unnormalized", 1),
    ]
    mod_type_lst = [(k, g) for k, g in mod_type_lst_all if k in cluster_info_mod]
    print(f"Modulation types to lesion ({len(mod_type_lst)}): {[k for k,_ in mod_type_lst]}")

    # Infer M from the first available type (same for all — shared modulation matrix size)
    _first_labels = cluster_info_mod[mod_type_lst[0][0]]["result_all_lst"][mod_type_lst[0][1]]["col_labels"]
    MM = len(_first_labels)
    M = int(np.sqrt(MM))
    assert M * M == MM, f"Expected square modulation matrix, got {MM} entries"

    def leison_prepost_inplace(net, cluster_index, preorpost, random=False):
        """Apply lesion in-place on net; returns (saved_state, leison_units).
        Call restore_leison(net, saved_state) to undo.  No deepcopy."""
        name = "input_normalized" if preorpost == "pre" else "hidden_normalized"
        leison_units = 0
        saved = {}

        if cluster_index is not None:
            neuron_index = cluster_info[name]["col_clusters"][cluster_index]
            class_N = len(neuron_index)

            if not random:
                neuron_index = torch.tensor(neuron_index, dtype=torch.long, device=net.W_output.device)
                leison_units = len(neuron_index)
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

    def leison_modulation_inplace(net, cluster_index, mod_col_clusters_, MM_, M_, random=False):
        """Lesion a modulation cluster by zeroing mp_layer1.W[post, pre] entries.
        flat_idx k → pre = k // M_, post = k % M_ → W[post, pre].
        Returns (saved_state, n_lesioned)."""
        if cluster_index is None:
            return {}, 0
        flat_idxs = np.array(mod_col_clusters_[cluster_index], dtype=int)
        n = len(flat_idxs)
        if random:
            flat_idxs = np.random.choice(MM_, size=n, replace=False)
        pre_t = torch.tensor(flat_idxs // M_, dtype=torch.long, device=net.mp_layer1.W.device)
        post_t = torch.tensor(flat_idxs % M_, dtype=torch.long, device=net.mp_layer1.W.device)
        with torch.no_grad():
            saved = {
                "preorpost": "modulation",
                "pre_t": pre_t,
                "post_t": post_t,
                "W_vals": net.mp_layer1.W[post_t, pre_t].clone(),
            }
            net.mp_layer1.W[post_t, pre_t] = 0.0
        return saved, n

    def leison_modulation_freeze_inplace(net, cluster_index, mod_col_clusters_, MM_, M_, random=False):
        """Freeze plasticity at a modulation cluster: M stays at its initial value
        at those (post, pre) positions throughout the trial, but W is untouched.
        Returns (saved_state, n_frozen)."""
        if cluster_index is None:
            return {}, 0
        flat_idxs = np.array(mod_col_clusters_[cluster_index], dtype=int)
        n = len(flat_idxs)
        if random:
            flat_idxs = np.random.choice(MM_, size=n, replace=False)
        pre_t = torch.tensor(flat_idxs // M_, dtype=torch.long, device=net.mp_layer1.W.device)
        post_t = torch.tensor(flat_idxs % M_, dtype=torch.long, device=net.mp_layer1.W.device)
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

    # register the size for each cluster — no model copy needed
    leison_units_all = []
    for tag, ci in all_comb:
        if ci is None:
            leison_units_all.append(0)
        else:
            name = "input_normalized" if tag == "pre" else "hidden_normalized"
            leison_units_all.append(len(cluster_info[name]["col_clusters"][ci]))
    for idx, units in enumerate(leison_units_all):
        print(f"Lesion condition: {all_comb_names_leison[idx]}, lesioned units: {units}")
    
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.bar(all_comb_names_leison, leison_units_all)
    ax.set_xticks(range(len(all_comb_names_leison)))
    ax.set_xticklabels(all_comb_names_leison, rotation=45, ha="right")
    ax.set_ylabel("# Lesioned Units", fontsize=10)
    ax.set_xlabel("Lesion Condition", fontsize=10)
    ax.tick_params(axis="both", length=2, pad=2)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/lesion_units_{aname}.png", dpi=300)
    
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

    # leisons for different input & hidden clusters through a leave-one-out manner
    ihtask_accs, wtask_accs = [], []
    ihrandomtask_accs = []

    for task in all_tasks:
        print(f"Evaluating task: {task}")
        # use a fixed test set for all lesion conditions to reduce variance
        # and make the comparison more fair
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
            task_params_c, test_n_batch, rules=[task],
            mode_input="random_batch", device=device, verbose=False
        )
        test_input, test_output, test_mask = test_data

        ihaccs, waccs = [], []
        ihrandomaccs = []

        # experiment for leisons on different clusters
        for idx, comb in enumerate(all_comb):
            print(f"Evaluating lesion condition: {all_comb_names_leison[idx]}")
            saved, _ = leison_prepost_inplace(model, cluster_index=comb[1], preorpost=comb[0])

            with torch.no_grad():
                net_out, _, db_test = model.iterate_sequence_batch(test_input, run_mode='track_states')
                acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input, 
                                           isvalid=True, mode=model.acc_measure)

            ihaccs.append(acc.item())
            restore_leison(model, saved)
            del net_out
            gc.collect()

            # what about randomly leisoning the same number of units from the same layer
            # this can serve as a sanity check to see if the specific clusters we identified are more important than random sets of neurons.
            random_leison_repeat = repeat_num
            rset = []
            for _ in range(random_leison_repeat):
                saved_r, _ = leison_prepost_inplace(model, cluster_index=comb[1], 
                                                    preorpost=comb[0], random=True)

                with torch.no_grad():
                    net_out, _, db_test = model.iterate_sequence_batch(test_input, run_mode='track_states')
                    acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input, 
                                               isvalid=True, mode=model.acc_measure)
                rset.append(acc.item())
                restore_leison(model, saved_r)
                del net_out

            ihrandomaccs.append(np.mean(rset))

        # pruning with different sparsity levels — swap precomputed W in/out
        for idx, W_pruned in enumerate(pruned_Ws):
            print(f"Evaluating pruning condition: {all_comb_names_prune[idx]}")
            model.mp_layer1.W.data.copy_(W_pruned)

            with torch.no_grad():
                net_out, _, db_test = model.iterate_sequence_batch(test_input, run_mode='track_states')
                acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                           isvalid=True, mode=model.acc_measure)

            waccs.append(acc.item())
            del net_out
            gc.collect()

        # restore original W after pruning loop
        model.mp_layer1.W.data.copy_(W_orig)
        
        ihtask_accs.append(ihaccs)
        wtask_accs.append(waccs)
        ihrandomtask_accs.append(ihrandomaccs)
        
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

    # Modulation lesion — loop over all clustering types and both lesion modes.
    # "zero_W": zero the static weight W at cluster synapses (original method).
    # "freeze_M": keep W intact but freeze plasticity (M stays at M_init) at cluster synapses.
    mod_lesion_modes = ["zero_W", "freeze_M"]
    mod_leison_results = {}

    for mod_type_key, mod_G_idx_cur in mod_type_lst:
        print(f"\n[modulation lesion] type={mod_type_key}  G={G_lst_leison[mod_G_idx_cur]}")

        mod_result_cur = cluster_info_mod[mod_type_key]["result_all_lst"][mod_G_idx_cur]
        mod_col_labels_cur = mod_result_cur["col_labels"]
        mod_col_clusters_cur = {}
        for flat_idx, label in enumerate(mod_col_labels_cur):
            mod_col_clusters_cur.setdefault(int(label), []).append(flat_idx)
        print(f"  {len(mod_col_clusters_cur)} modulation clusters")

        all_comb_mod = [("mod", None)] + [("mod", i) for i in sorted(mod_col_clusters_cur.keys())]
        all_comb_names_mod = ["mod_noleison"] + [f"mod_c{i}" for i in sorted(mod_col_clusters_cur.keys())]

        for mod_lesion_mode in mod_lesion_modes:
            print(f"  [mode={mod_lesion_mode}]")

            if mod_lesion_mode == "zero_W":
                _lesion_fn = leison_modulation_inplace
            else:
                _lesion_fn = leison_modulation_freeze_inplace

            modtask_accs = []
            modrandomtask_accs = []

            for task in all_tasks:
                print(f"    Evaluating task: {task}")
                test_data, _ = mpn_tasks.generate_trials_wrap(
                    task_params_c, test_n_batch, rules=[task],
                    mode_input="random_batch", device=device, verbose=False
                )
                test_input, test_output, test_mask = test_data

                modaccs = []
                modrandomaccs = []

                for tag, ci in all_comb_mod:
                    saved, _ = _lesion_fn(
                        model, cluster_index=ci,
                        mod_col_clusters_=mod_col_clusters_cur, MM_=MM, M_=M,
                    )
                    with torch.no_grad():
                        net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='track_states')
                        acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                                   isvalid=True, mode=model.acc_measure)
                    modaccs.append(acc.item())
                    restore_leison(model, saved)
                    del net_out
                    gc.collect()

                    rset = []
                    for _ in range(random_leison_repeat):
                        saved_r, _ = _lesion_fn(
                            model, cluster_index=ci,
                            mod_col_clusters_=mod_col_clusters_cur, MM_=MM, M_=M,
                            random=True,
                        )
                        with torch.no_grad():
                            net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode='track_states')
                            acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                                       isvalid=True, mode=model.acc_measure)
                        rset.append(acc.item())
                        restore_leison(model, saved_r)
                        del net_out
                    modrandomaccs.append(np.mean(rset))

                modtask_accs.append(modaccs)
                modrandomtask_accs.append(modrandomaccs)

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
        },
        "prune": {
            "wtask_accs": wtask_accs,
            "all_comb_names_prune": all_comb_names_prune,
            "all_tasks": all_tasks,
        },
        "random_leison": {
            "ihrandomtask_accs": ihrandomtask_accs,
            "all_comb_names_leison": all_comb_names_leison,
            "all_tasks": all_tasks,
        },
        "mod_leison": mod_leison_results,
        "cluster_similarity": {
            "corr_matrices": corr_matrices,       # {name: (n_clusters, n_clusters)}
            "l1_dist_matrices": l1_dist_matrices, # {name: (n_clusters, n_clusters)}
            "cluster_means": cluster_means_cache, # {name: (n_tasks, n_clusters)}
        },
    }

    with open(f"{save_dir}/lesion_prune_results_{aname}.pkl", "wb") as f:
        pickle.dump(saved_dict, f)
        
if __name__ == "__main__":
    saved_nets = sorted(Path("multiple_tasks").glob("savednet_everything_seed*+angle.pt"))
    param_lst = []
    for p in saved_nets:
        m = re.match(r"savednet_everything_seed(\d+)_(\w+)\+hidden\d+\+batch\d+\+angle\.pt", p.name)
        if m:
            param_lst.append((int(m.group(1)), m.group(2)))

    print(f"Found {len(param_lst)} saved models: {param_lst}")

    param_lst = [(749, "L21e4")]

    for seed, feature in param_lst:
        aname = f"everything_seed{seed}_{feature}+hidden300+batch128+angle"
        cluster_path     = Path(f"./multiple_tasks/{aname}/cluster_info_{aname}.pkl")
        cluster_path_mod = Path(f"./multiple_tasks/{aname}/cluster_info_mod_{aname}.pkl")
        if not cluster_path.exists() or not cluster_path_mod.exists():
            print(f"Skipping {aname}: cluster files not found (run multiple_task_analysis.py first)")
            continue
        main(seed, feature)