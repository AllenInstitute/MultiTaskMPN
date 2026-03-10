from pathlib import Path 
import json 
import numpy as np 
import seaborn as sns 
import pickle
import copy 
import gc
import sys  
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch 

import mpn 
import mpn_tasks

if __name__ == "__main__":
    aname = "everything_seed749_L21e4+hidden300+batch128+angle"
    normalized = "_normalized"
    
    out_param_path = Path("multiple_tasks/" + f"param_{aname}_param.json")    
    cluster_path = Path("./multiple_tasks/" + f"cluster_info_{aname}{normalized}.pkl")

    with out_param_path.open() as f: 
        raw_cfg_param = json.load(f)
        
    with cluster_path.open("rb") as f:
        cluster_info = pickle.load(f)
        
    # need to remind what the clusters are
    cnames = ["input", "hidden"]
    for cname in cnames:
        print(f"{cname} clusters; row_clusters: {len(cluster_info[cname]['row_clusters'])}, col_clusters: {len(cluster_info[cname]['col_clusters'])}")
        
    task_params, train_params, net_params = raw_cfg_param["task_params"], raw_cfg_param["train_params"], raw_cfg_param["net_params"]
    
    netpathname = "multiple_tasks/" + f"savednet_{aname}.pt"
    checkpoint = torch.load(netpathname, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    print(state_dict.keys())
    load_net_params = checkpoint["net_params"]
    print(load_net_params)
    
    model = mpn.DeepMultiPlasticNet(load_net_params, verbose=False, forzihan=True)

    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
    print("missing:", missing)
    print("unexpected:", unexpected)

    model.eval()

    task_params_c, train_params_c, net_params_c = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )
    
    all_tasks = task_params_c['rules']
    
    pre_n = len(cluster_info["input"]["col_clusters"])
    post_n = len(cluster_info["hidden"]["col_clusters"])

    all_comb = (
        [("pre", i) for i in range(pre_n + 1)] +
        [("post", i) for i in range(post_n + 1)]
    )

    rename = {"pre_c0": "pre_noleison", "post_c0": "post_noleison"}
    all_comb_names_leison = [rename.get(f"{tag}_c{i}", f"{tag}_c{i}") for tag, i in all_comb]
    print(f"all_comb_names_leison: {all_comb_names_leison}")
    
    def leison_prepost(net, cluster_info, cluster_index, preorpost):
        net_copy = copy.deepcopy(net)
        name = "input" if preorpost == "pre" else "hidden"
        
        leison_units = 0 
        if cluster_index != 0: 
            neuron_index = cluster_info[name]["col_clusters"][cluster_index]
            leison_units = len(neuron_index)
            
            if preorpost == "pre":
                net_copy.W_initial_linear.weight.data[neuron_index, :] = 0.0
                net_copy.mp_layer1.W[:, neuron_index].zero_()
            elif preorpost == "post":
                net_copy.mp_layer1.W[neuron_index, :].zero_()
                # the bias of MPN layer is on the postsynaptic side
                net_copy.mp_layer1.b[neuron_index].zero_()
                net_copy.W_output[:, neuron_index].zero_()
        
        return net_copy, leison_units
    
    def prune_w(net, sparsity):
        """
        L2 (magnitude) pruning on net_copy.mp_layer.W:
        keep the top (1 - sparsity) fraction of entries by L2 norm (for scalar entries: |w|)
        and zero out the rest.

        Returns:
            net_copy: deep-copied network with pruned W.
        """
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

        net_copy = copy.deepcopy(net)

        if not hasattr(net_copy, "mp_layer1") or not hasattr(net_copy.mp_layer1, "W"):
            raise AttributeError("Expected net.mp_layer1.W to exist.")

        W = net_copy.mp_layer1.W  
        if not torch.is_tensor(W):
            raise TypeError(f"net_copy.mp_layer1.W must be a torch Tensor/Parameter, got {type(W)}")

        with torch.no_grad():
            numel = W.numel()
            k = int((1.0 - float(sparsity)) * numel)  

            if k <= 0:
                W.zero_()
                return net_copy
            if k >= numel:
                return net_copy 

            w_abs = W.abs().view(-1)
            _, idx = torch.topk(w_abs, k, largest=True, sorted=False)
            mask = torch.zeros(numel, dtype=torch.bool, device=W.device)
            mask[idx] = True
            mask = mask.view_as(W)
            
            W.mul_(mask)

        return net_copy
        
    # register the size for each cluster
    leison_units_all = []
    for idx, comb in enumerate(all_comb):
        _, leison_units = leison_prepost(model, cluster_info, cluster_index=comb[1], preorpost=comb[0])
        leison_units_all.append(leison_units)
        print(f"Lesion condition: {all_comb_names_leison[idx]}, lesioned units: {leison_units}")
    
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.bar(all_comb_names_leison, leison_units_all)
    ax.set_xticks(range(len(all_comb_names_leison)))
    ax.set_xticklabels(all_comb_names_leison, rotation=45, ha="right")
    ax.set_ylabel("# Lesioned Units", fontsize=10)
    ax.set_xlabel("Lesion Condition", fontsize=10)
    ax.tick_params(axis="both", length=2, pad=2)
    fig.tight_layout()
    fig.savefig(f"./multiple_tasks_perf/lesion_units_{aname}.png", dpi=300)
    
    # setup the evaluation dataset generator
    test_n_batch = 10
    task_params_c['hp']['batch_size_train'] = test_n_batch
    
    # pruning for W
    K_lst = [0.0, 10.0, 50.0, 90.0, 95.0, 98.0, 99.0, 99.90]
    sparsity_lst = [k / 100.0 for k in K_lst]
    all_comb_prune = [("prune", k) for k in sparsity_lst]
    all_comb_names_prune = [f"prune_{k:.3f}%" for k in sparsity_lst]
    
    wtask_accs = []

    # leisons for different input & hidden clusters through a leave-one-out manner
    ihtask_accs = []
    
    for task in all_tasks:   
        print(f"Evaluating task: {task}")
        # use a fixed test set for all lesion conditions to reduce variance 
        # and make the comparison more fair
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
            task_params_c, test_n_batch, rules=[task],
            mode_input="random_batch", fix=True, device="cpu", verbose=False
        )
        test_input, test_output, test_mask = test_data
        
        ihaccs, waccs = [], []
        
        for idx, comb in enumerate(all_comb): 
            print(f"Evaluating lesion condition: {all_comb_names_leison[idx]}")
            model_copy, _ = leison_prepost(model, cluster_info, cluster_index=comb[1], preorpost=comb[0])
            
            with torch.no_grad():
                net_out, _, db_test = model_copy.iterate_sequence_batch(test_input, run_mode='track_states')
                acc, _ = model_copy.compute_acc(net_out, test_output, test_mask, test_input, isvalid=True, mode=model_copy.acc_measure)
            
            ihaccs.append(acc.item())
            
            del net_out
            del model_copy
            gc.collect()
            
        for idx in range(len(all_comb_prune)):
            k = all_comb_prune[idx][1]
            print(f"Evaluating pruning condition: {all_comb_names_prune[idx]}") 
            model_copy = prune_w(model, sparsity=k)
            
            with torch.no_grad():
                net_out, _, db_test = model_copy.iterate_sequence_batch(test_input, run_mode='track_states')
                acc, _ = model_copy.compute_acc(net_out, test_output, test_mask, test_input, isvalid=True, mode=model_copy.acc_measure)
            
            waccs.append(acc.item())
            
            del net_out
            del model_copy
            gc.collect()
        
        ihtask_accs.append(ihaccs)
        wtask_accs.append(waccs)

    def plot_heatmap(input_matrix, all_comb_names_, all_tasks_, xlabel, ylabel, savename):
        """
        """
        A = np.asarray(input_matrix, dtype=float)
        mask = ~np.isfinite(A)
        fig_w = max(6, 0.55 * len(all_tasks_) + 2.5)
        fig_h = max(4, 0.40 * len(all_comb_names_) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
        
        hm = sns.heatmap(
            A,
            mask=mask,
            cmap="magma_r",
            vmin=0.0, vmax=1.0,                
            annot=True,
            fmt=".2f",                          
            annot_kws={"fontsize": 8},
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Accuracy", "shrink": 0.9, "pad": 0.02},
            ax=ax,
        )

        ax.set_xticklabels(all_comb_names_, rotation=45, ha="right")
        ax.set_yticklabels(all_tasks_, rotation=0)    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="both", length=2, pad=2)

        cbar = hm.collections[0].colorbar
        cbar.ax.yaxis.set_major_locator(MaxNLocator(6))

        for t in hm.texts:
            try:
                v = float(t.get_text())
                t.set_color("white" if v < 0.55 else "black")
            except ValueError:
                pass

        fig.tight_layout()
        fig.savefig(f"./multiple_tasks_perf/{savename}_heatmap_{aname}.png", dpi=300)
        
    plot_heatmap(
        ihtask_accs, all_comb_names_leison, all_tasks, xlabel="Lesion Condition", ylabel="Task", savename="lesion"
    )   
    
    plot_heatmap(
        wtask_accs, all_comb_names_prune, all_tasks, xlabel="Pruning Condition", ylabel="Task", savename="pruning"
    )
