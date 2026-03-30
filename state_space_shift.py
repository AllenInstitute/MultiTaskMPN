from pathlib import Path 
import json 
import numpy as np 
import seaborn as sns 
import pickle
import copy 
import gc
import sys  

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

from sklearn.decomposition import PCA

import torch 

import mpn 
import mpn_tasks
import color_func
import helper

c_vals = color_func.rainbow_generate(15)
c_vals_l = color_func.rainbow_generate(30)

def generate_response_stimulus(task_params, test_trials): 
    """
    """
    labels_resp, labels_stim1, labels_stim2 = [], [], []
    rules_epochs = {} 
    for rule_idx, rule in enumerate(task_params['rules']):
        rules_epochs[rule] = test_trials[rule_idx].epochs
        # print(test_trials[rule_idx].meta.keys())
        
        if rule in ('dmsgo','dmcgo','dmsnogo','dmcnogo',):
            labels_resp.append(test_trials[rule_idx].meta['matches'])                
        else:
            labels_resp.append(test_trials[rule_idx].meta['resp1'])           
            
        labels_stim1.append(test_trials[rule_idx].meta['stim1']) 
        
        try: 
            labels_stim2.append(test_trials[rule_idx].meta['stim2'])
        except KeyError:
            labels_stim2.append(np.full_like(test_trials[rule_idx].meta['stim1'], np.nan))

    
    labels_resp = np.concatenate(labels_resp, axis=0).reshape(-1,1)
    labels_stim1 = np.concatenate(labels_stim1, axis=0).reshape(-1,1)
    labels_stim2 = np.concatenate(labels_stim2, axis=0).reshape(-1,1)

    return labels_resp, labels_stim1, labels_stim2, rules_epochs

if __name__ == "__main__":
    aname = "everything_seed749_L21e4+hidden300+batch128+angle"
    normalized = "_normalized"
    
    out_param_path = Path("multiple_tasks/" + f"param_{aname}_param.json")   
    
    with out_param_path.open() as f: 
        raw_cfg_param = json.load(f)
    
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
    
    # setup the evaluation dataset generator
    test_n_batch = 10
    task_params_c['hp']['batch_size_train'] = test_n_batch
    
    test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
        task_params_c, 
        test_n_batch, 
        rules=all_tasks,
        mode_input="random", 
        device="cpu", 
        verbose=False
    )
    test_input, test_output, test_mask = test_data
    _, test_trials, test_rule_idxs = test_trials_extra
    
    with torch.no_grad():
        net_out, _, db_test = model.iterate_sequence_batch(test_input, run_mode='track_states')
        acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input, isvalid=True, mode=model.acc_measure)
        
    print(f"acc: {acc:.2f}")
        
    Ms_orig = db_test["M1"].cpu().numpy()
    Ms = Ms_orig.reshape(Ms_orig.shape[0], Ms_orig.shape[1], -1) 
    xs = db_test["input1"].cpu().numpy()
    hs = db_test["hidden1"].cpu().numpy()
    print(f"Ms_orig.shape: {Ms_orig.shape}; Ms.shape: {Ms.shape}; xs.shape: {xs.shape}; hs.shape: {hs.shape}")
    
    _, labels_stim1, labels_stim2, rules_epochs = generate_response_stimulus(task_params_c, test_trials)
    
    labels_stim1 = labels_stim1.flatten()
    all_rules = task_params_c['rules']
        
    ctx_endfix = []
    ctx_rule_labels = []

    for idx, rule in enumerate(all_rules):
        ctx_endtime = rules_epochs[rule]['fix1'][1]
        states = hs[test_rule_idxs == idx, ctx_endtime - 1, :]
        ctx_endfix.append(states)
        ctx_rule_labels.append(np.full(states.shape[0], idx))

    ctx_extract = np.concatenate(ctx_endfix, axis=0)
    ctx_rule_labels = np.concatenate(ctx_rule_labels, axis=0)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(ctx_extract)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for idx, rule in enumerate(all_rules):
        ctx_values = X_2d[ctx_rule_labels == idx]
        ax.scatter(ctx_values[:, 0], ctx_values[:, 1], label=rule, color=c_vals[idx], alpha=0.5)

    ax.set_xlabel("Context endpt. state PC1")
    ax.set_ylabel("Context endpt. state PC2")
    ax.legend(frameon=True, loc='best', fontsize=6)
    fig.tight_layout()
    fig.savefig("./state_space/state_space_shift.png", dpi=300)
    
    # distance between initial conditions vs. angle between first step (closer to Fig 4C)
    fig, axs = plt.subplots(1, 2, figsize=(4*2, 4))
    
    datas = [hs, Ms]
    
    for idx, data in enumerate(datas):
        all_x, all_y = [], []

        for idx1, rule1 in enumerate(all_rules):
            for idx2, rule2 in enumerate(all_rules):
                if idx1 >= idx2:
                    continue

                ctx_endtime1 = rules_epochs[rule1]['fix1'][1]
                ctx_endtime2 = rules_epochs[rule2]['fix1'][1]

                stim_dists = []
                stim_angles = []

                for stimidx in range(8):
                    match1 = (labels_stim1 == stimidx) & (test_rule_idxs == idx1)
                    match2 = (labels_stim1 == stimidx) & (test_rule_idxs == idx2)

                    idxs1 = np.where(match1)[0]
                    idxs2 = np.where(match2)[0]

                    if len(idxs1) == 0 or len(idxs2) == 0:
                        continue

                    # initial conditions: end of context / just before stimulus onset
                    h0_1 = data[idxs1, ctx_endtime1-1, :]   # shape (n1, N)
                    h0_2 = data[idxs2, ctx_endtime2-1, :]   # shape (n2, N)
                    
                    shift_time = 3

                    # shift_time+1 step vectors at stimulus onset
                    dh1 = data[idxs1, ctx_endtime1+shift_time, :] - h0_1
                    dh2 = data[idxs2, ctx_endtime2+shift_time, :] - h0_2

                    # compare matched trial pairs
                    # if counts differ, use all cross-pairs
                    pair_dists = []
                    pair_angles = []

                    for i in range(len(idxs1)):
                        for j in range(len(idxs2)):
                            d = np.linalg.norm(h0_1[i] - h0_2[j])

                            n1 = np.linalg.norm(dh1[i])
                            n2 = np.linalg.norm(dh2[j])
                            if n1 < 1e-12 or n2 < 1e-12:
                                continue

                            cosang = np.dot(dh1[i], dh2[j]) / (n1 * n2)
                            cosang = np.clip(cosang, -1.0, 1.0)
                            ang = np.degrees(np.arccos(cosang))

                            pair_dists.append(d)
                            pair_angles.append(ang)

                    if len(pair_dists) > 0:
                        stim_dists.append(np.mean(pair_dists))
                        stim_angles.append(np.mean(pair_angles))

                if len(stim_dists) > 0:
                    axs[idx].scatter(np.mean(stim_dists), np.mean(stim_angles), color='gray', alpha=0.5)
                    all_x.append(np.mean(stim_dists))
                    all_y.append(np.mean(stim_angles))
                    
        x_fit, y_fit, r_value, slope, intercept, p_value = helper.linear_regression(np.array(all_x), np.array(all_y), log=False, through_origin=True)
        axs[idx].plot(x_fit, y_fit, color='red', label=f"Fit: slope={slope:.2f}, r={r_value:.2f}, p={p_value:.3f}")
    
        axs[idx].set_xlabel("Distance between initial conditions")
        axs[idx].set_ylabel(f"Angle between {shift_time+1} step of trajectories (deg.)")
        axs[idx].legend(frameon=True, loc='best', fontsize=6)
        
    fig.tight_layout()
    fig.savefig(f"./state_space/initial_condition_distance_vs_angle_{shift_time+1}.png", dpi=300)
                    
    
    
    
    
        
