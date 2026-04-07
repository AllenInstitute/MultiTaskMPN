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
import multiple_task_performance as mpf 

c_vals = color_func.rainbow_generate(15)
c_vals_l = color_func.rainbow_generate(30)

def eval_one(netpathname):
    """
    """
    hidden_size, l2_info = mpf.parse_hidden_and_l2(netpathname)
    
    aname = netpathname[24:-3]
    
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
    
    noise_level = 0.01
    task_params["sigma_x"] = noise_level

    task_params_c, train_params_c, net_params_c = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )
    
    all_tasks = task_params_c['rules']
    
    # setup the evaluation dataset generator
    test_n_batch = 50
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
    modulation_W = state_dict["mp_layer1.W"].cpu().numpy()
    eff_Ms_orig = Ms_orig * modulation_W
        
    Ms = Ms_orig.reshape(Ms_orig.shape[0], Ms_orig.shape[1], -1) 
    eff_Ms = eff_Ms_orig.reshape(eff_Ms_orig.shape[0], eff_Ms_orig.shape[1], -1)
    xs = db_test["input1"].cpu().numpy()
    hs = db_test["hidden1"].cpu().numpy()
    print(f"Ms_orig.shape: {Ms_orig.shape}; Ms.shape: {Ms.shape}; xs.shape: {xs.shape}; hs.shape: {hs.shape}")
    
    _, labels_stim1, _, rules_epochs = helper.generate_response_stimulus(task_params_c, test_trials)
    
    labels_stim1 = labels_stim1.flatten()
    all_rules = task_params_c['rules']
    
    # rule -> (paper-style computation category, plotting color)
    # https://www.nature.com/articles/s41593-024-01668-6/figures/4
    rule_motif_mapping = {
        "fdgo":           ("Pro Delayed",    "blue"),
        "fdanti":         ("Anti Delayed",   "red"),
        "delaygo":        ("Pro Delayed",    "blue"),
        "delayanti":      ("Anti Delayed",   "red"),
        "reactgo":        ("Pro Reaction",   "green"),
        "reactanti":      ("Anti Reaction",  "orange"),

        "contextdelaydm1": ("Pro Integration", "steelblue"),
        "contextdelaydm2": ("Pro Integration", "steelblue"),
        "delaydm1":        ("Pro Integration", "steelblue"),
        "delaydm2":        ("Pro Integration", "steelblue"),
        "multidelaydm":    ("Pro Integration", "steelblue"),

        "dmsgo":          ("Categorization", "green"),
        "dmsnogo":        ("Categorization", "orange"),
        "dmcgo":          ("Categorization", "deeppink"),
        "dmcnogo":        ("Categorization", "deeppink"),
    }

    assert set(all_rules).issubset(set(rule_motif_mapping.keys()))
    assert len(np.unique([v[0] for v in rule_motif_mapping.values()])) == 6
    
    embed_data_names = ["hidden", "mod", "eff_mod"]
    embed_data = [hs, Ms, eff_Ms]
    
    for data_name, data in zip(embed_data_names, embed_data):
        print(f"Processing {data_name}...")
        ctx_endfix = []
        ctx_rule_labels = []

        for idx, rule in enumerate(all_rules):
            ctx_endtime = rules_epochs[rule]['fix1'][1]
            states = data[test_rule_idxs == idx, ctx_endtime - 1, :]
            ctx_endfix.append(states)
            ctx_rule_labels.append(np.full(states.shape[0], idx))

        ctx_extract = np.concatenate(ctx_endfix, axis=0)
        ctx_rule_labels = np.concatenate(ctx_rule_labels, axis=0)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(ctx_extract)

        fig, axs = plt.subplots(1,2,figsize=(4*2,4))

        # -------------------------
        # Panel 1: color by rule
        # -------------------------
        for idx, rule in enumerate(all_rules):
            sel = (ctx_rule_labels == idx)
            ctx_values = X_2d[sel]
            axs[0].scatter(
                ctx_values[:, 0],
                ctx_values[:, 1],
                label=rule,
                color=c_vals[idx],
                alpha=0.5,
                s=18,
            )

        axs[0].set_title("Colored by rule")
        axs[0].set_xlabel("Context endpoint state PC1")
        axs[0].set_ylabel("Context endpoint state PC2")
        axs[0].legend(frameon=True, loc='best', fontsize=6)

        # -------------------------
        # Panel 2: color by paper computation category
        # -------------------------
        category_order = [
            "Pro Delayed",
            "Anti Delayed",
            "Pro Reaction",
            "Anti Reaction",
            "Pro Integration",
            "Categorization",
        ]

        category_to_color = {}
        for rule, (cat, color) in rule_motif_mapping.items():
            category_to_color[cat] = color

        for cat in category_order:
            rule_idxs_in_cat = [
                idx for idx, rule in enumerate(all_rules)
                if rule_motif_mapping[rule][0] == cat
            ]
            sel = np.isin(ctx_rule_labels, rule_idxs_in_cat)
            ctx_values = X_2d[sel]
            axs[1].scatter(
                ctx_values[:, 0],
                ctx_values[:, 1],
                label=cat,
                color=category_to_color[cat],
                alpha=0.5,
                s=18,
            )

        axs[1].set_title("Colored by computation category")
        axs[1].set_xlabel("Context endpoint state PC1")
        axs[1].set_ylabel("Context endpoint state PC2")
        axs[1].legend(frameon=True, loc='best', fontsize=7)

        fig.tight_layout()
        fig.savefig(
            f"./state_space/state_space_shift_{aname}_{data_name}_noise{noise_level}.png",
            dpi=300,
            bbox_inches="tight",
        )
            
    # distance between initial conditions vs. angle between first step (closer to Fig 4C)
    def fig4c(shift_time):
        fig, axs = plt.subplots(1, len(embed_data), figsize=(4*len(embed_data),4))
        rval_dict = {}
    
        for idx, data in enumerate(embed_data):
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
            rval_dict[embed_data_names[idx]] = (r_value, slope, p_value)
            axs[idx].plot(x_fit, y_fit, color='red', label=f"Fit: slope={slope:.2f}, r={r_value:.2f}, p={p_value:.3f}")
        
            axs[idx].set_xlabel("Distance between initial conditions")
            axs[idx].set_ylabel(f"Angle between {shift_time+1} step of trajectories (deg.)")
            axs[idx].set_title(f"{embed_data_names[idx]}")
            axs[idx].legend(frameon=True, loc='best', fontsize=6)
            
        fig.tight_layout()
        fig.savefig(f"./state_space/initial_condition_distance_vs_angle_{aname}_{shift_time+1}_noise{noise_level}.png", dpi=300)
        
        return rval_dict
    
    rval_dict = fig4c(shift_time=0)
    
    return aname, hidden_size, l2_info, rval_dict

def run_all():
    pt_paths = mpf.list_pt_files("./multiple_tasks", recursive=False)
    
    result_dict = {}
    for netpathname in pt_paths:
        aname, hidden_size, l2_info, rval_dict = eval_one(netpathname)
        result_dict[aname] = {"hidden_size": hidden_size, "l2_info": l2_info, "rval_dict": rval_dict}
        
    with open("./state_space/initial_condition_distance_vs_angle_results.pkl", "wb") as f:
        pickle.dump(result_dict, f)
        
def summarize():
    with open("./state_space/initial_condition_distance_vs_angle_results.pkl", "rb") as f:
        result_dict = pickle.load(f)
    
    # Organize r-values by data type
    data_types = ["hidden", "mod", "eff_mod"]
    r_values = {dt: [] for dt in data_types}
    slopes = {dt: [] for dt in data_types}
    p_values = {dt: [] for dt in data_types}
    hidden_sizes = []
    l2_infos = []
    anames = []
    
    # Extract all values
    for aname, results in result_dict.items():
        anames.append(aname)
        hidden_sizes.append(results["hidden_size"])
        l2_infos.append(results["l2_info"])
        
        for dt in data_types:
            if dt in results["rval_dict"]:
                r_val, slope, p_val = results["rval_dict"][dt]
                r_values[dt].append(r_val)
                slopes[dt].append(slope)
                p_values[dt].append(p_val)
    
    # Print summary statistics
    print("=" * 80)
    print(f"Summary of {len(result_dict)} networks")
    print("=" * 80)
    
    for dt in data_types:
        print(f"\n{dt.upper()}:")
        r_arr = np.array(r_values[dt])
        slope_arr = np.array(slopes[dt])
        p_arr = np.array(p_values[dt])
        
        print(f"  R-values: mean={np.mean(r_arr):.3f}, std={np.std(r_arr):.3f}, "
              f"min={np.min(r_arr):.3f}, max={np.max(r_arr):.3f}")
        print(f"  Slopes:   mean={np.mean(slope_arr):.3f}, std={np.std(slope_arr):.3f}, "
              f"min={np.min(slope_arr):.3f}, max={np.max(slope_arr):.3f}")
        print(f"  P-values: mean={np.mean(p_arr):.4f}, significant (p<0.05): {np.sum(p_arr < 0.05)}/{len(p_arr)}")
    
    # Create comparison visualization
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    
    # Plot 1: R-values comparison
    positions = np.arange(len(data_types))
    r_means = [np.mean(r_values[dt]) for dt in data_types]
    r_stds = [np.std(r_values[dt]) for dt in data_types]
    
    ax.bar(positions, r_means, yerr=r_stds, capsize=5, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_xticks(positions)
    ax.set_xticklabels(data_types, rotation=45, ha='right')
    ax.set_ylabel('R-value')
    ax.set_title('Mean R-values across networks')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig('./state_space/summary_r_values.png', dpi=300)
    print("\nSaved summary plot to ./state_space/summary_r_values.png")

if __name__ == "__main__":    
    run_all()
    summarize()
