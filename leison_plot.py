import numpy as np 
import pandas as pd 

import pickle 

import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib.ticker import MaxNLocator

import leison as leison_helper

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

if __name__ == "__main__": 
    aname = "everything_seed749_L21e4+hidden300+batch128+angle"
    
    pickle_name = f"./multiple_tasks_perf/lesion_prune_results_{aname}.pkl"
    with open(pickle_name, 'rb') as f:
        results = pickle.load(f)
        
    all_comb_names_leison = results["leison"]["all_comb_names_leison"]
    all_comb_names_leison_ = [k for k in all_comb_names_leison if k not in ("pre_cNone", "post_cNone")]
    all_tasks = results["leison"]["all_tasks"]
    ihtask_accs = np.asarray(results["leison"]["ihtask_accs"], dtype=float)
    ihrandomtask_accs = np.asarray(results["random_leison"]["ihrandomtask_accs"], dtype=float)
    
    select_props = []
    for key_idx, key in enumerate(all_comb_names_leison):
        if key not in ("pre_cNone", "post_cNone"):
            leison = ihtask_accs[:, key_idx]
            random_leison = ihrandomtask_accs[:, key_idx]
            noleison = ihtask_accs[:,0]
            leison_diff = noleison - leison
            random_leison_diff = noleison - random_leison
            normalized_leison_diff = random_leison - leison
            select_props.append(normalized_leison_diff) 
            
    select_props = np.array(select_props).T           
    
    leison_helper.plot_heatmap(select_props, all_comb_names_leison_, all_tasks, 
                               xlabel="Lesion Condition", ylabel="Task", savename="normalized_leison", 
                               aname=aname, label="Normalized Accuracy")
            
    
    