# %%
import numpy as np  
from numpy.linalg import norm 
from pathlib import Path
import json
import psutil
import copy
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix as sk_contingency_matrix
ticker.Locator.MAXTICKS = 10000 
import seaborn as sns 

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


from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

import torch
from torch.serialization import add_safe_globals

import helper           
import clustering
import clustering_metric
import color_func
import mpn 
import mpn_tasks

c_vals = [
    "#e53e3e",  # red
    "#3182ce",  # blue
    "#38a169",  # green
    "#d69e2e",  # yellow-gold
    "#d53f8c",  # pink-magenta
    "#4c51bf",  # indigo
    "#dd6b20",  # orange
    "#0ea5e9",  # sky blue
    "#22c55e",  # bright green
    "#a855f7",  # purple
    "#f43f5e",  # red-pink
    "#0f766e",  # teal
    "#b83280",  # magenta-violet
    "#ca8a04",  # amber
    "#2b6cb0",  # deep blue
] * 10

c_vals_l = [
    "#feb2b2",  # light red
    "#90cdf4",  # light blue
    "#9ae6b4",  # light green
    "#faf089",  # light yellow-gold
    "#fbb6ce",  # light magenta
    "#c3dafe",  # light indigo
    "#fed7aa",  # light orange
    "#bae6fd",  # light sky blue
    "#bbf7d0",  # light bright green
    "#e9d5ff",  # light purple
    "#fecdd3",  # light red-pink
    "#a7f3d0",  # light teal
    "#f9a8d4",  # light violet-magenta
    "#fde68a",  # light amber
    "#bfdbfe",  # light deep blue
] * 10

c_vals_d = [
    "#9b2c2c",  # dark red
    "#2c5282",  # dark blue
    "#276749",  # dark green
    "#975a16",  # dark golden
    "#97266d",  # dark magenta
    "#4338ca",  # dark indigo
    "#7b341e",  # dark orange
    "#0369a1",  # dark sky blue
    "#15803d",  # dark bright green
    "#6b21a8",  # dark purple
    "#9f1239",  # dark red-pink
    "#0f4c3a",  # dark teal
    "#702459",  # dark violet-magenta
    "#854d0e",  # dark amber
    "#1e3a8a",  # dark deep blue
] * 10


l_vals = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
markers_vals = ['o', 'v', '*', '+', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
linestyles = ["-", "--", "-."]
cs = "coolwarm"


def main(seed, feature): 
    """
    """
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / 1e9:.2f} GB")
    print(f"Available: {mem.available / 1e9:.2f} GB")
    print(f"Used: {mem.used / 1e9:.2f} GB")
    print(f"Percentage: {mem.percent}%")

    task = "everything"
    hidden = "300"
    batch = "128"
    accfeature = "+angle" 

    aname = f"{task}_seed{seed}_{feature}+hidden{hidden}+batch{batch}{accfeature}"
    out_path_name = "multiple_tasks/" + f"param_{aname}_result.npz"
    out_path = Path(out_path_name)

    size_bytes = out_path.stat().st_size  
    size_gb = size_bytes / 1024**3 
    print(f"{out_path} = {size_gb:.3f} GiB")

    with np.load(out_path_name, allow_pickle=True) as data:
        rules_epochs = data["rules_epochs"].item()
        hyp_dict = data["hyp_dict"].item()
        all_rules = data["all_rules"]
        test_task = data["test_task"]
        Ms_orig = data["Ms_orig"]
        hs = data["hs"]
        bs = data["bs"]
        xs = data["xs"]

    print(f"Ms_orig: {Ms_orig.shape}")
    print(f"hs: {hs.shape}")
    print(f"xs: {xs.shape}")
    print(f"bs: {bs.shape}")
    print(f"test_task: {test_task.shape}")
    print(f"all_rules: {all_rules}")

    out_param_path = "multiple_tasks/" + f"param_{aname}_param.json"
    out_param_path = Path(out_param_path)

    with out_param_path.open() as f: 
        raw_cfg_param = json.load(f)

    task_params, train_params, net_params = raw_cfg_param["task_params"], raw_cfg_param["train_params"], raw_cfg_param["net_params"]

    savefigure_name = f"{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}"
    savefigure_name_base = copy.deepcopy(savefigure_name)

    # %%
    # 2025-11-19: make sure the bias is only cell-dependent but not time- or trail-dependent
    ref = bs[0, 0, :]              
    same_per_k = np.all(bs == ref, axis=(0, 1))  
    all_k_constant = np.all(same_per_k)

    # %%
    add_safe_globals([np.core.multiarray._reconstruct])

    netpathname = "multiple_tasks/" + f"savednet_{aname}.pt"
    checkpoint = torch.load(netpathname, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    print(state_dict.keys())
    load_net_params = checkpoint["net_params"]
    print(load_net_params)

    # reload the network
    model = mpn.DeepMultiPlasticNet(load_net_params, verbose=False, forzihan=True)

    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
    print("missing:", missing)
    print("unexpected:", unexpected)
    del checkpoint  # free duplicated weights; state_dict already extracted above

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on: {device}")

    task_params_c, train_params_c, net_params_c = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )

    def shared_run(addtask): 
        """
        Trying to replicate Fig 3D-E in Driscoll etal 2024
        """
        assert addtask in ("dmcgo", "delaydm1", )
        
        task_params_dmcgo = copy.deepcopy(task_params_c)

        if addtask == "dmcgo":
            task_params_dmcgo["rules"] = ["dmcgo", "dmcnogo"]
        elif addtask == "delaydm1":
            task_params_dmcgo["rules"] = ["delaydm1", "delaydm2"]
            
        test_n_batch = 100

        task_params_dmcgo['hp']['batch_size_train'] = test_n_batch
        task_params_dmcgo["long_delay"] = "normal"
        
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(task_params_dmcgo, 
                                                                    test_n_batch, 
                                                                    rules=task_params_dmcgo['rules'], 
                                                                    mode_input="random", 
                                                                    device="cpu",
                                                                    verbose=True)
        test_input, test_output, _ = test_data

        print(f"test_input.shape: {test_input.shape}")

        test_task = helper.find_task(task_params_dmcgo, test_input.detach().cpu().numpy(), 0)
        test_task = [int(test_task_ - min(test_task)) for test_task_ in test_task]

        _, test_trials, _ = test_trials_extra

        stim1_choices = np.concatenate([test_trials[0].meta["stim1"], test_trials[1].meta["stim1"]])
        print(f"stim1_choices: {stim1_choices}")

        epochs = test_trials[0].epochs
        
        if addtask in ("dmcgo", "delaydm1", ):
            delay_period = epochs["delay1"]
        
        # is delay period start inclusive and end exclusive?
        print(f"delay_period: {delay_period}")

        assert len(test_task) == len(stim1_choices)

        net_out, _, db_test = model.iterate_sequence_batch(test_input.to(device), run_mode='track_states')

        if addtask in ("dmcgo", "delaydm1", ):
            hidden_test = db_test["hidden1"][:, delay_period[0]:delay_period[1]-1, :]
            em_test = (db_test["M1"][:, delay_period[0]:delay_period[1]-1, :, :] * state_dict["mp_layer1.W"]).cpu().numpy()
            m_test = db_test["M1"][:, delay_period[0]:delay_period[1]-1, :, :].cpu().numpy()

        print(f"hidden_test: {hidden_test.shape}; em_test: {em_test.shape}; m_test: {m_test.shape}")

        fig, axs = plt.subplots(5,2,figsize=(5*2,5*2))
        for i in range(5):
            for inp in range(test_input.shape[2]):
                axs[i,0].plot(test_input[i,:,inp].detach().cpu().numpy(), color=c_vals[inp], alpha=0.5)
            for inp in range(net_out.shape[2]):
                axs[i,1].plot(net_out[i,:,inp].detach().cpu().numpy(), color=c_vals[inp], alpha=0.5)
            for outp in range(test_output.shape[2]):
                axs[i,1].plot(test_output[i,:,outp].detach().cpu().numpy(), color=c_vals[outp], 
                            alpha=0.5, linestyle="--")
        fig.tight_layout()  
        fig.savefig(f"./multiple_tasks/{addtask}_{savefigure_name}.png", dpi=300)

        def combo_indices_16(A, B):
            A = np.asarray(A)
            B = np.asarray(B)
            if A.shape != B.shape:
                raise ValueError(f"Length mismatch: A{A.shape} vs B{B.shape}")

            out = {}
            for a in (0, 1):
                for b in range(8):
                    out[(a, b)] = np.flatnonzero((A == a) & (B == b)).tolist()
            return out

        idx_map = combo_indices_16(test_task, stim1_choices)

        stacked_inputs = [[], []]
        stacked_inputs_labels = [[], []]
        for i in range(2):
            for b in range(8):
                # add multiple trials of the same kind of stimulus
                trial_num = 5
                for trial_idx in range(trial_num): 
                    # for trials that are considered to have the same stim1 and same task
                    # their input should be identical (quantified via sum) up to the end 
                    # of the delay period; only check for dmcgo since for delaydm1, the magnitude
                    # might be different across batches even the stim identity is the same 
                    if addtask == "dmcgo":
                        input_check = []
                        for key in idx_map[(i,b)]:
                            input_check.append(torch.sum(test_input[key,:delay_period[1],:]).item())
                        input_check = np.array(input_check, dtype=float)
                        ok = np.all(np.isclose(input_check, input_check[0]))
                        print(f"input_check: {input_check}")
                        assert ok 
                    
                    idxs = idx_map[(i, b)][trial_idx]
                    test_input_chosen = test_input[idxs]
                    stacked_inputs[i].append(test_input_chosen)
                    # add the stim1 label (for this trial) as well
                    stacked_inputs_labels[i].append(b)
                
            xs = stacked_inputs[i]  
            stacked_inputs[i] = torch.stack(
                [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in xs],
                dim=0
            )
            
        assert stacked_inputs_labels[0] == stacked_inputs_labels[1], "The two conditions should have the same set of stimuli for interpolation to make sense"

        # common delay period PCA
        as_hidden_wantperiod = hidden_test.reshape(-1, hidden_test.shape[-1])
        pca_delay_h = PCA(n_components=3, random_state=np.random.randint(0, 10000), svd_solver="auto")
        pca_delay_h.fit(as_hidden_wantperiod)

        as_em_wantperiod = em_test.reshape(-1, em_test.shape[-1] * em_test.shape[-2])
        pca_delay_em = PCA(n_components=3, random_state=np.random.randint(0, 10000), svd_solver="auto")
        pca_delay_em.fit(as_em_wantperiod)

        as_m_wantperiod = m_test.reshape(-1, m_test.shape[-1] * m_test.shape[-2])
        pca_delay_m = PCA(n_components=3, random_state=np.random.randint(0, 10000), svd_solver="auto")
        pca_delay_m.fit(as_m_wantperiod)

        plot_all = [["hidden", pca_delay_h], ["e_modulation", pca_delay_em], ["m_modulation", pca_delay_m]]

        for plot_name, pca_delay in plot_all:
            # interpolate between the two conditions (stim1 and stim2) in the input space, 
            # and track how the fixed points evolve in the PCA space of the delay period activity
            alpha_lst = np.linspace(0,1,20)
            interpolate_inputs = [(1-a) * stacked_inputs[0] + a * stacked_inputs[1] for a in alpha_lst]
            fixed_points_all = []
            traj_all = []

            for idx, interpolate_input in enumerate(interpolate_inputs):
                _, _, db = model.iterate_sequence_batch(interpolate_input.to(device), run_mode="track_states", save_to_cpu=True, detach_saved=True)

                if plot_name == "hidden":
                    data = db["hidden1"]
                    as_flat = data.reshape(-1, data.shape[-1])
                elif plot_name == "e_modulation":
                    data = (db["M1"] * state_dict["mp_layer1.W"]).cpu().numpy()
                    as_flat = data.reshape(-1, data.shape[-1] * data.shape[-2])
                elif plot_name == "m_modulation":
                    data = db["M1"]
                    as_flat = data.reshape(-1, data.shape[-1] * data.shape[-2])
                del db  # free activation dict before next iteration

                hidden_tf = pca_delay.transform(as_flat)
                projected_data = hidden_tf.reshape(data.shape[0], data.shape[1], -1)
                del data, as_flat  # intermediates no longer needed

                if addtask in ("dmcgo", "delaydm1", ):
                    fixed_points_all.append(projected_data[:,delay_period[1]-1,:])

                if idx in (0, len(interpolate_inputs) - 1, ):
                    if addtask in ("dmcgo", "delaydm1", ):
                        traj_all.append(projected_data[:, delay_period[0]:delay_period[1], :])
                del projected_data
                
            del interpolate_inputs  # free the 20 interpolated input tensors
            fixed_points_all_arr = np.stack(fixed_points_all, axis=0)  # (n_alpha, n_stim, n_pc)
            print(f"fixed_points_all_arr: {fixed_points_all_arr.shape}")
            n_alpha, n_stim, n_pc = fixed_points_all_arr.shape

            figmap, axsmap = plt.subplots(1,3,figsize=(5*3,5))
            pcs = [[0,1],[0,2],[1,2]]

            for pc_idx, (pc_x, pc_y) in enumerate(pcs):
                for stim in range(n_stim):
                    traj_fp = fixed_points_all_arr[:, stim, :] 

                    axsmap[pc_idx].plot(traj_fp[:, pc_x], traj_fp[:, pc_y], "-o", 
                                        color=c_vals[stacked_inputs_labels[0][stim]], linewidth=1.5, markersize=4, 
                                        alpha=0.5, 
                                        label=f"stim {(stim // trial_num + 1)}" if stim % trial_num == 0 else None)  

                    axsmap[pc_idx].scatter(traj_fp[0, pc_x],  traj_fp[0, pc_y],  color=c_vals[stacked_inputs_labels[0][stim]], 
                                        marker="s", s=50, zorder=3)
                    axsmap[pc_idx].scatter(traj_fp[-1, pc_x], traj_fp[-1, pc_y], color=c_vals[stacked_inputs_labels[0][stim]], 
                                        marker="^", s=50, zorder=3)

                # plot the trajectory during the delay period for the first and last alpha (i.e. the two original conditions)
                delay_traj = False
                if delay_traj: 
                    for be in range(len(traj_all)):
                        proj = traj_all[be]
                        for stim in range(proj.shape[0]):
                            xs = proj[stim, :, pc_x]
                            ys = proj[stim, :, pc_y]
                            axsmap[pc_idx].plot(xs, ys, color=c_vals[stacked_inputs_labels[0][stim]], 
                                                alpha=0.3, linewidth=1.0, linestyle=l_vals[be+1])
                            axsmap[pc_idx].scatter(xs[0], ys[0], color=c_vals[stacked_inputs_labels[0][stim]], 
                                                marker="o", s=35, zorder=4, alpha=0.9)

                axsmap[pc_idx].set_xlabel(f"Memory State PC{pc_x+1}", fontsize=12)
                axsmap[pc_idx].set_ylabel(f"Memory State PC{pc_y+1}", fontsize=12)
                axsmap[pc_idx].legend(frameon=True)
                # axsmap[pc_idx].set_xscale("symlog", linthresh=1e-3)
                # axsmap[pc_idx].set_yscale("symlog", linthresh=1e-3)
                
            figmap.tight_layout()
            figmap.savefig(f"./multiple_tasks/{addtask}_fixed_points_{plot_name}_{savefigure_name}.png", dpi=300)

    # shared_run("delaydm1") 
    # shared_run("dmcgo")

    # analyze the fitted weight matrices; we focus on the first layer of modulation and the output layer, since they are more interpretable than the hidden layer
    output_W = state_dict["W_output"].cpu().numpy()
    input_W = state_dict["W_initial_linear.weight"].cpu().numpy()
    modulation_W = state_dict["mp_layer1.W"].cpu().numpy()

    def heatmap_with_top_left_marginals(
        M,
        ax,
        cmap="coolwarm",
        center=0,
        vmin=None,
        vmax=None,
        xlabel="",
        ylabel="",
        label_fs=15,
        tick_fs=10,
        marginal_frac=0.18,   # thickness of top/left strips (in ax coords)
        marginal_pad=0.03,    # gap between heatmap and strips (in ax coords)
        marginal_lw=1.2,
        cbar=True,
        square=False
    ):
        """
        Heatmap on `ax` + marginals:
        - top inset:  col-wise sum(abs(M)) (length = n_cols)
        - left inset: row-wise sum(abs(M)) (length = n_rows), plotted vertically

        Layout: col-sum at top, row-sum at left. No child dirs, no extra axes elsewhere.
        """
        M = np.asarray(M)
        n_rows, n_cols = M.shape

        if vmin is None:
            vmin = np.nanmin(M)
        if vmax is None:
            vmax = np.nanmax(M)

        # ---- main heatmap
        sns.heatmap(
            M,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            cbar=cbar,
            square=square,
        )
        ax.set_xlabel(xlabel, fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)

        # ---- marginals
        col_sum = np.nansum(np.abs(M), axis=0)  # (n_cols,)
        row_sum = np.nansum(np.abs(M), axis=1)  # (n_rows,)

        # inset axes positions are in the parent ax's coordinate system
        top_rect  = [0.0, 1.0 + marginal_pad, 1.0, marginal_frac]
        left_rect = [-(marginal_frac + marginal_pad), 0.0, marginal_frac, 1.0]

        ax_top  = ax.inset_axes(top_rect,  transform=ax.transAxes)
        ax_left = ax.inset_axes(left_rect, transform=ax.transAxes)

        # ---- top: col sums (aligned with heatmap columns)
        x = np.arange(n_cols)
        ax_top.plot(x, col_sum, lw=marginal_lw)
        ax_top.set_xlim(-0.5, n_cols - 0.5)
        ax_top.set_xticks([])
        ax_top.tick_params(axis="y", labelsize=tick_fs)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        # ---- left: row sums (vertical, aligned with heatmap rows)
        y = np.arange(n_rows)
        ax_left.plot(row_sum, y, lw=marginal_lw)  # x=row_sum, y=row index
        ax_left.set_ylim(n_rows - 0.5, -0.5)      # match heatmap's y-direction (top row at top)
        ax_left.set_yticks([])
        ax_left.tick_params(axis="x", labelsize=tick_fs)
        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)
        
        ax.set_xticks([]); ax.set_yticks([]) 

        return ax, ax_top, ax_left

    def plot_weight_triplet_with_top_left_marginals(
        output_W,
        input_W,
        modulation_W,
        aname,
        cs="coolwarm",
        save_dir="./multiple_tasks",
    ):
        figoutput, axsoutput = plt.subplots(1,1,figsize=(10, 12))

        heatmap_with_top_left_marginals(
            output_W,
            ax=axsoutput,
            cmap=cs,
            center=0,
            vmin=np.nanmin(output_W),
            vmax=np.nanmax(output_W),
            xlabel="Hidden 2 Index",
            ylabel="Output Index",
            square=False
        )
        
        figinput, axsinput = plt.subplots(1,1,figsize=(10, 12))
        heatmap_with_top_left_marginals(
            input_W.T,
            ax=axsinput,
            cmap=cs,
            center=0,
            vmin=np.nanmin(input_W),
            vmax=np.nanmax(input_W),
            xlabel="Hidden 1 Index",
            ylabel="Input Index",
            square=False
        )
        
        figmodulation, axsmodulation = plt.subplots(1,1,figsize=(10, 12))
        heatmap_with_top_left_marginals(
            modulation_W.T,
            ax=axsmodulation,
            cmap=cs,
            center=0,
            vmin=np.nanmin(modulation_W),
            vmax=np.nanmax(modulation_W),
            xlabel="Hidden 1 Index",
            ylabel="Hidden 2 Index",
            square=False
        )

        figoutput.tight_layout()
        figoutput.savefig(f"{save_dir}/weight_matrices_{aname}.png", dpi=300)
        plt.close(figoutput)
        
        figinput.tight_layout()
        figinput.savefig(f"{save_dir}/weight_matrices_input_{aname}.png", dpi=300)
        plt.close(figinput)
        
        figmodulation.tight_layout()
        figmodulation.savefig(f"{save_dir}/weight_matrices_modulation_{aname}.png", dpi=300)
        plt.close(figmodulation)
        
        return

    def plot_static_pathway_analysis(
        output_W,
        input_W,
        modulation_W,
        input_labels,
        aname,
        save_dir="./multiple_tasks",
    ):
        """End-to-end pathway W_out @ W_mod @ W_in and its SVD spectrum."""
        W_path = output_W @ modulation_W @ input_W  # (n_output, n_input)
        _, s, _ = np.linalg.svd(W_path, full_matrices=False)

        n_output, n_input = W_path.shape
        fig, axs = plt.subplots(
            1, 2,
            figsize=(max(8, n_input * 0.35 + 4), max(4, n_output * 0.45 + 2)),
            gridspec_kw={"width_ratios": [3, 1]},
        )

        vabs = float(np.nanmax(np.abs(W_path)))
        sns.heatmap(
            W_path,
            ax=axs[0],
            cmap="coolwarm",
            center=0,
            vmin=-vabs,
            vmax=vabs,
            cbar=True,
            xticklabels=input_labels,
            yticklabels=np.arange(n_output),
        )
        axs[0].set_xticklabels(input_labels, rotation=90, fontsize=7)
        axs[0].set_xlabel("Input Feature", fontsize=10)
        axs[0].set_ylabel("Output Dim", fontsize=10)
        axs[0].set_title("End-to-end pathway  W_out @ W_mod @ W_in", fontsize=9)

        axs[1].bar(np.arange(len(s)), s, color="steelblue")
        axs[1].set_xlabel("Mode", fontsize=10)
        axs[1].set_ylabel("Singular value", fontsize=10)
        axs[1].set_title("SV spectrum", fontsize=9)
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(f"{save_dir}/pathway_static_{aname}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    plot_weight_triplet_with_top_left_marginals(
        output_W=output_W,
        input_W=input_W,
        modulation_W=modulation_W,
        aname=aname,
        cs=cs,
        save_dir="./multiple_tasks",
    )

    all_input = ["Fix On", "Fix Off", "Stim 1 Cos", "Stim 1 Sin", "Stim 2 Cos", "Stim 2 Sin"] + task_params_c["rules"]
    input_corr = np.corrcoef(input_W.T)
    mask = np.triu(np.ones_like(input_corr, dtype=bool), k=1)
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    sns.heatmap(input_corr, ax=ax, cmap=cs, center=0, mask=mask, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.5})
    ax.set_xticks(np.arange(len(all_input)) + 0.5, labels=all_input, rotation=90)
    ax.set_yticks(np.arange(len(all_input)) + 0.5, labels=all_input, rotation=0)
    ax.set_xlabel("Input Index", fontsize=12)
    ax.set_ylabel("Input Index", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"./multiple_tasks/input_weight_correlation_{aname}.png", dpi=300)
    plt.close(fig)

    plot_static_pathway_analysis(
        output_W=output_W,
        input_W=input_W,
        modulation_W=modulation_W,
        input_labels=all_input,
        aname=aname,
        save_dir="./multiple_tasks",
    )

    # should we re-evaluate the result? 
    reevaluate = True 
    if reevaluate:
        test_n_batch = 80 # number of batches for each task 
        task_params_c['hp']['batch_size_train'] = test_n_batch
            
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
            task_params_c, 
            test_n_batch, 
            rules=task_params_c['rules'],
            mode_input="random", 
            device="cpu", 
            verbose=False,
            long_all=True
        )
        
        test_input, test_output, test_mask = test_data
        print(f"test_input.shape: {test_input.shape}; test_output.shape: {test_output.shape}; test_mask.shape: {test_mask.shape}")
        _, test_trials, test_rule_idxs = test_trials_extra
        
        test_input = test_input.to(device)
        test_output = test_output.to(device)
        test_mask = test_mask.to(device)

        forward_chunk_size = 20  
        B_total = test_input.shape[0]
        chunk_net_outs, chunk_db = [], {}
        with torch.no_grad():
            for chunk_start in range(0, B_total, forward_chunk_size):
                chunk_end = min(chunk_start + forward_chunk_size, B_total)
                chunk_out, _, chunk_db_i = model.iterate_sequence_batch(
                    test_input[chunk_start:chunk_end], run_mode='track_states'
                )
                chunk_net_outs.append(chunk_out.cpu())
                for key, val in chunk_db_i.items():
                    if torch.is_tensor(val):
                        chunk_db.setdefault(key, []).append(val.cpu())
                    else:
                        chunk_db.setdefault(key, []).append(val)

        net_out = torch.cat(chunk_net_outs, dim=0)
        del chunk_net_outs  # free per-chunk output list now that net_out is assembled
        db_test = {
            key: torch.cat(chunks, dim=0) if torch.is_tensor(chunks[0]) else [x for c in chunks for x in c]
            for key, chunks in chunk_db.items()
        }
        del chunk_db  # free per-chunk state lists now that db_test is assembled
        acc, _ = model.compute_acc(net_out.to(device), test_output, test_mask, test_input, isvalid=True, mode=model.acc_measure)
        print(f"Accuracy: {acc}")
        del net_out, test_output, test_mask  # no longer needed after accuracy; test_input kept for find_task below

        Ms_orig = db_test["M1"].cpu().numpy()
        xs = db_test["input1"].cpu().numpy()
        hs = db_test["hidden1"].cpu().numpy()
        del db_test  # all needed arrays extracted; free the large activation tensors

        print(f"Ms_orig: {Ms_orig.shape}; xs: {xs.shape}; hs: {hs.shape}")
        print(f"modulation_W: {modulation_W.shape}")
        
        all_rules = np.array(task_params["rules"])
        
        def generate_response_stimulus(task_params, test_trials): 
            """
            """
            labels_resp, labels_stim1, labels_stim2 = [], [], []
            rules_epochs = {} 
            for rule_idx, rule in enumerate(task_params['rules']):
                rules_epochs[rule] = test_trials[rule_idx].epochs
                
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
        
        _, labels_stim1, _, rules_epochs = generate_response_stimulus(task_params, test_trials)
        
        test_task = helper.find_task(task_params, test_input.detach().cpu().numpy(), 0)
        test_task = np.array([int(c) for c in test_task]).flatten()

    # %%
    weighted_Ms_orig = Ms_orig * modulation_W
    print(f"weighted_Ms_orig: {weighted_Ms_orig.shape}")

    # 2026-03-29: for the modulation-weighted activity, we will not normalize across rules, 
    # otherwise per-neuron normalization
    # will wash out the multiplication effect
    # 2026-04-06: downstream analysis implicitly require the order of
    clustering_data_analysis = [xs, xs, hs, hs, Ms_orig, Ms_orig, weighted_Ms_orig, Ms_orig]
    clustering_data_analysis_names = ["input", "input", "hidden", "hidden", \
        "modulation_all", "modulation_all", "modulation_all_weighted", "modulation_all_var_weighted"]
    clustering_data_normalize = [True, False, True, False, True, False, False, False]
    c_metrics = ["euclidean", "euclidean", "euclidean", "euclidean", "euclidean", "euclidean", "euclidean", "euclidean"]
    c_methods = ["ward", "ward", "ward", "ward", "ward", "ward", "ward", "ward"]
    assert len(clustering_data_analysis) == len(clustering_data_analysis_names) \
        == len(clustering_data_normalize)

    # data registertion buffer
    clustering_data_hierarchy = {}
    clustering_corr_info = {}
    col_clusters_all, row_clusters_all = {}, {}
    row_cluster_breaker_all = {}
    input_hidden_comparison = {}
    base_data = {}
    metrics_all_all = {}
    rbreaks_all, cbreaks_all = {}, {}
    cluster_info_save = {}
    cluster_info_save_mod = {}

    selection_key = ["CH_blocks", "DB_blocks"]

    upper_cluster = 300
    lower_cluster = 5 # for input & hidden
    lower_cluster_mod = 10 # for modulation
    silhouette_tol = 0.05
    tol_mode = "relative"   # or "relative"
    tol_k_select = "gap"
    score_quantile = 0.50
    unresponsive_norm_frac = 5e-3

    assert tol_mode in ("absolute", "relative"), f"Invalid tol_mode: {tol_mode}"

    for clustering_index in range(len(clustering_data_analysis)): 
        print("======================================================")
        clustering_data = clustering_data_analysis[clustering_index]
        clustering_name = clustering_data_analysis_names[clustering_index]
        clustering_normalize = clustering_data_normalize[clustering_index]
        # 2026-04-09: only used for the modulation-related clustering analysis
        c_metric = c_metrics[clustering_index]
        c_method = c_methods[clustering_index]
        
        print(
            f"[clustering]\n"
            f"  name      : {clustering_name}\n"
            f"  data shape: {clustering_data.shape}\n"
            f"  normalize : {clustering_normalize}"
        )
        
        if hyp_dict['ruleset'] == "everything": 
            phase_to_indices = [
                ("stim1",  [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                ("stim2",  [6, 7, 8, 9, 10]),
                ("delay1", [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                ("delay2", [6, 7, 8, 9, 10]),
                ("go1",    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            ]
        
        tb_break = [
            [idx, rules_epochs[all_rules[idx]][phase]]
            for phase, indices in phase_to_indices
            for idx in indices
        ]
        
        tb_break_name = [
            f"{all_rules[idx]}-{phase}"
            for phase, indices in phase_to_indices
            for idx in indices
        ]
        
        tb_break_name = np.array(tb_break_name)
        
        cell_vars_rules = [] 
        
        for el in range(len(tb_break)):
            n_rules = len(task_params['rules'])
            n_cells = clustering_data.shape[-1]
                
            rule_idx, period_time = tb_break[el][0], tb_break[el][1]
            
            # print('Rule {} (idx {}), {}'.format(all_rules[rule_idx], rule_idx, period_time))
            # print(f"tb_break_name[el]: {tb_break_name[el]}")
            
            # 2026-03-29: select the stimulus information
            labels_stim = labels_stim1
            # labels_stim = labels_stim2 if ("stim2" in tb_break_name[el] or "delay2" in tb_break_name[el]) else labels_stim1
            
            if len(clustering_data.shape) == 3: # input or hidden 
                # 2025-11-19: if period_time[1] is None, then go until the end along that axis
                rule_cluster = clustering_data[test_task == rule_idx, period_time[0]:period_time[1], :]
                varval = helper.task_variance_period_numpy(rule_cluster, labels_stim[test_task == rule_idx].flatten())
                cell_vars_rules.append(varval) 
            else:
                clustering_data_old = clustering_data                
                if "all" in clustering_name: # modulation 
                    clustering_data = clustering_data.reshape(clustering_data.shape[0], clustering_data.shape[1], -1)
                    rule_cluster = clustering_data[test_task == rule_idx, 
                                                   period_time[0]:period_time[1], :]
                    varval = helper.task_variance_period_numpy(rule_cluster, 
                                                               labels_stim[test_task == rule_idx].flatten())
                    cell_vars_rules.append(varval) 
                        
        cell_vars_rules = np.array(cell_vars_rules)    
        cell_vars_rules_norm = np.zeros_like(cell_vars_rules)

        print(f"cell_vars_rules.shape: {cell_vars_rules.shape}")
        
        # normalize for each neuron 
        if clustering_normalize: 
            cell_max_var = np.max(cell_vars_rules, axis=0) # Across rules
            print(f"cell_max_var.shape: {cell_max_var.shape}")

            for period_idx in range(len(tb_break)):
                cell_vars_rules_norm[period_idx] = np.where(
                    cell_max_var > 0., cell_vars_rules[period_idx] / cell_max_var, 0.
                )
                
            savefigure_name = savefigure_name_base + "_normalized"
            clustering_save_name = clustering_name + "_normalized"
            # with normalization, then when plotting the variance matrix, the entries 
            # are all between 0 and 1, so we can set a fixed color range for better comparability across periods
            vmins, vmaxs = 0, 1
        else:
            cell_vars_rules_norm = cell_vars_rules.copy()
            savefigure_name = savefigure_name_base + "_unnormalized"
            clustering_save_name = clustering_name + "_unnormalized"
            vmins, vmaxs = None, None

        # for var_weighted: compute variance on Ms_orig then weight by modulation_W
        if "var_weighted" in clustering_name:
            cell_vars_rules_norm = cell_vars_rules_norm * modulation_W.flatten()[np.newaxis, :]

        # modulation only, reshape to (N, pre, post) shape after calculating the variance
        # N here as the number of sessions after breakdown
        # 2026-04-06: the code implicitly requires the modulation to be a square matrix
        if "all" in clustering_name: 
            N, MM = cell_vars_rules_norm.shape
            M = int(np.sqrt(MM))
            cell_vars_rules_norm_keepshape = cell_vars_rules_norm.reshape(N, M, M)
        
        # build rule-wise value lists and corresponding field names dynamically
        rule_vals  = [cell_vars_rules_norm[i].tolist() for i in range(n_rules)]
        # print(f"rule_vals: {rule_vals}")
        rule_names = [f"rule{i}" for i in range(n_rules)]
        
        # structured array whose fields are rule0, rule1, …, rule{n_rules-1}
        dtype = np.dtype([(name, float) for name in rule_names])
        rules_struct = np.array(list(zip(*rule_vals)), dtype=dtype)
        
        ## 2026-04-09: this part is important and should be treated delicately
        ## currently no sorting is applied, i.e. we keep the original order of the neurons; 
        # this is for better interpretability and also to avoid potential 
        # undesirable bugs in the clustering code that might be triggered by a non-identity sorting
        
        # sort_idxs = np.argsort(rules_struct, order=rule_names)[::-1] # descending lexicographic sort across all rule columns 
        sort_idxs = np.arange(rules_struct.shape[0], dtype=np.intp) # identity map for better interpretability

        # 2025-07-07: first sorting based on the normalized magnitude
        # all the following should be aligned with this change
        # 2025-08-22: sort based on the variance ordering OR using an identity map (i.e. do nothing)
        cell_vars_rules_sorted_norm = cell_vars_rules_norm[:, sort_idxs]
        base_data[clustering_save_name] = cell_vars_rules_sorted_norm
        print(f"cell_vars_rules_sorted_norm: {cell_vars_rules_sorted_norm.shape}")  

        # 2026-04-06: analyze input and hidden
        if not ("all" in clustering_name):
            # For unnormalized variance data (all non-negative), apply log1p to compress
            # the dynamic range before clustering so that cosine distance captures
            # selectivity profile shape rather than raw amplitude differences.
            # Normalized data is passed through unchanged.
            if clustering_normalize:
                V_for_clustering = cell_vars_rules_sorted_norm
            else:
                V_for_clustering = np.log1p(cell_vars_rules_sorted_norm)

            # clustering & grouping & re-ordering
            # first loop on input, second loop in hidden
            jitter_std = 0.02 
            resample_features_frac = 1.0 
            n_repeats = 100
            if jitter_std == 0 and resample_features_frac == 1.0:
                n_repeats = 1  # no need to repeat if no randomness is introduced
                
            result = clustering.cluster_variance_matrix_repeat(V_for_clustering,
                                                               k_min=lower_cluster,
                                                               k_max=upper_cluster,
                                                               metric=c_metric,
                                                               method=c_method,
                                                               n_repeats=n_repeats,
                                                               resample_features_frac=resample_features_frac,
                                                               jitter_std=jitter_std,
                                                               silhouette_tol=silhouette_tol,
                                                               tol_mode=tol_mode,
                                                               score_quantile=score_quantile,
                                                               unresponsive_norm_frac=unresponsive_norm_frac,
                                                               skip_unresponsive_detection=clustering_normalize,
                                                               tol_k_select=tol_k_select)
            
            # sanity check to make sure no undesirable bug happens with in the clustering code
            assert sorted(result["row_order"]) == list(range(cell_vars_rules_sorted_norm.shape[0]))
            assert sorted(result["col_order"]) == list(range(cell_vars_rules_sorted_norm.shape[1]))

            assert len(result["row_labels"]) == cell_vars_rules_sorted_norm.shape[0]
            assert len(result["col_labels"]) == cell_vars_rules_sorted_norm.shape[1]

            # Allow one extra label (k+1) for the unresponsive cluster that
            # clustering.py appends when unresponsive rows are detected.
            assert np.unique(result["row_labels"]).size in (result["row_k"], result["row_k"] + 1)
            assert np.unique(result["col_labels"]).size in (result["col_k"], result["col_k"] + 1)
            
            eval_res = clustering_metric.evaluate_bicluster_clustering(
                cell_vars_rules_sorted_norm, row_labels=result["row_tol_labels"], col_labels=result["col_tol_labels"]
            )
            
            eval_metrics = eval_res["metrics"]
            eval_blocks = eval_res["blocks"]
            eval_stdmean = np.nanmedian(eval_blocks["std"] / eval_blocks["means"])

            eval_random_metrics_all = []
            eval_random_blocks_all = []
            for _ in range(1000):
                rng = np.random.default_rng(seed=np.random.randint(0, 10000))
                row_arr = rng.permutation(result["row_tol_labels"])
                col_arr = rng.permutation(result["col_tol_labels"])
                eval_res_random = clustering_metric.evaluate_bicluster_clustering(
                    cell_vars_rules_sorted_norm, row_labels=row_arr, col_labels=col_arr
                )
                eval_random_metrics_all.append(eval_res_random["metrics"])
                eval_random_blocks_all.append(eval_res_random["blocks"])

            eval_random_stdmean = [np.nanmedian(eval_random_blocks["std"] / eval_random_blocks["means"]) 
                                   for eval_random_blocks in eval_random_blocks_all]

            metrics_all = {}
            for metric_key in selection_key: 
                optimized_value = eval_metrics[metric_key]
                random_values = [eval_random_metrics[metric_key] 
                                for eval_random_metrics in eval_random_metrics_all]
                metrics_all[metric_key] = [optimized_value, np.mean(random_values), 
                                           np.std(random_values, ddof=1)/np.sqrt(len(random_values))]

            metrics_all["std/mean"] = [eval_stdmean, 
                                       np.mean(eval_random_stdmean), 
                                       np.std(eval_random_stdmean, ddof=1)/np.sqrt(len(eval_random_stdmean))]
            
            # registeration
            metrics_all_all[clustering_save_name] = metrics_all

            input_hidden_comparison[clustering_save_name] = [result, cell_vars_rules_sorted_norm]
            
            # reorder the original matrix based on the clustering result
            cell_vars_rules_sorted_norm_ordered = cell_vars_rules_sorted_norm[
                np.ix_(result["row_order"], result["col_order"])]
            
            rl = np.asarray(result["row_tol_labels"])[result["row_order"]]
            cl = np.asarray(result["col_tol_labels"])[result["col_order"]]
            rbreaks = clustering._breaks(rl)
            cbreaks = clustering._breaks(cl)
            rbreaks_all[clustering_save_name] = rbreaks
            cbreaks_all[clustering_save_name] = cbreaks

            # Print k decisions across all selection strategies for comparison.
            row_sil_mean = result["row_score_recording_mean"]
            col_sil_mean = result["col_score_recording_mean"]
            row_ari_mean = result["row_ari_recording_mean"]
            col_ari_mean = result["col_ari_recording_mean"]

            row_strict_k = result["row_strict_best_k"]
            col_strict_k = result["col_strict_best_k"]
            row_strict_sil = row_sil_mean.get(row_strict_k, float("nan"))
            col_strict_sil = col_sil_mean.get(col_strict_k, float("nan"))

            # Recompute tolerance band (mirrors clustering._score_threshold_from_best)
            def _tol_thresh(score):
                if tol_mode == "relative":
                    return score * (1.0 - silhouette_tol)
                return score - silhouette_tol

            row_band = sorted(
                k for k in result["row_k_values"]
                if row_sil_mean.get(k, -np.inf) >= _tol_thresh(row_strict_sil)
            )
            col_band = sorted(
                k for k in result["col_k_values"]
                if col_sil_mean.get(k, -np.inf) >= _tol_thresh(col_strict_sil)
            )

            def _band_select(band, strategy, fallback):
                if not band:
                    return fallback
                if strategy == "min":
                    return min(band)
                if strategy == "max":
                    return max(band)
                # "mean": k closest to the arithmetic mean; ties broken by smaller k
                m = float(np.mean(band))
                return min(band, key=lambda k: (abs(k - m), k))

            strat_ks = {
                "min":  (_band_select(row_band, "min",  row_strict_k), _band_select(col_band, "min",  col_strict_k)),
                "max":  (_band_select(row_band, "max",  row_strict_k), _band_select(col_band, "max",  col_strict_k)),
                "mean": (_band_select(row_band, "mean", row_strict_k), _band_select(col_band, "mean", col_strict_k)),
                "ari":  (result["row_ari_tol_k"],  result["col_ari_tol_k"]),
                "gap":  (result["row_gap_tol_k"],  result["col_gap_tol_k"]),
            }

            print(
                f"  strict argmax — row: {row_strict_k} (sil={row_strict_sil:.4f}); "
                f"col: {col_strict_k} (sil={col_strict_sil:.4f})"
            )
            for strat, (rk, ck) in strat_ks.items():
                marker = "  <-- selected" if strat == tol_k_select else ""
                r_sil = row_sil_mean.get(rk, float("nan"))
                c_sil = col_sil_mean.get(ck, float("nan"))
                r_ari = row_ari_mean.get(rk, float("nan"))
                c_ari = col_ari_mean.get(ck, float("nan"))
                print(
                    f"  {strat:4s}: row={rk:3d} (sil={r_sil:.4f}, ari={r_ari:.4f}); "
                    f"col={ck:3d} (sil={c_sil:.4f}, ari={c_ari:.4f}){marker}"
                )
            
            # extract the grouping information, i.e. which neuron belong to which cluster
            # instead of the view of dendrogram
            # 2025-10-20: we register the tolerant version of optimal cluster selection
            col_labels, col_k = result["col_tol_labels"], result["col_tol_k"]
            # group the neuron based on the labels
            col_clusters = {int(lab): np.where(col_labels == lab)[0] for lab in np.unique(col_labels)}
            # 2025-11-04: do similar things for row separation
            # we checked so that the cluster label (name) is monotonically increasing 
            # i.e. near by cluster (e.g. cluster 1 and cluster 2) should be more similiar in population 
            # as well, since increasing the cutoff threshold in dendrogram will "merge" these clusters
            row_labels, row_k = result["row_tol_labels"], result["row_tol_k"]
            row_clusters = {int(lab): np.where(row_labels == lab)[0] for lab in np.unique(row_labels)}
            print(f"col_clusters: {len(col_clusters)}; row_clusters: {len(row_clusters)}")
            
            # registeration
            col_clusters_all[clustering_save_name] = col_clusters
            row_clusters_all[clustering_save_name] = row_clusters
            
            cluster_info_save[clustering_save_name] = {
                "col_clusters": col_clusters,
                "row_clusters": row_clusters,
                "tb_break_name": tb_break_name,
                "cell_vars_rules_sorted_norm": cell_vars_rules_sorted_norm,
                "result": result,
            }
            
            # plot the optimization score as a function of number of clustering
            # also plot the indicator for the optimal number of cluster (and with tolerance version)
            # 2026-04-16: add more details in the plotting

            def _gap_curve(Z, k_vals):
                """Compute the Ward merge-height gap for each k from a linkage matrix."""
                n = Z.shape[0] + 1
                gaps = []
                for k in k_vals:
                    idx_above = n - k
                    idx_below = n - k - 1
                    h_above = float(Z[idx_above, 2]) if 0 <= idx_above < Z.shape[0] else np.inf
                    h_below = float(Z[idx_below, 2]) if 0 <= idx_below < Z.shape[0] else 0.0
                    gaps.append(h_above - h_below)
                return np.asarray(gaps, dtype=float)

            # solid vline = strict silhouette argmax; dashed vline = tol_k_select decision
            row_decision_k, col_decision_k = strat_ks[tol_k_select]

            figscore, (axscore, axari, axgap) = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

            # ---------- silhouette: row ----------
            row_kvals   = np.asarray(result["row_k_values"], dtype=int)
            row_sil_all = np.asarray(result["row_score_recording_all"], dtype=float)  # (n_repeats, n_k)
            row_sil_arr = np.asarray([result["row_score_recording_mean"][k] for k in row_kvals], dtype=float)
            row_sil_std = np.asarray([result["row_score_recording_std"][k]  for k in row_kvals], dtype=float)

            for r in range(row_sil_all.shape[0]):
                axscore.plot(row_kvals, row_sil_all[r], color=c_vals[0], alpha=0.10, linewidth=0.8)
            axscore.fill_between(row_kvals, row_sil_arr - row_sil_std, row_sil_arr + row_sil_std,
                                 color=c_vals[0], alpha=0.20)
            axscore.plot(row_kvals, row_sil_arr, color=c_vals[0], linewidth=2.5, label="row")
            axscore.axvline(row_strict_k,   color=c_vals[0], linestyle="-",  linewidth=1.5)
            axscore.axvline(row_decision_k, color=c_vals[0], linestyle="--", linewidth=1.5)

            # ---------- silhouette: col ----------
            col_kvals   = np.asarray(result["col_k_values"], dtype=int)
            col_sil_all = np.asarray(result["col_score_recording_all"], dtype=float)  # (n_repeats, n_k)
            col_sil_arr = np.asarray([result["col_score_recording_mean"][k] for k in col_kvals], dtype=float)
            col_sil_std = np.asarray([result["col_score_recording_std"][k]  for k in col_kvals], dtype=float)

            for r in range(col_sil_all.shape[0]):
                axscore.plot(col_kvals, col_sil_all[r], color=c_vals[1], alpha=0.10, linewidth=0.8)
            axscore.fill_between(col_kvals, col_sil_arr - col_sil_std, col_sil_arr + col_sil_std,
                                 color=c_vals[1], alpha=0.20)
            axscore.plot(col_kvals, col_sil_arr, color=c_vals[1], linewidth=2.5, label="col")
            axscore.axvline(col_strict_k,   color=c_vals[1], linestyle="-",  linewidth=1.5)
            axscore.axvline(col_decision_k, color=c_vals[1], linestyle="--", linewidth=1.5)

            axscore.set_ylabel("Silhouette Score")
            axscore.legend(frameon=True)
            axscore.set_title(f"{clustering_name}", fontsize=15)

            # ---------- ARI stability: row ----------
            row_ari_arr = np.asarray([result["row_ari_recording_mean"][k] for k in row_kvals], dtype=float)
            row_ari_std = np.asarray([result["row_ari_recording_std"][k]  for k in row_kvals], dtype=float)
            axari.fill_between(row_kvals, row_ari_arr - row_ari_std, row_ari_arr + row_ari_std,
                               color=c_vals[0], alpha=0.20)
            axari.plot(row_kvals, row_ari_arr, color=c_vals[0], linewidth=2.5, label="row")
            axari.axvline(row_strict_k,   color=c_vals[0], linestyle="-",  linewidth=1.5)
            axari.axvline(row_decision_k, color=c_vals[0], linestyle="--", linewidth=1.5)

            # ---------- ARI stability: col ----------
            col_ari_arr = np.asarray([result["col_ari_recording_mean"][k] for k in col_kvals], dtype=float)
            col_ari_std = np.asarray([result["col_ari_recording_std"][k]  for k in col_kvals], dtype=float)
            axari.fill_between(col_kvals, col_ari_arr - col_ari_std, col_ari_arr + col_ari_std,
                               color=c_vals[1], alpha=0.20)
            axari.plot(col_kvals, col_ari_arr, color=c_vals[1], linewidth=2.5, label="col")
            axari.axvline(col_strict_k,   color=c_vals[1], linestyle="-",  linewidth=1.5)
            axari.axvline(col_decision_k, color=c_vals[1], linestyle="--", linewidth=1.5)

            axari.set_ylabel("Mean Pairwise ARI")
            axari.legend(frameon=True)

            # ---------- gap score: row ----------
            row_gap_arr = _gap_curve(result["row_linkage"], row_kvals)
            axgap.plot(row_kvals, row_gap_arr, color=c_vals[0], linewidth=2.5, label="row")
            axgap.axvline(row_strict_k,   color=c_vals[0], linestyle="-",  linewidth=1.5)
            axgap.axvline(row_decision_k, color=c_vals[0], linestyle="--", linewidth=1.5)

            # ---------- gap score: col ----------
            col_gap_arr = _gap_curve(result["col_linkage"], col_kvals)
            axgap.plot(col_kvals, col_gap_arr, color=c_vals[1], linewidth=2.5, label="col")
            axgap.axvline(col_strict_k,   color=c_vals[1], linestyle="-",  linewidth=1.5)
            axgap.axvline(col_decision_k, color=c_vals[1], linestyle="--", linewidth=1.5)

            axgap.set_xlabel("Number of Clusters")
            axgap.set_ylabel("Gap Score")
            axgap.set_xscale("log")
            axgap.legend(frameon=True)

            figscore.tight_layout()
            figscore.savefig(f"./multiple_tasks/{clustering_name}_variance_cluster_score_{savefigure_name}.png", dpi=300)
            plt.close(figscore)

            # 
            ordered_row_name = tb_break_name[result["row_order"]]
            row_breakers = [0]
            for row_group in row_clusters.values():
                row_breakers.append(row_breakers[-1] + len(row_group))
            row_breakers = row_breakers[1:]
            print(f"row_breakers: {row_breakers}")
            row_cluster_breaker_all[clustering_save_name] = row_breakers
            
            # pearson correlation matrix
            figcorr, axcorrs = plt.subplots(1,3,figsize=(8*3,8))
            metrics = ["Correlation", "Cosine Similarity", "L2 Distance"]
            cell_vars_rules_sorted_norm_ordered_measure = np.corrcoef(cell_vars_rules_sorted_norm_ordered, rowvar=True)
            cell_vars_rules_sorted_norm_ordered_measure_cos = cosine_similarity(cell_vars_rules_sorted_norm_ordered)
            cell_vars_rules_sorted_norm_ordered_measure_L2  = squareform(pdist(cell_vars_rules_sorted_norm_ordered, metric='euclidean'))

            # set uniform colorbar to cross-compare between analysis
            sns.heatmap(cell_vars_rules_sorted_norm_ordered_measure, cmap=cs, square=True, 
                        vmin=-0.5, vmax=1.0, ax=axcorrs[0])
            sns.heatmap(cell_vars_rules_sorted_norm_ordered_measure_cos, cmap=cs, square=True, 
                        vmin=-0.5, vmax=1.0, ax=axcorrs[1])
            sns.heatmap(cell_vars_rules_sorted_norm_ordered_measure_L2, cmap=cs, square=True, ax=axcorrs[2])

            for axcorr_index in range(len(axcorrs)): 
                axcorr = axcorrs[axcorr_index]
                # plot the group information (delimiter between different cluster)
                nn = cell_vars_rules_sorted_norm_ordered_measure.shape[0]
                boundaries = row_breakers
                for b in boundaries:
                    axcorr.axvline(b, 0, 1, color="k", linewidth=1.2)
                    axcorr.axhline(b, 0, 1, color="k", linewidth=1.2)
                
                axcorr.set_xticks(np.arange(len(tb_break_name)))
                axcorr.set_xticklabels(tb_break_name[result["row_order"]], rotation=45, ha='right', va='center', \
                                    rotation_mode='anchor', fontsize=9)    
                axcorr.set_yticks(np.arange(len(tb_break_name)))
                axcorr.set_yticklabels(tb_break_name[result["row_order"]], rotation=0, ha='right', va='center', \
                                    rotation_mode='anchor', fontsize=9) 
                axcorr.tick_params(axis="both", length=0)
                axcorr.set_title(f"{clustering_name}-{metrics[axcorr_index]}", fontsize=15)
            
            figcorr.tight_layout()
            figcorr.savefig(f"./multiple_tasks/{clustering_name}_variance_cluster_corr_{savefigure_name}.png", dpi=300)
            plt.close(figcorr)

            # register correlation information
            clustering_corr_info[clustering_save_name] = [
                cell_vars_rules_sorted_norm_ordered_measure, ordered_row_name, result["col_order"]
            ]

            # plot the effect of grouping & ordering through the feature axis
            # For unnormalized data use a log-scale colorbar spanning ~3 orders
            # of magnitude so that structure is visible despite large amplitude
            # differences across neurons. The data itself is NOT transformed.
            if clustering_normalize:
                heatmap_norm = None
                scale_label = ""
            else:
                data_max = np.max(cell_vars_rules_sorted_norm)
                data_min = max(np.min(cell_vars_rules_sorted_norm[cell_vars_rules_sorted_norm > 0]),
                               data_max / 1000.0)
                heatmap_norm = LogNorm(vmin=data_min, vmax=data_max)
                scale_label = " [log scale]"

            fig, ax = plt.subplots(2,1,figsize=(16,8*2))
            sns.heatmap(cell_vars_rules_sorted_norm, ax=ax[0], cmap=cs, cbar=True,
                        norm=heatmap_norm, vmin=vmins, vmax=vmaxs)
            sns.heatmap(cell_vars_rules_sorted_norm_ordered, ax=ax[1], cmap=cs, cbar=True,
                        norm=heatmap_norm, vmin=vmins, vmax=vmaxs)
            # Cluster boundaries: thick white line with a thin black outline for
            # visibility against any colormap value.
            for rb in rbreaks:
                ax[1].axhline(rb, color="w", lw=3.0, zorder=3)
                ax[1].axhline(rb, color="k", lw=0.8, zorder=4)
            for cb in cbreaks:
                ax[1].axvline(cb, color="w", lw=3.0, zorder=3)
                ax[1].axvline(cb, color="k", lw=0.8, zorder=4)
            ax[0].set_title(f"Before Clustering{scale_label}; best k row: {row_decision_k}; best k col: {col_decision_k}",
                            fontsize=15)
            ax[0].set_ylabel('Rule / Break-name', fontsize=12, labelpad=12)
            # Seaborn heatmap cells are centered at 0.5, 1.5, ...; use + 0.5 offset
            # so tick labels sit at the vertical center of each cell row.
            ax[0].set_yticks(np.arange(len(tb_break_name)) + 0.5)
            ax[0].set_yticklabels(tb_break_name, rotation=0, ha='right', va='center', fontsize=9)
            ax[1].set_title(f"After Clustering{scale_label}; best k row: {row_decision_k}; best k col: {col_decision_k}",
                            fontsize=15)
            ax[1].set_ylabel('Rule / Break-name', fontsize=12, labelpad=12)
            ax[1].set_yticks(np.arange(len(tb_break_name)) + 0.5)
            ax[1].set_yticklabels(tb_break_name[result["row_order"]], rotation=0, ha='right', va='center', fontsize=9)
            fig.savefig(f"./multiple_tasks/{clustering_name}_variance_cluster_{savefigure_name}.png", dpi=300)
            plt.close(fig)

            # 2025-11-04: for plotting purpose
            rbreaks_ = [0] + rbreaks + [cell_vars_rules_sorted_norm_ordered.shape[0]]
            cbreaks_ = [0] + cbreaks + [cell_vars_rules_sorted_norm_ordered.shape[1]]
            figvarmean, axsvarmean = plt.subplots(1,2,figsize=(4*2,4))
            varmean = np.zeros((len(rbreaks_) - 1, len(cbreaks_) - 1))
            for rr in range(len(rbreaks_) - 1):
                for cc in range(len(cbreaks_) - 1):
                    # 2025-11-04: mean covariance in this bicluster
                    varmean_ = np.mean(cell_vars_rules_sorted_norm_ordered[rbreaks_[rr]:rbreaks_[rr+1], 
                                                                           cbreaks_[cc]:cbreaks_[cc+1]])
                    varmean[rr,cc] = varmean_
            sns.heatmap(varmean, ax=axsvarmean[0], cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs)
            axsvarmean[0].set_ylabel("Session Clusters", fontsize=15)
            axsvarmean[0].set_xlabel("Neuron Clusters", fontsize=15)
            axsvarmean[0].set_title("Mean Variance in Each Bicluster", fontsize=15)
            
            # 2025-11-04: for sanity check; since the neuron clusters are ordered so that the adjacent ones 
            # are more similar to each other than the ones that are further, therefore the correlation matrix
            # should have larger value near the diagonal 
            varmeanC = np.corrcoef(varmean, rowvar=False) 
            sns.heatmap(varmeanC, ax=axsvarmean[1], cmap=cs, cbar=True, vmin=0, vmax=1)
            axsvarmean[1].set_xlabel("Neuron Clusters", fontsize=15)
            axsvarmean[1].set_ylabel("Neuron Clusters", fontsize=15)
            axsvarmean[1].set_title("Correlation Between Neuron Clusters", fontsize=15)
            figvarmean.tight_layout()
            figvarmean.savefig(f"./multiple_tasks/{clustering_name}_variance_cluster_mean_{savefigure_name}.png", dpi=300)
            plt.close(figvarmean)

            # plot the norm of normalized variance of each session (across the neuron dimension)
            session_norm = cell_vars_rules_sorted_norm_ordered.sum(axis=1)
            norm_order = np.argsort(-session_norm)
            session_norm = session_norm[norm_order]
            session_norm_name = (tb_break_name[result["row_order"]])[norm_order]
            
            fig, ax = plt.subplots(1,1,figsize=(15,5))
            ax.plot([i for i in range(len(session_norm))], session_norm, "-o")
            ax.set_xticks([i for i in range(len(session_norm))])
            ax.set_xticklabels(session_norm_name, rotation=45, ha="right")
            ax.set_ylabel("Summation of Normalized Variance", fontsize=15)
            fig.tight_layout()
            fig.savefig(f"./multiple_tasks/{clustering_name}_variance_norm_{savefigure_name}.png", dpi=300)
            plt.close(fig)
            
            # register hierarchy clustering
            clustering_data_hierarchy[clustering_name] = result["col_linkage"]

        # align the correlation matrix for input and hidden based on an identical ordering 
        # this loop will be run during the hidden analysis iteration (not modulation iteration)
        # this IF condition will work when hidden is calculated but not yet move to the modulation iteration
        # 2026-04-10: == 4 instead, since we have 4 iterations before modulation
        if (len(clustering_corr_info) == 4) and ("all" not in clustering_name):
            corr_comparison_pairs = [
                ("input_normalized",   "hidden_normalized",   "normalized"),
                ("input_unnormalized", "hidden_unnormalized", "unnormalized"),
            ]
            for inp_key, hid_key, norm_desc in corr_comparison_pairs:
                input_order_row  = clustering_corr_info[inp_key][1]
                hidden_order_row = clustering_corr_info[hid_key][1]
                input_corr       = clustering_corr_info[inp_key][0]
                hidden_corr      = clustering_corr_info[hid_key][0]

                # reorder hidden rows/cols to match input's task-condition ordering
                shuffle_hidden_to_input = helper.permutation_indices_b_to_a(
                    input_order_row, hidden_order_row
                )
                hidden_corr_reordered = hidden_corr[
                    np.ix_(shuffle_hidden_to_input, shuffle_hidden_to_input)
                ]

                figinputhiddencorr, axinputhiddencorr = plt.subplots(1, 2, figsize=(8*2, 8))
                sns.heatmap(input_corr,           ax=axinputhiddencorr[0], cmap=cs, square=True, vmin=-0.5, vmax=1.0)
                sns.heatmap(hidden_corr_reordered, ax=axinputhiddencorr[1], cmap=cs, square=True, vmin=-0.5, vmax=1.0)

                for ax in axinputhiddencorr:
                    ax.set_xticks(np.arange(len(input_order_row)) + 0.5)
                    ax.set_xticklabels(input_order_row, rotation=45, ha='right', va='center',
                                       rotation_mode='anchor', fontsize=9)
                    ax.set_yticks(np.arange(len(input_order_row)) + 0.5)
                    ax.set_yticklabels(input_order_row, rotation=0, ha='right', va='center',
                                       rotation_mode='anchor', fontsize=9)
                    ax.tick_params(axis="both", length=0)

                axinputhiddencorr[0].set_title(f"Input Correlation ({norm_desc})", fontsize=15)
                axinputhiddencorr[1].set_title(
                    f"Hidden Correlation — reordered by input ({norm_desc})", fontsize=15
                )
                # draw cluster breaks using input's break positions
                for ax in axinputhiddencorr:
                    for b in row_cluster_breaker_all[inp_key]:
                        ax.axvline(b, 0, 1, color="k", linewidth=1.2)
                        ax.axhline(b, 0, 1, color="k", linewidth=1.2)

                figinputhiddencorr.suptitle(f"Input vs Hidden Correlation ({norm_desc})")
                figinputhiddencorr.savefig(
                    f"./multiple_tasks/input2hidden_variance_hierarchy_{norm_desc}_{savefigure_name_base}.png",
                    dpi=300
                )
                plt.close(figinputhiddencorr)
        
        # 2025-11-04: check the consistency of row clusters between input & hidden
        if (len(clustering_corr_info) == 4) and ("all" not in clustering_name):
            # save the clustering information for input & hidden
            with open(f"./multiple_tasks/cluster_info_{savefigure_name_base}.pkl", "wb") as f:
                pickle.dump(cluster_info_save, f)
                
            print(cluster_info_save.keys())

            # ----------------------------------------------------------------
            # 2026-04-12: Input vs Hidden clustering comparison
            # Compare optimal-k row labels (task conditions) between input and
            # hidden for each normalisation variant (normalized / unnormalized).
            #
            # NOTE: col_labels (neurons/features) are NOT compared here because
            # input_dim != hidden_dim — those label vectors have different lengths
            # and index different feature spaces, so no meaningful comparison exists.
            # Row labels (task conditions) have the same length in both cases
            # (same tb_break) so ARI / NMI / contingency matrix are all valid.
            #
            # Metrics: Adjusted Rand Index (ARI) and Normalized Mutual Info (NMI)
            # — both permutation-invariant and handle different k.
            # Visual: contingency-matrix heatmap (counts + row-norm fractions).
            # ----------------------------------------------------------------
            comparison_pairs = [
                ("input_normalized",   "hidden_normalized",   "normalized"),
                ("input_unnormalized", "hidden_unnormalized", "unnormalized"),
            ]

            for input_key, hidden_key, norm_desc in comparison_pairs:
                if input_key not in cluster_info_save or hidden_key not in cluster_info_save:
                    continue

                r_input  = cluster_info_save[input_key]["result"]
                r_hidden = cluster_info_save[hidden_key]["result"]

                lbl_input  = np.asarray(r_input["row_tol_labels"])
                lbl_hidden = np.asarray(r_hidden["row_tol_labels"])
                k_input    = r_input["row_tol_k"]
                k_hidden   = r_hidden["row_tol_k"]

                assert len(lbl_input) == len(lbl_hidden), (
                    f"row_labels length mismatch: input={len(lbl_input)}, "
                    f"hidden={len(lbl_hidden)}"
                )

                ari = adjusted_rand_score(lbl_input, lbl_hidden)
                nmi = normalized_mutual_info_score(lbl_input, lbl_hidden)

                # contingency matrix: rows = input clusters, cols = hidden clusters
                cm      = sk_contingency_matrix(lbl_input, lbl_hidden)
                cm_frac = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                n_input_k, n_hidden_k = cm.shape

                fig_w = max(6, n_hidden_k * 0.7 + 2) * 2
                fig_h = max(4, n_input_k  * 0.6 + 2)
                fig, axes = plt.subplots(
                    1, 2, figsize=(fig_w, fig_h),
                    gridspec_kw={"wspace": 0.5},
                )

                # left panel: raw counts
                sns.heatmap(
                    cm, ax=axes[0], annot=True, fmt="d", cmap="Blues",
                    xticklabels=[f"H{i}" for i in range(n_hidden_k)],
                    yticklabels=[f"I{i}" for i in range(n_input_k)],
                    cbar=False,
                )
                axes[0].set_xlabel(f"Hidden clusters  (k={k_hidden})")
                axes[0].set_ylabel(f"Input clusters  (k={k_input})")
                axes[0].set_title("Count")

                # right panel: row-normalised fractions
                sns.heatmap(
                    cm_frac, ax=axes[1], annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f"H{i}" for i in range(n_hidden_k)],
                    yticklabels=[f"I{i}" for i in range(n_input_k)],
                    vmin=0, vmax=1, cbar=True,
                )
                axes[1].set_xlabel(f"Hidden clusters  (k={k_hidden})")
                axes[1].set_ylabel(f"Input clusters  (k={k_input})")
                axes[1].set_title("Row-normalised fraction")

                fig.suptitle(
                    f"Input vs Hidden — task conditions  ({norm_desc})  |  "
                    f"ARI = {ari:.3f}   NMI = {nmi:.3f}   "
                    f"(input k={k_input}, hidden k={k_hidden})",
                    fontsize=10,
                )
                out_path = (
                    f"./multiple_tasks/input_vs_hidden_row_"
                    f"{norm_desc}_{savefigure_name_base}.png"
                )
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(
                    f"[input vs hidden] {norm_desc} — task conditions: "
                    f"ARI={ari:.3f}  NMI={nmi:.3f}  "
                    f"(input k={k_input}  hidden k={k_hidden})"
                )
            
            ih_keys = ["input_normalized", "hidden_normalized"]
            belonging = np.zeros((2, len(tb_break_name)))
            for ttind in range(len(tb_break_name)):
                for rowind in range(2):
                    cluster_name = helper.find_key_by_membership(row_clusters_all[ih_keys[rowind]], ttind)
                    belonging[rowind, ttind] = cluster_name

            figbelonging, axbelonging = plt.subplots(1,1,figsize=(10,2))
            sns.heatmap(belonging, ax=axbelonging, cmap=cs, cbar=True, square=True)
            axbelonging.set_xticks([i for i in range(len(tb_break_name))])
            axbelonging.set_xticklabels(tb_break_name, rotation=45, ha='right', va='center', rotation_mode='anchor', fontsize=9)
            figbelonging.tight_layout()
            figbelonging.savefig(f"./multiple_tasks/inputhidden_row_membership_{savefigure_name}.png", dpi=300)
            plt.close(figbelonging)

            [result_input, cell_vars_rules_sorted_norm_input]  = input_hidden_comparison["input_normalized"]
            [result_hidden, cell_vars_rules_sorted_norm_hidden] = input_hidden_comparison["hidden_normalized"]
            cell_vars_rules_sorted_norm_ordered_input = cell_vars_rules_sorted_norm_input[np.ix_(result_input["row_order"], 
                                                                                                 result_input["col_order"])]
            cell_vars_rules_sorted_norm_ordered_hidden = cell_vars_rules_sorted_norm_hidden[np.ix_(result_input["row_order"], 
                                                                                                   result_hidden["col_order"])]
            
            figsame, axssame = plt.subplots(2,1,figsize=(24,8*2))
            sns.heatmap(cell_vars_rules_sorted_norm_ordered_input, ax=axssame[0], cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs)
            sns.heatmap(cell_vars_rules_sorted_norm_ordered_hidden, ax=axssame[1], cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs)
            for tem, tem_key in enumerate(["input_normalized", "hidden_normalized"]):
                # 2025-11-05: use input row breaks for both; col breaks per data type
                for rb in rbreaks_all["input_normalized"]:
                    axssame[tem].axhline(rb, color="k", lw=0.6)
                for cb in cbreaks_all[tem_key]:
                    axssame[tem].axvline(cb, color="k", lw=0.6)

            axssame[0].set_title("Input Using Input Session Break", fontsize=15)
            axssame[1].set_title("Hidden Using Input Session Break", fontsize=15)
            for axsind in range(2):
                axssame[axsind].set_xlabel("Hidden Neuron Index / Cluster", fontsize=15)
                axssame[axsind].set_ylabel("Input Neuron Index / Cluster", fontsize=15)
            figsame.tight_layout()
            figsame.savefig(f"./multiple_tasks/inputhidden_samerow_{savefigure_name}.png", dpi=300)
            plt.close(figsame)

            # 2025-11-18: whether to use the common session grouping based on input or hidden
            activecorr_lst = []
            for ih_index, ih_key in enumerate(["input_normalized", "hidden_normalized"]):
                common_rbreaks_ = [0] + rbreaks_all[ih_key]             + [cell_vars_rules_sorted_norm_input.shape[0]]
                cbreaks_input_  = [0] + cbreaks_all["input_normalized"]  + [cell_vars_rules_sorted_norm_input.shape[1]]
                cbreaks_hidden_ = [0] + cbreaks_all["hidden_normalized"] + [cell_vars_rules_sorted_norm_hidden.shape[1]]
        
                varmeaninput = np.zeros((len(common_rbreaks_) - 1, len(cbreaks_input_) - 1))
                varmeanhidden = np.zeros((len(common_rbreaks_) - 1, len(cbreaks_hidden_) - 1))
                for rr in range(len(common_rbreaks_) - 1):
                    for cc in range(len(cbreaks_input_) - 1):
                        varmeaninput[rr,cc] = np.mean(cell_vars_rules_sorted_norm_ordered_input[common_rbreaks_[rr]:common_rbreaks_[rr+1], 
                                                                                                cbreaks_input_[cc]:cbreaks_input_[cc+1]])
                    for cc2 in range(len(cbreaks_hidden_) - 1):
                        varmeanhidden[rr,cc2] = np.mean(cell_vars_rules_sorted_norm_ordered_hidden[common_rbreaks_[rr]:common_rbreaks_[rr+1], 
                                                                                                cbreaks_hidden_[cc2]:cbreaks_hidden_[cc2+1]])
        
                figmeanact, axsmeanact = plt.subplots(1,1,figsize=(10,4))
                varmeanconcatenate = np.concatenate((varmeaninput, varmeanhidden), axis=1)
                sns.heatmap(varmeanconcatenate, ax=axsmeanact, cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs, square=True)
                axsmeanact.axvline(len(cbreaks_input_), color="k", lw=0.6)
                axsmeanact.set_xlabel("Input / Hidden Cluster", fontsize=15)
                axsmeanact.set_ylabel("Common Session Cluster", fontsize=15)
                figmeanact.tight_layout()
                figmeanact.savefig(f"./multiple_tasks/inputhidden_aligned_activation_ih{ih_index}_{savefigure_name}.png", dpi=300)
                plt.close(figmeanact)

                figcoactive, axscoactive = plt.subplots(1,3,figsize=(4*3,4))
                s1, s2 = varmeaninput.shape[1], varmeanhidden.shape[1]
                activecorr, activeL1, activecosine = np.zeros((s1,s2)), np.zeros((s1,s2)), np.zeros((s1,s2))
                for inputind in range(varmeaninput.shape[1]):
                    for hiddenind in range(varmeanhidden.shape[1]):
                        activecorr[inputind, hiddenind] = np.corrcoef(varmeaninput[:,inputind], varmeanhidden[:,hiddenind])[0,1]
                        activeL1[inputind, hiddenind] = np.sum(np.abs(varmeaninput[:,inputind] - varmeanhidden[:,hiddenind]))
                        activecosine[inputind, hiddenind] = np.dot(
                            varmeaninput[:, inputind],
                            varmeanhidden[:, hiddenind]
                        ) / (
                            norm(varmeaninput[:, inputind]) * norm(varmeanhidden[:, hiddenind])
                        )
                        
                sns.heatmap(activecorr, ax=axscoactive[0], cmap=cs, cbar=True, square=True)
                activecorr_lst.append(activecorr)
                sns.heatmap(activeL1, ax=axscoactive[1], cmap=cs, cbar=True, square=True)
                sns.heatmap(activecosine, ax=axscoactive[2], cmap=cs, cbar=True, square=True)
                
                for axsind in range(2):
                    axscoactive[axsind].set_xlabel("Hidden Cluster", fontsize=15)
                    axscoactive[axsind].set_ylabel("Input Cluster", fontsize=15)
                    
                axscoactive[0].set_title("Pearson Correlation", fontsize=15)
                axscoactive[1].set_title("L1 Distance", fontsize=15)
                axscoactive[2].set_title("Cosine Similarity", fontsize=15)
                figcoactive.tight_layout()
                figcoactive.savefig(f"./multiple_tasks/inputhidden_coactivation_ih{ih_index}_{savefigure_name}.png", dpi=300)
                plt.close(figcoactive)

            # 2025-11-19: this result should be aligned by using / comparing the session clusters of 
            figcoactivecompare, axcoactivecompare = plt.subplots(1,1,figsize=(4,4))
            axcoactivecompare.scatter(activecorr_lst[0].flatten(), activecorr_lst[1].flatten(), c=c_vals[0], alpha=0.7)
            x_fit, y_fit, r_value, slope, _, p_val = helper.linear_regression(activecorr_lst[0].flatten(), 
                                                                              activecorr_lst[1].flatten(), 
                                                                              log=False, 
                                                                              through_origin=True)
            axcoactivecompare.plot(x_fit, y_fit, linestyle="--", label=f"R={r_value:.3f}; Slope: {slope:.3f}; p-value: {p_val:.3f}")
            axcoactivecompare.set_xlabel("Input Common Session Coactivation", fontsize=15)
            axcoactivecompare.set_ylabel("Hidden Common Session Coactivation", fontsize=15)
            axcoactivecompare.axhline(0, color="k", lw=0.6)
            axcoactivecompare.axvline(0, color="k", lw=0.6)
            axcoactivecompare.legend()
            figcoactivecompare.tight_layout()
            figcoactivecompare.savefig(f"./multiple_tasks/inputhidden_coactivation_ihcompare_{savefigure_name}.png", dpi=300)
            plt.close(figcoactivecompare)
            
        # use the clustering result for input and hidden to order the modulation information
        # and/or cross-compare the clustering result from input & hidden 
        # trying to observe consistency in between
        # 2026-04-06: this part of analysis is specifically targeting to the modulation-related data
        ih_mod_run = False # 2026-04-15: control this part is running or not
        if (len(clustering_corr_info) == 4) and ("all" in clustering_name) and ih_mod_run:
            # Apply log1p to unnormalized modulation data before clustering,
            # matching the same transform used for unnormalized input/hidden data.
            # Variance values are non-negative, so log1p is valid and compresses
            # the dynamic range so Euclidean distance captures pattern shape
            # rather than raw amplitude. Normalized data is passed unchanged.
            if clustering_normalize:
                V_for_clustering_mod = cell_vars_rules_sorted_norm
            else:
                V_for_clustering_mod = np.log1p(cell_vars_rules_sorted_norm)

            input_order_col_ind  = clustering_corr_info["input_normalized"][2]
            hidden_order_col_ind = clustering_corr_info["hidden_normalized"][2]
            # cell_vars_rules_norm_keepshape: 3D array
            # sort the modulation matrix based on the pre (input) and post (hidden) neuron ordering
            cell_vars_rules_norm_keepshape_ih = cell_vars_rules_norm_keepshape[:, input_order_col_ind, :]
            cell_vars_rules_norm_keepshape_ih = cell_vars_rules_norm_keepshape_ih[:, :, hidden_order_col_ind]

            flatten_by_pre = cell_vars_rules_norm_keepshape_ih.reshape(N, M*M)
            flatten_by_post = cell_vars_rules_norm_keepshape_ih.transpose(0,2,1).reshape(N, M*M)

            fig, axs = plt.subplots(2,1,figsize=(24,8*2))
            sns.heatmap(flatten_by_pre, ax=axs[0], cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs)
            sns.heatmap(flatten_by_post, ax=axs[1], cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs)
            axs[0].set_title("Flatten By Pre (Input)", fontsize=15)
            axs[1].set_title("Flatten By Post (Hidden)", fontsize=15)
            for ax in axs: 
                ax.set_yticks(np.arange(len(tb_break_name)))
                ax.set_yticklabels(tb_break_name, rotation=0, ha='right', va='center', fontsize=9)
            fig.tight_layout()
            fig.savefig(f"./multiple_tasks/modulation_input2hidden_variance_hierarchy_{savefigure_name}.png", dpi=300)
            plt.close(fig)

            #
            print(cell_vars_rules_sorted_norm.shape)
            if clustering_normalize:
                cluster_input  = col_clusters_all["input_normalized"]
                cluster_hidden = col_clusters_all["hidden_normalized"]
            else:
                cluster_input  = col_clusters_all["input_unnormalized"]
                cluster_hidden = col_clusters_all["hidden_unnormalized"]
                
            cluster_combine = {}
            newc = 0
            for c1 in cluster_input.keys(): # input cluster name
                for c2 in cluster_hidden.keys(): # hidden cluster name
                    newc += 1 # iterative counter of new cell type
                    # different newc stands for a unique cell type 
                    # (defined as a multiplication cluster of pre [input] and post [hidden])
                    nc1_input, nc2_hidden = cluster_input[c1], cluster_hidden[c2]
                    nc_combine = [(int(i),int(j)) for i in nc1_input for j in nc2_hidden]
                    # transform to reshaped matrix index
                    nc_combine = [cc[0] * M + cc[1] for cc in nc_combine]
                    cluster_combine[newc] = nc_combine

            # 2026-10-08: qualitatively, training MPN without regularization will cause more hidden cluster
            cluster_input_num, cluster_hidden_num = len(cluster_input), len(cluster_hidden)
            print(f"cluster_input_num: {cluster_input_num}; cluster_hidden_num: {cluster_hidden_num}")
            # by the setup, cluster_combine will be organized by shared pre
            # same pre neuron will be placed adjacently
            cluster_combine_pre = copy.deepcopy(cluster_combine)
            # create order of key to put post neuron together 
            post_order = []
            for c1 in range(cluster_input_num):
                for c2 in range(cluster_hidden_num):
                    post_order.append(c1 + c2 * cluster_input_num + 1) 

            pre_order = list(cluster_combine.keys())

            pre_order_group, post_order_group = [], [] 
            for ind in range(cluster_hidden_num):
                pre_order_group.append([i+1 for i in range(ind * cluster_input_num, (ind+1) * cluster_input_num)])
            for ind in range(cluster_input_num):
                block = []
                for ind2 in range(cluster_hidden_num):
                    block.append(ind + ind2 * cluster_input_num + 1)
                post_order_group.append(block)
            
            group_neurons_comb_pre, group_neurons_comb_post = [], [] 
            for preo in pre_order:
                group_neurons_comb_pre.extend(cluster_combine[preo])
            for posto in post_order:
                group_neurons_comb_post.extend(cluster_combine[posto])

            # 2025-09-02: we dont want random order of cell type
            # but based on certain pre or post neuron clustering 
            # random_celltype_orders = helper.concat_random_samples(cluster_combine, n_samples=1, seed=42)
            # random_celltype_orders = random_celltype_orders[0] # take the first sample
            # group_neurons_comb = random_celltype_orders[0] # concatenated list
            # assert len(group_neurons_comb) == len(set(group_neurons_comb)) # every element is unique
            # group_neurons = random_celltype_orders[1] # list of list, the order does not matter here

            group_neurons = [group for group in cluster_combine.values()]

            group_neurons_pre = []
            for pre_order_ in pre_order_group:
                block = []
                for ind in pre_order_:
                    block.extend(group_neurons[ind-1])
                group_neurons_pre.append(block)

            group_neurons_post = []
            for post_order_ in post_order_group:
                block = []
                for ind in post_order_:
                    block.extend(group_neurons[ind-1])
                group_neurons_post.append(block)

            # assert helper.all_leq(group_neurons_comb, cell_vars_rules_sorted_norm.shape[1])

            # cell_vars_rules_sorted_norm_r1 = cell_vars_rules_sorted_norm[:,group_neurons_comb]
            cell_vars_rules_sorted_norm_outer_bypre = cell_vars_rules_sorted_norm[:,group_neurons_comb_pre]
            cell_vars_rules_sorted_norm_outer_bypost = cell_vars_rules_sorted_norm[:,group_neurons_comb_post]

            fig, axs = plt.subplots(6,1,figsize=(24,8*6))
            sns.heatmap(cell_vars_rules_sorted_norm, ax=axs[0], cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs)
            sns.heatmap(cell_vars_rules_sorted_norm_outer_bypre, ax=axs[1], cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs)
            sns.heatmap(cell_vars_rules_sorted_norm_outer_bypost, ax=axs[2], cmap=cs, cbar=True, vmin=vmins, vmax=vmaxs)
            axs[0].set_title("Original", fontsize=15)
            axs[1].set_title(f"Ordering based on the Outer Product of Input & Hidden Clusters [Input Adjacent]", fontsize=15)
            axs[2].set_title(f"Ordering based on the Outer Product of Input & Hidden Clusters [Hidden Adjacent]", fontsize=15)
            for iii in range(3):
                axs[iii].set_yticks(np.arange(len(tb_break_name)))
                axs[iii].set_yticklabels(tb_break_name, rotation=0, ha='right', va='center', fontsize=9)

            group_all = [group_neurons, group_neurons_pre, group_neurons_post]
            group_all_names = ["Outer Product of Input & Hidden Clusters", "Shared Hidden", "Shared Input"]

            for group_neurons_index in range(len(group_all)):
                group_neurons_ = group_all[group_neurons_index]
                result_outer = clustering.cluster_variance_matrix_forgroup(V_for_clustering_mod,
                                                                           row_groups=None,
                                                                           col_groups_all_lst=[group_neurons_],
                                                                           k_min=lower_cluster,
                                                                           k_max=upper_cluster,
                                                                           silhouette_tol=silhouette_tol,
                                                                           tol_mode=tol_mode,
                                                                           metric=c_metric,
                                                                           score_quantile=score_quantile,
                                                                           unresponsive_norm_frac=unresponsive_norm_frac,
                                                                           tol_k_select=tol_k_select,
                                                                           skip_unresponsive_detection=clustering_normalize)
                prior_cluster_num = len(group_neurons_)
                cell_vars_rules_sorted_norm_inputhidden = cell_vars_rules_sorted_norm[np.ix_(result_outer["row_order"], 
                                                                                            result_outer["col_order"])]
        
                sns.heatmap(cell_vars_rules_sorted_norm_inputhidden, ax=axs[3+group_neurons_index], cmap=cs, cbar=True, 
                            vmin=vmins, vmax=vmaxs)
                axs[3+group_neurons_index].set_yticks(np.arange(len(tb_break_name)))
                axs[3+group_neurons_index].set_yticklabels(tb_break_name[result_outer["row_order"]], 
                                                           rotation=0, ha='right', va='center', fontsize=9)
                axs[3+group_neurons_index].set_title(f"Ordering based on the {group_all_names[group_neurons_index]}, #{prior_cluster_num} [Post-Clustering]", fontsize=15)
                
            fig.tight_layout()
            fig.savefig(f"./multiple_tasks/modulation_input2hiddentogether_variance_hierarchy_{savefigure_name}.png", dpi=300)
            plt.close(fig)
        
          
        if ("all" in clustering_name): 
            # ----------------------------------------------------------------
            # Modulation-specific clustering analysis
            # ----------------------------------------------------------------
            # The flattened modulation matrix (N_sessions x pre*post) is clustered
            # in three ways along the column (synapse) axis:
            #   1. "pre"  — columns grouped by shared presynaptic neuron
            #   2. "post" — columns grouped by shared postsynaptic neuron
            #   3. "all"  — columns grouped via MiniBatchKMeans with varying G
            # Row (session) clustering is always done without prior grouping.
            # For each scheme we obtain row/col orders, cluster labels, and
            # silhouette-based optimal k, then plot heatmaps and compute
            # pre/post membership statistics.
            # ----------------------------------------------------------------

            # Apply log1p to unnormalized modulation data before clustering,
            # matching the same transform used for unnormalized input/hidden data.
            # Normalized data is passed unchanged.
            if clustering_normalize:
                V_for_clustering_mod = cell_vars_rules_sorted_norm
            else:
                V_for_clustering_mod = np.log1p(cell_vars_rules_sorted_norm)

            # 2026-04-10: just a buffer to save the 4D shape of the modulation matrix
            assert len(clustering_data_old.shape) == 4
            pre_num, post_num = clustering_data_old.shape[2], clustering_data_old.shape[3]
            feature_group_post = [] 
            for i in range(post_num):
                feature_group_post.append(helper.basic_sort([i + j * post_num 
                                                             for j in range(pre_num)], sort_idxs))
            feature_group_pre = []
            for i in range(pre_num):
                feature_group_pre.append(helper.basic_sort([j for j in range(post_num * i, post_num * (i+1))], sort_idxs))

            result_pre = clustering.cluster_variance_matrix_forgroup(V_for_clustering_mod,
                                                                    row_groups=None,
                                                                    col_groups_all_lst=[feature_group_pre],
                                                                    k_min=lower_cluster,
                                                                    k_max=upper_cluster,
                                                                    silhouette_tol=silhouette_tol,
                                                                    tol_mode=tol_mode,
                                                                    metric=c_metric,
                                                                    score_quantile=score_quantile,
                                                                    unresponsive_norm_frac=unresponsive_norm_frac,
                                                                    tol_k_select=tol_k_select,
                                                                    skip_unresponsive_detection=clustering_normalize)

            result_post = clustering.cluster_variance_matrix_forgroup(V_for_clustering_mod,
                                                                    row_groups=None,
                                                                    col_groups_all_lst=[feature_group_post],
                                                                    k_min=lower_cluster,
                                                                    k_max=upper_cluster,
                                                                    silhouette_tol=silhouette_tol,
                                                                    tol_mode=tol_mode,
                                                                    metric=c_metric,
                                                                    score_quantile=score_quantile,
                                                                    unresponsive_norm_frac=unresponsive_norm_frac,
                                                                    tol_k_select=tol_k_select,
                                                                    skip_unresponsive_detection=clustering_normalize)

            for _lbl, _res in [("result_pre", result_pre), ("result_post", result_post)]:
                _row_k_strict = _res["row_strict_best_k"]
                _col_k_strict = _res["col_strict_best_k"]
                _row_k_tol    = _res["row_k"]
                _col_k_tol    = _res["col_k"]
                _row_sil_strict = _res["row_score_recording_mean"].get(_row_k_strict, float("nan"))
                _col_sil_strict = _res["col_score_recording_mean"].get(_col_k_strict, float("nan"))
                _row_sil_tol    = _res["row_score_recording_mean"].get(_row_k_tol, float("nan"))
                _col_sil_tol    = _res["col_score_recording_mean"].get(_col_k_tol, float("nan"))
                print(
                    f"[{_lbl}] row_k (strict): {_row_k_strict} "
                    f"(sil={_row_sil_strict:.4f}); "
                    f"col_k (strict): {_col_k_strict} "
                    f"(sil={_col_sil_strict:.4f})"
                )
                print(
                    f"[{_lbl}] row_k (tol):    {_row_k_tol} "
                    f"(sil={_row_sil_tol:.4f}); "
                    f"col_k (tol):    {_col_k_tol} "
                    f"(sil={_col_sil_tol:.4f})"
                )

            # 2025-10-06: do not give any prior grouping prior to the modulation information
            # simply grouping and considering each individual column as separate
            # having smaller G, e.g. 200, will make the following calculation in determining pre- and post-
            # belonging identity more time costly
            G_lst = [100, 300, 1000]
            figcol, axscol = plt.subplots(1,len(G_lst),figsize=(4*len(G_lst),4))
            figcolcluster, axscolcluster = plt.subplots(1,len(G_lst),figsize=(4*len(G_lst),4))
            figppshare, axsppshare = plt.subplots(2,len(G_lst),figsize=(4*len(G_lst),4*2))

            result_all_lst = []
            result_all_name_lst = []

            # 2025-11-04: input cluster & hidden cluster along the neuron dimension (N)
            if clustering_normalize:
                cluster_input  = col_clusters_all["input_normalized"]
                cluster_hidden = col_clusters_all["hidden_normalized"]
            else:
                cluster_input  = col_clusters_all["input_unnormalized"]
                cluster_hidden = col_clusters_all["hidden_unnormalized"]
                
            # sanity check: order it based on the key
            cluster_input = dict(sorted(cluster_input.items()))
            cluster_hidden = dict(sorted(cluster_hidden.items()))
            
            for G_idx in range(len(G_lst)): 
                G = G_lst[G_idx]
                assert G <= cell_vars_rules_sorted_norm.shape[0] * cell_vars_rules_sorted_norm.shape[1]
                
                col_groups_all_lst = []
                # 2025-10-06: since K-means has randomness, we run it multiple times to obtain 
                # the distribution of grouping statistics,
                krun = 10
                print(f"Running K-means with G={G} for {krun} times to obtain the distribution of grouping statistics...")
                for _ in range(krun):
                    random_seed = np.random.randint(0, 10000)
                    col_groups_all = clustering.make_col_groups_with_kmeans(V_for_clustering_mod,
                                                                            n_groups=G,
                                                                            random_state=random_seed)[0]
                    col_groups_all_lst.append(col_groups_all)
        
                # plot the statistics of grouping on all modulations without considering their pre/post identity 
                for gi in range(5): 
                    group_lengths = [len(sublist) for sublist in col_groups_all_lst[gi]]
                    axscol[G_idx].hist(group_lengths, color=c_vals[gi], bins=20, edgecolor='black', alpha=0.7)
                # if uniform grouping, what is the expected length for each group 
                axscol[G_idx].axvline(MM / G, linestyle="--", color="black")
                axscol[G_idx].set_xlabel('Length of Formed Group', fontsize=15)
                axscol[G_idx].set_ylabel('Frequency', fontsize=15)

                # Modulation analysis: varying G changes the column-side grouped representation,
                # because columns are first averaged within K-means groups before hierarchical clustering.
                # Note that the row-side data are unchanged here; only the searched k-range also changes
                # because we currently pass k_max=G to both axes.
                result_all = clustering.cluster_variance_matrix_forgroup(V_for_clustering_mod,
                                                                         row_groups=None,
                                                                         col_groups_all_lst=col_groups_all_lst,
                                                                         k_min=lower_cluster_mod,
                                                                         k_max=G,
                                                                         silhouette_tol=silhouette_tol,
                                                                         tol_mode=tol_mode,
                                                                         metric=c_metric,
                                                                         score_quantile=score_quantile,
                                                                         unresponsive_norm_frac=unresponsive_norm_frac,
                                                                         tol_k_select=tol_k_select,
                                                                         skip_unresponsive_detection=clustering_normalize)
        
                _r_sil      = result_all["row_score_recording_mean"]
                _c_sil      = result_all["col_score_recording_mean"]
                _r_strict_k = result_all["row_strict_best_k"]
                _c_strict_k = result_all["col_strict_best_k"]
                # Tolerance band stored directly in the result — no need to recompute.
                _r_band     = result_all["row_primary_candidates"]
                _c_band     = result_all["col_primary_candidates"]

                def _bsel(band, strategy, fallback, Z=None):
                    if not band:
                        return fallback
                    if strategy == "min":
                        return min(band)
                    if strategy == "max":
                        return max(band)
                    if strategy == "gap":
                        return clustering._gap_k(Z, band) if Z is not None else fallback
                    m = float(np.mean(band))
                    return min(band, key=lambda kk: (abs(kk - m), kk))

                _strat_ks = {
                    "min":  (_bsel(_r_band, "min",  _r_strict_k),
                             _bsel(_c_band, "min",  _c_strict_k)),
                    "max":  (_bsel(_r_band, "max",  _r_strict_k),
                             _bsel(_c_band, "max",  _c_strict_k)),
                    "mean": (_bsel(_r_band, "mean", _r_strict_k),
                             _bsel(_c_band, "mean", _c_strict_k)),
                    "gap":  (_bsel(_r_band, "gap",  _r_strict_k, result_all["row_linkage"]),
                             _bsel(_c_band, "gap",  _c_strict_k, result_all["col_linkage"])),
                }

                print(
                    f"[G={G}]   strict argmax — "
                    f"row: {_r_strict_k} (sil={_r_sil.get(_r_strict_k, float('nan')):.4f}); "
                    f"col: {_c_strict_k} (sil={_c_sil.get(_c_strict_k, float('nan')):.4f})"
                )
                for _strat, (_rk, _ck) in _strat_ks.items():
                    _marker = "  <-- selected" if _strat == tol_k_select else ""
                    print(
                        f"[G={G}]   {_strat:4s}: row={_rk:3d} (sil={_r_sil.get(_rk, float('nan')):.4f}); "
                        f"col={_ck:3d} (sil={_c_sil.get(_ck, float('nan')):.4f}){_marker}"
                    )
                assert result_all["col_k"] < G
                axscol[G_idx].set_title(f"G={G}, Neuron Cluster={result_all['col_k']}", fontsize=15)

                # Plot the size distribution of the actual modulation clusters
                # after cluster_variance_matrix_forgroup expands labels back to
                # the original modulation columns.
                actual_cluster_ids, actual_cluster_sizes = np.unique(
                    np.asarray(result_all["col_labels"]), return_counts=True
                )
                assert actual_cluster_ids.size in (result_all["col_k"], result_all["col_k"] + 1)
                axscolcluster[G_idx].hist(
                    actual_cluster_sizes,
                    color=c_vals[G_idx],
                    bins="auto",
                    edgecolor='black',
                    alpha=0.7
                )
                axscolcluster[G_idx].axvline(
                    MM / result_all["col_k"], linestyle="--", color="black"
                )
                axscolcluster[G_idx].set_xlabel('Length of Actual Cluster', fontsize=15)
                axscolcluster[G_idx].set_ylabel('Frequency', fontsize=15)
                axscolcluster[G_idx].set_title(
                    f"G={G}, Actual Cluster={result_all['col_k']}", fontsize=15
                )

                result_all_lst.append(result_all)
                result_all_name_lst.append(f"G={G}")
        
                # 2025-10-21: after grouping and clustering the modulation, 
                # we ask for two modulation within the same cluster, 
                # how likely they share the same presynaptic neuron (or neuron cluster), 
                # postsynaptic neuron (or neuron cluster), or neither, or both for neuron cluster 
                # it is not possible for different modulation sharing the same pre and post neuron,
                # but they may still share the same neuron cluster
                # here we are curious about their collective behavior and calculate the total summation of count
                row_all, col_all = result_all["row_labels"], result_all["col_labels"]
                assert np.max(col_all) in (result_all["col_k"], result_all["col_k"] + 1)
                
                print(f"col_all: {col_all}")

                # some helper function
                def find_key_by_element(d, element):
                    for key, values in d.items():
                        if element in values:
                            return key
                    return None
                    
                # Exclude the unresponsive modulation cluster (label = col_k + 1).
                # flat_idx preserves original positions so i//M and i%M stay correct.
                # Note: when clustering_normalize=True, skip_unresponsive_detection=True is
                # passed to cluster_variance_matrix_forgroup, so col_all only contains labels
                # 1..col_k and the mask is all-True (no-op). The exclusion is only meaningful
                # when clustering_normalize=False, where unresponsive modulations get label col_k+1.
                _unres_mod_label = result_all["col_k"] + 1
                _active_mod_mask = col_all != _unres_mod_label
                _col_all_active  = col_all[_active_mod_mask]
                _flat_idx_active = np.where(_active_mod_mask)[0]

                # Additionally exclude modulations whose pre-neuron (input) or post-neuron
                # (hidden) is in the unresponsive neuron cluster.  This gives "fully responsive
                # triplets": the modulation itself AND both its endpoints are task-sensitive.
                # Only meaningful when clustering_normalize=False (unresponsive detection was run
                # for input/hidden neurons).  When clustering_normalize=True this block is a no-op
                # because no col_k+1 label was assigned to input/hidden neurons.
                if not clustering_normalize:
                    _input_col_k  = cluster_info_save["input_unnormalized"]["result"]["col_tol_k"]
                    _hidden_col_k = cluster_info_save["hidden_unnormalized"]["result"]["col_tol_k"]
                    _unres_input_neurons  = set(cluster_input.get(_input_col_k  + 1, []))
                    _unres_hidden_neurons = set(cluster_hidden.get(_hidden_col_k + 1, []))

                    if _unres_input_neurons or _unres_hidden_neurons:
                        _pre_idx  = _flat_idx_active // M
                        _post_idx = _flat_idx_active % M
                        _endpoint_mask = (
                            ~np.isin(_pre_idx,  list(_unres_input_neurons)) &
                            ~np.isin(_post_idx, list(_unres_hidden_neurons))
                        )
                        _col_all_active  = _col_all_active[_endpoint_mask]
                        _flat_idx_active = _flat_idx_active[_endpoint_mask]
                        n_dropped = int((~_endpoint_mask).sum())
                        print(f"  Endpoint mask: dropped {n_dropped} modulations touching "
                              f"unresponsive pre/post neurons "
                              f"({len(_unres_input_neurons)} unresponsive input, "
                              f"{len(_unres_hidden_neurons)} unresponsive hidden neurons)")

                # 2025-10-21: membership using the actual clustering information
                same_pre_all, same_post_all, no_same_pre_post_all, same_pre_cluster_all, same_post_cluster_all, \
                    same_pre_post_cluster_all, no_same_pre_post_cluster_all = clustering_metric.count_pairs_with_clusters(
                        _col_all_active, M, cluster_input, cluster_hidden,
                        flat_idx=_flat_idx_active)
                # (control) membership using the random clustering information
                # 2026-03-03: increase the repeat times for a more reliable null distribution
                same_pre_all_c, same_post_all_c, no_same_pre_post_all_c, same_pre_cluster_all_c, same_post_cluster_all_c, \
                    same_pre_post_cluster_all_c, no_same_pre_post_cluster_all_c = clustering_metric.count_pairs_with_clusters_control(
                        _col_all_active, M, cluster_input, cluster_hidden,
                        repeat=10000, flat_idx=_flat_idx_active)
                
                print(f"same_pre_all: {same_pre_all}; same_post_all: {same_post_all}; no_same_pre_post_all: {no_same_pre_post_all}")
                print(f"same_pre_cluster_all: {same_pre_cluster_all}; same_post_cluster_all: {same_post_cluster_all}")
                print(f"same_pre_post_cluster_all: {same_pre_post_cluster_all}; no_same_pre_post_cluster_all: {no_same_pre_post_cluster_all}")

                print(f"same_pre_all_c: {same_pre_all_c}; same_post_all_c: {same_post_all_c}; no_same_pre_post_all_c: {no_same_pre_post_all_c}")
                print(f"same_pre_cluster_all_c: {same_pre_cluster_all_c}; same_post_cluster_all_c: {same_post_cluster_all_c}")
                print(f"same_pre_post_cluster_all_c: {same_pre_post_cluster_all_c}; no_same_pre_post_cluster_all_c: {no_same_pre_post_cluster_all_c}")
        
                bar_all_lst = [[same_pre_all, same_post_all, no_same_pre_post_all], 
                               [same_pre_cluster_all, same_post_cluster_all, 
                                same_pre_post_cluster_all, no_same_pre_post_cluster_all]]
                bar_all_ctrl_lst = [[same_pre_all_c, same_post_all_c, no_same_pre_post_all_c], 
                                    [same_pre_cluster_all_c, same_post_cluster_all_c, 
                                     same_pre_post_cluster_all_c, no_same_pre_post_cluster_all_c]]
                bar_name_lst = [["Share-Pre", "Share-Post", "Neither"], 
                                ["Share-Pre-Cluster", "Share-Post-Cluster", "Share-Both-Cluster", "Neither"]]

                N_cluster = np.max(col_all)
                
                ppshare_row_names = ["Same Neuron Check", "Same Neuron Cluster Check"]
                n_pre_clusters  = len(cluster_input)
                n_post_clusters = len(cluster_hidden)
                for idx in range(len(bar_all_lst)):
                    bar_all = np.array(bar_all_lst[idx])
                    bar_all_c = np.array(bar_all_ctrl_lst[idx])

                    over_membership = (bar_all - bar_all_c) / bar_all_c

                    axsppshare[idx,G_idx].bar([i for i in range(len(over_membership))], over_membership)
                    axsppshare[idx,G_idx].set_xticks([i for i in range(len(over_membership))])
                    axsppshare[idx,G_idx].set_xticklabels(bar_name_lst[idx], rotation=45, ha="right")
                    axsppshare[idx,G_idx].set_ylabel("Over-membership", fontsize=15)
                    if idx == 0:
                        title = f"{ppshare_row_names[idx]} | G={G}; #Cluster={N_cluster}"
                    else:
                        title = (f"{ppshare_row_names[idx]} | G={G}; #Cluster={N_cluster}\n"
                                 f"#PreCluster={n_pre_clusters}; #PostCluster={n_post_clusters}")
                    axsppshare[idx,G_idx].set_title(title)

                # 2025-10-21: next analyze for each individual modulation cluster, 
                # the belonging to different individual 
                # pre and post cluster; only test in the minimal modulation cluster selection case
                # 2025-11-17: revise to plot for G=300 case
                if G_idx == 1:
                    print(f"Plot for G={G_lst[G_idx]} Case")
                    mp = 5; cnt = 0
                    # Use active (responsive) modulations only, consistent with count_pairs_with_clusters.
                    # _col_all_active / _flat_idx_active are all-inclusive no-ops when
                    # clustering_normalize=True (unresponsive detection is skipped).
                    all_choice_order_dict = helper.value_counts_desc(_col_all_active)
                    all_choice_order = list(all_choice_order_dict.keys())

                    # cluster sizes come directly from value_counts_desc (no need to re-scan col_all)
                    cluster_size_all = np.array([all_choice_order_dict[k] for k in all_choice_order])
                    cluster_size_percent = cluster_size_all / cluster_size_all.sum()
                    assert np.isclose(cluster_size_percent.sum(), 1.0)

                    # all_num is the outer product of input and hidden cluster sizes
                    in_num = np.array([len(cluster_input[k]) for k in cluster_input])
                    hid_num = np.array([len(cluster_hidden[k]) for k in cluster_hidden])
                    all_num = np.outer(in_num, hid_num).astype(float)
                    flat_all_num = all_num.flatten()
                    n_in, n_hid = len(cluster_input), len(cluster_hidden)

                    figac, axac = plt.subplots(1,1,figsize=(4,4))
                    sns.heatmap(all_num, ax=axac, cmap=cs)
                    axac.set_xlabel("Hidden Cluster Index", fontsize=15)
                    axac.set_ylabel("Input Cluster Index", fontsize=15)
                    axac.set_title("Number of Total Modulation", fontsize=15)
                    figac.tight_layout()
                    figac.savefig(f"./multiple_tasks/{clustering_name}_inhidpair_num_{savefigure_name}.png", dpi=300)
                    plt.close(figac)

                    # Precompute neuron -> cluster index (0-based) lookup arrays once.
                    # Keys in cluster_input/cluster_hidden are 1-indexed, so subtract 1.
                    max_pre = max(max(v) for v in cluster_input.values())
                    max_post = max(max(v) for v in cluster_hidden.values())
                    pre_lookup = np.empty(max_pre + 1, dtype=int)
                    for key, neurons in cluster_input.items():
                        pre_lookup[np.asarray(neurons)] = key - 1
                    post_lookup = np.empty(max_post + 1, dtype=int)
                    for key, neurons in cluster_hidden.items():
                        post_lookup[np.asarray(neurons)] = key - 1

                    # Precompute cluster assignment using original flat positions (_flat_idx_active)
                    # so that pre = flat_idx // M and post = flat_idx % M remain correct after
                    # unresponsive entries are removed.
                    pre_clusters  = pre_lookup[_flat_idx_active // M]   # shape (N_active,), 0-based
                    post_clusters = post_lookup[_flat_idx_active % M]   # shape (N_active,), 0-based

                    corr_lst, om_lst, num_lst = [], [], []
                    over_membership_lst = []
                    z_count_lst = []

                    figsm, axsm = plt.subplots(2,mp,figsize=(4*mp,4*2))
                    for cidx, cluster_num in enumerate(all_choice_order):
                        idx = np.where(_col_all_active == cluster_num)[0]

                        # Build Z_count via bincount on linearised (pre, post) indices —
                        # replaces the per-element Python loop + dict lookup.
                        flat_idx = pre_clusters[idx] * n_hid + post_clusters[idx]
                        Z_count = np.bincount(flat_idx, minlength=n_in * n_hid).reshape(n_in, n_hid).astype(float)

                        corr = np.corrcoef(flat_all_num, Z_count.flatten())[0, 1]
                        corr_lst.append(corr)

                        all_num_avg = all_num * cluster_size_percent[cidx]
                        over_membership = Z_count / all_num_avg
                        om = np.mean(np.abs(over_membership))
                        om_lst.append(om); num_lst.append(len(idx))
                        over_membership_lst.append(over_membership)
                        z_count_lst.append(Z_count)

                        if cnt < mp:
                            sns.heatmap(Z_count, ax=axsm[0,cnt], cmap=cs)
                            sns.heatmap(over_membership, ax=axsm[1,cnt], cmap=cs, vmin=0, vmax=2)
                            for axindex in range(2):
                                axsm[axindex,cnt].set_xlabel("Hidden Cluster Index", fontsize=15)
                                axsm[axindex,cnt].set_ylabel("Input Cluster Index", fontsize=15)
                            axsm[0,cnt].set_title(f"Modulation Cluster {cluster_num}; corr: {corr:.3f}", fontsize=12)
                            axsm[1,cnt].set_title(f"Modulation Cluster {cluster_num}; om: {om:.3f}", fontsize=12)
                            cnt += 1

                    figsm.tight_layout()
                    figsm.savefig(f"./multiple_tasks/{clustering_name}_specific_case_{savefigure_name}.png", dpi=300)
                    plt.close(figsm)

                    # Directional asymmetry analysis:
                    # Each modulation cluster groups synapses (pre, post) with similar Hebbian dynamics.
                    # The question is whether co-clustered synapses share a common pre-synaptic (input)
                    # neuron cluster, a common post-synaptic (hidden) neuron cluster, or neither.
                    # Over-membership (OM) = observed count / expected count under random assignment.
                    # OM > 1 means the modulation cluster is enriched for that (input or hidden) cluster.
                    # The log2 asymmetry ratio log2(peak_input_OM / peak_hidden_OM) > 0 means
                    # a modulation cluster is better explained by shared input identity than shared
                    # hidden identity, suggesting pre-synaptic organisation of plasticity.
                    om_stack = np.stack(over_membership_lst)           # (N_cls, n_in, n_hid)
                    N_cls = len(over_membership_lst)
                    # Mean |OM| collapsed over the other dimension
                    over_membership_array_input  = np.mean(np.abs(om_stack), axis=2)  # (N_cls, n_in)
                    over_membership_array_hidden = np.mean(np.abs(om_stack), axis=1)  # (N_cls, n_hid)

                    # Per-mod-cluster peak alignment strength in each dimension
                    max_input_om  = over_membership_array_input.max(axis=1)   # (N_cls,)
                    max_hidden_om = over_membership_array_hidden.max(axis=1)  # (N_cls,)

                    # Log2 asymmetry ratio: >0 = input-dominant, <0 = hidden-dominant
                    log_asym = np.log2(
                        np.clip(max_input_om, 1e-6, None) /
                        np.clip(max_hidden_om, 1e-6, None)
                    )
                    input_dominant_frac = float(np.mean(log_asym > 0))
                    mean_log_asym = float(np.mean(log_asym))

                    figomcluster, axsomcluster = plt.subplots(2, 2, figsize=(14, 10))
                    x_cls = np.arange(N_cls)

                    # Panel [0,0]: scatter — max input OM vs max hidden OM, sized by cluster size
                    ax_sc = axsomcluster[0, 0]
                    sizes = cluster_size_percent / cluster_size_percent.max() * 200 + 20
                    sc = ax_sc.scatter(
                        max_input_om, max_hidden_om,
                        s=sizes, c=x_cls, cmap="viridis",
                        alpha=0.8, edgecolors="k", linewidths=0.5,
                    )
                    lim = max(float(max_input_om.max()), float(max_hidden_om.max())) * 1.15
                    ax_sc.plot([0, lim], [0, lim], "--", color="gray", lw=1.2, label="y = x")
                    ax_sc.set_xlim(0, lim); ax_sc.set_ylim(0, lim)
                    ax_sc.set_xlabel("Peak OM — input cluster", fontsize=12)
                    ax_sc.set_ylabel("Peak OM — hidden cluster", fontsize=12)
                    ax_sc.set_title(
                        f"Per-cluster peak alignment\n"
                        f"{input_dominant_frac:.0%} input-dominant  |  mean log₂ ratio = {mean_log_asym:+.2f}",
                        fontsize=10,
                    )
                    ax_sc.legend(fontsize=8)
                    plt.colorbar(sc, ax=ax_sc, label="Mod cluster order (0=largest)")

                    # Panel [0,1]: log asymmetry ratio bar chart, sorted by cluster size
                    ax_bar = axsomcluster[0, 1]
                    bar_colors = ["#d73027" if v > 0 else "#4575b4" for v in log_asym]
                    ax_bar.bar(x_cls, log_asym, color=bar_colors, edgecolor="k", linewidth=0.4)
                    ax_bar.axhline(0, color="black", lw=1)
                    ax_bar.set_xlabel("Modulation cluster (largest → smallest)", fontsize=12)
                    ax_bar.set_ylabel("log₂(peak input OM / peak hidden OM)", fontsize=11)
                    ax_bar.set_title(
                        "Directional asymmetry per mod cluster\n"
                        "red = input-dominant   blue = hidden-dominant",
                        fontsize=10,
                    )

                    # Panel [1,0]: input-cluster OM profiles across mod clusters
                    ax_in = axsomcluster[1, 0]
                    in_colors = color_func.rainbow_generate(n_in)
                    for ci in range(n_in):
                        ax_in.plot(
                            x_cls, over_membership_array_input[:, ci],
                            color=in_colors[ci], alpha=0.8, lw=1.5, label=f"In C{ci}",
                        )
                    ax_in.axhline(1.0, color="gray", lw=1, linestyle="--", label="Expected = 1")
                    ax_in.set_xlabel("Modulation cluster (largest → smallest)", fontsize=12)
                    ax_in.set_ylabel("Mean |OM| — input cluster", fontsize=12)
                    ax_in.set_title("Input-cluster alignment profiles", fontsize=10)
                    ax_in.legend(fontsize=8, loc="upper right")

                    # Panel [1,1]: hidden-cluster OM profiles across mod clusters
                    ax_hid = axsomcluster[1, 1]
                    hid_colors = color_func.rainbow_generate(n_hid)
                    for ci in range(n_hid):
                        ax_hid.plot(
                            x_cls, over_membership_array_hidden[:, ci],
                            color=hid_colors[ci], alpha=0.8, lw=1.5, label=f"Hid C{ci}",
                        )
                    ax_hid.axhline(1.0, color="gray", lw=1, linestyle="--", label="Expected = 1")
                    ax_hid.set_xlabel("Modulation cluster (largest → smallest)", fontsize=12)
                    ax_hid.set_ylabel("Mean |OM| — hidden cluster", fontsize=12)
                    ax_hid.set_title("Hidden-cluster alignment profiles", fontsize=10)
                    ax_hid.legend(fontsize=8, loc="upper right")

                    figomcluster.suptitle(
                        f"Modulation cluster ↔ Input / Hidden cluster alignment  "
                        f"| {n_in} input clusters × {n_hid} hidden clusters × {N_cls} mod clusters",
                        fontsize=11,
                    )
                    figomcluster.tight_layout()
                    figomcluster.savefig(f"./multiple_tasks/{clustering_name}_om_across_cluster_{savefigure_name}.png", dpi=300)
                    plt.close(figomcluster)

                    corr_arr = np.array(corr_lst)
                    om_arr   = np.array(om_lst)
                    num_arr  = np.array(num_lst)

                    figgroupcorr, axsgroupcorr = plt.subplots(1,5,figsize=(4*5,4))
                    axsgroupcorr[0].hist(corr_arr, bins="auto")
                    axsgroupcorr[0].set_xlabel("Correlation per Group", fontsize=15)
                    axsgroupcorr[1].hist(om_arr, bins="auto")
                    axsgroupcorr[1].set_xlabel("Mean Absolute Average OM", fontsize=15)
                    axsgroupcorr[2].scatter(corr_arr, om_arr, c=c_vals[0], alpha=0.7)
                    axsgroupcorr[3].scatter(num_arr, corr_arr, c=c_vals[0], alpha=0.7)
                    axsgroupcorr[4].scatter(num_arr, om_arr,  c=c_vals[0], alpha=0.7)

                    x_fit0, y_fit0, r0, slope0, _, p0 = helper.linear_regression(
                        corr_lst, om_lst, log=False, through_origin=False)
                    axsgroupcorr[2].plot(x_fit0, y_fit0,
                        label=f"R={r0:.3f}; slope={slope0:.3f}; p={p0:.3e}")

                    x_fit, y_fit, r1, slope1, _, p1 = helper.linear_regression(
                        num_lst, corr_lst, log=False, through_origin=False)
                    axsgroupcorr[3].plot(x_fit, y_fit,
                        label=f"R={r1:.3f}; slope={slope1:.3f}; p={p1:.3e}")

                    x_fit2, y_fit2, r2, slope2, _, p2 = helper.linear_regression(
                        num_lst, om_lst, log=False, through_origin=False)
                    axsgroupcorr[4].plot(x_fit2, y_fit2,
                        label=f"R={r2:.3f}; slope={slope2:.3f}; p={p2:.3e}")

                    axsgroupcorr[2].set_xlabel("Corr", fontsize=15)
                    axsgroupcorr[2].set_ylabel("OM", fontsize=15)
                    axsgroupcorr[3].set_xlabel("Modulation Cluster Size", fontsize=15)
                    axsgroupcorr[3].set_ylabel("Corr", fontsize=15)
                    axsgroupcorr[4].set_xlabel("Modulation Cluster Size", fontsize=15)
                    axsgroupcorr[4].set_ylabel("OM", fontsize=15)
                    for _ax in axsgroupcorr[2:]:
                        _ax.legend(fontsize=7)

                    figgroupcorr.tight_layout()
                    figgroupcorr.savefig(f"./multiple_tasks/{clustering_name}_corr_allgroup_case_{savefigure_name}.png", dpi=300)
                    plt.close(figgroupcorr)

                    # Global assignment heatmap: all modulation clusters × flattened
                    # (input-cluster, hidden-cluster) pairs.  Each row is one modulation
                    # cluster (largest → smallest); each column is one neuron-cluster pair
                    # grouped by input cluster.  Color = over-membership (1 = chance level).
                    om_global = om_stack.reshape(N_cls, n_in * n_hid)  # (N_cls, n_in*n_hid)

                    y_labels = [
                        f"MC{cid} ({pct:.1%})"
                        for cid, pct in zip(all_choice_order, cluster_size_percent)
                    ]
                    x_labels = [
                        f"({i},{j})" for i in range(n_in) for j in range(n_hid)
                    ]

                    # Symmetric colormap range around 1 (clipped at 99th percentile of deviation)
                    max_dev = float(np.nanpercentile(np.abs(om_global - 1.0), 99))
                    max_dev = max(max_dev, 0.1)   # avoid degenerate all-uniform case
                    vmin_gah = max(0.0, 1.0 - max_dev)
                    vmax_gah = 1.0 + max_dev

                    fig_gah, ax_gah = plt.subplots(
                        1, 1,
                        figsize=(max(8, n_in * n_hid * 0.7), max(4, N_cls * 0.35 + 1.5))
                    )
                    sns.heatmap(
                        om_global,
                        ax=ax_gah,
                        cmap="RdBu_r",
                        vmin=vmin_gah, vmax=vmax_gah,
                        center=1.0,
                        xticklabels=x_labels,
                        yticklabels=y_labels,
                        linewidths=0.3,
                        linecolor="lightgray",
                        cbar_kws={"label": "Over-membership (1 = expected)"},
                    )
                    # Vertical lines separating input-cluster groups
                    for _i in range(1, n_in):
                        ax_gah.axvline(_i * n_hid, color="black", lw=1.5)
                    # Input-cluster group labels below x-axis ticks
                    ax_gah.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
                    ax_gah.set_yticklabels(y_labels, fontsize=8)
                    ax_gah.set_xlabel("(Input cluster, Hidden cluster) pair", fontsize=12)
                    ax_gah.set_ylabel("Modulation cluster (largest → smallest)", fontsize=12)
                    ax_gah.set_title(
                        f"Global assignment heatmap: mod-cluster over-membership per neuron-cluster block\n"
                        f"{N_cls} mod clusters  ×  {n_in} input clusters  ×  {n_hid} hidden clusters",
                        fontsize=11,
                    )
                    fig_gah.tight_layout()
                    fig_gah.savefig(
                        f"./multiple_tasks/{clustering_name}_global_assignment_heatmap_{savefigure_name}.png",
                        dpi=300,
                    )
                    plt.close(fig_gah)

                    # -------------------------------------------------------
                    # Analysis 2: Block purity / entropy
                    #
                    # Direction A — mod-cluster entropy over neuron-cluster blocks:
                    #   For each modulation cluster c, treat the distribution of its
                    #   synapses over (input-cluster, hidden-cluster) pairs as a
                    #   probability vector and compute its Shannon entropy.
                    #   Low entropy  → cluster is concentrated in one block (specialised).
                    #   High entropy → cluster spans many blocks (diffuse).
                    #
                    # Direction B — block entropy over mod clusters:
                    #   For each (input-cluster, hidden-cluster) pair, treat the
                    #   distribution of its synapses across modulation clusters as a
                    #   probability vector.
                    #   Low entropy  → block is "owned" by one mod cluster (clean partition).
                    #   High entropy → block is shared across many mod clusters.
                    # -------------------------------------------------------
                    z_count_stack = np.stack(z_count_lst)  # (N_cls, n_in, n_hid)

                    # Direction A: entropy of each mod cluster over (n_in*n_hid) blocks
                    z_flat = z_count_stack.reshape(N_cls, -1).astype(float)     # (N_cls, n_in*n_hid)
                    row_sum = z_flat.sum(axis=1, keepdims=True)
                    p_mod = z_flat / np.maximum(row_sum, 1e-12)                 # (N_cls, n_in*n_hid)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        h_mod = -np.sum(np.where(p_mod > 0, p_mod * np.log2(p_mod), 0.0), axis=1)
                    h_mod_norm = h_mod / np.log2(n_in * n_hid)                  # (N_cls,) in [0,1]

                    # Direction B: entropy of each (i,j) block over N_cls mod clusters
                    col_sum = z_count_stack.sum(axis=0, keepdims=True)          # (1, n_in, n_hid)
                    p_block = z_count_stack / np.maximum(col_sum, 1e-12)        # (N_cls, n_in, n_hid)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        h_block = -np.sum(np.where(p_block > 0, p_block * np.log2(p_block), 0.0), axis=0)
                    h_block_norm = h_block / np.log2(max(N_cls, 2))             # (n_in, n_hid) in [0,1]

                    fig_ent, axs_ent = plt.subplots(2, 3, figsize=(16, 9))

                    # [0,0] Bar chart: mod-cluster normalised entropy, sorted by cluster size
                    ax = axs_ent[0, 0]
                    bar_colors = plt.cm.RdYlGn_r(h_mod_norm)
                    ax.bar(np.arange(N_cls), h_mod_norm, color=bar_colors, edgecolor="k", linewidth=0.4)
                    ax.axhline(1.0, color="gray", lw=1, linestyle="--", label="Max entropy (uniform)")
                    ax.set_xticks(np.arange(N_cls))
                    ax.set_xticklabels(
                        [f"MC{cid}" for cid in all_choice_order],
                        rotation=45, ha="right", fontsize=7,
                    )
                    ax.set_ylim(0, 1.1)
                    ax.set_xlabel("Modulation cluster (largest → smallest)", fontsize=11)
                    ax.set_ylabel("Normalised entropy", fontsize=11)
                    ax.set_title(
                        "Mod-cluster entropy over neuron-cluster blocks\n"
                        "0 = one block owns it all,  1 = uniform across all blocks",
                        fontsize=10,
                    )
                    ax.legend(fontsize=8)

                    # [0,1] Scatter: cluster size vs mod-cluster entropy
                    ax = axs_ent[0, 1]
                    sc = ax.scatter(
                        cluster_size_percent * 100, h_mod_norm,
                        c=h_mod_norm, cmap="RdYlGn_r", vmin=0, vmax=1,
                        s=60, edgecolors="k", linewidths=0.5, alpha=0.85,
                    )
                    plt.colorbar(sc, ax=ax, label="Normalised entropy")
                    ax.set_xlabel("Modulation cluster size (%)", fontsize=11)
                    ax.set_ylabel("Normalised entropy", fontsize=11)
                    ax.set_title("Cluster size vs specialisation", fontsize=10)

                    # [0,2] Histogram of mod-cluster normalised entropy
                    ax = axs_ent[0, 2]
                    ax.hist(h_mod_norm, bins="auto", color="#4575b4", edgecolor="k", alpha=0.8)
                    ax.axvline(np.mean(h_mod_norm), color="red", lw=1.5,
                               label=f"Mean = {np.mean(h_mod_norm):.2f}")
                    ax.set_xlabel("Normalised entropy", fontsize=11)
                    ax.set_ylabel("Count", fontsize=11)
                    ax.set_title("Distribution of mod-cluster entropy", fontsize=10)
                    ax.legend(fontsize=8)

                    # [1,0] Heatmap: block entropy over mod clusters (n_in × n_hid)
                    ax = axs_ent[1, 0]
                    sns.heatmap(
                        h_block_norm,
                        ax=ax,
                        cmap="RdYlGn_r",
                        vmin=0, vmax=1,
                        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                        cbar_kws={"label": "Normalised entropy"},
                        xticklabels=[f"Hid C{j}" for j in range(n_hid)],
                        yticklabels=[f"In C{i}" for i in range(n_in)],
                    )
                    ax.set_xlabel("Hidden cluster", fontsize=11)
                    ax.set_ylabel("Input cluster", fontsize=11)
                    ax.set_title(
                        "Block entropy over mod clusters\n"
                        "0 = owned by one mod cluster,  1 = shared equally",
                        fontsize=10,
                    )

                    # [1,1] Histogram of block normalised entropy
                    ax = axs_ent[1, 1]
                    ax.hist(h_block_norm.flatten(), bins="auto", color="#d73027", edgecolor="k", alpha=0.8)
                    ax.axvline(np.mean(h_block_norm), color="navy", lw=1.5,
                               label=f"Mean = {np.mean(h_block_norm):.2f}")
                    ax.set_xlabel("Normalised entropy", fontsize=11)
                    ax.set_ylabel("Count", fontsize=11)
                    ax.set_title("Distribution of block entropy", fontsize=10)
                    ax.legend(fontsize=8)

                    # [1,2] Scatter: mod-cluster entropy vs block entropy of the peak block
                    # (the (i,j) block where that mod cluster has the highest count)
                    peak_block_flat = z_flat.argmax(axis=1)                     # (N_cls,)
                    peak_block_entropy = h_block_norm.flatten()[peak_block_flat] # (N_cls,)
                    ax = axs_ent[1, 2]
                    sc2 = ax.scatter(
                        h_mod_norm, peak_block_entropy,
                        c=cluster_size_percent * 100, cmap="viridis",
                        s=60, edgecolors="k", linewidths=0.5, alpha=0.85,
                    )
                    plt.colorbar(sc2, ax=ax, label="Cluster size (%)")
                    ax.set_xlabel("Mod-cluster entropy (specialisation)", fontsize=11)
                    ax.set_ylabel("Entropy of its peak block", fontsize=11)
                    ax.set_title(
                        "Specialised clusters → do they own exclusive blocks?\n"
                        "Bottom-left = both specific",
                        fontsize=10,
                    )

                    fig_ent.suptitle(
                        f"Block purity / entropy analysis  |  "
                        f"{N_cls} mod clusters  ×  {n_in} input clusters  ×  {n_hid} hidden clusters",
                        fontsize=12,
                    )
                    fig_ent.tight_layout()
                    fig_ent.savefig(
                        f"./multiple_tasks/{clustering_name}_block_entropy_{savefigure_name}.png",
                        dpi=300,
                    )
                    plt.close(fig_ent)

            figcol.tight_layout()
            figcol.savefig(f"./multiple_tasks/{clustering_name}_allneuron_grouplength_{savefigure_name}.png", dpi=300)  
            plt.close(figcol)
            figcolcluster.tight_layout()
            figcolcluster.savefig(f"./multiple_tasks/{clustering_name}_actualcluster_length_{savefigure_name}.png", dpi=300)
            plt.close(figcolcluster)
            figppshare.tight_layout()
            figppshare.savefig(f"./multiple_tasks/{clustering_name}_prepost_belonging_{savefigure_name}.png", dpi=300)  
            plt.close(figppshare)

            # all following analysis based on the maximal G (G_lst[-1])
            cell_vars_rules_sorted_norm_pre = cell_vars_rules_sorted_norm[np.ix_(result_pre["row_order"], result_pre["col_order"])]
            cell_vars_rules_sorted_norm_post = cell_vars_rules_sorted_norm[np.ix_(result_post["row_order"], result_post["col_order"])]

            # 2025-11-12: for all G results (after reordering)
            # up to here, the covariance matrix is determined. 
            cell_vars_rules_sorted_norm_all_lst = []
            for result_all_ in result_all_lst:
                cell_vars_rules_sorted_norm_all = cell_vars_rules_sorted_norm[np.ix_(result_all_["row_order"], result_all_["col_order"])]
                cell_vars_rules_sorted_norm_all_lst.append(cell_vars_rules_sorted_norm_all)

            clustering_data_hierarchy["modulation_all_pre"] = result_pre["col_linkage"]
            clustering_data_hierarchy["modulation_all_post"] = result_post["col_linkage"]

            # check if length is consistent
            assert len(G_lst) == len(result_all_lst)

            # plot original, pre, post, plus all results under different G
            tf = 3 + len(result_all_lst)
            figprepost, axprepost = plt.subplots(tf,1,figsize=(24,8*tf))

            # For unnormalized data use a log colorbar bounded below at 1e-4;
            # compute a shared vmax across all subplots for consistent color scale.
            if not clustering_normalize:
                _all_mats = (
                    [cell_vars_rules_sorted_norm,
                     cell_vars_rules_sorted_norm_pre,
                     cell_vars_rules_sorted_norm_post]
                    + list(cell_vars_rules_sorted_norm_all_lst)
                )
                _vmax_log = float(max(m.max() for m in _all_mats))
                _heatmap_kw = dict(norm=LogNorm(vmin=1e-4, vmax=_vmax_log))
            else:
                _heatmap_kw = dict(vmin=vmins, vmax=vmaxs)

            sns.heatmap(cell_vars_rules_sorted_norm, ax=axprepost[0], cmap=cs, cbar=True, **_heatmap_kw)
            sns.heatmap(cell_vars_rules_sorted_norm_pre, ax=axprepost[1], cmap=cs, cbar=True, **_heatmap_kw)
            sns.heatmap(cell_vars_rules_sorted_norm_post, ax=axprepost[2], cmap=cs, cbar=True, **_heatmap_kw)

            # plot the headmap for all G result
            for k in range(len(cell_vars_rules_sorted_norm_all_lst)):
                sns.heatmap(cell_vars_rules_sorted_norm_all_lst[k], ax=axprepost[3+k], cmap=cs, cbar=True, **_heatmap_kw)

            for ax in axprepost:
                ax.set_yticks(np.arange(len(tb_break_name)))
                
            axprepost[0].set_yticklabels(tb_break_name, rotation=0, ha='right', va='center', fontsize=9)
            axprepost[1].set_yticklabels(tb_break_name[result_pre["row_order"]], rotation=0, ha='right', va='center', fontsize=9)
            axprepost[2].set_yticklabels(tb_break_name[result_post["row_order"]], rotation=0, ha='right', va='center', fontsize=9)
            
            # save for external plotting 
            cell_vars_rules_sorted_norm_all_save = {}
            cell_vars_rules_sorted_norm_all_save["cell_vars_rules_sorted_norm_all_lst"] = cell_vars_rules_sorted_norm_all_lst

            # 2025-10-30: modulation, clustering by G groups from MinibatchKmeans, "border" between clusters
            modulation_cluster_norm = []
            modulation_cluster_boundary = []
            for k in range(len(cell_vars_rules_sorted_norm_all_lst)):
                axprepost[3+k].set_yticklabels(tb_break_name[result_all_lst[k]["row_order"]], 
                                               rotation=0, ha='right', va='center', fontsize=9)
                result_all = result_all_lst[k]
                rl_all = np.asarray(result_all["row_labels"])[result_all["row_order"]]
                cl_all = np.asarray(result_all["col_labels"])[result_all["col_order"]]
                rbreaks_all_ = clustering._breaks(rl_all)
                cbreaks_all_ = clustering._breaks(cl_all)
                for rb in rbreaks_all_:
                    axprepost[3+k].axhline(rb, color="w", lw=3.0, zorder=3)
                    axprepost[3+k].axhline(rb, color="k", lw=0.8, zorder=4)
                for cb in cbreaks_all_:
                    axprepost[3+k].axvline(cb, color="w", lw=3.0, zorder=3)
                    axprepost[3+k].axvline(cb, color="k", lw=0.8, zorder=4)
                    
                modulation_cluster_boundary.append([rbreaks_all_, cbreaks_all_])

                # 2025-11-12: calculate the mean of the reordered covaraince matrix under different G value
                # notice G indicates the group size, not the actual 
                rbreaks_all_end = [0] + rbreaks_all_ + [cell_vars_rules_sorted_norm_all_lst[0].shape[0]]
                cbreaks_all_end = [0] + cbreaks_all_ + [cell_vars_rules_sorted_norm_all_lst[0].shape[1]]

                varmeanmod = np.zeros((len(rbreaks_all_end) - 1, len(cbreaks_all_end) - 1))
                for rr in range(len(rbreaks_all_end) - 1):
                    for cc in range(len(cbreaks_all_end) - 1):
                        varmeanmod[rr,cc] = np.mean(cell_vars_rules_sorted_norm_all_lst[k][rbreaks_all_end[rr]:rbreaks_all_end[rr+1], 
                                                                                        cbreaks_all_end[cc]:cbreaks_all_end[cc+1]])

                modulation_cluster_norm.append(varmeanmod)    
                
            cell_vars_rules_sorted_norm_all_save["modulation_cluster_boundary"] = modulation_cluster_boundary      
                
            axprepost[0].set_title("Original", fontsize=15)
            axprepost[1].set_title(f"Group by Pre (row_k={result_pre['row_k']}, col_k={result_pre['col_k']})", fontsize=15)
            axprepost[2].set_title(f"Group by Post (row_k={result_post['row_k']}, col_k={result_post['col_k']})", fontsize=15)

            # 2025-10-30: modulation, clustering by pre/post, "border" between clusters
            rl_pre = np.asarray(result_pre["row_labels"])[result_pre["row_order"]]
            cl_pre = np.asarray(result_pre["col_labels"])[result_pre["col_order"]]
            rbreaks_pre = clustering._breaks(rl_pre)
            cbreaks_pre = clustering._breaks(cl_pre)
            for rb in rbreaks_pre:
                axprepost[1].axhline(rb, color="w", lw=3.0, zorder=3)
                axprepost[1].axhline(rb, color="k", lw=0.8, zorder=4)
            for cb in cbreaks_pre:
                axprepost[1].axvline(cb, color="w", lw=3.0, zorder=3)
                axprepost[1].axvline(cb, color="k", lw=0.8, zorder=4)

            rl_post = np.asarray(result_post["row_labels"])[result_post["row_order"]]
            cl_post = np.asarray(result_post["col_labels"])[result_post["col_order"]]
            rbreaks_post = clustering._breaks(rl_post)
            cbreaks_post = clustering._breaks(cl_post)
            for rb in rbreaks_post:
                axprepost[2].axhline(rb, color="w", lw=3.0, zorder=3)
                axprepost[2].axhline(rb, color="k", lw=0.8, zorder=4)
            for cb in cbreaks_post:
                axprepost[2].axvline(cb, color="w", lw=3.0, zorder=3)
                axprepost[2].axvline(cb, color="k", lw=0.8, zorder=4)
            
            for k in range(len(cell_vars_rules_sorted_norm_all_lst)):
                axprepost[3+k].set_title(f"Group by All, G={G_lst[k]} (row_k={result_all_lst[k]['row_k']}, col_k={result_all_lst[k]['col_k']})", fontsize=15)
                
            figprepost.tight_layout()
            figprepost.savefig(f"./multiple_tasks/{clustering_name}_bygroup_variance_{savefigure_name}.png", dpi=300)
            plt.close(figprepost)
            
            # save the fitting result under different number of predefined modulation cluster
            with open(f"./multiple_tasks/{clustering_name}_clustering_result_all_{savefigure_name}.pkl", "wb") as f:
                pickle.dump(cell_vars_rules_sorted_norm_all_save, f)

            # plot the norm 
            figmodnorm, axsmodnorm = plt.subplots(len(modulation_cluster_norm),1,figsize=(10,4*len(modulation_cluster_norm)))
            for idx in range(len(modulation_cluster_norm)):
                sns.heatmap(modulation_cluster_norm[idx], ax=axsmodnorm[idx], cmap=cs)
                axsmodnorm[idx].set_xlabel("Modulation Index", fontsize=15)
                axsmodnorm[idx].set_ylabel("Session Index", fontsize=15)

            figmodnorm.tight_layout()
            figmodnorm.savefig(f"./multiple_tasks/{clustering_name}_modulation_clusternorm_{savefigure_name}.png", dpi=300)
            plt.close(figmodnorm)

            # plot the score
            def _gap_curve_mod(Z, k_vals):
                """Compute Ward merge-height gap for each k from a linkage matrix."""
                if Z is None:
                    return np.full(len(k_vals), np.nan)
                n = Z.shape[0] + 1
                gaps = []
                for k in k_vals:
                    idx_above = n - k
                    idx_below = n - k - 1
                    h_above = float(Z[idx_above, 2]) if 0 <= idx_above < Z.shape[0] else np.inf
                    h_below = float(Z[idx_below, 2]) if 0 <= idx_below < Z.shape[0] else 0.0
                    gaps.append(h_above - h_below)
                return np.asarray(gaps, dtype=float)

            figscore, (axscore, axgap) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            # result_pre / result_post: single grouping → col_score_recording_mean has no meaningful std
            pre_ks  = np.asarray(result_pre["col_k_values"],  dtype=int)
            post_ks = np.asarray(result_post["col_k_values"], dtype=int)

            # ---------- silhouette: pre ----------
            axscore.plot(pre_ks, [result_pre["col_score_recording_mean"][k] for k in pre_ks],
                         label="col pre", color=c_vals[1])
            axscore.scatter(result_pre["col_k"],
                            result_pre["col_score_recording_mean"][result_pre["col_k"]],
                            color=c_vals[1], zorder=5, s=60, marker="*")
            axscore.axvline(result_pre["col_strict_best_k"], color=c_vals[1], linestyle="-",  linewidth=1.0)
            axscore.axvline(result_pre["col_k"],             color=c_vals[1], linestyle="--", linewidth=1.0)

            # ---------- silhouette: post ----------
            axscore.plot(post_ks, [result_post["col_score_recording_mean"][k] for k in post_ks],
                         label="col post", color=c_vals[3])
            axscore.scatter(result_post["col_k"],
                            result_post["col_score_recording_mean"][result_post["col_k"]],
                            color=c_vals[3], zorder=5, s=60, marker="*")
            axscore.axvline(result_post["col_strict_best_k"], color=c_vals[3], linestyle="-",  linewidth=1.0)
            axscore.axvline(result_post["col_k"],             color=c_vals[3], linestyle="--", linewidth=1.0)

            # ---------- silhouette: result_all_lst (mean ± std across groupings) ----------
            for k_idx in range(len(result_all_lst)):
                r     = result_all_lst[k_idx]
                ks    = np.asarray(r["col_k_values"], dtype=int)
                mean  = np.asarray([r["col_score_recording_mean"][ki] for ki in ks], dtype=float)
                std   = np.asarray([r["col_score_recording_std"][ki]  for ki in ks], dtype=float)
                color = c_vals[5 + k_idx]
                axscore.plot(ks, mean, label=f"col all, G={G_lst[k_idx]}", color=color, linestyle="--")
                axscore.fill_between(ks, mean - std, mean + std, color=color, alpha=0.2)
                axscore.scatter(r["col_k"], r["col_score_recording_mean"][r["col_k"]],
                                color=color, zorder=5, s=60, marker="*")
                axscore.axvline(r["col_strict_best_k"], color=color, linestyle="-",  linewidth=1.0)
                axscore.axvline(r["col_k"],             color=color, linestyle="--", linewidth=1.0)

            axscore.set_ylabel("Silhouette Score", fontsize=15)
            axscore.legend()

            # ---------- gap: pre ----------
            axgap.plot(pre_ks, _gap_curve_mod(result_pre["col_linkage"], pre_ks),
                       label="col pre", color=c_vals[1])
            axgap.axvline(result_pre["col_strict_best_k"], color=c_vals[1], linestyle="-",  linewidth=1.0)
            axgap.axvline(result_pre["col_k"],             color=c_vals[1], linestyle="--", linewidth=1.0)

            # ---------- gap: post ----------
            axgap.plot(post_ks, _gap_curve_mod(result_post["col_linkage"], post_ks),
                       label="col post", color=c_vals[3])
            axgap.axvline(result_post["col_strict_best_k"], color=c_vals[3], linestyle="-",  linewidth=1.0)
            axgap.axvline(result_post["col_k"],             color=c_vals[3], linestyle="--", linewidth=1.0)

            # ---------- gap: result_all_lst ----------
            for k_idx in range(len(result_all_lst)):
                r     = result_all_lst[k_idx]
                ks    = np.asarray(r["col_k_values"], dtype=int)
                color = c_vals[5 + k_idx]
                axgap.plot(ks, _gap_curve_mod(r["col_linkage"], ks),
                           label=f"col all, G={G_lst[k_idx]}", color=color, linestyle="--")
                axgap.axvline(r["col_strict_best_k"], color=color, linestyle="-",  linewidth=1.0)
                axgap.axvline(r["col_k"],             color=color, linestyle="--", linewidth=1.0)

            axgap.set_xlabel("Number of Clusters", fontsize=15)
            axgap.set_ylabel("Gap Score", fontsize=15)
            axgap.set_xscale("log")
            axgap.legend()

            figscore.tight_layout()
            figscore.savefig(f"./multiple_tasks/{clustering_name}_variance_cluster_score_{savefigure_name}.png", dpi=300)
            plt.close(figscore)


            # compare modulation grouping result
            # either using pre, post, or different level of K-means pre-clusters
            all_mod = [result_pre, result_post] + result_all_lst
            all_mod_name = ["Pre", "Post"] + result_all_name_lst

            all_mod_metric_allk = []
            # loop through downsampled clusters along the neuron dimension
            select_col_allk = np.arange(lower_cluster_mod, 30, 3)
            valid_select_col_allk = []
            for select_col_k_ in select_col_allk:
                if all(
                    (select_col_k_ in mod_["row_labels_by_k"]) and
                    (select_col_k_ in mod_["col_labels_by_k"])
                    for mod_ in all_mod
                ):
                    valid_select_col_allk.append(select_col_k_)

            if len(valid_select_col_allk) == 0:
                print("No shared modulation clustering keys found for comparison; skipping metric plot.")
                all_mod_metric_allk = np.empty((0, len(all_mod), 3))
            else:
                for select_col_k_ in valid_select_col_allk:
                    all_mod_metric = []
                    for mod_ in all_mod: 
                        eval_res_modulation = clustering_metric.evaluate_bicluster_clustering(
                            cell_vars_rules_sorted_norm, 
                            row_labels=mod_["row_labels_by_k"][select_col_k_], 
                            col_labels=mod_["col_labels_by_k"][select_col_k_]
                        )
                        
                        all_mod_metric.append([
                            eval_res_modulation["metrics"]["CH_blocks"],
                            eval_res_modulation["metrics"]["DB_blocks"],
                            eval_res_modulation["metrics"]["XB_blocks"], 
                        ])

                    all_mod_metric_allk.append(all_mod_metric)

                all_mod_metric_allk = np.array(all_mod_metric_allk)
            print(all_mod_metric_allk.shape)
            
            # compare the effect by clustering based on presynaptic neuron, postsynaptic neuron, or all
            # hypothesis: all >≈ postsynaptic > presynaptic
            metric_labels = [
                ("Calinski-Harabasz (↑)", True),   # (label, higher_is_better)
                ("Davies-Bouldin (↓)", False),
                ("Xie-Beni (↓)", False),
            ]
            if all_mod_metric_allk.shape[0] > 0:
                n_metrics = all_mod_metric_allk.shape[2]
                fig, ax = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
                for metric_index in range(n_metrics):
                    for model_index in range(all_mod_metric_allk.shape[1]):
                        ax[metric_index].plot(
                            valid_select_col_allk,
                            all_mod_metric_allk[:, model_index, metric_index],
                            "-o", markersize=4, linewidth=1.5,
                            label=all_mod_name[model_index],
                            color=c_vals[model_index],
                        )
                    ax[metric_index].set_ylabel(metric_labels[metric_index][0], fontsize=13)
                    ax[metric_index].set_xlabel("# Cluster (col k)", fontsize=13)
                    ax[metric_index].set_yscale("log")
                    ax[metric_index].legend(fontsize=8, frameon=True, loc="best")
                    ax[metric_index].tick_params(labelsize=10)
                    ax[metric_index].grid(True, which="both", ls="--", alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"./multiple_tasks/{clustering_name}_between_modulation_{savefigure_name}.png", dpi=300)
                plt.close(fig)

            cluster_info_save_mod[clustering_save_name] = {
                "result_pre": result_pre,
                "result_post": result_post,
                "result_all_lst": result_all_lst,
                "result_all_name_lst": result_all_name_lst,
                "tb_break_name": tb_break_name,
                "cell_vars_rules_sorted_norm": cell_vars_rules_sorted_norm,
            }
    
    # save this only at the end     
    with open(f"./multiple_tasks/cluster_info_mod_{savefigure_name_base}.pkl", "wb") as f:
        pickle.dump(cluster_info_save_mod, f)
        

if __name__ == "__main__":
    # Clean up old output files
    clean = False 
    if clean: 
        for f in Path("multiple_tasks").glob("*.png"):
            f.unlink()
        for f in Path("multiple_tasks").glob("*.pkl"):
            f.unlink()
        
    import re
    saved_nets = sorted(Path("multiple_tasks").glob("savednet_everything_seed*+angle.pt"))
    param_lst = []
    for p in saved_nets:
        m = re.match(r"savednet_everything_seed(\d+)_(\w+)\+hidden\d+\+batch\d+\+angle\.pt", p.name)
        if m:
            param_lst.append((int(m.group(1)), m.group(2)))
            
    print(f"Found {len(param_lst)} saved models: {param_lst}")
    
    param_lst = [[749, "L21e4"]]
    
    for seed, feature in param_lst:
        main(seed, feature)
