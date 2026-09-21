"""
State space geometry analysis across tasks.

Examines how the network's internal representations (hidden states, modulation
matrices, effective modulation) are geometrically organized across tasks in
high-dimensional space. Inspired by Fig 4C of Yang et al. (2024, Nature
Neuroscience), this script tests whether tasks that start from nearby initial
conditions also produce similar dynamical trajectories.

Analyses:
1. Context-end PCA — projects hidden/modulation states at the end of the
   fixation period (just before stimulus onset) into 2D via PCA, colored by
   task and by computational category (pro/anti, delayed/reaction, etc.).
2. Initial condition distance vs. trajectory angle — for each pair of tasks
   sharing the same stimulus, computes the Euclidean distance between their
   pre-stimulus states (initial conditions) and the angle between their
   first-step displacement vectors after stimulus onset. A positive correlation
   indicates that the network separates tasks via distinct initial conditions
   that lead to diverging trajectories.

These analyses are run on three representations: hidden states, raw modulation
M, and effective modulation (W ⊙ M).

For this analysis only, every task's first stimulus is aligned to the same
requested 500-ms fixation duration.  The shared task generator retains its
original variable-timing behavior unless this script explicitly opts in.

Outputs saved to ./state_space/.

By default, the batch entry point analyzes only the paper cohorts with tanh
activation, a 300-dimensional input projection, a 300-dimensional plastic
hidden layer, and L2 regularization 1e-5, 1e-4, 1e-3, or 1e-2 (feature tags
``L21e5``, ``L21e4``, ``L21e3``, and ``L21e2`` respectively).
"""
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

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import mpn 
import mpn_tasks
import color_func
import helper
import multiple_task_performance as mpf 

c_vals = color_func.rainbow_generate(15)
c_vals_l = color_func.rainbow_generate(30)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Default paper cohort.  ``linear_embed`` is the input width of the plastic
# layer (called the projection dimension elsewhere in the repository), while
# n_neurons[1] is its output/hidden width.
TARGET_ACTIVATION = "tanh"
TARGET_PROJECTION_DIM = 300
TARGET_HIDDEN_DIM = 300
FIXED_FIXATION_MS = 500
TARGET_FEATURE_REG_LAMBDAS = {
    "L21e5": 1e-5,
    "L21e4": 1e-4,
    "L21e3": 1e-3,
    "L21e2": 1e-2,
}
STATE_SPACE_DIR = Path("state_space")


def _checkpoint_aname(checkpoint_path):
    """Return the run identifier encoded by a savednet checkpoint path."""
    stem = Path(checkpoint_path).stem
    prefix = "savednet_"
    if not stem.startswith(prefix):
        raise ValueError(f"Expected a {prefix}*.pt checkpoint, got {checkpoint_path}")
    return stem[len(prefix):]


def _raw_config_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    aname = _checkpoint_aname(checkpoint_path)
    return checkpoint_path.with_name(f"param_{aname}_param.json")


def _matches_target_cohort(checkpoint_path):
    """Whether a checkpoint belongs to the default state-space cohort.

    Dimensions and activation are read from the saved parameter JSON rather
    than inferred from the filename.  The exact feature tag is checked as an
    additional guard against silently mixing regularization cohorts.
    """
    config_path = _raw_config_path(checkpoint_path)
    if not config_path.exists():
        return False

    try:
        feature = mpf.parse_feature(str(checkpoint_path))
    except ValueError:
        return False

    with config_path.open() as f:
        config = json.load(f)

    net_params = config.get("net_params", {})
    train_params = config.get("train_params", {})
    n_neurons = net_params.get("n_neurons", [])
    if len(n_neurons) != 3:
        return False

    target_reg_lambda = TARGET_FEATURE_REG_LAMBDAS.get(feature)
    return (
        target_reg_lambda is not None
        and str(net_params.get("activation", "")).casefold() == TARGET_ACTIVATION
        and net_params.get("input_layer_add") is True
        and int(net_params.get("linear_embed", -1)) == TARGET_PROJECTION_DIM
        and int(n_neurons[1]) == TARGET_HIDDEN_DIM
        and train_params.get("weight_reg") == "L2"
        and np.isclose(float(train_params.get("reg_lambda", np.nan)),
                       target_reg_lambda)
    )


def _select_target_checkpoints(checkpoint_paths):
    """Filter checkpoint paths to the default cohort, preserving input order."""
    return [path for path in checkpoint_paths if _matches_target_cohort(path)]


def _clean_state_space_results(output_dir=STATE_SPACE_DIR):
    """Remove existing result files before regenerating the selected cohort.

    Subdirectories are deliberately preserved: the state-space pipeline owns
    the files at the top level of its output directory, but should not recurse
    into a directory another workflow may have placed there.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    removed = []
    for path in output_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed.append(path.name)
    print(f"Cleared {len(removed)} existing file(s) from {output_dir}/")
    return removed


def eval_one(netpathname):
    """
    """
    hidden_size, l2_info = mpf.parse_hidden_and_l2(netpathname)

    netpathname = Path(netpathname)
    aname = _checkpoint_aname(netpathname)
    out_param_path = _raw_config_path(netpathname)
    
    with out_param_path.open() as f: 
        raw_cfg_param = json.load(f)
    
    task_params, train_params, net_params = raw_cfg_param["task_params"], raw_cfg_param["train_params"], raw_cfg_param["net_params"]
    
    # Keep the checkpoint copy on CPU; only the live model needs to occupy GPU
    # memory.  This also makes the frozen W used below immediately NumPy-safe.
    checkpoint = torch.load(netpathname, map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"]
    print(state_dict.keys())
    load_net_params = checkpoint["net_params"]
    print(load_net_params)

    model = mpn.DeepMultiPlasticNet(load_net_params, verbose=False, forzihan=True)

    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
    print("missing:", missing)
    print("unexpected:", unexpected)

    model.to(device)
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
    dt_ms = task_params_c['hp']['dt']
    fixed_fixation_steps = max(1, int(FIXED_FIXATION_MS / dt_ms))
    
    test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
        task_params_c, 
        test_n_batch, 
        rules=all_tasks,
        mode_input="random", 
        device="cpu", 
        verbose=False,
        fixed_fixation_steps=fixed_fixation_steps,
    )
    test_input, test_output, test_mask = test_data
    _, test_trials, test_rule_idxs = test_trials_extra

    fixation_endpoints = {
        int(trial.epochs['fix1'][1]) for trial in test_trials
    }
    if fixation_endpoints != {fixed_fixation_steps}:
        raise RuntimeError(
            "State-space trials were not aligned to one fixation endpoint: "
            f"{sorted(fixation_endpoints)}"
        )
    print(
        f"Fixed fixation across tasks: {fixed_fixation_steps} steps "
        f"({fixed_fixation_steps * dt_ms:g} ms; requested "
        f"{FIXED_FIXATION_MS} ms)"
    )

    test_input = test_input.to(device)
    test_output = test_output.to(device)
    test_mask = test_mask.to(device)

    with torch.no_grad():
        # Accuracy needs only the outputs, so compute it without retaining the
        # enormous M trajectory on CUDA.
        net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode="minimal")
        acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input, isvalid=True, mode=model.acc_measure)
        del net_out

        # A 750 x ~118 x 300 x 300 float32 M trace is about 30 GiB.  Save each
        # time step directly to CPU so the selected 300x300 cohort can run on a
        # conventional GPU.  detach_saved is explicit even under no_grad so the
        # storage contract remains clear if this block is later refactored.
        tracked_out, tracked_hidden, db_test = model.iterate_sequence_batch(
            test_input,
            run_mode="track_states",
            save_to_cpu=True,
            detach_saved=True,
        )
        del tracked_out, tracked_hidden

    print(f"acc: {acc:.2f}")

    # db_test is already on CPU.  NumPy views avoid another full copy of M.
    Ms_orig = db_test["M1"].numpy()
    modulation_W = state_dict["mp_layer1.W"].numpy()
    eff_Ms_orig = Ms_orig * modulation_W
        
    Ms = Ms_orig.reshape(Ms_orig.shape[0], Ms_orig.shape[1], -1) 
    eff_Ms = eff_Ms_orig.reshape(eff_Ms_orig.shape[0], eff_Ms_orig.shape[1], -1)
    hs = db_test["hidden1"].numpy()
    print(f"Ms_orig.shape: {Ms_orig.shape}; Ms.shape: {Ms.shape}; hs.shape: {hs.shape}")

    # All remaining work is NumPy/scikit-learn.  Release CUDA storage before
    # the comparatively long PCA and pairwise geometry calculations.
    del model, checkpoint, state_dict, db_test
    del test_input, test_output, test_mask, test_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    _, labels_stim1, _, rules_epochs = helper.generate_response_stimulus(task_params_c, test_trials)
    
    labels_stim1 = labels_stim1.flatten()
    all_rules = task_params_c['rules']
    
    # rule -> (paper-style computation category, plotting color)
    # https://www.nature.com/articles/s41593-024-01668-6/figures/4
    # Hexes mirrored from paper_plot._RULE_MOTIF (the paper-side source of
    # truth) so analysis and paper figures agree. The dms pair keeps the
    # go/anti pairing deliberately: dmsgo shares Pro Reaction's green and
    # dmsnogo shares Anti Reaction's orange (match/non-match is a pro/anti
    # response rule); only the dmc pair takes Categorization's own deeppink.
    rule_motif_mapping = {
        "fdgo":           ("Pro Delayed",    "#3182ce"),  # blue
        "fdanti":         ("Anti Delayed",   "#e53e3e"),  # red
        "delaygo":        ("Pro Delayed",    "#3182ce"),
        "delayanti":      ("Anti Delayed",   "#e53e3e"),
        "reactgo":        ("Pro Reaction",   "#38a169"),  # green
        "reactanti":      ("Anti Reaction",  "#dd6b20"),  # orange

        "contextdelaydm1": ("Pro Integration", "#805ad5"),  # purple
        "contextdelaydm2": ("Pro Integration", "#805ad5"),
        "delaydm1":        ("Pro Integration", "#805ad5"),
        "delaydm2":        ("Pro Integration", "#805ad5"),
        "multidelaydm":    ("Pro Integration", "#805ad5"),

        "dmsgo":          ("Categorization", "#38a169"),  # green, pairs reactgo
        "dmsnogo":        ("Categorization", "#dd6b20"),  # orange, pairs reactanti
        "dmcgo":          ("Categorization", "#ff1493"),  # deeppink
        "dmcnogo":        ("Categorization", "#ff1493"),
    }

    assert set(all_rules).issubset(set(rule_motif_mapping.keys()))
    assert len(np.unique([v[0] for v in rule_motif_mapping.values()])) == 6
    
    embed_data_names = ["hidden", "mod", "eff_mod"]
    embed_data = [hs, Ms, eff_Ms]

    # Collect PCA results for saving
    pca_results = {}

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

        pca_results[data_name] = {
            "X_2d": X_2d,
            "ctx_rule_labels": ctx_rule_labels,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

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
            STATE_SPACE_DIR / f"state_space_shift_{aname}_{data_name}_noise{noise_level}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Save PCA data for paper_plot reuse
    pca_save_path = STATE_SPACE_DIR / f"state_space_pca_{aname}_noise{noise_level}.pkl"
    with open(pca_save_path, "wb") as f:
        pickle.dump({
            "pca_results": pca_results,
            "all_rules": list(all_rules),
            "rule_motif_mapping": rule_motif_mapping,
            "noise_level": noise_level,
            "fixed_fixation_steps": fixed_fixation_steps,
            "fixed_fixation_ms": fixed_fixation_steps * dt_ms,
            "aname": aname,
        }, f)
    print(f"Saved PCA data to {pca_save_path}")

    # distance between initial conditions vs. angle between first step (closer to Fig 4C)
    def fig4c(shift_time):
        fig, axs = plt.subplots(1, len(embed_data), figsize=(4*len(embed_data),4))
        rval_dict = {}
        # Raw per-task-pair scatter (x = initial-condition distance, y = first-
        # step angle), saved alongside the fit so paper_plot can redraw the
        # scatter without re-running the model forwards.
        scatter_dict = {}

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
            scatter_dict[embed_data_names[idx]] = {
                "dists": np.asarray(all_x, dtype=float),
                "angles_deg": np.asarray(all_y, dtype=float),
            }
            axs[idx].plot(x_fit, y_fit, color='red', label=f"Fit: slope={slope:.2f}, r={r_value:.2f}, p={p_value:.3f}")
        
            axs[idx].set_xlabel("Distance between initial conditions")
            axs[idx].set_ylabel(f"Angle between {shift_time+1} step of trajectories (deg.)")
            axs[idx].set_title(f"{embed_data_names[idx]}")
            axs[idx].legend(frameon=True, loc='best', fontsize=6)
            
        fig.tight_layout()
        fig.savefig(
            STATE_SPACE_DIR
            / f"initial_condition_distance_vs_angle_{aname}_{shift_time+1}_noise{noise_level}.png",
            dpi=300,
        )
        plt.close(fig)

        return rval_dict, scatter_dict

    rval_dict, scatter_dict = fig4c(shift_time=0)

    # Cleanup to prevent CPU memory compounding across experiments.
    del test_trials_extra
    del Ms_orig, eff_Ms_orig, Ms, eff_Ms, hs
    del embed_data
    gc.collect()

    return (aname, hidden_size, l2_info, rval_dict, scatter_dict,
            fixed_fixation_steps, fixed_fixation_steps * dt_ms)

def run_all():
    all_pt_paths = mpf.list_pt_files("./multiple_tasks", recursive=False)
    pt_paths = _select_target_checkpoints(all_pt_paths)
    target_features = ",".join(TARGET_FEATURE_REG_LAMBDAS)
    if not pt_paths:
        raise FileNotFoundError(
            "No state-space checkpoints matched "
            f"activation={TARGET_ACTIVATION}, projection={TARGET_PROJECTION_DIM}, "
            f"hidden={TARGET_HIDDEN_DIM}, features={target_features}."
        )

    print(
        f"Selected {len(pt_paths)}/{len(all_pt_paths)} state-space checkpoints: "
        f"activation={TARGET_ACTIVATION}, projection={TARGET_PROJECTION_DIM}, "
        f"hidden={TARGET_HIDDEN_DIM}, features={target_features}"
    )
    for path in pt_paths:
        print(f"  {_checkpoint_aname(path)}")

    # This script is the sole producer of top-level state_space result files.
    # The user requested a clean regeneration, so remove every previous file
    # only after confirming that the target cohort is nonempty.
    _clean_state_space_results()

    result_dict = {}
    for netpathname in pt_paths:
        (aname, hidden_size, l2_info, rval_dict, scatter_dict,
         fixed_fixation_steps, fixed_fixation_ms) = eval_one(netpathname)
        result_dict[aname] = {"hidden_size": hidden_size, "l2_info": l2_info,
                              "rval_dict": rval_dict, "scatter": scatter_dict,
                              "fixed_fixation_steps": fixed_fixation_steps,
                              "fixed_fixation_ms": fixed_fixation_ms}
        
    with (STATE_SPACE_DIR / "initial_condition_distance_vs_angle_results.pkl").open("wb") as f:
        pickle.dump(result_dict, f)
        
def summarize():
    with (STATE_SPACE_DIR / "initial_condition_distance_vs_angle_results.pkl").open("rb") as f:
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
    fig.savefig(STATE_SPACE_DIR / "summary_r_values.png", dpi=300)
    plt.close(fig)
    print(f"\nSaved summary plot to {STATE_SPACE_DIR / 'summary_r_values.png'}")

if __name__ == "__main__":    
    run_all()
    summarize()
