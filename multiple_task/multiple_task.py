#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic Libraries
import sys
import time
import gc
import random
import copy 
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json 
from pathlib import Path 

# PyTorch Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Data Handling and Image Processing
from torchvision import datasets, transforms

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim
import h5py
from hdf5plugin import Blosc 

# Style for Matplotlib
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

# Scientific Computing and Machine Learning
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import networks as nets  # Contains RNNs
import net_helpers
import mpn_tasks
import helper
import mpn
import clustering

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

# Memory Optimization
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',] * 10
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',] * 10
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',] * 10 
l_vals = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
markers_vals = ['o', 'v', '*', '+', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
linestyles = ["-", "--", "-."]

# ─── Experiment-wide configuration ───────────────────────────────────────────
# Number of independent trials (different random seeds) to train, under the SAME
# parameter setting. Mirrors two_task.py so the multi-task net can be trained
# many times for robustness across seeds.
N_TRIALS = 5
# Fixed list of seeds, or None to draw N_TRIALS random seeds (fresh entropy each
# run, so reruns explore different seeds).
SEED_LIST = None

RULESET = 'everything'          # low_dim, all, test, everything, ...
CHOSEN_NETWORK = "dmpn"         # mpn1, dmpn, vanilla, gru
N_HIDDEN = 300
ADDON_NAME = "L21e2"            # +hidden{N_HIDDEN}+batch{n_batches}+{acc} appended below
train = True                    # whether or not to train the network
verbose = True

# Reload modules if changes have been made to them
from importlib import reload

reload(nets)
reload(net_helpers)

accept_rules = ('fdgo', 'fdanti', 'delaygo', 'delayanti', 'reactgo', 'reactanti',
                'delaydm1', 'delaydm2', 'dmsgo', 'dmcgo', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsnogo', 'dmcnogo')

rules_dict = \
    {'all' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
     'low_dim' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
     'delayfamily': ['delaygo', 'delayanti'], 
     'dmsgo': ['dmsgo'],
     'dmcgo': ['dmcgo'], 
     'contextdelaydm1': ['contextdelaydm1'], 
     'delaygo': ['delaygo'],
     'delaydm1': ['delaydm1'], 
     'simplegofamily': ['fdgo', 'fdanti', 'reactgo', 'reactanti'],
     'gofamily': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti'],
     'gofamily_delaydm': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti', 'delaydm1', 'delaydm2'],
     'dm_family': ['delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm'],
     'go_dm_family': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti', \
                      'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm'],
     'everything': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti', 
                    'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 
                    'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
    }

# only work if adjust_task_prop == False, otherwise will be overwritten
rules_dict_frequency = {
    'delaygo': np.array([1]),
    'dmsgo': np.array([1]), 
    'dmcgo': np.array([1]),
    'contextdelaydm1': np.array([1]), 
    'delaydm1': np.array([1]), 
    'go_dm_family': np.array([1, 1, 1, 1, 1, 1, 
                              3, 3, 3, 3, 3,
    ]), 
    'everything': np.array([1, 1, 1, 1, 1, 1, 
                              1, 1, 1, 1, 1,
                              1, 1, 1, 1
    ])
}

    

mpn_depth = 1
n_hidden = N_HIDDEN

# for coding
if CHOSEN_NETWORK in ("gru", "vanilla"):
    mpn_depth = 1


def current_basic_params(hyp_dict):
    task_params = {
        'task_type': hyp_dict['task_type'],
        'rules': rules_dict[hyp_dict['ruleset']],
        'rules_probs': rules_dict_frequency[hyp_dict['ruleset']], 
        'dt': 40, # ms, directly influence sequence lengths,
        'ruleset': hyp_dict['ruleset'],
        'n_eachring': 8, # Number of distinct possible inputs on each ring
        'in_out_mode': 'low_dim',  # high_dim or low_dim or low_dim_pos (Robert vs. Laura's paper, resp)
        'sigma_x': 0.00, # Laura raised to 0.1 to prevent overfitting (Robert uses 0.01)
        'mask_type': 'cost', # 'cost', None
        'fixate_off': True, # Second fixation signal goes on when first is off
        'task_info': True, 
        'randomize_inputs': False,
        'n_input': 20, # Only used if inputs are randomized,
        'modality_diff': True,
        'label_strength': True, 
        'long_stimulus': 'normal', 
        'long_fixation': 'normal', 
        'long_delay': 'normal',
        'long_response': 'normal',
        'adjust_task_prop': True,
        'adjust_task_decay': 0.9, 
    }

    assert task_params["fixate_off"], "Accuracy calculation is partially depended on that"

    print(f"Fixation_off: {task_params['fixate_off']}; Task_info: {task_params['task_info']}")

    train_params = {
        'lr': 1e-3,
        'n_batches': 128,
        'batch_size': 128,
        'gradient_clip': 10,
        'valid_n_batch': min(max(15, int(200/len(rules_dict[hyp_dict['ruleset']]))), 50),
        'n_datasets': 100000,
        'valid_check': None, 
        'n_epochs_per_set': 1, 
        'weight_reg': 'L2',
        'activity_reg': 'L2', 
        'reg_lambda': 1e-2,
        
        'scheduler': {
            'type': 'ReduceLROnPlateau',  # or 'StepLR'
            'mode': 'min',                # for ReduceLROnPlateau
            'factor': 0.95,                # factor to reduce LR
            'patience': 30,                # epochs to wait before reducing LR
            'min_lr': 1e-8,
            'step_size': 30,              # for StepLR (step every 30 datasets)
            'gamma': 0.1                  # for StepLR (multiply LR by 0.1)
        },
    }

    print(f"valid_n_batch: {train_params['valid_n_batch']}")

    if not train: # some 
        assert train_params['n_epochs_per_set'] == 0

    net_params = {
        'net_type': hyp_dict['chosen_network'], # mpn1, dmpn, vanilla
        'n_neurons': [1] + [n_hidden] * mpn_depth + [1],
        'linear_embed': n_hidden, 
        'output_bias': False, # Turn off biases for easier interpretation
        'loss_type': 'MSE', # XE, MSE
        'activation': 'tanh', # linear, ReLU, sigmoid, tanh, tanh_re, tukey, heaviside
        'cuda': True,
        'monitor_freq': train_params["n_epochs_per_set"],
        'monitor_valid_out': True, # Whether or not to save validation output throughout training
        'output_matrix': '',# "" (default); "untrained", or "orthogonal"
        'input_layer_add': True, 
        'input_layer_add_trainable': True, # revise this is effectively to [randomize_inputs], tune this
        'input_layer_bias': False, 
        'input_layer': "trainable", # for RNN only
        'acc_measure': 'angle', 
        
        # for one-layer MPN, GRU or Vanilla
        'ml_params': {
            'bias': True, # Bias of layer
            'mp_type': 'mult',
            'm_update_type': 'hebb_assoc', # hebb_assoc, hebb_pre
            'eta_type': 'scalar', # scalar, pre_vector, post_vector, matrix
            'eta_train': True,
            # 'eta_init': 'mirror_gaussian', #0.0,
            'lam_type': 'scalar', # scalar, pre_vector, post_vector, matrix
            'm_time_scale': 4000, # ms, sets lambda
            'lam_train': False,
            'W_freeze': False, # different combination with [input_layer_add_trainable]
        },

        # Vanilla RNN params
        'leaky': True,
        'alpha': 0.2,
    }

    # Ensure the two options are *not* activated at the same time
    assert not (task_params["randomize_inputs"] and net_params["input_layer_add"]), (
        "task_params['randomize_inputs'] and net_params['input_layer_add'] cannot both be True."
    )

    # for multiple MPN layers, assert 
    if mpn_depth > 1:
        for mpl_idx in range(mpn_depth - 1):
            assert f'ml_params{mpl_idx}' in net_params.keys()

    # actually I don't think it is needed
    # putting here to warn the parameter checking every time 
    # when switching network
    if hyp_dict['chosen_network'] in ("gru", "vanilla"):
        assert f'ml_params' in net_params.keys()

    return task_params, train_params, net_params


def run_trial(seed):
    """Train one multi-task network under the shared parameter setting and save
    its checkpoint + result traces (flat layout under ./multiple_tasks/, parsed
    by multiple_task_analysis.py via the aname identifier)."""
    print(f"\n{'='*70}\nTrial seed = {seed}\n{'='*70}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    hyp_dict = {
        'task_type': 'multitask',          # int, NeuroGym, multitask
        'mode_for_all': "random_batch",
        'ruleset': RULESET,
        'run_mode': 'minimal',             # minimal, debug
        'chosen_network': CHOSEN_NETWORK,
        'addon_name': ADDON_NAME + f"+hidden{N_HIDDEN}",
    }

    task_params, train_params, net_params = current_basic_params(hyp_dict)
    # add batch information to the parameters
    print(f"Accuracy Measure: {net_params['acc_measure']}")
    hyp_dict['addon_name'] += f"+batch{train_params['n_batches']}+{net_params['acc_measure']}"

    # save the setting result
    config = {
        "task_params": task_params,
        "train_params": train_params,
        "net_params": net_params,
    }

    out_path = Path(f"./multiple_tasks/param_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}_param.json")
    with out_path.open("w") as f:
        json.dump(config, f, indent=4, default=helper.as_jsonable)

    shift_index = 1 if not task_params['fixate_off'] else 0

    if hyp_dict['task_type'] in ('multitask',):
        task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
            (task_params, train_params, net_params)
        )

        net_params['prefs'] = mpn_tasks.get_prefs(task_params['hp'])

        print('Rules: {}'.format(task_params['rules']))
        print('  Input size {}, Output size {}'.format(
            task_params['n_input'], task_params['n_output'],
        ))
    else:
        raise NotImplementedError()

    if net_params['cuda']:
        print('Using CUDA...')
        device = torch.device('cuda')
    else:
        print('Using CPU...')
        device = torch.device('cpu')

    # how many epoch each dataset will be trained on
    epoch_multiply = train_params["n_epochs_per_set"]

    params = task_params, train_params, net_params

    if net_params['net_type'] == 'mpn1':
        netFunction = mpn.MultiPlasticNet
    elif net_params['net_type'] == 'dmpn':
        netFunction = mpn.DeepMultiPlasticNet
    elif net_params['net_type'] == 'vanilla':
        netFunction = nets.VanillaRNN
    elif net_params['net_type'] == 'gru':
        netFunction = nets.GRU

    test_n_batch = train_params["valid_n_batch"]
    color_by = "stim"  # or "resp"

    if task_params['task_type'] in ('multitask',):  # Test batch consists of all the rules
        task_params['hp']['batch_size_train'] = test_n_batch
        test_mode_for_all = "random"
        # generate test data using "random"
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(task_params,
                                                                      test_n_batch,
                                                                      rules=task_params['rules'],
                                                                      mode_input=test_mode_for_all)

        _, test_trials, test_rule_idxs = test_trials_extra
        task_params['dataset_name'] = 'multitask'

        if task_params['in_out_mode'] in ('low_dim_pos',):
            output_dim_labels = ('Fixate', 'Cos', '-Cos', 'Sin', '-Sin')
        elif task_params['in_out_mode'] in ('low_dim',):
            output_dim_labels = ('Fixate', 'Cos', 'Sin')
        else:
            raise NotImplementedError()

        labels_resp, labels_stim, labels_stim2, rules_epochs = helper.generate_response_stimulus(task_params, test_trials)

    labels = labels_stim if color_by == "stim" else labels_resp

    test_input, test_output, _ = test_data
    print(f"test_input.device: {test_input.device}")

    permutation = np.random.permutation(test_input.shape[0])
    test_input = test_input[permutation]
    test_output = test_output[permutation]
    labels = labels[permutation]

    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()

    del test_output

    # Total number of batches, might be different than test_n_batch
    n_batch_all = test_input_np.shape[0]

    test_task = helper.find_task(task_params, test_input_np, shift_index)

    # actual fitting: use net at different training stages on the same test_input
    net, _, (counter_lst, netout_lst, db_lst, Winput_lst, Winputbias_lst,
             Woutput_lst, Wall_lst, marker_lst, loss_lst, acc_lst), _ = net_helpers.train_network(
        params, device=device, verbose=verbose, train=train, hyp_dict=hyp_dict,
        netFunction=netFunction, test_input=[test_input], print_frequency=100)

    if hyp_dict['chosen_network'] == "dmpn":
        if net_params["input_layer_add"]:
            counter_lst = [x * epoch_multiply + 1 for x in counter_lst]  # avoid log plot issue
            fignorm, axsnorm = plt.subplots(1, 1, figsize=(4, 4))
            axsnorm.plot(counter_lst, [np.linalg.norm(Winput_matrix) for Winput_matrix in Winput_lst], "-o")
            axsnorm.set_xscale("log")
            axsnorm.set_ylabel("Frobenius Norm")
            plt.close(fignorm)

    # sanity check: if W_freeze, the recorded W for the modulation layer is unchanged
    if net_params["ml_params"]["W_freeze"]:
        assert np.allclose(Wall_lst[-1][0], Wall_lst[0][0])

    if net_params["input_layer_bias"]:
        assert net_params["input_layer_add"] is True

    if train:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.plot(net.hist['iters_monitor'][1:], net.hist['train_acc'][1:],
                color=c_vals[0], label='Full train accuracy')
        ax.plot(net.hist['iters_monitor'][1:], net.hist['valid_acc'][1:],
                color=c_vals[1], label='Full valid accuracy')
        if net.weight_reg is not None:
            ax.plot(net.hist['iters_monitor'], net.hist['train_loss_output_label'],
                    color=c_vals_l[0], zorder=-1, label='Output label')
            ax.plot(net.hist['iters_monitor'], net.hist['train_loss_reg_term'],
                    color=c_vals_l[0], zorder=-1, label='Reg term', linestyle='dashed')
            ax.plot(net.hist['iters_monitor'], net.hist['valid_loss_output_label'],
                    color=c_vals_l[1], zorder=-1, label='Output valid label')
            ax.plot(net.hist['iters_monitor'], net.hist['valid_loss_reg_term'],
                    color=c_vals_l[1], zorder=-1, label='Reg valid term', linestyle='dashed')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('# Batches')
        plt.savefig(f"./multiple_tasks/loss_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png", dpi=100)
        plt.close(fig)

    print('Done!')

    if train:
        net_helpers.net_eta_lambda_analysis(net, net_params, hyp_dict)

    # final-stage outputs (use the last recorded training stage)
    ind = len(marker_lst) - 1
    network_at_percent = (marker_lst[ind] + 1) / train_params['n_datasets'] * 100
    print(f"Using network at {network_at_percent}%")
    net_out = netout_lst[0][ind]
    db = db_lst[0][ind]
    W_output = Woutput_lst[ind]
    W_ = Wall_lst[ind][0]

    def plot_input_output(test_input_np, net_out, test_output_np, test_task=None, tag="", batch_num=5, savefig=True):
        test_input_np = helper.to_ndarray(test_input_np)
        net_out = helper.to_ndarray(net_out)
        test_output_np = helper.to_ndarray(test_output_np)

        fig_all, axs_all = plt.subplots(batch_num, 2, figsize=(4 * 2, batch_num * 2))

        if test_output_np.shape[-1] == 1:
            for batch_idx, ax in enumerate(axs_all):
                ax.plot(net_out[batch_idx, :, 0], color=c_vals[batch_idx])
                ax.plot(test_output_np[batch_idx, :, 0], color=c_vals_l[batch_idx])
        else:
            for batch_idx in range(batch_num):
                for out_idx in range(test_output_np.shape[-1]):
                    axs_all[batch_idx, 0].plot(net_out[batch_idx, :, out_idx], color=c_vals[out_idx], label=out_idx)
                    axs_all[batch_idx, 0].plot(test_output_np[batch_idx, :, out_idx], color=c_vals_l[out_idx], linewidth=5, alpha=0.3)
                    if test_task is not None:
                        axs_all[batch_idx, 0].set_title(f"{task_params['rules'][test_task[batch_idx]]}")

                input_batch = test_input_np[batch_idx, :, :]
                if task_params["randomize_inputs"]:
                    input_batch = input_batch @ np.linalg.pinv(task_params["randomize_matrix"])
                for inp_idx in range(input_batch.shape[-1]):
                    axs_all[batch_idx, 1].plot(input_batch[:, inp_idx], color=c_vals[inp_idx], label=inp_idx, alpha=1.0)
                    if test_task is not None:
                        axs_all[batch_idx, 1].set_title(f"{task_params['rules'][test_task[batch_idx]]}")

        for ax in axs_all.flatten():
            ax.set_ylim([-2, 2])
        fig_all.tight_layout()

        if savefig:
            fig_all.savefig(f"./multiple_tasks/lowD_{hyp_dict['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}_{tag}.png", dpi=100)
        return fig_all, axs_all

    fig_all, axs_all = plot_input_output(test_input_np, net_out, test_output_np, test_task, tag="",
                                         batch_num=20 if len(rules_dict[hyp_dict['ruleset']]) > 1 else 10)
    plt.close(fig_all)

    # db is selected based on learning-stage selection
    layer_index = 0  # 1 layer MPN
    if net_params["input_layer_add"]:
        layer_index += 1

    max_seq_len = test_input.shape[1]

    def modulation_extraction(db, max_seq_len, layer_index):
        n_batch_all_ = test_input.shape[0]
        Ms = np.concatenate((db[f'M{layer_index}'].reshape(n_batch_all_, max_seq_len, -1),), axis=-1)
        Ms_orig = np.concatenate((db[f'M{layer_index}'],), axis=-1)
        bs = np.concatenate((db[f'b{layer_index}'],), axis=-1)
        hs = np.concatenate((db[f'hidden{layer_index}'].reshape(n_batch_all_, max_seq_len, -1),), axis=-1)
        xs = np.concatenate((db[f'input{layer_index}'].reshape(n_batch_all_, max_seq_len, -1),), axis=-1)
        return Ms, Ms_orig, hs, bs, xs

    print(f"rules_epochs: {rules_epochs}")
    all_rules = np.array(task_params["rules"])
    print(f"all_rules: {all_rules}")
    test_task = np.array(test_task)
    print(f"test_task: {test_task}")

    Ms, Ms_orig, hs, bs, xs = modulation_extraction(db_lst[0][-1], max_seq_len, layer_index)
    print(f"Ms.shape:{Ms.shape}")
    print(f"Ms_orig.shape:{Ms_orig.shape}")
    print(f"hs.shape:{hs.shape}")
    print(f"bs.shape:{bs.shape}")
    print(f"xs.shape:{xs.shape}")

    # save result traces
    pathname = f"./multiple_tasks/param_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}_result.npz"
    np.savez_compressed(pathname,
                        rules_epochs=rules_epochs,
                        hyp_dict=hyp_dict,
                        all_rules=all_rules,
                        test_task=test_task,
                        Ms_orig=Ms_orig,
                        hs=hs,
                        bs=bs,
                        xs=xs)

    # save the network information
    netpathname = f"./multiple_tasks/savednet_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.pt"
    save_dict = {
        "state_dict": net.state_dict(),  # trained result
        "net_params": net_params,        # network parameter
    }
    torch.save(save_dict, netpathname)
    print("Network parameter saving is done")

    aname = f"{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}"

    # Free this trial's large objects before returning. The __main__ loop also
    # runs a finally-block cleanup (figures + gc + CUDA cache) on every trial, so
    # memory does not compound from trial to trial. Here we additionally drop the
    # big per-trial tensors/arrays explicitly. Move net off-GPU first so its
    # parameters' device memory is released promptly.
    try:
        net.to("cpu")
    except Exception:
        pass
    del net, db, db_lst, netout_lst, Wall_lst, Woutput_lst, Winput_lst, Winputbias_lst
    del net_out, W_output, W_, Ms, Ms_orig, hs, bs, xs
    del test_input, test_input_np, test_output_np, test_data, test_trials_extra
    plt.close("all")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return aname


# ─── Run K independent trials (same params, different seeds) ──────────────────
if __name__ == "__main__":
    Path("multiple_tasks").mkdir(parents=True, exist_ok=True)

    if SEED_LIST is not None:
        seeds = list(SEED_LIST)
    else:
        rng = random.Random()  # fresh entropy each run -> different seed pool
        seeds = rng.sample(range(1, 1000), N_TRIALS)

    print(f"Running {len(seeds)} independent trials: seeds={seeds}")
    anames = []
    for seed in seeds:
        try:
            anames.append(run_trial(seed))
        except Exception as exc:
            print(f"Trial seed={seed} FAILED: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset per-trial memory so cost does NOT compound across trials and
            # a long run won't OOM. Runs on BOTH success and failure:
            #  - plt.close("all"): drop every matplotlib figure (incl. any made by
            #    net_eta_lambda_analysis) from pyplot's global registry.
            #  - gc.collect(): break ref cycles and release a failed trial's frame
            #    locals (the traceback that held `net`/GPU tensors is cleared once
            #    the except clause exits, before this finally runs).
            #  - empty_cache / ipc_collect: return freed blocks to the CUDA driver.
            plt.close("all")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    print(f"\nCompleted {len(anames)}/{len(seeds)} trials.")
    for a in anames:
        print(f"  {a}")

    # Manifest of the runs produced THIS invocation (mirrors two_task.py), so a
    # pipeline can analyze exactly what was just trained.
    manifest_path = Path("multiple_tasks") / "last_run_anames.txt"
    with manifest_path.open("w") as mf:
        mf.write("\n".join(anames) + ("\n" if anames else ""))
    print(f"Wrote manifest: {manifest_path}")












