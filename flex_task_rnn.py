########################################################
# Test for RNN testing on different set of tasks
########################################################

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
import pickle

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

for _ in range(5): 
    hyp_dict = {}
    
    # Reload modules if changes have been made to them
    from importlib import reload
    
    reload(nets)
    reload(net_helpers)
    
    fixseed = False # randomize setting the seed may lead to not perfectly solved results
    seed = random.randint(1,1000) if not fixseed else 8 # random set the seed to test robustness by default
    print(f"Set seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    hyp_dict['task_type'] = 'multitask' # int, NeuroGym, multitask
    hyp_dict['mode_for_all'] = "random_batch"
    hyp_dict['ruleset'] = 'delaydm1' # low_dim, all, test
    
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
         'fdgofamily': ['fdgo', 'fdanti'],
         'dmsgo': ['dmsgo'],
         'dmcgo': ['dmcgo'], 
         'contextdelaydm1': ['contextdelaydm1'], 
         'delaygo': ['delaygo'],
         'delaydm1': ['delaydm1'], 
         'delaydmfamily': ['delaydm1', 'delaydm2'],
         'simplegofamily': ['fdgo', 'fdanti', 'reactgo', 'reactanti'],
         'gofamily': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti'],
         'gofamily_delaydm': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti', 'delaydm1', 'delaydm2'],
         'dm_family': ['delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm'],
         'go_dm_family': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti', \
                          'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm'],
         'everything': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti', \
                          'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 
                           'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
        }
    
    # only work if adjust_task_prop == False, otherwise will be overwritten
    rules_dict_frequency = {
        'delaygo': np.array([1]),
        'delayfamily': np.array([1,1]), 
        'fdgofamily': np.array([1,1]), 
        'dmsgo': np.array([1]), 
        'dmcgo': np.array([1]),
        'contextdelaydm1': np.array([1]), 
        'delaydm1': np.array([1]), 
        'delaydmfamily': np.array([1,1]),
        'go_dm_family': np.array([1, 1, 1, 1, 1, 1, 
                                  3, 3, 3, 3, 3,
        ]), 
        'everything': np.array([1, 1, 1, 1, 1, 1, 
                                  1, 1, 1, 1, 1,
                                  1, 1, 1, 1
        ])
    }
    
        
    # This can either be used to set parameters OR set parameters and train
    train = True # whether or not to train the network
    verbose = True
    hyp_dict['run_mode'] = 'minimal' # minimal, debug
    hyp_dict['chosen_network'] = "vanilla"
    
    # suffix for saving images
    # inputadd, Wfix, WL2, hL2
    # inputrandom, Wtrain
    # noise001
    # largeregularization
    # trainetalambda
    
    mpn_depth = 1
    n_hidden = 200
    
    hyp_dict['addon_name'] = "L2"
    hyp_dict['addon_name'] += f"+hidden{n_hidden}"
    
    # for coding 
    if hyp_dict['chosen_network'] in ("gru", "vanilla"):
        mpn_depth = 1
    
    def current_basic_params():
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
            'long_delay': 'normal',
            'long_response': 'normal', 
            'adjust_task_prop': True,
            'adjust_task_decay': 0.9, 
        }
    
        assert task_params["fixate_off"], "Accuracy calculation is partially depended on that"
    
        print(f"Fixation_off: {task_params['fixate_off']}; Task_info: {task_params['task_info']}")
    
        train_params = {
            'lr': 1e-2,
            'n_batches': 128,
            'batch_size': 128,
            'gradient_clip': 10,
            'valid_n_batch': 100, 
            'n_datasets': 3000, 
            'valid_check': None, 
            'n_epochs_per_set': 1, 
            'weight_reg': 'L2',
            'activity_reg': 'L2', 
            'reg_lambda': 1e-4,
            
            'scheduler': {
                'type': 'ReduceLROnPlateau',  # or 'StepLR'
                'mode': 'min',                # for ReduceLROnPlateau
                'factor': 0.9,                # factor to reduce LR
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
            'output_bias': False, # Turn off biases for easier interpretation
            'loss_type': 'MSE', # XE, MSE
            'activation': 'tanh', # linear, ReLU, sigmoid, tanh, tanh_re, tukey, heaviside
            'cuda': True,
            'monitor_freq': 1,
            'monitor_valid_out': True, # Whether or not to save validation output throughout training
            'output_matrix': '',# "" (default); "untrained", or "orthogonal"
            'input_layer_add': True, # for MPN
            'input_layer_add_trainable': True, # for MPN, revise this is effectively to [randomize_inputs], tune this
            'recurrent_layer_add': False, # for MPN
            'input_layer_bias': False, # for MPN
            'input_layer': "trainable", # for RNN 
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
            'alpha': 0.8,
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
    
    task_params, train_params, net_params = current_basic_params()
    # add batch information to the parameters
    print("Accuracy Measure: {net_params['acc_measure']}")
    hyp_dict['addon_name'] += f"+batch{train_params['n_batches']}+{net_params['acc_measure']}+rec{net_params['recurrent_layer_add']}+lr{train_params['lr']:.0e}"
    
    # save the setting result
    config = {
        "task_params": task_params, 
        "train_params": train_params, 
        "net_params": net_params,
    }
    
    out_path = Path(f"./flextask/param_rnn_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}_param.json")
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
    
    # In[6]:
    
    test_n_batch = train_params["valid_n_batch"]
    color_by = "stim" # or "resp" 
    
    task_random_fix = True
    if task_random_fix:
        print(f"Align {task_params['rules']} With Same Time")
    
    if task_params['task_type'] in ('multitask',): # Test batch consists of all the rules
        task_params['hp']['batch_size_train'] = test_n_batch
        # using homogeneous cutting off if multiple tasks are presented in the pool
        # if single task, using inhomogeneous cutoff to show diversity & robustness
        # test_mode_for_all = "random" if len(rules_dict[hyp_dict['ruleset']]) > 1 else "random_batch"
        test_mode_for_all = "random"
        # ZIHAN
        # generate test data using "random"
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(task_params, test_n_batch, \
                    rules=task_params['rules'], mode_input=test_mode_for_all, fix=task_random_fix
        )
        _, test_trials, test_rule_idxs = test_trials_extra
        task_params['dataset_name'] = 'multitask'
    
        if task_params['in_out_mode'] in ('low_dim_pos',):
            output_dim_labels = ('Fixate', 'Cos', '-Cos', 'Sin', '-Sin')
        elif task_params['in_out_mode'] in ('low_dim',):
            output_dim_labels = ('Fixate', 'Cos', 'Sin')
        else:
            raise NotImplementedError()
    
        def generate_response_stimulus(task_params, test_trials): 
            """
            """
            labels_resp, labels_stim = [], []
            rules_epochs = {} 
            for rule_idx, rule in enumerate(task_params['rules']):
                print(rule)
                if rule in accept_rules:
                    rules_epochs[rule] = test_trials[rule_idx].epochs
                    if hyp_dict['ruleset'] in ('dmsgo','dmcgo',):
                        labels_resp.append(test_trials[rule_idx].meta['matches'])
                        labels_stim.append(test_trials[rule_idx].meta['stim1']) 
                    else:
                        try: 
                            labels_resp.append(test_trials[rule_idx].meta['resp1'])
                        except Exception as e:
                            labels_resp.append(test_trials[rule_idx].meta['matches'])
                        labels_stim.append(test_trials[rule_idx].meta['stim1']) 
        
                else:
                    raise NotImplementedError()
    
            print(rules_epochs)
            
            labels_resp = np.concatenate(labels_resp, axis=0).reshape(-1,1)
            labels_stim = np.concatenate(labels_stim, axis=0).reshape(-1,1)
    
            return labels_resp, labels_stim, rules_epochs
    
        labels_resp, labels_stim, rules_epochs = generate_response_stimulus(task_params, test_trials)
    
    
    labels = labels_stim if color_by == "stim" else labels_resp
        
    test_input, test_output, _ = test_data
    print(f"test_input.device: {test_input.device}")
    
    permutation = np.random.permutation(test_input.shape[0])
    test_input = test_input[permutation]
    test_output = test_output[permutation]
    # test_mask = test_mask[permutation]
    labels = labels[permutation]
    
    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()
    
    del test_output
    
    # Total number of batches, might be different than test_n_batch
    # this should be the same regardless of variety of test_input
    n_batch_all = test_input_np.shape[0] 
    
    def find_task(task_params, test_input_np, shift_index):
        """
        """
        test_task = [] # which task
        for batch_idx in range(test_input_np.shape[0]):
            
            if task_params["randomize_inputs"]: 
                test_input_np_ = test_input_np @ np.linalg.pinv(task_params["randomize_matrix"])
            else: 
                test_input_np_ = test_input_np
                
            task_label = test_input_np_[batch_idx, 0, 6-shift_index:]
            
            task_label = np.asarray(task_label)       
            dist = np.abs(task_label - 1)     
            mask = dist == dist.min() 
            
            indices = np.where(mask)[0]
            
            if indices.size:                
                task_label_index = indices[0]   
            else:
                raise ValueError("No entry close enough to 1 found")
                
            test_task.append(task_label_index)
    
        return test_task  
    
    test_task = find_task(task_params, test_input_np, shift_index)
    
    # actual fitting
    # we use net at different training stage on the same test_input
    net, _, (counter_lst, netout_lst, db_lst, Winput_lst, Winputbias_lst,\
             Woutput_lst, Wall_lst, marker_lst, loss_lst, acc_lst) = net_helpers.train_network(params, device=device, verbose=verbose,\
                                                                                               train=train, hyp_dict=hyp_dict,\
                                                                                               netFunction=netFunction,\
                                                                                               test_input=[test_input])
    
    if hyp_dict['chosen_network'] == "dmpn":
        if net_params["input_layer_add"]:
            counter_lst = [x * epoch_multiply + 1 for x in counter_lst] # avoid log plot issue    
            fignorm, axsnorm = plt.subplots(1,1,figsize=(4,4))
            axsnorm.plot(counter_lst, [np.linalg.norm(Winput_matrix) for Winput_matrix in Winput_lst], "-o")
            axsnorm.set_xscale("log")
            axsnorm.set_ylabel("Frobenius Norm")
    
    # sanity check, if W_freeze, then the recorded W matrix for the modulation layer should not be changed
    if net_params["ml_params"]["W_freeze"]: 
        assert np.allclose(Wall_lst[-1][0], Wall_lst[0][0])
    
    if net_params["input_layer_bias"]: 
        assert net_params["input_layer_add"] is True 
    
    if train:
        fig, ax = plt.subplots(1,1,figsize=(3,3))
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
        
        # ax.set_yscale('log')
        ax.legend()
        # ax.set_ylim([0.5, 1.05])
        ax.set_ylim([0, 1])
        # ax.set_ylabel('Loss ({})'.format(net.loss_type))
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('# Batches')
        plt.savefig(f"./flextask/loss_rnn_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png", dpi=100)
        
    print('Done!')
    
    if train:
        net_helpers.net_eta_lambda_analysis(net, net_params, hyp_dict)

    # get network output
    ind = len(marker_lst)-1 
    net_out = netout_lst[0][ind]
    
    def plot_input_output(test_input_np, net_out, test_output_np, test_task=None, tag="", batch_num=5):
        """
        """
        test_input_np = helper.to_ndarray(test_input_np)
        net_out = helper.to_ndarray(net_out)
        test_output_np = helper.to_ndarray(test_output_np)
        
        fig_all, axs_all = plt.subplots(batch_num,2,figsize=(4*2,batch_num*2))
        
        if test_output_np.shape[-1] == 1:
            for batch_idx, ax in enumerate(axs):
                ax.plot(net_out[batch_idx, :, 0], color=c_vals[batch_idx])
                ax.plot(test_output_np[batch_idx, :, 0], color=c_vals_l[batch_idx])
        
        else:
            for batch_idx in range(batch_num):
                for out_idx in range(test_output_np.shape[-1]):
                    axs_all[batch_idx,0].plot(net_out[batch_idx, :, out_idx], color=c_vals[out_idx], label=out_idx)
                    axs_all[batch_idx,0].plot(test_output_np[batch_idx, :, out_idx], color=c_vals_l[out_idx], linewidth=5, alpha=0.3)
                    if test_task is not None: 
                        axs_all[batch_idx,0].set_title(f"{task_params['rules'][test_task[batch_idx]]}")
        
                input_batch = test_input_np[batch_idx,:,:]
                if task_params["randomize_inputs"]: 
                    input_batch = input_batch @ np.linalg.pinv(task_params["randomize_matrix"])
                for inp_idx in range(input_batch.shape[-1]):
                    axs_all[batch_idx,1].plot(input_batch[:,inp_idx], color=c_vals[inp_idx], label=inp_idx, alpha=1.0)
                    if test_task is not None: 
                        axs_all[batch_idx,1].set_title(f"{task_params['rules'][test_task[batch_idx]]}")
    
        for ax in axs_all.flatten(): 
            ax.set_ylim([-2, 2])
        fig_all.tight_layout()
        fig_all.savefig(f"./flextask/lowD_rnn_{hyp_dict['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}_{tag}.png", dpi=100)
    
    plot_input_output(test_input_np, net_out, test_output_np, test_task, tag="", batch_num=20 if len(rules_dict[hyp_dict['ruleset']]) > 1 else 10)
    
    loss_dict = {
        "batch_idx": net.hist['iters_monitor'][1:], 
        "training_acc": net.hist['train_acc'][1:], 
        "validation_acc": net.hist['valid_acc'][1:]
    }
    
    loss_name = f"./flextask/loss_rnn_{hyp_dict['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}.pkl"
    
    with open(loss_name, "wb") as f:
        pickle.dump(loss_dict, f)