#!/usr/bin/env python
# coding: utf-8
"""
Pretraining → Post-training transfer experiment.

Tests whether an MPN's Hebbian plasticity can support learning a new task
when all network parameters are frozen except the task-indicator input column.

Protocol:
  Stage 1 (Pretraining):  Train a DeepMultiPlasticNet (200 hidden units) on
                           a pair of tasks (e.g. fdgo + delaygo) until
                           convergence. The input layer W_initial_linear is
                           created with one extra column (pretraining_shift_pre=1)
                           to reserve space for the post-training task indicator;
                           this column receives zeros during Stage 1. All
                           parameters are trainable.

  Stage 2 (Post-training): Load the pretrained network and freeze ALL
                           parameters via expand_and_freeze(option=1). A
                           gradient hook restricts learning to only the last
                           column of W_initial_linear — the task-indicator
                           input weights. The held-out task (e.g. delayanti)
                           is trained using input data zero-padded in the
                           Stage 1 task-indicator slots (pretraining_shift=2).
                           The plasticity matrix M still evolves within-trial
                           via the Hebbian rule (eta, lam), but eta, lam, the
                           static recurrent weight W, and the output layer
                           W_output are all frozen.

The experiment is repeated over multiple random seeds (outer loop) to
assess robustness. For each seed the script saves:
  - Network checkpoint (.pt)
  - Hyperparameters (.json)
  - Hidden states, modulation matrices, and activations for both stages (.npz)
  - Validation outputs at each stage (.npz)
  - Diagnostic figures (input weight comparison, loss curves, sample I/O)

All outputs are written to ./pretraining/.
"""

# In[1]:


# Basic Libraries
import sys
import time
import gc
import random
import copy
import pickle
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

import matplotlib as mpl 
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

hyp_dict = {}

from importlib import reload

reload(nets)
reload(net_helpers)

for _ in range(5): 
    seed = random.randint(1,1000)
    print(f"Set seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    hyp_dict['task_type'] = 'multitask'
    hyp_dict['mode_for_all'] = "random_batch"
    hyp_dict['ruleset'] = 'fdanti_delaygo'

    accept_rules = ('fdgo', 'fdanti', 'delaygo', 'delayanti', 'reactgo', 'reactanti', 
                    'delaydm1', 'delaydm2', 'dmsgo', 'dmcgo', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm', 'dmsnogo', 'dmcnogo')

    rules_dict = {
        'fdgo_delaygo': ['fdgo', 'delaygo'], 
        'fdanti_delaygo': ['fdanti', 'delaygo'],
        'delayanti': ['delayanti']

    }

    # only work if adjust_task_prop == False, otherwise will be overwritten
    rules_dict_frequency = {
        'fdgo_delaygo': np.array([1,1]), 
        'fdanti_delaygo': np.array([1,1]), 
        'delayanti': np.array([1])
    }


    # This can either be used to set parameters OR set parameters and train
    train = True # whether or not to train the network
    verbose = True
    hyp_dict['run_mode'] = 'minimal' # minimal, debug
    hyp_dict['chosen_network'] = "dmpn"

    mpn_depth = 1
    n_hidden = 100

    hyp_dict['addon_name'] = ""
    hyp_dict['addon_name'] += f"+hidden{n_hidden}+L21e4"

    # for coding 
    if hyp_dict['chosen_network'] in ("gru", "vanilla"):
        mpn_depth = 1

    def current_basic_params(hyp_dict_input):
        task_params = {
            'task_type': hyp_dict_input['task_type'],
            'rules': rules_dict[hyp_dict_input['ruleset']],
            'rules_probs': rules_dict_frequency[hyp_dict_input['ruleset']], 
            'dt': 40, # ms, directly influence sequence lengths,
            'ruleset': hyp_dict_input['ruleset'],
            'n_eachring': 8, # Number of distinct possible inputs on each ring
            'in_out_mode': 'low_dim',  # high_dim or low_dim or low_dim_pos (Robert vs. Laura's paper, resp)
            'sigma_x': 0.00, # Laura raised to 0.1 to prevent overfitting (Robert uses 0.01)
            'mask_type': 'cost', # 'cost', None
            'fixate_off': True, # Second fixation signal goes on when first is off
            'task_info': True, 
            'randomize_inputs': False, # outdated 
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
            # 'valid_n_batch': min(max(50, int(200/len(rules_dict[hyp_dict_input['ruleset']]))), 50),
            'valid_n_batch': 200, 
            'n_datasets': 60000, 
            'valid_check': 600, 
            'pretrain_min': 1000, 
            'n_epochs_per_set': 1,  
            'weight_reg': 'L2',
            'activity_reg': 'L2', 
            'reg_lambda': 1e-4,
            
            'scheduler': {
                'type': 'ReduceLROnPlateau',  # or 'StepLR'
                'mode': 'min',                # for ReduceLROnPlateau
                'factor': 0.9,                # factor to reduce LR
                'patience': 20,                # epochs to wait before reducing LR
                'min_lr': 1e-8,
                'step_size': 30,              # for StepLR (step every 30 datasets)
                'gamma': 0.1                  # for StepLR (multiply LR by 0.1)
            },
        }

        print(f"valid_n_batch: {train_params['valid_n_batch']}")

        if not train: # some 
            assert train_params['n_epochs_per_set'] == 0

        net_params = {
            'net_type': hyp_dict_input['chosen_network'], # mpn1, dmpn, vanilla
            'n_neurons': [1] + [n_hidden] * mpn_depth + [1],
            'linear_embed': n_hidden, 
            'output_bias': False, 
            'hidden_bias': False, 
            'input_bias': False, 
            'loss_type': 'MSE', # XE, MSE
            'activation': 'tanh', # linear, ReLU, sigmoid, tanh, tanh_re, tukey, heaviside
            'W_rec_init': 'diag', 
            'W_rec_diag_scale': 0.8, 
            'cuda': True,
            'monitor_freq': 1,
            'monitor_valid_out': True, # Whether or not to save validation output throughout training
            'output_matrix': '', # "" (default); "untrained", or "orthogonal"
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

        # 2025-11-16: make sure the input bias control are consistent between vanilla RNN and dmpn
        assert net_params["input_bias"] == net_params["input_layer_bias"]

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
        if hyp_dict_input['chosen_network'] in ("gru", "vanilla"):
            assert f'ml_params' in net_params.keys()

        return task_params, train_params, net_params

    hyp_dict_old = copy.deepcopy(hyp_dict)

    task_params, train_params, net_params = current_basic_params(hyp_dict_old)

    print(f"Accuracy Measure: {net_params['acc_measure']}")

    # 2025-11-19: this part should either have "+L2" or not 
    hyp_dict['addon_name'] += f"+batch{train_params['n_batches']}+{net_params['acc_measure']}"
    hyp_dict_old['addon_name'] += f"+batch{train_params['n_batches']}+{net_params['acc_measure']}"

    hyp_dict['ruleset'] = 'delayanti'
    task_params2, train_params2, net_params2 = current_basic_params(hyp_dict)

    # save the setting result
    config = {
        "task_params": task_params, 
        "train_params": train_params, 
        "net_params": net_params,
    }

    out_path = Path(f"./pretraining/param_{hyp_dict_old['ruleset']}_seed{seed}_{hyp_dict['addon_name']}_param.json")
    with out_path.open("w") as f: 
        json.dump(config, f, indent=4, default=helper.as_jsonable)

    shift_index = 1 if not task_params['fixate_off'] else 0

    if hyp_dict['task_type'] in ('multitask',):
        task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
            (task_params, train_params, net_params)
        )
        task_params2, train_params2, net_params2 = mpn_tasks.convert_and_init_multitask_params(
            (task_params2, train_params2, net_params2)
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

    # adjust the training information
    train_params2["n_datasets"] = 80000 
    train_params2['n_epochs_per_set'] = 1
    # net_params2['acc_measure'] = "angle"

    # In[5]:
    params = task_params, train_params, net_params
    params2 = task_params2, train_params2, net_params2

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

    # how much the second stage input should be shifted/paddled 
    # zero-paddle to the training data of post-training; pretraining_shift = number of pre-training task
    pretraining_shift = len(task_params['rules']) 
    # zero-paddle to the training data of pre-training; pretraining_shift_pre = number of post-training task = 1
    pretraining_shift_pre = len(task_params2['rules'])
    # this should be 1, since we always pre-training on multiple and test on 1
    # hard-coded for simplicity
    assert pretraining_shift_pre == 1 

    if task_params['task_type'] in ('multitask',): # Test batch consists of all the rules
        task_params['hp']['batch_size_train'] = test_n_batch
        task_params2['hp']['batch_size_train'] = test_n_batch
        # using homogeneous cutting off if multiple tasks are presented in the pool
        # if single task, using inhomogeneous cutoff to show diversity & robustness
        # test_mode_for_all = "random" if len(rules_dict[hyp_dict['ruleset']]) > 1 else "random_batch"
        test_mode_for_all = "random"
        # ZIHAN
        # generate test data using "random"
        task_params_test = copy.deepcopy(task_params)
        long_response_change = "normal"
        task_params_test["long_response"] = long_response_change
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(task_params_test, test_n_batch,
                                                                    rules=task_params_test['rules'], mode_input=test_mode_for_all,
                                                                    pretraining_shift_pre=pretraining_shift_pre)

        # Oct 15th: make the response period to be longer
        # so that the hidden activity analysis might be more reliable
        task_params2_test = copy.deepcopy(task_params2)
        task_params2_test["long_response"] = long_response_change
        test_data2, test_trials_extra2 = mpn_tasks.generate_trials_wrap(task_params2_test, test_n_batch,
                                                                        rules=task_params2_test['rules'], mode_input=test_mode_for_all,
                                                                        pretraining_shift=pretraining_shift)
        _, test_trials, test_rule_idxs = test_trials_extra
        _, test_trials2, test_rule_idxs2 = test_trials_extra2
        
        task_params['dataset_name'] = 'multitask'
        task_params2['dataset_name'] = 'multitask'

        if task_params['in_out_mode'] in ('low_dim_pos',):
            output_dim_labels = ('Fixate', 'Cos', '-Cos', 'Sin', '-Sin')
        elif task_params['in_out_mode'] in ('low_dim',):
            output_dim_labels = ('Fixate', 'Cos', 'Sin')
        else:
            raise NotImplementedError()

        def generate_response_stimulus(task_params, test_trials, hyp_dict_input): 
            """
            """
            labels_resp, labels_stim = [], []
            rules_epochs = {} 
            for rule_idx, rule in enumerate(task_params['rules']):
                print(rule)
                if rule in accept_rules:
                    rules_epochs[rule] = test_trials[rule_idx].epochs
                    if hyp_dict_input['ruleset'] in ('dmsgo','dmcgo',):
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

        labels_resp, labels_stim, rules_epochs = generate_response_stimulus(task_params, test_trials, hyp_dict_old)
        labels_resp2, labels_stim2, rules_epochs2 = generate_response_stimulus(task_params2, test_trials2, hyp_dict)

    labels = labels_stim if color_by == "stim" else labels_resp
    labels2 = labels_stim2 if color_by == "stim" else labels_resp2
        
    test_input, test_output, _ = test_data
    test_input2, test_output2, _ = test_data2

    permutation = np.random.permutation(test_input.shape[0])
    permutation2 = np.random.permutation(test_input2.shape[0])

    test_input, test_output, labels = test_input[permutation], test_output[permutation], labels[permutation]
    test_input2, test_output2, label2 = test_input2[permutation2], test_output2[permutation2], labels2[permutation2]

    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()
    test_input2_np = test_input2.detach().cpu().numpy()
    test_output2_np = test_output2.detach().cpu().numpy()
    labels_np = labels
    labels2_np = labels2

    del test_output, test_output2

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
    test_task2 = find_task(task_params2, test_input2_np, shift_index)
    # adjust the task label information to shift back
    test_task2 = [i - len(task_params["rules"]) for i in test_task2]

    # In[7]:

    # actual fitting
    # we use net at different training stage on the same test_input
    print("================================= Stage 1 =================================")
    net_pretrain, _, (_, netout_stage1_lst, db_stage1_lst, _, _, _, _, marker_stage1_lst, _, _), pretrain_stop = net_helpers.train_network(params, device=device,
                                                                                                                                        verbose=verbose, 
                                                                                                                                        train=train,
                                                                                                                                        hyp_dict=hyp_dict_old, 
                                                                                                                                        netFunction=netFunction, 
                                                                                                                                        test_input=[test_input],
                                                                                                                                        pretraining_shift_pre=1, 
                                                                                                                                        print_frequency=100)

    # overwrite the early stopping in the post-training
    params2[1]["valid_check"] = None
    net_stage1 = copy.deepcopy(net_pretrain)

    # compare the input layer after pretraining and after posttraining
    if hyp_dict_old["chosen_network"] == "dmpn":
        input_orig  = net_pretrain.W_initial_linear.weight.detach().cpu().clone()
    elif hyp_dict_old["chosen_network"] == "vanilla":
        input_orig = net_pretrain.W_input.detach().cpu().clone()

    print("================================= Stage 2 =================================")
    net, _, (counter_lst, netout_lst, db_lst, Winput_lst, Winputbias_lst,\
            Woutput_lst, Wall_lst, marker_lst, loss_lst, acc_lst), _ = net_helpers.train_network(params2, net=net_pretrain, device=device,
                                                                                            verbose=verbose, train=train, hyp_dict=hyp_dict,
                                                                                            netFunction=netFunction, test_input=[test_input2],
                                                                                            pretraining_shift=len(task_params["rules"]), print_frequency=100)

    print("================================= End  =================================")

    if hyp_dict_old["chosen_network"] == "dmpn":
        input_after  = net.W_initial_linear.weight.detach().cpu().clone()
    elif hyp_dict_old["chosen_network"] == "vanilla":
        input_after = net.W_input.detach().cpu().clone()                                                                                    

    # Sanity check that stage-2 freezing actually kept every column of the
    # input layer except the last identical to its pretraining value.
    # (The visual heatmap comparison was removed so this script only writes
    # data artifacts; diagnostic plots live in pretraining_analysis.py.)
    diff = (input_orig[:, :-1] - input_after[:, :-1]).abs()
    assert torch.all(diff < 1e-4)

    # In[ ]:


    if hyp_dict['chosen_network'] == "dmpn":
        if net_params["input_layer_add"]:
            # Keep the counter_lst transform because downstream code still
            # expects the log-shifted iterations, even though the norm plot
            # itself has been dropped.
            counter_lst = [x * epoch_multiply + 1 for x in counter_lst]


    # In[ ]:


    # sanity check, if W_freeze, then the recorded W matrix for the modulation layer should not be changed
    if net_params["ml_params"]["W_freeze"]: 
        assert np.allclose(Wall_lst[-1][0], Wall_lst[0][0])

    if net_params["input_layer_bias"]: 
        assert net_params["input_layer_add"] is True 


    # In[ ]:


    print('Done!')


    # In[ ]:


    # net_eta_lambda_analysis saves diagnostic eta/lambda figures under
    # ./results/ — suppressed here so this script writes data artifacts only.
    # The relevant eta/lambda info is accessible post-hoc from the saved
    # state_dict if needed.

    # In[ ]:

    use_finalstage = False
    if use_finalstage:
        # plotting output in the validation set
        net_out_final, db = net.iterate_sequence_batch(test_input, run_mode='track_states')
        
        W_output = net.W_output.detach().cpu().numpy()

        W_all_ = []
        for i in range(len(net.mp_layers)):
            W_all_.append(net.mp_layers[i].W.detach().cpu().numpy())
        W_ = W_all_[0]
        
    else:
        ind = len(marker_lst)-1 
        ind_stage1 = len(marker_stage1_lst)-1
        network_at_percent = (marker_lst[ind]+1)/train_params2['n_datasets']*100
        print(f"Using network at {network_at_percent}%")
        # by default using the first test_input 
        net_out_final = netout_lst[0][ind]
        net_out_stage1_final = netout_stage1_lst[0][ind_stage1]
        
        db = db_lst[0][ind]
        W_output = Woutput_lst[ind]
        # W_ = Wall_lst[ind][0]


    # In[ ]:


    # Input/output qualitative visualization (`lowD_*_stage{1,2}.png`)
    # was removed — see pretraining_analysis.py
    # (`run_final_net_sanity_check`) for the post-hoc equivalent, which
    # runs the reloaded network on both stages' inputs directly.

    # save to the output
    pathname_stage1output = f"./pretraining/output_{hyp_dict_old['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}_stage1.npz"
    np.savez_compressed(pathname_stage1output, \
                        test_input_np=test_input_np, 
                        net_out_stage1_final=net_out_stage1_final,
                        test_output_np=test_output_np, 
                        rules_epochs=rules_epochs,
                        task_params=task_params, 
                        test_task=test_task
    )

    print(f"test_input_np: {test_input_np.shape}")
    print(f"net_out_stage1_final: {net_out_stage1_final.shape}")
    print(f"test_output_np: {test_output_np.shape}")


    pathname_stage2output = f"./pretraining/output_{hyp_dict_old['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}_stage2.npz"
    np.savez_compressed(pathname_stage2output, \
                        test_input_np=test_input2_np, 
                        net_out_final=net_out_final,
                        test_output_np=test_output2_np, 
                        rules_epochs2=rules_epochs2,
                        task_params=task_params2, 
                        test_task=test_task2
    )

    # In[ ]:


    # here db is selected based on learning stage selection 

    layer_index = 0 # 1 layer MPN 
    if net_params["input_layer_add"]:
        layer_index += 1 

    max_seq_len1 = test_input.shape[1]
    max_seq_len2 = test_input2.shape[1]
        
    def modulation_extraction(db_, max_seq_len_, layer_index, half=False, nettype="dmpn"):
        """
        """
        print(db_.keys())
        devider = 1 if not half else 2
        n_batch_all_ = test_input.shape[0]
        
        if nettype == "dmpn": 
            Ms = np.concatenate((
                db_[f'M{layer_index}'].reshape(int(n_batch_all_ / devider), max_seq_len_, -1),
            ), axis=-1)
        
            Ms_orig = np.concatenate((
                db_[f'M{layer_index}'],
            ), axis=-1)
        
            bs = np.concatenate((
                db_[f'b{layer_index}'],
            ), axis=-1) 
        
            hs = np.concatenate((
                db_[f'hidden{layer_index}'].reshape(int(n_batch_all_ / devider), max_seq_len_, -1),
            ), axis=-1)
        
            xs = np.concatenate((
                db_[f'input{layer_index}'].reshape(int(n_batch_all_ / devider), max_seq_len_, -1),
            ), axis=-1)
        
            return Ms, Ms_orig, hs, bs, xs
        
        elif nettype == "vanilla":
            hs = np.concatenate((
                db_[f'hidden'].reshape(int(n_batch_all_ / devider), max_seq_len_, -1),
            ), axis=-1)

            return None, None, hs, None, None
            

    print(f"rules_epochs: {rules_epochs}")
    print(f"rules_epochs2: {rules_epochs2}")

    all_rules = np.array(task_params["rules"])
    print(f"all_rules: {all_rules}")
    test_task = np.array(test_task)
    print(f"test_task: {test_task}")

    Ms_stage1, Ms_orig_stage1, hs_stage1, bs_stage1, xs_stage1 = modulation_extraction(db_stage1_lst[0][-1], max_seq_len1, layer_index, nettype=hyp_dict["chosen_network"])
    # since we only have half of the batches (one-task in post-training vs. two-task in pre-training)
    # so in reshape, we need to adjust the desired 
    Ms_stage2, Ms_orig_stage2, hs_stage2, bs_stage2, xs_stage2 = modulation_extraction(db_lst[0][-1], max_seq_len2, layer_index, half=True, nettype=hyp_dict["chosen_network"])

    # print(f"Ms_stage1.shape:{Ms_stage1.shape}")
    # print(f"Ms_orig_stage1.shape:{Ms_orig_stage1.shape}")
    print(f"hs_stage1.shape:{hs_stage1.shape}")
    # print(f"bs_stage1.shape:{bs_stage1.shape}")
    # print(f"xs_stage1.shape:{xs_stage1.shape}")

    # print(f"Ms_stage2.shape:{Ms_stage2.shape}")
    # print(f"Ms_orig_stage2.shape:{Ms_orig_stage2.shape}")
    print(f"hs_stage2.shape:{hs_stage2.shape}")
    # print(f"bs_stage2.shape:{bs_stage2.shape}")
    # print(f"xs_stage2.shape:{xs_stage2.shape}")

    assert hs_stage1.shape[-1] == hs_stage2.shape[-1]

    # save
    pathname = f"./pretraining/param_{hyp_dict_old['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}_result.npz"
    np.savez_compressed(pathname, \
                        rules_epochs=rules_epochs, 
                        rules_epochs2=rules_epochs2, 
                        hyp_dict_old=hyp_dict_old,
                        hyp_dict=hyp_dict, \
                        all_rules=all_rules, \

                        Ms_orig_stage1=Ms_orig_stage1, \
                        hs_stage1=hs_stage1, \
                        bs_stage1=bs_stage1, \
                        xs_stage1=xs_stage1, \
                        
                        Ms_orig_stage2=Ms_orig_stage2, \
                        hs_stage2=hs_stage2, \
                        bs_stage2=bs_stage2, \
                        xs_stage2=xs_stage2, \

                        pretrain_stop=pretrain_stop, \
                        valid_acc_iter=net.hist['iters_monitor'][1:], \
                        valid_acc=net.hist['valid_acc'][1:]
    )

    # 2025-10-20: save the network 
    netpathname = f"./pretraining/savednet_{hyp_dict_old['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}.pt"
    save_dict = {
        "state_dict": net.state_dict(), # trained result
        "net_params": net_params # network parameter
    }
    torch.save(save_dict, netpathname)
    print("Network parameter saving is done")

    # Save training history for both stages
    def _hist_to_serializable(hist):
        """Convert hist dict to numpy-friendly format for pickling."""
        out = {}
        for k, v in hist.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                out[k] = [t.detach().cpu().numpy() for t in v]
            else:
                out[k] = v
        return out

    hist_pathname = f"./pretraining/hist_{hyp_dict_old['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}.pkl"
    with open(hist_pathname, "wb") as f:
        pickle.dump({
            "stage1": _hist_to_serializable(net_stage1.hist),
            "stage2": _hist_to_serializable(net.hist),
            "pretrain_stop": pretrain_stop,
        }, f)
    print(f"Training history saved: {hist_pathname}")

    # # try to reload
    # checkpoint = torch.load(netpathname, map_location="cpu", weights_only=True)
    # net_params_loaded = checkpoint["net_params"]

    # net = mpn.DeepMultiPlasticNet(net_params_loaded)
    # net.load_state_dict(checkpoint["state_dict"])
    # net.eval()   
    # print("Reload Check is done")

