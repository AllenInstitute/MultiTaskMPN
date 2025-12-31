# %%
# Basic Libraries
import sys
import time
import gc
import random
import copy 
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PyTorch Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Data Handling and Image Processing
from torchvision import datasets, transforms

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim

# Style for Matplotlib
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

# Custom Modules and Extensions
sys.path.append("../netrep/")
sys.path.append("../svcca/")

import networks as nets  # Contains RNNs
import net_helpers
import mpn_tasks
import helper
import mpn

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

# Memory Optimization
gc.collect()
torch.cuda.empty_cache()

# %%
# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',] * 10
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',] * 10
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',] * 10 
l_vals = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
markers_vals = ['o', 'v', '*', '+', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
linestyles = ["-", "--", "-."]

# %%
# Reload modules if changes have been made to them
from importlib import reload

reload(nets)
reload(net_helpers)

def run():
    fixseed = False # randomize setting the seed may lead to not perfectly solved results
    seed = random.randint(1,1000) if not fixseed else 8 # random set the seed to test robustness by default
    print(f"Set seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    hyp_dict = {}

    hyp_dict['task_type'] = 'multitask' # int, NeuroGym, multitask
    hyp_dict['mode_for_all'] = "random_batch"
    hyp_dict['ruleset'] = 'delaygofamily' # low_dim, all, test

    accept_rules = ('fdgo', 'fdanti', 'delaygo', 'delayanti', 'reactgo', 'reactanti', 
                    'delaydm1', 'delaydm2', 'dmsgo', 'dmcgo', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm')


    rules_dict = \
        {'all' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
        'low_dim' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                    'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                    'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
        'gofamily': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti'],
        'delaygo': ['delaygo'],
        'delaygofamily': ['delaygo', 'delayanti'],
        'fdgo': ['fdgo'],
        'fdfamily': ['fdgo', 'fdanti'],
        'reactgo': ['reactgo'],
        'reactfamily': ['reactgo', 'reactanti'],
        'delaydm1': ['delaydm1'],
        'delaydmfamily': ['delaydm1', 'delaydm2'],
        'dmsgofamily': ['dmsgo', 'dmsnogo'],
        'dmsgo': ['dmsgo'],
        'dmcgo': ['dmcgo'],
        'contextdelayfamily': ['contextdelaydm1', 'contextdelaydm2'],
        }
        

    # This can either be used to set parameters OR set parameters and train
    train = True # whether or not to train the network
    verbose = True
    hyp_dict['run_mode'] = 'minimal' # minimal, debug
    hyp_dict['chosen_network'] = "dmpn"

    # suffix for saving images
    # inputadd, Wfix, WL2, hL2
    # inputrandom, Wtrain
    # noise001
    # largeregularization
    # trainetalambda

    mpn_depth = 1
    n_hidden = 200

    hyp_dict['addon_name'] = "inputrandom+Wtrain+WL2+hL2"
    hyp_dict['addon_name'] += f"+hidden{n_hidden}"

    # for coding 
    if hyp_dict['chosen_network'] in ("gru", "vanilla"):
        mpn_depth = 1

    def current_basic_params():
        task_params = {
            'task_type': hyp_dict['task_type'],
            'rules': rules_dict[hyp_dict['ruleset']],
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
            'modality_diff': False,
            'label_strength': False, 
            'long_delay': 'normal',
            'long_response': 'normal',
            'adjust_task_prop': True,
            'adjust_task_decay': 0.9, 
        }

        print(f"Fixation_off: {task_params['fixate_off']}; Task_info: {task_params['task_info']}")

        train_params = {
            'lr': 1e-3,
            'n_batches': 128,
            'batch_size': 128,
            'gradient_clip': 10,
            'valid_n_batch': 100,
            'n_datasets': 100, # Number of distinct batches
            'valid_check': None, 
            'n_epochs_per_set': 1, # longer/shorter training
            'weight_reg': 'L2',
            'activity_reg': 'L2', 
            'reg_lambda': 1e-4,
        }

        if not train: # some 
            assert train_params['n_epochs_per_set'] == 0

        net_params = {
            'net_type': hyp_dict['chosen_network'], # mpn1, dmpn, vanilla
            'n_neurons': [1] + [n_hidden] * mpn_depth + [1],
            'output_bias': False, # Turn off biases for easier interpretation
            'loss_type': 'MSE', # XE, MSE
            'activation': 'tanh', # linear, ReLU, sigmoid, tanh, tanh_re, tukey, heaviside
            'cuda': True,
            'monitor_freq': train_params["n_epochs_per_set"],
            'monitor_valid_out': True, # Whether or not to save validation output throughout training
            'output_matrix': '',# "" (default); "untrained", or "orthogonal"
            'input_layer_add': True, 
            'input_layer_add_trainable': False, # revise this is effectively to [randomize_inputs], tune this
            'input_layer_bias': False, 
            'input_layer': "trainable", # for RNN only
            'acc_measure': 'stimulus', 
            
            # for one-layer MPN, GRU or Vanilla
            'ml_params': {
                'bias': True, # Bias of layer
                'mp_type': 'mult',
                'm_update_type': 'hebb_assoc', # hebb_assoc, hebb_pre
                'eta_type': 'scalar', # scalar, pre_vector, post_vector, matrix
                'eta_train': False,
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

    task_params, train_params, net_params = current_basic_params()

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

    # %%
    hyp_dict["mess_with_training"] = False

    if hyp_dict['mess_with_training']:
        hyp_dict['addon_name'] += "messwithtraining"

    params = task_params, train_params, net_params

    if net_params['net_type'] == 'mpn1':
        netFunction = mpn.MultiPlasticNet
    elif net_params['net_type'] == 'dmpn':
        netFunction = mpn.DeepMultiPlasticNet
    elif net_params['net_type'] == 'vanilla':
        netFunction = nets.VanillaRNN
    elif net_params['net_type'] == 'gru':
        netFunction = nets.GRU

    # %%
    test_n_batch = train_params["valid_n_batch"]
    color_by = "stim" # or "resp" 

    task_random_fix = True
    if task_random_fix:
        print(f"Align {task_params['rules']} With Same Time")

    if task_params['task_type'] in ('multitask',): # Test batch consists of all the rules
        task_params['hp']['batch_size_train'] = test_n_batch
        # using homogeneous cutting off
        test_mode_for_all = "random"
        # ZIHAN
        # generate test data using "random"
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(task_params, test_n_batch, \
                    rules=task_params['rules'], mode_input=test_mode_for_all, fix=task_random_fix
        )
        _, test_trials, test_rule_idxs = test_trials_extra

        task_params_attractor = copy.deepcopy(task_params)
        task_params_attractor["long_delay"] = "long"
        test_data_attractor, test_trials_extra_attractor = mpn_tasks.generate_trials_wrap(task_params_attractor, test_n_batch, \
                                                                                        rules=task_params_attractor['rules'], \
                                                                                        mode_input=test_mode_for_all, fix=task_random_fix)
        
        _, test_trials_attractor, test_rule_idxs_attractor = test_trials_extra_attractor

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
            for rule_idx, rule in enumerate(task_params['rules']):
                print(rule)
                if rule in accept_rules:
                    if hyp_dict['ruleset'] in ('dmsgo', 'dmcgo'):
                        labels.append(test_trials[rule_idx].meta['matches'])
                    else:
                        labels_resp.append(test_trials[rule_idx].meta['resp1'])
                        labels_stim.append(test_trials[rule_idx].meta['stim1']) 
        
                else:
                    raise NotImplementedError()
                    
            labels_resp = np.concatenate(labels_resp, axis=0).reshape(-1,1)
            labels_stim = np.concatenate(labels_stim, axis=0).reshape(-1,1)

            return labels_resp, labels_stim

        labels_resp, labels_stim = generate_response_stimulus(task_params, test_trials)

    labels = labels_stim if color_by == "stim" else labels_resp
        
    test_input, test_output, test_mask = test_data
    test_input_attractor, test_output_attractor, test_mask_attractor = test_data_attractor
    print(test_input_attractor.shape)
    print(test_output_attractor.shape)

    permutation = np.random.permutation(test_input.shape[0])
    test_input = test_input[permutation]
    test_output = test_output[permutation]
    test_mask = test_mask[permutation]
    labels = labels[permutation]

    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()

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
            # task_label_index = np.where(task_label == 1)[0][0]
            
            # tol = 1e-3      
            # mask = np.isclose(task_label, 1, atol=tol)
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
    test_task_attractor = find_task(task_params_attractor, test_input_attractor.detach().cpu().numpy(), shift_index)

    # %%
    # we use net at different training stage on the same test_input
    start_time = time.time()
    net, _, (counter_lst, netout_lst, db_lst, Winput_lst, Winputbias_lst,\
            Woutput_lst, Wall_lst, marker_lst, loss_lst, acc_lst), _ = net_helpers.train_network(params, device=device, verbose=verbose,
                                                                                                train=train, hyp_dict=hyp_dict,\
                                                                                                netFunction=netFunction,\
                                                                                                test_input=[test_input, test_input_attractor],
                                                                                                print_frequency=1)

    end_time = time.time()
    print(f"Running Time: {end_time - start_time}")
    counter_lst = [x * epoch_multiply + 1 for x in counter_lst] # avoid log plot issue  
    
    test_bundle = {
        "test_input": test_input,
        "test_output": test_output,
        "test_mask": test_mask,

        "test_input_attractor": test_input_attractor,
        "test_output_attractor": test_output_attractor,
        "test_mask_attractor": test_mask_attractor,

        "labels_resp": labels_resp,
        "labels_stim": labels_stim,
        "labels": labels,  # if you want exactly what you used after permutation

        "test_task": np.array(test_task),
        "test_task_attractor": np.array(test_task_attractor),

    }

    training_bundle_heavy = {
        "counter_lst": counter_lst,
        "marker_lst": marker_lst,
        "loss_lst": loss_lst,
        "acc_lst": acc_lst,
        "db_lst": db_lst,
        "netout_lst": netout_lst,
        "Wall_lst": Wall_lst,
        "Woutput_lst": Woutput_lst,
        "Winput_lst": Winput_lst,
        "Winputbias_lst": Winputbias_lst,
    }

    ckpt_path = f"./twotasks_training/{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.pt"
    net_helpers.save_checkpoint(
        ckpt_path,
        net=net,
        params=params,
        hyp_dict=hyp_dict,
        seed=seed,
        test_bundle=test_bundle,
        training_bundle=training_bundle_heavy,   # or None if you want even smaller
    )


if __name__ == "__main__":
    run()