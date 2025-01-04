import torch
import gc
gc.collect()
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles

from scipy.spatial.distance import cosine

import sys
import time
sys.path.append('/content/drive/MyDrive/neuro_research/mpns_more/')
sys.path.append("../netrep/")
sys.path.append("../svcca/")

import cca_core
from netrep.metrics import LinearMetric

import networks as nets # Contains RNNs
import net_helpers as net_helpers # A bunch helpers for both MPNs/RNNs
import tasks
import helper
import mpn

# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',]
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',]
l_vals = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
markers_vals = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']


# Reload modules if changes have been made to them
from importlib import reload

reload(nets)
reload(net_helpers)
# reload(analysis)

seed = 6

np.random.seed(seed)
torch.manual_seed(seed)

task_type = 'multitask' # int, NeuroGym, multitask
mode_for_all = "random_batch"
ruleset = 'delaygo' # low_dim, all, test

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
train = True
verbose = True
run_mode = 'minimal' # minimal, debug
chosen_network = "dmpn"

# suffix for saving images
addon_name = ""

mpn_depth = 1

# for coding 
if chosen_network in ("gru", "vanilla"):
    mpn_depth = 1

def current_basic_params():
    task_params = {
        'task_type': task_type,
        'rules': rules_dict[ruleset],
        'dt': 40, # ms, directly influence sequence lengths,
        'ruleset': ruleset,
        'n_eachring': 8, # Number of distinct possible inputs on each ring
        'in_out_mode': 'low_dim',  # high_dim or low_dim or low_dim_pos (Robert vs. Laura's paper, resp)
        'sigma_x': 0.00, # Laura raised to 0.1 to prevent overfitting (Robert uses 0.01)
        'mask_type': 'cost', # 'cost', None
        'fixate_off': True, # Second fixation signal goes on when first is off
        'randomize_inputs': False,
        'n_input': 20, # Only used if inputs are randomized
    }

    train_params = {
        'lr': 1e-3,
        'n_batches': 640,
        'batch_size': 640,
        'gradient_clip': 10,
        'valid_n_batch': 200,
        'n_datasets': 100, # Number of distinct batches
        'n_epochs_per_set': 130, # longer/shorter training
        'task_mask': None, # None, task
        # 'weight_reg': 'L2',
        # 'reg_lambda': 1e-4,
    }

    n_hidden = 100

    net_params = {
        'net_type': chosen_network, # mpn1, dmpn, vanilla
        'n_neurons': [1] + [n_hidden] * mpn_depth + [1],
        'output_bias': False, # Turn off biases for easier interpretation
        'loss_type': 'MSE', # XE, MSE
        'activation': 'tanh', # linear, ReLU, sigmoid, tanh, tanh_re, tukey, heaviside
        'cuda': True,
        'monitor_freq': 100,
        'monitor_valid_out': True, # Whether or not to save validation output throughout training
        
        # for one-layer MPN, GRU or Vanilla

        'ml_params': {
            'bias': True, # Bias of layer
            'mp_type': 'mult',
            'm_update_type': 'hebb_assoc', # hebb_assoc, hebb_pre
            'eta_type': 'scalar', # scalar, pre_vector, post_vector, matrix
            'eta_train': False,
            # 'eta_init': 'mirror_gaussian', #0.0,
            'lam_type': 'scalar', # scalar, pre_vector, post_vector, matrix
            'm_time_scale': 400, # ms, sets lambda
            'lam_train': False,
        },

        # for multiple layer MPN

        # 'ml_params0': {
        #     'bias': False,
        #     'mp_type': 'mult',
        #     'm_update_type': 'hebb_assoc', # hebb_assoc, hebb_pre

        #     'eta_type': 'matrix', # scalar, pre_vector, post_vector, matrix
        #     'eta_train': True,
        #     'eta_init': 'xavier',

        #     'lam_type': 'matrix', # scalar, pre_vector, post_vector, matrix
        #     'lam_clamp': 0.99,
        #     'lam_train': True,
        # },

        # 'ml_params1': {
        #     'bias': False,
        #     'mp_type': 'mult',
        #     'm_update_type': 'hebb_assoc', # hebb_assoc, hebb_pre

        #     'eta_type': 'matrix', # scalar, pre_vector, post_vector, matrix
        #     'eta_train': True,
        #     'eta_init': 'xavier',

        #     'lam_type': 'matrix', # scalar, pre_vector, post_vector, matrix
        #     'lam_clamp': 0.99,
        #     'lam_train': True,
        # },

        # 'ml_params2': {
        #     'bias': False,
        #     'mp_type': 'mult',
        #     'm_update_type': 'hebb_assoc', # hebb_assoc, hebb_pre

        #     'eta_type': 'matrix', # scalar, pre_vector, post_vector, matrix
        #     'eta_train': True,
        #     'eta_init': 'xavier',

        #     'lam_type': 'matrix', # scalar, pre_vector, post_vector, matrix
        #     'lam_clamp': 0.99,
        #     'lam_train': True,
        # },

        # Vanilla RNN params
        'leaky': True,
        'alpha': 0.2,
    }

    # for multiple MPN layers, assert 
    if mpn_depth > 1:
        for mpl_idx in range(mpn_depth - 1):
            assert f'ml_params{mpl_idx}' in net_params.keys()

    # actually I don't think it is needed
    # putting here to warn the parameter checking every time 
    # when switching network
    if chosen_network in ("gru", "vanilla"):
        assert f'ml_params' in net_params.keys()

    return task_params, train_params, net_params

task_params, train_params, net_params = current_basic_params()

lag = "trained" if train_params["n_epochs_per_set"] > 10 else "untrained"

if task_type in ('multitask',):
    task_params, train_params, net_params = tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )

    net_params['prefs'] = tasks.get_prefs(task_params['hp'])

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

######
mess_with_training = False
if mess_with_training:
    addon_name += "messwithtraining"

def train_network(params, net=None, device=torch.device('cuda'),
                  verbose=False,):

    task_params, train_params, net_params = params

    if task_params['task_type'] in ('multitask',):
        def generate_train_data(device='cuda'):
            # ZIHAN
            # correct thing to do
            # "training batches should only be over one rule"
            # "but validation should mix them"
            if not mess_with_training:
                train_data, _ = tasks.generate_trials_wrap(
                    task_params, train_params['n_batches'], device=device, verbose=False, mode_input=mode_for_all
                )
            else:
                print("=== Mess with generating training data ===")
                train_data, _ = tasks.generate_trials_wrap(
                    task_params, train_params['n_batches'], rules=task_params['rules'], device=device, verbose=False, mode_input=mode_for_all, mess_with_training=True
                )

            return train_data
        def generate_valid_data(device='cuda'):
            valid_data, _ = tasks.generate_trials_wrap(
                task_params, train_params['valid_n_batch'], rules=task_params['rules'], device=device, mode_input=mode_for_all
            )
            return valid_data
    else:
        raise ValueError('Task type not recognized.')

    if net is None: # Create a new network
        if net_params['net_type'] == 'mpn1':
            netFunction = mpn.MultiPlasticNet
        elif net_params['net_type'] == 'dmpn':
            netFunction = mpn.DeepMultiPlasticNet
        elif net_params['net_type'] == 'vanilla':
            netFunction = nets.VanillaRNN
        elif net_params['net_type'] == 'gru':
            netFunction = nets.GRU

        net = netFunction(net_params, verbose=verbose)

    # Puts network on device
    net.to(device)

    # check_net_cell_types(net) # Pre-train cell types check

    valid_data = generate_valid_data(device=device)

    for dataset_idx in range(train_params['n_datasets']):
        # Regenerate new data
        train_data = generate_train_data(device=device)
        new_thresh = True if dataset_idx == 0 else False
        _ = net.fit(train_params, train_data, valid_batch=valid_data, new_thresh=new_thresh, run_mode=run_mode)

    return net, (train_data, valid_data)

if train:

    params = task_params, train_params, net_params

    net, _ = train_network(params, device=device, verbose=verbose)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(net.hist['iters_monitor'], net.hist['train_loss'], color=c_vals[0], label='Full train loss')
    ax.plot(net.hist['iters_monitor'], net.hist['valid_loss'], color=c_vals[1], label='Full valid loss')
    if net.weight_reg is not None:
        ax.plot(net.hist['iters_monitor'], net.hist['train_loss_output_label'], color=c_vals_l[0], zorder=-1, label='Output label')
        ax.plot(net.hist['iters_monitor'], net.hist['train_loss_reg_term'], color=c_vals_l[0], zorder=-1, label='Reg term', linestyle='dashed')
        ax.plot(net.hist['iters_monitor'], net.hist['valid_loss_output_label'], color=c_vals_l[1], zorder=-1, label='Output valid label')
        ax.plot(net.hist['iters_monitor'], net.hist['valid_loss_reg_term'], color=c_vals_l[1], zorder=-1, label='Reg valid term', linestyle='dashed')

    ax.set_yscale('log')
    ax.legend()
    ax.set_ylabel('Loss ({})'.format(net.loss_type))
    ax.set_xlabel('# Batches')
    plt.savefig(f"./results/loss_{ruleset}_{chosen_network}_{addon_name}.png")

    if net_params['loss_type'] in ('MSE',):
        helper.plot_some_ouputs(params, net, mode_for_all, nameadd=f"{ruleset}_{chosen_network}_{addon_name}")

else:
    print('Not training, set train=True to train.')

print('Done!')

#######
#######
#######
#######

# only make sense for dmpn for eta and lambda information extraction
if net_params['net_type'] in ("dmpn",):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for mpl_idx, mp_layer in enumerate(net.mp_layers):
        if net.mp_layers[mpl_idx].eta_type in ('pre_vector', 'post_vector', 'matrix',):
            full_eta = np.concatenate([
                eta.flatten()[np.newaxis, :] for eta in net.hist['eta{}'.format(mpl_idx)]
            ], axis=0)
        else:
            full_eta = net.hist['eta{}'.format(mpl_idx)]

        if net.mp_layers[mpl_idx].lam_type in ('pre_vector', 'post_vector', 'matrix',):
            full_lam = np.concatenate([
                lam.flatten()[np.newaxis, :] for lam in net.hist['lam{}'.format(mpl_idx)]
            ], axis=0)
        else:
            full_lam = net.hist['lam{}'.format(mpl_idx)]
        ax1.plot(net.hist['iters_monitor'], full_eta, color=c_vals[mpl_idx], label='MPL{}'.format(mpl_idx))
        ax2.plot(net.hist['iters_monitor'], full_lam, color=c_vals[mpl_idx], label='MPL{}'.format(mpl_idx))

    ax1.axhline(0.0, color='k', linestyle='dashed')

    ax1.set_ylabel('Eta')
    ax2.set_ylabel('Lambda')

    if net.mp_layers[mpl_idx].eta_type not in ('pre_vector', 'post_vector', 'matrix',):
        ax1.legend()
        ax2.legend()

    fig.savefig(f"./results/eta_lambda_{ruleset}_{chosen_network}_{addon_name}.png")

    # only for deep mpn with multiple layers
    n_mplayers = len(net.mp_layers)
    if n_mplayers > 1:
        fig, axs = plt.subplots(1, n_mplayers, figsize=(4+4*n_mplayers, 4))
        n_bins = 50

        for mpl_idx, (mp_layer, ax) in enumerate(zip(net.mp_layers, axs)):

            init_eta = net.hist['eta{}'.format(mpl_idx)][0].flatten()
            final_eta = net.hist['eta{}'.format(mpl_idx)][-1].flatten()

            max_eta = np.max((np.max(np.abs(init_eta)), np.max(np.abs(final_eta))))

            bins = np.linspace(-max_eta, max_eta, n_bins+1)[1:]

            ax.hist(init_eta, bins=bins, color=c_vals_l[mpl_idx], alpha=0.3, label='init')
            ax.hist(final_eta, bins=bins, color=c_vals[mpl_idx], alpha=0.3, label='final')

            ax.set_ylabel('Count')
            ax.set_xlabel('Eta value')
            ax.legend()

        fig.savefig(f"./results/eta_distribution_{ruleset}_{chosen_network}_{addon_name}.png")

#######
#######
#######
#######

# test_n_batch = 100 # Per rule evaluation
test_n_batch = train_params["valid_n_batch"]

if task_params['task_type'] in ('multitask',): # Test batch consists of all the rules
    task_params['hp']['batch_size_train'] = test_n_batch
    # using homogeneous cutting off
    test_mode_for_all = "random"
    # ZIHAN
    # generate test data using "random"
    test_data, test_trials_extra = tasks.generate_trials_wrap(task_params, test_n_batch, \
                rules=task_params['rules'], mode_input=test_mode_for_all)
    _, test_trials, test_rule_idxs = test_trials_extra

    task_params['dataset_name'] = 'multitask'

    if task_params['in_out_mode'] in ('low_dim_pos',):
        output_dim_labels = ('Fixate', 'Cos', '-Cos', 'Sin', '-Sin')
    elif task_params['in_out_mode'] in ('low_dim',):
        output_dim_labels = ('Fixate', 'Cos', 'Sin')
    else:
        raise NotImplementedError()

    labels = []
    for rule_idx, rule in enumerate(task_params['rules']):
        print(rule)
        if rule in accept_rules:
            if ruleset in ('dmsgo', 'dmcgo'):
                labels.append(test_trials[rule_idx].meta['matches'])
            else:
                labels.append(test_trials[rule_idx].meta['resp1'])

        else:
            raise NotImplementedError()
    labels = np.concatenate(labels, axis=0)

test_input, test_output, test_mask = test_data
print(f"test_input: {test_input.shape}")

test_input_np = test_input.detach().cpu().numpy()
test_output_np = test_output.detach().cpu().numpy()

n_batch_all = test_input_np.shape[0] # Total number of batches, might be different than test_n_batch
# print(n_batch_all)
max_seq_len = test_input_np.shape[1]

# plotting output in the validation set
net_out, db = net.iterate_sequence_batch(test_input, run_mode='track_states')

W_output = net.W_output.detach().cpu().numpy()
W_all_ = []
for i in range(len(net.mp_layers)):
    W_ = net.mp_layers[i].W.detach().cpu().numpy()
    W_all_.append(W_)


net_out = net_out.detach().cpu().numpy()

if net_params['loss_type'] in ('MSE',):
    test_out_np = test_output.cpu().numpy()

    fig, axs = plt.subplots(4, 1, figsize=(8, 8))

    if test_out_np.shape[-1] == 1:
        for batch_idx, ax in enumerate(axs):
            ax.plot(net_out[batch_idx, :, 0], color=c_vals[batch_idx])
            ax.plot(test_out_np[batch_idx, :, 0], color=c_vals_l[batch_idx], zorder=-1)

            # ax.set_ylim((0, 2.25))
    else:
        for batch_idx, ax in enumerate(axs):
            for out_idx in range(test_out_np.shape[-1]):
                ax.plot(net_out[batch_idx, :, out_idx], color=c_vals[out_idx])
                ax.plot(test_out_np[batch_idx, :, out_idx], color=c_vals_l[out_idx], zorder=-1)

            # ax.set_ylim((-3, 2.25))

    fig.suptitle("Validation Set Output Comparison")
    fig.tight_layout()
    fig.savefig(f"./results/lowD_{ruleset}_{chosen_network}_{addon_name}.png")


#######
#######
#######
#######


if net_params["net_type"] in ("dmpn", ):
    if mpn_depth == 1:
        Ms = np.concatenate((
            db['M0'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1),
            # db['M1'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1)
        ), axis=-1)

        Ms_orig = np.concatenate((
            db['M0'].detach().cpu().numpy(),
            # db['M1'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1)
        ), axis=-1)

        hs = np.concatenate((
            db['hidden0'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1),
        ), axis=-1)

    else:
        # raise NotImplementedError() # Let's focusing on one layer first
        modulations, hiddens = [], []
        for i in range(mpn_depth):
            modulations.append(db[f'M{i}'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1))
            hiddens.append(db[f'hidden{i}'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1),)

        # Ms = np.concatenate(modulations, axis=-1)
        Ms = modulations[0]
        hs = hiddens[0]
        
elif net_params["net_type"] in ("vanilla", "gru"):
    hs = db['hidden'].detach().cpu().numpy()

# print(f"Ms.shape: {Ms.shape}")
# print(f"hs.shape: {hs.shape}")

pca_type = 'full' # full, cell_types
pca_target_lst = ['hs', 'Ms'] # hs, 'Ms' 
if net_params["net_type"] in ("vanilla", "gru"):
    pca_target_lst = ['hs'] # if not dmpn, no M information effectively


# using recorded information
recordkyle_all, recordkyle_nameall = [], []
for test_subtrial in test_trials:
    metaepoch = test_subtrial.epochs
    periodname = list(metaepoch.keys())
    recordkyle, recordkyle_name = [], []
    # print(metaepoch)
    for keyiter in range(len(periodname)):
        try:
            recordkyle_name.append(periodname[keyiter])
            if test_mode_for_all == "random":
                recordkyle.append(metaepoch[periodname[keyiter]][1])
            elif test_mode_for_all == "random_batch":
                recordkyle.append(list(metaepoch[periodname[keyiter]][1]))
        except Exception as e:
            print(e)
    
    if test_mode_for_all in ("random",):
        fillrecordkyle = []
        for timestamp in recordkyle:
            fillrecordkyle.append([timestamp for _ in range(hs.shape[0])])
        recordkyle = fillrecordkyle

    recordkyle.insert(0, [0 for _ in range(len(recordkyle[1]))])
    recordkyle = np.array(recordkyle).T.tolist()
    recordkyle_all.extend(recordkyle)
    recordkyle_nameall.append(recordkyle_name)

session_breakdown = []
for sindex in range(0,len(recordkyle[0])-1):
    session_breakdown.append([recordkyle[0][sindex], recordkyle[0][sindex+1]]) # all sessions should be the same
session_breakdown.append([recordkyle[0][0], recordkyle[0][-1]])

def to_unit_vector(arr):
    norm = np.linalg.norm(arr)
    
    if norm == 0:
        return arr
    
    unit_vector = arr / norm
    return unit_vector

# break down time
breaks = [cut[1] for cut in session_breakdown[:-1]]
print(f"breaks: {breaks}")

# Sanity check from Equation 2-7
figexh1, axsexh1 = plt.subplots(3,3,figsize=(4*3,4*3))  
figexh2, axsexh2 = plt.subplots(3,3,figsize=(4*3,4*3))  


for batch_iter in range(test_input.shape[0]):
    res_eq26, res_eq8, res_eq11 = [], [], []
    res_meta = []

    # analyze the change of M
    M_beforestim = Ms_orig[batch_iter, breaks[0], :, :]
    M_afterstim = Ms_orig[batch_iter, breaks[0]+1, :, :]
    h_s_beforesstim = hs[batch_iter, breaks[0], :].reshape(-1,1)

    saver_shape = (3,3)
    saver1 = np.empty((saver_shape[0], saver_shape[1]), dtype=object)
    saver2 = np.empty((saver_shape[0], saver_shape[1]), dtype=object)

    for i in range(saver_shape[0]):
        for j in range(saver_shape[1]):
            saver1[i, j] = np.array([])
            saver2[i, j] = np.array([])


    for time_iter in range(test_input.shape[1]):
        x = test_input[batch_iter, time_iter, :].cpu().numpy().reshape(-1,1)
        
        x_fixon = np.array([x[0,0],0,0,0,0,0,0]).reshape(-1,1) # one-hot encoded vector for fixation
        x_fixoff = np.array([0,x[1,0],0,0,0,0,0]).reshape(-1,1) # one-hot encoded vector for fixation off
        x_stimulus = np.array([0,0,x[2,0],x[3,0],x[4,0],x[5,0],0]).reshape(-1,1) # one-hot encoded vector for stimulus
        x_task = np.array([0,0,0,0,0,0,x[6,0]]).reshape(-1,1) # one-hot encoded vector for task
        
        Mt = Ms_orig[batch_iter, time_iter, :, :]
        
        middle =  W_ + W_ * Mt
        
        y_fix = W_output[0,:].reshape(1,-1)
        Y_resp1 = W_output[1,:].reshape(1,-1)
        Y_resp2 = W_output[2,:].reshape(1,-1)

        allX1 = [x_fixon+x_task, x_fixoff+x_task, x_stimulus+x_fixon+x_task]
        allX1name = ["x_fixon+x_task", "x_fixoff+x_task", "x_stimulus+x_fixon+x_task"]
        allX2 = [x_fixon, x_fixoff, x_stimulus]
        allX2name = ["x_fixon", "x_fixoff", "x_stimulus"]
        allY = [y_fix, Y_resp1, Y_resp2]
        allYname = ["y_fix", "Y_resp1", "Y_resp2"]

        for yiter in range(len(allY)):
            for xiter in range(len(allX1)):

                res1 = to_unit_vector(allY[yiter]) @ to_unit_vector(middle @ allX1[xiter])
                saver1[xiter, yiter] = np.append(saver1[xiter, yiter], res1[0,0])

                res2 = to_unit_vector(allY[yiter]) @ to_unit_vector(middle @ allX2[xiter])
                saver2[xiter, yiter] = np.append(saver2[xiter, yiter], res2[0,0])

    # plot single batch result, colored based on the response
    for i in range(saver_shape[0]):
        for j in range(saver_shape[1]):
            axsexh1[i,j].plot(saver1[i,j], color=c_vals[labels[batch_iter]])
            axsexh2[i,j].plot(saver2[i,j], color=c_vals[labels[batch_iter]])

for i in range(saver_shape[0]):
    for j in range(saver_shape[1]):
        axsexh1[i,j].set_title(f"{allX1name[i]} & {allYname[j]}")
        axsexh2[i,j].set_title(f"{allX2name[i]} & {allYname[j]}")

for ax in np.concatenate((axsexh1.flatten(), axsexh2.flatten())):
    for bb in breaks:
        ax.axvline(bb, linestyle="--")

figexh1.savefig(f"./results/exhaustive1_{lag}_{ruleset}_{chosen_network}_{addon_name}.png")
figexh2.savefig(f"./results/exhaustive2_{lag}_{ruleset}_{chosen_network}_{addon_name}.png")

print("done")
time.sleep(10000)


for session_iter in range(0,len(session_breakdown)):
    session_part = session_breakdown[session_iter]

    for pca_target in pca_target_lst:
        print(f"session_part: {session_part}; pca_target: {pca_target}")

        if net.hidden_cell_types is not None:
            cell_types = net.hidden_cell_types.detach().cpu().numpy()[0] # Remove batch idx
        cell_type_names = ('Inh.', 'Exc.')
        pca_sort_type = 'ratios' # How to sort cell type PCA; vars, ratios

        if pca_target in ('hs',):
            n_activity = hs.shape[-1]
            # Truncate into session specifically
            hs_cut = hs[:,session_part[0]:session_part[1],:]
            as_shape = hs_cut.shape 
            as_flat = hs_cut.reshape((-1, n_activity,))
        elif pca_target in ('Ms',):
            n_activity = Ms.shape[-1]
            # Truncate into session specifically
            Ms_cut = Ms[:,session_part[0]:session_part[1],:]
            # as_shape = (Ms.shape[0], Ms.shape[1], n_activity) 
            as_shape = Ms_cut.shape
            as_flat = Ms_cut.reshape((-1, n_activity))

        print(f"as_shape: {as_shape}")
        as_shape_saver = as_shape

        total_vars = np.var(as_flat, axis=0) # Var per activity dimension
        activity_zero = np.zeros((1, n_activity,))

        if pca_type in ('full',):
            activity_pca = PCA()

            activity_pca.fit(as_flat)
            
            # Corrects signs of components so that their mean/average is positive
            activity_pca.components_ = np.sign(np.sum(activity_pca.components_, axis=-1))[:, np.newaxis] * activity_pca.components_
            
            as_pca = activity_pca.transform(as_flat)
            # as_pca = as_pca.reshape(as_shape) # Separates back into batches and sequences
            as_pca = as_pca.reshape(as_shape_saver[0],as_shape_saver[1],-1)

            zeros_pca = activity_pca.transform(activity_zero)
            if pca_target in ('hs',): # Some transformations only make sense for certain activities
                W_output_pca = activity_pca.transform(W_output)

            # ZIHAN: only do for all period trial (last)
            #
            # Should we use PCA result or the original result?
            #
            if session_iter == len(session_breakdown)-1 and pca_target in ('hs', ):
                output_num = W_output.shape[0]
                assert output_num == 3 # low_dim
                figtest, axstest = plt.subplots(1,2,figsize=(5*2,5))
                # ? Original or PCA
                fixation_pca = W_output[0,:].reshape(1,-1)
                # fixation_pca = fixation_pca / np.linalg.norm(fixation_pca, axis=1, keepdims=True)
                stimulus_pca = W_output[1:,:]
                # stimulus_pca = stimulus_pca / np.linalg.norm(stimulus_pca, axis=1, keepdims=True)
                batch_record_fixate, batch_record_stimulus = [], []
                
                for batchiter in range(hs_cut.shape[0]):
                    # ? Original or PCA
                    batch_activity_pca = hs_cut[batchiter,:,:]
                    
                    batch_record1, batch_record2 = [], []
                    for timestamp in range(batch_activity_pca.shape[0]):
                        batch_time_activity_pca = batch_activity_pca[timestamp,:].reshape(1,-1)
                        #
                        # batch_time_activity_pca = batch_time_activity_pca / np.linalg.norm(batch_time_activity_pca, axis=1, keepdims=True)

                        projection_magnitude_fix = helper.magnitude_of_projection(batch_time_activity_pca, fixation_pca)
                        batch_record1.append(projection_magnitude_fix)
                        projection_magnitude_stimulus = helper.magnitude_of_projection(batch_time_activity_pca, stimulus_pca)
                        batch_record2.append(projection_magnitude_stimulus)
                        # print(f"projection_magnitude_fix: {projection_magnitude_fix}; projection_magnitude_stimulus: {projection_magnitude_stimulus}")
                        # time.sleep(10000)
                    batch_record_fixate.append(batch_record1)
                    batch_record_stimulus.append(batch_record2)

                batch_record_fixate, batch_record_stimulus = np.array(batch_record_fixate), np.array(batch_record_stimulus)
                mean_fix, std_fix = np.mean(batch_record_fixate, axis=0), np.std(batch_record_fixate, axis=0)
                mean_stimulus, std_stimulus = np.mean(batch_record_stimulus, axis=0), np.std(batch_record_stimulus, axis=0)
                xxx = [i for i in range(mean_fix.shape[0])]
                axstest[0].plot(xxx, mean_fix)
                axstest[0].fill_between(xxx, mean_fix-std_fix, mean_fix+std_fix, alpha=0.7, color="red")
                axstest[0].set_title("Projection Magnitude on 1D Subspace of Fixation Period")
                axstest[1].plot(xxx, mean_stimulus)
                axstest[1].fill_between(xxx, mean_stimulus-std_stimulus, mean_stimulus+std_stimulus, alpha=0.7, color="red")
                axstest[1].set_title("Projection Magnitude on 2D Subspace of Fixation Period")
                for ax in axstest:
                    for spp in session_breakdown[:-1]:
                        ax.axvline(spp[1], linestyle="--")
                figtest.tight_layout()
                # figtest.savefig(f"./results/zz_test_{pca_target}_{ruleset}_{chosen_network}_{addon_name}_test.png")
                

            print('PR: {:.2f}'.format(
                helper.participation_ratio_vector(activity_pca.explained_variance_ratio_)
            ))
            print('PCA component PRs - PC1: {:.1f}, PC2: {:.1f}, PC3: {:.1f}'.format(
                    helper.participation_ratio_vector(np.abs(activity_pca.components_[0, :])),
                    helper.participation_ratio_vector(np.abs(activity_pca.components_[1, :])),
                    helper.participation_ratio_vector(np.abs(activity_pca.components_[2, :])),
                ))

        elif pca_type in ('cell_types',):
            raise NotImplementedError('Need to correct this for Ms activity')
            cell_type_vals = np.unique(cell_types) # Gets unique cell type idxs

            n_cell_types = cell_type_vals.shape[0]
            pcas = [PCA() for _ in range(n_cell_types)]

            cell_types_pca = [] # This needs to be diferent from cell_types because may do cell types in a different order
            hs_pca = []
            explained_vars = []
            explained_var_ratios = []

            zeros_pca = []
            W_output_pca = []
            # Fit each PCA individually
            for cell_type_idx, (cell_type_val, cell_type_name) in enumerate(zip(
                cell_type_vals, cell_type_names
            )):
                print('Cell type: {}'.format(cell_type_name))
                cell_type_filter = (cell_types == cell_type_val)
                n_cells_type = np.sum(cell_type_filter.astype(np.int32))

                print(' Ratio of population: {:.2f}, variance: {:.2f}'.format(
                    n_cells_type / n_cells,
                    np.sum(total_vars[cell_type_filter]) / np.sum(total_vars)
                ))

                pcas[cell_type_idx].fit(hs_flat[:, cell_type_filter])
                # Corrects signs of components so that their mean/average is positive
                pcas[cell_type_idx].components_ = np.sign(np.sum(pcas[cell_type_idx].components_, axis=-1))[:, np.newaxis] * pcas[cell_type_idx].components_

                hs_pca_type = pcas[cell_type_idx].transform(hs_flat[:, cell_type_filter])

                print(' PR: {:.2f}'.format(
                    helper.participation_ratio_vector(pcas[cell_type_idx].explained_variance_ratio_)
                ))
                print(' PCA component PRs - PC1: {:.1f}, PC2: {:.1f}, PC3: {:.1f}'.format(
                    helper.participation_ratio_vector(np.abs(pcas[cell_type_idx].components_[0, :])),
                    helper.participation_ratio_vector(np.abs(pcas[cell_type_idx].components_[1, :])),
                    helepr.participation_ratio_vector(np.abs(pcas[cell_type_idx].components_[2, :])),
                ))

                cell_types_pca.append([cell_type_val for _ in range(n_cells_type)])
                hs_pca.append(hs_pca_type)
                explained_vars.append(pcas[cell_type_idx].explained_variance_)
                explained_var_ratios.append(pcas[cell_type_idx].explained_variance_ratio_)

                zeros_pca.append(pcas[cell_type_idx].transform(hidden_zero[:, cell_type_filter]))
                W_output_pca.append(pcas[cell_type_idx].transform(W_output[:, cell_type_filter]))

            explained_vars = np.concatenate(explained_vars, axis=-1)
            explained_var_ratios = np.concatenate(explained_var_ratios, axis=-1)
            # Now sort based on explained variances/explained variance ratio
            if pca_sort_type in ('vars',):
                pca_type_sort = np.argsort(explained_vars)[::-1] # largest to smallest
            elif pca_sort_type in ('ratios',):
                pca_type_sort = np.argsort(explained_var_ratios)[::-1] # largest to smallest

            cell_types_pca =  np.concatenate(cell_types_pca, axis=-1)[pca_type_sort]
            hs_pca = np.concatenate(hs_pca, axis=-1)[:, pca_type_sort]
            explained_vars = explained_vars[pca_type_sort]
            explained_var_ratios = explained_var_ratios[pca_type_sort]
            zeros_pca = np.concatenate(zeros_pca, axis=-1)[:, pca_type_sort]
            W_output_pca = np.concatenate(W_output_pca, axis=-1)[:, pca_type_sort]
            print('Overall:')
            print(' PR: {:.2f}'.format(
                participation_ratio_vector(explained_var_ratios)
            ))
            hs_pca = hs_pca.reshape(hs.shape)

        if session_iter == len(session_breakdown)-1:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            n_pcs_plot = 3

            if pca_type in ('full',):
                ax1.scatter(np.arange(n_activity), activity_pca.explained_variance_ratio_, color=c_vals[2])
                cutoff = np.sum(activity_pca.explained_variance_ratio_ > 0.1) # PC with > 0.1 

                for pc_idx in range(n_pcs_plot):
                    ax2.plot(activity_pca.components_[pc_idx, :], color=c_vals[pc_idx], label='PC{}'.format(pc_idx+1),
                            zorder=5-pc_idx)

            ax1.set_xlabel('PC')
            ax1.set_ylabel('Explained var ratio')
            ax1.set_title(f"Good PC: {cutoff}")

            for ax in (ax2, ax3):
                ax.legend()
                ax.set_xlabel('Neuron idx')
                ax.set_ylabel('PC weight')

            fig.savefig(f"./results/PC_{pca_target}_{ruleset}_{chosen_network}_{addon_name}.png")

        #######
        #######
        #######
        #######

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        figdomain, axdomain = plt.subplots(1, 3, figsize=(24, 8))
        figmatch, axmatch = plt.subplots(1, 3, figsize=(24, 8))


        # single_rule: color by labels
        # all_rules: color by rule

        # if multiple testrules are detected, automatically plot based on the rule type
        plot_mode = 'single_rule' if len(task_params["rules"]) == 1 else 'all_rules'

        if plot_mode == 'single_rule':
            rule_idx = 0 # Only used for individual labels
            rule = task_params['rules'][rule_idx]
            rule_batch_idxs = np.arange(n_batch_all)[test_rule_idxs == rule_idx]
        elif plot_mode == 'all_rules':
            rule_batch_idxs = np.arange(n_batch_all)

        if pca_target in ('hs'):
            pc1_idx, pc2_idx, pc3_idx = 0,1,2
        elif pca_target in ('Ms'):
            pc1_idx, pc2_idx, pc3_idx = 0,1,2

        # batch_plot = (0, 1, 2,)
        if task_params['dataset_name'] in ('DelayMatchSample-v0',):
            batch_plot = [False, False, False, False] # 4 distinct paths
        else:
            # batch_plot = (0, 1, 3,)
            batch_plot = (0,1,2)
        
        # ZIHAN
        # additional analysis
        if session_iter == len(session_breakdown)-1 and pca_target in ('hs', 'Ms'):
            all_normal_vectors = []
            # minus the zero activity pca
            pc_indices_consider = [pc1_idx, pc2_idx, pc3_idx]

            cosine_pc = np.hstack([W_output_pca[1, idx] - zeros_pca[:, idx] for idx in pc_indices_consider])
            # cosine_pc = np.vstack((W_output_pca[1,pc1_idx]-zeros_pca[:,pc1_idx], W_output_pca[1,pc2_idx]-zeros_pca[:,pc2_idx], W_output_pca[1,pc3_idx]-zeros_pca[:,pc3_idx])).reshape(-1)
            cosine_pc = cosine_pc / np.linalg.norm(cosine_pc)

            sine_pc = np.hstack([W_output_pca[2, idx] - zeros_pca[:, idx] for idx in pc_indices_consider])
            # sine_pc = np.vstack((W_output_pca[2,pc1_idx]-zeros_pca[:,pc1_idx], W_output_pca[2,pc2_idx]-zeros_pca[:,pc2_idx], W_output_pca[2,pc3_idx]-zeros_pca[:,pc3_idx])).reshape(-1)
            sine_pc = sine_pc / np.linalg.norm(sine_pc)

            normal_cosine_sine = np.cross(cosine_pc, sine_pc)
            # normalization
            normal_cosine_sine = normal_cosine_sine / np.linalg.norm(normal_cosine_sine)
            all_normal_vectors.append(normal_cosine_sine)

            figscatter = plt.figure(figsize=(21, 7))
            ax1_3d = figscatter.add_subplot(131, projection='3d')
            ax2_3d = figscatter.add_subplot(132, projection='3d')
            ax3 = figscatter.add_subplot(133)

            init_point_3d = np.stack((zeros_pca[:, pc1_idx], zeros_pca[:, pc2_idx], zeros_pca[:, pc3_idx]), axis=1)
            ax1_3d.scatter(init_point_3d[0,0], init_point_3d[0,1], init_point_3d[0,2], marker="s", color="black")

            ax1_3d.plot([init_point_3d[0,0], init_point_3d[0,0] + cosine_pc[0]],
                        [init_point_3d[0,1], init_point_3d[0,1] + cosine_pc[1]],
                        [init_point_3d[0,2], init_point_3d[0,2] + cosine_pc[2]],
                        color='red', label="cosine")

            ax1_3d.plot([init_point_3d[0,0], init_point_3d[0,0] + sine_pc[0]],
                        [init_point_3d[0,1], init_point_3d[0,1] + sine_pc[1]],
                        [init_point_3d[0,2], init_point_3d[0,2] + sine_pc[2]],
                        color='blue', label="sine")

            ax1_3d.plot([init_point_3d[0,0], init_point_3d[0,0] + normal_cosine_sine[0]],
                        [init_point_3d[0,1], init_point_3d[0,1] + normal_cosine_sine[1]],
                        [init_point_3d[0,2], init_point_3d[0,2] + normal_cosine_sine[2]],
                        color='green', label="cosine * sine")
            
            ax1_3d.legend()

            for ii in range(len(session_breakdown[:-1])):
                subsession = session_breakdown[ii]
                clustered_points = as_pca[:, subsession[0]:subsession[1], :]
                
                # Extract the principal components for the scatter plot
                pc1_points = clustered_points[:, :, pc1_idx].flatten()
                pc2_points = clustered_points[:, :, pc2_idx].flatten()
                pc3_points = clustered_points[:, :, pc3_idx].flatten()
                
                ax1_3d.scatter(pc1_points, pc2_points, pc3_points, c=c_vals[ii], alpha=0.15)

                points_3d = np.vstack((pc1_points, pc2_points, pc3_points)).T
                # points_3d_centered = points_3d - np.mean(points_3d, axis=0)
                scaler = StandardScaler()
                points_3d_standardized = scaler.fit_transform(points_3d)
                pca = PCA(n_components=3)
                pca.fit(points_3d_standardized)
                normal_vector = pca.components_[-1]
                normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)
                all_normal_vectors.append(normalized_normal_vector)

                point = np.mean(points_3d, axis=0)
                d = -point.dot(normal_vector)
                xx, yy = np.meshgrid(np.linspace(np.min(points_3d[:, 0]), np.max(points_3d[:, 0]), 10), 
                                    np.linspace(np.min(points_3d[:, 1]), np.max(points_3d[:, 1]), 10))
                zz = (-normal_vector[0] * xx - normal_vector[1] * yy - d) * 1. / normal_vector[2]
                
                ax2_3d.plot_surface(xx, yy, zz, alpha=0.5, color=c_vals_d[ii])

            for thisax in [ax1_3d, ax2_3d]:
                thisax.set_xlabel(f"PCA {pc1_idx+1}")
                thisax.set_ylabel(f"PCA {pc2_idx+1}")
                thisax.set_zlabel(f"PCA {pc3_idx+1}")

            nnn = len(all_normal_vectors)
            normal_align = np.zeros((nnn, nnn))
            for n1 in range(nnn):
                for n2 in range(nnn):
                    dot_product = np.dot(all_normal_vectors[n1], all_normal_vectors[n2])
                    angle_deg = np.degrees(np.arccos(dot_product))
                    # consider both alignment and anti-alignment
                    angle_deg = min(angle_deg, 180.0-angle_deg)
                    normal_align[n1,n2] = angle_deg

            np.fill_diagonal(normal_align, np.nan)
            # mark_labels = ["response"] + [f"period {i+1}" for i in range(nnn-1)]
            mark_labels = ["response"] + recordkyle_nameall[0]

            sns.heatmap(normal_align, cbar=True, annot=True, ax=ax3, cmap='coolwarm', xticklabels=mark_labels, yticklabels=mark_labels)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
            ax3.set_yticklabels(ax3.get_yticklabels(), rotation=45)

            figscatter.tight_layout()
            # figscatter.savefig(f"./results/zz_test_{pca_target}_{ruleset}_{chosen_network}_{addon_name}_scatter.png")

            # what if not based on PC -> Aug 3nd
            
            # middle of each session
            breaks_1afterstart = [int((cut[0]+cut[1])/2) for cut in session_breakdown[:-1]]

            # plot more frames 
            considerall = 0
            if considerall:
                for cut in session_breakdown[:-1]:
                    breaks_1afterstart.append(cut[0])
                    breaks_1afterstart.append(cut[1]-1)
            breaks_1afterstart = list(np.sort(breaks_1afterstart))
            print(session_breakdown)

            if pca_target in ('hs',): # in random
                figall, axsall = plt.subplots(3,2,figsize=(2*4,3*4))
                figavg, axsavg = plt.subplots(3,2,figsize=(2*4,3*4))

                assert mpn_depth == 1 # for now
                W_ = W_all_[0]

                # plot two trials
                figW, axsW = plt.subplots(len(breaks_1afterstart)+1,4,figsize=(12*4,4*(len(breaks_1afterstart)+1)))
                select_batch = [29,49,69,89]
                for pp in select_batch:
                    sns.heatmap(W_.T, cbar=True, cmap="coolwarm", ax=axsW[0,select_batch.index(pp)])
                    axsW[0,select_batch.index(pp)].set_title("W")
                    # take average across batches
                    # Ms_samplebatch = np.mean(Ms_orig, axis=0) 
                    Ms_samplebatch = Ms_orig[pp,:,:,:]
                    print(Ms_samplebatch.shape)

                    for bb in breaks_1afterstart:
                        sns.heatmap(Ms_samplebatch[bb,:,:].T, cbar=True, cmap="coolwarm", center=0, \
                                            vmin=-1, vmax=1, ax=axsW[breaks_1afterstart.index(bb)+1,select_batch.index(pp)])

                    for bb in breaks_1afterstart:
                        axsW[breaks_1afterstart.index(bb)+1,select_batch.index(pp)].set_title(f"M: Time: {bb}")
                        axsW[breaks_1afterstart.index(bb)+1,select_batch.index(pp)].set_xlabel(f"Neuron")
                        axsW[breaks_1afterstart.index(bb)+1,select_batch.index(pp)].set_ylabel(f"Input Index")

                figW.tight_layout()
                figW.savefig(f"./results/zz_test_{pca_target}_{ruleset}_{chosen_network}_{addon_name}_W.png")

                results = {
                    'fix_hs_Woutput_all': [],
                    'response_hs_Woutput_all': [],
                    'fix_Ms_Woutput_all': [],
                    'response_Ms_Woutput_all': [],
                    'fix_MsWs_Woutput_all': [],
                    'response_MsWs_Woutput_all': [],
                }

                pltout, axsout = plt.subplots(figsize=(6,6))
                sns.heatmap(W_output, ax=axsout, cmap="coolwarm", center=0, cbar=True, square=True)
                pltout.savefig(f"./results/zz_test_{pca_target}_{ruleset}_{chosen_network}_{addon_name}_Woutput.png")

                W_output_fixation = W_output[0,:].reshape(-1,1) # (N,1)
                W_output_response = W_output[1:3,:].T # (N,2)


                for batch_iter in range(hs.shape[0]):
                    hs_per_batch = hs[batch_iter,:,:]
                    Ms_per_batch = Ms_orig[batch_iter,:,:,:]

                    fix_hs_Woutput, response_hs_Woutput = [], []
                    fix_Ms_Woutput, response_Ms_Woutput = [], []
                    fix_MsWs_Woutput, response_MsWs_Woutput = [], []
                    
                    for time_cut in range(hs_per_batch.shape[0]):
                        hs_per_batch_per_time = hs_per_batch[time_cut,:].reshape(-1,1) # (N,1)
                        Ms_per_batch_per_time = Ms_per_batch[time_cut,:,:] # (N, n_input)

                        ang_fix_hs_Ws = subspace_angles(hs_per_batch_per_time, W_output_fixation)[0]
                        fix_hs_Woutput.append(np.degrees(ang_fix_hs_Ws))

                        ang_response_hs_Ws = subspace_angles(hs_per_batch_per_time, W_output_response)[0]
                        response_hs_Woutput.append(np.degrees(ang_response_hs_Ws))

                        try:
                            ang_fix_Ms_Ws = subspace_angles(Ms_per_batch_per_time, W_output_fixation)[0]
                            fix_Ms_Woutput.append(np.degrees(ang_fix_Ms_Ws))
                        except:
                            fix_Ms_Woutput.append(np.nan)
                        
                        try:
                            ang_response_Ms_Ws = subspace_angles(Ms_per_batch_per_time, W_output_response)[0]
                            response_Ms_Woutput.append(np.degrees(ang_response_Ms_Ws))
                        except:
                            response_Ms_Woutput.append(np.nan)


                        try:
                            ang_fix_MsWs_Ws = subspace_angles(Ms_per_batch_per_time * W_ + W_, W_output_fixation)[0]
                            fix_MsWs_Woutput.append(np.degrees(ang_fix_MsWs_Ws))
                        except:
                            fix_MsWs_Woutput.append(np.nan)
                        
                        try:
                            ang_response_MsWs_Ws = subspace_angles(Ms_per_batch_per_time * W_ + W_, W_output_response)[0]
                            response_MsWs_Woutput.append(np.degrees(ang_response_MsWs_Ws))
                        except:
                            response_MsWs_Woutput.append(np.nan)


                    results['fix_hs_Woutput_all'].append(fix_hs_Woutput)
                    results['response_hs_Woutput_all'].append(response_hs_Woutput)
                    results['fix_Ms_Woutput_all'].append(fix_Ms_Woutput)
                    results['response_Ms_Woutput_all'].append(response_Ms_Woutput)
                    results['fix_MsWs_Woutput_all'].append(fix_MsWs_Woutput)
                    results['response_MsWs_Woutput_all'].append(response_MsWs_Woutput)

                for key in results:
                    results[key] = np.array(results[key])
                
                result_key = list(results.keys())
                axsall = axsall.flatten()
                for batch_iter in range(hs.shape[0]):
                    for key in result_key:
                        axsall[result_key.index(key)].plot([x for x in range(results[key].shape[1])], results[key][batch_iter,:], color=c_vals[labels[batch_iter]])             
                        axsall[result_key.index(key)].set_title(key)

                for ax in axsall.flatten():
                    for bb in breaks:
                        ax.axvline(bb, color='r', linestyle="--")

                breaks = [0] + breaks
                axsavg = axsavg.flatten()
                for key in result_key:
                    means, stds = [], []
                    for bb in range(len(breaks)-1):
                        meanp = np.nanmean(results[key][:,breaks[bb]:breaks[bb+1]])
                        stdp = np.nanstd(results[key][:,breaks[bb]:breaks[bb+1]])
                        means.append(meanp)
                        stds.append(stdp)
                    means, stds = np.array(means), np.array(stds)
                    axsavg[result_key.index(key)].plot([x for x in range(len(means))], means, "-o")
                    axsavg[result_key.index(key)].fill_between([x for x in range(len(means))], means-stds, means+stds, alpha=0.3, color="red")
                    axsavg[result_key.index(key)].set_title(key)

                    
                figall.savefig(f"./results/zz_test_{ruleset}_{chosen_network}_{addon_name}_all2all.png")
                figavg.savefig(f"./results/zz_test_{ruleset}_{chosen_network}_{addon_name}_all2allavg.png")

        for ax, pc_idxs in zip((ax1, ax2, ax3),((pc1_idx, pc2_idx), (pc1_idx, pc3_idx), (pc2_idx, pc3_idx))):
            domain_save = []
            for batch_idx_idx, batch_idx in enumerate(rule_batch_idxs):
                if plot_mode == 'single_rule':
                    batch_color_idx = labels[batch_idx]
                elif plot_mode == 'all_rules':
                    batch_color_idx = int(test_rule_idxs[batch_idx])

                # MODIFY BY ZIHAN
                # only plot the categorization at the whole final period
                # where all sections are included in plotting
                if session_iter == len(session_breakdown)-1:
                    cutoff_evidence = recordkyle_all
                    
                    temp = []
                    for jj in range(len(cutoff_evidence[batch_idx])-1):
                        xx = as_pca[batch_idx, cutoff_evidence[batch_idx][jj]:cutoff_evidence[batch_idx][jj+1], pc_idxs[0]]
                        yy = as_pca[batch_idx, cutoff_evidence[batch_idx][jj]:cutoff_evidence[batch_idx][jj+1], pc_idxs[1]]
                        temp.append([xx, yy])
                        ax.scatter(xx, yy, color=c_vals_l[batch_color_idx], alpha=0.7, marker=markers_vals[jj], s=8)
                    domain_save.append(temp)

                    if batch_idx_idx in batch_plot:
                        for jj in range(len(cutoff_evidence[batch_idx])-1):
                            ax.plot(as_pca[batch_idx, cutoff_evidence[batch_idx][jj]:cutoff_evidence[batch_idx][jj+1], pc_idxs[0]], 
                                    as_pca[batch_idx, cutoff_evidence[batch_idx][jj]:cutoff_evidence[batch_idx][jj+1], pc_idxs[1]], 
                                    color=c_vals[batch_color_idx], 
                                    linestyle=l_vals[jj], alpha=1.0, zorder=10
                            )

                else:
                    ax.scatter(as_pca[batch_idx, :, pc_idxs[0]], as_pca[batch_idx, :, pc_idxs[1]], color=c_vals_l[batch_color_idx], alpha=0.7, marker="o", s=8)

            if pca_type in ('full',):
                ax.set_xlabel('PC {}'.format(pc_idxs[0]+1))
                ax.set_ylabel('PC {}'.format(pc_idxs[1]+1))
            elif pca_type in ('cell_types',):
                ax.set_xlabel('PC {} (cell type: {})'.format(pc_idxs[0]+1, cell_types_pca[pc_idxs[0]]))
                ax.set_ylabel('PC {} (cell type: {})'.format(pc_idxs[1]+1, cell_types_pca[pc_idxs[1]]))
            # Plot zero point
            ax.scatter(zeros_pca[:, pc_idxs[0]], zeros_pca[:, pc_idxs[1]], color='k',
                    marker='s')

            if ax == ax1:
                ax.legend()

            # Plot readouts
            ro_vector_dir_all = []
            init_points = []
            # if pca_target in ('Ms',):
            #     continue
            for out_idx, output_dim_label in enumerate(output_dim_labels):
                if pca_target in ('Ms',):
                    RO_SCALE = 5
                elif pca_target in ('hs',):
                    RO_SCALE = 1

                ro_vector_dir = np.array((
                    W_output_pca[out_idx, pc_idxs[0]] - zeros_pca[:, pc_idxs[0]],
                    W_output_pca[out_idx, pc_idxs[1]] - zeros_pca[:, pc_idxs[1]],
                ))

                norm = np.linalg.norm(ro_vector_dir)

                if norm != 0:
                    ro_vector_dir = ro_vector_dir / norm
                else:
                    ro_vector_dir = ro_vector_dir

                ro_vector_dir = RO_SCALE * ro_vector_dir
                ro_vector_dir_all.append(ro_vector_dir)

                init_point = [zeros_pca[:, pc_idxs[0]], zeros_pca[:, pc_idxs[1]]]

                ax.plot((zeros_pca[:, pc_idxs[0]], zeros_pca[:, pc_idxs[0]] + ro_vector_dir[0]),
                        (zeros_pca[:, pc_idxs[1]], zeros_pca[:, pc_idxs[1]] + ro_vector_dir[1]),
                        color=c_vals[out_idx], label=output_dim_label, zorder=9)

                init_points.append(init_point)
                
                ax.legend()

            # reload and regenerate the PCA projections separately

            # direction_vector_all = []
            # if session_iter == len(session_breakdown)-1 and pca_target in ('hs',):
            #     for evi in range(len(cutoff_evidence[batch_idx])-1):
            #         domain = [temp[evi] for temp in domain_save]
            #         domain_x = [batch_domain[0] for batch_domain in domain]
            #         domain_y = [batch_domain[1] for batch_domain in domain]
            #         flatten_domain_x = [item for sublist in domain_x for item in sublist]
            #         flatten_domain_y = [item for sublist in domain_y for item in sublist]
            #         axdomain[count].scatter(flatten_domain_x, flatten_domain_y, marker=markers_vals[evi], c=c_vals[evi])

            #         # directional analysis
            #         pts = [[flatten_domain_x[i], flatten_domain_y[i]] for i in range(len(flatten_domain_x))]
            #         points = np.array(pts)
            #         mean = np.mean(points, axis=0)
            #         centered_points = points - mean
            #         covariance_matrix = np.cov(centered_points, rowvar=False)
            #         eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            #         direction_vector = eigenvectors[:, np.argmax(eigenvalues)]
            #         direction_vector = direction_vector / np.linalg.norm(direction_vector)
            #         direction_vector_all.append(direction_vector)

            #         start = [1,1] # dummy initial point
            #         axdomain[count].plot((start[0], start[0] + direction_vector[0]), \
            #                             (start[1], start[1] + direction_vector[1]), 
            #                             c=c_vals[evi], linestyle="--")

            #         if evi == 0:
            #             for jjj in range(len(output_dim_labels)):
            #                 init_xx, init_yy = init_points[jjj][0], init_points[evi][1]
            #                 ro_vector_dir = ro_vector_dir_all[jjj]

            #                 axdomain[count].plot((init_xx, init_xx + ro_vector_dir[0]), \
            #                                     (init_yy, init_yy + ro_vector_dir[1]), \
            #                                     color=c_vals[jjj], label=output_dim_labels[jjj], zorder=9)

            # if session_iter == len(session_breakdown)-1 and pca_target in ('hs',):
            #     matching_angle = np.zeros((len(direction_vector_all), len(ro_vector_dir_all)))
            #     for v1 in range(len(direction_vector_all)):
            #         for v2 in range(len(ro_vector_dir_all)):
            #             vec1, vec2 = direction_vector_all[v1].flatten(), ro_vector_dir_all[v2].flatten()
            #             dot_product = np.dot(vec1, vec2)
            #             norm1 = np.linalg.norm(vec1)
            #             norm2 = np.linalg.norm(vec2)
            #             cosine_similarity = dot_product / (norm1 * norm2)
            #             matching_angle[v1,v2] = np.abs(cosine_similarity)

            #     sns.heatmap(matching_angle, annot=True, cmap='coolwarm', cbar=True, ax=axmatch[count])
            #     figmatch.savefig(f"./results/zz_test_{pca_target}_{ruleset}_{chosen_network}_{addon_name}_match.png")

                    
            # count += 1

        # if pca_type in ('cell_types',): # Some additional PC plots of only one cell type
        #     print('Yes')

        # if session_iter == len(session_breakdown)-1 and pca_target in ('hs',):
        #     for ax in axdomain:
        #         ax.legend()
                
        fig.savefig(f"./results/trajectory_{pca_target}_{ruleset}_{chosen_network}_{addon_name}_period{session_iter}.png")
        # figdomain.savefig(f"./results/zz_test_{pca_target}_{ruleset}_{chosen_network}_{addon_name}.png")


#######
#######
#######
#######

print(db.keys())

if len(rules_dict[ruleset]) > 1:
    hs_all = []
    if net_params['net_type'] in ('dmpn',):
        layer_idx_lst = [i for i in range(mpn_depth)]
        for layer_idx in layer_idx_lst:
            if layer_idx == 0:
                hs = db['hidden0'].detach().cpu().numpy()
            elif layer_idx == 1:
                hs = db['hidden1'].detach().cpu().numpy()
            hs_all.append(hs)
    elif net_params['net_type'] in ('gru',):
        layer_idx_lst = [0]
        hs_all = [db['hidden'].detach().cpu().numpy()]

    for hsiter in range(len(hs_all)):
        hs = hs_all[hsiter]
        layer_name = layer_idx_lst[hsiter]

        cell_vars_tot = np.var(hs, axis=(0, 1)) # Var over batch and sequence
        n_rules = len(task_params['rules'])
        n_cells = hs.shape[-1]

        # cell_vars_dtypes = [('rule{}'.format(rule_idx), np.float) for rule_idx in range(n_rules)]# Useful for sorting later
        cell_vars_rules = np.zeros((n_rules, n_cells,))
        cell_vars_rules_norm = np.zeros_like(cell_vars_rules)

        for rule_idx, rule in enumerate(task_params['rules']):
            print('Rule {} (idx {})'.format(rule, rule_idx))
            rule_hs = hs[test_rule_idxs == rule_idx, :, :]
            cell_vars_rules[rule_idx] = np.var(rule_hs, axis=(0, 1)) # Var over batch and sequence

        # Now normalize everything
        cell_max_var = np.max(cell_vars_rules, axis=0) # Across rules
        for rule_idx, rule in enumerate(task_params['rules']):
            cell_vars_rules_norm[rule_idx] = np.where(
                cell_max_var > 0., cell_vars_rules[rule_idx] / cell_max_var, 0.
            )

        # Now sort
        if n_rules > 1:
            rule0_vals = cell_vars_rules_norm[0].tolist()
            rule1_vals = cell_vars_rules_norm[1].tolist()

        rule01_vals = np.array(list(zip(rule0_vals, rule1_vals)), dtype=[('rule0', float), ('rule1', float)])
        sort_idxs = np.argsort(rule01_vals, order=['rule0', 'rule1'])[::-1]

        # sort_idxs = np.argsort(cell_vars_rules_norm[0])[::-1]
        cell_vars_rules_sorted_norm = cell_vars_rules_norm[:, sort_idxs]

        fig, ax = plt.subplots(2, 1, figsize=(12, 4*2))

        for rule_idx, rule in enumerate(task_params['rules']):
            ax[0].plot(cell_vars_rules_sorted_norm[rule_idx], color=c_vals[rule_idx],
                    label=task_params['rules'][rule_idx])

        ax[0].legend()

        ax[0].set_xlabel('Cell_idx')
        ax[0].set_ylabel('Norm. task var.')


        ax[1].matshow(cell_vars_rules_sorted_norm, aspect='auto', vmin=0.0, vmax=1.0,)
        ax[1].set_yticks(np.arange(n_rules))
        ax[1].set_yticklabels(task_params['rules'])
        ax[1].set_xlabel('Cell idx')
        fig.savefig(f"./results/categorization_{ruleset}_{chosen_network}_layer_{layer_name}_{addon_name}.png")