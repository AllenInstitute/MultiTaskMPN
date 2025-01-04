import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

import sys
import time
sys.path.append('/content/drive/MyDrive/neuro_research/mpns_more/')

import networks as nets # Contains RNNs
import net_helpers as net_helpers # A bunch helpers for both MPNs/RNNs
import tasks
import helper
import mpn

# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',]
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',]


# Reload modules if changes have been made to them
from importlib import reload

reload(nets)
reload(net_helpers)
# reload(analysis)

seed = 6

np.random.seed(seed)
torch.manual_seed(seed)

task_type = 'multitask' # int, NeuroGym, multitask
rules_dict = \
    {'all' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
     'low_dim' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
    #  'test' : ['delaygo', 'delayanti'], # ['fdgo', 'fdanti', 'delaygo', 'delayanti'],
     'test': ['fdgo', 'fdanti'],
     'test2' : ['delaydm1'],
     'delaygo': ['delaygo'],
     'delayanti': ['delayanti'],
     'fdgo': ['fdgo'],
     'reactgo': ['reactgo']
    }
    
ruleset = 'fdgo' # low_dim, all, test

# This can either be used to set parameters OR set parameters and train
train = True
verbose = True
run_mode = 'minimal' # minimal, debug

mpn_depth = 1

def current_basic_params():
    task_params = {
        'task_type': task_type,
        'rules': rules_dict[ruleset],
        'dt': 40, # ms, directly influence sequence lengths,
        'ruleset': ruleset,
        'n_eachring': 8, # Number of distinct possible inputs on each ring
        'in_out_mode': 'low_dim_pos',  # high_dim or low_dim or low_dim_pos (Robert vs. Laura's paper, resp)
        'sigma_x': 0.01, # Laura raised to 0.1 to prevent overfitting (Robert uses 0.01)
        'mask_type': 'cost', # 'cost', None
        'fixate_off': True, # Second fixation signal goes on when first is off
        'randomize_inputs': True,
        'n_input': 20, # Only used if inputs are randomized
    }

    train_params = {
        'lr': 1e-3,
        'n_batches': 64,
        'batch_size': 64,
        'gradient_clip': 10,
        'valid_n_batch': 10,
        'n_datasets': 100, # Number of distinct batches
        'n_epochs_per_set': 10, # longer/shorter training
        'task_mask': None, # None, task
        # 'weight_reg': 'L1',
        # 'reg_lambda': 1e-3,
    }

    n_hidden = 100

    net_params = {
        'net_type': 'dmpn', # mpn1, dmpn, vanilla
        'n_neurons': [1] + [n_hidden] * mpn_depth + [1],
        'output_bias': False, # Turn off biases for easier interpretation
        'loss_type': 'MSE', # XE, MSE
        'activation': 'tanh', # linear, ReLU, sigmoid, tanh, tanh_re, tukey, heaviside
        'cuda': True,
        'monitor_freq': 100,
        'monitor_valid_out': True, # Whether or not to save validation output throughout training

        'ml_params': {
            'bias': True, # Bias of layer
            'mp_type': 'mult',
            'm_update_type': 'hebb_assoc', # hebb_assoc, hebb_pre
            'eta_type': 'scalar', # scalar, pre_vector, post_vector, matrix
            'eta_train': True,
            # 'eta_init': 'mirror_gaussian', #0.0,
            'lam_type': 'scalar', # scalar, pre_vector, post_vector, matrix
            'm_time_scale': 400, # ms, sets lambda
            'lam_train': False,
        },
        # 'ml_params0': {
        #     'bias': False,
        #     'mp_type': 'mult',
        #     'm_update_type': 'hebb_pre', # hebb_assoc, hebb_pre
        #     'eta_type': 'scalar', # scalar, pre_vector, post_vector, matrix
        #     'eta_train': False,
        #     'eta_init': -1.0,
        #     'lam_type': 'scalar', # scalar, pre_vector, post_vector, matrix
        #     'lam_clamp': 0.99,
        #     'lam_train': False,
        # },

        # Vanilla RNN params
        'leaky': True,
        'alpha': 0.2,
    }

    return task_params, train_params, net_params

task_params, train_params, net_params = current_basic_params()

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

def train_network(params, net=None, device=torch.device('cuda'),
                  verbose=False,):

    task_params, train_params, net_params = params

    if task_params['task_type'] in ('multitask',):
        def generate_train_data(device='cuda'):
            train_data, _ = tasks.generate_trials_wrap(
                task_params, train_params['n_batches'], device=device, verbose=False
            )
            return train_data
        def generate_valid_data(device='cuda'):
            valid_data, _ = tasks.generate_trials_wrap(
                task_params, train_params['valid_n_batch'], rules=task_params['rules'], device=device
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

        net = netFunction(net_params, verbose=verbose)

    # Puts network on device
    net.to(device)

    # print(net.params)
    # for name, param in net.named_parameters():
    #     print('Name {} device {}'.format(name, param.device))

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
    plt.savefig("./results/loss.png")

    if net_params['loss_type'] in ('MSE',):
        helper.plot_some_ouputs(params, net)

else:
    print('Not training, set train=True to train.')

print('Done!')

#######
#######
#######
#######

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

fig.savefig("./results/eta_lambda.png")

#######
#######
#######
#######
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

    fig.savefig("./results/eta_distribution.png")

#######
#######
#######
#######

test_n_batch = 50 # Per rule evaluation

if task_params['task_type'] in ('multitask',): # Test batch consists of all the rules
    task_params['hp']['batch_size_train'] = test_n_batch
    test_data, test_trials_extra = tasks.generate_trials_wrap(task_params, test_n_batch, rules=task_params['rules'])
    _, test_trials, test_rule_idxs = test_trials_extra

    task_params['dataset_name'] = 'multitask'

    if task_params['in_out_mode'] in ('low_dim_pos',):
        output_dim_labels = ('Fixate', 'Cos', '-Cos', 'Sin', '-Sin')
    else:
        raise NotImplementedError()

    labels = []
    for rule_idx, rule in enumerate(task_params['rules']):
        print(rule)
        if rule in ('fdgo', 'fdanti', 'delaygo', 'delayanti','reactgo'):
            labels.append(test_trials[rule_idx].meta['resp1'])
        else:
            raise NotImplementedError()
    labels = np.concatenate(labels, axis=0)

test_input, test_output, test_mask = test_data
print(f"test_input: {test_input.shape}")

test_input_np = test_input.detach().cpu().numpy()
test_output_np = test_output.detach().cpu().numpy()

n_batch_all = test_input_np.shape[0] # Total number of batches, might be different than test_n_batch
print(n_batch_all)
max_seq_len = test_input_np.shape[1]

net_out, db = net.iterate_sequence_batch(test_input, run_mode='track_states')
print(db)


W_output = net.W_output.detach().cpu().numpy()

net_out = net_out.detach().cpu().numpy()

if net_params['loss_type'] in ('MSE',):
    test_out_np = test_output.cpu().numpy()

    fig, axs = plt.subplots(4, 1, figsize=(8, 8))

    if test_out_np.shape[-1] == 1:
        for batch_idx, ax in enumerate(axs):
            ax.plot(net_out[batch_idx, :, 0], color=c_vals[batch_idx])
            ax.plot(test_out_np[batch_idx, :, 0], color=c_vals_l[batch_idx], zorder=-1)

            ax.set_ylim((0, 2.25))
    else:
        for batch_idx, ax in enumerate(axs):
            for out_idx in range(test_out_np.shape[-1]):
                ax.plot(net_out[batch_idx, :, out_idx], color=c_vals[out_idx])
                ax.plot(test_out_np[batch_idx, :, out_idx], color=c_vals_l[out_idx], zorder=-1)

            ax.set_ylim((-3, 2.25))


    fig.savefig("./results/lowD.png")


#######
#######
#######
#######

print(db.keys())

if mpn_depth == 1:
    Ms = np.concatenate((
        db['M0'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1),
        # db['M1'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1)
    ), axis=-1)

    hs = np.concatenate((
        db['hidden0'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1),
    ), axis=-1)

else:
    raise NotImplementedError() # Let's focusing on one layer first
    matrices = []
    for i in range(mpn_depth):
        matrices.append(db[f'M{i}'].detach().cpu().numpy().reshape(n_batch_all, max_seq_len, -1))
    Ms = np.concatenate(matrices, axis=-1)

print(f"Ms.shape: {Ms.shape}")
print(f"hs.shape: {hs.shape}")

pca_type = 'full' # full, cell_types
pca_target_lst = ['hs', 'Ms'] # hs, 'Ms'

for pca_target in pca_target_lst:

    if net.hidden_cell_types is not None:
        cell_types = net.hidden_cell_types.detach().cpu().numpy()[0] # Remove batch idx
    cell_type_names = ('Inh.', 'Exc.')
    pca_sort_type = 'ratios' # How to sort cell type PCA; vars, ratios

    if pca_target in ('hs',):
        n_activity = hs.shape[-1]
        # Flatten hidden activity across batches and sequence
        as_shape = hs.shape # This will be the reshape used after PCA
        as_flat = hs.reshape((-1, n_activity,))
    elif pca_target in ('Ms',):
        # n_activity = Ms.shape[-2] * Ms.shape[-1]
        n_activity = Ms.shape[-1]
        # Just make sure this is flattening correctly, so doing multiple calls
        as_shape = (Ms.shape[0], Ms.shape[1], n_activity) # This will be the reshape used after PCA, modulations flattened
        # as_flat = Ms.reshape((Ms.shape[0], Ms.shape[1], n_activity)).reshape((-1, n_activity))
        as_flat = Ms.reshape((-1, n_activity))

    print(f"as_shape: {as_shape}")

    total_vars = np.var(as_flat, axis=0) # Var per activity dimension
    activity_zero = np.zeros((1, n_activity,))

    if pca_type in ('full',):
        activity_pca = PCA()

        activity_pca.fit(as_flat)

        # Corrects signs of components so that their mean/average is positive
        activity_pca.components_ = np.sign(np.sum(activity_pca.components_, axis=-1))[:, np.newaxis] * activity_pca.components_

        print(as_flat.shape)
        
        as_pca = activity_pca.transform(as_flat)
        print(as_pca.shape)
        as_pca = as_pca.reshape(as_shape) # Separates back into batches and sequences

        zeros_pca = activity_pca.transform(activity_zero)
        if pca_target in ('hs',): # Some transformations only make sense for certain activities
            W_output_pca = activity_pca.transform(W_output)

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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    n_pcs_plot = 3

    if pca_type in ('full',):
        ax1.scatter(np.arange(n_activity), activity_pca.explained_variance_ratio_, color=c_vals[2])

        for pc_idx in range(n_pcs_plot):
            ax2.plot(activity_pca.components_[pc_idx, :], color=c_vals[pc_idx], label='PC{}'.format(pc_idx+1),
                    zorder=5-pc_idx)

    ax1.set_xlabel('PC')
    ax1.set_ylabel('Explained var ratio')
    ax1.set_xlabel

    for ax in (ax2, ax3):
        ax.legend()
        ax.set_xlabel('Neuron idx')
        ax.set_ylabel('PC weight')

    fig.savefig(f"./results/PC_{pca_target}.png")

    #######
    #######
    #######
    #######

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    # single_rule: color by labels
    # all_rules: color by rule
    plot_mode = 'single_rule'

    if plot_mode == 'single_rule':
        rule_idx = 0 # Only used for individual labels
        rule = task_params['rules'][rule_idx]
        rule_batch_idxs = np.arange(n_batch_all)[test_rule_idxs == rule_idx]
    elif plot_mode == 'all_rules':
        rule_batch_idxs = np.arange(n_batch_all)

    if pca_target in ('hs'):
        pc1_idx, pc2_idx, pc3_idx = 0,2,3
    elif pca_target in ('Ms'):
        pc1_idx, pc2_idx, pc3_idx = 0,2,3

    # batch_plot = (0, 1, 2,)
    if task_params['dataset_name'] in ('DelayMatchSample-v0',):
        batch_plot = [False, False, False, False] # 4 distinct paths
    else:
        # batch_plot = (0, 1, 3,)
        batch_plot = (0,)

    for ax, pc_idxs in zip((ax1, ax2, ax3),((pc1_idx, pc2_idx), (pc1_idx, pc3_idx), (pc2_idx, pc3_idx))):
        for batch_idx_idx, batch_idx in enumerate(rule_batch_idxs):

            if plot_mode == 'single_rule':
                batch_color_idx = labels[batch_idx]
            elif plot_mode == 'all_rules':
                batch_color_idx = int(test_rule_idxs[batch_idx])

            ax.scatter(as_pca[batch_idx, :, pc_idxs[0]], as_pca[batch_idx, :, pc_idxs[1]],
                    color=c_vals_l[batch_color_idx], alpha=0.7, marker='.')

            if batch_idx_idx in batch_plot:
                ax.plot(as_pca[batch_idx, :, pc_idxs[0]], as_pca[batch_idx, :, pc_idxs[1]], color=c_vals[batch_color_idx],
                        alpha=1.0, zorder=10)

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
        if pca_target in ('Ms',):
            continue
        for out_idx, output_dim_label in enumerate(output_dim_labels):
            RO_SCALE = 1

            ro_vector_dir = np.array((
                W_output_pca[out_idx, pc_idxs[0]] - zeros_pca[:, pc_idxs[0]],
                W_output_pca[out_idx, pc_idxs[1]] - zeros_pca[:, pc_idxs[1]],
            ))
            # ax.plot((zeros_pca[:, pc_idxs[0]], zeros_pca[:, pc_idxs[0]] + ro_vector_dir[0]),
            #         (zeros_pca[:, pc_idxs[1]], zeros_pca[:, pc_idxs[1]] + ro_vector_dir[1]),
            #         color=c_vals[out_idx], label=output_dim_label, zorder=10)

            ro_vector_dir = RO_SCALE * ro_vector_dir

            ax.plot((zeros_pca[:, pc_idxs[0]], zeros_pca[:, pc_idxs[0]] + ro_vector_dir[0]),
                    (zeros_pca[:, pc_idxs[1]], zeros_pca[:, pc_idxs[1]] + ro_vector_dir[1]),
                    color=c_vals[out_idx], label=output_dim_label, zorder=9)

    if pca_type in ('cell_types',): # Some additional PC plots of only one cell type
        print('Yes')

    fig.savefig(f"./results/trajectory_{pca_target}.png")


#######
#######
#######
#######

if len(rules_dict[ruleset]) > 1:
    if net_params['net_type'] in ('dmpn',):
        layer_idx = 0
        if layer_idx == 0:
            hs = db['hidden0'].detach().cpu().numpy()
        elif layer_idx == 1:
            hs = db['hidden1'].detach().cpu().numpy()

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

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    for rule_idx, rule in enumerate(task_params['rules']):
        ax.plot(cell_vars_rules_sorted_norm[rule_idx], color=c_vals[rule_idx],
                label=task_params['rules'][rule_idx])

    ax.legend()

    ax.set_xlabel('Cell_idx')
    ax.set_ylabel('Norm. task var.')
    fig.savefig("./results/categorization1.png")

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    ax.matshow(cell_vars_rules_sorted_norm, aspect='auto', vmin=0.0, vmax=1.0,)
    ax.set_yticks(np.arange(n_rules))
    ax.set_yticklabels(task_params['rules'])
    ax.set_xlabel('Cell idx')
    fig.savefig("./results/categorization2.png")