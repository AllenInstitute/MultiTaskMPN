import time # For debugging
import copy 
import numpy as np
import matplotlib.pyplot as plt 
import math 
import seaborn as sns 
import copy 

import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import lognorm

import mpn_tasks 
import helper 

# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',]
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',]
l_vals = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
markers_vals = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

def train_network(params, net=None, device=torch.device('cuda'), verbose=False, \
    train=True, hyp_dict=None, netFunction=None, test_input=None):
    """
    """
    assert isinstance(test_input, list), "test_input must be a list"

    task_params, train_params, net_params = params

    if task_params['task_type'] in ('multitask',):
        
        def generate_train_data(device='cuda'):
            # ZIHAN
            # correct thing to do
            # "training batches should only be over one rule"
            # "but validation should mix them"
            if not hyp_dict['mess_with_training']:
                # print(f"Task Output with Strength: {task_params['label_strength']}")
                train_data, (_, train_trails, _ ) = mpn_tasks.generate_trials_wrap(
                    task_params, train_params['n_batches'], device=device, verbose=False, \
                        mode_input=hyp_dict['mode_for_all']
                )
            else:
                print("=== Mess with generating training data ===")
                train_data, (_, train_trails, _) = mpn_tasks.generate_trials_wrap(
                    task_params, train_params['n_batches'], rules=task_params['rules'], device=device, verbose=False, \
                        mode_input=hyp_dict['mode_for_all'], mess_with_training=True
                )

            return train_data, train_trails
            
        def generate_valid_data(device='cuda'):
            valid_data, (_, valid_trails, _) = mpn_tasks.generate_trials_wrap(
                task_params, train_params['valid_n_batch'], rules=task_params['rules'], device=device, mode_input=hyp_dict['mode_for_all']
            )
            return valid_data, valid_trails
            
    else:
        raise ValueError('Task type not recognized.')

    if net is None: # Create a new network
        net = netFunction(net_params, verbose=verbose)

    # Puts network on device
    net.to(device)

    # check_net_cell_types(net) # Pre-train cell types check

    valid_data, valid_trails = generate_valid_data(device=device)

    counter_lst = []
    # to customize if multiple test data are used
    netout_lst, db_lst  = [[] for _ in range(len(test_input))], [[] for _ in range(len(test_input))]
    Winput_lst, Woutput_lst, Winputbias_lst, Wall_lst, marker_lst, loss_lst, acc_lst = [], [], [], [], [], [], []
    net_lst = []

    def is_power_of_2_or_zero(n):
        return n == 0 or (n > 0 and (n & (n - 1)) == 0)

    for dataset_idx in range(train_params['n_datasets']):
        # Regenerate new data
        train_data, train_trails = generate_train_data(device=device)
        new_thresh = True if dataset_idx == 0 else False
        if train: 
            if test_input is not None and (is_power_of_2_or_zero(dataset_idx) or dataset_idx == train_params['n_datasets'] - 1):
                print(f"How about Test Data at dataset {dataset_idx}")
                counter_lst.append(dataset_idx)
                # test data for each stage
                for test_input_index, test_input_ in enumerate(test_input): 
                    net_out, net_hidden, db = net.iterate_sequence_batch(test_input_, run_mode='track_states')
                    net_out = net_out.detach().cpu().numpy()
                    netout_lst[test_input_index].append(net_out)
                    db_lst[test_input_index].append(db)
                
                Woutput_lst.append(net.W_output.detach().cpu().numpy())
                
                if net_params["input_layer_add"] and net_params["net_type"] == "dmpn":
                    Winput_lst.append(net.W_initial_linear.weight.detach().cpu().numpy())
                    
                    if net_params["input_layer_bias"]:
                        Winputbias_lst.append(net.W_initial_linear.bias.detach().cpu().numpy())
                        
                elif net_params["input_layer"] and net_params["net_type"] == "vanilla":
                    Winput_lst.append(net.W_input.detach().cpu().numpy())
                    # no input bias is set for vanilla RNN in kyle's original design

                W_all_ = []
                if params[2]["net_type"] == "dmpn":
                    for i in range(len(net.mp_layers)):
                        W_all_.append(net.mp_layers[i].W.detach().cpu().numpy())
                Wall_lst.append(W_all_)

                marker_lst.append(dataset_idx)

            _, monitor_loss, monitor_acc = net.fit(train_params, train_data, train_trails, valid_batch=valid_data, valid_trails=valid_trails, new_thresh=new_thresh, run_mode=hyp_dict['run_mode'])

            if test_input is not None and (is_power_of_2_or_zero(dataset_idx) or dataset_idx == train_params['n_datasets'] - 1):
                # print(f"monitor_loss: {monitor_loss}")
                loss_lst.append(monitor_loss)
                acc_lst.append(monitor_acc)
		
    return net, (train_data, valid_data), (counter_lst, netout_lst, db_lst, Winput_lst, Winputbias_lst, \
        		Woutput_lst, Wall_lst, marker_lst, loss_lst, acc_lst)

def net_eta_lambda_analysis(net, net_params, hyp_dict=None, verbose=False):
    """
    """
    layer_index = 0 # 1 layer MPN 
    if net_params["input_layer_add"]:
        layer_index += 1 # 2 layer MPN
  
    # only make sense for dmpn for eta and lambda information extraction
    if net_params['net_type'] in ("dmpn",):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        for mpl_idx, mp_layer in enumerate(net.mp_layers):
            if net.mp_layers[mpl_idx].eta_type in ('pre_vector', 'post_vector', 'matrix',):
                full_eta = np.concatenate([
                    eta.flatten()[np.newaxis, :] for eta in net.hist['eta{}'.format(mpl_idx+layer_index)]
                ], axis=0)
            else:
                full_eta = net.hist['eta{}'.format(mpl_idx+layer_index)]
    
            if net.mp_layers[mpl_idx].lam_type in ('pre_vector', 'post_vector', 'matrix',):
                full_lam = np.concatenate([
                    lam.flatten()[np.newaxis, :] for lam in net.hist['lam{}'.format(mpl_idx+layer_index)]
                ], axis=0)
            else:
                full_lam = net.hist['lam{}'.format(mpl_idx+layer_index)]
            ax1.plot(net.hist['iters_monitor'], full_eta, color=c_vals[mpl_idx], label='MPL{}'.format(mpl_idx+layer_index))
            ax2.plot(net.hist['iters_monitor'], full_lam, color=c_vals[mpl_idx], label='MPL{}'.format(mpl_idx+layer_index))

        ax1.axhline(0.0, color='k', linestyle='dashed')
    
        ax1.set_ylabel('Eta')
        ax2.set_ylabel('Lambda')
    
        if net.mp_layers[mpl_idx].eta_type not in ('pre_vector', 'post_vector', 'matrix',):
            ax1.legend()
            ax2.legend()

        if verbose:
            fig.savefig(f"./results/eta_lambda_{hyp_dict['ruleset']}_{hyp_dict['chosen_network']}_{hyp_dict['addon_name']}.png")
    
        # only for deep mpn with multiple layers
        n_mplayers = len(net.mp_layers)
        if n_mplayers > 1:
            fig, axs = plt.subplots(1, n_mplayers, figsize=(4+4*n_mplayers, 4))
            n_bins = 50
    
            for mpl_idx, (mp_layer, ax) in enumerate(zip(net.mp_layers, axs)):
    
                init_eta = net.hist['eta{}'.format(mpl_idx+layer_index)][0].flatten()
                final_eta = net.hist['eta{}'.format(mpl_idx+layer_index)][-1].flatten()
    
                max_eta = np.max((np.max(np.abs(init_eta)), np.max(np.abs(final_eta))))
    
                bins = np.linspace(-max_eta, max_eta, n_bins+1)[1:]
    
                ax.hist(init_eta, bins=bins, color=c_vals_l[mpl_idx], alpha=0.3, label='init')
                ax.hist(final_eta, bins=bins, color=c_vals[mpl_idx], alpha=0.3, label='final')
    
                ax.set_ylabel('Count')
                ax.set_xlabel('Eta value')
                ax.legend()

            if verbose:
                fig.savefig(f"./results/eta_distribution_{hyp_dict['ruleset']}_{hyp_dict['chosen_network']}_{hyp_dict['addon_name']}.png")

def rand_weight_init(n_inputs, n_outputs=None, init_type='gaussian', cell_types=None,
					 ei_balance_val=None, sparsity=None, weight_norm=None, self_couplings=True):
	"""
	Returns a random weight initialization of the specified type, of size
	(n_inputs,) if n_ouputs is None, otherwise (n_outputs, n_inputs).

	Note: uses numpy throughout, converted to tensor after called

	ei_balance_val: balances strength of excitation and inhibition
	spasity: override sparsity 

	self_couplings: whether or not diagonal couplings can be nonzero

	"""
	sparsity_p = 1.0 if sparsity is None else sparsity

	if n_outputs is not None: # 2d case
		weight_shape = (n_outputs, n_inputs,)
	else: # 1d case
		weight_shape = (n_inputs,)

	if init_type == 'xavier':
		if n_outputs is not None: # 2d case
			xavier_bound = np.sqrt(6/(n_inputs + n_outputs)) if weight_norm is None else weight_norm
		else: # 1d case
			xavier_bound = np.sqrt(6/(n_inputs)) if weight_norm is None else weight_norm
		rand_weights = np.random.uniform(low=-xavier_bound, high=xavier_bound, size=weight_shape)
	elif init_type in ('gaussian', 'sparse_gaussian', 'sparse_gaussian_ln_in'):
		norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm
		rand_weights = norm_factor * np.random.normal(scale=1.0, size=weight_shape)
	elif init_type in ('mirror_gaussian',): # Two gaussians peaks centered at opposite means (fixed mean/scale ratio for now)
		norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm
		rand_weights = norm_factor * (
			np.random.choice([-1, 1], size=weight_shape) * np.random.normal(loc=1.0, scale=0.5, size=weight_shape)
		)
	elif init_type in ('log_normal', 'sparse_log_normal'):
		norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm # if init_type == 'log_normal' else n_outputs * sparsity_p
		rand_weights = norm_factor * np.random.lognormal(mean=0.0, sigma=1.0, size=weight_shape)
	elif init_type in ('ones', 'sparse_ones') or type(init_type) == float:
		if n_outputs is not None: # 2d case
			if n_outputs > 1:
				raise NotImplementedError('Normalization is weird for this, should think about it if we use this again')
			norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm #if init_type == 'ones' else n_outputs * sparsity_p
		else: # 1d case
			norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm #if init_type == 'ones' else n_inputs * n_outputs * sparsity_p
		if init_type in ('ones', 'sparse_ones'):
			rand_weights = norm_factor * np.ones(weight_shape)
		else:
			rand_weights = init_type * np.ones(weight_shape)
	elif init_type in ('rand_one_hot',):
		assert len(weight_shape) == 2
		shuffle_idxs = np.random.permutation(max(weight_shape))
		# Shuffle an identity matrix that is the larger of the two weight dimensions
		square_weights = np.eye(max(weight_shape))[shuffle_idxs, :]
		# Clip to appropriate size
		rand_weights = square_weights[:weight_shape[0], :weight_shape[1]]
	elif init_type in ('zeros',):
		rand_weights = np.zeros(weight_shape)
	else:
		raise NotImplementedError('Random weight init type {} not recognized!'.format(init_type))

	# Creates sparsification masks (masks that determine which elements are zero)
	if init_type in ('sparse_gaussian', 'sparse_log_normal', 'sparse_ones', 'sparse_gaussian_ln_in'):

		if init_type in ('sparse_gaussian', 'sparse_ones',): # Note this creates an in-degree distribution that is Gaussian distributed
			weight_mask = np.random.uniform(0, 1, size=rand_weights.shape) > sparsity_p # Which weights to zero
		elif init_type in ('sparse_log_normal', 'sparse_gaussian_ln_in',): 
			LOG_NORM_SHAPE = 0.25 # This could be adjusted to experimental data eventually

			# Adjusts scale relative to standard log normal to set desired mean
			standard_ln = lognorm(s=LOG_NORM_SHAPE)
			scaled_ln = lognorm(s=LOG_NORM_SHAPE, scale=sparsity_p / standard_ln.mean())

			# Draws twice as many as needed since will be truncating all > 1.0
			in_degrees = scaled_ln.rvs(size=2*n_outputs)
			in_degrees = in_degrees[in_degrees <=1.0] # Truncates to <1.0 in degrees

			if in_degrees.shape[0] < n_outputs:
				raise ValueError('Did not draw enough in-degrees that met threshold (only {}).'.format(
					in_degrees.shape[0]
				))

			weight_mask = np.zeros(weight_shape, dtype=bool)
			for out_idx in range(n_outputs): # Generates mask one output at a time
				in_degree_mask = np.zeros((n_inputs,), dtype=bool) # True for elements set to zero
				# Set all elements beyond index to be True, corresponding to no connection
				# (small in_degree[out_idx] corresponds to majority 1s)
				in_degree_mask[int(np.floor(in_degrees[out_idx] * n_inputs)):] = True
				np.random.shuffle(in_degree_mask)

				weight_mask[out_idx] = in_degree_mask

		if not self_couplings: # Sets all diagonal elements of mask so weights are set to zero
			assert n_inputs == n_outputs
			for neuron_idx in range(n_inputs):
				weight_mask[neuron_idx, neuron_idx] = True

		rand_weights[weight_mask] = 0.0

	if (cell_types is not None) and (n_outputs is not None): # Assigns cell types
		cell_types = cell_types.detach().cpu().numpy()
		assert cell_types.shape[0] == 1
		assert cell_types.shape[1] == n_inputs

		if ei_balance_val is not None:
			 # Signed based on cell type, but also enhances strength appropriately
			cell_weights = np.where(cell_types < 0, ei_balance_val * cell_types, cell_types)
		else:
			cell_weights = cell_types # Just 1s or -1s to correct sign

		rand_weights = cell_weights * np.abs(rand_weights)

	# print(' Init type:', init_type)
	# print(' Cell types:', cell_types)
	# perc_pos = np.sum(rand_weights > 0) / np.prod(rand_weights.shape)
	# perc_neg = np.sum(rand_weights < 0) / np.prod(rand_weights.shape)
	# print(' Perc pos: {:.2f} perc neg: {:.2f}'.format(perc_pos, perc_neg))

	return rand_weights

def append_to_average(avg_raw, raw, n_window=1):
	""" Rolling average """
	if len(raw) < n_window:
		avg_raw.append(np.mean(raw))
	else:
		avg_raw.append(np.mean(raw[-n_window:]))
	return avg_raw

def accumulate_decay(decay_raw, raw, n_window=10, base_val=0.0):
	""" Average by accumulation and decay """
	gamma = 1. - 1./n_window
	
	if len(decay_raw) == 0: # Use base_val for previous value
		if np.isnan(raw[-1]): # Skips nans and just set to base value
			decay_raw.append(base_val)
		else:
			decay_raw.append(gamma*base_val + 1/n_window * raw[-1])
	else:
		if np.isnan(raw[-1]): # Skips nans and just holds decay_raw constant
			decay_raw.append(decay_raw[-1])
		else:
			decay_raw.append(gamma*decay_raw[-1] + 1/n_window * raw[-1])
		# print('Raw {} new avg {}'.format(decay_raw[-1], decay_raw[-1]))
	return decay_raw

def relu_fn(x):
	return torch.maximum(x, torch.zeros_like(x))
def relu_fn_np(x):
	return np.maximum(x, np.zeros_like(x))
def relu_fn_p(x):
	return torch.heaviside(x, torch.zeros_like(x)) # For x = 0, return 0 just like pytorch default

def sigmoid(x):
	return 1 / (1 + torch.exp(-x))    
def sigmoid_np(x):
	return 1 / (1 + np.exp(-x))
def sigmoid_p(x):
	sx = sigmoid(x)
	return sx * (1 - sx)

def tanh_p(x):
	return (1/torch.cosh(x))**2 # Sech**2

def tanh_re(x):
	return torch.maximum(torch.tanh(x), torch.zeros_like(x))
def tanh_re_np(x):
	return np.maximum(np.tanh(x), np.zeros_like(x))    
def tanh_re_p(x): # Have to use where here instead of maximum because we want this to return 0 for x = 0, like ReLU (and sech(0) = 1)
	return torch.where(x > 1e-6, (1/torch.cosh(x))**2, 0.) # Sech**2

def tanh_re_super(x, alpha=2.0):
	return torch.maximum(torch.tanh(alpha*x), torch.zeros_like(x))
def tanh_re_super_np(x, alpha=2.0):
	return np.maximum(np.tanh(alpha*x), np.zeros_like(x))    
def tanh_re_super_p(x): # Have to use where here instead of maximum because we want this to return 0 for x = 0, like ReLU (and sech(0) = 1)
	raise NotImplementedError()
	return torch.where(x > 0., (1/torch.cosh(x))**2, 0.) # Sech**2

def linear_fn(x):
	return x     
def linear_fn_p(x):
	return torch.ones_like(x)

def cubed_re(x):
	return torch.maximum(x**3, torch.zeros_like(x))
def cubed_re_p(x):
	return torch.maximum(3*x**2, torch.zeros_like(x))

def tukey_fn(x):
	raw = 1/2 - 1/2 * torch.cos(np.pi * x) 
	raw[x < 0.] = 0.0
	raw[x > 1.] = 1.0
	return raw
def tukey_fn_np(x):
	raw = 1/2 - 1/2 * np.cos(np.pi * x) 
	raw[x < 0.] = 0.0
	raw[x > 1.] = 1.0
	return raw
def tukey_fn_p(x):
	raw = np.pi / 2 * torch.sin(np.pi * x)
	raw[x < 0.] = 0.0
	raw[x > 1.] = 0.0

def heaviside_p(x):
	raise NotImplementedError()

def get_activation_function(act_fn):
	""" Returns pytorch version, numpy version, and pytorch derivative functions """
	if act_fn == 'ReLU':
		return relu_fn, relu_fn_np, relu_fn_p
	elif act_fn == 'sigmoid':
		return sigmoid, sigmoid_np, sigmoid_p
	elif act_fn == 'tanh_re':
		return tanh_re, tanh_re_np, tanh_re_p
	elif act_fn == 'tanh_re_super': # Supra linear version of Tanh
		return tanh_re_super, tanh_re_super_np, tanh_re_super_p
	elif act_fn == 'tanh':
		return torch.tanh, np.tanh, tanh_p
	elif act_fn == 'tukey':
		return tukey_fn, tukey_fn_np, tukey_fn_p
	elif act_fn == 'linear':
		return linear_fn, linear_fn, linear_fn_p
	elif act_fn == 'cubed_re':
		return cubed_re, None, cubed_re_p
	elif act_fn == 'heaviside':
		return lambda x : torch.heaviside(x, torch.tensor(0.5)), lambda x : np.heaviside(x, 0.5), heaviside_p
	else:
		raise ValueError('Activation function: {} not recoginized!'.format(act_fn))

def cosine_similarity_loss(output, pred):
	"""
	Cosine similiarty loss, 
	expects output and pred to both be: (B, Ny)
	"""
	cosine_sim = nn.CosineSimilarity(dim=-1)

	return torch.mean(torch.abs(cosine_sim(output, pred)))

def mse_loss_weighted(output, pred, weight):
	"""
	Weighted version of MSE loss. For fitting curves weighted by mean squared error.
	"""  
	return torch.mean(weight * (output - pred) ** 2)
	

def shuffle_dataset(inputs, labels, masks=None):
	""" Shuffles a dataset over its batch index """
	
	assert inputs.shape[0] == labels.shape[0] # Checks batch indexes are equal
	shuffle_idxs = np.arange(inputs.shape[0])
	np.random.shuffle(shuffle_idxs)
	inputs = inputs[shuffle_idxs, :, :]
	labels = labels[shuffle_idxs, :, :]
	
	if masks is not None:
		assert inputs.shape[0] == masks.shape[0]
		masks = masks[shuffle_idxs, :, :]
	else:
		masks = None

	return inputs, labels, masks

def round_to_values(array, round_vals):
	""" Round all elements of an array to closest value in round_vals, numpy version """

	assert len(round_vals.shape) == 1

	dims_unsqueeze = [i for i in range(len(array.shape))] # How many times to unsqueeze round_vals based on array shape
	dists = np.abs(
		np.expand_dims(array, axis=-1) - # shape: (array.shape*, 1)
		np.expand_dims(round_vals, axis=dims_unsqueeze) 
	)

	return round_vals[np.argmin(dists, axis=-1)]

def round_to_values_torch(array, round_vals):
	""" Round all elements of an array to closest value in round_vals, pytorch version """

	assert len(round_vals.shape) == 1

	round_vals_us = torch.clone(round_vals) # Copy of round_vals to be unsqueezed
	dims_unsqueeze = len(array.shape) # How many times to unsqueeze based on array shape
	for _ in range(dims_unsqueeze):
		round_vals_us = round_vals_us.unsqueeze(0)
	dists = torch.abs(
		array.unsqueeze(-1) - round_vals_us
	)

	return round_vals[torch.argmin(dists, axis=-1)]

##############################################################################################
##############################################################################################
##############################################################################################

class BaseNetworkFunctions(nn.Module):
	"""
	Functions that may be used in networks and subnetwork layers. 

	Separate from BaseNetwork so that certain subnetwork layers can use these functions without having
	to fully init a new network each time.
	"""

	def __init__(self):

		super().__init__()


	def parameter_or_buffer(self, name, tensor=None, verbose=False):
		"""
		Helper function to register certain variables as parameters or buffers in networks.
		Also has capability of updating prameters to buffers or vice versa, in which case 
		a tensor value does not need to be passed

		If "name" is in the "self.params" tuple, then registers as parameter, otherwise buffer.
		"""

		# If the attribute already exists, copy it to tensor then delete it
		update_name = True

		if hasattr(self, name): # Automatrically update if it doesnt exist, otherwise some checks

			# Find out if this is a parameter or buffer (probably a better way to do this)
			param = False
			buffer = False
			for buff_name, _ in self.named_buffers():
				if buff_name == name:
					buffer = True
			for param_name, _ in self.named_parameters():
				if param_name == name:
					param = True

			assert not (buffer and param)

			# If already the correct type
			if (buffer and (name not in self.params)) or (param and (name in self.params)):
				update_name = False
			else:
				if verbose: print('Updating atribute: ', name)
				tensor = getattr(self, name).detach().clone()
				delattr(self, name)

		if update_name:
			if name in self.params:
				self.register_parameter(name, nn.Parameter(tensor))
			else:
				self.register_buffer(name, tensor)


class BaseNetwork(BaseNetworkFunctions):

	def __init__(self, net_params, verbose=False):

		super().__init__()
		
		self.loss_type = net_params.get('loss_type', 'XE')

		if self.loss_type == 'XE':
			self.loss_fn = F.cross_entropy
		elif self.loss_type in ('MSE', 'MSE_grads',):
			self.loss_fn = F.mse_loss
		elif self.loss_type == 'MSE_weighted':
			self.loss_fn = mse_loss_weighted
		elif self.loss_type == 'CS': # cosine similarity
			self.loss_fn = cosine_similarity_loss
		elif self.loss_type in ('rl_direct_output', 'rl_output_deviations',): # RL loss functions implemented manually for now
			self.loss_fn = None
		else:
			raise ValueError('Loss type {} not recognized.'.format(self.loss_type))

		# Prefs can be used to compute approximate accuracy in loss setups where accuracy is not well-defined
		if self.loss_type in ('MSE', 'MSE_grads', 'MSE_weighted', 'CS',):
			if net_params.get('prefs', None) is not None:
				self.register_buffer('prefs', torch.tensor(net_params.get('prefs')))
			else:
				self.prefs = None

		self.verbose = verbose
		self.param_clamping = False # Will be overridden if needed.

		# Time difference between discrete steps, used to set biologically-realistic parameters
		self.dt = net_params.get('dt', 40) # units of ms

		self.hist = None

		self.monitor_freq = net_params.get('monitor_freq', 1000)
		self.monitor_valid_out = net_params.get('monitor_valid_out', 10)   

	def fit(self, train_params, train_data, train_trails, valid_batch=None, valid_trails=None,new_thresh=True, run_mode='minimal'):
		"""
		Fit a network to the train_data using the train_parameters. 

		INPUTS:
		train_data: either a labelled dataset OR an envicorment

		run_mode: Controls exactly what is collected and reported back during full fit
		- 'minimal': Minimal runthrough
		- 'track_states': Collect additional data during run through for analysis, sometimes needed for different gradient types

		"""

		self.train_type = train_params.get('train_type', 'supervised')
		self.gradient_type = train_params.get('gradient_type', 'backprop')

		if self.gradient_type in ('3_factor_hebbian', '3_factor_hebbian_trunc', 'full_debug'):
			self.rho = train_params.get('rho', None) # Subtraction from postsynaptic activity, useful for positive definite activations

		# How to combine errors and compute gradients (e.g. error for each time versus mean over time)
		self.error_reduction = train_params.get('error_reduction', 'mean')
		if self.train_type in ('supervised',):
			train_fn = self.train_epochs
		elif self.train_type in ('rl', 'rl_test',):
			raise NotImplementedError('Removed from this code base.')
		else:
			raise NotImplementedError('Training type {} not recognized.'.format(self.train_type))

		# Regularization
		self.weight_reg = train_params.get('weight_reg', None)
		self.reg_lambda = train_params.get('reg_lambda', 0.0)
		self.activity_reg = train_params.get('activity_reg', None) # Activity regularization
		self.reg_omit = train_params.get('reg_omit', []) # List of named parameters to omit from regularization

		if new_thresh or self.hist is None:
			init_string = 'Train parameters:'

			# Init optimizer, sometimes even for unsupervised training just in case parameter adjustment is needed for other reasons
			self.lr = train_params.get('lr', 1e-3)
			self.optim_type = train_params.get('optimizer', 'adam')
			if list(self.parameters()) != []:
				if self.optim_type in ('adam',):
					self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)      
				elif self.optim_type in ('sgd',):
					self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) 

			if self.train_type in ('supervised', 'rl',):
				init_string += '\n  Loss: {} // LR: {:.2e} // Optim: {}\n  Grad type: {} '.format(
					self.loss_type, self.lr, self.optim_type, self.gradient_type
				)

			self.gradient_clip = train_params.get('gradient_clip', None)
			if self.gradient_clip is not None and self.gradient_type in ('backprop', 'full_debug',):
				init_string += '// Gradient clip: {:.1e}'.format(self.gradient_clip)      
			else:
				init_string += '// Gradient clip: None'

			if self.weight_reg is not None:
				init_string += '\nWeight reg: {}, coef: {:.1e}'.format(
					self.weight_reg, self.reg_lambda
				)
			else:
				init_string += '\nWeight reg: None'
    
			if self.activity_reg is not None:
				init_string += '\nActivity reg: {}, coef: {:.1e}'.format(
					self.weight_reg, self.reg_lambda
				)
			else:
				init_string += '\nActivity reg: None'

			if self.verbose:
				print(init_string) 

		self.train() # put in train mode (doesn't really do anything unless we are using dropout/batch norm)
		db, monitor_loss, monitor_acc = train_fn(train_params, train_data, train_trails, valid_batch=valid_batch, valid_trails=valid_trails,
					  new_thresh=new_thresh, run_mode=run_mode)
        
		self.eval() # return to eval mode

		return db, monitor_loss, monitor_acc
			 
	
	def train_base(self, train_params, train_data, train_trails=None, valid_batch=None, valid_trails=None, new_thresh=True, 
				   monitor=True, run_mode='minimal'):
		"""
		Some training attributes and checks that are shared across different train types 
		(i.e. both supervised and rl setups). Performs initial monitor that initializes
		many tracked values.
		
		train_type: supervised, unsupervised, rl, rl_test
		output_credit_assign: optional different credit assignment across an output layer  

		"""

		# Trigger these only on first train call
		if not hasattr(self, 'output_credit_assign'):
			self.output_credit_assign = train_params.get('output_credit_assign', 'W_output')

		if self.train_type in ('supervised', 'rl',): # Only relevant if training
			assert self.gradient_type in ('backprop',)
			assert self.output_credit_assign in ('W_output',) # Backprop without this credit assignment not implemented
			self.run_mode = run_mode
		elif self.train_type in ('unsupervised',): # Not training, sets parameters to default
			self.run_mode = run_mode
		else:
			raise NotImplementedError('Training type {} not recognized.'.format(self.train_type))
		

		# Whether or not to do one final monitor at the end of the train function 
		self.end_monitor = train_params.get('end_monitor', False)

		if new_thresh or self.hist is None:
			self._monitor_init(train_params, train_data, train_trails, valid_batch=valid_batch, valid_trails=valid_trails)

		# A quick pre-training check to make sure all parameters are named
		# (named parameters are used in regularization, so this can be removed if not regularizing)
		assert sum(1 for _ in self.named_parameters()) == sum(1 for _ in self.parameters())

	# @torch.no_grad()
	def train_epochs(self, train_params, train_data, train_trails, valid_batch=None, valid_trails=None, new_thresh=True, 
					 monitor=True, run_mode='minimal'):
		"""
		This will either perform supervised/unsupervised training on the train_data. In both cases, the input data represents 
		some sequence of inputs, and the M matrices will go through the training during said sequence. For the case 
		of supervised training, the weights of the network will be trained as well. 

		Each batch is assumed to keep track of its own internal state (the SM matrix).

		INPUTS:
		train_data = train_inputs, train_labels, train_masks
		  train_inputs shape: (batches, seq_len, input_size)
		  train_labeles: None for 'unsupervised'. For 'supervised': 
			shape: (batches, seq_len)
		  train_masks shape: (batches, seq_len, output_size) used for calculating loss and uneven sequence lengths
		valid_batch = 
		"""

		self.train_base(train_params, train_data, train_trails=train_trails, valid_batch=valid_batch, valid_trails=valid_trails, new_thresh=new_thresh, 
						monitor=monitor, run_mode=run_mode)

		assert len(train_data) == 3
		train_inputs, train_labels, train_masks = train_data
		# assert len(train_trails) == 1 # let's do single trail first...
		train_go_info = train_trails[0].epochs['go1']
		valid_go_info = valid_trails[0].epochs['go1']

		B = train_params['batch_size']

		if self.train_type in ('supervised',): # Checks to matke sure labels and inputs are same size
			assert train_inputs.shape[0] == train_labels.shape[0]
			assert train_inputs.shape[1] == train_labels.shape[1]
			assert train_inputs.shape[0] == train_masks.shape[0]
			assert train_inputs.shape[1] == train_masks.shape[1]
		
		losses = []
		accs = []
		for epoch_idx in range(train_params['n_epochs_per_set']):
			for b in range(0, train_inputs.shape[0], B):
				train_inputs_batch = train_inputs[b:b+B, :, :]
				train_labels_batch = train_labels[b:b+B, :]
				train_masks_batch = train_masks[b:b+B, :, :]

				# train_go_info_batch = [train_go_info[0][b:b+B], train_go_info[1][b:b+B]]
				train_go_info_batch = None
				
				self.optimizer.zero_grad()

				if self.run_mode in ('timing',): t0 = time.time() # Forward start

				# Note that labels and masks are only used for special types of gradient calculations
				output, hidden, db = self.iterate_sequence_batch(
					train_inputs_batch, batch_labels=train_labels_batch, batch_masks=train_masks_batch, run_mode=self.run_mode
				)
				
				loss, loss_components, error_term = self.compute_loss(output, train_labels_batch, train_masks_batch, hidden=hidden)
				acc = self.compute_acc(output, train_labels_batch, train_masks_batch) 
				losses.append(loss.cpu().detach().numpy())
				accs.append(acc.cpu().detach().numpy())

				if self.run_mode in ('timing',): 
					self.hist['forward_times'].append(time.time() - t0) # Forward end
					t0 = time.time() # Backward start
				
				if self.gradient_type in ('backprop', 'full_debug',): # Standard supervised training procedures to compute gradient
					loss.backward()
					if self.gradient_clip is not None:
						torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

				if self.run_mode in ('timing',): self.hist['backward_times'].append(time.time() - t0) # Forward end # Backward end
				
				self.optimizer.step() 

				if self.param_clamping: # Enforces parameter clamping (e.g. from cell types or parameter bounds)
					self.param_clamp()

				self.hist['iter'] += 1 # Note this will always be one more than corresponding seq_idx
				
				if monitor and ((self.hist['iter'] % self.monitor_freq == 0) or # Do a monitor for a set amount of iterations
								(self.end_monitor and seq_idx == train_inputs.shape[1]-1)): # Do one monitor at the end of train set
					self._monitor((train_inputs_batch, train_labels_batch, train_masks_batch), train_go_info_batch, valid_go_info,
								  output=output, loss=loss, loss_components=loss_components, 
								  valid_batch=valid_batch)

			# At the end of the epoch, shuffle over batch dimension
			train_inputs, train_labels, train_masks = shuffle_dataset(
				train_inputs, train_labels, train_masks
			)
            
		return db, np.mean(losses), np.mean(accs)

	def iterate_sequence_batch(self, batch_inputs, batch_labels=None, batch_masks=None, run_mode='minimal'):
		"""
		Default iteration through the batch sequence, used in supervised learning setups.

		Computes all outputs and then returns externally to compute loss.
		"""
		
		B = batch_inputs.shape[0]

		batch_output = torch.zeros((B, batch_inputs.shape[1], self.n_output), 
									dtype=torch.float, device=batch_inputs.device)
		batch_hidden = torch.zeros((B, batch_inputs.shape[1], self.n_hidden),
									dtype=torch.float, device=batch_inputs.device)
		self.reset_state(B=B)

		if run_mode in ('track_states',): 
			# For debugging, will keep track of all pyTorch tensors through entire sequence. 
			# This is a placeholder dict, that will get populated after the first network_step
			db_seq = {}
		else:
			db_seq = None

		# Now iterate through the full input sequence, note that network_step of children classes will
		# update the internal state unless told otherwise. 
		for seq_idx in range(batch_inputs.shape[1]):

			step_output, step_activity, db = self.network_step(batch_inputs[:, seq_idx, :], run_mode=run_mode)
			
			batch_output[:, seq_idx, :] = step_output
			batch_hidden[:, seq_idx, :] = step_activity[-1] # assume 1-layer MPN for now

			if run_mode in ('track_states',):
				if seq_idx == 0: # Initializes db_seq based off of first db return
					for key in db:
						if type(db[key]) == torch.Tensor:
							# Batch, seq, rest of dimensions
							db_seq[key] = torch.zeros((batch_inputs.shape[0], batch_inputs.shape[1], *db[key].shape[1:],)).to(db[key].device) 
				for key in db: 
					if type(db[key]) == torch.Tensor:
						db_seq[key][:, seq_idx] = db[key]

		return batch_output, batch_hidden, db_seq


	def compute_reg_term(self):
		
		# Omits certain paramters from regularization
		p_reg = []
		
		for p_name, p in self.named_parameters():
			# print(p_name)
			# if torch.prod(torch.tensor(p.shape)) > 1:
			# 	p_reg.append(p)
			if p_name not in self.reg_omit:
				p_reg.append(p)

		if self.weight_reg == 'L2':
			l2_norm = sum(p.pow(2.0).sum()
								for p in p_reg)
			reg_term = self.reg_lambda * l2_norm/2
		elif self.weight_reg == 'L1':
			l1_norm = sum(p.abs().sum()
								for p in p_reg)
			reg_term = self.reg_lambda * l1_norm
			# print('Batch size: {}, L1 Norm {:0.5f}, Reg Term {:0.5f}'.format(B, l1_norm, reg_term))
		return reg_term

	def compute_loss(self, output, labels, output_mask, hidden):
		"""
		Seperate function here for additional contributions to loss such as regularization terms
		"""

		# Computes regularization term
		if self.weight_reg in ('L1', 'L2'):
			reg_term = self.compute_reg_term()
		else:
			reg_term = torch.tensor(0.0)
   
		activity_reg_term = torch.tensor(0.0, device=output.device)
		if self.activity_reg in ('L2',): 
			activity_reg_term = torch.mean(hidden ** 2) * self.reg_lambda 
			reg_term += activity_reg_term

		if output_mask.dtype == torch.bool:
			# Modify output by the mask (note just uses first output idx to mask)   
			masked_output = output[output_mask[:, :, 0]] # [B, T, out] -> [B*T_mask, out]
			masked_labels = labels[output_mask[:, :, 0]] # [B, T, label_size] -> [B*T_mask, label_size]
		elif output_mask.dtype == torch.float32:
			# Mask by some cost function instead
			masked_output = (output_mask * output).reshape((-1, output.shape[-1])) # [B, T, out] -> [B*T, out]
			masked_labels = (output_mask * labels).reshape((-1, output.shape[-1])) # [B, T, label_size] -> [B*T, label_size]
		else:
			raise ValueError('Mask type {} not recognized.'.format(output_mask.dtype))

		if self.loss_type == 'XE':
			masked_labels = masked_labels.squeeze(-1) # [B*T_mask, 1] -> [B*T_mask]

		output_label_term = self.loss_fn(masked_output, masked_labels)

		loss_components = (output_label_term.item(), reg_term.item(),)

		with torch.no_grad():
			error_term = None
			if self.gradient_type in ('3_factor_hebbian', '3_factor_hebbian_trunc',):
				error_term = masked_labels - masked_output # Under predicting yields positive term (weights should increase)
				raise NotImplementedError('May want to make sure this is similar to the other rules below')
			elif self.gradient_type in ('full_debug', 'local_fo',):
				error_term = output_mask.reshape((-1, output.shape[-1])) * (masked_output - masked_labels) # Two factors of mask because derivative

		return self.loss_fn(masked_output, masked_labels) + reg_term, loss_components, error_term

	def compute_acc(self, output, labels, output_mask, round_type='prefs', verbose=False):
		"""
		output shape: (batches, seq_len, output_size)
		labels shape: (batches, seq_len)
		output_mask shape: (batches, seq_len, output_size)
		"""			
		if verbose: 
			fig, ax = plt.subplots(1,5,figsize=(10*5,4))
			for i in range(5): 
				sns.heatmap(output_mask[i].cpu().detach().numpy(), ax=ax[i], cbar=True)
			fig.savefig("output_mask.png")

		output_mask_alter = torch.full(output_mask.shape, float('nan'), device=output_mask.device) # batches * seq_len * output_size

		for batch_iter in range(output_mask.shape[0]-1):
			# identify chunks of zeros 
			# 1st: first 100ms; 2nd: ideally fix_offs and check_ons; 3rd: end of trail (paddling)
			ll = helper.find_zero_chunks(output_mask[batch_iter, :, :].clone().cpu().detach().numpy())

			if len(ll) < 3: # for delay task...
				ll.append([output_mask.shape[1], output_mask.shape[1]])
			# only focusing on the alignment during response (go) period; always placed at the final
			response_start, response_end = ll[-2][1] + 1, ll[-1][0]

			cut_proportion = 1/8 
			response_duration = response_end - response_start
			response_end = response_end
			response_start = response_start + int(response_duration * cut_proportion)
   
			output_part = output_mask[batch_iter, response_start:response_end, :]
			# originally the mask is not binary, but scaled based on different period to let the cost function work properly
			# that makes sense since we do not want to over-emphasize on periods e.g. fixon and delay
			# when calculating MSE loss
			output_part = (output_part > 0).int()

			output_mask_alter[batch_iter, response_start:response_end, :] = output_part

		assert self.loss_type in ('XE', 'MSE',)

		if self.loss_type in ('XE',):
			# Modify output by the mask (note just uses first output idx to mask)   
			masked_output = output[output_mask[:, :, 0]] # [B, T, out] -> [B*T_mask, out]
			masked_labels = labels[output_mask[:, :, 0]].squeeze(-1) # [B, T, 1] -> [B*T_mask, 1] -> [B*T_mask]
			return (torch.argmax(masked_output, dim=-1) == masked_labels).float().mean() 

		elif self.loss_type in ('MSE',):
			if output_mask.dtype == torch.bool:
				# Modify output by the mask (note just uses first output idx to mask)   
				masked_output = output[output_mask[:, :, 0]] # [B, T, 1] -> [B*T_mask, 1]
				masked_labels = labels[output_mask[:, :, 0]] # [B, T, 1] -> [B*T_mask, 1]
			elif output_mask.dtype == torch.float32:# Mask by some cost function instead
				masked_output = (output_mask_alter * output).reshape((-1, output.shape[-1])) # [B, T, out] -> [B*T, out]
				masked_labels = (output_mask_alter * labels).reshape((-1, output.shape[-1])) # [B, T, label_size] -> [B*T, label_size]
			else:
				raise ValueError('Mask type {} not recognized.'.format(output_mask.dtype))

			if round_type in ('prefs',) and self.prefs is not None: # Round both the labels and the ouputs to self.prefs
				mo = masked_output[~torch.all(torch.isnan(masked_output), dim=1)]
				ml = masked_labels[~torch.all(torch.isnan(masked_labels), dim=1)]
    
				mo_angle = torch.remainder(
					torch.atan2(mo[:, 1], mo[:, 2]),            
					2 * math.pi                              
				).reshape(-1, 1)
    
				ml_angle = torch.remainder(
					torch.atan2(ml[:, 1], ml[:, 2]),            
					2 * math.pi                              
				).reshape(-1, 1)
				
				append_pref = torch.tensor([2 * math.pi], device=self.prefs.device)
				temp_prefs = torch.cat((self.prefs, append_pref), dim=0)

				masked_output_round = round_to_values_torch(mo_angle, temp_prefs) 
				masked_labels_round = round_to_values_torch(ml_angle, temp_prefs)
    
				def clean(x: torch.Tensor, tol: float = 1e-6):
					"""Replace entries equal (within tol) to 2π with 0, in-place."""
					two_pi = torch.tensor(2 * math.pi, device=x.device, dtype=x.dtype)
					x.masked_fill_(torch.isclose(x, two_pi, atol=tol), 0.)
					return x   
     
				masked_output_round = clean(masked_output_round)
				masked_labels_round = clean(masked_labels_round)

			elif round_type in ('int',):
				masked_output_round = torch.round(masked_output)
				masked_labels_round = masked_labels
			else:
				raise ValueError(f'Round type {round_type} not recognized or invalid.')

			return (masked_output_round == masked_labels_round).float().mean() 

	def compute_gradients(self):
		raise NotImplementedError('Should be implemented in children.')

	@torch.no_grad()
	def param_clamp(self):

		for p_name, p in self.named_parameters():
			
			if hasattr(self, p_name + '_bounds'): # If bounds exist, enforce them
				bounds = getattr(self, p_name + '_bounds')
				p.data.clamp_(bounds[0], bounds[1])

	@torch.no_grad()
	def build_param_zero_bounds(self):
		
		MAX_VAL = 1e8 # Magnitude bound for parameters

		if self.verbose: print('Maintaining parameter sparsity:')
		for p_name, p in self.named_parameters():

			if not p.requires_grad:
				continue

			if self.verbose:
				print(' {} nonzero: {:.3f}'.format(
					p_name, torch.mean(torch.where(p == 0., torch.zeros_like(p), torch.ones_like(p)))
				))

			upper = torch.where(p == 0., torch.zeros_like(p), MAX_VAL * torch.ones_like(p))
			lower = -1 * torch.clone(upper)

			self.register_buffer(f'{p_name}_bounds', torch.cat((
				lower.unsqueeze(0), upper.unsqueeze(0)
			), dim=0))

	@torch.no_grad()
	def _monitor_init(self, train_params, train_data, train_trails, valid_batch=None, valid_trails=None):

		if self.hist is None: # Initialize self.hist if needed (can be initialized by children or previous training)
			self.hist = {
				'iter': 0
			}
		if 'iters_monitor' not in self.hist: # This allows self.hist to be initialized in children classes first, and then all default keys are added here  
			self.hist['iters_monitor'] =  []
			self.hist['monitor_thresh'] =  [0,] # used to calculate rolling average loss/accuracy
			if self.train_type in ('supervised',):
				self.hist['epoch'] = 0
				self.hist['train_loss'] =  [] 
				self.hist['train_loss_output_label'] =  [] 
				self.hist['train_loss_reg_term'] =  [] 
				self.hist['train_acc'] =  []
				# Validation monitors
				self.hist['valid_output'] = []
				self.hist['valid_loss'] = []
				self.hist['valid_loss_output_label'] = []
				self.hist['valid_loss_reg_term'] = []
				self.hist['valid_acc']  = []
				self.hist['avg_valid_loss'] = []
				self.hist['avg_valid_acc']  = []
			if self.train_type in ('rl', 'rl_test',):
				self.hist['running_reward'] = 0. 
				self.hist['running_return'] = 0. 
			# Timing (only used for debugging)
			self.hist['forward_times'] = []
			self.hist['backward_times'] = []
		else: 
			if self.verbose:
				print('Network already partially trained. Last iter {}'.format(self.hist['iter']))    

			# Will be used in _monitor to calculate the rolling loss relative to new iter value
			self.hist['monitor_thresh'].append(len(self.hist['iters_monitor']))

		if self.train_type in ('supervised',): # For supervised, gets valid loss and accuracies
			# For the intial monitor, just run the first batch in the sequence
			train_inputs, train_labels, train_masks = train_data

			train_batch = (
				train_inputs[:train_params['batch_size'], :, :], 
				train_labels[:train_params['batch_size'], :], 
				train_masks[:train_params['batch_size'], :, :], 
			)
			
			self._monitor(train_batch, train_trails, valid_trails, valid_batch=valid_batch)

	@torch.no_grad()
	def _monitor(self, train_batch, train_go_info_batch, valid_go_info, output=None, loss=None, loss_components=None, valid_batch=None):  		
		train_inputs_batch, train_labels_batch, train_masks_batch = train_batch # (seq_lens assumed not to matter since these are only used for init monitor)
 
		# Stores iterations where monitor was called
		self.hist['iters_monitor'].append(self.hist['iter'])

		### Train Set ###
		if self.train_type in ('supervised',): # For supervised, gets valid loss and accuracies

			if output is None: # If output is not passed (e.g. in _monitor_init), just does the first example in the sequence
				assert train_inputs_batch.shape[0] == train_labels_batch.shape[0]	
				output, hidden, _ = self.iterate_sequence_batch(train_inputs_batch)
				loss, loss_components, _ = self.compute_loss(output, train_labels_batch, train_masks_batch, hidden=hidden)                
			if loss is None or loss_components is None:
				raise NotImplementedError('Cannot have output without a loss computed.')

			self.hist['train_loss'].append(loss.item()) 
			self.hist['train_loss_output_label'].append(loss_components[0]) 
			self.hist['train_loss_reg_term'].append(loss_components[1])       
			
			# Note this assumes all parameters have the same learning rate so 
			lr = self.optimizer.param_groups[0]['lr']

			moitor_str = 'Iter: {}, LR: {:.3e} - train_loss:{:.3e}'.format(self.hist['iter'], lr, loss)

			if self.loss_type in ('XE', 'MSE',): # Accuracy computations if relevant
				acc = self.compute_acc(output, train_labels_batch, train_masks_batch)
				self.hist['train_acc'].append(acc.item())
				if self.loss_type in ('XE',):
					moitor_str += ', train_acc:{:.3f}'.format(acc)
				else:
					moitor_str += ', rounded train_acc:{:.3f}'.format(acc)
			else:
				self.hist['train_acc'].append(torch.tensor(float('nan')))                

		### Validation Set ###
		# Assumes validation set is already in batches
		valid_inputs_batch, valid_labels_batch, valid_masks_batch = valid_batch if valid_batch else (None, None, None)

		if valid_batch is not None:
			
			valid_output, valid_hidden, _ = self.iterate_sequence_batch(valid_inputs_batch)

			if self.monitor_valid_out: # Saves validation output if needed
				self.hist['valid_output'].append(valid_output.cpu().numpy())
			else:
				self.hist['valid_output'].append(None)
			
			if self.train_type in ('supervised',): # For supervised, gets valid loss and accuracies
				valid_loss, valid_loss_components, _ = self.compute_loss(valid_output, valid_labels_batch, valid_masks_batch, hidden=valid_hidden)  
				
				self.hist['valid_loss'].append(valid_loss.item())
				self.hist['valid_loss_output_label'].append(valid_loss_components[0]) 
				self.hist['valid_loss_reg_term'].append(valid_loss_components[1])                 

				# Rolling average values, with adjustable window to account for early training times
				N_AVG_WINDOW = 10
				avg_window = min((N_AVG_WINDOW, len(self.hist['iters_monitor'])-self.hist['monitor_thresh'][-1]))

				self.hist['avg_valid_loss'].append(np.mean(self.hist['valid_loss'][-avg_window:]))

				moitor_str += ', valid_loss:{:.3e}'.format(valid_loss)

				if self.loss_type in ('XE', 'MSE',): # Accuracy computations if relevant
					valid_acc = self.compute_acc(valid_output, valid_labels_batch, valid_masks_batch, verbose=False) 
					self.hist['valid_acc'].append(valid_acc.item())
					self.hist['avg_valid_acc'].append(np.mean(self.hist['valid_acc'][-avg_window:]))
					if self.loss_type in ('XE',):
						moitor_str += ', valid_acc:{:.3f}'.format(valid_acc)
					else:
						moitor_str += ', rounded valid_acc:{:.3f}'.format(valid_acc)
				else:
					self.hist['valid_acc'].append(torch.tensor(float('nan')))   
					self.hist['avg_valid_acc'].append(torch.tensor(float('nan')))  

		else: # Still update these quantities to keep self.hist in sync
			self.hist['valid_loss'].append(torch.tensor(float('nan')))                
			self.hist['valid_acc'].append(torch.tensor(float('nan')))
			self.hist['avg_valid_loss'].append(torch.tensor(float('nan')))
			self.hist['avg_valid_acc'].append(torch.tensor(float('nan')))   
		
		if self.verbose:
			print(moitor_str)

	def save(self, path):
		"""
		Just save the state dictionary.
		"""   
		
		current_device = next(self.buffers()).device

		self.to('cpu') 
		 
		state = self.state_dict()
		state.update({'hist':self.hist}) # Put self.hist in state_dict, removed later
		
		torch.save(state, path)

		# Put net back on device it started on
		self.to(current_device) 

	
	def load(self, path):

		state = torch.load(path)
		self.hist = state.pop('hist')
		
		self.load_state_dict(state, strict=False)