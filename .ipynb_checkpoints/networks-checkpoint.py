import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np

import copy

from net_helpers import BaseNetwork
from net_helpers import rand_weight_init, get_activation_function

import time

##############################################################################################
##############################################################################################
##############################################################################################

class VanillaRNN(BaseNetwork):
	

	def __init__(self, net_params, verbose=False):
		super().__init__(net_params, verbose=verbose)  

		init_string = 'Vanilla RNN:\n'

		# Default code that should not change based on cell types
		init_string = self.standard_init(net_params, init_string, verbose=verbose)

		self.cell_types = net_params.get('cell_types', None)
		
		# Assigns cell types if they are being used
		if self.cell_types is not None:
			raise NotImplementedError('Cell types not yet implemented for VanillaRNN')
		else:
			self.input_cell_types = None
			self.hidden_cell_types = None


		### Weight/bias initialization ###
		# Input weights
		if self.input_layer in ('trainable', 'frozen',):
			self.W_input_init = net_params.get('W_input_init', 'xavier')
			self.parameter_or_buffer('W_input', torch.tensor(
				rand_weight_init(self.n_input, self.n_hidden, init_type=self.W_input_init,
								 cell_types=self.input_cell_types),
				dtype=torch.float)
			)

		# Recurrent weights
		self.W_rec_init = net_params.get('W_rec_init', 'xavier')
		self.W_rec_sparsity = net_params.get('W_rec_sparsity', None)

		self.parameter_or_buffer('W_rec', torch.tensor(
			rand_weight_init(self.n_hidden, self.n_hidden, init_type=self.W_rec_init,
							 cell_types=self.hidden_cell_types, sparsity=self.W_rec_sparsity),
			dtype=torch.float)
		)

		# Output weights
		if self.output_layer in ('trainable', 'frozen',):
			self.W_output_init = net_params.get('W_output_init', 'xavier')
			self.parameter_or_buffer('W_output', torch.tensor(
				rand_weight_init(self.n_hidden, self.n_output, init_type=self.W_output_init,
								 cell_types=self.hidden_cell_types),
				dtype=torch.float)
			)

		# Bias terms
		if self.b_hidden_active:
			self.b_hidden_init = net_params.get('b_hidden_init', 'gaussian')
		else:
			self.b_hidden_init = 'zeros'
		self.parameter_or_buffer('b_hidden', torch.tensor(
			rand_weight_init(self.n_hidden, init_type=self.b_hidden_init),
			dtype=torch.float)
		)

		if self.output_layer in ('trainable', 'frozen',):
			if self.b_output_active: 
				self.b_output_init = 'gaussian'
			else:
				self.b_output_init = 'zeros'
			self.parameter_or_buffer('b_output', torch.tensor(
				rand_weight_init(self.n_output, init_type=self.b_output_init),
				dtype=torch.float)
			)

		self.keep_param_zeros = net_params.get('keep_param_zeros', False)

		if self.keep_param_zeros:
			if self.cell_types is not None:
				raise NotImplementedError('Multiple types of bounds not yet implemented.')
			
			self.param_clamping = True

			self.build_param_zero_bounds()

		if verbose:
			print(init_string)

	def standard_init(self, net_params, init_string, verbose=False):
		"""
		This is the portion of the initializaiton that should not change whether or not a copy of the network
		is being created or not. This is so that we don't need to repeat all this code in the VanillaRNN with 
		cell types.
		"""

		if 'n_neurons' in net_params:
			self.n_input = net_params['n_neurons'][0]
			self.n_hidden = net_params['n_neurons'][1]
			self.n_output = net_params['n_neurons'][2]
		else:
			self.n_hidden = net_params['n_hidden']
			self.n_input = net_params['n_input']
			self.n_output = net_params['n_output']
 
		init_string += '  n_neurons - input: {}, hidden: {}, output: {}\n'.format(
			self.n_input, self.n_hidden, self.n_output
		)

		# Get list of variable that should be parameters of the network (for later matching to experiment)
		self.params = [] # Biases can be added below if used.

		### Whether or not each layer is present and trainable ###
		self.input_layer = net_params.get('input_layer', 'trainable')
		if self.input_layer in ('trainable',):
			self.params.append('W_input')
		elif self.input_layer in (None,):
			self.n_input = self.n_hidden
		elif self.input_layer not in ('frozen',):
			raise ValueError('Input layer option {} not recognized.'.format(self.input_layer))
		init_string += '  input layer: {} //'.format(self.input_layer)

		self.rec_layer = net_params.get('rec_layer', 'trainable')
		if self.rec_layer in ('trainable',):
			self.params.append('W_rec')
		elif self.rec_layer in (None,):
			raise NotImplementedError('No recurrent layer not yet implemented.')
		elif self.rec_layer not in ('frozen',):
			raise ValueError('Recurrent layer option {} not recognized.'.format(self.rec_layer))
		init_string += ' rec layer: {} //'.format(self.rec_layer)

		self.output_layer = net_params.get('output_layer', 'trainable')
		if self.output_layer in ('trainable',):
			self.params.append('W_output')
		elif self.output_layer in (None,):
			self.n_output = self.n_hidden
		elif self.output_layer not in ('frozen',):
			raise ValueError('Output layer option {} not recognized.'.format(self.output_layer))
		init_string += ' output layer: {}\n'.format(self.output_layer)

		# Numpy equivalents only used for debugging purposes
		self.act = net_params.get('activation', 'linear')
		self.act_fn, self.act_fn_np, self.act_fn_p = get_activation_function(self.act)
		
		init_string += '  Act: {}\n'.format(
			self.act,
		)

		self.b_hidden_active = net_params.get('hidden_bias', True)
		self.b_output_active = net_params.get('output_bias', True)

		if self.b_hidden_active:
			self.params.append('b_hidden')
		if self.b_output_active: 
			self.params.append('b_output')

		# Leaky setup parameters
		self.leaky = net_params.get('leaky', False)
		if self.leaky:
			if 'leaky_timescale' in net_params:
				self.leaky_timescale = net_params.get('leaky_timescale', 125) # Rate of leakyness, if dt = 50 ms then alpha = 0.6 by default
				self.alpha = 1 - self.dt / self.leaky_timescale
			else:
				self.alpha = net_params.get('alpha', 0.2)
				self.leaky_timescale = self.dt / (1 - self.alpha)
			init_string += '  Leaky updates, timescale {:.0f} ms (old: {:.1f} new: {:.1f})'.format(self.leaky_timescale, self.alpha, 1-self.alpha)
		else:
			self.alpha = 0.0

		# By default, these are set to none but are overridden in the initialization if cell types are used
		self.ei_balance = None
		self.input_cell_types = None
		self.hidden_cell_types = None

		return init_string

	def reset_state(self, B=1):
		"""
		Resets internal states of the network. 
		"""

		# Gets device from a buffer that should be on the correct device of the whole net
		device = self.W_rec.device

		self.h0 = torch.zeros((B, self.n_hidden,), device=device)

		# Initialize states to desired B
		self.hidden = torch.clone(self.h0)
  
	def forward(self, inputs, run_mode='minimal', verbose=False):

		if type(inputs) == torch.Tensor: # Current input
			x = torch.clone(inputs)
			x_hidden = None
		elif type(inputs) == tuple:
			assert len(inputs) == 2
			x = torch.clone(inputs[0])
			x_hidden = torch.clone(inputs[1])
		else:
			raise NotImplementedError('Input type not recognized.')

		if self.input_layer in ('trainable', 'frozen',):
			h_input = torch.einsum('iI, BI -> Bi', self.W_input, x)
		else:
			h_input = x

		h_rec = torch.einsum('iI, BI -> Bi', self.W_rec, self.hidden)
		
		hidden_pre = h_input + h_rec + self.b_hidden.unsqueeze(0)

		if x_hidden is not None: # Direct input injection to hidden layer (usually just noise)
			hidden_pre = hidden_pre + x_hidden

		if self.leaky:
			hidden = self.alpha * self.hidden + (1. - self.alpha) * self.act_fn(hidden_pre)
		else: # Equivalent to alpha = 1.0
			hidden = self.act_fn(hidden_pre)

		if self.output_layer in ('trainable', 'frozen',):
			output_hidden = torch.einsum('iI, BI -> Bi', self.W_output, hidden)
			output = output_hidden + self.b_output.unsqueeze(0)
		else:
			output = hidden

		if run_mode in ('track_states',):
			db = {
				'hidden_pre': hidden_pre.detach(),
				'hidden': hidden.detach(),
			}
		else:
			db = None

		return output, hidden, db

	def network_step(self, current_input, run_mode='minimal', monitor_set=False, verbose=False):
		"""
		Performs a single batch pass forward for the network. This mostly consists of a forward pass and 
		the associated updates internal states.
		
		This should not be passed a full sequence of data, only data from a given time point

		current_input: Input into the RNN at the current time step. Currently two forms implemented:
		- type == torch.Tensor: Raw input to be passed through input layer
		- tpye == tuple: Tuple consisting of (raw_input, noise_injections), former goes through input, latter goes directly into hidden activity
		"""

		output, current_hidden, db = self.forward(current_input, run_mode=run_mode, verbose=verbose)

		# Sets internal states using forward
		self.hidden = current_hidden
		# if run_mode in ('minimal',):
		# 	db['hidden'] = current_hidden.detach()

		return output, db
		
	def update_parameters(self, net_params):
		"""
		From the list of params, either turns buffers into parameters or vice versa
		"""

		# Updates self.params
		self.params = net_params.get('params', ())

		self.parameter_or_buffer('W_input')
		self.parameter_or_buffer('W_rec')
		self.parameter_or_buffer('W_output')

		self.parameter_or_buffer('b_hidden')
		self.parameter_or_buffer('b_output')

	def state_detach(self):
		""" Detach internal states for truncated BPTT """
		self.hidden.detach_()

	def get_device(self):
		""" Return current device the network is on """
		return self.W_rec.device

	# @torch.no_grad()
	# def param_clamp(self):
	# 	"""
	# 	Clamps all weights (only called if cell types are enforced)

	# 	"""

	# 	if self.input_cell_types is not None and self.input_layer in ('trainable', 'frozen',):
	# 		self.W_input.data.clamp_(self.W_input_bounds[0], self.W_input_bounds[1])
	# 	if self.hidden_cell_types is not None:
	# 		if self.cell_types not in ('E_in_and_out',): # Has hidden cell types but doesn't enforce them in recurrent layer
	# 			self.W_rec.data.clamp_(self.W_rec_bounds[0], self.W_rec_bounds[1])
	# 		self.W_output.data.clamp_(self.W_output_bounds[0], self.W_output_bounds[1])

	@torch.no_grad()
	def _monitor_init(self, train_params, train_data, valid_batch=None):

		# Additional quantities to track during training, note initializes these first so that _monitor call
		# inside super()._monitor_init can append additional quantities.
		if self.hist is None: 
			self.hist = {
				'iter': 0,
			}

			if self.gradient_type in ('full_debug',): # Additional quantities to track for full gradient debugging
				self.hist['local_term_mag'] = []
				self.hist['nonlocal_term_mag'] = []
				self.hist['leak_mag'] = []

				self.hist['local_term_sparsity'] = []
				self.hist['nonlocal_term_sparsity'] = []
				self.hist['leak_sparsity'] = []

				self.hist['positive_grad_ratio'] = []
				self.hist['local_deviation_mag'] = []

		super()._monitor_init(train_params, train_data, valid_batch=valid_batch)


	@torch.no_grad()
	def _monitor(self, train_batch, train_type='supervised', output=None, loss=None, loss_components=None, 
				 acc=None, valid_batch=None): 

		super()._monitor(train_batch, output=output, loss=loss, loss_components=loss_components,
						 valid_batch=valid_batch)


class GRU(VanillaRNN):
	def __init__(self, net_params, verbose=False):
		super().__init__(net_params, verbose=verbose)  
		
		init_string = 'GRU:\n'

		# Extend vanilla parameters (biases may be turned on below)
		self.params.extend(['Wz_input', 'Wz_rec', 'Wr_input', 'Wr_rec',])


		# Z gate weights (assumes same initializations types as Vanilla equivalents)
		self.parameter_or_buffer('Wz_input', torch.tensor(
			rand_weight_init(self.n_input, self.n_hidden, init_type=self.W_input_init),
			dtype=torch.float)
		)
		self.parameter_or_buffer('Wz_rec', torch.tensor(
			rand_weight_init(self.n_hidden, self.n_hidden, init_type=self.W_rec_init),
			dtype=torch.float)
		)
		if self.b_hidden_active:
			self.params.append('bz_hidden')
			self.b_hidden_init = 'gaussian'
		else:
			self.b_hidden_init = 'zeros'
		self.parameter_or_buffer('bz_hidden', torch.tensor(
			rand_weight_init(self.n_hidden, init_type=self.b_hidden_init),
			dtype=torch.float)
		)


		# Reset gate weights (assumes same initializations types as Vanilla equivalents)
		self.parameter_or_buffer('Wr_input', torch.tensor(
			rand_weight_init(self.n_input, self.n_hidden, init_type=self.W_input_init),
			dtype=torch.float)
		)
		self.parameter_or_buffer('Wr_rec', torch.tensor(
			rand_weight_init(self.n_hidden, self.n_hidden, init_type=self.W_rec_init),
			dtype=torch.float)
		)
		if self.b_hidden_active:
			self.params.append('br_hidden')
			self.b_hidden_init = 'gaussian'
		else:
			self.b_hidden_init = 'zeros'
		self.parameter_or_buffer('br_hidden', torch.tensor(
			rand_weight_init(self.n_hidden, init_type=self.b_hidden_init),
			dtype=torch.float)
		)
		
		self.reset_state()

		if verbose:
			print(init_string)

	
	def forward(self, inputs, run_mode='minimal', verbose=False):
		
		x = torch.clone(inputs)

		# Update gate
		u_input = torch.einsum('iI, BI -> Bi', self.Wz_input, x)
		u_rec = torch.einsum('iI, BI -> Bi', self.Wz_rec, self.hidden)

		u_pre = u_input + u_rec + self.bz_hidden.unsqueeze(0)
		update = torch.sigmoid(u_pre)

		# Reset gate
		r_input = torch.einsum('iI, BI -> Bi', self.Wr_input, x)
		r_rec = torch.einsum('iI, BI -> Bi', self.Wr_rec, self.hidden)

		r_pre = r_input + r_rec + self.br_hidden.unsqueeze(0)
		reset = torch.sigmoid(r_pre)

		# Main computation
		h_input = torch.einsum('iI, BI -> Bi', self.W_input, x)
		h_rec = torch.einsum('iI, BI -> Bi', self.W_rec, self.hidden * reset)

		hidden_pre = h_input + h_rec + self.b_hidden.unsqueeze(0)
		hidden_new = self.act_fn(hidden_pre)
		
		hidden = (torch.ones_like(update) - update) * hidden_new + update * self.hidden
		
		# Output layer
		output_hidden = torch.einsum('iI, BI -> Bi', self.W_output, hidden)
		
		output = output_hidden + self.b_output.unsqueeze(0)

		if run_mode in ('track_states',):
			db = {
				'hidden_pre': hidden_pre,
				'hidden': hidden
			}
		else:
			db = None

		return output, hidden, db

	def update_parameters(self, net_params):
		"""
		From the list of params, either turns buffers into parameters or vice versa
		"""

		super().update_parameters(net_params) 

		self.parameter_or_buffer('Wz_input')
		self.parameter_or_buffer('Wz_rec')
		self.parameter_or_buffer('bz_hidden')

		self.parameter_or_buffer('Wr_input')
		self.parameter_or_buffer('Wr_rec')
		self.parameter_or_buffer('br_hidden')