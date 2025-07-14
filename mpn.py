import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.nn.init import orthogonal_

import math

from net_helpers import BaseNetwork, BaseNetworkFunctions
from net_helpers import rand_weight_init, get_activation_function

import numpy as np

import time 

class MultiPlasticLayer(BaseNetworkFunctions):
    """
    Fully-connected layer with multi-plasticity.

    This functions very similarly to a fully connected PyTorch layer. However,
    in addition to the implementation of the layer in a regular forward pass,
    the modulations need to be correctly kept track of through the "reset_state"
    and "update_M_matrix" functions. The latter needs to be called after every
    forward pass where the modulations should be updated.
    """
    def __init__(self, ml_params, output_matrix, verbose=True):
        super().__init__()

        init_string=''

        self.verbose=verbose
        # Name appended to various parameters to disti
        self.mp_layer_name = ml_params.get('mpl_name', '')

        self.n_input = ml_params['n_input']
        self.n_output = ml_params['n_output']

        init_string += '  MP Layer{} parameters:\n'.format(self.mp_layer_name)
        init_string += '    n_neurons - input: {}, output: {}'.format(
            self.n_input, self.n_output
        )

        # Determines whether or not layer weights are trainable parameters
        self.layer_bias = ml_params.get('bias', True)
        self.freeze_layer = ml_params.get('freeze_layer', False)
        if self.freeze_layer:
            init_string += '    W: Frozen // '
            if self.layer_bias:
                init_string += 'b: Frozen //'
        else:
            self.params = ['W',] # This is a local params list that can be merged with full network params if needed
            if self.layer_bias:
                self.params.append('b')

        ### Weight/bias initialization ###
        # Input weights
        self.W_init = ml_params.get('W_init', 'xavier')
        W_freeze = ml_params.get('W_freeze', False)

        # Initialize the weight tensor once.
        W_tensor = torch.tensor(
            rand_weight_init(self.n_input, self.n_output, init_type=self.W_init, cell_types=None),
            dtype=torch.float
        )

        if W_freeze:
            print("MPN Layer W Frozen")
            self.register_buffer('W', W_tensor)
        else:
            self.parameter_or_buffer('W', W_tensor)


        # Bias term
        if self.layer_bias:
            self.b_init = 'gaussian'
        else:
            self.b_init = 'zeros'
        self.parameter_or_buffer('b', torch.tensor(
            rand_weight_init(self.n_output, init_type=self.b_init),
            dtype=torch.float)
        )

        ###### M matrix-related specs ########
        init_string += '\n    M matrix parameters:'

        self.mp_type = ml_params.get('mp_type', 'mult')
        # Controls the update equation of the M matrix (calculation of \Delta M)
        self.m_update_type = ml_params.get('m_update_type', 'hebb_assoc')
        # Activation function to pass M through after update (can enforce bounds)
        self.m_act = ml_params.get('m_activation', 'linear')
        self.m_act_fn, self.m_act_fn_np, self.m_act_fn_p = get_activation_function(self.m_act)

        # Initial modulation values
        self.register_buffer('M_init', torch.zeros((self.n_output, self.n_input,), dtype=torch.float))

        # Controls maximum and minimum values of modulations so weights don't change signs
        self.modulation_bounds = ml_params.get('modulation_bounds', True)
        if self.modulation_bounds:
            self.M_bound_vals = ml_params.get('m_bounds', (-1.0, 1.0,)) # (min, max)

            M_bounds, init_string = self.build_M_bounds(init_string=init_string)
            self.register_buffer('M_bounds', M_bounds)

            # These bounds will need to be continually updated if W is variable and M is additive, which is not yet implemented
            if self.mp_type == 'add' and 'W' in self.params:
                raise NotImplementedError('Need to continuously update bounds in this case.')

        init_string += '      type: {} // Update - type: {} // Act fn: {}'.format(
            self.mp_type, self.m_update_type, self.m_act
        )

        ### Eta dependencies ###
        self.eta_train = ml_params.get('eta_train', True)
        eta_train_str = 'fixed'
        if self.eta_train:
            self.params.append('eta')
            eta_train_str = 'train'
        self.eta_type = ml_params.get('eta_type', 'scalar')
        self.eta_init = ml_params.get('eta_init', 'eta_clamp')

        self.eta_clamp = ml_params.get('eta_clamp', 1.00)

        self.parameter_or_buffer('eta', torch.tensor(
            self.init_M_parameter(param_type=self.eta_type, init_type=self.eta_init),
        dtype=torch.float))

        if self.eta_type in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += '\n      Eta: {} ({}) // '.format(self.eta_type, eta_train_str)
        else:
            raise ValueError('eta_type: {} not recognized'.format(self.eta_type))

        ### Lambda dependencies ###
        self.lam_train = ml_params.get('lam_train', True)
        lam_train_str = 'fixed'
        if self.lam_train:
            self.params.append('lam')
            lam_train_str = 'train'
        self.lam_type = ml_params.get('lam_type', 'scalar')
        self.lam_init = ml_params.get('lam_init', 'lam_clamp')
        # Maximum lambda value/corresponding decay time constant (both always computed)
        if 'm_time_scale' in ml_params:
            self.m_time_scale = ml_params.get('m_time_scale')
            self.lam_clamp = 1. - ml_params['dt'] / self.m_time_scale
        else:
            self.lam_clamp = ml_params.get('lam_clamp', 0.95)
            self.m_time_scale = ml_params['dt'] / (1. - self.lam_clamp)

        self.parameter_or_buffer('lam', torch.tensor(
            np.abs(self.init_M_parameter(param_type=self.lam_type, init_type=self.lam_init)), # Always positive
        dtype=torch.float))

        if self.lam_type in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += 'Lambda: {} ({}) // Lambda_max: {:.2f} (tau: {:.1e})'.format(
                self.lam_type, lam_train_str, self.lam_clamp, self.m_time_scale
            )
        else:
            raise ValueError('lam_type: {} not recognized'.format(self.lam_type))

        if self.verbose: # Full summary of mp_layer parameters
            print(init_string)

    def reset_state(self, B=1):
        """
        Resets/initializes modulations values
        """

        self.M = torch.ones(B, *self.W.shape, device=self.W.device) #shape: (B, n_input, n_output)
        self.M = self.M * self.M_init.unsqueeze(0) # (B, n_input, n_output) x (1, n_input, n_output)

        self.M_pre = torch.zeros_like(self.M)

    @torch.no_grad()
    def param_clamp(self):
        """ Enforce lambda bounds. Doesn't track gradients, since this is always called after weight updates. """
        self.lam.data.clamp_(0., self.lam_clamp)

    def init_M_parameter(self, param_type='scalar', init_type='gaussian'):
        """
        Initialize different forms of the various M parameters (e.g. eta and lambda).
        Default is just one of each parameter for each layer, but can make them
        post- and/or presynaptic cell dependent.

        Turned into a buffer/parameter externally.
        """

        if type(init_type) == float:
            if param_type == 'scalar': # Just directly set to init_type
                param = init_type
            elif param_type == 'pre_vector': # Serves as mean to distribution
                param = init_type + rand_weight_init(self.n_input, init_type='guassian', weight_norm=1.0)
            elif param_type == 'post_vector':
                param = init_type + rand_weight_init(self.n_output, init_type='guassian', weight_norm=1.0)
            elif param_type == 'matrix':
                param = init_type + rand_weight_init(self.n_input, self.n_output, init_type='guassian', weight_norm=1.0)
        elif type(init_type) == str:
            if param_type == 'scalar':
                if init_type in ('lam_clamp',):
                    param = self.lam_clamp
                elif init_type in ('eta_clamp',):
                    param = self.eta_clamp
                else:
                    param = rand_weight_init(1, init_type=init_type, weight_norm=1.0)
            elif param_type == 'pre_vector':
                if init_type in ('lam_clamp',):
                    param = self.lam_clamp * np.ones((self.n_input,))
                elif init_type in ('eta_clamp',):
                    param = self.eta_clamp * np.ones((self.n_input,))
                else:
                    param = rand_weight_init(self.n_input, init_type=init_type, weight_norm=1.0)
            elif param_type == 'post_vector':
                if init_type in ('lam_clamp',):
                    param = self.lam_clamp * np.ones((self.n_output,))
                elif init_type in ('eta_clamp',):
                    param = self.eta_clamp * np.ones((self.n_output,))
                else:
                    param = rand_weight_init(self.n_output, init_type=init_type, weight_norm=1.0)
            elif param_type == 'matrix':
                if init_type in ('lam_clamp',):
                    param = self.lam_clamp * np.random.rand(self.n_output, self.n_input,)
                elif init_type in ('eta_clamp',):
                    param = self.eta_clamp * np.random.rand(self.n_output, self.n_input,)
                else:
                    param = rand_weight_init(self.n_input, self.n_output, init_type=init_type, weight_norm=1.0)
        else:
            raise ValueError('Init of type {} not recognized: {}'.format(type(init_type), init_type))

        return param

    def build_M_parameter(self, param, param_type='scalar'):
        """
        Returns M parameters (e.g. eta and lambda) of the appropriate dimensions
        given their type

        OUTPUTS:
        param_matrix shape is simply something that can be cast to the shape of
        M without batch dim: (n_output, n_input), so len(param_expanded.shape)
        == 2 always
        """

        if param_type == 'scalar':
            param_expanded = param.unsqueeze(0) # shape (1, 1)
        elif param_type == 'pre_vector':
            param_expanded = param.unsqueeze(0) # shape (1, n_input)
        elif param_type == 'post_vector':
            param_expanded = param.unsqueeze(-1) # shape (n_output, 1)
        elif param_type == 'matrix':
            param_expanded = param # shape (n_output, n_input)

        return param_expanded

    def build_M_bounds(self, init_string=''):
        """
        Controls maximum and minimum values of modulations. Generally used so
        weights don't change signs (since these are often tied to cell type).

        Bounds are in order: (upper_vals, lower_vals)
        """

        W_fixed = self.W.detach()

        if self.mp_type == 'add':
            MAX_ADD = self.M_bound_vals[1] # Default: 1.0, Controls how much a weight can be strengthened, 1.0 means the weight's mag can be doubled, 0.0 means it cant be strengthened
            MIN_ADD = self.M_bound_vals[0] # Default: 0.0, Any value >0 prevents weight from being fully weakened, e.g. 0.2 means the weight can be weakened to at most 20% of its original value

            # Expanation of this expression: (with example values MAX_ADD = 2.0 and MIN_ADD = 0.2)
            #   First line: Upper bounds on Ms
            #       For W_ij > 0: Maximum M value is MAX_ADD * W_ij, so 2 * W_ij > 0, meaning positive weights can be strengthened to 3x their initial value
            #       For W_ij < 0: Maximum M value is -1 * (1 - MIN_ADD) * W_ij = -0.8 * W_ij > 0, since W_ij is negative. Since a positive M_ij would cancel the
            #           negative W_ij, this means that at most W_ij can be weakened to 0.2 x its original value
            #   Second line: Lower bounds for M
            #       For W_ij > 0: Minimum M value is -1 * (1 - MIN_ADD) * W_ij = -0.8 * W_ij < 0, since W_ij is positive, a negative M_ij that saturates this bound
            #           would reduce W_ij to 0.2 x its original value.
            #       For W_ij < 0: Minimum M value is MAX_ADD * W_ij = 2 * W_ij < 0, since W_ij is negative. So can strengthen negative weight to 3x its initial value

            M_bounds = torch.cat((
                (MAX_ADD * W_fixed * (W_fixed > 0) - 1 * (1 - MIN_ADD) * W_fixed * (W_fixed < 0)).unsqueeze(0),
                (MAX_ADD * W_fixed * (W_fixed < 0) - 1 * (1 - MIN_ADD) * W_fixed * (W_fixed > 0)).unsqueeze(0)
            ))

            init_string += '    update bounds - Max add: {}, Min add: {}\n'.format(MAX_ADD, MIN_ADD)
        elif self.mp_type == 'mult':
            max_mult = self.M_bound_vals[1] # Controls how much a weight can be enhanced, 1.0 means the weight's mag can be doubled
            min_mult = self.M_bound_vals[0] # Controls how much a weight can be depressed, -1.0 means it can be fully depressed
            M_bounds = torch.cat((
                max_mult * torch.ones_like(W_fixed).unsqueeze(0),
                min_mult * torch.ones_like(W_fixed).unsqueeze(0)
            ))
            init_string += '    update bounds - Max mult: {}, Min mult: {}\n'.format(max_mult, min_mult)
        else:
            raise ValueError('MP type not recognized in build_M_bounds.')

        return M_bounds, init_string

    def update_M_matrix(self, pre, post, update_mask=None):
        """
        Updates the modulation matrix from one time step to the next.
        Should only be called in the network_step pass once. Directly updates self.M.

        Note that this is NOT automatically called in MP layer's "forward" call, since the
        postsynaptic activity could be dependent on other factors (e.g. other layers).

        M updates can be frozen from the update_mask, if a given bactch idx is
        False (e.g. because it is beyond the end of the sequence)

        pre.shape: (B, n_input)
        post.shape: (B, n_output)
        update_mask: (B,)
        """

        eta = self.build_M_parameter(self.eta, self.eta_type)
        lam = self.build_M_parameter(self.lam, self.lam_type)
        M = self.M

        delta_M = torch.zeros_like(M)

        if update_mask is not None: # Update each batch_idx individually using the update_mask
            for batch_idx in range(M.shape[0]):
                if update_mask[batch_idx]: # Only calculates delta_M if batch is being updated (I think this saves time?)
                    if self.m_update_type in ('hebb_pre',):
                        post = 1 / math.sqrt(post.shape[-1]) * torch.ones_like(post)

                    if self.m_update_type in ('hebb_assoc', 'hebb_pre',):
                        delta_M[batch_idx] = - M[batch_idx] + lam * M[batch_idx] + eta * torch.einsum(
                            'i, I -> iI', post[batch_idx], pre[batch_idx]
                        )
                    elif self.m_update_type in ('oja',):
                        delta_M[batch_idx] = (eta * torch.einsum('i, I -> iI', post[batch_idx], pre[batch_idx]) -
                                              torch.abs(eta) * torch.einsum('i, iI -> iI', post[batch_idx]**2, M[batch_idx]))
        else: # Update all M at once
            if self.m_update_type in ('hebb_pre',):
                post = 1 / math.sqrt(post.shape[-1]) * torch.ones_like(post)

            if self.m_update_type in ('hebb_assoc', 'hebb_pre',):
                delta_M = - M + lam.unsqueeze(0) * M + eta.unsqueeze(0) * torch.einsum(
                    'Bi, BI -> BiI', post, pre
                )

            elif self.m_update_type in ('oja',):
                raise NotImplementedError('Need to update to a batched version.')
                # delta_M[batch_idx] = (eta * torch.einsum('i, I -> iI', post[batch_idx], pre[batch_idx]) -
                #                       torch.abs(eta) * torch.einsum('i, iI -> iI', post[batch_idx]**2, M[batch_idx]))

        self.M_pre = self.M + delta_M
        self.M = self.m_act_fn(self.M_pre)

        # # Masks batches of delta_M
        # delta_M_masked = torch.einsum('B, BiI -> BiI', update_mask, delta_M)

        # Update M matrices, while being sure update holds matrix within bounds
        # (this may error if not self.ei_types, but this is always true in our settings)
        # (note: updates to restristed cell types is built into the eta matrix)
        if self.modulation_bounds:
            self.M = torch.clamp(self.M, min=self.M_bounds[1], max=self.M_bounds[0])

        return delta_M # This is only used for theory matching

    def get_modulated_weights(self, M=None):
        """
        Returns modulated weights, taking into account exactly how M and W are combined.
        Note the batch size of the modulated weights is set by the batch size of M.

        If M is not None, the passed M matrix is used, otherwise just uses stored Ms (this
        is be used for analysis of the network after training)
        """

        W = self.W
        if M is None:
            M = self.M

        # Fixed weights and M matrix, either multiplicative or additive
        if self.mp_type == 'mult':
            modulated_weights = W.unsqueeze(0) + W.unsqueeze(0) * M
        elif self.mp_type == 'add':
            modulated_weights = W.unsqueeze(0) + M

        return modulated_weights

    def forward(self, x, run_mode='minimal'):
        """
        Passes inputs through the modulated weights. Activation are handled
        externally to allow for more complicated architectures.

        Updating of modulations is handled externally

        INPUTS:
        x.shape: [B, n_input]

        INTERNALS:
        b.shape: [n_output]
        modulated_weights.shape: [B, n_output, n_input],

        OUTPUTS:
        pre_act.shape: [B, n_output]

        """

        modulated_weights = self.get_modulated_weights()
        pre_act_no_bias =  torch.einsum('BiI, BI -> Bi', modulated_weights, x)

        pre_act = pre_act_no_bias + self.b.unsqueeze(0)

        if run_mode in ('track_states',):
            db = {
                'pre_act_no_bias': pre_act_no_bias.detach(),
                'M': self.M.detach(),
                'b': self.b.detach().unsqueeze(0), # record bias information as well 
            }
            if self.m_act:
                db['M_pre'] = self.M_pre.detach()
        else:
            db = None

        return pre_act, db

class MultiPlasticNetBase(BaseNetwork):
    """
    Base network for the multiplastic network. Initializes things like the output
    layers and activation functions that are mostly shared across all types of
    MPNs, no matter the connections that lead from input to output.
    """

    def __init__(self, net_params, n_output_pre, output_matrix="", verbose=False):
        # Note that this assumes self.output has already been set in child
        assert hasattr(self, 'n_output')

        super().__init__(net_params, verbose=verbose)

        init_string = 'MultiPlastic Net:\n'

        init_string += '  output neurons: {}\n'.format(
            self.n_output
        )

        # Get list that should be parameters of the network (for later matching to experiment)
        self.params = ['W_output',] # Biases can be added below if used.

        # Numpy equivalents only used for debugging purposes
        self.act = net_params.get('activation', 'linear')
        self.act_fn, self.act_fn_np, self.act_fn_p = get_activation_function(self.act)

        init_string += '  Act: {}\n'.format(
            self.act,
        )

        self.b_output_active = net_params.get('output_bias', True)
        if self.b_output_active:
            self.params.append('b_output')
            self.b_output_init = 'gaussian'
        else:
            self.b_output_init = 'zeros'

        # By default, these are set to none but are overridden in the initialization if cell types are used
        self.ei_balance = None
        self.input_cell_types = None
        self.hidden_cell_types = None

        self.cell_types = net_params.get('cell_types', None)
        if self.cell_types:
            raise NotImplementedError()

        # Output weights
        self.W_output_init = net_params.get('W_output_init', 'xavier')
        self.parameter_or_buffer('W_output', torch.tensor(
            rand_weight_init(n_output_pre, self.n_output, init_type=self.W_output_init,
                             cell_types=self.hidden_cell_types),
            dtype=torch.float)
        )

        # overwrite 
        if output_matrix == "":
            pass
        elif output_matrix == "untrained":
            print("Output Matrix Untrained")
            self.W_output.requires_grad = False
        elif output_matrix == "orthogonal":
            print("Output Matrix Orthogonal and Untrained")
            W_output_init = torch.empty(self.n_output, n_output_pre)  # Create a matrix of size (n_output, n_output_pre)
            orthogonal_(W_output_init)  # In-place orthogonal initialization
            W_output_init = W_output_init.T
            self.parameter_or_buffer('W_output', torch.tensor(W_output_init, dtype=torch.float))
            self.W_output.requires_grad = False
        else:
            raise Exception("Output Matrix not recognized")

        self.parameter_or_buffer('b_output', torch.tensor(
            rand_weight_init(self.n_output, init_type=self.b_output_init),
            dtype=torch.float)
        )

        if verbose: # Full summary of readout parameters (MP layer prints out internally)
            print(init_string)

    def reset_state(self, B=1):
        """ Resets states of all internal layer M matrices """

        for mp_layer in self.mp_layers:
            mp_layer.reset_state(B=B)

    def param_clamp(self):
        # mp_layer call doesn't track gradients, since this is always called after weight updates
        for mp_layer in self.mp_layers:
            mp_layer.param_clamp()

    @torch.no_grad()
    def _monitor_init(self, train_params, train_data, train_trails=None, valid_batch=None, valid_trails=None):

        # Additional quantities to track during training, note initializes these first so that _monitor call
        # inside super()._monitor_init can append additional quantities.
        if self.hist is None:
            self.hist = {
                'iter': 0,
            }
            for mpl_idx, mp_layer in enumerate(self.mp_layers):
                self.hist['eta{}'.format(mp_layer.mp_layer_name)] = []
                self.hist['lam{}'.format(mp_layer.mp_layer_name)] = []

        super()._monitor_init(train_params, train_data, train_trails=train_trails, valid_batch=valid_batch, valid_trails=valid_trails)

    @torch.no_grad()
    def _monitor(self, train_batch, train_go_info_batch, valid_go_info_batch, train_type='supervised', output=None, loss=None, loss_components=None,
                 acc=None, valid_batch=None):

        super()._monitor(train_batch, train_go_info_batch, valid_go_info_batch, output=output, loss=loss, loss_components=loss_components,
                         valid_batch=valid_batch)

        for mpl_idx, mp_layer in enumerate(self.mp_layers):
            self.hist['eta{}'.format(mp_layer.mp_layer_name)].append(
                mp_layer.eta.detach().cpu().numpy()
            )
            self.hist['lam{}'.format(mp_layer.mp_layer_name)].append(
                mp_layer.lam.detach().cpu().numpy()
            )

class MultiPlasticNet(MultiPlasticNetBase):
    """
    Two-layer feedforward setup, with single multi-plastic layer followed by a readout layer.
    """

    def __init__(self, net_params, verbose=False):

        if 'n_neurons' in net_params:
            assert len(net_params['n_neurons']) == 3
            self.n_input = net_params['n_neurons'][0]
            self.n_hidden = net_params['n_neurons'][1]
            self.n_output = net_params['n_neurons'][2]
        else:
            self.n_input = net_params['n_input']
            self.n_hidden = net_params['n_hidden']
            self.n_output = net_params['n_output']

        super().__init__(net_params, self.n_hidden, verbose=verbose)

        # Creates the input MP layer
        self.param_clamping = True # Always have param clamping for MP layers because lam bounds
        net_params['ml_params']['n_input'] = self.n_input
        net_params['ml_params']['n_output'] = self.n_hidden
        net_params['ml_params']['dt'] = self.dt
        self.mp_layer = MultiPlasticLayer(net_params['ml_params'], output_matrix=self.output_matrix, verbose=verbose)
        self.params.extend(self.mp_layer.params)

        self.mp_layers = [self.mp_layer,] # List of all mp_layers in this network

    def forward(self, inputs, run_mode='minimal', verbose=False):

        x = torch.clone(inputs)

        # Returns pre-activation
        hidden_pre, db_mp = self.mp_layer(x, run_mode=run_mode)

        hidden = self.act_fn(hidden_pre)

        output_hidden = torch.einsum('iI, BI -> Bi', self.W_output, hidden)
        output = output_hidden + self.b_output.unsqueeze(0)

        if run_mode in ('track_states'):
            db = {
                'M': db_mp['M'],
                'hidden_pre': hidden_pre.detach(),
                'hidden': hidden.detach(),
                "input": x.detach(), 
            }
        else:
            db = None

        return output, hidden, db

    def network_step(self, current_input, run_mode='minimal', verbose=False):
        """
        Performs a single batch pass forward for the network. This mostly consists of a forward pass and
        the associated updates to internal states (i.e. the modulations)

        This should not be passed a full sequence of data, only data from a given time point
        """

        assert len(current_input.shape) == 2

        output, current_hidden, db = self.forward(current_input, run_mode=run_mode, verbose=verbose)

        # M updated internally when this is called, M here is only used if finding fixed points (not yet implemented)
        M = self.mp_layer.update_M_matrix(current_input, current_hidden)

        return output, db

class DeepMultiPlasticNet(MultiPlasticNetBase):
    """
    N-layer feedforward setup, with N-1 multi-plastic layers followed by a single readout layer.
    """

    def __init__(self, net_params, verbose=False):

        # Mar 16th: add input layer
        self.input_layer_active = net_params.get('input_layer_add', False)
        self.input_layer_active_trainable = net_params.get('input_layer_add_trainable', False)

        if self.input_layer_active:
            net_params['n_neurons'].insert(1, net_params['n_neurons'][1])

        print(net_params['n_neurons'])
        self.n_hidden = net_params['n_neurons'][1]

        n_layers = len(net_params['n_neurons']) - 1        

        self.n_input = net_params['n_neurons'][0]
        self.n_output = net_params['n_neurons'][-1]

        self.output_matrix = net_params['output_matrix']

        super().__init__(net_params, net_params['n_neurons'][-2], output_matrix=self.output_matrix, verbose=verbose)

        # Creates all the MP layers
        self.param_clamping = True # Always have param clamping for MP layers because lam bounds

        if self.input_layer_active:
            self.W_initial_linear = nn.Linear(net_params['n_neurons'][0], net_params['n_neurons'][1])

            self.W_initial_linear.weight.data = torch.tensor(
                rand_weight_init(net_params['n_neurons'][0], net_params['n_neurons'][1], init_type=net_params.get('W_init', 'xavier')),
                dtype=torch.float
            )
            
            if not self.input_layer_active_trainable:
                print("Input Layer Frozen")
                self.W_initial_linear.weight.requires_grad = False

            if net_params.get('input_layer_bias', False):
                self.W_initial_linear.bias.data = torch.tensor(
                    rand_weight_init(net_params['n_neurons'][1], init_type='gaussian'),
                    dtype=torch.float
                )
            else:
                self.W_initial_linear.bias = None

        self.mp_layers = []
        
        start_layer_count = 1 if self.input_layer_active else 0
        # if additional input layer is added, shift the starting index of layer counting from 1
        for mpl_idx in range(start_layer_count, n_layers - 1): # (e.g. three-layer has two MP layers)
            # Can specify layer with either layer-specific 'ml_params' or just a single set of parameters
            if f'ml_params{mpl_idx}' in net_params.keys():
                print("=== Layer Specific Setup ===")
                ml_params_key = f'ml_params{mpl_idx}'
            else:
                print("=== Layer Universal Setup ===")
                ml_params_key = 'ml_params'

            # Updates some parameters for each new MPL
            net_params[ml_params_key]['dt'] = self.dt
            net_params[ml_params_key]['mpl_name'] = str(mpl_idx)
            net_params[ml_params_key]['n_input'] = net_params['n_neurons'][mpl_idx]
            print(net_params['n_neurons'][mpl_idx])
            net_params[ml_params_key]['n_output'] = net_params['n_neurons'][mpl_idx + 1]

            setattr(self, 'mp_layer{}'.format(mpl_idx),
                    MultiPlasticLayer(net_params[ml_params_key], output_matrix=self.output_matrix, verbose=verbose))

            self.mp_layers.append(getattr(self, 'mp_layer{}'.format(mpl_idx)))
            self.params.extend([param+str(mpl_idx) for param in self.mp_layers[-1].params])

    def forward(self, inputs, run_mode='minimal', verbose=False):
    
        if self.input_layer_active:
            x = self.W_initial_linear(inputs)
            x = self.act_fn(x)
        else:
            x = inputs

        layer_input = torch.clone(x)

        mpl_activities = [x,] # Used for updating the M matrices

        db = {} if run_mode in ('track_states',) else None

        for mpl_idx, mp_layer in enumerate(self.mp_layers):
            # Returns pre-activation
            layer_input_old = copy.deepcopy(layer_input)
            hidden_pre, db_mp = mp_layer(layer_input, run_mode=run_mode)
            layer_input = self.act_fn(hidden_pre)

            print(layer_input_old.shape)
            print(hidden_pre.shape)
            print(layer_input.shape)
            time.sleep(1000)

            if run_mode in ('debug',):
                print(f'  MP Layer {mpl_idx} forward.')
                print('   Pre-act mean {:.2e} Post-act mean {:.2e}'.format(
                    torch.mean(hidden_pre.detach()), torch.mean(layer_input.detach())
                ))

            mpl_activities.append(layer_input) # Postsyn activity
            if run_mode in ('track_states',):
                db['hidden_pre{}'.format(mp_layer.mp_layer_name)] = hidden_pre.detach()
                db['hidden{}'.format(mp_layer.mp_layer_name)] = layer_input.detach()
                db['M{}'.format(mp_layer.mp_layer_name)] = db_mp['M']
                db['b{}'.format(mp_layer.mp_layer_name)] = db_mp['b']

        if run_mode in ('debug',): print(f'  Output layer forward.')
        output_hidden = torch.einsum('iI, BI -> Bi', self.W_output, layer_input)
        output = output_hidden + self.b_output.unsqueeze(0)

        
        return output, mpl_activities, db

    def network_step(self, current_input, run_mode='minimal', verbose=False):
        """
        Performs a single batch pass forward for the network. This mostly consists of a forward pass and
        the associated updates to internal states (i.e. the modulations)

        This should not be passed a full sequence of data, only data from a given time point
        """

        assert len(current_input.shape) == 2
        if run_mode in ('debug',): print(f' Network step:')

        output, mpl_activities, db = self.forward(current_input, run_mode=run_mode, verbose=verbose)

        # M updated internally when this is called
        for mpl_idx, mp_layer in enumerate(self.mp_layers):
            if run_mode in ('debug',):
                print(f'  MP Layer {mpl_idx} M update.')
                print('   Pre mean {:.2e} post mean {:.2e}'.format(
                    torch.mean(mpl_activities[mpl_idx].detach()), torch.mean(mpl_activities[mpl_idx + 1].detach())
                ))
                print('   M mag mean {:.2e} max {:.2e}'.format(
                    torch.mean(torch.abs(mp_layer.M.detach())), torch.max(torch.abs(mp_layer.M.detach()))
                ))

            _ = mp_layer.update_M_matrix(mpl_activities[mpl_idx], mpl_activities[mpl_idx + 1])

        return output, mpl_activities, db