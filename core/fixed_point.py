"""
Gradient-based fixed-point finding for the plastic modulation matrix M.

The one-task / two-task attractor analyses estimate "fixed points" as the state
at the last timestep of an artificially-lengthened trial period (a settling
proxy). This module instead solves for TRUE fixed points of the modulation
dynamics by gradient descent, following Sussillo & Barak (2013) as adapted for
plastic networks.

Idea (mirrors the classic RNN fixed-point trick): freeze the trained network's
parameters, then treat a batch of candidate STATES as the trainable parameters
of a small optimizer problem. Minimize the "speed"

    q(M) = 1/2 || F(M; x) - M ||^2

where F(M; x) is one network update step of the modulation matrix under a fixed,
constant input x. A minimizer with q ≈ 0 is a fixed point M* = F(M*; x).

For the MPN the evolving state is the modulation matrix M (shape (B, post, pre))
of the single multi-plastic layer, and F is one `network_step` under constant
input. This differs from the RNN/GRU/HebbNet template (whose state is a hidden
vector self.h / self.A); here the state is M and we drive it via the layer's own
`update_M_matrix`.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mp_layer(net):
    """The single multi-plastic layer of a (Deep)MultiPlasticNet."""
    return net.mp_layers[0]


class ModulationFixedPointNetwork(nn.Module):
    """
    Finds fixed points of the modulation matrix M for an MPN under a constant
    input.

    The trained network's weights are frozen; the trainable parameters are a
    batch of candidate M matrices (`self.states`, shape (B, post, pre)). One
    forward pass applies a single network update step under the constant input
    and returns the resulting M; training with MSE(M_next, M) drives the batch
    toward fixed points.
    """

    def __init__(self, network, init_states):
        """
        network     : a trained (Deep)MultiPlasticNet (its parameters are frozen)
        init_states : (B, post, pre) array/tensor of initial candidate M matrices
                      (e.g. the recorded M at the middle/end of a trial period).
        """
        super().__init__()
        self.eval()
        self.name = self.__class__.__name__

        # Frozen copy of the analyzed network. deepcopy fails if the network
        # carries non-leaf state tensors from prior forward passes (mp.M / M_pre
        # are re-derived each step and are attached to a graph). Temporarily
        # swap those transient buffers for detached leaves, copy, then restore
        # the originals on the caller's network so it is left untouched.
        stashed = []
        for lyr in network.mp_layers:
            for attr in ("M", "M_pre"):
                val = getattr(lyr, attr, None)
                if isinstance(val, torch.Tensor):
                    stashed.append((lyr, attr, val))
                    setattr(lyr, attr, val.detach().clone())
        try:
            net_fzn = copy.deepcopy(network)
        finally:
            for lyr, attr, val in stashed:
                setattr(lyr, attr, val)   # restore caller's original tensors

        for param in net_fzn.parameters():
            param.requires_grad = False
        net_fzn.eval()
        self.net = net_fzn
        self.mp = _mp_layer(net_fzn)

        # The optimized states ARE the modulation matrices.
        init = torch.as_tensor(np.asarray(init_states), dtype=torch.float)
        assert init.dim() == 3, (
            f"init_states must be (B, post, pre); got shape {tuple(init.shape)}")
        self.states = nn.Parameter(init.clone())

        print("FP Network - NetType: {}, States (M) size: {}".format(
            type(net_fzn).__name__, tuple(self.states.shape)))

    # ── One update step of M under a constant input ──────────────────────────
    def _step_M(self, inputs, current_states):
        """
        Return F(M; x): the modulation matrix after ONE network step, starting
        from `current_states` (B, post, pre) under constant input `inputs`
        (B, n_input). Restores the layer's stored M afterward so repeated calls
        are side-effect free.

        `update_M_matrix` mutates layer.M in place, so we set M = current_states,
        run one step (forward + M update), read the updated M as F(M), then
        restore the layer's original M.
        """
        mp = self.mp
        saved_M = mp.M
        saved_M_pre = getattr(mp, "M_pre", None)

        # Seed the layer with the candidate states. Assign a TENSOR (not the
        # nn.Parameter itself): nn.Module.__setattr__ would try to register a
        # Parameter as a submodule-parameter and later fail when update_M_matrix
        # reassigns mp.M with a plain tensor. Multiplying by 1.0 yields a
        # grad-tracking tensor view of the parameter, so gradients still flow
        # back to self.states.
        mp.M = current_states * 1.0
        # network_step: forward (uses mp.M via get_modulated_weights) + M update.
        self.net.network_step(inputs, run_mode="minimal")
        next_M = mp.M

        # Restore so the layer state is unchanged for the next call.
        mp.M = saved_M
        if saved_M_pre is not None:
            mp.M_pre = saved_M_pre
        return next_M

    def forward(self, inputs, current_states=None):
        """One-step update of M. Uses the optimized `self.states` by default, or
        `current_states` if provided (e.g. for measuring speeds of given points).
        Returns the next M (B, post, pre)."""
        states = self.states if current_states is None else current_states
        return self._step_M(inputs, states)

    # ── Speeds q(M) = 1/2 ||F(M) - M||^2 ─────────────────────────────────────
    def get_speeds(self, inputs, current_states=None):
        """Per-point speed q(M) for the batch (numpy, shape (B,)). Norm is over
        the (post, pre) matrix dims."""
        with torch.no_grad():
            ref = self.states if current_states is None else current_states
            next_state = self(inputs, current_states=current_states)
            return (0.5 * torch.norm(next_state - ref, dim=(1, 2)) ** 2).cpu().numpy()

    # ── Optimize the batch toward fixed points ───────────────────────────────
    def find_fixed_points(self, inputs, steps, learningRate=1e-3, printPeriod=10):
        """
        Gradient-descend the candidate states toward fixed points of M under the
        constant input `inputs` (B, n_input).

        Returns (states, loss_hist, final_speeds):
          states       : detached (B, post, pre) tensor of found fixed points
          loss_hist    : list of per-step MSE losses
          final_speeds : per-point speed q(M*) (numpy, (B,)); small ⇒ good FP.
        """
        inputs = torch.as_tensor(np.asarray(inputs), dtype=torch.float,
                                 device=self.states.device)

        init_speeds = self.get_speeds(inputs)
        print("Init speeds - Max: {:.2e} / Min: {:.2e}".format(
            float(np.max(init_speeds)), float(np.min(init_speeds))))

        self.optimizer = torch.optim.Adam([self.states], lr=learningRate)

        loss_hist = []
        for step in range(steps):
            self.optimizer.zero_grad()
            next_state = self(inputs)                    # F(M), (B, post, pre)
            # Labels are the current states: drive F(M) -> M.
            loss = F.mse_loss(next_state, self.states, reduction="mean")
            loss_hist.append(loss.item())
            loss.backward()
            self.optimizer.step()
            if step % printPeriod == 0:
                print("  Step {} - Loss: {:.3e}".format(step, loss.item()))

        final_speeds = self.get_speeds(inputs)
        print("Final speeds - Max: {:.2e} / Min: {:.2e}".format(
            float(np.max(final_speeds)), float(np.min(final_speeds))))

        return self.states.detach(), loss_hist, final_speeds


def find_modulation_fixed_points(network, init_M, inputs, steps=2000,
                                 learningRate=1e-3, printPeriod=200, device=None):
    """
    Convenience wrapper: build a ModulationFixedPointNetwork seeded at `init_M`
    and optimize it under constant `inputs`.

    network : trained (Deep)MultiPlasticNet.
    init_M  : (B, post, pre) initial modulation matrices (e.g. recorded M at a
              period midpoint), one per candidate / stimulus.
    inputs  : (B, n_input) constant per-candidate input held fixed during the
              relaxation (e.g. the fixation-only input for a delay fixed point).
    Returns (fixed_M, loss_hist, final_speeds) with fixed_M as a numpy array.
    """
    fpn = ModulationFixedPointNetwork(network, init_M)
    if device is not None:
        fpn.to(device)
        fpn.states.data = fpn.states.data.to(device)
    fixed_M, loss_hist, final_speeds = fpn.find_fixed_points(
        inputs, steps, learningRate=learningRate, printPeriod=printPeriod)
    return fixed_M.cpu().numpy(), loss_hist, final_speeds
