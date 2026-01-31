# nonlinear_dagmpg.py
# Per-sample (per-input) plasticity DAG-MPN
# Hidden state is computed as a *nonlinear DAG perceptron* in topological order:
#   h_j = activation_fn(v_j + sum_{i<j} A_{i,j} h_i)
# where v = W_in x + b_in, and A = W ⊙ (1 + M) (strictly upper-triangular).
#
# Includes:
#   - memory-safe per-sample M (no graph persistence)
#   - time-chunk checkpointing to control activation memory
#   - two h-computation modes:
#       * "sequential"  : loop over neurons (clearest)
#       * "blockwise"   : loop over blocks, uses batched matmuls (more vectorized)
#
# NOTE: A is NOT materialized as (B,H,H) to avoid wasting memory.
#       We slice W and M on-the-fly.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class _LayerView:
    W: torch.nn.Parameter


class DeepMultiPlasticNet(nn.Module):
    """
    Per-sample plasticity DAG network with nonlinear hidden units.

    Per time step:
      v = W_in x + b_in
      A^{(b)} = W ⊙ (1 + M^{(b)})   (strictly upper-triangular)
      h_j^{(b)} = activation_fn( v_j^{(b)} + sum_{i<j} A_{i,j}^{(b)} h_i^{(b)} )
      M_{t+1}^{(b)} = lam * M_t^{(b)} + eta * P(h h^T)

    In the loop:
      - forward_sequence_checkpointed(x_TBD, chunk_size, Ms0=None, run_mode="minimal") -> logits_last (B,d_out)
      - param_clamp()
    """

    def __init__(self, net_params: Dict[str, Any], verbose: bool = True, forzihan: bool = False):
        super().__init__()
        self.verbose = verbose

        n_neurons = net_params.get("n_neurons", [1, 300, 10])
        self.d_in = int(n_neurons[0])
        self.d_h = int(n_neurons[1])
        self.d_out = int(n_neurons[-1])

        # Single activation hook (replaces self.sigma_in)
        act = str(net_params.get("activation", "tanh")).lower()
        if act == "tanh":
            self.activation_fn = torch.tanh
        elif act == "relu":
            self.activation_fn = F.relu
        elif act == "gelu":
            self.activation_fn = F.gelu
        else:
            self.activation_fn = torch.tanh

        input_bias = bool(net_params.get("input_layer_bias", True))
        output_bias = bool(net_params.get("output_bias", True))

        mlp = net_params.get("ml_params", {})
        m_time_scale = float(mlp.get("m_time_scale", 100.0))
        lam = float(mlp.get("lam", None) or torch.exp(torch.tensor(-1.0 / m_time_scale)).item())
        self.lam = lam

        # eta: default eta=(1-lam)*eta0
        eta0 = float(mlp.get("eta0", 0.1))
        eta = float(mlp.get("eta", (1.0 - lam) * eta0))
        self.eta = eta

        W_freeze = bool(mlp.get("W_freeze", False))

        # Hidden compute mode
        #   "sequential"  : neuron-by-neuron (simple)
        #   "blockwise"   : block-by-block with batched matmul (more vectorized)
        self.h_mode = str(net_params.get("h_mode", "blockwise")).lower()
        self.h_block = int(net_params.get("h_block", 64))
        self.h_block = max(1, min(self.d_h, self.h_block))

        # Parameters
        self.W_in = nn.Parameter(torch.empty(self.d_h, self.d_in))
        self.b_in = nn.Parameter(torch.zeros(self.d_h)) if input_bias else None

        self.W = nn.Parameter(torch.empty(self.d_h, self.d_h))
        if W_freeze:
            self.W.requires_grad_(False)

        self.W_out = nn.Parameter(torch.empty(self.d_out, self.d_h))
        self.b_out = nn.Parameter(torch.zeros(self.d_out)) if output_bias else None

        # For logging: net.mp_layers[0].W
        self.mp_layers = [_LayerView(self.W)]

        # Buffers
        upper = torch.triu(torch.ones(self.d_h, self.d_h), diagonal=1)
        self.register_buffer("upper_mask", upper)

        self._init_parameters(net_params)

        if self.verbose:
            print(
                f"[Nonlinear DAG-MPN per-sample M] d_in={self.d_in}, d_h={self.d_h}, d_out={self.d_out}, "
                f"lam={self.lam:.6f}, eta={self.eta:.6g}, h_mode={self.h_mode}, h_block={self.h_block}"
            )

    def _init_parameters(self, net_params: Dict[str, Any]) -> None:
        nn.init.xavier_uniform_(self.W_in)
        if self.b_in is not None:
            nn.init.zeros_(self.b_in)

        # W init: fan-in scaling + diagonal-distance decay + mild row L1 normalization
        g = float(net_params.get("dag_W_gain", 0.3))
        alpha = float(net_params.get("dag_W_diag_decay", 0.05))
        c = float(net_params.get("dag_row_l1_target", 0.5))

        with torch.no_grad():
            W = torch.randn(self.d_h, self.d_h, device=self.W.device, dtype=torch.float32)

            idx = torch.arange(self.d_h, device=W.device)
            k = (self.d_h - idx - 1).clamp(min=1).float()
            row_std = (g / torch.sqrt(k)).view(self.d_h, 1)

            ii = idx.view(-1, 1)
            jj = idx.view(1, -1)
            dist = (jj - ii).clamp(min=0).float()
            decay = torch.exp(-alpha * dist)

            W = W * row_std * decay
            W = W * self.upper_mask

            if c > 0:
                l1 = W.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
                W = W * (c / l1)

            self.W.copy_(W.to(dtype=self.W.dtype))

        # W_out init
        wout_init = str(net_params.get("W_output_init", "xavier")).lower()
        if wout_init == "xavier":
            nn.init.xavier_uniform_(self.W_out)
        else:
            nn.init.kaiming_uniform_(self.W_out, a=math.sqrt(5))

        if self.b_out is not None:
            nn.init.zeros_(self.b_out)

    @torch.no_grad()
    def param_clamp(self) -> None:
        w_clip = 3.0
        self.W.clamp_(-w_clip, w_clip)
        self.W_in.clamp_(-w_clip, w_clip)
        self.W_out.clamp_(-w_clip, w_clip)
        if self.b_in is not None:
            self.b_in.clamp_(-w_clip, w_clip)
        if self.b_out is not None:
            self.b_out.clamp_(-w_clip, w_clip)

    def _project_upper(self, M: torch.Tensor) -> torch.Tensor:
        # Works for (..., H, H)
        return M * self.upper_mask

    def _compute_h_sequential(self, v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Neuron-by-neuron DAG perceptron computation.
        Does NOT materialize A=(B,H,H).
        """
        B, H = v.shape
        h = v.new_zeros((B, H))

        # A_{i,j}^{(b)} = W_{i,j} * (1 + M_{i,j}^{(b)}), with i<j
        for j in range(H):
            if j == 0:
                pre = v[:, 0]
            else:
                # A_col: (B, j) for i=0..j-1
                # W_col: (j,)
                W_col = self.W[:j, j]  # (j,)
                A_col = (1.0 + M[:, :j, j]) * W_col.unsqueeze(0)  # (B, j)
                rec = (h[:, :j] * A_col).sum(dim=1)  # (B,)
                pre = v[:, j] + rec

            h[:, j] = self.activation_fn(pre)

        return h

    def _compute_h_blockwise(self, v: torch.Tensor, M: torch.Tensor, block: int) -> torch.Tensor:
        """
        Blockwise DAG perceptron computation (more vectorized):
          For a block of neurons [j0:j1):
             rec = h_prev @ A_prev_to_block   (batched matmul)
             h_block = activation_fn(v_block + rec)
        Does NOT materialize A=(B,H,H).
        """
        B, H = v.shape
        h = v.new_zeros((B, H))

        for j0 in range(0, H, block):
            j1 = min(j0 + block, H)

            v_blk = v[:, j0:j1]  # (B, blk)

            if j0 == 0:
                # no previous neurons
                pre_blk = v_blk
            else:
                # A_prev_to_block: (B, j0, blk)
                #   = W[:j0, j0:j1] * (1 + M[:, :j0, j0:j1])
                W_sub = self.W[:j0, j0:j1]  # (j0, blk)
                A_sub = (1.0 + M[:, :j0, j0:j1]) * W_sub.unsqueeze(0)  # (B, j0, blk)

                # rec = h_prev @ A_sub  -> (B, blk)
                # Use bmm to avoid expanding to (B,j0,1) etc.
                h_prev = h[:, :j0].unsqueeze(1)  # (B, 1, j0)
                rec = torch.bmm(h_prev, A_sub).squeeze(1)  # (B, blk)

                pre_blk = v_blk + rec

            h[:, j0:j1] = self.activation_fn(pre_blk)

        return h

    def _step(self, x_t: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-sample step.

        Args:
            x_t: (B, d_in)
            M:   (B, d_h, d_h)  strictly upper-triangular (masked)

        Returns:
            h:   (B, d_h)
            M1:  (B, d_h, d_h)
        """
        # v = W_in x + b  (nonlinearity happens in hidden nodes, not here)
        v = F.linear(x_t, self.W_in, self.b_in)  # (B,H)

        # Hidden computation (no (I-A)^{-1}, no solve)
        if self.h_mode == "sequential":
            h = self._compute_h_sequential(v, M)
        else:
            h = self._compute_h_blockwise(v, M, self.h_block)

        # Per-sample plasticity update: dM = P(h h^T)
        # Allocate (B,H,H) once per step; then mask to keep upper-triangular.
        dM = torch.einsum("bi,bj->bij", h, h) / math.sqrt(self.d_h)
        dM = self._project_upper(dM)

        # Update M and re-mask to prevent drift into lower triangle
        M1 = (self.lam * M) + (self.eta * dM)
        M1 = self._project_upper(M1)

        return h, M1

    def forward_sequence_checkpointed(
        self,
        x_TBD: torch.Tensor,
        chunk_size: int = 32,
        Ms0: Optional[torch.Tensor] = None,
        run_mode: str = "minimal",
    ) -> torch.Tensor:
        """
        Args:
            x_TBD: (T, B, d_in)
            Ms0: optional initial M.
                 - if None: zeros
                 - if shape (H,H): broadcast to (B,H,H)
                 - if shape (B,H,H): use directly

        Returns:
            logits_last: (B, d_out)
        """
        assert x_TBD.dim() == 3, "Expected x_TBD with shape (T,B,D)"
        T, B, D = x_TBD.shape
        assert D == self.d_in, f"Input dim mismatch: got {D}, expected {self.d_in}"

        # Initialize per-sample M as a LOCAL tensor (prevents graph persistence/leaks)
        if Ms0 is None:
            M = x_TBD.new_zeros((B, self.d_h, self.d_h))
        else:
            Ms0 = Ms0.to(device=x_TBD.device, dtype=x_TBD.dtype)
            if Ms0.dim() == 2:
                M = Ms0.unsqueeze(0).expand(B, -1, -1).contiguous()
            elif Ms0.dim() == 3:
                assert Ms0.shape[0] == B, "Ms0 batch mismatch"
                M = Ms0
            else:
                raise ValueError("Ms0 must be None, (H,H), or (B,H,H)")
            M = self._project_upper(M)

        # Dummy tensor: makes checkpoint keep autograd on (otherwise grads can be dropped)
        dummy = x_TBD.new_zeros((), requires_grad=True)

        def run_chunk(
            x_chunk: torch.Tensor, M_in: torch.Tensor, dummy_in: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            M_local = M_in
            h_local = None
            Tc = x_chunk.shape[0]
            for t in range(Tc):
                h_local, M_local = self._step(x_chunk[t], M_local)
            return h_local, M_local, dummy_in

        # Chunked checkpointing across time
        for t0 in range(0, T, chunk_size):
            x_chunk = x_TBD[t0 : min(t0 + chunk_size, T)]
            h_last, M, dummy = checkpoint(run_chunk, x_chunk, M, dummy, use_reentrant=False)

        logits_last = F.linear(h_last, self.W_out, self.b_out)
        return logits_last