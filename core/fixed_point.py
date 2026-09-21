"""
Gradient-based fixed-point finding for the plastic modulation matrix M.

The one-task / two-task attractor analyses estimate "fixed points" as the state
at the last timestep of an artificially-lengthened trial period (a settling
proxy). This module instead solves for TRUE fixed points of the modulation
dynamics by gradient descent, following Sussillo & Barak (2013) as adapted for
plastic networks.

Idea (mirrors the classic RNN fixed-point trick): freeze the trained network's
parameters, then treat a batch of candidate STATES as the trainable parameters
of a small optimizer problem. The raw diagnostic speed is

    q(M) = 1/2 || F(M; x) - M ||^2

where F(M; x) is one network update step under a fixed input x. Optimization uses
the leak-normalized residual (F(M; x)-M)/(1-lambda), which has the same zeros but
does not become artificially flat for slow modulation. A zero is a fixed point
M* = F(M*; x); raw q is still returned for diagnostics and compatibility.

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
    and returns the resulting M; training with the leak-normalized residual
    MSE((M_next-M)/(1-lambda), 0) drives the batch toward fixed points without
    making slow-plasticity models artificially easy to optimize.
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

        # F(M) - M is proportional to the modulation leak (1 - lambda).  For
        # slow-plasticity runs (lambda ~= 0.99) the unscaled MSE is therefore
        # smaller by ~1e4 and its gradients are needlessly flat, even though the
        # location of every zero is unchanged.  Optimize the leak-normalized
        # residual while continuing to report the raw one-step speed below.
        lam = self.mp.build_M_parameter(
            self.mp.lam, self.mp.lam_type).detach().to(dtype=torch.float)
        leak = torch.clamp(torch.abs(1.0 - lam), min=1e-6)
        self.register_buffer("residual_leak", leak)

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
    def _speed_loss(self, inputs, current_states=None):
        """Leak-normalized MSE(F(M), M).

        Dividing each modulation entry by its own ``1-lambda`` factor leaves the
        fixed points unchanged but prevents a slow modulation time constant from
        making the objective and gradients look artificially converged.
        """
        states = self.states if current_states is None else current_states
        next_state = self(inputs, current_states=states)  # F(M), (B, post, pre)
        residual = (next_state - states) / self.residual_leak.unsqueeze(0)
        return torch.mean(residual.square())

    def relative_residuals(self, inputs, current_states=None):
        """Per-candidate leak-normalized relative residual.

        This is the exact convergence quantity used by the analysis, generalized
        to element-wise leak factors:

            ||(F(M)-M)/(1-lambda)|| / ||M||.
        """
        states = self.states if current_states is None else current_states
        if torch.is_tensor(inputs):
            inputs = inputs.to(dtype=torch.float, device=states.device)
        else:
            inputs = torch.as_tensor(np.asarray(inputs), dtype=torch.float,
                                     device=states.device)
        with torch.no_grad():
            next_state = self(inputs, current_states=states)
            residual = (next_state - states) / self.residual_leak.unsqueeze(0)
            numerator = torch.linalg.vector_norm(
                residual.reshape(residual.shape[0], -1), dim=1)
            denominator = torch.clamp(torch.linalg.vector_norm(
                states.reshape(states.shape[0], -1), dim=1), min=1e-12)
        return (numerator / denominator).detach().cpu().numpy()

    def polish_candidates_individually(self, inputs, indices, lbfgs_steps=200):
        """L-BFGS-polish selected candidates independently and keep improvements.

        The ordinary batch L-BFGS pass uses one line search for every candidate.
        A minority of difficult candidates can therefore plateau even when most
        of the batch is already accurate. Here each selected state gets its own
        parameter and line search while reusing the same frozen network copy.
        A proposal is committed only when its normalized relative residual is
        finite and lower than the original value.

        Returns a dict containing attempted/accepted masks and before/after
        residuals for auditability.
        """
        if torch.is_tensor(inputs):
            inputs = inputs.to(dtype=torch.float, device=self.states.device)
        else:
            inputs = torch.as_tensor(np.asarray(inputs), dtype=torch.float,
                                     device=self.states.device)
        indices = np.asarray(indices, dtype=int).reshape(-1)
        n = int(self.states.shape[0])
        attempted = np.zeros(n, dtype=bool)
        accepted = np.zeros(n, dtype=bool)
        before = self.relative_residuals(inputs)

        for raw_idx in indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= n:
                raise IndexError(f"candidate index {idx} outside batch of {n}")
            attempted[idx] = True
            candidate = nn.Parameter(
                self.states[idx:idx + 1].detach().clone())
            optimizer = torch.optim.LBFGS(
                [candidate], max_iter=int(lbfgs_steps), lr=1.0,
                tolerance_grad=1e-16, tolerance_change=1e-18,
                history_size=20, line_search_fn="strong_wolfe")
            input_i = inputs[idx:idx + 1]

            def _closure():
                optimizer.zero_grad()
                loss = self._speed_loss(input_i, current_states=candidate)
                loss.backward()
                return loss

            try:
                optimizer.step(_closure)
                proposed = self.relative_residuals(
                    input_i, current_states=candidate)[0]
            except (RuntimeError, ValueError) as exc:
                print(f"  [rescue] candidate {idx} failed: {exc}")
                continue
            if np.isfinite(proposed) and proposed < before[idx]:
                with torch.no_grad():
                    self.states[idx].copy_(candidate[0])
                accepted[idx] = True

        after = self.relative_residuals(inputs)
        return {
            "attempted": attempted,
            "accepted": accepted,
            "rel_step_undamped_before": before,
            "rel_step_undamped_after": after,
            "lbfgs_steps": int(lbfgs_steps),
        }

    def find_fixed_points(self, inputs, steps, learningRate=1e-3, printPeriod=10,
                          lbfgs_steps=500, loss_tol=1e-8):
        """
        Descend the candidate states toward fixed points of M under the constant
        input `inputs` (B, n_input), in two stages:

          1. Adam — a robust first pass from the recorded seed toward the
             fixed-point basin. The optimized MSE uses the leak-normalized
             residual (F(M)-M)/(1-lambda), so its scale is comparable across
             modulation time constants. It runs until that loss drops to
             `loss_tol` or `steps` is reached. `steps` is therefore a maximum,
             not a fixed count. Set `loss_tol=0` (or None) to run all steps.
          2. L-BFGS for up to `lbfgs_steps` iterations (strong-Wolfe line search)
             — second-order polishing that drives the speed q(M) orders of
             magnitude lower than Adam alone can (the standard Sussillo & Barak
             refinement). Set `lbfgs_steps=0` to skip.

        Returns (states, loss_hist, final_speeds):
          states       : detached (B, post, pre) tensor of found fixed points
          loss_hist    : list of per-step MSE losses (Adam stage)
          final_speeds : per-point speed q(M*) (numpy, (B,)); small ⇒ good FP.
        """
        inputs = torch.as_tensor(np.asarray(inputs), dtype=torch.float,
                                 device=self.states.device)

        init_speeds = self.get_speeds(inputs)
        print("Init speeds - Max: {:.2e} / Min: {:.2e}".format(
            float(np.max(init_speeds)), float(np.min(init_speeds))))

        # ── Stage 1: Adam (run until loss <= loss_tol, capped at `steps`) ─────
        self.optimizer = torch.optim.Adam([self.states], lr=learningRate)
        loss_hist = []
        last_step, last_loss = 0, float("inf")
        for step in range(steps):
            self.optimizer.zero_grad()
            loss = self._speed_loss(inputs)              # drive F(M) -> M
            loss_val = loss.item()
            loss_hist.append(loss_val)
            last_step, last_loss = step, loss_val
            loss.backward()
            self.optimizer.step()
            if step % printPeriod == 0:
                print("  [adam] Step {} - Loss: {:.3e}".format(step, loss_val))
            # Early stop once the speed loss has converged to the tolerance.
            if loss_tol and loss_val <= loss_tol:
                print("  [adam] converged: Step {} - Loss: {:.3e} "
                      "(<= tol {:.1e})".format(step, loss_val, loss_tol))
                break
        else:
            if loss_tol:
                print("  [adam] hit max steps ({}) without reaching tol {:.1e}; "
                      "last loss {:.3e}".format(steps, loss_tol, last_loss))

        adam_speeds = self.get_speeds(inputs)
        print("Post-Adam speeds - Max: {:.2e} / Min: {:.2e}".format(
            float(np.max(adam_speeds)), float(np.min(adam_speeds))))

        # ── Stage 2: L-BFGS polishing ────────────────────────────────────────
        # Second-order refinement on the same speed objective. The closure is
        # re-evaluated by the line search, so each call rebuilds the graph.
        if lbfgs_steps and lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                [self.states], max_iter=int(lbfgs_steps), lr=1.0,
                tolerance_grad=1e-16, tolerance_change=1e-18,
                history_size=50, line_search_fn="strong_wolfe")

            def _closure():
                lbfgs.zero_grad()
                loss = self._speed_loss(inputs)
                loss.backward()
                return loss

            lbfgs.step(_closure)
            polish_speeds = self.get_speeds(inputs)
            print("Post-LBFGS speeds - Max: {:.2e} / Min: {:.2e}".format(
                float(np.max(polish_speeds)), float(np.min(polish_speeds))))

        final_speeds = self.get_speeds(inputs)
        print("Final speeds - Max: {:.2e} / Min: {:.2e}".format(
            float(np.max(final_speeds)), float(np.min(final_speeds))))

        return self.states.detach(), loss_hist, final_speeds


def characterize_fixed_point_stability(network, fixed_M, inputs, k=16,
                                       marginal_tol=5e-2, device=None):
    """
    Linear-stability analysis of modulation fixed points, following the RNN
    fixed-point tradition (Sussillo & Barak 2013): linearize the update map
    F(M; x) about each M* and read stability from the Jacobian eigenvalues.

    The state here is the modulation matrix M (post*pre ≈ tens of thousands of
    dims), so the Jacobian J = ∂F/∂M is far too large to form densely. Instead we
    apply an iterative eigensolver to a matrix-free operator: J^T acts on a vector
    v by a single reverse-mode VJP of one `network_step` at M*, and since J and
    J^T share eigenvalues, the leading-|λ| spectrum is obtained with only ~O(k)
    backward passes per fixed point.

    Discrete-map reading of the eigenvalues λ:
      |λ| < 1  contracting direction;  |λ| > 1  expanding (unstable) direction.
      spectral_radius = max|λ|:  < 1 ⇒ attracting fixed point.
      marginal directions (|λ − 1| < marginal_tol) are near-neutral flow. A lone
      marginal eigenvalue with all others < 1 is a necessary ring-attractor
      candidate condition; confirming a ring additionally requires alignment of
      that eigenmode with the manifold tangent.

    network      : trained (Deep)MultiPlasticNet.
    fixed_M      : (B, post, pre) solved fixed points (numpy or tensor).
    inputs       : (B, n_input) constant input each M* was solved under.
    k            : number of leading (largest-magnitude) eigenvalues per point.
    marginal_tol : |λ − 1| threshold counting a direction as marginal/neutral.

    Returns a dict of per-point arrays (all length B):
      eigenvalues     : (B, k) complex, largest |λ| first.
      spectral_radius : (B,) max|λ|.
      n_unstable      : (B,) count of |λ| > 1 + marginal_tol.
      n_marginal      : (B,) count of |λ − 1| < marginal_tol.
      is_strict_stable: (B,) bool, spectral_radius < 1 - marginal_tol.
      is_nonunstable  : (B,) bool, spectral_radius <= 1 + marginal_tol.
      is_stable       : deprecated alias of is_nonunstable, retained for old
                        pickle/caller compatibility.
    """
    from scipy.sparse.linalg import LinearOperator, eigs

    fpn = ModulationFixedPointNetwork(network, fixed_M)
    if device is not None:
        fpn.to(device)
        fpn.states.data = fpn.states.data.to(device)
    dev = fpn.states.device
    inp = torch.as_tensor(np.asarray(inputs), dtype=torch.float, device=dev)

    B, post, pre = fpn.states.shape
    n = post * pre                       # per-point state dimension

    eig_all = np.zeros((B, k), dtype=complex)
    radius = np.zeros(B)
    n_unstable = np.zeros(B, dtype=int)
    n_marginal = np.zeros(B, dtype=int)

    kk = min(k, n - 2)                   # eigs needs k < n-1
    for b in range(B):
        M_b = fpn.states[b:b + 1].detach().clone().requires_grad_(True)  # (1,post,pre)
        x_b = inp[b:b + 1]                                               # (1,n_input)
        # F(M_b) for this single point; graph retained for repeated VJPs.
        F_b = fpn._step_M(x_b, M_b)                                      # (1,post,pre)

        def _matvec(v):
            # J^T v : reverse-mode VJP of F at M_b with cotangent v.
            vt = torch.as_tensor(v.real, dtype=torch.float, device=dev).reshape(1, post, pre)
            (g,) = torch.autograd.grad(F_b, M_b, grad_outputs=vt, retain_graph=True)
            return g.detach().cpu().numpy().reshape(-1)

        JT = LinearOperator((n, n), matvec=_matvec, dtype=float)
        try:
            vals = eigs(JT, k=kk, which="LM", return_eigenvectors=False,
                        maxiter=n * 10)
        except Exception as exc:
            print(f"    [stability] point {b}: eigs failed ({exc}); NaN spectrum.")
            eig_all[b, :] = np.nan
            radius[b] = np.nan
            continue
        vals = vals[np.argsort(-np.abs(vals))]      # largest |λ| first
        eig_all[b, :vals.size] = vals
        mag = np.abs(vals)
        radius[b] = float(mag.max()) if mag.size else np.nan
        n_unstable[b] = int(np.sum(mag > 1.0 + marginal_tol))
        n_marginal[b] = int(np.sum(np.abs(vals - 1.0) < marginal_tol))

    is_strict_stable = radius < (1.0 - marginal_tol)
    is_nonunstable = radius <= (1.0 + marginal_tol)
    return {
        "eigenvalues": eig_all,
        "spectral_radius": radius,
        "n_unstable": n_unstable,
        "n_marginal": n_marginal,
        "is_strict_stable": is_strict_stable,
        "is_nonunstable": is_nonunstable,
        "is_stable": is_nonunstable,
        "marginal_tol": float(marginal_tol),
    }


def find_modulation_fixed_points(network, init_M, inputs, steps=2000,
                                 learningRate=1e-3, printPeriod=200,
                                 lbfgs_steps=500, loss_tol=1e-8, device=None,
                                 rescue_rel_tol_undamped=None,
                                 rescue_lbfgs_steps=200,
                                 return_diagnostics=False):
    """
    Convenience wrapper: build a ModulationFixedPointNetwork seeded at `init_M`
    and optimize it under constant `inputs`.

    network     : trained (Deep)MultiPlasticNet.
    init_M      : (B, post, pre) initial modulation matrices (e.g. recorded M at
                  a period midpoint), one per candidate / stimulus.
    inputs      : (B, n_input) constant per-candidate input held fixed during the
                  relaxation (e.g. the fixation-only input for a delay fixed pt).
    steps       : MAX Adam iterations (first pass); Adam stops early once the
                  speed loss reaches `loss_tol`.
    loss_tol    : Adam early-stop threshold on the leak-normalized MSE residual
                  (default 1e-8).
    lbfgs_steps : L-BFGS polishing iterations after Adam (0 disables); drives the
                  normalized residual far lower than Adam alone.
    rescue_rel_tol_undamped : when set, candidates still above this normalized
                  relative-residual threshold after the batch solve receive an
                  independent L-BFGS pass. Only improvements are retained.
    rescue_lbfgs_steps : maximum iterations for each independent rescue pass.
    return_diagnostics : append a fourth return value describing rescue attempts.
    Returns (fixed_M, loss_hist, final_speeds) with fixed_M as a numpy array, plus
    diagnostics when ``return_diagnostics=True``.
    """
    fpn = ModulationFixedPointNetwork(network, init_M)
    if device is not None:
        fpn.to(device)
        fpn.states.data = fpn.states.data.to(device)
    fixed_M, loss_hist, final_speeds = fpn.find_fixed_points(
        inputs, steps, learningRate=learningRate, printPeriod=printPeriod,
        lbfgs_steps=lbfgs_steps, loss_tol=loss_tol)
    diagnostics = {
        "attempted": np.zeros(fixed_M.shape[0], dtype=bool),
        "accepted": np.zeros(fixed_M.shape[0], dtype=bool),
        "rel_step_undamped_before": fpn.relative_residuals(inputs),
        "rel_step_undamped_after": fpn.relative_residuals(inputs),
        "lbfgs_steps": int(rescue_lbfgs_steps),
    }
    if rescue_rel_tol_undamped is not None and rescue_lbfgs_steps > 0:
        failed = np.flatnonzero(
            diagnostics["rel_step_undamped_before"]
            > float(rescue_rel_tol_undamped))
        if failed.size:
            print(f"  [rescue] independently polishing {failed.size}/"
                  f"{fixed_M.shape[0]} candidate(s) above normalized residual "
                  f"{float(rescue_rel_tol_undamped):g}")
            diagnostics = fpn.polish_candidates_individually(
                inputs, failed, lbfgs_steps=rescue_lbfgs_steps)
            print(f"  [rescue] accepted {int(diagnostics['accepted'].sum())}/"
                  f"{failed.size} lower-residual proposal(s); strict after rescue "
                  f"{int((diagnostics['rel_step_undamped_after'] <= float(rescue_rel_tol_undamped)).sum())}/"
                  f"{fixed_M.shape[0]}")
            fixed_M = fpn.states.detach()
            final_speeds = fpn.get_speeds(inputs)

    result = (fixed_M.cpu().numpy(), loss_hist, final_speeds)
    return (*result, diagnostics) if return_diagnostics else result


# ═════════════════════════════════════════════════════════════════════════════
# Hidden-state fixed points (vanilla RNN / GRU)
# ═════════════════════════════════════════════════════════════════════════════
# The MPN's state is the modulation matrix M, so everything above solves
# M* = F(M*; x). A vanilla RNN has no M — its state is the HIDDEN VECTOR, and the
# same question becomes h* = F(h*; x) with
#     F(h; x) = alpha*h + (1-alpha)*act(W_rec h + W_input x + b_input + b_hidden)
# (the leaky update in networks.VanillaRNN.forward; alpha = 0 for a non-leaky
# net). The classes below mirror the modulation ones one-for-one so the two
# analyses can be read side by side.


class HiddenFixedPointNetwork(nn.Module):
    """
    Finds fixed points of an RNN's HIDDEN state under a constant input.

    The trained network's weights are frozen; the trainable parameters are a
    batch of candidate hidden vectors (`self.states`, shape (B, n_hidden)). One
    forward pass applies a single network update under the constant input and
    returns the resulting hidden state; minimizing MSE(F(h), h) drives the batch
    toward fixed points. Direct analog of `ModulationFixedPointNetwork`.
    """

    def __init__(self, network, init_states):
        """
        network     : a trained VanillaRNN / GRU (its parameters are frozen).
        init_states : (B, n_hidden) initial candidate hidden vectors, e.g. the
                      recorded hidden state at the end of a trial period.
        """
        super().__init__()
        self.eval()
        self.name = self.__class__.__name__

        # `network.hidden` is a live tensor from earlier forward passes and may
        # carry a graph, which deepcopy cannot follow. Swap it for a detached
        # leaf while copying, then put the caller's tensor back untouched — the
        # same dance ModulationFixedPointNetwork does for mp.M.
        stashed = getattr(network, "hidden", None)
        if isinstance(stashed, torch.Tensor):
            network.hidden = stashed.detach().clone()
        try:
            net_fzn = copy.deepcopy(network)
        finally:
            if isinstance(stashed, torch.Tensor):
                network.hidden = stashed

        for param in net_fzn.parameters():
            param.requires_grad = False
        net_fzn.eval()
        self.net = net_fzn

        init = torch.as_tensor(np.asarray(init_states), dtype=torch.float)
        assert init.dim() == 2, (
            f"init_states must be (B, n_hidden); got shape {tuple(init.shape)}")
        self.states = nn.Parameter(init.clone())

        print("FP Network - NetType: {}, States (h) size: {}".format(
            type(net_fzn).__name__, tuple(self.states.shape)))

    # ── One update step of h under a constant input ──────────────────────────
    def _step_h(self, inputs, current_states):
        """
        Return F(h; x): the hidden state after ONE network step, starting from
        `current_states` (B, n_hidden) under constant input `inputs`
        (B, n_input). The layer's stored hidden state is restored afterward so
        repeated calls are side-effect free.

        Calls `forward` rather than `network_step`: the latter also assigns
        self.hidden and may call state_detach() (tbptt), which would cut the
        gradient path back to `self.states`.
        """
        saved_hidden = self.net.hidden
        # Assign a TENSOR, not the nn.Parameter itself (see the note in
        # ModulationFixedPointNetwork._step_M); the *1.0 keeps it grad-tracking.
        self.net.hidden = current_states * 1.0
        _, next_h, _ = self.net.forward(inputs, run_mode="minimal")
        self.net.hidden = saved_hidden
        return next_h

    def forward(self, inputs, current_states=None):
        """One-step update of h. Uses the optimized `self.states` by default, or
        `current_states` if provided. Returns the next hidden state (B, n_hidden)."""
        states = self.states if current_states is None else current_states
        return self._step_h(inputs, states)

    # ── Speeds q(h) = 1/2 ||F(h) - h||^2 ─────────────────────────────────────
    def get_speeds(self, inputs, current_states=None):
        """Per-point speed q(h) for the batch (numpy, shape (B,)); the norm runs
        over the hidden dimension."""
        with torch.no_grad():
            ref = self.states if current_states is None else current_states
            next_state = self(inputs, current_states=current_states)
            return (0.5 * torch.norm(next_state - ref, dim=1) ** 2).cpu().numpy()

    def _speed_loss(self, inputs):
        """MSE(F(h), h): minimizing it drives the total speed q(h) to zero."""
        next_state = self(inputs)
        return F.mse_loss(next_state, self.states, reduction="mean")

    def find_fixed_points(self, inputs, steps, learningRate=1e-3, printPeriod=200,
                          lbfgs_steps=500, loss_tol=1e-8):
        """
        Descend the candidate hidden states toward fixed points under the
        constant input `inputs` (B, n_input). Two stages, identical in spirit to
        `ModulationFixedPointNetwork.find_fixed_points`:
          1. Adam until the MSE speed loss reaches `loss_tol`, capped at `steps`
             (so `steps` is a MAX-iteration cap, not a fixed count);
          2. L-BFGS polishing (strong-Wolfe), the Sussillo & Barak refinement,
             which drives q(h*) orders of magnitude below what Adam reaches.

        Returns (states, loss_hist, final_speeds).
        """
        inputs = torch.as_tensor(np.asarray(inputs), dtype=torch.float,
                                 device=self.states.device)

        init_speeds = self.get_speeds(inputs)
        print("Init speeds - Max: {:.2e} / Min: {:.2e}".format(
            float(np.max(init_speeds)), float(np.min(init_speeds))))

        self.optimizer = torch.optim.Adam([self.states], lr=learningRate)
        loss_hist = []
        last_loss = float("inf")
        for step in range(steps):
            self.optimizer.zero_grad()
            loss = self._speed_loss(inputs)
            loss_val = loss.item()
            loss_hist.append(loss_val)
            last_loss = loss_val
            loss.backward()
            self.optimizer.step()
            if step % printPeriod == 0:
                print("  [adam] Step {} - Loss: {:.3e}".format(step, loss_val))
            if loss_tol and loss_val <= loss_tol:
                print("  [adam] converged: Step {} - Loss: {:.3e} "
                      "(<= tol {:.1e})".format(step, loss_val, loss_tol))
                break
        else:
            if loss_tol:
                print("  [adam] hit max steps ({}) without reaching tol {:.1e}; "
                      "last loss {:.3e}".format(steps, loss_tol, last_loss))

        if lbfgs_steps and lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                [self.states], max_iter=int(lbfgs_steps), lr=1.0,
                tolerance_grad=1e-16, tolerance_change=1e-18,
                history_size=50, line_search_fn="strong_wolfe")

            def _closure():
                lbfgs.zero_grad()
                loss = self._speed_loss(inputs)
                loss.backward()
                return loss

            lbfgs.step(_closure)

        final_speeds = self.get_speeds(inputs)
        print("Final speeds - Max: {:.2e} / Min: {:.2e}".format(
            float(np.max(final_speeds)), float(np.min(final_speeds))))

        return self.states.detach(), loss_hist, final_speeds


def find_hidden_fixed_points(network, init_h, inputs, steps=2000,
                             learningRate=1e-3, printPeriod=200,
                             lbfgs_steps=500, loss_tol=1e-8, device=None):
    """
    Convenience wrapper: build a HiddenFixedPointNetwork seeded at `init_h` and
    optimize it under constant `inputs`. The hidden-state counterpart of
    `find_modulation_fixed_points`.

    network : trained VanillaRNN / GRU.
    init_h  : (B, n_hidden) initial hidden vectors (e.g. the recorded state at a
              period's end), one per candidate / stimulus.
    inputs  : (B, n_input) constant per-candidate input held fixed during the
              relaxation (e.g. the delay-period input for a memory fixed point).
    Returns (fixed_h, loss_hist, final_speeds) with fixed_h as a numpy array.
    """
    fpn = HiddenFixedPointNetwork(network, init_h)
    if device is not None:
        fpn.to(device)
        fpn.states.data = fpn.states.data.to(device)
    fixed_h, loss_hist, final_speeds = fpn.find_fixed_points(
        inputs, steps, learningRate=learningRate, printPeriod=printPeriod,
        lbfgs_steps=lbfgs_steps, loss_tol=loss_tol)
    return fixed_h.cpu().numpy(), loss_hist, final_speeds


def characterize_hidden_fixed_point_stability(network, fixed_h, inputs,
                                              k=16, marginal_tol=5e-2,
                                              device=None):
    """
    Linear-stability analysis of hidden-state fixed points (Sussillo & Barak
    2013): linearize F(h; x) about each h* and read stability off the Jacobian
    eigenvalues.

    Unlike the modulation case — where the state has post*pre ≈ 10^4-10^5 dims
    and the Jacobian must be handled matrix-free — the hidden state is only
    n_hidden (a few hundred) dims, so J is formed DENSELY here and the FULL
    spectrum is computed exactly. No ARPACK, no k < n-1 restriction; `k` only
    decides how many of the leading eigenvalues are reported back.

    Discrete-map reading of the eigenvalues λ:
      |λ| < 1 contracting, |λ| > 1 expanding; spectral_radius = max|λ| < 1 ⇒ an
      attracting fixed point. A lone marginal direction (|λ − 1| < marginal_tol)
      with everything else contracting is a ring-attractor candidate; tangent
      alignment is still required to establish the manifold interpretation.

    Returns the same dict of per-point arrays as
    `characterize_fixed_point_stability`, so downstream code can treat the two
    interchangeably.
    """
    fpn = HiddenFixedPointNetwork(network, fixed_h)
    if device is not None:
        fpn.to(device)
        fpn.states.data = fpn.states.data.to(device)
    dev = fpn.states.device
    inp = torch.as_tensor(np.asarray(inputs), dtype=torch.float, device=dev)

    B, n = fpn.states.shape
    k_report = min(k, n)

    eig_all = np.zeros((B, k_report), dtype=complex)
    radius = np.zeros(B)
    n_unstable = np.zeros(B, dtype=int)
    n_marginal = np.zeros(B, dtype=int)

    for b in range(B):
        h_b = fpn.states[b:b + 1].detach().clone().requires_grad_(True)  # (1, n)
        x_b = inp[b:b + 1]                                               # (1, n_input)
        F_b = fpn._step_h(x_b, h_b)                                      # (1, n)
        # Dense Jacobian, one reverse-mode pass per output coordinate. n is a few
        # hundred, so this is cheap and exact.
        J = np.zeros((n, n))
        for i in range(n):
            seed = torch.zeros_like(F_b)
            seed[0, i] = 1.0
            (g,) = torch.autograd.grad(F_b, h_b, grad_outputs=seed,
                                       retain_graph=True)
            J[i, :] = g.detach().cpu().numpy().reshape(-1)
        vals = np.linalg.eigvals(J)
        vals = vals[np.argsort(-np.abs(vals))]      # largest |λ| first
        eig_all[b, :] = vals[:k_report]
        mag = np.abs(vals)
        radius[b] = float(mag.max())
        n_unstable[b] = int(np.sum(mag > 1.0 + marginal_tol))
        n_marginal[b] = int(np.sum(np.abs(vals - 1.0) < marginal_tol))

    is_strict_stable = radius < (1.0 - marginal_tol)
    is_nonunstable = radius <= (1.0 + marginal_tol)
    return {
        "eigenvalues": eig_all,
        "spectral_radius": radius,
        "n_unstable": n_unstable,
        "n_marginal": n_marginal,
        "is_strict_stable": is_strict_stable,
        "is_nonunstable": is_nonunstable,
        "is_stable": is_nonunstable,
        "marginal_tol": float(marginal_tol),
    }
