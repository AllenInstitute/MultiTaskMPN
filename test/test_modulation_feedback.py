"""Tests for the plastic layer's feedback and saturation analysis."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401
import numpy as np
import torch

import modulation_feedback as feedback
import mpn

LAM, ETA = 0.99, 1.7


def _toy_state(seed=0, n_trials=2, n_post=5, n_pre=4):
    """Random h, x, W and an unclipped mask with some clipped synapses, in double."""
    generator = torch.Generator().manual_seed(seed)
    h = torch.tanh(torch.randn(n_trials, n_post, generator=generator, dtype=torch.float64))
    x = torch.randn(n_trials, n_pre, generator=generator, dtype=torch.float64)
    W = torch.randn(n_post, n_pre, generator=generator, dtype=torch.float64)
    unclipped = torch.rand(n_trials, n_post, n_pre, generator=generator) > 0.3
    return h, x, W, unclipped


def _tiny_network(eta=None):
    """A 4-input, 5-embedding, 6-hidden multiplicative MPN with lam = 0.99."""
    net_params = {
        "net_type": "dmpn", "n_neurons": [4, 6, 3], "linear_embed": 5, "activation": "tanh",
        "output_matrix": "", "input_layer_add": True, "input_layer_add_trainable": True,
        "input_layer_bias": False, "output_bias": False, "loss_type": "MSE",
        "acc_measure": "angle", "dt": 40,
        "ml_params": {"bias": True, "mp_type": "mult", "m_update_type": "hebb_assoc",
                      "eta_type": "scalar", "lam_type": "scalar", "m_time_scale": 4000,
                      "eta_train": True, "lam_train": False, "W_freeze": False},
    }
    torch.manual_seed(0)
    model = mpn.DeepMultiPlasticNet(net_params, verbose=False, forzihan=True)
    if eta is not None:
        model.mp_layers[0].eta.data.fill_(eta)
    return model.eval()


class ReadableModeRateTests(unittest.TestCase):
    def test_mu_is_the_one_eigenvalue_feedback_moves(self):
        h, x, W, unclipped = _toy_state()
        mu = feedback.readable_mode_rate(h, x, W, unclipped, LAM, ETA)
        for trial in range(h.shape[0]):
            for post in range(h.shape[1]):
                kept = unclipped[trial, post].numpy()
                u = ETA * (1 - h[trial, post].item() ** 2) * x[trial].numpy()
                v = (W[post] * x[trial]).numpy()
                jacobian = np.diag(kept.astype(float)) @ (LAM * np.eye(len(v)) + np.outer(u, v))
                eigenvalues = np.linalg.eigvals(jacobian).real
                self.assertTrue(np.isclose(eigenvalues, mu[trial, post].item()).any())
                self.assertEqual(np.isclose(eigenvalues, LAM).sum(), kept.sum() - 1)
                self.assertEqual(np.isclose(eigenvalues, 0.0).sum(), (~kept).sum())

    def test_without_a_mask_every_synapse_counts(self):
        h, x, W, _ = _toy_state()
        everything = torch.ones(W.shape, dtype=torch.bool).expand(h.shape[0], *W.shape)
        torch.testing.assert_close(feedback.readable_mode_rate(h, x, W, None, LAM, ETA),
                                   feedback.readable_mode_rate(h, x, W, everything, LAM, ETA))


class LinearResponseTests(unittest.TestCase):
    def test_matches_the_derivative_of_the_update(self):
        generator = torch.Generator().manual_seed(1)
        n_post, n_pre = 5, 4
        W = torch.randn(n_post, n_pre, generator=generator, dtype=torch.float64)
        b = torch.randn(n_post, generator=generator, dtype=torch.float64)
        x = torch.randn(n_pre, generator=generator, dtype=torch.float64)
        # Entries near +-1 make some updates clip.
        M = 0.97 * torch.tanh(2 * torch.randn(n_post, n_pre, generator=generator,
                                              dtype=torch.float64))
        delta = torch.randn(n_post, n_pre, generator=generator, dtype=torch.float64)

        def update(M):
            h = torch.tanh((W * (1 + M)) @ x + b)
            return torch.clamp(LAM * M + ETA * torch.outer(h, x), -1.0, 1.0)

        _, derivative = torch.autograd.functional.jvp(update, M, delta)
        h = torch.tanh((W * (1 + M)) @ x + b)
        M_pre = LAM * M + ETA * torch.outer(h, x)
        unclipped = (M_pre.abs() < 1.0).unsqueeze(0)
        self.assertFalse(unclipped.all())
        predicted = feedback.linear_response(delta.unsqueeze(0), h.unsqueeze(0), x.unsqueeze(0),
                                             W, unclipped, LAM, ETA)[0]
        torch.testing.assert_close(predicted, derivative)

    def test_the_probes_are_the_predicted_modes(self):
        h, x, W, unclipped = _toy_state(seed=2)
        mu = feedback.readable_mode_rate(h, x, W, unclipped, LAM, ETA)
        probes = feedback.linearization_probes(x, W, unclipped, torch.Generator().manual_seed(0))
        for probe in probes.values():
            norms = probe.norm(dim=-1)
            torch.testing.assert_close(norms[norms > 0], torch.ones_like(norms[norms > 0]))
        for name in ("readable", "unreadable"):
            self.assertTrue((probes[name][~unclipped] == 0).all())

        def respond(probe):
            return feedback.linear_response(probe, h, x, W, unclipped, LAM, ETA)

        torch.testing.assert_close(respond(probes["readable"]),
                                   mu.unsqueeze(-1) * probes["readable"])
        torch.testing.assert_close(respond(probes["unreadable"]), LAM * probes["unreadable"])


class NetworkTests(unittest.TestCase):
    def test_the_real_step_follows_the_derivation_with_clipping(self):
        model = _tiny_network(eta=8.0)
        inputs = 2.0 * torch.randn(3, 12, 4, generator=torch.Generator().manual_seed(0))
        record, outputs = feedback.measure(model, inputs, chunk=2)
        self.assertEqual(record["mu"].shape, (3, 12, 6))
        self.assertEqual(outputs.shape, (3, 12, 3))
        self.assertGreater((record["above"] + record["below"]).max(), 0.0)
        errors = feedback.check_linearization(model, inputs, n_steps=5)
        self.assertEqual(set(errors), {"random", "readable", "unreadable"})
        for error in errors.values():
            self.assertLess(error["max"], 1e-5)

    def test_rejects_a_layer_the_derivation_does_not_cover(self):
        model = _tiny_network()
        model.mp_layers[0].mp_type = "add"
        with self.assertRaises(ValueError):
            feedback.plastic_layer(model)


class SummaryTests(unittest.TestCase):
    def test_time_constant_uses_the_magnitude_of_mu(self):
        tau = feedback.readable_tau([0.5, -0.5, 0.99, 1.2, -1.5], 40.0)
        np.testing.assert_allclose(tau, [80.0, 80.0, 4000.0, np.inf, np.inf])

    def test_stats_leave_out_rows_without_a_mode(self):
        stats = feedback.mu_stats([0.5, 0.995, 1.2, -1.5, np.nan], LAM, 40.0)
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["share_above_one"], 0.25)
        self.assertEqual(stats["share_below_minus_one"], 0.25)
        self.assertEqual(stats["share_below_lam"], 0.5)

    def test_epoch_bounds_fill_the_open_ends(self):
        trial = SimpleNamespace(tdim=np.array([30, 30]), epochs={
            "go1": (np.array([10, 10]), None), "fix1": (None, 5), "stim1": (5, 10)})
        bounds = feedback.epoch_bounds(trial)
        self.assertEqual(bounds, {"fix1": (0, 5), "stim1": (5, 10), "go1": (10, 30)})
        self.assertEqual(list(bounds), ["fix1", "stim1", "go1"])
        trial.epochs["go1"] = (np.array([10, 11]), None)
        with self.assertRaises(ValueError):
            feedback.epoch_bounds(trial)


class NamingTests(unittest.TestCase):
    def test_result_names(self):
        aname = "everything_seed749_L21e4+hidden300+batch128+angle"
        self.assertEqual(feedback.result_stem(749, "L21e4"), f"modulation_feedback_{aname}")
        self.assertEqual(feedback.result_stem(749, "L21e4", 3),
                         f"modulation_feedback_{aname}_init3")

    def test_discovers_the_feature_s_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("everything_seed21_L21e4", "everything_seed7_L21e4",
                         "everything_seed5_L21e3"):
                (Path(directory) / f"savednet_{name}+hidden300+batch128+angle.pt").touch()
            self.assertEqual(feedback.discover_seeds("L21e4", directory), [7, 21])


if __name__ == "__main__":
    unittest.main()
