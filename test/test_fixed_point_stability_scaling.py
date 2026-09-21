"""Unit tests for leak-normalized modulation fixed-point diagnostics."""

import unittest
from pathlib import Path
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401
from core.fixed_point import (
    ModulationFixedPointNetwork,
    find_modulation_fixed_points,
)
from core.grad_fixed_points import (
    classify_spectral_radius,
    convergence_from_rel_step,
    convergence_quality_masks,
    raw_tolerance,
    spectral_radius_timescale,
)
import paper_plot
from one_task import one_task_analysis


class _LossHarness(nn.Module):
    """Minimal object that exercises the production modulation loss method."""

    _speed_loss = ModulationFixedPointNetwork._speed_loss

    def __init__(self, residual, leak):
        super().__init__()
        self.states = nn.Parameter(torch.zeros_like(residual))
        self.register_buffer("next_state", residual)
        self.register_buffer("residual_leak", leak)

    def forward(self, inputs, current_states=None):
        states = self.states if current_states is None else current_states
        return states + self.next_state


class _LinearModulationLayer(nn.Module):
    """Tiny layer exposing the attributes required by the fixed-point wrapper."""

    def __init__(self):
        super().__init__()
        self.lam = torch.tensor(0.5)
        self.lam_type = "scalar"
        self.M = torch.zeros(1, 1, 2)
        self.M_pre = torch.zeros(1, 1, 2)

    def build_M_parameter(self, value, value_type):
        return value


class _LinearModulationNet(nn.Module):
    """Map F(M; x) = 0.5 M + 0.5 x, whose fixed point is exactly x."""

    def __init__(self):
        super().__init__()
        self.mp_layers = nn.ModuleList([_LinearModulationLayer()])

    def network_step(self, inputs, run_mode="minimal"):
        layer = self.mp_layers[0]
        inputs = torch.as_tensor(inputs, dtype=layer.M.dtype,
                                 device=layer.M.device)
        target = inputs.reshape(inputs.shape[0], 1, 2)
        layer.M = 0.5 * layer.M + 0.5 * target


class FixedPointStabilityScalingTests(unittest.TestCase):
    def test_optimizer_loss_removes_leak_scale(self):
        residual = torch.full((2, 3, 4), 0.01)
        leak = torch.full((3, 4), 0.01)
        model = _LossHarness(residual, leak)
        self.assertAlmostEqual(
            float(model._speed_loss(None).detach()), 1.0, places=6)

    def test_convergence_uses_normalized_residual(self):
        normalized, mask, raw_limit = convergence_from_rel_step(
            np.array([5e-5, 1e-4, 2e-4]), leak=0.01,
            rel_tol_undamped=0.01)
        np.testing.assert_allclose(normalized, [0.005, 0.01, 0.02])
        np.testing.assert_array_equal(mask, [True, True, False])
        self.assertAlmostEqual(raw_limit, 1e-4)

    def test_convergence_quality_separates_approximate_candidates(self):
        strict, approximate, unconverged = convergence_quality_masks(
            np.array([0.005, 0.01, 0.02, 0.05, 0.051, np.nan]))
        np.testing.assert_array_equal(
            strict, [True, True, False, False, False, False])
        np.testing.assert_array_equal(
            approximate, [False, False, True, True, False, False])
        np.testing.assert_array_equal(
            unconverged, [False, False, False, False, True, True])

    def test_individual_rescue_polishes_only_failed_candidates(self):
        inputs = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
        initial = np.zeros((2, 1, 2), dtype=np.float32)
        initial[0, 0] = inputs[0]  # already exact; rescue must leave it alone
        fixed, _, _, rescue = find_modulation_fixed_points(
            _LinearModulationNet(), initial,
            inputs, steps=0, lbfgs_steps=0, loss_tol=0,
            rescue_rel_tol_undamped=1e-5, rescue_lbfgs_steps=50,
            return_diagnostics=True)
        np.testing.assert_array_equal(rescue["attempted"], [False, True])
        np.testing.assert_array_equal(rescue["accepted"], [False, True])
        self.assertEqual(rescue["rel_step_undamped_after"][0], 0.0)
        self.assertLess(rescue["rel_step_undamped_after"][1],
                        rescue["rel_step_undamped_before"][1])
        np.testing.assert_allclose(fixed.reshape(2, 2), inputs, atol=1e-5)

    def test_paper_upgrades_legacy_classification_counts(self):
        counts = paper_plot._fixed_point_classification_counts({
            "counts": {"stable": 1, "marginal": 0, "unstable": 1,
                       "unconverged": 4},
            "rel_step_undamped": np.array(
                [0.005, 0.008, 0.02, 0.05, 0.051, np.nan]),
            "rel_tol_undamped": 0.01,
        })
        self.assertEqual(counts["approximate"], 2)
        self.assertEqual(counts["unconverged"], 2)

    def test_marginal_band_scales_with_leak(self):
        tol = raw_tolerance(0.05, leak=0.01)
        self.assertAlmostEqual(tol, 5e-4)
        radii = np.array([0.9990, 0.9996, 1.0004, 1.0010, np.nan])
        np.testing.assert_array_equal(
            classify_spectral_radius(radii, tol), [0, 1, 1, 2, -1])

    def test_growth_rate_and_timescale(self):
        radii = np.array([np.exp(-0.04), 1.0, np.exp(0.04)])
        rate, timescale = spectral_radius_timescale(radii, dt_ms=40)
        np.testing.assert_allclose(rate, [-1.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(timescale[[0, 2]], [1000.0, 1000.0])
        self.assertTrue(np.isinf(timescale[1]))

    def test_paper_reinterprets_legacy_modulation_pickle(self):
        entry = {
            "rel_step_undamped": np.array([0.005, 0.02]),
            "is_fixed": np.array([True, True]),
            "leak": 0.01,
            "marginal_tol": 0.05,
        }
        np.testing.assert_array_equal(
            paper_plot._fixed_point_mask(entry, 2), [True, False])
        self.assertAlmostEqual(paper_plot._fixed_point_marginal_tol(entry), 5e-4)

    def test_one_task_stability_plot_excludes_unconverged_candidates(self):
        entry = {
            "period_title": "Delay",
            "stim": np.arange(3),
            "spectral_radius": np.array([0.8, 1.1, 1.2]),
            "eigenvalues": np.array([
                [0.8 + 0j, 0.5 + 0j],
                [1.1 + 0j, 0.4 + 0j],
                [1.2 + 0j, 0.3 + 0j],
            ]),
            "is_fixed_strict": np.array([True, False, True]),
            "marginal_tol_undamped": 0.01,
        }
        data = {
            "angles": np.array([0.0, 1.0, 2.0]),
            "results": {"longdelay": entry},
        }
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot._render_fixed_point_stability(data, "unused.png")
        figure = save.call_args.args[0]
        try:
            self.assertEqual(len(figure.axes[0].collections), 2)
            np.testing.assert_allclose(
                figure.axes[1].lines[0].get_ydata(), [0.8, 1.2])
        finally:
            paper_plot.plt.close(figure)

    def test_one_task_classifier_reinterprets_legacy_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {
                "period_title": "Delay",
                "stim": np.arange(3),
                "spectral_radius": np.array([0.9990, 1.0002, 1.0010]),
                "eigenvalues": np.array([
                    [0.9990 + 0j, 0.9 + 0j],
                    [1.0002 + 0j, 0.9 + 0j],
                    [1.0010 + 0j, 0.9 + 0j],
                ]),
                "rel_step_undamped": np.array([0.005, 0.005, 0.02]),
                # Legacy values that must not override the corrected defaults.
                "is_fixed": np.ones(3, dtype=bool),
                "marginal_tol": 0.05,
                "leak": 0.01,
            }
            source = root / "fixed_points_grad_run.pkl"
            source.write_bytes(pickle.dumps({
                "aname": "run", "leak": 0.01,
                "angles": np.arange(3, dtype=float),
                "results": {"longdelay": entry},
            }))
            one_task_analysis.classify_fixed_point_stability("run", root)
            with (root / "fixed_point_classification_run.pkl").open("rb") as handle:
                result = pickle.load(handle)["per_period"]["longdelay"]
            self.assertEqual(
                result["counts"],
                {"stable": 1, "marginal": 1, "unstable": 0,
                 "approximate": 1, "unconverged": 0})
            np.testing.assert_array_equal(
                result["is_approximate"], [False, False, True])
            np.testing.assert_array_equal(
                result["ring_spectrum_candidate"], [False, True, False])


if __name__ == "__main__":
    unittest.main()
