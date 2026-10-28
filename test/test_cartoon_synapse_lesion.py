"""The lesion schematic in cartoon.py draws the procedure it explains."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import _bootstrap  # noqa: F401
import cartoon


class LesionSchematicTests(unittest.TestCase):
    def render(self, seed=0):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cartoon, "_save_fig") as save:
            cartoon.plot_synapse_lesion_schematic(out_dir=directory, seed=seed)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(cartoon.plt.close, figure)
        return figure, Path(path), save.call_args.kwargs["extra"]

    def test_frozen_trace_differs_from_intact_only_in_time(self):
        t_ms = np.linspace(0.0, 3000.0, cartoon.N_SAMPLES)
        m = cartoon._modulation_trace(np.random.RandomState(0), t_ms)
        self.assertLess(abs(m[0]), 0.1)            # M starts near its initial value
        self.assertGreater(m[-1], 0.3)             # ... and has grown by the end
        figure, axis = cartoon.plt.subplots()
        self.addCleanup(cartoon.plt.close, figure)
        intact, frozen = cartoon._freeze_m_traces(axis, t_ms, 0.8, m, "k")
        np.testing.assert_allclose(intact, 0.8 * (1 + m))
        np.testing.assert_array_equal(frozen, 0.8)
        self.assertGreater(np.ptp(intact), 0.2)
        self.assertEqual(len(axis.lines), 2)
        labels = [text.get_text() for text in axis.texts]
        self.assertEqual(len(labels), 2)
        self.assertFalse(any("W \\leftarrow 0" in label for label in labels))
        self.assertTrue(any("M(t) \\leftarrow M(0)" in label for label in labels))
        self.assertFalse(any("zero_W" in label or "freeze_M" in label for label in labels))
        self.assertEqual([tick.get_text() for tick in axis.get_xticklabels()], ["0", "T"])
        self.assertEqual(axis.get_ylabel(), r"$W_{\mathrm{eff}}$")
        # Text sizes follow paper_plot: 7 pt ticks, 8 pt axis labels.
        self.assertTrue(all(tick.get_fontsize() == 7 for tick in axis.get_xticklabels()))
        self.assertTrue(all(tick.get_fontsize() == 7 for tick in axis.get_yticklabels()))
        self.assertEqual(axis.yaxis.label.get_fontsize(), 8)
        self.assertEqual(axis.xaxis.label.get_fontsize(), 8)

    def test_y_axis_spans_the_traces_not_zero(self):
        """W is the only y tick and the axis hugs the two traces, so no empty
        band runs from zero up to W."""
        t_ms = np.linspace(0.0, 3000.0, cartoon.N_SAMPLES)
        m = cartoon._modulation_trace(np.random.RandomState(0), t_ms)
        figure, axis = cartoon.plt.subplots()
        self.addCleanup(cartoon.plt.close, figure)
        intact, _ = cartoon._freeze_m_traces(axis, t_ms, 0.8, m, "k")
        self.assertEqual([tick.get_text() for tick in axis.get_yticklabels()], ["W"])
        np.testing.assert_allclose(axis.get_yticks(), [0.8])
        low, high = axis.get_ylim()
        self.assertGreater(low, 0.0)
        self.assertLessEqual(low, min(intact.min(), 0.8))
        self.assertGreaterEqual(high, intact.max())
        span = high - low
        self.assertLess(0.8 - low, 0.25 * span)    # W sits near the bottom

    def test_figure_is_the_single_trace_panel(self):
        figure, path, extra = self.render()
        self.assertEqual(path.name, "cartoon_synapse_lesion_schematic.png")
        self.assertIn("SYNTHETIC", extra)
        np.testing.assert_allclose(figure.get_size_inches(), cartoon._LESION_FIGSIZE)
        self.assertEqual(len(figure.axes), 1)
        (axis,) = figure.axes
        self.assertEqual(axis.get_title(), "Synapse lesion: frozen M")
        self.assertEqual(axis.title.get_fontsize(), 8)
        self.assertEqual(len(axis.lines), 2)
        self.assertEqual(len(axis.patches), 0)          # no wiring diagram

    def test_same_seed_reproduces_the_trace(self):
        t_ms = np.linspace(0.0, 3000.0, cartoon.N_SAMPLES)
        first = cartoon._modulation_trace(np.random.RandomState(7), t_ms)
        second = cartoon._modulation_trace(np.random.RandomState(7), t_ms)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
