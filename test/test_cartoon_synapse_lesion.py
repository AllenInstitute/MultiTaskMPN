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

    def test_synapse_clusters_interleave_across_neuron_pairs(self):
        """Every drawn (pre, post) pair has a cluster, and each cluster touches
        every post neuron: the lesioned set is not any neuron's fan-in."""
        n = cartoon._LESION_N_NODES
        edges = cartoon._LESION_EDGE_CLUSTERS
        self.assertEqual(set(edges), {(pre, post) for pre in range(n) for post in range(n)})
        for cluster in (0, 1):
            posts = {post for (pre, post), c in edges.items() if c == cluster}
            self.assertEqual(posts, set(range(n)))

    def test_mode_traces_differ_only_in_time(self):
        t_ms = np.linspace(0.0, 3000.0, cartoon.N_SAMPLES)
        m = cartoon._modulation_trace(np.random.RandomState(0), t_ms)
        self.assertLess(abs(m[0]), 0.1)            # M starts near its initial value
        self.assertGreater(m[-1], 0.3)             # ... and has grown by the end
        figure, axis = cartoon.plt.subplots()
        self.addCleanup(cartoon.plt.close, figure)
        intact, zero, frozen = cartoon._lesion_mode_traces(axis, t_ms, 0.8, m, "k")
        np.testing.assert_allclose(intact, 0.8 * (1 + m))
        np.testing.assert_array_equal(zero, 0.0)
        np.testing.assert_array_equal(frozen, 0.8)
        self.assertGreater(np.ptp(intact), 0.2)
        self.assertEqual(len(axis.lines), 3)
        labels = [text.get_text() for text in axis.texts]
        self.assertEqual(len(labels), 3)
        self.assertTrue(any("W \\leftarrow 0" in label for label in labels))
        self.assertTrue(any("M(t) \\leftarrow M(0)" in label for label in labels))
        self.assertFalse(any("zero_W" in label or "freeze_M" in label for label in labels))
        self.assertEqual([tick.get_text() for tick in axis.get_xticklabels()], ["0", "T"])
        self.assertEqual(axis.get_ylabel(), r"$W_{\mathrm{eff}}$")
        # Text sizes follow paper_plot: 7 pt ticks, 8 pt axis labels.
        self.assertTrue(all(tick.get_fontsize() == 7 for tick in axis.get_xticklabels()))
        self.assertTrue(all(tick.get_fontsize() == 7 for tick in axis.get_yticklabels()))
        self.assertEqual(axis.yaxis.label.get_fontsize(), 8)
        self.assertEqual(axis.xaxis.label.get_fontsize(), 8)

    def test_figure_shows_a_neuron_lesion_a_synapse_lesion_and_the_traces(self):
        figure, path, extra = self.render()
        self.assertEqual(path.name, "cartoon_synapse_lesion_schematic.png")
        self.assertIn("SYNTHETIC", extra)
        np.testing.assert_allclose(figure.get_size_inches(), cartoon._LESION_FIGSIZE)
        self.assertEqual(len(figure.axes), 3)
        ax_neuron, ax_synapse, ax_trace = figure.axes
        self.assertEqual(ax_neuron.get_title(), "Neuron-cluster lesion")
        self.assertEqual(ax_synapse.get_title(), "Synapse-cluster lesion")
        self.assertEqual(ax_trace.get_title(), "Synapse lesion: two modes")
        self.assertTrue(all(axis.title.get_fontsize() == 8 for axis in figure.axes))
        n = cartoon._LESION_N_NODES
        n_edges = len(cartoon._LESION_EDGE_CLUSTERS)
        red = cartoon.to_rgb(cartoon._LESION_CROSS_COLOR)
        for axis in (ax_neuron, ax_synapse):
            self.assertEqual(len(axis.patches), 2 * n)          # the neurons
        # The post neurons of one cluster are crossed out and their synapses are
        # faint; no edge carries a cross of its own.
        n_lesioned_neurons = len(cartoon._LESION_LESIONED_NEURONS)
        self.assertGreater(n_lesioned_neurons, 1)          # a cluster, not one neuron
        self.assertLess(n_lesioned_neurons, n)             # ... and not the whole layer
        crosses = [line for line in ax_neuron.lines if cartoon.to_rgb(line.get_color()) == red]
        self.assertEqual(len(crosses), 2 * n_lesioned_neurons)
        faint = [line for line in ax_neuron.lines
                 if cartoon.to_rgb(line.get_color()) != red and line.get_alpha() < 0.5]
        self.assertEqual(len(faint), n * n_lesioned_neurons)
        # Every edge of the lesioned synapse cluster is crossed, no neuron is.
        n_lesioned = sum(c == cartoon._LESION_LESIONED_SYNAPSE_CLUSTER
                         for c in cartoon._LESION_EDGE_CLUSTERS.values())
        crosses = [line for line in ax_synapse.lines if cartoon.to_rgb(line.get_color()) == red]
        self.assertEqual(len(crosses), 2 * n_lesioned)
        edges = [line for line in ax_synapse.lines if cartoon.to_rgb(line.get_color()) != red]
        self.assertEqual(len(edges), n_edges)
        self.assertTrue(all(line.get_alpha() > 0.5 for line in edges))
        self.assertEqual(len(ax_trace.lines), 3)

    def test_same_seed_reproduces_the_trace(self):
        t_ms = np.linspace(0.0, 3000.0, cartoon.N_SAMPLES)
        first = cartoon._modulation_trace(np.random.RandomState(7), t_ms)
        second = cartoon._modulation_trace(np.random.RandomState(7), t_ms)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
