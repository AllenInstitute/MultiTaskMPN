"""Small cached-data checks for the synapse-cluster enrichment example figures."""

import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


class OvermembershipExampleTests(unittest.TestCase):
    def cache(self):
        # Input cluster C1 (row 0) has too few surviving synapses in every block
        # (expected 0, 0.5 / 4 and 0.3 / 2.4 for shares 0.1 / 0.8), so it is
        # dropped; every other block expects 10 or 80 synapses and is shown.
        # Cluster C3 (index 1) peaks at OM 8, C4 (index 0) at 6, C1 (index 2)
        # at 0; the ratios of 100 sit in the masked input row and do not count.
        active = np.full((4, 3), 100.)
        active[0] = [0., 5., 3.]
        ratios = np.zeros((3, 4, 3))
        ratios[0, 1, 2] = 6.
        ratios[1, 3, 1] = 8.
        ratios[:, 0, 1] = 100.
        # Neuron level: 8 input (pre) and 6 hidden (post) neurons, flat synapse
        # index post * 8 + pre. Cluster C3 owns the four synapses from input
        # neurons 6, 7 (cluster C4) onto hidden neurons 2, 3 (cluster C2) plus
        # one stray (input 0 -> hidden 5); C4 owns input 2, 3 -> hidden 4, 5.
        synapse_labels = np.full(48, 21)
        for post in (2, 3):
            for pre in (6, 7):
                synapse_labels[post * 8 + pre] = 3
        synapse_labels[5 * 8 + 0] = 3
        for post in (4, 5):
            for pre in (2, 3):
                synapse_labels[post * 8 + pre] = 4
        return {"modulation_all_var_weighted_unnormalized": {
            "global_assignment_fixed_k20": {
                "fixed_k": 20,
                "all_choice_order": [4, 3, 1],
                "cluster_size_percent": [0.1, 0.1, 0.8],
                "n_active_block": active,
                "om_stack": ratios,
            },
            "global_assignment": {"om_stack": "not this mapping"},
            "mod_shape": (6, 8),
            "result_all_lst": [{"col_group_labels": np.zeros(300),
                                "col_labels_by_k": {20: synapse_labels}}],
        }}

    # Fixed-k neuron labels of the fixture: input neurons in clusters C1..C4
    # (two each), hidden neurons in C1..C3 (two each), no unresponsive class.
    NEURON_LABELS = {"input": (np.repeat([1, 2, 3, 4], 2), None),
                     "hidden": (np.repeat([1, 2, 3], 2), None)}

    def weights(self):
        """Static weights (post 6 x pre 8): negative in the first input column, else positive."""
        w = np.ones((6, 8))
        w[:, 0] = -1.
        return w

    def render(self, cache, min_expected=5.0, clusters=None, select_min_expected=5.0, n_examples=2):
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=cache), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED", min_expected), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_SELECT_MIN_EXPECTED",
                             select_min_expected), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_CLUSTERS", clusters), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_N", n_examples), \
                patch.object(paper_plot, "_fixed_k_neuron_labels",
                             side_effect=lambda side: self.NEURON_LABELS[side]), \
                patch.object(paper_plot, "_plastic_weights", return_value=self.weights()), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot.plot_overmembership_examples()
        saved = {call.args[1].name: call for call in save.call_args_list}
        self.assertEqual(len(saved), save.call_count)
        for call in save.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        return saved

    def test_draws_the_two_most_concentrated_clusters_on_shared_scales(self):
        self.assertEqual(paper_plot.OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED, paper_plot.OM_MIN_EXPECTED)
        self.assertGreater(paper_plot.OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED, 0)
        self.assertIsNone(paper_plot.OVERMEMBERSHIP_EXAMPLE_CLUSTERS)
        self.assertEqual(paper_plot.OVERMEMBERSHIP_EXAMPLE_N, 2)
        cache = self.cache()
        saved = self.render(cache)
        # Two heatmaps, two dot matrices, two synapse maps, their keys and the colorbar.
        self.assertEqual(set(saved), {
            "multitask_overmembership_examples_1.png", "multitask_overmembership_examples_2.png",
            "multitask_overmembership_examples_dots_1.png", "multitask_overmembership_examples_dots_2.png",
            "multitask_overmembership_examples_synapses_1.png",
            "multitask_overmembership_examples_synapses_2.png",
            "multitask_overmembership_examples_synapses_raw_1.png",
            "multitask_overmembership_examples_synapses_raw_2.png",
            "multitask_overmembership_examples_synapses_legend.png",
            "multitask_overmembership_examples_dots_legend.png",
            "multitask_overmembership_examples_colorbar.png"})
        expected = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        # Rank 1 is C3 (peak OM 8), rank 2 is C4 (peak 6); both use clim (0, 8).
        for rank, cluster_id, index, labels in ((1, 3, 1, (["C4"], ["C2"])), (2, 4, 0, (["C2"], ["C3"]))):
            figure = saved[f"multitask_overmembership_examples_{rank}.png"].args[0]
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            self.assertEqual(axis.get_title(), f"Synapse cluster C{cluster_id}")
            self.assertEqual(axis.get_xlabel(), "Input cluster")
            self.assertEqual(axis.get_ylabel(), "Hidden cluster")
            # Only the deciding block's clusters are labelled.
            self.assertEqual([label.get_text() for label in axis.get_xticklabels()], labels[0])
            self.assertEqual([label.get_text() for label in axis.get_yticklabels()], labels[1])
            displayed = axis.collections[0].get_array().reshape(3, 3)
            self.assertFalse(displayed.mask.any() if np.ma.is_masked(displayed) else False)
            np.testing.assert_allclose(np.asarray(displayed), expected["om_stack"][index][1:, :].T)
            self.assertEqual(axis.collections[0].get_clim(), (0, 8))
            self.assertEqual(axis.collections[0].get_cmap().name, paper_plot._MULTITASK_HEATMAP_CMAP)
            self.assertEqual(len(axis.texts), 0)
            self.assertEqual(len(axis.patches), 0)
            self.assertEqual(len(figure.texts), 0)
            np.testing.assert_allclose(figure.get_size_inches(), paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
            extra = saved[f"multitask_overmembership_examples_{rank}.png"].kwargs["extra"]
            self.assertIn(f"example {rank}: synapse cluster C{cluster_id}", extra)
            self.assertIn("fixed k=20", extra)
            self.assertIn("hatched: expected < 5", extra)
            self.assertIn("dropped (no shown block): input C1", extra)
            self.assertIn("runners-up: C1 0.0", extra)
        self.assertEqual(tuple(paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE), (2.0, 2.0))
        colorbar = saved["multitask_overmembership_examples_colorbar.png"].args[0]
        self.assertEqual(colorbar.axes[0].get_ylim(), (0, 8))
        self.assertIn(1., colorbar.axes[0].get_yticks())
        self.assertIs(paper_plot.FIGURES_BY_MODE["multiple_tasks"]["overmembership_examples"],
                      paper_plot.plot_overmembership_examples)

    def test_dot_matrices_share_one_area_scale_and_one_size_key(self):
        cache = self.cache()
        saved = self.render(cache)
        # C3's 80 observed synapses set the scale; C4's 60 take 3/4 of that area.
        areas = {}
        for rank, cluster_id, position in ((1, 3, (2.5, 1.5)), (2, 4, (0.5, 2.5))):
            figure = saved[f"multitask_overmembership_examples_dots_{rank}.png"].args[0]
            axis = figure.axes[0]
            self.assertEqual(axis.get_title(), f"Synapse cluster C{cluster_id}")
            self.assertIsNone(axis.get_legend())
            dots = [c for c in axis.collections if isinstance(c, paper_plot.mpl.collections.PathCollection)]
            self.assertEqual(len(dots), 2)
            filled, rings = (dots[0], dots[1]) if len(dots[0].get_offsets()) == 1 else (dots[1], dots[0])
            np.testing.assert_allclose(filled.get_offsets(), [list(position)])
            self.assertEqual(len(rings.get_offsets()), 8)
            np.testing.assert_allclose(rings.get_sizes(), [paper_plot._OVERMEMBERSHIP_DOT_EMPTY_PT2])
            self.assertEqual(rings.get_facecolors().size, 0)        # hollow
            areas[rank] = float(filled.get_sizes()[0])
            np.testing.assert_allclose(figure.get_size_inches(), paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
        self.assertAlmostEqual(areas[2] / areas[1], 60. / 80.)
        cmap = paper_plot.plt.get_cmap(paper_plot._MULTITASK_HEATMAP_CMAP)
        rank1 = saved["multitask_overmembership_examples_dots_1.png"].args[0].axes[0]
        filled = next(c for c in rank1.collections if len(c.get_offsets()) == 1)
        np.testing.assert_allclose(filled.get_facecolors()[0], cmap(1.0))   # OM 8 of vmax 8
        # Labels: only the enriched block's clusters are significant.
        self.assertEqual([label.get_text() for label in rank1.get_xticklabels()], ["C4"])
        self.assertEqual([label.get_text() for label in rank1.get_yticklabels()], ["C2"])
        self.assertIn("labelled: input C4, hidden C2",
                      saved["multitask_overmembership_examples_dots_1.png"].kwargs["extra"])
        key = saved["multitask_overmembership_examples_dots_legend.png"].args[0]
        legend = key.legends[0]
        self.assertEqual(legend.get_title().get_text(), "synapses")
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["10", "20", "50"])
        sizes = [handle.get_markersize() ** 2 for handle in legend.legend_handles]
        self.assertAlmostEqual(sizes[2], areas[1] * 50. / 80.)

    def test_synapse_map_places_every_member_by_neuron_and_colors_by_weight_sign(self):
        saved = self.render(self.cache())
        figure = saved["multitask_overmembership_examples_synapses_1.png"].args[0]
        axis = figure.axes[0]
        self.assertEqual(axis.get_title(), "Synapse cluster C3")
        self.assertEqual(axis.get_xlabel(), "Input neuron")
        self.assertEqual(axis.get_ylabel(), "Hidden neuron")
        np.testing.assert_allclose(figure.get_size_inches(), paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
        dots = {tuple(np.round(c.get_facecolors()[0][:3], 3)): c for c in axis.collections
                if isinstance(c, paper_plot.mpl.collections.PathCollection)}
        to_rgb = paper_plot.mpl.colors.to_rgb
        positive = dots[tuple(np.round(to_rgb(paper_plot._OVERMEMBERSHIP_SYNAPSE_COLORS["W > 0"]), 3))]
        negative = dots[tuple(np.round(to_rgb(paper_plot._OVERMEMBERSHIP_SYNAPSE_COLORS["W < 0"]), 3))]
        # Neurons keep their index order (labels already sorted), so positions
        # are the neuron indices themselves: four positive synapses in the
        # input 6-7 x hidden 2-3 block, one negative stray at input 0, hidden 5.
        expected_positive = sorted((pre + 0.5, post + 0.5) for post in (2, 3) for pre in (6, 7))
        self.assertEqual(sorted(map(tuple, positive.get_offsets())), expected_positive)
        np.testing.assert_allclose(negative.get_offsets(), [[0.5, 5.5]])
        np.testing.assert_allclose(positive.get_sizes(), [paper_plot._OVERMEMBERSHIP_SYNAPSE_PT2])
        # Cluster boundaries between neuron clusters: 3 vertical, 2 horizontal lines.
        verticals = [line for line in axis.lines if line.get_xdata()[0] == line.get_xdata()[1]]
        horizontals = [line for line in axis.lines if line.get_ydata()[0] == line.get_ydata()[1]]
        self.assertEqual(sorted(line.get_xdata()[0] for line in verticals), [2, 4, 6])
        self.assertEqual(sorted(line.get_ydata()[0] for line in horizontals), [2, 4])
        self.assertEqual(len(axis.patches), 0)                 # no unresponsive band in the fixture
        # Ticks sit at the centers of the dot matrix's labelled clusters.
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["C4"])
        np.testing.assert_allclose(axis.get_xticks(), [7.0])
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], ["C2"])
        np.testing.assert_allclose(axis.get_yticks(), [3.0])
        extra = saved["multitask_overmembership_examples_synapses_1.png"].kwargs["extra"]
        self.assertIn("5 synapses of C3, 4 with W > 0, 1 with W < 0", extra)
        self.assertIn("3 input and 3 hidden neurons", extra)
        key = saved["multitask_overmembership_examples_synapses_legend.png"].args[0]
        self.assertEqual([text.get_text() for text in key.legends[0].get_texts()], ["W > 0", "W < 0"])

    def test_raw_synapse_map_keeps_index_order_without_cluster_structure(self):
        saved = self.render(self.cache())
        axis = saved["multitask_overmembership_examples_synapses_raw_1.png"].args[0].axes[0]
        self.assertEqual(axis.get_title(), "Synapse cluster C3")
        self.assertEqual(len(axis.lines), 0)
        self.assertEqual(len(axis.patches), 0)
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["1", "8"])
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], ["1", "6"])
        offsets = np.vstack([c.get_offsets() for c in axis.collections
                             if isinstance(c, paper_plot.mpl.collections.PathCollection)])
        expected = sorted([(pre + 0.5, post + 0.5) for post in (2, 3) for pre in (6, 7)] + [(0.5, 5.5)])
        self.assertEqual(sorted(map(tuple, offsets)), expected)
        self.assertIn("[neurons in index order]",
                      saved["multitask_overmembership_examples_synapses_raw_1.png"].kwargs["extra"])
        # With a shuffled cluster order the raw map still sits at neuron indices.
        labels = np.array([2, 1, 3, 1, 2, 3, 4, 4])
        positions, sorted_labels = paper_plot._neuron_axis_positions(labels, unresponsive=None)
        figure, _ = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        with patch.object(paper_plot, "_save_fig"):
            paper_plot._plot_om_synapse_map(
                np.array([1]), np.array([2]), np.array([-1.]),
                (positions, sorted_labels, None), (np.arange(3), np.array([1, 1, 2]), None),
                (["C1"], ["C1"]), "t", Path("x.png"), order_by_cluster=False)
        drawn = paper_plot.plt.gcf().axes[0]
        np.testing.assert_allclose(drawn.collections[0].get_offsets(), [[2.5, 1.5]])
        self.assertEqual(len(drawn.lines), 0)
        paper_plot.plt.close(paper_plot.plt.gcf())

    def test_unresponsive_neurons_are_shaded_and_sorted_last(self):
        labels = np.array([2, 1, 3, 1, 2, 3, 4, 4])
        positions, sorted_labels = paper_plot._neuron_axis_positions(labels, unresponsive=1)
        np.testing.assert_array_equal(sorted_labels, [2, 2, 3, 3, 4, 4, 1, 1])
        np.testing.assert_array_equal(positions, [0, 6, 2, 7, 1, 3, 4, 5])
        figure, axis = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        with patch.object(paper_plot, "_save_fig"):
            paper_plot._plot_om_synapse_map(
                np.array([0]), np.array([0]), np.array([1.]),
                (positions, sorted_labels, 1), (np.arange(3), np.array([1, 1, 2]), None),
                (["C2"], ["C1"]), "t", Path("x.png"), drop_unresponsive=False)
        drawn = paper_plot.plt.gcf().axes[0]
        self.assertEqual(len(drawn.patches), 1)              # the input U band, nothing else
        self.assertEqual([label.get_text() for label in drawn.get_xticklabels()], ["C2", "U"])
        paper_plot.plt.close(paper_plot.plt.gcf())

    def test_unresponsive_neurons_are_left_out_by_default(self):
        self.assertTrue(paper_plot._OVERMEMBERSHIP_SYNAPSE_DROP_UNRESPONSIVE)
        # Input neurons 1 and 3 are unresponsive (label 1); hidden neuron 2 is (label 9).
        in_labels = np.array([2, 1, 3, 1, 2, 3, 4, 4])
        hid_labels = np.array([5, 5, 9, 6])
        in_axis = (*paper_plot._neuron_axis_positions(in_labels, 1), 1)
        hid_axis = (*paper_plot._neuron_axis_positions(hid_labels, 9), 9)
        # Members: (post 0, pre 0) kept; (post 2, pre 0) on the silent hidden
        # neuron and (post 3, pre 1) on a silent input neuron are dropped.
        posts, pres = np.array([0, 2, 3]), np.array([0, 0, 1])
        figure, _ = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        with patch.object(paper_plot, "_save_fig") as save:
            paper_plot._plot_om_synapse_map(posts, pres, np.array([1., 1., -1.]), in_axis, hid_axis,
                                            (["C2"], ["C5"]), "t", Path("x.png"), extra="  (base)")
        drawn = paper_plot.plt.gcf().axes[0]
        self.assertEqual(drawn.get_xlim(), (0, 6))            # 8 - 2 silent inputs
        self.assertEqual(drawn.get_ylim(), (3, 0))            # 4 - 1 silent hidden
        self.assertEqual(len(drawn.patches), 0)               # no U band
        self.assertEqual([label.get_text() for label in drawn.get_xticklabels()], ["C2"])
        self.assertEqual([label.get_text() for label in drawn.get_yticklabels()], ["C5"])
        dots = [c for c in drawn.collections if isinstance(c, paper_plot.mpl.collections.PathCollection)]
        self.assertEqual(len(dots), 1)
        # Input neuron 0 (label 2) is the first responsive input in cluster
        # order; hidden neuron 0 (label 5) the first hidden: position (0.5, 0.5).
        np.testing.assert_allclose(dots[0].get_offsets(), [[0.5, 0.5]])
        self.assertIn("2 member synapses on them not drawn", save.call_args.kwargs["extra"])
        self.assertIn("6 input x 3 hidden shown", save.call_args.kwargs["extra"])
        paper_plot.plt.close(paper_plot.plt.gcf())

    def test_synapse_map_drops_a_tick_that_would_overlap_a_higher_priority_one(self):
        # 300 input neurons, clusters C1 (neurons 0-9) and C2 (10-19) are
        # 10 neurons apart: far less than one label width at 2 in, so only the
        # higher-priority C2 keeps its tick; U is far away and stays.
        labels = np.concatenate([np.full(10, 1), np.full(10, 2), np.full(280, 3)])
        positions, sorted_labels = paper_plot._neuron_axis_positions(labels, unresponsive=3)
        figure, _ = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, figure)
        with patch.object(paper_plot, "_save_fig"):
            paper_plot._plot_om_synapse_map(
                np.array([0]), np.array([0]), np.array([1.]),
                (positions, sorted_labels, 3), (np.arange(2), np.array([1, 1]), None),
                (["C2", "C1"], ["C1"]), "t", Path("x.png"), drop_unresponsive=False)
        drawn = paper_plot.plt.gcf().axes[0]
        self.assertEqual([label.get_text() for label in drawn.get_xticklabels()], ["C2", "U"])
        paper_plot.plt.close(paper_plot.plt.gcf())

    def test_missing_weights_skip_only_the_synapse_maps(self):
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=self.cache()), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED", 5.0), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_SELECT_MIN_EXPECTED", 5.0), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_CLUSTERS", None), \
                patch.object(paper_plot, "_fixed_k_neuron_labels",
                             side_effect=lambda side: self.NEURON_LABELS[side]), \
                patch.object(paper_plot, "_plastic_weights", return_value=None), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("builtins.print") as output:
            paper_plot.plot_overmembership_examples()
        names = {call.args[1].name for call in save.call_args_list}
        for call in save.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        self.assertEqual(len(names), 6)
        self.assertFalse(any("synapses" in name for name in names))
        self.assertTrue(any("synapse maps skipped" in str(call.args[0])
                            for call in output.call_args_list if call.args))

    def test_dot_area_is_proportional_to_observed_count_and_labels_follow_significance(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["om_stack"][1, 2, 1] = 2.   # a second block for C3: 20 observed vs 80
        assignment["om_stack"][1, 3, 2] = 1.5  # input C4 x hidden C3: 15 observed (z 1.6)
        assignment["om_stack"][1, 2, 0] = 0.5  # input C3 x hidden C1: depleted, never labelled
        saved = self.render(cache)
        axis = saved["multitask_overmembership_examples_dots_1.png"].args[0].axes[0]
        dots = [c for c in axis.collections if isinstance(c, paper_plot.mpl.collections.PathCollection)]
        filled = next(c for c in dots if len(c.get_offsets()) == 4)
        by_position = {tuple(offset): size for offset, size in zip(filled.get_offsets(), filled.get_sizes())}
        self.assertAlmostEqual(by_position[(1.5, 1.5)] / by_position[(2.5, 1.5)], 20. / 80.)
        # Enriched inputs C3 and C4, enriched hiddens C2 and C3, in axis order
        # (three clusters per side give wide cells, so neither pair collides).
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["C3", "C4"])
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], ["C2", "C3"])

    def test_significant_clusters_are_capped_and_ranked_by_poisson_z(self):
        expected = np.full((5, 4), 10.)
        observed = np.full((5, 4), 10.)
        observed[0, 0], observed[1, 1], observed[2, 2], observed[3, 3], observed[4, 0] = 40., 30., 25., 20., 12.
        shown = np.ones((5, 4), bool)
        shown[4, 0] = False                  # the weakest enrichment sits in an unshown block
        inputs, hiddens = paper_plot._om_significant_clusters(observed, expected, shown, max_labels=3)
        self.assertEqual(inputs, [0, 1, 2])  # z 9.5, 6.3, 4.7 beat input 3's 3.2
        self.assertEqual(hiddens, [0, 1, 2])
        inputs, hiddens = paper_plot._om_significant_clusters(observed, expected, shown, max_labels=5)
        self.assertEqual(inputs, [0, 1, 2, 3])   # input 4 has no shown enrichment
        self.assertEqual(hiddens, [0, 1, 2, 3])
        # Most significant first: swapping the strengths swaps the order.
        observed[0, 0], observed[1, 1] = 30., 40.
        inputs, _ = paper_plot._om_significant_clusters(observed, expected, shown, max_labels=2)
        self.assertEqual(inputs, [1, 0])

    def test_colliding_labels_keep_the_more_significant_neighbour(self):
        names = [f"C{index}" for index in range(1, 11)]
        # Significance order 4, 5, 9: C5 (one cell from C4) collides with C4
        # on narrow cells and is dropped; C9 is far enough and stays.
        kept = paper_plot._non_colliding_labels([3, 4, 8], names, cell_pt=5.0, which="x")
        self.assertEqual(kept, [3, 8])
        kept = paper_plot._non_colliding_labels([3, 4, 8], names, cell_pt=30.0, which="x")
        self.assertEqual(kept, [3, 4, 8])            # wide cells: nothing collides
        kept = paper_plot._non_colliding_labels([4, 3], names, cell_pt=5.0, which="y")
        self.assertEqual(kept, [4])                   # y: the more significant C5 wins
        self.assertEqual(paper_plot._non_colliding_labels([], names, 5.0, "x"), [])

    def test_ranking_ignores_high_ratios_on_thinly_supported_blocks(self):
        """With the display threshold at 1 synapse, cluster C1 (share 0.8) shows
        its input-C1 x hidden-C2 block (expected 4) with OM 100, yet the
        examples stay C3 then C4 because only blocks expecting at least 5
        synapses rank them."""
        saved = self.render(self.cache(), min_expected=1.0)
        titles = [saved[f"multitask_overmembership_examples_{rank}.png"].args[0].axes[0].get_title()
                  for rank in (1, 2)]
        self.assertEqual(titles, ["Synapse cluster C3", "Synapse cluster C4"])
        self.assertIn("blocks with expected >= 5 rank the examples",
                      saved["multitask_overmembership_examples_1.png"].kwargs["extra"])
        # Lowering the ranking threshold to the display threshold lets that
        # thin block decide, and C1 becomes the first example.
        saved = self.render(self.cache(), min_expected=1.0, select_min_expected=1.0)
        axis = saved["multitask_overmembership_examples_1.png"].args[0].axes[0]
        self.assertEqual(axis.get_title(), "Synapse cluster C1")
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["C1"])
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], ["C2"])

    def test_forced_cluster_ids_override_the_automatic_choice(self):
        saved = self.render(self.cache(), clusters=(4,))
        self.assertIn("multitask_overmembership_examples_1.png", saved)
        self.assertNotIn("multitask_overmembership_examples_2.png", saved)
        self.assertNotIn("multitask_overmembership_examples_synapses_2.png", saved)
        axis = saved["multitask_overmembership_examples_1.png"].args[0].axes[0]
        self.assertEqual(axis.get_title(), "Synapse cluster C4")
        self.assertEqual(axis.collections[0].get_clim(), (0, 6))
        saved = self.render(self.cache(), clusters=(1, 3))
        titles = [saved[f"multitask_overmembership_examples_{rank}.png"].args[0].axes[0].get_title()
                  for rank in (1, 2)]
        self.assertEqual(titles, ["Synapse cluster C1", "Synapse cluster C3"])

    def test_partially_masked_cluster_stays_with_hatched_blocks(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"][0] = [0., 5., 100.]   # C1 keeps one shown block (share 0.1)
        saved = self.render(cache)
        axis = saved["multitask_overmembership_examples_1.png"].args[0].axes[0]
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["C4"])
        np.testing.assert_allclose(axis.get_xticks(), [3.5])     # input C1 is kept now
        displayed = axis.collections[0].get_array().reshape(3, 4)
        self.assertTrue(displayed.mask[0, 0] and displayed.mask[1, 0])
        self.assertFalse(displayed.mask[2, 0])
        self.assertEqual(len(axis.texts), 0)
        # One hatched patch per masked block, nothing else drawn on top.
        self.assertEqual(len(axis.patches), int(displayed.mask.sum()))
        self.assertTrue(all(patch.get_hatch() == "////" for patch in axis.patches))
        self.assertNotIn("dropped", saved["multitask_overmembership_examples_1.png"].kwargs["extra"])

    def test_missing_data_unsupported_cache_or_absent_forced_cluster_skips_without_saving(self):
        unsupported = self.cache()
        # Every block expects fewer synapses than the threshold: nothing to show.
        unsupported["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]["n_active_block"] = np.full((4, 3), 4.)
        adaptive_only = self.cache()
        adaptive_only["modulation_all_var_weighted_unnormalized"].pop("global_assignment_fixed_k20")
        wrong_k = self.cache()
        wrong_k["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]["fixed_k"] = 10
        for data, clusters in ((None, None), (unsupported, None), (adaptive_only, None),
                               (wrong_k, None), (self.cache(), (3, 9))):
            with self.subTest(data_available=data is not None, clusters=clusters), \
                    patch.object(paper_plot, "_load_cluster_info_mod", return_value=data), \
                    patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED", 5.0), \
                    patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_SELECT_MIN_EXPECTED", 5.0), \
                    patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_CLUSTERS", clusters), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_overmembership_examples()
                save.assert_not_called()

    def test_unresponsive_class_is_never_an_example(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["unresponsive_mod_label"] = 3          # the would-be winner is the silent class
        saved = self.render(cache)
        titles = [saved[f"multitask_overmembership_examples_{rank}.png"].args[0].axes[0].get_title()
                  for rank in (1, 2)]
        self.assertEqual(titles, ["Synapse cluster C4", "Synapse cluster C1"])

    def test_fixed_twenty_axes_drop_the_empty_unresponsive_group(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"] = np.full((21, 21), 100.)
        assignment["n_active_block"][20, :] = 0
        assignment["n_active_block"][:, 20] = 0
        assignment["om_stack"] = np.ones((3, 21, 21))
        assignment["om_stack"][0, 1, 13] = 4.
        assignment["om_stack"][1, 3, 13] = 5.
        saved = self.render(cache)
        axis = saved["multitask_overmembership_examples_1.png"].args[0].axes[0]
        # C3 peaks at input C4 x hidden C14 (om_stack[1, 3, 13]).
        self.assertEqual(axis.get_title(), "Synapse cluster C3")
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["C4"])
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], ["C14"])
        np.testing.assert_allclose(axis.get_xticks(), [3.5])
        np.testing.assert_allclose(axis.get_yticks(), [13.5])
        self.assertEqual(axis.collections[0].get_array().reshape(20, 20).shape, (20, 20))
        self.assertIn("dropped (no shown block): input U; hidden U",
                      saved["multitask_overmembership_examples_1.png"].kwargs["extra"])
        self.assertEqual(saved["multitask_overmembership_examples_2.png"].args[0].axes[0].get_title(),
                         "Synapse cluster C4")


if __name__ == "__main__":
    unittest.main()
