"""Small cached-data checks for the synapse-cluster enrichment example figures."""

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


class OvermembershipExampleTests(unittest.TestCase):
    def cache(self):
        # Input cluster C1 (row 0) has too few surviving synapses in every block
        # (expected 0, 0.5 / 4 and 0.3 / 2.4 for shares 0.1 / 0.8), so it is
        # dropped; every other block expects 10 or 80 synapses and is shown.
        # Cluster C3 (index 1) peaks at OM 7, C4 (index 0) at 5, C1 (index 2)
        # at -1; cached ratios stay 8, 6, 0. Ratios of 100 sit in the masked
        # input row and do not count.
        active = np.full((4, 3), 100.)
        active[0] = [0., 5., 3.]
        ratios = np.zeros((3, 4, 3))
        ratios[0, 1, 2] = 6.
        ratios[1, 3, 1] = 8.
        ratios[:, 0, 1] = 100.
        return {"modulation_all_var_weighted_unnormalized": {
            "global_assignment_fixed_k20": {
                "fixed_k": 20,
                "all_choice_order": [4, 3, 1],
                "cluster_size_percent": [0.1, 0.1, 0.8],
                "n_active_block": active,
                "om_stack": ratios,
            },
            "global_assignment": {"om_stack": "not this mapping"},
        }}

    def render(self, cache, min_expected=5.0, clusters=None, select_min_expected=5.0, n_examples=2,
               cluster_names=True):
        """Render the examples; cluster names are on by default here so the
        label-choice tests see them (the figure's own default is off)."""
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=cache), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_CLUSTER_NAMES", cluster_names), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED", min_expected), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_SELECT_MIN_EXPECTED",
                             select_min_expected), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_CLUSTERS", clusters), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_N", n_examples), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot.plot_overmembership_examples()
        saved = {call.args[1].name: call for call in save.call_args_list}
        self.assertEqual(len(saved), save.call_count)
        for call in save.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        return saved

    def test_draws_the_two_most_enriched_clusters_on_shared_scales(self):
        self.assertEqual(paper_plot.OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED, 1.0)
        self.assertGreater(paper_plot.OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED, 0)
        self.assertIsNone(paper_plot.OVERMEMBERSHIP_EXAMPLE_CLUSTERS)
        self.assertEqual(paper_plot.OVERMEMBERSHIP_EXAMPLE_N, 2)
        cache = self.cache()
        saved = self.render(cache)
        # Two dot matrices, the size key and the colorbar; no heatmap.
        self.assertEqual(set(saved), {
            "multitask_overmembership_examples_dots_1.png", "multitask_overmembership_examples_dots_2.png",
            "multitask_overmembership_examples_dots_legend.png",
            "multitask_overmembership_examples_colorbar.png"})
        # Rank 1 is C3 (peak OM 7 at input C4 x hidden C2), rank 2 is C4 (peak 5
        # at input C2 x hidden C3); both share the colorbar's (-1, 7).
        for rank, cluster_id, peak in ((1, 3, "peak OM 7.00 at input C4 / hidden C2"),
                                       (2, 4, "peak OM 5.00 at input C2 / hidden C3")):
            figure = saved[f"multitask_overmembership_examples_dots_{rank}.png"].args[0]
            # The matrix, the input bars above it (which carry the title) and
            # the hidden bars to its right.
            self.assertEqual(len(figure.axes), 3)
            axis, top, right = figure.axes
            self.assertEqual(top.get_title(), f"Synapse cluster C{cluster_id}")
            self.assertEqual(axis.get_title(), "")
            self.assertEqual(axis.get_xlabel(), "Input cluster")
            self.assertEqual(axis.get_ylabel(), "Hidden cluster")
            # Input C1 has no shown block and is dropped: 3 inputs x 3 hiddens;
            # the bar axes share the matrix's limits.
            self.assertEqual(axis.get_xlim(), (0, 3))
            self.assertEqual(axis.get_ylim(), (3, 0))
            self.assertEqual(top.get_xlim(), (0, 3))
            self.assertEqual(right.get_ylim(), (3, 0))
            self.assertEqual(len(axis.texts), 0)
            self.assertEqual(len(figure.texts), 0)
            np.testing.assert_allclose(figure.get_size_inches(), paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
            extra = saved[f"multitask_overmembership_examples_dots_{rank}.png"].kwargs["extra"]
            self.assertIn(f"example {rank}: synapse cluster C{cluster_id}", extra)
            self.assertIn("fixed k=20", extra)
            self.assertIn("blank: expected < 5", extra)
            self.assertIn(peak, extra)
            self.assertIn("dropped (no shown block): input C1", extra)
            self.assertIn("runners-up: C1 -1.0", extra)
        self.assertEqual(tuple(paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE), (2.5, 2.5))
        colorbar = saved["multitask_overmembership_examples_colorbar.png"].args[0]
        self.assertEqual(colorbar.axes[0].get_ylim(), (-1, 7))
        self.assertIn(0., colorbar.axes[0].get_yticks())
        self.assertEqual(colorbar.axes[0].get_ylabel(), "OM (observed / expected − 1)")
        self.assertIs(paper_plot.FIGURES_BY_MODE["multiple_tasks"]["overmembership_examples"],
                      paper_plot.plot_overmembership_examples)

    def test_dot_matrices_share_one_area_scale_and_one_size_key(self):
        cache = self.cache()
        saved = self.render(cache)
        # C3's 80 observed synapses set the scale (every shown block expects
        # 10); C4's 60 take 3/4 of that area. Sorting puts each example's one
        # enriched block at the top left, and every shown block gets a ring
        # with the expected count's area, drawn over the discs.
        areas = {}
        for rank, cluster_id in ((1, 3), (2, 4)):
            figure = saved[f"multitask_overmembership_examples_dots_{rank}.png"].args[0]
            axis = figure.axes[0]
            self.assertEqual(figure.axes[1].get_title(), f"Synapse cluster C{cluster_id}")
            self.assertIsNone(axis.get_legend())
            dots = [c for c in axis.collections if isinstance(c, paper_plot.mpl.collections.PathCollection)]
            self.assertEqual(len(dots), 2)
            filled, rings = (dots[0], dots[1]) if len(dots[0].get_offsets()) == 1 else (dots[1], dots[0])
            np.testing.assert_allclose(filled.get_offsets(), [[0.5, 0.5]])
            self.assertEqual(len(rings.get_offsets()), 9)
            self.assertEqual(rings.get_facecolors().size, 0)        # hollow
            self.assertGreater(rings.get_zorder(), filled.get_zorder())
            areas[rank] = float(filled.get_sizes()[0])
            np.testing.assert_allclose(rings.get_sizes(), np.full(9, areas[1] * 10. / 80.))
            np.testing.assert_allclose(figure.get_size_inches(), paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
        self.assertAlmostEqual(areas[2] / areas[1], 60. / 80.)
        cmap = paper_plot.plt.get_cmap(paper_plot._MULTITASK_HEATMAP_CMAP)
        rank1 = saved["multitask_overmembership_examples_dots_1.png"].args[0].axes[0]
        filled = next(c for c in rank1.collections if len(c.get_offsets()) == 1)
        np.testing.assert_allclose(filled.get_facecolors()[0], cmap(1.0))   # OM 7 of vmax 7
        # Labels: only the enriched block's clusters are significant, and the
        # sort puts them first on both axes.
        self.assertEqual([label.get_text() for label in rank1.get_xticklabels()], ["C4"])
        self.assertEqual([label.get_text() for label in rank1.get_yticklabels()], ["C2"])
        np.testing.assert_allclose(rank1.get_xticks(), [0.5])
        np.testing.assert_allclose(rank1.get_yticks(), [0.5])
        extra = saved["multitask_overmembership_examples_dots_1.png"].kwargs["extra"]
        self.assertIn("labelled: input C4, hidden C2", extra)
        self.assertIn("disc area = observed synapses, ring area = expected synapses", extra)
        self.assertIn("axes sorted by observed/expected over shown blocks, input C4, C2, C3; "
                      "hidden C2, C1, C3", extra)
        # The size key: discs for round observed counts and one ring, the
        # expected count, at the largest reference.
        key = saved["multitask_overmembership_examples_dots_legend.png"].args[0]
        legend = key.legends[0]
        self.assertEqual(legend.get_title().get_text(), "synapses")
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["10", "20", "50", "expected"])
        sizes = [handle.get_markersize() ** 2 for handle in legend.legend_handles]
        self.assertAlmostEqual(sizes[2], areas[1] * 50. / 80.)
        self.assertAlmostEqual(sizes[3], sizes[2])
        self.assertEqual(legend.legend_handles[3].get_markerfacecolor(), "none")
        self.assertIn("ring: expected count",
                      saved["multitask_overmembership_examples_dots_legend.png"].kwargs["extra"])

    def test_cluster_names_are_off_by_default_and_go_to_the_saved_message(self):
        """With the figure's default, the dot matrix carries no ticks but
        keeps its axis titles."""
        self.assertFalse(paper_plot.OVERMEMBERSHIP_EXAMPLE_CLUSTER_NAMES)
        saved = self.render(self.cache(), cluster_names=False)
        axis = saved["multitask_overmembership_examples_dots_1.png"].args[0].axes[0]
        self.assertEqual(len(axis.get_xticks()), 0)
        self.assertEqual(len(axis.get_yticks()), 0)
        self.assertEqual((axis.get_xlabel(), axis.get_ylabel()), ("Input cluster", "Hidden cluster"))
        self.assertIn("most significant (not drawn): input C4, hidden C2",
                      saved["multitask_overmembership_examples_dots_1.png"].kwargs["extra"])

    def test_dot_area_is_proportional_to_observed_count_and_labels_follow_significance(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["om_stack"][1, 2, 1] = 2.   # a second block for C3: 20 observed vs 80
        assignment["om_stack"][1, 3, 2] = 1.5  # input C4 x hidden C3: 15 observed (z 1.6)
        assignment["om_stack"][1, 2, 0] = 0.5  # input C3 x hidden C1: depleted, never labelled
        assignment["om_stack"][1, 1, 0] = 1.0  # chance: OM 0, still 10 observed synapses
        cached_ratios = assignment["om_stack"].copy()
        saved = self.render(cache)
        np.testing.assert_array_equal(assignment["om_stack"], cached_ratios)
        axis = saved["multitask_overmembership_examples_dots_1.png"].args[0].axes[0]
        dots = [c for c in axis.collections if isinstance(c, paper_plot.mpl.collections.PathCollection)]
        filled = next(c for c in dots if len(c.get_offsets()) == 5)
        # Inputs C4, C3 and C2 hold 95, 25 and 10 of C3's synapses against 30
        # expected each, hiddens C2, C1 and C3 hold 100, 15 and 15, so the
        # sorted axes run C4, C3, C2 and C2, C1, C3: the peak block C4 x C2
        # is at (0.5, 0.5), C3 x C2 at (1.5, 0.5), C2 x C1 at (2.5, 1.5),
        # C3 x C1 at (1.5, 1.5) and C4 x C3 at (0.5, 2.5).
        by_position = {tuple(offset): size for offset, size in zip(filled.get_offsets(), filled.get_sizes())}
        self.assertAlmostEqual(by_position[(1.5, 0.5)] / by_position[(0.5, 0.5)], 20. / 80.)
        self.assertAlmostEqual(by_position[(2.5, 1.5)] / by_position[(0.5, 0.5)], 10. / 80.)
        self.assertAlmostEqual(by_position[(1.5, 1.5)] / by_position[(0.5, 0.5)], 5. / 80.)
        self.assertAlmostEqual(by_position[(0.5, 2.5)] / by_position[(0.5, 0.5)], 15. / 80.)
        # Chance and depletion use OM 0 and -0.5 on the same colorbar as the
        # peak (OM 7), while their nonzero areas retain the observed counts.
        colors = {tuple(offset): color for offset, color in zip(filled.get_offsets(), filled.get_facecolors())}
        colorbar = saved["multitask_overmembership_examples_colorbar.png"].args[0].axes[0]._colorbar
        np.testing.assert_allclose(colors[(2.5, 1.5)], colorbar.cmap(colorbar.norm(0.0)))
        np.testing.assert_allclose(colors[(1.5, 1.5)], colorbar.cmap(colorbar.norm(-0.5)))
        # Enriched inputs C4 and C3, enriched hiddens C2 and C3, in axis order
        # (three clusters per side give wide cells, so neither pair collides).
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["C4", "C3"])
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], ["C2", "C3"])
        np.testing.assert_allclose(axis.get_xticks(), [0.5, 1.5])
        np.testing.assert_allclose(axis.get_yticks(), [0.5, 2.5])

    def test_marginal_bars_show_log2_enrichment_on_one_symmetric_axis(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["om_stack"][1, 2, 1] = 2.
        assignment["om_stack"][1, 3, 2] = 1.5
        assignment["om_stack"][1, 2, 0] = 0.5
        assignment["om_stack"][1, 1, 0] = 1.0
        saved = self.render(cache)
        figure = saved["multitask_overmembership_examples_dots_1.png"].args[0]
        axis, top, right = figure.axes
        # One bar per kept cluster, centered on its cell, log2 of the sorted
        # marginal ratios of the previous test (95, 25, 10 and 100, 15, 15
        # over 30 expected).
        np.testing.assert_allclose([patch.get_x() + patch.get_width() / 2 for patch in top.patches],
                                   [0.5, 1.5, 2.5])
        np.testing.assert_allclose([patch.get_height() for patch in top.patches],
                                   np.log2([95 / 30, 25 / 30, 10 / 30]))
        np.testing.assert_allclose([patch.get_y() + patch.get_height() / 2 for patch in right.patches],
                                   [0.5, 1.5, 2.5])
        np.testing.assert_allclose([patch.get_width() for patch in right.patches],
                                   np.log2([100 / 30, 15 / 30, 15 / 30]))
        # Chance lines at 0; the axis is symmetric and covers every example
        # (C3's hidden C2 at 100/30 needs two doublings), labelled as ratios.
        self.assertEqual([line.get_ydata()[0] for line in top.lines], [0])
        self.assertEqual([line.get_xdata()[0] for line in right.lines], [0])
        self.assertEqual(top.get_ylim(), (-2, 2))
        self.assertEqual(right.get_xlim(), (-2, 2))
        self.assertEqual([label.get_text() for label in top.get_yticklabels()], ["1/4", "1", "4"])
        self.assertEqual([label.get_text() for label in right.get_xticklabels()], ["1/4", "1", "4"])
        self.assertEqual(top.get_ylabel(), "obs / exp")
        self.assertIn("marginal bars log2(observed/expected) on +-2 doublings",
                      saved["multitask_overmembership_examples_dots_1.png"].kwargs["extra"])
        # Example 2 (C4) has synapses in input C2 and hidden C3 only (60 of 30
        # expected, one doubling): the other clusters sit at the floor.
        second = saved["multitask_overmembership_examples_dots_2.png"].args[0]
        np.testing.assert_allclose([patch.get_height() for patch in second.axes[1].patches], [1., -2., -2.])
        np.testing.assert_allclose([patch.get_width() for patch in second.axes[2].patches], [1., -2., -2.])
        self.assertEqual(second.axes[1].get_ylim(), (-2, 2))
        # With every example within one doubling the axis stays at +-1.
        saved = self.render(self.cache(), clusters=(4,))
        self.assertEqual(saved["multitask_overmembership_examples_dots_1.png"].args[0].axes[1].get_ylim(),
                         (-1, 1))
        self.assertEqual([label.get_text() for label in
                          saved["multitask_overmembership_examples_dots_1.png"].args[0].axes[1].get_yticklabels()],
                         ["1/2", "1", "2"])

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
        its input-C1 x hidden-C2 block (expected 4) with OM 99, yet the
        examples stay C3 then C4 because only blocks expecting at least 5
        synapses rank them."""
        saved = self.render(self.cache(), min_expected=1.0)
        titles = [saved[f"multitask_overmembership_examples_dots_{rank}.png"].args[0].axes[1].get_title()
                  for rank in (1, 2)]
        self.assertEqual(titles, ["Synapse cluster C3", "Synapse cluster C4"])
        self.assertIn("blocks with expected >= 5 rank the examples",
                      saved["multitask_overmembership_examples_dots_1.png"].kwargs["extra"])
        # Lowering the ranking threshold to the display threshold lets that
        # thin block decide, and C1 becomes the first example.
        saved = self.render(self.cache(), min_expected=1.0, select_min_expected=1.0)
        first = saved["multitask_overmembership_examples_dots_1.png"]
        self.assertEqual(first.args[0].axes[1].get_title(), "Synapse cluster C1")
        self.assertIn("peak OM 99.00 at input C1 / hidden C2", first.kwargs["extra"])

    def test_forced_cluster_ids_override_the_automatic_choice(self):
        saved = self.render(self.cache(), clusters=(4,))
        self.assertIn("multitask_overmembership_examples_dots_1.png", saved)
        self.assertNotIn("multitask_overmembership_examples_dots_2.png", saved)
        figure = saved["multitask_overmembership_examples_dots_1.png"].args[0]
        self.assertEqual(figure.axes[1].get_title(), "Synapse cluster C4")
        colorbar = saved["multitask_overmembership_examples_colorbar.png"].args[0]
        self.assertEqual(colorbar.axes[0].get_ylim(), (-1, 5))
        saved = self.render(self.cache(), clusters=(1, 3))
        titles = [saved[f"multitask_overmembership_examples_dots_{rank}.png"].args[0].axes[1].get_title()
                  for rank in (1, 2)]
        self.assertEqual(titles, ["Synapse cluster C1", "Synapse cluster C3"])

    def test_partially_measurable_input_keeps_its_column(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"][0] = [0., 5., 100.]   # C1 keeps one shown block (share 0.1)
        saved = self.render(cache)
        first = saved["multitask_overmembership_examples_dots_1.png"]
        axis = first.args[0].axes[0]
        self.assertEqual(axis.get_xlim(), (0, 4))                 # input C1 is kept now
        # C4 holds the only synapses and sorts first; the others, with none,
        # tie at the floor and keep their cluster order.
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["C4"])
        np.testing.assert_allclose(axis.get_xticks(), [0.5])
        self.assertIn("input C4, C1, C2, C3;", first.kwargs["extra"])
        # Its two unmeasurable blocks stay blank: 10 of 12 blocks are drawn.
        self.assertIn("10/12 blocks shown", first.kwargs["extra"])
        self.assertNotIn("dropped", first.kwargs["extra"])

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
        titles = [saved[f"multitask_overmembership_examples_dots_{rank}.png"].args[0].axes[1].get_title()
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
        first = saved["multitask_overmembership_examples_dots_1.png"]
        axis = first.args[0].axes[0]
        # C3 peaks at input C4 x hidden C14 (om_stack[1, 3, 13]), its only
        # enriched block, so the only labelled clusters, sorted to the front.
        self.assertEqual(first.args[0].axes[1].get_title(), "Synapse cluster C3")
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["C4"])
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], ["C14"])
        np.testing.assert_allclose(axis.get_xticks(), [0.5])
        np.testing.assert_allclose(axis.get_yticks(), [0.5])
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), ((0, 20), (20, 0)))
        self.assertIn("dropped (no shown block): input U; hidden U", first.kwargs["extra"])
        self.assertEqual(saved["multitask_overmembership_examples_dots_2.png"].args[0].axes[1].get_title(),
                         "Synapse cluster C4")


if __name__ == "__main__":
    unittest.main()
