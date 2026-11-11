"""Small cached-data checks for the synapse-cluster enrichment example figures."""

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


def block_cells(axis):
    """{(input position, hidden position): (facecolor, square side)} of the
    blocks a block map draws, from its PolyCollection."""
    cells = {}
    for collection in axis.collections:
        if not isinstance(collection, paper_plot.mpl.collections.PolyCollection):
            continue
        for path, color in zip(collection.get_paths(), collection.get_facecolors()):
            corners = path.vertices[:4]
            center = (round(float(corners[:, 0].mean()), 6), round(float(corners[:, 1].mean()), 6))
            cells[center] = (color, round(float(np.ptp(corners[:, 0])), 6))
    return cells


def circle_cells(axis):
    """{(input position, hidden position): (ring radius, disc radius or
    None)} of the blocks a circle map draws, and the frame's lower-left
    corner, from its patches."""
    Circle, Rectangle = paper_plot.mpl.patches.Circle, paper_plot.mpl.patches.Rectangle
    cells, frames = {}, []
    for patch_ in axis.patches:
        if isinstance(patch_, Rectangle):
            frames.append(tuple(round(float(value), 6) for value in patch_.get_xy()))
            continue
        assert isinstance(patch_, Circle), patch_
        center = tuple(round(float(value), 6) for value in patch_.center)
        ring, disc = cells.get(center, (None, None))
        if patch_.get_facecolor()[3] == 0:          # the hollow expected ring
            np.testing.assert_allclose(patch_.get_edgecolor()[:3], 0.)
            ring = float(patch_.radius)
        else:
            assert (paper_plot.mpl.colors.to_rgb(patch_.get_facecolor())
                    == paper_plot.mpl.colors.to_rgb(paper_plot._OM_CIRCLE_FILL))
            disc = float(patch_.radius)
        cells[center] = (ring, disc)
    assert len(frames) == 1, frames
    return cells, frames[0]


def deciding_ring(axis):
    """The one hollow black-edged marker of a block map, its deciding block."""
    rings = [c for c in axis.collections if isinstance(c, paper_plot.mpl.collections.PathCollection)]
    assert len(rings) == 1, rings
    ring = rings[0]
    assert ring.get_facecolors().size == 0
    np.testing.assert_allclose(ring.get_edgecolors()[0][:3], 0.)
    return ring


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

    def render(self, cache, min_expected=5.0, clusters=None, select_min_expected=5.0, n_examples=2):
        """Render the examples and return the saved figures by file name."""
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=cache), \
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
        # Two block maps with their colorbar, two circle maps with their key.
        self.assertEqual(set(saved), {
            "multitask_overmembership_examples_supported_1.png",
            "multitask_overmembership_examples_supported_2.png",
            "multitask_overmembership_examples_colorbar.png",
            "multitask_overmembership_examples_circles_1.png",
            "multitask_overmembership_examples_circles_2.png",
            "multitask_overmembership_examples_circles_legend.png"})
        # Rank 1 is C3 (peak OM 7 at input C4 x hidden C2), rank 2 is C4 (peak 5
        # at input C2 x hidden C3); both share the colorbar's (-1, 7).
        for rank, cluster_id, peak in ((1, 3, "peak OM 7.00 at input C4 / hidden C2"),
                                       (2, 4, "peak OM 5.00 at input C2 / hidden C3")):
            figure = saved[f"multitask_overmembership_examples_supported_{rank}.png"].args[0]
            # The map, the input bars above it (which carry the title) and
            # the hidden bars to its right.
            self.assertEqual(len(figure.axes), 3)
            axis, top, right = figure.axes
            self.assertEqual(top.get_title(), f"Synapse cluster C{cluster_id}")
            self.assertEqual(axis.get_title(), "")
            self.assertEqual(axis.get_xlabel(), "Input cluster")
            self.assertEqual(axis.get_ylabel(), "Hidden cluster")
            # Input C1 has no block to draw and is dropped: 3 inputs x 3 hiddens;
            # the bar axes share the map's limits.
            self.assertEqual(axis.get_xlim(), (0, 3))
            self.assertEqual(axis.get_ylim(), (3, 0))
            self.assertEqual(top.get_xlim(), (0, 3))
            self.assertEqual(right.get_ylim(), (3, 0))
            self.assertEqual(len(axis.texts), 0)
            self.assertEqual(len(figure.texts), 0)
            np.testing.assert_allclose(figure.get_size_inches(), paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE)
            extra = saved[f"multitask_overmembership_examples_supported_{rank}.png"].kwargs["extra"]
            self.assertIn(f"example {rank}: synapse cluster C{cluster_id}", extra)
            self.assertIn("fixed k=20", extra)
            self.assertIn("the 9 blocks expecting >= 5 synapses", extra)
            self.assertIn("grey: block not assessed (expected < 5)", extra)
            self.assertIn(peak, extra)
            self.assertNotIn("C1,", extra.split("input ")[1].split(";")[0] + ",")
            self.assertIn("runners-up: C1 -1.0", extra)
            self.assertEqual(paper_plot.mpl.colors.to_rgb(axis.get_facecolor()),
                             paper_plot.mpl.colors.to_rgb(paper_plot._OM_BLANK_COLOR))
        self.assertEqual(tuple(paper_plot._OVERMEMBERSHIP_EXAMPLE_FIGSIZE), (2.5, 2.5))
        colorbar = saved["multitask_overmembership_examples_colorbar.png"].args[0]
        self.assertEqual(colorbar.axes[0].get_ylim(), (-1, 7))
        self.assertIn(0., colorbar.axes[0].get_yticks())
        self.assertEqual(colorbar.axes[0].get_ylabel(), "OM (observed / expected − 1)")
        norm = colorbar.axes[0]._colorbar.norm
        self.assertIsInstance(norm, paper_plot.mpl.colors.TwoSlopeNorm)
        self.assertEqual((norm.vmin, norm.vcenter, norm.vmax), (-1, 0, 7))
        self.assertIs(paper_plot.FIGURES_BY_MODE["multiple_tasks"]["overmembership_examples"],
                      paper_plot.plot_overmembership_examples)

    def test_blocks_are_cells_colored_by_om_about_chance_with_the_deciding_block_ringed(self):
        saved = self.render(self.cache())
        cmap, norm = paper_plot._om_block_cmap_norm(7.0)
        # Chance is the diverging map's center, no synapse its blue end.
        self.assertEqual(norm(0.0), 0.5)
        self.assertEqual(norm(-1.0), 0.0)
        self.assertEqual(norm(7.0), 1.0)
        # Sorting puts each example's one enriched block at the top left; the
        # other eight drawn blocks hold no synapse (OM -1). Every cell fills
        # its block.
        for rank, cluster_id, om in ((1, 3, 7.0), (2, 4, 5.0)):
            figure = saved[f"multitask_overmembership_examples_supported_{rank}.png"].args[0]
            axis = figure.axes[0]
            self.assertEqual(figure.axes[1].get_title(), f"Synapse cluster C{cluster_id}")
            self.assertIsNone(axis.get_legend())
            cells = block_cells(axis)
            self.assertEqual(len(cells), 9)
            self.assertEqual({side for _, side in cells.values()}, {1.0})
            np.testing.assert_allclose(cells[(0.5, 0.5)][0], cmap(norm(om)))
            for center, (color, _) in cells.items():
                if center != (0.5, 0.5):
                    np.testing.assert_allclose(color, cmap(0.0))
            ring = deciding_ring(axis)
            np.testing.assert_allclose(ring.get_offsets(), [[0.5, 0.5]])
            self.assertGreater(ring.get_zorder(), 2)
        rank1 = saved["multitask_overmembership_examples_supported_1.png"].args[0].axes[0]
        # The axes carry their titles and no cluster names; the saved
        # message lists the clusters in their drawn order.
        self.assertEqual(len(rank1.get_xticks()), 0)
        self.assertEqual(len(rank1.get_yticks()), 0)
        self.assertEqual((rank1.get_xlabel(), rank1.get_ylabel()), ("Input cluster", "Hidden cluster"))
        extra = saved["multitask_overmembership_examples_supported_1.png"].kwargs["extra"]
        self.assertNotIn("labelled", extra)
        self.assertIn("cell color = OM (observed/expected - 1), diverging about chance, saturating above 7",
                      extra)
        self.assertIn("black ring: deciding block", extra)
        self.assertIn("axes sorted by observed/expected over the blocks expecting >= 5, "
                      "input C4, C2, C3; hidden C2, C1, C3", extra)

    def enriched_cache(self):
        """The base cache with four more C3 blocks: C3 x C2 at OM 1 (20
        observed of 10), C4 x C3 at 0.5 (15), C3 x C1 depleted at -0.5 (5)
        and C2 x C1 at chance (10)."""
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["om_stack"][1, 2, 1] = 2.
        assignment["om_stack"][1, 3, 2] = 1.5
        assignment["om_stack"][1, 2, 0] = 0.5
        assignment["om_stack"][1, 1, 0] = 1.0
        return cache

    def test_cell_colors_follow_om(self):
        cache = self.enriched_cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        cached_ratios = assignment["om_stack"].copy()
        saved = self.render(cache)
        np.testing.assert_array_equal(assignment["om_stack"], cached_ratios)
        axis = saved["multitask_overmembership_examples_supported_1.png"].args[0].axes[0]
        cmap, norm = paper_plot._om_block_cmap_norm(7.0)
        # Inputs C4, C3 and C2 hold 95, 25 and 10 of C3's synapses against 30
        # expected each, hiddens C2, C1 and C3 hold 100, 15 and 15, so the
        # sorted axes run C4, C3, C2 and C2, C1, C3: the peak block C4 x C2
        # is at (0.5, 0.5), C3 x C2 at (1.5, 0.5), C2 x C1 at (2.5, 1.5),
        # C3 x C1 at (1.5, 1.5) and C4 x C3 at (0.5, 2.5).
        cells = block_cells(axis)
        for center, om in (((0.5, 0.5), 7.), ((1.5, 0.5), 1.), ((2.5, 1.5), 0.),
                           ((1.5, 1.5), -0.5), ((0.5, 2.5), 0.5)):
            np.testing.assert_allclose(cells[center][0], cmap(norm(om)))
        # Chance is white, depletion on the blue side, enrichment on the red.
        np.testing.assert_allclose(cells[(2.5, 1.5)][0][:3], cmap(0.5)[:3])
        self.assertGreater(cells[(1.5, 1.5)][0][2], cells[(1.5, 1.5)][0][0])   # blue > red
        self.assertGreater(cells[(0.5, 0.5)][0][0], cells[(0.5, 0.5)][0][2])   # red > blue
        np.testing.assert_allclose(deciding_ring(axis).get_offsets(), [[0.5, 0.5]])

    def test_marginal_bars_show_log2_enrichment_on_one_symmetric_axis(self):
        saved = self.render(self.enriched_cache())
        figure = saved["multitask_overmembership_examples_supported_1.png"].args[0]
        axis, top, right = figure.axes
        # One bar per kept cluster, centered on its cell, log2 of the sorted
        # marginal ratios (95, 25, 10 and 100, 15, 15 over 30 expected).
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
        self.assertIn("marginal bars log2(observed/expected) over those blocks on +-2 doublings",
                      saved["multitask_overmembership_examples_supported_1.png"].kwargs["extra"])
        # Example 2 (C4) has synapses in input C2 and hidden C3 only (60 of 30
        # expected, one doubling): the other clusters sit at the floor.
        second = saved["multitask_overmembership_examples_supported_2.png"].args[0]
        np.testing.assert_allclose([patch.get_height() for patch in second.axes[1].patches], [1., -2., -2.])
        np.testing.assert_allclose([patch.get_width() for patch in second.axes[2].patches], [1., -2., -2.])
        self.assertEqual(second.axes[1].get_ylim(), (-2, 2))
        # With every example within one doubling the axis stays at +-1.
        saved = self.render(self.cache(), clusters=(4,))
        self.assertEqual(saved["multitask_overmembership_examples_supported_1.png"].args[0].axes[1].get_ylim(),
                         (-1, 1))
        self.assertEqual([label.get_text() for label in
                          saved["multitask_overmembership_examples_supported_1.png"].args[0].axes[1].get_yticklabels()],
                         ["1/2", "1", "2"])

    def test_thin_blocks_are_left_out_of_the_map_but_counted_in_the_bars(self):
        """A block expecting 1 to 5 synapses counts toward the axis order and
        the marginal bars (threshold 1) but is not drawn, does not rank the
        example and does not stretch the color scale: input C1's block with
        hidden C2 expects 2 synapses of C3 at a cached ratio of 100, so its
        cell is left grey, while the peak over well-supported blocks (OM 7)
        sets vmax and the deciding block."""
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"][0] = [0., 20., 100.]
        # With 20 active synapses that block expects 16 of C1 (share 0.8), a
        # supported block; C1's cached ratio there drops to 0 so it does not
        # become the first example.
        assignment["om_stack"][2, 0, 1] = 0.
        saved = self.render(cache, min_expected=1.0)
        first = saved["multitask_overmembership_examples_supported_1.png"]
        axis, top, right = first.args[0].axes
        self.assertEqual(top.get_title(), "Synapse cluster C3")
        # Input C1 keeps its supported block with hidden C3, so no cluster goes.
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), ((0, 4), (3, 0)))
        self.assertIn("the 10 blocks expecting >= 5 synapses", first.kwargs["extra"])
        cmap, norm = paper_plot._om_block_cmap_norm(7.0)
        cells = block_cells(axis)
        self.assertEqual(len(cells), 10)
        self.assertTrue(all(side == 1.0 for _, side in cells.values()))
        # C1 holds 200 of C3's 12 expected synapses and sorts first, hidden C2
        # likewise: the thin block would sit at (0.5, 0.5) and is not drawn;
        # its supported neighbour C1 x C3 is an empty cell.
        self.assertNotIn((0.5, 0.5), cells)
        np.testing.assert_allclose(cells[(0.5, 2.5)][0], cmap(0.0))
        to_rgb = paper_plot.mpl.colors.to_rgb
        self.assertEqual(to_rgb(axis.get_facecolor()), to_rgb(paper_plot._OM_BLANK_COLOR))
        self.assertIn("grey: block not assessed", first.kwargs["extra"])
        colorbar = saved["multitask_overmembership_examples_colorbar.png"].args[0]
        self.assertEqual(colorbar.axes[0].get_ylim(), (-1, 7))
        # The deciding block stays C4 x C2 (input position 1 after C1).
        np.testing.assert_allclose(deciding_ring(axis).get_offsets(), [[1.5, 0.5]])
        self.assertIn("peak OM 7.00 at input C4 / hidden C2", first.kwargs["extra"])
        self.assertIn("input C1, C4, C2, C3; hidden C2, C1, C3", first.kwargs["extra"])
        # The bars still count the thin block: input C1 holds 200 of 12
        # expected synapses (0 of 10 without it), hidden C2 280 of 32; the
        # clusters without a synapse sit at the floor of the +-5 axis.
        self.assertEqual(top.get_ylim(), (-5, 5))
        np.testing.assert_allclose([patch_.get_height() for patch_ in top.patches],
                                   [np.log2(200 / 12), np.log2(80 / 30), -5., -5.])
        np.testing.assert_allclose([patch_.get_width() for patch_ in right.patches],
                                   [np.log2(280 / 32), -5., -5.])
        # A cluster left without a supported block leaves the axes: input C1
        # is below the ranking threshold everywhere here.
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"][0] = [0., 20., 0.]
        assignment["om_stack"][2, 0, 1] = 0.
        saved = self.render(cache, min_expected=1.0)
        self.assertEqual(saved["multitask_overmembership_examples_supported_1.png"].args[0].axes[0].get_xlim(),
                         (0, 3))

    def test_circle_map_draws_the_clusters_of_the_most_extreme_blocks(self):
        """C3's three most enriched blocks (ratios 8, 2, 1.5) lie on inputs
        C4 and C3 and hiddens C2 and C3; its most depleted ones (no synapse,
        the first three in block order) add input C2. The 3 x 2 grid is
        drawn in the block map's order, a ring per block with a disc of
        ratio times its area."""
        saved = self.render(self.enriched_cache())
        first = saved["multitask_overmembership_examples_circles_1.png"]
        figure = first.args[0]
        self.assertEqual(len(figure.axes), 1)           # no marginal bars
        axis = figure.axes[0]
        self.assertEqual(axis.get_title(), "Synapse cluster C3")
        self.assertEqual((axis.get_xlabel(), axis.get_ylabel()), ("Input cluster", "Hidden cluster"))
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), ((0, 3), (2, 0)))
        self.assertEqual((len(axis.get_xticks()), len(axis.get_yticks())), (0, 0))
        self.assertEqual(len(axis.collections), 0)      # no colored cells
        np.testing.assert_allclose(figure.get_size_inches(), paper_plot._OM_CIRCLE_FIGSIZE)
        cells, frame = circle_cells(axis)
        # One ring size for both examples: the largest disc (ratio 8) has the
        # maximum radius.
        ring = paper_plot._OM_CIRCLE_MAX_RADIUS / np.sqrt(8.)
        ratios = {(0.5, 0.5): 8., (1.5, 0.5): 2., (2.5, 0.5): 0.,
                  (0.5, 1.5): 1.5, (1.5, 1.5): 0., (2.5, 1.5): 0.}
        self.assertEqual(set(cells), set(ratios))
        for center, ratio in ratios.items():
            self.assertAlmostEqual(cells[center][0], ring)
            if ratio:
                self.assertAlmostEqual(cells[center][1], ring * np.sqrt(ratio))
            else:
                self.assertIsNone(cells[center][1])     # a ring alone: no synapse
        np.testing.assert_allclose(frame, (0.02, 0.02))  # the deciding block, C4 x C2
        extra = first.kwargs["extra"]
        self.assertIn("example 1: synapse cluster C3", extra)
        self.assertIn("3 most enriched and 3 most depleted blocks expecting >= 5 synapses: "
                      "input C4, C3, C2; hidden C2, C3", extra)
        self.assertIn("6/6 blocks drawn", extra)
        # 115 of the cluster's 180 synapses (50 of them in a masked thin block).
        self.assertIn("holding 64% of the cluster's synapses", extra)
        self.assertIn("disc area: observed/expected (0.00 to 8.00)", extra)
        # The second example shares the ring size although its peak is 6.
        second, _ = circle_cells(saved["multitask_overmembership_examples_circles_2.png"].args[0].axes[0])
        self.assertTrue(all(abs(found - ring) < 1e-12 for found, _ in second.values()))
        self.assertAlmostEqual(max(disc for _, disc in second.values() if disc), ring * np.sqrt(6.))
        # The key: three examples of the two circles, saved with the legends.
        key = saved["multitask_overmembership_examples_circles_legend.png"].args[0].axes[0]
        key_cells, key_discs = {}, []
        for patch_ in key.patches:
            key_cells.setdefault(round(float(patch_.center[0]), 6), []).append(float(patch_.radius))
        self.assertEqual(sorted(key_cells), [0.5, 1.5, 2.5])
        for (ratio, label), radii in zip(paper_plot._OM_CIRCLE_LEGEND_RATIOS, key_cells.values()):
            self.assertAlmostEqual(min(radii) / max(radii), np.sqrt(min(ratio, 1 / ratio)))
            key_discs.append(label)
        self.assertEqual([text.get_text() for text in key.texts][:3], key_discs)

    def test_circle_map_leaves_thin_blocks_empty_and_needs_legends_for_its_key(self):
        """A block below the ranking threshold inside the selected grid is
        not drawn at all, which a block without a synapse never is (it keeps
        its ring); the key is a legend figure."""
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"][0] = [0., 20., 100.]
        assignment["om_stack"][2, 0, 1] = 0.
        saved = self.render(cache, min_expected=1.0)
        first = saved["multitask_overmembership_examples_circles_1.png"]
        cells, frame = circle_cells(first.args[0].axes[0])
        # Inputs C1, C4, C2 x hiddens C2, C1, C3; C1 x C2 expects 2 synapses
        # and C1 x C1 none.
        self.assertIn("input C1, C4, C2; hidden C2, C1, C3", first.kwargs["extra"])
        self.assertIn("7/9 blocks drawn", first.kwargs["extra"])
        self.assertEqual(len(cells), 7)
        self.assertNotIn((0.5, 0.5), cells)
        self.assertNotIn((0.5, 1.5), cells)
        np.testing.assert_allclose(frame, (1.02, 0.02))  # C4 x C2
        with patch.object(paper_plot, "SHOW_LEGEND", False):
            saved = self.render(self.cache())
        self.assertIn("multitask_overmembership_examples_circles_1.png", saved)
        self.assertNotIn("multitask_overmembership_examples_circles_legend.png", saved)

    def test_ranking_ignores_high_ratios_on_thinly_supported_blocks(self):
        """With the display threshold at 1 synapse, cluster C1 (share 0.8) shows
        its input-C1 x hidden-C2 block (expected 4) with OM 99, yet the
        examples stay C3 then C4 because only blocks expecting at least 5
        synapses rank them, and that block does not stretch the color scale."""
        saved = self.render(self.cache(), min_expected=1.0)
        titles = [saved[f"multitask_overmembership_examples_supported_{rank}.png"].args[0].axes[1].get_title()
                  for rank in (1, 2)]
        self.assertEqual(titles, ["Synapse cluster C3", "Synapse cluster C4"])
        self.assertIn("blocks with expected >= 5 rank the examples",
                      saved["multitask_overmembership_examples_supported_1.png"].kwargs["extra"])
        self.assertEqual(saved["multitask_overmembership_examples_colorbar.png"].args[0].axes[0].get_ylim(),
                         (-1, 7))
        # Lowering the ranking threshold to the display threshold lets that
        # thin block decide, and C1 becomes the first example with its OM 99
        # at the top of the scale.
        saved = self.render(self.cache(), min_expected=1.0, select_min_expected=1.0)
        first = saved["multitask_overmembership_examples_supported_1.png"]
        self.assertEqual(first.args[0].axes[1].get_title(), "Synapse cluster C1")
        self.assertIn("peak OM 99.00 at input C1 / hidden C2", first.kwargs["extra"])
        self.assertEqual(saved["multitask_overmembership_examples_colorbar.png"].args[0].axes[0].get_ylim(),
                         (-1, 99))

    def test_forced_cluster_ids_override_the_automatic_choice(self):
        saved = self.render(self.cache(), clusters=(4,))
        self.assertIn("multitask_overmembership_examples_supported_1.png", saved)
        self.assertNotIn("multitask_overmembership_examples_supported_2.png", saved)
        figure = saved["multitask_overmembership_examples_supported_1.png"].args[0]
        self.assertEqual(figure.axes[1].get_title(), "Synapse cluster C4")
        colorbar = saved["multitask_overmembership_examples_colorbar.png"].args[0]
        self.assertEqual(colorbar.axes[0].get_ylim(), (-1, 5))
        saved = self.render(self.cache(), clusters=(1, 3))
        titles = [saved[f"multitask_overmembership_examples_supported_{rank}.png"].args[0].axes[1].get_title()
                  for rank in (1, 2)]
        self.assertEqual(titles, ["Synapse cluster C1", "Synapse cluster C3"])

    def test_partially_measurable_input_keeps_its_column(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"][0] = [0., 5., 100.]   # C1 keeps one shown block (share 0.1)
        saved = self.render(cache)
        first = saved["multitask_overmembership_examples_supported_1.png"]
        axis = first.args[0].axes[0]
        self.assertEqual(axis.get_xlim(), (0, 4))                 # input C1 is kept now
        # C4 holds the only synapses and sorts first; the others, with none,
        # tie at the floor and keep their cluster order.
        self.assertIn("input C4, C1, C2, C3;", first.kwargs["extra"])
        # Its two unmeasurable blocks stay grey: 10 of 12 blocks are drawn.
        self.assertIn("the 10 blocks expecting >= 5 synapses", first.kwargs["extra"])
        self.assertEqual(len(block_cells(axis)), 10)

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
        titles = [saved[f"multitask_overmembership_examples_supported_{rank}.png"].args[0].axes[1].get_title()
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
        first = saved["multitask_overmembership_examples_supported_1.png"]
        axis = first.args[0].axes[0]
        # C3 peaks at input C4 x hidden C14 (om_stack[1, 3, 13]), its only
        # enriched block, sorted to the front of both axes.
        self.assertEqual(first.args[0].axes[1].get_title(), "Synapse cluster C3")
        np.testing.assert_allclose(deciding_ring(axis).get_offsets(), [[0.5, 0.5]])
        self.assertIn("input C4,", first.kwargs["extra"])
        self.assertIn("hidden C14,", first.kwargs["extra"])
        # The empty unresponsive group U is on neither axis.
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), ((0, 20), (20, 0)))
        self.assertNotIn("U", first.kwargs["extra"].split("axes sorted")[1].split("marginal bars")[0])
        self.assertEqual(saved["multitask_overmembership_examples_supported_2.png"].args[0].axes[1].get_title(),
                         "Synapse cluster C4")


if __name__ == "__main__":
    unittest.main()
