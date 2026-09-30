"""Small cached-data checks for individual modulation-cluster enrichment plots."""

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

    def render(self, cache, min_expected=5.0):
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=cache), \
                patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED", min_expected), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot.plot_overmembership_examples()
        save.assert_called_once()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        return figure, save.call_args.args[1]

    def test_masks_low_expectation_blocks_drops_empty_clusters_and_annotates_nothing(self):
        self.assertEqual(paper_plot.OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED, paper_plot.OM_MIN_EXPECTED)
        self.assertGreater(paper_plot.OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED, 0)
        cache = self.cache()
        figure, path = self.render(cache)
        self.assertEqual(path.name, "multitask_overmembership_examples.png")
        self.assertEqual(len(figure.axes), 3)
        expected = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        for axis, cluster_id, index in zip(figure.axes[:2], (3, 4), (1, 0)):
            self.assertEqual(axis.get_title(), f"Modulation C{cluster_id}")
            self.assertEqual(axis.get_xlabel(), "Input cluster")
            self.assertEqual(axis.get_ylabel(), "Hidden cluster")
            self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                             ["C2", "C3", "C4"])
            self.assertEqual([label.get_text() for label in axis.get_yticklabels()],
                             ["C1", "C2", "C3"])
            displayed = axis.collections[0].get_array().reshape(3, 3)
            self.assertFalse(displayed.mask.any() if np.ma.is_masked(displayed) else False)
            np.testing.assert_allclose(np.asarray(displayed),
                                       expected["om_stack"][index][1:, :].T)
            self.assertEqual(axis.collections[0].get_clim(), (0, 8))
            self.assertEqual(axis.collections[0].get_cmap().name, paper_plot._MULTITASK_HEATMAP_CMAP)
            # No per-cell numbers, no peak caption, no highlighted block.
            self.assertEqual(len(axis.texts), 0)
            self.assertEqual(len(axis.patches), 0)
        footer = [text.get_text() for text in figure.texts if "Fixed k" in text.get_text()][0]
        self.assertIn("hatched: expected < 5", footer)
        self.assertIn("dropped (no shown block): input C1", footer)
        self.assertNotIn("N/A", footer)
        self.assertIn(1., figure.axes[2].get_yticks())
        self.assertIs(paper_plot.FIGURES_BY_MODE["multiple_tasks"]["overmembership_examples"],
                      paper_plot.plot_overmembership_examples)

    def test_partially_masked_cluster_stays_with_hatched_blocks(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"][0] = [0., 5., 100.]   # C1 keeps one shown block (share 0.1)
        figure, _ = self.render(cache)
        for axis in figure.axes[:2]:
            self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                             ["C1", "C2", "C3", "C4"])
            displayed = axis.collections[0].get_array().reshape(3, 4)
            self.assertTrue(displayed.mask[0, 0] and displayed.mask[1, 0])
            self.assertFalse(displayed.mask[2, 0])
            self.assertEqual(len(axis.texts), 0)
            # One hatched patch per masked block, nothing else drawn on top.
            self.assertEqual(len(axis.patches), int(displayed.mask.sum()))
            self.assertTrue(all(patch.get_hatch() == "////" for patch in axis.patches))
        footer = [text.get_text() for text in figure.texts if "Fixed k" in text.get_text()][0]
        self.assertNotIn("dropped", footer)

    def test_missing_data_or_unsupported_example_skips_without_saving(self):
        unsupported = self.cache()
        # Every block expects fewer synapses than the threshold: nothing to show.
        unsupported["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]["n_active_block"] = np.full((4, 3), 4.)
        adaptive_only = self.cache()
        adaptive_only["modulation_all_var_weighted_unnormalized"].pop("global_assignment_fixed_k20")
        wrong_k = self.cache()
        wrong_k["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]["fixed_k"] = 10
        for data in (None, unsupported, adaptive_only, wrong_k):
            with self.subTest(data_available=data is not None), \
                    patch.object(paper_plot, "_load_cluster_info_mod", return_value=data), \
                    patch.object(paper_plot, "OVERMEMBERSHIP_EXAMPLE_MIN_EXPECTED", 5.0), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_overmembership_examples()
                save.assert_not_called()

    def test_fixed_twenty_axes_drop_the_empty_unresponsive_group(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"] = np.full((21, 21), 100.)
        assignment["n_active_block"][20, :] = 0
        assignment["n_active_block"][:, 20] = 0
        assignment["om_stack"] = np.ones((3, 21, 21))
        assignment["om_stack"][0, 1, 13] = 4.
        assignment["om_stack"][1, 3, 13] = 5.
        figure, _ = self.render(cache)
        expected_labels = [f"C{index}" for index in range(1, 21)]
        for axis in figure.axes[:2]:
            self.assertEqual([label.get_text() for label in axis.get_xticklabels()], expected_labels)
            self.assertEqual([label.get_text() for label in axis.get_yticklabels()], expected_labels)
            self.assertEqual(axis.collections[0].get_array().reshape(20, 20).shape, (20, 20))
        footer = [text.get_text() for text in figure.texts if "Fixed k" in text.get_text()][0]
        self.assertIn("input U; hidden U", footer)


if __name__ == "__main__":
    unittest.main()
