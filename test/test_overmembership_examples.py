"""Small cached-data checks for individual modulation-cluster enrichment plots."""

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


class OvermembershipExampleTests(unittest.TestCase):
    def cache(self):
        active = np.full((4, 3), 100.)
        active[0, 0] = 0
        active[0, 1] = 5
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

    def test_renders_original_ids_transposed_axes_raw_ratios_and_shared_scale(self):
        cache = self.cache()
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=cache), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot.plot_overmembership_examples()
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_overmembership_examples.png")
        self.assertEqual(len(figure.axes), 3)
        for axis, cluster_id, index, peak, count in zip(
            figure.axes[:2], (3, 4), (1, 0), ((3, 1), (1, 2)), (80, 60)):
            self.assertEqual(axis.get_title(), f"Modulation C{cluster_id}")
            self.assertEqual(axis.get_xlabel(), "Input cluster")
            self.assertEqual(axis.get_ylabel(), "Hidden cluster")
            self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                             ["C1", "C2", "C3", "C4"])
            mesh = axis.collections[0]
            displayed = mesh.get_array().reshape(3, 4)
            expected = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
            np.testing.assert_array_equal(displayed.mask, (expected["n_active_block"] == 0).T)
            np.testing.assert_allclose(displayed.compressed(),
                                       np.ma.array(expected["om_stack"][index].T,
                                                   mask=displayed.mask).compressed())
            self.assertEqual(mesh.get_clim(), (0, 100))
            self.assertEqual(displayed[1, 0], 100.)
            annotations = {text.get_position(): text.get_text() for text in axis.texts}
            self.assertEqual(annotations[(0.5, 1.5)], "100.0")
            self.assertEqual(annotations[(0.5, 0.5)], "N/A")
            self.assertEqual(annotations[(1.5, 0.5)], "0.0")
            np.testing.assert_allclose(axis.patches[0].get_xy(), np.asarray(peak) + 0.06)
            self.assertTrue(any(f"{count} observed / 10.00 expected" in text.get_text()
                                for text in axis.texts))
            self.assertFalse(any(text.get_text() == "--" for text in axis.texts))
            self.assertEqual(sum(text.get_text() == "N/A" for text in axis.texts), 1)
        self.assertIn(1., figure.axes[2].get_yticks())
        self.assertIs(paper_plot.FIGURES_BY_MODE["multiple_tasks"]["overmembership_examples"],
                      paper_plot.plot_overmembership_examples)

    def test_missing_data_or_unsupported_example_skips_without_saving(self):
        unsupported = self.cache()
        unsupported["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]["om_stack"] *= 0
        adaptive_only = self.cache()
        adaptive_only["modulation_all_var_weighted_unnormalized"].pop("global_assignment_fixed_k20")
        wrong_k = self.cache()
        wrong_k["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]["fixed_k"] = 10
        for data in (None, unsupported, adaptive_only, wrong_k):
            with self.subTest(data_available=data is not None), \
                    patch.object(paper_plot, "_load_cluster_info_mod", return_value=data), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_overmembership_examples()
                save.assert_not_called()

    def test_fixed_twenty_axes_label_extra_unresponsive_group(self):
        cache = self.cache()
        assignment = cache["modulation_all_var_weighted_unnormalized"]["global_assignment_fixed_k20"]
        assignment["n_active_block"] = np.full((21, 21), 100.)
        assignment["n_active_block"][20, :] = 0
        assignment["n_active_block"][:, 20] = 0
        assignment["om_stack"] = np.ones((3, 21, 21))
        assignment["om_stack"][0, 1, 13] = 4.
        assignment["om_stack"][1, 3, 13] = 5.
        assignment["om_stack"][:, 8, 5] = 30.8
        with patch.object(paper_plot, "_load_cluster_info_mod", return_value=cache), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot.plot_overmembership_examples()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        figure.canvas.draw()
        expected_labels = [f"C{index}" for index in range(1, 21)] + ["U"]
        for axis in figure.axes[:2]:
            self.assertEqual([label.get_text() for label in axis.get_xticklabels()], expected_labels)
            self.assertEqual([label.get_text() for label in axis.get_yticklabels()], expected_labels)
            values = axis.collections[0].get_array().reshape(21, 21)
            self.assertTrue(values.mask[-1, :].all())
            self.assertTrue(values.mask[:, -1].all())
            self.assertEqual(sum(text.get_text() == "N/A" for text in axis.texts), 41)
            cell_width = axis.get_window_extent().width / 21
            for text in axis.texts:
                if text.get_text() in ("30.8", "N/A"):
                    self.assertLess(text.get_window_extent().width, cell_width)


if __name__ == "__main__":
    unittest.main()