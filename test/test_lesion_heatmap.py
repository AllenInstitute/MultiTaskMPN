"""Input, hidden and modulation lesion panels use saved effects and identities."""

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


class LesionHeatmapTests(unittest.TestCase):
    def cache(self):
        return {
            "schema_version": 1,
            "aname": paper_plot.ANAME,
            "entries": {
                "leison_unnorm": {
                    "definition": "random_minus_lesion",
                    "units": "fraction",
                    "tasks": ["fdgo", "fdanti"],
                    "conditions": ["post_c7", "pre_c4", "post_noleison", "pre_c2",
                                   "post_c3", "pre_noleison"],
                    "effect": [[0.1, -0.8, 0., 0.3, 0.2, 0.],
                               [0.2, 0.6, 0., -0.1, 0.4, 0.]],
                },
                "modulation_all_var_weighted_unnormalized__freeze_M": {
                    "definition": "random_minus_lesion",
                    "units": "fraction",
                    "tasks": ["fdanti", "fdgo"],
                    "conditions": ["mod_c9", "mod_noleison", "mod_c1"],
                    "effect": [[0.5, 0., -0.2], [0.3, 0., 0.1]],
                },
            },
        }

    def test_input_is_top_and_all_panels_share_task_order_and_color_limits(self):
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_pkl_or_skip", return_value=self.cache()), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch.object(paper_plot, "_save_standalone_colorbar") as colorbar:
            paper_plot.plot_lesion_heatmap()
        save.assert_called_once()
        figure, output_path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(output_path.name, "multitask_lesion_heatmap_unnorm.png")
        self.assertEqual(len(figure.axes), 3)
        expected_effects = (
            [[-80., 30.], [60., -10.]],
            [[10., 20.], [20., 40.]],
            [[30., 10.], [50., -20.]],
        )
        cluster_labels = (["C4", "C2"], ["C7", "C3"], ["C9", "C1"])
        for axis, expected, labels in zip(figure.axes, expected_effects, cluster_labels):
            mesh = axis.collections[0]
            np.testing.assert_allclose(mesh.get_array().reshape(2, 2), expected)
            self.assertEqual(mesh.get_clim(), (-80., 80.))
            self.assertEqual([text.get_text() for text in axis.get_yticklabels()],
                             [paper_plot._TASK_DISPLAY[rule] for rule in ("fdgo", "fdanti")])
            self.assertEqual([axis.xaxis.get_major_formatter()(index)
                              for index in axis.get_xticks()], labels)
        self.assertEqual(figure.axes[-1].get_xlabel(), "Cluster")
        self.assertTrue(all(not axis.get_xticklabels() for axis in figure.axes[:2]))
        colorbar.assert_called_once()
        self.assertEqual(colorbar.call_args.kwargs["vmin"], -80.)
        self.assertEqual(colorbar.call_args.kwargs["vmax"], 80.)

    def test_missing_input_or_nonfinite_input_does_not_save_partial_figure(self):
        for problem in ("missing", "nonfinite"):
            data = self.cache()
            neurons = data["entries"]["leison_unnorm"]
            if problem == "missing":
                indices = [index for index, name in enumerate(neurons["conditions"])
                           if not name.startswith("pre_c")]
                neurons["conditions"] = [neurons["conditions"][index] for index in indices]
                neurons["effect"] = np.asarray(neurons["effect"])[:, indices]
            else:
                neurons["effect"][0][1] = np.nan
            with self.subTest(problem=problem), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_load_pkl_or_skip", return_value=data), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch.object(paper_plot, "_save_standalone_colorbar") as colorbar:
                paper_plot.plot_lesion_heatmap()
                save.assert_not_called()
                colorbar.assert_not_called()


if __name__ == "__main__":
    unittest.main()