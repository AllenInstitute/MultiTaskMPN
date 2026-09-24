"""Logarithmic cluster-size panels preserve the saved percentages and order."""

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


class LesionClusterSizeTests(unittest.TestCase):
    def data(self, hidden_sizes, modulation_sizes):
        names = [f"post_c{index + 1}" for index in range(len(hidden_sizes))]
        return {
            "leison_unnorm": {
                "lesion_units": dict(zip(names, hidden_sizes)),
                "all_comb_names_leison": ["post_noleison", *names],
            },
            "mod_leison": {
                "modulation_all_var_weighted_unnormalized__freeze_M": {
                    "mod_col_clusters": {
                        index + 1: list(range(size))
                        for index, size in reversed(list(enumerate(modulation_sizes)))
                    },
                },
            },
        }

    def test_both_panels_are_logarithmic_without_changing_percentages(self):
        for hidden_sizes, modulation_sizes in (
                ([1, 9, 90], [1, 99, 900]),
                ([1, 0, 99], [0, 100, 900])):
            with self.subTest(hidden_sizes=hidden_sizes), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_load_lesion_results",
                                 return_value=self.data(hidden_sizes, modulation_sizes)), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_lesion_cluster_sizes()
            save.assert_called_once()
            figure, path = save.call_args.args[:2]
            self.addCleanup(paper_plot.plt.close, figure)
            self.assertEqual(path.name, "multitask_lesion_cluster_sizes.png")
            figure.canvas.draw()
            for axis, sizes in zip(figure.axes, (hidden_sizes, modulation_sizes)):
                pct = np.asarray(sizes, dtype=float) / sum(sizes) * 100
                positive = pct > 0
                self.assertEqual(axis.get_yscale(), "log")
                self.assertIn("log scale", axis.get_ylabel())
                np.testing.assert_allclose([bar.get_height() for bar in axis.patches],
                                           pct[positive])
                np.testing.assert_allclose(
                    [bar.get_x() + bar.get_width() / 2 for bar in axis.patches],
                    np.flatnonzero(positive))
                self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                                 ["C1", "C2", "C3"])
                self.assertGreater(axis.get_ylim()[0], 0)
                self.assertLess(axis.get_ylim()[0], pct[positive].min())
                self.assertGreater(axis.get_ylim()[1], pct.max())
                self.assertEqual([text.get_text() for text in axis.texts],
                                 ["0%"] * np.count_nonzero(~positive))

    def test_all_zero_sizes_skip_instead_of_producing_invalid_log_axis(self):
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_lesion_results",
                             return_value=self.data([0, 0], [1, 2])), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot.plot_lesion_cluster_sizes()
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()