"""Tests for conditional transfer-speed summaries and reached-seed counts."""

import unittest
from unittest.mock import patch, mock_open

import numpy as np

import _bootstrap  # noqa: F401
import paper_plot as paper


class TransferSpeedTests(unittest.TestCase):
    def test_quartiles_missing_and_single_reaching_seed(self):
        values = [[1, np.nan, 8], [2, np.nan, np.nan],
                  [3, np.nan, np.nan], [100, np.nan, np.nan]]
        median, lower, upper, reached, total = paper._transfer_speed_summary(values)
        np.testing.assert_allclose(median, [2.5, np.nan, 8])
        np.testing.assert_allclose(lower, [1.75, np.nan, 8])
        np.testing.assert_allclose(upper, [27.25, np.nan, 8])
        np.testing.assert_array_equal(reached, [4, 0, 1])
        self.assertEqual(total, 4)

    def test_invalid_times_are_rejected(self):
        for values in ([1, 2], [[0]], [[-1]], [[np.inf]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                paper._transfer_speed_summary(values)

    def test_plot_displays_seed_dots_and_median_without_count_panel(self):
        data = {"thresholds": np.array([0.5, 0.95, 0.99]), "by_ruleset": {
            "fdgo_delaygo": {"per_seed_iters": [[10, 100, np.nan], [30, 900, np.nan]], "n_seeds": 2},
            "fdanti_delaygo": {"per_seed_iters": [[2, 4, 8], [4, 8, 16]], "n_seeds": 2},
        }}
        with patch.object(paper, "_ensure_out_dir"), \
                patch.object(paper.Path, "exists", return_value=True), \
                patch.object(paper.Path, "glob", return_value=iter([paper.Path("test.pkl")])), \
                patch("builtins.open", mock_open()), \
                patch.object(paper.pickle, "load", return_value=data), \
                patch.object(paper, "_save_fig") as save:
            paper.plot_transfer_speed()
        figure = save.call_args.args[0]
        try:
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            np.testing.assert_allclose(axis.lines[0].get_xdata(), [20, 500, np.nan])
            np.testing.assert_allclose(axis.lines[1].get_xdata(), [3, 6, 12])
            dots = [collection for collection in axis.collections
                    if isinstance(collection, paper.mpl.collections.PathCollection)]
            self.assertEqual(sum(len(collection.get_offsets()) for collection in dots), 10)
            self.assertEqual(len(axis.texts), 0)
            self.assertEqual(axis.get_xscale(), "log")
        finally:
            paper.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
