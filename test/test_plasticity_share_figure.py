"""The plasticity-share paper figure redraws lesion_plot's saved cache only."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


class PlasticityShareFigureTests(unittest.TestCase):
    def cache(self):
        tasks = ["fdgo", "reactgo", "delaygo", "dmcgo"]
        family = ["reaction", "reaction", "memory", "memory"]
        share = np.full((4, 3), np.nan)
        share[0] = [0.1, 0.3, np.nan]
        share[1] = [0.2, np.nan, np.nan]
        share[2] = [0.9, 1.1, 2.5]      # 2.5 lies above the y-limits
        share[3] = [0.8, -0.9, 1.0]     # -0.9 lies below the y-limits
        medians = np.array([np.nanmedian(row) for row in share])
        return {"share": share, "sig_zero_w": np.isfinite(share), "tasks": tasks,
                "task_family": family, "task_median_share": medians,
                "mwu_U": 4.0, "mwu_p_one_sided": 0.0125,
                "mod_type": paper_plot.PLASTICITY_SHARE_MOD_TYPE, "fdr_q": 0.05}

    def write(self, root, cache):
        path = root / f"plasticity_share_var-weighted-unnormalized_{paper_plot.LESION_ANAME}.pkl"
        with path.open("wb") as stream:
            pickle.dump(cache, stream)

    def render(self, root, show_legend=True):
        with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                patch.object(paper_plot, "SHOW_LEGEND", show_legend), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("scipy.stats.mannwhitneyu", side_effect=AssertionError("No retesting")):
            paper_plot.plot_plasticity_share()
        return save

    def test_draws_cells_medians_clipped_markers_and_saved_p(self):
        cache = self.cache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, cache)
            save = self.render(root)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_plasticity_share.png")
        self.assertIn("2 beyond y-limits", save.call_args.kwargs["extra"])
        axis = figure.axes[0]
        labels = [label.get_text() for label in axis.get_xticklabels()]
        self.assertEqual(labels, ["DelayPro", "ReactPro", "MemoryPro", "ReactCategoryPro"])
        self.assertEqual(axis.get_ylim(), paper_plot._PLASTICITY_SHARE_YLIM)
        low, high = paper_plot._PLASTICITY_SHARE_YLIM
        offsets = np.concatenate([collection.get_offsets() for collection in axis.collections])
        y_values = offsets[:, 1]
        # 2 + 1 + 3 + 3 cells plus 4 medians; clipped cells sit exactly on the limits
        self.assertEqual(len(y_values), 9 + 4)
        self.assertEqual(int(np.sum(y_values == high)), 1)
        self.assertEqual(int(np.sum(y_values == low)), 1)
        self.assertTrue(np.all((y_values >= low) & (y_values <= high)))
        medians = cache["task_median_share"]
        for position, value in enumerate(medians):
            self.assertTrue(np.any(np.isclose(offsets, [position, value]).all(axis=1)))
        legend = axis.get_legend()
        self.assertIn("MWU p = 0.013", legend.get_title().get_text())
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["Task median"])
        texts = [text.get_text() for text in axis.texts]
        self.assertIn("No working memory", texts)
        self.assertIn("Working memory", texts)
        # reference lines at 0 and 1 plus the family divider
        self.assertEqual(len(axis.lines), 3)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["plasticity_share"],
                      paper_plot.plot_plasticity_share)

    def test_no_legend_mode_and_missing_or_incompatible_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, self.cache())
            save = self.render(root, show_legend=False)
            figure = save.call_args.args[0]
            self.addCleanup(paper_plot.plt.close, figure)
            self.assertIsNone(figure.axes[0].get_legend())
        for problem in ("missing", "wrong_type", "bad_median", "no_cells"):
            cache = self.cache()
            if problem == "wrong_type":
                cache["mod_type"] = "modulation_all_weighted_unnormalized"
            elif problem == "bad_median":
                cache["task_median_share"][0] = 0.99
            elif problem == "no_cells":
                cache["share"][:] = np.nan
                cache["task_median_share"][:] = np.nan
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if problem != "missing":
                    self.write(root, cache)
                self.render(root).assert_not_called()


class PlasticityShareSeedsTests(PlasticityShareFigureTests):
    def test_seed_summary_joins_runs_and_marks_significance(self):
        import re
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            root.mkdir()
            specs = {11: (0.2, 0.9, 0.01), 22: (0.5, 0.4, 0.6), 33: (0.1, 0.8, 0.02)}
            for seed, (no_memory, memory, p_value) in specs.items():
                aname = re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME)
                cache = self.cache()
                cache["task_median_share"] = np.array([no_memory, no_memory, memory, memory])
                cache["share"] = np.array([[no_memory] * 3, [no_memory] * 3,
                                           [memory] * 3, [memory] * 3])
                cache["mwu_p_one_sided"] = p_value
                run_dir = base / aname
                run_dir.mkdir()
                with (run_dir / f"plasticity_share_var-weighted-unnormalized_{aname}.pkl").open("wb") as f:
                    pickle.dump(cache, f)
            with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_plasticity_share_seeds()
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_plasticity_share_seeds.png")
        self.assertIn("memory > no memory in 2/3 runs", save.call_args.kwargs["extra"])
        axis = figure.axes[0]
        points = [c for c in axis.collections
                  if isinstance(c, paper_plot.mpl.collections.PathCollection)]
        self.assertEqual(len(points), 3)   # one scatter (two markers) per run
        filled = sum(np.allclose(c.get_facecolors()[0][:3], paper_plot.mpl.colors.to_rgb("#3182ce"))
                     for c in points)
        self.assertEqual(filled, 2)
        self.assertEqual(len(axis.lines), 2 + 3)   # reference lines at 0 and 1, one join per run
        medians = [c for c in axis.collections
                   if isinstance(c, paper_plot.mpl.collections.LineCollection)]
        values = sorted(float(seg[0][1]) for c in medians for seg in c.get_segments())
        self.assertEqual(values, [0.2, 0.8])
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["No\nworking memory", "Working\nmemory"])
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["plasticity_share_seeds"],
                      paper_plot.plot_plasticity_share_seeds)

    def test_seed_summary_needs_two_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / paper_plot.LESION_ANAME
            root.mkdir()
            self.write(root, self.cache())
            with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_plasticity_share_seeds()
            save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
