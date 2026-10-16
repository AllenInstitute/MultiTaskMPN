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

    def test_draws_cells_medians_and_clipped_markers_without_a_test(self):
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
        # The figure shows the values only: the cache's Mann-Whitney U test is
        # neither drawn nor reported.
        self.assertNotIn("MWU", save.call_args.kwargs["extra"])
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
        self.assertEqual(legend.get_title().get_text(), "")
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


class RemovedSeedsFigureTests(unittest.TestCase):
    def test_plasticity_share_seeds_figure_is_gone(self):
        self.assertNotIn("plasticity_share_seeds", paper_plot.FIGURES_BY_MODE["lesion"])
        self.assertFalse(hasattr(paper_plot, "plot_plasticity_share_seeds"))


class PlasticityShareCohortTests(PlasticityShareFigureTests):
    """The overlay of every sibling run's saved cells and task medians."""

    def write_runs(self, base, specs):
        """One sibling run directory per (seed, share matrix) in `specs`."""
        import re
        for seed, share in specs.items():
            aname = re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME)
            cache = self.cache()
            cache["share"] = np.asarray(share, dtype=float)
            cache["sig_zero_w"] = np.isfinite(cache["share"])
            cache["task_median_share"] = np.array([np.nanmedian(row) if np.isfinite(row).any()
                                                   else np.nan for row in cache["share"]])
            run_dir = base / aname
            run_dir.mkdir()
            with (run_dir / f"plasticity_share_var-weighted-unnormalized_{aname}.pkl").open("wb") as f:
                pickle.dump(cache, f)

    def render_cohort(self, root):
        with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("scipy.stats.mannwhitneyu", side_effect=AssertionError("No retesting")):
            paper_plot.plot_plasticity_share_cohort()
        return save

    def test_cohort_overlays_cells_run_medians_and_pooled_medians(self):
        nan = np.nan
        specs = {
            11: [[0.1, 0.3, nan], [0.2, nan, nan], [0.9, 1.1, nan], [0.8, 1.0, nan]],
            22: [[0.5, nan, nan], [0.4, 0.6, nan], [1.3, nan, nan], [nan, nan, nan]],
            33: [[0.0, 0.2, 0.4], [nan, nan, nan], [2.5, 0.7, nan], [0.6, nan, nan]],
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            root.mkdir()
            self.write_runs(base, specs)
            save = self.render_cohort(root)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_plasticity_share_cohort.png")
        self.assertIn("3 runs, 15 cells, 1 beyond y-limits", save.call_args.kwargs["extra"])
        axis = figure.axes[0]
        collections = paper_plot.mpl.collections
        points = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
        offsets = np.concatenate([c.get_offsets() for c in points])
        # 15 pooled cells (the 2.5 clipped to the upper limit) plus 4 pooled medians.
        self.assertEqual(len(offsets), 15 + 4)
        high = paper_plot._PLASTICITY_SHARE_YLIM[1]
        self.assertEqual(int(np.sum(offsets[:, 1] == high)), 1)
        # Task order: reaction tasks first (fdgo, reactgo), then delaygo, dmcgo.
        pooled = {0: np.median([0.1, 0.3, 0.5, 0.0, 0.2, 0.4]), 1: np.median([0.2, 0.4, 0.6]),
                  2: np.median([0.9, 1.1, 1.3, 2.5, 0.7]), 3: np.median([0.8, 1.0, 0.6])}
        for position, value in pooled.items():
            self.assertTrue(np.any(np.isclose(offsets, [position, value]).all(axis=1)))
        # One gray dash per run that has cells for the task: 3, 2, 3, 2.
        dashes = [c for c in axis.collections if isinstance(c, collections.LineCollection)]
        counts = [len(c.get_segments()) for c in dashes]
        self.assertEqual(counts, [3, 2, 3, 2])
        legend = axis.get_legend()
        self.assertEqual(legend.get_title().get_text(), "3 runs")
        self.assertEqual([text.get_text() for text in legend.get_texts()],
                         ["Pooled median", "Run median"])
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["plasticity_share_cohort"],
                      paper_plot.plot_plasticity_share_cohort)

    def test_cohort_needs_two_runs_and_drops_runs_with_other_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / paper_plot.LESION_ANAME
            root.mkdir()
            self.write(root, self.cache())
            self.render_cohort(root).assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            root.mkdir()
            self.write_runs(base, {11: self.cache()["share"], 22: self.cache()["share"]})
            other = base / paper_plot.LESION_ANAME.replace("seed", "seed9")
            other.mkdir()
            cache = self.cache()
            cache["tasks"] = ["fdgo", "reactgo", "delaygo", "dmsgo"]
            with (other / f"plasticity_share_var-weighted-unnormalized_{other.name}.pkl").open("wb") as f:
                pickle.dump(cache, f)
            with patch("builtins.print") as output:
                save = self.render_cohort(root)
        save.assert_called_once()
        self.assertIn("2 runs", save.call_args.kwargs["extra"])
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("task list differs", printed)


if __name__ == "__main__":
    unittest.main()
