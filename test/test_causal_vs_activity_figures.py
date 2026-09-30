"""Task-level causal-vs-activity paper figures redraw lesion_plot's caches only."""

import pickle
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


def association(rho, p_perm, n_tasks=5):
    return {"statistic": "spearman", "rho": rho, "p_perm": p_perm, "n_clusters": n_tasks,
            "permutation_unit": "cluster_label", "side": "two-sided", "n_perm": 10}


class CausalVsActivityFigureTests(unittest.TestCase):
    def cache(self, side, aname=paper_plot.LESION_ANAME, rho=0.4, p_perm=0.02, trend=True):
        tasks = ["fdgo", "reactgo", "delaygo", "dmcgo", "dmsgo"]
        rng = np.random.default_rng(1 if side == "hidden" else 2)
        activity = np.corrcoef(rng.random((5, 8)))
        causal = np.corrcoef(rng.random((5, 8)))
        iu = np.triu_indices(5, k=1)
        return {
            "schema_version": 2, "neuron_variant": "unnormalized",
            "aname": aname, "side": side, "tasks": tasks,
            "activity_similarity": activity, "causal_similarity": causal,
            "activity_pairs": activity[iu], "causal_pairs": causal[iu],
            "x_definition": paper_plot._CAUSAL_VS_ACTIVITY_X_DEFINITION,
            "y_definition": paper_plot._CAUSAL_VS_ACTIVITY_Y_DEFINITION,
            "association": association(rho, p_perm),
            "trend_line": {"method": "ols", "slope": 0.5, "intercept": 0.1} if trend else None,
            "pearson_mantel_one_sided": {"r": 0.3, "p_perm": 0.04, "n_perm": 10},
        }

    def write(self, directory, cache):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"causal_vs_activity_tasksim_{cache['side']}_{cache['aname']}.pkl").open("wb") as f:
            pickle.dump(cache, f)

    def render(self, root, function, show_legend=True):
        with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                patch.object(paper_plot, "SHOW_LEGEND", show_legend), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("scipy.stats.spearmanr", side_effect=AssertionError("No refitting")):
            function()
        figures = {call.args[1].name: call.args[0] for call in save.call_args_list}
        for figure in figures.values():
            self.addCleanup(paper_plot.plt.close, figure)
        return figures

    def test_per_run_figures_draw_saved_pairs_line_and_rank_label(self):
        hidden = self.cache("hidden")
        inputs = self.cache("input", rho=-0.1, p_perm=0.6, trend=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / paper_plot.LESION_ANAME
            self.write(root, hidden)
            self.write(root, inputs)
            figures = self.render(root, paper_plot.plot_causal_vs_activity)
        self.assertEqual(set(figures), {"multitask_causal_vs_activity_tasks_hidden.png",
                                        "multitask_causal_vs_activity_tasks_input.png"})
        for name, cache, n_lines in (("multitask_causal_vs_activity_tasks_hidden.png", hidden, 1),
                                     ("multitask_causal_vs_activity_tasks_input.png", inputs, 0)):
            axis = figures[name].axes[0]
            np.testing.assert_allclose(axis.collections[0].get_offsets(),
                                       np.column_stack((cache["activity_pairs"], cache["causal_pairs"])))
            self.assertEqual(len(axis.lines), n_lines)
            if n_lines:
                np.testing.assert_allclose(axis.lines[0].get_ydata(), 0.1 + 0.5 * axis.lines[0].get_xdata())
            legend = axis.get_legend().get_texts()[0].get_text()
            self.assertIn(f"= {cache['association']['rho']:.2f}", legend)
            self.assertIn("(n = 5 tasks)", legend)
            self.assertIn("lesion profiles", axis.get_ylabel())
            np.testing.assert_allclose(figures[name].get_size_inches(), paper_plot._CLUSTER_CORR_FIGSIZE)
        self.assertIn("hidden activity", figures["multitask_causal_vs_activity_tasks_hidden.png"].axes[0].get_xlabel())

    def test_per_run_figure_rejects_other_run_side_or_legacy_cache(self):
        for problem in ("aname", "side", "schema", "variant", "pearson", "pairs"):
            cache = self.cache("hidden")
            if problem == "aname":
                cache["aname"] = "another_run"
            elif problem == "side":
                cache["side"] = "input"
            elif problem == "schema":
                cache["schema_version"] = 1
            elif problem == "variant":
                cache["neuron_variant"] = "normalized"
            elif problem == "pearson":
                cache["association"]["statistic"] = "pearson"
            else:
                cache["activity_pairs"] = cache["activity_pairs"][:-1]
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / paper_plot.LESION_ANAME
                cache_side = "hidden"
                cache["side"] = cache["side"]
                directory_path = Path(root)
                directory_path.mkdir(parents=True)
                with (directory_path / f"causal_vs_activity_tasksim_{cache_side}_{paper_plot.LESION_ANAME}.pkl").open("wb") as f:
                    pickle.dump(cache, f)
                self.assertEqual(self.render(root, paper_plot.plot_causal_vs_activity), {})

    def test_seed_summary_joins_runs_and_marks_significance(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            runs = {re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME): values
                    for seed, values in ((11, (0.3, 0.01, -0.1, 0.5)),
                                         (22, (0.2, 0.2, 0.05, 0.7)),
                                         (33, (0.4, 0.001, -0.2, 0.04)))}
            for aname, (rho_h, p_h, rho_i, p_i) in runs.items():
                self.write(base / aname, self.cache("hidden", aname=aname, rho=rho_h, p_perm=p_h))
                self.write(base / aname, self.cache("input", aname=aname, rho=rho_i, p_perm=p_i))
            # a run missing its input cache is ignored
            partial = re.sub(r"seed\d+", "seed44", paper_plot.LESION_ANAME)
            self.write(base / partial, self.cache("hidden", aname=partial))
            root.mkdir(exist_ok=True)
            figures = self.render(root, paper_plot.plot_causal_vs_activity_seeds)
        self.assertEqual(set(figures), {"multitask_causal_vs_activity_seeds.png"})
        axis = figures["multitask_causal_vs_activity_seeds.png"].axes[0]
        points = [collection for collection in axis.collections
                  if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
        medians = [collection for collection in axis.collections
                   if isinstance(collection, paper_plot.mpl.collections.LineCollection)]
        self.assertEqual(len(points), 6)   # 3 runs x 2 sides
        filled = sum(np.allclose(collection.get_facecolors()[0][:3],
                                 paper_plot.mpl.colors.to_rgb("#3182ce"))
                     for collection in points)
        self.assertEqual(filled, 3)        # hidden 11, hidden 33, input 33
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["Hidden activity", "Input activity"])
        self.assertEqual(len(axis.lines), 1 + 3)   # zero line + one join per run
        # medians: hidden 0.3, input -0.1 drawn as hlines
        hline_values = sorted(float(segment[0][1]) for collection in medians
                              for segment in collection.get_segments())
        self.assertEqual(hline_values, [-0.1, 0.3])
        self.assertIn("3 runs", axis.get_legend().get_title().get_text())

    def test_seed_summary_needs_two_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / paper_plot.LESION_ANAME
            self.write(root, self.cache("hidden"))
            self.write(root, self.cache("input"))
            self.assertEqual(self.render(root, paper_plot.plot_causal_vs_activity_seeds), {})


if __name__ == "__main__":
    unittest.main()
