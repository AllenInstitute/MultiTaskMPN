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
        # Neuron sides carry neuron_variant and the activity definitions, the
        # modulation side its clustering, lesion mode and synapse definitions.
        return {
            "schema_version": 2,
            **paper_plot._CAUSAL_VS_ACTIVITY_CACHE_FIELDS[side],
            "aname": aname, "side": side, "tasks": tasks,
            "activity_similarity": activity, "causal_similarity": causal,
            "activity_pairs": activity[iu], "causal_pairs": causal[iu],
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
            np.testing.assert_allclose(figures[name].get_size_inches(), paper_plot._LESION_SCATTER_FIGSIZE)
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

    def seeds_artists(self, axis):
        """Points (x, y, filled), hidden-input joins, median dashes, IQR bars,
        the bracket and the divider of the seeds figure."""
        mpl = paper_plot.mpl
        grey = mpl.colors.to_rgb(paper_plot._LESION_POINT_COLOR)
        points = []
        for collection in axis.collections:
            if not isinstance(collection, mpl.collections.PathCollection):
                continue
            offsets = collection.get_offsets()
            if not len(offsets):
                continue
            filled = np.allclose(collection.get_facecolors()[0][:3], grey)
            if not filled:     # hollow: white face, grey ring, never black
                np.testing.assert_allclose(collection.get_facecolors()[0][:3], (1., 1., 1.))
                np.testing.assert_allclose(collection.get_edgecolors()[0][:3], grey)
            points.extend((float(x), float(y), filled) for x, y in offsets)
        segments = [segment for collection in axis.collections
                    if isinstance(collection, mpl.collections.LineCollection)
                    for segment in collection.get_segments()]
        join_grey = mpl.colors.to_rgb("0.85")
        return {
            "points": points,
            "joins": [line for line in axis.lines if mpl.colors.to_rgb(line.get_color()) == join_grey],
            # A dash spans x; an IQR bar does not (and may be a single point).
            "medians": sorted(float(s[0][1]) for s in segments if not np.isclose(s[0][0], s[1][0])),
            "iqr": sorted((float(s[0][0]), float(s[0][1]), float(s[1][1])) for s in segments
                          if np.isclose(s[0][0], s[1][0])),
            "brackets": [line for line in axis.lines if len(line.get_xdata()) == 4],
            "dividers": [line for line in axis.lines if line.get_linestyle() == ":"],
        }

    def test_seed_summary_pairs_hidden_with_input_and_marks_significance(self):
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
        self.assertEqual(set(figures), {"multitask_causal_vs_activity_seeds.png",
                                        "multitask_causal_vs_activity_seeds_legend.png"})
        axis = figures["multitask_causal_vs_activity_seeds.png"].axes[0]
        artists = self.seeds_artists(axis)
        jitter = paper_plot._CAUSAL_SEEDS_JITTER
        columns = {"hidden": [0.3, 0.2, 0.4], "input": [-0.1, 0.05, -0.2]}
        for side, values in columns.items():
            center = paper_plot._CAUSAL_SEEDS_POSITIONS[side]
            ys = sorted(y for x, y, _ in artists["points"] if abs(x - center) <= jitter)
            self.assertEqual(ys, sorted(values))
        # filled: hidden 11 and 33, input 33
        self.assertEqual(sorted(y for _, y, filled in artists["points"] if filled), [-0.2, 0.3, 0.4])
        # Plain column names; significance is carried by the marker fill.
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["Hidden", "Input"])
        # One join per run, from its hidden to its input point.
        self.assertEqual(len(artists["joins"]), 3)
        self.assertEqual(sorted(tuple(line.get_ydata()) for line in artists["joins"]),
                         [(0.2, 0.05), (0.3, -0.1), (0.4, -0.2)])
        self.assertEqual(artists["medians"], [-0.1, 0.3])
        self.assertEqual([center for center, _, _ in artists["iqr"]], [0.0, 1.0])
        # The bracket spans hidden to input; with three runs the paired
        # Wilcoxon p (hidden minus input) is 0.25, the smallest two-sided value.
        self.assertEqual(len(artists["brackets"]), 1)
        np.testing.assert_allclose(artists["brackets"][0].get_xdata(), [0., 0., 1., 1.])
        self.assertEqual([text.get_text() for text in axis.texts], ["p = 0.250"])
        self.assertEqual(artists["dividers"], [])
        # The legend is its own figure; the panel carries none, nor a run count.
        self.assertIsNone(axis.get_legend())
        legend = figures["multitask_causal_vs_activity_seeds_legend.png"].legends[0]
        self.assertEqual([text.get_text() for text in legend.get_texts()],
                         ["permutation p < 0.05", r"permutation p $\geq$ 0.05", "median, IQR"])
        self.assertEqual(legend.get_title().get_text(), "")

    def test_seed_panel_matches_the_scatter_box_and_drops_its_legend_without_legends(self):
        """The seeds panel is drawn in the square box the single-run scatter gets,
        so the two sit side by side at one scale; legends off, no legend figure."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            for seed in (11, 22):
                aname = re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME)
                for side in ("hidden", "input"):
                    self.write(base / aname, self.cache(side, aname=aname))
            self.write(root, self.cache("hidden"))
            seeds = self.render(root, paper_plot.plot_causal_vs_activity_seeds)
            scatter = self.render(root, paper_plot.plot_causal_vs_activity)
            bare = self.render(root, paper_plot.plot_causal_vs_activity_seeds, show_legend=False)

        def box(figure):
            figure.canvas.draw()
            extent = figure.axes[0].get_window_extent().transformed(
                figure.dpi_scale_trans.inverted())
            return extent.width, extent.height

        panel = seeds["multitask_causal_vs_activity_seeds.png"]
        reference = scatter["multitask_causal_vs_activity_tasks_hidden.png"]
        np.testing.assert_allclose(panel.get_size_inches(), reference.get_size_inches())
        np.testing.assert_allclose(box(panel), box(reference), atol=0.01)
        self.assertEqual(set(bare), {"multitask_causal_vs_activity_seeds.png"})

    def test_modulation_side_gets_its_own_figure_and_rejects_another_lesion_mode(self):
        modulation = self.cache("modulation", rho=0.3, p_perm=0.01)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / paper_plot.LESION_ANAME
            self.write(root, self.cache("hidden"))
            self.write(root, modulation)
            figures = self.render(root, paper_plot.plot_causal_vs_activity)
        self.assertEqual(set(figures), {"multitask_causal_vs_activity_tasks_hidden.png",
                                        "multitask_causal_vs_activity_tasks_modulation.png"})
        axis = figures["multitask_causal_vs_activity_tasks_modulation.png"].axes[0]
        np.testing.assert_allclose(axis.collections[0].get_offsets(),
                                   np.column_stack((modulation["activity_pairs"],
                                                    modulation["causal_pairs"])))
        self.assertIn("modulation", axis.get_xlabel())
        for field, value in (("mod_lesion_mode", "zero_W"), ("neuron_variant", "unnormalized"),
                             ("x_definition", paper_plot._CAUSAL_VS_ACTIVITY_X_DEFINITION)):
            broken = self.cache("modulation")
            broken[field] = value
            if field == "neuron_variant":
                del broken["modulation_type"]
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / paper_plot.LESION_ANAME
                self.write(root, broken)
                self.assertEqual(self.render(root, paper_plot.plot_causal_vs_activity), {})

    def test_seed_summary_adds_modulation_for_the_runs_that_have_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            runs = {re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME): modulation
                    for seed, modulation in ((11, (0.5, 0.01)), (22, (0.1, 0.4)), (33, None))}
            for aname, modulation in runs.items():
                self.write(base / aname, self.cache("hidden", aname=aname))
                self.write(base / aname, self.cache("input", aname=aname, rho=-0.1, p_perm=0.6))
                if modulation is not None:
                    self.write(base / aname, self.cache("modulation", aname=aname,
                                                        rho=modulation[0], p_perm=modulation[1]))
            # modulation alone does not make a run
            lone = re.sub(r"seed\d+", "seed44", paper_plot.LESION_ANAME)
            self.write(base / lone, self.cache("modulation", aname=lone))
            root.mkdir(exist_ok=True)
            figures = self.render(root, paper_plot.plot_causal_vs_activity_seeds)
        axis = figures["multitask_causal_vs_activity_seeds.png"].axes[0]
        artists = self.seeds_artists(axis)
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["Hidden", "Input", "Modulation"])
        self.assertEqual(len(artists["points"]), 3 * 2 + 2)
        center = paper_plot._CAUSAL_SEEDS_POSITIONS["modulation"]
        self.assertEqual(sorted(y for x, y, _ in artists["points"]
                                if abs(x - center) <= paper_plot._CAUSAL_SEEDS_JITTER), [0.1, 0.5])
        # Modulation is set apart and never joined to the neuron sides.
        self.assertEqual(len(artists["joins"]), 3)
        self.assertTrue(all(max(line.get_xdata()) < paper_plot._CAUSAL_SEEDS_DIVIDER
                            for line in artists["joins"]))
        self.assertEqual(len(artists["dividers"]), 1)
        self.assertEqual(artists["dividers"][0].get_xdata()[0], paper_plot._CAUSAL_SEEDS_DIVIDER)
        self.assertEqual(artists["medians"], [-0.1, 0.3, 0.4])   # input, modulation, hidden

    def test_seed_summary_needs_two_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / paper_plot.LESION_ANAME
            self.write(root, self.cache("hidden"))
            self.write(root, self.cache("input"))
            self.assertEqual(self.render(root, paper_plot.plot_causal_vs_activity_seeds), {})


if __name__ == "__main__":
    unittest.main()
