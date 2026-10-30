"""The task-specificity paper figures redraw lesion_plot's saved statistics only."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


class TaskSpecificityFigureTests(unittest.TestCase):
    def sharing(self, relations, means):
        by_relation = {}
        for name, mean in zip(relations, means):
            values = np.array([mean - 0.1, mean, mean + 0.1])
            by_relation[name] = {"pairs": [("a", "b")] * 3, "values": values,
                                 "mean": mean, "p_perm": 0.5 if name == "other" else 0.02,
                                 "null_mean": np.zeros(4)}
        by_relation["related"] = {"pairs": [], "values": np.array([0.4, 0.5]), "mean": 0.45,
                                  "p_perm": 0.004, "null_mean": np.zeros(4)}
        return {"relations": relations, "by_relation": by_relation, "permutation_unit": "task_label",
                "side": "greater", "n_perm": 4, "seed": 0}

    # Null distributions of one cluster's count over the 6 tasks (0..6).
    HIDDEN_PMF = np.array([0.1, 0.3, 0.3, 0.2, 0.1, 0.0, 0.0])
    MODULATION_PMF = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0])

    THRESHOLDS = (0.02, 0.05, 0.1, 0.2)

    def robustness(self, counts, sizes, rho_significant, rho_threshold):
        """The size and fixed-threshold fields of one type: the counts at
        every threshold (the 0.05 one `counts`, the others shifted) and the
        size correlations (`rho_threshold` at 0.05)."""
        counts = np.asarray(counts)
        threshold_counts = {thr: np.clip(counts + k - 1, 0, 6)
                            for k, thr in enumerate(self.THRESHOLDS)}
        correlation = {"significant": rho_significant}
        correlation.update({thr: rho_threshold - 0.1 * (k - 1)
                            for k, thr in enumerate(self.THRESHOLDS)})
        return {"cluster_sizes": np.asarray(sizes, dtype=float),
                "threshold_counts": threshold_counts, "size_correlation": correlation,
                "effect": np.zeros((6, counts.size))}

    def cache(self, pools=None, control_pool="all"):
        """A schema-4 cache; `pools` adds counts under other control pools
        ({pool: {type: counts}}) beside the "all" pool's, which are the
        significance counts."""
        tasks = [f"t{k}" for k in range(6)]
        relations = ["response rule", "timing", "other"]
        types = {
            "hidden": {"cluster_labels": ["h1", "h2", "h3", "h4"],
                       "dispersion": {"counts": np.array([0, 1, 1, 5]), "p_perm": 0.01,
                                      "null_var": np.array([0.5, 1.0, 1.5, 1.0]),
                                      "null_count_pmf": self.HIDDEN_PMF.copy(),
                                      "side": "greater"},
                       "sharing": self.sharing(relations, [0.6, 0.5, 0.2]),
                       **self.robustness([1, 2, 2, 4], [3, 1, 5, 20], -0.4, -0.1)},
            "modulation": {"cluster_labels": ["c1", "c2", "c3"],
                           "dispersion": {"counts": np.array([2, 2, 2]), "p_perm": 0.7,
                                          "null_var": np.full(4, 0.25),
                                          "null_count_pmf": self.MODULATION_PMF.copy(),
                                          "side": "greater"},
                           "sharing": self.sharing(relations, [0.3, 0.3, 0.3]),
                           **self.robustness([2, 3, 1], [50, 200, 1000], -0.3, 0.05)},
        }
        pool_counts = {"all": {name: entry["dispersion"]["counts"].copy()
                                for name, entry in types.items()}}
        for pool, by_type in (pools or {}).items():
            pool_counts[pool] = {name: np.asarray(counts) for name, counts in by_type.items()}
        return {"schema_version": 4, "aname": paper_plot.LESION_ANAME, "tasks": tasks,
                "n_perm": 4, "modulation_lesion_mode": "freeze_M",
                "effect_thresholds": list(self.THRESHOLDS), "types": types,
                "control_pool": control_pool, "pool_counts": pool_counts}

    def render(self, cache, show_legend=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if cache is not None:
                with (root / f"task_specificity_{paper_plot.LESION_ANAME}.pkl").open("wb") as f:
                    pickle.dump(cache, f)
            with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                    patch.object(paper_plot, "SHOW_LEGEND", show_legend), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_task_specificity()
        figures = {call.args[1].name: call.args[0] for call in save.call_args_list}
        for figure in figures.values():
            self.addCleanup(paper_plot.plt.close, figure)
        return figures

    def test_counts_figure_uses_saved_values(self):
        import lesion_plot

        figures = self.render(self.cache())
        self.assertEqual(set(figures), {"multitask_task_specificity_counts.png"})
        figure = figures["multitask_task_specificity_counts.png"]
        # One row per saved type (hidden, modulation), titled with the type
        # and its cluster count on the left and the saved dispersion p on the
        # right; the rows share both axes.
        hidden_axis, synapse_axis = figure.axes
        self.assertEqual([axis.get_title(loc="left") for axis in figure.axes],
                         ["Hidden clusters (n = 4)", "Synapse clusters (n = 3)"])
        self.assertEqual([axis.get_title(loc="right") for axis in figure.axes],
                         ["p = 0.010", "p = 0.700"])
        self.assertEqual(hidden_axis.get_xlim(), synapse_axis.get_xlim())
        self.assertEqual(hidden_axis.get_ylim(), synapse_axis.get_ylim())
        # Every cluster is one dot stacked over its count, one cluster per
        # unit of height: hidden 0, 1, 1, 5 and modulation 2, 2, 2.
        collections = paper_plot.mpl.collections
        for axis, expected in ((hidden_axis, [(0, 0.5), (1, 0.5), (1, 1.5), (5, 0.5)]),
                               (synapse_axis, [(2, 0.5), (2, 1.5), (2, 2.5)])):
            dots = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
            offsets = np.concatenate([c.get_offsets() for c in dots])
            self.assertEqual(sorted(map(tuple, offsets.tolist())), expected)
        # The gray histogram is the expected number of clusters per count
        # under the saved null: clusters x the saved count distribution.
        for axis, n_clusters, pmf in ((hidden_axis, 4, self.HIDDEN_PMF),
                                      (synapse_axis, 3, self.MODULATION_PMF)):
            steps = [patch for patch in axis.patches
                     if isinstance(patch, paper_plot.mpl.patches.StepPatch)]
            self.assertEqual(len(steps), 1)
            np.testing.assert_allclose(steps[0].get_data().values, n_clusters * pmf)
            np.testing.assert_allclose(steps[0].get_data().edges, np.arange(8) - 0.5)
        # The red median line: hidden median 1, modulation 2.
        for axis, median in ((hidden_axis, 1.0), (synapse_axis, 2.0)):
            lines = [line for line in axis.get_lines()
                     if line.get_color() == paper_plot._TREND_LINE_COLOR]
            self.assertEqual([line.get_xdata()[0] for line in lines], [median])
        self.assertEqual(synapse_axis.get_xlabel(), "Tasks impaired per cluster")
        self.assertEqual(synapse_axis.get_ylabel(), "Clusters")
        self.assertGreaterEqual(hidden_axis.get_ylim()[1], 3)
        legend_texts = [text.get_text() for text in figure.legends[0].get_texts()]
        self.assertEqual(legend_texts, ["cluster", "mixing null, expected", "median"])
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_specificity"],
                      paper_plot.plot_task_specificity)
        # The synapse clusters are tested with the lesion heatmap's mode.
        self.assertEqual(lesion_plot.TASK_SPECIFICITY_MOD_MODE, paper_plot.LESION_HEATMAP_MODE)
        self.assertEqual(lesion_plot.TASK_SPECIFICITY_SCHEMA_VERSION,
                         paper_plot._TASK_SPECIFICITY_SCHEMA_VERSION)

    def test_counts_do_not_depend_on_the_sharing_statistics(self):
        cache = self.cache()
        for entry in cache["types"].values():
            entry.pop("sharing")
        figures = self.render(cache)
        self.assertEqual(set(figures), {"multitask_task_specificity_counts.png"})
        self.assertEqual(len(figures["multitask_task_specificity_counts.png"].axes), 2)

    def test_no_legend_and_incompatible_caches(self):
        figures = self.render(self.cache(), show_legend=False)
        for figure in figures.values():
            self.assertEqual(figure.legends, [])
        for problem in ("missing", "run", "schema", "mode", "type", "counts", "null", "pmf",
                        "threshold", "sizes", "rho", "pool", "pool_shape", "pool_all"):
            cache = self.cache()
            if problem == "missing":
                cache = None
            elif problem == "run":
                cache["aname"] = "other"
            elif problem == "schema":
                cache["schema_version"] = 2
            elif problem == "mode":
                cache["modulation_lesion_mode"] = "zero_W"
            elif problem == "null":
                cache["types"]["hidden"]["dispersion"].pop("null_count_pmf")
            elif problem == "pmf":
                cache["types"]["hidden"]["dispersion"]["null_count_pmf"] = self.HIDDEN_PMF[:-1]
            elif problem == "type":
                cache["types"]["synapse"] = cache["types"].pop("modulation")
            elif problem == "threshold":     # no counts at the figure's threshold
                cache["types"]["hidden"]["threshold_counts"].pop(0.05)
            elif problem == "sizes":
                cache["types"]["hidden"]["cluster_sizes"] = np.array([3.0, 1.0, 5.0])
            elif problem == "rho":
                cache["types"]["hidden"]["size_correlation"].pop("significant")
            elif problem == "pool":          # schema 4 records the control pool
                cache.pop("control_pool")
            elif problem == "pool_shape":
                cache["pool_counts"]["all"]["hidden"] = np.array([0, 1])
            elif problem == "pool_all":      # the all pool's counts are always saved
                cache["pool_counts"] = {"responsive": cache["pool_counts"]["all"]}
            else:
                cache["types"]["hidden"]["dispersion"]["counts"] = np.array([0, 9, 1, 1])
            with self.subTest(problem=problem):
                self.assertEqual(self.render(cache), {})


class TaskSpecificitySeedsTests(TaskSpecificityFigureTests):
    """The median impaired-task count of every sibling run, per cluster type."""

    def write_run(self, base, seed, hidden_counts, modulation_counts, p_hidden=0.01,
                  threshold=None, rho=None, pools=None, control_pool="all"):
        """Cache one run: the significance counts per type, and optionally
        `threshold` ({type: counts at 0.05}), `rho` ({type: (significant,
        threshold) size correlations}) and `pools` ({pool: {type: counts}}
        under other control pools)."""
        import re
        aname = re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME)
        cache = self.cache(pools=pools, control_pool=control_pool)
        cache["aname"] = aname
        cache["types"]["hidden"]["dispersion"]["counts"] = np.array(hidden_counts)
        cache["types"]["hidden"]["dispersion"]["p_perm"] = p_hidden
        cache["types"]["modulation"]["dispersion"]["counts"] = np.array(modulation_counts)
        cache["pool_counts"]["all"] = {"hidden": np.array(hidden_counts),
                                       "modulation": np.array(modulation_counts)}
        for type_name, counts in (("hidden", hidden_counts), ("modulation", modulation_counts)):
            entry = cache["types"][type_name]
            sizes = np.arange(1, len(counts) + 1, dtype=float)
            entry.update(self.robustness(
                (threshold or {}).get(type_name, counts), sizes,
                *(rho or {}).get(type_name, (-0.5, 0.0))))
        run_dir = base / aname
        run_dir.mkdir()
        with (run_dir / f"task_specificity_{aname}.pkl").open("wb") as f:
            pickle.dump(cache, f)
        return run_dir

    def render_seeds(self, root, function=None):
        with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("builtins.print") as output:
            (function or paper_plot.plot_task_specificity_seeds)()
        return save, output

    def test_seeds_draw_each_runs_median_count_per_type(self):
        from scipy.stats import wilcoxon

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6], p_hidden=0.01)
            self.write_run(base, 22, [1, 2, 3], [3, 5, 5], p_hidden=0.30)
            self.write_run(base, 33, [1, 1, 2], [5, 5, 6], p_hidden=0.01)
            save, _ = self.render_seeds(root)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_task_specificity_seeds.png")
        axis = figure.axes[0]
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["Hidden", "Synapse"])
        collections = paper_plot.mpl.collections
        dots = [c for c in axis.collections
                if isinstance(c, collections.PathCollection) and len(c.get_offsets())]
        offsets = np.concatenate([c.get_offsets() for c in dots])
        # Run medians: hidden 1, 2, 1 at x = 0 (the tied 1s side by side),
        # modulation 4, 5, 5 at x = 1.
        hidden = offsets[np.isclose(offsets[:, 0], 0, atol=0.3)]
        self.assertEqual(sorted(hidden[:, 1].tolist()), [1, 1, 2])
        tied = hidden[np.isclose(hidden[:, 1], 1), 0]
        self.assertGreater(tied.max() - tied.min(), 0)
        self.assertEqual(sorted(offsets[np.isclose(offsets[:, 0], 1, atol=0.3), 1].tolist()),
                         [4, 5, 5])
        # Hollow (white face) where the saved dispersion p is >= 0.05: the
        # hidden point of seed 22 (p = 0.30) and every modulation point (0.7).
        hollow = [c for c in dots if np.allclose(c.get_facecolors()[0][:3], 1.0)]
        self.assertEqual(sorted(np.concatenate([c.get_offsets() for c in hollow])[:, 1].tolist()),
                         [2, 4, 5, 5])
        # Median dashes over runs, and the bracket's paired Wilcoxon p.
        dashes = [c for c in axis.collections if isinstance(c, collections.LineCollection)
                  and np.isclose(c.get_segments()[0][0, 1], c.get_segments()[0][1, 1])]
        self.assertEqual(sorted(c.get_segments()[0][0, 1] for c in dashes), [1.0, 5.0])
        expected = wilcoxon([4 - 1, 5 - 2, 5 - 1]).pvalue
        self.assertEqual([text.get_text() for text in axis.texts], [f"p = {expected:.3f}"])
        self.assertIn("3 runs", axis.get_legend().get_title().get_text())
        self.assertIn("modulation vs hidden paired Wilcoxon", save.call_args.kwargs["extra"])
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_specificity_seeds"],
                      paper_plot.plot_task_specificity_seeds)

    def test_threshold_seeds_draw_the_fixed_threshold_medians_all_filled(self):
        from scipy.stats import wilcoxon

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6],
                                  threshold={"hidden": [1, 2, 2, 6], "modulation": [4, 6, 6]})
            self.write_run(base, 22, [1, 2, 3], [3, 5, 5],
                           threshold={"hidden": [3, 3, 4], "modulation": [5, 5, 6]})
            self.write_run(base, 33, [1, 1, 2], [5, 5, 6], p_hidden=0.3,
                           threshold={"hidden": [0, 1, 1], "modulation": [6, 6, 6]})
            save, _ = self.render_seeds(root, paper_plot.plot_task_specificity_threshold_seeds)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_task_specificity_threshold_seeds.png")
        axis = figure.axes[0]
        collections = paper_plot.mpl.collections
        dots = [c for c in axis.collections
                if isinstance(c, collections.PathCollection) and len(c.get_offsets())]
        offsets = np.concatenate([c.get_offsets() for c in dots])
        # Run medians of the threshold counts, not of the significance counts:
        # hidden 2, 3, 1; modulation 6, 5, 6. Every point is filled.
        self.assertEqual(sorted(offsets[np.isclose(offsets[:, 0], 0, atol=0.3), 1].tolist()),
                         [1, 2, 3])
        self.assertEqual(sorted(offsets[np.isclose(offsets[:, 0], 1, atol=0.3), 1].tolist()),
                         [5, 6, 6])
        self.assertFalse(any(np.allclose(c.get_facecolors()[0][:3], 1.0) for c in dots))
        expected = wilcoxon([6 - 2, 5 - 3, 6 - 1]).pvalue
        self.assertEqual([text.get_text() for text in axis.texts], [f"p = {expected:.3f}"])
        self.assertIn("effect > 5 pp", axis.get_ylabel())
        legend = axis.get_legend()
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["run", "median, IQR"])
        self.assertIn("3 runs", legend.get_title().get_text())
        extra = save.call_args.kwargs["extra"]
        self.assertIn("hidden: median 1 at > 2 pp, 2 at > 5 pp, 3 at > 10 pp, 4 at > 20 pp", extra)
        self.assertIn("modulation vs hidden paired Wilcoxon", extra)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_specificity_threshold_seeds"],
                      paper_plot.plot_task_specificity_threshold_seeds)
        self.assertEqual(paper_plot._TASK_SPECIFICITY_FIGURE_THRESHOLD, 0.05)
        self.assertIn(paper_plot._TASK_SPECIFICITY_FIGURE_THRESHOLD, self.THRESHOLDS)

    def test_seeds_need_two_runs_and_drop_disagreeing_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6])
            save, output = self.render_seeds(root)
        save.assert_not_called()
        self.assertIn("need >= 2", output.call_args.args[0])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6])
            self.write_run(base, 22, [1, 2, 3], [3, 5, 5])
            # A run of the old zero_W cache is incompatible; one without the
            # synapse clusters disagrees with the others.
            old = self.write_run(base, 33, [1, 1, 2], [5, 5, 6])
            cache = load_pickle(old / f"task_specificity_{old.name}.pkl")
            cache["modulation_lesion_mode"] = "zero_W"
            dump_pickle(cache, old / f"task_specificity_{old.name}.pkl")
            odd = self.write_run(base, 44, [1, 1, 2], [5, 5, 6])
            cache = load_pickle(odd / f"task_specificity_{odd.name}.pkl")
            cache["types"].pop("modulation")
            dump_pickle(cache, odd / f"task_specificity_{odd.name}.pkl")
            save, output = self.render_seeds(root)
        save.assert_called_once()
        self.addCleanup(paper_plot.plt.close, save.call_args.args[0])
        self.assertIn("2 runs",
                      save.call_args.args[0].axes[0].get_legend().get_title().get_text())
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("incompatible", printed)
        self.assertIn("cluster types differ", printed)


class TaskSpecificitySizeTests(TaskSpecificitySeedsTests):
    """The per-run correlation of cluster size with the impaired-task count,
    under the significance and the fixed-threshold rule."""

    RHO = {11: {"hidden": (-0.6, -0.1), "modulation": (-0.3, 0.05)},
           22: {"hidden": (-0.4, 0.1), "modulation": (-0.2, 0.0)},
           33: {"hidden": (-0.5, 0.0), "modulation": (-0.4, 0.1)}}

    def test_size_figure_draws_both_rules_per_type(self):
        from scipy.stats import wilcoxon

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = None
            for seed, rho in self.RHO.items():
                run_dir = self.write_run(base, seed, [0, 1, 1, 5], [2, 4, 6], rho=rho)
                root = root or run_dir
            save, _ = self.render_seeds(root, paper_plot.plot_task_specificity_size)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_task_specificity_size.png")
        axis = figure.axes[0]
        collections = paper_plot.mpl.collections
        dots = [c for c in axis.collections
                if isinstance(c, collections.PathCollection) and len(c.get_offsets())]
        # Two columns per type (significant, threshold) at 0, 1 and 3, 4, each
        # holding the three runs' saved correlations, in the lesion grey.
        self.assertEqual(len(dots), 4)
        expected = {0: [-0.6, -0.5, -0.4], 1: [-0.1, 0.0, 0.1],
                    3: [-0.4, -0.3, -0.2], 4: [0.0, 0.05, 0.1]}
        for position, values in expected.items():
            column = np.concatenate([c.get_offsets() for c in dots
                                     if np.isclose(c.get_offsets()[0][0], position, atol=0.3)])
            np.testing.assert_allclose(sorted(column[:, 1]), values)
        for c in dots:
            np.testing.assert_allclose(
                c.get_facecolors()[0][:3],
                paper_plot.mpl.colors.to_rgb(paper_plot._LESION_POINT_COLOR))
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["Signif.", "> 5 pp"] * 2)
        # Medians per column, the type labels under the groups and one
        # two-sided Wilcoxon p above each column.
        dashes = [c for c in axis.collections if isinstance(c, collections.LineCollection)
                  and np.isclose(c.get_segments()[0][0, 1], c.get_segments()[0][1, 1])]
        np.testing.assert_allclose(sorted(c.get_segments()[0][0, 1] for c in dashes),
                                   [-0.5, -0.3, 0.0, 0.05])
        texts = [text.get_text() for text in axis.texts]
        self.assertEqual(texts[:2], ["Hidden", "Synapse"])
        self.assertEqual(texts[2], paper_plot._short_p(wilcoxon([-0.6, -0.4, -0.5]).pvalue))
        self.assertEqual(len(texts), 6)
        self.assertIn("3 runs", axis.get_legend().get_title().get_text())
        self.assertIn("hidden/Signif.: median rho -0.50, 3/3 < 0",
                      save.call_args.kwargs["extra"])
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_specificity_size"],
                      paper_plot.plot_task_specificity_size)

    def test_size_figure_drops_undefined_correlations_and_needs_two_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6], rho=self.RHO[11])
            self.write_run(base, 22, [1, 2, 3], [3, 5, 5], rho=self.RHO[22])
            # A constant count has no correlation: the run is left out.
            self.write_run(base, 33, [1, 1, 1], [5, 5, 6],
                           rho={"hidden": (np.nan, 0.0), "modulation": (-0.4, 0.1)})
            save, output = self.render_seeds(root, paper_plot.plot_task_specificity_size)
        save.assert_called_once()
        self.addCleanup(paper_plot.plt.close, save.call_args.args[0])
        self.assertIn("2 runs", save.call_args.kwargs["extra"])
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("undefined size correlation", printed)
        self.assertIn("seed33", printed)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6], rho=self.RHO[11])
            save, output = self.render_seeds(root, paper_plot.plot_task_specificity_size)
        save.assert_not_called()
        self.assertIn("need >= 2", output.call_args.args[0])


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path):
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


class TaskSharingCohortTests(TaskSpecificityFigureTests):
    """The per-run excess sharing of impaired clusters, over sibling runs."""

    def run_cache(self, aname, hidden_means, modulation_means, p_rule=0.02):
        """A task-specificity cache for run `aname` whose sharing means are
        (response rule, timing, other) per type; "related" is their mean."""
        cache = self.cache()
        cache["aname"] = aname
        for type_name, means in (("hidden", hidden_means), ("modulation", modulation_means)):
            sharing = self.sharing(["response rule", "timing", "other"], list(means))
            sharing["by_relation"]["response rule"]["p_perm"] = p_rule
            sharing["by_relation"]["related"]["mean"] = float(np.mean(means[:2]))
            sharing["by_relation"]["other"]["p_perm"] = np.nan
            cache["types"][type_name]["sharing"] = sharing
        return cache

    def write_runs(self, base, specs):
        import re
        for seed, (hidden, modulation, p_rule) in specs.items():
            aname = re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME)
            run_dir = base / aname
            run_dir.mkdir()
            with (run_dir / f"task_specificity_{aname}.pkl").open("wb") as f:
                pickle.dump(self.run_cache(aname, hidden, modulation, p_rule), f)

    def render_cohort(self, root):
        with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("builtins.print") as output:
            paper_plot.plot_task_sharing_cohort()
        return save, output

    def test_cohort_draws_excess_sharing_per_run_relation_and_type(self):
        from scipy.stats import wilcoxon

        specs = {11: ((0.6, 0.5, 0.2), (0.7, 0.7, 0.6), 0.02),
                 22: ((0.5, 0.3, 0.3), (0.8, 0.5, 0.6), 0.30),
                 33: ((0.4, 0.4, 0.1), (0.9, 0.6, 0.7), 0.01)}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            root.mkdir()
            self.write_runs(base, specs)
            save, _ = self.render_cohort(root)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_task_sharing_cohort.png")
        # One panel per saved type, titled; the bottom panel carries the
        # column labels: the pooled column, then the relations.
        self.assertEqual([axis.get_title(loc="left") for axis in figure.axes],
                         ["Hidden neuron clusters", "Synapse clusters"])
        self.assertEqual([label.get_text() for label in figure.axes[-1].get_xticklabels()],
                         ["All\nrelated", "Response\nrule", "Timing"])
        collections = paper_plot.mpl.collections
        hidden_axis = figure.axes[0]
        dots = [c for c in hidden_axis.collections if isinstance(c, collections.PathCollection)]
        offsets = np.concatenate([c.get_offsets() for c in dots])
        self.assertEqual(len(offsets), 3 * 3)        # 3 columns x 3 runs
        # Excess = relation mean - other mean per run: response rule 0.4, 0.2, 0.3.
        rule = sorted(offsets[np.isclose(offsets[:, 0], 1, atol=0.3), 1].round(6).tolist())
        self.assertEqual(rule, [0.2, 0.3, 0.4])
        # Filled when the run's saved relation p < 0.05: seeds 11 and 33 for
        # the response rule.
        rule_dots = [c for c in dots if np.isclose(c.get_offsets()[0][0], 1, atol=0.3)]
        filled = sum(np.allclose(c.get_facecolors()[0][:3],
                                 paper_plot.mpl.colors.to_rgb(
                                     paper_plot._TASK_SPECIFICITY_COLORS["hidden"]))
                     for c in rule_dots)
        self.assertEqual(filled, 2)
        dashes = [c for c in hidden_axis.collections if isinstance(c, collections.LineCollection)]
        np.testing.assert_allclose([c.get_segments()[0][0, 1] for c in dashes],
                                   [np.median([0.35, 0.1, 0.3]), 0.3, 0.3])
        # Column texts carry the one-sided cohort Wilcoxon p of excess > 0.
        expected = wilcoxon([0.4, 0.2, 0.3], alternative="greater").pvalue
        texts = [text.get_text() for text in hidden_axis.texts]
        self.assertEqual(len(texts), 3)
        self.assertEqual(texts[1], paper_plot._short_p(expected))
        self.assertEqual(paper_plot._short_p(0.0004), "p<.001")
        self.assertEqual(paper_plot._short_p(0.004), "p=.004")
        self.assertEqual(paper_plot._short_p(0.54), "p=.54")
        self.assertEqual(len(figure.legends), 1)
        self.assertIn("3 runs", save.call_args.kwargs["extra"])
        self.assertIn("response rule 3/3 > 0, 2 significant", save.call_args.kwargs["extra"])
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_sharing_cohort"],
                      paper_plot.plot_task_sharing_cohort)

    def test_cohort_needs_two_runs_and_drops_disagreeing_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            root.mkdir()
            self.write_runs(base, {11: ((0.6, 0.5, 0.2), (0.7, 0.7, 0.6), 0.02)})
            save, output = self.render_cohort(root)
        save.assert_not_called()
        self.assertIn("need >= 2", output.call_args.args[0])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / paper_plot.LESION_ANAME
            root.mkdir()
            self.write_runs(base, {11: ((0.6, 0.5, 0.2), (0.7, 0.7, 0.6), 0.02),
                                   22: ((0.5, 0.3, 0.3), (0.8, 0.5, 0.6), 0.30)})
            # A run whose sharing test is not the task-label permutation is
            # incompatible; one with other relation groups disagrees.
            bad = self.run_cache("bad", (0.1, 0.1, 0.1), (0.1, 0.1, 0.1))
            bad["types"]["hidden"]["sharing"]["permutation_unit"] = "point"
            bad_dir = base / paper_plot.LESION_ANAME.replace("seed", "seed8")
            bad_dir.mkdir()
            bad["aname"] = bad_dir.name
            with (bad_dir / f"task_specificity_{bad_dir.name}.pkl").open("wb") as f:
                pickle.dump(bad, f)
            odd = self.run_cache("odd", (0.1, 0.1, 0.1), (0.1, 0.1, 0.1))
            odd_dir = base / paper_plot.LESION_ANAME.replace("seed", "seed9")
            odd_dir.mkdir()
            odd["aname"] = odd_dir.name
            odd["types"]["hidden"]["sharing"]["relations"] = ["timing", "response rule", "other"]
            with (odd_dir / f"task_specificity_{odd_dir.name}.pkl").open("wb") as f:
                pickle.dump(odd, f)
            save, output = self.render_cohort(root)
        save.assert_called_once()
        self.assertIn("2 runs", save.call_args.kwargs["extra"])
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("incompatible", printed)
        self.assertIn("relations differ", printed)


if __name__ == "__main__":
    unittest.main()


class TaskSpecificityPoolTests(TaskSpecificitySeedsTests):
    """The median impaired-task count of every run under each random-control
    pool, per cluster type."""

    def test_pool_seeds_draw_each_pools_medians_per_type(self):
        from scipy.stats import wilcoxon

        responsive = ({"hidden": [0, 0, 1, 2], "modulation": [1, 2, 3]},
                      {"hidden": [0, 1, 1], "modulation": [2, 2, 4]},
                      {"hidden": [1, 1, 1], "modulation": [3, 3, 3]})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6],
                                  pools={"responsive": responsive[0]})
            self.write_run(base, 22, [1, 2, 3], [3, 5, 5], pools={"responsive": responsive[1]})
            self.write_run(base, 33, [1, 1, 2], [5, 5, 6], pools={"responsive": responsive[2]})
            save, output = self.render_seeds(root, paper_plot.plot_task_specificity_pool_seeds)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_task_specificity_pool_seeds.png")
        axis = figure.axes[0]
        # Two types x two pools: the columns are labelled by pool, the types
        # named below each pair.
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["All units", "Responsive", "All units", "Responsive"])
        texts = [text.get_text() for text in axis.texts]
        self.assertEqual([text for text in texts if text in ("Hidden", "Synapse")],
                         ["Hidden", "Synapse"])
        collections = paper_plot.mpl.collections
        dots = [c for c in axis.collections
                if isinstance(c, collections.PathCollection) and len(c.get_offsets())]
        offsets = np.concatenate([c.get_offsets() for c in dots])
        columns = sorted(set(np.round(offsets[:, 0]).astype(int).tolist()))
        self.assertEqual(len(columns), 4)
        # Per-run medians: hidden all 1, 2, 1 / responsive 0.5, 1, 1;
        # synapse all 4, 5, 5 / responsive 2, 2, 3.
        by_column = [sorted(offsets[np.isclose(np.round(offsets[:, 0]), column), 1].tolist())
                     for column in columns]
        self.assertEqual(by_column, [[1, 1, 2], [0.5, 1, 1], [4, 5, 5], [2, 2, 3]])
        self.assertFalse(any(np.allclose(c.get_facecolors()[0][:3], 1.0) for c in dots))
        # The paired test between the responsive and the all pool, per type.
        expected = [wilcoxon([0.5 - 1, 1 - 2, 1 - 1]).pvalue, wilcoxon([2 - 4, 2 - 5, 3 - 5]).pvalue]
        p_texts = [text for text in texts if text.startswith("p")]
        self.assertEqual(p_texts, [f"p = {p:.3f}" for p in expected])
        extra = save.call_args.kwargs["extra"]
        self.assertIn("hidden: median 1 (all), 1 (responsive)", extra)
        self.assertIn("modulation: median 5 (all), 2 (responsive)", extra)
        self.assertIn("modulation responsive vs all paired Wilcoxon p", extra)
        self.assertIn("3 runs", axis.get_legend().get_title().get_text())
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("all control pool", printed)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_specificity_pool_seeds"],
                      paper_plot.plot_task_specificity_pool_seeds)

    def test_pool_seeds_need_a_second_pool_in_every_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6])
            self.write_run(base, 22, [1, 2, 3], [3, 5, 5],
                           pools={"responsive": {"hidden": [0, 1, 1], "modulation": [2, 2, 4]}})
            save, output = self.render_seeds(root, paper_plot.plot_task_specificity_pool_seeds)
        save.assert_not_called()
        self.assertIn("all control pool only", output.call_args.args[0])
        self.assertIn("lesion_controls.py", output.call_args.args[0])

    def test_runs_of_another_control_pool_are_left_out(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6])
            self.write_run(base, 22, [1, 2, 3], [3, 5, 5])
            self.write_run(base, 33, [1, 1, 2], [5, 5, 6], control_pool="responsive")
            save, output = self.render_seeds(root)
        save.assert_called_once()
        self.addCleanup(paper_plot.plt.close, save.call_args.args[0])
        self.assertIn("2 runs", save.call_args.args[0].axes[0].get_legend().get_title().get_text())
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("control pool (responsive) differs", printed)

    def test_loader_keeps_every_pools_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.write_run(base, 11, [0, 1, 1, 5], [2, 4, 6],
                                  pools={"responsive": {"hidden": [0, 0, 1, 2]},
                                         "all_same_trials": {"hidden": [0, 1, 1, 4],
                                                             "modulation": [2, 3, 6]}})
            with patch("builtins.print"):
                saved = paper_plot._load_task_specificity(root, root.name)
        self.assertEqual(saved["control_pool"], "all")
        hidden = saved["types"]["hidden"]["pool_counts"]
        self.assertEqual(set(hidden), {"all", "responsive", "all_same_trials"})
        np.testing.assert_array_equal(hidden["responsive"], [0, 0, 1, 2])
        np.testing.assert_array_equal(hidden["all"], [0, 1, 1, 5])
        self.assertEqual(set(saved["types"]["modulation"]["pool_counts"]), {"all", "all_same_trials"})
