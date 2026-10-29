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

    def cache(self):
        tasks = [f"t{k}" for k in range(6)]
        relations = ["response rule", "timing", "other"]
        types = {
            "hidden": {"cluster_labels": ["h1", "h2", "h3", "h4"],
                       "dispersion": {"counts": np.array([0, 1, 1, 5]), "p_perm": 0.01,
                                      "null_var": np.array([0.5, 1.0, 1.5, 1.0]),
                                      "null_count_pmf": self.HIDDEN_PMF.copy(),
                                      "side": "greater"},
                       "sharing": self.sharing(relations, [0.6, 0.5, 0.2])},
            "modulation": {"cluster_labels": ["c1", "c2", "c3"],
                           "dispersion": {"counts": np.array([2, 2, 2]), "p_perm": 0.7,
                                          "null_var": np.full(4, 0.25),
                                          "null_count_pmf": self.MODULATION_PMF.copy(),
                                          "side": "greater"},
                           "sharing": self.sharing(relations, [0.3, 0.3, 0.3])},
        }
        return {"schema_version": 2, "aname": paper_plot.LESION_ANAME, "tasks": tasks,
                "n_perm": 4, "modulation_lesion_mode": "freeze_M", "types": types}

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
        for problem in ("missing", "run", "schema", "mode", "type", "counts", "null", "pmf"):
            cache = self.cache()
            if problem == "missing":
                cache = None
            elif problem == "run":
                cache["aname"] = "other"
            elif problem == "schema":
                cache["schema_version"] = 1
            elif problem == "mode":
                cache["modulation_lesion_mode"] = "zero_W"
            elif problem == "null":
                cache["types"]["hidden"]["dispersion"].pop("null_count_pmf")
            elif problem == "pmf":
                cache["types"]["hidden"]["dispersion"]["null_count_pmf"] = self.HIDDEN_PMF[:-1]
            elif problem == "type":
                cache["types"]["synapse"] = cache["types"].pop("modulation")
            else:
                cache["types"]["hidden"]["dispersion"]["counts"] = np.array([0, 9, 1, 1])
            with self.subTest(problem=problem):
                self.assertEqual(self.render(cache), {})


class TaskSpecificitySeedsTests(TaskSpecificityFigureTests):
    """The median impaired-task count of every sibling run, per cluster type."""

    def write_run(self, base, seed, hidden_counts, modulation_counts, p_hidden=0.01):
        import re
        aname = re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME)
        cache = self.cache()
        cache["aname"] = aname
        cache["types"]["hidden"]["dispersion"]["counts"] = np.array(hidden_counts)
        cache["types"]["hidden"]["dispersion"]["p_perm"] = p_hidden
        cache["types"]["modulation"]["dispersion"]["counts"] = np.array(modulation_counts)
        run_dir = base / aname
        run_dir.mkdir()
        with (run_dir / f"task_specificity_{aname}.pkl").open("wb") as f:
            pickle.dump(cache, f)
        return run_dir

    def render_seeds(self, root):
        with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("builtins.print") as output:
            paper_plot.plot_task_specificity_seeds()
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
