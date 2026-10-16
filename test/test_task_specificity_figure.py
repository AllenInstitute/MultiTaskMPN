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

    def cache(self):
        tasks = [f"t{k}" for k in range(6)]
        relations = ["response rule", "timing", "other"]
        types = {
            "hidden": {"cluster_labels": ["h1", "h2", "h3", "h4"],
                       "dispersion": {"counts": np.array([0, 1, 1, 5]), "p_perm": 0.01,
                                      "null_var": np.array([0.5, 1.0, 1.5, 1.0]),
                                      "side": "greater"},
                       "sharing": self.sharing(relations, [0.6, 0.5, 0.2])},
            "modulation": {"cluster_labels": ["c1", "c2", "c3"],
                           "dispersion": {"counts": np.array([2, 2, 2]), "p_perm": 0.7,
                                          "null_var": np.full(4, 0.25), "side": "greater"},
                           "sharing": self.sharing(relations, [0.3, 0.3, 0.3])},
        }
        return {"schema_version": 1, "aname": paper_plot.LESION_ANAME, "tasks": tasks,
                "n_perm": 4, "types": types}

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
        figures = self.render(self.cache())
        # Only the counts figure: the task-pair sharing figures were removed.
        self.assertEqual(set(figures), {"multitask_task_specificity_counts.png"})
        self.assertFalse(hasattr(paper_plot, "_short_p"))
        self.assertFalse(hasattr(paper_plot, "_TASK_SHARING_RELATION_LABELS"))
        axis = figures["multitask_task_specificity_counts.png"].axes[0]
        # One column per saved type (hidden, modulation), each tick label
        # naming the type, its cluster count and the saved dispersion p.
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["Hidden\nclusters\nn = 4\np = 0.010",
                          "Synapse\nclusters\nn = 3\np = 0.700"])
        collections = paper_plot.mpl.collections
        dots = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
        offsets = np.concatenate([c.get_offsets() for c in dots])
        # Every cluster is one dot at its count: hidden 0, 1, 1, 5 at x = 0
        # (the two 1s spread sideways), modulation 2, 2, 2 at x = 1.
        self.assertEqual(len(offsets), 7)
        self.assertEqual(sorted(offsets[np.isclose(offsets[:, 0], 0, atol=0.3), 1].tolist()),
                         [0, 1, 1, 5])
        self.assertEqual(sorted(offsets[np.isclose(offsets[:, 0], 1, atol=0.3), 1].tolist()),
                         [2, 2, 2])
        ties = offsets[np.isclose(offsets[:, 1], 1)][:, 0]
        self.assertEqual(len(ties), 2)
        self.assertGreater(ties.max() - ties.min(), 0)
        # Median dashes and the mixing-null bands (mean ± sqrt(median null var)).
        dashes = [c for c in axis.collections if isinstance(c, collections.LineCollection)]
        self.assertEqual([c.get_segments()[0][0, 1] for c in dashes], [1.0, 2.0])
        bands = [c for c in axis.collections if isinstance(c, collections.PolyCollection)]
        self.assertEqual(len(bands), 2)
        hidden_band = bands[0].get_paths()[0].vertices[:, 1]
        np.testing.assert_allclose([hidden_band.min(), hidden_band.max()], [0.75, 2.75])
        synapse_band = bands[1].get_paths()[0].vertices[:, 1]
        np.testing.assert_allclose([synapse_band.min(), synapse_band.max()], [1.5, 2.5])
        self.assertEqual(axis.get_ylabel(), "Tasks impaired per cluster")
        self.assertGreater(axis.get_ylim()[1], 6)
        legend_texts = [text.get_text() for text in axis.get_legend().get_texts()]
        self.assertEqual(legend_texts, ["median", "mixing null, mean ± SD"])
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_specificity"],
                      paper_plot.plot_task_specificity)

    def test_counts_do_not_depend_on_the_sharing_statistics(self):
        cache = self.cache()
        for entry in cache["types"].values():
            entry.pop("sharing")
        figures = self.render(cache)
        self.assertEqual(set(figures), {"multitask_task_specificity_counts.png"})
        axis = figures["multitask_task_specificity_counts.png"].axes[0]
        self.assertEqual(len(axis.get_xticklabels()), 2)

    def test_no_legend_and_incompatible_caches(self):
        figures = self.render(self.cache(), show_legend=False)
        for figure in figures.values():
            self.assertIsNone(figure.axes[0].get_legend())
        for problem in ("missing", "run", "type", "counts", "null"):
            cache = self.cache()
            if problem == "missing":
                cache = None
            elif problem == "run":
                cache["aname"] = "other"
            elif problem == "null":
                cache["types"]["hidden"]["dispersion"].pop("null_var")
            elif problem == "type":
                cache["types"]["synapse"] = cache["types"].pop("modulation")
            else:
                cache["types"]["hidden"]["dispersion"]["counts"] = np.array([0, 9, 1, 1])
            with self.subTest(problem=problem):
                self.assertEqual(self.render(cache), {})


if __name__ == "__main__":
    unittest.main()
