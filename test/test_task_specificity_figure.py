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
                       "dispersion": {"counts": np.array([0, 1, 1, 5]), "p_perm": 0.01},
                       "sharing": self.sharing(relations, [0.6, 0.5, 0.2])},
            "modulation": {"cluster_labels": ["c1", "c2", "c3"],
                           "dispersion": {"counts": np.array([2, 2, 2]), "p_perm": 0.7},
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
        counts_axis = figures["multitask_task_specificity_counts.png"].axes[0]
        self.assertEqual(len(counts_axis.lines), 2)   # hidden and modulation only
        hidden_line = counts_axis.lines[0]
        np.testing.assert_allclose(hidden_line.get_xdata(), np.arange(7))
        np.testing.assert_allclose(hidden_line.get_ydata(), [0.25, 0.5, 0, 0, 0, 0.25, 0])
        legend_texts = [text.get_text() for text in counts_axis.get_legend().get_texts()]
        self.assertIn("Hidden neuron clusters (n = 4; p = 0.010)", legend_texts)
        self.assertIn("Synapse clusters (n = 3; p = 0.700)", legend_texts)
        self.assertEqual(counts_axis.get_ylim()[0], 0)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_specificity"],
                      paper_plot.plot_task_specificity)

    def test_counts_do_not_depend_on_the_sharing_statistics(self):
        cache = self.cache()
        for entry in cache["types"].values():
            entry.pop("sharing")
        figures = self.render(cache)
        self.assertEqual(set(figures), {"multitask_task_specificity_counts.png"})
        self.assertEqual(len(figures["multitask_task_specificity_counts.png"].axes[0].lines), 2)

    def test_no_legend_and_incompatible_caches(self):
        figures = self.render(self.cache(), show_legend=False)
        for figure in figures.values():
            self.assertIsNone(figure.axes[0].get_legend())
        for problem in ("missing", "run", "type", "counts"):
            cache = self.cache()
            if problem == "missing":
                cache = None
            elif problem == "run":
                cache["aname"] = "other"
            elif problem == "type":
                cache["types"]["synapse"] = cache["types"].pop("modulation")
            else:
                cache["types"]["hidden"]["dispersion"]["counts"] = np.array([0, 9, 1, 1])
            with self.subTest(problem=problem):
                self.assertEqual(self.render(cache), {})


if __name__ == "__main__":
    unittest.main()
