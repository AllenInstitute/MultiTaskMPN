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

    def test_counts_and_sharing_figures_use_saved_values(self):
        figures = self.render(self.cache())
        self.assertEqual(set(figures), {"multitask_task_specificity_counts.png",
                                        "multitask_task_sharing_relations_hidden.png",
                                        "multitask_task_sharing_relations_modulation.png"})
        counts_axis = figures["multitask_task_specificity_counts.png"].axes[0]
        self.assertEqual(len(counts_axis.lines), 2)   # hidden and modulation only
        hidden_line = counts_axis.lines[0]
        np.testing.assert_allclose(hidden_line.get_xdata(), np.arange(7))
        np.testing.assert_allclose(hidden_line.get_ydata(), [0.25, 0.5, 0, 0, 0, 0.25, 0])
        legend_texts = [text.get_text() for text in counts_axis.get_legend().get_texts()]
        self.assertIn("Hidden neuron clusters (n = 4; p = 0.010)", legend_texts)
        self.assertIn("Synapse clusters (n = 3; p = 0.700)", legend_texts)
        self.assertEqual(counts_axis.get_ylim()[0], 0)

        sharing_axis = figures["multitask_task_sharing_relations_hidden.png"].axes[0]
        self.assertEqual([label.get_text() for label in sharing_axis.get_xticklabels()],
                         ["Response\nrule\np=.02", "Timing\np=.02", "Other"])
        self.assertEqual(paper_plot._short_p(0.0004), "p<.001")
        self.assertEqual(paper_plot._short_p(0.004), "p=.004")
        self.assertEqual(paper_plot._short_p(0.54), "p=.54")
        offsets = np.concatenate([c.get_offsets() for c in sharing_axis.collections])
        # three pair points plus one median per relation
        self.assertEqual(len(offsets), 3 * 4)
        self.assertTrue(np.any(np.isclose(offsets, [0, 0.6]).all(axis=1)))
        self.assertEqual(len(sharing_axis.texts), 0)   # p values live in the tick labels
        self.assertIn("p = 0.004", sharing_axis.get_legend().get_title().get_text())
        self.assertEqual(sharing_axis.get_title(loc="left"), "Hidden neuron clusters")
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["task_specificity"],
                      paper_plot.plot_task_specificity)

    def test_no_legend_and_incompatible_caches(self):
        figures = self.render(self.cache(), show_legend=False)
        for figure in figures.values():
            self.assertIsNone(figure.axes[0].get_legend())
        for problem in ("missing", "run", "unit", "type", "counts"):
            cache = self.cache()
            if problem == "missing":
                cache = None
            elif problem == "run":
                cache["aname"] = "other"
            elif problem == "unit":
                cache["types"]["hidden"]["sharing"]["permutation_unit"] = "point"
            elif problem == "type":
                cache["types"]["synapse"] = cache["types"].pop("modulation")
            else:
                cache["types"]["hidden"]["dispersion"]["counts"] = np.array([0, 9, 1, 1])
            with self.subTest(problem=problem):
                self.assertEqual(self.render(cache), {})


if __name__ == "__main__":
    unittest.main()
