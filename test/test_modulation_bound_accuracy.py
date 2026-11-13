"""Regression checks for the matched mb1/mb2 accuracy figure."""

import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import _bootstrap  # noqa: F401
import paper_plot


class ModulationBoundAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "task_params": {"ruleset": "everything"},
            "train_params": {
                "weight_reg": "L2",
                "reg_lambda": 1e-4,
                "batch_size": 128,
                "n_batches": 128,
            },
            "net_params": {
                "net_type": "dmpn",
                "n_neurons": [1, 300, 1],
                "linear_embed": 300,
                "activation": "tanh",
                "input_layer_add": True,
                "ml_params": {},
            },
        }

    def test_groups_only_metadata_matched_regular_mb1_and_mb2_runs(self):
        mb1 = copy.deepcopy(self.config)
        mb2 = copy.deepcopy(self.config)
        mb2["net_params"]["ml_params"]["m_bounds"] = [-2.0, 2.0]
        mb2_e3 = copy.deepcopy(mb2)
        mb2_e3["train_params"]["reg_lambda"] = 1e-3
        wrong_projection = copy.deepcopy(mb2)
        wrong_projection["net_params"]["linear_embed"] = 200
        configs = {
            "regular_mb1": mb1,
            "regular_mb2": mb2,
            "regular_mb2_e3": mb2_e3,
            "wrong_projection": wrong_projection,
        }
        results = {
            "regular_mb1": {"feature": "L21e4", "acc": 0.80},
            "regular_mb2": {"feature": "L21e4mb2", "acc": 0.90},
            "regular_mb2_e3": {"feature": "L21e3mb2", "acc": 0.70},
            "wrong_projection": {"feature": "L21e4mb2", "acc": 0.99},
        }

        def open_config(path, *args, **kwargs):
            name = path.name.removeprefix("param_").removesuffix("_param.json")
            return io.StringIO(json.dumps(configs[name]))

        with patch.object(Path, "open", new=open_config):
            groups = paper_plot._modulation_bound_accuracy_groups(results)

        self.assertEqual(
            tuple(groups), paper_plot.MODULATION_BOUND_L2_STRENGTHS)
        self.assertEqual(groups[1e-4]["mb1"], [80.0])
        self.assertEqual(groups[1e-4]["mb2"], [90.0])
        self.assertEqual(groups[1e-3]["mb2"], [70.0])
        self.assertEqual(groups[1e-5]["mb1"], [])

    def _render(self, groups):
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "open", return_value=io.StringIO("{}")), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(
                    paper_plot, "_modulation_bound_accuracy_groups",
                    return_value=groups), \
                patch.object(paper_plot, "_save_fig") as save, \
                contextlib.redirect_stdout(io.StringIO()) as console:
            paper_plot.plot_modulation_bound_accuracy()
        figure, output_path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        return figure, output_path, console.getvalue()

    @staticmethod
    def _marks(axis):
        return [(text.get_text(), text.xy[0]) for text in axis.texts]

    def test_plot_is_one_panel_and_is_registered_in_acc_plot(self):
        groups = {
            strength: {
                bound_name: [60.0 + 5.0 * bound_index]
                for bound_index, (bound_name, _, _, _) in enumerate(
                    paper_plot._MODULATION_BOUND_SPECS)
            }
            for strength in paper_plot.MODULATION_BOUND_L2_STRENGTHS
        }
        figure, output_path, _ = self._render(groups)
        self.assertEqual(len(figure.axes), 1)
        axis = figure.axes[0]
        self.assertEqual(
            [label.get_text() for label in axis.get_xticklabels()],
            ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"])
        self.assertEqual(axis.get_xlabel(), "L2 regularization strength")
        self.assertEqual(axis.get_ylabel(), "Test accuracy (%)")
        # the two bounds sit side by side at every strength
        dots = [dots.get_offsets() for dots in axis.collections
                if isinstance(dots, paper_plot.mpl.collections.PathCollection)]
        self.assertEqual(len(dots), 8)
        offset = paper_plot._MODULATION_BOUND_OFFSET
        self.assertEqual([tuple(points[0]) for points in dots[:2]],
                         [(-offset, 60.0), (offset, 65.0)])
        self.assertEqual(
            output_path.name, "multitask_modulation_bound_accuracy.png")
        self.assertIs(
            paper_plot.FIGURES_BY_MODE["acc_plot"]
            ["modulation_bound_accuracy"],
            paper_plot.plot_modulation_bound_accuracy,
        )
        self.assertLess(max(figure.get_size_inches()), 3.5)
        # One run per group: nothing is tested, bracketed or marked.
        self.assertEqual((self._marks(axis), len(axis.lines)), ([], 0))
        self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                         ["$M \\in [-1, 1]$ (mb1)", "$M \\in [-2, 2]$ (mb2)", "median, IQR"])

    def test_mb2_is_tested_against_mb1_of_the_same_l2(self):
        mb1, mb2 = [80.0, 81.0, 82.0, 83.0, 84.0], [90.0, 91.0, 92.0, 93.0, 94.0]
        groups = {strength: {"mb1": mb1, "mb2": mb2 if index else mb1}
                  for index, strength in enumerate(paper_plot.MODULATION_BOUND_L2_STRENGTHS)}
        figure, _, console = self._render(groups)
        axis = figure.axes[0]
        # Holm over the four strengths' tests: 4 * 2/252 = .032 where the bounds differ.
        self.assertEqual(self._marks(axis), [("n.s.", 0), ("*", 1), ("*", 2), ("*", 3)])
        # one bracket per tested pair, over the pair's highest network
        self.assertEqual(len(axis.lines), 4)
        offset = paper_plot._MODULATION_BOUND_OFFSET
        np.testing.assert_allclose(axis.lines[1].get_xdata(),
                                   [1 - offset, 1 - offset, 1 + offset, 1 + offset])
        self.assertGreater(min(axis.lines[1].get_ydata()), 94.0)
        self.assertGreater(axis.get_ylim()[1], max(axis.lines[1].get_ydata()))
        self.assertEqual(
            axis.get_legend().get_texts()[-1].get_text(),
            "Holm p < 0.05 / 0.01 / 0.001\nvs mb1 at the same L2 (Mann\u2013Whitney)")
        self.assertIn(
            "L2=0.0001, mb2 vs mb1: Mann-Whitney p=0.00794, Holm-adjusted p=0.0317", console)


if __name__ == "__main__":
    unittest.main()
