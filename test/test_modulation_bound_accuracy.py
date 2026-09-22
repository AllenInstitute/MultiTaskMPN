"""Regression checks for the matched mb1/mb2 accuracy figure."""

import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

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

    def test_plot_has_four_l2_panels_and_is_registered_in_acc_plot(self):
        groups = {
            strength: {
                bound_name: [60.0 + 5.0 * bound_index]
                for bound_index, (bound_name, _, _, _) in enumerate(
                    paper_plot._MODULATION_BOUND_SPECS)
            }
            for strength in paper_plot.MODULATION_BOUND_L2_STRENGTHS
        }
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "open", return_value=io.StringIO("{}")), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(
                    paper_plot, "_modulation_bound_accuracy_groups",
                    return_value=groups), \
                patch.object(paper_plot, "_save_fig") as save:
            paper_plot.plot_modulation_bound_accuracy()

        figure, output_path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            [
                "a   L2 = $10^{-5}$",
                "b   L2 = $10^{-4}$",
                "c   L2 = $10^{-3}$",
                "d   L2 = $10^{-2}$",
            ],
        )
        self.assertEqual(len(figure.axes[-1].get_xticklabels()), 2)
        self.assertEqual(
            [label.get_text() for label in figure.axes[-1].get_xticklabels()],
            ["$[-1, 1]$\n(mb1)", "$[-2, 2]$\n(mb2)"],
        )
        self.assertEqual(len(figure.axes[0].collections), 2)
        self.assertEqual(
            figure.axes[0].lines[0].get_ydata().tolist(), [60.0, 65.0])
        self.assertEqual(
            output_path.name, "multitask_modulation_bound_accuracy.png")
        self.assertIs(
            paper_plot.FIGURES_BY_MODE["acc_plot"]
            ["modulation_bound_accuracy"],
            paper_plot.plot_modulation_bound_accuracy,
        )
        self.assertEqual(figure.axes[0].get_ylabel(), "")


if __name__ == "__main__":
    unittest.main()
