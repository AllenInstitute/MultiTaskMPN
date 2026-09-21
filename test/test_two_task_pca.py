"""Checks for two-task cross-period PCA representations and paper rendering."""

import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import _bootstrap  # noqa: F401
import paper_plot
from two_task import two_task_analysis


class TwoTaskPCATests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.hidden = rng.normal(size=(6, 6, 4))
        self.modulation = rng.normal(size=(6, 6, 2, 3))
        self.task_id = np.array([0, 0, 0, 1, 1, 1])
        self.periods = {
            0: {"context": (0, 3), "stim": (3, 6)},
            1: {"context": (0, 3), "stim": (3, 6)},
        }

    def test_series_contains_raw_and_effective_modulation(self):
        series = two_task_analysis._cross_task_period_pca_series(
            self.hidden, self.modulation, np.ones((2, 3)), self.task_id,
            self.periods, top_k=2, max_pcs=3, center="none")
        by_name = dict(series)
        self.assertEqual(
            list(by_name), ["hidden", "modulation", "w_modulation"])
        np.testing.assert_allclose(
            by_name["modulation"]["__cross_task__"]["fve_k_all"],
            by_name["w_modulation"]["__cross_task__"]["fve_k_all"])

    def test_none_centering_is_case_insensitive(self):
        lower = two_task_analysis.figure2A_pca_fve(
            self.hidden, self.task_id, self.periods, center="none")
        title_case = two_task_analysis.figure2A_pca_fve(
            self.hidden, self.task_id, self.periods, center="None")
        np.testing.assert_allclose(
            lower["__cross_task__"]["fve_k_all"],
            title_case["__cross_task__"]["fve_k_all"])

    def test_input_interpolation_returns_only_interpolated_inputs(self):
        labels = np.array([[stim, task] for stim in range(8)
                           for task in (0, 1)])
        inputs = torch.arange(16 * 2 * 3, dtype=torch.float).reshape(16, 2, 3)
        alphas, interpolated = two_task_analysis.input_interpolation(
            inputs, labels, expand_stimulus=False, n_alpha=2)

        self.assertEqual(alphas, [0.0, 0.5, 1.0])
        self.assertEqual(len(interpolated), 3)
        anti = inputs[1::2]
        pro = inputs[0::2]
        torch.testing.assert_close(interpolated[0], anti)
        torch.testing.assert_close(interpolated[1], 0.5 * (anti + pro))
        torch.testing.assert_close(interpolated[2], pro)

    def test_paper_d_combine_renders_all_three_representations(self):
        labels = [
            "Pro Context", "Anti Context", "Pro Stim", "Anti Stim",
            "Pro Memory", "Anti Memory", "Pro Response", "Anti Response",
        ]
        entry = {
            "fve_k_all": np.eye(8),
            "labels": labels,
            "vmin": 0.0,
            "vmax": 1.0,
            "top_k": 2,
        }
        data = {name: dict(entry) for name in
                ("hidden", "modulation", "w_modulation")}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            with (run_dir / "d_combine_run.pkl").open("wb") as handle:
                pickle.dump(data, handle)
            with patch.object(paper_plot, "TWOTASKS_DIR", root), \
                    patch.object(paper_plot, "TWOTASK_ANAME", "run"), \
                    patch.object(paper_plot, "OUT_DIR", root), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_two_task_d_combine()

        figure = save.call_args.args[0]
        try:
            self.assertEqual(
                [ax.get_title() for ax in figure.axes[:3]],
                ["Hidden", "Modulation", "Eff. modulation"])
        finally:
            paper_plot.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
