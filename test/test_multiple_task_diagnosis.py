"""Tests for the sibling-task row-cluster diagnosis and its paper figure."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest

import multiple_task_diagnosis as diagnosis
import paper_plot


# Rows of the synthetic clusterings: "{rule}-{phase}" for every phase a rule has.
_RULE_PHASES = {
    "fdgo": ("stim1", "delay1", "go1"),
    "dmcgo": ("stim1", "delay1", "go1"),
    "dmcnogo": ("stim1", "delay1", "go1"),
    "delaydm1": ("stim1", "stim2", "delay1", "delay2", "go1"),
    "delaydm2": ("stim1", "stim2", "delay1", "delay2", "go1"),
}
_NAMES = np.array([f"{rule}-{phase}" for rule, phases in _RULE_PHASES.items()
                   for phase in phases])


def _labels(*shared_groups):
    """One row cluster per row, except that the rows of each group in
    `shared_groups` (lists of row names) share one cluster."""
    labels = np.arange(_NAMES.size)
    for group in shared_groups:
        rows = [int(np.flatnonzero(_NAMES == name)[0]) for name in group]
        labels[rows] = labels[rows[0]]
    return labels


def _experiment():
    """Every memory-1 row in one cluster in every representation (chance 1),
    and the two DMC response rows sharing a cluster (chance 1 of the 10
    response pairs); everything else apart."""
    labels = _labels([name for name in _NAMES if name.endswith("-delay1")],
                     ["dmcgo-go1", "dmcnogo-go1"])
    return {rep: {"row_labels": labels, "tb_break_name": _NAMES}
            for rep in ("input", "hidden", "modulation")}


class ChanceLevelTests(unittest.TestCase):
    def test_chance_is_the_sharing_of_two_tasks_rows_of_one_phase(self):
        rep_data = {"tb_break_name": np.array(["a-stim1", "b-stim1", "c-stim1",
                                               "a-delay1", "b-delay1"]),
                    "row_labels": np.array([0, 0, 1, 2, 2])}
        # stim1: (a, b) share, (a, c) and (b, c) do not.
        self.assertAlmostEqual(diagnosis._same_phase_chance(rep_data, "stim1"), 1 / 3)
        self.assertEqual(diagnosis._same_phase_chance(rep_data, "delay1"), 1.0)
        self.assertTrue(np.isnan(diagnosis._same_phase_chance(rep_data, "go1")))

    def test_poisson_binomial_p_matches_the_binomial_test_at_one_chance(self):
        for n_same in (0, 3, 9, 10):
            self.assertAlmostEqual(
                diagnosis._poisson_binomial_p(n_same, [0.3] * 10),
                binomtest(n_same, 10, 0.3).pvalue)
        # Certain sharing makes any other count impossible.
        self.assertEqual(diagnosis._poisson_binomial_p(4, [1.0] * 4), 1.0)
        self.assertTrue(np.isnan(diagnosis._poisson_binomial_p(1, [0.5, np.nan])))
        with self.assertRaises(ValueError):
            diagnosis._poisson_binomial_p(3, [0.5, 0.5])

    def test_pair_result_reports_chance_and_the_test_against_it(self):
        experiments = [(f"run{seed}", _experiment()) for seed in range(8)]
        result = diagnosis.diagnose_category_same_cluster(experiments)
        self.assertEqual(result["phases"], ["stim1", "delay1", "go1"])
        response = result["per_phase"]["go1"]["hidden"]
        self.assertEqual((response["n_same"], response["n_total"]), (8, 8))
        self.assertAlmostEqual(response["chance_mean"], 0.1)
        self.assertAlmostEqual(response["expected_same"], 0.8)
        self.assertAlmostEqual(response["p_vs_chance"], 0.1 ** 8)
        # Memory 1 shares in every run, but so does every task pair: chance 1.
        memory = result["per_phase"]["delay1"]["input"]
        self.assertEqual((memory["n_same"], memory["chance_mean"], memory["p_vs_chance"]),
                         (8, 1.0, 1.0))
        stimulus = result["per_phase"]["stim1"]["modulation"]
        self.assertEqual((stimulus["n_same"], stimulus["chance_mean"]), (0, 0.0))


class DiagnosisDriverTests(unittest.TestCase):
    def _write_run(self, root, aname):
        run_dir = root / aname
        run_dir.mkdir(parents=True)
        labels = _experiment()["input"]["row_labels"]
        neuron = {"result": {"row_tol_labels": labels}, "tb_break_name": _NAMES}
        with open(run_dir / f"cluster_info_{aname}.pkl", "wb") as stream:
            pickle.dump({"input_normalized": neuron, "hidden_normalized": neuron}, stream)
        modulation = {"result_all_lst": [None, {"row_tol_labels": labels}],
                      "tb_break_name": _NAMES}
        with open(run_dir / f"cluster_info_mod_{aname}.pkl", "wb") as stream:
            pickle.dump({"modulation_all_normalized": modulation}, stream)

    def test_a_run_replaces_its_own_tag_and_keeps_the_others(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis, out = root / "analysis", root / "out"
            for seed in (1, 2):
                self._write_run(analysis, f"everything_seed{seed}_L21e4+hidden300+batch128+angle")
            out.mkdir()
            kept = ["diagnosis_L21e3.pkl", "diagnosis_L21e4_seed1.json",
                    "same_cluster_diagnoses_L21e3.png"]
            for name in kept + ["diagnosis_L21e4.json", "same_cluster_diagnoses_L21e4.png"]:
                (out / name).write_text("old")
            with patch.object(diagnosis, "MULTI_DIR", analysis), \
                    patch.object(diagnosis, "OUT_DIR", out), \
                    patch("builtins.print"):
                diagnosis.main(feature="L21e4")
            for name in kept:
                self.assertEqual((out / name).read_text(), "old")
            with open(out / "diagnosis_L21e4.pkl", "rb") as stream:
                payload = pickle.load(stream)
            self.assertEqual(payload["version"], diagnosis.RESULT_VERSION)
            self.assertEqual(payload["feature"], "L21e4")
            self.assertEqual(len(payload["experiments"]), 2)
            go = payload["results"]["category_same_cluster"]["per_phase"]["go1"]["modulation"]
            self.assertAlmostEqual(go["chance_mean"], 0.1)
            # The json copy and the figure were rewritten.
            self.assertEqual(json.loads((out / "diagnosis_L21e4.json").read_text())["version"],
                             diagnosis.RESULT_VERSION)
            self.assertNotEqual((out / "same_cluster_diagnoses_L21e4.png").read_bytes(), b"old")


class RowClusterSharingFigureTests(unittest.TestCase):
    def _payload(self):
        experiments = [(f"run{seed}", _experiment()) for seed in range(8)]
        return {
            "version": paper_plot._ROW_SHARING_RESULT_VERSION,
            "feature": paper_plot.multitask_feature(),
            "seed": None,
            "experiments": [name for name, _ in experiments],
            "results": {
                "category_same_cluster": diagnosis.diagnose_category_same_cluster(experiments),
                "integration_modality_same_cluster":
                    diagnosis.diagnose_integration_modality_same_cluster(experiments),
            },
        }

    def _render(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if payload is not None:
                with open(root / f"diagnosis_{paper_plot.multitask_feature()}.pkl", "wb") as stream:
                    pickle.dump(payload, stream)
            with patch.object(paper_plot, "MULTITASK_DIAGNOSIS_ROOT", root), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "SHOW_LEGEND", True), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as printed:
                paper_plot.plot_sibling_row_cluster_sharing()
        return save, printed

    def test_phases_share_columns_and_bars_sit_on_their_chance(self):
        payload = self._payload()
        save, _ = self._render(payload)
        save.assert_called_once()
        fig = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, fig)
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_sibling_row_cluster_sharing.png")
        axes = np.array(fig.axes).reshape(2, 5)
        self.assertEqual([ax.get_title() for ax in axes[0]],
                         ["Stimulus 1", "Stimulus 2", "Memory 1", "Memory 2", "Response"])
        # The DMC pair has no second stimulus or memory: those cells are empty.
        self.assertEqual([ax.axison for ax in axes[0]], [True, False, True, False, True])
        self.assertTrue(all(ax.axison for ax in axes[1]))
        self.assertIn("ReactCategoryPro", axes[0, 0].get_ylabel())
        # DMC response: every bar shares in 8/8 runs against a chance of 10%,
        # so the dash sits at 10, the label above the bar and starred.
        response = axes[0, 4]
        dashes = [segment[0][1] for collection in response.collections
                  for segment in collection.get_segments()]
        np.testing.assert_allclose(dashes, [10.0, 10.0, 10.0])
        self.assertEqual([text.get_text() for text in response.texts], ["8/8*"] * 3)
        self.assertEqual([text.get_position()[1] for text in response.texts], [103.0] * 3)
        # DMC first stimulus: no sharing, chance 0, nothing starred.
        self.assertEqual([text.get_text() for text in axes[0, 0].texts], ["0/8"] * 3)
        self.assertIn("ReactCategoryPro vs ReactCategoryAnti Response hidden 8/8",
                      save.call_args.kwargs["extra"])

    def test_skips_a_missing_or_pre_chance_result(self):
        save, printed = self._render(None)
        save.assert_not_called()
        self.assertIn("multiple_task_diagnosis.py --feature", str(printed.call_args_list))
        payload = self._payload()
        payload["version"] = 1
        save, printed = self._render(payload)
        save.assert_not_called()
        self.assertIn("no chance levels", str(printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
