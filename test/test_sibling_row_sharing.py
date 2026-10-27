"""Tests for the seven-pair sibling row-cluster sharing and its summary."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np

import multiple_task_diagnosis as diagnosis
import paper_plot
import sibling_row_sharing as sharing


# Rows of the synthetic clusterings: "{rule}-{phase}" for every phase a rule
# has. Two pro/anti pairs and one modality pair; the other pairs have no rows.
_RULE_PHASES = {
    "fdgo": ("stim1", "go1"),
    "fdanti": ("stim1", "go1"),
    "dmcgo": ("stim1", "delay1", "go1"),
    "dmcnogo": ("stim1", "delay1", "go1"),
    "delaydm1": ("stim1", "stim2", "delay1", "delay2", "go1"),
    "delaydm2": ("stim1", "stim2", "delay1", "delay2", "go1"),
}
_NAMES = np.array([f"{rule}-{phase}" for rule, phases in _RULE_PHASES.items()
                   for phase in phases])


def _experiment():
    """Every memory row of a phase in one cluster (chance 1) and the two
    DelayPro/DelayAnti response rows sharing one (1 of the 15 response pairs:
    chance 1/15); every other row in a cluster of its own (chance 0)."""
    labels = np.arange(_NAMES.size)
    for group in ([n for n in _NAMES if n.endswith("-delay1")],
                  [n for n in _NAMES if n.endswith("-delay2")],
                  ["fdgo-go1", "fdanti-go1"]):
        rows = [int(np.flatnonzero(_NAMES == name)[0]) for name in group]
        labels[rows] = labels[rows[0]]
    return {rep: {"row_labels": labels, "tb_break_name": _NAMES}
            for rep in ("input", "hidden", "modulation")}


def _experiments(n_runs=6):
    return [(f"run{seed}", _experiment()) for seed in range(n_runs)]


def _task_experiment():
    """Every memory row, of either memory phase, in one cluster (so a task's
    two memory rows share at chance 1) and ReactCategoryPro's stimulus and
    response rows sharing one (chance 0: no two different tasks' stimulus
    and response rows share); every other row in a cluster of its own."""
    labels = np.arange(_NAMES.size)
    for group in ([n for n in _NAMES if n.endswith(("-delay1", "-delay2"))],
                  ["dmcgo-stim1", "dmcgo-go1"]):
        rows = [int(np.flatnonzero(_NAMES == name)[0]) for name in group]
        labels[rows] = labels[rows[0]]
    return {rep: {"row_labels": labels, "tb_break_name": _NAMES}
            for rep in ("input", "hidden", "modulation")}


class PairGroupTests(unittest.TestCase):
    def test_groups_cover_seven_known_pairs_and_phase_types_every_phase(self):
        self.assertEqual({group: len(pairs) for group, pairs in sharing.PAIR_GROUPS.items()},
                         {"Pro / anti": 5, "Modality": 2})
        rules = [rule for pairs in sharing.PAIR_GROUPS.values()
                 for pair in pairs for rule in pair]
        self.assertEqual(len(rules), len(set(rules)))
        self.assertTrue(set(rules) <= set(diagnosis._TASK_DISPLAY))
        phases = [phase for group in sharing.PHASE_TYPES.values() for phase in group]
        self.assertEqual(sorted(phases), sorted(diagnosis._PHASE_DISPLAY))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.summary = sharing.summarize_by_phase_type(
            sharing.analyze_pairs(_experiments()))

    def test_response_counts_cells_against_their_chance_per_run(self):
        # Pro / anti response cells per run: DelayPro/Anti shares (chance
        # 1/15), ReactCategoryPro/Anti does not (chance 1/15).
        stats = self.summary["Pro / anti"]["Response"]["modulation"]
        self.assertEqual((stats["n_same"], stats["n_total"], stats["n_runs"]), (6, 12, 6))
        self.assertAlmostEqual(stats["pct_same_cluster"], 50.0)
        self.assertAlmostEqual(stats["chance_mean"], 1 / 15)
        for run in stats["per_run"].values():
            self.assertEqual((run["observed"], run["n_cells"]), (1, 2))
            self.assertAlmostEqual(run["expected"], 2 / 15)
        np.testing.assert_allclose(stats["excess"], [13 / 15] * 6)
        self.assertFalse(stats["unspecific"])
        self.assertLess(stats["p_excess"], sharing.SIGNIFICANCE)
        # The modality pair's response rows stay apart: below chance.
        below = self.summary["Modality"]["Response"]["hidden"]
        self.assertEqual((below["n_same"], below["n_total"]), (0, 6))
        self.assertLess(below["mean_excess"], 0)
        self.assertLess(below["p_excess"], sharing.SIGNIFICANCE)

    def test_stimulus_pools_both_stimulus_phases(self):
        # Modality: stim1 and stim2 rows, never shared, chance 0: no excess.
        stats = self.summary["Modality"]["Stimulus"]["input"]
        self.assertEqual((stats["n_same"], stats["n_total"]), (0, 12))
        self.assertEqual((stats["chance_mean"], stats["p_excess"]), (0.0, 1.0))

    def test_memory_at_chance_one_is_reported_but_not_tested(self):
        stats = self.summary["Pro / anti"]["Memory"]["input"]
        self.assertEqual((stats["n_same"], stats["n_total"], stats["chance_mean"]),
                         (6, 6, 1.0))
        self.assertTrue(stats["unspecific"])
        self.assertTrue(np.isnan(stats["p_excess"]))
        self.assertEqual(sharing._summary_mark(stats), "†")

    def test_excess_test_drops_runs_without_excess(self):
        self.assertEqual(sharing._excess_test([0.0, 0.0, 0.0]), 1.0)
        self.assertEqual(sharing._excess_test([0.0, 0.5]), 1.0)
        self.assertLess(sharing._excess_test([0.3, 0.6, 0.2, 0.9, 0.4, 0.8]), 0.05)


class SameTaskTests(unittest.TestCase):
    def setUp(self):
        self.same_task = sharing.summarize_same_task(
            [(f"run{seed}", _task_experiment()) for seed in range(6)])

    def test_cross_phase_chance_counts_only_two_different_tasks(self):
        rep_data = {"tb_break_name": np.array(["a-stim1", "b-stim1", "a-go1", "b-go1"]),
                    "row_labels": np.array([0, 1, 1, 2])}
        # (a-stim1, b-go1) apart, (b-stim1, a-go1) together; a's own rows,
        # also apart, do not count.
        self.assertEqual(sharing._cross_phase_chance(rep_data, "stim1", "go1"), 0.5)
        self.assertTrue(np.isnan(sharing._cross_phase_chance(rep_data, "stim1", "delay1")))

    def test_a_task_sharing_its_rows_beyond_chance(self):
        # Pro / anti cells per run: DelayPro and DelayAnti one each
        # (stimulus, response), ReactCategoryPro and Anti three each; only
        # ReactCategoryPro's stimulus and response rows share, at chance 0.
        pooled = self.same_task["Pro / anti"]["pooled"]["modulation"]
        self.assertEqual((pooled["n_same"], pooled["n_total"]), (6, 48))
        self.assertAlmostEqual(pooled["pct_same_cluster"], 12.5)
        self.assertEqual(pooled["chance_mean"], 0.0)
        self.assertLess(pooled["p_excess"], sharing.SIGNIFICANCE)
        tasks = self.same_task["Pro / anti"]["tasks"]
        self.assertEqual(list(tasks), ["fdgo", "fdanti", "dmcgo", "dmcnogo"])
        self.assertAlmostEqual(tasks["dmcgo"]["hidden"]["pct_same_cluster"], 100 / 3)
        self.assertEqual((tasks["fdgo"]["input"]["pct_same_cluster"],
                          tasks["fdgo"]["input"]["p_excess"]), (0.0, 1.0))

    def test_cells_at_chance_one_are_left_out_and_counted(self):
        # IntegrationModality1 and 2 have ten phase pairs each; their two
        # memory rows share at chance 1, so nine cells per task and run remain.
        modality = self.same_task["Modality"]
        self.assertEqual(modality["n_unspecific_cells"],
                         {"input": 12, "hidden": 12, "modulation": 12})
        self.assertEqual(modality["tasks"]["delaydm1"]["input"]["n_total"], 54)
        self.assertEqual(modality["pooled"]["input"]["n_same"], 0)

    def test_figure_puts_the_pooled_block_first_and_marks_the_excess(self):
        with patch.object(sharing.plt.Figure, "savefig", autospec=True) as save:
            sharing.plot_same_task(self.same_task, Path("same_task.png"), "test")
        axes = save.call_args.args[0].axes[:2]
        self.assertEqual([tick.get_text() for tick in axes[0].get_xticklabels()],
                         ["All", "DelayPro", "DelayAnti", "ReactCategoryPro",
                          "ReactCategoryAnti"])
        self.assertEqual([tick.get_text() for tick in axes[1].get_xticklabels()],
                         ["All", "IntegrationModality1", "IntegrationModality2"])
        # Stars over the pooled block and ReactCategoryPro, three bars each.
        self.assertEqual([text.get_text() for text in axes[0].texts], ["*"] * 6)
        self.assertIn("left out", axes[1].get_title(loc="left"))
        self.assertEqual(axes[0].get_xlim(), axes[1].get_xlim())


def _organized_experiment():
    """Four tasks with three phases each: input and hidden rows clustered by
    phase, modulation rows by task."""
    rules, phases = ("a", "b", "c", "d"), ("stim1", "delay1", "go1")
    names = np.array([f"{rule}-{phase}" for rule in rules for phase in phases])
    by_phase = np.tile(np.arange(len(phases)), len(rules))
    by_task = np.repeat(np.arange(len(rules)), len(phases))
    return {"input": {"row_labels": by_phase, "tb_break_name": names},
            "hidden": {"row_labels": by_phase, "tb_break_name": names},
            "modulation": {"row_labels": by_task, "tb_break_name": names}}


class OrganizationTests(unittest.TestCase):
    def test_pair_classes_and_enrichment_over_any_two_rows(self):
        rep_data = {"tb_break_name": np.array(["a-stim1", "b-stim1", "a-go1", "b-go1"]),
                    "row_labels": np.array([0, 0, 1, 2])}
        counts = sharing._organization_counts(rep_data)
        self.assertEqual(counts, {"period": {"n_pairs": 2, "n_shared": 1},
                                  "task": {"n_pairs": 2, "n_shared": 0},
                                  "neither": {"n_pairs": 2, "n_shared": 0},
                                  "any": {"n_pairs": 6, "n_shared": 1}})
        # Half of the period pairs share against one in six of any two rows.
        self.assertAlmostEqual(sharing._enrichment(counts, "period"), 3.0)
        self.assertEqual(sharing._enrichment(counts, "task"), 0.0)

    def test_period_and_task_organization_and_their_contrast(self):
        organization = sharing.summarize_organization(
            [(f"run{seed}", _organized_experiment()) for seed in range(6)])
        reps = organization["representations"]
        # 12 rows, 66 pairs. By phase: 18 shared pairs, all 18 period pairs
        # among them; by task: 12 shared, all 12 task pairs.
        self.assertAlmostEqual(reps["input"]["median"]["period"], 66 / 18)
        self.assertEqual(reps["input"]["median"]["task"], 0.0)
        self.assertAlmostEqual(reps["modulation"]["median"]["task"], 66 / 12)
        self.assertEqual(reps["modulation"]["median"]["period"], 0.0)
        self.assertLess(reps["input"]["p_vs_one"]["period"], sharing.SIGNIFICANCE)
        self.assertLess(reps["modulation"]["p_vs_one"]["task"], sharing.SIGNIFICANCE)
        self.assertEqual(len(reps["hidden"]["enrichment"]["task"]), 6)
        contrast = organization["contrast_p"]
        self.assertEqual(set(contrast), {"input_vs_hidden", "input_vs_modulation",
                                         "hidden_vs_modulation"})
        self.assertLess(contrast["hidden_vs_modulation"], sharing.SIGNIFICANCE)
        self.assertEqual(contrast["input_vs_hidden"], 1.0)

    def test_same_task_over_every_task_of_the_runs(self):
        entry = sharing.summarize_same_task_all(
            [(f"run{seed}", _task_experiment()) for seed in range(6)])
        self.assertEqual(list(entry["tasks"]),
                         ["fdgo", "fdanti", "delaydm1", "delaydm2", "dmcgo", "dmcnogo"])
        self.assertAlmostEqual(entry["tasks"]["dmcgo"]["modulation"]["pct_same_cluster"],
                               100 / 3)
        self.assertEqual(entry["n_unspecific_cells"]["input"], 12)


class DriverTests(unittest.TestCase):
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

    def test_writes_results_and_figures_and_keeps_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis, out = root / "analysis", root / "out"
            for seed in (1, 2, 3):
                self._write_run(analysis, f"everything_seed{seed}_L21e4+hidden300+batch128+angle")
            out.mkdir()
            kept = ["diagnosis_L21e4.pkl", "sibling_row_sharing_L21e3.pkl",
                    "sibling_row_sharing_summary_L21e4_seed1.png"]
            for name in kept + ["sibling_row_sharing_L21e4.json"]:
                (out / name).write_text("old")
            with patch.object(diagnosis, "MULTI_DIR", analysis), \
                    patch.object(sharing, "OUT_DIR", out), \
                    patch("builtins.print"):
                sharing.main(feature="L21e4")
            for name in kept:
                self.assertEqual((out / name).read_text(), "old")
            with open(out / "sibling_row_sharing_L21e4.pkl", "rb") as stream:
                payload = pickle.load(stream)
            self.assertEqual(payload["version"], sharing.RESULT_VERSION)
            self.assertEqual(len(payload["experiments"]), 3)
            self.assertEqual(set(payload["pairs"]["Pro / anti"]),
                             {"fdgo_vs_fdanti", "delaygo_vs_delayanti", "reactgo_vs_reactanti",
                              "dmsgo_vs_dmsnogo", "dmcgo_vs_dmcnogo"})
            # Pairs without rows have no phases and no summary cells.
            self.assertEqual(payload["pairs"]["Pro / anti"]["reactgo_vs_reactanti"]["phases"], [])
            self.assertEqual(payload["summary"]["Pro / anti"]["Response"]["input"]["n_total"], 6)
            self.assertEqual(json.loads((out / "sibling_row_sharing_L21e4.json").read_text())
                             ["version"], sharing.RESULT_VERSION)
            self.assertEqual(set(payload["same_task"]), {"Pro / anti", "Modality"})
            self.assertIn("dmcgo", payload["same_task_all"]["tasks"])
            self.assertEqual(set(payload["organization"]["representations"]),
                             {"input", "hidden", "modulation"})
            for name in ("sibling_row_sharing_pairs_L21e4.png",
                         "sibling_row_sharing_summary_L21e4.png",
                         "sibling_row_sharing_same_task_L21e4.png"):
                self.assertTrue((out / name).stat().st_size > 0)


class PairFigureTests(unittest.TestCase):
    def test_a_pair_without_the_first_phase_keeps_its_name(self):
        # DelayPro/DelayAnti has only a first stimulus, ReactPro/ReactAnti
        # only a response.
        names = np.array(["fdgo-stim1", "fdanti-stim1", "reactgo-go1", "reactanti-go1"])
        labels = np.array([0, 1, 2, 2])
        experiments = [("run0", {rep: {"row_labels": labels, "tb_break_name": names}
                                 for rep in ("input", "hidden", "modulation")})]
        results = {"delay": diagnosis.diagnose_pair_same_cluster(experiments, "fdgo", "fdanti"),
                   "react": diagnosis.diagnose_pair_same_cluster(
                       experiments, "reactgo", "reactanti")}
        with patch.object(diagnosis.plt.Figure, "savefig", autospec=True) as save:
            diagnosis.plot_combined_same_cluster(results, Path("pairs.png"))
        axes = np.array(save.call_args.args[0].axes).reshape(2, 2)
        # ReactPro/ReactAnti's empty first cell still shows the pair's name...
        self.assertTrue(axes[1, 0].axison)
        self.assertFalse(axes[1, 0].patches)
        self.assertEqual(axes[1, 0].get_ylabel(), "ReactPro\nvs\nReactAnti")
        # ...while an empty later cell is switched off.
        self.assertFalse(axes[0, 1].axison)
        self.assertEqual(axes[0, 0].get_ylabel(), "DelayPro\nvs\nDelayAnti")


class SummaryFigureTests(unittest.TestCase):
    def test_rows_are_groups_and_columns_phase_types(self):
        summary = sharing.summarize_by_phase_type(sharing.analyze_pairs(_experiments()))
        with patch.object(sharing.plt.Figure, "savefig", autospec=True) as save:
            sharing.plot_summary(summary, Path("summary.png"), "test")
        save.assert_called_once()
        fig = save.call_args.args[0]
        axes = np.array(fig.axes).reshape(2, 3)
        self.assertEqual([ax.get_title().split("\n")[0] for ax in axes[0]],
                         ["Stimulus", "Memory", "Response"])
        self.assertEqual([ax.get_ylabel() for ax in axes[:, 0]],
                         ["Pro / anti\n(5 pairs)", "Modality\n(2 pairs)"])
        self.assertTrue(all(ax.axison for ax in axes.flat))

    def test_panel_marks_tested_and_untested_cells(self):
        summary = sharing.summarize_by_phase_type(sharing.analyze_pairs(_experiments()))
        fig, axes = sharing.plt.subplots(1, 2)
        self.addCleanup(sharing.plt.close, fig)
        sharing._draw_summary_panel(axes[0], summary["Pro / anti"]["Response"])
        sharing._draw_summary_panel(axes[1], summary["Pro / anti"]["Memory"])
        self.assertEqual([text.get_text() for text in axes[0].texts], ["6/12*"] * 3)
        self.assertEqual([text.get_text() for text in axes[1].texts], ["6/6†"] * 3)
        # The chance dash sits at the mean chance of each bar.
        dashes = [segment[0][1] for collection in axes[0].collections
                  if hasattr(collection, "get_segments")
                  for segment in collection.get_segments()]
        np.testing.assert_allclose(dashes, [100 / 15] * 3)


class PaperFigureTests(unittest.TestCase):
    def _payload(self):
        return {
            "version": paper_plot._ROW_SHARING_RESULT_VERSION,
            "feature": paper_plot.multitask_feature(),
            "seed": None,
            "experiments": [f"run{seed}" for seed in range(6)],
            "organization": sharing.summarize_organization(
                [(f"run{seed}", _organized_experiment()) for seed in range(6)]),
            "same_task_all": sharing.summarize_same_task_all(
                [(f"run{seed}", _task_experiment()) for seed in range(6)]),
        }

    def _untrained_payload(self):
        """Untrained networks whose synapse rows cluster in pairs of tasks:
        a and b in one cluster, c and d in another."""
        def experiment():
            data = _organized_experiment()
            data["modulation"] = dict(data["modulation"],
                                      row_labels=np.repeat([0, 0, 1, 1], 3))
            return data
        return dict(self._payload(), untrained=True, organization=sharing.summarize_organization(
            [(f"init{seed}", experiment()) for seed in range(6)]))

    def _render(self, plot, payload, untrained=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = paper_plot.multitask_feature()
            for tag, saved in ((feature, payload), (f"{feature}_untrained", untrained)):
                if saved is not None:
                    with open(root / f"sibling_row_sharing_{tag}.pkl", "wb") as stream:
                        pickle.dump(saved, stream)
            with patch.object(paper_plot, "MULTITASK_DIAGNOSIS_ROOT", root), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "SHOW_LEGEND", True), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as printed:
                plot()
        if save.called:
            self.addCleanup(paper_plot.plt.close, save.call_args.args[0])
        return save, printed

    def test_organization_panels_put_period_and_task_on_one_log_scale(self):
        save, _ = self._render(paper_plot.plot_row_cluster_organization, self._payload())
        save.assert_called_once()
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_organization.png")
        axes = save.call_args.args[0].axes
        self.assertEqual([ax.get_title() for ax in axes],
                         ["Same period,\ndifferent task", "Same task,\ndifferent period"])
        self.assertEqual([tick.get_text() for tick in axes[0].get_xticklabels()],
                         ["Input\nneurons", "Hidden\nneurons", "Synapses"])
        self.assertEqual(axes[0].get_yscale(), "log")
        # Median dashes: input and hidden group by period (66/18), modulation
        # by task (66/12); a class with no sharing (0x) has no point to draw.
        medians = [[segment[0][1] for collection in ax.collections
                    if hasattr(collection, "get_segments")
                    for segment in collection.get_segments()] for ax in axes]
        np.testing.assert_allclose(medians[0], [66 / 18, 66 / 18])
        np.testing.assert_allclose(medians[1], [66 / 12])
        self.assertIn("paired Wilcoxon", save.call_args.kwargs["extra"])

    def test_untrained_networks_sit_beside_the_trained_runs(self):
        save, printed = self._render(paper_plot.plot_row_cluster_organization,
                                     self._payload(), untrained=self._untrained_payload())
        save.assert_called_once()
        task_panel = save.call_args.args[0].axes[1]
        # Synapses: trained 66/12 (left of the column's center, red dash),
        # untrained 66/30 (right, gray dash, hollow points).
        dashes = [(segment[0][0], segment[0][1]) for collection in task_panel.collections
                  if hasattr(collection, "get_segments")
                  for segment in collection.get_segments()]
        self.assertEqual(len(dashes), 2)
        self.assertLess(dashes[0][0], 2.0)
        self.assertGreater(dashes[1][0], 2.0)
        np.testing.assert_allclose([dash[1] for dash in dashes], [66 / 12, 66 / 30])
        hollow = [collection for collection in task_panel.collections
                  if hasattr(collection, "get_facecolors")
                  and not hasattr(collection, "get_segments")
                  and np.allclose(collection.get_facecolors()[:, :3], 1.0)]
        self.assertEqual(len(hollow), 1)
        extra = save.call_args.kwargs["extra"]
        self.assertIn("6 untrained networks", extra)
        self.assertIn("task modulation trained 5.50x vs untrained 2.20x", extra)

    def test_untrained_provenance_names_the_config_and_init_seeds(self):
        names = [f"everything_seed749_L21e4+hidden300+batch128+angle_init{k}" for k in (2, 0, 1)]
        self.assertEqual(paper_plot._untrained_provenance(names), "seed 749 config, init 0, 1, 2")
        self.assertEqual(paper_plot._untrained_provenance(["init0", "init1"]), "")

    def test_without_untrained_networks_the_trained_runs_are_drawn_alone(self):
        save, printed = self._render(paper_plot.plot_row_cluster_organization, self._payload())
        save.assert_called_once()
        self.assertIn("No untrained control", str(printed.call_args_list))
        self.assertIn("untrained_clustering.py", str(printed.call_args_list))
        self.assertIn("0 untrained networks", save.call_args.kwargs["extra"])

    def test_same_task_bars_per_task_with_stars_on_the_excess(self):
        save, _ = self._render(paper_plot.plot_row_cluster_same_task, self._payload())
        save.assert_called_once()
        ax = save.call_args.args[0].axes[0]
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                         ["DelayPro", "DelayAnti", "IntegrationModality1",
                          "IntegrationModality2", "ReactCategoryPro", "ReactCategoryAnti"])
        # Only ReactCategoryPro's stimulus and response rows share (chance 0).
        self.assertEqual([text.get_text() for text in ax.texts], ["*"] * 3)
        self.assertIn("no period pairs", save.call_args.kwargs["extra"])

    def test_skips_a_missing_or_older_result(self):
        for plot in (paper_plot.plot_row_cluster_organization,
                     paper_plot.plot_row_cluster_same_task):
            save, printed = self._render(plot, None)
            save.assert_not_called()
            self.assertIn("sibling_row_sharing.py --feature", str(printed.call_args_list))
            payload = self._payload()
            payload["version"] = 2
            save, printed = self._render(plot, payload)
            save.assert_not_called()
            self.assertIn("is not a version-3", str(printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
