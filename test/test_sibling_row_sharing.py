"""Tests for restricted row-sharing statistics, paper plots and legacy diagnostics."""

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
        self.experiments = [(f"run{seed}", _task_experiment()) for seed in range(6)]
        self.same_task = sharing.summarize_same_task(self.experiments, n_permutations=2000)

    def test_sharing_is_compared_to_the_restricted_null(self):
        pooled = self.same_task["Pro / anti"]["pooled"]["modulation"]
        self.assertEqual((pooled["n_same"], pooled["n_total"]), (6, 48))
        self.assertAlmostEqual(pooled["pct_same_cluster"], 12.5)
        self.assertGreater(pooled["chance_mean"], 0)
        self.assertLess(pooled["q_fdr"], sharing.SIGNIFICANCE)
        tasks = self.same_task["Pro / anti"]["tasks"]
        self.assertEqual(list(tasks), ["fdgo", "fdanti", "dmcgo", "dmcnogo"])
        stats = tasks["dmcgo"]["hidden"]
        self.assertAlmostEqual(stats["pct_same_cluster"], 100 / 3)
        self.assertAlmostEqual(stats["chance_mean"], 1 / 108, delta=0.004)
        self.assertLess(stats["q_fdr"], sharing.SIGNIFICANCE)

    def test_chance_one_pairs_are_kept_in_every_representation(self):
        modality = self.same_task["Modality"]
        for rep in sharing.REPRESENTATIONS:
            self.assertEqual(modality["tasks"]["delaydm1"][rep]["n_total"], 60)
            self.assertEqual(modality["pooled"][rep]["n_same"], 12)
            self.assertAlmostEqual(modality["pooled"][rep]["pct_same_cluster"], 10)
        self.assertNotIn("n_unspecific_cells", modality)

    def test_figure_puts_the_pooled_block_first_and_marks_corrected_tests(self):
        with patch.object(sharing.plt.Figure, "savefig", autospec=True) as save:
            sharing.plot_same_task(self.same_task, Path("same_task.png"), "test")
        axes = save.call_args.args[0].axes[:2]
        self.assertEqual([tick.get_text() for tick in axes[0].get_xticklabels()],
                         ["All", "DelayPro", "DelayAnti", "ReactCategoryPro",
                          "ReactCategoryAnti"])
        self.assertEqual([text.get_text() for text in axes[0].texts], ["*"] * 6)
        self.assertNotIn("left out", axes[1].get_title(loc="left"))
        self.assertEqual(axes[0].get_xlim(), axes[1].get_xlim())


class SamePeriodTests(unittest.TestCase):
    def test_every_phase_keeps_all_task_pairs(self):
        entry = sharing.summarize_same_period_all(_experiments(), n_permutations=2000)
        periods = entry["periods"]
        self.assertEqual(list(periods), ["stim1", "stim2", "delay1", "delay2", "go1"])
        memory = periods["delay1"]["input"]
        self.assertEqual((memory["n_same"], memory["n_total"]), (36, 36))
        self.assertGreater(memory["chance_mean"], 0)
        self.assertLess(memory["q_fdr"], sharing.SIGNIFICANCE)
        response = periods["go1"]["modulation"]
        self.assertEqual((response["n_same"], response["n_total"]), (6, 90))
        # Sharing by two response rows once per run is not automatically significant.
        self.assertGreater(response["p_perm"], sharing.SIGNIFICANCE)
        pooled = entry["pooled"]["input"]
        self.assertEqual((pooled["n_same"], pooled["n_total"]), (48, 228))
        self.assertNotIn("n_unspecific_cells", entry)


class RestrictedPermutationTests(unittest.TestCase):
    @staticmethod
    def data(names, labels):
        return {rep: {"tb_break_name": np.array(names), "row_labels": np.array(labels)}
                for rep in sharing.REPRESENTATIONS}

    def test_row_shuffle_preserves_blocks_and_missing_slots(self):
        blocks = np.array(["stim1", "go1", "stim1", "go1", "go1"])
        indices = sharing._restricted_permutations(blocks, 50, np.random.default_rng(2))
        np.testing.assert_array_equal(blocks[indices], np.tile(blocks, (50, 1)))
        np.testing.assert_array_equal(np.sort(indices, axis=1),
                                      np.tile(np.arange(5), (50, 1)))
        self.assertGreater(len(np.unique(indices, axis=0)), 1)

    def test_single_phase_tasks_and_missing_phases_are_retained(self):
        data = self.data(["a-stim1", "a-go1", "b-go1", "c-go1"], [0, 1, 1, 1])
        tasks = sharing.summarize_same_task_all([("run", data)], n_permutations=100)
        periods = sharing.summarize_same_period_all([("run", data)], n_permutations=100)
        self.assertEqual(list(tasks["tasks"]), ["a"])
        self.assertEqual(tasks["pooled"]["input"]["n_total"], 1)
        self.assertEqual(list(periods["periods"]), ["go1"])
        self.assertEqual(periods["pooled"]["input"]["n_total"], 3)
        self.assertEqual(periods["pooled"]["input"]["n_same"], 3)

    def test_all_one_cluster_is_chance_one_with_p_one(self):
        data = self.data(_NAMES, np.zeros(len(_NAMES), dtype=int))
        for summarize in (sharing.summarize_same_task_all, sharing.summarize_same_period_all):
            entry = summarize([("run", data)], n_permutations=100)
            for stats in entry["pooled"].values():
                self.assertEqual(stats["pct_same_cluster"], 100)
                self.assertEqual(stats["chance_mean"], 1)
                self.assertEqual(stats["p_perm"], 1)
                self.assertEqual(stats["q_fdr"], 1)

    def test_pair_counts_do_not_depend_on_representation_labels(self):
        data = _task_experiment()
        data["input"] = dict(data["input"], row_labels=np.zeros(len(_NAMES), dtype=int))
        for summarize in (sharing.summarize_same_task_all, sharing.summarize_same_period_all):
            entry = summarize([("run", data)], n_permutations=100)
            counts = [entry["pooled"][rep]["n_total"] for rep in sharing.REPRESENTATIONS]
            self.assertEqual(len(set(counts)), 1)
            self.assertEqual(entry["pooled"]["input"]["pct_same_cluster"], 100)

    def test_missing_or_duplicate_rows_fail_instead_of_changing_comparison(self):
        for invalid in ("missing_rep", "missing_row", "duplicate_row"):
            data = _experiment()
            if invalid == "missing_rep":
                del data["hidden"]
            elif invalid == "missing_row":
                data["hidden"] = {key: value[:-1] for key, value in data["hidden"].items()}
            else:
                names = _NAMES.copy()
                names[0] = names[1]
                data["hidden"] = dict(data["hidden"], tb_break_name=names)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sharing.summarize_same_task_all([("run", data)], n_permutations=10)

    def test_reordered_rows_and_runs_are_reproducible(self):
        data = _experiment()
        data["hidden"] = {key: value[::-1] for key, value in data["hidden"].items()}
        first = sharing.summarize_same_task_all(_experiments(2), n_permutations=100,
                                               permutation_seed=17)
        second = sharing.summarize_same_task_all([("run1", data), ("run0", data)],
                                                n_permutations=100, permutation_seed=17)
        self.assertEqual(first, second)

    def test_whole_row_null_preserves_dependent_pairs(self):
        # Two tasks, two phases. Enumerate all four within-phase permutations.
        data = self.data(["a-stim1", "a-go1", "b-stim1", "b-go1"], [0, 0, 1, 1])
        permutations = np.array([[0, 1, 2, 3], [2, 1, 0, 3],
                                 [0, 3, 2, 1], [2, 3, 0, 1]])
        with patch.object(sharing, "_restricted_permutations", return_value=permutations), \
                patch.object(sharing, "_permutation_p", wraps=sharing._permutation_p) as test:
            sharing.summarize_same_task_all([("run", data)], n_permutations=4)
        # A shuffle matches either both tasks or neither, never just one.
        np.testing.assert_array_equal(test.call_args_list[0].args[1], [1, 0, 0, 1])

    def test_binary_zero_sharing_uses_the_discrete_null(self):
        # Each task's two phase labels have null sharing probability 1/8.
        # Across 17 runs, zero observed successes is not unusual.
        names = [f"t{i}-{phase}" for i in range(8) for phase in ("stim1", "go1")]
        labels = [label for i in range(8) for label in (i, (i + 1) % 8)]
        data = self.data(names, labels)
        entry = sharing.summarize_same_task_all(
            [(f"run{i:02d}", data) for i in range(17)], n_permutations=10000)
        stats = entry["tasks"]["t0"]["input"]
        self.assertEqual(stats["n_same"], 0)
        self.assertAlmostEqual(stats["chance_mean"], 1 / 8, delta=0.004)
        self.assertAlmostEqual(stats["p_perm"], 2 * (7 / 8) ** 17, delta=0.025)
        self.assertGreater(stats["q_fdr"], 0.05)

    def test_runs_are_equally_weighted_when_pair_counts_differ(self):
        first = self.data(["a-stim1", "a-go1"], [0, 0])  # 1/1
        second = self.data(["a-stim1", "a-delay1", "a-go1"], [0, 1, 2])  # 0/3
        entry = sharing.summarize_same_task_all(
            [("one", first), ("three", second)], n_permutations=100)
        stats = entry["tasks"]["a"]["input"]
        self.assertEqual((stats["n_same"], stats["n_total"]), (1, 4))
        self.assertEqual(stats["pct_same_cluster"], 50)
        self.assertEqual(stats["chance_mean"], 0.5)

    def test_add_one_ties_and_bh_adjustment(self):
        self.assertEqual(sharing._permutation_p(0, np.ones(9)), 0.2)
        self.assertEqual(sharing._permutation_p(0.5, np.full(9, 0.5 + 1e-15)), 1)
        stats = [{"p_perm": p} for p in (0.04, 0.001, 0.02, 0.5)]
        sharing._adjust_fdr(stats)
        np.testing.assert_allclose([s["q_fdr"] for s in stats], [0.04 * 4 / 3, 0.004, 0.04, 0.5])

    def test_group_and_all_task_views_reuse_identical_tests(self):
        entry, groups = sharing._sharing_permutations(_experiments(2), "task", 100, 2)
        self.assertIs(entry["tasks"]["fdgo"], groups["Pro / anti"]["tasks"]["fdgo"])
        self.assertEqual(entry["permutation"]["n_permutations"], 100)
        self.assertEqual(entry["permutation"]["permutation_seed"], 2)
        statistics = list(entry["pooled"].values())
        statistics += [s for reps in entry["tasks"].values() for s in reps.values()]
        statistics += [s for group in groups.values() for s in group["pooled"].values()]
        self.assertTrue(all(s["q_fdr"] >= s["p_perm"] for s in statistics))


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
        self.assertEqual(entry["tasks"]["delaydm1"]["input"]["n_total"], 60)


class DriverTests(unittest.TestCase):
    def test_invalid_rows_do_not_delete_previous_results(self):
        data = _experiment()
        del data["hidden"]
        with patch.object(diagnosis, "_discover_anames", return_value=["run"]), \
                patch.object(diagnosis, "_load_experiment", return_value=data), \
                patch.object(sharing, "_clear_previous_results") as clear, \
                patch("builtins.print"), self.assertRaisesRegex(ValueError, "missing hidden"):
            sharing.main(n_permutations=10)
        clear.assert_not_called()

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
                sharing.main(feature="L21e4", n_permutations=1000)
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
            self.assertIn("delay1", payload["same_period_all"]["periods"])
            self.assertEqual(payload["same_task_all"]["permutation"]["n_permutations"], 1000)
            self.assertEqual(payload["same_period_all"]["permutation"]["permutation_seed"], 0)
            stats = payload["same_task_all"]["tasks"]["fdgo"]["input"]
            self.assertIn("p_perm", stats)
            self.assertIn("q_fdr", stats)
            self.assertNotIn("p_excess", stats)
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
            "same_period_all": sharing.summarize_same_period_all(_experiments()),
        }

    def _untrained_payload(self):
        """Untrained networks whose synapse rows cluster in pairs of tasks:
        a and b in one cluster, c and d in another."""
        def experiment():
            data = _organized_experiment()
            data["modulation"] = dict(data["modulation"],
                                      row_labels=np.repeat([0, 0, 1, 1], 3))
            return data
        return dict(self._payload(), untrained=True, feature=None, architecture="",
                    config_features=["L21e4"],
                    organization=sharing.summarize_organization(
                        [(f"init{seed}", experiment()) for seed in range(6)]))

    def _render(self, plot, payload, untrained=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = paper_plot.multitask_feature()
            for tag, saved in ((feature, payload),
                               (paper_plot.untrained_sharing_tag(feature), untrained)):
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

    def test_the_untrained_control_is_shared_across_l2_but_not_architectures(self):
        # The control file carries no feature: whichever L2 cohort is
        # selected, the same file is read...
        with patch.object(paper_plot, "MULTITASK_L2", "1e4"):
            save, _ = self._render(paper_plot.plot_row_cluster_organization,
                                   dict(self._payload(), feature="L21e4"),
                                   untrained=self._untrained_payload())
        self.assertIn("6 untrained networks", save.call_args.kwargs["extra"])
        # ...but a control of another architecture is refused.
        save, printed = self._render(paper_plot.plot_row_cluster_organization, self._payload(),
                                     untrained=dict(self._untrained_payload(),
                                                    architecture="lam0.9"))
        self.assertIn("0 untrained networks", save.call_args.kwargs["extra"])
        self.assertIn("default-architecture untrained", str(printed.call_args_list))
        self.assertEqual(paper_plot._row_sharing_path(untrained=True).name,
                         "sibling_row_sharing_untrained.pkl")

    def test_stale_control_requests_statistics_only(self):
        control = dict(self._untrained_payload(), version=4)
        save, printed = self._render(paper_plot.plot_row_cluster_same_task_seeds,
                                     self._payload(), untrained=control)
        save.assert_called_once()
        message = str(printed.call_args_list)
        self.assertIn("is not a version-5", message)
        self.assertIn("sibling_row_sharing.py --feature", message)
        self.assertIn("--untrained", message)
        self.assertNotIn("untrained_clustering.py", message)

    def test_stars_read_corrected_q_not_raw_p(self):
        stats = {"pct_same_cluster": 50.0, "chance_mean": 0.25,
                 "p_perm": 0.001, "q_fdr": 0.2,
                 "per_run": {"a": {"observed": 1, "expected": 0.5, "n_cells": 2}}}
        fig, ax = paper_plot.plt.subplots()
        self.addCleanup(paper_plot.plt.close, fig)
        paper_plot._draw_row_sharing_bars(ax, 0, {"input": stats})
        self.assertEqual(len(ax.texts), 0)
        stats["q_fdr"] = 0.04
        paper_plot._draw_row_sharing_bars(ax, 1, {"input": stats})
        self.assertEqual([text.get_text() for text in ax.texts], ["*"])

    def test_bars_use_network_means_with_unequal_pair_counts(self):
        stats = {"pct_same_cluster": 50.0, "chance_mean": 0.5,
                 "per_run": {"a": {"observed": 1, "expected": 1, "n_cells": 1},
                             "b": {"observed": 0, "expected": 0, "n_cells": 3}}}
        height, chance, spread = paper_plot._row_sharing_bar_values(stats)
        self.assertEqual((height, chance), (50.0, 50.0))
        self.assertAlmostEqual(spread, np.std([100, 0], ddof=1))

    def test_same_task_bars_per_task_with_stars_on_corrected_tests(self):
        save, _ = self._render(paper_plot.plot_row_cluster_same_task_seeds, self._payload())
        save.assert_called_once()
        ax = save.call_args.args[0].axes[0]
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                         ["DelayPro", "DelayAnti", "IntegrationModality1",
                          "IntegrationModality2", "ReactCategoryPro", "ReactCategoryAnti"])
        # Only ReactCategoryPro has significant within-task alignment.
        self.assertEqual([text.get_text() for text in ax.texts], ["*"] * 3)
        self.assertIn("no period pairs", save.call_args.kwargs["extra"])

    def test_untrained_networks_get_a_hollow_twin_of_every_bar(self):
        for plot in (paper_plot.plot_row_cluster_same_task_seeds,
                     paper_plot.plot_row_cluster_same_period_seeds):
            with self.subTest(plot=plot.__name__):
                save, _ = self._render(plot, self._payload(), untrained=self._untrained_payload())
                save.assert_called_once()
                bars = save.call_args.args[0].axes[0].patches
                hollow = [bar for bar in bars if np.allclose(bar.get_facecolor()[:3], 1.0)]
                self.assertEqual(len(hollow), len(bars) // 2)
                self.assertIn("6 untrained networks", save.call_args.kwargs["extra"])

    def test_seed_figures_draw_the_mean_over_runs_with_sd_whiskers(self):
        # Three runs share the DelayPro/DelayAnti response rows (1 of 15
        # response pairs), three keep them apart: per-run response % of
        # 20/3 and 0 in every representation.
        def apart():
            data = _experiment()
            row = int(np.flatnonzero(_NAMES == "fdanti-go1")[0])
            for rep_data in data.values():
                rep_data["row_labels"] = rep_data["row_labels"].copy()
                rep_data["row_labels"][row] = row
            return data
        experiments = _experiments(3) + [(f"run{seed}", apart()) for seed in range(3, 6)]
        payload = dict(self._payload(),
                       same_period_all=sharing.summarize_same_period_all(experiments))
        stats = payload["same_period_all"]["periods"]["go1"]["input"]
        shared, chance = paper_plot._row_sharing_run_values(stats)
        np.testing.assert_allclose(shared, [20 / 3] * 3 + [0.0] * 3)
        self.assertTrue(np.all(chance > 0))
        height, dash, spread = paper_plot._row_sharing_bar_values(stats)
        self.assertAlmostEqual(height, 10 / 3)
        self.assertAlmostEqual(dash, chance.mean())
        self.assertAlmostEqual(spread, float(np.std(shared, ddof=1)))
        for plot, name in ((paper_plot.plot_row_cluster_same_period_seeds,
                            "multitask_row_cluster_same_period_seeds.png"),
                           (paper_plot.plot_row_cluster_same_task_seeds,
                            "multitask_row_cluster_same_task_seeds.png")):
            with self.subTest(plot=plot.__name__):
                save, _ = self._render(plot, payload, untrained=self._untrained_payload())
                save.assert_called_once()
                self.assertEqual(Path(save.call_args.args[1]).name, name)
                self.assertIn("whiskers: SD over runs", save.call_args.kwargs["extra"])
                fig = save.call_args.args[0]
                self.assertIn("mean ± SD over runs",
                              [text.get_text() for text in fig.legends[0].get_texts()])
        # The response bars of the period figure carry the whiskers: one
        # errorbar per representation (the untrained twins do not vary).
        save, _ = self._render(paper_plot.plot_row_cluster_same_period_seeds, payload,
                               untrained=self._untrained_payload())
        ax = save.call_args.args[0].axes[0]
        whiskers = [container for container in ax.containers
                    if isinstance(container, paper_plot.mpl.container.ErrorbarContainer)]
        self.assertEqual(len(whiskers), 3)
        response = [bar for bar in ax.patches if abs(bar.get_height() - 10 / 3) < 1e-9]
        self.assertEqual(len(response), 3)
        self.assertLess(ax.get_ylim()[0], float(shared.mean() - shared.std(ddof=1)))

    def test_same_period_bars_per_period_with_stars_on_corrected_tests(self):
        save, _ = self._render(paper_plot.plot_row_cluster_same_period_seeds, self._payload())
        save.assert_called_once()
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_same_period_seeds.png")
        ax = save.call_args.args[0].axes[0]
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                         ["Stimulus 1", "Stimulus 2", "Memory 1", "Memory 2", "Response"])
        expected_stars = sum(stats["q_fdr"] < sharing.SIGNIFICANCE
                             for by_rep in self._payload()["same_period_all"]["periods"].values()
                             for stats in by_rep.values())
        self.assertEqual([text.get_text() for text in ax.texts], ["*"] * expected_stars)
        self.assertEqual(len(ax.patches), 15)

    def test_skips_a_missing_or_older_result(self):
        for plot in (paper_plot.plot_row_cluster_organization,
                     paper_plot.plot_row_cluster_same_task_seeds,
                     paper_plot.plot_row_cluster_same_period_seeds):
            save, printed = self._render(plot, None)
            save.assert_not_called()
            self.assertIn("sibling_row_sharing.py --feature", str(printed.call_args_list))
            payload = self._payload()
            payload["version"] = 4
            save, printed = self._render(plot, payload)
            save.assert_not_called()
            self.assertIn("is not a version-5", str(printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
