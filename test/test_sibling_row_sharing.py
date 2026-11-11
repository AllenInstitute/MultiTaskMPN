"""Tests for restricted row-sharing statistics, paper plots and legacy diagnostics."""

import json
import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from matplotlib.container import ErrorbarContainer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import false_discovery_control, mannwhitneyu

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


# Selected K per representation (row, column) of the counted fixtures.
_SELECTED_K = {"input": (5, 5), "hidden": (5, 6), "modulation": (19, 69)}


def _counted_experiments(n_runs=6, col_offset=0):
    """`_experiments` whose representations also carry the selected K:
    row K alternates between networks, column K rises with the network."""
    runs = []
    for seed in range(n_runs):
        data = _experiment()
        for rep, (row_k, col_k) in _SELECTED_K.items():
            data[rep] = dict(data[rep], row_k=row_k + seed % 2, col_k=col_k + col_offset + seed)
        runs.append((f"run{seed}", data))
    return runs


def _complete_experiment():
    """Extend the existing fixture with all five prespecified complete tasks."""
    names = list(_NAMES)
    names += [f"{rule}-{phase}" for rule in sharing.MATCHED_TASKS
              for phase in sharing.MATCHED_PHASES if f"{rule}-{phase}" not in names]
    labels = np.concatenate([_experiment()["input"]["row_labels"],
                             np.arange(len(_NAMES), len(names))])
    return {rep: {"tb_break_name": np.array(names), "row_labels": labels.copy()}
            for rep in sharing.REPRESENTATIONS}


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


class CoassignmentTests(unittest.TestCase):
    def test_aligns_rows_and_ignores_cluster_numbering_across_networks(self):
        names = np.array(["a-delay1", "a-stim1", "b-stim1"])
        label_sets = ({"input": [4, 4, 9], "hidden": [3, 3, 3], "modulation": [1, 2, 3]},
                      {"input": [10, 20, 20], "hidden": [7, 8, 9], "modulation": [6, 5, 6]})
        experiments = []
        for seed, label_set in enumerate(label_sets):
            rows = {}
            for index, (rep, labels) in enumerate(label_set.items()):
                order = np.roll(np.arange(3), seed + index)
                rows[rep] = {"tb_break_name": names[order],
                             "row_labels": np.asarray(labels)[order]}
            experiments.append((f"run{seed}", rows))
        result = sharing.summarize_row_coassignment(iter(experiments))
        self.assertEqual(result["row_names"], names.tolist())
        self.assertEqual(result["experiments"], ["run0", "run1"])
        self.assertEqual(result["n_runs"], 2)
        expected = {
            "input": [[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]],
            "hidden": [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
            "modulation": [[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]],
        }
        for rep, matrix in expected.items():
            np.testing.assert_array_equal(result["representations"][rep], matrix)

    def test_rejects_missing_or_invalid_rows_instead_of_changing_the_denominator(self):
        incomplete = copy.deepcopy(_experiment())
        incomplete.pop("modulation")
        different_rows = copy.deepcopy(_experiment())
        for rows in different_rows.values():
            rows["tb_break_name"] = rows["tb_break_name"][:-1]
            rows["row_labels"] = rows["row_labels"][:-1]
        nonfinite = copy.deepcopy(_experiment())
        nonfinite["hidden"]["row_labels"] = np.full(len(_NAMES), np.nan)
        duplicate = copy.deepcopy(_experiment())
        duplicate["input"]["tb_break_name"][0] = duplicate["input"]["tb_break_name"][1]
        for case, runs in enumerate(([], [("run0", None)],
                                    [("run0", _experiment()), ("run0", _experiment())],
                                    [("run0", incomplete)], [("run0", nonfinite)],
                                    [("run0", duplicate)],
                                    [("run0", _experiment()), ("run1", different_rows)])):
            with self.subTest(case=case), self.assertRaises(ValueError):
                sharing.summarize_row_coassignment(runs)


class ClusterCountTests(unittest.TestCase):
    def test_counts_follow_the_saved_selected_k_per_representation(self):
        runs = _counted_experiments(3)
        entry = sharing.summarize_cluster_counts(runs)
        self.assertEqual(entry["schema_version"], 1)
        self.assertEqual(entry["representations"]["input"]["col_k"], [5, 6, 7])
        self.assertEqual(entry["representations"]["hidden"]["row_k"], [5, 6, 5])
        self.assertEqual(entry["representations"]["modulation"]["anames"], ["run0", "run1", "run2"])
        self.assertEqual(entry["column_units"]["modulation"], "synapses")
        del runs[1][1]["hidden"]["col_k"]
        with self.assertRaisesRegex(sharing.MissingClusterCounts, "hidden.*run1"):
            sharing.summarize_cluster_counts(runs)
        # Without any saved K (older or synthetic caches) the driver skips the entry.
        with self.assertRaises(sharing.MissingClusterCounts):
            sharing.summarize_cluster_counts(_experiments(2))


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


class MatchedTaskTests(unittest.TestCase):
    def test_same_task_keeps_bars_but_recomputes_null_within_five_tasks(self):
        data = _complete_experiment()
        for source in data.values():
            source["row_labels"] = np.arange(len(source["row_labels"])) + 100
            selected = [str(name).rsplit("-", 1)[0] in sharing.MATCHED_TASKS
                        for name in source["tb_break_name"]]
            source["row_labels"][selected] = 0
        original = sharing.summarize_same_task_all([("run", data)], 1000)
        entry = sharing.summarize_same_task_matched_tasks([("run", data)], 1000)
        self.assertEqual(set(entry["tasks"]), set(sharing.MATCHED_TASKS))
        self.assertEqual(entry["permutation"]["n_fdr_tests"], 18)
        for rule, reps in entry["tasks"].items():
            for rep, stats in reps.items():
                self.assertEqual(stats["pct_same_cluster"],
                                 original["tasks"][rule][rep]["pct_same_cluster"])
                self.assertEqual(stats["per_run"]["run"]["n_cells"], 10)
                self.assertEqual((stats["chance_mean"], stats["p_perm"]), (1, 1))
                self.assertLess(original["tasks"][rule][rep]["chance_mean"], 1)

    def test_organization_uses_all_300_selected_pairs_and_preserves_zero(self):
        data = _complete_experiment()
        for source in data.values():
            source["row_labels"] = np.array([
                sharing.MATCHED_TASKS.index(str(name).rsplit("-", 1)[0])
                if str(name).rsplit("-", 1)[0] in sharing.MATCHED_TASKS else 99
                for name in source["tb_break_name"]])
        before = pickle.dumps(data)
        entry = sharing.summarize_organization_matched_tasks([("run", data)])
        for rep in sharing.REPRESENTATIONS:
            stats = entry["representations"][rep]
            counts = stats["counts"]["run"]
            self.assertEqual(counts["period"], {"n_pairs": 50, "n_shared": 0})
            self.assertEqual(counts["task"], {"n_pairs": 50, "n_shared": 50})
            self.assertEqual(counts["neither"], {"n_pairs": 200, "n_shared": 0})
            self.assertEqual(counts["any"], {"n_pairs": 300, "n_shared": 50})
            self.assertEqual(stats["enrichment"]["period"], [0.0])
            self.assertEqual(stats["enrichment"]["task"], [6.0])
        self.assertEqual(pickle.dumps(data), before)

    def test_fixed_pairs_preserve_labels_and_exclude_other_tasks(self):
        data = _complete_experiment()
        before = pickle.dumps(data)
        entry = sharing.summarize_same_period_matched_tasks([("run", data)], 500)
        self.assertEqual(entry["tasks"], list(sharing.MATCHED_TASKS))
        self.assertEqual(entry["phases"], list(sharing.MATCHED_PHASES))
        self.assertEqual(entry["n_pairs_per_period"], 10)
        for phase, reps in entry["periods"].items():
            for rep, stats in reps.items():
                source = data[rep]
                mask = np.isin(source["tb_break_name"],
                               [f"{task}-{phase}" for task in sharing.MATCHED_TASKS])
                labels = source["row_labels"][mask]
                observed = sum(labels[i] == labels[j] for i in range(5) for j in range(i))
                self.assertEqual(stats["per_run"]["run"]["n_cells"], 10)
                self.assertEqual(stats["pct_same_cluster"], 10 * observed)
        self.assertEqual(entry["pooled"]["input"]["n_total"], 50)
        self.assertEqual(pickle.dumps(data), before)
        # Nonselected rows and representation-specific ordering cannot affect
        # a statistic conditional on the saved labels of the selected rows.
        for source in data.values():
            other = [str(n).split("-")[0] not in sharing.MATCHED_TASKS
                     for n in source["tb_break_name"]]
            source["row_labels"][other] = -99
        data["hidden"] = {key: value[::-1] for key, value in data["hidden"].items()}
        self.assertEqual(entry, sharing.summarize_same_period_matched_tasks([("run", data)], 500))

    def test_missing_task_phase_fails_instead_of_changing_task_population(self):
        data = _complete_experiment()
        for rep, source in data.items():
            keep = source["tb_break_name"] != "multidelaydm-delay2"
            data[rep] = {key: value[keep] for key, value in source.items()}
        for summarize in (sharing.summarize_same_period_matched_tasks,
                          sharing.summarize_same_task_matched_tasks,
                          sharing.summarize_organization_matched_tasks):
            with self.subTest(summarize=summarize.__name__), self.assertRaisesRegex(
                    ValueError, "multidelaydm-delay2"):
                summarize([("run", data)])

    def test_paired_test_uses_five_networks_not_fifty_task_pairs(self):
        data = _complete_experiment()
        for source in data.values():
            names = source["tb_break_name"]
            labels = np.arange(len(names))
            labels[np.array([str(n).endswith(("-stim2", "-delay2")) for n in names])] = 999
            source["row_labels"] = labels
        entry = sharing.summarize_same_period_matched_tasks(
            [(f"run{i}", data) for i in range(5)], 1000)
        paired = entry["period_contrasts"]
        self.assertEqual(paired["n_fdr_tests"], 6)
        for reps in paired["contrasts"].values():
            for stats in reps.values():
                self.assertEqual(stats["mean_difference_pp"], 100)
                self.assertEqual(stats["n_runs"], 5)
                # All five differences have one sign: two of 2**5 sign maps.
                self.assertEqual(stats["p_perm"], 2 / 32)
                self.assertEqual(stats["q_fdr"], 2 / 32)
        self.assertEqual(entry["permutation"]["n_fdr_tests"], 18)

    def test_zero_differences_and_single_network_are_not_significant(self):
        data = _complete_experiment()
        for source in data.values():
            source["row_labels"][:] = 0
        entry = sharing.summarize_same_period_matched_tasks([("run", data)], 50)
        for reps in entry["period_contrasts"]["contrasts"].values():
            for stats in reps.values():
                self.assertEqual(stats["mean_difference_pp"], 0)
                self.assertEqual((stats["p_perm"], stats["q_fdr"]), (1, 1))


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


def _tree_experiment():
    data = _complete_experiment()
    n = len(data["modulation"]["row_labels"])
    tree = linkage(np.random.default_rng(7).normal(size=(n, 4)), method="average")
    data["modulation"].update(row_linkage=tree, row_labels=fcluster(tree, 5, criterion="maxclust"))
    return data


class CutRobustnessTests(unittest.TestCase):
    def test_recut_full_tree_before_filtering_and_preserve_network_order(self):
        data = _tree_experiment()
        original = data["modulation"]["row_labels"].copy()
        runs = [("run1", data), ("run0", data)]
        with patch.object(sharing, "fcluster", wraps=fcluster) as cut:
            result = sharing.summarize_cut_robustness_matched_tasks(runs, (3, 5))
        self.assertTrue(all(call.args[0].shape[0] == len(original) - 1
                            for call in cut.call_args_list))
        self.assertGreater(len(original), 25)
        expected = sharing.summarize_organization_matched_tasks(runs)["representations"]["modulation"]
        self.assertEqual(result["cuts"][5]["counts"], expected["counts"])
        self.assertEqual(result["cuts"][5]["anames"], ["run1", "run0"])
        np.testing.assert_array_equal(data["modulation"]["row_labels"], original)
        for k in (3, 5):
            for counts in result["cuts"][k]["counts"].values():
                self.assertEqual(counts["any"]["n_pairs"], 300)
                self.assertEqual(counts["task"]["n_pairs"], 50)
                self.assertEqual(counts["period"]["n_pairs"], 50)

    def test_all_task_recut_reproduces_organization_counts_without_a_matched_grid(self):
        """Every row enters: at the saved K the counts are the organization
        check's, the selected-row count is the full-tree count, and rows of
        the five-task grid are not required."""
        data = _tree_experiment()  # labels are the tree's K=5 cut
        runs = [("run1", data), ("run0", data)]
        result = sharing.summarize_cut_robustness_all(runs, (3, 5))
        expected = sharing.summarize_organization(runs)["representations"]["modulation"]
        self.assertEqual(result["cuts"][5]["counts"], expected["counts"])
        self.assertEqual(result["cuts"][5]["anames"], ["run1", "run0"])
        n_rows = len(data["modulation"]["row_labels"])
        for k in (3, 5):
            cut = result["cuts"][k]
            self.assertEqual(cut["actual_k"], [k, k])
            self.assertEqual(cut["matched_k"], cut["actual_k"])
            for counts in cut["counts"].values():
                self.assertEqual(counts["any"]["n_pairs"], n_rows * (n_rows - 1) // 2)
        self.assertEqual(result["schema_version"], 1)
        self.assertNotIn("task_names", result)
        partial = _experiment()  # no ContextInt or IntegrationMultimodal rows
        n_rows = len(partial["modulation"]["row_labels"])
        tree = linkage(np.random.default_rng(3).normal(size=(n_rows, 4)), method="average")
        partial["modulation"].update(row_linkage=tree,
                                     row_labels=fcluster(tree, 4, criterion="maxclust"))
        with self.assertRaises(sharing.MissingMatchedTaskRows):
            sharing.summarize_cut_robustness_matched_tasks([("run", partial)], (4,))
        cut = sharing.summarize_cut_robustness_all([("run", partial)], (4,))["cuts"][4]
        self.assertEqual(cut["counts"]["run"], sharing._organization_counts(partial["modulation"]))

    def test_tied_identical_profiles_are_not_forced_into_k_clusters(self):
        data = _complete_experiment()
        n = len(data["modulation"]["row_labels"])
        data["modulation"].update(row_labels=np.ones(n), row_linkage=linkage(np.zeros((n, 1))))
        result = sharing.summarize_cut_robustness_matched_tasks([("run", data)], (3, 5))
        for cut in result["cuts"].values():
            self.assertEqual(cut["actual_k"], [1])
            self.assertEqual(cut["task_bias"], [0])
            self.assertEqual(cut["enrichment"]["period"], [1])

    def test_no_selected_pair_sharing_stays_undefined(self):
        names = np.array([f"{r}-{p}" for r in sharing.MATCHED_TASKS for p in sharing.MATCHED_PHASES]
                         + ["other-stim1", "other-delay1", "other-go1"])
        tree = linkage(np.r_[np.arange(1, 26), [0, 0, 0]][:, None])
        labels = fcluster(tree, 26, criterion="maxclust")
        data = {rep: {"tb_break_name": names, "row_labels": labels}
                for rep in sharing.REPRESENTATIONS}
        data["modulation"]["row_linkage"] = tree
        cut = sharing.summarize_cut_robustness_matched_tasks([("run", data)], (26,))["cuts"][26]
        self.assertEqual(cut["counts"]["run"]["any"]["n_shared"], 0)
        self.assertTrue(np.isnan(cut["task_bias"][0]))
        self.assertTrue(np.isnan(cut["enrichment"]["period"][0]))

    def test_rejects_wrong_tree_alignment_and_distinguishes_missing_trees(self):
        with self.assertRaises(sharing.MissingRowLinkage):
            sharing.summarize_cut_robustness_matched_tasks([("run", _complete_experiment())])
        data = _tree_experiment()
        # Swap leaves across clusters while preserving K. Singleton labels
        # are a valid maxclust cut in newer SciPy releases, not a bad tree.
        labels = data["modulation"]["row_labels"].copy()
        clusters, counts = np.unique(labels, return_counts=True)
        largest = clusters[np.argmax(counts)]
        i, j = np.flatnonzero(labels == largest)[0], np.flatnonzero(labels != largest)[0]
        labels[i], labels[j] = labels[j], labels[i]
        data["modulation"]["row_labels"] = labels
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            sharing.summarize_cut_robustness_matched_tasks([("run", data)])
        data["modulation"]["row_linkage"] = data["modulation"]["row_linkage"][:-1]
        with self.assertRaisesRegex(ValueError, "leaves do not align"):
            sharing.summarize_cut_robustness_matched_tasks([("run", data)])

    def test_update_uses_saved_cohort_and_keeps_previous_statistics(self):
        data = _tree_experiment()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sibling_row_sharing_L21e4.pkl"
            payload = {"version": sharing.RESULT_VERSION, "feature": "L21e4", "seed": None,
                       "variance_method": diagnosis.VARIANCE_METHOD,
                       "architecture": "", "mod_g_index": diagnosis.MOD_G_INDEX,
                       "experiments": ["run"], "old_statistics": {"keep": [0.1, 0.2]},
                       "organization": sharing.summarize_organization([("run", data)]),
                       "organization_matched_tasks": sharing.summarize_organization_matched_tasks(
                           [("run", data)])}
            with path.open("wb") as f:
                pickle.dump(payload, f)
            with patch.object(sharing, "OUT_DIR", Path(directory)), \
                    patch.object(diagnosis, "_load_experiment", return_value=data) as load, \
                    patch.object(diagnosis, "_discover_anames") as discover, patch("builtins.print"):
                updated = sharing.main(cut_robustness_only=True)
            discover.assert_not_called()
            load.assert_called_once_with("run", root=None)
            self.assertEqual(updated["experiments"], ["run"])
            cut_keys = ("cut_robustness_all", "cut_robustness_matched_tasks")
            for key in cut_keys:
                self.assertEqual(updated[key]["cuts"][sharing.CUT_ROBUSTNESS_K[0]]["anames"], ["run"])
            self.assertEqual({k: v for k, v in updated.items() if k not in cut_keys}, payload)
            saved = path.read_bytes()
            data["hidden"]["row_labels"] = np.ones(len(data["hidden"]["row_labels"]))
            with patch.object(sharing, "OUT_DIR", Path(directory)), \
                    patch.object(diagnosis, "_load_experiment", return_value=data), \
                    self.assertRaisesRegex(ValueError, "Cluster caches changed"):
                sharing.main(cut_robustness_only=True)
            self.assertEqual(path.read_bytes(), saved)

    def test_update_rejects_stale_task_or_phase_metadata_before_loading_runs(self):
        entry = sharing.summarize_organization_matched_tasks([("run", _tree_experiment())])
        for key in ("task_names", "phases"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sibling_row_sharing_L21e4.pkl"
                payload = {"version": sharing.RESULT_VERSION, "feature": "L21e4", "seed": None,
                           "variance_method": diagnosis.VARIANCE_METHOD,
                           "architecture": "", "mod_g_index": diagnosis.MOD_G_INDEX,
                           "experiments": ["run"],
                           "organization_matched_tasks": dict(entry, **{key: ["stale"]})}
                with path.open("wb") as f:
                    pickle.dump(payload, f)
                before = path.read_bytes()
                with patch.object(sharing, "OUT_DIR", Path(directory)), \
                        patch.object(diagnosis, "_load_experiment") as load, \
                        self.assertRaisesRegex(ValueError, "current matched-task summary"):
                    sharing.main(cut_robustness_only=True)
                load.assert_not_called()
                self.assertEqual(path.read_bytes(), before)


class DriverTests(unittest.TestCase):
    def test_one_missing_row_skips_all_supplements_without_dropping_networks(self):
        names = [f"everything_seed{i}_L21e4+hidden300+batch128+angle" for i in (1, 2)]
        complete, partial = _complete_experiment(), _complete_experiment()
        for rep, rows in partial.items():
            keep = rows["tb_break_name"] != "multidelaydm-delay2"
            partial[rep] = {key: value[keep] for key, value in rows.items()}
        runs = dict(zip(names, (complete, partial)))
        for data in (complete, partial):  # saved trees: the all-task cut needs no grid
            n_rows = len(data["modulation"]["row_labels"])
            tree = linkage(np.random.default_rng(n_rows).normal(size=(n_rows, 4)), method="average")
            data["modulation"].update(row_linkage=tree,
                                      row_labels=fcluster(tree, 5, criterion="maxclust"))
        supplement_keys = ("same_period_matched_tasks", "same_task_matched_tasks",
                           "organization_matched_tasks", "cut_robustness_matched_tasks")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result_path = out / "sibling_row_sharing_L21e4.pkl"
            # A previous successful supplement must not survive the new run.
            with result_path.open("wb") as stream:
                pickle.dump({key: "stale" for key in supplement_keys}, stream)
            with patch.object(diagnosis, "_discover_anames", return_value=names), \
                    patch.object(diagnosis, "_load_experiment", side_effect=lambda name, **kw: runs[name]), \
                    patch.object(sharing, "OUT_DIR", out), \
                    patch("builtins.print") as printed:
                sharing.main(n_permutations=50)
            self.assertIn("multidelaydm-delay2", str(printed.call_args_list))
            with result_path.open("rb") as stream:
                payload = pickle.load(stream)
            saved_json = json.loads((out / "sibling_row_sharing_L21e4.json").read_text())
            self.assertEqual(payload["experiments"], names)
            for key in supplement_keys:
                self.assertNotIn(key, payload)
                self.assertNotIn(key, saved_json)
            self.assertEqual(payload["cut_robustness_all"]["cuts"][sharing.CUT_ROBUSTNESS_K[0]]["anames"], names)
            self.assertEqual(saved_json["cut_robustness_all"]["k_values"],
                             list(sharing.CUT_ROBUSTNESS_K))
            for key in ("same_task_all", "same_period_all"):
                self.assertEqual(set(payload[key]["pooled"]["input"]["per_run"]), set(names))
            self.assertEqual(payload["organization"]["representations"]["input"]["anames"], names)

    def test_unexpected_supplement_errors_are_not_swallowed(self):
        with patch.object(diagnosis, "_discover_anames", return_value=["run"]), \
                patch.object(diagnosis, "_load_experiment", return_value=_complete_experiment()), \
                patch.object(sharing, "summarize_same_task_matched_tasks",
                             side_effect=ValueError("unexpected calculation error")), \
                patch.object(sharing, "_clear_previous_results") as clear, \
                patch("builtins.print"), self.assertRaisesRegex(ValueError, "unexpected calculation"):
            sharing.main(n_permutations=10)
        clear.assert_not_called()

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
        data = _complete_experiment()["input"]
        labels, names = data["row_labels"], data["tb_break_name"]
        neuron = {"variance_method": diagnosis.VARIANCE_METHOD,
                  "result": {"row_tol_labels": labels, "row_tol_k": 4, "col_tol_k": 6},
                  "tb_break_name": names}
        with open(run_dir / f"cluster_info_{aname}.pkl", "wb") as stream:
            pickle.dump({"input_normalized": neuron, "hidden_normalized": neuron}, stream)
        modulation = {"variance_method": diagnosis.VARIANCE_METHOD,
                      "result_all_lst": [None, {"row_tol_labels": labels, "row_tol_k": 4,
                                                "col_tol_k": 60}],
                      "tb_break_name": names}
        with open(run_dir / f"cluster_info_mod_{aname}.pkl", "wb") as stream:
            pickle.dump({"modulation_all_normalized": modulation}, stream)

    def test_writes_results_and_figures_and_keeps_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis, out = root / "analysis", root / "out"
            for seed in (1, 2, 3):
                self._write_run(analysis, f"everything_seed{seed}_L21e4+hidden300+batch128+angle")
            partial = "everything_seed4_L21e4+hidden300+batch128+angle"
            self._write_run(analysis, partial)
            (analysis / partial / f"cluster_info_mod_{partial}.pkl").unlink()
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
            self.assertEqual(payload["same_period_matched_tasks"]["tasks"],
                             list(sharing.MATCHED_TASKS))
            self.assertEqual(payload["same_period_matched_tasks"]["pooled"]["input"]["n_total"], 150)
            self.assertEqual(payload["same_task_matched_tasks"]["permutation"]["n_fdr_tests"], 18)
            self.assertEqual(payload["organization_matched_tasks"]["task_names"],
                             list(sharing.MATCHED_TASKS))
            for key in ("cut_robustness_all", "cut_robustness_matched_tasks"):
                self.assertNotIn(key, payload)  # older fixture has no trees
            counts = payload["cluster_counts"]["representations"]
            self.assertEqual(counts["input"]["col_k"], [6, 6, 6])
            self.assertEqual(counts["modulation"]["col_k"], [60, 60, 60])
            self.assertEqual(counts["modulation"]["row_k"], [4, 4, 4])
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
            "variance_method": diagnosis.VARIANCE_METHOD,
            "feature": paper_plot.multitask_feature(),
            "seed": None,
            "mod_g_index": diagnosis.MOD_G_INDEX,
            "experiments": [f"run{seed}" for seed in range(6)],
            "organization": sharing.summarize_organization(
                [(f"run{seed}", _organized_experiment()) for seed in range(6)]),
            "same_task_all": sharing.summarize_same_task_all(
                [(f"run{seed}", _task_experiment()) for seed in range(6)]),
            "same_period_all": sharing.summarize_same_period_all(_experiments()),
            "cluster_counts": sharing.summarize_cluster_counts(_counted_experiments()),
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
                    cluster_counts=sharing.summarize_cluster_counts(
                        _counted_experiments(col_offset=10)),
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

    def test_cut_plot_shows_shared_pair_fractions_per_class_on_one_axis(self):
        """One panel per pair class with the fraction of those pairs in one
        cluster, cohort means over networks, on a shared 0-1 axis."""
        payload = self._payload()
        payload["cut_robustness_matched_tasks"] = sharing.summarize_cut_robustness_matched_tasks(
            [(name, _tree_experiment()) for name in payload["experiments"]], (3, 5))
        save, _ = self._render(paper_plot.plot_row_cluster_cut_robustness_matched_tasks, payload)
        axes = save.call_args.args[0].axes
        self.assertEqual([ax.get_title().split(",")[0] for ax in axes],
                         ["Same task", "Same period", "Different task\nand period"])
        self.assertTrue(axes[0].get_shared_y_axes().joined(*axes[:2]))
        self.assertEqual(axes[0].get_ylim(), (0, 1))
        cuts = payload["cut_robustness_matched_tasks"]["cuts"]
        for ax, pair_class in zip(axes, ("task", "period", "neither")):
            bars = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
            self.assertEqual(len(bars), 1)  # trained only; the grey baseline is a plain line
            expected = [np.mean([counts[pair_class]["n_shared"] / counts[pair_class]["n_pairs"]
                                 for counts in cuts[k]["counts"].values()]) for k in (3, 5)]
            np.testing.assert_allclose(bars[0].lines[0].get_ydata(), expected)
            np.testing.assert_allclose(bars[0].lines[0].get_xdata(), [3, 5])
        bars = [c for c in axes[0].containers if isinstance(c, ErrorbarContainer)]
        # Identical networks: SD 0, so every whisker collapses onto its mean.
        heights = [seg[1, 1] - seg[0, 1] for seg in bars[0].lines[2][0].get_segments()]
        np.testing.assert_allclose(heights, 0, atol=1e-9)
        self.assertIn("SDs=", save.call_args.kwargs["extra"])
        self.assertIn("actual full-tree K ranges", save.call_args.kwargs["extra"])

    def test_coassignment_plot_uses_only_saved_trained_cohort_with_shared_axes_and_scale(self):
        payload = self._payload()
        experiments = [(name, _experiment()) for name in payload["experiments"]]
        payload["organization"] = sharing.summarize_organization(experiments)
        with patch.object(sharing.diag, "_load_experiment",
                          side_effect=lambda name, root: dict(experiments)[name]) as load:
            save, _ = self._render(paper_plot.plot_row_cluster_coassignment, payload)
        self.assertEqual([call.args[0] for call in load.call_args_list], payload["experiments"])
        self.assertTrue(all(call.kwargs == {"root": paper_plot.MULTITASK_ANALYSIS_ROOT}
                            for call in load.call_args_list))
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_coassignment.png")
        axes = save.call_args.args[0].axes[:3]
        ticklabels = [tick.get_text() for tick in axes[0].get_xticklabels()]
        row_names = [tick.get_gid() for tick in axes[0].get_xticklabels()]
        self.assertEqual(len(ticklabels), len(_NAMES))
        self.assertEqual(ticklabels[:2], ["DelayPro", "DelayPro"])
        self.assertEqual(row_names[:2], ["fdgo-stim1", "fdgo-go1"])
        # The integration-task rows must follow chronological order, not the
        # row order of a source cache or a representation-specific dendrogram.
        self.assertEqual([name for name in row_names if name.startswith("delaydm1-")],
                         ["delaydm1-" + phase for phase in
                          ("stim1", "delay1", "stim2", "delay2", "go1")])
        row_lookup = {name: i for i, name in enumerate(_NAMES)}
        order = [row_lookup[name] for name in row_names]
        labels = _experiment()["input"]["row_labels"][order]
        for ax in axes:
            self.assertEqual(ax.images[0].get_clim(), (0, 1))
            displayed = ax.images[0].get_array()
            np.testing.assert_array_equal(np.ma.getdata(displayed), labels[:, None] == labels)
            np.testing.assert_array_equal(np.ma.getmaskarray(displayed),
                                           np.triu(np.ones((len(labels), len(labels)), dtype=bool)))
            self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ticklabels)
            self.assertEqual(ax.images[0].get_cmap().name, paper_plot._MULTITASK_HEATMAP_CMAP)
            for tick, name in zip(ax.get_xticklabels(), row_names):
                np.testing.assert_allclose(tick.get_bbox_patch().get_facecolor(),
                                           paper_plot.mpl.colors.to_rgba(
                                               paper_plot._PHASE_COLORS[paper_plot._phase_of(name)],
                                               alpha=0.8))
        self.assertIn("Trained networks (n=6)", save.call_args.args[0]._suptitle.get_text())

    def test_coassignment_plot_skips_missing_or_changed_saved_cohort_member(self):
        payload = self._payload()
        payload["organization"] = sharing.summarize_organization(
            [(name, _experiment()) for name in payload["experiments"]])
        changed = copy.deepcopy(_experiment())
        changed["input"]["row_labels"][:] = 0
        for case, rows in enumerate((None, changed)):
            with self.subTest(case=case), \
                    patch.object(sharing.diag, "_load_experiment", return_value=rows):
                save, printed = self._render(paper_plot.plot_row_cluster_coassignment, payload)
                save.assert_not_called()
                self.assertIn("Skipped co-clustering heatmap", str(printed.call_args_list))

    def test_matched_cut_plot_ticks_carry_k_alone_and_log_the_25_row_counts(self):
        """The ticks show the target K as in the all-task twin; the clusters
        each cut forms among the selected rows go to the save message."""
        payload, control = self._payload(), self._untrained_payload()
        for saved in (payload, control):
            saved["cut_robustness_matched_tasks"] = sharing.summarize_cut_robustness_matched_tasks(
                [(name, _tree_experiment()) for name in saved["experiments"]], (3, 5))
        cuts = payload["cut_robustness_matched_tasks"]["cuts"]
        for k in (3, 5):  # one tree for every network: a single count, not a range
            self.assertEqual(len(set(cuts[k]["matched_k"])), 1)
            self.assertEqual(cuts[k]["actual_k"], [k] * len(payload["experiments"]))
        for args in ((payload, control), (payload,)):
            save, _ = self._render(paper_plot.plot_row_cluster_cut_robustness_matched_tasks, *args)
            for ax in save.call_args.args[0].axes:
                self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ["3", "5"])
                self.assertEqual(ax.get_xlabel(), "Full-tree cluster target K")
            extra = save.call_args.kwargs["extra"]
            self.assertIn("trained clusters among the 25 rows=" + str(
                [(cuts[k]["matched_k"][0], cuts[k]["matched_k"][0]) for k in (3, 5)]), extra)
            self.assertEqual("untrained clusters among the 25 rows=" in extra, len(args) == 2)

    def test_all_task_cut_plot_ticks_carry_k_alone(self):
        """Every row enters, so the ticks show the target K (with the full-tree
        count only when ties keep it below K) and no per-cohort row counts."""
        payload, control = self._payload(), self._untrained_payload()
        for saved in (payload, control):
            saved["cut_robustness_all"] = sharing.summarize_cut_robustness_all(
                [(name, _tree_experiment()) for name in saved["experiments"]], (3, 5))
        save, _ = self._render(paper_plot.plot_row_cluster_cut_robustness, payload, control)
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_cut_robustness.png")
        axes = save.call_args.args[0].axes
        for ax in axes:
            self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ["3", "5"])
            self.assertEqual(ax.get_xlabel(), "Full-tree cluster target K")
        bars = [c for c in axes[0].containers if isinstance(c, ErrorbarContainer)]
        self.assertEqual(len(bars), 2)  # trained, untrained, both at the same K
        np.testing.assert_allclose(bars[0].lines[0].get_xdata(), [3, 5])
        np.testing.assert_allclose(bars[1].lines[0].get_xdata(), [3, 5])
        self.assertTrue(save.call_args.args[0]._suptitle.get_text().startswith("All tasks"))
        extra = save.call_args.kwargs["extra"]
        self.assertIn("actual full-tree K ranges", extra)
        self.assertNotIn("clusters among", extra)

    def test_cluster_count_bars_show_the_selected_k_with_hollow_untrained_twins(self):
        payload, control = self._payload(), self._untrained_payload()
        save, printed = self._render(paper_plot.plot_row_cluster_counts_seeds, payload, control)
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_counts_seeds.png")
        axes = save.call_args.args[0].axes
        self.assertEqual([ax.get_title().split("\n")[0] for ax in axes],
                         ["Row clusters", "Column clusters"])
        for ax, key in zip(axes, ("row_k", "col_k")):
            bars = ax.patches
            self.assertEqual(len(bars), 6)  # three representations x (trained, untrained)
            # Medians, not means: the counts are skewed.
            expected = [np.median(saved["cluster_counts"]["representations"][rep][key])
                        for rep in ("input", "hidden", "modulation")
                        for saved in (payload, control)]
            np.testing.assert_allclose([bar.get_height() for bar in bars], expected)
            self.assertEqual(sum(bar.get_facecolor()[:3] == (1.0, 1.0, 1.0) for bar in bars), 3)
            # No per-network dots; one test mark per trained/untrained pair.
            self.assertFalse(any(isinstance(c, paper_plot.mpl.collections.PathCollection)
                                 for c in ax.collections))
            marks = [text.get_text() for text in ax.texts]
            self.assertEqual(len(marks), 3)
            self.assertTrue(set(marks) <= {"*", "n.s."})
            self.assertEqual(ax.get_ylim()[0], 0)
        extra = save.call_args.kwargs["extra"]
        self.assertIn("Mann-Whitney", extra)
        # The marks follow the BH q of the six two-sided Mann-Whitney tests.
        from scipy.stats import false_discovery_control, mannwhitneyu
        pairs = [(key, rep) for key in ("row_k", "col_k") for rep in ("input", "hidden", "modulation")]
        p_values = [mannwhitneyu(payload["cluster_counts"]["representations"][rep][key],
                                 control["cluster_counts"]["representations"][rep][key],
                                 alternative="two-sided").pvalue for key, rep in pairs]
        q_values = false_discovery_control(p_values, method="bh")
        drawn = [text.get_text() for ax in axes for text in ax.texts]
        self.assertEqual(drawn, ["*" if q < 0.05 else "n.s." for q in q_values])
        self.assertIn(f"q = {q_values[-1]:.2g}", extra)
        self.assertIn("Mann", save.call_args.args[0].legends[0].get_texts()[-1].get_text())
        self.assertIn("col_k modulation: trained median 71.5, range 69-74", extra)
        # Without the control: full-width trained bars only, and a request for it.
        del control["cluster_counts"]
        save, printed = self._render(paper_plot.plot_row_cluster_counts_seeds, payload, control)
        self.assertEqual(len(save.call_args.args[0].axes[0].patches), 3)
        self.assertEqual(len(save.call_args.args[0].axes[0].texts), 0)
        self.assertIn("--untrained", str(printed.call_args_list))
        self.assertNotIn("Mann-Whitney", save.call_args.kwargs["extra"])

    def test_cut_plot_rejects_changed_membership(self):
        payload = self._payload()
        payload["cut_robustness_matched_tasks"] = sharing.summarize_cut_robustness_matched_tasks(
            [("different-run", _tree_experiment())], (3, 5))
        with self.assertRaisesRegex(ValueError, "saved cohort"):
            self._render(paper_plot.plot_row_cluster_cut_robustness_matched_tasks, payload)

    def test_missing_cut_result_requests_a_command_its_cache_can_run(self):
        payload = self._payload()
        for has_matched in (False, True):
            with self.subTest(has_matched=has_matched):
                if has_matched:
                    payload["organization_matched_tasks"] = (
                        sharing.summarize_organization_matched_tasks([("run", _tree_experiment())]))
                for plot in (paper_plot.plot_row_cluster_cut_robustness_matched_tasks,
                             paper_plot.plot_row_cluster_cut_robustness):
                    save, printed = self._render(plot, payload)
                    save.assert_not_called()
                    message = str(printed.call_args_list)
                    self.assertIn("sibling_row_sharing.py --feature", message)
                    self.assertEqual("--cut-robustness-only" in message, has_matched)

    def test_missing_untrained_cut_requests_full_summary_without_matched_metadata(self):
        payload, control = self._payload(), self._untrained_payload()
        payload["cut_robustness_matched_tasks"] = sharing.summarize_cut_robustness_matched_tasks(
            [(name, _tree_experiment()) for name in payload["experiments"]], (3, 5))
        for has_matched in (False, True):
            with self.subTest(has_matched=has_matched):
                if has_matched:
                    control["organization_matched_tasks"] = (
                        sharing.summarize_organization_matched_tasks([("run", _tree_experiment())]))
                save, printed = self._render(
                    paper_plot.plot_row_cluster_cut_robustness_matched_tasks, payload, control)
                save.assert_called_once()
                message = str(printed.call_args_list)
                self.assertIn("--untrained", message)
                self.assertEqual("--cut-robustness-only" in message, has_matched)

    def test_organization_panels_show_period_and_task_on_log_axes(self):
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
        self.assertIn("is not a version-6", message)
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

    def test_matched_figure_reads_supplement_and_labels_fixed_task_population(self):
        entry = sharing.summarize_same_period_matched_tasks(
            [(f"run{i}", _complete_experiment()) for i in range(6)], 200)
        payload = dict(self._payload(), same_period_matched_tasks=entry)
        control = dict(self._untrained_payload(), same_period_matched_tasks=entry)
        save, printed = self._render(
            paper_plot.plot_row_cluster_same_period_matched_tasks_seeds, payload, control)
        save.assert_called_once()
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_same_period_matched_tasks_seeds.png")
        ax = save.call_args.args[0].axes[0]
        self.assertIn("Same five tasks", ax.get_title())
        self.assertIn("Original clustering retained", ax.get_title())
        self.assertEqual(len(ax.patches), 30)
        expected = [stats["pct_same_cluster"] for reps in entry["periods"].values()
                    for stats in reps.values()]
        np.testing.assert_allclose([bar.get_height() for bar in ax.patches[::2]], expected)
        self.assertIn("10 task pairs", save.call_args.kwargs["extra"])
        self.assertIn("paired contrast", str(printed.call_args_list))

    def test_synapse_comparison_uses_network_fractions_including_zeros(self):
        payload, control = self._payload(), self._untrained_payload()
        samples = ([0.0, 0.0, 100.0], [25.0, 75.0])
        for source, counts, expected in (
                (payload, [(0, 1), (0, 9), (1, 1)], samples[0]),
                (control, [(1, 4), (3, 4)], samples[1])):
            source["experiments"] = [f"run{i}" for i in range(len(counts))]
            for reps in source["same_period_all"]["periods"].values():
                reps["modulation"] = {
                    "pct_same_cluster": np.mean(expected), "chance_mean": 0.1,
                    "q_fdr": 0.0,  # old chance stars must not appear
                    "per_run": {f"run{i}": {"observed": observed, "n_cells": total,
                                              "expected": 0.1 * total}
                                for i, (observed, total) in enumerate(counts)}}
        with patch("scipy.stats.mannwhitneyu", wraps=mannwhitneyu) as compare:
            save, _ = self._render(paper_plot.plot_row_cluster_same_period_synapses,
                                   payload, control)
        # Five one-sided tests of the raw fractions, then the same five tests
        # of the fractions minus each network's own chance (10% here).
        self.assertEqual(compare.call_count, 10)
        for call in compare.call_args_list[:5]:
            for actual, expected in zip(call.args, samples):
                np.testing.assert_allclose(actual, expected)
            self.assertEqual(call.kwargs, {"alternative": "greater",
                                           "method": "asymptotic", "use_continuity": True})
        for call in compare.call_args_list[5:]:
            for actual, expected in zip(call.args, samples):
                np.testing.assert_allclose(actual, np.asarray(expected) - 10.0)
            self.assertEqual(call.kwargs, {"alternative": "greater",
                                           "method": "asymptotic", "use_continuity": True})
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_same_period_synapses.png")
        ax = save.call_args.args[0].axes[0]
        np.testing.assert_allclose([bar.get_height() for bar in ax.patches],
                                   [100 / 3, 50] * 5)
        self.assertLess(ax.get_ylim()[0], np.mean(samples[0]) - np.std(samples[0], ddof=1))
        # One mark per period: the trained values rank lower, so no star.
        self.assertEqual([text.get_text() for text in ax.texts], ["n.s."] * 5)
        self.assertIn("trained > untrained", ax.get_title())
        self.assertIn("one-sided", ax.get_title())
        # The figure has the row-sharing bar figures' width.
        figure = save.call_args.args[0]
        self.assertAlmostEqual(figure.get_size_inches()[0],
                               1.0 + paper_plot._ROW_SHARING_GROUP_IN[True] * 5)
        self.assertEqual(figure.axes[1].get_title(), "Chance-corrected sharing")
        expected_p = mannwhitneyu(*samples, alternative="greater",
                                  method="asymptotic").pvalue
        # The trained values rank lower here: the greater alternative must
        # give a large p, not half of the two-sided p regardless of direction.
        self.assertGreater(expected_p, 0.5)
        self.assertIn(f"p={expected_p:.6g}, q={expected_p:.6g}",
                      save.call_args.kwargs["extra"])
        self.assertIn("one-sided Mann-Whitney U, trained > untrained",
                      save.call_args.kwargs["extra"])

    def test_synapse_comparison_corrects_only_the_five_cohort_tests(self):
        # Different from all saved within-cohort p/q values; BH in period order.
        # The five raw-fraction tests come first, then the five
        # chance-corrected ones, each family adjusted on its own.
        with patch("scipy.stats.mannwhitneyu", side_effect=[
                SimpleNamespace(pvalue=p) for p in (0.04, 0.001, 0.03, 0.8, 0.2,
                                                    0.5, 0.001, 0.002, 0.3, 0.9)]):
            save, _ = self._render(paper_plot.plot_row_cluster_same_period_synapses,
                                   self._payload(), self._untrained_payload())
        # q = 0.0667, 0.005, 0.0667, 0.8, 0.25: a raw p of 0.04 loses its star.
        self.assertEqual([text.get_text() for text in save.call_args.args[0].axes[0].texts],
                         ["n.s.", "*", "n.s.", "n.s.", "n.s."])
        # q = 0.625, 0.005, 0.005, 0.5, 0.9.
        self.assertEqual([text.get_text() for text in save.call_args.args[0].axes[1].texts],
                         ["n.s.", "*", "*", "n.s.", "n.s."])
        extra = save.call_args.kwargs["extra"]
        self.assertIn("p=0.04, q=0.0666667", extra)
        self.assertIn("above chance: trained", extra)
        self.assertIn("p=0.002, q=0.005", extra)

    def test_synapse_comparison_skips_missing_control_or_period(self):
        payload, control = self._payload(), self._untrained_payload()
        del control["same_period_all"]["periods"]["stim1"]
        for untrained in (None, control):
            with self.subTest(has_control=untrained is not None):
                save, printed = self._render(paper_plot.plot_row_cluster_same_period_synapses,
                                             payload, untrained)
                save.assert_not_called()
                self.assertIn("Skipped synapse cohort comparison", str(printed.call_args_list))

    def test_matched_synapse_comparison_uses_matched_counts_and_its_own_five_tests(self):
        payload, control = self._payload(), self._untrained_payload()
        samples = []
        for source, observed in ((payload, [0, 1, 2, 3, 4, 5]),
                                 (control, [0, 0, 0, 0, 0, 1])):
            entry = sharing.summarize_same_period_matched_tasks(
                [(name, _complete_experiment()) for name in source["experiments"]], 200)
            source["same_period_matched_tasks"] = entry
            cohort_samples = []
            for period_index, reps in enumerate(entry["periods"].values()):
                counts = np.asarray(observed) + period_index
                cohort_samples.append(10.0 * counts)
                reps["modulation"] = {
                    "pct_same_cluster": 10.0 * counts.mean(), "chance_mean": 0.15,
                    "q_fdr": 0.0,  # within-cohort stars must not enter this comparison
                    "per_run": {name: {"observed": count, "n_cells": 10, "expected": 1.5}
                                for name, count in zip(source["experiments"], counts)}}
            samples.append(cohort_samples)
        with patch("scipy.stats.mannwhitneyu", wraps=mannwhitneyu) as compare, \
                patch("scipy.stats.false_discovery_control",
                      wraps=false_discovery_control) as adjust:
            save, _ = self._render(
                paper_plot.plot_row_cluster_same_period_synapses_matched_tasks, payload, control)
        # Five tests of the raw fractions, then five of the chance-corrected
        # ones (every network's null is 15% here), each family with its own BH.
        self.assertEqual(compare.call_count, 10)
        expected_p = []
        for call, group in zip(compare.call_args_list[:5], zip(*samples)):
            for actual, expected in zip(call.args, group):
                np.testing.assert_allclose(actual, expected)
            self.assertEqual(call.kwargs, {"alternative": "greater",
                                           "method": "asymptotic", "use_continuity": True})
            expected_p.append(mannwhitneyu(*group, **call.kwargs).pvalue)
        expected_excess_p = []
        for call, group in zip(compare.call_args_list[5:], zip(*samples)):
            for actual, expected in zip(call.args, group):
                np.testing.assert_allclose(actual, expected - 15.0)
            self.assertEqual(call.kwargs, {"alternative": "greater",
                                           "method": "asymptotic", "use_continuity": True})
            expected_excess_p.append(mannwhitneyu(*group, **call.kwargs).pvalue)
        self.assertEqual(adjust.call_count, 2)
        np.testing.assert_allclose(adjust.call_args_list[0].args[0], expected_p)
        np.testing.assert_allclose(adjust.call_args_list[1].args[0], expected_excess_p)
        self.assertEqual(adjust.call_args.kwargs, {"method": "bh"})
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_same_period_synapses_matched_tasks.png")
        ax = save.call_args.args[0].axes[0]
        np.testing.assert_allclose([bar.get_height() for bar in ax.patches],
                                   [values.mean() for group in zip(*samples) for values in group])
        np.testing.assert_allclose(
            [segment[:, 1] for collection in ax.collections[1::2]
             for segment in collection.get_segments()], [[15.0, 15.0]] * 10)
        self.assertEqual([text.get_text() for text in ax.texts],
                         ["*" if q < 0.05 else "n.s."
                          for q in false_discovery_control(expected_p, method="bh")])
        self.assertIn("Same five tasks", ax.get_title())
        self.assertIn("10 task pairs per period per network", save.call_args.kwargs["extra"])
        # Bottom panel: the same bars minus each network's own chance, with a
        # zero line and its own p/q annotations.
        bottom = save.call_args.args[0].axes[1]
        np.testing.assert_allclose([bar.get_height() for bar in bottom.patches],
                                   [values.mean() - 15.0 for group in zip(*samples) for values in group])
        self.assertEqual([text.get_text() for text in bottom.texts],
                         ["*" if q < 0.05 else "n.s."
                          for q in false_discovery_control(expected_excess_p, method="bh")])
        self.assertIn("Chance-corrected", bottom.get_title())
        self.assertTrue(any(np.allclose(line.get_ydata(), 0.0) for line in bottom.lines))
        self.assertIn("above chance: trained", save.call_args.kwargs["extra"])

    def test_matched_synapse_comparison_never_falls_back_to_all_task_counts(self):
        payload, control = self._payload(), self._untrained_payload()
        entry = sharing.summarize_same_period_matched_tasks(
            [(name, _complete_experiment()) for name in payload["experiments"]], 200)
        for case, (trained_entry, control_entry) in enumerate((
                (None, entry), (entry, None),
                (entry, dict(entry, task_names=list(reversed(entry["task_names"])))),
                (entry, dict(entry, phases=entry["phases"][:-1])))):
            with self.subTest(case=case):
                trained = dict(payload)
                untrained = dict(control)
                if trained_entry is not None:
                    trained["same_period_matched_tasks"] = trained_entry
                if control_entry is not None:
                    untrained["same_period_matched_tasks"] = control_entry
                save, printed = self._render(
                    paper_plot.plot_row_cluster_same_period_synapses_matched_tasks,
                    trained, untrained)
                save.assert_not_called()
                self.assertIn("Skipped", str(printed.call_args_list))

    def test_matched_task_and_organization_figures_use_their_own_entries(self):
        experiments = [(f"run{i}", _complete_experiment()) for i in range(6)]
        payload = dict(self._payload(),
                       same_task_matched_tasks=sharing.summarize_same_task_matched_tasks(
                           experiments, 200),
                       organization_matched_tasks=sharing.summarize_organization_matched_tasks(
                           experiments))
        control = dict(self._untrained_payload(), **{
            key: payload[key] for key in ("same_task_matched_tasks", "organization_matched_tasks")})
        save, _ = self._render(paper_plot.plot_row_cluster_same_task_matched_tasks_seeds,
                               payload, control)
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_same_task_matched_tasks_seeds.png")
        ax = save.call_args.args[0].axes[0]
        self.assertEqual(len(ax.patches), 30)
        self.assertEqual(len(ax.get_xticklabels()), 5)
        self.assertIn("Same five tasks", ax.get_title())
        self.assertNotIn("no period pairs", save.call_args.kwargs["extra"])

        # Every network stays in the linear plot and the cohort test, including
        # the networks with zero enrichment that a logarithmic plot would hide.
        reps = payload["organization_matched_tasks"]["representations"]
        for entry in reps.values():
            entry["enrichment"]["period"][0] = 0.0
        from scipy.stats import mannwhitneyu
        with patch("scipy.stats.mannwhitneyu", wraps=mannwhitneyu) as tested:
            save, _ = self._render(paper_plot.plot_row_cluster_organization_matched_tasks,
                                   payload, control)
        self.assertEqual(Path(save.call_args.args[1]).name,
                         "multitask_row_cluster_organization_matched_tasks.png")
        axes = save.call_args.args[0].axes
        self.assertEqual(axes[0].get_yscale(), "linear")
        self.assertLessEqual(axes[0].get_ylim()[0], 0)
        self.assertEqual(len(tested.call_args_list), 6)
        for call in tested.call_args_list[:3]:
            self.assertEqual(len(call.args[0]), 6)
            self.assertEqual(len(call.args[1]), 6)
            self.assertIn(0.0, call.args[0])
            self.assertIn(0.0, call.args[1])

    def test_new_matched_views_skip_missing_or_incompatible_controls(self):
        for plot, key, summarize in (
                (paper_plot.plot_row_cluster_same_task_matched_tasks_seeds,
                 "same_task_matched_tasks", sharing.summarize_same_task_matched_tasks),
                (paper_plot.plot_row_cluster_organization_matched_tasks,
                 "organization_matched_tasks", sharing.summarize_organization_matched_tasks)):
            with self.subTest(key=key):
                save, _ = self._render(plot, self._payload())
                save.assert_not_called()
                entry = summarize([("run", _complete_experiment())])
                control = dict(self._untrained_payload(), **{
                    key: dict(entry, task_names=["different_task"])})
                save, printed = self._render(plot, dict(self._payload(), **{key: entry}), control)
                save.assert_called_once()
                self.assertIn("--untrained", str(printed.call_args_list))

    def test_missing_supplement_does_not_fall_back_to_all_tasks(self):
        payload = self._payload()
        save, printed = self._render(
            paper_plot.plot_row_cluster_same_period_matched_tasks_seeds, payload)
        save.assert_not_called()
        self.assertIn("sibling_row_sharing.py --feature", str(printed.call_args_list))
        save, _ = self._render(paper_plot.plot_row_cluster_same_period_seeds, payload)
        save.assert_called_once()
        entry = sharing.summarize_same_period_matched_tasks(
            [("run", _complete_experiment())], 20)
        save, printed = self._render(
            paper_plot.plot_row_cluster_same_period_matched_tasks_seeds,
            dict(payload, same_period_matched_tasks=entry), self._untrained_payload())
        self.assertEqual(len(save.call_args.args[0].axes[0].patches), 15)
        self.assertIn("--untrained", str(printed.call_args_list))

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
            self.assertIn("is not a version-6", str(printed.call_args_list))

    def test_current_schema_without_estimator_provenance_requires_reclustering(self):
        payload = self._payload()
        del payload["variance_method"]
        save, printed = self._render(paper_plot.plot_row_cluster_same_period_seeds, payload)
        save.assert_not_called()
        self.assertIn("First rerun clustering", str(printed.call_args_list))
        control = self._untrained_payload()
        del control["variance_method"]
        save, printed = self._render(paper_plot.plot_row_cluster_same_period_seeds,
                                     self._payload(), untrained=control)
        save.assert_called_once()
        self.assertIn("untrained_clustering.py", str(printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
