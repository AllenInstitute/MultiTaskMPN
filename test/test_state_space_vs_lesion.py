"""Tests of the state-space-vs-lesion link: the Spearman Mantel test, the
analysis over the saved caches and the two paper figures that redraw it."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import spearmanr

import paper_plot
import state_space_vs_lesion as link
from core import state_space_geometry as geometry


RULES = list(paper_plot._RULE_MOTIF)
UPPER = np.triu_indices(len(RULES), 1)


def _distance_matrix(rng, n, dim=3):
    points = rng.normal(size=(n, dim))
    return np.linalg.norm(points[:, None] - points[None], axis=-1)


def _similarity_matrix(rng, n):
    profiles = rng.normal(size=(n, 6))
    return np.corrcoef(profiles)


def _structure_cache(aname, l2_info, seed):
    rng = np.random.default_rng(seed)
    return {"aname": aname, "l2_info": l2_info, "all_rules": list(RULES),
            "representations": {representation: {"distances": _distance_matrix(rng, len(RULES))}
                                for representation in link.REPRESENTATIONS}}


def _tasksim_cache(aname, tasks, similarity):
    return {"schema_version": 2, "aname": aname, "tasks": list(tasks),
            "causal_similarity": similarity, "y_definition": link.NEURON_SIMILARITY_DEFINITION}


def _effects_cache(aname, tasks, effect, conditions):
    return {"schema_version": 1, "aname": aname, "primary_modulation_mode": "zero_W",
            "entries": {link.SYNAPSE_EFFECT_ENTRY: {
                "effect": effect, "tasks": list(tasks), "conditions": list(conditions),
                "definition": "random_minus_lesion", "units": "fraction",
                "mod_lesion_mode": "zero_W"}}}


def _prune_results(conditions, unresponsive_label="missing"):
    record = {"all_comb_names_mod": ["mod_nolesion"] + list(conditions), "mod_lesion_mode": "zero_W"}
    if unresponsive_label != "missing":
        record["unresponsive_label"] = unresponsive_label
    return {"mod_lesion": {link.SYNAPSE_EFFECT_ENTRY: record}}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream)


class SpearmanMantelTests(unittest.TestCase):
    def test_rho_and_null_are_spearman_over_the_permuted_pairs(self):
        rng = np.random.default_rng(1)
        n = 7
        x = _distance_matrix(rng, n)
        y = _similarity_matrix(rng, n)
        result = geometry.mantel_spearman_permutation(x, y, n_permutations=40, seed=3)
        upper = np.triu_indices(n, 1)
        self.assertAlmostEqual(result["rho"], spearmanr(x[upper], y[upper]).correlation)
        expected_rng = np.random.default_rng(3)
        expected = []
        for _ in range(40):
            order = expected_rng.permutation(n)
            permuted = y[np.ix_(order, order)]
            expected.append(spearmanr(x[upper], permuted[upper]).correlation)
        np.testing.assert_allclose(result["null"], expected)
        self.assertEqual(result["p_value"],
                         (1 + np.sum(np.abs(expected) >= abs(result["rho"]))) / 41)
        self.assertEqual(result["statistic"], "spearman")
        self.assertEqual(result["sided"], "two-sided")
        self.assertNotIn("r", result)

    def test_rho_is_invariant_to_monotone_transforms_and_detects_perfect_order(self):
        rng = np.random.default_rng(0)
        n = 8
        x = _distance_matrix(rng, n)
        y = _similarity_matrix(rng, n)
        plain = geometry.mantel_spearman_permutation(x, y, n_permutations=30, seed=0)
        warped = geometry.mantel_spearman_permutation(x ** 3, np.exp(y), n_permutations=30, seed=0)
        self.assertAlmostEqual(plain["rho"], warped["rho"])
        np.testing.assert_allclose(plain["null"], warped["null"])
        reversed_order = geometry.mantel_spearman_permutation(x, -np.sqrt(x), n_permutations=999)
        self.assertAlmostEqual(reversed_order["rho"], -1.0)
        self.assertLess(reversed_order["p_value"], 0.01)
        with self.assertRaises(ValueError):
            geometry.mantel_spearman_permutation(x, x[:-1, :-1], n_permutations=10)


class LesionSimilarityTests(unittest.TestCase):
    aname = "everything_seed5_L21e4+hidden300+batch128+angle"
    conditions = ["mod_c1", "mod_c2", "mod_c3", "mod_c4", "mod_c5"]

    def write_synapse_caches(self, root, effect, unresponsive_label="missing"):
        _write(root / "norm" / self.aname / f"normalized_lesion_effects_{self.aname}.pkl",
               _effects_cache(self.aname, RULES, effect, self.conditions))
        _write(root / "perf" / self.aname / f"lesion_prune_results_{self.aname}.pkl",
               _prune_results(self.conditions, unresponsive_label))

    def test_synapse_similarity_drops_the_recorded_unresponsive_class(self):
        rng = np.random.default_rng(2)
        effect = rng.normal(size=(len(RULES), len(self.conditions)))
        effect[:, 1] = -0.8          # a huge constant column, as the real class has
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_synapse_caches(root, effect, unresponsive_label=2)
            tasks, similarity, excluded = link.synapse_cluster_similarity(
                self.aname, root / "norm", root / "perf")
            self.assertEqual(excluded, "mod_c2")
            np.testing.assert_allclose(similarity, np.corrcoef(effect[:, [0, 2, 3, 4]]))
            self.assertEqual(tasks, RULES)
            # No flagged class: every cluster stays.
            self.write_synapse_caches(root, effect, unresponsive_label=None)
            _, similarity, excluded = link.synapse_cluster_similarity(
                self.aname, root / "norm", root / "perf")
            self.assertIsNone(excluded)
            np.testing.assert_allclose(similarity, np.corrcoef(effect))
            # Legacy lesion pickles without the label: the last cluster, as
            # lesion_plot reads them.
            self.write_synapse_caches(root, effect)
            _, similarity, excluded = link.synapse_cluster_similarity(
                self.aname, root / "norm", root / "perf")
            self.assertEqual(excluded, "mod_c5")
            np.testing.assert_allclose(similarity, np.corrcoef(effect[:, :4]))
            # A label that names no saved condition is an error, not silently kept.
            self.write_synapse_caches(root, effect, unresponsive_label=9)
            with self.assertRaisesRegex(ValueError, "mod_c9"):
                link.synapse_cluster_similarity(self.aname, root / "norm", root / "perf")
            self.assertIsNone(link.synapse_cluster_similarity("other_run", root / "norm", root / "perf"))

    def test_neuron_similarity_requires_the_current_cache_schema(self):
        rng = np.random.default_rng(3)
        similarity = _similarity_matrix(rng, len(RULES))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / self.aname / f"causal_vs_activity_tasksim_hidden_{self.aname}.pkl"
            _write(path, _tasksim_cache(self.aname, RULES, similarity))
            tasks, loaded = link.neuron_cluster_similarity(self.aname, root)
            self.assertEqual(tasks, RULES)
            np.testing.assert_array_equal(loaded, similarity)
            legacy = _tasksim_cache(self.aname, RULES, similarity)
            legacy["schema_version"] = 1
            _write(path, legacy)
            with self.assertRaisesRegex(ValueError, "legacy"):
                link.neuron_cluster_similarity(self.aname, root)
            other = _tasksim_cache(self.aname, RULES, similarity)
            other["y_definition"] = "something else"
            _write(path, other)
            with self.assertRaisesRegex(ValueError, "definition"):
                link.neuron_cluster_similarity(self.aname, root)
            self.assertIsNone(link.neuron_cluster_similarity("other_run", root))


class AnalyzeRunTests(unittest.TestCase):
    def test_entry_aligns_lesion_tasks_and_tests_every_pairing(self):
        aname = "everything_seed7_L21e4+hidden300+batch128+angle"
        cache = _structure_cache(aname, 1e-4, 0)
        rng = np.random.default_rng(4)
        neuron = _similarity_matrix(rng, len(RULES))
        # The lesion caches list the tasks in another order; the entry must
        # follow the structure cache's order.
        order = rng.permutation(len(RULES))
        shuffled_tasks = [RULES[index] for index in order]
        synapse = -cache["representations"]["eff_mod"]["distances"]      # a perfect inverse order
        entry = link.analyze_run(
            aname, 1e-4, RULES, {name: value["distances"] for name, value in cache["representations"].items()},
            {"neuron": (shuffled_tasks, neuron[np.ix_(order, order)]),
             "synapse": (RULES, synapse, "mod_c21")},
            n_permutations=25, seed=1)
        self.assertEqual(entry["aname"], aname)
        self.assertEqual(entry["l2_info"], 1e-4)
        self.assertEqual(len(entry["pairs"]), len(UPPER[0]))
        self.assertEqual(entry["pairs"][0], (RULES[0], RULES[1]))
        self.assertEqual(entry["excluded_conditions"], {"synapse": "mod_c21"})
        np.testing.assert_allclose(entry["similarities"]["neuron"], neuron[UPPER])
        for representation in link.REPRESENTATIONS:
            distances = cache["representations"][representation]["distances"]
            np.testing.assert_allclose(entry["distances"][representation], distances[UPPER])
            for cluster_type, similarity in (("neuron", neuron), ("synapse", synapse)):
                test = entry["tests"][representation][cluster_type]
                self.assertEqual(test["statistic"], "spearman")
                self.assertEqual(test["n_permutations"], 25)
                self.assertAlmostEqual(
                    test["rho"], spearmanr(distances[UPPER], similarity[UPPER]).correlation)
        self.assertAlmostEqual(entry["tests"]["eff_mod"]["synapse"]["rho"], -1.0)
        self.assertLess(entry["tests"]["eff_mod"]["synapse"]["p_value"], 0.1)
        with self.assertRaisesRegex(ValueError, "different tasks"):
            link.analyze_run(aname, 1e-4, RULES,
                             {"hidden": cache["representations"]["hidden"]["distances"]},
                             {"neuron": (RULES[:-1] + ["unknown"], neuron)}, n_permutations=5)


class RunAllTests(unittest.TestCase):
    def test_run_all_analyzes_runs_with_both_caches_and_writes_the_results(self):
        with_lesions = "everything_seed1_L21e4+hidden300+batch128+angle"
        without = "everything_seed2_L21e3+hidden300+batch128+angle"
        rng = np.random.default_rng(5)
        conditions = [f"mod_c{k}" for k in range(1, 6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_space = root / "state_space"
            for aname, l2_info, seed in ((with_lesions, 1e-4, 0), (without, 1e-3, 1)):
                _write(state_space / f"state_space_structure_{aname}_noise0.01.pkl",
                       _structure_cache(aname, l2_info, seed))
            (state_space / "state_space_structure_other_noise0.1.pkl").write_bytes(b"other noise")
            _write(root / "norm" / with_lesions / f"causal_vs_activity_tasksim_hidden_{with_lesions}.pkl",
                   _tasksim_cache(with_lesions, RULES, _similarity_matrix(rng, len(RULES))))
            _write(root / "norm" / with_lesions / f"normalized_lesion_effects_{with_lesions}.pkl",
                   _effects_cache(with_lesions, RULES, rng.normal(size=(len(RULES), 5)), conditions))
            _write(root / "perf" / with_lesions / f"lesion_prune_results_{with_lesions}.pkl",
                   _prune_results(conditions, 5))
            with patch("builtins.print") as output:
                results = link.run_all(state_space, root / "norm", root / "perf",
                                       n_permutations=12, seed=2)
            with (state_space / link.RESULT_NAME).open("rb") as stream:
                saved = pickle.load(stream)
            printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
            # Without lesion caches anywhere, the run is an error, not an empty file.
            with self.assertRaisesRegex(FileNotFoundError, "lesion_plot"):
                link.run_all(state_space, root / "elsewhere", root / "perf", n_permutations=3)
            with self.assertRaisesRegex(FileNotFoundError, "state_space_structure"):
                link.run_all(root / "empty", root / "norm", root / "perf", n_permutations=3)
        self.assertEqual(set(results), {with_lesions})
        self.assertEqual(saved.keys(), results.keys())
        entry = saved[with_lesions]
        self.assertEqual(entry["n_permutations"], 12)
        self.assertEqual(entry["permutation_seed"], 2)
        self.assertEqual(entry["excluded_conditions"], {"synapse": "mod_c5"})
        self.assertEqual(entry["tests"]["hidden"]["neuron"]["null"].shape, (12,))
        self.assertIn(f"{without}: no lesion caches", printed)
        self.assertIn("L2 = 0.0001: 1 runs", printed)
        self.assertIn("eff_mod distance vs synapse lesion similarity", printed)


class PaperFigureTests(unittest.TestCase):
    def results(self, anames, seed=0):
        rng = np.random.default_rng(seed)
        combined = {}
        for aname in anames:
            cache = _structure_cache(aname, 1e-4, int(rng.integers(1000)))
            distances = {name: value["distances"] for name, value in cache["representations"].items()}
            combined[aname] = link.analyze_run(
                aname, 1e-4, RULES, distances,
                {"neuron": (RULES, _similarity_matrix(rng, len(RULES))),
                 "synapse": (RULES, -distances["eff_mod"] + 0.1 * _similarity_matrix(rng, len(RULES)),
                             "mod_c21")},
                n_permutations=20, seed=0)
        return combined

    def render(self, function, combined, anames, show_legend=True):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for aname in anames:
                (base / aname).mkdir()
            result_path = base / "results.pkl"
            if combined is not None:
                _write(result_path, combined)
            with patch.object(paper_plot, "STATE_SPACE_VS_LESION_RESULT_PATH", result_path), \
                    patch.object(paper_plot, "LESION_NORM_DIR", base / paper_plot.LESION_ANAME), \
                    patch.object(paper_plot, "SHOW_LEGEND", show_legend), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as output:
                function()
        for call in save.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        return save, output

    def test_scatter_draws_the_saved_pairs_with_the_saved_statistics(self):
        collections = paper_plot.mpl.collections
        aname = paper_plot.LESION_ANAME
        combined = self.results([aname])
        save, _ = self.render(paper_plot.plot_state_space_vs_lesion, combined, [aname])
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, "multitask_state_space_vs_lesion.png")
        # Rows: hidden-state then W * M distance; columns: neuron then synapse
        # clusters. Column titles sit on the top row only.
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["Neuron clusters", "Synapse clusters", "", ""])
        entry = combined[aname]
        panels = [(representation, cluster_type)
                  for representation in ("hidden", "eff_mod") for cluster_type in ("neuron", "synapse")]
        for axis, (representation, cluster_type) in zip(figure.axes, panels):
            points = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
            self.assertEqual(len(points), 1)
            offsets = points[0].get_offsets()
            np.testing.assert_allclose(offsets[:, 0], entry["distances"][representation])
            np.testing.assert_allclose(offsets[:, 1], entry["similarities"][cluster_type])
            self.assertEqual(len(axis.lines), 0)            # no fitted line for a rank statistic
            test = entry["tests"][representation][cluster_type]
            legend_text = axis.get_legend().get_texts()[0].get_text()
            self.assertIn(f"{test['rho']:.2f}", legend_text)
            self.assertIn(paper_plot._permutation_p_text(test["p_value"]), legend_text)
            self.assertIn(paper_plot._SSL_DISTANCE_LABELS[representation], axis.get_xlabel())
            # Undecorated dots and the scatter family's tick size.
            self.assertEqual(len(points[0].get_edgecolors()), 0)
            self.assertTrue(all(label.get_fontsize() == paper_plot._STAT_SCATTER_TICK_FONTSIZE
                                for label in axis.get_xticklabels() + axis.get_yticklabels()))
        # Each panel is one single-scatter square.
        np.testing.assert_allclose(figure.get_size_inches(),
                                   2 * np.asarray(paper_plot._LESION_SCATTER_FIGSIZE))
        self.assertEqual([axis.get_ylabel() != "" for axis in figure.axes], [True, False, True, False])
        self.assertAlmostEqual(entry["tests"]["eff_mod"]["synapse"]["rho"], -1.0, places=0)
        self.assertIn("eff_mod/synapse: rho = -", save.call_args.kwargs["extra"])

    def test_scatter_is_skipped_without_results_for_the_selected_run(self):
        aname = paper_plot.LESION_ANAME
        sibling = aname.replace("seed921", "seed11") if "seed921" in aname else aname.replace("seed", "seed1", 1)
        save, output = self.render(paper_plot.plot_state_space_vs_lesion, None, [aname])
        save.assert_not_called()
        save, output = self.render(paper_plot.plot_state_space_vs_lesion, self.results([sibling]), [aname])
        save.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("state_space_vs_lesion.py", printed)
        broken = self.results([aname])
        del broken[aname]["tests"]["hidden"]["synapse"]
        save, output = self.render(paper_plot.plot_state_space_vs_lesion, broken, [aname])
        save.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("incompatible", printed)

    def test_seed_figure_draws_every_sibling_run_with_the_cohort_test(self):
        import re

        collections = paper_plot.mpl.collections
        aname = paper_plot.LESION_ANAME
        anames = [re.sub(r"seed\d+", f"seed{seed}", aname) for seed in (3, 5, 8)]
        combined = self.results(anames)
        combined["everything_seed4_L21e9+hidden300+batch128+angle"] = combined[anames[0]]  # another cohort
        save, _ = self.render(paper_plot.plot_state_space_vs_lesion_seeds, combined, anames)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, "multitask_state_space_vs_lesion_seeds.png")
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["Neuron clusters", "Synapse clusters"])
        for axis, cluster_type in zip(figure.axes, ("neuron", "synapse")):
            dots = [c for c in axis.collections if isinstance(c, collections.PathCollection)]
            self.assertEqual(len(dots), 3 * 2)              # 3 runs x 2 representations
            offsets = np.concatenate([c.get_offsets() for c in dots])
            for position, representation in enumerate(("hidden", "eff_mod")):
                tests = [combined[name]["tests"][representation][cluster_type] for name in anames]
                rhos = sorted(test["rho"] for test in tests)
                drawn = sorted(offsets[np.isclose(offsets[:, 0], position, atol=0.3), 1])
                np.testing.assert_allclose(drawn, rhos)
                filled = sum(np.allclose(c.get_facecolors()[0][:3],
                                         paper_plot.mpl.colors.to_rgb(paper_plot._LESION_POINT_COLOR))
                             for c in dots if np.isclose(c.get_offsets()[0][0], position, atol=0.3))
                self.assertEqual(filled, sum(test["p_value"] < 0.05 for test in tests))
                label = axis.get_xticklabels()[position].get_text()
                self.assertIn(paper_plot._wilcoxon_vs_zero([test["rho"] for test in tests])[1], label)
            dashes = [c for c in axis.collections if isinstance(c, collections.LineCollection)]
            medians = sorted(c.get_segments()[0][0, 1] for c in dashes)
            expected = sorted(np.median([combined[name]["tests"][representation][cluster_type]["rho"]
                                         for name in anames])
                              for representation in ("hidden", "eff_mod"))
            np.testing.assert_allclose(medians, expected)
        self.assertIn("Hidden", figure.axes[0].get_xticklabels()[0].get_text())
        self.assertEqual(len(figure.legends), 1)
        self.assertIn("3 runs", save.call_args.kwargs["extra"])
        # One run is not a cohort.
        save, output = self.render(paper_plot.plot_state_space_vs_lesion_seeds,
                                   self.results(anames[:1]), anames[:1])
        save.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("need >= 2", printed)

    def test_figures_are_registered_in_the_lesion_mode(self):
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["state_space_vs_lesion"],
                      paper_plot.plot_state_space_vs_lesion)
        self.assertIs(paper_plot.FIGURES_BY_MODE["lesion"]["state_space_vs_lesion_seeds"],
                      paper_plot.plot_state_space_vs_lesion_seeds)


if __name__ == "__main__":
    unittest.main()
