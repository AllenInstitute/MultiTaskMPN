"""Balanced and incomplete task/period partitions, shared updates and rendering."""
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import false_discovery_control, mannwhitneyu

import multiple_task_diagnosis as diag
import sibling_row_sharing as sharing
import paper_plot


def _data(seed=0):
    names = np.array([f"{task}-{period}" for task in sharing.MATCHED_TASKS
                      for period in sharing.MATCHED_PHASES])
    a = np.arange(5, dtype=float)[:, None] - 2
    b = a.T
    return {rep: {"tb_break_name": names.copy(), "row_labels": np.repeat(np.arange(5), 5),
                  "variance_features": (0.5 + (0.04 + 0.003 * seed) * a
                                        + (0.03 + 0.006 * i) * b + 0.01 * a * b).reshape(25, 1)}
            for i, rep in enumerate(sharing.REPRESENTATIONS)}


def _payload(runs, untrained=False):
    return {"version": sharing.RESULT_VERSION, "variance_method": diag.VARIANCE_METHOD,
            "feature": None if untrained else "L21e4", "architecture": "",
            "seed": None, "untrained": untrained, "mod_g_index": diag.MOD_G_INDEX,
            "experiments": [name for name, _ in runs],
            "organization": sharing.summarize_organization(runs),
            "previous_results": {"untouched": [1, 2, 3]}}


def _entry(runs, matched=True):
    return sharing.summarize_variance_partitions([
        (name, sharing._partition_features(name, data, matched=matched))
        for name, data in runs], matched=matched)


def _write_cache(root, name, data):
    folder = root / name
    folder.mkdir(parents=True)
    entries = {}
    for rep, rows in data.items():
        entry = {"variance_method": diag.VARIANCE_METHOD,
                 "tb_break_name": rows["tb_break_name"],
                 "result": {"row_tol_labels": rows["row_labels"]}}
        if "variance_features" in rows:
            entry["cell_vars_rules_sorted_norm"] = rows["variance_features"]
        entries[rep] = entry
    with (folder / f"cluster_info_{name}.pkl").open("wb") as stream:
        pickle.dump({f"{rep}_normalized": entries[rep] for rep in ("input", "hidden")}, stream)
    mod = entries["modulation"]
    mod["result_all_lst"] = [None, mod.pop("result")]
    with (folder / f"cluster_info_mod_{name}.pkl").open("wb") as stream:
        pickle.dump({"modulation_all_normalized": mod}, stream)


class PartitionTests(unittest.TestCase):
    def test_general_model_recovers_balanced_orthogonal_decomposition(self):
        values = np.random.default_rng(19).normal(size=(5, 5, 7))
        centered = values - values.mean(axis=(0, 1), keepdims=True)
        task = centered.mean(axis=1, keepdims=True)
        period = centered.mean(axis=0, keepdims=True)
        expected = [5 * np.sum(task ** 2), 5 * np.sum(period ** 2),
                    np.sum((centered - task - period) ** 2), np.sum(centered ** 2)]
        grid = sharing.partition_task_period_variance(values)
        rows = sharing.partition_task_period_variance(
            values.reshape(25, 7), np.repeat(np.arange(5), 5), np.tile(np.arange(5), 5))
        for result in (grid, rows):
            np.testing.assert_allclose(list(result["ss"].values()), expected, rtol=1e-12)

    def test_incomplete_grid_uses_observed_rows_and_average_entry_order(self):
        # Only A/x, A/y, B/y exist. The response depends purely on task, but
        # period explains part of it through the unbalanced task-period design.
        values = np.array([[0.], [0.], [1.]])
        tasks, periods = np.array(["A", "A", "B"]), np.array(["x", "y", "y"])
        result = sharing.partition_task_period_variance(values, tasks, periods)
        np.testing.assert_allclose(list(result["fractions"].values()), [7/8, 1/8, 0], atol=1e-14)
        np.testing.assert_allclose(list(result["model_ss"].values()), [2/3, 1/6, 2/3])
        reverse = sharing.partition_task_period_variance(values[::-1], tasks[::-1], periods[::-1])
        np.testing.assert_allclose(list(reverse["fractions"].values()), [7/8, 1/8, 0], atol=1e-14)
        swapped = sharing.partition_task_period_variance(values, periods, tasks)
        np.testing.assert_allclose(list(swapped["fractions"].values()), [1/8, 7/8, 0], atol=1e-14)
        # Filling the absent B/x cell with zero would instead give thirds.
        padded = sharing.partition_task_period_variance(np.array([0., 0., 0., 1.]).reshape(2, 2, 1))
        np.testing.assert_allclose(list(padded["fractions"].values()), [1/3] * 3)

    def test_rank_deficient_confounded_factors_split_shared_information_equally(self):
        result = sharing.partition_task_period_variance(
            np.array([[0., 1.], [2., 2.], [4., 5.]]), ["A", "B", "C"], ["x", "y", "z"])
        np.testing.assert_allclose(list(result["fractions"].values()), [0.5, 0.5, 0], atol=1e-14)

    def test_pure_effects_and_known_mixture(self):
        a = np.arange(5, dtype=float)[:, None] - 2
        b = a.T
        terms = [np.broadcast_to(a, (5, 5)), np.broadcast_to(b, (5, 5)), a * b]
        for i, term in enumerate(terms):
            result = sharing.partition_task_period_variance(term[..., None])
            np.testing.assert_allclose(list(result["fractions"].values()), np.eye(3)[i], atol=1e-14)
        # Distinct feature dimensions and unequal amplitudes: sum SS before
        # dividing, rather than averaging each feature's explained fraction.
        values = np.stack([terms[0], 2 * terms[1], 3 * terms[2]], axis=-1)
        result = sharing.partition_task_period_variance(values)
        np.testing.assert_allclose(list(result["ss"].values()), [50, 200, 900, 1150])
        np.testing.assert_allclose(list(result["fractions"].values()), np.array([50, 200, 900]) / 1150)
        shifted = sharing.partition_task_period_variance(values + [3., 10., 70.])
        np.testing.assert_allclose(list(result["ss"].values()), list(shifted["ss"].values()))

    def test_constant_profiles_are_undefined_and_invalid_inputs_rejected(self):
        result = sharing.partition_task_period_variance(np.full((5, 5, 2), [0.3, 0.7]))
        self.assertEqual(result["ss"]["total"], 0)
        self.assertTrue(np.isnan(list(result["fractions"].values())).all())
        for bad in (np.zeros((5, 5)), np.zeros((1, 5, 2)), np.zeros((5, 5, 0)),
                    np.full((5, 5, 1), np.nan), np.full((5, 5, 1), np.inf)):
            with self.assertRaises(ValueError):
                sharing.partition_task_period_variance(bad)

    def test_named_rows_are_aligned_independently_and_columns_are_not_renormalized(self):
        data = _data()
        expected = sharing._partition_features("run", data)
        rng = np.random.default_rng(23)
        for rows in data.values():
            order = rng.permutation(25)
            for key in rows:
                rows[key] = rows[key][order]
        actual = sharing._partition_features("run", data)
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(actual["input"]["ss"]["task"], 50 * 0.04 ** 2)

    def test_missing_and_malformed_features_are_distinct(self):
        for kind in ("features", "row", "duplicate", "nan", "shape"):
            with self.subTest(kind=kind):
                data = _data()
                rows = data["hidden"]
                if kind == "features":
                    del rows["variance_features"]
                    error = sharing.MissingVarianceFeatures
                elif kind == "row":
                    rows["tb_break_name"] = rows["tb_break_name"][:-1]
                    error = sharing.MissingMatchedTaskRows
                else:
                    error = ValueError
                    if kind == "duplicate":
                        rows["tb_break_name"][1] = rows["tb_break_name"][0]
                    elif kind == "nan":
                        rows["variance_features"][0] = np.nan
                    else:
                        rows["variance_features"] = np.zeros((24, 1))
                with self.assertRaises(error):
                    sharing._partition_features("run", data)

    def test_missing_matched_cell_keeps_all_observed_rows_and_requires_common_coverage(self):
        data = _data()
        for rows in data.values():
            for key in rows:
                rows[key] = rows[key][:-1]
        with patch("builtins.print"):
            views = sharing._partition_variance_views("partial", data)
            entries = sharing._summarize_variance_views([("partial", views)])
        self.assertEqual(set(entries), {"variance_partition_all"})
        self.assertEqual(len(entries["variance_partition_all"]["row_names"]), 24)
        with self.assertRaisesRegex(ValueError, "identical task-period rows"):
            _entry([("partial", data), ("complete", _data())], matched=False)
        data["hidden"]["tb_break_name"][0] = "different-stim1"
        with self.assertRaisesRegex(ValueError, "inconsistent all-task feature rows"):
            sharing._partition_features("invalid", data, matched=False)


class CacheTests(unittest.TestCase):
    def test_feature_loading_is_optional_and_preserves_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root, data = Path(directory), _data()
            _write_cache(root, "run", data)
            small = diag._load_experiment("run", root=root)
            full = diag._load_experiment("run", root=root, include_features=True)
            for rep in sharing.REPRESENTATIONS:
                self.assertNotIn("variance_features", small[rep])
                np.testing.assert_array_equal(full[rep]["variance_features"], data[rep]["variance_features"])
                np.testing.assert_array_equal(full[rep]["tb_break_name"], data[rep]["tb_break_name"])

    def test_update_uses_saved_cohort_and_preserves_all_other_results(self):
        for untrained in (False, True):
            with self.subTest(untrained=untrained), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runs = [("run2", _data(2)), ("run1", _data(1))]
                for name, data in runs + [("new_run", _data(3))]:
                    _write_cache(root, name, data)
                payload = _payload(runs, untrained)
                tag = "untrained" if untrained else "L21e4"
                path = root / f"sibling_row_sharing_{tag}.pkl"
                path.write_bytes(pickle.dumps(payload))
                with patch.object(sharing, "OUT_DIR", root), \
                        patch.object(sharing, "UNTRAINED_DIR", root), \
                        patch.object(diag, "MULTI_DIR", root), \
                        patch.object(diag, "_discover_anames") as discover, \
                        patch.object(sharing, "_sharing_permutations") as permute, \
                        patch("builtins.print"):
                    result = sharing.main(untrained=untrained, variance_partition_only=True)
                discover.assert_not_called()
                permute.assert_not_called()
                for key, matched in sharing.VARIANCE_VIEWS.items():
                    entry = result.pop(key)
                    self.assertEqual(entry["representations"]["input"]["anames"], ["run2", "run1"])
                    self.assertEqual(entry, _entry(runs, matched=matched))
                self.assertEqual(result, payload)
                saved = json.loads(path.with_suffix(".json").read_text())
                self.assertTrue(sharing.VARIANCE_VIEWS.keys() <= saved.keys())

    def test_failed_update_keeps_previous_files(self):
        for kind in ("missing", "stale", "changed", "nan"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root, data = Path(directory), _data()
                payload = _payload([("run", copy.deepcopy(data))])
                if kind == "missing":
                    del data["hidden"]["variance_features"]
                elif kind == "stale":
                    payload["variance_method"] = "old"
                elif kind == "changed":
                    data["hidden"]["row_labels"] = np.arange(25)
                else:
                    data["hidden"]["variance_features"][0] = np.nan
                _write_cache(root, "run", data)
                path = root / "sibling_row_sharing_L21e4.pkl"
                before = pickle.dumps(payload)
                path.write_bytes(before)
                path.with_suffix(".json").write_text("previous JSON")
                with patch.object(sharing, "OUT_DIR", root), patch.object(diag, "MULTI_DIR", root), \
                        self.assertRaises(ValueError):
                    sharing.main(variance_partition_only=True)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(path.with_suffix(".json").read_text(), "previous JSON")

    def test_normal_driver_adds_or_skips_the_whole_supplement(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                names = [f"everything_seed{i}_L21e4+hidden300+batch128+angle" for i in (1, 2)]
                for i, name in enumerate(names):
                    data = _data(i)
                    if missing and i == 1:
                        del data["modulation"]["variance_features"]
                    _write_cache(root, name, data)
                with patch.object(sharing, "OUT_DIR", root), patch.object(diag, "MULTI_DIR", root), \
                        patch("builtins.print"):
                    result = sharing.main(n_permutations=9)
                self.assertEqual(result["experiments"], names)
                for key in sharing.VARIANCE_VIEWS:
                    self.assertEqual(key in result, not missing)
                self.assertIn("same_period_all", result)


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.runs = [(f"run{i}", _data(i)) for i in range(4)]
        self.control_runs = [(f"init{i}", _data(i + 7)) for i in range(4)]
        self.payload = _payload(self.runs)
        self.control = _payload(self.control_runs, True)
        self.entry, self.control_entry = _entry(self.runs), _entry(self.control_runs)

    def test_six_test_correction_matches_direct_network_comparisons(self):
        a = paper_plot._variance_partition_samples(self.entry, self.payload["experiments"])
        b = paper_plot._variance_partition_samples(self.control_entry, self.control["experiments"])
        tests = paper_plot._variance_partition_comparisons(a, b)
        p = [mannwhitneyu(a[rep][:, i], b[rep][:, i], alternative="two-sided",
                          method="asymptotic", use_continuity=True).pvalue
             for rep in sharing.REPRESENTATIONS for i in range(2)]
        np.testing.assert_allclose([test["p"] for test in tests.values()], p)
        np.testing.assert_allclose([test["q"] for test in tests.values()], false_discovery_control(p))

    def test_plot_bars_match_equal_network_means_and_only_six_tests(self):
        for view, matched in sharing.VARIANCE_VIEWS.items():
            with self.subTest(view=view):
                entries = [_entry(runs, matched) for runs in (self.runs, self.control_runs)]
                with patch.object(paper_plot, "_load_row_sharing_view", return_value=(
                        self.payload, entries[0], self.control, entries[1])), \
                        patch.object(paper_plot, "_save_fig") as save:
                    getattr(paper_plot, f"plot_{view}")()
                fig, path = save.call_args.args[:2]
                self.addCleanup(paper_plot.plt.close, fig)
                self.assertEqual(Path(path).name, f"multitask_{view}.png")
                self.assertEqual(len(fig.axes), 1)
                ax = fig.axes[0]
                expected = [100 * np.mean(entry["representations"][rep]["fractions"][key])
                            for key in sharing.VARIANCE_COMPONENTS for rep in sharing.REPRESENTATIONS
                            for entry in entries]
                np.testing.assert_allclose([bar.get_height() for bar in ax.patches], expected)
                self.assertEqual(sum(text.get_text() == "*" for text in ax.texts), 6)
                self.assertTrue(all(label.get_rotation() == 0 for label in ax.get_xticklabels()))

    def test_both_views_delegate_to_one_plotter(self):
        with patch.object(paper_plot, "_plot_variance_partition") as draw:
            for view in sharing.VARIANCE_VIEWS:
                getattr(paper_plot, f"plot_{view}")()
                draw.assert_called_with(view, f"{view}.png")

    def test_all_task_plot_rejects_controls_with_different_observed_cells(self):
        entry, control = _entry(self.runs, False), _entry(self.control_runs, False)
        control["row_names"] = control["row_names"][:-1]  # same factor levels, different cells
        with patch.object(paper_plot, "_load_row_sharing_view", return_value=(
                self.payload, entry, self.control, control)), \
                patch.object(paper_plot, "_save_fig") as save, patch("builtins.print") as printed:
            paper_plot.plot_variance_partition_all()
        fig = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, fig)
        self.assertEqual(len(fig.axes[0].patches), 9)  # trained only, no mismatched control
        self.assertIn("methods/rows differ", str(printed.call_args_list))
        self.assertFalse(any(text.get_text() == "*" for text in fig.axes[0].texts))

    def test_undefined_networks_and_invalid_cohort_or_fractions(self):
        data = _data()
        for rows in data.values():
            rows["variance_features"][:] = 0.3
        runs = self.runs + [("constant", data)]
        entry = _entry(runs)
        names = [name for name, _ in runs]
        values = paper_plot._variance_partition_samples(entry, names)
        self.assertEqual(values["input"].shape, (4, 3))
        safe = diag._json_safe(entry)
        self.assertIsNone(safe["representations"]["input"]["fractions"]["task"][-1])
        with self.assertRaisesRegex(ValueError, "saved cohort"):
            paper_plot._variance_partition_samples(entry, names[::-1])
        entry["representations"]["input"]["fractions"]["task"][0] = 0.999
        with self.assertRaisesRegex(ValueError, "fractions"):
            paper_plot._variance_partition_samples(entry, names)

    def test_missing_entry_suggests_fast_update(self):
        for view in sharing.VARIANCE_VIEWS:
            with patch.object(paper_plot, "_load_row_cluster_sharing", return_value=self.payload), \
                    patch.object(paper_plot, "_ensure_out_dir"), patch("builtins.print") as printed:
                self.assertIsNone(paper_plot._load_row_sharing_view(view))
            self.assertIn("--variance-partition-only", str(printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
