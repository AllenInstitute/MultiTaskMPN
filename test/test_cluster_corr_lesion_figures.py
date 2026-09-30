"""Cluster/lesion paper figures reuse exact saved scale-free comparisons."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot


def association(rho, p_perm=0.012, n_clusters=4):
    return {"statistic": "spearman", "rho": rho, "p_perm": p_perm,
            "n_clusters": n_clusters, "permutation_unit": "cluster_label",
            "side": "two-sided", "n_perm": 10}


class ClusterCorrLesionFigureTests(unittest.TestCase):
    def entry(self, exclude_last=False, rho=0.5, aname=paper_plot.LESION_ANAME):
        tuning = np.array([-0.2, 0.1, 0.4, 0.8, 0.3, 0.6])
        return {
            "schema_version": 2,
            "aname": aname,
            "x_definition": paper_plot.CLUSTER_CORR_X_DEFINITION,
            "y_definition": paper_plot.CLUSTER_CORR_Y_DEFINITION,
            "exclude_last_cluster": exclude_last,
            "included_clusters": ["c1", "c2", "c3", "c4"],
            "tuning_corr": tuning.tolist(),
            "lesion_profile_dissim": (1.0 - rho * tuning).tolist(),
            "association": association(rho),
            "trend_line": {"method": "ols", "slope": -rho, "intercept": 1.0},
            "l1": {"tuning_cos_sim": (0.5 + 0.5 * tuning).tolist(),
                   "lesion_l1_dist": (2.0 + tuning).tolist()},
        }

    def write(self, root, suffix, entries):
        path = root / f"cluster_corr_vs_{suffix}_{paper_plot.LESION_ANAME}.pkl"
        with path.open("wb") as stream:
            pickle.dump(entries, stream)

    def root(self, directory):
        return Path(directory)

    def render(self, root, show_legend=True, weighted_only=False):
        with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "SHOW_LEGEND", show_legend), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("scipy.stats.linregress", side_effect=AssertionError("No refitting")), \
                patch("scipy.stats.spearmanr", side_effect=AssertionError("No refitting")):
            if weighted_only:
                paper_plot.plot_cluster_corr_vs_lesion_weighted()
            else:
                paper_plot.plot_cluster_corr_vs_lesion()
        figures = {call.args[1].name: call.args[0] for call in save.call_args_list}
        self.assertEqual(len(figures), save.call_count)
        for figure in figures.values():
            self.addCleanup(paper_plot.plt.close, figure)
        return figures

    def assert_main_figure(self, figure, entry, tuning_word="Tuning"):
        self.assertEqual(len(figure.axes), 1)
        axis = figure.axes[0]
        np.testing.assert_allclose(
            axis.collections[0].get_offsets(),
            np.column_stack((entry["tuning_corr"], entry["lesion_profile_dissim"])))
        self.assertEqual(len(axis.lines), 1)
        line = axis.lines[0]
        np.testing.assert_allclose(line.get_ydata(),
                                   1.0 - entry["association"]["rho"] * line.get_xdata())
        self.assertAlmostEqual(line.get_xdata().min(), min(entry["tuning_corr"]))
        self.assertAlmostEqual(line.get_xdata().max(), max(entry["tuning_corr"]))
        legend = axis.get_legend().get_texts()[0].get_text()
        self.assertIn("permutation p = 0.012", legend)
        self.assertIn(f"= {entry['association']['rho']:.2f}", legend)
        self.assertIn("(n = 4)", legend)
        self.assertEqual(axis.get_xlabel(), f"{tuning_word} profile correlation")
        self.assertIn("Lesion profile dissimilarity", axis.get_ylabel())
        self.assertEqual(axis.get_ylim()[0], 0)
        np.testing.assert_allclose(np.diff(axis.get_yticks()), 0.5)
        np.testing.assert_allclose(figure.get_size_inches(), [3.0, 2.8])

    def test_entry_point_draws_one_figure_per_entry_and_no_supplement(self):
        normalized = self.entry()
        unnormalized = self.entry(exclude_last=True, rho=-0.4)
        abs_weighted = self.entry(exclude_last=True, rho=-0.3)
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(directory)
            for variant, entry in (("normalized", normalized), ("unnormalized", unnormalized)):
                suffix = "normalized_lesion_effect" + ("_unnorm" if variant == "unnormalized" else "")
                self.write(root, suffix, {f"{layer}_{variant}_k20": entry
                                          for layer in ("input", "hidden")})
            self.write(root, "mod_lesion_effect_normalized_zero-W",
                       {"normalized_zero-W": normalized})
            self.write(root, "mod_lesion_effect_var-weighted-unnormalized_zero-W",
                       {"var-weighted-unnormalized_zero-W": unnormalized})
            self.write(root, "mod_lesion_effect_abs-weighted-unnormalized_zero-W",
                       {"abs-weighted-unnormalized_zero-W": abs_weighted})
            self.write(root, "mod_lesion_effect_unnormalized_zero-W",
                       {"unnormalized_zero-W": self.entry(True, rho=0.99)})
            self.write(root, "mod_lesion_effect_var-weighted-unnormalized_freeze-M",
                       {"var-weighted-unnormalized_freeze-M": self.entry(True, rho=0.99)})
            figures = self.render(root)

        expected = {
            "multitask_cluster_corr_vs_lesion_input_norm.png": normalized,
            "multitask_cluster_corr_vs_lesion_hidden_norm.png": normalized,
            "multitask_cluster_corr_vs_lesion_input_unnorm.png": unnormalized,
            "multitask_cluster_corr_vs_lesion_hidden_unnorm.png": unnormalized,
            "multitask_cluster_corr_vs_lesion_modulation_norm_zero_W.png": normalized,
            "multitask_cluster_corr_vs_lesion_modulation_var_weighted_unnorm_zero_W.png": unnormalized,
            "multitask_cluster_corr_vs_lesion_modulation_abs_weighted_unnorm_zero_W.png": abs_weighted,
        }
        self.assertEqual(set(figures), set(expected))
        for name, entry in expected.items():
            self.assert_main_figure(figures[name], entry)

    def test_legacy_cache_filenames_remain_readable_without_renaming(self):
        variants = (
            ("normalized_lesion_effect", "hidden_normalized_k20", False, False),
            ("normalized_lesion_effect_unnorm", "hidden_unnormalized_k20", True, False),
            ("mod_lesion_effect_normalized_zero-W", "normalized_zero-W", False, False),
            ("mod_lesion_effect_var-weighted-unnormalized_zero-W",
             "var-weighted-unnormalized_zero-W", True, False),
            ("mod_lesion_effect_weighted-unnormalized_zero-W",
             "weighted-unnormalized_zero-W", True, True),
        )
        for suffix, key, exclude_last, weighted_only in variants:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                entry = self.entry(exclude_last)
                self.write(root, suffix.replace("lesion", "leison"), {key: entry})
                legacy = next(root.glob("*.pkl"))
                before = legacy.read_bytes()
                figures = self.render(root, weighted_only=weighted_only)
                self.assertEqual(len(figures), 1)
                figure = next(iter(figures.values()))
                self.assert_main_figure(figure, entry, tuning_word=(
                    r"$\mathrm{Var}(WM)$ tuning" if weighted_only else "Tuning"))
                self.assertEqual(list(root.glob("*.pkl")), [legacy])
                self.assertEqual(legacy.read_bytes(), before)

    def test_entries_from_another_run_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(directory)
            self.write(root, "normalized_lesion_effect_unnorm",
                       {"input_unnormalized_k20": self.entry(True),
                        "hidden_unnormalized_k20": self.entry(True, aname="another_run")})
            figures = self.render(root)
        self.assertEqual(set(figures), {"multitask_cluster_corr_vs_lesion_input_unnorm.png"})

    def test_cache_without_trend_line_still_renders_without_a_line(self):
        entry = self.entry()
        entry.pop("trend_line")
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(directory)
            self.write(root, "mod_leison_effect_normalized_zero-W", {"normalized_zero-W": entry})
            figures = self.render(root)
        self.assertEqual(len(figures), 1)
        axis = next(iter(figures.values())).axes[0]
        self.assertEqual(len(axis.lines), 0)
        self.assertIsNotNone(axis.get_legend())

    def test_non_finite_trend_line_is_rejected(self):
        entry = self.entry()
        entry["trend_line"]["slope"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(directory)
            self.write(root, "mod_leison_effect_normalized_zero-W", {"normalized_zero-W": entry})
            self.assertEqual(self.render(root), {})

    def test_missing_modulation_variant_does_not_fall_back_or_block_others(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(directory)
            self.write(root, "mod_lesion_effect_normalized_zero-W",
                       {"normalized_zero-W": self.entry()})
            for tag in ("unnormalized_zero-W", "var-weighted-unnormalized_freeze-M"):
                self.write(root, f"mod_lesion_effect_{tag}", {tag: self.entry(True)})
            figures = self.render(root)
        self.assertEqual(set(figures),
                         {"multitask_cluster_corr_vs_lesion_modulation_norm_zero_W.png"})

    def test_incompatible_entries_are_skipped_without_refitting(self):
        problems = ("legacy_schema", "wrong_key", "wrong_mode", "wrong_x", "wrong_y",
                    "wrong_run", "wrong_exclusion", "missing_exclusion",
                    "missing_association", "pearson_association", "nonfinite")
        for problem in problems:
            entry = self.entry(True)
            name = "var-weighted-unnormalized_zero-W"
            if problem == "legacy_schema":
                entry = {"aname": paper_plot.LESION_ANAME, "exclude_last_cluster": True,
                         "y_definition": "sum_over_tasks_abs_effect_difference",
                         "tuning_cos_sim": [0.1, 0.4, 0.8], "lesion_l1_dist": [3., 4., 5.],
                         "regression": {"slope": 2., "intercept": 3., "r": 1., "p": 0.}}
            elif problem == "wrong_key":
                name = "var-weighted-unnormalized_freeze-M"
            elif problem == "wrong_mode":
                entry["mod_lesion_mode"] = "freeze_M"
            elif problem == "wrong_x":
                entry["x_definition"] = "cosine"
            elif problem == "wrong_y":
                entry["y_definition"] = "sum_over_tasks_abs_effect_difference"
            elif problem == "wrong_run":
                entry["aname"] = "another_run"
            elif problem == "wrong_exclusion":
                entry["exclude_last_cluster"] = False
            elif problem == "missing_exclusion":
                entry.pop("exclude_last_cluster")
            elif problem == "missing_association":
                entry.pop("association")
            elif problem == "pearson_association":
                entry["association"]["statistic"] = "pearson"
            else:
                entry["lesion_profile_dissim"][0] = np.nan
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = self.root(directory)
                self.write(root, "mod_lesion_effect_var-weighted-unnormalized_zero-W", {name: entry})
                self.assertEqual(self.render(root), {})

    def test_weighted_uses_only_var_wm_cache(self):
        entry = self.entry(True, rho=-0.3)
        tag = "weighted-unnormalized_zero-W"
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(directory)
            for distractor in ("var-weighted-unnormalized_zero-W",
                               "weighted-unnormalized_freeze-M", "unnormalized_zero-W"):
                self.write(root, f"mod_lesion_effect_{distractor}",
                           {distractor: self.entry(True, rho=0.99)})
            self.assertEqual(self.render(root, weighted_only=True), {})
            self.write(root, f"mod_lesion_effect_{tag}", {tag: entry})
            figures = self.render(root, weighted_only=True)
        stem = "multitask_cluster_corr_vs_lesion_modulation_weighted_unnorm_zero_W"
        self.assertEqual(set(figures), {f"{stem}.png"})
        self.assert_main_figure(figures[f"{stem}.png"], entry,
                                tuning_word=r"$\mathrm{Var}(WM)$ tuning")

    def test_weighted_rejects_wrong_identity_and_exclusion(self):
        tag = "weighted-unnormalized_zero-W"
        for problem in ("key", "mode", "exclusion"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = self.root(directory)
                entry = self.entry(True)
                key = tag
                if problem == "key":
                    key = "var-weighted-unnormalized_zero-W"
                elif problem == "mode":
                    entry["mod_lesion_mode"] = "freeze_M"
                else:
                    entry["exclude_last_cluster"] = False
                self.write(root, f"mod_lesion_effect_{tag}", {key: entry})
                self.assertEqual(self.render(root, weighted_only=True), {})

    def test_no_legend_keeps_figure_without_legend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(directory)
            self.write(root, "mod_lesion_effect_normalized_zero-W",
                       {"normalized_zero-W": self.entry()})
            figures = self.render(root, show_legend=False)
        self.assertEqual(len(figures), 1)
        axis = next(iter(figures.values())).axes[0]
        self.assertIsNone(axis.get_legend())
        self.assertEqual(len(axis.lines), 1)


if __name__ == "__main__":
    unittest.main()
