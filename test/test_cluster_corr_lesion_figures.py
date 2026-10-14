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


def binned_medians(tuning, dissim, n_bins=2):
    """The quantile-bin summary lesion_plot.py saves: median x, median y with
    quartiles and count per bin of the tuning correlation."""
    tuning = np.asarray(tuning, dtype=float)
    dissim = np.asarray(dissim, dtype=float)
    edges = np.unique(np.quantile(tuning, np.linspace(0, 1, n_bins + 1)))
    ids = np.searchsorted(edges[1:-1], tuning, side="right")
    groups = [ids == index for index in np.unique(ids)]
    return {"method": "quantile", "n_bins_requested": n_bins, "edges": edges,
            "x": np.array([np.median(tuning[g]) for g in groups]),
            "y": np.array([np.median(dissim[g]) for g in groups]),
            "y_q25": np.array([np.percentile(dissim[g], 25) for g in groups]),
            "y_q75": np.array([np.percentile(dissim[g], 75) for g in groups]),
            "counts": np.array([np.count_nonzero(g) for g in groups])}


class ClusterCorrLesionFigureTests(unittest.TestCase):
    def entry(self, exclude_last=False, rho=0.5, aname=paper_plot.LESION_ANAME):
        tuning = np.array([-0.2, 0.1, 0.4, 0.8, 0.3, 0.6])
        dissim = 1.0 - rho * tuning
        return {
            "schema_version": 2,
            "aname": aname,
            "x_definition": paper_plot.CLUSTER_CORR_X_DEFINITION,
            "y_definition": paper_plot.CLUSTER_CORR_Y_DEFINITION,
            "exclude_last_cluster": exclude_last,
            "included_clusters": ["c1", "c2", "c3", "c4"],
            "tuning_corr": tuning.tolist(),
            "lesion_profile_dissim": dissim.tolist(),
            "association": association(rho),
            "trend_line": {"method": "ols", "slope": -rho, "intercept": 1.0},
            "binned_medians": binned_medians(tuning, dissim),
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
        # Caches hold 1 - r; the figure shows r, with rho and the bin medians
        # flipped (and their quartiles swapped). No OLS line is drawn.
        points = [c for c in axis.collections
                  if isinstance(c, paper_plot.mpl.collections.PathCollection)]
        np.testing.assert_allclose(
            points[0].get_offsets(),
            np.column_stack((entry["tuning_corr"], 1.0 - np.asarray(entry["lesion_profile_dissim"]))))
        medians = entry["binned_medians"]
        bars = [c for c in axis.containers
                if isinstance(c, paper_plot.mpl.container.ErrorbarContainer)]
        self.assertEqual(len(bars), 1)
        self.assertEqual(len(axis.lines), 1)
        line = axis.lines[0]
        np.testing.assert_allclose(line.get_xdata(), medians["x"])
        np.testing.assert_allclose(line.get_ydata(), 1.0 - medians["y"])
        segments = bars[0].lines[2][0].get_segments()
        np.testing.assert_allclose([segment[0, 1] for segment in segments], 1.0 - medians["y_q75"])
        np.testing.assert_allclose([segment[1, 1] for segment in segments], 1.0 - medians["y_q25"])
        self.assertEqual(line.get_color(), paper_plot._TREND_LINE_COLOR)
        legend_box = axis.get_legend()
        legend = legend_box.get_title().get_text()
        self.assertIn("permutation p = 0.012", legend)
        self.assertIn(f"= {-entry['association']['rho']:.2f}", legend)
        self.assertIn("(n = 4)", legend)
        self.assertEqual([text.get_text() for text in legend_box.get_texts()], ["Bin median, IQR"])
        self.assertEqual(axis.get_xlabel(), f"{tuning_word} correlation")
        self.assertIn("Lesion similarity", axis.get_ylabel())
        self.assertEqual(axis.get_box_aspect(), 1)            # square axes box, shared style
        # Round tick steps and short labels: at most two decimals, no trailing zeros.
        figure.canvas.draw()
        for tick_axis in (axis.xaxis, axis.yaxis):
            labels = [label.get_text() for label in tick_axis.get_ticklabels()]
            self.assertTrue(all(len(label.split(".")[-1]) <= 2 if "." in label else True
                                for label in labels), labels)
            self.assertFalse(any(label.endswith("0") and "." in label for label in labels), labels)
        self.assertEqual([paper_plot._short_tick(v) for v in (0.4, 0.80, 1.0, -0.5, 0.0, 0.75)],
                         ["0.4", "0.8", "1", "-0.5", "0", "0.75"])
        self.assertLessEqual(len(axis.get_yticks()), paper_plot._STAT_SCATTER_MAX_TICKS + 1)
        np.testing.assert_array_equal(points[0].get_sizes(),
                                      [paper_plot._SCATTER_POINT_STYLE["s"]])
        np.testing.assert_allclose(figure.get_size_inches(), paper_plot._LESION_SCATTER_FIGSIZE)

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

    def test_cache_without_binned_medians_still_renders_without_a_connector(self):
        entry = self.entry()
        entry.pop("binned_medians")
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(directory)
            self.write(root, "mod_leison_effect_normalized_zero-W", {"normalized_zero-W": entry})
            with patch("builtins.print") as output:
                figures = self.render(root)
        self.assertEqual(len(figures), 1)
        axis = next(iter(figures.values())).axes[0]
        self.assertEqual(len(axis.lines), 0)
        self.assertEqual(axis.containers, [])
        self.assertIsNotNone(axis.get_legend())
        self.assertTrue(any("no saved binned medians" in str(call.args[0])
                            for call in output.call_args_list if call.args))

    def test_invalid_binned_medians_are_rejected(self):
        for problem in ("nonfinite", "quartile_order", "length", "zero_count"):
            entry = self.entry()
            medians = entry["binned_medians"]
            if problem == "nonfinite":
                medians["y"][0] = np.nan
            elif problem == "quartile_order":
                medians["y_q25"][0] = medians["y"][0] + 1.0
            elif problem == "length":
                medians["x"] = medians["x"][:-1]
            else:
                medians["counts"][0] = 0
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
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

    def test_seeds_figure_pools_every_sibling_run_with_both_sides(self):
        """One hidden/input pair per run, similarity sign, filled when p < 0.05."""
        import re
        runs = {}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for seed, (rho_hidden, p_hidden, rho_input, p_input) in {
                    11: (-0.4, 0.010, 0.1, 0.600), 22: (-0.2, 0.080, 0.2, 0.300),
                    33: (-0.5, 0.001, -0.1, 0.700)}.items():
                aname = re.sub(r"seed\d+", f"seed{seed}", paper_plot.LESION_ANAME)
                hidden = self.entry(True, rho=rho_hidden, aname=aname)
                hidden["association"]["p_perm"] = p_hidden
                inputs = self.entry(True, rho=rho_input, aname=aname)
                inputs["association"]["p_perm"] = p_input
                (base / aname).mkdir()
                with (base / aname / f"cluster_corr_vs_normalized_lesion_effect_unnorm_{aname}.pkl"
                      ).open("wb") as stream:
                    pickle.dump({"hidden_unnormalized_k20": hidden,
                                 "input_unnormalized_k20": inputs}, stream)
                runs[aname] = (rho_hidden, p_hidden, rho_input, p_input)
            # A run lacking the input entry is skipped entirely, never half-drawn.
            partial = re.sub(r"seed\d+", "seed44", paper_plot.LESION_ANAME)
            (base / partial).mkdir()
            with (base / partial / f"cluster_corr_vs_normalized_lesion_effect_unnorm_{partial}.pkl"
                  ).open("wb") as stream:
                pickle.dump({"hidden_unnormalized_k20": self.entry(True, aname=partial)}, stream)
            with patch.object(paper_plot, "LESION_NORM_DIR", base / paper_plot.LESION_ANAME), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as output:
                paper_plot.plot_cluster_corr_vs_lesion_seeds()
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_cluster_corr_vs_lesion_seeds.png")
        self.assertIn("hidden: 3/3 rho > 0, 2 significant", save.call_args.kwargs["extra"])
        self.assertIn("input: 1/3 rho > 0, 0 significant", save.call_args.kwargs["extra"])
        from scipy.stats import wilcoxon
        expected_p = {side: float(wilcoxon([-values[index] for values in runs.values()]).pvalue)
                      for side, index in (("hidden", 0), ("input", 2))}
        self.assertIn(f"Wilcoxon p = {expected_p['hidden']:.3f}", save.call_args.kwargs["extra"])
        self.assertEqual([label.get_text() for label in figure.axes[0].get_xticklabels()],
                         ["Hidden", "Input"])
        self.assertEqual(len(figure.axes[0].texts), 0)
        self.assertEqual(figure.axes[0].get_legend().get_title().get_text(),
                         "3 runs; Wilcoxon signed-rank vs 0:\n"
                         f"hidden p = {expected_p['hidden']:.3f}, input p = {expected_p['input']:.3f}")
        self.assertTrue(any("seed44" in str(call.args[0]) for call in output.call_args_list if call.args))
        axis = figure.axes[0]
        # Scatter points only: the median dashes are LineCollections in `collections` too.
        dots = [c for c in axis.collections
                if isinstance(c, paper_plot.mpl.collections.PathCollection)]
        points = sorted((round(float(c.get_offsets()[0, 0])), float(c.get_offsets()[0, 1]),
                         bool(np.allclose(c.get_facecolor()[0][:3], (1, 1, 1))))
                        for c in dots)
        expected = sorted([(0, -rh, not (ph < 0.05)) for rh, ph, _, _ in runs.values()]
                          + [(1, -ri, not (pi < 0.05)) for _, _, ri, pi in runs.values()])
        for (x, y, hollow), (ex, ey, ehollow) in zip(points, expected):
            self.assertEqual(x, ex)
            self.assertAlmostEqual(y, ey)
            self.assertEqual(hollow, ehollow)
        np.testing.assert_allclose(figure.get_size_inches(), paper_plot._LESION_SCATTER_FIGSIZE)

    def test_seeds_figure_needs_two_runs(self):
        import re
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            aname = re.sub(r"seed\d+", "seed11", paper_plot.LESION_ANAME)
            (base / aname).mkdir()
            with (base / aname / f"cluster_corr_vs_normalized_lesion_effect_unnorm_{aname}.pkl"
                  ).open("wb") as stream:
                pickle.dump({"hidden_unnormalized_k20": self.entry(True, aname=aname),
                             "input_unnormalized_k20": self.entry(True, aname=aname)}, stream)
            with patch.object(paper_plot, "LESION_NORM_DIR", base / paper_plot.LESION_ANAME), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as output:
                paper_plot.plot_cluster_corr_vs_lesion_seeds()
        save.assert_not_called()
        self.assertIn("need >= 2", output.call_args.args[0])

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
