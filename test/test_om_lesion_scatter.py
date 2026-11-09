"""OM paper plots consume saved rank statistics and medians without fitting."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot
import lesion_plot

MODE = paper_plot.OM_LESION_MODE
MODE_TAG = MODE.replace("_", "-")


def scatter_run(root, highlight=None):
    """Point the OM-vs-lesion scatter at the fixture's run directory, with the
    example-block highlight given (none by default: no cluster cache here)."""
    return patch.multiple(paper_plot,
                          _om_scatter_run=MagicMock(return_value=(root, paper_plot.LESION_ANAME)),
                          _om_scatter_highlight=MagicMock(return_value=highlight))


class OmLesionScatterTests(unittest.TestCase):
    def test_scatter_reads_the_lesion_heatmap_mode(self):
        """freeze_M, like every synapse-lesion figure except plasticity share
        (the scatter read zero_W until 2026-11-08)."""
        self.assertEqual(MODE, "freeze_M")
        self.assertEqual(MODE, paper_plot.LESION_HEATMAP_MODE)

    def test_scatter_reads_the_clustering_example_run_not_the_lesion_example(self):
        """The deliberate exception: ANAME (the OM example panels' run), not the
        LESION_ANAME every other lesion figure reads; the saved message says so."""
        cache = self.cache(False)
        cache["aname"] = "clustering_run"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "clustering_run"
            run_dir.mkdir()
            path = run_dir / f"om_vs_lesion_diff_var-weighted-unnormalized_{MODE_TAG}_unnorm_clustering_run.pkl"
            with path.open("wb") as stream:
                pickle.dump(cache, stream)
            with patch.object(paper_plot, "LESION_NORM_ROOT", Path(directory)), \
                    patch.object(paper_plot, "ANAME", "clustering_run"), \
                    patch.object(paper_plot, "LESION_ANAME", "lesion_run"), \
                    patch.object(paper_plot, "_om_example_selection", return_value=None), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                self.assertEqual(paper_plot._om_scatter_run(), (run_dir, "clustering_run"))
                paper_plot.plot_om_vs_lesion()
        save.assert_called_once()
        self.addCleanup(paper_plot.plt.close, save.call_args.args[0])
        self.assertIn("not the lesion example lesion_run", save.call_args.kwargs["extra"])

    def selection(self, ratio=2.5):
        """A one-example OM selection whose deciding block (input C2, hidden
        C1 of cluster C7) has cached observed/expected `ratio` and whose
        upper OM color limit is 3."""
        ratios = np.zeros((1, 2, 2))
        ratios[0, 1, 0] = ratio
        return {"ratios": ratios, "cluster_ids": [7], "chosen": [0], "tops": [(1, 0)], "vmax": 3.0}

    def test_highlight_matches_identity_despite_repeated_ratios_and_point_reordering(self):
        identities = np.array([[7, 2, 2], [7, 2, 1], [8, 2, 1]])
        values = np.array([2.5, 2.5, 1.0])
        with patch.object(paper_plot, "ANAME", "run"), \
                patch.object(paper_plot, "_om_example_selection", return_value=self.selection()):
            found = paper_plot._om_scatter_highlight(values, "run", identities)
            self.assertEqual(found, {"point": 1, "om": 1.5, "cluster_id": 7,
                                     "input": 2, "hidden": 1, "vmax": 3.0})
            order = [1, 2, 0]
            reordered = paper_plot._om_scatter_highlight(values[order], "run", identities[order])
            self.assertEqual(reordered, dict(found, point=0))

    def test_highlight_skips_missing_ambiguous_or_stale_identity_without_ratio_fallback(self):
        with patch.object(paper_plot, "ANAME", "run"), \
                patch.object(paper_plot, "_om_example_selection", return_value=self.selection()), \
                patch("builtins.print"):
            for values, run, identities in (
                ([2.5], "run", None),                         # legacy cache
                ([2.5], "other_run", [[7, 2, 1]]),            # different network
                ([2.5], "run", [[8, 2, 1]]),                 # same ratio, wrong identity
                ([2.5, 1.0], "run", [[7, 2, 1], [7, 2, 1]]), # duplicate identity
                ([2.5, 1.0], "run", [[8, 2, 1], [7, 2, 1]]), # changed ratio at the right ID
            ):
                with self.subTest(values=values, run=run, identities=identities):
                    self.assertIsNone(paper_plot._om_scatter_highlight(
                        np.asarray(values), run, identities))

    def test_highlight_is_drawn_on_top_in_the_examples_color(self):
        cache = self.cache(False)
        highlight = {"point": 0, "om": float(cache["om_vals"][0]) - 1.0,
                     "cluster_id": 7, "input": 2, "hidden": 1, "vmax": 3.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, cache, False)
            with scatter_run(root, highlight), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_om_vs_lesion()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        axis = figure.axes[0]
        ringed = [c for c in axis.collections if isinstance(c, paper_plot.mpl.collections.PathCollection)
                  and len(c.get_edgecolors()) and np.allclose(c.get_edgecolors()[0][:3], 0.)]
        self.assertEqual(len(ringed), 1)
        np.testing.assert_allclose(ringed[0].get_offsets(),
                                   [[cache["om_vals"][0] - 1.0, cache["lesion_diffs"][0]]])
        # The block maps' diverging scale (vmax 3): this block's OM -1 is its blue end.
        cmap, norm = paper_plot._om_block_cmap_norm(highlight["vmax"])
        np.testing.assert_allclose(ringed[0].get_facecolors()[0], cmap(norm(highlight["om"])))
        self.assertEqual(norm(highlight["om"]), 0.0)
        self.assertGreater(ringed[0].get_zorder(), 3)
        self.assertIn("Example block", [text.get_text() for text in axis.get_legend().get_texts()])
        self.assertIn("highlighted: example 1 block, synapse cluster C7 at input C2 / hidden C1",
                      save.call_args.kwargs["extra"])

    def test_single_cache_of_the_other_mode_is_rejected(self):
        cache = self.cache(False)
        cache["mod_lesion_mode"] = "zero_W" if MODE == "freeze_M" else "freeze_M"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f"om_vs_lesion_diff_var-weighted-unnormalized_{MODE_TAG}_unnorm_{paper_plot.LESION_ANAME}.pkl"
            with path.open("wb") as stream:
                pickle.dump(cache, stream)
            with patch.object(paper_plot, "LESION_NORM_DIR", root):
                self.assertIsNone(paper_plot._load_om_lesion_scatter())

    def summary(self):
        return lesion_plot._om_scatter_summary(
            [[0., 0.2, 0.4], [0.8, 0.1, 0.3]], [[0., 2., 4.], [1., 3., 5.]],
            [[[1., 0.3, 0.1], [0.3, 0.2, 0.6], [0., 0.4, 0.1]],
             [[0., 0.7, 0.2], [0.5, 0.1, 0.3], [0.8, 0.3, 0.2]]], n_perm=13, seed=4)

    def cache(self, combined):
        entry = self.summary()
        entry["point_ids"] = np.array([[3, 1, 1], [3, 1, 2], [3, 2, 1],
                                       [7, 1, 1], [7, 1, 2], [7, 2, 1]])
        result = {
            "variance_method": paper_plot.VARIANCE_METHOD,
            "schema_version": 2, "aname": paper_plot.LESION_ANAME,
            "variant": "unnorm", "min_expected": 3.,
            "y_definition": "task-profile L1/T: mean_t |mod_effect(t) - combined_effect(t)|",
        }
        if combined:
            other = self.summary()
            other["association"]["rho"] = -0.99
            result.update(base_key="modulation_all_var_weighted_unnormalized",
                          mode_data={MODE: entry, "freeze_M" if MODE == "zero_W" else "zero_W": other})
        else:
            result.update(entry)
            result.update(mod_type_key="modulation_all_var_weighted_unnormalized",
                          mod_lesion_mode=MODE)
        return result

    def write(self, root, cache, combined):
        mode = "combined" if combined else MODE_TAG
        path = root / f"om_vs_lesion_diff_var-weighted-unnormalized_{mode}_unnorm_{paper_plot.LESION_ANAME}.pkl"
        with path.open("wb") as stream:
            pickle.dump(cache, stream)

    def test_cache_preserves_point_id_order_and_accepts_legacy_statistics_without_ids(self):
        for combined in (False, True):
            cache = self.cache(combined)
            entry = cache["mode_data"][MODE] if combined else cache
            with self.subTest(combined=combined), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write(root, cache, combined)
                loaded = paper_plot._load_om_lesion_scatter(root)
                np.testing.assert_array_equal(loaded["point_ids"], entry["point_ids"])
                entry.pop("point_ids")
                self.write(root, cache, combined)
                legacy = paper_plot._load_om_lesion_scatter(root)
                self.assertIsNone(legacy["point_ids"])
                np.testing.assert_array_equal(loaded["om_vals"], legacy["om_vals"])
                self.assertEqual(loaded["rho"], legacy["rho"])

    def test_malformed_or_duplicate_point_ids_are_rejected(self):
        for problem in ("shape", "float", "nonpositive", "duplicate"):
            cache = self.cache(True)
            entry = cache["mode_data"][MODE]
            if problem == "shape":
                entry["point_ids"] = entry["point_ids"][:-1]
            elif problem == "float":
                entry["point_ids"] = entry["point_ids"].astype(float)
            elif problem == "nonpositive":
                entry["point_ids"][0, 0] = 0
            else:
                entry["point_ids"][1] = entry["point_ids"][0]
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write(root, cache, True)
                self.assertIsNone(paper_plot._load_om_lesion_scatter(root))

    def test_both_cache_formats_render_saved_medians_and_rank_without_refitting(self):
        for combined in (False, True):
            cache = self.cache(combined)
            entry = cache["mode_data"][MODE] if combined else cache
            with self.subTest(combined=combined), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write(root, cache, combined)
                with scatter_run(root), \
                        patch.object(paper_plot, "_ensure_out_dir"), \
                        patch.object(paper_plot, "_save_fig") as save, \
                        patch.object(lesion_plot, "_om_scatter_summary",
                                     side_effect=AssertionError("Must use saved statistics")):
                    paper_plot.plot_om_vs_lesion()
                save.assert_called_once()
                figure, path = save.call_args.args[:2]
                self.addCleanup(paper_plot.plt.close, figure)
                self.assertEqual(path.name, "multitask_om_vs_lesion_scatter.png")
                self.assertEqual(len(figure.axes), 2)
                axis = figure.axes[0]
                self.assertEqual(axis.get_ylim()[0], 0)
                self.assertEqual(len(axis.lines), 1)
                np.testing.assert_array_equal(axis.lines[0].get_xdata(), entry["binned_medians"]["x"] - 1.0)
                np.testing.assert_array_equal(axis.lines[0].get_ydata(), entry["binned_medians"]["y"])
                np.testing.assert_allclose(axis.collections[0].get_offsets(),
                                           np.column_stack((entry["om_vals"] - 1.0, entry["lesion_diffs"])))
                self.assertIn("observed / expected − 1", axis.get_xlabel())
                scatter = axis.collections[0]
                self.assertEqual(scatter.get_alpha(), paper_plot._DENSE_SCATTER_ALPHA)
                self.assertEqual(scatter.get_zorder(), 3)
                np.testing.assert_array_equal(scatter.get_sizes(), [paper_plot._SCATTER_POINT_STYLE["s"]])
                np.testing.assert_array_equal(scatter.get_linewidths(),
                                              [paper_plot._SCATTER_POINT_STYLE["linewidths"]])
                np.testing.assert_allclose(scatter.get_facecolors(),
                                           [paper_plot.mpl.colors.to_rgba(
                                               paper_plot._LESION_POINT_COLOR,
                                               paper_plot._DENSE_SCATTER_ALPHA)])
                # No outline around the dots.
                self.assertEqual(len(scatter.get_edgecolors()), 0)
                legend = axis.get_legend()
                self.assertIn("Spearman", legend.get_title().get_text())
                self.assertIn(f"p_perm = {entry['association']['p_perm']:.3f}", legend.get_title().get_text())
                self.assertEqual([text.get_text() for text in legend.get_texts()], ["Binned median"])
                corr = entry["profile_correlation"]
                right = figure.axes[1]
                self.assertIn("Pearson r", right.get_ylabel())
                self.assertLess(right.get_ylim()[0], -1)
                self.assertGreater(right.get_ylim()[1], 1)
                np.testing.assert_allclose(right.collections[0].get_offsets(),
                                           np.column_stack((entry["om_vals"] - 1., corr["values"])))
                np.testing.assert_allclose(right.lines[0].get_ydata(), corr["binned_medians"]["y"])
                self.assertIn(f"p_perm = {corr['association']['p_perm']:.3f}",
                              right.get_legend().get_title().get_text())
                self.assertIn("one-sided >", right.get_legend().get_title().get_text())

    def test_pearson_entry_is_required_only_for_the_two_panel_figure(self):
        cache = self.cache(False)
        cache.pop("profile_correlation")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, cache, False)
            self.assertIsNotNone(paper_plot._load_om_lesion_scatter(root))
            with scatter_run(root), patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_om_vs_lesion()
            save.assert_not_called()

    def test_wrong_pearson_tail_and_out_of_range_values_are_rejected(self):
        for problem in ("tail", "range", "excluded"):
            cache = self.cache(False)
            corr = cache["profile_correlation"]
            if problem == "tail":
                corr["association"]["side"] = "less"
            elif problem == "range":
                corr["values"][0] = 1.1
            else:
                corr["n_excluded"] = 1
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write(root, cache, False)
                self.assertIsNone(paper_plot._load_om_lesion_scatter(root))

    def test_undefined_pearson_profiles_are_not_drawn_at_zero(self):
        cache = self.cache(False)
        cache.update(lesion_plot._om_scatter_summary(
            [[0.], [0.8]], [[0., 2., 4.], [1., 3., 5.]],
            [[[1.], [0.3], [0.]], [[0.], [0.5], [0.8]]], n_perm=13))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, cache, False)
            with scatter_run(root), patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_om_vs_lesion()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(len(figure.axes[0].collections[0].get_offsets()), 6)
        self.assertEqual(len(figure.axes[1].collections[0].get_offsets()), 0)
        self.assertIn("Pearson excluded 6", save.call_args.kwargs["extra"])

    def test_legacy_mismatched_or_incomplete_statistics_are_rejected(self):
        for problem in ("legacy", "pearson", "wrong_tail", "point_permutation", "no_bins", "negative_median"):
            cache = self.cache(True)
            entry = cache["mode_data"][MODE]
            if problem == "legacy":
                cache.pop("schema_version")
            elif problem == "pearson":
                entry["association"]["statistic"] = "pearson"
            elif problem == "wrong_tail":
                entry["association"]["side"] = "greater"
            elif problem == "point_permutation":
                entry["association"]["permutation_unit"] = "point"
            elif problem == "no_bins":
                entry.pop("binned_medians")
            else:
                entry["binned_medians"]["y"][0] = -0.1
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write(root, cache, True)
                with scatter_run(root), \
                        patch.object(paper_plot, "_ensure_out_dir"), \
                        patch.object(paper_plot, "_save_fig") as save:
                    paper_plot.plot_om_vs_lesion()
                save.assert_not_called()

    def test_new_single_cache_is_used_when_combined_cache_is_legacy(self):
        legacy = self.cache(True)
        legacy.pop("schema_version")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, legacy, True)
            self.write(root, self.cache(False), False)
            with patch.object(paper_plot, "LESION_NORM_DIR", root):
                loaded = paper_plot._load_om_lesion_scatter()
            self.assertIn(MODE_TAG, loaded["path"].name)
            self.assertEqual(loaded["mode"], MODE)

    def test_undefined_rank_still_displays_shifted_om_without_legend(self):
        cache = self.cache(False)
        cache["association"]["rho"] = np.nan
        cache["association"]["p_perm"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, cache, False)
            with scatter_run(root), \
                    patch.object(paper_plot, "SHOW_LEGEND", False), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_om_vs_lesion()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertIsNone(figure.axes[0].get_legend())
        self.assertEqual(figure.axes[0].get_ylim()[0], 0)

class OmLesionSeedsTests(unittest.TestCase):
    """The paired cohort view reads both saved associations from the same runs."""

    scatter = OmLesionScatterTests()

    def run_dir(self, root, seed, rho=None, p_perm=None, legacy=False,
                correlation=None, missing_correlation=False):
        """A sibling run directory holding a combined OM cache for that run."""
        aname = f"everything_seed{seed}_L21e4+hidden300+batch128+angle"
        directory = root / aname
        directory.mkdir()
        cache = self.scatter.cache(True)
        cache["aname"] = aname
        association = cache["mode_data"][MODE]["association"]
        if rho is not None:
            association.update(rho=rho, p_perm=p_perm)
        if correlation is not None:
            cache["mode_data"][MODE]["profile_correlation"]["association"].update(
                rho=correlation[0], p_perm=correlation[1])
        if missing_correlation:
            cache["mode_data"][MODE].pop("profile_correlation")
        if legacy:
            cache.pop("schema_version")
        path = directory / f"om_vs_lesion_diff_var-weighted-unnormalized_combined_unnorm_{aname}.pkl"
        with path.open("wb") as stream:
            pickle.dump(cache, stream)
        return directory

    def draw(self, directories):
        with patch.object(paper_plot, "_sibling_run_dirs", return_value=directories), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch("builtins.print") as output, \
                patch.object(lesion_plot, "_om_scatter_summary",
                             side_effect=AssertionError("Must use saved statistics")):
            paper_plot.plot_om_vs_lesion_seeds()
        return save, output

    def test_one_point_per_run_with_a_one_sided_cohort_test(self):
        from scipy.stats import wilcoxon

        rhos, p_values = [-0.6, -0.3, -0.1], [0.01, 0.2, 0.4]
        corr_rhos, corr_p_values = [0.15, 0.3, -0.05], [0.02, 0.1, 0.6]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = [self.run_dir(root, seed, rho, p,
                                       correlation=(corr_rhos[seed], corr_p_values[seed]))
                           for seed, (rho, p) in enumerate(zip(rhos, p_values))]
            # A legacy cache is left out rather than drawn.
            directories.append(self.run_dir(root, 9, legacy=True))
            save, output = self.draw(directories)
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_om_vs_lesion_seeds.png")
        self.assertEqual(len(figure.axes), 2)
        axis = figure.axes[0]
        points = [collection for collection in axis.collections
                  if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
        np.testing.assert_allclose([point.get_offsets()[0][1] for point in points], rhos)
        filled = [paper_plot.mpl.colors.to_hex(point.get_facecolors()[0])
                  == paper_plot.mpl.colors.to_hex(paper_plot._LESION_POINT_COLOR)
                  for point in points]
        self.assertEqual(filled, [True, False, False])
        dashes = [collection for collection in axis.collections
                  if isinstance(collection, paper_plot.mpl.collections.LineCollection)]
        np.testing.assert_allclose(dashes[0].get_segments()[0][:, 1], [-0.3, -0.3])
        expected_p = wilcoxon(rhos, alternative="less").pvalue
        title = axis.get_legend().get_title().get_text()
        self.assertIn("3 runs", title)
        self.assertIn("one-sided", title)
        self.assertIn(f"p = {expected_p:.3f}", title)
        self.assertEqual([tick.get_text() for tick in axis.get_xticklabels()],
                         ["Synapse clusters"])
        self.assertIn(f"3/3 rho < 0, 1 significant, one-sided Wilcoxon p = {expected_p:.3f}",
                      save.call_args.kwargs["extra"])
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("everything_seed9_L21e4", printed)
        right = figure.axes[1]
        right_points = [collection for collection in right.collections
                        if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
        np.testing.assert_allclose([point.get_offsets()[0][1] for point in right_points], corr_rhos)
        # The same network occupies the same horizontal offset in both panels.
        np.testing.assert_allclose([point.get_offsets()[0][0] for point in points],
                                   [point.get_offsets()[0][0] for point in right_points])
        right_filled = [paper_plot.mpl.colors.to_hex(point.get_facecolors()[0])
                        == paper_plot.mpl.colors.to_hex(paper_plot._LESION_POINT_COLOR)
                        for point in right_points]
        self.assertEqual(right_filled, [True, False, False])
        right_dashes = [collection for collection in right.collections
                        if isinstance(collection, paper_plot.mpl.collections.LineCollection)]
        np.testing.assert_allclose(right_dashes[0].get_segments()[0][:, 1], [0.15, 0.15])
        expected_corr_p = wilcoxon(corr_rhos, alternative="greater").pvalue
        right_title = right.get_legend().get_title().get_text()
        self.assertIn("3 runs", right_title)
        self.assertIn(r"$\rho$ > 0", right_title)
        self.assertIn(f"p = {expected_corr_p:.3f}", right_title)
        self.assertIn("Spearman", right.get_ylabel())
        self.assertIn("profile Pearson r", right.get_ylabel())

    def test_missing_or_undefined_correlation_excludes_run_from_both_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = [self.run_dir(root, 1, -0.5, 0.01),
                           self.run_dir(root, 2, -0.2, 0.1),
                           self.run_dir(root, 3, -0.9, 0.001, missing_correlation=True),
                           self.run_dir(root, 4, -0.8, 0.001, correlation=(np.nan, np.nan))]
            save, output = self.draw(directories)
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        for axis in figure.axes:
            points = [collection for collection in axis.collections
                      if isinstance(collection, paper_plot.mpl.collections.PathCollection)]
            self.assertEqual(len(points), 2)
            self.assertIn("2 runs", axis.get_legend().get_title().get_text())
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("Missing Pearson task-profile panel", printed)
        self.assertIn("undefined saved L1/Pearson OM rank statistics", printed)

    def test_fewer_than_two_runs_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save, output = self.draw([self.run_dir(root, 1, -0.5, 0.01)])
        save.assert_not_called()
        self.assertIn("need >= 2", output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
