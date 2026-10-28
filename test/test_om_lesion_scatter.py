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
    def test_scatter_keeps_zero_w_unlike_the_heatmap(self):
        self.assertEqual(MODE, "zero_W")
        self.assertNotEqual(MODE, paper_plot.LESION_HEATMAP_MODE)

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

    def selection(self, om=2.5):
        """A one-example OM selection whose deciding block (input C2, hidden
        C1 of cluster C7) has OM `om` and whose color limit is 3."""
        ratios = np.zeros((1, 2, 2))
        ratios[0, 1, 0] = om
        return {"ratios": ratios, "cluster_ids": [7], "chosen": [0], "tops": [(1, 0)], "vmax": 3.0}

    def test_highlight_finds_the_example_block_by_its_om(self):
        with patch.object(paper_plot, "ANAME", "run"), \
                patch.object(paper_plot, "_om_example_selection", return_value=self.selection()):
            found = paper_plot._om_scatter_highlight(np.array([0.5, 2.5, 1.0]), "run")
            self.assertEqual(found, {"point": 1, "om": 2.5, "cluster_id": 7,
                                     "input": 2, "hidden": 1, "vmax": 3.0})
            with patch("builtins.print"):
                # Two points with the block's OM: ambiguous, nothing highlighted.
                self.assertIsNone(paper_plot._om_scatter_highlight(np.array([2.5, 2.5]), "run"))
                # A scatter of another run cannot hold the example block.
                self.assertIsNone(paper_plot._om_scatter_highlight(np.array([2.5]), "other_run"))

    def test_highlight_is_drawn_on_top_in_the_examples_color(self):
        cache = self.cache(False)
        highlight = {"point": 0, "om": 1.5, "cluster_id": 7, "input": 2, "hidden": 1, "vmax": 3.0}
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
                                   [[cache["om_vals"][0], cache["lesion_diffs"][0]]])
        expected = paper_plot.plt.get_cmap(paper_plot._MULTITASK_HEATMAP_CMAP)(0.5)
        np.testing.assert_allclose(ringed[0].get_facecolors()[0], expected)
        self.assertGreater(ringed[0].get_zorder(), 3)
        self.assertIn("Example block", [text.get_text() for text in axis.get_legend().get_texts()])
        self.assertIn("highlighted: example 1 block, synapse cluster C7 at input C2 / hidden C1",
                      save.call_args.kwargs["extra"])

    def test_single_cache_of_the_other_mode_is_rejected(self):
        cache = self.cache(False)
        cache["mod_lesion_mode"] = "freeze_M"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f"om_vs_lesion_diff_var-weighted-unnormalized_{MODE_TAG}_unnorm_{paper_plot.LESION_ANAME}.pkl"
            with path.open("wb") as stream:
                pickle.dump(cache, stream)
            with patch.object(paper_plot, "LESION_NORM_DIR", root):
                self.assertIsNone(paper_plot._load_om_lesion_scatter())

    def summary(self):
        return lesion_plot._om_scatter_summary(
            [[0.], [0.8]], [[0., 2., 4.], [1., 3., 5.]],
            [[[1.], [0.3], [0.]], [[0.], [0.5], [0.8]]], n_perm=13, seed=4)

    def cache(self, combined):
        entry = self.summary()
        result = {
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
                axis = figure.axes[0]
                self.assertEqual(axis.get_ylim()[0], 0)
                self.assertEqual(len(axis.lines), 1)
                np.testing.assert_array_equal(axis.lines[0].get_xdata(), entry["binned_medians"]["x"])
                np.testing.assert_array_equal(axis.lines[0].get_ydata(), entry["binned_medians"]["y"])
                np.testing.assert_allclose(axis.collections[0].get_offsets(),
                                           np.column_stack((entry["om_vals"], entry["lesion_diffs"])))
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

    def test_undefined_rank_still_displays_nonnegative_data_without_legend(self):
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
    """The cohort view reads every sibling run's saved OM association."""

    scatter = OmLesionScatterTests()

    def run_dir(self, root, seed, rho=None, p_perm=None, legacy=False):
        """A sibling run directory holding a combined OM cache for that run."""
        aname = f"everything_seed{seed}_L21e4+hidden300+batch128+angle"
        directory = root / aname
        directory.mkdir()
        cache = self.scatter.cache(True)
        cache["aname"] = aname
        association = cache["mode_data"][MODE]["association"]
        if rho is not None:
            association.update(rho=rho, p_perm=p_perm)
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = [self.run_dir(root, seed, rho, p)
                           for seed, (rho, p) in enumerate(zip(rhos, p_values))]
            # A legacy cache is left out rather than drawn.
            directories.append(self.run_dir(root, 9, legacy=True))
            save, output = self.draw(directories)
        figure, path = save.call_args.args[:2]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertEqual(path.name, "multitask_om_vs_lesion_seeds.png")
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

    def test_fewer_than_two_runs_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save, output = self.draw([self.run_dir(root, 1, -0.5, 0.01)])
        save.assert_not_called()
        self.assertIn("need >= 2", output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()