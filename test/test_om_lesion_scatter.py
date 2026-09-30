"""OM paper plots consume saved rank statistics and medians without fitting."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot
import lesion_plot

MODE = paper_plot.OM_LESION_MODE
MODE_TAG = MODE.replace("_", "-")


class OmLesionScatterTests(unittest.TestCase):
    def test_scatter_uses_the_heatmap_lesion_mode(self):
        self.assertEqual(MODE, "zero_W")

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
                with patch.object(paper_plot, "LESION_NORM_DIR", root), \
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
                self.assertEqual(scatter.get_alpha(), 0.8)
                self.assertEqual(scatter.get_zorder(), 3)
                np.testing.assert_array_equal(scatter.get_sizes(), [40])
                np.testing.assert_array_equal(scatter.get_linewidths(), [0.5])
                np.testing.assert_allclose(scatter.get_facecolors(),
                                           [paper_plot.mpl.colors.to_rgba("#3182ce", 0.8)])
                np.testing.assert_allclose(scatter.get_edgecolors(),
                                           [paper_plot.mpl.colors.to_rgba("k", 0.8)])
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
                with patch.object(paper_plot, "LESION_NORM_DIR", root), \
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
            with patch.object(paper_plot, "LESION_NORM_DIR", root), \
                    patch.object(paper_plot, "SHOW_LEGEND", False), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save:
                paper_plot.plot_om_vs_lesion()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        self.assertIsNone(figure.axes[0].get_legend())
        self.assertEqual(figure.axes[0].get_ylim()[0], 0)

if __name__ == "__main__":
    unittest.main()