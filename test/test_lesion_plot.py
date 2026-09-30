"""Mode-specific normalized modulation effects exported for paper figures."""

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import lesion_plot
import paper_plot


class NormalizedModulationEffectTests(unittest.TestCase):
    def raw(self, mode, accuracy):
        return {
            "mod_lesion_mode": mode,
            "all_tasks": ["fdanti", "fdgo"],
            "all_comb_names_mod": ["mod_c21", "mod_nolesion", "mod_c3"],
            "modtask_accs": np.full((2, 3), accuracy),
            "modrandomtask_accs": np.array([[0.8, 1., 0.6], [0.7, 1., 0.9]]),
        }

    def test_exports_each_mode_with_its_own_control_and_original_order(self):
        base_key = "modulation_all_var_weighted_unnormalized"
        for mode, accuracy in (("zero_W", 0.2), ("freeze_M", 0.5)):
            with self.subTest(mode=mode):
                raw = self.raw(mode, accuracy)
                base, actual_mode, record = lesion_plot._normalized_modulation_effect_record(
                    f"{base_key}__{mode}", raw, ["unused"])
                self.assertEqual((base, actual_mode), (base_key, mode))
                self.assertEqual(record["mod_lesion_mode"], mode)
                self.assertEqual(record["tasks"], ["fdanti", "fdgo"])
                self.assertEqual(record["conditions"], ["mod_c21", "mod_c3"])
                self.assertEqual(record["definition"], "random_minus_lesion")
                self.assertEqual(record["units"], "fraction")
                np.testing.assert_allclose(record["effect"],
                                           raw["modrandomtask_accs"][:, [0, 2]] - accuracy)
                self.assertEqual(raw["all_comb_names_mod"][1], "mod_nolesion")

    def test_legacy_record_defaults_to_zero_w_and_excludes_old_baseline(self):
        raw = self.raw("zero_W", 0.2)
        del raw["mod_lesion_mode"]
        del raw["all_tasks"]
        raw["all_comb_names_mod"][1] = "mod_cNone"
        base, mode, record = lesion_plot._normalized_modulation_effect_record(
            "modulation_all_var_weighted_unnormalized", raw, ["fdanti", "fdgo"])
        self.assertEqual(mode, "zero_W")
        self.assertEqual(base, "modulation_all_var_weighted_unnormalized")
        self.assertEqual(record["conditions"], ["mod_c21", "mod_c3"])
        self.assertEqual(record["tasks"], ["fdanti", "fdgo"])

    def test_exported_modes_feed_primary_heatmap_without_recomputing_effects(self):
        entries = {"lesion_unnorm": lesion_plot._normalized_effect_record(
            [[0.1, 0.2], [0.2, 0.1]], ["fdgo", "fdanti"], ["pre_c21", "post_c21"])}
        base_key = "modulation_all_var_weighted_unnormalized"
        for mode, accuracy in (("zero_W", 0.2), ("freeze_M", 0.5)):
            base, exported_mode, record = lesion_plot._normalized_modulation_effect_record(
                f"{base_key}__{mode}", self.raw(mode, accuracy), [])
            entries[f"{base}__{exported_mode}"] = record
        cache = {"schema_version": 1, "aname": paper_plot.LESION_ANAME, "entries": entries,
                 "primary_modulation_mode": "zero_W"}
        with patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_pkl_or_skip", return_value=cache), \
                patch.object(paper_plot, "_save_fig") as save, \
                patch.object(paper_plot, "_save_standalone_colorbar"):
            paper_plot.plot_lesion_heatmap()
        save.assert_called_once()
        figure = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, figure)
        np.testing.assert_allclose(figure.axes[2].collections[0].get_array().reshape(2, 2),
                                   [[50., 70.], [60., 40.]])
        self.assertEqual([figure.axes[2].xaxis.get_major_formatter()(index)
                          for index in figure.axes[2].get_xticks()], ["U", "C3"])

    def test_mislabeled_interventions_and_mismatched_controls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            lesion_plot._normalized_modulation_effect_record(
                "modulation_all__zero_W", self.raw("freeze_M", 0.5), [])
        raw = self.raw("zero_W", 0.2)
        raw["modrandomtask_accs"] = np.ones((1, 3))
        with self.assertRaisesRegex(ValueError, "shapes"):
            lesion_plot._normalized_modulation_effect_record(
                "modulation_all__zero_W", raw, [])


if __name__ == "__main__":
    unittest.main()