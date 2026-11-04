"""Regression coverage for state-space provenance and undefined response angles."""
import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot
import state_space_shift as shift
import state_space_structure as structure
import state_space_transplant as transplant
import state_space_vs_lesion as lesion
from test_state_space_distance_angle import _distance_angle_entry
from test_state_space_structure import _pca_cache
from test_state_space_transplant import _tiny_network, _trials, RULES, CONTEXT_STEPS


class AlignmentProvenanceTests(unittest.TestCase):
    def test_structure_propagates_alignment_and_rejects_unknown_inputs(self):
        data = _pca_cache("run", 1e-4, 0)
        result = structure.analyze_run(data, n_permutations=5)
        for entry in (result, structure.summarize_run(result),
                      structure.cohort_mean_structure({"run": result}, n_permutations=5)):
            self.assertEqual(entry["m_alignment"], shift.M_ALIGNMENT)
        for marker in (None, "pre_update"):
            data["m_alignment"] = marker
            with self.assertRaisesRegex(ValueError, "m_alignment"):
                structure.analyze_run(data, n_permutations=5)
            stale = {**result, "m_alignment": marker}
            with self.assertRaisesRegex(ValueError, "m_alignment"):
                structure.summarize_run(stale)
            with self.assertRaisesRegex(ValueError, "m_alignment"):
                structure.cohort_mean_structure({"run": stale}, n_permutations=5)

    def test_structure_readers_reject_unmarked_derived_caches(self):
        result = structure.analyze_run(_pca_cache("run", 1e-4, 0), n_permutations=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = structure.structure_cache_path("run", root)
            for marker in (shift.M_ALIGNMENT, None, "pre_update"):
                result["m_alignment"] = marker
                with path.open("wb") as stream:
                    pickle.dump(result, stream)
                with patch.object(paper_plot, "STATE_SPACE_DIR", root), patch("builtins.print"):
                    loaded = paper_plot._load_state_space_structure("run")
                    combined = paper_plot._load_aligned_structure(path)
                if marker == shift.M_ALIGNMENT:
                    self.assertIsNotNone(loaded)
                    self.assertIsNotNone(combined)
                    self.assertEqual(lesion.load_structure_distances(path)[0], "run")
                else:
                    self.assertIsNone(loaded)
                    self.assertIsNone(combined)
                    with self.assertRaisesRegex(ValueError, "m_alignment"):
                        lesion.load_structure_distances(path)

    def test_lesion_requires_provenance_before_analysis_and_plotting(self):
        with self.assertRaisesRegex(ValueError, "m_alignment"):
            lesion.analyze_run("run", 1e-4, [], {}, {}, m_alignment=None)
        with patch("builtins.print") as output:
            self.assertIsNone(paper_plot._state_space_vs_lesion_entry({"run": {}}, "run"))
        self.assertIn("m_alignment", output.call_args.args[0])

    def test_source_summary_and_paper_plot_exclude_the_same_stale_entries(self):
        current = _distance_angle_entry()
        current["rval_dict"] = {key: (0.2, 1.0, 0.01) for key in current["rval_dict"]}
        old = copy.deepcopy(current)
        old["schema_version"] = 2
        old["rval_dict"] = {key: (0.9, 1.0, 0.01) for key in old["rval_dict"]}
        unmarked = {**old, "schema_version": shift.DISTANCE_ANGLE_SCHEMA_VERSION}
        unmarked.pop("m_alignment")
        results = {"current": current, "old": old, "unmarked": unmarked}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / shift.RESULTS_NAME).open("wb") as stream:
                pickle.dump(results, stream)
            with patch("matplotlib.figure.Figure.savefig", autospec=True) as save, \
                    patch("builtins.print") as output:
                shift.summarize(root)
                grouped = paper_plot._state_space_r_values_by_l2(results, ["hidden"])
            figure = save.call_args.args[0]
            np.testing.assert_allclose([bar.get_height() for bar in figure.axes[0].patches], 0.2)
            self.assertIn("linear decay removed", figure.axes[0].get_title())
            self.assertEqual([entry["aname"] for entry in grouped[1e-4]["hidden"]], ["current"])
            self.assertTrue(any("Skipped 2 stale" in call.args[0] for call in output.call_args_list))


class UndefinedTransplantResponseTests(unittest.TestCase):
    def analyze(self, mode):
        model = _tiny_network()
        inputs, indices = _trials()
        if mode == "one":
            selected = indices == 0
            inputs[selected, CONTEXT_STEPS] = inputs[selected, CONTEXT_STEPS - 1]
        elif mode == "all":
            inputs[:, CONTEXT_STEPS] = inputs[:, CONTEXT_STEPS - 1]
        elif mode == "constant":
            inputs[:] = inputs[0].clone()
        with patch.object(transplant, "DISTANCE_ANGLE_N_PERMUTATIONS", 20):
            return transplant.transplant_analysis(model, inputs, indices, RULES, CONTEXT_STEPS, 0.9)

    def test_zero_hidden_response_does_not_drop_modulation_pairs(self):
        r_values, scatter, _, _ = self.analyze("one")
        self.assertEqual(len(scatter["hidden"]["dists"]), 3)
        self.assertIsNone(scatter["hidden"]["permutation"])
        self.assertIn("defined response angles", scatter["hidden"]["permutation_skipped"])
        self.assertNotIn("stimulus", scatter["hidden"]["permutation_skipped"])
        self.assertTrue(np.isfinite(r_values["hidden"][0]))
        for name in ("mod", "eff_mod"):
            self.assertEqual(len(scatter[name]["dists"]), 6)
            self.assertEqual(scatter[name]["n_steps_skipped"], 0)
            self.assertIsNotNone(scatter[name]["permutation"])
            self.assertTrue(np.isfinite(r_values[name][0]))

    def test_all_hidden_responses_zero_are_saved_as_undefined_without_aborting(self):
        r_values, scatter, norms, check = self.analyze("all")
        self.assertEqual(scatter["hidden"]["dists"].shape, (0,))
        self.assertEqual(scatter["hidden"]["angles_deg_by_host"].shape, (0, 2))
        self.assertTrue(np.isnan(r_values["hidden"][0]))
        self.assertIn("fewer than two", scatter["hidden"]["regression"]["skipped"])
        for name in ("mod", "eff_mod"):
            self.assertEqual(len(scatter[name]["dists"]), 6)
            self.assertTrue(np.isfinite(r_values[name][0]))
        with tempfile.TemporaryDirectory() as tmp, patch("builtins.print"):
            transplant._draw_transplant_figure(scatter, r_values, Path(tmp) / "diagnostic.png")
            transplant.summarize({"run": {"rval_dict": r_values, "scatter": scatter,
                                         "step_norms": norms, "native_check": check}}, "test")

    def test_constant_distances_do_not_crash_regression_or_permutation(self):
        r_values, scatter, _, _ = self.analyze("constant")
        for name in transplant.REPRESENTATIONS:
            self.assertEqual(len(scatter[name]["dists"]), 6)
            self.assertTrue(np.isnan(r_values[name][0]))
            self.assertIsNone(scatter[name]["permutation"])
            self.assertIn("constant", scatter[name]["regression"]["skipped"])

    def test_undefined_test_is_reported_as_undefined_instead_of_legacy(self):
        r_values, scatter, _, _ = self.analyze("all")
        entry = {"schema_version": transplant.TRANSPLANT_SCHEMA_VERSION,
                 "rval_dict": r_values, "scatter": scatter}
        group = {name: [] for name in transplant.REPRESENTATIONS}
        with patch("builtins.print") as output:
            legacy = paper_plot._add_r_value_entry(
                group, "run", entry, transplant.REPRESENTATIONS,
                transplant.TRANSPLANT_SCHEMA_VERSION)
        self.assertEqual(legacy, 0)
        self.assertEqual(group["hidden"], [])
        self.assertEqual(len(group["mod"]), 1)
        self.assertTrue(any("fewer than two" in call.args[0] for call in output.call_args_list))


if __name__ == "__main__":
    unittest.main()
