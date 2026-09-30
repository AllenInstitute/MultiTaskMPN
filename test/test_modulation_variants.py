"""The five modulation clustering variants agree across the pipeline stages."""

import ast
import inspect
import unittest

import _bootstrap  # noqa: F401
import numpy as np
import lesion
import lesion_plot
import modulation_variants as mv
import multiple_task_analysis


class VarianceWeightingTests(unittest.TestCase):
    def setUp(self):
        self.W = np.array([[0.5, -2.0], [0.0, 3.0]])          # (post, pre)
        self.vars = np.arange(1, 9, dtype=float).reshape(2, 4)  # (conditions, synapses)

    def test_unweighted_variants_return_a_copy(self):
        for name in ("modulation_all", "modulation_all_weighted"):
            out = mv.weight_modulation_variance(self.vars, name, self.W)
            np.testing.assert_array_equal(out, self.vars)
            self.assertIsNot(out, self.vars)

    def test_signed_and_absolute_weighting(self):
        signed = mv.weight_modulation_variance(self.vars, "modulation_all_var_weighted", self.W)
        np.testing.assert_allclose(signed, self.vars * np.array([0.5, -2.0, 0.0, 3.0]))
        absolute = mv.weight_modulation_variance(self.vars, "modulation_all_abs_weighted", self.W)
        np.testing.assert_allclose(absolute, self.vars * np.array([0.5, 2.0, 0.0, 3.0]))
        self.assertTrue((absolute >= 0).all())

    def test_unknown_variant_and_shape_mismatch_raise(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            mv.weight_modulation_variance(self.vars, "modulation_all_abs_weightd", self.W)
        with self.assertRaises(ValueError):
            mv.weight_modulation_variance(self.vars[:, :3], "modulation_all_abs_weighted", self.W)


class RegistryTests(unittest.TestCase):
    def test_five_lesion_types_include_abs_weighted_and_have_colors(self):
        self.assertEqual(len(mv.LESION_MODULATION_TYPES), 5)
        self.assertIn("modulation_all_abs_weighted_unnormalized", mv.LESION_MODULATION_TYPES)
        self.assertEqual(len(set(mv.LESION_MODULATION_TYPES)), 5)
        for type_key in mv.LESION_MODULATION_TYPES:
            self.assertIn(mv.modulation_type_tag(type_key), mv.MODULATION_TYPE_COLORS)
        self.assertEqual(mv.modulation_type_tag("modulation_all_abs_weighted_unnormalized"),
                         "abs-weighted-unnormalized")
        self.assertEqual(len(set(mv.MODULATION_TYPE_COLORS.values())),
                         len(mv.MODULATION_TYPE_COLORS))

    def test_every_saved_name_has_a_variance_weight_rule(self):
        for type_key in mv.LESION_MODULATION_TYPES:
            base = type_key.replace("_unnormalized", "").replace("_normalized", "")
            self.assertIn(base, mv.MODULATION_VARIANCE_WEIGHTS)

    def test_analysis_clustering_list_produces_exactly_the_lesion_types(self):
        """The load-bearing list in multiple_task_analysis.main names every variant."""
        source = inspect.getsource(multiple_task_analysis.main)
        tree = ast.parse(source)
        names = normalize = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "clustering_data_analysis_names":
                    names = ast.literal_eval(node.value)
                elif node.targets[0].id == "clustering_data_normalize":
                    normalize = ast.literal_eval(node.value)
        self.assertIsNotNone(names)
        self.assertEqual(len(names), len(normalize))
        saved = [f"{name}_{'normalized' if flag else 'unnormalized'}"
                 for name, flag in zip(names, normalize) if "all" in name]
        self.assertEqual(saved, list(mv.LESION_MODULATION_TYPES))

    def test_lesion_and_plot_stages_use_the_shared_registry(self):
        self.assertIn("LESION_MODULATION_TYPES", inspect.getsource(lesion.main))
        plot_source = inspect.getsource(lesion_plot.main)
        self.assertIn("LESION_MODULATION_TYPES", plot_source)
        self.assertIn("MODULATION_TYPE_COLORS", plot_source)
        self.assertNotIn('"var-weighted-unnormalized": "#e7298a"', plot_source)


if __name__ == "__main__":
    unittest.main()
