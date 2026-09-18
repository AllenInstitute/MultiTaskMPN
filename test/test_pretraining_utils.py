"""Contract tests for metadata shared by pretraining analysis drivers."""

import unittest

from pretrain import pretraining_utils


class PretrainingUtilsTests(unittest.TestCase):
    def test_current_ruleset_contract(self):
        self.assertEqual(
            set(pretraining_utils.RULESET_SPECS),
            {"fdanti_delaygo", "fdgo_delaygo", "fdanti", "fdgo"},
        )
        self.assertEqual(
            pretraining_utils.stage1_tasks_for("fdanti_delaygo"),
            ["fdanti", "delaygo"],
        )
        self.assertEqual(
            pretraining_utils.stage1_tasks_for("fdgo_delaygo"),
            ["fdgo", "delaygo"],
        )
        self.assertEqual(
            pretraining_utils.stage1_tasks_for("fdanti"), ["fdanti"])
        self.assertEqual(
            pretraining_utils.stage1_tasks_for("fdgo"), ["fdgo"])
        for ruleset in pretraining_utils.RULESET_SPECS:
            self.assertEqual(
                pretraining_utils.stage2_tasks_for(ruleset), ["delayanti"])

    def test_display_names(self):
        expected = {
            "fdgo": "DelayPro",
            "fdanti": "DelayAnti",
            "delaygo": "MemoryPro",
            "delayanti": "MemoryAnti",
        }
        self.assertEqual(pretraining_utils.RULE_DISPLAY_NAMES, expected)
        for rule, label in expected.items():
            self.assertEqual(pretraining_utils.display_rule(rule), label)

    def test_variant_and_run_names_preserve_existing_format(self):
        self.assertEqual(
            pretraining_utils.variant_addon(200, "L21e3"),
            "+hidden200+L21e3+batch128+angle",
        )
        self.assertEqual(
            pretraining_utils.run_name(
                "fdanti_delaygo", "dmpn", 862, 200, "L21e3"),
            "fdanti_delaygo_dmpn_seed862_+hidden200+L21e3+batch128+angle",
        )

    def test_seed_selection_is_stable_and_ruleset_specific(self):
        seeds = range(10, 20)
        expected = {
            "fdgo_delaygo": [10, 11, 17],
            "fdanti_delaygo": [14, 17, 19],
            "fdanti": [11, 17, 18],
            "fdgo": [16, 18, 19],
        }
        for ruleset, chosen in expected.items():
            self.assertEqual(
                pretraining_utils.select_seeds(
                    seeds, total_seed=3, test_seed=17, ruleset=ruleset),
                chosen,
            )
        self.assertEqual(
            pretraining_utils.select_seeds(
                [7, 2, 5], total_seed=None, test_seed=17,
                ruleset="fdanti_delaygo"),
            [2, 5, 7],
        )

    def test_layout_for_one_and_two_parent_rulesets(self):
        for ruleset, stage1_rules, width in (
            ("fdanti_delaygo", ["fdanti", "delaygo"], 9),
            ("fdgo_delaygo", ["fdgo", "delaygo"], 9),
            ("fdanti", ["fdanti"], 8),
            ("fdgo", ["fdgo"], 8),
        ):
            stage1 = {"rules": stage1_rules, "hp": {"rule_start": 6}}
            stage2 = {"rules": ["delayanti"], "hp": {"rule_start": 6}}
            layout = pretraining_utils.build_task_layout(
                stage1, stage2, ruleset, input_width=width)
            self.assertEqual(layout["input_width"], width)
            self.assertEqual(layout["novel_rule_index"], len(stage1_rules))
            self.assertEqual(layout["novel_weight_column"], width - 1)
            self.assertEqual(
                layout["pretraining_weight_slice"],
                slice(6, 6 + len(stage1_rules)),
            )

    def test_layout_rejects_wrong_saved_tasks(self):
        stage1 = {"rules": ["fdgo", "delaygo"], "hp": {"rule_start": 6}}
        stage2 = {"rules": ["delayanti"], "hp": {"rule_start": 6}}
        with self.assertRaisesRegex(ValueError, "unexpected stage task"):
            pretraining_utils.build_task_layout(
                stage1, stage2, "fdanti_delaygo", input_width=9)


if __name__ == "__main__":
    unittest.main()
