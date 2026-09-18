"""Tests for the standalone sibling-family analysis boundary."""

import tempfile
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
import sibling_delay_analysis as sibling


class SiblingDelayAnalysisTests(unittest.TestCase):
    def test_family_names_expand_to_both_rules(self):
        self.assertEqual(
            sibling.SIBLING_FAMILIES["delaydm1"],
            ("delaydm1", "delaydm2"),
        )
        self.assertEqual(
            sibling.SIBLING_FAMILIES["dmcgo"],
            ("dmcgo", "dmcnogo"),
        )

    def test_run_identifier_matches_existing_convention(self):
        self.assertEqual(
            sibling.build_aname(921, "L21e4"),
            "everything_seed921_L21e4+hidden300+batch128+angle",
        )

    def test_cleanup_is_scoped_to_selected_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            delay_paths = [
                root / "fixed_points_grad_run_delaydm1.pkl",
                root / "fixed_points_grad_run_delaydm2.pkl",
                root / "delaydm1_delay_pc_projections_run.pkl",
            ]
            retained_paths = [
                root / "fixed_points_grad_run_dmcgo.pkl",
                root / "cluster_info_run.pkl",
            ]
            for path in delay_paths + retained_paths:
                path.touch()

            removed = sibling._clean_stale_sibling_artifacts(
                root, ("delaydm1",))

            self.assertEqual(set(removed), set(delay_paths))
            self.assertTrue(all(not path.exists() for path in delay_paths))
            self.assertTrue(all(path.exists() for path in retained_paths))

    def test_artifact_ownership_covers_both_families(self):
        self.assertTrue(sibling.is_sibling_artifact(
            "fixed_points_grad_run_delaydm2.pkl"))
        self.assertTrue(sibling.is_sibling_artifact(
            "dmcgo_delay_pc_projections_run.pkl"))
        self.assertFalse(sibling.is_sibling_artifact("cluster_info_run.pkl"))

    def test_cli_requires_an_explicit_family(self):
        parser = sibling.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--seed", "921", "--feature", "L21e4"])
        args = parser.parse_args([
            "--seed", "921", "--feature", "L21e4",
            "--families", "delaydm1",
        ])
        self.assertEqual(args.families, ["delaydm1"])


if __name__ == "__main__":
    unittest.main()
