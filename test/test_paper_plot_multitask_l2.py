"""The MULTITASK_L2 control selects one cohort for the multiple_tasks / lesion figures."""

import importlib.util
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import paper_plot


def aname(seed, l2="1e3"):
    return f"everything_seed{seed}_L2{l2}+hidden300+batch128+angle"


def write_cache(root, run, stem="lesion_prune_results"):
    directory = Path(root) / run
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_{run}.pkl"
    with path.open("wb") as stream:
        pickle.dump({"source": run}, stream)
    return path


class MultitaskRunResolutionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.roots = (self.root / "perf", self.root / "norm")

    def resolve(self, l2="1e3", category="lesion"):
        with patch.dict(paper_plot._MULTITASK_CACHE_ROOTS, {category: self.roots}):
            return paper_plot._resolve_multitask_run(l2, category)

    def test_run_names_follow_the_cohort(self):
        self.assertEqual(paper_plot.multitask_feature("1e4"), "L21e4")
        self.assertEqual(paper_plot.multitask_aname(86, "1e3"), aname(86))
        self.assertEqual(paper_plot.multitask_aname("*", "1e4"), aname("*", "1e4"))

    def test_designated_seed_wins_when_either_root_has_its_cache(self):
        write_cache(self.roots[1], aname(921, "1e4"))
        for root in self.roots:
            write_cache(root, aname(86, "1e4"))
        with patch("builtins.print") as output:
            self.assertEqual(self.resolve("1e4"), aname(921, "1e4"))
        output.assert_not_called()

    def test_without_designated_seed_the_best_covered_lowest_seed_is_chosen(self):
        write_cache(self.roots[0], aname(86))
        for root in self.roots:
            write_cache(root, aname(196), stem="leison_prune_results")
        with patch("builtins.print") as output:
            self.assertEqual(self.resolve(), aname(196))
        output.assert_not_called()
        write_cache(self.roots[1], aname(86))
        self.assertEqual(self.resolve(), aname(86))

    def test_missing_designated_seed_falls_back_within_the_cohort_and_reports(self):
        for seed in (196, 86):
            write_cache(self.roots[0], aname(seed, "1e4"))
        with patch("builtins.print") as output:
            self.assertEqual(self.resolve("1e4"), aname(86, "1e4"))
        message = output.call_args.args[0]
        self.assertIn(aname(921, "1e4"), message)
        self.assertIn("seed 86", message)

    def test_other_cohorts_empty_directories_and_foreign_pickles_do_not_count(self):
        for root in self.roots:
            write_cache(root, aname(86, "1e4"))
        (self.roots[0] / aname(1)).mkdir(parents=True)
        stray = self.roots[1] / aname(196)
        stray.mkdir(parents=True)
        (stray / f"lesion_prune_results_{aname(86)}.pkl").touch()
        with patch("builtins.print") as output:
            self.assertEqual(self.resolve("1e3"), aname(0))
        self.assertIn("no cached L21e3 run", output.call_args.args[0])
        # With nothing cached at all, the designated name is kept so the
        # per-figure "Skipped" messages name the intended run.
        with patch.dict(paper_plot._MULTITASK_CACHE_ROOTS,
                        {"multiple_tasks": (self.root / "empty",)}), \
                patch("builtins.print"):
            self.assertEqual(paper_plot._resolve_multitask_run("1e4", "multiple_tasks"),
                             aname(749, "1e4"))

    def test_unknown_cohort_is_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve("1e2")


class MultitaskL2ControlTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def load_module(self):
        spec = importlib.util.spec_from_file_location("paper_plot_l2", paper_plot.__file__)
        module = importlib.util.module_from_spec(spec)
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            with patch("builtins.print"):
                spec.loader.exec_module(module)
                return module
        finally:
            os.chdir(previous_cwd)

    def test_import_resolves_both_categories_from_the_default_cohort(self):
        clustering = aname(86)
        lesion = aname(196)
        write_cache(self.root / "multiple_tasks_analysis", clustering, "cluster_info")
        write_cache(self.root / "multiple_tasks_analysis", aname(196, "1e4"), "cluster_info")
        write_cache(self.root / "multiple_tasks_perf", lesion)
        write_cache(self.root / "multiple_tasks_norm", lesion, "normalized_lesion_effects")
        write_cache(self.root / "multiple_tasks_norm", aname(921, "1e4"), "normalized_lesion_effects")
        module = self.load_module()
        self.assertEqual(module.MULTITASK_L2, "1e3")
        self.assertEqual(module.ANAME, clustering)
        self.assertEqual(module.LESION_ANAME, lesion)
        self.assertEqual(module.DATA_DIR, Path("multiple_tasks_analysis") / clustering)
        self.assertEqual(module.LESION_DIR, Path("multiple_tasks_perf") / lesion)
        self.assertEqual(module.LESION_NORM_DIR, Path("multiple_tasks_norm") / lesion)
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertEqual(module._load_cluster_info(), {"source": clustering})
            self.assertEqual(module._load_lesion_results(), {"source": lesion})
            self.assertEqual([d.name for d in module._find_experiment_dirs()], [clustering])
            self.assertEqual([d.name for d in module._sibling_run_dirs()], [lesion])
            module._set_multitask_l2("1e4")
        finally:
            os.chdir(previous_cwd)
        self.assertEqual(module.MULTITASK_L2, "1e4")
        self.assertEqual(module.ANAME, aname(196, "1e4"))
        self.assertEqual(module.LESION_ANAME, aname(921, "1e4"))
        self.assertEqual(module.LESION_NORM_DIR, Path("multiple_tasks_norm") / aname(921, "1e4"))

    def test_command_line_override_switches_cohort_before_plotting(self):
        modes = {mode: {name: Mock(name=name) for name in group}
                 for mode, group in paper_plot.FIGURES_BY_MODE.items()}
        figures = {name: fn for group in modes.values() for name, fn in group.items()}
        for arguments, expected_calls in ((["lesion"], []),
                                          (["--multitask-l2", paper_plot.MULTITASK_L2, "lesion"], []),
                                          (["--multitask-l2", "1e4", "lesion"], ["1e4"])):
            with self.subTest(arguments=arguments), \
                    patch.object(paper_plot, "FIGURES_BY_MODE", modes), \
                    patch.object(paper_plot, "ALL_FIGURES", figures), \
                    patch.object(paper_plot, "OUT_DIR", self.root), \
                    patch.object(paper_plot, "_set_pretraining_bound"), \
                    patch.object(paper_plot, "_set_multitask_l2") as setter, \
                    patch("sys.argv", ["paper_plot.py", *arguments]), \
                    patch("builtins.print") as output:
                paper_plot.main()
                self.assertEqual([call.args[0] for call in setter.call_args_list], expected_calls)
                self.assertTrue(any(
                    f"lesion: {paper_plot.LESION_ANAME} (L2={paper_plot.MULTITASK_L2}" in str(call.args[0])
                    for call in output.call_args_list if call.args))
            for figure in modes["lesion"].values():
                figure.reset_mock()


if __name__ == "__main__":
    unittest.main()
