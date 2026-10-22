"""Tests for the automatic per-panel choice of the sibling-family example runs."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np

import paper_plot


N_ALPHA = 3
N_DIRECTIONS = 4
REPRESENTATIONS = ("hidden", "e_modulation")


def _aname(seed, l2=None):
    return paper_plot.multitask_aname(seed, paper_plot.SIBLING_L2 if l2 is None else l2)


def _ring(stim, separation):
    """PC1/PC2 offsets placing each stimulus direction on a ring."""
    angles = 2 * np.pi * stim / N_DIRECTIONS
    return separation * np.column_stack([np.cos(angles), np.sin(angles)])


def _delaydm_caches(rng, separation, sweep_separation=None, endpoint_fixed=True,
                    sweep_fixed=True):
    """(endpoint entry, sweeps) of a DelayDM run whose endpoint directions sit
    `separation` apart and sweep directions `sweep_separation` (default the
    same); `*_fixed` False marks one point unconverged and None drops the
    convergence mask (a cache written before settling)."""
    if sweep_separation is None:
        sweep_separation = separation
    stim = np.tile(np.repeat(np.arange(N_DIRECTIONS), 2), 2)
    proj = rng.normal(scale=0.1, size=(stim.size, 6))
    proj[:, :2] += _ring(stim, separation)
    entry = {"proj": proj, "stim_idx": stim,
             "task_idx": np.repeat([0, 1], stim.size // 2)}
    if endpoint_fixed is not None:
        entry["is_fixed"] = np.ones(stim.size, dtype=bool)
        entry["is_fixed"][0] = endpoint_fixed

    sweeps = {}
    sweep_stim = np.repeat(np.arange(N_DIRECTIONS), 2)
    for base_rule, target_rule in (("delaydm1", "delaydm2"), ("delaydm2", "delaydm1")):
        sweep_proj = rng.normal(scale=0.1, size=(N_ALPHA, sweep_stim.size, 6))
        sweep_proj[:, :, :2] += _ring(sweep_stim, sweep_separation)
        sweep_proj[:, :, 2] += np.linspace(0.0, 1.0, N_ALPHA)[:, None]
        sweep = {"proj": sweep_proj, "stim_idx": sweep_stim,
                 "target_rule": target_rule}
        if sweep_fixed is not None:
            sweep["is_fixed"] = np.ones((N_ALPHA, sweep_stim.size), dtype=bool)
            sweep["is_fixed"][0, 0] = sweep_fixed
        sweeps[base_rule] = sweep
    return entry, sweeps


def _dmc_caches(rng):
    """(endpoint entry, sweeps) of a DMC run whose memory clusters are
    perfectly separated along PC1 in both tasks."""
    task = np.repeat([0, 1], 8)
    stim = np.tile(np.arange(8), 2)
    groups = ((task == 0) == (stim < 4)).astype(int)
    proj = rng.normal(scale=0.1, size=(task.size, 6))
    proj[:, 0] += 2.0 * (2 * groups - 1)
    entry = {"proj": proj, "stim_idx": stim, "task_idx": task,
             "group_labels": groups, "is_fixed": np.ones(task.size, dtype=bool)}
    sweeps = {"dmcgo": {
        "proj": rng.normal(size=(N_ALPHA, 8, 6)), "stim_idx": np.arange(8),
        "target_rule": "dmcnogo", "is_fixed": np.ones((N_ALPHA, 8), dtype=bool)}}
    return entry, sweeps


class SiblingRunSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "two_in_multiples"
        self.root.mkdir()
        self.out = Path(directory.name) / "paper_plot"
        self.rng = np.random.RandomState(0)
        for patcher in (
                patch.object(paper_plot, "TWO_IN_MULTIPLES_DIR", self.root),
                patch.object(paper_plot, "OUT_DIR", self.out),
                patch.object(paper_plot, "DELAYDM_ANAME", None),
                patch.object(paper_plot, "DMCGO_ANAME", None),
                patch.dict(paper_plot._SIBLING_SELECTED_RUNS, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        print_patcher = patch("builtins.print")
        self.output = print_patcher.start()
        self.addCleanup(print_patcher.stop)

    def _write_run(self, family, aname, caches, with_sweep=True):
        entry, sweeps = caches
        run_dir = self.root / aname
        run_dir.mkdir()
        endpoint_path = (run_dir
                         / f"{family}_long_delay_endpoint_pc_projections_{aname}.pkl")
        with endpoint_path.open("wb") as stream:
            pickle.dump({"representations": {rep: entry for rep in REPRESENTATIONS}},
                        stream)
        if not with_sweep:
            return
        sweep_path = paper_plot._multitask_cue_interp_projection_path(
            run_dir, family, aname, "long_delay_endpoint")
        with sweep_path.open("wb") as stream:
            pickle.dump({"kind": "cue_interp",
                         "alphas": np.linspace(0.0, 1.0, N_ALPHA),
                         "representations": {rep: {"sweeps": sweeps}
                                             for rep in REPRESENTATIONS}},
                        stream)

    def _report(self, name):
        return json.loads(
            (self.out / f"multitask_sibling_run_scores_{name}.json").read_text())

    def test_each_panel_takes_its_own_best_run_from_one_ranking(self):
        # Seed 3 has the clearer endpoints, seed 7 the clearer sweeps.
        self._write_run("delaydm1", _aname(3),
                        _delaydm_caches(self.rng, 3.0, sweep_separation=0.5))
        self._write_run("delaydm1", _aname(7),
                        _delaydm_caches(self.rng, 0.5, sweep_separation=3.0))
        # A better run of another L2 cohort is never a candidate.
        self._write_run("delaydm1", _aname(1, "1e3"), _delaydm_caches(self.rng, 9.0))
        with patch.object(paper_plot, "_sibling_run_ranking",
                          wraps=paper_plot._sibling_run_ranking) as ranking:
            selected = {panel: paper_plot._multitask_sibling_aname("delaydm1", panel)
                        for panel in paper_plot._SIBLING_PANELS}
        ranking.assert_called_once_with("delaydm1")
        self.assertEqual(selected, {
            "endpoint hidden": _aname(3), "endpoint e_modulation": _aname(3),
            "sweep hidden": _aname(7), "sweep e_modulation": _aname(7)})

        report = self._report("delaydm")
        self.assertEqual([row["seed"] for row in report["runs"]], [3, 7])
        self.assertEqual(sorted(report["runs"][0]["components"]),
                         sorted(paper_plot._SIBLING_PANELS))
        for panel, seeds in (("endpoint hidden", [3, 7]), ("sweep hidden", [7, 3])):
            choice = report["panels"][panel]
            self.assertEqual(choice["metric"], panel)
            self.assertEqual([row["seed"] for row in choice["ranking"]], seeds)
        messages = [str(call.args[0]) for call in self.output.call_args_list]
        self.assertTrue(any("delaydm1 sweep hidden run selected automatically: seed 7"
                            in text for text in messages))

    def test_runs_that_are_unconverged_unsettled_or_incomplete_are_skipped(self):
        self._write_run("delaydm1", _aname(3), _delaydm_caches(self.rng, 0.5))
        self._write_run("delaydm1", _aname(7),
                        _delaydm_caches(self.rng, 3.0, endpoint_fixed=False))
        self._write_run("delaydm1", _aname(9),
                        _delaydm_caches(self.rng, 3.0, sweep_fixed=None))
        self._write_run("delaydm1", _aname(11), _delaydm_caches(self.rng, 3.0),
                        with_sweep=False)
        for panel in paper_plot._SIBLING_PANELS:
            self.assertEqual(paper_plot._multitask_sibling_aname("delaydm1", panel),
                             _aname(3))

        reasons = {row["seed"]: row["reason"]
                   for row in self._report("delaydm")["skipped"]}
        self.assertEqual(sorted(reasons), [7, 9, 11])
        self.assertIn("above the fixed-point criterion", reasons[7])
        self.assertIn("predate settling", reasons[9])
        self.assertIn("cue_interp", reasons[11])

    def test_dmc_sweep_panels_follow_their_endpoint_panels(self):
        for seed in (3, 7, 9):
            (self.root / _aname(seed)).mkdir()
        components = {
            _aname(3): {"endpoint hidden": 1.0, "endpoint e_modulation": 0.0},
            _aname(7): {"endpoint hidden": 0.4, "endpoint e_modulation": 0.4},
            # Ties seed 3 everywhere; the lower seed wins.
            _aname(9): {"endpoint hidden": 1.0, "endpoint e_modulation": 0.0},
        }
        with patch.object(paper_plot, "_sibling_run_components",
                          side_effect=lambda family, aname: components[aname]):
            selected = paper_plot._select_sibling_runs("dmcgo")
        self.assertEqual(selected, {
            "endpoint hidden": _aname(3), "sweep hidden": _aname(3),
            "endpoint e_modulation": _aname(7), "sweep e_modulation": _aname(7)})
        self.assertEqual(self._report("dmcgo")["panels"]["sweep hidden"]["metric"],
                         "endpoint hidden")

    def test_dmc_accuracy_is_rescaled_so_chance_is_zero(self):
        self._write_run("dmcgo", _aname(5), _dmc_caches(self.rng))
        self.assertEqual(paper_plot._sibling_run_components("dmcgo", _aname(5)),
                         {"endpoint hidden": 1.0, "endpoint e_modulation": 1.0})

    def test_a_pinned_run_serves_every_panel_without_ranking(self):
        with patch.object(paper_plot, "DMCGO_ANAME", "pinned_run"), \
                patch.object(paper_plot, "_sibling_run_ranking") as ranking:
            for panel in paper_plot._SIBLING_PANELS:
                self.assertEqual(paper_plot._multitask_sibling_aname("dmcgo", panel),
                                 "pinned_run")
        ranking.assert_not_called()

    def test_without_a_scorable_run_every_panel_gets_the_seed_zero_name(self):
        for panel in paper_plot._SIBLING_PANELS:
            self.assertEqual(paper_plot._multitask_sibling_aname("dmcgo", panel),
                             _aname(0))
        self.assertEqual(self._report("dmcgo")["runs"], [])
        messages = [str(call.args[0]) for call in self.output.call_args_list]
        self.assertTrue(any("no scorable L21e4 run" in text for text in messages))


if __name__ == "__main__":
    unittest.main()
