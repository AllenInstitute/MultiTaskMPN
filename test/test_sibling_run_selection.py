"""Tests for the automatic per-representation choice of the sibling example runs."""

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


def _delaydm_entry(rng, separation, fixed=True):
    """DelayDM endpoints of both tasks whose directions sit `separation`
    apart; `fixed` False marks one endpoint unconverged and None drops the
    convergence mask (a cache written before settling)."""
    stim = np.tile(np.repeat(np.arange(N_DIRECTIONS), 2), 2)
    proj = rng.normal(scale=0.1, size=(stim.size, 6))
    proj[:, :2] += _ring(stim, separation)
    entry = {"proj": proj, "stim_idx": stim,
             "task_idx": np.repeat([0, 1], stim.size // 2)}
    if fixed is not None:
        entry["is_fixed"] = np.ones(stim.size, dtype=bool)
        entry["is_fixed"][0] = fixed
    return entry


def _sweeps(rng, base_rules, fixed=True):
    """Rule-cue sweeps of each base rule (unscored, but their cache and
    convergence decide whether a run can be shown)."""
    sweeps = {}
    for base_rule, target_rule in base_rules:
        sweep = {"proj": rng.normal(size=(N_ALPHA, 8, 6)),
                 "stim_idx": np.repeat(np.arange(N_DIRECTIONS), 2),
                 "target_rule": target_rule}
        if fixed is not None:
            sweep["is_fixed"] = np.full((N_ALPHA, 8), bool(fixed))
        sweeps[base_rule] = sweep
    return sweeps


DELAYDM_BASES = (("delaydm1", "delaydm2"), ("delaydm2", "delaydm1"))


def _dmc_entry(rng, separated_by_response):
    """DMC endpoints of both tasks with a 4-unit task gap on PC1 and a
    2-unit split on PC2 by the response category, or (False) by the seen
    stimulus category, which in DMCNoGo is the opposite response category."""
    task = np.repeat([0, 1], 8)
    stim = np.tile(np.arange(8), 2)
    response = ((task == 0) == (stim < 4)).astype(int)
    split = response if separated_by_response else (stim < 4).astype(int)
    proj = rng.normal(scale=0.1, size=(task.size, 6))
    proj[:, 0] += 4.0 * task
    proj[:, 1] += 2.0 * split
    return {"proj": proj, "stim_idx": stim, "task_idx": task,
            "group_labels": response, "is_fixed": np.ones(task.size, dtype=bool)}


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

    def _write_run(self, family, aname, entries, sweeps=None):
        """Cache one run: `entries` maps each representation to its endpoint
        projection; `sweeps` (None: no sweep cache) serve both."""
        run_dir = self.root / aname
        run_dir.mkdir()
        endpoint_path = (run_dir
                         / f"{family}_long_delay_endpoint_pc_projections_{aname}.pkl")
        with endpoint_path.open("wb") as stream:
            pickle.dump({"representations": entries}, stream)
        if sweeps is None:
            return
        sweep_path = paper_plot._multitask_cue_interp_projection_path(
            run_dir, family, aname, "long_delay_endpoint")
        with sweep_path.open("wb") as stream:
            pickle.dump({"kind": "cue_interp",
                         "alphas": np.linspace(0.0, 1.0, N_ALPHA),
                         "representations": {rep: {"sweeps": sweeps}
                                             for rep in REPRESENTATIONS}},
                        stream)

    def _write_delaydm(self, seed, hidden, e_modulation, endpoint_fixed=True,
                       sweep_fixed=True, with_sweep=True):
        entries = {"hidden": _delaydm_entry(self.rng, hidden, endpoint_fixed),
                   "e_modulation": _delaydm_entry(self.rng, e_modulation)}
        sweeps = _sweeps(self.rng, DELAYDM_BASES, sweep_fixed) if with_sweep else None
        self._write_run("delaydm1", _aname(seed), entries, sweeps)

    def _report(self, name):
        return json.loads(
            (self.out / f"multitask_sibling_run_scores_{name}.json").read_text())

    def test_each_representation_takes_its_own_best_run_from_one_ranking(self):
        self._write_delaydm(3, hidden=3.0, e_modulation=0.5)
        self._write_delaydm(7, hidden=0.5, e_modulation=3.0)
        # A better run of another L2 cohort is never a candidate.
        self._write_run("delaydm1", _aname(1, "1e3"),
                        {rep: _delaydm_entry(self.rng, 9.0) for rep in REPRESENTATIONS},
                        _sweeps(self.rng, DELAYDM_BASES))
        with patch.object(paper_plot, "_sibling_run_ranking",
                          wraps=paper_plot._sibling_run_ranking) as ranking:
            selected = {rep: paper_plot._multitask_sibling_aname("delaydm1", rep)
                        for rep in REPRESENTATIONS}
        ranking.assert_called_once_with("delaydm1")
        self.assertEqual(selected, {"hidden": _aname(3), "e_modulation": _aname(7)})

        report = self._report("delaydm")
        self.assertEqual([row["seed"] for row in report["runs"]], [3, 7])
        self.assertEqual(sorted(report["runs"][0]["scores"]), sorted(REPRESENTATIONS))
        self.assertEqual(
            [row["seed"] for row in report["representations"]["hidden"]["ranking"]],
            [3, 7])
        self.assertIn("stimulus-direction separation", report["criterion"])
        messages = [str(call.args[0]) for call in self.output.call_args_list]
        self.assertTrue(any("delaydm1 e_modulation run selected automatically: seed 7"
                            in text for text in messages))

    def test_runs_that_are_unconverged_unsettled_or_incomplete_are_skipped(self):
        self._write_delaydm(3, hidden=0.5, e_modulation=0.5)
        self._write_delaydm(7, hidden=3.0, e_modulation=3.0, endpoint_fixed=False)
        self._write_delaydm(9, hidden=3.0, e_modulation=3.0, sweep_fixed=None)
        self._write_delaydm(11, hidden=3.0, e_modulation=3.0, with_sweep=False)
        for rep in REPRESENTATIONS:
            self.assertEqual(paper_plot._multitask_sibling_aname("delaydm1", rep),
                             _aname(3))

        reasons = {row["seed"]: row["reason"]
                   for row in self._report("delaydm")["skipped"]}
        self.assertEqual(sorted(reasons), [7, 9, 11])
        self.assertIn("above the fixed-point criterion", reasons[7])
        self.assertIn("predate settling", reasons[9])
        self.assertIn("cue_interp", reasons[11])

    def test_ties_go_to_the_lowest_seed(self):
        for seed in (3, 7, 9):
            (self.root / _aname(seed)).mkdir()
        scores = {
            _aname(3): {"hidden": 1.0, "e_modulation": 0.2},
            _aname(7): {"hidden": 0.4, "e_modulation": 0.4},
            _aname(9): {"hidden": 1.0, "e_modulation": 0.2},
        }
        with patch.object(paper_plot, "_sibling_run_scores",
                          side_effect=lambda family, aname: scores[aname]):
            selected = paper_plot._select_sibling_runs("dmcgo")
        self.assertEqual(selected, {"hidden": _aname(3), "e_modulation": _aname(7)})

    def test_dmc_runs_are_ranked_by_response_category_separation(self):
        dmc_bases = (("dmcgo", "dmcnogo"),)
        for seed, by_response in ((3, False), (7, True)):
            self._write_run(
                "dmcgo", _aname(seed),
                {rep: _dmc_entry(self.rng, by_response) for rep in REPRESENTATIONS},
                _sweeps(self.rng, dmc_bases))
        for rep in REPRESENTATIONS:
            self.assertEqual(paper_plot._multitask_sibling_aname("dmcgo", rep),
                             _aname(7))
        ranking = self._report("dmcgo")["representations"]["hidden"]["ranking"]
        self.assertGreater(ranking[0]["score"], 5 * ranking[1]["score"])

    def test_a_pinned_run_serves_every_representation_without_ranking(self):
        with patch.object(paper_plot, "DMCGO_ANAME", "pinned_run"), \
                patch.object(paper_plot, "_sibling_run_ranking") as ranking:
            for rep in REPRESENTATIONS:
                self.assertEqual(paper_plot._multitask_sibling_aname("dmcgo", rep),
                                 "pinned_run")
        ranking.assert_not_called()

    def test_without_a_scorable_run_every_representation_gets_seed_zero(self):
        for rep in REPRESENTATIONS:
            self.assertEqual(paper_plot._multitask_sibling_aname("dmcgo", rep),
                             _aname(0))
        self.assertEqual(self._report("dmcgo")["runs"], [])
        messages = [str(call.args[0]) for call in self.output.call_args_list]
        self.assertTrue(any("no scorable L21e4 run" in text for text in messages))


if __name__ == "__main__":
    unittest.main()
