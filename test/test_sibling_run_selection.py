"""Tests for the automatic per-figure choice of the sibling example runs."""

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


def _sweeps(rng, base_rules, fixed=True, separation=1.0, jump=0.0, drift=0.0):
    """Rule-cue sweeps of each base rule whose directions sit `separation`
    apart at every alpha; `fixed` as for `_delaydm_entry`. Every trial keeps
    its offset along the sweep, so none switches attractor unless `jump`
    moves trial 0 that far along PC6, off the directions' plane, at the last
    alpha. `drift` moves every trial that far along PC1 and PC2 over the
    sweep, which the separation ratio ignores but which squeezes the
    directions together in each drawn panel."""
    sweeps = {}
    stim = np.repeat(np.arange(N_DIRECTIONS), 2)
    for base_rule, target_rule in base_rules:
        offsets = rng.normal(scale=0.1, size=(stim.size, 6))
        proj = np.repeat(offsets[None], N_ALPHA, axis=0)
        proj[:, :, :2] += _ring(stim, separation)
        proj[:, :, :2] += drift * np.linspace(0.0, 1.0, N_ALPHA)[:, None, None]
        proj[-1, 0, 5] += jump
        sweep = {"proj": proj, "stim_idx": stim, "target_rule": target_rule,
                 "stimulus_magnitude": np.ones(stim.size)}
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


def _dmc_sweeps(rng, swap):
    """A DMCGo-based DMC sweep: PC1 keeps the response clusters 2 apart in the
    same order and PC2 swaps them by `swap` (from +swap/2 to -swap/2)."""
    stim = np.tile(np.arange(8), 2)
    labels = (stim < 4).astype(int)
    sign = np.where(labels == 1, 1.0, -1.0)
    alphas = np.linspace(0.0, 1.0, N_ALPHA)
    proj = rng.normal(scale=0.1, size=(N_ALPHA, stim.size, 6))
    proj[:, :, 0] += sign[None, :]
    proj[:, :, 1] += 0.5 * swap * (1.0 - 2.0 * alphas)[:, None] * sign[None, :]
    return {"dmcgo": {"proj": proj, "stim_idx": stim, "group_labels": labels,
                      "target_rule": "dmcnogo",
                      "is_fixed": np.ones((N_ALPHA, stim.size), dtype=bool)}}


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

    def _write_delaydm(self, seed, hidden, e_modulation, sweep=1.0,
                       endpoint_fixed=True, sweep_fixed=True, with_sweep=True,
                       jump=0.0, drift=0.0):
        entries = {"hidden": _delaydm_entry(self.rng, hidden, endpoint_fixed),
                   "e_modulation": _delaydm_entry(self.rng, e_modulation)}
        sweeps = (_sweeps(self.rng, DELAYDM_BASES, sweep_fixed, sweep, jump, drift)
                  if with_sweep else None)
        self._write_run("delaydm1", _aname(seed), entries, sweeps)

    def _report(self, name):
        return json.loads(
            (self.out / f"multitask_sibling_run_scores_{name}.json").read_text())

    def _selected(self, family):
        return {f"{figure} {rep}": paper_plot._multitask_sibling_aname(family, rep, figure)
                for figure in paper_plot._SIBLING_FIGURES for rep in REPRESENTATIONS}

    def test_each_figure_takes_its_own_best_run_from_one_ranking(self):
        # Seed 3 has the clearer hidden endpoints, seed 7 the clearer
        # e_modulation endpoints and the clearer sweeps.
        self._write_delaydm(3, hidden=3.0, e_modulation=0.5, sweep=0.5)
        self._write_delaydm(7, hidden=0.5, e_modulation=3.0, sweep=3.0)
        # A better run of another L2 cohort is never a candidate.
        self._write_run("delaydm1", _aname(1, "1e3"),
                        {rep: _delaydm_entry(self.rng, 9.0) for rep in REPRESENTATIONS},
                        _sweeps(self.rng, DELAYDM_BASES, separation=9.0))
        with patch.object(paper_plot, "_sibling_run_ranking",
                          wraps=paper_plot._sibling_run_ranking) as ranking:
            selected = self._selected("delaydm1")
        ranking.assert_called_once_with("delaydm1")
        self.assertEqual(selected, {
            "endpoint hidden": _aname(3), "endpoint e_modulation": _aname(7),
            "sweep hidden": _aname(7), "sweep e_modulation": _aname(7)})

        report = self._report("delaydm")
        self.assertEqual([row["seed"] for row in report["runs"]], [3, 7])
        self.assertEqual(sorted(report["runs"][0]["scores"]), ["endpoint", "sweep"])
        for key, seeds in (("endpoint hidden", [3, 7]), ("sweep hidden", [7, 3])):
            choice = report["choices"][key]
            self.assertEqual([row["seed"] for row in choice["ranking"]], seeds)
        self.assertIn("stimulus-direction separation", report["metrics"]["endpoint"])
        self.assertIn("minimum over alpha", report["metrics"]["sweep"])
        messages = [str(call.args[0]) for call in self.output.call_args_list]
        self.assertTrue(any("delaydm1 sweep hidden run selected automatically: seed 7"
                            in text for text in messages))

    def test_a_sweep_figure_weighs_separation_by_smooth_legible_lines(self):
        # Seed 3 separates its sweep directions better, but in each sweep
        # one trial jumps to another attractor at the last alpha; seed 7 is
        # smooth; seed 9 separates best by the ratio, but a common drift
        # squeezes its directions onto one line in every drawn panel.
        self._write_delaydm(3, hidden=3.0, e_modulation=3.0, sweep=3.0, jump=2.0)
        self._write_delaydm(7, hidden=0.5, e_modulation=0.5, sweep=1.0)
        self._write_delaydm(9, hidden=0.5, e_modulation=0.5, sweep=9.0, drift=1e4)
        # Every sweep trial drawn first, the jumping one included.
        with patch.object(paper_plot, "SIBLING_ONE_TRIAL_PER_STIMULUS", False):
            selected = self._selected("delaydm1")
        self.assertEqual(selected, {
            "endpoint hidden": _aname(3), "endpoint e_modulation": _aname(3),
            "sweep hidden": _aname(7), "sweep e_modulation": _aname(7)})
        report = self._report("delaydm")
        ranking = report["choices"]["sweep hidden"]["ranking"]
        self.assertEqual([row["seed"] for row in ranking], [7, 3, 9])
        by_seed = {row["seed"]: row for row in ranking}
        self.assertGreater(by_seed[9]["score"], by_seed[3]["score"])
        self.assertGreater(by_seed[3]["score"], by_seed[7]["score"])
        # Seed 7 keeps its score; seed 3's is cut by its jump.
        self.assertEqual(by_seed[7]["ranked_score"], by_seed[7]["score"])
        self.assertAlmostEqual(
            by_seed[3]["ranked_score"],
            by_seed[3]["score"] * 0.1 / by_seed[3]["largest_attractor_switch"])
        self.assertLess(by_seed[9]["direction_spread"],
                        paper_plot._SIBLING_MIN_DIRECTION_SPREAD)
        self.assertGreater(by_seed[7]["direction_spread"], 0.5)
        self.assertNotIn("ranked_score",
                         report["choices"]["endpoint hidden"]["ranking"][0])
        self.assertIn("largest drawn attractor switch", report["metrics"]["sweep"])
        # One trial per direction leaves the jumping trial out, so seed 3
        # keeps its score and wins; seed 9 still draws one line.
        paper_plot._SIBLING_SELECTED_RUNS.clear()
        with patch.object(paper_plot, "SIBLING_ONE_TRIAL_PER_STIMULUS", True):
            self.assertEqual(
                paper_plot._multitask_sibling_aname("delaydm1", "hidden", "sweep"),
                _aname(3))
        ranking = self._report("delaydm")["choices"]["sweep hidden"]["ranking"]
        self.assertEqual([row["seed"] for row in ranking], [3, 7, 9])

    def test_runs_that_are_unconverged_unsettled_or_incomplete_are_skipped(self):
        self._write_delaydm(3, hidden=0.5, e_modulation=0.5)
        self._write_delaydm(7, hidden=3.0, e_modulation=3.0, endpoint_fixed=False)
        self._write_delaydm(9, hidden=3.0, e_modulation=3.0, sweep_fixed=None)
        self._write_delaydm(11, hidden=3.0, e_modulation=3.0, with_sweep=False)
        self.assertEqual(set(self._selected("delaydm1").values()), {_aname(3)})

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
            _aname(3): {"endpoint": {"hidden": 1.0, "e_modulation": 0.2},
                        "sweep": {"hidden": 0.3, "e_modulation": 0.9}},
            _aname(7): {"endpoint": {"hidden": 0.4, "e_modulation": 0.4},
                        "sweep": {"hidden": 0.5, "e_modulation": 0.1}},
            # Ties seed 3 everywhere; the lower seed wins.
            _aname(9): {"endpoint": {"hidden": 1.0, "e_modulation": 0.2},
                        "sweep": {"hidden": 0.3, "e_modulation": 0.9}},
        }
        legibility = {"hidden": None, "e_modulation": None}
        with patch.object(paper_plot, "_sibling_run_scores",
                          side_effect=lambda family, aname: (scores[aname], legibility)):
            selected = paper_plot._select_sibling_runs("dmcgo")
        self.assertEqual(selected, {
            "endpoint hidden": _aname(3), "endpoint e_modulation": _aname(7),
            "sweep hidden": _aname(7), "sweep e_modulation": _aname(3)})

    def test_dmc_runs_are_ranked_on_their_own_endpoints_and_sweeps(self):
        # Seed 3: endpoints split by the seen stimulus (the wrong grouping)
        # but a sweep whose clusters swap widely; seed 7: endpoints split by
        # the response category but a sweep that barely swaps.
        for seed, by_response, swap in ((3, False, 2.0), (7, True, 0.2)):
            self._write_run(
                "dmcgo", _aname(seed),
                {rep: _dmc_entry(self.rng, by_response) for rep in REPRESENTATIONS},
                _dmc_sweeps(self.rng, swap))
        selected = self._selected("dmcgo")
        self.assertEqual({key: aname for key, aname in selected.items()
                          if key.startswith("endpoint")},
                         {"endpoint hidden": _aname(7),
                          "endpoint e_modulation": _aname(7)})
        self.assertEqual({key: aname for key, aname in selected.items()
                          if key.startswith("sweep")},
                         {"sweep hidden": _aname(3), "sweep e_modulation": _aname(3)})
        report = self._report("dmcgo")
        ranking = report["choices"]["endpoint hidden"]["ranking"]
        self.assertGreater(ranking[0]["score"], 5 * ranking[1]["score"])
        self.assertIn("kept and a swapped PC", report["metrics"]["sweep"])
        # DMC sweeps are not scored for attractor switches: every trial
        # takes the response-cluster swap.
        self.assertNotIn("attractor switch", report["metrics"]["sweep"])
        self.assertNotIn("ranked_score",
                         report["choices"]["sweep hidden"]["ranking"][0])

    def test_a_pinned_run_serves_every_figure_without_ranking(self):
        with patch.object(paper_plot, "DMCGO_ANAME", "pinned_run"), \
                patch.object(paper_plot, "_sibling_run_ranking") as ranking:
            self.assertEqual(set(self._selected("dmcgo").values()), {"pinned_run"})
        ranking.assert_not_called()

    def test_without_a_scorable_run_every_figure_gets_seed_zero(self):
        self.assertEqual(set(self._selected("dmcgo").values()), {_aname(0)})
        self.assertEqual(self._report("dmcgo")["runs"], [])
        messages = [str(call.args[0]) for call in self.output.call_args_list]
        self.assertTrue(any("no scorable L21e4 run" in text for text in messages))


if __name__ == "__main__":
    unittest.main()
