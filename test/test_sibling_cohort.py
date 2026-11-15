"""Tests of the sibling-task cohort statistics: the per-run metrics on
constructed fixed points, the script over a cache tree, and the five paper
figures that draw the saved results."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np

import paper_plot
import sibling_cohort as cohort


DIM = 6
N_ALPHA = 3
DIRECTIONS = 4
MAGNITUDES = (0.6, 1.0)
FEATURE = "L21e4"
SUFFIX = "+hidden300+batch128+angle"
REPRESENTATIONS = ("hidden", "e_modulation")


def _aname(seed, feature=FEATURE):
    return f"everything_seed{seed}_{feature}{SUFFIX}"


def _unit(index):
    vector = np.zeros(DIM)
    vector[index] = 1.0
    return vector


def _as_cache_states(points):
    """The two representations of one state array as the caches store them:
    the hidden state flat, W * M as a (..., 2, 3) matrix per point."""
    points = np.asarray(points, dtype=float)
    return {"fixed_hidden": points, "fixed_WM": points.reshape(*points.shape[:-1], 2, 3)}


def _delaydm_caches(rng, translation, end="sibling", fixed=True, noise=0.02):
    """Endpoint and rule-cue sweep caches of one DelayDM run: task 0 stores
    direction d at magnitude m on a ring in the first two dimensions, task 1
    the same point plus `translation`. Each base rule's sweep runs from its
    own endpoints to the sibling's matched endpoints (`end` "sibling"), stays
    on its own ("own") or stops halfway ("half"). `fixed` False marks one
    endpoint unconverged, None leaves the convergence mask out."""
    stim = np.repeat(np.arange(DIRECTIONS), len(MAGNITUDES))
    magnitude = np.tile(MAGNITUDES, DIRECTIONS)
    angles = 2 * np.pi * stim / DIRECTIONS
    base = rng.normal(scale=noise, size=(stim.size, DIM))
    base[:, 0] += magnitude * np.cos(angles)
    base[:, 1] += magnitude * np.sin(angles)
    points = np.concatenate([base, base + translation])
    task = np.repeat([0, 1], stim.size)
    endpoint = {"task_idx": task, "stim_idx": np.concatenate([stim, stim]),
                "stimulus_magnitude": np.concatenate([magnitude, magnitude]),
                "representations": _as_cache_states(points)}
    if fixed is not None:
        endpoint["is_fixed"] = np.ones(points.shape[0], dtype=bool)
        endpoint["is_fixed"][0] = fixed
    alphas = np.linspace(0.0, 1.0, N_ALPHA)
    sweeps = {}
    for base_rule, base_task in (("delaydm1", 0), ("delaydm2", 1)):
        own, sibling = points[task == base_task], points[task == 1 - base_task]
        target = {"sibling": sibling, "own": own, "half": (own + sibling) / 2}[end]
        path = own[None] + alphas[:, None, None] * (target - own)[None]
        sweeps[base_rule] = {"target_rule": "delaydm2" if base_task == 0 else "delaydm1",
                             "stim_idx": stim, "stimulus_magnitude": magnitude,
                             "is_fixed": np.ones((N_ALPHA, stim.size), dtype=bool),
                             "representations": _as_cache_states(path)}
    return endpoint, {"kind": "cue_interp", "alphas": alphas, "sweeps": sweeps}


def _dmc_caches(rng, code="orthogonal", end="sibling", noise=0.02):
    """Endpoint and sweep caches of one DMC run, eight stimuli per task: task
    0 (DMCGo) codes the stimulus category along the first axis, task 1
    (DMCNoGo) along the same axis (`code` "same"), its reverse ("reversed")
    or the second axis ("orthogonal"), four units away along the third. The
    DMCGo sweep ends on the DMCNoGo centroid of each trial's category (`end`
    "sibling") or stays on its own endpoint ("own")."""
    stim = np.tile(np.arange(8), 2)
    task = np.repeat([0, 1], 8)
    category = (stim < 4).astype(int)
    axis = {"same": _unit(0), "reversed": -_unit(0), "orthogonal": _unit(1)}[code]
    points = rng.normal(scale=noise, size=(stim.size, DIM))
    points[task == 0] += (category[task == 0] - 0.5)[:, None] * _unit(0)
    points[task == 1] += (category[task == 1] - 0.5)[:, None] * axis + 4.0 * _unit(2)
    endpoint = {"task_idx": task, "stim_idx": stim, "stimulus_magnitude": np.ones(stim.size),
                "group_labels": np.where(task == 0, category, 1 - category),
                "is_fixed": np.ones(stim.size, dtype=bool),
                "representations": _as_cache_states(points)}
    own = points[task == 0]
    centroids = {label: points[(task == 1) & (category == label)].mean(axis=0) for label in (0, 1)}
    target = (np.stack([centroids[label] for label in category[task == 0]])
              if end == "sibling" else own)
    alphas = np.linspace(0.0, 1.0, N_ALPHA)
    path = own[None] + alphas[:, None, None] * (target - own)[None]
    sweeps = {"dmcgo": {"target_rule": "dmcnogo", "stim_idx": stim[task == 0],
                        "stimulus_magnitude": np.ones(8),
                        "is_fixed": np.ones((N_ALPHA, 8), dtype=bool),
                        "representations": _as_cache_states(path)}}
    return endpoint, {"kind": "cue_interp", "alphas": alphas, "sweeps": sweeps}


def _memory_caches(rng, code="same", end="sibling", repeats=2, noise=0.0, fixed=True):
    """Endpoint and sweep caches of one MemoryPro / MemoryAnti run, eight
    stimulus directions `repeats` times per task. MemoryPro holds direction k
    on a unit ring in the first two dimensions. MemoryAnti holds it three
    units away along the fourth, on the same ring (`code` "same"), on the
    ring turned by pi, i.e. at the place of the opposite direction
    ("opposite"), or on a ring in the last two dimensions ("orthogonal").
    The MemoryPro sweep ends on the MemoryAnti endpoint of each trial's
    stimulus (`end` "sibling") or stays on its own ("own")."""
    stim = np.tile(np.arange(8), repeats)
    angles = 2 * np.pi * stim / 8
    pro = rng.normal(scale=noise, size=(stim.size, DIM)) if noise else np.zeros((stim.size, DIM))
    pro[:, 0] += np.cos(angles)
    pro[:, 1] += np.sin(angles)
    anti = np.zeros((stim.size, DIM))
    turn = np.pi if code == "opposite" else 0.0
    axes = (4, 5) if code == "orthogonal" else (0, 1)
    anti[:, axes[0]] = np.cos(angles + turn)
    anti[:, axes[1]] = np.sin(angles + turn)
    anti += 3.0 * _unit(3)
    points = np.concatenate([pro, anti])
    task = np.repeat([0, 1], stim.size)
    endpoint = {"task_idx": task, "stim_idx": np.concatenate([stim, stim]),
                "stimulus_magnitude": np.ones(2 * stim.size),
                "is_fixed": np.ones(2 * stim.size, dtype=bool),
                "representations": _as_cache_states(points)}
    endpoint["is_fixed"][0] = fixed
    target = anti if end == "sibling" else pro
    alphas = np.linspace(0.0, 1.0, N_ALPHA)
    path = pro[None] + alphas[:, None, None] * (target - pro)[None]
    sweeps = {"delaygo": {"target_rule": "delayanti", "stim_idx": stim,
                          "stimulus_magnitude": np.ones(stim.size),
                          "is_fixed": np.ones((N_ALPHA, stim.size), dtype=bool),
                          "representations": _as_cache_states(path)}}
    return endpoint, {"kind": "cue_interp", "alphas": alphas, "sweeps": sweeps}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def _write_run(root, aname, delaydm=None, dmc=None, memory=None):
    run_dir = root / aname
    run_dir.mkdir(parents=True, exist_ok=True)
    if memory is not None:
        endpoint, sweep = memory
        _write(cohort.endpoint_path(run_dir, "delaygo", aname), endpoint)
        if sweep is not None:
            _write(cohort.sweep_path(run_dir, "delaygo", aname), sweep)
    if delaydm is not None:
        endpoint, sweep = delaydm
        _write(cohort.endpoint_path(run_dir, "delaydm1", aname), endpoint)
        if sweep is not None:
            _write(cohort.sweep_path(run_dir, "delaydm1", aname), sweep)
    if dmc is not None:
        endpoint, sweep = dmc
        _write(cohort.endpoint_path(run_dir, "dmcgo", aname), endpoint)
        if sweep is not None:
            _write(cohort.sweep_path(run_dir, "dmcgo", aname), sweep)


class DelayDMStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.translation = 3.0 * _unit(3) + 1.0 * _unit(4)

    def test_a_pure_translation_is_explained_and_the_cue_carries_it(self):
        endpoint, sweep = _delaydm_caches(self.rng, self.translation)
        for rep_key in cohort.REPRESENTATIONS.values():
            stats = cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"], rep_key)
            self.assertGreater(stats["translation_explained"], 0.99)
            self.assertAlmostEqual(stats["translation_length"], np.sqrt(10.0), places=1)
            self.assertGreater(stats["geometry_r"], 0.99)
            self.assertGreater(stats["alignment_gain"], 1.5)
            self.assertAlmostEqual(stats["separation_aligned"] / stats["separation_raw"],
                                   stats["alignment_gain"])
            self.assertAlmostEqual(stats["cue_traversed"], 1.0, places=2)
            self.assertLess(stats["cue_end_residual"], 0.02)
            self.assertEqual(stats["cue_end_nearer_sibling"], 1.0)
            self.assertEqual(stats["n_conditions"], DIRECTIONS * len(MAGNITUDES))
            self.assertEqual(stats["n_sweep_trials"], 2 * DIRECTIONS * len(MAGNITUDES))

    def test_a_sweep_that_stays_traverses_nothing(self):
        endpoint, sweep = _delaydm_caches(self.rng, self.translation, end="own")
        stats = cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"],
                                          "fixed_hidden")
        self.assertAlmostEqual(stats["cue_traversed"], 0.0, places=2)
        self.assertAlmostEqual(stats["cue_end_residual"], 1.0, places=1)
        self.assertEqual(stats["cue_end_nearer_sibling"], 0.0)
        endpoint, sweep = _delaydm_caches(self.rng, self.translation, end="half")
        stats = cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"],
                                          "fixed_hidden")
        self.assertAlmostEqual(stats["cue_traversed"], 0.5, places=2)
        self.assertAlmostEqual(stats["cue_end_residual"], 0.5, places=1)

    def test_a_direction_dependent_displacement_is_not_a_translation(self):
        endpoint, sweep = _delaydm_caches(self.rng, self.translation)
        points = endpoint["representations"]["fixed_hidden"].copy()
        task, stim = endpoint["task_idx"], endpoint["stim_idx"]
        # Task 1 mirrors the ring instead of shifting it: the mean displacement
        # cancels over the directions.
        points[task == 1] = points[task == 0].copy()
        points[task == 1, 0] *= -1
        points[task == 1, 1] *= -1
        endpoint["representations"] = _as_cache_states(points)
        stats = cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"],
                                          "fixed_hidden")
        self.assertLess(stats["translation_explained"], 0.05)
        self.assertGreater(stats["geometry_r"], 0.99)
        del stim

    def test_unsettled_or_mismatched_caches_are_errors(self):
        endpoint, sweep = _delaydm_caches(self.rng, self.translation, fixed=False)
        with self.assertRaisesRegex(ValueError, "fixed-point criterion"):
            cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"], "fixed_hidden")
        endpoint, sweep = _delaydm_caches(self.rng, self.translation, fixed=None)
        with self.assertRaisesRegex(ValueError, "predate settling"):
            cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"], "fixed_hidden")
        endpoint, sweep = _delaydm_caches(self.rng, self.translation)
        endpoint["stimulus_magnitude"][0] = 0.7     # task 0 lacks a condition of task 1
        with self.assertRaisesRegex(ValueError, "conditions differ"):
            cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"], "fixed_hidden")
        endpoint, sweep = _delaydm_caches(self.rng, self.translation)
        sweep["sweeps"]["delaydm1"]["stimulus_magnitude"][0] = 0.7
        with self.assertRaisesRegex(ValueError, "no matched endpoint"):
            cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"], "fixed_hidden")
        endpoint, sweep = _delaydm_caches(self.rng, self.translation)
        sweep["sweeps"]["delaydm2"]["is_fixed"][-1, 0] = False
        with self.assertRaisesRegex(ValueError, "sweep from delaydm2"):
            cohort.delaydm_statistics(endpoint, sweep, cohort.FAMILIES["delaydm1"], "fixed_hidden")


class DMCStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_cosine_tells_a_shared_reversed_and_orthogonal_category_code_apart(self):
        for code, expected in (("same", 1.0), ("reversed", -1.0), ("orthogonal", 0.0)):
            endpoint, sweep = _dmc_caches(self.rng, code)
            for rep_key in cohort.REPRESENTATIONS.values():
                stats = cohort.dmc_statistics(endpoint, sweep, cohort.FAMILIES["dmcgo"], rep_key)
                self.assertAlmostEqual(stats["category_axis_cosine"], expected, delta=0.06)
                self.assertGreater(stats["separation_task"], 5.0)
                self.assertLess(stats["cue_end_residual"], 0.02)
                self.assertEqual(stats["cue_end_nearer_same"], 1.0)
                self.assertEqual(stats["n_endpoints"], 16)
                self.assertEqual(stats["n_sweep_trials"], 8)
                # Both rules hold the category: far apart, the best of every
                # balanced split, perfectly decodable.
                for rule in cohort.FAMILIES["dmcgo"]:
                    self.assertGreater(stats[f"category_separation_{rule}"], 5.0)
                    self.assertLess(stats[f"category_best_other_split_{rule}"],
                                    stats[f"category_separation_{rule}"])
                    self.assertTrue(stats[f"category_is_best_split_{rule}"])
                    self.assertEqual(stats[f"category_nearest_centroid_accuracy_{rule}"], 1.0)
            # With one shared code the response labels group the tasks'
            # categories crosswise, so the stimulus grouping wins; reversed,
            # the response grouping wins.
            if code == "same":
                self.assertGreater(stats["separation_stimulus_aligned"],
                                   stats["separation_response_aligned"])
            elif code == "reversed":
                self.assertGreater(stats["separation_response_aligned"],
                                   stats["separation_stimulus_aligned"])

    def test_a_sweep_that_stays_is_a_task_gap_away(self):
        endpoint, sweep = _dmc_caches(self.rng, "same", end="own")
        stats = cohort.dmc_statistics(endpoint, sweep, cohort.FAMILIES["dmcgo"], "fixed_hidden")
        self.assertAlmostEqual(stats["cue_end_residual"], 1.0, places=1)

    def test_a_rule_grouping_the_stimuli_by_another_split_holds_no_category(self):
        endpoint, sweep = _dmc_caches(self.rng, "same")
        points = endpoint["representations"]["fixed_hidden"].copy()
        task, stim = endpoint["task_idx"], endpoint["stim_idx"]
        # DMCGo's endpoints split the ring as {0, 1, 4, 5} against the rest,
        # so each category holds two of each group.
        other = np.isin(stim[task == 0], [0, 1, 4, 5]).astype(float)
        points[task == 0] = self.rng.normal(scale=0.02, size=(8, DIM))
        points[task == 0, 0] += other - 0.5
        endpoint["representations"] = _as_cache_states(points)
        stats = cohort.dmc_statistics(endpoint, sweep, cohort.FAMILIES["dmcgo"], "fixed_hidden")
        self.assertLess(stats["category_separation_dmcgo"], 0.5)
        self.assertGreater(stats["category_best_other_split_dmcgo"], 5.0)
        self.assertFalse(stats["category_is_best_split_dmcgo"])
        self.assertLess(stats["category_nearest_centroid_accuracy_dmcgo"], 0.8)
        # DMCNoGo is untouched.
        self.assertGreater(stats["category_separation_dmcnogo"], 5.0)
        self.assertTrue(stats["category_is_best_split_dmcnogo"])
        self.assertEqual(stats["category_nearest_centroid_accuracy_dmcnogo"], 1.0)

    def test_a_ring_code_does_not_single_out_the_category_split(self):
        endpoint, sweep = _dmc_caches(self.rng, "same")
        points = endpoint["representations"]["fixed_hidden"].copy()
        task, stim = endpoint["task_idx"], endpoint["stim_idx"]
        # DMCGo stores the stimulus identity on a ring: the category split is
        # one of four contiguous halves and separates no better than they do.
        angles = 2 * np.pi * (stim[task == 0] + 0.5) / 8
        points[task == 0] = self.rng.normal(scale=0.01, size=(8, DIM))
        points[task == 0, 0] += np.cos(angles)
        points[task == 0, 1] += np.sin(angles)
        endpoint["representations"] = _as_cache_states(points)
        stats = cohort.dmc_statistics(endpoint, sweep, cohort.FAMILIES["dmcgo"], "fixed_hidden")
        self.assertAlmostEqual(stats["category_best_other_split_dmcgo"]
                               / stats["category_separation_dmcgo"], 1.0, delta=0.1)
        self.assertEqual(stats["category_nearest_centroid_accuracy_dmcgo"], 1.0)

    def test_balanced_splits_and_the_accuracy_helper(self):
        splits = cohort.balanced_splits(8)
        self.assertEqual(len(splits), 35)
        self.assertEqual(splits[0], frozenset(range(4)))
        self.assertEqual(len(set(splits)), 35)
        self.assertTrue(all(0 in split and len(split) == 4 for split in splits))
        points = np.array([[0.0], [0.1], [1.0], [1.1]])
        self.assertEqual(cohort.nearest_centroid_accuracy(points, [0, 0, 1, 1]), 1.0)
        self.assertEqual(cohort.nearest_centroid_accuracy(points, [0, 1, 0, 1]), 0.0)
        with self.assertRaisesRegex(ValueError, "two classes"):
            cohort.nearest_centroid_accuracy(points, [0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "two points"):
            cohort.nearest_centroid_accuracy(points, [0, 1, 1, 1])

    def test_response_labels_must_be_the_documented_ones(self):
        endpoint, sweep = _dmc_caches(self.rng, "same")
        endpoint["group_labels"] = (endpoint["stim_idx"] < 4).astype(int)   # stimulus, not response
        with self.assertRaisesRegex(ValueError, "response categories"):
            cohort.dmc_statistics(endpoint, sweep, cohort.FAMILIES["dmcgo"], "fixed_hidden")
        self.assertEqual(cohort.dmc_category(np.arange(8)).tolist(), [1, 1, 1, 1, 0, 0, 0, 0])


class ProAntiStatisticsTests(unittest.TestCase):
    """MemoryPro / MemoryAnti, settled in the stimulus period."""
    RULES = ("delaygo", "delayanti")

    def setUp(self):
        self.rng = np.random.default_rng(4)

    def statistics(self, rep_key="fixed_hidden", **kwargs):
        return cohort.pro_anti_statistics(*_memory_caches(self.rng, **kwargs), self.RULES, rep_key)

    def test_ring_cosine_tells_stimulus_from_response_coding(self):
        expected = {"same": (1.0, -1.0), "opposite": (-1.0, 1.0), "orthogonal": (0.0, 0.0)}
        for code, (same, opposite) in expected.items():
            for rep_key in cohort.REPRESENTATIONS.values():
                with self.subTest(code=code, rep_key=rep_key):
                    stats = self.statistics(rep_key, code=code)
                    self.assertAlmostEqual(stats["ring_cosine_same"], same, places=9)
                    self.assertAlmostEqual(stats["ring_cosine_opposite"], opposite, places=9)
                    # The shift between the rules is three units for a unit ring.
                    self.assertAlmostEqual(stats["translation_length"], 3.0)
                    self.assertAlmostEqual(stats["ring_radius"], 1.0)
                    self.assertAlmostEqual(stats["translation_to_ring"], 3.0)
                    self.assertEqual(stats["n_conditions"], 8)
                    self.assertEqual(stats["n_endpoints"], 32)

    def test_one_translation_explains_the_matching_that_holds(self):
        same = self.statistics(code="same")
        self.assertAlmostEqual(same["translation_explained_same"], 1.0)
        # Matched by response, the same ring is turned by pi: the displacement
        # is the shift (9) plus a diameter squared (4) on average.
        self.assertAlmostEqual(same["translation_explained_opposite"], 9 / 13)
        self.assertAlmostEqual(same["geometry_r"], 1.0)
        opposite = self.statistics(code="opposite")
        self.assertAlmostEqual(opposite["translation_explained_opposite"], 1.0)
        self.assertAlmostEqual(opposite["translation_explained_same"], 9 / 13)
        # Removing the rule means aligns the directions under the matching
        # that holds, and only under that one.
        noisy = self.statistics(code="same", noise=0.02)
        self.assertGreater(noisy["alignment_gain_same"], 5.0)
        self.assertGreater(noisy["separation_aligned_same"],
                           20 * noisy["separation_aligned_opposite"])
        turned = self.statistics(code="opposite", noise=0.02)
        self.assertGreater(turned["alignment_gain_opposite"], 5.0)
        self.assertGreater(turned["separation_aligned_opposite"],
                           20 * turned["separation_aligned_same"])
        # Exact opposite points cancel: no raw separation, so no gain.
        self.assertTrue(np.isnan(same["alignment_gain_opposite"]))
        self.assertEqual(same["separation_aligned_opposite"], 0.0)

    def test_cue_sweep_is_read_against_the_sibling_endpoint_of_the_stimulus(self):
        landed = self.statistics(code="opposite", end="sibling")
        self.assertAlmostEqual(landed["cue_traversed"], 1.0)
        self.assertAlmostEqual(landed["cue_end_residual"], 0.0)
        self.assertEqual(landed["cue_end_nearer_sibling"], 1.0)
        self.assertEqual(landed["n_sweep_trials"], 16)
        stayed = self.statistics(code="opposite", end="own")
        self.assertAlmostEqual(stayed["cue_traversed"], 0.0)
        self.assertEqual(stayed["cue_end_nearer_sibling"], 0.0)
        self.assertGreater(stayed["cue_end_residual"], 1.0)

    def test_incomplete_or_unsettled_caches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed-point criterion"):
            self.statistics(fixed=False)
        endpoint, sweep = _memory_caches(self.rng)
        endpoint["stim_idx"] = np.where(endpoint["stim_idx"] == 7, 6, endpoint["stim_idx"])
        with self.assertRaisesRegex(ValueError, "lacks a stimulus direction"):
            cohort.pro_anti_statistics(endpoint, sweep, self.RULES, "fixed_hidden")
        endpoint, sweep = _memory_caches(self.rng)
        sweep["sweeps"]["delaygo"]["is_fixed"][1, 0] = False
        with self.assertRaisesRegex(ValueError, "sweep from delaygo"):
            cohort.pro_anti_statistics(endpoint, sweep, self.RULES, "fixed_hidden")
        endpoint, _ = _memory_caches(self.rng)
        with self.assertRaisesRegex(ValueError, "no rule-cue sweep"):
            cohort.pro_anti_statistics(endpoint, {"sweeps": {}}, self.RULES, "fixed_hidden")


class ScriptTests(unittest.TestCase):
    def test_optional_family_enters_only_the_runs_that_have_its_caches(self):
        rng = np.random.default_rng(5)
        translation = 2.0 * _unit(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "two_in_multiples"
            _write_run(root, _aname(1), _delaydm_caches(rng, translation),
                       _dmc_caches(rng, "orthogonal"), _memory_caches(rng, "opposite"))
            _write_run(root, _aname(2), _delaydm_caches(rng, translation),
                       _dmc_caches(rng, "orthogonal"))                       # not run yet
            _write_run(root, _aname(3), _delaydm_caches(rng, translation),
                       _dmc_caches(rng, "orthogonal"), _memory_caches(rng, fixed=False))
            with patch("builtins.print") as output:
                results = cohort.run_all(root, FEATURE, out_dir=root / "out")
            printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        # Every run keeps its DelayDM and DMC statistics; none is skipped.
        self.assertEqual(list(results["runs"]), [_aname(1), _aname(2), _aname(3)])
        self.assertEqual(results["skipped"], [])
        self.assertEqual(cohort.REQUIRED_FAMILIES + cohort.OPTIONAL_FAMILIES,
                         tuple(cohort.FAMILIES))
        with_memory = results["runs"][_aname(1)]
        self.assertEqual(set(with_memory["families"]), set(cohort.FAMILIES))
        self.assertEqual(with_memory["optional_missing"], {})
        self.assertAlmostEqual(
            with_memory["families"]["delaygo"]["e_modulation"]["ring_cosine_same"], -1.0)
        for seed, reason in ((2, "no delaygo_long_stimulus_endpoints"),
                             (3, "fixed-point criterion")):
            run = results["runs"][_aname(seed)]
            self.assertEqual(set(run["families"]), set(cohort.REQUIRED_FAMILIES))
            self.assertIn(reason, run["optional_missing"]["delaygo"])
        self.assertIn("delaygo / delayanti (1 of 3 runs have its caches)", printed)
        self.assertIn("ring_cosine_same", printed)
        self.assertIn("1/1 runs below 0", printed)
        self.assertIn("MemoryPro/Anti: no stimulus-period caches", printed)
        self.assertIn("MemoryPro/Anti ring cosine -1.00", printed)
        self.assertEqual(
            cohort.endpoint_path(Path("d"), "delaygo", "run").name,
            "delaygo_long_stimulus_endpoints_run.pkl")
        self.assertEqual(
            cohort.sweep_path(Path("d"), "dmcgo", "run").name,
            "dmcgo_cue_interp_long_delay_endpoints_run.pkl")

    def test_run_all_writes_every_settled_run_of_the_feature_and_lists_the_rest(self):
        rng = np.random.default_rng(2)
        translation = 2.0 * _unit(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "two_in_multiples"
            for seed in (2, 1):
                _write_run(root, _aname(seed), _delaydm_caches(rng, translation),
                           _dmc_caches(rng, "orthogonal"))
            _write_run(root, _aname(3, "L21e3"), _delaydm_caches(rng, translation),
                       _dmc_caches(rng, "same"))                                   # other cohort
            _write_run(root, _aname(4), _delaydm_caches(rng, translation, fixed=False),
                       _dmc_caches(rng, "same"))                                   # unsettled
            endpoint, _ = _dmc_caches(rng, "same")
            # seed 5: no DMC sweep
            _write_run(root, _aname(5), _delaydm_caches(rng, translation), (endpoint, None))
            (root / "notes.txt").write_text("not a run")
            with patch("builtins.print") as output:
                results = cohort.run_all(root, FEATURE, out_dir=root / "out")
            with (root / "out" / cohort.RESULT_NAME).open("rb") as stream:
                saved = pickle.load(stream)
            printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
            # No run of the feature at all, and no run that can be scored,
            # are errors rather than empty files.
            with self.assertRaisesRegex(FileNotFoundError, "sibling_fixed_point_analysis"):
                cohort.run_all(root, "L21e9", out_dir=root / "out")
            only_bad = Path(directory) / "bad"
            _write_run(only_bad, _aname(7), _delaydm_caches(rng, translation, fixed=False),
                       _dmc_caches(rng, "same"))
            with patch("builtins.print"), \
                    self.assertRaisesRegex(FileNotFoundError, "settled endpoint"):
                cohort.run_all(only_bad, FEATURE)
            self.assertFalse((only_bad / cohort.RESULT_NAME).exists())
        self.assertEqual(saved["schema_version"], cohort.SCHEMA_VERSION)
        self.assertEqual(saved["feature"], FEATURE)
        self.assertEqual(list(saved["runs"]), [_aname(1), _aname(2)])
        self.assertEqual(saved["runs"][_aname(2)]["seed"], 2)
        self.assertEqual(set(saved["runs"][_aname(1)]["families"]),
                         set(cohort.REQUIRED_FAMILIES))
        self.assertEqual(set(saved["runs"][_aname(1)]["families"]["dmcgo"]), set(REPRESENTATIONS))
        self.assertEqual(results["runs"].keys(), saved["runs"].keys())
        self.assertEqual([row["seed"] for row in saved["skipped"]], [4, 5])
        self.assertIn("fixed-point criterion", saved["skipped"][0]["reason"])
        self.assertIn("dmcgo_cue_interp_long_delay_endpoints", saved["skipped"][1]["reason"])
        self.assertEqual(set(saved["definitions"]), {
            "translation_explained", "alignment_gain", "cue_traversed", "cue_end_residual",
            "category_axis_cosine", "category_separation", "category_best_other_split",
            "category_is_best_split", "category_nearest_centroid_accuracy",
            "ring_cosine_same", "ring_cosine_opposite", "translation_to_ring"})
        self.assertIn("category_is_best_split_dmcnogo", printed)
        self.assertIn(f"{_aname(4)}: skipped", printed)
        self.assertIn("2 runs, 2 skipped", printed)
        self.assertIn("2/2 runs above 1", printed)
        self.assertIn("category_axis_cosine", printed)
        self.assertNotIn(_aname(3, "L21e3"), printed)


class FigureTests(unittest.TestCase):
    def results(self, seeds, feature=FEATURE, schema=cohort.SCHEMA_VERSION, code="orthogonal",
                memory=()):
        """Saved cohort results of `seeds`; the seeds of `memory` also carry
        the optional MemoryPro / MemoryAnti statistics (response-coded for
        the odd ones)."""
        rng = np.random.default_rng(3)
        runs = {}
        for seed in seeds:
            translation = (2.0 + 0.5 * seed) * _unit(3)
            delaydm = _delaydm_caches(rng, translation, end="half" if seed % 2 else "sibling")
            dmc = _dmc_caches(rng, code)
            families = {
                "delaydm1": {name: cohort.delaydm_statistics(
                    *delaydm, cohort.FAMILIES["delaydm1"], rep_key)
                    for name, rep_key in cohort.REPRESENTATIONS.items()},
                "dmcgo": {name: cohort.dmc_statistics(*dmc, cohort.FAMILIES["dmcgo"], rep_key)
                          for name, rep_key in cohort.REPRESENTATIONS.items()},
            }
            if seed in memory:
                caches = _memory_caches(rng, "opposite" if seed % 2 else "same")
                families["delaygo"] = {
                    name: cohort.pro_anti_statistics(*caches, cohort.FAMILIES["delaygo"], rep_key)
                    for name, rep_key in cohort.REPRESENTATIONS.items()}
            runs[_aname(seed, feature)] = {"aname": _aname(seed, feature), "seed": seed,
                                           "families": families}
        return {"schema_version": schema, "feature": feature, "runs": runs, "skipped": [],
                "definitions": dict(cohort.DEFINITIONS)}

    def render(self, function, results, show_legend=True):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            if results is not None:
                _write(base / cohort.RESULT_NAME, results)
            with patch.object(paper_plot, "TWO_IN_MULTIPLES_DIR", base), \
                    patch.object(paper_plot, "SIBLING_L2", "1e4"), \
                    patch.object(paper_plot, "SHOW_LEGEND", show_legend), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as output:
                function()
        for call in save.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        return save, output

    def _points(self, axis):
        collections = paper_plot.mpl.collections
        return [c for c in axis.collections if isinstance(c, collections.PathCollection)]

    def _medians(self, axis):
        collections = paper_plot.mpl.collections
        dashes = [c for c in axis.collections if isinstance(c, collections.LineCollection)
                  and c.get_segments()[0][0, 0] != c.get_segments()[0][1, 0]]   # horizontal
        return sorted(c.get_segments()[0][0, 1] for c in dashes)

    def _check(self, function, name, groups):
        """Render `function` on three runs and check that every column of
        `groups` ([(family, key)], two representation columns each, the
        groups one column apart) holds the saved values of the runs by seed,
        with their median dashes, labels and legend."""
        seeds = (5, 3, 8)
        results = self.results(seeds)
        save, _ = self.render(function, results)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, name)
        axis = figure.axes[0]
        points = self._points(axis)
        self.assertEqual(len(points), 2 * len(groups))
        medians = []
        for group, (family, key) in enumerate(groups):
            for column, representation in enumerate(REPRESENTATIONS):
                values = [results["runs"][_aname(seed)]["families"][family][representation][key]
                          for seed in sorted(seeds)]
                drawn = points[2 * group + column].get_offsets()
                np.testing.assert_allclose(drawn[:, 0], 3 * group + column, atol=0.3)
                np.testing.assert_allclose(drawn[:, 1], values)
                np.testing.assert_allclose(
                    points[2 * group + column].get_facecolors()[0][:3],
                    paper_plot.mpl.colors.to_rgb(paper_plot._SCATTER_POINT_COLOR))
                medians.append(np.median(values))
        np.testing.assert_allclose(self._medians(axis), sorted(medians))
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["Hidden", r"W $\odot$ M"] * len(groups))
        self.assertIsNotNone(axis.get_legend())
        self.assertIn("3 runs", axis.get_legend().get_title().get_text())
        self.assertIn("3 runs", save.call_args.kwargs["extra"])
        return figure, axis, results

    def test_translation_figure_draws_the_explained_fractions(self):
        _, axis, _ = self._check(paper_plot.plot_sibling_cohort_translation,
                                 "multitask_sibling_cohort_translation.png",
                                 [("delaydm1", "translation_explained")])
        self.assertEqual(axis.get_ylim(), (0.0, 1.08))

    def test_alignment_figure_is_on_a_log_axis_with_one_marked(self):
        _, axis, _ = self._check(paper_plot.plot_sibling_cohort_alignment,
                                 "multitask_sibling_cohort_alignment.png",
                                 [("delaydm1", "alignment_gain")])
        self.assertEqual(axis.get_yscale(), "log")
        self.assertIn(1.0, [line.get_ydata()[0] for line in axis.lines if len(line.get_xdata()) == 2
                            and line.get_linestyle() == "--"])

    def test_cue_transfer_figure_holds_both_families(self):
        _, axis, _ = self._check(paper_plot.plot_sibling_cohort_cue_transfer,
                                 "multitask_sibling_cohort_cue_transfer.png",
                                 [("delaydm1", "cue_end_residual"), ("dmcgo", "cue_end_residual")])
        self.assertEqual([text.get_text() for text in axis.texts], ["DelayDM", "DMC"])
        # Room below 0, so points at 0 are not cut by the axis.
        self.assertLess(axis.get_ylim()[0], 0.0)
        self.assertGreater(axis.get_ylim()[0], -0.1)

    def test_category_axis_figure_marks_the_three_codes(self):
        figure, axis, _ = self._check(paper_plot.plot_sibling_cohort_category_axis,
                                      "multitask_sibling_cohort_category_axis.png",
                                      [("dmcgo", "category_axis_cosine")])
        self.assertEqual(axis.get_ylim(), (-1.15, 1.15))
        levels = sorted(line.get_ydata()[0] for line in axis.lines if len(line.get_xdata()) == 2
                        and line.get_color() == "0.6")
        self.assertEqual(levels, [-1.0, 0.0, 1.0])
        # The levels are named on a secondary axis on the right (a child of the
        # axes, not a figure axes), off the points.
        self.assertEqual(len(figure.axes), 1)
        names = [label.get_text() for label in axis.child_axes[0].get_yticklabels()]
        self.assertEqual(names, ["reversed code", "orthogonal", "same code"])
        self.assertEqual(len(axis.texts), 0)

    def test_category_memory_figure_holds_both_rules_on_a_log_axis(self):
        _, axis, _ = self._check(paper_plot.plot_sibling_cohort_category_memory,
                                 "multitask_sibling_cohort_category_memory.png",
                                 [("dmcgo", "category_separation_dmcgo"),
                                  ("dmcgo", "category_separation_dmcnogo")])
        self.assertEqual(axis.get_yscale(), "log")
        self.assertEqual([text.get_text() for text in axis.texts],
                         ["ReactCategoryPro", "ReactCategoryAnti"])
        self.assertIn(1.0, [line.get_ydata()[0] for line in axis.lines if len(line.get_xdata()) == 2
                            and line.get_linestyle() == "--"])
        # A rule that does not separate the categories, or whose separation
        # runs into the thousands, is a hollow point at the edge of the
        # axis, not a point lost beyond it.
        floor, ceiling = paper_plot._SIBLING_COHORT_MEMORY_LIMITS
        results = self.results((1, 2))
        hidden = results["runs"][_aname(1)]["families"]["dmcgo"]["hidden"]
        hidden["category_separation_dmcgo"] = 0.0
        hidden["category_separation_dmcnogo"] = 5000.0
        save, _ = self.render(paper_plot.plot_sibling_cohort_category_memory, results)
        axis = save.call_args.args[0].axes[0]
        self.assertEqual(axis.get_ylim(), (floor / 1.4, ceiling * 1.4))
        hollow = [c for c in self._points(axis)
                  if len(c.get_offsets()) and np.allclose(c.get_facecolors()[0][:3], 1.0)]
        self.assertEqual(len(hollow), 2)
        drawn = sorted((c.get_offsets()[0, 0], c.get_offsets()[0, 1]) for c in hollow)
        self.assertAlmostEqual(drawn[0][0], 0.0, delta=0.3)    # DMCGo hidden, at the floor
        self.assertAlmostEqual(drawn[0][1], floor)
        self.assertAlmostEqual(drawn[1][0], 3.0, delta=0.3)    # DMCNoGo hidden, at the ceiling
        self.assertAlmostEqual(drawn[1][1], ceiling)

    def test_figures_are_skipped_without_usable_results(self):
        for results, note in ((None, "not found"),
                              (self.results((1, 2), feature="L21e3"), "not the L21e4 cohort"),
                              (self.results((1, 2), schema=99), "schema_version"),
                              (self.results((1,)), "need >= 2")):
            with self.subTest(note=note):
                save, output = self.render(paper_plot.plot_sibling_cohort_translation, results)
                save.assert_not_called()
                printed = "\n".join(str(call.args[0]) for call in output.call_args_list
                                    if call.args)
                self.assertIn(note, printed)
                if note != "need >= 2":
                    self.assertIn("sibling_cohort.py --feature L21e4", printed)

    def test_ring_cosine_figure_draws_the_runs_that_have_the_optional_family(self):
        results = self.results((5, 3, 8, 6), memory=(5, 3, 8))
        save, _ = self.render(paper_plot.plot_sibling_cohort_ring_cosine, results)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, "multitask_sibling_cohort_ring_cosine.png")
        axis = figure.axes[0]
        # Three of the four runs have the family; by seed: 3 and 5 hold a
        # direction at the opposite one's place (-1), 8 at the same place (+1).
        points = self._points(axis)
        self.assertEqual(len(points), 2)
        for column in points:
            np.testing.assert_allclose(column.get_offsets()[:, 1], [-1.0, -1.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(self._medians(axis), [-1.0, -1.0], atol=1e-9)
        self.assertEqual(axis.get_ylim(), (-1.15, 1.15))
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         ["Hidden", r"W $\odot$ M"])
        codes = figure.axes[0].child_axes[0]
        self.assertEqual([label.get_text() for label in codes.get_yticklabels()],
                         ["opposite direction", "orthogonal", "same direction"])
        self.assertEqual(axis.get_legend().get_title().get_text(), "3 runs")
        extra = save.call_args.kwargs["extra"]
        self.assertIn("3 runs; MemoryPro/Anti ring_cosine_same: hidden -1.000", extra)
        self.assertIn("MemoryPro/Anti translation_to_ring", extra)
        self.assertIn("hidden 2/3 runs below 0", extra)
        # The delay families' figures still draw all four runs.
        save, _ = self.render(paper_plot.plot_sibling_cohort_translation, results)
        self.assertEqual(save.call_args.args[0].axes[0].get_legend().get_title().get_text(),
                         "4 runs")
        # Fewer than two runs with the family: skipped, with the commands.
        save, output = self.render(paper_plot.plot_sibling_cohort_ring_cosine,
                                   self.results((5, 3, 8), memory=(5,)))
        save.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("1 of 3 cohort runs have MemoryPro / MemoryAnti", printed)
        self.assertIn("--families delaygo --method settled_endpoint", printed)

    @staticmethod
    def _memory_projection(aname, rule_gap=0.1, settled=True):
        """A stimulus-period projection cache: two repeats of eight directions
        per rule on a ring in PC3 and PC1, MemoryAnti shifted by `rule_gap`
        within that plane."""
        stim = np.tile(np.arange(8), 2)
        angles = 2 * np.pi * stim / 8
        base = np.zeros((stim.size, 6))
        base[:, 2] = 3.0 * np.cos(angles)            # the widest spread: PC3
        base[:, 0] = 2.0 * np.sin(angles)            # then PC1
        proj = np.concatenate([base, base + rule_gap * _unit(0)[:6]])
        path = np.linspace(0.0, 1.0, 5)[None, :, None] * proj[:, None, :]
        is_fixed = np.ones(proj.shape[0], dtype=bool)
        is_fixed[0] = settled
        entry = {"proj": proj, "trajectory_proj": path,
                 "task_idx": np.repeat([0, 1], stim.size), "stim_idx": np.tile(stim, 2),
                 "task_names": ["delaygo", "delayanti"], "is_fixed": is_fixed,
                 "explained_variance_ratio": np.array([0.3, 0.1, 0.4, 0.05, 0.05, 0.05])}
        return {"aname": aname, "family": "delaygo", "method": "long_stimulus_endpoint",
                "representations": {"hidden": entry, "e_modulation": entry}}

    @staticmethod
    def _memory_sweep(aname, jump=0.0):
        """A rule-cue sweep projection cache: the eight directions spread
        along PC3 (and less along PC1) and every trial moves along PC6; the
        first trial also hops by `jump` along PC6 halfway through."""
        stim = np.tile(np.arange(8), 2)
        angles = 2 * np.pi * stim / 8
        base = np.zeros((stim.size, 6))
        base[:, 2] = 3.0 * np.cos(angles)
        base[:, 0] = 2.0 * np.sin(angles)
        alphas = np.linspace(0.0, 1.0, 11)
        proj = base[None] + 5.0 * alphas[:, None, None] * _unit(5)[None, None, :6]
        proj[5:, 0, 5] += jump
        sweep = {"target_rule": "delayanti", "proj": proj, "stim_idx": stim,
                 "stimulus_magnitude": np.ones(stim.size),
                 "is_fixed": np.ones(proj.shape[:2], dtype=bool)}
        entry = {"sweeps": {"delaygo": sweep}}
        return {"kind": "cue_interp", "aname": aname, "family": "delaygo",
                "method": "long_stimulus_endpoint", "alphas": alphas,
                "task_names": ["delaygo", "delayanti"],
                "representations": {"hidden": entry, "e_modulation": entry}}

    def _render_memory(self, function, caches):
        """Run a MemoryPro / MemoryAnti illustration on the cached runs of
        `caches`, {seed: (endpoint rule gap, sweep jump, endpoints settled)}."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for seed, (rule_gap, jump, settled) in caches.items():
                aname = _aname(seed)
                _write(base / aname / f"delaygo_long_stimulus_endpoint_pc_projections_{aname}.pkl",
                       self._memory_projection(aname, rule_gap, settled))
                _write(base / aname / ("delaygo_cue_interp_long_stimulus_endpoint_pc_projections_"
                                       f"{aname}.pkl"), self._memory_sweep(aname, jump))
            # A run of another cohort is never a candidate.
            other = _aname(1, "L21e3")
            _write(base / other / f"delaygo_long_stimulus_endpoint_pc_projections_{other}.pkl",
                   self._memory_projection(other, 0.001))
            with patch.object(paper_plot, "TWO_IN_MULTIPLES_DIR", base), \
                    patch.object(paper_plot, "SIBLING_L2", "1e4"), \
                    patch.object(paper_plot, "SHOW_LEGEND", True), \
                    patch.object(paper_plot, "_ensure_out_dir"), \
                    patch.object(paper_plot, "_save_fig") as save, \
                    patch("builtins.print") as output:
                function()
        for call in save.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        return save, "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)

    # {seed: (endpoint rule gap, sweep jump, endpoints settled)}: seed 5 has
    # the clearest endpoints among the settled runs, seed 8 the smoothest sweep.
    MEMORY_CACHES = {3: (0.5, 2.0, True), 5: (0.1, 1.0, True), 8: (0.3, 0.0, True),
                     2: (0.01, 3.0, False)}

    def test_memory_endpoint_figure_draws_the_most_legible_run_in_its_widest_plane(self):
        self.assertEqual(paper_plot._endpoint_spread_pc_pair(
            np.array([[0.0, 5.0, 1.0], [2.0, 5.0, -1.0], [-2.0, 5.0, 1.0]])), (0, 2))
        # Neighboring directions of the (3 cos, 2 sin) ring are at least
        # 1.665 apart; the two rules' points of a stimulus `rule_gap` apart.
        self.assertAlmostEqual(
            paper_plot._memory_endpoint_legibility(self._memory_projection("run", 0.1)),
            16.65, places=2)
        save, printed = self._render_memory(
            paper_plot.plot_multitask_memory_stimulus_endpoints, self.MEMORY_CACHES)
        save.assert_called_once()
        figure, path = save.call_args.args[:2]
        self.assertEqual(path.name, "multitask_memory_stimulus_endpoints.png")
        self.assertEqual([ax.get_title() for ax in figure.axes],
                         ["Hidden state", r"W $\odot$ M"])
        for ax in figure.axes:
            # Eight directions per rule: one marker each, its path behind it,
            # and one grey connector per stimulus.
            self.assertEqual(len(self._points(ax)), 16)
            self.assertEqual(len(ax.lines), 16 + 8)
            connectors = [line for line in ax.lines
                          if isinstance(line.get_color(), str) and line.get_color() == "0.45"]
            self.assertEqual(len(connectors), 8)
            self.assertEqual(ax.get_xlabel(), "Joint Stimulus PC3 (40.0%)")
            self.assertEqual(ax.get_ylabel(), "Joint Stimulus PC1 (30.0%)")
            # Hue is the stimulus direction, for both rules.
            np.testing.assert_allclose(self._points(ax)[0].get_facecolor()[0][:3],
                                       paper_plot.stim_color(0, 8)[:3])
        self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()],
                         ["MemoryPro", "MemoryAnti", "same stimulus"])
        extra = save.call_args.kwargs["extra"]
        # Seed 2 would score highest but has an unsettled endpoint.
        self.assertIn(f"{_aname(5)}: most legible of 3 cached runs: smallest direction gap "
                      "over largest rule gap 16.65", extra)
        self.assertIn("hidden PC3-PC1", extra)
        self.assertIn(f"Skipped {_aname(2)}", printed)
        self.assertIn("above the fixed-point criterion", printed)

    def test_memory_cue_sweep_figures_draw_the_smoothest_run(self):
        cache = self._memory_sweep("run")
        sweep = cache["representations"]["hidden"]["sweeps"]["delaygo"]
        self.assertEqual(paper_plot._memory_sweep_pc_pair(sweep), (3, 6))
        self.assertAlmostEqual(paper_plot._memory_sweep_legibility(cache, "hidden"), 0.0)
        # A hop of 1 beyond the drawn trials' common (median) step, on a
        # rule PC that then spans 6.
        self.assertAlmostEqual(paper_plot._memory_sweep_legibility(
            self._memory_sweep("run", jump=1.0), "hidden"), -1.0 / 6.0)
        save, _ = self._render_memory(paper_plot.plot_multitask_memory_cue_interp,
                                      self.MEMORY_CACHES)
        self.assertEqual(
            [call.args[1].name for call in save.call_args_list],
            [f"multitask_memory_cue_interp_long_stimulus_endpoint_{name}_{view}.png"
             for name in ("emodulation", "hidden") for view in ("2d", "3d")])
        figure = save.call_args_list[0].args[0]                 # the 2d view
        stimulus, rule = figure.axes
        self.assertEqual(stimulus.get_ylabel(), "Joint Stimulus PC3")
        self.assertEqual(rule.get_ylabel(), "Joint Stimulus PC6")
        self.assertEqual(stimulus.get_title(), "MemoryPro stimulus\n→ MemoryAnti cue")
        # One trial per stimulus direction; the stimulus PC stays flat along
        # the sweep and the rule PC moves, without the other runs' hop.
        self.assertEqual(len(stimulus.lines), 8)
        for line in stimulus.lines:
            self.assertAlmostEqual(float(np.ptp(line.get_ydata())), 0.0)
        for line in rule.lines:
            np.testing.assert_allclose(line.get_ydata(), 5.0 * np.linspace(0.0, 1.0, 11))
        for call in save.call_args_list:
            self.assertIn("Joint Stimulus PC3-PC6, stimulus PC and rule PC", call.kwargs["extra"])
            self.assertIn(f"{_aname(8)}, most legible of 4 cached runs: largest drawn "
                          "departure from the common motion 0.00 of a panel",
                          call.kwargs["extra"])

    def test_memory_figures_break_ties_by_seed_and_skip_without_a_cache(self):
        same = {8: (0.1, 0.0, True), 5: (0.1, 0.0, True)}
        for function in (paper_plot.plot_multitask_memory_stimulus_endpoints,
                         paper_plot.plot_multitask_memory_cue_interp):
            save, _ = self._render_memory(function, same)
            self.assertIn(_aname(5), save.call_args.kwargs["extra"])
            save, printed = self._render_memory(function, {})
            save.assert_not_called()
            self.assertIn("no scorable MemoryPro / MemoryAnti", printed)
            self.assertIn("--families delaygo --method settled_endpoint", printed)

    def test_no_legend_leaves_the_axes_bare(self):
        save, _ = self.render(paper_plot.plot_sibling_cohort_translation, self.results((1, 2)),
                              show_legend=False)
        self.assertIsNone(save.call_args.args[0].axes[0].get_legend())

    def test_figures_are_registered_and_agree_with_the_script(self):
        mode = paper_plot.FIGURES_BY_MODE["two_in_multiple"]
        self.assertIs(mode["sibling_cohort_translation"],
                      paper_plot.plot_sibling_cohort_translation)
        self.assertIs(mode["sibling_cohort_alignment"], paper_plot.plot_sibling_cohort_alignment)
        self.assertIs(mode["sibling_cohort_cue_transfer"],
                      paper_plot.plot_sibling_cohort_cue_transfer)
        self.assertIs(mode["sibling_cohort_category_axis"],
                      paper_plot.plot_sibling_cohort_category_axis)
        self.assertIs(mode["sibling_cohort_category_memory"],
                      paper_plot.plot_sibling_cohort_category_memory)
        self.assertEqual(paper_plot.SIBLING_COHORT_RESULT_NAME, cohort.RESULT_NAME)
        self.assertEqual(paper_plot._SIBLING_COHORT_SCHEMA_VERSION, cohort.SCHEMA_VERSION)
        # The delay-family figures draw the two required families; the
        # optional stimulus-period family has its own two figures.
        self.assertEqual(dict(paper_plot._MULTITASK_SIBLING_FAMILIES),
                         {family: cohort.FAMILIES[family]
                          for family in cohort.REQUIRED_FAMILIES})
        self.assertEqual([name for name, _ in paper_plot._SIBLING_COHORT_REPRESENTATIONS],
                         list(cohort.REPRESENTATIONS))
        self.assertEqual(set(paper_plot._SIBLING_COHORT_FAMILY_LABELS),
                         set(cohort.REQUIRED_FAMILIES))
        self.assertEqual(set(paper_plot._SIBLING_COHORT_OPTIONAL_FAMILY_LABELS),
                         set(cohort.OPTIONAL_FAMILIES))
        self.assertEqual(paper_plot._MEMORY_ENDPOINT_TAG,
                         cohort.ENDPOINT_TAGS[paper_plot._MEMORY_FAMILY])
        self.assertIs(mode["sibling_cohort_ring_cosine"],
                      paper_plot.plot_sibling_cohort_ring_cosine)
        self.assertIs(mode["memory_stimulus_endpoints"],
                      paper_plot.plot_multitask_memory_stimulus_endpoints)
        self.assertIs(mode["memory_cue_interp"], paper_plot.plot_multitask_memory_cue_interp)


if __name__ == "__main__":
    unittest.main()
