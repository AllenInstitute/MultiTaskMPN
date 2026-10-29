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


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def _write_run(root, aname, delaydm=None, dmc=None):
    run_dir = root / aname
    run_dir.mkdir(parents=True, exist_ok=True)
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


class ScriptTests(unittest.TestCase):
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
            with self.assertRaisesRegex(FileNotFoundError, "sibling_delay_analysis"):
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
        self.assertEqual(set(saved["runs"][_aname(1)]["families"]), set(cohort.FAMILIES))
        self.assertEqual(set(saved["runs"][_aname(1)]["families"]["dmcgo"]), set(REPRESENTATIONS))
        self.assertEqual(results["runs"].keys(), saved["runs"].keys())
        self.assertEqual([row["seed"] for row in saved["skipped"]], [4, 5])
        self.assertIn("fixed-point criterion", saved["skipped"][0]["reason"])
        self.assertIn("dmcgo_cue_interp_long_delay_endpoints", saved["skipped"][1]["reason"])
        self.assertEqual(set(saved["definitions"]), {
            "translation_explained", "alignment_gain", "cue_traversed", "cue_end_residual",
            "category_axis_cosine", "category_separation", "category_best_other_split",
            "category_is_best_split", "category_nearest_centroid_accuracy"})
        self.assertIn("category_is_best_split_dmcnogo", printed)
        self.assertIn(f"{_aname(4)}: skipped", printed)
        self.assertIn("2 runs, 2 skipped", printed)
        self.assertIn("2/2 runs above 1", printed)
        self.assertIn("category_axis_cosine", printed)
        self.assertNotIn(_aname(3, "L21e3"), printed)


class FigureTests(unittest.TestCase):
    def results(self, seeds, feature=FEATURE, schema=cohort.SCHEMA_VERSION, code="orthogonal"):
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
        self.assertEqual(dict(paper_plot._MULTITASK_SIBLING_FAMILIES), cohort.FAMILIES)
        self.assertEqual([name for name, _ in paper_plot._SIBLING_COHORT_REPRESENTATIONS],
                         list(cohort.REPRESENTATIONS))
        self.assertEqual(set(paper_plot._SIBLING_COHORT_FAMILY_LABELS), set(cohort.FAMILIES))


if __name__ == "__main__":
    unittest.main()
