"""Tests for the opt-in balanced stimulus directions (DelayDM) and stimulus
modalities (the tasks whose stimulus may come on either ring)."""

import inspect
import unittest

import numpy as np

import _bootstrap  # noqa: F401
import mpn_tasks


def _config(seed=17):
    return {
        "dt": 40,
        "rng": np.random.RandomState(seed),
        "n_eachring": 8,
        "in_out_mode": "low_dim",
        "fixate_off": True,
        "n_input": 6,
        "n_output": 3,
        "loss_type": "lsq",
        "sigma_x": 0.0,
        "alpha": 0.2,
        "get_meta": True,
    }


def _delaydm_trial(config, stim_mod):
    return mpn_tasks.delaydm_(
        config, "random", stim_mod, False, True,
        "long", "normal", "normal", "normal", False,
        batch_size=32, balanced_stim1=True)


class BalancedStimulusGenerationTests(unittest.TestCase):
    def test_every_direction_has_four_trials_and_siblings_are_aligned(self):
        config = _config()
        rng_state = config["rng"].get_state()
        first = _delaydm_trial(config, 1)
        config["rng"].set_state(rng_state)
        second = _delaydm_trial(config, 2)

        directions, counts = np.unique(first.meta["stim1"], return_counts=True)
        np.testing.assert_array_equal(directions, np.arange(8))
        np.testing.assert_array_equal(counts, np.full(8, 4))
        np.testing.assert_array_equal(first.meta["stim1"], second.meta["stim1"])
        np.testing.assert_allclose(
            first.meta["stim1_strs"], second.meta["stim1_strs"])

    def test_nondivisible_trial_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            mpn_tasks._balanced_stimulus_locations(
                np.random.RandomState(1), 30, 8)

    def test_balancing_is_opt_in(self):
        parameter = inspect.signature(
            mpn_tasks.generate_trials_wrap
        ).parameters["balanced_stimulus_directions"]
        self.assertFalse(parameter.default)


N_FIXATE = 2          # fixation and fixation-off channels of `_config`
RING_WIDTH = 2        # (sin, cos) of one low_dim ring


def _ring_of(trial, window):
    """Per-trial modality (1 or 2) of the stimulus shown in `window`."""
    start, stop = window
    shown = np.abs(trial.x[start:stop]).max(axis=0)          # (trials, channels)
    on_ring = np.stack([
        shown[:, N_FIXATE + ring * RING_WIDTH:N_FIXATE + (ring + 1) * RING_WIDTH].max(axis=1)
        for ring in (0, 1)], axis=1) > 0
    assert np.all(on_ring.sum(axis=1) == 1), "each trial shows exactly one ring"
    return np.argmax(on_ring, axis=1) + 1


# name -> (generator call taking **kwargs, epoch of the first stimulus)
FREE_MODALITY_TASKS = {
    "delaygo": (lambda config, **kw: mpn_tasks.delaygo_(
        config, "random", False, True, True, "normal", "normal", "normal", "normal", False,
        **kw), "stim1"),
    "reactgo": (lambda config, **kw: mpn_tasks.reactgo_(
        config, "random", False, False, **kw), "go1"),
    "fdgo": (lambda config, **kw: mpn_tasks.fdgo_(
        config, "random", True, "normal", "normal", "normal", False, **kw), "stim1"),
    "dms": (lambda config, **kw: mpn_tasks.dms_(
        config, "random", 0, True, True, "normal", "normal", "normal", "normal", False,
        **kw), "stim1"),
    "dmc": (lambda config, **kw: mpn_tasks.dmc_(
        config, "random", 1, True, True, "normal", "normal", "normal", "normal", False,
        **kw), "stim1"),
}


class BalancedStimulusModalityTests(unittest.TestCase):
    def test_helper_gives_each_modality_half_of_the_trials(self):
        modalities = mpn_tasks._balanced_stimulus_modalities(np.random.RandomState(3), 50)
        np.testing.assert_array_equal(np.bincount(modalities, minlength=3), [0, 25, 25])
        # Shuffled, and reproducible from the generator's state.
        self.assertFalse(np.array_equal(modalities, np.tile([1, 2], 25)))
        np.testing.assert_array_equal(
            modalities, mpn_tasks._balanced_stimulus_modalities(np.random.RandomState(3), 50))
        for batch_size in (0, 7):
            with self.assertRaisesRegex(ValueError, "even batch_size"):
                mpn_tasks._balanced_stimulus_modalities(np.random.RandomState(3), batch_size)

    def test_every_free_modality_task_shows_both_rings_at_one_timing(self):
        for name, (generate, epoch) in FREE_MODALITY_TASKS.items():
            with self.subTest(task=name):
                trial = generate(_config(), batch_size=32, balanced_stimulus_modalities=True)
                rings = _ring_of(trial, trial.epochs[epoch])
                np.testing.assert_array_equal(np.bincount(rings, minlength=3), [0, 16, 16])
                # The timing stays the scalar one of the "random" mode.
                self.assertEqual(np.ndim(trial.tdim), 0)
                for start, stop in trial.epochs.values():
                    self.assertEqual(np.ndim(start), 0)
                    self.assertEqual(np.ndim(stop), 0)

    def test_without_the_option_a_batch_keeps_its_single_modality(self):
        for name, (generate, epoch) in FREE_MODALITY_TASKS.items():
            with self.subTest(task=name):
                trial = generate(_config(), batch_size=32)
                self.assertEqual(np.unique(_ring_of(trial, trial.epochs[epoch])).size, 1)

    def test_second_stimulus_of_the_match_tasks_follows_the_first(self):
        for name in ("dms", "dmc"):
            with self.subTest(task=name):
                generate, _ = FREE_MODALITY_TASKS[name]
                trial = generate(_config(), batch_size=32, balanced_stimulus_modalities=True)
                first = _ring_of(trial, trial.epochs["stim1"])
                go_on, go_off = trial.epochs["go1"]
                second = _ring_of(trial, (go_on, trial.tdim if go_off is None else go_off))
                # separate_input: the second stimulus is on the other ring.
                np.testing.assert_array_equal(second, 3 - first)
                np.testing.assert_array_equal(trial.meta["stim1_mod"], first)
                np.testing.assert_array_equal(trial.meta["stim2_mod"], second)

    def test_a_task_with_its_own_modality_ignores_the_option(self):
        for stim_mod in (1, 2):
            trial = mpn_tasks.delaydm_(
                _config(), "random", stim_mod, False, True,
                "normal", "normal", "normal", "normal", False,
                batch_size=32, balanced_stimulus_modalities=True)
            np.testing.assert_array_equal(
                _ring_of(trial, trial.epochs["stim1"]), np.full(32, stim_mod))

    def test_the_wrapper_option_is_opt_in_and_needs_the_random_mode(self):
        parameter = inspect.signature(
            mpn_tasks.generate_trials_wrap).parameters["balanced_stimulus_modalities"]
        self.assertFalse(parameter.default)
        with self.assertRaisesRegex(ValueError, "mode_input='random'"):
            mpn_tasks.generate_trials_wrap(
                {"rules": ["delaygo"]}, 4, rules=["delaygo"], mode_input="random_batch",
                balanced_stimulus_modalities=True)



class SeededGeneratorTests(unittest.TestCase):
    def test_multisensory_trials_depend_on_the_task_generator_only(self):
        """The multisensory tasks drew their modality split from the global
        np.random state, so a seeded generator did not reproduce them."""
        # multidm takes no long_response / long_all argument.
        calls = {
            mpn_tasks.multidelaydm: ("normal", "normal", "normal", "normal", False),
            mpn_tasks.multidm: ("normal", "normal", "normal"),
        }
        for task, periods in calls.items():
            with self.subTest(task=task.__name__):
                trials = []
                for global_seed in (1, 2):
                    np.random.seed(global_seed)
                    trials.append(task(
                        _config(seed=5), "random", True, True, *periods, batch_size=8))
                np.testing.assert_array_equal(trials[0].x, trials[1].x)
                np.testing.assert_array_equal(trials[0].y, trials[1].y)


if __name__ == "__main__":
    unittest.main()
