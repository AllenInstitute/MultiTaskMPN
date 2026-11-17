"""Tests for the opt-in balanced stimulus directions (DelayDM) and stimulus
modalities (the tasks whose stimulus may come on either ring)."""

import inspect
import unittest

import numpy as np

import _bootstrap  # noqa: F401
import mpn_tasks
from core.task_variance import trial_epoch_windows


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
    "delaygo": (lambda config, mode="random", **kw: mpn_tasks.delaygo_(
        config, mode, False, True, True, "normal", "normal", "normal", "normal", False,
        **kw), "stim1"),
    "reactgo": (lambda config, mode="random", **kw: mpn_tasks.reactgo_(
        config, mode, False, False, **kw), "go1"),
    "fdgo": (lambda config, mode="random", **kw: mpn_tasks.fdgo_(
        config, mode, True, "normal", "normal", "normal", False, **kw), "stim1"),
    "dms": (lambda config, mode="random", **kw: mpn_tasks.dms_(
        config, mode, 0, True, True, "normal", "normal", "normal", "normal", False,
        **kw), "stim1"),
    "dmc": (lambda config, mode="random", **kw: mpn_tasks.dmc_(
        config, mode, 1, True, True, "normal", "normal", "normal", "normal", False,
        **kw), "stim1"),
}


def _trial_rings(trial, epoch):
    """Per-trial modality of the stimulus in `epoch` when every trial has
    its own timing: each trial read inside its own window."""
    starts, ends = trial_epoch_windows(trial, epoch)
    rings = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        shown = np.abs(trial.x[start:end, index]).max(axis=0)
        on_ring = [shown[N_FIXATE + ring * RING_WIDTH:N_FIXATE + (ring + 1) * RING_WIDTH].max() > 0
                   for ring in (0, 1)]
        assert sum(on_ring) == 1, "each trial shows exactly one ring"
        rings.append(1 + on_ring.index(True))
    return np.array(rings)


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

    def test_per_trial_timing_takes_the_same_exact_balance(self):
        """'random_batch' draws the ring per trial, balanced only on average;
        the option makes it exact while every trial keeps its own timing."""
        for name, (generate, epoch) in FREE_MODALITY_TASKS.items():
            with self.subTest(task=name):
                trial = generate(_config(), "random_batch", batch_size=32,
                                 balanced_stimulus_modalities=True)
                self.assertEqual(np.shape(trial.tdim), (32,))
                self.assertGreater(np.unique(trial.tdim).size, 1)
                rings = _trial_rings(trial, epoch)
                np.testing.assert_array_equal(np.bincount(rings, minlength=3), [0, 16, 16])
                if name in ("dms", "dmc"):          # the second stimulus: the other ring
                    np.testing.assert_array_equal(_trial_rings(trial, "go1"), 3 - rings)
                free = generate(_config(), "random_batch", batch_size=32)
                self.assertEqual(set(np.unique(_trial_rings(free, epoch))), {1, 2})

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

    def test_the_wrapper_option_is_opt_in_and_needs_a_drawn_mode(self):
        parameter = inspect.signature(
            mpn_tasks.generate_trials_wrap).parameters["balanced_stimulus_modalities"]
        self.assertFalse(parameter.default)
        with self.assertRaisesRegex(ValueError, "'random' or 'random_batch'"):
            mpn_tasks.generate_trials_wrap(
                {"rules": ["delaygo"]}, 4, rules=["delaygo"], mode_input="test",
                balanced_stimulus_modalities=True)




class PinnedStimulusModalityTests(unittest.TestCase):
    def test_every_free_modality_task_shows_the_pinned_ring_only(self):
        for modality in (1, 2):
            for name, (generate, epoch) in FREE_MODALITY_TASKS.items():
                with self.subTest(task=name, modality=modality):
                    trial = generate(_config(), batch_size=32, stimulus_modality=modality)
                    np.testing.assert_array_equal(
                        _ring_of(trial, trial.epochs[epoch]), np.full(32, modality))

    def test_the_pin_changes_nothing_but_the_ring(self):
        """The generator's own draw is still made, so timing and directions
        are those of the unpinned batch of the same generator state."""
        for name, (generate, epoch) in FREE_MODALITY_TASKS.items():
            with self.subTest(task=name):
                free = generate(_config(), batch_size=32)
                pinned = generate(_config(), batch_size=32, stimulus_modality=1)
                self.assertEqual(free.epochs, pinned.epochs)
                np.testing.assert_array_equal(free.meta["stim1"], pinned.meta["stim1"])
                np.testing.assert_array_equal(free.y, pinned.y)

    def test_second_stimulus_of_the_match_tasks_takes_the_other_ring(self):
        for name in ("dms", "dmc"):
            with self.subTest(task=name):
                generate, _ = FREE_MODALITY_TASKS[name]
                trial = generate(_config(), batch_size=32, stimulus_modality=1)
                go_on, go_off = trial.epochs["go1"]
                second = _ring_of(trial, (go_on, trial.tdim if go_off is None else go_off))
                np.testing.assert_array_equal(second, np.full(32, 2))

    def test_the_pin_comes_from_the_task_parameters_and_is_validated(self):
        for value, mode, balanced, message in (
                (3, "random", False, "must be 1 or 2"),
                (1, "test", False, "requires mode_input 'random' or 'random_batch'"),
                (1, "random", True, "excludes balanced")):
            with self.subTest(value=value, mode=mode, balanced=balanced):
                with self.assertRaisesRegex(ValueError, message):
                    mpn_tasks.generate_trials_wrap(
                        {"rules": ["delaygo"], "stimulus_modality": value}, 4,
                        rules=["delaygo"], mode_input=mode,
                        balanced_stimulus_modalities=balanced)


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
