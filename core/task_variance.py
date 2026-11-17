"""Shared trial-window and cache contract for task-period variance analyses."""

import pickle
from pathlib import Path

import numpy as np

from core.modulation_variants import LESION_MODULATION_TYPES


# First version pools trials and time directly and excludes trial-end padding.
VARIANCE_METHOD = "trial_time"
NEURON_VARIANTS = ("input_normalized", "input_unnormalized",
                   "hidden_normalized", "hidden_unnormalized")


def trial_epoch_bounds(trial, phase):
    """Aligned (start, end) indices within this task's actual trial length.

    For a batch whose trials share one timing. The clustering analysis uses
    trial_epoch_windows since its trials have per-trial timing (2026-11-16);
    this stays for test/variance_conditioning_check.py."""
    def scalar(value):
        values = np.asarray(value)
        if values.size == 0 or not np.all(values == values.flat[0]):
            raise ValueError("Task variance requires aligned trial timings")
        return int(values.flat[0])

    length = scalar(trial.tdim)
    start, end = trial.epochs[phase]
    start = 0 if start is None else scalar(start)
    end = length if end is None else scalar(end)
    if not 0 <= start <= end <= length:
        raise ValueError(f"{phase}: epoch {(start, end)} exceeds trial length {length}")
    return start, end


def trial_epoch_windows(trial, phase):
    """Per-trial (starts, ends) index arrays of `phase`, each window within
    its own trial's length.

    For a batch whose trials have their own timing (the generator's
    'random_batch' mode): epoch edges and the trial length may be scalars or
    one value per trial, a missing start is the trial's first step and a
    missing end its own last one, never the padded end of the batch. The
    arrays have one entry per trial of the batch, in batch order."""
    n_trials = int(trial.batch_size)

    def per_trial(value, name):
        values = np.asarray(value)
        if values.ndim == 0:
            return np.full(n_trials, int(values), dtype=int)
        if values.shape != (n_trials,):
            raise ValueError(f"{phase}: {name} has {values.shape}, expected {n_trials} trials")
        return values.astype(int)

    lengths = per_trial(trial.tdim, "trial length")
    start, end = trial.epochs[phase]
    starts = np.zeros(n_trials, dtype=int) if start is None else per_trial(start, "start")
    ends = lengths.copy() if end is None else per_trial(end, "end")
    if not np.all((0 <= starts) & (starts <= ends) & (ends <= lengths)):
        raise ValueError(f"{phase}: an epoch window exceeds its trial's length")
    return starts, ends


def variance_cache_is_current(payload, required_keys=()):
    """Cluster entries all use the current estimator; no metadata sibling key."""
    return (isinstance(payload, dict) and bool(payload)
            and set(required_keys).issubset(payload)
            and all(isinstance(entry, dict)
                    and entry.get("variance_method") == VARIANCE_METHOD
                    for entry in payload.values()))


def require_current_variance(payload, source):
    if not variance_cache_is_current(payload):
        raise ValueError(f"{source}: stale task-variance cache; rerun clustering "
                         "via run_pipeline.py (or untrained_clustering.py) "
                         "for trial_time variance and actual trial endpoints.")


def require_current_variance_result(payload, source):
    """Reject derived results from another estimator, including old scatter entries."""
    if not (isinstance(payload, dict) and
            (payload.get("variance_method") == VARIANCE_METHOD
             or variance_cache_is_current(payload))):
        raise ValueError(f"{source}: stale task-variance result; rerun clustering and "
                         "its downstream analysis for trial_time variance.")


def cluster_caches_current(paths):
    """Completion requires both full cluster pickles with current provenance."""
    if len(paths) != 2:
        return False
    for path, required in zip(paths, (NEURON_VARIANTS, LESION_MODULATION_TYPES)):
        try:
            with Path(path).open("rb") as stream:
                payload = pickle.load(stream)
        except (OSError, EOFError, pickle.UnpicklingError, ValueError):
            return False
        if not variance_cache_is_current(payload, required):
            return False
    return True
