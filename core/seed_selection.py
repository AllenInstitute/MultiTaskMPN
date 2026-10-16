"""Seed selection shared by the training batch scripts.

Every training script (one_task, one_task_rnn, two_task, multiple_task,
pretraining) trains N_TRIALS independent networks under one setting and
names each run ``<ruleset>_seed<N>_<setting tags>``. A batch should add new
seeds to that setting's cohort instead of reusing a hand-edited list, and
must never overwrite a saved run. The scripts differ only in how a run's
name is built and which files mark a run as finished or started; this
module holds the shared part:

- ``seed_pattern(run_name)`` turns a script's own run-name function into a
  regex that captures the seed, so the scan and the training agree on the
  name by construction;
- ``seeds_with_file`` / ``seeds_matching`` list the seeds present in an
  output directory;
- ``draw_new_seeds`` draws from the seed pool;
- ``select_seeds`` validates explicit seeds or draws fresh ones, and
  ``choose_seeds`` wraps the scan and the selection for the scripts whose
  runs leave a ``savednet_<name>.pt`` when finished and a
  ``param_<name>_param.json`` when started.
"""
import random
import re
from pathlib import Path

SEED_POOL = range(1, 1001)
COMPLETED_FILE = ("savednet_", ".pt")       # the last file a trial writes
STARTED_FILE = ("param_", "_param.json")    # written before training starts
_SENTINEL = "\x00seed\x00"


def seed_pattern(run_name):
    """Regex matching ``run_name(seed)`` for any seed, the seed captured as
    group 1. ``run_name`` must embed its argument exactly once, as text."""
    escaped = re.escape(run_name(_SENTINEL))
    marker = re.escape(_SENTINEL)
    if escaped.count(marker) != 1:
        raise ValueError("run_name must embed its seed argument exactly once")
    return re.compile(escaped.replace(marker, r"(\d+)"))


def seeds_matching(pattern, directory):
    """Sorted seeds captured by ``pattern`` (group 1) over the entry names of
    ``directory``; [] when the directory does not exist."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    matches = (pattern.fullmatch(entry.name) for entry in directory.iterdir())
    return sorted({int(match.group(1)) for match in matches if match})


def seeds_with_file(directory, run_name, prefix, suffix, per_run_directory=False):
    """Sorted seeds whose run has the file ``<prefix><run name><suffix>``:
    directly in ``directory``, or, with ``per_run_directory``, inside the
    run's own subdirectory ``directory/<run name>/``."""
    pattern = seed_pattern(run_name)
    if not per_run_directory:
        return seeds_matching(re.compile(re.escape(prefix) + pattern.pattern + re.escape(suffix)),
                              directory)
    directory = Path(directory)
    if not directory.is_dir():
        return []
    seeds = set()
    for entry in directory.iterdir():
        match = pattern.fullmatch(entry.name)
        if match and (entry / f"{prefix}{entry.name}{suffix}").exists():
            seeds.add(int(match.group(1)))
    return sorted(seeds)


def draw_new_seeds(n_runs, taken, *, pool=SEED_POOL, rng=None):
    """``n_runs`` distinct seeds of ``pool`` that are not in ``taken``."""
    rng = random.Random() if rng is None else rng
    candidates = sorted(set(pool) - set(taken))
    if len(candidates) < n_runs:
        raise ValueError(f"Only {len(candidates)} unused seeds remain in "
                         f"{min(pool)}..{max(pool)}; cannot add {n_runs} runs")
    return rng.sample(candidates, n_runs)


def select_seeds(n_trials, seeds, completed, started=(), *, label="", pool=SEED_POOL, rng=None):
    """The seeds a batch trains: ``seeds`` when given, else ``n_trials`` fresh ones.

    ``completed`` are the seeds with a saved run of the setting, ``started``
    those with a run that began (in progress or failed midway). A fresh draw
    skips both, so no saved run is overwritten and no running one is
    duplicated. Explicit seeds must be distinct nonnegative integers; one
    with a saved run is rejected rather than overwritten, one whose run only
    started is allowed with a warning, so a failed run can be redone.
    """
    completed = sorted(set(completed))
    started = sorted(set(started) - set(completed))
    print(f"Runs of {label or 'this setting'}: completed seeds {completed}, "
          f"started but unfinished {started}")
    if seeds is None:
        return draw_new_seeds(n_trials, set(completed) | set(started), pool=pool, rng=rng)
    seeds = list(seeds)
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct to avoid overwriting runs")
    clashes = sorted(set(seeds) & set(completed))
    if clashes:
        raise ValueError(f"seeds {clashes} already have a saved run of {label or 'this setting'}; "
                         "choose other seeds or remove those outputs")
    unfinished = sorted(set(seeds) & set(started))
    if unfinished:
        print(f"Warning: seeds {unfinished} have a started but unfinished run "
              f"(in progress or failed); their outputs will be overwritten")
    return seeds


def choose_seeds(n_trials, seeds, directory, run_name, *, per_run_directory=False,
                 completed_file=COMPLETED_FILE, started_file=STARTED_FILE, pool=SEED_POOL, rng=None):
    """``select_seeds`` for a script whose runs are named by ``run_name`` and
    saved under ``directory``, with the checkpoint marking a finished run and
    the parameter file a started one (see COMPLETED_FILE / STARTED_FILE)."""
    return select_seeds(
        n_trials, seeds,
        seeds_with_file(directory, run_name, *completed_file, per_run_directory),
        seeds_with_file(directory, run_name, *started_file, per_run_directory),
        label=run_name("*"), pool=pool, rng=rng)
