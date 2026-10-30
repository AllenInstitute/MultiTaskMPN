"""
Activity-matched random controls for the single-cluster lesions.

lesion.py compares every cluster lesion with size-matched random lesions
drawn from ALL units of that side or layer. More than half of the hidden
neurons and of the plastic synapses belong to the unresponsive class, so
those controls are mostly silent units and barely hurt the network: the
significance test then asks "is this cluster more important than a
half-silent random set", and its control spread tracks the cluster size.
This script scores the same single-cluster conditions again, on a seeded
test set, with controls drawn from the RESPONSIVE units only (``--pool
responsive``; ``--pool all`` repeats the historical pool on the same trials,
for a direct comparison). The combined lesions and the pruning sweep are
not repeated; they keep lesion.py's controls.

Conditions, clusters and lesion operations come from core/lesion_ops.py,
shared with lesion.py, and are checked against the run's
lesion_prune_results pickle (condition names, cluster sizes, synapse
cluster membership, fixed k) before anything is scored. A condition larger
than the selected pool (the unresponsive class itself) falls back to the
"all" pool; `condition_pools` records the pool used by every condition.

Output: multiple_tasks_perf/<aname>/lesion_controls_<pool>_<aname>.pkl, a
pickle mirroring lesion_prune_results' ``lesion`` / ``random_lesion`` /
``lesion_unnorm`` / ``random_lesion_unnorm`` / ``mod_lesion`` records (same
shapes and condition order) plus the pool, repeat count, test seed and
per-task test-set checksums. lesion_plot.py --control-pool <pool> overlays
it on the lesion pickle in memory; nothing here is written into the lesion
pickle.

Run from the repository root:
    python multiple_task/lesion_controls.py --seed 921 --feature L21e4
    python multiple_task/lesion_controls.py --seed all --feature L21e4 --skip-completed
run_pipeline.py calls `main(seed, feature, ...)` as its third step.
"""
import argparse
import gc
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import lesion_cache
import lesion_ops
from lesion_cache import load_lesion_pickle
from modulation_variants import LESION_MODULATION_TYPES
from run_logging import tee_output

SCHEMA_VERSION = 1
# The pools, defined once for this writer and its readers (lesion_plot.py).
CONTROL_POOLS = lesion_cache.CONTROL_POOLS
DEFAULT_POOL = "responsive"
DEFAULT_REPEATS = 30
DEFAULT_TEST_SEED = 0
# Trials per task, as in lesion.py's single-cluster sweep.
TEST_N_BATCH = 200
PERF_DIR = Path("multiple_tasks_perf")
RUN_PATTERN = re.compile(r"everything_seed(\d+)_([^+]+)\+hidden300\+batch128\+angle$")


def controls_pickle_path(aname, pool, root=Path(".")):
    """Where this script keeps the controls of one run and pool: the name
    lesion_cache.controls_pickle_path derives from the lesion pickle's."""
    return lesion_cache.controls_pickle_path(lesion_results_path(aname, root), pool)


def lesion_results_path(aname, root=Path(".")):
    return Path(root) / PERF_DIR / aname / f"lesion_prune_results_{aname}.pkl"


# ── Pools ────────────────────────────────────────────────────────────────────

def responsive_neuron_pool(cluster_entry):
    """Units outside the entry's unresponsive class (every unit when it has none)."""
    units = lesion_ops.default_neuron_pool(cluster_entry)
    label = cluster_entry["unresponsive_label"]
    if label is None:
        return units
    return np.setdiff1d(units, np.asarray(cluster_entry["col_clusters"][label]))


def responsive_synapse_pool(mod_col_clusters, unresponsive_label, n_synapses):
    """Synapses outside the unresponsive class (all of them when it has none)."""
    synapses = np.arange(n_synapses)
    if unresponsive_label is None:
        return synapses
    return np.setdiff1d(synapses, np.asarray(mod_col_clusters[unresponsive_label]))


def choose_pool(pools, selected, size):
    """(pool name, pool) a control of `size` units draws from: the selected
    pool, or "all" when the condition is larger than it (the unresponsive
    class itself can only be matched from every unit)."""
    if size <= len(pools[selected]):
        return selected, pools[selected]
    return "all", pools["all"]


# ── Consistency with the lesion pickle ───────────────────────────────────────

def check_against_lesion_results(results, fixed_k, neuron_records, synapse_clusters):
    """Raise unless the clusters recomputed here are those lesion.py scored.

    `neuron_records` maps the lesion-pickle record key ("lesion",
    "lesion_unnorm") to (condition names, {name: size}); `synapse_clusters`
    maps a modulation type to (mod_col_clusters, unresponsive_label).
    """
    problems = []
    if int(results.get("fixed_k", -1)) != int(fixed_k):
        problems.append(f"fixed_k {results.get('fixed_k')} in the lesion pickle, {fixed_k} here")
    for key, (names, sizes) in neuron_records.items():
        saved = results[key]
        if list(saved["all_comb_names_lesion"]) != list(names):
            problems.append(f"{key}: condition names differ")
        saved_units = saved.get("lesion_units", {})
        if {name: int(saved_units.get(name, -1)) for name in names} != {n: int(s) for n, s in sizes.items()}:
            problems.append(f"{key}: cluster sizes differ")
    for type_key, (clusters, unresponsive) in synapse_clusters.items():
        for mode in lesion_ops.SYNAPSE_LESION_MODES:
            saved = results["mod_lesion"].get(f"{type_key}__{mode}")
            if saved is None:
                problems.append(f"{type_key}__{mode}: missing from the lesion pickle")
                continue
            saved_clusters = {int(label): list(map(int, members))
                              for label, members in saved["mod_col_clusters"].items()}
            if saved_clusters != {int(label): list(map(int, members))
                                  for label, members in clusters.items()}:
                problems.append(f"{type_key}__{mode}: synapse clusters differ")
            if saved.get("unresponsive_label") != unresponsive:
                problems.append(f"{type_key}__{mode}: unresponsive label differs")
    if problems:
        raise ValueError("lesion_controls does not match the lesion pickle: "
                         + "; ".join(problems))


# ── Sweeps and scoring ───────────────────────────────────────────────────────

RANDOM_RECORD_KEYS = {"lesion": "random_lesion", "lesion_unnorm": "random_lesion_unnorm"}


def neuron_sweep(cluster_info, variant, pools_by_side, selected):
    """The single-cluster sweep of one neuron variant: a dict with the
    conditions, their names and sizes, the pool each control draws from
    (`condition_pools`, None for the baselines) and the control cache keys
    (`keys`, None for the baselines, which are their own control)."""
    pre_n = len(cluster_info[lesion_ops.neuron_cluster_key("pre", variant)]["col_clusters"])
    post_n = len(cluster_info[lesion_ops.neuron_cluster_key("post", variant)]["col_clusters"])
    conditions, names = lesion_ops.neuron_conditions(pre_n, post_n)
    sizes = [lesion_ops.neuron_cluster_size(cluster_info, condition, variant)
             for condition in conditions]
    condition_pools = [None if size == 0 else choose_pool(pools_by_side[side], selected, size)[0]
                       for (side, _), size in zip(conditions, sizes)]
    keys = [None if size == 0 else (side, size, name)
            for (side, _), size, name in zip(conditions, sizes, condition_pools)]
    return {"variant": variant, "conditions": conditions, "names": names, "sizes": sizes,
            "condition_pools": condition_pools, "keys": keys}


def synapse_sweep(mod_col_clusters, pools, selected):
    """The sweep of one synapse clustering, laid out like `neuron_sweep`."""
    conditions, names = lesion_ops.synapse_conditions(mod_col_clusters)
    sizes = [0 if label is None else len(mod_col_clusters[label]) for _, label in conditions]
    condition_pools = [None if size == 0 else choose_pool(pools, selected, size)[0] for size in sizes]
    keys = [None if size == 0 else (size, name) for size, name in zip(sizes, condition_pools)]
    return {"conditions": conditions, "names": names, "sizes": sizes,
            "condition_pools": condition_pools, "keys": keys}


def forward_count(n_tasks, keys, repeats):
    """Forward passes a sweep costs: every condition once, plus `repeats` per
    distinct control key, per task."""
    return n_tasks * (len(keys) + repeats * len({key for key in keys if key is not None}))


def _score_sweep(model, sweep, lesion, test_sets, tasks, repeats, evaluate, log, label):
    """Per-task (accuracies, control means, control repeats) of one sweep;
    `evaluate` None means lesion_ops.evaluate_accuracy, looked up at call time."""
    if evaluate is None:
        evaluate = lesion_ops.evaluate_accuracy
    key_of = dict(zip(sweep["conditions"], sweep["keys"]))
    accs, means, raw = [], [], []
    for task in tasks:
        log(f"  [{label}] task {task}")
        task_accs, task_means, task_raw = lesion_ops.evaluate_conditions(
            model, test_sets[task], sweep["conditions"], lesion, key_of.get, repeats,
            evaluate=evaluate)
        accs.append(task_accs)
        means.append(task_means)
        raw.append(task_raw)
        gc.collect()
    return accs, means, raw


def score_neuron_sweep(model, cluster_info, sweep, pools_by_side, test_sets, tasks, repeats,
                       rng, evaluate=None, log=print):
    """Score a `neuron_sweep` on every task. Returns the lesion and
    random_lesion records in lesion_prune_results' layout, each carrying
    `condition_pools`."""
    variant = sweep["variant"]
    pool_of = {condition: (None if name is None else pools_by_side[condition[0]][name])
               for condition, name in zip(sweep["conditions"], sweep["condition_pools"])}

    def lesion(condition, random):
        saved, _ = lesion_ops.lesion_prepost_inplace(
            model, cluster_info, condition[1], condition[0], variant,
            random=random, pool=pool_of[condition], rng=rng)
        return saved

    accs, means, raw = _score_sweep(model, sweep, lesion, test_sets, tasks, repeats, evaluate,
                                    log, variant)
    labels, unresponsive_conditions = lesion_ops.unresponsive_neuron_conditions(cluster_info, variant)
    names = sweep["names"]
    lesion_record = {
        "ihtask_accs": accs, "all_comb_names_lesion": names, "all_tasks": list(tasks),
        "lesion_units": dict(zip(names, sweep["sizes"])),
        "unresponsive_labels": labels, "unresponsive_conditions": unresponsive_conditions,
        "condition_pools": sweep["condition_pools"],
    }
    random_record = {
        "ihrandomtask_accs": means, "ihrandomtask_accs_raw": raw,
        "all_comb_names_lesion": names, "all_tasks": list(tasks),
        "condition_pools": sweep["condition_pools"],
    }
    return lesion_record, random_record


def score_synapse_sweep(model, sweep, mod_col_clusters, unresponsive_label, n_pre, mode, pools,
                        test_sets, tasks, repeats, rng, evaluate=None,
                        log=print):
    """Score a `synapse_sweep` in one lesion mode on every task. Returns the
    mod_lesion record in lesion_prune_results' layout plus `condition_pools`."""
    pool_of = {condition: (None if name is None else pools[name])
               for condition, name in zip(sweep["conditions"], sweep["condition_pools"])}

    def lesion(condition, random):
        saved, _ = lesion_ops.lesion_modulation_inplace(
            model, condition[1], mod_col_clusters, n_pre, mode=mode,
            random=random, pool=pool_of[condition], rng=rng)
        return saved

    accs, means, raw = _score_sweep(model, sweep, lesion, test_sets, tasks, repeats, evaluate,
                                    log, mode)
    return {
        "modtask_accs": accs, "modrandomtask_accs": means, "modrandomtask_accs_raw": raw,
        "all_comb_names_mod": sweep["names"], "all_tasks": list(tasks),
        "mod_col_clusters": mod_col_clusters, "mod_lesion_mode": mode,
        "unresponsive_label": unresponsive_label, "condition_pools": sweep["condition_pools"],
    }


# ── Entry point ──────────────────────────────────────────────────────────────

def main(seed, feature, pool=DEFAULT_POOL, repeats=DEFAULT_REPEATS, mod_types=None,
         test_seed=DEFAULT_TEST_SEED, device=None, root=Path(".")):
    """Score the single-cluster lesions of one run with `pool` controls and
    write its lesion_controls pickle."""
    if pool not in CONTROL_POOLS:
        raise ValueError(f"pool must be one of {CONTROL_POOLS}, got {pool!r}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    aname = lesion_ops.run_name(seed, feature)
    print(f"[lesion_controls] {aname}: pool={pool}, repeats={repeats}, test seed={test_seed}, "
          f"device={device}")

    results = load_lesion_pickle(lesion_results_path(aname, root))
    cluster_info, cluster_info_mod = lesion_ops.load_cluster_info(aname, root)
    model, task_params_c, tasks = lesion_ops.load_network(aname, device, root)
    fixed_k, _ = lesion_ops.infer_fixed_k(cluster_info_mod)
    lesion_ops.add_fixed_k_neuron_clusters(cluster_info, fixed_k)
    variants = {"lesion": lesion_ops.neuron_variant(fixed_k, normalized=True),
                "lesion_unnorm": lesion_ops.neuron_variant(fixed_k, normalized=False)}
    n_post, n_pre = model.mp_layer1.W.shape

    # Synapse clusters of every requested type lesion.py scored, at the G
    # index its record names, so the clusters are the ones in the pickle.
    requested = list(LESION_MODULATION_TYPES) if mod_types is None else list(mod_types)
    synapse_clusters = {}
    for type_key in requested:
        record = results["mod_lesion"].get(f"{type_key}__zero_W")
        if type_key not in cluster_info_mod or record is None:
            print(f"  skipping {type_key}: not in cluster_info_mod or the lesion pickle")
            continue
        clusters, unresponsive, _ = lesion_ops.fixed_k_synapse_clusters(
            cluster_info_mod[type_key], record["mod_G_idx"], fixed_k)
        synapse_clusters[type_key] = (clusters, unresponsive)

    neuron_pools = {}
    for variant in variants.values():
        neuron_pools[variant] = {}
        for side in ("pre", "post"):
            entry = cluster_info[lesion_ops.neuron_cluster_key(side, variant)]
            neuron_pools[variant][side] = {"all": lesion_ops.default_neuron_pool(entry),
                                           "responsive": responsive_neuron_pool(entry)}
        print(f"  responsive pool {variant}: " + ", ".join(
            f"{side} {len(p['responsive'])}/{len(p['all'])}"
            for side, p in neuron_pools[variant].items()))
    synapse_pools = {}
    for type_key, (clusters, unresponsive) in synapse_clusters.items():
        synapse_pools[type_key] = {
            "all": np.arange(n_post * n_pre),
            "responsive": responsive_synapse_pool(clusters, unresponsive, n_post * n_pre)}
        print(f"  responsive pool {type_key}: "
              f"{len(synapse_pools[type_key]['responsive'])}/{n_post * n_pre} synapses")

    neuron_sweeps = {key: neuron_sweep(cluster_info, variant, neuron_pools[variant], pool)
                     for key, variant in variants.items()}
    synapse_sweeps = {type_key: synapse_sweep(clusters, synapse_pools[type_key], pool)
                      for type_key, (clusters, _) in synapse_clusters.items()}
    check_against_lesion_results(
        results, fixed_k,
        {key: (sweep["names"], dict(zip(sweep["names"], sweep["sizes"])))
         for key, sweep in neuron_sweeps.items()},
        synapse_clusters)
    n_forward = (sum(forward_count(len(tasks), sweep["keys"], repeats)
                     for sweep in neuron_sweeps.values())
                 + sum(len(lesion_ops.SYNAPSE_LESION_MODES)
                       * forward_count(len(tasks), sweep["keys"], repeats)
                       for sweep in synapse_sweeps.values()))
    print(f"  {n_forward} forward passes of {TEST_N_BATCH} trials planned")

    test_sets = lesion_ops.make_test_sets(task_params_c, tasks, TEST_N_BATCH, device, seed=test_seed)
    rng = np.random.default_rng(test_seed)
    output = {
        "schema_version": SCHEMA_VERSION, "aname": aname, "control_pool": pool,
        "repeat_num": repeats, "test_seed": test_seed, "test_n_batch": TEST_N_BATCH,
        "test_checksums": {task: lesion_ops.test_set_checksum(test_sets[task]) for task in tasks},
        "fixed_k": fixed_k, "all_tasks": list(tasks),
        "source_pickle": lesion_results_path(aname, root).name,
        "pools": {"neurons": {variant: {side: p["responsive"] for side, p in by_side.items()}
                              for variant, by_side in neuron_pools.items()},
                  "synapses": {type_key: p["responsive"] for type_key, p in synapse_pools.items()}},
        "mod_lesion": {},
    }
    for key, sweep in neuron_sweeps.items():
        print(f"[neurons] {sweep['variant']}")
        output[key], output[RANDOM_RECORD_KEYS[key]] = score_neuron_sweep(
            model, cluster_info, sweep, neuron_pools[sweep["variant"]], test_sets, tasks,
            repeats, rng)
    for type_key, sweep in synapse_sweeps.items():
        clusters, unresponsive = synapse_clusters[type_key]
        source = results["mod_lesion"][f"{type_key}__zero_W"]
        print(f"[synapses] {type_key}")
        for mode in lesion_ops.SYNAPSE_LESION_MODES:
            record = score_synapse_sweep(
                model, sweep, clusters, unresponsive, n_pre, mode, synapse_pools[type_key],
                test_sets, tasks, repeats, rng)
            record["mod_G_idx"] = source["mod_G_idx"]
            record["mod_k"] = source.get("mod_k")
            output["mod_lesion"][f"{type_key}__{mode}"] = record

    out_path = controls_pickle_path(aname, pool, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name and renamed, so an interrupted write never
    # leaves a truncated pickle that --skip-completed would count as done.
    partial = out_path.with_name(out_path.name + ".partial")
    with partial.open("wb") as stream:
        pickle.dump(output, stream)
    partial.replace(out_path)
    print(f"Saved {out_path} ({time.time() - t0:.0f}s)")
    return out_path


def discover_runs(root=Path(".")):
    """(seed, feature) of every run with a finished lesion pickle."""
    runs = []
    for directory in sorted((Path(root) / PERF_DIR).glob("everything_seed*")):
        match = RUN_PATTERN.match(directory.name)
        if match and lesion_results_path(directory.name, root).exists():
            runs.append((int(match.group(1)), match.group(2)))
    return runs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Score the single-cluster lesions again on a seeded test set with "
                    "size-matched random controls drawn from the responsive units "
                    "(or from every unit, --pool all). Needs the run's "
                    "lesion_prune_results pickle. Run from the repository root.")
    parser.add_argument("--seed", type=str, required=True,
                        help="Seed of the run (e.g. 921), or 'all' for every lesioned run "
                             "of --feature.")
    parser.add_argument("--feature", type=str, default=None,
                        help="Feature tag of the run(s) (e.g. 'L21e4'); required with --seed all.")
    parser.add_argument("--pool", choices=CONTROL_POOLS, default=DEFAULT_POOL,
                        help="Units the random controls are drawn from (default: responsive).")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"Random controls per (side, size) and task (default: {DEFAULT_REPEATS}).")
    parser.add_argument("--mod-types", nargs="+", choices=LESION_MODULATION_TYPES, default=None,
                        help="Synapse clustering types to score (default: every type the "
                             "lesion pickle holds).")
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED,
                        help=f"Seed of the shared test set and control draws (default: {DEFAULT_TEST_SEED}).")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Leave out runs whose controls pickle for --pool already exists.")
    args = parser.parse_args(argv)
    if args.seed == "all" and args.feature is None:
        parser.error("--seed all requires --feature so the batch is explicit")
    return args


def select_runs(args, root=Path(".")):
    """The (seed, feature) runs the arguments ask for, in seed order."""
    runs = discover_runs(root)
    if args.feature is not None:
        runs = [run for run in runs if run[1] == args.feature]
    if args.seed != "all":
        runs = [run for run in runs if run[0] == int(args.seed)]
        if len(runs) > 1:
            raise SystemExit(f"Ambiguous selection: {runs}; pass --feature.")
    if args.skip_completed:
        runs = [run for run in runs
                if not controls_pickle_path(lesion_ops.run_name(*run), args.pool, root).exists()]
    if not runs:
        raise SystemExit(f"No lesioned run matches seed={args.seed} feature={args.feature!r}"
                         + (" without a controls pickle" if args.skip_completed else ""))
    return runs


def cli():
    args = parse_args()
    runs = select_runs(args)
    print(f"Running lesion controls for {len(runs)} run(s): {runs}")
    for seed, feature in runs:
        main(seed, feature, pool=args.pool, repeats=args.repeats, mod_types=args.mod_types,
             test_seed=args.test_seed)


if __name__ == "__main__":
    with tee_output("lesion_controls"):
        cli()
