"""In-place lesions of a trained multi-task MPN and the bookkeeping around them.

Shared by ``multiple_task/lesion.py`` (the full lesion experiment: single,
combined and modulation lesions plus pruning) and
``multiple_task/lesion_controls.py`` (the same single-cluster lesions scored
again on a seeded test set, with size-matched random controls drawn from the
responsive units only). Both scripts therefore lesion exactly the same
clusters with the same operations; the difference between them is where the
random control units come from (`pool`) and which trials are scored.

Conventions (shared with lesion_plot.py):
- Neuron clusters are the fixed-k recut of the saved dendrograms
  (``clustering.fixed_k_col_clusters``), injected into ``cluster_info`` under
  ``"<base>_k<fixed_k>"`` keys; ``"pre"`` lesions remove input-side (embedding)
  units, ``"post"`` lesions hidden units. Condition names are
  ``pre_nolesion, pre_c1, ..., post_nolesion, post_c1, ...``.
- Synapse clusters are the fixed-k labels saved per modulation clustering
  type; flat synapse index ``k`` of the (post, pre) plastic layer is
  ``post = k // n_pre, pre = k % n_pre``. Conditions are ``mod_nolesion,
  mod_c<label>, ...``; ``"zero_W"`` zeroes the static weight at the cluster's
  synapses, ``"freeze_M"`` keeps W and holds M at its trial-onset value there.
- A size-matched random control draws the lesioned units uniformly without
  replacement from `pool` (default: every unit of that side or layer, the
  historical design). Controls depend only on the control key (side or
  layer, size, pool), never on the cluster's identity, so conditions sharing
  a key share one set of control draws.
"""
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch

import clustering
import mpn
import mpn_tasks

# Fallback for cluster_info_mod pickles written before the fixed-k key existed;
# must match FIXED_K_OM in multiple_task_analysis.py.
FALLBACK_FIXED_K = 20
NEURON_CLUSTER_BASES = ("input_normalized", "hidden_normalized",
                        "input_unnormalized", "hidden_unnormalized")
SYNAPSE_LESION_MODES = ("zero_W", "freeze_M")
NEURON_BASELINE_NAMES = {"pre_cNone": "pre_nolesion", "post_cNone": "post_nolesion"}


def run_name(seed, feature):
    """Identifier of a multi-task run, shared by its checkpoint and analysis files."""
    return f"everything_seed{seed}_{feature}+hidden300+batch128+angle"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_network(aname, device, root=Path(".")):
    """(model, task_params_c, tasks) of a saved run: the checkpoint in eval
    mode on `device`, the converted task parameters and the task list."""
    root = Path(root)
    with (root / "multiple_tasks" / f"param_{aname}_param.json").open() as stream:
        raw_cfg = json.load(stream)
    checkpoint = torch.load(root / "multiple_tasks" / f"savednet_{aname}.pt",
                            map_location=device)
    model = mpn.DeepMultiPlasticNet(checkpoint["net_params"], verbose=False, forzihan=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.to(device)
    task_params_c, _, _ = mpn_tasks.convert_and_init_multitask_params(
        (raw_cfg["task_params"], raw_cfg["train_params"], raw_cfg["net_params"]))
    return model, task_params_c, list(task_params_c["rules"])


def load_cluster_info(aname, root=Path(".")):
    """(cluster_info, cluster_info_mod) pickles written by multiple_task_analysis.py."""
    analysis_dir = Path(root) / "multiple_tasks_analysis" / aname
    with (analysis_dir / f"cluster_info_{aname}.pkl").open("rb") as stream:
        cluster_info = pickle.load(stream)
    with (analysis_dir / f"cluster_info_mod_{aname}.pkl").open("rb") as stream:
        cluster_info_mod = pickle.load(stream)
    return cluster_info, cluster_info_mod


# ── Fixed-k clusters ─────────────────────────────────────────────────────────

def infer_fixed_k(cluster_info_mod):
    """(fixed_k, source key) read from the first modulation type's
    ``global_assignment_fixed_k<N>`` key, or (FALLBACK_FIXED_K, None)."""
    first_type = next(iter(cluster_info_mod))
    keys = [key for key in cluster_info_mod[first_type]
            if key.startswith("global_assignment_fixed_k")]
    if not keys:
        return FALLBACK_FIXED_K, None
    return int(re.search(r"fixed_k(\d+)", keys[0]).group(1)), keys[0]


def neuron_variant(fixed_k, normalized=True):
    """Variant tag of the fixed-k neuron clusters, e.g. ``normalized_k20``."""
    return f"{'normalized' if normalized else 'unnormalized'}_k{fixed_k}"


def neuron_cluster_key(preorpost, variant):
    """cluster_info key holding the clusters a ``"pre"`` / ``"post"`` lesion reads."""
    if preorpost not in ("pre", "post"):
        raise ValueError(f"preorpost must be 'pre' or 'post', got {preorpost!r}")
    return f"{'input' if preorpost == 'pre' else 'hidden'}_{variant}"


def add_fixed_k_neuron_clusters(cluster_info, fixed_k):
    """Inject the fixed-k recut of every neuron clustering base into
    `cluster_info` under ``"<base>_k<fixed_k>"`` and return those keys.

    Each entry carries the recut ``col_clusters`` ({1-based label: neuron
    indices}), the base's row clusters, period breaks, task-variance matrix
    and clustering result, and ``unresponsive_label`` (the key of the
    silent-neuron class, or None), so downstream code never has to assume the
    last cluster is the unresponsive one.
    """
    keys = []
    for base in NEURON_CLUSTER_BASES:
        clusters, unresponsive = clustering.fixed_k_col_clusters(
            cluster_info[base], fixed_k, return_unresponsive_label=True)
        key = f"{base}_k{fixed_k}"
        cluster_info[key] = {
            "col_clusters": clusters,
            "row_clusters": cluster_info[base]["row_clusters"],
            "tb_break_name": cluster_info[base]["tb_break_name"],
            "cell_vars_rules_sorted_norm": cluster_info[base]["cell_vars_rules_sorted_norm"],
            "result": cluster_info[base]["result"],
            "unresponsive_label": unresponsive,
        }
        keys.append(key)
    return keys


def unresponsive_neuron_conditions(cluster_info, variant):
    """({"pre": label|None, "post": label|None}, [condition names]) of the
    unresponsive input / hidden class of one neuron variant."""
    labels = {side: cluster_info[neuron_cluster_key(side, variant)]["unresponsive_label"]
              for side in ("pre", "post")}
    conditions = [f"{side}_c{label}" for side, label in labels.items() if label is not None]
    return labels, conditions


def fixed_k_synapse_clusters(mod_type_entry, G_index, fixed_k):
    """Synapse clusters of one modulation clustering type at the fixed k.

    `mod_type_entry` is ``cluster_info_mod[type_key]`` and `G_index` selects
    its per-G clustering solution (``result_all_lst``). Returns
    ``(mod_col_clusters, unresponsive_label, k_used)`` with
    ``mod_col_clusters`` = {label: [flat synapse indices]}. The fixed-k labels
    are used when saved; otherwise the smallest saved k, else the optimal k
    (the same fallbacks lesion.py has always applied). The unresponsive
    label comes from the saved mask, or the k + 1 convention for pickles
    written before the mask existed.
    """
    result = mod_type_entry["result_all_lst"][G_index]
    labels_by_k = result.get("col_labels_by_k") or {}
    if fixed_k in labels_by_k:
        k_used, labels = fixed_k, labels_by_k[fixed_k]
    elif labels_by_k:
        k_used = sorted(labels_by_k)[0]
        labels = labels_by_k[k_used]
    else:
        k_used, labels = result["col_k"], result["col_labels"]
    labels = np.asarray(labels)
    mask = result.get("col_unresponsive_mask")
    if mask is None:
        mask = labels == (int(k_used) + 1)
    unresponsive = clustering.unresponsive_label(labels, mask)
    mod_col_clusters = {}
    for flat_index, label in enumerate(labels):
        mod_col_clusters.setdefault(int(label), []).append(flat_index)
    return mod_col_clusters, unresponsive, int(k_used)


# ── Conditions ───────────────────────────────────────────────────────────────

def neuron_conditions(pre_n, post_n):
    """(conditions, names) of the single-cluster sweep: the no-lesion baseline
    and every cluster of each side, as ``("pre"|"post", label|None)``."""
    conditions = ([("pre", None)] + [("pre", i) for i in range(1, pre_n + 1)]
                  + [("post", None)] + [("post", i) for i in range(1, post_n + 1)])
    names = [NEURON_BASELINE_NAMES.get(f"{side}_c{label}", f"{side}_c{label}")
             for side, label in conditions]
    return conditions, names


def synapse_conditions(mod_col_clusters):
    """(conditions, names) of one synapse clustering: the baseline, then every
    cluster by label, as ``("mod", label|None)``."""
    labels = sorted(mod_col_clusters)
    conditions = [("mod", None)] + [("mod", label) for label in labels]
    names = ["mod_nolesion"] + [f"mod_c{label}" for label in labels]
    return conditions, names


def neuron_cluster_size(cluster_info, condition, variant):
    """Number of neurons a single-cluster condition removes (0 for the baseline)."""
    side, label = condition
    if label is None:
        return 0
    return len(cluster_info[neuron_cluster_key(side, variant)]["col_clusters"][label])


# ── In-place lesions ─────────────────────────────────────────────────────────

def _draw(pool, size, rng):
    """`size` distinct entries of `pool`, from `rng` or numpy's global state."""
    pool = np.asarray(pool)
    if size > pool.size:
        raise ValueError(f"cannot draw {size} units from a pool of {pool.size}")
    chooser = np.random if rng is None else rng
    return np.asarray(chooser.choice(pool, size=size, replace=False))


def default_neuron_pool(cluster_entry):
    """Every neuron the clustering assigned (the historical control pool)."""
    return np.arange(max(np.max(indices) for indices in
                         cluster_entry["col_clusters"].values()) + 1)


def lesion_neurons_inplace(net, neuron_index, preorpost):
    """Zero the connections of the given units in place and return the saved
    state `restore_lesion` needs. ``"pre"`` removes input-side (embedding)
    units: their input projection and their column of the plastic weights;
    ``"post"`` removes hidden units: their plastic-weight row, bias and
    readout column."""
    device = net.W_output.device
    neuron_index = torch.as_tensor(np.asarray(neuron_index), dtype=torch.long, device=device)
    with torch.no_grad():
        if preorpost == "pre":
            saved = {"preorpost": "pre", "neuron_index": neuron_index,
                     "W_initial": net.W_initial_linear.weight.data[neuron_index, :].clone(),
                     "mp_W_col": net.mp_layer1.W[:, neuron_index].clone()}
            net.W_initial_linear.weight.data[neuron_index, :] = 0.0
            net.mp_layer1.W[:, neuron_index] = 0.0
        elif preorpost == "post":
            saved = {"preorpost": "post", "neuron_index": neuron_index,
                     "mp_W_row": net.mp_layer1.W[neuron_index, :].clone(),
                     "mp_b": net.mp_layer1.b[neuron_index].clone(),
                     "W_output": net.W_output[:, neuron_index].clone()}
            net.mp_layer1.W[neuron_index, :] = 0.0
            net.mp_layer1.b[neuron_index] = 0.0   # the plastic layer's bias is postsynaptic
            net.W_output[:, neuron_index] = 0.0
        else:
            raise ValueError(f"preorpost must be 'pre' or 'post', got {preorpost!r}")
    return saved


def lesion_prepost_inplace(net, cluster_info, cluster_index, preorpost, variant,
                           random=False, pool=None, rng=None):
    """Lesion one neuron cluster (or, with `random`, as many units drawn from
    `pool`, default every unit of that side). Returns (saved_state, n_units);
    the baseline ``cluster_index=None`` is a no-op returning ({}, 0)."""
    if cluster_index is None:
        return {}, 0
    entry = cluster_info[neuron_cluster_key(preorpost, variant)]
    neuron_index = np.asarray(entry["col_clusters"][cluster_index])
    if random:
        neuron_index = _draw(default_neuron_pool(entry) if pool is None else pool,
                             len(neuron_index), rng)
    return lesion_neurons_inplace(net, neuron_index, preorpost), len(neuron_index)


def lesion_synapses_inplace(net, flat_indices, n_pre, mode):
    """Lesion the given flat synapse indices of the plastic layer in place:
    ``"zero_W"`` zeroes their static weight, ``"freeze_M"`` holds their
    modulation at its trial-onset value. Returns the saved state."""
    flat_indices = np.asarray(flat_indices, dtype=int)
    device = net.mp_layer1.W.device
    post_t = torch.tensor(flat_indices // n_pre, dtype=torch.long, device=device)
    pre_t = torch.tensor(flat_indices % n_pre, dtype=torch.long, device=device)
    if mode == "zero_W":
        with torch.no_grad():
            saved = {"preorpost": "modulation", "pre_t": pre_t, "post_t": post_t,
                     "W_vals": net.mp_layer1.W[post_t, pre_t].clone()}
            net.mp_layer1.W[post_t, pre_t] = 0.0
    elif mode == "freeze_M":
        net.mp_layer1.set_plasticity_freeze(post_t, pre_t)
        saved = {"preorpost": "modulation_freeze", "pre_t": pre_t, "post_t": post_t}
    else:
        raise ValueError(f"Unknown synapse lesion mode: {mode!r}")
    return saved


def lesion_modulation_inplace(net, cluster_index, mod_col_clusters, n_pre, mode="zero_W",
                              random=False, pool=None, rng=None):
    """Lesion one synapse cluster (or, with `random`, as many synapses drawn
    from `pool`, default all of the plastic layer). Returns (saved_state,
    n_synapses); the baseline ``cluster_index=None`` is a no-op."""
    if cluster_index is None:
        return {}, 0
    flat_indices = np.asarray(mod_col_clusters[cluster_index], dtype=int)
    if random:
        if pool is None:
            pool = np.arange(net.mp_layer1.W.shape[0] * n_pre)
        flat_indices = _draw(pool, len(flat_indices), rng)
    return lesion_synapses_inplace(net, flat_indices, n_pre, mode), len(flat_indices)


def restore_lesion(net, saved):
    """Undo a lesion from its saved state (a no-op for the baseline's {})."""
    if not saved:
        return
    with torch.no_grad():
        kind = saved["preorpost"]
        if kind == "pre":
            index = saved["neuron_index"]
            net.W_initial_linear.weight.data[index, :] = saved["W_initial"]
            net.mp_layer1.W[:, index] = saved["mp_W_col"]
        elif kind == "post":
            index = saved["neuron_index"]
            net.mp_layer1.W[index, :] = saved["mp_W_row"]
            net.mp_layer1.b[index] = saved["mp_b"]
            net.W_output[:, index] = saved["W_output"]
        elif kind == "modulation":
            net.mp_layer1.W[saved["post_t"], saved["pre_t"]] = saved["W_vals"]
        elif kind == "modulation_freeze":
            net.mp_layer1.clear_plasticity_freeze()
        else:
            raise ValueError(f"Unknown saved lesion state: {kind!r}")


# ── Test sets and scoring ────────────────────────────────────────────────────

def make_test_sets(task_params_c, tasks, n_trials, device, seed=None):
    """One (input, output, mask) batch of `n_trials` trials per task, shared by
    every condition. With `seed` the trials are reproducible (numpy and torch
    are seeded before drawing); without it they are not, as in lesion.py."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    task_params_c["hp"]["batch_size_train"] = n_trials
    test_sets = {}
    for task in tasks:
        test_data, _ = mpn_tasks.generate_trials_wrap(
            task_params_c, n_trials, rules=[task], mode_input="random_batch",
            device=device, verbose=False)
        test_sets[task] = tuple(test_data)
    return test_sets


def test_set_checksum(test_set):
    """SHA-1 of a test set's input batch, to tell apart test sets drawn with
    different seeds (or without one) when their accuracies are compared."""
    test_input = test_set[0]
    array = (test_input.detach().cpu().numpy() if torch.is_tensor(test_input)
             else np.asarray(test_input))
    return hashlib.sha1(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def evaluate_accuracy(model, test_set):
    """Accuracy of the model as it currently stands on one task's test set."""
    test_input, test_output, test_mask = test_set
    with torch.inference_mode():
        net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode="minimal")
        acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input,
                                   isvalid=True, mode=model.acc_measure)
    del net_out
    return float(acc.item())


def evaluate_conditions(model, test_set, conditions, lesion, control_key, repeats,
                        evaluate=evaluate_accuracy):
    """Score every condition and its size-matched random control on one test set.

    `lesion(condition, random)` applies the condition (or a random control of
    its size) in place and returns the saved state; `control_key(condition)`
    is the hashable key the control draws depend on (side or layer, size and
    pool), so conditions with the same key share one set of `repeats` draws.
    A key of None marks a condition that is its own control (the no-lesion
    baseline): its accuracy is repeated instead of scoring `repeats` no-op
    lesions. Returns (accuracies, control means, control repeats) as lists
    over the conditions; the repeats of same-key conditions are identical
    copies.
    """
    accuracies, control_means, control_raw = [], [], []
    cache = {}
    for condition in conditions:
        saved = lesion(condition, False)
        accuracies.append(evaluate(model, test_set))
        restore_lesion(model, saved)
        key = control_key(condition)
        if key is None:
            control_means.append(accuracies[-1])
            control_raw.append([accuracies[-1]] * repeats)
            continue
        if key not in cache:
            repeats_acc = []
            for _ in range(repeats):
                saved = lesion(condition, True)
                repeats_acc.append(evaluate(model, test_set))
                restore_lesion(model, saved)
            cache[key] = repeats_acc
        control_means.append(float(np.mean(cache[key])))
        control_raw.append(list(cache[key]))
    return accuracies, control_means, control_raw
