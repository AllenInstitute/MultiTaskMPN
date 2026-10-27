"""
How much of a synapse's task tuning is its endpoints' cluster identity?

The multi-task pipeline clusters the plastic synapses of ``mp_layer1`` on
their task-variance profiles (W * Var(M) over the task periods,
``multiple_task_analysis.py``) and, independently, the presynaptic
(input-layer) and postsynaptic (hidden-layer) neurons on theirs. The
over-membership analysis asks whether a synapse cluster is concentrated in a
few pre x post blocks. This script asks the complementary, quantitative
question over every active synapse: what share of the variance of the
synapse tuning profiles is explained by the clusters of the two endpoint
neurons, by the endpoint neurons themselves, and by the data-driven synapse
clusters beyond that?

Observations are the active synapses (the clustering's unresponsive synapse
class excluded, as the over-membership analysis does), responses one value
per task period in two versions of the feature: ``magnitude``,
``log1p(|W| * Var(M))``, the primary one, because the sign of W is a
per-synapse attribute rather than tuning (the signed clustering separates
positive- and negative-W synapses into sign-pure clusters, and the sign
alone accounts for about a fifth of the signed variance), and ``signed``,
``signed_log1p(W * Var(M))``, exactly what the clustering saw. Every R^2 is
the share of the total sum of squares about the grand mean captured by a
model's least-squares fit:

  pre                  cluster means of the presynaptic neuron's cluster
  post                 cluster means of the postsynaptic neuron's cluster
  additive             pre + post cluster effects (two-way, no interaction)
  block                pre x post cell means; interaction = block - additive
  synapse              the data-driven synapse clusters themselves (reference)
  block_plus_synapse   joint (block, synapse cluster) cells;
                       synapse_beyond_block = block_plus_synapse - block
  neuron level         pre neuron, post neuron and their additive fit: the
                       strongest "determined by the endpoints" model without
                       an interaction, which is saturated at one synapse per
                       neuron pair

Two label sets are evaluated per feature version: ``fixed_k20`` (the lesion
convention: pre, post and synapse clusters cut at k = FIXED_K plus the
unresponsive class) and ``tolerance`` (the tolerance-selected clusterings of
the analysis figures). Each version also records the share explained by the
sign of W alone (``sign_of_w_r2``, descriptive).

Nulls (N_PERMUTATIONS permutations, seed PERMUTATION_SEED, one-sided p of the
observed R^2 exceeding the null): the cluster models permute the cluster
labels over the neurons (pre labels among the input neurons, post labels
among the hidden neurons, cluster sizes kept), so they test the specific
clusters against arbitrary neuron groupings of the same sizes; ``synapse``
permutes the synapse labels over the active synapses; ``synapse_beyond_block``
permutes them within each block, which keeps every block's composition and
removes the mechanical gain of the extra cells; the neuron-level models
permute the synapse positions (the chance level of that many parameters).

Reads ``multiple_tasks_analysis/<aname>/cluster_info_<aname>.pkl`` and
``cluster_info_mod_<aname>.pkl``, writes
``multiple_tasks_analysis/<aname>/synapse_variance_decomposition_<aname>.pkl``
and prints a per-cohort summary. Nothing is recomputed from network activity.

    python multiple_task/synapse_variance_decomposition.py                 # every run
    python multiple_task/synapse_variance_decomposition.py --seed 921 --feature L21e4
    python multiple_task/synapse_variance_decomposition.py --seed all --feature L21e4
"""
from pathlib import Path
import argparse
import pickle
import re

import numpy as np
from scipy import sparse

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from clustering import fixed_k_col_clusters
from modulation_variants import signed_log1p

ANALYSIS_ROOT = Path("multiple_tasks_analysis")
MODULATION_TYPE = "modulation_all_var_weighted_unnormalized"
GROUPING_INDEX = 1                   # the G = 300 grouping, which the lesion experiment reads
GROUPING_NAME = "G=300"
FIXED_K = 20
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 0
SCHEMA_VERSION = 1
NEURON_SIDES = (("pre", "input_unnormalized"), ("post", "hidden_unnormalized"))
CLUSTER_MODELS = ("pre", "post", "additive", "block", "interaction", "synapse",
                  "block_plus_synapse", "synapse_beyond_block")
NEURON_MODELS = ("pre_neuron", "post_neuron", "additive_neuron")
FEATURE_VERSIONS = {
    "magnitude": ("log1p(|W| * Var(M))", lambda values: np.log1p(np.abs(values))),
    "signed": ("signed_log1p(W * Var(M))", signed_log1p),
}
PRIMARY_VERSION = "magnitude"
PRIMARY_LABEL_SET = "fixed_k20"
ANAME_PATTERN = re.compile(r"everything_seed(\d+)_([^+]+)\+hidden300\+batch128\+angle$")


def result_name(aname):
    return f"synapse_variance_decomposition_{aname}.pkl"


# ── Sums of squares ───────────────────────────────────────────────────────────

def relabel(labels):
    """0-based contiguous labels and their count."""
    unique, inverse = np.unique(np.asarray(labels), return_inverse=True)
    return inverse, int(unique.size)


def group_sums(labels, n_groups, values):
    """(n_groups, p) column sums of `values` within each label."""
    indicator = sparse.csr_matrix((np.ones(labels.size), (labels, np.arange(labels.size))),
                                  shape=(n_groups, labels.size))
    return indicator @ values


def explained_by_groups(labels, n_groups, centered):
    """Sum of squares captured by the group means of `labels`: sum over groups
    of n_g ||mean_g||^2, for `centered` rows (grand mean removed)."""
    counts = np.bincount(labels, minlength=n_groups)
    sums = group_sums(labels, n_groups, centered)
    occupied = counts > 0
    return float(((sums[occupied] ** 2).sum(axis=1) / counts[occupied]).sum())


def explained_by_additive(labels_a, n_a, labels_b, n_b, centered):
    """Sum of squares captured by the least-squares two-way additive model
    (an effect per level of a plus an effect per level of b, no interaction),
    through its normal equations: the Gram matrix of the two one-hot designs
    holds the level counts on the diagonal and their contingency table off
    it, and the explained sum of squares is trace(R^T G^+ R) for the moment
    matrix R of per-level sums. Exact for unbalanced designs."""
    counts_a = np.bincount(labels_a, minlength=n_a)
    counts_b = np.bincount(labels_b, minlength=n_b)
    cross = np.bincount(labels_a * n_b + labels_b, minlength=n_a * n_b).reshape(n_a, n_b)
    gram = np.block([[np.diag(counts_a), cross], [cross.T, np.diag(counts_b)]]).astype(float)
    moments = np.vstack([group_sums(labels_a, n_a, centered), group_sums(labels_b, n_b, centered)])
    coefficients = np.linalg.lstsq(gram, moments, rcond=None)[0]
    return float((coefficients * moments).sum())


def joint_labels(labels_a, n_a, labels_b, n_b):
    """One label per (a, b) cell, 0-based, with the cell count n_a * n_b."""
    return labels_a * n_b + labels_b, n_a * n_b


def cluster_model_fractions(centered, total_ss, pre, n_pre, post, n_post, synapse, n_synapse):
    """R^2 of every cluster-level model for one labeling of the synapses."""
    block, n_block = joint_labels(pre, n_pre, post, n_post)
    joint, n_joint = joint_labels(block, n_block, synapse, n_synapse)
    fractions = {
        "pre": explained_by_groups(pre, n_pre, centered) / total_ss,
        "post": explained_by_groups(post, n_post, centered) / total_ss,
        "additive": explained_by_additive(pre, n_pre, post, n_post, centered) / total_ss,
        "block": explained_by_groups(block, n_block, centered) / total_ss,
        "synapse": explained_by_groups(synapse, n_synapse, centered) / total_ss,
        "block_plus_synapse": explained_by_groups(joint, n_joint, centered) / total_ss,
    }
    fractions["interaction"] = fractions["block"] - fractions["additive"]
    fractions["synapse_beyond_block"] = fractions["block_plus_synapse"] - fractions["block"]
    return fractions


def neuron_model_fractions(centered, total_ss, pre_index, n_input, post_index, n_hidden):
    """R^2 of the neuron-level models: each synapse's presynaptic neuron, its
    postsynaptic neuron, and both additively."""
    return {
        "pre_neuron": explained_by_groups(pre_index, n_input, centered) / total_ss,
        "post_neuron": explained_by_groups(post_index, n_hidden, centered) / total_ss,
        "additive_neuron": explained_by_additive(pre_index, n_input, post_index, n_hidden,
                                                 centered) / total_ss,
    }


def permute_within_groups(labels, groups, rng):
    """`labels` permuted among the entries sharing a value of `groups`, so
    every group keeps its own multiset of labels."""
    by_group = np.argsort(groups, kind="stable")
    shuffled = np.lexsort((rng.random(groups.size), groups))
    permuted = np.empty_like(labels)
    permuted[shuffled] = labels[by_group]
    return permuted


def _test_summary(observed, null):
    null = np.asarray(null, dtype=float)
    return {
        "r2": float(observed),
        "null": null,
        "null_mean": float(null.mean()),
        "null_quantile_95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + np.count_nonzero(null >= observed)) / (1 + null.size)),
        "sided": "greater",
    }


# ── One run ───────────────────────────────────────────────────────────────────

def decompose(centered, pre_index, n_input, post_index, n_hidden, neuron_labels, synapse_labels,
              n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED):
    """The decomposition of one run for one label set.

    `centered`: (n_active, n_features) features about their grand mean;
    `pre_index` / `post_index`: each active synapse's input and hidden neuron;
    `neuron_labels`: {"pre": (n_input,) cluster label per input neuron,
    "post": (n_hidden,) per hidden neuron}; `synapse_labels`: (n_active,)
    synapse cluster label. Returns {model: test summary} for the cluster
    models and, under "neuron_level", the neuron-level models.
    """
    total_ss = float((centered ** 2).sum())
    if total_ss <= 0:
        raise ValueError("The synapse features do not vary.")
    pre_neuron, n_pre = relabel(neuron_labels["pre"])
    post_neuron, n_post = relabel(neuron_labels["post"])
    synapse, n_synapse = relabel(synapse_labels)
    pre, post = pre_neuron[pre_index], post_neuron[post_index]
    observed = cluster_model_fractions(centered, total_ss, pre, n_pre, post, n_post, synapse, n_synapse)
    block, n_block = joint_labels(pre, n_pre, post, n_post)

    rng = np.random.default_rng(seed)
    null = {model: np.empty(n_permutations) for model in CLUSTER_MODELS}
    for index in range(n_permutations):
        # Cluster labels permuted over the neurons: arbitrary groupings of the
        # same sizes.
        pre_perm = pre_neuron[rng.permutation(n_input)][pre_index]
        post_perm = post_neuron[rng.permutation(n_hidden)][post_index]
        block_perm, _ = joint_labels(pre_perm, n_pre, post_perm, n_post)
        null["pre"][index] = explained_by_groups(pre_perm, n_pre, centered) / total_ss
        null["post"][index] = explained_by_groups(post_perm, n_post, centered) / total_ss
        null["additive"][index] = explained_by_additive(pre_perm, n_pre, post_perm, n_post,
                                                        centered) / total_ss
        null["block"][index] = explained_by_groups(block_perm, n_block, centered) / total_ss
        # Synapse labels permuted over the active synapses, and within blocks.
        synapse_perm = synapse[rng.permutation(synapse.size)]
        null["synapse"][index] = explained_by_groups(synapse_perm, n_synapse, centered) / total_ss
        within = permute_within_groups(synapse, block, rng)
        joint_perm, n_joint = joint_labels(block, n_block, within, n_synapse)
        null["block_plus_synapse"][index] = explained_by_groups(joint_perm, n_joint, centered) / total_ss
    null["interaction"] = null["block"] - null["additive"]
    null["synapse_beyond_block"] = null["block_plus_synapse"] - observed["block"]

    result = {model: _test_summary(observed[model], null[model]) for model in CLUSTER_MODELS}
    result["n_groups"] = {"pre": n_pre, "post": n_post, "synapse": n_synapse,
                          "block_occupied": int(np.unique(block).size)}

    observed_neuron = neuron_model_fractions(centered, total_ss, pre_index, n_input, post_index, n_hidden)
    null_neuron = {model: np.empty(n_permutations) for model in NEURON_MODELS}
    for index in range(n_permutations):
        order = rng.permutation(pre_index.size)
        permuted = neuron_model_fractions(centered, total_ss, pre_index[order], n_input,
                                          post_index[order], n_hidden)
        for model in NEURON_MODELS:
            null_neuron[model][index] = permuted[model]
    result["neuron_level"] = {model: _test_summary(observed_neuron[model], null_neuron[model])
                              for model in NEURON_MODELS}
    return result


def _unresponsive_mask(result):
    """Boolean mask of the unresponsive synapses of a saved synapse clustering."""
    mask = result.get("col_unresponsive_mask")
    if mask is None:
        mask = np.asarray(result["col_tol_labels"]) == result["col_tol_k"] + 1
    return np.asarray(mask, dtype=bool)


def _labels_from_clusters(clusters, n_neurons):
    """(n_neurons,) label vector from a {label: neuron indices} dict covering
    every neuron exactly once."""
    labels = np.full(n_neurons, -1, dtype=int)
    for label, members in clusters.items():
        labels[np.asarray(members, dtype=int)] = int(label)
    if (labels < 0).any():
        raise ValueError("neuron clusters do not cover every neuron")
    return labels


def label_sets(cluster_info, synapse_result, n_input, n_hidden):
    """{set name: {"neuron": {"pre": labels, "post": labels}, "synapse": labels
    over all synapses, "unresponsive": {...}}} for the fixed-k and the
    tolerance-selected clusterings."""
    fixed, tolerance = {}, {}
    unresponsive = {"fixed_k20": {}, "tolerance": {}}
    for side, key in NEURON_SIDES:
        entry = cluster_info[key]
        n_neurons = n_input if side == "pre" else n_hidden
        clusters, label = fixed_k_col_clusters(entry, FIXED_K, return_unresponsive_label=True)
        fixed[side] = _labels_from_clusters(clusters, n_neurons)
        unresponsive["fixed_k20"][side] = label
        tolerance[side] = _labels_from_clusters(entry["col_clusters"], n_neurons)
        unresponsive["tolerance"][side] = entry.get("unresponsive_label")
    fixed_synapse = np.asarray(synapse_result["col_labels_by_k"][FIXED_K])
    unresponsive["fixed_k20"]["synapse"] = synapse_result.get("col_unresponsive_label_by_k", {}).get(FIXED_K)
    unresponsive["tolerance"]["synapse"] = synapse_result.get("col_unresponsive_label")
    return {
        "fixed_k20": {"neuron": fixed, "synapse": fixed_synapse, "unresponsive": unresponsive["fixed_k20"]},
        "tolerance": {"neuron": tolerance, "synapse": np.asarray(synapse_result["col_labels"]),
                      "unresponsive": unresponsive["tolerance"]},
    }


def load_run(aname, analysis_root=ANALYSIS_ROOT):
    """(raw features (synapses, periods), pre_index, post_index, active mask,
    label sets, metadata) of one run from its cluster_info and
    cluster_info_mod pickles."""
    run_dir = Path(analysis_root) / aname
    with (run_dir / f"cluster_info_{aname}.pkl").open("rb") as stream:
        cluster_info = pickle.load(stream)
    with (run_dir / f"cluster_info_mod_{aname}.pkl").open("rb") as stream:
        cluster_info_mod = pickle.load(stream)
    entry = cluster_info_mod[MODULATION_TYPE]
    names = list(entry["result_all_name_lst"])
    if names[GROUPING_INDEX] != GROUPING_NAME:
        raise ValueError(f"{aname}: grouping {GROUPING_INDEX} is {names[GROUPING_INDEX]}, "
                         f"not {GROUPING_NAME}")
    synapse_result = entry["result_all_lst"][GROUPING_INDEX]
    raw = np.asarray(entry["cell_vars_rules_sorted_norm"], dtype=float)      # (periods, synapses)
    n_hidden, n_input = (int(size) for size in entry["mod_shape"])            # (post, pre)
    if raw.shape[1] != n_hidden * n_input:
        raise ValueError(f"{aname}: {raw.shape[1]} synapse features for a {entry['mod_shape']} W")
    flat = np.arange(raw.shape[1])
    pre_index, post_index = flat % n_input, flat // n_input                  # C-order (post, pre)
    active = ~_unresponsive_mask(synapse_result)
    sets = label_sets(cluster_info, synapse_result, n_input, n_hidden)
    metadata = {
        "modulation_type": MODULATION_TYPE, "grouping": GROUPING_NAME,
        "n_features": int(raw.shape[0]),
        "feature_names": [str(name) for name in entry["tb_break_name"]],
        "n_input": n_input, "n_hidden": n_hidden,
        "n_synapses": int(raw.shape[1]), "n_active": int(active.sum()),
    }
    return raw.T, pre_index, post_index, active, sets, metadata


def analyze_run(aname, analysis_root=ANALYSIS_ROOT, n_permutations=N_PERMUTATIONS,
                seed=PERMUTATION_SEED):
    """The results entry of one run (see the module docstring)."""
    raw, pre_index, post_index, active, sets, metadata = load_run(aname, analysis_root)
    match = ANAME_PATTERN.match(aname)
    entry = {
        "schema_version": SCHEMA_VERSION, "aname": aname,
        "seed": int(match.group(1)) if match else None,
        "feature": match.group(2) if match else None,
        "n_permutations": int(n_permutations), "permutation_seed": int(seed),
        "primary": {"version": PRIMARY_VERSION, "label_set": PRIMARY_LABEL_SET},
        "versions": {}, **metadata,
    }
    # W * Var(M) carries the sign of W in every period, so a synapse's sign is
    # read off its raw features (Var(M) >= 0).
    negative_w = (raw[active].mean(axis=1) < 0).astype(int)
    for version, (transform_name, transform) in FEATURE_VERSIONS.items():
        features = transform(raw[active])
        centered = features - features.mean(axis=0)
        total_ss = float((centered ** 2).sum())
        results = {"feature_transform": transform_name, "label_sets": {},
                   "sign_of_w_r2": explained_by_groups(negative_w, 2, centered) / total_ss,
                   "n_negative_w": int(negative_w.sum())}
        for name, labels in sets.items():
            result = decompose(centered, pre_index[active], metadata["n_input"], post_index[active],
                               metadata["n_hidden"], labels["neuron"], labels["synapse"][active],
                               n_permutations=n_permutations, seed=seed)
            result["unresponsive"] = labels["unresponsive"]
            results["label_sets"][name] = result
        entry["versions"][version] = results
    return entry


def _format(summary):
    return f"{summary['r2'] * 100:5.1f}% (null {summary['null_mean'] * 100:4.1f}%, p = {summary['p_value']:.3f})"


def _model_tests(result):
    """[(model, test summary)] of one decomposition, cluster models first."""
    return ([(model, result[model]) for model in CLUSTER_MODELS]
            + [(model, result["neuron_level"][model]) for model in NEURON_MODELS])


def print_run(entry):
    for version, results in entry["versions"].items():
        print(f"  {version} features ({results['feature_transform']}): sign of W alone "
              f"{results['sign_of_w_r2'] * 100:.1f}%, {results['n_negative_w']} negative-W synapses")
        for name, result in results["label_sets"].items():
            groups = result["n_groups"]
            print(f"  [{version} / {name}] pre {groups['pre']}, post {groups['post']}, synapse "
                  f"{groups['synapse']} clusters, {groups['block_occupied']} occupied blocks")
            for model, test in _model_tests(result):
                print(f"    {model:>22}: {_format(test)}")


def print_summary(entries, version=PRIMARY_VERSION, label_set=PRIMARY_LABEL_SET):
    """Per cohort (feature tag): the median R^2 of every model and the
    number of runs whose permutation p is below 0.05."""
    features = sorted({entry["feature"] for entry in entries}, key=str)
    print("=" * 78)
    print(f"Synapse tuning variance explained by endpoint clusters [{version} / {label_set}], "
          f"{len(entries)} runs")
    print("=" * 78)
    for feature in features:
        cohort = [entry for entry in entries if entry["feature"] == feature]
        results = [entry["versions"][version]["label_sets"][label_set] for entry in cohort]
        print(f"\n{feature}: {len(cohort)} runs, "
              f"{int(np.median([entry['n_active'] for entry in cohort]))} active synapses (median), "
              f"sign of W alone {np.median([entry['versions'][version]['sign_of_w_r2'] for entry in cohort]) * 100:.1f}%")
        for model in CLUSTER_MODELS + NEURON_MODELS:
            tests = [dict(_model_tests(result))[model] for result in results]
            values = np.array([test["r2"] for test in tests]) * 100
            significant = sum(test["p_value"] < 0.05 for test in tests)
            print(f"  {model:>22}: median {np.median(values):5.1f}% "
                  f"(range {values.min():5.1f} to {values.max():5.1f}), "
                  f"{significant}/{len(tests)} runs with p < 0.05")
        residual = [100 * (1 - result["block"]["r2"]) for result in results]
        print(f"  {'residual (1 - block)':>22}: median {np.median(residual):5.1f}%")


def candidate_runs(analysis_root=ANALYSIS_ROOT):
    """[(seed, feature, aname)] of every run whose analysis directory holds
    both clustering pickles."""
    runs = []
    for directory in sorted(Path(analysis_root).glob("everything_seed*")):
        match = ANAME_PATTERN.match(directory.name)
        if not match:
            continue
        aname = directory.name
        if ((directory / f"cluster_info_{aname}.pkl").exists()
                and (directory / f"cluster_info_mod_{aname}.pkl").exists()):
            runs.append((int(match.group(1)), match.group(2), aname))
    return runs


def run(anames, analysis_root=ANALYSIS_ROOT, n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED):
    """Analyze the named runs, write one pickle per run and print the summary."""
    entries = []
    for aname in anames:
        print(f"{aname}")
        entry = analyze_run(aname, analysis_root, n_permutations=n_permutations, seed=seed)
        print_run(entry)
        out_path = Path(analysis_root) / aname / result_name(aname)
        with out_path.open("wb") as stream:
            pickle.dump(entry, stream)
        print(f"  Saved {out_path}")
        entries.append(entry)
    for version in FEATURE_VERSIONS:
        print_summary(entries, version=version)
    return entries


def select_runs(seed, feature, runs):
    """The candidate runs chosen by the CLI arguments (None = no filter)."""
    selected = [run_ for run_ in runs
                if (feature is None or run_[1] == feature) and (seed is None or run_[0] == seed)]
    if not selected:
        available = ", ".join(f"seed{s}/{f}" for s, f, _ in runs) or "none found"
        raise SystemExit(f"No run matches seed={seed!r}, feature={feature!r}. Available: {available}")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decompose the synapse tuning variance by endpoint cluster identity "
                    "(reads multiple_tasks_analysis/, cheap). Run from the repository root.")
    parser.add_argument("--seed", type=str, default=None,
                        help="Seed of the run (e.g. 921), or 'all' for every run matching --feature; "
                             "omitted: every analyzed run.")
    parser.add_argument("--feature", type=str, default=None, help="Feature tag (e.g. L21e4).")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()
    if args.seed == "all" and args.feature is None:
        raise SystemExit("--seed all requires --feature so the batch run is explicit.")
    seed_filter = None if args.seed in (None, "all") else int(args.seed)
    chosen = select_runs(seed_filter, args.feature, candidate_runs())
    print(f"Analyzing {len(chosen)} run(s)")
    run([aname for _, _, aname in chosen], n_permutations=args.n_permutations)
