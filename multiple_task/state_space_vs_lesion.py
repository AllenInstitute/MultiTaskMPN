"""
Context-end task geometry vs causal task dependence.

Links the state-space and lesion analyses over the task pairs. For every run
that has both a state-space structure cache
(``state_space/state_space_structure_<aname>_noise0.01.pkl``, from
``state_space_structure.py``) and lesion caches (``multiple_tasks_norm/<aname>/``,
from ``lesion_plot.py`` with lesion.py's every-unit controls, the ``all``
pool, which every entry records), it relates, over the 105 task pairs, the distance
between the two tasks' full-dimensional context-end centroids (hidden state
and W * M, the saved distance matrices) to the similarity of their
lesion-effect profiles:

- neuron clusters: the saved Pearson correlation of the two tasks'
  lesion-effect profiles (random-control minus lesion accuracy) over all
  unnormalized input and hidden neuron clusters (``causal_similarity`` in
  ``causal_vs_activity_tasksim_hidden_<aname>.pkl``, the matrix the
  causal-vs-activity figures use);
- synapse clusters: the Pearson correlation of the two tasks' freeze_M
  lesion-effect profiles (``SYNAPSE_LESION_MODE``, the intervention of the
  paper's lesion heatmap) over the var-weighted unnormalized synapse clusters
  (``normalized_lesion_effects_<aname>.pkl``), computed here without the
  unresponsive class. That class holds most synapses, so its size-matched
  random control destroys the network while the class itself does little, a
  large negative effect for every task that would dominate the profiles; its
  label is read from the lesion pickle (``multiple_tasks_perf/<aname>/``),
  as the task-specificity analysis does.

Each (representation, cluster type) pair gets a two-sided Spearman task-label
permutation (Mantel) test (``N_PERMUTATIONS`` permutations, seed
``PERMUTATION_SEED``), because the pairs share tasks, and a descriptive OLS
guide line for the figure (as the causal-vs-activity scatter has; the
statistic stays the rank correlation). A negative rho means that tasks whose
context-end states lie closer depend on more similar clusters. Nothing is
recomputed from network activity; the script reads caches and runs in
seconds. Rerun it after ``state_space_structure.py`` or ``lesion_plot.py``
(the state-space batch run clears its output too).

Output: ``state_space/state_space_vs_lesion_results.pkl``, {aname: entry}
with the per-pair coordinates, the tests (null arrays included), the guide
lines and the run's L2 strength; ``paper_plot.py`` draws them in ``lesion`` mode
(``state_space_vs_lesion`` and ``state_space_vs_lesion_seeds``).
"""
from pathlib import Path
import pickle

import numpy as np

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from lesion_cache import load_lesion_pickle, normalize_lesion_cache, resolve_lesion_cache_path
from state_space_geometry import mantel_spearman_permutation

STATE_SPACE_DIR = Path("state_space")
LESION_NORM_ROOT = Path("multiple_tasks_norm")
LESION_PERF_ROOT = Path("multiple_tasks_perf")
NOISE_LEVEL = 0.01
N_PERMUTATIONS = 10000
PERMUTATION_SEED = 0
# 2: every (representation, cluster type) test has a saved OLS guide line.
SCHEMA_VERSION = 2
RESULT_NAME = "state_space_vs_lesion_results.pkl"
REPRESENTATIONS = ("hidden", "eff_mod")
CLUSTER_TYPES = ("neuron", "synapse")
NEURON_TASKSIM_NAME = "causal_vs_activity_tasksim_hidden"
NEURON_SIMILARITY_DEFINITION = "pearson_corr_of_task_lesion_effect_profiles_over_neuron_clusters"
# Synapse intervention of the synapse-cluster column: freeze_M (M held at its
# initial value, W kept), as in paper_plot's lesion heatmap
# (LESION_HEATMAP_MODE); test_state_space_vs_lesion checks the two agree.
SYNAPSE_LESION_MODE = "freeze_M"
SYNAPSE_EFFECT_ENTRY = f"modulation_all_var_weighted_unnormalized__{SYNAPSE_LESION_MODE}"
SYNAPSE_SIMILARITY_DEFINITION = (f"pearson_corr_of_task_{SYNAPSE_LESION_MODE}_lesion_effect_profiles"
                                 "_over_var_weighted_synapse_clusters")
SYNAPSE_CONDITION_PREFIX = "mod_c"
SYNAPSE_BASELINE_CONDITIONS = ("mod_nolesion", "mod_cNone")


def structure_cache_paths(state_space_dir=STATE_SPACE_DIR, noise_level=NOISE_LEVEL):
    """Every per-run structure cache of state_space_structure.py, sorted."""
    return sorted(Path(state_space_dir).glob(f"state_space_structure_*_noise{noise_level}.pkl"))


def load_structure_distances(path):
    """(aname, l2_info, tasks, {representation: distance matrix}) of one
    structure cache: the Euclidean distances between the full-dimensional
    context-end task centroids."""
    path = Path(path)
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    tasks = list(saved["all_rules"])
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"{path.name}: duplicate task names")
    distances = {}
    for representation in REPRESENTATIONS:
        matrix = np.asarray(saved["representations"][representation]["distances"], dtype=float)
        if matrix.shape != (len(tasks), len(tasks)) or not np.isfinite(matrix).all():
            raise ValueError(f"{path.name}: invalid {representation} distance matrix")
        distances[representation] = matrix
    return saved["aname"], float(saved["l2_info"]), tasks, distances


def neuron_cluster_similarity(aname, lesion_root=LESION_NORM_ROOT):
    """(tasks, matrix) with the saved Pearson correlation between the two
    tasks' z-scored lesion-effect profiles over the unnormalized input and
    hidden neuron clusters; None when the run has no such cache."""
    path = resolve_lesion_cache_path(Path(lesion_root) / aname / f"{NEURON_TASKSIM_NAME}_{aname}.pkl")
    if not path.exists():
        return None
    saved = load_lesion_pickle(path)
    if saved.get("schema_version") != 2 or saved.get("aname") != aname:
        raise ValueError(f"{path.name}: legacy cache or another run")
    if saved.get("y_definition") != NEURON_SIMILARITY_DEFINITION:
        raise ValueError(f"{path.name}: unexpected lesion similarity definition")
    tasks = list(saved["tasks"])
    matrix = np.asarray(saved["causal_similarity"], dtype=float)
    if matrix.shape != (len(tasks), len(tasks)) or not np.isfinite(matrix).all():
        raise ValueError(f"{path.name}: invalid causal similarity matrix")
    return tasks, matrix


def synapse_unresponsive_condition(aname, lesion_perf_root=LESION_PERF_ROOT):
    """Condition name (``mod_c<label>``) of the unresponsive synapse class in
    the run's var-weighted SYNAPSE_LESION_MODE lesion record, or None when none was
    flagged. Mirrors lesion_plot's ``_mod_unresponsive_label`` with
    ``legacy_last``: pickles written before the label was recorded take the
    last ``mod_c`` condition, the k + 1 class appended by every clustering."""
    path = Path(lesion_perf_root) / aname / f"lesion_prune_results_{aname}.pkl"
    if not resolve_lesion_cache_path(path).exists():
        raise FileNotFoundError(f"{aname}: no lesion pickle under {path.parent}")
    record = load_lesion_pickle(path)["mod_lesion"][SYNAPSE_EFFECT_ENTRY]
    if "unresponsive_label" in record:
        label = record["unresponsive_label"]
        return None if label is None else f"{SYNAPSE_CONDITION_PREFIX}{int(label)}"
    ids = sorted(int(name[len(SYNAPSE_CONDITION_PREFIX):]) for name in record["all_comb_names_mod"]
                 if name.startswith(SYNAPSE_CONDITION_PREFIX))
    return f"{SYNAPSE_CONDITION_PREFIX}{ids[-1]}" if ids else None


def synapse_cluster_similarity(aname, lesion_root=LESION_NORM_ROOT, lesion_perf_root=LESION_PERF_ROOT):
    """(tasks, matrix, excluded condition) with the Pearson correlation
    between the two tasks' SYNAPSE_LESION_MODE (freeze_M) lesion-effect
    profiles over the var-weighted unnormalized synapse clusters, leaving out
    the recorded unresponsive class; None when the run has no
    normalized-effects cache. An entry recording another mode is an error,
    never a substitute."""
    path = resolve_lesion_cache_path(Path(lesion_root) / aname / f"normalized_lesion_effects_{aname}.pkl")
    if not path.exists():
        return None
    saved = normalize_lesion_cache(load_lesion_pickle(path))
    if saved.get("schema_version") != 1 or saved.get("aname") != aname:
        raise ValueError(f"{path.name}: legacy cache or another run")
    entry = saved["entries"][SYNAPSE_EFFECT_ENTRY]
    if (entry.get("mod_lesion_mode", SYNAPSE_LESION_MODE) != SYNAPSE_LESION_MODE
            or entry.get("definition") != "random_minus_lesion"):
        raise ValueError(f"{path.name}: unexpected synapse lesion mode or effect definition")
    conditions = list(entry["conditions"])
    excluded = synapse_unresponsive_condition(aname, lesion_perf_root)
    if excluded is not None and excluded not in conditions:
        raise ValueError(f"{path.name}: unresponsive class {excluded} is not a saved condition")
    columns = [index for index, name in enumerate(conditions)
               if name not in SYNAPSE_BASELINE_CONDITIONS and name != excluded]
    if len(columns) < 3:
        raise ValueError(f"{path.name}: too few synapse clusters")
    effect = np.asarray(entry["effect"], dtype=float)[:, columns]
    tasks = list(entry["tasks"])
    if effect.shape[0] != len(tasks) or not np.isfinite(effect).all():
        raise ValueError(f"{path.name}: invalid synapse lesion effects")
    similarity = np.corrcoef(effect)
    if not np.isfinite(similarity).all():
        raise ValueError(f"{path.name}: a task has a constant synapse lesion profile")
    return tasks, similarity, excluded


def lesion_control_pool(aname, lesion_root=LESION_NORM_ROOT):
    """The random-control pool the run's lesion_plot.py caches were computed
    with (``control_pool`` of the normalized-effects cache; "all" for caches
    written before it was recorded, lesion.py's every-unit controls)."""
    path = resolve_lesion_cache_path(Path(lesion_root) / aname / f"normalized_lesion_effects_{aname}.pkl")
    return str(load_lesion_pickle(path).get("control_pool", "all"))


def lesion_similarities(aname, lesion_root=LESION_NORM_ROOT, lesion_perf_root=LESION_PERF_ROOT):
    """{cluster type: (tasks, matrix[, excluded])} of a run, or None when
    either lesion cache is missing."""
    neuron = neuron_cluster_similarity(aname, lesion_root)
    synapse = synapse_cluster_similarity(aname, lesion_root, lesion_perf_root)
    if neuron is None or synapse is None:
        return None
    return {"neuron": neuron, "synapse": synapse}


def _aligned(matrix, source_tasks, target_tasks):
    """``matrix`` with rows and columns reordered from ``source_tasks`` to
    ``target_tasks``, which must name the same tasks."""
    if set(source_tasks) != set(target_tasks) or len(source_tasks) != len(target_tasks):
        raise ValueError("lesion and state-space caches list different tasks")
    index = [source_tasks.index(task) for task in target_tasks]
    return matrix[np.ix_(index, index)]


def descriptive_trend_line(x, y):
    """Ordinary least-squares line through one panel's pairs, as a visual
    guide only (the same guide lesion_plot.py saves for the causal-vs-activity
    scatter). The reported association is the Spearman rho with its
    task-label permutation p; a rank correlation has no line in the raw
    coordinates, so this carries no inferential claim. None when x is
    constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-12:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return {"method": "ols", "slope": float(slope), "intercept": float(intercept),
            "role": "descriptive_guide_for_spearman_scatter"}


def analyze_run(aname, l2_info, tasks, distances, similarities,
                n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED, control_pool="all"):
    """The results entry of one run: per task pair (``pairs``, i < j in the
    structure cache's task order) the centroid distances per representation
    and the lesion-profile similarities per cluster type, and under
    ``tests[representation][cluster_type]`` the Spearman Mantel test of
    distance against similarity, under ``trend_lines`` (same keys) its
    descriptive OLS guide; ``control_pool`` names the random-control pool
    of the lesion caches the similarities came from."""
    upper = np.triu_indices(len(tasks), 1)
    entry = {
        "schema_version": SCHEMA_VERSION, "aname": aname, "l2_info": l2_info,
        "control_pool": str(control_pool),
        "tasks": list(tasks), "pairs": [(tasks[i], tasks[j]) for i, j in zip(*upper)],
        "n_permutations": int(n_permutations), "permutation_seed": int(seed),
        "distance_definition": "euclidean distance between full-dimensional context-end task centroids",
        "similarity_definitions": {
            "neuron": NEURON_SIMILARITY_DEFINITION,
            "synapse": SYNAPSE_SIMILARITY_DEFINITION,
        },
        "distances": {name: matrix[upper] for name, matrix in distances.items()},
        "similarities": {}, "excluded_conditions": {}, "tests": {name: {} for name in distances},
        "trend_lines": {name: {} for name in distances},
    }
    for cluster_type, (sim_tasks, matrix, *extra) in similarities.items():
        aligned = _aligned(matrix, sim_tasks, tasks)
        entry["similarities"][cluster_type] = aligned[upper]
        if extra:
            entry["excluded_conditions"][cluster_type] = extra[0]
        for representation, distance in distances.items():
            entry["tests"][representation][cluster_type] = mantel_spearman_permutation(
                distance, aligned, n_permutations=n_permutations, seed=seed)
            entry["trend_lines"][representation][cluster_type] = descriptive_trend_line(
                distance[upper], aligned[upper])
    return entry


def _format_p(p_value):
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def print_summary(results):
    """Per L2 cohort, representation and cluster type: the median rho over
    runs and the number of runs with permutation p below 0.05."""
    strengths = sorted({entry["l2_info"] for entry in results.values()})
    print("=" * 78)
    print(f"Context-end task distance vs lesion-profile similarity, {len(results)} runs")
    print("=" * 78)
    for strength in strengths:
        cohort = [entry for entry in results.values() if entry["l2_info"] == strength]
        print(f"\nL2 = {strength:g}: {len(cohort)} runs")
        for representation in REPRESENTATIONS:
            for cluster_type in CLUSTER_TYPES:
                tests = [entry["tests"][representation][cluster_type] for entry in cohort
                         if cluster_type in entry["tests"][representation]]
                if not tests:
                    continue
                rhos = np.array([test["rho"] for test in tests])
                significant = sum(test["p_value"] < 0.05 for test in tests)
                print(f"  {representation} distance vs {cluster_type} lesion similarity: "
                      f"rho median {np.median(rhos):+.3f} (range {rhos.min():+.3f} to "
                      f"{rhos.max():+.3f}), {significant}/{len(tests)} runs with p < 0.05")


def run_all(state_space_dir=STATE_SPACE_DIR, lesion_root=LESION_NORM_ROOT,
            lesion_perf_root=LESION_PERF_ROOT, n_permutations=N_PERMUTATIONS,
            seed=PERMUTATION_SEED, noise_level=NOISE_LEVEL):
    """Analyze every run with both caches, write the combined results and
    return them."""
    state_space_dir = Path(state_space_dir)
    paths = structure_cache_paths(state_space_dir, noise_level)
    if not paths:
        raise FileNotFoundError(
            f"No state_space_structure_*_noise{noise_level}.pkl in {state_space_dir}/; "
            "run python multiple_task/state_space_structure.py first.")
    results = {}
    for path in paths:
        aname, l2_info, tasks, distances = load_structure_distances(path)
        similarities = lesion_similarities(aname, lesion_root, lesion_perf_root)
        if similarities is None:
            print(f"{aname}: no lesion caches under {Path(lesion_root) / aname}; skipped")
            continue
        entry = analyze_run(aname, l2_info, tasks, distances, similarities,
                            n_permutations=n_permutations, seed=seed,
                            control_pool=lesion_control_pool(aname, lesion_root))
        results[aname] = entry
        test = entry["tests"]["eff_mod"]["synapse"]
        print(f"{aname}: W*M distance vs synapse lesion similarity rho = {test['rho']:+.3f} "
              f"({_format_p(test['p_value'])})")
    if not results:
        raise FileNotFoundError(
            f"No run has lesion caches under {lesion_root}/; run multiple_task/lesion_plot.py first.")
    out_path = state_space_dir / RESULT_NAME
    with out_path.open("wb") as stream:
        pickle.dump(results, stream)
    print(f"Saved {out_path}")
    print_summary(results)
    return results


if __name__ == "__main__":
    run_all()
