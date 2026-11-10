"""
Context-end task geometry vs causal task dependence.

Links the state-space and lesion analyses over the task pairs. For every run
that has both a state-space structure cache
(``state_space/state_space_structure_<aname>_noise0.01.pkl``, from
``state_space_structure.py``) and lesion caches (``multiple_tasks_norm/<aname>/``,
from ``lesion_plot.py`` with lesion.py's every-unit controls, the ``all``
pool, which every entry records), it relates, over the 105 task pairs, the distance
between the two tasks' full-dimensional context-end centroids (hidden state
and W * M, the saved distance matrices) to two measures of how alike their
lesion-effect profiles (random-control minus lesion accuracy, one value per
cluster) are, the two the OM-vs-lesion scatter compares its clusters with:

- the Pearson correlation of the two profiles (``similarities``), which
  removes each profile's mean and scale;
- their L1/C distance (``profile_distances``), the mean over clusters of the
  absolute effect difference in accuracy fraction, which keeps the effect
  magnitudes.

Both are computed over the same clusters per cluster type:

- neuron clusters: all unnormalized input and hidden neuron clusters, the
  ``lesion_unnorm`` entry of ``normalized_lesion_effects_<aname>.pkl``. Their
  Pearson matrix is the one the causal-vs-activity figures use
  (``causal_similarity`` in ``causal_vs_activity_tasksim_hidden_<aname>.pkl``,
  the correlation of these same effects), which is read and checked against
  the recomputation;
- synapse clusters: the var-weighted unnormalized synapse clusters lesioned
  with freeze_M (``SYNAPSE_LESION_MODE``, the intervention of the paper's
  lesion heatmap), from the same cache, without the unresponsive class. That
  class holds most synapses, so its size-matched random control destroys the
  network while the class itself does little, a large negative effect for
  every task that would dominate the profiles; its label is read from the
  lesion pickle (``multiple_tasks_perf/<aname>/``), as the task-specificity
  analysis does.

Each (representation, cluster type, measure) gets a two-sided Spearman
task-label permutation (Mantel) test (``N_PERMUTATIONS`` permutations, seed
``PERMUTATION_SEED``), because the pairs share tasks, and a descriptive OLS
guide line for the figure (as the causal-vs-activity scatter has; the
statistic stays the rank correlation). Tasks whose context-end states lie
closer depending on more similar clusters shows as a negative rho for the
similarity and a positive rho for the distance. Because the tasks form
three families (``TASK_FAMILIES``: go, decision-making, matching) whose
members lie together in state space and share lesion profiles, each test is
also run with the labels permuted within families only (restricted Mantel,
``core.state_space_geometry.mantel_spearman_restricted_permutation``): that
p says whether the arrangement of the tasks within their families adds to
the association family membership alone produces, and its ``within`` part
is the rho over the same-family pairs alone; ``per_family`` repeats this
for each family on its own pairs (labels shuffled within that family only;
four matching tasks allow 24 distinct permutations, so that p is coarse).
Nothing is recomputed from network activity; the script reads caches and
runs in seconds. Rerun it after ``state_space_structure.py`` or
``lesion_plot.py`` (the state-space batch run clears its output too).

Output: ``state_space/state_space_vs_lesion_results.pkl``, {aname: entry}
with the per-pair coordinates of both measures, the full and restricted
tests (null arrays included), the guide lines, the task families and the
run's L2 strength; ``paper_plot.py`` draws them in ``lesion`` mode
(``state_space_vs_lesion`` and ``state_space_vs_lesion_seeds``, their
``_families`` versions and ``state_space_vs_lesion_family_seeds``).
"""
from pathlib import Path
import math
import pickle

import numpy as np

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from lesion_cache import load_lesion_pickle, normalize_lesion_cache, resolve_lesion_cache_path
from state_space_geometry import mantel_spearman_permutation, mantel_spearman_restricted_permutation
from state_space_cache import M_ALIGNMENT, require_aligned
from core.task_variance import VARIANCE_METHOD

STATE_SPACE_DIR = Path("state_space")
LESION_NORM_ROOT = Path("multiple_tasks_norm")
LESION_PERF_ROOT = Path("multiple_tasks_perf")
NOISE_LEVEL = 0.01
N_PERMUTATIONS = 10000
PERMUTATION_SEED = 0
# 2: every (representation, cluster type) test has a saved OLS guide line.
# 3: the L1/C lesion-profile distance beside the Pearson similarity, with its
#    own tests and guide lines (MEASURES).
# 4: the within-family restricted permutation test of every panel, with the
#    rho over the same-family pairs, and the task families.
# 5: each restricted test also holds, under per_family, the test over one
#    family's pairs alone, for every family.
SCHEMA_VERSION = 5
RESULT_NAME = "state_space_vs_lesion_results.pkl"
REPRESENTATIONS = ("hidden", "eff_mod")
CLUSTER_TYPES = ("neuron", "synapse")
NEURON_TASKSIM_NAME = "causal_vs_activity_tasksim_hidden"
NEURON_SIMILARITY_DEFINITION = "pearson_corr_of_task_lesion_effect_profiles_over_neuron_clusters"
# The normalized-effects entry of the unnormalized input and hidden neuron
# clusters, the profiles the causal-vs-activity matrix correlates.
NEURON_EFFECT_ENTRY = "lesion_unnorm"
NEURON_DISTANCE_DEFINITION = "mean_over_neuron_clusters_abs_task_lesion_effect_difference"
# Synapse intervention of the synapse-cluster column: freeze_M (M held at its
# initial value, W kept), as in paper_plot's lesion heatmap
# (LESION_HEATMAP_MODE); test_state_space_vs_lesion checks the two agree.
SYNAPSE_LESION_MODE = "freeze_M"
SYNAPSE_EFFECT_ENTRY = f"modulation_all_var_weighted_unnormalized__{SYNAPSE_LESION_MODE}"
SYNAPSE_SIMILARITY_DEFINITION = (f"pearson_corr_of_task_{SYNAPSE_LESION_MODE}_lesion_effect_profiles"
                                 "_over_var_weighted_synapse_clusters")
SYNAPSE_DISTANCE_DEFINITION = (f"mean_over_var_weighted_synapse_clusters_abs_task_{SYNAPSE_LESION_MODE}"
                               "_lesion_effect_difference")
SYNAPSE_CONDITION_PREFIX = "mod_c"
SYNAPSE_BASELINE_CONDITIONS = ("mod_nolesion", "mod_cNone")
# The two lesion-profile measures of a profile record (its key) and the
# results-entry keys holding their pair values, Mantel tests, guide lines and
# within-family restricted tests; paper_plot._SSL_MEASURES mirrors this.
MEASURES = (("similarity", "similarities", "tests", "trend_lines", "restricted_tests"),
            ("distance", "profile_distances", "profile_distance_tests",
             "profile_distance_trend_lines", "profile_distance_restricted_tests"))
# The ruleset's three task families (README): the restricted null permutes
# task labels within them.
TASK_FAMILIES = {
    "go": ("fdgo", "fdanti", "delaygo", "delayanti", "reactgo", "reactanti"),
    "decision": ("delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm"),
    "matching": ("dmsgo", "dmsnogo", "dmcgo", "dmcnogo"),
}


def task_families(tasks):
    """The TASK_FAMILIES family of each task, in order."""
    family_of = {task: family for family, members in TASK_FAMILIES.items() for task in members}
    unknown = [task for task in tasks if task not in family_of]
    if unknown:
        raise ValueError(f"tasks of no family: {unknown}")
    return [family_of[task] for task in tasks]


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
    require_aligned(saved, path.name)
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


def _profile_measures(effect, source):
    """(similarity, distance) task x task matrices of a (tasks, clusters)
    lesion-effect matrix: the Pearson correlation of the two tasks' profiles
    and their L1/C distance, the mean over clusters of the absolute effect
    difference. A constant profile has no correlation and is an error."""
    effect = np.asarray(effect, dtype=float)
    if effect.ndim != 2 or effect.shape[1] < 3 or not np.isfinite(effect).all():
        raise ValueError(f"{source}: invalid lesion effects")
    # A profile without spread has no correlation: the OM scatter's constant
    # criterion (centered norm <= 1e-12), not NaN alone, because a constant
    # row of rounded accuracies leaves a 1e-17 residual that corrcoef accepts.
    centered = effect - effect.mean(axis=1, keepdims=True)
    if (np.linalg.norm(centered, axis=1) <= 1e-12).any():
        raise ValueError(f"{source}: a task has a constant lesion profile")
    similarity = np.corrcoef(effect)
    distance = np.abs(effect[:, None, :] - effect[None, :, :]).mean(axis=2)
    return similarity, distance


def _normalized_effects(aname, lesion_root):
    """(path, cache) of the run's validated normalized-effects cache
    (lesion_plot.py), or None when the run has none."""
    path = resolve_lesion_cache_path(Path(lesion_root) / aname / f"normalized_lesion_effects_{aname}.pkl")
    if not path.exists():
        return None
    saved = normalize_lesion_cache(load_lesion_pickle(path))
    if saved.get("schema_version") != 1 or saved.get("aname") != aname:
        raise ValueError(f"{path.name}: legacy cache or another run")
    return path, saved


def neuron_cluster_profiles(aname, lesion_root=LESION_NORM_ROOT):
    """{"tasks", "similarity", "distance", "excluded"} of the two tasks'
    lesion-effect profiles over all unnormalized input and hidden neuron
    clusters: the similarity is the saved causal-vs-activity Pearson matrix,
    the distance their L1/C from the normalized-effects cache's
    NEURON_EFFECT_ENTRY, whose correlation must reproduce that matrix (the
    two caches describe the same effects). None when the run lacks either
    cache; nothing is excluded."""
    path = resolve_lesion_cache_path(Path(lesion_root) / aname / f"{NEURON_TASKSIM_NAME}_{aname}.pkl")
    effects = _normalized_effects(aname, lesion_root)
    if not path.exists() or effects is None:
        return None
    saved = load_lesion_pickle(path)
    if saved.get("schema_version") != 2 or saved.get("aname") != aname:
        raise ValueError(f"{path.name}: legacy cache or another run")
    if saved.get("y_definition") != NEURON_SIMILARITY_DEFINITION:
        raise ValueError(f"{path.name}: unexpected lesion similarity definition")
    tasks = list(saved["tasks"])
    similarity = np.asarray(saved["causal_similarity"], dtype=float)
    if similarity.shape != (len(tasks), len(tasks)) or not np.isfinite(similarity).all():
        raise ValueError(f"{path.name}: invalid causal similarity matrix")
    effects_path, effects = effects
    entry = effects["entries"].get(NEURON_EFFECT_ENTRY)
    if entry is None:
        raise ValueError(f"{effects_path.name}: no {NEURON_EFFECT_ENTRY} entry")
    if entry.get("definition") != "random_minus_lesion" or list(entry["tasks"]) != tasks:
        raise ValueError(f"{effects_path.name}: unexpected neuron effect definition or tasks")
    recomputed, distance = _profile_measures(entry["effect"], effects_path.name)
    if not np.allclose(recomputed, similarity, atol=1e-6):
        raise ValueError(f"{effects_path.name}: the {NEURON_EFFECT_ENTRY} effects do not reproduce "
                         f"the causal similarity of {path.name}")
    return {"tasks": tasks, "similarity": similarity, "distance": distance, "excluded": None}


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


def synapse_cluster_profiles(aname, lesion_root=LESION_NORM_ROOT, lesion_perf_root=LESION_PERF_ROOT):
    """{"tasks", "similarity", "distance", "excluded"} of the two tasks'
    SYNAPSE_LESION_MODE (freeze_M) lesion-effect profiles over the
    var-weighted unnormalized synapse clusters, leaving out the recorded
    unresponsive class (``excluded``, its condition name); None when the run
    has no normalized-effects cache. An entry recording another mode is an
    error, never a substitute."""
    effects = _normalized_effects(aname, lesion_root)
    if effects is None:
        return None
    path, saved = effects
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
    if effect.shape[0] != len(tasks):
        raise ValueError(f"{path.name}: invalid synapse lesion effects")
    similarity, distance = _profile_measures(effect, path.name)
    return {"tasks": tasks, "similarity": similarity, "distance": distance, "excluded": excluded}


def lesion_control_pool(aname, lesion_root=LESION_NORM_ROOT):
    """The random-control pool the run's lesion_plot.py caches were computed
    with (``control_pool`` of the normalized-effects cache; "all" for caches
    written before it was recorded, lesion.py's every-unit controls)."""
    path = resolve_lesion_cache_path(Path(lesion_root) / aname / f"normalized_lesion_effects_{aname}.pkl")
    return str(load_lesion_pickle(path).get("control_pool", "all"))


def lesion_profiles(aname, lesion_root=LESION_NORM_ROOT, lesion_perf_root=LESION_PERF_ROOT):
    """{cluster type: profile record} of a run, or None when a lesion cache
    is missing."""
    neuron = neuron_cluster_profiles(aname, lesion_root)
    synapse = synapse_cluster_profiles(aname, lesion_root, lesion_perf_root)
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


def family_tests(distance, aligned, tasks, families, n_permutations=N_PERMUTATIONS,
                 seed=PERMUTATION_SEED):
    """{family: test over that family's pairs alone}: the ``within`` part of
    a restricted test that shuffles that family's tasks only (every other
    task a group of its own), with the family's size and its number of
    distinct label permutations, which bounds the p's resolution (four
    matching tasks: 24, so p >= about 0.04)."""
    tests = {}
    for family in TASK_FAMILIES:
        groups = [family if own == family else task for task, own in zip(tasks, families)]
        result = mantel_spearman_restricted_permutation(
            distance, aligned, groups, n_permutations=n_permutations, seed=seed)
        size = families.count(family)
        tests[family] = {**result["within"], "n_items": size,
                         "n_distinct_permutations": math.factorial(size)}
    return tests


def analyze_run(aname, l2_info, tasks, distances, profiles,
                n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED, control_pool="all",
                *, m_alignment):
    """The results entry of one run: per task pair (``pairs``, i < j in the
    structure cache's task order) the centroid distances per representation
    and, per cluster type, the lesion-profile similarities and L1/C distances
    of ``profiles`` ({cluster type: profile record}); for each MEASURES row,
    under ``<tests>[representation][cluster_type]`` the Spearman Mantel test
    of centroid distance against that measure, under ``<trend lines>`` (same
    keys) its descriptive OLS guide and under ``<restricted tests>`` the
    within-family restricted test with, under ``per_family``, the test over
    each family's pairs alone (``families`` maps each task to its
    TASK_FAMILIES family, ``pair_same_family`` flags the pairs of one
    family). ``excluded_conditions`` names the cluster condition each profile
    left out; ``control_pool`` the random-control pool of the lesion caches
    the profiles came from. The caller must pass the validated structure
    distances' time-alignment marker as ``m_alignment``."""
    require_aligned({"m_alignment": m_alignment}, "structure distances")
    upper = np.triu_indices(len(tasks), 1)
    families = task_families(tasks)
    entry = {
        "schema_version": SCHEMA_VERSION, "aname": aname, "l2_info": l2_info,
        "variance_method": VARIANCE_METHOD,
        "control_pool": str(control_pool),
        "m_alignment": m_alignment,
        "tasks": list(tasks), "pairs": [(tasks[i], tasks[j]) for i, j in zip(*upper)],
        "n_permutations": int(n_permutations), "permutation_seed": int(seed),
        "distance_definition": "euclidean distance between full-dimensional context-end task centroids",
        "similarity_definitions": {
            "neuron": NEURON_SIMILARITY_DEFINITION,
            "synapse": SYNAPSE_SIMILARITY_DEFINITION,
        },
        "profile_distance_definitions": {
            "neuron": NEURON_DISTANCE_DEFINITION,
            "synapse": SYNAPSE_DISTANCE_DEFINITION,
        },
        "distances": {name: matrix[upper] for name, matrix in distances.items()},
        "excluded_conditions": {},
        "families": dict(zip(tasks, families)),
        "pair_same_family": np.array([families[i] == families[j] for i, j in zip(*upper)]),
    }
    for _, values_key, tests_key, trends_key, restricted_key in MEASURES:
        entry[values_key] = {}
        for key in (tests_key, trends_key, restricted_key):
            entry[key] = {name: {} for name in distances}
    for cluster_type, profile in profiles.items():
        if profile["excluded"] is not None:
            entry["excluded_conditions"][cluster_type] = profile["excluded"]
        for measure, values_key, tests_key, trends_key, restricted_key in MEASURES:
            aligned = _aligned(profile[measure], profile["tasks"], tasks)
            entry[values_key][cluster_type] = aligned[upper]
            for representation, distance in distances.items():
                entry[tests_key][representation][cluster_type] = mantel_spearman_permutation(
                    distance, aligned, n_permutations=n_permutations, seed=seed)
                entry[trends_key][representation][cluster_type] = descriptive_trend_line(
                    distance[upper], aligned[upper])
                restricted = mantel_spearman_restricted_permutation(
                    distance, aligned, families, n_permutations=n_permutations, seed=seed)
                restricted["per_family"] = family_tests(
                    distance, aligned, tasks, families, n_permutations=n_permutations, seed=seed)
                entry[restricted_key][representation][cluster_type] = restricted
    return entry


def _format_p(p_value):
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def print_summary(results):
    """Per L2 cohort, representation, cluster type and measure: the median
    rho over runs and the number of runs with permutation p below 0.05, then
    the same under the within-family restricted test, the median rho over
    the same-family pairs alone, and each family's own rho and count."""
    strengths = sorted({entry["l2_info"] for entry in results.values()})
    print("=" * 78)
    print(f"Context-end task distance vs lesion-profile similarity, {len(results)} runs")
    print("=" * 78)
    for strength in strengths:
        cohort = [entry for entry in results.values() if entry["l2_info"] == strength]
        print(f"\nL2 = {strength:g}: {len(cohort)} runs")
        for representation in REPRESENTATIONS:
            for cluster_type in CLUSTER_TYPES:
                for measure, _, tests_key, _, restricted_key in MEASURES:
                    tests = [entry[tests_key][representation][cluster_type] for entry in cohort
                             if cluster_type in entry[tests_key][representation]]
                    if not tests:
                        continue
                    restricted = [entry[restricted_key][representation][cluster_type]
                                  for entry in cohort]
                    rhos = np.array([test["rho"] for test in tests])
                    within = np.array([test["within"]["rho"] for test in restricted])
                    significant = sum(test["p_value"] < 0.05 for test in tests)
                    significant_restricted = sum(test["p_value"] < 0.05 for test in restricted)
                    print(f"  {representation} distance vs {cluster_type} lesion-profile {measure}: "
                          f"rho median {np.median(rhos):+.3f} (range {rhos.min():+.3f} to "
                          f"{rhos.max():+.3f}), {significant}/{len(tests)} runs with p < 0.05; "
                          f"within-family permutation: {significant_restricted}/{len(tests)} runs "
                          f"with p < 0.05, same-family pairs rho median {np.median(within):+.3f}")
                    per_family = ", ".join(
                        f"{family} {np.median([test['per_family'][family]['rho'] for test in restricted]):+.3f}"
                        f" ({sum(test['per_family'][family]['p_value'] < 0.05 for test in restricted)} sig.)"
                        for family in TASK_FAMILIES)
                    print(f"    per family (rho median over runs, runs with p < 0.05): {per_family}")


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
        profiles = lesion_profiles(aname, lesion_root, lesion_perf_root)
        if profiles is None:
            print(f"{aname}: no lesion caches under {Path(lesion_root) / aname}; skipped")
            continue
        entry = analyze_run(aname, l2_info, tasks, distances, profiles,
                            n_permutations=n_permutations, seed=seed,
                            control_pool=lesion_control_pool(aname, lesion_root),
                            m_alignment=M_ALIGNMENT)
        results[aname] = entry
        similarity = entry["tests"]["eff_mod"]["synapse"]
        distance = entry["profile_distance_tests"]["eff_mod"]["synapse"]
        restricted = entry["profile_distance_restricted_tests"]["eff_mod"]["synapse"]
        print(f"{aname}: W*M distance vs synapse lesion-profile similarity rho = "
              f"{similarity['rho']:+.3f} ({_format_p(similarity['p_value'])}), "
              f"vs L1/C distance rho = {distance['rho']:+.3f} ({_format_p(distance['p_value'])}; "
              f"within-family {_format_p(restricted['p_value'])})")
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
