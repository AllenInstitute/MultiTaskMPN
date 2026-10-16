"""
Category structure and factor additivity of the context-end task centroids.

Reads every ``state_space/state_space_pca_<aname>_noise0.01.pkl`` written by
``state_space_shift.py`` and, for each of its three representations (hidden
state, modulation M, effective modulation W * M), asks two questions of the
full-dimensional task centroids saved there:

1. Category structure. Is the category-balanced separation score
   S = (between - within) / (between + within), computed with the six paper
   color groups, larger than for a random regrouping of the same tasks? The
   null shuffles the category labels over the 15 tasks (one-sided permutation
   p). The 15 x 15 centroid distance matrix and its average-linkage ordering
   are saved for the representational dissimilarity figure.
2. Task-vector additivity. For the task pairs that differ in one factor
   (``state_space_geometry.TASK_FACTOR_PAIRS``: pro to anti, delay to memory,
   delay to reaction, modality, context cue, match to category) the
   centroid-difference vectors are compared by their mean pairwise cosine;
   the null permutes which centroid belongs to which task (one-sided p per
   factor and pooled over factors).

Per L2 cohort (runs sharing ``l2_info``) the runs' distance matrices are also
averaged after dividing each by its mean task-pair distance, and the mean
matrix gets its own tree and category test, together with the seed-to-seed
Spearman consistency of the per-run matrices (the paper's RDM figure draws
this cohort mean; the single-run matrix is kept as the example figure).

The untrained networks of ``state_space_shift.py --untrained`` (PCA caches in
``state_space/untrained/``, one per config and init seed) get the same
per-network analysis. Their summaries are kept apart, under ``untrained`` in
the combined results, and never enter the L2 cohorts or their mean matrices:
an untrained network has no L2 strength, since L2 only acts during training.

Nothing is recomputed from network activity: the centroids come from the PCA
caches, so this runs in minutes on CPU. Run it after ``state_space_shift.py``
(whose batch run clears the top-level ``state_space/`` files, including the
caches written here) and before ``paper_plot.py state_space``.

Outputs, in ``state_space/``:
  state_space_structure_<aname>_noise0.01.pkl   per run, with the null arrays
  untrained/state_space_structure_<aname>_init<seed>_noise0.01.pkl
                                                 per untrained network, likewise
  state_space_structure_results.pkl              every run's summary statistics,
                                                 the per-cohort mean matrices and
                                                 the untrained networks' summaries
"""
from pathlib import Path
import pickle

import numpy as np

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from state_space_geometry import (
    category_separation_permutation,
    category_separation_permutation_from_distances,
    factor_vector_consistency,
    mean_task_distance_matrix,
    task_centroid_distances,
    task_centroid_linkage,
    task_distance_consistency,
)

STATE_SPACE_DIR = Path("state_space")
STATE_SPACE_NOISE_LEVEL = 0.01
REPRESENTATIONS = ("hidden", "mod", "eff_mod")
N_PERMUTATIONS = 10000
PERMUTATION_SEED = 0
LINKAGE_METHOD = "average"
SCHEMA_VERSION = 1
COMBINED_RESULT_NAME = "state_space_structure_results.pkl"
# Subdirectory of the untrained networks' PCA caches (state_space_shift.py
# UNTRAINED_DIR_NAME); the per-run glob below never descends into it.
UNTRAINED_DIR_NAME = "untrained"
_PCA_PREFIX = "state_space_pca_"


def pca_cache_paths(state_space_dir=STATE_SPACE_DIR, noise_level=STATE_SPACE_NOISE_LEVEL):
    """Every per-run PCA cache of the state-space analysis, sorted by name."""
    return sorted(Path(state_space_dir).glob(f"{_PCA_PREFIX}*_noise{noise_level}.pkl"))


def untrained_pca_cache_paths(state_space_dir=STATE_SPACE_DIR,
                              noise_level=STATE_SPACE_NOISE_LEVEL):
    """Every untrained network's PCA cache (state_space_shift.py --untrained),
    sorted by name; empty when none was made."""
    return pca_cache_paths(Path(state_space_dir) / UNTRAINED_DIR_NAME, noise_level)


def cache_stem(pca_path, noise_level=STATE_SPACE_NOISE_LEVEL):
    """The identifier between "state_space_pca_" and "_noise<level>.pkl" of a
    PCA cache name: the aname of a run, aname + "_init<seed>" of an untrained
    network."""
    name = Path(pca_path).name
    suffix = f"_noise{noise_level}.pkl"
    if not (name.startswith(_PCA_PREFIX) and name.endswith(suffix)):
        raise ValueError(f"Not a PCA cache name: {name}")
    return name[len(_PCA_PREFIX):-len(suffix)]


def structure_cache_path(aname, state_space_dir=STATE_SPACE_DIR,
                         noise_level=STATE_SPACE_NOISE_LEVEL):
    return Path(state_space_dir) / f"state_space_structure_{aname}_noise{noise_level}.pkl"


def task_categories(data):
    """Per-task category ids and names from the cache's rule -> (name, color)
    mapping. The color is the category id: it is what the paper groups by,
    and the dms pair deliberately shares the Reaction colors (see
    state_space_shift.py), so two names can map to one category."""
    mapping = data["rule_motif_mapping"]
    all_rules = list(data["all_rules"])
    if len(set(all_rules)) != len(all_rules):
        raise ValueError("Task names in the PCA cache must be unique.")
    missing = [rule for rule in all_rules if rule not in mapping]
    if missing:
        raise ValueError(f"No category for task(s) {missing} in the PCA cache.")
    return ([mapping[rule][1] for rule in all_rules],
            [mapping[rule][0] for rule in all_rules])


def _centroids(record):
    if record.get("centroid_space") != "original_features":
        raise ValueError("No original-feature task centroids; rerun state_space_shift.py.")
    return np.asarray(record["task_centroids"], dtype=np.float64)


def analyze_representation(record, all_rules, categories, n_permutations=N_PERMUTATIONS,
                           seed=PERMUTATION_SEED):
    """Distances, linkage, category test and factor-vector test of one
    representation's saved centroids."""
    centroids = _centroids(record)
    if centroids.shape[0] != len(all_rules):
        raise ValueError("Task centroids must have one row per task.")
    distances = task_centroid_distances(centroids)
    tree = task_centroid_linkage(distances, method=LINKAGE_METHOD)
    return {
        "distances": distances,
        "linkage": tree["linkage"],
        "leaf_order": tree["leaf_order"],
        "linkage_method": tree["method"],
        "separation": category_separation_permutation(
            centroids, categories, n_permutations=n_permutations, seed=seed),
        "factor_vectors": factor_vector_consistency(
            centroids, all_rules, n_permutations=n_permutations, seed=seed),
    }


def analyze_run(data, n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED,
                representations=REPRESENTATIONS):
    """Full structure result of one PCA cache (``pickle.load`` of the file)."""
    all_rules = list(data["all_rules"])
    categories, category_names = task_categories(data)
    result = {
        "schema_version": SCHEMA_VERSION,
        "aname": data["aname"],
        "l2_info": float(data["l2_info"]),
        "noise_level": data.get("noise_level", STATE_SPACE_NOISE_LEVEL),
        "all_rules": all_rules,
        "categories": categories,
        "category_names": category_names,
        "n_permutations": int(n_permutations),
        "permutation_seed": int(seed),
        "representations": {},
    }
    for representation in representations:
        record = data["pca_results"][representation]
        result["representations"][representation] = analyze_representation(
            record, all_rules, categories, n_permutations=n_permutations, seed=seed)
    return result


_SEPARATION_SUMMARY_KEYS = ("score", "within_distance", "between_distance", "p_value",
                            "null_mean", "null_sd", "null_quantile_95", "n_permutations")
_POOLED_SUMMARY_KEYS = ("mean_cosine", "p_value", "null_mean", "null_sd", "null_quantile_95",
                        "null_quantile_025", "null_quantile_975")
_FACTOR_SUMMARY_KEYS = _POOLED_SUMMARY_KEYS + ("shared_variance_fraction",)


def summarize_run(result):
    """The run's statistics without the null arrays, for the combined file."""
    summary = {"l2_info": result["l2_info"], "aname": result["aname"], "representations": {}}
    for representation, entry in result["representations"].items():
        factors = entry["factor_vectors"]
        summary["representations"][representation] = {
            "separation": {key: entry["separation"][key] for key in _SEPARATION_SUMMARY_KEYS},
            "factor_vectors": {
                factor: {**{key: stats[key] for key in _FACTOR_SUMMARY_KEYS},
                         "n_pairs": len(stats["pairs"])}
                for factor, stats in factors["factors"].items()
            },
            "pooled": {key: factors["pooled"][key] for key in _POOLED_SUMMARY_KEYS},
            "skipped_factors": list(factors["skipped_factors"]),
        }
    return summary


def cohort_mean_structure(results, n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED,
                          representations=REPRESENTATIONS):
    """Mean task distance matrix of one cohort's runs (``results``: aname ->
    ``analyze_run`` output, all with the same task order), per representation,
    with its average-linkage tree, the category test on the mean matrix and
    the Spearman consistency of the runs' matrices."""
    anames = sorted(results)
    if not anames:
        raise ValueError("A cohort needs at least one run.")
    first = results[anames[0]]
    for aname in anames[1:]:
        if (results[aname]["all_rules"] != first["all_rules"]
                or results[aname]["categories"] != first["categories"]):
            raise ValueError(f"{aname} lists its tasks differently from {anames[0]}.")
    cohort = {
        "anames": anames,
        "n_runs": len(anames),
        "all_rules": list(first["all_rules"]),
        "categories": list(first["categories"]),
        "category_names": list(first["category_names"]),
        "normalization": "each run's distances divided by its mean off-diagonal distance",
        "representations": {},
    }
    for representation in representations:
        matrices = [results[aname]["representations"][representation]["distances"]
                    for aname in anames]
        mean_distances, scales = mean_task_distance_matrix(matrices)
        tree = task_centroid_linkage(mean_distances, method=LINKAGE_METHOD)
        cohort["representations"][representation] = {
            "distances": mean_distances,
            "run_scales": scales,
            "linkage": tree["linkage"],
            "leaf_order": tree["leaf_order"],
            "linkage_method": tree["method"],
            "separation": category_separation_permutation_from_distances(
                mean_distances, cohort["categories"], n_permutations=n_permutations, seed=seed),
            "consistency": task_distance_consistency(matrices),
        }
    return cohort


def analyze_untrained(state_space_dir=STATE_SPACE_DIR, n_permutations=N_PERMUTATIONS,
                      seed=PERMUTATION_SEED, noise_level=STATE_SPACE_NOISE_LEVEL):
    """``analyze_run`` for every untrained network's PCA cache: each full
    result goes next to its cache in the untrained subdirectory, and the
    returned {cache stem: summary} adds to ``summarize_run`` the network's
    ``init_seed`` and ``config_source`` (the run whose config it used). A
    cache not marked ``untrained`` is skipped with a message rather than
    counted as an untrained network."""
    untrained_dir = Path(state_space_dir) / UNTRAINED_DIR_NAME
    summaries = {}
    for path in untrained_pca_cache_paths(state_space_dir, noise_level):
        with path.open("rb") as stream:
            data = pickle.load(stream)
        if data.get("untrained") is not True:
            print(f"Skipped {path}: not marked as an untrained network.")
            continue
        stem = cache_stem(path, noise_level)
        result = analyze_run(data, n_permutations=n_permutations, seed=seed)
        result.update(untrained=True, init_seed=data.get("init_seed"),
                      config_source=data.get("config_source", data["aname"]))
        out_path = structure_cache_path(stem, untrained_dir, noise_level)
        with out_path.open("wb") as stream:
            pickle.dump(result, stream)
        summaries[stem] = {**summarize_run(result), "init_seed": result["init_seed"],
                           "config_source": result["config_source"]}
        eff_mod = result["representations"]["eff_mod"]["separation"]
        print(f"untrained {stem}: eff_mod S = {eff_mod['score']:.4f} "
              f"({_format_p(eff_mod['p_value'])}); saved {out_path}")
    return summaries


def group_runs_by_l2(results):
    """aname -> run result, grouped into {l2_info: {aname: result}}."""
    grouped = {}
    for aname in sorted(results):
        grouped.setdefault(float(results[aname]["l2_info"]), {})[aname] = results[aname]
    return grouped


def _format_p(p_value):
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def print_summary(combined):
    """Per L2 cohort and representation: seeds with a significant category
    structure, the median score, and per factor the significant seed count."""
    runs = combined["runs"]
    strengths = sorted({entry["l2_info"] for entry in runs.values()})
    print("=" * 78)
    print(f"State-space structure of {len(runs)} runs "
          f"({combined['n_permutations']} permutations, seed {combined['permutation_seed']})")
    print("=" * 78)
    for strength in strengths:
        cohort = [entry for entry in runs.values() if entry["l2_info"] == strength]
        print(f"\nL2 = {strength:g}: {len(cohort)} runs")
        mean_structure = combined.get("cohorts", {}).get(strength)
        for representation in REPRESENTATIONS:
            entries = [entry["representations"][representation] for entry in cohort
                       if representation in entry["representations"]]
            if not entries:
                continue
            scores = np.array([entry["separation"]["score"] for entry in entries])
            p_values = np.array([entry["separation"]["p_value"] for entry in entries])
            print(f"  {representation}: category separation S median {np.median(scores):.4f} "
                  f"(range {scores.min():.4f} to {scores.max():.4f}), "
                  f"{int(np.sum(p_values < 0.05))}/{len(entries)} runs with p < 0.05")
            if mean_structure is not None:
                mean_entry = mean_structure["representations"][representation]
                print(f"      mean RDM over {mean_structure['n_runs']} runs: "
                      f"S = {mean_entry['separation']['score']:.4f} "
                      f"({_format_p(mean_entry['separation']['p_value'])}); seed-to-seed "
                      f"Spearman median {mean_entry['consistency']['median']:.2f} "
                      f"(range {mean_entry['consistency']['min']:.2f} to "
                      f"{mean_entry['consistency']['max']:.2f})")
            factors = entries[0]["factor_vectors"]
            for factor in factors:
                cosines = np.array([entry["factor_vectors"][factor]["mean_cosine"]
                                    for entry in entries])
                significant = sum(entry["factor_vectors"][factor]["p_value"] < 0.05
                                  for entry in entries)
                print(f"      {factor}: mean cosine median {np.median(cosines):.3f}, "
                      f"{significant}/{len(entries)} runs with p < 0.05")
            pooled = [entry["pooled"]["p_value"] for entry in entries]
            print(f"      pooled over factors: {sum(p < 0.05 for p in pooled)}/{len(entries)} "
                  "runs with p < 0.05")

    untrained = combined.get("untrained") or {}
    if untrained:
        print(f"\nUntrained networks: {len(untrained)}")
        for representation in REPRESENTATIONS:
            separations = [entry["representations"][representation]["separation"]
                           for entry in untrained.values()
                           if representation in entry["representations"]]
            if not separations:
                continue
            scores = np.array([separation["score"] for separation in separations])
            significant = sum(separation["p_value"] < 0.05 for separation in separations)
            print(f"  {representation}: category separation S median {np.median(scores):.4f} "
                  f"(range {scores.min():.4f} to {scores.max():.4f}), "
                  f"{significant}/{len(separations)} networks with p < 0.05")


def run_all(state_space_dir=STATE_SPACE_DIR, n_permutations=N_PERMUTATIONS,
            seed=PERMUTATION_SEED, noise_level=STATE_SPACE_NOISE_LEVEL):
    """Analyze every PCA cache, write the per-run and combined pickles and
    return the combined result."""
    state_space_dir = Path(state_space_dir)
    paths = pca_cache_paths(state_space_dir, noise_level)
    if not paths:
        raise FileNotFoundError(
            f"No state_space_pca_*_noise{noise_level}.pkl in {state_space_dir}/; "
            "run python multiple_task/state_space_shift.py first.")
    combined = {
        "schema_version": SCHEMA_VERSION,
        "noise_level": noise_level,
        "n_permutations": int(n_permutations),
        "permutation_seed": int(seed),
        "linkage_method": LINKAGE_METHOD,
        "runs": {},
        "cohorts": {},
        "untrained": {},
    }
    results = {}
    for path in paths:
        with path.open("rb") as stream:
            data = pickle.load(stream)
        result = analyze_run(data, n_permutations=n_permutations, seed=seed)
        out_path = structure_cache_path(result["aname"], state_space_dir, noise_level)
        with out_path.open("wb") as stream:
            pickle.dump(result, stream)
        results[result["aname"]] = result
        combined["runs"][result["aname"]] = summarize_run(result)
        eff_mod = result["representations"]["eff_mod"]["separation"]
        print(f"{result['aname']}: eff_mod S = {eff_mod['score']:.4f} "
              f"({_format_p(eff_mod['p_value'])}); saved {out_path}")
    for strength, cohort_results in group_runs_by_l2(results).items():
        combined["cohorts"][strength] = cohort_mean_structure(
            cohort_results, n_permutations=n_permutations, seed=seed)
    combined["untrained"] = analyze_untrained(
        state_space_dir, n_permutations=n_permutations, seed=seed, noise_level=noise_level)
    combined_path = state_space_dir / COMBINED_RESULT_NAME
    with combined_path.open("wb") as stream:
        pickle.dump(combined, stream)
    print(f"Saved {combined_path}")
    print_summary(combined)
    return combined


if __name__ == "__main__":
    run_all()
