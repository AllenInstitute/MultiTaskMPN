"""High-dimensional task-centroid geometry for state-space examples.

Everything here works on the per-task context-end centroids that
``state_space_shift.py`` saves (one row per task, all original features):
the category-balanced separation score and its label-permutation test, the
task-by-task distance matrix with its average-linkage ordering, and the
additivity of task-difference vectors along the factors that distinguish the
tasks. No PCA, whitening or feature normalization is applied anywhere.
"""

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform


# Pairs of tasks that differ in exactly one task factor, listed as (from, to)
# so the difference vector centroid[to] - centroid[from] points the same way
# within a factor. If the context state encoded the factors additively, every
# vector of one factor would be (nearly) parallel.
TASK_FACTOR_PAIRS = {
    "pro to anti": [("fdgo", "fdanti"), ("delaygo", "delayanti"),
                    ("reactgo", "reactanti"), ("dmsgo", "dmsnogo"),
                    ("dmcgo", "dmcnogo")],
    "delay to memory": [("fdgo", "delaygo"), ("fdanti", "delayanti")],
    "delay to reaction": [("fdgo", "reactgo"), ("fdanti", "reactanti")],
    "modality 1 to 2": [("delaydm1", "delaydm2"),
                        ("contextdelaydm1", "contextdelaydm2")],
    "context cue": [("delaydm1", "contextdelaydm1"),
                    ("delaydm2", "contextdelaydm2")],
    "match to category": [("dmsgo", "dmcgo"), ("dmsnogo", "dmcnogo")],
}


def context_task_centroids(states, rule_labels, n_rules):
    """Average context-end trials within each task, preserving rule order."""
    states = np.asarray(states)
    rule_labels = np.asarray(rule_labels)
    if (states.ndim < 2 or rule_labels.shape != (states.shape[0],)
            or not np.issubdtype(rule_labels.dtype, np.integer)):
        raise ValueError("Context states and integer task labels must align.")
    if (n_rules < 1 or states.size == 0 or not np.all(np.isfinite(states))
            or np.any(rule_labels < 0) or np.any(rule_labels >= n_rules)):
        raise ValueError("Context states must be finite with valid task labels.")
    flattened = states.reshape(states.shape[0], -1)
    counts = np.bincount(rule_labels, minlength=n_rules)
    if np.any(counts == 0):
        raise ValueError("Every task must have context-end samples.")
    centers = np.stack([
        flattened[rule_labels == index].mean(axis=0, dtype=np.float64)
        for index in range(n_rules)
    ])
    return centers, counts


def _validated_centroids(centroids):
    centroids = np.asarray(centroids, dtype=np.float64)
    if (centroids.ndim != 2 or centroids.shape[0] < 2 or centroids.shape[1] == 0
            or not np.all(np.isfinite(centroids))):
        raise ValueError("Need a finite (tasks x features) centroid matrix with two or more tasks.")
    return centroids


def task_centroid_distances(centroids):
    """Euclidean distance between every pair of task centroids, (tasks x tasks)."""
    centroids = _validated_centroids(centroids)
    return squareform(pdist(centroids, metric="euclidean"))


def _category_membership(categories, n_tasks):
    """One-hot (categories x tasks) membership, requiring two tasks per category."""
    categories = np.asarray(categories)
    if categories.shape != (n_tasks,):
        raise ValueError("Category labels must align with the task centroids.")
    unique, codes, counts = np.unique(categories, return_inverse=True, return_counts=True)
    if unique.size < 2 or np.any(counts < 2):
        raise ValueError("Need at least two categories with two tasks each.")
    membership = np.zeros((unique.size, n_tasks), dtype=np.float64)
    membership[codes, np.arange(n_tasks)] = 1.0
    return membership, counts


def _separation_from_membership(distances, membership, counts):
    """Category-balanced within/between means from a membership matrix.

    The (categories x categories) block sums ``membership @ distances @
    membership.T`` hold twice each within-category pair sum on the diagonal
    and the between-category pair sums off it.
    """
    block_sums = membership @ distances @ membership.T
    within_means = np.diag(block_sums) / (counts * (counts - 1))
    pair_sizes = np.outer(counts, counts)
    upper = np.triu_indices(counts.size, 1)
    between_means = block_sums[upper] / pair_sizes[upper]
    within_mean = float(within_means.mean())
    between_mean = float(between_means.mean())
    total = within_mean + between_mean
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Task centroids have no finite nonzero separation.")
    return {
        "score": (between_mean - within_mean) / total,
        "within_distance": within_mean,
        "between_distance": between_mean,
        "within_pair_count": int((counts * (counts - 1) // 2).sum()),
        "between_pair_count": int(pair_sizes[upper].sum()),
    }


def task_centroid_separation_from_distances(distances, categories):
    """``task_centroid_separation`` for a precomputed task distance matrix."""
    distances = np.asarray(distances, dtype=np.float64)
    if (distances.ndim != 2 or distances.shape[0] != distances.shape[1]
            or not np.all(np.isfinite(distances))):
        raise ValueError("Need a finite square task distance matrix.")
    membership, counts = _category_membership(categories, distances.shape[0])
    metrics = _separation_from_membership(distances, membership, counts)
    metrics.update(n_tasks=int(distances.shape[0]), n_categories=int(counts.size))
    return metrics


def task_centroid_separation(centroids, categories):
    """Compare Euclidean task-center distances with equal category weights.

    Within-category means receive equal weight. Between-category means receive
    equal weight per unordered category pair. The normalized contrast
    (between - within) / (between + within) is scale invariant; larger is better.
    All coordinates are used, with no PCA, whitening, or feature normalization.
    """
    centroids = _validated_centroids(centroids)
    metrics = task_centroid_separation_from_distances(
        task_centroid_distances(centroids), categories)
    metrics["n_features"] = int(centroids.shape[1])
    return metrics


def _permutation_p_value(observed, null):
    """One-sided p for the observed statistic exceeding the null, with the
    observed arrangement counted as one permutation (never zero)."""
    null = np.asarray(null, dtype=np.float64)
    return float((1 + np.count_nonzero(null >= observed)) / (1 + null.size))


def _null_summary(observed, null):
    null = np.asarray(null, dtype=np.float64)
    return {
        "null": null,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if null.size > 1 else float("nan"),
        "null_quantile_95": float(np.quantile(null, 0.95)),
        "null_quantile_025": float(np.quantile(null, 0.025)),
        "null_quantile_975": float(np.quantile(null, 0.975)),
        "p_value": _permutation_p_value(observed, null),
        "n_permutations": int(null.size),
    }


def category_separation_permutation_from_distances(distances, categories, n_permutations=10000,
                                                   seed=0):
    """``category_separation_permutation`` for a precomputed task distance
    matrix, e.g. the normalized mean over a cohort's runs."""
    if n_permutations < 1:
        raise ValueError("Need at least one permutation.")
    distances = _validated_distance_matrix(distances)
    membership, counts = _category_membership(categories, distances.shape[0])
    observed = _separation_from_membership(distances, membership, counts)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for index in range(n_permutations):
        order = rng.permutation(distances.shape[0])
        null[index] = _separation_from_membership(
            distances, membership[:, order], counts)["score"]
    result = dict(observed)
    result.update(n_tasks=int(distances.shape[0]), n_categories=int(counts.size),
                  seed=int(seed),
                  null_statistic="category-balanced separation score under shuffled task labels")
    result.update(_null_summary(observed["score"], null))
    return result


def category_separation_permutation(centroids, categories, n_permutations=10000, seed=0):
    """Label-permutation test of the category-balanced separation score.

    The null shuffles which task carries which category label (category sizes
    are preserved, the centroid geometry is untouched) and recomputes the
    score. One-sided: the observed categories separate the task centroids more
    than a random grouping of the same sizes. Returns the observed metrics with
    the null scores, their 95th percentile and the p-value.
    """
    centroids = _validated_centroids(centroids)
    result = category_separation_permutation_from_distances(
        task_centroid_distances(centroids), categories, n_permutations=n_permutations,
        seed=seed)
    result["n_features"] = int(centroids.shape[1])
    return result


def _validated_distance_matrix(distances):
    distances = np.asarray(distances, dtype=np.float64)
    if (distances.ndim != 2 or distances.shape[0] != distances.shape[1]
            or distances.shape[0] < 2 or not np.all(np.isfinite(distances))
            or not np.allclose(distances, distances.T) or np.any(np.diag(distances) != 0)):
        raise ValueError("Need a finite symmetric task distance matrix with a zero diagonal.")
    return distances


def task_centroid_linkage(distances, method="average"):
    """Hierarchical clustering of the task distance matrix and its leaf order.

    Returns the scipy linkage matrix (for a dendrogram) and the leaf order that
    places joined tasks next to each other, so a heatmap ordered by it shows
    the cluster blocks.
    """
    distances = _validated_distance_matrix(distances)
    merge_tree = linkage(squareform(distances, checks=False), method=method)
    return {
        "linkage": merge_tree,
        "leaf_order": leaves_list(merge_tree).astype(int),
        "method": method,
    }


def _stacked_distance_matrices(distance_matrices):
    matrices = [_validated_distance_matrix(matrix) for matrix in distance_matrices]
    if not matrices or any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("Need one or more task distance matrices of one shape.")
    return np.stack(matrices)


def mean_task_distance_matrix(distance_matrices):
    """Average of several runs' task distance matrices on a common scale.

    Each run's matrix is first divided by its mean off-diagonal distance, so a
    run whose representation is simply larger does not dominate the mean and
    the result is in units of the typical task-pair distance. Returns the
    mean matrix and the per-run scales that were divided out.
    """
    stack = _stacked_distance_matrices(distance_matrices)
    upper = np.triu_indices(stack.shape[1], 1)
    scales = stack[:, upper[0], upper[1]].mean(axis=1)
    if np.any(scales <= 0):
        raise ValueError("Every run must have a positive mean task distance.")
    return (stack / scales[:, None, None]).mean(axis=0), scales


def task_distance_consistency(distance_matrices):
    """How alike the runs' task geometries are: the Spearman rank correlation
    between every two runs' upper-triangle distances (scale-free), summarized
    by its median, minimum and maximum over run pairs. With fewer than two
    runs the summaries are NaN."""
    from scipy.stats import spearmanr

    stack = _stacked_distance_matrices(distance_matrices)
    upper = np.triu_indices(stack.shape[1], 1)
    vectors = stack[:, upper[0], upper[1]]
    pairwise = np.array([
        spearmanr(vectors[first], vectors[second]).correlation
        for first in range(len(vectors)) for second in range(first + 1, len(vectors))
    ], dtype=np.float64)
    summary = {"n_runs": int(len(vectors)), "n_run_pairs": int(pairwise.size),
               "pairwise_spearman": pairwise}
    for name, function in (("median", np.median), ("min", np.min), ("max", np.max)):
        summary[name] = float(function(pairwise)) if pairwise.size else float("nan")
    return summary


def _resolve_factor_pairs(rules, factor_pairs):
    """Index pairs per factor for the tasks present; factors with fewer than
    two complete pairs are returned separately as skipped."""
    rules = list(rules)
    if len(set(rules)) != len(rules):
        raise ValueError("Task names must be unique.")
    position = {rule: index for index, rule in enumerate(rules)}
    resolved, skipped = {}, []
    for factor, pairs in factor_pairs.items():
        indices = [(position[first], position[second]) for first, second in pairs
                   if first in position and second in position]
        if len(indices) < 2:
            skipped.append(factor)
        else:
            resolved[factor] = indices
    return resolved, skipped


def _difference_gram(gram, pairs):
    """Gram matrix of the difference vectors c[b] - c[a] for ``pairs`` from the
    centroid Gram matrix ``gram`` = C C^T: (c_b - c_a) . (c_d - c_c) =
    G[b, d] - G[b, c] - G[a, d] + G[a, c]."""
    first = np.array([a for a, _ in pairs])
    second = np.array([b for _, b in pairs])
    return (gram[np.ix_(second, second)] - gram[np.ix_(second, first)]
            - gram[np.ix_(first, second)] + gram[np.ix_(first, first)])


def _consistency_from_gram(difference_gram):
    """Mean pairwise cosine among the difference vectors and the share of
    their total variance (about the origin) along their leading direction."""
    norms = np.sqrt(np.clip(np.diag(difference_gram), 0.0, None))
    if np.any(norms <= 0):
        raise ValueError("Task-difference vectors must be nonzero.")
    cosines = np.clip(difference_gram / np.outer(norms, norms), -1.0, 1.0)
    upper = np.triu_indices(norms.size, 1)
    eigenvalues = np.linalg.eigvalsh(difference_gram)
    return {
        "mean_cosine": float(cosines[upper].mean()),
        "pairwise_cosines": cosines[upper],
        "shared_variance_fraction": float(eigenvalues[-1] / eigenvalues.sum()),
    }


def factor_vector_consistency(centroids, rules, factor_pairs=TASK_FACTOR_PAIRS,
                              n_permutations=10000, seed=0):
    """How parallel the task-difference vectors of each task factor are.

    For every factor in ``factor_pairs`` the difference vectors
    ``centroid[to] - centroid[from]`` of its pairs are compared by their mean
    pairwise cosine (1: one shared direction, 0: unrelated), together with
    the fraction of their summed squared length along their leading singular
    direction. The null permutes which centroid belongs to which task, keeping
    the centroid cloud and the pair structure, and recomputes every factor's
    mean cosine (and the mean over factors, ``pooled``) from the permuted Gram
    matrix. One-sided: observed vectors are more parallel than vectors between
    arbitrary tasks. Factors with fewer than two pairs among ``rules`` are
    listed under ``skipped_factors``.
    """
    if n_permutations < 1:
        raise ValueError("Need at least one permutation.")
    centroids = _validated_centroids(centroids)
    rules = list(rules)
    if len(rules) != centroids.shape[0]:
        raise ValueError("Task names must align with the centroid rows.")
    resolved, skipped = _resolve_factor_pairs(rules, factor_pairs)
    if not resolved:
        raise ValueError("No task factor has two complete pairs among the tasks.")
    gram = centroids @ centroids.T
    observed = {factor: _consistency_from_gram(_difference_gram(gram, pairs))
                for factor, pairs in resolved.items()}
    factors = list(resolved)
    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, len(factors)))
    for index in range(n_permutations):
        order = rng.permutation(centroids.shape[0])
        permuted = gram[np.ix_(order, order)]
        for column, factor in enumerate(factors):
            null[index, column] = _consistency_from_gram(
                _difference_gram(permuted, resolved[factor]))["mean_cosine"]
    result = {"factors": {}, "skipped_factors": skipped, "seed": int(seed),
              "n_permutations": int(n_permutations),
              "null_statistic": "mean pairwise cosine under shuffled task identities"}
    for column, factor in enumerate(factors):
        entry = dict(observed[factor])
        entry["pairs"] = [(rules[a], rules[b]) for a, b in resolved[factor]]
        entry.update(_null_summary(entry["mean_cosine"], null[:, column]))
        result["factors"][factor] = entry
    pooled = float(np.mean([observed[factor]["mean_cosine"] for factor in factors]))
    result["pooled"] = {"mean_cosine": pooled, "factors": factors}
    result["pooled"].update(_null_summary(pooled, null.mean(axis=1)))
    return result


# ─── Pair-level association with a task-label permutation (Mantel) test ──────
# The distance-angle analysis of state_space_shift.py relates two quantities
# defined per task pair. Pairs share tasks, so the nominal OLS p of a fit over
# the pairs assumes an independence that does not hold; the appropriate null
# relabels the tasks (rows and columns of one matrix together), as the lesion
# analyses do for cluster pairs (multiple_task/lesion_plot.py, Spearman).

# Permutations are drawn one at a time but evaluated in blocks of this size.
_PERMUTATION_BLOCK = 1000


def pair_values_to_matrix(values, pairs, n_items):
    """Symmetric (n_items x n_items) matrix from one value per unordered pair.

    ``pairs`` holds the (i, j) index pair of each value. Every unordered pair
    must appear exactly once, so the matrix is complete; the diagonal is zero.
    """
    values = np.asarray(values, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=int)
    expected = n_items * (n_items - 1) // 2
    if (n_items < 2 or values.ndim != 1 or pairs.shape != (values.size, 2)
            or values.size != expected or not np.all(np.isfinite(values))):
        raise ValueError(f"Need one finite value for each of the {expected} unordered pairs.")
    if np.any(pairs < 0) or np.any(pairs >= n_items) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("Pair indices must name two different items.")
    matrix = np.zeros((n_items, n_items), dtype=np.float64)
    filled = np.zeros((n_items, n_items), dtype=bool)
    for (first, second), value in zip(pairs, values):
        if filled[first, second]:
            raise ValueError(f"Pair ({first}, {second}) appears more than once.")
        matrix[first, second] = matrix[second, first] = value
        filled[first, second] = filled[second, first] = True
    return matrix


def _pearson_rows(rows, centered_reference, reference_norm):
    """Pearson correlation of every row of ``rows`` with one centered vector."""
    centered = rows - rows.mean(axis=-1, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=-1))
    return centered @ centered_reference / (norms * reference_norm)


def mantel_pearson_permutation(x_matrix, y_matrix, n_permutations=10000, seed=0):
    """Two-sided Pearson Mantel test between two symmetric pair matrices.

    r is the Pearson correlation of the strict upper triangles. The null
    permutes the item (task) labels of ``y_matrix``, rows and columns
    together, so every pair keeps its dependence on the shared items, and
    recomputes r. p is two-sided: the share of permutations with |r| at least
    the observed |r|, with the observed labeling counted once. Returns the
    observed r, p, the null correlations and the test metadata.
    """
    x_matrix = _validated_distance_like(x_matrix)
    y_matrix = _validated_distance_like(y_matrix)
    n_items = x_matrix.shape[0]
    if y_matrix.shape != x_matrix.shape or n_items < 3:
        raise ValueError("Mantel test needs two square matrices of one size over >= 3 items.")
    if n_permutations < 1:
        raise ValueError("Need at least one permutation.")
    upper = np.triu_indices(n_items, 1)
    x_pairs = x_matrix[upper]
    y_pairs = y_matrix[upper]
    if np.std(x_pairs) < 1e-12 or np.std(y_pairs) < 1e-12:
        raise ValueError("Pair values must vary for a correlation to exist.")
    centered_x = x_pairs - x_pairs.mean()
    norm_x = np.sqrt((centered_x ** 2).sum())
    observed = float(_pearson_rows(y_pairs[None, :], centered_x, norm_x)[0])
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for start in range(0, n_permutations, _PERMUTATION_BLOCK):
        count = min(_PERMUTATION_BLOCK, n_permutations - start)
        orders = np.stack([rng.permutation(n_items) for _ in range(count)])
        permuted = y_matrix[orders[:, upper[0]], orders[:, upper[1]]]
        null[start:start + count] = _pearson_rows(permuted, centered_x, norm_x)
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (1 + null.size))
    return {
        "r": observed,
        "p_value": p_value,
        "null": null,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if null.size > 1 else float("nan"),
        "n_permutations": int(null.size),
        "n_items": int(n_items),
        "seed": int(seed),
        "sided": "two-sided",
        "statistic": "pearson",
        "null_statistic": "Pearson r over item pairs under permuted item labels (Mantel)",
    }


def _validated_distance_like(matrix):
    """A finite symmetric square matrix (any diagonal), as float64."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T)):
        raise ValueError("Need a finite symmetric square matrix.")
    return matrix


def _rank_pair_matrix(matrix):
    """The symmetric matrix whose pair values are the ranks (ties averaged)
    of ``matrix``'s strict upper-triangle values; zero diagonal."""
    from scipy.stats import rankdata

    matrix = _validated_distance_like(matrix)
    upper = np.triu_indices(matrix.shape[0], 1)
    ranked = np.zeros_like(matrix)
    ranked[upper] = rankdata(matrix[upper])
    return ranked + ranked.T


def mantel_spearman_permutation(x_matrix, y_matrix, n_permutations=10000, seed=0):
    """Two-sided Spearman Mantel test between two symmetric pair matrices.

    Spearman rho is Pearson r on the ranks of the pair values, and relabeling
    the items only permutes the set of unordered pairs, so the permuted
    matrix's pair ranks are the permuted ranks: the test is
    ``mantel_pearson_permutation`` on the two rank matrices (same null, same
    RNG stream). Returns rho under ``"rho"`` with the test metadata.
    """
    result = mantel_pearson_permutation(_rank_pair_matrix(x_matrix), _rank_pair_matrix(y_matrix),
                                        n_permutations=n_permutations, seed=seed)
    result["rho"] = result.pop("r")
    result["statistic"] = "spearman"
    result["null_statistic"] = "Spearman rho over item pairs under permuted item labels (Mantel)"
    return result
