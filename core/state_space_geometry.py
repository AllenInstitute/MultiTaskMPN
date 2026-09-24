"""High-dimensional task-centroid geometry for state-space examples."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


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


def task_centroid_separation(centroids, categories):
    """Compare Euclidean task-center distances with equal category weights.

    Within-category means receive equal weight. Between-category means receive
    equal weight per unordered category pair. The normalized contrast
    (between - within) / (between + within) is scale invariant; larger is better.
    All coordinates are used, with no PCA, whitening, or feature normalization.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    categories = np.asarray(categories)
    if (centroids.ndim != 2 or centroids.shape[1] == 0
            or categories.shape != (centroids.shape[0],)
            or not np.all(np.isfinite(centroids))):
        raise ValueError("Finite task centroids and category labels must align.")
    unique, counts = np.unique(categories, return_counts=True)
    if unique.size < 2 or np.any(counts < 2):
        raise ValueError("Need at least two categories with two tasks each.")
    distances = squareform(pdist(centroids, metric="euclidean"))
    within_means, between_means = [], []
    within_count = between_count = 0
    for category_index, category in enumerate(unique):
        members = np.flatnonzero(categories == category)
        within = distances[np.ix_(members, members)][
            np.triu_indices(members.size, 1)]
        within_means.append(float(within.mean()))
        within_count += int(within.size)
        for other in unique[category_index + 1:]:
            others = np.flatnonzero(categories == other)
            between = distances[np.ix_(members, others)]
            between_means.append(float(between.mean()))
            between_count += int(between.size)
    within_mean = float(np.mean(within_means))
    between_mean = float(np.mean(between_means))
    total = within_mean + between_mean
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Task centroids have no finite nonzero separation.")
    return {
        "score": (between_mean - within_mean) / total,
        "within_distance": within_mean,
        "between_distance": between_mean,
        "n_tasks": int(centroids.shape[0]),
        "n_features": int(centroids.shape[1]),
        "n_categories": int(unique.size),
        "within_pair_count": within_count,
        "between_pair_count": between_count,
    }