"""Task-specific two-dimensional selection metrics for sibling endpoints."""

import numpy as np


DELAYDM_METRIC_NAME = "direction separation / within-direction dispersion"
DMCGO_METRIC_NAME = "cross-task category balanced accuracy"


def _finite_rows(points, *labels):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected an n×2 projection, got {points.shape}")
    arrays = [np.asarray(values) for values in labels]
    if any(values.shape != (points.shape[0],) for values in arrays):
        raise ValueError("projection and endpoint labels disagree")
    finite = np.isfinite(points).all(axis=1)
    return points[finite], [values[finite] for values in arrays]


def delaydm_direction_score(points, stim_idx):
    """Between-direction centroid distance divided by within-direction spread."""
    points, (stim_idx,) = _finite_rows(points, stim_idx)
    directions = np.unique(stim_idx)
    if directions.size < 2:
        raise ValueError("DelayDM score needs at least two stimulus directions")

    centroids = []
    within_direction = []
    for direction in directions:
        group = points[stim_idx == direction]
        if group.shape[0] < 2:
            raise ValueError(
                f"stimulus direction {direction} has fewer than two endpoints")
        centroid = np.mean(group, axis=0)
        centroids.append(centroid)
        within_direction.append(
            float(np.mean(np.linalg.norm(group - centroid, axis=1))))
    centroids = np.asarray(centroids)
    centroid_distances = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(centroid_distances, np.inf)
    # Use each direction's nearest competing centroid so a projection cannot
    # score well by separating only a subset while collapsing other directions.
    between_mean = float(np.mean(np.min(centroid_distances, axis=1)))
    within_mean = float(np.mean(within_direction))
    scale = max(between_mean, 1.0)
    return between_mean / max(within_mean, np.finfo(float).eps * scale)


def _nearest_centroid_balanced_accuracy(train_points, train_labels,
                                        test_points, test_labels):
    categories = np.unique(train_labels)
    if categories.size != 2 or not np.array_equal(
            categories, np.unique(test_labels)):
        raise ValueError("cross-task DMC score requires the same two categories")
    centroids = np.stack([
        np.mean(train_points[train_labels == category], axis=0)
        for category in categories
    ])
    distances = np.sum(
        (test_points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    predicted = categories[np.argmin(distances, axis=1)]
    recalls = [
        np.mean(predicted[test_labels == category] == category)
        for category in categories
    ]
    return float(np.mean(recalls))


def dmcgo_cross_task_category_score(points, group_labels, task_idx):
    """Symmetric category decoding accuracy from one sibling task to the other."""
    points, (group_labels, task_idx) = _finite_rows(
        points, group_labels, task_idx)
    tasks = np.unique(task_idx)
    if tasks.size != 2:
        raise ValueError("DMC cross-task score requires exactly two tasks")
    accuracies = []
    for train_task, test_task in ((tasks[0], tasks[1]),
                                  (tasks[1], tasks[0])):
        train = task_idx == train_task
        test = task_idx == test_task
        accuracies.append(_nearest_centroid_balanced_accuracy(
            points[train], group_labels[train],
            points[test], group_labels[test]))
    return float(np.mean(accuracies))


def task_specific_pc_score(points, family, *, stim_idx=None,
                           group_labels=None, task_idx=None):
    """Score one two-PC endpoint projection using its family-specific goal."""
    if family == "delaydm1":
        if stim_idx is None:
            raise ValueError("DelayDM score requires stim_idx")
        return delaydm_direction_score(points, stim_idx), DELAYDM_METRIC_NAME
    if family == "dmcgo":
        if group_labels is None or task_idx is None:
            raise ValueError("DMCGo score requires group_labels and task_idx")
        return (dmcgo_cross_task_category_score(
            points, group_labels, task_idx), DMCGO_METRIC_NAME)
    raise ValueError(f"no endpoint selection metric for {family!r}")


def best_task_specific_pc_pair(projection, family, *, stim_idx=None,
                               group_labels=None, task_idx=None):
    """Return the highest-scoring pair among every pair of saved PCs."""
    projection = np.asarray(projection, dtype=float)
    if projection.ndim != 2 or projection.shape[1] < 2:
        raise ValueError(f"invalid endpoint projection shape {projection.shape}")
    candidates = []
    metric_name = None
    for pc_x in range(projection.shape[1]):
        for pc_y in range(pc_x + 1, projection.shape[1]):
            try:
                score, metric_name = task_specific_pc_score(
                    projection[:, [pc_x, pc_y]], family,
                    stim_idx=stim_idx, group_labels=group_labels,
                    task_idx=task_idx)
            except ValueError:
                continue
            if np.isfinite(score):
                candidates.append((float(score), pc_x, pc_y))
    if not candidates:
        raise ValueError(f"cannot score any PC pair for {family}")
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    score, pc_x, pc_y = candidates[0]
    return (pc_x + 1, pc_y + 1), score, metric_name
