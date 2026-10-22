"""PC-pair selection for the sibling-family fixed-point figures.

Every pair of the saved PCs is scored by how well it separates the fixed
points into their groups, and the best pair is drawn. The groups are the
stimulus directions for DelayDM and the response categories for DMC (the
memory clusters "respond to category 1 / 2"); both sibling tasks' fixed points
are pooled.
"""

import numpy as np


DELAYDM_METRIC_NAME = "stimulus-direction separation / within-direction spread"
DMCGO_METRIC_NAME = "response-category separation / within-category spread"


def _finite_rows(points, labels):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected an n×2 projection, got {points.shape}")
    labels = np.asarray(labels)
    if labels.shape != (points.shape[0],):
        raise ValueError("projection and fixed-point labels disagree")
    finite = np.isfinite(points).all(axis=1)
    return points[finite], labels[finite]


def separation_components(points, labels):
    """(between, within): mean distance from each group's centroid to the
    nearest other group's centroid, and mean distance of the points from their
    own group's centroid."""
    points, labels = _finite_rows(points, labels)
    groups = np.unique(labels)
    if groups.size < 2:
        raise ValueError("separation needs at least two groups")

    centroids = []
    within_group = []
    for group in groups:
        members = points[labels == group]
        if members.shape[0] < 2:
            raise ValueError(f"group {group} has fewer than two fixed points")
        centroid = np.mean(members, axis=0)
        centroids.append(centroid)
        within_group.append(
            float(np.mean(np.linalg.norm(members - centroid, axis=1))))
    centroids = np.asarray(centroids)
    centroid_distances = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(centroid_distances, np.inf)
    # Each group's nearest competing centroid, so a plane cannot score well by
    # separating only a subset of the groups while collapsing the others.
    between_mean = float(np.mean(np.min(centroid_distances, axis=1)))
    within_mean = float(np.mean(within_group))
    return between_mean, within_mean


def separation_score(points, labels):
    """Between-group centroid distance divided by within-group spread."""
    between_mean, within_mean = separation_components(points, labels)
    scale = max(between_mean, 1.0)
    return between_mean / max(within_mean, np.finfo(float).eps * scale)


def task_specific_pc_score(points, family, *, stim_idx=None, group_labels=None):
    """Score one two-PC projection by the separation of the family's groups:
    stimulus directions for DelayDM, response categories for DMC."""
    if family == "delaydm1":
        if stim_idx is None:
            raise ValueError("DelayDM score requires stim_idx")
        return separation_score(points, stim_idx), DELAYDM_METRIC_NAME
    if family == "dmcgo":
        if group_labels is None:
            raise ValueError("DMCGo score requires group_labels")
        return separation_score(points, group_labels), DMCGO_METRIC_NAME
    raise ValueError(f"no fixed-point separation metric for {family!r}")


def best_task_specific_pc_pair(projection, family, *, stim_idx=None,
                               group_labels=None):
    """Return ((pc_x, pc_y), score, metric_name) for the highest-scoring pair
    among every pair of saved PCs (1-based; ties to the lowest PCs)."""
    projection = np.asarray(projection, dtype=float)
    if projection.ndim != 2 or projection.shape[1] < 2:
        raise ValueError(f"invalid fixed-point projection shape {projection.shape}")
    candidates = []
    metric_name = None
    for pc_x in range(projection.shape[1]):
        for pc_y in range(pc_x + 1, projection.shape[1]):
            try:
                score, metric_name = task_specific_pc_score(
                    projection[:, [pc_x, pc_y]], family,
                    stim_idx=stim_idx, group_labels=group_labels)
            except ValueError:
                continue
            if np.isfinite(score):
                candidates.append((float(score), pc_x, pc_y))
    if not candidates:
        raise ValueError(f"cannot score any PC pair for {family}")
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    score, pc_x, pc_y = candidates[0]
    return (pc_x + 1, pc_y + 1), score, metric_name
