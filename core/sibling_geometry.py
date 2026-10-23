"""PC-pair selection for the sibling-family fixed-point figures.

Every pair of the saved PCs is scored by how well it separates the fixed
points a figure draws into their groups, and the best pair is drawn. For the
long-delay endpoint figures the groups are the stimulus directions for DelayDM
and the response categories for DMC (the memory clusters "respond to
category 1 / 2"), with both sibling tasks' fixed points pooled. For the DelayDM
rule-cue sweep the fixed points at every alpha are grouped by stimulus
direction and the pair must separate them all along the sweep.
"""

import numpy as np


DELAYDM_METRIC_NAME = "stimulus-direction separation / within-direction spread"
DMCGO_METRIC_NAME = "response-category separation / within-category spread"
SWEEP_METRIC_NAME = ("minimum over alpha of stimulus-direction separation / "
                     "within-direction spread")


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


def _best_pc_pair(n_pc, score_pair, context):
    """((pc_x, pc_y), score) maximizing `score_pair(i, j)` over every pair of
    `n_pc` PCs (1-based; ties to the lowest PCs). Pairs that raise ValueError
    or score non-finite are skipped."""
    candidates = []
    for pc_x in range(n_pc):
        for pc_y in range(pc_x + 1, n_pc):
            try:
                score = float(score_pair(pc_x, pc_y))
            except ValueError:
                continue
            if np.isfinite(score):
                candidates.append((score, pc_x, pc_y))
    if not candidates:
        raise ValueError(f"cannot score any PC pair for {context}")
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    score, pc_x, pc_y = candidates[0]
    return (pc_x + 1, pc_y + 1), score


def best_task_specific_pc_pair(projection, family, *, stim_idx=None,
                               group_labels=None):
    """Return ((pc_x, pc_y), score, metric_name) for the pair of saved PCs
    that best separates the fixed points (`task_specific_pc_score`)."""
    projection = np.asarray(projection, dtype=float)
    if projection.ndim != 2 or projection.shape[1] < 2:
        raise ValueError(f"invalid fixed-point projection shape {projection.shape}")
    metric_name = {"delaydm1": DELAYDM_METRIC_NAME,
                   "dmcgo": DMCGO_METRIC_NAME}.get(family)
    pair, score = _best_pc_pair(
        projection.shape[1],
        lambda i, j: task_specific_pc_score(
            projection[:, [i, j]], family,
            stim_idx=stim_idx, group_labels=group_labels)[0],
        family)
    return pair, score, metric_name


def sweep_separation_score(projections, stim_indices):
    """Separation of a rule-cue sweep's fixed points by stimulus direction,
    taken at its worst: `separation_score` at every alpha of every base rule's
    sweep, minimized. ``projections`` are ``(n_alpha, n_trials, 2)`` arrays,
    one per base rule, with matching ``stim_indices``. A trial that hops to
    another direction's attractor at some alpha lowers the score there, as
    does a merge of the directions anywhere along the sweep."""
    scores = []
    for projection, stim in zip(projections, stim_indices):
        projection = np.asarray(projection, dtype=float)
        if projection.ndim != 3 or projection.shape[2] != 2:
            raise ValueError("expected (n_alpha, n_trials, 2) sweep projections, "
                             f"got {projection.shape}")
        scores.extend(separation_score(points, stim) for points in projection)
    if not scores:
        raise ValueError("no sweep to score")
    return float(np.min(scores))


def best_sweep_pc_pair(projections, stim_indices):
    """Return ((pc_x, pc_y), score, metric_name) for the pair of saved PCs
    whose rule-cue sweeps stay best separated by stimulus direction at every
    alpha (`sweep_separation_score`). ``projections`` are
    ``(n_alpha, n_trials, n_pc)`` arrays, one per base rule."""
    projections = [np.asarray(p, dtype=float) for p in projections]
    if not projections or any(p.ndim != 3 for p in projections):
        raise ValueError("expected (n_alpha, n_trials, n_pc) sweep projections")
    n_pc = projections[0].shape[2]
    if any(p.shape[2] != n_pc for p in projections) or n_pc < 2:
        raise ValueError("sweep projections must share at least two PCs")
    pair, score = _best_pc_pair(
        n_pc,
        lambda i, j: sweep_separation_score(
            [p[:, :, [i, j]] for p in projections], stim_indices),
        "the rule-cue sweep")
    return pair, score, SWEEP_METRIC_NAME
