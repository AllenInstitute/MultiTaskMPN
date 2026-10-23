"""PC-pair selection for the sibling-family fixed-point figures.

Every pair of the saved PCs is scored by how well it separates the fixed
points a figure draws into their groups, and the best pair is drawn. For the
long-delay endpoint figures the groups are the stimulus directions for DelayDM
and the response categories for DMC (the memory clusters "respond to
category 1 / 2"), with both sibling tasks' fixed points pooled. The DelayDM
tasks store the same stimulus at places offset by a task-wide translation, so
each task's mean is removed first (`remove_task_offsets`): the score asks
whether the same stimulus clusters across tasks once that offset is gone. For
the DelayDM rule-cue sweep the fixed points at every alpha are grouped by
stimulus direction and the pair must separate them all along the sweep. The
DMC sweep
figure draws each PC on its own row and shows the response clusters swapping
under the sibling's cue, so it pairs the PC on which the clusters keep their
order with the PC on which they swap.
"""

import numpy as np


DELAYDM_METRIC_NAME = ("stimulus-direction separation / within-direction spread, "
                       "each task's mean removed")
DMCGO_METRIC_NAME = "response-category separation / within-category spread"
SWEEP_METRIC_NAME = ("minimum over alpha of stimulus-direction separation / "
                     "within-direction spread")
DMC_SWEEP_METRIC_NAME = ("response-cluster separation at both ends of the "
                         "sweep, on a kept and a swapped PC")
DMC_SWEEP_KEEP_EVERY_ALPHA_METRIC_NAME = (
    "response-cluster separation on a PC kept at every alpha and a PC "
    "swapped between the ends of the sweep")


def _finite_rows(points, labels):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError(f"expected an n×d projection, got {points.shape}")
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


def task_offsets(points, task_idx):
    """{task: mean of that task's finite points}, the task-wide translation."""
    points = np.asarray(points, dtype=float)
    task_idx = np.asarray(task_idx)
    if task_idx.shape != (points.shape[0],):
        raise ValueError("points and task labels disagree")
    offsets = {}
    for task in np.unique(task_idx):
        members = points[task_idx == task]
        members = members[np.isfinite(members).all(axis=1)]
        if members.shape[0] == 0:
            raise ValueError(f"task {task} has no finite points")
        offsets[task] = members.mean(axis=0)
    return offsets


def remove_task_offsets(points, task_idx):
    """`points` with each task's mean (`task_offsets`) subtracted, so tasks
    that store the same stimuli at translated places overlap."""
    points = np.asarray(points, dtype=float)
    task_idx = np.asarray(task_idx)
    offsets = task_offsets(points, task_idx)
    return points - np.stack([offsets[task] for task in task_idx])


def task_specific_pc_score(points, family, *, stim_idx=None, group_labels=None,
                           task_idx=None):
    """Score one two-PC projection by the separation of the family's groups:
    stimulus directions for DelayDM, with each task's mean removed first, and
    response categories for DMC."""
    if family == "delaydm1":
        if stim_idx is None or task_idx is None:
            raise ValueError("DelayDM score requires stim_idx and task_idx")
        return (separation_score(remove_task_offsets(points, task_idx), stim_idx),
                DELAYDM_METRIC_NAME)
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
                               group_labels=None, task_idx=None):
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
            projection[:, [i, j]], family, stim_idx=stim_idx,
            group_labels=group_labels, task_idx=task_idx)[0],
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


def dmc_sweep_axis_scores(projection, group_labels, keep_at_every_alpha=False):
    """(keep, swap): per-PC scores of one DMC rule-cue sweep.

    ``projection`` is ``(n_alpha, n_trials, n_pc)`` and ``group_labels`` are
    the trials' response clusters at alpha = 0 (two groups). On each PC the
    clusters' separation along that PC alone (`separation_score`) is taken at
    both ends of the sweep, and the smaller of the two scores the PC as a
    kept axis when the clusters keep their order from alpha = 0 to alpha = 1,
    or as a swapped axis when the order reverses (the response flips under
    the sibling's cue); its other score is 0. With `keep_at_every_alpha` a
    kept PC must keep the order at every alpha, scoring its smallest
    separation over all of them, so the clusters stay apart along the whole
    top row; a swapped PC is still judged at the ends, since a swap has to
    cross somewhere in between.
    """
    projection = np.asarray(projection, dtype=float)
    labels = np.asarray(group_labels)
    if projection.ndim != 3 or projection.shape[0] < 2:
        raise ValueError("expected (n_alpha >= 2, n_trials, n_pc) sweep "
                         f"projections, got {projection.shape}")
    groups = np.unique(labels)
    if groups.size != 2:
        raise ValueError("the DMC sweep score needs exactly two response clusters")
    n_pc = projection.shape[2]
    keep, swap = np.zeros(n_pc), np.zeros(n_pc)
    for pc in range(n_pc):
        separations, orders = [], []
        for points in projection:
            column = points[:, [pc]]
            separations.append(separation_score(column, labels))
            orders.append(np.sign(np.nanmean(column[labels == groups[1], 0])
                                  - np.nanmean(column[labels == groups[0], 0])))
        ends = min(separations[0], separations[-1])
        if orders[0] * orders[-1] < 0:
            swap[pc] = ends
        elif orders[0] * orders[-1] > 0:
            if not keep_at_every_alpha:
                keep[pc] = ends
            elif all(order == orders[0] for order in orders):
                keep[pc] = min(separations)
    return keep, swap


def best_dmc_sweep_pc_pair(projections, group_labels, keep_at_every_alpha=False):
    """Return ((keep_pc, swap_pc), score, metric_name) for the DMC rule-cue
    sweep: the PC on which the response clusters stay apart in the same order
    and the PC on which they swap, each the best of its kind
    (`dmc_sweep_axis_scores`, with `keep_at_every_alpha` passed on; 1-based,
    the kept PC first, ties to the lowest PC). The score is the smaller of
    the two, so both rows of the figure show the clusters. ``projections``
    are ``(n_alpha, n_trials, n_pc)`` arrays, one per base rule, with
    matching ``group_labels``; a PC must keep (or swap) in every one, its
    scores being their minimum."""
    projections = [np.asarray(p, dtype=float) for p in projections]
    if not projections or any(p.ndim != 3 for p in projections):
        raise ValueError("expected (n_alpha, n_trials, n_pc) sweep projections")
    n_pc = projections[0].shape[2]
    if any(p.shape[2] != n_pc for p in projections) or n_pc < 2:
        raise ValueError("sweep projections must share at least two PCs")
    keeps, swaps = zip(*(dmc_sweep_axis_scores(p, labels, keep_at_every_alpha)
                         for p, labels in zip(projections, group_labels)))
    keep, swap = np.min(keeps, axis=0), np.min(swaps, axis=0)
    keep_pc = int(np.argmax(keep))
    swap_pc = next(int(pc) for pc in np.argsort(-swap, kind="stable")
                   if pc != keep_pc)
    score = float(min(keep[keep_pc], swap[swap_pc]))
    metric_name = (DMC_SWEEP_KEEP_EVERY_ALPHA_METRIC_NAME if keep_at_every_alpha
                   else DMC_SWEEP_METRIC_NAME)
    return (keep_pc + 1, swap_pc + 1), score, metric_name
