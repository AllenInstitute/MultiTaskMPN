import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score
from typing import List, Sequence, Optional, Tuple, Dict, Any

def _hierarchical_clustering(data, k_min=3, k_max=40, metric="euclidean", method="ward"):
    """
    Hierarchical clustering on `data` (observations × features) with Ward by default.
    Chooses k in [k_min, k_max] by max silhouette (precomputed distances).
    Returns leaf order for visualization and the labels at the optimal k.
    """
    n_obs = data.shape[0]
    k_min = max(k_min, 2)
    k_max = min(k_max, n_obs - 1)
    if k_max < k_min:
        raise ValueError("Not enough observations for the requested k-range.")

    # --- Build the linkage ---
    if method.lower() == "ward":
        if metric.lower() != "euclidean":
            warnings.warn("Ward requires Euclidean distances; overriding metric to 'euclidean'.")
        # Ward must use the observation matrix directly
        Z = linkage(data, method="ward", metric="euclidean")
        pairwise_dists = pdist(data, metric="euclidean")
    else:
        # For other methods, a condensed distance matrix is fine
        pairwise_dists = pdist(data, metric=metric)
        Z = linkage(pairwise_dists, method=method)

    # Optimal leaf ordering for a nicer dendrogram/heatmap
    Z = optimal_leaf_ordering(Z, pairwise_dists)

    # Precompute full distance matrix for silhouette
    D_square = squareform(pairwise_dists)

    # --- Model selection by silhouette ---
    best_k, best_score, best_labels = None, -np.inf, None
    score_recording, cut_thresholds = {}, {}

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        # Silhouette on *distances* to match the description
        score = silhouette_score(D_square, labels, metric="precomputed")
        score_recording[k] = score

        cut_idx = (n_obs - k) - 1  # zero-based
        if 0 <= cut_idx < Z.shape[0]:
            cut_thresholds[k] = float(np.nextafter(Z[cut_idx, 2], np.inf))
        else:
            cut_thresholds[k] = None

        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    leaf_order = dendrogram(Z, no_plot=True)["leaves"]

    return dict(
        linkage=Z,
        leaf_order=leaf_order,
        labels=best_labels,
        k=best_k,
        silhouette=best_score,
        score_recording=score_recording,
        cut_threshold=cut_thresholds.get(best_k, None)
    )


def cluster_variance_matrix(V, k_min=3, k_max=40, metric="euclidean", method="ward"):
    """
    Cluster a variance matrix V (rows = one axis of units/tasks, cols = the other).
    Runs hierarchical clustering on both axes, selects k by silhouette (3..40).
    """
    V = np.asarray(V)

    row_res = _hierarchical_clustering(V,   k_min, k_max, metric, method)
    col_res = _hierarchical_clustering(V.T, k_min, k_max, metric, method)

    return dict(
        row_order=row_res["leaf_order"],
        col_order=col_res["leaf_order"],
        row_labels=row_res["labels"],
        col_labels=col_res["labels"],
        row_k=row_res["k"],
        col_k=col_res["k"],
        row_linkage=row_res["linkage"],
        col_linkage=col_res["linkage"],
        row_score_recording=row_res["score_recording"],
        col_score_recording=col_res["score_recording"],
        row_cut_threshold=row_res["cut_threshold"],
        col_cut_threshold=col_res["cut_threshold"],
    )

def _hierarchical_clustering_repeat(
    data,
    k_min=3,
    k_max=40,
    metric="euclidean",
    method="ward",
    *,
    n_repeats=1,
    resample_features_frac=1.0,
    jitter_std=0.0,
    random_state=None
):
    """
    Hierarchical clustering on `data` (observations × features) with stability repeats.

    - For each repeat, optionally resample features (columns) and add tiny Gaussian jitter
      to break ties and probe stability.
    - For each k in k_range, compute silhouette; aggregate mean/std across repeats.
    - Choose k with best mean silhouette.
    - For that k, pick the labeling whose ARI-agreement with other repeats is maximal.
    """
    rng = np.random.default_rng(random_state)
    n_obs, n_feat = data.shape

    k_range = range(max(k_min, 2), min(k_max, n_obs - 1) + 1)

    # Per-repeat storage
    per_repeat = []  # list of dicts: { 'Z':..., 'leaf_order':..., 'labels_by_k':{k:labels}, 'scores_by_k':{k:score} }

    for r in range(n_repeats):
        # 1) feature resampling (bagging in feature space)
        if resample_features_frac < 1.0:
            m = max(1, int(np.ceil(resample_features_frac * n_feat)))
            feat_idx = rng.choice(n_feat, size=m, replace=True)
            Xr = data[:, feat_idx]
        else:
            Xr = data

        # 2) tiny jitter (optional) to break exact distance ties
        if jitter_std > 0.0:
            Xr = Xr + rng.normal(0.0, jitter_std, size=Xr.shape)

        pairwise_dists = pdist(Xr, metric=metric)
        Z = linkage(pairwise_dists, method=method)
        Z = optimal_leaf_ordering(Z, pairwise_dists)

        D_square = squareform(pairwise_dists)

        labels_by_k = {}
        scores_by_k = {}

        for k in k_range:
            labels = fcluster(Z, k, criterion="maxclust")
            score = silhouette_score(D_square, labels, metric="precomputed")
            labels_by_k[k] = labels
            scores_by_k[k] = score

        leaf_order = dendrogram(Z, no_plot=True)["leaves"]
        per_repeat.append(dict(Z=Z, leaf_order=leaf_order,
                               labels_by_k=labels_by_k, scores_by_k=scores_by_k))

    # Aggregate silhouettes over repeats
    score_recording_mean = {k: np.mean([rep["scores_by_k"][k] for rep in per_repeat]) for k in k_range}
    score_recording_std  = {k: np.std( [rep["scores_by_k"][k] for rep in per_repeat]) for k in k_range}

    best_k = max(score_recording_mean, key=score_recording_mean.get)
    best_score_mean = score_recording_mean[best_k]

    # Pick a stable labeling for best_k: choose repeat with highest mean ARI to others
    labels_list = [rep["labels_by_k"][best_k] for rep in per_repeat]
    if n_repeats == 1:
        # trivial case
        best_rep_idx = 0
    else:
        R = len(labels_list)
        ari_mat = np.zeros((R, R))
        for i in range(R):
            for j in range(i+1, R):
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                ari_mat[i, j] = ari_mat[j, i] = ari
        mean_ari = ari_mat.mean(axis=1)
        best_rep_idx = int(np.argmax(mean_ari))

    best_labels = labels_list[best_rep_idx]
    Z_best      = per_repeat[best_rep_idx]["Z"]
    leaf_order  = per_repeat[best_rep_idx]["leaf_order"]

    return dict(
        linkage=Z_best,
        leaf_order=leaf_order,
        labels=best_labels,
        k=best_k,
        silhouette=best_score_mean,
        score_recording_mean=score_recording_mean,
        score_recording_std=score_recording_std,
        _repeats=len(per_repeat),
        _params=dict(resample_features_frac=resample_features_frac, jitter_std=jitter_std)
    )


def cluster_variance_matrix_repeat(
    V,
    k_min=3, k_max=40, metric="euclidean", method="ward",
    *,
    n_repeats=10,
    resample_features_frac=1.0,
    jitter_std=0.0,
    random_state=None
):
    """
    Cluster a variance matrix V (shape: N features × M neurons)
    for both rows (features) and columns (neurons), with repeat-based stabilization.
    """
    V = np.asarray(V)

    row_res = _hierarchical_clustering_repeat(
        V, k_min, k_max, metric, method,
        n_repeats=n_repeats,
        resample_features_frac=resample_features_frac,
        jitter_std=jitter_std,
        random_state=random_state
    )

    # For columns, observations are neurons (columns), features are rows; we resample rows of V for stability
    col_res = _hierarchical_clustering_repeat(
        V.T, k_min, k_max, metric, method,
        n_repeats=n_repeats,
        resample_features_frac=resample_features_frac,
        jitter_std=jitter_std,
        random_state=None if random_state is None else (random_state + 1)  # different stream
    )

    return dict(
        row_order=row_res["leaf_order"],
        col_order=col_res["leaf_order"],
        row_labels=row_res["labels"],
        col_labels=col_res["labels"],
        row_k=row_res["k"],
        col_k=col_res["k"],
        row_linkage=row_res["linkage"],
        col_linkage=col_res["linkage"],
        row_score_recording_mean=row_res["score_recording_mean"],
        row_score_recording_std=row_res["score_recording_std"],
        col_score_recording_mean=col_res["score_recording_mean"],
        col_score_recording_std=col_res["score_recording_std"]
    )
    
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def _hierarchical_clustering_forgroup(
    data: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    Run Ward hierarchical clustering on `data` (observations × features).

    Returns: {
        'linkage'   : condensed linkage matrix (scipy format),
        'leaf_order': list of observation indices in optimized dendrogram order,
        'labels'    : 1-based flat-cluster labels at best_k,
        'k'         : number of clusters chosen by silhouette maximisation,
        'silhouette': best silhouette score,
    }
    """
    n_obs = data.shape[0]
    if n_obs < 2:                             # nothing to cluster
        return dict(
            linkage=None,
            leaf_order=list(range(n_obs)),
            labels=np.ones(n_obs, dtype=int),
            k=1,
            silhouette=np.nan,
        )

    pairwise = pdist(data, metric=metric)
    Z = optimal_leaf_ordering(linkage(pairwise, method="ward"), pairwise)

    best_k, best_score, best_labels = None, -np.inf, None
    D_sq = squareform(pairwise)
    k_range = range(max(k_min, 2), min(k_max, n_obs - 1) + 1)

    score_recording = {}
    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        score = silhouette_score(D_sq, labels, metric="precomputed")
        score_recording[k] = score # register
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    leaf_order = dendrogram(Z, no_plot=True)["leaves"]

    return dict(
        linkage=Z,
        leaf_order=leaf_order,
        labels=best_labels,
        k=best_k,
        silhouette=best_score,
        score_recording=score_recording
    )



def _build_groups(
    n_items: int,
    blocks: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Ensure a full, disjoint partition of {0,…,n_items-1}.

    Parameters
    ----------
    n_items : total number of elements along the axis
    blocks  : list of lists (possibly None).  Each sub-list is a user-defined block.

    Returns
    -------
    groups  : list of lists, length = n_groups
    item_to_group : int array of length n_items mapping each item to its group index
    """
    item_to_group = -np.ones(n_items, dtype=int)
    groups: List[List[int]] = []

    # 1. user-defined blocks
    if blocks is not None:
        for b in blocks:
            b = sorted(set(b))
            if len(b) == 0:
                continue
            g_idx = len(groups)
            groups.append(b)
            for i in b:
                if i < 0 or i >= n_items:
                    raise IndexError(f"index {i} out of range 0…{n_items-1}")
                if item_to_group[i] != -1:
                    raise ValueError(f"index {i} appears in multiple blocks")
                item_to_group[i] = g_idx

    # 2. singletons for anything not covered above
    for i in range(n_items):
        if item_to_group[i] == -1:
            item_to_group[i] = len(groups)
            groups.append([i])

    return groups, item_to_group


def _aggregate_along_axis(
    arr: np.ndarray,
    groups: List[List[int]],
    axis: int = 0,
    reduce: str = "mean",
) -> np.ndarray:
    """
    Stack aggregated slices along the chosen axis.

    * axis=0 : groups act on rows   → output shape (n_groups, …)
    * axis=1 : groups act on cols   → output shape (…, n_groups)
    """
    if reduce not in {"mean", "median"}:
        raise ValueError("reduce must be 'mean' or 'median'")

    agg_fn = np.mean if reduce == "mean" else np.median
    out = []
    for idxs in groups:
        if axis == 0:
            out.append(agg_fn(arr[idxs, :], axis=0))
        else:  # axis==1
            out.append(agg_fn(arr[:, idxs], axis=1))
    return np.stack(out, axis=axis)


def cluster_variance_matrix_forgroup(
    V: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
    row_groups: Optional[Sequence[Sequence[int]]] = None,
    col_groups: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    """
    Bi-directional hierarchical clustering of a variance matrix V
    (shape: N_features × M_neurons).

    Parameters
    ----------
    V           : (N × M) variance matrix
    k_min/k_max : min / max candidate #clusters for silhouette selection
    row_groups  : optional list of feature-index blocks that must stay together
    col_groups  : optional list of neuron-index blocks that must stay together

    Returns
    -------
    dict with:
      row_order           – list of feature indices (full, flattened order)
      col_order           – list of neuron  indices
      row_group_order     – order of the row blocks
      col_group_order     – order of the col blocks
      row_labels          – per-feature     flat-cluster labels
      col_labels          – per-neuron      flat-cluster labels
      row_group_labels    – per-row-block   labels
      col_group_labels    – per-col-block   labels
      row_k / col_k       – chosen k for rows / cols
      row_linkage / col_linkage – full linkage matrices on blocks
    """
    V = np.asarray(V)

    # ---------- rows --------------------------------------------------
    row_blocks, row_map = _build_groups(V.shape[0], row_groups)
    V_row_grp = _aggregate_along_axis(V, row_blocks, axis=0, reduce="mean")
    row_res = _hierarchical_clustering_forgroup(V_row_grp, k_min, k_max)

    # expand group labels → per-feature
    row_labels = np.take(row_res["labels"], row_map)

    # rebuild full feature order: block-by-block following dendrogram
    row_order = [idx for g in row_res["leaf_order"] for idx in row_blocks[g]]

    col_blocks, col_map = _build_groups(V.shape[1], col_groups)
    V_col_grp = _aggregate_along_axis(V, col_blocks, axis=1, reduce="mean")
    col_res = _hierarchical_clustering_forgroup(V_col_grp.T, k_min, k_max)

    col_labels = np.take(col_res["labels"], col_map)
    col_order = [idx for g in col_res["leaf_order"] for idx in col_blocks[g]]

    return dict(
        # fine-grained ordering / labels
        row_order=row_order,
        col_order=col_order,
        row_labels=row_labels,
        col_labels=col_labels,
        # block-level results
        row_group_order=row_res["leaf_order"],
        col_group_order=col_res["leaf_order"],
        row_group_labels=row_res["labels"],
        col_group_labels=col_res["labels"],
        row_k=row_res["k"],
        col_k=col_res["k"],
        row_linkage=row_res["linkage"],
        col_linkage=col_res["linkage"],
        row_score_recording=row_res["score_recording"], 
        col_score_recording=col_res["score_recording"]
    )