import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from typing import List, Sequence, Optional, Tuple, Dict, Any

def _hierarchical_clustering(data, k_min=3, k_max=40, metric="euclidean"):
    """
    Hierarchical Ward clustering on `data` (observations × features).
    """
    n_obs = data.shape[0]

    pairwise_dists = pdist(data, metric=metric)
    Z = linkage(pairwise_dists, method="ward")
    Z = optimal_leaf_ordering(Z, pairwise_dists)

    best_k, best_score, best_labels = None, -np.inf, None
    k_range = range(max(k_min, 2), min(k_max, n_obs - 1) + 1)
    D_square = squareform(pairwise_dists)

    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        score = silhouette_score(D_square, labels, metric="precomputed")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    leaf_order = dendrogram(Z, no_plot=True)["leaves"]

    return dict(
        linkage=Z,
        leaf_order=leaf_order,
        labels=best_labels,
        k=best_k,
        silhouette=best_score,
    )


def cluster_variance_matrix(V, k_min=3, k_max=40):
    """
    Cluster a variance matrix V (shape: N features × M neurons)
    for both rows and columns.
    """
    V = np.asarray(V)

    row_res = _hierarchical_clustering(V,  k_min, k_max)
    col_res = _hierarchical_clustering(V.T, k_min, k_max)

    return dict(
        row_order=row_res["leaf_order"],
        col_order=col_res["leaf_order"],
        row_labels=row_res["labels"],
        col_labels=col_res["labels"],
        row_k=row_res["k"],
        col_k=col_res["k"],
        row_linkage=row_res["linkage"],   # full row hierarchy
        col_linkage=col_res["linkage"],   # full column hierarchy
    )

# ---------------------------------------------------------------------
#  ---------------------------------------------------
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

    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        score = silhouette_score(D_sq, labels, metric="precomputed")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    leaf_order = dendrogram(Z, no_plot=True)["leaves"]

    return dict(
        linkage=Z,
        leaf_order=leaf_order,
        labels=best_labels,
        k=best_k,
        silhouette=best_score,
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
    V_row_grp = _aggregate_along_axis(V, row_blocks, axis=0)
    row_res = _hierarchical_clustering_forgroup(V_row_grp, k_min, k_max)

    # expand group labels → per-feature
    row_labels = np.take(row_res["labels"], row_map)

    # rebuild full feature order: block-by-block following dendrogram
    row_order = [idx for g in row_res["leaf_order"] for idx in row_blocks[g]]

    col_blocks, col_map = _build_groups(V.shape[1], col_groups)
    V_col_grp = _aggregate_along_axis(V, col_blocks, axis=1)
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
    )