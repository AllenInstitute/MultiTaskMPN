import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.stats import wilcoxon

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import List, Sequence, Optional, Tuple, Dict, Any
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------
# Original Clustering 
# ---------------------------------------------------------------------

def _hierarchical_clustering(data, k_min=3, k_max=40, metric="euclidean", method="ward"):
    """
    """
    n_obs = data.shape[0]
    k_min = max(k_min, 2)
    k_max = min(k_max, n_obs - 1)
    if k_max < k_min:
        raise ValueError("Not enough observations for the requested k-range.")

    if method.lower() == "ward":
        Z = linkage(data, method="ward", metric="euclidean")
        pairwise_dists = pdist(data, metric="euclidean")
    else:
        pairwise_dists = pdist(data, metric=metric)
        Z = linkage(pairwise_dists, method=method)

    Z = optimal_leaf_ordering(Z, pairwise_dists)

    D_square = squareform(pairwise_dists)

    best_k, best_score, best_labels = None, -np.inf, None
    score_recording, cut_thresholds = {}, {}

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        score = silhouette_score(D_square, labels, metric="precomputed")
        score_recording[k] = score

        cut_idx = (n_obs - k) - 1  
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

# ---------------------------------------------------------------------
# Clustering with multiple repeats
# ---------------------------------------------------------------------

def _breaks(lbls):
    """
    """
    idx = np.nonzero(np.diff(lbls))[0] + 1
    return idx.tolist()

def labels_at_k(Z, k):
    """
    """
    # Z has shape (n_obs-1, 4); Z[i,2] is the merge height of the (i+1)-th merge
    n_obs = Z.shape[0] + 1
    # index of the last merge included when you still have exactly k clusters
    cut_idx = (n_obs - k) - 1           # 0-based
    # choose a threshold strictly between the two adjacent heights
    t_low  = Z[cut_idx, 2]
    t_high = Z[cut_idx + 1, 2] if (cut_idx + 1) < Z.shape[0] else np.inf
    t = np.nextafter(t_low, t_high)     # just above t_low, still below t_high
    return fcluster(Z, t, criterion="distance")

def _score_threshold_from_best(best_score, silhouette_tol=0.02, tol_mode="relative"):
    """
    Compute the score threshold used for tolerance-based model selection.

    Parameters
    ----------
    best_score : float
        Reference silhouette score.
    silhouette_tol : float
        Tolerance value. Interpreted according to ``tol_mode``.
    tol_mode : {"relative", "absolute"}
        - "relative": threshold = best_score * (1 - silhouette_tol)
        - "absolute": threshold = best_score - silhouette_tol
    """
    if tol_mode == "relative":
        return float(best_score) * (1.0 - float(silhouette_tol))
    if tol_mode == "absolute":
        return float(best_score) - float(silhouette_tol)
    raise ValueError("tol_mode must be 'relative' or 'absolute'.")


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
    random_state=None,
    select_min_k_within_tol=True,
    silhouette_tol=0.02,
    tol_mode="relative",
):
    """
    Hierarchical clustering with stability repeats.

    Key behavior:
        - Repeats (feature bootstrap / jitter) are used ONLY to estimate the
          silhouette-vs-k curve.
        - The returned linkage / leaf_order are computed ONCE on the original
          unperturbed data.
        - The primary returned labels / k correspond to the strict argmax of
          the mean silhouette across repeats.
        - The alt/tol returned labels / k correspond to the smallest k whose
          mean silhouette is within tolerance of that strict best mean score.
        - We still pick a representative repeat (max mean ARI at the selected
          tolerant k) for diagnostics, but we do NOT return its linkage as the
          final dendrogram.
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n_obs, n_feat = data.shape

    def _compute_linkage_leaforder(X):
        """Compute linkage + optimal leaf order; return (Z, condensed_dists, leaf_order)."""
        X = np.asarray(X)

        if method.lower() == "ward":
            Z = linkage(X, method="ward")
            d = pdist(X, metric="euclidean")
        else:
            d = pdist(X, metric=metric)
            Z = linkage(d, method=method)

        Z = optimal_leaf_ordering(Z, d)
        leaf_order = dendrogram(Z, no_plot=True)["leaves"]
        return Z, d, leaf_order

    def _cut_threshold_from_Z(Z, k):
        """Threshold just above the last merge height when there are k clusters."""
        cut_idx = (n_obs - k) - 1
        if 0 <= cut_idx < Z.shape[0]:
            return float(np.nextafter(Z[cut_idx, 2], np.inf))
        return None

    k_min = max(int(k_min), 2)
    k_max = min(int(k_max), n_obs - 1)
    if k_max < k_min:
        raise ValueError("Not enough observations for the requested k-range.")
    k_range = range(k_min, k_max + 1)
    k_values = np.array(list(k_range), dtype=int)

    per_repeat = []
    for _ in range(int(n_repeats)):
        if resample_features_frac < 1.0:
            m = max(1, int(np.ceil(resample_features_frac * n_feat)))
            feat_idx = rng.choice(n_feat, size=m, replace=True)
            Xr = data[:, feat_idx]
        else:
            Xr = data

        if jitter_std > 0.0:
            Xr = Xr + rng.normal(0.0, jitter_std, size=Xr.shape)

        Zr, dr, leaf_order_r = _compute_linkage_leaforder(Xr)
        D_square = squareform(dr)

        labels_by_k = {}
        scores_by_k = {}
        cut_thresholds = {}
        for k in k_range:
            labels = fcluster(Zr, k, criterion="maxclust")
            n_clusters = np.unique(labels).size

            if n_clusters < 2 or n_clusters >= n_obs:
                scores_by_k[k] = -np.inf
            else:
                scores_by_k[k] = float(silhouette_score(D_square, labels, metric="precomputed"))

            labels_by_k[k] = labels
            cut_thresholds[k] = _cut_threshold_from_Z(Zr, k)

        per_repeat.append(
            dict(
                Z=Zr,
                leaf_order=leaf_order_r,
                labels_by_k=labels_by_k,
                scores_by_k=scores_by_k,
                cut_thresholds=cut_thresholds,
            )
        )

    score_recording_all = np.array(
        [[rep["scores_by_k"][k] for k in k_values] for rep in per_repeat],
        dtype=float,
    )
    score_recording_mean = {
        int(k): float(np.mean(score_recording_all[:, idx])) for idx, k in enumerate(k_values)
    }
    score_recording_std = {
        int(k): float(np.std(score_recording_all[:, idx])) for idx, k in enumerate(k_values)
    }

    strict_best_k = max(score_recording_mean, key=score_recording_mean.get)
    strict_best_score = score_recording_mean[strict_best_k]

    if select_min_k_within_tol:
        primary_thresh = _score_threshold_from_best(
            strict_best_score, silhouette_tol=silhouette_tol, tol_mode=tol_mode
        )
        primary_candidates = [k for k in k_range if score_recording_mean[k] >= primary_thresh]
        best_k = min(primary_candidates) if primary_candidates else strict_best_k
    else:
        primary_thresh = strict_best_score
        primary_candidates = [strict_best_k]
        best_k = strict_best_k

    best_k_score_mean = score_recording_mean[best_k]

    labels_list = [rep["labels_by_k"][best_k] for rep in per_repeat]
    if len(labels_list) == 1:
        best_rep_idx = 0
        mean_ari_val = 1.0
    else:
        R = len(labels_list)
        ari_mat = np.zeros((R, R), dtype=float)
        for i in range(R):
            for j in range(i + 1, R):
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                ari_mat[i, j] = ari_mat[j, i] = ari
        mean_ari = ari_mat.mean(axis=1)
        best_rep_idx = int(np.argmax(mean_ari))
        mean_ari_val = float(mean_ari[best_rep_idx])

    best_rep = per_repeat[best_rep_idx]

    alt_thresh = _score_threshold_from_best(
        strict_best_score, silhouette_tol=silhouette_tol, tol_mode=tol_mode
    )
    alt_candidates = [k for k in k_range if score_recording_mean[k] >= alt_thresh]
    alt_k = min(alt_candidates) if alt_candidates else strict_best_k

    Z0, d0, leaf0 = _compute_linkage_leaforder(data)

    strict_labels = fcluster(Z0, strict_best_k, criterion="maxclust")
    strict_cut_threshold = _cut_threshold_from_Z(Z0, strict_best_k)

    final_alt_labels = fcluster(Z0, alt_k, criterion="maxclust")
    final_alt_cut_threshold = _cut_threshold_from_Z(Z0, alt_k)

    return dict(
        linkage=Z0,
        leaf_order=leaf0,
        labels=strict_labels,
        k=strict_best_k,
        cut_threshold=strict_cut_threshold,

        alt_k=alt_k,
        alt_labels=final_alt_labels,
        alt_linkage=Z0,
        alt_leaf_order=leaf0,
        alt_cut_threshold=final_alt_cut_threshold,

        silhouette_strict_best=strict_best_score,
        silhouette_at_k=best_k_score_mean,
        score_recording_mean=score_recording_mean,
        score_recording_std=score_recording_std,
        score_recording_all=score_recording_all,
        k_values=k_values,

        rep_linkage=best_rep["Z"],
        rep_leaf_order=best_rep["leaf_order"],
        rep_labels_at_k=best_rep["labels_by_k"][best_k],
        rep_cut_threshold_at_k=best_rep["cut_thresholds"][best_k],

        _repeats=len(per_repeat),
        _params=dict(
            resample_features_frac=resample_features_frac,
            jitter_std=jitter_std,
            method=method,
            metric=metric,
            mean_ari_at_best_k=mean_ari_val,
            select_min_k_within_tol=select_min_k_within_tol,
            silhouette_tol=silhouette_tol,
            tol_mode=tol_mode,
            chosen_k=best_k,
            chosen_score=best_k_score_mean,
            strict_best_k=strict_best_k,
            strict_best_score=strict_best_score,
            primary_thresh=primary_thresh,
            primary_candidates=primary_candidates,
            alt_thresh=alt_thresh,
            alt_candidates=alt_candidates,
        ),
    )

def cluster_variance_matrix_repeat(
    V,
    k_min=3, k_max=40, metric="euclidean", method="ward",
    *,
    n_repeats=10,
    resample_features_frac=1.0,
    jitter_std=0.0,
    random_state=None,
    select_min_k_within_tol=True,
    silhouette_tol=0.02,
    tol_mode="relative",
):
    """
    Cluster a variance matrix V (shape: N_features * M_neurons)
    for both rows (features) and columns (neurons), with repeat-based stabilization.

    Returns mean/std silhouette curves for rows and cols.
    row_k / col_k correspond to the strict mean-silhouette argmax.
    row_tol_* / col_tol_* correspond to the smaller tolerance-selected solution.
    """
    V = np.asarray(V)

    row_res = _hierarchical_clustering_repeat(
        V, k_min, k_max, metric, method,
        n_repeats=n_repeats,
        resample_features_frac=resample_features_frac,
        jitter_std=jitter_std,
        random_state=random_state,
        select_min_k_within_tol=select_min_k_within_tol,
        silhouette_tol=silhouette_tol,
        tol_mode=tol_mode,
    )

    col_res = _hierarchical_clustering_repeat(
        V.T, k_min, k_max, metric, method,
        n_repeats=n_repeats,
        resample_features_frac=resample_features_frac,
        jitter_std=jitter_std,
        random_state=None if random_state is None else (random_state + 1),
        select_min_k_within_tol=select_min_k_within_tol,
        silhouette_tol=silhouette_tol,
        tol_mode=tol_mode,
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
        row_score_recording_all=row_res["score_recording_all"],
        row_k_values=row_res["k_values"],
        col_score_recording_mean=col_res["score_recording_mean"],
        col_score_recording_std=col_res["score_recording_std"],
        col_score_recording_all=col_res["score_recording_all"],
        col_k_values=col_res["k_values"],
        row_cut_threshold=row_res["cut_threshold"],
        col_cut_threshold=col_res["cut_threshold"],
        _row_meta=row_res["_params"],
        _col_meta=col_res["_params"],
        row_tol_labels=row_res["alt_labels"],
        col_tol_labels=col_res["alt_labels"],
        row_tol_k=row_res["alt_k"],
        col_tol_k=col_res["alt_k"],
        row_cut_tol_threshold=row_res["alt_cut_threshold"],
        col_cut_tol_threshold=col_res["alt_cut_threshold"],
    )

    
# ---------------------------------------------------------------------
# Group clustering (taken some grouping prior and 
# re-ordering/grouping based on this information)
# ---------------------------------------------------------------------

def _pca1_order(X):
    """
    """
    # X: (n_items, n_features)
    Xc = X - X.mean(axis=0, keepdims=True)
    # first PC score via SVD (stable)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    score = U[:, 0] * S[0]
    return np.argsort(score)

def _within_block_leaf_order(
    X,                 
    metric="euclidean",
    method="ward",
    max_items=2000,
    fallback="pca1",       
): 
    n = X.shape[0]
    if n <= 2:
        return np.arange(n)

    # Ward is only coherent with Euclidean
    if method == "ward" and metric != "euclidean":
        metric = "euclidean"

    if n > max_items:
        return _pca1_order(X) if fallback == "pca1" else np.arange(n)

    d = pdist(X, metric=metric)
    Z = linkage(d, method=method)
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    return np.asarray(leaves, dtype=int)

def _cut_distance_for_k(Z: np.ndarray, k: int) -> float:
    """
    Given a SciPy linkage matrix Z (n-1 merges; Z[:,2] are merge heights)
    return a distance threshold t such that fcluster(Z, t, criterion='distance')
    yields exactly k clusters (for standard monotone linkages like 'ward').

    We choose t midway between the (i-1)-th and i-th merge heights, where
    i = n - k (number of merges included). Handle edges robustly.
    """
    n = Z.shape[0] + 1              # number of original observations
    d = Z[:, 2]                     # non-decreasing merge heights
    if n <= 1:
        return 0.0

    i = n - k                       # merges included to reach k clusters
    if i <= 0:
        # No merges: want n clusters → pick any t < first height
        return (d[0] * 0.5) if d.size > 0 else 0.0
    elif i >= n - 1:
        # All merges: want 1 cluster → pick t just above max height
        return (d[-1] + np.finfo(float).eps) if d.size > 0 else 0.0
    else:
        # Midpoint between previous and next merge heights
        return 0.5 * (d[i - 1] + d[i])

def _build_groups(
    n_items: int,
    blocks: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Ensure a full, disjoint partition of {0,…,n_items-1}.
    """
    item_to_group = -np.ones(n_items, dtype=int)
    groups: List[List[int]] = []

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

def _hierarchical_clustering_forgroup(
    data: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
    metric: str = "euclidean",
    *,
    # NEW tolerance knobs (default keeps old behavior but prefers simpler k)
    select_min_k_within_tol: bool = True,
    silhouette_tol: float = 0.02,
    tol_mode: str = "relative",  # {"relative","absolute"}
) -> Dict[str, Any]:
    """
    Ward hierarchical clustering on `data` (observations × features)
    with tolerance-based k selection and an alternative k near the chosen k.
    """
    n_obs = data.shape[0]
    if n_obs < 2:
        return dict(
            linkage=None,
            leaf_order=list(range(n_obs)),
            labels=np.ones(n_obs, dtype=int),
            k=1,
            silhouette=np.nan,
            score_recording={},
            cophenet_score=np.nan,
            cut_distance_by_k={},
            best_cut_distance=None,
            labels_by_k={},
            # NEW fields for consistency
            strict_best_k=1,
            alt_k=1,
            alt_labels=np.ones(n_obs, dtype=int),
            alt_cut_distance=None,
            primary_candidates=[1],
        )

    pairwise = pdist(data, metric=metric)
    Z = linkage(pairwise, method="ward")
    c, _ = cophenet(Z, pdist(data))  # cophenetic correlation

    D_sq = squareform(pairwise)
    k_range = range(max(k_min, 2), min(k_max, n_obs - 1) + 1)
    # print(f"k_range: {k_range}")

    score_recording: Dict[int, float] = {}
    cut_distance_by_k: Dict[int, float] = {}
    labels_by_k: Dict[int, np.ndarray] = {}

    # Precompute cut distances
    for k in k_range:
        cut_distance_by_k[k] = _cut_distance_for_k(Z, k)

    # Silhouette per k
    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        labels_by_k[k] = labels
        score = silhouette_score(D_sq, labels, metric="precomputed")
        score_recording[k] = float(score)

    # Strict argmax (global best)
    strict_best_k = max(score_recording, key=score_recording.get)
    strict_best_score = score_recording[strict_best_k]

    # Tolerance threshold for primary selection (relative to strict best)
    if select_min_k_within_tol:
        if tol_mode == "relative":
            primary_thresh = strict_best_score * (1.0 - float(silhouette_tol))
        elif tol_mode == "absolute":
            primary_thresh = strict_best_score - float(silhouette_tol)
        else:
            raise ValueError("tol_mode must be 'relative' or 'absolute'.")
        primary_candidates = [k for k in k_range if score_recording[k] >= primary_thresh]
        best_k = min(primary_candidates) if primary_candidates else strict_best_k
    else:
        best_k = strict_best_k
        primary_candidates = [strict_best_k]

    best_labels = labels_by_k[best_k]
    best_score = score_recording[best_k]
    best_cut_distance = cut_distance_by_k.get(best_k, _cut_distance_for_k(Z, best_k))

    # Secondary tolerance (alt_k) relative to chosen best_k (smallest k within tol of best_k)
    if tol_mode == "relative":
        alt_thresh = best_score * (1.0 - float(silhouette_tol))
    else:
        alt_thresh = best_score - float(silhouette_tol)
    alt_candidates = [k for k in k_range if score_recording[k] >= alt_thresh]
    alt_k = min(alt_candidates) if alt_candidates else best_k
    alt_labels = labels_by_k[alt_k]
    alt_cut_distance = cut_distance_by_k.get(alt_k, _cut_distance_for_k(Z, alt_k))

    leaf_order = dendrogram(Z, no_plot=True)["leaves"]

    return dict(
        linkage=Z,
        leaf_order=leaf_order,
        labels=best_labels,
        k=best_k,
        silhouette=best_score,                    # silhouette at chosen k
        score_recording=score_recording,
        cophenet_score=c,
        cut_distance_by_k=cut_distance_by_k,
        best_cut_distance=best_cut_distance,
        labels_by_k=labels_by_k,
        # NEW: introspection + alternative near chosen k
        strict_best_k=strict_best_k,
        strict_best_score=strict_best_score,
        primary_candidates=primary_candidates,
        alt_k=alt_k,
        alt_labels=alt_labels,
        alt_cut_distance=alt_cut_distance,
    )


def cluster_variance_matrix_forgroup(
    V: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
    row_groups: Optional[Sequence[Sequence[int]]] = None,
    col_groups_all_lst: Optional[List[Sequence[Sequence[int]]]] = None,
    *,
    select_min_k_within_tol: bool = True,
    silhouette_tol: float = 0.02,
    tol_mode: str = "relative",
) -> Dict[str, Any]:
    """
    Bi-directional hierarchical clustering of a variance matrix V
    (shape: N_features * M_neurons) with tolerance-based k selection.

    col_groups_all_lst : list of col_groups, each from a different trial/seed.
        Each element is a grouping of column indices (same format as the old
        col_groups argument).  All groupings are used to compute per-k
        silhouette scores; the scores are averaged across groupings to
        determine the best column k collectively.  The first element is used
        as the primary grouping for the final linkage, leaf ordering, and
        label assignment.
    """
    V = np.asarray(V)

    # ----- rows (unchanged) -----
    row_blocks, row_map = _build_groups(V.shape[0], row_groups)
    V_row_grp = _aggregate_along_axis(V, row_blocks, axis=0, reduce="mean")
    row_res = _hierarchical_clustering_forgroup(
        V_row_grp, k_min, k_max,
        select_min_k_within_tol=select_min_k_within_tol,
        silhouette_tol=silhouette_tol,
        tol_mode=tol_mode,
    )

    # ----- cols: aggregate silhouette across all provided groupings -----
    if col_groups_all_lst is None or len(col_groups_all_lst) == 0:
        col_groups_all_lst = [None]

    # Run clustering for every col_groups; collect per-k silhouette curves.
    col_res_list: List[Dict[str, Any]] = []
    col_blocks_list: List[List[List[int]]] = []
    col_V_grp_list: List[np.ndarray] = []

    for col_groups_i in col_groups_all_lst:
        col_blocks_i, _ = _build_groups(V.shape[1], col_groups_i)
        V_col_grp_i = _aggregate_along_axis(V, col_blocks_i, axis=1, reduce="mean")
        col_res_i = _hierarchical_clustering_forgroup(
            V_col_grp_i.T, k_min, k_max,
            select_min_k_within_tol=select_min_k_within_tol,
            silhouette_tol=silhouette_tol,
            tol_mode=tol_mode,
        )
        col_res_list.append(col_res_i)
        col_blocks_list.append(col_blocks_i)
        col_V_grp_list.append(V_col_grp_i)

    # Average silhouette scores across groupings for each k.
    all_k_col: set = set()
    for col_res_i in col_res_list:
        all_k_col.update(col_res_i["score_recording"].keys())

    col_score_mean: Dict[int, float] = {}
    col_score_std: Dict[int, float] = {}
    for k in sorted(all_k_col):
        vals = [col_res_i["score_recording"][k]
                for col_res_i in col_res_list
                if k in col_res_i["score_recording"]]
        col_score_mean[k] = float(np.mean(vals))
        col_score_std[k] = float(np.std(vals))

    # Re-select best col k from the averaged silhouette curve.
    col_strict_best_k = max(col_score_mean, key=col_score_mean.get)
    col_strict_best_score = col_score_mean[col_strict_best_k]

    if select_min_k_within_tol:
        if tol_mode == "relative":
            col_primary_thresh = col_strict_best_score * (1.0 - float(silhouette_tol))
        elif tol_mode == "absolute":
            col_primary_thresh = col_strict_best_score - float(silhouette_tol)
        else:
            raise ValueError("tol_mode must be 'relative' or 'absolute'.")
        col_primary_candidates = [k for k in col_score_mean
                                   if col_score_mean[k] >= col_primary_thresh]
        best_k_col = min(col_primary_candidates) if col_primary_candidates else col_strict_best_k
    else:
        best_k_col = col_strict_best_k
        col_primary_candidates = [col_strict_best_k]

    # Primary grouping (first element) provides linkage, ordering, and labels.
    col_res = col_res_list[0]
    col_blocks = col_blocks_list[0]
    V_col_grp = col_V_grp_list[0]
    col_blocks_primary, col_map = _build_groups(V.shape[1], col_groups_all_lst[0])

    # Labels / cut distance at the collectively determined best k.
    if best_k_col in col_res["labels_by_k"]:
        col_group_labels_at_best_k = col_res["labels_by_k"][best_k_col]
        col_best_cut_distance = col_res["cut_distance_by_k"].get(
            best_k_col, _cut_distance_for_k(col_res["linkage"], best_k_col)
        )
    else:
        col_group_labels_at_best_k = col_res["labels"]
        col_best_cut_distance = col_res["best_cut_distance"]

    # ----- within-block ordering -----
    row_order: List[int] = []
    for g in row_res["leaf_order"]:
        idxs = np.asarray(row_blocks[g], dtype=int)
        if idxs.size <= 2:
            row_order.extend(idxs.tolist())
            continue
        X = V_col_grp[idxs, :]          # (n_in_block, C_blk) — primary col grouping
        local = _within_block_leaf_order(X, metric="euclidean", method="ward",
                                         max_items=2000, fallback="pca1")
        row_order.extend(idxs[local].tolist())

    col_order: List[int] = []
    for g in col_res["leaf_order"]:
        idxs = np.asarray(col_blocks[g], dtype=int)
        if idxs.size <= 2:
            col_order.extend(idxs.tolist())
            continue
        X = V_row_grp[:, idxs].T        # (n_in_block, R_blk)
        local = _within_block_leaf_order(X, metric="euclidean", method="ward",
                                         max_items=2000, fallback="pca1")
        col_order.extend(idxs[local].tolist())

    row_labels = np.take(row_res["labels"], row_map)
    row_labels_by_k = {k: np.take(lbls, row_map) for k, lbls in row_res["labels_by_k"].items()}

    col_labels = np.take(col_group_labels_at_best_k, col_map)
    col_labels_by_k = {k: np.take(lbls, col_map) for k, lbls in col_res["labels_by_k"].items()}

    return dict(
        # fine-grained ordering / labels
        row_order=row_order,
        col_order=col_order,
        row_labels=row_labels,
        col_labels=col_labels,
        # block-level results (primary grouping for cols)
        row_group_order=row_res["leaf_order"],
        col_group_order=col_res["leaf_order"],
        row_group_labels=row_res["labels"],
        col_group_labels=col_group_labels_at_best_k,
        row_k=row_res["k"],
        col_k=best_k_col,
        row_linkage=row_res["linkage"],
        col_linkage=col_res["linkage"],
        row_score_recording=row_res["score_recording"],
        # col silhouette: mean/std across all groupings
        col_score_recording_mean=col_score_mean,
        col_score_recording_std=col_score_std,
        col_score_recording_per_grouping=[r["score_recording"] for r in col_res_list],
        row_cophenet_score=row_res["cophenet_score"],
        col_cophenet_score=col_res["cophenet_score"],
        row_cut_distance_by_k=row_res["cut_distance_by_k"],
        col_cut_distance_by_k=col_res["cut_distance_by_k"],
        row_best_cut_distance=row_res["best_cut_distance"],
        col_best_cut_distance=col_best_cut_distance,
        row_labels_by_k=row_labels_by_k,
        col_labels_by_k=col_labels_by_k,
        # introspection + strict best k
        row_strict_best_k=row_res["strict_best_k"],
        col_strict_best_k=col_strict_best_k,
        row_alt_k=row_res["alt_k"],
        col_alt_k=col_res["alt_k"],
        row_alt_cut_distance=row_res["alt_cut_distance"],
        col_alt_cut_distance=col_res["alt_cut_distance"],
        row_primary_candidates=row_res["primary_candidates"],
        col_primary_candidates=col_primary_candidates,
    )

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def make_col_groups_with_kmeans(V, 
                                n_groups=1000, 
                                batch_size=8192, 
                                random_state=0,
                                verbose=False):
    """
    Use MiniBatchKMeans to cluster columns (neurons) of V into groups.
    This produces a 'col_groups' list compatible with cluster_variance_matrix_forgroup.
    """
    n_cols = V.shape[1]
    
    if verbose:
        print(f"Running MiniBatchKMeans on {n_cols} columns ...")
    
    mbk = MiniBatchKMeans(
        n_clusters=n_groups,
        batch_size=batch_size,
        n_init='auto',
        random_state=random_state
    )
    
    col_labels = mbk.fit_predict(V.T)  
    centroids = mbk.cluster_centers_.T  
    col_groups = [np.where(col_labels == g)[0].tolist() for g in range(n_groups)]
    col_groups = [g for g in col_groups if len(g) > 0]
    
    if verbose:
        print(f"Formed {len(col_groups)} groups.")
        
    assert len(col_groups) <= n_groups, "Unexpected: more groups than requested."
    
    return col_groups, col_labels, centroids
    