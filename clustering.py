import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cophenet, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import List, Sequence, Optional, Tuple, Dict, Any
from matplotlib.patches import Rectangle
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt
import seaborn as sns 
import scienceplots
import gc, psutil, os
plt.style.use('science')
plt.style.use(['no-latex'])

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
# ---------------------------------------------------------------------

def _breaks(lbls):
    """
    """
    idx = np.nonzero(np.diff(lbls))[0] + 1
    return idx.tolist()

def show_A_ordered_by_B(A, res_B, row_name):
    """
    """
    A_ord = A[np.ix_(res_B["row_order"], res_B["col_order"])]
    fig, ax = plt.subplots(1,1,figsize=(24,8))
    hm = sns.heatmap(A_ord, ax=ax, cmap="coolwarm", cbar=True, vmin=0, vmax=1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    rl = np.asarray(res_B["row_labels"])[res_B["row_order"]]
    cl = np.asarray(res_B["col_labels"])[res_B["col_order"]]
    
    rbreaks = _breaks(rl)
    cbreaks = _breaks(cl)
    for rb in rbreaks:
        ax.axhline(rb, color="k", lw=0.6)
    for cb in cbreaks:
        ax.axvline(cb, color="k", lw=0.6)
    ax.set_title("A reordered by B clustering", fontsize=20)
    # set new row name
    ax.set_yticks(np.arange(len(row_name)))
    ax.set_yticklabels(row_name[res_B["row_order"]], rotation=0, ha='right', va='center', fontsize=9)
    plt.tight_layout()
    return fig, ax 

def transfer_and_score(A, B, res_A, res_B, metric="euclidean"):
    """
    Evaluate how well B’s clustering explains A.
    Assumes A and B share the same row/col entities in the same order.
    If not, align by your IDs first, then pass aligned matrices.
    """
    # B→A labels
    lblB_row = np.asarray(res_B["row_labels"])
    lblB_col = np.asarray(res_B["col_labels"])

    # A’s own labels (for partition agreement only)
    lblA_row = np.asarray(res_A["row_labels"])
    lblA_col = np.asarray(res_A["col_labels"])

    # Pairwise distances in A for silhouette
    D_row = squareform(pdist(A, metric=metric))       # rows as observations
    D_col = squareform(pdist(A.T, metric=metric))     # cols as observations

    # --- Cohesion/separation on A with B’s labels
    row_sil = float(silhouette_score(D_row, lblB_row, metric="precomputed"))
    col_sil = float(silhouette_score(D_col, lblB_col, metric="precomputed"))
    row_ch  = float(calinski_harabasz_score(A,  lblB_row))
    col_ch  = float(calinski_harabasz_score(A.T, lblB_col))
    row_db  = float(davies_bouldin_score(A,  lblB_row))
    col_db  = float(davies_bouldin_score(A.T, lblB_col))

    # --- Variance explained in A by B’s labels
    row_R2 = _cluster_R2(A,  lblB_row)
    col_R2 = _cluster_R2(A.T, lblB_col)

    # --- Agreement of partitions: A’s own labels vs B’s (not a performance metric per se, but useful)
    row_ari = float(adjusted_rand_score(lblA_row, lblB_row))
    col_ari = float(adjusted_rand_score(lblA_col, lblB_col))
    row_nmi = float(normalized_mutual_info_score(lblA_row, lblB_row))
    col_nmi = float(normalized_mutual_info_score(lblA_col, lblB_col))

    ccc_row = _cross_cophenetic_corr(res_B["row_linkage"], A, metric=metric)
    ccc_col = _cross_cophenetic_corr(res_B["col_linkage"], A.T, metric=metric)

    # --- 2D block R^2 using both row+col labels from B
    block_R2 = _block_R2(A, lblB_row, lblB_col)

    return dict(
        # A’s structure under B’s partition
        row_silhouette_on_A=row_sil,
        col_silhouette_on_A=col_sil,
        row_CH_on_A=row_ch, col_CH_on_A=col_ch,
        row_DB_on_A=row_db, col_DB_on_A=col_db,
        row_R2_on_A=row_R2, col_R2_on_A=col_R2,
        block_R2_on_A=block_R2,
        # Partition agreement (A’s own clustering vs B’s)
        row_ARI_A_vs_B=row_ari, col_ARI_A_vs_B=col_ari,
        row_NMI_A_vs_B=row_nmi, col_NMI_A_vs_B=col_nmi,
        # Geometry agreement
        row_cophenetic_corr_Btree_vs_A=ccc_row,
        col_cophenetic_corr_Btree_vs_A=ccc_col,
        # Useful counts
        row_k_B=int(res_B["row_k"]),
        col_k_B=int(res_B["col_k"])
    )

def _cross_cophenetic_corr(Z_ref, X, metric="euclidean"):
    """
    Correlation between cophenetic distances implied by Z_ref
    and the actual pairwise distances in X (same entities/order).
    """
    # Optional sanity check: (n_obs - 1) rows in linkage
    n_ref = Z_ref.shape[0] + 1
    n_X = X.shape[0]
    if n_ref != n_X:
        raise ValueError(f"Size mismatch: linkage has {n_ref} leaves but X has {n_X} rows.")
    c, _ = cophenet(Z_ref, pdist(X, metric=metric))
    return float(c)

def _cluster_R2(X, labels):
    """
    Univariate-free ‘k-means ANOVA’ style R^2 across multi-d features.
    """
    X = np.asarray(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    TSS = np.sum(Xc**2)
    WSS = 0.0
    for g in np.unique(labels):
        Xg = X[labels == g]
        Xg_c = Xg - Xg.mean(axis=0, keepdims=True)
        WSS += np.sum(Xg_c**2)
    return float(1.0 - WSS / (TSS + 1e-12))

def _block_R2(M, row_labels, col_labels):
    """
    2D ‘ANOVA’ style R^2: how much of matrix energy is explained by row×col blocks.
    Equivalent to replacing each block by its mean and computing 1 - SSE/TSS.
    """
    M = np.asarray(M)
    M_centered = M - M.mean()
    TSS = float(np.sum(M_centered**2))

    # Build block means
    row_u = np.unique(row_labels)
    col_u = np.unique(col_labels)
    M_hat = np.zeros_like(M, dtype=float)
    for r in row_u:
        rmask = (row_labels == r)
        for c in col_u:
            cmask = (col_labels == c)
            block = M[np.ix_(rmask, cmask)]
            M_hat[np.ix_(rmask, cmask)] = block.mean() if block.size else 0.0

    SSE = float(np.sum((M - M_hat)**2))
    return float(1.0 - SSE / (TSS + 1e-12))

# ---------------------------------------------------------------------
# Clustering with multiple repeats
# ---------------------------------------------------------------------

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

    Mirrors `_hierarchical_clustering`:
      - Ward shortcut (linkage on data, euclidean only) vs generic metric
      - Optimal leaf ordering
      - Silhouette on precomputed distances
      - Cut thresholds computed from Z rows

    Repeats:
      - Optional feature resampling and Gaussian jitter
      - Aggregate silhouette mean/std across repeats for each k
      - Choose k with best mean silhouette
      - Pick labeling from the repeat that maximizes mean ARI to other repeats at that k
      - Return that repeat's linkage, leaf order, labels, and cut threshold
    """
    rng = np.random.default_rng(random_state)
    n_obs, n_feat = data.shape

    # Resolve k-range as in the single-run version
    k_min = max(k_min, 2)
    k_max = min(k_max, n_obs - 1)
    if k_max < k_min:
        raise ValueError("Not enough observations for the requested k-range.")

    k_range = range(k_min, k_max + 1)
    per_repeat = []

    for r in range(n_repeats):
        if resample_features_frac < 1.0:
            m = max(1, int(np.ceil(resample_features_frac * n_feat)))
            feat_idx = rng.choice(n_feat, size=m, replace=True)
            Xr = data[:, feat_idx]
        else:
            Xr = data

        if jitter_std > 0.0:
            Xr = Xr + rng.normal(0.0, jitter_std, size=Xr.shape)

        if method.lower() == "ward":
            Z = linkage(Xr, method="ward", metric="euclidean")
            pairwise_dists = pdist(Xr, metric="euclidean")
        else:
            pairwise_dists = pdist(Xr, metric=metric)
            Z = linkage(pairwise_dists, method=method)

        # Optimal leaf ordering
        Z = optimal_leaf_ordering(Z, pairwise_dists)

        D_square = squareform(pairwise_dists)

        labels_by_k = {}
        scores_by_k = {}
        cut_thresholds = {}

        for k in k_range:
            # labels = fcluster(Z, k, criterion="maxclust")
            labels = labels_at_k(Z, k)

            score = silhouette_score(D_square, labels, metric="precomputed")
            labels_by_k[k] = labels
            scores_by_k[k] = float(score)

            # cut threshold extraction aligned with single-run version
            cut_idx = (n_obs - k) - 1
            if 0 <= cut_idx < Z.shape[0]:
                cut_thresholds[k] = float(np.nextafter(Z[cut_idx, 2], np.inf))
            else:
                cut_thresholds[k] = None

        leaf_order = dendrogram(Z, no_plot=True)["leaves"]
        per_repeat.append(dict(
            Z=Z,
            leaf_order=leaf_order,
            labels_by_k=labels_by_k,
            scores_by_k=scores_by_k,
            cut_thresholds=cut_thresholds
        ))

    score_recording_mean = {k: float(np.mean([rep["scores_by_k"][k] for rep in per_repeat])) for k in k_range}
    score_recording_std  = {k: float(np.std( [rep["scores_by_k"][k] for rep in per_repeat])) for k in k_range}

    best_k = max(score_recording_mean, key=score_recording_mean.get)
    best_score_mean = score_recording_mean[best_k]

    labels_list = [rep["labels_by_k"][best_k] for rep in per_repeat]

    if n_repeats == 1:
        best_rep_idx = 0
        mean_ari_val = 1.0
    else:
        R = len(labels_list)
        ari_mat = np.zeros((R, R))
        for i in range(R):
            for j in range(i + 1, R):
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                ari_mat[i, j] = ari_mat[j, i] = ari
        mean_ari = ari_mat.mean(axis=1)  # mean ARI to others (zeros on diagonal)
        best_rep_idx = int(np.argmax(mean_ari))
        mean_ari_val = float(mean_ari[best_rep_idx])

    best_labels = labels_list[best_rep_idx]
    Z_best      = per_repeat[best_rep_idx]["Z"]
    leaf_order  = per_repeat[best_rep_idx]["leaf_order"]
    best_cut_threshold = per_repeat[best_rep_idx]["cut_thresholds"][best_k]

    return dict(
        linkage=Z_best,
        leaf_order=leaf_order,
        labels=best_labels,
        k=best_k,
        silhouette=best_score_mean,
        score_recording_mean=score_recording_mean,
        score_recording_std=score_recording_std,
        cut_threshold=best_cut_threshold,
        _repeats=len(per_repeat),
        _params=dict(
            resample_features_frac=resample_features_frac,
            jitter_std=jitter_std,
            method=method,
            metric=metric,
            mean_ari_at_best_k=mean_ari_val
        ),
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
    Cluster a variance matrix V (shape: N_features × M_neurons)
    for both rows (features) and columns (neurons), with repeat-based stabilization.

    Returns mean/std silhouette curves for rows and cols, chosen k per axis,
    and the cut thresholds (from the chosen repeat) for convenience.
    """
    V = np.asarray(V)

    row_res = _hierarchical_clustering_repeat(
        V, k_min, k_max, metric, method,
        n_repeats=n_repeats,
        resample_features_frac=resample_features_frac,
        jitter_std=jitter_std,
        random_state=random_state
    )

    col_res = _hierarchical_clustering_repeat(
        V.T, k_min, k_max, metric, method,
        n_repeats=n_repeats,
        resample_features_frac=resample_features_frac,
        jitter_std=jitter_std,
        random_state=None if random_state is None else (random_state + 1)
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
        col_score_recording_std=col_res["score_recording_std"],
        row_cut_threshold=row_res["cut_threshold"],
        col_cut_threshold=col_res["cut_threshold"],
        _row_meta=row_res["_params"],
        _col_meta=col_res["_params"],
    )
    
# ---------------------------------------------------------------------
# Group clustering (taken some grouping prior and 
# re-ordering/grouping based on this information)
# ---------------------------------------------------------------------

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

def _hierarchical_clustering_forgroup(
    data: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    Run Ward hierarchical clustering on `data` (observations × features).
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
    # Z = optimal_leaf_ordering(linkage(pairwise, method="ward"), pairwise)
    Z = linkage(pairwise, method="ward")

    c, _ = cophenet(Z, pdist(data))

    best_k, best_score, best_labels = None, -np.inf, None
    D_sq = squareform(pairwise)
    k_range = range(max(k_min, 2), min(k_max, n_obs - 1) + 1)

    score_recording = {}
    cut_distance_by_k = {}
    labels_by_k: Dict[int, np.ndarray] = {}

    for k in k_range:
        cut_distance_by_k[k] = _cut_distance_for_k(Z, k)
        
    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        labels_by_k[k] = labels
        score = silhouette_score(D_sq, labels, metric="precomputed")
        score_recording[k] = score # register
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    leaf_order = dendrogram(Z, no_plot=True)["leaves"]
    best_cut_distance = cut_distance_by_k.get(best_k, _cut_distance_for_k(Z, best_k))

    return dict(
        linkage=Z,
        leaf_order=leaf_order,
        labels=best_labels,
        k=best_k,
        silhouette=best_score,
        score_recording=score_recording,
        cophenet_score=c,
        cut_distance_by_k=cut_distance_by_k,
        best_cut_distance=best_cut_distance,
        labels_by_k=labels_by_k
    )

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
    """
    V = np.asarray(V)

    row_blocks, row_map = _build_groups(V.shape[0], row_groups)
    V_row_grp = _aggregate_along_axis(V, row_blocks, axis=0, reduce="mean")
    row_res = _hierarchical_clustering_forgroup(V_row_grp, k_min, k_max)
    
    row_labels = np.take(row_res["labels"], row_map)
    row_labels_by_k = {
        k: np.take(lbls, row_map) for k, lbls in row_res["labels_by_k"].items()
    }
    row_order = [idx for g in row_res["leaf_order"] for idx in row_blocks[g]]

    col_blocks, col_map = _build_groups(V.shape[1], col_groups)
    V_col_grp = _aggregate_along_axis(V, col_blocks, axis=1, reduce="mean")
    col_res = _hierarchical_clustering_forgroup(V_col_grp.T, k_min, k_max)

    col_labels = np.take(col_res["labels"], col_map)
    col_labels_by_k = {
        k: np.take(lbls, col_map) for k, lbls in col_res["labels_by_k"].items()
    }
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
        col_score_recording=col_res["score_recording"],
        row_cophenet_score=row_res["cophenet_score"],
        col_cophenet_score=col_res["cophenet_score"],
        row_cut_distance_by_k=row_res["cut_distance_by_k"],
        col_cut_distance_by_k=col_res["cut_distance_by_k"],
        row_best_cut_distance=row_res["best_cut_distance"],
        col_best_cut_distance=col_res["best_cut_distance"],
        row_labels_by_k=row_labels_by_k,
        col_labels_by_k=col_labels_by_k
    )

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def make_col_groups_with_kmeans(V: np.ndarray, n_groups: int = 1000, \
                                batch_size: int = 8192, \
                                random_state: int = 0):
    """
    Use MiniBatchKMeans to cluster columns (neurons) of V into groups.
    This produces a 'col_groups' list compatible with cluster_variance_matrix_forgroup.
    """
    n_cols = V.shape[1]
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
    
    print(f"Formed {len(col_groups)} groups.")
    return col_groups, col_labels, centroids
    