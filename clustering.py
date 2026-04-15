import warnings
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
    score_quantile=None,
    unresponsive_norm_frac=1e-3,
    skip_unresponsive_detection=False,
    tol_k_select="min",
):
    """
    Hierarchical clustering with stability repeats.

    Key behavior:
        - Repeats (feature bootstrap / jitter) are used ONLY to estimate the
          silhouette-vs-k curve.
        - The returned linkage / leaf_order are computed ONCE on the original
          unperturbed data.
        - The primary returned labels / k correspond to the strict argmax of
          the aggregated silhouette across repeats.
        - The alt/tol returned labels / k correspond to the smallest (or
          largest, depending on tol_k_select) k whose aggregated silhouette
          is within tolerance of that strict best score.
        - Aggregation is mean by default; set score_quantile (e.g. 0.75, 0.90)
          to use that quantile across repeats instead.
        - We still pick a representative repeat (max mean ARI at the selected
          tolerant k) for diagnostics, but we do NOT return its linkage as the
          final dendrogram.
        - jitter_std is a RELATIVE fraction: each entry is multiplied by
          (1 + N(0, jitter_std)), so jitter_std=0.01 means ±1% per-entry noise.
          This is scale-invariant and behaves consistently across normalized and
          unnormalized matrices. Zero entries receive no jitter.
        - tol_k_select controls whether the smallest ("min") or largest
          ("max") k within the tolerance band is selected.
    """
    if tol_k_select not in ("min", "max"):
        raise ValueError("tol_k_select must be 'min' or 'max'.")
    _k_select = min if tol_k_select == "min" else max

    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n_obs, n_feat = data.shape

    # ------------------------------------------------------------------
    # Unresponsive row detection (active for all metrics).
    #
    # Rows whose L2 norm is < unresponsive_norm_frac * max_norm are
    # considered "unresponsive" (silent neuron or silent task period).
    # For cosine/correlation metrics this is essential because the
    # distance is undefined for zero vectors. For Euclidean/Ward it is
    # also beneficial: near-zero rows form a trivial cluster that can
    # inflate silhouette scores and obscure meaningful structure among
    # the active rows. Excluding them and assigning a dedicated label
    # (k+1) keeps the clustering focused on informative observations.
    # Using a relative threshold (fraction of the max norm) keeps the
    # criterion scale-invariant across normalised and unnormalised data.
    # ------------------------------------------------------------------
    if skip_unresponsive_detection:
        unresponsive_mask = np.zeros(n_obs, dtype=bool)
    else:
        norms = np.linalg.norm(data, axis=1)
        max_norm = norms.max() if norms.max() > 0 else 1.0
        unresponsive_mask = norms < unresponsive_norm_frac * max_norm

    n_unresponsive = int(unresponsive_mask.sum())
    if n_unresponsive > 0:
        warnings.warn(
            f"{n_unresponsive} unresponsive row(s) detected "
            f"(norm < {unresponsive_norm_frac} * max_norm = "
            f"{unresponsive_norm_frac * max_norm:.3g}); "
            f"excluding from clustering and assigning to dedicated cluster (label = k+1).",
            RuntimeWarning,
            stacklevel=3,
        )

    active_mask = ~unresponsive_mask
    active_idx  = np.where(active_mask)[0]        # original row indices of active rows
    unres_idx   = np.where(unresponsive_mask)[0]  # original row indices of unresponsive rows
    active_data = data[active_mask]               # shape: (n_active, n_feat)
    n_active    = active_data.shape[0]

    def _expand_labels(labels_active, k_active):
        """Map labels from n_active rows back to n_obs; unresponsive rows get label k+1."""
        full = np.zeros(n_obs, dtype=int)
        full[active_mask] = labels_active
        if n_unresponsive > 0:
            full[unresponsive_mask] = k_active + 1
        return full

    def _expand_leaf_order(leaf_order_active):
        """Map leaf order from active-row indices back to original indices.

        Unresponsive rows are appended at the end so they form a contiguous
        block in downstream heatmaps.
        """
        return list(active_idx[np.array(leaf_order_active)]) + list(unres_idx)

    def _compute_linkage_leaforder(X):
        """Compute linkage + optimal leaf order; return (Z, condensed_dists, leaf_order).

        X must already be free of unresponsive rows — handled by the caller.
        leaf_order indices are relative to the rows of X (0..len(X)-1).
        """
        X = np.asarray(X, dtype=float)

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
        cut_idx = (n_active - k) - 1
        if 0 <= cut_idx < Z.shape[0]:
            return float(np.nextafter(Z[cut_idx, 2], np.inf))
        return None

    k_min = max(int(k_min), 2)
    k_max = min(int(k_max), n_active - 1)
    if k_max < k_min:
        raise ValueError("Not enough observations for the requested k-range.")
    k_range = range(k_min, k_max + 1)
    k_values = np.array(list(k_range), dtype=int)

    per_repeat = []
    for _ in range(int(n_repeats)):
        if resample_features_frac < 1.0:
            m = max(1, int(np.ceil(resample_features_frac * n_feat)))
            feat_idx = rng.choice(n_feat, size=m, replace=True)
            Xr = active_data[:, feat_idx]
        else:
            Xr = active_data

        if jitter_std > 0.0:
            # Relative jitter: each entry is perturbed by jitter_std fraction of
            # its own magnitude, i.e. Xr_ij *= (1 + N(0, jitter_std)).
            # Zero entries receive no jitter, which is intentional.
            Xr = Xr * (1.0 + rng.normal(0.0, jitter_std, size=Xr.shape))

        Zr, dr, leaf_order_r = _compute_linkage_leaforder(Xr)
        D_square = squareform(dr)

        labels_by_k = {}
        scores_by_k = {}
        cut_thresholds = {}
        for k in k_range:
            labels = fcluster(Zr, k, criterion="maxclust")
            n_clusters = np.unique(labels).size

            if n_clusters < 2 or n_clusters >= n_active:
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

    if score_quantile is not None:
        score_recording_agg = {
            int(k): float(np.quantile(score_recording_all[:, idx], score_quantile))
            for idx, k in enumerate(k_values)
        }
    else:
        score_recording_agg = score_recording_mean

    strict_best_k = max(score_recording_agg, key=score_recording_agg.get)
    strict_best_score = score_recording_agg[strict_best_k]

    if select_min_k_within_tol:
        primary_thresh = _score_threshold_from_best(
            strict_best_score, silhouette_tol=silhouette_tol, tol_mode=tol_mode
        )
        primary_candidates = [k for k in k_range if score_recording_agg[k] >= primary_thresh]
        best_k = _k_select(primary_candidates) if primary_candidates else strict_best_k
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
    alt_candidates = [k for k in k_range if score_recording_agg[k] >= alt_thresh]
    alt_k = _k_select(alt_candidates) if alt_candidates else strict_best_k

    # Final linkage and labels on the original unperturbed active rows.
    Z0, d0, leaf0 = _compute_linkage_leaforder(active_data)
    leaf0_full = _expand_leaf_order(leaf0)

    strict_labels     = _expand_labels(fcluster(Z0, strict_best_k, criterion="maxclust"), strict_best_k)
    strict_cut_threshold = _cut_threshold_from_Z(Z0, strict_best_k)

    final_alt_labels  = _expand_labels(fcluster(Z0, alt_k, criterion="maxclust"), alt_k)
    final_alt_cut_threshold = _cut_threshold_from_Z(Z0, alt_k)

    return dict(
        linkage=Z0,
        leaf_order=leaf0_full,
        labels=strict_labels,
        k=strict_best_k,
        cut_threshold=strict_cut_threshold,

        alt_k=alt_k,
        alt_labels=final_alt_labels,
        alt_linkage=Z0,
        alt_leaf_order=leaf0_full,
        alt_cut_threshold=final_alt_cut_threshold,

        silhouette_strict_best=strict_best_score,
        silhouette_at_k=best_k_score_mean,
        score_recording_mean=score_recording_mean,
        score_recording_std=score_recording_std,
        score_recording_all=score_recording_all,
        k_values=k_values,

        rep_linkage=best_rep["Z"],
        rep_leaf_order=_expand_leaf_order(best_rep["leaf_order"]),
        rep_labels_at_k=_expand_labels(best_rep["labels_by_k"][best_k], best_k),
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
            score_quantile=score_quantile,
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
    row_metric=None, row_method=None,
    col_metric=None, col_method=None,
    *,
    n_repeats=10,
    resample_features_frac=1.0,
    jitter_std=0.0,
    random_state=None,
    select_min_k_within_tol=True,
    silhouette_tol=0.02,
    tol_mode="relative",
    score_quantile=None,
    skip_unresponsive_detection=False,
    tol_k_select="min",
):
    """
    Cluster a variance matrix V (shape: N_features * M_neurons)
    for both rows (features) and columns (neurons), with repeat-based stabilization.

    Returns mean/std silhouette curves for rows and cols.
    row_k / col_k correspond to the strict argmax of the aggregated silhouette.
    row_tol_* / col_tol_* correspond to the smaller tolerance-selected solution.
    Aggregation is mean by default; set score_quantile (e.g. 0.75, 0.90) to use
    that quantile across repeats instead.

    Row and column axes can use different distance metrics and linkage methods via
    row_metric / row_method / col_metric / col_method. When any of these is None
    it falls back to the shared metric / method defaults. This allows, for example,
    keeping euclidean + ward for rows (task periods) while using cosine + average
    for columns (neurons) on unnormalized data.
    """
    # Resolve per-axis metric/method, falling back to shared defaults.
    _row_metric = row_metric if row_metric is not None else metric
    _row_method = row_method if row_method is not None else method
    _col_metric = col_metric if col_metric is not None else metric
    _col_method = col_method if col_method is not None else method

    print(f"Row  — method: {_row_method}, metric: {_row_metric}")
    print(f"Col  — method: {_col_method}, metric: {_col_metric}")

    V = np.asarray(V)

    row_res = _hierarchical_clustering_repeat(
        V, k_min, k_max, _row_metric, _row_method,
        n_repeats=n_repeats,
        resample_features_frac=resample_features_frac,
        jitter_std=jitter_std,
        random_state=random_state,
        select_min_k_within_tol=select_min_k_within_tol,
        silhouette_tol=silhouette_tol,
        tol_mode=tol_mode,
        score_quantile=score_quantile,
        skip_unresponsive_detection=skip_unresponsive_detection,
        tol_k_select=tol_k_select,
    )

    col_res = _hierarchical_clustering_repeat(
        V.T, k_min, k_max, _col_metric, _col_method,
        n_repeats=n_repeats,
        resample_features_frac=resample_features_frac,
        jitter_std=jitter_std,
        random_state=None if random_state is None else (random_state + 1),
        select_min_k_within_tol=select_min_k_within_tol,
        silhouette_tol=silhouette_tol,
        tol_mode=tol_mode,
        score_quantile=score_quantile,
        skip_unresponsive_detection=skip_unresponsive_detection,
        tol_k_select=tol_k_select,
    )

    return dict(
        row_order=row_res["leaf_order"],
        col_order=col_res["leaf_order"],
        # Tolerance-selected labels and k (primary result).
        # Use row_strict_best_k / col_strict_best_k for the raw silhouette argmax.
        row_labels=row_res["alt_labels"],
        col_labels=col_res["alt_labels"],
        row_k=row_res["alt_k"],
        col_k=col_res["alt_k"],
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
        row_cut_threshold=row_res["alt_cut_threshold"],
        col_cut_threshold=col_res["alt_cut_threshold"],
        _row_meta=row_res["_params"],
        _col_meta=col_res["_params"],
        # Strict argmax (before tolerance).
        row_strict_best_k=row_res["k"],
        col_strict_best_k=col_res["k"],
        row_strict_labels=row_res["labels"],
        col_strict_labels=col_res["labels"],
        row_strict_cut_threshold=row_res["cut_threshold"],
        col_strict_cut_threshold=col_res["cut_threshold"],
        # Aliases — tol_* == primary (row_k / row_labels) for API consistency
        # with cluster_variance_matrix_forgroup.
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
    method: str = "ward",
    *,
    select_min_k_within_tol: bool = True,
    silhouette_tol: float = 0.02,
    tol_mode: str = "relative",
    tol_k_select: str = "min",
    skip_unresponsive_detection: bool = False,
    unresponsive_norm_frac: float = 1e-3,
) -> Dict[str, Any]:
    """
    Hierarchical clustering on `data` (observations × features)
    with tolerance-based k selection and an alternative k near the chosen k.
    metric/method: use "euclidean"/"ward" (default) or "cosine"/"average".

    tol_k_select : "min" selects the smallest k within the tolerance band of
        the best silhouette score; "max" selects the largest.
    skip_unresponsive_detection : when False (default), rows whose L2 norm is
        below unresponsive_norm_frac * max_norm are excluded from clustering
        and assigned a dedicated label (k+1), appended at the end of leaf_order.
        Set to True to disable detection (e.g. already-normalised data).
    """
    if tol_k_select not in ("min", "max"):
        raise ValueError("tol_k_select must be 'min' or 'max'.")
    _k_select = min if tol_k_select == "min" else max

    n_obs = data.shape[0]

    # ------------------------------------------------------------------
    # Unresponsive observation detection.
    # Rows whose L2 norm is < unresponsive_norm_frac * max_norm are
    # excluded from clustering.  They are assigned label k+1 and appended
    # at the end of leaf_order so they form a contiguous block in heatmaps.
    # ------------------------------------------------------------------
    if skip_unresponsive_detection:
        unresponsive_mask = np.zeros(n_obs, dtype=bool)
    else:
        norms = np.linalg.norm(data, axis=1)
        max_norm = norms.max() if norms.max() > 0 else 1.0
        unresponsive_mask = norms < unresponsive_norm_frac * max_norm

    n_unresponsive = int(unresponsive_mask.sum())
    if n_unresponsive > 0:
        warnings.warn(
            f"{n_unresponsive} unresponsive observation(s) detected "
            f"(norm < {unresponsive_norm_frac} * max_norm = "
            f"{unresponsive_norm_frac * (np.linalg.norm(data, axis=1).max() if not skip_unresponsive_detection else 0.):.3g}); "
            f"excluding from clustering and assigning to dedicated cluster (label = k+1).",
            RuntimeWarning,
            stacklevel=3,
        )

    active_mask = ~unresponsive_mask
    active_idx  = np.where(active_mask)[0]
    unres_idx   = np.where(unresponsive_mask)[0]
    active_data = data[active_mask]
    n_active    = active_data.shape[0]

    def _expand_labels(labels_active, k_active):
        """Map labels from n_active observations back to n_obs; unresponsive get label k+1."""
        full = np.zeros(n_obs, dtype=int)
        full[active_mask] = labels_active
        if n_unresponsive > 0:
            full[unresponsive_mask] = k_active + 1
        return full

    def _expand_leaf_order(leaf_order_active):
        """Map active-observation leaf order back to original indices, unresponsive appended."""
        return list(active_idx[np.array(leaf_order_active)]) + list(unres_idx)

    # Degenerate case: fewer than 2 active observations
    if n_active < 2:
        k_deg = 1
        labels_deg = _expand_labels(np.ones(max(n_active, 1), dtype=int), k_deg)
        return dict(
            linkage=None,
            leaf_order=_expand_leaf_order(list(range(n_active))),
            labels=labels_deg,
            k=k_deg,
            silhouette=np.nan,
            score_recording={},
            cophenet_score=np.nan,
            cut_distance_by_k={},
            best_cut_distance=None,
            labels_by_k={},
            strict_best_k=k_deg,
            strict_best_score=np.nan,
            alt_k=k_deg,
            alt_labels=labels_deg,
            alt_cut_distance=None,
            primary_candidates=[k_deg],
        )

    pairwise = pdist(active_data, metric=metric)
    Z = linkage(pairwise, method=method)
    c, _ = cophenet(Z, pairwise)  # cophenetic correlation

    D_sq = squareform(pairwise)
    k_range = range(max(k_min, 2), min(k_max, n_active - 1) + 1)

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

    # Tolerance threshold for primary selection
    if select_min_k_within_tol:
        if tol_mode == "relative":
            primary_thresh = strict_best_score * (1.0 - float(silhouette_tol))
        elif tol_mode == "absolute":
            primary_thresh = strict_best_score - float(silhouette_tol)
        else:
            raise ValueError("tol_mode must be 'relative' or 'absolute'.")
        primary_candidates = [k for k in k_range if score_recording[k] >= primary_thresh]
        best_k = _k_select(primary_candidates) if primary_candidates else strict_best_k
    else:
        best_k = strict_best_k
        primary_candidates = [strict_best_k]

    best_labels = labels_by_k[best_k]
    best_score = score_recording[best_k]
    best_cut_distance = cut_distance_by_k.get(best_k, _cut_distance_for_k(Z, best_k))

    # Secondary tolerance (alt_k) relative to chosen best_k
    if tol_mode == "relative":
        alt_thresh = best_score * (1.0 - float(silhouette_tol))
    else:
        alt_thresh = best_score - float(silhouette_tol)
    alt_candidates = [k for k in k_range if score_recording[k] >= alt_thresh]
    alt_k = _k_select(alt_candidates) if alt_candidates else best_k
    alt_labels = labels_by_k[alt_k]
    alt_cut_distance = cut_distance_by_k.get(alt_k, _cut_distance_for_k(Z, alt_k))

    leaf_order_active = dendrogram(Z, no_plot=True)["leaves"]

    return dict(
        linkage=Z,
        leaf_order=_expand_leaf_order(leaf_order_active),
        labels=_expand_labels(best_labels, best_k),
        k=best_k,
        silhouette=best_score,
        score_recording=score_recording,
        cophenet_score=c,
        cut_distance_by_k=cut_distance_by_k,
        best_cut_distance=best_cut_distance,
        labels_by_k={k: _expand_labels(lbls, k) for k, lbls in labels_by_k.items()},
        strict_best_k=strict_best_k,
        strict_best_score=strict_best_score,
        primary_candidates=primary_candidates,
        alt_k=alt_k,
        alt_labels=_expand_labels(alt_labels, alt_k),
        alt_cut_distance=alt_cut_distance,
    )


def cluster_variance_matrix_forgroup(
    V: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
    row_groups: Optional[Sequence[Sequence[int]]] = None,
    col_groups_all_lst: Optional[List[Sequence[Sequence[int]]]] = None,
    *,
    metric: str = "euclidean",
    row_metric: Optional[str] = None,
    row_method: Optional[str] = None,
    col_metric: Optional[str] = None,
    col_method: Optional[str] = None,
    select_min_k_within_tol: bool = True,
    silhouette_tol: float = 0.02,
    tol_mode: str = "relative",
    tol_k_select: str = "min",
    skip_unresponsive_detection: bool = False,
    score_quantile: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Bi-directional hierarchical clustering of a variance matrix V
    (shape: N_features * M_neurons) with tolerance-based k selection.

    metric : shared fallback when row_metric / col_metric are not provided.
        Supported values: "euclidean" (Ward linkage) or "cosine" (average linkage).
    row_metric / row_method : override metric/method for the row axis.
    col_metric / col_method : override metric/method for the column axis.
        method defaults to the canonical choice for the given metric
        ("euclidean" → "ward", "cosine" → "average") when not specified.

    col_groups_all_lst : list of col_groups, each from a different trial/seed.
        All groupings are used to compute per-k silhouette scores; the scores
        are aggregated across groupings to determine the best column k.
        The first element is used as the primary grouping for the final
        linkage, leaf ordering, and label assignment.

    score_quantile : if given (e.g. 0.5, 0.75), use that quantile across
        col groupings when aggregating silhouette scores for k selection,
        rather than the mean.  Mirrors the same parameter in
        cluster_variance_matrix_repeat.  Has no effect when only one col
        grouping is provided (mean == quantile).
    """
    _METRIC_TO_METHOD = {"euclidean": "ward", "cosine": "average"}

    _row_metric = row_metric if row_metric is not None else metric
    _col_metric = col_metric if col_metric is not None else metric

    for _ax, _m in (("row", _row_metric), ("col", _col_metric)):
        if _m not in _METRIC_TO_METHOD:
            raise ValueError(
                f"{_ax}_metric must be one of {list(_METRIC_TO_METHOD)}; got {_m!r}"
            )

    _row_method = row_method if row_method is not None else _METRIC_TO_METHOD[_row_metric]
    _col_method = col_method if col_method is not None else _METRIC_TO_METHOD[_col_metric]

    if tol_k_select not in ("min", "max"):
        raise ValueError("tol_k_select must be 'min' or 'max'.")
    _k_select = min if tol_k_select == "min" else max

    print(f"Row  — method: {_row_method}, metric: {_row_metric}")
    print(f"Col  — method: {_col_method}, metric: {_col_metric}")

    V = np.asarray(V)

    # shared kwargs forwarded to every _hierarchical_clustering_forgroup call
    _hcfg_kw = dict(
        select_min_k_within_tol=select_min_k_within_tol,
        silhouette_tol=silhouette_tol,
        tol_mode=tol_mode,
        tol_k_select=tol_k_select,
        skip_unresponsive_detection=skip_unresponsive_detection,
    )

    # ----- rows -----
    row_blocks, row_map = _build_groups(V.shape[0], row_groups)
    V_row_grp = _aggregate_along_axis(V, row_blocks, axis=0, reduce="mean")
    row_res = _hierarchical_clustering_forgroup(
        V_row_grp, k_min, k_max,
        metric=_row_metric, method=_row_method,
        **_hcfg_kw,
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
            metric=_col_metric, method=_col_method,
            **_hcfg_kw,
        )
        col_res_list.append(col_res_i)
        col_blocks_list.append(col_blocks_i)
        col_V_grp_list.append(V_col_grp_i)

    # Aggregate silhouette scores across groupings for each k.
    # col_score_mean is always the mean (used for reporting/plotting).
    # col_score_agg uses score_quantile if specified; otherwise equals col_score_mean.
    # k selection is driven by col_score_agg.
    all_k_col: set = set()
    for col_res_i in col_res_list:
        all_k_col.update(col_res_i["score_recording"].keys())

    col_score_mean: Dict[int, float] = {}
    col_score_std: Dict[int, float] = {}
    col_score_agg: Dict[int, float] = {}
    for k in sorted(all_k_col):
        vals = [col_res_i["score_recording"][k]
                for col_res_i in col_res_list
                if k in col_res_i["score_recording"]]
        col_score_mean[k] = float(np.mean(vals))
        col_score_std[k] = float(np.std(vals))
        col_score_agg[k] = (
            float(np.quantile(vals, score_quantile))
            if score_quantile is not None
            else col_score_mean[k]
        )

    # Re-select best col k from the aggregated silhouette curve.
    col_strict_best_k = max(col_score_agg, key=col_score_agg.get)
    col_strict_best_score = col_score_agg[col_strict_best_k]

    if select_min_k_within_tol:
        if tol_mode == "relative":
            col_primary_thresh = col_strict_best_score * (1.0 - float(silhouette_tol))
        elif tol_mode == "absolute":
            col_primary_thresh = col_strict_best_score - float(silhouette_tol)
        else:
            raise ValueError("tol_mode must be 'relative' or 'absolute'.")
        col_primary_candidates = [k for k in col_score_agg
                                   if col_score_agg[k] >= col_primary_thresh]
        best_k_col = _k_select(col_primary_candidates) if col_primary_candidates else col_strict_best_k
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
        local = _within_block_leaf_order(X, metric=_row_metric, method=_row_method,
                                         max_items=2000, fallback="pca1")
        row_order.extend(idxs[local].tolist())

    col_order: List[int] = []
    for g in col_res["leaf_order"]:
        idxs = np.asarray(col_blocks[g], dtype=int)
        if idxs.size <= 2:
            col_order.extend(idxs.tolist())
            continue
        X = V_row_grp[:, idxs].T        # (n_in_block, R_blk)
        local = _within_block_leaf_order(X, metric=_col_metric, method=_col_method,
                                         max_items=2000, fallback="pca1")
        col_order.extend(idxs[local].tolist())

    row_labels = np.take(row_res["labels"], row_map)
    row_labels_by_k = {k: np.take(lbls, row_map) for k, lbls in row_res["labels_by_k"].items()}

    col_labels = np.take(col_group_labels_at_best_k, col_map)
    col_labels_by_k = {k: np.take(lbls, col_map) for k, lbls in col_res["labels_by_k"].items()}

    # --- pre-compute arrays for score-curve keys ---
    row_score_recording = row_res["score_recording"]
    sorted_k_row = sorted(row_score_recording.keys())
    sorted_k_col = sorted(all_k_col)

    # col_score_recording_all: shape (n_groupings, n_k_col)
    col_score_recording_all = np.array(
        [[col_res_i["score_recording"].get(k, np.nan) for k in sorted_k_col]
         for col_res_i in col_res_list],
        dtype=float,
    )

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
        row_score_recording=row_score_recording,
        # col silhouette: mean/std/agg across all groupings
        col_score_recording_mean=col_score_mean,
        col_score_recording_std=col_score_std,
        col_score_recording_agg=col_score_agg,          # used for k selection
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
        # --- API aliases for consistency with cluster_variance_matrix_repeat ---
        # In cluster_variance_matrix_repeat:  row_k  = strict best;  row_tol_k = tolerance.
        # Here:  row_k  = tolerance-selected  (what _hierarchical_clustering_forgroup calls
        #        best_k);  row_strict_best_k  is the raw argmax.
        # Aliases below ensure code that always reads row_tol_labels / row_tol_k
        # works consistently across both functions.
        row_tol_labels=row_labels,
        col_tol_labels=col_labels,
        row_tol_k=row_res["k"],
        col_tol_k=best_k_col,
        row_cut_threshold=row_res["best_cut_distance"],
        col_cut_threshold=col_best_cut_distance,
        # score-curve arrays matching cluster_variance_matrix_repeat return shape
        row_score_recording_mean=row_score_recording,           # single run: mean == value
        row_score_recording_std={k: 0.0 for k in row_score_recording},
        row_score_recording_all=np.array(
            [[row_score_recording[k] for k in sorted_k_row]], dtype=float
        ),                                                       # shape (1, n_k_row)
        row_k_values=np.array(sorted_k_row, dtype=int),
        col_k_values=np.array(sorted_k_col, dtype=int),
        col_score_recording_all=col_score_recording_all,        # shape (n_groupings, n_k_col)
        # introspection meta dicts
        _row_meta=dict(
            method=_row_method,
            metric=_row_metric,
            select_min_k_within_tol=select_min_k_within_tol,
            silhouette_tol=silhouette_tol,
            tol_mode=tol_mode,
            score_quantile=score_quantile,
            tol_k_select=tol_k_select,
            chosen_k=row_res["k"],
            chosen_score=row_res["silhouette"],
            strict_best_k=row_res["strict_best_k"],
            strict_best_score=row_res["strict_best_score"],
            primary_candidates=row_res["primary_candidates"],
        ),
        _col_meta=dict(
            method=_col_method,
            metric=_col_metric,
            n_groupings=len(col_res_list),
            select_min_k_within_tol=select_min_k_within_tol,
            silhouette_tol=silhouette_tol,
            tol_mode=tol_mode,
            score_quantile=score_quantile,
            tol_k_select=tol_k_select,
            chosen_k=best_k_col,
            strict_best_k=col_strict_best_k,
            strict_best_score=col_strict_best_score,
            primary_candidates=col_primary_candidates,
        ),
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
    