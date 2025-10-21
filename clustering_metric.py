import numpy as np 

def count_pairs_with_clusters(col_all: np.ndarray,
                              M: int,
                              pre_cluster: dict,
                              post_cluster: dict):
    """
    Vectorized counts over unordered pairs within each group in col_all.

    Returns a tuple:
      (
        same_pre_all,                  # i//M equal
        same_post_all,                 # i%M equal
        no_same_all,                   # neither same pre nor same post

        same_pre_cluster_all,          # same pre *cluster*
        same_post_cluster_all,         # same post *cluster*
        both_pre_post_cluster_all,     # same pre-cluster AND same post-cluster
        no_pre_post_cluster_all        # neither same pre-cluster nor same post-cluster
      )
    """

    N = col_all.size
    if N == 0:
        return (0, 0, 0, 0, 0, 0, 0)

    # Precompute index → (pre, post)
    idx_all  = np.arange(N)
    pre_all  = idx_all // M
    post_all = idx_all % M

    # -------- Build fast reverse maps: index -> cluster id (int) --------
    # Map cluster keys to consecutive ints and fill arrays in O(N)
    def build_reverse_map(cluster_dict, size_hint):
        # cluster_dict: {cluster_key: [indices]}
        # returns: (arr_of_len>=max_index+1 with cluster ids, n_clusters)
        arr = np.full(size_hint, -1, dtype=int)
        for cid, (_, inds) in enumerate(cluster_dict.items()):
            inds = np.asarray(inds, dtype=int)
            if inds.size:
                if inds.max() >= arr.size:
                    # grow if needed (rare; defensive)
                    arr = np.pad(arr, (0, inds.max() + 1 - arr.size), constant_values=-1)
                arr[inds] = cid
        n_clusters = max(cid + 1, 0) if cluster_dict else 0
        return arr, n_clusters

    # Size hints from observed indices
    pre_size_hint  = pre_all.max() + 1
    post_size_hint = post_all.max() + 1   # typically == M

    pre_to_cluster, n_pre_clusters   = build_reverse_map(pre_cluster,  pre_size_hint)
    post_to_cluster, n_post_clusters = build_reverse_map(post_cluster, post_size_hint)

    if (pre_to_cluster[pre_all] < 0).any():
        raise ValueError("Some pre indices are missing from pre_cluster.")
    if (post_to_cluster[post_all] < 0).any():
        raise ValueError("Some post indices are missing from post_cluster.")

    # Cluster id per edge index
    preC_all  = pre_to_cluster[pre_all]
    postC_all = post_to_cluster[post_all]

    # -------------------- Aggregate per group --------------------
    same_pre_all = same_post_all = no_same_all = 0
    same_pre_cluster_all = same_post_cluster_all = 0
    both_pre_post_cluster_all = no_pre_post_cluster_all = 0

    for g in np.unique(col_all):
        idx = np.flatnonzero(col_all == g)
        n = idx.size
        if n < 2:
            continue

        pre   = pre_all[idx]
        post  = post_all[idx]
        preC  = preC_all[idx]
        postC = postC_all[idx]

        total_pairs = n * (n - 1) // 2

        # exact pre equality
        cnt_pre  = np.bincount(pre)
        same_pre = np.sum(cnt_pre * (cnt_pre - 1) // 2)

        # exact post equality
        cnt_post  = np.bincount(post, minlength=M)
        same_post = np.sum(cnt_post * (cnt_post - 1) // 2)

        no_same = total_pairs - same_pre - same_post

        # same pre *cluster*
        cnt_preC = np.bincount(preC, minlength=n_pre_clusters)
        same_preC = np.sum(cnt_preC * (cnt_preC - 1) // 2)

        # same post *cluster*
        cnt_postC = np.bincount(postC, minlength=n_post_clusters)
        same_postC = np.sum(cnt_postC * (cnt_postC - 1) // 2)

        # both same pre-cluster AND same post-cluster
        comb_id = np.ravel_multi_index((preC, postC), (n_pre_clusters, n_post_clusters))
        cnt_both = np.bincount(comb_id, minlength=n_pre_clusters * n_post_clusters)
        both_clusters = np.sum(cnt_both * (cnt_both - 1) // 2)

        # neither same pre-cluster nor same post-cluster
        # Use inclusion-exclusion: |none| = total - |preC| - |postC| + |both|
        no_pre_post_cluster = total_pairs - (same_preC + same_postC - both_clusters)

        # accumulate
        same_pre_all += int(same_pre)
        same_post_all += int(same_post)
        no_same_all += int(no_same)

        same_pre_cluster_all += int(same_preC)
        same_post_cluster_all += int(same_postC)
        both_pre_post_cluster_all += int(both_clusters)
        no_pre_post_cluster_all += int(no_pre_post_cluster)

    return (
        same_pre_all,
        same_post_all,
        no_same_all,
        same_pre_cluster_all,
        same_post_cluster_all,
        both_pre_post_cluster_all,
        no_pre_post_cluster_all,
    )

def _to_zero_based(labels):
    """
    """
    labels = np.asarray(labels)
    uniques = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniques)}
    z = np.vectorize(remap.get)(labels)
    K = len(uniques)
    return z.astype(int), K

def _block_stats(V, row_labels, col_labels):
    """
    Compute per-block counts, sums, sums of squares, means, variances
    using only matrix multiplications with one-hot membership.
    """
    V = np.asarray(V, dtype=np.float64)
    n_r, n_c = V.shape
    r, Kr = _to_zero_based(row_labels)
    c, Kc = _to_zero_based(col_labels)

    # One-hot membership matrices
    R = np.zeros((n_r, Kr), dtype=np.float64)
    R[np.arange(n_r), r] = 1.0
    C = np.zeros((n_c, Kc), dtype=np.float64)
    C[np.arange(n_c), c] = 1.0

    # Block counts = outer(row_counts, col_counts)
    row_counts = R.sum(axis=0)           # (Kr,)
    col_counts = C.sum(axis=0)           # (Kc,)
    counts = np.outer(row_counts, col_counts)  # (Kr, Kc), all > 0

    # Block sums / sums of squares
    sum_blocks   = R.T @ V        @ C    # (Kr, Kc)
    sumsq_blocks = R.T @ (V*V)    @ C    # (Kr, Kc)

    means = sum_blocks / counts
    var   = np.maximum(0.0, sumsq_blocks / counts - means**2)  # population var
    sse   = sumsq_blocks - counts * means**2                   # within-block SSE

    # Global stats
    N = V.size
    mu_global = float(V.mean())
    totss = float(np.sum(V*V) - N * mu_global**2)              # total SS about global mean
    withinss = float(np.sum(sse))
    betweenss = float(totss - withinss)

    return {
        "Kr": Kr, "Kc": Kc, "K": Kr*Kc,
        "counts": counts, "means": means, "var": var, "std": np.sqrt(var),
        "sse": sse, "totss": totss, "withinss": withinss, "betweenss": betweenss,
        "N": N, "mu_global": mu_global
    }

def evaluate_bicluster_clustering(V, row_labels, col_labels):
    """
    Evaluate the *product* (row × col) clustering as K=Kr*Kc blocks.

    Metrics (all higher is better except Davies–Bouldin):
      - CH_blocks: Calinski–Harabasz using blocks as clusters of cells (squared L2)
      - DB_blocks: Davies–Bouldin using block centroids and within-block scatter
      - Dunn_blocks: Dunn index using centroid separations and within-block scatter
      - silhouette_sq_blocks: exact silhouette with *squared* Euclidean distances
      - silhouette_L2approx_blocks: same but with sqrt to approximate L2-silhouette

    Returns:
      dict with metrics and a 'blocks' sub-dict containing per-block stats.
    """
    stats = _block_stats(V, row_labels, col_labels)
    Kr, Kc, K = stats["Kr"], stats["Kc"], stats["K"]
    counts, means, var, std = stats["counts"], stats["means"], stats["var"], stats["std"]
    withinss, betweenss, N = stats["withinss"], stats["betweenss"], stats["N"]

    # ---- Calinski–Harabasz (blocks; higher is better) ----
    if K >= 2 and (N - K) > 0 and withinss > 0:
        ch = (betweenss / (K - 1)) / (withinss / (N - K))
    else:
        ch = np.nan

    # Flatten block stats
    mu = means.ravel()
    s  = np.sqrt(np.maximum(0.0, var.ravel()))
    n  = counts.ravel().astype(np.float64)

    # ---- Davies–Bouldin (robust; lower is better) ----
    # R_ij = (s_i + s_j) / ||mu_i - mu_j||
    # Handle same-centroid cases:
    #  - if numerator==0 (both zero scatter), treat as 0 (not confusable)
    #  - if numerator>0, treat as +inf (overlapping)
    Kb = mu.size
    if Kb >= 2:
        diff = np.abs(mu[:, None] - mu[None, :])
        num  = s[:, None] + s[None, :]

        R = np.empty_like(diff)
        same = (diff == 0.0)

        # Regular ratios
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(num, diff, out=R, where=~same)

        # Identical centroids
        R[same & (num == 0.0)] = 0.0
        R[same & (num  > 0.0)] = np.inf

        # Exclude self
        np.fill_diagonal(R, -np.inf)
        db = float(np.mean(np.max(R, axis=1)))
    else:
        db = np.nan

    # ---- Dunn index (higher is better) ----
    delta = float(np.min(diff)) if K >= 2 else np.nan          # min centroid sep
    bigD  = float(2.0 * np.max(s)) if K >= 1 else np.nan       # max within diameter ~ 2*std
    dunn = delta / bigD if (K >= 2 and bigD > 0) else np.nan

    # ---- Silhouette with squared L2 (exact, higher is better) ----
    var_flat = var.ravel()
    a2 = np.zeros_like(var_flat)
    mask_gt1 = (n > 1)
    a2[mask_gt1] = (2.0 * n[mask_gt1] / (n[mask_gt1] - 1.0)) * var_flat[mask_gt1]
    a2[~mask_gt1] = 0.0

    mudiff2 = (mu[:, None] - mu[None, :])**2
    b2_mat = var_flat[:, None] + var_flat[None, :] + mudiff2
    np.fill_diagonal(b2_mat, np.inf)
    b2 = np.min(b2_mat, axis=1)

    denom2 = np.maximum(a2, b2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sil2 = (b2 - a2) / denom2
    silhouette_sq = float(np.sum(n * sil2) / np.sum(n)) if Kb >= 2 else np.nan

    # L2-approx silhouette (sqrt of a2/b2)
    a = np.sqrt(a2)
    b = np.sqrt(b2)
    denom = np.maximum(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        sil_l2 = (b - a) / denom
    silhouette_l2_approx = float(np.sum(n * sil_l2) / np.sum(n)) if Kb >= 2 else np.nan

    # ---- Xie–Beni index (lower is better) ----
    if Kb >= 2:
        # min squared separation between distinct block means
        offdiag = ~np.eye(Kb, dtype=bool)
        sep2 = mudiff2[offdiag]
        # consider strictly positive separations to avoid zero-distance degeneracy
        pos_sep2 = sep2[sep2 > 0.0]
        if pos_sep2.size > 0:
            min_sep2 = float(np.min(pos_sep2))
            xb = float(withinss) / (float(N) * min_sep2) if N > 0 else np.nan
        else:
            # All centroids coincide pairwise: degenerate. If withinss>0 -> inf, else 0.
            xb = 0.0 if withinss == 0.0 else np.inf
    else:
        xb = np.nan

    return {
        "metrics": {
            "CH_blocks": ch,
            "DB_blocks": db,
            "Dunn_blocks": dunn,
            "silhouette_sq_blocks": silhouette_sq,
            "silhouette_L2approx_blocks": silhouette_l2_approx,
            "XB_blocks": xb, 
            "K_row": Kr, "K_col": Kc, "K_blocks": K,
        },
        "blocks": {
            "counts": counts, "means": means, "var": var, "std": std,
            "withinss": withinss, "betweenss": betweenss, "totss": stats["totss"],
            "global_mean": stats["mu_global"],
        },
    }
