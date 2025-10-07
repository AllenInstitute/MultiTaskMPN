import numpy as np 

def _to_zero_based(labels):
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

    return {
        "metrics": {
            "CH_blocks": ch,
            "DB_blocks": db,
            "Dunn_blocks": dunn,
            "silhouette_sq_blocks": silhouette_sq,
            "silhouette_L2approx_blocks": silhouette_l2_approx,
            "K_row": Kr, "K_col": Kc, "K_blocks": K,
        },
        "blocks": {
            "counts": counts, "means": means, "var": var, "std": std,
            "withinss": withinss, "betweenss": betweenss, "totss": stats["totss"],
            "global_mean": stats["mu_global"],
        },
    }