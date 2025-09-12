import numpy as np

def _to_zero_based(labels):
    """
    """
    labels = np.asarray(labels)
    # fcluster returns 1..K; make contiguous 0..K-1
    uniques = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniques)}
    z = np.vectorize(remap.get)(labels)
    K = len(uniques)
    return z.astype(int), K

def _block_stats(V, row_labels, col_labels):
    """
    """
    V = np.asarray(V, dtype=np.float64)
    n_r, n_c = V.shape
    r, Kr = _to_zero_based(row_labels)
    c, Kc = _to_zero_based(col_labels)

    R = np.zeros((n_r, Kr), dtype=np.float64)
    R[np.arange(n_r), r] = 1.0
    C = np.zeros((n_c, Kc), dtype=np.float64)
    C[np.arange(n_c), c] = 1.0

    row_counts = R.sum(axis=0)          
    col_counts = C.sum(axis=0)        
    counts = np.outer(row_counts, col_counts)  

    sum_blocks   = R.T @ V @ C   
    sumsq_blocks = R.T @ (V*V) @ C   

    means = sum_blocks / counts
    var   = np.maximum(0.0, sumsq_blocks / counts - means**2) 
    sse   = sumsq_blocks - counts * means**2                 

    N = V.size
    mu_global = float(V.mean())
    totss = float(np.sum(V*V) - N * mu_global**2)              
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
    """
    stats = _block_stats(V, row_labels, col_labels)
    Kr, Kc, K = stats["Kr"], stats["Kc"], stats["K"]
    counts, means, var, std = stats["counts"], stats["means"], stats["var"], stats["std"]
    withinss, betweenss, N = stats["withinss"], stats["betweenss"], stats["N"]

    if K >= 2 and (N - K) > 0 and withinss > 0:
        ch = (betweenss / (K - 1)) / (withinss / (N - K))
    else:
        ch = np.nan

    mu = means.ravel()
    s  = np.sqrt(np.maximum(0.0, var.ravel()))
    n  = counts.ravel().astype(np.float64)

    diff = np.abs(mu[:, None] - mu[None, :]) 
    np.fill_diagonal(diff, np.inf) 

    with np.errstate(divide="ignore", invalid="ignore"):
        R = (s[:, None] + s[None, :]) / diff
    np.fill_diagonal(R, -np.inf)
    db = float(np.mean(np.max(R, axis=1))) if K >= 2 else np.nan

    delta = float(np.min(diff)) if K >= 2 else np.nan         
    bigD  = float(2.0 * np.max(s)) if K >= 1 else np.nan      
    dunn = delta / bigD if (K >= 2 and bigD > 0) else np.nan

    var_flat = var.ravel()
    a2 = np.zeros_like(var_flat)
    mask_gt1 = (n > 1)
    a2[mask_gt1] = (2.0 * n[mask_gt1] / (n[mask_gt1] - 1.0)) * var_flat[mask_gt1]
    a2[~mask_gt1] = 0.0

    mudiff2 = (mu[:, None] - mu[None, :])**2
    b2_mat = var_flat[:, None] + var_flat[None, :] + mudiff2
    np.fill_diagonal(b2_mat, np.inf)
    b2 = np.min(b2_mat, axis=1)

    denom = np.maximum(a2, b2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sil2_blocks = (b2 - a2) / denom

    silhouette_sq = float(np.sum(n * sil2_blocks) / np.sum(n)) if K >= 2 else np.nan

    a = np.sqrt(a2)
    b = np.sqrt(b2)
    denom_l2 = np.maximum(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        sil_l2_blocks = (b - a) / denom_l2
    silhouette_l2_approx = float(np.sum(n * sil_l2_blocks) / np.sum(n)) if K >= 2 else np.nan

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
            "global_mean": stats["mu_global"]
        }
    }
