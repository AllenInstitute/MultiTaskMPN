import numpy as np 

# ---------------------------------------------------------------------
# Find if modulation are belonging to some pre/post neuron and/or 
# pre/post neuron cluster
# ---------------------------------------------------------------------


def _build_reverse_map(cluster_dict, size_hint):
    """Map cluster_dict {key: [indices]} → (lookup_array, n_clusters).

    lookup_array[index] == cluster_id (0-based consecutive int).
    """
    arr = np.full(size_hint, -1, dtype=int)
    n_clusters = 0
    for cid, (_, inds) in enumerate(cluster_dict.items()):
        inds = np.asarray(inds, dtype=int)
        if inds.size:
            if inds.max() >= arr.size:
                arr = np.pad(arr, (0, inds.max() + 1 - arr.size), constant_values=-1)
            arr[inds] = cid
        n_clusters = cid + 1
    return arr, n_clusters


def _precompute_pair_arrays(N, M, pre_cluster, post_cluster, flat_idx=None):
    """Precompute all index-level arrays shared across every shuffle repeat.

    Returns a dict with:
      pre_all, post_all          — pre/post neuron index per modulation
      preC_all, postC_all        — pre/post cluster id per modulation
      comb_id                    — ravelled (preC, postC) pair id per modulation
      n_pre_clusters, n_post_clusters

    flat_idx : optional array of original flat indices (length N).
        Pass when col_all has been filtered (e.g. unresponsive entries removed)
        so that pre = flat_idx // M and post = flat_idx % M remain correct.
        Defaults to np.arange(N).
    """
    if flat_idx is None:
        flat_idx = np.arange(N)
    pre_all = flat_idx // M
    post_all = flat_idx % M

    pre_to_cluster, n_pre_clusters = _build_reverse_map(pre_cluster, pre_all.max() + 1)
    post_to_cluster, n_post_clusters = _build_reverse_map(post_cluster, post_all.max() + 1)

    if (pre_to_cluster[pre_all] < 0).any():
        raise ValueError("Some pre indices are missing from pre_cluster.")
    if (post_to_cluster[post_all] < 0).any():
        raise ValueError("Some post indices are missing from post_cluster.")

    preC_all = pre_to_cluster[pre_all]
    postC_all = post_to_cluster[post_all]
    comb_id = np.ravel_multi_index((preC_all, postC_all), (n_pre_clusters, n_post_clusters))

    return dict(
        pre_all=pre_all, post_all=post_all,
        preC_all=preC_all, postC_all=postC_all,
        comb_id=comb_id,
        n_pre_clusters=n_pre_clusters, n_post_clusters=n_post_clusters,
    )


def _aggregate_pair_counts(col_all, M, arrays):
    """Compute the 7 pair-count statistics for one assignment of col_all.

    Uses pre-computed arrays from _precompute_pair_arrays to avoid
    rebuilding lookup tables.
    """
    pre_all = arrays["pre_all"]
    post_all = arrays["post_all"]
    preC_all = arrays["preC_all"]
    postC_all = arrays["postC_all"]
    comb_id = arrays["comb_id"]
    n_pre_clusters = arrays["n_pre_clusters"]
    n_post_clusters = arrays["n_post_clusters"]

    same_pre_all = same_post_all = no_same_all = 0
    same_pre_cluster_all = same_post_cluster_all = 0
    both_pre_post_cluster_all = no_pre_post_cluster_all = 0

    for g in np.unique(col_all):
        idx = np.flatnonzero(col_all == g)
        n = idx.size
        if n < 2:
            continue

        pre = pre_all[idx]
        post = post_all[idx]
        preC = preC_all[idx]
        postC = postC_all[idx]
        comb = comb_id[idx]

        total_pairs = n * (n - 1) // 2

        cnt_pre = np.bincount(pre)
        same_pre = int(np.sum(cnt_pre * (cnt_pre - 1) // 2))

        cnt_post = np.bincount(post, minlength=M)
        same_post = int(np.sum(cnt_post * (cnt_post - 1) // 2))

        cnt_preC = np.bincount(preC, minlength=n_pre_clusters)
        same_preC = int(np.sum(cnt_preC * (cnt_preC - 1) // 2))

        cnt_postC = np.bincount(postC, minlength=n_post_clusters)
        same_postC = int(np.sum(cnt_postC * (cnt_postC - 1) // 2))

        cnt_both = np.bincount(comb, minlength=n_pre_clusters * n_post_clusters)
        both_clusters = int(np.sum(cnt_both * (cnt_both - 1) // 2))

        same_pre_all += same_pre
        same_post_all += same_post
        no_same_all += total_pairs - same_pre - same_post
        same_pre_cluster_all += same_preC
        same_post_cluster_all += same_postC
        both_pre_post_cluster_all += both_clusters
        no_pre_post_cluster_all += total_pairs - (same_preC + same_postC - both_clusters)

    return (
        same_pre_all, same_post_all, no_same_all,
        same_pre_cluster_all, same_post_cluster_all,
        both_pre_post_cluster_all, no_pre_post_cluster_all,
    )


def count_pairs_with_clusters(col_all,
                              M,
                              pre_cluster,
                              post_cluster,
                              flat_idx=None):
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

    flat_idx : see _precompute_pair_arrays.
    """
    N = col_all.size
    if N == 0:
        return (0, 0, 0, 0, 0, 0, 0)

    arrays = _precompute_pair_arrays(N, M, pre_cluster, post_cluster, flat_idx=flat_idx)
    return _aggregate_pair_counts(col_all, M, arrays)


def _count_same_pairs_all_repeats(shuffled_groups, feature_all, feature_range, n_groups, repeat, N):
    """Count same-group AND same-feature pairs for all repeats simultaneously.

    Uses the combined-key flat-bincount trick:
      key[r, i] = shuffled_groups[r, i] * feature_range + feature_all[i]
    Encoding all (repeat, key_range) counts into one bincount call.

    Returns (repeat,) int64 array.
    """
    key = shuffled_groups * feature_range + feature_all[None, :]   # (repeat, N)
    key_range = n_groups * feature_range
    # Offset each repeat's keys so all repeats can share one bincount
    offsets = np.arange(repeat, dtype=np.int64)[:, None] * key_range
    key_flat = (key + offsets).ravel()
    counts_flat = np.bincount(key_flat, minlength=repeat * key_range)
    counts_2d = counts_flat.reshape(repeat, key_range)
    sum_sq = np.einsum("ij,ij->i", counts_2d, counts_2d)   # (repeat,) — Σ count_k^2
    return (sum_sq - N) // 2


def count_pairs_with_clusters_control(col_all,
                                      M,
                                      pre_cluster,
                                      post_cluster,
                                      repeat=10,
                                      flat_idx=None):
    """
    Control distribution for pre/post neuron & neuron clustering belonging.

    "Are these group labels more aligned with pre/post (or pre/post clusters)
    than random assignment given group sizes?"

    Vectorized: precomputes static arrays once, then processes all `repeat`
    shuffles simultaneously via combined-key flat bincount — O(repeat * N)
    instead of O(repeat * N * n_groups).

    flat_idx : see _precompute_pair_arrays.
    """
    N = col_all.size
    if N == 0:
        return np.zeros(7)

    arrays = _precompute_pair_arrays(N, M, pre_cluster, post_cluster, flat_idx=flat_idx)
    pre_all = arrays["pre_all"]
    post_all = arrays["post_all"]
    preC_all = arrays["preC_all"]
    postC_all = arrays["postC_all"]
    comb_id = arrays["comb_id"]
    n_pre_clusters = arrays["n_pre_clusters"]
    n_post_clusters = arrays["n_post_clusters"]

    # Remap group labels to 0-based consecutive ints (required by key encoding)
    unique_groups = np.unique(col_all)
    n_groups = unique_groups.size
    group_remap = np.empty(unique_groups.max() + 1, dtype=np.int64)
    group_remap[unique_groups] = np.arange(n_groups, dtype=np.int64)
    col_mapped = group_remap[col_all]  # (N,)

    # Fixed total pairs per group — identical for every shuffle (it's a permutation)
    group_sizes = np.bincount(col_mapped, minlength=n_groups)
    total_pairs = int(np.sum(group_sizes * (group_sizes - 1) // 2))

    # Generate all permutations at once → shuffled group labels (repeat, N)
    perm_matrix = np.stack([np.random.permutation(N) for _ in range(repeat)])  # (repeat, N)
    shuffled_groups = col_mapped[perm_matrix].astype(np.int64)                  # (repeat, N)

    # Vectorized pair counts for every feature across all repeats
    same_pre = _count_same_pairs_all_repeats(
        shuffled_groups, pre_all.astype(np.int64), pre_all.max() + 1, n_groups, repeat, N)
    same_post = _count_same_pairs_all_repeats(
        shuffled_groups, post_all.astype(np.int64), M, n_groups, repeat, N)
    same_preC = _count_same_pairs_all_repeats(
        shuffled_groups, preC_all.astype(np.int64), n_pre_clusters, n_groups, repeat, N)
    same_postC = _count_same_pairs_all_repeats(
        shuffled_groups, postC_all.astype(np.int64), n_post_clusters, n_groups, repeat, N)
    same_both = _count_same_pairs_all_repeats(
        shuffled_groups, comb_id.astype(np.int64), n_pre_clusters * n_post_clusters,
        n_groups, repeat, N)

    no_same = total_pairs - same_pre - same_post
    no_both = total_pairs - (same_preC + same_postC - same_both)

    control_stats = np.stack(
        [same_pre, same_post, no_same, same_preC, same_postC, same_both, no_both],
        axis=1,
    ).astype(float)   # (repeat, 7)

    # Sanity checks (mirror the original assertions)
    total_neuron_group = control_stats[:, 0] + control_stats[:, 1] + control_stats[:, 2]
    assert np.unique(total_neuron_group).size == 1, "Elements in total_neuron_group are not identical"
    total_cluster_group = control_stats[:, 3] + control_stats[:, 4] - control_stats[:, 5] + control_stats[:, 6]
    assert np.unique(total_cluster_group).size == 1, "Elements in total_cluster_group are not identical"

    return np.mean(control_stats, axis=0)

# ---------------------------------------------------------------------
# Evaluation metric to clustering
# ---------------------------------------------------------------------

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
