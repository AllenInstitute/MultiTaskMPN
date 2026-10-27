"""Shared registry of the modulation (synapse) clustering variants.

The multi-task pipeline clusters the plastic synapses of `mp_layer1` on
task-conditioned variance features. Five feature definitions exist, and the
three pipeline stages must agree on their names, order and colors:

  multiple_task_analysis.py  clusters each variant and saves it under
                             `{clustering_name}_{normalized|unnormalized}` in
                             cluster_info_mod_{aname}.pkl
  lesion.py                  lesions every saved variant listed in
                             LESION_MODULATION_TYPES
  lesion_plot.py             orders and colors its per-variant comparisons by
                             LESION_MODULATION_TYPES / MODULATION_TYPE_COLORS

| clustering_name             | feature per synapse and task period            |
|-----------------------------|------------------------------------------------|
| modulation_all (normalized) | Var(M), divided by the synapse's max over rules|
| modulation_all              | Var(M)                                          |
| modulation_all_weighted     | Var(W * M)  (= W^2 Var(M) for a static W)       |
| modulation_all_var_weighted | W * Var(M)  (signed; positive- and negative-W   |
|                             | synapses form separate clusters)                |
| modulation_all_abs_weighted | |W| * Var(M) (magnitude-weighted, sign-blind)   |

Signed features need two sign-aware steps that are identities for the
non-negative variants: `signed_log1p` compresses magnitudes symmetrically before
clustering, and clustering.unresponsive_row_mask judges responsiveness by mean
|value|. Before these fixes (2026-09-25) every negative-W synapse of
`modulation_all_var_weighted` was classified as unresponsive, so its lesion
results described positive-W synapses only.

All three weighted variants are computed from Var(M) by
`weight_modulation_variance`, which weights the VARIANCE afterwards: by W^2 for
`modulation_all_weighted`, by W and |W| for the other two. W is static, so
W^2 Var(M) is Var(W * M) up to floating-point rounding. Until 2026-10-27
multiple_task_analysis.py built `Ms_orig * W` and took its variance, which
held a second copy of the M trajectory (about 32 GiB at 126 time steps).
"""
import numpy as np

# clustering_name -> the post-variance weight it applies (None = none).
MODULATION_VARIANCE_WEIGHTS = {
    "modulation_all": None,
    "modulation_all_weighted": "squared",
    "modulation_all_var_weighted": "signed",
    "modulation_all_abs_weighted": "absolute",
}

# Saved-name order used by the lesion experiment and every per-variant panel.
LESION_MODULATION_TYPES = (
    "modulation_all_normalized",
    "modulation_all_unnormalized",
    "modulation_all_weighted_unnormalized",
    "modulation_all_var_weighted_unnormalized",
    "modulation_all_abs_weighted_unnormalized",
)


def modulation_type_tag(type_key):
    """`modulation_all_var_weighted_unnormalized` -> `var-weighted-unnormalized`."""
    return type_key.replace("modulation_all_", "").replace("_", "-")


# One color per type tag (ColorBrewer Dark2), for the lesion_plot diagnostics.
MODULATION_TYPE_COLORS = {
    "normalized": "#1b9e77",
    "unnormalized": "#d95f02",
    "weighted-unnormalized": "#7570b3",
    "var-weighted-unnormalized": "#e7298a",
    "abs-weighted-unnormalized": "#66a61e",
}


def weight_modulation_variance(cell_vars, clustering_name, modulation_W):
    """Apply a variant's post-variance synapse weighting.

    cell_vars: (n_conditions, n_synapses) variance features, synapses in the
    C-order flattening of `modulation_W` (post, pre). Returns a new array:
    unchanged for `modulation_all`, multiplied by W^2 for
    `modulation_all_weighted` (Var(W * M)), by W for
    `modulation_all_var_weighted`, by |W| for `modulation_all_abs_weighted`.
    """
    if clustering_name not in MODULATION_VARIANCE_WEIGHTS:
        raise ValueError(f"Unknown modulation clustering variant: {clustering_name!r}")
    cell_vars = np.asarray(cell_vars, dtype=float)
    weights = np.asarray(modulation_W, dtype=float).ravel()
    if cell_vars.ndim != 2 or cell_vars.shape[1] != weights.size:
        raise ValueError("cell_vars must be (n_conditions, n_synapses) with one "
                         "column per entry of modulation_W")
    kind = MODULATION_VARIANCE_WEIGHTS[clustering_name]
    if kind is None:
        return cell_vars.copy()
    if kind == "absolute":
        weights = np.abs(weights)
    elif kind == "squared":
        weights = weights ** 2
    return cell_vars * weights[np.newaxis, :]


def signed_log1p(values):
    """sign(x) * log1p(|x|): the log1p range compression used before clustering
    unnormalized features, made symmetric so signed features (W * Var(M)) are
    compressed the same way on both sides of zero. Identical to np.log1p for
    non-negative input, and finite for any finite input."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))
