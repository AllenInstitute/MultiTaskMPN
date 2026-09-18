#!/usr/bin/env python
# coding: utf-8
"""Draw a directed subnetwork of fully proofread MICrONS BC neurons.

This is intentionally the only analysis in this file. It reuses the connectome
loader in ``biology/connectivity_helper.py`` and restricts both ends of every
displayed connection to basket cells (BC) whose axon and dendrite are both
proofread. A deterministic connected core keeps the network legible while every
node and edge shown still comes from the measured connectome.

Run on the compute node with the project environment::

    conda activate mpn
    python biology.py

The default outputs are ``cartoon_plot/biology_connectivity_network.png`` and
``cartoon_plot/biology_connectivity_network_legend.png``.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

_HELPERS = Path(__file__).resolve().parent / "biology"
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))

import connectivity_helper as conn  # noqa: E402 (requires path shim above)


OUT_DIR = Path("cartoon_plot")
CELL_TYPE = "BC"
PROOFREAD = "both"  # status_axon == "extended" AND full_dendrite == True
DEFAULT_N_NODES = 48
DEFAULT_MIN_SYNAPSES = 1
DEFAULT_LAYOUT_SEED = 7

_NODE_COLOR = "#8f8f8f"
_EDGE_COLOR = "#777777"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save_connectivity_legend(out_dir):
    """Save the neuron/connectivity key as a standalone transparent figure."""
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=_NODE_COLOR, markeredgecolor="white",
               markeredgewidth=0.25, label="Neuron"),
        Line2D([], [], color=_EDGE_COLOR, linewidth=1.5,
               label="Connectivity"),
    ]
    fig = plt.figure(figsize=(2.2, 1.2))
    fig.legend(handles=handles, loc="center", ncol=1, frameon=False,
               fontsize=11, handlelength=1.6, handletextpad=0.65,
               labelspacing=0.5)
    legend_path = Path(out_dir) / "biology_connectivity_network_legend.png"
    fig.savefig(legend_path, dpi=600, bbox_inches="tight",
                pad_inches=0.02, transparent=True)
    plt.close(fig)
    return legend_path


def _connectivity_core_indices(W, syn_count, n_nodes, min_synapses=1):
    """Select one deterministic, connected and edge-rich induced subgraph.

    Selection starts at the cell with largest directed degree and repeatedly adds
    the remaining cell with the most observed links to the current set. Synapse
    count into the set, whole-graph degree and stable source index break ties.
    This defines the subset before plotting and avoids searching many subsets for
    the visually most attractive result.
    """
    W = np.asarray(W, dtype=bool)
    syn_count = np.asarray(syn_count)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or syn_count.shape != W.shape:
        raise ValueError("W and syn_count must be equal-size square matrices")

    n_total = W.shape[0]
    n_nodes = int(n_nodes)
    min_synapses = int(min_synapses)
    if not 3 <= n_nodes <= n_total:
        raise ValueError(f"n_nodes must be between 3 and {n_total}; got {n_nodes}")
    if min_synapses < 1:
        raise ValueError("min_synapses must be >= 1")

    directed = W & (syn_count >= min_synapses)
    np.fill_diagonal(directed, False)
    undirected = directed | directed.T
    degree = directed.sum(axis=0) + directed.sum(axis=1)
    strength = ((syn_count * directed).sum(axis=0)
                + (syn_count * directed).sum(axis=1))
    seed = int(np.lexsort((-np.arange(n_total), strength, degree))[-1])

    selected = [seed]
    available = np.ones(n_total, dtype=bool)
    available[seed] = False
    while len(selected) < n_nodes:
        candidates = np.flatnonzero(available)
        current = np.asarray(selected, dtype=int)
        links = undirected[np.ix_(candidates, current)].sum(axis=1)
        if int(links.max()) == 0:
            raise ValueError(
                f"the connected BC component contains only {len(selected)} cells, "
                f"fewer than n_nodes={n_nodes}; request fewer nodes or lower "
                "min_synapses")
        internal_strength = (
            syn_count[np.ix_(candidates, current)].sum(axis=1)
            + syn_count[np.ix_(current, candidates)].sum(axis=0))
        winner = np.lexsort((-candidates, degree[candidates],
                             internal_strength, links))[-1]
        chosen = int(candidates[winner])
        selected.append(chosen)
        available[chosen] = False
    return np.asarray(selected, dtype=int)


def _spring_layout(adjacency, seed=DEFAULT_LAYOUT_SEED, iterations=450):
    """Deterministic Fruchterman-Reingold layout without a graph dependency."""
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    if adjacency.shape != (n, n):
        raise ValueError("adjacency must be square")

    rng = np.random.RandomState(int(seed))
    theta = 2 * np.pi * np.arange(n) / max(n, 1)
    pos = np.column_stack((np.cos(theta), np.sin(theta)))
    pos += rng.normal(scale=0.08, size=pos.shape)
    edge_i, edge_j = np.where(np.triu(adjacency | adjacency.T, 1))
    k = 1.35 / np.sqrt(max(n, 1))

    for step in range(int(iterations)):
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(distance, np.inf)
        displacement = ((k * k / np.maximum(distance, 1e-6) ** 2)[..., None]
                        * delta).sum(axis=1)
        for i, j in zip(edge_i, edge_j):
            d = pos[i] - pos[j]
            r = max(float(np.linalg.norm(d)), 1e-6)
            pull = d * (r / k)
            displacement[i] -= pull
            displacement[j] += pull
        length = np.linalg.norm(displacement, axis=1)
        temperature = 0.18 * (1.0 - (step + 1) / max(int(iterations), 1))
        step_size = np.minimum(length, temperature)
        pos += (displacement / np.maximum(length[:, None], 1e-12)
                * step_size[:, None])
        pos -= pos.mean(axis=0, keepdims=True)

    # Fill a square plotting region independently along x and y. The layout has
    # no anatomical metric, so this affine rescaling changes no encoded quantity;
    # it only avoids a long/flat graph with wasted whitespace.
    span = np.maximum(np.ptp(pos, axis=0), 1e-12)
    pos /= span[None, :]
    return pos


def plot_bc_connectivity_network(out_dir=OUT_DIR, n_nodes=DEFAULT_N_NODES,
                                 min_synapses=DEFAULT_MIN_SYNAPSES,
                                 layout_seed=DEFAULT_LAYOUT_SEED,
                                 data_dir=conn.DATA_DIR, result=None):
    """Plot measured BC→BC connections among fully proofread MICrONS cells.

    ``result`` may be a previously loaded ``conn.l2it_connectivity`` dictionary.
    Otherwise the function loads only cells annotated ``BC`` with proofread mode
    ``both`` and synapses whose pre- and postsynaptic root IDs both belong to that
    set. Arrow direction is presynaptic→postsynaptic. Width and opacity increase
    with the number of synapses supporting that directed connection.
    """
    if result is None:
        result = conn.l2it_connectivity(
            cell_type=CELL_TYPE, proofread=PROOFREAD, data_dir=data_dir)

    full_W = np.asarray(result["W"], dtype=bool)
    full_count = np.asarray(result["syn_count"], dtype=float)
    keep = _connectivity_core_indices(
        full_W, full_count, n_nodes, min_synapses=min_synapses)
    W = full_W[np.ix_(keep, keep)]
    syn_count = full_count[np.ix_(keep, keep)]
    visible = W & (syn_count >= int(min_synapses))
    np.fill_diagonal(visible, False)
    positions = _spring_layout(visible, seed=layout_seed)

    # Matplotlib scatter `s` is marker AREA. Map total directed degree linearly
    # into a deliberately visible area range: a node's degree counts every
    # incoming plus every outgoing connection within the displayed BC core.
    total_degree = visible.sum(axis=0) + visible.sum(axis=1)
    degree_span = int(np.ptp(total_degree))
    if degree_span > 0:
        degree_fraction = ((total_degree - total_degree.min()) / degree_span)
        node_sizes = 24.0 + 66.0 * degree_fraction
    else:
        node_sizes = np.full(total_degree.shape, 48.0)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    edge_scale = np.log1p(syn_count[visible])
    edge_scale /= max(float(edge_scale.max()), 1.0)
    for edge_k, (i, j) in enumerate(zip(*np.where(visible))):
        scale = float(edge_scale[edge_k])
        rad = (0.08 if visible[j, i] and i < j else
               -0.08 if visible[j, i] else 0.0)
        ax.add_patch(FancyArrowPatch(
            positions[i], positions[j], arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}", color=_EDGE_COLOR,
            linewidth=0.30 + 0.75 * scale, alpha=0.18 + 0.38 * scale,
            mutation_scale=5.5, shrinkA=4.8, shrinkB=6.0, zorder=1))

    ax.scatter(positions[:, 0], positions[:, 1],
               s=node_sizes, color=_NODE_COLOR,
               edgecolor="white", linewidth=0.25, zorder=3)
    limit = 1.06 * max(float(np.abs(positions).max()), 0.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "biology_connectivity_network.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    legend_path = _save_connectivity_legend(out_dir)

    edges = int(visible.sum())
    print(f"Saved: {out_path}  ({len(keep)} of {full_W.shape[0]} fully "
          f"proofread BC neurons; {edges} directed BC→BC connections with >= "
          f"{min_synapses} synapse(s); layout seed {layout_seed})")
    print(f"Saved: {legend_path}")
    return {
        "path": out_path,
        "legend_path": legend_path,
        "indices": keep,
        "root_ids": np.asarray(result["ids"])[keep],
        "positions": positions,
        "W": visible,
        "syn_count": syn_count,
        "total_degree": total_degree,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"output directory (default: {OUT_DIR})")
    parser.add_argument("--data-dir", default=str(conn.DATA_DIR),
                        help="directory containing the MICrONS cell and synapse "
                             "feather tables")
    parser.add_argument("--n-nodes", type=int, default=DEFAULT_N_NODES,
                        help=f"BC neurons in the connected core "
                             f"(default: {DEFAULT_N_NODES})")
    parser.add_argument("--min-synapses", type=int,
                        default=DEFAULT_MIN_SYNAPSES,
                        help="minimum synapses required to display an edge "
                             f"(default: {DEFAULT_MIN_SYNAPSES})")
    parser.add_argument("--layout-seed", type=int, default=DEFAULT_LAYOUT_SEED,
                        help=f"deterministic layout seed "
                             f"(default: {DEFAULT_LAYOUT_SEED})")
    args = parser.parse_args()

    plot_bc_connectivity_network(
        out_dir=args.out_dir, n_nodes=args.n_nodes,
        min_synapses=args.min_synapses, layout_seed=args.layout_seed,
        data_dir=args.data_dir)


if __name__ == "__main__":
    main()
