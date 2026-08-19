#!/usr/bin/env python
# coding: utf-8
"""
Structural half of the MICrONS dataset: who is wired to whom.

The counterpart of `activity_helper.py` — not what the neurons did, but what they
are connected to. Same two feather tables `fitting_helper.py` reads (it names them
under `./microns_v1dd/sven/`; here they sit directly in this directory):

    microns_cell_annos_*.feather     one row per reconstructed cell — `pt_root_id`,
                                     `cell_type`, `region`, `layer`, position, and
                                     the proofreading columns `status_axon` /
                                     `full_dendrite`
    synapses_minnie65_*.feather      191 M rows, one per synapse — `pre_pt_root_id`,
                                     `post_pt_root_id`, `size` and positions

`l2it_connectivity` is the entry point: it turns those into a square, directed
connectivity matrix over the proofread cells of one type. No plotting lives here;
`biology.py` at the repository root imports this module by putting `biology/` on
`sys.path` (the `_bootstrap.py` idiom the experiment directories use for `core/`).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

# The tables sit beside this file; see `activity_helper.DATA_DIR` for why they are
# located from `__file__` rather than from the working directory.
DATA_DIR = Path(__file__).resolve().parent

CELL_TABLE_FILE = "microns_cell_annos_CV_250904.feather"
SYNAPSE_TABLE_FILE = ("synapses_minnie65_phase3_v1_1507_combined_filtered"
                      "_incl_trafo_250904.feather")

# Layer 2 intratelencephalic cells, the `cell_type` label used by the annotations.
L2IT = "L2IT"

# What counts as proofread, as the pair of columns the annotations use.
# "both" is the default and the only one under which a SQUARE matrix means what it
# looks like: an entry (i, j) is a claim about i's axon and j's dendrite, so unless
# every neuron has both reconstructed, the matrix mixes measured zeros with zeros
# that only say "we did not trace that far". The L2IT numbers show what that costs:
# 205 cells have both and connect at 3.6% density, while the 3921 with a complete
# dendrite but a mostly untraced axon come out at 0.30% — a twelvefold "result"
# that is entirely an artifact of which axons were followed.
_PROOFREAD_MASKS = {
    "both": lambda t: (t["status_axon"] == "extended") & (t["full_dendrite"]),
    "axon": lambda t: t["status_axon"] == "extended",
    "dendrite": lambda t: t["full_dendrite"] == True,      # noqa: E712 (pandas)
    "either": lambda t: (t["status_axon"] == "extended") | (t["full_dendrite"]),
}

# Public list of the above, for callers building a CLI or validating an argument.
PROOFREAD_MODES = tuple(sorted(_PROOFREAD_MASKS))


# ─── Table loading ────────────────────────────────────────────────────────────

def _table_path(filename, data_dir):
    """Path of one EM table, checked to exist."""
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. The EM tables live in "
                                f"{DATA_DIR} and are not tracked by git.")
    return path


def load_proofread_cells(cell_type=L2IT, proofread="both", region=None,
                         celltype_column="cell_type", data_dir=DATA_DIR):
    """The rows of the cell table for one proofread cell type, sorted by id.

    `proofread` selects the reconstruction requirement (see `_PROOFREAD_MASKS`);
    `region` optionally restricts to one visual area, as `cell_synapse_table_reader`
    does for microns with "V1" — left off by default, so the set is every proofread
    cell of the type wherever it sits. `celltype_column` picks the annotation column
    ("cell_type", or "cell_type_v2" for the layer-split labels).

    Sorted by `pt_root_id` so the matrix's row/column order depends on the cell set
    alone, not on the order rows happen to sit in the feather. The returned frame
    carries every annotation column, so a caller can reorder by depth or subset
    further before building the matrix."""
    if proofread not in _PROOFREAD_MASKS:
        raise ValueError(f"proofread must be one of {list(PROOFREAD_MODES)}; "
                         f"got {proofread!r}")
    path = _table_path(CELL_TABLE_FILE, data_dir)
    table = pd.read_feather(path)
    if celltype_column not in table.columns:
        raise KeyError(f"{celltype_column!r} is not a column of {path.name}; "
                       f"have e.g. {[c for c in table.columns if 'type' in c]}")
    selected = table[(table[celltype_column] == cell_type)
                     & _PROOFREAD_MASKS[proofread](table)]
    if region is not None:
        selected = selected[selected["region"] == region]
    if selected.empty:
        raise ValueError(f"no {proofread}-proofread {cell_type} cells"
                         + (f" in {region}" if region else ""))
    return selected.sort_values("pt_root_id").reset_index(drop=True)


def load_synapses_between(pt_root_ids, data_dir=DATA_DIR):
    """Every synapse whose pre AND post cell are both in `pt_root_ids`.

    The synapse table is 191 M rows over 10 columns; the three columns
    `create_connectivity_as_whole` needs are still ~4.6 GB if read whole. So the
    id filter is pushed down into the feather read (pyarrow streams the record
    batches and keeps only matching rows), which turns a multi-GB load into a few
    thousand rows in a few seconds. The result is exactly what a full read followed
    by the same filter would give."""
    import pyarrow.dataset as pads

    path = _table_path(SYNAPSE_TABLE_FILE, data_dir)
    ids = np.asarray(pt_root_ids)
    dataset = pads.dataset(path, format="feather")
    table = dataset.to_table(
        columns=["pre_pt_root_id", "post_pt_root_id", "size"],
        filter=(pads.field("pre_pt_root_id").isin(ids)
                & pads.field("post_pt_root_id").isin(ids)))
    return table.to_pandas()


# ─── Building the matrices ────────────────────────────────────────────────────

def create_connectivity_as_whole(
        cell_table: pd.DataFrame,
        synapse_table: pd.DataFrame,
        record_syn_values: bool = False):
    """
    Build pair-wise connectivity matrices.

    Parameters
    ----------
    cell_table : DataFrame
        Must contain column 'pt_root_id'.
    synapse_table : DataFrame
        Must contain columns 'pre_pt_root_id', 'post_pt_root_id', 'size'.
    record_syn_values : bool, default False
        If True, also return a 2-D list `synValues[i][j]` with every
        individual 'size' value for the connection i → j.
        Skipping this saves substantial RAM / runtime.

    Returns (always five objects, to keep the signature stable)
    -------
    W          : (N, N) int8   - binary adjacency (1 = ≥1 synapse)
    totalSyn   : (N, N) int64  - sum of 'size' per pair
    synCount   : (N, N) int32  - number of synapses per pair
    synValues  : list[list[list]] | None
                 If `record_syn_values` is False, this is None.
                 Else synValues[i][j] is a Python list of sizes.
    id_to_index: dict          - pt_root_id → row/col index
    """
    start_time = time.time()

    all_neuron_ids = cell_table["pt_root_id"].unique()
    num_neuron = len(all_neuron_ids)

    W = np.zeros((num_neuron, num_neuron), dtype=np.int8)
    totalSyn = np.zeros((num_neuron, num_neuron), dtype=np.int64)
    synCount = np.zeros((num_neuron, num_neuron), dtype=np.int32)
    synValues = None if not record_syn_values else \
        [[[] for _ in range(num_neuron)] for _ in range(num_neuron)]

    id_to_index = {nid: idx for idx, nid in enumerate(all_neuron_ids)}

    # filter synapses whose pre & post neurons exist in cell_table
    mask = (
        synapse_table['pre_pt_root_id'].isin(id_to_index) &
        synapse_table['post_pt_root_id'].isin(id_to_index)
    )
    syn_filtered = synapse_table.loc[mask].copy()

    # map to indices
    syn_filtered["pre_index"] = syn_filtered["pre_pt_root_id"].map(id_to_index)
    syn_filtered["post_index"] = syn_filtered["post_pt_root_id"].map(id_to_index)

    # choose aggregation recipe depending on whether we need the lists
    if record_syn_values:
        grouped = (
            syn_filtered
            .groupby(["pre_index", "post_index"])
            .agg(
                syn_sum=("size", "sum"),
                syn_count=("size", "size"),
                syn_values=("size", list)
            )
            .reset_index()
        )
    else:
        grouped = (
            syn_filtered
            .groupby(["pre_index", "post_index"])
            .agg(
                syn_sum=("size", "sum"),
                syn_count=("size", "size")
            )
            .reset_index()
        )

    # populate outputs
    for row in grouped.itertuples(index=False):
        i, j = row.pre_index, row.post_index
        W[i, j] = 1
        totalSyn[i, j] = row.syn_sum
        synCount[i, j] = row.syn_count
        if record_syn_values:
            synValues[i][j] = row.syn_values

    # remove self-loops
    np.fill_diagonal(W,        0)
    np.fill_diagonal(totalSyn, 0)
    np.fill_diagonal(synCount, 0)
    if record_syn_values:
        for k in range(num_neuron):
            synValues[k][k] = []

    if record_syn_values:
        synValues_ndarray = np.empty((num_neuron, num_neuron), dtype=object)
        for i in range(num_neuron):
            for j in range(num_neuron):
                synValues_ndarray[i, j] = synValues[i][j]
        synValues = synValues_ndarray
    else:  # dummy placeholder
        synValues = np.empty((num_neuron, num_neuron), dtype=object)

    end_time = time.time()
    print(f"Connectivity matrices created in {end_time - start_time:.2f} seconds.",
          flush=True)

    return W, totalSyn, synCount, synValues, id_to_index


def l2it_connectivity(cell_type=L2IT, proofread="both", region=None,
                      celltype_column="cell_type", record_syn_values=False,
                      data_dir=DATA_DIR, verbose=True):
    """Square connectivity matrix over the proofread L2IT cells.

    Selects the cells (`load_proofread_cells`), pulls only the synapses internal to
    that set (`load_synapses_between`) and hands both to
    `create_connectivity_as_whole`. Row and column i are the same neuron, so the
    matrix is directed and square: `W[i, j] == 1` means i's axon synapses onto j's
    dendrite, and `W` is *not* symmetric.

    With `proofread="both"` (the default) every cell has a reconstructed axon and a
    complete dendrite, which is what makes a zero informative: it is "these two
    fully traced neurons do not touch", not "we stopped tracing". Self-loops are
    zeroed by the builder, so the diagonal is empty by construction rather than by
    measurement. `cell_type` is a parameter, so the same call builds the matrix for
    any annotated type; L2IT is only the default.

    Returns a dict — six things that must travel together, and positional returns
    would make the call sites unreadable:
        W          (N, N) int8   binary adjacency
        total_syn  (N, N) int64  summed synapse size per pair
        syn_count  (N, N) int32  number of synapses per pair
        syn_values (N, N) object per-pair list of sizes (empty unless requested)
        ids        (N,)   int64  pt_root_id of each row/column, in order
        cells      DataFrame     the selected rows of the cell table, same order
    """
    cells = load_proofread_cells(cell_type=cell_type, proofread=proofread,
                                 region=region, celltype_column=celltype_column,
                                 data_dir=data_dir)
    ids = cells["pt_root_id"].to_numpy()
    if verbose:
        print(f"{len(ids)} {proofread}-proofread {cell_type} cells"
              + (f" in {region}" if region else "") + "; reading synapses...",
              flush=True)
    synapses = load_synapses_between(ids, data_dir=data_dir)
    if verbose:
        print(f"{len(synapses)} synapses internal to the set.", flush=True)

    W, total_syn, syn_count, syn_values, id_to_index = create_connectivity_as_whole(
        cells, synapses, record_syn_values=record_syn_values)
    # `create_connectivity_as_whole` indexes by first appearance in `cell_table`,
    # which for a de-duplicated, id-sorted frame is exactly `ids` — assert it rather
    # than trust it, since every downstream row/column label depends on it.
    assert all(id_to_index[i] == k for k, i in enumerate(ids)), \
        "row order does not match the cell table order"
    return {"W": W, "total_syn": total_syn, "syn_count": syn_count,
            "syn_values": syn_values, "ids": ids, "cells": cells}


def connectivity_summary(result):
    """One-line-per-fact summary of an `l2it_connectivity` result."""
    W, syn_count = result["W"], result["syn_count"]
    n = W.shape[0]
    edges = int(W.sum())
    possible = n * (n - 1)                      # no self-loops
    reciprocal = int((W & W.T).sum() // 2)
    multi = int((syn_count > 1).sum())
    return "\n".join([
        f"  neurons          {n}",
        f"  connections      {edges} / {possible} possible "
        f"({100 * edges / max(possible, 1):.2f}% density)",
        f"  reciprocal pairs {reciprocal}",
        f"  multi-synapse    {multi} of {edges} connections",
        f"  synapses         {int(syn_count.sum())}",
    ])
