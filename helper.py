import mpn_tasks
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 
from scipy.linalg import null_space
import time

from typing import Dict, List, Sequence, Iterable, Iterator, TypeVar
from itertools import permutations, chain, islice
import random
import math

import torch

T = TypeVar("T")

def all_leq(seq, limit):
    """
    check if all elements in the list is smaller or equal to certain value
    """
    return all(x <= limit for x in seq)
    
def concat_random_samples(
    d: Dict[float, Sequence[T]], n_samples: int, seed: int
) -> List[List[T]]:
    """
    Return `n_samples` random concatenations by shuffling the keys each time.
    Sampling is *with replacement* over the space of permutations.
    """
    rng = random.Random(seed)
    keys = list(d.keys())
    out: List[List[T]] = []
    for _ in range(n_samples):
        rng.shuffle(keys)
        out.append(list(chain.from_iterable(d[k] for k in keys)))
    return out
    
def is_power_of_4_or_zero(n: int) -> bool:
    """
    Returns True if n is 0 **or** an exact power of 4 (1, 4, 16, 64, …).
    """
    return (
        n == 0
        or (
            n > 0
            and (n & (n - 1)) == 0      # power-of-2 check
            and (n & 0x55555555) != 0    # even-bit position  ➜ power-of-4
            )
        )

def basic_sort(lst, sort_idxs):
    """
    Map each element in `lst` to its corresponding entry in `sort_idxs`.
    """
    return [int(sort_idxs[i]) for i in lst]
    
def permutation_indices_b_to_a(a, b):
    """
    Return indices p such that [b[i] for i in p] == a (no duplicates).
    """
    pos = {v: i for i, v in enumerate(b)}
    return [pos[v] for v in a]
    
def as_jsonable(obj):
    """
    Convert *obj* into something json-serialisable.

    Handles:
    • np.ndarray      → nested Python list
    • np.integer/float→ plain int/float
    • list/tuple      → element-wise conversion (keeps same type as list)
    • dict            → value-wise conversion (keys left untouched)
    Falls back to the original object if it is already JSON-friendly,
    otherwise raises TypeError so json can try its own default.
    """
    # --- NumPy types --------------------------------------------------
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # --- Container types ----------------------------------------------
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(el) for el in obj]         # keep outer list
    if isinstance(obj, dict):
        return {k: as_jsonable(v) for k, v in obj.items()}

    # --- Anything JSON already understands (str, int, float, bool, None) ----
    # Leave it unchanged; json.dumps can handle it.
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # --- Unknown type: let json default handler decide -----------------
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
    
        
def to_ndarray(x):
    """
    Return *x* as a NumPy ndarray.
    
    • If x is a PyTorch tensor, move it to CPU (if needed),
      detach it from the autograd graph, and convert to ndarray.
    • If x is already an ndarray, return it untouched.
    • Otherwise fall back to np.asarray for scalars, lists, etc.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)
    
def sample_upper_means(mat, k=8, n_iter=1000, seed=None):
    """
    """
    rng = np.random.default_rng(seed)

    # 1. Pull out only the finite (non-NaN) entries once.
    valid_vals = mat[~np.isnan(mat)].ravel()

    # 2. Sanity check.
    if valid_vals.size < k:
        raise ValueError(f"Need at least {k} finite values, got {valid_vals.size}")

    # 3. Repeat: sample k numbers → take their mean.
    means = np.array([
        rng.choice(valid_vals, size=k, replace=False).mean()
        for _ in range(n_iter)
    ])

    return [np.mean(means), np.std(means)]

def find_zero_chunks(array):
    """
    Find consecutive rows of all zeros in an N*3 array.
    
    Args:
        array (numpy.ndarray): Input N*3 array.
    
    Returns:
        list: List of [start_index, end_index] for each chunk of consecutive zero rows.
    """
    zero_rows = np.all(array == 0, axis=1)  # Identify rows that are all zero
    chunks = []
    start = None
    
    for i, is_zero in enumerate(zero_rows):
        if is_zero and start is None: 
            start = i
        elif not is_zero and start is not None: 
            chunks.append([start, i - 1])
            start = None
    
    if start is not None:
        chunks.append([start, len(zero_rows) - 1])
    
    return chunks

def find_zero_chunks_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x : (T, C) tensor
    returns an (n_chunks, 2) LongTensor with [start, end] (inclusive) indices,
    ordered from earliest to latest.  If no zero rows, returns empty tensor.
    """
    zero_rows = (x == 0).all(dim=1)                     # (T,) bool
    if not torch.any(zero_rows):
        return torch.empty(0, 2, dtype=torch.long, device=x.device)

    # prepend/append False so diff catches edges
    padded = torch.cat([zero_rows.new_zeros(1), zero_rows, zero_rows.new_zeros(1)])
    diff   = padded[1:].int() - padded[:-1].int()        # +1 at starts, −1 at ends-1
    starts = (diff ==  1).nonzero(as_tuple=False).squeeze(1)
    ends   = (diff == -1).nonzero(as_tuple=False).squeeze(1) - 1
    return torch.stack([starts, ends], dim=1)            # (n_chunks, 2)

def build_altered_mask(output_mask: torch.Tensor,
                       cut_proportion: float = 0.25) -> torch.Tensor:
    """
    output_mask : (B, T, C)  *non-binary* mask with task-period scaling
    returns     : (B, T, C)  binary mask limited to late-response period
    """
    B, T, C = output_mask.shape
    device  = output_mask.device
    # out_bin = torch.zeros_like(output_mask, dtype=torch.int8, device=device)
    out_bin = torch.full_like(output_mask, float('nan'))

    for b in range(B):
        # --- locate zero chunks along time for this trial ----------------------
        chunks = find_zero_chunks_torch(output_mask[b])
        if chunks.shape[0] < 3:                          # delay tasks etc.
            # append dummy “padding” chunk at the end
            chunks = torch.cat([chunks,
                                torch.as_tensor([[T, T]], device=device)])

        # response period = chunk −2 (zeros) → chunk −1 (zeros)
        response_start = chunks[-2, 1] + 1
        response_end   = chunks[-1, 0]
        dur            = response_end - response_start
        response_start = response_start + int(dur * cut_proportion)

        # --- slice, binarise, write back --------------------------------------
        seg = output_mask[b, response_start:response_end] > 0
        out_bin[b, response_start:response_end] = seg.int()

    return out_bin

def generate_rainbow_colors(length):
    """
    Generate a list of colors transitioning from red to purple in a rainbow-like gradient.

    Args:
        length (int): Number of colors to generate.

    Returns:
        list: List of RGB color strings.
    """
    if length <= 0:
        raise ValueError("Length must be greater than 0.")

    # Generate evenly spaced values for hue (red to purple in HSV space)
    hues = np.linspace(0, 0.8, length)  # Hue from 0 (red) to ~0.8 (purple)
    colors = [mcolors.hsv_to_rgb((hue, 1, 1)) for hue in hues]

    # Convert RGB values to hex for easier visualization
    hex_colors = [mcolors.to_hex(color) for color in colors]

    return hex_colors

def to_unit_vector(arr):
    """Convert a vector to a unit vector."""
    norm = np.linalg.norm(arr)
    return arr / norm if norm != 0 else arr

def participation_ratio_vector(C):
    """Computes the participation ratio of a vector of variances."""
    return np.sum(C) ** 2 / np.sum(C*C)

def plot_some_outputs(params, net, mode_for_all="random_batch", n_outputs=4, nameadd="", verbose=False):
    """
    """
    # in case out of range
    c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',] * 10
    c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',] * 10
    c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',] * 10

    task_params, train_params, net_params = params

    if task_params['task_type'] in ('multitask',):
        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(task_params, n_outputs, mode_input=mode_for_all)
        # figure out sessions
        # assert mode_for_all == "random_batch" # lazy...
        _, test_trials, test_rule_idxs = test_trials_extra
        recordkyle_all = []
        for test_subtrial in test_trials:
            metaepoch = test_subtrial.epochs
            # print(metaepoch)
            periodname = list(metaepoch.keys())
            recordkyle = []
            for keyiter in range(len(periodname)):
                recordkyle.append(list(metaepoch[periodname[keyiter]][1]))
            recordkyle.insert(0, [0 for _ in range(len(recordkyle[1]))])
            recordkyle = np.array(recordkyle).T.tolist()
            # print(recordkyle)
            recordkyle_all.extend(recordkyle)
                    
    elif task_params['task_type'] in ('NeuroGym',):
        test_data, _ = convert_ngym_dataset(
                    task_params, set_size=n_outputs, device=device,
                    mask_type=tasks_masks[task_params['dataset_name']]
            )

    test_input, test_output, test_masks = test_data

    net_out, db = net.iterate_sequence_batch(test_input, run_mode='track_states')

    net_out = net_out.detach().cpu().numpy()
    test_input_np = test_input.cpu().numpy()
    test_out_np = test_output.cpu().numpy()

    test_masks_np = test_masks.cpu().numpy()

    fig, axs = plt.subplots(n_outputs, 2, figsize=(8*2, 2*n_outputs))

    if task_params['task_type'] in ('NeuroGym',):
        for batch_idx, ax in enumerate(axs[:,0]):
            ax.plot(net_out[batch_idx, :, 0], color=c_vals[batch_idx])
            ax.plot(test_out_np[batch_idx, :, 0], color=c_vals_l[batch_idx], zorder=-1)

            ax.set_ylim((0, 2.25))
    elif task_params['task_type'] in ('multitask',):

        masks = (test_masks_np > 0.).astype(np.int32) # Hides output where not relevant

        for batch_idx, ax in enumerate(axs[:,0]):
            for out_idx in range(test_out_np.shape[-1]):
                ax.plot(net_out[batch_idx, :, out_idx] * masks[batch_idx, :, out_idx], color=c_vals[out_idx])
                ax.plot(test_out_np[batch_idx, :, out_idx], color=c_vals_l[out_idx], zorder=-1)

            ax.set_ylim((-1.125, 1.125))

        # ADD BY ZIHAN
        for batch_idx, ax in enumerate(axs[:,1]):
            for out_idx in range(test_input_np.shape[-1]):
                ax.plot(test_input_np[batch_idx, :, out_idx], color=c_vals_l[out_idx], zorder=-1, label=out_idx)

            ax.set_ylim((-1.125, 1.125))
            ax.legend()

        # ZIHAN: add session cutoff
        for axind in range(2):
            for outputind in range(n_outputs):
                session_period = recordkyle_all[outputind]
                for session_break in session_period:
                    axs[outputind, axind].axvline(session_break, linestyle="--")

    if verbose:
        fig.savefig(f"./results/results_{nameadd}.png")

def find_consecutive_zero_indices(arr):
    zero_indices = []
    start_index = None

    for i, row in enumerate(arr):
        if np.all(row == 0):
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                zero_indices.append(list(range(start_index, i)))
                start_index = None

    # Handle case where the last rows are all zeros
    if start_index is not None:
        zero_indices.append(list(range(start_index, len(arr))))

    return zero_indices

def magnitude_of_projection(v, basis_vectors):
    """
    Calculate the magnitude of the projection of vector v onto the subspace spanned by basis_vectors.
    """
    v = np.array(v).flatten() 
    basis_vectors = np.array(basis_vectors)

    if basis_vectors.shape[0] == 1:
        # 1D subspace
        u = basis_vectors[0]
        norm_u = np.linalg.norm(u)
        projection_scalar = np.dot(v, u) / norm_u**2
        projection_vector = projection_scalar * u
        magnitude = np.linalg.norm(projection_vector)
    elif basis_vectors.shape[0] == 2:
        # 2D subspace
        U = basis_vectors.T  # (N, 2)
        P = U @ np.linalg.inv(U.T @ U) @ U.T  # Projection matrix
        projection_vector = P @ v
        magnitude = np.linalg.norm(projection_vector)
    else:
        raise ValueError("Only 1D or 2D subspaces are supported.")
    
    return magnitude

if __name__ == "__main__":
    vector1 = np.random.rand(1, 100)
    vector2 = np.random.rand(1, 100)

    vector1_normalized = vector1 / np.linalg.norm(vector1)
    vector2_normalized = vector2 / np.linalg.norm(vector2)
    vectors = np.vstack((vector1_normalized, vector2_normalized))

    alpha = np.random.rand()
    beta = np.random.rand()
    vector_in_plane = alpha * vector1_normalized + beta * vector2_normalized
    vector_in_plane_normalized = vector_in_plane / np.linalg.norm(vector_in_plane)

    magnitude = magnitude_of_projection(vector_in_plane_normalized, vectors)
    print(magnitude)
