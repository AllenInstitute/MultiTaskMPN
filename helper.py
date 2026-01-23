import mpn_tasks
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 
from scipy.linalg import null_space
import time
from scipy.stats import linregress, t

from typing import Dict, Sequence, TypeVar, List, Tuple
from itertools import chain
import random
import math

import torch

T = TypeVar("T")

def find_key_by_membership(d, value):
    """
    Return the key whose array/list contains `value`, else None.
    """
    for k, arr in d.items():
        if value in arr:         
            return k
    return None

def linear_regression(x1, y1, log=True, through_origin=False):
    """
    Fit a (log-)linear model using only pairs where both transformed
    values are finite (not NaN/±∞).

    Returns:
        x_fit, y_fit, r_value, slope, intercept, p_value
    """
    x = np.asarray(x1, dtype=float)
    y = np.asarray(y1, dtype=float)

    # Apply log first (may introduce –∞/NaN for 0 or negative values)
    if log:
        x = np.log10(x)
        y = np.log10(y)

    # Keep only points where *both* transformed values are finite
    good = np.isfinite(x) & np.isfinite(y)
    if not np.any(good):
        raise ValueError("No finite data points remain after transform.")

    x, y = x[good], y[good]

    if x.size < 2:
        raise ValueError("Need at least two valid points for regression.")

    # correlation coefficient (same as linregress.rvalue)
    r_value = np.corrcoef(x, y)[0, 1]

    if through_origin:
        # ----- Linear regression through the origin -----
        Sxx = np.dot(x, x)
        if Sxx == 0.0:
            raise ValueError("All x are zero; cannot fit through-origin regression.")

        slope = np.dot(x, y) / Sxx
        intercept = 0.0

        # residuals and variance estimate
        y_hat = slope * x
        resid = y - y_hat
        n = len(x)
        if n <= 1:
            raise ValueError("Not enough data points for p-value computation.")

        # unbiased estimate of residual variance
        s2 = np.dot(resid, resid) / (n - 1)
        se_slope = np.sqrt(s2 / Sxx)

        # t-test for slope != 0
        t_stat = slope / se_slope
        p_value = 2 * t.sf(np.abs(t_stat), df=n - 1)

    else:
        # ----- Standard linear regression with intercept -----
        lr = linregress(x, y)
        slope      = lr.slope
        intercept  = lr.intercept
        r_value    = lr.rvalue
        p_value    = lr.pvalue
        # lr.stderr is the std error of the slope if you need it

    # build fitted line for plotting in transformed space
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept

    return x_fit, y_fit, r_value, slope, intercept, p_value

def all_leq(seq, limit):
    """
    check if all elements in the list is smaller or equal to certain value
    """
    return all(x <= limit for x in seq)
    
def concat_random_samples(
    d: Dict[float, Sequence[T]], n_samples: int, seed: int
) -> List[Tuple[List[T], List[List[T]]]]:
    """
    Return `n_samples` random concatenations by shuffling the keys each time.
    For each sample, also return the ordered list-of-lists used to build it.

    Sampling is *with replacement* over the space of permutations.

    Returns
    -------
    samples : list of (combined, parts)
        combined : List[T]         # the concatenated list for this permutation
        parts    : List[List[T]]   # the component lists in the shuffled order
    """
    rng = random.Random(seed)
    keys = list(d.keys())
    out: List[Tuple[List[T], List[List[T]]]] = []

    for _ in range(n_samples):
        rng.shuffle(keys)
        parts = [list(d[k]) for k in keys]              
        combined = list(chain.from_iterable(parts))     
        out.append((combined, parts))

    return out
    
def is_power_of_n_or_zero(x: int, n: int) -> bool:
    """
    Returns True if x is 0 or an exact power of n (n^0, n^1, n^2, ...),
    with n >= 2.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    if x == 0:
        return True

    if x < 0:
        return False

    while x % n == 0:
        x //= n

    return x == 1


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
