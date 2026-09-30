"""Read lesion caches while retaining compatibility with the historical typo."""

import pickle
from pathlib import Path

import numpy as np


def resolve_lesion_cache_path(path):
    """Prefer a correctly spelled file, falling back to an existing legacy name."""
    path = Path(path)
    if path.exists():
        return path
    legacy = path.with_name(path.name.replace("lesion", "leison"))
    return legacy if legacy.exists() else path


def normalize_lesion_cache(value):
    """Correct metadata in memory without modifying input containers or numbers.

    Numerical arrays are reused, not copied. Colliding canonical/legacy keys
    are rejected instead of silently discarding either record. Only strings
    containing the historical spelling are changed; cache schemas stay intact.
    """
    if isinstance(value, str):
        return value.replace("leison", "lesion")
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            canonical_key = normalize_lesion_cache(key)
            if canonical_key in normalized:
                raise ValueError(f"Conflicting legacy and canonical lesion cache key: {canonical_key!r}")
            normalized[canonical_key] = normalize_lesion_cache(item)
        return normalized
    if isinstance(value, list):
        return [normalize_lesion_cache(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_lesion_cache(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "U":
            return np.char.replace(value, "leison", "lesion")
        if value.dtype.kind == "O":
            normalized = np.empty_like(value)
            for index in np.ndindex(value.shape):
                normalized[index] = normalize_lesion_cache(value[index])
            return normalized
    return value


def load_lesion_pickle(path):
    """Load a trusted local pickle with legacy filename/key support, read-only."""
    with resolve_lesion_cache_path(path).open("rb") as stream:
        return normalize_lesion_cache(pickle.load(stream))