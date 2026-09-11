"""Export PCA bases from trusted local fixed-point caches, without model solves.

Backfill existing one-task, RNN, and two-task caches from the repository root:
    python core/fixed_point_pca.py onetask onetask_rnn twotasks

Paths may also name individual fixed_points_grad_* or fixed_points_hidden_*
pickles. Compact .pca.npz sidecars leave the original pickles unchanged.
Task-interpolation alpha figures use the reference rule's exported basis;
their interpolation arrays are not fitted separately here.
"""

import argparse
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np


SCHEMA_VERSION = 1
PERIODS = ("longfixation", "longstimulus", "longdelay", "longresponse")
REPRESENTATIONS = ("fixed_M", "fixed_WM", "fixed_hidden")


def sidecar_path(source_path):
    return Path(source_path).with_suffix(".pca.npz")


def source_signature(source_path):
    path = Path(source_path)
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fit_fixed_point_bases(data):
    """Fit each available canonical period's basis using the original paper settings.

    All saved points participate, including nonconverged points, as in the old
    paper renderer. Period, representation, and reference rule remain distinct.
    fit_projection retains PCA.fit_transform coordinates for dense-angle panels.
    """
    import sklearn
    from sklearn.decomposition import PCA

    bases = {}
    results = data.get("results", {})
    for period in PERIODS:
        if period not in results:
            continue
        for representation in REPRESENTATIONS:
            values = results[period].get(representation)
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            if values.ndim not in (2, 3):
                raise ValueError(f"{period}/{representation}: expected per-point feature arrays, got {values.shape}")
            flat = values.reshape(values.shape[0], -1)
            if min(flat.shape) < 2:
                print(f"Skipped PCA {period}/{representation}: fewer than two samples or features")
                continue
            pca = PCA(n_components=2, svd_solver="auto", random_state=0)
            projected = pca.fit_transform(flat)
            bases.setdefault(period, {})[representation] = {
                "mean": pca.mean_, "components": pca.components_,
                "explained_variance": pca.explained_variance_,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "singular_values": pca.singular_values_,
                "fit_projection": projected,
                "source_aname": data.get("aname"), "source_rule": data.get("rule"),
                "source_period": period, "representation": representation,
                "source_shape": values.shape, "n_samples": len(values),
                "n_components": 2, "random_state": 0, "svd_solver": "auto",
                "sklearn_version": sklearn.__version__, "fit_dtype": str(flat.dtype),
                "sample_selection": "all_saved_points",
            }
    if not bases:
        raise ValueError("No canonical fixed-point period has a supported PCA representation")
    return {"schema_version": SCHEMA_VERSION, "bases": bases}


def export_fixed_point_pca(source_path, *, force=False):
    """Create or refresh a compact sidecar; never modify or rerun the source solve."""
    source_path = Path(source_path)
    signature = source_signature(source_path)
    destination = sidecar_path(source_path)
    if destination.exists() and not force:
        try:
            with np.load(destination, allow_pickle=True) as saved:
                artifact = saved["artifact"].item()
            if artifact.get("schema_version") == SCHEMA_VERSION and artifact.get("source") == signature:
                print(f"Up to date: {destination}")
                return destination
        except (OSError, ValueError, KeyError):
            pass
    with source_path.open("rb") as handle:
        data = pickle.load(handle)
    artifact = fit_fixed_point_bases(data)
    artifact["source"] = signature
    if source_signature(source_path) != signature:
        raise RuntimeError(f"Source changed during PCA export: {source_path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, artifact=np.array(artifact, dtype=object))
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved PCA bases: {destination}")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Fixed-point pickles or experiment directories")
    parser.add_argument("--force", action="store_true", help="Refit even when sidecars match the source")
    args = parser.parse_args(argv)
    sources = set()
    for path in args.paths:
        if path.is_file():
            sources.add(path)
        elif path.is_dir():
            for pattern in ("fixed_points_grad_*.pkl", "fixed_points_hidden_*.pkl"):
                sources.update(path.rglob(pattern))
        else:
            parser.error(f"Path does not exist: {path}")
    if not sources:
        parser.error("No fixed-point caches found")
    failures = []
    for path in sorted(sources):
        try:
            export_fixed_point_pca(path, force=args.force)
        except Exception as error:
            failures.append(f"{path}: {error}")
            print(f"FAILED {path}: {error}")
    if failures:
        raise SystemExit("PCA export failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()