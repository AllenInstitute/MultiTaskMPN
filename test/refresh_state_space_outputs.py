"""Refresh derived state-space caches and labels without evaluating networks.

Manual, data-dependent check: run from the repository root. New structure and
lesion-link caches are staged, compared against existing numerical results,
and published only after every existing value is unchanged. PCA, distance-
angle and transplant result pickles are inputs and are never rewritten.
"""
import pickle
import shutil
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import paper_plot
import state_space_shift as shift
import state_space_structure as structure
import state_space_transplant as transplant
import state_space_vs_lesion as lesion


def read(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def unchanged(old, new, location):
    if isinstance(old, dict):
        if set(old) - {"m_alignment"} != set(new) - {"m_alignment"}:
            raise AssertionError(f"Changed fields: {location}")
        for key in old:
            if key != "m_alignment":
                unchanged(old[key], new[key], f"{location}/{key}")
    elif isinstance(old, (list, tuple)):
        if len(old) != len(new):
            raise AssertionError(f"Changed length: {location}")
        for index, (left, right) in enumerate(zip(old, new)):
            unchanged(left, right, f"{location}/{index}")
    elif isinstance(old, (np.ndarray, float, np.floating)):
        np.testing.assert_allclose(old, new, rtol=1e-10, atol=1e-12,
                                   equal_nan=True, err_msg=location)
    elif old != new:
        raise AssertionError(f"Changed value: {location}: {old!r} -> {new!r}")


def main():
    root = Path("state_space").resolve()
    old_combined = read(root / structure.COMBINED_RESULT_NAME)
    old_lesion = read(root / lesion.RESULT_NAME)
    with tempfile.TemporaryDirectory(prefix="state_space_refresh_") as tmp:
        staged = Path(tmp)
        for subdir in (Path("."), Path("untrained")):
            (staged / subdir).mkdir(exist_ok=True)
            for path in structure.pca_cache_paths(root / subdir):
                (staged / subdir / path.name).symlink_to(path)
        combined = structure.run_all(
            staged, n_permutations=old_combined["n_permutations"],
            seed=old_combined["permutation_seed"])
        representative = next(iter(old_lesion.values()))
        pools = {entry.get("control_pool", "all") for entry in old_lesion.values()}
        if pools != {"all"}:
            raise ValueError(f"This refresh expects the all control pool, found {pools}")
        linked = lesion.run_all(
            staged, lesion_root=Path("multiple_tasks_norm").resolve(),
            lesion_perf_root=Path("multiple_tasks_perf").resolve(),
            n_permutations=representative["n_permutations"],
            seed=representative["permutation_seed"])
        unchanged(old_combined, combined, "structure combined")
        unchanged(old_lesion, linked, "lesion combined")
        generated = [path for path in staged.rglob("*.pkl") if not path.is_symlink()]
        for path in generated:
            target = root / path.relative_to(staged)
            if target.exists():
                unchanged(read(target), read(path), str(target))
        for path in generated:
            target = root / path.relative_to(staged)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        print(f"Verified unchanged numerical results and refreshed {len(generated)} derived files; "
              f"{len(combined['runs'])} trained, {len(combined['untrained'])} untrained, "
              f"{len(linked)} lesion-linked runs.")

    for directory in (root, root / "untrained"):
        if (directory / shift.RESULTS_NAME).exists():
            shift.summarize(directory)
    diagnostics = 0
    for path in (root / "transplant").rglob("transplant_results*.pkl"):
        for stem, entry in read(path).items():
            figure_stem = transplant._figure_stem(stem, entry["evaluation_seed"])
            transplant._draw_transplant_figure(
                entry["scatter"], entry["rval_dict"], path.parent / f"transplant_{figure_stem}.png")
            diagnostics += 1
    paper_plot.plot_state_space_transplant_r_values()
    print(f"Redrew {diagnostics} transplant diagnostics, the paper transplant figure, "
          "and both distance-angle summaries from existing results.")


if __name__ == "__main__":
    main()
