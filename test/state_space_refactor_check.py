"""Manual before/after numerical check for the state-space refactor.

Run before and after a refactor with the same snapshot path. Uses tiny fixtures,
never project result caches or trained checkpoints.
"""
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import numpy as np
import state_space_shift as shift
import state_space_structure as structure
import state_space_transplant as transplant
from test_state_space_timecourse import _states
from test_state_space_structure import _pca_cache
from test_state_space_transplant import _tiny_network, _trials, RULES, CONTEXT_STEPS


def snapshot():
    states = _states()
    with patch.object(shift, "DISTANCE_ANGLE_N_PERMUTATIONS", 30), \
            patch.object(shift, "_draw_distance_angle_figure"), patch("builtins.print"):
        r, scatter = shift._distance_angle_for_states(states, Path("."), "fixture", n_steps=5)
    inputs, indices = _trials()
    transplant_result = transplant.transplant_analysis(
        _tiny_network(), inputs, indices, RULES, CONTEXT_STEPS, 0.9)
    structure_result = structure.analyze_run(_pca_cache("fixture", 1e-4, 0), n_permutations=30)
    return {"shift": (r, scatter, states["timecourse"]),
            "transplant": transplant_result, "structure": structure_result}


def compare(expected, actual, path="result"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            compare(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual) and len(expected) == len(actual), path
        for i, value in enumerate(expected):
            compare(value, actual[i], f"{path}/{i}")
    elif isinstance(expected, (np.ndarray, np.number, float, int)):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    else:
        assert expected == actual, (path, expected, actual)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    result = snapshot()
    if path.exists():
        with path.open("rb") as stream:
            compare(pickle.load(stream), result)
        print("All three analyses match the baseline exactly, including five-step results and permutation nulls.")
    else:
        with path.open("wb") as stream:
            pickle.dump(result, stream)
        print(f"Saved baseline to {path}")
