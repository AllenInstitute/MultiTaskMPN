"""Read-only manual audit of existing state-space transplant results.

Run directly; intentionally excluded from automated test discovery.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ("hidden", "mod", "eff_mod")


def load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def median_range(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max())}


def check_zero_hidden_responses():
    """Show whether undefined hidden angles also remove valid M pairs."""
    import _bootstrap  # noqa: F401
    import state_space_transplant as transplant
    from test_state_space_transplant import _tiny_network, _trials, RULES, CONTEXT_STEPS

    for scope in ("one_host", "all_hosts"):
        model = _tiny_network()
        inputs, labels = _trials()
        hosts = labels == 0 if scope == "one_host" else np.ones(labels.size, dtype=bool)
        inputs[hosts, CONTEXT_STEPS] = inputs[hosts, CONTEXT_STEPS - 1]
        state, _ = transplant.modulation_after_steps(model, inputs, CONTEXT_STEPS)
        donor = np.roll(np.arange(labels.size), 3)
        native = transplant.branch_step(model, state, inputs[:, CONTEXT_STEPS - 1],
                                         inputs[:, CONTEXT_STEPS])
        other = transplant.branch_step(model, state[donor], inputs[:, CONTEXT_STEPS - 1],
                                        inputs[:, CONTEXT_STEPS])
        finite_mod = int(np.isfinite(transplant.paired_angles(native["mod"], other["mod"])).sum())
        report = {"edge_case": scope, "finite_M_angles_before_filter": finite_mod,
                  "trials": len(labels)}
        try:
            _, scatter, _, _ = transplant.transplant_analysis(
                model, inputs, labels, RULES, CONTEXT_STEPS, 0.9)
            report["saved_pairs"] = {name: len(row["dists"]) for name, row in scatter.items()}
            report["M_permutation_skipped"] = scatter["mod"]["permutation_skipped"]
        except ValueError as exc:
            report["error"] = str(exc)
        print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-cases", action="store_true")
    if parser.parse_args().edge_cases:
        check_zero_hidden_responses()
        return
    directory = ROOT / "state_space" / "transplant"
    grouped = defaultdict(list)
    for path in sorted(directory.rglob("transplant_results*.pkl")):
        results = load(path)
        print(json.dumps({"file": str(path.relative_to(ROOT)), "runs": len(results),
                          "evaluation_seeds": sorted({v["evaluation_seed"] for v in results.values()}),
                          "schemas": sorted({v["schema_version"] for v in results.values()})}))
        if path.name != "transplant_results.pkl":
            continue
        for name, entry in results.items():
            group = "untrained" if entry.get("untrained") else str(entry["l2_info"])
            grouped[group].append((name, entry))
    for group, runs in sorted(grouped.items()):
        entries = [entry for _, entry in runs]
        report = {"group": group, "n": len(entries), "representations": {}}
        for representation in REPRESENTATIONS:
            scatter = [entry["scatter"][representation] for entry in entries]
            r_values = [entry["rval_dict"][representation][0] for entry in entries]
            p_values = [row["permutation"]["p_value"] for row in scatter if row["permutation"]]
            norm_ratios = [entry["step_norms"][representation]["transplant_step"] /
                           entry["step_norms"][representation]["native_step"] for entry in entries]
            report["representations"][representation] = {
                "r": median_range(r_values), "p_below_0.05": sum(p < 0.05 for p in p_values),
                "pairs": sorted({len(row["dists"]) for row in scatter}),
                "skipped_steps": sum(row["n_steps_skipped"] for row in scatter),
                "cross_task_angle": median_range([np.mean(row["angles_deg"]) for row in scatter]),
                "within_task_angle": median_range([row["within_task_angle_deg"] for row in scatter]),
                "transplant_native_step_norm_ratio": median_range(norm_ratios),
            }
        report["clip_native"] = median_range([
            entry["step_norms"]["mod"]["clipped_fraction_native"] for entry in entries])
        report["clip_transplant"] = median_range([
            entry["step_norms"]["mod"]["clipped_fraction_transplant"] for entry in entries])
        for kind in ("native_step", "transplant_step"):
            report["decay_over_" + kind] = median_range([
                entry["step_norms"]["mod"]["decay_term"] / entry["step_norms"]["mod"][kind]
                for entry in entries])
        report["self_check_max"] = max(max(entry["native_check"]["max_abs_hidden_diff"],
                                          entry["native_check"]["max_abs_m_diff"])
                                        for entry in entries)
        print(json.dumps(report))
    state_space = ROOT / "state_space"
    for pattern in ("*results*.pkl", "untrained/*results*.pkl"):
        for path in sorted(state_space.glob(pattern)):
            value = load(path)
            if isinstance(value, dict):
                print(json.dumps({"result": str(path.relative_to(ROOT)),
                                  "schema": value.get("schema_version"),
                                  "m_alignment": value.get("m_alignment"),
                                  "keys": list(value)[:6]}))


if __name__ == "__main__":
    main()
