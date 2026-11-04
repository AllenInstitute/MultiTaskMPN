"""Read-only summaries for interpreting the existing state-space figures."""
from collections import defaultdict
from pathlib import Path
import json
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "state_space"


def read(name):
    with (ROOT / name).open("rb") as stream:
        return pickle.load(stream)


def describe(rows, statistic, p_key="p_value"):
    values = np.array([row[statistic] for row in rows], dtype=float)
    return {"n": len(rows), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()),
            "significant": sum(row[p_key] < 0.05 for row in rows)}


def main():
    structure = read("state_space_structure_results.pkl")
    groups = defaultdict(list)
    for entry in structure["runs"].values():
        groups[str(entry["l2_info"])].append(entry)
    groups["untrained"] = list(structure["untrained"].values())
    for group, entries in sorted(groups.items()):
        for representation in ("hidden", "mod", "eff_mod"):
            rows = [entry["representations"][representation] for entry in entries]
            report = {"kind": "category", "group": group, "representation": representation,
                      **describe([row["separation"] for row in rows], "score")}
            print(json.dumps(report))
            factors = rows[0]["factor_vectors"]
            print(json.dumps({"kind": "factor_vectors", "group": group,
                              "representation": representation,
                              "factors": {key: describe([row["factor_vectors"][key] for row in rows],
                                                       "mean_cosine") for key in factors},
                              "pooled": describe([row["pooled"] for row in rows], "mean_cosine")}))
    for group, cohort in sorted(structure["cohorts"].items()):
        entry = cohort["representations"]["eff_mod"]
        print(json.dumps({"kind": "mean_RDM", "group": group, "n": cohort["n_runs"],
                          "S": entry["separation"]["score"],
                          "p": entry["separation"]["p_value"],
                          "rho": {key: entry["consistency"][key]
                                  for key in ("median", "min", "max")}}))
    for directory in ("", "untrained/"):
        groups = defaultdict(list)
        for entry in read(directory + "initial_condition_distance_vs_angle_results.pkl").values():
            group = "untrained" if directory else str(entry["l2_info"])
            groups[group].append(entry)
        for group, entries in sorted(groups.items()):
            for representation in ("hidden", "mod", "eff_mod", "mod_hebb", "eff_mod_hebb"):
                rows = [{"r": entry["rval_dict"][representation][0],
                         "p_value": entry["scatter"][representation]["permutation"]["p_value"]}
                        for entry in entries]
                print(json.dumps({"kind": "natural_distance_angle", "group": group,
                                  "representation": representation, **describe(rows, "r")}))
    groups = defaultdict(list)
    for entry in read("state_space_vs_lesion_results.pkl").values():
        groups[str(entry["l2_info"])].append(entry)
    for group, entries in sorted(groups.items()):
        for representation in ("hidden", "eff_mod"):
            for cluster in ("neuron", "synapse"):
                rows = [entry["tests"][representation][cluster] for entry in entries]
                print(json.dumps({"kind": "distance_vs_lesion", "group": group,
                                  "representation": representation, "cluster": cluster,
                                  **describe(rows, "rho")}))


if __name__ == "__main__":
    main()
