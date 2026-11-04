"""Read-only manual review of state-space caches; excluded from test discovery.

Run with the project's Python environment. This does not regenerate or edit
analysis outputs. The PCA diagnostic concerns between-task centroid variance,
not the total trial variance that IncrementalPCA optimizes.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(values.size), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


def main():
    groups = defaultdict(list)
    for folder, untrained in ((ROOT / "state_space", False),
                              (ROOT / "state_space" / "untrained", True)):
        results = read_pickle(folder / "initial_condition_distance_vs_angle_results.pkl")
        for name, entry in results.items():
            if entry.get("m_alignment") != "post_update":
                raise ValueError(f"Unaligned cache: {name}")
            group = "untrained" if untrained else f"L2={entry['l2_info']:g}"
            groups[group].append(entry)
    for group, entries in groups.items():
        for representation in ("hidden", "mod", "eff_mod", "mod_hebb", "eff_mod_hebb"):
            values = [entry["rval_dict"][representation][0] for entry in entries]
            print(json.dumps({"kind": "distance_angle", "group": group,
                              "representation": representation, **summary(values)}), flush=True)

    selected = set()
    for path in (ROOT / "paper_plot").glob("*state_space_centroid_scores*.json"):
        data = json.loads(path.read_text())
        selected.add(data["best_aname"])
        print(json.dumps({"kind": "selection", "file": path.name,
                          "best_aname": data["best_aname"], "n_scored": data["n_scored"],
                          "scores": summary([row["score"] for row in data["rankings"]])}), flush=True)

    diagnostics = []
    for folder in (ROOT / "state_space", ROOT / "state_space" / "untrained"):
        for path in sorted(folder.glob("state_space_pca_*_noise0.01.pkl")):
            data = read_pickle(path)
            if data.get("m_alignment") != "post_update":
                raise ValueError(f"Unaligned PCA cache: {path}")
            for representation, record in data["pca_results"].items():
                centers = np.asarray(record["task_centroids"], dtype=np.float64)
                weights = np.asarray(record["task_trial_counts"], dtype=np.float64)
                weights /= weights.sum()
                centered = (centers - weights @ centers) * np.sqrt(weights[:, None])
                eigenvalues = np.linalg.eigvalsh(centered @ centered.T)
                labels = np.asarray(record["ctx_rule_labels"])
                coordinates = np.asarray(record["X_2d"], dtype=np.float64)
                projected = np.stack([coordinates[labels == i].mean(axis=0)
                                      for i in range(len(centers))])
                projected -= weights @ projected
                captured = float(np.sum(weights[:, None] * projected ** 2))
                total = float(eigenvalues.sum())
                row = {"kind": "centroid_projection", "aname": data["aname"],
                       "untrained": bool(data.get("untrained")),
                       "init_seed": data.get("init_seed"),
                       "representation": representation,
                       "saved_capture": captured / total,
                       "optimal_centroid_capture": float(eigenvalues[-2:].sum() / total),
                       "fraction_of_centroid_optimum": captured / float(eigenvalues[-2:].sum())}
                diagnostics.append(row)
                if data["aname"] in selected and not data.get("untrained"):
                    print(json.dumps(row), flush=True)
    for representation in ("hidden", "mod", "eff_mod"):
        values = [row["fraction_of_centroid_optimum"] for row in diagnostics
                  if row["representation"] == representation and not row["untrained"]]
        print(json.dumps({"kind": "centroid_projection_summary", "representation": representation,
                          **summary(values)}), flush=True)


if __name__ == "__main__":
    main()
