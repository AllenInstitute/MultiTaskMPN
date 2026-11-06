"""Manual, data-dependent comparison of two task-period variance definitions.

Uses identical fresh production-style trials for both estimators and both
networks; never rewrites clustering caches. This isolates S1 conditioning,
not all differences from Driscoll et al.'s evaluation trial design. Streams
moments to avoid retaining full modulation trajectories. The historical
``*_result.json`` reports used padded endpoints, explicitly recorded in their
metadata; new checks use actual trial endpoints. Example:

    python test/variance_conditioning_check.py --eval-seed 0
"""

import argparse
import copy
import json
from pathlib import Path
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

import helper
import mpn_tasks
from network_loading import build_network
from core.task_variance import trial_epoch_bounds


PHASE_RULES = {
    "stim1": [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "stim2": [6, 7, 8, 9, 10],
    "delay1": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "delay2": [6, 7, 8, 9, 10],
    "go1": list(range(15)),
}
REPRESENTATIONS = {"input": "input1", "hidden": "hidden1", "synapse": "M1"}
METHODS = ("s1_mean", "all_trials", "weighted_s1", "within_s1")


def stimulus_conditioned_variance(h, labels):
    """Historical S1-bin estimator, retained only as a manual comparison."""
    means = np.stack([h[labels == k].mean(axis=0) for k in range(8) if np.any(labels == k)])
    return means.astype(float).var(axis=(0, 1))


class Moments:
    def __init__(self, n_features):
        self.sums = np.zeros((7, n_features), dtype=np.float64)
        self.steps = 0

    def add(self, x, groups, weights):
        # Match the historical estimator's float32 means before float64 moments.
        means = np.stack([x[group].mean(axis=0) for group in groups]).astype(float)
        values = x.astype(float)
        raw_mean = values.mean(axis=0)
        raw_second = np.square(values).mean(axis=0)
        mean_second = np.square(means)
        weighted_second = np.sum(weights[:, None] * mean_second, axis=0)
        self.sums += np.stack([
            means.mean(axis=0), mean_second.mean(axis=0),
            raw_mean, raw_second,
            np.sum(weights[:, None] * means, axis=0), weighted_second,
            raw_second - weighted_second,
        ])
        self.steps += 1

    def finish(self):
        s = self.sums / self.steps
        result = {
            "s1_mean": np.maximum(s[1] - s[0] ** 2, 0),
            "all_trials": np.maximum(s[3] - s[2] ** 2, 0),
            "weighted_s1": np.maximum(s[5] - s[4] ** 2, 0),
            "within_s1": np.maximum(s[6], 0),
        }
        np.testing.assert_allclose(
            result["all_trials"], result["weighted_s1"] + result["within_s1"],
            rtol=2e-4, atol=2e-7,
        )
        return result


@torch.no_grad()
def collect(model, inputs, trials, rule_indices, rules):
    values = {}
    for rule_index, (rule, trial) in enumerate(zip(rules, trials)):
        started = time.monotonic()
        batch = inputs[np.asarray(rule_indices) == rule_index]
        labels = np.asarray(trial.meta["stim1"]).reshape(-1)
        groups = [labels == k for k in range(8) if np.any(labels == k)]
        weights = np.array([group.mean() for group in groups])
        epochs = {
            phase: trial_epoch_bounds(trial, phase)
            for phase, indices in PHASE_RULES.items() if rule_index in indices
        }
        moments = {}
        samples = {name: [] for name in REPRESENTATIONS}
        sample_indices = {}
        model.reset_state(B=len(batch))
        for t in range(max(end for _, end in epochs.values())):
            _, _, states = model.network_step(batch[:, t], run_mode="track_states", seq_idx=t)
            for name, key in REPRESENTATIONS.items():
                x = states[key].cpu().numpy().reshape(len(batch), -1)
                if name not in sample_indices:
                    sample_indices[name] = np.linspace(0, x.shape[1] - 1, 16, dtype=int)
                samples[name].append(x[:, sample_indices[name]].copy())
                for phase, (start, end) in epochs.items():
                    if start <= t < end:
                        if (phase, name) not in moments:
                            moments[(phase, name)] = Moments(x.shape[1])
                        entry = moments[(phase, name)]
                        entry.add(x, groups, weights)
        for (phase, name), entry in moments.items():
            result = entry.finish()
            start, end = epochs[phase]
            sample = np.stack(samples[name], axis=1)[:, start:end]
            # Validate against the actual helper and a direct ungrouped variance.
            np.testing.assert_allclose(
                result["s1_mean"][sample_indices[name]],
                stimulus_conditioned_variance(sample, labels), rtol=2e-4, atol=2e-7)
            np.testing.assert_allclose(
                result["all_trials"][sample_indices[name]],
                helper.task_variance_period_numpy(sample), rtol=2e-5, atol=2e-9)
            values[(phase, rule_index, name)] = result
        print(f"  {rule}: {time.monotonic() - started:.1f}s", flush=True)
    rows = [(phase, rule_index) for phase, indices in PHASE_RULES.items() for rule_index in indices]
    matrices = {
        name: {method: np.stack([values[(phase, idx, name)][method] for phase, idx in rows])
               for method in METHODS}
        for name in REPRESENTATIONS
    }
    return rows, matrices


def normalize(matrix):
    maximum = matrix.max(axis=0)
    return np.divide(matrix, maximum, out=np.zeros_like(matrix), where=maximum > 0)


def summarize(rows, matrices):
    summary = {}
    for name, estimates in matrices.items():
        current, full = (normalize(estimates[key]) for key in ("s1_mean", "all_trials"))
        current_dist, full_dist = pdist(current), pdist(full)
        trees = [linkage(d, method="ward") for d in (current_dist, full_dist)]
        phases = {}
        for phase in PHASE_RULES:
            idx = [i for i, row in enumerate(rows) if row[0] == phase]
            total = estimates["all_trials"][idx].sum()
            # Constant embedded inputs in delays can have zero variance.
            # Ratios there are undefined; do not turn rounding error into a %.
            def fraction(method):
                return float(estimates[method][idx].sum() / total) if total > 1e-12 else None
            phases[phase] = {
                "s1_mean_to_full_total_variance": fraction("s1_mean"),
                "weighted_s1_retained_fraction": fraction("weighted_s1"),
                "within_s1_fraction": fraction("within_s1"),
                "normalized_mean_absolute_difference": float(np.abs(current[idx] - full[idx]).mean()),
            }
        summary[name] = {
            "phase": phases,
            "normalized_matrix_correlation": float(np.corrcoef(current.ravel(), full.ravel())[0, 1]),
            "normalized_mean_absolute_difference": float(np.abs(current - full).mean()),
            "row_distance_correlation": float(np.corrcoef(current_dist, full_dist)[0, 1]),
            # Fixed cuts are a sensitivity diagnostic, not production's selected k.
            "fixed_cut_row_ari": {
                str(k): float(adjusted_rand_score(*[fcluster(z, k, criterion="maxclust") for z in trees]))
                for k in (10, 15, 20)
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-seed", type=int, default=749)
    parser.add_argument("--untrained-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--network", choices=("both", "trained", "untrained"), default="both")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("/tmp/variance_conditioning_check.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    run = f"everything_seed{args.trained_seed}_L21e4+hidden300+batch128+angle"
    cfg = json.loads((root / "multiple_tasks" / f"param_{run}_param.json").read_text())
    task_params, _, _ = mpn_tasks.convert_and_init_multitask_params(
        tuple(copy.deepcopy(cfg[key]) for key in ("task_params", "train_params", "net_params")))
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    task_params["hp"]["seed"] = args.eval_seed
    task_params["hp"]["rng"] = np.random.RandomState(args.eval_seed)
    task_params["hp"]["batch_size_train"] = args.trials
    data, extra = mpn_tasks.generate_trials_wrap(
        task_params, args.trials, rules=task_params["rules"], mode_input="random",
        device="cpu", verbose=False, long_all=False)
    inputs = data[0]
    _, trials, rule_indices = extra
    checkpoint = torch.load(root / "multiple_tasks" / f"savednet_{run}.pt", map_location="cpu", weights_only=False)
    metadata = {
        "checkpoint": run, "eval_seed": args.eval_seed, "trials_per_task": args.trials,
        "untrained_init_seed": args.untrained_seed, "sigma_x": task_params["hp"]["sigma_x"],
        "scope": "Same current random trials; isolates variance estimator, not a paper replication.",
        "none_epoch_end": "Actual task trial end; padding excluded.",
    }
    report = {"metadata": metadata, "networks": {}}
    for label, init_seed in (("trained", None), ("untrained", args.untrained_seed)):
        if args.network not in ("both", label):
            continue
        print(f"Starting {label}", flush=True)
        model, _ = build_network(checkpoint, init_seed)
        model.eval()
        rows, matrices = collect(model, inputs, trials, rule_indices, task_params["rules"])
        report["networks"][label] = summarize(rows, matrices)
        archive = {}
        for name, estimates in matrices.items():
            for method, matrix in estimates.items():
                archive[f"{label}_{name}_{method}"] = matrix
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        archive["rows"] = np.array([f"{task_params['rules'][idx]}-{phase}" for phase, idx in rows])
        np.savez_compressed(args.output.with_name(f"{args.output.stem}_{label}.npz"), **archive)
        del model, matrices, archive, estimates, matrix
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
