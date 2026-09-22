"""GPU diagnosis of L2=1e-4 multi-task activation experiments.

Covers all five trained activations. ReLU and linear are the controls that
separate the candidate explanations for the softplus / sigmoid deficit: ReLU is
positive-only with a zero-derivative ("saturated") majority of hidden units yet
trains best, so neither sign nor saturation can be the cause; what singles out
softplus and sigmoid is the non-zero output at zero input, i.e. the constant
background (offset) in the plastic layer's presynaptic embedding.

This analysis loads every matching checkpoint and must be run manually on a
compute node. Its JSON output is consumed by ``paper_plot.py acc_plot``.
"""

import argparse
import copy
import json
import socket
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
import helper
import mpn
import mpn_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "multiple_tasks"
PERFORMANCE_PATH = REPO_ROOT / "multiple_tasks_perf" / "performance_results.json"
# Diagnosis label -> run feature tag. Ordered as the paper's activation figure.
ACTIVATION_FEATURES = {
    "linear": "L21e4linear",
    "relu": "L21e4relu",
    "softplus": "L21e4softplus",
    "sigmoid": "L21e4sigmoid",
    "tanh": "L21e4",
}
REFERENCE_ACTIVATION = "tanh"
METRIC_PATHS = (
    "database_accuracy",
    "evaluation_accuracy",
    "embedding.activation_mean",
    "embedding.activation_std",
    "embedding.derivative_mean",
    "embedding.derivative_fraction_lt_0p1",
    "embedding.offset_energy_fraction",
    "embedding.effective_dimension",
    "hidden.activation_mean",
    "hidden.activation_std",
    "hidden.derivative_mean",
    "hidden.derivative_fraction_lt_0p1",
    "hidden.offset_energy_fraction",
    "hidden.effective_dimension",
    "hidden.response_task_separation_ratio",
    "hebbian.dc_energy_fraction",
    "hebbian.negative_outer_product_fraction",
    "final_modulation.mean",
    "final_modulation.std",
    "final_modulation.common_across_trials_energy_fraction",
    "final_modulation.trial_effective_dimension",
    "final_modulation.upper_bound_fraction",
    "final_modulation.lower_bound_fraction",
)


def _load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _load_model(model_name, device):
    checkpoint_path = MODEL_DIR / f"savednet_{model_name}.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    model = mpn.DeepMultiPlasticNet(
        copy.deepcopy(checkpoint["net_params"]), verbose=False, forzihan=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint_path


def _common_dataset(reference_name, n_per_rule, seed):
    config = _load_json(MODEL_DIR / f"param_{reference_name}_param.json")
    task_params, _, _ = mpn_tasks.convert_and_init_multitask_params((
        copy.deepcopy(config["task_params"]),
        copy.deepcopy(config["train_params"]),
        copy.deepcopy(config["net_params"]),
    ))
    task_params["hp"]["batch_size_train"] = int(n_per_rule)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    data, extra = mpn_tasks.generate_trials_wrap(
        task_params, int(n_per_rule), rules=task_params["rules"],
        mode_input="random_batch", device="cpu", verbose=False)
    task_idx = np.asarray(extra[2], dtype=int).reshape(-1)
    if task_idx.size != data[0].shape[0]:
        task_idx = np.asarray(helper.find_task(
            task_params, data[0].numpy(), 0), dtype=int).reshape(-1)
    return task_params, data, task_idx


def _activation_derivative(name, preactivation):
    preactivation = np.asarray(preactivation, dtype=np.float64)
    if name == "tanh":
        activated = np.tanh(preactivation)
        return 1.0 - activated * activated
    if name == "relu":
        return (preactivation > 0.0).astype(np.float64)
    if name == "linear":
        return np.ones_like(preactivation)
    clipped = np.clip(preactivation, -50.0, 50.0)
    sigmoid = 1.0 / (1.0 + np.exp(-clipped))
    if name == "sigmoid":
        return sigmoid * (1.0 - sigmoid)
    if name == "softplus":
        return sigmoid
    raise ValueError(name)


def _effective_dimension(samples, max_samples=5000):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape[0] > max_samples:
        index = np.linspace(0, samples.shape[0] - 1, max_samples).astype(int)
        samples = samples[index]
    samples -= samples.mean(axis=0, keepdims=True)
    covariance = samples.T @ samples / max(samples.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    denominator = float(np.square(eigenvalues).sum())
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(eigenvalues.sum() ** 2 / denominator)


def _trial_effective_dimension(flat_values):
    values = np.asarray(flat_values, dtype=np.float64)
    values -= values.mean(axis=0, keepdims=True)
    gram = values @ values.T / max(values.shape[1], 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    denominator = float(np.square(eigenvalues).sum())
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(eigenvalues.sum() ** 2 / denominator)


def _activation_summary(name, preactivation, activation):
    preactivation = np.asarray(preactivation, dtype=np.float64).reshape(-1)
    activation = np.asarray(activation, dtype=np.float64).reshape(-1)
    derivative = _activation_derivative(name, preactivation)
    second_moment = float(np.mean(np.square(activation)))
    return {
        "preactivation_mean": float(preactivation.mean()),
        "preactivation_std": float(preactivation.std()),
        "activation_mean": float(activation.mean()),
        "activation_std": float(activation.std()),
        "activation_p01": float(np.quantile(activation, 0.01)),
        "activation_median": float(np.quantile(activation, 0.50)),
        "activation_p99": float(np.quantile(activation, 0.99)),
        "negative_activation_fraction": float(np.mean(activation < 0.0)),
        "derivative_mean": float(derivative.mean()),
        "derivative_median": float(np.median(derivative)),
        "derivative_p10": float(np.quantile(derivative, 0.10)),
        "derivative_fraction_lt_0.05": float(np.mean(derivative < 0.05)),
        "derivative_fraction_lt_0p1": float(np.mean(derivative < 0.10)),
        "offset_energy_fraction": (
            float(activation.mean() ** 2 / second_moment)
            if second_moment > np.finfo(float).eps else 0.0),
    }


def _response_task_separation(response_representation, task_idx):
    values = np.asarray(response_representation, dtype=np.float64)
    labels = np.asarray(task_idx, dtype=int)
    global_mean = values.mean(axis=0)
    between_sum = 0.0
    within_sum = 0.0
    total = 0
    for label in np.unique(labels):
        selected = values[labels == label]
        centroid = selected.mean(axis=0)
        between_sum += selected.shape[0] * float(
            np.mean(np.square(centroid - global_mean)))
        within_sum += float(np.square(selected - centroid).sum()) / values.shape[1]
        total += selected.shape[0]
    between = between_sum / max(total, 1)
    within = within_sum / max(total, 1)
    return float(between / max(within, np.finfo(float).eps))


def _accuracy_by_task(model, outputs, targets, masks, inputs, task_idx, rules):
    accuracies = {}
    for rule_index, rule in enumerate(rules):
        selected = np.flatnonzero(task_idx == rule_index)
        if selected.size == 0:
            accuracies[rule] = None
            continue
        selected_device = torch.as_tensor(selected, device=outputs.device)
        value, _ = model.compute_acc(
            outputs.index_select(0, selected_device),
            targets.index_select(0, selected_device),
            masks.index_select(0, selected_device),
            inputs.index_select(0, selected_device),
            isvalid=True, mode=model.acc_measure)
        accuracies[rule] = float(value)
    return accuracies


def _diagnose_model(model_name, activation_name, database_entry,
                    task_params, data, task_idx, device):
    model, checkpoint_path = _load_model(model_name, device)
    inputs, targets, masks = (tensor.to(device) for tensor in data)
    batch_size, n_steps = inputs.shape[:2]
    outputs = torch.empty(
        batch_size, n_steps, targets.shape[-1], device=device)
    response_sum = torch.zeros(batch_size, model.n_hidden, device=device)
    response_count = torch.zeros(batch_size, 1, device=device)
    embedding_pre_samples = []
    embedding_samples = []
    hidden_pre_samples = []
    hidden_samples = []
    hebbian_dc = []
    hebbian_negative = []

    model.reset_state(B=batch_size)
    with torch.no_grad():
        for seq_idx in range(n_steps):
            current_input = inputs[:, seq_idx]
            embedding_pre = model.W_initial_linear(current_input)
            output, _, db = model.network_step(
                current_input, run_mode="track_states", seq_idx=seq_idx)
            embedding = db["input1"]
            hidden_pre = db["hidden_pre1"]
            hidden = db["hidden1"]
            outputs[:, seq_idx] = output

            embedding_pre_samples.append(embedding_pre.cpu().numpy())
            embedding_samples.append(embedding.cpu().numpy())
            hidden_pre_samples.append(hidden_pre.cpu().numpy())
            hidden_samples.append(hidden.cpu().numpy())

            pre_dc = (embedding.mean(dim=1).square()
                      / embedding.square().mean(dim=1).clamp_min(1e-12))
            post_dc = (hidden.mean(dim=1).square()
                       / hidden.square().mean(dim=1).clamp_min(1e-12))
            hebbian_dc.append(float((pre_dc * post_dc).mean()))
            pre_negative = (embedding < 0).float().mean(dim=1)
            post_negative = (hidden < 0).float().mean(dim=1)
            negative_outer = (
                pre_negative * (1.0 - post_negative)
                + (1.0 - pre_negative) * post_negative)
            hebbian_negative.append(float(negative_outer.mean()))

            response_active = (masks[:, seq_idx, 0] > 0).float().unsqueeze(1)
            response_sum += hidden * response_active
            response_count += response_active

    accuracy, _ = model.compute_acc(
        outputs, targets, masks, inputs, isvalid=True, mode=model.acc_measure)
    per_task = _accuracy_by_task(
        model, outputs, targets, masks, inputs, task_idx, task_params["rules"])
    embedding_pre = np.concatenate(embedding_pre_samples, axis=0)
    embedding = np.concatenate(embedding_samples, axis=0)
    hidden_pre = np.concatenate(hidden_pre_samples, axis=0)
    hidden = np.concatenate(hidden_samples, axis=0)
    response = (response_sum / response_count.clamp_min(1.0)).cpu().numpy()

    final_modulation = model.mp_layers[0].M.detach().cpu().numpy()
    flat_modulation = final_modulation.reshape(batch_size, -1)
    common_energy = float(
        np.square(flat_modulation.mean(axis=0)).mean()
        / max(np.square(flat_modulation).mean(), np.finfo(float).eps))
    upper_bound, lower_bound = model.mp_layers[0].M_bound_vals[1], model.mp_layers[0].M_bound_vals[0]
    modulation_summary = {
        "mean": float(final_modulation.mean()),
        "std": float(final_modulation.std()),
        "mean_absolute": float(np.abs(final_modulation).mean()),
        "common_across_trials_energy_fraction": common_energy,
        "trial_effective_dimension": _trial_effective_dimension(flat_modulation),
        "upper_bound_fraction": float(
            np.mean(final_modulation >= float(upper_bound) - 1e-5)),
        "lower_bound_fraction": float(
            np.mean(final_modulation <= float(lower_bound) + 1e-5)),
    }
    embedding_summary = _activation_summary(
        activation_name, embedding_pre, embedding)
    embedding_summary["effective_dimension"] = _effective_dimension(embedding)
    hidden_summary = _activation_summary(
        activation_name, hidden_pre, hidden)
    hidden_summary["effective_dimension"] = _effective_dimension(hidden)
    hidden_summary["response_task_separation_ratio"] = (
        _response_task_separation(response, task_idx))

    layer = model.mp_layers[0]
    record = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "activation": activation_name,
        "database_accuracy": float(database_entry["acc"]),
        "database_accuracy_per_task": database_entry.get("acc_per_task", {}),
        "evaluation_accuracy": float(accuracy),
        "evaluation_accuracy_per_task": per_task,
        "embedding": embedding_summary,
        "hidden": hidden_summary,
        "hebbian": {
            "dc_energy_fraction": float(np.mean(hebbian_dc)),
            "negative_outer_product_fraction": float(
                np.mean(hebbian_negative)),
        },
        "final_modulation": modulation_summary,
        "parameters": {
            "eta": float(layer.eta.detach().cpu().reshape(-1)[0]),
            "lambda": float(layer.lam.detach().cpu().reshape(-1)[0]),
            "input_weight_frobenius": float(
                torch.linalg.vector_norm(model.W_initial_linear.weight)),
            "plastic_weight_frobenius": float(
                torch.linalg.vector_norm(layer.W)),
            "output_weight_frobenius": float(
                torch.linalg.vector_norm(model.W_output)),
            "plastic_bias_mean": float(layer.b.detach().mean()),
            "plastic_bias_std": float(layer.b.detach().std()),
        },
    }

    del model, inputs, targets, masks, outputs, final_modulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _run_accuracy_mode(model_name, data, device, mode):
    model, _ = _load_model(model_name, device)
    inputs, targets, masks = (tensor.to(device) for tensor in data)
    outputs = torch.empty(
        inputs.shape[0], inputs.shape[1], targets.shape[-1], device=device)
    model.reset_state(B=inputs.shape[0])
    with torch.no_grad():
        for seq_idx in range(inputs.shape[1]):
            output, activities, _ = model.forward(
                inputs[:, seq_idx], run_mode="minimal")
            outputs[:, seq_idx] = output
            if mode == "normal":
                for layer_index, layer in enumerate(model.mp_layers):
                    layer.update_M_matrix(
                        activities[layer_index], activities[layer_index + 1])
            elif mode == "centered_hebb":
                for layer_index, layer in enumerate(model.mp_layers):
                    pre = activities[layer_index]
                    post = activities[layer_index + 1]
                    layer.update_M_matrix(
                        pre - pre.mean(dim=1, keepdim=True),
                        post - post.mean(dim=1, keepdim=True))
            elif mode != "frozen_modulation":
                raise ValueError(mode)
    accuracy, _ = model.compute_acc(
        outputs, targets, masks, inputs, isvalid=True, mode=model.acc_measure)
    value = float(accuracy)
    del model, inputs, targets, masks, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return value


def _nested_value(record, path):
    value = record
    for key in path.split("."):
        value = value[key]
    return float(value)


def _aggregate(records):
    grouped = {}
    for activation_name in ACTIVATION_FEATURES:
        selected = [
            record for record in records
            if record["activation"] == activation_name]
        grouped[activation_name] = {
            "n_models": len(selected),
            "metrics": {
                path: {
                    "mean": float(np.mean([
                        _nested_value(record, path) for record in selected])),
                    "std": float(np.std([
                        _nested_value(record, path) for record in selected])),
                }
                for path in METRIC_PATHS
            },
        }
        tasks = selected[0]["database_accuracy_per_task"]
        grouped[activation_name]["task_accuracy_mean"] = {
            task: float(np.mean([
                record["database_accuracy_per_task"][task]
                for record in selected
                if record["database_accuracy_per_task"].get(task) is not None]))
            for task in tasks
        }

    accuracy = np.asarray([
        record["database_accuracy"] for record in records], dtype=float)
    correlations = {}
    for path in METRIC_PATHS[2:]:
        values = np.asarray([
            _nested_value(record, path) for record in records], dtype=float)
        correlations[path] = (
            float(np.corrcoef(accuracy, values)[0, 1])
            if accuracy.std() > 0 and values.std() > 0 else None)

    tanh_tasks = grouped[REFERENCE_ACTIVATION]["task_accuracy_mean"]
    task_deficits = {}
    for activation_name in ACTIVATION_FEATURES:
        if activation_name == REFERENCE_ACTIVATION:
            continue
        task_deficits[activation_name] = sorted(
            ({
                "task": task,
                "accuracy": grouped[activation_name]["task_accuracy_mean"][task],
                "tanh_accuracy": tanh_tasks[task],
                "deficit_vs_tanh": (
                    grouped[activation_name]["task_accuracy_mean"][task]
                    - tanh_tasks[task]),
            } for task in tanh_tasks),
            key=lambda entry: entry["deficit_vs_tanh"])
    return grouped, correlations, task_deficits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=(REPO_ROOT / "multiple_task"
                 / "activation_function_diagnosis_results.json"))
    parser.add_argument("--trials-per-rule", type=int, default=2)
    parser.add_argument("--intervention-trials-per-rule", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA node")
    device = torch.device("cuda:0")
    performance = _load_json(PERFORMANCE_PATH)
    candidates = []
    for activation_name, feature in ACTIVATION_FEATURES.items():
        for model_name, entry in sorted(performance.items()):
            if entry.get("feature") != feature:
                continue
            checkpoint = MODEL_DIR / f"savednet_{model_name}.pt"
            if checkpoint.exists():
                candidates.append((model_name, activation_name, entry))
    counts = {
        activation_name: sum(
            candidate[1] == activation_name for candidate in candidates)
        for activation_name in ACTIVATION_FEATURES
    }
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"missing activation checkpoints: {counts}")

    reference_name = candidates[0][0]
    task_params, data, task_idx = _common_dataset(
        reference_name, args.trials_per_rule, args.seed)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_name(
        f"{args.output.stem}.partial{args.output.suffix}")
    for index, (model_name, activation_name, entry) in enumerate(candidates, 1):
        print(f"[{index}/{len(candidates)}] {model_name}", flush=True)
        records.append(_diagnose_model(
            model_name, activation_name, entry,
            task_params, data, task_idx, device))
        with partial_output.open("w") as stream:
            json.dump({"records": records}, stream, indent=2)

    grouped, correlations, task_deficits = _aggregate(records)
    representatives = {}
    for activation_name in ACTIVATION_FEATURES:
        selected = [
            record for record in records
            if record["activation"] == activation_name]
        median_accuracy = float(np.median([
            record["database_accuracy"] for record in selected]))
        representatives[activation_name] = min(
            selected,
            key=lambda record: abs(
                record["database_accuracy"] - median_accuracy))["model_name"]

    _, intervention_data, _ = _common_dataset(
        reference_name, args.intervention_trials_per_rule, args.seed + 1)
    interventions = {}
    for activation_name, model_name in representatives.items():
        print(f"[intervention] {activation_name}: {model_name}", flush=True)
        activation_interventions = {"model_name": model_name}
        for mode in ("normal", "frozen_modulation", "centered_hebb"):
            activation_interventions[mode] = _run_accuracy_mode(
                model_name, intervention_data, device, mode)
        interventions[activation_name] = activation_interventions

    result = {
        "environment": {
            "hostname": socket.gethostname(),
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "seed": args.seed,
            "trials_per_rule": args.trials_per_rule,
            "intervention_trials_per_rule": args.intervention_trials_per_rule,
        },
        "model_counts": counts,
        "records": records,
        "group_summary": grouped,
        "metric_accuracy_correlations": correlations,
        "largest_task_deficits_vs_tanh": task_deficits,
        "interventions": interventions,
    }
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2)
    partial_output.unlink(missing_ok=True)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
