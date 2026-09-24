"""Standalone delay-memory geometry analysis for sibling task families.

The supported families are DelayDM1/DelayDM2 and DMCGo/DMCNoGo inside a trained
``everything`` multi-task network. This module loads the run, generates aligned
sibling trials, and fits shared delay-trajectory PCA bases. ``--method`` then
selects either gradient fixed-point solving or very-long-delay settling
endpoints. Gradient analysis uses IncrementalPCA, while long-delay analysis
selects among randomized-PCA candidates with family-specific endpoint metrics;
their method-specific bases are stored separately. All outputs are isolated
under ``two_in_multiples/{aname}/``.

Run one or both families without rerunning clustering or lesion analysis::

    python multiple_task/sibling_delay_analysis.py \
    --seed 749 --feature L21e4 --families delaydm1 --method gradient

Select ``--method long_delay_endpoint`` instead to use the final state of a
generated very-long delay as the fixed-point proxy and save its sampled path.
"""
import copy
import gc
import json
import pickle
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
import torch
from torch.serialization import add_safe_globals

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper
import mpn
import mpn_tasks
from grad_fixed_points import solve_period_modulation_fixed_points
from sibling_geometry import best_task_specific_pc_pair

__all__ = [
    "SIBLING_FAMILIES",
    "SIBLING_ANALYSIS_DIR",
    "DELAY_PCA_SCOPES",
    "SIBLING_METHODS",
    "SIBLING_FIXED_POINT_N_SEEDS",
    "SIBLING_FIXED_POINT_STEPS",
    "SIBLING_NORMAL_GPU_BATCH_SIZE",
    "SIBLING_NORMAL_TRIALS_PER_RULE",
    "SIBLING_LONG_DELAY_TRAJECTORY_SAMPLES",
    "build_aname",
    "extract_long_delay_endpoints",
    "is_sibling_artifact",
    "run_sibling_analysis",
    "save_long_delay_endpoint_pc_projections",
    "save_sibling_fixed_point_pc_projections",
    "stream_normal_delay_analysis",
]

SIBLING_FAMILIES = {
    "dmcgo": ("dmcgo", "dmcnogo"),
    "delaydm1": ("delaydm1", "delaydm2"),
}

# Keep sibling-task analysis independent from the general multi-task clustering
# database. paper_plot.py uses the same root when loading two_in_multiple data.
SIBLING_ANALYSIS_DIR = Path("two_in_multiples")
SIBLING_METHODS = ("gradient", "long_delay_endpoint")

# Match the task generator's native eight stimulus directions. This sibling
# analysis deliberately does not solve additional between-direction inputs.
SIBLING_FP_N_STIM = 8

# Both DelayDM rules probe whether different in-distribution stimulus magnitudes
# at the same angle relax to the same or different delay fixed points.
DELAYDM_FP_STIM_MAGNITUDES = (0.6, 0.8, 1.0, 1.2, 1.4)

DELAY_PCA_SCOPES = ("joint", "first_task_only")
# Deterministic task-template seeds tried independently for each sibling rule.
# core/grad_fixed_points.py saves only the seed with the lowest median rel_step.
SIBLING_FIXED_POINT_N_SEEDS = 1
# Maximum Adam steps for each candidate seed. The optimizer can stop earlier
# when its fixed-point speed loss reaches the configured tolerance.
SIBLING_FIXED_POINT_STEPS = 200000
# Normal-delay trajectories used to fit the shared PCA bases. Trials stay on
# CPU and only this many are transferred to the GPU at once.
SIBLING_NORMAL_TRIALS_PER_RULE = 100
SIBLING_NORMAL_GPU_BATCH_SIZE = 200
# Balanced long-delay trials for the settling-endpoint proxy. Each sibling rule
# receives four amplitude/coherence realizations at each of eight directions.
# Aligned RNG state makes every local trial a matched cross-task condition.
SIBLING_LONG_DELAY_TRIALS_PER_DIRECTION = 4
SIBLING_LONG_DELAY_TRIALS_PER_RULE = (
    SIBLING_FP_N_STIM * SIBLING_LONG_DELAY_TRIALS_PER_DIRECTION)
SIBLING_LONG_DELAY_TRAJECTORY_SAMPLES = 51
# Long-delay analysis uses conventional sklearn PCA with randomized SVD so
# reproducible seeds can be compared by the family's task-specific metric.
SIBLING_LONG_DELAY_PCA_SEEDS = tuple(range(10))

# Match the multi-task plotting palette in the input/output sanity figure.
_TRACE_COLORS = [
    "#e53e3e", "#3182ce", "#38a169", "#d69e2e", "#d53f8c",
    "#4c51bf", "#dd6b20", "#0ea5e9", "#22c55e", "#a855f7",
    "#f43f5e", "#0f766e", "#b83280", "#ca8a04", "#2b6cb0",
] * 10


def build_aname(seed, feature):
    """Return the established identifier for an ``everything`` network run."""
    return f"everything_seed{int(seed)}_{feature}+hidden300+batch128+angle"


def _validate_families(families):
    families = tuple(families)
    if not families:
        raise ValueError("select at least one sibling family")
    unknown = set(families) - set(SIBLING_FAMILIES)
    if unknown:
        raise ValueError(f"unknown sibling families: {sorted(unknown)}")
    return families


def is_sibling_artifact(path, families=None):
    """Whether a filename belongs to one of the sibling-analysis families."""
    selected = (tuple(SIBLING_FAMILIES) if families is None
                else _validate_families(families))
    tokens = {
        rule
        for family in selected
        for rule in SIBLING_FAMILIES[family]
    }
    return any(token in Path(path).name for token in tokens)


def _artifact_method(path):
    """Classify a sibling artifact as method-specific or shared preprocessing."""
    path = Path(path)
    name = path.name
    if "long_delay_endpoint" in name:
        return "long_delay_endpoint"
    if "_delay_trajectory_pca" in name:
        # Before PCA filenames became method-specific, randomized long-delay
        # candidates used the gradient filename. Inspect that legacy artifact
        # so method-scoped cleanup can still assign it to the correct owner.
        try:
            with path.open("rb") as stream:
                artifact = pickle.load(stream)
        except (OSError, EOFError, pickle.PickleError, TypeError, ValueError):
            return "gradient"
        if artifact.get("pca_strategy") == "randomized_candidates":
            return "long_delay_endpoint"
        return "gradient"
    if (name.startswith("fixed_points_grad_")
            or "_delay_pc_projections" in name
            or "_delay_pc_gallery" in name):
        return "gradient"
    return None


def _clean_stale_sibling_artifacts(save_dir, families, method=None):
    """Remove selected-family outputs owned by one method plus shared files.

    ``method=None`` retains the legacy behavior of removing every artifact for
    the selected families. A method-specific run preserves results produced by
    the other fixed-point method.
    """
    save_dir = Path(save_dir)
    families = _validate_families(families)
    if method is not None and method not in SIBLING_METHODS:
        raise ValueError(f"unknown sibling method {method!r}; choose from "
                         f"{SIBLING_METHODS}")
    stale = sorted(
        path for path in save_dir.iterdir()
        if (path.is_file()
            and is_sibling_artifact(path, families)
            and (method is None or _artifact_method(path) in (None, method)))
    )
    for path in stale:
        path.unlink()

    if stale:
        print(f"  [cleanup] removed {len(stale)} stale sibling-analysis "
              f"artifact(s) from {save_dir}:")
        for path in stale:
            print(f"    - {path.name}")
    else:
        print(f"  [cleanup] no stale sibling-analysis artifacts in {save_dir}")
    return stale


def _run_sibling_family(*, aname, save_name, save_dir, family, method,
                        model, device,
                        task_params, train_params, net_params,
                        converted_task_params, W_fp):
    """Analyze one pair of sibling tasks using a loaded multi-task network."""
    rules = list(SIBLING_FAMILIES[family])
    task_params_family = copy.deepcopy(converted_task_params)
    task_params_family["rules"] = rules
    task_params_family["hp"]["batch_size_train"] = SIBLING_NORMAL_TRIALS_PER_RULE
    task_params_family["long_delay"] = "normal"

    # Resetting the shared RNG for each rule aligns their epoch boundaries. A
    # single delay window can then be used safely for both tasks' PCA samples.
    norm_data, norm_extra = mpn_tasks.generate_trials_wrap(
        task_params_family, SIBLING_NORMAL_TRIALS_PER_RULE,
        rules=rules, mode_input="random",
        device="cpu", verbose=True, align_periods=True)
    norm_input, norm_output, norm_mask = norm_data
    norm_task = helper.find_task(
        task_params_family, norm_input.detach().cpu().numpy(), 0)
    norm_task = [int(t - min(norm_task)) for t in norm_task]

    norm_task_arr = np.asarray(norm_task)
    _, norm_trials, _ = norm_extra
    acc_all, per_task_acc, norm_out_preview = stream_normal_delay_analysis(
        aname, save_dir, family, rules, model, device,
        norm_input, norm_output, norm_mask, norm_trials, norm_task_arr, W_fp,
        layer_index=1, gpu_batch_size=SIBLING_NORMAL_GPU_BATCH_SIZE,
        n_components=6,
        pca_strategy=("randomized_candidates"
                      if method == "long_delay_endpoint" else "incremental"))
    print(f"  [acc] {family} (normal delay, both tasks): {acc_all:.3f}")
    for task_i, task_name in enumerate(rules):
        n_task = int(np.sum(norm_task_arr == task_i))
        print(f"  [acc] {task_name} (normal delay): "
              f"{per_task_acc[task_name]:.3f}  (n={n_task})")

    acc_str = ", ".join(
        f"{name} acc={per_task_acc[name]:.3f}"
        for name in rules if name in per_task_acc)
    fig, axs = plt.subplots(5, 2, figsize=(10, 10))
    for trial_idx in range(5):
        for inp in range(norm_input.shape[2]):
            axs[trial_idx, 0].plot(
                norm_input[trial_idx, :, inp].detach().cpu().numpy(),
                color=_TRACE_COLORS[inp], alpha=0.5)
        for outp in range(norm_out_preview.shape[2]):
            axs[trial_idx, 1].plot(
                norm_out_preview[trial_idx, :, outp].numpy(),
                color=_TRACE_COLORS[outp], alpha=0.5)
        for outp in range(norm_output.shape[2]):
            axs[trial_idx, 1].plot(
                norm_output[trial_idx, :, outp].detach().cpu().numpy(),
                color=_TRACE_COLORS[outp], alpha=0.5, linestyle="--")
    fig.suptitle(f"{family} (normal delay)  |  {acc_str}", fontsize=12)
    fig.tight_layout()
    io_path = Path(save_dir) / f"{family}_{save_name}.png"
    fig.savefig(io_path, dpi=300)
    plt.close(fig)
    print(f"  Saved sibling IO figure: {io_path}")

    del norm_extra, norm_trials, norm_out_preview
    del norm_data, norm_input, norm_output, norm_mask

    if method == "long_delay_endpoint":
        endpoint_path = extract_long_delay_endpoints(
            aname, save_dir, family, rules, model, device,
            converted_task_params, W_fp, layer_index=1)
        for basis_scope in DELAY_PCA_SCOPES:
            save_long_delay_endpoint_pc_projections(
                aname, save_dir, family, rules, endpoint_path,
                basis_scope=basis_scope)
        return

    if method != "gradient":
        raise ValueError(f"unknown sibling method {method!r}; choose from "
                         f"{SIBLING_METHODS}")

    cfg_fp = {
        "task_params": task_params,
        "train_params": train_params,
        "net_params": net_params,
    }
    solved_rules = []
    failed_rules = []
    for rule in rules:
        try:
            solve_period_modulation_fixed_points(
                aname, Path(save_dir), model, cfg_fp, device,
                rule=rule, out_suffix=f"_{rule}", layer_index=1, W=W_fp,
                periods=("longdelay",), n_interp=SIBLING_FP_N_STIM,
                stim_magnitudes=(DELAYDM_FP_STIM_MAGNITUDES
                                 if rule in ("delaydm1", "delaydm2") else None),
                n_seeds=SIBLING_FIXED_POINT_N_SEEDS,
                steps=SIBLING_FIXED_POINT_STEPS,
                save_all_trajectories=False,
                analyze_stability=False,
                cross_seed_probes=False,
                naive_seed_probes=False,
                traj_seed_probes=False)
            solved_rules.append(rule)
        except Exception as exc:
            failed_rules.append(rule)
            print(f"  [grad-fp/{rule}] failed: {exc}")
            import traceback
            traceback.print_exc()

    if len(solved_rules) == len(rules):
        for basis_scope in DELAY_PCA_SCOPES:
            save_sibling_fixed_point_pc_projections(
                aname, save_dir, family, solved_rules,
                basis_scope=basis_scope)
    else:
        raise RuntimeError(
            f"{family}: fixed-point solve failed for {failed_rules}; "
            "PC projection was not generated")


def run_sibling_analysis(seed, feature, families, method):
    """Run selected sibling analyses without clustering or lesion experiments."""
    families = _validate_families(families)
    if method not in SIBLING_METHODS:
        raise ValueError(f"unknown sibling method {method!r}; choose from "
                         f"{SIBLING_METHODS}")
    aname = build_aname(seed, feature)
    result_path = Path("multiple_tasks") / f"param_{aname}_result.npz"
    param_path = Path("multiple_tasks") / f"param_{aname}_param.json"
    checkpoint_path = Path("multiple_tasks") / f"savednet_{aname}.pt"

    with np.load(result_path, allow_pickle=True) as data:
        hyp_dict = data["hyp_dict"].item()
    with param_path.open() as stream:
        raw_cfg = json.load(stream)
    task_params = raw_cfg["task_params"]
    train_params = raw_cfg["train_params"]
    net_params = raw_cfg["net_params"]

    save_name = f"{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}"
    save_dir = SIBLING_ANALYSIS_DIR / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    _clean_stale_sibling_artifacts(save_dir, families, method=method)

    add_safe_globals([np.core.multiarray._reconstruct])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model = mpn.DeepMultiPlasticNet(
        checkpoint["net_params"], verbose=False, forzihan=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print("missing:", missing)
    print("unexpected:", unexpected)
    del checkpoint

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on: {device}")
    converted_task_params, _, _ = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params))
    W_fp = state_dict["mp_layer1.W"].detach().cpu().numpy()

    failures = []
    for family in families:
        rules = SIBLING_FAMILIES[family]
        print(f"\n{'=' * 70}\n[sibling] family {family} "
              f"({' + '.join(rules)}) | method={method}\n{'=' * 70}", flush=True)
        try:
            _run_sibling_family(
                aname=aname, save_dir=save_dir, family=family, method=method,
                save_name=save_name,
                model=model, device=device, task_params=task_params,
                train_params=train_params, net_params=net_params,
                converted_task_params=converted_task_params, W_fp=W_fp)
        except Exception as exc:
            failures.append((family, exc))
            print(f"  [{family}] family failed: {exc}")
            import traceback
            traceback.print_exc()

    del model, state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if failures:
        details = "; ".join(f"{family}: {exc}" for family, exc in failures)
        raise RuntimeError(f"sibling analysis failed ({details})")
    return save_dir


def _delay_pca_scope_spec(rules, basis_scope):
    """Metadata shared by PCA fitting and fixed-point projection."""
    if basis_scope == "joint":
        return {
            "fit_rule_indices": tuple(range(len(rules))),
            "source": "concatenated normal-trial delay1 trajectories from both tasks",
            "artifact_suffix": "",
            "title": "joint",
        }
    if basis_scope == "first_task_only":
        reference_rule = rules[0]
        return {
            "fit_rule_indices": (0,),
            "source": (f"normal-trial delay1 trajectories from "
                       f"{reference_rule} only"),
            "artifact_suffix": f"_{reference_rule}_only",
            "title": f"{reference_rule}-only",
        }
    raise ValueError(f"unknown basis_scope {basis_scope!r}; choose from "
                     f"{DELAY_PCA_SCOPES}")


def _delay_pca_path(save_dir, aname, addtask, rules, basis_scope, method):
    """Return the method-specific PCA artifact path for one family/scope."""
    scope = _delay_pca_scope_spec(rules, basis_scope)
    if method == "gradient":
        # Preserve the historical gradient filename for compatibility.
        method_suffix = ""
    elif method == "long_delay_endpoint":
        method_suffix = "_long_delay_endpoint"
    else:
        raise ValueError(f"unknown sibling method {method!r}; choose from "
                         f"{SIBLING_METHODS}")
    return (Path(save_dir)
            / f"{addtask}{method_suffix}_delay_trajectory_pca"
              f"{scope['artifact_suffix']}_{aname}.pkl")


def _tracked_numpy(value):
    """Convert one tracked-state entry to a single NumPy array."""
    if isinstance(value, (list, tuple)):
        return np.concatenate([_tracked_numpy(v) for v in value], axis=-1)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pca_record(pca, n_samples, method):
    """Portable PCA parameters consumed by paper_plot.py (no sklearn pickle)."""
    return {
        "mean": np.asarray(pca.mean_, dtype=np.float32),
        "components": np.asarray(pca.components_, dtype=np.float32),
        "explained_variance": np.asarray(pca.explained_variance_, dtype=np.float32),
        "explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype=np.float32),
        "n_samples": int(n_samples),
        "method": str(method),
    }


def _aligned_delay1_window(rules, trials):
    """Return the common scalar ``delay1`` window for aligned sibling trials."""
    if len(trials) != len(rules):
        raise ValueError(f"got {len(trials)} trial objects for {len(rules)} rules")
    windows = []
    for rule, trial in zip(rules, trials):
        if "delay1" not in trial.epochs:
            raise KeyError(f"{rule}: trial has no delay1 epoch")
        start, stop = trial.epochs["delay1"]
        if np.ndim(start) or np.ndim(stop):
            raise ValueError(f"{rule}: delay1 must be a scalar aligned window, "
                             f"got {(start, stop)}")
        windows.append((int(start), int(stop)))
    if len(set(windows)) != 1:
        raise ValueError("sibling delay1 windows are not aligned: "
                         f"{dict(zip(rules, windows))}")
    start, stop = windows[0]
    if stop <= start:
        raise ValueError(f"empty delay1 window {windows[0]}")
    return start, stop


def _run_delay_tracking_chunk(model, inputs, delay_start, delay_stop,
                              layer_index):
    """Run one GPU batch while retaining states only inside ``delay1``."""
    if not (0 <= delay_start < delay_stop <= inputs.shape[1]):
        raise ValueError(f"invalid delay window {(delay_start, delay_stop)} "
                         f"for sequence length {inputs.shape[1]}")
    model.reset_state(B=inputs.shape[0])
    outputs, hidden_delay, modulation_delay = [], [], []
    with torch.no_grad():
        for seq_idx in range(inputs.shape[1]):
            track = delay_start <= seq_idx < delay_stop
            output, _, db = model.network_step(
                inputs[:, seq_idx, :],
                run_mode="track_states" if track else "minimal",
                seq_idx=seq_idx)
            outputs.append(output.detach().cpu())
            if track:
                hidden_delay.append(
                    np.asarray(_tracked_numpy(db[f"hidden{layer_index}"]),
                               dtype=np.float32))
                modulation_delay.append(
                    np.asarray(_tracked_numpy(db[f"M{layer_index}"]),
                               dtype=np.float32))
    return (
        torch.stack(outputs, dim=1),
        np.stack(hidden_delay, axis=1),
        np.stack(modulation_delay, axis=1),
    )


def _partial_fit_effective_modulation(pca, modulation, W,
                                      chunk_samples=128):
    """Incrementally fit W⊙M without materializing all flattened samples."""
    samples = modulation.reshape(-1, *modulation.shape[-2:])
    n_samples = int(samples.shape[0])
    n_components = int(pca.n_components)
    if n_samples < n_components:
        raise ValueError(f"PCA update has {n_samples} samples but needs at least "
                         f"{n_components}")
    chunk_samples = max(int(chunk_samples), n_components)
    start = 0
    n_chunks = 0
    while start < n_samples:
        remaining = n_samples - start
        stop = (n_samples if remaining <= chunk_samples + n_components - 1
                else start + chunk_samples)
        wm = (samples[start:stop] * W[None, :, :]).reshape(stop - start, -1)
        pca.partial_fit(wm)
        start = stop
        n_chunks += 1
    return n_chunks


def _write_effective_modulation(destination, start, modulation, W,
                                chunk_samples=128):
    """Write flattened W⊙M samples to a bounded-memory matrix or memmap."""
    samples = modulation.reshape(-1, *modulation.shape[-2:])
    chunk_samples = max(1, int(chunk_samples))
    for chunk_start in range(0, samples.shape[0], chunk_samples):
        chunk_stop = min(chunk_start + chunk_samples, samples.shape[0])
        destination[start + chunk_start:start + chunk_stop] = (
            samples[chunk_start:chunk_stop] * W[None, :, :]
        ).reshape(chunk_stop - chunk_start, -1)
    return int(samples.shape[0])


def _fit_randomized_pca_candidates(samples, n_components, seeds):
    """Fit reproducible conventional-PCA candidates using randomized SVD."""
    candidates = []
    for seed in seeds:
        pca = PCA(
            n_components=int(n_components), svd_solver="randomized",
            random_state=int(seed), copy=True)
        pca.fit(samples)
        record = _pca_record(
            pca, samples.shape[0],
            f"PCA(svd_solver=randomized, random_state={int(seed)})")
        record["random_seed"] = int(seed)
        candidates.append(record)
        del pca
        gc.collect()
    return candidates


def stream_normal_delay_analysis(
        aname, save_dir, addtask, rules, model, device,
        norm_input, norm_output, norm_mask, norm_trials, norm_task, W,
        layer_index=1, gpu_batch_size=SIBLING_NORMAL_GPU_BATCH_SIZE,
        pca_chunk_samples=128, n_components=6,
        pca_strategy="incremental"):
    """Evaluate normal-delay trials and fit the requested PCA candidates.

    Inputs, labels, and masks remain on CPU. Each configured trial batch is
    moved to the GPU, and only delay-period hidden/M states are copied back.
    ``incremental`` immediately updates bounded-memory IncrementalPCA models.
    ``randomized_candidates`` writes W⊙M to temporary memory-mapped matrices
    and fits ten conventional PCA candidates sequentially; this avoids keeping
    the multi-gigabyte effective-modulation matrix in RAM.
    """
    delay_start, delay_stop = _aligned_delay1_window(rules, norm_trials)
    task_idx = np.asarray(norm_task, dtype=int)
    n_trials = int(norm_input.shape[0])
    if task_idx.shape != (n_trials,):
        raise ValueError(f"{addtask}: {task_idx.size} task labels for "
                         f"{n_trials} trajectories")
    trial_counts = {rule: int(np.sum(task_idx == i))
                    for i, rule in enumerate(rules)}
    if any(count == 0 for count in trial_counts.values()):
        raise ValueError(f"{addtask}: missing normal-delay trials: {trial_counts}")

    W = np.asarray(W, dtype=np.float32)
    n_components = int(n_components)
    if pca_strategy not in ("incremental", "randomized_candidates"):
        raise ValueError(f"unknown PCA strategy {pca_strategy!r}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timepoints_per_trial = int(delay_stop - delay_start)
    fitters = {}
    candidate_stores = {}
    temporary_store = None
    if pca_strategy == "incremental":
        for basis_scope in DELAY_PCA_SCOPES:
            fitters[basis_scope] = {
                "hidden": IncrementalPCA(n_components=n_components),
                "fixed_WM": IncrementalPCA(n_components=n_components),
                "hidden_samples": 0,
                "wm_samples": 0,
                "wm_chunks": 0,
            }
    else:
        temporary_store = tempfile.TemporaryDirectory(
            prefix=f".{addtask}_pca_", dir=save_dir)
        temporary_root = Path(temporary_store.name)
        for basis_scope in DELAY_PCA_SCOPES:
            scope = _delay_pca_scope_spec(rules, basis_scope)
            n_scope_trials = sum(
                trial_counts[rules[index]]
                for index in scope["fit_rule_indices"])
            n_scope_samples = n_scope_trials * timepoints_per_trial
            wm_path = temporary_root / f"{basis_scope}_effective_modulation.dat"
            candidate_stores[basis_scope] = {
                "hidden_chunks": [],
                "fixed_WM": np.memmap(
                    wm_path, mode="w+", dtype=np.float32,
                    shape=(n_scope_samples, W.size)),
                "offset": 0,
                "n_samples": n_scope_samples,
            }

    gpu_batch_size = max(1, int(gpu_batch_size))
    n_gpu_batches = (n_trials + gpu_batch_size - 1) // gpu_batch_size
    progress_every = max(1, n_gpu_batches // 20)
    acc_weighted = 0.0
    acc_count = 0
    task_acc_weighted = {i: 0.0 for i in range(len(rules))}
    task_acc_count = {i: 0 for i in range(len(rules))}
    preview_output = None

    for batch_number, start in enumerate(
            range(0, n_trials, gpu_batch_size), start=1):
        stop = min(start + gpu_batch_size, n_trials)
        input_cpu = norm_input[start:stop]
        output_cpu = norm_output[start:stop]
        mask_cpu = norm_mask[start:stop]
        input_device = input_cpu.to(device)
        predicted_cpu, hidden_delay, modulation_delay = _run_delay_tracking_chunk(
            model, input_device, delay_start, delay_stop, layer_index)

        if preview_output is None:
            preview_output = predicted_cpu[:5].clone()

        predicted_device = predicted_cpu.to(device)
        output_device = output_cpu.to(device)
        mask_device = mask_cpu.to(device)
        acc, _ = model.compute_acc(
            predicted_device, output_device, mask_device, input_device,
            isvalid=True, mode=model.acc_measure)
        batch_n = stop - start
        acc_weighted += float(acc) * batch_n
        acc_count += batch_n

        batch_tasks = task_idx[start:stop]
        for task_i in range(len(rules)):
            selected = np.flatnonzero(batch_tasks == task_i)
            if selected.size == 0:
                continue
            selected_device = torch.as_tensor(selected, device=device)
            acc_i, _ = model.compute_acc(
                predicted_device.index_select(0, selected_device),
                output_device.index_select(0, selected_device),
                mask_device.index_select(0, selected_device),
                input_device.index_select(0, selected_device),
                isvalid=True, mode=model.acc_measure)
            task_acc_weighted[task_i] += float(acc_i) * selected.size
            task_acc_count[task_i] += int(selected.size)

        if tuple(modulation_delay.shape[-2:]) != tuple(W.shape):
            raise ValueError(f"{addtask}: streamed M/W shapes disagree: "
                             f"{modulation_delay.shape[-2:]} vs {W.shape}")
        for basis_scope in DELAY_PCA_SCOPES:
            scope = _delay_pca_scope_spec(rules, basis_scope)
            selected = np.isin(batch_tasks, scope["fit_rule_indices"])
            if not np.any(selected):
                continue
            if np.all(selected):
                hidden_selected = hidden_delay
                modulation_selected = modulation_delay
            else:
                hidden_selected = hidden_delay[selected]
                modulation_selected = modulation_delay[selected]
            hidden_samples = hidden_selected.reshape(
                -1, hidden_selected.shape[-1])
            if pca_strategy == "incremental":
                fitter = fitters[basis_scope]
                fitter["hidden"].partial_fit(hidden_samples)
                fitter["hidden_samples"] += int(hidden_samples.shape[0])
                fitter["wm_samples"] += int(
                    np.prod(modulation_selected.shape[:2]))
                fitter["wm_chunks"] += _partial_fit_effective_modulation(
                    fitter["fixed_WM"], modulation_selected, W,
                    chunk_samples=pca_chunk_samples)
            else:
                store = candidate_stores[basis_scope]
                store["hidden_chunks"].append(
                    np.asarray(hidden_samples, dtype=np.float32).copy())
                written = _write_effective_modulation(
                    store["fixed_WM"], store["offset"],
                    modulation_selected, W, chunk_samples=pca_chunk_samples)
                store["offset"] += written

        del input_device, predicted_device, output_device, mask_device
        del predicted_cpu, hidden_delay, modulation_delay
        if (batch_number == 1 or batch_number == n_gpu_batches
                or batch_number % progress_every == 0):
            print(f"  [{addtask}/stream] GPU batch {batch_number}/"
                  f"{n_gpu_batches}; trials {start}:{stop}")

    for basis_scope in DELAY_PCA_SCOPES:
        scope = _delay_pca_scope_spec(rules, basis_scope)
        fit_rules = [rules[i] for i in scope["fit_rule_indices"]]
        fit_trial_counts = {rule: trial_counts[rule] for rule in fit_rules}
        if pca_strategy == "incremental":
            fitter = fitters[basis_scope]
            representations = {
                "fixed_hidden": _pca_record(
                    fitter["hidden"], fitter["hidden_samples"],
                    f"IncrementalPCA(gpu_batch_size={gpu_batch_size})"),
                "fixed_WM": _pca_record(
                    fitter["fixed_WM"], fitter["wm_samples"],
                    f"IncrementalPCA(chunk_samples={pca_chunk_samples}, "
                    f"gpu_batch_size={gpu_batch_size})"),
            }
            pca_candidates = None
            wm_samples = fitter["wm_samples"]
            fit_summary = f"PCA chunks={fitter['wm_chunks']}"
        else:
            store = candidate_stores[basis_scope]
            if store["offset"] != store["n_samples"]:
                raise RuntimeError(
                    f"{addtask}/{basis_scope}: wrote {store['offset']} PCA "
                    f"samples, expected {store['n_samples']}")
            store["fixed_WM"].flush()
            hidden_samples = np.concatenate(store["hidden_chunks"], axis=0)
            pca_candidates = {
                "fixed_hidden": _fit_randomized_pca_candidates(
                    hidden_samples, n_components,
                    SIBLING_LONG_DELAY_PCA_SEEDS),
                "fixed_WM": _fit_randomized_pca_candidates(
                    store["fixed_WM"], n_components,
                    SIBLING_LONG_DELAY_PCA_SEEDS),
            }
            # The endpoint-based selection happens after long-delay endpoints
            # exist. Seed 0 is only a temporary placeholder until then.
            representations = {
                key: candidates[0]
                for key, candidates in pca_candidates.items()
            }
            wm_samples = store["n_samples"]
            fit_summary = (
                f"PCA candidates={len(SIBLING_LONG_DELAY_PCA_SEEDS)}")
        artifact = {
            "version": 6 if pca_candidates is not None else 4,
            "aname": aname,
            "family": addtask,
            "rules": list(rules),
            "basis_scope": basis_scope,
            "fit_rules": fit_rules,
            "period": "delay1",
            "period_window": (int(delay_start), int(delay_stop)),
            "source": scope["source"],
            "trial_counts": trial_counts,
            "fit_trial_counts": fit_trial_counts,
            "timepoints_per_trial": timepoints_per_trial,
            "n_components": n_components,
            "gpu_batch_size": gpu_batch_size,
            "pca_strategy": pca_strategy,
            "representations": representations,
        }
        if pca_candidates is not None:
            artifact["pca_candidates"] = pca_candidates
        method = ("gradient" if pca_strategy == "incremental"
                  else "long_delay_endpoint")
        out_path = _delay_pca_path(
            save_dir, aname, addtask, rules, basis_scope, method)
        with out_path.open("wb") as stream:
            pickle.dump(artifact, stream)
        print(f"  [{addtask}/delay-pca/{basis_scope}] trials="
              f"{fit_trial_counts}, samples={wm_samples}, "
              f"GPU batch={gpu_batch_size}, {fit_summary}")
        print(f"  Saved {basis_scope} delay-trajectory PCA basis: {out_path}")

    if temporary_store is not None:
        for store in candidate_stores.values():
            store["fixed_WM"]._mmap.close()
        temporary_store.cleanup()

    per_task_acc = {
        rules[i]: task_acc_weighted[i] / task_acc_count[i]
        for i in range(len(rules))
    }
    return acc_weighted / acc_count, per_task_acc, preview_output


def _run_to_delay_endpoint(model, inputs, delay_stop, layer_index):
    """Run through delay1 while retaining only its final full-dimensional state."""
    if not (1 <= int(delay_stop) <= inputs.shape[1]):
        raise ValueError(f"delay stop {delay_stop} outside sequence length "
                         f"{inputs.shape[1]}")
    model.reset_state(B=inputs.shape[0])
    endpoint_db = None
    with torch.no_grad():
        for seq_idx in range(int(delay_stop)):
            track = seq_idx == int(delay_stop) - 1
            _, _, db = model.network_step(
                inputs[:, seq_idx, :],
                run_mode="track_states" if track else "minimal",
                seq_idx=seq_idx)
            if track:
                endpoint_db = db
    if endpoint_db is None:
        raise RuntimeError("long-delay run did not capture its endpoint")
    modulation = _tracked_numpy(endpoint_db[f"M{layer_index}"]).astype(
        np.float32, copy=True)
    hidden = _tracked_numpy(endpoint_db[f"hidden{layer_index}"]).astype(
        np.float32, copy=True)
    return modulation, hidden


def _run_long_delay_projected_trajectory(
        model, inputs, delay_start, delay_stop, layer_index, W, basis,
        max_samples=SIBLING_LONG_DELAY_TRAJECTORY_SAMPLES):
    """Run a long delay and retain sparse six-PC trajectories plus its endpoint.

    Projection happens immediately on the compute device. Only the resulting
    six coordinates are copied to CPU, so memory does not scale as
    ``delay_steps × batch × hidden × embedding``.
    """
    if not (0 <= delay_start < delay_stop <= inputs.shape[1]):
        raise ValueError(f"invalid delay window {(delay_start, delay_stop)} "
                         f"for sequence length {inputs.shape[1]}")
    n_delay_steps = int(delay_stop - delay_start)
    n_samples = min(max(2, int(max_samples)), n_delay_steps)
    relative_indices = np.unique(np.rint(
        np.linspace(0, n_delay_steps - 1, n_samples)).astype(int))
    sample_indices = delay_start + relative_indices
    sample_set = set(sample_indices.tolist())

    projectors = {}
    for key in ("fixed_hidden", "fixed_WM"):
        record = basis.get("representations", {}).get(key)
        if record is None:
            raise KeyError(f"trajectory PCA basis has no {key!r}")
        mean = np.asarray(record["mean"], dtype=np.float32)
        components = np.asarray(record["components"], dtype=np.float32)[:6]
        if components.shape != (6, mean.size):
            raise ValueError(f"{key}: expected six PCA components, got "
                             f"{components.shape} for mean {mean.shape}")
        projectors[key] = (
            torch.as_tensor(mean, device=inputs.device),
            torch.as_tensor(components, device=inputs.device),
        )

    W_device = torch.as_tensor(
        np.asarray(W, dtype=np.float32), device=inputs.device)
    model.reset_state(B=inputs.shape[0])
    projected = {"fixed_hidden": [], "fixed_WM": []}
    endpoint_M = endpoint_hidden = None
    with torch.no_grad():
        for seq_idx in range(int(delay_stop)):
            track = seq_idx in sample_set
            _, _, db = model.network_step(
                inputs[:, seq_idx, :],
                run_mode="track_states" if track else "minimal",
                seq_idx=seq_idx)
            if not track:
                continue
            hidden_state = db[f"hidden{layer_index}"]
            modulation_state = db[f"M{layer_index}"]
            if not torch.is_tensor(hidden_state):
                hidden_state = torch.as_tensor(
                    _tracked_numpy(hidden_state), device=inputs.device)
            if not torch.is_tensor(modulation_state):
                modulation_state = torch.as_tensor(
                    _tracked_numpy(modulation_state), device=inputs.device)

            hidden_mean, hidden_components = projectors["fixed_hidden"]
            hidden_projection = (
                hidden_state.reshape(hidden_state.shape[0], -1) - hidden_mean
            ) @ hidden_components.T
            wm_mean, wm_components = projectors["fixed_WM"]
            effective = (modulation_state * W_device[None, :, :]).reshape(
                modulation_state.shape[0], -1)
            wm_projection = (effective - wm_mean) @ wm_components.T
            projected["fixed_hidden"].append(
                hidden_projection.detach().cpu().numpy().astype(
                    np.float32, copy=False))
            projected["fixed_WM"].append(
                wm_projection.detach().cpu().numpy().astype(
                    np.float32, copy=False))

            if seq_idx == int(delay_stop) - 1:
                endpoint_M = _tracked_numpy(modulation_state).astype(
                    np.float32, copy=True)
                endpoint_hidden = _tracked_numpy(hidden_state).astype(
                    np.float32, copy=True)

    if endpoint_M is None or endpoint_hidden is None:
        raise RuntimeError("long-delay trajectory did not capture its endpoint")
    projected = {
        key: np.stack(values, axis=1)
        for key, values in projected.items()
    }
    return endpoint_M, endpoint_hidden, projected, sample_indices


def _select_endpoint_pca_candidates(
        aname, save_dir, addtask, rules, endpoint_values, *, stim_idx,
        task_idx, group_labels=None):
    """Select and activate the PCA seed with the best task-specific score."""
    selected_bases = {}
    for basis_scope in DELAY_PCA_SCOPES:
        basis_path = _delay_pca_path(
            save_dir, aname, addtask, rules, basis_scope,
            "long_delay_endpoint")
        with basis_path.open("rb") as stream:
            basis = pickle.load(stream)
        all_candidates = basis.get("pca_candidates")
        if not all_candidates:
            selected_bases[basis_scope] = basis
            continue
        candidate_counts = {
            len(candidates) for candidates in all_candidates.values()
        }
        if len(candidate_counts) != 1:
            raise ValueError(
                f"{basis_path}: PCA representations have unequal seed counts")
        n_seed_candidates = candidate_counts.pop()

        selection = {}
        for representation, values in endpoint_values.items():
            flattened = np.asarray(values, dtype=float).reshape(
                values.shape[0], -1)
            scored = []
            for candidate in all_candidates[representation]:
                mean = np.asarray(candidate["mean"], dtype=float)
                components = np.asarray(candidate["components"], dtype=float)
                projection = (flattened - mean) @ components.T
                pair, score, metric_name = best_task_specific_pc_pair(
                    projection, addtask, stim_idx=stim_idx,
                    group_labels=group_labels, task_idx=task_idx)
                scored.append((score, int(candidate["random_seed"]), pair,
                               candidate))
            scored.sort(key=lambda item: (-item[0], item[1]))
            score, seed, pair, selected = scored[0]
            basis["representations"][representation] = selected
            selection[representation] = {
                "random_seed": seed,
                "best_pc_pair": pair,
                "task_specific_score": score,
            }
            print(f"  [{addtask}/delay-pca/{basis_scope}/{representation}] "
                  f"selected seed={seed}, PC{pair[0]}-PC{pair[1]}, "
                  f"{metric_name}={score:.4f}")
        basis["candidate_selection"] = {
            "strategy": "maximum endpoint task-specific score",
            "metric": metric_name,
            "n_seeds": n_seed_candidates,
            "representations": selection,
        }
        with basis_path.open("wb") as stream:
            pickle.dump(basis, stream)
        selected_bases[basis_scope] = basis
    return selected_bases


def extract_long_delay_endpoints(
        aname, save_dir, addtask, rules, model, device,
        converted_task_params, W, layer_index=1,
        n_trials_per_rule=SIBLING_LONG_DELAY_TRIALS_PER_RULE):
    """Use the end of a generated very-long ``delay1`` as a fixed-point proxy.

    This is deliberately a second method, not a replacement for the gradient
    solve. Sibling rules are generated with aligned RNG state and equal local
    trial order, so ``condition_idx`` matches the same natural trial across the
    two rules. Full-dimensional state is retained only at the endpoint; up to
    51 delay frames are retained as compact six-PC trajectories.
    """
    rules = list(rules)
    tp = copy.deepcopy(converted_task_params)
    tp["rules"] = rules
    tp["long_delay"] = "long"
    tp["hp"]["batch_size_train"] = int(n_trials_per_rule)
    data, extra = mpn_tasks.generate_trials_wrap(
        tp, int(n_trials_per_rule), rules=rules, mode_input="random",
        device="cpu", verbose=True, align_periods=True,
        balanced_stimulus_directions=(addtask == "delaydm1"))
    long_input = data[0]
    _, trials, _ = extra
    delay_start, delay_stop = _aligned_delay1_window(rules, trials)

    task_idx = np.asarray(
        helper.find_task(tp, long_input.detach().cpu().numpy(), 0), dtype=int)
    task_idx -= task_idx.min()
    expected = np.repeat(
        np.arange(len(rules), dtype=int), int(n_trials_per_rule))
    if not np.array_equal(task_idx, expected):
        raise ValueError(f"{addtask}: unexpected aligned sibling task order")

    stim_idx = np.concatenate([
        np.asarray(trial.meta["stim1"], dtype=int).reshape(-1)
        for trial in trials
    ])
    stimulus_magnitude = np.concatenate([
        np.asarray(
            trial.meta.get("stim1_strs", np.ones(int(n_trials_per_rule))),
            dtype=float).reshape(-1)
        for trial in trials
    ])
    condition_idx = np.tile(
        np.arange(int(n_trials_per_rule), dtype=int), len(rules))
    if not (stim_idx.size == stimulus_magnitude.size == task_idx.size):
        raise ValueError(
            f"{addtask}: endpoint metadata length does not match trial count")
    direction_counts = None
    if addtask == "delaydm1":
        direction_counts = {}
        expected_per_direction = int(n_trials_per_rule) // SIBLING_FP_N_STIM
        for task_i, rule in enumerate(rules):
            counts = np.bincount(
                stim_idx[task_idx == task_i], minlength=SIBLING_FP_N_STIM)
            if (counts.size != SIBLING_FP_N_STIM
                    or not np.all(counts == expected_per_direction)):
                raise ValueError(
                    f"{rule}: unbalanced long-delay directions {counts.tolist()}")
            direction_counts[rule] = counts.tolist()

    W = np.asarray(W, dtype=np.float32)
    long_input_device = long_input.to(device)
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(long_input_device.device)
        if long_input_device.is_cuda else None
    )
    numpy_rng_state = np.random.get_state()
    M, hidden = _run_to_delay_endpoint(
        model, long_input_device, delay_stop, layer_index)
    if tuple(M.shape[-2:]) != tuple(W.shape):
        raise ValueError(f"{addtask}: endpoint M/W shapes disagree: "
                         f"{M.shape[-2:]} vs {W.shape}")
    effective_M = (M * W[None, :, :]).astype(np.float32, copy=False)

    group_labels = None
    if addtask == "dmcgo":
        group_labels = _dmc_category_labels(
            task_idx, stim_idx, int(stim_idx.max()) + 1)
    elif addtask != "delaydm1":
        raise ValueError(
            f"no endpoint PCA task-specific metric for {addtask!r}")
    selected_bases = _select_endpoint_pca_candidates(
        aname, save_dir, addtask, rules,
        {"fixed_hidden": hidden, "fixed_WM": effective_M},
        stim_idx=stim_idx, task_idx=task_idx, group_labels=group_labels)
    trajectory_basis = selected_bases["joint"]
    basis_path = _delay_pca_path(
        save_dir, aname, addtask, rules, "joint", "long_delay_endpoint")
    # Replay exactly the same stochastic forward pass now that the winning PCA
    # basis is known. This keeps the saved endpoint and projected trajectory on
    # the same realization even when the network injects recurrent noise.
    torch.random.set_rng_state(torch_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, long_input_device.device)
    np.random.set_state(numpy_rng_state)
    M, hidden, projected_trajectories, trajectory_sample_indices = (
        _run_long_delay_projected_trajectory(
            model, long_input_device, delay_start, delay_stop, layer_index,
            W, trajectory_basis))
    effective_M = (M * W[None, :, :]).astype(np.float32, copy=False)

    artifact = {
        "version": 2,
        "method": "long_delay_endpoint",
        "aname": aname,
        "family": addtask,
        "task_names": rules,
        "long_delay_setting": "long",
        "delay_epoch": "delay1",
        "delay_window": (int(delay_start), int(delay_stop)),
        "delay_endpoint_index": int(delay_stop - 1),
        "delay_steps": int(delay_stop - delay_start),
        "delay_ms": int((delay_stop - delay_start) * tp["dt"]),
        "trajectory_basis_scope": "joint",
        "trajectory_pca_source": str(basis_path),
        "trajectory_sample_indices": np.asarray(
            trajectory_sample_indices, dtype=int),
        "trajectory_delay_steps": np.asarray(
            trajectory_sample_indices - delay_start, dtype=int),
        "trajectory_time_ms": np.asarray(
            (trajectory_sample_indices - delay_start) * tp["dt"], dtype=int),
        "task_idx": task_idx,
        "condition_idx": condition_idx,
        "stim_idx": stim_idx,
        "stimulus_magnitude": stimulus_magnitude,
        "balanced_stimulus_directions": addtask == "delaydm1",
        "representations": {
            "fixed_M": M,
            "fixed_WM": effective_M,
            "fixed_hidden": hidden,
        },
        "projected_trajectories": projected_trajectories,
    }
    if direction_counts is not None:
        artifact["direction_counts"] = direction_counts
    if addtask == "dmcgo":
        artifact["group_labels"] = group_labels

    out_path = (Path(save_dir)
                / f"{addtask}_long_delay_endpoints_{aname}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  [{addtask}/long-delay-endpoint] saved {task_idx.size} endpoints "
          f"and {trajectory_sample_indices.size}-step projected trajectories "
          f"from delay1 step {delay_stop - 1} ({artifact['delay_ms']} ms): "
          f"{out_path}")
    del long_input_device, long_input, data
    return out_path


_FIXED_POINT_REPS = (("hidden", "fixed_hidden"),
                     ("e_modulation", "fixed_WM"))


def save_long_delay_endpoint_pc_projections(
        aname, save_dir, addtask, rules, endpoint_path,
        basis_scope="joint"):
    """Project long-delay endpoints through a saved trajectory-PCA basis.

    Endpoint coordinates are produced for every requested basis. Compact delay
    trajectories are attached only when their online projection basis matches
    ``basis_scope``; currently they are sampled in the joint sibling basis.
    """
    save_dir = Path(save_dir)
    scope = _delay_pca_scope_spec(rules, basis_scope)
    artifact_suffix = scope["artifact_suffix"]
    basis_path = _delay_pca_path(
        save_dir, aname, addtask, rules, basis_scope,
        "long_delay_endpoint")
    with open(basis_path, "rb") as f:
        basis = pickle.load(f)
    with open(endpoint_path, "rb") as f:
        endpoints = pickle.load(f)
    if endpoints.get("method") != "long_delay_endpoint":
        raise ValueError(f"{endpoint_path}: not a long-delay endpoint artifact")
    if list(endpoints.get("task_names", [])) != list(rules):
        raise ValueError(f"{endpoint_path}: expected rules {rules}, got "
                         f"{endpoints.get('task_names')}")

    out = {
        "version": 3,
        "method": "long_delay_endpoint",
        "aname": aname,
        "family": addtask,
        "task_names": list(rules),
        "basis_scope": basis_scope,
        "pca_source": str(basis_path),
        "pca_selection": basis.get("candidate_selection"),
        "endpoint_source": str(endpoint_path),
        "delay_window": tuple(endpoints["delay_window"]),
        "delay_steps": int(endpoints["delay_steps"]),
        "delay_ms": int(endpoints["delay_ms"]),
        "n_components": 6,
        "representations": {},
    }
    for plot_name, endpoint_key in _FIXED_POINT_REPS:
        record = basis["representations"][endpoint_key]
        mean = np.asarray(record["mean"], dtype=float)
        components = np.asarray(record["components"], dtype=float)[:6]
        values = np.asarray(
            endpoints["representations"][endpoint_key], dtype=float)
        values = values.reshape(values.shape[0], -1)
        if values.shape[1] != mean.size or components.shape != (6, mean.size):
            raise ValueError(f"{addtask}/{plot_name}: endpoint/PCA dimensions "
                             f"disagree ({values.shape[1]}, {components.shape}, "
                             f"mean={mean.size})")
        rep_out = {
            "proj": np.asarray((values - mean) @ components.T, dtype=np.float32),
            "task_idx": np.asarray(endpoints["task_idx"], dtype=int),
            "condition_idx": np.asarray(endpoints["condition_idx"], dtype=int),
            "stim_idx": np.asarray(endpoints["stim_idx"], dtype=int),
            "stimulus_magnitude": np.asarray(
                endpoints["stimulus_magnitude"], dtype=float),
            "task_names": list(rules),
            "explained_variance_ratio": np.asarray(
                record.get("explained_variance_ratio", []), dtype=float)[:6],
        }
        if (basis_scope == endpoints.get("trajectory_basis_scope")
                and endpoint_key in endpoints.get("projected_trajectories", {})):
            trajectory = np.asarray(
                endpoints["projected_trajectories"][endpoint_key],
                dtype=np.float32)
            if (trajectory.ndim != 3
                    or trajectory.shape[0] != values.shape[0]
                    or trajectory.shape[2] != 6):
                raise ValueError(f"{addtask}/{plot_name}: invalid projected "
                                 f"trajectory shape {trajectory.shape}")
            rep_out["trajectory_proj"] = trajectory
            rep_out["trajectory_sample_indices"] = np.asarray(
                endpoints["trajectory_sample_indices"], dtype=int)
            rep_out["trajectory_delay_steps"] = np.asarray(
                endpoints["trajectory_delay_steps"], dtype=int)
            rep_out["trajectory_time_ms"] = np.asarray(
                endpoints["trajectory_time_ms"], dtype=int)
        if addtask == "dmcgo" and "group_labels" in endpoints:
            rep_out["group_labels"] = np.asarray(endpoints["group_labels"])
        gallery_suffix = f"_long_delay_endpoint{artifact_suffix}"
        rep_out["gallery_path"] = str(_save_pc_pair_gallery(
            save_dir, aname, addtask, gallery_suffix, basis_scope,
            plot_name, rep_out))
        out["representations"][plot_name] = rep_out

    out_path = (save_dir
                / f"{addtask}_long_delay_endpoint_pc_projections"
                  f"{artifact_suffix}_{aname}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"  Saved {basis_scope} {addtask} long-delay endpoint projections: "
          f"{out_path}")
    return out_path


def _load_fixed_point_rules(aname, save_dir, rules):
    """Load sibling-rule fixed-point pickles in one shared, validated order."""
    per_rule = []
    for rule in rules:
        path = Path(save_dir) / f"fixed_points_grad_{aname}_{rule}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            per_rule.append((rule, pickle.load(f)))
    return per_rule


def _stimulus_color(stim, n_stim):
    """Match paper_plot.py's red-to-purple circular-stimulus color ramp."""
    frac = (int(stim) % n_stim) / max(n_stim - 1, 1)
    return mpl.colors.hsv_to_rgb((0.83 * frac, 0.85, 0.9))


def _adaptive_limits(xy, padding=0.10):
    """Independent x/y limits from the joint extent of all displayed points."""
    xy = np.asarray(xy, dtype=float)
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return (-1.0, 1.0), (-1.0, 1.0)
    lo = xy[finite].min(axis=0)
    hi = xy[finite].max(axis=0)
    center = 0.5 * (lo + hi)
    span = hi - lo
    fallback = max(float(np.max(span)), float(np.max(np.abs(center))), 1e-9) * 0.1
    span = np.where(span > np.finfo(float).eps, span, fallback)
    half = 0.5 * span * (1.0 + 2.0 * padding)
    return ((center[0] - half[0], center[0] + half[0]),
            (center[1] - half[1], center[1] + half[1]))


def _save_pc_pair_gallery(save_dir, aname, addtask, artifact_suffix,
                          basis_scope, plot_name, entry):
    """Draw all 15 pairwise views of a six-PC fixed-point projection.

    The gallery itself computes no score, highlights no panel, and writes no
    selected-plane metadata. ``paper_plot.py`` uses an explicit plane for the
    gradient method and task-specific automatic selection for long-delay
    endpoints.
    """
    proj = np.asarray(entry["proj"], dtype=float)
    if proj.ndim != 2 or proj.shape[1] != 6:
        raise ValueError(f"{addtask}/{basis_scope}/{plot_name}: gallery expects "
                         f"six PCs, got {proj.shape}")
    task_idx = np.asarray(entry["task_idx"], dtype=int)
    stim_idx = np.asarray(entry["stim_idx"], dtype=int)
    # Gradient-solver projections may carry a convergence mask. Long-delay
    # settling endpoints intentionally do not: every generated endpoint is
    # shown, so the default is an all-filled mask.
    is_fixed = np.asarray(
        entry.get("is_fixed", np.ones(proj.shape[0], dtype=bool)), dtype=bool)
    task_names = list(entry["task_names"])
    n_stim = int(stim_idx.max()) + 1
    evr = np.asarray(entry.get("explained_variance_ratio", []), dtype=float)
    pairs = [(x, y) for x in range(6) for y in range(x + 1, 6)]
    markers = ("s", "^")

    fig, axs = plt.subplots(3, 5, figsize=(13.5, 8.2), squeeze=False)
    for ax, (pc_x, pc_y) in zip(axs.flat, pairs):
        for task, rule in enumerate(task_names):
            sel_task = task_idx == task
            for stim in np.unique(stim_idx[sel_task]):
                sel = sel_task & (stim_idx == stim)
                color = _stimulus_color(stim, n_stim)
                good = sel & is_fixed
                bad = sel & ~is_fixed
                if np.any(good):
                    ax.scatter(proj[good, pc_x], proj[good, pc_y],
                               color=color, edgecolor="none", linewidth=0,
                               marker=markers[task % len(markers)], s=25,
                               zorder=2)
                if np.any(bad):
                    ax.scatter(proj[bad, pc_x], proj[bad, pc_y],
                               color="none", edgecolor=color, linewidth=0.9,
                               marker=markers[task % len(markers)], s=25,
                               zorder=2)

        xlim, ylim = _adaptive_limits(proj[:, [pc_x, pc_y]])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        x_var = f" ({100 * evr[pc_x]:.1f}%)" if evr.size > pc_x else ""
        y_var = f" ({100 * evr[pc_y]:.1f}%)" if evr.size > pc_y else ""
        ax.set_xlabel(f"PC{pc_x + 1}{x_var}", fontsize=7)
        ax.set_ylabel(f"PC{pc_y + 1}{y_var}", fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([], [], marker=markers[t % len(markers)],
                          color="0.3", linestyle="", markersize=5, label=rule)
               for t, rule in enumerate(task_names)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=len(handles), frameon=True, fontsize=8)
    fig.suptitle(f"{addtask} | {basis_scope} delay PCA | {plot_name} | "
                 "all 15 PC pairs (selection deferred to paper_plot)", fontsize=12,
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = (Path(save_dir)
            / f"{addtask}_delay_pc_gallery{artifact_suffix}_{plot_name}_{aname}.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"  Saved 15-pair PC gallery: {path}")
    return path


def save_sibling_fixed_point_pc_projections(
        aname, save_dir, addtask, rules, probe="longdelay", basis_scope="joint"):
    """Save one sibling family's fixed points in a six-PC trajectory basis.

    This upstream step deliberately makes no two-dimensional plane choice and
    computes no alignment score. The stored six-dimensional coordinates contain
    every possible PC-pair projection; paper_plot.py selects a pair explicitly.
    DMC category labels are included as metadata without scoring their separation.
    """
    if addtask not in ("delaydm1", "dmcgo"):
        raise ValueError("six-PC projection is defined for delayDM and DMC")
    if len(rules) != 2:
        raise ValueError(f"expected two sibling rules, got {rules}")

    save_dir = Path(save_dir)
    scope = _delay_pca_scope_spec(rules, basis_scope)
    artifact_suffix = scope["artifact_suffix"]
    basis_path = _delay_pca_path(
        save_dir, aname, addtask, rules, basis_scope, "gradient")
    with open(basis_path, "rb") as f:
        basis_artifact = pickle.load(f)
    if basis_artifact.get("basis_scope") != basis_scope:
        raise ValueError(f"{basis_path}: expected basis_scope={basis_scope!r}, "
                         f"got {basis_artifact.get('basis_scope')!r}")
    if list(basis_artifact.get("rules", [])) != list(rules):
        raise ValueError(f"{basis_path}: expected rules {rules}, got "
                         f"{basis_artifact.get('rules')}")

    per_rule = _load_fixed_point_rules(aname, save_dir, rules)

    out = {
        "version": 2,
        "aname": aname,
        "family": addtask,
        "task_names": list(rules),
        "probe": probe,
        "basis_scope": basis_scope,
        "pca_source": str(basis_path),
        "pca_fit_source": basis_artifact["source"],
        "n_components": 6,
        "available_pc_pairs": [(x + 1, y + 1)
                               for x in range(6) for y in range(x + 1, 6)],
        "representations": {},
    }
    for plot_name, fp_key in _FIXED_POINT_REPS:
        record = basis_artifact.get("representations", {}).get(fp_key)
        if record is None:
            raise KeyError(f"{basis_path}: missing PCA representation {fp_key!r}")
        mean = np.asarray(record["mean"], dtype=float)
        components = np.asarray(record["components"], dtype=float)
        if components.ndim != 2 or components.shape[0] < 6:
            raise ValueError(f"{basis_path}: {fp_key} needs >=6 trajectory PCs, "
                             f"got {components.shape}")
        components = components[:6]

        projected, task_labels, stim_labels = [], [], []
        magnitude_labels, fixed_labels = [], []
        for task, (rule, data) in enumerate(per_rule):
            entry = data.get("results", {}).get(probe)
            if entry is None or entry.get(fp_key) is None:
                raise KeyError(f"{rule}: missing {probe!r}/{fp_key!r}")
            values = np.asarray(entry[fp_key], dtype=float)
            values = values.reshape(values.shape[0], -1)
            if values.shape[1] != mean.size:
                raise ValueError(f"{rule}/{fp_key}: {values.shape[1]} features, "
                                 f"trajectory PCA expects {mean.size}")
            projected.append((values - mean) @ components.T)
            stim = np.asarray(entry["stim"], dtype=int)
            magnitude = np.asarray(
                entry.get("stimulus_magnitude", np.ones(stim.size)), dtype=float)
            if magnitude.shape != stim.shape:
                raise ValueError(f"{rule}: stimulus_magnitude shape "
                                 f"{magnitude.shape} != stim shape {stim.shape}")
            task_labels.extend([task] * stim.size)
            stim_labels.extend(stim.tolist())
            magnitude_labels.extend(magnitude.tolist())
            fixed_labels.extend(np.asarray(
                entry.get("is_fixed", np.ones(stim.size, bool)), dtype=bool).tolist())

        proj = np.vstack(projected)
        task_idx = np.asarray(task_labels, dtype=int)
        stim_idx = np.asarray(stim_labels, dtype=int)
        stimulus_magnitude = np.asarray(magnitude_labels, dtype=float)
        is_fixed = np.asarray(fixed_labels, dtype=bool)
        rep_out = {
            "proj": np.asarray(proj, dtype=np.float32),
            "task_idx": task_idx,
            "stim_idx": stim_idx,
            "stimulus_magnitude": stimulus_magnitude,
            "is_fixed": is_fixed,
            "task_names": list(rules),
            "explained_variance_ratio": np.asarray(
                record.get("explained_variance_ratio", []), dtype=float)[:6],
        }
        if addtask == "dmcgo":
            n_stim = int(stim_idx.max()) + 1
            rep_out["group_labels"] = _dmc_category_labels(
                task_idx, stim_idx, n_stim)
        gallery_path = _save_pc_pair_gallery(
            save_dir, aname, addtask, artifact_suffix, basis_scope,
            plot_name, rep_out)
        rep_out["gallery_path"] = str(gallery_path)
        out["representations"][plot_name] = rep_out
        print(f"  [{addtask}/pc-projection/{basis_scope}] {plot_name}: saved "
              f"{proj.shape[0]} points in all six Delay PCs")

    out_path = (save_dir
                / f"{addtask}_delay_pc_projections{artifact_suffix}_{aname}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"  Saved {basis_scope} {addtask} fixed-point PC projections: "
          f"{out_path}")
    return out_path


def _dmc_category_labels(task_idx, stim_idx, n_stim):
    """DMC category label per solved fixed point, pooled across the two rules.

    The split cuts across task and stimulus identity:
        Group A: task0 stim {0..n/2-1} + task1 stim {n/2..n-1}
        Group B: task0 stim {n/2..n-1} + task1 stim {0..n/2-1}
    Thus group A iff (task == 0) == (stim < n_stim/2). A separating plane
    encodes remembered category rather than stimulus identity or task cue.
    """
    half = n_stim // 2
    return np.asarray([int((t == 0) == (a < half))
                       for t, a in zip(task_idx, stim_idx)], dtype=int)


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=("Analyze DelayDM or DMC sibling-task memory geometry "
                     "without running clustering or lesion experiments."))
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed of the trained everything network (for example 749).")
    parser.add_argument("--feature", required=True,
                        help="Run feature token (for example L21e4).")
    parser.add_argument(
        "--families", nargs="+", required=True,
        choices=tuple(SIBLING_FAMILIES),
        help=("Sibling families to analyze. 'delaydm1' runs DelayDM1 and "
              "DelayDM2; 'dmcgo' runs DMCGo and DMCNoGo."))
    parser.add_argument(
        "--method", required=True, choices=SIBLING_METHODS,
        help=("Fixed-point method to run. 'gradient' runs the optimizer-based "
              "solver; 'long_delay_endpoint' runs very-long-delay trials and "
              "uses the last delay state. Only the selected method is run."))
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_sibling_analysis(args.seed, args.feature, args.families, args.method)


if __name__ == "__main__":
    main()
