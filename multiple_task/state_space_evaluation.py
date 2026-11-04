"""Shared checkpoint selection and evaluation trials for state-space analyses.

Both natural-trajectory and transplant analyses use these public helpers.
This module does not import either analysis script or set plotting styles;
device selection happens when prepare_evaluation is called.
"""
from pathlib import Path
import json

import numpy as np
import torch

import _bootstrap  # noqa: F401
import mpn_tasks
from network_loading import build_network
import multiple_task_performance as mpf

# Default paper cohort.  ``linear_embed`` is the input width of the plastic
# layer (called the projection dimension elsewhere in the repository), while
# n_neurons[1] is its output/hidden width.
TARGET_ACTIVATION = "tanh"
TARGET_PROJECTION_DIM = 300
TARGET_HIDDEN_DIM = 300
FIXED_FIXATION_MS = 500
EVALUATION_SEED = 42
TARGET_FEATURE_REG_LAMBDAS = {
    "L21e5": 1e-5,
    "L21e4": 1e-4,
    "L21e3": 1e-3,
    "L21e2": 1e-2,
}
STATE_SPACE_DIR = Path("state_space")
NOISE_LEVEL = 0.01

# Evaluation trials per task (one batch of every rule).
TRIALS_PER_RULE = 50

# Untrained control (eval_untrained): a fresh initialization of a run's
# architecture and config, drawn after seeding with UNTRAINED_INIT_SEED, whose
# PCA cache goes to <output dir>/UNTRAINED_DIR_NAME/.
UNTRAINED_DIR_NAME = "untrained"
UNTRAINED_INIT_SEED = 0


def checkpoint_aname(checkpoint_path):
    """Return the run identifier encoded by a savednet checkpoint path."""
    stem = Path(checkpoint_path).stem
    prefix = "savednet_"
    if not stem.startswith(prefix):
        raise ValueError(f"Expected a {prefix}*.pt checkpoint, got {checkpoint_path}")
    return stem[len(prefix):]


def _raw_config_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    aname = checkpoint_aname(checkpoint_path)
    return checkpoint_path.with_name(f"param_{aname}_param.json")


def matches_target_cohort(checkpoint_path):
    """Whether a checkpoint belongs to the default state-space cohort.

    Dimensions and activation are read from the saved parameter JSON rather
    than inferred from the filename.  The exact feature tag is checked as an
    additional guard against silently mixing regularization cohorts.
    """
    config_path = _raw_config_path(checkpoint_path)
    if not config_path.exists():
        return False

    try:
        feature = mpf.parse_feature(str(checkpoint_path))
    except ValueError:
        return False

    with config_path.open() as f:
        config = json.load(f)

    net_params = config.get("net_params", {})
    train_params = config.get("train_params", {})
    n_neurons = net_params.get("n_neurons", [])
    if len(n_neurons) != 3:
        return False

    target_reg_lambda = TARGET_FEATURE_REG_LAMBDAS.get(feature)
    return (
        target_reg_lambda is not None
        and str(net_params.get("activation", "")).casefold() == TARGET_ACTIVATION
        and net_params.get("input_layer_add") is True
        and int(net_params.get("linear_embed", -1)) == TARGET_PROJECTION_DIM
        and int(n_neurons[1]) == TARGET_HIDDEN_DIM
        and train_params.get("weight_reg") == "L2"
        and np.isclose(float(train_params.get("reg_lambda", np.nan)),
                       target_reg_lambda)
    )


def select_target_checkpoints(checkpoint_paths):
    """Filter checkpoint paths to the default cohort, preserving input order."""
    return [path for path in checkpoint_paths if matches_target_cohort(path)]


def modulation_retention(state_dict, load_net_params, dt_ms):
    """The plastic layer's retention lam per step, read from the checkpoint.

    Only a scalar lam is supported (every training script uses one). When the
    saved net params carry ``m_time_scale``, lam must equal 1 - dt / tau, so a
    checkpoint whose two records disagree is rejected rather than analyzed
    with the wrong decay term.
    """
    lam = state_dict.get("mp_layer1.lam")
    if lam is None:
        raise KeyError("Checkpoint has no mp_layer1.lam; the decay term cannot be removed.")
    lam = np.asarray(torch.as_tensor(lam).detach().cpu().numpy(), dtype=np.float64)
    if lam.size != 1:
        raise ValueError(f"Only a scalar lam is supported, got shape {lam.shape}.")
    lam = float(lam.reshape(-1)[0])
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must lie in (0, 1), got {lam}.")
    m_time_scale = (load_net_params or {}).get("ml_params", {}).get("m_time_scale")
    if m_time_scale is not None and not np.isclose(lam, 1.0 - dt_ms / m_time_scale, atol=1e-6):
        raise ValueError(f"lam {lam} disagrees with m_time_scale {m_time_scale} ms at dt {dt_ms} ms.")
    return lam


def prepare_evaluation(netpathname, untrained_init_seed=None, evaluation_seed=EVALUATION_SEED,
                       device=None):
    """Build the network of checkpoint ``netpathname`` and draw the evaluation
    trials every state-space analysis runs on.

    The network (trained, or untrained with ``untrained_init_seed``; see
    network_loading.build_network) is moved to ``device`` in eval mode. The
    trials are drawn after reseeding NumPy, torch and the task generator
    with ``evaluation_seed``: TRIALS_PER_RULE per task, input noise
    NOISE_LEVEL, every task's Context aligned to FIXED_FIXATION_MS (the
    trials do not depend on the network). Returns a dict: the run's
    identifiers (``aname``, ``hidden_size``, ``l2_info``), ``model``,
    ``state_dict``, the converted ``task_params``, the trial tensors
    ``inputs``, ``outputs`` and ``masks`` on the CPU, the ``trials`` objects,
    each trial's task index (``rule_indices``), ``retention`` (lam),
    ``dt_ms``, ``fixed_fixation_steps`` and ``evaluation_seed``. Shift and
    transplant use this helper so both experiments see the same trials.
    device defaults to CUDA when available, otherwise CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size, l2_info = mpf.parse_hidden_and_l2(netpathname)

    netpathname = Path(netpathname)
    aname = checkpoint_aname(netpathname)
    with _raw_config_path(netpathname).open() as f:
        raw_cfg_param = json.load(f)

    task_params, train_params, net_params = raw_cfg_param["task_params"], raw_cfg_param["train_params"], raw_cfg_param["net_params"]

    # Keep the checkpoint copy on CPU; only the live model needs to occupy GPU
    # memory.  This also makes the frozen W used below immediately NumPy-safe.
    checkpoint = torch.load(netpathname, map_location="cpu", weights_only=False)
    load_net_params = checkpoint["net_params"]
    print(checkpoint["state_dict"].keys())
    print(load_net_params)

    model, state_dict = build_network(checkpoint, untrained_init_seed)
    model.to(device)
    model.eval()

    task_params["sigma_x"] = NOISE_LEVEL
    np.random.seed(evaluation_seed)
    torch.manual_seed(evaluation_seed)

    task_params_c, _, _ = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )

    all_tasks = task_params_c['rules']

    # setup the evaluation dataset generator
    task_params_c['hp']['batch_size_train'] = TRIALS_PER_RULE
    task_params_c['hp']['seed'] = evaluation_seed
    task_params_c['hp']['rng'] = np.random.RandomState(evaluation_seed)
    dt_ms = task_params_c['hp']['dt']
    retention = modulation_retention(state_dict, load_net_params, dt_ms)
    print(f"Modulation retention lam = {retention:.6g} per {dt_ms:g} ms step")
    fixed_fixation_steps = max(1, int(FIXED_FIXATION_MS / dt_ms))

    test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
        task_params_c,
        TRIALS_PER_RULE,
        rules=all_tasks,
        mode_input="random",
        device="cpu",
        verbose=False,
        fixed_fixation_steps=fixed_fixation_steps,
    )
    test_input, test_output, test_mask = test_data
    _, test_trials, test_rule_idxs = test_trials_extra

    fixation_endpoints = {
        int(trial.epochs['fix1'][1]) for trial in test_trials
    }
    if fixation_endpoints != {fixed_fixation_steps}:
        raise RuntimeError(
            "State-space trials were not aligned to one fixation endpoint: "
            f"{sorted(fixation_endpoints)}"
        )
    print(
        f"Fixed fixation across tasks: {fixed_fixation_steps} steps "
        f"({fixed_fixation_steps * dt_ms:g} ms; requested "
        f"{FIXED_FIXATION_MS} ms)"
    )

    return {
        "aname": aname,
        "hidden_size": hidden_size,
        "l2_info": l2_info,
        "model": model,
        "state_dict": state_dict,
        "task_params": task_params_c,
        "inputs": test_input,
        "outputs": test_output,
        "masks": test_mask,
        "trials": test_trials,
        "rule_indices": test_rule_idxs,
        "retention": retention,
        "dt_ms": dt_ms,
        "fixed_fixation_steps": fixed_fixation_steps,
        "evaluation_seed": evaluation_seed,
    }


def untrained_file_stem(aname, init_seed):
    """File-name stem of an untrained network built from run ``aname``'s
    config with weights from ``init_seed``: one cache per seed, so several
    initializations of one config never overwrite each other."""
    return f"{aname}_init{int(init_seed)}"


def existing_checkpoints(checkpoint_paths):
    """The given paths as Paths, checked to be savednet_*.pt files."""
    paths = [Path(path) for path in checkpoint_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
    for path in paths:
        checkpoint_aname(path)
    return paths
