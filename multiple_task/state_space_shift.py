"""
State space geometry analysis across tasks.

Examines how the network's internal representations (hidden states, modulation
matrices, effective modulation) are geometrically organized across tasks in
high-dimensional space. Inspired by Fig 4C of Driscoll et al. (2024, Nature
Neuroscience), this script tests whether tasks that start from nearby initial
conditions also produce similar dynamical trajectories.

Analyses:
1. Context-end PCA — projects hidden/modulation states at the end of the
    Context period (just before stimulus onset) into 2D via PCA, colored by
    task and by the six paper color groups.
2. Initial condition distance vs. trajectory angle — for each pair of tasks
    and each shared stimulus index, averages Euclidean initial-state distances
    and first-step displacement angles over all cross-task trial pairs, then
    averages equally over the available stimulus indices. The task-pair means
    are fitted with free-intercept OLS (angle = intercept + slope * distance).
    Reported r is Pearson correlation. Two p-values are saved: the nominal OLS
    slope-test p, which assumes independent task pairs although pairs share
    tasks, and the two-sided task-label permutation (Mantel) p of r, which is
    the one to report (``DISTANCE_ANGLE_N_PERMUTATIONS`` permutations).
    A positive correlation indicates an association between more distant
    initial states and more different initial motion directions, not causation.

The PCA runs on three representations: hidden states, raw modulation M, and
effective modulation (W ⊙ M). The distance-angle analysis adds, for M and
W ⊙ M, a second step definition. The modulation update is
M[t] = lam * M[t-1] + eta * post (x) pre, clamped to the M bounds
(core/mpn.py, update_M_matrix), so the full first step M[t] - M[t-1] contains
-(1 - lam) * M[t-1]: a term along the initial condition itself, which relates
the step angle to the initial-condition distance without any task-dependent
computation. The ``mod_hebb`` and ``eff_mod_hebb`` representations therefore
use the Hebbian part only, x[t] - lam * x[t-1] with lam read from the
checkpoint, so that comparing them with ``mod`` and ``eff_mod`` shows how much
of the correlation the decay term accounts for. The hidden state has no decay
term and keeps the one definition.

For this analysis only, every task's first stimulus is aligned to the same
requested 500-ms Context duration (rounded down to whole simulation steps).
The shared task generator retains its
original variable-timing behavior unless this script explicitly opts in.

Outputs saved to ./state_space/. Without arguments the script clears that
directory's top-level files and analyzes the whole cohort; with
``--checkpoint PATH [PATH ...]`` it analyzes only those runs and merges their
entries into the results pickle without clearing anything, and ``--out-dir``
redirects every output (for example to test one run in a scratch directory).
``--untrained --checkpoint PATH`` instead runs the context-end PCA on
untrained networks with that run's architecture and config, one per
``--init-seed`` (fresh weights, default seed ``UNTRAINED_INIT_SEED``; eta at
its initial value), on the same evaluation trials, writing only their PCA
caches and diagnostic figures to ``<out-dir>/untrained/`` with ``_init<seed>``
in the names; the batch cleanup preserves that directory.

The same PCA caches retain original-feature task centroids and trial counts.
After analysis, run ``python paper_plot.py --only state_space_combined`` to
rank L2=1e-4 seeds by category-balanced high-dimensional centroid separation,
save the per-seed ranking/summary, and display the best seed's PCA panels.
Display PCA is incremental; quantitative scoring uses the full task centers.

By default, the batch entry point analyzes only the paper cohorts with tanh
activation, a 300-dimensional input projection, a 300-dimensional plastic
hidden layer, and L2 regularization 1e-5, 1e-4, 1e-3, or 1e-2 (feature tags
``L21e5``, ``L21e4``, ``L21e3``, and ``L21e2`` respectively).
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pickle
import gc

import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], 
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,   
    "ps.fonttype": 42,
})

from scipy.spatial.distance import cdist
from sklearn.decomposition import IncrementalPCA

import torch

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import mpn
import mpn_tasks
import color_func
import helper
import multiple_task_performance as mpf
from state_space_geometry import mantel_pearson_permutation, pair_values_to_matrix

c_vals = color_func.rainbow_generate(15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Default paper cohort.  ``linear_embed`` is the input width of the plastic
# layer (called the projection dimension elsewhere in the repository), while
# n_neurons[1] is its output/hidden width.
TARGET_ACTIVATION = "tanh"
TARGET_PROJECTION_DIM = 300
TARGET_HIDDEN_DIM = 300
FIXED_FIXATION_MS = 500
EVALUATION_SEED = 42
PCA_BATCH_SIZE = 16
TARGET_FEATURE_REG_LAMBDAS = {
    "L21e5": 1e-5,
    "L21e4": 1e-4,
    "L21e3": 1e-3,
    "L21e2": 1e-2,
}
STATE_SPACE_DIR = Path("state_space")
RESULTS_NAME = "initial_condition_distance_vs_angle_results.pkl"
NOISE_LEVEL = 0.01

# Distance-angle analysis. Each representation compares the task pairs'
# initial conditions x[t-1] (end of Context) with their first post-stimulus
# step. STEP_FULL is x[t] - x[t-1]; STEP_HEBBIAN is x[t] - lam * x[t-1], the
# modulation update without its decay term (see the module docstring), and
# applies to M and W * M only, since the hidden state has no decay term.
STEP_FULL = "full"
STEP_HEBBIAN = "hebbian"
# (saved name, source representation, step definition)
DISTANCE_ANGLE_REPRESENTATIONS = (
    ("hidden", "hidden", STEP_FULL),
    ("mod", "mod", STEP_FULL),
    ("eff_mod", "eff_mod", STEP_FULL),
    ("mod_hebb", "mod", STEP_HEBBIAN),
    ("eff_mod_hebb", "eff_mod", STEP_HEBBIAN),
)
N_STIMULUS_DIRECTIONS = 8
# Task pairs share tasks, so r is tested by permuting the task labels
# (Mantel-type, two-sided), as the lesion analyses do for cluster pairs.
DISTANCE_ANGLE_N_PERMUTATIONS = 10000
DISTANCE_ANGLE_PERMUTATION_SEED = 0
# Results entries carrying the permutation test and the Hebbian-only steps.
DISTANCE_ANGLE_SCHEMA_VERSION = 2

# Evaluation trials per task (one batch of every rule).
TRIALS_PER_RULE = 50

# Untrained control (eval_untrained): a fresh initialization of a run's
# architecture and config, drawn after seeding with UNTRAINED_INIT_SEED, whose
# PCA cache goes to <output dir>/UNTRAINED_DIR_NAME/.
UNTRAINED_DIR_NAME = "untrained"
UNTRAINED_INIT_SEED = 0

# rule -> (paper-style computation category, plotting color)
# https://www.nature.com/articles/s41593-024-01668-6/figures/4
# Hexes mirrored from paper_plot._RULE_MOTIF (the paper-side source of
# truth) so analysis and paper figures agree. The dms pair keeps the
# go/anti pairing deliberately: dmsgo shares Pro Reaction's green and
# dmsnogo shares Anti Reaction's orange (match/non-match is a pro/anti
# response rule); only the dmc pair takes Categorization's own deeppink.
RULE_MOTIF_MAPPING = {
    "fdgo":           ("Pro Delayed",    "#3182ce"),  # blue
    "fdanti":         ("Anti Delayed",   "#e53e3e"),  # red
    "delaygo":        ("Pro Delayed",    "#3182ce"),
    "delayanti":      ("Anti Delayed",   "#e53e3e"),
    "reactgo":        ("Pro Reaction",   "#38a169"),  # green
    "reactanti":      ("Anti Reaction",  "#dd6b20"),  # orange

    "contextdelaydm1": ("Pro Integration", "#805ad5"),  # purple
    "contextdelaydm2": ("Pro Integration", "#805ad5"),
    "delaydm1":        ("Pro Integration", "#805ad5"),
    "delaydm2":        ("Pro Integration", "#805ad5"),
    "multidelaydm":    ("Pro Integration", "#805ad5"),

    "dmsgo":          ("Categorization", "#38a169"),  # green, pairs reactgo
    "dmsnogo":        ("Categorization", "#dd6b20"),  # orange, pairs reactanti
    "dmcgo":          ("Categorization", "#ff1493"),  # deeppink
    "dmcnogo":        ("Categorization", "#ff1493"),
}
assert len({color for _, color in RULE_MOTIF_MAPPING.values()}) == 6

# The six categories of the diagnostic PCA figures, one per color above.
CATEGORY_STYLES = [
    ("Pro Delayed", "#3182ce"),
    ("Anti Delayed", "#e53e3e"),
    ("Pro Reaction / Match", "#38a169"),
    ("Anti Reaction / Non-match", "#dd6b20"),
    ("Integration", "#805ad5"),
    ("Category", "#ff1493"),
]


def _checkpoint_aname(checkpoint_path):
    """Return the run identifier encoded by a savednet checkpoint path."""
    stem = Path(checkpoint_path).stem
    prefix = "savednet_"
    if not stem.startswith(prefix):
        raise ValueError(f"Expected a {prefix}*.pt checkpoint, got {checkpoint_path}")
    return stem[len(prefix):]


def _raw_config_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    aname = _checkpoint_aname(checkpoint_path)
    return checkpoint_path.with_name(f"param_{aname}_param.json")


def _matches_target_cohort(checkpoint_path):
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


def _select_target_checkpoints(checkpoint_paths):
    """Filter checkpoint paths to the default cohort, preserving input order."""
    return [path for path in checkpoint_paths if _matches_target_cohort(path)]


def _clean_state_space_results(output_dir=STATE_SPACE_DIR):
    """Remove existing result files before regenerating the selected cohort.

    Subdirectories are deliberately preserved: the state-space pipeline owns
    the files at the top level of its output directory, but should not recurse
    into a directory another workflow may have placed there.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    removed = []
    for path in output_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed.append(path.name)
    print(f"Cleared {len(removed)} existing file(s) from {output_dir}/")
    return removed


def _context_pca_record(states, rule_labels, n_rules, batch_size=PCA_BATCH_SIZE):
    """Return original-feature task means, counts and incremental display PCA."""
    states = np.asarray(states)
    rule_labels = np.asarray(rule_labels)
    if (states.ndim < 2 or rule_labels.shape != (states.shape[0],)
            or not np.issubdtype(rule_labels.dtype, np.integer)
            or states.shape[0] < 2 or n_rules < 1 or batch_size < 2
            or np.any(rule_labels < 0) or np.any(rule_labels >= n_rules)):
        raise ValueError("Need aligned task labels and at least two PCA samples per batch.")
    states = states.reshape(states.shape[0], -1)
    if states.shape[1] < 2:
        raise ValueError("Display PCA requires at least two original features.")
    counts = np.bincount(rule_labels, minlength=n_rules)
    if np.any(counts == 0):
        raise ValueError("Every task must have context-end samples.")

    sums = np.zeros((n_rules, states.shape[1]), dtype=np.float64)
    pca = IncrementalPCA(n_components=2, batch_size=batch_size)
    start = 0
    while start < states.shape[0]:
        stop = min(start + batch_size, states.shape[0])
        if states.shape[0] - stop == 1:
            stop += 1
        samples = np.array(states[start:stop], dtype=np.float32, copy=True)
        if not np.all(np.isfinite(samples)):
            raise ValueError("Context states must be finite.")
        labels = rule_labels[start:stop]
        for rule_index in np.unique(labels):
            sums[rule_index] += samples[labels == rule_index].sum(
                axis=0, dtype=np.float64)
        pca.partial_fit(samples)
        del samples
        start = stop

    projection = np.empty((states.shape[0], 2), dtype=np.float32)
    for start in range(0, states.shape[0], batch_size):
        stop = min(start + batch_size, states.shape[0])
        projection[start:stop] = pca.transform(states[start:stop])
    return {
        "X_2d": projection,
        "ctx_rule_labels": rule_labels.astype(int, copy=False),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "task_centroids": sums / counts[:, None],
        "task_trial_counts": counts,
        "centroid_space": "original_features",
        "pca_method": "IncrementalPCA",
        "pca_batch_size": int(batch_size),
    }


def _modulation_retention(state_dict, load_net_params, dt_ms):
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


def _first_step(data, trial_indices, onset, retention=None):
    """Initial conditions x[t-1] and first steps of the given trials, where t
    is the first stimulus step ``onset``. With ``retention`` (lam) the step is
    x[t] - lam * x[t-1], the modulation update without its decay term;
    without it the full step x[t] - x[t-1]. Both in float64."""
    initial = np.asarray(data[trial_indices, onset - 1, :], dtype=np.float64)
    following = np.asarray(data[trial_indices, onset, :], dtype=np.float64)
    coefficient = 1.0 if retention is None else float(retention)
    return initial, following - coefficient * initial


def _cross_pair_statistics(initial_1, initial_2, step_1, step_2):
    """Mean Euclidean distance between the initial conditions and mean angle
    (degrees) between the steps over every cross-task trial pair, skipping
    pairs in which either step is (numerically) zero. Returns None when no
    pair remains."""
    norms_1 = np.linalg.norm(step_1, axis=1)
    norms_2 = np.linalg.norm(step_2, axis=1)
    valid = (norms_1 >= 1e-12)[:, None] & (norms_2 >= 1e-12)[None, :]
    if not valid.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = (step_1 @ step_2.T) / np.outer(norms_1, norms_2)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    distances = cdist(initial_1, initial_2)
    return float(distances[valid].mean()), float(angles[valid].mean()), int(valid.sum())


def _task_pair_distance_angle(data, rule_indices, stimulus_labels, onset_by_rule,
                              n_rules, retention=None, n_stimuli=N_STIMULUS_DIRECTIONS):
    """Per task pair: the mean initial-condition distance and first-step angle,
    averaged over cross-task trial pairs within each shared stimulus index and
    then equally over the stimulus indices. Returns (pairs, distances,
    angles) with ``pairs`` the (idx1, idx2) task indices, idx1 < idx2; a pair
    without any shared stimulus is left out."""
    pairs, distances, angles = [], [], []
    for idx1 in range(n_rules):
        for idx2 in range(idx1 + 1, n_rules):
            per_stimulus = []
            for stimulus in range(n_stimuli):
                trials_1 = np.flatnonzero((stimulus_labels == stimulus) & (rule_indices == idx1))
                trials_2 = np.flatnonzero((stimulus_labels == stimulus) & (rule_indices == idx2))
                if trials_1.size == 0 or trials_2.size == 0:
                    continue
                initial_1, step_1 = _first_step(data, trials_1, onset_by_rule[idx1], retention)
                initial_2, step_2 = _first_step(data, trials_2, onset_by_rule[idx2], retention)
                statistics = _cross_pair_statistics(initial_1, initial_2, step_1, step_2)
                if statistics is not None:
                    per_stimulus.append(statistics[:2])
            if per_stimulus:
                pairs.append((idx1, idx2))
                distances.append(float(np.mean([entry[0] for entry in per_stimulus])))
                angles.append(float(np.mean([entry[1] for entry in per_stimulus])))
    return pairs, np.asarray(distances, dtype=float), np.asarray(angles, dtype=float)


def _step_norm_summary(data, rule_indices, onset_by_rule, n_rules, retention=None):
    """Mean trial norms of the initial condition and the first step and, with
    ``retention``, of the step's decay term (1 - lam) * x[t-1] and Hebbian
    term x[t] - lam * x[t-1]: how much of the full step the decay contributes."""
    sums = {"initial": 0.0, "full_step": 0.0}
    if retention is not None:
        sums.update(decay_term=0.0, hebbian_term=0.0)
    count = 0
    for rule_index in range(n_rules):
        trials = np.flatnonzero(rule_indices == rule_index)
        if trials.size == 0:
            continue
        initial, full_step = _first_step(data, trials, onset_by_rule[rule_index])
        sums["initial"] += np.linalg.norm(initial, axis=1).sum()
        sums["full_step"] += np.linalg.norm(full_step, axis=1).sum()
        if retention is not None:
            sums["decay_term"] += (1.0 - retention) * np.linalg.norm(initial, axis=1).sum()
            sums["hebbian_term"] += np.linalg.norm(
                full_step + (1.0 - retention) * initial, axis=1).sum()
        count += trials.size
    if count == 0:
        raise ValueError("No trials to summarize.")
    return {key: float(value / count) for key, value in sums.items()}


def _permutation_test_or_reason(pairs, distances, angles, n_rules):
    """The task-label permutation test of r over the task pairs, or the reason
    it was skipped (an incomplete pair set cannot be relabeled)."""
    expected = n_rules * (n_rules - 1) // 2
    if len(pairs) != expected:
        return None, f"only {len(pairs)} of {expected} task pairs share a stimulus"
    return mantel_pearson_permutation(
        pair_values_to_matrix(distances, pairs, n_rules),
        pair_values_to_matrix(angles, pairs, n_rules),
        n_permutations=DISTANCE_ANGLE_N_PERMUTATIONS,
        seed=DISTANCE_ANGLE_PERMUTATION_SEED), None


def distance_angle_analysis(source_data, rule_indices, stimulus_labels, onset_by_rule,
                            all_rules, retention):
    """Initial-condition distance vs first-step angle for every entry of
    DISTANCE_ANGLE_REPRESENTATIONS.

    ``source_data`` maps "hidden", "mod" and "eff_mod" to (trials x time x
    features) arrays. Returns (rval_dict, scatter_dict): rval_dict keeps the
    (Pearson r, slope, nominal OLS p) tuples per saved name; scatter_dict holds
    per saved name the task-pair coordinates with their task names, the OLS
    fit, the task-label permutation test of r, the step definition and the
    step-norm summary.
    """
    all_rules = list(all_rules)
    n_rules = len(all_rules)
    rval_dict, scatter_dict = {}, {}
    for name, source, step_kind in DISTANCE_ANGLE_REPRESENTATIONS:
        step_retention = retention if step_kind == STEP_HEBBIAN else None
        data = source_data[source]
        pairs, distances, angles = _task_pair_distance_angle(
            data, rule_indices, stimulus_labels, onset_by_rule, n_rules, step_retention)
        _, _, r_value, slope, intercept, p_value = helper.linear_regression(
            distances, angles, log=False, through_origin=False)
        permutation, skipped = _permutation_test_or_reason(pairs, distances, angles, n_rules)
        rval_dict[name] = (r_value, slope, p_value)
        scatter_dict[name] = {
            "dists": distances,
            "angles_deg": angles,
            "pair_rules": [(all_rules[idx1], all_rules[idx2]) for idx1, idx2 in pairs],
            "regression": {
                "through_origin": False,
                "intercept": float(intercept),
                "p_value_method": "OLS slope t-test assuming independent task pairs",
            },
            "permutation": permutation,
            "permutation_skipped": skipped,
            "step": {
                "kind": step_kind,
                "source": source,
                "retention": step_retention,
                "definition": ("x[t] - x[t-1]" if step_kind == STEP_FULL
                               else "x[t] - lam * x[t-1] (decay term removed)"),
            },
            # The decay/Hebbian split describes the source's full step, so it
            # is attached to the modulation representations only.
            "step_norms": _step_norm_summary(
                data, rule_indices, onset_by_rule, n_rules,
                retention if source != "hidden" else None),
        }
    return rval_dict, scatter_dict


def _draw_distance_angle_figure(scatter_dict, rval_dict, out_path):
    """Per-run diagnostic: one panel per representation with the OLS guide,
    r, the nominal p and the permutation p."""
    names = list(scatter_dict)
    fig, axs = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    for ax, name in zip(np.atleast_1d(axs), names):
        entry = scatter_dict[name]
        r_value, slope, p_value = rval_dict[name]
        x = entry["dists"]
        ax.scatter(x, entry["angles_deg"], color="gray", alpha=0.5)
        permutation = entry["permutation"]
        perm_text = (f"permutation p={permutation['p_value']:.4f}" if permutation
                     else f"permutation skipped: {entry['permutation_skipped']}")
        if x.size >= 2:
            x_fit = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_fit, entry["regression"]["intercept"] + slope * x_fit, color="red",
                    label=(f"Fit: slope={slope:.2f}, r={r_value:.2f}, nominal p={p_value:.3f}\n"
                           f"{perm_text}"))
        ax.set_xlabel("Distance between initial conditions")
        ax.set_ylabel("Angle between first step of trajectories (deg.)")
        ax.set_title(f"{name} ({entry['step']['definition']})")
        ax.legend(frameon=True, loc="best", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _build_network(checkpoint, untrained_init_seed=None):
    """The network to analyze and the state dict its W and lam are read from.

    By default the checkpoint's trained weights are loaded. With
    ``untrained_init_seed`` the same architecture and config
    (``checkpoint["net_params"]``) is initialized afresh after seeding
    NumPy's and torch's global generators with it, and no weights are loaded:
    an untrained network whose eta keeps its initial value. This is not the
    training run's own initialization, which was never saved and cannot be
    replayed, since training drew other random numbers before building its
    network.
    """
    if untrained_init_seed is not None:
        np.random.seed(untrained_init_seed)
        torch.manual_seed(untrained_init_seed)
    model = mpn.DeepMultiPlasticNet(checkpoint["net_params"], verbose=False, forzihan=True)
    if untrained_init_seed is None:
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
        print("missing:", missing)
        print("unexpected:", unexpected)
        return model, checkpoint["state_dict"]
    return model, {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _context_states(netpathname, untrained_init_seed=None):
    """Run one checkpoint's network on the evaluation trials and return what
    the analyses need, as a dict: the run's identifiers (aname, hidden_size,
    l2_info), lam (``retention``), dt, the fixed Context length, the accuracy,
    the task names, each trial's task index and first-stimulus label, the
    task epochs, and the (trials x time x features) trajectories of the
    hidden state, M and W * M. With ``untrained_init_seed`` the network is
    untrained (see _build_network); the evaluation trials do not depend on
    it, since they are drawn after reseeding with EVALUATION_SEED. Full
    hidden and modulation trajectories are retained on CPU.
    """
    hidden_size, l2_info = mpf.parse_hidden_and_l2(netpathname)

    netpathname = Path(netpathname)
    aname = _checkpoint_aname(netpathname)
    with _raw_config_path(netpathname).open() as f:
        raw_cfg_param = json.load(f)

    task_params, train_params, net_params = raw_cfg_param["task_params"], raw_cfg_param["train_params"], raw_cfg_param["net_params"]

    # Keep the checkpoint copy on CPU; only the live model needs to occupy GPU
    # memory.  This also makes the frozen W used below immediately NumPy-safe.
    checkpoint = torch.load(netpathname, map_location="cpu", weights_only=False)
    load_net_params = checkpoint["net_params"]
    print(checkpoint["state_dict"].keys())
    print(load_net_params)

    model, state_dict = _build_network(checkpoint, untrained_init_seed)
    model.to(device)
    model.eval()

    task_params["sigma_x"] = NOISE_LEVEL
    np.random.seed(EVALUATION_SEED)
    torch.manual_seed(EVALUATION_SEED)

    task_params_c, _, _ = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )

    all_tasks = task_params_c['rules']

    # setup the evaluation dataset generator
    task_params_c['hp']['batch_size_train'] = TRIALS_PER_RULE
    task_params_c['hp']['seed'] = EVALUATION_SEED
    task_params_c['hp']['rng'] = np.random.RandomState(EVALUATION_SEED)
    dt_ms = task_params_c['hp']['dt']
    retention = _modulation_retention(state_dict, load_net_params, dt_ms)
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

    test_input = test_input.to(device)
    test_output = test_output.to(device)
    test_mask = test_mask.to(device)

    with torch.no_grad():
        # Accuracy needs only the outputs, so compute it without retaining the
        # enormous M trajectory on CUDA.
        net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode="minimal")
        acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input, isvalid=True, mode=model.acc_measure)
        del net_out

        # A 750 x ~118 x 300 x 300 float32 M trace is about 30 GiB.  Save each
        # time step directly to CPU so the selected 300x300 cohort can run on a
        # conventional GPU.  detach_saved is explicit even under no_grad so the
        # storage contract remains clear if this block is later refactored.
        tracked_out, tracked_hidden, db_test = model.iterate_sequence_batch(
            test_input,
            run_mode="track_states",
            save_to_cpu=True,
            detach_saved=True,
        )
        del tracked_out, tracked_hidden

    print(f"acc: {acc:.2f}")

    # db_test is already on CPU.  NumPy views avoid another full copy of M.
    Ms_orig = db_test["M1"].numpy()
    modulation_W = state_dict["mp_layer1.W"].numpy()
    eff_Ms_orig = Ms_orig * modulation_W

    Ms = Ms_orig.reshape(Ms_orig.shape[0], Ms_orig.shape[1], -1)
    eff_Ms = eff_Ms_orig.reshape(eff_Ms_orig.shape[0], eff_Ms_orig.shape[1], -1)
    hs = db_test["hidden1"].numpy()
    print(f"Ms_orig.shape: {Ms_orig.shape}; Ms.shape: {Ms.shape}; hs.shape: {hs.shape}")

    # All remaining work is NumPy/scikit-learn.  Release CUDA storage before
    # the comparatively long PCA and pairwise geometry calculations.
    del model, checkpoint, state_dict, db_test
    del test_input, test_output, test_mask, test_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _, labels_stim1, _, rules_epochs = helper.generate_response_stimulus(task_params_c, test_trials)
    all_rules = list(task_params_c['rules'])
    assert set(all_rules).issubset(set(RULE_MOTIF_MAPPING.keys()))
    del test_trials_extra, test_trials

    return {
        "aname": aname,
        "hidden_size": hidden_size,
        "l2_info": l2_info,
        "retention": retention,
        "dt_ms": dt_ms,
        "fixed_fixation_steps": fixed_fixation_steps,
        "accuracy": float(acc),
        "all_rules": all_rules,
        "rule_indices": test_rule_idxs,
        "stimulus_labels": labels_stim1.flatten(),
        "rules_epochs": rules_epochs,
        "hidden": hs,
        "mod": Ms,
        "eff_mod": eff_Ms,
    }


def _draw_context_pca_figure(X_2d, ctx_rule_labels, all_rules, out_path):
    """Per-run diagnostic of one representation's context-end PCA: colored
    by task (left) and by the six paper computation categories (right)."""
    fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 4))

    for idx, rule in enumerate(all_rules):
        sel = (ctx_rule_labels == idx)
        ctx_values = X_2d[sel]
        axs[0].scatter(ctx_values[:, 0], ctx_values[:, 1], label=rule,
                       color=c_vals[idx], alpha=0.5, s=18)
    axs[0].set_title("Colored by rule")
    axs[0].set_xlabel("Context endpoint state PC1")
    axs[0].set_ylabel("Context endpoint state PC2")
    axs[0].legend(frameon=True, loc='best', fontsize=6)

    for category, color in CATEGORY_STYLES:
        rule_idxs_in_cat = [idx for idx, rule in enumerate(all_rules)
                            if RULE_MOTIF_MAPPING[rule][1] == color]
        sel = np.isin(ctx_rule_labels, rule_idxs_in_cat)
        ctx_values = X_2d[sel]
        axs[1].scatter(ctx_values[:, 0], ctx_values[:, 1], label=category,
                       color=color, alpha=0.5, s=18)
    axs[1].set_title("Colored by computation category")
    axs[1].set_xlabel("Context endpoint state PC1")
    axs[1].set_ylabel("Context endpoint state PC2")
    axs[1].legend(frameon=True, loc='best', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def untrained_file_stem(aname, init_seed):
    """File-name stem of an untrained network built from run ``aname``'s
    config with weights from ``init_seed``: one cache per seed, so several
    initializations of one config never overwrite each other."""
    return f"{aname}_init{int(init_seed)}"


def _save_context_pca(states, output_dir, extra=None, file_stem=None):
    """Context-end PCA of the hidden state, M and W * M of one ``states``
    dict (``_context_states``): a diagnostic figure per representation and
    the PCA cache paper_plot reads, both in ``output_dir`` and named after
    ``file_stem`` (default: the run's aname). ``extra`` adds or overrides
    cache fields. Returns the cache path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aname, all_rules = states["aname"], states["all_rules"]
    file_stem = aname if file_stem is None else file_stem
    rule_indices, rules_epochs = states["rule_indices"], states["rules_epochs"]

    pca_results = {}
    for data_name in ("hidden", "mod", "eff_mod"):
        print(f"Processing {data_name}...")
        data = states[data_name]
        ctx_endfix = []
        ctx_rule_labels = []
        for idx, rule in enumerate(all_rules):
            ctx_endtime = rules_epochs[rule]['fix1'][1]
            context_end = data[rule_indices == idx, ctx_endtime - 1, :]
            ctx_endfix.append(context_end)
            ctx_rule_labels.append(np.full(context_end.shape[0], idx))
        ctx_extract = np.concatenate(ctx_endfix, axis=0)
        ctx_rule_labels = np.concatenate(ctx_rule_labels, axis=0)

        pca_results[data_name] = _context_pca_record(ctx_extract, ctx_rule_labels, len(all_rules))
        _draw_context_pca_figure(
            pca_results[data_name]["X_2d"], ctx_rule_labels, all_rules,
            output_dir / f"state_space_shift_{file_stem}_{data_name}_noise{NOISE_LEVEL}.png")

    cache = {
        "pca_results": pca_results,
        "all_rules": list(all_rules),
        "rule_motif_mapping": RULE_MOTIF_MAPPING,
        "noise_level": NOISE_LEVEL,
        "evaluation_seed": EVALUATION_SEED,
        "trials_per_rule": TRIALS_PER_RULE,
        "l2_info": float(states["l2_info"]),
        "fixed_fixation_steps": states["fixed_fixation_steps"],
        "fixed_fixation_ms": states["fixed_fixation_steps"] * states["dt_ms"],
        "aname": aname,
        "accuracy": states["accuracy"],
        "untrained": False,
    }
    cache.update(extra or {})
    pca_save_path = output_dir / f"state_space_pca_{file_stem}_noise{NOISE_LEVEL}.pkl"
    with open(pca_save_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved PCA data to {pca_save_path}")
    return pca_save_path


def eval_one(netpathname, output_dir=None):
    """Analyze one checkpoint's Context geometry and first-step directions.

    Save PCA coordinates, high-dimensional task centers and diagnostic plots
    under ``output_dir`` (STATE_SPACE_DIR by default).
    Return (aname, hidden_size, l2_info, rval_dict, scatter_dict,
    fixed_fixation_steps, fixed_fixation_ms, retention): the distance-angle
    results of ``distance_angle_analysis`` and the checkpoint's lam.
    Full hidden and modulation trajectories are retained on CPU during analysis.
    """
    output_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir)
    states = _context_states(netpathname)
    _save_context_pca(states, output_dir)
    aname, all_rules = states["aname"], states["all_rules"]

    # Distance between initial conditions vs. angle between first steps
    # (closer to Fig 4C of Driscoll et al.), for the full and Hebbian steps.
    onset_by_rule = {idx: int(states["rules_epochs"][rule]['fix1'][1])
                     for idx, rule in enumerate(all_rules)}
    source_data = {name: states[name] for name in ("hidden", "mod", "eff_mod")}
    rval_dict, scatter_dict = distance_angle_analysis(
        source_data, states["rule_indices"], states["stimulus_labels"], onset_by_rule,
        all_rules, states["retention"])
    _draw_distance_angle_figure(
        scatter_dict, rval_dict,
        output_dir / f"initial_condition_distance_vs_angle_{aname}_1_noise{NOISE_LEVEL}.png")
    for name, entry in scatter_dict.items():
        permutation = entry["permutation"]
        text = (f"permutation p = {permutation['p_value']:.4f}" if permutation
                else f"permutation skipped ({entry['permutation_skipped']})")
        print(f"  {name}: r = {rval_dict[name][0]:.3f}, {text}; "
              f"step norms {entry['step_norms']}")

    result = (aname, states["hidden_size"], states["l2_info"], rval_dict, scatter_dict,
              states["fixed_fixation_steps"], states["fixed_fixation_steps"] * states["dt_ms"],
              states["retention"])
    # Cleanup to prevent CPU memory compounding across experiments.
    del states, source_data
    gc.collect()
    return result


def eval_untrained(netpathname, output_dir=None, init_seed=UNTRAINED_INIT_SEED):
    """Context-end PCA of an untrained network with the architecture and
    config of the checkpoint ``netpathname``: fresh weights drawn after
    seeding with ``init_seed``, none of the trained weights used (see
    _build_network), run on the same evaluation trials as the trained
    analysis.

    Only the PCA cache and its diagnostic figures are written, to
    ``<output_dir>/untrained/`` (output_dir defaults to STATE_SPACE_DIR),
    named ``untrained_file_stem(aname, init_seed)`` so every seed keeps its
    own files. The
    subdirectory keeps the untrained cache out of the trained caches' readers
    (state_space_structure.py's glob, paper_plot's seed rankings), and the
    batch run's cleanup, which removes top-level files only, keeps it. The
    cache is marked ``untrained`` and records ``init_seed`` and the
    checkpoint it took its config from. Returns the cache path.
    """
    untrained_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir) / UNTRAINED_DIR_NAME
    states = _context_states(netpathname, untrained_init_seed=init_seed)
    path = _save_context_pca(states, untrained_dir, extra={
        "untrained": True,
        "init_seed": int(init_seed),
        "config_source": states["aname"],
    }, file_stem=untrained_file_stem(states["aname"], init_seed))
    del states
    gc.collect()
    return path


def _result_entry(evaluation):
    """The results-pickle entry of one ``eval_one`` return value."""
    (aname, hidden_size, l2_info, rval_dict, scatter_dict,
     fixed_fixation_steps, fixed_fixation_ms, retention) = evaluation
    return aname, {
        "schema_version": DISTANCE_ANGLE_SCHEMA_VERSION,
        "hidden_size": hidden_size,
        "l2_info": l2_info,
        "rval_dict": rval_dict,
        "scatter": scatter_dict,
        "fixed_fixation_steps": fixed_fixation_steps,
        "fixed_fixation_ms": fixed_fixation_ms,
        "retention": retention,
        "n_permutations": DISTANCE_ANGLE_N_PERMUTATIONS,
        "permutation_seed": DISTANCE_ANGLE_PERMUTATION_SEED,
    }


def _results_path(output_dir=None):
    return Path(STATE_SPACE_DIR if output_dir is None else output_dir) / RESULTS_NAME


def _save_results(result_dict, output_dir=None):
    with _results_path(output_dir).open("wb") as f:
        pickle.dump(result_dict, f)


def run_all(output_dir=None):
    """Analyze the whole target cohort after clearing the top-level files of
    ``output_dir`` (STATE_SPACE_DIR by default). The results pickle is
    rewritten after every run, so an interrupted batch keeps what it finished.
    """
    output_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir)
    all_pt_paths = mpf.list_pt_files("./multiple_tasks", recursive=False)
    pt_paths = _select_target_checkpoints(all_pt_paths)
    target_features = ",".join(TARGET_FEATURE_REG_LAMBDAS)
    if not pt_paths:
        raise FileNotFoundError(
            "No state-space checkpoints matched "
            f"activation={TARGET_ACTIVATION}, projection={TARGET_PROJECTION_DIM}, "
            f"hidden={TARGET_HIDDEN_DIM}, features={target_features}."
        )

    print(
        f"Selected {len(pt_paths)}/{len(all_pt_paths)} state-space checkpoints: "
        f"activation={TARGET_ACTIVATION}, projection={TARGET_PROJECTION_DIM}, "
        f"hidden={TARGET_HIDDEN_DIM}, features={target_features}"
    )
    for path in pt_paths:
        print(f"  {_checkpoint_aname(path)}")

    # This script is the sole producer of top-level state_space result files.
    # The user requested a clean regeneration, so remove every previous file
    # only after confirming that the target cohort is nonempty.
    _clean_state_space_results(output_dir)

    result_dict = {}
    for netpathname in pt_paths:
        aname, entry = _result_entry(eval_one(netpathname, output_dir))
        result_dict[aname] = entry
        _save_results(result_dict, output_dir)
    return result_dict


def run_checkpoints(checkpoint_paths, output_dir=None):
    """Analyze only the given ``savednet_*.pt`` checkpoints and merge their
    entries into the results pickle of ``output_dir`` (STATE_SPACE_DIR by
    default), which is created if needed and never cleared. A checkpoint
    outside the default cohort is analyzed with a warning, since the caller
    named it explicitly."""
    output_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir)
    paths = _existing_checkpoints(checkpoint_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = _results_path(output_dir)
    result_dict = {}
    if results_path.exists():
        with results_path.open("rb") as f:
            result_dict = pickle.load(f)
    for path in paths:
        if not _matches_target_cohort(path):
            print(f"Warning: {path} is outside the default state-space cohort.")
        aname, entry = _result_entry(eval_one(path, output_dir))
        result_dict[aname] = entry
        _save_results(result_dict, output_dir)
        print(f"Merged {aname} into {results_path} ({len(result_dict)} run(s))")
    return result_dict


def _existing_checkpoints(checkpoint_paths):
    """The given paths as Paths, checked to be savednet_*.pt files."""
    paths = [Path(path) for path in checkpoint_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
    for path in paths:
        _checkpoint_aname(path)
    return paths


def run_untrained(checkpoint_paths, output_dir=None, init_seeds=(UNTRAINED_INIT_SEED,)):
    """``eval_untrained`` for each named checkpoint and each of ``init_seeds``:
    untrained networks with that run's architecture and config, one per
    seed, written to <output_dir>/untrained/. Nothing else in ``output_dir``
    is touched. Returns the cache paths."""
    paths = _existing_checkpoints(checkpoint_paths)
    seeds = [int(seed) for seed in init_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Need one or more distinct init seeds, got {seeds}.")
    cache_paths = []
    for path in paths:
        if not _matches_target_cohort(path):
            print(f"Warning: {path} is outside the default state-space cohort.")
        for seed in seeds:
            cache_paths.append(eval_untrained(path, output_dir, seed))
    return cache_paths


def summarize(output_dir=None):
    """Print per-representation r, nominal p and permutation p over the
    analyzed runs, the decay term's share of the full modulation step, and
    save a bar chart of the mean r (summary_r_values.png)."""
    output_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir)
    with _results_path(output_dir).open("rb") as f:
        result_dict = pickle.load(f)

    names = [name for name, _, _ in DISTANCE_ANGLE_REPRESENTATIONS]
    collected = {name: {"r": [], "slope": [], "p_nominal": [], "p_perm": []}
                 for name in names}
    decay_share = {"mod": [], "eff_mod": []}
    for results in result_dict.values():
        scatter = results.get("scatter", {})
        for name in names:
            if name not in results["rval_dict"]:
                continue
            r_val, slope, p_val = results["rval_dict"][name]
            collected[name]["r"].append(r_val)
            collected[name]["slope"].append(slope)
            collected[name]["p_nominal"].append(p_val)
            permutation = scatter.get(name, {}).get("permutation")
            if permutation is not None:
                collected[name]["p_perm"].append(permutation["p_value"])
        for source in decay_share:
            norms = scatter.get(source, {}).get("step_norms")
            if norms and "decay_term" in norms:
                decay_share[source].append(norms["decay_term"] / norms["full_step"])

    print("=" * 80)
    print(f"Summary of {len(result_dict)} networks")
    print("=" * 80)
    present = [name for name in names if collected[name]["r"]]
    for name in present:
        entry = collected[name]
        r_arr = np.array(entry["r"])
        slope_arr = np.array(entry["slope"])
        p_arr = np.array(entry["p_nominal"])
        print(f"\n{name.upper()}:")
        print(f"  R-values: mean={np.mean(r_arr):.3f}, std={np.std(r_arr):.3f}, "
              f"min={np.min(r_arr):.3f}, max={np.max(r_arr):.3f}")
        print(f"  Slopes:   mean={np.mean(slope_arr):.3f}, std={np.std(slope_arr):.3f}, "
              f"min={np.min(slope_arr):.3f}, max={np.max(slope_arr):.3f}")
        print(f"  Nominal P-values (task-pair independence assumed): "
              f"mean={np.mean(p_arr):.4f}, p<0.05: {np.sum(p_arr < 0.05)}/{len(p_arr)}")
        if entry["p_perm"]:
            p_perm = np.array(entry["p_perm"])
            print(f"  Task-label permutation P-values: median={np.median(p_perm):.4f}, "
                  f"p<0.05: {np.sum(p_perm < 0.05)}/{len(p_perm)}")
    for source, shares in decay_share.items():
        if shares:
            print(f"\n{source}: decay term / full first step (mean trial norms): "
                  f"mean={np.mean(shares):.3f}, min={np.min(shares):.3f}, "
                  f"max={np.max(shares):.3f}")

    if not present:
        print("\nNo distance-angle results to plot.")
        return
    source_colors = {"hidden": "skyblue", "mod": "lightcoral", "eff_mod": "lightgreen"}
    steps = {name: (source, step) for name, source, step in DISTANCE_ANGLE_REPRESENTATIONS}
    fig, ax = plt.subplots(1, 1, figsize=(1.5 + 0.8 * len(present), 4))
    positions = np.arange(len(present))
    r_means = [np.mean(collected[name]["r"]) for name in present]
    r_stds = [np.std(collected[name]["r"], ddof=1) if len(collected[name]["r"]) > 1 else 0.0
              for name in present]
    bars = ax.bar(positions, r_means, yerr=r_stds, capsize=5,
                  color=[source_colors[steps[name][0]] for name in present],
                  edgecolor="k", linewidth=0.6)
    for bar, name in zip(bars, present):
        if steps[name][1] == STEP_HEBBIAN:
            bar.set_hatch("///")
    ax.set_xticks(positions)
    ax.set_xticklabels(present, rotation=45, ha='right')
    ax.set_ylabel('R-value')
    ax.set_title('Mean R-values across networks\n(hatched: Hebbian term only)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_r_values.png", dpi=300)
    plt.close(fig)
    print(f"\nSaved summary plot to {output_dir / 'summary_r_values.png'}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="State-space geometry of the multi-task runs (see the module docstring).")
    parser.add_argument(
        "--checkpoint", nargs="+", metavar="PATH", default=None,
        help="analyze only these savednet_*.pt checkpoints and merge them into the "
             "results pickle of --out-dir without clearing it (default: the whole "
             "cohort, after clearing the top-level files of --out-dir)")
    parser.add_argument(
        "--out-dir", default=str(STATE_SPACE_DIR), metavar="DIR",
        help=f"directory for every output (default: {STATE_SPACE_DIR}/)")
    parser.add_argument(
        "--untrained", action="store_true",
        help="instead analyze an untrained network with the architecture and config of "
             "each --checkpoint (fresh weights from --init-seed, no trained weight used): "
             f"context-end PCA only, written to <out-dir>/{UNTRAINED_DIR_NAME}/")
    parser.add_argument(
        "--init-seed", type=int, nargs="+", default=None, metavar="SEED",
        help=f"seeds of the untrained weights, one network per seed (default: "
             f"{UNTRAINED_INIT_SEED}); only with --untrained")
    args = parser.parse_args(argv)
    if args.untrained and not args.checkpoint:
        parser.error("--untrained needs --checkpoint to name the run whose config it uses")
    if args.init_seed is not None and not args.untrained:
        parser.error("--init-seed only applies with --untrained")
    if args.init_seed is None:
        args.init_seed = [UNTRAINED_INIT_SEED]
    return args


def main(argv=None):
    args = _parse_args(argv)
    output_dir = Path(args.out_dir)
    if args.untrained:
        run_untrained(args.checkpoint, output_dir, args.init_seed)
        return
    if args.checkpoint:
        run_checkpoints(args.checkpoint, output_dir)
    else:
        run_all(output_dir)
    summarize(output_dir)


if __name__ == "__main__":
    main()
