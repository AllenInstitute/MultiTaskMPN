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
    With --steps N, repeat for local updates at stimulus steps 1..N, keeping
    the first step's context-end distances and task-pair set fixed. This is
    not the cumulative displacement from context end. The window must stay
    within every task's first post-context phase.

Time alignment. The network records M before it is updated
(core/mpn.py: ``network_step`` runs ``forward``, which logs the M that
processes x[t], and updates M afterwards), while the recorded hidden state
h[t] is the response to x[t]. The recorded M[t] is therefore the state
before the update that x[t] drives. Both analyses need M on the hidden
state's clock, so ``align_modulation_to_hidden`` shifts the recorded M and
W ⊙ M trajectories by one step (M[t] := the state after x[t]'s update) and
drops the last step of every representation, which has no recorded
successor. Caches and results entries carry ``m_alignment`` (``M_ALIGNMENT``)
and schema version ``DISTANCE_ANGLE_SCHEMA_VERSION``; readers skip earlier
caches, whose modulation context states and first steps were one step early.

The PCA runs on three representations: hidden states, raw modulation M, and
effective modulation (W ⊙ M). The distance-angle analysis adds, for M and
W ⊙ M, a second step definition. The modulation update is
M[t] = lam * M[t-1] + eta * post (x) pre, clamped to the M bounds
(core/mpn.py, update_M_matrix), so the full first step M[t] - M[t-1] contains
-(1 - lam) * M[t-1]: a term along the initial condition itself, which relates
the step angle to the initial-condition distance without any task-dependent
computation. The ``mod_hebb`` and ``eff_mod_hebb`` representations therefore
use the step with the linear decay removed, x[t] - lam * x[t-1] with lam read
from the checkpoint, so that comparing them with ``mod`` and ``eff_mod`` shows
how much of the correlation the decay term accounts for. That remainder is
not the Hebbian outer product alone: the bound clipping and the M activation
of the update stay in it, so figures and messages name it by what is removed
("decay removed"); the saved keys keep their names. The hidden state has no
decay term and keeps the one definition.

For this analysis only, every task's first stimulus is aligned to the same
Context duration, ``--context-ms`` (default FIXED_FIXATION_MS, 480 ms; rounded
down to whole simulation steps). The shared task generator retains its
original variable-timing behavior unless this script explicitly opts in.

Outputs saved to ./state_space/. Without arguments the script clears that
directory's top-level files and analyzes the whole cohort; with
``--checkpoint PATH [PATH ...]`` it analyzes only those runs and merges their
entries into the results pickle without clearing anything, and ``--out-dir``
redirects every output (for example to test one run in a scratch directory).
A Context length other than the default writes to ``state_space/context<ms>ms/``
instead (``context_dir_of``), with the same layout, so the lengths never share
a results pickle; ``paper_plot.py --only state_space_r_vs_context`` draws the
r of every length found.
``--steps 5 --skip-pca`` adds a five-step timecourse, recording only Context,
the requested stimulus window and one extra step for M alignment. It merges
results even in cohort mode, preserving PCA and derived caches; full-trial
accuracy is not evaluated. Existing top-level statistics remain the first step.
``--untrained --checkpoint PATH`` instead runs the context-end PCA and the
distance-angle analysis on untrained networks with that run's architecture
and config, one per ``--init-seed`` (fresh weights, default seed
``UNTRAINED_INIT_SEED``; eta at its initial value), on the same evaluation
trials: the no-learning control of both analyses. Everything goes to
``<out-dir>/untrained/``: PCA caches and diagnostic figures with
``_init<seed>`` in the names, and the networks' own distance-angle results
pickle (entries keyed by those stems, ``l2_info`` None); the batch cleanup
preserves that directory.

The same PCA caches retain original-feature task centroids and trial counts.
After analysis, run ``python figures/paper_plot.py --only state_space_combined`` to
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
import color_func
import helper
import multiple_task_performance as mpf
import state_space_evaluation as evaluation_utils
from state_space_evaluation import (
    STATE_SPACE_DIR, NOISE_LEVEL, EVALUATION_SEED, FIXED_FIXATION_MS, TRIALS_PER_RULE,
    TARGET_ACTIVATION, TARGET_PROJECTION_DIM, TARGET_HIDDEN_DIM, TARGET_FEATURE_REG_LAMBDAS,
    UNTRAINED_DIR_NAME, UNTRAINED_INIT_SEED, untrained_file_stem, context_dir_of,
)
from state_space_geometry import (
    distance_angle_statistics, DISTANCE_ANGLE_N_PERMUTATIONS, DISTANCE_ANGLE_PERMUTATION_SEED,
)
from state_space_cache import M_ALIGNMENT, DISTANCE_ANGLE_SCHEMA_VERSION, distance_angle_is_current

c_vals = color_func.rainbow_generate(15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


PCA_BATCH_SIZE = 16
RESULTS_NAME = "initial_condition_distance_vs_angle_results.pkl"

# Distance-angle analysis. Each representation compares the task pairs'
# initial conditions x[t-1] (end of Context) with their first post-stimulus
# step. STEP_FULL is x[t] - x[t-1]; STEP_HEBBIAN is x[t] - lam * x[t-1], the
# modulation update with its linear decay removed (clipping and M activation
# included, see the module docstring; the saved kind string is kept), and
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


def align_modulation_to_hidden(hidden, *modulations):
    """Put recorded modulation trajectories on the hidden state's clock.

    ``hidden`` is (trials, time, ...) with hidden[t] the response to input
    x[t]; each of ``modulations`` is (trials, time, ...) as the network
    records it, with M[t] the state that processed x[t], before the update
    x[t] drives (core/mpn.py, network_step). Returns ``hidden`` followed by
    the modulations shifted by one step, so that index t of every array is
    the state after x[t]: hidden[:, :-1] and M[:, 1:]. The last recorded
    step has no successor and is dropped from every array (as views, no
    copy). Raises ValueError when the lengths differ or fewer than two steps
    were recorded.
    """
    n_steps = hidden.shape[1]
    if n_steps < 2:
        raise ValueError("Aligning M to the hidden state needs at least two recorded steps.")
    for modulation in modulations:
        if modulation.shape[1] != n_steps:
            raise ValueError("The hidden state and M were recorded over different step counts.")
    return (hidden[:, :-1],) + tuple(modulation[:, 1:] for modulation in modulations)


def _step_vectors(data, trial_indices, onset, retention=None, step=1, with_initial=True):
    """Context-end initial states and the update at stimulus step `step`.
    Arrays use the aligned, post-input clock. The initial state always comes
    from onset - 1; only the update moves forward. With retention, remove
    this step's linear decay, using its immediate predecessor. Without
    ``with_initial`` the initial state is None: later steps keep the first
    step's distances and need only the update."""
    if step < 1 or onset < 1 or onset + step > data.shape[1]:
        raise ValueError("Requested stimulus step is outside the aligned trajectory.")
    initial = (np.asarray(data[trial_indices, onset - 1, :], dtype=np.float64)
               if with_initial else None)
    previous = np.asarray(data[trial_indices, onset + step - 2, :], dtype=np.float64)
    following = np.asarray(data[trial_indices, onset + step - 1, :], dtype=np.float64)
    coefficient = 1.0 if retention is None else float(retention)
    return initial, following - coefficient * previous


def _cross_pair_statistics(initial_1, initial_2, step_1, step_2, compute_distance=True,
                           norms_1=None, norms_2=None):
    """Mean Euclidean distance between the initial conditions and mean angle
    (degrees) between the steps over every cross-task trial pair, skipping
    pairs in which either step is (numerically) zero. Returns None when no
    pair remains. ``norms_1`` and ``norms_2`` are the steps' trial norms when
    the caller has them already."""
    if norms_1 is None:
        norms_1 = np.linalg.norm(step_1, axis=1)
    if norms_2 is None:
        norms_2 = np.linalg.norm(step_2, axis=1)
    valid = (norms_1 >= 1e-12)[:, None] & (norms_2 >= 1e-12)[None, :]
    if not valid.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = (step_1 @ step_2.T) / np.outer(norms_1, norms_2)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    distance = float(cdist(initial_1, initial_2)[valid].mean()) if compute_distance else float("nan")
    return distance, float(angles[valid].mean()), int(valid.sum())


def _task_pair_distance_angle(data, rule_indices, stimulus_labels, onset_by_rule,
                              n_rules, retention=None, n_stimuli=N_STIMULUS_DIRECTIONS,
                              step=1, compute_distance=True):
    """Per task pair: the mean context-end distance and angle at stimulus step,
    averaged over cross-task trial pairs within each shared stimulus index and
    then equally over the stimulus indices. Returns (pairs, distances,
    angles) with ``pairs`` the (idx1, idx2) task indices, idx1 < idx2; a pair
    without any shared stimulus is left out."""
    # Every (task, stimulus) group of trials meets the same group of every
    # other task: extract its vectors and step norms once, not once per
    # partner (the extraction from the trials x time x features trace
    # dominated the runtime).
    groups = {}
    for rule_index in range(n_rules):
        for stimulus in range(n_stimuli):
            trials = np.flatnonzero((stimulus_labels == stimulus) & (rule_indices == rule_index))
            if trials.size == 0:
                continue
            initial, step_vectors = _step_vectors(data, trials, onset_by_rule[rule_index], retention,
                                                  step, with_initial=compute_distance)
            groups[(rule_index, stimulus)] = (initial, step_vectors,
                                              np.linalg.norm(step_vectors, axis=1))
    pairs, distances, angles = [], [], []
    for idx1 in range(n_rules):
        for idx2 in range(idx1 + 1, n_rules):
            per_stimulus = []
            for stimulus in range(n_stimuli):
                if (idx1, stimulus) not in groups or (idx2, stimulus) not in groups:
                    continue
                initial_1, step_1, norms_1 = groups[(idx1, stimulus)]
                initial_2, step_2, norms_2 = groups[(idx2, stimulus)]
                statistics = _cross_pair_statistics(initial_1, initial_2, step_1, step_2,
                                                    compute_distance=compute_distance,
                                                    norms_1=norms_1, norms_2=norms_2)
                if statistics is not None:
                    per_stimulus.append(statistics[:2])
            if per_stimulus:
                pairs.append((idx1, idx2))
                distances.append(float(np.mean([entry[0] for entry in per_stimulus])))
                angles.append(float(np.mean([entry[1] for entry in per_stimulus])))
    return pairs, np.asarray(distances, dtype=float), np.asarray(angles, dtype=float)


def _step_norm_summary(data, rule_indices, onset_by_rule, n_rules, retention=None, step=1):
    """Mean trial norms of the context-end initial state and the chosen step and, with
    ``retention``, of the step's decay term (1 - lam) * x[t-1] and of the
    remainder x[t] - lam * x[t-1] (saved as ``hebbian_term``: the Hebbian
    outer product with the clipping and M activation of that step): how much
    of the full step the decay contributes."""
    sums = {"initial": 0.0, "full_step": 0.0}
    if retention is not None:
        sums.update(decay_term=0.0, hebbian_term=0.0)
    count = 0
    for rule_index in range(n_rules):
        trials = np.flatnonzero(rule_indices == rule_index)
        if trials.size == 0:
            continue
        onset = onset_by_rule[rule_index]
        initial, full_step = _step_vectors(data, trials, onset, step=step)
        sums["initial"] += np.linalg.norm(initial, axis=1).sum()
        sums["full_step"] += np.linalg.norm(full_step, axis=1).sum()
        if retention is not None:
            previous = np.asarray(data[trials, onset + step - 2], dtype=np.float64)
            sums["decay_term"] += (1.0 - retention) * np.linalg.norm(previous, axis=1).sum()
            sums["hebbian_term"] += np.linalg.norm(
                full_step + (1.0 - retention) * previous, axis=1).sum()
        count += trials.size
    if count == 0:
        raise ValueError("No trials to summarize.")
    return {key: float(value / count) for key, value in sums.items()}


def distance_angle_analysis(source_data, rule_indices, stimulus_labels, onset_by_rule,
                            all_rules, retention, step=1, initial_scatter=None):
    """Context-end distance vs angle at a stimulus step for every entry of
    DISTANCE_ANGLE_REPRESENTATIONS.

    ``source_data`` maps "hidden", "mod" and "eff_mod" to (trials x time x
    features) arrays. Returns (rval_dict, scatter_dict): rval_dict keeps the
    (Pearson r, slope, nominal OLS p) tuples per saved name; scatter_dict holds
    per saved name the task-pair coordinates with their task names, the OLS
    fit, the task-label permutation test of r, the step definition and the
    step-norm summary. Later steps use `initial_scatter` to keep exactly the
    first step\'s distances and task pairs. If a later angle is undefined,
    its r is undefined rather than computed on a different set of task pairs.
    """
    all_rules = list(all_rules)
    n_rules = len(all_rules)
    rval_dict, scatter_dict = {}, {}
    step_norms_by_source = {}
    for name, source, step_kind in DISTANCE_ANGLE_REPRESENTATIONS:
        step_retention = retention if step_kind == STEP_HEBBIAN else None
        data = source_data[source]
        if source not in step_norms_by_source:
            # The decay/remainder split describes the source's full step, so
            # it is attached to the modulation representations only and is the
            # same for both step definitions of a source.
            step_norms_by_source[source] = _step_norm_summary(
                data, rule_indices, onset_by_rule, n_rules,
                retention if source != "hidden" else None, step=step)
        pairs, distances, angles = _task_pair_distance_angle(
            data, rule_indices, stimulus_labels, onset_by_rule, n_rules, step_retention,
            step=step, compute_distance=initial_scatter is None)
        if initial_scatter is not None:
            baseline = initial_scatter[name]
            angle_by_pair = dict(zip(pairs, angles))
            rule_index = {rule: index for index, rule in enumerate(all_rules)}
            pairs = [(rule_index[a], rule_index[b]) for a, b in baseline["pair_rules"]]
            distances = np.asarray(baseline["dists"], dtype=float).copy()
            angles = np.array([angle_by_pair.get(pair, np.nan) for pair in pairs])
        rval_dict[name], scatter = distance_angle_statistics(
            pairs, distances, angles, all_rules,
            n_permutations=DISTANCE_ANGLE_N_PERMUTATIONS,
            seed=DISTANCE_ANGLE_PERMUTATION_SEED)
        scatter_dict[name] = {
            **scatter,
            "step": {
                "index": int(step),
                "kind": step_kind,
                "source": source,
                "retention": step_retention,
                "definition": ("x[t] - x[t-1]" if step_kind == STEP_FULL
                               else "x[t] - lam * x[t-1] (decay term removed)"),
            },
            "step_norms": dict(step_norms_by_source[source]),
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
        if x.size >= 2 and np.isfinite(slope):
            x_fit = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_fit, entry["regression"]["intercept"] + slope * x_fit, color="red",
                    label=(f"Fit: slope={slope:.2f}, r={r_value:.2f}, nominal p={p_value:.3f}\n"
                           f"{perm_text}"))
        ax.set_xlabel("Distance between initial conditions")
        ax.set_ylabel("Angle between first step of trajectories (deg.)")
        ax.set_title(f"{name} ({entry['step']['definition']})")
        if np.isfinite(slope):
            ax.legend(frameon=True, loc="best", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _validate_step_window(rules_epochs, n_steps, aligned_length=None, trial_lengths=None):
    """Keep every local update within the first post-context task phase."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    for rule, epochs in rules_epochs.items():
        onset = int(epochs["fix1"][1])
        boundaries = [int(value) for interval in epochs.values() for edge in interval
                      if edge is not None for value in np.asarray(edge).reshape(-1)
                      if int(value) > onset]
        if trial_lengths is not None:
            boundaries.append(int(np.min(trial_lengths[rule])))
        if boundaries and onset + n_steps > min(boundaries):
            raise ValueError(
                f"{rule}: {n_steps} stimulus steps cross the next task-phase boundary "
                f"({min(boundaries) - onset} steps after onset).")
        if aligned_length is not None and onset + n_steps > aligned_length:
            raise ValueError(f"{rule}: trajectory is too short for {n_steps} stimulus steps.")


def _context_states(netpathname, untrained_init_seed=None, trace_steps=None,
                    context_ms=FIXED_FIXATION_MS):
    """Run one checkpoint's network on the evaluation trials and return what
    the analyses need, as a dict: the run's identifiers (aname, hidden_size,
    l2_info), lam (``retention``), dt, the fixed Context length, the accuracy,
    the task names, each trial's task index and first-stimulus label, the
    task epochs, and the (trials x time x features) trajectories of the
    hidden state, M and W * M, all on the hidden state's clock
    (align_modulation_to_hidden: index t is the state after input x[t], and
    the last recorded step is dropped). With ``untrained_init_seed`` the network is
    untrained (see network_loading.build_network); the evaluation trials do
    not depend on it, since they are drawn after reseeding with
    EVALUATION_SEED. Full hidden and modulation trajectories are retained on
    CPU. With trace_steps, record only Context + that many stimulus steps
    + one extra step for post-update M alignment; omit full-trial accuracy.
    ``context_ms`` is the Context length of the evaluation trials.
    """
    evaluation = evaluation_utils.prepare_evaluation(netpathname, untrained_init_seed, device=device,
                                                     context_ms=context_ms)
    model, state_dict = evaluation["model"], evaluation["state_dict"]
    aname, hidden_size, l2_info = (evaluation["aname"], evaluation["hidden_size"],
                                   evaluation["l2_info"])
    task_params_c, test_trials = evaluation["task_params"], evaluation["trials"]
    test_rule_idxs = evaluation["rule_indices"]
    retention, dt_ms = evaluation["retention"], evaluation["dt_ms"]
    fixed_fixation_steps = evaluation["fixed_fixation_steps"]
    _, labels_stim1, _, rules_epochs = helper.generate_response_stimulus(task_params_c, test_trials)
    all_rules = list(task_params_c["rules"])
    assert set(all_rules).issubset(set(RULE_MOTIF_MAPPING.keys()))
    trace_input = evaluation["inputs"]
    trial_lengths = {rule: getattr(trial, "tdim", trace_input.shape[1])
                     for rule, trial in zip(all_rules, test_trials)}
    if trace_steps is not None:
        _validate_step_window(rules_epochs, trace_steps, trace_input.shape[1] - 1, trial_lengths)
        # M is logged before each input: one extra record supplies the state
        # after the last analyzed stimulus update.
        trace_input = trace_input[:, :fixed_fixation_steps + trace_steps + 1]
    test_input = trace_input.to(device)

    with torch.no_grad():
        acc = float("nan")
        if trace_steps is None:
            # Full-trial accuracy is only part of the PCA workflow.
            net_out, _, _ = model.iterate_sequence_batch(test_input, run_mode="minimal")
            acc, _ = model.compute_acc(
                net_out, evaluation["outputs"].to(device), evaluation["masks"].to(device),
                test_input, isvalid=True, mode=model.acc_measure)
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

    print(f"acc: {acc:.2f}" if trace_steps is None else "Short trace: full-trial accuracy not evaluated.")

    # db_test is already on CPU.  NumPy views avoid another full copy of M.
    Ms_orig = db_test["M1"].numpy()
    modulation_W = state_dict["mp_layer1.W"].numpy()
    eff_Ms_orig = Ms_orig * modulation_W

    Ms = Ms_orig.reshape(Ms_orig.shape[0], Ms_orig.shape[1], -1)
    eff_Ms = eff_Ms_orig.reshape(eff_Ms_orig.shape[0], eff_Ms_orig.shape[1], -1)
    hs = db_test["hidden1"].numpy()
    # The recorded M[t] is the state before x[t]'s update; put M and W * M on
    # the hidden state's clock (views: the 30 GiB trace is not copied).
    hs, Ms, eff_Ms = align_modulation_to_hidden(hs, Ms, eff_Ms)
    print(f"Ms_orig.shape: {Ms_orig.shape}; aligned Ms.shape: {Ms.shape}; hs.shape: {hs.shape}")

    # All remaining work is NumPy/scikit-learn.  Release CUDA storage before
    # the comparatively long PCA and pairwise geometry calculations.
    del model, evaluation, state_dict, db_test
    del test_input, trace_input
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    del test_trials

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
        "trial_lengths": trial_lengths,
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
        "m_alignment": M_ALIGNMENT,
    }
    cache.update(extra or {})
    pca_save_path = output_dir / f"state_space_pca_{file_stem}_noise{NOISE_LEVEL}.pkl"
    with open(pca_save_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved PCA data to {pca_save_path}")
    return pca_save_path


def eval_one(netpathname, output_dir=None, n_steps=1, skip_pca=False,
             context_ms=FIXED_FIXATION_MS):
    """Analyze one checkpoint's Context geometry and stimulus-step directions.

    Save PCA and diagnostic plots under output_dir (STATE_SPACE_DIR by default).
    With skip_pca, retain only the short trace and preserve existing PCA caches.
    Return the original eight first-step fields, plus a timecourse when n_steps > 1.
    ``context_ms`` is the Context length of the evaluation trials.
    """
    output_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = (_context_states(netpathname, trace_steps=n_steps, context_ms=context_ms) if skip_pca
              else _context_states(netpathname, context_ms=context_ms))
    if not skip_pca:
        _save_context_pca(states, output_dir)
    result = _evaluation_tuple(states, *_distance_angle_for_states(
        states, output_dir, states["aname"], n_steps=n_steps))
    # Cleanup to prevent CPU memory compounding across experiments.
    del states
    gc.collect()
    return result


def _distance_angle_for_states(states, output_dir, file_stem, n_steps=1):
    """First-step distance-angle results and diagnostic; optionally attach
    the local-step timecourse to states, reusing the first-step distances."""
    all_rules = states["all_rules"]
    _validate_step_window(states["rules_epochs"], n_steps, states["hidden"].shape[1],
                          states.get("trial_lengths"))
    onset_by_rule = {idx: int(states["rules_epochs"][rule]['fix1'][1])
                     for idx, rule in enumerate(all_rules)}
    source_data = {name: states[name] for name in ("hidden", "mod", "eff_mod")}
    rval_dict, scatter_dict = distance_angle_analysis(
        source_data, states["rule_indices"], states["stimulus_labels"], onset_by_rule,
        all_rules, states["retention"])
    if n_steps > 1:
        by_step = {1: {"rval_dict": rval_dict, "scatter": scatter_dict}}
        for step in range(2, n_steps + 1):
            step_r, step_scatter = distance_angle_analysis(
                source_data, states["rule_indices"], states["stimulus_labels"],
                onset_by_rule, all_rules, states["retention"],
                step=step, initial_scatter=scatter_dict)
            by_step[step] = {"rval_dict": step_r, "scatter": step_scatter}
            print(f"  Stimulus step {step}: " + ", ".join(
                f"{name} r={values[0]:.3f}" for name, values in step_r.items()))
        states["timecourse"] = {
            "schema_version": 1,
            "n_steps": n_steps,
            "dt_ms": states["dt_ms"],
            "distance_reference": "context_end_step1_pairs",
            "by_step": by_step,
        }
    _draw_distance_angle_figure(
        scatter_dict, rval_dict,
        Path(output_dir) / f"initial_condition_distance_vs_angle_{file_stem}_1_noise{NOISE_LEVEL}.png")
    for name, entry in scatter_dict.items():
        permutation = entry["permutation"]
        text = (f"permutation p = {permutation['p_value']:.4f}" if permutation
                else f"permutation skipped ({entry['permutation_skipped']})")
        print(f"  {name}: r = {rval_dict[name][0]:.3f}, {text}; "
              f"step norms {entry['step_norms']}")
    return rval_dict, scatter_dict


def _evaluation_tuple(states, rval_dict, scatter_dict):
    """Original eight first-step fields, optionally followed by the timecourse."""
    result = (states["aname"], states["hidden_size"], states["l2_info"], rval_dict, scatter_dict,
              states["fixed_fixation_steps"], states["fixed_fixation_steps"] * states["dt_ms"],
              states["retention"])
    return result + (states["timecourse"],) if "timecourse" in states else result


def eval_untrained(netpathname, output_dir=None, init_seed=UNTRAINED_INIT_SEED,
                   n_steps=1, skip_pca=False, context_ms=FIXED_FIXATION_MS):
    """Context-end PCA and distance-angle analysis of an untrained network
    with the architecture and config of the checkpoint ``netpathname``: fresh
    weights drawn after seeding with ``init_seed``, none of the trained
    weights used (see network_loading.build_network), run on the same
    evaluation trials as the trained analysis. It is the control for both: what the rule-cue
    inputs and the Hebbian update produce without any learning.

    The PCA cache and the diagnostic figures go to ``<output_dir>/untrained/``
    (output_dir defaults to STATE_SPACE_DIR), named
    ``untrained_file_stem(aname, init_seed)`` so every seed keeps its own
    files. The subdirectory keeps them out of the trained caches' readers
    (state_space_structure.py's glob, paper_plot's seed rankings), and the
    batch run's cleanup, which removes top-level files only, keeps it. The
    cache is marked ``untrained`` and records ``init_seed`` and the
    checkpoint it took its config from.

    Returns (file stem, distance-angle results entry): the entry has the
    fields of a trained run's (``_result_entry``) plus the same marks, with
    ``l2_info`` set to None, since an untrained network has no L2 strength
    (the config's is kept as ``config_l2_info``); ``run_untrained`` collects
    the entries in the untrained subdirectory's own results pickle.
    """
    untrained_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir) / UNTRAINED_DIR_NAME
    untrained_dir.mkdir(parents=True, exist_ok=True)
    states = (_context_states(netpathname, untrained_init_seed=init_seed, trace_steps=n_steps,
                              context_ms=context_ms)
              if skip_pca else _context_states(netpathname, untrained_init_seed=init_seed,
                                               context_ms=context_ms))
    stem = untrained_file_stem(states["aname"], init_seed)
    marks = {"untrained": True, "init_seed": int(init_seed), "config_source": states["aname"]}
    if not skip_pca:
        _save_context_pca(states, untrained_dir, extra=marks, file_stem=stem)
    rval_dict, scatter_dict = _distance_angle_for_states(
        states, untrained_dir, stem, n_steps=n_steps)
    _, entry = _result_entry(_evaluation_tuple(states, rval_dict, scatter_dict))
    entry.update(marks, l2_info=None, config_l2_info=entry["l2_info"])
    del states
    gc.collect()
    return stem, entry


def _result_entry(evaluation):
    """The results-pickle entry of one ``eval_one`` return value."""
    (aname, hidden_size, l2_info, rval_dict, scatter_dict,
     fixed_fixation_steps, fixed_fixation_ms, retention, *extra) = evaluation
    entry = {
        "schema_version": DISTANCE_ANGLE_SCHEMA_VERSION,
        "m_alignment": M_ALIGNMENT,
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
    if extra:
        entry["timecourse"] = extra[0]
    return aname, entry


def _results_path(output_dir=None):
    return Path(STATE_SPACE_DIR if output_dir is None else output_dir) / RESULTS_NAME


def _save_results(result_dict, output_dir=None):
    with _results_path(output_dir).open("wb") as f:
        pickle.dump(result_dict, f)


def run_all(output_dir=None, n_steps=1, skip_pca=False, context_ms=FIXED_FIXATION_MS):
    """Analyze the target cohort, clearing top-level files unless skip_pca.
    With skip_pca, merge distance-angle entries and preserve other files.
    Save after every run so an interrupted batch keeps what it finished.
    ``context_ms`` is the Context length of the evaluation trials.
    """
    output_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir)
    all_pt_paths = mpf.list_pt_files("./multiple_tasks", recursive=False)
    pt_paths = evaluation_utils.select_target_checkpoints(all_pt_paths)
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
        print(f"  {evaluation_utils.checkpoint_aname(path)}")

    if skip_pca:
        # Supplement distance-angle results without deleting PCA or any derived caches.
        return run_checkpoints(pt_paths, output_dir, n_steps=n_steps, skip_pca=True,
                               context_ms=context_ms)
    _clean_state_space_results(output_dir)

    result_dict = {}
    for netpathname in pt_paths:
        aname, entry = _result_entry(eval_one(netpathname, output_dir, n_steps=n_steps,
                                              skip_pca=skip_pca, context_ms=context_ms))
        result_dict[aname] = entry
        _save_results(result_dict, output_dir)
    return result_dict


def run_checkpoints(checkpoint_paths, output_dir=None, n_steps=1, skip_pca=False,
                    context_ms=FIXED_FIXATION_MS):
    """Analyze only the given ``savednet_*.pt`` checkpoints and merge their
    entries into the results pickle of ``output_dir`` (STATE_SPACE_DIR by
    default), which is created if needed and never cleared. A checkpoint
    outside the default cohort is analyzed with a warning, since the caller
    named it explicitly. ``context_ms`` is the Context length of the
    evaluation trials."""
    output_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir)
    paths = evaluation_utils.existing_checkpoints(checkpoint_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = _results_path(output_dir)
    result_dict = {}
    if results_path.exists():
        with results_path.open("rb") as f:
            result_dict = pickle.load(f)
    for path in paths:
        if not evaluation_utils.matches_target_cohort(path):
            print(f"Warning: {path} is outside the default state-space cohort.")
        aname, entry = _result_entry(eval_one(path, output_dir, n_steps=n_steps, skip_pca=skip_pca,
                                              context_ms=context_ms))
        result_dict[aname] = entry
        _save_results(result_dict, output_dir)
        print(f"Merged {aname} into {results_path} ({len(result_dict)} run(s))")
    return result_dict


def untrained_dir_of(output_dir=None):
    """The untrained networks' subdirectory of ``output_dir``."""
    return Path(STATE_SPACE_DIR if output_dir is None else output_dir) / UNTRAINED_DIR_NAME


def run_untrained(checkpoint_paths, output_dir=None, init_seeds=(UNTRAINED_INIT_SEED,),
                  n_steps=1, skip_pca=False, context_ms=FIXED_FIXATION_MS):
    """``eval_untrained`` for each named checkpoint and each of ``init_seeds``:
    untrained networks with that run's architecture and config, one per
    seed, written to <output_dir>/untrained/, whose own distance-angle
    results pickle (RESULTS_NAME) gains one entry per network, keyed by its
    file stem, and is rewritten after every network. Nothing else in
    ``output_dir`` is touched. ``context_ms`` is the Context length of the
    evaluation trials. Returns the merged results."""
    paths = evaluation_utils.existing_checkpoints(checkpoint_paths)
    seeds = [int(seed) for seed in init_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Need one or more distinct init seeds, got {seeds}.")
    untrained_dir = untrained_dir_of(output_dir)
    untrained_dir.mkdir(parents=True, exist_ok=True)
    results_path = _results_path(untrained_dir)
    result_dict = {}
    if results_path.exists():
        with results_path.open("rb") as f:
            result_dict = pickle.load(f)
    for path in paths:
        if not evaluation_utils.matches_target_cohort(path):
            print(f"Warning: {path} is outside the default state-space cohort.")
        for seed in seeds:
            stem, entry = eval_untrained(path, output_dir, seed, n_steps=n_steps, skip_pca=skip_pca,
                                         context_ms=context_ms)
            result_dict[stem] = entry
            _save_results(result_dict, untrained_dir)
            print(f"Merged {stem} into {results_path} ({len(result_dict)} network(s))")
    return result_dict


def summarize(output_dir=None):
    """Print per-representation r, nominal p and permutation p over the
    analyzed runs, the decay term's share of the full modulation step, and
    save a bar chart of the mean r (summary_r_values.png)."""
    output_dir = Path(STATE_SPACE_DIR if output_dir is None else output_dir)
    with _results_path(output_dir).open("rb") as f:
        result_dict = pickle.load(f)

    current = {name: entry for name, entry in result_dict.items()
               if distance_angle_is_current(entry)}
    if len(current) != len(result_dict):
        print(f"Skipped {len(result_dict) - len(current)} stale distance-angle entries "
              "(schema or m_alignment); rerun only those networks with state_space_shift.py.")
    result_dict = current
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
    context_lengths = sorted({(entry["fixed_fixation_steps"], entry["fixed_fixation_ms"])
                              for entry in result_dict.values() if "fixed_fixation_ms" in entry})
    print("Context length: " + ", ".join(f"{ms:g} ms ({steps} steps)"
                                         for steps, ms in context_lengths))
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
    ax.set_title('Mean R-values across networks\n(hatched: linear decay removed)')
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
        "--out-dir", default=None, metavar="DIR",
        help=f"directory for every output (default: {STATE_SPACE_DIR}/, or "
             f"{STATE_SPACE_DIR}/context<ms>ms/ with a non-default --context-ms)")
    parser.add_argument(
        "--context-ms", type=float, default=FIXED_FIXATION_MS, metavar="MS",
        help="Context length of the evaluation trials in ms, rounded down to whole "
             f"simulation steps (default: {FIXED_FIXATION_MS})")
    parser.add_argument(
        "--untrained", action="store_true",
        help="instead analyze untrained networks with the architecture and config of "
             "each --checkpoint (fresh weights from --init-seed, no trained weight used): "
             "context-end PCA and distance-angle analysis, written to "
             f"<out-dir>/{UNTRAINED_DIR_NAME}/ with its own results pickle")
    parser.add_argument(
        "--init-seed", type=int, nargs="+", default=None, metavar="SEED",
        help=f"seeds of the untrained weights, one network per seed (default: "
             f"{UNTRAINED_INIT_SEED}); only with --untrained")
    parser.add_argument(
        "--steps", type=int, default=1, metavar="N",
        help="analyze local update angles at stimulus steps 1..N, holding the "
             "Context-end distances and first-step task pairs fixed (default: 1)")
    parser.add_argument(
        "--skip-pca", action="store_true",
        help="record only the short stimulus window and update distance-angle results; "
             "preserve existing PCA and derived caches, including in cohort mode")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not args.context_ms > 0:
        parser.error("--context-ms must be positive")
    if args.out_dir is None:
        args.out_dir = str(context_dir_of(STATE_SPACE_DIR, args.context_ms))
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
        run_untrained(args.checkpoint, output_dir, args.init_seed,
                      n_steps=args.steps, skip_pca=args.skip_pca, context_ms=args.context_ms)
        summarize(untrained_dir_of(output_dir))
        return
    if args.checkpoint:
        run_checkpoints(args.checkpoint, output_dir, n_steps=args.steps, skip_pca=args.skip_pca,
                        context_ms=args.context_ms)
    else:
        run_all(output_dir, n_steps=args.steps, skip_pca=args.skip_pca, context_ms=args.context_ms)
    summarize(output_dir)


if __name__ == "__main__":
    main()
