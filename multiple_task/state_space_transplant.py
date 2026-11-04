"""
Modulation transplant: how does Context-end M change response directions?

The distance-angle analysis of state_space_shift.py relates the distance
between two tasks' Context-end states to the angle between their first
post-stimulus steps over cross-task trial pairs that are matched by stimulus
label only. Trials of different tasks still differ in their onset inputs
(modality, exact direction, strength, a second stimulus) and in their rule
cue, so part of that angle is an input difference, not a state difference.
This experiment removes the input difference by construction: every
comparison holds the input fixed and changes only the modulation state.

Design. The network has one plastic layer and no hidden recurrence, so M is
its whole memory. Every trial runs through its Context period and the live
M of the plastic layer is cloned after the last Context update (the aligned
Context-end state of state_space_shift.py). For an ordered task pair (host
A, donor B) the k-th trial of A hosts the k-th trial of B: from M_A (the
host trial's own state) and from M_B (the donor's) the network takes one
step under the host's onset input x_on. The response vectors are

    modulation        v(M) = F_x_on(M) - M, with F the network's own update
                      (Hebbian term, decay, M activation and clipping);
    eff. modulation   W * v(M);
    hidden state      v_h(M) = h(M, x_on) - h(M, x_ctx): the response to the
                      onset input minus the response to the host's last
                      Context input, both read with M frozen.

Each branch subtracts its own starting point, so the jump the transplant
itself causes is not counted. The angle between v(M_A) and v(M_B) is the
change of response direction that the state change alone produces; the
distance is ||M_A - M_B|| over the paired trials. Both directions (A
hosting B, B hosting A) are kept and averaged into one value per unordered
pair, and the pairs give Pearson r (free-intercept OLS) with the task-label
permutation p of state_space_shift.py. The x axis is the M distance for
every representation: the question is how the response change follows the
state change, not how each representation's own distance does.

Controls recorded with every network. The self transplant (M_A into its own
trials) must reproduce the uninterrupted run's onset response and M to
NATIVE_CHECK_TOLERANCE, which the run checks and aborts on. Same-task
transplants (the k-th trial of A hosting the (k-1)-th) give the angle that
Context input noise alone produces, a noise baseline (not a lower bound). Per
representation the mean norms of the response vectors (native and transplanted),
of the decay term (1 - lam) M and of the start state are kept, with the
share of M entries at the bounds after the step. Steps of (numerically)
zero norm are skipped, never given an angle.

Statistics. r and its permutation p describe the association of two
task-pair matrices (M distance, response change). The intervention is on
M, but the correlation is not a test of a causal effect, and transplanting
the state does not show that a task's behavior or rule moves with it.

Evaluation seed. The two branches of one comparison see the same input
tensor, but the inputs themselves (stimuli, Context noise, hence M) depend
on the evaluation seed; ``--evaluation-seed`` reruns everything on another
draw into a results file named after the seed, to check that the r values
do not hinge on one draw.

Outputs go to ``state_space/transplant/`` (TRANSPLANT_DIR_NAME under
``--out-dir``, a subdirectory that state_space_shift.py's batch cleanup
leaves alone): ``transplant_results.pkl`` keyed by run name, rewritten
after every run, and one diagnostic scatter per run. Without arguments the
whole state-space cohort is analyzed from scratch; ``--checkpoint PATH
[PATH ...]`` analyzes those runs and merges them. ``--untrained
--checkpoint PATH --init-seed S [S ...]`` analyzes untrained networks of
that run's config (network_loading.build_network) into
``state_space/transplant/untrained/`` with their own results pickle
(entries keyed by ``<aname>_init<seed>``, ``l2_info`` None). Draw with
``python figures/paper_plot.py --only state_space_transplant_r_values``.
"""
from pathlib import Path
import argparse
import gc
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper
import multiple_task_performance as mpf
import state_space_shift as shift

TRANSPLANT_DIR_NAME = "transplant"
RESULTS_NAME = "transplant_results.pkl"
# Results entries of this experiment (paper_plot reads this version or later).
TRANSPLANT_SCHEMA_VERSION = 1
EXPERIMENT = "modulation_transplant"
REPRESENTATIONS = ("hidden", "mod", "eff_mod")
STEP_DEFINITIONS = {
    "hidden": "h(M, x_on) - h(M, x_ctx), M frozen",
    "mod": "F_x_on(M) - M, the network's full update",
    "eff_mod": "W * (F_x_on(M) - M)",
}
DISTANCE_DEFINITION = "||M_A - M_B|| over the paired trials"
# The self transplant must reproduce the uninterrupted run to this absolute
# difference (float32 arithmetic in batches of another size).
NATIVE_CHECK_TOLERANCE = 1e-4
ZERO_STEP_NORM = 1e-12
BOUND_TOLERANCE = 1e-6


def plastic_layer(model):
    """The single plastic layer of ``model``. The transplant moves its M, so
    the network must hold no other memory: one plastic layer and no hidden
    recurrence."""
    layers = list(model.mp_layers)
    if len(layers) != 1:
        raise ValueError(f"The transplant moves one plastic layer's M; the network has {len(layers)}.")
    if getattr(model, "recurrent_active", False):
        raise ValueError("The transplant takes M as the network's only memory; hidden recurrence is on.")
    return layers[0]


def modulation_after_steps(model, inputs, n_steps):
    """Run every trial of ``inputs`` (trials x time x channels) through its
    first ``n_steps`` inputs. Returns the live M of the plastic layer after
    the last update, cloned (trials x post x pre), and the hidden responses
    (trials x n_steps x hidden). The live M, not the recorded one, is the
    state after the last input (SCHEME.md: the network records M before
    each update)."""
    with torch.no_grad():
        _, hidden, _ = model.iterate_sequence_batch(inputs[:, :n_steps], run_mode="minimal")
        return plastic_layer(model).M.detach().clone(), hidden.detach()


def branch_step(model, m_start, x_context, x_onset):
    """One step of ``model`` from the modulation states ``m_start`` (trials x
    post x pre) under the host trials' last Context input ``x_context`` and
    onset input ``x_onset`` (trials x channels). Returns the response
    vectors of the three representations (float64 arrays, trials x
    features; see the module docstring), the frozen-M Context response
    ``h_context`` (float64), the onset response ``h_onset`` and the updated
    modulation ``m_after`` (tensors, for the self-transplant check) and the
    share of M entries at the bounds after the step (NaN without bounds)."""
    layer = plastic_layer(model)
    with torch.no_grad():
        layer.M = m_start.clone()
        _, activities, _ = model.forward(x_context)
        h_context = activities[-1]
        _, activities, _ = model.network_step(x_onset)
        h_onset = activities[-1]
        m_after = layer.M.detach()
        step = m_after - m_start
        clipped = float("nan")
        if getattr(layer, "modulation_bounds", False):
            at_bound = ((m_after >= layer.M_bounds[0] - BOUND_TOLERANCE)
                        | (m_after <= layer.M_bounds[1] + BOUND_TOLERANCE))
            clipped = float(at_bound.double().mean())
        return {
            "hidden": (h_onset - h_context).double().cpu().numpy(),
            "mod": step.flatten(1).double().cpu().numpy(),
            "eff_mod": (step * layer.W.detach().unsqueeze(0)).flatten(1).double().cpu().numpy(),
            "h_context": h_context.double().cpu().numpy(),
            "h_onset": h_onset,
            "m_after": m_after,
            "clipped_fraction": clipped,
        }


def paired_angles(steps_1, steps_2):
    """Angle in degrees between the k-th rows of two (trials x features)
    arrays; a row of (numerically) zero norm in either gives NaN."""
    norms_1 = np.linalg.norm(steps_1, axis=1)
    norms_2 = np.linalg.norm(steps_2, axis=1)
    valid = (norms_1 >= ZERO_STEP_NORM) & (norms_2 >= ZERO_STEP_NORM)
    angles = np.full(steps_1.shape[0], np.nan)
    cosines = (steps_1[valid] * steps_2[valid]).sum(axis=1) / (norms_1[valid] * norms_2[valid])
    angles[valid] = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return angles


def _mean_angle(angles):
    """Mean over the finite angles, NaN when there is none."""
    finite = angles[np.isfinite(angles)]
    return float(finite.mean()) if finite.size else float("nan")


def _check_native_branch(check, native, hidden_onset, m_onset):
    """Compare the self transplant's onset response and updated M with the
    uninterrupted run's; keep the largest differences in ``check`` and abort
    beyond the tolerance."""
    hidden_diff = float((native["h_onset"] - hidden_onset).abs().max())
    m_diff = float((native["m_after"] - m_onset).abs().max())
    check["max_abs_hidden_diff"] = max(check["max_abs_hidden_diff"], hidden_diff)
    check["max_abs_m_diff"] = max(check["max_abs_m_diff"], m_diff)
    if max(hidden_diff, m_diff) > check["tolerance"]:
        raise RuntimeError(
            "The self transplant does not reproduce the uninterrupted run: max abs "
            f"difference {max(hidden_diff, m_diff):.3g} in the onset hidden state or M "
            f"(tolerance {check['tolerance']:g}).")


def _norm_mean(tensor):
    """Mean over trials of the Euclidean norm of each trial's entries."""
    return float(tensor.flatten(1).double().norm(dim=1).mean())


def transplant_analysis(model, inputs, rule_indices, all_rules, context_steps, retention):
    """The transplant experiment of one network (module docstring) on
    ``inputs`` (trials x time x channels, on the model's device), whose
    trials' ``rule_indices`` index ``all_rules`` and whose Context ends after
    ``context_steps`` inputs (the onset input is inputs[:, context_steps]).
    ``retention`` is lam. Trials are paired by position within their tasks,
    up to the smallest task's count.

    Returns (rval_dict, scatter_dict, step_norms, native_check). rval_dict
    maps each representation to (Pearson r, slope, nominal OLS p) of the
    response angle over the M distance across the unordered task pairs.
    scatter_dict holds per representation the pairs' distances and mean
    angles, the angles of both hosting directions (``angles_deg_by_host``:
    first task hosting, second task hosting), the task names, the OLS fit,
    the task-label permutation test, the same-task control angle (mean and
    per task), the count of skipped zero-norm steps and the definitions.
    step_norms holds per representation the mean norms of the start state
    and of the native and transplanted response vectors, for the modulation
    representations the decay term's norm, and for raw M the share of
    entries at the bounds after native and transplanted steps. native_check
    holds the self-transplant differences.
    """
    rule_indices = np.asarray(rule_indices)
    all_rules = list(all_rules)
    n_rules = len(all_rules)
    trials_by_rule = [np.flatnonzero(rule_indices == index) for index in range(n_rules)]
    if any(trials.size == 0 for trials in trials_by_rule):
        raise ValueError("Every task needs at least one trial.")
    n_hosted = min(trials.size for trials in trials_by_rule)
    layer = plastic_layer(model)
    weights = layer.W.detach().unsqueeze(0)
    m_context, _ = modulation_after_steps(model, inputs, context_steps)
    m_onset, hidden = modulation_after_steps(model, inputs, context_steps + 1)
    hidden_onset = hidden[:, context_steps]

    directional = {}
    within = {name: np.full(n_rules, np.nan) for name in REPRESENTATIONS}
    norms = {name: {"initial": [], "native_step": [], "transplant_step": []}
             for name in REPRESENTATIONS}
    clipped = {"native": [], "transplant": []}
    skipped = {name: 0 for name in REPRESENTATIONS}
    native_check = {"max_abs_hidden_diff": 0.0, "max_abs_m_diff": 0.0,
                    "tolerance": NATIVE_CHECK_TOLERANCE}
    for host in range(n_rules):
        hosts = trials_by_rule[host][:n_hosted]
        x_context, x_onset = inputs[hosts, context_steps - 1], inputs[hosts, context_steps]
        native = branch_step(model, m_context[hosts], x_context, x_onset)
        _check_native_branch(native_check, native, hidden_onset[hosts], m_onset[hosts])
        clipped["native"].append(native["clipped_fraction"])
        norms["hidden"]["initial"].append(float(np.linalg.norm(native["h_context"], axis=1).mean()))
        norms["mod"]["initial"].append(_norm_mean(m_context[hosts]))
        norms["eff_mod"]["initial"].append(_norm_mean(m_context[hosts] * weights))
        for name in REPRESENTATIONS:
            norms[name]["native_step"].append(float(np.linalg.norm(native[name], axis=1).mean()))
        # Same-task control: the previous trial of the host's own task donates.
        control = branch_step(model, m_context[np.roll(trials_by_rule[host], 1)[:n_hosted]],
                              x_context, x_onset)
        for name in REPRESENTATIONS:
            within[name][host] = _mean_angle(paired_angles(native[name], control[name]))
        for donor in range(n_rules):
            if donor == host:
                continue
            donors = trials_by_rule[donor][:n_hosted]
            branch = branch_step(model, m_context[donors], x_context, x_onset)
            clipped["transplant"].append(branch["clipped_fraction"])
            entry = {"distance": _norm_mean(m_context[hosts] - m_context[donors]), "angles": {}}
            for name in REPRESENTATIONS:
                angles = paired_angles(native[name], branch[name])
                skipped[name] += int(np.count_nonzero(~np.isfinite(angles)))
                entry["angles"][name] = _mean_angle(angles)
                norms[name]["transplant_step"].append(
                    float(np.linalg.norm(branch[name], axis=1).mean()))
            directional[(host, donor)] = entry

    rval_dict, scatter_dict = {}, {}
    for name in REPRESENTATIONS:
        # Undefined hidden responses must not discard valid modulation updates.
        pairs, distances, by_host = [], [], []
        for first in range(n_rules):
            for second in range(first + 1, n_rules):
                forward, backward = directional[(first, second)], directional[(second, first)]
                angles = (forward["angles"][name], backward["angles"][name])
                if not np.isfinite(angles).all():
                    continue
                pairs.append((first, second))
                distances.append(0.5 * (forward["distance"] + backward["distance"]))
                by_host.append(angles)
        distances = np.asarray(distances, dtype=float)
        angles_by_host = np.asarray(by_host, dtype=float).reshape(-1, 2)
        angles = angles_by_host.mean(axis=1)
        regression_reason = None
        if distances.size < 2:
            regression_reason = "fewer than two task pairs have defined response angles in both directions"
        elif np.ptp(distances) == 0 or np.ptp(angles) == 0:
            regression_reason = "Pearson correlation is undefined for constant distances or angles"
        if regression_reason:
            r_value = slope = intercept = p_value = float("nan")
        else:
            _, _, r_value, slope, intercept, p_value = helper.linear_regression(
                distances, angles, log=False, through_origin=False)
        permutation, reason = shift._permutation_test_or_reason(
            pairs, distances, angles, n_rules,
            pair_requirement="have defined response angles in both directions")
        rval_dict[name] = (r_value, slope, p_value)
        scatter_dict[name] = {
            "dists": distances,
            "angles_deg": angles,
            "angles_deg_by_host": angles_by_host,
            "pair_rules": [(all_rules[first], all_rules[second]) for first, second in pairs],
            "regression": {
                "through_origin": False,
                "intercept": float(intercept),
                "p_value_method": "OLS slope t-test assuming independent task pairs",
                "skipped": regression_reason,
            },
            "permutation": permutation,
            "permutation_skipped": reason,
            "within_task_angle_deg": _mean_angle(within[name]),
            "within_task_angles_deg": {rule: float(within[name][index])
                                       for index, rule in enumerate(all_rules)},
            "n_steps_skipped": skipped[name],
            "step": {"kind": "transplant", "source": name,
                     "definition": STEP_DEFINITIONS[name], "distance": DISTANCE_DEFINITION},
        }

    step_norms = {name: {key: float(np.mean(values)) for key, values in norms[name].items()}
                  for name in REPRESENTATIONS}
    for name in ("mod", "eff_mod"):
        step_norms[name]["decay_term"] = (1.0 - float(retention)) * step_norms[name]["initial"]
    step_norms["mod"]["clipped_fraction_native"] = _mean_angle(np.asarray(clipped["native"]))
    step_norms["mod"]["clipped_fraction_transplant"] = _mean_angle(np.asarray(clipped["transplant"]))
    return rval_dict, scatter_dict, step_norms, native_check


def _draw_transplant_figure(scatter_dict, rval_dict, out_path):
    """Per-network diagnostic: one panel per representation with the task
    pairs, the OLS guide, r, the nominal and the permutation p, and the
    same-task control angle as a dotted line."""
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
        if entry["regression"].get("skipped"):
            ax.text(0.03, 0.97, "Undefined r: " + entry["regression"]["skipped"],
                    transform=ax.transAxes, va="top", fontsize=6, wrap=True)
        elif not permutation:
            ax.text(0.03, 0.97, perm_text, transform=ax.transAxes, va="top", fontsize=6, wrap=True)
        if np.isfinite(entry["within_task_angle_deg"]):
            ax.axhline(entry["within_task_angle_deg"], color="k", linestyle=":",
                       label=f"same-task donors: {entry['within_task_angle_deg']:.2f} deg")
        ax.set_xlabel("Distance between the transplanted M states")
        ax.set_ylabel("Angle between response vectors\nfrom M_A and M_B, same input (deg.)")
        ax.set_title(f"{name}: {entry['step']['definition']}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=True, loc="best", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def results_name(evaluation_seed=shift.EVALUATION_SEED):
    """The results pickle of an evaluation seed: RESULTS_NAME for the
    standard seed, otherwise named after the seed so a check never
    overwrites the main results."""
    if int(evaluation_seed) == shift.EVALUATION_SEED:
        return RESULTS_NAME
    return f"transplant_results_eval{int(evaluation_seed)}.pkl"


def _figure_stem(stem, evaluation_seed):
    """The diagnostic figure's stem, marked by any non-standard seed."""
    if int(evaluation_seed) == shift.EVALUATION_SEED:
        return stem
    return f"{stem}_eval{int(evaluation_seed)}"


def transplant_dir_of(output_dir=None):
    """The experiment's subdirectory of the state-space directory."""
    return Path(shift.STATE_SPACE_DIR if output_dir is None else output_dir) / TRANSPLANT_DIR_NAME


def untrained_dir_of(output_dir=None):
    """The untrained networks' subdirectory of the experiment's directory."""
    return transplant_dir_of(output_dir) / shift.UNTRAINED_DIR_NAME


def analyze_network(evaluation, directory, file_stem):
    """Run the experiment on a prepared evaluation
    (state_space_shift.prepare_evaluation), save the diagnostic figure
    ``transplant_<file_stem>.png`` in ``directory`` and return the results
    entry."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    all_rules = list(evaluation["task_params"]["rules"])
    inputs = evaluation["inputs"].to(shift.device)
    context_steps = int(evaluation["fixed_fixation_steps"])
    rval_dict, scatter_dict, step_norms, native_check = transplant_analysis(
        evaluation["model"], inputs, evaluation["rule_indices"], all_rules,
        context_steps, evaluation["retention"])
    _draw_transplant_figure(scatter_dict, rval_dict, directory / f"transplant_{file_stem}.png")
    for name in REPRESENTATIONS:
        permutation = scatter_dict[name]["permutation"]
        text = (f"permutation p = {permutation['p_value']:.4f}" if permutation
                else f"permutation skipped ({scatter_dict[name]['permutation_skipped']})")
        print(f"{name}: r = {rval_dict[name][0]:.3f}, {text}, same-task control "
              f"{scatter_dict[name]['within_task_angle_deg']:.2f} deg")
    print(f"Self transplant max abs difference: hidden {native_check['max_abs_hidden_diff']:.2e}, "
          f"M {native_check['max_abs_m_diff']:.2e}; M entries at the bounds: native "
          f"{step_norms['mod']['clipped_fraction_native']:.3f}, transplanted "
          f"{step_norms['mod']['clipped_fraction_transplant']:.3f}")
    return {
        "schema_version": TRANSPLANT_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "hidden_size": evaluation["hidden_size"],
        "l2_info": evaluation["l2_info"],
        "retention": evaluation["retention"],
        "dt_ms": evaluation["dt_ms"],
        "fixed_fixation_steps": context_steps,
        "fixed_fixation_ms": context_steps * evaluation["dt_ms"],
        "evaluation_seed": int(evaluation["evaluation_seed"]),
        "trials_per_rule": shift.TRIALS_PER_RULE,
        "noise_level": shift.NOISE_LEVEL,
        "all_rules": all_rules,
        "rval_dict": rval_dict,
        "scatter": scatter_dict,
        "step_norms": step_norms,
        "native_check": native_check,
        "n_permutations": shift.DISTANCE_ANGLE_N_PERMUTATIONS,
        "permutation_seed": shift.DISTANCE_ANGLE_PERMUTATION_SEED,
        "untrained": False,
    }


def _release(evaluation):
    """Drop a prepared evaluation and its network from memory."""
    del evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_one(netpathname, output_dir=None, evaluation_seed=shift.EVALUATION_SEED):
    """The experiment on one trained checkpoint. Returns (aname, entry)."""
    evaluation = shift.prepare_evaluation(netpathname, evaluation_seed=evaluation_seed)
    aname = evaluation["aname"]
    entry = analyze_network(evaluation, transplant_dir_of(output_dir),
                            _figure_stem(aname, evaluation_seed))
    _release(evaluation)
    return aname, entry


def eval_untrained(netpathname, output_dir=None, init_seed=shift.UNTRAINED_INIT_SEED,
                   evaluation_seed=shift.EVALUATION_SEED):
    """The experiment on an untrained network with the config of checkpoint
    ``netpathname`` and fresh weights from ``init_seed``
    (network_loading.build_network), on the same trials. Returns (file stem,
    entry): the entry carries the marks of state_space_shift.py's untrained
    entries (``untrained``, ``init_seed``, ``config_source``; ``l2_info``
    None with the config's L2 as ``config_l2_info``)."""
    evaluation = shift.prepare_evaluation(netpathname, untrained_init_seed=init_seed,
                                          evaluation_seed=evaluation_seed)
    stem = shift.untrained_file_stem(evaluation["aname"], init_seed)
    entry = analyze_network(evaluation, untrained_dir_of(output_dir),
                            _figure_stem(stem, evaluation_seed))
    entry.update(untrained=True, init_seed=int(init_seed), config_source=evaluation["aname"],
                 config_l2_info=entry["l2_info"], l2_info=None)
    _release(evaluation)
    return stem, entry


def _load_results(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        return pickle.load(stream)


def _save_results(result_dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(result_dict, stream)


def run_all(output_dir=None, evaluation_seed=shift.EVALUATION_SEED):
    """The whole state-space cohort (state_space_shift.py's selection) into
    a fresh results pickle, rewritten after every run."""
    all_pt_paths = mpf.list_pt_files("./multiple_tasks", recursive=False)
    pt_paths = shift._select_target_checkpoints(all_pt_paths)
    if not pt_paths:
        raise FileNotFoundError("No checkpoint of the state-space cohort found under multiple_tasks/.")
    print(f"Selected {len(pt_paths)}/{len(all_pt_paths)} state-space checkpoints")
    results_path = transplant_dir_of(output_dir) / results_name(evaluation_seed)
    result_dict = {}
    for path in pt_paths:
        aname, entry = eval_one(path, output_dir, evaluation_seed)
        result_dict[aname] = entry
        _save_results(result_dict, results_path)
        print(f"Saved {aname} to {results_path} ({len(result_dict)} run(s))")
    return result_dict


def run_checkpoints(checkpoint_paths, output_dir=None, evaluation_seed=shift.EVALUATION_SEED):
    """The named checkpoints, merged into the results pickle, which is never
    cleared. A checkpoint outside the cohort is analyzed with a warning."""
    paths = shift._existing_checkpoints(checkpoint_paths)
    results_path = transplant_dir_of(output_dir) / results_name(evaluation_seed)
    result_dict = _load_results(results_path)
    for path in paths:
        if not shift._matches_target_cohort(path):
            print(f"Warning: {path} is outside the default state-space cohort.")
        aname, entry = eval_one(path, output_dir, evaluation_seed)
        result_dict[aname] = entry
        _save_results(result_dict, results_path)
        print(f"Merged {aname} into {results_path} ({len(result_dict)} run(s))")
    return result_dict


def run_untrained(checkpoint_paths, output_dir=None, init_seeds=(shift.UNTRAINED_INIT_SEED,),
                  evaluation_seed=shift.EVALUATION_SEED):
    """``eval_untrained`` for each named checkpoint and init seed, merged
    into the untrained subdirectory's own results pickle."""
    paths = shift._existing_checkpoints(checkpoint_paths)
    seeds = [int(seed) for seed in init_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Need one or more distinct init seeds, got {seeds}.")
    results_path = untrained_dir_of(output_dir) / results_name(evaluation_seed)
    result_dict = _load_results(results_path)
    for path in paths:
        if not shift._matches_target_cohort(path):
            print(f"Warning: {path} is outside the default state-space cohort.")
        for seed in seeds:
            stem, entry = eval_untrained(path, output_dir, seed, evaluation_seed)
            result_dict[stem] = entry
            _save_results(result_dict, results_path)
            print(f"Merged {stem} into {results_path} ({len(result_dict)} network(s))")
    return result_dict


def summarize(result_dict, label):
    """Print per representation the r values, the permutation p counts and
    the same-task control angles over ``result_dict``, and the clipping
    shares and self-transplant differences."""
    print("=" * 80)
    print(f"Transplant summary of {len(result_dict)} {label}")
    print("=" * 80)
    entries = list(result_dict.values())
    if not entries:
        return
    for name in REPRESENTATIONS:
        r_values = np.array([entry["rval_dict"][name][0] for entry in entries])
        r_values = r_values[np.isfinite(r_values)]
        p_values = np.array([entry["scatter"][name]["permutation"]["p_value"]
                             for entry in entries if entry["scatter"][name]["permutation"]])
        controls = np.array([entry["scatter"][name]["within_task_angle_deg"] for entry in entries])
        angles = np.array([_mean_angle(entry["scatter"][name]["angles_deg"]) for entry in entries])
        if r_values.size:
            print(f"\n{name}: r mean={r_values.mean():.3f}, sd={r_values.std(ddof=1) if r_values.size > 1 else 0.0:.3f}, "
                  f"min={r_values.min():.3f}, max={r_values.max():.3f} "
                  f"({r_values.size}/{len(entries)} defined)")
        else:
            print(f"\n{name}: r undefined for all {len(entries)} networks")
        if p_values.size:
            print(f"  permutation p < 0.05: {np.count_nonzero(p_values < 0.05)}/{p_values.size}")
        print(f"  cross-task angle mean={_mean_angle(angles):.2f} deg; same-task control "
              f"mean={_mean_angle(controls):.2f} deg")
    clipped = np.array([[entry["step_norms"]["mod"]["clipped_fraction_native"],
                         entry["step_norms"]["mod"]["clipped_fraction_transplant"]]
                        for entry in entries])
    decay = np.array([entry["step_norms"]["mod"]["decay_term"]
                      / entry["step_norms"]["mod"]["native_step"] for entry in entries
                      if entry["step_norms"]["mod"]["native_step"] >= ZERO_STEP_NORM])
    checks = np.array([[entry["native_check"]["max_abs_hidden_diff"],
                        entry["native_check"]["max_abs_m_diff"]] for entry in entries])
    print(f"\nM entries at the bounds after the step: native median {np.median(clipped[:, 0]):.3f}, "
          f"transplanted median {np.median(clipped[:, 1]):.3f}")
    print(f"Decay term / native M step: mean {_mean_angle(decay):.3f}, "
          f"max {decay.max() if decay.size else float('nan'):.3f}")
    print(f"Self-transplant max abs difference: hidden {checks[:, 0].max():.2e}, M {checks[:, 1].max():.2e}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Modulation transplant experiment of the multi-task runs (module docstring).")
    parser.add_argument(
        "--checkpoint", nargs="+", metavar="PATH", default=None,
        help="analyze only these savednet_*.pt checkpoints and merge them into the "
             "results pickle (default: the whole state-space cohort, from scratch)")
    parser.add_argument(
        "--out-dir", default=str(shift.STATE_SPACE_DIR), metavar="DIR",
        help=f"state-space directory; outputs go to its {TRANSPLANT_DIR_NAME}/ subdirectory "
             f"(default: {shift.STATE_SPACE_DIR}/)")
    parser.add_argument(
        "--untrained", action="store_true",
        help="instead analyze untrained networks with the config of each --checkpoint "
             "(fresh weights from --init-seed), written to "
             f"<out-dir>/{TRANSPLANT_DIR_NAME}/{shift.UNTRAINED_DIR_NAME}/")
    parser.add_argument(
        "--init-seed", type=int, nargs="+", default=None, metavar="SEED",
        help=f"seeds of the untrained weights, one network per seed (default: "
             f"{shift.UNTRAINED_INIT_SEED}); only with --untrained")
    parser.add_argument(
        "--evaluation-seed", type=int, default=shift.EVALUATION_SEED, metavar="SEED",
        help="seed of the evaluation trials; another value writes a results pickle named "
             f"after it (default: {shift.EVALUATION_SEED}, the state-space trials)")
    args = parser.parse_args(argv)
    if args.untrained and not args.checkpoint:
        parser.error("--untrained needs --checkpoint to name the run whose config it uses")
    if args.init_seed is not None and not args.untrained:
        parser.error("--init-seed only applies with --untrained")
    if args.init_seed is None:
        args.init_seed = [shift.UNTRAINED_INIT_SEED]
    return args


def main(argv=None):
    args = _parse_args(argv)
    output_dir = Path(args.out_dir)
    if args.untrained:
        results = run_untrained(args.checkpoint, output_dir, args.init_seed, args.evaluation_seed)
        summarize(results, "untrained networks")
        return
    if args.checkpoint:
        results = run_checkpoints(args.checkpoint, output_dir, args.evaluation_seed)
    else:
        results = run_all(output_dir, args.evaluation_seed)
    summarize(results, "trained runs")


if __name__ == "__main__":
    main()
