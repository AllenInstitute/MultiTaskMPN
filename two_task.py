#!/usr/bin/env python
# coding: utf-8
"""
Two-task training for a DeepMultiPlasticNet.

Trains an MPN on a PAIR of cognitive tasks (a go/anti family, e.g.
`delaygofamily` = [delaygo, delayanti]) and saves everything the downstream
analysis (two_task_analysis.py) needs.

This is the "run the network and save the result" half of what used to be the
monolithic `two_task_analysis.ipynb`. It mirrors the split already done for the
single-task pipeline (one_task.py / one_task_analysis.py): this script trains
and records, two_task_analysis.py reloads and analyzes.

Unlike one_task_analysis.py — which is purely trace-driven — the two-task
analysis repeatedly needs the LIVE trained network (it runs the net on freshly
interpolated inputs, prunes its weights, reads its weight matrices). So in
addition to the per-stage training traces, this script saves a full network
checkpoint that two_task_analysis.py rebuilds and runs.

Outputs (one self-contained subfolder per trial: ./twotasks/{aname}/):
  savednet_{aname}.pt         — state_dict + (converted) params + hyp_dict
  param_{aname}_param.json    — human-readable hyperparameters
  bundle_{aname}.pkl          — full training bundle + test datasets + labels
  loss_{ruleset}_seed{seed}_{addon}.png  — training accuracy / loss curve

`aname` shared identifier (matches the notebook's filename convention):
  {ruleset}_seed{seed}_{addon_name}      # addon_name already embeds +hidden{n}
"""
import gc
import time
import copy
import random
import json
import pickle
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networks as nets
import net_helpers
import mpn_tasks
import helper
import mpn

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ─── Experiment-wide configuration ───────────────────────────────────────────
# Number of independent trials (different random seeds) to train.
N_TRIALS = 1
# Fixed list of seeds, or None to draw N_TRIALS random seeds.
SEED_LIST = None

RULESET = 'delaygofamily'      # go/anti pair, e.g. [delaygo, delayanti]
CHOSEN_NETWORK = "dmpn"
N_HIDDEN = 200
ADDON_NAME = "reg1e4"   # +hidden{N_HIDDEN} appended automatically below
train = True
verbose = True

# Plotting palette (notebook cell 2) — kept here only for the loss curve.
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795',
          '#718096', '#d53f8c', '#d69e2e'] * 10
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9',
            '#e2e8f0', '#fbb6ce', '#faf089'] * 10

accept_rules = ('fdgo', 'fdanti', 'delaygo', 'delayanti', 'reactgo', 'reactanti',
                'delaydm1', 'delaydm2', 'dmsgo', 'dmcgo', 'contextdelaydm1',
                'contextdelaydm2', 'multidelaydm')

rules_dict = {
    'all': ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
            'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
            'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
            'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
    'low_dim': ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
    'gofamily': ['fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti'],
    'delaygo': ['delaygo'],
    'delaygofamily': ['delaygo', 'delayanti'],
    'fdgo': ['fdgo'],
    'fdfamily': ['fdgo', 'fdanti'],
    'reactgo': ['reactgo'],
    'reactfamily': ['reactgo', 'reactanti'],
    'delaydm1': ['delaydm1'],
    'delaydmfamily': ['delaydm1', 'delaydm2'],
    'dmsgofamily': ['dmsgo', 'dmsnogo'],
    'dmsgo': ['dmsgo'],
    'dmcgo': ['dmcgo'],
    'contextdelayfamily': ['contextdelaydm1', 'contextdelaydm2'],
}

mpn_depth = 1
if CHOSEN_NETWORK in ("gru", "vanilla"):
    mpn_depth = 1


def current_basic_params(hyp_dict):
    """Task / training / network hyperparameters (notebook cell 3)."""
    task_params = {
        'task_type': hyp_dict['task_type'],
        'rules': rules_dict[hyp_dict['ruleset']],
        'dt': 40,
        'ruleset': hyp_dict['ruleset'],
        'n_eachring': 8,
        'in_out_mode': 'low_dim',
        'sigma_x': 0.00,
        'mask_type': 'cost',
        'fixate_off': False,
        'task_info': True,
        'randomize_inputs': False,
        'n_input': 20,
        'modality_diff': False,
        'label_strength': False,
        'long_delay': 'normal',
        'long_response': 'normal',
        'long_stimulus': 'normal',
        'long_fixation': 'normal',
        'adjust_task_prop': True,
        'adjust_task_decay': 0.9,
    }

    print(f"Fixation_off: {task_params['fixate_off']}; Task_info: {task_params['task_info']}")

    train_params = {
        'lr': 1e-3,
        'n_batches': 128,
        'batch_size': 128,
        'gradient_clip': 10,
        'valid_n_batch': 60,
        'n_datasets': 6000,
        'valid_check': None,
        'n_epochs_per_set': 1,
        'weight_reg': 'L2',
        'activity_reg': 'L2',
        'reg_lambda': 1e-4,

        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-8,
            'step_size': 30,
            'gamma': 0.1,
        },
    }

    if not train:
        assert train_params['n_epochs_per_set'] == 0

    net_params = {
        'net_type': hyp_dict['chosen_network'],
        'n_neurons': [1] + [N_HIDDEN] * mpn_depth + [1],
        'linear_embed': 200,
        'output_bias': False,
        'loss_type': 'MSE',
        'activation': 'tanh',
        'cuda': True,
        'monitor_freq': train_params["n_epochs_per_set"],
        'monitor_valid_out': True,
        'output_matrix': '',
        'input_layer_add': True,
        'input_layer_add_trainable': True,
        'input_init_type': "xavier",
        'input_layer_bias': False,
        'input_layer': "trainable",
        'acc_measure': 'angle',

        'ml_params': {
            'bias': True,
            'mp_type': 'mult',
            'm_activation': 'linear',
            'm_update_type': 'hebb_assoc',
            'eta_type': 'scalar',
            'eta_train': False,
            'lam_type': 'scalar',
            'm_time_scale': 4000,
            'lam_train': False,
            'W_freeze': False,
        },

        'leaky': True,
        'alpha': 0.2,
    }

    assert not (task_params["randomize_inputs"] and net_params["input_layer_add"]), (
        "task_params['randomize_inputs'] and net_params['input_layer_add'] cannot both be True."
    )

    if mpn_depth > 1:
        for mpl_idx in range(mpn_depth - 1):
            assert f'ml_params{mpl_idx}' in net_params.keys()

    if hyp_dict['chosen_network'] in ("gru", "vanilla"):
        assert 'ml_params' in net_params.keys()

    return task_params, train_params, net_params


# Local 2-value variant of generate_response_stimulus (notebook cell 5). The
# shared helper.generate_response_stimulus returns 4 values; the two-task
# analysis pipeline relies on this (resp, stim) form (see make_label_task_comb).
def generate_response_stimulus(task_params, test_trials, ruleset):
    labels_resp, labels_stim = [], []
    for rule_idx, rule in enumerate(task_params['rules']):
        print(rule)
        if rule in accept_rules:
            labels_resp.append(test_trials[rule_idx].meta['resp1'])
            labels_stim.append(test_trials[rule_idx].meta['stim1'])
        else:
            raise NotImplementedError()
    labels_resp = np.concatenate(labels_resp, axis=0).reshape(-1, 1)
    labels_stim = np.concatenate(labels_stim, axis=0).reshape(-1, 1)
    return labels_resp, labels_stim


def make_label_task_comb(task_params, test_trials, test_task, ruleset, color_by="stim"):
    """[stimulus-or-response-label, task_id] pair per trial (notebook cell 20)."""
    labels_resp, labels_stim = generate_response_stimulus(task_params, test_trials, ruleset)
    labels_ = labels_stim if color_by == "stim" else labels_resp
    return np.column_stack((labels_[:, 0], test_task))


OUT_DIR = Path("twotasks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_trial(seed):
    """Train one independent two-task network and save its full training bundle
    plus a checkpoint that two_task_analysis.py can rebuild and run."""
    print(f"\n{'='*70}\nTrial seed = {seed}\n{'='*70}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    hyp_dict = {
        'task_type': 'multitask',
        'mode_for_all': "random_batch",
        'ruleset': RULESET,
        'run_mode': 'minimal',
        'chosen_network': CHOSEN_NETWORK,
        'addon_name': ADDON_NAME + f"+hidden{N_HIDDEN}",
        'mess_with_training': False,
    }

    task_params, train_params, net_params = current_basic_params(hyp_dict)

    aname = f"{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}"
    print(f"aname: {aname}")

    # All outputs for this trial (param, loss curve, checkpoint, bundle) go in a
    # per-run subfolder twotasks/{aname}/ so each trial is self-contained.
    run_dir = OUT_DIR / aname
    run_dir.mkdir(parents=True, exist_ok=True)

    shift_index = 1 if not task_params['fixate_off'] else 0

    # Persist human-readable hyperparameters BEFORE convert_and_init_multitask_params
    # (which injects a non-serialisable RandomState into task_params['hp']), so a
    # crash still leaves a clean record — same ordering as one_task.py.
    config = {"task_params": task_params, "train_params": train_params, "net_params": net_params}
    with (run_dir / f"param_{aname}_param.json").open("w") as f:
        json.dump(config, f, indent=4, default=helper.as_jsonable)

    if hyp_dict['task_type'] in ('multitask',):
        task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
            (task_params, train_params, net_params)
        )
        net_params['prefs'] = mpn_tasks.get_prefs(task_params['hp'])
        print('Rules: {}'.format(task_params['rules']))
        print('  Input size {}, Output size {}'.format(task_params['n_input'], task_params['n_output']))
    else:
        raise NotImplementedError()

    device = torch.device('cuda') if net_params['cuda'] and torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device}")

    epoch_multiply = train_params["n_epochs_per_set"]

    params = task_params, train_params, net_params

    if net_params['net_type'] == 'mpn1':
        netFunction = mpn.MultiPlasticNet
    elif net_params['net_type'] == 'dmpn':
        netFunction = mpn.DeepMultiPlasticNet
    elif net_params['net_type'] == 'vanilla':
        netFunction = nets.VanillaRNN
    elif net_params['net_type'] == 'gru':
        netFunction = nets.GRU

    # ─── Test/validation datasets ────────────────────────────────────────────
    # The normal trial plus 4 long-period variants (used for fixed-point /
    # attractor analyses). Per-stage traces saved below are aligned to exactly
    # these trials, so the variants are saved and reloaded by the analysis.
    test_n_batch = train_params["valid_n_batch"]
    color_by = "stim"

    task_params['hp']['batch_size_train'] = test_n_batch
    test_mode_for_all = "random"

    # Align the epoch timing of the two tasks so their periods (fixation /
    # stimulus / delay / response) coincide. Without this, generate_trials_wrap
    # draws an independent random timeline per rule, so e.g. delaygo and
    # delayanti would have different delay-end timesteps while the analysis uses
    # a single shared timeline — comparing the two tasks at mismatched times.
    # align_periods resets the shared RNG before each rule so both draw identical
    # scalar timing (requires mode_input='random').
    task_random_fix = True
    if task_random_fix:
        print(f"Align {task_params['rules']} With Same Time")

    test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
        task_params, test_n_batch, rules=task_params['rules'],
        mode_input=test_mode_for_all, device=device,
        align_periods=task_random_fix,
    )
    _, test_trials, test_rule_idxs = test_trials_extra

    def gen_long(period_key):
        tp = copy.deepcopy(task_params)
        tp[period_key] = "long"
        data, extra = mpn_tasks.generate_trials_wrap(
            tp, test_n_batch, rules=tp['rules'],
            mode_input=test_mode_for_all, device=device,
            align_periods=task_random_fix,
        )
        _, trials, _ = extra
        return tp, data, trials

    task_params_longdelay, test_data_longdelay, test_trials_longdelay = gen_long("long_delay")
    task_params_longresponse, test_data_longresponse, test_trials_longresponse = gen_long("long_response")
    task_params_longstimulus, test_data_longstimulus, test_trials_longstimulus = gen_long("long_stimulus")
    task_params_longfixation, test_data_longfixation, test_trials_longfixation = gen_long("long_fixation")

    task_params['dataset_name'] = 'multitask'

    if task_params['in_out_mode'] not in ('low_dim', 'low_dim_pos'):
        raise NotImplementedError()

    labels_resp, labels_stim = generate_response_stimulus(task_params, test_trials, hyp_dict['ruleset'])
    labels = labels_stim if color_by == "stim" else labels_resp

    test_input, test_output, test_mask = test_data
    test_input_longfixation, test_output_longfixation, test_mask_longfixation = test_data_longfixation
    test_input_longstimulus, test_output_longstimulus, test_mask_longstimulus = test_data_longstimulus
    test_input_longdelay, test_output_longdelay, test_mask_longdelay = test_data_longdelay
    test_input_longresponse, test_output_longresponse, test_mask_longresponse = test_data_longresponse

    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()

    # Task id per trial for each variant.
    test_task = helper.find_task(task_params, test_input_np, shift_index)
    test_task_longfixation = helper.find_task(task_params_longfixation, test_input_longfixation.detach().cpu().numpy(), shift_index)
    test_task_longstimulus = helper.find_task(task_params_longstimulus, test_input_longstimulus.detach().cpu().numpy(), shift_index)
    test_task_longdelay = helper.find_task(task_params_longdelay, test_input_longdelay.detach().cpu().numpy(), shift_index)
    test_task_longresponse = helper.find_task(task_params_longresponse, test_input_longresponse.detach().cpu().numpy(), shift_index)

    # [label, task] pairs per variant (notebook cell 20).
    label_task_comb = make_label_task_comb(task_params, test_trials, test_task, hyp_dict['ruleset'], color_by)
    label_task_comb_longdelay = make_label_task_comb(task_params_longdelay, test_trials_longdelay, test_task_longdelay, hyp_dict['ruleset'], color_by)
    label_task_comb_longresponse = make_label_task_comb(task_params_longresponse, test_trials_longresponse, test_task_longresponse, hyp_dict['ruleset'], color_by)
    label_task_comb_longstimulus = make_label_task_comb(task_params_longstimulus, test_trials_longstimulus, test_task_longstimulus, hyp_dict['ruleset'], color_by)
    label_task_comb_longfixation = make_label_task_comb(task_params_longfixation, test_trials_longfixation, test_task_longfixation, hyp_dict['ruleset'], color_by)

    # ─── Train ────────────────────────────────────────────────────────────────
    start_time = time.time()
    net, _, (counter_lst, netout_lst, db_lst, Winput_lst, Winputbias_lst,
             Woutput_lst, Wall_lst, marker_lst, loss_lst, acc_lst), _ = net_helpers.train_network(
        params, device=device, verbose=verbose, train=train, hyp_dict=hyp_dict,
        netFunction=netFunction,
        test_input=[test_input, test_input_longfixation, test_input_longstimulus,
                    test_input_longdelay, test_input_longresponse],
        print_frequency=100,
    )
    end_time = time.time()
    print(f"Running Time: {end_time - start_time}")
    counter_lst = [x * epoch_multiply + 1 for x in counter_lst]  # avoid log plot issue

    # ─── Loss / accuracy curve (notebook cell 10) ──────────────────────────────
    if train:
        figl, axl = plt.subplots(1, 1, figsize=(3, 3))
        axl.plot(net.hist['iters_monitor'][1:], net.hist['train_acc'][1:], color=c_vals[0], label='Full train accuracy')
        axl.plot(net.hist['iters_monitor'][1:], net.hist['valid_acc'][1:], color=c_vals[1], label='Full valid accuracy')
        if net.weight_reg is not None:
            axl.plot(net.hist['iters_monitor'], net.hist['train_loss_output_label'], color=c_vals_l[0], zorder=-1, label='Output label')
            axl.plot(net.hist['iters_monitor'], net.hist['train_loss_reg_term'], color=c_vals_l[0], zorder=-1, label='Reg term', linestyle='dashed')
            axl.plot(net.hist['iters_monitor'], net.hist['valid_loss_output_label'], color=c_vals_l[1], zorder=-1, label='Output valid label')
            axl.plot(net.hist['iters_monitor'], net.hist['valid_loss_reg_term'], color=c_vals_l[1], zorder=-1, label='Reg valid term', linestyle='dashed')
        axl.legend()
        axl.set_ylim([0.90, 1.05])
        axl.set_ylabel('Accuracy')
        axl.set_xlabel('# Batches')
        figl.savefig(run_dir / f"loss_{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}.png", dpi=300)
        plt.close(figl)
        print('Done!')

    # Sanity checks (notebook cell 9).
    if net_params["ml_params"]["W_freeze"]:
        assert np.allclose(Wall_lst[-1][0], Wall_lst[0][0])
    if net_params["input_layer_bias"]:
        assert net_params["input_layer_add"] is True

    if train:
        net_helpers.net_eta_lambda_analysis(net, net_params, hyp_dict)

    # ─── Save checkpoint (rebuildable by two_task_analysis.py) ─────────────────
    net_path = run_dir / f"savednet_{aname}.pt"
    torch.save({
        "state_dict": net.state_dict(),
        "net_params": net_params,
        "task_params": task_params,
        "train_params": train_params,
        "hyp_dict": hyp_dict,
    }, net_path)
    print(f"Saved network: {net_path}")

    # ─── Save training bundle + test datasets + labels ────────────────────────
    def to_np(t):
        return t.detach().cpu().numpy()

    # Slim the giant per-stage traces down to exactly what two_task_analysis.py
    # reads, before pickling. db_lst / netout_lst are indexed [variant][stage].
    #   • db_lst: the normal variant (index 0) is consumed across ALL training
    #     stages (the attractor-over-learning curves); the 4 long variants are
    #     only ever read at the final stage. The long variants (T ~ 5x longer)
    #     saved across every stage are what blow the bundle up to >100 GB, so we
    #     keep only their final stage.
    #   • netout_lst: only ever read at the final stage, for every variant.
    # Both keep their [variant][stage] nesting so the analysis's [-1] / [0]
    # indexing stays valid unchanged.
    db_lst = [db_lst[0]] + [[stages[-1]] for stages in db_lst[1:]]
    netout_lst = [[stages[-1]] for stages in netout_lst]

    bundle = {
        "aname": aname,
        "seed": seed,
        "hyp_dict": hyp_dict,
        "shift_index": shift_index,
        "color_by": color_by,
        "epoch_multiply": epoch_multiply,
        "test_mode_for_all": test_mode_for_all,

        # Training traces (all 5 test inputs × all recorded stages).
        "counter_lst": counter_lst,
        "netout_lst": netout_lst,
        "db_lst": db_lst,
        "Winput_lst": Winput_lst,
        "Winputbias_lst": Winputbias_lst,
        "Woutput_lst": Woutput_lst,
        "Wall_lst": Wall_lst,
        "marker_lst": marker_lst,
        "loss_lst": loss_lst,
        "acc_lst": acc_lst,

        # Test datasets (numpy) — normal + 4 long variants.
        "test_input_np": to_np(test_input),
        "test_output_np": to_np(test_output),
        "test_mask_np": to_np(test_mask),
        "test_input_longfixation_np": to_np(test_input_longfixation),
        "test_output_longfixation_np": to_np(test_output_longfixation),
        "test_input_longstimulus_np": to_np(test_input_longstimulus),
        "test_output_longstimulus_np": to_np(test_output_longstimulus),
        "test_input_longdelay_np": to_np(test_input_longdelay),
        "test_output_longdelay_np": to_np(test_output_longdelay),
        "test_input_longresponse_np": to_np(test_input_longresponse),
        "test_output_longresponse_np": to_np(test_output_longresponse),

        # Derived labels / task ids / pairs per variant.
        "labels": labels,
        "labels_stim": labels_stim,
        "labels_resp": labels_resp,
        "test_task": test_task,
        "test_task_longfixation": test_task_longfixation,
        "test_task_longstimulus": test_task_longstimulus,
        "test_task_longdelay": test_task_longdelay,
        "test_task_longresponse": test_task_longresponse,
        "label_task_comb": label_task_comb,
        "label_task_comb_longfixation": label_task_comb_longfixation,
        "label_task_comb_longstimulus": label_task_comb_longstimulus,
        "label_task_comb_longdelay": label_task_comb_longdelay,
        "label_task_comb_longresponse": label_task_comb_longresponse,
    }

    bundle_path = run_dir / f"bundle_{aname}.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved bundle: {bundle_path}")
    print(f"Done trial seed={seed}!")

    # Free memory before the next trial.
    del net, db_lst, netout_lst, Wall_lst, Woutput_lst, Winput_lst, bundle
    del test_input, test_output, test_mask, test_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return aname


# ─── Run K independent trials ────────────────────────────────────────────────
if __name__ == "__main__":
    if SEED_LIST is not None:
        seeds = list(SEED_LIST)
    else:
        rng = random.Random()  # fresh entropy each run -> different seed pool
        seeds = rng.sample(range(1, 1000), N_TRIALS)

    print(f"Running {len(seeds)} independent trials: seeds={seeds}")
    anames = []
    for seed in seeds:
        try:
            anames.append(run_trial(seed))
        except Exception as exc:
            print(f"Trial seed={seed} FAILED: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\nCompleted {len(anames)}/{len(seeds)} trials.")
    for a in anames:
        print(f"  {a}")

    # Manifest of the runs produced THIS invocation, so a pipeline can analyze
    # exactly what was just trained (rather than every run on disk).
    manifest_path = OUT_DIR / "last_run_anames.txt"
    with manifest_path.open("w") as mf:
        mf.write("\n".join(anames) + ("\n" if anames else ""))
    print(f"Wrote manifest: {manifest_path}")
