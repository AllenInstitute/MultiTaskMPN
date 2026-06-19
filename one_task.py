#!/usr/bin/env python
# coding: utf-8
"""
Single-task training for a DeepMultiPlasticNet.

Trains an MPN on ONE cognitive task (e.g. delaydm1) and saves everything the
downstream analysis needs. Unlike multiple_task_analysis.py — which re-evaluates
the final network — one_task_analysis.py inspects how the weights and the fast
modulation matrix M evolve ACROSS training, so this script records and saves the
full per-checkpoint ("stage") traces produced by net_helpers.train_network.

Outputs (under ./onetask/):
  savednet_{aname}.pt         — final network state_dict + net_params
  param_{aname}_param.json    — task/train/net hyperparameters
  param_{aname}_result.npz    — per-stage traces + test data needed for analysis
  loss_{aname}.png            — training accuracy / loss curve

`aname` shared identifier:
  {ruleset}_seed{seed}_{addon}hidden{hidden}+batch{batch}+{acc_measure}
"""
import gc
import random
import json
from pathlib import Path

import numpy as np
import torch

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
N_TRIALS = 3
# Fixed list of seeds, or None to draw N_TRIALS random seeds.
SEED_LIST = None

RULESET = 'delaygo'
CHOSEN_NETWORK = "dmpn"
ADDON_NAME = ""
train = True
verbose = True

accept_rules = ('fdgo', 'fdanti', 'delaygo', 'delayanti', 'reactgo', 'reactanti',
                'delaydm1', 'delaydm2', 'dmsgo', 'dmcgo', 'contextdelaydm1',
                'contextdelaydm2', 'multidelaydm', 'dm1')

rules_dict = {
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
    'contextdelaydm1': ['contextdelaydm1'],
    'contextdelayfamily': ['contextdelaydm1', 'contextdelaydm2'],
}

mpn_depth = 1
if CHOSEN_NETWORK in ("gru", "vanilla"):
    mpn_depth = 1


def current_basic_params(hyp_dict):
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
        'long_fixation': 'normal',
        'long_stimulus': 'normal',
        'long_delay': 'normal',
        'long_response': 'normal',
        'adjust_task_prop': True,
        'adjust_task_decay': 0.9,
    }

    print(f"Fixation_off: {task_params['fixate_off']}; Task_info: {task_params['task_info']}")

    train_params = {
        'lr': 1e-3,
        'n_batches': 128,
        'batch_size': 128,
        'gradient_clip': 10,
        'valid_n_batch': 50,
        'n_datasets': 3000,
        'valid_check': None,
        'n_epochs_per_set': 1,
        'task_mask': None,
        'weight_reg': 'L2',
        'activity_reg': 'L2',
        'reg_lambda': 1e-4,

        'scheduler': {
            'type': 'ReduceLROnPlateau',  # or 'StepLR'
            'mode': 'min',                # for ReduceLROnPlateau
            'factor': 0.95,               # factor to reduce LR
            'patience': 30,               # epochs to wait before reducing LR
            'min_lr': 1e-8,
            'step_size': 30,              # for StepLR (step every 30 datasets)
            'gamma': 0.1                  # for StepLR (multiply LR by 0.1)
        },
    }

    if not train:
        assert train_params['n_epochs_per_set'] == 0

    n_hidden = 200
    linear_embed = n_hidden

    net_params = {
        'net_type': hyp_dict['chosen_network'],
        'n_neurons': [1] + [n_hidden] * mpn_depth + [1],
        'linear_embed': linear_embed,
        'output_bias': False,
        'loss_type': 'MSE',
        'activation': 'tanh',
        'cuda': True,
        'monitor_freq': train_params["n_epochs_per_set"],
        'monitor_valid_out': True,
        'output_matrix': '',
        'input_layer_add': True,
        'input_layer_add_trainable': True,
        'input_layer_bias': False,
        'input_layer': "trainable",
        'acc_measure': 'angle',
        'ml_params': {
            'bias': True,
            'mp_type': 'mult',
            'm_update_type': 'hebb_assoc',
            'eta_type': 'scalar',
            'eta_train': False,
            'lam_type': 'scalar',
            'm_time_scale': 400,
            'lam_train': False,
            'W_freeze': False,
        },
        'leaky': True,
        'alpha': 0.2,
    }

    return task_params, train_params, net_params


OUT_DIR = Path("onetask")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_trial(seed):
    """Train one independent network on the chosen single task and save its
    per-stage traces / checkpoint. Each trial uses its own random seed."""
    print(f"\n{'='*70}\nTrial seed = {seed}\n{'='*70}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    hyp_dict = {
        'task_type': 'multitask',
        'mode_for_all': "random_batch",
        'ruleset': RULESET,
        'run_mode': 'minimal',
        'chosen_network': CHOSEN_NETWORK,
        'addon_name': ADDON_NAME,
    }

    task_params, train_params, net_params = current_basic_params(hyp_dict)

    # Shared identifier (see module docstring).
    n_hidden = net_params['n_neurons'][1]
    aname = (f"{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}"
             f"hidden{n_hidden}+batch{train_params['n_batches']}+{net_params['acc_measure']}")
    print(f"aname: {aname}")

    # Persist hyperparameters before training so a crash still leaves a record.
    config = {"task_params": task_params, "train_params": train_params, "net_params": net_params}
    with (OUT_DIR / f"param_{aname}_param.json").open("w") as f:
        json.dump(config, f, indent=4, default=helper.as_jsonable)

    shift_index = 1 if not task_params['fixate_off'] else 0

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

    params = task_params, train_params, net_params

    if net_params['net_type'] == 'mpn1':
        netFunction = mpn.MultiPlasticNet
    elif net_params['net_type'] == 'dmpn':
        netFunction = mpn.DeepMultiPlasticNet
    elif net_params['net_type'] == 'vanilla':
        netFunction = nets.VanillaRNN
    elif net_params['net_type'] == 'gru':
        netFunction = nets.GRU

    # ─── Test/validation dataset ──────────────────────────────────────────────────
    test_n_batch = train_params["valid_n_batch"]
    color_by = "stim"

    task_params['hp']['batch_size_train'] = test_n_batch
    test_mode_for_all = "random"
    test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
        task_params, test_n_batch, rules=task_params['rules'],
        mode_input=test_mode_for_all, device=device,
    )
    _, test_trials, test_rule_idxs = test_trials_extra
    task_params['dataset_name'] = 'multitask'

    if task_params['in_out_mode'] == 'low_dim':
        output_dim_labels = ('Fixate', 'Cos', 'Sin')
    elif task_params['in_out_mode'] == 'low_dim_pos':
        output_dim_labels = ('Fixate', 'Cos', '-Cos', 'Sin', '-Sin')
    else:
        raise NotImplementedError()

    labels = []
    for rule_idx, rule in enumerate(task_params['rules']):
        print(rule)
        if rule not in accept_rules:
            raise NotImplementedError()
        if hyp_dict['ruleset'] in ('dmsgo', 'dmcgo'):
            labels.append(test_trials[rule_idx].meta['matches'])
        else:
            labels.append(test_trials[rule_idx].meta['resp1' if color_by == "resp" else 'stim1'])
    labels = np.concatenate(labels, axis=0).reshape(-1, 1)

    test_input, test_output, test_mask = test_data

    permutation = np.random.permutation(test_input.shape[0])
    test_input = test_input[permutation]
    test_output = test_output[permutation]
    test_mask = test_mask[permutation]
    labels = labels[permutation]

    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()
    print(f"test_input.shape: {test_input.shape}")

    # ─── Train ────────────────────────────────────────────────────────────────────
    net, _, (counter_lst, netout_lst, db_lst, Winput_lst, Winputbias_lst,
             Woutput_lst, Wall_lst, marker_lst, loss_lst, acc_lst), _ = net_helpers.train_network(
        params, device=device, verbose=verbose, train=train, hyp_dict=hyp_dict,
        netFunction=netFunction, test_input=[test_input], print_frequency=100,
        record_frequency=8
    )

    # Training-history arrays (train/valid accuracy + loss components across
    # batches). Saved for one_task_analysis.py to render the training curve.
    # Only the scalar-per-monitor-step keys the curve needs are kept; some
    # net.hist entries are ragged (variable-length per step) and would not
    # convert to a clean array.
    _hist_keys = [
        "iters_monitor", "train_acc", "valid_acc",
        "train_loss_output_label", "train_loss_reg_term",
        "valid_loss_output_label", "valid_loss_reg_term",
    ]
    net_hist = {}
    if train:
        for k in _hist_keys:
            if k in net.hist:
                try:
                    net_hist[k] = np.asarray(net.hist[k], dtype=float)
                except (ValueError, TypeError):
                    print(f"  [warn] skipping ragged net.hist['{k}']")
    weight_reg_on = net.weight_reg is not None

    if train:
        # net_eta_lambda_analysis writes to ./results/; ensure it exists.
        Path("results").mkdir(parents=True, exist_ok=True)
        net_helpers.net_eta_lambda_analysis(net, net_params, hyp_dict)

    # ─── Extract per-stage modulation traces ──────────────────────────────────────
    # layer_index = 1 because input_layer_add prepends a linear input layer.
    layer_index = 1 if net_params["input_layer_add"] else 0
    max_seq_len = test_input.shape[1]
    n_batch_all = test_input.shape[0]


    def modulation_extraction(db, layer_index):
        """Pull (Ms_orig, hs, bs, xs) for one recorded stage's activation dict."""
        Ms_orig = db[f'M{layer_index}']
        bs = db[f'b{layer_index}']
        hs = db[f'hidden{layer_index}'].reshape(n_batch_all, max_seq_len, -1)
        xs = db[f'input{layer_index}'].reshape(n_batch_all, max_seq_len, -1)
        return Ms_orig, hs, bs, xs


    stages_num = len(Wall_lst)
    print(f"Recorded {stages_num} training stages")

    Ms_orig_stages, hs_stages, bs_stages, xs_stages = [], [], [], []
    Wall_stages, Woutput_stages, Winput_stages = [], [], []
    for stage_iter in range(stages_num):
        Ms_orig_s, hs_s, bs_s, xs_s = modulation_extraction(db_lst[0][stage_iter], layer_index)
        Ms_orig_stages.append(np.asarray(Ms_orig_s))
        hs_stages.append(np.asarray(hs_s))
        bs_stages.append(np.asarray(bs_s))
        xs_stages.append(np.asarray(xs_s))
        Wall_stages.append(np.asarray(Wall_lst[stage_iter][0]))
        Woutput_stages.append(np.asarray(Woutput_lst[stage_iter]))
        Winput_stages.append(np.asarray(Winput_lst[stage_iter]) if net_params["input_layer_add"] else None)

    Ms_orig_stages = np.stack(Ms_orig_stages, axis=0)
    hs_stages = np.stack(hs_stages, axis=0)
    bs_stages = np.stack(bs_stages, axis=0)
    xs_stages = np.stack(xs_stages, axis=0)
    Wall_stages = np.stack(Wall_stages, axis=0)
    Woutput_stages = np.stack(Woutput_stages, axis=0)
    if net_params["input_layer_add"]:
        Winput_stages = np.stack(Winput_stages, axis=0)

    # Final-stage network output on the test set (for I/O trace figures).
    net_out_final = np.asarray(netout_lst[0][-1])
    # Final-stage input weight matrix (W_initial_linear) for the heatmap figure.
    input_matrix_final = (np.asarray(Winput_lst[-1]) if net_params["input_layer_add"] else np.array([]))

    # ─── Epoch / period breakdown (needed for period-wise analysis) ───────────────
    recordkyle_all = []
    for test_subtrial in test_trials:
        metaepoch = test_subtrial.epochs
        periodname = list(metaepoch.keys())
        recordkyle = []
        for keyiter in range(len(periodname)):
            try:
                if test_mode_for_all == "random":
                    recordkyle.append(metaepoch[periodname[keyiter]][1])
                elif test_mode_for_all == "random_batch":
                    recordkyle.append(list(metaepoch[periodname[keyiter]][1]))
            except Exception as e:
                print(e)
        if test_mode_for_all == "random":
            fillrecordkyle = [[timestamp for _ in range(hs_stages.shape[1])] for timestamp in recordkyle]
            recordkyle = fillrecordkyle
        recordkyle.insert(0, [0 for _ in range(len(recordkyle[1]))])
        recordkyle = np.array(recordkyle).T.tolist()
        recordkyle_all.extend(recordkyle)

    unique_recordkyle_all = [list(lst) for lst in set(tuple(lst) for lst in recordkyle_all)]
    all_session_breakdown = []
    for task_specific_time in unique_recordkyle_all:
        session_breakdown = []
        for sindex in range(0, len(task_specific_time) - 1):
            session_breakdown.append([task_specific_time[sindex], task_specific_time[sindex + 1]])
        session_breakdown.append([task_specific_time[0], task_specific_time[-1]])
        all_session_breakdown.append(session_breakdown)

    all_breaks = []
    for session_breakdown in all_session_breakdown:
        breaks = [cut[1] for cut in session_breakdown[:-1]]
        all_breaks.append(breaks)

    assert len(all_breaks)
    response_start = all_breaks[0][-2]
    stimulus_start = all_breaks[0][0]
    stimulus_end = all_breaks[0][1]
    print(f"response_start={response_start}, stimulus_start={stimulus_start}, stimulus_end={stimulus_end}")

    # ─── Save everything analysis needs ───────────────────────────────────────────
    result_path = OUT_DIR / f"param_{aname}_result.npz"
    np.savez_compressed(
        result_path,
        hyp_dict=hyp_dict,
        seed=seed,
        shift_index=shift_index,
        color_by=color_by,
        layer_index=layer_index,
        max_seq_len=max_seq_len,
        counter_lst=np.asarray(counter_lst),
        marker_lst=np.asarray(marker_lst),
        loss_lst=np.asarray(loss_lst),
        acc_lst=np.asarray(acc_lst),
        net_hist=net_hist,
        weight_reg_on=weight_reg_on,
        test_input_np=test_input_np,
        test_output_np=test_output_np,
        net_out_final=net_out_final,
        input_matrix_final=input_matrix_final,
        network_at_percent=(marker_lst[-1] + 1) / train_params['n_datasets'] * 100,
        labels=labels,
        Ms_orig_stages=Ms_orig_stages,
        hs_stages=hs_stages,
        bs_stages=bs_stages,
        xs_stages=xs_stages,
        Wall_stages=Wall_stages,
        Woutput_stages=Woutput_stages,
        Winput_stages=Winput_stages if net_params["input_layer_add"] else np.array([]),
        all_breaks=np.array(all_breaks, dtype=object),
        response_start=response_start,
        stimulus_start=stimulus_start,
        stimulus_end=stimulus_end,
    )
    print(f"Saved traces: {result_path}")

    # Final network checkpoint.
    net_path = OUT_DIR / f"savednet_{aname}.pt"
    torch.save({"state_dict": net.state_dict(), "net_params": net_params}, net_path)
    print(f"Saved network: {net_path}")
    print(f"Done trial seed={seed}!")

    # Free memory before the next trial so traces don't compound.
    del net, db_lst, netout_lst, Wall_lst, Woutput_lst, Winput_lst
    del Ms_orig_stages, hs_stages, bs_stages, xs_stages
    del Wall_stages, Woutput_stages, Winput_stages
    del test_input, test_output, test_mask, test_data, test_trials_extra
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return aname


# ─── Run K independent trials ────────────────────────────────────────────────
if __name__ == "__main__":
    if SEED_LIST is not None:
        seeds = list(SEED_LIST)
    else:
        rng = random.Random(0)  # reproducible draw of distinct seeds
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
