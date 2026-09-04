#!/usr/bin/env python
# coding: utf-8
"""
Pretraining -> Post-training transfer experiment.

This module keeps the existing pretraining setup and file outputs, but is
organized like multiple_task_analysis.py: one main orchestrator per seed,
small helpers for each stage, and a narrow script entry point.

Protocol
--------
Stage 1 (Pretraining)
    Train a DeepMultiPlasticNet on a pair of tasks while reserving one extra
    task-indicator column for the held-out post-training task.

Stage 2 (Post-training)
    Reload the pretrained network state in-memory, freeze all parameters via
    expand_and_freeze(option=1), and continue training only the last input
    column on the held-out task.

Outputs are written to ./pretraining/ with the same naming convention as the
previous script.
"""

import copy
import gc
import json
import pickle
import random
from importlib import reload
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
from hdf5plugin import Blosc
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
from scipy.linalg import subspace_angles
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper
import mpn
import mpn_tasks
import net_helpers
import networks as nets

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

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e'] * 10
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089'] * 10
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16'] * 10

ACCEPT_RULES = (
    'fdgo', 'fdanti', 'delaygo', 'delayanti', 'reactgo', 'reactanti',
    'delaydm1', 'delaydm2', 'dmsgo', 'dmcgo', 'contextdelaydm1',
    'contextdelaydm2', 'multidelaydm', 'dmsnogo', 'dmcnogo'
)
RULES_DICT = {
    'fdgo_delaygo': ['fdgo', 'delaygo'],
    'fdanti_delaygo': ['fdanti', 'delaygo'],
    'delayanti': ['delayanti'],
}
RULES_DICT_FREQUENCY = {
    'fdgo_delaygo': np.array([1, 1]),
    'fdanti_delaygo': np.array([1, 1]),
    'delayanti': np.array([1]),
}
OUT_DIR = Path("./pretraining")

reload(nets)
reload(net_helpers)


def _set_seed(seed):
    print(f"Set seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_file_tag(hyp_dict_old, hyp_dict, seed):
    return f"{hyp_dict_old['ruleset']}_{hyp_dict['chosen_network']}_seed{seed}_{hyp_dict['addon_name']}"


def _select_net_function(net_type):
    if net_type == 'mpn1':
        return mpn.MultiPlasticNet
    if net_type == 'dmpn':
        return mpn.DeepMultiPlasticNet
    if net_type == 'vanilla':
        return nets.VanillaRNN
    if net_type == 'gru':
        return nets.GRU
    raise ValueError(f"Unknown net_type: {net_type}")


def _current_basic_params(hyp_dict_input, *, train, n_hidden, mpn_depth):
    task_params = {
        'task_type': hyp_dict_input['task_type'],
        'rules': RULES_DICT[hyp_dict_input['ruleset']],
        'rules_probs': RULES_DICT_FREQUENCY[hyp_dict_input['ruleset']],
        'dt': 40,
        'ruleset': hyp_dict_input['ruleset'],
        'n_eachring': 8,
        'in_out_mode': 'low_dim',
        'sigma_x': 0.00,
        'mask_type': 'cost',
        'fixate_off': True,
        'task_info': True,
        'randomize_inputs': False,
        'n_input': 20,
        'modality_diff': True,
        'label_strength': True,
        'long_stimulus': 'normal',
        'long_fixation': 'normal',
        'long_delay': 'normal',
        'long_response': 'normal',
        'adjust_task_prop': True,
        'adjust_task_decay': 0.9,
    }

    assert task_params["fixate_off"], "Accuracy calculation is partially depended on that"
    print(f"Fixation_off: {task_params['fixate_off']}; Task_info: {task_params['task_info']}")

    train_params = {
        'lr': 1e-3,
        'n_batches': 128,
        'batch_size': 128,
        'gradient_clip': 10,
        'valid_n_batch': 200,
        'n_datasets': 60000,
        'valid_check': 600,
        'pretrain_min': 1000,
        'n_epochs_per_set': 1,
        'weight_reg': 'L2',
        'activity_reg': 'L2',
        'reg_lambda': 1e-3,
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': 0.9,
            'patience': 20,
            'min_lr': 1e-8,
            'step_size': 30,
            'gamma': 0.1,
        },
    }

    print(f"valid_n_batch: {train_params['valid_n_batch']}")
    if not train:
        assert train_params['n_epochs_per_set'] == 0

    net_params = {
        'net_type': hyp_dict_input['chosen_network'],
        'n_neurons': [1] + [n_hidden] * mpn_depth + [1],
        'linear_embed': n_hidden,
        'output_bias': False,
        'hidden_bias': False,
        'input_bias': False,
        'loss_type': 'MSE',
        'activation': 'tanh',
        'W_rec_init': 'diag',
        'W_rec_diag_scale': 0.8,
        'cuda': True,
        'monitor_freq': 1,
        'monitor_valid_out': True,
        'output_matrix': '',
        'input_layer_add': True,
        'input_layer_add_trainable': True,
        'input_layer_bias': False,
        'input_layer': 'trainable',
        'acc_measure': 'angle',
        'ml_params': {
            'bias': True,
            'mp_type': 'mult',
            'm_update_type': 'hebb_assoc',
            'eta_type': 'scalar',
            'eta_train': True,
            'lam_type': 'scalar',
            'm_time_scale': 4000,
            'lam_train': False,
            'W_freeze': False,
        },
        'leaky': True,
        'alpha': 0.2,
    }

    assert net_params["input_bias"] == net_params["input_layer_bias"]
    assert not (task_params["randomize_inputs"] and net_params["input_layer_add"]), (
        "task_params['randomize_inputs'] and net_params['input_layer_add'] cannot both be True."
    )

    if mpn_depth > 1:
        for mpl_idx in range(mpn_depth - 1):
            assert f'ml_params{mpl_idx}' in net_params.keys()

    if hyp_dict_input['chosen_network'] in ("gru", "vanilla"):
        assert 'ml_params' in net_params.keys()

    return task_params, train_params, net_params


def _generate_response_stimulus(task_params, test_trials, hyp_dict_input):
    labels_resp, labels_stim = [], []
    rules_epochs = {}
    for rule_idx, rule in enumerate(task_params['rules']):
        print(rule)
        if rule not in ACCEPT_RULES:
            raise NotImplementedError()
        rules_epochs[rule] = test_trials[rule_idx].epochs
        if hyp_dict_input['ruleset'] in ('dmsgo', 'dmcgo'):
            labels_resp.append(test_trials[rule_idx].meta['matches'])
            labels_stim.append(test_trials[rule_idx].meta['stim1'])
        else:
            try:
                labels_resp.append(test_trials[rule_idx].meta['resp1'])
            except Exception:
                labels_resp.append(test_trials[rule_idx].meta['matches'])
            labels_stim.append(test_trials[rule_idx].meta['stim1'])

    print(rules_epochs)
    labels_resp = np.concatenate(labels_resp, axis=0).reshape(-1, 1)
    labels_stim = np.concatenate(labels_stim, axis=0).reshape(-1, 1)
    return labels_resp, labels_stim, rules_epochs


def _find_task(task_params, test_input_np, shift_index):
    test_task = []
    for batch_idx in range(test_input_np.shape[0]):
        if task_params["randomize_inputs"]:
            test_input_np_ = test_input_np @ np.linalg.pinv(task_params["randomize_matrix"])
        else:
            test_input_np_ = test_input_np

        task_label = np.asarray(test_input_np_[batch_idx, 0, 6 - shift_index:])
        dist = np.abs(task_label - 1)
        mask = dist == dist.min()
        indices = np.where(mask)[0]
        if not indices.size:
            raise ValueError("No entry close enough to 1 found")
        test_task.append(indices[0])
    return test_task


def _modulation_extraction(db_, max_seq_len_, layer_index, n_batch_all, *, half=False, nettype="dmpn"):
    print(db_.keys())
    divider = 1 if not half else 2
    if nettype == "dmpn":
        Ms = np.concatenate((
            db_[f'M{layer_index}'].reshape(int(n_batch_all / divider), max_seq_len_, -1),
        ), axis=-1)
        Ms_orig = np.concatenate((db_[f'M{layer_index}'],), axis=-1)
        bs = np.concatenate((db_[f'b{layer_index}'],), axis=-1)
        hs = np.concatenate((
            db_[f'hidden{layer_index}'].reshape(int(n_batch_all / divider), max_seq_len_, -1),
        ), axis=-1)
        xs = np.concatenate((
            db_[f'input{layer_index}'].reshape(int(n_batch_all / divider), max_seq_len_, -1),
        ), axis=-1)
        return Ms, Ms_orig, hs, bs, xs

    if nettype == "vanilla":
        hs = np.concatenate((
            db_['hidden'].reshape(int(n_batch_all / divider), max_seq_len_, -1),
        ), axis=-1)
        return None, None, hs, None, None

    raise ValueError(f"Unsupported nettype: {nettype}")


def _hist_to_serializable(hist):
    out = {}
    for key, value in hist.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            out[key] = [item.detach().cpu().numpy() for item in value]
        else:
            out[key] = value
    return out


def _build_experiment_hyp_dicts(feature, pretrain_ruleset, posttrain_ruleset, *, n_hidden, chosen_network):
    if pretrain_ruleset not in RULES_DICT:
        raise ValueError(f"Unknown pretrain_ruleset: {pretrain_ruleset}")
    if posttrain_ruleset not in RULES_DICT:
        raise ValueError(f"Unknown posttrain_ruleset: {posttrain_ruleset}")

    base_hyp_dict = {
        'task_type': 'multitask',
        'mode_for_all': 'random_batch',
        'run_mode': 'minimal',
        'chosen_network': chosen_network,
        'addon_name': f"+hidden{n_hidden}+{feature}",
    }
    hyp_dict_old = copy.deepcopy(base_hyp_dict)
    hyp_dict_old['ruleset'] = pretrain_ruleset

    hyp_dict = copy.deepcopy(base_hyp_dict)
    hyp_dict['ruleset'] = posttrain_ruleset
    return hyp_dict_old, hyp_dict


def main(seed=None, feature="L21e3", pretrain_ruleset="fdanti_delaygo", posttrain_ruleset="delayanti"):
    train = True
    verbose = True

    if seed is None:
        seed = random.randint(1, 1000)
    _set_seed(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mpn_depth = 1
    n_hidden = 200
    chosen_network = 'dmpn'
    hyp_dict_old, hyp_dict = _build_experiment_hyp_dicts(
        feature,
        pretrain_ruleset,
        posttrain_ruleset,
        n_hidden=n_hidden,
        chosen_network=chosen_network,
    )
    if hyp_dict['chosen_network'] in ("gru", "vanilla"):
        mpn_depth = 1

    task_params, train_params, net_params = _current_basic_params(
        hyp_dict_old, train=train, n_hidden=n_hidden, mpn_depth=mpn_depth
    )
    print(f"Accuracy Measure: {net_params['acc_measure']}")

    hyp_dict['addon_name'] += f"+batch{train_params['n_batches']}+{net_params['acc_measure']}"
    hyp_dict_old['addon_name'] += f"+batch{train_params['n_batches']}+{net_params['acc_measure']}"
    task_params2, train_params2, net_params2 = _current_basic_params(
        hyp_dict, train=train, n_hidden=n_hidden, mpn_depth=mpn_depth
    )

    config = {
        "task_params": task_params,
        "train_params": train_params,
        "net_params": net_params,
    }
    config_path = OUT_DIR / f"param_{hyp_dict_old['ruleset']}_seed{seed}_{hyp_dict['addon_name']}_param.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=4, default=helper.as_jsonable)

    shift_index = 1 if not task_params['fixate_off'] else 0
    if hyp_dict['task_type'] not in ('multitask',):
        raise NotImplementedError()

    task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )
    task_params2, train_params2, net_params2 = mpn_tasks.convert_and_init_multitask_params(
        (task_params2, train_params2, net_params2)
    )
    net_params['prefs'] = mpn_tasks.get_prefs(task_params['hp'])

    print(f"Rules: {task_params['rules']}")
    print(f"  Input size {task_params['n_input']}, Output size {task_params['n_output']}")

    if net_params['cuda']:
        print('Using CUDA...')
        device = torch.device('cuda')
    else:
        print('Using CPU...')
        device = torch.device('cpu')

    epoch_multiply = train_params["n_epochs_per_set"]
    train_params2["n_datasets"] = 80000
    train_params2['n_epochs_per_set'] = 1

    params = (task_params, train_params, net_params)
    params2 = (task_params2, train_params2, net_params2)
    netFunction = _select_net_function(net_params['net_type'])

    test_n_batch = train_params["valid_n_batch"]
    color_by = "stim"

    pretraining_shift = len(task_params['rules'])
    pretraining_shift_pre = len(task_params2['rules'])
    assert pretraining_shift_pre == 1

    task_params['hp']['batch_size_train'] = test_n_batch
    task_params2['hp']['batch_size_train'] = test_n_batch
    test_mode_for_all = "random"

    task_params_test = copy.deepcopy(task_params)
    task_params_test["long_response"] = "normal"
    test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
        task_params_test,
        test_n_batch,
        rules=task_params_test['rules'],
        mode_input=test_mode_for_all,
        pretraining_shift_pre=pretraining_shift_pre,
    )

    task_params2_test = copy.deepcopy(task_params2)
    task_params2_test["long_response"] = "normal"
    test_data2, test_trials_extra2 = mpn_tasks.generate_trials_wrap(
        task_params2_test,
        test_n_batch,
        rules=task_params2_test['rules'],
        mode_input=test_mode_for_all,
        pretraining_shift=pretraining_shift,
    )
    _, test_trials, _ = test_trials_extra
    _, test_trials2, _ = test_trials_extra2

    task_params['dataset_name'] = 'multitask'
    task_params2['dataset_name'] = 'multitask'

    labels_resp, labels_stim, rules_epochs = _generate_response_stimulus(
        task_params, test_trials, hyp_dict_old
    )
    labels_resp2, labels_stim2, rules_epochs2 = _generate_response_stimulus(
        task_params2, test_trials2, hyp_dict
    )

    labels = labels_stim if color_by == "stim" else labels_resp
    labels2 = labels_stim2 if color_by == "stim" else labels_resp2

    test_input, test_output, _ = test_data
    test_input2, test_output2, _ = test_data2

    permutation = np.random.permutation(test_input.shape[0])
    permutation2 = np.random.permutation(test_input2.shape[0])
    test_input = test_input[permutation]
    test_output = test_output[permutation]
    labels = labels[permutation]
    test_input2 = test_input2[permutation2]
    test_output2 = test_output2[permutation2]
    labels2 = labels2[permutation2]
    del labels, labels2

    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()
    test_input2_np = test_input2.detach().cpu().numpy()
    test_output2_np = test_output2.detach().cpu().numpy()

    n_batch_all = test_input_np.shape[0]
    test_task = _find_task(task_params, test_input_np, shift_index)
    test_task2 = _find_task(task_params2, test_input2_np, shift_index)
    test_task2 = [idx - len(task_params["rules"]) for idx in test_task2]

    print("================================= Stage 1 =================================")
    net_pretrain, _, (_, netout_stage1_lst, db_stage1_lst, _, _, _, _, marker_stage1_lst, _, _), pretrain_stop = net_helpers.train_network(
        params,
        device=device,
        verbose=verbose,
        train=train,
        hyp_dict=hyp_dict_old,
        netFunction=netFunction,
        test_input=[test_input],
        pretraining_shift_pre=1,
        print_frequency=100,
    )

    params2[1]["valid_check"] = None
    net_stage1 = copy.deepcopy(net_pretrain)

    if hyp_dict_old["chosen_network"] == "dmpn":
        input_orig = net_pretrain.W_initial_linear.weight.detach().cpu().clone()
    elif hyp_dict_old["chosen_network"] == "vanilla":
        input_orig = net_pretrain.W_input.detach().cpu().clone()
    else:
        input_orig = None

    print("================================= Stage 2 =================================")
    net, _, (counter_lst, netout_lst, db_lst, _, _, Woutput_lst, Wall_lst, marker_lst, _, _), _ = net_helpers.train_network(
        params2,
        net=net_pretrain,
        device=device,
        verbose=verbose,
        train=train,
        hyp_dict=hyp_dict,
        netFunction=netFunction,
        test_input=[test_input2],
        pretraining_shift=len(task_params["rules"]),
        print_frequency=100,
    )
    print("================================= End  =================================")

    if hyp_dict_old["chosen_network"] == "dmpn":
        input_after = net.W_initial_linear.weight.detach().cpu().clone()
    elif hyp_dict_old["chosen_network"] == "vanilla":
        input_after = net.W_input.detach().cpu().clone()
    else:
        input_after = None

    if input_orig is not None and input_after is not None:
        diff = (input_orig[:, :-1] - input_after[:, :-1]).abs()
        assert torch.all(diff < 1e-4)

    if hyp_dict['chosen_network'] == "dmpn" and net_params["input_layer_add"]:
        counter_lst = [x * epoch_multiply + 1 for x in counter_lst]
        del counter_lst

    if net_params["ml_params"]["W_freeze"]:
        assert np.allclose(Wall_lst[-1][0], Wall_lst[0][0])
    if net_params["input_layer_bias"]:
        assert net_params["input_layer_add"] is True

    print('Done!')

    use_finalstage = False
    if use_finalstage:
        net_out_final, db = net.iterate_sequence_batch(test_input, run_mode='track_states')
    else:
        ind = len(marker_lst) - 1
        ind_stage1 = len(marker_stage1_lst) - 1
        network_at_percent = (marker_lst[ind] + 1) / train_params2['n_datasets'] * 100
        print(f"Using network at {network_at_percent}%")
        net_out_final = netout_lst[0][ind]
        net_out_stage1_final = netout_stage1_lst[0][ind_stage1]
        db = db_lst[0][ind]

    stage1_output_path = OUT_DIR / f"output_{_build_file_tag(hyp_dict_old, hyp_dict, seed)}_stage1.npz"
    np.savez_compressed(
        stage1_output_path,
        test_input_np=test_input_np,
        net_out_stage1_final=net_out_stage1_final,
        test_output_np=test_output_np,
        rules_epochs=rules_epochs,
        task_params=task_params,
        test_task=test_task,
    )

    print(f"test_input_np: {test_input_np.shape}")
    print(f"net_out_stage1_final: {net_out_stage1_final.shape}")
    print(f"test_output_np: {test_output_np.shape}")

    stage2_output_path = OUT_DIR / f"output_{_build_file_tag(hyp_dict_old, hyp_dict, seed)}_stage2.npz"
    np.savez_compressed(
        stage2_output_path,
        test_input_np=test_input2_np,
        net_out_final=net_out_final,
        test_output_np=test_output2_np,
        rules_epochs2=rules_epochs2,
        task_params=task_params2,
        test_task=test_task2,
    )

    layer_index = 1 if net_params["input_layer_add"] else 0
    max_seq_len1 = test_input.shape[1]
    max_seq_len2 = test_input2.shape[1]

    print(f"rules_epochs: {rules_epochs}")
    print(f"rules_epochs2: {rules_epochs2}")

    all_rules = np.array(task_params["rules"])
    test_task = np.array(test_task)
    print(f"all_rules: {all_rules}")
    print(f"test_task: {test_task}")

    Ms_stage1, Ms_orig_stage1, hs_stage1, bs_stage1, xs_stage1 = _modulation_extraction(
        db_stage1_lst[0][-1], max_seq_len1, layer_index, n_batch_all,
        nettype=hyp_dict["chosen_network"],
    )
    Ms_stage2, Ms_orig_stage2, hs_stage2, bs_stage2, xs_stage2 = _modulation_extraction(
        db_lst[0][-1], max_seq_len2, layer_index, n_batch_all,
        half=True, nettype=hyp_dict["chosen_network"],
    )

    print(f"hs_stage1.shape:{hs_stage1.shape}")
    print(f"hs_stage2.shape:{hs_stage2.shape}")
    assert hs_stage1.shape[-1] == hs_stage2.shape[-1]

    result_path = OUT_DIR / f"param_{_build_file_tag(hyp_dict_old, hyp_dict, seed)}_result.npz"
    np.savez_compressed(
        result_path,
        rules_epochs=rules_epochs,
        rules_epochs2=rules_epochs2,
        hyp_dict_old=hyp_dict_old,
        hyp_dict=hyp_dict,
        all_rules=all_rules,
        Ms_orig_stage1=Ms_orig_stage1,
        hs_stage1=hs_stage1,
        bs_stage1=bs_stage1,
        xs_stage1=xs_stage1,
        Ms_orig_stage2=Ms_orig_stage2,
        hs_stage2=hs_stage2,
        bs_stage2=bs_stage2,
        xs_stage2=xs_stage2,
        pretrain_stop=pretrain_stop,
        valid_acc_iter=net.hist['iters_monitor'][1:],
        valid_acc=net.hist['valid_acc'][1:],
    )

    net_path = OUT_DIR / f"savednet_{_build_file_tag(hyp_dict_old, hyp_dict, seed)}.pt"
    torch.save({
        "state_dict": net.state_dict(),
        "net_params": net_params,
    }, net_path)
    print("Network parameter saving is done")

    hist_path = OUT_DIR / f"hist_{_build_file_tag(hyp_dict_old, hyp_dict, seed)}.pkl"
    with hist_path.open("wb") as f:
        pickle.dump({
            "stage1": _hist_to_serializable(net_stage1.hist),
            "stage2": _hist_to_serializable(net.hist),
            "pretrain_stop": pretrain_stop,
        }, f)
    print(f"Training history saved: {hist_path}")

    return {
        "seed": seed,
        "feature": feature,
        "pretrain_ruleset": pretrain_ruleset,
        "posttrain_ruleset": posttrain_ruleset,
        "config_path": config_path,
        "result_path": result_path,
        "net_path": net_path,
        "hist_path": hist_path,
    }


def run_many(n_runs=5, feature="L21e3", pretrain_ruleset="fdanti_delaygo", posttrain_ruleset="delayanti"):
    results = []
    for _ in range(n_runs):
        results.append(main(
            feature=feature,
            pretrain_ruleset=pretrain_ruleset,
            posttrain_ruleset=posttrain_ruleset,
        ))
    return results


if __name__ == "__main__":
    run_many()