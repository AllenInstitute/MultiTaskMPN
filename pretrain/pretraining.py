#!/usr/bin/env python
# coding: utf-8
"""
Pretraining -> Post-training transfer experiment.

run_trial runs one seed; main uses the configuration below to run a batch.
run_many remains a callable batch entry point. Fixed SEED_LIST values override
N_TRIALS; otherwise each invocation selects distinct random seeds.

Protocol
--------
Stage 1 (Pretraining)
    Train a DeepMultiPlasticNet on the tasks of the chosen pretraining ruleset
    (a pair for the Proper/Improper motifs, or one of the single-task
    DelayAnti/DelayPro controls) while reserving one extra task-indicator
    column for the held-out post-training task.

Stage 2 (Post-training)
    Reuse the pretrained network, freeze all parameters via
    expand_and_freeze(option=1), and continue training only the last input
    column on the held-out task.

Outputs in ./pretraining/ include configuration, stage-specific test inputs,
targets and task metadata, recorded M and hidden states, accuracy curves,
the final checkpoint, and both stages' full histories. Unused predictions,
bias/input traces, and duplicate result metadata are not saved.
The feature argument labels files; it does not set the regularization strength.
"""

import copy
import gc
import json
import pickle
import random
import traceback
from importlib import reload
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
from run_logging import tee_output
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

ACCEPT_RULES = (
    'fdgo', 'fdanti', 'delaygo', 'delayanti', 'reactgo', 'reactanti',
    'delaydm1', 'delaydm2', 'dmsgo', 'dmcgo', 'contextdelaydm1',
    'contextdelaydm2', 'multidelaydm', 'dmsnogo', 'dmcnogo'
)
# Pretraining rulesets (keys are underscore-joined internal rule names and
# prefix every output filename, so each mode saves to its own files):
#   fdgo_delaygo   — Improper motif   (DelayPro + MemoryPro)
#   fdanti_delaygo — Proper motif     (DelayAnti + MemoryPro)
#   fdanti         — DelayAnti       (DelayAnti only; tests whether MemoryPro
#                    pretraining is necessary for MemoryAnti transfer)
#   fdgo           — Improper motif + (DelayPro only; single-task control
#                    matched to the DelayAnti-only condition)
#   delayanti      — the post-training task itself (MemoryAnti)
# A single-rule pretraining reserves one held-out column as usual, so its
# checkpoints have one fewer task-indicator channel than the two-rule motifs.
RULES_DICT = {
    'fdgo_delaygo': ['fdgo', 'delaygo'],
    'fdanti_delaygo': ['fdanti', 'delaygo'],
    'fdanti': ['fdanti'],
    'fdgo': ['fdgo'],
    'delayanti': ['delayanti'],
}
RULES_DICT_FREQUENCY = {
    'fdgo_delaygo': np.array([1, 1]),
    'fdanti_delaygo': np.array([1, 1]),
    'fdanti': np.array([1]),
    'fdgo': np.array([1]),
    'delayanti': np.array([1]),
}
OUT_DIR = Path("./pretraining")

N_TRIALS = 10
SEED_LIST = None
PRETRAIN_RULESET = "fdgo"
POSTTRAIN_RULESET = "delayanti"
FEATURE = "L21e3"

# Multiplicative modulation bounds (min, max) for M. The default (-1, 1)
# keeps W_eff = W * (1 + M) within [0, 2W], so no synapse can flip the sign
# of its weight; (-2, 2) gives 1 + M in [-1, 3] and allows per-synapse sign
# inversion. Non-default bounds are appended to the feature label
# automatically (e.g. L21e3 -> L21e3mb2), so their output files can never
# overwrite or be confused with default-bound runs, and downstream analyses
# select them explicitly via their feature string.
M_BOUNDS = (-1.0, 1.0)


def _feature_with_bounds(feature, m_bounds=None):
    """Append a bounds tag to the feature label for non-default M bounds."""
    m_bounds = tuple(M_BOUNDS if m_bounds is None else m_bounds)
    if m_bounds == (-1.0, 1.0):
        return feature
    if m_bounds[0] == -m_bounds[1]:
        return f"{feature}mb{m_bounds[1]:g}"
    return f"{feature}mb{m_bounds[0]:g}to{m_bounds[1]:g}"

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
            'm_bounds': M_BOUNDS,
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


def _extract_rule_epochs(task_params, test_trials):
    rules_epochs = {}
    for rule_idx, rule in enumerate(task_params['rules']):
        print(rule)
        if rule not in ACCEPT_RULES:
            raise NotImplementedError()
        rules_epochs[rule] = test_trials[rule_idx].epochs

    print(rules_epochs)
    return rules_epochs


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


def _modulation_extraction(db_, max_seq_len_, layer_index, n_batch, *, nettype="dmpn"):
    """Extract modulation matrices and reshape hidden states for one stage."""
    print(db_.keys())
    if nettype == "dmpn":
        Ms_orig = np.concatenate((db_[f'M{layer_index}'],), axis=-1)
        hidden = db_[f'hidden{layer_index}']
        hs = np.concatenate((
            hidden.reshape(n_batch, max_seq_len_, hidden.shape[-1]),
        ), axis=-1)
        return Ms_orig, hs

    if nettype == "vanilla":
        hidden = db_['hidden']
        hs = np.concatenate((
            hidden.reshape(n_batch, max_seq_len_, hidden.shape[-1]),
        ), axis=-1)
        return None, hs

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


def run_trial(seed=None, feature="L21e3", pretrain_ruleset="fdanti_delaygo", posttrain_ruleset="delayanti"):
    train = True
    verbose = True

    if seed is None:
        seed = random.randint(1, 1000)
    _set_seed(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mpn_depth = 1
    n_hidden = 200
    chosen_network = 'dmpn'
    feature = _feature_with_bounds(feature)
    print(f"Feature label (with M-bound tag if non-default): {feature}; "
          f"m_bounds={M_BOUNDS}")
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

    train_params2["n_datasets"] = 80000
    train_params2['n_epochs_per_set'] = 1

    params = (task_params, train_params, net_params)
    params2 = (task_params2, train_params2, net_params2)
    netFunction = _select_net_function(net_params['net_type'])

    test_n_batch = train_params["valid_n_batch"]

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

    rules_epochs = _extract_rule_epochs(task_params, test_trials)
    rules_epochs2 = _extract_rule_epochs(task_params2, test_trials2)

    test_input, test_output, _ = test_data
    test_input2, test_output2, _ = test_data2

    permutation = np.random.permutation(test_input.shape[0])
    permutation2 = np.random.permutation(test_input2.shape[0])
    test_input = test_input[permutation]
    test_output = test_output[permutation]
    test_input2 = test_input2[permutation2]
    test_output2 = test_output2[permutation2]

    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()
    test_input2_np = test_input2.detach().cpu().numpy()
    test_output2_np = test_output2.detach().cpu().numpy()

    n_batch_stage1 = test_input_np.shape[0]
    n_batch_stage2 = test_input2_np.shape[0]
    test_task = _find_task(task_params, test_input_np, shift_index)
    test_task2 = _find_task(task_params2, test_input2_np, shift_index)
    test_task2 = [idx - len(task_params["rules"]) for idx in test_task2]

    print("================================= Stage 1 =================================")
    net_pretrain, _, (_, _, db_stage1_lst, _, _, _, _, _, _, _), pretrain_stop = net_helpers.train_network(
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
    stage1_end_iter = int(net_pretrain.hist["iter"])
    net_stage1 = copy.deepcopy(net_pretrain)

    if hyp_dict_old["chosen_network"] == "dmpn":
        input_orig = net_pretrain.W_initial_linear.weight.detach().cpu().clone()
    elif hyp_dict_old["chosen_network"] == "vanilla":
        input_orig = net_pretrain.W_input.detach().cpu().clone()
    else:
        input_orig = None

    print("================================= Stage 2 =================================")
    net, _, (_, _, db_lst, _, _, _, Wall_lst, marker_lst, _, _), _ = net_helpers.train_network(
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

    if net_params["ml_params"]["W_freeze"]:
        assert np.allclose(Wall_lst[-1][0], Wall_lst[0][0])
    if net_params["input_layer_bias"]:
        assert net_params["input_layer_add"] is True

    print('Done!')

    ind = len(marker_lst) - 1
    network_at_percent = (marker_lst[ind] + 1) / train_params2['n_datasets'] * 100
    print(f"Using network at {network_at_percent}%")

    stage1_output_path = OUT_DIR / f"output_{_build_file_tag(hyp_dict_old, hyp_dict, seed)}_stage1.npz"
    np.savez_compressed(
        stage1_output_path,
        test_input_np=test_input_np,
        test_output_np=test_output_np,
        rules_epochs=rules_epochs,
        task_params=task_params,
        test_task=test_task,
    )

    print(f"test_input_np: {test_input_np.shape}")
    print(f"test_output_np: {test_output_np.shape}")

    stage2_output_path = OUT_DIR / f"output_{_build_file_tag(hyp_dict_old, hyp_dict, seed)}_stage2.npz"
    np.savez_compressed(
        stage2_output_path,
        test_input_np=test_input2_np,
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

    Ms_orig_stage1, hs_stage1 = _modulation_extraction(
        db_stage1_lst[0][-1], max_seq_len1, layer_index, n_batch_stage1,
        nettype=hyp_dict["chosen_network"],
    )
    Ms_orig_stage2, hs_stage2 = _modulation_extraction(
        db_lst[0][-1], max_seq_len2, layer_index, n_batch_stage2,
        nettype=hyp_dict["chosen_network"],
    )

    print(f"hs_stage1.shape:{hs_stage1.shape}")
    print(f"hs_stage2.shape:{hs_stage2.shape}")
    assert hs_stage1.shape[-1] == hs_stage2.shape[-1]

    result_path = OUT_DIR / f"param_{_build_file_tag(hyp_dict_old, hyp_dict, seed)}_result.npz"
    np.savez_compressed(
        result_path,
        Ms_orig_stage1=Ms_orig_stage1,
        hs_stage1=hs_stage1,
        Ms_orig_stage2=Ms_orig_stage2,
        hs_stage2=hs_stage2,
        pretrain_stop=pretrain_stop,
        stage1_end_iter=stage1_end_iter,
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
            "stage1_end_iter": stage1_end_iter,
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


def run_many(n_runs=5, feature="L21e3", pretrain_ruleset="fdanti_delaygo", posttrain_ruleset="delayanti", *, seeds=None):
    """Run independent seeds, continue after failures, and list successful runs."""
    if seeds is None:
        seeds = random.Random().sample(range(1, 1001), n_runs)
    else:
        seeds = list(seeds)
        if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("seeds must contain nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be distinct to avoid overwriting runs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(seeds)} independent trials: seeds={seeds}")
    results = []
    for seed in seeds:
        try:
            results.append(run_trial(
                seed=seed,
                feature=feature,
                pretrain_ruleset=pretrain_ruleset,
                posttrain_ruleset=posttrain_ruleset,
            ))
        except Exception as exc:
            print(f"Trial seed={seed} FAILED: {exc}")
            traceback.print_exc()
        finally:
            plt.close("all")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    anames = [result["net_path"].stem.removeprefix("savednet_") for result in results]
    manifest_path = OUT_DIR / "last_run_anames.txt"
    with manifest_path.open("w") as manifest:
        manifest.write("\n".join(anames) + ("\n" if anames else ""))
    print(f"Completed {len(results)}/{len(seeds)} trials.")
    for aname in anames:
        print(f"  {aname}")
    print(f"Wrote manifest: {manifest_path}")
    return results


def main():
    """Run the configured seed pool without changing the two-stage protocol."""
    return run_many(
        n_runs=N_TRIALS,
        feature=FEATURE,
        pretrain_ruleset=PRETRAIN_RULESET,
        posttrain_ruleset=POSTTRAIN_RULESET,
        seeds=SEED_LIST,
    )


if __name__ == "__main__":
    with tee_output("pretraining"):
        main()
