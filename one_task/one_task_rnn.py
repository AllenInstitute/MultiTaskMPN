#!/usr/bin/env python
# coding: utf-8
"""
Single-task training for a vanilla RNN — the RNN counterpart of one_task.py.

Trains a `networks.VanillaRNN` on ONE cognitive task (delaygo by default) so its
HIDDEN-state fixed points can be compared against the MPN's modulation fixed
points. The two scripts share the task generation, the trainer and the naming
convention; what differs is the architecture and, consequently, what is worth
saving:

  one_task.py (MPN)  the state is the fast modulation matrix M, so it records the
                     full per-stage M / hidden / bias traces that
                     one_task_analysis.py needs to watch M evolve.
  this script (RNN)  there is no M. The state is the hidden vector, and the only
                     downstream analysis is the hidden-state fixed-point solve,
                     which re-runs the trained network itself. So the saved npz
                     is deliberately LEAN — training curves, one test batch and
                     the trial-period boundaries — and the checkpoint carries the
                     rest.

Outputs (under ./onetask_rnn/, kept separate from the MPN runs in ./onetask/ so
the two never collide or get analyzed by the wrong script):
  savednet_{aname}.pt         — final network state_dict + net_params
  param_{aname}_param.json    — task/train/net hyperparameters
  param_{aname}_result.npz    — training curves, test batch, period boundaries
  loss_acc_{aname}.png        — training accuracy / loss curve

`aname` follows the same convention as one_task.py, with `rnn` in the addon slot:
  {ruleset}_seed{seed}_{addon}{reg_tag}+hidden{hidden}+batch{batch}+{acc_measure}

Usage:
    python one_task/one_task_rnn.py            # N_TRIALS seeds
    python one_task/one_task_rnn.py --seeds 1 2 3
"""
import gc
import random
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import networks as nets
import net_helpers
import mpn_tasks
import helper

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ─── Experiment-wide configuration ───────────────────────────────────────────
N_TRIALS = 1
SEED_LIST = None

RULESET = 'delaygo'
CHOSEN_NETWORK = "vanilla"      # vanilla | gru — NOT dmpn (that is one_task.py)
ADDON_NAME = "rnn"
train = True
verbose = True

accept_rules = ('fdgo', 'fdanti', 'delaygo', 'delayanti', 'reactgo', 'reactanti',
                'delaydm1', 'delaydm2', 'dmsgo', 'dmcgo', 'contextdelaydm1',
                'contextdelaydm2', 'multidelaydm', 'dm1')

rules_dict = {
    'delaygo': ['delaygo'],
    'delayanti': ['delayanti'],
    'delaygofamily': ['delaygo', 'delayanti'],
    'fdgo': ['fdgo'],
    'reactgo': ['reactgo'],
    'delaydm1': ['delaydm1'],
    'dmsgo': ['dmsgo'],
    'dmcgo': ['dmcgo'],
}

OUT_DIR = Path("onetask_rnn")


def current_basic_params(hyp_dict):
    """Task / training / network hyperparameters for one RNN run.

    Kept deliberately close to one_task.py so the MPN and RNN runs differ only
    where the architecture forces it; the differences are flagged inline."""
    task_params = {
        'task_type': hyp_dict['task_type'],
        'rules': rules_dict[hyp_dict['ruleset']],
        'dt': 40,                       # ms per step (SCHEME.md)
        'ruleset': hyp_dict['ruleset'],
        'n_eachring': 64 if hyp_dict.get('more_stimulus') else 8,
        'in_out_mode': 'low_dim',
        'sigma_x': 0.00,
        'mask_type': 'cost',
        'fixate_off': False,
        'task_info': True,
        # Must stay False: VanillaRNN uses its own trainable input layer
        # (`input_layer`), and net_params asserts the two are not both on.
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

    train_params = {
        'lr': 1e-3,
        'n_batches': 128,
        'batch_size': 128,
        'gradient_clip': 10,
        'valid_n_batch': 50,
        'n_datasets': 2000,
        'valid_check': None,
        'n_epochs_per_set': 1,
        'task_mask': None,
        'weight_reg': 'L2',
        'activity_reg': 'L2',
        'reg_lambda': 1e-4,
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': 0.95,
            'patience': 30,
            'min_lr': 1e-8,
            'step_size': 30,
            'gamma': 0.1
        },
    }

    if hyp_dict.get('more_stimulus'):
        train_params['n_datasets'] = max(train_params['n_datasets'], 10000)
        print(f"more_stimulus: n_datasets set to {train_params['n_datasets']}")

    if not train:
        assert train_params['n_epochs_per_set'] == 0

    n_hidden = 200

    net_params = {
        'net_type': hyp_dict['chosen_network'],
        'n_neurons': [1] + [n_hidden] + [1],
        'linear_embed': n_hidden,
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
        'input_layer': "trainable",     # the RNN's own input projection W_input
        'acc_measure': 'angle',
        # `ml_params` is unused by VanillaRNN (it has no plastic layer), but
        # net_helpers asserts the key exists for the gru/vanilla path, so it is
        # kept as an inert placeholder.
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
        # Leaky RNN. NB alpha is the RETENTION factor here
        # (h <- alpha*h + (1-alpha)*act(...)), so tau = dt/(1-alpha) = 200 ms at
        # alpha=0.8 — the same time constant SCHEME.md quotes. one_task.py's
        # alpha=0.2 belongs to the MPN's convention and would mean tau = 50 ms
        # here, far too fast to hold a delay.
        'leaky': True,
        'alpha': 0.8,
    }

    assert not (task_params["randomize_inputs"] and net_params["input_layer_add"]), (
        "task_params['randomize_inputs'] and net_params['input_layer_add'] "
        "cannot both be True.")
    assert net_params['net_type'] in ("vanilla", "gru"), (
        f"this script trains RNNs; got net_type={net_params['net_type']!r}. "
        "Use one_task.py for the MPN.")

    return task_params, train_params, net_params


def run_trial(seed, more_stimulus=False):
    """Train one RNN on the chosen single task and save its checkpoint + the
    lean trace bundle one_task_rnn_analysis.py needs."""
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
        'more_stimulus': more_stimulus,
    }

    task_params, train_params, net_params = current_basic_params(hyp_dict)

    n_hidden = net_params['n_neurons'][1]
    _rl = train_params['reg_lambda']
    _mant, _exp = f"{_rl:.0e}".split("e")
    reg_tag = f"L2{_mant}e{abs(int(_exp))}"
    stim_tag = "+morestimulus" if hyp_dict.get('more_stimulus') else ""
    aname = (f"{hyp_dict['ruleset']}_seed{seed}_{hyp_dict['addon_name']}{reg_tag}+"
             f"hidden{n_hidden}+batch{train_params['n_batches']}+"
             f"{net_params['acc_measure']}{stim_tag}")
    print(f"aname: {aname}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Persist hyperparameters before training so a crash still leaves a record.
    config = {"task_params": task_params, "train_params": train_params,
              "net_params": net_params}
    with (OUT_DIR / f"param_{aname}_param.json").open("w") as f:
        json.dump(config, f, indent=4, default=helper.as_jsonable)

    shift_index = 1 if not task_params['fixate_off'] else 0

    task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params))
    net_params['prefs'] = mpn_tasks.get_prefs(task_params['hp'])
    print('Rules: {}'.format(task_params['rules']))
    print('  Input size {}, Output size {}'.format(
        task_params['n_input'], task_params['n_output']))

    device = (torch.device('cuda') if net_params['cuda'] and torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Using {device}")

    params = task_params, train_params, net_params
    netFunction = nets.GRU if net_params['net_type'] == 'gru' else nets.VanillaRNN

    # ─── Test/validation dataset ─────────────────────────────────────────────
    test_n_batch = train_params["valid_n_batch"]
    color_by = "stim"
    task_params['hp']['batch_size_train'] = test_n_batch
    test_mode_for_all = "random"
    test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
        task_params, test_n_batch, rules=task_params['rules'],
        mode_input=test_mode_for_all, device=device)
    _, test_trials, _ = test_trials_extra
    task_params['dataset_name'] = 'multitask'

    labels = []
    for rule_idx, rule in enumerate(task_params['rules']):
        if rule not in accept_rules:
            raise NotImplementedError(f"rule {rule} not in accept_rules")
        if hyp_dict['ruleset'] in ('dmsgo', 'dmcgo'):
            labels.append(test_trials[rule_idx].meta['matches'])
        else:
            labels.append(test_trials[rule_idx].meta[
                'resp1' if color_by == "resp" else 'stim1'])
    labels = np.concatenate(labels, axis=0).reshape(-1, 1)

    test_input, test_output, test_mask = test_data
    permutation = np.random.permutation(test_input.shape[0])
    test_input, test_output = test_input[permutation], test_output[permutation]
    test_mask, labels = test_mask[permutation], labels[permutation]
    test_input_np = test_input.detach().cpu().numpy()
    test_output_np = test_output.detach().cpu().numpy()
    print(f"test_input.shape: {test_input.shape}")

    # ─── Train ───────────────────────────────────────────────────────────────
    net, _, (counter_lst, netout_lst, db_lst, Winput_lst, Winputbias_lst,
             Woutput_lst, Wall_lst, marker_lst, loss_lst,
             acc_lst), _ = net_helpers.train_network(
        params, device=device, verbose=verbose, train=train, hyp_dict=hyp_dict,
        netFunction=netFunction, test_input=[test_input], print_frequency=100,
        record_frequency=8)

    # NB no per-stage M/hidden trace extraction here, and no
    # net_helpers.net_eta_lambda_analysis: both read the MP layer's M / eta /
    # lambda, which a VanillaRNN does not have. `Wall_lst` likewise comes back as
    # a list of EMPTY lists for non-dmpn nets (net_helpers only fills it for
    # dmpn), so it is deliberately not saved.
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

    net_out_final = np.asarray(netout_lst[0][-1])

    # ─── Trial-period boundaries (same derivation as one_task.py) ────────────
    recordkyle_all = []
    for test_subtrial in test_trials:
        metaepoch = test_subtrial.epochs
        periodname = list(metaepoch.keys())
        recordkyle = []
        for keyiter in range(len(periodname)):
            try:
                recordkyle.append(metaepoch[periodname[keyiter]][1])
            except Exception as e:
                print(e)
        fillrecordkyle = [[ts for _ in range(test_input.shape[0])]
                          for ts in recordkyle]
        recordkyle = fillrecordkyle
        recordkyle.insert(0, [0 for _ in range(len(recordkyle[1]))])
        recordkyle_all.extend(np.array(recordkyle).T.tolist())

    unique_recordkyle_all = [list(t) for t in set(tuple(r) for r in recordkyle_all)]
    all_breaks = []
    for task_specific_time in unique_recordkyle_all:
        all_breaks.append([task_specific_time[i + 1]
                           for i in range(len(task_specific_time) - 2)])
    assert len(all_breaks)
    response_start = all_breaks[0][-1]
    stimulus_start = all_breaks[0][0]
    stimulus_end = all_breaks[0][1]
    print(f"response_start={response_start}, stimulus_start={stimulus_start}, "
          f"stimulus_end={stimulus_end}")

    # ─── Training curve ──────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(counter_lst, loss_lst, "-o", color="#e53e3e")
    ax1.set_ylabel("MSE Loss", color="#e53e3e", fontsize=13)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel("# Dataset", fontsize=13)
    ax2 = ax1.twinx()
    ax2.plot(counter_lst, acc_lst, "-o", color="#3182ce")
    ax2.axhline(1 / 8, linestyle="--", color="0.5", label="By chance")
    ax2.set_ylabel("Accuracy", color="#3182ce", fontsize=13)
    ax2.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"loss_acc_{aname}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {OUT_DIR / f'loss_acc_{aname}.png'}")

    # ─── Save what the analysis needs ────────────────────────────────────────
    result_path = OUT_DIR / f"param_{aname}_result.npz"
    np.savez_compressed(
        result_path,
        hyp_dict=hyp_dict,
        seed=seed,
        shift_index=shift_index,
        color_by=color_by,
        counter_lst=np.asarray(counter_lst),
        marker_lst=np.asarray(marker_lst),
        loss_lst=np.asarray(loss_lst),
        acc_lst=np.asarray(acc_lst),
        net_hist=net_hist,
        test_input_np=test_input_np,
        test_output_np=test_output_np,
        net_out_final=net_out_final,
        labels=labels,
        all_breaks=np.array(all_breaks, dtype=object),
        response_start=response_start,
        stimulus_start=stimulus_start,
        stimulus_end=stimulus_end,
    )
    print(f"Saved traces: {result_path}")

    net_path = OUT_DIR / f"savednet_{aname}.pt"
    torch.save({"state_dict": net.state_dict(), "net_params": net_params}, net_path)
    print(f"Saved network: {net_path}")

    del net, db_lst, netout_lst, Wall_lst, Woutput_lst, Winput_lst, Winputbias_lst
    del test_input, test_output, test_mask, test_data, test_trials_extra
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return aname


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seeds to train (default: SEED_LIST, or "
                             f"{N_TRIALS} random ones).")
    parser.add_argument("--more-stimulus", action="store_true",
                        help="Train on 64 ring directions instead of 8; tags the "
                             "aname with 'morestimulus'.")
    args = parser.parse_args()

    if args.seeds:
        seeds = list(args.seeds)
    elif SEED_LIST is not None:
        seeds = list(SEED_LIST)
    else:
        seeds = random.Random().sample(range(1, 1000), N_TRIALS)

    print(f"Running {len(seeds)} RNN trial(s): seeds={seeds}")
    anames = []
    for seed in seeds:
        try:
            anames.append(run_trial(seed, more_stimulus=args.more_stimulus))
        except Exception as exc:
            print(f"Trial seed={seed} FAILED: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\nCompleted {len(anames)}/{len(seeds)} trials.")
    for a in anames:
        print(f"  {a}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "last_run_anames.txt"
    with manifest_path.open("w") as mf:
        mf.write("\n".join(anames) + ("\n" if anames else ""))
    print(f"Wrote manifest: {manifest_path}")
