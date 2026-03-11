#!/usr/bin/env python3
"""
Single-task delayedDM1 training script.

Architecture used here is strictly:
        x -> h -> y
where the multi-plastic modulation M is only on x -> h.

Implementation details:
- input_layer_add=False: no extra linear embedding before the MP layer
- mp_layer0 maps input x directly to hidden h
- W_output maps hidden h to output y
- training uses supervised backprop/BPTT over time

Examples:
    # Repo-consistent GPU run (if CUDA is available)
    python single_task_delayeddm1.py --preset repo

    # Explicit custom GPU run
    python single_task_delayeddm1.py --hidden 300 --n-batches 128 --batch-size 128 --n-datasets 100000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import helper
import mpn
import mpn_tasks
import net_helpers


def build_params(args):
    task_params = {
        "task_type": "multitask",
        "rules": ["delaydm1"],
        "rules_probs": np.array([1.0]),
        "dt": args.dt,
        "ruleset": "delaydm1",
        "n_eachring": 8,
        "in_out_mode": "low_dim",
        "sigma_x": args.sigma_x,
        "mask_type": "cost",
        "fixate_off": True,
        "task_info": True,
        "randomize_inputs": False,
        "n_input": 20,
        "modality_diff": True,
        "label_strength": True,
        "long_stimulus": "normal",
        "long_fixation": "normal",
        "long_delay": "normal",
        "long_response": "normal",
        "adjust_task_prop": True,
        "adjust_task_decay": 0.9,
    }

    train_params = {
        "train_type": "supervised",
        "gradient_type": "backprop",
        "lr": args.lr,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "gradient_clip": 10,
        "valid_n_batch": args.valid_n_batch,
        "n_datasets": args.n_datasets,
        "valid_check": None,
        "n_epochs_per_set": 1,
        "weight_reg": "L2",
        "activity_reg": "L2",
        "reg_lambda": args.reg_lambda,
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.95,
            "patience": 30,
            "min_lr": 1e-8,
            "step_size": 30,
            "gamma": 0.1,
        },
    }

    net_params = {
        "net_type": "dmpn",
        "n_neurons": [1, args.hidden, 1],
        # Keep architecture as x -> h -> y, with MP modulation on x -> h only.
        "linear_embed": args.hidden,
        "output_bias": False,
        "loss_type": "MSE",
        "activation": "tanh",
        "cuda": args.cuda,
        "monitor_freq": train_params["n_epochs_per_set"],
        "monitor_valid_out": True,
        "output_matrix": "",
        "input_layer_add": False,
        "input_layer_add_trainable": False,
        "input_layer_bias": False,
        "acc_measure": "angle",
        "ml_params": {
            "bias": True,
            "mp_type": "mult",
            "m_update_type": "hebb_assoc",
            "eta_type": "scalar",
            "eta_train": False,
            "lam_type": "scalar",
            "m_time_scale": 4000,
            "lam_train": False,
            "W_freeze": False,
        },
    }

    return task_params, train_params, net_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a single-layer MPN on the single delayedDM1 task."
    )
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--preset",
        type=str,
        choices=["none", "repo"],
        default="none",
        help="Use a predefined setting; 'repo' matches common settings in this repo.",
    )
    parser.add_argument("--hidden", type=int, default=300)
    parser.add_argument("--dt", type=int, default=40)
    parser.add_argument("--sigma-x", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg-lambda", type=float, default=1e-3)
    parser.add_argument("--n-batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--valid-n-batch", type=int, default=20)
    parser.add_argument("--n-datasets", type=int, default=1000)
    parser.add_argument("--print-frequency", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="single_task_delayeddm1")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.preset == "repo":
        # Common setting pattern used in other experiment scripts in this repo.
        args.hidden = 300
        args.dt = 40
        args.lr = 1e-3
        args.reg_lambda = 1e-3
        args.n_batches = 128
        args.batch_size = 128
        args.valid_n_batch = 20
        args.n_datasets = 100000
        args.print_frequency = 100

    if args.quick:
        args.n_datasets = min(args.n_datasets, 20)
        args.n_batches = min(args.n_batches, 32)
        args.batch_size = min(args.batch_size, 64)
        args.valid_n_batch = min(args.valid_n_batch, 8)
        args.print_frequency = min(args.print_frequency, 5)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    hyp_dict = {
        "task_type": "multitask",
        "mode_for_all": "random_batch",
        "run_mode": "minimal",
        "chosen_network": "dmpn",
        "ruleset": "delaydm1",
        "addon_name": f"single_delaydm1_hidden{args.hidden}_seed{args.seed}",
    }

    task_params, train_params, net_params = build_params(args)

    params = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )
    task_params, train_params, net_params = params
    net_params["prefs"] = mpn_tasks.get_prefs(task_params["hp"])

    test_data, _ = mpn_tasks.generate_trials_wrap(
        task_params,
        train_params["valid_n_batch"],
        rules=task_params["rules"],
        mode_input="random",
        fix=True,
        device=device,
    )
    test_input, _, _ = test_data

    net, _, training_bundle, early_stop_idx = net_helpers.train_network(
        params,
        device=device,
        verbose=True,
        train=True,
        hyp_dict=hyp_dict,
        netFunction=mpn.DeepMultiPlasticNet,
        test_input=[test_input],
        print_frequency=args.print_frequency,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `task_params['hp']` contains a numpy RandomState object, which is not JSON serializable.
    task_params_json = dict(task_params)
    if "hp" in task_params_json:
        hp_json = dict(task_params_json["hp"])
        if "rng" in hp_json:
            hp_json["rng"] = "numpy.random.RandomState"
        task_params_json["hp"] = hp_json

    config_path = out_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "device": str(device),
                "hyp_dict": hyp_dict,
                "task_params": task_params_json,
                "train_params": train_params,
                "net_params": net_params,
                "early_stop_idx": early_stop_idx,
            },
            f,
            indent=2,
            default=helper.as_jsonable,
        )

    net_helpers.save_checkpoint(
        str(out_dir / "checkpoint.pt"),
        net,
        params,
        hyp_dict,
        args.seed,
    )

    # Keep a compact copy of monitor curves for quick plotting/debug.
    np.savez(
        out_dir / "history.npz",
        iters=np.array(net.hist.get("iters_monitor", [])),
        train_acc=np.array(net.hist.get("train_acc", [])),
        valid_acc=np.array(net.hist.get("valid_acc", [])),
        train_loss=np.array(net.hist.get("train_loss", [])),
        valid_loss=np.array(net.hist.get("valid_loss", [])),
        marker_lst=np.array(training_bundle[7]),
        loss_lst=np.array(training_bundle[8]),
        acc_lst=np.array(training_bundle[9]),
    )

    print("Training finished.")
    print(f"Saved config: {config_path}")
    print(f"Saved checkpoint: {out_dir / 'checkpoint.pt'}")
    print(f"Saved history: {out_dir / 'history.npz'}")


if __name__ == "__main__":
    main()
