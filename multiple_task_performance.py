"""
Evaluate trained multi-task MPN models and save results.

Loads each checkpoint, runs inference, computes accuracy, and saves
the result dictionary to a JSON file for downstream plotting.

Usage:
    python multiple_task_performance.py
"""
from pathlib import Path
import re
import gc
import json

import torch

import mpn
import mpn_tasks


def natural_key(p: Path):
    s = p.as_posix()
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_pt_files(dir_path: str, recursive: bool = True):
    root = Path(dir_path)
    if recursive:
        files = [p for p in root.rglob("*.pt") if p.is_file()]
    else:
        files = [p for p in root.glob("*.pt") if p.is_file()]

    files.sort(key=natural_key)
    return [str(p) for p in files]


def parse_hidden_and_l2(path_str: str):
    s = Path(path_str).name

    m_h = re.search(r'hidden(\d+)', s)
    if not m_h:
        raise ValueError(f"Couldn't find hidden size in: {s}")
    hidden_size = int(m_h.group(1))

    m_l2_custom = re.search(r'L2(\d+(?:\.\d+)?)e(\d+)', s)
    m_l2_standard = re.search(r'L2([0-9]+(?:\.[0-9]+)?(?:e-?\d+)?)', s)

    if m_l2_custom:
        base = float(m_l2_custom.group(1))
        exp = int(m_l2_custom.group(2))
        l2 = base * (10.0 ** (-exp))
    elif m_l2_standard:
        l2 = float(m_l2_standard.group(1))
    else:
        raise ValueError(f"Couldn't find L2 in: {s}")

    return hidden_size, l2


RESULT_PATH = Path("multiple_tasks_perf") / "performance_results.json"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path("multiple_tasks_perf").mkdir(parents=True, exist_ok=True)

    def eval_one(netpathname):
        hidden_size, l2_info = parse_hidden_and_l2(netpathname)
        core_name = netpathname[24:-3]

        out_param_path = Path("multiple_tasks") / f"param_{core_name}_param.json"
        with out_param_path.open() as f:
            raw_cfg_param = json.load(f)

        task_params, train_params, net_params = raw_cfg_param["task_params"], raw_cfg_param["train_params"], raw_cfg_param["net_params"]
        task_params_c, train_params_c, net_params_c = mpn_tasks.convert_and_init_multitask_params((task_params, train_params, net_params))

        test_n_batch = 50
        task_params_c['hp']['batch_size_train'] = test_n_batch

        test_data, test_trials_extra = mpn_tasks.generate_trials_wrap(
            task_params_c, test_n_batch, rules=task_params_c['rules'],
            mode_input="random_batch", device="cpu", verbose=False
        )
        test_input, test_output, test_mask = test_data

        checkpoint = torch.load(netpathname, map_location=device)
        model = mpn.DeepMultiPlasticNet(checkpoint["net_params"], verbose=False, forzihan=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        model.eval()

        test_input = test_input.to(device)
        test_output = test_output.to(device)
        test_mask = test_mask.to(device)

        with torch.no_grad():
            net_out, _, db_test = model.iterate_sequence_batch(test_input, run_mode='track_states')
            acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input, isvalid=True, mode="angle")

        del test_data, test_trials_extra, test_input, test_output, test_mask
        del checkpoint, model, net_out, db_test
        del raw_cfg_param, task_params, train_params, net_params, task_params_c, train_params_c, net_params_c
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return core_name, hidden_size, l2_info, float(acc)

    pt_paths = list_pt_files("./multiple_tasks", recursive=False)

    result_dict = {}
    for netpathname in pt_paths:
        core_name, hidden_size, l2_info, acc = eval_one(netpathname)
        result_dict[core_name] = {"hidden_size": hidden_size, "l2_info": l2_info, "acc": acc}
        print(f"  {core_name}: acc={acc:.4f}")

    with open(RESULT_PATH, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"Saved results to {RESULT_PATH}")
