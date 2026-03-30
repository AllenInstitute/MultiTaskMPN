from pathlib import Path
import re
import gc 
import json 
import matplotlib.pyplot as plt

import torch 

import mpn 
import mpn_tasks

c_vals = [
    "#e53e3e",  # red
    "#3182ce",  # blue
    "#38a169",  # green
    "#d69e2e",  # yellow-gold
    "#d53f8c",  # pink-magenta
    "#4c51bf",  # indigo
    "#dd6b20",  # orange
    "#0ea5e9",  # sky blue
    "#22c55e",  # bright green
    "#a855f7",  # purple
    "#f43f5e",  # red-pink
    "#0f766e",  # teal
    "#b83280",  # magenta-violet
    "#ca8a04",  # amber
    "#2b6cb0",  # deep blue
] * 10

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

    # hidden size: "hidden300"
    m_h = re.search(r'hidden(\d+)', s)
    if not m_h:
        raise ValueError(f"Couldn't find hidden size in: {s}")
    hidden_size = int(m_h.group(1))

    m_l2_custom = re.search(r'L2(\d+(?:\.\d+)?)e(\d+)', s)     
    m_l2_standard = re.search(r'L2([0-9]+(?:\.[0-9]+)?(?:e-?\d+)?)', s)  

    if m_l2_custom:
        base = float(m_l2_custom.group(1))
        exp = int(m_l2_custom.group(2))
        l2 = base * (10.0 ** (-exp))   # your rule: e3 -> 10^{-3}
    elif m_l2_standard:
        l2 = float(m_l2_standard.group(1))  # standard float parsing
    else:
        raise ValueError(f"Couldn't find L2 in: {s}")

    return hidden_size, l2     

if __name__ == "__main__":
    pt_paths = list_pt_files("./multiple_tasks", recursive=False)
        
    def eval_one(netpathname, c_vals):
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

        checkpoint = torch.load(netpathname, map_location="cpu")
        model = mpn.DeepMultiPlasticNet(checkpoint["net_params"], verbose=False, forzihan=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()

        with torch.no_grad():
            net_out, _, db_test = model.iterate_sequence_batch(test_input, run_mode='track_states')
            acc, _ = model.compute_acc(net_out, test_output, test_mask, test_input, isvalid=True, mode="angle")

        fig, axs = plt.subplots(5,2,figsize=(10,10))
        for i in range(5):
            for inp in range(test_input.shape[2]):
                axs[i,0].plot(test_input[i,:,inp].detach().cpu().numpy(), color=c_vals[inp], alpha=0.5)
            for inp in range(net_out.shape[2]):
                axs[i,1].plot(net_out[i,:,inp].detach().cpu().numpy(), color=c_vals[inp], alpha=0.5)
            for outp in range(test_output.shape[2]):
                axs[i,1].plot(test_output[i,:,outp].detach().cpu().numpy(), color=c_vals[outp], alpha=0.5, linestyle="--")

        fig.tight_layout()
        fig.savefig(f"./multiple_tasks_perf/{core_name}_out.png", dpi=300)
        plt.close(fig)

        # hard cleanup before returning
        del test_data, test_trials_extra, test_input, test_output, test_mask
        del checkpoint, model, net_out, db_test
        del raw_cfg_param, task_params, train_params, net_params, task_params_c, train_params_c, net_params_c
        gc.collect()

        return core_name, hidden_size, l2_info, float(acc)

    result_dict = {}
    for netpathname in pt_paths:
        core_name, hidden_size, l2_info, acc = eval_one(netpathname, c_vals)
        result_dict[core_name] = {"hidden_size": hidden_size, "l2_info": l2_info, "acc": acc}
        
    l2_info_uniq = set(d["l2_info"] for d in result_dict.values())
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    for l2_info in l2_info_uniq:
        y = [d["acc"] for d in result_dict.values() if d["l2_info"] == l2_info]
        for y_ in y:
            ax.scatter(l2_info, y_, color=c_vals[0], alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("L2 Regularization Strength", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    fig.tight_layout()
    fig.savefig("./multiple_tasks_perf/l2_vs_acc.png", dpi=300)
    plt.close(fig)
        
    
