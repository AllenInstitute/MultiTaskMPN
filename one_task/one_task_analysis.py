#!/usr/bin/env python
# coding: utf-8
"""
Post-training analysis of a single-task MPN.

Reloads the per-stage training traces saved by one_task.py and reproduces the
single-task analyses:

1. Loss / accuracy across training.
2. Input weight matrix heatmap (W_initial_linear).
2b. Example single-trial input & network/target output.
3. Fixon vs task projection onto the readout — the "cancellation" mechanism
   (Eq. 2-7 sanity check) tracked across training. At the final stage also
   emits the exhaustive-search (es1/es2), fixon-task difference (diff), and
   per-stimulus cancellation (show) figures.
4. Modulation-change / synaptic & hidden correlation across learning.
5. Weight-component projection to output across learning.
6. Low-D PCA of the modulation matrix M during the stimulus period.

All outputs go into ./onetask/{aname}/. Aggregated correlation curves (across
seeds) are written to ./onetask_data/ and re-plotted if multiple runs exist.

Usage:
    python one_task_analysis.py                 # newest run in ./onetask/
    python one_task_analysis.py --aname <name>  # a specific run
"""
import os
import glob
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
ticker.Locator.MAXTICKS = 10000
import seaborn as sns

import _bootstrap  # noqa: F401  -- prepends repo-root/core to sys.path
import helper

# Match the plotting style used in multiple_task_analysis.py
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

c_vals = [
    "#e53e3e", "#3182ce", "#38a169", "#d69e2e", "#d53f8c",
    "#4c51bf", "#dd6b20", "#0ea5e9", "#22c55e", "#a855f7",
    "#f43f5e", "#0f766e", "#b83280", "#ca8a04", "#2b6cb0",
] * 10

c_vals_l = [
    "#feb2b2", "#90cdf4", "#9ae6b4", "#faf089", "#fbb6ce",
    "#c3dafe", "#fed7aa", "#bae6fd", "#bbf7d0", "#e9d5ff",
    "#fecdd3", "#a7f3d0", "#f9a8d4", "#fde68a", "#bfdbfe",
] * 10

c_vals_d = [
    "#9b2c2c", "#2c5282", "#276749", "#975a16", "#97266d",
    "#4338ca", "#7b341e", "#0369a1", "#15803d", "#6b21a8",
    "#9f1239", "#0f4c3a", "#702459", "#854d0e", "#1e3a8a",
] * 10

l_vals = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
markers_vals = ['o', 'v', '*', '+', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
linestyles = ["-", "--", "-."]

ONETASK_DIR = Path("onetask")
ONETASK_DATA_DIR = Path("onetask_data")


def _generate_random_orthonormal_matrix(N, num_columns=3):
    """N x num_columns matrix with orthonormal columns."""
    Q, _ = np.linalg.qr(np.random.randn(N, num_columns))
    return Q[:, :num_columns]


def main(aname):
    result_path = ONETASK_DIR / f"param_{aname}_result.npz"
    param_path = ONETASK_DIR / f"param_{aname}_param.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Result traces not found: {result_path}")

    with open(param_path) as f:
        cfg = json.load(f)
    task_params = cfg["task_params"]
    net_params = cfg["net_params"]
    fixate_off = task_params["fixate_off"]

    data = np.load(result_path, allow_pickle=True)
    hyp_dict = data["hyp_dict"].item()
    seed = int(data["seed"])
    shift_index = int(data["shift_index"])
    color_by = str(data["color_by"])
    counter_lst = data["counter_lst"]
    loss_lst = data["loss_lst"]
    acc_lst = data["acc_lst"]
    # test_input_np is the SAVED validation set, kept because the per-stage
    # modulation traces (Ms_orig_stages etc.) are aligned to exactly these
    # trials.
    test_input_np = data["test_input_np"]
    test_output_np = data["test_output_np"]
    net_out_final = data["net_out_final"]   # final-stage network output on the test set
    input_matrix_final = data["input_matrix_final"]
    labels = data["labels"]
    Ms_orig_stages = data["Ms_orig_stages"]          # (stage, batch, T, hidden, input)
    hs_stages = data["hs_stages"]                    # (stage, batch, T, hidden)
    bs_stages = data["bs_stages"]                    # (stage, batch, T, hidden)
    Wall_stages = data["Wall_stages"]                # (stage, hidden, hidden) — MP layer W on embedded input
    Woutput_stages = data["Woutput_stages"]          # (stage, out, hidden)
    Winput_stages = data["Winput_stages"]            # (stage, hidden, n_raw_input) — input embedding
    input_layer_add = bool(net_params.get("input_layer_add", False))
    response_start = int(data["response_start"])
    stimulus_start = int(data["stimulus_start"])
    stimulus_end = int(data["stimulus_end"])
    all_breaks = data["all_breaks"].tolist()

    stages_num = Ms_orig_stages.shape[0]
    batch_nums = Ms_orig_stages.shape[1]
    print(f"stages={stages_num}, batch={batch_nums}, "
          f"stim=({stimulus_start},{stimulus_end}), response_start={response_start}")

    # counter_lst / loss_lst / acc_lst can differ in length by one because
    # train_network appends to them at slightly different points. Align every
    # per-stage quantity (and anything plotted against counter_lst) to a single
    # common length so x/y dimensions always match.
    n_common = min(len(counter_lst), len(loss_lst), len(acc_lst), stages_num)
    if n_common != stages_num or n_common != len(counter_lst):
        print(f"  [warn] trimming traces to common length {n_common} "
              f"(counter={len(counter_lst)}, loss={len(loss_lst)}, "
              f"acc={len(acc_lst)}, stages={stages_num})")
    counter_lst = np.asarray(counter_lst)[:n_common]
    loss_lst = np.asarray(loss_lst)[:n_common]
    acc_lst = np.asarray(acc_lst)[:n_common]
    # Use the trailing n_common stages so the final (best-trained) stage is kept.
    stage_sel = slice(stages_num - n_common, stages_num)
    Ms_orig_stages = Ms_orig_stages[stage_sel]
    hs_stages = hs_stages[stage_sel]
    bs_stages = bs_stages[stage_sel]
    Wall_stages = Wall_stages[stage_sel]
    Woutput_stages = Woutput_stages[stage_sel]
    if input_layer_add and Winput_stages.size > 0:
        Winput_stages = Winput_stages[stage_sel]
    stages_num = n_common

    save_dir = ONETASK_DIR / aname
    save_dir.mkdir(parents=True, exist_ok=True)
    for _old in save_dir.iterdir():
        if _old.is_file():
            _old.unlink()

    # ── Figure: loss / accuracy across training ──────────────────────────────
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(counter_lst, loss_lst, "-o", c=c_vals[0])
    ax1.set_ylabel("MSE Loss", color=c_vals[0], fontsize=15)
    ax1.tick_params(axis='y', colors=c_vals[0], labelsize=12)
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(counter_lst, acc_lst, "-o", c=c_vals[1])
    ax2.axhline(y=1 / 8, linestyle="--", label="By Chance")
    ax2.set_ylabel("Accuracy", color=c_vals[1], fontsize=15)
    ax2.tick_params(axis='y', colors=c_vals[1], labelsize=12)
    ax2.legend(loc='best', frameon=True, fontsize=12)
    ax1.set_xlabel("# Dataset", fontsize=15)
    ax1.set_xscale("log")
    fig.tight_layout()
    fig.savefig(save_dir / f"loss_acc_{aname}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {save_dir / f'loss_acc_{aname}.png'}")

    # ── Input weight matrix heatmap (notebook cell 9) ────────────────────────
    if net_params["input_layer_add"] and input_matrix_final.size > 0:
        figinp, axinp = plt.subplots(1, 1, figsize=(4, 4))
        sns.heatmap(input_matrix_final, ax=axinp, square=True, cmap='coolwarm')
        axinp.set_title("Input weight (W_initial_linear)", fontsize=12)
        figinp.tight_layout()
        figinp.savefig(save_dir / f"input_weight_{aname}.png", dpi=300)
        plt.close(figinp)
        print(f"  Saved figure: {save_dir / f'input_weight_{aname}.png'}")

    # ── Example single-trial input & output ──────────────────────────────────
    # One representative trial. Input layout (shift_index=1): channel 0 =
    # fixation, channels [1,2] and [3,4] are two stimulus (cos,sin) groups, the
    # last channel = task cue. Only ONE stimulus group is active per trial, so
    # we plot 4 channels: Fixation, Stim Cos, Stim Sin, Task Cue.
    # Output (3 channels): Fixation, Output Cos, Output Sin (network output).
    b0 = 0
    fix_ch = 0
    task_ch = test_input_np.shape[-1] - 1
    # The two candidate stimulus groups; pick the one carrying signal in trial b0.
    groups = [(1, 2), (3, 4)]
    group_energy = [np.abs(test_input_np[b0, :, list(g)]).sum() for g in groups]
    cos_ch, sin_ch = groups[int(np.argmax(group_energy))]

    figex, axex = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    in_specs = [(fix_ch, "Fixation"), (cos_ch, "Stim Cos"),
                (sin_ch, "Stim Sin"), (task_ch, "Task Cue")]
    for k, (ch, lab) in enumerate(in_specs):
        axex[0].plot(test_input_np[b0, :, ch], color=c_vals[k % len(c_vals)], label=lab)
    axex[0].set_ylabel("Input", fontsize=12)
    axex[0].set_title(f"Example trial (stimulus = {int(labels[b0, 0])})", fontsize=11)

    out_labels = ["Fixation", "Output Cos", "Output Sin"]
    for out_idx in range(min(test_output_np.shape[-1], len(out_labels))):
        # Target output as a faded shadow (no legend entry), network output on top.
        axex[1].plot(test_output_np[b0, :, out_idx], color=c_vals_l[out_idx % len(c_vals_l)],
                     linewidth=4, alpha=0.6)
        axex[1].plot(net_out_final[b0, :, out_idx], color=c_vals[out_idx % len(c_vals)],
                     label=out_labels[out_idx])
    axex[1].set_ylabel("Output", fontsize=12)
    axex[1].set_xlabel("Time step", fontsize=12)

    for ax in axex:
        ax.legend(fontsize=7, frameon=True, loc="best", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    figex.tight_layout()
    figex.savefig(save_dir / f"example_trial_{aname}.png", dpi=300)
    plt.close(figex)
    print(f"  Saved example trial figure: {save_dir / f'example_trial_{aname}.png'}")

    # Save the exact traces so paper_plot can re-render this figure identically:
    # the chosen trial's input channels (with their labels) and the network /
    # target output channels.
    import pickle as _pickle
    example_trial_pkl = {
        "aname": aname,
        "stimulus": int(labels[b0, 0]),
        "input_specs": [(int(ch), lab) for ch, lab in in_specs],
        "input": np.asarray(test_input_np[b0]),          # (T, n_input)
        "output_labels": list(out_labels),
        "net_output": np.asarray(net_out_final[b0]),      # (T, n_output)
        "target_output": np.asarray(test_output_np[b0]),  # (T, n_output)
        # Trial-period boundaries, for shading sessions like onetask_show.
        "stimulus_start": int(stimulus_start),
        "stimulus_end": int(stimulus_end),
        "response_start": int(response_start),
    }
    with open(save_dir / f"example_trial_{aname}.pkl", "wb") as _f:
        _pickle.dump(example_trial_pkl, _f)
    print(f"  Saved example trial data: {save_dir / f'example_trial_{aname}.pkl'}")

    # ── Fixon vs Task projection onto readout (cancellation) ─────────────────
    # For each training stage, project the modulated weight's response to the
    # fixation-on and task inputs onto the readout directions, and track how the
    # network learns to CANCEL the fixon contribution against the task input
    # before the response period.
    def plot_trajectory_by_index(label_index, stage_iter, verbose=False):
        """Replicates the notebook's plot_trajectory_by_index.

        Projects the modulated weight's response to decomposed input components
        (fixon/fixoff/stimulus/task) onto the readout directions. saver1 uses
        combined components (allX1); saver2 uses individual components (allX2).
        When verbose, emits the exhaustive-search, fixon-task-diff, and per-stim
        cancellation ("show") figures for this stage.

        The MP-layer weight W operates on the EMBEDDED input (W_initial_linear @
        raw_input), so each raw input component is mapped through W_input first;
        W_input is identity when there is no input layer."""
        W_ = Wall_stages[stage_iter]
        W_output = Woutput_stages[stage_iter]
        Ms_orig = Ms_orig_stages[stage_iter]
        bs = bs_stages[stage_iter]
        if input_layer_add and Winput_stages.size > 0:
            W_input = Winput_stages[stage_iter]
        else:
            W_input = np.eye(W_.shape[1])

        T = test_input_np.shape[1]

        if verbose:
            figsize1, figsize2 = 3, 6
            figexh1, axsexh1 = plt.subplots(3, 3, figsize=(figsize2 * 3, figsize1 * 3))
            figexh2, axsexh2 = plt.subplots(4, 3, figsize=(figsize2 * 3, figsize1 * 4))
            figdiff, axsdiff = plt.subplots(1, 2, figsize=(4 * 2, 2))

        task_labels_across_batch = []
        saver_shape1 = (3, 3)
        saver1 = np.empty((batch_nums, saver_shape1[0], saver_shape1[1]), dtype=object)
        saver_shape2 = (4, 3)
        saver2 = np.empty((batch_nums, saver_shape2[0] + 1, saver_shape2[1]), dtype=object)
        saver2_random = np.empty((batch_nums, saver_shape2[0] + 1, saver_shape2[1]), dtype=object)
        random_output_Y_lst = [_generate_random_orthonormal_matrix(W_output.shape[1]) for _ in range(10)]

        allX1name = ["x_fixon+x_task", "x_fixoff+x_task", "x_stimulus+x_fixon+x_task"]
        allX2name = ["x_fixon", "x_fixoff", "x_stimulus", "x_task"]
        allYname = ["y_fix", "Y_resp1", "Y_resp2"]

        for batch_iter in range(batch_nums):
            labels_for_batch = labels[batch_iter, 0]
            if labels_for_batch not in label_index:
                continue

            x_batch_taskinfo = test_input_np[batch_iter, :, :][:, 6 - shift_index:][0, :]
            task_specific = np.where(x_batch_taskinfo == 1)[0]
            assert len(task_specific) == 1
            task_specific = task_specific[0]
            task_labels_across_batch.append(task_specific)

            for i in range(saver_shape1[0]):
                for j in range(saver_shape1[1]):
                    saver1[batch_iter, i, j] = np.array([])
            for i in range(saver_shape2[0] + 1):
                for j in range(saver_shape2[1]):
                    saver2[batch_iter, i, j] = np.array([])
                    saver2_random[batch_iter, i, j] = np.array([])

            for time_iter in range(T):
                x = test_input_np[batch_iter, time_iter, :].reshape(-1, 1)
                input_length = len(x)
                x_fixon, x_fixoff, x_stimulus, x_task = [np.zeros((input_length, 1)) for _ in range(4)]
                x_fixon[0, 0] = x[0, 0]
                x_fixoff[1, 0] = x[1, 0] if fixate_off else 0
                x_stimulus[2 - shift_index:6 - shift_index, 0] = x[2 - shift_index:6 - shift_index, 0]
                x_task[6 - shift_index:, 0] = x[6 - shift_index:, 0]

                Mt = Ms_orig[batch_iter, time_iter, :, :]
                bt = bs[batch_iter, time_iter, :].reshape(-1, 1)
                middle = W_ + W_ * Mt

                y_fix = W_output[0, :].reshape(1, -1)
                Y_resp1 = W_output[1, :].reshape(1, -1)
                Y_resp2 = W_output[2, :].reshape(1, -1)

                # Combined-component inputs (allX1) and individual ones (allX2),
                # each embedded into the hidden-dim space via W_input.
                if fixate_off:
                    allX1 = [x_fixon + x_task, x_fixoff + x_task, x_stimulus + x_fixon + x_task]
                else:
                    allX1 = [x_fixon + x_task, x_task, x_stimulus + x_fixon + x_task]
                allX1 = [W_input @ xc for xc in allX1]
                allX2 = [W_input @ xc for xc in (x_fixon, x_fixoff, x_stimulus, x_task)]
                allY = [y_fix, Y_resp1, Y_resp2]

                for yiter in range(len(allY)):
                    for xiter in range(len(allX1)):
                        step1 = middle @ allX1[xiter] + bt
                        res1 = allY[yiter] @ step1
                        saver1[batch_iter, xiter, yiter] = np.append(
                            saver1[batch_iter, xiter, yiter], res1[0, 0])

                for y1 in range(len(allY)):
                    for x1 in range(len(allX2)):
                        step1 = middle @ allX2[x1]
                        res2 = allY[y1] @ step1
                        res2_random = [((rY[:, y1].reshape(1, -1)) @ middle @ allX2[x1])[0, 0]
                                       for rY in random_output_Y_lst]
                        saver2[batch_iter, x1, y1] = np.append(saver2[batch_iter, x1, y1], res2[0, 0])
                        saver2_random[batch_iter, x1, y1] = np.append(
                            saver2_random[batch_iter, x1, y1], np.mean(res2_random))

                # bias projection onto each readout
                for y_iter2 in range(len(allY)):
                    res2 = allY[y_iter2] @ bt
                    saver2[batch_iter, len(allX2), y_iter2] = np.append(
                        saver2[batch_iter, len(allX2), y_iter2], res2[0, 0])

            if verbose:
                ls = l_vals[task_specific % len(l_vals)]
                cb = c_vals[labels_for_batch % len(c_vals)]
                cbl = c_vals_l[labels_for_batch % len(c_vals_l)]
                for i in range(saver_shape1[0]):
                    for j in range(saver_shape1[1]):
                        axsexh1[i, j].plot(saver1[batch_iter, i, j], color=cb, linestyle=ls)
                for i in range(saver_shape2[0]):
                    for j in range(saver_shape2[1]):
                        axsexh2[i, j].plot(saver2[batch_iter, i, j], color=cb, linestyle=ls)
                axsdiff[0].plot(saver2[batch_iter, 0, 1] + saver2[batch_iter, 3, 1], color=cb, linestyle=ls)
                axsdiff[0].plot(saver2_random[batch_iter, 0, 1] + saver2_random[batch_iter, 3, 1], color=cbl, linestyle=ls)
                axsdiff[1].plot(saver2[batch_iter, 0, 2] + saver2[batch_iter, 3, 2], color=cb, linestyle=ls)
                axsdiff[1].plot(saver2_random[batch_iter, 0, 2] + saver2_random[batch_iter, 3, 2], color=cbl, linestyle=ls)

        if verbose:
            # per-stimulus fixon/task/combine cancellation ("show") figure
            figpaper, axspaper = plt.subplots(8, 1, figsize=(6, figsize1 * 8))
            temp_saver = []
            show_save = {}   # stimulus label -> {fixon, task, combine} traces
            for batch_iter in range(batch_nums):
                labels_for_batch = labels[batch_iter, 0]
                if labels_for_batch in label_index and labels_for_batch not in temp_saver:
                    f_fixon = saver2[batch_iter, 0, 1]
                    f_task = saver2[batch_iter, 3, 1]
                    f_bias = saver2[batch_iter, -1, 1]
                    k = len(temp_saver)
                    if k >= len(axspaper):
                        break
                    axspaper[k].plot(f_fixon, color=c_vals[0], linestyle=l_vals[0], label="Fixon")
                    axspaper[k].plot(f_task + f_bias, color=c_vals[1], linestyle=l_vals[1], label="Task")
                    axspaper[k].plot(f_fixon + f_task + f_bias, color=c_vals[2], linestyle=l_vals[3],
                                     linewidth=3, label="Combine")
                    axspaper[k].axhline(0, color=c_vals[3])
                    axspaper[k].set_xlabel("Timestep", fontsize=15)
                    axspaper[k].set_ylabel("Modulation Component", fontsize=15)
                    show_save[int(labels_for_batch)] = {
                        "fixon": np.asarray(f_fixon, dtype=float),
                        "task": np.asarray(f_task + f_bias, dtype=float),
                        "combine": np.asarray(f_fixon + f_task + f_bias, dtype=float),
                    }
                    temp_saver.append(labels_for_batch)
            for axsp in axspaper:
                axsp.legend(loc="best", frameon=True, fontsize=12)
                axsp.set_ylim([-2.0, 2.0])
            figpaper.tight_layout()
            figpaper.savefig(save_dir / f"show_{aname}.png", dpi=300)
            plt.close(figpaper)
            print(f"  Saved figure: {save_dir / f'show_{aname}.png'}")

            # Save the underlying traces so paper_plot can re-render this figure.
            import pickle as _pickle
            show_pkl = {
                "aname": aname,
                "stage_iter": int(stage_iter),
                "all_breaks": all_breaks,
                "response_start": int(response_start),
                "stimulus_start": int(stimulus_start),
                "stimulus_end": int(stimulus_end),
                "per_stimulus": show_save,  # {stim_label: {fixon, task, combine}}
            }
            with open(save_dir / f"show_{aname}.pkl", "wb") as _f:
                _pickle.dump(show_pkl, _f)
            print(f"  Saved show data: {save_dir / f'show_{aname}.pkl'}")

            for i in range(saver_shape1[0]):
                for j in range(saver_shape1[1]):
                    axsexh1[i, j].set_ylim([-1.2, 1.2])
                    axsexh1[i, j].set_title(f"{allX1name[i]} & {allYname[j]}")
            for i in range(saver_shape2[0]):
                for j in range(saver_shape2[1]):
                    axsexh2[i, j].set_ylim([-1.2, 1.2])
                    axsexh2[i, j].set_title(f"{allX2name[i]} & {allYname[j]}")
            for ax in np.concatenate((axsexh1.flatten(), axsexh2.flatten())):
                for bi, breaks in enumerate(all_breaks):
                    for bb in breaks:
                        ax.axvline(bb, linestyle="--", c=c_vals[bi % len(c_vals)])

            figexh1.suptitle(f"Exhaustive Search 1 {color_by} at Stage {stage_iter}")
            figexh1.tight_layout()
            figexh1.savefig(save_dir / f"es1_{aname}.png", dpi=300)
            plt.close(figexh1)
            print(f"  Saved figure: {save_dir / f'es1_{aname}.png'}")
            figexh2.suptitle(f"Exhaustive Search 2 {color_by} Stage {stage_iter}")
            figexh2.tight_layout()
            figexh2.savefig(save_dir / f"es2_{aname}.png", dpi=300)
            plt.close(figexh2)
            print(f"  Saved figure: {save_dir / f'es2_{aname}.png'}")
            axsdiff[0].set_title("Stimulus 1")
            axsdiff[1].set_title("Stimulus 2")
            figdiff.suptitle(f"Fixon-Task at Stage {stage_iter}")
            figdiff.tight_layout()
            figdiff.savefig(save_dir / f"diff_{aname}.png", dpi=300)
            plt.close(figdiff)
            print(f"  Saved figure: {save_dir / f'diff_{aname}.png'}")

        return task_labels_across_batch, saver2, saver2_random

    all_trajectory, all_trajectory_random = [], []
    label_index = np.unique(labels)
    for stage_iter in range(stages_num):
        _, saver2, saver2_random = plot_trajectory_by_index(
            label_index, stage_iter, verbose=(stage_iter == stages_num - 1))
        all_trajectory.append(saver2)
        all_trajectory_random.append(saver2_random)

    def analyze_trajectory(save_trajectory, save_trajectory_random):
        def process(trajectory, ind=False):
            results = []
            for batch in trajectory:
                if batch[0, 1] is None:
                    continue
                stim1_fixon = batch[0, 1][stimulus_start:response_start]
                stim1_task = batch[3, 1][stimulus_start:response_start]
                bias = batch[4, 1][stimulus_start:response_start] if ind else np.zeros_like(stim1_fixon)
                results.append([np.mean(np.abs(stim1_fixon + stim1_task + bias)),
                                np.mean(np.abs(stim1_fixon)), np.mean(np.abs(stim1_task))])
            return np.array(results)
        result = process(save_trajectory, True)
        result_random = process(save_trajectory_random)
        return (np.mean(result[:, 0]), np.mean(result[:, 1]), np.mean(result[:, 2]),
                np.mean(result_random[:, 0]), np.mean(result_random[:, 1]), np.mean(result_random[:, 2]))

    fixon_task_diff = np.array([analyze_trajectory(all_trajectory[i], all_trajectory_random[i])
                                for i in range(stages_num)])

    figc, axc = plt.subplots(figsize=(6, 3))
    axc.plot(counter_lst, fixon_task_diff[:, 0], "-o", c=c_vals[0], label=r"|Fix − Task|")
    axc.plot(counter_lst, fixon_task_diff[:, 1], "-o", c=c_vals[1], label=r"|Fix|")
    axc.plot(counter_lst, fixon_task_diff[:, 2], "-o", c=c_vals[2], label=r"|Task|")
    axc.legend(loc="best", fontsize=12, frameon=True)
    axc.set_ylabel("Magnitude Projection", fontsize=15)
    axc.set_xlabel("# Dataset", fontsize=15)
    axc.set_xscale("log")
    figc.tight_layout()
    figc.savefig(save_dir / f"cancel_{aname}.png", dpi=300)
    plt.close(figc)
    print(f"  Saved figure: {save_dir / f'cancel_{aname}.png'}")

    # ── Modulation-change / synaptic & hidden correlation across learning ────
    modulation_dict_diff_lst, modulation_dict_lst = [], []
    hidden_output_dict_lst, hidden_dict_lst = [], []

    # Fixon column of the input embedding (see m_pca note): M's last axis is the
    # embedded input, so the fixon-channel modulation effect on hidden units is
    # M @ W_input[:, fixon_col], not M[..., 0].
    _fixon_col = 0
    for stage_iter in range(stages_num):
        Woutput = Woutput_stages[stage_iter]
        Ms_orig = Ms_orig_stages[stage_iter]
        hs = hs_stages[stage_iter]
        hs_stimulus = hs[:, stimulus_start:stimulus_end, :]
        if input_layer_add and Winput_stages.size > 0:
            w_fixon = Winput_stages[stage_iter][:, _fixon_col]   # (embed,)
            Ms_fixon_proj = Ms_orig @ w_fixon                    # (batch, T, hidden)
        else:
            Ms_fixon_proj = Ms_orig[:, :, :, _fixon_col]
        # Per-period fixon-modulation traces (projected onto the raw fixon input).
        Mf_fix = Ms_fixon_proj[:, :stimulus_start, :]
        Mf_stimulus = Ms_fixon_proj[:, stimulus_start:stimulus_end, :]
        Mf_delay = Ms_fixon_proj[:, stimulus_end:response_start, :]
        Mf_response = Ms_fixon_proj[:, response_start:, :]
        Mf_all = [Mf_fix, Mf_stimulus, Mf_delay, Mf_response]

        modulation_diff_dict, modulation_dict, hidden_output_dict, hidden_dict = {}, {}, {}, {}
        for batch_iter in range(batch_nums):
            hs_stim_batch = hs_stimulus[batch_iter, :, :]
            hs_stim_out = hs_stim_batch @ Woutput.T
            # change of fixon modulation (end - start) per period
            Ms_fixon = [Mf[batch_iter, -1, :] - Mf[batch_iter, 0, :] for Mf in Mf_all]
            modulation_diff_dict[labels[batch_iter, 0]] = Ms_fixon
            # fixon modulation at end of stimulus (for synaptic-cosine analysis)
            modulation_dict[labels[batch_iter, 0]] = Mf_stimulus[batch_iter, -1, :]
            hidden_output_dict[labels[batch_iter, 0]] = hs_stim_out
            hidden_dict[labels[batch_iter, 0]] = hs_stim_batch[-1, :]
        modulation_dict_diff_lst.append(modulation_diff_dict)
        modulation_dict_lst.append(modulation_dict)
        hidden_output_dict_lst.append(hidden_output_dict)
        hidden_dict_lst.append(hidden_dict)

    modulation_change_stage = [[], [], [], []]
    m_corr_stage, h_corr_stage = [], []
    fig_hc, axs_hc = plt.subplots(2, 1, figsize=(6, 3 * 2))

    def analyze_hm_change(lst, i, index=None):
        md = lst[i]
        if index is None:
            md_m = [np.array(v) for v in md.values()]
        else:
            md_m = [np.array(v[index]) for v in md.values()]
        md_m = np.column_stack(md_m).T  # num_stimulus x hidden
        mc_stage = list(np.mean(np.abs(md_m), axis=1))
        synaptic_corr = cosine_similarity(md_m)
        # Mean pairwise cosine over the STRICT upper triangle (k=1): exclude the
        # all-ones diagonal and avoid double-counting / the zeroed lower triangle,
        # so the result is a genuine cosine in [-1, 1].
        n = synaptic_corr.shape[0]
        iu = np.triu_indices(n, k=1)
        mean_cos = float(np.nanmean(synaptic_corr[iu])) if n > 1 else np.nan
        return mean_cos, mc_stage, md_m

    for i in range(stages_num):
        m_mean_corr, _, _ = analyze_hm_change(modulation_dict_lst, i)
        h_mean_corr, _, _ = analyze_hm_change(hidden_dict_lst, i)
        m_corr_stage.append(m_mean_corr)
        h_corr_stage.append(h_mean_corr)
        md_m_diff_stim = md_m_diff_response = None
        for p in range(4):
            _, mc_stage, md_m_diff = analyze_hm_change(modulation_dict_diff_lst, i, p)
            if p == 1:
                md_m_diff_stim = md_m_diff
            elif p == 3:
                md_m_diff_response = md_m_diff
            modulation_change_stage[p].append(mc_stage)
        if i == stages_num - 1:
            sns.heatmap(md_m_diff_stim, ax=axs_hc[0], cmap="coolwarm")
            sns.heatmap(md_m_diff_response, ax=axs_hc[1], cmap="coolwarm")
            for ax_hc in axs_hc:
                ax_hc.set_xticks([]); ax_hc.set_yticks([])
                ax_hc.set_xlabel("Hidden", fontsize=15)
                ax_hc.set_ylabel("Stimuli", fontsize=15)
            fig_hc.tight_layout()
            fig_hc.savefig(save_dir / f"modulation_heatmap_{aname}.png", dpi=300)
            plt.close(fig_hc)
            print(f"  Saved figure: {save_dir / f'modulation_heatmap_{aname}.png'}")

            # Save the two heatmap matrices for paper_plot reuse.
            import pickle as _pickle
            with open(save_dir / f"modulation_heatmap_{aname}.pkl", "wb") as _f:
                _pickle.dump({
                    "aname": aname,
                    "stage_iter": int(i),
                    "stim_change": np.asarray(md_m_diff_stim, dtype=float),       # (n_stim, hidden)
                    "response_change": np.asarray(md_m_diff_response, dtype=float),
                }, _f)
            print(f"  Saved modulation heatmap data: "
                  f"{save_dir / f'modulation_heatmap_{aname}.pkl'}")

    modulation_change_stage = np.array(modulation_change_stage)
    m_corr_stage = np.array(m_corr_stage)
    h_corr_stage = np.array(h_corr_stage)
    period_names = ["Fixation", "Stimulus", "Delay", "Response"]

    figmc, axsmc = plt.subplots(3, 1, figsize=(6, 3 * 3))
    for i in range(4):
        mcs = modulation_change_stage[i]
        axsmc[0].plot(counter_lst, np.mean(mcs, axis=1), "-o", c=c_vals[i], label=period_names[i])
        axsmc[0].fill_between(counter_lst, np.mean(mcs, axis=1) - np.std(mcs, axis=1),
                              np.mean(mcs, axis=1) + np.std(mcs, axis=1), color=c_vals_l[i])
    axsmc[0].set_ylabel("Change of Modulation", fontsize=15)
    axsmc[0].legend(loc="best", frameon=True, fontsize=12)
    axsmc[1].plot(counter_lst, m_corr_stage, "-o")
    axsmc[1].set_ylabel("Synaptic Cosine\nbetween Stimulus", fontsize=13)
    axsmc[2].plot(counter_lst, h_corr_stage, "-o")
    axsmc[2].set_ylabel("Hidden Activity Cosine\nbetween Stimulus", fontsize=13)
    for ax in axsmc:
        ax.set_xlabel("# Dataset", fontsize=15)
        ax.set_xscale("log")
    figmc.tight_layout()
    figmc.savefig(save_dir / f"modulation_change_{aname}.png", dpi=300)
    plt.close(figmc)
    print(f"  Saved figure: {save_dir / f'modulation_change_{aname}.png'}")

    # Save the raw (un-normalized) mean cosine curves for cross-seed plotting.
    ONETASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ONETASK_DATA_DIR / f"corr_{aname}.npz",
             counter_lst=counter_lst,
             m_corr_stage=m_corr_stage,
             h_corr_stage=h_corr_stage)

    # ── Hidden-trajectory length across learning ─────────────────────────────
    def traj_length(arr):
        return np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))

    hidden_length_all = []
    for stage_iter in range(stages_num):
        hstage = hidden_output_dict_lst[stage_iter]
        hstage = {k: hstage[k] for k in sorted(hstage.keys())}
        hidden_length_all.append([traj_length(arr) for arr in hstage.values()])
    hidden_length_all = np.array(hidden_length_all)

    figt, axst = plt.subplots(figsize=(6, 3))
    for i in range(hidden_length_all.shape[1]):
        axst.plot(counter_lst, hidden_length_all[:, i], "-o", c=c_vals[i % len(c_vals)])
    axst.set_xlabel("# Dataset", fontsize=15)
    axst.set_xscale("log")
    axst.set_ylabel("Length of Hidden\nState Trajectory", fontsize=13)
    figt.tight_layout()
    figt.savefig(save_dir / f"length_hidden_state_{aname}.png", dpi=300)
    plt.close(figt)
    print(f"  Saved figure: {save_dir / f'length_hidden_state_{aname}.png'}")

    # ── Weight-component projection to output across learning ────────────────
    fixon_task_projoutput = []
    for stage_iter in range(stages_num):
        W = Wall_stages[stage_iter]
        W_output = Woutput_stages[stage_iter]
        bias = np.mean(bs_stages[stage_iter], axis=0)
        # W is in the embedded space; compose the input embedding so we read the
        # raw fixon / task channels' effective weight onto the hidden units.
        if input_layer_add and Winput_stages.size > 0:
            W_input = Winput_stages[stage_iter]
            W_fixon = (W @ W_input[:, 0]).reshape(-1, 1)
            W_task = (W @ W_input[:, 6 - shift_index]).reshape(-1, 1)
        else:
            W_fixon = W[:, 0].reshape(-1, 1)
            W_task = W[:, 6 - shift_index].reshape(-1, 1)
        fixon_output, task_output = W_output[1:, :] @ W_fixon, W_output[1:, :] @ W_task
        bias_output = np.mean(bias @ (W_output[1:, :].T), axis=0)
        fixon_task_projoutput.append([
            fixon_output[0][0] + bias_output[0], task_output[0][0],
            fixon_output[1][0] + bias_output[1], task_output[1][0],
        ])
    fixon_task_projoutput = np.array(fixon_task_projoutput)

    figw, axw = plt.subplots(figsize=(6, 3))
    axw.plot(counter_lst, fixon_task_projoutput[:, 0], marker="o", color=c_vals[0], linestyle=l_vals[0], label="fixon→out1")
    axw.plot(counter_lst, fixon_task_projoutput[:, 1], marker="o", color=c_vals[0], linestyle=l_vals[1], label="task→out1")
    axw.plot(counter_lst, fixon_task_projoutput[:, 0] + fixon_task_projoutput[:, 1], marker="o",
             color=c_vals[1], linestyle=l_vals[2], linewidth=1, label="fixon+task→out1")
    axw.axhline(0, color=c_vals[1], linestyle=l_vals[2])
    axw.legend(loc="lower left", fontsize=12, frameon=True)
    axw.set_xlabel("# Dataset", fontsize=15)
    axw.set_xscale("log")
    axw.set_ylabel("Weight Component\nProjection", fontsize=13)
    figw.tight_layout()
    figw.savefig(save_dir / f"w_to_output_{aname}.png", dpi=300)
    plt.close(figw)
    print(f"  Saved figure: {save_dir / f'w_to_output_{aname}.png'}")

    # ── Low-D PCA of fixon modulation during the stimulus period (final stage) ─
    #
    # Goal: visualize how the fast plasticity acting on the FIXON input pathway
    # is organized across hidden units, and whether that organization separates
    # by stimulus direction during the stimulus epoch.
    #
    # Step 1 — recover the fixon modulation.
    #   The recorded modulation tensor M has shape (batch, T, hidden, embed):
    #   its last axis is the EMBEDDED input (output of the trained input layer
    #   W_initial_linear), NOT the raw input channels. So M[..., 0] would be
    #   embedded-dim 0, which is a meaningless mixed coordinate — not fixon.
    #   The effective modulation of the raw fixon input on each hidden unit is
    #   obtained by contracting M's embedded-input axis with the fixon column of
    #   the embedding (w_fixon = W_input[:, fixon_col], shape (embed,)):
    #       fixon_mod[b, t, hidden] = sum_e  M[b, t, hidden, e] * w_fixon[e]
    #   giving a per-timestep hidden-unit vector (batch, T, hidden).
    #   (Without an input layer, M's last axis IS the raw input, so slice it.)
    #
    # Step 2 — build the PCA basis from the FIXON modulation itself.
    #   We deliberately fit PCA on fixon_mod (not the full M), using only the
    #   end-of-stimulus state pooled across trials. This puts the maximal
    #   variance of the quantity we plot (fixon modulation) into the top PCs, so
    #   any stimulus structure in the fixon pathway is maximally visible. (An
    #   alternative — fitting on the full M and slicing fixon afterward — would
    #   instead show fixon within a shared, channel-agnostic modulation frame.)
    #
    # Step 3 — project the whole trial and plot the stimulus-period trajectory
    #   in three PC planes (PC1-2, PC1-3, PC2-3), colored by stimulus direction.
    fighs, axshs = plt.subplots(3, 1, figsize=(6, 3 * 3), squeeze=False)
    stage_iter = stages_num - 1
    PCA_downsample = 3
    Ms_orig = Ms_orig_stages[stage_iter]               # (batch, T, hidden, embed)
    fixon_col = 0  # raw input column for the fixation-on channel

    if input_layer_add and Winput_stages.size > 0:
        w_fixon = Winput_stages[stage_iter][:, fixon_col]   # (embed,)
        fixon_mod = Ms_orig @ w_fixon                       # (batch, T, hidden)
    else:
        fixon_mod = Ms_orig[:, :, :, fixon_col]             # raw input already

    # PCA basis fit on end-of-stimulus fixon modulation, pooled over trials.
    pca = PCA(n_components=PCA_downsample)
    fixon_end = fixon_mod[:, stimulus_end:stimulus_end + 1, :]
    pca.fit(fixon_end.reshape(-1, fixon_end.shape[-1]))
    lowd = pca.transform(
        fixon_mod.reshape(-1, fixon_mod.shape[-1])
    ).reshape(fixon_mod.shape[0], fixon_mod.shape[1], PCA_downsample)

    pairs = [(0, 0, 1), (1, 0, 2), (2, 1, 2)]
    for i in range(lowd.shape[0]):
        data_batch = lowd[i, :, :]
        color = c_vals[labels[i, 0] % len(c_vals)]
        # All three panels show the stimulus period only.
        for row, xpc, ypc in pairs:
            axshs[row, 0].plot(data_batch[stimulus_start:stimulus_end, xpc],
                               data_batch[stimulus_start:stimulus_end, ypc],
                               marker=markers_vals[0], markersize=3, c=color, alpha=0.5)
        for row, xpc, ypc in pairs:
            axshs[row, 0].set_xlabel(f"PC {xpc+1}", fontsize=13)
            axshs[row, 0].set_ylabel(f"PC {ypc+1}", fontsize=13)
    fighs.tight_layout()
    fighs.savefig(save_dir / f"m_pca_{aname}.png", dpi=300)
    plt.close(fighs)
    print(f"  Saved figure: {save_dir / f'm_pca_{aname}.png'}")

    # Save the projected trajectories so paper_plot can re-render this figure.
    import pickle as _pickle
    with open(save_dir / f"m_pca_{aname}.pkl", "wb") as _f:
        _pickle.dump({
            "aname": aname,
            "stage_iter": int(stage_iter),
            "lowd": np.asarray(lowd, dtype=float),          # (batch, T, n_pc)
            "labels": np.asarray(labels).reshape(-1),       # stimulus label per trial
            "pairs": pairs,                                  # PC-plane index pairs
            "stimulus_start": int(stimulus_start),
            "stimulus_end": int(stimulus_end),
            "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=float),
        }, _f)
    print(f"  Saved m_pca data: {save_dir / f'm_pca_{aname}.pkl'}")

    print(f"All figures saved to {save_dir}/")

    # ── Cross-seed aggregate of correlation curves ───────────────────────────
    _plot_aggregate_corr(save_dir, aname)


def _plot_aggregate_corr(save_dir, aname):
    """Average the m_corr / h_corr curves across all saved corr_*.npz runs."""
    files = sorted(glob.glob(str(ONETASK_DATA_DIR / "corr_*.npz")))
    if len(files) < 1:
        return
    counters, m_all, h_all = [], [], []
    for f in files:
        d = np.load(f)
        counters.append(d["counter_lst"])
        m_all.append(d["m_corr_stage"])
        h_all.append(d["h_corr_stage"])
    # only aggregate runs with a common length
    lens = [len(c) for c in counters]
    common = min(lens)
    counters = np.array([c[:common] for c in counters])
    m_all = np.array([m[:common] for m in m_all])
    h_all = np.array([h[:common] for h in h_all])

    mean_counter = np.mean(counters, axis=0)
    figm, axm = plt.subplots(2, 1, figsize=(6, 3 * 2))
    axm[0].plot(mean_counter, m_all.mean(0), "-o", color=c_vals[0])
    axm[0].fill_between(mean_counter, m_all.mean(0) - m_all.std(0), m_all.mean(0) + m_all.std(0),
                        color=c_vals_l[0], alpha=0.2)
    axm[0].set_ylabel("Cos of Modulation", fontsize=14)
    axm[1].plot(mean_counter, h_all.mean(0), "-o", color=c_vals[0])
    axm[1].fill_between(mean_counter, h_all.mean(0) - h_all.std(0), h_all.mean(0) + h_all.std(0),
                        color=c_vals_l[0], alpha=0.2)
    axm[1].set_ylabel("Cos of Hidden Activity", fontsize=14)
    for ax in axm:
        ax.set_xlabel("# Dataset", fontsize=15)
        ax.set_xscale("log")
    figm.tight_layout()
    figm.savefig(save_dir / f"modulation_analysis_during_learning_{aname}.png", dpi=300)
    plt.close(figm)
    print(f"Aggregated {len(files)} runs into modulation_analysis_during_learning_{aname}.png")


def _discover_anames():
    """Return all experiment identifiers (param_*_result.npz) in onetask/,
    sorted by modification time (oldest first)."""
    results = sorted(ONETASK_DIR.glob("param_*_result.npz"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise FileNotFoundError("No param_*_result.npz found in ./onetask/. Run one_task.py first.")
    return [p.name[len("param_"):-len("_result.npz")] for p in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aname", type=str, default=None,
                        help="Experiment identifier. Omit to analyze ALL runs in ./onetask/.")
    args = parser.parse_args()

    anames = [args.aname] if args.aname else _discover_anames()
    print(f"Analyzing {len(anames)} run(s).")
    for a in anames:
        print(f"\n── Analyzing: {a} ──")
        try:
            main(a)
        except Exception as exc:
            print(f"  FAILED {a}: {exc}")
            import traceback
            traceback.print_exc()
