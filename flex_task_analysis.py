import os 
import numpy as np 
import matplotlib.pyplot as plt 

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',] * 10

def find_pkl_files_with_keywords(path, keywords):
    """
    """
    matched_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".pkl") and all(k in file for k in keywords):
                matched_files.append(os.path.join(root, file))
    return matched_files

fig, axs = plt.subplots(1,2,figsize=(4*2,4))

task_name = "delayfamily"
hiddennum = "200"
lr = "1e-03"

path = "./flextask/"
keywords_mpn = ["loss", "recFalse", "dmpn", task_name, hiddennum, lr]
results_mpn = find_pkl_files_with_keywords(path, keywords_mpn)
print(f"results_mpn: {results_mpn}")

mpn_acc = []
for idx, mpn_file in enumerate(results_mpn): 
    data = np.load(mpn_file, allow_pickle=True)
    mpn_acc.append(data["validation_acc"])
        
mpn_acc_mean = np.mean(np.array(mpn_acc), axis=0)
mpn_acc_std = np.std(np.array(mpn_acc), axis=0) / np.sqrt(np.array(mpn_acc).shape[0])
for ax in axs:
    ax.plot(data["batch_idx"], mpn_acc_mean, c=c_vals[0], label="MPN without Recurrent Hidden")
    ax.fill_between(data["batch_idx"], mpn_acc_mean-mpn_acc_std,  mpn_acc_mean+mpn_acc_std, color=c_vals[0], alpha=0.5)

path = "./flextask/"
keywords_rnn = ["loss", "recFalse", "rnn", task_name, hiddennum, lr]
results_rnn = find_pkl_files_with_keywords(path, keywords_rnn)
print(f"results_rnn: {results_rnn}")

rnn_acc = []
for idx, rnn_file in enumerate(results_rnn): 
    data = np.load(rnn_file, allow_pickle=True)
    rnn_acc.append(data["validation_acc"])

rnn_acc_mean = np.mean(np.array(rnn_acc), axis=0)
rnn_acc_std = np.std(np.array(rnn_acc), axis=0) / np.sqrt(np.array(rnn_acc).shape[0])
for ax in axs:
    ax.plot(data["batch_idx"], rnn_acc_mean, c=c_vals[1], label="RNN")
    ax.fill_between(data["batch_idx"], rnn_acc_mean-rnn_acc_std, rnn_acc_mean+rnn_acc_std, color=c_vals[1], alpha=0.5)

# 2025-11-04: whether to further compare with the dmpn with recurrent layer added 
# currently consider as a benchmark
mpn_recurrent = False
if mpn_recurrent:
    path = "./flextask/"
    keywords_mpn_rec = ["loss", "recTrue", "dmpn", task_name, hiddennum]
    results_mpn_rec = find_pkl_files_with_keywords(path, keywords_mpn_rec)
    print(f"results_mpn_rec: {results_mpn_rec}")

    mpn_rec_acc = []
    for idx, mpnrec_file in enumerate(results_mpn_rec): 
        data = np.load(mpnrec_file, allow_pickle=True)
        mpn_rec_acc.append(data["validation_acc"])
        # for ax in axs:
            # ax.plot(data["batch_idx"], data["validation_acc"], c=c_vals[2], alpha=0.1)

    mpn_rec_acc_mean = np.mean(np.array(mpn_rec_acc), axis=0)
    mpn_rec_acc_std = np.std(np.array(mpn_rec_acc), axis=0) / np.sqrt(np.array(mpn_rec_acc).shape[0])
    for ax in axs:
        ax.plot(data["batch_idx"], mpn_rec_acc_mean, c=c_vals[2], label="MPN with Recurrent Hidden")
        ax.fill_between(data["batch_idx"], mpn_rec_acc_mean-mpn_rec_acc_std, mpn_rec_acc_mean+mpn_rec_acc_std, color=c_vals[2])
            
for ax in axs: 
    ax.legend()
    ax.set_xlabel("# Batches", fontsize=15)
    ax.set_ylabel("Validation Accuracy", fontsize=15)
    ax.set_ylim([0, 1.1])
axs[1].set_xlim([0, 300])
axs[0].set_title("Learning Trajectory", fontsize=15)
axs[1].set_title("Zoomed In Learning Trajectory", fontsize=15)

fig.tight_layout()
fig.savefig(f"./flextask/compare_dmpn_rnn_{task_name}_N{hiddennum}_lr{lr}.png", dpi=300)








