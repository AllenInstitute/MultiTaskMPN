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

fig, ax = plt.subplots(1,1,figsize=(4,4))

path = "./flextask/"
keywords_mpn = ["loss", "recFalse", "dmpn"]
results_mpn = find_pkl_files_with_keywords(path, keywords_mpn)
print(f"results_mpn: {results_mpn}")

for idx, mpn_file in enumerate(results_mpn): 
    data = np.load(mpn_file, allow_pickle=True)
    if idx == 0:
        ax.plot(data["batch_idx"], data["validation_acc"], c=c_vals[0], label="MPN without Hidden")
    else:
        ax.plot(data["batch_idx"], data["validation_acc"], c=c_vals[0])

path = "./flextask/"
keywords_rnn = ["loss", "recFalse", "rnn"]
results_rnn = find_pkl_files_with_keywords(path, keywords_rnn)
print(f"results_rnn: {results_rnn}")

for idx, rnn_file in enumerate(results_rnn): 
    data = np.load(rnn_file, allow_pickle=True)
    if idx == 0:
        ax.plot(data["batch_idx"], data["validation_acc"], c=c_vals[1], label="RNN")
    else:
        ax.plot(data["batch_idx"], data["validation_acc"], c=c_vals[1])

ax.legend()
ax.set_xlabel("# Batches", fontsize=15)
ax.set_ylabel("Validation Accuracy", fontsize=15)
fig.tight_layout()
fig.savefig("./flextask/compare_dmpn_rnn.png", dpi=300)








