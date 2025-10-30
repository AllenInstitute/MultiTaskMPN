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

path = "./onetask/"
keywords_recTrue = ["loss", "recTrue"]
results_recTrue = find_pkl_files_with_keywords(path, keywords_recTrue)

fig, ax = plt.subplots(1,1,figsize=(4,4))
for recTrue_file in results_recTrue: 
    data = np.load(recTrue_file, allow_pickle=True)
    ax.plot(data["batch_idx"], data["testing_acc"], c=c_vals[0])

ax.set_xlabel("# Batches", fontsize=15)
ax.set_ylabel("Validation Accuracy", fontsize=15)
fig.tight_layout()
fig.savefig("./onetask/compare_w_wo_rec.png", dpi=300)