# %%
import os
import math
import time
import copy
import numpy as np
import gc 
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mpn 

class SequentialMNIST(torch.utils.data.Dataset):
    """
    """
    def __init__(self, root, train=True, download=True, normalize=True):
        tfms = [transforms.ToTensor()]
        self.ds = datasets.MNIST(root=root, train=train, download=download, transform=transforms.Compose(tfms))
        self.normalize = normalize

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]             
        x = x.view(-1)                  
        if self.normalize:
            x = (x - 0.1307) / 0.3081
        x_seq = x.unsqueeze(-1)        
        return x_seq, y


def collate_seq(batch):
    """
    """
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


# %%
def print_cuda_tensor_shapes(limit=None, sort_by_numel=True, include_nonleaf=True):
    """
    Prints shapes (and a bit more) for all live torch tensors on CUDA.

    Notes:
    - This lists tensors that are still referenced by Python (reachable by GC).
    - It may include duplicates (views). We de-duplicate by storage data_ptr.
    """
    cuda_tensors = []
    seen = set()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                t = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                # Parameters and some wrappers
                t = obj.data
            else:
                continue

            if t.is_cuda:
                # de-dup by underlying storage pointer (works for views)
                try:
                    key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.size()), str(t.dtype))
                except Exception:
                    key = (t.data_ptr(), tuple(t.size()), str(t.dtype))

                if key in seen:
                    continue
                seen.add(key)

                if (not include_nonleaf) and (t.grad_fn is not None):
                    continue

                cuda_tensors.append(t)
        except Exception:
            pass

    if sort_by_numel:
        cuda_tensors.sort(key=lambda x: x.numel(), reverse=True)

    if limit is not None:
        cuda_tensors = cuda_tensors[:limit]

    total_bytes = 0
    for i, t in enumerate(cuda_tensors, 1):
        nbytes = t.numel() * t.element_size()
        total_bytes += nbytes
        print(
            f"[{i:04d}] shape={tuple(t.shape)} dtype={t.dtype} "
            f"device={t.device} requires_grad={t.requires_grad} "
            f"bytes={nbytes/1024**2:.2f}MB"
        )

    print(f"\nCount: {len(cuda_tensors)} tensors")
    print(f"Estimated total (sum of listed tensor sizes): {total_bytes/1024**2:.2f}MB")


# %%
@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device) 
        y = y.to(device)

        B, T, D = x.shape
        net.reset_state(B=B)

        K = 32
        logits_tail = []
        
        for t in range(T):
            out, _, _ = net.network_step(x[:, t, :], run_mode="minimal", seq_idx=t)
            if t >= T - K:
                logits_tail.append(out)
        
        out_agg = torch.stack(logits_tail, dim=0).mean(dim=0)  
        pred = out_agg.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)

def count_parameter(net):
    """
    """
    trainable = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    
    # Print a readable summary
    total = 0
    for n, p in trainable:
        num = p.numel()
        total += num
        print(f"{n:50s}  shape={tuple(p.shape)}  numel={num}")
    
    print(f"\nTotal trainable parameters: {total}")

def train_sequential_mnist(
    device="cuda",
    data_root="./data",
    hidden_dim=256,
    batch_size=64,
    lr=1e-3,
    epochs=5,
    mpn_depth=5
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Data
    train_ds = SequentialMNIST(root=data_root, train=True, download=True, normalize=True)
    test_ds  = SequentialMNIST(root=data_root, train=False, download=True, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=True, collate_fn=collate_seq)
    test_loader  = DataLoader(test_ds, batch_size=batch_size*10, shuffle=False, num_workers=2,
                              pin_memory=True, collate_fn=collate_seq)

    net_params = {
        "n_neurons": [1] + [hidden_dim] * mpn_depth + [10],
        "linear_embed": 100, 
        "dt": 1.0,
        "activation": "tanh",            
        "output_bias": True,
        "W_output_init": "xavier",
        "input_layer_add": True, 
        'input_layer_add_trainable': True, 
        'input_layer_bias': False, 
        "output_matrix": "", 

        'ml_params': {
            'bias': True, # Bias of layer
            'mp_type': 'mult',
            'm_activation': 'linear',
            'm_update_type': 'hebb_assoc', # hebb_assoc, hebb_pre
            'eta_type': 'matrix', # scalar, pre_vector, post_vector, matrix
            'eta_train': True,
            'eta_init': 'gaussian', 
            'lam_type': 'matrix', # scalar, pre_vector, post_vector, matrix
            'm_time_scale': 1000, 
            'lam_train': True,
            'W_freeze': False, # different combination with [input_layer_add_trainable]
        },
    }

    net = mpn.DeepMultiPlasticNet(net_params, verbose=True, forzihan=False).to(device)
    
    count_parameter(net)

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    stats = {
        "test_acc": []
    }

    for ep in range(1, epochs + 1):
        net.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)  
            y = y.to(device, non_blocking=True)

            B, T, D = x.shape
            net.reset_state(B=B)

            opt.zero_grad(set_to_none=True)

            K = 32
            logits_tail = []
            
            for t in range(T):
                out, _, _ = net.network_step(x[:, t, :], run_mode="minimal", seq_idx=t)
                if t >= T - K:
                    logits_tail.append(out) 

            out_agg = torch.stack(logits_tail, dim=0).mean(dim=0)  
            loss = criterion(out_agg, y)
            loss.backward()

            if (n_batches % 100) == 0:
                mpl = net.mp_layers[0]
                with torch.no_grad():
                    M_abs_mean = mpl.M.abs().mean().item()
                    M_abs_max  = mpl.M.abs().max().item()
                    W_abs_mean = mpl.W.abs().mean().item()
                print(f"[ep {ep} | batch {n_batches}] loss={loss.item():.4f} "
                      f"|M| mean={M_abs_mean:.3e} max={M_abs_max:.3e} |W| mean={W_abs_mean:.3e}")

            opt.step()
            scheduler.step()
            net.param_clamp()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        test_acc = evaluate(net, test_loader, device=device)

        current_lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:02d}/{epochs} | lr={current_lr:.2e} | loss={train_loss:.4f} | test_acc={test_acc*100:.2f}%")

        stats["test_acc"].append(test_acc)

    return net, net_params, stats

# %%
trained_net, net_params, state_dict = train_sequential_mnist(
                device="cuda",
                hidden_dim=128,
                batch_size=64,
                lr=1e-3,
                epochs=30,
                mpn_depth=3)

# %%
ckpt_path = "./mnist/mpn_seqmnist_ckpt.pt"

checkpoint = {
    "model_class": "DeepMultiPlasticNet",
    "net_params": net_params,  # the same dict used to build the net
    "state_dict": trained_net.state_dict(),
}

torch.save(checkpoint, ckpt_path)
print(f"Saved checkpoint to: {ckpt_path}")

# %%
import seaborn as sns 
import matplotlib.pyplot as plt

# %%
W1, W2, W3 = trained_net.mp_layer1.W.cpu().detach(), trained_net.mp_layer2.W.cpu().detach(), trained_net.mp_layer3.W.cpu().detach()
Ws = [W1, W2, W3]
fig, axs = plt.subplots(1,3,figsize=(4*3,4))
for i in range(len(Ws)): 
    sns.heatmap(Ws[i], ax=axs[i], square=True, cmap="coolwarm", center=0, vmax=1, vmin=-1)
    axs[i].set_title(f"Mean: {torch.mean(Ws[i]):1f}")
fig.tight_layout()

# %%
W1, W2, W3 = trained_net.mp_layer1.eta.cpu().detach(), trained_net.mp_layer2.eta.cpu().detach(), trained_net.mp_layer3.eta.cpu().detach()
Ws = [W1, W2, W3]
fig, axs = plt.subplots(1,3,figsize=(4*3,4))
for i in range(len(Ws)): 
    sns.heatmap(Ws[i], ax=axs[i], square=True, cmap="coolwarm", center=0, vmax=2, vmin=-2)
    axs[i].set_title(f"Mean: {torch.mean(Ws[i]):1f}")
fig.tight_layout()

# %%
W1, W2, W3 = trained_net.mp_layer1.lam.cpu().detach(), trained_net.mp_layer2.lam.cpu().detach(), trained_net.mp_layer3.lam.cpu().detach()
Ws = [W1, W2, W3]
fig, axs = plt.subplots(1,3,figsize=(4*3,4))
for i in range(len(Ws)): 
    sns.heatmap(Ws[i], ax=axs[i], square=True, cmap="coolwarm", center=0.5, vmax=1, vmin=0)
    axs[i].set_title(f"Mean: {torch.mean(Ws[i]):1f}")
fig.tight_layout()

# %%



