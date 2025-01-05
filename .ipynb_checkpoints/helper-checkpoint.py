import tasks
import matplotlib.pyplot as plt
import numpy as np 
from scipy.linalg import null_space

import time

def to_unit_vector(arr):
    """Convert a vector to a unit vector."""
    norm = np.linalg.norm(arr)
    return arr / norm if norm != 0 else arr

def participation_ratio_vector(C):
    """Computes the participation ratio of a vector of variances."""
    return np.sum(C) ** 2 / np.sum(C*C)

def plot_some_ouputs(params, net, mode_for_all="random_batch", n_outputs=4, nameadd="", verbose=False):
    """
    """
    # in case out of range
    c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',] * 10
    c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',] * 10
    c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',] * 10

    task_params, train_params, net_params = params

    if task_params['task_type'] in ('multitask',):
        test_data, test_trials_extra = tasks.generate_trials_wrap(task_params, n_outputs, mode_input=mode_for_all)
        # figure out sessions
        # assert mode_for_all == "random_batch" # lazy...
        _, test_trials, test_rule_idxs = test_trials_extra
        recordkyle_all = []
        for test_subtrial in test_trials:
            metaepoch = test_subtrial.epochs
            # print(metaepoch)
            periodname = list(metaepoch.keys())
            recordkyle = []
            for keyiter in range(len(periodname)):
                recordkyle.append(list(metaepoch[periodname[keyiter]][1]))
            recordkyle.insert(0, [0 for _ in range(len(recordkyle[1]))])
            recordkyle = np.array(recordkyle).T.tolist()
            # print(recordkyle)
            recordkyle_all.extend(recordkyle)
                    
    elif task_params['task_type'] in ('NeuroGym',):
        test_data, _ = convert_ngym_dataset(
                    task_params, set_size=n_outputs, device=device,
                    mask_type=tasks_masks[task_params['dataset_name']]
            )

    test_input, test_output, test_masks = test_data

    net_out, db = net.iterate_sequence_batch(test_input, run_mode='track_states')

    net_out = net_out.detach().cpu().numpy()
    test_input_np = test_input.cpu().numpy()
    test_out_np = test_output.cpu().numpy()

    test_masks_np = test_masks.cpu().numpy()

    fig, axs = plt.subplots(n_outputs, 2, figsize=(8*2, 2*n_outputs))

    if task_params['task_type'] in ('NeuroGym',):
        for batch_idx, ax in enumerate(axs[:,0]):
            ax.plot(net_out[batch_idx, :, 0], color=c_vals[batch_idx])
            ax.plot(test_out_np[batch_idx, :, 0], color=c_vals_l[batch_idx], zorder=-1)

            ax.set_ylim((0, 2.25))
    elif task_params['task_type'] in ('multitask',):

        masks = (test_masks_np > 0.).astype(np.int32) # Hides output where not relevant

        for batch_idx, ax in enumerate(axs[:,0]):
            for out_idx in range(test_out_np.shape[-1]):
                ax.plot(net_out[batch_idx, :, out_idx] * masks[batch_idx, :, out_idx], color=c_vals[out_idx])
                ax.plot(test_out_np[batch_idx, :, out_idx], color=c_vals_l[out_idx], zorder=-1)

            ax.set_ylim((-1.125, 1.125))

        # ADD BY ZIHAN
        for batch_idx, ax in enumerate(axs[:,1]):
            for out_idx in range(test_input_np.shape[-1]):
                ax.plot(test_input_np[batch_idx, :, out_idx], color=c_vals_l[out_idx], zorder=-1, label=out_idx)

            ax.set_ylim((-1.125, 1.125))
            ax.legend()

        # ZIHAN: add session cutoff
        for axind in range(2):
            for outputind in range(n_outputs):
                session_period = recordkyle_all[outputind]
                for session_break in session_period:
                    axs[outputind, axind].axvline(session_break, linestyle="--")

    if verbose:
        fig.savefig(f"./results/results_{nameadd}.png")

def find_consecutive_zero_indices(arr):
    zero_indices = []
    start_index = None

    for i, row in enumerate(arr):
        if np.all(row == 0):
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                zero_indices.append(list(range(start_index, i)))
                start_index = None

    # Handle case where the last rows are all zeros
    if start_index is not None:
        zero_indices.append(list(range(start_index, len(arr))))

    return zero_indices

def magnitude_of_projection(v, basis_vectors):
    """
    Calculate the magnitude of the projection of vector v onto the subspace spanned by basis_vectors.
    """
    v = np.array(v).flatten() 
    basis_vectors = np.array(basis_vectors)

    if basis_vectors.shape[0] == 1:
        # 1D subspace
        u = basis_vectors[0]
        norm_u = np.linalg.norm(u)
        projection_scalar = np.dot(v, u) / norm_u**2
        projection_vector = projection_scalar * u
        magnitude = np.linalg.norm(projection_vector)
    elif basis_vectors.shape[0] == 2:
        # 2D subspace
        U = basis_vectors.T  # (N, 2)
        P = U @ np.linalg.inv(U.T @ U) @ U.T  # Projection matrix
        projection_vector = P @ v
        magnitude = np.linalg.norm(projection_vector)
    else:
        raise ValueError("Only 1D or 2D subspaces are supported.")
    
    return magnitude

if __name__ == "__main__":
    vector1 = np.random.rand(1, 100)
    vector2 = np.random.rand(1, 100)

    vector1_normalized = vector1 / np.linalg.norm(vector1)
    vector2_normalized = vector2 / np.linalg.norm(vector2)
    vectors = np.vstack((vector1_normalized, vector2_normalized))

    alpha = np.random.rand()
    beta = np.random.rand()
    vector_in_plane = alpha * vector1_normalized + beta * vector2_normalized
    vector_in_plane_normalized = vector_in_plane / np.linalg.norm(vector_in_plane)

    magnitude = magnitude_of_projection(vector_in_plane_normalized, vectors)
    print(magnitude)
