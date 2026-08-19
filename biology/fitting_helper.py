import pandas as pd
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import seaborn as sns

def shift_synapse_table_by_cell_table(cell_table, synapse_table):
    """ 
    """
    syn_x, syn_y, syn_z = synapse_table["pt_position_x_trafo"].values, synapse_table["pt_position_y_trafo"].values, \
        synapse_table["pt_position_z_trafo"].values
    shift_x, shift_y, shift_z = np.max(cell_table["pt_position_x_trafo"].values), np.max(cell_table["pt_position_y_trafo"].values), \
        np.max(cell_table["pt_position_z_trafo"].values)
    synapse_table["pt_position_x_trafo"] += shift_x
    synapse_table["pt_position_y_trafo"] += shift_y
    synapse_table["pt_position_z_trafo"] += shift_z
    print(f"Translate All Synapses by: {shift_x}, {shift_y}, {shift_z}")

    return synapse_table, (shift_x, shift_y, shift_z)

def cell_synapse_table_reader(dataload_index, celltypechoice, pattern, version):
    """
    if pattern = "normal", as in most cases, gererate the cell table with anatomical filtering
    if pattern = "all", use all neurons in the V1 region (for microns) or all neurons (for v1dd)
    """
    print(f"======================= Start Cell & Synapse Table Reading =======================")
    print(
        f"dataload_index: {dataload_index}; celltypechoice: {celltypechoice}; pattern: {pattern}")

    assert pattern in ("normal", "all"), "Undefined pattern"

    filenames = {
        # after Sep 9th 2025
        "new": {
            "microns/": {
                "cell_table": "./microns_v1dd/sven/microns_cell_annos_CV_250904.feather",
                "synapse_table": "./microns_v1dd/sven/synapses_minnie65_phase3_v1_1507_combined_filtered_incl_trafo_250904.feather"
            },
        },
    }

    if dataload_index == "microns/":
        cell_table_name = filenames[version]["microns/"]["cell_table"]
        cell_table = pd.read_feather(cell_table_name)
        print(f"cell_table_name: {cell_table_name}")
        # print(cell_table["region"].value_counts())

        if pattern == "normal":
            cell_table = cell_table[
                ((cell_table["status_axon"] == "extended")
                 | (cell_table["full_dendrite"] == True))
                & (cell_table["region"] == "V1")
            ]
        elif pattern == "all":
            cell_table = cell_table[
                (cell_table["region"] == "V1")
            ]

    elif dataload_index == "v1dd/":
        cell_table_name = filenames[version]["v1dd/"]["cell_table"]
        cell_table = pd.read_feather(cell_table_name)
        print(f"cell_table_name: {cell_table_name}")

        if pattern == "normal":
            cell_table = cell_table[
                ((cell_table["status_axon"] == "extended")
                 | (cell_table["full_dendrite"] == True))
            ]
        elif pattern == "all":
            cell_table = cell_table

    # assert filtered_df.empty, "Dataframe has rows where only one column is -1."
    # only select neurons that have *both* axon and dendrite CVs values available
    # Mar 4th: however, based on the newest dataset Sven provides, many neurons have -1 in the CV values, regardless of the way of categorization
    # should we treat the pre & post differently?
    # Mar 7th: maybe not... confirmed with Stefan B -- still take the neurons with both CV values available
    # Mar 15th: consider CV inclusion axon and dendrite separately

    # dummy setting for all neurons fitting
    cell_table['all'] = 'all'

    # sort the cell_table based to cluster the cell type and cv choices (combination of pre&post)
    cell_table = cell_table.sort_values(
        by=[celltypechoice, f"{celltypechoice}_CV_axon", f"{celltypechoice}_CV_dendrite"])

    # print the cell type distribution in the cell table
    # will later cross compare with the category of precomputed mixture models
    # 2025-10-27: old_synpase_table is the raw synapse table directly from loading into, without any filtering
    if dataload_index == "microns/":
        synapse_table_name = filenames[version]["microns/"]["synapse_table"]
        old_synapse_table = pd.read_feather(synapse_table_name)
        print(f"synapse_table_name: {synapse_table_name}")
    elif dataload_index == "v1dd/":
        synapse_table_name = filenames[version]["v1dd/"]["synapse_table"]
        old_synapse_table = pd.read_feather(synapse_table_name)
        print(f"synapse_table_name: {synapse_table_name}")

    # shift the synapses altogether based on the cell table information
    old_synapse_table, shift_vector = shift_synapse_table_by_cell_table(
        cell_table, old_synapse_table)

    pt_root_ids = set(cell_table["pt_root_id"])

    # make sure the synapse table only contains synapses formed by the neurons in the cell table
    # namely either pre or post pt_root_id is in the cell table
    # 2025-10-27: this should not change ANY functionality in the calculations;
    # only save memory and speed up the process
    synapse_table = old_synapse_table[
        old_synapse_table["pre_pt_root_id"].isin(pt_root_ids) |
        old_synapse_table["post_pt_root_id"].isin(pt_root_ids)
    ]

    cell_table_cp = copy.deepcopy(cell_table)
    cell_table_cp = cell_table_cp.reset_index()

    # define the boolean masks
    # for pre/post cells, we use axon/dendrite status + CV values
    if pattern == "normal":
        pre_mask = (cell_table["status_axon"] == "extended") & cell_table[f"{celltypechoice}_CV_axon"].isin(
            [0, 1, 2, 3, 4])
        post_mask = (cell_table["full_dendrite"] ==
                     True) & cell_table[f"{celltypechoice}_CV_dendrite"].isin([0, 1, 2, 3, 4])
    elif pattern == "all":
        pre_mask = np.ones(len(cell_table), dtype=bool)
        post_mask = np.ones(len(cell_table), dtype=bool)

    print(f"pre_mask: {pre_mask.sum()}; post_mask: {post_mask.sum()}")

    # Get indices from cell_table_cp using the same masks
    indices_row = cell_table_cp.index[pre_mask].tolist()
    indices_column = cell_table_cp.index[post_mask].tolist()

    pre_type = cell_table[pre_mask][celltypechoice].value_counts()
    post_type = cell_table[post_mask][celltypechoice].value_counts()

    # Get the 'pt_root_id' values from cell_table using the masks
    # 2025-10-27: pre/post_cells_index are the indices of ones that are respectively structurally proofreaded
    # because both selections are order-preserving projections of the same base DataFrame, 
    # their induced ordering is identical.
    pre_cells_index = cell_table.loc[pre_mask, 'pt_root_id'].tolist()
    post_cells_index = cell_table.loc[post_mask, 'pt_root_id'].tolist()

    return cell_table, synapse_table, indices_row, indices_column, pre_cells_index, post_cells_index, old_synapse_table, [pre_type, post_type], shift_vector

def load_prediction_vs_groundtruth(dataload_index, categorization, pattern, 
                                   dataset_version, prediction_path, verbose=True):
    """
    """
    cell_table, synapse_table, indices_row, indices_column, pre_cells_index, post_cells_index, _, [pre_type, post_type], _ \
        = cell_synapse_table_reader(dataload_index, categorization, pattern=pattern, version=dataset_version)
                    
    with open(prediction_path, "rb") as file:
        predictions = pickle.load(file)
        
    binary_prediction = predictions["matrices"][0]
    # unpack from previous calculation (will redo from scratch later)
    ground_truth = predictions["ground_truth"]
    assert binary_prediction.shape == (len(pre_cells_index), len(post_cells_index)), \
        "Prediction matrix shape does not match the expected shape based on filtered cell table."
        
    rng = np.random.default_rng(seed=42) 
    n_pre = len(pre_cells_index)
    n_post = len(post_cells_index)
        
    if verbose:
        for _ in range(100):
            # randomly sample pre and post indices
            i_pre = rng.integers(0, n_pre)
            i_post = rng.integers(0, n_post)

            # pt_root_id for pre- and post-synaptic cells
            pre_cell_id = pre_cells_index[i_pre]
            post_cell_id = post_cells_index[i_post]

            pre_cat = cell_table.loc[
                cell_table["pt_root_id"] == pre_cell_id, categorization
            ].values[0]

            post_cat = cell_table.loc[
                cell_table["pt_root_id"] == post_cell_id, categorization
            ].values[0]

            # prediction value
            pred_value = binary_prediction[i_pre, i_post]

            # ground truth value
            gt = synapse_table[
                (synapse_table["pre_pt_root_id"] == pre_cell_id) &
                (synapse_table["post_pt_root_id"] == post_cell_id)
            ]
            gt_value = int(not gt.empty)

            print(
                f"Pre[{i_pre}] {pre_cell_id}-{pre_cat} | "
                f"Post[{i_post}] {post_cell_id}-{post_cat} | "
                f"Prediction: {pred_value:.2f} | "
                f"Ground Truth: {gt_value}"
            )
    
    # visualize the prediction vs ground truth as heatmaps     
    fig, axs = plt.subplots(1, 2, figsize=(10*2, 10))
    sns.heatmap(binary_prediction, ax=axs[0], cmap="viridis")
    axs[0].set_title("Predicted Connectivity Matrix", fontsize=16)
    sns.heatmap(ground_truth, ax=axs[1], cmap="viridis")
    axs[1].set_title("Ground Truth Connectivity Matrix", fontsize=16)
    fig.tight_layout()
    fig.savefig("zz_prediction_vs_groundtruth_heatmap.png", dpi=300)
      