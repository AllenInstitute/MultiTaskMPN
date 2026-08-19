import numpy as np 
import pandas as pd
import xarray as xr
import time 

session_info = 4
scan_info = 7

coreg = pd.read_csv("./coregistration_manual_v4_1507.csv")
celltable = pd.read_feather("./microns_cell_annos_CV_250904.feather")
activity_name = f"./functional_xr/functional_session_{session_info}_scan_{scan_info}.nc"
session_ds = xr.open_dataset(activity_name)

# access the specific session & scan
coreg_select = coreg[(coreg["session"] == session_info) & (coreg["scan_idx"] == scan_info)]
unit_ids = coreg_select["unit_id"].values.tolist()
fields = coreg_select["field"].values.tolist()

print(f"Session {session_info} Scan {scan_info} Coregistered Neurons: {len(coreg_select)}")

# check overlap
coreg_select_pt_root_id = coreg_select["pt_root_id"].values.tolist()
celltable_pt_root_id = celltable["pt_root_id"].values.tolist()
overlap_count = sum(pt_root_id in celltable_pt_root_id for pt_root_id in coreg_select_pt_root_id)
overlap_pt_root_id = [pt_root_id for pt_root_id in coreg_select_pt_root_id if pt_root_id in celltable_pt_root_id]
print(f"Overlap with celltable: {overlap_count} / {len(coreg_select_pt_root_id)} = {overlap_count / len(coreg_select_pt_root_id):.2f}")

# for overlapped neuron, check proofread status
axon_pf_count, dendrite_pf_count, both_pf_count = 0, 0, 0
for pt_root_id in overlap_pt_root_id:
    row = celltable[celltable["pt_root_id"] == pt_root_id].iloc[0]
    status_axon = row["status_axon"]
    status_dendrite = row["full_dendrite"]
    axon_pf_count += int(status_axon == "extended")
    dendrite_pf_count += int(status_dendrite == True)
    both_pf_count += int(status_axon == "extended" and status_dendrite == True)
    
print(f"Proofread Axon: {axon_pf_count} / {overlap_count} = {axon_pf_count / overlap_count:.2f}")
print(f"Proofread Dendrite: {dendrite_pf_count} / {overlap_count} = {dendrite_pf_count / overlap_count:.2f}")
print(f"Proofread Both: {both_pf_count} / {overlap_count} = {both_pf_count / overlap_count:.2f}")

# access to activity data 
for row in coreg_select.itertuples():
    # only for neurons that are also in cell table, i.e. we know its structural informatio
    if row.pt_root_id in overlap_pt_root_id: 
        unit_id, field = row.unit_id, row.field
        check_unitid = session_ds["unit_id"].values[unit_id-1]
        check_field = session_ds["field"].values[unit_id-1]
        # sanity check
        assert max(unit_ids) <= np.max(session_ds["unit_id"].values)
        assert max(fields) <= np.max(session_ds["field"].values)
        assert unit_id == check_unitid
        assert field == check_field
        # that is the activity of the coregistered neuron
        activity_neuron = session_ds["activity"].values[unit_id-1,:]
