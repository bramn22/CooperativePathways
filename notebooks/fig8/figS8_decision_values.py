#%%
import os
os.chdir('..')

import numpy as np
import seaborn as sb
import notebooks.CoordinatedCellTypes.fig3.fig3_preprocessing as fig3_preprocessing
import notebooks.CoordinatedCellTypes.fig3.fig3_encoding_unified as fig3_encoding_unified

#%%
cwdir = os.getcwd()
print(cwdir)
os.chdir('../..')

#%% Figure parameters and path
# reset path
from notebooks.CoordinatedCellTypes.final.fig_params import *
fig_path = r''

#%%
tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)
vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
stim_types = ['EXPA', 'EXPA_60', 'EXPA_25', 'EXPW', 'EXPW_60','DIMM']
stim_types_plotting = [0,1,2,3,4,5]
stim_types_labels = ['black loom 100%','black loom 60%', 'black loom 25%', 'white loom 100%', 'white loom 60%', 'dimming']
stim_types_colors = [vlag_cmap[-1], vlag_cmap[-3], vlag_cmap[-4], vlag_cmap[0], vlag_cmap[2], tab_cmap[5]]


#%% Load data (only visually responsive cells)
from notebooks.CoordinatedCellTypes.final.datajoint_utils import get_cluster_frs

conditions = ['sSC', 'dSC', 'LP']

exp_dict = {}
for cond in conditions:
    print(cond)
    frs_dict = get_cluster_frs([], stim_types, dataset_name=f'{cond}_VR_good_fr25_20260116', means=False, subsession_id='01', fix_sweeping=False)
    frs, depths, keys_used = frs_dict['cluster_frs'], frs_dict['cluster_depths'], frs_dict['cluster_keys']


sSC_frs, sSC_depths, sSC_keys_used = exp_dict['sSC']
dSC_frs, dSC_depths, dSC_keys_used = exp_dict['dSC']
LP_frs, LP_depths, LP_keys_used = exp_dict['LP']

#%% --- Save model decision values for fig S8
stim_types_plotting = [0,1,2,3,4,5]
time_win = [205, 215]
base_win = [50, 350] # Adjust if your baseline window indices differ
dec_classes = [[0], [2, 3, 5]]

sSC_frs_mats, sSC_frs_trials_mats, sSC_keys_mats, _, _ = fig3_preprocessing.preprocess(sSC_frs, keys_used=sSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
dSC_frs_mats, dSC_frs_trials_mats, dSC_keys_mats, _, _ = fig3_preprocessing.preprocess(dSC_frs, keys_used=dSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
LP_frs_mats, LP_frs_trials_mats, LP_keys_mats, _, _ = fig3_preprocessing.preprocess(LP_frs, keys_used=LP_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)

dSC_data = fig3_preprocessing.unflatten_frs_mat(np.concatenate(dSC_frs_trials_mats, axis=0), n_stim=6, n_trials=8, n_timepoints=800)
# Generate the 1D arrays for dSC
dSC_dv_dict = fig3_encoding_unified.generate_model_decision_variables(
    data=dSC_data, 
    target_groups=dec_classes, 
    time_window=time_win, 
    baseline_window=base_win,
    subsample_n=30,
    k_dim=10, 
    use_pca=False, 
    # modulation_strengths=[0.0, 0.5, 1.0, 1.5, 2]
    modulation_strengths=np.arange(0, 4.1, step=0.1)
    # n_pseudo=100, 
    # n_bootstraps=10000
)

LP_data = fig3_preprocessing.unflatten_frs_mat(np.concatenate(LP_frs_trials_mats, axis=0), n_stim=6, n_trials=8, n_timepoints=800)
# Generate the 1D arrays for LP
LP_dv_dict = fig3_encoding_unified.generate_model_decision_variables(
    data=LP_data, 
    target_groups=dec_classes, 
    time_window=time_win, 
    baseline_window=base_win,
    subsample_n=30,
    k_dim=10, 
    use_pca=False, 
    # modulation_strengths=[0.0, 0.5, 1.0, 1.5, 2]
    modulation_strengths=np.arange(0, 4.1, step=0.1)
    # n_pseudo=100, 
    # n_bootstraps=10000
)

#%%
dSC_dv_dict
dSC_EXPA, dSC_EXPA25, dSC_EXPW, dSC_DIMM, dSC_Base = dSC_dv_dict['stim_0'], dSC_dv_dict['stim_2'],dSC_dv_dict['stim_3'], dSC_dv_dict['stim_5'], dSC_dv_dict['base_0']
LP_EXPA, LP_EXPA25, LP_EXPW, LP_DIMM, LP_Base = LP_dv_dict['stim_0'], LP_dv_dict['stim_2'],LP_dv_dict['stim_3'], LP_dv_dict['stim_5'], LP_dv_dict['base_0']
dSC_EXPA, dSC_EXPA25, dSC_EXPW, dSC_DIMM, dSC_Base = -1*dSC_EXPA, -1*dSC_EXPA25, -1*dSC_EXPW, -1*dSC_DIMM, -1*dSC_Base
LP_EXPA, LP_EXPA25, LP_EXPW, LP_DIMM, LP_Base = -1*LP_EXPA, -1*LP_EXPA25, -1*LP_EXPW, -1*LP_DIMM, -1*LP_Base

#%%
np.mean(dSC_EXPA, axis=-1), np.mean(LP_EXPA, axis=-1)
#%%
import pickle

data_to_save = {
    # dSC variables
    "mod_strengths": np.arange(0, 4.1, step=0.1),
    "dSC_data": dSC_data,
    "dSC_EXPA": dSC_EXPA,
    "dSC_EXPA25": dSC_EXPA25,
    "dSC_EXPW": dSC_EXPW,
    "dSC_DIMM": dSC_DIMM,
    "dSC_Base": dSC_Base,
    
    # LP variables
    "LP_data": LP_data,
    "LP_EXPA": LP_EXPA,
    "LP_EXPA25": LP_EXPA25,
    "LP_EXPW": LP_EXPW,
    "LP_DIMM": LP_DIMM,
    "LP_Base": LP_Base,
}

# Save the dictionary to a pickle file
filename = "model_decision_variables_30.pkl"
with open(filename, "wb") as f:
    pickle.dump(data_to_save, f)

print(f"Variables successfully saved to {filename}")
