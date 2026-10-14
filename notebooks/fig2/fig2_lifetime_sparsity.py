#%%
import os
import pandas as pd
import datajoint as dj
# from datetime import datenid
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import kruskal, mannwhitneyu

#%% Figure parameters and path
from fig_params import *
fig_path = r''


#%% Load data (only visually responsive cells)
from datajoint_utils import get_cluster_frs

stim_types = ['EXPA', 'EXPA_60', 'EXPA_25', 'EXPW', 'EXPW_60','DIMM']

sSC_dict = get_cluster_frs([], stim_types, dataset_name='sSC_VR_good_fr25_20260116', means=False, subsession_id='01')
dSC_dict = get_cluster_frs([], stim_types, dataset_name='dSC_VR_good_fr25_20260116', means=False, subsession_id='01')
LP_dict = get_cluster_frs([], stim_types, dataset_name='LP_VR_good_fr25_20260116', means=False, subsession_id='01')

sSC_frs, sSC_depths, sSC_keys_used = sSC_dict['cluster_frs'], sSC_dict['cluster_depths'], sSC_dict['cluster_keys']
dSC_frs, dSC_depths, dSC_keys_used = dSC_dict['cluster_frs'], dSC_dict['cluster_depths'], dSC_dict['cluster_keys']
LP_frs, LP_depths, LP_keys_used = LP_dict['cluster_frs'], LP_dict['cluster_depths'], LP_dict['cluster_keys']
sum([len(v) for k,v in sSC_frs.items()]), sum([len(v) for k,v in dSC_frs.items()]), sum([len(v) for k,v in LP_frs.items()])
    
#%% Calculate trial means and baseline subtraction
for key in sSC_frs:
    sSC_frs[key] = np.nanmean(sSC_frs[key][:,:,:], axis=2)
    # baseline subtraction
    sSC_frs[key] = sSC_frs[key] - np.nanmean(sSC_frs[key][:,:,50:350], axis=2, keepdims=True)
for key in dSC_frs:
    dSC_frs[key] = np.nanmean(dSC_frs[key][:,:,:], axis=2)
    dSC_frs[key] = dSC_frs[key] - np.nanmean(dSC_frs[key][:,:,50:350], axis=2, keepdims=True)
for key in LP_frs:
    LP_frs[key] = np.nanmean(LP_frs[key][:,:,:], axis=2)
    LP_frs[key] = LP_frs[key] - np.nanmean(LP_frs[key][:,:,50:350], axis=2, keepdims=True)


#%% Lifetime sparsity using adapted Treves-Rolls Sparsity Index
def treves_rolls_sparsity(responses):
    print(responses.shape)
    n = len(responses)
    if n == 0:
        return np.nan
    sum_r = np.sum(responses)/n
    sum_r2 = np.sum(responses**2)/n
    if sum_r == 0:
        return np.nan
    S = (1-((sum_r**2) / sum_r2)) / (1 - (1/n))
    # S = ((sum_r**2) / sum_r2) 
    return S

def get_response_mat(frs):
    # Get peak (or mean) fr during stimulus period only
    n_cells = frs.shape[0]
    n_stimuli = frs.shape[1]
    response_mat = np.zeros((n_cells, n_stimuli))
    for cell_idx in range(n_cells):
        for stim_idx in range(n_stimuli):
            peak_response = np.percentile(np.abs(frs[cell_idx,stim_idx,400:400+100]), 98)
            # peak_response = np.mean(np.abs(frs[cell_idx,stim_idx,400:400+100]))
            response_mat[cell_idx, stim_idx] = peak_response
    return response_mat

def get_mean_sparsity(frs_dict, mean=True):
    sparsity_dict = {}
    for key in frs_dict:
        # exclude experiments with less than 5 cells
        if frs_dict[key].shape[0] < 5:
            continue
        resp_mat = get_response_mat(frs_dict[key][:,stim_indices,:]) # n_cells x n_stimuli
        sparsity = np.apply_along_axis(treves_rolls_sparsity, 1, resp_mat)
        if mean:
            sparsity_dict[key] = np.nanmean(sparsity)
        else:
            sparsity_dict[key] = sparsity
    return sparsity_dict

# stim_indices = [0,2,3,5] # EXPA, EXPA_25, EXPW, DIMM
stim_indices = [0,1,2,3,4,5] # EXPA, EXPA_25, EXPW, DIMM

sSC_sparsity_dict = get_mean_sparsity(sSC_frs, mean=False)
dSC_sparsity_dict = get_mean_sparsity(dSC_frs, mean=False)
LP_sparsity_dict = get_mean_sparsity(LP_frs, mean=False)

sSC_sparsity_concat = np.concatenate([sp for _, sp in sSC_sparsity_dict.items()]) # Shape: (n_cells, n_stimuli, n_timepoints) (eg 273, 11, 2000)
dSC_sparsity_concat = np.concatenate([sp for _, sp in dSC_sparsity_dict.items()])
LP_sparsity_concat = np.concatenate([sp for _, sp in LP_sparsity_dict.items()])

sSC_sparsity_means = ([np.nanmean(sp) for _, sp in sSC_sparsity_dict.items()]) # Shape: (n_cells, n_stimuli, n_timepoints) (eg 273, 11, 2000)
dSC_sparsity_means = ([np.nanmean(sp) for _, sp in dSC_sparsity_dict.items()])
LP_sparsity_means = ([np.nanmean(sp) for _, sp in LP_sparsity_dict.items()])

print("sSC mean sparsity per experiment:", sSC_sparsity_means)
print("dSC mean sparsity per experiment:", dSC_sparsity_means)
print("LP mean sparsity per experiment:", LP_sparsity_means)

#%%
fig, ax = plt.subplots(1, 1, figsize=(3*cm, 3*cm))
data_resp = [sSC_sparsity_means, dSC_sparsity_means, LP_sparsity_means]

labels = ['sSC', 'dSC', 'LP']

shift = 0.13

# Plot pointplot (median + IQR)
for i, d in enumerate(data_resp):
    median = np.median(d)
    q1, q3 = np.percentile(d, [25, 75])
    ax.errorbar(i + shift, median, yerr=[[median-q1], [q3-median]], fmt='o', color=colors[i], 
                elinewidth=1, capsize=2, markersize=4, zorder=2)

# Plot jittered scatter (stripplot) to the right of the boxplots
for i, d in enumerate(data_resp):
    # x = np.random.normal(i - shift, 0.04, size=len(d))
    x = np.random.uniform(i - shift-0.08, i - shift+0.08, size=len(d))
    ax.scatter(x, d, color=colors[i], s=1, alpha=0.2, zorder=1)

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel('Sparsity Index')
# ax.set_ylim([0, 1.1])
sb.despine()
# plt.savefig(os.path.join(fig_path, 'lifetime_sparsity_means_raincloud_simple.pdf'), bbox_inches='tight', dpi=300)
plt.show()

# statistical test
mw_res_sSC_dSC = mannwhitneyu(sSC_sparsity_means, dSC_sparsity_means)
mw_res_sSC_LP = mannwhitneyu(sSC_sparsity_means, LP_sparsity_means)
mw_res_dSC_LP = mannwhitneyu(dSC_sparsity_means, LP_sparsity_means)
print("Mann-Whitney U test result:", mw_res_sSC_dSC,mw_res_sSC_LP, mw_res_dSC_LP)
# Save stats test results to txt file
with open(os.path.join(fig_path, 'lifetime_sparsity_stats.txt'), 'w') as f:
    f.write(f"MWU (sSC vs dSC): {mw_res_sSC_dSC}\n")
    f.write(f"MWU (sSC vs LP): {mw_res_sSC_LP}\n")
    f.write(f"MWU (dSC vs LP): {mw_res_dSC_LP}\n")
    # N's
    f.write(f"N (sSC): {len(sSC_sparsity_means)}\n")
    f.write(f"N (dSC): {len(dSC_sparsity_means)}\n")
    f.write(f"N (LP): {len(LP_sparsity_means)}\n")