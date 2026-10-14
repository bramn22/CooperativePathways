#%%
import os
import pandas as pd
import datajoint as dj
# from datetime import datenid
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


#%%
import os
os.chdir('../..')
import main

drive_path = r"V:\Data\datajoint\DJEphys"
data_path = r"T:\Data"

ft = main.load(drive_path, data_path)


#%%
exp_table = ft.Experiment()
sess_table = ft.Session()
mouse_table = ft.Mouse()
subsess_table = ft.Subsession()
raw_table = ft.ephys_tables.EphysRaw()
raw_helper_table = ft.ephys_tables.EphysRawHelper()
events_table = ft.event_tables.SubsessionEvents()
sess_split_table = ft.SessionSplit() 
sorted_table = ft.ephys_tables.SpikeSorted()
metrics_table = ft.ephys_tables.ClusterMetrics()
subsess_spikes_table = ft.ephys_tables.SubsessionSpikes()
trials_table = ft.ephys_tables.TrialSpikes()
opto_table = ft.ephys_tables.OptoTagging()
opto_cluster_table = ft.ephys_tables.OptoCluster()
cluster_table = ft.ephys_tables.Cluster()
opto_clustering_table = ft.ephys_tables.OptoClusterClustering()
cluster_trials_table = ft.ephys_tables.ClusterTrial()
behavior_table = ft.behavior_tables.Behavioral()
behavior_trials_table = ft.behavior_tables.TrialBehavioral()
cluster_sweep_metrics_table = ft.ephys_analysis_tables.ClusterSweepMetrics()
cluster_loom_metrics_table = ft.ephys_analysis_tables.ClusterLoomMetrics()
cluster_depth_table = ft.ephys_tables.ClusterDepth()
cluster_trial_sweeping_table = ft.ephys_tables.TrialSweeping()
cluster_trials2_table = ft.ephys_tables.ClusterTrial2()


#%% Figure parameters and path
from fig_params import *
fig_path = r''


#%% Load data (only visually responsive cells)
from datajoint_utils import get_cluster_frs

stim_types = ['EXPA', 'EXPA_60', 'EXPA_25', 'EXPW', 'EXPW_60','DIMM']

sSC_dict = get_cluster_frs([], stim_types, dataset_name='sSC_good_fr25_20260116', means=False, subsession_id='01')
dSC_dict = get_cluster_frs([], stim_types, dataset_name='dSC_good_fr25_20260116', means=False, subsession_id='01')
LP_dict = get_cluster_frs([], stim_types, dataset_name='LP_good_fr25_20260116', means=False, subsession_id='01')

sSC_frs, sSC_depths, sSC_keys_used = sSC_dict['cluster_frs'], sSC_dict['cluster_depths'], sSC_dict['cluster_keys']
dSC_frs, dSC_depths, dSC_keys_used = dSC_dict['cluster_frs'], dSC_dict['cluster_depths'], dSC_dict['cluster_keys']
LP_frs, LP_depths, LP_keys_used = LP_dict['cluster_frs'], LP_dict['cluster_depths'], LP_dict['cluster_keys']
sum([len(v) for k,v in sSC_frs.items()]), sum([len(v) for k,v in dSC_frs.items()]), sum([len(v) for k,v in LP_frs.items()])
    
#%% Calculate trial means and baseline subtraction
for key in sSC_frs:
    sSC_frs[key] = np.nanmean(sSC_frs[key][:,:,:], axis=2)
    sSC_frs[key] = sSC_frs[key] - np.nanmean(sSC_frs[key][:,:,50:350], axis=2, keepdims=True) # baseline subtraction
for key in dSC_frs:
    dSC_frs[key] = np.nanmean(dSC_frs[key][:,:,:], axis=2)
    dSC_frs[key] = dSC_frs[key] - np.nanmean(dSC_frs[key][:,:,50:350], axis=2, keepdims=True)
for key in LP_frs:
    LP_frs[key] = np.nanmean(LP_frs[key][:,:,:], axis=2)
    LP_frs[key] = LP_frs[key] - np.nanmean(LP_frs[key][:,:,50:350], axis=2, keepdims=True)


#%% Lifetime sparsity
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


#%% Responsiveness across stimuli
stat_stims = ['EXPW', 'EXPW_60', 'EXPA_25', 'EXPA_60', 'EXPA', 'DIMM']
# flatten keys_used dictionaries
sSC_keys_used_flat = [key for key_list in sSC_keys_used.values() for key in key_list]
dSC_keys_used_flat = [key for key_list in dSC_keys_used.values() for key in key_list]
LP_keys_used_flat = [key for key_list in LP_keys_used.values() for key in key_list]

# Get counts of clusters being responsive to each stimulus type
def get_zeta_p_counts(stim_types, keys_used_flat):
    counts = np.zeros((len(stim_types), len(keys_used_flat)))
    for i, stim_type in enumerate(stim_types):
        if 'SWEP' in stim_type:
            zeta_ps = (cluster_sweep_metrics_table & keys_used_flat & {'stim_type': stim_type, 'subsession_id': '01'}).fetch('sm_zeta_p')
        else:
            zeta_ps = (cluster_loom_metrics_table & keys_used_flat & {'stim_type': stim_type, 'subsession_id': '01'}).fetch('lm_zeta_p')
        
        counts[i,zeta_ps<0.05/len(stat_stims)] = 1
    return counts

sSC_counts = get_zeta_p_counts(stat_stims, sSC_keys_used_flat)
dSC_counts = get_zeta_p_counts(stat_stims, dSC_keys_used_flat)
LP_counts = get_zeta_p_counts(stat_stims, LP_keys_used_flat)

# Plot counts of clusters being responsive to each stimulus type
# Calculate the percentage of clusters that are responsive to each stimulus type
sSC_counts_perc = np.sum(sSC_counts, axis=1)/sSC_counts.shape[1]*100
dSC_counts_perc = np.sum(dSC_counts, axis=1)/dSC_counts.shape[1]*100
LP_counts_perc = np.sum(LP_counts, axis=1)/LP_counts.shape[1]*100
# Make it a pandas dataframe
counts_df = pd.DataFrame({'sSC': sSC_counts_perc, 'dSC': dSC_counts_perc, 'LP': LP_counts_perc}, index=stat_stims)

fig, axs = plt.subplots(3,1, figsize=[3*cm,5*cm])
counts_df['sSC'].loc[stat_stims].plot(kind='bar', ax=axs[0], color='k', width=0.8)
axs[0].set(ylabel='sSC', xlabel='', yticks=[0, 20, 40, 60], yticklabels=['0%', '20%', '40%', '60%'], xticks=[])

counts_df['dSC'].loc[stat_stims].plot(kind='bar', ax=axs[1], color='k', width=0.8)
axs[1].set(ylabel='dSC', xlabel='', yticks=[0, 10, 20], yticklabels=['0%', '10%', '20%'], xticks=[])

counts_df['LP'].loc[stat_stims].plot(kind='bar', ax=axs[2], color='k', width=0.8)
axs[2].set(ylabel='LP', xlabel='', yticks=[0, 5, 10], yticklabels=['0%', '5%', '10%'])

plt.xticks(rotation=45)
plt.savefig(os.path.join(fig_path, 'responsive_clusters_stat_stims.pdf'), bbox_inches='tight', dpi=300)

# plt.tight_layout()
plt.show()
# %% Proportion of responsive cells
sSC_resp_probs = []
for key in sSC_keys_used:
    # exclude experiments with less than 10 cells
    if len(sSC_keys_used[key]) < 5:
        continue
    # Get proportion of responsive cells to any stimulus by first setting each cell to 1 if responsive, 0 otherwise
    zeta_p_counts = get_zeta_p_counts(stat_stims, sSC_keys_used[key]) # stimuli x cells
    resp_cells_vec = np.sum(zeta_p_counts, 0) > 0
    sSC_resp_probs.append(np.sum(resp_cells_vec)/resp_cells_vec.shape[0])
dSC_resp_probs = []
for key in dSC_keys_used:
    # exclude experiments with less than 10 cells
    if len(dSC_keys_used[key]) < 5:
        continue
    zeta_p_counts = get_zeta_p_counts(stat_stims, dSC_keys_used[key]) # stimuli x cells
    resp_cells_vec = np.sum(zeta_p_counts, 0) > 0
    dSC_resp_probs.append(np.sum(resp_cells_vec)/resp_cells_vec.shape[0])
LP_resp_probs = []
for key in LP_keys_used:
    # exclude experiments with less than 10 cells
    if len(LP_keys_used[key]) < 5:
        continue
    zeta_p_counts = get_zeta_p_counts(stat_stims, LP_keys_used[key]) # stimuli x cells
    resp_cells_vec = np.sum(zeta_p_counts, 0) > 0
    LP_resp_probs.append(np.sum(resp_cells_vec)/resp_cells_vec.shape[0])

# %% Boxplot + swarmplot
np.mean(sSC_resp_probs), np.mean(dSC_resp_probs), np.mean(LP_resp_probs)
# Plot boxplots
fig, ax = plt.subplots(1, 1, figsize=(3*cm, 3*cm))
sb.boxplot(data=[sSC_resp_probs, dSC_resp_probs, LP_resp_probs], ax=ax, palette=colors, fliersize=0)
sb.swarmplot(data=[sSC_resp_probs, dSC_resp_probs, LP_resp_probs], color='k', size=2, ax=ax)
ax.set_xticklabels(['sSC', 'dSC', 'LP'])
ax.set_ylabel('Proportion of responsive cells')
plt.savefig(os.path.join(fig_path, 'proportion_responsive_cells.pdf'), bbox_inches='tight', dpi=300)
plt.show()

#%% Raincloud plot for proportion of responsive cells (Boxplot + Scatter)
fig, ax = plt.subplots(1, 1, figsize=(3*cm, 3*cm))
data_resp = [sSC_resp_probs, dSC_resp_probs, LP_resp_probs]
labels = ['sSC', 'dSC', 'LP']

shift = 0.13

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Plot pointplot (median + IQR)
for i, d in enumerate(data_resp):
    median = np.median(d)
    q1, q3 = np.percentile(d, [25, 75])
    ax.errorbar(i + shift, median, yerr=[[median-q1], [q3-median]], fmt='o', color=colors[i], 
                elinewidth=1, capsize=2, markersize=4, zorder=2)

# Plot jittered scatter (stripplot) to the right of the boxplots
for i, d in enumerate(data_resp):
    x = np.random.normal(i - shift, 0.04, size=len(d))
    ax.scatter(x, d, color=colors[i], s=2, alpha=0.4, zorder=1)

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel('Proportion of responsive cells')
ax.set_ylim([0, 1.1])
sb.despine()
plt.savefig(os.path.join(fig_path, 'proportion_responsive_cells_raincloud_simple.pdf'), bbox_inches='tight', dpi=300)
plt.show()


# statistical test
mw_res = mannwhitneyu(sSC_resp_probs, dSC_resp_probs)
print("Mann-Whitney U test result (sSC vs dSC):", mannwhitneyu(sSC_resp_probs, dSC_resp_probs),mannwhitneyu(sSC_resp_probs, LP_resp_probs), mannwhitneyu(LP_resp_probs, dSC_resp_probs))
# statistical test
mw_res_sSC_dSC = mannwhitneyu(sSC_resp_probs, dSC_resp_probs)
mw_res_sSC_LP = mannwhitneyu(sSC_resp_probs, LP_resp_probs)
mw_res_dSC_LP = mannwhitneyu(dSC_resp_probs, LP_resp_probs)
print("Mann-Whitney U test result:", mw_res_sSC_dSC,mw_res_sSC_LP, mw_res_dSC_LP)
# Save stats test results to txt file
with open(os.path.join(fig_path, 'proportion_responsive_cells_stats.txt'), 'w') as f:
    f.write(f"MWU (sSC vs dSC): {mw_res_sSC_dSC}\n")
    f.write(f"MWU (sSC vs LP): {mw_res_sSC_LP}\n")
    f.write(f"MWU (dSC vs LP): {mw_res_dSC_LP}\n")
    # Median percent responsive cells
    f.write(f"% (sSC): {np.median(sSC_resp_probs)}\n")
    f.write(f"% (dSC): {np.median(dSC_resp_probs)}\n")
    f.write(f"% (LP): {np.median(LP_resp_probs)}\n")
