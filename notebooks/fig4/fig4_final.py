#%%
import os
os.chdir('..')

import pandas as pd
# from datetime import datenid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import mannwhitneyu
import notebooks.CoordinatedCellTypes.fig2.fig2_preprocessing as fig2_preprocessing
from joblib import Parallel, delayed
import notebooks.CoordinatedCellTypes.final.fig3_encoding_CVPR as fig3_encoding_CVPR
import notebooks.CoordinatedCellTypes.fig4.fig4_encoding_unified as fig4_encoding_unified
import notebooks.CoordinatedCellTypes.fig3.fig2_encoding_pca_presvis as fig2_encoding_pca_presvis

#%%
cwdir = os.getcwd()
print(cwdir)
os.chdir('../..')
import main

drive_path = r"V:\Data\datajoint\DJEphys"
data_path = r"T:\Data"

ft = main.load(drive_path, data_path)

os.chdir(cwdir)

#%%
mouse_table = ft.Mouse()
subsess_spikes_table = ft.ephys_tables.SubsessionSpikes()
trials_table = ft.ephys_tables.TrialSpikes()
opto_table = ft.ephys_tables.OptoTagging()
opto_cluster_table = ft.ephys_tables.OptoCluster()
cluster_table = ft.ephys_tables.Cluster()
opto_clustering_table = ft.ephys_tables.OptoClusterClustering()
cluster_trials_table = ft.ephys_tables.ClusterTrial()
cluster_sweep_metrics_table = ft.ephys_analysis_tables.ClusterSweepMetrics()
cluster_loom_metrics_table = ft.ephys_analysis_tables.ClusterLoomMetrics()
cluster_depth_table = ft.ephys_tables.ClusterDepth()
cluster_trial_sweeping_table = ft.ephys_tables.TrialSweeping()
cluster_trials2_table = ft.ephys_tables.ClusterTrial2()
opto_curation_table = ft.ephys_tables.OptoCuration()
depth_table = ft.ephys_tables.SortingDepth()

#%% Figure parameters and path
# reset path
from notebooks.CoordinatedCellTypes.final.fig_params import *
fig_path = r''
tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)
vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
color_all = color_CTRL # matplotlib.colors.to_rgba('#9E9E9E')
color_opto = color_GRP # matplotlib.colors.to_rgba('#C84B31') # color_NTSR
color_late = color_GRP # matplotlib.colors.to_rgba('#C84B31') # color_NTSR
#%% Load data (only visually responsive cells)
from notebooks.CoordinatedCellTypes.final.datajoint_utils import get_cluster_frs

stim_types = ['EXPA', 'EXPA_60', 'EXPA_25', 'EXPW', 'EXPW_60','DIMM']
conditions = ['dSC'] # , 'LP'

exp_dict = {}
for cond in conditions:
    print(cond)
    frs_dict = get_cluster_frs([], stim_types, dataset_name=f'{cond}_VR_all_fr25_20260116', means=False, subsession_id='01', fix_sweeping=False)
    frs, depths, keys_used = frs_dict['cluster_frs'], frs_dict['cluster_depths'], frs_dict['cluster_keys']
    
dSC_frs, dSC_depths, dSC_keys_used = exp_dict['dSC'] # LP

#%% -- Get metrics --
response_onsets = []
response_probs = []
labels = []
thr_labels = []
depths = []
for exp_key, cl_keys in dSC_keys_used.items():
    for i, cl_key in enumerate(cl_keys):
        cl_key_full = (opto_curation_table & cl_key).fetch('KEY')
        if len(cl_key_full) > 0:
            response_onset_ms, response_prob, label = (opto_curation_table & cl_key_full).fetch1('response_onset_ms', 'response_prob', 'label')
            response_times, response_prob = (opto_cluster_table & cl_key_full).fetch('median_response_time', 'response_prob')
            response_onset_ms = np.min(response_times)
            response_prob = np.max(response_prob)
            response_onsets.append(response_onset_ms)
            response_probs.append(response_prob)
            labels.append(label)
            shank = (cluster_depth_table & cl_key).fetch1('shank')
            ssc_surface, ssc_bottom = (depth_table & cl_key & {'shank': shank}).fetch1('ssc_surface', 'ssc_bottom')

            depths.append(np.squeeze(dSC_depths[exp_key][i])) # -ssc_size
            thr_labels.append('early' if (response_onset_ms < 9) and (response_prob > 0.1) else 'late')

def group_labels(labels):
    label_summary = []
    for label in labels:
        if label in [10, 11]:
            label_summary.append('Early')
        else:
            label_summary.append('Late')
    return label_summary

# Combine into pandas dataframe
df = pd.DataFrame({
    'response_onset': response_onsets,
    'response_prob': response_probs,
    'label': group_labels(labels),
    'threshold_label': thr_labels,
    'depth': np.array(depths).astype('float32')
})

# Plot histogram of response onsets
plt.figure(figsize=(3*cm, 2*cm))
sb.histplot(data=df, x='response_onset', bins=np.arange(0, 20, 1), multiple='stack', color='k')
plt.xlabel('Response Onset (ms)')
plt.axvline(9, c='r', ls='--')
plt.ylabel('Count')
plt.savefig(os.path.join(fig_path, 'opto_response_onset_histogram.pdf'), bbox_inches='tight', dpi=300)
plt.show()

# Create a temporary dummy column so Seaborn groups them onto 1 shared axis
df_plot = df.copy()
df_plot['dummy'] = ''

plt.figure(figsize=(2 * cm, 3 * cm))

sb.violinplot(
    data=df_plot,
    y='depth',
    x='dummy',  # Places both on the same central position
    hue='threshold_label',
    split=True,  # Splits the single violin into left (early) and right (late)
    palette={'early': 'k', 'late': 'k'},
    hue_order=['early', 'late'],  # Ensures consistent left/right assignment
    fill=False,
    inner='quartile',
    bw_adjust=0.8, common_norm=False,
    cut=0,
)

plt.axvline(0, color='r', linewidth=1)  # Central y-axis divider
plt.ylabel('Collicular Depth')
plt.xlabel('')
plt.gca().invert_yaxis()  # Invert so deeper points downwards

# Clean up legend
# plt.legend(title='Label', loc='upper right')
# remove legend
plt.legend([], [], frameon=False)
sb.despine()

plt.savefig(
    os.path.join(fig_path, 'opto_response_depth_split_sns.pdf'),
    bbox_inches='tight',
    dpi=300,
)
plt.show()

def filter_exp_dict(frs, depths, keys_used, filter=None):
    frs_filt, depths_filt, keys_used_filt = {}, {}, {}
    early_idxs = {}
    late_idxs = {}
    for exp_key, cl_keys in keys_used.items():
        # if exp_key != ('01942', '74R', '20230307'):
        #     continue
        for i, cl_key in enumerate(cl_keys):
            response_times, response_prob = (opto_cluster_table & cl_key).fetch('median_response_time', 'response_prob')
            if len(response_times) == 0:
                continue 
            response_time_ms = np.min(response_times)
            cl_key_full = (opto_curation_table & cl_key).fetch('KEY')
            if len(cl_key_full) > 0:
                if exp_key not in early_idxs:
                    early_idxs[exp_key] = np.zeros(len(cl_keys), dtype=int)
                    late_idxs[exp_key] = np.zeros(len(cl_keys), dtype=int)

            if filter == 'early':
                early_label_filter= '(response_prob > 0.1)'
                cl_key_full = (opto_curation_table & cl_key & early_label_filter).fetch('KEY')
                if (len(cl_key_full) > 0) and (response_time_ms < 9):
                    if exp_key not in keys_used_filt:
                        keys_used_filt[exp_key] = []
                        frs_filt[exp_key] = []
                        depths_filt[exp_key] = []
                    frs_filt[exp_key].append(frs[exp_key][i])
                    depths_filt[exp_key].append(depths[exp_key][i])
                    keys_used_filt[exp_key].append(cl_key)
                    early_idxs[exp_key][i] = 1

            elif filter == 'late':
                cl_key_full = (opto_curation_table & cl_key).fetch('KEY')
                if (len(cl_key_full) > 0) and (response_time_ms >= 9):
                    if exp_key not in keys_used_filt:
                        keys_used_filt[exp_key] = []
                        frs_filt[exp_key] = []
                        depths_filt[exp_key] = []
                    frs_filt[exp_key].append(frs[exp_key][i])
                    depths_filt[exp_key].append(depths[exp_key][i])
                    keys_used_filt[exp_key].append(cl_key)
                    late_idxs[exp_key][i] = 1
            else:
                cl_key_full = (opto_curation_table & cl_key).fetch('KEY')
                if len(cl_key_full) > 0:
                    if exp_key not in keys_used_filt:
                        keys_used_filt[exp_key] = []
                        frs_filt[exp_key] = []
                        depths_filt[exp_key] = []
                    frs_filt[exp_key].append(frs[exp_key][i])
                    depths_filt[exp_key].append(depths[exp_key][i])
                    keys_used_filt[exp_key].append(cl_key)
        if exp_key in frs_filt.keys():
            frs_filt[exp_key] = np.array(frs_filt[exp_key])
            depths_filt[exp_key] = np.array(depths_filt[exp_key])
            keys_used_filt[exp_key] = np.array(keys_used_filt[exp_key])
            if filter == 'all':
                frs_filt[exp_key] = frs[exp_key]
                depths_filt[exp_key] = depths[exp_key]
                keys_used_filt[exp_key] = keys_used[exp_key]
    return frs_filt, depths_filt, keys_used_filt, early_idxs, late_idxs

sSC_frs, sSC_depths, sSC_keys_used, _, late_idxs = filter_exp_dict(dSC_frs, dSC_depths, dSC_keys_used, filter='late')
dSC_frs, dSC_depths, dSC_keys_used, _, _ = filter_exp_dict(dSC_frs, dSC_depths, dSC_keys_used, filter='all')
LP_frs, LP_depths, LP_keys_used, early_idxs, _ = filter_exp_dict(dSC_frs, dSC_depths, dSC_keys_used, filter='early')

#%% === Panel C | Participation Ratio ===
stim_types_plotting = [0,1,2,3,4,5]
late_frs_mats, late_frs_mats1, late_frs_mats2, _, _ = fig3_encoding_CVPR.preprocess(sSC_frs, sSC_keys_used, stim_types_plotting)
all_frs_mats, all_frs_mats1, all_frs_mats2, _, _ = fig3_encoding_CVPR.preprocess(dSC_frs, dSC_keys_used, stim_types_plotting)
early_frs_mats, early_frs_mats1, early_frs_mats2, _, _ = fig3_encoding_CVPR.preprocess(LP_frs, LP_keys_used, stim_types_plotting)

# # print(sSC_frs_mats[0].shape)
late_pca, _ = fig3_encoding_CVPR.run_pca(late_frs_mats,6, n_cells=30)
all_pca, _ = fig3_encoding_CVPR.run_pca(all_frs_mats,6, n_cells=30)
early_pca, _ = fig3_encoding_CVPR.run_pca(early_frs_mats,6, n_cells=30)

# # visualize_cumul_var_expl_combined(sSC_pca, dSC_pca, LP_pca)
fig3_encoding_CVPR.visualize_participation_ratio(late_pca, 'late')
fig3_encoding_CVPR.visualize_participation_ratio(all_pca, 'all')
fig3_encoding_CVPR.visualize_participation_ratio(early_pca, 'early')

fig3_encoding_CVPR.visualize_cumul_var_expl_combined(pca_list=[late_pca, all_pca, early_pca], pca_labels=['late', 'all', 'early'], pca_colors=[color_late, color_all, color_opto], log_scale=False, fig_path=fig_path)
n_cells = 30
n_bootstraps = 10000
n_stim = 6

print("Bootstrapping late...")
late_var_bootstraps = fig3_encoding_CVPR.get_variance_bootstraps(late_frs_mats, n_stim, n_cells, n_bootstraps)
print("Bootstrapping all...")
all_var_bootstraps = fig3_encoding_CVPR.get_variance_bootstraps(all_frs_mats, n_stim, n_cells, n_bootstraps)
print("Bootstrapping early...")
early_var_bootstraps = fig3_encoding_CVPR.get_variance_bootstraps(early_frs_mats, n_stim, n_cells, n_bootstraps)

# 2. Run the statistical tests on individual PC variance differences
fig3_encoding_CVPR.test_pc_variance_difference(late_var_bootstraps, all_var_bootstraps, 'late', 'all')
fig3_encoding_CVPR.test_pc_variance_difference(all_var_bootstraps, early_var_bootstraps, 'all', 'early')
fig3_encoding_CVPR.test_pc_variance_difference(late_var_bootstraps, early_var_bootstraps, 'late', 'early')

# 3. Generate the cumulative variance figure with CIs using the same data
fig3_encoding_CVPR.visualize_cumul_var_expl_combined_ci(
    bootstraps_list=[all_var_bootstraps, early_var_bootstraps, late_var_bootstraps], 
    pca_labels=['all', 'early', 'late'], 
    pca_colors=[color_all, color_opto, color_late], #[cite: 1]
    fig_path=fig_path, log_scale=True #[cite: 1]
)

fig3_encoding_CVPR.visualize_var_expl_barplot(
    bootstraps_list=[all_var_bootstraps, early_var_bootstraps, late_var_bootstraps], 
    pca_labels=['all', 'early', 'late'], 
    pca_colors=[color_all, color_opto, color_late], #[cite: 1]
    k=3, # Change this to show more or fewer PCs
    fig_path=fig_path #[cite: 1]
)

#%% --- decoding accuracy & variance explained & weights
stim_types_plotting = [0,1,2,3,4,5]
target_window = [200, 250]

late_frs_mats, late_frs_trials_mats, late_keys_mats, _, _ = fig2_preprocessing.preprocess(sSC_frs, keys_used=sSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
all_frs_mats, all_frs_trials_mats, all_keys_mats, _, _ = fig2_preprocessing.preprocess(dSC_frs, keys_used=dSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
early_frs_mats, early_frs_trials_mats, early_keys_mats, _, _ = fig2_preprocessing.preprocess(LP_frs, keys_used=LP_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)

early_idxs
early_idxs_concat = np.concatenate(list(early_idxs.values()))
late_idxs_concat = np.concatenate(list(late_idxs.values()))
assert np.all(early_idxs_concat+late_idxs_concat<2)

#%% ----- Sufficiency: decoding accuracy 2 ------
# Get decoding accuracy of early cells
# all_frs_mats
target_window = [205, 215]

early_data_decod = fig2_preprocessing.unflatten_frs_mat(np.concatenate(early_frs_trials_mats, axis=0), n_stim=6, n_trials=8, n_timepoints=800)
all_data_decod = fig2_preprocessing.unflatten_frs_mat(np.concatenate(all_frs_trials_mats, axis=0), n_stim=6, n_trials=8, n_timepoints=800)
# filter all data to be only non-optotagged
all_data_decod = all_data_decod[(early_idxs_concat==0)]
print(len(early_data_decod), len(all_data_decod))
n_cells = len(early_data_decod)
print("n_cells", n_cells)

a_r_all, var_r, _, _, _, _, _, _ = fig4_encoding_unified.unified_decoding_engine(
        early_data_decod, 
        decoding_classes, 
        k_dim=10, 
        use_pca=True,
        n_shuffles=0, # <-- 0 shuffles
        seed=42,
        time_window=target_window # <-- Only compute these 10 bins
    )
    
early_result = np.mean(a_r_all[:, target_window[0]:target_window[1]])
early_result_var = np.mean(var_r[:, target_window[0]:target_window[1]])
print("Early:", early_result)
all_results = []
all_results_var = []
def run_decod():
    dynamic_seed = np.random.randint(10000)
    sub_data = all_data_decod[np.random.choice(all_data_decod.shape[0], n_cells, replace=False)]
    a_r_all, var_r, _, _, _, _, _, _ = fig4_encoding_unified.unified_decoding_engine(
        sub_data, 
        decoding_classes, 
        k_dim=10, 
        use_pca=True,
        n_shuffles=0, # <-- 0 shuffles
        seed=dynamic_seed,
        time_window=target_window # <-- Only compute these 10 bins
    )
    print(np.mean(a_r_all[:, target_window[0]:target_window[1]]))
    return a_r_all[:, target_window[0]:target_window[1]], var_r[:, target_window[0]:target_window[1]]

boot_accs = Parallel(n_jobs=5, verbose=10)(delayed(run_decod)() for _ in range(1000))
boot_accs_results = list(zip(*boot_accs))
all_results.append(boot_accs_results[0])
all_results_var.append(boot_accs_results[1])
# all_results.append(boot_accs)

#%%
print(np.mean(np.mean(np.squeeze(np.array(all_results)), axis=(1,2))))
fig, ax = plt.subplots(1, 1, figsize=(5*cm, 4*cm))
ax.hist(np.mean(np.squeeze(np.array(all_results)), axis=(1,2)), bins=20, alpha=0.5, color=color_all, label='all')
ax.axvline(early_result, color=color_opto, label='early', linewidth=2)
ax.set_xlabel('Decoding accuracy')
# plt.savefig(os.path.join(fig_path, 'opto_decoding_accuracy_bootstrap_hist.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# boot_accs_means = np.mean(np.squeeze(np.array(all_results)), axis=(1,2))
# # Calc p value comparing early vs null distr
# p_value = np.sum(np.array(boot_accs_means) >= early_result) / len(boot_accs_means)
# print(f"P-value (early vs null): {p_value}")
boot_accs_means = np.mean(np.squeeze(np.array(all_results)), axis=(1,2))
null_mean_acc = np.mean(boot_accs_means)
# Calc ONE-TAILED p value comparing early vs null distr
p_value_1t = np.sum(np.array(boot_accs_means) >= early_result) / len(boot_accs_means)

# Calc TWO-TAILED p value comparing early vs null distr
obs_diff_acc = np.abs(early_result - null_mean_acc)
p_value_2t = np.sum(np.abs(np.array(boot_accs_means) - null_mean_acc) >= obs_diff_acc) / len(boot_accs_means)

print(f"Accuracy P-value (early vs null) - One-tailed: {p_value_1t}")
print(f"Accuracy P-value (early vs null) - Two-tailed: {p_value_2t}")

print(np.mean(np.mean(np.squeeze(np.array(all_results_var)), axis=(1,2))))
fig, ax = plt.subplots(1, 1, figsize=(5*cm, 4*cm))
ax.hist(np.mean(np.squeeze(np.array(all_results_var)), axis=(1,2)), bins=20, alpha=0.5, color=color_all, label='all')
ax.axvline(early_result_var, color=color_opto, label='early', linewidth=2)
ax.set_xlabel('Variance explained')
# plt.savefig(os.path.join(fig_path, 'opto_decoding_variance_bootstrap_hist.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# boot_vars_means = np.mean(np.squeeze(np.array(all_results_var)), axis=(1,2))
# # Calc p value comparing early vs null distr
# p_value = np.sum(np.array(boot_vars_means) >= early_result_var) / len(boot_vars_means)
# print(f"P-value (early vs null): {p_value}")
boot_vars_means = np.mean(np.squeeze(np.array(all_results_var)), axis=(1,2))
null_mean_var = np.mean(boot_vars_means)

# Calc ONE-TAILED p value comparing early vs null distr
p_value_var_1t = np.sum(np.array(boot_vars_means) >= early_result_var) / len(boot_vars_means)

# Calc TWO-TAILED p value comparing early vs null distr
obs_diff_var = np.abs(early_result_var - null_mean_var)
p_value_var_2t = np.sum(np.abs(np.array(boot_vars_means) - null_mean_var) >= obs_diff_var) / len(boot_vars_means)

print(f"Variance P-value (early vs null) - One-tailed: {p_value_var_1t}")
print(f"Variance P-value (early vs null) - Two-tailed: {p_value_var_2t}")

#%% ----- Necessity: decoding accuracy ------
stim_types_plotting = [0,1,2,3,4,5]
target_window = [205, 215]

late_frs_mats, late_frs_trials_mats, late_keys_mats, _, _ = fig2_preprocessing.preprocess(sSC_frs, keys_used=sSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
all_frs_mats, all_frs_trials_mats, all_keys_mats, _, _ = fig2_preprocessing.preprocess(dSC_frs, keys_used=dSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
early_frs_mats, early_frs_trials_mats, early_keys_mats, _, _ = fig2_preprocessing.preprocess(LP_frs, keys_used=LP_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)

early_idxs_concat = np.concatenate(list(early_idxs.values()))
late_idxs_concat = np.concatenate(list(late_idxs.values()))
assert np.all(early_idxs_concat+late_idxs_concat<2)
# Get decoding accuracy of all excluding early cells (drop-out accuracy)
all_data_decod = fig2_preprocessing.unflatten_frs_mat(np.concatenate(all_frs_trials_mats, axis=0), n_stim=6, n_trials=8, n_timepoints=800)
assert len(all_data_decod) == len(early_idxs_concat)
exclearly_data_decod = all_data_decod[(early_idxs_concat==0)]
print("exclearly_data_decod.shape", exclearly_data_decod.shape)
a_r_all, var_r, _, _, _, _, _, _ = fig4_encoding_unified.unified_decoding_engine(
        exclearly_data_decod, 
        decoding_classes, 
        k_dim=10, 
        use_pca=True,
        n_shuffles=0, # <-- 0 shuffles
        seed=42,
        time_window=target_window # <-- Only compute these 10 bins
    )
exclearly_result = np.mean(a_r_all[:, target_window[0]:target_window[1]])
exclearly_result_var = np.mean(var_r[:, target_window[0]:target_window[1]])
print("Excluding early:", exclearly_result)

exclall_results = []
exclall_results_var = []
def run_decod_ablate():
    dynamic_seed = np.random.randint(10000)
    # Randomly exclude non-early cells
    # all_idxs_where = np.where((early_idxs_concat==0)&(late_idxs_concat==0))[0]
    all_idxs_where = np.where((early_idxs_concat==0))[0]
    cells_to_exclude = np.random.choice(all_idxs_where, np.sum((early_idxs_concat==1)), replace=False)
    sub_data = np.delete(all_data_decod, cells_to_exclude, axis=0)
    print("sub_data.shape", sub_data.shape)
    # assert len(sub_data) == np.sum(early_idxs_concat==0)
    # sub_data = all_data_decod[np.random.choice(all_data_decod.shape[0], n_cells, replace=False)]
    a_r_all, var_r, _, _, _, _, _, _ = fig4_encoding_unified.unified_decoding_engine(
        sub_data, 
        decoding_classes, 
        k_dim=10, 
        use_pca=True,
        n_shuffles=0, # <-- 0 shuffles
        seed=dynamic_seed,
        time_window=target_window # <-- Only compute these 10 bins
    )
    print(np.mean(a_r_all[:, target_window[0]:target_window[1]]))
    return a_r_all[:, target_window[0]:target_window[1]], var_r[:, target_window[0]:target_window[1]]

boot_accs = Parallel(n_jobs=5, verbose=10)(delayed(run_decod_ablate)() for _ in range(1000))
boot_accs_results = list(zip(*boot_accs))
exclall_results.append(boot_accs_results[0])
exclall_results_var.append(boot_accs_results[1])
#%%
print(n_cells)
print(len(all_data_decod))
#%%
print(np.mean(np.mean(np.squeeze(np.array(exclall_results)), axis=(1,2))))
fig, ax = plt.subplots(1, 1, figsize=(5*cm, 4*cm))
ax.hist(np.mean(np.squeeze(np.array(exclall_results)), axis=(1,2)), bins=20, alpha=0.5, color=color_all, label='all')
ax.axvline(exclearly_result, color=color_opto, label='early', linewidth=2)
ax.set_xlabel('Decoding accuracy')
# plt.savefig(os.path.join(fig_path, 'opto_ablation_decoding_accuracy_bootstrap_hist2.pdf'), dpi=300, bbox_inches='tight')
plt.show()

boot_accs_means = np.mean(np.squeeze(np.array(exclall_results)), axis=(1,2))
# Calc p value comparing early vs null distr
p_value = np.sum(np.array(boot_accs_means) <= exclearly_result) / len(boot_accs_means)
print(f"P-value (early vs null): {p_value}")

print(np.mean(np.mean(np.squeeze(np.array(exclall_results_var)), axis=(1,2))))
fig, ax = plt.subplots(1, 1, figsize=(5*cm, 4*cm))
ax.hist(np.mean(np.squeeze(np.array(exclall_results_var)), axis=(1,2)), bins=20, alpha=0.5, color=color_all, label='all')
ax.axvline(exclearly_result_var, color=color_opto, label='early', linewidth=2)
ax.set_xlabel('Variance explained')
# plt.savefig(os.path.join(fig_path, 'opto_ablation_decoding_variance_bootstrap_hist2.pdf'), dpi=300, bbox_inches='tight')
plt.show()

boot_var_means = np.mean(np.squeeze(np.array(exclall_results_var)), axis=(1,2))
# Calc p value comparing early vs null distr
p_value = np.sum(np.array(boot_var_means) <= exclearly_result_var) / len(boot_var_means)
print(f"P-value (early vs null): {p_value}")
# bootstrapped (null): Get decoding accuracy of all excluding random sampled non-tagged cells
