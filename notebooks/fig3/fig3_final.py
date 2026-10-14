#%%
import os
os.chdir('..')

# from datetime import datenid
import numpy as np
import seaborn as sb
import notebooks.CoordinatedCellTypes.fig3.fig3_preprocessing as fig3_preprocessing
import notebooks.CoordinatedCellTypes.fig3.fig3_encoding_CVPR as fig3_encoding_CVPR
import notebooks.CoordinatedCellTypes.fig3.fig3_encoding_unified as fig3_encoding_unified
import notebooks.CoordinatedCellTypes.fig3.fig3_encoding_pca_presvis as fig3_encoding_pca_presvis

cwdir = os.getcwd()
print(cwdir)
os.chdir('../..')
import main

drive_path = r"V:\Data\datajoint\DJEphys"
data_path = r"T:\Data"

ft = main.load(drive_path, data_path)

os.chdir(cwdir)

#%%
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

#%%
len(LP_frs)

#%% === Panel C | Participation Ratio ===
stim_types_plotting = [0,1,2,3,4,5]
sSC_frs_mats, sSC_frs_mats1, sSC_frs_mats2, _, _ = fig3_encoding_CVPR.preprocess(sSC_frs, sSC_keys_used, stim_types_plotting)
dSC_frs_mats, dSC_frs_mats1, dSC_frs_mats2, _, _ = fig3_encoding_CVPR.preprocess(dSC_frs, dSC_keys_used, stim_types_plotting)
LP_frs_mats, LP_frs_mats1, LP_frs_mats2, _, _ = fig3_encoding_CVPR.preprocess(LP_frs, LP_keys_used, stim_types_plotting)

# print(sSC_frs_mats[0].shape)
sSC_pca, _ = fig3_encoding_CVPR.run_pca(sSC_frs_mats,6, n_cells=100)
dSC_pca, _ = fig3_encoding_CVPR.run_pca(dSC_frs_mats,6, n_cells=100)
LP_pca, _ = fig3_encoding_CVPR.run_pca(LP_frs_mats,6, n_cells=100)

# visualize_cumul_var_expl_combined(sSC_pca, dSC_pca, LP_pca)
fig3_encoding_CVPR.visualize_participation_ratio(sSC_pca, 'sSC')
fig3_encoding_CVPR.visualize_participation_ratio(dSC_pca, 'dSC')
fig3_encoding_CVPR.visualize_participation_ratio(LP_pca, 'LP')

# fig2_encoding_CVPR.visualize_cumul_var_expl_combined(pca_list=[sSC_pca, dSC_pca, LP_pca], pca_labels=['sSC', 'dSC', 'LP'], pca_colors=[color_sSC, color_dSC, color_LP], log_scale=False)

sSC_PRs = fig3_encoding_CVPR.run_bootstrapped_cv_participation_ratio(sSC_frs_mats1, sSC_frs_mats2, n_bootstrap=500, n_cells=100)
dSC_PRs = fig3_encoding_CVPR.run_bootstrapped_cv_participation_ratio(dSC_frs_mats1, dSC_frs_mats2, n_bootstrap=500, n_cells=100)
LP_PRs = fig3_encoding_CVPR.run_bootstrapped_cv_participation_ratio(LP_frs_mats1, LP_frs_mats2, n_bootstrap=500, n_cells=100)

fig3_encoding_CVPR.plot_PRs([sSC_PRs, dSC_PRs, LP_PRs], ['sSC', 'dSC', 'LP'], [color_sSC, color_dSC, color_LP], fig_path)
fig3_encoding_CVPR.print_statistics([sSC_PRs, dSC_PRs, LP_PRs], ['sSC', 'dSC', 'LP'], os.path.join(fig_path, 'PR_regions_stats.txt'))

#%% === Panel H | Decoding performance ===
stim_types_plotting = [0,1,2,3,4,5]
target_window = [205, 215]

sSC_frs_mats, sSC_frs_trials_mats, sSC_keys_mats, _, _ = fig3_preprocessing.preprocess(sSC_frs, keys_used=sSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
dSC_frs_mats, dSC_frs_trials_mats, dSC_keys_mats, _, _ = fig3_preprocessing.preprocess(dSC_frs, keys_used=dSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
LP_frs_mats, LP_frs_trials_mats, LP_keys_mats, _, _ = fig3_preprocessing.preprocess(LP_frs, keys_used=LP_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)

decoding_classes = [[0], [2,3,5]]

fig3_encoding_unified.run_decoding_analysis_time([dSC_frs_trials_mats, LP_frs_trials_mats], 
                                            labels=['dSC', 'LP'], 
                                            colors=[color_dSC, color_LP], 
                                            decoding_classes=decoding_classes, fig_path=os.path.join(fig_path, f'final_100'), n_bootstraps=500, n_perms=500, k_dim=10, use_saved=False)
fig3_encoding_unified.run_decoding_analysis_cellcounts([sSC_frs_trials_mats, dSC_frs_trials_mats, LP_frs_trials_mats], 
                                            labels=['sSC', 'dSC', 'LP'], 
                                            colors=[color_sSC, color_dSC, color_LP], 
                                            decoding_classes=decoding_classes, fig_path=fig_path, n_bootstraps=100, n_perms=100, target_window=target_window)


#%% === Panel H | Low-dimensional representation ===
stim_types_plotting = [0,1,2,3,4,5]
sSC_frs_mats, sSC_frs_trials_mats, sSC_keys_mats, _, _ = fig3_preprocessing.preprocess(sSC_frs, keys_used=sSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
dSC_frs_mats, dSC_frs_trials_mats, dSC_keys_mats, _, _ = fig3_preprocessing.preprocess(dSC_frs, keys_used=dSC_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)
LP_frs_mats, LP_frs_trials_mats, LP_keys_mats, _, _ = fig3_preprocessing.preprocess(LP_frs, keys_used=LP_keys_used, stim_types=stim_types_plotting,
                              t_start=200, t_end=1000, base_start=50, base_end=350)

tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)
vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
stim_types = ['EXPA', 'EXPA_60', 'EXPA_25', 'EXPW', 'EXPW_60','DIMM', 'SWEP_4', 'SWEP_12', 'SWEP_20', 'SWEP_50']
stim_types_plotting = [0,1,2,3,4,5]
stim_types_labels = ['black loom 100%','black loom 60%', 'black loom 25%', 'white loom 100%', 'white loom 60%', 'dimming', 'sweeping 4', 'sweeping 12', 'sweeping 20', 'sweeping 50']
stim_types_colors = [vlag_cmap[-1], vlag_cmap[-3], vlag_cmap[-4], vlag_cmap[0], vlag_cmap[2], tab_cmap[5]]

sSC_pca, sSC_frs_mat_pca = fig3_encoding_pca_presvis.run_pca(sSC_frs_mats, n_stim=len(stim_types_plotting))
dSC_pca, dSC_frs_mat_pca = fig3_encoding_pca_presvis.run_pca(dSC_frs_mats, n_stim=len(stim_types_plotting))
LP_pca, LP_frs_mat_pca = fig3_encoding_pca_presvis.run_pca(LP_frs_mats, n_stim=len(stim_types_plotting))

fig3_encoding_pca_presvis.visualize_pca_components(sSC_frs_mat_pca, sSC_pca, stim_types_plotting, np.array(stim_types_labels)[stim_types_plotting], 'sSC', 
                         latent_dims=3, stim_types_colors=np.array(stim_types_colors)[stim_types_plotting], fig_path=fig_path)
fig3_encoding_pca_presvis.visualize_pca_components(dSC_frs_mat_pca, dSC_pca, stim_types_plotting, np.array(stim_types_labels)[stim_types_plotting], 'dSC', 
                         latent_dims=3, stim_types_colors=np.array(stim_types_colors)[stim_types_plotting], fig_path=fig_path)
fig3_encoding_pca_presvis.visualize_pca_components(LP_frs_mat_pca, LP_pca, stim_types_plotting, np.array(stim_types_labels)[stim_types_plotting], 'LP', 
                         latent_dims=3, stim_types_colors=np.array(stim_types_colors)[stim_types_plotting], fig_path=fig_path)

stim_types_plotting = [0,2,3,5]
stim_types_colors = [vlag_cmap[-1], vlag_cmap[-4], vlag_cmap[0], tab_cmap[5]]
ani = fig3_encoding_pca_presvis.visualize_pca_3d(sSC_frs_mat_pca, stim_types_plotting, np.array(stim_types_labels)[stim_types_plotting], 'sSC', 
                 stim_types_colors=stim_types_colors, animation=False, t_start=150, t_end=301, rotate=True, repeats=3)#301
ani = fig3_encoding_pca_presvis.visualize_pca_3d(dSC_frs_mat_pca, stim_types_plotting, np.array(stim_types_labels)[stim_types_plotting], 'dSC', 
                 stim_types_colors=stim_types_colors, animation=False, t_start=150, t_end=301, rotate=True, repeats=3)
ani = fig3_encoding_pca_presvis.visualize_pca_3d(LP_frs_mat_pca, stim_types_plotting, np.array(stim_types_labels)[stim_types_plotting], 'LP', 
                 stim_types_colors=stim_types_colors, animation=False, t_start=150, t_end=301, rotate=True, repeats=3)