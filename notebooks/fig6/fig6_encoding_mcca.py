#%%
import os
import datajoint as dj
# from datetime import datenid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import copy
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import wilcoxon
from cca_zoo.linear import MCCA
from utils import behavior_ethogram

#%% Figure parameters and path
# reset path
from fig_params import *
fig_path = r''


#%% Load data (only visually responsive cells)
from datajoint_utils import get_cluster_frs

stim_types = ['EXPA', 'EXPA_60', 'EXPA_25', 'EXPW', 'EXPW_60','DIMM']
sSC_dict = get_cluster_frs([], stim_types, dataset_name='sSC_VR_all_fr25_20260116', means=False, subsession_id='01')
dSC_dict = get_cluster_frs([], stim_types, dataset_name='dSC_VR_all_fr25_20260116', means=False, subsession_id='01')
LP_dict = get_cluster_frs([], stim_types, dataset_name='LP_VR_all_fr25_20260116', means=False, subsession_id='01')
sSC_frs, sSC_depths, sSC_keys_used = sSC_dict['cluster_frs'], sSC_dict['cluster_depths'], sSC_dict['cluster_keys']
dSC_frs, dSC_depths, dSC_keys_used = dSC_dict['cluster_frs'], dSC_dict['cluster_depths'], dSC_dict['cluster_keys']
LP_frs, LP_depths, LP_keys_used = LP_dict['cluster_frs'], LP_dict['cluster_depths'], LP_dict['cluster_keys']
sum([len(v) for k,v in sSC_frs.items()]), sum([len(v) for k,v in dSC_frs.items()]), sum([len(v) for k,v in LP_frs.items()])

#%% Preprocessing
stim_types_plotting = [0,1,2,3,4,5]

def baseline_subtract(frs, baseline_start=50, baseline_end=350):
    """For each trial, subtract baseline from firing rates. """
    baseline = np.mean(frs[:, :, baseline_start:baseline_end], axis=-1, keepdims=True)
    return frs - baseline

# Flatten frs_mat for PCA and CCA
def flatten_frs_mat(frs_mat):
    n_cells, n_stim, n_trials, n_timepoints = frs_mat.shape
    return frs_mat.reshape(n_cells, n_stim * n_trials * n_timepoints)

def unflatten_frs_mat(frs_mat_flat, n_stim, n_trials, n_timepoints):
    n_cells = frs_mat_flat.shape[0]
    return frs_mat_flat.reshape(n_cells, n_stim, n_trials, n_timepoints)

def preprocess(frs_dict, keys_used, trial_pre_from=2, trial_pre_to=10, t_start=200, t_end=1000,stim_types=stim_types_plotting):
    frs_all = copy.deepcopy(frs_dict)
    
    frs_means = []
    frs_stds = []
    frs_means_trials = []
    frs_stds_trials = []
    frs_mats = []
    frs_trials_mats = []
    keys_mats = []

    for exp_key in frs_all:
        stim_frs_all = []
        stim_frs_trials_all = []
        stim_keys_all = []
        for stim_idx in stim_types:
            # baseline subtraction
            print(frs_all[exp_key].shape)
            frs = frs_all[exp_key][:,stim_idx,:]
            print("frs shape", frs.shape)
            frs_basesub = baseline_subtract(frs, baseline_start=0, baseline_end=200)[:,:,t_start:t_end]
            # trial average
            stim_frs_trials_all.append(frs_basesub[:,trial_pre_from:trial_pre_to])
            frs_mean = np.mean(frs_basesub[:, trial_pre_from:trial_pre_to], axis=1)
            frs_mean = frs_mean[:, np.newaxis, :]

            stim_frs_all.append(frs_mean)
            stim_keys_all.append(np.array(keys_used[exp_key]))
        if len(stim_frs_all) == 0:
            continue

        stim_frs_all = np.array(stim_frs_all)
        stim_frs_all = stim_frs_all.swapaxes(0,1)  # shape: n_cells, n_stim, 1, n_timepoints
        print(stim_frs_all.shape)

        stim_frs_trials_all = np.array(stim_frs_trials_all)
        stim_frs_trials_all = stim_frs_trials_all.swapaxes(0,1)  # shape: n_cells, n_stim, n_trials, n_timepoints

        frs_mats_flat = flatten_frs_mat(stim_frs_all)
        frs_trials_mats_flat = flatten_frs_mat(stim_frs_trials_all)
        print("frs_mats_flat.shape", frs_mats_flat.shape)
        # soft normalization
        means = np.mean(frs_mats_flat, axis=1, keepdims=True)
        stds = np.std(frs_mats_flat, axis=1, keepdims=True)
        means_trials = np.mean(frs_trials_mats_flat, axis=1, keepdims=True)
        stds_trials = np.std(frs_trials_mats_flat, axis=1, keepdims=True)

        frs_mat_flat_std = (frs_mats_flat)/(stds+1) # baseline subtracted & divided by std across concatenated trials
        frs_trials_mat_flat_std = (frs_trials_mats_flat)/(stds_trials+1)

        frs_mats.append(frs_mat_flat_std) #np.array(stim_frs_all)
        frs_trials_mats.append(frs_trials_mat_flat_std)
        frs_means.append(means)
        frs_stds.append(stds)
        frs_means_trials.append(means_trials)
        frs_stds_trials.append(stds_trials)
        
        keys_mats.append(stim_keys_all)
    return frs_mats, frs_trials_mats, keys_mats, frs_means, frs_stds, frs_means_trials, frs_stds_trials


#%%
sSC_frs_mats, sSC_frs_trials_mats, sSC_keys_mats, _, _, _, _ = preprocess(sSC_frs, sSC_keys_used, stim_types=stim_types_plotting)
dSC_frs_mats, dSC_frs_trials_mats, dSC_keys_mats, _, _, _, _ = preprocess(dSC_frs, dSC_keys_used, stim_types=stim_types_plotting)
LP_frs_mats, LP_frs_trials_mats, LP_keys_mats, _, _, _, _ = preprocess(LP_frs, LP_keys_used, stim_types=stim_types_plotting)
print(sSC_frs_mats[0].shape)

# Filter out experiments with less than 15 cells ...
dSC_frs_mats = [mat for mat in dSC_frs_mats if mat.shape[0]>=15] # 15
dSC_keys_mats = [mat for mat in dSC_keys_mats if len(mat[0])>=15] # 15
region = 'dSC'


#%%
vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)

from cca_zoo.linear import MCCA
def run_mcca(frs_mats, n_stim, latent_dims=3, pca=True):
    frs_mats_transpose = [mat.T for mat in frs_mats]

    mcca = MCCA(latent_dimensions=latent_dims, c=1, pca=pca)
    mcca.fit(frs_mats_transpose)
    frs_mats_flat_cca_transpose = mcca.transform(frs_mats_transpose)
    frs_mats_flat_cca = [unflatten_frs_mat(mat.T, n_stim=n_stim, n_trials=1, n_timepoints=800) for mat in frs_mats_flat_cca_transpose]
    return frs_mats_flat_cca, mcca

latent_dims = 5
res, mcca = run_mcca(dSC_frs_mats, n_stim=len(stim_types_plotting), pca=True, latent_dims=latent_dims)

#%%
def get_mean_pairwise_correlations(transformed_views, n_dims):
    """Calculates the average pairwise Pearson correlation across all sessions for each latent dimension."""
    n_views = len(transformed_views)
    mean_rs = []
    
    for dim in range(n_dims):
        r_sum = 0
        count = 0
        # Iterate through all unique pairs of sessions
        for i in range(n_views):
            for j in range(i + 1, n_views):
                # Extract the 1D array for the specific latent dimension
                cv_i = transformed_views[i][:, dim]
                cv_j = transformed_views[j][:, dim]
                
                # Calculate Pearson r
                r = np.corrcoef(cv_i, cv_j)[0, 1]
                r_sum += r
                count += 1
                
        mean_rs.append(r_sum / count)
        
    return np.array(mean_rs)

def run_mcca_alignment_quantification(frs_mats, n_stim, n_timepoints=800, latent_dims=3, n_shuffles=100):
    """
    Fits true MCCA and generates a null distribution of correlations via stimulus block-shuffling.
    """
    n_views = len(frs_mats)
    
    # 1. Compute True MCCA
    frs_mats_transpose = [mat.T for mat in frs_mats]
    true_mcca = MCCA(latent_dimensions=latent_dims, c=1, pca=True)
    true_mcca.fit(frs_mats_transpose)

    true_rs = true_mcca.average_pairwise_correlations(frs_mats_transpose)
    
    # 2. Compute Null MCCA (Stimulus Block-Shuffle)
    null_rs_dist = np.zeros((n_shuffles, latent_dims))
    
    print(f"Running {n_shuffles} MCCA permutations...")
    for it in range(n_shuffles):
        shuffled_views = []
        
        for view_idx in range(n_views):
            mat = frs_mats[view_idx] # Shape: (n_cells, n_stim * n_timepoints)
            
            # Reshape to isolate the stimulus blocks: (n_cells, n_stim, n_timepoints)
            # mat_blocked = mat.reshape(mat.shape[0], n_stim, n_timepoints)
            mat_blocked = unflatten_frs_mat(mat, n_stim=n_stim, n_trials=1, n_timepoints=800)
            
            # Shuffle the stimulus blocks (axis 1)
            shuffled_indices = np.random.permutation(n_stim)
            mat_shuffled = mat_blocked[:, shuffled_indices, :]
            
            # Flatten back and transpose for cca_zoo
            mat_shuffled_flat = mat_shuffled.reshape(mat.shape[0], -1).T
            shuffled_views.append(mat_shuffled_flat)
            
        # Re-fit MCCA on the broken alignment
        null_mcca = MCCA(latent_dimensions=latent_dims, c=1, pca=True)
        null_mcca.fit(shuffled_views)
        
        null_rs_dist[it, :] = null_mcca.average_pairwise_correlations(shuffled_views)
    return true_rs, null_rs_dist

# --- Execution ---
# Assuming dSC_frs_mats and stim_types_plotting are defined
true_rs, null_rs_dist = run_mcca_alignment_quantification(
    dSC_frs_mats, 
    n_stim=len(stim_types_plotting), 
    latent_dims=latent_dims, 
    n_shuffles=100 # 1000
)

# --- Plotting Panel 2 ---
fig, ax = plt.subplots(figsize=(4*cm, 4*cm))

dims = np.arange(1, latent_dims+1)

# Plot True Correlations
ax.plot(dims, true_rs, marker='o', color=color_dSC, linewidth=2, label='True')

# Plot Null Distribution (95% Confidence Interval)
null_mean = np.mean(null_rs_dist, axis=0)
null_upper = np.percentile(null_rs_dist, 97.5, axis=0)
null_lower = np.percentile(null_rs_dist, 2.5, axis=0)

ax.plot(dims, null_mean, linestyle='--', color='gray', label='Null')
ax.fill_between(dims, null_lower, null_upper, color='gray', alpha=0.3)

ax.set_xticks(dims)
ax.set_xlabel('Canonical Dimension (CV)')
ax.set_ylabel('Mean Pairwise Correlation (r)')
ax.set_title('Cross-Session Alignment (dSC)')
ax.set_ylim([0, 1])
ax.legend(bbox_to_anchor=(1, 1.05), frameon=False, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set(yticks=[0,0.25,0.5,0.75,1])

plt.tight_layout()
plt.savefig(os.path.join(fig_path, f'MCCA_{region}_meanpairwisecorr.pdf'), dpi=300)
plt.show()

# Save p-values to a text file
with open(os.path.join(fig_path, f'MCCA_{region}_p_values.txt'), 'w') as f:
    f.write(f"MCCA Alignment P-values ({region}):\n")
    for dim in range(latent_dims):
        p_val = np.sum(null_rs_dist[:, dim] >= true_rs[dim]) / null_rs_dist.shape[0]
        print(f"CV{dim+1} p-value: {p_val}")
        f.write(f"CV{dim+1}: {p_val:.5f}\n")


#%% ===================== Link to freely moving experiments ========================
import pickle

def load_data(dataset_name):
    data_folder = rf'E:\results\CoordinatedCellTypes\data'
    data_path = os.path.join(data_folder, f'{dataset_name}.pkl')

    # If data is saved locally, load it
    if os.path.exists(data_path):
        print("Found data locally, loading...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

sSC_hfx = load_data('sSC_VR_freehfx_all_fr25_20260715')
dSC_hfx = load_data('dSC_VR_freehfx_all_fr25_20260715')
LP_hfx = load_data('LP_VR_freehfx_all_fr25_20260715')
sSC_free = load_data('sSC_VR_freefree_all_fr25_20260715')
dSC_free = load_data('dSC_VR_freefree_all_fr25_20260715')
LP_free = load_data('LP_VR_freefree_all_fr25_20260715')
SC_behav = load_data('SC_free_20260715')
LP_behav = load_data('LP_free_20260715')

dSC_hfx_frs, dSC_hfx_keys, dSC_hfx_depths = dSC_hfx['cluster_frs'], dSC_hfx['cluster_keys'], dSC_hfx['cluster_depths']
dSC_free_frs, dSC_free_keys, dSC_free_depths, dSC_stims = dSC_free['cluster_frs'], dSC_free['cluster_keys'], dSC_free['cluster_depths'], dSC_free['stims']
dSC_free_frs = {key: np.expand_dims(val, axis=1) for key, val in dSC_free_frs.items()}
SC_speed, SC_cornerdist, SC_stims = SC_behav['speed'], SC_behav['corner_distance'], SC_behav['stims']

speed, cornerdist, stims = SC_speed, SC_cornerdist, SC_stims

#%% --- Fix recordings shift (temporary fix until events_table issues are resolved)
for key in dSC_free_frs:
    if key in [('02583', '844LL', '20260204'), ('02585', '841L', '20260204'), ('02661', '860L', '20260410')]:
        dSC_free_frs[key] = dSC_free_frs[key][:,:,:,:-200]
    else:
        dSC_free_frs[key] = dSC_free_frs[key][:,:,:,200:]

#%% --- Combine all head-fixed recordings (freely moving is separately normalized ...)
baseline_slice = slice(50, 350)

dSC_hfx.keys()
dSC_hfx_frs_mats, dSC_hfx_frs_trials_mats, dSC_hfx_keys_mats, dSC_hfx_frs_means, dSC_hfx_frs_stds, dSC_hfx_frs_means_trials, dSC_hfx_frs_stds_trials = preprocess(dSC_hfx_frs, dSC_hfx_keys, trial_pre_from=2, trial_pre_to=10, stim_types=stim_types_plotting)
# Preprocess freely moving data
dSC_free_frs_mats = []
for i, exp_key in enumerate(dSC_free_frs):
    print(exp_key)
    print(dSC_free_frs[exp_key].shape)
    frs = dSC_free_frs[exp_key][:,:,:,:]
    print("frs shape", frs.shape)

    # Baseline subtraction
    baseline = np.mean(frs[:, :, :, baseline_slice], axis=-1, keepdims=True)
    frs_basesub = frs - baseline
    # TODO: try calc baseline across trials

    frs_mats_flat = flatten_frs_mat(frs_basesub)
    print("frs_mats_flat.shape", frs_mats_flat.shape)
    frs_mat_flat_std = (frs_mats_flat)/(dSC_hfx_frs_stds_trials[i]+1) # baseline subtracted and divided by head-fixed std
    dSC_free_frs_mats.append(frs_mat_flat_std)

# Filter out experiments with less than 15 cells ...
dSC_hfx_frs_mats = [mat for mat in dSC_hfx_frs_mats[:] if mat.shape[0]>=15] # 15
dSC_free_frs_mats = [mat for mat in dSC_free_frs_mats[:] if mat.shape[0]>=15] # 15
dSC_hfx_keys_mats = [mat for mat in dSC_hfx_keys_mats[:] if len(mat[0])>=15] # 15
region = 'dSC'
#%% --- Combine recordings
dSC_frs_mats_combined = dSC_frs_mats + dSC_hfx_frs_mats

#%% --- Fit mCCA
latent_dims = 5
res, mcca = run_mcca(dSC_frs_mats_combined, n_stim=len(stim_types_plotting), latent_dims=latent_dims,pca=True)

#%% --- Visualize latents
stim_types = ['EXPA', 'EXPA_60', 'EXPA_25', 'EXPW', 'EXPW_60','DIMM']
stim_types_labels = ['black expansion 100%','black expansion 60%', 'black expansion 25%', 'white expansion 100%', 'white expansion 60%', 'dimming']
stim_types_colors = [vlag_cmap[-1], vlag_cmap[-3], vlag_cmap[-4], vlag_cmap[0], vlag_cmap[2], tab_cmap[5]]

def run_pca(frs_mats, n_stim, n_components=3):
    pca_results = []
    pca_objects = []
    for mat in frs_mats:
        pca = PCA(n_components=n_components)
        # mat is (n_cells, n_stim * n_timepoints)
        pca_flat = pca.fit_transform(mat.T).T # (n_components, n_stim * n_timepoints)
        pca_unflat = unflatten_frs_mat(pca_flat, n_stim=n_stim, n_trials=1, n_timepoints=800)
        pca_results.append(pca_unflat)
        pca_objects.append(pca)
    return pca_results, pca_objects

#%% --- Get reconstuction error for freely moving experiments
frs_mats_transpose = [mat.T for mat in dSC_frs_mats_combined]
transformed_views = mcca.transform(frs_mats_transpose) # List of 24 arrays, each (4800, 3)

zscored_views = []
zscored_means = []
zscored_stds = []
for session_cv in transformed_views:
    session_cv_mean = np.mean(session_cv, axis=0)
    print(session_cv_mean.shape)
    session_cv_std = np.std(session_cv, axis=0)
    session_zscored = (session_cv - session_cv_mean)/(session_cv_std+1)
    zscored_views.append(session_zscored)
    zscored_means.append(session_cv_mean)
    zscored_stds.append(session_cv_std)

#%% --- Fit global decoder of threat
n_stim = 6
n_timepoints = 800
window_start = 205
window_end = 215

# Binary labels for the 6 stimuli (e.g., 1 for Threat, 0 for Non-Threat)
# Assuming index 0 and 1 are the "Black Looming" stimuli.
stim_labels_6 = np.array([1, 1, 0, 0, 0, 0]) 

n_sessions = len(zscored_views)
windowed_features = []

for session_cv in zscored_views:
    # Reshape from (4800, 3) to (6, 800, 3)
    reshaped_cv = session_cv.reshape(n_stim, n_timepoints, -1)
    
    # Slice the specific time window and take the mean across time
    # Shape becomes (6, 3)
    mean_response = np.mean(reshaped_cv[:, window_start:window_end, :], axis=1)
    windowed_features.append(mean_response)

# 3. Run Leave-One-Session-Out Decoding
accuracies = []
balanced_accuracies = []

for test_idx in range(n_sessions):
    # Test set: 6 samples from the left-out session
    test_X = windowed_features[test_idx]
    test_y = stim_labels_6
    
    # Train set: Pool the 6 samples from the remaining 23 sessions (138 samples total)
    train_X_list = [windowed_features[i] for i in range(n_sessions) if i != test_idx]
    train_X = np.vstack(train_X_list)
    
    # Replicate the 6 labels for the 23 sessions
    train_y = np.tile(stim_labels_6, n_sessions - 1)
    
    # Train Linear SVM
    clf = SVC(kernel='linear', class_weight='balanced')
    clf.fit(train_X, train_y)
    
    # Test on the left-out session
    preds = clf.predict(test_X)
    
    # Evaluate
    accuracies.append(accuracy_score(test_y, preds))
    balanced_accuracies.append(balanced_accuracy_score(test_y, preds))

# 4. Print Results
print(f"Mean Accuracy: {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
print(f"Mean Balanced Acc: {np.mean(balanced_accuracies):.3f} +/- {np.std(balanced_accuracies):.3f}")

#%%
# 1. Define Parameters and Labels
n_stim = 6
n_timepoints = 800
window_start = 205
window_end = 215
stim_labels_6 = np.array([1, 1, 0, 0, 0, 0]) 

n_sessions = len(zscored_views)

# 2. Extract Features
windowed_features = []
for session_cv in zscored_views:
    reshaped_cv = session_cv.reshape(n_stim, n_timepoints, -1)
    mean_response = np.mean(reshaped_cv[:, window_start:window_end, :], axis=1)
    windowed_features.append(mean_response)

# 3. Run Leave-One-Session-Out Decoding
balanced_accuracies = []

for test_idx in range(n_sessions):
    test_X = windowed_features[test_idx]
    test_y = stim_labels_6
    
    train_X = np.vstack([windowed_features[i] for i in range(n_sessions) if i != test_idx])
    train_y = np.tile(stim_labels_6, n_sessions - 1)
    
    clf = SVC(kernel='linear', class_weight='balanced')
    clf.fit(train_X, train_y)
    
    preds = clf.predict(test_X)
    balanced_accuracies.append(balanced_accuracy_score(test_y, preds))

balanced_accuracies = np.array(balanced_accuracies)

# 4. Statistical Testing (Wilcoxon signed-rank test against 0.5 chance level)
chance_level = 0.5
stat, p_value = wilcoxon(balanced_accuracies - chance_level, alternative='two-sided')

mean_acc = np.mean(balanced_accuracies)
std_acc = np.std(balanced_accuracies)

print(f"Mean Balanced Acc: {mean_acc:.3f} +/- {std_acc:.3f}")
print(f"Wilcoxon statistic: {stat}, p-value: {p_value:.4e}")

# 5. Plotting Distribution of Accuracies across Mice
fig, ax = plt.subplots(figsize=(1*cm, 3*cm))
data = [balanced_accuracies]
colors = [color_dSC]
# Add individual points
for i, d in enumerate(data):
    y = d
    x = np.random.normal(i, 0.1, size=len(y))
    ax.plot(x, y, c=colors[i], marker='.', linestyle='', alpha=0.3)

sb.pointplot(data, ax=ax, palette=[color_dSC], errorbar='ci')
ax.set_xticklabels(['dSC'])
ax.set(yticks=[0,0.25,0.5,0.75,1])
ax.set_ylabel('Balanced Accuracy')
ax.set_title('Decoding Performance')
ax.set_ylim([0, 1.01])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0.5, c='k', linestyle='--')
plt.savefig(os.path.join(fig_path, 'MCCA_decoding_pointplot.pdf'), dpi=300)

plt.show()

import numpy as np
from scipy.stats import bootstrap

# Assuming 'balanced_accuracies' is your 1D numpy array of 24 accuracies
mean_acc = np.mean(balanced_accuracies)

# Run a non-parametric bootstrap to get the 95% Confidence Interval
bootstrap_result = bootstrap((balanced_accuracies,), np.mean, confidence_level=0.95, method='BCa')
ci_lower, ci_upper = bootstrap_result.confidence_interval

print(f"Mean Accuracy: {mean_acc:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")

#%% ---------------------

# clf = SVC(kernel='line
# Plot clf coefs
fig, ax = plt.subplots(figsize=(4*cm, 4*cm))
coefs = clf.coef_[0]
ax.bar([f'{i+1}' for i in range(latent_dims)], coefs, color='gray')
ax.set_ylabel('SVM Weight')
ax.set_xlabel('Canonical Dimension (CV)')
ax.set_title('Threat Decoder Weights')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.tight_layout()
# plt.savefig()
plt.savefig(os.path.join(fig_path, f'MCCA_{region}_SVM_weights.pdf'), dpi=300)
plt.show()

 #%% --- Transform freely moving trials into latent space

def project_freely_moving(mcca, original_hf_mats, fm_mat, fm_index):
    # Create a copy of the original list to serve as a template
    mixed_mats = []
    for mat in original_hf_mats:
        print(mat.shape)
        new_mat = np.zeros(shape=(mat.shape[0], fm_mat.shape[1])).T
        mixed_mats.append(new_mat)

    mixed_mats[fm_index] = fm_mat.T
    
    # Transform the mixed list
    mixed_transformed = mcca.transform(views=mixed_mats)
    
    # Extract only the transformed freely-moving data
    return mixed_transformed[fm_index].T

transformed_trials_all = []
transformed_keys = []
for idx, (key, mat) in enumerate(zip(dSC_free_frs.keys(), dSC_free_frs_mats)):
    # Unflatten
    print(dSC_free_frs[key].shape[2])
    print(mat.shape)
    mat_unflat = unflatten_frs_mat(mat, n_stim=dSC_free_frs[key].shape[2], n_trials=1, n_timepoints=dSC_free_frs[key].shape[3])
    print(mat_unflat.shape)

    # Transform
    transformed_trials = []
    for i in range(mat_unflat.shape[1]):
        trial = project_freely_moving(mcca, dSC_frs_mats_combined, mat_unflat[:,i,0,:], len(dSC_frs_mats)+idx)
        transformed_trials.append(trial)
        
    transformed_trials_all.append(np.array(transformed_trials))
    transformed_keys.append(key)

#%%
# z-score using head-fixed data!...
zscored_free_views = []
for i, session_cv in enumerate(transformed_trials_all):
    session_zscored = (session_cv.swapaxes(0,1) - zscored_means[len(dSC_frs_mats)+i][:,None,None])/(zscored_stds[len(dSC_frs_mats)+i][:,None,None]+1)
    zscored_free_views.append(session_zscored)

#%%
# Run clf on freely moving data
decision_distances = []
for i, session_cv in enumerate(zscored_free_views):
    trial_decision_distances = []
    # Get distance to the decision boundary for each timepoint/stimulus
    # clf.decision_function returns the signed distance to the hyperplane
    for trial_idx in range(session_cv.shape[1]):
        # TODO: check!
        # Apply moving binning (binsize=10) on session_cv[:,trial_idx,:]
        session_cv_binned = ndimage.uniform_filter1d(session_cv, size=10, axis=-1)
        print(session_cv_binned.shape)
        dist = clf.decision_function(session_cv_binned[:,trial_idx,:].T)
        trial_decision_distances.append(dist)
    decision_distances.append(np.array(trial_decision_distances))

#%%
exp_idx = 5
key = transformed_keys[exp_idx]
exp_trials_dec = decision_distances[exp_idx]
exp_trials_latents = zscored_free_views[exp_idx] # transformed_trials_all[exp_idx]

session_baseline_activity = exp_trials_dec[:, baseline_slice].flatten()

# 2. Calculate the 99th percentile noise floor
empirical_threshold = np.percentile(session_baseline_activity, 99)
empirical_threshold2 = np.percentile(session_baseline_activity, 1)
print(f"Session {exp_idx} Empirical Threshold: {empirical_threshold:.2f}")
print(exp_trials_dec.shape)
exp_trial_idxs = [i for i, stim in enumerate(stims[key]) if stim in ['EXPA', 'EXPW', 'DIMM']]

fig, axs = plt.subplots(5, len(exp_trial_idxs), figsize=(len(exp_trial_idxs), 3), sharey='row', sharex=True)

for i, trial_idx in enumerate(exp_trial_idxs):
    stim = stims[key][trial_idx]
    trial_dec = exp_trials_dec[trial_idx]
    trial_latents = exp_trials_latents[:,trial_idx]
    axs[0,i].plot(np.arange(0, len(speed[key][trial_idx])/60, 1/60), speed[key][trial_idx], c='k')
    for j in range(3):
        axs[j+1,i].plot(np.arange(0, 2000/200, 1/200), trial_latents[j,:2000], c='k')
    axs[4,i].plot(np.arange(0, 2000/200, 1/200), trial_dec[:2000], c='k')
    for t in [2, 3, 4]:
        for ax in axs[:, i]:
            ax.axvspan(t, t + 0.5, color='gray', alpha=0.2, lw=0)


    axs[0,i].set_title(stim)
    axs[4,i].axhline(0, color='k', linestyle='--')
    axs[4,0].set(xticks=[0,2,4,6,8])
axs[0,0].set(ylabel='running\nspeed')
axs[1,0].set(ylabel='CV1')
axs[2,0].set(ylabel='CV2')
axs[3,0].set(ylabel='CV3')
axs[4,0].set(ylabel='SVM\ndecision')
plt.suptitle(key)
plt.tight_layout()
# plt.savefig(os.path.join(fig_path, f'MCCA_freely_moving_decision_boundary_{key[0]}_{key[1]}_{key[2]}.pdf'), dpi=300)
plt.show()

#%% --- Visualize head-fixed trials!

# 1. Select a head-fixed session to visualize
hf_exp_idx = 0 
global_idx = len(dSC_frs_mats) + hf_exp_idx  # Index in the combined MCCA dataset
hf_key = dSC_hfx_keys_mats[hf_exp_idx][0][0]

# dSC_hfx_frs_trials_mats holds the flattened trials for the head-fixed dataset
hf_trials_mat = dSC_hfx_frs_trials_mats[hf_exp_idx]

# Reconstruct shape: n_stim, n_trials, n_timepoints
n_stim_hf = len(stim_types_plotting)
n_trials_hf = 8 # Derived from preprocess logic (trial_pre_from=2 to trial_pre_to=10)
n_timepoints_hf = 800

hf_trials_unflat = unflatten_frs_mat(hf_trials_mat, n_stim=n_stim_hf, n_trials=n_trials_hf, n_timepoints=n_timepoints_hf)

# 2. Project single trials into MCCA space
hf_transformed_trials = np.zeros((n_stim_hf, n_trials_hf, latent_dims, n_timepoints_hf))

for i in range(n_stim_hf):
    for j in range(n_trials_hf):
        trial_mat = hf_trials_unflat[:, i, j, :]
        # Project single trial using the existing transformation mapping
        trial_proj = project_freely_moving(mcca, dSC_frs_mats_combined, trial_mat, global_idx)
        hf_transformed_trials[i, j] = trial_proj

# 3. Z-score the latents using the session's overall mean and std
sess_mean = zscored_means[global_idx][:, None]
sess_std = zscored_stds[global_idx][:, None]

hf_zscored_trials = (hf_transformed_trials - sess_mean) / (sess_std + 1)

# 4. Compute SVM decision boundary distance for each trial
hf_decision_distances = np.zeros((n_stim_hf, n_trials_hf, n_timepoints_hf))

for i in range(n_stim_hf):
    for j in range(n_trials_hf):
        trial_latents = hf_zscored_trials[i, j]
        # Moving average binning (matching the freely moving pipeline)
        trial_latents_binned = ndimage.uniform_filter1d(trial_latents, size=50, axis=-1)
        dist = clf.decision_function(trial_latents_binned.T)
        hf_decision_distances[i, j] = dist

# 5. Plotting Configuration
# Select stimuli to plot (e.g., EXPA, EXPW, DIMM map to indices 0, 3, 5 based on stim_types_plotting)
stim_idxs_to_plot = [0,3,5]
trial_to_plot = 0 # Selecting an example trial (e.g., first trial)

fig, axs = plt.subplots(4, len(stim_idxs_to_plot), figsize=(len(stim_idxs_to_plot)*1.5, 3.5), sharey='row', sharex=True)

for col_idx, stim_idx in enumerate(stim_idxs_to_plot):
    stim_label = stim_types_labels[stim_types_plotting[stim_idx]]
    trial_dec = hf_decision_distances[stim_idx, trial_to_plot]
    trial_latents = hf_zscored_trials[stim_idx, trial_to_plot]
    
    time_vector = np.arange(0, n_timepoints_hf/200, 1/200)
    
    # Plot latents (CV1, CV2, CV3)
    for j in range(3):
        axs[j, col_idx].plot(time_vector, trial_latents[j, :], c='k')
        
    # Plot SVM decision boundary
    axs[3, col_idx].plot(time_vector, trial_dec, c='k')
    
    # Add shading for stimulus (Starts at index 200 = 1.0s)
    for t in [1, 1+0.85, 1+0.85*2]: 
        for ax in axs[:, col_idx]:
            ax.axvspan(t, t + 0.5, color='gray', alpha=0.2, lw=0)
            
    axs[0, col_idx].set_title(stim_label)
    axs[3, col_idx].axhline(0, color='k', linestyle='--')
    
# Clean up axes and attach labels
axs[0, 0].set(ylabel='CV1')
axs[1, 0].set(ylabel='CV2')
axs[2, 0].set(ylabel='CV3')
axs[3, 0].set(ylabel='SVM\ndecision')

axs[3, 0].set(xticks=[0, 1, 2, 3, 4])
for col_idx in range(len(stim_idxs_to_plot)):
    axs[3, col_idx].set_xlabel('Time (s)')

plt.suptitle(f"Head-fixed Latents & Decoder | Session {global_idx}")
plt.tight_layout()
plt.savefig(os.path.join(fig_path, f'MCCA_headfixed_decision_boundary_{key[0]}_{key[1]}_{key[2]}.pdf'), dpi=300)
plt.show()


#%% --- Get escape metrics --- 
exp_idx = 0
print(key)

peak_speeds = []
peak_dec = []
mean_dec = []
time_to_peak_dec = []
time_to_peak_speed = []
dt_peak_speed = []
for exp_idx, key in enumerate(dSC_free_frs.keys()):
    exp_trial_idxs = [i for i, stim in enumerate(stims[key]) if stim in ['EXPA', 'EXPW', 'DIMM']] # all EXPA in this experiments have an escape response
    if len(exp_trial_idxs) == 0:
        exp_trial_idxs = [i for i, stim in enumerate(stims[key]) if stim in ['EXPAL','EXPWL','DIMML']] # all EXPA in this experiments have an escape response
    print(key)
    key = transformed_keys[exp_idx]
    sess_speed = speed[key]
    sess_cornerdist = cornerdist[key]
    sess_running_bouts, sess_stopping_bouts, sess_escape_bouts = behavior_ethogram.get_ethogram(sess_speed, sess_cornerdist)
    # Transform sess_escape_bouts to binary array indicating whether there was an escape or not
    sess_escape_bouts_binary = np.array([len(bout)>0 for bout in sess_escape_bouts])
    print(sess_escape_bouts_binary[exp_trial_idxs])

    print(key)
    exp_trials_dec = decision_distances[exp_idx]
    exp_trials_latents = zscored_free_views[exp_idx] # transformed_trials_all[exp_idx]

    for i, trial_idx in enumerate(exp_trial_idxs):
        # filter escapes
        if not sess_escape_bouts_binary[trial_idx]:
            continue
        stim = stims[key][trial_idx]    
        trial_dec = exp_trials_dec[trial_idx]
        trial_latents = exp_trials_latents[:,trial_idx]
        trial_speed = speed[key][trial_idx]
        print(trial_dec.shape)
        peak_speeds.append(np.max(trial_speed[60*2:60*5]))
        time_to_peak_speed.append(np.argmax(trial_speed[60*2:60*5])/60)
        t_peak_speed = time_to_peak_speed[-1]+2
        t_peak_speed_min = time_to_peak_speed[-1]+2-2
        print(t_peak_speed)
        peak_dec.append(np.max(trial_dec[int(200*t_peak_speed_min):int(200*t_peak_speed)]))
        mean_dec.append(np.mean(trial_dec[int(200*t_peak_speed_min):int(200*t_peak_speed)]))
        time_to_peak_dec.append(t_peak_speed_min-2+np.argmax(trial_dec[int(200*t_peak_speed_min):int(200*t_peak_speed)])/200)
        dt_peak_speed.append(time_to_peak_speed[-1] - time_to_peak_dec[-1])

fig, axs = plt.subplots(3, 2, figsize=(8*cm, 10*cm))
axs[0,0].scatter(peak_dec, peak_speeds)
axs[0,0].set_ylabel('Peak Speed')
axs[0,0].set_xlabel('Peak Decision')
axs[1,0].scatter(peak_dec, time_to_peak_speed)
axs[1,0].set_ylabel('Time to Peak Speed')
axs[1,0].set_xlabel('Peak Decision')
axs[0,1].scatter(mean_dec, peak_speeds)
axs[0,1].set_ylabel('Peak Speed')
axs[0,1].set_xlabel('Mean Decision')
axs[1,1].scatter(time_to_peak_dec, time_to_peak_speed)
axs[1,1].set_ylabel('Time to Peak Speed')
axs[1,1].set_xlabel('Time to Peak Decision')
axs[2,1].scatter(mean_dec, dt_peak_speed)
axs[2,1].set_ylabel('DT to Peak Speed')
axs[2,1].set_xlabel('Mean Decision')
axs[2,0].scatter(peak_dec, dt_peak_speed)
axs[2,0].set_ylabel('DT to Peak Speed')
axs[2,0].set_xlabel('Peak Decision')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(6*cm, 3*cm))
axs[0].scatter(peak_dec, peak_speeds, c='k')
axs[0].set_ylabel('Peak Speed')
axs[0].set_xlabel('Peak Decision')
axs[0].set(ylim=(0,None))

axs[1].scatter(peak_dec, dt_peak_speed, c='k')
axs[1].set_ylabel('DT to Peak Speed')
axs[1].set_xlabel('Peak Decision')
plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'MCCA_corrs_dec_speed.pdf'))
plt.show()


#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr

def fit_robust_and_correlate(x, y):
    """
    Fits a Robust Linear Model (RLM) for the line coordinates, 
    and calculates the Pearson correlation coefficient and its p-value.
    """
    # Statsmodels needs an explicit column for the intercept (constant)
    X = sm.add_constant(x)
    
    # Fit the robust linear model (HuberT is default) for plotting the line
    model = sm.RLM(y, X).fit()
    
    # Generate line coordinates for plotting
    x_line = np.linspace(min(x), max(x), 100)
    X_line = sm.add_constant(x_line)
    y_line = model.predict(X_line)
    
    # Calculate standard Pearson correlation coefficient and p-value
    r, p_val = pearsonr(x, y)
    
    return x_line, y_line, r, p_val

fig, axs = plt.subplots(1, 2, figsize=(7.5*cm, 4*cm)) 

# --- Plot 1: Peak Speed ---
axs[0].scatter(peak_dec, peak_speeds, c='k', s=3)
x_line1, y_line1, r1, p1 = fit_robust_and_correlate(peak_dec, peak_speeds)
axs[0].plot(x_line1, y_line1, color='red', linewidth=1.5)

axs[0].set_ylabel('Peak Speed')
axs[0].set_xlabel('Peak Decision')
axs[0].set(ylim=(0, None))

# --- Plot 2: DT to Peak Speed ---
axs[1].scatter(peak_dec, dt_peak_speed, c='k', s=3)
x_line2, y_line2, r2, p2 = fit_robust_and_correlate(peak_dec, dt_peak_speed)
axs[1].plot(x_line2, y_line2, color='red', linewidth=1.5)

axs[1].set_ylabel('DT to Peak Speed')
axs[1].set_xlabel('Peak Decision')

print(f"Plot 1: r = {r1:.3f}, p = {p1:.3f}")
print(f"Plot 2: r = {r2:.3f}, p = {p2:.3f}")

plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'MCCA_corrs_dec_speed_stats.pdf'))
plt.show()

#%% --- Combined ROC
y_true_list = []
y_scores_list = []
for exp_idx, key in enumerate(dSC_free_frs.keys()):
    print(key)

    sess_speed = speed[key]
    sess_cornerdist = cornerdist[key]
    sess_running_bouts, sess_stopping_bouts, sess_escape_bouts = behavior_ethogram.get_ethogram(sess_speed, sess_cornerdist)
    len(stims[key]), len(sess_escape_bouts)

    exp_trial_idxs = [i for i, stim in enumerate(stims[key]) if stim in ['EXPA','EXPW','DIMM']] # all EXPA in this experiments have an escape response
    if len(exp_trial_idxs) == 0:
        exp_trial_idxs = [i for i, stim in enumerate(stims[key]) if stim in ['EXPAL','EXPWL','DIMML']] # all EXPA in this experiments have an escape response
    
    # Transform sess_escape_bouts to binary array indicating whether there was an escape or not
    sess_escape_bouts_binary = np.array([len(bout)>0 for bout in sess_escape_bouts])
    sess_escape_bouts_binary[exp_trial_idxs]

    sess_decs = decision_distances[exp_idx]

    from sklearn.metrics import roc_curve, auc

    # Flatten the decision distances to find max value per trial in the stimulus window
    # Window: 2s to 6s (indices 400 to 1200 at 200Hz)
    y_true = sess_escape_bouts_binary[exp_trial_idxs]
    y_scores = [np.max(sess_decs[i, 200*2:200*5]) for i in exp_trial_idxs]
    y_true_list.append(y_true)
    y_scores_list.append(y_scores)

print("N experiments:", len(y_true_list), len(y_scores_list))
y_true_list = np.concatenate(y_true_list)
y_scores_list = np.concatenate(y_scores_list)
print("N trials:", len(y_true_list), len(y_scores_list))

fpr, tpr, thresholds = roc_curve(y_true_list, y_scores_list)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(4*cm,4*cm))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, f'MCCA_freely_moving_ROC_all.pdf'))
plt.show()

from sklearn.metrics import roc_auc_score
# Calculate the ROC AUC
roc_auc = roc_auc_score(y_true_list, y_scores_list)
print(f"ROC AUC: {roc_auc:.4f}")

#%%
from sklearn.metrics import confusion_matrix

# 1. Find the best threshold using Youden's J statistic (maximizes TPR - FPR)
optimal_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold (Youden's J): {best_threshold:.4f}")

# 2. Binarize the predicted scores using the optimal threshold
y_pred = (y_scores_list >= best_threshold).astype(int)

# 3. Calculate the confusion matrix
conf_mat = confusion_matrix(y_true_list, y_pred)
print("Confusion Matrix:\n", cm)

# 4. Plot the confusion matrix
fig, ax = plt.subplots(figsize=(4*cm, 4*cm)) # Assuming 'cm' is defined in your environment as 1/2.54 for inches

conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

sb.heatmap(conf_mat_norm.T, annot=True, fmt='.2f', cmap='Blues', 
           xticklabels=['Threat', 'No Threat'], 
           yticklabels=['Escape', 'No Escape'], ax=ax, cbar=False)
ax.set_xlabel('SVM prediction')
ax.set_ylabel('Behavior')

plt.tight_layout()
plt.savefig(os.path.join(fig_path, f'MCCA_freely_moving_confmat.pdf'))

plt.show()

#%%
from sklearn.metrics import accuracy_score

# Calculate overall accuracy
accuracy = accuracy_score(y_true_list, y_pred)
n_samples = len(y_true_list)

# Calculate 95% confidence interval (Z-score for 95% is ~1.96)
z_score = 1.96
margin_of_error = z_score * np.sqrt((accuracy * (1 - accuracy)) / n_samples)

ci_lower = accuracy - margin_of_error
ci_upper = accuracy + margin_of_error

print(f"Accuracy: {accuracy:.4f} ± {margin_of_error:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")