import numpy as np
import copy

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

def preprocess(frs_dict, keys_used, stim_types, t_start=200, t_end=1000, base_start=50, base_end=350, precal_stds=None, precal_stds_trials=None, std_norm=True, basesub=True, trial_slice=(2,10)):
    trial_pre_from = trial_slice[0]
    trial_pre_to = trial_slice[1]
    frs_all = copy.deepcopy(frs_dict)
    
    frs_mats = []
    frs_trials_mats = []
    keys_mats = []
    frs_stds = []
    frs_stds_trials = []

    for exp_idx, exp_key in enumerate(frs_all):
        stim_frs_all = []
        stim_frs_trials_all = []
        stim_keys_all = []
        for stim_idx in stim_types:
            # baseline subtraction
            print(frs_all[exp_key].shape)
            frs = frs_all[exp_key][:,stim_idx,:]
            print("frs shape", frs.shape)
            if basesub:
                frs_basesub = baseline_subtract(frs, baseline_start=base_start, baseline_end=base_end)[:,:,t_start:t_end] # TODO: be careful, removing 200 ...
            else:
                frs_basesub = frs[:,:,t_start:t_end]

            # trial average
            stim_frs_trials_all.append(frs_basesub[:, trial_pre_from:trial_pre_to])
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

        if precal_stds is None:
            stds = np.std(frs_mats_flat, axis=1, keepdims=True)
        else:
            stds = precal_stds[exp_idx]
        print("frs_mats_flat.shape", frs_mats_flat.shape)
        if precal_stds_trials is None:
            stds_trials = np.std(frs_trials_mats_flat, axis=1, keepdims=True)
        else:
            stds_trials = precal_stds_trials[exp_idx]
        
        if std_norm:
            frs_mat_flat_std = (frs_mats_flat)/(stds+1)
            frs_trials_mat_flat_std = (frs_trials_mats_flat)/(stds_trials+1)
        else:
            frs_mat_flat_std = (frs_mats_flat)
            frs_trials_mat_flat_std = (frs_trials_mats_flat)
        frs_mats.append(frs_mat_flat_std)
        frs_trials_mats.append(frs_trials_mat_flat_std)
        frs_stds.append(stds)
        frs_stds_trials.append(stds_trials)
        
        keys_mats.append(stim_keys_all)
    return frs_mats, frs_trials_mats, keys_mats, frs_stds, frs_stds_trials