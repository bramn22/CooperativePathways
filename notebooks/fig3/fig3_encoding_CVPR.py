#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import copy
from sklearn.decomposition import PCA
from notebooks.CoordinatedCellTypes.final.fig_params import *

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

def preprocess(frs_dict, keys_used, stim_types_plotting):
    trial_pre_from = 2
    trial_pre_to = 10
    frs_all = copy.deepcopy(frs_dict)
    
    frs_mats = []
    frs_mats1 = []
    frs_mats2 = []
    frs_trials_mats = []
    keys_mats = []

    for exp_key in frs_all:
        stim_frs_all = []
        stim_frs_all1 = []
        stim_frs_all2 = []
        stim_frs_trials_all = []
        stim_keys_all = []
        for stim_idx in stim_types_plotting:
            # baseline subtraction
            print(frs_all[exp_key].shape)
            frs = frs_all[exp_key][:,stim_idx,:]
            print("frs shape", frs.shape)
            frs_basesub = baseline_subtract(frs)[:,:,200:1000]
            frs_smooth = frs_basesub

            # trial average
            stim_frs_trials_all.append(frs_smooth[:,trial_pre_from:trial_pre_to])
            frs_mean = np.mean(frs_smooth[:, trial_pre_from:trial_pre_to], axis=1)
            frs_mean = frs_mean[:, np.newaxis, :]
            frs_mean1 = np.mean(frs_smooth[:, trial_pre_from:trial_pre_to][:,::2], axis=1)
            frs_mean1 = frs_mean1[:, np.newaxis, :]
            frs_mean2 = np.mean(frs_smooth[:, trial_pre_from:trial_pre_to][:,1::2], axis=1)
            frs_mean2 = frs_mean2[:, np.newaxis, :]

            stim_frs_all.append(frs_mean)
            stim_frs_all1.append(frs_mean1)
            stim_frs_all2.append(frs_mean2)
            stim_keys_all.append(np.array(keys_used[exp_key]))
        if len(stim_frs_all) == 0:
            continue

        stim_frs_all = np.array(stim_frs_all)
        stim_frs_all = stim_frs_all.swapaxes(0,1)  # shape: n_cells, n_stim, 1, n_timepoints
        stim_frs_all1 = np.array(stim_frs_all1)
        stim_frs_all1 = stim_frs_all1.swapaxes(0,1)  # shape: n_cells, n_stim, 1, n_tim
        stim_frs_all2 = np.array(stim_frs_all2)
        stim_frs_all2 = stim_frs_all2.swapaxes(0,1)  # shape: n_cells, n_stim, 1, n_tim
        print(stim_frs_all.shape, stim_frs_all1.shape, stim_frs_all2.shape)

        stim_frs_trials_all = np.array(stim_frs_trials_all)
        stim_frs_trials_all = stim_frs_trials_all.swapaxes(0,1)  # shape: n_cells, n_stim, n_trials, n_timepoints

        frs_mats_flat = flatten_frs_mat(stim_frs_all)
        frs_mats_flat1 = flatten_frs_mat(stim_frs_all1)
        frs_mats_flat2 = flatten_frs_mat(stim_frs_all2)
        frs_trials_mats_flat = flatten_frs_mat(stim_frs_trials_all)
        print("frs_mats_flat.shape", frs_mats_flat.shape)
        frs_mat_flat_std = (frs_mats_flat)/(np.std(frs_mats_flat, axis=1, keepdims=True)+1)
        frs_mat_flat_std1 = (frs_mats_flat1)/(np.std(frs_mats_flat1, axis=1, keepdims=True)+1)
        frs_mat_flat_std2 = (frs_mats_flat2)/(np.std(frs_mats_flat2, axis=1, keepdims=True)+1)

        frs_trials_mat_flat_std = (frs_trials_mats_flat)/(np.std(frs_trials_mats_flat, axis=1, keepdims=True)+1)

        frs_mats.append(frs_mat_flat_std) #np.array(stim_frs_all)
        frs_mats1.append(frs_mat_flat_std1) #np.array(stim_frs_all)
        frs_mats2.append(frs_mat_flat_std2) #np.array(stim_frs_all
        frs_trials_mats.append(frs_trials_mat_flat_std)
        
        keys_mats.append(stim_keys_all)
        # break
    return frs_mats, frs_mats1, frs_mats2, frs_trials_mats, keys_mats

#%%
vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)

def run_cvpca(mats_split1, mats_split2, n_cells=None, concat=True):
    """
    Computes cross-validated eigenvalues from two independent trial splits.
    """
    if concat:
        X1 = np.concatenate(mats_split1, axis=0)
        X2 = np.concatenate(mats_split2, axis=0)
    else:
        X1 = mats_split1
        X2 = mats_split2
        
    if n_cells is not None:
        # We must sample the same cells for both splits
        idx = np.random.choice(X1.shape[0], n_cells, replace=False)
        X1 = X1[idx, :]
        X2 = X2[idx, :]
        
    # Standard PCA mean-centers the features (cells). 
    # Since our shape is (n_cells, n_timepoints), we subtract the mean across time.
    X1 = X1 - X1.mean(axis=1, keepdims=True)
    X2 = X2 - X2.mean(axis=1, keepdims=True)
    
    n_timepoints = X1.shape[1]
    
    # Calculate the cross-covariance matrix (n_cells x n_cells)
    C_cv = (X1 @ X2.T) / (n_timepoints - 1)
    
    # Symmetrize the matrix to ensure real eigenvalues
    C_sym = (C_cv + C_cv.T) / 2
    
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eigh(C_sym)
    
    # Sort in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Cross-validation isolates pure signal. If a dimension is pure noise, 
    # its cv-eigenvalue will hover around 0 and sometimes dip slightly negative.
    # We clip negative eigenvalues to 0 to prevent PR math errors.
    eigenvalues[eigenvalues < 0] = 0
    
    return eigenvalues

def participation_ratio(eigenvalues):
    """Calculate the participation ratio given eigenvalues."""
    numerator = (np.sum(eigenvalues))**2
    denominator = np.sum(eigenvalues**2)
    return numerator / denominator

def run_bootstrapped_cv_participation_ratio(mats_split1, mats_split2, n_bootstrap=100, n_cells=200):
    PRs = []
    for _ in range(n_bootstrap):
        # Get cross-validated eigenvalues for a random subset of 200 cells
        cv_eigenvalues = run_cvpca(mats_split1, mats_split2, n_cells=n_cells, concat=True)
        
        eff_dim = participation_ratio(cv_eigenvalues)
        PRs.append(eff_dim)
        
    return np.array(PRs)

def run_pca(frs_mats, n_stim, n_cells=None, concat=True):
    if concat:
        frs_mats_flat_concat = np.concatenate(frs_mats, axis=0)
    else:
        frs_mats_flat_concat = frs_mats
    if n_cells is not None:
        # randomly sample cells without replacement
        frs_mats_flat_concat = frs_mats_flat_concat[np.random.choice(frs_mats_flat_concat.shape[0], n_cells, replace=False), :]

    # Apply PCA
    pca = PCA(n_components=None)
    frs_mat_flat_pca = pca.fit_transform(frs_mats_flat_concat.T).T  # shape: n_components, n_samples
    frs_mat_pca = unflatten_frs_mat(frs_mat_flat_pca, n_stim=n_stim, n_trials=1, n_timepoints=800)
    return pca, frs_mat_pca

def visualize_participation_ratio(pca, area):
    # get eigenvalues from PCA
    eigenvalues = pca.explained_variance_
    eff_dim = participation_ratio(eigenvalues)
    print(f'Effective dimensionality (Participation Ratio): {eff_dim}')

    # 2. Variance explained curve
    cumulative_variance_explained = np.cumsum(pca.explained_variance_ratio_)
    # Calculate how many components explain 80% of variance
    n_components_80 = np.argmax(cumulative_variance_explained >= 0.8) + 1
    print(f'Number of components explaining 80% variance: {n_components_80}')

    # Plot cumulative variance explained and participation ratio
    fig, ax = plt.subplots(figsize=(6*cm, 5*cm), constrained_layout=True)
    ax.plot(np.arange(1, len(cumulative_variance_explained)+1), cumulative_variance_explained, marker='o')
    ax.axhline(y=0.8, color='r', linestyle='--', label=f'80% Variance Explained: {n_components_80}')
    ax.axvline(x=eff_dim, color='g', linestyle='--', label=f'Participation Ratio: {eff_dim:.2f}')
    ax.set(xlim=[0, len(cumulative_variance_explained)+1], ylim=[0, 1.05])
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Variance Explained by PCA Components')
    ax.legend()
    ax.set(xlim=[0, 100])
    plt.suptitle(area)

    # plt.savefig(f'{area}_PCA_variance_explained.pdf', dpi=300)
    plt.show()

def visualize_cumul_var_expl_combined(pca_list, pca_labels, pca_colors, fig_path, log_scale=False):

    fig, ax = plt.subplots(figsize=(5*cm, 4*cm), constrained_layout=True)
    for pca, label, color in zip(pca_list, pca_labels, pca_colors):
        var_expl = pca.explained_variance_ratio_
    
        ax.plot(np.arange(1, len(var_expl)+1), var_expl, marker='o', label=label, c=color)
    
    if log_scale:
        ax.set_xscale('log')
    # ax.set(xlim=[0,15])
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Variance Explained')
    ax.legend()
    plt.savefig(os.path.join(fig_path, 'var_expl_combined.pdf'), bbox_inches='tight', dpi=300)
    plt.show()


    fig, ax = plt.subplots(figsize=(5*cm, 4*cm), constrained_layout=True)
    for pca, label, color in zip(pca_list, pca_labels, pca_colors):
        cum_var_expl = np.cumsum(pca.explained_variance_ratio_)
    
        ax.plot(np.arange(1, len(cum_var_expl)+1), cum_var_expl, marker='o', label=label, c=color)
    
    if log_scale:
        ax.set_xscale('log')
    # ax.set(xlim=[0,15])
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.legend()
    plt.savefig(os.path.join(fig_path, 'cumul_var_expl_combined.pdf'), bbox_inches='tight', dpi=300)
    plt.show()

def get_variance_bootstraps(frs_mats, n_stim, n_cells=30, n_bootstraps=1000):
    """
    Runs PCA bootstraps and returns the raw variance explained for each PC.
    """
    all_vars = []
    
    for _ in range(n_bootstraps):
        pca, _ = run_pca(frs_mats, n_stim, n_cells=n_cells, concat=True) #[cite: 1]
        var_expl = pca.explained_variance_ratio_ #[cite: 1]
        
        # Safeguard against rank deficiency: pad missing PCs with 0.0 variance
        if len(var_expl) < n_cells:
            pad_length = n_cells - len(var_expl)
            var_expl = np.pad(var_expl, (0, pad_length), constant_values=0.0)
            
        all_vars.append(var_expl[:n_cells])
        
    return np.array(all_vars)  # Shape: (n_bootstraps, n_cells)

def test_pc_variance_difference(var_boot_A, var_boot_B, name_A, name_B):
    """
    Calculates the 95% CI of the difference between two bootstrap distributions
    of individual PC variance explained and returns empirical p-values.
    """
    # Difference matrix: Shape (n_bootstraps, n_cells)
    diff_matrix = var_boot_B - var_boot_A

    # Calculate 95% CI of the difference at each PC 
    ci_low = np.percentile(diff_matrix, 2.5, axis=0)
    ci_high = np.percentile(diff_matrix, 97.5, axis=0)
    median_diff = np.median(diff_matrix, axis=0)

    # Empirical two-tailed p-value at each PC
    p_vals = np.minimum(np.mean(diff_matrix <= 0, axis=0), np.mean(diff_matrix >= 0, axis=0)) * 2

    # Check significance (0 is outside the CI)
    significant_pcs = np.where((ci_low > 0) | (ci_high < 0))[0] + 1 

    print(f"\n--- {name_B} vs {name_A} (Individual PC Variance Explained) ---")
    print(f"Significant PCs (alpha=0.05): {significant_pcs}")
    
    # Print detailed stats for the first 3 PCs
    for i in range(3):
        print(f"PC{i+1}: Median Diff = {median_diff[i]:.4f}, 95% CI = [{ci_low[i]:.4f}, {ci_high[i]:.4f}], p = {p_vals[i]:.4f}")

    return median_diff, ci_low, ci_high, p_vals, significant_pcs

def visualize_cumul_var_expl_combined_ci(bootstraps_list, pca_labels, pca_colors, fig_path=None, log_scale=False):
    """
    Plots the mean cumulative variance explained with shaded 95% confidence intervals 
    using pre-computed raw variance bootstraps.
    """
    fig, ax = plt.subplots(figsize=(4*cm, 4*cm), constrained_layout=True) #[cite: 1]
    
    for var_boot, label, color in zip(bootstraps_list, pca_labels, pca_colors):
        # 1. Convert raw variance to cumulative variance along the PC axis (axis=1)
        cum_var_boot = np.cumsum(var_boot, axis=1)
        
        # 2. Calculate mean and CIs across the bootstrap iterations (axis=0)
        mean_cum_var = np.mean(cum_var_boot, axis=0)
        ci_low = np.percentile(cum_var_boot, 2.5, axis=0)
        ci_high = np.percentile(cum_var_boot, 97.5, axis=0)
        
        x_axis = np.arange(1, len(mean_cum_var) + 1)
        
        # 3. Plotting
        if label == 'late':
            ax.plot(x_axis, mean_cum_var, marker='o', linestyle='--', markerfacecolor='none', markersize=3, label=label, c=color)
        else:
            ax.plot(x_axis, mean_cum_var, marker='o', markersize=3, label=label, c=color)
        ax.fill_between(x_axis, ci_low, ci_high, color=color, alpha=0.3, edgecolor='none')

    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.legend(loc='lower right')
    sb.despine() #[cite: 1]
    
    if fig_path:
        plt.savefig(os.path.join(fig_path, 'cumul_var_expl_combined_ci.pdf'), bbox_inches='tight', dpi=300) #[cite: 1]
    
    plt.show()

def visualize_var_expl_barplot(bootstraps_list, pca_labels, pca_colors, k=3, fig_path=None):
    """
    Plots a grouped bar chart of the variance explained for the first k Principal Components
    with 95% confidence intervals using pre-computed bootstrap distributions.
    """
    # Assuming `cm` is imported from fig_params as in your script[cite: 1]
    fig, ax = plt.subplots(figsize=(4*cm, 4*cm), constrained_layout=True) #[cite: 1]
    
    x = np.arange(k)  # Base x locations for the groups
    width = 0.8 / len(bootstraps_list)  # Width of individual bars based on number of groups
    
    for i, (var_boot, label, color) in enumerate(zip(bootstraps_list, pca_labels, pca_colors)):
        # 1. Extract only the first k principal components
        var_boot_k = var_boot[:, :k]
        
        # 2. Calculate mean and CIs across the bootstrap iterations (axis=0)
        mean_var = np.mean(var_boot_k, axis=0)
        ci_low = np.percentile(var_boot_k, 2.5, axis=0)
        ci_high = np.percentile(var_boot_k, 97.5, axis=0)
        
        # 3. Matplotlib's yerr expects relative error margins: [lower_errors, upper_errors]
        yerr = [mean_var - ci_low, ci_high - mean_var]
        
        # 4. Calculate offset to group bars side-by-side
        offset = (i - len(bootstraps_list)/2 + 0.5) * width
        
        # 5. Plot the bar and error bars
        if label == 'late':
            ax.bar(x + offset, mean_var, width, yerr=yerr, label=label, color=color, 
                capsize=2, edgecolor=color, fill=False, linewidth=1, error_kw={'elinewidth': 1, 'alpha': 0.8})
        else:
            ax.bar(x + offset, mean_var, width, yerr=yerr, label=label, color=color, 
                capsize=2, edgecolor=color, fill=True, linewidth=1, error_kw={'elinewidth': 1, 'alpha': 0.8})

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{j+1}' for j in range(k)])
    ax.legend(loc='upper right', frameon=False, fontsize='small')
    sb.despine() #[cite: 1]
    
    if fig_path:
        plt.savefig(os.path.join(fig_path, f'var_expl_top_{k}_barplot.pdf'), bbox_inches='tight', dpi=300) #[cite: 1]
    
    plt.show()

#%%
# visualize
def plot_PRs(PR_lists, PR_labels, PR_colors, fig_path):
    fig, ax = plt.subplots(1, 1, figsize=(len(PR_lists)*cm, 4*cm))
    data_resp = PR_lists
    # data_resp = [sSC_sparsity_means, dSC_sparsity_means, LP_sparsity_means]

    labels = PR_labels
    colors = PR_colors
    shift = 0.13

    # Plot pointplot (95th confidence interval) previous: (median + IQR)
    for i, d in enumerate(data_resp):
        median = np.median(d)
        q1, q3 = np.percentile(d, [2.5, 97.5])
        ax.errorbar(i + shift, median, yerr=[[median-q1], [q3-median]], fmt='o', color=colors[i], 
                    elinewidth=1, capsize=2, markersize=4, zorder=2)

    # Plot jittered scatter (stripplot) to the right of the boxplots
    for i, d in enumerate(data_resp):
        # x = np.random.normal(i - shift, 0.04, size=len(d))
        x = np.random.uniform(i - shift-0.08, i - shift+0.08, size=len(d))
        ax.scatter(x, d, color=colors[i], s=1, alpha=0.2, zorder=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks([5,10,15,20])
    ax.set_ylabel('Participation Ratio')
    # ax.set_ylim([0, 1.1])
    sb.despine()
    plt.savefig(os.path.join(fig_path, 'lifetime_sparsity_means_raincloud_simple.pdf'), bbox_inches='tight', dpi=300)
    plt.show()

#%% === Final 95% CI statistics ===
def test_pseudo_population_difference(PRs_A, PRs_B, name_A, name_B):
    """
    Computes the 95% CI of the difference between two independent bootstrap distributions.
    """
    # Calculate all 10,000 pairwise differences (B - A)
    diff_matrix = PRs_B[:, None] - PRs_A[None, :]
    diffs = diff_matrix.flatten()

    # Calculate the 95% CI of the DIFFERENCE itself
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    median_diff = np.median(diffs)
    
    p_val_empirical = min(np.mean(diffs <= 0), np.mean(diffs >= 0)) * 2
    print(f"--- {name_B} vs {name_A} ---")
    print(f"Median Difference: {median_diff:.3f}")
    print(f"95% CI of Difference: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Empirical p-value: {p_val_empirical:.4f}")
    
    if ci_low > 0 or ci_high < 0:
        print(f"Conclusion: SIGNIFICANT (0 is outside the 95% CI)")
    else:
        print(f"Conclusion: NOT SIGNIFICANT (0 is inside the 95% CI)")
    return median_diff, ci_low, ci_high, p_val_empirical

def print_statistics(PR_lists, PR_labels, file_path):
    with open(file_path, 'w') as f:

        f.write(f"95% CI of the difference between two independent bootstrap distribution:\n")
        for i in range(len(PR_lists)):
            PR_ci = np.percentile(PR_lists[i], [2.5, 97.5])
            f.write(f"{PR_labels[i]}: Median={np.median(PR_lists[i]):.2f}, 95% CI=[{PR_ci[0]:.2f}, {PR_ci[1]:.2f}]\n")
            for j in range(i+1, len(PR_lists)):
                print(f"--- {PR_labels[i]} vs {PR_labels[j]} ---")
                median_diff, ci_low, ci_high, p_val_empirical = test_pseudo_population_difference(PR_lists[i], PR_lists[j], PR_labels[i], PR_labels[j])

                f.write(f"{PR_labels[i]} vs {PR_labels[j]}: Median Difference: {median_diff:.3f}, 95% CI=[{ci_low:.2f}, {ci_high:.2f}], p-value: {p_val_empirical:.2e}\n")

    print(f"Participation ratio stats saved to {file_path}")#%% Statistical comparison of Participation Ratio

