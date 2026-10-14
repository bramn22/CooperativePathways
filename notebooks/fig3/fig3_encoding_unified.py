#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from scipy.integrate import simpson
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.svm import LinearSVC
from numpy.linalg import norm
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut

from notebooks.CoordinatedCellTypes.final.fig_params import *

#%%
vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)

#%%
# --- CORE UTILITIES ---
def generate_pseudo_trials(data, n_pseudo_trials, rng):
    """Resamples physical trials independently per neuron."""
    n_neurons, n_stimuli, n_phys_trials, n_timebins = data.shape
    pseudo_data = np.zeros((n_neurons, n_stimuli, n_pseudo_trials, n_timebins))
    for n in range(n_neurons):
        for s in range(n_stimuli):
            idx = rng.integers(0, n_phys_trials, size=n_pseudo_trials)
            pseudo_data[n, s, :, :] = data[n, s, idx, :]
    return pseudo_data

def unified_decoding_engine(data, target_groups, k_dim=10, n_pseudo=100, n_shuffles=10, seed=42, time_window=None, use_pca=True, noise_std=0.0):
    rng = np.random.default_rng(seed)
    n_neurons, n_stimuli, n_phys_trials, n_timebins = data.shape
    
    group_map = {s: g_idx for g_idx, group in enumerate(target_groups) for s in group}
    valid_stims = list(group_map.keys())
    n_valid_stims = len(valid_stims)

    bins_to_run = range(n_timebins) if time_window is None else range(time_window[0], time_window[1])
    loo = LeaveOneOut()
    n_splits = loo.get_n_splits(np.arange(n_phys_trials))
    
    acc_real = np.zeros((n_splits, n_timebins))
    var_real = np.zeros((n_splits, n_timebins))
    acc_null = np.zeros((n_splits, n_timebins))
    var_null = np.zeros((n_splits, n_timebins))

    align_real = np.zeros((n_splits, n_timebins, k_dim))
    align_null = np.zeros((n_splits, n_timebins, k_dim))

    # Aggregators for out-of-fold projections to calculate d'
    all_proj_real = np.zeros((n_splits, n_valid_stims, n_timebins))
    all_proj_null = np.zeros((n_splits, n_valid_stims, n_timebins, max(1, n_shuffles)))
    all_y_test = np.zeros((n_splits, n_valid_stims))

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(np.arange(n_phys_trials))):
        
        pseudo_train = generate_pseudo_trials(data[:, :, train_idx, :], n_pseudo, rng)
        X_train = pseudo_train.transpose(2, 1, 0, 3).reshape(-1, n_neurons, n_timebins)
        y_train_stim = np.tile(np.arange(n_stimuli), n_pseudo)
        

        test_data = data[:, :, test_idx, :]
        n_test_phys = test_data.shape[2]
        X_test = test_data.transpose(2, 1, 0, 3).reshape(-1, n_neurons, n_timebins)
        y_test_stim = np.tile(np.arange(n_stimuli), n_test_phys)
        
        # --- DELIBERATE NOISE INJECTION ---
        if noise_std > 0:
            X_test += rng.normal(loc=0, scale=noise_std, size=X_test.shape)
        
        if use_pca:
            PSTH_train = np.stack([np.mean(X_train[y_train_stim == c], axis=0) for c in np.unique(y_train_stim)], axis=1)
            actual_k = min(k_dim, n_neurons)
            pca = PCA(n_components=actual_k)
            pca.fit(PSTH_train.reshape(n_neurons, -1).T)
            sig_var_ratios = pca.explained_variance_ratio_
            
            X_train_centered = X_train - pca.mean_[np.newaxis, :, np.newaxis]
            X_test_centered = X_test - pca.mean_[np.newaxis, :, np.newaxis]
            
            Z_train = np.einsum('cn, int -> ict', pca.components_, X_train_centered)
            Z_test = np.einsum('cn, int -> ict', pca.components_, X_test_centered)
        else:
            Z_train = X_train
            Z_test = X_test
            sig_var_ratios = np.zeros(n_neurons) 
        
        mask_train = np.isin(y_train_stim, valid_stims)
        mask_test = np.isin(y_test_stim, valid_stims)
        
        y_task_train = np.array([group_map[s] for s in y_train_stim[mask_train]])
        y_task_test = np.array([group_map[s] for s in y_test_stim[mask_test]])
        
        all_y_test[fold_idx, :] = y_task_test
        
        for t in bins_to_run:
            Z_tr_t = Z_train[mask_train, :, t]
            Z_te_t = Z_test[mask_test, :, t]
            
            clf = LinearSVC(dual=False, class_weight='balanced', C=0.1, max_iter=2000)
            clf.fit(Z_tr_t, y_task_train)
            
            y_pred = clf.predict(Z_te_t)
            acc_real[fold_idx, t] = balanced_accuracy_score(y_task_test, y_pred)
            
            w = clf.coef_[0]
            if norm(w) > 0:
                var_real[fold_idx, t] = np.sum(((w / norm(w))**2) * sig_var_ratios) * 100
                u = w / norm(w)
                all_proj_real[fold_idx, :, t] = np.dot(Z_te_t, u)

                if use_pca:
                    align_real[fold_idx, t, :len(u)] = np.abs(u)
                
            if n_shuffles > 0:
                fold_acc_nulls = []
                fold_var_nulls = []
                fold_align_nulls = []
                for shuf_idx in range(n_shuffles):
                    y_shuffled = rng.permutation(y_task_train)
                    clf_null = LinearSVC(dual=False, class_weight='balanced', C=0.1, max_iter=2000)
                    clf_null.fit(Z_tr_t, y_shuffled)
                    
                    y_pred_null = clf_null.predict(Z_te_t)
                    fold_acc_nulls.append(balanced_accuracy_score(y_task_test, y_pred_null))
                    
                    w_null = clf_null.coef_[0]
                    if norm(w_null) > 0:
                        fold_var_nulls.append(np.sum(((w_null / norm(w_null))**2) * sig_var_ratios) * 100)
                        u_null = w_null / norm(w_null)
                        all_proj_null[fold_idx, :, t, shuf_idx] = np.dot(Z_te_t, u_null)

                        if use_pca:
                            fold_align_nulls.append(np.abs(u_null))

                acc_null[fold_idx, t] = np.mean(fold_acc_nulls)
                var_null[fold_idx, t] = np.mean(fold_var_nulls)
                if use_pca and fold_align_nulls:
                    align_null[fold_idx, t, :len(u_null)] = np.mean(fold_align_nulls, axis=0)

    # Calculate final pooled d' across all folds
    dprime_real = np.zeros(n_timebins)
    dprime_null = np.zeros(n_timebins)
    
    unique_classes = np.unique(all_y_test)
    if len(unique_classes) == 2:
        class_0, class_1 = unique_classes[0], unique_classes[1]
        y_flat = all_y_test.flatten()
        
        for t in bins_to_run:
            p_flat = all_proj_real[:, :, t].flatten()
            p0, p1 = p_flat[y_flat == class_0], p_flat[y_flat == class_1]
            
            var0 = np.var(p0, ddof=1) if len(p0) > 1 else 0
            var1 = np.var(p1, ddof=1) if len(p1) > 1 else 0
            pooled_sd = np.sqrt(0.5 * (var0 + var1))
            
            if pooled_sd > 0:
                dprime_real[t] = np.abs(np.mean(p0) - np.mean(p1)) / pooled_sd
                
            if n_shuffles > 0:
                dp_shufs = []
                for s in range(n_shuffles):
                    p_flat_n = all_proj_null[:, :, t, s].flatten()
                    p0_n, p1_n = p_flat_n[y_flat == class_0], p_flat_n[y_flat == class_1]
                    var0_n = np.var(p0_n, ddof=1) if len(p0_n) > 1 else 0
                    var1_n = np.var(p1_n, ddof=1) if len(p1_n) > 1 else 0
                    psd_n = np.sqrt(0.5 * (var0_n + var1_n))
                    dp_shufs.append(np.abs(np.mean(p0_n) - np.mean(p1_n)) / psd_n if psd_n > 0 else 0)
                dprime_null[t] = np.mean(dp_shufs)

    return acc_real, var_real, acc_null, var_null, dprime_real, dprime_null, align_real, align_null
# --- PANEL GENERATORS ---
def fit_full_decoding_model(data, target_groups, k_dim=10, n_pseudo=100, seed=42, time_window=None, use_pca=True, window_mode='mean'):
    """
    Fits a decoding model on the full dataset without cross-validation splits.
    
    Returns:
        dict: Contains 'models' (a dictionary of fitted LinearSVC models mapped by time bin)
              and 'pca' (the fitted PCA object, or None if use_pca is False).
    """
    rng = np.random.default_rng(seed)
    n_neurons, n_stimuli, n_phys_trials, n_timebins = data.shape
    
    group_map = {s: g_idx for g_idx, group in enumerate(target_groups) for s in group}
    valid_stims = list(group_map.keys())
    
    bins_to_run = range(n_timebins) if time_window is None else range(time_window[0], time_window[1])
    
    # Generate pseudo-trials using ALL physical trials
    pseudo_train = generate_pseudo_trials(data, n_pseudo, rng)
    X_train = pseudo_train.transpose(2, 1, 0, 3).reshape(-1, n_neurons, n_timebins)
    y_train_stim = np.tile(np.arange(n_stimuli), n_pseudo)
    
    # Optional PCA dimensionality reduction
    if use_pca:
        # Calculate PSTHs across all stimuli to fit PCA
        PSTH_train = np.stack([np.mean(X_train[y_train_stim == c], axis=0) for c in np.unique(y_train_stim)], axis=1)
        actual_k = min(k_dim, n_neurons)
        pca = PCA(n_components=actual_k)
        pca.fit(PSTH_train.reshape(n_neurons, -1).T)
        
        # Center and project data
        X_train_centered = X_train - pca.mean_[np.newaxis, :, np.newaxis]
        Z_train = np.einsum('cn, int -> ict', pca.components_, X_train_centered)
    else:
        pca = None
        Z_train = X_train
        
    # Filter for valid stimuli mapping to our target classes
    mask_train = np.isin(y_train_stim, valid_stims)
    y_task_train = np.array([group_map[s] for s in y_train_stim[mask_train]])
    
    fitted_models = {}
    
    # Train a model for each specified time bin
    if window_mode == 'single':
        for t in bins_to_run:
            Z_tr_t = Z_train[mask_train, :, t]
            
            clf = LinearSVC(dual=False, class_weight='balanced', C=0.1, max_iter=8000)
            clf.fit(Z_tr_t, y_task_train)
            
            fitted_models[t] = clf
    elif window_mode == 'mean':
        # Average across the specified time window
        Z_tr_window = np.mean(Z_train[mask_train, :, time_window[0]:time_window[1]], axis=2)
        
        clf = LinearSVC(dual=False, class_weight='balanced', C=0.1, max_iter=8000)
        clf.fit(Z_tr_window, y_task_train)
        
        fitted_models['mean_window'] = clf

    return {
        'models': fitted_models,
        'pca': pca,
        'bins_run': list(bins_to_run)
    }

def _panel_GJK_worker(data, n_cells, target_groups, k_dim):
    sub_data = data[np.random.choice(data.shape[0], n_cells, replace=False)]
    dynamic_seed = np.random.randint(10000)
    
    a_r, v_r, a_n, v_n, dp_r, dp_n, al_r, al_n = unified_decoding_engine(
        sub_data, target_groups, k_dim=k_dim, seed=dynamic_seed
    )
    
    # d' is already pooled to 1D over time, so we don't average across folds here
    return np.mean(a_r, axis=0), np.mean(v_r, axis=0), np.mean(a_n, axis=0), np.mean(v_n, axis=0), dp_r, dp_n, np.mean(al_r, axis=0), np.mean(al_n, axis=0)

def compute_gjk_results(data_dict, target_groups, k_dim=10, n_bootstraps=10):
    results = {}
    for region, data in data_dict.items():
        print(f"Running full time-resolved engine for {region} ({n_bootstraps} bootstraps)...")
        # n_cells = min(30, data.shape[0]) # TODO: 100
        n_cells = 100
        
        parallel_res = Parallel(n_jobs=-1)(
            delayed(_panel_GJK_worker)(data, n_cells, target_groups, k_dim) 
            for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {region}")
        )
        
        boot_acc_r, boot_var_r, boot_acc_n, boot_var_n, boot_dp_r, boot_dp_n, boot_al_r, boot_al_n = zip(*parallel_res)
            
        results[region] = {
            'acc_real': np.vstack(boot_acc_r),
            'var_real': np.vstack(boot_var_r),
            'acc_null': np.vstack(boot_acc_n),
            'var_null': np.vstack(boot_var_n),
            'dprime_real': np.vstack(boot_dp_r),
            'dprime_null': np.vstack(boot_dp_n),
            'align_real': np.stack(boot_al_r), # shape: (n_bootstraps, n_timebins, k_dim)
            'align_null': np.stack(boot_al_n)
        }
    return results


def plot_gjk_results(results, colors, fig_path, onset_bin=200, bin_size_ms=5, target_ms=(40,41)):
    """Handles the visualization of pre-computed decoding results."""
    # Dynamically extract timebins from the results dictionary
    sample_region = list(results.keys())[0]
    timebins = results[sample_region]['acc_real'].shape[1]
    
    time_vector = (np.arange(timebins) - onset_bin) * bin_size_ms
    # target_bin = onset_bin + int(target_ms / bin_size_ms)
    target_bin = slice(onset_bin+int(target_ms[0]/bin_size_ms), onset_bin+int(target_ms[1]/bin_size_ms)) 

    cm = 1/2.54  # Centimeters in inches for matplotlib

    # --- Plot Panel G: Accuracy over time ---
    fig_g, ax_g = plt.subplots(figsize=(5*cm, 4*cm))
    for region, color in zip(results.keys(), colors):
        mean_acc = np.mean(results[region]['acc_real'], axis=0)
        sem_acc = np.std(results[region]['acc_real'], axis=0)
        mean_null = np.mean(results[region]['acc_null'], axis=0)
        
        ax_g.plot(time_vector, mean_acc, color=color, lw=1.5, label=region)
        ax_g.fill_between(time_vector, mean_acc-sem_acc, mean_acc+sem_acc, color=color, alpha=0.3, lw=0)
        ax_g.plot(time_vector, mean_null, color=color, linestyle='--', lw=1)
        
    ax_g.axvline(0, color='k', linestyle='--', lw=0.8)
    ax_g.set(ylabel='Accuracy', xlim=[-200, 500])
    sb.despine(ax=ax_g)
    plt.savefig(os.path.join(fig_path, 'unidec_accuracy_time_resolved.pdf'), dpi=300)
    plt.show()

    # --- Plot Panel J: Variance over time ---
    fig_j, ax_j = plt.subplots(figsize=(5*cm, 4*cm))
    for region, color in zip(results.keys(), colors):
        mean_var = np.mean(results[region]['var_real'], axis=0)
        sem_var = np.std(results[region]['var_real'], axis=0)
        # mean_var_null = np.mean(results[region]['var_null'], axis=0)
        # sem_var_null = np.std(results[region]['var_null'], axis=0)

        ax_j.plot(time_vector, mean_var, color=color, lw=1.5, label=region)
        ax_j.fill_between(time_vector, mean_var-sem_var, mean_var+sem_var, color=color, alpha=0.3, lw=0)
        # ax_j.plot(time_vector, mean_var_null, color='gray', lw=1.5, label=region, linestyle='--')
        # ax_j.fill_between(time_vector, mean_var_null-sem_var_null, mean_var_null+sem_var_null, color='gray', alpha=0.3, lw=0)
       
    ax_j.axvline(0, color='k', linestyle='--', lw=0.8)
    ax_j.axvline(target_ms[0], color='k', linestyle=':', lw=0.8)
    ax_j.axvline(target_ms[1], color='k', linestyle=':', lw=0.8)
    ax_j.set(ylabel='% variance explained', xlim=[-200, 500])
    # ax_j.set(ylim=[0, 22])
    sb.despine(ax=ax_j)
    plt.savefig(os.path.join(fig_path, 'unidec_var_explained_time_resolved.pdf'), dpi=300)
    plt.show()

    # --- Plot Panel K supplementary: barplot at target_ms with nulls---
    fig_k, ax_k = plt.subplots(figsize=(len(colors)*2*cm, 4*cm))
    bar_width = 0.35
    x = np.arange(len(colors))
    
    real_means, real_sems, null_means, null_sems = [], [], [], []
    for region, color in zip(results.keys(), colors):
        real_means.append(np.mean(np.mean(results[region]['var_real'][:, target_bin], axis=-1)))
        real_sems.append(np.std(np.mean(results[region]['var_real'][:, target_bin], axis=-1)))
        null_means.append(np.mean(np.mean(results[region]['var_null'][:, target_bin], axis=-1)))
        null_sems.append(np.std(np.mean(results[region]['var_null'][:, target_bin], axis=-1)))


    ax_k.bar(x - bar_width/2, real_means, bar_width, yerr=real_sems, color=colors, label='Real')
    ax_k.bar(x + bar_width/2, null_means, bar_width, yerr=null_sems, color='gray', label='Null')
    
    ax_k.set_xticks(x)
    ax_k.set_xticklabels(list(results.keys()))
    ax_k.set_ylabel('% Signal Var Explained')
    ax_k.set_title(f'Variance at {target_ms}ms')
    ax_k.legend(frameon=False)
    sb.despine(ax=ax_k)
    plt.savefig(os.path.join(fig_path, 'suppl_unidec_var_explained.pdf'), dpi=300)
    plt.show()

    # --- Plot Panel K supplementary: Violinplot at target_ms ---
    fig_k, ax_k = plt.subplots(figsize=(len(colors)*1.5*cm, 4*cm))

    data_to_plot = []

    # Collect the raw data distributions
    for region in results.keys():
        data_to_plot.append(np.max(results[region]['var_real'][:, target_bin], axis=-1))

    # Create the Seaborn violin plot
    # 'palette' applies your custom colors automatically
    # 'inner="box"' draws a miniature boxplot inside (median and quartiles)
    sb.violinplot(data=data_to_plot, ax=ax_k, palette=colors, inner="quartile")

    # Seaborn assigns numeric x-ticks (0, 1, 2...) when given a list of arrays, 
    # so we apply your region names to those positions
    ax_k.set_xticks(range(len(results.keys())))
    ax_k.set_xticklabels(list(results.keys()))

    ax_k.set_ylabel('% Signal Var Explained')
    ax_k.set_title(f'Variance at {target_ms}ms')
    sb.despine(ax=ax_k)

    # Using tight_layout to ensure nothing gets cropped out
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'unidec_var_explained_sns_violin.pdf'), dpi=300)
    plt.show()

    # Slightly increased the width multiplier (from 2 to 2.5) to give the labels more room
    fig_k, ax_k = plt.subplots(figsize=(len(colors)*1.5*cm, 4.5*cm)) 
    x = np.arange(len(colors))

    for i, (region, color) in enumerate(zip(results.keys(), colors)):
        data = np.mean(results[region]['var_real'][:, target_bin], axis=-1)
        
        mean_val = np.mean(data)
        ci_lower = np.percentile(data, 2.5)
        ci_upper = np.percentile(data, 97.5)
        
        err_lower = mean_val - ci_lower
        err_upper = ci_upper - mean_val
        
        ax_k.errorbar(x[i], mean_val, 
                    yerr=[[err_lower], [err_upper]], 
                    fmt='o', 
                    color=color, 
                    capsize=4,        # Slightly smaller caps for publication aesthetics
                    markersize=6,     # Scaled down marker size slightly
                    linewidth=1.5)    # Refined line width

    ax_k.set_xticks(x)

    # FIX 1: Rotate the labels 45 degrees and align them to the right
    ax_k.set_xticklabels(list(results.keys()), rotation=45, ha='right')

    # FIX 2: Manually set x-limits to add padding on the left and right of the points
    ax_k.set_xlim(-0.5, len(colors) - 0.5)

    ax_k.set_ylabel('% Signal Var Explained')
    ax_k.set_title(f'Variance at {target_ms}ms')
    # ax_k.set_ylim([0, 22])
    sb.despine(ax=ax_k)

    # FIX 3: Automatically adjust subplot parameters so the rotated labels aren't cut off
    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, 'unidec_var_explained_point.pdf'), dpi=300)
    plt.show()

    # --- Plot PC Alignment (Grouped Barplot) ---
    fig_align, ax_align = plt.subplots(figsize=(len(colors)*2*cm + 2, 4*cm))
    
    # Extract k_dim from the cached arrays
    k_dim = results[list(results.keys())[0]]['align_real'].shape[-1]
    x = np.arange(k_dim)
    n_regions = len(results.keys())
    
    # Bar width calculations to group bars per PC
    total_bar_width = 0.8
    single_bar_width = total_bar_width / n_regions
    
    for i, (region, color) in enumerate(zip(results.keys(), colors)):
        # Mean across target time window, preserving bootstraps: shape (bootstraps, k_dim)
        align_window = np.mean(results[region]['align_real'][:, target_bin, :], axis=1)
        
        mean_val = np.mean(align_window, axis=0)
        err_val = np.std(align_window, axis=0)
        
        offset = (i - n_regions/2 + 0.5) * single_bar_width
        
        ax_align.bar(x + offset, mean_val, width=single_bar_width, yerr=err_val, 
                     color=color, label=region, capsize=2, edgecolor='none')

    ax_align.set_xticks(x)
    ax_align.set_xticklabels([f'PC{i+1}' for i in range(k_dim)])
    ax_align.set_ylabel('|Cosine Similarity|')
    ax_align.set_title(f'Decoding Axis PC Alignment ({target_ms}ms)')
    ax_align.legend(frameon=False)
    sb.despine(ax=ax_align)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'unidec_pc_alignment_barplot.pdf'), dpi=300)
    plt.show()

    # --- Plot Discriminability (d') over time ---
    cm = 1/2.54
    fig_dp, ax_dp = plt.subplots(figsize=(5*cm, 4*cm))
    for region, color in zip(results.keys(), colors):
        mean_dp = np.mean(results[region]['dprime_real'], axis=0)
        sem_dp = np.std(results[region]['dprime_real'], axis=0)
        mean_dp_null = np.mean(results[region]['dprime_null'], axis=0)
        
        ax_dp.plot(time_vector, mean_dp, color=color, lw=1.5, label=region)
        ax_dp.fill_between(time_vector, mean_dp-sem_dp, mean_dp+sem_dp, color=color, alpha=0.3, lw=0)
        ax_dp.plot(time_vector, mean_dp_null, color=color, linestyle='--', lw=1)
        
    ax_dp.axvline(0, color='k', linestyle='--', lw=0.8)
    ax_dp.set(ylabel="Discriminability (d')", xlim=[-200, 500])
    sb.despine(ax=ax_dp)
    plt.savefig(os.path.join(fig_path, 'unidec_dprime_time_resolved.pdf'), dpi=300)
    plt.show()

def generate_panels_G_J_K(data_dict, colors, target_groups, fig_path, 
                          onset_bin=200, bin_size_ms=5, target_ms=40.0, 
                          k_dim=10, n_bootstraps=10, use_saved=True, cache_filename='gjk_results_align_cache.pkl'):
    """
    Main wrapper that manages computation, caching, and plotting.
    Set use_saved=False to force a re-computation.
    """
    cache_path = os.path.join(fig_path, cache_filename)
    
    if use_saved and os.path.exists(cache_path):
        print(f"Loading cached analysis results from {cache_path}...")
        with open(cache_path, 'rb') as f:
            results = pickle.load(f)
    else:
        print("Computing new analysis results...")
        results = compute_gjk_results(data_dict, target_groups, k_dim, n_bootstraps)
        
        # Save the results
        # os.makedirs(fig_path, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results cached to {cache_path}.")

    # Proceed to plot with the results (either loaded or freshly computed)
    plot_gjk_results(results, colors, fig_path, onset_bin, bin_size_ms, target_ms)
    return results

def compute_panel_K_stats(data_dict, results_dict, target_groups, target_bins, file_path, k_dim=10, n_perms=1000, n_cells=100):
    print("\n--- Panel K Statistical Testing ---")
    with open(file_path, 'w') as f:
        f.write(f"Variance explained | Statistical Testing:\n")
        
        # 1. Within-Region: Real vs. Null (Exact Empirical P-value)
        for region in results_dict.keys():
            # real_var = np.max(results_dict[region]['var_real'][:, target_bins], axis=-1)
            # null_var = np.max(results_dict[region]['var_null'][:, target_bins], axis=-1)
            real_var = np.mean(results_dict[region]['var_real'][:, target_bins], axis=-1)
            null_var = np.mean(results_dict[region]['var_null'][:, target_bins], axis=-1)
            
            # Calculate how often the null model variance equaled or exceeded the real variance
            p_val_within = (1 + np.sum(null_var >= np.mean(real_var))) / (1 + len(null_var))
            print(f"{region} Real vs Null p-value: {p_val_within:.4f}")
            f.write(f"{region} Real vs Null p-value: {p_val_within:.4f}\n")
        
        # ---------------------------------------------------------
        # NEW: 2. Between-Region: Bootstrap CI of the Difference
        # ---------------------------------------------------------
        print(f"\nRunning Between-Region Bootstrap CI of Difference...")
        f.write(f"\nRunning Between-Region Bootstrap CI of Difference...\n")
        alpha_levels = [0.05, 0.01, 0.001]
        keys = list(data_dict.keys())

        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                key_A = keys[i]
                key_B = keys[j]
                
                # Extract the 1D arrays of bootstrapped maximum variance in the target window
                # var_A_boots = np.max(results_dict[key_A]['var_real'][:, target_bins], axis=-1)
                # var_B_boots = np.max(results_dict[key_B]['var_real'][:, target_bins], axis=-1)
                var_A_boots = np.mean(results_dict[key_A]['var_real'][:, target_bins], axis=-1)
                var_B_boots = np.mean(results_dict[key_B]['var_real'][:, target_bins], axis=-1)
                
                # Since these are independent bootstraps, element-wise subtraction yields the distribution of the difference
                diff_boots = var_A_boots - var_B_boots
                mean_diff_boot = np.mean(diff_boots)
                
                # Print/Write Header for the comparison
                header_str = f"\n--- Bootstrap Diff ({key_A} - {key_B}): {mean_diff_boot:.2f}% ---"
                print(header_str)
                f.write(header_str + "\n")
                
                # 1. Loop through different alpha levels and calculate their specific CIs
                for alpha in alpha_levels:
                    # Calculate the lower and upper percentiles for a two-tailed test
                    lower_pct = 100 * (alpha / 2)
                    upper_pct = 100 * (1 - (alpha / 2))
                    
                    ci_lower = np.percentile(diff_boots, lower_pct)
                    ci_upper = np.percentile(diff_boots, upper_pct)
                    
                    # If zero is NOT in the interval, it is significant at this alpha level
                    is_significant = not (ci_lower <= 0 <= ci_upper)
                    
                    ci_level_str = f"{(1-alpha)*100:.1f}% CI"
                    result_str = f"{ci_level_str}: [{ci_lower:.2f}%, {ci_upper:.2f}%] | Significant (p < {alpha})? {is_significant}"
                    
                    print(result_str)
                    f.write(result_str + "\n")
                    
                # 2. Calculate the exact two-tailed empirical p-value
                p_val_empirical = 2 * min(np.mean(diff_boots <= 0), np.mean(diff_boots >= 0))
                
                # 3. Calculate and print the p-value cap
                n_bootstraps = len(diff_boots)
                p_val_cap = 1 / n_bootstraps
                p_val_reported = max(p_val_empirical, p_val_cap)
                
                cap_str = f"Minimum measurable p-value (cap): p < {p_val_cap:.5f} (based on N={n_bootstraps})"
                print(cap_str)
                f.write(cap_str + "\n")
                
                pval_str = f"Exact Empirical p-value: {p_val_reported:.5f}"
                if p_val_reported == p_val_cap and p_val_empirical == 0:
                    pval_str += " (Result hit the cap: True p-value is likely lower)"
                    
                print(pval_str)
                f.write(pval_str + "\n")
    print(f"Variance explained | stats saved to {file_path}")
    
    return p_val_within, p_val_empirical

def _panel_H_worker(data, n_cells, target_groups, target_window):
    """
    Executes a single bootstrap iteration for cell subsampling decoding.
    Decodes in native cell space (use_pca=False) and skips null distributions.
    """
    n_available_cells = data.shape[0]
    actual_cells = min(n_cells, n_available_cells)
    
    # Subsample neurons randomly without replacement
    sub_data = data[np.random.choice(n_available_cells, actual_cells, replace=False)]
    dynamic_seed = np.random.randint(10000) 
    
    # Run core decoding engine
    a_r, _, _, _, _, _, _, _ = unified_decoding_engine(
        sub_data, 
        target_groups, 
        n_shuffles=0,  
        seed=dynamic_seed,
        time_window=target_window,
        use_pca=False
    )
    
    # Return the mean accuracy across folds and specified time bins
    return np.mean(a_r[:, target_window[0]:target_window[1]])

def generate_panel_H_parallel(data_dict, colors, target_groups, fig_path, cell_counts=[1,10, 20,35, 50, 75, 100], target_window=[205, 215], n_bootstraps=100):
    """Generates Accuracy vs. Number of Cells using multi-core processing."""
    from joblib import Parallel, delayed
    
    mean_accs = {key: [] for key in data_dict.keys()}
    err_accs = {key: [] for key in data_dict.keys()}
    
    for region in data_dict.keys():
        print(f"Running parallel cell subsampling for {region}...")
        region_data = data_dict[region]
        
        for n_cells in cell_counts:
            print(n_cells)
            # Parallelize all bootstraps for the current cell count tier
            boot_accs = Parallel(n_jobs=5)(
                delayed(_panel_H_worker)(region_data, n_cells, target_groups, target_window)
                for _ in range(n_bootstraps)
            )
            
            # Aggregate metrics
            mean_accs[region].append(np.mean(boot_accs))
            err_accs[region].append(np.std(boot_accs))# / np.sqrt(len(boot_accs)))

    # --- PLOTTING ---
    fig_h, ax_h = plt.subplots(figsize=(4*cm, 4*cm))
    for region, color in zip(data_dict.keys(), colors):
        ax_h.plot(cell_counts, mean_accs[region], color=color, lw=2, label=region)
        ax_h.fill_between(
            cell_counts, 
            np.array(mean_accs[region]) - np.array(err_accs[region]), 
            np.array(mean_accs[region]) + np.array(err_accs[region]), 
            color=color, alpha=0.3, lw=0
        )

    ax_h.axhline(0.5, color='k', linestyle='--', lw=0.8) 
    ax_h.set(xlabel='Number of cells', ylabel='Balanced Accuracy', ylim=[0.4, 1.05])
    ax_h.legend(frameon=False)
    sb.despine(ax=ax_h)
    plt.savefig(os.path.join(fig_path, 'acc_n_cells.pdf'), dpi=300)
    plt.show()
    return mean_accs, err_accs

# --- 2. FULLY COUPLED PANEL H STATS ---
def permutation_test_AUC_panel_H(data_dict, target_groups, cell_counts, target_window, plotted_mean_accs, file_path, n_perms=1000):
    print("\n--- Accuracy vs Number of cells | AUC Statistical Testing ---")

    def _get_single_auc(data, n_cells_list):
        """Helper to get AUC for a single permutation."""
        curve = []
        for n in n_cells_list:
            actual_cells = min(n, data.shape[0])
            sub_data = data[np.random.choice(data.shape[0], actual_cells, replace=False)]
            dynamic_seed = np.random.randint(10000)
            
            a_r, _, _, _, _, _, _, _ = unified_decoding_engine(sub_data, target_groups, n_shuffles=0, seed=dynamic_seed, time_window=target_window, use_pca=False)
            curve.append(np.mean(a_r[:, target_window[0]:target_window[1]]))
        return simpson(y=curve, x=n_cells_list)

    with open(file_path, 'w') as f:
        f.write(f"Accuracy vs Number of cells | AUC Statistical Testing:\n")
        for i in range(len(data_dict)):
            for j in range(i+1, len(data_dict)):
                key_A = list(data_dict.keys())[i]
                key_B = list(data_dict.keys())[j]
                data_A = data_dict[key_A]
                data_B = data_dict[key_B]

                # 1. OBSERVED DIFFERENCE (Strictly coupled to the figure)
                obs_auc_A = simpson(y=plotted_mean_accs[key_A], x=cell_counts)
                obs_auc_B = simpson(y=plotted_mean_accs[key_B], x=cell_counts)
                obs_diff = obs_auc_A - obs_auc_B
                
                print(f"Plotted AUC {key_A}: {obs_auc_A:.2f}")
                print(f"Plotted AUC {key_B}: {obs_auc_B:.2f}")
                print(f"Exact Plotted Difference: {obs_diff:.2f}")
                f.write(f"Plotted AUC {key_A}: {obs_auc_A:.2f}\n")
                f.write(f"Plotted AUC {key_B}: {obs_auc_B:.2f}\n")
                f.write(f"Exact Plotted Difference: {obs_diff:.2f}\n")

                # 2. NULL DISTRIBUTION (Permuting the raw data)
                pooled_data = np.concatenate([data_A, data_B], axis=0)
                total_cells = pooled_data.shape[0]
                print("pooled_data.shape", pooled_data.shape)
                
                def _perm_auc_worker():
                    # Shuffle labels, maintaining original group sizes
                    perm_indices = np.random.permutation(total_cells)
                    pseudo_A = pooled_data[perm_indices[:data_A.shape[0]]]
                    pseudo_B = pooled_data[perm_indices[data_A.shape[0]:]]
                    
                    auc_pseudo_A = _get_single_auc(pseudo_A, cell_counts)
                    auc_pseudo_B = _get_single_auc(pseudo_B, cell_counts)
                    
                    return auc_pseudo_A - auc_pseudo_B

                # Explicitly restricted to 10 cores
                null_diffs = Parallel(n_jobs=10)(delayed(_perm_auc_worker)() for _ in tqdm(range(n_perms), desc="Building Null AUC"))
                
                # Two-tailed empirical p-value
                p_val_auc = (1 + np.sum(np.abs(null_diffs) >= np.abs(obs_diff))) / (1 + n_perms)
                print(f"Panel H (AUC) Between-Region p-value: {p_val_auc:.4f}")
                f.write(f"AUC Between-Region p-value: {p_val_auc:.4f}\n")
    print(f"Accuracy vs Number of cells | stats saved to {file_path}")#%% Statistical comparison of Participation Ratio

    return p_val_auc

def test_decoding_above_chance(data_dict, target_groups, target_window, file_path, max_cells=100, n_perms=1000):
    """
    Tests if each region can decode visual stimuli significantly above empirical chance.
    Generates a null distribution by shuffling stimulus labels while preserving neural noise.
    """
    from joblib import Parallel, delayed
    
    print("\n" + "="*50)
    print(f"WITHIN-REGION DECODING VS. CHANCE (at {max_cells} cells)")
    print("="*50)
    
    with open(file_path, 'a') as f:
        f.write(f"\nWithin-Region Decoding vs. Chance (at {max_cells} cells):\n")
        
        for region, data in data_dict.items():
            actual_cells = min(max_cells, data.shape[0])
            
            # Draw the cells once to maintain consistency between Real and Null
            sub_data = data[np.random.choice(data.shape[0], actual_cells, replace=False)]
            
            # 1. Get REAL accuracy
            dynamic_seed = np.random.randint(10000)
            a_r, _, _, _, _, _, _, _ = unified_decoding_engine(sub_data, target_groups, n_shuffles=0, seed=dynamic_seed, time_window=target_window, use_pca=False)
            real_acc = np.mean(a_r[:, target_window[0]:target_window[1]])
            
            # 2. Get NULL distribution
            def _null_worker():
                # We scramble the 'stimulus' dimension (axis 1) independently for each trial (axis 2).
                # This perfectly preserves the cell-to-cell noise correlations (axis 0) while destroying the signal.
                scrambled_data = np.copy(sub_data)
                for tr in range(scrambled_data.shape[2]):
                    scrambled_data[:, :, tr, :] = scrambled_data[:, np.random.permutation(scrambled_data.shape[1]), tr, :]
                    
                null_seed = np.random.randint(10000)
                a_null, _, _, _, _, _, _, _ = unified_decoding_engine(scrambled_data, target_groups, n_shuffles=0, seed=null_seed, time_window=target_window, use_pca=False)
                return np.mean(a_null[:, target_window[0]:target_window[1]])
            
            # Run the permutations in parallel
            null_accs = Parallel(n_jobs=-1)(delayed(_null_worker)() for _ in tqdm(range(n_perms), desc=f"Null Dist: {region}"))
            
            # 3. Calculate Empirical P-value
            p_val = (1 + np.sum(np.array(null_accs) >= real_acc)) / (1 + n_perms)
            
            print(f"{region} | Real Acc: {real_acc:.3f} | p-value vs chance: {p_val:.4f}")
            f.write(f"{region} | Real Acc: {real_acc:.3f} | p-value vs chance: {p_val:.4f}\n")

# %% Decoding using PCA-reduced trials
def run_decoding_analysis_time(mats_list, labels, colors, decoding_classes, fig_path, n_bootstraps=100, n_perms=500, k_dim=7, use_saved=True):
    data_dict = {}
    for mats, label in zip(mats_list, labels):
        print([mat.shape for mat in mats])
        data_pcadecod = fig3_preprocessing.unflatten_frs_mat(np.concatenate(mats, axis=0), n_stim=6, n_trials=8, n_timepoints=800)
        data_dict[label] = data_pcadecod 
        print("data_pcadecod.shape (n_neurons, n_stimuli, 8, 800):", data_pcadecod.shape) # Check that (n_neurons, n_stimuli, 8, 800)


    # Define timing constants to match your plotting logic
    onset_bin = 200
    bin_size_ms = 5
    target_ms = (25, 75)
    target_bins = slice(onset_bin+int(target_ms[0]/bin_size_ms), onset_bin+int(target_ms[1]/bin_size_ms)) 
    
    results_dict = generate_panels_G_J_K(data_dict, colors, decoding_classes, fig_path=fig_path, onset_bin=onset_bin, bin_size_ms=bin_size_ms, target_ms=target_ms, n_bootstraps=n_bootstraps, k_dim=k_dim, use_saved=use_saved, cache_filename='gjk_results_cache.pkl')
    # if not use_saved:
    compute_panel_K_stats(data_dict, results_dict, decoding_classes, target_bins, file_path=os.path.join(fig_path, 'decoding_cellcount_stats.txt'), n_perms=n_perms)


def run_decoding_analysis_cellcounts(mats_list, labels, colors, decoding_classes, fig_path, target_window, n_bootstraps=100, n_perms=500):
    data_dict = {}
    for mats, label in zip(mats_list, labels):
        print([mat.shape for mat in mats])
        data_pcadecod = fig3_preprocessing.unflatten_frs_mat(np.concatenate(mats, axis=0), n_stim=6, n_trials=8, n_timepoints=800)
        data_dict[label] = data_pcadecod 
        print("data_pcadecod.shape (n_neurons, n_stimuli, 8, 800):", data_pcadecod.shape) # Check that (n_neurons, n_stimuli, 8, 800)

    # Define timing constants to match your plotting logic
    # cell_counts = [10, 25, 50, 75, 100]
    cell_counts = [1,10,25,50]

    plotted_curve_data = generate_panel_H_parallel(data_dict, colors, decoding_classes, cell_counts=cell_counts, target_window=target_window, n_bootstraps=n_bootstraps, fig_path=fig_path)
    permutation_test_AUC_panel_H(data_dict, decoding_classes, cell_counts, target_window, file_path=os.path.join(fig_path, 'varexpl_stats.txt'), plotted_mean_accs=plotted_curve_data[0], n_perms=n_perms)
    test_decoding_above_chance(data_dict, decoding_classes, target_window, file_path=os.path.join(fig_path, 'varexpl_stats.txt'), max_cells=100, n_perms=n_perms)


def _to_slice(w):
    """Helper to convert list/tuple/slice to a slice object."""
    if isinstance(w, slice):
        return w
    return slice(w[0], w[1])

def generate_model_decision_variables(data, target_groups, time_window, baseline_window, 
                                      subsample_n=20, n_models=100, n_test_per_model=100, 
                                      k_dim=10, use_pca=True, seed=42,
                                      modulation_strengths=[0.0, 0.5, 1.0, 1.5]):
    """
    Extracts 1D decision variables for the Generative Readout Model using Approach B.
    Simulates sparse downstream connectivity and applies population-wide scaling 
    during the testing phase to model optogenetic/pharmacological manipulation.
    """
    rng = np.random.default_rng(seed)
    n_neurons, n_stimuli, n_phys_trials, n_timebins = data.shape
    
    if subsample_n is None or subsample_n >= n_neurons:
        subsample_n = n_neurons
        n_models = 1
        n_test_per_model = n_models * n_test_per_model
        
    group_map = {s: g_idx for g_idx, group in enumerate(target_groups) for s in group}
    valid_stims = list(group_map.keys())
    
    t_slice = _to_slice(time_window)
    b_slice = _to_slice(baseline_window)
    
    target_means_full = np.mean(data[:, :, :, t_slice], axis=3)
    baseline_means_full = np.mean(data[:, :, :, b_slice], axis=3)
    
    n_mods = len(modulation_strengths)
    
    # Initialize dictionary to store a list of outputs for each modulation strength
    dv_dict = {f'stim_{s}': [[] for _ in range(n_mods)] for s in range(n_stimuli)}
    dv_dict.update({f'base_{s}': [[] for _ in range(n_mods)] for s in range(n_stimuli)})
    
    for model_idx in range(n_models):
        # 1. THE ANATOMICAL BOTTLENECK
        selected_cells = rng.choice(n_neurons, subsample_n, replace=False)
        
        # 2. GENERATE PSEUDO-TRAIN SET FOR THIS SPECIFIC PATHWAY
        n_train_pseudo = 100 
        X_train = np.zeros((subsample_n, n_stimuli, n_train_pseudo, n_timebins))
        for i, cell in enumerate(selected_cells):
            for s in range(n_stimuli):
                idx = rng.integers(0, n_phys_trials, size=n_train_pseudo)
                X_train[i, s] = data[cell, s, idx]
                
        X_train_flat = X_train.transpose(2, 1, 0, 3).reshape(-1, subsample_n, n_timebins)
        y_train_stim = np.tile(np.arange(n_stimuli), n_train_pseudo)
        
        # 3. FIT LATENT SPACE & HYPERPLANE
        actual_k = min(k_dim, subsample_n)
        
        if use_pca:
            PSTH_train = np.stack([np.mean(X_train_flat[y_train_stim == c], axis=0) 
                                   for c in np.unique(y_train_stim)], axis=1)
            pca = PCA(n_components=actual_k)
            pca.fit(PSTH_train.reshape(subsample_n, -1).T)
            
            X_tr_centered = X_train_flat - pca.mean_[np.newaxis, :, np.newaxis]
            Z_train = np.einsum('cn, int -> ict', pca.components_, X_tr_centered)
            Z_tr_window = np.mean(Z_train[:, :, t_slice], axis=2)
            pca_mean = pca.mean_
            pca_comp = pca.components_
        else:
            Z_tr_window = np.mean(X_train_flat[:, :, t_slice], axis=2)
            
        mask_train = np.isin(y_train_stim, valid_stims)
        y_task_train = np.array([group_map[s] for s in y_train_stim[mask_train]])
        Z_tr_task = Z_tr_window[mask_train]
        
        clf = LinearSVC(dual=False, class_weight='balanced', C=0.1, max_iter=5000)
        clf.fit(Z_tr_task, y_task_train)
        
        w = clf.coef_[0]
        b = clf.intercept_[0]
        if norm(w) > 0:
            b = b / norm(w)
            w = w / norm(w)
            
        # 4. BOMBARD WITH NOISE & APPLY MODULATION
        for stim_idx in range(n_stimuli):
            chosen_trials = rng.integers(0, n_phys_trials, size=(subsample_n, n_test_per_model))
            
            sampled_frs = target_means_full[selected_cells[:, None], stim_idx, chosen_trials]
            sampled_base_frs = baseline_means_full[selected_cells[:, None], stim_idx, chosen_trials]
            
            for mod_idx, mod_strength in enumerate(modulation_strengths):
                # Scale the raw firing rates
                mod_target_frs = sampled_frs * mod_strength
                mod_base_frs = sampled_base_frs * mod_strength
                
                if use_pca:
                    delta_target = mod_target_frs - pca_mean[:, None]
                    delta_base = mod_base_frs - pca_mean[:, None]
                    z_target = np.dot(pca_comp, delta_target)
                    z_base = np.dot(pca_comp, delta_base)
                else:
                    z_target = mod_target_frs
                    z_base = mod_base_frs
                    
                dv_dict[f'stim_{stim_idx}'][mod_idx].extend(np.dot(w, z_target) + b)
                dv_dict[f'base_{stim_idx}'][mod_idx].extend(np.dot(w, z_base) + b)

    # Convert output lists to matrices of shape (n_conditions, n_decision_values)
    for key in dv_dict:
        dv_dict[key] = np.array(dv_dict[key])
        
    return dv_dict