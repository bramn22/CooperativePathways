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
import copy
from scipy import ndimage
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from decoding import get_baseline_and_test, run_discrimination_singlecell, run_discrimination_wrapper
import pickle
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal

#%% Figure parameters and path
from fig_params import *
fig_path = r'E:\KU Leuven\FarrowLab-Colab - VEIDB\Manuscript\Figures\Ephys\raw\FigBehav'

#%% Load data (only visually responsive cells)
stim_types = ['EXPA', 'EXPA_60', 'EXPA_25', 'EXPW', 'EXPW_60','DIMM']
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
LP_hfx = load_data('LP_VR_freehfx_all_fr25_20260821')
sSC_free = load_data('sSC_VR_freefree_all_fr25_20260715')
dSC_free = load_data('dSC_VR_freefree_all_fr25_20260715')
LP_free = load_data('LP_VR_freefree_all_fr25_20260821')
SC_behav = load_data('SC_free_20260715')
LP_behav = load_data('LP_free_20260821')

sSC_hfx_frs, sSC_depths, sSC_keys_used = sSC_hfx['cluster_frs'], sSC_hfx['cluster_depths'], sSC_hfx['cluster_keys']
dSC_hfx_frs, dSC_depths, dSC_keys_used = dSC_hfx['cluster_frs'], dSC_hfx['cluster_depths'], dSC_hfx['cluster_keys']
LP_hfx_frs, LP_depths, LP_keys_used = LP_hfx['cluster_frs'], LP_hfx['cluster_depths'], LP_hfx['cluster_keys']
sSC_free_frs, sSC_free_depths, sSC_free_keys_used, sSC_free_stims = sSC_free['cluster_frs'], sSC_free['cluster_depths'], sSC_free['cluster_keys'], sSC_free['stims']
dSC_free_frs, dSC_free_depths, dSC_free_keys_used, dSC_free_stims = dSC_free['cluster_frs'], dSC_free['cluster_depths'], dSC_free['cluster_keys'], dSC_free['stims']
LP_free_frs, LP_free_depths, LP_free_keys_used, LP_free_stims = LP_free['cluster_frs'], LP_free['cluster_depths'], LP_free['cluster_keys'], LP_free['stims']
SC_speed, SC_cornerdist, SC_stims = SC_behav['speed'], SC_behav['corner_distance'], SC_behav['stims']
LP_speed, LP_cornerdist, LP_stims = LP_behav['speed'], LP_behav['corner_distance'], LP_behav['stims']

speed, cornerdist, stims = SC_speed, SC_cornerdist, SC_stims
free_frs, free_depths, free_keys_used = dSC_free_frs, dSC_free_depths, dSC_free_keys_used
hfx_frs = dSC_hfx_frs
# speed, cornerdist, stims = LP_speed, LP_cornerdist, LP_stims
# free_frs, free_depths, free_keys_used = LP_free_frs, LP_free_depths, LP_free_keys_used
# hfx_frs = LP_hfx_frs

color_area = color_dSC

#%%

def get_hfx_stim_kernel(stim_onset=400, dur=2000, kernel_type='default'):
    kernel = np.zeros(dur)
    if kernel_type == 'default':
        kernel[stim_onset:int(stim_onset+0.5*200)] = 1
        kernel[int(stim_onset+0.85*200):int(stim_onset+(0.85+0.25)*200)] = 1
        kernel[int(stim_onset+2*0.85*200):int(stim_onset+(2*0.85+0.25)*200)] = 1
    elif kernel_type == 'dynamic':
        kernel[stim_onset:int(stim_onset+0.25*200)] = 1
        kernel[int(stim_onset+0.85*200):int(stim_onset+(0.85+0.25)*200)] = 1
        kernel[int(stim_onset+2*0.85*200):int(stim_onset+(2*0.85+0.25)*200)] = 1
    elif kernel_type == 'onset':
        kernel[stim_onset:int(stim_onset+0.25*200)] = 1
    return kernel

def get_free_stim_kernel(stim_onset=400, dur=2000):
    kernel = np.zeros(dur)
    kernel[stim_onset:int(stim_onset+0.5*200)] = 1
    kernel[stim_onset+1*200:int(stim_onset+1.5*200)] = 1
    kernel[stim_onset+2*200:int(stim_onset+2.5*200)] = 1
    return kernel

def get_onsets_and_offsets(trace, min_duration):
    # returns onsets and offsets of 1 values in trace
    diff = np.diff(trace, prepend=0, append=0)
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0] - 1
    # print(onsets, offsets)
    # filter bouts by min_duration
    assert len(onsets) == len(offsets)
    bouts = []
    for onset, offset in zip(onsets, offsets):
        if offset-onset >= min_duration:
            # print(onset, offset)
            bouts.append((onset, offset))
    return bouts

def find_true_escape_onset(speed_trace, trigger_frame, min_frame=0, max_lookback_frames=60, low_speed_thresh=5.0):
    """
    Walks backward from a high-speed trigger frame (e.g. 25 cm/s) to find 
    the exact frame where acceleration into the escape response initiated.
    """
    # 1. Smooth trace slightly to ignore single-frame noise jitter
    smoothed_speed = ndimage.gaussian_filter1d(speed_trace, sigma=5.0) # TODO: reset sigma=2
    acc = np.gradient(smoothed_speed)  # Acceleration trace
    
    search_start = trigger_frame
    search_end = max(min_frame, trigger_frame - max_lookback_frames)
    
    onset_frame = search_start
    
    for f in range(search_start, search_end, -1):
        # Stop condition A: Reached baseline / low speed
        if smoothed_speed[f] <= low_speed_thresh:
            onset_frame = f
            break
            
        # Stop condition B: Reached local minimum in speed (acceleration <= 0)
        # Going backward, when acc <= 0, speed was no longer increasing forward in time
        if acc[f] <= 0:
            onset_frame = f
            break
            
        onset_frame = f
        
    return onset_frame

def get_manual_escape_trials(sess_name):
    # Sessions where automatic escape detection failed
    manual_annot_dict = {('02585','20260204'): [(11,290),(12,290),(13,250),(14,200)],
                    ('02585','20260217'): [(2,150),(12,390)],
                    ('02584','20260204'): [(3,250),(7,140)],
                    ('02584','20260210'): [(11,200),(12,190),(14,180),(23,180)]}
    return manual_annot_dict.get(sess_name, None)

# Ethogram
def get_ethogram(speed_mat, cornerdist_mat, sess_name):
    # smooth speed_mat using gaussian filter
    speed_mat = ndimage.gaussian_filter(speed_mat, sigma=(0,5), order=0, mode='reflect')

    # Detect running bout
    running_threshold = 10
    running_n_frames = 5
    running_mat = speed_mat > running_threshold
    # median filter
    print("running_mat shape", running_mat.shape)
    running_mat = ndimage.median_filter(running_mat, size=(1,3), mode='reflect')
    running_bouts = [get_onsets_and_offsets(trial, running_n_frames) for trial in running_mat]
    
    # Detect stopping bout
    stopping_threshold = 5
    stopping_n_frames = 5
    stopping_mat = speed_mat < stopping_threshold
    # for each trial, extract bout beginning and ending where animal stopped for more than stopping_n_frames
    stopping_mat = ndimage.median_filter(stopping_mat, size=(1,3), mode='reflect')
    stopping_bouts = [get_onsets_and_offsets(trial, stopping_n_frames) for trial in stopping_mat]
 
    # Detect escape bout (running bout + end in corner)
    escape_running_threshold = 25 # 25
    escape_corner_threshold  = 25 # 30
    escape_running_n_frames = 5
    # Escape detection -- running bout that ended up in shelter
    escape_running_mat = speed_mat > escape_running_threshold
    # median filter
    escape_running_mat = ndimage.median_filter(escape_running_mat, size=(1,3), mode='reflect')
     # If experiment is manually annotated, use the manual annotation instead of the thresholds
    manual_escape_trials = get_manual_escape_trials(sess_name)
    manual_annotation = False
    print("manual_escape_trials", manual_escape_trials)
    if manual_escape_trials is not None:
        manual_annotation = True
        manual_escape_mat = np.zeros_like(escape_running_mat)
        for trial_idx, escape_t in manual_escape_trials:
            manual_escape_mat[trial_idx, escape_t-5:escape_t+5] = 1
        escape_running_mat = manual_escape_mat
    escape_running_bouts = [get_onsets_and_offsets(trial, escape_running_n_frames) for trial in escape_running_mat]
    print("manual_escape_running_bouts", escape_running_bouts)

    escape_bouts = []
    for trial_idx, trial_bouts in enumerate(escape_running_bouts):
        trial_escape_bouts = []
        trial_speed = speed_mat[trial_idx]
        
        for bout in trial_bouts:
            if (cornerdist_mat[trial_idx][bout[1]] < escape_corner_threshold) or manual_annotation:
                # REFINEMENT: Find true acceleration onset prior to bout[0]
                refined_onset = find_true_escape_onset(
                    speed_trace=trial_speed, 
                    trigger_frame=bout[0], 
                    min_frame=0, 
                    max_lookback_frames=60,  # Look back up to 1 second (at 60 Hz)
                    low_speed_thresh=5.0
                )
                
                # Store bout with refined true onset
                trial_escape_bouts.append((refined_onset, bout[1]))
                
        escape_bouts.append(trial_escape_bouts)
    
    return running_bouts, stopping_bouts, escape_bouts

# ...
def get_correlation(sig_1, sig_2):
    # Assume sig_1_low is (51, 200) and sig_2_low is (200,)
    # 1. Mean-center the signals (required for Pearson Correlation)
    sig_1_c = sig_1 - np.mean(sig_1, axis=1, keepdims=True)
    sig_2_c = sig_2 - np.mean(sig_2)

    # 2. Calculate Cross-Correlation
    # Using mode='same' keeps the output length at 200
    # Zero-lag will be at index 100 (len // 2)
    corr = signal.correlate(sig_1_c, sig_2_c[np.newaxis, :], mode='same', method='fft')

    # 3. Calculate Normalization Factor (the denominator)
    norm1 = np.sqrt(np.sum(sig_1_c**2, axis=1))
    norm2 = np.sqrt(np.sum(sig_2_c**2))
    denom = norm1 * norm2

    # Avoid division by zero for silent channels
    denom[denom == 0] = 1e-10

    # 4. Normalize
    corr_norm = corr / denom[:, np.newaxis]

    # 5. Extract the Peak within 5 bins around 0-lag
    # In scipy.signal.correlate with mode='same', 0-lag is at N // 2
    zero_lag_idx = sig_2.shape[0] // 2
    peak_corrs = corr_norm[:,zero_lag_idx]
    return peak_corrs#[:,0]

#%%
stims_list = []
hfx_frs_mats = []
free_frs_mats = []
running_mats = []
running_frs_mats = []
running_stims_mats = []
stopping_mats = []
stopping_frs_mats = []
stopping_stims_mats = []
escape_mats = []
escape_frs_mats = []
escape_stims_mats = []
speed_downsampled_mats = []
frs_downsampled_mats = []
speed_correlation_mats = []
acc_correlation_mats = []
for key, mat in speed.items():
    # Exclude experiments without escape responses
    if key in [('02583', '844LL', '20260210'), ('02662', '861R', '20260402')]: # ('02584', '843LR', '20260210'), 
        continue
    print(key)
    hfx_stim_idx = 0
    hfx_frs_mats.append(hfx_frs[key][:,hfx_stim_idx])#.shape
    stims_list.append(stims[key])
    print(mat.shape)
    running_bouts, stopping_bouts, escape_bouts = get_ethogram(mat, cornerdist[key], (key[0],key[2]))
    vis_mat = np.zeros_like(mat)
    for trial_idx, trial_bouts in enumerate(running_bouts):
        for bout in trial_bouts:
            vis_mat[trial_idx, bout[0]:bout[1]] = 1
    for trial_idx, trial_bouts in enumerate(stopping_bouts):
        for bout in trial_bouts:
            vis_mat[trial_idx, bout[0]:bout[1]] = -1
    for trial_idx, trial_bouts in enumerate(escape_bouts):
        for bout in trial_bouts:
            vis_mat[trial_idx, bout[0]:bout[1]] = 2

    fig, axs = plt.subplots(1, 2, figsize=(8,3), sharex=True, sharey=True)
    colors = [matplotlib.colors.to_rgba("#224cbe"),
                matplotlib.colors.to_rgba("#727272"),
                matplotlib.colors.to_rgba('#de8f05'),
                matplotlib.colors.to_rgba("#e64716")]
    cmap = LinearSegmentedColormap.from_list("custom_map", colors)
    sb.heatmap(mat, ax=axs[0], cmap='Grays', vmin=0, vmax=40)
    sb.heatmap(vis_mat, ax=axs[1], cmap=cmap, vmin=-1, vmax=2)
    axs[0].set(xticks=np.arange(0, 600, 120), xticklabels=np.arange(0, 600,120)/60)
    # set axs[0] yticklabels to yticklabels=LP_stims[key] with horizontal orientation
    axs[0].set_yticklabels(stims[key], rotation=0)
    axs[1].set(xticks=np.arange(0, 600, 120), xticklabels=np.arange(0, 600,120)/60)
    axs[1].set_yticklabels(stims[key], rotation=0)

    for ax in axs:
        ax.axvline(120, c='k', linestyle='--')
    plt.suptitle(key)
    plt.show()

    # Get specifically for black looming stimuli
    min_t = 0#120# 180 # 240
    max_t = mat.shape[1] # 120
    # Get neural activity at different bout onsets ...
    # Get running bout onsets and offsets (any - incl during stimulus)
    min_bout_length = 15
    if key in [('02583', '844LL', '20260204'), ('02585', '841L', '20260204'), ('02661', '860L', '20260410')]:
        np_open_offset=400-400 # account for missing 1 second of pre-stimulus baseline
        free_stim_kernel = get_free_stim_kernel(stim_onset=400, dur=3400)

    else:
        np_open_offset=600-400 # -400 to remove first 2 seconds baseline of free
        free_stim_kernel = get_free_stim_kernel(stim_onset=600, dur=3400)

    running_mat = []
    running_fr_mat = []
    running_stim_mat = []
    for trial_idx, trial_bouts in enumerate(running_bouts):
        for bout in trial_bouts:
            if bout[1]-bout[0]>min_bout_length:
                if (bout[0]-60>=min_t) and (bout[0]+60<=max_t):
                    print(bout)
                    running_mat.append(speed[key][trial_idx, bout[0]-60:bout[0]+60])
                    running_fr_mat.append(free_frs[key][:,trial_idx,int(np_open_offset+bout[0]*200/60-200):int(np_open_offset+bout[0]*200/60+200)])
                    running_stim_mat.append(free_stim_kernel[int(np_open_offset+bout[0]*200/60-200):int(np_open_offset+bout[0]*200/60+200)])
    running_mat = np.array(running_mat)
    running_fr_mat = np.array(running_fr_mat)
    running_stim_mat = np.array(running_stim_mat)
    running_mats.append(running_mat)
    running_frs_mats.append(running_fr_mat)
    running_stims_mats.append(running_stim_mat)
    
    fig, ax = plt.subplots(1,1,figsize=(2,2))
    sb.heatmap(running_mat, ax=ax, center=10, cmap='tab10', vmin=8, vmax=12)
    plt.show()

    stopping_mat = []
    stopping_fr_mat = []
    stopping_stim_mat = []
    for trial_idx, trial_bouts in enumerate(stopping_bouts):
        for bout in trial_bouts:
            if bout[1]-bout[0]>min_bout_length:
                if (bout[0]-60>=min_t) and (bout[0]+60<=max_t):
                    stopping_mat.append(speed[key][trial_idx, bout[0]-60:bout[0]+60])
                    stopping_fr_mat.append(free_frs[key][:,trial_idx,int(np_open_offset+bout[0]*200/60-200):int(np_open_offset+bout[0]*200/60+200)])
                    stopping_stim_mat.append(free_stim_kernel[int(np_open_offset+bout[0]*200/60-200):int(np_open_offset+bout[0]*200/60+200)])
    stopping_mat = np.array(stopping_mat)
    stopping_fr_mat = np.array(stopping_fr_mat)
    stopping_stim_mat = np.array(stopping_stim_mat)
    stopping_mats.append(stopping_mat)
    stopping_frs_mats.append(stopping_fr_mat)
    stopping_stims_mats.append(stopping_stim_mat)

    escape_mat = []
    escape_fr_mat = []
    escape_stim_mat = []
    for trial_idx, trial_bouts in enumerate(escape_bouts):
        for bout in trial_bouts:
            if bout[1]-bout[0]>min_bout_length:
                if (bout[0]-60>=min_t) and (bout[1]+60<=max_t):
                    escape_mat.append(speed[key][trial_idx, bout[0]-60:bout[0]+60])
                    escape_fr_mat.append(free_frs[key][:,trial_idx,int(np_open_offset+bout[0]*200/60-200):int(np_open_offset+bout[0]*200/60+200)])
                    escape_stim_mat.append(free_stim_kernel[int(np_open_offset+bout[0]*200/60-200):int(np_open_offset+bout[0]*200/60+200)])
    escape_mat = np.array(escape_mat)
    escape_fr_mat = np.array(escape_fr_mat)
    escape_stim_mat = np.array(escape_stim_mat)
    escape_mats.append(escape_mat)
    escape_frs_mats.append(escape_fr_mat)
    escape_stims_mats.append(escape_stim_mat)

    stim_trial_idxs = np.where((np.array(stims[key])=='EXPA')|(np.array(stims[key])=='EXPAL'))[0]
    free_frs_mat = []
    for trial_idx in stim_trial_idxs:
        free_frs_mat.append(free_frs[key][:,trial_idx,np_open_offset:np_open_offset+1200])
    free_frs_mat = np.array(free_frs_mat)
    free_frs_mats.append(free_frs_mat)

    fig, axs = plt.subplots(2, 3, figsize=(6, 2), sharey='row')
    axs[0,0].plot(np.mean(running_mat, axis=0))
    axs[0,1].plot(np.mean(stopping_mat, axis=0))
    
    axs[0,0].axvline(60, c='k', linestyle='--')
    axs[0,1].axvline(60, c='k', linestyle='--')

    sb.heatmap(np.mean(running_fr_mat, axis=(0))-np.mean(running_fr_mat[:,:,:200], axis=(0,2))[:,None], ax=axs[1,0], cmap='vlag', center=0)
    sb.heatmap(np.mean(stopping_fr_mat, axis=(0))-np.mean(stopping_fr_mat[:,:,:200], axis=(0,2))[:,None], ax=axs[1,1], cmap='vlag', center=0)

    if len(escape_fr_mat) > 0:
        axs[0,2].plot(np.mean(escape_mat, axis=0))
        # axs[1,2].plot(np.mean(escape_fr_mat, axis=(0,1)))
        sb.heatmap(np.mean(escape_fr_mat, axis=(0)), ax=axs[1,2])
        axs[0,2].axvline(60, c='k', linestyle='--')
    plt.show()

    # show visual responses
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    mean_fr_responses = np.mean(free_frs[key][:,:,np_open_offset:np_open_offset+2000], axis=1)
    mean_stim_responses = free_stim_kernel[np_open_offset:np_open_offset+2000]
    # mean_fr_responses = np.mean(free_frs[key][:,:,], axis=0)
    sb.heatmap(mean_fr_responses, ax=ax, cmap='Grays')
    ax.plot(mean_stim_responses*50, c='r')
    ax.axvline(400, c='r', linestyle='--')
    ax.axvline(600, c='r', linestyle='--')
    ax.axvline(800, c='r', linestyle='--')
    plt.show()

    # Calculate correlation between speed and neural activity
    speed_correlations = []
    acc_correlations = []
    speed_downsampled = []
    frs_downsampled = []
    for trial_idx in range(speed[key].shape[0]):
        trial_speed = speed[key][trial_idx] # (600)
        trial_frs = free_frs[key][:,trial_idx] # (n_neurons x 3400)
        # downsample free_frs from 200Hz to 60Hz
        trial_frs = trial_frs[:,np_open_offset:np_open_offset+2000] # (n_neurons x 2000)
        trial_frs = signal.resample_poly(trial_frs, up=1, down=10, axis=1)
        trial_speed = signal.resample_poly(trial_speed, up=1, down=3)
        trial_acceleration = np.diff(trial_speed, axis=0)
        print(trial_frs.shape)
        speed_correlations.append(get_correlation(trial_frs, trial_speed))
        acc_correlations.append(get_correlation(trial_frs[:,:-1], trial_acceleration))
        speed_downsampled.append(trial_speed)
        frs_downsampled.append(trial_frs)
    
    speed_correlations = np.array(speed_correlations)
    acc_correlations = np.array(acc_correlations)
    speed_downsampled = np.array(speed_downsampled)
    frs_downsampled = np.array(frs_downsampled)
    speed_downsampled_mats.append(speed_downsampled)
    frs_downsampled_mats.append(frs_downsampled)
    speed_correlation_mats.append(speed_correlations)
    acc_correlation_mats.append(acc_correlations)

    fig, axs = plt.subplots(1,3,figsize=(9,3))
    sb.heatmap(speed_correlations.T, ax=axs[0], cmap='vlag', center=0, vmin=-1, vmax=1)
    sb.heatmap(acc_correlations.T, ax=axs[1], cmap='vlag', center=0, vmin=-1, vmax=1)
    axs[2].scatter(np.mean(speed_correlations, axis=0), np.mean(acc_correlations, axis=0))
    axs[2].axvline(0, c='k', linestyle='--')
    axs[2].axhline(0, c='k', linestyle='--')
    axs[2].set(xlim=(-0.5,0.5), ylim=(-0.5,0.5), xlabel='Speed correlation', ylabel='Acceleration correlation')
    plt.tight_layout()
    plt.show()

#%% Combined heatmap showing escapes (running speed)
stim_speed_mats_dict = {}
escape_latency_dict = {}
for key, mat in speed.items():
    # Exclude experiments without escape responses
    if key in [('02583', '844LL', '20260210'), ('02662', '861R', '20260402')]:
        continue
    print(key)
    stims_list = stims[key]
    _, _, escape_bouts = get_ethogram(mat, cornerdist[key], (key[0], key[2]))
    
    print(mat.shape, len(escape_bouts))
    for i, (stim, bouts) in enumerate(zip(stims_list, escape_bouts)):
        if stim in ['EXPAL', 'EXPWL', 'DIMML']:
            stim = stim[:-1]
        if stim not in stim_speed_mats_dict:
            stim_speed_mats_dict[stim] = []
            escape_latency_dict[stim] = []
        stim_speed_mats_dict[stim].append(mat[i])
        escape_latency_dict[stim].append(bouts[0][0] if (len(bouts)>0) else 999)

stims_to_plot = ['EXPA', 'EXPW', 'DIMM']
cmap = 'Grays'
stim_intervals = [(120, 150), (180, 210), (240, 270)]
fig, axs = plt.subplots(1,len(stims_to_plot), figsize=(len(stims_to_plot)*cm*3.5,4*cm), constrained_layout=True)
for stim, ax in zip(stims_to_plot, axs):
    sort_idxs = np.argsort(escape_latency_dict[stim])
    im = ax.imshow(np.array(stim_speed_mats_dict[stim])[:,int(60*1):int(60*2+60*5)][sort_idxs], aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=50)
    ax.set_title(stim)
    ax.set(xticks=[60, 120, 180, 240, 300], xticklabels=[-1, 0, 1, 2, 3], xlabel='time (s)', yticks=[0,len(sort_idxs)-1])
    ax.tick_params(axis='y', which='both', length=0)
    for start, end in stim_intervals:
        ax.plot([start-60, end-60], [1.03, 1.03], color='k', linewidth=1.5, 
            transform=ax.get_xaxis_transform(), clip_on=False, solid_capstyle='butt')
    ax.axvline(60, c='r', linestyle='--')
cbar = fig.colorbar(im, ax=axs, location="right", shrink=0.8)
cbar.set_label("Speed")  # Adjust label as needed

# plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'behavior_heatmaps.pdf'), dpi=300)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Process and concatenate all matrices vertically
stacked_mats = []
boundaries = []
y_tick_positions = []
current_y = 0

for stim in stims_to_plot:
    sort_idxs = np.argsort(escape_latency_dict[stim])
    mat = np.array(stim_speed_mats_dict[stim])[:, int(60*1):int(60*2+60*5)][sort_idxs]
    
    num_rows = mat.shape[0]
    stacked_mats.append(mat)
    
    # Store midpoint for labeling each stimulus on the Y-axis
    y_tick_positions.append(current_y + num_rows / 2)
    
    # Track row boundary
    current_y += num_rows
    boundaries.append(current_y)

# Combine into a single matrix
combined_mat = np.vstack(stacked_mats)

# 2. Plot single heatmap
fig, ax = plt.subplots(figsize=(6*cm, 8*cm), constrained_layout=True)  # Adjust figure height as needed

im = ax.imshow(combined_mat, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=50)

# 3. Draw horizontal separating lines (-0.5 positions line exactly on pixel grid boundary)
for boundary in boundaries[:-1]:  # Exclude final boundary at the very bottom
    ax.axhline(boundary - 0.5, color='r', linewidth=1.5)  # Change color to 'black' or 'k' if preferred

# 4. Set Y-axis labels at the center of each condition block
ax.set_yticks(y_tick_positions)
ax.set_yticklabels(stims_to_plot)
ax.tick_params(axis='y', which='both', length=0)

# 5. Set X-axis ticks
ax.set(xticks=[60, 120, 180, 240, 300], xticklabels=[-1, 0, 1, 2, 3], xlabel='time (s)')

# 6. Add stimulus top bars & vertical marker
for start, end in stim_intervals:
    ax.plot([start-60, end-60], [1.02, 1.02], color='k', linewidth=1.5, 
            transform=ax.get_xaxis_transform(), clip_on=False, solid_capstyle='butt')

ax.axvline(60, c='r', linestyle='--')

# 7. Add colorbar
cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.8)
cbar.set_label("Speed")

# plt.savefig(os.path.join(fig_path, 'behavior_heatmaps.pdf'), dpi=300)
plt.show()

#%% Combine data from the experiment - visualize
fig = plt.figure(constrained_layout=True, figsize=(8,3))
gs = fig.add_gridspec(2,5, width_ratios=[1,1,1,1,1], height_ratios=[1, 2])
ax00 = fig.add_subplot(gs[0,0])
ax01 = fig.add_subplot(gs[0,1])
ax02 = fig.add_subplot(gs[0,2])
ax10 = ax00.twinx()
ax11 = ax01.twinx()
ax11.sharey(ax10)
ax12 = ax02.twinx()
ax12.sharey(ax10)
ax13 = fig.add_subplot(gs[0,3])
ax14 = fig.add_subplot(gs[0,4])
ax20 = fig.add_subplot(gs[1,0], sharex=ax10)
ax21 = fig.add_subplot(gs[1,1], sharex=ax11)
ax22 = fig.add_subplot(gs[1,2], sharex=ax12)
ax23 = fig.add_subplot(gs[1,3], sharex=ax13)
ax24 = fig.add_subplot(gs[1,4], sharex=ax14)

escape_mats_filt = [f for f in escape_mats if len(f)>0]
escape_stims_mats_filt = [f for f in escape_stims_mats if len(f)>0]
escape_frs_mats_filt = [f for f in escape_frs_mats if len(f)>0]

mean_running = np.mean(np.concatenate(running_mats, axis=0), axis=0)
mean_stopping = np.mean(np.concatenate(stopping_mats, axis=0), axis=0)
std_running = np.std(np.concatenate(running_mats, axis=0), axis=0)
std_stopping = np.std(np.concatenate(stopping_mats, axis=0), axis=0)

ax10.plot(np.arange(0,len(mean_running))*200/60,mean_running, c='k')
ax11.plot(np.arange(0,len(mean_stopping))*200/60,mean_stopping, c='k')

mean_escape = np.mean(np.concatenate(escape_mats_filt, axis=0), axis=0)
ax12.plot(np.arange(0,len(mean_escape))*200/60, mean_escape, c='k')

mean_visual_responses = np.mean(np.concatenate(hfx_frs_mats, axis=0), axis=1)[:,200:1000]

mean_running_responses = np.vstack([np.mean(running_frs_mat, axis=0) for running_frs_mat in running_frs_mats])
mean_stopping_responses = np.vstack([np.mean(stopping_frs_mat, axis=0) for stopping_frs_mat in stopping_frs_mats])
mean_escape_responses = np.vstack([np.mean(escape_frs_mat, axis=0) for escape_frs_mat in escape_frs_mats_filt])
mean_free_visual_responses = np.vstack([np.mean(free_frs_mat, axis=0) for free_frs_mat in free_frs_mats])

# Sort responses... (time to peak | correlation to running speed)
mean_running_responses_sorted_idxs = np.argsort(np.argmax(mean_running_responses, axis=1))
mean_stopping_responses_sorted_idxs = np.argsort(np.argmax(mean_stopping_responses, axis=1))
mean_escape_responses_sorted_idxs = np.argsort(np.argmax(mean_escape_responses, axis=1))
mean_visual_responses_sorted_idxs = np.argsort(np.argmax(mean_visual_responses, axis=1))
mean_free_visual_responses_sorted_idxs = np.argsort(np.argmax(mean_free_visual_responses, axis=1))

cells_mean_speed_correlations = np.concatenate([np.mean(corrs, axis=0) for corrs in speed_correlation_mats])
cells_mean_acc_correlations = np.concatenate([np.mean(corrs, axis=0) for corrs in acc_correlation_mats])
correlation_sorted_idxs = np.argsort(cells_mean_speed_correlations)
sort_on = 'time_to_peak' # time_to_peak
if sort_on == 'time_to_peak':
    mean_running_responses = mean_running_responses[mean_running_responses_sorted_idxs]
    mean_stopping_responses = mean_stopping_responses[mean_stopping_responses_sorted_idxs]
    mean_escape_responses = mean_escape_responses[mean_escape_responses_sorted_idxs]
    mean_visual_responses = mean_visual_responses[mean_visual_responses_sorted_idxs]
    mean_free_visual_responses = mean_free_visual_responses[mean_visual_responses_sorted_idxs]
elif sort_on =='correlation':
    mean_running_responses = mean_running_responses[correlation_sorted_idxs]
    mean_stopping_responses = mean_stopping_responses[correlation_sorted_idxs]
    mean_visual_responses = mean_visual_responses[correlation_sorted_idxs]

mean_running_responses_norm = (mean_running_responses-np.mean(mean_running_responses, axis=-1)[:,None])/(1+np.std(mean_running_responses, axis=-1))[:,None]
mean_stopping_responses_norm = (mean_stopping_responses-np.mean(mean_stopping_responses, axis=-1)[:,None])/(1+np.std(mean_stopping_responses, axis=-1))[:,None]
mean_escape_responses_norm = (mean_escape_responses-np.mean(mean_escape_responses, axis=-1)[:,None])/(1+np.std(mean_escape_responses, axis=-1))[:,None]
mean_visual_responses_norm = (mean_visual_responses-np.mean(mean_visual_responses, axis=-1)[:,None])/(1+np.std(mean_visual_responses, axis=-1))[:,None]
mean_free_visual_responses_norm = (mean_free_visual_responses-np.mean(mean_free_visual_responses, axis=-1)[:,None])/(1+np.std(mean_free_visual_responses, axis=-1))[:,None]

T_behav_frs = mean_running_responses_norm.shape[-1]
T_free_visual_frs = mean_free_visual_responses_norm.shape[-1]
T_hfx_visual_frs = mean_visual_responses_norm.shape[-1]

ax00.plot(np.arange(T_behav_frs), np.mean(mean_running_responses_norm, axis=0), c=color_area)
ax01.plot(np.arange(T_behav_frs), np.mean(mean_stopping_responses_norm, axis=0), c=color_area)
ax02.plot(np.arange(T_behav_frs), np.mean(mean_escape_responses_norm, axis=0), c=color_area)
ax14.plot(np.arange(T_hfx_visual_frs), np.mean(mean_visual_responses_norm, axis=0), c=color_area)
ax13.plot(np.arange(T_free_visual_frs), np.mean(mean_free_visual_responses_norm, axis=0), c=color_area)

def draw_imshow(data, ax, cmap='Grays'):
    im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')
    return im

# Replace your seaborn calls with these:
draw_imshow(mean_running_responses_norm, ax20)
draw_imshow(mean_stopping_responses_norm, ax21)
draw_imshow(mean_escape_responses_norm, ax22)
draw_imshow(mean_free_visual_responses_norm, ax23)
draw_imshow(mean_visual_responses_norm, ax24)
ax20.axvline(200, c='r', linestyle='--')
ax21.axvline(200, c='r', linestyle='--')
ax22.axvline(200, c='r', linestyle='--')
ax23.axvline(400, c='r', linestyle='--')
ax24.axvline(200, c='r', linestyle='--')

ax10.set(ylabel='running speed')
ax00.set(ylabel='mean normalized\nfiring rate')
ax20.set(ylabel='neurons')
ax00.axvline(1*200, c='r', linestyle='--')
ax01.axvline(1*200, c='r', linestyle='--')
ax02.axvline(1*200, c='r', linestyle='--')
ax13.axvline(2*200, c='r', linestyle='--')
ax13.axvline(3*200, c='k', linestyle='--')
ax13.axvline(4*200, c='k', linestyle='--')
ax14.axvline(1*200, c='r', linestyle='--')
ax14.axvline((1+(0.85))*200, c='k', linestyle='--')
ax14.axvline((1+(2*0.85))*200, c='k', linestyle='--')

ax10.set(title='running')
ax11.set(title='stopping')
ax12.set(title='escape')
ax13.set(title='openfield\nvisual responses')
ax14.set(title='head-fixed\nvisual responses')
plt.savefig(os.path.join(fig_path, 'behavior_neural_alignment.pdf'), dpi=300, bbox_inches='tight')

plt.show()


#%% --- Visualize AUC before escape
fig, ax = plt.subplots(1,1, figsize=(4*cm, 4*cm))
ax_twin = ax.twinx()
responses_cumulative = np.cumsum(mean_escape_responses_norm, axis=-1) # TODO: Replace with mean_escape_responses 
ax.plot(np.arange(T_behav_frs)/200, np.mean(responses_cumulative, axis=0), c=color_area, linewidth=2)
ax_twin.plot(np.arange(0,len(mean_escape))/60, mean_escape, c='k')
# ax.axvline(np.argmax(np.mean(responses_cumulative, axis=0))/200, c='k', linestyle='--')
ax.axvline(1, c='k', linestyle='--')

ax.set(ylabel='AUC', xlabel='Time (s)')
ax_twin.set(ylabel='Speed (cm/s)')
plt.savefig(os.path.join(fig_path, 'escape_neural_cumulative.pdf'), dpi=300, bbox_inches='tight')
plt.show()

#%% === ROC AUC cumulative response with Null Distribution ===
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# --- 1. Define Window Parameters (-2.0s to +1.0s relative to Escape Onset) ---
fs = 200  # Sampling rate in Hz
pre_sec = 2.0
post_sec = 1.0

pre_bins = int(pre_sec * fs)    # 400 bins
post_bins = int(post_sec * fs)  # 200 bins
win_bins = pre_bins + post_bins  # 600 bins total

# Time vector spanning -2.0s to +1.0s (t = 0 at index pre_bins)
t_vec = (np.arange(win_bins) - pre_bins) / float(fs)
zero_idx = pre_bins  # Index corresponding to t = 0.0s (Escape Onset)

# Define Evaluation Window around t = 0s (-50ms to +50ms)
window_ms = 100# 100
window_bins = int((window_ms / 1000.0) * fs)  # 10 bins = 50 ms
eval_window_indices = np.arange(zero_idx - window_bins, zero_idx + window_bins + 1)

# --- 2. Extract Latencies and Prepare Trial Data ---
trials_info = []
observed_latencies_bins = []  # Escape latencies in 200Hz bin offsets relative to stim onset

for key in speed.keys():
    # Exclude experiments without escape trials
    if key in [('02583', '844LL', '20260210'), ('02662', '861R', '20260402')]: # ('02584', '843LR', '20260210')
        continue
    running_bouts, stopping_bouts, escape_bouts = get_ethogram(speed[key], cornerdist[key], (key[0], key[2]))
    expa_trial_idxs = [i for i, s in enumerate(stims[key]) if s in ['EXPA','EXPW','DIMM']] # Left out EXPAL to avoidtrials whre the animal did not see the stimulus
    print(key, len(expa_trial_idxs))
    np_open_offset = 0 if key in [('02583', '844LL', '20260204'), ('02585', '841L', '20260204'), ('02661', '860L', '20260410')] else 200
    stim_onset_bin = np_open_offset + 400

    sess_frs = free_frs[key]

    for t_idx in expa_trial_idxs:
        # Check for escape bouts post stimulus onset (frame 120 = 2.0s at 60Hz video)
        valid_escapes = [b for b in escape_bouts[t_idx] if b[0] >= 120] if len(escape_bouts[t_idx]) > 0 else []
        has_escape = len(valid_escapes) > 0

        # Baseline-subtracted population firing rate trace
        tr_frs = sess_frs[:, t_idx, :]
        base = np.mean(tr_frs[:, stim_onset_bin - 200 : stim_onset_bin], axis=-1, keepdims=True)
        base_std = np.std(tr_frs[:, stim_onset_bin - 200 : stim_onset_bin], axis=-1, keepdims=True)

        pop_trace = np.mean((tr_frs - base)/(base_std+1), axis=0)  # Average across cells

        if has_escape:
            esc_frame = valid_escapes[0][0]
            lat_bins = int((esc_frame - 120) * (200.0 / 60.0))
            observed_latencies_bins.append(lat_bins)
            esc_bin = stim_onset_bin + lat_bins
        else:
            lat_bins = None
            esc_bin = None

        trials_info.append({
            'is_escape': 1 if has_escape else 0,
            'stim_onset_bin': stim_onset_bin,
            'escape_bin': esc_bin,
            'pop_trace': pop_trace
        })

observed_latencies_bins = np.array(observed_latencies_bins)
print(f"Collected {len(trials_info)} trials across sessions.")
print(f"Observed escape latencies: {np.mean(observed_latencies_bins)/200:.2f}s +/- {np.std(observed_latencies_bins)/200:.2f}s")

# --- 3. Helper Function: Extract -2s to +1s Cumulative Traces ---
def build_aligned_cum_traces(trials_info, latencies_pool, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    cum_traces = []
    labels = []

    for trial in trials_info:
        if trial['is_escape']:
            align_bin = trial['escape_bin']
        else:
            mock_lat_bin = np.random.choice(latencies_pool)
            align_bin = trial['stim_onset_bin'] + mock_lat_bin

        start_bin = align_bin - pre_bins
        end_bin = align_bin + post_bins

        trace = trial['pop_trace']
        
        pad_left = max(0, -start_bin)
        pad_right = max(0, end_bin - len(trace))
        
        slice_start = max(0, start_bin)
        slice_end = min(len(trace), end_bin)
        
        window = trace[slice_start:slice_end]
        if pad_left > 0 or pad_right > 0:
            window = np.pad(window, (pad_left, pad_right), mode='edge')

        cum_traces.append(np.cumsum(window)) # TODO:re-enable!
        # cum_traces.append(window)
        labels.append(trial['is_escape'])

    return np.array(cum_traces), np.array(labels)

# --- 4. Compute Observed ROC AUC Across Time ---
n_mock_draws = 500
roc_auc_time_draws = np.zeros((n_mock_draws, win_bins))

for d in range(n_mock_draws):
    X_cum, y_true = build_aligned_cum_traces(trials_info, observed_latencies_bins, random_state=d)
    for t in range(win_bins):
        roc_auc_time_draws[d, t] = roc_auc_score(y_true, X_cum[:, t])

roc_auc_time = np.mean(roc_auc_time_draws, axis=0)

# Average ROC AUC within the [-50ms, +50ms] window
final_auc_val = np.mean(roc_auc_time[eval_window_indices])

# --- 5. Bootstrapping & Permutation Null (Evaluated over [-50ms, +50ms] window) ---
n_iterations = 1000
boot_window_aucs = np.zeros(n_iterations)
null_window_aucs = np.zeros(n_iterations)
null_auc_time = np.zeros((n_iterations, win_bins))

for i in range(n_iterations):
    print(i)
    # Bootstrap Observed Data
    X_cum, y_true = build_aligned_cum_traces(trials_info, observed_latencies_bins)
    n_trials = len(y_true)
    
    boot_idxs = np.random.choice(n_trials, size=n_trials, replace=True)
    while len(np.unique(y_true[boot_idxs])) < 2:
        boot_idxs = np.random.choice(n_trials, size=n_trials, replace=True)
        
    # Mean ROC AUC across window for this bootstrap draw
    boot_aucs = [roc_auc_score(y_true[boot_idxs], X_cum[boot_idxs, t]) for t in eval_window_indices]
    boot_window_aucs[i] = np.mean(boot_aucs)

    # Label Permutation Null
    y_shuffled = np.random.permutation(y_true)
    for t in range(win_bins):
        null_auc_time[i, t] = roc_auc_score(y_shuffled, X_cum[:, t])
    
    null_window_aucs[i] = np.mean(null_auc_time[i, eval_window_indices])

observed_sd = np.std(boot_window_aucs)
null_mean = np.mean(null_window_aucs)
null_sd = np.std(null_window_aucs)
null_time_mean = np.mean(null_auc_time, axis=0)

#%% Permutation Test p-value over the [-50ms, +50ms] window
p_val_perm = (np.sum(null_window_aucs >= final_auc_val)) / (n_iterations)

print("=" * 60)
print(f"dSC ROC AUC (Window [-50ms, +50ms]): {final_auc_val:.3f} +/- {observed_sd:.3f} (S.D.)")
print(f"Null ROC AUC (Window [-50ms, +50ms]): {null_mean:.3f} +/- {null_sd:.3f} (S.D.)")
print(f"Permutation Test:                      p = {p_val_perm:.4e}")
print("=" * 60)


# --- 6. Plotting Figure ---
fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(6.5*cm, 5.0*cm), gridspec_kw={'width_ratios': [1.2, 2]})

# --- Left Panel: Barplot for Windowed ROC AUC ---
ax_bar.bar(['dSC', 'Null'], [final_auc_val, null_mean], 
           yerr=[observed_sd, null_sd], 
           color=[color_area, '#8c8c8c'], 
           capsize=3, width=0.5, error_kw={'lw': 1.2})

ax_bar.set_ylim([0.4, 1.0])
ax_bar.axhline(0.5, color='k', linestyle='--', lw=0.8, alpha=0.7)
ax_bar.set_ylabel('ROC AUC', fontsize=10)

def p_to_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'n.s.'

y_max = max(final_auc_val + observed_sd, null_mean + null_sd) + 0.04
ax_bar.plot([0, 0, 1, 1], [y_max, y_max + 0.02, y_max + 0.02, y_max], color='k', lw=1.0)
ax_bar.text(0.5, y_max + 0.03, p_to_stars(p_val_perm), ha='center', va='bottom', fontsize=11, fontweight='bold')

sb.despine(ax=ax_bar)

# --- Right Panel: Time-Series relative to Escape Onset ---
ax_line.plot(t_vec, roc_auc_time, color=color_area, lw=1.8, label='dSC')
ax_line.fill_between(t_vec, 0.5, roc_auc_time, color=color_area, alpha=0.15)

# Null baseline overlay
ax_line.plot(t_vec, null_time_mean, color='#8c8c8c', lw=1.2, linestyle='--', label='Null')

# Dotted vertical line at Escape Onset (t = 0s)
ax_line.axvline(0, color='k', linestyle=':', lw=1.2)
ax_line.set_ylim([0.45, max(0.75, np.max(roc_auc_time) + 0.05)])
ax_line.set_xlabel('Time from escape onset (s)', fontsize=10)
ax_line.set_ylabel('ROC AUC', fontsize=10)

sb.despine(ax=ax_line)

plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'roc_auc_escape_aligned_windowed.pdf'), dpi=300, bbox_inches='tight')
plt.show()

#%% --- Cell responses during no escape vs escape trials ---
# Storage for cell traces (for line plots)
escape_cell_means = []
no_escape_cell_means = []

# Storage for trial-level population peaks (for statistical test)
esc_peaks = []
no_esc_peaks = []

# Bin and frame thresholds
stim_onset_frame = 120  # Frame 120 = 2.0s in 60 Hz behavior video
peak_win_start = 200    # 0.0s post-stimulus (bin 200 in -1.0s to +5.5s trace)
peak_win_end   = 300    # 0.5s post-stimulus (100 bins = 500ms)

# --- Single Loop: Extract Traces & Peak Responses ---
for key in speed.keys():
    # Exclude experiments without escape trials
    if key in [('02583', '844LL', '20260210'), ('02662', '861R', '20260402')]: # ('02584', '843LR', '20260210')
        continue

    # 1. Detect ethograms & EXPA/EXPAL trial indices
    running_bouts, stopping_bouts, escape_bouts = get_ethogram(speed[key], cornerdist[key], (key[0], key[2]))
    expa_trial_idxs = [i for i, s in enumerate(stims[key]) if s in ['EXPA', 'EXPAL']]
    np_open_offset = 0 if key in [('02583', '844LL', '20260204'), ('02585', '841L', '20260204'), ('02661', '860L', '20260410')] else 200
    stim_onset_bin = np_open_offset + 400
    sess_frs = free_frs[key]

    win_start = stim_onset_bin - 200   # -1.0 s
    win_end   = stim_onset_bin + 1100  # +5.5 s

    escape_trial_idxs = []
    no_escape_trial_idxs = []

    # 2. Classify trials as Escape vs No Escape
    for t_idx in expa_trial_idxs:
        has_escape = any(b[0] >= stim_onset_frame for b in escape_bouts[t_idx]) if len(escape_bouts[t_idx]) > 0 else False
        if has_escape:
            escape_trial_idxs.append(t_idx)
        else:
            no_escape_trial_idxs.append(t_idx)

        # Calculate population average trace and peak response for this trial
        tr_frs = sess_frs[:, t_idx, win_start:win_end]
        tr_frs_bs = tr_frs - np.mean(tr_frs[:, :200], axis=-1, keepdims=True)  # Baseline subtract
        pop_trial_trace = np.mean(tr_frs_bs, axis=0)                          # Average across cells
        peak_val = np.max(pop_trial_trace[peak_win_start:peak_win_end])      # Peak in 0 - 0.5s window

        if has_escape:
            esc_peaks.append(peak_val)
        else:
            no_esc_peaks.append(peak_val)
    # Ensure recording of the same population!
    if (len(escape_trial_idxs)>0) and (len(no_escape_trial_idxs)>0):
        # 3. Extract cell-level traces across all condition trials
        if len(escape_trial_idxs) > 0:
            esc_resp = np.mean(sess_frs[:, escape_trial_idxs, win_start:win_end], axis=1)
            esc_resp_bs = esc_resp - np.mean(esc_resp[:, :200], axis=-1, keepdims=True)
            escape_cell_means.append(esc_resp_bs)

        if len(no_escape_trial_idxs) > 0:
            no_esc_resp = np.mean(sess_frs[:, no_escape_trial_idxs, win_start:win_end], axis=1)
            no_esc_resp_bs = no_esc_resp - np.mean(no_esc_resp[:, :200], axis=-1, keepdims=True)
            no_escape_cell_means.append(no_esc_resp_bs)

# Convert arrays
all_escape_cells = np.vstack(escape_cell_means) if len(escape_cell_means) > 0 else np.array([])
all_no_escape_cells = np.vstack(no_escape_cell_means) if len(no_escape_cell_means) > 0 else np.array([])
esc_peaks = np.array(esc_peaks)
no_esc_peaks = np.array(no_esc_peaks)

print("escape cells:", all_escape_cells.shape, "no escape cells:", all_no_escape_cells.shape)
u_stat, p_val = mannwhitneyu(np.mean(all_escape_cells[:,200:300], axis=-1), np.mean(all_no_escape_cells[:,200:300], axis=-1), alternative='two-sided')

# Print statistical summary
print("=" * 50)
print(f"Escape trials   (n = {len(all_escape_cells)}): {np.mean(np.mean(all_escape_cells[:,200:300], axis=-1)):.2f} +/- {np.std(np.mean(all_escape_cells[:,200:300], axis=-1))/np.sqrt(len(all_escape_cells)):.2f} Hz (Mean +/- SEM)")
print(f"No Escape trials (n = {len(all_no_escape_cells)}): {np.mean(np.mean(all_no_escape_cells[:,200:300], axis=-1)):.2f} +/- {np.std(np.mean(all_escape_cells[:,200:300], axis=-1))/np.sqrt(len(all_no_escape_cells)):.2f} Hz (Mean +/- SEM)")

print(f"Escape trials   (n = {len(esc_peaks)}): {np.mean(esc_peaks):.2f} +/- {np.std(esc_peaks)/np.sqrt(len(esc_peaks)):.2f} Hz (Mean +/- SEM)")
print(f"No Escape trials (n = {len(no_esc_peaks)}): {np.mean(no_esc_peaks):.2f} +/- {np.std(no_esc_peaks)/np.sqrt(len(no_esc_peaks)):.2f} Hz (Mean +/- SEM)")
print("-" * 50)
print(f"Mann-Whitney U Test (Peak 0-0.5s): U = {u_stat:.1f}, p = {p_val:.4e}")
print("=" * 50)
# Compute time vector (seconds), means, and SEMs
t_vec = (np.arange(1300) - 200) / 200.0

mean_no_escape = np.mean(all_no_escape_cells, axis=0)
sem_no_escape = np.std(all_no_escape_cells, axis=0) / np.sqrt(all_no_escape_cells.shape[0])

mean_escape = np.mean(all_escape_cells, axis=0)
sem_escape = np.std(all_escape_cells, axis=0) / np.sqrt(all_escape_cells.shape[0])

# % Calculate percentage of No Escape vs Escape trials
total_trials = len(all_no_escape_cells) + len(all_escape_cells)
pct_no_escape = int(round((len(all_no_escape_cells) / total_trials) * 100)) if total_trials > 0 else 0

fig, axs = plt.subplots(2, 1, figsize=(4*cm, 6*cm), sharex=True, sharey=True)

axs[0].plot(t_vec, mean_no_escape, color=color_area, lw=2)
axs[0].fill_between(t_vec, mean_no_escape - sem_no_escape, mean_no_escape + sem_no_escape, color=color_area, alpha=0.25)
axs[0].axvline(0, color='k', linestyle=':', lw=1.5)
axs[0].axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.5)

axs[0].text(-0.3, 0.5, 'No Escape', transform=axs[0].transAxes, rotation=90, va='center', ha='right')

# Bottom Panel: Escape
axs[1].plot(t_vec, mean_escape, color=color_area, lw=2)
axs[1].fill_between(t_vec, mean_escape - sem_escape, mean_escape + sem_escape, color=color_area, alpha=0.25)
axs[1].axvline(0, color='k', linestyle=':', lw=1.5)
axs[1].axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.5)

axs[1].text(-0.3, 0.5, 'Escape', transform=axs[1].transAxes, rotation=90, va='center', ha='right')

for ax in axs:
    sb.despine(ax=ax)
    ax.set_xlim([-1.0, 5.5])

# Formatting subplots
sb.despine(ax=ax)
ax.set_xlim([-1.0, 1])
ax.set_xlabel('Time (s)', fontsize=12)
# ax.set_ylim([None, 5])
# plt.legend()
plt.suptitle(f'% no escape: {pct_no_escape}%')

plt.tight_layout()

# Save output figure
plt.savefig(os.path.join(fig_path, 'escape_vs_no_escape_responses.pdf'), dpi=300, bbox_inches='tight')
plt.show()