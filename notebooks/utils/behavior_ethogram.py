import numpy as np
from scipy import ndimage, signal



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
    # filter bouts by min_duration
    assert len(onsets) == len(offsets)
    bouts = []
    for onset, offset in zip(onsets, offsets):
        if offset-onset >= min_duration:
            # print(onset, offset)
            bouts.append((onset, offset))
    return bouts

# Ethogram
def get_ethogram(speed_mat, cornerdist_mat):
    # smooth speed_mat using gaussian filter
    speed_mat = ndimage.gaussian_filter(speed_mat, sigma=(0,5), order=0, mode='reflect')

    # Detect running bout
    running_threshold = 10# 10
    running_n_frames = 5
    running_mat = speed_mat > running_threshold
    # median filter
    print("running_mat shape", running_mat.shape)
    running_mat = ndimage.median_filter(running_mat, size=(1,3), mode='reflect')
    running_bouts = [get_onsets_and_offsets(trial, running_n_frames) for trial in running_mat]
    # fill vis_mat with ones when the animal is running
    
    
    # Detect stopping bout
    stopping_threshold = 5
    stopping_n_frames = 5
    stopping_mat = speed_mat < stopping_threshold
    # for each trial, extract bout beginning and ending where animal stopped for more than stopping_n_frames
    stopping_mat = ndimage.median_filter(stopping_mat, size=(1,3), mode='reflect')
    stopping_bouts = [get_onsets_and_offsets(trial, stopping_n_frames) for trial in stopping_mat]
 
    # Detect escape bout (running bout + end in corner)
    escape_running_threshold = 25
    escape_corner_threshold  = 25 # 30
    escape_running_n_frames = 5
    # Escape detection -- running bout that ended up in shelter
    escape_running_mat = speed_mat > escape_running_threshold
    # median filter
    escape_running_mat = ndimage.median_filter(escape_running_mat, size=(1,3), mode='reflect')
    escape_running_bouts = [get_onsets_and_offsets(trial, escape_running_n_frames) for trial in escape_running_mat]
    escape_bouts = []
    for trial_idx, trial_bouts in enumerate(escape_running_bouts):
        trial_escape_bouts = []
        for bout in trial_bouts:
            if cornerdist_mat[trial_idx][bout[1]] < escape_corner_threshold:
                trial_escape_bouts.append(bout)
        escape_bouts.append(trial_escape_bouts)
    
    return running_bouts, stopping_bouts, escape_bouts
