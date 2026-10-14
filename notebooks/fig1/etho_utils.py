
import numpy as np
from scipy.ndimage import gaussian_filter1d
import kfopenfield

def get_running_bouts(df, speed_thr=3, dur_thr=10):
    running_vec = np.zeros_like(df['measures', 'speed'])
    running_vec[df['measures', 'speed'] > speed_thr] = 1
    running_vec_diff = np.diff(running_vec)
    running_onsets, running_offsets = np.where(running_vec_diff == 1)[0], np.where(running_vec_diff == -1)[0]
    if (len(running_onsets) == 0) and (len(running_offsets) != 0):
        running_onsets = [0]
    elif (len(running_onsets) != 0) and (len(running_offsets) == 0):
        running_offsets = [len(df['measures', 'speed'])-1]
    elif (len(running_onsets) == 0) and (len(running_offsets) == 0):
        return [], []
    
    if running_offsets[0] < running_onsets[0]:
        running_onsets = np.insert(running_onsets, 0, 0) # Add 0 to the beginning of running_onsets
    if running_onsets[-1] > running_offsets[-1]:
        running_offsets = np.append(running_offsets, len(df['measures', 'speed'])-1)

    # Filter out bouts that are too short
    include_bouts = np.where(running_offsets - running_onsets > dur_thr)[0]
    if len(include_bouts) == 0:
        return [], []

    # Necessary check to avoid errors with indexing a one dimensional array
    if (len(running_onsets) == 1) and len(include_bouts)==1:
        return running_onsets, running_offsets
    
    running_onsets, running_offsets = running_onsets[include_bouts], running_offsets[include_bouts]
    return running_onsets, running_offsets

def get_stopping_bouts(df):
    stopping_vec = np.zeros_like(df['measures', 'speed'])
    stopping_vec[df['measures', 'speed'] < 2] = 1 # used to be < 1 - changed it to 2 to match freezing detection
    stopping_vec_diff = np.diff(stopping_vec)
    stopping_onsets, stopping_offsets = np.where(stopping_vec_diff == 1)[0], np.where(stopping_vec_diff == -1)[0]
    if (len(stopping_onsets) == 0) and (len(stopping_offsets) != 0):
        stopping_onsets = [0]
    elif (len(stopping_onsets) != 0) and (len(stopping_offsets) == 0):
        stopping_offsets = [len(df['measures', 'speed'])-1]
    elif (len(stopping_onsets) == 0) and (len(stopping_offsets) == 0):
        return [], []
    
    if stopping_offsets[0] < stopping_onsets[0]:
        stopping_onsets = np.insert(stopping_onsets, 0, 0) # Add 0 to the beginning of running_onsets
    if stopping_onsets[-1] > stopping_offsets[-1]:
        stopping_offsets = np.append(stopping_offsets, len(df['measures', 'speed'])-1)

    # Filter out bouts that are too short
    include_bouts = np.where(stopping_offsets - stopping_onsets > 10)[0]
    if len(include_bouts) == 0:
        return [], []

    # Strange check that's necessary to avoid errors with indexing a one dimensional array
    if (len(stopping_onsets) == 1) and len(include_bouts)==1:
        return stopping_onsets, stopping_offsets
    
    stopping_onsets, stopping_offsets = stopping_onsets[include_bouts], stopping_offsets[include_bouts]
    return stopping_onsets, stopping_offsets

def get_freezing_bouts(df):
    trial_mat = df[[('Nose','x'),('Nose','y'),('Body','x'),('Body','y'),('TailBase','x'),('TailBase','y')]].to_numpy()
    df['measures', 'full_body_speed'] = kfopenfield.measures.calc_speed_bodyparts_combined(trial_mat)
    stopping_vec = np.zeros_like(df['measures', 'full_body_speed'])
    stopping_vec[df['measures', 'full_body_speed'] < 2] = 1
    stopping_vec_diff = np.diff(stopping_vec)
    stopping_onsets, stopping_offsets = np.where(stopping_vec_diff == 1)[0], np.where(stopping_vec_diff == -1)[0]
    if (len(stopping_onsets) == 0) and (len(stopping_offsets) != 0):
        stopping_onsets = [0]
    elif (len(stopping_onsets) != 0) and (len(stopping_offsets) == 0):
        stopping_offsets = [len(df['measures', 'full_body_speed'])-1]
    elif (len(stopping_onsets) == 0) and (len(stopping_offsets) == 0):
        return [], []
    
    if stopping_offsets[0] < stopping_onsets[0]:
        stopping_onsets = np.insert(stopping_onsets, 0, 0) # Add 0 to the beginning of running_onsets
    if stopping_onsets[-1] > stopping_offsets[-1]:
        stopping_offsets = np.append(stopping_offsets, len(df['measures', 'full_body_speed'])-1)

    # Filter out bouts that are too short
    include_bouts = np.where(stopping_offsets - stopping_onsets > 10)[0]
    if len(include_bouts) == 0:
        return [], []

    # Strange check that's necessary to avoid errors with indexing a one dimensional array
    if (len(stopping_onsets) == 1) and len(include_bouts)==1:
        return stopping_onsets, stopping_offsets
    
    stopping_onsets, stopping_offsets = stopping_onsets[include_bouts], stopping_offsets[include_bouts]
    return stopping_onsets, stopping_offsets

def in_shelter(x, y, shelter_boundaries, shelter_margin=[0,0]):
    if (x<shelter_boundaries[0]) and (y>shelter_boundaries[1]):
        return True
    return False

def get_shelter_boundaries(sess_date):
    if (int(sess_date)>20240626) and (int(sess_date)<20240705):
        shelter_boundaries = [190, 190]
    else:
        shelter_boundaries = [390, 598-190-10] # add 20 as buffer
    return shelter_boundaries

def get_escape_bouts(df, speed_thr=3, dur_thr=10, shelter_margin_start=[0,0], shelter_margin_end=[0,0]):
    # excludes trials where nose is in the shelter at the beginning of the running bout!!
    running_onsets, running_offsets = get_running_bouts(df, speed_thr=speed_thr, dur_thr=dur_thr)
    print(running_onsets, running_offsets)

    if len(running_offsets) == 0:
        return [], []
    
    escape_onsets, escape_offsets = [], []
    for running_onset, running_offset in zip(running_onsets, running_offsets):
        escape = True
        shelter_boundaries = df['shelter_boundaries'][['x','y']].iloc[0].to_numpy()
        print(shelter_boundaries)
        if not in_shelter(df['Body', 'x'].iloc[running_offset], df['Body', 'y'].iloc[running_offset], shelter_boundaries, shelter_margin_end):
            escape = False
        # filter out bouts that start in the shelter
        nose_x = df['Nose', 'x'].iloc[running_onset]
        nose_y = df['Nose', 'y'].iloc[running_onset]
        if in_shelter(nose_x, nose_y, shelter_boundaries, shelter_margin_start):
            escape = False
        if escape:
            escape_onsets.append(running_onset)
            escape_offsets.append(running_offset)
    return np.array(escape_onsets), np.array(escape_offsets)
    
def get_ethogram(df, offset=300, return_onoffs=False):
    etho = np.zeros(shape=len(df['measures', 'speed']))
    df['measures','speed'] = gaussian_filter1d(df['measures','speed'], sigma=3)
    arr_body = df['Body'][['x', 'y']].to_numpy()
    df['measures','corner_distance'] = kfopenfield.measures.calc_corners_distance(arr_body, corners=[(0, 572)])[0]

    escape_onsets, escape_offsets = get_escape_bouts(df.iloc[offset:])
    for escape_onset, escape_offset in zip(escape_onsets, escape_offsets):
        etho[escape_onset+offset:escape_offset+offset] = 1
        
    stopping_onsets, stopping_offsets = get_stopping_bouts(df.iloc[offset:])
    for stopping_onset, stopping_offset in zip(stopping_onsets, stopping_offsets):
        etho[stopping_onset+offset:stopping_offset+offset] = 2

    freezing_onsets, freezing_offsets = get_freezing_bouts(df.iloc[offset:])
    for freezing_onset, freezing_offset in zip(freezing_onsets, freezing_offsets):
        etho[freezing_onset+offset:freezing_offset+offset] = 3 # Will overwrite some stopping bouts!

    if return_onoffs:
        return (escape_onsets+offset, escape_offsets+offset), (stopping_onsets+offset, stopping_offsets+offset)
    return etho

def get_escape_onsets(etho_mat, start, end):
    escape_onsets = []
    for etho_trial in etho_mat:
        escape_indexes = np.where(etho_trial[start:end]==1)[0]+start
        if len(escape_indexes)>0:
            escape_onset = escape_indexes[0]
        else:
            escape_onset = 9999
        escape_onsets = np.append(escape_onsets, escape_onset)

    return escape_onsets.astype(int)

def get_stopping_onsets(etho_mat, start, end):
    escape_onsets = []
    for etho_trial in etho_mat:
        escape_indexes = np.where(etho_trial[start:end]==2)[0]+start
        if len(escape_indexes)>0:
            escape_onset = escape_indexes[0]
        else:
            escape_onset = 9999
        escape_onsets = np.append(escape_onsets, escape_onset)

    return escape_onsets.astype(int)

def get_freezing_onsets(etho_mat, start, end, min_bout_dur=10):
    freeze_onsets = []
    for etho_trial in etho_mat:
        # freeze_indexes = np.where(etho_trial[300:540]==3)[0]+300
        freeze_trial = np.zeros_like(etho_trial)
        freeze_trial[np.where(etho_trial[start:end]==3)[0]+start] = 1
        freeze_ons = np.where(np.diff(freeze_trial, prepend=0)==1)[0]-1 # -1 to account for prepend
        freeze_offs = np.where(np.diff(freeze_trial, append=0)==-1)[0]
        freeze_indexes = []
        for on, off in zip(freeze_ons, freeze_offs):
            if off-on>=min_bout_dur:
                freeze_indexes.append(on)
        if len(freeze_indexes)>0:
            freeze_onset = freeze_indexes[0]
        else:
            freeze_onset = 9999
        freeze_onsets = np.append(freeze_onsets, freeze_onset)

    return freeze_onsets.astype(int)

def get_sorted_indexes(etho_mat, start, end, sort_by='escape_onset'):
    if sort_by == 'escape_onset':
        escape_onsets = get_escape_onsets(etho_mat, start, end)
        sorted_indexes = np.argsort(escape_onsets)

    return sorted_indexes