import numpy as np
import os
import pickle


def get_cluster_frs(cluster_keys, stim_types, dataset_name, means=False, subsession_id='01'):
    # Check whether data is saved locally
    data_folder = rf'E:\results\CoordinatedCellTypes\data'
    data_path = os.path.join(data_folder, f'{dataset_name}.pkl')

    # If data is saved locally, load it
    if os.path.exists(data_path):
        print("Found data locally, loading...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # If data is not saved locally, extract from datajoint and save locally
    print("Data not found locally, extracting from datajoint...")
    return None
