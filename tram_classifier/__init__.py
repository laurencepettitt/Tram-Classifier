import os

# Paths
import pickle
from pathlib import Path

import tram_classifier.dataset
import tram_classifier.features

base_path = os.path.dirname(__file__)
# Data paths
training_dir = os.path.join(base_path, 'training')
saved_models_dir = os.path.join(training_dir, 'saved_models')
training_logs_dir = os.path.join(training_dir, 'logs')
# Cache paths
cache_dir_name = os.path.join(base_path, 'cache')


def load_cached(data_loader, cache_file_path):
    """
    Helper function to load data from cache_file_path, or from data_loader (if cache not available) and save it
    :param data_loader: loads data for cache
    :param cache_file_path: string path to cache file
    :return: data
    """
    Path(os.path.dirname(cache_file_path)).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(cache_file_path):
        return pickle.load(open(cache_file_path, 'rb'))
    else:
        data = data_loader()
        pickle.dump(data, file=open(cache_file_path, 'wb'))
        return data
