import ntpath
import os
import pickle
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pandas as pd

# Definitions & Parameters
num_classes = 9
classes = range(num_classes)
sample_rate = 22050

# Paths
base_path = os.path.dirname(__file__)
# Data paths
examples_path = os.path.join(base_path, 'examples')
test_files_path = os.path.join(base_path, 'test_files')
# Cache paths
cache_dir_name = os.path.join(base_path, 'cache')
examples_cache_file_path = os.path.join(cache_dir_name, 'examples_cache.pkl')
test_files_cache_path = os.path.join(cache_dir_name, 'test_files.pkl')

# Class string name to integer label encoding
states = ['accelerating', 'braking']
models = ['1_New', '2_CKD_Long', '3_CKD_Short', '4_Old']
classes_list = [('positive', state, model) for state in states for model in models] + [('negative',)]


def _get_example_class_from_path(example_path):
    """
    Given an examples path (in the data set), returns the examples class
    :param example_path:
    :return: class of example
    """
    path_in_dataset = os.path.relpath(example_path, examples_path)
    path_parts = Path(os.path.dirname(path_in_dataset)).parts
    return classes_list.index(tuple(path_parts))


def _recursive_scandir(base_dir):
    """
    Generates Entries for all wav files in a directory (and subdidrectories)
    :param base_dir:
    :return:
    """
    for entry in os.scandir(base_dir):
        if entry.is_file():
            if entry.name.endswith('.wav'):
                yield entry
        else:
            yield from _recursive_scandir(entry.path)

    return


def load_audio_example(file_path):
    audio, _ = librosa.load(file_path, sr=sample_rate, res_type='kaiser_fast')
    return audio


def _clean_name(file_name):
    basename = ntpath.basename(file_name)
    start = len('tram-')
    end = -len('.mp4.wav')
    return basename[start:end]


def _get_name_parts(file_name):
    clean_name = _clean_name(file_name)
    parts = clean_name.split(sep='_')
    timestamp = str(parts[0])
    other = '_'.join((str(parts[1]), str(parts[2])))
    return timestamp, other


def load_cached(data_loader, cache_file_path):
    Path(os.path.dirname(cache_file_path)).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(cache_file_path):
        return pickle.load(open(cache_file_path, 'rb'))
    else:
        data = data_loader()
        pickle.dump(data, file=open(cache_file_path, 'wb'))
        return data


def _examples_data_loader() -> pd.DataFrame:
    """
    Loads Examples dataset in pandas.DataFrame
    :return: dataset of examples and attributed information
    """
    _ensure_dataset_files()
    entries = _recursive_scandir(examples_path)
    rows = []
    for entry in entries:
        time_series = load_audio_example(entry.path)
        class_label = _get_example_class_from_path(entry.path)
        timestamp, other = _get_name_parts(entry.name)
        relative_path = os.path.relpath(examples_path, entry.path)
        file_name = timestamp + '_' + other
        row = [time_series, class_label, timestamp, file_name, relative_path]
        rows.append(row)

    return pd.DataFrame(data=rows,
                        columns=['time_series', 'class', 'timestamp', 'file_name', 'relative_path'])


def load_examples_full_dataset():
    return load_cached(_examples_data_loader, examples_cache_file_path)


def load_examples() -> Tuple[np.ndarray, np.ndarray]:
    df = load_examples_full_dataset()
    return df['time_series'].reset_index(drop=True).to_numpy(), df['class'].reset_index(drop=True).to_numpy()


def _test_files_data_loader():
    _ensure_dataset_files()
    entries = _recursive_scandir(examples_path)
    rows = []
    for entry in entries:
        time_series = load_audio_example(entry.path)
        relative_path = os.path.relpath(examples_path, entry.path)
        file_name = entry.name
        row = [time_series, file_name, relative_path]
        rows.append(row)

    return pd.DataFrame(data=rows,
                        columns=['time_series', 'file_name', 'relative_path'])


def load_test_files_full_dataset():
    return load_cached(_test_files_data_loader, test_files_cache_path)


def load_test_files():
    return load_test_files_full_dataset()['time_series'].reset_index(drop=True).to_numpy()


def _ensure_dataset_files():
    if os.path.isdir(examples_path) and os.path.isdir(test_files_path):
        return
    _download_dataset_files()


def _download_dataset_files():
    import dload
    dataset_files_url = 'https://cunicz-my.sharepoint.com/:u:/g/personal/53500436_cuni_cz/EYh2GS4MFKVGoNTn5_Wm840BaYe6ZQ5ihouRjm0kAVed_A?download=1'
    dataset_files_dir = base_path
    print("Downloading dataset files (~1GB), could take a while..")
    dload.save_unzip(dataset_files_url, dataset_files_dir)
