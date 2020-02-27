import os

import numpy as np
import sklearn
import tensorflow as tf

import tram_classifier as cf

# Cache paths
inputs_targets_cache_file_path = os.path.join(cf.cache_dir_name, 'inputs_targets_cache.pkl')


def get_sequence_length_seconds():
    """
    This will be the length of the model's input in seconds
    :return: length of model's input in seconds
    """
    return 5


def manipulate(time_series, sampling_rate, shift_max, shift_direction):
    """
    Randomly shifts time_series by at most shift_max in shift_direction
    :param time_series: time_series to shift, must have time along 0th axis
    :param sampling_rate: sample rate of time_series
    :param shift_max: maximum time in seconds to shift
    :param shift_direction: one of ('left', 'right')
    :return: shifted version of time_series
    """
    assert shift_direction in ['left', 'right']
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift

    shifted_series = np.roll(time_series, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        shifted_series[:shift] = 0
    else:
        shifted_series[shift:] = 0
    return shifted_series


def augment_shift(time_series_objects, classes):
    """
    Shifts each time series object in time_series_objects randomly left and right to create two more time series objects
    :param time_series_objects: time ax
    :param classes:
    :return:
    """
    max_shift = 0.5
    time_series_objects_left = [
        manipulate(time_series_object, cf.dataset.sample_rate, shift_max=max_shift, shift_direction='left') for
        time_series_object in time_series_objects]
    time_series_objects_right = [
        manipulate(time_series_object, cf.dataset.sample_rate, shift_max=max_shift, shift_direction='right') for
        time_series_object in time_series_objects]
    time_series_objects = np.concatenate([time_series_objects, time_series_objects_left, time_series_objects_right])
    classes = np.concatenate([classes, classes, classes])
    return time_series_objects, classes


def _inputs_targets_loader():
    """
    Loads inputs and targets for training in model
    :return: tuple(input_train, input_test, target_train, target_test)
    """
    time_series_objects, classes = cf.dataset.load_examples()
    print("Number of samples in dataset: " + str(len(time_series_objects)))

    time_series_objects, classes = augment_shift(time_series_objects, classes)
    print("Number of samples in dataset after shift augmentation: " + str(len(time_series_objects)))

    # time_series_objects, classes = _example_classes(time_series_objects, classes, n=1)
    print("Number of samples actually using: " + str(len(time_series_objects)))

    # Pad sequences to max_sequence_length  # Todo - pad sequences _after_ fourier-transfrom, only truncate before
    max_sequence_length = cf.dataset.sample_rate * get_sequence_length_seconds()
    time_series_objects = tf.keras.preprocessing.sequence.pad_sequences(time_series_objects, maxlen=max_sequence_length,
                                                          dtype=np.float32)

    print("Transforming time series to spectrums...")
    spectrum_objects = (cf.features.time_series_to_spectrum(time_series) for time_series in time_series_objects)

    # Convert to numpy array
    spectrum_objects = np.array(list(spectrum_objects), dtype=np.float32)
    print("Shape of time-frequency spectrum objects list: " + str(spectrum_objects.shape))

    # Split into train/test set
    input_train, input_test, target_train, target_test = sklearn.model_selection.train_test_split(spectrum_objects,
                                                                                                  classes,
                                                                                                  test_size=0.2,
                                                                                                  random_state=43)

    # Adapt input shape for input to model
    input_shape = (*input_train.shape[1:], 1)
    print("CNN input shape: " + str(input_shape))
    input_train = input_train.reshape(input_train.shape[0], *input_shape)  # convert to "images" with one channel
    input_test = input_test.reshape(input_test.shape[0], *input_shape)

    return input_train, input_test, target_train, target_test


def get_inputs_targets():
    """
    Returns inputs and targets for training in model
    :return: tuple(input_train, input_test, target_train, target_test)
    """
    return cf.load_cached(_inputs_targets_loader, inputs_targets_cache_file_path)
