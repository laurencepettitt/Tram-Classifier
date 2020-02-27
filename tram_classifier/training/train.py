import datetime
import os
import pathlib

import numpy as np
import pandas
import sklearn
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical

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


def init_model(input_shape):
    """
    Returns the built and compiled Keras model
    :param input_shape: tuple(num_examples, num_frequency_bins, num_time_frames, num_channels)
    :return: keras model
    """
    num_classes = len(cf.dataset.classes)

    # Construct model
    model = Sequential(name='spectrum_cnn_3')
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 1), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(learning_rate=1.0), metrics=['accuracy'])

    return model


def examples_normalized_class_weights(classes):
    return pandas.Series(classes).value_counts(normalize=True).apply(lambda f: 1/f).to_dict()


def train_model(model, input_train, input_test, target_train, target_test):
    """
    Fits model to data
    """
    num_classes = len(cf.dataset.classes)

    # Convert class labels to one hot vectors
    target_train_hot = to_categorical(target_train, num_classes=num_classes)
    target_test_hot = to_categorical(target_test, num_classes=num_classes)

    datetime_now = '{date:%Y%m%d-%H%M}'.format(date=datetime.datetime.now())

    # Save logs every epoch for tensorboard visualisation
    log_dir = os.path.join(cf.training_logs_dir, datetime_now)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Save model every epoch
    model_checkpoint_dir = os.path.join(cf.saved_models_dir, datetime_now)
    pathlib.Path(model_checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model_checkpoint_file = os.path.join(model_checkpoint_dir, "model.{epoch:02d}-{val_loss:.2f}.h5")
    model_checkpoint = ModelCheckpoint(model_checkpoint_file)

    # Apply custom class weights
    class_weights = class_weight.compute_class_weight('balanced', cf.dataset.classes, target_train)

    # Fit data to model
    model.fit(input_train, target_train_hot, batch_size=32, epochs=20, verbose=1, class_weight=class_weights,
              validation_data=(input_test, target_test_hot), callbacks=[tensorboard_callback, model_checkpoint])

    print("Saved models to: " + str(model_checkpoint_dir))
    return model


def evaluate(model, input_test, target_test):
    """
    Evaluates model and prints report with precision, recall and f1-score for each class as well as overall averages.
    """
    # Print classification report
    pred_test = np.argmax(model.predict(input_test, batch_size=4), axis=1)
    print(classification_report(target_test, pred_test))


def main():
    input_train, input_test, target_train, target_test = get_inputs_targets()
    model = init_model(input_shape=input_train.shape[1:])
    print(model.summary())
    model = train_model(model, input_train, input_test, target_train, target_test)
    evaluate(model, input_test, target_test)


if __name__ == '__main__':
    main()
