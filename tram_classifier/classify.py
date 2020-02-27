import os
import sys
from math import floor

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import tram_classifier as cf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def sequence_to_sliding_windows(seq, win_length, hop_length=1):
    """
    Converts sequence seq into a sequence of sliding windows
    :param seq: 2d sequence along 0th axis
    :param win_length: length of window
    :param hop_length: size of hop between windows
    :return: ndarray shape (seq.shape[0]/num_hops, seq.shape[1], win_length
    """
    # win_height = seq.shape[0]
    seq_length = seq.shape[1]
    num_hops = floor(seq_length / hop_length)
    windows = []
    for i in range(num_hops):
        start = i * hop_length
        end = start + win_length
        if end > seq_length:
            break
        win_slice = slice(start, end)
        windows.append(seq[..., win_slice])
    return np.stack(windows, axis=0)


def _get_win_length(model):
    """
    Window length depends on the size of the input the model takes.
    :param model: trained keras model
    :return: window length (width of input to first layer of keras model)
    """
    return model.layers[0].input_shape[2]  # stft frames


def windows_to_time_plot_axis_formatter():
    """
    Simple function to convert matplotlib axis from windows to time for better readability
    :return: matplotlib FuncFormatter
    """
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda x_val, tick_pos: "{:.1f}".format(window_to_time(x_val)))


def get_window_heatmap(predictions):
    """
    Returns a matplotlib heatmap of predictions
    :param predictions: 2d numpy array with windows (time) on 0th axis
    :return: figure, axis
    """
    predictions = np.transpose(predictions)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(predictions, aspect='auto')
    ax.xaxis.set_major_formatter(windows_to_time_plot_axis_formatter())
    fig.tight_layout()
    return (fig, ax)


def predictions_heatmap(raw_preds, final_preds):
    """
    Shows a heatmap of raw_preds overlain with scatter plot of final_preds
    :param raw_preds: 2d numpy array with windows (time) on 0th axis
    :param final_preds: list of tuples of (time, class) of predictions
    :return: None
    """
    # First get window heatmap of raw (unprocessed) predictions
    fig, ax = get_window_heatmap(raw_preds)
    # Then overlay a scatter graph of the final (processed) predictions
    x = [time_to_window(pred_time) for pred_time, _ in final_preds]
    y = [pred_class for _, pred_class in final_preds]
    ax.scatter(x, y, color='red', marker='x')
    plt.show()


def window_to_time(window):
    """
    Converts a window number to a time in seconds
    :param window: window number
    :return: time in seconds
    """
    time = window * cf.features.get_hop_length() / cf.dataset.sample_rate
    return time


def time_to_window(t):
    """
    Converts a time in seconds to a window number
    :param t: time in seconds
    :return: window number
    """
    return t / window_to_time(1)


def process_predictions(predictions):
    """
    Turns raw class predictions (1 per window) from the model into a sequence of unique positive class event predictions
    :param predictions: ndarray(num_windows, num_classes)
    :return: list of tuples of (time, class) predicting the times at which a class event occurred
    """
    # Take moving average
    moving_avg_window = 6
    predictions = pd.DataFrame(predictions).rolling(window=moving_avg_window, axis=0).mean().dropna().to_numpy()

    # Get maximum probability class from each frame
    # such that max_preds[i][0] is class with maximum probability at window i and max_preds[i][1] is its probability
    max_preds = [(np.argmax(p, axis=0), np.max(p, axis=0)) for p in predictions]
    # Now we have a time series where one class event happens at each time (with a given probability)

    # Ignore adjacent duplicates and negative events
    prev_unique_pred_class = 8  # All eights will be ignored anyway
    preds = []
    for w, (pred_class, pred_prob) in enumerate(max_preds):
        if pred_class != prev_unique_pred_class:
            prev_unique_pred_class = pred_class
            if pred_class != 8:  # class 8 is a negative event (background noise)
                preds.append((w, pred_class, pred_prob))

    # Remove predictions with probability less than min_prob
    #min_prob = 0.3
    #preds = [(w, pred_class, pred_prob) for w, pred_class, pred_prob in preds if pred_prob > min_prob]

    # Convert keys from windows to times
    preds = [(window_to_time(w), pred_class, pred_prob) for w, pred_class, pred_prob in preds]

    # We can view time_preds as having "peaks" where a positives classes probability is very high
    # Finally, we prune any peaks which are very near another larger peak
    result = []
    shoulder_width_seconds = 8
    for i, (i_time, i_class, i_prob) in enumerate(preds):
        # Inner loop tries to find a bigger peak nearby this one
        # Looks at most max_look_around peaks around (mostly just to avoid using a while true loop)
        max_look_around = 100  # something much larger then the number of peaks we expect in shoulder_width_seconds/2
        include = True
        d_start = max(0, i - max_look_around)
        d_end = min(len(preds), i + max_look_around)
        for j in range(d_start, d_end):
            # by definition range(d_start, d_min) ensures 0 <= j < len(time_preds)
            j_time, j_class, j_prob = preds[j]
            # Break if peak j is more than shoulder_width_seconds/2 ahead of peak i
            diff = j_time - i_time
            if diff > shoulder_width_seconds / 2:
                break
            # Skip forward if peak j is more than shoulder_width_seconds/2 behind peak i
            if abs(diff) > shoulder_width_seconds / 2:
                continue
            # If we get here, do not include peak i if peak j is bigger than peak i
            if i_prob < j_prob:
                include = False
                break
        if include:
            result.append((i_time, i_class))

    return result


def application():
    """
    Given the path to a wave file, this application will print (in csv) details of tram events in the recording
    :return: None
    """
    # Load model
    model_file = os.path.join(cf.base_path, 'model-final.h5')
    model = tf.keras.models.load_model(model_file)

    # Load input
    audio_file_path = sys.argv[1]
    audio, _ = librosa.load(audio_file_path, sr=cf.dataset.sample_rate, res_type='kaiser_fast')
    spectrum_object = cf.features.time_series_to_spectrum(audio)
    windows = sequence_to_sliding_windows(spectrum_object, _get_win_length(model))
    data = windows.reshape(*windows.shape, 1)  # Convert to 'image' with 1 channel for cnn input

    # Make predictions
    raw_preds = model.predict(data)
    final_preds = process_predictions(raw_preds)

    # Print CSV header
    positive_classes_slice = slice(0, -1)
    print("seconds_offset," + ",".join(
        ["{}_{}".format(state, model) for _, state, model in cf.dataset.classes_list[positive_classes_slice]]))

    # Print CSV rows
    for t, c in final_preds:
        one_hot = ["1" if i == c else "0" for i in cf.dataset.classes]
        print("{:.1f},{}".format(t, ",".join(one_hot)))

    # Uncomment for predictions heatmap output (for debugging/development)
    # predictions_heatmap(raw_preds, final_preds)


if __name__ == '__main__':
    application()
