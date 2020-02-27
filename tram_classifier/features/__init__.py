import librosa
import numpy as np


def get_win_length():
    """
    Number of samples in stft window (and frame)
    """
    return 2**13


def get_hop_length():
    """
    Number of samples between successive stft windows (frames)
    """
    return 2**12


def get_num_mels():
    return 200


def time_series_to_spectrum(time_series):
    """
    Converts a tram_dataset time_series into a particular time-frequency spectrum
    :param time_series: time_series sampled at tram_dataset.sample_rate
    :return: 2d numpy array spectrum, time on 0th axis
    """
    # Apply short-term Fourier transform on each time series object
    spectrum_object = librosa.core.stft(time_series, n_fft=get_win_length(), hop_length=get_hop_length())

    # Get only amplitudes of short-term fourier transforms
    spectrum_object = np.abs(spectrum_object)
    # Convert amplitudes to powers
    spectrum_object = spectrum_object ** 2
    # Convert powers to decibel
    spectrum_object = librosa.power_to_db(spectrum_object, ref=np.max)
    # Convert frequencies to Mel-scale
    spectrum_object = librosa.feature.melspectrogram(S=spectrum_object, n_mels=get_num_mels())
    return spectrum_object

