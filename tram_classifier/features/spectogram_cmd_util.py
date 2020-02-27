import sys

import librosa.display
import matplotlib.pyplot as plt

import tram_classifier as cf

if __name__ == '__main__':
    audio_file_path = sys.argv[1]
    post_fix = sys.argv[2] if len(sys.argv) >= 3 else ""

    spectogram_file_path = audio_file_path \
                           + "_" + str(cf.features.get_win_length()) \
                           + '-' + str(cf.features.get_hop_length()) \
                           + '_' + str(cf.features.get_num_mels()) \
                           + post_fix + ".png"
    spectogram = cf.features.time_series_to_spectrum(cf.dataset.load_audio_example(audio_file_path))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectogram, x_axis='time', y_axis='mel', sr=cf.dataset.sample_rate)
    plt.colorbar(format='%+2.0f dB')
    class_id = cf.dataset._get_example_class_from_path(audio_file_path)
    class_attrs = cf.dataset.classes_list[class_id]
    plt.title('Mel-frequency spectrogram - ' + " ".join(class_attrs))
    plt.tight_layout()
    plt.savefig(spectogram_file_path)
    print(spectogram_file_path)
