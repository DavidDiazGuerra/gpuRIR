import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


def create_spectrogram_from_file(inner_file_path, title="", channel_count=1):
    fs, source = wavfile.read(inner_file_path)
    for channel in range(0, channel_count):
        create_spectrogram(source, fs, channel, title)


def create_spectrogram_from_data(source, fs, title=""):
    channel_count = source.shape[1]
    for channel in range(0, channel_count):
        create_spectrogram(source, fs, channel, title)


def create_spectrogram(source, fs, channel, title=""):
    channel_name = 'left' if channel == 0 else 'right'
    x = source[:, channel]
    plt.rcParams.update({'font.size': 18})
    f, t, Sxx = signal.spectrogram(x, fs, nfft=512)
    plt.pcolormesh(t, f/1000, 10*np.log10(Sxx/Sxx.max()),
                   vmin=-80, vmax=0, cmap='inferno')
    plt.ylabel('Frequenz [kHz]')
    plt.xlabel('Zeit [s]')
    plt.title(f"{title} Channel: {channel_name}")
    plt.colorbar(label='dB').ax.yaxis.set_label_position('left')
    plt.show()


if len(sys.argv) > 1:
    file_path = sys.argv[1]
    if len(sys.argv) > 2:
        create_spectrogram_from_file(file_path, sys.argv[2])
    else:
        create_spectrogram_from_file(file_path, file_path)
