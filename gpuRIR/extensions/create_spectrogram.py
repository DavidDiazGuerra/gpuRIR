import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


def create_spectrogram_from_file(inner_file_path, title="", channel_count=1):
    """ Shows a spectrogram from a provided file.

    Parameters
    ----------
    inner_file_path : str
        Path to the sound file that we want to visualize (WAV format)
    title : str, optional
        Title of spectrogram
    channel_count : int, optional
        Number of channels
    """
    fs, source = wavfile.read(inner_file_path)
    for channel in range(0, channel_count):
        channel_name = 'left' if channel == 0 else 'right'
        x = source[:, channel]
        plt.rcParams.update({'font.size': 18})
        f, t, Sxx = signal.spectrogram(x, fs, nfft=512)
        plt.pcolormesh(t, f/1000, 10*np.log10(Sxx/Sxx.max()),
                    vmin=-80, vmax=0, cmap='inferno')
        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Time [s]')
        plt.title(f"{title} Channel: {channel_name}")
        plt.colorbar(label='dB').ax.yaxis.set_label_position('left')
        plt.show()


def create_spectrogram_from_data(source, fs, channel_name="", title=""):
    """ Shows a spectrogram from a provided file.

    Parameters
    ----------
    source : ndarray
        Sound data that we want to visualize (stereo or mono)
    fs : int
        Sampling rate (Hertz) of sound data.
    title : str, optional
        Title of spectrogram
    channel_count : int, optional
        Number of channels
    """
    plt.rcParams.update({'font.size': 18})
    f, t, Sxx = signal.spectrogram(source, fs, nfft=512)
    plt.pcolormesh(t, f/1000, 10*np.log10(Sxx/Sxx.max()),
                    vmin=-80, vmax=0, cmap='inferno')
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [s]')
    plt.title(f"{title} {channel_name}")
    plt.colorbar(label='dB').ax.yaxis.set_label_position('left')
    plt.show()


""" Can be run from a terminal. First argument is sound file path, second argument is diagram title.
"""
if len(sys.argv) > 1:
    file_path = sys.argv[1]
    if len(sys.argv) > 2:
        create_spectrogram_from_file(file_path, sys.argv[2])
    else:
        create_spectrogram_from_file(file_path, file_path)
