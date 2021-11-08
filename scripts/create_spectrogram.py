import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


def create_spectrogram(inner_file_path, title=""):
    fs, x = wavfile.read(inner_file_path)
    x = x[:, 0]
    plt.rcParams.update({'font.size': 18})

    f, t, Sxx = signal.spectrogram(x, fs, nfft=512)
    plt.pcolormesh(t, f/1000, 10*np.log10(Sxx/Sxx.max()),
                   vmin=-80, vmax=0, cmap='inferno')
    plt.ylabel('Frequenz [kHz]')
    plt.xlabel('Zeit [s]')
    plt.colorbar(label='dB').ax.yaxis.set_label_position('left')
    plt.show()

    
if len(sys.argv) > 1:
    file_path = sys.argv[1]
    if len(sys.argv) > 2:
        create_spectrogram(file_path, sys.argv[2])
    else:
        create_spectrogram(file_path, file_path)
