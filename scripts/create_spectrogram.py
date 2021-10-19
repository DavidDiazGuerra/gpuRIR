
#import librosa
#import librosa.display
import sys
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import wavfile



def create_spectrogram(inner_file_path, title=""):
    samplingFrequency, signalData = wavfile.read(inner_file_path)
    signalData=signalData[:,0]
    # normalize data
    signalData = signalData * (1 / np.max(signalData))

    dbCap = 150
    vmin = 20*np.log10(np.max(signalData)) - dbCap

    #plot.subplot(211)
    plot.rc('font', size=12)
    plot.title(title)
    plot.plot(signalData)
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')
    plot.show()

    #plot.subplot(212)
    plot.title(title)
    _, _, _, cax = plot.specgram(signalData,Fs=samplingFrequency, vmin=vmin, cmap='inferno')
    plot.xlabel('Zeit [s]')
    plot.ylabel('Frequenz [Hz]')
    plot.colorbar(cax, label='dB').ax.yaxis.set_label_position('left')
    plot.show()

if len(sys.argv) > 1:
    file_path = sys.argv[1]
    if len(sys.argv) > 2:
        create_spectrogram(file_path, sys.argv[2])
    else:
        create_spectrogram(file_path, file_path)
