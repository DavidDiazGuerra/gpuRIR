""" 
Generates an impulse response WAV file (IR) with optional filters.
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""
import librosa

import filters.air_absorption_calculation as aa
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
#from scipy.io import wavfile
import time
import gpuRIR
from scipy.io import wavfile
from scipy import signal

PARTITIONS = 36
PLOT_SPECTROGRAM = False
PLOT_WAVEFORM = True


def generate_RIR(src_degree):
    '''
    Generates RIRs from the gpuRIR library.

    :return: Receiver channels (mono)
    '''
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(False)

    rad = src_degree*np.pi/180

    room_sz = [16, 8, 3]  # Size of the room [m]
    nb_src = 1  # Number of sources
    pos_src = np.array([[4, 4, 1.7]])  # Positions of the sources ([m]
    nb_rcv = 1  # Number of receivers
    pos_rcv = np.array([[12, 4, 1.7]])	 # Position of the receivers [m]
    # Vectors pointing in the same direction than the receivers
    orV_src = np.matlib.repmat(
        np.array([np.cos(rad), np.sin(rad), 0]), nb_src, 1)
    orV_rcv = np.matlib.repmat(np.array([1, 0, 0]), nb_rcv, 1)
    spkr_pattern = "bidir"  # Source polar pattern
    mic_pattern = "omni"  # Receiver polar pattern
    abs_weights = [0.9]*5+[0.5]  # Absortion coefficient ratios of the walls
    T60 = 0.21	 # Time for the RIR to reach 60dB of attenuation [s]
    # Attenuation when start using the diffuse reverberation model [dB]
    att_diff = 15.0
    att_max = 60.0  # Attenuation at the end of the simulation [dB]
    fs = 44100  # Sampling frequency [Hz]
    # Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit
    bit_depth = np.int32

    beta = 6*[0.1]
    # Time to start the diffuse reverberation model [s]
    Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)
    # Time to stop the simulation [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)
    # Number of image sources in each dimension
    nb_img = gpuRIR.t2n(Tdiff, room_sz)
    RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img,
                              Tmax, fs, Tdiff=Tdiff, orV_src=orV_src, orV_rcv=orV_rcv, spkr_pattern=spkr_pattern, mic_pattern=mic_pattern)

    # return receiver channels (mono), number of receivers, sampling frequency and bit depth from RIRs.
    return RIRs[0], pos_rcv, fs, bit_depth


limit = 1


def create_waveform(x, fig, i, title=""):
    global limit
    plt.rcParams.update({'font.size': 10})
    ax = fig.add_subplot(int(np.sqrt(PARTITIONS)),
                         int(np.sqrt(PARTITIONS)), i+1)
    if i == 0:
        limit = np.abs(np.max(x))
    plt.ylim(top=limit, bottom=-limit)
    plt.title(title)
    plt.plot(x)


def create_spectrogram(x, fs, fig, i, title=""):
    #x = x[:, 0]
    plt.rcParams.update({'font.size': 10})
    f, t, Sxx = signal.spectrogram(x, fs, nfft=512)

    ax = fig.add_subplot(int(np.sqrt(PARTITIONS)),
                         int(np.sqrt(PARTITIONS)), i+1)

    plt.title(title)

    plt.pcolormesh(t, f/1000, 10*np.log10(Sxx/Sxx.max()),
                   vmin=-100, vmax=0, cmap='inferno')
    #plt.ylabel('Frequenz [kHz]')
    #plt.xlabel('Zeit [s]')
    # plt.colorbar(label='dB').ax.yaxis.set_label_position('left')


if __name__ == "__main__":
    fig = plt.figure(1)
    for i in range(0, PARTITIONS):
        degree = i * (360 / PARTITIONS)

        # Prepare sound data arrays.
        receiver_channels, pos_rcv, fs, bit_depth = generate_RIR(degree)

        # Stack array vertically
        impulseResponseArray = np.vstack(receiver_channels[0])

        # Increase Amplitude to usable levels
        impulseResponseArray = automatic_gain_increase(
            impulseResponseArray, bit_depth, 3)

        # Create stereo file (dual mono)
        impulseResponseArray = np.concatenate(
            (impulseResponseArray, impulseResponseArray), axis=1)

        # Create spectrogram
        if PLOT_WAVEFORM:
            create_waveform(receiver_channels[0], fig, i, f"{degree}°")
        if PLOT_SPECTROGRAM:
            create_spectrogram(receiver_channels[0], fs, fig, i, f"{degree}°")

    plt.show()
