import gpuRIR.extensions.filters.air_absorption_calculation as aa

import librosa
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
import time
import gpuRIR
from scipy.io import wavfile
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

""" 
Generates an impulse response WAV file (IR) with optional filters.
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""

# Feel free to change these parameters
PARTITIONS = 81
PLOT_SPECTROGRAM = False  # Works only with 16 or less partitions!
PLOT_WAVEFORM = False  # Works only with 16 or less partitions!
PLOT_POLAR = True


# Don't change these things
POLAR_PATTERNS = np.array(
    ["omni", "homni", "card", "hypcard", "subcard", "bidir"])
MAX_VALUES = np.zeros((POLAR_PATTERNS.shape[0], PARTITIONS))
PARTITION_LIMIT = 16 # only for spectrogram and waveform plots


# Check if spectrogram / waveform plot count limit is violated
if PLOT_SPECTROGRAM or PLOT_WAVEFORM:
    assert(int(np.sqrt(PARTITIONS)) ** 2 == PARTITIONS)


def generate_RIR(src_degree, polar_pattern):
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
    pos_rcv = np.array([[10, 4, 2]])	 # Position of the receivers [m]
    # Vectors pointing in the same direction than the receivers
    orV_src = np.matlib.repmat(
        np.array([np.cos(rad), np.sin(rad), 0]), nb_src, 1)
    orV_rcv = np.matlib.repmat(np.array([1, 0, 0]), nb_rcv, 1)
    spkr_pattern = polar_pattern  # Source polar pattern
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


WAVEFORM_LIMIT = 1  # Ensure waveform values don't get overwritten


def create_waveform(x, fig, i, title=""):
    global WAVEFORM_LIMIT
    plt.rcParams.update({'font.size': 10})
    ax = fig.add_subplot(int(np.sqrt(PARTITIONS)),
                         int(np.sqrt(PARTITIONS)), i+1)
    if i == 0:
        WAVEFORM_LIMIT = np.abs(np.max(x))
    plt.ylim(top=WAVEFORM_LIMIT, bottom=-WAVEFORM_LIMIT)
    plt.title(title)
    plt.plot(x)


def create_spectrogram(x, fs, fig, i, title=""):
    plt.rcParams.update({'font.size': 10})
    f, t, Sxx = signal.spectrogram(x, fs, nfft=512)

    ax = fig.add_subplot(int(np.sqrt(PARTITIONS)),
                         int(np.sqrt(PARTITIONS)), i+1)

    plt.title(title)

    plt.pcolormesh(t, f/1000, 10*np.log10(Sxx/Sxx.max()),
                   vmin=-100, vmax=0, cmap='inferno')


def normalize_amps(amps):
    amps_normalized = np.copy(amps)
    max_value = np.max(amps_normalized)
    for i in range(0, len(amps_normalized)):
        amps_normalized[i] /= max_value
    return np.abs(amps_normalized)


COLORS = ['mediumseagreen', 'darkorange',
          'mediumpurple', 'magenta', 'limegreen', 'royalblue']


def create_polar_plot(fig, i, amps, name):
    fig.add_trace(go.Scatterpolar(
        r=normalize_amps(amps),
        theta=np.linspace(0, 360, len(amps)),
        mode='lines',
        name=name,
        line_color=COLORS[i],
        line_width=3,
        subplot=f"polar{i+1}"
    ), 1 if i <= 2 else 2, (i % 3) + 1)


def create_polar_plots(title=""):
    fig_polar = make_subplots(rows=2, cols=3, specs=[[{'type': 'polar'}]*3]*2)
    for i in range(6):
        create_polar_plot(fig_polar, i, MAX_VALUES[i], POLAR_PATTERNS[i])
    fig_polar.update_layout(
        font_size=22,
        showlegend=True,

        # TODO: Make this stuff dynamic
        polar1=dict(
            radialaxis=dict(type="log", tickangle=45),
            radialaxis_range=[-1, 0.1],
            radialaxis_tickfont_size=18
        ),
        polar2=dict(
            radialaxis=dict(type="log", tickangle=45),
            radialaxis_range=[-1, 0.1],
            radialaxis_tickfont_size=18
        ),
        polar3=dict(
            radialaxis=dict(type="log", tickangle=45),
            radialaxis_range=[-1, 0.1],
            radialaxis_tickfont_size=18
        ),
        polar4=dict(
            radialaxis=dict(type="log", tickangle=45),
            radialaxis_range=[-1, 0.1],
            radialaxis_tickfont_size=18
        ),
        polar5=dict(
            radialaxis=dict(type="log", tickangle=45),
            radialaxis_range=[-1, 0.1],
            radialaxis_tickfont_size=18
        ),
        polar6=dict(
            radialaxis=dict(type="log", tickangle=45),
            radialaxis_range=[-1, 0.1],
            radialaxis_tickfont_size=18
        ),
    )
    fig_polar.show()


if __name__ == "__main__":
    fig_wf = plt.figure(1)
    fig_sp = plt.figure(2)

    for p in range(len(POLAR_PATTERNS)):
        for i in range(0, PARTITIONS):
            degree = i * (360 / PARTITIONS)

            # Prepare sound data arrays.
            receiver_channels, pos_rcv, fs, bit_depth = generate_RIR(
                degree, POLAR_PATTERNS[p])

            # Stack array vertically
            impulseResponseArray = np.vstack(receiver_channels[0])

            # Extract biggest peak for polar pattern plotting
            MAX_VALUES[p][i] = np.max(impulseResponseArray)
            negative_peak = np.abs(np.min(impulseResponseArray))
            if MAX_VALUES[p][i] < negative_peak:
                MAX_VALUES[p][i] = negative_peak

            # Create plots
            if PLOT_WAVEFORM and PARTITIONS <= PARTITION_LIMIT:
                plt.figure(1)
                create_waveform(receiver_channels[0], fig_wf, i, f"{degree}°")
            if PLOT_SPECTROGRAM and PARTITIONS <= PARTITION_LIMIT:
                plt.figure(2)
                create_spectrogram(
                    receiver_channels[0], fs, fig_sp, i, f"{degree}°")

    if PLOT_POLAR:
        create_polar_plots()

    if (PLOT_SPECTROGRAM or PLOT_WAVEFORM) and PARTITIONS <= PARTITION_LIMIT:
        plt.show()
