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
Visualizes polar patterns of the source by rotating source by 360, repeatedly calling gpuRIR.
Warning: Takes up to 2-3 minutes depending on your hardware!
"""

# Feel free to change these parameters
# Resolution of polar plot (amount to divide 360 degrees into)
PARTITIONS = 360

# gpuRIR parameters
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

# Don't change these things
POLAR_PATTERNS = np.array(
    ["omni", "homni", "card", "hypcard", "subcard", "bidir"])
MAX_VALUES = np.zeros((POLAR_PATTERNS.shape[0], PARTITIONS))


def normalize_amps(amps):
    '''
    Normalizing amplitude. Change all amplitude such that loudest sample has the amplitude of 1.
    :param amps Array of samples.
    '''
    amps_normalized = np.copy(amps)
    max_value = np.max(amps_normalized)
    for i in range(0, len(amps_normalized)):
        amps_normalized[i] /= max_value
    return np.abs(amps_normalized)


COLORS = ['mediumseagreen', 'darkorange',
          'mediumpurple', 'magenta', 'limegreen', 'royalblue']


def create_polar_plot(fig, i, amps, name):
    '''
    Creates single polar plot.

    :param fig Plot figure.
    :param i Index of plot
    :param name Name of polar pattern
    '''
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
    '''
    Creates compilation of polar plots using plotly.

    :param title Title of plot.
    '''
    fig_polar = make_subplots(rows=2, cols=3, specs=[[{'type': 'polar'}]*3]*2)
    for i in range(6):
        create_polar_plot(fig_polar, i, MAX_VALUES[i], POLAR_PATTERNS[i])
    fig_polar.update_layout(
        font_size=22,
        showlegend=True,

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
            rad = degree*np.pi/180

            # RIR parameters
            room_sz=[16, 8, 3]  # Size of the room [m]
            pos_src=np.array([[4, 4, 1.7]])  # Positions of the sources [m]
            pos_rcv=np.array([[10, 4, 2]])  # Positions of the receivers [m]
            # Steering vector of source(s)
            orV_src=np.matlib.repmat(np.array([np.cos(rad), np.sin(rad), 0]), 1, 1)
            # Steering vector of receiver(s)
            orV_rcv=np.matlib.repmat(np.array([1, 0, 0]), 1, 1)
            spkr_pattern=POLAR_PATTERNS[p]  # Source polar pattern
            mic_pattern="omni"  # Receiver polar pattern
            T60=0.21  # Time for the RIR to reach 60dB of attenuation [s]
            # Attenuation when start using the diffuse reverberation model [dB]
            att_diff=15.0
            att_max=60.0  # Attenuation at the end of the simulation [dB]
            fs=44100  # Sampling frequency [Hz]
            # Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit
            bit_depth=np.int32
            beta=6*[0.1] # Reflection coefficients
            Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
            Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
            nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension

            # Prepare sound data arrays.
            receiver_channels = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern, orV_src=orV_src, spkr_pattern=spkr_pattern)

            # Stack array vertically
            impulseResponseArray = np.vstack(receiver_channels[0])

            # Extract biggest peak for polar pattern plotting
            MAX_VALUES[p][i] = np.max(impulseResponseArray)
            negative_peak = np.abs(np.min(impulseResponseArray))
            if MAX_VALUES[p][i] < negative_peak:
                MAX_VALUES[p][i] = negative_peak

    create_polar_plots()
