""" 
Generates an impulse response WAV file (IR).
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""
import air_absorption as aa
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import wavfile
from scipy.signal import istft, stft
import time
import gpuRIR
#from create_spectrogram import create_spectrogram

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

room_sz = [6,6,5]  # Size of the room [m]
nb_src = 1  # Number of sources
pos_src = np.array([[1,1,2.5]]) # Positions of the sources ([m]
nb_rcv = 1 # Number of receivers
pos_rcv = np.array([[5.9,5.9,4.9]])	 # Position of the receivers [m]
orV_rcv = np.matlib.repmat(np.array([0,1,0]), nb_rcv, 1) # Vectors pointing in the same direction than the receivers
mic_pattern = "card" # Receiver polar pattern
abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls
T60 = 1.0	 # Time for the RIR to reach 60dB of attenuation [s]
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]
fs=44100 # Sampling frequency [Hz]
bit_depth=np.int32 # Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit

beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)

receiver_channels = RIRs[0] # Extract receiver channels (mono) from RIRs.

'''
Parameters relating to air absorption
'''
enable_air_absorption=True # Determines if air absorption is applied.
divisions=1 # How many partitions the frequency spectrum gets divided into. Roughly correlates to quality / accuracy.
min_frequency=20.0 # [Hz] Lower frequency boundary.
max_frequency=20000.0 # [Hz] Upper frequency boundary.

frequency_range=max_frequency - min_frequency

'''
Calculates how much distance the sound has travelled. [m]
'''
def distance_travelled(sample_number, sampling_frequency, c):
    seconds_passed=sample_number*(sampling_frequency**(-1))
    return (seconds_passed*c) # [m]


def STFT_air_absorption(RIR, T, hr, ps, fs, nFFT=256, noverlap=int(256*0.75), window='hanning'):

  # Calculate STFT of RIR (like spectrogram)
    f, t, RIR_TF = stft(RIR, fs, window, nperseg=nFFT, noverlap=noverlap, nfft=nFFT, boundary='even', padded=True, return_onesided=False)

  # Get air absorption coeffs for given configuration, coeffs are in dB/meter
    alphas_t0, alpha_iso, c, c_iso = aa.air_absorption(f, T, hr, ps)

  # Get air absorption coeffs over distance/time for each band
    alphas = np.zeros((len(f), len(t)))
    for ii in range(len(t)):
        alphas[:, ii] = -alphas_t0*t[ii]*c

  # Set lower bound for attenuation
    alphas[alphas < -100] = -100

  # Get linear absorption coeffs and apply them to the STFT of the RIR
    RIR_TF_processed = RIR_TF*np.power(10, (alphas/20))

  # Transform processed STFT of RIR back to time domain
    _, RIR_processed = istft(RIR_TF_processed, fs, window, nperseg=nFFT, noverlap=noverlap, nfft=nFFT)
    RIR_processed = RIR_processed[0:len(RIR)]
    return RIR_processed



'''
Increases amplitude (loudness) to defined ceiling.
'''
def automatic_gain_increase(data, bit_depth, ceiling):
    peak = np.max(data)
    negative_peak = np.abs(np.min(data))

    # Check if the negative or positive peak is of a higher magnitude
    if peak < negative_peak:
        peak = negative_peak

    max_gain = np.iinfo(bit_depth).max*10**(-ceiling/10)
    factor = max_gain/peak

    return data*factor


for i in range(0, len(pos_rcv)):
    # Prepare sound data arrays.
    source_signal=np.copy(receiver_channels[i])

    # Apply STFT based air absorption
    source_signal=STFT_air_absorption(source_signal, 25, 50, 1, fs)

    print(source_signal)

    # Stack array vertically
    impulseResponseArray = np.vstack(source_signal)

    # Increase Amplitude to usable levels 
    # ye olde timey way to increase ampliude: impulseResponseArray = impulseResponseArray * np.iinfo(bit_depth).max
    impulseResponseArray=automatic_gain_increase(impulseResponseArray, bit_depth, 3)

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate((impulseResponseArray, impulseResponseArray), axis=1)

    print(impulseResponseArray)

    # Write impulse response file
    filename = f'impulse_response_rcv_atten_{i}_{time.time()}.wav'
    wavfile.write(filename, fs, impulseResponseArray.astype(bit_depth))
    #create_spectrogram(filename)

    # Visualize waveform of IR
    plt.plot(impulseResponseArray)

t = np.arange(int(ceil(Tmax * fs))) / fs
plt.show()


