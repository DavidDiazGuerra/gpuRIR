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
from scipy.signal import butter, lfilter, buttord
import time
import gpuRIR

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
enable_air_absorption=False # Determines if air absorption is applied.
divisions=10 # How many partitions the frequency spectrum gets divided into. Roughly correlates to quality / accuracy.
min_frequency=20.0 # [Hz] Lower frequency boundary.
max_frequency=20000.0 # [Hz] Upper frequency boundary.

frequency_range=max_frequency - min_frequency

'''
Calculates how much distance the sound has travelled. [m]
'''
def distance_travelled(sample_number, sampling_frequency, c):
    seconds_passed=sample_number*(sampling_frequency**(-1))
    return (seconds_passed*c) # [m]


'''
Returns a butterworth bandpass filter.
'''
def create_bandpass_filter(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = (lowcut / nyq)
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


'''
Applies a butterworth bandpass filter.
'''
def apply_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = create_bandpass_filter(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

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
    combined_signals=np.zeros(len(source_signal))

    # Divide frequency range into defined frequency bands
    for j in range(1, divisions + 1):
        # Upper ceiling of each band
        band_max = ((frequency_range / divisions) * j)

        # Lower ceiling of each band and handling of edge case
        if j == 1:
            band_min = min_frequency
        else:
            band_min = ((frequency_range / divisions) * (j - 1))

        # Calculating mean frequency of band which determines the attenuation.
        band_mean = (band_max+band_min)/2
        print(f"Band {j} frequencies: min: {band_min} max: {band_max} mean:{band_mean}")

        # Prepare + apply bandpass filter
        filtered_signal = apply_bandpass_filter(source_signal, band_min, band_max, fs, 3)

        # Apply attenuation
        if enable_air_absorption:
            for k in range(0, len(filtered_signal)):
                alpha, alpha_iso, c, c_iso = aa.air_absorption(band_mean)
                distance = distance_travelled(k, fs, c)
                attenuation = distance*alpha  # [dB]

                filtered_signal[k] *= 10**(-attenuation / 10)

        # Summing the different bands together
        for k in range(0, len(combined_signals)):
            combined_signals[k] += filtered_signal[k]

    # Stack array vertically
    impulseResponseArray = np.vstack(combined_signals)

    # Increase Amplitude to usable levels 
    # ye olde timey way to increase ampliude: impulseResponseArray = impulseResponseArray * np.iinfo(bit_depth).max
    impulseResponseArray=automatic_gain_increase(impulseResponseArray, bit_depth, 3)

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate((impulseResponseArray, impulseResponseArray), axis=1)

    #impulseResponseArray=impulseResponseArray[1]
    print(impulseResponseArray)

    # Write impulse response file
    wavfile.write(f'impulse_response_rcv_atten_{i}_{time.time()}.wav', fs, impulseResponseArray.astype(bit_depth))

    # Visualize waveform of IR
    plt.plot(impulseResponseArray)

t = np.arange(int(ceil(Tmax * fs))) / fs
plt.show()
