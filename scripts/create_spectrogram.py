
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

waveform, sample_rate = librosa.load('impulse_response_rcv_atten_2_1631652651.3554087.wav')

def to_decibles(signal):
    # Perform short time Fourier Transformation of signal and take absolute value of results
    stft = np.abs(librosa.stft(signal))
    # Convert to dB
    D = librosa.amplitude_to_db(stft, ref = np.max) # Set reference value to the maximum value of stft.
    return D # Return converted audio signal

# Function to plot the converted audio signal
def plot_spec(D, sr, instrument):
    fig, ax = plt.subplots(figsize = (30,10))
    spec = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title = f'Sample rate: {sample_rate}')
    fig.colorbar(spec)

plot_spec(to_decibles(waveform), sample_rate, 'Guitar')
plt.show()