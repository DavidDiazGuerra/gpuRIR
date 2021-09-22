
import librosa
import librosa.display
import sys
import numpy as np
import matplotlib.pyplot as plt

file_path = 'impulse_response_rcv_atten_0_1632315495.4742222.wav'
sample_rate = 0

if len(sys.argv) > 1:
    file_path = sys.argv[1]

def create_spectrogram(inner_file_path):
    waveform, sample_rate = librosa.load(inner_file_path)
    plot_spec(to_decibles(waveform), sample_rate, 'Guitar')
    plt.show()

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

print(f"Opening {file_path}...")
create_spectrogram(file_path)