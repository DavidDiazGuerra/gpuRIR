import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import soundfile

'''
Converts relative response in dB to ratio (logarithmic scale), needed for signal.firls
Makes it easier to manually enter existing microphone or speaker frequency responses
'''


def rel_db_response_to_ratio(gain_dB):
    return 10**(gain_dB/10)


if __name__ == '__main__':
    y, fs = soundfile.read('clean_woman.wav')

    # Simulate mic response using a band-pass FIR filter
    # in real worls b_siumulated_mic would be measured with real microphones and loudspeakers
    # choose which bands should be filtered
    relative_response = [
        -10, -4, 0, -1, -0.1, 0, 0.05, 0.1, 2, 5, 6.5, 3, 2.5, 4, 3.5, -10]
    bands = [
        50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000, 6000, 7000, 8000, 9000, 10000, fs/2]  # band frequencies in Hz

    for i in range(len(relative_response)):
        relative_response[i] = rel_db_response_to_ratio(relative_response[i])

    print(f"fs: {fs}")

    print(f"len resp: {len(relative_response)}")
    print(f"len band: {len(bands)}")


    b_siumulated_mic = scipy.signal.firls(
        51, bands, relative_response, fs=fs)  # design filter

    # Plot frequency resonse of filter
    w, response = scipy.signal.freqz(b_siumulated_mic)
    freq = w/np.pi*fs/2
    fig, ax1 = plt.subplots()
    ax1.set_title('Simulated Mic Freq Response')
    ax1.plot(freq, 20 * np.log10(abs(response)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [Hz]')
    plt.show()

    # Apply filter to audio signal
    y_filtered = scipy.signal.lfilter(b_siumulated_mic, 1, y)
    soundfile.write('filtered.wav', y_filtered, fs)
