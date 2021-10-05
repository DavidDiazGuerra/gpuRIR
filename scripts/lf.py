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

    #Shure SM57 dynamic microphone. Standard mic for US presidential speeches
    sm57_freq_response=np.array([ 
        # Frequency in Hz | Relative response in dB
        [50, -10],
        [100, -4],
        [200, 0],
        [400, -1],
        [700, -0.1],
        [1000, 0],
        [1500, 0.05],
        [2000, 0.1],
        [3000, 2],
        [4000, 3],
        [5000, 5],
        #[6000, 6.5],
        [7000, 3],
        [8000, 2.5],
        [9000, 4],
        [10000, 3.5],
        [fs/2, -10]
    ])

    relative_response = sm57_freq_response[:,1]
    print(relative_response)
    bands = sm57_freq_response[:,0]  # band frequencies in Hz
    print(bands)

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
