import numpy as np

'''
Generates white noise
'''
def generate_white_noise(Tdiff, Tmax, fs, n_src=1, n_rcv=1):
    # Check if parameters exceed 1 for n_src and n_rcv
    assert(n_src == 1 and n_rcv == 1), "multiple sources & receivers not yet implemented"

    nSamplesISM = np.ceil(Tdiff*fs)
    nSamplesISM += nSamplesISM % 2
    nSamples = np.ceil(Tmax*fs)
    nSamples += nSamples % 2
    nSamplesDiff = nSamples - nSamplesISM

    # Gauss distribution
    mean = 0
    std = 1
    return np.random.normal(mean, std, size=int((nSamplesDiff*n_src*n_rcv)/20))