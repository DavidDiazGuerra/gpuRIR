import numpy as np

'''
Generates white noise
'''
def generate_white_noise(Tdiff, Tmax, fs, n_src=1, n_rcv=1):
    # Check if parameters exceed 1 for n_src and n_rcv
    # TODO expand to more rcv's/src's
    assert(n_src == 1 and n_rcv ==
           1), "multiple sources & receivers not yet implemented"

    nThreadsRed = 128

    nSamplesISM = np.ceil(Tdiff*fs)
    nSamplesISM += nSamplesISM % 2
    nSamples = np.ceil(Tmax*fs)
    nSamples += nSamples % 2
    nSamplesDiff = nSamples - nSamplesISM

    # Uniform distribution
    low = 0
    high = 1
    size = int((nSamplesDiff*n_src*n_rcv)/nThreadsRed)
    print(f"booba size {size} (very lorge)")
    return np.random.uniform(low, high, size)
