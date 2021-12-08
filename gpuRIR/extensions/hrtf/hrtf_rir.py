import numpy as np
from scipy.io import loadmat
from scipy.signal import spectrogram
from matplotlib import pyplot as plt

class HRTF_RIR:
    def __init__(self):
        self.azimuths = np.float32([
            -80, -65, -55, -45, -40,
            -35, -30, -25, -20, -15,
            -10, -5, 0, 5, 10,
            15, 20, 25, 30, 35,
            40, 45, 55, 65, 80
        ])

        self.elev_step = (360/64)
        self.elevations = np.arange(-45, 230.625 + self.elev_step, self.elev_step, dtype=np.float32)
        self.hrir = loadmat('gpuRIR/extensions/hrtf/cipic_hrir/hrir_final.mat')

    def azimuth_to_idx(self, azimuth):
        return int(np.argmin(np.abs(self.azimuths - azimuth)))

    def elevation_to_idx(self, elevation):
        return int(np.argmin(np.abs(self.elevations - elevation)))


    def get_hrtf_rir(self, elevation, azimuth, channel, visualize=False):
        hrir_channel = self.hrir['hrir_'+channel][:, self.elevation_to_idx(elevation), :]

        if visualize:
            x = hrir_channel[self.azimuth_to_idx(azimuth)]
            plt.title("HRTF Frequency Response")
            plt.rcParams.update({'font.size': 18})
            plt.plot(x)
            plt.xlabel("Sample [n]")
            plt.ylabel("Amplitude")
            plt.show()
        return hrir_channel[self.azimuth_to_idx(azimuth)]

obj = HRTF_RIR()
obj.get_hrtf_rir(0, 0, 'r', True)
