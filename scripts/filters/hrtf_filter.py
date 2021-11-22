import numpy as np
from filters.filter import FilterStrategy
from hrtf.hrtf_rir import HRTF_RIR


class HRTF_Filter(FilterStrategy):
    def __init__(self, channel, params):
        self.channel = channel
        self.NAME = "HRTF"
        self.params = params
        self.hrtf_rir = HRTF_RIR()

    @staticmethod
    def find_angle(u, v):
        return np.arccos((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))


    # Find elevation between head and source
    @staticmethod
    def calculate_elevation(pos_src, pos_rcv, rcv_orV):
        # Height of source
        opposite = np.abs(pos_src[2]-pos_rcv[2]) 

        # Length of floor distance between head and source
        adjacent = np.linalg.norm(
            np.array([pos_src[0], pos_src[1]]) - np.array([pos_rcv[0], pos_rcv[1]]))

        # Find elevation between head and source positions
        el_rcv_src = np.arctan(opposite / adjacent)

        # Edge case if source is below head
        if pos_rcv[2] > pos_src[2]:
            el_rcv_src = -el_rcv_src

        # Height of receiver
        opposite = np.abs(rcv_orV[2])

        # Length of floor distance between head and head direction vector
        adjacent = np.linalg.norm(rcv_orV)

        # Calculate elevation between head and head direction
        el_rcv_dir = np.arctan(opposite / adjacent)

        # Subtract elevation between head and source and between head and head direction
        return el_rcv_src - el_rcv_dir


    @staticmethod
    def calculate_azimuth(pos_src, pos_rcv, rcv_orV):
        # 3D vector from head position (origin) to source
        head_to_src = pos_src - pos_rcv
        # Extract 2D array from 3D
        head_to_src = np.array([head_to_src[0], head_to_src[1]])
        headdir_xy = [rcv_orV[0], rcv_orV[1]]  # Extract 2D array from 3D
        return HRTF_Filter.find_angle(headdir_xy, head_to_src)


    def hrtf_convolve(self, IR):
        elevation = self.calculate_elevation(
            self.params.pos_src[0], self.params.pos_rcv[0], self.params.orV_rcv[0])

        print(f"Elevation = {elevation * (180 / np.pi)}")

        azimuth = self.calculate_azimuth(
            self.params.pos_src[0], self.params.pos_rcv[0], self.params.orV_rcv[0])

        print(f"Azimuth = {azimuth * (180 / np.pi)}")

        hrir_channel = self.hrtf_rir.get_hrtf_rir(
            elevation, azimuth, self.channel)

        return np.convolve(IR[0], hrir_channel, mode='same')


    def apply(self, IR):
        return self.hrtf_convolve(IR)
