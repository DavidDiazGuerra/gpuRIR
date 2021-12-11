import numpy as np

from gpuRIR.extensions.filters.filter import FilterStrategy
from gpuRIR.extensions.hrtf.hrtf_rir import HRTF_RIR


class HRTF_Filter(FilterStrategy):
    ANGLE_90 = np.pi/2
    ANGLE_180 = np.pi

    def __init__(self, channel, params, verbose=False, visualize=False):
        self.channel = channel
        self.NAME = "HRTF"
        self.params = params
        self.hrtf_rir = HRTF_RIR()
        self.verbose = verbose
        self.visualize = visualize

    @staticmethod
    def find_angle(u, v):
        '''
        Find angle via trigonometry
        '''
        norm_product = (np.linalg.norm(u) * np.linalg.norm(v))
        
        if norm_product != 0:
            return np.arccos((u @ v) / norm_product) 
        
        return 0

    # Find elevation between head and source

    @staticmethod
    def calculate_elevation(pos_src, pos_rcv, head_direction):
        # Height of source
        opposite = np.abs(pos_src[2] - pos_rcv[2])

        # Length of floor distance between head and source
        adjacent = np.linalg.norm(
            np.array([pos_src[0], pos_src[1]]) - np.array([pos_rcv[0], pos_rcv[1]]))

        # Find elevation between head and source positions
        if adjacent != 0:
            el_rcv_src = np.arctan(opposite / adjacent) 
        else:
            el_rcv_src=np.arctan(np.inf)

        # Edge case if source is below head
        if pos_rcv[2] > pos_src[2]:
            el_rcv_src = -el_rcv_src

        # Height of receiver
        opposite = np.abs(head_direction[2])

        # Length of floor distance between head and head direction vector
        adjacent = np.linalg.norm(head_direction)

        # Calculate elevation between head and head direction
        el_rcv_dir = np.arctan(opposite / adjacent)
        elevation_angle = el_rcv_src - el_rcv_dir

        # Edge case if source is behind head
        angle, _, _ = HRTF_Filter.vector_between_points(
            pos_src, pos_rcv, head_direction)
        if angle > HRTF_Filter.ANGLE_90:
            # Source is behind head
            elevation_angle = HRTF_Filter.ANGLE_180 - elevation_angle

        # Subtract elevation between head and source and between head and head direction
        return elevation_angle

    @staticmethod
    def vector_between_points(pos_src, pos_rcv, head_direction):
        # 3D vector from head position (origin) to source
        head_to_src = pos_src - pos_rcv

        # Extract 2D array from 3D
        head_to_src = np.array([head_to_src[0], head_to_src[1]])
        # Extract 2D array from 3D
        headdir_xy = [head_direction[0], head_direction[1]]

        # Return angle using trigonometry
        return HRTF_Filter.find_angle(headdir_xy, head_to_src), head_to_src, headdir_xy

    @staticmethod
    def calculate_azimuth(pos_src, pos_rcv, head_direction):
        # Find angle using trigonometry
        angle, head_to_src, headdir_xy = HRTF_Filter.vector_between_points(
            pos_src, pos_rcv, head_direction)

        # Check if azimuth goes above 90°
        if angle > HRTF_Filter.ANGLE_90:
            angle = np.pi - angle

        # Check left/right. If positive direction is left, if negative direction is right.
        side = np.sign(np.linalg.det([headdir_xy, head_to_src]))

        return angle * (-side)

    def hrtf_convolve(self, IR):
        elevation = self.calculate_elevation(
            self.params.pos_src[0], self.params.head_position, self.params.head_direction)

        if self.verbose:
            print(f"Elevation = {elevation * (180 / np.pi)}")

        azimuth = self.calculate_azimuth(
            self.params.pos_src[0], self.params.head_position, self.params.head_direction)

        if self.verbose:
            print(f"Azimuth = {azimuth * (180 / np.pi)}")

        hrir_channel = self.hrtf_rir.get_hrtf_rir(
            elevation, azimuth, self.channel)

        return np.convolve(IR[0], hrir_channel, mode='same')

    def apply(self, IR):
        return self.hrtf_convolve(IR)
