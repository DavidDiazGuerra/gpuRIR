from pytest import approx
import numpy as np
from gpuRIR.extensions.filters.hrtf_filter import HRTF_Filter

ANGLE_180 = approx(np.pi)

ANGLE_90 = approx(np.pi / 2)
ANGLE_NEG_90 = approx(-(np.pi / 2))

ANGLE_45 = approx(np.pi / 4)
ANGLE_NEG_45 = approx(-(np.pi / 4))

ANGLE_135 = approx(3 * (np.pi / 4))

ANGLE_225 = approx(np.pi + (np.pi / 4))

def test_azimuth():
    # Front
    pos_src = np.array([0, 0, 0])
    pos_rcv = np.array([1, 0, 0])
    head_direction = [-1, 0, 0]
    assert HRTF_Filter.calculate_azimuth(pos_src, pos_rcv, head_direction) == 0

    # Rear
    pos_src = np.array([1, 0, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [1, 0, 0]
    assert HRTF_Filter.calculate_azimuth(pos_src, pos_rcv, head_direction) == 0

    # Left (-90 degrees)
    pos_src = np.array([0, 0, 0])
    pos_rcv = np.array([0, 1, 0])
    head_direction = [-1, 0, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == ANGLE_NEG_90

    # Right (+90 degrees)
    pos_src = np.array([0, 0, 0])
    pos_rcv = np.array([0, 1, 0])
    head_direction = [1, 0, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == ANGLE_90

    # Front Left (-45 degrees)
    pos_src = np.array([-1, 1, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == ANGLE_NEG_45

    # Front Right (+45 degrees)
    pos_src = np.array([1, 1, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == ANGLE_45

    # Rear Left (-45 degrees)
    pos_src = np.array([-1, -1, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == ANGLE_NEG_45

    # Rear Right (+45 degrees)
    pos_src = np.array([-1, 1, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, -1, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == ANGLE_45


def test_elevation():
    # In front of head (+45 degrees)
    pos_src = np.array([0, 1, 1])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == ANGLE_45

    # In front of head (-45 degrees)
    pos_src = np.array([0, 1, -1])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == ANGLE_NEG_45

    # Above head (90 degrees)
    pos_src = np.array([0, 0, 1])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [1, 1, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == ANGLE_90

    # Below head (-90 degrees)
    pos_src = np.array([0, 0, 1])
    pos_rcv = np.array([0, 0, 2])
    head_direction = [1, 1, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == ANGLE_NEG_90

    # In front of head (0 degrees) (same elevation)
    pos_src = np.array([0, 0, 0])
    pos_rcv = np.array([1, 0, 0])
    head_direction = [-1, 0, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == 0

    # Behind head (180 degrees) (same elevation)
    pos_src = np.array([-1, 0, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [1, 0, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == ANGLE_180

    # Behind head (224 degrees)
    pos_src = np.array([-1, 0, -1])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [1, 0, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == ANGLE_225

    # Behind head (135 degrees)
    pos_src = np.array([-1, 0, 1])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [1, 0, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == ANGLE_135

    # Behind head (153 degrees)
    pos_src = np.array([-1, 0, 0.5])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [1, 0, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == approx(np.pi * .8524163) 
