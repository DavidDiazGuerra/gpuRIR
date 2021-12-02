from pytest import approx
import numpy as np
from filters.hrtf_filter import HRTF_Filter


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

    # Left (-90°)
    pos_src = np.array([0, 0, 0])
    pos_rcv = np.array([0, 1, 0])
    head_direction = [-1, 0, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == -np.pi/2

    # Right (+90°)
    pos_src = np.array([0, 0, 0])
    pos_rcv = np.array([0, 1, 0])
    head_direction = [1, 0, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == np.pi/2

    # Front Left (-45°)
    pos_src = np.array([-1, 1, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == approx(-np.pi/4)

    # Front Right (+45°)
    pos_src = np.array([1, 1, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == approx(np.pi/4)

    # Rear Left (-45°)
    pos_src = np.array([-1, -1, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == approx(-np.pi/4)

    # Rear Right (+45°)
    pos_src = np.array([-1, 1, 0])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, -1, 0]
    assert HRTF_Filter.calculate_azimuth(
        pos_src, pos_rcv, head_direction) == approx(np.pi/4)


def test_elevation():
    # Same elevation
    pos_src = np.array([0, 0, 0])
    pos_rcv = np.array([1, 0, 0])
    head_direction = [-1, 0, 0]
    assert HRTF_Filter.calculate_elevation(pos_src, pos_rcv, head_direction) == 0

    # +45°
    pos_src = np.array([0, 1, 1])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    assert HRTF_Filter.calculate_elevation(
        pos_src, pos_rcv, head_direction) == np.pi/4

    # Above head
    pos_src = np.array([0, 0, 0])
    pos_rcv = np.array([0, 0, 1])
    head_direction = [1, 0, 0]
    assert HRTF_Filter.calculate_elevation(pos_src, pos_rcv, head_direction) == np.pi/2

    # -45°
    pos_src = np.array([0, 1, 1])
    pos_rcv = np.array([0, 0, 0])
    head_direction = [0, 1, 0]
    #assert HRTF_Filter.calculate_elevation(
    #    pos_src, pos_rcv, head_direction) == np.pi/4

    
