from pytest import approx
import numpy as np
from gpuRIR.extensions.hrtf.hrtf_binaural_receiver import BinauralReceiver


def test_binaural_receiver():
    # Head pointing east
    head_position = [0, 0, 0]
    head_direction = [1, 0, 0]
    head = BinauralReceiver(head_position, head_direction)

    assert(head.ear_position_l[0] ==
           head_position[0]-BinauralReceiver.PINNA_OFFSET_BACK)
    assert(head.ear_position_r[0] ==
           head_position[0]-BinauralReceiver.PINNA_OFFSET_BACK)

    assert(head.ear_position_l[1] ==
           head_position[1]+BinauralReceiver.HEAD_WIDTH/2)
    assert(head.ear_position_r[1] ==
           head_position[1]-BinauralReceiver.HEAD_WIDTH/2)

    assert(head.ear_position_l[2] ==
           head_position[2]-BinauralReceiver.PINNA_OFFSET_DOWN)
    assert(head.ear_position_r[2] ==
           head_position[2]-BinauralReceiver.PINNA_OFFSET_DOWN)

    # Head pointing west
    head_position = [0, 0, 0]
    head_direction = [-1, 0, 0]
    head = BinauralReceiver(head_position, head_direction)

    assert(head.ear_position_l[0] ==
           head_position[0]+BinauralReceiver.PINNA_OFFSET_BACK)
    assert(head.ear_position_r[0] ==
           head_position[0]+BinauralReceiver.PINNA_OFFSET_BACK)

    assert(head.ear_position_l[1] ==
           head_position[1]-BinauralReceiver.HEAD_WIDTH/2)
    assert(head.ear_position_r[1] ==
           head_position[1]+BinauralReceiver.HEAD_WIDTH/2)

    assert(head.ear_position_l[2] ==
           head_position[2]-BinauralReceiver.PINNA_OFFSET_DOWN)
    assert(head.ear_position_r[2] ==
           head_position[2]-BinauralReceiver.PINNA_OFFSET_DOWN)

    # Head pointing north
    head_position = [0, 0, 0]
    head_direction = [0, 1, 0]
    head = BinauralReceiver(head_position, head_direction)
    
    assert(head.ear_position_l[0] ==
           head_position[0]-BinauralReceiver.HEAD_WIDTH/2)
    assert(head.ear_position_r[0] ==
           head_position[0]+BinauralReceiver.HEAD_WIDTH/2)

    assert(head.ear_position_l[1] ==
           head_position[1]-BinauralReceiver.PINNA_OFFSET_BACK)
    assert(head.ear_position_r[1] ==
           head_position[1]-BinauralReceiver.PINNA_OFFSET_BACK)

    assert(head.ear_position_l[2] ==
           head_position[2]-BinauralReceiver.PINNA_OFFSET_DOWN)
    assert(head.ear_position_r[2] ==
           head_position[2]-BinauralReceiver.PINNA_OFFSET_DOWN)

    # Head pointing south
    head_position = [0, 0, 0]
    head_direction = [0, -1, 0]
    head = BinauralReceiver(head_position, head_direction)
    
    assert(head.ear_position_l[0] ==
           head_position[0]+BinauralReceiver.HEAD_WIDTH/2)
    assert(head.ear_position_r[0] ==
           head_position[0]-BinauralReceiver.HEAD_WIDTH/2)

    assert(head.ear_position_l[1] ==
           head_position[1]+BinauralReceiver.PINNA_OFFSET_BACK)
    assert(head.ear_position_r[1] ==
           head_position[1]+BinauralReceiver.PINNA_OFFSET_BACK)

    assert(head.ear_position_l[2] ==
           head_position[2]-BinauralReceiver.PINNA_OFFSET_DOWN)
    assert(head.ear_position_r[2] ==
           head_position[2]-BinauralReceiver.PINNA_OFFSET_DOWN)
    
    # Head pointing -45°
    head_position = [0, 0, 0]
    head_direction = [1, 1, -1]
    head = BinauralReceiver(head_position, head_direction)

    assert(head.ear_position_l[0] ==
           approx(-0.08747573))
    assert(head.ear_position_r[0] ==
           approx(0.05742427))
    
    assert(head.ear_position_l[1] ==
           approx(0.05742427))
    assert(head.ear_position_r[1] ==
           approx(-0.08747573))
    
    assert(head.ear_position_l[2] ==
           approx(-0.02208404))
    assert(head.ear_position_r[2] ==
           approx(-0.02208404))

    # Head pointing +45° 
    head_position = [0, 0, 0]
    head_direction = [1, 1, 1]
    head = BinauralReceiver(head_position, head_direction)
        
    assert(head.ear_position_l[0] ==
           approx(-0.06273589))
    assert(head.ear_position_r[0] ==
           approx(0.08216411))
    
    assert(head.ear_position_l[1] ==
           approx(0.08216411))
    assert(head.ear_position_r[1] ==
           approx(-0.06273589))
    
    assert(head.ear_position_l[2] ==
           approx(-0.02739566))
    assert(head.ear_position_r[2] ==
           approx(-0.02739566))