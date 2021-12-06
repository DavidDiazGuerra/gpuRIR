
# Virtual wall materials section ------------------------------------------------------------------------------------

# If True, apply frequency dependent wall absorption coefficients to simulate realistic wall/ceiling/floor materials.
# Caution: Needs more resources!
freq_dep_abs_coeff = False

visualize = True

# Wall, floor and ceiling materials the room is consisting of
# Structure: Array of six materials (use 'mat.xxx') corresponding to:
# Left wall | Right wall | Front wall | Back wall | Floor | Ceiling
wall_materials = 4 * [mat.cinema_screen]+[mat.carpet_10mm]+[mat.concrete]

# HRTF section -------------------------------------------------------------------------------------------------------

use_hrtf = False

# Parameters referring to head related transfer functions (HRTF).
head_width = 0.1449  # [m]
head_position = [1.5, 1.5, 1.6]  # [m]
head_direction = [0, -1, 0]  # [m]

pinna_offset_down = 0.0303  # [m]
pinna_offset_back = 0.0046  # [m]

def rotate_z_plane(vec, angle):
    vec_copy = np.copy(vec)

    z_rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    vec_copy[2] = 0
    return vec_copy @ z_rotation


ear_direction_r = np.round(rotate_z_plane(head_direction, np.pi/2))
ear_direction_l = -ear_direction_r

ear_offset_vector = pinna_offset_back * \
    (head_direction / np.linalg.norm(head_direction, 2))

ear_position_r = (head_position + ear_direction_r * (head_width / 2)) - \
    np.array([0, 0, pinna_offset_down]) - ear_offset_vector
ear_position_l = (head_position + ear_direction_l * (head_width / 2)) - \
    np.array([0, 0, pinna_offset_down]) - ear_offset_vector


# Common gpuRIR parameters (applied to both channels) ------------------------------------------------------------------

room_sz = [5, 4, 3]  # Size of the room [m]
pos_src = [[1.5, 1.8, 1.8]]  # Positions of the sources [m]
orV_src = [0, -1, 0]  # Steering vector of source(s)
spkr_pattern = "omni"  # Source polar pattern
mic_pattern = "homni"  # Receiver polar patterny
T60 = 1.0  # Time for the RIR to reach 60dB of attenuation [s]
# Attenuation when start using the diffuse reverberation model [dB]
att_diff = 15.0
att_max = 60.0  # Attenuation at the end of the simulation [dB]
fs = 44100  # Sampling frequency [Hz]
# Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit
bit_depth = np.int32

if visualize:
    ax = plt.figure().add_subplot(projection='3d')
    x, y, z = np.meshgrid(
        np.arange(0, room_sz[0], 0.2),
        np.arange(0, room_sz[1], 0.2),
        np.arange(0, room_sz[2], 0.2)
    )

    # Plot origin -> head position
    ax.quiver(0, 0, 0, head_position[0], head_position[1], head_position[2],
                arrow_length_ratio=0, color='gray', label="Head position")

    # Plot ear directions
    ax.quiver(ear_position_l[0], ear_position_l[1], ear_position_l[2], ear_direction_l[0],
                ear_direction_l[1], ear_direction_l[2], length=head_width/2, color='b', label="Left ear direction")
    ax.quiver(ear_position_r[0], ear_position_r[1], ear_position_r[2], ear_direction_r[0], ear_direction_r[1],
                ear_direction_r[2], length=head_width/2, color='r', label="Right ear direction")

    # Plot head direction
    ax.quiver(head_position[0], head_position[1], head_position[2], head_direction[0],
                head_direction[1], head_direction[2], length=0.2, color='orange', label="Head direction")

    # Plot head -> signal source
    ax.quiver(head_position[0], head_position[1], head_position[2], pos_src[0][0] - head_position[0], pos_src[0][1] -
                head_position[1], pos_src[0][2] - head_position[2], arrow_length_ratio=0.1, color='g', label="Signal source")
    # Plot head -> signal source
    ax.quiver(pos_src[0][0], pos_src[0][1], pos_src[0][2], orV_src[0], orV_src[1],
                orV_src[2], arrow_length_ratio=0.1, color='black', label="Source steering")

    plt.legend()
    plt.show()

# Define room parameters
params_left = rp.RoomParameters(
    room_sz=room_sz,
    pos_src=pos_src,
    orV_src=orV_src,
    spkr_pattern=spkr_pattern,
    mic_pattern=mic_pattern,
    T60=T60,
    att_diff=att_diff,
    att_max=att_max,
    fs=fs,
    bit_depth=bit_depth,
    wall_materials=wall_materials,

    # Positions of the receivers [m]
    pos_rcv=[ear_position_l],  # Position of left ear
    orV_rcv=ear_direction_l,  # Steering vector of left ear
    head_direction=head_direction,
    head_position=head_position
)

params_right = rp.RoomParameters(
    room_sz=room_sz,
    pos_src=pos_src,
    orV_src=orV_src,
    spkr_pattern=spkr_pattern,
    mic_pattern=mic_pattern,
    T60=T60,
    att_diff=att_diff,
    att_max=att_max,
    fs=fs,
    bit_depth=bit_depth,
    wall_materials=wall_materials,

    # Positions of the receivers [m]
    pos_rcv=[ear_position_r],  # Position of right ear
    orV_rcv=ear_direction_r,  # Steering vector of right ear
    head_direction=head_direction,
    head_position=head_position
)

# Generate two room impulse responses (RIR) with given parameters for each ear
if freq_dep_abs_coeff:
    receiver_channel_r = fdac.generate_RIR_freq_dep_walls(params_right)
    receiver_channel_l = fdac.generate_RIR_freq_dep_walls(params_left)

else:
    receiver_channel_r = generate_RIR(params_right)
    receiver_channel_l = generate_RIR(params_left)

# All listed filters wil be applied in that order.
# Leave filters array empty if no filters should be applied.
filters_both = [
    # AirAbsBandpass(),
]

filters_r = filters_both + [HRTF_Filter('r', params_right)]
filters_l = filters_both + [HRTF_Filter('l', params_left)]

generate_stereo_IR(receiver_channel_r, receiver_channel_l,
                    filters_r, filters_l, bit_depth, fs)