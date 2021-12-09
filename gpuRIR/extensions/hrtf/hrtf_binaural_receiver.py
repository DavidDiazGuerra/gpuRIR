import numpy as np
import matplotlib.pyplot as plt

# Helper methods for HRTF.


class BinauralReceiver():
    '''
    TODO: Doc
    '''

    HEAD_WIDTH = 0.1449  # [m]
    PINNA_OFFSET_DOWN = 0.0303  # [m]
    PINNA_OFFSET_BACK = 0.0046  # [m]

    def __init__(self, head_position, head_direction, verbose=False):
        assert(head_direction[0] != 0 or head_direction[1] != 0), \
            "Ear directions are undefined for head pointing straight up or down! Try tilting a bit"
        self.position = head_position
        self.direction = head_direction
        self.update(head_position, head_direction, verbose)

    @staticmethod
    def find_spine_vector(head_direction):
        '''
        Finding a vector going down 90° from head direction (direction of spine).

        :param head_direction Direction the head is pointing towards.
        :returns New vector pointing down 90° from head direction (direction of spine).
        '''
        h = np.copy(head_direction)
        if h[2] == 0:
            h[0] = 0
            h[1] = 0
            h[2] = -1
        else:
            h[2] = -(h[0]**2 + h[1]**2) / h[2]
        return h / np.linalg.norm(h)

    @staticmethod
    def rotate_z_plane(vec, angle):
        '''
        Rotates the Z plane of a vector by given angle.

        :param vec Vector to turn.
        :param angle Angle the vector is being turned.
        :returns Rotated vector.
        '''
        vec_copy = np.copy(vec)

        z_rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
        vec_copy[2] = 0

        return vec_copy @ z_rotation

    def update(self, head_position, head_direction, verbose=False):
        '''
        Updates head position & direction in 3D space, calculates left and right ear position.
        TODO: Define paramenters
        '''
        self.position = head_position
        self.direction = head_direction

        self.ear_direction_r = np.round(
            BinauralReceiver.rotate_z_plane(self.direction, np.pi/2))
        self.ear_direction_l = -self.ear_direction_r

        ear_offset_back = BinauralReceiver.PINNA_OFFSET_BACK * \
            (self.direction / np.linalg.norm(self.direction, 2))

        ear_offset_down = BinauralReceiver.find_spine_vector(
            head_direction) * BinauralReceiver.PINNA_OFFSET_DOWN

        self.ear_position_r = (self.position + self.ear_direction_r * (BinauralReceiver.HEAD_WIDTH / 2)) #+ ear_offset_down - ear_offset_back

        self.ear_position_l = (self.position + self.ear_direction_l * (BinauralReceiver.HEAD_WIDTH / 2)) #+ ear_offset_down - ear_offset_back

        if head_direction[2] < 0:
            self.ear_position_r += -ear_offset_down - ear_offset_back
            self.ear_position_l += -ear_offset_down - ear_offset_back
        else:
            self.ear_position_r += ear_offset_down - ear_offset_back
            self.ear_position_l += ear_offset_down - ear_offset_back

        if verbose:
            print(f"Head position: {self.position}")
            print(f"Head direction: {self.direction}")
            print(f"Ear position L: {self.ear_position_l}")
            print(f"Ear position R: {self.ear_position_r}")
            print(f"Ear direction L: {self.ear_direction_l}")
            print(f"Ear direction R: {self.ear_direction_r}")
            print(f"Ear offset down: {ear_offset_down}")
            print(f"Ear offset back: {ear_offset_back}")

    def visualize(self, room_sz, pos_src, orV_src):
        '''
        TODO: Doc
        '''
        ax = plt.figure().add_subplot(projection='3d')
        np.meshgrid(
            np.arange(0, room_sz[0], 0.2),
            np.arange(0, room_sz[1], 0.2),
            np.arange(0, room_sz[2], 0.2)
        )

        # Plot origin -> head position
        ax.quiver(0, 0, 0, self.position[0], self.position[1], self.position[2],
                  arrow_length_ratio=0, color='gray', label="Head position")

        # Plot ear directions
        ax.quiver(self.ear_position_l[0], self.ear_position_l[1], self.ear_position_l[2], self.ear_direction_l[0],
                  self.ear_direction_l[1], self.ear_direction_l[2], length=BinauralReceiver.HEAD_WIDTH/2, color='b', label="Left ear direction")
        ax.quiver(self.ear_position_r[0], self.ear_position_r[1], self.ear_position_r[2], self.ear_direction_r[0], self.ear_direction_r[1],
                  self.ear_direction_r[2], length=BinauralReceiver.HEAD_WIDTH/2, color='r', label="Right ear direction")

        # Plot head direction
        ax.quiver(self.position[0], self.position[1], self.position[2], self.direction[0],
                  self.direction[1], self.direction[2], length=0.2, color='orange', label="Head direction")

        # Plot head -> signal source
        ax.quiver(self.position[0], self.position[1], self.position[2], pos_src[0][0] - self.position[0], pos_src[0][1] -
                  self.position[1], pos_src[0][2] - self.position[2], arrow_length_ratio=0.1, color='g', label="Signal source")
        # Plot head -> signal source
        ax.quiver(pos_src[0][0], pos_src[0][1], pos_src[0][2], orV_src[0], orV_src[1],
                  orV_src[2], arrow_length_ratio=0.1, color='black', label="Source steering")

        plt.legend()
        plt.show()
