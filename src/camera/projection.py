from typing import List, Tuple, Union

import math
import numpy as np


def calculate_image_plane(camera_distance: int, camera_fov_degree: int):
    """
    Calculate the approximate size of the image plane based on camera properties.
    """
    camera_fov = math.radians(camera_fov_degree)
    plane_size = 2 * camera_distance * math.tan(camera_fov / 2)
    return plane_size


def calculate_image_plane_hv(camera_distance: int, camera_hfov_degree: int, 
                                                   camera_vfov_degree: int):
    """
    Calculate the approximate size of the image plane 
        in case FoV|horizontal != FoV|vertical.
    """
    plane_size_h = calculate_image_plane(camera_distance, camera_hfov_degree)
    plane_size_v = calculate_image_plane(camera_distance, camera_vfov_degree)
    return plane_size_h, plane_size_v


def compute_intrinsic_matrix(
        resolution_h: int, 
        resolution_v: int,

        # Focal approach
        f: int = None,  # focal length
        fx: int = None,  # sensor size in horizontal
        fy: int = None,  # sensor size in vertical

        # Resolution approach
        camera_hfov_degree: int = None, 
        camera_vfov_degree: int = None,
    ):
    Cx = resolution_h // 2
    Cy = resolution_v // 2

    if (f is not None) and (fx is not None) and (fy is not None):
        Fx = f * resolution_h / fx
        Fy = f * resolution_v / fy

    elif (camera_hfov_degree is not None) and (camera_vfov_degree is not None):
        camera_hfov = math.radians(camera_hfov_degree)
        camera_vfov = math.radians(camera_vfov_degree)
        Fx = resolution_h / (2 * math.tan(camera_hfov / 2))
        Fy = resolution_v / (2 * math.tan(camera_vfov / 2))

    else:
        raise ValueError('Cannot calculate Fx, Fy with current inputs!')

    K = [[Fx,  0, Cx], 
         [ 0, Fy, Cy], 
         [ 0,  0,  1]]

    return np.array(K)


def compute_extrinsic_matrix(
        world_coord: Tuple[int],    #       x,      y,       z
         rot_angles: Tuple[int],    # pitch(x), yaw(y), roll(z)
    ):

    x, y, z = world_coord
    pitch, yaw, roll = rot_angles

    yaw = math.radians(yaw)
    roll = math.radians(roll)
    pitch = math.radians(pitch)

    # Calculate translation vector
    t = np.array([x, y, z]).reshape(3, 1)

    # Calculate rotation matrices
    Rx = [[1,               0,                0],
          [0, math.cos(pitch), -math.sin(pitch)],
          [0, math.sin(pitch),  math.cos(pitch)]]

    Ry = [[ math.cos(yaw), 0, math.sin(yaw)],
          [             0, 1,             0],
          [-math.sin(yaw), 0, math.cos(yaw)]]

    Rz = [[math.cos(roll), -math.sin(roll), 0],
          [math.sin(roll),  math.cos(roll), 0],
          [             0,               0, 1]]

    Rx = np.array(Rx)
    Ry = np.array(Ry)
    Rz = np.array(Rz)

    R = Rz @ Ry @ Rx

    # Combine
    T = np.hstack((R, t))
    T = np.vstack((T, [0, 0, 0, 1]))

    return T
    

if __name__ == "__main__":

    # np.set_printoptions(precision=2)

    T = compute_extrinsic_matrix((0, 0, 10), (90, 0, 0))
    print(np.round(T, decimals=2))


