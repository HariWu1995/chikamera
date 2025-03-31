"""
Dataset Structure: 

    └── ICSens (stereo camera)
        ├── images
        ├── calibration
        │   ├── view1
        │   ├── view2
        │   └── view3
        │       ├── absolute.txt
        │       ├── extrinsics.txt
        │       └── intrinsics.txt
        └── ...

absolute.txt
    X0: shape (3,) 
    rpy: shape (3,)
    R: shape (9,)

extrinsics.txt

    Covar: !!opencv-matrix
        rows: 24
        cols: 24
        dt: d
        data: [...]

    F: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [...]

    E: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [...]

    R: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [...]

    T: !!opencv-matrix
        rows: 3
        cols: 1
        dt: d
        data: [...]

    R1: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [...]

    R2: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [...]

    P1: !!opencv-matrix
        rows: 3
        cols: 4
        dt: d
        data: [...]

    P2: !!opencv-matrix
        rows: 3
        cols: 4
        dt: d
        data: [...]

    Q: !!opencv-matrix
        rows: 4
        cols: 4
        dt: d
        data: [...]

    mirrorY: 1
    ROI1x: 0
    ROI1y: 0
    ROI1w: 1920
    ROI1h: 1213
    ROI2x: 0
    ROI2y: 1
    ROI2w: 1920
    ROI2h: 1215
    Offsetx: 0.
    Offsety: 0.
    Offsetz: 0.

intrinsics.txt

    M1: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [...]

    D1: !!opencv-matrix
        rows: 5
        cols: 1
        dt: d
        data: [...]

    Cov1: !!opencv-matrix
        rows: 9
        cols: 9
        dt: d
        data: [...]

    s1: 6144
    w1: 1920
    h1: 1216

    M2: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [...]

    D2: !!opencv-matrix
        rows: 5
        cols: 1
        dt: d
        data: [...]

    Cov2: !!opencv-matrix
        rows: 9
        cols: 9
        dt: d
        data: [...]

    s2: 6144
    w2: 1920
    h2: 1216

"""
import os
import json
import itertools

import cv2
import numpy as np
import scipy


Cam2Id = {
    f"view{vi}_{cam}": i 
    for i, (vi, cam) in enumerate(itertools.product(list(range(1, 4)), ['left','right']))
}

Id2Cam = {v: k for k, v in Cam2Id.items()}


def load_camera_parameters(intr_file, extr_file, cam_id: int = 1):
    """
    Load extrinsic and intrinsic parameters from XML files.

    Args:
        cam_id: 1 = left, 2 = right

    Projection Matrix:
        https://camtools.readthedocs.io/en/stable/camera.html

    Homography matrix:
        https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
    """
    # Load absolute parameters
    # with open(abs_file, 'r') as file:
    #     lines = file.readlines()
    #     x, y, z = [float(x) for x in lines[0].split(':')[1].strip().split()]  # Position in World Coordinates
    #     r, p, y = [float(x) for x in lines[1].split(':')[1].strip().split()]  # Roll, Pitch, Yaw Angles
    #     R = np.array([float(x) for x in lines[2].split(':')[1].strip().split()]).reshape(3, 3)

    # Load extrinsic parameters
    fs_extr = cv2.FileStorage(extr_file, cv2.FILE_STORAGE_READ)
    R = fs_extr.getNode("R").mat()
    T = fs_extr.getNode("T").mat()
    fs_extr.release()
    
    # Load intrinsic parameters
    fs_intr = cv2.FileStorage(intr_file, cv2.FILE_STORAGE_READ)
    M = fs_intr.getNode(f"M{cam_id}").mat()
    fs_intr.release()

    # Construct extrinsic matrix [R | t]
    extrinsic_matrix = np.hstack((R, T))

    # Compute projection matrix P = K * [R | t].
    proj_matrix = np.dot(M, extrinsic_matrix)

    # Consider only the first 2 columns of R and t (ignoring depth component)
    homo_matrix = np.dot(M, np.hstack((R[:, :2], T)))

    all_params = dict()
    all_params["projection_matrix"] = proj_matrix
    all_params["homography_matrix"] = homo_matrix
    return all_params


class Camera:

    z = 0.15 # FIXME: assumption
    cam_ids = dict(left = 1, right = 2)

    def __init__(self, in_config_path, ex_config_path, stero_relative):

        cam_id = os.path.normpath(in_config_path).split(os.sep)[-2]
        cam_id = f"{cam_id}_{stero_relative}"

        # index (str) in whole dataset
        self.name = cam_id
        self.idx = Cam2Id[self.name]

        all_params = load_camera_parameters(in_config_path, 
                                            ex_config_path,
                                            cam_id = self.cam_ids[stero_relative])

        self.project_mat = all_params["projection_matrix"]
        self.project_inv = scipy.linalg.pinv(self.project_mat)

        self.homo_mat = all_params["homography_matrix"]
        self.homo_inv = np.linalg.inv(self.homo_mat)

        self.pos = np.linalg.inv(self.project_mat[:,:-1]) @ - self.project_mat[:,-1]

        self.homo_feet = self.homo_mat.copy()
        self.homo_feet[:, -1] = self.homo_feet[:, -1] + self.project_mat[:, 2] * self.z
        self.homo_feet_inv = np.linalg.inv(self.homo_feet)


if __name__ == "__main__":

    root_path = "F:/__Datasets__/ICSens/calibration/view1"
    camera = Camera(
             # abs_config_path=f"{root_path}/absolute.txt",
                in_config_path=f"{root_path}/intrinsics.txt",
                ex_config_path=f"{root_path}/extrinsics.txt",
                stero_relative='right',
            )
    print(camera.homo_feet_inv)
