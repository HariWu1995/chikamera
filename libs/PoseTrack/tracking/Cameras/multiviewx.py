"""
Dataset Structure:

    └── MultiviewX
        ├── Image_subsets
        ├── calibrations
        │   ├── extrinsic 
        │   │   ├── extr_Camera1.xml
        │   │   └── ...
        │   └── intrinsic
        │       ├── intr_Camera1.xml
        │       └── ...
        └── ...

extr_Camera1.xml:

    <opencv_storage>

        <rvec type_id="opencv-matrix">
            <rows>3</rows>
            <cols>1</cols>
            <dt>d</dt>
            <data> [...] </data>
        </rvec>

        <tvec type_id="opencv-matrix">
            <rows>3</rows>
            <cols>1</cols>
            <dt>d</dt>
            <data> [...] </data>
        </tvec>

    </opencv_storage>

intr_Camera1.xml:

    <opencv_storage>

        <camera_matrix type_id="opencv-matrix">
            <rows>3</rows>
            <cols>3</cols>
            <dt>d</dt>
            <data> 8.9999963676899733e+02 0. 9.5999969734520107e+02 0. 8.9999954513583339e+02 5.4000000686696694e+02 0. 0. 1.</data>
        </camera_matrix>

        <distortion_coefficients type_id="opencv-matrix">
            <rows>1</rows>
            <cols>5</cols>
            <dt>d</dt>
            <data> 9.4697674822818266e-07 -1.5758488655936218e-06 1.7699546431889683e-08 -1.6041668775652409e-07 5.6752029903185227e-07</data>
        </distortion_coefficients>

    </opencv_storage>
"""
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import scipy


def load_camera_parameters(intr_file, extr_file):
    """
    Load extrinsic and intrinsic parameters from XML files.

    Projection Matrix:
        https://camtools.readthedocs.io/en/stable/camera.html

    Homography matrix:
        https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
    """
    fs_extr = cv2.FileStorage(extr_file, cv2.FILE_STORAGE_READ)
    rvec = fs_extr.getNode("rvec").mat()
    tvec = fs_extr.getNode("tvec").mat()
    fs_extr.release()

    fs_intr = cv2.FileStorage(intr_file, cv2.FILE_STORAGE_READ)
    intrinsic_matrix = fs_intr.getNode("camera_matrix").mat()
    fs_intr.release()
    
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Construct extrinsic matrix [R | t]
    extrinsic_matrix = np.hstack((R, tvec))

    # Compute projection matrix P = K * [R | t].
    proj_matrix = intrinsic_matrix @ extrinsic_matrix

    # Consider only the first 2 columns of R and t (ignoring depth component)
    homo_matrix = np.dot(intrinsic_matrix, np.hstack((R[:, :2], tvec)))

    all_params = dict()
    all_params["height"] = tvec[1, 0]  # NOTE: Y-axis: 1 / Z-axis: 2
    all_params["intrinsic_matrix"] = intrinsic_matrix
    all_params["extrinsic_matrix"] = extrinsic_matrix
    all_params["projection_matrix"] = proj_matrix
    all_params["homography_matrix"] = homo_matrix
    return all_params


class Camera:

    z = 0.15 # FIXME: assumption

    def __init__(self, in_config_path, ex_config_path):

        # index (str) in whole dataset
        self.name = os.path.splitext(in_config_path)[0][-1]
        self.idx = int(self.name)

        all_params = load_camera_parameters(in_config_path, ex_config_path)
        self.height = all_params["height"]
        self.intr_mat = all_params["intrinsic_matrix"]
        self.extr_mat = all_params["extrinsic_matrix"]

        self.project_mat = all_params["projection_matrix"]
        self.project_inv = scipy.linalg.pinv(self.project_mat)

        # _3d_to_2d_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, self.z], [0, 0, 1]])
        # self.project2d_mat = self.project_mat @ _3d_to_2d_

        self.homo_mat = all_params["homography_matrix"]
        self.homo_inv = np.linalg.inv(self.homo_mat)

        self.pos = np.linalg.inv(self.project_mat[:,:-1]) @ - self.project_mat[:,-1]

        self.homo_feet = self.homo_mat.copy()
        self.homo_feet[:, -1] = self.homo_feet[:, -1] + self.project_mat[:, 2] * self.z
        self.homo_feet_inv = np.linalg.inv(self.homo_feet)


if __name__ == "__main__":

    root_path = "F:/__Datasets__/MultiviewX/calibrations"
    # camera = Camera(in_config_path=f"{root_path}/intrinsic/intr_Camera1.xml",
    #                 ex_config_path=f"{root_path}/extrinsic/extr_Camera1.xml")
    # print(camera.__dict__)
    
    Ks, Ts, hs = [], [], []
    for cam_id in range(1, 7):
        cam = Camera(in_config_path=f"{root_path}/intrinsic/intr_Camera{cam_id}.xml",
                     ex_config_path=f"{root_path}/extrinsic/extr_Camera{cam_id}.xml")
        hs.append(cam.height)
        Ks.append(cam.intr_mat)
        Ts.append(np.concatenate([cam.extr_mat, np.array([[0, 0, 0, 1]])], axis=0))
    print(hs)

    import sys
    sys.path.append("src/camera")

    from visualization import create_camera_frustums, visualize_camera_mesh
    camera_mesh = create_camera_frustums(Ks, Ts, camera_distance=np.max(hs), 
                                                 randomize_color=True)
    visualize_camera_mesh(camera_mesh, interactive=True)

