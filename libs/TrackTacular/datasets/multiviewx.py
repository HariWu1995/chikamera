import os

import cv2
import numpy as np

from torchvision.datasets import VisionDataset


# Camera Setting
intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml',
                                     'intr_Camera4.xml', 'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml',
                                     'extr_Camera4.xml', 'extr_Camera5.xml', 'extr_Camera6.xml']
worldcoord_from_worldgrid_matrix = [[0.025, 0    , 0], 
                                    [0    , 0.025, 0], 
                                    [0    , 0    , 1]]


class MultiviewX(VisionDataset):
    """
    Multiview-X
        im_shape: (C, H, W) / (C, N_row, N_col)
        im_area: H * W = 640 * 1000
        indexing: (x, y) / (w, h) / (n_col, n_row)
    x is \in [0,1000), y \in [0,640)
    Unit: meter (m) for calibration & pos annotation
    """
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'MultiviewX'

        self.img_shape = [1080, 1920]   # (H, W) ~ (N_row, N_col)
        self.grid_shape = [640, 1000]

        self.num_cam = 6
        self.num_frame = 400
        self.frame_step = 1

        # world x,y correspond to w,h
        self.worldcoord_from_worldgrid_matrix = np.array(worldcoord_from_worldgrid_matrix)
        
        self.intrinsic_matrices, \
        self.extrinsic_matrices = zip(*[self.get_intrinsic_extrinsic_matrix(cam) 
                                                                        for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {
            cam: {} for cam in range(self.num_cam)
        }
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            if camera_folder == '.DS_Store':
                continue
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1000
        grid_y = pos // 1000
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_matrix = self.get_intrinsic_matrix(camera_i)
        extrinsic_matrix = self.get_extrinsic_matrix(camera_i)
        return intrinsic_matrix, extrinsic_matrix

    def get_intrinsic_matrix(self, camera_i):
        camera_fn = intrinsic_camera_matrix_filenames[camera_i]
        camera_path = os.path.join(self.root, 'calibrations', 'intrinsic', camera_fn)
        params_file = cv2.FileStorage(camera_path, flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = params_file.getNode('camera_matrix').mat()
        params_file.release()
        return intrinsic_matrix

    def get_extrinsic_matrix(self, camera_i):
        camera_fn = extrinsic_camera_matrix_filenames[camera_i]
        camera_path = os.path.join(self.root, 'calibrations', 'extrinsic', camera_fn)
        params_file = cv2.FileStorage(camera_path, flags=cv2.FILE_STORAGE_READ)
        rvec = params_file.getNode('rvec').mat().squeeze()
        tvec = params_file.getNode('tvec').mat().squeeze()
        params_file.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
        return extrinsic_matrix
