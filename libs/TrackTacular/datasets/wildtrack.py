import os
import re

import xml.etree.ElementTree as ET

import numpy as np
import cv2

from torchvision.datasets import VisionDataset


# Camera Setting
intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']
worldcoord_from_worldgrid_matrix = [[  0, 2.5, -300], 
                                    [2.5,   0, -900], 
                                    [  0,   0,    1]]


class Wildtrack(VisionDataset):
    """
    WILDTRACK
        im_shape: (C, H, W) / (C, N_row, N_col)
        im_area: H * W = 480 * 1440
        indexing: (x, y) / (w, h) / (n_col, n_row)
    x is \in [0,480), y \in [0,1440)
    Unit: centimeter (cm) for calibration & pos annotation
    """
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'Wildtrack'

        self.img_shape = [1080, 1920]   # (H, W) ~ (N_row, N_col)
        self.grid_shape = [480, 1440]

        self.num_cam = 7
        self.num_frame = 2000
        self.frame_step = 5

        # world x,y actually means i,j in Wildtrack, which correspond to h,w
        self.worldcoord_from_worldgrid_matrix = np.array(worldcoord_from_worldgrid_matrix)

        self.intrinsic_matrices, \
        self.extrinsic_matrices = zip(*[self.get_intrinsic_extrinsic_matrix(cam) 
                                                                        for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {
            cam: {} for cam in range(self.num_cam)
        }
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_y = pos % 480
        grid_x = pos // 480
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_matrix = self.get_intrinsic_matrix(camera_i)
        extrinsic_matrix = self.get_extrinsic_matrix(camera_i)
        return intrinsic_matrix, extrinsic_matrix

    def get_intrinsic_matrix(self, camera_i):
        camera_fn = intrinsic_camera_matrix_filenames[camera_i]
        camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero', camera_fn)
        params_file = cv2.FileStorage(camera_path, flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = params_file.getNode('camera_matrix').mat()
        params_file.release()
        return intrinsic_matrix

    def get_extrinsic_matrix(self, camera_i):
        camera_fn = extrinsic_camera_matrix_filenames[camera_i]
        camera_path = os.path.join(self.root, 'calibrations', 'extrinsic', camera_fn)
        params_root = ET.parse(camera_path).getroot()

        rvec = params_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        tvec = params_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
        return extrinsic_matrix
