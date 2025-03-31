"""
Dataset Structure:

    ├── train
    │   ├── scene_001
    │   │   ├── camera_0001
    │   │   │   ├── calibration.json
    │   │   │   └── video.mp4
    ... ... ...

calibration.json:
{
    "camera projection matrix": [[]],
    "homography matrix": [[]],
}
"""
import os
import json

import cv2
import numpy as np
import scipy


class Camera:

    z = 0.15 # FIXME: assumption

    def __init__(self, config_path):

        # index (str) in whole dataset
        self.name = config_path.split("/")[-2][-4:]
        self.idx = int(self.name)

        with open(config_path, 'r') as file:
            config = json.load(file)

        self.project_mat = np.array(config["camera projection matrix"])
        self.project_inv = scipy.linalg.pinv(self.project_mat)

        self.homo_mat = np.array(config["homography matrix"])
        self.homo_inv = np.linalg.inv(self.homo_mat)

        self.pos = np.linalg.inv(self.project_mat[:,:-1]) @ - self.project_mat[:,-1]

        self.homo_feet = self.homo_mat.copy()
        self.homo_feet[:, -1] = self.homo_feet[:, -1] + self.project_mat[:, 2] * self.z
        self.homo_feet_inv = np.linalg.inv(self.homo_feet)

