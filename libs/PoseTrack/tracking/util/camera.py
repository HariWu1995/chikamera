import os
import json

import cv2
import numpy as np
import scipy


eps = 1e-5


class Camera():

    z = 0.15 # FIXME: assumption

    def __init__(self, config_path):
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

        # index (str) in whole dataset
        self.idx = config_path.split("/")[-2][-4:]
        self.idx_int = int(self.idx)


def cross(R, V):
    h = [R[1] * V[2] - R[2] * V[1],
         R[2] * V[0] - R[0] * V[2],
         R[0] * V[1] - R[1] * V[0]]
    return h


def Point2LineDist(p_3d, pos, ray):
    return np.linalg.norm(np.cross(p_3d - pos, ray), axis=-1)


def Line2LineDist(pA, rayA, pB, rayB):
    if np.abs(np.dot(rayA, rayB)) > (1 - eps) * np.linalg.norm(rayA, axis=-1) \
                                              * np.linalg.norm(rayB, axis=-1):  # quasi vertical
        return Point2LineDist(pA, pB, rayA)
    else:
        rayCP =  np.cross(rayA, rayB)
        return np.abs((pA - pB).dot(rayCP / np.linalg.norm(rayCP, axis=-1), axis=-1))


def Line2LineDist_norm(pA, rayA, pB, rayB):
    rayCP = np.cross(rayA, rayB, axis=-1)
    rayCP_norm = np.linalg.norm(rayCP, axis=-1) + eps
    return np.abs(np.sum((pA - pB) * (rayCP / rayCP_norm[:, None]), -1))
    # return np.where(
    #     rayCP_norm < eps, 
    #     Point2LineDist(pA, pB, rayA),
    #     np.abs(np.sum((pA-pB) * (rayCP / rayCP_norm[:, None]), -1))
    # )

    # if np.abs(np.dot(rayA, rayB)) > (1 - eps):  # quasi parallel
    #     return Point2LineDist(pA, pB, rayA)
    # else:
    #     rayCP = np.cross(rayA, rayB, axis=-1)
    #     return np.abs(np.sum((pA - pB) * (rayCP / np.linalg.norm(rayCP, axis=-1)), axis=-1))
    

def epipolar_3d_score(pA, rayA, pB, rayB, alpha_epi):
    dist = Line2LineDist(pA, rayA, pB, rayB)
    return 1 - dist / alpha_epi


def epipolar_3d_score_norm(pA, rayA, pB, rayB, alpha_epi):
    dist = Line2LineDist_norm(pA, rayA, pB, rayB)
    return 1 - dist / alpha_epi


import aic_cpp

epipolar_3d_score_norm = aic_cpp.epipolar_3d_score_norm

# def epipolar_3d_score(rayA, rayB, alpha_epi):
#     dist = Line2LineDist(rayA, rayB)
#     return 1- dist / alpha_epi
