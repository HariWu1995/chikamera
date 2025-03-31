import numpy as np


eps = 1e-5


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

