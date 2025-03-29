import itertools
from copy import deepcopy

import cv2
import numpy as np

from ezpose.format import format_openpose
from ezpose.draw import draw_openpose


aux_columns = ['frame_id','x1','y1','x2','y2','bbox_score']
all_columns = deepcopy(aux_columns)
kpt_coord_columns = []
kpt_score_columns = []

for k in range(133):
    all_columns.extend([f'kpt{k:03d}_x',f'kpt{k:03d}_y',f'kpt{k:03d}_score'])
    kpt_coord_columns.extend([f'kpt{k:03d}_x',f'kpt{k:03d}_y'])
    kpt_score_columns.append(f'kpt{k:03d}_score')


def mmpose_to_openpose(keypoints, scores):

    keypoints_info = np.concatenate((keypoints, scores), axis=-1)

    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)

    # neck score when visualizing pred
    neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, 
                                  keypoints_info[:, 6, 2:4] > 0.3).astype(int)
    _keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]

    _keypoints_info[:, openpose_idx] = _keypoints_info[:, mmpose_idx]
    keypoints_info = _keypoints_info

    keypts = keypoints_info[..., :2]
    scores = keypoints_info[..., 2]
    return keypts, scores


def visualize(image, keypoints, scores, include_face: bool = True, 
                                        include_hands: bool = True, 
                                            opacity: float = 0.32):
    height, width, _ = image.shape

    keypoints, scores = mmpose_to_openpose(keypoints, scores)
    pose = format_openpose(keypoints, scores, width, height)

    pose_only = draw_openpose(pose, height=height, include_face=include_face, 
                                      width=width, include_hands=include_hands)
    pose_image = cv2.addWeighted(pose_only, opacity, image, 1 - opacity, 0)
    return pose_image
    
