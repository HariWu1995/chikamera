"""
Notation:
    - mv:  multi-view ~  multi-camera
    - sv: single-view ~ single-camera
"""
import os
from tqdm import tqdm

import cv2
import numpy as np
import torch

from Tracker.states import TrackState
from Tracker.entity import DetectedEntity
from Tracker.PoseTracker import PoseTracker


def run_pipeline(
        calibs, 
        det_data, 
        kpt_data, 
        reid_data, 
        bbox_thresh: float = 0.69,
        **kwargs
    ):

    # NOTE: person = 1
    class_id = int(kwargs.get('class_id', 1))

    # Filter 17 COCO-keypoints from 133 MMPose-keypoints
    print('\n\nReducing #keypoints ...')
    kpt_temp = []
    for kpt in kpt_data:
        k_meta = kpt[:, :-(133 * 3)]
        k_kpts = kpt[:, -(133 * 3):].reshape(-1, 133, 3)[:, :17, :]\
                                    .reshape(-1, 17 * 3)
        kpt_temp.append(np.concatenate([k_meta, k_kpts], axis=1))
    kpt_data = kpt_temp
    
    # Calculate (global) number of frames
    max_frame = []
    for det_sv in det_data:
        if len(det_sv) > 0:
            max_frame.append(np.max(det_sv[:, 0]))
    max_frame = int(np.max(max_frame))

    # Pipeline
    tracker = PoseTracker(calibs, thresh_bbox=bbox_thresh)
    all_results = []

    print('\n\nRunning ...')
    pbar = tqdm(range(max_frame+1))
    for frame_id in pbar:
        
        tracked_entities_mv = []

        for cam_id in range(tracker.num_cam):
            tracked_entities_sv = []

            det_sv = det_data[cam_id]
            if len(det_sv) == 0:
                tracked_entities_mv.append(tracked_entities_sv)
                continue

            idx = det_sv[:, 0] == frame_id
            curr_det  =  det_data[cam_id][idx]
            curr_kpt  =  kpt_data[cam_id][idx]
            curr_reid = reid_data[cam_id][idx]

            for det, kpt, reid in zip(curr_det, curr_kpt, curr_reid):
                
                if det[-1] < bbox_thresh or \
                int(det[1]) != class_id:
                    continue

                # Filter data
                det = det[2:].astype(int)       # frame_id, class_id, x1, y1, x2, y2, score
                reid = reid[6:]                 # frame_id, x1, y1, x2, y2, score, (2048 features)
                kpt = kpt[6:].reshape(17, 3)    # frame_id, x1, y1, x2, y2, score, (17 keypoints * 3)
            
                # Add entity to track list
                entity = DetectedEntity(bbox=det, kpts=kpt, reid=reid, 
                                        cam_id=cam_id, frame_id=frame_id)
                tracked_entities_sv.append(entity)

                pbar.set_description(f"frame {frame_id} - camera {cam_id} - #entities = {len(tracked_entities_sv)}")
            
            tracked_entities_mv.append(tracked_entities_sv)

        # num_entities = np.max([len(te_sv) for te_sv in tracked_entities_mv])
        # pbar.set_description(f"frame {frame_id} - #entities = {num_entities}")

        pbar.set_description(f"frame {frame_id} - Update multi-view")
        tracker.update_mv(tracked_entities_mv, frame_id, 
                        pbar=pbar, desc=f"frame {frame_id}")

        results = tracker.output(frame_id)
        all_results.append(results)
    
    if len(all_results) == 0:
        return np.array([[]])

    all_results = np.concatenate(all_results, axis=0)
    print(all_results.shape)
    sorted_idx = np.lexsort((all_results[:, 2], 
                             all_results[:, 0]))
    all_results = np.ascontiguousarray(all_results[sorted_idx])
    return all_results

