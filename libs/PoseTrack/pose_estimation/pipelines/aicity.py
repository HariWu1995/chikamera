"""
Dataset Structure:

    ├── train
    │   ├── scene_001
    │   │   ├── camera_0001
    │   │   │   ├── calibration.json
    │   │   │   └── video.mp4
    │   │   ├── ...
    │   │   └── ground_truth.txt
    │   ├── scene_002
    │   └── ...
    ├── val
    │   └── ...
    └── test
        └── ...
"""
import os
import os.path as osp
from tqdm import tqdm

import cv2
import numpy as np

from .prediction import estimate_pose_kpts, output_formats


def run_pipeline(pose_estimator, args):
    
    vid_root = os.path.join(args.root_path, args.subset)
    det_root = os.path.join(args.root_path, "results_posetrack/detection")
    save_root = os.path.join(args.root_path, "results_posetrack/keypoints")

    scene_id = f"scene_{args.scene_id:03d}"
        
    det_dir = os.path.join(det_root, scene_id)
    vid_dir = os.path.join(vid_root, scene_id)
    save_dir = os.path.join(save_root, scene_id)

    cams = os.listdir(vid_dir)
    cams = sorted([c for c in cams if c.startswith("camera")])
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    
    pbar = tqdm()
    for cam in cams:

        save_path = os.path.join(save_dir, f"{cam}.txt")
        det_path = os.path.join(det_dir, f"{cam}.txt")
        det_annot = np.loadtxt(det_path, delimiter=",")

        all_results = []
        line_idx = 0
        frame_id = 0
        
        vid_path = os.path.join(vid_dir, cam, args.video_name)
        cap = cv2.VideoCapture(vid_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            dets = det_annot[det_annot[:, 0] == frame_id]
            bboxes_score = dets[:, 2:7]
            bboxes_score = bboxes_score[bboxes_score[:, -1] > args.det_thresh]
            
            n_bboxes = len(bboxes_score)
            pbar.set_description(f'Camera {cam} - #frame {frame_id} - #bbox {n_bboxes}')
            pbar.update()

            if len(bboxes_score) == 0:
                continue

            result = estimate_pose_kpts(args, frame, bboxes_score, pose_estimator)
            frames = np.ones((len(result), 1)) * frame_id
            result = np.concatenate((frames, result), axis=1)
            all_results.append(result)
            frame_id += 1

        if len(all_results) == 0:
            cap.release()
            continue

        all_results = np.concatenate(all_results, axis=0)
        np.savetxt(save_path, all_results, fmt=output_formats)
        cap.release()


