"""
Dataset Structure:

    └── MultiviewX
        ├── Image_subsets
        │   ├── C1
        │   │   ├── 0000.png (1920 x 1080)
        │   │   └── ...
        │   ├── ...
        │   └── C6
        │       ├── 0000.png
        │       └── ...
        ├── calibrations
        │   ├── extrinsic 
        │   │   ├── extr_Camera1.xml
        │   │   └── ...
        │   └── intrinsic
        │       ├── intr_Camera1.xml
        │       └── ...
        └── ...
"""
import os
import os.path as osp
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
import torch


def run_pipeline(reid_featractor, args):
    
    img_dir = os.path.join(args.root_path, args.subset)
    det_dir = os.path.join(args.root_path, "results_posetrack/detection")
    save_dir = os.path.join(args.root_path, "results_posetrack/reid_feats")
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    cameras = sorted(os.listdir(img_dir))
    for cam in cameras:
        
        images_list = glob(os.path.join(img_dir, cam, args.img_regext))

        save_path = os.path.join(save_dir, f"{cam}.npy")
        det_path = os.path.join(det_dir, f"{cam}.txt")
        det_annot = np.loadtxt(det_path, delimiter=",")        
        all_results = []

        pbar = tqdm(enumerate(images_list))
        for frame_id, frame_path in pbar:
            frame = cv2.imread(frame_path)

            dets = det_annot[det_annot[:, 0] == frame_id]
            bboxes_score = dets[:, 2:7]
            bboxes_score = bboxes_score[bboxes_score[:, -1] > args.det_thresh]
            
            n_bboxes = len(bboxes_score)
            pbar.set_description(f'Camera {cam} - #frame {frame_id} - #bbox {n_bboxes}')
            pbar.update()

            if len(bboxes_score) == 0:
                continue

            with torch.no_grad():
                feats = reid_featractor.process(frame, bboxes_score[:, :-1])

            frames = np.ones((len(feats), 1)) * frame_id
            result = np.concatenate((frames, bboxes_score, feats), axis=1)

            all_results.append(result)

        all_results = np.concatenate(all_results, axis=0)
        np.save(save_path, all_results)

