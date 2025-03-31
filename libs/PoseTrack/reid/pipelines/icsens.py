"""
Dataset Structure: 

    └── ICSens (stereo camera)
        ├── images
        │   ├── view1 (1920 x 1216)
        │   ├── view2 (1521 x 691)
        │   └── view3 (1920 x 1200)
        │       ├── 0000 (scene_id)
        │       ├── ...
        │       └── 0009
        │           ├── time_stamp.csv
        │           ├── left
        │           └── right
        │               ├── 000000.png
        │               └── ...
        ├── calibration
        │   ├── view1
        │   ├── view2
        │   └── view3
        │       ├── absolute.txt
        │       ├── extrinsics.txt
        │       └── intrinsics.txt
        └── ...
"""
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import itertools

import cv2
import numpy as np
import torch


def run_pipeline(reid_featractor, args):
    
    img_dir = os.path.join(args.root_path, args.subset)
    det_dir = os.path.join(args.root_path, "results_posetrack/detection", f"{args.scene_id:04d}")
    save_dir = os.path.join(args.root_path, "results_posetrack/reid_feats", f"{args.scene_id:04d}")
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    views = sorted(os.listdir(img_dir))
    
    for view, cam in itertools.product(views, ['left','right']):

        images_list = glob(os.path.join(img_dir, view, f"{args.scene_id:04d}", cam, args.img_regext))

        save_path = os.path.join(save_dir, f"{view}_{cam}.npy")
        det_path = os.path.join(det_dir, f"{view}_{cam}.txt")
        det_annot = np.loadtxt(det_path, delimiter=",")        
        all_results = []

        pbar = tqdm(enumerate(images_list))
        for frame_id, frame_path in pbar:
            frame = cv2.imread(frame_path)

            dets = det_annot[det_annot[:, 0] == frame_id]
            bboxes_score = dets[:, 2:7]
            bboxes_score = bboxes_score[bboxes_score[:, -1] > args.det_thresh]
            
            n_bboxes = len(bboxes_score)
            pbar.set_description(f'Camera {view}-{cam} - #frame {frame_id} - #bbox {n_bboxes}')
            pbar.update()

            if len(bboxes_score) == 0:
                continue

            with torch.no_grad():
                feats = reid_featractor.process(frame, bboxes_score[:, :-1])

            frames = np.ones((len(feats), 1)) * frame_id
            result = np.concatenate((frames, bboxes_score, feats), axis=1)
            all_results.append(result)

        if len(all_results) == 0:
            continue

        all_results = np.concatenate(all_results, axis=0)
        np.save(save_path, all_results)

