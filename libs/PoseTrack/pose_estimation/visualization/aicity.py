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
import pandas as pd

from .draw import visualize, all_columns, kpt_coord_columns, \
                             aux_columns, kpt_score_columns


def run_viz(args):

    if args.root_path is None:
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.abspath(__file__)))
    else:
        root_path = args.root_path
    out_path = osp.join(root_path, args.outset, f"scene_{args.scene_id:03d}")
    in_path = osp.join(root_path, args.subset, f"scene_{args.scene_id:03d}")
    
    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    cameras = sorted(os.listdir(in_path))
    
    for cam in cameras:
        if int(cam.split('_')[1]) < 0:
            continue

        keypts_df = pd.read_csv(
                    os.path.join(out_path, f'{cam}.txt'), delimiter=' ', header=None)
        keypts_df.columns = all_columns

        pbar = tqdm()
        writer = None
        frame_id = 0
        
        video_path = os.path.join(in_path, cam, args.video_name)
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # choose codec according to format needed
            if not writer:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                writer = cv2.VideoWriter(
                        os.path.join(out_path, f'{cam}.mp4'), fourcc, args.frame_rate, (width, height))

            frame_keypts_df = keypts_df[(keypts_df['frame_id'] == frame_id)]

            frame_kpt_coords = frame_keypts_df[kpt_coord_columns].values.reshape(-1, 133, 2)
            frame_kpt_scores = frame_keypts_df[kpt_score_columns].values.reshape(-1, 133, 1)

            frame_annot = visualize(frame, frame_kpt_coords, frame_kpt_scores)
    
            writer.write(frame_annot)

            pbar.update()
            pbar.set_description(f'Processing camera {cam} - frame {frame_id}')

            frame_id += 1

        cv2.destroyAllWindows()
        writer.release()


