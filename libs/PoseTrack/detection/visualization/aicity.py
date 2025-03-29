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

from .draw import visualize


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

        bboxes_df = pd.read_csv(
                    os.path.join(out_path, f'{cam}.txt'), delimiter=',', header=None)
        bboxes_df.columns = ['frame_id','class_id','x1','y1','x2','y2','confidence']
        # bboxes_df = bboxes_df.astype({k: int for k in bboxes_df.columns[:-1]})

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
    
            frame_bboxes_df = bboxes_df[(bboxes_df['frame_id'] == frame_id) & \
                                        (bboxes_df['confidence'] > args.conf_thresh)]
            frame_bboxes = frame_bboxes_df.drop(columns=['frame_id']).values.tolist()
            frame_annot = visualize(frame, frame_bboxes)
    
            writer.write(frame_annot)

            pbar.update()
            pbar.set_description(f'Processing camera {cam} - frame {frame_id}')

            frame_id += 1

        cv2.destroyAllWindows()
        writer.release()


