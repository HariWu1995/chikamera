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
import pandas as pd

from .draw import visualize


def run_viz(args):

    if args.root_path is None:
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.abspath(__file__)))
    else:
        root_path = args.root_path
    out_path = osp.join(root_path, args.outset)
    in_path = osp.join(root_path, args.subset)
    
    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    cameras = sorted(os.listdir(in_path))
    
    for cam in cameras:
        
        images_list = glob(os.path.join(in_path, cam, args.img_regext))
        if len(images_list) == 0:
            continue
        image_init = cv2.imread(images_list[0])
        height, width = image_init.shape[:2]

        bboxes_df = pd.read_csv(
                    os.path.join(out_path, f'{cam}.txt'), delimiter=',', header=None)
        bboxes_df.columns = ['frame_id','class_id','x1','y1','x2','y2','confidence']
        # bboxes_df = bboxes_df.astype({k: int for k in bboxes_df.columns[:-1]})

        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        writer = cv2.VideoWriter(
                os.path.join(out_path, f'{cam}.mp4'), fourcc, args.frame_rate, (width, height))
    
        pbar = tqdm(enumerate(images_list))

        for frame_id, frame_path in pbar:
            frame = cv2.imread(frame_path)

            frame_bboxes_df = bboxes_df[(bboxes_df['frame_id'] == frame_id) & \
                                        (bboxes_df['confidence'] > args.conf_thresh)]
            frame_bboxes = frame_bboxes_df.drop(columns=['frame_id']).values.tolist()
            frame_annot = visualize(frame, frame_bboxes)
    
            writer.write(frame_annot)

            pbar.update()
            pbar.set_description(f'Processing camera {cam} - frame {frame_id}')

        cv2.destroyAllWindows()
        writer.release()

