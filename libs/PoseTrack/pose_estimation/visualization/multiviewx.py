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

from .draw import visualize, all_columns, kpt_coord_columns, \
                             aux_columns, kpt_score_columns


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

        keypts_df = pd.read_csv(
                    os.path.join(out_path, f'{cam}.txt'), delimiter=' ', header=None)
        keypts_df.columns = all_columns

        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        writer = cv2.VideoWriter(
                os.path.join(out_path, f'{cam}.mp4'), fourcc, args.frame_rate, (width, height))
    
        pbar = tqdm(enumerate(images_list))

        for frame_id, frame_path in pbar:
            frame = cv2.imread(frame_path)
            if frame_id > 100:
                break

            frame_keypts_df = keypts_df[(keypts_df['frame_id'] == frame_id)]

            frame_kpt_coords = frame_keypts_df[kpt_coord_columns].values.reshape(-1, 133, 2)
            frame_kpt_scores = frame_keypts_df[kpt_score_columns].values.reshape(-1, 133, 1)

            frame_annot = visualize(frame, frame_kpt_coords, frame_kpt_scores)
    
            writer.write(frame_annot)

            pbar.update()
            pbar.set_description(f'Processing camera {cam} - frame {frame_id}')

        cv2.destroyAllWindows()
        writer.release()

