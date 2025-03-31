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
    out_path = osp.join(root_path, args.outset, f"{args.scene_id:04d}")
    in_path = osp.join(root_path, args.subset)
    
    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    views = sorted(os.listdir(in_path))

    for view, cam in itertools.product(views, ['left','right']):

        images_list = glob(os.path.join(in_path, view, f"{args.scene_id:04d}", cam, args.img_regext))
        if len(images_list) == 0:
            continue
        image_init = cv2.imread(images_list[0])
        height, width = image_init.shape[:2]

        keypts_df = pd.read_csv(
                    os.path.join(out_path, f'{view}_{cam}.txt'), delimiter=' ', header=None)
        keypts_df.columns = all_columns

        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        writer = cv2.VideoWriter(
                os.path.join(out_path, f'{view}_{cam}.mp4'), fourcc, args.frame_rate, (width, height))
    
        pbar = tqdm(enumerate(images_list))

        for frame_id, frame_path in pbar:
            frame = cv2.imread(frame_path)
            frame_keypts_df = keypts_df[(keypts_df['frame_id'] == frame_id)]

            frame_kpt_coords = frame_keypts_df[kpt_coord_columns].values.reshape(-1, 133, 2)
            frame_kpt_scores = frame_keypts_df[kpt_score_columns].values.reshape(-1, 133, 1)

            frame_annot = visualize(frame, frame_kpt_coords, frame_kpt_scores)
            writer.write(frame_annot)

            pbar.update()
            pbar.set_description(f'Processing view {view[-1]} camera {cam} - frame {frame_id}')

        cv2.destroyAllWindows()
        writer.release()


