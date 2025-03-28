# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
from tqdm import tqdm

import cv2
import numpy as np
import torch

import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)

from pipeline import build_model, run_pipeline
    

def build_parser(
        default_cfg_path: str = "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
        default_ckpt_path: str = "./checkpoints/td_hm_hrnet.pth",
        default_data_root: str = "./temp/posetrack",
        default_data_subset: str = "",
            default_device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):

    parser = ArgumentParser("Pose Keypoint Estimation")
    
    parser.add_argument('--use_mmpose', action='store_true', help='Using MMPose Framework which are complex and difficult to install.')
    parser.add_argument('--config_path', default=default_cfg_path, help='Config file for pose-estimation model')
    parser.add_argument('--ckpt_path', default=default_ckpt_path, help='Checkpoint file for pose-estimation model')
    
    parser.add_argument("--root_path", default=default_data_root, type=str, help="path to dataset root")
    parser.add_argument("--subset", default=default_data_subset, type=str, help="subset to dataset root")
    parser.add_argument("--video_name", default='video.mp4', type=str, help="video name in root_path/subset")

    parser.add_argument('--device', default=default_device, help='Device used for inference')

    parser.add_argument('--class-id', default=[0], nargs='+', help='Category id for detection model. 0 means `human`')
    parser.add_argument('--bbox-thresh', default=0.3, type=float, help='Bounding box score threshold')
    parser.add_argument('--nms-thresh', default=0.3, type=float, help='IoU threshold for bounding box NMS')
    parser.add_argument('--kpt-thresh', default=0.3, type=float, help='Visualizing keypoint thresholds')

    parser.add_argument('--output-root', default='', type=str, help='root of the output img file.')
    parser.add_argument('--show-interval', default=0, type=int, help='Sleep seconds per frame')
    parser.add_argument('--show', '-viz', action='store_true', help='whether to show img')
    parser.add_argument('--skeleton-style', default='mmpose', type=str, 
                                            choices=['mmpose', 'openpose'], help='Skeleton style selection')
    parser.add_argument('--save-predictions', action='store_true', help='whether to save predicted results')
    parser.add_argument('--draw-heatmap', action='store_true', help='Draw heatmap predicted by the model')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument('--show-kpt-idx', action='store_true', help='Whether to show the index of keypoints')

    parser.add_argument('--alpha', default=0.8, type=float, help='The transparency of bboxes')
    parser.add_argument('--radius', default=3, type=int, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', default=1, type=int, help='Link thickness for visualization')

    return parser


if __name__ == '__main__':

    # Test MMPose
    # cfg_path = "./libs/PoseTrack/pose_estimation/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    # ckpt_path = "./checkpoints/td_hm_hrnet.pth"

    # Test DWPose
    cfg_path = None
    ckpt_path = "./checkpoints"

    parser = build_parser(
        default_cfg_path = cfg_path,
        default_ckpt_path = ckpt_path,
        default_data_root = "./temp/posetrack",
        default_data_subset = "BodyUpperRotation",
    )
    args = parser.parse_args()

    pose_estimator = build_model(args)
    run_pipeline(pose_estimator, args)


