# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import torch

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)

from pipelines import (
    build_model, 
    run_pipeline_aicity, 
    run_pipeline_multiviewx, 
    run_pipeline_icsens,
)
    

def build_parser(
        default_cfg_path: str = "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
        default_ckpt_path: str = "./checkpoints/td_hm_hrnet.pth",
            default_device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):

    parser = ArgumentParser("Pose Keypoint Estimation")

    #################################
    #       Model arguments         #
    #################################
    
    parser.add_argument('--use_mmpose', action='store_true', help='Using MMPose Framework which are complex and difficult to install.')
    parser.add_argument('--config_path', default=default_cfg_path, help='Config file for pose-estimation model')
    parser.add_argument('--ckpt_path', default=default_ckpt_path, help='Checkpoint file for pose-estimation model')
    parser.add_argument('--device', default=default_device, help='Device used for inference')

    parser.add_argument('--class-id', default=[0], nargs='+', help='Category id for detection model. 0 means `human`')
    parser.add_argument('--det-thresh', default=0.3, type=float, help='Detection confidence score threshold')

    #################################
    #        Data arguments         #
    #################################
    
    parser.add_argument("-sc", "--scene_id", default=-1, type=int, help='scene number')
    parser.add_argument("-r", "--root_path", default='', type=str, help="path to dataset root")
    parser.add_argument("-ss", "--subset", default='', type=str, help="subset to dataset root")

    parser.add_argument("--video_name", default='video.mp4', type=str, help="video name in root_path/subset")
    parser.add_argument("--img_regext", default='*.png', type=str, help="image extension as regular expression")

    return parser


if __name__ == '__main__':

    # Test MMPose
    # cfg_path = "./libs/PoseTrack/pose_estimation/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    # ckpt_path = "./checkpoints/HRNet/td_hm_hrnet.pth"

    # Test DWPose
    cfg_path = None
    ckpt_path = "./checkpoints"

    parser = build_parser(
        default_cfg_path = cfg_path,
        default_ckpt_path = ckpt_path,
    )
    args = parser.parse_args()
    pose_estimator = build_model(args)

    #####################################
    #       Test case: MultiviewX       #
    #####################################
    # args.root_path = "F:/__Datasets__/MultiviewX"
    # args.subset = "Image_subsets"

    # print(f"\n\n[MultiviewX] Estimating pose-keypoints in single-scene ...")
    # run_pipeline_multiviewx(pose_estimator, args)

    #################################
    #       Test case: ICSens       #
    #################################
    # args.root_path = "F:/__Datasets__/ICSens"
    # args.subset = "images"

    # for scene_id in range(10):
    #     print(f"\n\n[ICSens] Estimating pose-keypoints in scene {scene_id} ...")
    #     args.scene_id = scene_id
    #     run_pipeline_icsens(pose_estimator, args)

    #################################
    #       Test case: AI City      #
    #################################
    args.root_path = "F:/__Datasets__/AI-City-Fake"
    args.subset = "videos"

    for scene_id in range(7):
        scene_id += 1
        print(f"\n\n[AI City] Estimating pose-keypoints in scene {scene_id} ...")
        args.scene_id = scene_id
        run_pipeline_aicity(pose_estimator, args)

