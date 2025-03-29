import os
import os.path as osp

import argparse
from tqdm import tqdm

import cv2
import numpy as np

from visualization import run_viz_aicity, run_viz_multiviewx, run_viz_icsens

    
def build_parser():

    parser = argparse.ArgumentParser("133-Keypoint Pose Visualization")

    parser.add_argument("-sc", "--scene_id", default=-1, type=int, help='scene number')
    parser.add_argument("-r", "--root_path", default='', type=str, help="path to dataset root")
    parser.add_argument("-ss", "--subset", default='', type=str, help="subset to dataset root")
    parser.add_argument("-os", "--outset", default='', type=str, help="subset to dataset root (output)")
    parser.add_argument("--frame_rate", default=10, type=int, help="output frame rate")
    parser.add_argument("--conf_thresh", default=0.32, type=float, help="threshold for detection confidence score")
    parser.add_argument("--img_regext", default='*.png', type=str, help="image extension as regular expression")
    parser.add_argument("--video_name", default='video.mp4', type=str, help="video name in root_path/subset")

    return parser


if __name__ == "__main__":

    parser = build_parser()
        
    args = parser.parse_args()

    #####################################
    #       Test case: MultiviewX       #
    #####################################
    args.root_path = "F:/__Datasets__/MultiviewX"
    args.subset = "Image_subsets"
    args.outset = "results_posetrack/keypoints"

    run_viz_multiviewx(args)

    #################################
    #       Test case: AI City      #
    #################################
    args.root_path = "F:/__Datasets__/AI-City-Fake"
    args.subset = "videos"
    args.outset = "results_posetrack/keypoints"

    for scene_id in range(7):
        scene_id += 1
        print(f"\n\nVisualizing in scene {scene_id} ...")
        args.scene_id = scene_id
        run_viz_aicity(args)

    #################################
    #       Test case: ICSens       #
    #################################
    args.root_path = "F:/__Datasets__/ICSens"
    args.subset = "images"
    args.outset = "results_posetrack/keypoints"

    for scene_id in range(10):
        print(f"\n\nVisualizing in scene {scene_id} ...")
        args.scene_id = scene_id
        run_viz_icsens(args)

