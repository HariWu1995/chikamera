import os
from argparse import ArgumentParser

import torch

import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)
sys.path.append("libs/FastReID")

from featractor import ReIdFeatractor
from pipeline import run_pipeline


def build_parser(
        default_ckpt_path: str = "./checkpoints/PoseTrack/aic24.pkl",
        default_data_root: str = "./temp/posetrack",
        default_data_subset: str = "",
            default_device: str = 'cuda' if torch.cuda.is_available() else 'cpu',        
    ):
    parser = ArgumentParser("ReID Feature Extraction")
    parser.add_argument('--ckpt_path', default=default_ckpt_path, help='Checkpoint file for pose-estimation model')
    parser.add_argument("--root_path", default=default_data_root, type=str, help="path to dataset root")
    parser.add_argument("--subset", default=default_data_subset, type=str, help="subset to dataset root")
    parser.add_argument('--device', default=default_device, help='Device used for inference')
    parser.add_argument("--video_name", default='video.mp4', type=str, help="video name in root_path/subset")
    # parser.add_argument("--height", default=2160, type=int, help="height of video frame")
    # parser.add_argument("--width", default=3840, type=int, help="width of video frame")
    return parser


if __name__ == '__main__':
    parser = build_parser(
        default_ckpt_path = "./checkpoints/PoseTrack/aic24.pkl",
        default_data_root = "./temp/posetrack",
        default_data_subset = "BodyUpperRotation",
    )
    args = parser.parse_args()
    
    reid_model = torch.load(args.ckpt_path, map_location='cpu').to(device=torch.device(args.device)).eval()
    featractor = ReIdFeatractor(reid_model)

    run_pipeline(featractor, args)

