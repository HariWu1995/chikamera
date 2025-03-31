from argparse import ArgumentParser

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)
sys.path.append("libs/FastReID")

from featractor import ReIdFeatractor
from pipelines import run_pipeline_aicity, run_pipeline_multiviewx, run_pipeline_icsens


def build_parser(
        default_ckpt_path: str = "./checkpoints/PoseTrack/aic24.pkl",
    ):
    parser = ArgumentParser("ReID Feature Extraction")
    parser.add_argument('--ckpt_path', default=default_ckpt_path, help='Checkpoint file for pose-estimation model')
    parser.add_argument("--det_thresh", default=0.3, type=int, help="Detection confidence threshold to run pipeline")

    # parser.add_argument("--height", default=2160, type=int, help="height of video frame")
    # parser.add_argument("--width", default=3840, type=int, help="width of video frame")
    
    parser.add_argument("-sc", "--scene_id", default=-1, type=int, help='scene number')
    parser.add_argument("-r", "--root_path", default='', type=str, help="path to dataset root")
    parser.add_argument("-ss", "--subset", default='', type=str, help="subset to dataset root")

    parser.add_argument("--video_name", default='video.mp4', type=str, help="video name in root_path/subset")
    parser.add_argument("--img_regext", default='*.png', type=str, help="image extension as regular expression")

    return parser


if __name__ == '__main__':
    parser = build_parser(default_ckpt_path="./checkpoints/PoseTrack/aic24.pkl")
    args = parser.parse_args()
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reid_model = torch.load(args.ckpt_path, 
                            map_location='cpu').to(device=device).eval()
    featractor = ReIdFeatractor(reid_model)

    #####################################
    #       Test case: MultiviewX       #
    #####################################
    args.root_path = "F:/__Datasets__/MultiviewX"
    args.subset = "Image_subsets"

    print(f"\n\n[MultiviewX] Extracting ReID features in single-scene ...")
    run_pipeline_multiviewx(featractor, args)

    #################################
    #       Test case: AI City      #
    #################################
    args.root_path = "F:/__Datasets__/AI-City-Fake"
    args.subset = "videos"

    for scene_id in range(7):
        scene_id += 1
        print(f"\n\n[AI City] Extracting ReID features in scene {scene_id} ...")
        args.scene_id = scene_id
        run_pipeline_aicity(featractor, args)

    #################################
    #       Test case: ICSens       #
    #################################
    args.root_path = "F:/__Datasets__/ICSens"
    args.subset = "images"

    for scene_id in range(10):
        print(f"\n\n[ICSens] Extracting ReID features in scene {scene_id} ...")
        args.scene_id = scene_id
        run_pipeline_icsens(featractor, args)


