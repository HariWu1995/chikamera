from argparse import ArgumentParser

import os
import sys
import inspect

current_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)

from preprocess import preprocess_aicity, preprocess_icsens, preprocess_multiviewx
from pipeline import run_pipeline


TITLE = "PoseTrack: Multi-Camera Person-Tracking"
DESC = f"{TITLE} with Geometric Consistency and State-aware Re-ID Correction"


def build_parser():
    parser = ArgumentParser(prog=TITLE, description=DESC)
    parser.add_argument("-sc", "--scene_id", default=-1, type=int, help='scene number')
    parser.add_argument("-r", "--root_path", default='', type=str, help="path to dataset root")
    parser.add_argument("-ss", "--subset", default='', type=str, help="subset to dataset root")
    parser.add_argument("--video_name", default='video.mp4', type=str, help="video name in root_path/subset")
    parser.add_argument("--img_regext", default='*.png', type=str, help="image extension as regular expression")
    parser.add_argument("--bbox_thresh", default=0.95, type=float, help="Detection confidence threshold")
    parser.add_argument("--reconcile", default=True, type=bool, help="Whether to reconcile data before preprocessing")
    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    #####################################
    #       Test case: MultiviewX       #
    #####################################
    # args.root_path = "F:/__Datasets__/MultiviewX"
    # args.subset = "Image_subsets"

    # print(f"\n\n[MultiviewX] Tracking in single-scene ...")
    # calibs, det_data, kpt_data, reid_data, save_path = preprocess_multiviewx(args)
    
    # all_results = run_pipeline(calibs, det_data, kpt_data, reid_data, args.bbox_thresh)
    # np.savetxt(save_path, all_results)

    #################################
    #       Test case: ICSens       #
    #################################
    args.root_path = "F:/__Datasets__/ICSens"
    args.subset = "images"

    for scene_id in range(10):
        args.scene_id = scene_id

        print(f"\n\n[ICSens] Tracking in scene {scene_id} ...")
        calibs, det_data, kpt_data, reid_data, save_path = preprocess_icsens(args)

        all_results = run_pipeline(calibs, det_data, kpt_data, reid_data, args.bbox_thresh)
        np.savetxt(save_path, all_results)

    #########################################
    #           Test case: AI City          #
    #   NOTE: missing camera calib files    #
    #########################################
    # args.root_path = "F:/__Datasets__/AI-City-Fake"
    # args.subset = "videos"

    # for scene_id in range(7):
    #     scene_id += 1
    #     args.scene_id = scene_id

    #     print(f"\n\n[AI-City] Tracking in scene {scene_id} ...")
    #     calibs, det_data, kpt_data, reid_data, save_path = preprocess_aicity(args)

    #     all_results = run_pipeline(calibs, det_data, kpt_data, reid_data, args.bbox_thresh)
    #     np.savetxt(save_path, all_results)

