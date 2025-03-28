import os
import os.path as osp
import argparse

from pipeline import run_pipeline


def build_parser(
        default_data_root: str = "./temp/posetrack",
        default_data_subset: str = "",
    ):
    parser = ArgumentParser("PoseTrack: Multi-Camera Person-Tracking with Geometric Consistency and State-aware Re-ID Correction")
    parser.add_argument("--root_path", default=default_data_root, type=str, help="path to dataset root")
    parser.add_argument("--subset", default=default_data_subset, type=str, help="subset to dataset root")
    return parser


if __name__ == '__main__':
    parser = build_parser(
        default_data_root = "./temp/posetrack",
        default_data_subset = "BodyUpperRotation",
    )
    args = parser.parse_args()
    run_pipeline(args)
