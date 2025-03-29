import os
import os.path as osp

import argparse
from loguru import logger
from tqdm import tqdm

import torch
import numpy as np

import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)
sys.path.append("libs/YOLOX")

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from yolox.utils.visualize import plot_tracking

from predictor import Predictor
from pipelines import run_pipeline_aicity, run_pipeline_multiviewx, run_pipeline_icsens
from utils.io import *


def init_predictor(exp, args):

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda:0" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.conf_thresh = args.conf
    if args.nms is not None:
        exp.nms_thresh = args.nms

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt_file
        logger.info("loading checkpoint")
        ckpt_dict = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt_dict["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        assert osp.exists(args.trt_file), \
            "TensorRT model is not found!\n Run python3 detection/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model,
        exp.test_size,
        exp.num_classes,
        exp.conf_thresh,
        exp.nms_thresh,
        batch_size=args.batch_size,
            fp16=args.fp16,
        device=args.device,
        decoder=decoder,
        trt_file=trt_file,
    )

    return predictor
    

def build_parser(
        default_exp_path: str = "detection/yolox/exps/example/mot/yolox_x_mix_det.py",
        default_ckpt_path: str = "./checkpoints/PoseTrack/bytetrack_x_mot17.pth.tar",
        default_trt_path: str = "./checkpoints/PoseTrack/yolox_trt.pth",
    ):

    parser = argparse.ArgumentParser("YOLOX Detection")

    #################################
    #     Experiment arguments      #
    #################################

    parser.add_argument("-n", "--name", default=None, type=str, help="model name")
    parser.add_argument("-f", "--exp_file", default=default_exp_path, type=str, help="path to experiment file for description")

    #################################
    #       Model arguments         #
    #################################

    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    parser.add_argument("-tf", "--trt_file", default=default_trt_path, type=str, help="path to checkpoint file for eval")
    parser.add_argument("-ck", "--ckpt_file", default=default_ckpt_path, type=str, help="path to checkpoint file for eval")

    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--conf", default=None, type=float, help="test confidence score")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")

    #################################
    #        Data arguments         #
    #################################

    parser.add_argument("-sc", "--scene_id", default=-1, type=int, help='scene number')
    parser.add_argument("-r", "--root_path", default='', type=str, help="path to dataset root")
    parser.add_argument("-ss", "--subset", default='', type=str, help="subset to dataset root")
    parser.add_argument("-os", "--outset", default='', type=str, help="subset to dataset root (output)")

    parser.add_argument("--video_name", default='video.mp4', type=str, help="video name in root_path/subset")
    parser.add_argument("--img_regext", default='*.png', type=str, help="image extension as regular expression")

    return parser


if __name__ == "__main__":

    parser = build_parser(
        default_exp_path = "./libs/YOLOX/yolox/exps/example/mot/yolox_x_mix_det.py",
        default_ckpt_path = "./checkpoints/bytetrack_x_mot17.pth.tar",
    )
        
    args = parser.parse_args()
    args.ablation = False

    exp = get_exp(args.exp_file, args.name)
    predictor = init_predictor(exp, args)

    #################################
    #       Test case: AI City      #
    #################################
    args.root_path = "F:/__Datasets__/AI-City-Fake"
    args.subset = "videos"
    args.outset = "results_posetrack/detection"

    for scene_id in range(7):
        scene_id += 1
        print(f"\n\nDetecting in scene {scene_id} ...")
        args.scene_id = scene_id
        run_pipeline_aicity(predictor, args)

    #####################################
    #       Test case: MultiviewX       #
    #####################################
    # args.root_path = "F:/__Datasets__/MultiviewX"
    # args.subset = "Image_subsets"
    # args.outset = "results_posetrack/detection"

    # print(f"\n\nDetecting in single-scene MultiviewX ...")
    # run_pipeline_multiviewx(predictor, args)

    #################################
    #       Test case: ICSens       #
    #################################
    # args.root_path = "F:/__Datasets__/ICSens"
    # args.subset = "images"
    # args.outset = "results_posetrack/detection"

    # for scene_id in range(10):
    #     print(f"\n\nDetecting in scene {scene_id} ...")
    #     args.scene_id = scene_id
    #     run_pipeline_icsens(predictor, args)

