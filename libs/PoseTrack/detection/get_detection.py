import os
import os.path as osp

import time
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
from pipelines import run_pipeline
from utils.io import *


def init_predictor(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

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
        trt_file = 'ckpt_weight/yolox_trt.pth'
        assert osp.exists(trt_file), "TensorRT model is not found!\n Run python3 detection/trt.py first!"
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
        trt_file=trt_file,
        decoder=decoder,
        fp16=args.fp16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return predictor
    

def build_parser(
        default_exp_path: str = "detection/yolox/exps/example/mot/yolox_x_mix_det.py",
        default_ckpt_path: str = "ckpt_weight/bytetrack_x_mot17.pth.tar",
        default_data_root: str = "./temp/posetrack",
        default_data_subset: str = "",
        default_scene_number: int = 1,
    ):

    parser = argparse.ArgumentParser("YOLOX Detection")
    parser.add_argument("-expn", "--experiment-name", default=None, type=str)
    parser.add_argument("-sc", "--scene_id", default=default_scene_number, type=int, help='scene number')
    parser.add_argument("-s", "--save_result", action="store_true", help="whether to save the inference result of image/video")

    # parser.add_argument("demo", default="image", choices=['image','video','webcam'], help="demo type, eg. image, video and webcam")
    # parser.add_argument("--path", default="", type=str, help="path to images or video")
    # parser.add_argument("--camid", default=0, type=int, help="webcam demo camera id")
    # parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")    

    parser.add_argument("-v", "--video_name", default='video.mp4', type=str, help="video name in root_path/subset")
    parser.add_argument("-ss", "--subset", default=default_data_subset, type=str, help="subset to dataset root")
    parser.add_argument("-r", "--root_path", default=default_data_root, type=str, help="path to dataset root")

    parser.add_argument("-f", "--exp_file", default=default_exp_path, type=str, help="path to experiment file for description")
    parser.add_argument("-c", "--ckpt_file", default=default_ckpt_path, type=str, help="path to checkpoint file for eval")
    parser.add_argument("-trt", "--trt_file", default=default_ckpt_path, type=str, help="path to checkpoint file for eval")

    parser.add_argument("-n", "--name", default=None, type=str, help="model name")    
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--conf", default=None, type=float, help="test confidence score")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")

    # inference args
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", default=0.6, type=float, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", default=30, type=int, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", default=0.8, type=float, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", default=1.6, type=float, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', default=10, type=float, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", action="store_true", dest="fuse_score", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default="fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default="pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser


if __name__ == "__main__":

    parser = build_parser(
        default_exp_path = "./libs/YOLOX/yolox/exps/example/mot/yolox_x_mix_det.py",
        default_ckpt_path = "./checkpoints/bytetrack_x_mot17.pth.tar",
        default_data_root = "F:/__Datasets__/MultiviewX",
        default_data_subset = "Image_subsets",
        default_scene_number = 0,
    )
        
    args = parser.parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    predictor = init_predictor(exp, args)

    num_scenes = 1 # MultiviewX: 1 - ICSens: 10 (0~9)
    for scene_id in range(1):
        scene_id += 1
        print(f"\n\nDetecting in scene {scene_id} ...")
        args.scene_id = scene_id
        run_pipeline(predictor, args)

