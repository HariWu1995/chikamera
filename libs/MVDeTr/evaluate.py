import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.append("libs/MVDeTr")

import argparse
import datetime
import tqdm

import shutil
from distutils.dir_util import copy_tree

import random as rd
import numpy as np

import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from multiview_detector.trainer import PerspectiveTrainer
from multiview_detector.datasets import MVDataset, MultiviewX, Wildtrack
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool


def build_arguments(
        dataset = "wildtrack",
        default_data_path = "./datasets",
        default_ckpt_path = "./checkpoints/MultiviewDetector.pth",
    ):

    parser = argparse.ArgumentParser(description='Multiview detector')

    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)

    # Model arguments
    parser.add_argument('--reID', action='store_true')    
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thresh', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--world_feat', type=str, default='deform_trans', choices=['conv', 'trans', 'deform_conv', 'deform_trans', 'aio'])
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    # Dataset arguments
    parser.add_argument('-p', '--data_path', type=str, default=default_data_path, help="Path to dataset")
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')

    args = parser.parse_args()
    return args


def main(args):

    # seed
    if args.seed is not None:
        rd.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    if args.dataset == 'wildtrack':
        Database = Wildtrack
    elif args.dataset == 'multiviewx':
        Database = MultiviewX
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')

    database = Database(args.data_path)
    kwarg_set = dict(world_reduce=args.world_reduce, 
                       img_reduce=args.img_reduce, 
                world_kernel_size=args.world_kernel_size,
                  img_kernel_size=args.img_kernel_size)
    test_set = MVDataset(database, train=False, **kwarg_set)
    # train_set = MVDataset(database, train=True, 
    #                               dropout=args.dropcam, 
    #                       semi_supervised=args.semi_supervised, 
    #                          augmentation=args.augmentation, **kwarg_set)

    args_loader = dict(batch_size=args.batch_size, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, **args_loader)
    # train_loader = DataLoader(train_set, shuffle=True, **args_loader)

    # logging
    log_dir = os.path.join(args.data_path, "results_MVDeTr")
    if os.path.isdir(log_dir) is False:
        os.makedirs(log_dir)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MVDeTr(test_set, args.arch, 
            world_feat_arch = args.world_feat,
             bottleneck_dim = args.bottleneck_dim, 
                outfeat_dim = args.outfeat_dim, 
                   droupout = args.dropout)

    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(device=device)
    model.eval()

    # evaluation
    trainer = PerspectiveTrainer(model, log_dir, args.cls_thresh, args.alpha)

    print('\n\nEvaluate loaded model...')
    eval_fpath = os.path.join(log_dir, 'evaluation.txt')
    loss, moda = trainer.test(None, test_loader, eval_fpath, visualize=True)


if __name__ == '__main__':

    #################################
    #       Test: Wildtrack         #
    #################################
    args = build_arguments(dataset='wildtrack')
    args.data_path = "F:/__Datasets__/Wildtrack"
    args.model_path = "./checkpoints/MVDeTr/wildtrack/MultiviewDetector.pth"
    main(args)

    #################################
    #       Test: MultiviewX        #
    #################################
    args = build_arguments(dataset='multiviewx')
    args.data_path = "F:/__Datasets__/MultiviewX"
    args.model_path = "./checkpoints/MVDeTr/multiviewx/MultiviewDetector.pth"
    main(args)

