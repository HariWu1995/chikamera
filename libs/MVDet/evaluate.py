import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append("libs/MVDet")

import tqdm
import argparse
import datetime
from collections import OrderedDict

import shutil
from distutils.dir_util import copy_tree

import numpy as np

import torch
import torchvision.transforms as T

from multiview_detector.datasets import MVDataset, Wildtrack, MultiviewX
from multiview_detector.trainer import PerspectiveTrainer
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.utils.image_utils import img_color_denormalize as Denormalize
from multiview_detector.utils.logger import Logger


def build_arguments(
        dataset = "wildtrack",
        default_data_path = "./datasets",
        default_ckpt_path = "./checkpoints/MultiviewDetector.pth",
    ):
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')

    # Model arguments
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thresh', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')

    parser.add_argument('--variant', type=str, default='default', choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-ckpt', '--model_path', type=str, default=default_ckpt_path, help="Path to model checkpoint")

    # Dataset arguments
    parser.add_argument('-p', '--data_path', type=str, default=default_data_path, help="Path to dataset")
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('-j', '--num_workers', type=int, default=4)

    # Trainer arguments
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')

    # Callback arguments
    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    print(args)

    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # transformations
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = Denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    data_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    
    # dataset
    if args.dataset == 'wildtrack':
        database = Wildtrack
    elif args.dataset == 'multiviewx':
        database = MultiviewX
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    kwargs_set = dict(base=database(args.data_path), transform=data_trans, grid_reduce=4)
    test_set = MVDataset(train=False, **kwargs_set)
    # train_set = MVDataset(train=True, **kwargs_set)

    args_loader = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **args_loader)
    # train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **args_loader)

    # model
    if args.variant == 'default':
        model = PerspTransDetector(test_set, args.arch)
    elif args.variant == 'img_proj':
        model = ImageProjVariant(test_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(test_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(test_set, args.arch)
    else:
        raise Exception('no support for this variant')

    # create a new state_dict [debug] [world->map]
    resume_dict = torch.load(args.model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in resume_dict.items():
        if 'world' in k:
            new_k = k.replace('world', 'map')
        else:
            new_k = k
        new_state_dict[new_k] = v

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(new_state_dict)
    model.to(device=device)
    model.eval()

    # logging
    log_dir = os.path.join(args.data_path, "results_MVDet")
    if os.path.isdir(log_dir) is False:
        os.makedirs(log_dir)

    # loss
    criterion = GaussianMSE().to(device)

    # evaluation
    trainer = PerspectiveTrainer(model, criterion, log_dir, denormalize, 
                                 args.cls_thresh, args.alpha)

    print('\n\nEvaluate loaded model...')
    eval_fpath = os.path.join(log_dir, 'evaluation.txt')
    loss, precision_perc, \
        moda = trainer.test(test_loader, eval_fpath, test_set.gt_fpath, visualize=True)


if __name__ == '__main__':

    #################################
    #       Test: Wildtrack         #
    #################################
    # args = build_arguments(dataset='wildtrack')
    # args.data_path = "F:/__Datasets__/Wildtrack"
    # args.model_path = "./checkpoints/MVDet/wildtrack/MultiviewDetector.pth"
    # main(args)

    #################################
    #       Test: MultiviewX        #
    #################################
    args = build_arguments(dataset='multiviewx')
    args.data_path = "F:/__Datasets__/MultiviewX"
    args.model_path = "./checkpoints/MVDet/multiviewx/MultiviewDetector.pth"
    main(args)

