import os
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm

import numpy as np
from scipy.stats import multivariate_normal

import torch
import torchvision.transforms as T
import torch.nn.functional as F

import sys
sys.path.append("libs/MVDet")

from multiview_detector.datasets import MVDataset, Wildtrack, MultiviewX
from multiview_detector.utils.image_utils import img_color_denormalize as denormalize


TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_MAP = cv2.COLORMAP_JET
COLOR = (87, 59, 233)


def create_map_kernel(map_kernel_size: int = 20, grid_reduce: int = 4):

    map_sigma = map_kernel_size // grid_reduce

    x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                       np.arange(-map_kernel_size, map_kernel_size + 1))
    pos = np.stack([x, y], axis=2)

    _map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
    _map_kernel = _map_kernel / _map_kernel.max()
    kernel_size = _map_kernel.shape[0]
    
    map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
    map_kernel[0, 0] = torch.from_numpy(_map_kernel)

    return map_kernel


def _target_transform(target, kernel):
    with torch.no_grad():
        target = F.conv2d(target, 
                          kernel.to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
    return target


def main(
        dataset_name,
        dataset_path,
    ):
    if dataset_name == 'multiviewx':
        dataset = MVDataset(MultiviewX(dataset_path), False)
    elif dataset_name == 'wildtrack':
        dataset = MVDataset(Wildtrack(dataset_path), False)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')

    grid_size = list(map(lambda x: x * 3, dataset.Rworld_shape))
    bbox_by_pos_cam = dataset.base.read_pom()

    # result_fpath = f'{dataset_path}/gt.txt'
    result_fpath = f'{dataset_path}/results_MVDeTr/evaluation.txt'
    results = np.loadtxt(result_fpath)

    denorm = denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # reshape = T.Resize(dataset.img_shape)
    reshape = T.Resize([1080//4, 1920//4])

    video_path = f'{dataset_path}/results_MVDeTr/evaluation.mp4'
    video_size = (1580, 1060)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 2, video_size)

    for index in tqdm(range(len(dataset))):
        img_comb = np.zeros([1060, 1580, 3]).astype('uint8')
        map_res = np.zeros(dataset.Rworld_shape)

        imgs, map_gt, imgs_gt, affine_mats, frame = dataset.__getitem__(index)
        imgs = reshape(denorm(imgs))

        res_map_grid = results[results[:, 0] == frame, 1:]
        for ij in res_map_grid:
            i, j = (ij / dataset.world_reduce).astype(int)
            if dataset.base.indexing == 'xy':
                map_res[j, i] = 1
            else:
                map_res[i, j] = 1

        map_kernel = create_map_kernel(dataset.world_kernel_size, 
                                       dataset.world_reduce)
        map_res = _target_transform(torch.from_numpy(map_res).unsqueeze(0).unsqueeze(0).float(), 
                                                                             map_kernel.float())
        map_res = F.interpolate(map_res, grid_size).squeeze().numpy()
        map_res = np.uint8(255 * map_res)
        map_res = cv2.applyColorMap(map_res, cv2.COLORMAP_JET)
        map_res = cv2.putText(map_res, 'Ground Plane', (0, 25), TEXT_FONT, 1, COLOR, 2, cv2.LINE_AA)

        img_comb[580 : 580 + grid_size[0], 
                 500 : 500 + grid_size[1]] = map_res

        res_posID = dataset.base.get_pos_from_worldgrid(res_map_grid.transpose())
        # gt_map_grid = map_gt[0].nonzero().cpu().numpy() * dataset.world_reduce
        # gt_posID = dataset.base.get_pos_from_worldgrid(gt_map_grid.transpose())

        for cam in range(dataset.num_cam):
            img = (imgs[cam].cpu().numpy().transpose([1, 2, 0]) * 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for posID in res_posID:
                bbox = bbox_by_pos_cam[posID][cam]
                if bbox is not None:
                    bbox = tuple(map(lambda x: int(x / 4), bbox))
                    cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)

            img = cv2.putText(img, f'Camera {cam + 1}', (0, 25), TEXT_FONT, 1, COLOR, 2, cv2.LINE_AA)
            i = cam // 3
            j = cam % 3
            img_comb[i * 290 : i * 290 + 270, 
                     j * 500 : j * 500 + 480] = img

        video.write(img_comb)

    video.release()


if __name__ == '__main__':
    main('multiviewx', "F:/__Datasets__/MultiviewX")
    main('wildtrack', "F:/__Datasets__/Wildtrack")
