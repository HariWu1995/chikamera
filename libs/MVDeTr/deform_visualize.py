import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
try:
    from kornia import warp_perspective
except ImportError:
    from kornia.geometry.transform import warp_perspective

import sys
sys.path.append("libs/MVDet")

from multiview_detector.datasets import *
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.utils.image_utils import img_color_denormalize as denormalize
from multiview_detector.utils import projection


if __name__ == '__main__':

    #################################
    #       Test: Wildtrack         #
    #################################
    # ckpt_path = "./checkpoints/MVDeTr/wildtrack/MultiviewDetector.pth"
    # data_path = "F:/__Datasets__/Wildtrack"
    # database = Wildtrack(data_path)

    #################################
    #       Test: MultiviewX        #
    #################################
    ckpt_path = "./checkpoints/MVDeTr/multiviewx/MultiviewDetector.pth"
    data_path = "F:/__Datasets__/MultiviewX"
    database = MultiviewX(data_path)

    # Projection
    dataset = MVDataset(database, train=False)
    multi = 10
    freq = 40

    imgs, world_gt, imgs_gt, affine_mats, frame = dataset.__getitem__(0)

    # add breakpoint to L112@'multiview_detector/models/ops/modules/ms_deform_attn.py',
    # run 'torch.save(sampling_locations.detach().cpu(), 'sampling_locations1')' to save the sampling locations,
    # run 'torch.save(attention_weights.detach().cpu(), 'attention_weights1')' to save the attention weights
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MVDeTr(dataset, world_feat_arch='deform_trans')
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(device=device)

    (world_heatmap, _), (_, _, _) = model(imgs.unsqueeze(0).cuda(), 
                                    affine_mats.unsqueeze(0))
    plt.imshow(world_heatmap.squeeze().detach().cpu().numpy())
    plt.show()
  
    world_gt_location = torch.zeros(np.prod(dataset.Rworld_shape))
    world_gt_location[world_gt['idx'][world_gt['reg_mask']]] = 1
    world_gt_location = (
        F.interpolate(world_gt_location.view([1, 1, ] + dataset.Rworld_shape), 
                        scale_factor=0.5,
                        mode='bilinear') > 0
    ).squeeze()
    print(world_gt_location.nonzero())
    plt.imshow(world_gt_location)
    plt.show()
    
    sampling_locations1 = torch.load('sampling_locations1')
    attention_weights1 = torch.load('attention_weights1')
    # len_q = n*h*w

    world_shape = (np.array(dataset.Rworld_shape) // 2).tolist()

    # ij indexing for gt locations in range [60, 180]
    # pos1_xy = np.array([33, 62])
    # pos2_xy = np.array([39, 101])
    pos1_xy = np.array([17, 78])
    pos2_xy = np.array([50, 25])

    pos_cam_offsets = np.prod(world_shape) * np.arange(dataset.num_cam)
    pos1s = pos1_xy[0] * world_shape[1] + pos1_xy[1]  # + pos_cam_offsets
    pos2s = pos2_xy[0] * world_shape[1] + pos2_xy[1]  # + pos_cam_offsets

    denorm = denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    to_pil = T.Compose([T.Resize(dataset.img_shape), T.ToPILImage()])

    for cam in range(dataset.num_cam):
        img = to_pil(denorm(imgs)[cam])
        world_img = T.ToTensor()(img).unsqueeze(0)
        mask_img = torch.ones_like(world_img)

        world_img = warp_perspective(world_img, dataset.world_from_img[[cam]], dataset.worldgrid_shape, align_corners=False)[0]
        world_mask = warp_perspective(mask_img, dataset.world_from_img[[cam]], dataset.worldgrid_shape, align_corners=False)[0, 0].bool().numpy()
        
        world_mask_grid = np.zeros_like(world_mask, dtype=bool)
        world_mask_grid[:, ::freq] = 1
        world_mask_grid[::freq, :] = 1
        world_mask = world_mask * world_mask_grid

        world_img = np.array(T.ToPILImage()(world_img))
        world_img[world_mask] = [0, 0, 255]

        # xy indexing -> ij indexing
        # N, Len_q, n_heads, n_levels, n_points, 2
        attend_points1 = (sampling_locations1[0, pos1s, :, cam].reshape([-1, 2])[:, [1, 0]].clip(0, 1 - 1e-3) * torch.tensor(world_shape)).int().long()
        attend_points2 = (sampling_locations1[0, pos2s, :, cam].reshape([-1, 2])[:, [1, 0]].clip(0, 1 - 1e-3) * torch.tensor(world_shape)).int().long()
        
        weight1 = attention_weights1[0, pos1s, :, cam].reshape([-1])
        weight2 = attention_weights1[0, pos2s, :, cam].reshape([-1])

        weight1 = (weight1 / weight1.max()) ** 0.5
        weight2 = (weight2 / weight2.max()) ** 0.5

        world_mask_points1 = torch.zeros(world_shape)
        world_mask_points2 = torch.zeros(world_shape)
        world_mask_points_og = torch.zeros(world_shape)

        world_mask_points1[attend_points1[:, 0], attend_points1[:, 1]] = weight1
        world_mask_points2[attend_points2[:, 0], attend_points2[:, 1]] = weight2

        world_mask_points_og[pos1_xy[0], pos1_xy[1]] = 1
        world_mask_points_og[pos2_xy[0], pos2_xy[1]] = 1

        world_mask_points1   = F.interpolate(  world_mask_points1.view([1, 1] + world_shape), dataset.worldgrid_shape)
        world_mask_points2   = F.interpolate(  world_mask_points2.view([1, 1] + world_shape), dataset.worldgrid_shape)
        world_mask_points_og = F.interpolate(world_mask_points_og.view([1, 1] + world_shape), dataset.worldgrid_shape)

        idx_1 = world_mask_points1.squeeze().bool()
        idx_2 = world_mask_points2.squeeze().bool()

        world_img[idx_1] = torch.tensor([[255, 192,  0]]) * world_mask_points1.squeeze()[idx_1].view([-1, 1]) + world_img[idx_1] * (1 - world_mask_points1.squeeze()[idx_1].view([-1, 1]).numpy())
        world_img[idx_2] = torch.tensor([[  0, 176, 80]]) * world_mask_points2.squeeze()[idx_2].view([-1, 1]) + world_img[idx_2] * (1 - world_mask_points2.squeeze()[idx_2].view([-1, 1]).numpy())

        world_img[world_mask_points_og.squeeze().bool()] = [255, 0, 0]
        world_img = Image.fromarray(world_img)
        world_img.save(f'{data_path}/results_MVDeTr/deform_world_grid_C{cam+1}.png')

        plt.imshow(world_img)
        plt.show()

        world_grid = np.zeros(np.array(dataset.worldgrid_shape) * multi + np.array([1, 1]))
        world_grid[:, ::freq * multi] = 1
        world_grid[::freq * multi, :] = 1
        world_grid = np.array(np.where(world_grid)) / multi

        world_coord = dataset.base.get_worldcoord_from_worldgrid(world_grid)[[1, 0]]
        img_coord = projection.get_imagecoord_from_worldcoord(world_coord, 
                                                              dataset.base.intrinsic_matrices[cam],
                                                              dataset.base.extrinsic_matrices[cam])
        img_coord = (img_coord).astype(int)
        img_coord = img_coord[:, np.where((img_coord[0] > 0) & (img_coord[0] < 1920) & \
                                          (img_coord[1] > 0) & (img_coord[1] < 1080))[0]]
        img = np.array(img)
        img[img_coord[1], img_coord[0]] = [0, 0, 255]
        img_mask_points1   = warp_perspective(world_mask_points1  , dataset.img_from_world[[cam]], dataset.img_shape, align_corners=False).squeeze()
        img_mask_points2   = warp_perspective(world_mask_points2  , dataset.img_from_world[[cam]], dataset.img_shape, align_corners=False).squeeze()
        img_mask_points_og = warp_perspective(world_mask_points_og, dataset.img_from_world[[cam]], dataset.img_shape, align_corners=False).squeeze()

        idx_1 = img_mask_points1.bool()
        idx_2 = img_mask_points2.bool()

        img[idx_1] = torch.tensor([[255, 192,  0]]) * img_mask_points1[idx_1].view([-1, 1]) + img[idx_1] * (1 - img_mask_points1[idx_1].view([-1, 1]).numpy())
        img[idx_2] = torch.tensor([[  0, 176, 80]]) * img_mask_points2[idx_2].view([-1, 1]) + img[idx_2] * (1 - img_mask_points2[idx_2].view([-1, 1]).numpy())
        
        img[img_mask_points_og.bool()] = [255, 0, 0]
        img = Image.fromarray(img)
        img.save(f'{data_path}/results_MVDeTr/deform_img_grid_C{cam+1}.png')

        plt.imshow(img)
        plt.show()

