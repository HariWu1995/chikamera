import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg11
from torchvision.models.mobilenet import mobilenet_v2
try:
    from kornia import warp_perspective
except ImportError:
    from kornia.geometry.transform import warp_perspective

from PIL import Image
import matplotlib.pyplot as plt
import cv2

from multiview_detector.models.resnet import resnet18, resnet50


class ImageProjVariant(nn.Module):

    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]

        if arch == 'vgg11':
            base = vgg11(in_channels=3 * self.num_cam + 2).features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True],
                                                in_channels=3 * self.num_cam + 2).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        pass

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        projected_imgs = []
        imgs_result = []
        for cam in range(self.num_cam):
            img_res = torch.zeros([B, 2, H, W], requires_grad=False).to('cuda:0')
            imgs_result.append(img_res)
            img_res = F.interpolate(imgs[:, cam].to('cuda:0'), self.upsample_shape, mode='bilinear')
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            img_feature = warp_perspective(img_res, proj_mat, self.reducedgrid_shape)
            if visualize:
                projected_image_rgb = img_feature[0, :].detach().cpu().numpy().transpose([1, 2, 0])
                projected_image_rgb = Image.fromarray((projected_image_rgb * 255).astype('uint8'))

                # xi = np.arange(0, self.reducedgrid_shape[0], 40)
                # yi = np.arange(0, self.reducedgrid_shape[1], 40)
                # world_grid = np.stack(np.meshgrid(yi, xi, indexing='ij')).reshape([2, -1]).transpose()
                #
                # projected_image_rgb = cv2.cvtColor(np.array(projected_image_rgb), cv2.COLOR_RGB2BGR)
                # for point in world_grid:
                #     cv2.circle(projected_image_rgb, tuple(point.astype(int)), 5, (0, 255, 0), -1)
                # projected_image_rgb = Image.fromarray(cv2.cvtColor(projected_image_rgb, cv2.COLOR_BGR2RGB))

                projected_image_rgb.save('map_grid_visualize.png')
                plt.imshow(projected_image_rgb)
                plt.show()
            projected_imgs.append(img_feature.to('cuda:0'))

        projected_imgs = torch.cat(projected_imgs + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        world_feature = self.base_pt1(projected_imgs.to('cuda:0'))
        world_feature = self.base_pt2(world_feature.to('cuda:0'))
        map_result = self.map_classifier(world_feature.to('cuda:0'))
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        return map_result, imgs_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.datasets.MVDataset import MVDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(), ])
    dataset = MVDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform, grid_reduce=1)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = ImageProjVariant(dataset)
    map_res, img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()
