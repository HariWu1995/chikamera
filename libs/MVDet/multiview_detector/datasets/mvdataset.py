import os
import json
from tqdm import tqdm
from PIL import Image

from scipy.sparse import coo_matrix
from scipy.stats import multivariate_normal

import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

import sys
import inspect

curr_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
repo_dir = os.path.dirname(
            os.path.dirname(curr_dir))
sys.path.append(repo_dir)

from multiview_detector.utils.projection import *


class MVDataset(VisionDataset):

    def __init__(self, base, train=True, 
                        transform=ToTensor(), target_transform=ToTensor(), reID=False, 
                        grid_reduce=4, img_reduce=4, train_ratio=0.9, force_prepare=False):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root = base.root
        self.num_cam = base.num_cam
        self.num_frame = base.num_frame
        self.img_shape = base.img_shape
        self.worldgrid_shape = base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.prepare(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_prepare:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)

        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)

        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]

        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)

    def prepare_gt(self):
        
        annot_dir = os.path.join(self.root, 'annotations_positions')
        print(f"\n\nPreparing groundtruth from {annot_dir} ...")

        og_gt = []
        for fname in tqdm(sorted(os.listdir(annot_dir))):
            frame = int(fname.split('.')[0])
            
            with open(os.path.join(annot_dir, fname)) as f:
                all_pedestrians = json.load(f)

            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1
                            and single_pedestrian['views'][cam]['xmax'] == -1
                            and single_pedestrian['views'][cam]['ymin'] == -1
                            and single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
                
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def prepare(self, frame_range):

        annot_dir = os.path.join(self.root, 'annotations_positions')
        print(f"\n\nPreparing head-foot from {annot_dir} ...")

        for fname in tqdm(sorted(os.listdir(annot_dir))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                
                with open(os.path.join(annot_dir, fname)) as f:
                    all_pedestrians = json.load(f)
                
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]
                
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])), shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])), shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def __getitem__(self, index):
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        map_gt = self.map_gt[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)

        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())

        return imgs, map_gt.float(), imgs_gt, frame

    def __len__(self):
        return len(self.map_gt.keys())


if __name__ == '__main__':

    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import itertools
    import cv2

    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord

    data_path = "F:/__Datasets__/MultiviewX"
    database = MultiviewX(data_path)
    dataset = MVDataset(database)

    # test projection
    xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
    H, W = xx.shape
    image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])

    pbar = tqdm(
        list(
            itertools.product(
                list(range(dataset.num_cam)),
                list(range(H)),
                list(range(W)),
            )
        )
    )

    world_grid_maps = {cam: None for cam in range(dataset.num_cam)}

    for cam, i, j in pbar:
        pbar.set_description(f"cam{cam}_H{i*20:04d}_W{j*20:04d}")

        world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(), 
                                                      dataset.base.intrinsic_matrices[cam],
                                                      dataset.base.extrinsic_matrices[cam])
        world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
        world_grid_map = np.zeros(dataset.worldgrid_shape)

        x, y = world_grids[i, j]
        if dataset.base.indexing == 'xy':
            if x in range(dataset.worldgrid_shape[1]) \
            and y in range(dataset.worldgrid_shape[0]):
                world_grid_map[int(y), int(x)] += 1
        else:
            if x in range(dataset.worldgrid_shape[0]) \
            and y in range(dataset.worldgrid_shape[1]):
                world_grid_map[int(x), int(y)] += 1

        world_grid_map = world_grid_map != 0
        world_grid_map = world_grid_map.astype(int) * 255
        
        if world_grid_maps[cam] is None:
            world_grid_maps[cam] = world_grid_map
        else:
            world_grid_maps[cam] += world_grid_map

        cv2.imwrite(f"./temp/MVDet/MultiviewX/cam_{cam}.png", np.clip(world_grid_maps[cam], 0, 255))
        # plt.imshow(world_grid_maps[cam])
        # plt.show()

    # world_grid_maps = {
    #         cam: cv2.imread(f"./temp/MVDet/MultiviewX/cam_{cam}.png") 
    #     for cam in range(dataset.num_cam)
    # }

    overlap_grid_map = [cam_gm[np.newaxis, ...] for cam_id, cam_gm in world_grid_maps.items()]
    overlap_grid_map = np.concatenate(overlap_grid_map, axis=0)
    overlap_grid_map = np.max(overlap_grid_map, axis=0)

    cv2.imwrite(f"./temp/MVDet/MultiviewX/overlap.png", overlap_grid_map)
    # plt.imshow(overlap_grid_map)
    # plt.show()

    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0)


