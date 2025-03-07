import os
import json

from tqdm import tqdm
from operator import itemgetter

from PIL import Image
import numpy as np

import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F

from ..utils.vox import VoxelUtil
from ..utils.misc import apply_4x4, img_transform
from ..utils.misc2d import draw_umich_gaussian
from ..utils.intrinsics import merge_intrinsics, split_intrinsics, scale_intrinsics
from ..utils.extrinsics import merge_rt


class PedestrianDataset(VisionDataset):
    """
    General Class for Wildtrack, MultiviewX and other similar datasets.

    Annotations in dataset
        MultiviewX: img_shape = [1080, 1920], grid_shape = [640, 1000] 
         Wildtrack: img_shape = [1080, 1920], grid_shape = [480, 1440]
    """
    def __init__(self,
                 base,
             is_train:  bool = True,
           resolution: tuple = (160, 4, 250),
               bounds: tuple = (-500, 500, -320, 320, 0, 2),
            final_dim: tuple = (720, 1280),
           resize_lim:  list = (0.8, 1.2),
        ):
        super().__init__(base.root)
        self.base = base

        self.root = base.root
        self.num_cam = base.num_cam
        self.num_frame = base.num_frame

        self.img_shape = base.img_shape
        self.grid_shape = base.grid_shape

        self.is_train = is_train

        self.bounds = bounds
        self.resolution = resolution
        self.kernel_size = 1.5
        self.max_objects = 60
        self.img_downsample = 4

        self.data_aug_conf = {
            'final_dim': final_dim, 
            'resize_lim': resize_lim,
        }

        self.Y, self.Z, self.X = self.resolution
        self.scene_centroid = torch.tensor((0., 0., 0.)).reshape([1, 3])

        self.vox_util = VoxelUtil(self.Y, self.Z, self.X,
                                    scene_centroid=self.scene_centroid,
                                            bounds=self.bounds,
                                        assert_cube=False)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.calibration = {}

        if self.is_train:
            frame_range = range(0, int(self.num_frame * 0.9))
        else:
            frame_range = range(int(self.num_frame * 0.9), self.num_frame)
        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.prepare_annot(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        self.prepare_gt()

        self.prepare_calib()

    def prepare_calib(self):
        intrinsic = self.base.intrinsic_matrices
        extrinsic = self.base.extrinsic_matrices

        intrinsic = torch.tensor(np.stack(intrinsic, axis=0), dtype=torch.float32)  # S,3,3
        extrinsic = torch.tensor(np.stack(extrinsic, axis=0), dtype=torch.float32)

        intrinsic = merge_intrinsics(
                    *split_intrinsics(intrinsic)).squeeze()  # S,4,4
        self.calibration['intrinsic'] = intrinsic

        self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
        self.calibration['extrinsic'][:, :3] = extrinsic

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])

            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_persons = json.load(json_file)

            for person in all_persons:
                def is_in_cam(cam):
                    return not (person['views'][cam]['xmin'] == -1 \                                single_pedestrian['views'][cam]['xmax'] == -1 and
                            and person['views'][cam]['ymin'] == -1 \                                single_pedestrian['views'][cam]['xmax'] == -1 and
                            and person['views'][cam]['ymax'] == -1)
                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue

                grid_x, grid_y = self.base.get_worldgrid_from_pos(person['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))

        og_gt = np.stack(og_gt, axis=0)
        np.savetxt(self.gt_fpath, og_gt, fmt='%d')

    def prepare_annot(self, frame_range):

        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0

        pbar = tqdm(sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))))
        for fname in pbar:
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1

                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_persons = json.load(json_file)

                world_pts = []
                world_pids = []
                img_bboxs = [[] for _ in range(self.num_cam)]
                img_pids = [[] for _ in range(self.num_cam)]

                for person in all_persons:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(person['positionID']).squeeze()
                    
                    if person['personID'] not in self.pid_dict:
                        self.pid_dict[person['personID']] = len(self.pid_dict)

                    num_world_bbox += 1

                    world_pts.append((grid_x, grid_y))
                    world_pids.append(person['personID'])

                    for cam in range(self.num_cam):
                        person_bbox = itemgetter('xmin', 'ymin', 'xmax', 'ymax')(person['views'][cam])
                        if person_bbox != (-1, -1, -1, -1):
                            img_bboxs[cam].append(person_bbox)
                            img_pids[cam].append(person['personID'])
                            num_imgs_bbox += 1
                            pbar.set_description(f'#frames = {num_frame} - #world_bboxes = {num_world_bbox} - #bboxes = {num_imgs_bbox}')

                self.world_gt[frame] = (torch.tensor(world_pts, dtype=torch.float32),
                                        torch.tensor(world_pids, dtype=torch.float32))
                self.imgs_gt[frame] = {
                                cam : (torch.tensor(img_bboxs[cam]), 
                                        torch.tensor(img_pids[cam])) for cam in range(self.num_cam)
                            }

    def get_bev_gt(self, mem_pts, mem_pts_prev, pids, pids_pre):
        center     = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        offset     = torch.zeros((4, self.Y, self.X), dtype=torch.float32)
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)
        person_ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)

        prev_pts = dict(zip(pids_pre.int().tolist(), mem_pts_prev[0]))

        for pts, pid in zip(mem_pts[0], pids):
            ct = pts[:2]
            ct_int = ct.int()

            if (ct_int[0] < 0) or (ct_int[0] >= self.X) \
            or (ct_int[1] < 0) or (ct_int[1] >= self.Y):
                continue

            for c in center:
                draw_umich_gaussian(c, ct_int, self.kernel_size)

            offset[   :2, ct_int[1], ct_int[0]] = ct - ct_int
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            person_ids[:, ct_int[1], ct_int[0]] = pid

            if pid in pids_pre:
                t_off = prev_pts[pid.int().item()][:2] - ct_int
                if t_off.abs().max() > 15:
                    continue
                offset[2:, ct_int[1], ct_int[0]] = t_off

        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, sx, sy, crop):

        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)

        center     = torch.zeros((3, H, W), dtype=torch.float32)
        offset     = torch.zeros((2, H, W), dtype=torch.float32)
        size       = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        center_pts = np.stack((( xmin + xmax) / 2, ( ymin + ymax) / 2), axis=1)
        size_pts   = np.stack(((-xmin + xmax)    , (-ymin + ymax)    ), axis=1)
        foot_pts   = np.stack((( xmin + xmax) / 2,   ymin            ), axis=1)
        head_pts   = np.stack((( xmin + xmax) / 2,   ymax            ), axis=1)

        center_pts = torch.tensor(center_pts, dtype=torch.float32)
        size_pts   = torch.tensor(  size_pts, dtype=torch.float32)
        foot_pts   = torch.tensor(  foot_pts, dtype=torch.float32)
        head_pts   = torch.tensor(  head_pts, dtype=torch.float32)

        for pt_idx, (pid, wh) in enumerate(zip(img_pids, size_pts)):
            for idx, pt in enumerate(foot_pts[pt_idx]):
                if (pt[0] < 0) or (pt[0] >= W) \
                or (pt[1] < 0) or (pt[1] >= H):
                    continue
                draw_umich_gaussian(center[idx], pt.int(), self.kernel_size)

            ct_int = foot_pts[pt_idx].int()
            if (ct_int[0] < 0) or (ct_int[0] >= W) \
            or (ct_int[1] < 0) or (ct_int[1] >= H):
                continue

            valid_mask[:, ct_int[1], ct_int[0]] = 1
            person_ids[:, ct_int[1], ct_int[0]] = pid
            offset[    :, ct_int[1], ct_int[0]] = foot_pts[pt_idx] - ct_int
            size[      :, ct_int[1], ct_int[0]] = wh

        return center, offset, size, person_ids, valid_mask

    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']

        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = int(self.data_aug_conf['resize_lim'][0] * \
                              self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        # validation / test
        else:
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        return resize_dims, crop

    def get_image_data(self, frame, cameras):
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []

        for cam in cameras:
            img = Image.open(self.img_fpaths[cam][frame]).convert('RGB')
            W, H = img.size

            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            extrin = self.calibration['extrinsic'][cam]
            intrin = self.calibration['intrinsic'][cam]
            intrin = scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = merge_intrinsics(fx, fy, new_x0, new_y0)

            intrin = pix_T_cam.squeeze(0)  # 4,4

            img = img_transform(img, resize_dims, crop)
            imgs.append(F.to_tensor(img))

            intrins.append(intrin)
            extrins.append(extrin)

            img_pts, img_pids = self.imgs_gt[frame][cam]
            
            center_img, offset_img, \
                size_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids, sx, sy, crop)

            centers.append(center_img)
            offsets.append(offset_img)
            valids.append(valid_img)
            sizes.append(size_img)
            pids.append(pid_img)

        return  torch.stack(imgs), torch.stack(intrins), torch.stack(extrins), \
                torch.stack(centers), torch.stack(offsets), torch.stack(sizes), \
                torch.stack(pids), torch.stack(valids)

    def __len__(self):
        return len(self.world_gt.keys())

    def __getitem__(self, index):
        frame     = list(self.world_gt.keys())[index]
        frame_pre = list(self.world_gt.keys())[max(index - 1, 0)]

        cameras = list(range(self.num_cam))

        # images
        imgs, intrins, extrins, \
        centers_img, offsets_img, sizes_img, \
            pids_img, valids_img = self.get_image_data(frame, cameras)

        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(self.base.worldcoord_from_worldgrid_mat, dtype=torch.float32)
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2,  3] = worldcoord_from_worldgrid2d[:2, 2]
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)

        worldgrid_pts_org, world_pids    = self.world_gt[frame]
        worldgrid_pts_pre, world_pid_pre = self.world_gt[frame_pre]

        worldgrid_pts     = torch.cat((worldgrid_pts_org, torch.zeros_like(worldgrid_pts_org[:, 0:1])), dim=1).unsqueeze(0)
        worldgrid_pts_pre = torch.cat((worldgrid_pts_pre, torch.zeros_like(worldgrid_pts_pre[:, 0:1])), dim=1)

        if self.is_train:
            Rz = torch.eye(3)
            off = 0.25
            scene_center = torch.tensor([0., 0., 0.], dtype=torch.float32)
            scene_center[:2].uniform_(-off, off)

            augment = merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()

            worldgrid_T_worldcoord = torch.matmul(augment, worldgrid_T_worldcoord)
            worldgrid_pts = apply_4x4(augment.unsqueeze(0), worldgrid_pts)

        mem_pts     = self.vox_util.Ref2Mem(worldgrid_pts                 , self.Y, self.Z, self.X)
        mem_pts_pre = self.vox_util.Ref2Mem(worldgrid_pts_pre.unsqueeze(0), self.Y, self.Z, self.X)

        center_bev, valid_bev, \
           pid_bev, offset_bev = self.get_bev_gt(mem_pts, mem_pts_pre, world_pids, world_pid_pre)

        grid_gt = torch.zeros((self.max_objects, 3), dtype=torch.long)
        grid_gt[:worldgrid_pts.shape[1], :2] = worldgrid_pts_org
        grid_gt[:worldgrid_pts.shape[1], 2] = world_pids

        item = {
                  'img': imgs,                      # S,3,H,W
            'intrinsic': intrins,                   # S,4,4
            'extrinsic': extrins,                   # S,4,4
         'ref_T_global': worldgrid_T_worldcoord,    # 4,4
         'sequence_num': int(0),
              'grid_gt': grid_gt,
                'frame': frame // self.base.frame_step,
        }

        target = {
            # BEV
             'valid_bev': valid_bev,    # 1,Y,X
            'center_bev': center_bev,   # 1,Y,X
            'offset_bev': offset_bev,   # 2,Y,X
               'pid_bev': pid_bev,      # 1,Y,X

            # img
            'center_img': centers_img,  # S, 1, H/8, W/8
            'offset_img': offsets_img,  # S, 2, H/8, W/8
              'size_img': sizes_img,    # S, 2, H/8, W/8
             'valid_img': valids_img,   # S, 1, H/8, W/8
               'pid_img': pids_img,     # S, 1, H/8, W/8
        }

        return item, target
