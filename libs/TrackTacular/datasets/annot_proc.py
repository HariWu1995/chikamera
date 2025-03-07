import os
import os.path as osp

from glob import glob
import json

import numpy as np
import torch

from ..utils.vox import VoxelUtil
from ..utils.misc import DEVICE
from ..utils.intrinsics import merge_intrinsics, split_intrinsics
from ..utils.extrinsics import merge_rt


def prepare_gt(root):

    Y, Z, X = 200, 4, 200
    bounds = (-75, 75, -75, 75, -1, 5)
    scene_centroid = (0.0, 0.0, 0.0)
    scene_centroid = torch.tensor(scene_centroid, device=DEVICE).reshape([1, 3])

    vox_util = VoxelUtil(Y, Z, X,
                        scene_centroid=scene_centroid,
                                bounds=bounds,
                            assert_cube=False)

    town_path_all = glob(osp.join(root, 'train', 'Town*-O-*'))
    for town_path in sorted(town_path_all):
        gt_fpath = osp.join(town_path, 'gt.txt')
        if osp.exists(gt_fpath):
            continue
        os.makedirs(osp.dirname(gt_fpath), exist_ok=True)

        cameras = {}
        town_name = osp.basename(town_path).split('-')[0]
        camera_file = osp.join(town_path, 'camera_name.txt')
        camera_names: list = np.loadtxt(camera_file, dtype=str).tolist()

        for _, camera_name in enumerate(camera_names):
            cam_id = int(camera_name[1:])

            calibration_path = osp.join(root, town_name, 'camera_info', f'camera_{cam_id}.txt')
            with open(calibration_path) as f:
                calibration_dict = json.load(f)

            intrinsic = torch.tensor(calibration_dict['intrinsic_matrix'])  # 3,3
            extrinsic = torch.tensor(calibration_dict['extrinsic_matrix'])  # w2c

            intrinsic = merge_intrinsics(
                        *split_intrinsics(intrinsic.unsqueeze(0))).squeeze()  # 4,4

            t = extrinsic[:3, 3:]  # 3,1
            r = extrinsic[:3, :3]  # 3,3
            t = torch.tensor([t[1], -t[2], t[0]])
            change = torch.tensor([[0, 1,  0],
                                   [0, 0, -1],
                                   [1, 0,  0]])
            r = torch.matmul(change.float(), r)

            cameras[camera_name] = {
                'rot': r,
                'tran': t,
                'intrin': intrinsic,
            }

        rots = torch.stack([c['rot'] for c in cameras.values()])
        trans = torch.stack([c['tran'] for c in cameras.values()])

        camXs_T_global = merge_rt(rots, trans.squeeze(-1))  # S 4 4
        global_T_camXs = torch.inverse(camXs_T_global)  # S 4 4
        cx, cy = global_T_camXs[:, :2, 3].mean(dim=0)

        bbox = []
        for f in sorted(glob(osp.join(town_path, camera_names[0], 'out_bbox', '*.txt'))):
            frame = osp.basename(f)[:-4]
            frame_name = int(frame)

            bbox_ids = set()
            for cam_name, cam in cameras.items():
                bbox_path = osp.join(town_path, cam_name, 'out_bbox', f'{frame}.txt')
                with open(bbox_path) as f:
                    raw_data = json.load(f)

                for v_id, v_class, box in zip(raw_data['vehicle_id'],
                                              raw_data['vehicle_class'],
                                              raw_data['world_coords']):
                    if v_class not in (0, 1, 2):
                        continue

                    if v_id in bbox_ids:
                        continue

                    bbox_ids.add(v_id)

                    xyz = torch.tensor(box[:3]).mean(1).unsqueeze(0).unsqueeze(0)
                    xyz = vox_util.Ref2Mem(xyz, Y, Z, X)
                    x = xyz[0, 0, 0].data
                    y = xyz[0, 0, 1].data
                    bbox.append(np.array([int(frame_name), x - cx, y - cy]))

        og_gt = np.stack(bbox, axis=0)
        np.savetxt(gt_fpath, og_gt, fmt='%d')


if __name__ == '__main__':
    prepare_gt('/home/ge97neh/datasets/synthehicle/')
