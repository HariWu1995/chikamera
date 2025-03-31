"""
Dataset Structure:

    └── MultiviewX
        ├── Image_subsets
        │   ├── C1
        │   │   ├── 0000.png (1920 x 1080)
        │   │   └── ...
        │   ├── ...
        │   └── C6
        │       ├── 0000.png
        │       └── ...
        ├── calibrations
        │   ├── extrinsic 
        │   │   ├── extr_Camera1.xml
        │   │   └── ...
        │   └── intrinsic
        │       ├── intr_Camera1.xml
        │       └── ...
        └── ...
"""
import os
from glob import glob
from tqdm import tqdm

import numpy as np

from Cameras.multiviewx import Camera
from preprocess.format import det_formats, kpt_formats, reid_formats
from preprocess.utils import load_preprocessed_file, reconcile


def list_dir(folder, ext: str, prefix: str = "C"):
    return sorted(glob(os.path.join(folder, f"{prefix}*{ext}")))


def reconcile_data_by_camera(det_file, kpt_file, reid_file):
    det_cam  = int(os.path.split( det_file)[1][1])
    kpt_cam  = int(os.path.split( kpt_file)[1][1])
    reid_cam = int(os.path.split(reid_file)[1][1])

    assert det_cam == kpt_cam == reid_cam, \
        f"{det_cam} ~ {kpt_cam} ~ {reid_cam}"

    if all([os.path.exists(f + '.rec') for f in [det_file, kpt_file, reid_file]]):
        return

    det_data  = load_preprocessed_file(det_file, delimiter=",")
    kpt_data  = load_preprocessed_file(kpt_file, delimiter=" ")
    reid_data = load_preprocessed_file(reid_file, use_numpy=True)

    det_data, kpt_data, reid_data = reconcile(det_data, kpt_data, reid_data)

    np.savetxt(det_file+'.rec', det_data, delimiter=",", fmt=det_formats)
    np.savetxt(kpt_file+'.rec', kpt_data, delimiter=" ", fmt=kpt_formats)
    with open(reid_file+'.rec', 'wb') as fwriter:
        np.save(fwriter, reid_data)


def load_preprocessed_folder(cam_files, **kwargs):
    data = []
    for cam_file in cam_files:
        cam_data = load_preprocessed_file(cam_file, **kwargs)
        data.append(cam_data)
    return data


def preprocessing(args):
    
    vid_dir = os.path.join(args.root_path, args.subset)
    det_dir = os.path.join(args.root_path, "results_posetrack/detection")
    kpt_dir = os.path.join(args.root_path, "results_posetrack/keypoints")
    reid_dir = os.path.join(args.root_path, "results_posetrack/reid_feats")
    save_dir = os.path.join(args.root_path, "results_posetrack/tracking")
    save_path = os.path.join(save_dir, "uniscene.txt")
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    det_files  = list_dir( det_dir, '.txt')
    kpt_files  = list_dir( kpt_dir, '.txt')
    reid_files = list_dir(reid_dir, '.npy')

    assert len(det_files) == len(kpt_files) == len(reid_files), \
        f"Mismatch when (detection = {len(det_files)}) " \
                    f"- (keypoints = {len(kpt_files)}) " \
                    f"- (reidfeats = {len(reid_files)})"

    #########################
    #   Reconcile data      #
    #########################
    if args.reconcile:
        print("\n\nReconciling data by camera ...")

        for det_file, kpt_file, reid_file in tqdm(zip(det_files, kpt_files, reid_files), 
                                            total=len(det_files)):
            reconcile_data_by_camera(det_file, kpt_file, reid_file)
        
        det_files  = [f + '.rec' for f in  det_files]
        kpt_files  = [f + '.rec' for f in  kpt_files]
        reid_files = [f + '.rec' for f in reid_files]

    #########################
    #   Preprocess data     #
    #########################

    det_data = load_preprocessed_folder(det_files, delimiter=",")
    kpt_data = load_preprocessed_folder(kpt_files, delimiter=" ")
    reid_data = load_preprocessed_folder(reid_files, use_numpy=True, use_memmap=True, apply_norm=True)

    calib_dir = os.path.join(args.root_path, "calibrations")
    calib_intr_path = calib_dir + "/intrinsic/intr_Camera{cam_id}.xml"
    calib_extr_path = calib_dir + "/extrinsic/extr_Camera{cam_id}.xml"

    calibs = []
    for cam_id in range(1, 7):
        calibs.append(
            Camera(in_config_path=calib_intr_path.format(cam_id=cam_id),
                   ex_config_path=calib_extr_path.format(cam_id=cam_id))
        )
    
    return calibs, det_data, kpt_data, reid_data, save_path


if __name__ == "__main__":
    pass

