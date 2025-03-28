"""
Notation:
    - mv:  multi-view ~  multi-camera
    - sv: single-view ~ single-camera
"""
import os
from tqdm import tqdm

import cv2
import numpy as np
import torch


from util.camera import Camera
from Tracker.PoseTracker import DetectedEntity, PoseTracker, TrackState


def load_preprocessed_data(scene_dir, use_npy: bool = False, apply_norm: bool = False):
    data = []
    cam_files = sorted(os.listdir(scene_dir))
    cam_files = sorted([f for f in cam_files if c.startswith("camera")])
    for f in cam_files:
        if use_npy:
            cam_data = np.load(os.path.join(scene_dir, f), mmap_mode='r')
        else:
            cam_data = np.loadtxt(os.path.join(scene_dir, f), delimiter=",")
        if apply_norm:
            cam_data = cam_data / np.linalg.norm(cam_data, axis=1, keepdims=True)
        data.append(cam_data)
    return data


def run_pipeline(args):
    
    vid_root = os.path.join(args.root_path, args.subset)
    det_root = os.path.join(args.root_path, "detection")
    kpt_root = os.path.join(args.root_path, "keypoints")
    reid_root = os.path.join(args.root_path, "reid_feats")
    save_root = os.path.join(args.root_path, "tracking")

    scenes = sorted(os.listdir(det_root))
    scenes = [s for s in scenes if s.startswith("scene")]

    for scene_id in scenes:

        print(f"\n\nTracking multi-camera in scene {scene_id} ...")

        vid_dir = os.path.join(vid_root, scene_id)
        det_dir = os.path.join(det_root, scene_id)
        kpt_dir = os.path.join(kpt_root, scene_id)
        reid_dir = os.path.join(reid_root, scene_id)
        save_path = os.path.join(save_root, f"{scene_id}.txt")
        
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        cams = os.listdir(vid_dir)
        cams = sorted([c for c in cams if c.startswith("camera")])

        calibs = []
        for cam in cams:
            calib_path = os.path.join(vid_dir, cam, "calibration.json")
            calibs.append(Camera(calib_path))

        det_data = load_preprocessed_data(det_dir)
        kpt_data = load_preprocessed_data(kpt_dir)
        reid_data = load_preprocessed_data(reid_dir, use_npy=True, apply_norm=True)
    
        max_frame = []
        for det_sv in det_data:
            if len(det_sv) > 0:
                max_frame.append(np.max(det_sv[:, 0]))
        max_frame = int(np.max(max_frame))

        tracker = PoseTracker(calibs)
        bbox_thresh = 0.3
        all_results = []

        pbar = tqdm(range(max_frame+1))
        for frame_id in pbar:
            
            tracked_entities_mv = []

            for v in range(tracker.num_cam):
                tracked_entities_sv = []

                det_sv = det_data[v]
                if len(det_sv) == 0:
                    tracked_entities_mv.append(tracked_entities_sv)
                    continue

                idx = det_sv[:, 0] == frame_id
                curr_det  =  det_data[v][idx]
                curr_kpt  =  kpt_data[v][idx]
                curr_reid = reid_data[v][idx]

                for det, kpt, reid in zip(curr_det, curr_pose, curr_reid):
                    if det[-1] < bbox_thresh or len(det)==0:
                        continue
                    entity = DetectedEntity(bbox = det[2:],
                                            kpts = kpt[6:].reshape(17, 3), 
                                            reid = reid, 
                                          cam_id = v, 
                                        frame_id = frame_id)
                    tracked_entities_sv.append(entity)
                tracked_entities_mv.append(tracked_entities_sv)

            pbar.set_description(f"frame {frame_id} - #entities = {np.max([len(te_sv) 
                                                                           for te_sv in tracked_entities_mv])}")
            tracker.mv_update_wo_pred(tracked_entities_mv, frame_id)
            results = tracker.output(frame_id)
            all_results += results
        
        all_results = np.concatenate(all_results, axis=0)
        sorted_idx = np.lexsort((all_results[:, 2], all_results[:, 0]))
        all_results = np.ascontiguousarray(all_results[sorted_idx])
        np.savetxt(save_path, all_results)


