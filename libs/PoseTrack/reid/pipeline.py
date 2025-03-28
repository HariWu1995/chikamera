import os
from tqdm import tqdm

import cv2
import numpy as np
import torch


def clipping(bboxes_score, width: int = 1920, height: int = 1080):
    bboxes_score[:, 0] = np.maximum(0, bboxes_score[:, 0])
    bboxes_score[:, 1] = np.maximum(0, bboxes_score[:, 1])
    bboxes_score[:, 2] = np.minimum( width, bboxes_score[:, 2])
    bboxes_score[:, 3] = np.minimum(height, bboxes_score[:, 3])
    return bboxes_score


def run_pipeline(reid_featractor, args):
    
    vid_root = os.path.join(args.root_path, args.subset)
    det_root = os.path.join(args.root_path, "detection")
    save_root = os.path.join(args.root_path, "reid_feats")

    scenes = sorted(os.listdir(det_root))
    scenes = [s for s in scenes if s.startswith("scene")]

    for scene_id in scenes:

        print(f"\n\nReID Feature-Extracting in scene {scene_id} ...")
        det_dir = os.path.join(det_root, scene_id)
        vid_dir = os.path.join(vid_root, scene_id)
        save_dir = os.path.join(save_root, scene_id)

        cams = os.listdir(vid_dir)
        cams = sorted([c for c in cams if c.startswith("camera")])
        
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        
        pbar = tqdm()
        for cam in cams:

            save_path = os.path.join(save_dir, f"{cam}.npy") # too many features to read, so choose .npy rather than .txt
            det_path = os.path.join(det_dir, f"{cam}.txt")
            det_annot = np.loadtxt(det_path, delimiter=",")
            det_annot = np.ascontiguousarray(det_annot)

            if len(det_annot) == 0:
                all_results = np.array([])
                np.save(save_path, all_results)
                continue

            all_results = []
            frame_id = 0
            
            vid_path = os.path.join(vid_dir, cam, args.video_name)
            cap = cv2.VideoCapture(vid_path)
            width = cv2.CAP_PROP_FRAME_WIDTH
            height = cv2.CAP_PROP_FRAME_HEIGHT

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                dets = det_annot[det_annot[:, 0] == frame_id]
                bboxes_score = dets[:, 2:7]
                bboxes_score = clipping(bboxes_score, width, height)
                
                frame_id += 1
                n_bboxes = len(bboxes_score)
                pbar.set_description(f'Camera {cam} - #frame {frame_id} - #bbox {n_bboxes}')
                pbar.update()

                if len(bboxes_score) == 0:
                    continue

                with torch.no_grad():
                    feats = reid_featractor.process(frame, bboxes_score[:, :-1])

                all_results.append(feats)

            all_results = np.concatenate(all_results)
            np.save(save_path, all_results)

            cap.release()


