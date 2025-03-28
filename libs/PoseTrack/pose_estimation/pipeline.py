import os
from tqdm import tqdm

from PIL import Image
import cv2

import numpy as np
import torch

try:
    import mmcv
    import mmdet
    from mmpose.apis import init_model as init_model_mmpose
    from mmpose.apis import inference_topdown
    IS_MMPOSE_INSTALLED = True

except (ModuleNotFoundError, ImportError) as e:
    # modify from https://github.com/reallyigor/easy_dwpose to ignore detection in inference
    from ezpose import DWposeDetector
    IS_MMPOSE_INSTALLED = False


def build_model(args):
    if args.use_mmpose:
        if not IS_MMPOSE_INSTALLED:
            raise ImportError("Fail to `import mmpose`. Please ignore `use_mmpose` to import `ezpose`.") 
        pose_estimator = init_model_mmpose(
                config = args.config_path,
            checkpoint = args.ckpt_path,
                device = args.device,
            cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap)))
        )
    else:
        pose_estimator = DWposeDetector(
            ckpt_dir = args.ckpt_path,
            use_det = False,
            device = torch.device(args.device),
        )
    return pose_estimator


def infer_mmpose(args, frame, bboxes_scores, pose_estimator):
    pose_results = inference_topdown(pose_estimator, frame, bboxes_scores[:, :4])
    records = []
    for i, result in enumerate(pose_results):
        keypts = result.pred_instances.keypoints[0]
        scores = result.pred_instances.keypoint_scores.T
        record = np.concatenate((keypts, scores), axis=1).flatten()
        records.append(record)
    records = np.array(records)
    records = np.concatenate((bboxes_scores, records), axis=1)
    return records


def infer_ezpose(args, frame, bboxes_scores, pose_estimator):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = Image.fromarray(frame)
    keypoints, scores = pose_estimator(frame, bboxes=bboxes_scores, return_mmpose=True, resolution=-1)
    scores = scores[..., np.newaxis]
    records = []
    for i in range(len(bboxes_scores)):
        record = np.concatenate((keypoints[i], scores[i]), axis=1).flatten()
        records.append(record)
    records = np.array(records)
    records = np.concatenate((bboxes_scores, records), axis=1)
    return records


def estimate_pose_kpts(args, frame, bboxes_scores, pose_estimator):
    # bboxes_scores = bboxes_scores[bboxes_scores[:, 4] > args.bbox_thresh]
    if IS_MMPOSE_INSTALLED:
        return infer_mmpose(args, frame, bboxes_scores, pose_estimator)
    else:
        return infer_ezpose(args, frame, bboxes_scores, pose_estimator)


def run_pipeline(pose_estimator, args):
    
    vid_root = os.path.join(args.root_path, args.subset)
    det_root = os.path.join(args.root_path, "detection")
    save_root = os.path.join(args.root_path, "keypoints")

    scenes = sorted(os.listdir(det_root))
    scenes = [s for s in scenes if s.startswith("scene")]

    for scene_id in scenes:
        
        print(f"\n\nPose-Estimating in scene {scene_id} ...")
        det_dir = os.path.join(det_root, scene_id)
        vid_dir = os.path.join(vid_root, scene_id)
        save_dir = os.path.join(save_root, scene_id)

        cams = os.listdir(vid_dir)
        cams = sorted([c for c in cams if c.startswith("camera")])
        
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        
        pbar = tqdm()
        for cam in cams:

            save_path = os.path.join(save_dir, f"{cam}.txt")
            det_path = os.path.join(det_dir, f"{cam}.txt")
            det_annot = np.loadtxt(det_path, delimiter=",")

            all_results = []
            line_idx = 0
            frame_id = 0
            
            vid_path = os.path.join(vid_dir, cam, args.video_name)
            cap = cv2.VideoCapture(vid_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                dets = det_annot[det_annot[:, 0] == frame_id]
                bboxes_score = dets[:, 2:7]
                
                frame_id += 1
                n_bboxes = len(bboxes_score)
                pbar.set_description(f'Camera {cam} - #frame {frame_id} - #bbox {n_bboxes}')
                pbar.update()

                if len(bboxes_score) == 0:
                    continue
    
                result = estimate_pose_kpts(args, frame, bboxes_score, pose_estimator)
                frames = np.ones((len(result), 1)) * frame_id
                result = np.concatenate((frames, result.astype(np.float32)), axis=1)
                all_results.append(result)

            all_results = np.concatenate(all_results)
            np.savetxt(save_path, all_results)

            cap.release()


