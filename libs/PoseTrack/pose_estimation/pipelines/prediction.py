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


output_formats = [
    '%d',   # frame id
    '%d','%d','%d','%d','%.3e', # 4-point bbox & score
]
for k in range(133):
    output_formats.extend(['%d','%d','%.3e'])   # keypoint (x, y, score)


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


