import os
from typing import Callable, Dict, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import torch
from huggingface_hub import hf_hub_download

from ezpose.body_estimation import Wholebody, resize_image
from ezpose.format import format_openpose
from ezpose.draw import draw_openpose


class DWposeDetector:

    def __init__(
            self, 
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            ckpt_dir: str = "./checkpoints",
            use_det: bool = True,
        ):

        det_model_path = os.path.join(ckpt_dir, "YOLO", "yolox_l.onnx")
        pose_model_path = os.path.join(ckpt_dir, "dw-ll_ucoco_384.onnx")

        self.use_det = use_det
        if use_det:
            if not os.path.isfile(det_model_path):
                hf_hub_download("RedHash/DWPose", "yolox_l.onnx", local_dir=ckpt_dir)
        else:
            det_model_path = None

        if not os.path.isfile(pose_model_path):
            hf_hub_download("RedHash/DWPose", "dw-ll_ucoco_384.onnx", local_dir=ckpt_dir)

        self.pipeline = Wholebody(
            model_det=det_model_path, 
            model_pose=pose_model_path, 
                device=device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        resolution: int = 512,
        output_type: str = "pil",
        bboxes = None,
        return_mmpose: bool = False,
        draw_pose: bool = False,
        **drawkwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict]:
        if type(image) != np.ndarray:
            image = np.array(image.convert("RGB"))

        image = image.copy()
        original_height, original_width, _ = image.shape

        if resolution > 0:
            image = resize_image(image, target_resolution=resolution)
            height, width, _ = image.shape
        else:
            height = original_height
            width = original_width

        candidates, scores = self.pipeline(image, bboxes, return_mmpose)

       # TODO: format & draw MMPose
        if return_mmpose:
            return candidates, scores

        pose = format_openpose(candidates, scores, width, height)

        if not draw_pose:
            return pose

        pose_image = draw_openpose(pose, height=height, width=width, **drawkwargs)
        pose_image = cv2.resize(pose_image, (original_width, original_height), cv2.INTER_LANCZOS4)

        if output_type == "pil":
            pose_image = PIL.Image.fromarray(pose_image)
        elif output_type == "np":
            pass
        else:
            raise ValueError("output_type should be 'pil' or 'np'")

        return pose_image
