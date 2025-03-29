import numpy as np
from onnxruntime import InferenceSession as Session

from .detector import inference_detector
from .pose import inference_pose


class Wholebody:
    """
    detect human pose by dwpose
    """
    def __init__(self, model_pose, model_det=None, device="cpu"):
        device = str(device)

        if device == "cpu":
            providers = ["CPUExecutionProvider"]
            provider_options = None
        else:
            providers = ["CUDAExecutionProvider"]
            if ":" in device:
                gpu_id = int(device.split(":")[1])
                provider_options = [{"device_id": gpu_id}]
            else:
                provider_options = [{"device_id": 0}]

        provider_config = dict(providers=providers, provider_options=provider_options)

        if not model_det:
            self.session_det = None
        else:
            self.session_det = Session( path_or_bytes=model_det, **provider_config)
        self.session_pose = Session(path_or_bytes=model_pose, **provider_config)

    def __call__(self, oriImg, bboxes=None, return_mmpose: bool = False):
        """
        call to process dwpose-detect

        Args:
            oriImg (np.ndarray): detected image
        """
        assert (self.session_det is not None or bboxes is not None), \
            "Detection model must be initiated or bboxes must be provided in inference."

        if self.session_det is not None:
            bboxes = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, bboxes, oriImg)
        
        if return_mmpose:
            return keypoints, scores

        #############################
        #   Format to OpenPose      #
        #############################

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)

        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, 
                                      keypoints_info[:, 6, 2:4] > 0.3).astype(int)

        _keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        
        _keypoints_info[:, openpose_idx] = _keypoints_info[:, mmpose_idx]
        keypoints_info = _keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        return keypoints, scores
