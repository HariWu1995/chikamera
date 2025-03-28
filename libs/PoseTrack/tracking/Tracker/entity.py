


class DetectedEntity:

    def __init__(self, bbox, kpts, reid, cam_id, frame_id):
        self.cam_id = cam_id
        self.frame_id = frame_id
        self.bbox = bbox
        self.kpts = kpts
        self.reid = reid


