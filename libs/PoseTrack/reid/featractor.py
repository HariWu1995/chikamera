import numpy as np

import torch
from torchvision.ops import roi_align, nms


RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


def clipping(bboxes_score, width: int = 1920, height: int = 1080):
    bboxes_score[:, 0] = np.maximum(0, bboxes_score[:, 0])
    bboxes_score[:, 1] = np.maximum(0, bboxes_score[:, 1])
    bboxes_score[:, 2] = np.minimum( width, bboxes_score[:, 2])
    bboxes_score[:, 3] = np.minimum(height, bboxes_score[:, 3])
    return bboxes_score


class ReIdFeatractor():
    """
    Re-Identification Feature Extractor
    """
    def __init__(self, model):
        self.model = model
        self.device = self.model.device
        self.mean = torch.tensor(RGB_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(RGB_STD).view(3, 1, 1)

    def extract(self, crops):
        features = self.model.backbone(crops)  # (bs, 2048, 16, 8)
        b1_feat = self.model.b1(features)
        b2_feat = self.model.b2(features)
        b3_feat = self.model.b3(features)

        b21_feat, b22_feat           = torch.chunk(b2_feat, chunks=2, dim=2)
        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, chunks=3, dim=2)

        b1_pool_feat = self.model.b1_head(b1_feat)
        b2_pool_feat = self.model.b2_head(b2_feat)
        b21_pool_feat = self.model.b21_head(b21_feat)
        b22_pool_feat = self.model.b22_head(b22_feat)
        b3_pool_feat = self.model.b3_head(b3_feat)
        b31_pool_feat = self.model.b31_head(b31_feat)
        b32_pool_feat = self.model.b32_head(b32_feat)
        b33_pool_feat = self.model.b33_head(b33_feat)

        all_feats = torch.cat([
            b1_pool_feat, 
            b2_pool_feat, 
            b3_pool_feat, 
            b21_pool_feat, 
            b22_pool_feat, 
            b31_pool_feat, 
            b32_pool_feat, 
            b33_pool_feat], dim=1)
        return all_feats
    
    def process_pad(self, frame, bboxes):
        frame = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2,0,1)
        frame = frame / 255.0
        
        frame_padded = torch.ones((3, 2160, 3840))
        frame_padded[0] = RGB_MEAN[0]
        frame_padded[1] = RGB_MEAN[1]
        frame_padded[2] = RGB_MEAN[2]

        frame_padded[:, 540:1620, 960:2880] = frame
        frame_padded.sub_(self.mean).div_(self.std)
        frame = frame_padded.unsqueeze(0)
        
        cbboxes = bboxes.copy()
        cbboxes[:, [1, 3]] += 540
        cbboxes[:, [0, 2]] += 960
        cbboxes = cbboxes.astype(np.float32)
        bboxes = torch.cat([torch.zeros(len(cbboxes), 1), torch.from_numpy(cbboxes)], dim=1)

        crops = roi_align(frame, bboxes, output_size=(384, 128)).to(self.device)
        feats = (self.extract(crops) + \
                 self.extract(crops.flip(3))).detach().cpu().numpy() / 2 # get average features with original and horizontal flip
        return feats

    def process(self, frame, bboxes):
        frame = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2,0,1)
        frame = frame / 255.0
        frame.sub_(self.mean).div_(self.std)
        frame = frame.unsqueeze(0)

        cbboxes = bboxes.copy()
        cbboxes = cbboxes.astype(np.float32)
        bboxes = torch.cat([torch.zeros(len(cbboxes), 1), torch.from_numpy(cbboxes)], dim=1)

        crops = roi_align(frame, bboxes, output_size=(384, 128)).to(self.device)
        feats = (self.extract(crops) + \
                 self.extract(crops.flip(3))) / 2 # get average features with original and horizontal flip
        return feats.detach().cpu().numpy()

