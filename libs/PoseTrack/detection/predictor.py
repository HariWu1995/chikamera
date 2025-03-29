import cv2
import numpy as np
import torch

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)
sys.path.append("libs/YOLOX")

from yolox.utils import postprocess


RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def preprocess(image, input_size, mean=None, std=None, swap=(0,3,1,2)):
    img = np.array(image)
    
    r = min(input_size[0] / img.shape[1], 
            input_size[1] / img.shape[2])

    padded_img = np.full((len(image), *input_size, 3), 114, dtype=np.uint8)
    for i in range(img.shape[0]):
        resized_sz = (int(img[i].shape[1] * r), 
                      int(img[i].shape[0] * r))
        resized_img = cv2.resize(img[i], resized_sz, interpolation=cv2.INTER_LINEAR)
        padded_img[i, : int(img.shape[1] * r), 
                      : int(img.shape[2] * r)] = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    padded_img = padded_img / np.float32(255.0)

    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
        
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class Predictor(object):

    def __init__(
        self,
        model,
        img_size,
        num_classes,
        conf_thresh=0.7,
        nms_thresh=0.45,
        batch_size=1,
        trt_file=None,
        decoder=None,
        fp16=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.img_size = img_size
        self.device = device
        self.fp16 = fp16
                
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, img_size[0], img_size[1]), device=device)
            self.model(x)
            self.model = model_trt
    
        repetition = (batch_size, img_size[0], img_size[1], 1)
        self.rgb_mean = np.tile(np.array(RGB_MEAN, dtype=np.float32).reshape(1, 1, 1, -1), repetition)
        self.rgb_std  = np.tile(np.array( RGB_STD, dtype=np.float32).reshape(1, 1, 1, -1), repetition)

    def inference(self, img, ratio):
        height, width = img.shape[1:3]

        img_info = dict()
        img_info["id"] = 0
        img_info["height"] = height
        img_info["width"] = width
        img_info["ratio"] = ratio

        # img, ratio = preprocess(img, self.img_size, self.rgb_mean, self.rgb_std)
        img = torch.from_numpy(img).float().to(self.device, non_blocking=True)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.conf_thresh, self.nms_thresh)
        return outputs, img_info

