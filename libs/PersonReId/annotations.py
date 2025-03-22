from __future__ import absolute_import

import math
import random as rd
# import numpy as np

# from torchvision.transforms import *


class RandomErasing(object):
    """ 
    Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    
    Args:
        prob: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value. 
    """
    def __init__(self, prob = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, 
                        mean = [0.4914, 0.4822, 0.4465]):
        self.prob = prob
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if rd.uniform(0, 1) > self.prob:
            return img

        for attempt in range(100):
            height, width = img.size()[1:3]
            area = height * width
       
            target_area = rd.uniform(self.sl, self.sh) * area
            aspect_ratio = rd.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = rd.randint(0, height - h)
                y1 = rd.randint(0, width - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


class RandomGrayscaleErasing(object):
    """ 
    Randomly selects a rectangle region in an image and use grayscale image instead of its pixels.
        'Local Grayscale Transfomation' by Yunpeng Gong.
        See https://arxiv.org/pdf/2101.08533.pdf

    Args:
        prob: The probability that the Random Grayscale Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
    """
    def __init__(self, prob: float = 0.2, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """
        Args:
            img: after ToTensor() and Normalize([...]), img's type is Tensor
        """
        if rd.uniform(0, 1) > self.prob:
            return img

        height = img.size()[-2]
        width = img.size()[-1]
        area = height * width

        for _ in range(100):

            target_area = rd.uniform(self.sl, self.sh) * area
            aspect_ratio = rd.uniform(self.r1, 1 / self.r1)  # height / width

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                # tl
                x = rd.randint(0, height - h)
                y = rd.randint(0, width - w)

                # unbind channel dim
                r, g, b = img.unbind(dim=-3)

                # Weighted average method -> grayscale patch
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
                l_img = l_img.unsqueeze(dim=-3)  # rebind channel
                
                # erasing
                img[0, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[1, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[2, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]

                return img

        return img
