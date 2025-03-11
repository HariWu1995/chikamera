#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from glob import glob
import cv2 as cv


def save_img(dname, fn, i, frame):
    cv.imwrite('{}/{}_{}_{}.png'.format(
        out_dir, 
        os.path.basename(dname),
        os.path.basename(fn).split('.')[0], 
        i
    ), frame)


out_dir = 'preprocessed/images'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for dname in sorted(glob('original/images/set*')):
    for fn in sorted(glob('{}/*.seq'.format(dname))):
        print(fn)
        cap = cv.VideoCapture(fn)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            save_img(dname, fn, i, frame)
            i += 1
