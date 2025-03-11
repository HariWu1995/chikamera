#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import cv2 as cv


annotations = json.load(open('preprocessed/annotations.json'))

out_dir = 'preprocessed/labels'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

img_fns = defaultdict(dict)

for fn in sorted(glob('preprocessed/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    img_fns[set_name] = defaultdict(dict)

for fn in sorted(glob('preprocessed/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    video_name = re.search('(V[0-9]+)', fn).groups()[0]
    img_fns[set_name][video_name] = []

for fn in sorted(glob('preprocessed/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    video_name = re.search('(V[0-9]+)', fn).groups()[0]
    n_frame = re.search('_([0-9]+)\.png', fn).groups()[0]
    img_fns[set_name][video_name].append((int(n_frame), fn))

n_objects = 0
for set_name in sorted(img_fns.keys()):
    for video_name in sorted(img_fns[set_name].keys()):
        print(set_name, video_name)

        for frame_i, fn in tqdm(sorted(img_fns[set_name][video_name])):
            frame_labels = []
            
            if str(frame_i) in annotations[set_name][video_name]['frames']:
                data = annotations[set_name][video_name]['frames'][str(frame_i)]
                for datum in data:
                    frame_labels.append([0, datum['id']]+[int(v) for v in datum['pos']])
                    n_objects += 1
            
            frame_anno_filepath = f"{out_dir}/{set_name}_{video_name}_{frame_i}.txt"
            with open(frame_anno_filepath, 'w') as f:
                f.write('\n'.join([' '.join([str(i) for i in l]) 
                                                    for l in frame_labels]))

print(n_objects)


