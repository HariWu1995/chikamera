"""
Dataset Structure: 

    └── ICSens (stereo camera)
        ├── images
        │   ├── view1 (1920 x 1216)
        │   ├── view2 (1521 x 691)
        │   └── view3 (1920 x 1200)
        │       ├── 0000 (scene_id)
        │       ├── ...
        │       └── 0009
        │           ├── time_stamp.csv
        │           ├── left
        │           └── right
        │               ├── 000000.png
        │               └── ...
        ├── calibration
        │   ├── view1
        │   ├── view2
        │   └── view3
        │       ├── absolute.txt
        │       ├── extrinsics.txt
        │       └── intrinsics.txt
        └── ...
"""
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import itertools

import cv2
import numpy as np

from utils.timer import Timer
from predictor import preprocess


def run_pipeline(predictor, args):

    if args.root_path is None:
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.abspath(__file__)))
    else:
        root_path = args.root_path
    out_path = osp.join(root_path, args.outset, f"{args.scene_id:04d}")
    in_path = osp.join(root_path, args.subset)
    
    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    views = sorted(os.listdir(in_path))
    
    def preprocess_worker(img):
        return preprocess(img, predictor.img_size, predictor.rgb_mean, predictor.rgb_std)
    
    batch_size = args.batch_size

    for view, cam in itertools.product(views, ['left','right']):

        images_list = glob(os.path.join(in_path, view, f"{args.scene_id:04d}", cam, args.img_regext))

        results = []
        id_bank = []
        memory_bank = []
        carry_flag = False
    
        pbar = tqdm(enumerate(images_list))
        timer = Timer()

        for frame_id, frame_path in pbar:
            frame = cv2.imread(frame_path)
            height, width = frame.shape[:2]

            scale = min(predictor.img_size[0] / height, 
                        predictor.img_size[1] / width)

            memory_bank.append(frame)
            id_bank.append(frame_id)

            pbar.update()
            frame_rate = batch_size / max(1e-5, timer.average_time)
            pbar.set_description('Processing camera {}-{} - frame {} - {:.2f} fps'.format(view, cam, frame_id, frame_rate))

            if frame_id % batch_size == 0:
                if memory_bank:
                    img_data = memory_bank
                    id_data = np.array(id_bank)
                    memory_bank = []
                    id_bank = []
                    carry_flag = True
                else:
                    break
            else:
                carry_flag = False
                continue
            
            if carry_flag:

                timer.tic()
                
                # Detect objects
                img_prep, ratio = preprocess_worker(img_data)
                outputs, img_info = predictor.inference(img_prep, ratio)
                img_info['memory'] = img_data

                outputs = outputs
                for out_id in range(len(outputs)):
                    out_item = outputs[out_id]
                    detections = []
                    if out_item is not None:
                        detections = out_item[:, :7].cpu().numpy()
                        detections[:, :4] /= scale  # model input size -> video size 
                        detections = detections[detections[:, 4] > 0.1]

                    for det in detections:
                        x1, y1, x2, y2, score, _, _ = det
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        results.append([cam, id_data[out_id], 1, int(x1), int(y1), int(x2), int(y2), score])
                        
                timer.toc()

        output_file = os.path.join(out_path, f'{view}_{cam}.txt')
        with open(output_file,'w') as f:
            for cam, frame_id, clss, x1, y1, x2, y2, score in results:
                f.write(f'{frame_id},{clss},{x1},{y1},{x2},{y2},{score}\n')

