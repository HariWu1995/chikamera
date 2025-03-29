"""
Dataset Structure:

    └── MultiviewX
        ├── Image_subsets
        │   ├── C1
        │   │   ├── 0000.png (1920 x 1080)
        │   │   └── ...
        │   ├── ...
        │   └── C6
        │       ├── 0000.png
        │       └── ...
        ├── calibrations
        │   ├── extrinsic 
        │   │   ├── extr_Camera1.xml
        │   │   └── ...
        │   └── intrinsic
        │       ├── intr_Camera1.xml
        │       └── ...
        ├── ...
        └── ...
"""
import os
import os.path as osp
from tqdm import tqdm

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
    out_path = osp.join(root_path, 'detection', f"scene_{args.scene_id:03d}")
    in_path = osp.join(root_path, args.subset, f"scene_{args.scene_id:03d}")
    
    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    cameras = sorted(os.listdir(in_path))
    scale = min(800 / 1080, 1440 / 1920)
    
    def preprocess_worker(img):
        return preprocess(img, predictor.img_size, predictor.rgb_mean, predictor.rgb_std)
    
    batch_size = args.batch_size
    
    for cam in cameras:
        if int(cam.split('_')[1]) < 0:
            continue
        
        frame_id = 0
        results = []
        video_path = os.path.join(in_path, cam, args.video_name)

        cap = cv2.VideoCapture(video_path)
        id_bank = []
        memory_bank = []
        carry_flag = False
        end_flag = False
    
        pbar = tqdm()
        timer = Timer()
        
        while cap.isOpened() and not end_flag:

            ret, frame = cap.read()
            if not ret:
                end_flag = True
                
            if not end_flag:
                memory_bank.append(frame)
                id_bank.append(frame_id)
            
            pbar.update()
            frame_id += 1
            frame_rate = batch_size / max(1e-5, timer.average_time)
            pbar.set_description('Processing cam {} - frame {} - {:.2f} fps'.format(cam, frame_id, frame_rate))

            if frame_id % batch_size == 0 or end_flag:
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
                        detections[:, :4] /= scale
                        detections = detections[detections[:, 4] > 0.1]

                    for det in detections:
                        x1, y1, x2, y2, score, _, _ = det
                        # x1 = max(0, x1)
                        # y1 = max(0, y1)
                        # x2 = min(1920, x2)
                        # y2 = min(1080, y2)
                        results.append([cam, id_data[out_id], 1, int(x1), int(y1), int(x2), int(y2), score])
                        
                timer.toc()

        output_file = os.path.join(out_path, f'{cam}.txt')
        with open(output_file,'w') as f:
            for cam, frame_id, clss, x1, y1, x2, y2, score in results:
                f.write(f'{frame_id},{clss},{x1},{y1},{x2},{y2},{score}\n')

