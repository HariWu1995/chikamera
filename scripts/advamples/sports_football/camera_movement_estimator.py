import os
import pickle
from tqdm import tqdm

import cv2
import numpy as np

import sys 
sys.path.append('../')

from utils import measure_distance, measure_xy_distance


FONT = cv2.FONT_HERSHEY_SIMPLEX


class CameraMovementEstimator():

    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        frame_init_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(frame_init_bw)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features,
        )

        self.display_alpha = 0.6
        self.display_color = (255, 255, 255)
        self.display_loc = [(0,0), (500, 100)]
        self.display_dx = (10, 30)
        self.display_dy = (10, 60)

    def update_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object_clss, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0],
                                         position[1] - camera_movement[1])
                    tracks[object_clss][frame_num][
                            track_id]['position_adjusted'] = position_adjusted
                    
    def get_camera_movement(self, frames, read_from_stub: bool = False, stub_path: str = ''):

        # Read the stub 
        if read_from_stub and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in tqdm(range(1, len(frames))):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, \
                _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x = 0
            camera_movement_y = 0

            for i, (new_pts, old_pts) in enumerate(zip(new_features, old_features)):
                new_features_point = new_pts.ravel()
                old_features_point = old_pts.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x,\
                    camera_movement_y = measure_xy_distance(old_features_point, new_features_point) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()
        
        if stub_path.endswith('.pkl'):
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()

            cv2.rectangle(overlay, *self.display_loc, self.display_color, -1)
            cv2.addWeighted(overlay, self.display_alpha, frame, 1-self.display_alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera ΔX: {x_movement:.2f}", self.display_dx, FONT, 1, (0,0,0), 3)
            frame = cv2.putText(frame,f"Camera ΔY: {y_movement:.2f}", self.display_dy, FONT, 1, (0,0,0), 3)

            output_frames.append(frame) 

        return output_frames