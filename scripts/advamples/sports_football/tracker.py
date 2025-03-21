import os
import pickle
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
import supervision as sv

import sys 
sys.path.append('../')

from utils import get_center_of_bbox, get_bbox_width, get_foot_position


FONT = cv2.FONT_HERSHEY_SIMPLEX


class Tracker:

    def __init__(self, model_path: str):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

        self.display_alpha = 0.4
        self.display_color = (255, 255, 255)
        self.display_loc = (1350, 850), (1900, 970)
        self.display_attr1 = (1400, 900)
        self.display_attr2 = (1400, 950)

    def add_position_to_tracks(sekf, tracks):
        for object_clss, object_tracks in tqdm(tracks.items()):
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_clss == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_clss][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        ball_positions_df = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()

        ball_positions = [{1: {"bbox": x}} for x in ball_positions_df.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames, confidence_threshold: float = 0.1):
        batch_size = 20 
        detections = [] 
        for i in tqdm(range(0, len(frames), batch_size)):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=confidence_threshold, verbose=False)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub: bool = False, stub_path: str = ''):

        if read_from_stub and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_sv = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)

            tracks["ball"].append({})
            tracks["players"].append({})
            tracks["referees"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path.endswith('.pkl'):
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35*width)),
                            angle=0.0, startAngle=-45, endAngle=235, # open-ellipse
                            color=color, thickness=2, lineType=cv2.LINE_4)

        if track_id is None:
            return frame

        rect_width = 40
        rect_height = 20
        rect_padding = 15

        x1_rect = int(x_center - rect_width // 2)
        x2_rect = int(x_center + rect_width // 2)
        y1_rect = int((y2- rect_height // 2) + rect_padding)
        y2_rect = int((y2+ rect_height // 2) + rect_padding)

        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

        # if track_id is None:
        #     return frame
        
        y1_text = y1_rect + 15
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -=10
        x1_text = int(x1_text)
        y1_text = int(y1_text)
        
        cv2.putText(frame, f"{track_id}", (x1_text, y1_text), FONT, 0.6, (0,0,0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,    y   ],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, *self.display_loc, self.display_color, -1)
        cv2.addWeighted(overlay, self.display_alpha, frame, 1 - self.display_alpha, 0, frame)

        # Get the number of time each team had ball control
        ball_control_nframes = team_ball_control[:frame_num+1]
        team_1_nframes = ball_control_nframes[ball_control_nframes==1].shape[0]
        team_2_nframes = ball_control_nframes[ball_control_nframes==2].shape[0]
        team_1_perc = team_1_nframes / (team_1_nframes + team_2_nframes) * 100
        team_2_perc = team_2_nframes / (team_1_nframes + team_2_nframes) * 100

        attr = 'Ball Possession'
        cv2.putText(frame, f"Team 1 {attr}: {team_1_perc:.2f}%", self.display_attr1, FONT, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 {attr}: {team_2_perc:.2f}%", self.display_attr2, FONT, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        out_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            ball_dict = tracks["ball"][frame_num]
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            out_frames.append(frame)

        return out_frames