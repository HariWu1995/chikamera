import os
import pickle
from tqdm import tqdm

import cv2
from ultralytics import YOLO 

import sys
sys.path.append('../')

from utils import measure_distance, get_center_of_bbox


FONT = cv2.FONT_HERSHEY_SIMPLEX


class PlayerTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def select_players(self, court_keypoints, person_detections):
        players = self.find_players(court_keypoints, person_detections[0])
        player_detections = []
        for person_dict in person_detections:
            player_dict = {players.index(track_id): bbox 
                                     for track_id, bbox in person_dict.items() 
                                      if track_id in players}
            player_detections.append(player_dict)
        return player_detections

    def find_players(self, court_keypoints, player_dict):

        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], 
                                  court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # Sort the distances in ascending order
        distances.sort(key = lambda x: x[1])

        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub: bool = False, stub_path: str = ''):
        player_detections = []

        if read_from_stub and \
        os.path.isfile(stub_path):
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in tqdm(frames):
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path != '':
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)[0]
        id_name_dict = results.names
        player_dict = {}
        for box in results.boxes:
            track_id = box.id.tolist()[0]
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[int(track_id)] = result
        return player_dict

    def draw_bboxes(self, video_frames, player_detections, color = (0, 0, 255)):
        output_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = [int(i) for i in bbox]
                cv2.putText(frame, f"Player ID: {track_id}",(x1, y1-10), FONT, 0.9, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            output_frames.append(frame)
        return output_frames

    