import os
import pickle
from tqdm import tqdm

import cv2
import pandas as pd

from ultralytics import YOLO 


FONT = cv2.FONT_HERSHEY_SIMPLEX


class BallTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [
                    x.get(1, []) for x in ball_positions]

        # convert the list into pandas dataframe
        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()

        ball_positions = [{1: x} for x in ball_positions_df.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self, ball_positions):

        ball_positions = [x.get(1, []) for x in ball_positions]

        # convert the list into pandas dataframe
        ball_positions_df = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        ball_positions_df['ball_hit'] = 0
        ball_positions_df['mid_y'] = (ball_positions_df['y1'] + ball_positions_df['y2'])/2
        ball_positions_df['mid_y|MA'] = ball_positions_df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        ball_positions_df['delta_y'] = ball_positions_df['mid_y|MA'].diff()

        min_change_frames = 25
        for i in range(1, len(ball_positions_df) - int(min_change_frames*1.2) ):
            negative_pos_change = ball_positions_df['delta_y'].iloc[i] > 0 and ball_positions_df['delta_y'].iloc[i+1] < 0
            positive_pos_change = ball_positions_df['delta_y'].iloc[i] < 0 and ball_positions_df['delta_y'].iloc[i+1] > 0

            if not (negative_pos_change or positive_pos_change):
                continue

            change_count = 0 
            for j in range(i+1, i+int(min_change_frames*1.2)+1):
                negative_pos_change_next = ball_positions_df['delta_y'].iloc[i] > 0 and ball_positions_df['delta_y'].iloc[j] < 0
                positive_pos_change_next = ball_positions_df['delta_y'].iloc[i] < 0 and ball_positions_df['delta_y'].iloc[j] > 0

                if negative_pos_change and negative_pos_change_next:
                    change_count += 1

                elif positive_pos_change and positive_pos_change_next:
                    change_count += 1
        
            if change_count > min_change_frames-1:
                ball_positions_df['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = ball_positions_df[ball_positions_df['ball_hit']==1].index.tolist()
        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub: bool = False, stub_path: str = ''):

        if read_from_stub and \
        os.path.isfile(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        ball_detections = []
        for frame in tqdm(frames):
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path != '':
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15, verbose=False)[0]
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    def draw_bboxes(self, video_frames, player_detections, color = (0, 255, 255)):
        output_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = [int(i) for i in bbox]
                cv2.putText(frame, f"Ball ID: {track_id}", (x1, y1-10), FONT, 0.9, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            output_frames.append(frame)
        return output_frames


    