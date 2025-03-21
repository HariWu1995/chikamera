import cv2
import numpy as np

import sys
sys.path.append('../')

import const
from utils import (
    convert_meters_to_pixels, convert_pixels_to_meters,
    measure_xy_distance, measure_distance,
    get_height_of_bbox, get_foot_position,
    get_center_of_bbox, get_closest_keypoint_index,
)


class CourtAnnotator():

    def __init__(self, frame, alpha: float = 0.5):
        self.alpha = alpha

        self.court_keypoints = [0] * 28       # 28 keypoints
        self.court_width = 250
        self.court_height = 500
        self.court_padding = 20
        self.padding = 50
        self.lines = [
            ( 0,  2),
            ( 4,  5),
            ( 6,  7),
            ( 1,  3),
            ( 0,  1),
            ( 8,  9),
            (10, 11),
            (10, 11),
            ( 2,  3)
        ]

        self.start_x, self.end_x = None, None
        self.start_y, self.end_y = None, None

        self.court_start_x, self.court_end_x = None, None
        self.court_start_y, self.court_end_y = None, None

        self.set_background_position(frame)     # update start_x, end_x, start_y, end_y
        self.set_court_positions()              # update court_width, court_start_x, court_end_x, court_start_y, court_end_y
        self.set_court_keypoints()              # update court_keypoints

    def convert(self, meters):
        return convert_meters_to_pixels(meters, const.DOUBLE_LINE_WIDTH, self.court_width)

    def set_court_keypoints(self):
        court_keypoints = [0] * 28

        # point 0 
        court_keypoints[0] = self.court_start_x
        court_keypoints[1] = self.court_start_y

        # point 1
        court_keypoints[2] = self.court_end_x
        court_keypoints[3] = self.court_start_y

        # point 2
        court_keypoints[4] = self.court_start_x
        court_keypoints[5] = self.court_start_y + self.convert(const.HALF_COURT_LINE_HEIGHT * 2)

        # point 3
        court_keypoints[6] = court_keypoints[0] + self.court_width
        court_keypoints[7] = court_keypoints[5] 

        # point 4
        court_keypoints[8] = court_keypoints[0] +  self.convert(const.DOUBLE_ALLY_DIFFERENCE)
        court_keypoints[9] = court_keypoints[1] 

        # point 5
        court_keypoints[10] = court_keypoints[4] + self.convert(const.DOUBLE_ALLY_DIFFERENCE)
        court_keypoints[11] = court_keypoints[5] 

        # point 6
        court_keypoints[12] = court_keypoints[2] - self.convert(const.DOUBLE_ALLY_DIFFERENCE)
        court_keypoints[13] = court_keypoints[3] 

        # point 7
        court_keypoints[14] = court_keypoints[6] - self.convert(const.DOUBLE_ALLY_DIFFERENCE)
        court_keypoints[15] = court_keypoints[7] 

        # point 8
        court_keypoints[16] = court_keypoints[8] 
        court_keypoints[17] = court_keypoints[9] + self.convert(const.NO_MANS_LAND_HEIGHT)

        # point 9
        court_keypoints[18] = court_keypoints[16] + self.convert(const.SINGLE_LINE_WIDTH)
        court_keypoints[19] = court_keypoints[17] 

        # point 10
        court_keypoints[20] = court_keypoints[10] 
        court_keypoints[21] = court_keypoints[11] - self.convert(const.NO_MANS_LAND_HEIGHT)

        # point 11
        court_keypoints[22] = court_keypoints[20] +  self.convert(const.SINGLE_LINE_WIDTH)
        court_keypoints[23] = court_keypoints[21] 

        # point 12
        court_keypoints[24] = (court_keypoints[16] + court_keypoints[18]) // 2
        court_keypoints[25] = court_keypoints[17] 
        
        # point 13
        court_keypoints[26] = (court_keypoints[20] + court_keypoints[22]) // 2
        court_keypoints[27] = court_keypoints[21] 

        self.court_keypoints = court_keypoints

    def set_court_positions(self):
        self.court_start_x = int(self.start_x + self.court_padding)
        self.court_start_y = int(self.start_y + self.court_padding)
        self.court_end_x = int(self.end_x - self.court_padding)
        self.court_end_y = int(self.end_y - self.court_padding)
        self.court_width = self.court_end_x - self.court_start_x

    def set_background_position(self, frame):
        frame = frame.copy()
        self.end_x =   frame.shape[1]  - self.padding
        self.end_y = self.court_height + self.padding
        self.start_x = self.end_x - self.court_width
        self.start_y = self.end_y - self.court_height

    def draw_court(self, frame):
        for i in range(0, len(self.court_keypoints), 2):
            x = int(self.court_keypoints[i])
            y = int(self.court_keypoints[i+1])
            cv2.circle(frame, (x,y), 5, (0,0,255), -1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.court_keypoints[line[0]*2]), int(self.court_keypoints[line[0]*2+1]))
            end_point   = (int(self.court_keypoints[line[1]*2]), int(self.court_keypoints[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (int(self.court_keypoints[0]), int((self.court_keypoints[1] + self.court_keypoints[5]) // 2))
        net_end_point   = (int(self.court_keypoints[2]), int((self.court_keypoints[1] + self.court_keypoints[5]) // 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background(self, frame):
        temp = np.zeros_like(frame, np.uint8)

        # Draw the rectangle
        cv2.rectangle(temp, (self.start_x, self.start_y), 
                              (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        mask = temp.astype(bool)
        out[mask] = cv2.addWeighted(frame, self.alpha, temp, 1 - self.alpha, 0)[mask]

        return out

    def draw_multi(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def draw_points(self, frames, postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames

    def get_start_point_of_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_court(self):
        return self.court_width

    def get_court_keypoints(self):
        return self.court_keypoints

    def get_court_coordinates(self, object_position,
                                    closest_keypoint, 
                                    closest_keypoint_index, 
                                    player_height_in_pixels,
                                    player_height_in_meters):
        
        dist_from_kpt_x_pixels, \
        dist_from_kpt_y_pixels = measure_xy_distance(object_position, closest_keypoint)

        # Conver pixels to meters
        dist_from_kpt_x_meters = convert_pixels_to_meters(dist_from_kpt_x_pixels,
                                                          player_height_in_meters,
                                                          player_height_in_pixels)

        dist_from_kpt_y_meters = convert_pixels_to_meters(dist_from_kpt_y_pixels,
                                                          player_height_in_meters,
                                                          player_height_in_pixels)
        
        # Convert to court coordinates
        court_x_dist_pixels = self.convert(dist_from_kpt_x_meters)
        court_y_dist_pixels = self.convert(dist_from_kpt_y_meters)

        # Estimate player position
        closest_coourt_keypoint = (self.court_keypoints[closest_keypoint_index*2],
                                   self.court_keypoints[closest_keypoint_index*2+1])

        court_player_position = (closest_coourt_keypoint[0] + court_x_dist_pixels,
                                 closest_coourt_keypoint[1] + court_y_dist_pixels)

        return  court_player_position

    def transform_bboxes(self, player_boxes, ball_boxes, court_keypoints):
        player_heights = {
            0: const.PLAYER_1_HEIGHT_METERS,
            1: const.PLAYER_2_HEIGHT_METERS,
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_bbox = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_bbox)
            ball_to_player_id = lambda x: measure_distance(ball_position, 
                                                        get_center_of_bbox(player_bbox[x]))
            closest_player_id_to_ball = min(player_bbox.keys(), key=ball_to_player_id)

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                if player_id not in player_heights.keys():
                    continue
                foot_position = get_foot_position(bbox)

                # Get The closest keypoint in pixels
                closest_keypoint_index = get_closest_keypoint_index(foot_position, court_keypoints, [0,2,12,13])
                closest_keypoint = (court_keypoints[closest_keypoint_index*2], 
                                    court_keypoints[closest_keypoint_index*2+1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) 
                                                                        for i in range (frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                court_player_position = self.get_court_coordinates(foot_position,
                                                                    closest_keypoint, 
                                                                    closest_keypoint_index, 
                                                                    max_player_height_in_pixels,
                                                                    player_heights[player_id])
                
                output_player_bboxes_dict[player_id] = court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_keypoint_index = get_closest_keypoint_index(ball_position, court_keypoints, [0,2,12,13])
                    closest_keypoint = (court_keypoints[closest_keypoint_index*2], 
                                        court_keypoints[closest_keypoint_index*2+1])
                    
                    court_player_position = self.get_court_coordinates(ball_position,
                                                                        closest_keypoint, 
                                                                        closest_keypoint_index, 
                                                                        max_player_height_in_pixels,
                                                                        player_heights[player_id])
                    output_ball_boxes.append({1: court_player_position})
    
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes

