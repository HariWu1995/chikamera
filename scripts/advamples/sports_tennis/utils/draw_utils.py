import numpy as np
import cv2


FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_player_stats(video_frames, player_stats):

    for index, row in player_stats.iterrows():

        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        frame = video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width = 350
        height = 230
        color = (255, 255, 255)
        color_bg = (0, 0, 0)
        alpha_bg = 0.5

        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 500
        end_x = start_x + width
        end_y = start_y + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), color_bg, -1)
        cv2.addWeighted(overlay, alpha_bg, frame, 1 - alpha_bg, 0, frame)

        text = "     Player 1     Player 2"
        frame = cv2.putText(frame, text, (start_x+80, start_y+30), FONT, 0.6, color, 2)
        
        text = "Shot Speed"
        frame = cv2.putText(frame, text, (start_x+10, start_y+80), FONT, 0.45, color, 1)

        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        frame = cv2.putText(frame, text, (start_x+130, start_y+80), FONT, 0.5, color, 2)

        text = "Player Speed"
        frame = cv2.putText(frame, text, (start_x+10, start_y+120), FONT, 0.45, color, 1)
        
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        frame = cv2.putText(frame, text, (start_x+130, start_y+120), FONT, 0.5, color, 2)
        
        text = "avg. S. Speed"
        frame = cv2.putText(frame, text, (start_x+10, start_y+160), FONT, 0.45, color, 1)

        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        frame = cv2.putText(frame, text, (start_x+130, start_y+160), FONT, 0.5, color, 2)
        
        text = "avg. P. Speed"
        frame = cv2.putText(frame, text, (start_x+10, start_y+200), FONT, 0.45, color, 1)

        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        frame = cv2.putText(frame, text, (start_x+130, start_y+200), FONT, 0.5, color, 2)
    
        video_frames[index] = frame

    return video_frames
