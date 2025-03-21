import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

from copy import deepcopy
from tqdm import tqdm

import cv2
import pandas as pd
import supervision as sv

import sys
sys.path.append('./scripts/advamples/sports_tennis')

import const
from trackers import PlayerTracker, BallTracker
from court_annotator import CourtAnnotator
from court_kpts_detector import CourtKptDetector

from utils.video_utils import read_video, save_video
from utils.draw_utils import draw_player_stats
from utils.bbox_utils import measure_distance
from utils.conversions import convert_pixels_to_meters


FONT = cv2.FONT_HERSHEY_SIMPLEX


def run_statistics(
        player_court_detections,
        ball_court_detections,
        ball_shot_frames,
        video_fps,
        court_width,
    ):

    player_stats_data = [{
        'frame_num': 0,

        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    } ]
    
    for ball_shot_id in tqdm(range(len(ball_shot_frames)-1)):
        start_frame = ball_shot_frames[ball_shot_id]
        end_frame   = ball_shot_frames[ball_shot_id+1]
        ball_shot_time = (end_frame - start_frame) / video_fps # seconds

        # Get distance covered by the ball
        distance_pixels = measure_distance(ball_court_detections[start_frame][1],
                                            ball_court_detections[end_frame][1])

        distance_meters = convert_pixels_to_meters(distance_pixels, const.DOUBLE_LINE_WIDTH, court_width) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_meters / ball_shot_time * 3.6

        # player who the ball
        player_positions = player_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), 
                                key=lambda player_id: measure_distance(player_positions[player_id],
                                                                    ball_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 0 if player_shot_ball == 1 else 1

        distance_pixels = measure_distance(player_court_detections[start_frame][opponent_player_id],
                                            player_court_detections[end_frame][opponent_player_id])

        distance_meters = convert_pixels_to_meters(distance_pixels, const.DOUBLE_LINE_WIDTH, court_width) 

        speed_of_opponent = distance_meters / ball_shot_time * 3.6

        player_shot_ball += 1
        opponent_player_id += 1

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    return player_stats_data


def process_video(
        source_video_path: str,
        target_video_path: str,
        model_court_path: str,
        model_player_path: str,
        model_ball_path: str,
        stub_ball_path: str,
        stub_player_path: str,
    ) -> None:

    # Read Video
    video_frames = read_video(source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    # Court Line Detector model
    print('\nDetecting court keypoints ...')
    court_detector = CourtKptDetector(model_court_path)
    court_keypoints = court_detector.predict(video_frames[0])

    # Detect Players and Ball
    print('\nTracking ...')

    player_tracker = PlayerTracker(model_path=model_player_path)
    ball_tracker   =   BallTracker(model_path=model_ball_path)

    person_detections = player_tracker.detect_frames(video_frames, read_from_stub=False, stub_path=stub_player_path)
    player_detections = player_tracker.select_players(court_keypoints, person_detections)
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False, stub_path=stub_ball_path)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Annotation
    print('\nAnnotating ...')
    court_annotator = CourtAnnotator(video_frames[0]) 

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to court coordinates
    player_court_detections, \
      ball_court_detections = court_annotator.transform_bboxes(player_detections, 
                                                                 ball_detections, court_keypoints)

    # Statistics
    print('\nStatisticizing ...')
    player_stats_data = run_statistics(
                            player_court_detections,
                            ball_court_detections,
                            ball_shot_frames,
                            video_fps = video_info.fps,
                            court_width = court_annotator.get_width_of_court(),
                        )

    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})

    player_stats_data_df = pd.DataFrame(player_stats_data)
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    for player_id in [1, 2]:
        for metric in ['shot_speed','player_speed']:
            player_stats_data_df[f'player_{player_id}_average_{metric}'] = \
            player_stats_data_df[f'player_{player_id}_total_{metric}'] / player_stats_data_df[f'player_{player_id}_number_of_shots']

    # Draw trackings
    print('\nVisualizing trackings ...')
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames =   ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = court_detector.draw_keypoints(output_frames, court_keypoints)

    # Draw BEV view
    print('\nVisualizing annotations (BEV) ...')
    output_frames = court_annotator.draw_multi(output_frames)
    output_frames = court_annotator.draw_points(output_frames, player_court_detections, color=(0,255,0))
    output_frames = court_annotator.draw_points(output_frames, ball_court_detections, color=(0,255,255))    

    # Draw Player Stats
    print('\nVisualizing statistics ...')
    output_frames = draw_player_stats(output_frames, player_stats_data_df)

    # Draw metadata
    print('\nVisualizing metadata ...')
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), FONT, 1, (0, 255, 0), 2)

    # Save video
    print('\nSaving ...')
    save_video(output_frames, target_video_path)


if __name__ == '__main__':

    ### Unit-test
    default_in_path = "./temp/supervision-2/tennis/input.mp4" 
    default_out_path = "./temp/supervision-2/tennis/output.mp4"

    default_ckpt_ball_path = "./checkpoints/YOLO/yolov5_tennisball.pt"
    default_stub_ball_path = "./scripts/advamples/sports_tennis/stubs/ball_detections.pkl"

    default_ckpt_player_path = "./checkpoints/YOLO/yolov8x.pt"
    default_stub_player_path = "./scripts/advamples/sports_tennis/stubs/player_detections.pkl"

    default_ckpt_court_path = "./checkpoints/resnet50_court_keypoints.pth"

    import argparse
    parser = argparse.ArgumentParser(description="Football Analysis")
    parser.add_argument("--source_video_path", default=default_in_path, 
                        type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", default=default_out_path,
                        type=str, help="Path to the target video file (output)")
    parser.add_argument("--model_court_path", default=default_ckpt_court_path, 
                        type=str, help="Path to the court-keypoints-detection weights file")
    parser.add_argument("--model_ball_path", default=default_ckpt_ball_path, 
                        type=str, help="Path to the ball-detection weights file")
    parser.add_argument("--stub_ball_path", default=default_stub_ball_path, 
                        type=str, help="Path to the ball tracking stub")
    parser.add_argument("--model_player_path", default=default_ckpt_player_path, 
                        type=str, help="Path to the ball-detection weights file")
    parser.add_argument("--stub_player_path", default=default_stub_player_path, 
                        type=str, help="Path to the player tracking stub")

    args = parser.parse_args()

    process_video(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        model_court_path=args.model_court_path,
        model_ball_path=args.model_ball_path,
        stub_ball_path=args.stub_ball_path,
        model_player_path=args.model_player_path,
        stub_player_path=args.stub_player_path,
    )

