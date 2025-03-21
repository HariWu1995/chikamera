from tqdm import tqdm

import cv2
import numpy as np

import sys
sys.path.append('./scripts/advamples/sports_football')

from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from view_transformer import ViewTransformer
from utils.video_utils import read_video, save_video


def process_video(
        model_weights_path: str,
        source_video_path: str,
        target_video_path: str,
        tracker_stub_path: str,
        cammov_stub_path: str,
    ) -> None:

    # Read Video
    video_frames = read_video(source_video_path)

    # Initialize Tracker
    print('\nTracking ...')
    tracker = Tracker(model_weights_path)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=tracker_stub_path)

    tracker.add_position_to_tracks(tracks)

    # Camera-movement Estimator
    print('\nEstimating camera-movement ...')
    camera_estimator = CameraMovementEstimator(video_frames[0])
    cammov_per_frame = camera_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path=cammov_stub_path)
    
    camera_estimator.update_positions_to_tracks(tracks, cammov_per_frame)

    # View Trasnformer
    print('\nTransforming view ...')
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    print('\nEstimating speed and distance ...')
    speed_n_dist_estimator = SpeedAndDistanceEstimator()
    speed_n_dist_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    print('\nAssigning team players ...')
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in tqdm(enumerate(tracks['players'])):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Possession
    print('\nAssigning ball possession ...')
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in tqdm(enumerate(tracks['players'])):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control= np.array(team_ball_control)

    # Draw output 
    print('\nVisualizing ...')
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = camera_estimator.draw_camera_movement(output_frames, cammov_per_frame)
    output_frames = speed_n_dist_estimator.draw_speed_and_distance(output_frames, tracks)

    # Save video
    print('\nSaving ...')
    save_video(output_frames, target_video_path)


if __name__ == '__main__':

    ### Unit-test
    default_in_path = "./temp/supervision-2/football/input.mp4" 
    default_out_path = "./temp/supervision-2/football/output.mp4"
    default_ckpt_path = "./temp/supervision-2/football/yolo-v5.pt"
    default_track_path = "./scripts/advamples/sports_football/stubs/tracking.pkl"
    default_cammov_path = "./scripts/advamples/sports_football/stubs/camera_movement.pkl"

    import argparse
    parser = argparse.ArgumentParser(description="Football Analysis")
    parser.add_argument("--model_weights_path", default=default_ckpt_path, 
                        type=str, help="Path to the source weights file")
    parser.add_argument("--source_video_path", default=default_in_path, 
                        type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", default=default_out_path,
                        type=str, help="Path to the target video file (output)")
    parser.add_argument("--tracker_stub_path", default=default_track_path, 
                        type=str, help="Path to the tracker stub")
    parser.add_argument("--cammov_stub_path", default=default_cammov_path,
                        type=str, help="Path to the camera movement stub")

    args = parser.parse_args()

    process_video(
            model_weights_path=args.model_weights_path,
            source_video_path=args.source_video_path,
            target_video_path=args.target_video_path,
            tracker_stub_path=args.tracker_stub_path,
            cammov_stub_path=args.cammov_stub_path,
    )

