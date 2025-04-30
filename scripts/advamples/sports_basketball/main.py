import os
import sys
sys.path.append('./scripts/advamples/sports_basketball')

from utils import read_video, save_video
from stats import SpeedAndDistanceCalculator
from viewer import TacticalViewConverter
from trackers import PlayerTracker, BallTracker, TeamAssigner
from detectors import CourtKeypointDetector, BallAquisitionDetector, PassInterceptionDetector
from drawers import (
    PlayerTracksDrawer, BallTracksDrawer,
    TacticalViewDrawer, CourtKeypointDrawer,
    FrameNumberDrawer, PassInterceptionDrawer,
    SpeedAndDistanceDrawer, TeamBallControlDrawer,
)


def process_video(
        source_video_path: str,
        target_video_path: str,
        team_1_class_name: str,
        team_2_class_name: str,
        court_image_path: str,
        model_court_path: str,
        model_player_path: str,
        model_ball_path: str,
        stub_player_path: str | None = None,
        stub_ball_path: str | None = None,
        stub_court_path: str | None = None,
        stub_assign_path: str | None = None,
    ) -> None:
    
    # Read Video
    video_frames = read_video(source_video_path)
    
    # Initialize Tracker
    player_tracker = PlayerTracker(model_player_path)
    ball_tracker = BallTracker(model_ball_path)

    # Initialize Court Detector
    court_detector = CourtKeypointDetector(model_court_path)

    # Run detections
    print('\nDetecting ...')
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_player_path)    
    ball_tracks   =   ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_ball_path)
    court_kpts =  court_detector.get_court_keypoints(video_frames, read_from_stub=True, stub_path=stub_court_path)

    # Postprocess
    print('\nPostprocessing ...')
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign Player Teams
    print('\nAssigning ...')
    team_assigner = TeamAssigner(team_1_class_name=team_1_class_name,
                                 team_2_class_name=team_2_class_name)
    players_assigned = team_assigner.get_player_teams_across_frames(
                            video_frames, player_tracks, read_from_stub=True, stub_path=stub_assign_path
                        )

    # Ball Acquisition
    print('\nAcquisiting ...')
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    # Detect Passes
    print('\nStatistizing ...')
    pass_and_interception_detector = PassInterceptionDetector()
    passes        = pass_and_interception_detector.detect_passes       (ball_aquisition, players_assigned)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition, players_assigned)

    # Tactical View
    print('\nViewing ...')
    tactical_view_converter = TacticalViewConverter(court_image_path=court_image_path)

    court_kpts = tactical_view_converter.validate_keypoints(court_kpts)
    player_pos = tactical_view_converter.transform_players_to_tactical_view(court_kpts, player_tracks)

    # Speed and Distance Calculator
    print('\nCalculating ...')
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters,
    )
    player_distances = speed_and_distance_calculator.calculate_distance(player_pos)
    player_speeds    = speed_and_distance_calculator.calculate_speed(player_distances)

    # Initialize Drawers
    ball_tracks_drawer = BallTracksDrawer()
    player_tracks_drawer = PlayerTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()
    speed_distance_drawer = SpeedAndDistanceDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    frame_number_drawer = FrameNumberDrawer()

    # Draw object Tracks
    print('\nDrawing tracks ...')
    output_frames = player_tracks_drawer.draw(video_frames, player_tracks, players_assigned, ball_aquisition)
    output_frames =   ball_tracks_drawer.draw(output_frames, ball_tracks)

    # Draw Court
    print('\nDrawing court keypoints ...')
    output_frames = court_keypoint_drawer.draw(output_frames, court_kpts)

    # Draw Frame Number
    print('\nDrawing frame number ...')
    output_frames = frame_number_drawer.draw(output_frames)

    # Draw Team Ball Control
    print('\nDrawing team ball control ...')
    output_frames = team_ball_control_drawer.draw(output_frames, players_assigned, ball_aquisition)

    # Draw Passes and Interceptions
    print('\nDrawing pass & interception ...')
    output_frames = pass_interception_drawer.draw(output_frames, passes, interceptions)
    
    # Speed and Distance Drawer
    print('\nDrawing speed & distance ...')
    output_frames = speed_distance_drawer.draw(output_frames, player_tracks, player_distances, player_speeds)

    # Draw Tactical View
    print('\nDrawing tactical view ...')
    output_frames = tactical_view_drawer.draw(output_frames,
                    tactical_view_converter.court_image_path,
                    tactical_view_converter.width,
                    tactical_view_converter.height,
                    tactical_view_converter.key_points,
                    player_pos, 
                    players_assigned, 
                    ball_aquisition
                )

    # Save video
    print('\nSaving ...')
    save_video(output_frames, target_video_path)


if __name__ == '__main__':

    from pathlib import Path
    default_court_path = Path(__file__).resolve().parent / "assets/basketball_court.png"
    assert default_court_path.exists()
    default_court_path = str(default_court_path)

    ### Unit-test
    default_in_path = "./temp/supervision-2/basketball/video_1.mp4" 
    default_out_path = "./temp/supervision-2/basketball/output_1.mp4"

    default_team1_desc = "white shirt"
    default_team2_desc = "dark blue shirt"

    ### Model Checkpoints & Stubs
    default_ckpt_ball_path   = "./temp/supervision-2/basketball/ball_detector.pt"
    default_ckpt_player_path = "./temp/supervision-2/basketball/player_detector.pt"
    default_ckpt_court_path  = "./temp/supervision-2/basketball/court_keypoint_detector.pt"

    default_stub_court_path  = "./temp/supervision-2/basketball/stub_court_track.pkl"
    default_stub_ball_path   = "./temp/supervision-2/basketball/stub_ball_track.pkl"
    default_stub_player_path = "./temp/supervision-2/basketball/stub_player_track.pkl"
    default_stub_assign_path = "./temp/supervision-2/basketball/stub_player_assigned.pkl"

    import argparse
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument("--source_video_path", default=default_in_path, type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", default=default_out_path, type=str, help="Path to the target video file (output)")
    parser.add_argument("--court_image_path", default=default_court_path, type=str, help="Path to the court layout for drawing")

    parser.add_argument("--team_1_class_name", default=default_team1_desc, type=str, help="Description of team 1 (shirt color)")
    parser.add_argument("--team_2_class_name", default=default_team2_desc, type=str, help="Description of team 2 (shirt color)")

    parser.add_argument("--model_court_path", default=default_ckpt_court_path, type=str, help="Path to the court-keypoints-detection weights file")
    parser.add_argument("--model_ball_path", default=default_ckpt_ball_path, type=str, help="Path to the ball-detection weights file")
    parser.add_argument("--model_player_path", default=default_ckpt_player_path, type=str, help="Path to the ball-detection weights file")

    parser.add_argument("--stub_ball_path", default=default_stub_ball_path, type=str, help="Path to the ball tracking stub")
    parser.add_argument("--stub_court_path", default=default_stub_court_path, type=str, help="Path to the court keypoints stub")
    parser.add_argument("--stub_player_path", default=default_stub_player_path, type=str, help="Path to the player tracking stub")
    parser.add_argument("--stub_assign_path", default=default_stub_assign_path, type=str, help="Path to the player assignment stub")
    
    args = parser.parse_args()

    process_video(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        team_1_class_name=args.team_1_class_name,
        team_2_class_name=args.team_2_class_name,
         court_image_path=args.court_image_path,
         model_court_path=args.model_court_path,
        model_player_path=args.model_player_path,
          model_ball_path=args.model_ball_path,
           stub_ball_path=args.stub_ball_path,
          stub_court_path=args.stub_court_path,
         stub_player_path=args.stub_player_path,
         stub_assign_path=args.stub_assign_path,
    )
    