import argparse

import cv2
from ultralytics import YOLO

import supervision as sv
from supervision.assets import VideoAssets, download_assets


def download_video() -> str:
    download_assets(VideoAssets.PEOPLE_WALKING)
    return VideoAssets.PEOPLE_WALKING.value


def track_with_heatmap(
        model_weights_path: str,
          source_video_path: str,
          target_video_path: str,
        confidence_threshold: float = 0.35,
               iou_threshold: float = 0.5,
               heatmap_alpha: float = 0.5,
                        radius: int = 25,
                track_seconds: int = 5,
                track_threshold: float = 0.35,
            min_match_threshold: float = 0.99,
    ) -> None:

    ### instantiate model
    model = YOLO(model_weights_path)

    ### heatmap config
    heat_map_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=heatmap_alpha,
        radius=radius,
        kernel_size=25,
        top_hue=0,
        low_hue=125,
    )

    ### annotation config
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    ### get the video fps
    cap = cv2.VideoCapture(source_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    ### tracker config
    byte_tracker = sv.ByteTrack(
                        frame_rate=fps,
                 lost_track_buffer=track_seconds*fps,
        track_activation_threshold=track_threshold,
        minimum_matching_threshold=min_match_threshold,
    )

    ### video config
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    video_frames = sv.get_video_frames_generator(source_path=source_video_path, stride=1)

    ### Detect, track, annotate, save
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:

        for frame in video_frames:
            results = model(
                source=frame,
                classes=[0],  # only person class
                conf=confidence_threshold,
                iou=iou_threshold,
                # show_conf = True,
                # save_txt = True,
                # save_conf = True,
                # save = True,
                device=0,  # use None = CPU, 0 = single GPU, or [0,1] = dual GPU
                verbose=True,
            )[0]

            detections = sv.Detections.from_ultralytics(results)            # get detections
            detections = byte_tracker.update_with_detections(detections)    # update tracker

            ### draw heatmap
            annotated_frame = heat_map_annotator.annotate(scene=frame.copy(), 
                                                        detections=detections)

            ### draw other attributes from `detections` object
            labels = [
                f"#{tracker_id}" for class_id, tracker_id in zip(detections.class_id, 
                                                                detections.tracker_id)
            ]

            annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":

    ### Unit-test
    default_ckpt_path = "./checkpoints/YOLO/yolov8x.pt"
    default_out_path = "./temp/supervision/out_heatmap.mp4"
    default_in_path = "./temp/supervision/people-walking.mp4" 
    # default_in_path = download_video()
    
    import os
    if not os.path.isdir('./temp/supervision'):
        os.makedirs('./temp/supervision')

    parser = argparse.ArgumentParser(description="[Supervision] Pedestrian Tracking & Heatmap")
    parser.add_argument("--model_weights_path", default=default_ckpt_path, 
                        type=str, help="Path to the source weights file")
    parser.add_argument("--source_video_path", default=default_in_path, 
                        type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", default=default_out_path,
                        type=str, help="Path to the target video file (output)")
    parser.add_argument("--confidence_threshold", default=0.35,
                        type=float, help="Confidence threshold for the model")
    parser.add_argument("--iou_threshold", default=0.5,
                        type=float, help="IOU threshold for the model")
    parser.add_argument("--heatmap_alpha", default=0.5,
                        type=float, help="Opacity of the overlay mask, between 0 and 1")
    parser.add_argument("--radius", default=25,
                        type=float, help="Radius of the heat circle")
    parser.add_argument("--track_threshold", default=0.35,
                        type=float, help="Detection confidence threshold for track activation")
    parser.add_argument("--track_seconds", default=5,
                        type=int, help="Number of seconds to buffer when a track is lost")
    parser.add_argument("--match_threshold", default=0.99,
                        type=float, help="Threshold for matching tracks with detections")

    args = parser.parse_args()

    track_with_heatmap(
        model_weights_path=args.model_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        heatmap_alpha=args.heatmap_alpha,
               radius=args.radius,
        track_seconds=args.track_seconds,
        track_threshold=args.track_threshold,
        min_match_threshold=args.match_threshold,
    )
