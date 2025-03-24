import argparse
from tqdm import tqdm

from ultralytics import YOLO

import supervision as sv


def process_video(
        model_weights_path: str,
        source_video_path: str,
        target_video_path: str,
        confidence_threshold: float = 0.3,
                iou_threshold: float = 0.7,
    ) -> None:

    model = YOLO(model_weights_path)
    tracker = sv.ByteTrack()

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_frames = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:

        for frame in tqdm(video_frames, total=video_info.total_frames):
            
            results = model.track(
                # classes=[0],  # only person class
                source=frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                # show_conf = True,
                # save_txt = True,
                # save_conf = True,
                # save = True,
                device=0,  # use None = CPU, 0 = single GPU, or [0,1] = dual GPU
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            annotated_frame = frame.copy()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":

    ### Unit-test
    default_ckpt_path = "./checkpoints/YOLO/yolo11x-seg.pt"
    default_out_path = "./temp/supervision/out_track_seg.mp4"
    default_in_path = "./temp/supervision/people-walking.mp4" 
    # default_in_path = download_video()
    
    import os
    if not os.path.isdir('./temp/supervision'):
        os.makedirs('./temp/supervision')

    parser = argparse.ArgumentParser(description="[Supervision] Object(s) Tracking")
    parser.add_argument("--model_weights_path", default=default_ckpt_path, 
                        type=str, help="Path to the source weights file")
    parser.add_argument("--source_video_path", default=default_in_path, 
                        type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", default=default_out_path,
                        type=str, help="Path to the target video file (output)")
    parser.add_argument("--confidence_threshold", default=0.35,
                        type=float, help="Confidence threshold for the model")
    parser.add_argument("--iou_threshold", default=0.7,
                        type=float, help="IOU threshold for the model")

    args = parser.parse_args()

    process_video(
        model_weights_path=args.model_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
    )
