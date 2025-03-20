import os
import argparse
from tqdm import tqdm

from inference.models.utils import get_roboflow_model

import supervision as sv


def process_video(
        api_key: str,
        model_id: str,
        source_video_path: str,
        target_video_path: str,
        confidence_threshold: float = 0.3,
                iou_threshold: float = 0.7,
    ) -> None:

    model = get_roboflow_model(model_id=model_id, api_key=api_key)
    tracker = sv.ByteTrack()

    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_frames = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:

        for frame in tqdm(video_frames, total=video_info.total_frames):
            results = model.infer(frame, confidence=confidence_threshold, 
                                        iou_threshold=iou_threshold)[0]

            detections = sv.Detections.from_inference(results)
            detections = tracker.update_with_detections(detections)

            annotated_frame = frame.copy()
            annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":

    ### Unit-test
    default_model_id = "yolov8x-1280"
    default_out_path = "./temp/supervision/out_track_all.mp4"
    default_in_path = "./temp/supervision/people-walking.mp4" 
    # default_in_path = download_video()
    
    import os
    if not os.path.isdir('./temp/supervision'):
        os.makedirs('./temp/supervision')

    parser = argparse.ArgumentParser(description="[Supervision] Object(s) Tracking using Roboflow API")
    parser.add_argument("--model_id", default=default_model_id,
                        type=str, help="Roboflow model ID")
    parser.add_argument("--roboflow_api_key", default=None,
                        type=str, help="Roboflow API key")
    parser.add_argument("--source_video_path", default=default_in_path, 
                        type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", default=default_out_path,
                        type=str, help="Path to the target video file (output)")
    parser.add_argument("--confidence_threshold", default=0.35,
                        type=float, help="Confidence threshold for the model")
    parser.add_argument("--iou_threshold", default=0.7,
                        type=float, help="IOU threshold for the model")

    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", args.roboflow_api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API key is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY as environment variable."
        )
    args.roboflow_api_key = api_key

    process_video(
        api_key=args.roboflow_api_key,
        model_id=args.model_id,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
    )
