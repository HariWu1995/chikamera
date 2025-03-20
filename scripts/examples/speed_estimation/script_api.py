import argparse
from collections import defaultdict, deque

import cv2
import numpy as np

from inference.models.utils import get_roboflow_model
import supervision as sv


window_name = "SuperVision"

# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

SOURCE = np.array([
    [1252, 787], 
    [2298, 803], 
    [5039, 2159], 
    [-550, 2159],
])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [               0,                 0],
    [TARGET_WIDTH - 1,                 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [               0, TARGET_HEIGHT - 1],
])


class ViewTransform:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def process_video(
                    model_id: str,
                     api_key: str,
            source_video_path: str,
            target_video_path: str,
         confidence_threshold: float = 0.3,
                iou_threshold: float = 0.7,
    ) -> None:

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    model = get_roboflow_model(model_id=model_id, api_key=api_key)
    tracker = sv.ByteTrack(frame_rate=video_info.fps, 
            track_activation_threshold=confidence_threshold)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    trace_annotator = sv.TraceAnnotator(thickness=thickness,
                                        trace_length=video_info.fps * 2,
                                        position=sv.Position.BOTTOM_CENTER)
    label_annotator = sv.LabelAnnotator(text_thickness=thickness,
                                        text_scale=text_scale,
                                        text_position=sv.Position.BOTTOM_CENTER)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transform = ViewTransform(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    with sv.VideoSink(target_video_path, video_info) as sink:

        for frame in frame_generator:
            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)
            detections = detections[detections.confidence > confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=iou_threshold)
            detections = tracker.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transform.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end   = coordinates[tracker_id][0]
                    dist = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = dist / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            sink.write_frame(annotated_frame)

        #     cv2.imshow(window_name, annotated_frame)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break

        # cv2.destroyAllWindows()


if __name__ == "__main__":

    ### Unit-test
    default_model_id = "yolov8x-640"
    default_out_path = "./temp/supervision/out_speed_est.mp4"
    default_in_path = "./temp/supervision/vehicles.mp4" 
    # default_in_path = download_video()
    
    import os
    if not os.path.isdir('./temp/supervision'):
        os.makedirs('./temp/supervision')

    parser = argparse.ArgumentParser(description="[Supervision] Speed Estimation")
    parser.add_argument("--model_id", default=default_model_id, 
                        type=str, help="Roboflow model ID")
    parser.add_argument("--roboflow_api_key", default=None,
                        type=str, help="Roboflow API KEY")
    parser.add_argument("--source_video_path", default=default_in_path, 
                        type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", default=default_out_path,
                        type=str, help="Path to the target video file (output)")
    parser.add_argument("--confidence_threshold", default=0.35,
                        type=float, help="Confidence threshold for the model")
    parser.add_argument("--iou_threshold", default=0.7,
                        type=float, help="IOU threshold for the model")

    args = parser.parse_args()

    api_key = args.roboflow_api_key
    api_key = os.environ.get("ROBOFLOW_API_KEY", api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API key is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    args.roboflow_api_key = api_key

    process_video(
                    model_id=args.model_id,
                     api_key=args.roboflow_api_key,
            source_video_path=args.source_video_path,
            target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
    )

