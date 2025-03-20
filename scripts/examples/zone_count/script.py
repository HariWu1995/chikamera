import argparse
import json

from tqdm import tqdm
from typing import List, Tuple

import cv2
import numpy as np
import supervision as sv
from supervision.assets import VideoAssets, download_assets

from ultralytics import YOLO


COLORS = sv.ColorPalette.DEFAULT


def download_video() -> str:
    download_assets(VideoAssets.MARKET_SQUARE)
    return VideoAssets.MARKET_SQUARE.value


def load_zones_config(file_path: str) -> List[np.ndarray]:
    """
    Load polygon zone configurations from a JSON file.

    This function reads a JSON file which contains polygon coordinates, and
    converts them into a list of NumPy arrays. 
    Each polygon is represented as a NumPy array of coordinates.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        List[np.ndarray]: A list of polygons, each represented as a NumPy array.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data["polygons"]]


def initiate_annotators(
    polygons: List[np.ndarray], resolution_wh: Tuple[int, int]
    ) -> Tuple[List[sv.PolygonZone], List[sv.PolygonZoneAnnotator], List[sv.BoxAnnotator]]:

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    zones = []
    zone_annotators = []
    bbox_annotators = []

    for index, polygon in enumerate(polygons):
        zone = sv.PolygonZone(polygon=polygon)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone,
                                        text_scale=text_scale*2, text_thickness=thickness*2,
                                        color=COLORS.by_idx(index), thickness=thickness)
        box_annotator = sv.BoxAnnotator(color=COLORS.by_idx(index), thickness=thickness)
        zones.append(zone)
        zone_annotators.append(zone_annotator)
        bbox_annotators.append(box_annotator)

    return zones, zone_annotators, bbox_annotators


def detect(
        frame: np.ndarray, model: YOLO, confidence_threshold: float = 0.5
    ) -> sv.Detections:
    """
    Detect objects in a frame using a YOLO model, filtering detections by class ID and
        confidence threshold.

    Args:
        frame (np.ndarray): The frame to process, expected to be a NumPy array.
        model (YOLO): The YOLO model used for processing the frame.
        confidence_threshold (float): The confidence threshold for filtering
            detections. Default is 0.5.

    Returns:
        sv.Detections: Filtered detections after processing the frame with the YOLO
            model.

    Note:
        This function is specifically tailored for a YOLO model and assumes class ID 0
            for filtering.
    """
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    filter_by_class = detections.class_id == 0  # person class id = 0
    filter_by_conf = detections.confidence > confidence_threshold
    return detections[filter_by_class & filter_by_conf]


def annotate(
        frame: np.ndarray,
        zones: List[sv.PolygonZone],
        zone_annotators: List[sv.PolygonZoneAnnotator],
        bbox_annotators: List[sv.BoxAnnotator],
        detections: sv.Detections,
    ) -> np.ndarray:
    """
    Annotate a frame with zone and box annotations based on given detections.

    Args:
        frame (np.ndarray): The original frame to be annotated.
        zones (List[sv.PolygonZone]): A list of polygon zones used for detection.
        zone_annotators (List[sv.PolygonZoneAnnotator]): A list of annotators for
            drawing zone annotations.
        bbox_annotators (List[sv.BoxAnnotator]): A list of annotators for
            drawing box annotations.
        detections (sv.Detections): Detections to be used for annotation.

    Returns:
        np.ndarray: The annotated frame.
    """
    annotated_frame = frame.copy()
    for zone,  zone_annotator,  bbox_annotator in zip(
        zones, zone_annotators, bbox_annotators
    ):
        zone_detections = detections[zone.trigger(detections=detections)]
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=zone_detections)
    return annotated_frame


def process_video(
        model_weights_path: str,
        zone_config_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
                iou_threshold: float = 0.7,
    ) -> None:

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    resolution = video_info.resolution_wh

    polygons = load_zones_config(zone_config_path)
    zones, zone_annotators, \
            bbox_annotators = initiate_annotators(polygons=polygons, resolution_wh=resolution)

    model = YOLO(model_weights_path)

    frames_generator = sv.get_video_frames_generator(args.source_video_path)

    if target_video_path is not None:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames):
                detections = detect(frame, model, confidence_threshold)
                annotated_frame = annotate(
                    frame=frame,
                    zones=zones,
                    zone_annotators=zone_annotators,
                    bbox_annotators=bbox_annotators,
                    detections=detections,
                )
                sink.write_frame(annotated_frame)
    else:
        for frame in tqdm(frames_generator, total=video_info.total_frames):
            detections = detect(frame, model, confidence_threshold)
            annotated_frame = annotate(
                frame=frame,
                zones=zones,
                zone_annotators=zone_annotators,
                bbox_annotators=bbox_annotators,
                detections=detections,
            )
            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":

    ### Unit-test
    default_zone_path = "./scripts/examples/count_in_zone/config/multi-zone-config.json"
    default_ckpt_path = "./checkpoints/YOLO/yolov8x.pt"
    default_out_path = "./temp/supervision/out_zone_count.mp4"
    default_in_path = "./temp/supervision/market-square.mp4" 
    # default_in_path = download_video()

    import os
    if not os.path.isdir('./temp/supervision'):
        os.makedirs('./temp/supervision')

    parser = argparse.ArgumentParser(description="[Supervision] Person Counting in Multi-Zone")
    parser.add_argument("--zone_config_path", default=default_zone_path, 
                        type=str, help="Path to the zone configuration JSON file")
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
          zone_config_path=args.zone_config_path,
        model_weights_path=args.model_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
    )

