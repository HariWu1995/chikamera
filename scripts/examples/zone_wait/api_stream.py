import argparse
from typing import List

import cv2
import numpy as np

from inference import get_model
import supervision as sv

import sys
sys.path.append("scripts/examples/zone_wait")

from utils.general import find_in_list, get_stream_frames_generator, load_zones_config
from utils.timers import ClockBasedTimer


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(color=COLORS, 
                                text_color=sv.Color.from_hex("#000000"))

window_name = "SuperVision"

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def main(
        model_id: str,
        rtsp_url: str,
        zone_config_path: str,
        confidence_threshold: float,
                iou_threshold: float,
        classes: List[int],
        device: str = 'cuda',
    ) -> None:

    model = get_model(model_id=model_id)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)

    frames_generator = get_stream_frames_generator(rtsp_url=rtsp_url)
    fps_monitor = sv.FPSMonitor()

    polygons = load_zones_config(file_path=zone_config_path)
    zones = [
        sv.PolygonZone(polygon=polygon, triggering_anchors=(sv.Position.CENTER,))
        for polygon in polygons
    ]
    timers = [ClockBasedTimer() for _ in zones]

    for frame in frames_generator:
        fps_monitor.tick()
        fps = fps_monitor.fps

        results = model.infer(frame, confidence=confidence_threshold, iou_threshold=iou_threshold)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = detections.with_nms(threshold=iou_threshold)
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                            scene=annotated_frame,
                        detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )

            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                detections=detections_in_zone,
                    labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    default_zone_path = "./temp/supervision/supermarket-checkout.json" 
    default_stream_url = "rtsp://localhost:8554/live0.stream" 
    default_model_id = "yolov8s-640"
    default_class_ids = [0]

    parser = argparse.ArgumentParser(description="[Supervision] Waiting Time in Multi-Zone, using RTSP stream.")
    parser.add_argument("--zone_config_path", default=default_zone_path, 
                        type=str, help="Path to the zone configuration JSON file")
    parser.add_argument("--rtsp_url", default=default_stream_url, 
                        type=str, help="RTSP URL for the video streaming. "
                                        "You can run `scripts/stream_from_video.py`")
    parser.add_argument("--model_id", default=default_model_id, 
                        type=str, help="Roboflow model ID.")
    parser.add_argument("--classes", nargs="*", default=default_class_ids,
                        type=int, help="List of class IDs to track. If empty, all classes are tracked.")
    parser.add_argument("--confidence_threshold", default=0.35,
                        type=float, help="Confidence threshold for the model")
    parser.add_argument("--iou_threshold", default=0.7,
                        type=float, help="IOU threshold for the model")

    args = parser.parse_args()

    main(
                model_id=args.model_id,
                classes=args.classes,
                rtsp_url=args.rtsp_url,
        zone_config_path=args.zone_config_path,
        confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
    )
    
    