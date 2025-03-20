import os
import json
import argparse
from typing import Any, Optional, Tuple

import cv2
import numpy as np

import supervision as sv


# Global Variables
KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

COLORS = sv.ColorPalette.DEFAULT
THICKNESS = 2
POLYGONS = [[]]

mouse_position: Optional[Tuple[int, int]] = None

WINDOW_NAME = "Annotation - Focus Zone(s)"

cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def resolve_source(source_path: str) -> Optional[np.ndarray]:
    if os.path.exists(source_path) is None:
        return None

    image = cv2.imread(source_path)
    if image is not None:
        return image

    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    frame = next(frame_generator)
    return frame


def save_polygons_to_json(polygons, target_path):
    data_to_save = polygons if polygons[-1] else polygons[:-1]
    with open(target_path, "w") as f:
        json.dump(data_to_save, f)


def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        POLYGONS[-1].append((x, y))


def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
    global POLYGONS, mouse_position
    image[:] = original_image.copy()
    for idx, polygon in enumerate(POLYGONS):
        color = COLORS.by_idx(idx).as_bgr() if idx < len(POLYGONS) - 1 \
                else sv.Color.WHITE.as_bgr()

        if len(polygon) > 1:
            for i in range(1, len(polygon)):
                cv2.line(
                    img=image,
                    pt1=polygon[i-1],
                    pt2=polygon[i],
                    color=color,
                    thickness=THICKNESS,
                )
            if idx < len(POLYGONS) - 1:
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )

        if idx == len(POLYGONS) - 1 and mouse_position is not None and polygon:
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=mouse_position,
                color=color,
                thickness=THICKNESS,
            )

    cv2.imshow(WINDOW_NAME, image)


def finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
    if len(POLYGONS[-1]) > 2:
        cv2.line(
            img=image,
            pt1=POLYGONS[-1][-1],
            pt2=POLYGONS[-1][0],
            color=COLORS.by_idx(0).as_bgr(),
            thickness=THICKNESS,
        )
    POLYGONS.append([])
    image[:] = original_image.copy()
    redraw_polygons(image)
    cv2.imshow(WINDOW_NAME, image)


def redraw_polygons(image: np.ndarray) -> None:
    for idx, polygon in enumerate(POLYGONS[:-1]):
        if len(polygon) > 1:
            color = COLORS.by_idx(idx).as_bgr()
            for i in range(len(polygon) - 1):
                cv2.line(
                    img=image,
                    pt1=polygon[i],
                    pt2=polygon[i + 1],
                    color=color,
                    thickness=THICKNESS,
                )
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=polygon[0],
                color=color,
                thickness=THICKNESS,
            )


def main(source_path: str, config_path: str) -> None:
    global mouse_position
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image = original_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER or key == KEY_NEWLINE:
            finalize_polygon(image, original_image)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS, config_path)
            print(f"Polygons saved to {config_path}")
            break
        redraw(image, original_image)
        if key == KEY_QUIT:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Interactively annotate polygons on images or video.")
    parser.add_argument("--source_path", "-s", required=True,
                        type=str, help="Path to the source image or video file for annotation.")
    parser.add_argument("--config_path", "-c", required=True,
                        type=str, help="Path to save the polygon annotations as a JSON file.")

    args = parser.parse_args()

    main(source_path=args.source_path, config_path=args.config_path)

