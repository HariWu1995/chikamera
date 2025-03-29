import cv2
import itertools


FONT = cv2.FONT_HERSHEY_SIMPLEX

COLORS = itertools.cycle([
    (  0,   0, 255), (  0, 255,   0), (255,   0,   0),
    (127,   0, 127), (  0, 127, 127), (127, 127,   0),
    (  0,   0, 127), (  0, 127,   0), (127,   0,   0),
    (  0, 169, 169), (169,   0, 169), (169,   0, 169),
])


def visualize(image, bboxes, opacity: float = 0.32):

    output = image.copy()
    overlay = image.copy()

    for i, bbox in enumerate(bboxes):
        # Bbox info
        clss, x1, y1, x2, y2, score = bbox
        x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]
        color = next(COLORS)
        # label = f"Class {clss}: {score*100:.1f}%"

        # Draw filled rectangle with opacity
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        output = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

        # Draw legend
        # (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        # cv2.putText(output, label, (x1, y1-th), FONT, 0.5, color, 2, cv2.LINE_AA)

    return output

