def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]


def get_width_of_bbox(bbox):
    return bbox[2] - bbox[0]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), \
           abs(p1[1] - p2[1])


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    keypoint_id = keypoint_indices[0]
    for kpt_id in keypoint_indices:
        keypoint = keypoints[kpt_id*2], keypoints[kpt_id*2+1]
        distance = abs(point[1] - keypoint[1])
        if distance < closest_distance:
            closest_distance = distance
            keypoint_id = kpt_id
    return keypoint_id
