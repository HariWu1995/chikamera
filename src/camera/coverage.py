import math


def get_points_covered_by_camera(camera_coord, image_plane_size: int, step: int = 1):
    """
    Return a set of grid-points covered by a camera.
    """
    x, y = camera_coord
    R = image_plane_size
    coverage = set()
    for dx in range(-R, R+1, step):
        for dy in range(-R, R+1, step):
            if math.sqrt(dx**2 + dy**2) <= R:
                coverage.add((x + dx, y + dy))
    return coverage


def evaluate_coverage(camera_coords, image_plane_size: int):
    """
    Computes the total area covered by all cameras.
    """
    covered = set()
    for cam_pt in camera_coords:
        covered.update(get_points_covered_by_camera(cam_pt, image_plane_size))
    return len(covered)

