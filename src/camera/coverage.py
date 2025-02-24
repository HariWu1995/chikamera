import math


def calculate_image_plane(camera_range: int, camera_fov_degree: int):
    """
    Calculate the approximate size of the image plane based on camera properties.
    """
    camera_fov = math.radians(camera_fov_degree / 2)
    plane_size = 2 * camera_range * math.tan(camera_fov)
    return plane_size


def get_points_covered_by_camera(camera_coord, camera_range: int, step: int = 1):
    """
    Return a set of grid-points covered by a camera.
    """
    x, y = camera_coord
    R = camera_range
    coverage = set()
    for dx in range(-R, R+1, step):
        for dy in range(-R, R+1, step):
            if math.sqrt(dx**2 + dy**2) <= R:
                coverage.add((x + dx, y + dy))
    return coverage


def evaluate_coverage(camera_coords, camera_range: int):
    """
    Computes the total area covered by all cameras.
    """
    covered = set()
    for cam_pt in camera_coords:
        covered.update(get_points_covered_by_camera(cam_pt, camera_range))
    return len(covered)

