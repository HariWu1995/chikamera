from src.camera.coverage import get_points_covered_by_camera


def init_uniform(width: int, length: int, image_plane: int):
    """
    Divide grid_points into equally-sized divisions,
        then, place 1 camera for each division.
    """
    x_intervals = list(range(image_plane // 2, width, image_plane))
    y_intervals = list(range(image_plane // 2, length, image_plane))
    camera_coords = [(x, y) for x in x_intervals for y in y_intervals]
    return camera_coords


def init_greedy(grid_points, num_cameras: int, camera_range: int):
    """
    Place cameras sequentially to maximize coverage greedily.
    """
    camera_coords = []
    covered_points = set()
    
    for _ in range(num_cameras):
        best_loc = None
        max_coverage = 0
        
        for pt in grid_points:
            covered_pts = get_points_covered_by_camera(pt, camera_range)
            new_coverage = len(covered_pts - covered_points)
            
            if new_coverage > max_coverage:
                max_coverage = new_coverage
                best_loc = pt
                
        if best_loc:
            camera_coords.append(best_loc)
            covered_points.update(get_points_covered_by_camera(best_loc, camera_range))
    
    return camera_coords


