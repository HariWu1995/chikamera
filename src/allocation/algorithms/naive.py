from src.camera.coverage import get_points_covered_by_camera


def init_uniform(space_size_h: int, space_size_v: int, 
                image_plane_h: int, image_plane_v: int = None):
    """
    Divide grid_points into equally-sized divisions,
        then, place 1 camera for each division.
    """
    if image_plane_v is None:
        image_plane_v = image_plane_h
    h_intervals = list(range(image_plane_h // 2, space_size_h, image_plane_h))
    v_intervals = list(range(image_plane_v // 2, space_size_v, image_plane_v))
    camera_coords = [(h, v) for h in h_intervals for v in v_intervals]
    return camera_coords


def init_greedy(grid_points, num_cameras: int, image_plane_size: int):
    """
    Place cameras sequentially to maximize coverage greedily.
    """
    camera_coords = []
    covered_points = set()
    
    for _ in range(num_cameras):
        best_loc = None
        max_coverage = 0
        
        for pt in grid_points:
            covered_pts = get_points_covered_by_camera(pt, image_plane_size)
            new_coverage = len(covered_pts - covered_points)
            
            if new_coverage > max_coverage:
                max_coverage = new_coverage
                best_loc = pt
                
        if best_loc:
            camera_coords.append(best_loc)
            covered_points.update(get_points_covered_by_camera(best_loc, image_plane_size))
    
    return camera_coords


