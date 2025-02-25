

def generate_space_grid(space_size_h: int, 
                        space_size_v: int, step: int = 1):
    """
    Generates a grid representation of the event space.
    """
    h_coords = list(range(0, space_size_h, step))
    v_coords = list(range(0, space_size_v, step))
    grid_pts = [(h, v) for h in h_coords 
                       for v in v_coords]
    return grid_pts


def get_priority_zones(space_size_h: int, space_size_v: int):
    """
    Defines key areas that need higher coverage.
    """
    return [(space_size_h//2, space_size_v//2), 
            (              0, space_size_v//2), 
            (space_size_h//2,               0)]


