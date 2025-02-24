

def generate_space_grid(width: int, length: int, step: int = 1):
    """
    Generates a grid representation of the event space.
    """
    x_coords = list(range(0, width, step))
    y_coords = list(range(0, length, step))
    grid_pts = [(x, y) for x in x_coords 
                       for y in y_coords]
    return grid_pts


def get_priority_zones(width: int, length: int):
    """
    Defines key areas that need higher coverage.
    """
    return [(width//2, length//2), (0, length//2), (width//2, 0)]


