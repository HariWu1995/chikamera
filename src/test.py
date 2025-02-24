import math

from src.utils.floorplan import generate_space_grid
from src.camera.coverage import calculate_image_plane

from src.algorithms.naive import init_greedy, init_uniform
from src.algorithms.probabilistic import simulated_annealing as optimize


# Camera Properties
CAMERA_FOV = 90  # Degrees
CAMERA_RANGE = 10  # Meters
CAMERA_HEIGHT = 3  # Meters
CAMERA_AMOUNT = 10


def test_uniform(width, length):
    image_plane = calculate_image_plane(CAMERA_RANGE, CAMERA_FOV)
    grid_cameras = init_uniform(width, length, round(image_plane))
    
    print("Grid Camera Locations:", grid_cameras)


def test_greedy(grid):

    initial_cameras = init_greedy(grid, CAMERA_AMOUNT, camera_range=CAMERA_RANGE)
    optimized_cameras = optimize(grid, initial_cameras, camera_range=CAMERA_RANGE)
    
    print("Initial Camera Locations:", initial_cameras)
    print("Optimized Camera Locations:", optimized_cameras)


if __name__ == "__main__":
    
    # Event dimensions in meters
    width, length = 50, 50
    grid = generate_space_grid(width, length)

    # Tests
    test_uniform(width, length)
    # test_greedy(grid)
