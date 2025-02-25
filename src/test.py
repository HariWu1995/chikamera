import math

from src.utils.floorplan import generate_space_grid
from src.camera.projection import calculate_image_plane, calculate_image_plane_hv, \
                                compute_intrinsic_matrix, compute_extrinsic_matrix
from src.camera.visualization import create_camera_frustums, visualize_camera_mesh

from src.algorithms.naive import init_greedy, init_uniform
from src.algorithms.probabilistic import simulated_annealing as optimize


# Camera Properties
#   https://www.thegioididong.com/camera-giam-sat/camera-ip-ngoai-troi-360-do-3mp-imou-cruiser-2c-s7cp-3m0we
#   https://imou.vn/camera-imou-wifi-ngoai-troi/imou-ipc-gs7ep-3m0we.html
CAMERA_HFOV = 90    # Degrees
CAMERA_VFOV = 45    # Degrees
CAMERA_HEIGHT = 3   # Meters
CAMERA_ANGLE = (-90, 0, 0)   # pitch, yaw, roll in degree

RESOLUTION_H = 2304
RESOLUTION_V = 1296

# 1 / 2.8 [inch] => 6.46 × 4.83 [mm]
F = 4.2         # assumption
FX = 6.4
FY = 4.8


def test_uniform(h_size, v_size):
    print('-' * 11)
    print('Test uniform distribution ...')

    image_plane_sz = calculate_image_plane(CAMERA_HEIGHT, CAMERA_HFOV)
    image_plane_sz = round(image_plane_sz)
    print("Image Plane Size:", image_plane_sz)

    grid_cameras = init_uniform(h_size, v_size, image_plane_sz)
    print("Grid Camera Locations:", grid_cameras)

    return grid_cameras


def test_uniform_hv(h_size, v_size):
    print('-' * 11)
    print('Test uniform distribution with hFoV != vFoV...')

    image_plane_h, \
    image_plane_v = calculate_image_plane_hv(CAMERA_HEIGHT, CAMERA_HFOV, CAMERA_VFOV)
    image_plane_h = round(image_plane_h)
    image_plane_v = round(image_plane_v)
    print("Image Plane h-Size:", image_plane_h)
    print("Image Plane v-Size:", image_plane_v)

    grid_cameras = init_uniform(h_size, v_size, image_plane_h, image_plane_v)
    print("Grid Camera Locations:", grid_cameras)

    return grid_cameras


def test_greedy(grid, num_cameras: int):
    print('-' * 11)
    print('Test greedy-and-optimize distribution ...')

    image_plane_sz = calculate_image_plane(CAMERA_HEIGHT, CAMERA_HFOV)
    image_plane_sz = round(image_plane_sz)
    print("Image Plane Size:", image_plane_sz)

    initial_cameras = init_greedy(grid, num_cameras, image_plane_size=image_plane_sz)
    optimized_cameras = optimize(grid, initial_cameras, image_plane_size=image_plane_sz)
    print("Initial Camera Locations:", initial_cameras)
    print("Optimized Camera Locations:", optimized_cameras)

    return optimized_cameras


if __name__ == "__main__":
    
    # Event dimensions in meters
    h_size, v_size = 15, 10
    grid = generate_space_grid(h_size, v_size)

    # Tests
    # grid_cameras = test_greedy(grid, num_cameras=10)
    # grid_cameras = test_uniform(h_size, v_size)
    grid_cameras = test_uniform_hv(h_size, v_size)

    # Camera Extrinsic -> Mesh
    H = CAMERA_HEIGHT
    Ts = [
        compute_extrinsic_matrix(tuple(list(p) + [H]), CAMERA_ANGLE) 
                                        for p in grid_cameras
    ]

    print('-' * 11)
    print('Extrinsic Matrix:')
    print(Ts[0])

    # Camera Intrinsic
    K = compute_intrinsic_matrix(resolution_h=RESOLUTION_H, 
                                 resolution_v=RESOLUTION_V,
                            camera_hfov_degree=CAMERA_HFOV,
                            camera_vfov_degree=CAMERA_VFOV)
    Ks = [K for _ in range(len(Ts))]

    print('-' * 11)
    print('Intrinsic Matrix:')
    print(Ks[0])

    # Visualization
    camera_mesh = create_camera_frustums(Ks, Ts, camera_distance=CAMERA_HEIGHT, 
                                                 randomize_color=True)
    visualize_camera_mesh(camera_mesh, interactive=True)


