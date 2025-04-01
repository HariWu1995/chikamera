import random as rd

import camtools
import open3d


def create_camera_frustums(Ks, Ts, camera_distance: int, 
                                   randomize_color: bool = False, **kwargs):
    if randomize_color:
        camera_colors = {
                c: (rd.uniform(0, 1), rd.uniform(0, 1), rd.uniform(0, 1)) 
            for c in range(len(Ts))
        }
    else:
        camera_colors = None

    camera_vizargs = dict(
                       size = camera_distance, 
        highlight_color_map = camera_colors, 
        center_line_color = (0,0,0),
        center_line = True,
        center_ray = True, 
        up_triangle = False,
    )
    camera_vizargs.update(**kwargs)

    camera_frustums = camtools.camera.create_camera_frustums(Ks, Ts, **camera_vizargs)
    return camera_frustums


def visualize_camera_mesh(camera_mesh, interactive: bool = True, 
                                        output_path: str = 'camera_mesh.png'):
    if interactive:
        window_vizargs = dict()
        open3d.visualization.draw_geometries([camera_mesh], **window_vizargs)

    else:
        vis = open3d.visualization.Visualizer() 
        vis.create_window() 
        vis.add_geometry(camera_mesh)
        # vis.update_geometry(camera_mesh)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path) 
        vis.destroy_window()

