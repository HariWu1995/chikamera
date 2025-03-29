"""
Convert 

    from PedestrianDynamics format:

        ├── videos_cam1
        │   ├── <scene_code_1>.mp4 (or <scene_code>.wav)
        │   ├── ... 
        │   └── <scene_code_x>.mp4 
        └── videos_cam2
            ├── <scene_code_1>.mp4 (or <scene_code>.wav)
            ├── ... 
            └── <scene_code_x>.mp4 

    to AI City Challenge format:

        ├── scene_001
        │   ├── camera_0001
        │   │   ├── calibration.json (not used)
        │   │   └── video.mp4
        │   ├── camera_0002
        │   ├── ...
        │   ├── camera_xxxx
        │   └── ground_truth.txt (not used)
        ├── scene_002
        ├── ...
        └── scene_xxx
"""
import os
import shutil


data_root = "F:/__Datasets__/PedestrianDynamics"
data_set = "BodyUpperRotation"

src_dir = f"{data_root}/{data_set}"
tgt_dir = f"F:/__Datasets__/AI-City-Fake/videos"

if os.path.isdir(tgt_dir) is False:
    os.makedirs(tgt_dir)

video_list = os.listdir(f"{src_dir}/videos_cam1")
for scene_id, video_name in enumerate(video_list):
    scene_id += 1
    scene, ext = os.path.splitext(video_name)

    if os.path.isfile(f"{src_dir}/videos_cam1/{video_name}") is False \
    or os.path.isfile(f"{src_dir}/videos_cam2/{video_name}") is False:
        continue

    print(f"{scene_id} / {len(video_list)} - Processing file {video_name} ...")

    # TODO: Convert .wav / .mov to .mp4
    if ext != '.mp4':
        pass

    scene_dir = f"{tgt_dir}/scene_{scene_id:03d}"
    if os.path.isdir(scene_dir) is False:
        os.makedirs(scene_dir)

    cam1_dir = f"{scene_dir}/camera_0001"
    if os.path.isdir(cam1_dir) is False:
        os.makedirs(cam1_dir)

    cam2_dir = f"{scene_dir}/camera_0002"
    if os.path.isdir(cam2_dir) is False:
        os.makedirs(cam2_dir)

    shutil.copyfile(f"{src_dir}/videos_cam1/{video_name}", f"{cam1_dir}/video{ext}")
    shutil.copyfile(f"{src_dir}/videos_cam2/{video_name}", f"{cam2_dir}/video{ext}")

