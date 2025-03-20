# waiting time in zone

[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=hAWpsIuem10)

## 👋 abstract

Practical demonstration on leveraging computer vision for analyzing wait times and monitoring the duration that objects or individuals spend in predefined areas of video frames. 

This example project, is well-designed for retail analytics or traffic management applications.

https://github.com/roboflow/supervision/assets/26109316/d051cc8a-dd15-41d4-aa36-d38b86334c39

## 💻 install

- clone repository and navigate to example directory

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
    cd supervision/examples/time_in_zone
    ```

- setup python environment and activate it \[optional\]

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

- install required dependencies

    ```bash
    pip install -r requirements.txt
    ```

## 🛠 scripts

### `download_from_youtube`

This script allows you to download a video from YouTube.

- `--url`: The full URL of the YouTube video you wish to download.
- `--output_path` (optional): Specifies the directory where the video will be saved.
- `--file_name` (optional): Sets the name of the saved video file.

```bash
python scripts/download_from_youtube.py \
    --url "https://www.youtube.com/watch?v=-8zyEwAa50Q" \
    --output_path "data/checkout" \
    --file_name "video.mp4"
```

```bash
python scripts/download_from_youtube.py \
    --url "https://www.youtube.com/watch?v=MNn9qKG2UFI" \
    --output_path "data/traffic" \
    --file_name "video.mp4"
```

Or you can download from [Yt1z.net](https://yt1z.net/en/)

### `stream_from_file`

This script allows you to stream video files from a directory. It's an awesome way to mock a live video stream for local testing. 

Video will be streamed in a loop under `rtsp://localhost:8554/live0.stream` URL. This script requires docker to be installed.

- `--video_directory`: Directory containing video files to stream.
- `--number_of_streams`: Number of video files to stream.

```bash
python scripts/stream_from_file.py \
    --video_directory "data/checkout" \
    --number_of_streams 1
```

### `draw_zones`

If you want to test zone time in zone analysis on your own video, you can use this script to design custom zones and save results as a JSON file. 

The script will open a window where you can draw polygons on the source image or video file. The polygons will be saved as a JSON file.

- `--source_path`: Path to the source image or video file for drawing polygons.
- `--zone_config_path`: Path where the polygon annotations will be saved as a JSON file.
- `enter` - finish drawing the current polygon.
- `escape` - cancel drawing the current polygon.
- `q` - quit the drawing window.
- `s` - save zone configuration to a JSON file.

```bash
python scripts/draw_zones.py \
    --source_path "data/checkout/video.mp4" \
    --zone_config_path "data/checkout/config.json"
```

https://github.com/roboflow/supervision/assets/26109316/9d514c9e-2a61-418b-ae49-6ac1ad6ae5ac

## 🎬 video & stream processing

General Arguments:
- `--zone_config_path`: Path to the zone configuration JSON file.
- `--device`: Computation device (`'cpu'`, `'mps'` or `'cuda'`). Default is `'cpu'`.
- `--classes`: List of class IDs to track. If empty, all classes are tracked.
    - Class ID for `person` is 0.
    - Class IDs for `vehicle` are [2 5 6 7].
- `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
- `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

Using Ultralytics:
- `--weights`: Path to the model weights file. Default is `'yolov8s.pt'`.

Using Roboflow Inference:
- `--model_id`: Roboflow model ID.

### `script`

Script to run object detection on a video file using the Ultralytics YOLOv8 model.
- `--source_video_path`: Path to the source video file.

```bash
python script.py \
    --zone_config_path "data/checkout/config.json" \
    --source_video_path "data/checkout/video.mp4" \
    --weights "yolov8x.pt" \
    --device "cuda" \
    --classes 0 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

### `script_stream`

Script to run object detection on a video stream using the Ultralytics YOLOv8 model.
- `--rtsp_url`: Complete RTSP URL for the video stream.

```bash
python ultralytics_stream_example.py \
    --zone_config_path "data/traffic/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --weights "yolov8x.pt" \
    --device "cuda" \
    --classes 2 5 6 7 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

<details>
<summary>👉 show Roboflow API examples</summary>

### `api_file`

```bash
python api_file.py \
    --zone_config_path "data/checkout/config.json" \
    --source_video_path "data/checkout/video.mp4" \
    --model_id "yolov8x-640" \
    --classes 0 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

https://github.com/roboflow/supervision/assets/26109316/d051cc8a-dd15-41d4-aa36-d38b86334c39

```bash
python inference_file_example.py \
    --zone_config_path "data/traffic/config.json" \
    --source_video_path "data/traffic/video.mp4" \
    --model_id "yolov8x-640" \
    --classes 2 5 6 7 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

https://github.com/roboflow/supervision/assets/26109316/5ec896d7-4b39-4426-8979-11e71666878b

### `api_stream`

```bash
python api_stream.py \
    --zone_config_path "data/checkout/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --model_id "yolov8x-640" \
    --classes 0 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

```bash
python api_stream.py \
    --zone_config_path "data/traffic/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --model_id "yolov8x-640" \
    --classes 2 5 6 7 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

</details>

## © license

This demo integrates two main components, each with its own licensing:

- ultralytics: The object detection model used in this demo, YOLOv8, is distributed under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).You can find more details about this license here.

- supervision: The analytics code that powers the zone-based analysis in this demo is based on the Supervision library, which is licensed under the [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). This makes the Supervision part of the code fully open source and freely usable in your projects.
