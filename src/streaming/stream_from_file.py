import os
import argparse
import tempfile
import yaml

from glob import glob
from threading import Thread

from ..utils.cmd import run_command, run_command_in_thread


SERVER_CONFIG = {
    "protocols": ["tcp"], 
    "paths": {
        "all": {
            "source": "publisher",
        },
    },
}


def get_video_list(directory: str, limit: int) -> list:
    video_formats = ["*.mp4", "*.webm"]
    video_paths = []
    for video_format in video_formats:
        video_paths.extend(glob(os.path.join(directory, video_format)))
    return video_paths[:limit]


def create_config_file(directory: str) -> str:
    config_path = os.path.join(directory, "rtsp-simple-server.yml")
    with open(config_path, "w") as config_file:
        yaml.dump(SERVER_CONFIG, config_file)
    return config_path


def run_rtsp_server(config_path: str) -> None:
    command = (
        "docker run --rm --name rtsp_server -d -v "
        f"{config_path}:/rtsp-simple-server.yml -p 8554:8554 "
        "aler9/rtsp-simple-server:v1.3.0"
    )
    if run_command(command.split()) != 0:
        raise RuntimeError("Could not start the RTSP server!")


def stop_rtsp_server() -> None:
    run_command("docker kill rtsp_server".split())


def stream_videos(video_files: list, stream_url: str) -> None:
    threads = []
    for index, video_file in enumerate(video_files):
        stream_url = f"{stream_url}{index}.stream"
        print(f"Streaming {video_file} under {stream_url}")
        thread = stream_video_to_url(video_file, stream_url)
        threads.append(thread)
    for thread in threads:
        thread.join()


def stream_video_to_url(video_path: str, stream_url: str) -> Thread:
    command = (
        f"ffmpeg -re -stream_loop -1 -i {video_path} "
        f"-f rtsp -rtsp_transport tcp {stream_url}"
    )
    return run_command_in_thread(command.split())


def main(video_directory: str, stream_url: str, number_of_streams: int) -> None:
    video_files = get_video_list(video_directory, number_of_streams)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file_path = create_config_file(temp_dir)
            run_rtsp_server(config_path=config_file_path)
            stream_videos(video_files, stream_url)
    finally:
        stop_rtsp_server()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to stream videos using RTSP protocol.")

    parser.add_argument("--video_directory", required=True,
                        type=str, help="Directory containing video files to stream.")
    parser.add_argument("--stream_url", default="rtsp://localhost:8554/live",
                        type=str, help="URL to stream video files.")
    parser.add_argument("--number_of_streams", default=6,
                        type=int, help="Number of video files to stream.")
    
    args = parser.parse_args()
    
    main(
        video_directory=args.video_directory,
        number_of_streams=args.number_of_streams,
                stream_url=args.stream_url,
    )
