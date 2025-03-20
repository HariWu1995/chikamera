import argparse
import os
from typing import Optional

from pytube import YouTube


def main(url: str, output_path: Optional[str], file_name: Optional[str] = None) -> None:
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    stream.download(output_path=output_path, filename=file_name)
    final_name = file_name if file_name else yt.title
    final_path = output_path if output_path else "current directory"
    print(f"Download completed! Video saved as '{final_name}' in '{final_path}'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download a specific YouTube video by providing its URL.")
    parser.add_argument("--url", required=True,
                        type=str, help="The full URL of the YouTube video you wish to download.")
    parser.add_argument("--output_path", default="data/source",
                        type=str, help="[Optional] Specifies the directory where the video will be saved.")
    parser.add_argument("--file_name", default=None,
                        type=str, help="[Optional] Sets the name of the saved video file.")

    args = parser.parse_args()
    main(url=args.url, output_path=args.output_path, file_name=args.file_name)
