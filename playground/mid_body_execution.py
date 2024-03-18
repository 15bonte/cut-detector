import os
from typing import Optional
from bigfish import stack

from cut_detector.widget_functions.mid_body_detection import (
    perform_mid_body_detection,
)
from cut_detector.data.tools import get_data_path


def main(
    data_set_path: Optional[str] = None,
):
    """
    If data_set_path is not None, it has to be a path containing
    folders: tracks, mitoses and videos.
    """
    if data_set_path is None:
        tracks_folder = get_data_path("tracks")
        mitoses_folder = get_data_path("mitoses")
        video_folder = get_data_path("videos")
    else:
        tracks_folder = os.path.join(data_set_path, "tracks")
        mitoses_folder = os.path.join(data_set_path, "mitoses")
        video_folder = os.path.join(data_set_path, "videos")

    video_files = os.listdir(video_folder)
    for video_index, video_file in enumerate(video_files):
        video_progress = f"{video_index + 1}/{len(video_files)}"
        print(f"\nVideo {video_progress}...")

        # Read video
        raw_video = stack.read_image(
            os.path.join(video_folder, video_file), sanity_check=False
        )  # TYXC
        video_name = os.path.basename(video_file).split(".")[0]

        perform_mid_body_detection(
            raw_video,
            video_name,
            mitoses_folder,
            tracks_folder,
        )


if __name__ == "__main__":
    main()
