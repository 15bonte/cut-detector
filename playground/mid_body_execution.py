import os
import numpy as np
import time
from typing import Optional, Callable, Union
from bigfish import stack
from laptrack import LapTrack

from cut_detector.widget_functions.mid_body_detection import (
    perform_mid_body_detection,
)
from cut_detector.data.tools import get_data_path
from cut_detector.factories.mb_support import detection, tracking

SOURCE_CHOICE = 2
SOURCES = {
    0: "eval_data/Data Standard",
    1: "eval_data/Data spastin",
    2: "eval_data/Data cep55",
}
DETECTION_METHOD = detection.cur_log
TRACKING_METHOD  = tracking.cur_spatial_laptrack
PARALLELIZATION = True


def start_execution(
    data_set_path: Optional[str] = None,
    mid_body_detection_method: Union[str, Callable[[np.ndarray], np.ndarray]] = detection.cur_log,
    mid_body_tracking_method: Union[str, LapTrack] = tracking.cur_spatial_laptrack,
    parallel_detection: bool = False,
) -> list[float]:
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
    deltas = []
    for video_index, video_file in enumerate(video_files):
        video_progress = f"{video_index + 1}/{len(video_files)}"
        print(f"\nVideo {video_progress}...")

        # Read video
        raw_video = stack.read_image(
            os.path.join(video_folder, video_file), sanity_check=False
        )  # TYXC
        video_name = os.path.basename(video_file).split(".")[0]

        start = time.time()
        perform_mid_body_detection(
            raw_video,
            video_name,
            mitoses_folder,
            tracks_folder,
            mid_body_detection_method=mid_body_detection_method,
            mid_body_tracking_method=mid_body_tracking_method,
            parallel_detection=parallel_detection
        )
        end = time.time()
        delta = end-start
        deltas.append(delta)

        print("\n=== Completion Time:", delta, "====\n\n\n")
    return deltas


if __name__ == "__main__":
    print("========")
    print("Executing:", SOURCES[SOURCE_CHOICE])
    print("detection with:", DETECTION_METHOD)
    print("Tracking with:", TRACKING_METHOD)
    print("Parallelization:", PARALLELIZATION)
    print("========")

    deltas = start_execution(
        SOURCES[SOURCE_CHOICE],
        mid_body_detection_method=DETECTION_METHOD,
        mid_body_tracking_method=TRACKING_METHOD,
        parallel_detection=PARALLELIZATION,
    )
    
    print("\n\nTime:")
    for idx, d in enumerate(deltas):
        print(f"- video {idx+1}/{len(deltas)}: {d}")
