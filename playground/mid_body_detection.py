"""Playground to run mid-body detection on a single mitosis.
Detected spots and tracks are printed in the console."""

import os
from typing import Optional

from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector.data.tools import get_data_path
from cut_detector.utils.tools import re_organize_channels
from cut_detector.widget_functions.mid_body_detection import (
    perform_mid_body_detection,
)


def main(
    videos_path: Optional[str] = get_data_path("videos"),
    video_name: Optional[str] = "example_video",
    mitoses_path: Optional[str] = get_data_path("mitoses"),
    tracks_path: Optional[str] = get_data_path("tracks"),
    save: Optional[bool] = False,
    movies_save_dir: Optional[str] = None,
) -> None:
    """
    Parameters
    ----------
    videos_path : str
        Path to the videos folder.
    video_name : str
        Name of the video.
    mitoses_path : str
        Path to the mitoses data.
    tracks_path : str
        Path to the tracks data.
    save : bool
        If True, update mitoses.
    """

    videos_path = os.path.join(videos_path, f"{video_name}.tif")

    # Read image and preprocess if needed
    image = TiffReader(videos_path, respect_initial_type=True).image  # TCZYX
    image = image.squeeze()  # TCYX
    image = re_organize_channels(image)  # TYXC

    perform_mid_body_detection(
        image,
        video_name,
        mitoses_path,
        tracks_path,
        save=save,
        movies_save_dir=movies_save_dir,
    )


if __name__ == "__main__":
    main()
