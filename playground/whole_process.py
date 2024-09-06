"""Playground to run cell segmentation."""

import os
from typing import Optional

from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector._widget import video_whole_process
from cut_detector.data.tools import get_data_path


def main(
    image_path: Optional[str] = get_data_path("videos"),
):
    """
    Parameters
    ----------
    image_path : str
        Path to the image to process.
    """
    # If image_path or model_path are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])

    # Read image and preprocess if needed
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX
    image = image.squeeze()  # CTYX
    image = image.transpose(1, 2, 3, 0)  # TYXC

    image_name = os.path.basename(image_path).split(".")[0]

    spots_dir_name = os.path.join(get_data_path("spots"), image_name)
    tracks_dir_name = os.path.join(get_data_path("tracks"), image_name)

    video_whole_process(
        image,
        image_name,
        default_model_check_box=True,
        segmentation_model="",
        save_check_box=False,
        movies_save_dir="",
        spots_dir_name=spots_dir_name,
        tracks_dir_name=tracks_dir_name,
        mitoses_dir_name=get_data_path("mitoses"),
    )


if __name__ == "__main__":
    main()
