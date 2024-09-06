"""Playground to run results saving."""

import os
from typing import Optional

from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector.data.tools import get_data_path
from cut_detector.utils.tools import re_organize_channels
from cut_detector.widget_functions.save_results import perform_results_saving


def main(
    image_path: Optional[str] = get_data_path("videos"),
    mitosis_path: Optional[str] = get_data_path("mitoses"),
    results_dir: Optional[str] = get_data_path("results"),
):
    """
    Parameters
    ----------
    image_path : str
        Path to the image to process.
    mitosis_path : str
        Path to the mitoses data.
    """
    # If paths are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])

    # Read image and preprocess if needed
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX
    image = re_organize_channels(image.squeeze())  # TYXC

    perform_results_saving(
        mitosis_path,
        show=False,
        save_dir=results_dir,
        verbose=True,
        video=image,
    )


if __name__ == "__main__":
    main()
