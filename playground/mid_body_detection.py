import os
from typing import Optional
from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector.data.tools import get_data_path
from cut_detector.factories.mid_body_detection_factory import (
    MidBodyDetectionFactory,
)


def main(
    image_path: Optional[str] = get_data_path("mitosis_movies"),
    path_output: Optional[str] = get_data_path("mid_bodies"),
):
    # If image_path is a directory, take its first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])

    # Read image
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX

    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC

    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    # Search for mid-body in mitosis movie
    factory = MidBodyDetectionFactory()

    spots_candidates = factory.detect_mid_body_spots(
        mitosis_movie=mitosis_movie, mask_movie=mask_movie, mode="h_maxima"
    )  # mode = "bigfish" or "h_maxima" (default)
    factory.generate_tracks_from_spots(
        spots_candidates,
    )
    factory.save_mid_body_tracking(
        spots_candidates, mitosis_movie, path_output
    )


if __name__ == "__main__":
    main()
