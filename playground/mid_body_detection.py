"""Playground to run mid-body detection on a single mitosis.
Detected spots and tracks are printed in the console."""

import os
from typing import Optional
import pickle

from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector.data.tools import get_data_path
from cut_detector.factories.mid_body_detection_factory import (
    MidBodyDetectionFactory,
)
from cut_detector.utils.mid_body_detection.tracking import TRACKING_FUNCTIONS
from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.utils.tools import re_organize_channels
from cut_detector.utils.track_generation import generate_tracks_from_spots


def main(
    image_path: Optional[str] = os.path.join(
        get_data_path("mitosis_movies"), "example_video_mitosis_0_0_to_5.tiff"
    ),
    mitosis_path: Optional[str] = get_data_path("mitoses"),
    show_points: bool = True,
    show_tracks: bool = True,
) -> None:
    """
    Parameters
    ----------
    image_path : str
        Path to the image to process.
    mitosis_path : str
        Path to the mitoses data.
    show_points : bool
        If True, plot the detected spots.
    show_tracks : bool
        If True, plot the detected tracks.
    """

    # If paths are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])
    if os.path.isdir(mitosis_path):
        mitosis_path = os.path.join(mitosis_path, os.listdir(mitosis_path)[0])

    # Read data: image, mitosis_track
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX
    with open(mitosis_path, "rb") as f:
        mitosis_track: MitosisTrack = pickle.load(f)

    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = re_organize_channels(mitosis_movie)  # TYXC

    # Search for mid-body in mitosis movie
    factory = MidBodyDetectionFactory()

    spots_candidates = factory.detect_mid_body_spots(
        mitosis_movie=mitosis_movie,
        method="difference_gaussian",
        parallelization=False,
        mitosis_track=mitosis_track,
    )

    if show_points:
        print("\nSpots candidates:")
        for frame, spots in spots_candidates.items():
            for spot in spots:
                print(
                    {
                        "fr": frame,
                        "x": spot.x,
                        "y": spot.y,
                        "mklp_int": spot.intensity,
                        "sir_int": spot.sir_intensity,
                    }
                )

    generate_tracks_from_spots(
        spots_candidates, TRACKING_FUNCTIONS["spatial_laptrack"]
    )

    if show_tracks:
        print("\nSpots candidates with tracks:")
        for frame, spots in spots_candidates.items():
            for spot in spots:
                print(
                    {
                        "fr": frame,
                        "x": spot.x,
                        "y": spot.y,
                        "mklp_int": spot.intensity,
                        "sir_int": spot.sir_intensity,
                        "track_id": spot.track_id,
                    }
                )


if __name__ == "__main__":
    main()
