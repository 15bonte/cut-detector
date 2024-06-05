import os
from time import time
from typing import Optional
import pickle
from pathlib import Path

import numpy as np
from aicsimageio import AICSImage

from cut_detector.data.tools import get_data_path
from cut_detector.factories.mid_body_detection_factory import (
    MidBodyDetectionFactory,
)
from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.utils.mb_support import detection, tracking
from cut_detector.utils.gen_track import generate_tracks_from_spots


D_METHOD = detection.cur_dog
T_METHOD = tracking.cur_spatial_laptrack

MEASURE_DETECTION_TIME = True
SHOW_POINTS = True
SHOW_TRACKS = True

DETECTION_STEP_COUNT = 1


def main(
    image_path: Optional[str] = os.path.join(
        get_data_path("mitosis_movies"), "example_video_mitosis_0_4_to_0.tiff"
    ),
    mitoses_path: Optional[str] = get_data_path("mitoses"),
    path_output: Optional[str] = get_data_path("mid_bodies"),
    save: bool = False,
    show_points: bool = True,
    show_tracks: bool = True,
) -> None:
    """Playground function to run mid-body detection on a single mitosis.
    Detected spots and tracks are printed in the console.
    Saving is possible - avoid with default data.

    Parameters
    ----------
    image_path : str
        Path to the image to process.
    mitoses_path : str
        Path to the mitoses data.
    path_output : str
        Path to the output folder.
    save : bool
        If True, save the results.
    show_points : bool
        If True, plot the detected spots.
    show_tracks : bool
        If True, plot the detected tracks.
    """

    mitosis_path = Path(mitoses_path) / f"{Path(image_path).stem}.bin"
    with open(mitosis_path, "rb") as f:
        track: MitosisTrack = pickle.load(f)

    # Read image
    image = read_tiff(image_path)  # TCZYX

    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC

    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    # Search for mid-body in mitosis movie
    factory = MidBodyDetectionFactory()

    spots_candidates = factory.detect_mid_body_spots(
        mitosis_movie=mitosis_movie,
        mask_movie=mask_movie,
        mitosis_track=track,
    )

    if show_points:
        print("Spots candidates:")
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

    generate_tracks_from_spots(spots_candidates, T_METHOD)

    if show_tracks:
        print("Spots candidates with tracks:")
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

    if save:  # To be removed -> everything has to be moved to the code
        factory.save_mid_body_tracking(
            spots_candidates, mitosis_movie, path_output
        )


def read_tiff(path: str) -> np.ndarray:
    """Duplicated function from cnn_framework.
    Rewritten here to avoid long useless imports.
    """
    aics_img = AICSImage(path)
    target_order = "TCZYX"
    original_order = aics_img.dims.order

    img = aics_img.data

    # Add missing dimensions if necessary
    for dim in target_order:
        if dim not in original_order:
            original_order = dim + original_order
            img = np.expand_dims(img, axis=0)

    indexes = [original_order.index(dim) for dim in target_order]

    return np.moveaxis(img, indexes, list(range(len(target_order))))


if __name__ == "__main__":
    main()
