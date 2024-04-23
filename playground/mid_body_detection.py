import os
from time import time
from typing import Optional
from cnn_framework.utils.readers.tiff_reader import TiffReader

import numpy as np 

from cut_detector.data.tools import get_data_path
from cut_detector.factories.mid_body_detection_factory import (
    MidBodyDetectionFactory,
)
from cut_detector.factories.mb_support import detection, tracking
from cut_detector.utils.mid_body_track import MidBodyTrack
from cut_detector.utils.gen_track import generate_tracks_from_spots

D_METHOD = detection.cur_dog
T_METHOD = tracking.cur_spatial_laptrack

MEASURE_DETECTION_TIME = True
SHOW_POINTS            = False
SHOW_TRACKS            = False
SHOULD_SAVE            = False

# PARALLELIZE = "pool"
# PARALLELIZE = "thread"
PARALLELIZE = "np_thread"

DETECTION_STEP_COUNT = 10

def main(
    image_path: Optional[str] = get_data_path("mitosis_movies"),
    path_output: Optional[str] = get_data_path("mid_bodies"),
):
    # If image_path is a directory, take its first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])

    print("src:", image_path)
    print("//:", PARALLELIZE)

    # Read image
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX

    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC

    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    # Search for mid-body in mitosis movie
    factory = MidBodyDetectionFactory()

    times = []

    for c in range(DETECTION_STEP_COUNT):
        if MEASURE_DETECTION_TIME: 
            start = time()
        spots_candidates = factory.detect_mid_body_spots(
            # mitosis_movie=mitosis_movie, mask_movie=mask_movie, mode="h_maxima"
            mitosis_movie=mitosis_movie,
            mask_movie=mask_movie,
            mode=D_METHOD,
            parallelization=PARALLELIZE,
        )
        if MEASURE_DETECTION_TIME:
            end = time()
            delta = end - start
            print(f"====== Detection time: {delta:.3f}s =========")
            times.append(delta)

    if SHOW_POINTS:
        for frame, spots in spots_candidates.items():
            for spot in spots:
                print(
                    {
                        "fr": frame,
                        "x": spot.x,
                        "y": spot.y,
                        "mlkp_int": spot.intensity,
                        "sir_int": spot.sir_intensity,
                    }
                )

    generate_tracks_from_spots(spots_candidates, T_METHOD)


    if SHOW_TRACKS:
        for frame, spots in spots_candidates.items():
            for spot in spots:
                print(
                    {
                        "fr": frame,
                        "x": spot.x,
                        "y": spot.y,
                        "mlkp_int": spot.intensity,
                        "sir_int": spot.sir_intensity,
                        "track_id": spot.track_id
                    }
                )

    if SHOULD_SAVE:
        factory.save_mid_body_tracking(
            spots_candidates, mitosis_movie, path_output
        )

    return times


if __name__ == "__main__":
    strats = ["pool", "thread", "np_thread", "max_thread"]
    times = []
    for s_p in strats:
        PARALLELIZE = s_p
        times.append(np.array(main(
            # "./src/cut_detector/data/mitosis_movies/example_video_mitosis_0_0_to_4.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/a_siLuci-1_mitosis_33_7_to_63.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/s1_siLuci-1_mitosis_14_158_to_227.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/s2_siLuci-1_mitosis_15_67_to_228,211.tiff",
            "./src/cut_detector/data/mid_bodies_movies_test/s3_siLuci-1_mitosis_17_170_to_195.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/s4_siLuci-1_mitosis_24_128_to_135.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/s5_siLuci-1_mitosis_27_22_to_93,87.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/s6_siLuci-1_mitosis_28_50_to_91.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/s7_siLuci-1_mitosis_31_19_to_73.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/s9_siLuci-1_mitosis_34_21_to_68,62.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/20231019-t1_siCep55-50-4_mitosis_21_25_to_117.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/cep2_20231019-t1_siCep55-50-4_mitosis_24_17_to_104.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/cep_1.tiff",
            # "./src/cut_detector/data/mid_bodies_movies_test/example_video_mitosis_0_0_to_4.tiff",
            # get_data_path("mid_bodies_tests")
        )))
    for idx, s_p in enumerate(strats):
        print("//:", s_p)
        print(f"min: {np.median(times[idx]):.2f}")
        print(f"q1: {np.percentile(times[idx], 25):.2f}")
        print(f"med: {np.median(times[idx]):.2f}")
        print(f"q3: {np.percentile(times[idx], 75):.2f}")
        print(f"max: {np.median(times[idx]):.2f}")
        print("")
