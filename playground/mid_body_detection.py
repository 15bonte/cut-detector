import os
from time import time
from typing import Optional

import numpy as np 
from aicsimageio import AICSImage

from cut_detector.data.tools import get_data_path
from cut_detector.factories.mid_body_detection_factory import (
    MidBodyDetectionFactory,
)
from cut_detector.utils.mb_support import detection, tracking
from cut_detector.utils.gen_track import generate_tracks_from_spots

MM_DIR   = "./src/cut_detector/data/mitosis_movies/"
MBMT_DIR = "./src/cut_detector/data/mid_bodies_movies_test/"
STD      = "./eval_data/Data Standard/movies/"
SPAS     = "./eval_data/Data spastin/movies/"
CEP      = "./eval_data/Data cep55/movies/"

SOURCE_CHOICE = 22
SOURCES = {
    0:  MM_DIR   + "example_video_mitosis_0_0_to_4.tiff",
    1:  MBMT_DIR + "a_siLuci-1_mitosis_33_7_to_63.tiff",
    5:  MBMT_DIR + "s1_siLuci-1_mitosis_14_158_to_227.tiff",
    6:  MBMT_DIR + "s2_siLuci-1_mitosis_15_67_to_228,211.tiff",
    4:  MBMT_DIR + "s3_siLuci-1_mitosis_17_170_to_195.tiff",
    7:  MBMT_DIR + "s4_siLuci-1_mitosis_24_128_to_135.tiff",
    8:  MBMT_DIR + "s5_siLuci-1_mitosis_27_22_to_93,87.tiff",
    9:  MBMT_DIR + "s6_siLuci-1_mitosis_28_50_to_91.tiff",
    10: MBMT_DIR + "s7_siLuci-1_mitosis_31_19_to_73.tiff",
    11: MBMT_DIR + "s9_siLuci-1_mitosis_34_21_to_68,62.tiff",
    12: MBMT_DIR + "20231019-t1_siCep55-50-4_mitosis_21_25_to_117.tiff",
    13: MBMT_DIR + "cep2_20231019-t1_siCep55-50-4_mitosis_24_17_to_104.tiff",
    14: MBMT_DIR + "cep_1.tiff",
    15: MBMT_DIR + "example_video_mitosis_0_0_to_4.tiff",

    16: STD + "converted t2_t3_F-1E5-35-12_mitosis_29_10_to_37.tiff",
    17: STD + "converted t2_t3_F-1E5-35-12_mitosis_4_13_to_189.tiff",
    18: STD + "converted t2_t3_F-1E5-35-8_mitosis_35_17_to_21.tiff",
    19: STD + "converted t2_t3_F-1E5-35-7_mitosis_12_14_to_36.tiff",
    20: STD + "converted t2_t3_F-1E5-35-11_mitosis_25_6_to_49.tiff",
    21: STD + "converted t2_t3_F-1E5-35-8_mitosis_9_95_to_196.tiff",
    22: STD + "converted t2_t3_F-1E5-35-7_mitosis_10_15_to_51.tiff",
    23: STD + "converted t2_t3_F-1E5-35-13_mitosis_17_0_to_34.tiff",
    24: STD + "converted t2_t3_F-1E5-35-11_mitosis_15_71_to_125.tiff",
    25: STD + "converted t2_t3_F-1E5-35-11_mitosis_27_2_to_42.tiff",

    26: SPAS + "20231019-t1_siSpastin-50-2_mitosis_5_136_to_176.tiff",
    27: SPAS + "20231019-t1_siSpastin-50-1_mitosis_16_5_to_25.tiff",
    28: SPAS + "20231019-t1_siSpastin-50-2_mitosis_29_17_to_47.tiff",

    29: CEP + "20231019-t1_siCep55-50-4_mitosis_5_8_to_228.tiff",
    30: CEP + "20231019-t1_siCep55-50-4_mitosis_24_17_to_104.tiff",
    31: CEP + "20231019-t1_siCep55-50-1_mitosis_5_41_to_155.tiff",
}

D_METHOD = detection.cur_dog
T_METHOD = tracking.cur_spatial_laptrack

MEASURE_DETECTION_TIME = True
SHOW_POINTS            = False
SHOW_TRACKS            = False
SHOULD_SAVE            = True
PARALLELIZE            = True

DETECTION_STEP_COUNT = 1 

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
    image = read_tiff(image_path) # TCZYX

    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC

    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    # Search for mid-body in mitosis movie
    factory = MidBodyDetectionFactory()

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

def read_tiff(path: str) -> np.ndarray:
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
    main(
        SOURCES[SOURCE_CHOICE]
    )
