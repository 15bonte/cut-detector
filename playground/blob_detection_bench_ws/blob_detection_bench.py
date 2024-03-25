""" Better version of detec_bench_playground
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from typing import Callable, Tuple, Any
from skimage.feature import blob_dog, blob_log, blob_doh
from cut_detector.sandbox import process_movie, plot_frame, min_max_norm

# from playground.blob_detection_bench_ws.minibencher import MiniBencher
# from playground.blob_detection_bench_ws.test_data_loading import load_movie

# from blob_detection_bench_ws.minibencher import MiniBencher
# from blob_detection_bench_ws.test_data_loading import load_movie

from minibencher import MiniBencher
from test_data_loading import load_movie

TEST_DIR = "./src/cut_detector/data/mid_bodies_movies_test"

FILE_CHOICE = 3 # index or slice
TEST_FILE_POOL = {
    #### 4 Channels ####
    0: "example_video_mitosis_0_0_to_4.tiff",
    1: "a_siLuci-1_mitosis_33_7_to_63.tiff",
    2: "s1_siLuci-1_mitosis_14_158_to_227.tiff",
    3: "s2_siLuci-1_mitosis_15_67_to_228,211.tiff",
    4: "s3_siLuci-1_mitosis_17_170_to_195.tiff",
    5: "s4_siLuci-1_mitosis_24_128_to_135.tiff",
    6: "s5_siLuci-1_mitosis_27_22_to_93,87.tiff",
    7: "s6_siLuci-1_mitosis_28_50_to_91.tiff",
    8: "s7_siLuci-1_mitosis_31_19_to_73.tiff",
    9: "s9_siLuci-1_mitosis_34_21_to_68,62.tiff",
    10: "20231019-t1_siCep55-50-4_mitosis_21_25_to_117.tiff",
}

# cur == current <=> as it is written in the factory
REFERENCE = "log_cur"
CANDIDATES = ["doh_005", "doh_0025", "doh_0010", "doh_0005", "doh_00040", "doh_00034", "doh_00025", "doh_00017", "doh_00014", "doh_0001"]
TEST_PARAMETERS = {
    "log_cur":     (blob_log, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.1}),

    "dog_cur":        (blob_dog, {"min_sigma": 2, "max_sigma":   5, "sigma_ratio": 1.2, "threshold": 0.1}),
    "dog_playground": (blob_dog, {"min_sigma": 5, "max_sigma":  10, "sigma_ratio": 1.2, "threshold": 0.1}),

    "doh_default": (blob_doh, {"min_sigma": 1, "max_sigma": 30, "num_sigma":    10, "threshold": 0.1}),
    "doh_loglike": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.1}),
    
    "doh_005":  (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.05}),
    "doh_0025": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.025}),
    "doh_0010": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.010}),
    "doh_0005": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.005}),
    "doh_00040": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.0040}),
    "doh_00034": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.0034}),
    "doh_00025": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.0025}),
    "doh_00017": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.0017}),
    "doh_00014": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.0014}),
    "doh_0001": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.001}),
}

SHOULD_PRINT = True
SHOULD_TIME = False



def main():
    movie = load_movie(TEST_DIR, TEST_FILE_POOL[FILE_CHOICE])
    nb_frames = movie.shape[0]

    ref_name = REFERENCE
    ref_funargs = TEST_PARAMETERS[ref_name]

    candidates = {k: TEST_PARAMETERS[k] for k in CANDIDATES}

    bencher = MiniBencher(
        (ref_name, ref_funargs),
        candidates,
        measure_time=SHOULD_TIME,
        print_blobs=SHOULD_PRINT
    )

    for frame in range(nb_frames):
        print(f"\nprocessing frame {frame+1}/{nb_frames}")
        mitosis_frame = movie[frame, :, :, :].squeeze()  # YXC

        mlkp = mitosis_frame[:, :, 1]
        min  = np.min(mlkp)
        max  = np.max(mlkp)
        mlkp = (mlkp-min) / (max-min)

        # plot_frame(mlkp, "test")

        bencher.bench_frame(mlkp)

    bencher.print_results(image_src=TEST_FILE_POOL[FILE_CHOICE])

if __name__ == "__main__":
    main()