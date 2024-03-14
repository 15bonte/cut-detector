""" Evaluation is an evaluation field from blob_log progress.
The idea is to rewrite and reoganize what was inside
- mid_body_log_sandbox (fully)
- fake_blob_log (to some extent, unknown for now)

Note: this file will be reorganized later, because it may scale a lot

---

How to use it:
Constants are defined first.
Then 2 important constants are defined:
- TEST_ENV: tuples of path to source and to ground truth
- TESTS: list of tests to run, tests are just dictionnary of values that
  define key events.
"""

################################################################################
############################ Evaluation Constants ##############################
################################################################################
# path to the test file, needs 4 channels
TFRAME_SOURCE_FILE_PATH  = "./src/cut_detector/data/mitosis_movies/example_video_mitosis_t28-30.tiff"
TFRAME_GROUND_TRUTH_PATH = None
# TFRAME_GROUND_TRUTH_PATH = "./src/cut_detector/data/mitosis_movies/example_video_mitosis_t28-30_truth.json"

# Should the plots be displayed at the end (usually yes)
PLOT_SHOW_AT_END = True

# Midbody and Sir Channels
MB_CHAN  = 1
SIR_CHAN = 0

# The simple binary test. A hard-coded but relatively generous threshold and AC/OC
SIMPLE_BIN_TEST = {
    "name": "Simple Binary Test",
    "normalize": None,
    "binarize": 135,
    "area_op": 50,
    "area_cl": 200,
    "method": {
        "kind": "lapgau",
        "mSig": 3,
        "MSig": 6,
        "nSig": 30,
        "threshold": .1
    },
    "debug": [
        "plot_cubes",
        "blobs"
    ]
}

RAW_TEST = {
    "name": "Raw Test",
    "method": {
        "kind": "lapgau",
        "mSig": 5,
        "MSig": 10,
        "nSig": 30,
        "threshold": .1
    },
    "debug": [
        # "raw",
        # "norm",
        # "bin",
        # "aop",
        # "acl",
        # "plot_cubes",
        "blobs"
    ]
}

RAW_NORM_TEST = {
    "name": "Raw Normalized Test",
    "normalize": "max",
    "method": {
        "kind": "lapgau",
        "mSig": 5,
        "MSig": 10,
        "nSig": 30,
        "threshold": .1
    },
    "debug": [
        # "raw",
        # "norm",
        # "bin",
        # "aop",
        # "acl",
        # "plot_cubes",
        "blobs"
    ]
}

################################################################################
TEST_ENV = [
    {"src": TFRAME_SOURCE_FILE_PATH, "truth": TFRAME_GROUND_TRUTH_PATH}
]

TESTS = [
    SIMPLE_BIN_TEST,
    # RAW_TEST,
    RAW_NORM_TEST,
]
################################################################################


################################################################################
################################################################################
################################################################################


import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import area_closing, area_opening
from skimage.feature import blob_log
from cnn_framework.utils.readers.tiff_reader import TiffReader

def main():
    for env in TEST_ENV:
        print(f"### Running test environment ###")
        print("src:", env["src"])
        print("truth:", env["truth"])
        print("################################\n\n")
        for test in TESTS:
            test_name = test["name"]
            print(f"=== Running Test {test_name} ===\n")
            mid_body_log_sandbox(env["src"], test)


def mid_body_log_sandbox(image_path: str, test):
    # If image_path is a directory, take its first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])

    # Read movie
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX
    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC
    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    # Start processing it
    detect_movie_spots(mitosis_movie, test)


def detect_movie_spots(movie: np.array, test_param: dict):
    n_frames = movie.shape[0]

    debug_param = test_param.get("debug", [])
    for k in debug_param:
        if not isinstance(k, str):
            print("debug list must only contains string, found:", k, file=sys.stderr)

    for frame in range(n_frames):
        print(f"processing frame {frame}:")
        mitosis_frame = movie[frame, :, :, :].squeeze()  # YXC
        image_sir = mitosis_frame[:, :, SIR_CHAN]
        mklp_frame = mitosis_frame[:, :, MB_CHAN]

        #### preprocessing
        ## Raw debug ?
        debug_image(mklp_frame, debug_param, "raw", f"Raw image frame {frame}")

        ## Normalization
        norm_param = test_param.get("normalize")
        if norm_param is None:
            pass
        elif norm_param == "max":
            # mklp_frame /= np.max(mklp_frame)
            # raw_image = mklp_frame / np.max(mklp_frame)
            mklp_frame = mklp_frame / np.max(mklp_frame)
        elif frame == 0: 
            print("unknown value for parameter 'normalize'", file=sys.stderr)
        debug_image(mklp_frame, debug_param, "norm", f"Normalized image frame {frame}")

        ## Binarization
        bin_param = test_param.get("binarize")
        if bin_param is None:
            pass
        elif isinstance(bin_param, int):
            mklp_frame = mklp_frame > bin_param
        elif frame == 0:
            print("unknown value for parameter 'binarize'", file=sys.stderr)
        debug_image(mklp_frame, debug_param, "bin", f"Binarized image frame {frame}")

        ## Area Opening
        area_op = test_param.get("area_op")
        if area_op is None:
            pass
        elif isinstance(area_op, int):
            mklp_frame = area_opening(mklp_frame, area_op)
        elif frame == 0:
            print("unknown value for parameter 'area_op'", file=sys.stderr)
        debug_image(mklp_frame, debug_param, "aop", f"AreaOp image frame {frame}")

        ## Area Closing
        area_cl = test_param.get("area_cl")
        if area_cl is None:
            pass
        elif isinstance(area_cl, int):
            mklp_frame = area_closing(mklp_frame, area_cl)
        elif frame == 0:
            print("unknown value for parameter 'area_cl'", file=sys.stderr)
        debug_image(mklp_frame, debug_param, "aoc", f"AreaCl image frame {frame}")

        ## Method
        method = test_param["method"]
        if method["kind"] == "lapgau":
            min_sig = method["mSig"]
            max_sig = method["MSig"]
            n_sig = method["nSig"]
            threshold = method["threshold"]
            blobs = blob_log(
                mklp_frame, 
                min_sigma=min_sig,
                max_sigma=max_sig,
                num_sigma=n_sig,
                threshold=threshold
            )
            if "blobs" in debug_param: print("found blobs (y/x/s):", blobs)
        elif frame == 0:
            print(f"unknown method name '{method}'", file=sys.stderr)

        
    plt.show()

def debug_image(img: np.array, debug_key: str, expected_keys: list[str], ttl: str):
    if expected_keys in debug_key:
        plt.figure()
        plt.title(ttl)
        plt.imshow(img)


if __name__ == "__main__":
    main()