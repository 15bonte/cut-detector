""" An alternative to mid_body_detection and mb_detection
"""

import numpy as np

from math import sqrt
from typing import Tuple, Callable
from skimage.feature import blob_log, blob_dog, blob_doh
from cnn_framework.utils.readers.tiff_reader import TiffReader
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from cut_detector.utils.mid_body_spot import MidBodySpot
from cut_detector.utils.image_tools import smart_cropping

MIDBODY_CHAN = 1
SIR_CHAN = 0
SOURCE_DIR = "./src/cut_detector/data/mid_bodies_movies_test"
OUT_DIR = "./src/cut_detector/data/mid_bodies"

SHOULD_SAVE = True
SHOW_TRACKING = False

SOURCE_CHOICE = -1 
SOURCE_LIST = {
    ### Positive or null indices: 4 channels as usual ###
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

    ### Strictly negative indices: 3 channels only ###
    -1: "longcep1_20231019-t1_siCep55-50-1.tif",
}

CANDIDATE_SELECTION = [
    "cur_log", 
]
CANDIDATES = {
    "cur_log": (blob_log, {"min_sigma": 5, "max_sigma": 10, "num_sigma": 5, "threshold": 0.1}),
    "cur_dog": (blob_dog, {"min_sigma": 2, "max_sigma": 5, "sigma_ratio": 1.2, "threshold": 0.1}),
    "cur_doh": (blob_doh, {"min_sigma": 5, "max_sigma": 10, "num_sigma": 5, "threshold": 0.0040}),

    "centered_log":      (blob_log, {"min_sigma": 3, "max_sigma": 7, "num_sigma": 5, "threshold": 0.1}),
    "mini_centered_log": (blob_log, {"min_sigma": 3, "max_sigma": 7, "num_sigma": 3, "threshold": 0.1}),
    "off_centered_log":  (blob_log, {"min_sigma": 3, "max_sigma": 11, "num_sigma": 5, "threshold": 0.1}),

    "log2":       (blob_log, {"min_sigma": 2, "max_sigma": 4, "num_sigma": 3, "threshold": 0.1}),
    "log2_wider": (blob_log, {"min_sigma": 2, "max_sigma": 8, "num_sigma": 4, "threshold": 0.1}),

    "mini_centered_log_0050": (blob_log, {"min_sigma": 3, "max_sigma": 7, "num_sigma": 3, "threshold": 0.05}),
    "mini_centered_log_0075": (blob_log, {"min_sigma": 3, "max_sigma": 7, "num_sigma": 3, "threshold": 0.05}),
    "mini_centered_log_0090": (blob_log, {"min_sigma": 3, "max_sigma": 7, "num_sigma": 3, "threshold": 0.05}),

    "cur_log_0050": (blob_log, {"min_sigma": 5, "max_sigma": 10, "num_sigma": 5, "threshold": 0.05}),
    "cur_log_0075": (blob_log, {"min_sigma": 5, "max_sigma": 10, "num_sigma": 5, "threshold": 0.075}),
    "cur_log_0090": (blob_log, {"min_sigma": 5, "max_sigma": 10, "num_sigma": 5, "threshold": 0.09}),
}

def process_image(mklp, sir, fun_args: Tuple[Callable, dict]) -> np.array:
    fun  = fun_args[0]
    args = fun_args[1]
    max  = np.max(mklp)
    min  = np.min(mklp)
    mklp = (mklp-min) / (max-min)
    blobs_log = fun(image=mklp, **args)
    print("found blobs (y/x/s):", blobs_log, sep="\n")
    # Compute radii in the 3rd column, since 3 column is sigma
    # and radius can be approximated by sigma * sqrt(2) according to doc
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    return blobs_log

def main():
    filepath = make_filepath(SOURCE_DIR, SOURCE_LIST[SOURCE_CHOICE])
    is_three_channel = SOURCE_CHOICE < 0
    mitosis_movie = load_movie(filepath, is_three_channel)
    factory = MidBodyDetectionFactory()

    for candidate in CANDIDATE_SELECTION:
        out_dir = f"{OUT_DIR}/{candidate}"
        fun_args = CANDIDATES[candidate]
        spots_candidates = detect_spots(
            mitosis_movie=mitosis_movie,
            fun_args=fun_args
        )

        print("=== detection results ===")
        for frame, spots in spots_candidates.items():
            print(f"\nframe{frame}")
            for spot in spots:
                print(f"x:{spot.x} y:{spot.y} mklp:{spot.intensity} sir:{spot.sir_intensity}")

        factory.generate_tracks_from_spots(
            spots_candidates,
            tracking_method="spatial_laptrack",
            # tracking_method="laptrack",
            show_tracking=SHOW_TRACKING,
        )

        if SHOULD_SAVE:
            factory.save_mid_body_tracking(
                spots_candidates, mitosis_movie, out_dir
            )
    
def detect_spots(mitosis_movie: np.array, fun_args: Tuple[Callable, dict]):
    spots_dictionary = {}
    nb_frames = mitosis_movie.shape[0]
    for frame in range(nb_frames):
        print(f"\nprocesssing frame {frame+1}/{nb_frames}:")
        mitosis_frame = mitosis_movie[frame, :, :, :].squeeze()  # YXC

        spots = spot_detection(mitosis_frame, frame, fun_args)

        spots_dictionary[frame] = spots

    return spots_dictionary
    
def spot_detection(mitosis_frame: np.array, frame: int, fun_args: Tuple[Callable, dict]):
    image_sir = mitosis_frame[:, :, SIR_CHAN]
    image_mklp = mitosis_frame[:, :, MIDBODY_CHAN]

    spots = [
        (int(spot[0]), int(spot[1]))
        for spot in process_image(image_mklp, image_sir, fun_args)
    ]

    # Convert spots to MidBodySpot objects (switch (y, x) to (x, y))
    mid_body_spots = [
        MidBodySpot(
            frame,
            x=position[1],
            y=position[0],
            intensity=get_average_intensity(position, image_mklp),
            sir_intensity=get_average_intensity(position, image_sir),
        )
        for position in spots
    ]
    return mid_body_spots

def get_average_intensity(
        position: tuple[int], image: np.array, margin=1
    ) -> int:
        """
        Parameters
        ----------
        position: (y, x)
        image: YX
        margin: int

        Returns
        ----------
        average_intensity: int
        """
        # Get associated crop
        crop = smart_cropping(
            image,
            margin,
            position[1],
            position[0],
            position[1] + 1,
            position[0] + 1,
        )

        # Return average intensity
        return int(np.mean(crop))

def make_filepath(dir: str, filename: str) -> str:
    if dir[-1] == "/" or dir[-1] == "\\":
        return f"{dir}{filename}"
    else:
        return f"{dir}/{filename}"
    

def load_movie(path: str, is_three_channel: bool) -> np.array:
    if is_three_channel:
        # raise RuntimeError("WIP")
        image = TiffReader(path, respect_initial_type=True).image  # ZCTYX
        print("image shape:", image.shape)
        movie = image[:, :3, ...].squeeze()  # Z C=3 TYX -> C=3 TYX
        movie = movie.transpose(1, 2, 3, 0)  # CTYX -(1,2,3,0)> TYXC
        # raise RuntimeError("WIP")
        return movie # -> TYXC
    else:
        image = TiffReader(path, respect_initial_type=True).image  # TCZYX
        print("image shape:", image.shape)
        movie = image[:, :3, ...].squeeze()  # T C=3 YX
        movie = movie.transpose(0, 2, 3, 1)  # TYXC
        # raise RuntimeError("WIP")
        return movie # -> TYXC

if __name__ == "__main__":
    main()