""" Mb Detection is an alternative to mid_body_detection.
Created because the newer videos have a different structure
(3 channels instead of 4).
"""

import os
from typing import Optional
from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector.data.tools import get_data_path
from cut_detector.factories.mid_body_detection_factory import (
    MidBodyDetectionFactory,
)

# "./src/cut_detector/data/mitosis_movies/example_video_mitosis_0_0_to_4.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/a_siLuci-1_mitosis_33_7_to_63.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/s1_siLuci-1_mitosis_14_158_to_227.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/s2_siLuci-1_mitosis_15_67_to_228,211.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/s3_siLuci-1_mitosis_17_170_to_195.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/s4_siLuci-1_mitosis_24_128_to_135.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/s5_siLuci-1_mitosis_27_22_to_93,87.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/s6_siLuci-1_mitosis_28_50_to_91.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/s7_siLuci-1_mitosis_31_19_to_73.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/s9_siLuci-1_mitosis_34_21_to_68,62.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/cep_1.tiff",
# "./src/cut_detector/data/mid_bodies_movies_test/example_video_mitosis_0_0_to_4.tiff",

""" Dict of all source files,
Set choice to the correct key, to choose which file to process.
(We are using a dictionnary because it is easy to just look at the index).
"""
SRC_CHOICE = 9
SOURCES = {
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
    #### 3 Channels ####
    10: "cep_1.tiff",
}

TIFF_TYPE_CHOICE: 1
TIFF_TYPE = {0: "4 Channels", 1: "3 Channels"}

"""The directory containing the source file. It is prepended to the source name"""
SOURCE_DIR = "./src/cut_detector/data/mid_bodies_movies_test"

"""The directory containing the result files. It is prepended to the result filename"""
OUTPUT_DIR = "./src/cut_detector/data/mid_bodies"


def mk_filepath(dir: str, filename: str) -> str:
    """Makes a file path from a name and a directory.
    Removes unecessary '/' at the end of dir, if any"""
    if dir[-1] == "/":
        return f"{dir}{filename}"
    else:
        return f"{dir}/{filename}"


def main():
    image_path = mk_filepath()
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX

    pass


# origine des points en haut à gauche, D>G, H>B
# autre test
def main(
    image_path: Optional[str] = get_data_path("mitosis_movies"),
    path_output: Optional[str] = get_data_path("mid_bodies"),
):
    # print(image_path)

    # If image_path is a directory, take its first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])
    print(image_path)

    # Read image
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX

    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC

    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    # Search for mid-body in mitosis movie
    factory = MidBodyDetectionFactory()

    # dict[int, list[MidBodySpot]]
    # int: frame
    spots_candidates = factory.detect_mid_body_spots(
        # mitosis_movie=mitosis_movie, mask_movie=mask_movie, mode="h_maxima"
        mitosis_movie=mitosis_movie,
        mask_movie=mask_movie,
        mode="lapgau",
    )  # mode = "bigfish" or "h_maxima" (default)

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

    factory.generate_tracks_from_spots(
        spots_candidates,
    )
    factory.save_mid_body_tracking(
        spots_candidates, mitosis_movie, path_output
    )


if __name__ == "__main__":
    # main()
    main(
        # "./src/cut_detector/data/mitosis_movies/example_video_mitosis_0_0_to_4.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/a_siLuci-1_mitosis_33_7_to_63.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/s1_siLuci-1_mitosis_14_158_to_227.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/s2_siLuci-1_mitosis_15_67_to_228,211.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/s3_siLuci-1_mitosis_17_170_to_195.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/s4_siLuci-1_mitosis_24_128_to_135.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/s5_siLuci-1_mitosis_27_22_to_93,87.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/s6_siLuci-1_mitosis_28_50_to_91.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/s7_siLuci-1_mitosis_31_19_to_73.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/s9_siLuci-1_mitosis_34_21_to_68,62.tiff",
        "./src/cut_detector/data/mid_bodies_movies_test/cep_1.tiff",
        # "./src/cut_detector/data/mid_bodies_movies_test/example_video_mitosis_0_0_to_4.tiff",
        # get_data_path("mid_bodies_tests")
    )
