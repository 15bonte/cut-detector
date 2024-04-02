from os.path import join
from pathlib import Path
import json
from bpkg.data_loader import Movie
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory

SOURCE_DIR = "./src/cut_detector/data/mid_bodies_movies_test"
OUT_DIR = "./src/cut_detector/data/mid_bodies_movies_test/gt"

DETECTION_MODE = "lapgau"

SOURCE_CHOICE = 0
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

def generate_ground_truth(filepath: str, fmt: str, detection_mode: str, out_file: str):
    """ Generates a ground truth from a pipeline
    """
    print("loading", filepath, "for ground truth generation")
    movie = Movie(filepath, fmt)
    factory = MidBodyDetectionFactory()
    spots_dict = factory.detect_mid_body_spots(movie.movie, mode=detection_mode)
    spots_idx_list = list(spots_dict.keys())
    spots_idx_list.sort()
    max_frame_idx = spots_idx_list[-1]
    output = {
        "file": filepath,
        "detection_mode": detection_mode,
        "max_frame_idx": max_frame_idx,
        "spots": {k: [{"x": v.x, "y": v.y} for v in spots_dict[k]] for k in spots_dict}
    }

    print("dumping ground truth to:", out_file)
    with open(out_file, "w") as file:
        json.dump(output, file, indent=2)
    

def run_generate_ground_truth():
    filepath = join(SOURCE_DIR, SOURCE_LIST[SOURCE_CHOICE])
    source_fp = Path(filepath)
    if SOURCE_CHOICE >= 0:
        fmt = "4c"
    else:
        fmt = "3c"
    
    out_fp = join(OUT_DIR, f"{source_fp.stem}_gt.json")
    generate_ground_truth(filepath, fmt, DETECTION_MODE, out_fp)

