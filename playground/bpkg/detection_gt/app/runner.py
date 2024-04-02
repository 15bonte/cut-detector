""" Allows one to start the app with default parameters
"""
from os.path import join
from pathlib import Path
from .app import start_app

SOURCE_DIR = "./src/cut_detector/data/mid_bodies_movies_test"
OUT_DIR = "./src/cut_detector/data/mid_bodies_movies_test/gt"

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

def run_app_default():
    src_fp = join(SOURCE_DIR, SOURCE_LIST[SOURCE_CHOICE])
    src_path = Path(src_fp)
    out_fp = join(OUT_DIR, f"{src_path.stem}_gt.json")
    start_app(src_fp, out_fp)
