from .source import Source

SRC_DIR = "src/cut_detector/data/mid_bodies_movies_test"

def src_dir(dir: str, filename: str, fmt: str) -> Source:
    from os.path import join
    p = join(dir, filename)
    return Source(p, fmt)

class SourceFiles:
    example  = src_dir(SRC_DIR, "example_video_mitosis_0_0_to_4.tiff",                "4c")
    a_siLuci = src_dir(SRC_DIR, "a_siLuci-1_mitosis_33_7_to_63.tiff",                 "4c")
    s1       = src_dir(SRC_DIR, "s1_siLuci-1_mitosis_14_158_to_227.tiff",             "4c")
    s2       = src_dir(SRC_DIR, "s2_siLuci-1_mitosis_15_67_to_228,211.tiff",          "4c")
    s3       = src_dir(SRC_DIR, "s3_siLuci-1_mitosis_17_170_to_195.tiff",             "4c")
    s4       = src_dir(SRC_DIR, "s4_siLuci-1_mitosis_24_128_to_135.tiff",             "4c")
    s5       = src_dir(SRC_DIR, "s5_siLuci-1_mitosis_27_22_to_93,87.tiff",            "4c")
    s6       = src_dir(SRC_DIR, "s6_siLuci-1_mitosis_28_50_to_91.tiff",               "4c")
    s7       = src_dir(SRC_DIR, "s7_siLuci-1_mitosis_31_19_to_73.tiff",               "4c")
    s8       = src_dir(SRC_DIR, "a_siLuci-1_mitosis_33_7_to_63.tiff",                 "4c")
    s9       = src_dir(SRC_DIR, "s9_siLuci-1_mitosis_34_21_to_68,62.tiff",            "4c")
    siCep    = src_dir(SRC_DIR, "20231019-t1_siCep55-50-4_mitosis_21_25_to_117.tiff", "4c")
    longCep1 = src_dir(SRC_DIR, "longcep1_20231019-t1_siCep55-50-1.tif",              "3c")

