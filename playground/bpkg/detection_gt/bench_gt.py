""" Benching pipelines against ground truths
"""

from os.path import join
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from bpkg.data_loader import Movie
from .loading import load_gt
from .bench_result import BenchResult


SOURCE_DIR = "./src/cut_detector/data/mid_bodies_movies_test"
GT_DIR = "./src/cut_detector/data/mid_bodies_movies_test/gt"

DETECTION_CHOICE = 4
DETECTION_MODE_LIST: MidBodyDetectionFactory.SPOT_DETECTION_MODE = {
    0: "lapgau",
    1: "log2_wider",
    2: "off_centered_log",
    3: "diffgau",
    4: "hessian"
}

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

def bench_detection(
        src_fp: str, 
        fmt: str,
        gt_fp: str, 
        detection_mode: str,
        allow_false_positives: bool = True):
    if not allow_false_positives:
        raise RuntimeError("disallowng false positives is not allowed yet")

    movie = Movie(src_fp, fmt)
    
    gt = load_gt(gt_fp)
    if detection_mode == gt.detection_mode:
        print("WARNING: ground truth has been generated with the same mode:", detection_mode)
    
    factory = MidBodyDetectionFactory()
    spots = factory.detect_mid_body_spots(movie.movie, mode=detection_mode)

    n_miss = 0
    distances = []

    for gt_frame, gt_spots in enumerate(gt.points):
        if len(gt_spots) != 0:
            gt_spots_raw = np.array([[s.x, s.y] for s in gt_spots])
            test_spots_raw = np.array([[s.x, s.y] for s in spots.get(gt_frame, [])])
            if len(test_spots_raw.shape) == 1: # no spots here: empty array
                n_miss += 1
            else:
                dists = cdist(gt_spots_raw, test_spots_raw, "euclidean")
                for l in range(dists.shape[0]):
                    best = dists[l].min()
                    distances.append(best)

    distances = np.array(distances)
    
    d_min = distances.min()
    d_max = distances.max()
    d_mean = distances.mean()

    return BenchResult(
        min_dist=d_min, 
        max_dist=d_max, 
        avg_dist=d_mean,
        n_miss=n_miss,
        same_method_bench_gt=(detection_mode == gt.detection_mode)
    )



def run_bench_detection():
    src_fp = join(SOURCE_DIR, SOURCE_LIST[SOURCE_CHOICE])
    src_fp_path = Path(src_fp)
    if SOURCE_CHOICE >= 0:
        fmt = "4c"
    else:
        fmt = "3c"
    gt_fp = join(GT_DIR, f"{src_fp_path.stem}_gt.json")
    detection_mode = DETECTION_MODE_LIST[DETECTION_CHOICE]
    result = bench_detection(src_fp, fmt, gt_fp, detection_mode)
    print("-- Result --")
    if result.same_method_bench_gt:
        print("/!\\ WARNING: same mode used for benchmark and gt generation")
    print("min distance:", result.min_dist)
    print("max distance:", result.max_dist)
    print("avg distance:", result.avg_dist)
    print("n miss:", result.n_miss)
