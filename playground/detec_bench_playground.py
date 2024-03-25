""" Test difference of Gaussian
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from typing import Callable, Tuple, Any
from skimage.feature import blob_dog, blob_log, blob_doh
from cut_detector.sandbox import load_movie, process_movie, plot_frame, min_max_norm
from cut_detector.sandbox.blob_bench import DetectBench

Bench = DetectBench()
doh_miss = 0
doh_fp = 0

MLKP_CHAN = 1 
DIR = "./src/cut_detector/data/mid_bodies_movies_test"

TF_CHOICE = 0
TEST_FILES = {
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

def diff_gauss(frame, filename, frame_idx):
    # diff_gauss_default(frame, filename, frame_idx)
    # diff_gauss_current(frame, filename, frame_idx)
    diff_gauss_current_bench(frame, filename, frame_idx)
    # diff_gauss_improve_doh_bench(frame, filename, frame_idx)
    # diff_gauss_doh_thresholds_vs_log(frame, filename, frame_idx)

def diff_gauss_default(frame, filename, frame_idx):
    mlkp_frame = frame[:,:,1]
    if frame_idx < 3: plot_frame(mlkp_frame, f"MKLP before norm - f{frame_idx}")
    img = min_max_norm(mlkp_frame)
    if frame_idx < 3: plot_frame(img, f"MKLP after norm - f{frame_idx}")
    print("log:", blob_log(img))
    print("dog", blob_dog(img))
    print("doh", blob_doh(img))
    input("[enter]")

def diff_gauss_current(frame, filename, frame_idx):
    mlkp_frame = frame[:,:,1]
    # if frame_idx < 3: plot_frame(mlkp_frame, f"MKLP before norm - f{frame_idx}")
    img = min_max_norm(mlkp_frame)
    # if frame_idx < 3: plot_frame(img, f"MKLP after norm - f{frame_idx}")
    print("log:", blob_log(img, 5, 10, 5, 0.1))
    print("dog", blob_dog(img, 5, 10, 1.2, 0.1))
    print("doh", blob_doh(img, 5, 10, 5, 0.0034))  #0.025 #0.025
    # DOH: 
    # 0.025: nothing
    # 0.020: around 50% similarity with log/dog
    # 0.015: around 85-90% similarity with log/dog
    # 0.010: around 98% similarity with log/dog
    # 0.005: around 98% similarity with log/dog,
    # 0.00475: around 98% similarity with log/dog,
    # 0.0045: around 98% similarity with log/doc + 1 FP
    # 0.004: around 98% similarity with log/dog + 1 FP
    # 0.0037: 2 miss / 2 FP
    # 0.0035: False positives (0 miss / 3FP)
    # 0.0025: False positives
    # 0.001: False positives
    # input("[enter]")

def diff_gauss_current_bench(frame, filename, frame_idx):
    global Bench
    if frame_idx == 0:
        Bench = DetectBench(
            filename,
        {
            "min_sigma": 5,
            "max_sigma": 10,
            "num_sigma": 5,
            "threshold": 0.1
        },{
            "min_sigma": 5,
            "max_sigma": 10,
            "sigma_ratio": 1.2,
            "threshold": 0.1,
        },{
            "min_sigma": 5,
            "max_sigma": 10,
            "num_sigma": 5,
            "threshold": 0.0025
        })
    mlkp_frame = frame[:,:,1]
    img = min_max_norm(mlkp_frame)
    Bench.bench_img(img)

def diff_gauss_improve_doh_bench(frame, filename, frame_idx):
    global Bench
    if frame_idx == 0:
        Bench = DetectBench(
            filename,
        {
            "min_sigma": 5,
            "max_sigma": 10,
            "num_sigma": 5,
            "threshold": 0.1
        },{
            "min_sigma": 5,
            "max_sigma": 10,
            "sigma_ratio": 1.2,
            "threshold": 0.1,
        },{
            "min_sigma": 5,
            "max_sigma": 10,
            "num_sigma": 5,
            "threshold": 0.0034
            # "threshold": 0.005,
        })
    mlkp_frame = frame[:,:,1]
    img = min_max_norm(mlkp_frame)
    Bench.bench_img(img)

def diff_gauss_doh_thresholds_vs_log(frame, filename, frame_idx):
    global doh_miss, doh_fp
    mlkp_frame = frame[:,:,1]
    img = min_max_norm(mlkp_frame)
    bl = blob_log(img, 5, 10, 5, 0.1)
    bh = blob_doh(img, 5, 10, 5, 0.0034)
    # DoH Thresholds:
    # LoG: 5 10 5 0.1
    # 0.025:           34 miss /  0 FP / 34E
    # 0.010 (default):  2 miss /  0 FP /  2E
    # 0.005:            2 miss /  0 FP /  2E
    # 0.00375:          2 miss /  1 FP /  3E
    # 0.00365:          2 miss /  2 FP /  4E
    # 0.003625:         2 miss /  3 FP /  5E
    # 0.0036:           1 miss /  3 FP /  4E
    # 0.00355:          0 miss /  3 FP /  3E
    # 0.0034:           0 miss /  3 FP /  3E
    # 0.00335:          0 miss /  3 FP /  3E
    # 0.003125:         0 miss /  6 FP /  6E
    # 0.0025:           0 miss /  7 FP /  7E
    # 0.001:            0 miss / 23 FP / 23E
    print("log:", bl)
    print("doh", bh) 
    
    if len(bh) == len(bl):
        pass
    elif len(bh) < len(bl):
        doh_miss += len(bl) - len(bh)
    elif len(bh) > len(bl):
        doh_fp += len(bh) - len(bl)
    else:
        print("[unknown case]")
    


def main():
    print("run")
    filename = TEST_FILES[TF_CHOICE]
    movie, mask = load_movie(DIR, filename)
    process_movie(movie, filename, diff_gauss)
    Bench.print_all()
    print("####")
    print("doh_miss:", doh_miss)
    print("doh_fp", doh_fp)

if __name__ == "__main__":
    main()