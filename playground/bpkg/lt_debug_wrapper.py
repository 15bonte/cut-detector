import numpy as np

import sys
from os.path import join
from scipy.spatial import distance
from .laptrack_debug import start_debug_run, SpatialLaptrackDebug

SOURCE_DIR = "./src/cut_detector/data/mid_bodies_movies_test"
LOG_OUT_DIR = "./src/cut_detector/data/mid_bodies"
OUT_DIR = "./src/cut_detector/data/mid_bodies"
OUT_DIR = None

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

LOG_FILE = "log.txt"
# LOG_FILE = None

# weight_mklp_intensity_factor=5.0,
#         weight_sir_intensity_factor=1.50,
#         # weight_sir_intensity_factor=15,
#         mid_body_linking_max_distance=175,

# lt = SpatialLapTrack(
#                 spatial_coord_slice=slice(0,2),
#                 spatial_metric="euclidean",
#                 track_dist_metric=dist_metric,
#                 track_cost_cutoff=max_distance,
#                 gap_closing_dist_metric=dist_metric,
#                 gap_closing_cost_cutoff=max_distance,
#                 gap_closing_max_frame_count=3,
#                 splitting_cost_cutoff=False,
#                 merging_cost_cutoff=False,
#                 # alternative_cost_percentile=1,
#                 alternative_cost_percentile=100,  # modified value
#                 # alternative_cost_percentile=90, # default value
#             )

# MAX_LINKING_DISTANCE = 175
MAX_LINKING_DISTANCE = 100
MKLP_WEIGHT = 5.0
SIR_WEIGHT = 1.50
def dist_metric(c1, c2):
    """Modified version of sqeuclidian distance
    Square Euclidian distance is applied to spatial coordinates
    x and y.
    while an 'intensity' distance is computed with MLKP and
    SIR intensities
    Finally values are combined by weighted addition
    """
    # unwrapping
    (x1, y1, mlkp1, sir1), (x2, y2, mlkp2, sir2) = c1, c2
    if np.isnan([x1, y1, x2, y2]).any():
        return MAX_LINKING_DISTANCE*2 
    
    spatial_e = distance.euclidean([x1, y1], [x2, y2])
    mkpl_penalty = (
        3
        * MKLP_WEIGHT
        * np.abs(mlkp1 - mlkp2) / (mlkp1 + mlkp2)
    )
    sir_penalty = (
        3
        * SIR_WEIGHT
        * np.abs(sir1 - sir2) / (sir1 + sir2)
    )
    penalty = (
        1
        + sir_penalty
        + mkpl_penalty
    )
    return (spatial_e * penalty)**2 

def start_wrapped_lt_run():
    global LogTarget

    if SOURCE_CHOICE < 0:
        movie_kind = "3c"
    else:
        movie_kind = "4c"

    lt = SpatialLaptrackDebug(
        spatial_coord_slice=slice(0,2),
        spatial_metric="euclidean",
        track_dist_metric=dist_metric,
        track_cost_cutoff=MAX_LINKING_DISTANCE,
        gap_closing_dist_metric=dist_metric,
        gap_closing_cost_cutoff=MAX_LINKING_DISTANCE,
        gap_closing_max_frame_count=3,
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
        alternative_cost_percentile=100,

        show_predict_link_debug=True,
        show_gap_closing_debug=True,
    )

    # if isinstance(LOG_FILE, str):
    #     log_path = join(OUT_DIR, LOG_FILE)
    #     print("logging to file: ", log_path)
    #     original_stdout = sys.stdout
    #     with open(log_path, "w") as f:
    #         sys.stdout = f
    #         start_debug_run(
    #             dir=SOURCE_DIR,
    #             filename=SOURCE_LIST[SOURCE_CHOICE],
    #             movie_kind=movie_kind,
    #             detection_method="lapgau",
    #             laptrack_method=lt,
    #             out_dir=OUT_DIR
    #         )
    #         sys.stdout = original_stdout

    # else:
    #     start_debug_run(
    #         dir=SOURCE_DIR,
    #         filename=SOURCE_LIST[SOURCE_CHOICE],
    #         movie_kind=movie_kind,
    #         detection_method="lapgau",
    #         laptrack_method=lt,
    #         out_dir=OUT_DIR
    #     )
    if isinstance(LOG_FILE, str):
        log_fp = join(LOG_OUT_DIR, LOG_FILE)
    else:
        log_fp = None
    start_debug_run(
        dir=SOURCE_DIR,
        filename=SOURCE_LIST[SOURCE_CHOICE],
        movie_kind=movie_kind,
        detection_method="lapgau",
        laptrack_method=lt,
        show_tracking=False,
        out_dir=OUT_DIR,
        log_fp=log_fp
    )

    