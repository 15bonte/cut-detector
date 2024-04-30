""" A simple binary that allows one to run detection benching on a single
detector or between several detectors
"""

from mbpkg.detection_multi_bench import generate_multi_bench_stat
from bin_env import SourceFiles, Detectors, StrDetectors
from bin_env import get_associated_gt_path, make_log_file, make_data_path

SOURCE = SourceFiles.a_siLuci
GT = get_associated_gt_path(SOURCE)
LOG_FP = make_log_file("mutibench_finale_a_siLuci.txt")
DATA_FP = make_data_path("multibench_finale_a_siLuci.json")

DETECTORS = [
    Detectors.cur_log,
    Detectors.cur_dog,
    Detectors.cur_doh,

    Detectors.log2_wider,
    Detectors.rshift_log,

    Detectors.very_fast_dog,
    Detectors.dog_005,
]

PARALLELIZATION = True


def run_detection_bench():
    print("--- running pipeline bench ----")
    print("source:", SOURCE)
    print("GT:", GT)
    print("LOG_FP:", LOG_FP)
    print("DATA_FP:", DATA_FP)
    for idx, d in enumerate(DETECTORS):
        print(f"Detector {idx+1}:", d)
    print("\n")

    r = generate_multi_bench_stat(
        SOURCE.load_movie(),
        DETECTORS,
        GT,
        PARALLELIZATION
    )
    r.report(LOG_FP, True)
    r.write_result(DATA_FP)

