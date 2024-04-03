from wrapper_env import WrapperEnv
from .bench import bench_detection_against_gt

def bench_detection_against_gt_runner():
    stats = bench_detection_against_gt(
        WrapperEnv.src_file,
        WrapperEnv.gt_filepath_from_source(WrapperEnv.src_file),
        WrapperEnv.main_spot_detection_method,
        ignore_false_positives=True
    )
    print("-- Results --")
    if stats.same_method_bench_gt:
        print("WARNING: ground truth has been generated with the same mode,")
        print("If you have not modified the file, the analysis will be biased")
    print("min distance:", stats.min())
    print("max distance:", stats.max())
    print("avg distance:", stats.avg())
    print("n miss:", stats.n_miss)