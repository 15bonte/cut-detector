import sys
from ..exec_bench import exec_bench
from ..bench_config import load_config
from wrapper_env import WrapperEnv
from detection_gt import BenchStat

def exec_runner():
    config = load_config(
        WrapperEnv.get_config_file_path("detectors.json")
    )
    gt_files = [WrapperEnv.gt_filepath_for_source(f) for f in WrapperEnv.bench_src_files]
    result = exec_bench(
        config,
        WrapperEnv.bench_src_files,
        gt_files,
        ignore_false_positives=True,
    )
    print_result(result)

    with open(WrapperEnv.log_filepath_for_purpose("d_bench_result.txt"), "w") as file:
        original_stdout = sys.stdout
        sys.stdout = file
        print_result(result)
        sys.stdout = original_stdout
    

def print_result(result: dict[str, dict[str, BenchStat]]):
    print("\n\n\n")
    print("==== Bench Results ====")
    for path in result:
        print(f"    File '{path}'    ")
        for detec_name, stat in result[path].items():
            print("Pipeline:", detec_name)
            print("n miss:", stat.n_miss)
            print("")
            print("min:", stat.min())
            print("max:", stat.max())
            print("med:", stat.median())
            print("")
            print("")
