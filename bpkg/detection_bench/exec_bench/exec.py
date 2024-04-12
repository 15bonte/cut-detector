from typing import Optional

from data_loading import Source
from detection_gt import bench_detection_against_gt, BenchStat
from ..bench_config import BenchConfig, DetectionConfig


def exec_bench(
        config: BenchConfig, 
        on_sources: list[Source],
        against_gt: list[str],
        ignore_false_positives: bool,
        measure_time: bool
        ) -> dict[str, dict[str, BenchStat]]:
    """ 
    Benchmarks the benchconfig on the given source files, with the associated
    ground truths files.
    """

    all_results = {}
    for (source, gt) in zip(on_sources, against_gt):
        print("######")
        print("source:", source)
        print("gt:", gt)
        single_source_results = {}
        for det_name, det_conf in config.detections.items():
            print("\n--- pipeline:", det_name, "----")
            stat = single_bench(source, gt, det_conf, ignore_false_positives, measure_time)
            single_source_results[det_name] = stat
        all_results[source.path] = single_source_results

    return all_results
    

def single_bench(
        source: Source, 
        gt_filepath: str, 
        det_config: DetectionConfig,
        ignore_false_positives: bool,
        measure_time: bool
        ) -> BenchStat:
    
    detector = det_config.make_associated_detector()
    
    return bench_detection_against_gt(
        source,
        gt_filepath,
        detector,
        ignore_false_positives,
        measure_time
    )