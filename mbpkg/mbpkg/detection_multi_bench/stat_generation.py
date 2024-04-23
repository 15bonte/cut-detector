from typing import Union

from mbpkg.movie_loading import Movie
from mbpkg.detection_truth import DetectionGT, load_gt
from mbpkg.better_detector import Detector
from mbpkg.detection_stat import generate_detection_stats

from .multi_bench_stat import MultiBenchStat

def generate_multi_bench_stat(
        movie: Movie, 
        detectors: list[Detector],
        ground_truth: Union[str, DetectionGT],
        parallelization: bool,
        ) -> MultiBenchStat:
    
    if isinstance(ground_truth, str):
        gt = load_gt(ground_truth)
    elif isinstance(ground_truth, DetectionGT):
        gt = ground_truth
    else:
        raise RuntimeError(f"Invalid ground truth type: {type(ground_truth)}")

    results = {
        d: generate_detection_stats(movie, d, gt, parallelization) 
        for d in detectors
    }

    return MultiBenchStat(
        movie.path,
        results
    )

