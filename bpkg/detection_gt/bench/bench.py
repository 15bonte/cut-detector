import numpy as np
from scipy.spatial.distance import cdist
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from importation import Source
from ..data import load_gt_file
from .bench_stat import BenchStat

def bench_detection_against_gt(
        src: Source, 
        gt_filepath: str, 
        detection_method: MidBodyDetectionFactory.SPOT_DETECTION_MODE,
        ignore_false_positives: bool = True):
    if not ignore_false_positives:
        raise RuntimeError("stats on false positives are not implemented yet")

    movie_data = src.load()
    gt = load_gt_file(gt_filepath)
    if detection_method == gt.detection_method:
        print("WARNING: ground truth has been generated with the same mode:", detection_method)
        print("If you have not modified the file, the analysis will be biased")
    
    factory = MidBodyDetectionFactory()
    spots = factory.detect_mid_body_spots(movie_data, mode=detection_method)

    n_miss = 0
    distances = []

    for gt_frame, gt_spots in enumerate(gt.points):
        if len(gt_spots) != 0:
            gt_spots_raw = np.array([[s.x, s.y] for s in gt_spots])
            test_spots_raw = np.array([[s.x, s.y] for s in spots.get(gt_frame, [])])
            if len(test_spots_raw.shape) == 1: # no spots here: empty array
                # n_miss += 1
                n_miss += len(gt_spots)
            else:
                dists = cdist(gt_spots_raw, test_spots_raw, "euclidean")
                for l in range(dists.shape[0]):
                    best = dists[l].min()
                    distances.append(best)

    distances = np.array(distances)

    return BenchStat(
        distances=distances,
        n_miss=n_miss,
        same_method_bench_gt=(detection_method==gt.detection_method)
    )