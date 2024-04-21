from time import time
from typing import Union

import numpy as np
from scipy.spatial.distance import cdist

from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory

from mbpkg.movie_loading import Movie
from mbpkg.detector import Detector
from mbpkg.detection_truth import load_gt, DetectionGT

from .detection_stat import DetectionStat

def generate_detection_stats(
        movie: Movie, 
        detector: Detector, 
        ground_truth: Union[str, DetectionGT],
        parallelization: bool,
        ) -> DetectionStat:
    
    frame_count = movie.get_framecount()

    factory_detector = detector.to_factory()
    factory = MidBodyDetectionFactory()

    start_time = time()
    spot_dict = factory.detect_mid_body_spots(
        movie.data,
        None,
        mode=factory_detector,
        parallelization=parallelization
    )
    end_time = time()
    duration = end_time - start_time

    if isinstance(ground_truth, str):
        gt = load_gt(ground_truth)
    elif isinstance(ground_truth, DetectionGT):
        gt = ground_truth
    else:
        raise RuntimeError(f"Invalid ground truth type: {type(ground_truth)}")

    fn_count = 0
    fp_count = 0

    all_distances = []

    for frame_idx in range(frame_count):
        test_spots = spot_dict.get(frame_idx, [])
        gt_spots = gt.spot_dict.get(frame_idx, [])
        d = len(gt_spots) - len(test_spots)
        fn_count += max(0, d)
        fp_count += max(0, -d)

        # converting spots to list (instead of nparray, so that we can delete values)
        test_spots = [[s.x, s.y] for s in test_spots]
        gt_spots   = [[s.x, s.y] for s in gt_spots]

        while len(test_spots) > 0 and len(gt_spots) > 0:
            distances = cdist(gt_spots, test_spots)

            # Returns an array of [x, y]; we take the first one (in case of several minima,
            # distances[x y] is one of the minimal values
            best = np.argwhere(distances == np.min(distances))[0]
            
            all_distances.append(distances[best[0], best[1]])
            del gt_spots[best[0]]
            del test_spots[best[1]]


    return DetectionStat(
        detector,
        movie.path,
        np.array(all_distances),
        duration,
        fn_count,
        fp_count
    )

