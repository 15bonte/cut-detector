""" Simple pipeline run, equivalent to mid_body_detection.
For real benchmarks, see pipeline_bench
"""
from time import time

from cut_detector.utils.mb_support import tracking
from cut_detector.utils.gen_track import generate_tracks_from_spots

from mbpkg.parallel_factory import ParallelFactory
from bin_env.sources import SourceFiles
from bin_env.better_detectors import Detectors
from bin_env.out import OUT_DIR

SOURCE   = SourceFiles.example
PLOT_DIR  = OUT_DIR
D_METHOD = Detectors.cur_dog
T_METHOD = tracking.cur_spatial_laptrack

MEASURE_DETECTION_TIME = True
SHOW_POINTS = False
SHOW_TRACKS = False
SHOULD_SAVE = False
PARALLELIZE = "pool"

def simple_pipeline_run():
    # simple printing
    print("source:", SOURCE)
    print("detection:", D_METHOD)
    print('tracking:', T_METHOD)
    print("parallelization:", PARALLELIZE)

    image = SOURCE.load_data()

    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC
    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    factory = ParallelFactory()
    if MEASURE_DETECTION_TIME: 
        start = time()
        spots_candidates = factory.detect_mid_body_spots(
            # mitosis_movie=mitosis_movie, mask_movie=mask_movie, mode="h_maxima"
            mitosis_movie=mitosis_movie,
            mask_movie=mask_movie,
            mode=D_METHOD.to_factory(),
            parallelization=PARALLELIZE,
        )
        if MEASURE_DETECTION_TIME:
            end = time()
            delta = end - start
            print(f"====== Detection time: {delta:.3f}s =========")

    if SHOW_POINTS:
        for frame, spots in spots_candidates.items():
            for spot in spots:
                print(
                    {
                        "fr": frame,
                        "x": spot.x,
                        "y": spot.y,
                        "mlkp_int": spot.intensity,
                        "sir_int": spot.sir_intensity,
                    }
                )

    generate_tracks_from_spots(spots_candidates, T_METHOD)


    if SHOW_TRACKS:
        for frame, spots in spots_candidates.items():
            for spot in spots:
                print(
                    {
                        "fr": frame,
                        "x": spot.x,
                        "y": spot.y,
                        "mlkp_int": spot.intensity,
                        "sir_int": spot.sir_intensity,
                        "track_id": spot.track_id
                    }
                )

    if SHOULD_SAVE:
        factory.save_mid_body_tracking(
            spots_candidates, mitosis_movie, PLOT_DIR
        )


