""" Starts a debug run of Spatial Laptrack
"""

import sys
from os.path import join
from laptrack import LapTrack
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory

from ..data_loader import load_movie, MOVIE_KIND_LITERAL
from .spatial_laptrack_debug import SpatialLaptrackDebug


def start_debug_run(
        dir: str, 
        filename: str, 
        movie_kind: MOVIE_KIND_LITERAL,
        detection_method: MidBodyDetectionFactory.SPOT_DETECTION_MODE,
        laptrack_method: MidBodyDetectionFactory.TRACKING_MODE | LapTrack = "spatial_laptrack",
        show_tracking: bool = True,
        out_dir: str | None = None,
        log_fp: str | None = None
    ):
    filepath = join(dir, filename)
    mitosis_movie = load_movie(filepath, movie_kind)
    factory = MidBodyDetectionFactory()

    spots_candidates = factory.detect_mid_body_spots(
        mitosis_movie=mitosis_movie,
        mask_movie=None,
        mode=detection_method,
    )

    if isinstance(log_fp, str):
        original_stdout = sys.stdout
        with open(log_fp, "w") as f:
            sys.stdout = f
            run_laptrack(factory, laptrack_method, spots_candidates, show_tracking)
            sys.stdout = original_stdout
    else:
        run_laptrack(factory, laptrack_method, spots_candidates, show_tracking)

    if isinstance(out_dir, str):
        factory.save_mid_body_tracking(
            spots_candidates, mitosis_movie, out_dir
        )

def run_laptrack(factory, laptrack_method, spots_candidates, show_tracking):
    if isinstance(laptrack_method, str):
        factory.generate_tracks_from_spots(
            spots_candidates=spots_candidates,
            tracking_method=laptrack_method,
            show_tracking=show_tracking,
        )
    elif isinstance(laptrack_method, LapTrack):
        factory.generate_tracks_from_spots(
            spots_candidates=spots_candidates,
            show_tracking=show_tracking,
            use_custom_laptrack=laptrack_method
        )
    else:
        raise RuntimeError(f"invalid laptrack method kind: {type(laptrack_method)}")
