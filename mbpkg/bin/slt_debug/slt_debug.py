from typing import Dict, List, Optional

from cut_detector.utils.mid_body_spot import MidBodySpot
from cut_detector.utils.mb_support.tracking import SpatialLapTrack
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory

from mbpkg.movie_loading import Source

from .spatial_laptrack_debug import SpatialLaptrackDebug

import sys

def spatial_laptrack_debug(
        source: Source,
        slt: SpatialLapTrack,
        detection_method: MidBodyDetectionFactory.SPOT_DETECTION_METHOD,
        log_fp: Optional[str],
        out_dir: Optional[str],
        show_tracking: bool = False):
    """ 
    Any SLT can be used, its settings will be imported into
    the debug versiob of SLT
    Runs a debug version of the spatial LapTrack
    If log_fp is a str, stdout is redirected to a file
    if out_dir is a str, result images are generated at the end to the specified
    directory.
    """
    movie_data = source.load_data()
    factory = MidBodyDetectionFactory()
    spots = factory.detect_mid_body_spots(
        mitosis_movie=movie_data,
        mode=detection_method
    )

    if isinstance(log_fp, str):
        original_stdout = sys.stdout
        with open(log_fp, "w") as log_file:
            sys.stdout = log_file
            run_laptrack(
                factory,
                slt, 
                spots,
                show_tracking
            )
            sys.stdout = original_stdout
    else:
        run_laptrack(
            factory,
            slt, 
            spots,
            show_tracking
        )

    if isinstance(out_dir, str):
        factory.save_mid_body_tracking(spots, movie_data, out_dir)


def run_laptrack(
        factory: MidBodyDetectionFactory, 
        slt: SpatialLapTrack, 
        spots: Dict[int, List[MidBodySpot]], 
        show_tracking: bool):
    
    debug_slt = SpatialLaptrackDebug(
        show_predict_link_debug=True,
        show_gap_closing_debug=True
    )
    debug_slt.import_settings_from_slt(slt)

    factory.generate_tracks_from_spots(
        spots,
        debug_slt,
        show_tracking=show_tracking
    )


