from data_loading import Source, SourceFiles
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from cut_detector.factories.mb_support.tracking import cur_spatial_laptrack, SpatialLapTrack

from pathlib import Path
from os.path import join

class WrapperEnv:
    src_file: Source = SourceFiles.example
    main_spot_detection_method: MidBodyDetectionFactory.SPOT_DETECTION_MODE = "lapgau"
    main_tracking_method: SpatialLapTrack = cur_spatial_laptrack
    main_slt_tracking_method: SpatialLapTrack = cur_spatial_laptrack
    out_dir: str = "src/cut_detector/data/mid_bodies"
    gt_dir:  str = "src/cut_detector/data/mid_bodies_movies_test"
    log_dir: str = "src/cut_detector/data/mid_bodies_movies_test"

    @staticmethod
    def gt_filepath_from_source(src: Source) -> str:
        p = Path(src.path)
        gt_out_dir = join(WrapperEnv.gt_dir, "gt")
        return join(gt_out_dir, f"{p.stem}__gt.json")
    
    @staticmethod
    def log_filepath_for_purpose(purpose: str) -> str:
        join(WrapperEnv.log_dir, "log", purpose)

