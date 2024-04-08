from data_loading import Source, SourceFiles
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from cut_detector.factories.mb_support.tracking import SpatialLapTrack
from cut_detector.factories.mb_support import detection, tracking

from pathlib import Path
from os.path import join

class WrapperEnv:
    src_file: Source = SourceFiles.siCep
    bench_src_files: list[Source] = [
        SourceFiles.example,
        SourceFiles.siCep
    ]
    
    main_spot_detection_method: MidBodyDetectionFactory.SPOT_DETECTION_MODE = detection.cur_log
    
    main_tracking_method: SpatialLapTrack = tracking.cur_spatial_laptrack
    main_slt_tracking_method: SpatialLapTrack = tracking.cur_spatial_laptrack

    out_dir:    str = "src/cut_detector/data/mid_bodies"
    gt_dir:     str = "src/cut_detector/data/mid_bodies_movies_test/gt"
    log_dir:    str = "src/cut_detector/data/mid_bodies_movies_test/log"
    config_dir: str = "src/cut_detector/data/mid_bodies_movies_test/config"

    @staticmethod
    def gt_filepath_for_source(src: Source) -> str:
        p = Path(src.path)
        return join(WrapperEnv.gt_dir, f"{p.stem}__gt.json")
    
    @staticmethod
    def log_filepath_for_purpose(purpose: str) -> str:
        return join(WrapperEnv.log_dir, purpose)

    @staticmethod
    def get_config_file_path(name: str) -> str:
        return join(WrapperEnv.config_dir, name)

