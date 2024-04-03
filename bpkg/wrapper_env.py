from importation import Source, SourceFiles
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory

from pathlib import Path
from os.path import join

class WrapperEnv:
    src_file: Source = SourceFiles.example
    main_spot_detection_method: MidBodyDetectionFactory.SPOT_DETECTION_MODE = "lapgau"
    out_dir: str = "src/cut_detector/data/mid_bodies_movies_test"
    # reference_detection_method: MidBodyDetectionFactory.SPOT_DETECTION_MODE = "lapgau"

    @staticmethod
    def gt_filepath_from_source(src: Source) -> str:
        p = Path(src.path)
        gt_out_dir = join(WrapperEnv.out_dir, "gt")
        return join(gt_out_dir, f"{p.stem}__gt.json")

