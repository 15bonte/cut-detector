import json

import numpy as np

from mbpkg.helpers.json_processing import extract, assert_extraction
from mbpkg.better_detector import Detector

from .detection_stat import DetectionStat
from .detection_loading_error import DetectionLoadingError


def load_detection_stat(filepath: str) -> DetectionStat:
    """ Loads a detection stat from a file.
    """
    with open(filepath, "r") as file:
        data = json.load(file)

    try:
        detector_str = extract(data, "detector",    str)
        source_path  = extract(data, "source_path", str)
        time         = extract(data, "time",        float, allow_missing=True)
        fn_count     = extract(data, "fn_count",    int)
        fp_count     = extract(data, "fp_count",    int)
        distances    = extract(data, "distances",   list)

    except DetectionLoadingError as e:
        raise DetectionLoadingError(e.args) from e

    return DetectionStat(
        Detector(detector_str),
        source_path,
        np.array(distances),
        time,
        fn_count,
        fp_count
    )
