from typing import Callable, Tuple
from functools import partial

from cut_detector.factories.mb_support import detection
from cut_detector.factories.mb_support.detection import detection_impl
from cut_detector.factories.mb_support.detection import detection_current

from .bench_config import BenchConfig, DetectionConfig

def generate_config() -> BenchConfig:
    """ Creates a default bench config with all available detectors
    """
    detectors = {}
    current_funcs = [f for _, f in detection_current if callable(f)]
    all_funcs     = [f for _, f in detection_impl    if callable(f)]
    
    for f in current_funcs:
        name, d_config = generate_detection(f)
        detectors[name] = d_config
    for f in all_funcs:
        name, d_config = generate_detection(f)
        detectors[name] = d_config

    return BenchConfig(detectors)


def generate_detection(fn: partial) -> Tuple[str, DetectionConfig]:
    """ Creates a DetectionConfig from a partial function fn of a known type.
    Known types are:
    - detect_minmax_log
    - detect_minmax_dog
    - detect_minmax_doh

    If unknown, the function raises an error
    """
    if fn.func == detection.detect_minmax_log:
        return ("log", DetectionConfig("log", fn.keywords))
    elif fn.func == detection.detect_minmax_dog:
        return ("dog", DetectionConfig("dog", fn.keywords))
    elif fn.func == detection.detect_minmax_doh:
        return ("dog", DetectionConfig("doh", fn.keywords))
    else:
        raise RuntimeError("Unknown function f{fn.func}")