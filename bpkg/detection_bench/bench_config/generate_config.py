from functools import partial

from cut_detector.factories.mb_support import detection
from cut_detector.factories.mb_support.detection import detection_impl
from cut_detector.factories.mb_support.detection import detection_current

from .bench_config import BenchConfig, DetectionConfig

def generate_config() -> BenchConfig:
    """ Creates a default bench config with all available detectors
    """
    detectors = {}

    funcs = {}
    current_funcs = {k: f for k, f in detection_current.__dict__.items() if callable(f)}
    all_funcs = {k: f for k, f in detection_impl.__dict__.items() if callable(f)}
    funcs.update(current_funcs)
    funcs.update(all_funcs)

    filtered_funcs = {}
    for k, f in funcs.items():
        if not (k == "partial" or k.startswith("detect_")):
            filtered_funcs[k] = f

    for k, f in filtered_funcs.items():
        d_config = generate_detection(f)
        detectors[k] = d_config

    return BenchConfig(detectors)


def generate_detection(fn: partial) -> DetectionConfig:
    """ Creates a DetectionConfig from a partial function fn of a known type.
    Known types are:
    - detect_minmax_log
    - detect_minmax_dog
    - detect_minmax_doh

    If unknown, the function raises an error
    """

    if fn.func == detection.detect_minmax_log:
        return DetectionConfig("log", fn.keywords)
    elif fn.func == detection.detect_minmax_dog:
        return DetectionConfig("dog", fn.keywords)
    elif fn.func == detection.detect_minmax_doh:
        return DetectionConfig("doh", fn.keywords)
    else:
        raise RuntimeError(f"Unknown function f{fn.func}")