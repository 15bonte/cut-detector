from typing import Any
from functools import partial

from cut_detector.utils.mb_support import detection

from .ext_api import Widget, NumericWidget, SPOT_DETECTION_METHOD

def mm_dog_widgets() -> list[Widget]:
    return [
        NumericWidget(0, "min_sigma", "Minimal Sigma"),
        NumericWidget(1, "max_sigma", "Maximal Sigma"),
        NumericWidget(2, "sigma_ratio", "Sigma ratio"),
        NumericWidget(3, "threshold", "Threshold"),
    ]

def mm_dog_param_extractor(detector: SPOT_DETECTION_METHOD) -> dict[str, Any]:
    if isinstance(detector, partial):
        return detector.keywords
    else:
        raise RuntimeError("Only the partial functions are supported")
    
def mm_dog_maker(params: dict[str, Any]) -> SPOT_DETECTION_METHOD:
    return partial(
        detection.detect_minmax_dog,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        sigma_ratio=params["sigma_ratio"],
        threshold=params["threshold"],
    )