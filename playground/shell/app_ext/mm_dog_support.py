from typing import Any
from functools import partial

import numpy as np
from skimage.feature import blob_dog
from skimage.util import img_as_float
from skimage._shared.utils import _supported_float_type
from skimage._shared.filters import gaussian

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
    
def mm_dog_layer_widget_maker(layer: str) -> list[Widget]:
    if layer == "Sigma Layer":
        return [
            NumericWidget(0, "sigma", "Initial Sigma"),
            NumericWidget(1, "ratio", "Sigma ratio"),
        ]
    else:
        raise RuntimeError(f"Unsupported layer {layer}")

def mm_dog_layer_param_extractor(detector: SPOT_DETECTION_METHOD, layer: str) -> dict[str, Any]:
    if isinstance(detector, partial):
        kws = detector.keywords
        if layer == "Sigma Layer":
            return {
                "sigma": kws["min_sigma"],
                "ratio": kws["sigma_ratio"],
            }
        else:
            raise RuntimeError(f"Unsupported layer {layer}")
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

def mm_dog_layer_detector_debug(layer: str, img: np.ndarray, args: dict[str, Any]) -> np.ndarray:
    if layer == "Sigma Layer":
        beg_sigma = args["sigma"]
        sigma_ratio = args["ratio"]
        if sigma_ratio <= 1.0:
            raise RuntimeError("sigma ratio muse be strictly greater than 1.0")
        end_sigma = beg_sigma * sigma_ratio

        return mm_dog_sigma_layer(img, beg_sigma, end_sigma, sigma_ratio)

    else:
        raise RuntimeError(f"Unsupported layer {layer}")
    

def mm_dog_sigma_layer(image: np.ndarray, beg_sigma: float, end_sigma: float, sigma_ratio: float) -> np.ndarray:
    # mm norm
    m = np.min(image)
    M = np.max(image)
    image = (image - m) / (M - m)
    
    # based on skimage.feature.blob_dog
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    beg_gaussian = gaussian(image, beg_sigma, mode="reflect")
    end_gaussian = gaussian(image, end_sigma, mode="reflect")
    diff = beg_gaussian - end_gaussian
    
    sf = 1 / (sigma_ratio - 1)
    diff *= sf

    return diff