from typing import Any

import numpy as np
import scipy.ndimage as ndi
from skimage.feature import blob_dog
from skimage.util import img_as_float
from skimage._shared.utils import _supported_float_type
from skimage._shared.filters import gaussian

from .debug_output import DebugOutputError, DebugOutput, SigmaLayoutOutput

def mm_dog_visualizer(
        detector_args: dict[str, Any], 
        img: np.ndarray, 
        debug_output: DebugOutput
        ) -> np.ndarray:
    
    m = np.min(img)
    M = np.max(img)
    processed_img = (img - m) / (M - m)

    if isinstance(debug_output, SigmaLayoutOutput):
        return debug_blob_dog_sigma_layer(processed_img, debug_output.sigma)
    else:
        raise DebugOutputError(f"Unknown debug output {debug_output}")
    

def debug_blob_dog_sigma_layer(
        img: np.ndarray, 
        base_sigma: float, 
        sigma_ratio: float,
        ) -> np.ndarray:
    
    raise RuntimeError("Dog not supported for now")
    
    if sigma_ratio <= 1.0:
        raise ValueError('sigma_ratio must be > 1.0')
    
    img = img_as_float(img)
    float_dtype = _supported_float_type(img.dtype)
    img = img.astype(float_dtype, copy=False)

    gaussian_previous = gaussian(img, base_sigma, mode='reflect')
    gaussian_current = gaussian(img, base_sigma*sigma_ratio, mode='reflect')
    sf = 1 / (sigma_ratio - 1)

    return (gaussian_previous - gaussian_current) * sf


