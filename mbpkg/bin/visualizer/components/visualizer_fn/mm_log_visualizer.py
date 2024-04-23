from typing import Any

import numpy as np
import scipy.ndimage as ndi
from skimage.feature import blob_log
from skimage.util import img_as_float
from skimage._shared.utils import _supported_float_type

from .debug_output import DebugOutputError, DebugOutput, SigmaLayoutOutput

def mm_log_visualizer(
        detector_args: dict[str, Any], 
        img: np.ndarray, 
        debug_output: DebugOutput
        ) -> np.ndarray:
    
    m = np.min(img)
    M = np.max(img)
    processed_img = (img - m) / (M - m)

    if isinstance(debug_output, SigmaLayoutOutput):
        return debug_blob_log_sigma_layer(processed_img, debug_output.sigma)
    else:
        raise DebugOutputError(f"Unknown debug output {debug_output}")


def debug_blob_log_sigma_layer(image: np.ndarray, sigma: float) -> np.ndarray:
    """ A modified version of skimage.feature.blob_log
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    return -ndi.gaussian_laplace(image, sigma) * np.mean(sigma)**2


