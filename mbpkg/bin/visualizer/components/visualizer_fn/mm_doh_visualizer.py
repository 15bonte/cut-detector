from typing import Any

import numpy as np
import scipy.ndimage as ndi
from skimage.feature import blob_doh
from skimage.transform import integral_image
from skimage.util import img_as_float
from skimage._shared.utils import _supported_float_type, check_nD
from skimage.feature._hessian_det_appx import _hessian_matrix_det

from .debug_output import DebugOutputError, DebugOutput, SigmaLayoutOutput

def mm_doh_visualizer(
        detector_args: dict[str, Any], 
        img: np.ndarray, 
        debug_output: DebugOutput
        ) -> np.ndarray:
    
    m = np.min(img)
    M = np.max(img)
    processed_img = (img - m) / (M - m)

    if isinstance(debug_output, SigmaLayoutOutput):
        return debug_blob_doh_sigma_layer(processed_img, debug_output.sigma)
    else:
        raise DebugOutputError(f"Unknown debug output {debug_output}")
    

def debug_blob_doh_sigma_layer(img: np.ndarray, sigma: float) -> np.ndarray:
    check_nD(img, 2)

    img = img_as_float(img)
    float_dtype = _supported_float_type(img.dtype)
    img = img.astype(float_dtype, copy=False)

    img = integral_image(img)

    return _hessian_matrix_det(img, sigma)