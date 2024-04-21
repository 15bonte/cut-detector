""" Debug verison of skimage's blob_log
"""

from typing import Any, Literal, Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.feature import blob_log
from skimage.util import img_as_float
from skimage._shared.utils import _supported_float_type
import plotly.express as px

def debug_log(
        image: np.ndarray, 
        output: Tuple[Literal["s_layer", "s_cube"], float], 
        args: dict[str, Any],
        ) -> dict:
    
    ####### Splitting and handling output ########  
    output_kind = output[0]
    output_param = output[1] # meaning changes based on output_kind
    if output_kind not in ["s_layer", "s_cube"]:
        return {}
    
    ####### Parameters turned into constants (since we are not using them) ########
    log_scale=False

    ####### Turning back args parameters into variables ########   
    # Note: Threshold is not used in this snippet
    min_sigma = args["min_sigma"]
    max_sigma = args["max_sigma"]
    num_sigma = args["num_sigma"]
    
    ######## Beginning of skimage.feature.blob_log snippet ########
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = (
        True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
    )

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float_dtype)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float_dtype)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float_dtype)
    max_sigma = np.asarray(max_sigma, dtype=float_dtype)

    if log_scale:
        start = np.log10(min_sigma)
        stop = np.log10(max_sigma)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    # computing gaussian laplace
    image_cube = np.empty(image.shape + (len(sigma_list),), dtype=float_dtype)

    ######## End of most of skimage.feature.blob_log snippet ########
    if output_kind == "s_layer":
        sigma_layer = -ndi.gaussian_laplace(image, output_param) * np.mean(output_param)**2
        return px.imshow(sigma_layer)
    
    elif output_kind == "s_cube":
        for i, s in enumerate(sigma_list):
            # average s**2 provides scale invariance
            image_cube[..., i] = -ndi.gaussian_laplace(image, s) * np.mean(s)**2
        threshold_indices = np.argwhere(np.absolute(image_cube) >= output_param)

        n_points = threshold_indices.shape[0]
        vs = image_cube[threshold_indices]
        xs = np.zeros(n_points)
        ys = np.zeros(n_points)
        zs = np.zeros(n_points)
        for idx, coord in enumerate(threshold_indices):
            xs[idx] = coord[0]
            ys[idx] = coord[1]
            zs[idx] = coord[2] 

        return px.scatter_3d(vs, xs, ys, zs)

    




