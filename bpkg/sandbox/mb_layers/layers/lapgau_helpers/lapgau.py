""" Laplacian of Gaussian and other functions copy-pasted from 
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/blob.py

This function's dependencies are either imported here (if they can be imported)
or copy-pasted in lapgau_dep otherwise

We are going to modify it to insert visualisation hooks for debugging.
"""
from typing import Optional
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.feature import peak_local_max
from skimage.feature.blob import _prune_blobs, _format_exclude_border
from .lapgau_dep import _supported_float_type

class BlobLogVisuSettings:
    def __init__(
            self, 
            cube_plotting_threshold: 
            Optional[float] = None, 
            sig_layers_idx: list[int] = []) -> None:
        self.cube_plotting_threshold = cube_plotting_threshold
        self.sig_layers_idx = sig_layers_idx
        

def blob_log_with_plotting(
    image,
    min_sigma=1,
    max_sigma=50,
    num_sigma=10,
    threshold=0.2,
    overlap=0.5,
    log_scale=False,
    *,
    threshold_rel=None,
    exclude_border=False,
    plot_settings: BlobLogVisuSettings=BlobLogVisuSettings()
):
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False

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
    for i, s in enumerate(sigma_list):
        # average s**2 provides scale invariance
        image_cube[..., i] = -ndi.gaussian_laplace(image, s) * np.mean(s) ** 2
        if i in plot_settings.sig_layers_idx:
            plt.figure()
            plt.imshow(image_cube[..., i])
            plt.title(f"LapOfGauss: (i:{i}/sigma:{s})")

    # Showing cube
    cube_threshold = plot_settings.cube_plotting_threshold
    if cube_threshold is not None:
        coords = np.argwhere((image_cube > cube_threshold) | (image_cube < -cube_threshold))
        x = np.zeros(coords.shape[0])
        y = np.zeros(coords.shape[0])
        z = np.zeros(coords.shape[0])
        v = np.zeros(coords.shape[0])
        for idx, c in enumerate(coords):
            # print("c shape:", c.shape)
            x[idx] = c[0]
            y[idx] = c[1]
            z[idx] = c[2]
            v[idx] = image_cube[c[0], c[1], c[2]]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, c=v)
        # ax.set_title("LapOfGauss Cube")
        plt.title("LapOfGauss Cube")
        # plt.show()


    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        footprint=np.ones((3,) * (image.ndim + 1)),
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))

    # Convert local_maxima to float64
    lm = local_maxima.astype(float_dtype)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]

    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)