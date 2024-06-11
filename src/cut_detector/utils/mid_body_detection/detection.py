"""Define optimal detection functions for every strategy."""

from functools import partial
import numpy as np
from skimage.feature import blob_log, blob_dog, blob_doh


def min_max(img: np.ndarray) -> np.ndarray:
    """Normalize the image between 0 and 1.

    Parameters
    ----------
    img : np.ndarray
        Image to normalize.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    img_max = np.max(img)
    img_min = np.min(img)
    return (img - img_min) / (img_max - img_min)


def detect_minmax_log(img: np.ndarray, **kwargs) -> np.ndarray:
    """Detect blobs using Laplacian of Gaussian with min-max normalization.

    Parameters
    ----------
    img : np.ndarray
        Image to detect blobs.
    **kwargs
        Extra arguments for blob_log.

    Returns
    -------
    np.ndarray
        Blobs detected."""
    img_norm = min_max(img)
    return blob_log(image=img_norm, **kwargs)


def detect_minmax_dog(img: np.ndarray, **kwargs) -> np.ndarray:
    """Detect blobs using Difference of Gaussian with min-max normalization.

    Parameters
    ----------
    img : np.ndarray
        Image to detect blobs.
    **kwargs
        Extra arguments for blob_dog.

    Returns
    -------
    np.ndarray
        Blobs detected."""
    img_norm = min_max(img)
    return blob_dog(img_norm, **kwargs)


def detect_minmax_doh(img: np.ndarray, **kwargs) -> np.ndarray:
    """Detect blobs using Determinant of Hessian with min-max normalization.

    Parameters
    ----------
    img : np.ndarray
        Image to detect blobs.
    **kwargs
        Extra arguments for blob_doh.

    Returns
    -------
    np.ndarray
        Blobs detected."""
    img_norm = min_max(img)
    return blob_doh(img_norm, **kwargs)


DETECTION_FUNCTIONS = {
    # Laplacian of Gaussian
    "laplacian_gaussian": partial(
        detect_minmax_log,
        min_sigma=5,
        max_sigma=10,
        num_sigma=5,
        threshold=0.1,
    ),
    "log2_wider": partial(
        detect_minmax_log, min_sigma=2, max_sigma=8, num_sigma=4, threshold=0.1
    ),
    "rshift_log": partial(
        detect_minmax_log,
        min_sigma=3,
        max_sigma=11,
        num_sigma=5,
        threshold=0.1,
    ),
    # Difference of Gaussian
    "difference_gaussian": partial(
        detect_minmax_dog,
        min_sigma=2,
        max_sigma=5,
        sigma_ratio=1.2,
        threshold=0.1,
    ),
    "very_fast_dog": partial(
        detect_minmax_dog,
        min_sigma=5,
        max_sigma=5,
        sigma_ratio=1.2,
        threshold=0.1,
    ),
    "dog_005": partial(
        detect_minmax_dog,
        min_sigma=2,
        max_sigma=5,
        sigma_ratio=1.2,
        threshold=0.05,
    ),
    # Determinant of Hessian
    "determinant_hessian": partial(
        detect_minmax_doh,
        min_sigma=5,
        max_sigma=10,
        num_sigma=5,
        threshold=0.0040,
    ),
}
