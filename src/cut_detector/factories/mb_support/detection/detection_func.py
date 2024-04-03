import numpy as np
from skimage.feature import blob_log, blob_dog, blob_doh


def min_max(img: np.ndarray) -> np.ndarray:
    max = np.max(img)
    min = np.min(img)
    return (img - min) / (max - min)


def detect_minmax_log(img: np.ndarray, **kwargs) -> np.ndarray:
    img_norm = min_max(img)
    return blob_log(image=img_norm, **kwargs)

def detect_minmax_dog(img: np.ndarray, **kwargs) -> np.ndarray:
    img_norm = min_max(img)
    return blob_dog(img_norm, **kwargs)

def detect_minmax_doh(img: np.ndarray, **kwargs) -> np.ndarray:
    img_norm = min_max(img)
    return blob_doh(img_norm, **kwargs)


