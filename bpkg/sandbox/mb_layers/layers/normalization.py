""" Normalization layers
"""
from typing import Union

import numpy as np
from skimage.morphology import area_closing, area_opening

from .layer import BlobLayer


class MaxNormalizer(BlobLayer):
    """Apply a normalization by dividing by maximum"""
    def apply(self, env: dict):
        img = env["img"]
        img = img / np.max(img)
        env["img"] = img


class MinMaxNormalizer(BlobLayer):
    """Apply a normlalization by substracting by min, and dividing by max-min"""
    def apply(self, env: dict):
        img = env["img"]
        min = np.min(img)
        max = np.max(img)
        img = (img-min) / (max-min)
        env["img"] = img

class MinPercentileNormalizer(BlobLayer):
    """Apply a normalization by substraction min, and dividing by percentile-min"""
    def __init__(self, percentile: int = 90):
        self.percentile = percentile
    
    def apply(self, env: dict):
        img = env["img"]
        min = np.min(img)
        p = np.percentile(img, self.percentile)
        img = (img-min) / (p-min)
        env["img"] = img

class HardBinaryNormalizer(BlobLayer):
    """Binarizes the img based on a hard-coded threshold value.
    pixels strictly greater than value are turned into 1
    pixels less than value are turned into 0
    """
    def __init__(self, threshold: Union[int, float]):
        self.threshold = threshold
    
    def apply(self, env: dict):
        img = env["img"]
        img = img > self.threshold
        env["img"] = img

class AreaOpeningNormalizer(BlobLayer):
    def __init__(self, area_threshold: int = 64):
        self.area_threshold = area_threshold
    
    def apply(self, env: dict):
        img = env["img"]
        img = area_opening(img, self.area_threshold)
        env["img"] = img


class AreaClosingNormalizer(BlobLayer):
    def __init__(self, area_threshold: int = 64):
        self.area_threshold = area_threshold
    
    def apply(self, env: dict):
        img = env["img"]
        img = area_closing(img, self.area_threshold)
        env["img"] = img