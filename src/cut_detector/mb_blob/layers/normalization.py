""" Normalization layers
"""
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


class HardBinaryNormalizer(BlobLayer):
    """Binarizes the img based on a hard-coded threshold value.
    pixels strictly greater than value are turned into 1
    pixels less than value are turned into 0
    """
    def __init__(self, threshold: int | float):
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