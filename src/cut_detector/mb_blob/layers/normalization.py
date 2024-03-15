""" Normalization layers
"""
import numpy as np
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