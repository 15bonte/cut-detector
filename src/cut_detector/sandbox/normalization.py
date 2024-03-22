""" Some normalization functions for frame processing
"""

import numpy as np

def binary_norm(frame: np.array, threshold: int) -> np.array:
    """ Hard binary normalization.
    1: for values strictly greater than threshold
    0: otherwise
    """
    return frame > threshold 

def min_max_norm(frame: np.array) -> np.array:
    min = np.min(frame)
    max = np.max(frame)
    return (frame - min) / (max - min)


def min_perc_norm(frame: np.array, percentile: float) -> np.array:
    """ MinMax normalization but using a percentile value for max.
    percentile must be between 0 and 100 inclusive
    """
    min = np.min(frame)
    perc = np.percentile(frame, percentile)
    return (frame - min) / (max - min)