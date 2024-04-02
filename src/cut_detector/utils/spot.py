from __future__ import annotations

import numpy as np


class Spot:
    """
    Class used for both cell and mid-body spots.
    """

    def __init__(self, frame: int, x: int, y: int):
        self.frame = frame
        self.x = x
        self.y = y

    def get_position(self) -> np.ndarray:
        """
        Return position as numpy array
        """
        return np.array([self.x, self.y])

    def distance_to(self, other_spot: Spot):
        """
        Compute distance between two spots
        """
        return np.linalg.norm(
            np.array(self.get_position()) - np.array(other_spot.get_position())
        )
