from __future__ import annotations

import numpy as np

from typing import Any
from abc import ABC, abstractmethod

class Spot(ABC):
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
    
    @staticmethod
    @abstractmethod
    def get_extra_features_name() -> list[str]:
        """
        To have better tracking for classes that inherit from this one.
        These classes can declare here their extra coordinate fields.

        Example:
        If you have extra features, like 'color' and 'shape', implement this
        the following way:
        ```python
        def get_extra_features_name() -> list[str]:
            return ["color", "shape"]
        ```
        """
        return []
    
    @abstractmethod
    def get_extra_coordinates(self) -> list[Any]:
        """
        To have better tracking for classes that inherit from this one.
        These classes can return here their extra coordinate fields.

        Example:
        If you have extra features, like 'color' and 'shape', implement this
        the following way:
        ```python
        def get_extra_features_name(self) -> list[str]:
            return [self.color, self.shape]
        ```
        """
        return []
