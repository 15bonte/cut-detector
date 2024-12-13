from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Spot(ABC):
    """Class used for both cell and mid-body spots.

    Parameters
    ----------
    frame : int
        Frame of the spot.
    x : int
        X coordinate of the spot.
    y : int
        Y coordinate of the spot.
    """

    def __init__(self, frame: int, x: int, y: int):
        self.frame = frame
        self.x = x
        self.y = y

    def get_position(self) -> np.ndarray:
        """Return position as numpy array.

        Returns
        -------
        np.ndarray
            Position as numpy array.
        """
        if hasattr(self, "position"):  # old versions
            return np.array(self.position)
        return np.array([self.x, self.y])

    def distance_to(self, other_spot: Spot) -> float:
        """Compute distance between two spots.

        Parameters
        ----------
        other_spot : Spot
            Other spot to compute distance to.

        Returns
        -------
        float
            Distance between two spots.
        """
        return np.linalg.norm(
            np.array(self.get_position()) - np.array(other_spot.get_position())
        )

    def temporal_distance_to(self, other_spot: Spot) -> float:
        """Compute temporal distance between two spots.

        Parameters
        ----------
        other_spot : Spot
            Other spot to compute distance to.

        Returns
        -------
        float
            Temporal distance between two spots.
        """
        return abs(self.frame - other_spot.frame)

    @staticmethod
    def get_extra_features_name() -> list[str]:
        """To have better tracking for classes that inherit from this one.
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

    def get_extra_coordinates(self) -> list:
        """To have better tracking for classes that inherit from this one.
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

    def __str__(self) -> str:
        return f"x:{self.x} y:{self.y}"
