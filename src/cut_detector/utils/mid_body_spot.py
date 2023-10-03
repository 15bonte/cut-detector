from __future__ import annotations

from typing import Optional
import numpy as np


class MidBodySpot:
    """
    Mid-body candidate spot
    """

    def __init__(
        self,
        frame: int,
        position: list[int],
        intensity: Optional[float] = None,
        sir_intensity: Optional[float] = None,
        area: Optional[float] = None,
        circularity: Optional[float] = None,
    ):
        self.frame = frame
        self.position = position  # (x, y)
        self.intensity = intensity
        self.sir_intensity = sir_intensity
        self.area = area
        self.circularity = circularity

        self.parent_spot: Optional[MidBodySpot] = None
        self.child_spot: Optional[MidBodySpot] = None

        self.track_id: Optional[int] = None

    def distance_to(self, other_spot: MidBodySpot) -> float:
        return np.sqrt(
            (self.position[0] - other_spot.position[0]) ** 2
            + (self.position[1] - other_spot.position[1]) ** 2
        )
