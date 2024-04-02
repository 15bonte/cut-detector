from __future__ import annotations

from typing import Optional

from .spot import Spot


class MidBodySpot(Spot):
    """
    Mid-body candidate spot
    """

    def __init__(
        self,
        frame: int,
        x: int,
        y: int,
        intensity: Optional[float] = None,
        sir_intensity: Optional[float] = None,
        area: Optional[float] = None,
        circularity: Optional[float] = None,
    ):
        super().__init__(frame, x, y)
        self.intensity = intensity
        self.sir_intensity = sir_intensity
        self.area = area
        self.circularity = circularity

        self.parent_spot: Optional[MidBodySpot] = None
        self.child_spot: Optional[MidBodySpot] = None

        self.track_id: Optional[int] = None
