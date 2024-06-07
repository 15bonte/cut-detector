from __future__ import annotations

from typing import Optional

from .spot import Spot


class MidBodySpot(Spot):
    """Mid-body spot.

    Parameters
    ----------
    frame : int
        Frame of the spot.
    x : int
        X coordinate of the spot.
    y : int
        Y coordinate of the spot.
    intensity : Optional[float], optional
        Intensity of the spot, by default None.
    sir_intensity : Optional[float], optional
        Sir intensity of the spot, by default None.
    area : Optional[float], optional
        Area of the spot, by default None.
    circularity : Optional[float], optional
        Circularity of the spot, by default None.
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

    @staticmethod
    def get_extra_features_name() -> list[str]:
        """Get the extra features name for tracking.

        Returns
        -------
        list[str]
            List of extra features name.
        """
        return ["mklp_intensity", "sir_intensity"]

    def get_extra_coordinates(self) -> list[float | None]:
        """Get the extra coordinates for tracking.

        Returns
        -------
        list[float | None]
            List of extra coordinates.
        """
        return [self.intensity, self.sir_intensity]
