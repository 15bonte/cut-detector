from __future__ import annotations

from typing import Optional, Any
from shapely.geometry.polygon import Polygon
from shapely import distance

from .spot import Spot


class CellSpot(Spot):
    """
    Useful information for a detected cell.

    Parameters
    ----------
    frame : int
        Frame number.
    x : int
        Centroid X coordinate.
    y : int
        Centroid Y coordinate.
    id_number : int
        Cell id.
    abs_min_x : int
        Absolute minimum x coordinate.
    abs_max_x : int
        Absolute maximum x coordinate.
    abs_min_y : int
        Absolute minimum y coordinate.
    abs_max_y : int
        Absolute maximum y coordinate.
    spot_points : list[list[int]]
        List of points that define the cell contour.
    """

    def __init__(
        self,
        frame: int,
        x: int,
        y: int,
        id_number: int,
        abs_min_x: int,
        abs_max_x: int,
        abs_min_y: int,
        abs_max_y: int,
        spot_points: list[list[int]],  # (x, y)
    ):
        super().__init__(frame, x, y)

        self.id = id_number
        self.abs_min_x = abs_min_x
        self.abs_max_x = abs_max_x
        self.abs_min_y = abs_min_y
        self.abs_max_y = abs_max_y
        self.spot_points = spot_points

        self.track_id: int = -1

        # Phase predicted by model
        self.predicted_phase: Optional[int] = None
        # Corresponding (closest) metaphase spot in track
        self.corresponding_metaphase_spot = None

    def is_stuck_to(
        self, other_spot: CellSpot, maximum_stuck_distance: float
    ) -> bool:
        """
        Distance between two spots hulls.

        Parameters
        ----------
        other_spot : CellSpot
            Other spot to compare.
        maximum_stuck_distance : float
            Maximum distance to consider two spots stuck.

        Returns
        -------
        bool
            True if the distance between the spots is less than the maximum distance.
        """
        return (
            distance(
                Polygon(self.spot_points), Polygon(other_spot.spot_points)
            )
            < maximum_stuck_distance
        )

    @staticmethod
    def get_extra_features_name() -> list[str]:
        return []

    def get_extra_coordinates(self) -> list[Any]:
        return []
