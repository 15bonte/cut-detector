from __future__ import annotations
from typing import Optional

import numpy as np
from shapely.geometry.polygon import Polygon
from shapely import distance


class TrackMateSpot:
    """
    Parse TrackMate spot from xml file.
    """

    def __init__(self, trackmate_spot, raw_video_shape: list[int]):
        self.x = float(trackmate_spot["@POSITION_X"])
        self.y = float(trackmate_spot["@POSITION_Y"])
        self.frame = int(trackmate_spot["@FRAME"])
        self.id = int(trackmate_spot["@ID"])
        self.track_id: int = -1

        # Get min and max positions
        raw_positions: str = trackmate_spot["#text"].split(" ")
        rel_positions_x = [float(raw_positions[i]) for i in range(0, len(raw_positions), 2)]
        rel_positions_y = [float(raw_positions[i]) for i in range(1, len(raw_positions), 2)]

        self.rel_min_x, self.rel_max_x = min(rel_positions_x), max(rel_positions_x)
        self.rel_min_y, self.rel_max_y = min(rel_positions_y), max(rel_positions_y)

        # Get spot points
        positions_x = [
            int(self.x + float(raw_positions[i])) for i in range(0, len(raw_positions), 2)
        ]
        positions_y = [
            int(self.y + float(raw_positions[i])) for i in range(1, len(raw_positions), 2)
        ]
        self.spot_points = [[x, y] for x, y in zip(positions_x, positions_y)]

        # Get min and max positions
        self.abs_min_x, self.abs_max_x = (
            int(self.x + self.rel_min_x),
            int(self.x + self.rel_max_x),
        )
        self.abs_min_y, self.abs_max_y = (
            int(self.y + self.rel_min_y),
            int(self.y + self.rel_max_y),
        )

        # Clip to video size
        self.abs_min_x, self.abs_max_x = (
            max(self.abs_min_x, 0),
            min(self.abs_max_x, raw_video_shape[2]),
        )
        self.abs_min_y, self.abs_max_y = (
            max(self.abs_min_y, 0),
            min(self.abs_max_y, raw_video_shape[1]),
        )

        # Phase predicted by model
        self.predicted_phase: Optional[int] = None

        # Corresponding (closest) metaphase spot in track
        self.corresponding_metaphase_spot = None

    def distance_to(self, other_spot: TrackMateSpot):
        return np.sqrt((self.x - other_spot.x) ** 2 + (self.y - other_spot.y) ** 2)

    def is_stuck_to(self, other_spot: TrackMateSpot, maximum_stuck_distance: float):
        """
        Distance between two spots hulls.
        """
        return (
            distance(Polygon(self.spot_points), Polygon(other_spot.spot_points))
            < maximum_stuck_distance
        )
