from __future__ import annotations
from typing import Callable, Optional
import numpy as np

from .track import Track
from .mid_body_spot import MidBodySpot


class MidBodyTrack(Track[MidBodySpot]):
    """
    Mid-body candidate track
    """

    max_frame_gap = 2

    def add_spot(self, spot: MidBodySpot) -> None:
        """
        Add spot to track.
        """
        self.spots[spot.frame] = spot
        spot.track_id = self.track_id
        self.length += 1
        # Add children recursively
        if spot.child_spot is not None:
            self.add_spot(spot.child_spot)

    def get_expected_distance(
        self, expected_positions: dict[int, list[int]], max_distance: float
    ) -> float:
        """
        Compute the average distance between mid-body expected positions and current
        track positions.
        """
        distances = []
        for frame, position in expected_positions.items():
            if frame not in self.spots:
                continue
            spot = self.spots[frame]
            distances.append(
                np.linalg.norm(spot.get_position() - np.array(position))
            )
        # If there are no frames in common, for sure track is not the right one
        if len(distances) == 0:
            return np.inf
        mean_distance = np.mean(distances)
        if mean_distance > max_distance:
            return np.inf
        return mean_distance

    @staticmethod
    def generate_tracks_from_spots(
        spots: dict[int, list[MidBodySpot]],
        linking_max_distance: int,
        gap_closing_max_distance: int = None,
        track_dist_metric: str | Callable = "sqeuclidean",
    ) -> list[MidBodyTrack]:
        """
        Generate tracks from spots.
        """
        max_frame_gap = MidBodyTrack.max_frame_gap
        raise NotImplementedError
