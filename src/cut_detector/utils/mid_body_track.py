import numpy as np

from .mid_body_spot import MidBodySpot


class MidBodyTrack:
    """
    Mid-body candidate track
    """

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.spots: dict[int, MidBodySpot] = {}
        self.length = 0

    def add_spot(self, spot: MidBodySpot) -> None:
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
            distances.append(
                np.linalg.norm(np.array(self.spots[frame].position) - np.array(position))
            )
        # If there are no frames in common, for sure track is not the right one
        if len(distances) == 0:
            return np.inf
        mean_distance = np.mean(distances)
        if mean_distance > max_distance:
            return np.inf
        return mean_distance
