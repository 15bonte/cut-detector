from __future__ import annotations
from abc import abstractmethod
from typing import TypeVar, Generic

from .cell_spot import CellSpot
from .mid_body_spot import MidBodySpot

T = TypeVar("T", MidBodySpot, CellSpot)


class Track(Generic[T]):
    """
    Class used for both cell and mid-body tracks.
    """

    def __init__(self, track_id: int) -> None:
        self.track_id = track_id

        self.spots: dict[int, T] = {}
        self.length = 0  # = len(self.spots)
        self.number_spots = (
            0  # can be different from length if we have a gap in the track
        )

    @staticmethod
    @abstractmethod
    def generate_tracks_from_spots(
        spots: dict[int, list[T]],
        linking_max_distance: int,
        gap_closing_max_distance: int,
    ) -> list[Track[T]]:
        """
        Generate tracks from spots.
        """
