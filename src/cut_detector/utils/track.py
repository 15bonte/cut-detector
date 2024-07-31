"""Track is the common ancestor for all specialized tracks of Specialized Points.

To create a new specialized track, you must do the following:
- Create your specialized XYZSpot class:
    - Make it inherit from Spot
    - implement (static function) Spot.get_extra_features_name()
    - implement (method) Spot.get_extra_features()
    (see their doc for additional help)
    
- Create your specialized XYZTrack class:
    - Make it inherit from Track
    - implement (static function) Track.track_df_to_track_list()

- In gen_track.py:
    - import your new specialized track,
    - import your new specialized spot,
    - add them to the SPOT_AND_TRACK_MAPPING dict,
"""

from __future__ import annotations
from typing import TypeVar, Generic
from abc import abstractmethod, ABC

import pandas as pd
from .spot import Spot
from .cell_spot import CellSpot
from .mid_body_spot import MidBodySpot

T = TypeVar("T", MidBodySpot, CellSpot)


class Track(ABC, Generic[T]):
    """
    Class used for both cell and mid-body tracks.

    Parameters
    ----------
    track_id : int
        Track id.
    """

    def __init__(self, track_id: int) -> None:
        self.track_id = track_id

        self.spots: dict[int, T] = {}
        self.length = 0  # = len(self.spots)

    def add_spot(self, spot: Spot) -> None:
        """
        Add spot to track.
        """
        self.spots[spot.frame] = spot
        spot.track_id = self.track_id
        self.length += 1
        # Add children recursively
        if hasattr(spot, "child_spot") and spot.child_spot is not None:
            self.add_spot(spot.child_spot)

    @staticmethod
    @abstractmethod
    def track_df_to_track_list(
        track_df: pd.DataFrame,
        spots: dict[int, list[Spot]],
    ) -> list[Track]:
        """Implemented in children classes."""
        return []
