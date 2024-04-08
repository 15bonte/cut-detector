from __future__ import annotations

import numpy as np
import pandas as pd

from .track import Track, TRACKING_METHOD
from .mid_body_spot import MidBodySpot


class MidBodyTrack(Track[MidBodySpot]):
    """
    Mid-body candidate track
    """

    @staticmethod
    def generate_tracks_from_spots(
        spots: dict[int, list[MidBodySpot]],
        method: TRACKING_METHOD,
        show_post_conv_df: bool = False,
        show_tracking_df: bool = False,
        show_tracking_plot: bool = False,
    ) -> list[Track[MidBodySpot]]:
        return Track[MidBodySpot].generate_tracks_from_spots(
            MidBodySpot,
            spots,
            method,
            show_post_conv_df,
            show_tracking_df,
            show_tracking_plot,
        )

    @staticmethod
    def track_df_to_mb_track(
        track_df: pd.DataFrame,
        spots: dict[int, list[MidBodySpot]],
    ) -> list[MidBodyTrack]:

        track_df.reset_index(inplace=True)
        track_df.dropna(inplace=True)
        id_to_track = {}

        for _, row in track_df.iterrows():
            track_id = row["track_id"]
            track: MidBodyTrack = id_to_track.get(track_id)
            if track is None:
                id_to_track[track_id] = MidBodyTrack(len(id_to_track))
                track = id_to_track[track_id]
            frame = row["frame"]
            idx_in_frame = row["idx_in_frame"]
            track.add_spot(spots[int(frame)][int(idx_in_frame)])

        return list(id_to_track.values())

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
