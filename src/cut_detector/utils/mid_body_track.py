from __future__ import annotations

import numpy as np
import pandas as pd

from .track import Track
from .mid_body_spot import MidBodySpot


class MidBodyTrack(Track[MidBodySpot]):
    """
    Mid-body candidate track
    """

    @staticmethod
    def track_df_to_track_list(
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
        self,
        expected_positions: dict[int, list[int]],
        spatial_resolution: int,
        max_distance=39.375,
    ) -> float:
        """
        Compute the average distance between mid-body expected positions and current
        track positions.

        Parameters
        ----------
        expected_positions : dict[int, list[int]]
            Expected mid-body positions.
        spatial_resolution : int
            Spatial resolution.
        max_distance : float, optional
            Maximum distance to consider the track (um).
        """
        max_distance_px = int(max_distance / spatial_resolution * 1000)  # px

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
        # If the mean distance is too high, discard the track
        if mean_distance > max_distance_px:
            return np.inf
        return mean_distance

    def fill_gaps(self):
        """
        Fill gaps in the track.
        """
        frames = list(self.spots.keys())
        min_frame, max_frame = min(frames), max(frames)
        for frame in range(min_frame, max_frame):
            next_frame = frame + 1
            # Look for the next frame with a spot
            while next_frame not in self.spots:
                next_frame += 1
            # If it was not the next frame, fill the gap
            if next_frame != frame + 1:
                gap_size = next_frame - (frame + 1)  # number of missing spots
                current_spot = self.spots[frame]
                next_spot = self.spots[next_frame]
                # Define interpolated values
                ranges = {}
                for attribute in [
                    "x",
                    "y",
                    "intensity",
                    "sir_intensity",
                    "area",
                    "circularity",
                ]:
                    if getattr(current_spot, attribute) is None:
                        ranges[attribute] = [None] * (gap_size + 2)
                    else:
                        ranges[attribute] = np.linspace(
                            getattr(current_spot, attribute),
                            getattr(next_spot, attribute),
                            gap_size + 2,
                        )
                for i in range(1, gap_size + 1):
                    new_spot = MidBodySpot(
                        frame=frame + i,
                        x=int(ranges["x"][i]),
                        y=int(ranges["y"][i]),
                        intensity=ranges["intensity"][i],
                        sir_intensity=ranges["sir_intensity"][i],
                        area=ranges["area"][i],
                        circularity=ranges["circularity"][i],
                    )
                    self.add_spot(new_spot)
        # All gaps should have been filled
        assert self.length == max_frame - min_frame + 1
