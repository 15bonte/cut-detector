from __future__ import annotations
from typing import TypeVar, Generic, Tuple, Literal, Union
from abc import abstractmethod, ABC

import pandas as pd
import matplotlib.pyplot as plt
from laptrack import LapTrack

from ..factories.mb_support import tracking
from .spot import Spot
from .cell_spot import CellSpot
from .mid_body_spot import MidBodySpot

T = TypeVar("T", MidBodySpot, CellSpot)

TRACKING_METHOD = Union[Literal["laptrack", "spatial_laptrack"], LapTrack]


class Track(ABC, Generic[T]):
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

    def add_spot(self, spot: Spot) -> None:
        """
        Add spot to track.
        """
        self.spots[spot.frame] = spot
        spot.track_id = self.track_id
        self.length += 1
        # Add children recursively
        if spot.child_spot is not None:
            self.add_spot(spot.child_spot)

    @staticmethod
    def generate_tracks_from_spots(
        spot_type: T,
        spots: dict[int, list[T]],
        method: TRACKING_METHOD,
        show_post_conv_df: bool = False,
        show_tracking_df: bool = False,
        show_tracking_plot: bool = False,
    ) -> list[Track[T]]:
        """
        Generate tracks from spots.
        """

        raise RuntimeError(
            "Sorry, this is broken for now, please use gen_track.generate_tracks_from_spots instead"
        )

        spot_df = Track.convert_spots_to_spotdf(
            spot_type,
            spots,
            show_post_conv_df,
        )
        track_df, _, _ = Track.apply_tracking(
            spot_type,
            spot_df,
            method,
            show_tracking_df,
        )
        if show_tracking_plot:
            Track.generate_tracking_plot(track_df)

        return Track.track_df_to_mb_track(track_df, spots)

    @staticmethod
    def convert_spots_to_spotdf(
        spot: T,
        spot_dict: dict[int, list[Spot]],
        show_post_conv_df: bool = False,
    ) -> list[Track]:

        cols = [
            "frame",
            "x",
            "y",
            "idx_in_frame",
        ]

        cols.extend(spot.get_extra_features_name())

        spot_df = pd.DataFrame({c: [] for c in cols})

        for frame, spots in spot_dict.items():
            if len(spots) == 0:
                spot_df.loc[len(spot_df.index)] = [
                    frame,
                    *[
                        None for _ in range(len(cols) - 1)
                    ],  # fills the rest of the cols with None
                ]
            else:
                for idx, spot in enumerate(spots):
                    features = [spot.frame, spot.x, spot.y, idx]
                    features.extend(spot.get_extra_coordinates())
                    spot_df.loc[len(spot_df.index)] = features

        if show_post_conv_df:
            print(spot_df)

        return spot_df

    @staticmethod
    def apply_tracking(
        spot: T,
        spot_df: pd.DataFrame,
        method: TRACKING_METHOD,
        show_tracking_df: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        tracker: LapTrack = None
        if isinstance(method, str):
            mapping: dict[str, LapTrack] = {
                "laptrack": tracking.cur_laptrack,
                "lt": tracking.lt,
                "spatial_laptrack": tracking.cur_spatial_laptrack,
                "slt": tracking.slt,
            }
            tracker = mapping.get(method, None)
            if tracker is None:
                raise RuntimeError(f"Unknown tracking string: [{method}]")
        elif isinstance(method, LapTrack):
            tracker = method
        else:
            raise RuntimeError(
                "mode must be either a str or a LapTrack object"
            )

        coord_cols = ["x", "y"]
        coord_cols.extend(spot.get_extra_features_name())

        df = tracker.predict_dataframe(
            spot_df,
            coord_cols,
            only_coordinate_cols=False,
        )

        if show_tracking_df:
            # df[0] is the tracking df, what we are mainly interested in.
            # [1] is splitting df, [2] is merging df (you can try the other values)
            # if you want.
            #
            # Pandas by default has a print size limit. You can change the settings
            # or call the 'to_string()' method on the dataframe
            # (as long as it is not too big, our case should be fine).
            print(df[0])

        return df

    @staticmethod
    def generate_tracking_plot(track_df: pd.DataFrame):
        plt.figure(figsize=(3, 3))
        frames = track_df.index.get_level_values("frame")
        frame_range = [frames.min(), frames.max()]
        # k1, k2 = "position_y", "position_x"
        k1, k2 = "y", "x"
        keys = [k1, k2]

        for _, grp in track_df.groupby("track_id"):
            df = grp.reset_index().sort_values("frame")
            plt.scatter(
                df[k1],
                df[k2],
                c=df["frame"],
                vmin=frame_range[0],
                vmax=frame_range[1],
            )
            for i in range(len(df) - 1):
                pos1 = df.iloc[i][keys]
                pos2 = df.iloc[i + 1][keys]
                plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], "-k")

        plt.show()

    @staticmethod
    @abstractmethod
    def track_df_to_mb_track(
        track_df: pd.DataFrame,
        spots: dict[int, list[Spot]],
    ) -> list[Track]:
        """This is the last step of the 'generate_tracks_from_spots'.
        It takes as input the pandas dataframe produced by 'generate_tracks_from_spots'.
        From there you have to implement the code that will transform this dataframe
        into a list of Track.

        You can use MidbodyTrack's implementation as a starting point.
        """
        return []
