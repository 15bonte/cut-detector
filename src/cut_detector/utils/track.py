from __future__ import annotations
from abc import abstractmethod
from typing import TypeVar, Generic, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from laptrack import LapTrack

from cut_detector.factories.mb_support import tracking
from .spot import Spot
from .cell_spot import CellSpot
from .mid_body_spot import MidBodySpot

T = TypeVar("T", MidBodySpot, CellSpot)
TRACKING_MODE = str


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
    @abstractmethod
    def generate_tracks_from_spots(
        spot_type: type,
        spots: dict[int, list[T]],
        mode: TRACKING_MODE | Callable[[np.ndarray], np.ndarray],
        show_post_conv_df: bool = False,
        show_tracking_df: bool = False,
        show_tracking_plot: bool = False,
    ) -> list[Track[T]]:
        """
        Generate tracks from spots.
        """
        return generate_tracks_from_spot_dict(
            spot_type,
            spots,
            mode,
            show_post_conv_df,
            show_tracking_df,
            show_tracking_plot
        )



def generate_tracks_from_spot_dict(
        spot_kind: type, 
        spot_dict: Dict[int, List[Spot]],
        mode: TRACKING_MODE | LapTrack = tracking.cur_spatial_laptrack,
        show_post_conv_df: bool = False,
        show_tracking_df: bool = False,
        show_tracking_plot: bool = False,
        ) -> List[Track]:
    """
    Although 'type' is used for spot_kind, the expected type
    in reality is a class that implements Spot.
    """
    spot_df = convert_spots_to_spotdf(
        spot_kind,
        spot_dict,
        show_post_conv_df,
    )
    track_df, _, _ = apply_tracking(
        spot_kind, 
        spot_df,
        mode,
        show_tracking_df,
    )
    if show_tracking_plot:
        generate_tracking_plot(track_df)

    return track_df_to_mb_track(track_df, spot_dict)


def convert_spots_to_spotdf(
        spot_kind: type, 
        spot_dict: Dict[int, List[Spot]],
        show_post_conv_df: bool = False) -> List[Track]:
    
    cols = [
        "frame",
        "x",
        "y",
        "idx_in_frame",
    ]
    cols.extend(spot_kind.get_extra_features_name())

    spot_df = pd.DataFrame({c: [] for c in cols})

    for frame, spots in spot_dict.items():
        if len(spots) == 0:
            spot_df.loc[len(spot_df.index)] = [
                frame,
                *[None for _ in range(cols)-1] # fills the rest of the cols with None
            ]
        else:
            for idx, spot in enumerate(spots):
                features = [spot.frame, spot.x, spot.y, idx]
                features.extend(spot.get_extra_coordinates())
                spot_df.loc[len(spot_df.index)] = features

    if show_post_conv_df:
        print(spot_df)

    return spot_df


def apply_tracking(
        spot: type,
        spot_df: pd.DataFrame, 
        mode: TRACKING_MODE | LapTrack,  # str support is for legacy code
        show_tracking_df: bool = False
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    tracker: LapTrack = None
    if isinstance(mode, str):
        mapping: Dict[str, LapTrack] = {
            "laptrack":         tracking.cur_laptrack,
            "lt":               tracking.lt,
            "spatial_laptrack": tracking.cur_spatial_laptrack,
            "slt":              tracking.slt,
        }
        tracker = mapping.get(mode, None)
        if tracker is None:
            raise RuntimeError(f"Unknown tracking string: [{mode}]")
    elif isinstance(mode, LapTrack):
        tracker = mode
    else:
        raise RuntimeError("mode must be either a str or a LapTrack object")
    
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


def generate_tracking_plot(track_df: pd.DataFrame):
    def get_track_end(track_df, keys, track_id, first=True):
        df = track_df[track_df["track_id"] == track_id].sort_index(
            level="frame"
        )
        return df.iloc[0 if first else -1][keys]

    keys = ["position_x", "position_y", "track_id", "tree_id"]
    plt.figure(figsize=(3, 3))
    frames = track_df.index.get_level_values("frame")
    frame_range = [frames.min(), frames.max()]
    # k1, k2 = "position_y", "position_x"
    k1, k2 = "y", "x"
    keys = [k1, k2]

    for track_id, grp in track_df.groupby("track_id"):
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


def track_df_to_mb_track(
        track_df: pd.DataFrame,
        spots: Dict[int, List[Spot]],
        ) -> List[Track]:
    
    track_df.reset_index(inplace=True)
    track_df.dropna(inplace=True)
    id_to_track = {}

    for _, row in track_df.iterrows():
        track_id = row["track_id"]
        track: Track = id_to_track.get(track_id)
        if track is None:
            id_to_track[track_id] = Track(len(id_to_track))
            track = id_to_track[track_id]
        frame        = row["frame"]
        idx_in_frame = row["idx_in_frame"]
        track.add_spot(spots[int(frame)][int(idx_in_frame)])

    return list(id_to_track.values())




