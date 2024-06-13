""" A module that relies on the Track Abstract Base Class in order to
generate specialized tracks from points.

See Track documentation to learn how to create new specialized Tracks/Spots
"""

from typing import Tuple, TypeVar

import pandas as pd
from laptrack import LapTrack

from .track import Track
from .spot import Spot
from .mid_body_track import MidBodyTrack
from .cell_track import CellTrack
from .mid_body_spot import MidBodySpot
from .cell_spot import CellSpot

SPOT_AND_TRACK_MAPPING = {
    MidBodySpot: MidBodyTrack,
    CellSpot: CellTrack,
}

T = TypeVar("T", *SPOT_AND_TRACK_MAPPING.values())
S = TypeVar("S", *SPOT_AND_TRACK_MAPPING.keys())


def generate_tracks_from_spots(
    spot_dict: dict[int, list[S]], method: LapTrack
) -> list[T]:
    """Generate a list of specialized tracks based on the underlying kind
    of spot.
    Spots must all be of the same type according to isinstance (more specifically,
    they must all share the same type as the first one).

    Parameters
    ----------
    spot_dict : dict[int, list[S]]
        A dictionary mapping frames to a list of spots.
    method : LapTrack
        The tracking method to use.

    Returns
    -------
    list[T]
        A list of specialized tracks.
    """

    if is_spot_dict_empty(spot_dict):
        return []

    inferred_spot_kind: S = validate_inferred_spot_kind(
        infer_spot_kind(spot_dict)
    )
    inferred_track_kind: T = infer_specialized_track_kind(inferred_spot_kind)

    spot_df = convert_spots_to_spotdf(spot_dict, inferred_spot_kind)
    track_df, _, _ = apply_tracking(inferred_spot_kind, spot_df, method)
    track_list = inferred_track_kind.track_df_to_track_list(
        track_df, spot_dict
    )

    return track_list


def is_spot_dict_empty(spot_dict: dict[int, list[Spot]]) -> bool:
    """Checks if a spot_dict is empty, meaning that all values are empty lists

    Parameters
    ----------
    spot_dict : dict[int, list[Spot]]
        A dictionary mapping frames to a list of spots

    Returns
    -------
    bool
        True if the spot_dict is empty, False otherwise
    """
    if len(spot_dict.values()) == 0:
        return True

    for v in spot_dict.values():
        if len(v) > 0:
            return False

    return True


def validate_inferred_spot_kind(inference: type) -> S:
    """Validates the inference of the kind of spot.

    Parameters
    ----------
    inference : type
        The inferred kind of spot

    Returns
    -------
    S
        The kind of spot if it is valid
    """
    if inference in SPOT_AND_TRACK_MAPPING:
        return inference
    raise RuntimeError(
        f"Tracks can only be built from {SPOT_AND_TRACK_MAPPING.keys()}, encountered {inference} instead"
    )


def infer_spot_kind(spot_dict: dict[int, list[Spot]]) -> type:
    """Infers the kind of spot from the first non-empty list of spots in the
    spot_dict.

    Parameters
    ----------
    spot_dict : dict[int, list[Spot]]
        A dictionary mapping frames to a list of spots

    Returns
    -------
    type
        The kind of spot
    """
    for v in spot_dict.values():
        if len(v) != 0:
            t = type(v[0])
            if t is None:
                raise RuntimeError(
                    "None in v[0] found in spot_dict:\n{spot_dict}"
                )
            return t

    # If we land here, it means that spot_dict.values is empty or only contains empty point lists.
    # This case must be handled earlier
    raise RuntimeError("No values in dict")


def infer_specialized_track_kind(spot_kind: type[Spot]) -> type[Track]:
    """Infers the specialized type of track from the kind of Spot.
    If no specialized kind of track is associated with the kind of
    spot, this function raises an error.

    Parameters
    ----------
    spot_kind : type[Spot]
        The kind of spot

    Returns
    -------
    type[Track]
        The specialized kind of track
    """
    kind_mapping = {MidBodySpot: MidBodyTrack, CellSpot: CellTrack}
    inferred_specialized_track_kind = kind_mapping.get(spot_kind)
    if infer_specialized_track_kind is None:
        raise RuntimeError(
            f"No known specialized track is associated with {type(spot_kind)}"
        )
    return inferred_specialized_track_kind


def convert_spots_to_spotdf(
    spot_dict: dict[int, list[Spot]], spot_kind: type[Spot]
) -> pd.DataFrame:
    """Converts a dictionary of spots to a DataFrame.

    Parameters
    ----------
    spot_dict : dict[int, list[Spot]]
        A dictionary mapping frames to a list of spots
    spot_kind : type[Spot]
        The kind of spot

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the spots
    """

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
                *[
                    None for _ in range(len(cols) - 1)
                ],  # fills the rest of the cols with None
            ]
        else:
            for idx, spot in enumerate(spots):
                assert (
                    spot.frame == frame
                ), "spot.frame and frame cols must be the same"
                features = [spot.frame, spot.x, spot.y, idx]
                features.extend(spot.get_extra_coordinates())
                spot_df.loc[len(spot_df.index)] = features

    # Cast frame to int for laptrack
    spot_df["frame"] = spot_df["frame"].astype(int)

    return spot_df


def apply_tracking(
    spot_kind: type[Spot],
    spot_df: pd.DataFrame,
    method: LapTrack,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Applies the tracking method to the spot DataFrame.

    Parameters
    ----------
    spot_kind : type[Spot]
        The kind of spot
    spot_df : pd.DataFrame
        A DataFrame containing the spots
    method : LapTrack
        The tracking method to use

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the DataFrame of the tracks, the DataFrame of the
        tracks' features, and the DataFrame of the tracks' links
    """
    coord_cols = ["x", "y"]
    coord_cols.extend(spot_kind.get_extra_features_name())

    tuple_df = method.predict_dataframe(
        spot_df,
        coord_cols,
        only_coordinate_cols=False,
    )

    return tuple_df
