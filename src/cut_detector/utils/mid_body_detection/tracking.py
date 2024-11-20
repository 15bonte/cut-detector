"""Define optimal tracking functions for every strategy."""

from functools import partial
from typing import Union
from laptrack import LapTrack
import numpy as np
from scipy.spatial import distance

from .spatial_laptrack import SpatialLapTrack


def spatial_intensity_dist(
    c1: tuple,
    c2: tuple,
    max_distance: Union[int, float],
    mklp_weight_factor: float,
    sir_weight_factor: float,
) -> float:
    """Modified version of sqeuclidian distance
    Square Euclidian distance is applied to spatial coordinates x and y.
    An 'intensity' distance is computed with MLKP and SIR intensities.
    Finally, values are combined by weighted addition.

    Parameters
    ----------
    c1 : tuple
        Tuple containing spatial coordinates and intensities of the first point.
    c2 : tuple
        Tuple containing spatial coordinates and intensities of the second point.
    max_distance : Union[int, float]
        Maximum distance for connection.
    mklp_weight_factor : float
        Weight factor for MLKP intensity.
    sir_weight_factor : float
        Weight factor for SIR intensity.

    Returns
    -------
    float
        Distance between two points.
    """

    # unwrapping
    (x1, y1, mlkp1, sir1), (x2, y2, mlkp2, sir2) = c1, c2

    # In case we have a None None point:
    if np.isnan([x1, y1, x2, y2]).any():
        return max_distance * 2  # connection is invalidated

    # spatial coordinates: euclidean
    spatial_e = distance.euclidean([x1, y1], [x2, y2])

    mklp_penalty = (
        3 * mklp_weight_factor * np.abs(mlkp1 - mlkp2) / (mlkp1 + mlkp2)
    )
    sir_penalty = 3 * sir_weight_factor * np.abs(sir1 - sir2) / (sir1 + sir2)
    penalty = 1 + mklp_penalty + sir_penalty
    return (spatial_e * penalty) ** 2


def get_tracking_method(
    method_name: str, spatial_resolution: int, max_distance=39.375
) -> Union[LapTrack, SpatialLapTrack]:
    """Get tracking method by name.

    Parameters
    ----------
    method_name : str
        Name of the tracking method.
    spatial_resolution : int
        Spatial resolution.
    max_distance : float
        Maximum distance for connection (um).

    Returns
    -------
    LapTrack or SpatialLapTrack
        Tracking method.
    """
    max_distance_px = int(max_distance / spatial_resolution * 1000)  # px

    if method_name == "spatial_laptrack":

        custom_spatial_distance = partial(
            spatial_intensity_dist,
            max_distance=max_distance_px,
            mklp_weight_factor=5.0,
            sir_weight_factor=1.50,
        )

        return SpatialLapTrack(
            spatial_coord_slice=slice(0, 2),
            spatial_metric="euclidean",
            track_dist_metric=custom_spatial_distance,
            track_cost_cutoff=max_distance_px,
            gap_closing_dist_metric=custom_spatial_distance,
            gap_closing_cost_cutoff=max_distance_px,
            gap_closing_max_frame_count=3,
            splitting_cost_cutoff=False,
            merging_cost_cutoff=False,
            alternative_cost_percentile=100,
        )

    if method_name == "laptrack":

        custom_distance = partial(
            spatial_intensity_dist,
            max_distance=max_distance_px,
            mklp_weight_factor=5.0,
            sir_weight_factor=1.50,
        )

        return LapTrack(
            track_dist_metric=custom_distance,
            track_cost_cutoff=max_distance_px**2,
            gap_closing_dist_metric=custom_distance,
            gap_closing_cost_cutoff=max_distance_px**2,
            gap_closing_max_frame_count=2,
            splitting_cost_cutoff=False,
            merging_cost_cutoff=False,
            alternative_cost_percentile=90,
        )

    raise ValueError(f"Method {method_name} not implemented.")
