""" Initialized version of SpatialLapTrack
"""
from functools import partial
from .dist_func import spatial_intensity_dist
from .spatial_laptrack import SpatialLapTrack
from laptrack import LapTrack, ParallelBackend


SLT_MAX_DISTANCE = 175

slt_dist = partial(
    spatial_intensity_dist,
    max_distance       = SLT_MAX_DISTANCE,
    mklp_weight_factor = 5.0,
    sir_weight_factor  = 1.50,
)

slt = SpatialLapTrack(
    spatial_coord_slice         = slice(0,2),
    spatial_metric              = "euclidean",
    track_dist_metric           = slt_dist,
    track_cost_cutoff           = SLT_MAX_DISTANCE,
    gap_closing_dist_metric     = slt_dist,
    gap_closing_cost_cutoff     = SLT_MAX_DISTANCE,
    gap_closing_max_frame_count = 3,
    splitting_cost_cutoff       = False,
    merging_cost_cutoff         = False,
    alternative_cost_percentile = 100,
)

parallel_slt = SpatialLapTrack(
    spatial_coord_slice         = slice(0,2),
    spatial_metric              = "euclidean",
    track_dist_metric           = slt_dist,
    track_cost_cutoff           = SLT_MAX_DISTANCE,
    gap_closing_dist_metric     = slt_dist,
    gap_closing_cost_cutoff     = SLT_MAX_DISTANCE,
    gap_closing_max_frame_count = 3,
    splitting_cost_cutoff       = False,
    merging_cost_cutoff         = False,
    alternative_cost_percentile = 100,
    parallel_backend            = ParallelBackend.ray
)


LT_MAX_DISTANCE = 175

lt_dist = partial(
    spatial_intensity_dist,
    max_distance       = LT_MAX_DISTANCE,
    mklp_weight_factor = 5.0,
    sir_weight_factor  = 1.50,
)

lt = LapTrack(
    track_dist_metric           = lt_dist,
    track_cost_cutoff           = LT_MAX_DISTANCE**2,
    gap_closing_dist_metric     = lt_dist,
    gap_closing_cost_cutoff     = LT_MAX_DISTANCE**2,
    gap_closing_max_frame_count = 2,
    splitting_cost_cutoff       = False,
    merging_cost_cutoff         = False,
    alternative_cost_percentile = 90
)

