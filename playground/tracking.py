import os
import pickle
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from cut_detector.data.tools import get_data_path
from cut_detector.utils.segmentation_tracking.mask_utils import (
    mask_to_polygons,
    simplify,
)
from cut_detector.factories.segmentation_tracking_factory import (
    SegmentationTrackingFactory,
)
from cut_detector.utils.mb_support.tracking.spatial_laptrack import (
    SpatialLapTrack,
)
from cut_detector.utils.trackmate_track import TrackMateTrack
from cut_detector.utils.trackmate_spot import TrackMateSpot
from cut_detector.utils.gen_track import generate_tracks_from_spots


def load_tracks_and_spots(
    trackmate_tracks_path: str, spots_path: str
) -> tuple[list[TrackMateTrack], list[TrackMateSpot]]:
    """
    Load saved spots and tracks generated from Trackmate xml file.
    """
    trackmate_tracks: list[TrackMateTrack] = []
    for track_file in os.listdir(trackmate_tracks_path):
        with open(os.path.join(trackmate_tracks_path, track_file), "rb") as f:
            trackmate_track: TrackMateTrack = pickle.load(f)
            trackmate_track.adapt_deprecated_attributes()
            trackmate_tracks.append(trackmate_track)

    spots: list[TrackMateSpot] = []
    for spot_file in os.listdir(spots_path):
        with open(os.path.join(spots_path, spot_file), "rb") as f:
            spots.append(pickle.load(f))

    return trackmate_tracks, spots


def barycenter(cellpose_results, frame):
    max = np.max(cellpose_results[frame])
    mean_x = []
    mean_y = []
    cells_per_frame = []
    for i in range(1, max + 1):
        cell_indices = np.where(cellpose_results[frame] == i)
        if len(cell_indices[0]) == 0 or len(cell_indices[1]) == 0:
            break
        Sx = np.sum(cell_indices[1])
        Sy = np.sum(cell_indices[0])
        mean_x.append(Sx / len(cell_indices[1]))
        mean_y.append(Sy / len(cell_indices[0]))
        cells_per_frame.append(cell_indices)
    return (mean_x, mean_y, cells_per_frame)


def main(
    segmentation_results_path: Optional[str] = os.path.join(
        get_data_path("segmentation_results"), "example_video.bin"
    ),
    trackmate_tracks_path: Optional[str] = os.path.join(
        get_data_path("tracks"), "example_video"
    ),
    spots_path: Optional[str] = os.path.join(
        get_data_path("spots"), "example_video"
    ),
):
    # Load Cellpose results
    with open(segmentation_results_path, "rb") as f:
        cellpose_results = pickle.load(f)

    # TODO: create spots from Cellpose results
    # TODO: perform tracking using laptrack

    # polygons = mask_to_polygons(cellpose_results[0])  # frame 0

    # simplified_polygons = []
    # for polygon in polygons:
    #     simplified_polygon = simplify(polygon, interval=2, epsilon=0.5)
    #     simplified_polygons.append(simplified_polygon)

    # # Plot polygons
    # plt.subplot(221)
    # for polygon in polygons:
    #     x, y = polygon.x, polygon.y
    #     plt.plot(y, x)
    # plt.imshow(cellpose_results[0], cmap="gray")

    # plt.subplot(222)
    # for polygon in simplified_polygons:
    #     x, y = polygon.x, polygon.y
    #     plt.plot(y, x)
    # plt.imshow(cellpose_results[0], cmap="gray")

    # Load TrackMate results to compare... make sure they match!

    # Frame of interest
    frame = 0

    # Plot cellpose_results
    trackmate_tracks, trackmate_spots = load_tracks_and_spots(
        trackmate_tracks_path, spots_path
    )

    # Plot trackmate_spots of frame number "frame"
    def plot_spots(frame):
        y = []
        x = []
        for s in trackmate_spots:
            if s.frame == frame:
                point_list = s.spot_points
                for i in range(len(point_list)):
                    x.append(point_list[i][0])
                    y.append(600 - point_list[i][1])
        return (x, y)
        # plt.scatter(x,y)
        # plt.show()

    factory = SegmentationTrackingFactory("")
    cell_dictionary = factory.get_spots_from_cellpose(cellpose_results)

    # Spot points can be created from the cell indices

    # The indices of points forming the convex hull
    plot = False
    if plot:
        for frame in range(len(cellpose_results)):
            fig, axarr = plt.subplots(1, 2)
            axarr[0].imshow(cellpose_results[frame])
            for i in range(len(cell_dictionary[frame])):
                axarr[0].plot(
                    cell_dictionary[frame][i].spot_points[:, 1],
                    cell_dictionary[frame][i].spot_points[:, 0],
                    "o",
                )
                axarr[0].plot(
                    cell_dictionary[frame][i].abs_max_x,
                    cell_dictionary[frame][i].abs_max_y,
                    "x",
                )
                axarr[0].plot(
                    cell_dictionary[frame][i].abs_min_x,
                    cell_dictionary[frame][i].abs_max_y,
                    "x",
                )
                axarr[0].plot(
                    cell_dictionary[frame][i].abs_min_x,
                    cell_dictionary[frame][i].abs_min_y,
                    "x",
                )
                axarr[0].plot(
                    cell_dictionary[frame][i].abs_max_x,
                    cell_dictionary[frame][i].abs_min_y,
                    "x",
                )
            axarr[0].plot(
                barycenter(cellpose_results, frame)[0],
                barycenter(cellpose_results, frame)[1],
                "x",
            )
            axarr[1].scatter(plot_spots(frame)[0], plot_spots(frame)[1], s=1)
            plt.show()

    tracking_method = SpatialLapTrack(
        spatial_coord_slice=slice(0, 2),
        spatial_metric="euclidean",
        track_dist_metric="euclidean",
        track_cost_cutoff=85.5,
        gap_closing_dist_metric="euclidean",
        gap_closing_cost_cutoff=85.5,
        gap_closing_max_frame_count=3,
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
        alternative_cost_percentile=100,
    )
    # TODO Compare cell_tracks with trackmate_tracks
    cell_tracks = generate_tracks_from_spots(cell_dictionary, tracking_method)
    print(cell_tracks == trackmate_tracks)


if __name__ == "__main__":
    main()
