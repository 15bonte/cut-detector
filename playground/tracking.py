import os
import pickle
from typing import Optional
import matplotlib.pyplot as plt

from cut_detector.data.tools import get_data_path
from cut_detector.utils.segmentation_tracking.mask_utils import (
    mask_to_polygons,
    simplify,
)
from cut_detector.utils.trackmate_track import TrackMateTrack
from cut_detector.utils.trackmate_spot import TrackMateSpot


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

    polygons = mask_to_polygons(cellpose_results[0])  # frame 0

    simplified_polygons = []
    for polygon in polygons:
        simplified_polygon = simplify(polygon, interval=2, epsilon=0.5)
        simplified_polygons.append(simplified_polygon)

    # Plot polygons
    plt.subplot(221)
    for polygon in polygons:
        x, y = polygon.x, polygon.y
        plt.plot(y, x)
    plt.imshow(cellpose_results[0], cmap="gray")

    plt.subplot(222)
    for polygon in simplified_polygons:
        x, y = polygon.x, polygon.y
        plt.plot(y, x)
    plt.imshow(cellpose_results[0], cmap="gray")

    # Load TrackMate results to compare... make sure they match!
    trackmate_tracks, trackmate_spots = load_tracks_and_spots(
        trackmate_tracks_path, spots_path
    )

    def plot_spots(frame):
        for s in trackmate_spots:
            y = []
            x = []
            if s.frame == frame:
                point_list = s.spot_points
                for i in range(len(point_list)):
                    x.append(point_list[i][0])
                    y.append(point_list[i][1])
            plt.plot(x, y)

    plt.subplot(224)
    plot_spots(0)
    plt.imshow(cellpose_results[0], cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()
