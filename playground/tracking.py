import os
import pickle
from typing import Optional
import matplotlib.pyplot as plt

from cut_detector.data.tools import get_data_path
from cut_detector.factories.segmentation_tracking_factory import (
    SegmentationTrackingFactory,
)
from cut_detector.utils.trackmate_track import CellTrack
from cut_detector.utils.cell_spot import CellSpot


def load_tracks_and_spots(
    trackmate_tracks_path: str, spots_path: str
) -> tuple[list[CellTrack], list[CellSpot]]:
    """
    Load saved spots and tracks generated from Trackmate xml file.
    """
    trackmate_tracks: list[CellTrack] = []
    for track_file in os.listdir(trackmate_tracks_path):
        with open(os.path.join(trackmate_tracks_path, track_file), "rb") as f:
            trackmate_track: CellTrack = pickle.load(f)
            trackmate_track.adapt_deprecated_attributes()
            trackmate_tracks.append(trackmate_track)

    spots: list[CellSpot] = []
    for spot_file in os.listdir(spots_path):
        with open(os.path.join(spots_path, spot_file), "rb") as f:
            spots.append(pickle.load(f))

    return trackmate_tracks, spots


def main(
    segmentation_results_path: Optional[str] = os.path.join(
        get_data_path("segmentation_results"), "example_video_old.bin"
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
                for point in point_list:
                    x.append(point[0])
                    y.append(point[1])
        return (x, y)

    def plot_bary(frame):
        y = []
        x = []
        for s in trackmate_spots:
            if s.frame == frame:
                x.append(s.x)
                y.append(s.y)
        return (x, y)

    factory = SegmentationTrackingFactory("")
    cell_spots, cell_tracks = factory.perform_tracking(cellpose_results, 171)

    # Spot points can be created from the cell indices
    plot = True
    if plot:
        # The indices of points forming the convex hull
        for frame, cellpose_result in enumerate(cellpose_results):
            frame_cells = [cell for cell in cell_spots if cell.frame == frame]
            _, axarr = plt.subplots(1, 2)
            axarr[0].imshow(cellpose_result)
            for local_cell in frame_cells:
                axarr[0].plot(
                    local_cell.spot_points[:, 0],
                    local_cell.spot_points[:, 1],
                    "o",
                )
                axarr[0].plot(
                    local_cell.x,
                    local_cell.y,
                    "x",
                )
            axarr[1].imshow(cellpose_result)
            axarr[1].scatter(
                plot_spots(frame)[0],
                plot_spots(frame)[1],
            )
            axarr[1].plot(
                plot_bary(frame)[0],
                plot_bary(frame)[1],
                "x",
            )
            plt.show()


if __name__ == "__main__":
    main()
