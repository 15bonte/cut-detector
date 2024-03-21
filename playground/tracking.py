import os
import pickle
from typing import Optional

from cut_detector.data.tools import get_data_path
from cut_detector.utils.cell_spot import CellSpot
from cut_detector.utils.trackmate_track import TrackMateTrack
from cut_detector.utils.trackmate_spot import TrackMateSpot

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

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
    plt.figure()
    plt.imshow(cellpose_results[0])
    #plt.show()
    plt.close()
    
    max = np.max(cellpose_results[0])
    for i in range(max):
           A = np.where(cellpose_results[0] == i)
           Sx = np.sum(A[0])
           Sy = np.sum(A[1])
           mx=Sx/len(A[0])
           my=Sy/len(A[1])



            
    # TODO: create spots from Cellpose results
    # TODO: perform tracking using laptrack

    # Load TrackMate results to compare... make sure they match!
    trackmate_tracks, trackmate_spots = load_tracks_and_spots(
        trackmate_tracks_path, spots_path
    )
    
    # Frame of interest
    frame = 0

    # Plot cellpose_results
    print(cellpose_results[frame])
    plt.figure()
    plt.imshow(cellpose_results[frame])
    plt.show()
    plt.close()

    # Plot trackmate_spots of frame number "frame"
    y = []
    x = []
    
    for s in trackmate_spots:
        if s.frame == frame:
            point_list = s.spot_points
            for i in range(len(point_list)):
                x.append(point_list[i][0])
                y.append(600 - point_list[i][1])
    plt.scatter(x,y)
    plt.show()

    # Finding barycenters of each cell
    for i in range(1,2):
        indices = np.where(cellpose_results[frame]==i)
        #print(indices)

    # TODO: generate CellSpot instances
    cell_dictionary: dict[int, list[CellSpot]] = {}
    # cell_spot = CellSpot(frame, x, y, id_number, abs_min_x, abs_max_x, abs_min_y, abs_max_y, spot_points)
    # Spot points can be created from the cell indices
    # hull = ConvexHull(indices)
    # The indices of points forming the convex hull
    # convex_hull_indices = indices[hull.vertices][:, ::-1]  # (x, y)


if __name__ == "__main__":
    main()

