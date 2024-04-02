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
    
       
    # TODO: create spots from Cellpose results
    # TODO: perform tracking using laptrack

    # Load TrackMate results to compare... make sure they match!
    trackmate_tracks, trackmate_spots = load_tracks_and_spots(
        trackmate_tracks_path, spots_path
    )
    
    # Frame of interest
    frame = 0

    # Plot cellpose_results
    
    plt.figure()
    plt.imshow(cellpose_results[frame])
    plt.show()
    plt.close()

    # Plot trackmate_spots of frame number "frame" and their barycenters
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
    def barycenters(frame):
        max = np.max(cellpose_results[0])
        mx=[]
        my=[]
        for i in range(1,max+1):
            A = np.where(cellpose_results[0] == i)
            Sx = np.sum(A[1])
            Sy = np.sum(A[0])
            mx.append(Sx/len(A[1]))
            my.append(Sy/len(A[0]))
        return(mx,my)
    barycenters(frame)
    plt.figure()
    plt.imshow(cellpose_results[frame])
    plt.plot(barycenters(frame)[0],barycenters(frame)[1],'o')
    plt.show()
    plt.close()


    # TODO: generate CellSpot instances
    cell_dictionary: dict[int, list[CellSpot]] = {}
    for frame in range(len(cellpose_results)):
        for id_number in range(1, np.max(cellpose_results[frame]) + 1):
            x,y = barycenters(frame)
            A = np.where(cellpose_results[frame] == id_number)
            hull = ConvexHull(A)
            convex_hull_indices = A[hull.vertices][:, ::-1]  # (x, y)
            spot_points = convex_hull_indices
            plt.figure()
            plt.plot(spot_points[:, 0], spot_points[:, 1], 'o')
            plt.show()
            plt.close()
            abs_min_x, abs_max_x, abs_min_y, abs_max_y = np.abs(np.min(A[1])), np.abs(np.max(A[1])), np.abs(np.min(A[0])), np.abs(np.max(A[0]))

            cell_spot = CellSpot(frame, x, y, id_number, abs_min_x, abs_max_x, abs_min_y, abs_max_y, spot_points)
    # Spot points can be created from the cell indices
     
    # The indices of points forming the convex hull
     


if __name__ == "__main__":
    main()

