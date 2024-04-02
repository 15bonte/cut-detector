import os
import pickle
from typing import Optional

from cut_detector.data.tools import get_data_path
from cut_detector.utils.trackmate_track import TrackMateTrack
from cut_detector.utils.trackmate_spot import TrackMateSpot

import matplotlib.pyplot as plt
import numpy as np

def load_tracks_and_spots(
    trackmate_tracks_path: str, spots_path: str
) -> tuple[list[TrackMateTrack], list[TrackMateSpot]]:
    """
    Load saved spots and tracks generated from Trackmate xml file.
    """
    trackmate_tracks: list[TrackMateTrack] = []
    for track_file in os.listdir(trackmate_tracks_path):
        with open(os.path.join(trackmate_tracks_path, track_file), "rb") as f:
            trackmate_tracks.append(pickle.load(f))

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
    print(cellpose_results[frame])
    plt.figure()
    plt.imshow(cellpose_results[frame])
    plt.show()
    plt.close()

    # Plot trackmate_spots of frame number "frame" and their barycenters
    y = []
    x = []
    
    for s in trackmate_spots:
        if s.frame == frame:
            list = s.spot_points
            for i in range(len(list)):
                x.append(list[i][0])
                y.append(600 - list[i][1])

    def barycenters(frame_number):
        b = []
        max = np.max(cellpose_results[0])
        for i in range(1,max+1):
            A = np.where(cellpose_results[frame_number]==i)
            l = len(x)
            if l == 0:
                break
            X = np.sum(A[0])
            Y = np.sum(A[1])
            b.append([X/len(A[0]),Y/len(A[1])])
        return b
    
    b = barycenters(frame)
    x_b = []
    y_b = []
    for i in range(len(b)):
        y_b.append(b[i][0])
        x_b.append(b[i][1])
    plt.figure()
    plt.scatter(x,y)
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(cellpose_results[frame])
    plt.scatter(x_b,y_b)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

