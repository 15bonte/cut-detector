import os
import pickle
from typing import Optional
import numpy as np

from matplotlib import pyplot as plt

from cut_detector.data.tools import get_data_path
from cut_detector.factories.mid_body_detection_factory import (
    MidBodyDetectionFactory,
)
from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.utils.trackmate_track import TrackMateTrack


def main(
    mitosis_path: Optional[str] = get_data_path("mitoses"),
    tracks_dir: Optional[str] = os.path.join(
        get_data_path("tracks"), "example_video"
    ),
):
    # Load mitoses
    mitosis_tracks: list[MitosisTrack] = []
    for state_path in os.listdir(mitosis_path):
        # Load mitosis track
        with open(os.path.join(mitosis_path, state_path), "rb") as f:
            mitosis_track: MitosisTrack = pickle.load(f)
            mitosis_track.adapt_deprecated_attributes()
            mitosis_tracks.append(mitosis_track)

    # Load tracks
    cell_tracks: list[TrackMateTrack] = []
    # Iterate over "bin" files in exported_tracks_dir
    for state_path in os.listdir(tracks_dir):
        # Load mitosis track
        with open(os.path.join(tracks_dir, state_path), "rb") as f:
            trackmate_track: TrackMateTrack = pickle.load(f)
            trackmate_track.adapt_deprecated_attributes()
            cell_tracks.append(trackmate_track)

    factory = MidBodyDetectionFactory()
    for mitosis_track in mitosis_tracks:
        positions, mother_points, daughter_points = (
            factory.get_expected_positions(mitosis_track, cell_tracks)
        )
        for frame, position in positions.items():
            absolute_frame = frame + mitosis_track.min_frame
            # Plot mother points with matplotlib
            plt.plot(
                np.array(mother_points[absolute_frame].spot_points)[:, 1],  # y
                np.array(mother_points[absolute_frame].spot_points)[:, 0],  # x
                "ro",
            )
            # Plot daughter points with matplotlib
            plt.plot(
                np.array(daughter_points[absolute_frame].spot_points)[:, 1],
                np.array(daughter_points[absolute_frame].spot_points)[:, 0],
                "bo",
            )
            # Plot expected position with matplotlib
            plt.plot(
                position[1]
                + mitosis_track.position.min_y,  # to make it absolute
                position[0] + mitosis_track.position.min_x,
                "go",
            )
            plt.show()


if __name__ == "__main__":
    main()
