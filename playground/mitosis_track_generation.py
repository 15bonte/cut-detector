"""Playground to run mitosis track generation."""

import os
from typing import Optional
from matplotlib import pyplot as plt

from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector.data.tools import get_data_path
from cut_detector.utils.cell_spot import CellSpot
from cut_detector.utils.cell_track import CellTrack
from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.utils.tools import re_organize_channels
from cut_detector.widget_functions.mitosis_track_generation import (
    perform_mitosis_track_generation,
)


def plot_predictions_evolution(
    raw_spots: list[CellSpot],
    raw_tracks: list[CellTrack],
    mitosis_tracks: list[MitosisTrack],
) -> None:
    """
    Plot predictions evolution. Used in playground.

    Parameters
    ----------
    raw_spots : list[CellSpot]
        Raw spots.
    raw_tracks : list[CellTrack]
        Raw tracks.
    mitosis_tracks : list[MitosisTrack]
        Mitosis tracks.
    """
    # Spots detected by Cellpose
    detected_spots = {}
    for raw_spot in raw_spots:
        if raw_spot.frame not in detected_spots:
            detected_spots[raw_spot.frame] = 0
        detected_spots[raw_spot.frame] += 1

    # Mitoses identified by current method, minus ending tracks
    detected_mitoses = {frame: detected_spots[0] for frame in detected_spots}
    max_frame = max(detected_spots.keys())
    for mitosis_track in mitosis_tracks:
        for frame in range(
            mitosis_track.metaphase_sequence.last_frame, max_frame + 1
        ):
            detected_mitoses[frame] += len(mitosis_track.daughter_track_ids)

    min_frame, max_frame = 0, max(detected_spots.keys())

    detected_tracks = {}
    metaphase_frames = []
    for track in raw_tracks:
        metaphase_frames = metaphase_frames + [
            metaphase_sequence.last_frame
            for metaphase_sequence in track.metaphase_sequences
        ]
        for frame in range(track.start, track.stop + 1):
            if frame not in detected_tracks:
                detected_tracks[frame] = 0
            detected_tracks[frame] += 1

    data_spots = [detected_spots[0]] + [
        detected_spots[i] if i in detected_spots else 0
        for i in range(min_frame, max_frame)
    ]
    data_tracks = [detected_tracks[0]] + [
        detected_tracks[i] if i in detected_tracks else 0
        for i in range(min_frame, max_frame)
    ]
    data_mitoses = [detected_mitoses[0]] + [
        detected_mitoses[i] if i in detected_mitoses else 0
        for i in range(min_frame, max_frame)
    ]

    _, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.step(
        list(range(min_frame, max_frame + 1)), data_spots, "r", linewidth=8.0
    )
    ax.step(
        list(range(min_frame, max_frame + 1)), data_tracks, "g", linewidth=8.0
    )
    ax.step(
        list(range(min_frame, max_frame + 1)), data_mitoses, "b", linewidth=8.0
    )

    ending_tracks = [track.stop + 1 for track in raw_tracks]

    # Plot first for legend
    ax.axvline(-10, color="y", linewidth=2.0)
    ax.axvline(-10, color="c", linewidth=2.0)
    ax.axvline(-10, color="k", linewidth=2.0, linestyle="--")

    # Potential mitoses
    metaphase_frames = list(set(metaphase_frames))
    for metaphase_spot_frame in metaphase_frames:
        ax.axvline(metaphase_spot_frame, color="y", linewidth=2.0)

    # Actual mitoses
    actual_mitoses = [
        mitosis.metaphase_sequence.last_frame for mitosis in mitosis_tracks
    ]
    for actual_mitosis in actual_mitoses:
        ax.axvline(actual_mitosis, color="c", linewidth=2.0)

    # Ending tracks
    for ending_track_frame in ending_tracks:
        ax.axvline(
            ending_track_frame, color="k", linewidth=2.0, linestyle="--"
        )

    ax.set_xlim(min_frame, max_frame)

    ax.legend(
        [
            "Spots",
            "Tracks",
            "Mitoses evolution",
            "Potential mitoses",
            "Actual mitoses",
            "Ending tracks",
        ],
        loc="best",
    )
    plt.show()


def main(
    image_path: Optional[str] = get_data_path("videos"),
    video_name: Optional[str] = "example_video",
    spots_dir: Optional[str] = get_data_path("spots"),
    tracks_dir: Optional[str] = get_data_path("tracks"),
):
    """
    Parameters
    ----------
    image_path : str
        Path to the image to process.
    video_name : str
        Video name.
    spots_dir : str
        Path to the spots directory.
    tracks_dir : str
        Path to the tracks directory.
    """
    # If image_path or model_path are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])

    # Read image and preprocess if needed
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX
    image = re_organize_channels(image.squeeze())  # TYXC

    mitosis_tracks, cell_spots, cell_tracks = perform_mitosis_track_generation(
        image, video_name, spots_dir, tracks_dir
    )

    plot_predictions_evolution(cell_spots, cell_tracks, mitosis_tracks)


if __name__ == "__main__":
    main()
