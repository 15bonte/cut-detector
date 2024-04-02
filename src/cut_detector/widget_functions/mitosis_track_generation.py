import os
from typing import Optional, Union
import pickle
import numpy as np
from matplotlib import pyplot as plt

from cnn_framework.utils.display_tools import display_progress

from ..utils.cell_spot import CellSpot
from ..utils.trackmate_track import TrackMateTrack
from ..factories.tracks_merging_factory import TracksMergingFactory
from ..models.tools import get_model_path
from ..utils.mitosis_track import MitosisTrack


def plot_predictions_evolution(
    raw_spots: list[CellSpot],
    raw_tracks: list[TrackMateTrack],
    mitosis_tracks: list[MitosisTrack],
) -> None:
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
        for frame in range(mitosis_track.metaphase_frame, max_frame + 1):
            detected_mitoses[frame] += len(mitosis_track.daughter_track_ids)

    min_frame, max_frame = 0, max(detected_spots.keys())

    # Tracks identified by TrackMate
    detected_tracks = {}
    metaphase_spots = []
    for track in raw_tracks:
        metaphase_spots = metaphase_spots + [
            metaphase_spot.frame for metaphase_spot in track.metaphase_spots
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
    metaphase_spots = list(set(metaphase_spots))
    for metaphase_spot_frame in metaphase_spots:
        ax.axvline(metaphase_spot_frame, color="y", linewidth=2.0)

    # Actual mitoses
    actual_mitoses = [mitosis.metaphase_frame for mitosis in mitosis_tracks]
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


def perform_mitosis_track_generation(
    raw_video: np.ndarray,
    video_name: str,
    xml_model_dir: str,
    mitoses_save_dir: Optional[str] = None,
    tracks_save_dir: Optional[str] = None,
    metaphase_model_path: Optional[str] = get_model_path("metaphase_cnn"),
    hmm_metaphase_parameters_file: Optional[str] = os.path.join(
        get_model_path("hmm"), "hmm_metaphase_parameters.npz"
    ),
    predictions_file: Optional[str] = None,
    only_predictions_update: bool = False,
    plot_evolution: bool = False,
) -> Union[list[MitosisTrack], None]:
    """
    Perform mitosis track generation.
    """
    # Create save_dir if not exists
    if mitoses_save_dir is not None and not os.path.exists(mitoses_save_dir):
        os.makedirs(mitoses_save_dir)

    # Create video tracks save dir if not exists
    if tracks_save_dir is not None:
        video_tracks_save_dir = os.path.join(tracks_save_dir, video_name)
        if not os.path.exists(video_tracks_save_dir):
            os.makedirs(video_tracks_save_dir)
    else:
        video_tracks_save_dir = None

    # Create factory instance, where useful functions are defined
    tracks_merging_factory = TracksMergingFactory()

    # Read useful information from xml file
    xml_model_path = os.path.join(xml_model_dir, f"{video_name}_model.xml")
    cell_tracks, cell_spots = tracks_merging_factory.read_trackmate_xml(
        xml_model_path, raw_video.shape
    )

    # Missing xml file case
    if cell_tracks is None:
        return None

    # Get dictionary of TrackMate spots (from xml file) for each track and detect metaphase spots
    tracks_merging_factory.pre_process_spots(
        cell_tracks,
        cell_spots,
        raw_video,
        metaphase_model_path,
        hmm_metaphase_parameters_file,
        predictions_file,
        video_name,
        only_predictions_update,
    )

    # If the goal is only to update predictions file, stop here
    if only_predictions_update:
        return None

    print("\nGet tracks to merge...")

    # Plug tracks occurring at frame>0 to closest metaphase
    mitosis_tracks = tracks_merging_factory.get_tracks_to_merge(cell_tracks)

    # Plot predictions evolution
    if plot_evolution:
        plot_predictions_evolution(cell_spots, cell_tracks, mitosis_tracks)

    # Update useful attributes for each track
    for i, mitosis_track in enumerate(mitosis_tracks):
        mitosis_track.id = i
        mitosis_track.update_mitosis_start_end(cell_tracks, mitosis_tracks)
        mitosis_track.update_key_events_frame(cell_tracks)
        mitosis_track.update_mitosis_position_dln(cell_tracks)
        mitosis_track.update_is_near_border(raw_video)

        # Save mitosis track
        if mitoses_save_dir is not None:
            daughter_track_ids = ",".join(
                [str(d) for d in mitosis_track.daughter_track_ids]
            )
            state_path = f"{video_name}_mitosis_{i}_{mitosis_track.mother_track_id}_to_{daughter_track_ids}.bin"
            save_path = os.path.join(
                mitoses_save_dir,
                state_path,
            )
            with open(save_path, "wb") as f:
                pickle.dump(mitosis_track, f)

        display_progress(
            "Mitosis tracks generation:",
            i + 1,
            len(mitosis_tracks),
            additional_message=f"Mitosis {i + 1}/{len(mitosis_tracks)}",
        )

    # Save updated cell tracks
    if video_tracks_save_dir is not None:
        for cell_track in cell_tracks:
            state_path = f"track_{cell_track.track_id}.bin"
            save_path = os.path.join(
                video_tracks_save_dir,
                state_path,
            )
            with open(save_path, "wb") as f:
                pickle.dump(cell_track, f)

    return mitosis_tracks
