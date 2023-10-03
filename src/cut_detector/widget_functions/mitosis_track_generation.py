import os
from typing import Optional, Union
import numpy as np
import xmltodict
import pickle

from cut_detector.constants.tracking import MIN_TRACK_SPOTS
from cut_detector.utils.cell_division_detection.tools import (
    plot_predictions_evolution,
    pre_process_spots,
    get_tracks_to_merge,
)
from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.utils.trackmate_frame_spots import TrackMateFrameSpots
from cut_detector.utils.trackmate_track import TrackMateTrack


def perform_mitosis_track_generation(
    raw_video: np.ndarray,
    video_name: str,
    xml_model_dir: str,
    mitoses_save_dir: str,
    tracks_save_dir: str,
    metaphase_model_path: str,
    hmm_metaphase_parameters_file: str,
    predictions_file: Optional[str] = None,
    only_predictions_update: bool = False,
    plot_evolution: bool = False,
) -> Union[list[MitosisTrack], None]:
    # Create save_dir if not exists
    if not os.path.exists(mitoses_save_dir):
        os.makedirs(mitoses_save_dir)

    # Create video tracks save dir if not exists
    video_tracks_save_dir = os.path.join(tracks_save_dir, video_name)
    if not os.path.exists(video_tracks_save_dir):
        os.makedirs(video_tracks_save_dir)

    # Read useful information from xml file
    xml_model_path = os.path.join(xml_model_dir, f"{video_name}_model.xml")
    if not os.path.exists(xml_model_path):
        print("No xml file found for this video.")
        return None
    with open(xml_model_path) as fd:
        doc = xmltodict.parse(fd.read())

    # Define custom classes to read xml file
    trackmate_tracks = list(
        filter(
            lambda track: len(track.track_spots_ids) > 2
            and track.stop - track.start + 1 >= MIN_TRACK_SPOTS,
            [TrackMateTrack(track) for track in doc["TrackMate"]["Model"]["AllTracks"]["Track"]],
        )
    )
    raw_spots = [
        TrackMateFrameSpots(spots, raw_video.shape)
        for spots in doc["TrackMate"]["Model"]["AllSpots"]["SpotsInFrame"]
    ]

    print("\nPreprocess spots...")

    # Get dictionary of TrackMate spots (from xml file) for each track and detect metaphase spots
    pre_process_spots(
        trackmate_tracks,
        raw_spots,
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
    mitosis_tracks = get_tracks_to_merge(trackmate_tracks)

    # Plot predictions evolution
    if plot_evolution:
        plot_predictions_evolution(raw_spots, trackmate_tracks, mitosis_tracks)

    # Update useful attributes for each track
    for i, mitosis_track in enumerate(mitosis_tracks):
        mitosis_track.id = i
        mitosis_track.update_mitosis_start_end(trackmate_tracks, mitosis_tracks)
        mitosis_track.update_key_events_frame(trackmate_tracks)
        mitosis_track.update_mitosis_position_dln(trackmate_tracks)
        mitosis_track.update_is_near_border(raw_video)

        # Save mitosis track
        daughter_track_ids = ",".join([str(d) for d in mitosis_track.daughter_track_ids])
        state_path = (
            f"{video_name}_mitosis_{i}_{mitosis_track.mother_track_id}_to_{daughter_track_ids}.bin"
        )
        save_path = os.path.join(
            mitoses_save_dir,
            state_path,
        )
        with open(save_path, "wb") as f:
            pickle.dump(mitosis_track, f)

    # Save updated trackmate tracks
    for trackmate_track in trackmate_tracks:
        state_path = f"track_{trackmate_track.track_id}.bin"
        save_path = os.path.join(
            video_tracks_save_dir,
            state_path,
        )
        with open(save_path, "wb") as f:
            pickle.dump(trackmate_track, f)

    return mitosis_tracks
