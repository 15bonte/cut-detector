import os
from typing import Optional, Union
import pickle
import numpy as np
from tqdm import tqdm

from ..utils.parameters import Parameters
from ..utils.cell_track import CellTrack
from ..utils.cell_spot import CellSpot
from ..factories.mitosis_track_generation_factory import (
    MitosisTrackGenerationFactory,
)
from ..models.tools import get_model_path
from ..utils.mitosis_track import MitosisTrack


def perform_mitosis_track_generation(
    video: np.ndarray,
    video_name: str,
    spots_dir: str,
    tracks_dir: str,
    mitoses_dir: Optional[str] = None,
    metaphase_model_path: Optional[str] = get_model_path("metaphase_cnn"),
    hmm_metaphase_parameters_file: Optional[str] = os.path.join(
        get_model_path("hmm"), "hmm_metaphase_parameters.npz"
    ),
    predictions_file: Optional[str] = None,
    only_predictions_update: bool = False,
    save: bool = True,
    params=Parameters(),
) -> tuple[Union[list[MitosisTrack], None], list[CellSpot], list[CellTrack]]:
    """Perform mitosis track generation.

    Parameters
    ----------
    video : np.ndarray
        Video. TYXC.
    video_name : str
        Video name.
    spots_dir : str
        Directory where spots are saved.
    tracks_dir : str
        Directory where tracks are saved.
    mitoses_dir : Optional[str], optional
        Directory where mitoses are saved, by default None.
    metaphase_model_path : Optional[str], optional
        Metaphase model path, by default get_model_path("metaphase_cnn").
    hmm_metaphase_parameters_file : Optional[str], optional
        HMM metaphase parameters file, by default os.path.join(get_model_path("hmm"), "hmm_metaphase_parameters.npz").
    predictions_file : Optional[str], optional
        Predictions file, by default None.
    only_predictions_update : bool, optional
        Only update predictions, by default False.
    save : bool, optional
        Save, by default True.
    params: Parameters, optional
        Video parameters, by default Parameters().

    Returns
    -------
    Union[list[MitosisTrack], None]
        List of mitosis tracks if only_predictions_update is False, None otherwise.
    list[CellSpot]
        Cell spots.
    list[CellTrack]
        Cell tracks.
    """

    print("\n### CELL DIVISION DETECTION ###")

    # Create save_dir if not exists
    if mitoses_dir is not None and not os.path.exists(mitoses_dir):
        os.makedirs(mitoses_dir)

    # Create factory instance, where useful functions are defined
    tracks_merging_factory = MitosisTrackGenerationFactory(params)

    # Load cell spots
    cell_spots: list[CellSpot] = []
    video_spots_save_dir = os.path.join(spots_dir, video_name)
    for state_path in os.listdir(video_spots_save_dir):
        with open(os.path.join(video_spots_save_dir, state_path), "rb") as f:
            cell_spot = CellSpot.load(f)
            cell_spots.append(cell_spot)

    # Load cell tracks
    cell_tracks: list[CellTrack] = []
    video_tracks_save_dir = os.path.join(tracks_dir, video_name)
    for state_path in os.listdir(video_tracks_save_dir):
        with open(os.path.join(video_tracks_save_dir, state_path), "rb") as f:
            cell_track = CellTrack.load(f)
            cell_tracks.append(cell_track)

    # Detect metaphase spots
    tracks_merging_factory.pre_process_spots(
        cell_tracks,
        cell_spots,
        video,
        metaphase_model_path,
        hmm_metaphase_parameters_file,
        predictions_file,
        video_name,
        only_predictions_update,
    )

    # If the goal is only to update predictions file, stop here
    if only_predictions_update:
        return None

    # Plug tracks occurring at frame>0 to closest metaphase
    mitosis_tracks = tracks_merging_factory.get_tracks_to_merge(cell_tracks)

    print(
        f"\nPredictions performed successfully. {len(mitosis_tracks)} divisions detected."
    )

    # Update useful attributes for each track
    print("Updating division attributes.")
    for i, mitosis_track in enumerate(tqdm(mitosis_tracks)):
        mitosis_track.id = i
        mitosis_track.update_mitosis_start_end(
            cell_tracks,
            mitosis_tracks,
            tracks_merging_factory.params.frames_around_metaphase,
        )
        mitosis_track.update_key_events_frame(cell_tracks)
        mitosis_track.update_mitosis_position_contour(cell_tracks)

        # Save mitosis track
        if mitoses_dir is not None and save:
            state_path = f"{mitosis_track.get_file_name(video_name)}.bin"
            save_path = os.path.join(
                mitoses_dir,
                state_path,
            )
            with open(save_path, "wb") as f:
                pickle.dump(mitosis_track, f)

    # Save updated cell tracks
    if save:
        for cell_track in cell_tracks:
            state_path = f"track_{cell_track.track_id}.bin"
            save_path = os.path.join(
                video_tracks_save_dir,
                state_path,
            )
            with open(save_path, "wb") as f:
                pickle.dump(cell_track, f)

    return mitosis_tracks, cell_spots, cell_tracks
