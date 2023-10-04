import os
import json
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_framework.utils.model_managers.cnn_model_manager import CnnModelManager
from cnn_framework.utils.data_managers.default_data_manager import DefaultDataManager
from cnn_framework.utils.metrics.classification_accuracy import ClassificationAccuracy
from cnn_framework.utils.preprocessing import normalize_array
from cnn_framework.utils.data_loader_generators.data_loader_generator import (
    collate_dataset_output,
)
from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.enum import PredictMode

from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.utils.trackmate_frame_spots import TrackMateFrameSpots
from cut_detector.utils.trackmate_spot import TrackMateSpot
from cut_detector.utils.trackmate_track import TrackMateTrack
from cut_detector.utils.hidden_markov_models import HiddenMarkovModel
from cut_detector.constants.tracking import MINIMUM_METAPHASE_INTERVAL

from cut_detector.utils.cell_division_detection.metaphase_cnn_data_set import (
    MetaphaseCnnDataSet,
)
from cut_detector.utils.cell_division_detection.metaphase_cnn import MetaphaseCnn
from cut_detector.utils.cell_division_detection.metaphase_cnn_model_params import (
    MetaphaseCnnModelParams,
)


def plot_predictions_evolution(
    raw_spots: list[TrackMateFrameSpots],
    raw_tracks: list[TrackMateTrack],
    mitosis_tracks: list[MitosisTrack],
) -> None:
    # Spots detected by Cellpose
    detected_spots = {raw_spot.frame: len(raw_spot.spots) for raw_spot in raw_spots}

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
        detected_spots[i] if i in detected_spots else 0 for i in range(min_frame, max_frame)
    ]
    data_tracks = [detected_tracks[0]] + [
        detected_tracks[i] if i in detected_tracks else 0 for i in range(min_frame, max_frame)
    ]
    data_mitoses = [detected_mitoses[0]] + [
        detected_mitoses[i] if i in detected_mitoses else 0 for i in range(min_frame, max_frame)
    ]

    _, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.step(list(range(min_frame, max_frame + 1)), data_spots, "r", linewidth=8.0)
    ax.step(list(range(min_frame, max_frame + 1)), data_tracks, "g", linewidth=8.0)
    ax.step(list(range(min_frame, max_frame + 1)), data_mitoses, "b", linewidth=8.0)

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
        ax.axvline(ending_track_frame, color="k", linewidth=2.0, linestyle="--")

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


def predict_metaphase_spots(metaphase_model_path: str, nuclei_crops: list[np.array]) -> list[int]:
    """
    Run CNN model to predict metaphase spots

    Parameters
    ----------
    metaphase_model: CNN model path
    nuclei_crops: [C, H, W]

    Returns
    -------
    predictions: [class predicted]
    """

    # Custom class to avoid loading images from folder
    class MetaphaseCnnDatasetFiles(MetaphaseCnnDataSet):
        def __init__(self, data_list, *args, **kwargs):
            self.data_list = (
                data_list  # not pythonic, but needed as super init calls generate_raw_images
            )
            super().__init__(*args, **kwargs)

        def generate_raw_images(self, filename):
            idx = int(filename.split(".")[0])
            nucleus_image = normalize_array(self.data_list[idx], None)  # C, H, W
            nucleus_image = np.moveaxis(nucleus_image, 0, -1)  # H, W, C
            return DatasetOutput(input=nucleus_image, target_array=np.asarray([0, 1, 0]))

    # Metaphase model parameters
    model_parameters = MetaphaseCnnModelParams()
    # Modify parameters for training
    model_parameters.train_ratio = 0
    model_parameters.val_ratio = 0
    model_parameters.test_ratio = 1

    # Model definition
    # Load pretrained model
    model = MetaphaseCnn(nb_classes=model_parameters.nb_classes)

    map_location = None
    if not torch.cuda.is_available():
        map_location = torch.device("cpu")
        print("No GPU found, using CPU.")
    model.load_state_dict(
        torch.load(
            metaphase_model_path,
            map_location=map_location,
        )
    )

    # Test (no sampler to keep order)
    dataset_test = MetaphaseCnnDatasetFiles(
        nuclei_crops,
        False,
        [f"{idx}.ext" for idx in range(len(nuclei_crops))],
        DefaultDataManager(""),
        model_parameters,
    )
    test_dl = DataLoader(
        dataset_test,
        batch_size=128,
        collate_fn=collate_dataset_output,
    )

    manager = CnnModelManager(model, model_parameters, ClassificationAccuracy)

    predictions = manager.predict(
        test_dl, predict_mode=PredictMode.GetPrediction, nb_images_to_save=0
    )  # careful, this is scores and not probabilities
    predictions = [int(np.argmax(p)) for p in predictions]

    return predictions


def correct_sequence(orig_sequence: list[int]) -> list[int]:
    """
    Correct sequence of states to fill the gap between two metaphase (1) subsequences
    separated by less than minimum_interval frames.

    Parameters
    ----------
    orig_seq: [class predicted]

    Returns
    -------
    seq: [class corrected]

    """
    corrected_sequence = np.copy(orig_sequence)
    # Get indexes of 1
    metaphase_index = [i for i, x in enumerate(corrected_sequence) if x == 1]
    for idx in range(1, len(metaphase_index)):
        if (
            metaphase_index[idx] - metaphase_index[idx - 1] != 1
            and metaphase_index[idx] - metaphase_index[idx - 1] < MINIMUM_METAPHASE_INTERVAL
        ):
            corrected_sequence[
                [i for i in range(metaphase_index[idx - 1], metaphase_index[idx])]
            ] = 1
    return corrected_sequence


def update_predictions_file(
    tracks: list[TrackMateTrack], predictions_file: str, video_name: str
) -> None:
    """
    Parameters
    ----------
    tracks: [TrackMateTrack]
    predictions_file: str
    video_name: str
    """
    if predictions_file is None:
        return

    # Read data from json prediction if exists
    if os.path.exists(predictions_file):
        with open(predictions_file) as json_file:
            predictions_data = json.load(json_file)
    else:
        predictions_data = {}

    # Retrieve predictions
    predictions = {
        int(track.track_id): [int(spot.predicted_phase) for spot in track.track_spots.values()]
        for track in tracks
    }
    predictions_data[video_name] = predictions
    # Save predictions data
    with open(predictions_file, "w") as json_file:
        json.dump(predictions_data, json_file)


def pre_process_spots(
    trackmate_tracks: list[TrackMateTrack],
    raw_spots: list[TrackMateFrameSpots],
    raw_video: np.array,
    metaphase_model_path: str,
    hmm_metaphase_parameters_file: str,
    predictions_file: str,
    video_name: str,
    only_predictions_update: bool,
) -> None:
    """
    Sort spots in track and predict metaphase.

    Parameters
    ----------
    raw_tracks: [TrackMateTrack]
    raw_spots: [TrackMateFrameSpots]
    raw_video: T, H, W, C
    metaphase_model_path: CNN model path
    hmm_metaphase_parameters_file: file with saved hmm parameters
    """

    nuclei_crops = []
    # Get list of possible metaphase spots
    for track in trackmate_tracks:
        # Get current track spots data & images
        current_nuclei_crops = track.get_spots_data(raw_spots, raw_video)
        # Merge current_nuclei_crops with nuclei_crops
        nuclei_crops = nuclei_crops + current_nuclei_crops

    # Apply CNN model to get metaphase spots, once for all
    predictions = predict_metaphase_spots(metaphase_model_path, nuclei_crops)

    # Load HMM parameters and create model
    if not only_predictions_update:
        if not os.path.exists(hmm_metaphase_parameters_file):
            raise FileNotFoundError(f"File {hmm_metaphase_parameters_file} not found")
        hmm_parameters = np.load(hmm_metaphase_parameters_file)
        hmm_model = HiddenMarkovModel(
            hmm_parameters["A"], hmm_parameters["B"], hmm_parameters["pi"]
        )

    # Get list of possible metaphase spots
    for track in trackmate_tracks:
        track_predictions = predictions[: track.number_spots]
        predictions = predictions[track.number_spots :]

        # Hidden Markov Model to smooth predictions
        # If we just want to get raw CNN predictions, we don't want to correct the predictions
        if not only_predictions_update:
            track_predictions, _ = hmm_model.viterbi_inference(track_predictions)
            track_predictions = correct_sequence(track_predictions)

        # Save prediction for each spot
        track.update_metaphase_spots(track_predictions)

    update_predictions_file(trackmate_tracks, predictions_file, video_name)


def get_tracks_to_merge(raw_tracks: list[TrackMateTrack]) -> list[MitosisTrack]:
    """
    Plug tracks occurring at frame>0 to closest metaphase.
    """
    ordered_tracks = sorted(raw_tracks, key=lambda x: x.track_start)
    mitosis_tracks: list[MitosisTrack] = []

    # Loop through all tracks beginning at frame > 0 and try to plug them to the previous metaphase
    for track in reversed(ordered_tracks):
        # Break when reaching tracking starting at first frame, as they are ordered
        track_first_frame = min(track.track_spots.keys())
        if track_first_frame == 0:
            break

        # Get all spots at same frame
        contemporary_spots = [
            raw_track.track_spots[track_first_frame]
            for raw_track in raw_tracks
            if track_first_frame in raw_track.track_spots and raw_track.track_id != track.track_id
        ]

        # Keep only stuck spots
        first_spot = track.track_spots[track_first_frame]
        stuck_spots: list[TrackMateSpot] = list(
            filter(lambda x: x.is_stuck_to(first_spot), contemporary_spots)
        )

        # Keep only spots with metaphase frame close to track first frame
        metaphase_spots = list(
            filter(
                lambda x: get_track_from_id(raw_tracks, x.track_id).has_close_metaphase(
                    x, track_first_frame
                ),
                stuck_spots,
            )
        )

        # If no candidate has been found, ignore track
        if len(metaphase_spots) == 0:
            continue

        # Order remaining spots by overlap
        selected_spot = sorted(
            metaphase_spots,
            key=lambda x: get_track_from_id(raw_tracks, x.track_id).compute_metaphase_iou(track),
        )[-1]

        # Mother cell spot is spot in metaphase for corresponding track
        mother_cell_spot = selected_spot.corresponding_metaphase_spot

        # Check if it should be merged to existing split (division into 3 cells)
        triple_division = False
        for mitosis_track in mitosis_tracks:
            if mitosis_track.is_same_mitosis(mother_cell_spot.track_id, mother_cell_spot.frame):
                mitosis_track.add_daughter_track(track.track_id)
                triple_division = True
                break

        # If not, create new split
        if not triple_division:
            mitosis_tracks.append(
                MitosisTrack(mother_cell_spot.track_id, track.track_id, mother_cell_spot.frame)
            )

    # Return dictionaries of tracks to merge
    return mitosis_tracks


def get_track_from_id(tracks: list[TrackMateTrack], track_id: int) -> TrackMateTrack:
    for track in tracks:
        if track.track_id == track_id:
            return track
    raise ValueError(f"Track {track_id} not found")
