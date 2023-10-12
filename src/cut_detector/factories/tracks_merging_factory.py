import json
import os
import numpy as np
import xmltodict
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

from ..utils.mitosis_track import MitosisTrack
from ..utils.trackmate_spot import TrackMateSpot
from ..utils.trackmate_track import TrackMateTrack
from ..utils.trackmate_frame_spots import TrackMateFrameSpots
from ..utils.hidden_markov_models import HiddenMarkovModel
from ..utils.cell_division_detection.metaphase_cnn_data_set import (
    MetaphaseCnnDataSet,
)
from ..utils.cell_division_detection.metaphase_cnn import MetaphaseCnn
from ..utils.cell_division_detection.metaphase_cnn_model_params import MetaphaseCnnModelParams


def get_track_from_id(tracks: list[TrackMateTrack], track_id: int) -> TrackMateTrack:
    for track in tracks:
        if track.track_id == track_id:
            return track
    raise ValueError(f"Track {track_id} not found")


class TracksMergingFactory:
    """
    Class to merge TrackMate tracks into mitosis tracks.

    Args:
        min_track_spots (int): Minimum spots in track to consider it.
        minimum_metaphase_interval (int): Minimum distance between two metaphases.
        max_spot_distance_for_split (int): Maximum distance between two spots to consider them.
    """

    def __init__(
        self, min_track_spots=10, minimum_metaphase_interval=10, max_spot_distance_for_split=20
    ) -> None:
        self.min_track_spots = min_track_spots
        self.minimum_metaphase_interval = minimum_metaphase_interval
        self.max_spot_distance_for_split = max_spot_distance_for_split

    def read_trackmate_xml(self, xml_model_path: str, raw_video_shape: np.ndarray) -> None:
        """
        Read useful information from xml file.
        """
        if not os.path.exists(xml_model_path):
            print("No xml file found for this video.")
            return None
        with open(xml_model_path) as fd:
            doc = xmltodict.parse(fd.read())

        # Define custom classes to read xml file
        trackmate_tracks = list(
            filter(
                lambda track: len(track.track_spots_ids) > 2
                and track.stop - track.start + 1 >= self.min_track_spots,
                [
                    TrackMateTrack(track)
                    for track in doc["TrackMate"]["Model"]["AllTracks"]["Track"]
                ],
            )
        )
        raw_spots = [
            TrackMateFrameSpots(spots, raw_video_shape)
            for spots in doc["TrackMate"]["Model"]["AllSpots"]["SpotsInFrame"]
        ]

        return trackmate_tracks, raw_spots

    def get_tracks_to_merge(self, raw_tracks: list[TrackMateTrack]) -> list[MitosisTrack]:
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
                if track_first_frame in raw_track.track_spots
                and raw_track.track_id != track.track_id
            ]

            # Keep only stuck spots
            first_spot = track.track_spots[track_first_frame]
            stuck_spots: list[TrackMateSpot] = list(
                filter(
                    lambda x: x.is_stuck_to(first_spot, self.max_spot_distance_for_split),
                    contemporary_spots,
                )
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
                key=lambda x: get_track_from_id(raw_tracks, x.track_id).compute_metaphase_iou(
                    track
                ),
            )[-1]

            # Mother cell spot is spot in metaphase for corresponding track
            mother_cell_spot = selected_spot.corresponding_metaphase_spot

            # Check if it should be merged to existing split (division into 3 cells)
            triple_division = False
            for mitosis_track in mitosis_tracks:
                if mitosis_track.is_same_mitosis(
                    mother_cell_spot.track_id, mother_cell_spot.frame
                ):
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

    def correct_sequence(self, orig_sequence: list[int]) -> list[int]:
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
                and metaphase_index[idx] - metaphase_index[idx - 1]
                < self.minimum_metaphase_interval
            ):
                corrected_sequence[list(range(metaphase_index[idx - 1], metaphase_index[idx]))] = 1
        return corrected_sequence

    def pre_process_spots(
        self,
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
        """

        nuclei_crops = []
        # Get list of possible metaphase spots
        for track in trackmate_tracks:
            # Get current track spots data & images
            current_nuclei_crops = track.get_spots_data(raw_spots, raw_video)
            # Merge current_nuclei_crops with nuclei_crops
            nuclei_crops = nuclei_crops + current_nuclei_crops

        # Apply CNN model to get metaphase spots, once for all
        predictions = self._predict_metaphase_spots(metaphase_model_path, nuclei_crops)

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
                track_predictions = self.correct_sequence(track_predictions)

            # Save prediction for each spot
            track.update_metaphase_spots(track_predictions)

        self.update_predictions_file(trackmate_tracks, predictions_file, video_name)

    @staticmethod
    def _predict_metaphase_spots(
        metaphase_model_path: str, nuclei_crops: list[np.array]
    ) -> list[int]:
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

    @staticmethod
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
