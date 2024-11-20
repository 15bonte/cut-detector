import json
import os
import numpy as np

from ..utils.parameters import Parameters
from ..utils.metaphase_sequence import MetaphaseSequence
from ..utils.cell_track import CellTrack
from ..utils.tools import perform_cnn_inference
from ..utils.mitosis_track import MitosisTrack
from ..utils.cell_spot import CellSpot
from ..utils.hidden_markov_models import HiddenMarkovModel
from ..utils.mitosis_track_generation.metaphase_cnn_model_params import (
    MetaphaseCnnModelParams,
)
from ..utils.mitosis_track_generation.metaphase_cnn_data_set import (
    MetaphaseCnnDataSet,
)
from ..utils.mitosis_track_generation.metaphase_cnn_model import (
    MetaphaseCnnModel,
)


def get_track_from_id(tracks: list[CellTrack], track_id: int) -> CellTrack:
    """Used to get track from its id.

    Parameters
    ----------
    tracks : list[CellTrack]
        List of all tracks.
    track_id : int
        Track id.

    Returns
    -------
    CellTrack
        Track corresponding to track_id.
    """
    for track in tracks:
        if track.track_id == track_id:
            return track
    raise ValueError(f"Track {track_id} not found")


class MitosisTrackGenerationFactory:
    """Class to merge cell tracks into mitosis tracks.

    Parameters
    ----------
    minimum_metaphase_interval : int
        Minimum time interval distance between two metaphases (minutes).
    max_spot_distance_for_split : int
        Maximum distance between two spots to consider them stuck (um).
    """

    def __init__(
        self,
        params,
        minimum_metaphase_interval=100,
        max_spot_distance_for_split=4.5,
    ) -> None:
        self.params = params
        self.minimum_metaphase_interval = int(
            minimum_metaphase_interval / params.time_resolution
        )
        self.max_spot_distance_for_split = int(
            max_spot_distance_for_split / params.spatial_resolution * 1000
        )

    def get_tracks_to_merge(
        self, raw_tracks: list[CellTrack]
    ) -> list[MitosisTrack]:
        """Plug tracks occurring at frame>0 to closest metaphase.

        Parameters
        ----------
        raw_tracks : list[CellTrack]
            List of all tracks.

        Returns
        -------
        list[MitosisTrack]
            List of mitosis tracks.
        """
        ordered_tracks = sorted(raw_tracks, key=lambda x: x.start)
        mitosis_tracks: list[MitosisTrack] = []

        # Loop through all tracks beginning at frame > 0 and try to plug them to the previous metaphase
        for track in reversed(ordered_tracks):
            # Break when reaching tracking starting at first frame, as they are ordered
            track_first_frame = min(track.spots.keys())
            if track_first_frame == 0:
                break

            # Get all spots at same frame
            contemporary_spots = [
                raw_track.spots[track_first_frame]
                for raw_track in raw_tracks
                if track_first_frame in raw_track.spots
                and raw_track.track_id != track.track_id
            ]

            # Keep only stuck spots
            first_spot = track.spots[track_first_frame]
            stuck_spots: list[CellSpot] = list(
                filter(
                    lambda x: x.is_stuck_to(
                        first_spot, self.max_spot_distance_for_split
                    ),
                    contemporary_spots,
                )
            )

            # Keep only spots with metaphase sequence close to track first frame
            metaphase_spots = list(
                filter(
                    lambda x: get_track_from_id(
                        raw_tracks, x.track_id
                    ).has_close_metaphase(
                        x,
                        track_first_frame,
                        self.params.frames_around_metaphase,
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
                key=lambda x: get_track_from_id(
                    raw_tracks, x.track_id
                ).compute_metaphase_iou(track),
            )[-1]

            metaphase_sequence: MetaphaseSequence = (
                selected_spot.corresponding_metaphase_sequence
            )

            # Check if it should be merged to existing split (division into 3 cells)
            triple_division = False
            for mitosis_track in mitosis_tracks:
                if metaphase_sequence.is_same(
                    mitosis_track.metaphase_sequence
                ):
                    mitosis_track.add_daughter_track(track.track_id)
                    triple_division = True
                    break

            # If not, create new split
            if not triple_division:
                mitosis_tracks.append(
                    MitosisTrack(
                        track.track_id,
                        metaphase_sequence,
                    )
                )

        # Return dictionaries of tracks to merge
        return mitosis_tracks

    def _correct_sequence(self, orig_sequence: list[int]) -> list[int]:
        """Correct sequence of states to fill the gap between two metaphase (1) subsequences
        separated by less than minimum_interval frames.

        Parameters
        ----------
        orig_seq : list[int]
            Predicted classes.

        Returns
        -------
        list[int]
            Corrected classes.
        """
        corrected_sequence = np.copy(orig_sequence)
        # Get indexes of 1
        metaphase_index = [
            i for i, x in enumerate(corrected_sequence) if x == 1
        ]
        for idx in range(1, len(metaphase_index)):
            if (
                metaphase_index[idx] - metaphase_index[idx - 1] != 1
                and metaphase_index[idx] - metaphase_index[idx - 1]
                < self.minimum_metaphase_interval
            ):
                corrected_sequence[
                    list(range(metaphase_index[idx - 1], metaphase_index[idx]))
                ] = 1
        return corrected_sequence

    def pre_process_spots(
        self,
        cell_tracks: list[CellTrack],
        raw_spots: list[CellSpot],
        raw_video: np.ndarray,
        metaphase_model_path: str,
        hmm_metaphase_parameters_file: str,
        predictions_file: str,
        video_name: str,
        only_predictions_update: bool,
    ) -> None:
        """Sort spots in track and predict metaphase.

        Parameters
        ----------
        cell_tracks : list[CellTrack]
            List of cell tracks.
        raw_spots : list[CellSpot]
            List of all spots.
        raw_video : np.ndarray
            Video. TYXC.
        metaphase_model_path : str
            CNN model path.
        hmm_metaphase_parameters_file : str
            HMM parameters file.
        predictions_file : str
            Predictions file to update if provided.
        video_name : str
            Video name.
        only_predictions_update : bool
            If True, only update predictions, do not apply HMM.
        """

        nuclei_crops, numbers_crops = [], []
        # Get list of possible metaphase spots
        for track in cell_tracks:
            # Get current track spots data & images
            current_nuclei_crops = track.get_spots_data(raw_spots, raw_video)
            # Merge current_nuclei_crops with nuclei_crops
            nuclei_crops = nuclei_crops + current_nuclei_crops  # CYX
            numbers_crops.append(len(current_nuclei_crops))

        # Apply CNN model to get metaphase spots, once for all
        predictions = self._predict_metaphase_spots(
            metaphase_model_path, nuclei_crops
        )

        # Load HMM parameters and create model
        if not only_predictions_update:
            if not os.path.exists(hmm_metaphase_parameters_file):
                raise FileNotFoundError(
                    f"File {hmm_metaphase_parameters_file} not found"
                )
            hmm_parameters = np.load(hmm_metaphase_parameters_file)
            hmm_model = HiddenMarkovModel(
                hmm_parameters["A"], hmm_parameters["B"], hmm_parameters["pi"]
            )

        # Get list of possible metaphase spots
        for track, number_crops in zip(cell_tracks, numbers_crops):
            track_predictions = predictions[:number_crops]
            predictions = predictions[number_crops:]

            # Hidden Markov Model to smooth predictions
            # If we just want to get raw CNN predictions, we don't want to correct the predictions
            if not only_predictions_update:
                track_predictions, _ = hmm_model.viterbi_inference(
                    track_predictions
                )
                track_predictions = self._correct_sequence(track_predictions)

            # Save prediction for each spot
            track.update_metaphase_spots(
                track_predictions,
                self.params.interphase_index,
                self.params.metaphase_index,
            )

        self._update_predictions_file(
            cell_tracks, predictions_file, video_name
        )

    @staticmethod
    def _predict_metaphase_spots(
        metaphase_model_path: str, nuclei_crops: list[np.array]
    ) -> list[int]:
        """Run CNN model to predict metaphase spots.

        Parameters
        ----------
        metaphase_model : str
            CNN model path.
        nuclei_crops :  list[np.array]
            list[CYX]

        Returns
        -------
        list[int]
            Predicted classes.
        """

        predictions = perform_cnn_inference(
            model_path=metaphase_model_path,
            images=nuclei_crops,
            cnn_model_params=MetaphaseCnnModelParams,
            cnn_data_set=MetaphaseCnnDataSet,
            cnn_classifier=MetaphaseCnnModel,
        )
        return predictions

    @staticmethod
    def _update_predictions_file(
        tracks: list[CellTrack], predictions_file: str, video_name: str
    ) -> None:
        """Update predictions file with new predictions.
        Used only to train HMM model.

        Parameters
        ----------
        tracks: [CellTrack]
            List of cell tracks.
        predictions_file: str
            Predictions file to update.
        video_name: str
            Video name.
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
            int(track.track_id): [
                int(spot.predicted_phase) for spot in track.spots.values()
            ]
            for track in tracks
        }
        predictions_data[video_name] = predictions
        # Save predictions data
        with open(predictions_file, "w") as json_file:
            json.dump(predictions_data, json_file)
