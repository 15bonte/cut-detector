import os
import numpy as np
from tqdm import tqdm

from ..utils.parameters import Parameters
from ..utils.mt_cut_detection.bridges_mt_model_manager import (
    BridgesMtModelManager,
)
from ..utils.mt_cut_detection.bridges_mt_cnn_model_params import (
    BridgesMtCnnModelParams,
)
from ..utils.tools import apply_hmm, perform_cnn_inference
from ..utils.mt_cut_detection.impossible_detection import (
    ImpossibleDetection,
)
from ..utils.mitosis_track import MitosisTrack


class MtCutDetectionFactory:
    """
    Class to detect first and second MT cut.

    Parameters
    ----------
    margin : int
        Number of pixels on each side of the midbody.
    """

    def __init__(
        self,
        params=Parameters(),
        margin=50,
    ) -> None:
        self.params = params
        self.margin = margin

    def _is_bridges_classification_impossible(
        self, mitosis_track: MitosisTrack, video: np.ndarray
    ) -> bool:
        """
        Bridges classification is impossible if:
        - no midbody spots
        - more than 2 daughter tracks
        - nucleus is near border
        - no midbody spot after cytokinesis
        - metaphase detected after cytokinesis

        Parameters
        ----------
        mitosis_track : MitosisTrack
            Mitosis track to check.
        video : np.ndarray
            Video of the mitosis track. TYXC.

        Returns
        -------
        bool
            True if classification is impossible, False otherwise.
        """

        if len(mitosis_track.daughter_track_ids) >= 2:
            mitosis_track.key_events_frame["first_mt_cut"] = (
                ImpossibleDetection.MORE_THAN_TWO_DAUGHTER_TRACKS
            )
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.MORE_THAN_TWO_DAUGHTER_TRACKS
            )
            return True

        if mitosis_track.is_near_border(
            video,
            self.params.cytokinesis_duration,
            self.params.spatial_resolution,
        ):
            mitosis_track.key_events_frame["first_mt_cut"] = (
                ImpossibleDetection.NEAR_BORDER
            )
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.NEAR_BORDER
            )
            return True

        if not mitosis_track.mid_body_spots:
            mitosis_track.key_events_frame["first_mt_cut"] = (
                ImpossibleDetection.NO_MID_BODY_DETECTED
            )
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.NO_MID_BODY_DETECTED
            )
            return True

        if (
            max(mitosis_track.mid_body_spots.keys())
            < mitosis_track.key_events_frame["no_mt_cut"]
        ):
            mitosis_track.key_events_frame["first_mt_cut"] = (
                ImpossibleDetection.NO_MID_BODY_DETECTED
            )
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.NO_MID_BODY_DETECTED
            )
            return True

        return False

    def update_mt_cut_detection(
        self,
        mitosis_tracks: list[MitosisTrack],
        video: np.ndarray,
        hmm_bridges_parameters_file: str,
        bridges_mt_cnn_model_path: str,
        debug_mode=False,
    ) -> dict[str, dict]:
        """
        Update micro-tubules cut detection using bridges classification.

        Parameters
        ----------
        mitosis_tracks : list[MitosisTrack]
            List of mitosis tracks to update.
        video : np.ndarray
            Video of the mitosis tracks. TYXC.
        hmm_bridges_parameters_file : str
            Path to the HMM parameters file.
        bridges_mt_cnn_model_path : str
            Path to the bridges CNN model.
        debug_mode : bool
            If True, return the classified bridges.

        Returns
        -------
        dict[str, dict]
            Classified bridges.
        """
        # Run CNN classification
        classified_bridges = self._classify_bridges(
            mitosis_tracks, video, bridges_mt_cnn_model_path
        )

        print("\nPredictions performed successfully.")

        # Read HMM parameters
        if not os.path.exists(hmm_bridges_parameters_file):
            raise FileNotFoundError(
                f"File {hmm_bridges_parameters_file} not found"
            )
        hmm_parameters = np.load(hmm_bridges_parameters_file)

        # Check if classification is impossible and smooth
        print("Correcting prediction inconsistencies.")
        for mitosis_track in tqdm(mitosis_tracks):
            classification_impossible = (
                self._is_bridges_classification_impossible(
                    mitosis_track, video
                )
            )

            if classification_impossible:
                continue

            # Force first frame to be class 0
            classified_bridges["predictions"][mitosis_track.id][0] = 0

            # Correct the sequence with HMM
            classified_bridges["predictions_after_hmm"][mitosis_track.id] = (
                apply_hmm(
                    hmm_parameters,
                    classified_bridges["predictions"][mitosis_track.id],
                )
            )

            # Get index of first element > 0 in sequence (first MT cut)
            first_mt_cut_frame_rel = next(
                (
                    i
                    for i, x in enumerate(
                        classified_bridges["predictions_after_hmm"][
                            mitosis_track.id
                        ]
                    )
                    if x > 0
                ),
                -1,
            )

            # Ignore if no MT cut detected
            if first_mt_cut_frame_rel == -1:
                mitosis_track.key_events_frame["first_mt_cut"] = (
                    ImpossibleDetection.NO_CUT_DETECTED
                )
                mitosis_track.key_events_frame["second_mt_cut"] = (
                    ImpossibleDetection.NO_CUT_DETECTED
                )
                continue

            # Ignore if cut is too short, i.e. less than 50 minutes
            first_frame = min(classified_bridges["frames"][mitosis_track.id])
            first_mt_cut_frame_abs = first_frame + first_mt_cut_frame_rel
            if (
                first_mt_cut_frame_abs
                - mitosis_track.key_events_frame["no_mt_cut"]
            ) <= 50 / self.params.time_resolution:  # 50 minutes
                mitosis_track.key_events_frame["first_mt_cut"] = (
                    ImpossibleDetection.TOO_SHORT_CUT
                )
                mitosis_track.key_events_frame["second_mt_cut"] = (
                    ImpossibleDetection.TOO_SHORT_CUT
                )
                continue

            # Update mitosis track accordingly
            mitosis_track.key_events_frame["first_mt_cut"] = (
                first_mt_cut_frame_abs
            )

            # Get index of first element > 1 in sequence (second MT cut)
            second_mt_cut_frame_rel = next(
                (
                    i
                    for i, x in enumerate(
                        classified_bridges["predictions_after_hmm"][
                            mitosis_track.id
                        ]
                    )
                    if x > 1
                ),
                -1,
            )

            # get the frame of the second MT cut
            if second_mt_cut_frame_rel == -1:
                mitosis_track.key_events_frame["second_mt_cut"] = (
                    ImpossibleDetection.NO_CUT_DETECTED
                )
                continue

            second_mt_cut_frame_abs = first_frame + second_mt_cut_frame_rel

            # Update mitosis track accordingly
            mitosis_track.key_events_frame["second_mt_cut"] = (
                second_mt_cut_frame_abs
            )

            mitosis_track.apply_consistency_checks()

            if debug_mode:
                # In debug, expect only one mitosis track
                return classified_bridges

        return classified_bridges

    def _classify_bridges(
        self,
        mitosis_tracks: list[MitosisTrack],
        video: np.ndarray,
        bridges_mt_cnn_model_path: str,
    ) -> dict[str, dict]:
        """
        Classify bridges using a CNN model.

        Parameters
        ----------
        mitosis_tracks : list[MitosisTrack]
            List of mitosis tracks to classify.
        video : np.ndarray
            Video of the mitosis tracks. TYXC.
        bridges_mt_cnn_model_path : str
            Path to the bridges CNN model.

        Returns
        -------
        dict[str, dict]
            Classified bridges.
        """
        # Get bridge crops
        bridges = {
            "images": {},
            "frames": {},
            "predictions": {},
            "predictions_after_hmm": {},
        }
        for mitosis_track in mitosis_tracks:
            images, frames = mitosis_track.get_bridge_images(
                video, self.margin
            )
            sir_tubulin_images = [
                np.expand_dims(image[0], axis=0) for image in images
            ]
            bridges["images"][mitosis_track.id] = sir_tubulin_images
            bridges["frames"][mitosis_track.id] = frames

        # Perform classification
        flatten_images = [
            image for images in bridges["images"].values() for image in images
        ]
        predictions = perform_cnn_inference(
            model_path=bridges_mt_cnn_model_path,
            images=flatten_images,
            cnn_model_params=BridgesMtCnnModelParams,
            model_manager=BridgesMtModelManager,
        )

        # Fill predictions
        for mitosis_track_id, images in bridges["images"].items():
            bridges["predictions"][mitosis_track_id] = predictions[
                : len(images)
            ]
            predictions = predictions[len(images) :]

        return bridges
