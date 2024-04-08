import os
import numpy as np

from ..utils.bridges_classification.bridges_mt_model_manager import (
    BridgesMtModelManager,
)
from ..utils.bridges_classification.bridges_mt_cnn_model_params import (
    BridgesMtCnnModelParams,
)
from ..utils.tools import apply_hmm, perform_cnn_inference
from ..utils.bridges_classification.impossible_detection import (
    ImpossibleDetection,
)
from ..utils.bridges_classification.template_type import TemplateType
from ..utils.mitosis_track import MitosisTrack


class MtCutDetectionFactory:
    """
    Class to detect first and second MT cut.

    Args:
        margin (int): Number of pixels on each side of the midbody.

        # Bridges classification parameters

        coeff_height_peak (float): Coeff for the height of the peak.
        circle_radius (int): Middle circle radius.
        circle_min_ratio (float): Min ratio distance on circle between two peaks.
        coeff_width_peak (float): Coeff for the width of the peak.
        diff_radius (float): Distance between the min/max and middle circle.
        window_size (float): Window size for the peak detection (% of the circle perimeter).
        min_peaks_by_group (int): Minimum number of peaks by window for the peak detection.
        overlap (int): Overlap between two windows for the peak detection.
        template_type (TemplateType): Template for SVC model.

        # Light spot detection parameters

        intensity_threshold_light_spot (int): Intensity threshold for the light spot detection.
        h_maxima_light_spot (int): h for the h_maxima function (light spot detection).
        center_tolerance_light_spot (int): Center tolerance to not count the light spots that are
        too close to the center.
        min_percentage_light_spot (float): Minimum percentage of frames with light spots to
        consider the mitosis as a light spot mitosis.
        crop_size_light_spot (int): Size of the crop for the light spot detection.
        length_light_spot (int): Length of the video to check around the mt cut for light spot
        detection.
    """

    def __init__(
        self,
        margin=50,
        coeff_height_peak=1.071,
        circle_radius=11,
        circle_min_ratio=0.2025,
        coeff_width_peak=0.25358974,
        diff_radius=4.54,
        window_size=0.04836,
        min_peaks_by_group=4,
        overlap=4,
        intensity_threshold_light_spot=350,
        h_maxima_light_spot=130,
        center_tolerance_light_spot=5,
        min_percentage_light_spot=0.1,
        crop_size_light_spot=20,
        length_light_spot=3,
        template_type=TemplateType.ALL,
    ) -> None:
        self.margin = margin
        self.coeff_height_peak = coeff_height_peak
        self.circle_radius = circle_radius
        self.circle_min_ratio = circle_min_ratio
        self.coeff_width_peak = coeff_width_peak
        self.diff_radius = diff_radius
        self.window_size = window_size
        self.min_peaks_by_group = min_peaks_by_group
        self.overlap = overlap
        self.intensity_threshold_light_spot = intensity_threshold_light_spot
        self.h_maxima_light_spot = h_maxima_light_spot
        self.center_tolerance_light_spot = center_tolerance_light_spot
        self.min_percentage_light_spot = min_percentage_light_spot
        self.crop_size_light_spot = crop_size_light_spot
        self.length_light_spot = length_light_spot
        self.template_type = template_type

    @staticmethod
    def _is_bridges_classification_impossible(
        mitosis_track: MitosisTrack,
    ) -> bool:
        """
        Bridges classification is impossible if:
        - no midbody spots
        - more than 2 daughter tracks
        - nucleus is near border
        - no midbody spot after cytokinesis
        """

        if mitosis_track.is_near_border:
            mitosis_track.key_events_frame["first_mt_cut"] = (
                ImpossibleDetection.NEAR_BORDER
            )
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.NEAR_BORDER
            )
            return True

        if len(mitosis_track.daughter_track_ids) >= 2:
            mitosis_track.key_events_frame["first_mt_cut"] = (
                ImpossibleDetection.MORE_THAN_TWO_DAUGHTER_TRACKS
            )
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.MORE_THAN_TWO_DAUGHTER_TRACKS
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
            < mitosis_track.key_events_frame["cytokinesis"]
        ):
            mitosis_track.key_events_frame["first_mt_cut"] = (
                ImpossibleDetection.NO_MID_BODY_DETECTED_AFTER_CYTOKINESIS
            )
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.NO_MID_BODY_DETECTED_AFTER_CYTOKINESIS
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
    ):
        """
        Update micro-tubules cut detection using bridges classification.
        Expect TYXC video.
        """
        # Run CNN classification
        list_class_bridges = self.classify_bridges(
            mitosis_tracks, video, bridges_mt_cnn_model_path
        )

        # Debug use
        results = {
            "list_class_bridges": list_class_bridges,
            "list_class_bridges_after_hmm": {},
            "crops": {},
        }

        # Read HMM parameters
        if not os.path.exists(hmm_bridges_parameters_file):
            raise FileNotFoundError(
                f"File {hmm_bridges_parameters_file} not found"
            )
        hmm_parameters = np.load(hmm_bridges_parameters_file)

        # Check if classification is impossible and smooth
        for mitosis_track in mitosis_tracks:
            classification_impossible = (
                self._is_bridges_classification_impossible(mitosis_track)
            )

            if classification_impossible:
                continue

            # Perform classification...
            ordered_mb_frames = sorted(mitosis_track.mid_body_spots.keys())
            first_mb_frame = ordered_mb_frames[0]
            first_frame = max(
                first_mb_frame,
                mitosis_track.key_events_frame["cytokinesis"] - 2,
            )  # -2 because cytokinesis frame may be a bit too late

            if debug_mode:  # save crops only in debug
                results["crops"][mitosis_track.id] = (
                    mitosis_track.get_bridge_images(video, self.margin)
                )

            # Make sure cytokinesis bridge is detected as class 0: "no MT cut"
            relative_cytokinesis_frame = (
                mitosis_track.key_events_frame["cytokinesis"] - first_frame
            )
            if (
                relative_cytokinesis_frame < 0
                or results["list_class_bridges"][mitosis_track.id][
                    relative_cytokinesis_frame
                ]
                != 0
            ):
                mitosis_track.key_events_frame["first_mt_cut"] = (
                    ImpossibleDetection.MT_CUT_AT_CYTOKINESIS
                )
                mitosis_track.key_events_frame["second_mt_cut"] = (
                    ImpossibleDetection.MT_CUT_AT_CYTOKINESIS
                )
                continue

            # Correct the sequence with HMM
            results["list_class_bridges_after_hmm"][mitosis_track.id] = (
                apply_hmm(
                    hmm_parameters,
                    results["list_class_bridges"][mitosis_track.id],
                )
            )

            # Get index of first element > 0 in sequence (first MT cut)
            first_mt_cut_frame_rel = next(
                (
                    i
                    for i, x in enumerate(
                        results["list_class_bridges_after_hmm"][
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

            first_mt_cut_frame_abs = first_frame + first_mt_cut_frame_rel
            if mitosis_track.light_spot_detected(
                video,
                first_mt_cut_frame_abs,
                self.length_light_spot,
                self.crop_size_light_spot,
                self.h_maxima_light_spot,
                self.intensity_threshold_light_spot,
                self.center_tolerance_light_spot,
                self.min_percentage_light_spot,
            ):
                mitosis_track.key_events_frame["first_mt_cut"] = (
                    ImpossibleDetection.LIGHT_SPOT
                )
                mitosis_track.key_events_frame["second_mt_cut"] = (
                    ImpossibleDetection.LIGHT_SPOT
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
                        results["list_class_bridges_after_hmm"][
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

            if debug_mode:
                # In debug, expect only one mitosis track
                return results

        return results

    def classify_bridges(
        self,
        mitosis_tracks: list[MitosisTrack],
        video: np.ndarray,
        bridges_mt_cnn_model_path: str,
    ) -> dict[int, list[int]]:
        """
        Classify bridges using a CNN model.
        """
        # Get bridge crops
        crops = {}
        for mitosis_track in mitosis_tracks:
            raw_crops = mitosis_track.get_bridge_images(video, self.margin)
            first_channel_crops = [
                np.expand_dims(raw_crop[0], axis=0) for raw_crop in raw_crops
            ]
            crops[mitosis_track.id] = first_channel_crops

        # Perform classification
        crops_list = [
            crop for crop_list in crops.values() for crop in crop_list
        ]
        predictions = perform_cnn_inference(
            model_path=bridges_mt_cnn_model_path,
            images=crops_list,
            cnn_model_params=BridgesMtCnnModelParams,
            model_manager=BridgesMtModelManager,
        )

        # Create prediction dictionary
        mitosis_predictions = {}
        for mitosis_track_id, crop_list in crops.items():
            mitosis_predictions[mitosis_track_id] = predictions[
                : len(crop_list)
            ]
            predictions = predictions[len(crop_list) :]

        return mitosis_predictions
