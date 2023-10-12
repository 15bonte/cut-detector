import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.signal import find_peaks
from skimage.feature import graycomatrix, graycoprops

from ..utils.bridges_classification.impossible_detection import ImpossibleDetection
from ..utils.bridges_classification.template_type import TemplateType
from ..utils.hidden_markov_models import HiddenMarkovModel
from ..utils.image_tools import smart_cropping
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
        length_light_spot (int): Length of the video to check around the mt cut for the light spot
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
        length_light_spot=10,
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
    def _is_bridges_classification_impossible(mitosis_track: MitosisTrack) -> bool:
        """
        Bridges classification is impossible if:
        - no midbody spots
        - more than 2 daughter tracks
        - nucleus is near border
        - no midbody spot after cytokinesis
        """

        if mitosis_track.is_near_border:
            mitosis_track.key_events_frame["first_mt_cut"] = ImpossibleDetection.NEAR_BORDER
            mitosis_track.key_events_frame["second_mt_cut"] = ImpossibleDetection.NEAR_BORDER
            return True

        if len(mitosis_track.daughter_track_ids) >= 2:
            mitosis_track.key_events_frame[
                "first_mt_cut"
            ] = ImpossibleDetection.MORE_THAN_TWO_DAUGHTER_TRACKS
            mitosis_track.key_events_frame[
                "second_mt_cut"
            ] = ImpossibleDetection.MORE_THAN_TWO_DAUGHTER_TRACKS
            return True

        if not mitosis_track.mid_body_spots:
            mitosis_track.key_events_frame[
                "first_mt_cut"
            ] = ImpossibleDetection.NO_MID_BODY_DETECTED
            mitosis_track.key_events_frame[
                "second_mt_cut"
            ] = ImpossibleDetection.NO_MID_BODY_DETECTED
            return True

        if (
            max(mitosis_track.mid_body_spots.keys())
            < mitosis_track.key_events_frame["cytokinesis"]
        ):
            mitosis_track.key_events_frame[
                "first_mt_cut"
            ] = ImpossibleDetection.NO_MID_BODY_DETECTED_AFTER_CYTOKINESIS
            mitosis_track.key_events_frame[
                "second_mt_cut"
            ] = ImpossibleDetection.NO_MID_BODY_DETECTED_AFTER_CYTOKINESIS
            return True

        return False

    def _get_circle_data(
        self,
        radius: int,
        image: np.ndarray,
    ) -> tuple[list[tuple[int]], list[float], float, np.ndarray, np.ndarray]:
        """
        Compute useful data for a circle of a given radius around the mid body spot:

        - circle_positions: list of the coordinates of the pixels on the circle
        - intensities: list of the intensities of the pixels on the circle
        - mean_intensity: mean intensity of the pixels on the circle
        - peaks: list of the indexes of the peaks of the intensities on the circle
        - norm_peaks: list of the normalized indexes of the peaks of the intensities on the circle
        """
        # Get image shape
        size_x = image.shape[1]
        size_y = image.shape[0]

        # Get the mid-body spot coordinates, middle of the image
        mid_body_position = (size_x // 2, size_y // 2)

        # Get associated perimeter, angles and coordinates
        perimeter = round(2 * np.pi * radius)
        angles = np.linspace(0, 2 * np.pi, perimeter)
        circle_positions = []
        for angle in angles:
            pos_x = round(mid_body_position[1] + radius * np.cos(angle))
            pos_y = round(mid_body_position[0] + radius * np.sin(angle))
            if (pos_x, pos_y) not in circle_positions:
                circle_positions.append((pos_x, pos_y))

        intensities = []
        for position in circle_positions:
            # Get the mean intensity around the mid spot
            total, inc = 0, 0
            for k in range(-1, 2):
                for j in range(-1, 2):
                    pos_x = position[0] + k
                    pos_y = position[1] + j
                    if 0 <= pos_x < size_x and 0 <= pos_y < size_y:
                        total += image[pos_x, pos_y]
                        inc += 1
            mean = total / inc
            intensities.append(mean)

        # Detect the peaks of the intensities on the circle
        mean_intensity = np.mean(intensities)

        # Prevent the case where the peak is on the border so we detect 2 peaks instead of 1
        concatenated_intensities = intensities + intensities

        # Insert lower intensity at the beginning and at the end of the list
        concatenated_intensities.insert(0, concatenated_intensities[0] - 1)
        concatenated_intensities.append(concatenated_intensities[-1] - 1)

        peaks, _ = find_peaks(
            concatenated_intensities,
            height=mean_intensity * self.coeff_height_peak,
            distance=len(intensities) * self.circle_min_ratio,
            width=(None, round(len(intensities) * self.coeff_width_peak)),
        )

        peaks = [i - 1 for i in peaks]
        peaks = [p % len(circle_positions) for p in peaks]
        peaks = [p for p in peaks if peaks.count(p) >= 2]
        peaks = list(set(peaks))

        norm_peaks = [peak / len(circle_positions) for peak in peaks]

        return circle_positions, intensities, mean_intensity, peaks, norm_peaks

    def _get_peaks(
        self,
        all_positions: list[list[tuple[int]]],
        all_intensities: list[list[float]],
        all_peaks: list[np.ndarray],
        all_norm_peaks: list[np.ndarray],
    ) -> tuple[list[tuple[int]], list[float]]:
        """
        Post-processing of peaks detection.

        Returns kept peaks (circle index, peak position) and their intensities.
        A typical output would be:
            final_peaks = [(0, 30), (0, 62)]
            peaks_intensity = [440, 416]
        Peak at position 30 on circle 0 was kept, with an intensity of 440.
        Peak at position 62 on circle 0 was kept, with an intensity of 416.
        """
        # Create the windows
        windows_starts = np.linspace(
            0, 1 - self.window_size, num=round(self.overlap / self.window_size)
        )
        windows_ends = np.linspace(self.window_size, 1, num=round(self.overlap / self.window_size))
        windows = list(zip(windows_starts, windows_ends))

        # Group the peaks by windows
        group_peaks = (
            []
        )  # list (window) of list of tuple (index of the circle, position of the peak)
        for window in windows:
            group = []
            for i, (peaks, norm_peaks) in enumerate(zip(all_peaks, all_norm_peaks)):
                for peak, norm_peak in zip(peaks, norm_peaks):
                    if window[0] <= norm_peak < window[1]:
                        group.append((i, peak))
            group_peaks.append(group)

        # Remove groups that have less than min_peaks_by_group peaks
        new_group_peaks = []
        for group_peak in group_peaks:
            if len(group_peak) >= self.min_peaks_by_group and group_peak not in new_group_peaks:
                new_group_peaks.append(group_peak)

        # If there is more than 1 peak, make sure there is no duplicate
        if len(new_group_peaks) >= 2:
            # Sort by maximum number of peaks, or higher mean intensity
            def get_length_intensity_key(item):
                return (len(item), np.mean([all_intensities[p[0]][p[1]] for p in item]))

            new_group_peaks.sort(key=get_length_intensity_key, reverse=True)

            # One peak must appear in one group only
            # Remove the peaks that are already seen in previous groups
            for i in range(len(new_group_peaks) - 1):
                for j in range(i + 1, len(new_group_peaks)):
                    new_group_peaks[j] = [
                        p for p in new_group_peaks[j] if p not in new_group_peaks[i]
                    ]

            # Remove groups that have less than min_peaks_by_group peaks
            new_group_peaks = [g for g in new_group_peaks if len(g) >= self.min_peaks_by_group]

            # Compute the mean position of the peaks in each group
            position_groups = [
                sum(pos / len(all_positions[circle]) for circle, pos in group) / len(group)
                for group in new_group_peaks
            ]

            # Remove the groups that are too close to each other, i.e. distance < circle_min_ratio
            i = 0
            while i < len(position_groups) - 1:
                j = i + 1
                while j < len(position_groups):
                    first_position = (position_groups[i] - position_groups[j]) % 1
                    second_position = (position_groups[j] - position_groups[i]) % 1
                    if min(first_position, second_position) < self.circle_min_ratio:
                        if len(new_group_peaks[i]) > len(new_group_peaks[j]):
                            new_group_peaks.remove(new_group_peaks[j])
                            position_groups.remove(position_groups[j])
                            j -= 1
                        else:
                            new_group_peaks.remove(new_group_peaks[i])
                            position_groups.remove(position_groups[i])
                            i -= 1
                    j += 1
                i += 1

            # Sort again by maximum number of peaks and mean intensity of the peaks in the group
            new_group_peaks.sort(key=get_length_intensity_key, reverse=True)

            # Should not be possible to have more than 2 MT, then keep 2 best ones
            if len(new_group_peaks) > 2:
                new_group_peaks = new_group_peaks[:2]

        # Compute peaks_intensity with the mean intensity of the peaks
        peaks_intensity = [
            sum(all_intensities[circle][pos] for circle, pos in group) / len(group)
            for group in new_group_peaks
        ]

        # final_peaks for plot
        final_peaks = [p[0] for p in new_group_peaks]

        return final_peaks, peaks_intensity

    @staticmethod
    def _get_haralick_features(
        list_intensity: list[float],
        properties=None,  # avoid list as default value
    ) -> np.ndarray:
        """
        Compute a few Haralick features from a list of intensities.
        """
        if properties is None:
            properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

        # Convert list_intensity to 2D array and int
        list_intensity_2D = np.array(list_intensity).astype(int).reshape(1, -1)

        # Compute all haralick features
        haralick_features = graycomatrix(
            list_intensity_2D,
            distances=[1],
            angles=[0],
            levels=max(list_intensity_2D[0]) + 1,
            symmetric=True,
            normed=True,
        )
        return np.array(
            [graycoprops(haralick_features, property)[0][0] for property in properties]
        )

    def _plot_circles(
        self,
        filename,
        image,
        final_peaks: list[tuple[int]],
        all_positions: list[list[tuple[int]]],
        mean_intensity: list[float],
        peaks_intensity: list[float],
        list_intensity: list[float],
    ) -> None:
        """
        Plot the image with the circle and the peaks.
        """

        # Plot the intensities of the circle around the mid body spot
        plt.subplot(2, 1, 1)
        plt.title("Number of peaks: " + str(len(final_peaks)))

        plt.plot(range(len(list_intensity)), list_intensity)
        plt.axhline(y=mean_intensity[0] * self.coeff_height_peak, color="red")
        plt.plot(
            [p[1] for p in final_peaks], peaks_intensity, marker="o", markersize=4, color="red"
        )

        # plot the image without the bridge line
        plt.subplot(2, 2, 3)
        plt.imshow(image)

        # plot the image with the bridge line
        plt.subplot(2, 2, 4)
        plt.imshow(image)

        if len(final_peaks) > 0:
            coord_peaks = [all_positions[p[0]][p[1]] for p in final_peaks]
            if len(final_peaks) == 2:
                plt.plot(
                    [coord_peaks[0][1], coord_peaks[1][1]],
                    [coord_peaks[0][0], coord_peaks[1][0]],
                    color="red",
                )
            elif len(final_peaks) == 1:
                plt.plot(
                    coord_peaks[0][1], coord_peaks[0][0], marker="o", markersize=4, color="red"
                )
        for i in range(len(all_positions)):
            x_cercle = [all_positions[i][k][0] for k in range(len(all_positions[i]))]
            y_cercle = [all_positions[i][k][1] for k in range(len(all_positions[i]))]
            plt.plot(y_cercle, x_cercle, color="green")

        plt.title(filename)
        plt.show()

    def get_bridge_template(
        self,
        image: np.ndarray,
        plot_enabled: bool,
        filename=None,
    ) -> np.ndarray:
        """
        Return bridge image embedding according to template provided in parameters.
        """

        # Create the coordinates list of the circle around the mid body spot
        # NB: first circle in list is the middle circle
        circle_radius, diff_radius = self.circle_radius, self.diff_radius
        list_radius = [
            circle_radius,
            circle_radius + 0.25 * diff_radius,
            circle_radius - 0.25 * diff_radius,
            circle_radius + 0.5 * diff_radius,
            circle_radius - 0.5 * diff_radius,
            circle_radius + 0.75 * diff_radius,
            circle_radius - 0.75 * diff_radius,
            circle_radius + diff_radius,
            circle_radius - diff_radius,
        ]

        # Get useful data for each circle
        all_positions, all_intensities, all_mean_intensity = [], [], []
        all_peaks, all_norm_peaks = [], []
        for radius in list_radius:
            positions, intensities, mean_intensity, peaks, norm_peaks = self._get_circle_data(
                radius, image
            )
            all_positions.append(positions)
            all_intensities.append(intensities)
            all_mean_intensity.append(mean_intensity)
            all_peaks.append(peaks)
            all_norm_peaks.append(norm_peaks)

        # Get peaks data
        final_peaks, peaks_intensity = self._get_peaks(
            all_positions, all_intensities, all_peaks, all_norm_peaks
        )

        # Get useful Haralick features from middle center circle
        haralick_features = self._get_haralick_features(all_intensities[0])

        # Plot if enabled
        if plot_enabled:
            self._plot_circles(
                filename,
                image,
                final_peaks,
                all_positions,
                all_mean_intensity,
                peaks_intensity,
                all_intensities[0],
            )

        # Get the template of the bridge
        if self.template_type == TemplateType.ALL_WITHOUT_HARALICK:
            return np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                    np.var(all_intensities[0]),  # 2 : variance of intensity on the circle
                    np.mean(all_intensities[0]),  # 3 : mean intensity on the circle
                ]
            )

        if self.template_type == TemplateType.NB_PEAKS:
            return np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                ]
            )

        if self.template_type == TemplateType.PEAKS_AND_INTENSITY:
            return np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                ]
            )

        if self.template_type == TemplateType.HARALICK:
            return haralick_features

        if self.template_type == TemplateType.ALL:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                    np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features))

        # Ablation study on All
        if self.template_type == TemplateType.ALL_ABLATION_1:
            template = np.array(
                [
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                    np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features))

        if self.template_type == TemplateType.All_Ablation2:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(all_intensities[0]),  # 1 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features))

        if self.template_type == TemplateType.All_Ablation3:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                ]
            )
            return np.concatenate((template, haralick_features))

        if self.template_type == TemplateType.All_Ablation4:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                    np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features[1:]))

        if self.template_type == TemplateType.All_Ablation5:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                    np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features[:1], haralick_features[2:]))

        if self.template_type == TemplateType.All_Ablation6:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                    np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features[:2], haralick_features[3:]))

        if self.template_type == TemplateType.All_Ablation7:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                    np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features[:3], haralick_features[4:]))

        if self.template_type == TemplateType.All_Ablation8:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(peaks_intensity)
                    if len(final_peaks) > 0
                    else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                    np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features[:4], haralick_features[5:]))

        raise ValueError("Unknown template type")

    def get_bridge_class(
        self,
        img_bridge: np.ndarray,
        scaler: StandardScaler,
        clf: SVC,
        plot_enabled=True,
    ) -> int:
        # Make sure crop has expected size
        assert img_bridge.shape[0] == self.margin * 2 and img_bridge.shape[1] == self.margin * 2

        # Get template and reshape to 2D data
        template = self.get_bridge_template(img_bridge, plot_enabled=plot_enabled).reshape(1, -1)

        # Scale template
        scaled_template = scaler.transform(template)

        # Predict the class
        frame_class = clf.predict(scaled_template)
        if frame_class[0] == "A":
            return 0
        if frame_class[0] == "B":
            return 1
        return 2

    @staticmethod
    def apply_hmm(hmm_parameters, sequence):
        """
        Correct the sequence of classes using HMM.
        """
        # Define observation sequence
        obs_seq = np.asarray(sequence, dtype=np.int32)

        # Define HMM model & run inference
        model = HiddenMarkovModel(hmm_parameters["A"], hmm_parameters["B"], hmm_parameters["pi"])
        states_seq, _ = model.viterbi_inference(obs_seq)

        return states_seq

    def update_mt_cut_detection(
        self,
        mitosis_track: MitosisTrack,
        video: np.ndarray,
        scaler_path: str,
        model_path: str,
        hmm_bridges_parameters_file: str,
    ) -> None:
        """
        Update micro-tubules cut detection using bridges classification.
        """
        classification_impossible = self._is_bridges_classification_impossible(mitosis_track)

        if classification_impossible:
            return

        # Perform classification...
        ordered_mb_frames = sorted(mitosis_track.mid_body_spots.keys())
        first_mb_frame = ordered_mb_frames[0]
        last_mb_frame = ordered_mb_frames[-1]
        first_frame = max(first_mb_frame, mitosis_track.key_events_frame["cytokinesis"] - 2)  # -2?

        # Load the classifier and scaler
        with open(model_path, "rb") as f:
            classifier: SVC = pickle.load(f)
        # Load the scaler
        with open(scaler_path, "rb") as f:
            scaler: StandardScaler = pickle.load(f)

        list_class_bridges = []
        # Iterate over frames and get the class of the bridge
        for frame in range(first_frame, last_mb_frame + 1):
            min_x = mitosis_track.position.min_x
            min_y = mitosis_track.position.min_y

            # Get midbody coordinates
            mb_coords = mitosis_track.mid_body_spots[frame].position
            x_pos, y_pos = min_x + mb_coords[0], min_y + mb_coords[1]

            # Extract frame image and crop around the midbody Sir-tubulin
            frame_image = video[frame, :, :, :].squeeze().transpose(2, 0, 1)  # C, H, W
            crop = smart_cropping(frame_image, self.margin, x_pos, y_pos, pad=True)[0, ...]  # H, W

            # Get the class of the bridge
            bridge_class = self.get_bridge_class(crop, scaler, classifier, plot_enabled=False)
            list_class_bridges.append(bridge_class)

        # Make sure cytokinesis bridge is detected as A (no MT cut)
        relative_cytokinesis_frame = mitosis_track.key_events_frame["cytokinesis"] - first_frame
        if relative_cytokinesis_frame < 0 or list_class_bridges[relative_cytokinesis_frame] != 0:
            mitosis_track.key_events_frame[
                "first_mt_cut"
            ] = ImpossibleDetection.MT_CUT_AT_CYTOKINESIS
            mitosis_track.key_events_frame[
                "second_mt_cut"
            ] = ImpossibleDetection.MT_CUT_AT_CYTOKINESIS
            return

        # Read HMM parameters
        if not os.path.exists(hmm_bridges_parameters_file):
            raise FileNotFoundError(f"File {hmm_bridges_parameters_file} not found")
        hmm_parameters = np.load(hmm_bridges_parameters_file)

        # Correct the sequence with HMM
        seq_after_hmm = self.apply_hmm(hmm_parameters, list_class_bridges)

        # Get index of first element > 0 in sequence (first MT cut)
        first_mt_cut_frame_rel = next((i for i, x in enumerate(seq_after_hmm) if x > 0), -1)

        # Ignore if no MT cut detected
        if first_mt_cut_frame_rel == -1:
            mitosis_track.key_events_frame["first_mt_cut"] = ImpossibleDetection.NO_CUT_DETECTED
            mitosis_track.key_events_frame["second_mt_cut"] = ImpossibleDetection.NO_CUT_DETECTED
            return

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
            mitosis_track.key_events_frame["first_mt_cut"] = ImpossibleDetection.LIGHT_SPOT
            mitosis_track.key_events_frame["second_mt_cut"] = ImpossibleDetection.LIGHT_SPOT
            return

        # Update mitosis track accordingly
        mitosis_track.key_events_frame["first_mt_cut"] = first_mt_cut_frame_abs

        # Get index of first element > 1 in sequence (second MT cut)
        second_mt_cut_frame_rel = next((i for i, x in enumerate(seq_after_hmm) if x > 1), -1)

        # get the frame of the second MT cut
        if second_mt_cut_frame_rel == -1:
            mitosis_track.key_events_frame["second_mt_cut"] = ImpossibleDetection.NO_CUT_DETECTED
            return

        second_mt_cut_frame_abs = first_frame + second_mt_cut_frame_rel

        # Update mitosis track accordingly
        mitosis_track.key_events_frame["second_mt_cut"] = second_mt_cut_frame_abs
