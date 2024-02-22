import os
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.signal import find_peaks

from skimage.feature import graycomatrix, graycoprops

from cnn_framework.utils.tools import save_tiff

from ..utils.peak import Peak
from ..utils.bridges_classification.impossible_detection import (
    ImpossibleDetection,
)
from ..utils.bridges_classification.template_type import TemplateType
from ..utils.hidden_markov_models import HiddenMarkovModel
from ..utils.image_tools import smart_cropping
from ..utils.mitosis_track import MitosisTrack
from ..utils.micro_tubules_augmentation import MicroTubulesAugmentation


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

    def _detect_peaks_from_intensities(
        self,
        intensities: list[float],
        circle_index: int,
        coeff_height_peak: float,
        coeff_width_peak: float,
        debug_plot: bool,
        circle_positions: Optional[list[tuple[int]]] = None,
    ) -> list[Peak]:
        """
        Given a list of intensities, detect peaks using scipy.

        Parameters
        ----------
        circle_positions : (y, x)
        """

        # Concatenate intensities to detect peaks even at borders
        concatenated_intensities = np.concatenate(
            (
                np.array(intensities),
                np.array(intensities),
                np.array(intensities),
            ),
            axis=0,
        )

        minimum_height = np.mean(intensities) * coeff_height_peak
        peaks_idx, peaks_data = find_peaks(
            concatenated_intensities,
            height=(
                minimum_height,
                None,
            ),  # absolute value
            distance=len(intensities) * self.circle_min_ratio,
            width=(None, round(len(intensities) * coeff_width_peak)),
            prominence=(None, None),
        )

        # Create peak instances
        # Regarding peak_idx, remove intensities length to focus on the first intensities list
        peaks = [
            Peak(
                relative_position=(peak_idx - len(intensities))
                / len(intensities),
                intensity=peaks_data["peak_heights"][idx],
                coordinates=(
                    circle_positions[peak_idx - len(intensities)]
                    if circle_positions is not None
                    else (0, 0)
                ),
                relative_intensity=peaks_data["peak_heights"][idx]
                / np.mean(intensities),
                position_index=peak_idx - len(intensities),
                circle_index=circle_index,
                prominence=peaks_data["prominences"][idx],
                width=peaks_data["widths"][idx],
                relative_width=peaks_data["widths"][idx] / len(intensities),
            )
            for idx, peak_idx in enumerate(peaks_idx)
            if len(intensities) <= peak_idx < 2 * len(intensities)
        ]

        if debug_plot:
            results_half = (
                peaks_data["widths"],
                peaks_data["width_heights"],
                peaks_data["left_ips"],
                peaks_data["right_ips"],
            )  # ie taken at half prominence
            plt.figure()
            plt.plot(concatenated_intensities)
            plt.plot(
                peaks_idx, np.array(concatenated_intensities)[peaks_idx], "x"
            )
            plt.hlines(*results_half[1:], color="C2")
            plt.hlines(
                minimum_height,
                xmin=0,
                xmax=len(concatenated_intensities),
                color="red",
            )
            contour_heights = (
                np.array(concatenated_intensities)[peaks_idx]
                - peaks_data["prominences"]
            )
            plt.vlines(
                x=peaks_idx,
                ymin=contour_heights,
                ymax=concatenated_intensities[peaks_idx],
                color="green",
            )
            plt.xlim(
                len(intensities), len(intensities) * 2
            )  # ignore rest of concatenated intensities
            plt.show()

        return peaks

    def _get_circle_data(
        self,
        radius: int,
        image: np.ndarray,
        circle_index: int,
        debug_plot: bool,
    ) -> tuple[list[tuple[int]], list[float], list[Peak]]:
        """
        Compute useful data for a circle of a given radius around the mid body spot:

        Parameters
        -------
        image : YX

        Returns
        -------

        circle_positions : list of the coordinates of the pixels on the circle
        intensities : list of the intensities of the pixels on the circle
        peaks : list of Peaks

        """
        # Get image shape
        size_y, size_x = image.shape[0], image.shape[1]

        # Get the mid-body spot coordinates, middle of the image
        mid_body_position = (size_x // 2, size_y // 2)

        # Get associated perimeter, angles and coordinates
        perimeter = round(2 * np.pi * radius)
        angles = np.linspace(0, 2 * np.pi, perimeter)
        circle_positions = []
        for angle in angles:
            pos_y = round(mid_body_position[0] + radius * np.cos(angle))
            pos_x = round(mid_body_position[1] + radius * np.sin(angle))
            if (pos_y, pos_x) not in circle_positions:
                circle_positions.append((pos_y, pos_x))

        intensities = []
        for position in circle_positions:
            # Get the mean intensity around the mid spot
            total, inc = 0, 0
            for k in range(-1, 2):
                for j in range(-1, 2):
                    pos_y = position[0] + k
                    pos_x = position[1] + j
                    if 0 <= pos_y < size_y and 0 <= pos_x < size_x:
                        total += image[pos_y, pos_x]
                        inc += 1
            mean = total / inc
            intensities.append(mean)

        peaks = self._detect_peaks_from_intensities(
            intensities,
            circle_index,
            self.coeff_height_peak,
            self.coeff_width_peak,
            debug_plot,
            circle_positions,
        )

        return circle_positions, intensities, peaks

    def _get_windows(self) -> list[tuple[float]]:
        """
        Return the windows for the peak detection.
        """
        windows_starts = np.linspace(
            0, 1 - self.window_size, num=round(self.overlap / self.window_size)
        )
        windows_ends = np.linspace(
            self.window_size, 1, num=round(self.overlap / self.window_size)
        )
        windows = list(zip(windows_starts, windows_ends))
        return windows

    def _post_process_peaks(
        self,
        all_peaks: list[list[Peak]],
    ) -> list[Peak]:
        """
        Post-processing of peaks detection.
        Returns kept peaks and their intensities.
        """
        # Group peaks by window and remove small groups
        window_peaks = Peak.group_peaks(
            all_peaks,
            self._get_windows(),
            minimum_size=self.min_peaks_by_group,
        )

        # If there is more than 1 peak, make sure there is no duplicate
        if len(window_peaks) >= 2:
            # Sort by maximum number of peaks, then higher mean intensity
            def get_length_intensity_key(item):
                return (
                    Peak.get_average_intensity(item),
                    len(item),
                )

            window_peaks.sort(key=get_length_intensity_key, reverse=True)

            unique_window_peaks = Peak.remove_duplicated_peaks(
                window_peaks, self.min_peaks_by_group
            )
            cleaned_window_peaks = Peak.remove_close_peaks(
                unique_window_peaks, self.circle_min_ratio
            )

            # Sort once again, after all cleanings
            cleaned_window_peaks.sort(
                key=get_length_intensity_key, reverse=True
            )

            # Should not be possible to have more than 2 MT, then keep 2 best ones
            if len(cleaned_window_peaks) > 2:
                cleaned_window_peaks = cleaned_window_peaks[:2]

        else:
            cleaned_window_peaks = window_peaks

        # Create average peaks
        average_peaks = [
            Peak.create_average_peak(peak) for peak in cleaned_window_peaks
        ]

        return average_peaks

    @staticmethod
    def _get_haralick_features(
        list_intensity: list[float],
        properties=None,  # avoid list as default value
    ) -> np.ndarray:
        """
        Compute a few Haralick features from a list of intensities.
        """
        if properties is None:
            properties = [
                "contrast",
                "dissimilarity",
                "homogeneity",
                "energy",
                "correlation",
            ]

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
            [
                graycoprops(haralick_features, property)[0][0]
                for property in properties
            ]
        )

    def _plot_circles(
        self,
        filename,
        sir_tubulin_image,
        filtered_image,
        final_peaks: list[Peak],
        all_positions: list[list[tuple[int]]],
        peaks_intensity: list[float],
        list_intensities: list[list[float]],
        average_circle_peaks: list[Peak],
    ) -> None:
        """
        Plot the image with the circle and the peaks.

        Parameters
        ----------

        sir_tubulin_image: YX
        filtered_image: YX
        """

        # Plot the intensities of the circle around the mid body spot
        plt.subplot(2, 1, 1)
        plt.title("Number of peaks: " + str(len(final_peaks)))

        # Colors - create a color list of 9 colors using usual matplotlib colors
        colors = [
            "blue",
            "orange",
            "green",
            "yellow",
            "purple",
            "brown",
            "pink",
            "grey",
            "olive",
        ]

        # Plot intensities along circles
        expected_points_nb = len(list_intensities[0])
        for circle_idx, list_intensity in enumerate(list_intensities):
            interpolated_x = np.linspace(
                0, len(list_intensity), expected_points_nb
            )
            interpolated_list_intensity = np.interp(
                interpolated_x,  # x
                range(len(list_intensity)),  # xp
                list_intensity,  # fp
            )
            plt.plot(
                range(expected_points_nb),
                interpolated_list_intensity,
                color=colors[circle_idx // 2],
            )
            plt.axhline(
                y=np.mean(interpolated_list_intensity)
                * self.coeff_height_peak,
                xmin=0,
                xmax=expected_points_nb,
                color=colors[circle_idx // 2],
            )

        # Plot peaks
        for idx, peak in enumerate(final_peaks):
            plt.plot(
                peak.relative_position * expected_points_nb,
                peaks_intensity[idx],
                marker="o",
                markersize=4,
                color="red",
            )
            # plt.text(
            #     peak.relative_position * expected_points_nb,
            #     peaks_intensity[idx],
            #     f"Prominence: {int(peak.prominence)}",
            # )
        for avg_peak in average_circle_peaks:
            if avg_peak.is_empty():
                continue
            plt.plot(
                avg_peak.relative_position * expected_points_nb,
                avg_peak.intensity,
                marker="*",
                markersize=4,
                color="green",
            )

        # plot initial image
        plt.subplot(2, 3, 4)
        plt.imshow(sir_tubulin_image)

        # plot filtered image
        plt.subplot(2, 3, 5)
        plt.imshow(filtered_image)

        # plot the image with the bridge line
        plt.subplot(2, 3, 6)
        plt.imshow(filtered_image)

        for peak in final_peaks:
            plt.scatter(
                peak.coordinates[1],
                peak.coordinates[0],
                marker="o",
                color="red",
            )

        if len(final_peaks) == 2:
            plt.plot(
                [final_peaks[0].coordinates[1], final_peaks[1].coordinates[1]],
                [final_peaks[0].coordinates[0], final_peaks[1].coordinates[0]],
                color="red",
            )

        for i in range(len(all_positions)):
            x_circle = [
                all_positions[i][k][0] for k in range(len(all_positions[i]))
            ]
            y_circle = [
                all_positions[i][k][1] for k in range(len(all_positions[i]))
            ]
            plt.plot(y_circle, x_circle, color="green")

        plt.title(filename)
        plt.show()

    def _get_average_circle_peaks(
        self, all_intensities: list[float]
    ) -> list[Peak]:
        """
        Compute 2 best peaks on average circle.
        """
        # First, compute average circle intensities
        expected_points_nb = len(all_intensities[0])
        average_circle_intensities = []
        for intensities in all_intensities:
            interpolated_points = np.linspace(
                0, len(intensities), expected_points_nb
            )
            interpolated_intensities = np.interp(
                interpolated_points,  # x
                range(len(intensities)),  # xp
                intensities,  # fp
            )
            average_circle_intensities.append(interpolated_intensities)
        average_circle_intensities = np.mean(
            average_circle_intensities, axis=0
        )

        # Then, detect peaks on average circle
        average_circle_peaks = self._detect_peaks_from_intensities(
            average_circle_intensities,
            circle_index=0,
            coeff_height_peak=0,  # minimum
            coeff_width_peak=len(average_circle_intensities),  # maximum
            debug_plot=False,
        )

        def get_peak_prominence(peak: Peak):
            return peak.prominence

        # Keep only 2 best peaks, i.e. with highest prominence
        average_circle_peaks.sort(key=get_peak_prominence, reverse=True)
        average_circle_peaks = average_circle_peaks[:2]

        # # Order them by relative position to assure consistency
        # NB: this was used to follow peaks - sigmoid fit test
        # def get_peak_relative_position(peak: Peak):
        #     return peak.relative_position
        # average_circle_peaks.sort(key=get_peak_relative_position)

        # Complete with empty Peak if length <2
        while len(average_circle_peaks) < 2:
            average_circle_peaks.append(Peak())

        return average_circle_peaks

    def _pre_process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Smooth image.
        1. Apply a mean filter to smooth the image
        2. Apply an adaptive histogram equalization to improve contrast
        """
        return image  # no pre-processing to apply so far
        # from skimage.exposure import equalize_adapthist
        # import scipy.ndimage as ndi
        # filtered_image = ndi.correlate(image, np.full((3, 3), 1 / 9))
        # filtered_image = equalize_adapthist(
        #     filtered_image, kernel_size=self.circle_radius * 2
        # )
        # return filtered_image

    def get_bridge_template(
        self,
        image: np.ndarray,
        debug_plot: bool,
        file_path=None,
        class_mode=None,
        save_dir=None,
    ) -> np.ndarray:
        """
        Return bridge image embedding according to template provided in parameters.

        image : CYX
        class_mode: int, used for debug
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

        # Apply image pre-processing
        sir_tubulin_image = image[0, ...].squeeze()  # YX
        filtered_image = self._pre_process_image(sir_tubulin_image)  # YX

        # Get useful data for each circle
        all_positions, all_intensities, all_peaks = [], [], []
        for circle_index, radius in enumerate(list_radius):
            (
                positions,
                intensities,
                peaks,
            ) = self._get_circle_data(
                radius, filtered_image, circle_index, debug_plot=False
            )
            all_positions.append(positions)
            all_intensities.append(intensities)
            all_peaks.append(peaks)

        # Alternative method: compute 2 best peaks on average circle
        average_circle_peaks = self._get_average_circle_peaks(all_intensities)

        # Get peaks data
        final_peaks = self._post_process_peaks(all_peaks)
        peaks_intensity = [peak.intensity for peak in final_peaks]

        # Get useful Haralick features from middle center circle
        haralick_features = self._get_haralick_features(all_intensities[0])

        # Save fake one cut image
        if save_dir is not None:
            file_name = os.path.basename(file_path).split("_c")[0]
            if class_mode == 0:
                micro_tubules_augmentation = MicroTubulesAugmentation(
                    average_circle_peaks
                )
                augmentations = (
                    micro_tubules_augmentation.generate_augmentations(image)
                )
                for title, augmented_image in augmentations.items():
                    save_path = os.path.join(
                        save_dir, f"{file_name}_{title}_c1.tiff"
                    )
                    save_tiff(augmented_image, save_path, original_order="YXC")
            if class_mode in (2, 4):
                micro_tubules_augmentation = MicroTubulesAugmentation()
                augmentations = (
                    micro_tubules_augmentation.generate_augmentations(image)
                )
                for title, augmented_image in augmentations.items():
                    save_path = os.path.join(
                        save_dir, f"{file_name}_{title}_c0.tiff"
                    )
                    save_tiff(augmented_image, save_path, original_order="YXC")

        # Plot if enabled
        if debug_plot:
            self._plot_circles(
                file_path,
                sir_tubulin_image,
                filtered_image,
                final_peaks,
                all_positions,
                peaks_intensity,
                all_intensities,
                average_circle_peaks,
            )

        # Get the template of the bridge
        if self.template_type == TemplateType.ALL_WITHOUT_HARALICK:
            return np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                    np.var(
                        all_intensities[0]
                    ),  # 2 : variance of intensity on the circle
                    np.mean(
                        all_intensities[0]
                    ),  # 3 : mean intensity on the circle
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
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                ]
            )

        if self.template_type == TemplateType.HARALICK:
            return haralick_features

        if self.template_type == TemplateType.ALL:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                    np.mean(
                        all_intensities[0]
                    ),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features))

        # Ablation study on All
        if self.template_type == TemplateType.ALL_ABLATION_1:
            template = np.array(
                [
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                    np.mean(
                        all_intensities[0]
                    ),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features))

        if self.template_type == TemplateType.ALL_ABLATION_2:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    np.mean(
                        all_intensities[0]
                    ),  # 1 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features))

        if self.template_type == TemplateType.ALL_ABLATION_3:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                ]
            )
            return np.concatenate((template, haralick_features))

        if self.template_type == TemplateType.ALL_ABLATION_4:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                    np.mean(
                        all_intensities[0]
                    ),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate((template, haralick_features[1:]))

        if self.template_type == TemplateType.ALL_ABLATION_5:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                    np.mean(
                        all_intensities[0]
                    ),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate(
                (template, haralick_features[:1], haralick_features[2:])
            )

        if self.template_type == TemplateType.ALL_ABLATION_6:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                    np.mean(
                        all_intensities[0]
                    ),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate(
                (template, haralick_features[:2], haralick_features[3:])
            )

        if self.template_type == TemplateType.ALL_ABLATION_7:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                    np.mean(
                        all_intensities[0]
                    ),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate(
                (template, haralick_features[:3], haralick_features[4:])
            )

        if self.template_type == TemplateType.ALL_ABLATION_8:
            template = np.array(
                [
                    len(final_peaks),  # 0 : number of peaks
                    (
                        np.mean(peaks_intensity)
                        if len(final_peaks) > 0
                        else np.mean(all_intensities[0])
                    ),  # 1 : intensity of the peaks
                    np.mean(
                        all_intensities[0]
                    ),  # 2 : mean intensity on the circle
                ]
            )
            return np.concatenate(
                (template, haralick_features[:4], haralick_features[5:])
            )

        if self.template_type == TemplateType.AVERAGE_CIRCLE:
            avg_peaks_template = np.array(
                [
                    average_circle_peaks[0].intensity,
                    average_circle_peaks[0].prominence,
                    average_circle_peaks[0].width,
                    average_circle_peaks[1].intensity,
                    average_circle_peaks[1].prominence,
                    average_circle_peaks[1].width,
                ]
            )
            return avg_peaks_template

        raise ValueError("Unknown template type")

    def get_bridge_class(
        self,
        img_bridge: np.ndarray,
        scaler: StandardScaler,
        clf: SVC,
        debug_plot=True,
    ):
        """
        Parameters 
        ----------

        img_bridge : CYX
        """
        # Make sure crop has expected size
        assert (
            img_bridge.shape[1] == self.margin * 2
            and img_bridge.shape[2] == self.margin * 2
        )

        # Get template and reshape to 2D data
        template = self.get_bridge_template(
            img_bridge, debug_plot=debug_plot
        ).reshape(1, -1)

        # This only occurs in debug mode
        if scaler.n_features_in_ != template.shape[1]:
            return -1, template, -1

        # Scale template
        scaled_template = scaler.transform(template)

        # Predict the class
        frame_class = clf.predict(scaled_template)
        if frame_class[0] == "A":
            predicted_class = 0
        elif frame_class[0] == "B":
            predicted_class = 1
        else:
            predicted_class = 2

        # Get distance to the hyperplane
        distance = clf.decision_function(scaled_template)

        return predicted_class, template, distance

    @staticmethod
    def apply_hmm(hmm_parameters, sequence):
        """
        Correct the sequence of classes using HMM.
        """
        # Define observation sequence
        obs_seq = np.asarray(sequence, dtype=np.int32)

        # Define HMM model & run inference
        model = HiddenMarkovModel(
            hmm_parameters["A"], hmm_parameters["B"], hmm_parameters["pi"]
        )
        states_seq, _ = model.viterbi_inference(obs_seq)

        return states_seq

    def update_mt_cut_detection(
        self,
        mitosis_track: MitosisTrack,
        video: np.ndarray,
        scaler_path: str,
        model_path: str,
        hmm_bridges_parameters_file: str,
        debug_plot=False,
    ):
        """
        Update micro-tubules cut detection using bridges classification.
        Expect TYXC video.
        """
        results = {
            "list_class_bridges": [],
            "list_class_bridges_after_hmm": [],
            "templates": [],
            "distances": [],
            "crops": [],
        }
        classification_impossible = self._is_bridges_classification_impossible(
            mitosis_track
        )

        if classification_impossible:
            return results

        # Perform classification...
        ordered_mb_frames = sorted(mitosis_track.mid_body_spots.keys())
        first_mb_frame = ordered_mb_frames[0]
        last_mb_frame = ordered_mb_frames[-1]
        first_frame = max(
            first_mb_frame, mitosis_track.key_events_frame["cytokinesis"] - 2
        )  # -2?

        # Load the classifier and scaler
        with open(model_path, "rb") as f:
            classifier: SVC = pickle.load(f)
        # Load the scaler
        with open(scaler_path, "rb") as f:
            scaler: StandardScaler = pickle.load(f)

        # Iterate over frames and get the class of the bridge
        for frame in range(first_frame, last_mb_frame + 1):
            min_x = mitosis_track.position.min_x
            min_y = mitosis_track.position.min_y

            # Get midbody coordinates
            mb_coords = mitosis_track.mid_body_spots[frame].position
            x_pos, y_pos = min_x + mb_coords[0], min_y + mb_coords[1]

            # Extract frame image and crop around the midbody Sir-tubulin
            frame_image = (
                video[frame, :, :, :].squeeze().transpose(2, 0, 1)
            )  # CYX
            crop = smart_cropping(
                frame_image, self.margin, x_pos, y_pos, pad=True
            )  # CYX

            # Get bridge class (and other data useful for debugging)
            bridge_class, template, distance = self.get_bridge_class(
                crop, scaler, classifier, debug_plot=debug_plot
            )

            results["list_class_bridges"].append(bridge_class)
            results["templates"].append(template)
            results["distances"].append(distance)
            results["crops"].append(crop)

        # Make sure cytokinesis bridge is detected as A (no MT cut)
        relative_cytokinesis_frame = (
            mitosis_track.key_events_frame["cytokinesis"] - first_frame
        )
        if (
            relative_cytokinesis_frame < 0
            or results["list_class_bridges"][relative_cytokinesis_frame] != 0
        ):
            mitosis_track.key_events_frame["first_mt_cut"] = (
                ImpossibleDetection.MT_CUT_AT_CYTOKINESIS
            )
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.MT_CUT_AT_CYTOKINESIS
            )
            return results

        # Read HMM parameters
        if not os.path.exists(hmm_bridges_parameters_file):
            raise FileNotFoundError(
                f"File {hmm_bridges_parameters_file} not found"
            )
        hmm_parameters = np.load(hmm_bridges_parameters_file)

        # Correct the sequence with HMM
        results["list_class_bridges_after_hmm"] = self.apply_hmm(
            hmm_parameters, results["list_class_bridges"]
        )

        # Get index of first element > 0 in sequence (first MT cut)
        first_mt_cut_frame_rel = next(
            (
                i
                for i, x in enumerate(results["list_class_bridges_after_hmm"])
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
            return results

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
            return results

        # Update mitosis track accordingly
        mitosis_track.key_events_frame["first_mt_cut"] = first_mt_cut_frame_abs

        # Get index of first element > 1 in sequence (second MT cut)
        second_mt_cut_frame_rel = next(
            (
                i
                for i, x in enumerate(results["list_class_bridges_after_hmm"])
                if x > 1
            ),
            -1,
        )

        # get the frame of the second MT cut
        if second_mt_cut_frame_rel == -1:
            mitosis_track.key_events_frame["second_mt_cut"] = (
                ImpossibleDetection.NO_CUT_DETECTED
            )
            return results

        second_mt_cut_frame_abs = first_frame + second_mt_cut_frame_rel

        # Update mitosis track accordingly
        mitosis_track.key_events_frame["second_mt_cut"] = (
            second_mt_cut_frame_abs
        )

        # Return proxies for testing
        return results
