from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks


from cut_detector.factories.mt_cut_detection_factory import (
    MtCutDetectionFactory,
)
from developers.ground_truth_generation.bridges.semi_images_generation.peak import (
    Peak,
)


class MtCutDetectionFactoryAdvanced(MtCutDetectionFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Min ratio distance on circle between two peaks
        self.circle_min_ratio = 0.2025

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

    def get_circle_data(
        self,
        radius: int,
        image: np.ndarray,
    ) -> tuple[list[tuple[int]], list[float], list[Peak]]:
        """
        Compute useful data for a circle of a given radius around the mid body spot:

        Parameters
        -------
        image : YX

        Returns
        -------

        intensities : list of the intensities of the pixels on the circle

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

        return intensities

    def get_average_circle_peaks(
        self, all_intensities: list[float], debug_plot: bool
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
            debug_plot=debug_plot,
        )

        def get_peak_prominence(peak: Peak):
            return peak.prominence

        # Keep only 2 best peaks, i.e. with highest prominence
        average_circle_peaks.sort(key=get_peak_prominence, reverse=True)
        average_circle_peaks = average_circle_peaks[:2]

        # Complete with empty Peak if length <2
        while len(average_circle_peaks) < 2:
            average_circle_peaks.append(Peak())

        return average_circle_peaks
