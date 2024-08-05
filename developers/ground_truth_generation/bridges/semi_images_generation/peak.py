from __future__ import annotations

import numpy as np


class Peak:
    """
    A class to store useful values regarding signal peaks.
    Used in the cut detection algorithm.
    """

    def __init__(
        self,
        relative_position=-1,
        intensity=0,
        coordinates=(0, 0),
        relative_intensity=0,
        position_index=-1,
        circle_index=-1,
        prominence=0,
        width=0,
        relative_width=0,
    ):
        self.relative_position = relative_position
        self.intensity = intensity
        self.coordinates = coordinates  # (y, x)
        self.relative_intensity = relative_intensity
        self.position_index = position_index
        self.circle_index = circle_index
        self.prominence = prominence
        self.width = width
        self.relative_width = relative_width

    def is_empty(self) -> bool:
        """
        Return True if the peak is empty, i.e. relative_position == -1.
        """
        return self.relative_position == -1

    @classmethod
    def get_average_intensity(cls, peaks: list[Peak]) -> float:
        """
        Return the average intensity of a list of peaks.
        """
        intensities = [peak.intensity for peak in peaks]
        return np.mean(intensities)

    @classmethod
    def get_average_relative_position(cls, peaks: list[Peak]) -> float:
        """
        Return the average relative position of a list of peaks.
        """
        relative_positions = [peak.relative_position for peak in peaks]
        return np.mean(relative_positions)

    @classmethod
    def get_average_relative_intensity(cls, peaks: list[Peak]) -> float:
        """
        Return the average relative intensity of a list of peaks.
        """
        relative_intensities = [peak.relative_intensity for peak in peaks]
        return np.mean(relative_intensities)

    @classmethod
    def get_average_prominence(cls, peaks: list[Peak]) -> float:
        """
        Return the average prominence of a list of peaks.
        """
        prominences = [peak.prominence for peak in peaks]
        return np.mean(prominences)

    @classmethod
    def get_average_width(cls, peaks: list[Peak]) -> float:
        """
        Return the average width of a list of peaks.
        """
        widths = [peak.width for peak in peaks]
        return np.mean(widths)

    @classmethod
    def get_average_relative_width(cls, peaks: list[Peak]) -> float:
        """
        Return the average relative width of a list of peaks.
        """
        relative_widths = [peak.relative_width for peak in peaks]
        return np.mean(relative_widths)

    @classmethod
    def group_peaks(
        cls,
        circle_peaks: list[list[Peak]],
        windows: list[tuple[float]],
        minimum_size: int,
    ) -> list[list[Peak]]:
        """
        Group the peaks by windows.
        """
        window_peaks = []
        for window in windows:
            group = []
            for peaks in circle_peaks:
                for peak in peaks:
                    if window[0] <= peak.relative_position < window[1]:
                        group.append(peak)
            window_peaks.append(group)

        # Filter out too small groups
        filtered_window_peaks = []
        for group_peak in window_peaks:
            if (
                len(group_peak) >= minimum_size
                and group_peak not in filtered_window_peaks
            ):
                filtered_window_peaks.append(group_peak)

        return filtered_window_peaks

    @classmethod
    def create_average_peak(cls, peaks: list[Peak]) -> Peak:
        """
        Create a peak with average values from a list of peaks.
        """
        average_relative_position = cls.get_average_relative_position(peaks)
        average_intensity = cls.get_average_intensity(peaks)
        average_coordinates = np.mean(
            [peak.coordinates for peak in peaks], axis=0
        )
        average_relative_intensity = cls.get_average_relative_intensity(peaks)
        average_prominence = cls.get_average_prominence(peaks)
        average_width = cls.get_average_width(peaks)
        average_relative_width = cls.get_average_relative_width(peaks)
        return cls(
            relative_position=average_relative_position,
            intensity=average_intensity,
            coordinates=average_coordinates,
            relative_intensity=average_relative_intensity,
            prominence=average_prominence,
            width=average_width,
            relative_width=average_relative_width,
        )

    @classmethod
    def remove_small_groups(
        cls, window_peaks: list[Peak], min_peaks_by_group: int
    ) -> list[Peak]:
        """
        Remove groups that have less than min_peaks_by_group peaks.
        """
        filtered_window_peaks = [
            group for group in window_peaks if len(group) >= min_peaks_by_group
        ]
        return filtered_window_peaks

    @classmethod
    def remove_duplicated_peaks(
        cls, window_peaks: list[Peak], min_peaks_by_group: int
    ) -> list[Peak]:
        """
        One peak must appear in one group only.
        Remove the peaks that are already seen in previous groups.
        Remove groups that have less than min_peaks_by_group peaks.
        """
        for i in range(len(window_peaks) - 1):
            for j in range(i + 1, len(window_peaks)):
                window_peaks[j] = [
                    p for p in window_peaks[j] if p not in window_peaks[i]
                ]
        filtered_window_peaks = cls.remove_small_groups(
            window_peaks, min_peaks_by_group
        )
        return filtered_window_peaks

    @classmethod
    def remove_close_peaks(
        cls, window_peaks: list[Peak], circle_min_ratio
    ) -> list[Peak]:
        """
        Remove the groups that are too close to each other, i.e. distance < circle_min_ratio.
        Choose the one with highest mean intensity (used to be highest number of peaks).
        """
        # Compute the mean position of the peaks in each group
        position_groups = [
            cls.get_average_relative_position(group) for group in window_peaks
        ]

        i = 0
        while i < len(position_groups) - 1:
            j = i + 1
            while j < len(position_groups):
                first_position = (position_groups[i] - position_groups[j]) % 1
                second_position = (position_groups[j] - position_groups[i]) % 1
                if min(first_position, second_position) < circle_min_ratio:
                    if cls.get_average_intensity(
                        window_peaks[i]
                    ) > cls.get_average_intensity(window_peaks[j]):
                        window_peaks.remove(window_peaks[j])
                        position_groups.remove(position_groups[j])
                        j -= 1
                    else:
                        window_peaks.remove(window_peaks[i])
                        position_groups.remove(position_groups[i])
                        i -= 1
                j += 1
            i += 1

        return window_peaks

    def intersects(self, relative_position: float) -> bool:
        """
        Return True if the peak intersects the given relative position.
        """
        relative_difference = min(
            abs(self.relative_position - relative_position),  # usual case
            1 - abs(self.relative_position - relative_position),  # wrap around
        )
        return relative_difference < self.relative_width / 2

    def enabled_augmentation(self) -> dict[str, int]:
        """
        Enable image augmentation.

        NB: for some reason, zero is at the bottom.
        Trigonometric rotation, so 0.25 is at the right, etc.
        """

        augmentations = {}

        # Top/bottom
        if self.intersects(0.25) or self.intersects(0.75):
            augmentations["top"] = -1
            augmentations["bottom"] = -1
        else:
            if (  # MT in top
                self.relative_position > 0.25 and self.relative_position < 0.75
            ):
                augmentations["top"] = 1
                augmentations["bottom"] = 0
            else:  # MT in left
                augmentations["top"] = 0
                augmentations["bottom"] = 1

        # Left/right
        if self.intersects(0.0) or self.intersects(0.5):
            augmentations["left"] = -1
            augmentations["right"] = -1
        else:
            if self.relative_position > 0.5:  # MT in left
                augmentations["left"] = 1
                augmentations["right"] = 0
            else:  # MT in right
                augmentations["left"] = 0
                augmentations["right"] = 1

        # Top right/bottom left
        if self.intersects(1 / 8) or self.intersects(5 / 8):
            augmentations["top_right"] = -1
            augmentations["bottom_left"] = -1
        else:
            if (  # MT in top right
                self.relative_position > 1 / 8
                and self.relative_position < 5 / 8
            ):
                augmentations["top_right"] = 1
                augmentations["bottom_left"] = 0
            else:  # MT in bottom left
                augmentations["top_right"] = 0
                augmentations["bottom_left"] = 1

        # Bottom right/top left
        if self.intersects(3 / 8) or self.intersects(7 / 8):
            augmentations["bottom_right"] = -1
            augmentations["top_left"] = -1
        else:
            if (  # MT in bottom right
                self.relative_position < 3 / 8
                or self.relative_position > 7 / 8
            ):
                augmentations["bottom_right"] = 1
                augmentations["top_left"] = 0
            else:  # MT in top left
                augmentations["bottom_right"] = 0
                augmentations["top_left"] = 1

        return augmentations
