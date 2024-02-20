from __future__ import annotations
from typing import Optional

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

    @classmethod
    def enabled_augmentation(cls, peaks: Optional[list[Peak]]) -> list[str]:
        """
        Enable image augmentation.
        Depends on the peaks: must not cut in the middle of a peak,
        and must have peak on both sides of the cut.

        NB: for some reason, zero is at the bottom.
        Trigonometric rotation, so 0.25 is at the right, etc.
        """

        if peaks is None:
            return [
                "top",
                "bottom",
                "left",
                "right",
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            ]

        assert len(peaks) == 2, "Only no cut is supported so far"

        augmentations = []

        # Top/bottom
        if (
            not peaks[0].intersects(0.25)
            and not peaks[1].intersects(0.25)
            and not peaks[0].intersects(0.75)
            and not peaks[1].intersects(0.75)
        ):
            if (  # first MT in top
                peaks[0].relative_position > 0.25
                and peaks[0].relative_position < 0.75
            ) or (  # second MT in top
                peaks[1].relative_position > 0.25
                and peaks[1].relative_position < 0.75
            ):
                augmentations.append("top")

            if (  # first MT in bottom
                peaks[0].relative_position < 0.25
                or peaks[0].relative_position > 0.75
            ) or (  # second MT in bottom
                peaks[1].relative_position < 0.25
                or peaks[1].relative_position > 0.75
            ):
                augmentations.append("bottom")

        # Left/right
        if (
            not peaks[0].intersects(0.0)
            and not peaks[1].intersects(0.0)
            and not peaks[0].intersects(0.5)
            and not peaks[1].intersects(0.5)
        ):
            if (  # first MT in left
                peaks[0].relative_position > 0.5
            ) or (  # second MT in left
                peaks[1].relative_position > 0.5
            ):
                augmentations.append("left")

            if (  # first MT in right
                peaks[0].relative_position < 0.5
            ) or (  # second MT in right
                peaks[1].relative_position < 0.5
            ):
                augmentations.append("right")

        # Top right/bottom left
        if (
            not peaks[0].intersects(1 / 8)
            and not peaks[1].intersects(1 / 8)
            and not peaks[0].intersects(5 / 8)
            and not peaks[1].intersects(5 / 8)
        ):
            if (  # first MT in bottom left
                peaks[0].relative_position < 1 / 8
                or peaks[0].relative_position > 5 / 8
            ) or (  # second MT in bottom left
                peaks[1].relative_position < 1 / 8
                or peaks[1].relative_position > 5 / 8
            ):
                augmentations.append("bottom_left")

            if (  # first MT in top right
                peaks[0].relative_position > 1 / 8
                and peaks[0].relative_position < 5 / 8
            ) or (  # second MT in top right
                peaks[1].relative_position > 1 / 8
                and peaks[1].relative_position < 5 / 8
            ):
                augmentations.append("top_right")

        # Bottom right/top left
        if (
            not peaks[0].intersects(3 / 8)
            and not peaks[1].intersects(3 / 8)
            and not peaks[0].intersects(7 / 8)
            and not peaks[1].intersects(7 / 8)
        ):
            if (  # first MT in bottom right
                peaks[0].relative_position < 3 / 8
                or peaks[0].relative_position > 7 / 8
            ) or (  # second MT in bottom right
                peaks[1].relative_position < 3 / 8
                or peaks[1].relative_position > 7 / 8
            ):
                augmentations.append("bottom_right")

            if (  # first MT in top left
                peaks[0].relative_position > 3 / 8
                and peaks[0].relative_position < 7 / 8
            ) or (  # second MT in top left
                peaks[1].relative_position > 3 / 8
                and peaks[1].relative_position < 7 / 8
            ):
                augmentations.append("top_left")

        return augmentations
