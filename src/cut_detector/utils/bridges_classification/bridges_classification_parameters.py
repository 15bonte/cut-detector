from pydantic.dataclasses import dataclass

from cut_detector.utils.bridges_classification.template_type import TemplateType


@dataclass
class BridgesClassificationParameters:
    """
    Best parameters for the bridges classification.
    """

    # Coeff for the height of the peak
    coeff_height_peak: float = 1.071

    # Middle circle radius
    circle_radius: int = 11

    # Min ratio distance on circle between two peaks
    circle_min_ratio: float = 0.2025

    # Coeff for the width of the peak
    coeff_width_peak: float = 0.25358974

    # Distance between the min/max and middle circle
    diff_radius: float = 4.54

    # Window size for the peak detection (% of the circle perimeter)
    window_size: float = 0.04836

    # Minimum number of peaks by window for the peak detection
    min_peaks_by_group: int = 4

    # Overlap between two windows for the peak detection
    overlap: int = 4

    # Intensity threshold for the light spot detection
    intensity_threshold_light_spot: int = 350

    # h for the h_maxima function (light spot detection)
    h_maxima_light_spot: int = 130

    # Center tolerance to not count the light spots that are too close to the center
    center_tolerance_light_spot: int = 5

    # Minimum percentage of frames with light spots to consider the mitosis as a light
    # spot mitosis
    min_percentage_light_spot: float = 0.1

    # Size of the crop for the light spot detection
    crop_size_light_spot: int = 20

    # Length of the video to check around the mt cut for the light spot detection
    length_light_spot: int = 10

    # Template for SVC model
    template_type: TemplateType = TemplateType.All
