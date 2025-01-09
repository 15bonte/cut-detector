from typing import Optional

from cut_detector.utils.mt_cut_detection.micro_tubules_augmentation import (
    MicroTubulesAugmentation,
)
from developers.ground_truth_generation.bridges.semi_images_generation.peak import (
    Peak,
)


class MicroTubulesAugmentationAdvanced(MicroTubulesAugmentation):
    """
    Manage the augmentation of images for microtubules detection.
    """

    def __init__(self, peaks: Optional[list[Peak]] = None):
        super().__init__()
        peak_augmentations = []
        if peaks is None:
            peaks = []
        for peak in peaks:
            peak_augmentations.append(peak.enabled_augmentation())
        self.augmentations = self.merge_augmentations(peak_augmentations)

    @classmethod
    def merge_augmentations(
        cls, peak_augmentations: Optional[list[dict[str, int]]] = None
    ) -> dict[str, int]:
        """
        Merge values given to augmentation categories by different peaks.
        """
        augmentations = {}
        for category in [
            "top",
            "bottom",
            "left",
            "right",
            "top_left",
            "bottom_right",
            "top_right",
            "bottom_left",
        ]:
            value = 0  # by default, no MT is seen
            if peak_augmentations is not None:
                for (
                    peak_augmentation
                ) in peak_augmentations:  # iterate over peaks
                    if (
                        peak_augmentation[category] == -1
                    ):  # if one peak forbids the augmentation
                        value = None
                        break
                    if (
                        peak_augmentation[category] == 1
                    ):  # if at least one peak found
                        value = 1
            if value is not None:
                augmentations[category] = value
        return augmentations
