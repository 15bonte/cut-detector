from __future__ import annotations
from enum import IntEnum


class ImpossibleDetection(IntEnum):
    """Enum to represent the different types of impossible detections."""

    NORMAL = 0
    NO_MID_BODY_DETECTED = -1
    MORE_THAN_TWO_DAUGHTER_TRACKS = -2
    NEAR_BORDER = -3
    NO_MID_BODY_DETECTED_AFTER_CYTOKINESIS = -4  # deprecated
    MT_CUT_AT_CYTOKINESIS = -5  # deprecated
    NO_CUT_DETECTED = -6
    LIGHT_SPOT = -7  # deprecated
    TOO_SHORT_CUT = -8
    METAPHASE_AFTER_CYTOKINESIS = -9  # deprecated

    @staticmethod
    def display(detection: ImpossibleDetection) -> bool:
        """Check if the detection status should be displayed.

        Some ImpossibleDetection status are usually linked
        to a false cell division detection, hence are not plotted.

        Parameters
        ----------
        detection : ImpossibleDetection
            Detection status.

        Returns
        -------
        bool
            Whether the mitosis should be displayed.
        """
        return detection in [
            ImpossibleDetection.NORMAL,
            ImpossibleDetection.NEAR_BORDER,
            ImpossibleDetection.NO_CUT_DETECTED,
        ]
